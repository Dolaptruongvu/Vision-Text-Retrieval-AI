import torch
import os
import time
import traceback
from typing import List, Dict, Optional
from dotenv import load_dotenv
import numpy as np
import logging
import pandas as pd
from datasets import Dataset
import sys
import math # Thêm import math để tính số lô

# --- Haystack Core & Standard Components ---
from haystack import Pipeline, Document, component
from haystack.components.builders import ChatPromptBuilder
from haystack.utils import Secret, ComponentDevice
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.dataclasses import ChatMessage
from haystack.components.converters import OutputAdapter

# --- Haystack Integrations ---
from haystack_integrations.components.generators.ollama import OllamaGenerator
try:
    from milvus_haystack import MilvusDocumentStore, MilvusEmbeddingRetriever
except ImportError: print("ERROR: milvus_haystack not found. pip install milvus-haystack"); sys.exit(1)
try:
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
except ImportError: print("ERROR: haystack-integrations[elasticsearch] not found. pip install 'haystack-integrations[elasticsearch]'"); sys.exit(1)

# --- Langchain Components (for Ragas wrappers) ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI # Dùng cho AI Studio API Key
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError: print("ERROR: Langchain google_genai/huggingface not found. pip install langchain-google-genai langchain-huggingface"); sys.exit(1)

# --- RAGAS Imports ---
try:
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError: print("ERROR: Ragas not found. pip install ragas"); sys.exit(1)

# --- Configuration Loading and Validation ---
print("--- Loading Configuration ---")
load_dotenv()

def get_env_var(var_name: str, is_critical: bool = True, default_value: Optional[str] = None) -> Optional[str]:
    value = os.getenv(var_name)
    env_var_key = f"{var_name}_API_KEY" if "GOOGLE_AI" in var_name else var_name
    if not value and env_var_key != var_name: value = os.getenv(env_var_key)

    if not value:
        if is_critical and default_value is None:
            print(f"\nCRITICAL ERROR: Environment variable '{var_name}' (or related key like {env_var_key} if applicable) not set and is required.")
            print("Please ensure it is defined in your .env file or system environment.")
            sys.exit(1)
        elif default_value is not None:
            print(f"INFO: Environment variable '{var_name}' not set, using default: '{default_value}'")
            return default_value
        else:
            return None

    is_sensitive = any(k in var_name for k in ["TOKEN", "KEY", "SECRET", "PASSWORD"])
    display_value = "***" if is_sensitive else value
    print(f"INFO: Loaded {var_name}={display_value}")
    return value

print("Checking environment variables...")
hf_token = get_env_var("HF_TOKEN", is_critical=False)
MILVUS_URI = get_env_var("MILVUS_HOST", default_value="http://localhost:19530")
ES_HOST = get_env_var("ES_HOST", default_value="http://127.0.0.1:9200")
OLLAMA_URL = get_env_var("OLLAMA_URL", default_value="http://localhost:11434")
OLLAMA_MODEL_NAME = get_env_var("OLLAMA_MODEL", default_value="gemma3:latest")
GOOGLE_API_KEY = get_env_var("GOOGLE_API_KEY") # Critical for Ragas AI Studio
GOOGLE_AI_MODEL_NAME_RAGAS = get_env_var("GOOGLE_AI_MODEL_NAME_RAGAS", default_value="gemini-1.5-flash-latest")

# --- Other Configurations ---
HF_TOKEN = Secret.from_token(hf_token) if hf_token else None
EXPECTED_EMBEDDING_DIM = 384 # Vẫn cần biết dim dự kiến
COLLECTION_NAME = "rag_blocks"
VECTOR_FIELD_NAME = "embedding"
TEXT_FIELD_NAME_MILVUS = "content"
MILVUS_INDEX_PARAMS = {"index_type": "DISKANN", "metric_type": "COSINE", "params": {"search_list": 100}}
MILVUS_SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"search_list": 100}}
MILVUS_TOP_K = 7
ES_INDEX_NAME = "my_rag_index"
ES_BM25_TOP_K = 7
RANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RANKER_FINAL_TOP_K = 5
EMBEDDER_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_TIMEOUT = 300
RAGAS_GOOGLE_AI_TIMEOUT = 360.0

# --- Cấu hình cho việc gọi API chậm lại ---
RAGAS_BATCH_SIZE = 1  # Số lượng mẫu xử lý mỗi lần gọi evaluate (giảm để ít request đồng thời)
DELAY_BETWEEN_BATCHES = 65 # Giây nghỉ giữa các lô ( > 60s để đảm bảo dưới 15 RPM)

print("-----------------------------")

# --- Device Determination ---
print("--- Determining Device ---")
try:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        current_device_index = torch.cuda.current_device()
        device = ComponentDevice.from_str(f"cuda:{current_device_index}")
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(current_device_index)} ({device.to_torch_str()})")
    else:
        device = ComponentDevice.from_str("cpu")
        print("CUDA not available. Using CPU.")
except Exception as e:
    print(f"Error determining device: {e}. Defaulting to CPU.")
    device = ComponentDevice.from_str("cpu")
print("-----------------------------")

# --- Haystack Component Initialization ---
print("--- Initializing Haystack Components ---")
try:
    # Document Stores
    # *** BỎ embedding_dim THEO YÊU CẦU - CÓ THỂ GÂY LỖI NẾU COLLECTION CHƯA TỒN TẠI/SAI SCHEMA ***
    print("WARNING: Initializing MilvusDocumentStore without explicitly setting embedding_dim. This might fail if the collection doesn't exist or has an incorrect schema.")
    milvus_store = MilvusDocumentStore(
        connection_args={"uri": MILVUS_URI},
        collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD_NAME,
        text_field=TEXT_FIELD_NAME_MILVUS,
        # embedding_dim=EXPECTED_EMBEDDING_DIM, # Bỏ dòng này
        index_params=MILVUS_INDEX_PARAMS,
        search_params=MILVUS_SEARCH_PARAMS
    )
    es_store = ElasticsearchDocumentStore(hosts=[ES_HOST], index=ES_INDEX_NAME)
    print(f"Milvus connection check successful. | ES: {es_store.count_documents()} docs.") # Không in count Milvus nếu chưa chắc có collection

    # Embedder
    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL_NAME, device=device, token=HF_TOKEN, normalize_embeddings=True)
    text_embedder.warm_up()

    # Retrievers
    milvus_retriever = MilvusEmbeddingRetriever(document_store=milvus_store, top_k=MILVUS_TOP_K)
    bm25_retriever = ElasticsearchBM25Retriever(document_store=es_store, top_k=ES_BM25_TOP_K)

    # Joiner & Ranker
    joiner = DocumentJoiner(join_mode="concatenate")
    ranker = TransformersSimilarityRanker(model=RANKER_MODEL_NAME, top_k=RANKER_FINAL_TOP_K, token=HF_TOKEN, device=device)
    ranker.warm_up()

    # Simple Prompt Builder for Evaluation
    eval_chat_prompt_template = [ChatMessage.from_system(
        "You are a Vietnamese agricultural AI assistant. Your mission is to provide accurate and helpful agricultural information to Vietnamese users. STRICTLY ADHERE to the following rules IN THE CORRECT ORDER:"
        "\n"
        "**RULE 1: PRIORITIZE HANDLING 'HEALTHY' IMAGE RESULTS**"
        "\n"
        "*   **DECISIVE CONDITION:** The `**Image Analysis Result:**` line exists AND it contains ONE OF THE FOLLOWING KEYWORDS: 'khỏe mạnh', 'healthy', 'không có bệnh', 'bình thường'."
        "*   **MANDATORY AND SOLE ACTION (If DECISIVE CONDITION is TRUE):**"
        "    1.  **COMPLETELY IGNORE EVERYTHING ELSE:** Do NOT read, consider, or use `{{ query }}`, `{{ memories }}`, and **POSITIVELY DO NOT LOOK AT or USE any content from `{{ documents }}`**. They are irrelevant and FORBIDDEN in this case."
        "    2.  Reply with EXACTLY and ONLY the following sentence in VIETNAMESE: 'Kết quả phân tích hình ảnh cho thấy cây trồng này khỏe mạnh, không có dấu hiệu bệnh rõ ràng.'"
        "    3.  **STOP IMMEDIATELY.** Do not execute any other Rules."
        "\n"
        "--- Only execute the rules below IF RULE 1 WAS NOT TRIGGERED ---"
        "\n"
        "**RULE 2: HANDLE OTHER (NON-'HEALTHY') IMAGE RESULTS**"
        "\n"
        "*   **CONDITION:** The `**Image Analysis Result:**` line exists AND it does NOT contain any keywords listed in Rule 1."
        "*   **ACTION:**"
        "    *   Identify the disease/issue from `**Image Analysis Result:**` (This is the **Current Context**)."
        "    *   Answer `{{ query }}` focusing on this **Current Context**."
        "    *   **ONLY** use `{{ documents }}` if they are directly relevant to the **Current Context**. Ignore all irrelevant documents."
        "    *   Avoid video links."
        "    *   Stop."
        "\n"
        "--- Only execute the rules below IF RULE 1 AND RULE 2 DID NOT APPLY ---"
        "\n"
        "**RULE 3: CHECK FOR OFF-TOPIC QUERIES (NO IMAGE)**"
        "\n"
        "*   **CONDITION:** No `**Image Analysis Result:**` exists AND `{{ query }}` is clearly NOT related to agriculture, plants, pests, diseases, fertilizers, or farming techniques (e.g., asking about politics, history, unrelated cooking, celebrities, world news, etc.)."
        "*   **MANDATORY ACTION:**"
        "    1.  **ABSOLUTELY DO NOT USE `{{ documents }}`.**"
        "    2.  Politely reply in VIETNAMESE that you are an agricultural assistant and cannot answer off-topic questions. Example: 'Tôi là trợ lý AI chuyên về nông nghiệp Việt Nam. Rất tiếc, tôi không thể trả lời câu hỏi của bạn về chủ đề này. Bạn có câu hỏi nào khác liên quan đến trồng trọt, sâu bệnh hoặc kỹ thuật nông nghiệp không?'"
        "    3.  **STOP IMMEDIATELY.** Do not execute Rule 4."
        "\n"
        "--- Only execute the rule below IF RULE 1, 2, AND 3 DID NOT APPLY ---"
        "\n"
        "**RULE 4: ANSWER NORMAL AGRICULTURAL QUERIES (NO IMAGE, ON-TOPIC)**"
        "\n"
        "*   **CONDITION:** No `**Image Analysis Result:**` exists AND `{{ query }}` is related to agriculture."
        "*   **ACTION:**"
        "    *   Check `{{ memories }}` for a recently discussed **Context** (disease/topic)."
        "    *   Answer `{{ query }}`: prioritize the **Context** (if available), otherwise answer generally."
        "    *   Use `{{ documents }}` to find information relevant to the **Context** (if available) or directly relevant to `{{ query }}`."
        "    *   Avoid video links."
        "\n"
        "**MOST CRITICAL REMINDERS:**"
        "\n"
        "1.  **ADHERE TO RULE ORDER: 1 -> 2 -> 3 -> 4.**"
        "2.  **ALWAYS RESPOND IN VIETNAMESE.** THIS IS MANDATORY."
        "3.  **RULE 1 IS ABSOLUTE:** When triggered, it overrides everything else and FORBIDS document usage."
        "4.  **RULE 3 ALSO FORBIDS DOCUMENT USAGE** for off-topic questions."
        "5.  Maintain **Context** once identified in Rule 2 or 4 for follow-up turns."
    ),
    ChatMessage.from_user(
         """**Chat History:**
{% for msg in memories %}
{% if msg.role == 'user' %}User: {{ (msg.to_dict()).content[0].text }}{% elif msg.role == 'assistant' %}Assistant: {{ (msg.to_dict()).content[0].text }}{% endif %}
{% else %}
(No chat history)
{% endfor %}

{% if identified_disease %}
**Image Analysis Result:** {{ identified_disease }}
{% endif %}

**Reference Documents:**
{% if documents %}
    {% for doc in documents %}
    ---
    {{ doc.content }}
    ---
    {% endfor %}
{% else %}
    (No reference documents)
{% endif %}

**User's Current Question:** {{ query }}

**Answer (Strictly follow ALL rules and critical reminders, RESPOND ONLY IN VIETNAMESE):**"""
    )]
    eval_chat_prompt_builder = ChatPromptBuilder(template=eval_chat_prompt_template)
    logging.getLogger("haystack.core.pipeline.pipeline").setLevel(logging.WARNING)

    # Adapter
    message_to_string_adapter = OutputAdapter(template="{{ (messages[-1].to_dict())['content'][0]['text'] }}", output_type=str)

    # Ollama Generator (for Haystack Pipeline)
    ollama_llm_generator = OllamaGenerator(model=OLLAMA_MODEL_NAME, url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT, generation_kwargs={"temperature": 0.1})
    print("Haystack components initialized.")
except Exception as e:
    print(f"\nFATAL ERROR initializing Haystack components: {e}")
    traceback.print_exc()
    sys.exit(1)
print("-----------------------------")


# --- Langchain/Ragas Wrapper Initialization ---
print("--- Initializing Ragas Wrappers (Google AI Studio for LLM, HF for Embeddings) ---")
ragas_metric_llm = None
ragas_embeddings = None
try:
    # Google AI Studio LLM for Ragas
    print(f"Attempting to initialize ChatGoogleGenerativeAI with model='{GOOGLE_AI_MODEL_NAME_RAGAS}'")
    langchain_google_ai_for_ragas = ChatGoogleGenerativeAI(
        model=GOOGLE_AI_MODEL_NAME_RAGAS,
        temperature=0.1,
        request_timeout=RAGAS_GOOGLE_AI_TIMEOUT,
        # google_api_key=GOOGLE_API_KEY # Langchain tự đọc từ env var
    )
    ragas_metric_llm = LangchainLLMWrapper(langchain_llm=langchain_google_ai_for_ragas)
    print(f"Ragas LLM Wrapper (Google AI Studio: {GOOGLE_AI_MODEL_NAME_RAGAS}) initialized.")

    # HF Embeddings for Ragas
    langchain_embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDER_MODEL_NAME,
        model_kwargs={'device': device.to_torch_str()},
        encode_kwargs={'normalize_embeddings': True}
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings=langchain_embeddings_model)
    print("Ragas Embeddings Wrapper (HuggingFace) initialized.")
except Exception as e:
    print(f"\nWARNING: Error initializing Ragas Wrappers: {e}")
    print("Ragas evaluation might fail or provide incomplete results.")
    traceback.print_exc()
print("-----------------------------")


# --- Build Haystack Evaluation Pipeline ---
print("--- Building Haystack Evaluation Pipeline (using Ollama Generator) ---")
eval_pipeline = Pipeline()
eval_pipeline.add_component("text_embedder", text_embedder)
eval_pipeline.add_component("milvus_retriever", milvus_retriever)
eval_pipeline.add_component("bm25_retriever", bm25_retriever)
eval_pipeline.add_component("joiner", joiner)
eval_pipeline.add_component("ranker", ranker)
eval_pipeline.add_component("eval_chat_prompt_builder", eval_chat_prompt_builder)
eval_pipeline.add_component("message_to_string_adapter", message_to_string_adapter)
eval_pipeline.add_component("llm", ollama_llm_generator) # Ollama generates the answer

eval_pipeline.connect("text_embedder.embedding", "milvus_retriever.query_embedding")
eval_pipeline.connect("milvus_retriever.documents", "joiner.documents")
eval_pipeline.connect("bm25_retriever.documents", "joiner.documents")
eval_pipeline.connect("joiner.documents", "ranker.documents")
eval_pipeline.connect("ranker.documents", "eval_chat_prompt_builder.documents")
eval_pipeline.connect("eval_chat_prompt_builder.prompt", "message_to_string_adapter.messages")
eval_pipeline.connect("message_to_string_adapter.output", "llm.prompt")
print("Evaluation pipeline built successfully.")
print("-----------------------------")

# --- Evaluation Dataset ---
# !!! USER ACTION REQUIRED: Define your evaluation data here !!!
evaluation_dataset_for_ragas = [
    {
        "question": "Triệu chứng của bệnh đốm lá Cercospora trên củ cải đường là gì?",
        "ground_truth": "Bệnh đốm lá Cercospora trên củ cải đường gây ra các đốm tròn, đường kính khoảng 3mm (hoặc 1/8 inch), có tâm màu xám tro và viền màu nâu sẫm hoặc đỏ tía. Khi bệnh nặng, lá có thể rụng, làm giảm năng suất và chất lượng củ cải."
    },
    {
        "question": "Làm thế nào để quản lý bệnh đốm lá Cercospora bằng biện pháp canh tác?",
        "ground_truth": "Các biện pháp canh tác bao gồm thăm dò đồng ruộng thường xuyên để phát hiện sớm, cày xới vụ thu để vùi lấp tàn dư cây bệnh, luân canh cây trồng (nghỉ củ cải đường ít nhất 2 năm), trồng xa các khu vực đã nhiễm bệnh trước đó (ít nhất 100 thước Anh), và sử dụng giống kháng bệnh (ví dụ: giống CR+)."
    },
    {
        "question": "Điều kiện môi trường nào thuận lợi cho bệnh đốm lá Cercospora phát triển?",
        "ground_truth": "Bệnh phát triển mạnh trong điều kiện thời tiết ấm, ẩm ướt. Cụ thể là nhiệt độ ban ngày từ 80-90°F (27-32°C), nhiệt độ ban đêm trên 60°F (15.5°C), và độ ẩm không khí cao (90-100%). Bệnh thường phổ biến sau khi tán cây khép lại."
    }
    # --- ADD MORE QUESTIONS AND GROUND TRUTHS HERE ---
]
print(f"--- Loaded {len(evaluation_dataset_for_ragas)} evaluation samples ---")
if not evaluation_dataset_for_ragas:
    print("ERROR: Evaluation dataset is empty. Please add question/ground_truth pairs.")
    sys.exit(1)
print("-----------------------------")

# --- Run Pipeline & Collect Data ---
print("--- Running Pipeline to Collect Data for Ragas (Using Ollama Generator) ---")
ragas_data_samples = []
start_collection_time = time.time()

if not evaluation_dataset_for_ragas:
    print("ERROR: `evaluation_dataset_for_ragas` is empty.") # Lỗi đã được xử lý ở trên
else:
    for i, item in enumerate(evaluation_dataset_for_ragas):
        question = item["question"]
        print(f"\nProcessing {i+1}/{len(evaluation_dataset_for_ragas)}: {question}")
        pipeline_input = {
            "text_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "ranker": {"query": question},
            "eval_chat_prompt_builder": {"query": question},
        }
        try:
            result = eval_pipeline.run(pipeline_input, include_outputs_from=["ranker", "llm"])
            contexts = [doc.content for doc in result.get("ranker", {}).get("documents", []) if doc.content and doc.content.strip()]
            answer = result.get("llm", {}).get("replies", [""])[0]
            if not answer: logging.warning(f"Empty answer from Ollama for: {question}")

            ragas_data_samples.append({
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": item["ground_truth"]
            })
            print(f"  Collected contexts: {len(contexts)}, Answer generated (by Ollama).")
        except Exception as e:
            print(f"  ERROR processing question '{question}': {e}")
            ragas_data_samples.append({
                "question": question, "contexts": [], "answer": f"Pipeline Error: {e}", "ground_truth": item["ground_truth"]
            })

end_collection_time = time.time()
print(f"\n--- Data Collection Finished in {end_collection_time - start_collection_time:.2f} seconds ---")
print("-----------------------------")

# --- Ragas Evaluation with Batching and Delay ---
if ragas_data_samples and ragas_metric_llm and ragas_embeddings:
    print(f"--- Starting Ragas Evaluation (using Google AI Studio for metrics) with Batch Size: {RAGAS_BATCH_SIZE}, Delay: {DELAY_BETWEEN_BATCHES}s ---")
    all_results_dfs = [] # List để lưu DataFrame kết quả của từng lô
    num_samples = len(ragas_data_samples)
    num_batches = math.ceil(num_samples / RAGAS_BATCH_SIZE)
    evaluation_start_time = time.time()

    for i in range(num_batches):
        batch_start_index = i * RAGAS_BATCH_SIZE
        batch_end_index = min((i + 1) * RAGAS_BATCH_SIZE, num_samples)
        current_batch_data = ragas_data_samples[batch_start_index:batch_end_index]
        batch_dataset = Dataset.from_list(current_batch_data)

        print(f"\nEvaluating Batch {i+1}/{num_batches} (Samples {batch_start_index + 1} to {batch_end_index})...")
        try:
            metrics_to_evaluate = [context_precision, context_recall, faithfulness, answer_relevancy]
            batch_start_eval_time = time.time()

            results = evaluate(
                batch_dataset,
                metrics=metrics_to_evaluate,
                llm=ragas_metric_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False # Rất quan trọng khi chạy theo lô
            )
            batch_end_eval_time = time.time()
            print(f"  Batch {i+1} evaluation finished in {batch_end_eval_time - batch_start_eval_time:.2f} seconds.")

            # Lưu kết quả của lô này
            if results: # Kiểm tra xem evaluate có trả về kết quả không
                 results_df_batch = results.to_pandas()
                 all_results_dfs.append(results_df_batch)
            else:
                 print(f"  WARNING: No results returned from evaluate for batch {i+1}.")


            # Nghỉ giữa các lô (trừ lô cuối cùng)
            if i < num_batches - 1:
                print(f"  Waiting for {DELAY_BETWEEN_BATCHES} seconds before next batch to avoid rate limits...")
                time.sleep(DELAY_BETWEEN_BATCHES)

        except Exception as e:
            print(f"\n  ERROR during Ragas evaluation for Batch {i+1}: {e}")
            traceback.print_exc()
            print(f"  Skipping batch {i+1} due to error.")
            # Có thể thêm placeholder lỗi vào all_results_dfs nếu muốn
            if i < num_batches - 1: # Vẫn nghỉ nếu lỗi không phải lô cuối
                 print(f"  Waiting for {DELAY_BETWEEN_BATCHES} seconds despite error...")
                 time.sleep(DELAY_BETWEEN_BATCHES)

    evaluation_end_time = time.time()
    print(f"\n--- Total Ragas Evaluation (with delays) Finished in {evaluation_end_time - evaluation_start_time:.2f} seconds ---")
    print("-----------------------------")

    # --- Kết hợp và Hiển thị Kết quả ---
    if all_results_dfs:
        final_results_df = pd.concat(all_results_dfs, ignore_index=True)
        print("\n--- Final Ragas Evaluation Results (Combined Batches) ---")
        pd.set_option('display.max_rows', None); pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000); pd.set_option('display.max_colwidth', 150)
        print(final_results_df)
        print("-----------------------------")

        print("--- Average Ragas Scores (Combined Batches) ---")
        numeric_cols = final_results_df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            print(final_results_df[numeric_cols].mean(skipna=True))
        else:
            print("No numeric metric columns found.")
        print("-----------------------------")

        # Optional: Save final results
        # filename = f"evaluation_results_batched_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        # final_results_df.to_csv(filename, index=False, encoding='utf-8-sig')
        # print(f"Final results saved to {filename}")
    else:
        print("ERROR: No results were collected from any evaluation batch.")

elif not ragas_data_samples:
    print("ERROR: No data collected for Ragas evaluation.")
else:
    print("ERROR: Ragas LLM (Google AI Studio) or Embeddings wrapper failed initialization. Cannot run evaluation.")

print("\n--- Evaluation Script Complete ---")