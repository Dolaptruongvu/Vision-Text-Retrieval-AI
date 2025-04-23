import torch
import os
import time
import traceback

# --- Haystack Core ---
from haystack import Pipeline, Document
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.utils import Secret, ComponentDevice
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker

# --- Haystack Integrations ---
from haystack_integrations.components.generators.ollama import OllamaGenerator
# Giữ style import gốc cho Milvus
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever
# Thêm import cho Elasticsearch
try:
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
except ImportError as e:
     print(f"Error importing Haystack integrations. Install: pip install haystack-integrations[elasticsearch,milvus]\nOriginal error: {e}")
     exit()

# --- Cấu hình ---
HF_TOKEN = Secret.from_token("") # Token Hugging Face cho Ranker
EXPECTED_EMBEDDING_DIM = 384 # Dimension của vector embedding

# Milvus Config
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_blocks"
VECTOR_FIELD_NAME = "embedding"
TEXT_FIELD_NAME_MILVUS = "content"
MILVUS_INDEX_PARAMS = {"index_type": "DISKANN", "metric_type": "COSINE", "params": {"search_list": 100}}
MILVUS_SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"search_list": 100}}
MILVUS_TOP_K = 7 # Số lượng lấy từ Milvus

# Elasticsearch Config
ES_HOST = "http://127.0.0.1:9200"
ES_INDEX_NAME = "my_rag_index"
ES_BM25_TOP_K = 7 # Số lượng lấy từ ES BM25

# Ranker Config
RANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RANKER_FINAL_TOP_K = 5 # Số lượng giữ lại sau re-ranking

# LLM Config
OLLAMA_MODEL_NAME = "llama3.1:8b-instruct-q6_K"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 180

# --- Xác định Device ---
print("--- Determining Device ---")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    try:
      gpu_device_str = f"cuda:{torch.cuda.current_device()}"
      device = ComponentDevice.from_str(gpu_device_str)
      print(f"CUDA available. Using: {device.to_torch_str()}")
    except Exception as e:
        print(f"CUDA error: {e}. Using CPU.")
        device = ComponentDevice.from_str("cpu")
else:
    device = ComponentDevice.from_str("cpu")
    print("CUDA not available. Using CPU.")
print("------------------------------------\n")

# --- 2. Khởi tạo Haystack Components ---
print("--- Initializing Haystack Components ---")

# -- Document Stores --
try:
    print(f"Connecting to Milvus ({MILVUS_URI})...")
    milvus_document_store = MilvusDocumentStore(
        connection_args={"uri": MILVUS_URI},
        collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD_NAME,
        text_field=TEXT_FIELD_NAME_MILVUS,
        index_params=MILVUS_INDEX_PARAMS,
        search_params=MILVUS_SEARCH_PARAMS
    )
    print(f"Milvus connected: {milvus_document_store.count_documents()} documents found.")
except Exception as e:
    print(f"Error connecting to Milvus: {e}")
    exit()

try:
    print(f"\nConnecting to Elasticsearch ({ES_HOST})...")
    es_document_store = ElasticsearchDocumentStore(hosts=[ES_HOST], index=ES_INDEX_NAME)
    print(f"Elasticsearch connected: {es_document_store.count_documents()} documents found.")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")
    exit()

# -- Embedder --
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"\nInitializing Embedder: {embedding_model_name}")
try:
    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model_name, device=device, normalize_embeddings=True)
    text_embedder.warm_up()
    print("Embedder initialized.")
except Exception as e:
    print(f"Error initializing Embedder: {e}")
    exit()

# -- Retrievers --
try:
    milvus_retriever = MilvusEmbeddingRetriever(document_store=milvus_document_store, top_k=MILVUS_TOP_K)
    print(f"Initialized Milvus Retriever (top_k={MILVUS_TOP_K})")
except Exception as e:
     print(f"Error initializing Milvus Retriever: {e}")
     exit()

try:
    bm25_retriever = ElasticsearchBM25Retriever(document_store=es_document_store, top_k=ES_BM25_TOP_K)
    print(f"Initialized BM25 Retriever (top_k={ES_BM25_TOP_K})")
except Exception as e:
    print(f"Error initializing BM25 Retriever: {e}")
    exit()

# -- Joiner --
joiner = DocumentJoiner(join_mode="concatenate")
print("Initialized Document Joiner.")

# -- Ranker --
print(f"\nInitializing Ranker: {RANKER_MODEL_NAME}")
try:
    ranker = TransformersSimilarityRanker(model=RANKER_MODEL_NAME, top_k=RANKER_FINAL_TOP_K, token=HF_TOKEN, device=device)
    ranker.warm_up()
    print(f"Ranker initialized (top_k={RANKER_FINAL_TOP_K}).")
except Exception as e:
    print(f"Error initializing Ranker: {e}")
    exit()

# -- Prompt Builder --
# Template hướng dẫn LLM trả lời dựa trên context tiếng Việt đã xếp hạng
prompt_template = """Bạn là một trợ lý AI hữu ích, chuyên về nông nghiệp và bệnh cây trồng tại Việt Nam.
Dựa *chỉ* vào các tài liệu tiếng Việt đã được xếp hạng theo mức độ liên quan dưới đây, hãy trả lời câu hỏi một cách trung thực và chính xác.
Nếu không tìm thấy thông tin trong tài liệu, hãy trả lời rõ ràng là "Thông tin không có sẵn trong tài liệu được cung cấp".

Tài liệu (Đã xếp hạng):
{% if documents %}
    {% for doc in documents %}
    ---
    Nội dung [{{ loop.index }}]: {{ doc.content }}
    {% if doc.meta and doc.meta.tags %}
    (Thẻ: {{ doc.meta.tags | join(', ') }})
    {% endif %}
    ---
    {% endfor %}
{% else %}
    Không tìm thấy tài liệu liên quan sau khi lọc và xếp hạng.
{% endif %}

Câu hỏi: {{ question }}

Hướng dẫn: Trả lời bằng **tiếng Việt**.

Câu trả lời (bằng tiếng Việt):
"""
prompt_builder = PromptBuilder(template=prompt_template)
print("Initialized Prompt Builder.")

# -- LLM Generator --
print(f"\nInitializing Ollama Generator: {OLLAMA_MODEL_NAME}")
try:
    llm_generator = OllamaGenerator(model=OLLAMA_MODEL_NAME, url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT,
                                   generation_kwargs={"num_predict": 700, "temperature": 0.6, "top_p": 0.9})
    print("Initialized Ollama Generator.")
except Exception as e:
    print(f"Error initializing Ollama Generator: {e}")
    exit()

# -- Answer Builder --
answer_builder = AnswerBuilder()
print("Initialized Answer Builder.")
print("-------------------------------------\n")

# --- 3. Xây dựng Pipeline ---
print("--- Building Hybrid RAG Pipeline ---")
pipeline = Pipeline()

# Thêm components
pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("milvus_retriever", milvus_retriever) 
pipeline.add_component("bm25_retriever", bm25_retriever)
pipeline.add_component("joiner", joiner)
pipeline.add_component("ranker", ranker)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm_generator)
pipeline.add_component("answer_builder", answer_builder)

# Kết nối luồng: Embed -> Milvus -> Joiner <- BM25 <- Ranker -> Prompt -> LLM -> Answer
pipeline.connect("text_embedder.embedding", "milvus_retriever.query_embedding")
pipeline.connect("milvus_retriever.documents", "joiner.documents")
pipeline.connect("bm25_retriever.documents", "joiner.documents")
pipeline.connect("joiner.documents", "ranker.documents")
pipeline.connect("ranker.documents", "prompt_builder.documents")
pipeline.connect("ranker.documents", "answer_builder.documents") 
pipeline.connect("prompt_builder.prompt", "llm.prompt")
pipeline.connect("llm.replies", "answer_builder.replies")

print("Pipeline built successfully.")
print("-----------------------------\n")

# --- 4. Chạy Pipeline ---
print("--- Running RAG Pipeline ---")
user_query = "Làm thế nào để trị bệnh thán thư trên cây xoài hiệu quả?"
print(f"User Query: {user_query}")

# Input ban đầu cho các điểm khởi đầu của pipeline
pipeline_input = {
    "text_embedder": {"text": user_query},
    "bm25_retriever": {"query": user_query},
    "ranker": {"query": user_query},
    "prompt_builder": {"question": user_query},
    "answer_builder": {"query": user_query}
}

try:
    start_run_time = time.time()
    result = pipeline.run(pipeline_input)
    end_run_time = time.time()
    print(f"\nPipeline execution time: {end_run_time - start_run_time:.2f} seconds")

    # Xử lý và in kết quả chính
    print("\n--- Pipeline Result ---")
    if "answer_builder" in result and result["answer_builder"]["answers"]:
         final_answer = result["answer_builder"]["answers"][0]
         print("\nGenerated Answer (Vietnamese):")
         print(final_answer.data)

         print(f"\nRetrieved & Ranked Documents (Top {RANKER_FINAL_TOP_K} used):")
         if final_answer.documents:
             for i, doc in enumerate(final_answer.documents):
                 score_str = f"{doc.score:.4f}" if hasattr(doc, 'score') and isinstance(doc.score, float) else "N/A"
                 print(f"--- Doc {i+1} (Score: {score_str}, ID: {doc.id}) ---")
                 if doc.meta: print(f"  Meta: {doc.meta}")
                 print(f"  Content Preview: {doc.content[:200]}...")
         else:
              print("  No documents associated with the answer.")

    elif not ("answer_builder" in result and result["answer_builder"]["answers"]):
         print("\nPipeline did not produce a final answer.")

except Exception as e:
    print(f"\n--- An error occurred during pipeline execution ---")
    print(e)
    traceback.print_exc()

print("\n------------------------")
print("RAG script finished.")