import torch
import os
import time # Thêm import time
# --- Import từ Haystack Core và Components chuẩn ---
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.utils import Secret, ComponentDevice
# from haystack.components.generators import HuggingFaceLocalGenerator # Không dùng nữa
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.dataclasses import Document
from haystack_integrations.components.generators.ollama import OllamaGenerator
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

# --- Không cần load LLM thủ công ---

# --- Xác định Device cho Embedder ---
print("--- Determining Device for Embedder ---")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = ComponentDevice.from_str("cuda:0")
    print(f"CUDA available. Embedder will use: {device.to_torch_str()}")
else:
    device = ComponentDevice.from_str("cpu")
    print("CUDA not available or no GPUs detected. Embedder will use: CPU")
print("------------------------------------\n")


# --- 2. Khởi tạo Haystack Components ---
print("--- Initializing Haystack Components (Vietnamese Focus) ---")

# -- Document Store (Milvus) --
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_blocks"
VECTOR_FIELD_NAME = "embedding" # Tên trường vector trong Milvus schema của bạn
EXPECTED_EMBEDDING_DIM = 384 # Phải khớp với embedding của bạn
# Index và Search Params phải khớp với index bạn đã tạo trong Milvus
EXPECTED_INDEX_PARAMS = {"index_type": "DISKANN", "metric_type": "COSINE", "params": {"search_list": 100}}
EXPECTED_SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"search_list": 100}}
TEXT_FIELD_NAME = "content"
try:
    print(f"Connecting to Milvus using URI: {MILVUS_URI}")
    document_store = MilvusDocumentStore(
        connection_args={"uri": MILVUS_URI},
        collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD_NAME,
        index_params=EXPECTED_INDEX_PARAMS, # Cung cấp để Haystack biết cách tương tác
        search_params=EXPECTED_SEARCH_PARAMS,
        text_field=TEXT_FIELD_NAME
    )
    # Kiểm tra số lượng documents
    doc_count = document_store.count_documents()
    print(f"Connected to Milvus: {doc_count} documents found in '{document_store.collection_name}'.")
    if doc_count == 0:
        print("Warning: Milvus collection is empty. Retrieval will not find relevant documents.")
except Exception as e:
    print(f"Error connecting to Milvus or accessing collection '{COLLECTION_NAME}': {e}")
    print(f"Please ensure Milvus is running at URI '{MILVUS_URI}', collection exists with field '{VECTOR_FIELD_NAME}' (dim={EXPECTED_EMBEDDING_DIM}) and a compatible index.")
    exit()


# -- Embedder --
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Model 384 chiều, đa ngôn ngữ
print(f"\n!!! Initializing Text Embedder: {embedding_model_name} - Ensure this matches your Milvus data embeddings !!!")
try:
    text_embedder = SentenceTransformersTextEmbedder(
        model=embedding_model_name,
        device=device, # Sử dụng device đã xác định (cuda hoặc cpu)
        normalize_embeddings=True # Quan trọng cho cosine similarity
    )
    print("Warming up Text Embedder (loading model)...")
    text_embedder.warm_up() # Tải model vào bộ nhớ
    print("Text Embedder initialized and warmed up.")
except Exception as e:
    print(f"Error initializing/warming up Text Embedder: {e}")
    exit()

# -- Retriever (Milvus) --
try:
    milvus_retriever = MilvusEmbeddingRetriever(
        document_store=document_store,
        top_k=5 # Số lượng documents muốn truy xuất
    )
    print(f"Initialized Milvus Retriever (top_k={milvus_retriever.top_k})")
except Exception as e:
     print(f"Error initializing MilvusEmbeddingRetriever: {e}")
     exit()


# -- Prompt Builder --
prompt_template = """Bạn là một trợ lý AI hữu ích, có kiến thức về nông nghiệp và bệnh cây trồng ở Việt Nam.
Hãy trả lời câu hỏi một cách trung thực và chính xác *chỉ dựa trên* các tài liệu tiếng Việt được cung cấp.
Nếu tài liệu không chứa câu trả lời, hãy nói rõ rằng thông tin không có sẵn trong các tài liệu được cung cấp.

Tài liệu:
{% if documents %}
    {% for doc in documents %}
    ---
    Nội dung: {{ doc.content }}
    {% if doc.meta and doc.meta.tags %}
    Thẻ: {{ doc.meta.tags | join(', ') }}
    {% endif %}
    ---
    {% endfor %}
{% else %}
    Không tìm thấy tài liệu liên quan.
{% endif %}

Câu hỏi: {{ question }}

Hướng dẫn: Cung cấp câu trả lời bằng **tiếng Việt**.

Câu trả lời (bằng tiếng Việt):
"""
prompt_builder = PromptBuilder(template=prompt_template)
print("Initialized Prompt Builder (Vietnamese focus).")


# -- Ollama Generator --
OLLAMA_MODEL_NAME = "llama3.1:8b-instruct-q6_K" # Đảm bảo bạn đã `ollama pull` model này

OLLAMA_URL = "http://localhost:11434"

print(f"\nInitializing Ollama Generator with model: {OLLAMA_MODEL_NAME}")
try:
    llm_generator = OllamaGenerator(
        model=OLLAMA_MODEL_NAME,
        url=OLLAMA_URL,
        generation_kwargs={
            "num_predict": 700,      # Số token tối đa Ollama tạo ra
            "temperature": 0.6,
            "top_p": 0.9,
            # Có thể thêm các tham số khác của Ollama tại đây nếu cần
            # "stop": ["\nUser:", "Câu hỏi:"]
        },
        timeout=120 # Giây, tăng nếu cần cho model/tác vụ phức tạp
    )
    # OllamaGenerator không yêu cầu warm_up() như HuggingFaceLocalGenerator
    print("Initialized Ollama Generator.")
except Exception as e:
    print(f"Error initializing OllamaGenerator: {e}")
    print(f"Ensure Ollama server is running at '{OLLAMA_URL.rsplit('/', 1)[0]}' and model '{OLLAMA_MODEL_NAME}' is available (use 'ollama list').")
    exit()


# -- Answer Builder --
answer_builder = AnswerBuilder()
print("Initialized Answer Builder.")

print("-------------------------------------\n")


# --- 3. Xây dựng Pipeline ---
print("--- Building Haystack Pipeline ---")
pipeline = Pipeline()
pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("retriever", milvus_retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm_generator) # Sử dụng Ollama Generator
pipeline.add_component("answer_builder", answer_builder)

# Kết nối các component
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.prompt")
pipeline.connect("llm.replies", "answer_builder.replies")
pipeline.connect("retriever.documents", "answer_builder.documents")

print("Pipeline built successfully.")
print("-----------------------------\n")


# --- 4. Chạy Pipeline ---
print("--- Running RAG Pipeline (Vietnamese Query) ---")
user_query = "Cần phun thuốc định kỳ bao nhiêu ngày ngày một lần để kiểm soát thrips" # Ví dụ câu hỏi

print(f"User Query: {user_query}")

# Dữ liệu đầu vào cho pipeline
pipeline_input = {
    "text_embedder": {"text": user_query},
    "prompt_builder": {"question": user_query},
    "answer_builder": {"query": user_query}
}

try:
    # Đo thời gian chạy
    start_run_time = time.time()
    result = pipeline.run(pipeline_input)
    end_run_time = time.time()
    print(f"\nPipeline execution time: {end_run_time - start_run_time:.2f} seconds")

    # Xử lý và in kết quả
    print("\n--- Pipeline Result ---")
    if "answer_builder" in result and "answers" in result["answer_builder"] and result["answer_builder"]["answers"]:
         final_answer = result["answer_builder"]["answers"][0]
         print("\nGenerated Answer (Vietnamese):")
         # Truy cập dữ liệu câu trả lời từ đối tượng Answer
         print(final_answer.data) # final_answer.data chứa chuỗi string trả lời

         print("\nRetrieved Documents:")
         if final_answer.documents: # Kiểm tra xem có documents được gắn vào answer không
             for i, doc in enumerate(final_answer.documents):
                 print(f"--- Document {i+1} (ID: {doc.id}) ---")
                 if doc.meta: print(f"  Meta: {doc.meta}")
                 print(f"  Content Preview: {doc.content[:200]}...") # In 200 ký tự đầu
         else:
              print("  No documents were associated with this answer (check retriever results).")

    # In thêm kết quả thô từ retriever để debug nếu cần
    if "retriever" in result and "documents" in result["retriever"]:
         print("\nRaw Retrieved Documents (for debugging):")
         if result["retriever"]["documents"]:
              for i, doc in enumerate(result["retriever"]["documents"]):
                  print(f"  [Retrieved {i+1}] ID: {doc.id}, Content: {doc.content[:100]}...")
         else:
              print("  Retriever found no documents.")

    # In nếu không có câu trả lời cuối cùng
    elif not ("answer_builder" in result and "answers" in result["answer_builder"] and result["answer_builder"]["answers"]):
         print("\nPipeline did not produce a final answer via AnswerBuilder.")
         print("Full result dictionary (truncated):")
         for key, value in result.items():
             # Giới hạn độ dài output để không quá dài
             print(f"- {key}: {str(value)[:500]}{'...' if len(str(value)) > 500 else ''}")

except Exception as e:
    print(f"\n--- An error occurred during pipeline execution ---")
    print(e)
    import traceback
    traceback.print_exc()

print("\n------------------------")
print("RAG script finished.")