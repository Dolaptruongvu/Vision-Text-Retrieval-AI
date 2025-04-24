import torch
import os
import time
import traceback

# --- Haystack Core ---
from haystack import Pipeline, Document
from haystack.components.builders import AnswerBuilder
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
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
# --- Thêm Imports cho Memory và Chat ---
from haystack.dataclasses import ChatMessage # Để làm việc với tin nhắn chat
from haystack.components.builders import ChatPromptBuilder # Builder cho prompt dạng chat
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack.components.joiners import ListJoiner
from typing import List
from haystack.components.converters import OutputAdapter

from haystack import component
from haystack.dataclasses import ChatMessage


from dotenv import load_dotenv
import os

load_dotenv()
@component
class StringListToChatMessages:
    """
    Chuyển đổi một List[str] thành List[ChatMessage] với vai trò 'assistant'.
    """
    @component.output_types(messages=List[ChatMessage])
    def run(self, replies: List[str]):
        """
        Thực hiện chuyển đổi.
        :param replies: Danh sách các chuỗi string trả lời từ LLM.
        :return: Dictionary chứa key 'messages' với giá trị là List[ChatMessage].
        """
        assistant_messages = [ChatMessage.from_assistant(reply) for reply in replies]
        return {"messages": assistant_messages}
    
# --- THÊM: Khởi tạo Component tùy chỉnh ---
str_to_chat_converter = StringListToChatMessages()
print("Initialized StringListToChatMessages converter.")
# --- Cấu hình ---
hf_token = os.getenv("HF_TOKEN")
HF_TOKEN = Secret.from_token(hf_token) # Token Hugging Face cho Ranker
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

# --- THÊM: Khởi tạo Components Memory ---
print("\nInitializing Memory Components...")
memory_store = InMemoryChatMessageStore()
memory_retriever = ChatMessageRetriever(memory_store)
memory_writer = ChatMessageWriter(memory_store)
# Khởi tạo ListJoiner để gộp tin nhắn user và assistant trước khi ghi
memory_joiner = ListJoiner(List[ChatMessage]) # Chỉ định type là List[ChatMessage]
print("Memory components initialized.")
# ---------------------------------------

# --- THÊM: Khởi tạo OutputAdapter ---
# Adapter này sẽ lấy nội dung của tin nhắn cuối cùng từ list ChatMessage
# mà ChatPromptBuilder tạo ra và chuyển thành string cho OllamaGenerator
message_to_string_adapter = OutputAdapter(
    template="{{ (messages[-1].to_dict())['content'][0]['text'] }}", # Lấy content của message cuối cùng
    output_type=str # Đảm bảo output là string
)
print("Initialized OutputAdapter (ChatMessage List to String).")

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

# -- Chat Prompt Builder --
# Template hướng dẫn LLM trả lời dựa trên context tiếng Việt đã xếp hạng
chat_prompt_template = [
    ChatMessage.from_system(
        "Bạn là một trợ lý AI hữu ích, chuyên về nông nghiệp và bệnh cây trồng tại Việt Nam. "
        "Trả lời câu hỏi chỉ dựa vào lịch sử trò chuyện và các tài liệu tiếng Việt đã xếp hạng được cung cấp."
    ),
    ChatMessage.from_user(
        """Dựa vào lịch sử trò chuyện và các tài liệu đã xếp hạng dưới đây, hãy trả lời câu hỏi của người dùng.
Nếu thông tin không có trong tài liệu hoặc lịch sử, hãy nói rõ là không tìm thấy.

Lịch sử trò chuyện:
{% for msg in memories %}
  {% if msg.role == 'user' %}User: {{ (msg.to_dict()).content[0].text }} {% else %}Assistant: {{ (msg.to_dict()).content[0].text }} {% endif %}
{% endfor %}

Tài liệu (Đã xếp hạng):
{% if documents %}
    {% for doc in documents %}
    ---
    Nội dung [{{ loop.index }}]: {{ doc.content }}
    {% if doc.meta and doc.meta.tags %} (Thẻ: {{ doc.meta.tags | join(', ') }}) {% endif %}
    ---
    {% endfor %}
{% else %}
    (Không có tài liệu liên quan được tìm thấy)
{% endif %}

Câu hỏi: {{ query }}

Câu trả lời (bằng tiếng Việt):"""
    )
]


# Khởi tạo ChatPromptBuilder
# Biến 'query', 'documents', 'memories' sẽ được truyền vào
chat_prompt_builder = ChatPromptBuilder(template=chat_prompt_template)
print("Initialized ChatPromptBuilder (replaces PromptBuilder).")

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

# --- 3. Xây dựng Pipeline (CẬP NHẬT LẠI KẾT NỐI) ---
print("--- Building Conversational RAG Pipeline ---")
pipeline = Pipeline()

# Thêm components (bao gồm cả adapter)
pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("milvus_retriever", milvus_retriever)
pipeline.add_component("bm25_retriever", bm25_retriever)
pipeline.add_component("joiner", joiner)
pipeline.add_component("ranker", ranker)
pipeline.add_component("chat_prompt_builder", chat_prompt_builder) # Vẫn dùng ChatPromptBuilder
pipeline.add_component("message_to_string_adapter", message_to_string_adapter) # THÊM Adapter
pipeline.add_component("llm", llm_generator)
pipeline.add_component("str_to_chat_converter", str_to_chat_converter)
pipeline.add_component("answer_builder", answer_builder)
pipeline.add_component("memory_retriever", memory_retriever)
pipeline.add_component("memory_writer", memory_writer)
pipeline.add_component("memory_joiner", memory_joiner)

# Kết nối luồng RAG (CẬP NHẬT)
pipeline.connect("text_embedder.embedding", "milvus_retriever.query_embedding")
pipeline.connect("milvus_retriever.documents", "joiner.documents")
pipeline.connect("bm25_retriever.documents", "joiner.documents")
pipeline.connect("joiner.documents", "ranker.documents")
pipeline.connect("ranker.documents", "chat_prompt_builder.documents")
pipeline.connect("ranker.documents", "answer_builder.documents")
# KẾT NỐI MỚI: ChatPromptBuilder -> Adapter -> LLM
pipeline.connect("chat_prompt_builder.prompt", "message_to_string_adapter.messages") # Output prompt (List[ChatMessage]) vào Adapter
pipeline.connect("message_to_string_adapter.output", "llm.prompt") # Output (str) từ Adapter vào LLM

# Kết nối luồng Memory (Giữ nguyên)
# Kết nối luồng Memory (CẬP NHẬT)
pipeline.connect("memory_retriever.messages", "chat_prompt_builder.memories")
pipeline.connect("llm.replies", "str_to_chat_converter.replies") 
pipeline.connect("str_to_chat_converter.messages", "memory_joiner.values")
# (Tin nhắn user vẫn được đưa vào memory_joiner từ input)
pipeline.connect("memory_joiner.values", "memory_writer.messages") # Joiner -> Writer

# Kết nối Output cuối cùng (Giữ nguyên)
pipeline.connect("llm.replies", "answer_builder.replies")

print("Conversational pipeline built successfully.")
#---------------------------------------------

# --- 4. Chạy Pipeline ---
print("\n--- Starting Conversation ---")
print("Nhập 'quit' hoặc 'exit' để kết thúc.")

while True:
    user_query = input("🧑 User: ")
    if user_query.lower() in ["quit", "exit"]:
        break

    # Dữ liệu đầu vào cho pipeline (CẬP NHẬT)
    pipeline_input = {
        # Input cho nhánh RAG
        "text_embedder": {"text": user_query},
        "bm25_retriever": {"query": user_query},
        "ranker": {"query": user_query},
        "chat_prompt_builder": {"query": user_query}, # Input query cho prompt builder
        # Input cho nhánh Memory (đưa tin nhắn user vào joiner)
        "memory_joiner": {"values": [ChatMessage.from_user(user_query)]},
        # Input query cho AnswerBuilder (để giữ cấu trúc output)
        "answer_builder": {"query": user_query}
    }

    try:
        start_run_time = time.time()
        # Chỉ cần lấy output từ AnswerBuilder
        result = pipeline.run(pipeline_input, include_outputs_from=["answer_builder"])
        end_run_time = time.time()
        print(f"(Debug: Pipeline took {end_run_time - start_run_time:.2f}s)")

        # Xử lý kết quả từ AnswerBuilder
        if "answer_builder" in result and result["answer_builder"]["answers"]:
             final_answer = result["answer_builder"]["answers"][0]
             print(f"🤖 Assistant: {final_answer.data}") # In câu trả lời text

             # (Tùy chọn) In tài liệu tham khảo nếu muốn
             # if final_answer.documents:
             #    print("   (Debug: Relevant documents found)")
             #    # for doc in final_answer.documents: print(f"     - {doc.id}")

        else:
             print("🤖 Assistant: Xin lỗi, tôi không thể tạo câu trả lời.")

    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(e)
        traceback.print_exc()

print("\n------------------------")
print("Conversation finished.")
#------------------------------------