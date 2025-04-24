import torch
import os
import time
import traceback
from typing import List, Any

# --- Haystack Core & Standard Components ---
from haystack import Pipeline, Document, component
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder, PromptBuilder
from haystack.utils import Secret, ComponentDevice
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner, ListJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.dataclasses import ChatMessage
from haystack.components.converters import OutputAdapter

# --- Haystack Integrations ---
from haystack_integrations.components.generators.ollama import OllamaGenerator
from milvus_haystack import MilvusDocumentStore, MilvusEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever

# --- Haystack Experimental (Memory) ---
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
import re # Thêm thư viện regex để parse
# --- Custom Component ---
@component
class StringListToChatMessages:
    """Chuyển đổi List[str] thành List[ChatMessage] với vai trò 'assistant'."""
    @component.output_types(messages=List[ChatMessage])
    def run(self, replies: List[str]):
        return {"messages": [ChatMessage.from_assistant(reply) for reply in replies]}

# --- Cấu hình ---
HF_TOKEN = Secret.from_token(os.getenv("HF_TOKEN"))
EXPECTED_EMBEDDING_DIM = 384
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_blocks"
VECTOR_FIELD_NAME = "embedding"
TEXT_FIELD_NAME_MILVUS = "content"
MILVUS_INDEX_PARAMS = {"index_type": "DISKANN", "metric_type": "COSINE", "params": {"search_list": 100}}
MILVUS_SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"search_list": 100}}
MILVUS_TOP_K = 7
ES_HOST = "http://127.0.0.1:9200"
ES_INDEX_NAME = "my_rag_index"
ES_BM25_TOP_K = 7
RANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RANKER_FINAL_TOP_K = 5
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
    milvus_document_store = MilvusDocumentStore(
        connection_args={"uri": MILVUS_URI}, collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD_NAME, text_field=TEXT_FIELD_NAME_MILVUS,
        index_params=MILVUS_INDEX_PARAMS, search_params=MILVUS_SEARCH_PARAMS
    )
    print(f"Milvus connected: {milvus_document_store.count_documents()} docs.")
except Exception as e: print(f"Error connecting to Milvus: {e}"); exit()

try:
    es_document_store = ElasticsearchDocumentStore(hosts=[ES_HOST], index=ES_INDEX_NAME)
    print(f"Elasticsearch connected: {es_document_store.count_documents()} docs.")
except Exception as e: print(f"Error connecting to Elasticsearch: {e}"); exit()

# -- Embedder --
try:
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device, normalize_embeddings=True)
    text_embedder.warm_up()
    print("Embedder initialized.")
except Exception as e: print(f"Error initializing Embedder: {e}"); exit()

# -- Retrievers --
try:
    milvus_retriever = MilvusEmbeddingRetriever(document_store=milvus_document_store, top_k=MILVUS_TOP_K)
    print(f"Initialized Milvus Retriever (top_k={MILVUS_TOP_K})")
except Exception as e: print(f"Error initializing Milvus Retriever: {e}"); exit()

try:
    bm25_retriever = ElasticsearchBM25Retriever(document_store=es_document_store, top_k=ES_BM25_TOP_K)
    print(f"Initialized BM25 Retriever (top_k={ES_BM25_TOP_K})")
except Exception as e: print(f"Error initializing BM25 Retriever: {e}"); exit()

# -- Joiner --
joiner = DocumentJoiner(join_mode="concatenate")
print("Initialized Document Joiner.")

# -- Ranker --
try:
    ranker = TransformersSimilarityRanker(model=RANKER_MODEL_NAME, top_k=RANKER_FINAL_TOP_K, token=HF_TOKEN, device=device)
    ranker.warm_up()
    print(f"Ranker initialized (top_k={RANKER_FINAL_TOP_K}).")
except Exception as e: print(f"Error initializing Ranker: {e}"); exit()

# -- Memory Components --
memory_store = InMemoryChatMessageStore()
memory_retriever = ChatMessageRetriever(memory_store)
memory_writer = ChatMessageWriter(memory_store)
memory_joiner = ListJoiner(List[ChatMessage])
print("Memory components initialized.")

# -- Query Rewriting Components --
query_rewrite_template = """Dựa vào lịch sử trò chuyện dưới đây, hãy viết lại câu hỏi cuối cùng của người dùng thành một câu hỏi độc lập, đầy đủ ngữ nghĩa để có thể dùng tìm kiếm thông tin **trong cơ sở dữ liệu về nông nghiệp/bệnh cây trồng**.
- Nếu câu hỏi cuối rõ ràng và liên quan đến nông nghiệp/bệnh cây trồng (kể cả dùng đại từ như 'nó', 'bệnh đó'), hãy viết lại cho đầy đủ. Ví dụ: "cách trị bệnh đó?" -> "cách trị bệnh thán thư trên cây xoài?".
- Nếu câu hỏi cuối KHÔNG liên quan đến nông nghiệp/bệnh cây trồng (ví dụ: chào hỏi, hỏi về bản thân bạn, hỏi về câu hỏi trước đó), hãy trả về một chuỗi rỗng hoặc một từ khóa đặc biệt như "NO_REWRITE_NEEDED".
- Nếu không có lịch sử trò chuyện, giữ nguyên câu hỏi.
Chỉ trả về câu hỏi đã viết lại hoặc chuỗi rỗng/"NO_REWRITE_NEEDED".

Lịch sử trò chuyện:
{% for msg in memories %}
  {% if msg.role == 'user' %}User: {{ (msg.to_dict()).content[0].text }}{% elif msg.role == 'assistant' %}Assistant: {{ (msg.to_dict()).content[0].text }}{% endif %}
{% endfor %}

Câu hỏi cuối của người dùng: {{ query }}

Câu hỏi đã viết lại (hoặc chuỗi rỗng/NO_REWRITE_NEEDED):"""
query_rewrite_prompt_builder = PromptBuilder(template=query_rewrite_template)
try:
    query_rewrite_llm_generator = OllamaGenerator(model=OLLAMA_MODEL_NAME, url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT, generation_kwargs={"temperature": 0.05, "top_p": 0.9, "num_predict": 200})
    print("Initialized Query Rewriting LLM.")
except Exception as e: print(f"Error initializing Query Rewriting LLM: {e}"); exit()
rewrite_output_adapter = OutputAdapter(template="{{ replies[0] }}", output_type=str)
print("Query rewriting components initialized.")

# -- Chat Prompt Builder (cho LLM cuối) --
chat_prompt_template = [
    ChatMessage.from_system(
        "Bạn là một trợ lý AI hữu ích, chuyên về nông nghiệp và bệnh cây trồng tại Việt Nam. "
        "Hãy trò chuyện và trả lời câu hỏi của người dùng một cách tự nhiên. "
        "QUAN TRỌNG: Luôn xem xét LỊCH SỬ TRÒ CHUYỆN và THÔNG TIN BỔ SUNG (biến `identified_disease`) nếu có."
        "- Nếu `identified_disease` chứa 'healthy', hãy thông báo cây trông khỏe mạnh và không có dấu hiệu bệnh dựa trên hình ảnh, tránh nói tên healthy ra" # Xử lý healthy
        "- Nếu `identified_disease` được cung cấp (và không phải healthy), hãy ưu tiên TÀI LIỆU THAM KHẢO để trả lời về bệnh đó. Nếu không có tài liệu, hãy nói rõ là không tìm thấy thông tin về bệnh này trong tài liệu."
        "- Nếu không có `identified_disease` và câu hỏi liên quan đến nông nghiệp/bệnh cây trồng, hãy dùng TÀI LIỆU THAM KHẢO để trả lời."
        "- Nếu câu hỏi là hội thoại thông thường (ví dụ: 'chào', 'bạn là ai?', 'tôi vừa hỏi gì?'), hãy trả lời trực tiếp dựa trên LỊCH SỬ TRÒ CHUYỆN và vai trò của bạn." # Xử lý câu hỏi meta
        "- Nếu không có thông tin từ bất kỳ nguồn nào (tài liệu, lịch sử) để trả lời một câu hỏi cụ thể, hãy nói bạn không biết hoặc không có thông tin."
        "- Trả lời bằng tiếng Việt."
    ),
    ChatMessage.from_user(
        """**Lịch sử trò chuyện (Để hiểu ngữ cảnh):**
{% for msg in memories %}
  {% if msg.role == 'user' %}User: {{ (msg.to_dict()).content[0].text }} {% else %}Assistant: {{ (msg.to_dict()).content[0].text }} {% endif %}
{% endfor %}

{% if identified_disease %}
**Thông tin bổ sung (Từ Vision Model):** Trạng thái/Bệnh được xác định là: **{{ identified_disease }}**
{% endif %}

**Tài liệu tham khảo (Chỉ dùng nếu câu hỏi liên quan và không phải trường hợp 'healthy'):**
{% if documents and (not identified_disease or 'healthy' not in identified_disease.lower()) %} {# Chỉ hiển thị docs nếu cần RAG #}
    {% for doc in documents %}
    ---
    Nội dung [{{ loop.index }}]: {{ doc.content }}
    {% if doc.meta and doc.meta.tags %} (Thẻ: {{ doc.meta.tags | join(', ') }}) {% endif %}
    ---
    {% endfor %}
{% elif identified_disease and 'healthy' not in identified_disease.lower() %}
    (Không tìm thấy tài liệu trong cơ sở dữ liệu về "{{ identified_disease }}")
{% elif not identified_disease %}
     (Không tìm thấy tài liệu liên quan cho câu hỏi này trong cơ sở dữ liệu)
{% endif %}

**Câu hỏi hiện tại của người dùng:** {{ query }}

**Câu trả lời của bạn (bằng tiếng Việt):**"""
    )
]
chat_prompt_builder = ChatPromptBuilder(template=chat_prompt_template)
print("Initialized ChatPromptBuilder.")

# -- Adapters cho LLM và Memory --
message_to_string_adapter = OutputAdapter(template="{{ (messages[-1].to_dict())['content'][0]['text'] }}", output_type=str)
str_to_chat_converter = StringListToChatMessages()
print("Initialized Adapters.")

# -- LLM Generator (chính) --
try:
    llm_generator = OllamaGenerator(model=OLLAMA_MODEL_NAME, url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT, generation_kwargs={"num_predict": 700, "temperature": 1, "top_p": 0.9})
    print("Initialized Main Ollama Generator.")
except Exception as e: print(f"Error initializing Main Ollama Generator: {e}"); exit()

# -- Answer Builder --
answer_builder = AnswerBuilder()
print("Initialized Answer Builder.")
print("-------------------------------------\n")

# --- 3. Xây dựng Pipeline ---
print("--- Building Conversational RAG Pipeline with Query Rewriting ---")
pipeline = Pipeline()

# Thêm components
pipeline.add_component("memory_retriever", memory_retriever)
pipeline.add_component("query_rewrite_prompt_builder", query_rewrite_prompt_builder)
pipeline.add_component("query_rewrite_llm", query_rewrite_llm_generator)
pipeline.add_component("rewrite_output_adapter", rewrite_output_adapter)
pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("milvus_retriever", milvus_retriever)
pipeline.add_component("bm25_retriever", bm25_retriever)
pipeline.add_component("joiner", joiner)
pipeline.add_component("ranker", ranker)
pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
pipeline.add_component("message_to_string_adapter", message_to_string_adapter)
pipeline.add_component("llm", llm_generator)
pipeline.add_component("str_to_chat_converter", str_to_chat_converter)
pipeline.add_component("memory_joiner", memory_joiner)
pipeline.add_component("memory_writer", memory_writer)
pipeline.add_component("answer_builder", answer_builder)

# --- Kết nối Pipeline ---
pipeline.connect("memory_retriever.messages", "query_rewrite_prompt_builder.memories")
pipeline.connect("query_rewrite_prompt_builder.prompt", "query_rewrite_llm.prompt")
pipeline.connect("query_rewrite_llm.replies", "rewrite_output_adapter.replies")
pipeline.connect("rewrite_output_adapter.output", "text_embedder.text")
pipeline.connect("rewrite_output_adapter.output", "bm25_retriever.query")
pipeline.connect("rewrite_output_adapter.output", "ranker.query")
pipeline.connect("text_embedder.embedding", "milvus_retriever.query_embedding")
pipeline.connect("milvus_retriever.documents", "joiner.documents")
pipeline.connect("bm25_retriever.documents", "joiner.documents")
pipeline.connect("joiner.documents", "ranker.documents")
pipeline.connect("ranker.documents", "chat_prompt_builder.documents")
pipeline.connect("ranker.documents", "answer_builder.documents")
pipeline.connect("memory_retriever.messages", "chat_prompt_builder.memories")
pipeline.connect("chat_prompt_builder.prompt", "message_to_string_adapter.messages")
pipeline.connect("message_to_string_adapter.output", "llm.prompt")
pipeline.connect("llm.replies", "str_to_chat_converter.replies")
pipeline.connect("str_to_chat_converter.messages", "memory_joiner.values")
pipeline.connect("memory_joiner.values", "memory_writer.messages")
pipeline.connect("llm.replies", "answer_builder.replies")

print("Pipeline built successfully.")
print("-----------------------------\n")
# --- 4. Chạy Pipeline ---
print("\n--- Starting Conversation ---")
print("Nhập 'quit' hoặc 'exit' để kết thúc.")
print("Định dạng gợi ý khi có Vision: <Câu hỏi>? ( <Tên bệnh từ Vision> )")

while True:
    user_input_raw = input("🧑 User: ")
    if user_input_raw.lower() in ["quit", "exit"]:
        break

    # --- THÊM: Parse Input để tách câu hỏi và tên bệnh ---
    original_query = user_input_raw # Lưu lại input gốc
    disease_name = None
    # Sử dụng regex để tìm nội dung trong dấu ngoặc đơn cuối cùng
    match = re.search(r'\(([^)]+)\)\s*$', user_input_raw)
    if match:
        disease_name = match.group(1).strip() # Lấy tên bệnh
        print(f"   (Debug: Identified disease: '{disease_name}')")

    query_for_rag = disease_name if disease_name else original_query

    pipeline_input = {
        "query_rewrite_prompt_builder": {"query": original_query}, # Viết lại dựa trên câu hỏi gốc
        "chat_prompt_builder": {
            "query": original_query,                             # Câu hỏi gốc vào prompt cuối
            "identified_disease": disease_name                   # Tên bệnh đã xác định vào prompt cuối
        },
        "memory_joiner": {"values": [ChatMessage.from_user(original_query)]}, # Ghi nhớ câu hỏi gốc
        "answer_builder": {"query": original_query}              # Câu hỏi gốc cho output Answer
    }
    try:
        start_run_time = time.time()
        result = pipeline.run(pipeline_input, include_outputs_from=["answer_builder"])
        end_run_time = time.time()

        # Xử lý kết quả
        if "answer_builder" in result and result["answer_builder"]["answers"]:
             final_answer = result["answer_builder"]["answers"][0]
             print(f"🤖 Assistant: {final_answer.data}")
        else:
             print("🤖 Assistant: Xin lỗi, tôi không thể tạo câu trả lời.")

    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(e)
        traceback.print_exc()

print("\n------------------------")
print("Conversation finished.")