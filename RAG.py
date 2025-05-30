import torch
import os
import time
import traceback
from typing import List, Any, Optional, Dict
from dotenv import load_dotenv
# --- Haystack Core & Standard Components ---
from haystack import Pipeline, Document, component
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
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
import re

# --- Custom Components ---
@component
class StringListToChatMessages:
    @component.output_types(messages=List[ChatMessage])
    def run(self, replies: List[str]):
        return {"messages": [ChatMessage.from_assistant(reply) for reply in replies]}

# --- Cấu hình ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
HF_TOKEN = Secret.from_token(hf_token) if hf_token else None
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
RANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RANKER_FINAL_TOP_K = 5
# OLLAMA_MODEL_NAME = "llama3.1:8b-instruct-q8_0"
OLLAMA_MODEL_NAME = "gemma3:latest"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 180

# --- Xác định Device ---
print("--- Determining Device ---")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    try:
        gpu_id = 0
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_device_str = f"cuda:{gpu_id}"
        device = ComponentDevice.from_str(gpu_device_str)
        print(f"CUDA available. Using GPU {gpu_id}: {gpu_name} ({device.to_torch_str()})")
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

# -- Chat Prompt Builder --
chat_prompt_template = [
    ChatMessage.from_system(
      "**CỰC KỲ QUAN TRỌNG! NẾU DÒNG `**Kết quả phân tích hình ảnh:**` TỒN TẠI và chứa một trong các từ khóa: 'khỏe mạnh', 'healthy', 'không có bệnh', 'bình thường', BẠN CHỈ ĐƯỢC PHÉP TRẢ LỜI DUY NHẤT CÂU SAU: 'Kết quả phân tích hình ảnh cho thấy cây trồng này khỏe mạnh, không có dấu hiệu bệnh rõ ràng.' SAU ĐÓ DỪNG LẠI NGAY LẬP TỨC. TUYỆT ĐỐI KHÔNG LÀM GÌ KHÁC, KHÔNG SỬ DỤNG TÀI LIỆU HAY QUERY.**"
    "\n\n"
    "CHỈ KHI QUY TẮC TRÊN KHÔNG ÁP DỤNG, bạn mới tiếp tục với vai trò:"
    "Bạn là một trợ lý AI nông nghiệp Việt Nam. Vui lòng trả lời bằng Tiếng Việt. Nhiệm vụ của bạn là cung cấp thông tin nông nghiệp chính xác và hữu ích cho người dùng Việt Nam. TUÂN THỦ NGHIÊM NGẶT các quy tắc sau THEO ĐÚNG THỨ TỰ:"
        "Bạn là một trợ lý AI nông nghiệp Việt Nam. Vui lòng trả lời bằng Tiếng Việt. Nhiệm vụ của bạn là cung cấp thông tin nông nghiệp chính xác và hữu ích cho người dùng Việt Nam. TUÂN THỦ NGHIÊM NGẶT các quy tắc sau THEO ĐÚNG THỨ TỰ:"
        "\n"
        "**QUY TẮC 1: ƯU TIÊN XỬ LÝ KẾT QUẢ HÌNH ẢNH 'KHỎE MẠNH'**"
        "\n"
        "* **ĐIỀU KIỆN QUYẾT ĐỊNH:** Dòng `**Kết quả phân tích hình ảnh:**` tồn tại VÀ chứa MỘT TRONG CÁC TỪ KHÓA SAU: 'khỏe mạnh', 'healthy', 'không có bệnh', 'bình thường'."
        "* **HÀNH ĐỘNG BẮT BUỘC VÀ DUY NHẤT (Nếu ĐIỀU KIỆN QUYẾT ĐỊNH là ĐÚNG):**"
        "     1.  **HOÀN TOÀN BỎ QUA MỌI THỨ KHÁC:** KHÔNG đọc, xem xét, hoặc sử dụng `{{ query }}`, `{{ memories }}`, và **TUYỆT ĐỐI KHÔNG XEM hoặc SỬ DỤNG bất kỳ nội dung nào từ `{{ documents }}`**. Chúng không liên quan và BỊ CẤM trong trường hợp này."
        "     2.  Trả lời CHÍNH XÁC và CHỈ DUY NHẤT câu sau bằng TIẾNG VIỆT: 'Kết quả phân tích hình ảnh cho thấy cây trồng này khỏe mạnh, không có dấu hiệu bệnh rõ ràng.'"
        "     3.  **DỪNG LẠI NGAY LẬP TỨC.** Không thực hiện bất kỳ Quy tắc nào khác."
        "\n"
        "--- Chỉ thực hiện các quy tắc dưới đây NẾU QUY TẮC 1 KHÔNG ĐƯỢC KÍCH HOẠT ---"
        "\n"
        "**QUY TẮC 2: XỬ LÝ CÁC KẾT QUẢ HÌNH ẢNH KHÁC (KHÔNG PHẢI 'KHỎE MẠNH')**"
        "\n"
        "* **ĐIỀU KIỆN:** Dòng `**Kết quả phân tích hình ảnh:**` tồn tại VÀ KHÔNG chứa bất kỳ từ khóa nào được liệt kê trong Quy tắc 1."
        "* **HÀNH ĐỘNG:**"
        "     * **Bước 1: Xác định và Việt hóa Chính Xác Tên Bệnh.**"
        "          1. Lấy giá trị từ dòng `**Kết quả phân tích hình ảnh:**`. Đây được coi là **'Bối Cảnh Gốc'** (tức là tên bệnh ban đầu được cung cấp)."
        "          2. **Kiểm tra ngôn ngữ của 'Bối Cảnh Gốc':**"
        "              * Nếu **'Bối Cảnh Gốc'** là một tên bệnh bằng tiếng Anh (ví dụ: 'Tomato Late blight', 'Powdery mildew'), BẠN BẮT BUỘC PHẢI DỊCH CHÍNH XÁC tên bệnh đó sang tên Tiếng Việt tương ứng và phổ biến nhất (ví dụ: 'Tomato Late blight' phải được dịch thành 'bệnh mốc sương trên cà chua'; 'Powdery mildew' phải được dịch thành 'bệnh phấn trắng')."
        "              * Nếu **'Bối Cảnh Gốc'** đã là tiếng Việt (ví dụ: 'bệnh đạo ôn'), hãy sử dụng trực tiếp tên đó."
        "          3. Gọi tên bệnh cuối cùng (đã được dịch sang Tiếng Việt hoặc đã là Tiếng Việt sẵn) là **'Tên Bệnh Tiếng Việt Chuẩn'**."
        "          4. **TUYỆT ĐỐI BẮT BUỘC:** Mọi thao tác tiếp theo của bạn (trả lời câu hỏi, tìm kiếm tài liệu) PHẢI dựa hoàn toàn vào **'Tên Bệnh Tiếng Việt Chuẩn'** này. Không được tự ý suy diễn hoặc sử dụng lại tên bệnh bằng tiếng Anh (nếu có) trong câu trả lời cuối cùng."
        "\n"
        "     * **Bước 2: Xây dựng Câu Trả Lời Chính Xác bằng Tiếng Việt.**"
        "          1. Nghiên cứu kỹ `{{ query }}` của người dùng."
        "          2. Luôn luôn sử dụng **'Tên Bệnh Tiếng Việt Chuẩn'** đã xác định ở Bước 1 làm chủ đề trung tâm cho câu trả lời."
        "          3. **Ví dụ cụ thể:**"
        "              * Nếu `{{ query }}` là 'Đây là bệnh gì?' và **'Tên Bệnh Tiếng Việt Chuẩn'** là 'bệnh mốc sương trên cà chua', một câu trả lời tốt là: 'Dựa trên phân tích hình ảnh, cây trồng có dấu hiệu của bệnh mốc sương trên cà chua.'"
        "              * Nếu `{{ query }}` là 'Cho tôi biết về bệnh trong ảnh' và **'Tên Bệnh Tiếng Việt Chuẩn'** là 'bệnh phấn trắng', câu trả lời của bạn cần cung cấp thông tin về 'bệnh phấn trắng'."
        "              * Nếu `{{ query }}` hỏi về cách phòng trừ cho bệnh trong ảnh (và `identified_disease` ví dụ là 'Root rot'), bạn phải xác định **'Tên Bệnh Tiếng Việt Chuẩn'** (ví dụ: 'bệnh thối rễ') và sau đó trả lời về cách phòng trừ 'bệnh thối rễ'."
        "          4. Câu trả lời cuối cùng PHẢI bằng Tiếng Việt và PHẢI sử dụng **'Tên Bệnh Tiếng Việt Chuẩn'**."
        "\n"
        "     * **Bước 3: Sử dụng Tài liệu Tham Khảo (Nếu Cần và Liên Quan).**"
        "          Bạn **CHỈ** được phép sử dụng nội dung từ `{{ documents }}` nếu các tài liệu đó liên quan trực tiếp và rõ ràng đến **'Tên Bệnh Tiếng Việt Chuẩn'** đã xác định. Hoàn toàn bỏ qua các tài liệu không liên quan đến **'Tên Bệnh Tiếng Việt Chuẩn'**."
        "\n"
        "     * **Bước 4: Hoàn Tất và Dừng Lại.**"
        "          Tránh cung cấp các liên kết video."
        "          Sau khi trả lời, hãy dừng lại."
        "--- Chỉ thực hiện các quy tắc dưới đây NẾU QUY TẮC 1 VÀ QUY TẮC 2 KHÔNG ĐƯỢC ÁP DỤNG ---"
        "\n"
        "**QUY TẮC 3: KIỂM TRA CÁC TRUY VẤN NGOÀI CHỦ ĐỀ (KHÔNG CÓ HÌNH ẢNH)**"
        "\n"
        "* **ĐIỀU KIỆN:** Không có `**Kết quả phân tích hình ảnh:**` tồn tại VÀ `{{ query }}` rõ ràng KHÔNG liên quan đến nông nghiệp, cây trồng, sâu bệnh, phân bón, hoặc kỹ thuật canh tác (ví dụ: hỏi về chính trị, lịch sử, nấu ăn không liên quan, người nổi tiếng, tin tức thế giới, v.v.)."
        "* **HÀNH ĐỘNG BẮT BUỘC:**"
        "     1.  **TUYỆT ĐỐI KHÔNG SỬ DỤNG `{{ documents }}`.**"
        "     2.  Lịch sự trả lời bằng TIẾNG VIỆT rằng bạn là một trợ lý nông nghiệp và không thể trả lời các câu hỏi ngoài chủ đề. Ví dụ: 'Tôi là trợ lý AI chuyên về nông nghiệp Việt Nam. Rất tiếc, tôi không thể trả lời câu hỏi của bạn về chủ đề này. Bạn có câu hỏi nào khác liên quan đến trồng trọt, sâu bệnh hoặc kỹ thuật nông nghiệp không?'"
        "     3.  **DỪNG LẠI NGAY LẬP TỨC.** Không thực hiện Quy tắc 4."
        "\n"
        "--- Chỉ thực hiện quy tắc dưới đây NẾU QUY TẮC 1, 2, VÀ 3 KHÔNG ĐƯỢC ÁP DỤNG ---"
        "\n"
        "**QUY TẮC 4: TRẢ LỜI CÁC TRUY VẤN NÔNG NGHIỆP THÔNG THƯỜNG (KHÔNG CÓ HÌNH ẢNH, ĐÚNG CHỦ ĐỀ)**"
        "\n"
        "* **ĐIỀU KIỆN:** Không có `**Kết quả phân tích hình ảnh:**` tồn tại VÀ `{{ query }}` liên quan đến nông nghiệp."
        "* **HÀNH ĐỘNG:**"
        "     * Kiểm tra `{{ memories }}` để tìm **Bối cảnh** (bệnh/chủ đề) đã được thảo luận gần đây."
        "     * Trả lời `{{ query }}`: ưu tiên **Bối cảnh** (nếu có), nếu không thì trả lời một cách tổng quát."
        "     * Sử dụng `{{ documents }}` để tìm thông tin liên quan đến **Bối cảnh** (nếu có) hoặc liên quan trực tiếp đến `{{ query }}`."
        "     * Tránh các liên kết video."
        "\n"
        "**CÁC LƯU Ý TỐI QUAN TRỌNG:**"
        "\n"
        "1.  **TUÂN THỦ THỨ TỰ QUY TẮC: 1 -> 2 -> 3 -> 4.**"
        "2.  **LUÔN LUÔN TRẢ LỜI BẰNG TIẾNG VIỆT.** ĐIỀU NÀY LÀ BẮT BUỘC."
        "3.  **QUY TẮC 1 LÀ TUYỆT ĐỐI:** Khi được kích hoạt, nó sẽ ghi đè lên mọi thứ khác và CẤM sử dụng tài liệu."
        "4.  **QUY TẮC 3 CŨNG CẤM SỬ DỤNG TÀI LIỆU** đối với các câu hỏi ngoài chủ đề."
        "5.  Duy trì **Bối cảnh** một khi đã được xác định trong Quy tắc 2 hoặc 4 cho các lượt trao đổi tiếp theo."
    ),
    ChatMessage.from_user(
    """
{# Logic để kiểm tra QUY TẮC 1 trước tiên #}
{% set rule1_triggered = false %}
{% if identified_disease %}
    {% set temp_disease_lower = identified_disease | lower %}
    {% if 'khỏe mạnh' in temp_disease_lower or \
          'healthy' in temp_disease_lower or \
          'không có bệnh' in temp_disease_lower or \
          'bình thường' in temp_disease_lower %}
        {% set rule1_triggered = true %}
    {% endif %}
{% endif %}

{% if not rule1_triggered %} {# CHỈ hiển thị lịch sử nếu QUY TẮC 1 KHÔNG kích hoạt #}
**Lịch sử trò chuyện:**
{% for msg in memories %}
{% if msg.role == 'user' %}Người dùng: {{ (msg.to_dict()).content[0].text }}{% elif msg.role == 'assistant' %}Trợ lý: {{ (msg.to_dict()).content[0].text }}{% endif %}
{% else %}
(Không có lịch sử trò chuyện)
{% endfor %}
{% endif %} {# Kết thúc điều kiện hiển thị lịch sử #}

{% if identified_disease %}
**Kết quả phân tích hình ảnh:** {{ identified_disease }}
{% endif %}

{% if not rule1_triggered %} {# CHỈ hiển thị tài liệu nếu QUY TẮC 1 KHÔNG kích hoạt #}
**Tài liệu tham khảo:**
{% if documents %}
    {% for doc in documents %}
    ---
    {{ doc.content }}
    ---
    {% endfor %}
{% else %}
    (Không có tài liệu tham khảo)
{% endif %}
{% endif %} {# Kết thúc điều kiện hiển thị tài liệu #}

**Câu hỏi hiện tại của người dùng:** {{ query }} {# Query vẫn cần cho các rule khác #}

**Câu trả lời (Tuân thủ nghiêm ngặt TẤT CẢ các quy tắc và nhắc nhở quan trọng, CHỈ TRẢ LỜI BẰNG TIẾNG VIỆT):**"""
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
    llm_generator = OllamaGenerator(model=OLLAMA_MODEL_NAME, url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT, generation_kwargs={"num_predict": 1000, "temperature": 1, "top_p": 0.95,"top_k": 65})
    print("Initialized Main Ollama Generator.")
except Exception as e: print(f"Error initializing Main Ollama Generator: {e}"); exit()

# -- Answer Builder --
answer_builder = AnswerBuilder()
print("Initialized Answer Builder.")
print("-------------------------------------\n")

# --- 3. Xây dựng Pipeline ---
print("--- Building Pipeline (No Document Filter Component) ---")
pipeline = Pipeline()

# Thêm components
pipeline.add_component("memory_retriever", memory_retriever)
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
pipeline.connect("text_embedder.embedding", "milvus_retriever.query_embedding")
pipeline.connect("milvus_retriever.documents", "joiner.documents")
pipeline.connect("bm25_retriever.documents", "joiner.documents")
pipeline.connect("joiner.documents", "ranker.documents")
pipeline.connect("ranker.documents", "chat_prompt_builder.documents")
pipeline.connect("memory_retriever.messages", "chat_prompt_builder.memories")
pipeline.connect("chat_prompt_builder.prompt", "message_to_string_adapter.messages")
pipeline.connect("message_to_string_adapter.output", "llm.prompt")
pipeline.connect("llm.replies", "str_to_chat_converter.replies")
pipeline.connect("str_to_chat_converter.messages", "memory_joiner.values")
pipeline.connect("memory_joiner.values", "memory_writer.messages")
pipeline.connect("llm.replies", "answer_builder.replies")
pipeline.connect("ranker.documents", "answer_builder.documents")

print("Pipeline built successfully (without doc_filter).")
print("-----------------------------\n")
# --- 4. Chạy Pipeline ---
print("\n--- Starting Conversation ---")
print("Nhập 'quit' hoặc 'exit' để kết thúc.")
print("Định dạng gợi ý khi có Vision: <Câu hỏi>? ( <Tên bệnh từ Vision> )")

while True:
    user_input_raw = input("🧑 User: ")
    if user_input_raw.lower() in ["quit", "exit"]:
        break

    original_query = user_input_raw
    disease_name = None
    # Đơn giản chỉ trích xuất disease_name nếu có
    match = re.search(r'\(([^)]+)\)\s*$', user_input_raw)
    if match:
        disease_name = match.group(1).strip()

    # Pipeline Input sử dụng original_query cho RAG
    pipeline_input = {
        "text_embedder": {"text": original_query},   # Luôn dùng original_query
        "bm25_retriever": {"query": original_query}, # Luôn dùng original_query
        "ranker": {"query": original_query},         # Luôn dùng original_query
        "chat_prompt_builder": {
            "query": original_query,        # Gửi query gốc
            "identified_disease": disease_name # Gửi disease_name (có thể là None)
        },
        "memory_joiner": {"values": [ChatMessage.from_user(original_query)]},
        "answer_builder": {"query": original_query}
    }
    try:
        start_run_time = time.time()
        result = pipeline.run(pipeline_input, include_outputs_from=["answer_builder"])
        end_run_time = time.time()

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