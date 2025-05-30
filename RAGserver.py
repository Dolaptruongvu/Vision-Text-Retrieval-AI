import torch
import sys
import os
import time
import traceback
from typing import List, Any, Optional, Dict
from dotenv import load_dotenv
from flask import Flask, request, jsonify # <<< THÊM CHO API FLASK

# --- Haystack Core & Standard Components ---
from haystack import Pipeline, Document, component
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.utils import Secret, ComponentDevice
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner, ListJoiner # Giữ lại ListJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.dataclasses import ChatMessage
from haystack.components.converters import OutputAdapter # Giữ lại OutputAdapter

# --- Haystack Integrations ---
from haystack_integrations.components.generators.ollama import OllamaGenerator
from milvus_haystack import MilvusDocumentStore, MilvusEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever

# --- Haystack Experimental (Memory) ---
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
import re # Giữ lại re vì bạn có thể dùng trong vòng lặp input nếu chạy standalone

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
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_blocks")
VECTOR_FIELD_NAME = "embedding"
TEXT_FIELD_NAME_MILVUS = "content"
MILVUS_INDEX_PARAMS = {"index_type": "DISKANN", "metric_type": "COSINE", "params": {"search_list": 100}}
MILVUS_SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"search_list": 100}}
MILVUS_TOP_K = int(os.getenv("MILVUS_TOP_K", 7))
ES_HOST = os.getenv("ES_HOST", "http://127.0.0.1:9200")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "my_rag_index")
ES_BM25_TOP_K = int(os.getenv("ES_BM25_TOP_K", 7))
RANKER_MODEL_NAME = os.getenv("RANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-12-v2")
RANKER_FINAL_TOP_K = int(os.getenv("RANKER_FINAL_TOP_K", 5))
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma3:latest")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 180))

# --- Khởi tạo Flask App cho RAG Service ---
rag_service_app = Flask(__name__) # Đặt tên mới cho Flask app của RAG service

# --- Biến toàn cục cho các components và pipeline của Haystack ---
# Khai báo để có thể truy cập từ các hàm và endpoint API
text_embedder = None
milvus_retriever = None
bm25_retriever = None
joiner = None
ranker = None
memory_store = None # Đây là memory_store gốc của bạn
memory_retriever = None
memory_writer = None
memory_joiner = None # Giữ lại memory_joiner của bạn
chat_prompt_builder = None
message_to_string_adapter = None # Giữ lại adapter này
llm_generator = None
str_to_chat_converter = None # Giữ lại custom component này
answer_builder = None
haystack_pipeline = None # Đặt tên mới cho pipeline

def initialize_and_build_haystack_pipeline_once():
    global text_embedder, milvus_retriever, bm25_retriever, joiner, ranker
    global memory_store, memory_retriever, memory_writer, memory_joiner
    global chat_prompt_builder, message_to_string_adapter, llm_generator
    global str_to_chat_converter, answer_builder, haystack_pipeline

    if haystack_pipeline is not None: # Chỉ khởi tạo một lần
        print("RAG SERVICE: Haystack pipeline already initialized.")
        return

    print("--- RAG SERVICE: Determining Device ---")
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

    print("--- RAG SERVICE: Initializing Haystack Components ---")
    try:
        # -- Document Stores --
        milvus_document_store = MilvusDocumentStore(
            connection_args={"uri": MILVUS_URI}, collection_name=COLLECTION_NAME,
            vector_field=VECTOR_FIELD_NAME, text_field=TEXT_FIELD_NAME_MILVUS,
            index_params=MILVUS_INDEX_PARAMS, search_params=MILVUS_SEARCH_PARAMS
        )
        print(f"Milvus connected: {milvus_document_store.count_documents()} docs.")

        es_document_store = ElasticsearchDocumentStore(hosts=[ES_HOST], index=ES_INDEX_NAME)
        print(f"Elasticsearch connected: {es_document_store.count_documents()} docs.")

        # -- Embedder --
        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device, normalize_embeddings=True)
        text_embedder.warm_up()
        print("Embedder initialized.")

        # -- Retrievers --
        milvus_retriever = MilvusEmbeddingRetriever(document_store=milvus_document_store, top_k=MILVUS_TOP_K)
        print(f"Initialized Milvus Retriever (top_k={MILVUS_TOP_K})")
        bm25_retriever = ElasticsearchBM25Retriever(document_store=es_document_store, top_k=ES_BM25_TOP_K)
        print(f"Initialized BM25 Retriever (top_k={ES_BM25_TOP_K})")

        # -- Joiner --
        joiner = DocumentJoiner(join_mode="concatenate")
        print("Initialized Document Joiner.")

        # -- Ranker --
        ranker = TransformersSimilarityRanker(model=RANKER_MODEL_NAME, top_k=RANKER_FINAL_TOP_K, token=HF_TOKEN, device=device)
        ranker.warm_up()
        print(f"Ranker initialized (top_k={RANKER_FINAL_TOP_K}).")

        # -- Memory Components (Giữ nguyên logic memory của bạn) --
        memory_store = InMemoryChatMessageStore()
        memory_retriever = ChatMessageRetriever(memory_store) # Sẽ lấy từ memory_store này
        memory_writer = ChatMessageWriter(memory_store)     # Sẽ ghi vào memory_store này
        memory_joiner = ListJoiner(List[ChatMessage])       # Dùng để join user_msg và assistant_msg cho writer
        print("Memory components initialized.")

        # -- Chat Prompt Builder --
        # !!! QUAN TRỌNG: BẠN SẼ TỰ ĐIỀN LẠI `chat_prompt_template` ĐẦY ĐỦ CỦA MÌNH VÀO ĐÂY !!!
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
        print("Initialized ChatPromptBuilder (VỚI TEMPLATE MẪU - BẠN CẦN THAY THẾ!).")

        # -- Adapters cho LLM và Memory (Giữ nguyên) --
        message_to_string_adapter = OutputAdapter(template="{{ (messages[-1].to_dict())['content'][0]['text'] }}", output_type=str)
        str_to_chat_converter = StringListToChatMessages()
        print("Initialized Adapters.")

        # -- LLM Generator (chính) --
        llm_generator = OllamaGenerator(model=OLLAMA_MODEL_NAME, url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT, 
                                        generation_kwargs={"num_predict": 1000, "temperature": 1, "top_p": 0.95,"top_k": 65}) # Giữ temp thấp
        print("Initialized Main Ollama Generator.")

        # -- Answer Builder --
        answer_builder = AnswerBuilder()
        print("Initialized Answer Builder.")
        print("-------------------------------------\n")

        # --- 3. Xây dựng Pipeline (Giữ nguyên pipeline của bạn) ---
        print("--- RAG SERVICE: Building Pipeline ---")
        haystack_pipeline = Pipeline() # Gán vào biến global

        haystack_pipeline.add_component("memory_retriever", memory_retriever)
        haystack_pipeline.add_component("text_embedder", text_embedder)
        haystack_pipeline.add_component("milvus_retriever", milvus_retriever)
        haystack_pipeline.add_component("bm25_retriever", bm25_retriever)
        haystack_pipeline.add_component("joiner", joiner)
        haystack_pipeline.add_component("ranker", ranker)
        haystack_pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
        haystack_pipeline.add_component("message_to_string_adapter", message_to_string_adapter)
        haystack_pipeline.add_component("llm", llm_generator)
        haystack_pipeline.add_component("str_to_chat_converter", str_to_chat_converter)
        haystack_pipeline.add_component("memory_joiner", memory_joiner)
        haystack_pipeline.add_component("memory_writer", memory_writer)
        haystack_pipeline.add_component("answer_builder", answer_builder)

        # --- Kết nối Pipeline (Giữ nguyên kết nối của bạn) ---
        haystack_pipeline.connect("text_embedder.embedding", "milvus_retriever.query_embedding")
        haystack_pipeline.connect("milvus_retriever.documents", "joiner.documents")
        haystack_pipeline.connect("bm25_retriever.documents", "joiner.documents")
        haystack_pipeline.connect("joiner.documents", "ranker.documents")
        haystack_pipeline.connect("ranker.documents", "chat_prompt_builder.documents")
        haystack_pipeline.connect("memory_retriever.messages", "chat_prompt_builder.memories")
        haystack_pipeline.connect("chat_prompt_builder.prompt", "message_to_string_adapter.messages")
        haystack_pipeline.connect("message_to_string_adapter.output", "llm.prompt")
        haystack_pipeline.connect("llm.replies", "str_to_chat_converter.replies")
        haystack_pipeline.connect("str_to_chat_converter.messages", "memory_joiner.values") # Input 1 cho memory_joiner
        # Input 2 cho memory_joiner (ChatMessage.from_user) sẽ được truyền qua pipeline_input
        haystack_pipeline.connect("memory_joiner.values", "memory_writer.messages")
        haystack_pipeline.connect("llm.replies", "answer_builder.replies")
        haystack_pipeline.connect("ranker.documents", "answer_builder.documents")

        print("RAG SERVICE: Haystack Pipeline built successfully.")
        print("-----------------------------\n")

    except Exception as e:
        print(f"FATAL ERROR initializing RAG SERVICE components: {e}")
        traceback.print_exc()
        sys.exit(1)


@rag_service_app.route('/get_rag_response', methods=['POST'])
def handle_rag_request():
    global haystack_pipeline, memory_store, memory_writer, memory_joiner, str_to_chat_converter # Cần truy cập để ghi memory

    if not haystack_pipeline:
        return jsonify({"error": "Haystack RAG Pipeline not initialized. Please wait or check logs."}), 500

    data = request.json
    user_query = data.get('query') # Đây là user_text_query từ client, có thể đã được client điều chỉnh
    identified_disease = data.get('identified_disease') # Tên bệnh đã làm sạch từ client
    # client_chat_history = data.get('memories', []) # Nhận lịch sử từ client nếu client gửi

    if not user_query:
        return jsonify({"error": "Parameter 'query' is required"}), 400

    print(f"\nRAG SERVICE Received: query='{user_query}', identified_disease='{identified_disease}'")

    # Tạo ChatMessage cho input người dùng để ghi vào memory của RAG service
    user_chat_message = ChatMessage.from_user(user_query) # Chỉ query gốc, vì prompt template sẽ tự xử lý identified_disease

    # Input cho pipeline RAG
    # memory_joiner sẽ nhận input này cho phần user message
    pipeline_input = {
        "text_embedder": {"text": user_query},
        "bm25_retriever": {"query": user_query},
        "ranker": {"query": user_query},
        "chat_prompt_builder": {
            "query": user_query,
            "identified_disease": identified_disease,
            # "memories": client_chat_history # Nếu bạn muốn truyền memories từ client vào prompt builder
        },
        "memory_joiner": {"values": [user_chat_message]}, # Truyền user message cho memory_joiner
        "answer_builder": {"query": user_query}
    }
    
    # Lưu ý: memory_retriever sẽ tự động lấy lịch sử từ memory_store của RAG service này.
    # Nếu bạn muốn mỗi client có session memory riêng biệt và RAG service này stateless,
    # thì client phải gửi toàn bộ 'memories' và bạn phải truyền nó vào 'chat_prompt_builder'.
    # Đồng thời, bỏ 'memory_retriever', 'memory_joiner', 'memory_writer' khỏi pipeline
    # và không ghi gì vào memory_store của RAG service.
    # Hiện tại, code đang giữ lại memory của Haystack RAG service.

    try:
        start_run_time = time.time()
        # Chạy pipeline
        result = haystack_pipeline.run(data=pipeline_input, include_outputs_from=["answer_builder", "llm", "str_to_chat_converter"])
        end_run_time = time.time()
        print(f"RAG SERVICE: Pipeline run time: {end_run_time - start_run_time:.2f}s")

        assistant_response_text = "Xin lỗi, tôi không thể tạo câu trả lời."

        if "answer_builder" in result and result["answer_builder"]["answers"]:
            final_answer_obj = result["answer_builder"]["answers"][0]
            if final_answer_obj.data:
                assistant_response_text = final_answer_obj.data
        elif "llm" in result and result["llm"]["replies"]: # Fallback nếu answer_builder trống
            assistant_response_text = result["llm"]["replies"][0]
        
        # Ghi câu trả lời của assistant vào memory của RAG service
        # str_to_chat_converter đã được gọi trong pipeline nếu llm.replies kết nối tới nó.
        # Output của str_to_chat_converter (là List[ChatMessage]) đã được kết nối tới memory_joiner.
        # memory_joiner cũng đã nhận user_chat_message từ pipeline_input.
        # Vậy memory_writer (đã được kết nối từ memory_joiner) sẽ ghi cả hai.
        # print(f"RAG SERVICE: Current messages in RAG's memory_store: {memory_store.count_messages()}")
            
        return jsonify({"answer": assistant_response_text})

    except Exception as e:
        print(f"\n--- RAG SERVICE: An error occurred during pipeline execution ---")
        print(e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_and_build_haystack_pipeline_once() # Khởi tạo pipeline một lần
    if haystack_pipeline is None:
        print("CRITICAL: Haystack RAG Pipeline không thể khởi tạo. API Service sẽ thoát.")
        exit()
    # Chạy Flask app cho RAG Service trên port 5002
    rag_service_app.run(debug=False, host='0.0.0.0', port=5002)