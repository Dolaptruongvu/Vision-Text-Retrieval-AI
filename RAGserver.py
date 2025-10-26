# LLM Service - RAG Pipeline with Flask API
# Port: 5002
# Based on RAG.py with improved source citation

import torch
import sys
import os
import time
import traceback
from typing import List, Any, Optional, Dict
from dotenv import load_dotenv
from flask import Flask, request, jsonify

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

# ============================================================================
# CUSTOM COMPONENTS
# ============================================================================
@component
class StringListToChatMessages:
    @component.output_types(messages=List[ChatMessage])
    def run(self, replies: List[str]):
        return {"messages": [ChatMessage.from_assistant(reply) for reply in replies]}

@component
class MarkdownURLExtractor:
    """
    Extract URLs from markdown links in document content.
    Parses [text](url) format and stores URLs in document metadata.
    """
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        import re
        
        for doc in documents:
            # Extract URLs from markdown links: [text](url)
            url_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            matches = re.findall(url_pattern, doc.content)
            extracted_urls = [url for text, url in matches]
            
            # Initialize metadata if not exists
            if not doc.meta:
                doc.meta = {}
            
            # Store extracted URLs in metadata
            doc.meta['extracted_urls'] = extracted_urls
            doc.meta['url_count'] = len(extracted_urls)
            
            # Extract first URL as primary source (for backward compatibility)
            if extracted_urls:
                doc.meta['url'] = extracted_urls[0]
        
        return {"documents": documents}

# ============================================================================
# CONFIGURATION (Same as RAG.py)
# ============================================================================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
HF_TOKEN = Secret.from_token(hf_token) if hf_token else None
EXPECTED_EMBEDDING_DIM = 896
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_collection_KaLM_embedding_multilingual_mini_v1"
VECTOR_FIELD_NAME = "embedding"
TEXT_FIELD_NAME_MILVUS = "content"
MILVUS_INDEX_PARAMS = {"index_type": "DISKANN", "metric_type": "COSINE", "params": {"search_list": 100}}
MILVUS_SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"search_list": 100}}
MILVUS_TOP_K = 15
ES_HOST = "http://127.0.0.1:9200"
ES_INDEX_NAME = "my_rag_index_final2"
ES_BM25_TOP_K = 15
RANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RANKER_FINAL_TOP_K = 8
OLLAMA_MODEL_NAME = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 180

# ============================================================================
# FLASK APP
# ============================================================================
rag_service_app = Flask(__name__)

# ============================================================================
# GLOBAL COMPONENTS
# ============================================================================
text_embedder = None
milvus_retriever = None
bm25_retriever = None
joiner = None
ranker = None
memory_store = None
memory_retriever = None
memory_writer = None
memory_joiner = None
url_extractor = None
chat_prompt_builder = None
message_to_string_adapter = None
llm_generator = None
str_to_chat_converter = None
answer_builder = None
pipeline = None

# ============================================================================
# INITIALIZATION FUNCTION
# ============================================================================
def initialize_rag_pipeline():
    """Initialize all components and build RAG pipeline (same as RAG.py)"""
    global text_embedder, milvus_retriever, bm25_retriever, joiner, ranker
    global memory_store, memory_retriever, memory_writer, memory_joiner
    global url_extractor, chat_prompt_builder, message_to_string_adapter, llm_generator
    global str_to_chat_converter, answer_builder, pipeline
    
    try:
        print("=" * 60)
        print("LLM SERVICE - INITIALIZING RAG PIPELINE")
        print("=" * 60)
        
        # --- Determine Device ---
        print("\n--- Determining Device ---")
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

        # --- Initialize Components ---
        print("--- Initializing Haystack Components ---")

        # Document Stores
        try:
            milvus_document_store = MilvusDocumentStore(
                connection_args={"uri": MILVUS_URI}, 
                collection_name=COLLECTION_NAME,
                vector_field=VECTOR_FIELD_NAME, 
                text_field=TEXT_FIELD_NAME_MILVUS,
                index_params=MILVUS_INDEX_PARAMS, 
                search_params=MILVUS_SEARCH_PARAMS
            )
            print(f"Milvus connected: {milvus_document_store.count_documents()} docs.")
        except Exception as e:
            print(f"Error connecting to Milvus: {e}")
            return False

        try:
            es_document_store = ElasticsearchDocumentStore(hosts=[ES_HOST], index=ES_INDEX_NAME)
            print(f"Elasticsearch connected: {es_document_store.count_documents()} docs.")
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {e}")
            return False

        # Embedder
        try:
            text_embedder = SentenceTransformersTextEmbedder(
                model="HIT-TMG/KaLM-embedding-multilingual-mini-v1", 
                device=device, 
                normalize_embeddings=True
            )
            text_embedder.warm_up()
            print("Embedder initialized.")
        except Exception as e:
            print(f"Error initializing Embedder: {e}")
            return False

        # Retrievers
        try:
            milvus_retriever = MilvusEmbeddingRetriever(document_store=milvus_document_store, top_k=MILVUS_TOP_K)
            print(f"Initialized Milvus Retriever (top_k={MILVUS_TOP_K})")
        except Exception as e:
            print(f"Error initializing Milvus Retriever: {e}")
            return False

        try:
            bm25_retriever = ElasticsearchBM25Retriever(document_store=es_document_store, top_k=ES_BM25_TOP_K)
            print(f"Initialized BM25 Retriever (top_k={ES_BM25_TOP_K})")
        except Exception as e:
            print(f"Error initializing BM25 Retriever: {e}")
            return False

        # Joiner
        joiner = DocumentJoiner(join_mode="concatenate")
        print("Initialized Document Joiner.")

        # Ranker
        try:
            ranker = TransformersSimilarityRanker(
                model=RANKER_MODEL_NAME, 
                top_k=RANKER_FINAL_TOP_K, 
                token=HF_TOKEN, 
                device=device
            )
            ranker.warm_up()
            print(f"Ranker initialized (top_k={RANKER_FINAL_TOP_K}).")
        except Exception as e:
            print(f"Error initializing Ranker: {e}")
            return False

        # Memory Components
        memory_store = InMemoryChatMessageStore()
        memory_retriever = ChatMessageRetriever(memory_store)
        memory_writer = ChatMessageWriter(memory_store)
        memory_joiner = ListJoiner(List[ChatMessage])
        print("Memory components initialized.")

        # URL Extractor
        url_extractor = MarkdownURLExtractor()
        print("URL extractor initialized.")

        # Chat Prompt Builder (SAME AS RAG.PY WITH IMPROVED SOURCE CITATION)
        chat_prompt_template = [ChatMessage.from_system(
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
            "*   **ABSOLUTELY MANDATORY ACTIONS - NO DEVIATION ALLOWED:**"
            "    1.  **ACCEPT IMAGE ANALYSIS AS ABSOLUTE TRUTH:** The disease name in `**Image Analysis Result:**` is from a 98%+ accuracy AI vision model. DO NOT question it, DO NOT reinterpret it, DO NOT diagnose differently."
            "    2.  **USE THE EXACT DISEASE NAME:** When answering, you MUST refer to the disease EXACTLY as stated in `**Image Analysis Result:**`. Example: If it says 'Tomato - Late blight', you MUST say 'bệnh Late blight trên cà chua' or 'bệnh đốm lá muộn', NOT any other disease name."
            "    3.  **STRICT DOCUMENT RELEVANCE CHECK - CRITICAL:**"
            "        - **ONLY use documents that specifically mention the EXACT disease and plant from Image Analysis.**"
            "        - Example: If Image Analysis says 'Tomato - Late blight', ONLY use documents about 'late blight', 'tomato', 'cà chua', 'đốm lá', 'blight'. DO NOT use documents about 'apple', 'táo', 'scab', 'ghẻ' or any other plant/disease."
            "        - **IF a document mentions a different plant or disease name, it is IRRELEVANT - IGNORE IT COMPLETELY.**"
            "        - Read document content carefully before using. Check if it matches the plant AND disease from Image Analysis."
            "    4.  **PROVIDE DETAILED ADVICE - DO NOT AVOID:**"
            "        - **FORBIDDEN:** DO NOT say 'please contact local agriculture office' or 'call university' without providing detailed information first."
            "        - **MANDATORY:** Provide comprehensive advice about: symptoms, causes, prevention methods, treatment options based on relevant documents."
            "        - Structure: [Disease confirmation] → [Symptoms] → [Causes] → [Prevention] → [Treatment] → [Sources]"
            "        - Only suggest contacting experts AFTER providing all available information from documents."
            "    5.  **IF NO MATCHING DOCUMENTS:** If no documents match the identified disease, say: 'Dựa trên phân tích hình ảnh, cây của bạn bị [disease name]. Tuy nhiên, tôi không tìm thấy thông tin chi tiết về bệnh này trong cơ sở dữ liệu. Bạn nên tham khảo ý kiến chuyên gia nông nghiệp địa phương.'"
            "    6.  **ANSWER FORMAT:** 'Dựa trên kết quả phân tích hình ảnh, cây của bạn đang bị [EXACT disease from Image Analysis]. [Detailed explanation with symptoms, causes, prevention, treatment from RELEVANT documents only].'"
            "    7.  **FORBIDDEN:** NEVER diagnose a different disease than what Image Analysis detected. NEVER cite documents about different plants/diseases. This is HALLUCINATION and UNACCEPTABLE."
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
            "    *   Provide detailed, comprehensive answers with practical advice."
            "    *   Avoid video links."
            "\n"
            "--- UNIVERSAL RULE THAT APPLIES TO ALL RESPONSES ---"
            "\n"
            "**RULE 5: MANDATORY SOURCE CITATION (ANTI-HALLUCINATION) ⚠️ CRITICAL**"
            "\n"
            "*   **CONDITION:** When using `{{ documents }}` in Rules 2 or 4 (NOT applicable to Rules 1 or 3)."
            "*   **ABSOLUTELY MANDATORY ACTION - YOU MUST DO THIS:**"
            "    1.  **CRITICAL: ONLY use documents that have '📚 Source URLs' section.** If a document has NO URLs, IGNORE it completely - do NOT use its content."
            "    2.  **NEVER cite Document IDs** - they are useless to users. Only cite actual clickable URLs."
            "    3.  At the END of your response, add a clear section: '\\n\\n---\\n**📚 Nguồn tham khảo:**\\n'"
            "    4.  For EACH document you used, copy its URLs EXACTLY as shown in '📚 Source URLs' section:"
            "        - Format: '- 🔗 [Complete URL exactly as shown]'"
            "        - Example: '- 🔗 https://globalcheck.com.vn/may-bay-phun-thuoc-nong-nghiep-cong-nghe-phun-ly-tam'"
            "    5.  **DO NOT INVENT URLs.** Only copy URLs that appear in the document metadata."
            "    6.  If NO documents have URLs, say: 'Xin lỗi, tôi không tìm thấy nguồn tham khảo cụ thể cho thông tin này.'"
            "    7.  End with: '\\n*Nguồn được trích xuất từ cơ sở dữ liệu nông nghiệp Việt Nam (Milvus + Elasticsearch)*'"
            "    8.  **FAILURE TO CITE REAL URLs = HALLUCINATION. This is NON-NEGOTIABLE.**"
            "\n"
            "**🚨 MOST CRITICAL REMINDERS:**"
            "\n"
            "1.  **ADHERE TO RULE ORDER: 1 -> 2 -> 3 -> 4, with Rule 5 applied when documents are used.**"
            "2.  **ALWAYS RESPOND IN VIETNAMESE.** THIS IS MANDATORY."
            "3.  **RULE 1 IS ABSOLUTE:** When triggered, it overrides everything else and FORBIDS document usage."
            "4.  **RULE 2 IS NON-NEGOTIABLE:** Image Analysis Result is from 98%+ accuracy AI vision model."
            "    - **YOU ARE FORBIDDEN FROM DIAGNOSING A DIFFERENT DISEASE** than what Image Analysis detected."
            "    - If Image Analysis says 'Late blight', you CANNOT say 'powdery mildew' or any other disease."
            "    - **CRITICAL: DOCUMENT RELEVANCE CHECK:**"
            "      * If Image Analysis: 'Tomato - Late blight' → Use docs about: tomato/cà chua + late blight/đốm lá"
            "      * If Image Analysis: 'Tomato - Late blight' → DO NOT use docs about: apple/táo, scab/ghẻ, pepper/ớt, etc."
            "      * **WRONG EXAMPLE:** Image='Tomato Late blight' but citing doc about 'apple scab' = HALLUCINATION = FORBIDDEN"
            "      * **RIGHT EXAMPLE:** Image='Tomato Late blight' citing doc about 'tomato blight symptoms' = CORRECT"
            "    - Always start response with: 'Dựa trên kết quả phân tích hình ảnh, cây của bạn đang bị [EXACT disease name]'"
            "    - Provide detailed advice (symptoms, treatment, prevention) - DO NOT just say 'contact someone' without details."
            "    - Use ONLY documents that match BOTH the plant AND disease from Image Analysis."
            "5.  **RULE 3 ALSO FORBIDS DOCUMENT USAGE** for off-topic questions."
            "6.  **RULE 5 IS ABSOLUTELY MANDATORY - NO EXCEPTIONS:**"
            "    - ONLY use documents that have URLs in '📚 Source URLs' section"
            "    - NEVER cite Document IDs - only cite real clickable URLs"
            "    - Copy URLs EXACTLY as shown - do NOT invent or modify them"
            "    - If no documents have URLs, explicitly state you cannot find sources"
            "    - NO REAL URLs = DO NOT CITE ANYTHING (no Document IDs, no made-up URLs)"
            "7.  Maintain **Context** once identified in Rule 2 or 4 for follow-up turns."
            "8.  **VERIFY EVERY FACT AGAINST DOCUMENTS. If unsure, cite the source.**"
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
            {% if doc.meta and doc.meta.extracted_urls and doc.meta.extracted_urls|length > 0 %}
            ---
            **✅ Document {{ loop.index }} (HAS SOURCES - USE THIS):**
            **Content:** {{ doc.content[:500] }}...
            **📚 Source URLs (CITE THESE EXACTLY):**
            {% for url in doc.meta.extracted_urls %}
            - 🔗 {{ url }}
            {% endfor %}
            ---
            {% else %}
            ---
            **❌ Document {{ loop.index }} (NO SOURCES - DO NOT USE):**
            **Content:** {{ doc.content[:300] }}...
            **⚠️ WARNING:** This document has NO verifiable URLs. Do NOT cite this document.
            ---
            {% endif %}
            {% endfor %}
        {% else %}
            (No reference documents)
        {% endif %}

        **User's Current Question:** {{ query }}

        **CORRECT ANSWER FORMAT EXAMPLE (When Image Analysis exists):**
        "Dựa trên kết quả phân tích hình ảnh, cây của bạn đang bị bệnh [disease name from Image Analysis].
        
        **Triệu chứng đặc trưng:**
        - [List symptoms from relevant documents]
        
        **Nguyên nhân:**
        - [Explain causes from relevant documents]
        
        **Cách phòng ngừa:**
        - [Prevention methods from relevant documents]
        
        **Cách xử lý:**
        - [Treatment methods from relevant documents]
        
        ---
        **📚 Nguồn tham khảo:**
        - 🔗 [URL 1 from document about THIS disease]
        - 🔗 [URL 2 from document about THIS disease]"

        **Answer (Strictly follow ALL rules and critical reminders, RESPOND ONLY IN VIETNAMESE, MUST CITE SOURCES):**"""
        )]
        
        chat_prompt_builder = ChatPromptBuilder(template=chat_prompt_template)
        print("Initialized ChatPromptBuilder with improved source citation.")

        # Adapters
        message_to_string_adapter = OutputAdapter(
            template="{{ (messages[-1].to_dict())['content'][0]['text'] }}", 
            output_type=str
        )
        str_to_chat_converter = StringListToChatMessages()
        print("Initialized Adapters.")

        # LLM Generator
        try:
            llm_generator = OllamaGenerator(
                model=OLLAMA_MODEL_NAME, 
                url=OLLAMA_URL, 
                timeout=OLLAMA_TIMEOUT,
                generation_kwargs={"temperature": 0.1}
            )
            print("Initialized Main Ollama Generator.")
        except Exception as e:
            print(f"Error initializing Main Ollama Generator: {e}")
            return False

        # Answer Builder
        answer_builder = AnswerBuilder()
        print("Initialized Answer Builder.")
        print("-------------------------------------\n")

        # --- Build Pipeline (SAME AS RAG.PY) ---
        print("--- Building Pipeline ---")
        pipeline = Pipeline()

        # Add components
        pipeline.add_component("memory_retriever", memory_retriever)
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("milvus_retriever", milvus_retriever)
        pipeline.add_component("bm25_retriever", bm25_retriever)
        pipeline.add_component("joiner", joiner)
        pipeline.add_component("ranker", ranker)
        pipeline.add_component("url_extractor", url_extractor)
        pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
        pipeline.add_component("message_to_string_adapter", message_to_string_adapter)
        pipeline.add_component("llm", llm_generator)
        pipeline.add_component("str_to_chat_converter", str_to_chat_converter)
        pipeline.add_component("memory_joiner", memory_joiner)
        pipeline.add_component("memory_writer", memory_writer)
        pipeline.add_component("answer_builder", answer_builder)

        # Connect Pipeline (WITH URL EXTRACTOR)
        pipeline.connect("text_embedder.embedding", "milvus_retriever.query_embedding")
        pipeline.connect("milvus_retriever.documents", "joiner.documents")
        pipeline.connect("bm25_retriever.documents", "joiner.documents")
        pipeline.connect("joiner.documents", "ranker.documents")
        pipeline.connect("ranker.documents", "url_extractor.documents")
        pipeline.connect("url_extractor.documents", "chat_prompt_builder.documents")
        pipeline.connect("memory_retriever.messages", "chat_prompt_builder.memories")
        pipeline.connect("chat_prompt_builder.prompt", "message_to_string_adapter.messages")
        pipeline.connect("message_to_string_adapter.output", "llm.prompt")
        pipeline.connect("llm.replies", "str_to_chat_converter.replies")
        pipeline.connect("str_to_chat_converter.messages", "memory_joiner.values")
        pipeline.connect("memory_joiner.values", "memory_writer.messages")
        pipeline.connect("llm.replies", "answer_builder.replies")
        pipeline.connect("url_extractor.documents", "answer_builder.documents")

        print("Pipeline built successfully.")
        print("-----------------------------\n")
        return True

    except Exception as e:
        print(f"FATAL ERROR initializing pipeline: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# API ENDPOINTS
# ============================================================================
@rag_service_app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'llm',
        'pipeline_loaded': pipeline is not None,
        'model': OLLAMA_MODEL_NAME,
        'embedding_model': 'HIT-TMG/KaLM-embedding-multilingual-mini-v1',
        'milvus_collection': COLLECTION_NAME,
        'es_index': ES_INDEX_NAME
    })

@rag_service_app.route('/get_rag_response', methods=['POST'])
def handle_rag_request():
    """
    LLM Service API Endpoint
    Receives request from Vision Service, returns AI response
    
    Input (from Vision Service):
        {
            "query": "user question text",
            "identified_disease": "disease name from vision (optional)"
        }
    
    Output (to Vision Service):
        {
            "success": true/false,
            "answer": "AI response with source citations",
            "error": "error message (if failed)"
        }
    """
    global pipeline, memory_joiner
    
    if not pipeline:
        return jsonify({
            'success': False,
            'error': 'Pipeline not initialized'
        }), 500
    
    try:
        # Parse input from Vision Service
        data = request.json
        user_query = data.get('query', '').strip()
        identified_disease = data.get('identified_disease')
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
        
        print(f"[LLM SERVICE] Received query: '{user_query}'")
        if identified_disease:
            print(f"[LLM SERVICE] Disease context: '{identified_disease}'")
        
        # ============================================================================
        # ENHANCED SEARCH QUERY - Use disease name for better retrieval
        # ============================================================================
        search_query = user_query
        
        if identified_disease:
            # Extract plant and disease type from "Plant - Disease" format
            # Example: "Strawberry - Leaf scorch" -> "Strawberry Leaf scorch"
            enhanced_query = identified_disease.replace(' - ', ' ')
            
            # If user asks generic questions, use disease name for search
            generic_questions = ['đây là bệnh gì', 'bệnh này', 'cây này', 'what disease', 'what is this']
            if any(gq in user_query.lower() for gq in generic_questions):
                search_query = enhanced_query
                print(f"[LLM SERVICE] Enhanced search query: '{search_query}' (generic question detected)")
            else:
                # Combine user query with disease name for better context
                search_query = f"{enhanced_query} {user_query}"
                print(f"[LLM SERVICE] Enhanced search query: '{search_query}' (combined)")
        
        # ============================================================================
        # SMART QUERY DETECTION - Skip RAG for simple greetings/small talk
        # ============================================================================
        query_lower = user_query.lower().strip()
        
        # Simple greetings (1-3 words)
        simple_greetings = [
            'hi', 'hello', 'chào', 'xin chào', 'hey', 'yo',
            'chào bạn', 'xin chao', 'hello there', 'hi there',
            'good morning', 'good afternoon', 'good evening',
            'chào buổi sáng', 'chào buổi chiều', 'chào buổi tối'
        ]
        
        # Check if query is just a greeting (no disease context)
        if not identified_disease and query_lower in simple_greetings:
            print(f"[LLM SERVICE] Detected simple greeting - Quick response (no RAG)")
            quick_response = """Xin chào! 👋 Tôi là trợ lý AI chuyên về bệnh cây trồng.

Tôi có thể giúp bạn:
• 🔍 Chẩn đoán bệnh từ ảnh cây trồng
• 💊 Tư vấn phương pháp phòng trừ bệnh
• 🌱 Giải đáp thắc mắc về triệu chứng bệnh
• 📚 Cung cấp thông tin về các loại bệnh phổ biến

Hãy upload ảnh cây trồng hoặc đặt câu hỏi để bắt đầu!"""
            
            return jsonify({
                'success': True,
                'answer': quick_response,
                'quick_reply': True
            })
        
        # Check if query is too short and has no disease context (likely not a real question)
        word_count = len(user_query.split())
        if not identified_disease and word_count <= 2:
            print(f"[LLM SERVICE] Query too short ({word_count} words) - Quick response")
            return jsonify({
                'success': True,
                'answer': 'Bạn có thể mô tả chi tiết hơn về vấn đề của cây trồng không? Hoặc upload ảnh để tôi có thể hỗ trợ tốt hơn.',
                'quick_reply': True
            })
        
        # ============================================================================
        # FULL RAG PIPELINE for real questions
        # ============================================================================
        
        # Create user message for memory
        user_message = ChatMessage.from_user(user_query)
        
        # Prepare pipeline input (Use search_query for retrieval, user_query for prompt)
        pipeline_input = {
            "text_embedder": {"text": search_query},  # Use enhanced query for better retrieval
            "bm25_retriever": {"query": search_query},  # Use enhanced query for better retrieval
            "ranker": {"query": search_query},  # Use enhanced query for better ranking
            "chat_prompt_builder": {
                "query": user_query,  # Keep original user query in prompt
                "identified_disease": identified_disease
            },
            "memory_joiner": {"values": [user_message]},
            "answer_builder": {"query": user_query}  # Keep original user query
        }
        
        # Run pipeline
        print(f"[LLM SERVICE] Running RAG pipeline...")
        start_time = time.time()
        result = pipeline.run(pipeline_input, include_outputs_from=["answer_builder"])
        end_time = time.time()
        
        print(f"[LLM SERVICE] Pipeline completed in {end_time - start_time:.2f}s")
        
        # Extract answer (SAME AS RAG.PY)
        assistant_response_text = None
        
        if "answer_builder" in result and result["answer_builder"]["answers"]:
            final_answer = result["answer_builder"]["answers"][0]
            assistant_response_text = final_answer.data
        
        if assistant_response_text:
            print(f"[LLM SERVICE] Response generated (length: {len(assistant_response_text)} chars)")
            
            # Return to Vision Service
            return jsonify({
                'success': True,
                'answer': assistant_response_text
            })
        else:
            print(f"[LLM SERVICE] No answer generated")
            return jsonify({
                'success': False,
                'error': 'No answer generated'
            }), 500
            
    except Exception as e:
        print(f"[LLM SERVICE] Error: {e}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("LLM SERVICE - RAG PIPELINE WITH FLASK API")
    print("=" * 60)
    
    # Initialize pipeline
    if not initialize_rag_pipeline():
        print("\n[CRITICAL] Failed to initialize RAG pipeline. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("LLM SERVICE - READY TO SERVE")
    print("=" * 60)
    print(f"Starting Flask server on port 5002...")
    print(f"API Endpoints:")
    print(f"  - GET  /health           : Health check")
    print(f"  - POST /get_rag_response : Get RAG response (called by Vision Service)")
    print("=" * 60)
    print(f"\nFlow: Client → Vision Service (5003) → LLM Service (5002) → Response")
    print("=" * 60 + "\n")
    
    # Run Flask app
    rag_service_app.run(debug=False, host='0.0.0.0', port=5002)
