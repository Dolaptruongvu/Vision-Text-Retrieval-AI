import json
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
from tqdm import tqdm # Thư viện để hiển thị thanh tiến trình
import traceback # Vẫn giữ lại để bắt lỗi nghiêm trọng

# --- Cấu hình Elasticsearch ---
ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "my_rag_index" # Đặt tên index bạn muốn tạo
EXPECTED_EMBEDDING_DIM = 384 # Kích thước embedding của bạn

# Định nghĩa cấu trúc (mapping) cho index
INDEX_MAPPING = {
    "properties": {
        "content": {"type": "text"},
        "tags": {"type": "keyword"},
        "embedding": {
            "type": "dense_vector",
            "dims": EXPECTED_EMBEDDING_DIM,
            "index": "true",
            "similarity": "cosine"
        }
    }
}

# --- Đường dẫn file dữ liệu ---
JSONL_FILE_PATH = "../Crawl4ai/encodedData/encoded_blocks4.jsonl"

# --- Hàm tạo actions cho Bulk API (đã bỏ print debug) ---
def generate_actions(filepath, index_name):
    """
    Đọc file JSONL và tạo ra các actions cho Elasticsearch Bulk API.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc_data = json.loads(line.strip())
                doc_id = doc_data.get("id")

                source_doc = {
                    "content": doc_data.get("content"),
                    "tags": doc_data.get("tags"),
                    "embedding": doc_data.get("embedding")
                }

                if not source_doc.get("embedding") or len(source_doc["embedding"]) != EXPECTED_EMBEDDING_DIM:
                     continue # Bỏ qua nếu embedding lỗi hoặc thiếu

                source_doc = {k: v for k, v in source_doc.items() if v is not None}

                action = {
                    "_index": index_name,
                    "_source": source_doc
                }
                if doc_id:
                    action["_id"] = doc_id

                yield action

            except (json.JSONDecodeError, KeyError, Exception):
                 # Bỏ qua dòng lỗi một cách lặng lẽ trong quá trình tạo action
                 # Bạn có thể thêm logging vào file nếu muốn theo dõi lỗi này
                 continue

# --- Kết nối Elasticsearch ---
es_client = None
print(f"Connecting to Elasticsearch at {ES_HOST}...")
try:
    es_client = Elasticsearch(
        hosts=[ES_HOST],
        verify_certs=False,
        ssl_show_warn=False,
        request_timeout=60
    )
    if not es_client.ping():
        raise ValueError("Connection failed (ping unsuccessful)")
    print("Connected to Elasticsearch successfully.")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")
    exit()

# --- Tạo Index nếu chưa tồn tại ---
print(f"Checking if index '{INDEX_NAME}' exists...")
try:
    if not es_client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' not found. Creating index with mapping...")
        es_client.indices.create(index=INDEX_NAME, mappings=INDEX_MAPPING)
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
except Exception as e:
    print(f"Error checking or creating index '{INDEX_NAME}': {e}")
    exit()

# --- Thực hiện Bulk Indexing ---
print(f"\nStarting bulk indexing from '{JSONL_FILE_PATH}' into '{INDEX_NAME}'...")
start_time = time.time()
success_count = 0
failed_count = 0
error_details = [] # Lưu trữ chi tiết lỗi nếu có

try:
    try:
         num_lines = sum(1 for line in open(JSONL_FILE_PATH, 'r', encoding='utf-8'))
         print(f"Total documents to process: {num_lines}")
         has_lines = num_lines > 0
    except FileNotFoundError:
         print(f"Error: File not found at '{JSONL_FILE_PATH}'")
         exit()

    if not has_lines:
        print("Input file is empty. No documents to index.")
    else:
        progress_bar = tqdm(total=num_lines, unit="docs", desc="Indexing")
        # SỬ DỤNG streaming_bulk Ở ĐÂY:
        for ok, result in streaming_bulk( # <--- THAY ĐỔI Ở ĐÂY
            client=es_client,
            actions=generate_actions(JSONL_FILE_PATH, INDEX_NAME),
            chunk_size=500,
            request_timeout=120,
            raise_on_error=False,
            raise_on_exception=False
        ):
            if ok:
                success_count += 1
            else:
                failed_count += 1
                # Lấy thông tin lỗi từ kết quả trả về (result bây giờ là dict thông tin)
                action_type = list(result.keys())[0] # Thường là 'index', 'create', 'delete', 'update'
                error_info = result.get(action_type, {}).get('error', {})
                error_type = error_info.get('type', 'Unknown error')
                error_reason = error_info.get('reason', 'Unknown reason')
                doc_id = result.get(action_type, {}).get('_id', 'N/A')
                error_details.append(f"Doc ID: {doc_id}, Type: {error_type}, Reason: {error_reason}")
                # Có thể bỏ phần print lỗi chi tiết ở đây nếu muốn gọn
                # if failed_count <= 10:
                #      print(f"\nFailed indexing doc_id: {doc_id}. Error: {error_type} - {error_reason}")
                # elif failed_count == 11:
                #      print("\n(Stopping detailed error logging for failures)")

            progress_bar.update(1)
            progress_bar.set_postfix({"Success": success_count, "Failed": failed_count})

        progress_bar.close()

except Exception as e:
    print(f"\nAn critical error occurred during bulk indexing process: {e}")
    traceback.print_exc()

# --- In Tổng kết ---
end_time = time.time()
print("\n--- Indexing Summary ---")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print(f"Successfully indexed documents: {success_count}")
print(f"Failed documents: {failed_count}")

# In chi tiết các lỗi đã lưu nếu có (tùy chọn)
if error_details and failed_count > 0:
    print("\n--- Failure Details (Sample) ---")
    for i, err in enumerate(error_details):
        if i < 20: # Chỉ in tối đa 20 lỗi
             print(err)
        elif i == 20:
             print(f"(... and {failed_count - 20} more failures)")
             break

print("\nImport script finished.")