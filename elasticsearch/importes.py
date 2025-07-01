import json
import time
import glob
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm
import traceback

# --- Cấu hình ---
ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "my_rag_index_final" # Đặt tên index bạn muốn
EXPECTED_EMBEDDING_DIM = 384

# --- CẤU HÌNH QUAN TRỌNG ---
# Đường dẫn đến thư mục chứa dữ liệu
DATA_DIRECTORY = "../Crawl4ai/encodedData/"
# Mẫu để tìm các file dữ liệu
FILE_PATTERN = "encoded_blocks*.jsonl"
# Đặt thành True nếu bạn muốn xóa sạch index cũ trước khi import
# Điều này đảm bảo không có dữ liệu trùng lặp từ các lần chạy trước
RECREATE_INDEX = True

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

def generate_actions_from_files(file_paths, index_name):
    """
    Đọc nhiều file JSONL, xác thực từng dòng và tạo ra các actions cho Bulk API.
    """
    skipped_count = 0
    
    for filepath in file_paths:
        print(f"\n[INFO] Processing file: {os.path.basename(filepath)}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_num = i + 1
                try:
                    doc_data = json.loads(line.strip())
                    doc_id = doc_data.get("id")
                    embedding = doc_data.get("embedding")

                    # Xác thực dữ liệu
                    if not embedding or len(embedding) != EXPECTED_EMBEDDING_DIM:
                        # Bỏ qua lặng lẽ để không làm rối log, bạn có thể thêm print nếu cần
                        skipped_count += 1
                        continue
                    
                    source_doc = {
                        "content": doc_data.get("content"),
                        "tags": doc_data.get("tags"),
                        "embedding": embedding
                    }
                    source_doc = {k: v for k, v in source_doc.items() if v is not None}
                    
                    action = {
                        "_index": index_name,
                        "_source": source_doc
                    }
                    if doc_id:
                        action["_id"] = doc_id
                    
                    yield action

                except (json.JSONDecodeError, KeyError, TypeError):
                    skipped_count += 1
                    continue
    
    if skipped_count > 0:
        print(f"\n[INFO] Total lines skipped due to validation errors: {skipped_count}")

# --- Bắt đầu Script ---

# 1. Tìm tất cả các file dữ liệu
search_pattern = os.path.join(DATA_DIRECTORY, FILE_PATTERN)
file_list = sorted(glob.glob(search_pattern)) # sorted để đảm bảo thứ tự xử lý

if not file_list:
    print(f"Error: No files found matching pattern '{search_pattern}'")
    exit()

print("--- Found data files to process ---")
for f in file_list:
    print(f"- {os.path.basename(f)}")
print("-" * 35)


# 2. Kết nối Elasticsearch
es_client = None
print(f"\nConnecting to Elasticsearch at {ES_HOST}...")
try:
    es_client = Elasticsearch(hosts=[ES_HOST], request_timeout=60)
    if not es_client.ping():
        raise ValueError("Connection failed")
    print("Connected to Elasticsearch successfully.")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")
    exit()

# 3. Xóa và/hoặc tạo Index
try:
    if RECREATE_INDEX and es_client.indices.exists(index=INDEX_NAME):
        print(f"RECREATE_INDEX is True. Deleting existing index '{INDEX_NAME}'...")
        es_client.indices.delete(index=INDEX_NAME)
        print(f"Index '{INDEX_NAME}' deleted.")

    if not es_client.indices.exists(index=INDEX_NAME):
        print(f"Creating index '{INDEX_NAME}' with mapping...")
        es_client.indices.create(index=INDEX_NAME, mappings=INDEX_MAPPING)
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists. Appending data.")
except Exception as e:
    print(f"Error managing index '{INDEX_NAME}': {e}")
    exit()

# 4. Thực hiện Bulk Indexing từ tất cả các file
print(f"\nStarting bulk indexing for {len(file_list)} files into '{INDEX_NAME}'...")
start_time = time.time()
success_count = 0
failed_count = 0
error_details = []

try:
    # Đếm tổng số dòng từ tất cả các file để tqdm hiển thị chính xác
    print("Calculating total documents...")
    total_lines = sum(1 for f in file_list for line in open(f, 'r', encoding='utf-8'))
    print(f"Total documents to process: {total_lines}")

    action_generator = generate_actions_from_files(file_list, INDEX_NAME)
    
    progress_bar = tqdm(total=total_lines, unit="docs", desc="Indexing to ES")
    
    for ok, result in streaming_bulk(
        client=es_client,
        actions=action_generator,
        chunk_size=1000, # Tăng chunk size
        request_timeout=120,
        raise_on_error=False,
        raise_on_exception=False
    ):
        if ok:
            success_count += 1
        else:
            failed_count += 1
            action_type = list(result.keys())[0]
            error_info = result.get(action_type, {}).get('error', {})
            error_reason = error_info.get('reason', 'Unknown reason')
            doc_id = result.get(action_type, {}).get('_id', 'N/A')
            error_details.append(f"Doc ID: {doc_id}, Reason: {error_reason}")
            
        progress_bar.update(1)
        progress_bar.set_postfix({"Success": success_count, "Failed": failed_count})

    progress_bar.close()

except Exception as e:
    print(f"\nA critical error occurred during bulk indexing: {e}")
    traceback.print_exc()

# --- 5. In Tổng kết ---
end_time = time.time()
print("\n--- GLOBAL INDEXING SUMMARY ---")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print(f"Successfully indexed operations: {success_count}")
print(f"Failed operations: {failed_count}")

# Lấy số lượng tài liệu thực tế trong index
try:
    final_count = es_client.count(index=INDEX_NAME)['count']
    print(f"Total documents actually in index '{INDEX_NAME}': {final_count}")
except Exception as e:
    print(f"Could not retrieve final count from index: {e}")


if error_details:
    print("\n--- Failure Details (Sample) ---")
    for i, err in enumerate(error_details[:20]):
        print(err)
    if len(error_details) > 20:
        print(f"(... and {len(error_details) - 20} more failures)")

print("\nImport script finished.")