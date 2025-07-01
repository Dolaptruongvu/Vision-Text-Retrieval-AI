import json
import os
import glob
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from tqdm import tqdm
import traceback

# --- Cấu hình ---
# Đổi tên collection mới để không bị ảnh hưởng bởi dữ liệu cũ
COLLECTION_NAME = "rag_collection_v2"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
BATCH_SIZE = 1000
EMBEDDING_DIM = 384
MAX_TAG_CAPACITY = 100 

# --- CẤU HÌNH QUAN TRỌNG ---
# Đường dẫn đến thư mục chứa dữ liệu
DATA_DIRECTORY = "../Crawl4ai/encodedData/"
# Mẫu để tìm các file dữ liệu
FILE_PATTERN = "encoded_blocks*.jsonl"
# Đặt thành True nếu bạn muốn xóa sạch collection cũ trước khi import
RECREATE_COLLECTION = True

# --- Định nghĩa Schema ---
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, auto_id=False),
    FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=MAX_TAG_CAPACITY, max_length=256),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
]
schema = CollectionSchema(fields, description="RAG Collection for Document Blocks")


# --- Bắt đầu Script ---

# 1. Tìm tất cả các file dữ liệu
search_pattern = os.path.join(DATA_DIRECTORY, FILE_PATTERN)
file_list = sorted(glob.glob(search_pattern))

if not file_list:
    print(f"Error: No files found matching pattern '{search_pattern}'")
    exit()

print("--- Found data files to process ---")
for f in file_list:
    print(f"- {os.path.basename(f)}")
print("-" * 35)

# 2. Đọc tất cả dữ liệu và loại bỏ trùng lặp trong bộ nhớ
print("\nStep 1: Reading all files and de-duplicating in memory...")
all_data = {}
skipped_log = []

for filepath in tqdm(file_list, desc="Reading files"):
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line_num = i + 1
            try:
                data = json.loads(line)
                doc_id = data.get("id")
                embedding = data.get("embedding")

                # Xác thực dữ liệu
                if not doc_id:
                    skipped_log.append(f"File {os.path.basename(filepath)}, Line {line_num}: Missing 'id'.")
                    continue
                if not embedding or len(embedding) != EMBEDDING_DIM:
                    skipped_log.append(f"File {os.path.basename(filepath)}, Line {line_num}, ID {doc_id}: Invalid 'embedding'.")
                    continue

                # Lưu vào dict, tự động ghi đè nếu ID trùng lặp
                all_data[doc_id] = {
                    "content": data.get("content", ""),
                    "tags": data.get("tags", []) or [], # Đảm bảo không phải None
                    "embedding": embedding
                }
            except (json.JSONDecodeError, TypeError) as e:
                skipped_log.append(f"File {os.path.basename(filepath)}, Line {line_num}: JSON/Type Error - {e}")
                continue

print(f"Reading complete. Found {len(all_data)} unique documents.")
if skipped_log:
    print(f"Skipped {len(skipped_log)} lines due to errors.")
    # In ra vài lỗi đầu tiên để debug
    for log_entry in skipped_log[:5]:
        print(f"  - {log_entry}")
    if len(skipped_log) > 5:
        print(f"  - (... and {len(skipped_log) - 5} more errors)")

# 3. Kết nối Milvus và quản lý Collection
try:
    print("\nStep 2: Connecting to Milvus and preparing collection...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if RECREATE_COLLECTION and utility.has_collection(COLLECTION_NAME):
        print(f"RECREATE_COLLECTION is True. Deleting existing collection '{COLLECTION_NAME}'...")
        utility.drop_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' deleted.")

    if not utility.has_collection(COLLECTION_NAME):
        collection = Collection(name=COLLECTION_NAME, schema=schema, using='default')
        print(f"Created new collection '{COLLECTION_NAME}'.")
    else:
        collection = Collection(name=COLLECTION_NAME)
        print(f"Using existing collection '{COLLECTION_NAME}'.")

    # 4. Chuẩn bị dữ liệu và thực hiện Insert theo batch
    print("\nStep 3: Preparing data and starting batch insert...")
    
    # Chuyển dict thành các list riêng biệt
    ids_to_insert = list(all_data.keys())
    docs_to_insert = list(all_data.values())
    
    contents = [doc['content'] for doc in docs_to_insert]
    tags = [doc['tags'] for doc in docs_to_insert]
    embeddings = [doc['embedding'] for doc in docs_to_insert]

    # Insert theo từng batch
    for i in tqdm(range(0, len(ids_to_insert), BATCH_SIZE), desc="Inserting to Milvus"):
        end_index = i + BATCH_SIZE
        batch_ids = ids_to_insert[i:end_index]
        batch_contents = contents[i:end_index]
        batch_tags = tags[i:end_index]
        batch_embeddings = embeddings[i:end_index]
        
        collection.insert([batch_ids, batch_contents, batch_tags, batch_embeddings])

    print("All batches inserted.")

    # 5. Flush, tạo Index và xác thực
    print("\nStep 4: Flushing data and creating index...")
    collection.flush()
    print(f"Flush complete. Total entities in collection: {collection.num_entities}")

    if not collection.has_index():
        print("Creating index...")
        index_params = {"index_type": "DISKANN", "metric_type": "COSINE", "params": {}}
        collection.create_index("embedding", index_params)
        utility.wait_for_index_building_complete(COLLECTION_NAME)
        print("Index created successfully.")
    else:
        print("Index already exists.")

    # 6. Tải và xác thực lần cuối
    print("\nStep 5: Final validation...")
    collection.load()
    final_count = collection.num_entities
    print(f"--- FINAL COUNT IN MILVUS: {final_count} ---")


except Exception as e:
    print(f"\nAn error occurred: {e}")
    traceback.print_exc()

finally:
    connections.disconnect("default")
    print("\nDisconnected from Milvus. Processing complete.")