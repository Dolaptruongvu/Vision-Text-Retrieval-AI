import json
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# --- Cấu hình CỐ ĐỊNH ---
JSONL_FILE_PATH = "./encodedData/encoded_blocks4.jsonl"
COLLECTION_NAME = "rag_blocks"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
BATCH_SIZE = 1000 # Tăng batch size để ít lần gọi insert hơn
EMBEDDING_DIM = 384
MAX_TAG_CAPACITY = 100 # Phải khớp với schema nếu collection đã tồn tại!

# --- Kết nối Milvus ---
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# --- Định nghĩa Schema ---
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=MAX_TAG_CAPACITY, max_length=256),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
]
schema = CollectionSchema(fields)

# --- Tạo Collection nếu chưa có (KHÔNG xóa cái cũ) ---
# LƯU Ý: Nếu collection cũ tồn tại với schema KHÁC, việc insert sẽ lỗi!
# Bạn cần xóa thủ công hoặc chạy script recreate trước nếu muốn đổi schema.
if not utility.has_collection(COLLECTION_NAME):
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"Created collection '{COLLECTION_NAME}'.") # Giữ lại print này
else:
    collection = Collection(name=COLLECTION_NAME)

# --- Đọc file và Insert dữ liệu ---
batch_ids = []
batch_contents = []
batch_tags = []
batch_embeddings = []

with open(JSONL_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            # Giả định dữ liệu hợp lệ và đúng kiểu/kích thước
            embedding_data = data.get("embedding")
            # Kiểm tra cơ bản nhất
            if isinstance(embedding_data, list) and len(embedding_data) == EMBEDDING_DIM:
                batch_ids.append(data["id"])
                batch_contents.append(data.get("content", ""))
                tags = data.get("tags", [])
                batch_tags.append(tags if tags is not None else []) # Đảm bảo tags là list
                batch_embeddings.append(embedding_data) # Giữ nguyên list số (hy vọng là float)
            # else: Bỏ qua dòng lỗi lặng lẽ

        except (json.JSONDecodeError, KeyError, TypeError):
            # Bỏ qua dòng lỗi JSON hoặc thiếu key bắt buộc (id, embedding) lặng lẽ
            continue

        # Insert batch khi đủ lớn
        if len(batch_ids) >= BATCH_SIZE:
            collection.insert([batch_ids, batch_contents, batch_tags, batch_embeddings])
            batch_ids, batch_contents, batch_tags, batch_embeddings = [], [], [], [] # Reset batch

# Insert phần còn lại (batch cuối)
if batch_ids:
    collection.insert([batch_ids, batch_contents, batch_tags, batch_embeddings])

# --- Flush dữ liệu ---
collection.flush()

# --- Tạo Index nếu chưa có ---
if not collection.has_index():
    index_params = {
        "index_type": "DISKANN",
        "metric_type": "COSINE", # Hoặc "L2"
        "params": {"search_list": 100}
    }
    collection.create_index("embedding", index_params)
    # utility.wait_for_index_building_complete(COLLECTION_NAME) # Bỏ chờ đợi cho ngắn gọn

# --- Ngắt kết nối ---
connections.disconnect("default")

print("Processing complete.") # Một thông báo cuối cùng