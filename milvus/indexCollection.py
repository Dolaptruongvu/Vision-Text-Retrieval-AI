from pymilvus import connections, Collection

# Kết nối đến Milvus
connections.connect(host="localhost", port="19530")

# Lấy collection
collection = Collection("rag_blocks")

# Tạo index cho trường "embedding" với DiskANN
collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "DISKANN",
        "metric_type": "COSINE"
    }
)

print("✅ Index created.")
