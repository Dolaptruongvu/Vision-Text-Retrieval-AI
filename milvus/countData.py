from pymilvus import connections, Collection, utility

# --- Cấu hình ---
COLLECTION_NAME = "rag_blocks"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

try:
    # --- 1. Kết nối Milvus ---
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Connection successful.")

    # --- 2. Kiểm tra collection có tồn tại không ---
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
    else:
        # --- 3. Lấy đối tượng collection ---
        collection = Collection(name=COLLECTION_NAME)
        
        # --- 4. Tải collection vào bộ nhớ (BẮT BUỘC) ---
        print(f"Loading collection '{COLLECTION_NAME}' into memory...")
        collection.load()
        print("Collection loaded.")
        
        # --- 5. Lấy và in ra tổng số entity ---
        # collection.num_entities sẽ lấy số liệu đã được flush và load
        total_entities = collection.num_entities
        print("\n--- Milvus Collection Stats ---")
        print(f"Total entities in collection '{COLLECTION_NAME}': {total_entities}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # --- 6. Ngắt kết nối ---
    connections.disconnect("default")
    print("\nDisconnected from Milvus.")