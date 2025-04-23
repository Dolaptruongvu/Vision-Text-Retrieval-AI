from pymilvus import connections, Collection, utility

# --- Thông tin kết nối và Collection ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_blocks"

# --- Kết nối đến Milvus ---
print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
try:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Successfully connected to Milvus.")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
    exit()

# --- Kiểm tra Collection tồn tại ---
if not utility.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' does not exist. Nothing to drop.")
else:
    # --- Thực hiện drop collection ---
    print(f"Attempting to drop collection: '{COLLECTION_NAME}'...")
    try:
        utility.drop_collection(COLLECTION_NAME)
        print(f"Successfully dropped collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"An error occurred while dropping the collection: {e}")

    finally:
        # --- Ngắt kết nối ---
        print("Disconnecting from Milvus.")
        connections.disconnect("default")