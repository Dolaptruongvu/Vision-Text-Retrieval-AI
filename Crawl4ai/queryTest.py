from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

connections.connect(host="localhost", port="19530")
collection = Collection("rag_blocks")

# Ensure collection is loaded into query node (won’t load full data!)
collection.load()

model = SentenceTransformer("all-MiniLM-L6-v2")
query_vector = model.encode("Cách chăm sóc cây cam hiệu quả", normalize_embeddings=True).tolist()

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"search_list": 100}},  # for DiskANN
    limit=5,
    output_fields=["id", "content", "tags"]
)

for i, hit in enumerate(results[0]):
    print(f"\n--- Result {i+1} ---")
    print("ID:", hit.entity.get("id"))
    print("Content:", hit.entity.get("content"))
    print("Tags:", hit.entity.get("tags"))
    print("Score:", hit.distance)
