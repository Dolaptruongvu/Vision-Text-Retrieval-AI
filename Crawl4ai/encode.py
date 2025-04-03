from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import os
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# Load model to GPU
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Load JSONL data
input_path = "./rawData/rag_blocks_milvus.jsonl"
output_path = "./encodedData/encoded_blocks.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

# Extract list of texts
texts = [item.get("content", "") for item in lines]

# Encode in batches using GPU
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=8,
    show_progress_bar=True
)

# Save encoded output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as out:
    for item, vec in zip(lines, embeddings):
        out.write(json.dumps({
            "id": item["id"],
            "content": item["content"],
            "tags": item.get("metadata", {}).get("tags", []),
            "embedding": vec.tolist()
        }, ensure_ascii=False) + "\n")
