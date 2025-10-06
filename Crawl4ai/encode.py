from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import os
import torch
import numpy as np

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    device = "cuda"
else:
    print("CUDA not available, using CPU.")
    device = "cpu"

# Load model to the determined device
# model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# model_name = "intfloat/multilingual-e5-large-instruct"
model_name = "HIT-TMG/KaLM-embedding-multilingual-mini-v1" 
# model_name = "ibm-granite/granite-embedding-278m-multilingual" 
print(f"\nLoading model '{model_name}' onto device '{device}'...")
try:
    model = SentenceTransformer(model_name, device=device)
    print("Model loaded successfully.")
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model embedding dimension: {embedding_dim}")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    exit()


# Load JSONL data safely
input_path = "./rawData/data5.jsonl"
output_path = "./encodedData_Final/encoded_blocks_HIT_TMG_KaLM_embedding_multilingual_mini_v1_5.jsonl"
print(f"\nReading and validating JSONL data from: {input_path}")

valid_lines_data = [] # List chứa các dictionary JSON hợp lệ đã đọc
texts_to_encode = []  # List chứa các content tương ứng để encode
skipped_lines = 0

try:
    with open(input_path, "r", encoding="utf-8") as f:
        # Sử dụng tqdm để xem tiến trình đọc file nếu file lớn
        for i, line in enumerate(tqdm(f, desc="Reading lines")):
            line = line.strip() # Loại bỏ khoảng trắng thừa
            if not line:       # Bỏ qua dòng trống
                skipped_lines += 1
                continue

            try:
                data = json.loads(line)
                # Kiểm tra các trường cơ bản cần thiết
                if "id" in data and "content" in data and data["id"] and isinstance(data["content"], str):
                     valid_lines_data.append(data) # Lưu lại data gốc hợp lệ
                     texts_to_encode.append(data["content"]) # Thêm content vào list để encode
                else:
                     print(f"Warning: Skipping line {i+1} due to missing/invalid 'id' or 'content'.")
                     skipped_lines += 1

            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {i+1}: {line}")
                skipped_lines += 1
            except Exception as e:
                 print(f"Warning: Error processing line {i+1}: {e}. Skipping.")
                 skipped_lines += 1

except FileNotFoundError:
    print(f"Error: Input file not found at {input_path}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the input file: {e}")
    exit()

print(f"Finished reading. Found {len(valid_lines_data)} valid entries. Skipped {skipped_lines} lines.")

if not texts_to_encode:
    print("No valid texts found to encode. Exiting.")
    exit()

# Encode in batches using GPU (or CPU if CUDA not available)
print(f"\nEncoding {len(texts_to_encode)} texts using model '{model_name}'...")
embeddings = model.encode(
    texts_to_encode,            # Chỉ encode các content hợp lệ
    normalize_embeddings=True,
    batch_size=32,            
    show_progress_bar=True,
    convert_to_numpy=True      
)
print("Encoding complete.")


if np.isnan(embeddings).any() or np.isinf(embeddings).any():
    print("WARNING: NaN or Infinity found in generated embeddings! This might cause issues during insertion or search.")
    

# Save encoded output
print(f"\nSaving encoded data to: {output_path}")
os.makedirs(os.path.dirname(output_path), exist_ok=True) 
saved_count = 0
try:
    with open(output_path, "w", encoding="utf-8") as out:
        # Lặp qua dữ liệu gốc hợp lệ và embeddings tương ứng
        for item, vec in zip(valid_lines_data, embeddings):
            try:
                # Tạo dictionary output
                 output_data = {
                    "id": item["id"],
                    "content": item["content"],
                    # Lấy tags an toàn hơn, đảm bảo là list
                    "tags": item.get("metadata", {}).get("tags", []) or [],
                    # Chuyển numpy array thành list[float]
                    "embedding": vec.tolist()
                 }
                 # Ghi vào file
                 json.dump(output_data, out, ensure_ascii=False)
                 out.write('\n')
                 saved_count += 1
            except Exception as write_e:
                 print(f"Error writing entry for ID {item.get('id', 'UNKNOWN')}: {write_e}")

    print(f"Successfully wrote {saved_count} encoded entries to {output_path}.")

except Exception as e:
    print(f"Error writing to output file {output_path}: {e}")

print("\nEncoding script finished.")