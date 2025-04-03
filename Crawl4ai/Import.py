from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType,utility
)
import os
# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encoding function for a single text input
def encode_text(text):
    if isinstance(text, str):
        return model.encode(text, normalize_embeddings=True).tolist()
    return []

# Initialize Spark session
spark = SparkSession.builder.appName("ImportJSONLtoMilvus").getOrCreate()

# Read the JSONL data file
df = spark.read.json("./data/rag_blocks_milvus.jsonl")
df.printSchema()

# Define UDF to generate embeddings
encode_udf = udf(encode_text, ArrayType(FloatType()))
df_encoded = df.withColumn("embedding", encode_udf(df["content"]))

# Connect to Milvus and define schema
connections.connect(host="localhost", port="19530")
collection_name = "rag_blocks"

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(
        name="tags",
        dtype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=10,
        max_length=256
    ),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
]


schema = CollectionSchema(fields)
existing_collections = utility.list_collections()

# Create collection if it does not exist
if collection_name not in existing_collections:
    Collection(name=collection_name, schema=schema)
    print(f"Created collection `{collection_name}`.")
else:
    print(f"Collection `{collection_name}` already exists.")

# Function to insert a partition of Spark DataFrame rows into Milvus
def insert_partition(rows):
    from pymilvus import connections, Collection
    connections.connect(host="localhost", port="19530")
    collection = Collection(collection_name)

    data_to_insert = []
    for row in rows:
        tags = row["metadata"].get("tags", []) if isinstance(row["metadata"], dict) else []
        data_to_insert.append([
            row["id"],
            row["content"],
            tags,
            row["embedding"]
        ])
    if data_to_insert:
        fields = list(zip(*data_to_insert))
        collection.insert(fields)
        collection.flush()

# Apply the insert function to each partition
df_encoded.foreachPartition(insert_partition)

print("Data encoding and import to Milvus completed.")
