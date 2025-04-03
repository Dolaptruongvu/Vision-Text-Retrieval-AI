from pyspark.sql import SparkSession
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)
import os
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"
# 1. Start Spark session
spark = SparkSession.builder.appName("ImportEncodedJSONLtoMilvus").getOrCreate()

# 2. Read the encoded JSONL
df = spark.read.json("./encodedData/encoded_blocks.jsonl")
df.printSchema()

# 3. Connect to Milvus
connections.connect(host="localhost", port="19530")
collection_name = "rag_blocks"

# 4. Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(
        name="tags",
        dtype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=10,
        max_length=256,
    ),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
]

schema = CollectionSchema(fields)

# 5. Create collection if not exists
if collection_name not in utility.list_collections():
    Collection(name=collection_name, schema=schema)
    print(f"Created collection `{collection_name}`.")
else:
    print(f"Collection `{collection_name}` already exists.")

# 6. Define partition insert logic
def insert_partition(rows):
    from pymilvus import connections, Collection
    connections.connect(host="localhost", port="19530")
    collection = Collection(collection_name)

    data_to_insert = []
    for row in rows:
        data_to_insert.append([
            row["id"],
            row["content"],
            row["tags"],
            row["embedding"]
        ])
    if data_to_insert:
        fields = list(zip(*data_to_insert))
        collection.insert(fields)
        collection.flush()

# 7. Insert each partition into Milvus
df.foreachPartition(insert_partition)

print("Imported encoded JSONL into Milvus using Spark.")

# 8. Build index (DISKANN or IVF_FLAT)
collection = Collection(collection_name)
collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "DISKANN",  # or IVF_FLAT
        "metric_type": "COSINE"
    }
)
print("Index created.")
