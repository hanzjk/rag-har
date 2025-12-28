"""
Shared configuration for PAMAP2 openset experiments
"""
import os

# Activity mapping for PAMAP2 dataset
ACTIVITY_MAPPING = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    24: "rope_jumping"
}

# Superclass grouping for activities
SUPERCLASS_MAPPING = {
    "group_1": [1, 2, 3, 4, 5, 12, 13, 6],
}

# Build reverse lookup: activity_id → superclass
ID_TO_SUPERCLASS = {
    activity_id: superclass
    for superclass, ids in SUPERCLASS_MAPPING.items()
    for activity_id in ids
}

# Dataset configuration
WINDOW_SIZE = 256
STEP_SIZE = 128
UNKNOWN_PERC = 0

# Milvus configuration
MILVUS_URI = os.getenv(
    "MILVUS_URI",
    "https://in03-df7c25264724124.serverless.aws-eu-central-1.cloud.zilliz.com"
)
MILVUS_TOKEN = os.getenv(
    "MILVUS_TOKEN",
    "9317f4a8922374f6fc023612f0d788322539108d7ece61c047bc636a99def0ebb71a277c335afbcd3db2365b912a06a085844305"
)

# Collection name template
def get_collection_name(window_size=WINDOW_SIZE, step_size=STEP_SIZE, unknown_perc=UNKNOWN_PERC):
    """Generate collection name based on parameters"""
    return f"activity_recognition_collection_w{window_size}_s{step_size}_{unknown_perc}"

# Base directory for descriptions
def get_base_dir(window_size=WINDOW_SIZE, step_size=STEP_SIZE, unknown_perc=UNKNOWN_PERC):
    """Generate base directory path for descriptions"""
    return f"../descriptions_w{window_size}_s{step_size}/unknown_{unknown_perc}percent/train"

# OpenAI configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
