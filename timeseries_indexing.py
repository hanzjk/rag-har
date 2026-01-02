import glob
import os
import json
import re
import concurrent.futures
import time
import argparse
import logging

import pandas as pd
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv
from langchain_core.documents import Document

from dataset_provider import get_provider

GRPC_TARGET = 38 * 1024 * 1024  # stay under ~48 MiB per RPC
MAX_ROWS = 500  # secondary guard


load_dotenv()

import math
import sys
from typing import Any, Dict, Iterable, List, Sequence, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

Row = Dict[str, Any]
Rows = List[Row]
Columns = Dict[str, Sequence[Any]]
MilvusData = Union[Rows, Columns]


def _is_rows(data: MilvusData) -> bool:
    return isinstance(data, list)


def _num_rows(data: MilvusData) -> int:
    if _is_rows(data):
        return len(data)  # list of row dicts
    # column dict: all columns should have equal length
    return len(next(iter(data.values()))) if data else 0


def _slice_rows(data: Rows, start: int, end: int) -> Rows:
    return data[start:end]


def _slice_columns(data: Columns, start: int, end: int) -> Columns:
    return {k: v[start:end] for k, v in data.items()}


def _sizeof_value(v: Any) -> int:
    """Approximate serialized size in bytes for gRPC budgeting."""
    try:
        import numpy as np  # optional, used if available
    except Exception:
        np = None

    if v is None:
        return 0
    if isinstance(v, (bytes, bytearray, memoryview)):
        return len(v)
    if isinstance(v, str):
        return len(v.encode("utf-8", errors="ignore"))
    if np is not None and isinstance(v, np.ndarray):
        # Milvus vectors are commonly float32
        return int(v.size * v.itemsize)
    if isinstance(v, (list, tuple)):
        # Assume homogeneous; estimate floats/ints as 8 bytes each (conservative)
        if v and isinstance(v[0], (float, int)):
            return len(v) * 8
        # Fallback: sum children (could be strings, nested lists, etc.)
        return sum(_sizeof_value(x) for x in v)
    if isinstance(v, (int, float, bool)):
        return 8
    if isinstance(v, dict):
        # Rare in Milvus rows, but handle just in case
        return sum(_sizeof_value(k) + _sizeof_value(val) for k, val in v.items())
    # Fallback: string repr
    return len(str(v).encode("utf-8", errors="ignore"))


def _estimate_row_size(row: Row) -> int:
    return sum(_sizeof_value(v) for v in row.values())


def _estimate_avg_row_size_rows(rows: Rows, sample: int = 50) -> int:
    if not rows:
        return 0
    n = min(len(rows), sample)
    return max(1, sum(_estimate_row_size(rows[i]) for i in range(n)) // n)


def _estimate_avg_row_size_columns(cols: Columns, sample: int = 50) -> int:
    if not cols:
        return 0
    keys = list(cols.keys())
    n_rows = len(cols[keys[0]]) if keys else 0
    if n_rows == 0:
        return 0
    n = min(n_rows, sample)
    total = 0
    for i in range(n):
        row_bytes = 0
        for k in keys:
            # Access i-th element of each column
            v = cols[k][i]
            row_bytes += _sizeof_value(v)
        total += row_bytes
    return max(1, total // n)


def _utf8_len(x):
    if x is None:
        return 0
    if isinstance(x, (bytes, bytearray)):
        return len(x)
    if isinstance(x, str):
        return len(x.encode("utf-8"))
    return len(str(x).encode("utf-8"))  # dicts, numbers -> string size


def _estimate_row_bytes(row: dict) -> int:
    # 4 vectors * 1536 * 4 bytes
    vec_bytes = (
        sum(
            (
                len(row.get(k) or [])
                for k in (
                    "activity_stats_emb",
                    "activity_stats_start_emb",
                    "activity_stats_mid_emb",
                    "activity_stats_end_emb",
                )
            )
        )
        * 4
    )
    str_bytes = sum(
        _utf8_len(row.get(k))
        for k in (
            "text",
            "timeseries_metadata",
            "stats_whole_text",
            "stats_start_text",
            "stats_mid_text",
            "stats_end_text",
        )
    )
    return vec_bytes + str_bytes + 256  # small overhead


class MultivariateTimeSeriesIndexer:
    def __init__(self, dataset_name, base_dir=None):
        self.dataset_name = dataset_name
        self.collection_name = f"{dataset_name}_collection"
        self.output_dir = f"output/{dataset_name}/documents"
        self.json_file_path = f"{self.output_dir}/{self.collection_name}_mv.json"
        self.base_dir = base_dir

        # Initialize embedding model and Milvus client
        self.embed = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.milvus_client = MilvusClient(
            uri=os.environ.get("ZILLIZ_CLOUD_URI"),
            token=os.environ.get("ZILLIZ_CLOUD_API_KEY"),
            grpc_channel_options=[
                ("grpc.max_send_message_length", 128 * 1024 * 1024),
                ("grpc.max_receive_message_length", 128 * 1024 * 1024),
            ],
        )
        logger.info("Milvus client initialized.")

        # Data storage
        self.multivariate_data_list = []

    def create_collection(self):
        """Create Milvus collection if it doesn't exist"""
        collections = self.milvus_client.list_collections()

        if self.collection_name not in collections:
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=50,
            )
            schema.add_field(
                field_name="timeseries_metadata",
                datatype=DataType.JSON,
                max_length=10000,
            )
            schema.add_field(
                field_name="stats_whole_text",
                datatype=DataType.VARCHAR,
                max_length=10000,
            )
            schema.add_field(
                field_name="stats_start_text",
                datatype=DataType.VARCHAR,
                max_length=10000,
            )
            schema.add_field(
                field_name="stats_mid_text", datatype=DataType.VARCHAR, max_length=10000
            )
            schema.add_field(
                field_name="stats_end_text", datatype=DataType.VARCHAR, max_length=10000
            )
            schema.add_field(
                field_name="activity_stats_emb",
                datatype=DataType.FLOAT_VECTOR,
                dim=1536,
            )
            schema.add_field(
                field_name="activity_stats_start_emb",
                datatype=DataType.FLOAT_VECTOR,
                dim=1536,
            )
            schema.add_field(
                field_name="activity_stats_mid_emb",
                datatype=DataType.FLOAT_VECTOR,
                dim=1536,
            )
            schema.add_field(
                field_name="activity_stats_end_emb",
                datatype=DataType.FLOAT_VECTOR,
                dim=1536,
            )

            index_params = self.milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="activity_stats_emb",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            index_params.add_index(
                field_name="activity_stats_start_emb",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            index_params.add_index(
                field_name="activity_stats_mid_emb",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            index_params.add_index(
                field_name="activity_stats_end_emb",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )

            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                metric_type="COSINE",
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    def embed_with_retry(self, text: str, max_retries: int = 3) -> List[float]:
        """Embed a single text with retry logic for connection errors"""
        for attempt in range(max_retries):
            try:
                return self.embed.embed_query(str(text))
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                time.sleep(60)  # Simple 60 second delay between retries

    def embed_batch(self, texts_batch):
        """Process a batch of texts to embed"""
        results = []
        for text in texts_batch:
            results.append(self.embed_with_retry(text))
        return results

    def parallel_embed(self, texts_list, max_workers=10, batch_size=20):
        """Embed a list of texts in parallel using concurrent.futures while maintaining order"""
        batches = [
            texts_list[i : i + batch_size]
            for i in range(0, len(texts_list), batch_size)
        ]
        results = [None] * len(batches)  # Pre-allocate results array to maintain order

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches and store future-to-index mapping
            future_to_index = {
                executor.submit(self.embed_batch, batch): i
                for i, batch in enumerate(batches)
            }

            # Process results while maintaining order
            for future in tqdm(
                concurrent.futures.as_completed(future_to_index),
                total=len(batches),
                desc="Embedding batches",
            ):
                batch_index = future_to_index[future]
                try:
                    batch_results = future.result()
                    results[batch_index] = batch_results  # Store in correct position
                except Exception as e:
                    logger.error(f"Error processing batch {batch_index}: {e}")
                    # Add None results to maintain order
                    results[batch_index] = [None] * len(batches[batch_index])

        # Flatten results while maintaining order
        flattened_results = []
        for batch_result in results:
            if batch_result is not None:
                flattened_results.extend(batch_result)
            else:
                flattened_results.extend([None] * batch_size)  # Maintain length

        return flattened_results

    def extract_sensor_sections(self, text):
        """
        Given the file content as a string, extract each sensor section for all segments.
        Returns a dict with structure:
        {
            'whole': {content for whole segment},
            'start': {content for start segment},
            'mid': {content for mid segment},
            'end': {content for end segment}
        }
        """
        segments = {"whole": "", "start": "", "mid": "", "end": ""}

        # Split text by segment headers
        segment_pattern = r"\[(Whole|Start|Mid|End) Segment\](.*?)(?=\[(?:Whole|Start|Mid|End) Segment\]|$)"
        segment_matches = re.findall(segment_pattern, text, re.DOTALL)

        for segment_name, segment_content in segment_matches:
            segments[segment_name.lower()] = segment_content.strip()

        return segments

    def extract_and_embed_data(self):
        """Extract multivariate time series data and create embeddings"""
        logger.info(
            f"Extracting and processing {self.dataset_name} time series data..."
        )

        # Prepare data for embedding
        stats_to_embed = []
        start_stats_to_embed = []
        mid_stats_to_embed = []
        end_stats_to_embed = []
        metadata_list = []

        logger.info(f"Processing {self.dataset_name} time series for embedding...")

        file_list = glob.glob(os.path.join(self.base_dir, "*.txt"))
        for file_path in file_list:
            try:
                base = os.path.basename(file_path)
                # Generic filename pattern: window_1000_activity_wlk_stats.txt
                m = re.match(r"window_(\d+)_activity_([A-Za-z0-9_-]+)_stats\.txt", base)
                if not m:
                    logger.warning(f"Filename not matched: {base}")
                    continue
                window_id, activity_id = m.groups()

                with open(file_path, "r") as f:
                    content = f.read()

                # Extract sensor sections
                logger.debug(
                    f"Extracting stat descriptions for window:{window_id} activity:{activity_id}"
                )
                sensors = self.extract_sensor_sections(content)
                stats_to_embed.append(sensors["whole"])
                start_stats_to_embed.append(sensors["start"])
                mid_stats_to_embed.append(sensors["mid"])
                end_stats_to_embed.append(sensors["end"])

                metadata_list.append(
                    {
                        "dataset": self.dataset_name,
                        "window_id": window_id,
                        "activity_id": activity_id,
                        "stats_whole_text": sensors["whole"],
                        "stats_start_text": sensors["start"],
                        "stats_mid_text": sensors["mid"],
                        "stats_end_text": sensors["end"],
                    }
                )
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

        # Process embeddings in parallel
        logger.info(f"Creating embeddings for {len(metadata_list)} time series...")
        stats_embedded_values = self.parallel_embed(
            stats_to_embed, max_workers=10, batch_size=20
        )
        start_stats_embedded_values = self.parallel_embed(
            start_stats_to_embed, max_workers=10, batch_size=20
        )
        mid_stats_embedded_values = self.parallel_embed(
            mid_stats_to_embed, max_workers=10, batch_size=20
        )
        end_stats_embedded_values = self.parallel_embed(
            end_stats_to_embed, max_workers=10, batch_size=20
        )

        # Create the final data list with proper metadata-embedding association
        self.multivariate_data_list = []
        for i, metadata in enumerate(metadata_list):
            text_id = f"{self.dataset_name}_window{metadata['window_id']}_activity_{metadata['activity_id']}"
            self.multivariate_data_list.append(
                {
                    "text": text_id,
                    "text_id": text_id,
                    "stats_whole_text": metadata["stats_whole_text"],
                    "stats_start_text": metadata["stats_start_text"],
                    "stats_mid_text": metadata["stats_mid_text"],
                    "stats_end_text": metadata["stats_end_text"],
                    "activity_stats_emb": stats_embedded_values[i],
                    "activity_stats_start_emb": start_stats_embedded_values[i],
                    "activity_stats_mid_emb": mid_stats_embedded_values[i],
                    "activity_stats_end_emb": end_stats_embedded_values[i],
                    "dataset": metadata["dataset"],
                    "window_id": metadata["window_id"],
                    "activity_id": metadata["activity_id"],
                }
            )

        logger.info(
            f"Successfully processed {len(self.multivariate_data_list)} time series data points"
        )

    def dicts_to_documents(self, data_list):
        """Convert data dictionaries to Document objects"""
        docs = []
        for d in data_list:
            doc_id = d["text_id"]
            docs.append(
                Document(
                    page_content=str(d["text"]),
                    metadata={
                        "doc_id": doc_id,
                        "dataset": d["dataset"],
                        "activity_id": d.get("activity_id"),
                        "window_id": d.get("window_id"),
                    },
                )
            )
        return docs

    def documents_to_serializable(self, documents, data_list):
        """Convert Document objects to serializable dictionaries"""
        serializable_docs = []
        for i, doc in enumerate(documents):
            serializable_docs.append(
                {
                    "page_content": doc.page_content,
                    "stats_whole_text": data_list[i]["stats_whole_text"],
                    "stats_start_text": data_list[i]["stats_start_text"],
                    "stats_mid_text": data_list[i]["stats_mid_text"],
                    "stats_end_text": data_list[i]["stats_end_text"],
                    "activity_stats_emb": data_list[i]["activity_stats_emb"],
                    "activity_stats_start_emb": data_list[i][
                        "activity_stats_start_emb"
                    ],
                    "activity_stats_mid_emb": data_list[i]["activity_stats_mid_emb"],
                    "activity_stats_end_emb": data_list[i]["activity_stats_end_emb"],
                    "metadata": doc.metadata,
                }
            )
        return serializable_docs

    def save_documents_to_file(self, documents):
        """Save documents to JSON file"""
        serializable_docs = self.documents_to_serializable(
            documents, self.multivariate_data_list
        )

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.json_file_path), exist_ok=True)

        with open(self.json_file_path, "w") as f:
            json.dump(list(serializable_docs), f, indent=2)
        logger.info(f"Documents saved to {self.json_file_path}")

    def load_documents_from_file(self):
        """Load documents from JSON file"""
        try:
            with open(self.json_file_path, "r") as f:
                serializable_docs = json.load(f)

            logger.info(
                f"Loaded {len(serializable_docs)} documents from {self.json_file_path}"
            )
            return serializable_docs
        except FileNotFoundError:
            logger.info(
                f"File {self.json_file_path} not found. Will extract and create new data."
            )
            return None
        except Exception as e:
            logger.error(f"Error loading documents from {self.json_file_path}: {e}")
            return None

    def prepare_data_for_milvus(self, serializable_docs):
        """Convert serializable documents to format expected by Milvus"""
        milvus_data = []
        for doc in serializable_docs:
            # Ensure we have all the required fields
            if "page_content" not in doc or "metadata" not in doc:
                logger.warning(f"Skipping document with missing fields: {doc.keys()}")
                continue

            # Use the page_content as the text field (original full content)
            mv_timeseries_text = doc["page_content"]
            # Truncate if too long for the field limit
            if len(mv_timeseries_text) > 5:  # Leave some buffer for the 5000 char limit
                mv_timeseries_text = mv_timeseries_text[:5] + "..."

            milvus_data.append(
                {
                    "text": doc["metadata"]["doc_id"],
                    "timeseries_metadata": doc["metadata"],
                    "stats_whole_text": doc.get("stats_whole_text", ""),
                    "stats_start_text": doc.get("stats_start_text", ""),
                    "stats_mid_text": doc.get("stats_mid_text", ""),
                    "stats_end_text": doc.get("stats_end_text", ""),
                    "activity_stats_emb": doc["activity_stats_emb"],
                    "activity_stats_start_emb": doc["activity_stats_start_emb"],
                    "activity_stats_mid_emb": doc["activity_stats_mid_emb"],
                    "activity_stats_end_emb": doc["activity_stats_end_emb"],
                }
            )
        return milvus_data

    def insert_data_to_milvus(self, serializable_docs):
        rows = self.prepare_data_for_milvus(serializable_docs)
        if not rows:
            logger.warning("No data to insert into Milvus")
            return

        total, batch, batch_bytes = 0, [], 0
        try:
            for r in rows:
                sz = _estimate_row_bytes(r)

                # flush if this row would push us over the target or max rows
                if batch and (batch_bytes + sz > GRPC_TARGET or len(batch) >= MAX_ROWS):
                    self.milvus_client.insert(
                        collection_name=self.collection_name, data=batch
                    )
                    total += len(batch)
                    logger.info(f"Inserted {len(batch)} docs (Total: {total})")
                    batch, batch_bytes = [], 0

                # if one row itself is big, send it alone
                if sz > GRPC_TARGET and not batch:
                    self.milvus_client.insert(
                        collection_name=self.collection_name, data=[r]
                    )
                    total += 1
                    logger.info(f"Inserted 1 large doc (Total: {total})")
                else:
                    batch.append(r)
                    batch_bytes += sz

            if batch:
                self.milvus_client.insert(
                    collection_name=self.collection_name, data=batch
                )
                total += len(batch)
                logger.info(f"Inserted {len(batch)} docs (Total: {total})")

            logger.info(
                f"Successfully inserted {total} documents into collection: {self.collection_name}"
            )
        except Exception as e:
            logger.error(f"Error inserting documents into Milvus: {e}")

    def process_and_index(self):
        """Main method to process data and index it"""
        # Check if JSON file exists
        serializable_docs = self.load_documents_from_file()

        if serializable_docs is None:
            # Extract and embed data if JSON file doesn't exist
            self.extract_and_embed_data()
            documents = self.dicts_to_documents(self.multivariate_data_list)
            self.save_documents_to_file(documents)
            serializable_docs = self.load_documents_from_file()

        # Create collection and insert data
        self.create_collection()
        self.insert_data_to_milvus(serializable_docs)

        return serializable_docs


def main():
    """
    Time Series Indexing Pipeline.
    Creates embeddings and indexes time series data to Milvus vector database.
    """
    parser = argparse.ArgumentParser(
        description="Index time series features to Milvus vector database"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to dataset configuration YAML file",
    )

    args = parser.parse_args()

    # Get dataset provider
    logger.info(f"Loading dataset configuration from {args.config}")
    provider = get_provider(args.config)

    # Automatic paths based on dataset name and split
    dataset_name = provider.dataset_name
    descriptions_dir = f"output/{dataset_name}/features/train/descriptions"

    logger.info("")
    logger.info("=" * 80)
    logger.info("TIME SERIES INDEXING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Descriptions directory: {descriptions_dir}")
    logger.info("=" * 80)
    logger.info("")

    # Create indexer
    indexer = MultivariateTimeSeriesIndexer(
        dataset_name=dataset_name,
        base_dir=descriptions_dir,
    )

    # Process and index
    documents = indexer.process_and_index()

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✓ Indexing complete!")
    logger.info(f"✓ {len(documents)} time series documents indexed")
    logger.info(f"✓ Collection: {indexer.collection_name}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
