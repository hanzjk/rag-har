"""
Shared utility functions for PAMAP2 openset experiments
"""
import glob
import os
import random
import re
import time
import asyncio
from typing import Dict, List, Any

import openai
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest

from shared_config import ACTIVITY_MAPPING


def extract_sensor_sections(text: str) -> Dict[str, str]:
    """
    Extract sensor sections from text for all temporal segments.

    Args:
        text: Raw text content containing segment markers

    Returns:
        Dict with structure:
        {
            'whole': <segment_content>,
            'start': <segment_content>,
            'mid': <segment_content>,
            'end': <segment_content>
        }
    """
    segments = {'whole': {}, 'start': {}, 'mid': {}, 'end': {}}

    # Split text by segment headers
    segment_pattern = r'\[(Whole|Start|Mid|End) Segment\](.*?)(?=\[(?:Whole|Start|Mid|End) Segment\]|$)'
    segment_matches = re.findall(segment_pattern, text, re.DOTALL)

    for segment_name, segment_content in segment_matches:
        segments[segment_name.lower()] = segment_content.strip()

    return segments


def get_samples_from_descriptions_directly(
    all_class_ids: List[str],
    samples_per_class: int,
    descriptions_dir: str,
    test_activities: Any  # Can be single int or list of ints
) -> List[Dict]:
    """
    Extract samples directly from description files instead of using RAG.
    Randomly selects samples_per_class from each activity class.

    Args:
        all_class_ids: List of class IDs to sample from
        samples_per_class: Number of samples to get from each class
        descriptions_dir: Base directory containing description files
        test_activities: Test activity ID(s) to exclude from sampling (int or list)

    Returns:
        List of mock documents with sensor data
    """
    # Ensure test_activities is a list
    if isinstance(test_activities, int):
        test_activities = [test_activities]

    all_selected_docs = []

    for class_id in all_class_ids:
        class_description_dir = os.path.join(
            descriptions_dir,
            f"activity{class_id}_{ACTIVITY_MAPPING.get(int(class_id))}"
        )

        if not os.path.exists(class_description_dir):
            print(f"Warning: Description directory for class {class_id} not found: {class_description_dir}")
            continue

        # Get all description files for this class
        description_files = glob.glob(os.path.join(class_description_dir, "*_stat.txt"))

        if len(description_files) == 0:
            print(f"Warning: No description files found for class {class_id}")
            continue

        # Randomly sample files
        random.seed(42)  # For reproducibility
        selected_files = random.sample(description_files, min(samples_per_class, len(description_files)))

        print(f"Selected {len(selected_files)} samples from class {class_id} ({ACTIVITY_MAPPING.get(int(class_id), 'unknown')})")

        for file_path in selected_files:
            try:
                with open(file_path, "r") as f:
                    content = f.read()

                # Extract sensor sections
                sensors = extract_sensor_sections(content)

                # Create a mock document structure similar to what RAG returns
                mock_doc = {
                    'class_id': class_id,
                    'whole_stats': sensors['whole'],
                    'start_stats': sensors['start'],
                    'mid_stats': sensors['mid'],
                    'end_stats': sensors['end'],
                    'file_path': file_path
                }

                all_selected_docs.append(mock_doc)

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

    return all_selected_docs


async def search_class_samples_async(
    milvus_client: MilvusClient,
    collection_name: str,
    class_id: str,
    stats_emb: List[float],
    start_stats_emb: List[float],
    mid_stats_emb: List[float],
    end_stats_emb: List[float],
    samples_per_class: int,
    semaphore: asyncio.Semaphore
) -> tuple:
    """
    Async function to search for samples from a specific class using hybrid search.

    Args:
        milvus_client: Milvus client instance
        collection_name: Name of the Milvus collection
        class_id: Activity class ID to search for
        stats_emb: Embedding for whole segment
        start_stats_emb: Embedding for start segment
        mid_stats_emb: Embedding for mid segment
        end_stats_emb: Embedding for end segment
        samples_per_class: Number of samples to retrieve
        semaphore: Asyncio semaphore for rate limiting

    Returns:
        Tuple of (class_id, search_results)
    """
    async with semaphore:  # Limit concurrent requests
        expr = f'timeseries_metadata["activity_id"] == "{class_id}"'

        req_1 = AnnSearchRequest(
            anns_field="activity_stats",
            data=[stats_emb],
            limit=samples_per_class,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            expr=expr
        )
        req_2 = AnnSearchRequest(
            anns_field="activity_stats_start",
            data=[start_stats_emb],
            limit=samples_per_class,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            expr=expr
        )
        req_3 = AnnSearchRequest(
            anns_field="activity_stats_mid",
            data=[mid_stats_emb],
            limit=samples_per_class,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            expr=expr
        )
        req_4 = AnnSearchRequest(
            anns_field="activity_stats_end",
            data=[end_stats_emb],
            limit=samples_per_class,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            expr=expr
        )

        # Run the hybrid search in an executor to make it async
        loop = asyncio.get_event_loop()
        class_docs = await loop.run_in_executor(
            None,
            lambda: milvus_client.hybrid_search(
                collection_name=collection_name,
                output_fields=["text", "timeseries_metadata", "stats_whole_text"],
                reqs=[req_1, req_2, req_3, req_4],
                limit=samples_per_class,
                ranker=WeightedRanker(0.25, 0.25, 0.25, 0.25)
            )
        )

        return class_id, class_docs


async def get_samples_from_all_classes_async(
    milvus_client: MilvusClient,
    collection_name: str,
    all_class_ids: List[str],
    stats_emb: List[float],
    start_stats_emb: List[float],
    mid_stats_emb: List[float],
    end_stats_emb: List[float],
    samples_per_class: int,
    max_concurrent: int = 5
) -> List[Any]:
    """
    Async function to get samples from all classes in parallel.

    Args:
        milvus_client: Milvus client instance
        collection_name: Name of the Milvus collection
        all_class_ids: List of class IDs to retrieve samples from
        stats_emb: Embedding for whole segment
        start_stats_emb: Embedding for start segment
        mid_stats_emb: Embedding for mid segment
        end_stats_emb: Embedding for end segment
        samples_per_class: Number of samples to retrieve per class
        max_concurrent: Maximum number of concurrent requests

    Returns:
        List of all retrieved documents
    """
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all classes
    tasks = [
        search_class_samples_async(
            milvus_client, collection_name, class_id,
            stats_emb, start_stats_emb, mid_stats_emb, end_stats_emb,
            samples_per_class, semaphore
        )
        for class_id in all_class_ids
    ]

    # Run all searches in parallel
    results = await asyncio.gather(*tasks)

    # Collect all documents
    all_selected_docs = []
    for class_id, class_docs in results:
        # Flatten the class_docs - each class_docs is a list containing search results
        class_doc_count = 0
        for doc_group in class_docs:
            for hit in doc_group:
                all_selected_docs.append(hit)
                class_doc_count += 1
        print(f"Selected {class_doc_count} samples from class {class_id} ({ACTIVITY_MAPPING.get(int(class_id), 'unknown')})")

    return all_selected_docs


def openai_api_call_with_retry(client, model: str, messages: List[Dict], response_format=None):
    """
    Call OpenAI API with automatic retry on rate limit or errors.

    Args:
        client: OpenAI client instance
        model: Model name to use
        messages: List of message dictionaries
        response_format: Optional Pydantic model for structured output

    Returns:
        Parsed response object
    """
    success = False
    while not success:
        try:
            print("Sending request to OpenAI API...")
            if response_format:
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format
                )
            else:
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages
                )

            print("Received response from OpenAI API")
            success = True
            return response

        except openai.RateLimitError:
            print("Rate limit reached. Waiting 65 seconds before retrying...")
            time.sleep(65)
        except Exception as e:
            print(f"OpenAI API error: {e}. Waiting 10 seconds before retrying...")
            time.sleep(10)
