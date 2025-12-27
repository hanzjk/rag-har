"""
RAG-based Activity Classifier - Adapted to dataset-agnostic architecture.

Uses hybrid search with temporal segmentation (whole, start, mid, end) and
LLM-based classification for activity recognition.
"""

import argparse
import glob
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv
import openai
from tqdm import tqdm

from dataset_provider import get_provider

load_dotenv()


def extract_sensor_sections(text: str) -> Dict[str, str]:
    """
    Extract sensor sections for temporal segments from description text.

    Args:
        text: File content as string

    Returns:
        Dict with structure: {'whole': ..., 'start': ..., 'mid': ..., 'end': ...}
    """
    segments = {'whole': {}, 'start': {}, 'mid': {}, 'end': {}}

    # Split text by segment headers
    segment_pattern = r'\[(Whole|Start|Mid|End) Segment\](.*?)(?=\[(?:Whole|Start|Mid|End) Segment\]|$)'
    segment_matches = re.findall(segment_pattern, text, re.DOTALL)

    for segment_name, segment_content in segment_matches:
        segments[segment_name.lower()] = segment_content.strip()

    return segments


class RAGActivityClassifier:
    """
    RAG-based classifier using hybrid search and LLM.

    Architecture:
    1. Extract temporal segments (whole, start, mid, end)
    2. Generate embeddings for each segment
    3. Hybrid search in Milvus with multiple ANN requests
    4. LLM-based classification using retrieved samples
    5. Track RAG quality metrics
    """

    def __init__(self, provider, model: str = "gpt-4o-mini", fewshot: int = 30,
                 out_fewshot: int = 20):
        """
        Initialize RAG classifier.

        Args:
            provider: DatasetProvider instance
            model: LLM model name for classification
            fewshot: Number of samples to retrieve per segment
            out_fewshot: Final number of samples after reranking
        """
        self.provider = provider
        self.config = provider.config
        self.dataset_name = provider.dataset_name
        self.model = model
        self.fewshot = fewshot
        self.out_fewshot = out_fewshot

        # Initialize OpenAI
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.embeddings = OpenAIEmbeddings(
            api_key=self.openai_api_key,
            model="text-embedding-3-small"
        )
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # Initialize Milvus
        milvus_uri = os.environ.get("ZILLIZ_CLOUD_URI")
        milvus_token = os.environ.get("ZILLIZ_CLOUD_API_KEY")

        if not milvus_uri or not milvus_token:
            raise ValueError(
                "ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_API_KEY environment variables must be set"
            )

        self.milvus_client = MilvusClient(uri=milvus_uri, token=milvus_token)
        self.collection_name = f"{self.dataset_name}_har_collection"

        # Get valid activity labels from config
        self.valid_labels = self.config['data_source']['activities']

        print(f"Initialized RAG Classifier for dataset: {self.dataset_name}")
        print(f"Collection: {self.collection_name}")
        print(f"Valid labels: {self.valid_labels}")
        print(f"LLM Model: {self.model}")
        print(f"Retrieval: {self.fewshot} per segment → {self.out_fewshot} final samples")

    def classify_window(self, window_file: str) -> Dict:
        """
        Classify a single window using RAG approach.

        Args:
            window_file: Path to window description file

        Returns:
            Dict with prediction results and metadata
        """
        # Extract window metadata from filename
        base = os.path.basename(window_file)
        m = re.match(r"window_(\d+)_activity_([A-Za-z0-9_]+)_stats\.txt", base)
        if not m:
            raise ValueError(f"Filename not matched: {base}")

        window_id, activity = m.groups()
        true_label = activity

        # Read window description
        with open(window_file, "r") as f:
            content = f.read()

        # Extract temporal segments
        segments = extract_sensor_sections(content)
        whole_stats = segments['whole']
        start_stats = segments['start']
        mid_stats = segments['mid']
        end_stats = segments['end']

        # Generate embeddings for each segment
        stats_emb = self.embeddings.embed_query(str(whole_stats))
        start_stats_emb = self.embeddings.embed_query(str(start_stats))
        mid_stats_emb = self.embeddings.embed_query(str(mid_stats))
        end_stats_emb = self.embeddings.embed_query(str(end_stats))

        # Create ANN search requests for each segment
        req_1 = AnnSearchRequest(
            anns_field="activity_stats",
            data=[stats_emb],
            limit=self.fewshot,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        )
        req_2 = AnnSearchRequest(
            anns_field="activity_stats_start",
            data=[start_stats_emb],
            limit=self.fewshot,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        )
        req_3 = AnnSearchRequest(
            anns_field="activity_stats_mid",
            data=[mid_stats_emb],
            limit=self.fewshot,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        )
        req_4 = AnnSearchRequest(
            anns_field="activity_stats_end",
            data=[end_stats_emb],
            limit=self.fewshot,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        )

        # Hybrid search with weighted ranker
        # Weight (1.0, 0.0, 0.0, 0.0) means only whole segment is used for ranking
        docs = self.milvus_client.hybrid_search(
            collection_name=self.collection_name,
            output_fields=["text", "timeseries_metadata", "stats_whole_text",
                         "stats_start_text", "stats_mid_text", "stats_end_text"],
            reqs=[req_1, req_2, req_3, req_4],
            limit=self.out_fewshot,
            ranker=WeightedRanker(1.0, 0.0, 0.0, 0.0)
        )

        # Process retrieved documents
        retrieved_labels = []
        sections = []
        for doc in docs:
            for hit in doc:
                entity = hit.entity
                whole_data = entity["stats_whole_text"]

                # Extract activity label from metadata
                # Handle different metadata structures
                metadata = entity.get('timeseries_metadata', {})
                if isinstance(metadata, dict):
                    sample_label = metadata.get('activity_id') or metadata.get('activity', 'unknown')
                else:
                    sample_label = 'unknown'

                retrieved_labels.append(sample_label)
                sections.append(
                    f"Activity Label: {sample_label}\n\n"
                    f"Stats for Whole Segment:\n{whole_data}\n"
                    f"Stats for Start Segment:\n{entity['stats_start_text']}\n"
                    f"Stats for Mid Segment:\n{entity['stats_mid_text']}\n"
                    f"Stats for End Segment:\n{entity['stats_end_text']}\n"
                )

        # Check if true label appears in retrieved samples (RAG quality metric)
        rag_hit = true_label in retrieved_labels

        # Construct prompt for LLM
        retrieved_data = "\n\n".join(sections)
        classes_str = str(self.valid_labels)

        system_prompt = f"""Use semantic similarity to compare the candidate statistics with the retrieved samples and output the activity label that maximizes similarity; respond with only the class label from {classes_str} and nothing else."""

        series = (
            f"Stats for Whole Segment:\n{whole_stats}\n"
            f"Stats for Start Segment:\n{start_stats}\n"
            f"Stats for Mid Segment:\n{mid_stats}\n"
            f"Stats for End Segment:\n{end_stats}\n"
        )

        user_prompt = f"""You are given summary statistics for sensor data across temporal segments for labeled samples and one unlabeled candidate.\n\n--- CANDIDATE ---\nTime Series:\n{series}\n\n--- LABELED SAMPLES ---\n{retrieved_data}\n"""

        # Call LLM with retry logic
        success = False
        while not success:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                prediction = response.choices[0].message.content.strip()
                success = True
            except openai.RateLimitError:
                print("Rate limit reached. Waiting 65 seconds...")
                time.sleep(65)
            except Exception as e:
                print(f"OpenAI API error: {e}. Waiting 10 seconds...")
                time.sleep(10)

        return {
            'window_id': window_id,
            'activity': activity,
            'true_label': true_label,
            'prediction': prediction,
            'rag_hit': rag_hit,
            'retrieved_labels': list(set(retrieved_labels)),
            'num_retrieved': len(retrieved_labels)
        }

    def evaluate(self, test_descriptions_dir: str, max_samples: int = None) -> Dict:
        """
        Evaluate classifier on test set.

        Args:
            test_descriptions_dir: Directory containing test description files
            max_samples: Maximum number of samples to evaluate (None = all)

        Returns:
            Dict with evaluation metrics and detailed results
        """
        # Get test files
        file_list = glob.glob(os.path.join(test_descriptions_dir, "*.txt"))

        if not file_list:
            raise ValueError(f"No test files found in {test_descriptions_dir}")

        # Shuffle for random sampling
        random.seed(42)
        random.shuffle(file_list)

        if max_samples:
            file_list = file_list[:max_samples]

        print(f"\nEvaluating on {len(file_list)} test samples...")
        print(f"Test descriptions: {test_descriptions_dir}")

        # Track results
        labels = []
        predictions = []
        rag_hit_rates = []
        all_results = []

        # Process each file
        for idx, file_path in enumerate(tqdm(file_list, desc="Classifying"), 1):
            try:
                result = self.classify_window(file_path)

                labels.append(result['true_label'])
                predictions.append(result['prediction'])
                rag_hit_rates.append(result['rag_hit'])
                all_results.append({
                    'file': os.path.basename(file_path),
                    **result
                })

                # Print progress
                if idx % 10 == 0:
                    current_acc = len([p for p, t in zip(predictions, labels) if p == t]) / len(labels)
                    current_rag_hit_rate = sum(rag_hit_rates) / len(rag_hit_rates) * 100
                    print(f"\nIteration {idx}/{len(file_list)}: "
                          f"Accuracy = {current_acc:.4f}, "
                          f"RAG Hit Rate = {current_rag_hit_rate:.1f}%")

            except Exception as e:
                print(f"\nError processing {file_path}: {e}")
                continue

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        rag_hit_rate = sum(rag_hit_rates) / len(rag_hit_rates) * 100 if rag_hit_rates else 0

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'rag_hit_rate': rag_hit_rate,
            'total_samples': len(labels),
            'labels': labels,
            'predictions': predictions,
            'detailed_results': all_results
        }


def main():
    parser = argparse.ArgumentParser(
        description="RAG-based Activity Classification with Hybrid Search"
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to dataset configuration YAML file')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='LLM model for classification (default: gpt-4o-mini)')
    parser.add_argument('--fewshot', type=int, default=30,
                       help='Number of samples to retrieve per segment (default: 30)')
    parser.add_argument('--out-fewshot', type=int, default=20,
                       help='Final number of samples after reranking (default: 20)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of test samples to evaluate (default: all)')

    args = parser.parse_args()

    # Load dataset provider
    provider = get_provider(args.config)
    dataset_name = provider.dataset_name

    # Auto-generated paths
    test_descriptions_dir = f"output/{dataset_name}/features/descriptions"
    output_dir = f"output/{dataset_name}/evaluation"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("RAG-BASED ACTIVITY CLASSIFICATION")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Test descriptions: {test_descriptions_dir}")
    print(f"Output directory: {output_dir}")
    print(f"LLM Model: {args.model}")
    print(f"Retrieval: {args.fewshot} per segment → {args.out_fewshot} final")
    print()

    # Initialize classifier
    classifier = RAGActivityClassifier(
        provider=provider,
        model=args.model,
        fewshot=args.fewshot,
        out_fewshot=args.out_fewshot
    )

    # Evaluate
    start_time = time.time()
    results = classifier.evaluate(
        test_descriptions_dir=test_descriptions_dir,
        max_samples=args.max_samples
    )
    end_time = time.time()

    # Save detailed results
    results_df = pd.DataFrame(results['detailed_results'])
    detailed_results_path = f'{output_dir}/detailed_results.csv'
    results_df.to_csv(detailed_results_path, index=False)

    # Save predictions
    predictions_df = pd.DataFrame({
        'label': results['labels'],
        'prediction': results['predictions']
    })
    predictions_path = f'{output_dir}/predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)

    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"RAG Hit Rate: {results['rag_hit_rate']:.1f}% "
          f"(true label in retrieved examples)")
    print(f"\nElapsed time: {end_time - start_time:.1f} seconds")
    print(f"\nResults saved to:")
    print(f"  - Predictions: {predictions_path}")
    print(f"  - Detailed: {detailed_results_path}")
    print("="*80)


if __name__ == "__main__":
    main()
