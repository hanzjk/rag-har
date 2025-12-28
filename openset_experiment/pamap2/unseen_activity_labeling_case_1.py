"""
Closed-set experiment for PAMAP2 dataset: Standard classification with RAG
Standard activity classification where test activity is excluded from retrieval
"""
import glob
import os
import random
import time
from collections import Counter

import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Import shared modules
from shared_config import (
    ACTIVITY_MAPPING,
    MILVUS_URI, MILVUS_TOKEN, OPENAI_EMBEDDING_MODEL,
    get_collection_name, get_base_dir
)
from shared_models import StandardActivityClassification
from shared_utils import extract_sensor_sections, openai_api_call_with_retry

load_dotenv()


def main():
    # TEST ACTIVITIES
    TEST_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

    # CONFIGURATION
    model = "gpt-5-mini"
    fewshot = 15
    window_count = 100

    # Initialize clients
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=OPENAI_EMBEDDING_MODEL)
    client = OpenAI(api_key=openai_api_key)
    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    base_dir = get_base_dir()
    collection_name = get_collection_name()

    # Track overall statistics
    all_accuracies = []
    all_f1_scores = []

    # Process each test activity
    for TEST_ACTIVITY in TEST_ACTIVITIES:
        print(f"\n{'='*100}")
        print(f"PROCESSING TEST ACTIVITY: {TEST_ACTIVITY} - {ACTIVITY_MAPPING.get(TEST_ACTIVITY, 'unknown')}")
        print(f"{'='*100}")

        try:
            # Activity-specific directories
            parent_directory_with_descriptions = f'../{base_dir}/activity{TEST_ACTIVITY}_{ACTIVITY_MAPPING.get(TEST_ACTIVITY)}'
            output_dir = f'output/exp1-v1/{str(fewshot)}/{model}/activity_{TEST_ACTIVITY}'
            os.makedirs(output_dir, exist_ok=True)

            # Initialize tracking variables
            labels = []
            predictions = []

            all_windows_raw = glob.glob(os.path.join(parent_directory_with_descriptions, "*_stat.txt"))
            if not all_windows_raw:
                print(f"No windows found for activity {TEST_ACTIVITY}, skipping...")
                continue

            random.seed(42)
            random.shuffle(all_windows_raw)
            all_windows = all_windows_raw[:window_count]
            total_samples = len(all_windows)

            print(f"Found {len(all_windows_raw)} windows, processing {total_samples} windows for activity {TEST_ACTIVITY}")

            for idx, file_path in enumerate(tqdm(all_windows, desc=f"Processing activity {TEST_ACTIVITY}"), 1):
                try:
                    # Parse activity_id, subject, window_x from path
                    filename = file_path.replace("\\", "/").split("/")[5]
                    parts = filename.replace(".txt", "").split("_")
                    activity_id = parts[0].replace("activity", "")
                    true_label = int(activity_id)
                    true_activity_name = ACTIVITY_MAPPING.get(true_label)
                    window_name = parts[-2]
                    subject_id = parts[-3]

                    if true_label != TEST_ACTIVITY:
                        print(f"Skipping file {file_path} as it does not match TEST_ACTIVITY {TEST_ACTIVITY}")
                        continue

                    print(f"\n[{idx}/{total_samples}] Processing: Activity={activity_id}-{ACTIVITY_MAPPING.get(int(activity_id))}")
                    print(f"True label: {true_label}-{true_activity_name}")

                    # Load Description
                    with open(file_path, "r") as f:
                        content = f.read()

                    print(f"Extracting stat descriptions for activity:{activity_id}")
                    sensors = extract_sensor_sections(content)
                    whole_stats = sensors['whole']
                    start_stats = sensors['start']
                    mid_stats = sensors['mid']
                    end_stats = sensors['end']

                    # Convert to embeddings for all segments
                    stats_emb = embeddings.embed_query(str(whole_stats))
                    start_stats_emb = embeddings.embed_query(str(start_stats))
                    mid_stats_emb = embeddings.embed_query(str(mid_stats))
                    end_stats_emb = embeddings.embed_query(str(end_stats))

                    # Retrieve samples excluding test activity
                    expr = f'timeseries_metadata["activity_id"] != "{TEST_ACTIVITY}"'
                    req_1 = AnnSearchRequest(
                        anns_field="activity_stats",
                        data=[stats_emb],
                        limit=fewshot,
                        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                        expr=expr
                    )
                    req_2 = AnnSearchRequest(
                        anns_field="activity_stats_start",
                        data=[start_stats_emb],
                        limit=fewshot,
                        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                        expr=expr
                    )
                    req_3 = AnnSearchRequest(
                        anns_field="activity_stats_mid",
                        data=[mid_stats_emb],
                        limit=fewshot,
                        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                        expr=expr
                    )
                    req_4 = AnnSearchRequest(
                        anns_field="activity_stats_end",
                        data=[end_stats_emb],
                        limit=fewshot,
                        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                        expr=expr
                    )

                    docs = milvus_client.hybrid_search(
                        collection_name=collection_name,
                        output_fields=["text", "timeseries_metadata", "stats_whole_text",
                                       "stats_start_text", "stats_mid_text", "stats_end_text"],
                        reqs=[req_1, req_2, req_3, req_4],
                        limit=fewshot,
                        ranker=WeightedRanker(0.25, 0.25, 0.25, 0.25)
                    )

                    total_hits = sum(len(hits) for hits in docs)
                    print(f"Retrieved {total_hits} documents for activity {activity_id}")

                    # Collect retrieved labels and sections
                    retrieved_labels = []
                    sections = []

                    for doc in docs:
                        for hit in doc:
                            entity = hit.entity
                            whole_data = entity["stats_whole_text"]
                            start_data = entity["stats_start_text"]
                            mid_data = entity["stats_mid_text"]
                            end_data = entity["stats_end_text"]

                            sample_label = entity['timeseries_metadata']['activity_id']
                            retrieved_labels.append(sample_label)

                            sections.append(
                                f"Activity: {ACTIVITY_MAPPING.get(int(sample_label))}\n\n"
                                f"[Whole Segment]\n"
                                f"\n{whole_data}\n"
                                f"[Start Segment]\n"
                                f"\n{start_data}\n"
                                f"[Mid Segment]\n"
                                f"\n{mid_data}\n"
                                f"[End Segment]\n"
                                f"\n{end_data}"
                            )

                    retrieved_label_names = [f"{label}:{ACTIVITY_MAPPING.get(int(label), 'unknown')}" for label in set(retrieved_labels)]
                    print(f"Retrieved labels: {retrieved_label_names}")

                    retrieved_data = "\n\n".join(sections)

                    # Create classes list
                    classes_list = []
                    for class_id, class_name in ACTIVITY_MAPPING.items():
                        classes_list.append(f"{class_id}. {class_name}")

                    classes_text = "\n    ".join(classes_list)

                    # System prompt
                    system_prompt = f"""
You are a multi-class activity classifier analyzing sensor data statistics.

AVAILABLE CLASSES:
{classes_text}

INPUT DATA:
- CANDIDATE: An unlabeled sample to classify (statistical features from accelerometer, gyroscope, magnetometer at chest, wrist, ankle across temporal segments)
- REFERENCE SAMPLES: Labeled samples that share SOME statistical similarities with the candidate but ARE FROM DIFFERENT ACTIVITIES

CRITICAL INSTRUCTIONS:
1. Classify the CANDIDATE based on its statistical patterns and your understanding of the activities into one of the AVAILABLE CLASSES.
2. IF YOUR OUTPUT MATCHES ANY OF THE REFERENCES, YOU HAVE FAILED
3. When doing the classification, consider all sensor modalities (accelerometer, gyroscope, magnetometer) across all body parts (chest, wrist, ankle) according to AXIS INTERPRETATION GUIDE.
4. The reference samples are DECOYS - they show activities that have SOME similar patterns but are NOT the correct class
5. You MUST choose from the AVAILABLE CLASSES list above, NOT from reference sample labels

OUTPUT FORMAT (valid JSON required):
{{
    "label": "<must be from AVAILABLE CLASSES list>",
    "reasoning": "<explain sensor pattern analysis and elimination process>"
}}

Remember: Reference samples are similar but WRONG - use them only to understand the data format, not for classification decisions.

"""
                    series = (
                        f"[Whole Segment]\n"
                        f"\n{whole_stats}\n"
                        f"[Start Segment]\n"
                        f"\n{start_stats}\n"
                        f"[Mid Segment]\n"
                        f"\n{mid_stats}\n"
                        f"[End Segment]\n"
                        f"\n{end_stats}"
                    )

                    user_prompt = f"""
                            --- CANDIDATE ---
                            {series}

                            --- REFERENCE SAMPLES ---
                            {retrieved_data}
                    """

                    response = openai_api_call_with_retry(
                        client=client,
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format=StandardActivityClassification
                    )

                    classification = response.choices[0].message.parsed
                    pred = classification.model_dump_json(indent=2)
                    predicted_label = classification.label
                    print(f"Received prediction from OpenAI API: \n{pred}")
                    print(f"Extracted class_label: {predicted_label}")

                    # Store results
                    predictions.append(predicted_label)
                    labels.append(true_activity_name)

                    # Calculate running accuracy
                    current_acc = len([p for p, t in zip(predictions, labels) if p == t]) / len(labels)

                    print(f"Iteration {idx}/{total_samples}: Accuracy = {current_acc:.4f}")
                    print("-" * 80)

                    # Save results to file
                    result_filename = f"{output_dir}/result_{activity_id}_{subject_id}_{window_name}.txt"
                    with open(result_filename, "w") as f:
                        f.write(f"=== CLASSIFICATION RESULT ===\n")
                        f.write(f"Activity ID: {activity_id}\n")
                        f.write(f"Activity: {ACTIVITY_MAPPING.get(int(activity_id), 'unknown')}\n")
                        f.write(f"Subject ID: {subject_id}\n")
                        f.write(f"Window: {window_name}\n")
                        f.write(f"Predicted Class Label: {predicted_label}\n")
                        f.write(f"Correct: {'Yes' if predicted_label == true_activity_name else 'No'}\n")

                        f.write(f"RETRIEVED LABELS ({len(retrieved_labels)} samples):\n")
                        for i, label in enumerate(retrieved_labels, 1):
                            activity_name = ACTIVITY_MAPPING.get(int(label), 'unknown')
                            f.write(f"  {i}. {label}:{activity_name}\n")

                        f.write(f"\nRETRIEVED LABELS DISTRIBUTION:\n")
                        label_counts = Counter(retrieved_labels)
                        for label, count in sorted(label_counts.items()):
                            activity_name = ACTIVITY_MAPPING.get(int(label), 'unknown')
                            f.write(f"  {label}:{activity_name}: {count} samples\n")

                        f.write(f"\nLLM Response: \n{pred}\n\n")

                    print(f"Results saved to: {result_filename}")

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue

            if len(predictions) == 0:
                print(f"No predictions to process for activity {TEST_ACTIVITY}")
                continue

            # Calculate final results
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

            df = pd.DataFrame({'label': labels + [acc], 'prediction': predictions + [f1]})
            df.to_csv(f'{output_dir}/results_activity_classification.csv', index=False)

            print(f"\n{'='*80}")
            print(f"FINAL RESULTS - Activity {TEST_ACTIVITY}")
            print(f"{'='*80}")
            print(f"Total samples processed: {len(labels)}")
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Results saved to: {output_dir}/results_activity_classification.csv")
            print(f"{'='*80}")

            # Store scores
            all_accuracies.append(acc)
            all_f1_scores.append(f1)

        except Exception as e:
            print(f"Error processing activity {TEST_ACTIVITY}: {e}")
            continue

    # Compute overall statistics
    if all_accuracies and all_f1_scores:
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        mean_f1 = np.mean(all_f1_scores)
        std_f1 = np.std(all_f1_scores)

        print(f"\n{'='*100}")
        print(f"OVERALL STATISTICS ACROSS ALL ACTIVITIES")
        print(f"{'='*100}")
        print(f"Number of activities processed: {len(all_accuracies)}")
        print(f"Individual accuracies: {[f'{acc:.4f}' for acc in all_accuracies]}")
        print(f"Individual F1 scores: {[f'{f1:.4f}' for f1 in all_f1_scores]}")
        print(f"")
        print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"{'='*100}")

        # Save overall results
        summary_file = f'output/exp1-v1/{str(fewshot)}/{model}/overall_summary.txt'
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)

        with open(summary_file, 'w') as f:
            f.write(f"OVERALL EXPERIMENT SUMMARY\n")
            f.write(f"{'='*50}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Few-shot samples: {fewshot}\n")
            f.write(f"Window count per activity: {window_count}\n")
            f.write(f"Activities processed: {len(all_accuracies)}\n")
            f.write(f"\n")
            f.write(f"Individual Results:\n")
            for i, (acc, f1) in enumerate(zip(all_accuracies, all_f1_scores)):
                activity_id = TEST_ACTIVITIES[i] if i < len(TEST_ACTIVITIES) else i + 1
                activity_name = ACTIVITY_MAPPING.get(activity_id, 'unknown')
                f.write(f"  Activity {activity_id} ({activity_name}): Accuracy={acc:.4f}, F1={f1:.4f}\n")
            f.write(f"\n")
            f.write(f"Summary Statistics:\n")
            f.write(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
            f.write(f"  Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\n")

        print(f"Overall summary saved to: {summary_file}")
    else:
        print(f"\n{'='*100}")
        print(f"No results to summarize - no activities were successfully processed")
        print(f"{'='*100}")

    print(f"\n{'='*100}")
    print(f"ALL ACTIVITIES COMPLETED")
    print(f"{'='*100}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Elapsed time: {end_time - start_time} seconds")
