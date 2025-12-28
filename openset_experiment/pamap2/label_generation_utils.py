"""
Utilities specific to label generation experiments
"""
from typing import Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

from shared_config import ACTIVITY_MAPPING


def get_pred_label(pred_label: str, mapping: Dict[int, str], device: str = "cuda:0") -> str:
    """
    Validate and match predicted label to known activities using semantic similarity.

    Args:
        pred_label: Predicted activity label from LLM
        mapping: Dictionary mapping activity IDs to activity names
        device: Device to run sentence transformer on

    Returns:
        Best matching activity label
    """
    # Normalize prediction
    norm_pred = pred_label.strip().lower().replace(" ", "_")

    # Step 1: Exact match
    for k, v in mapping.items():
        if norm_pred == v.lower():
            return norm_pred

    # Step 2: Compute Embedding Similarity
    print("Computing embedding similarity for prediction validation...")

    # Initialize sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Convert prediction to embedding
    pred_embedding = model.encode([pred_label])

    best_match = pred_label
    best_s = 0

    # Compare against each activity in the mapping
    for k, v in mapping.items():
        # Convert activity name to embedding using sentence transformer
        activity_embedding = model.encode([v])

        # Compute cosine similarity for sentence transformer
        similarity_score = util.cos_sim(pred_embedding, activity_embedding).item()

        if similarity_score > best_s:
            best_s = similarity_score
            best_match = v

        print(f"Sentence Transformer similarity between predicted '{pred_label}' and '{v}': {similarity_score:.4f}")

    return best_match


def get_semantic_matching_activity(
    created_label: str,
    pred_reasoning: str,
    openai_client,
    openai_model: str
) -> str:
    """
    Find the semantically most similar activity from known activities using LLM.

    Args:
        created_label: Newly created activity label
        pred_reasoning: Reasoning behind the label creation
        openai_client: OpenAI client instance
        openai_model: Model name to use

    Returns:
        Most semantically similar known activity label
    """
    system_prompt = """You are an expert in semantic similarity of human activities. Your task is to find the most semantically similar activity from a list of known activities.

    Given:
    1. A newly created activity label and the reasoning behind its creation
    2. A list of known activity labels

    Find which known activity is most semantically similar to the new label. Consider the physical movements, body posture, energy expenditure, and overall nature of the activities.
    Return only the most similar activity label and no extra text.
"""

    known_activities_list = list(ACTIVITY_MAPPING.values())

    user_prompt = f"""New activity label: {created_label}\n\n Reasoning: {pred_reasoning}\n\n

Known activities: {', '.join(known_activities_list)}
"""

    try:
        response = openai_client.beta.chat.completions.parse(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        result = response.choices[0].message.content
        return result
    except Exception as e:
        print(f"Error in semantic matching LLM call: {e}")
        return list(ACTIVITY_MAPPING.values())[0]  # Default to first activity


def print_aggregate_results(
    all_activities_summary,
    model: str,
    fewshot: int,
    window_count: int,
    use_rag_extraction: bool,
    output_dir: str = None
):
    """
    Print and save aggregate results across all activities/activity pairs.

    Args:
        all_activities_summary: List of dictionaries containing results for each activity/pair
        model: Model name used for classification
        fewshot: Number of few-shot samples used
        window_count: Number of windows processed per activity
        use_rag_extraction: Whether RAG extraction was used
        output_dir: Optional output directory for saving results
    """
    if not all_activities_summary:
        print("\nNo results to aggregate.")
        return

    # Determine if we're aggregating single activities or pairs
    is_pairs = 'test_activities' in all_activities_summary[0]

    print(f"\n{'='*100}")
    if is_pairs:
        print(f"AGGREGATE RESULTS ACROSS ALL ACTIVITY PAIRS")
    else:
        print(f"AGGREGATE RESULTS ACROSS ALL ACTIVITIES")
    print(f"{'='*100}")

    # Extract metrics for statistical analysis
    binary_accuracies = [result['binary_accuracy'] for result in all_activities_summary]
    binary_f1s = [result['binary_f1'] for result in all_activities_summary]
    multi_class_accuracies = [result['multi_class_accuracy'] for result in all_activities_summary]
    multi_class_f1s = [result['multi_class_f1'] for result in all_activities_summary]

    # Calculate means and standard deviations
    binary_acc_mean = np.mean(binary_accuracies)
    binary_acc_std = np.std(binary_accuracies)
    binary_f1_mean = np.mean(binary_f1s)
    binary_f1_std = np.std(binary_f1s)

    multi_acc_mean = np.mean(multi_class_accuracies)
    multi_acc_std = np.std(multi_class_accuracies)
    multi_f1_mean = np.mean(multi_class_f1s)
    multi_f1_std = np.std(multi_class_f1s)

    # Print summary statistics
    print(f"Configuration:")
    print(f"  Model: {model}")
    print(f"  Few-shot samples: {fewshot}")
    print(f"  Window count per activity: {window_count}")
    print(f"  Extraction method: {'RAG' if use_rag_extraction else 'Direct Description'}")
    if is_pairs:
        print(f"  Total activity pairs processed: {len(all_activities_summary)}")
    else:
        print(f"  Total activities processed: {len(all_activities_summary)}")

    print(f"\nBinary Classification (Known/Unknown Detection):")
    print(f"  Binary Accuracy: {binary_acc_mean:.4f} ± {binary_acc_std:.4f}")
    print(f"  Binary F1 Score: {binary_f1_mean:.4f} ± {binary_f1_std:.4f}")

    print(f"\nMulti-Class Classification:")
    print(f"  Classification Accuracy: {multi_acc_mean:.4f} ± {multi_acc_std:.4f}")
    print(f"  Classification F1 Score: {multi_f1_mean:.4f} ± {multi_f1_std:.4f}")

    # Print per-activity breakdown
    if is_pairs:
        print(f"\nPer-Activity Pair Results:")
        print(f"{'Activities':<20} {'Names':<30} {'Samples':<8} {'Bin_Acc':<8} {'Bin_F1':<8} {'MC_Acc':<8} {'MC_F1':<8}")
        print(f"{'-'*100}")
        for result in all_activities_summary:
            activities_str = str(result['test_activities'])
            print(f"{activities_str:<20} {result['activity_names']:<30} {result['total_samples']:<8} "
                  f"{result['binary_accuracy']:<8.4f} {result['binary_f1']:<8.4f} "
                  f"{result['multi_class_accuracy']:<8.4f} {result['multi_class_f1']:<8.4f}")
    else:
        print(f"\nPer-Activity Results:")
        print(f"{'Activity':<15} {'Name':<20} {'Samples':<8} {'Bin_Acc':<8} {'Bin_F1':<8} {'MC_Acc':<8} {'MC_F1':<8}")
        print(f"{'-'*80}")
        for result in all_activities_summary:
            print(f"{result['test_activity']:<15} {result['activity_name']:<20} {result['total_samples']:<8} "
                  f"{result['binary_accuracy']:<8.4f} {result['binary_f1']:<8.4f} "
                  f"{result['multi_class_accuracy']:<8.4f} {result['multi_class_f1']:<8.4f}")

    # Save aggregate results to CSV if output directory provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        summary_df = pd.DataFrame(all_activities_summary)
        if is_pairs:
            summary_csv_path = f'{output_dir}/all_activity_pairs_summary.csv'
        else:
            summary_csv_path = f'{output_dir}/all_activities_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)

        # Save aggregate statistics
        aggregate_stats = {
            'metric': ['Binary_Accuracy', 'Binary_F1', 'Classification_Accuracy', 'Classification_F1'],
            'mean': [binary_acc_mean, binary_f1_mean, multi_acc_mean, multi_f1_mean],
            'std': [binary_acc_std, binary_f1_std, multi_acc_std, multi_f1_std],
            'mean_plus_std': [f"{binary_acc_mean:.4f} ± {binary_acc_std:.4f}",
                              f"{binary_f1_mean:.4f} ± {binary_f1_std:.4f}",
                              f"{multi_acc_mean:.4f} ± {multi_acc_std:.4f}",
                              f"{multi_f1_mean:.4f} ± {multi_f1_std:.4f}"]
        }

        aggregate_df = pd.DataFrame(aggregate_stats)
        aggregate_csv_path = f'{output_dir}/aggregate_statistics.csv'
        aggregate_df.to_csv(aggregate_csv_path, index=False)

        print(f"\nAggregate results saved to:")
        print(f"  Summary: {summary_csv_path}")
        print(f"  Statistics: {aggregate_csv_path}")

    print(f"{'='*100}")
