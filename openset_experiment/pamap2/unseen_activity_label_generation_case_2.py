"""
Open-set experiment for PAMAP2 dataset: Multiple unseen activities testing
Tests multiple activities together as unknown, generates new labels, and finds semantic matches
"""
import glob
import os
import random
import asyncio
import time
from collections import Counter

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Import shared modules
from shared_config import (
    ACTIVITY_MAPPING, ID_TO_SUPERCLASS, SUPERCLASS_MAPPING,
    MILVUS_URI, MILVUS_TOKEN, OPENAI_EMBEDDING_MODEL,
    get_collection_name, get_base_dir
)
from shared_models import ActivityClassification, KnownUnknownClassification
from shared_utils import (
    extract_sensor_sections,
    get_samples_from_descriptions_directly,
    get_samples_from_all_classes_async,
    openai_api_call_with_retry
)
from label_generation_utils import (
    get_semantic_matching_activity,
    print_aggregate_results
)

load_dotenv()


async def classify_known_unknown(client, model, user_prompt, classes_text):
    """
    Initial LLM call to determine if the activity is known or unknown based on retrieved samples
    """
    system_prompt = f"""
    You are an open-set activity classifier: build a feature vector from the candidate's per-sensor per-axis statistics across Whole/Start/Mid/End, compare to the retrieved labeled samples using cosine similarity, aggregate similarities by class, and return the class with the highest score only if its confidence exceeds a conservative data-driven threshold; otherwise return unknown; output only the chosen label or unknown.

    CLASSES:
    {classes_text}

    """

    response = openai_api_call_with_retry(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=KnownUnknownClassification
    )

    classification = response.choices[0].message.parsed
    print(f"Known/Unknown classification result: is_unknown={classification.is_unknown}")
    return classification


async def process_multiple_activities(
    TEST_ACTIVITIES, model, fewshot, window_count, use_rag_extraction,
    embeddings, client, milvus_client, collection_name, output_dir
):
    """Process multiple test activities together and return results summary"""
    base_dir = get_base_dir()

    # Create combined activity identifier for output
    activity_ids_str = "_".join(map(str, TEST_ACTIVITIES))
    activity_names = [ACTIVITY_MAPPING.get(act, 'unknown') for act in TEST_ACTIVITIES]
    activity_names_str = "_".join(activity_names)

    extraction_method = "rag" if use_rag_extraction else "direct"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tracking variables
    labels = []
    predictions = []
    is_unknown_predictions = []
    llm_predicted_labels = []

    # Collect windows from all test activities
    all_windows = []
    activity_window_mapping = {}

    for TEST_ACTIVITY in TEST_ACTIVITIES:
        parent_directory_with_descriptions = f'{base_dir}/activity{TEST_ACTIVITY}_{ACTIVITY_MAPPING.get(TEST_ACTIVITY)}'
        all_windows_raw = glob.glob(os.path.join(parent_directory_with_descriptions, "*_stat.txt"))
        random.seed(42)
        random.shuffle(all_windows_raw)
        activity_windows = all_windows_raw[:window_count]

        # Add to combined list and track source activity
        for window in activity_windows:
            all_windows.append(window)
            activity_window_mapping[window] = TEST_ACTIVITY

        print(f"Collected {len(activity_windows)} windows from activity {TEST_ACTIVITY} ({ACTIVITY_MAPPING.get(TEST_ACTIVITY, 'unknown')})")

    # Shuffle the combined windows to mix activities
    random.seed(42)
    random.shuffle(all_windows)
    total_samples = len(all_windows)

    if total_samples == 0:
        print(f"Warning: No windows found for activities {TEST_ACTIVITIES}, skipping...")
        return None

    # Get the superclasses of the test activities
    test_superclasses = set()
    for TEST_ACTIVITY in TEST_ACTIVITIES:
        test_superclass = ID_TO_SUPERCLASS.get(TEST_ACTIVITY)
        if test_superclass:
            test_superclasses.add(test_superclass)

    if test_superclasses:
        # Get all class IDs from the same superclasses, excluding the test activities
        all_class_ids = []
        for superclass in test_superclasses:
            same_superclass_ids = SUPERCLASS_MAPPING.get(superclass, [])
            for class_id in same_superclass_ids:
                if class_id not in TEST_ACTIVITIES and str(class_id) not in all_class_ids:
                    all_class_ids.append(str(class_id))

        print(f"Test activities: {TEST_ACTIVITIES} - {[ACTIVITY_MAPPING.get(act, 'unknown') for act in TEST_ACTIVITIES]}")
        print(f"Test activities belong to superclasses: {list(test_superclasses)}")
        print(f"Using class IDs from same superclasses: {all_class_ids}")
    else:
        all_class_ids = [str(i) for i in range(1, 13) if i not in TEST_ACTIVITIES]
        print(f"Warning: Superclass not found for test activities {TEST_ACTIVITIES}, using all classes")
        print(f"Test activities: {TEST_ACTIVITIES} - {[ACTIVITY_MAPPING.get(act, 'unknown') for act in TEST_ACTIVITIES]}")
        print(f"Using reference class IDs: {all_class_ids}")

    samples_per_class = 2

    # Pre-load descriptions if not using RAG
    if not use_rag_extraction:
        print("Using direct description extraction...")
        all_selected_docs = get_samples_from_descriptions_directly(
            all_class_ids, samples_per_class, base_dir, TEST_ACTIVITIES
        )

        print(f"Pre-loaded {len(all_selected_docs)} documents for direct extraction")

        # Pre-process the sections for reuse
        retrieved_labels = []
        sections = []

        for doc in all_selected_docs:
            whole_data = doc["whole_stats"]
            start_data = doc["start_stats"]
            mid_data = doc["mid_stats"]
            end_data = doc["end_stats"]

            sample_label = doc['class_id']
            retrieved_labels.append(sample_label)
            activity_label = ACTIVITY_MAPPING.get(int(sample_label))

            sections.append(
                f"Activity Label: {activity_label}"
                f"\n[Whole Segment]\n"
                f"\n{whole_data}\n"
                f"[Start Segment]\n"
                f"\n{start_data}\n"
                f"[Mid Segment]\n"
                f"\n{mid_data}\n"
                f"[End Segment]\n"
                f"\n{end_data}\n"
            )

        retrieved_data = "\n\n".join(sections)

    for idx, file_path in enumerate(tqdm(all_windows, desc=f"Processing Activities {activity_ids_str}"), 1):
        try:
            # Parse activity_id, subject, window_x from path
            filename = file_path.replace("\\", "/").split("/")[5]
            parts = filename.replace(".txt", "").split("_")
            activity_id = parts[0].replace("activity", "")
            true_label = ACTIVITY_MAPPING.get(int(activity_id))
            window_name = parts[-2]
            subject_id = parts[-3]

            print(f"\n[{idx}/{total_samples}] Processing: Activity={activity_id}-{ACTIVITY_MAPPING.get(int(activity_id))}, Subject={subject_id}, Window={window_name}")

            # Load Description
            with open(file_path, "r") as f:
                content = f.read()

            print(f"Extracting stat descriptions for activity:{activity_id} subject:{subject_id} window:{window_name}")
            sensors = extract_sensor_sections(content)
            whole_stats = sensors['whole']
            start_stats = sensors['start']
            mid_stats = sensors['mid']
            end_stats = sensors['end']

            # RETRIEVAL
            if use_rag_extraction:
                # Convert to embeddings for all segments
                stats_emb = embeddings.embed_query(str(whole_stats))
                start_stats_emb = embeddings.embed_query(str(start_stats))
                mid_stats_emb = embeddings.embed_query(str(mid_stats))
                end_stats_emb = embeddings.embed_query(str(end_stats))

                print("Using RAG extraction...")
                all_selected_docs = await get_samples_from_all_classes_async(
                    milvus_client, collection_name, all_class_ids,
                    stats_emb, start_stats_emb, mid_stats_emb, end_stats_emb,
                    samples_per_class
                )

                print(f"Retrieved {len(all_selected_docs)} documents for activity {activity_id}")

                retrieved_labels = []
                sections = []

                for hit in all_selected_docs:
                    entity = hit.entity
                    whole_data = entity["stats_whole_text"]

                    sample_label = entity['timeseries_metadata']['activity_id']
                    retrieved_labels.append(sample_label)
                    activity_label = ACTIVITY_MAPPING.get(int(sample_label))

                    sections.append(
                        f"Activity: {activity_label}\n\n"
                        f"\n{whole_data}\n"
                    )

                retrieved_data = "\n\n".join(sections)
            else:
                print(f"Using pre-loaded direct extraction data ({len(retrieved_labels)} samples)")

            retrieved_label_names = [f"{label}:{ACTIVITY_MAPPING.get(int(label), 'unknown')}" for label in set(retrieved_labels)]
            print(f"Retrieved labels: {retrieved_label_names}")

            series = f"""--- REFERENCE SAMPLES  ---\n{retrieved_data}"""

            user_prompt = f"""
                --- CANDIDATE ---
                \n[Whole Segment]\n
                {whole_stats}\n
                [Start Segment]\n
                {start_stats}\n
                [Mid Segment]\n
                {mid_stats}\n
                [End Segment]\n
                {end_stats}\n
                \n\n
                {series}
            """

            # STEP 1: KNOWN/UNKNOWN CLASSIFICATION
            print("\n--- STEP 1: Known/Unknown Classification ---")
            print("skipping known/unknown classification, setting all to unknown for testing...")
            is_unknown = True
            is_unknown_predictions.append(is_unknown)

            # STEP 2: ACTIVITY CLASSIFICATION (only if unknown)
            if True:
                print("\n--- STEP 2: Activity Classification (Unknown Activity Detected) ---")

                system_prompt = f"""You are an expert that can recognize human activities. Your input consists of tri-site sensor readings of an accelerometer, gyroscope and magnetometer placed on subject's chest, right wrist and left ankle that have been summarized into statistical features across temporal segments.

IMPORTANT: The initial analysis has determined that candidate activity is UNKNOWN (not matching any reference samples). Your task is to create a new, appropriate activity label.

TASK
1) Find a realistic activity class which would match with given sample.
2) Use the statical features of the reference samples and your domain knowledge on human activities to find the new activity label.
3) When finding a new label follow CRITICAL GUIDELINES FOR LABEL GENERATION:
4) I repeat, DO NOT RELY ONLY ON REFERENCED SAMPLES ONLY, USE YOUR DOMAIN KNOWLEDGE to create the activity label.
5) CRITICAL: Keep in mind that candidate activity doesn't belong to any of the reference classes.


CRITICAL GUIDELINES FOR LABEL GENERATION:
✓ DO: Generate labels for whole-body physical activities or postures
✓ DO: Use common, intuitive activity names (walking, sitting, jogging, etc.)
✓ DO: Consider both dynamic activities AND static postures
✓ DO: Base your decision primarily on sensor patterns

✗ DON'T: Generate labels for object interactions (holding phone, typing)
✗ DON'T: Use vague terms (resting, being active)
✗ DON'T: Generate compound activities (sitting-and-reading)
✗ DON'T: Focus on what hands are doing unless it affects whole-body movement



OUTPUT REQUIREMENTS:
Output must be a valid JSON object.
- The object must include:
  - `class_label`: assign a plausible new activity name.
  - `reasoning`: explain how you compared the candidate statistics with other classes, why you eliminated alternatives, and why the chosen class is the best match.

"""

                response = openai_api_call_with_retry(
                    client=client,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=ActivityClassification
                )

                classification = response.choices[0].message.parsed
                pred = classification.model_dump_json(indent=2)
                pred_reasoning = classification.reasoning
                predicted_class_label_from_llm = classification.class_label
                print(f"Received prediction from OpenAI API: \n{pred}")

                # STEP 3: Find semantically similar activity
                print("\n--- STEP 3:  Semantic Matching (OpenAI) ---")
                predicted_class_label = get_semantic_matching_activity(
                    predicted_class_label_from_llm, pred_reasoning, client, model
                )
                print(f"Semantically Similar Activity from LLM: {predicted_class_label}")
            else:
                predicted_class_label_from_llm = "N/A"
                predicted_class_label = "known_activity"
                pred = "N/A - Classified as known activity"

            # Store results
            predictions.append(predicted_class_label.lower())
            labels.append(true_label)
            llm_predicted_labels.append(predicted_class_label_from_llm)

            # Calculate running accuracy
            current_binary_accuracy = sum(1 for pred in is_unknown_predictions if pred == True) / len(is_unknown_predictions) if is_unknown_predictions else 0
            current_acc = len([p for p, t in zip(predictions, labels) if p == t]) / len(labels)
            current_f1 = f1_score(labels, predictions, average='macro')
            print(f"Iteration {idx}/{total_samples}: Binary Acc={current_binary_accuracy:.4f} Multi Class Acc={current_acc:.4f}")
            print("-" * 80)

            # Save results to file
            result_filename = f"{output_dir}/result_{activity_id}_{subject_id}_{window_name}.txt"
            with open(result_filename, "w") as f:
                f.write(f"=== CLASSIFICATION RESULT ===\n")
                f.write(f"Extraction Method: {'RAG' if use_rag_extraction else 'Direct Description'}\n")
                f.write(f"Activity ID: {activity_id}\n")
                f.write(f"Activity: {ACTIVITY_MAPPING.get(int(activity_id), 'unknown')}\n")
                f.write(f"Subject ID: {subject_id}\n")
                f.write(f"Window: {window_name}\n")

                f.write(f"\n=== STEP 1: KNOWN/UNKNOWN CLASSIFICATION ===\n")
                f.write(f"Classification: {'UNKNOWN' if is_unknown else 'KNOWN'}\n")
                f.write(f"Binary Classification Correct: {'Yes' if (is_unknown) == True else 'No'}\n")

                f.write(f"\n=== STEP 2: ACTIVITY CLASSIFICATION ===\n")
                f.write(f"Executed: {'Yes' if is_unknown else 'No (classified as known)'}\n")
                f.write(f"LLM Predicted Label: {predicted_class_label_from_llm}\n")
                f.write(f"Semantically Similar Class Label: {predicted_class_label}\n")
                f.write(f"Multi-class Correct: {'Yes' if predicted_class_label == true_label else 'No'}\n")

                f.write(f"\nRETRIEVED LABELS ({len(retrieved_labels)} samples):\n")
                for i, label in enumerate(retrieved_labels, 1):
                    activity_name = ACTIVITY_MAPPING.get(int(label), 'unknown')
                    f.write(f"  {i}. {label}:{activity_name}\n")

                f.write(f"\nRETRIEVED LABELS DISTRIBUTION:\n")
                label_counts = Counter(retrieved_labels)
                for label, count in sorted(label_counts.items()):
                    activity_name = ACTIVITY_MAPPING.get(int(label), 'unknown')
                    f.write(f"  {label}:{activity_name}: {count} samples\n")

                f.write(f"\nLLM Response (Activity Classification): \n{pred}\n\n")

            print(f"Results saved to: {result_filename}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    if len(predictions) == 0:
        print(f"No predictions to process for activities {TEST_ACTIVITIES}")
        return None

    # Compute metrics
    binary_accuracy = sum(1 for pred in is_unknown_predictions if pred == True) / len(is_unknown_predictions) if is_unknown_predictions else 0
    binary_true_labels = [True] * len(is_unknown_predictions) if is_unknown_predictions else []
    binary_f1 = f1_score(binary_true_labels, is_unknown_predictions, zero_division=0) if is_unknown_predictions else 0

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    # Save results
    df = pd.DataFrame({
        'label': labels + [acc] + [binary_accuracy] + [binary_f1],
        'prediction': predictions + [f1] + ['binary_acc'] + ['binary_f1'],
        'llm_predicted_label': llm_predicted_labels + ['N/A'] + ['N/A'] + ['N/A']
    })
    results_path = f'{output_dir}/results_activity_classification.csv'
    df.to_csv(results_path, index=False)

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - Activities {activity_ids_str}")
    print(f"{'='*80}")
    print(f"Extraction Method: {'RAG' if use_rag_extraction else 'Direct Description'}")
    print(f"Test Activities: {TEST_ACTIVITIES} - {activity_names}")
    print(f"Total samples processed: {len(labels)}")
    print(f"Multi-class Accuracy: {acc:.4f}")
    print(f"Multi-class F1 Score: {f1:.4f}")
    print(f"Binary Accuracy (Known/Unknown): {binary_accuracy:.4f}")
    print(f"Binary F1 Score (Known/Unknown): {binary_f1:.4f}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*80}")

    return {
        'test_activities': TEST_ACTIVITIES,
        'activity_names': activity_names_str,
        'total_samples': len(labels),
        'multi_class_accuracy': acc,
        'multi_class_f1': f1,
        'binary_accuracy': binary_accuracy,
        'binary_f1': binary_f1,
        'extraction_method': extraction_method,
        'model': model,
        'fewshot': fewshot,
        'window_count': window_count
    }


async def main():
    # TEST ACTIVITY PAIRS
    TEST_ACTIVITY_PAIRS = [[1, 3], [1, 5], [4, 3], [4, 5], [3, 5], [1, 3, 4], [3, 4, 5], [1, 3, 5], [3, 4, 12]]

    # CONFIGURATION
    model = "gpt-5-mini"
    fewshot = 15
    window_count = 100
    use_rag_extraction = False

    # Initialize clients
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=OPENAI_EMBEDDING_MODEL)
    client = OpenAI(api_key=openai_api_key)
    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    collection_name = get_collection_name()

    # Summary results
    all_activities_summary = []

    # Process each test activity pair
    for TEST_ACTIVITIES in TEST_ACTIVITY_PAIRS:
        activity_names = [ACTIVITY_MAPPING.get(act, 'unknown') for act in TEST_ACTIVITIES]
        print(f"\n{'='*100}")
        print(f"PROCESSING TEST ACTIVITY PAIR: {TEST_ACTIVITIES} - {activity_names}")
        print(f"{'='*100}")

        activity_ids_str = "_".join(map(str, TEST_ACTIVITIES))
        extraction_method = "rag" if use_rag_extraction else "direct"
        output_dir = f'output/exp2-v2/{str(fewshot)}/{model}/{extraction_method}/activities_{activity_ids_str}'

        activity_results = await process_multiple_activities(
            TEST_ACTIVITIES, model, fewshot, window_count, use_rag_extraction,
            embeddings, client, milvus_client, collection_name, output_dir
        )

        if activity_results:
            all_activities_summary.append(activity_results)

        print(f"\nCompleted processing for activity pair {TEST_ACTIVITIES}")

    # Print aggregate results
    aggregate_output_dir = f'output/exp2-v2/{str(fewshot)}/{model}/{extraction_method}/aggregate_results'
    print_aggregate_results(all_activities_summary, model, fewshot, window_count, use_rag_extraction, aggregate_output_dir)


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()

    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Elapsed time: {end_time - start_time} seconds")
