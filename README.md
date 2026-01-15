# RAG-HAR Pipeline

A dataset-agnostic Retrieval-Augmented Generation pipeline for Human Activity Recognition using vector similarity search.

---

## Prerequisites

### 1. OpenAI API Key

The pipeline uses OpenAI for embeddings and LLM-based classification.

**Get your API key:**
1. Sign up at [OpenAI Platform](https://platform.openai.com/)
2. Navigate to [API Keys](https://platform.openai.com/api-keys)
3. Create a new secret key


### 2. Milvus Vector Database (Zilliz Cloud)

The pipeline uses Milvus for vector storage and similarity search.

**Get free cloud instance:**
1. Sign up at [Zilliz Cloud](https://cloud.zilliz.com/signup)
2. Create a new cluster (free tier available)
3. Get credentials from cluster details page

---

## Supported Datasets

RAG-HAR is implemented for the following publicly available HAR datasets:

| Dataset | # Activities / Gestures | Sensors | Download |
|--------|--------------------------|---------|----------|
| **PAMAP2** | 12 activities | 3 IMUs placed on hand, chest, and ankle | [UCI Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) |
| **MHEALTH** | 12 activities | 3 IMUs placed on arm, ankle, and chest | [UCI Repository](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) |
| **USC-HAD** | 12 activities | Accelerometer and gyroscope | [USC](https://sipi.usc.edu/had/) |
| **HHAR** | 6 activities | Accelerometer and gyroscope (smartphones & smartwatches) | [UCI Repository](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) |
| **GOTOV** | 16 activities | 3 accelerometers placed on hand, chest, and ankle | [4TUResearch Data](https://data.4tu.nl/datasets/f9bae0cd-ec4e-4cfb-aaa5-41bd1c5554ce/1) |
| **Skoda** | 10 gestures | 20 accelerometers attached to worker’s arms | [Data Repository](https://drive.google.com/file/d/15Q8oV02h2_e94IWJ9rnKLrSCKPCTW5FS/view?usp=drive_link) |

**Note:** Download the datasets and update the `data_source` paths in the corresponding config files (`datasets/*.yaml`).

---

## Overview

This pipeline processes sensor data through four stages to enable RAG-based activity classification:

```
Raw Sensor Data (CSV)
    ↓
[Stage 1] Preprocessing → CSV Windows
    ↓
[Stage 2] Feature Extraction → descriptions/
    ↓
[Stage 3] Vector Indexing → Vector Database
    ↓
[Stage 4] Classification → Predictions + Evaluation
```

## Quick Start

**Note:** The following instructions use PAMAP2 as an example. For other datasets, simply change the config file path (e.g., `--config datasets/mhealth_config.yaml`).

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="sk-your-key-here"
export ZILLIZ_CLOUD_URI="https://your-cluster.api.gcp-us-west1.zillizcloud.com"
export ZILLIZ_CLOUD_API_KEY="your-token-here"
```

### 3. Download Dataset

Download PAMAP2 from [UCI Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) and update the data path in `datasets/pamap2_config.yaml`.

### 4. Run Pipeline Stages

Each stage is run independently using its own script.

---

## Stage 1: Preprocessing

**Purpose:** Dataset-specific preprocessing - each dataset implements its own logic.

**Script:** `preprocessing.py` (wrapper that calls provider's preprocess())

**Command:**

```bash
python preprocessing.py --config datasets/pamp2_config.yaml
```

**Parameters:**

- `--config`: Path to dataset configuration YAML file

**Auto-generated paths:**

- Output directory: `output/{dataset_name}/windows/`

**How it works:**

- The script calls `provider.preprocess(output_dir)`
- Each dataset provider implements its own `preprocess()` method
- Providers can use the `Preprocessor` utility class or implement completely custom logic

**Outputs:**

All datasets follow this **standardized structure**:

```
output/{dataset_name}/train-test-splits/
├── train/
│   ├── {activity_name}/
│   │   ├── subject{idx}_window{idx}_activity{idx}_{activity_label}.csv
│   │   ├── subject{idx}_window{idx}_activity{idx}_{activity_label}.csv
│   │   └── ...
│   └── {another_activity}/
│       └── ...
└── test/
    ├── {activity_name}/
    │   └── ...
    └── ...
```

**File naming convention:**
- Pattern: `subject{idx}_window{idx}_activity{idx}_{activity_label}.csv`
- Example: `subject101_window0000_activity1_walking.csv`

**What it does:**

1. Loads raw CSV files via DatasetProvider
2. Applies dataset-specific preprocessing (normalization, filtering, etc.)
3. Segments into overlapping windows (e.g., 200 samples with 50% overlap)
4. Splits into train/test sets
5. Organizes windows into activity-based folders following the standardized structure

---

## Stage 2: Feature Extraction

**Purpose:** Calculate statistical features for each window and generate human-readable descriptions.

**Script:** `generate_stats.py`

**Command:**

```bash
python generate_stats.py --config datasets/pamp2_config.yaml
```

**Parameters:**

- `--config`: Path to dataset configuration YAML file

**Auto-generated paths:**

- Train Input: `output/{dataset_name}/train-test-splits/train/`
- Test Input: `output/{dataset_name}/train-test-splits/test/`
- Train Output: `output/{dataset_name}/features/train/`
- Test Output: `output/{dataset_name}/features/test/`

**Outputs:**

```
output/{dataset_name}/features/
├── train/
│   └── descriptions/
│       ├── window_{idx}_activity_{activity_label}_stats.txt
│       ├── window_{idx}_activity_{activity_label}_stats.txt
│       └── ...
└── test/
    └── descriptions/
        └── ...
```

**Feature file naming:**
- Pattern: `window_{idx}_activity_{activity_label}_stats.txt` 
- Example: `window_0_activity_walking_stats.txt`

**What it does:**

1. For each window CSV, calculates statistical features (mean, std, min, max, median, etc.)
2. Computes features per sensor axis (x, y, z)
3. Segments window into temporal parts (whole, start, mid, end) for richer features
4. Generates human-readable text descriptions of the features

**Dataset-Specific Feature Extraction:**
Each dataset specific implementation knows how to extract features from its own column structure:

- **HAR Demo**: Handled by `providers/har_demo/features.py` (extracts from `accel_x`, `gyro_y`, `mag_z`)
- **Your dataset**: Create `providers/your_dataset/features.py` with your column names


---

## Stage 3: Vector Database Indexing

**Purpose:** Create embeddings from feature descriptions and index them into a vector database.

**Script:** `timeseries_indexing.py`

**Command:**

```bash
python timeseries_indexing.py --config datasets/pamp2_config.yaml
```

**Parameters:**

- `--config`: Path to dataset configuration YAML file

**Auto-generated paths:**

- Input: `output/{dataset_name}/features/descriptions/`
- Collection name: `{dataset_name}_collection`

**Outputs:**

- **Milvus:** Cloud storage

**What it does:**

1. Reads all window description text files
2. Generates embeddings for each description
3. Indexes embeddings into vector database with metadata (activity, window_id)
4. Creates similarity search index

---

## Stage 4: Classification & Evaluation

**Purpose:** RAG-based activity classification using hybrid search with temporal segmentation and LLM reasoning.

**Script:** `classifier.py`

**Command:**

```bash
python classifier.py --config datasets/pamp2_config.yaml
```

**Parameters:**

- `--config`: Path to dataset configuration YAML file (required)
- `--model`: [Optional] LLM model for classification (default: `gpt-5-mini`)
- `--fewshot`: [Optional] Number of samples to retrieve per temporal segment
- `--out-fewshot`: [Optional] Final number of samples after hybrid reranking 

**Auto-generated paths:**

- Test descriptions: `output/{dataset_name}/features/descriptions/`
- Collection name: `{dataset_name}_collection`
- Output directory: `output/{dataset_name}/evaluation/`

**Outputs:**

- `output/{dataset_name}/evaluation/predictions.csv` - Labels and predictions with Full results
- Console output with accuracy, F1 score, and RAG hit rate

**What it does:**

1. For each test window:
   - Extracts temporal segments (whole, start, mid, end)
   - Generates embeddings for each segment
   - Performs hybrid search in Milvus with multiple ANN requests
   - Retrieves top-k similar samples using weighted ranker
   - Uses LLM to classify based on semantic similarity to retrieved samples
   - Tracks RAG quality (whether true label appears in retrieved samples)
2. Calculates evaluation metrics (accuracy, F1 score, RAG hit rate)
3. Saves detailed results

---

## Complete Example Workflow

```bash
# Set required environment variables
export OPENAI_API_KEY="your-api-key-here"
export ZILLIZ_CLOUD_URI="your-milvus-uri"
export ZILLIZ_CLOUD_API_KEY="your-milvus-api-key"

# Step 1: Preprocessing (dataset-specific)
python preprocessing_new.py --config datasets/pamp2_config.yaml

# Step 2: Feature Extraction (temporal segmentation always enabled)
python generate_stats.py --config datasets/pamp2_config.yaml

# Step 3: Vector Indexing (creates 4 indexes: whole, start, mid, end)
python timeseries_indexing_new.py --config datasets/pamp2_config.yaml

# Step 4: Classification & Evaluation (hybrid search + LLM)
python classifier_new.py --config datasets/pamp2_config.yaml
```

**That's it!** All paths are automatically determined from the dataset name in the config file.

---

## Customizing Classification Prompts

The system prompts used for LLM classification can be customized per dataset in the configuration file. This allows you to tailor the classification instructions based on dataset-specific characteristics.

### Configuration

Add a `prompts` section to your dataset config file:

```yaml
# Classification prompts configuration (optional)
# If not specified, default prompts will be used
prompts:
  system_prompt: "Use semantic similarity to compare the candidate statistics with the retrieved samples and output the activity label that maximizes similarity; respond with only the class label from {classes} and nothing else."
  user_prompt: |
    You are given summary statistics for sensor data across temporal segments for labeled samples and one unlabeled candidate.

    --- CANDIDATE ---
    Time Series:
    {candidate_series}

    --- LABELED SAMPLES ---
    {retrieved_data}
```



### How It Works

1. The `PromptProvider` class (in `prompt_provider.py`) loads prompt templates from the dataset config
2. During classification, prompts are dynamically generated using these templates
3. This allows different datasets to use different classification strategies while maintaining the same codebase

---

## Adding a New Dataset

To add support for a new HAR dataset, follow these steps:

### 1. Create Dataset Configuration

Create `datasets/my_dataset_config.yaml`:

```yaml
dataset_name: "my_dataset"

data_source:
  folder_path: "/path/to/dataset"
  activities:
    - walking
    - running
    - sitting

preprocessing:
  window_size: 200
  step_size: 50

features:
  statistics:
    - mean
    - std
    - min
    - max
```

### 2. Required File Structure

**IMPORTANT:** Your dataset provider MUST follow this standardized structure:

**Preprocessing Step Outputs:**
```
output/{dataset_name}/train-test-splits/
├── train/
│   ├── {activity_name}/
│   │   ├── subject{idx}_window{idx}_activity{idx}_{activity_label}.csv
│   │   ├── subject{idx}_window{idx}_activity{idx}_{activity_label}.csv
│   │   └── ...
│   └── {another_activity}/
│       └── ...
└── test/
    ├── {activity_name}/
    │   └── ...
    └── ...
```

**Window CSV Naming:**
- Standard pattern: `subject{idx}_window{idx}_activity{idx}_{activity_label}.csv`
- Example: `subject101_window0000_activity1_walking.csv`

**Feature Extraction Step Outputs:**
```
output/{dataset_name}/features/
├── train/
│   └── descriptions/
│       ├── window_0_activity_walking_stats.txt
│       ├── window_1_activity_walking_stats.txt
│       └── ...
└── test/
    └── descriptions/
        └── ...
```

**Feature File Naming:**
- Pattern: `window_{idx}_activity_{activity_label}_stats.txt`
- Example: `window_0_activity_walking_stats.txt`

### 3. Create Dataset Provider

Create `providers/my_dataset/provider.py` implementing:

```python
class MyDatasetProvider(DatasetProvider):
    def preprocess(self, output_dir: str) -> str:
        # 1. Load your raw data
        # 2. Apply preprocessing (normalization, filtering)
        # 3. Create sliding windows
        # 4. Save in standardized format:
        #    output/{dataset}/train-test-splits/{train|test}/{activity}/subject{idx}_window{idx}_activity{idx}_{activity_label}.csv
        # 5. Return path to train-test-splits directory
        pass

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        from .features import MyDatasetFeatureExtractor
        extractor = MyDatasetFeatureExtractor(self.config)
        return extractor.extract_features(windows_dir, output_dir)
```

### 4. Create Feature Extractor

Create `providers/my_dataset/features.py`:

```python
class MyDatasetFeatureExtractor:
    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        # 1. Search for windows: windows_path.rglob("subject*.csv") or windows_path.rglob("*.csv")
        # 2. Parse filename to extract window_idx and activity_name
        # 3. Extract features per window (temporal segments: whole, start, mid, end)
        # 4. Generate descriptions
        # 5. Save as: window_{idx}_activity_{name}_stats.txt
        pass
```

### 5. Register Provider

Add to `dataset_provider.py`:

```python
provider_registry = {
    "my_dataset": ("providers.my_dataset.provider", "MyDatasetProvider"),
}
```

### 6. Run Pipeline

```bash
python preprocessing.py --config datasets/my_dataset_config.yaml
python generate_stats.py --config datasets/my_dataset_config.yaml
python timeseries_indexing.py --config datasets/my_dataset_config.yaml
python classifier.py --config datasets/my_dataset_config.yaml
```

**Reference implementations:** See `providers/pamap2/` for complete working examples

---
