# RAG-HAR Pipeline

A **dataset-agnostic** Retrieval-Augmented Generation pipeline for Human Activity Recognition using vector similarity search.

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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Pipeline Stages

Each stage is run independently using its own script.

---

## Stage 1: Preprocessing

**Purpose:** Dataset-specific preprocessing - each dataset implements its own logic.

**Script:** `preprocessing.py` (wrapper that calls provider's preprocess())

**Command:**

```bash
python preprocessing.py --config datasets/har_demo_config.yaml
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

- `output/{dataset_name}/train-test-splits/` - Train/Test split directories
  - `train/`
    - `subject1/walking/subject1_window_0_walking.csv`
    - `subject1/walking/subject1_window_1_walking.csv`
    - `subject1/running/subject1_window_0_running.csv`
    - ...
  - `test/`
    - `subject1/walking/subject1_window_10_walking.csv`
    - `subject2/running/subject2_window_5_running.csv`
    - ...
- `output/{dataset_name}/scaler.pkl` - Normalization scaler for later use
- `output/{dataset_name}/preprocessing_summary.txt` - Statistics with train/test split info

**What it does:**

1. Loads raw CSV files via DatasetProvider
2. Provide the preprocessing of the dataset accordingly
3. Segments into overlapping windows (e.g., 200 samples with 50% overlap)
4. Splits into train/test sets

---

## Stage 2: Feature Extraction

**Purpose:** Calculate statistical features for each window and generate human-readable descriptions.

**Script:** `generate_stats.py`

**Command:**

```bash
python generate_stats.py --config datasets/har_demo_config.yaml
```

**Parameters:**

- `--config`: Path to dataset configuration YAML file

**Auto-generated paths:**

- Train Input: `output/{dataset_name}/train-test-splits/train/`
- Test Input: `output/{dataset_name}/train-test-splits/test/`
- Train Output: `output/{dataset_name}/features/train/`
- Test Output: `output/{dataset_name}/features/test/`

**Outputs:**

- `output/{dataset_name}/features/train/` - Training set features
  - `features.pkl` - Feature vectors (numpy arrays)
  - `descriptions/` - Human-readable text descriptions
  - `feature_summary.txt` - Statistics
- `output/{dataset_name}/features/test/` - Test set features (same structure)

**What it does:**

1. For each window, calculates a set statistical features (mean, std, min, max, median, etc.)
2. Computes features per sensor axis (x, y, z)
3. Segments window into temporal parts (whole, start, mid, end) for richer features
4. Generates human-readable text descriptions of the features

**Dataset-Specific Feature Extraction:**
Each dataset knows how to extract features from its own column structure:

- **HAR Demo**: Handled by `providers/har_demo/features.py` (extracts from `accel_x`, `gyro_y`, `mag_z`)
- **Your dataset**: Create `providers/your_dataset/features.py` with your column names

No configuration needed - each provider's `features.py` defines which columns to extract!

---

## Stage 3: Vector Database Indexing

**Purpose:** Create embeddings from feature descriptions and index them into a vector database.

**Script:** `timeseries_indexing.py`

**Command:**

```bash
export OPENAI_API_KEY="your-api-key-here"
export ZILLIZ_CLOUD_URI="your-milvus-uri"
export ZILLIZ_CLOUD_API_KEY="your-milvus-api-key"

python timeseries_indexing.py --config datasets/har_demo_config.yaml
```

**Parameters:**

- `--config`: Path to dataset configuration YAML file

**Auto-generated paths:**

- Input: `output/{dataset_name}/features/descriptions/`
- Collection name: `{dataset_name}_har_collection`
- Embedding model: `openai`

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
export OPENAI_API_KEY="your-api-key-here"
export ZILLIZ_CLOUD_URI="your-milvus-uri"
export ZILLIZ_CLOUD_API_KEY="your-milvus-api-key"

python classifier.py --config datasets/har_demo_config.yaml
```

**Parameters:**

- `--config`: Path to dataset configuration YAML file (required)
- `--model`: LLM model for classification (default: `gpt-4o-mini`)
- `--fewshot`: Number of samples to retrieve per temporal segment (default: 30)
- `--out-fewshot`: Final number of samples after hybrid reranking (default: 20)

**Auto-generated paths:**

- Test descriptions: `output/{dataset_name}/features/descriptions/`
- Collection name: `{dataset_name}_har_collection`
- Output directory: `output/{dataset_name}/evaluation/`

**Outputs:**

- `output/{dataset_name}/evaluation/predictions.csv` - Labels and predictions
- `output/{dataset_name}/evaluation/detailed_results.csv` - Full results with RAG metrics
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
python preprocessing_new.py --config datasets/har_demo_config.yaml

# Step 2: Feature Extraction (temporal segmentation always enabled)
python generate_stats.py --config datasets/har_demo_config.yaml

# Step 3: Vector Indexing (creates 4 indexes: whole, start, mid, end)
python timeseries_indexing_new.py --config datasets/har_demo_config.yaml

# Step 4: Classification & Evaluation (hybrid search + LLM)
python classifier_new.py --config datasets/har_demo_config.yaml
```

**That's it!** All paths are automatically determined from the dataset name in the config file.

---

## Adding a New Dataset

### 1. Create Dataset Configuration

Create `datasets/my_dataset_config.yaml`:

### 2. Create Dataset Provider Folder

Create `providers/my_dataset/` with three files:

**a) `providers/my_dataset/__init__.py`:**

```python
"""My Dataset Provider"""

from .provider import MyDatasetProvider

__all__ = ['MyDatasetProvider']
```

**b) `providers/my_dataset/provider.py`:**

```python
"""
My Dataset Provider - Loads and preprocesses my custom HAR dataset.
"""

import sys
import pickle
from pathlib import Path
from typing import Dict
import pandas as pd
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))
from dataset_provider import DatasetProvider

logger = logging.getLogger(__name__)


class MyDatasetProvider(DatasetProvider):
    """Dataset provider for my custom dataset."""

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data from your custom format."""
        base_path = Path(self.config['data_source']['base_path'])
        data = {}

        # TODO: Implement your loading logic
        # for file in base_path.glob("*.csv"):
        #     df = pd.read_csv(file)
        #     activity = extract_activity(file)
        #     data[activity] = df

        return data

    def preprocess(self, output_dir: str) -> str:
        """Dataset-specific preprocessing."""
        logger.info("Starting preprocessing...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # TODO: Implement your preprocessing
        # 1. Load data: data = self.load_raw_data()
        # 2. Normalize: normalized = self._normalize(data)
        # 3. Segment: windows = self._segment(normalized)
        # 4. Save: csv_windows_dir = self._save(windows, output_path)
        # 5. Return: return str(csv_windows_dir)

        raise NotImplementedError(
            "See providers/har_demo/provider.py for example"
        )

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """Delegates to MyDatasetFeatureExtractor."""
        from .features import MyDatasetFeatureExtractor

        extractor = MyDatasetFeatureExtractor(self.config)
        return extractor.extract_features(windows_dir, output_dir)
```

**c) `providers/my_dataset/features.py`:**

```python
"""
My Dataset Feature Extraction
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

from ..common.feature_utils import FeatureExtractorUtils

logger = logging.getLogger(__name__)


class MyDatasetFeatureExtractor:
    """Feature extractor for my dataset."""

    def __init__(self, config: Dict):
        self.config = config
        feature_config = config.get('features', {})

        self.statistics = feature_config.get('statistics',
            ['mean', 'std', 'min', 'max', 'median', 'p25', 'p75'])
        self.compute_magnitude = feature_config.get('magnitude', True)
        self.per_axis = feature_config.get('per_axis', True)

        # TODO: Define your sensor columns
        # Example for standard sensors:
        self.sensor_columns = ['accel', 'gyro', 'mag']

        # Example for multi-location sensors:
        # self.sensor_columns = [
        #     'hand_accel', 'hand_gyro', 'hand_mag',
        #     'torso_accel', 'torso_gyro', 'torso_mag'
        # ]

        self.utils = FeatureExtractorUtils()

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """Extract features from all windows."""
        # TODO: Implement feature extraction
        # See providers/har_demo/features.py for complete example
        raise NotImplementedError(
            "See providers/har_demo/features.py for example"
        )
```

### 3. Register Provider

Edit `dataset_provider.py`, add your provider to the registry:

```python
provider_registry = {
    "har_demo": ("providers.har_demo.provider", "HARDemoProvider"),
    "gotov": ("providers.gotov.provider", "GOTOVProvider"),
    "my_dataset": ("providers.my_dataset.provider", "MyDatasetProvider"),  # Add this
}
```

### 4. Run Pipeline

```bash
export OPENAI_API_KEY="your-api-key-here"
export ZILLIZ_CLOUD_URI="your-milvus-uri"
export ZILLIZ_CLOUD_API_KEY="your-milvus-api-key"

python preprocessing_new.py --config datasets/my_dataset_config.yaml
python generate_stats.py --config datasets/my_dataset_config.yaml
python timeseries_indexing_new.py --config datasets/my_dataset_config.yaml
python classifier_new.py --config datasets/my_dataset_config.yaml
```

All paths are automatically created as `output/my_dataset/{windows,features}/`

---
