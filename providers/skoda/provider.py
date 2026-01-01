"""
Skoda Dataset Provider - Loads data from the Skoda MiniCheckpoint dataset.
Handles .mat files from SkodaMiniCP_2015_08/Skoda.mat

Preprocessing for Skoda:
- Load pre-split data (train, validation, test)
- Fixed window size: 24 samples
- Train: 50% overlap (step=12), Test: no overlap (step=24)
- Saves to CSV files for RAG/LLM classification
"""

import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import scipy.io
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from dataset_provider import DatasetProvider

logger = logging.getLogger(__name__)


class SkodaProvider(DatasetProvider):
    """
    Dataset provider for Skoda MiniCheckpoint dataset.
    Loads .mat file from SkodaMiniCP_2015_08/Skoda.mat
    """

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from .mat file.
        Returns pre-split train, validation, and test sets.

        Returns:
            Dict mapping split names ('train', 'valid', 'test') to DataFrames
        """
        mat_file = self.config['data_source']['mat_file']
        logger.info(f"Loading Skoda .mat file from: {mat_file}")

        try:
            data = scipy.io.loadmat(mat_file)
        except FileNotFoundError:
            logger.error(f"Mat file not found: {mat_file}")
            raise

        # Extract data and labels
        X_train = data['X_train']
        X_valid = data['X_valid']
        X_test = data['X_test']
        y_train = data['y_train'].reshape(-1)
        y_valid = data['y_valid'].reshape(-1)
        y_test = data['y_test'].reshape(-1)

        logger.info(f"Loaded train: {X_train.shape}, valid: {X_valid.shape}, test: {X_test.shape}")

        # Create column headers for Skoda sensor layout
        headers = self._create_headers()

        # Convert to DataFrames
        splits = {}
        for split_name, X, y in [('train', X_train, y_train),
                                  ('valid', X_valid, y_valid),
                                  ('test', X_test, y_test)]:
            # Convert one-hot encoded labels back to label strings
            label_df = pd.get_dummies(y, prefix='labels')
            labels = label_df.idxmax(axis=1)

            # Combine labels and features
            combined = np.concatenate([labels.values.reshape(-1, 1), X], axis=1)
            df = pd.DataFrame(combined, columns=headers)
            splits[split_name] = df

            logger.info(f"  {split_name}: {len(df)} samples")

        return splits

    def _create_headers(self) -> List[str]:
        """Create column headers for Skoda sensor layout."""
        headers = ['label']
        left_sensor_ids = [3, 17, 19, 20, 23, 25, 26, 28, 30, 31]
        right_sensor_ids = [1, 2, 14, 16, 18, 21, 22, 24, 27, 29]

        for arm in ['left', 'right']:
            sensor_ids = left_sensor_ids if arm == 'left' else right_sensor_ids
            for s in range(10):
                sensor_id = sensor_ids[s]
                for axis_name in ['x', 'y', 'z']:
                    headers.append(f'{arm}_sensor{sensor_id}_acc_{axis_name}_calib')

        return headers

    def preprocess(self, output_dir: str) -> str:
        """
        Skoda specific preprocessing.

        Steps:
        1. Load raw data (pre-split train/valid/test)
        2. Combine train and valid for training set
        3. Segment into 24-sample windows
        4. Train: 50% overlap (step=12), Test: no overlap (step=24)
        5. Save as CSV files

        Args:
            output_dir: Directory to save preprocessed windows

        Returns:
            Path to saved windows directory
        """
        logger.info("="*60)
        logger.info("SKODA PREPROCESSING")
        logger.info("="*60)
        logger.info("Dataset: Skoda MiniCheckpoint 2015_08")
        logger.info("Sensors: 20 accelerometers (10 left arm, 10 right arm)")
        logger.info("Window size: 24 samples")
        logger.info("Train overlap: 50% (step=12), Test overlap: 0% (step=24)")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load data
        logger.info("Step 1: Loading raw data from .mat file...")
        splits = self.load_raw_data()

        # Step 2: Combine train and valid
        logger.info("Step 2: Combining train and valid sets...")
        train_df = pd.concat([splits['train'], splits['valid']], ignore_index=True)
        test_df = splits['test']
        logger.info(f"  Combined train: {len(train_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")

        # Get null class inclusion setting
        include_null_class = self.config['preprocessing'].get('include_null_class', True)

        # Step 3: Segment into windows
        logger.info("Step 3: Segmenting into windows...")
        train_test_dir = output_path / 'train-test-splits'
        train_test_dir.mkdir(exist_ok=True)

        train_dir = train_test_dir / 'train'
        test_dir = train_test_dir / 'test'
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        # Segment train with 50% overlap
        logger.info("  Segmenting training data (window=24, step=12)...")
        self._segment_and_save(train_df, train_dir, window=24, step=12,
                              include_null_class=include_null_class)

        # Segment test with no overlap
        logger.info("  Segmenting test data (window=24, step=24)...")
        self._segment_and_save(test_df, test_dir, window=24, step=24,
                              include_null_class=include_null_class)

        # Save summary
        self._save_summary(train_dir, test_dir, output_path)

        logger.info("")
        logger.info(f"✓ Output: {train_test_dir}")

        return str(train_test_dir)

    def _segment_and_save(self, df: pd.DataFrame, out_dir: Path,
                         window: int, step: int, include_null_class: bool = True):
        """
        Segment dataframe into windows and save as CSV files.

        Args:
            df: DataFrame to segment
            out_dir: Output directory
            window: Window size in samples
            step: Step size in samples (overlap = window - step)
            include_null_class: Whether to include null class (label_0.0)
        """
        window_count = 0

        for label, group in df.groupby('label'):
            # Skip null class if not included
            if not include_null_class and label == "labels_0.0":
                logger.info(f"  Skipping null class: {label}")
                continue

            # Clean activity name
            activity_name = str(label).replace('labels_', '').replace('.', '_')

            # Create activity folder
            activity_dir = out_dir / activity_name
            activity_dir.mkdir(exist_ok=True)

            group = group.reset_index(drop=True)
            label_windows = 0
            window_idx = 0

            for i in tqdm(range(0, len(group) - window + 1, step),
                         desc=f'  Processing {label}', leave=False):
                window_df = group.iloc[i:i+window]

                # Only save if all samples in window have same label
                if window_df['label'].nunique() == 1:
                    # New filename format: window_{idx}_activity_{name}.csv
                    filename = f'window_{window_idx}_activity_{activity_name}.csv'
                    out_path = activity_dir / filename
                    window_df.to_csv(out_path, index=False)
                    label_windows += 1
                    window_count += 1
                    window_idx += 1

            logger.info(f"  {label}: {label_windows} windows")

    def _save_summary(self, train_dir: Path, test_dir: Path, output_path: Path):
        """Save preprocessing summary."""
        summary_file = output_path / 'preprocessing_summary.txt'

        # Count windows
        train_files = list(train_dir.glob('*.csv'))
        test_files = list(test_dir.glob('*.csv'))

        # Count by label
        train_counts = {}
        for f in train_files:
            label = f.stem.split('_window_')[0]
            train_counts[label] = train_counts.get(label, 0) + 1

        test_counts = {}
        for f in test_files:
            label = f.stem.split('_window_')[0]
            test_counts[label] = test_counts.get(label, 0) + 1

        with open(summary_file, 'w') as f:
            f.write("Skoda Preprocessing Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total windows: {len(train_files) + len(test_files)}\n")
            f.write(f"Train windows: {len(train_files)}\n")
            f.write(f"Test windows: {len(test_files)}\n\n")
            f.write(f"Window size: 24 samples\n")
            f.write(f"Train step: 12 samples (50% overlap)\n")
            f.write(f"Test step: 24 samples (no overlap)\n\n")

            f.write("Train windows per label:\n")
            for label, count in sorted(train_counts.items()):
                f.write(f"  {label}: {count}\n")

            f.write("\nTest windows per label:\n")
            for label, count in sorted(test_counts.items()):
                f.write(f"  {label}: {count}\n")

        logger.info(f"  Saved summary to {summary_file}")

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        Skoda specific feature extraction.
        Delegates to SkodaFeatureExtractor in features.py

        Args:
            windows_dir: Path to preprocessed windows directory
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved features file
        """
        from .features import SkodaFeatureExtractor

        extractor = SkodaFeatureExtractor(self.config)
        return extractor.extract_features(windows_dir, output_dir)
