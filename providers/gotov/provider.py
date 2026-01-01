"""
GOTOV Dataset Provider - Loads data from the GOTOV Activity Recognition dataset.
Handles multi-sensor data from ankle, wrist, and chest sensors.

Preprocessing for GOTOV:
- Load and merge data from 3 sensor locations (ankle, wrist, chest)
- Normalize using training set statistics (z-score normalization)
- Subject-based train/test/validation split
- Fixed window size: 24 samples
- Train: 50% overlap (step=12), Test: no overlap (step=24)
- Saves to CSV files for RAG/LLM classification
"""

import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import logging
from tqdm import tqdm
from glob import glob
from functools import reduce
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from dataset_provider import DatasetProvider

logger = logging.getLogger(__name__)


class GOTOVProvider(DatasetProvider):
    """
    Dataset provider for GOTOV Activity Recognition dataset.
    Loads data from multiple sensor locations (ankle, wrist, chest).
    """

    def __init__(self, config_path: str):
        """Initialize GOTOV provider."""
        super().__init__(config_path)

        # Subject splits from config
        self.exclude_subjects = set(self.config['preprocessing'].get('exclude_subjects', [2, 3, 4, 12, 19, 23]))
        self.test_subjects = set(self.config['preprocessing'].get('test_subjects', [5, 15, 30]))
        self.val_subjects = set(self.config['preprocessing'].get('val_subjects', [13, 21, 29]))

        # Sensor locations
        self.sensor_types = ['ankle', 'wrist', 'chest']

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from all subjects.
        Combines data from 3 sensor locations for each subject.

        Returns:
            Dict mapping subject IDs to merged DataFrames
        """
        dataset_dir = self.config['data_source']['dataset_dir']
        dataset_path = Path(dataset_dir)

        subject_folders = [f for f in dataset_path.iterdir() if f.is_dir() and f.name.startswith("GOTOV")]
        logger.info(f"Found {len(subject_folders)} subject folders")

        subjects_data = {}

        for folder in subject_folders:
            subject_id = self._get_subject_id(folder.name)

            # Skip excluded subjects
            if subject_id in self.exclude_subjects:
                logger.debug(f"Skipping excluded subject {subject_id}")
                continue

            # Load and merge sensor data for this subject
            df = self._combine_sensor_files(folder)

            if df is not None and len(df) > 0:
                subjects_data[f"subject_{subject_id}"] = df
                logger.info(f"Subject {subject_id}: {len(df)} samples")

        return subjects_data

    def _get_subject_id(self, folder_name: str) -> int:
        """Extract subject ID from folder name (e.g., GOTOV1 -> 1)."""
        return int(folder_name.replace("GOTOV", ""))

    def _combine_sensor_files(self, subject_folder: Path) -> pd.DataFrame:
        """
        Load and merge data from all sensor locations for a subject.

        Args:
            subject_folder: Path to subject's data folder

        Returns:
            Merged DataFrame with all sensor data
        """
        sensor_dfs = {}

        # Load and prepare each sensor file
        for sensor in self.sensor_types:
            pattern = str(subject_folder / f"GOTOV*-*-{sensor}.csv")
            files = glob(pattern)

            if not files:
                logger.warning(f"No {sensor} sensor file found for {subject_folder.name}")
                continue

            df = pd.read_csv(files[0], engine='python')

            # Remove N/A and NaN in label
            if 'labels' in df.columns:
                df = df[df['labels'].notna() & (df['labels'] != 'N/A')]
            else:
                df['labels'] = 'unknown'

            # Rename axes to include sensor location
            df = df.rename(columns={
                'x': f'{sensor}_x',
                'y': f'{sensor}_y',
                'z': f'{sensor}_z'
            })

            # Ensure time is float
            df['time'] = df['time'].astype(float)

            # Only keep necessary columns
            sensor_dfs[sensor] = df[['time', f'{sensor}_x', f'{sensor}_y', f'{sensor}_z', 'labels']]

        if len(sensor_dfs) < 3:
            logger.warning(f"Missing sensor data in {subject_folder.name}. Found: {list(sensor_dfs.keys())}")

        if not sensor_dfs:
            return None

        # Merge all sensor dataframes on time and labels
        merged_df = reduce(
            lambda left, right: pd.merge(
                left, right,
                on=["time", "labels"],
                how="outer"
            ),
            sensor_dfs.values()
        )

        # Drop rows with missing values
        merged_df_cleaned = merged_df.dropna().reset_index(drop=True)

        return merged_df_cleaned

    def preprocess(self, output_dir: str) -> str:
        """
        GOTOV specific preprocessing.

        Steps:
        1. Load raw data from all subjects
        2. Compute normalization statistics from training subjects
        3. Normalize all subjects using training statistics
        4. Segment into 24-sample windows
        5. Train: 50% overlap (step=12), Test: no overlap (step=24)
        6. Save windows as CSV files

        Args:
            output_dir: Directory to save preprocessed windows

        Returns:
            Path to saved windows directory
        """
        logger.info("="*60)
        logger.info("GOTOV PREPROCESSING")
        logger.info("="*60)
        logger.info("Dataset: GOTOV Activity Recognition")
        logger.info("Sensors: Ankle, Wrist, Chest (3-axis accelerometers)")
        logger.info("Window size: 24 samples")
        logger.info("Train overlap: 50% (step=12), Test overlap: 0% (step=24)")
        logger.info("Normalization: Z-score using training set statistics")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load all subjects
        logger.info("Step 1: Loading raw data from all subjects...")
        subjects_data = self.load_raw_data()

        # Step 2: Compute normalization statistics from training subjects
        logger.info("Step 2: Computing normalization statistics from training subjects...")
        train_dfs = []
        for subject_name, df in subjects_data.items():
            subject_id = int(subject_name.split('_')[1])
            if subject_id not in self.test_subjects and subject_id not in self.val_subjects:
                train_dfs.append(df)

        normalization_stats = self._compute_normalization_stats(train_dfs)
        self._save_normalization_stats(normalization_stats, output_path)

        # Step 3: Process all subjects with normalization and segmentation
        logger.info("Step 3: Normalizing and segmenting data...")

        train_test_dir = output_path / 'train-test-splits'
        train_dir = train_test_dir / 'train'
        test_dir = train_test_dir / 'test'
        val_dir = train_test_dir / 'validation'

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        window_size = self.config['preprocessing']['window_size']

        for subject_name, df in tqdm(subjects_data.items(), desc="Processing subjects"):
            subject_id = int(subject_name.split('_')[1])

            # Normalize using training statistics
            df_normalized = self._normalize_dataframe(df, normalization_stats)

            # Determine split and save windows
            if subject_id in self.test_subjects:
                self._save_windows(df_normalized, subject_id, test_dir, window_size, step_size=24, split_type="test")
            elif subject_id in self.val_subjects:
                self._save_full_sample(df_normalized, subject_id, val_dir, split_type="validation")
            else:
                self._save_windows(df_normalized, subject_id, train_dir, window_size, step_size=12, split_type="train")

        # Save summary
        self._save_summary(train_dir, test_dir, val_dir, output_path)

        logger.info("")
        logger.info(f"✓ Output: {train_test_dir}")

        return str(train_test_dir)

    def _compute_normalization_stats(self, train_dfs: List[pd.DataFrame]) -> Dict:
        """
        Compute mean and std from all training data for z-score normalization.

        Args:
            train_dfs: List of training DataFrames

        Returns:
            Dict with normalization statistics for each sensor column
        """
        # Get sensor column names (exclude time and labels)
        sensor_columns = []
        for df in train_dfs:
            sensor_columns.extend([col for col in df.columns if col not in ['time', 'labels']])
        sensor_columns = list(set(sensor_columns))

        # Concatenate all training data
        all_train_data = pd.concat(train_dfs, ignore_index=True)

        # Compute statistics for each sensor column
        normalization_stats = {}
        for col in sensor_columns:
            if col in all_train_data.columns:
                normalization_stats[col] = {
                    'mean': float(all_train_data[col].mean()),
                    'std': float(all_train_data[col].std())
                }

        logger.info(f"  Computed normalization stats for {len(normalization_stats)} sensor columns")
        return normalization_stats

    def _normalize_dataframe(self, df: pd.DataFrame, normalization_stats: Dict) -> pd.DataFrame:
        """
        Normalize a dataframe using precomputed statistics (z-score normalization).

        Args:
            df: DataFrame to normalize
            normalization_stats: Dict with mean and std for each column

        Returns:
            Normalized DataFrame
        """
        df_normalized = df.copy()

        for col, stats in normalization_stats.items():
            if col in df_normalized.columns:
                if stats['std'] > 0:  # Avoid division by zero
                    df_normalized[col] = (df_normalized[col] - stats['mean']) / stats['std']
                else:
                    df_normalized[col] = df_normalized[col] - stats['mean']

        return df_normalized

    def _save_normalization_stats(self, stats: Dict, output_dir: Path):
        """Save normalization statistics to a JSON file."""
        stats_path = output_dir / 'train-test-splits' / 'normalization_stats.json'
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"  Saved normalization statistics to {stats_path}")

    def _save_windows(self, df: pd.DataFrame, subject_id: int, out_dir: Path,
                     window_size: int, step_size: int, split_type: str = "train"):
        """
        Split dataframe into windows and save in activity-based folders.
        Uses PAMAP2-style structure: split/activity_name/filename.csv

        Args:
            df: DataFrame to segment
            subject_id: Subject identifier
            out_dir: Output directory
            window_size: Window size in samples
            step_size: Step size in samples
            split_type: 'train', 'test', or 'validation'
        """
        num_windows = (len(df) - window_size) // step_size + 1
        skipped = 0
        saved = 0

        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            window = df.iloc[start:end]

            if len(window) == window_size:
                unique_labels = window['labels'].unique()
                if len(unique_labels) == 1:
                    activity_name = str(unique_labels[0]).replace(' ', '_').replace('/', '_')

                    # Create activity-based folder
                    activity_dir = out_dir / activity_name
                    activity_dir.mkdir(exist_ok=True)

                    # New filename format: subject{id}_window{num}_activity_{name}.csv
                    filename = f"subject{subject_id}_window{i}_activity_{activity_name}.csv"
                    out_path = activity_dir / filename
                    window.to_csv(out_path, index=False)
                    saved += 1
                else:
                    skipped += 1

        logger.info(f"  Subject {subject_id} ({split_type}): Saved {saved} windows, skipped {skipped} mixed-label windows")

    def _save_full_sample(self, df: pd.DataFrame, subject_id: int, out_dir: Path, split_type: str = "validation"):
        """
        Save full subject data without windowing (for validation).

        Args:
            df: DataFrame to save
            subject_id: Subject identifier
            out_dir: Output directory
            split_type: Split type name
        """
        out_path = out_dir / f"subject{subject_id}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  Subject {subject_id} ({split_type}): Saved full sample with {len(df)} rows")

    def _save_summary(self, train_dir: Path, test_dir: Path, val_dir: Path, output_path: Path):
        """Save preprocessing summary."""
        summary_file = output_path / 'preprocessing_summary.txt'

        # Count windows
        train_files = list(train_dir.glob('*.csv'))
        test_files = list(test_dir.glob('*.csv'))
        val_files = list(val_dir.glob('*.csv'))

        with open(summary_file, 'w') as f:
            f.write("GOTOV Preprocessing Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total windows: {len(train_files) + len(test_files) + len(val_files)}\n")
            f.write(f"Train windows: {len(train_files)}\n")
            f.write(f"Test windows: {len(test_files)}\n")
            f.write(f"Validation files: {len(val_files)}\n\n")

            window_size = self.config['preprocessing']['window_size']
            f.write(f"Window size: {window_size} samples\n")
            f.write(f"Train step: 12 samples (50% overlap)\n")
            f.write(f"Test step: 24 samples (no overlap)\n")
            f.write(f"Normalization: Z-score using training set statistics\n\n")

            f.write(f"Excluded subjects: {sorted(self.exclude_subjects)}\n")
            f.write(f"Test subjects: {sorted(self.test_subjects)}\n")
            f.write(f"Validation subjects: {sorted(self.val_subjects)}\n")

        logger.info(f"  Saved summary to {summary_file}")

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        GOTOV specific feature extraction.
        Delegates to GOTOVFeatureExtractor in features.py

        Args:
            windows_dir: Path to preprocessed windows directory
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved features file
        """
        from .features import GOTOVFeatureExtractor

        extractor = GOTOVFeatureExtractor(self.config)
        return extractor.extract_features(windows_dir, output_dir)
