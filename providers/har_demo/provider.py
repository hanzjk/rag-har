"""
HAR Demo Dataset Provider - Loads data from the Flutter mobile app.
Handles CSV files from server/collected_data/

Preprocessing for HAR Demo:
- No normalization (raw sensor values preserved for RAG/LLM classification)
- Fixed window size: 200 samples (4 seconds at 50Hz)
- 50% overlap (100 sample step)
- No filtering needed (high-quality mobile sensor data)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from dataset_provider import DatasetProvider

logger = logging.getLogger(__name__)


class HARDemoProvider(DatasetProvider):
    """
    Dataset provider for HAR Demo mobile app data.
    Loads CSV files from server/collected_data/
    """

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw CSV data from collected_data directory.
        Handles subject-based folder structure: collected_data/subject1/walking.csv, etc.

        Returns:
            Dict mapping activity labels to DataFrames
        """
        base_path = Path(self.config['data_source']['base_path'])
        activities = self.config['data_source']['activities']

        # Find all subject folders
        subject_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('subject')]

        if not subject_folders:
            logger.warning(f"No subject folders found in {base_path}")
            return {}

        logger.info(f"Found {len(subject_folders)} subject folders: {[f.name for f in subject_folders]}")

        # Collect data by activity and subject
        data = {activity: [] for activity in activities}

        for subject_folder in subject_folders:
            subject_name = subject_folder.name
            logger.info(f"Loading data from {subject_name}...")

            for activity in activities:
                activity_file = subject_folder / f"{activity}.csv"

                if activity_file.exists():
                    df = pd.read_csv(activity_file)
                    
                    # Omit first and last 4 seconds (200 samples at 50Hz)
                    samples_to_omit = 200
                    if len(df) > 2 * samples_to_omit:
                        df = df.iloc[samples_to_omit:-samples_to_omit].reset_index(drop=True)
                        logger.info(f"  {subject_name}/{activity}.csv: {len(df)} samples (after omitting first/last 4s)")
                    else:
                        logger.warning(f"  {subject_name}/{activity}.csv: {len(df)} samples - too short to omit edges, skipping file")
                        continue
                    
                    # Add subject identifier column
                    df['subject_id'] = subject_name
                    data[activity].append(df)
                else:
                    logger.warning(f"  {subject_name}/{activity}.csv: not found")

        # Combine data from all subjects for each activity
        combined_data = {}
        for activity, dfs in data.items():
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_data[activity] = combined_df
                logger.info(f"Combined {activity}: {len(combined_df)} samples from {len(dfs)} subjects")
            else:
                logger.warning(f"No data found for activity: {activity}")

        return combined_data

    def preprocess(self, output_dir: str) -> str:
        """
        HAR Demo specific preprocessing.

        Steps:
        1. Load raw data (already has standardized column names)
        2. Segment into 200-sample windows with 50% overlap
        3. Train/test split (10% test per activity)
        4. Save as CSV files

        Args:
            output_dir: Directory to save preprocessed windows

        Returns:
            Path to saved windows file
        """
        logger.info("="*60)
        logger.info("HAR DEMO PREPROCESSING")
        logger.info("="*60)
        logger.info("Dataset: Clean mobile sensor data")
        logger.info("Sensors: Accelerometer, Gyroscope, Magnetometer (50Hz)")
        logger.info("Normalization: None (raw sensor values preserved)")
        logger.info("Window size: 200 samples (4 seconds)")
        logger.info("Overlap: 50% (100 sample step)")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load data
        logger.info("Step 1: Loading raw data...")
        data = self.load_raw_data()

        # Step 2: Segment into windows
        logger.info("Step 2: Segmenting into windows...")
        windows = self._segment_windows(data)

        # Step 3: Train/Test split (10% test per activity)
        logger.info("Step 3: Splitting train/test (10% test per activity)...")
        train_windows, test_windows = self._split_train_test(windows, test_ratio=0.1)

        # Step 4: Save windows
        logger.info("Step 4: Saving windows...")
        train_test_dir = output_path / 'train-test-splits'
        train_test_dir.mkdir(exist_ok=True)

        self._save_windows(train_windows, train_test_dir, split_name='train')
        self._save_windows(test_windows, train_test_dir, split_name='test')

        # Save summary
        self._save_summary(train_windows, test_windows, output_path)

        logger.info("")
        logger.info(f"✓ Train: {len(train_windows)} windows")
        logger.info(f"✓ Test: {len(test_windows)} windows")
        logger.info(f"✓ Output: {train_test_dir}")

        return str(train_test_dir)

    def _segment_windows(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Segment data into 200-sample windows with 50% overlap."""
        window_size = 200
        step_size = 100  # 50% overlap
        windows = []

        for activity, df in data.items():
            num_windows = (len(df) - window_size) // step_size + 1
            window_id = 0  # Reset window counter for each activity

            for i in range(num_windows):
                start_idx = i * step_size
                end_idx = start_idx + window_size

                if end_idx > len(df):
                    break

                window_data = df.iloc[start_idx:end_idx].copy()

                # Extract subject_id from the window data (all rows should have same subject_id)
                subject_id = window_data['subject_id'].iloc[0] if 'subject_id' in window_data.columns else 'unknown'

                windows.append({
                    'window_id': window_id,
                    'activity': activity,
                    'subject_id': subject_id,
                    'data': window_data,
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'num_samples': len(window_data)
                })

                window_id += 1

            logger.info(f"  {activity}: {num_windows} windows")

        return windows

    def _split_train_test(self, windows: List[Dict], test_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """
        Split windows into train and test sets.
        Takes 10% from each activity for testing.

        Args:
            windows: List of all windows
            test_ratio: Ratio of test data (default 0.1 = 10%)

        Returns:
            Tuple of (train_windows, test_windows)
        """
        import random

        # Group windows by activity
        windows_by_activity = {}
        for window in windows:
            activity = window['activity']
            if activity not in windows_by_activity:
                windows_by_activity[activity] = []
            windows_by_activity[activity].append(window)

        train_windows = []
        test_windows = []

        # Split each activity
        for activity, activity_windows in windows_by_activity.items():
            # Shuffle to ensure random split
            random.shuffle(activity_windows)

            # Calculate split point
            n_test = max(1, int(len(activity_windows) * test_ratio))
            n_train = len(activity_windows) - n_test

            # Split
            train_windows.extend(activity_windows[:n_train])
            test_windows.extend(activity_windows[n_train:])

            logger.info(f"  {activity}: {n_train} train, {n_test} test")

        return train_windows, test_windows

    def _save_windows(self, windows: List[Dict], output_path: Path, split_name: str) -> Path:
        """
        Save windows as CSV files.
        CSV structure: train/subject1/walking/subject1_window_0_walking.csv

        Args:
            windows: List of window dictionaries
            output_path: Base output path (train-test-splits directory)
            split_name: 'train' or 'test'

        Returns:
            Path to the split directory
        """
        # Save CSVs with organized folder structure
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)

        for window in windows:
            subject_id = window.get('subject_id', 'unknown')
            activity = window['activity']
            window_id = window['window_id']

            # Create subject/activity folder structure
            activity_dir = split_dir / subject_id / activity
            activity_dir.mkdir(parents=True, exist_ok=True)

            # Filename includes subject_id
            csv_file = activity_dir / f"{subject_id}_window_{window_id}_{activity}.csv"
            window['data'].to_csv(csv_file, index=False)

        logger.info(f"  Saved {len(windows)} {split_name} CSV files to {split_dir}")

        return split_dir

    def _save_summary(self, train_windows: List[Dict], test_windows: List[Dict], output_path: Path):
        """Save preprocessing summary with train/test split information."""
        summary_file = output_path / 'preprocessing_summary.txt'

        # Count activities for train
        train_activity_counts = {}
        for window in train_windows:
            activity = window['activity']
            train_activity_counts[activity] = train_activity_counts.get(activity, 0) + 1

        # Count activities for test
        test_activity_counts = {}
        for window in test_windows:
            activity = window['activity']
            test_activity_counts[activity] = test_activity_counts.get(activity, 0) + 1

        with open(summary_file, 'w') as f:
            f.write("HAR Demo Preprocessing Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total windows: {len(train_windows) + len(test_windows)}\n")
            f.write(f"Train windows: {len(train_windows)}\n")
            f.write(f"Test windows: {len(test_windows)}\n")
            f.write(f"Test ratio: 10%\n\n")
            f.write(f"Window size: 200 samples (4 seconds)\n")
            f.write(f"Step size: 100 samples (50% overlap)\n")
            f.write(f"Sampling rate: 50 Hz\n")
            f.write(f"Normalization: None (raw sensor values)\n\n")

            f.write("Train windows per activity:\n")
            for activity, count in sorted(train_activity_counts.items()):
                f.write(f"  {activity}: {count}\n")

            f.write("\nTest windows per activity:\n")
            for activity, count in sorted(test_activity_counts.items()):
                f.write(f"  {activity}: {count}\n")

        logger.info(f"  Saved summary to {summary_file}")

    def extract_features(self, windows_file: str, output_dir: str) -> str:
        """
        HAR Demo specific feature extraction.
        Delegates to HARDemoFeatureExtractor in features.py

        Args:
            windows_file: Path to preprocessed windows pickle file
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved features file
        """
        from .features import HARDemoFeatureExtractor

        extractor = HARDemoFeatureExtractor(self.config)
        return extractor.extract_features(windows_file, output_dir)
