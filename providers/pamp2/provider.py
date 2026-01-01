"""
PAMAP2 Dataset Provider - Loads data from the PAMAP2 Physical Activity Monitoring dataset.
Handles multi-sensor IMU data from hand, chest, and ankle locations.

Preprocessing for PAMAP2:
- Load data from subject .dat files
- Remove transient activity (activity_id=0)
- Remove rows with NaN values
- Window size: 256 samples (2.56 seconds at 100Hz)
- Configurable overlap (default: 50% = step size 128)
- Saves to CSV files for RAG/LLM classification
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from dataset_provider import DatasetProvider

logger = logging.getLogger(__name__)


class PAMAP2Provider(DatasetProvider):
    """
    Dataset provider for PAMAP2 (Physical Activity Monitoring) dataset.
    Handles 3 IMU sensors (hand, chest, ankle) with accelerometer, gyroscope, and magnetometer data.
    """

    def __init__(self, config_path: str):
        """Initialize PAMAP2 provider."""
        super().__init__(config_path)

        # PAMAP2 column definitions (54 columns total)
        self.column_names = [
            'timestamp', 'activity_id', 'heartrate',
            # Hand IMU (4-20: 17 columns)
            'hand_temp', 'hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
            'hand_acc6_x', 'hand_acc6_y', 'hand_acc6_z',
            'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
            'hand_mag_x', 'hand_mag_y', 'hand_mag_z',
            'hand_orient_x', 'hand_orient_y', 'hand_orient_z', 'hand_orient_w',
            # Chest IMU (21-37: 17 columns)
            'chest_temp', 'chest_acc16_x', 'chest_acc16_y', 'chest_acc16_z',
            'chest_acc6_x', 'chest_acc6_y', 'chest_acc6_z',
            'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
            'chest_mag_x', 'chest_mag_y', 'chest_mag_z',
            'chest_orient_x', 'chest_orient_y', 'chest_orient_z', 'chest_orient_w',
            # Ankle IMU (38-54: 17 columns)
            'ankle_temp', 'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z',
            'ankle_acc6_x', 'ankle_acc6_y', 'ankle_acc6_z',
            'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
            'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
            'ankle_orient_x', 'ankle_orient_y', 'ankle_orient_z', 'ankle_orient_w'
        ]

        # Keep only acc16, gyro, and mag columns
        self.keep_columns = [
            'timestamp', 'activity_id',
            # Hand IMU
            'hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
            'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
            'hand_mag_x', 'hand_mag_y', 'hand_mag_z',
            # Chest IMU
            'chest_acc16_x', 'chest_acc16_y', 'chest_acc16_z',
            'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
            'chest_mag_x', 'chest_mag_y', 'chest_mag_z',
            # Ankle IMU
            'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z',
            'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
            'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
        ]

        # Activity mapping
        self.activity_map = {
            0: 'transient',
            1: 'lying',
            2: 'sitting',
            3: 'standing',
            4: 'walking',
            5: 'running',
            6: 'cycling',
            7: 'nordic_walking',
            9: 'watching_tv',
            10: 'computer_work',
            11: 'car_driving',
            12: 'ascending_stairs',
            13: 'descending_stairs',
            16: 'vacuum_cleaning',
            17: 'ironing',
            18: 'folding_laundry',
            19: 'house_cleaning',
            20: 'playing_soccer',
            24: 'rope_jumping'
        }

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from subject .dat files.

        Returns:
            Dict mapping subject IDs to DataFrames
        """
        folder_path = Path(self.config['data_source']['folder_path'])

        subjects_data = {}

        # Find all subject .dat files
        dat_files = sorted(folder_path.glob("subject*.dat"))

        if not dat_files:
            logger.warning(f"No subject*.dat files found in {folder_path}")
            return subjects_data

        logger.info(f"Found {len(dat_files)} subject files")

        for dat_file in dat_files:
            subject_id = dat_file.stem.replace('subject', '')

            try:
                # Load the data file (space-separated, no header)
                data = pd.read_csv(dat_file, sep=' ', header=None, na_values='NaN')

                # Check column count
                if len(data.columns) != len(self.column_names):
                    logger.error(f"Column count mismatch in {dat_file.name}! "
                               f"Expected {len(self.column_names)}, got {len(data.columns)}")
                    continue

                # Assign column names
                data.columns = self.column_names

                # Convert timestamp to datetime
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

                subjects_data[subject_id] = data
                logger.info(f"Subject {subject_id}: {len(data)} samples")

            except Exception as e:
                logger.error(f"Error loading {dat_file.name}: {e}")
                continue

        return subjects_data

    def preprocess(self, output_dir: str) -> str:
        """
        PAMAP2 specific preprocessing.

        Steps:
        1. Load raw data from all subjects
        2. Remove transient activity (activity_id=0)
        3. Keep only acc16, gyro, mag columns
        4. Remove rows with NaN values
        5. Create sliding windows
        6. Train/test split (one-subject-out)
        7. Save windows as CSV files

        Args:
            output_dir: Directory to save preprocessed windows

        Returns:
            Path to saved train-test-splits directory
        """
        logger.info("="*60)
        logger.info("PAMAP2 PREPROCESSING")
        logger.info("="*60)
        logger.info("Dataset: PAMAP2 Physical Activity Monitoring")
        logger.info("Sensors: 3 IMUs (hand, chest, ankle) - acc16, gyro, mag")
        logger.info(f"Window size: {self.config['preprocessing']['window_size']} samples")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load data
        logger.info("Step 1: Loading raw data from subject files...")
        subjects_data = self.load_raw_data()

        if not subjects_data:
            logger.error("No subject data loaded. Exiting.")
            return str(output_path)

        # Step 2: Remove transient and clean data
        logger.info("Step 2: Removing transient activity and cleaning data...")
        subjects_clean = {}
        for subject_id, data in subjects_data.items():
            # Remove activity_id=0 (transient)
            data_filtered = data[data['activity_id'] != 0].copy()

            # Keep only selected columns
            data_selected = data_filtered[self.keep_columns].copy()

            # Remove NaN rows
            original_len = len(data_selected)
            data_clean = data_selected.dropna()

            subjects_clean[subject_id] = data_clean
            logger.info(f"  Subject {subject_id}: {len(data_clean)} samples "
                       f"(removed {original_len - len(data_clean)} NaN rows)")

        # Step 3: Create sliding windows
        logger.info("Step 3: Creating sliding windows...")
        window_size = self.config['preprocessing']['window_size']
        step_size = self.config['preprocessing'].get('step_size', window_size)

        all_windows = []
        for subject_id, data in tqdm(subjects_clean.items(), desc="Processing subjects"):
            windows = self._create_sliding_windows(data, subject_id, window_size, step_size)

            if windows:
                all_windows.extend(windows)
            else:
                logger.warning(f"No valid windows created for subject {subject_id}")

        logger.info(f"  Total windows created: {len(all_windows)}")

        # Step 4: Train/test split (one-subject-out)
        split_config = self.config.get('train_test_split', {})
        test_subject_id = split_config.get('test_subject_id')

        if test_subject_id:
            logger.info(f"Step 4: Creating one-subject-out split (test subject: {test_subject_id})...")
            train_windows, test_windows = self._split_one_subject_out(all_windows, test_subject_id)

            # Step 5: Save windows
            logger.info("Step 5: Saving windows...")
            train_test_dir = output_path / 'train-test-splits'
            train_test_dir.mkdir(parents=True, exist_ok=True)

            self._save_split_windows(train_windows, train_test_dir, 'train')
            self._save_split_windows(test_windows, train_test_dir, 'test')

            # Save summary
            self._save_split_summary(train_windows, test_windows, output_path, test_subject_id)

            logger.info("")
            logger.info(f"✓ Train windows: {len(train_windows)}")
            logger.info(f"✓ Test windows: {len(test_windows)}")
            logger.info(f"✓ Output: {train_test_dir}")

            return str(train_test_dir)
        else:
            logger.warning("No test_subject_id specified in config. Saving all windows without split...")
            # Save windows without split
            windows_dir = output_path / 'windows'
            windows_dir.mkdir(exist_ok=True)

            for window_data, window_idx, activity_id, subject_id in all_windows:
                activity_name = self.activity_map.get(activity_id, f"activity_{activity_id}")
                subject_dir = windows_dir / f"subject{subject_id}"
                subject_dir.mkdir(exist_ok=True)

                filename = f"subject{subject_id}_window{window_idx:04d}_activity{activity_id}_{activity_name}.csv"
                filepath = subject_dir / filename

                window_data_copy = window_data.copy()
                window_data_copy['subject_id'] = subject_id
                window_data_copy['window_index'] = window_idx
                window_data_copy['activity_name'] = activity_name
                window_data_copy.to_csv(filepath, index=False)

            self._save_summary(output_path, len(all_windows))
            logger.info(f"✓ Total windows: {len(all_windows)}")
            logger.info(f"✓ Output: {windows_dir}")

            return str(windows_dir)

    def _create_sliding_windows(self, data: pd.DataFrame, subject_id: str,
                               window_size: int, step_size: int) -> List[Tuple[pd.DataFrame, int, int, str]]:
        """
        Create sliding windows from the data.

        Returns:
            List of tuples: (window_data, window_index, activity_id, subject_id)
        """
        windows = []

        # Group by activity to maintain activity consistency within windows
        for activity_id, activity_group in data.groupby('activity_id'):
            if len(activity_group) < window_size:
                logger.debug(f"Skipping activity {activity_id} for subject {subject_id}: "
                           f"insufficient data ({len(activity_group)} samples)")
                continue

            # Reset index for proper slicing
            activity_group = activity_group.reset_index(drop=True)

            # Create sliding windows within each activity segment
            for start_idx in range(0, len(activity_group) - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window_data = activity_group.iloc[start_idx:end_idx].copy()

                # Verify single activity in window
                unique_activities = window_data['activity_id'].unique()
                if len(unique_activities) == 1:
                    window_index = len(windows)
                    windows.append((window_data, window_index, int(activity_id), subject_id))
                else:
                    logger.debug(f"Skipping mixed window for subject {subject_id}")
                    continue

        return windows

    def _split_one_subject_out(self, windows: List[Tuple], test_subject_id: str) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Split windows into train and test using one-subject-out method.

        Args:
            windows: List of tuples (window_data, window_idx, activity_id, subject_id)
            test_subject_id: Subject ID to use for test set

        Returns:
            Tuple of (train_windows, test_windows)
        """
        train_windows = []
        test_windows = []

        for window in windows:
            window_data, window_idx, activity_id, subject_id = window

            if subject_id == test_subject_id:
                test_windows.append(window)
            else:
                train_windows.append(window)

        logger.info(f"  Train: {len(train_windows)} windows (excluding subject {test_subject_id})")
        logger.info(f"  Test: {len(test_windows)} windows (subject {test_subject_id} only)")

        return train_windows, test_windows

    def _save_split_windows(self, windows: List[Tuple], output_path: Path, split_name: str):
        """
        Save windows to train or test split with activity-based folder structure.

        Args:
            windows: List of tuples (window_data, window_idx, activity_id, subject_id)
            output_path: Base output path (e.g., train-test-splits/S101_out)
            split_name: 'train' or 'test'
        """
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)

        activity_counts = {}

        for window_data, window_idx, activity_id, subject_id in tqdm(windows, desc=f"  Saving {split_name} windows"):
            activity_name = self.activity_map.get(activity_id, f"activity_{activity_id}")

            # Create activity folder
            activity_dir = split_dir / activity_name
            activity_dir.mkdir(exist_ok=True)

            # Create filename
            filename = f"subject{subject_id}_window{window_idx:04d}_activity{activity_id}_{activity_name}.csv"
            filepath = activity_dir / filename

            # Add metadata columns
            window_data_copy = window_data.copy()
            window_data_copy['subject_id'] = subject_id
            window_data_copy['window_index'] = window_idx
            window_data_copy['activity_name'] = activity_name

            # Save to CSV
            window_data_copy.to_csv(filepath, index=False)

            # Update counts
            activity_counts[activity_name] = activity_counts.get(activity_name, 0) + 1

        logger.info(f"  Saved {len(windows)} {split_name} windows")
        for activity, count in sorted(activity_counts.items()):
            logger.info(f"    {activity}: {count}")

    def _save_split_summary(self, train_windows: List[Tuple], test_windows: List[Tuple],
                           output_path: Path, test_subject_id: str):
        """Save preprocessing summary with train/test split information."""
        summary_file = output_path / f'preprocessing_summary_S{test_subject_id}_out.txt'

        # Count activities for train
        train_activity_counts = {}
        for _, _, activity_id, _ in train_windows:
            activity_name = self.activity_map.get(activity_id, f"activity_{activity_id}")
            train_activity_counts[activity_name] = train_activity_counts.get(activity_name, 0) + 1

        # Count activities for test
        test_activity_counts = {}
        for _, _, activity_id, _ in test_windows:
            activity_name = self.activity_map.get(activity_id, f"activity_{activity_id}")
            test_activity_counts[activity_name] = test_activity_counts.get(activity_name, 0) + 1

        window_size = self.config['preprocessing']['window_size']
        step_size = self.config['preprocessing'].get('step_size', window_size)
        sampling_rate = self.config['preprocessing']['sampling_rate']
        overlap_pct = int((1 - step_size / window_size) * 100)

        with open(summary_file, 'w') as f:
            f.write("PAMAP2 Preprocessing Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Split method: One-Subject-Out\n")
            f.write(f"Test subject: {test_subject_id}\n\n")
            f.write(f"Total windows: {len(train_windows) + len(test_windows)}\n")
            f.write(f"Train windows: {len(train_windows)}\n")
            f.write(f"Test windows: {len(test_windows)}\n\n")

            f.write(f"Window size: {window_size} samples ({window_size/sampling_rate:.2f} seconds at {sampling_rate}Hz)\n")
            f.write(f"Step size: {step_size} samples ({overlap_pct}% overlap)\n")
            f.write(f"Sampling rate: {sampling_rate} Hz\n\n")
            f.write("Sensors: Hand, Chest, Ankle\n")
            f.write("Data: acc16 (3-axis), gyro (3-axis), mag (3-axis)\n\n")

            f.write("Train windows per activity:\n")
            for activity, count in sorted(train_activity_counts.items()):
                f.write(f"  {activity}: {count}\n")

            f.write("\nTest windows per activity:\n")
            for activity, count in sorted(test_activity_counts.items()):
                f.write(f"  {activity}: {count}\n")

        logger.info(f"  Saved summary to {summary_file}")

    def _save_summary(self, output_path: Path, total_windows: int):
        """Save preprocessing summary."""
        summary_file = output_path / 'preprocessing_summary.txt'

        with open(summary_file, 'w') as f:
            f.write("PAMAP2 Preprocessing Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total windows: {total_windows}\n\n")

            window_size = self.config['preprocessing']['window_size']
            step_size = self.config['preprocessing'].get('step_size', window_size)
            sampling_rate = self.config['preprocessing']['sampling_rate']
            overlap_pct = int((1 - step_size / window_size) * 100)

            f.write(f"Window size: {window_size} samples ({window_size/sampling_rate:.2f} seconds at {sampling_rate}Hz)\n")
            f.write(f"Step size: {step_size} samples ({overlap_pct}% overlap)\n")
            f.write(f"Sampling rate: {sampling_rate} Hz\n\n")
            f.write("Sensors: Hand, Chest, Ankle\n")
            f.write("Data: acc16 (3-axis), gyro (3-axis), mag (3-axis)\n")

        logger.info(f"  Saved summary to {summary_file}")

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        PAMAP2 specific feature extraction.
        Delegates to PAMAP2FeatureExtractor in features.py

        Args:
            windows_dir: Path to preprocessed windows directory
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved features file
        """
        from .features import PAMAP2FeatureExtractor

        extractor = PAMAP2FeatureExtractor(self.config)
        return extractor.extract_features(windows_dir, output_dir)
