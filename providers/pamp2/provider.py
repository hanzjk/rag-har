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
        6. Save windows as CSV files

        Args:
            output_dir: Directory to save preprocessed windows

        Returns:
            Path to saved windows directory
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

        total_windows = 0
        for subject_id, data in tqdm(subjects_clean.items(), desc="Processing subjects"):
            windows = self._create_sliding_windows(data, subject_id, window_size, step_size)

            if windows:
                self._save_windows(windows, subject_id, output_path)
                total_windows += len(windows)
            else:
                logger.warning(f"No valid windows created for subject {subject_id}")

        # Save summary
        self._save_summary(output_path, total_windows)

        logger.info("")
        logger.info(f"✓ Total windows created: {total_windows}")
        logger.info(f"✓ Output: {output_path}")

        return str(output_path)

    def _create_sliding_windows(self, data: pd.DataFrame, subject_id: str,
                               window_size: int, step_size: int) -> List[Tuple[pd.DataFrame, int, int]]:
        """
        Create sliding windows from the data.

        Returns:
            List of tuples: (window_data, window_index, activity_id)
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
                    windows.append((window_data, window_index, int(activity_id)))
                else:
                    logger.debug(f"Skipping mixed window for subject {subject_id}")
                    continue

        return windows

    def _save_windows(self, windows: List[Tuple[pd.DataFrame, int, int]],
                     subject_id: str, output_path: Path):
        """Save each window as a separate CSV file."""
        subject_dir = output_path / f"subject{subject_id}"
        subject_dir.mkdir(parents=True, exist_ok=True)

        for window_data, window_idx, activity_id in windows:
            activity_name = self.activity_map.get(activity_id, f"activity_{activity_id}")

            # Create filename
            filename = f"subject{subject_id}_window{window_idx:04d}_activity{activity_id}_{activity_name}.csv"
            filepath = subject_dir / filename

            # Add metadata columns
            window_data_copy = window_data.copy()
            window_data_copy['subject_id'] = subject_id
            window_data_copy['window_index'] = window_idx
            window_data_copy['activity_name'] = activity_name

            # Save to CSV
            window_data_copy.to_csv(filepath, index=False)

        logger.debug(f"Saved {len(windows)} windows for subject {subject_id}")

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
