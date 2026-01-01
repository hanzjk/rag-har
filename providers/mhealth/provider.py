"""
MHEALTH Dataset Provider - Loads data from the mHealth Activity Recognition dataset.
Handles multi-sensor data from chest, ankle (left), and arm (right lower) locations.

Preprocessing for MHEALTH:
- Load data from 10 subjects
- Remove null class (label=0, transient activities)
- Fixed window size: 200 samples (4 seconds at 50Hz)
- Configurable overlap (default: 75% = step size 50)
- Subject-based train/test split
- Saves to CSV files for RAG/LLM classification
"""

import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from dataset_provider import DatasetProvider

logger = logging.getLogger(__name__)


# Column mapping based on mHealth dataset README
MHEALTH_COLUMNS = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',  # chest accelerometer
    'ecg_lead1', 'ecg_lead2',  # ECG signals
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',  # left ankle accelerometer
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',  # left ankle gyroscope
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',  # left ankle magnetometer
    'arm_acc_x', 'arm_acc_y', 'arm_acc_z',  # right lower arm accelerometer
    'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',  # right lower arm gyroscope
    'arm_mag_x', 'arm_mag_y', 'arm_mag_z',  # right lower arm magnetometer
    'label'  # activity label
]


class MHEALTHProvider(DatasetProvider):
    """
    Dataset provider for MHEALTH (Mobile Health) Activity Recognition dataset.
    Loads data from 10 subjects with multiple body-worn sensors.
    """

    def __init__(self, config_path: str):
        """Initialize MHEALTH provider."""
        super().__init__(config_path)

        # Activity mapping
        self.activity_map = {
            0: 'null_class',  # transient activities - should be discarded
            1: 'standing_still',
            2: 'sitting_relaxing',
            3: 'lying_down',
            4: 'walking',
            5: 'climbing_stairs',
            6: 'waist_bends_forward',
            7: 'frontal_elevation_arms',
            8: 'knees_bending_crouching',
            9: 'cycling',
            10: 'jogging',
            11: 'running',
            12: 'jump_front_back'
        }

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from all subjects.

        Returns:
            Dict mapping subject IDs to DataFrames
        """
        folder_path = Path(self.config['data_source']['folder_path'])
        num_subjects = self.config['data_source'].get('num_subjects', 10)

        subjects_data = {}

        for i in range(1, num_subjects + 1):
            file_path = folder_path / f"mHealth_subject{i}.log"

            if file_path.exists():
                subject = pd.read_csv(file_path, header=None, sep='\t')
                subject.columns = MHEALTH_COLUMNS
                subject['subject_id'] = i
                subjects_data[f"subject_{i}"] = subject
                logger.info(f"Subject {i}: {len(subject)} samples")
            else:
                logger.warning(f"File not found for subject {i}: {file_path}")

        return subjects_data

    def preprocess(self, output_dir: str) -> str:
        """
        MHEALTH specific preprocessing.

        Steps:
        1. Load raw data from all subjects
        2. Remove null class (label=0, transient activities)
        3. Drop rows with missing values
        4. Create sliding windows (200 samples, configurable overlap)
        5. Split by subject (one for test, rest for train)
        6. Save windows as CSV files

        Args:
            output_dir: Directory to save preprocessed windows

        Returns:
            Path to saved windows directory
        """
        logger.info("="*60)
        logger.info("MHEALTH PREPROCESSING")
        logger.info("="*60)
        logger.info("Dataset: mHealth Activity Recognition")
        logger.info("Sensors: Chest (acc), Ankle (acc+gyro+mag), Arm (acc+gyro+mag)")
        logger.info("Window size: 200 samples (4 seconds at 50Hz)")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load data
        logger.info("Step 1: Loading raw data from all subjects...")
        subjects_data = self.load_raw_data()

        # Step 2: Remove null class
        logger.info("Step 2: Removing null class (label=0)...")
        subjects_filtered = self._remove_null_class(subjects_data)

        # Step 3: Drop missing values
        logger.info("Step 3: Dropping rows with missing values...")
        subjects_clean = self._drop_missing_values(subjects_filtered)

        # Step 4: Create sliding windows
        logger.info("Step 4: Creating sliding windows...")
        segments_dir = output_path / 'mhealth_segments'
        segments_dir.mkdir(exist_ok=True)

        window_size = self.config['preprocessing']['window_size']
        step_size = self.config['preprocessing'].get('step_size', 50)

        total_windows = self._create_sliding_windows(
            subjects_clean,
            segments_dir,
            window_size,
            step_size
        )

        # Step 5: Split by subject
        logger.info("Step 5: Splitting by subject...")
        test_subject_id = self.config['preprocessing'].get('test_subject_id', 1)

        train_test_dir = output_path / 'train-test-splits'
        train_dir = train_test_dir / 'train'
        test_dir = train_test_dir / 'test'

        self._split_by_subject(segments_dir, train_dir, test_dir, test_subject_id)

        # Save summary
        self._save_summary(train_dir, test_dir, output_path, test_subject_id)

        logger.info("")
        logger.info(f"✓ Total windows created: {total_windows}")
        logger.info(f"✓ Output: {train_test_dir}")

        return str(train_test_dir)

    def _remove_null_class(self, subjects_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Remove data labeled with activity_id=0 (null class/transient activities)."""
        filtered = {}

        for subject_name, subject_df in subjects_data.items():
            original_len = len(subject_df)
            subject_filtered = subject_df[subject_df['label'] != 0].copy()
            subject_filtered = subject_filtered.reset_index(drop=True)
            filtered[subject_name] = subject_filtered

            subject_id = subject_df['subject_id'].iloc[0]
            logger.info(f"  Subject {subject_id}: Removed {original_len - len(subject_filtered)} null class samples")

        return filtered

    def _drop_missing_values(self, subjects_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Drop rows with missing values."""
        cleaned = {}

        for subject_name, subject_df in subjects_data.items():
            subject_id = subject_df['subject_id'].iloc[0]
            original_len = len(subject_df)
            subject_clean = subject_df.dropna()
            cleaned[subject_name] = subject_clean

            dropped = original_len - len(subject_clean)
            if dropped > 0:
                logger.info(f"  Subject {subject_id}: Dropped {dropped} rows with missing values")

        return cleaned

    def _create_sliding_windows(self, subjects_data: Dict[str, pd.DataFrame],
                                output_dir: Path, window_size: int, step_size: int) -> int:
        """
        Create sliding window segments from time series data.

        Args:
            subjects_data: Dict of subject DataFrames
            output_dir: Output directory for segments
            window_size: Window size in samples
            step_size: Step size in samples (0 for non-overlapping)

        Returns:
            Total number of windows created
        """
        total_windows = 0

        # Handle non-sliding window case
        if step_size == 0:
            step_size = window_size
            logger.info("  Non-sliding window mode: Creating non-overlapping windows")
        else:
            overlap_pct = int((1 - step_size / window_size) * 100)
            logger.info(f"  Sliding window mode: {overlap_pct}% overlap")

        for subject_name, subject_data in tqdm(subjects_data.items(), desc="Processing subjects"):
            subject_id = subject_data['subject_id'].iloc[0]

            # Process each activity separately
            for activity_id in sorted(subject_data['label'].unique()):
                activity_data = subject_data[subject_data['label'] == activity_id].copy()
                activity_data = activity_data.reset_index(drop=True)

                if len(activity_data) < window_size:
                    logger.debug(f"Subject {subject_id}, Activity {activity_id}: Too few samples ({len(activity_data)})")
                    continue

                win_num = 0
                # Window segmentation
                for start in range(0, len(activity_data) - window_size + 1, step_size):
                    end = start + window_size
                    window_data = activity_data.iloc[start:end].copy()

                    # Create output directory structure: activity_id/subject_id/
                    activity_name = self.activity_map.get(int(activity_id), f'activity_{int(activity_id)}')
                    win_folder = output_dir / str(int(activity_id)) / f"subject_{subject_id}"
                    win_folder.mkdir(parents=True, exist_ok=True)

                    # Save window as CSV
                    out_path = win_folder / f"window_{win_num}.csv"
                    window_data.to_csv(out_path, index=False)
                    win_num += 1
                    total_windows += 1

                logger.debug(f"  Subject {subject_id}, Activity {activity_id} ({activity_name}): {win_num} windows")

        return total_windows

    def _split_by_subject(self, source_root: Path, train_root: Path,
                         test_root: Path, test_subject_id: int):
        """
        Split data by subject: one subject for testing, rest for training.
        Uses PAMAP2-style folder structure: train/activity_name/filename.csv

        Args:
            source_root: Directory containing segmented data
            train_root: Output directory for training data
            test_root: Output directory for test data
            test_subject_id: Subject ID to use for testing (1-10)
        """
        logger.info(f"  Test subject: {test_subject_id}, Train subjects: all others")

        total_train = 0
        total_test = 0

        # For each activity directory
        for activity_dir in sorted(source_root.iterdir()):
            if not activity_dir.is_dir():
                continue

            activity_id = int(activity_dir.name)
            activity_name = self.activity_map.get(activity_id, f'activity_{activity_id}')

            # For each subject directory within the activity
            for subject_dir in sorted(activity_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue

                # Extract subject ID from directory name (format: "subject_X")
                try:
                    subject_id = int(subject_dir.name.split('_')[1])
                except (IndexError, ValueError):
                    logger.warning(f"Could not extract subject ID from {subject_dir.name}")
                    continue

                # Get all window files for this subject
                window_files = list(subject_dir.glob("window_*.csv"))

                if len(window_files) == 0:
                    continue

                # Determine if this subject goes to train or test
                if subject_id == test_subject_id:
                    dest_root = test_root
                    total_test += len(window_files)
                else:
                    dest_root = train_root
                    total_train += len(window_files)

                # Create destination directory using activity name (not nested subject folder)
                dest_dir = dest_root / activity_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Copy and rename window files with metadata in filename
                for window_file in window_files:
                    import shutil
                    # Extract window number from filename
                    window_num = window_file.stem.split('_')[1]
                    # New filename format: subject{id}_window{num}_activity{id}_{name}.csv
                    new_filename = f"subject{subject_id}_window{window_num}_activity{activity_id}_{activity_name}.csv"
                    dest_file = dest_dir / new_filename
                    shutil.copy2(window_file, dest_file)

        logger.info(f"  Training windows: {total_train}")
        logger.info(f"  Test windows: {total_test}")

    def _save_summary(self, train_dir: Path, test_dir: Path, output_path: Path, test_subject_id: int):
        """Save preprocessing summary."""
        summary_file = output_path / 'preprocessing_summary.txt'

        # Count windows
        train_files = list(train_dir.rglob('*.csv'))
        test_files = list(test_dir.rglob('*.csv'))

        with open(summary_file, 'w') as f:
            f.write("MHEALTH Preprocessing Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total windows: {len(train_files) + len(test_files)}\n")
            f.write(f"Train windows: {len(train_files)}\n")
            f.write(f"Test windows: {len(test_files)}\n\n")

            window_size = self.config['preprocessing']['window_size']
            step_size = self.config['preprocessing'].get('step_size', 50)
            overlap_pct = int((1 - step_size / window_size) * 100)

            f.write(f"Window size: {window_size} samples (4 seconds at 50Hz)\n")
            f.write(f"Step size: {step_size} samples ({overlap_pct}% overlap)\n")
            f.write(f"Sampling rate: 50 Hz\n\n")

            f.write(f"Test subject: {test_subject_id}\n")
            f.write(f"Train subjects: all others (1-10 except {test_subject_id})\n")

        logger.info(f"  Saved summary to {summary_file}")

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        MHEALTH specific feature extraction.
        Delegates to MHEALTHFeatureExtractor in features.py

        Args:
            windows_dir: Path to preprocessed windows directory
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved features file
        """
        from .features import MHEALTHFeatureExtractor

        extractor = MHEALTHFeatureExtractor(self.config)
        return extractor.extract_features(windows_dir, output_dir)
