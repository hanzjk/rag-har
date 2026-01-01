"""
USC-HAD Dataset Provider - Loads data from the USC Human Activity Dataset.
Handles accelerometer and gyroscope data with demographic metadata.

Preprocessing for USC-HAD:
- Load data from 14 subjects with demographic metadata
- Downsample from ~100Hz to ~33Hz
- Window size: 66 samples (2 seconds at 33Hz)
- Subject-based train/test split (subjects 13, 14 for testing)
- Multiple normalization options: zscore, minmax100, revin, none
- Saves to CSV files for RAG/LLM classification
"""

import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import scipy.io
from collections import Counter
import glob
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from dataset_provider import DatasetProvider

logger = logging.getLogger(__name__)


class USCHADProvider(DatasetProvider):
    """
    Dataset provider for USC-HAD (USC Human Activity Dataset).
    Handles accelerometer and gyroscope data with demographic metadata.
    """

    def __init__(self, config_path: str):
        """Initialize USC-HAD provider."""
        super().__init__(config_path)

        # Epsilon for numerical stability
        self.EPS = 1e-8

        # Activity mapping (12 activities)
        self.activity_map = {
            1: 'walking_forward',
            2: 'walking_left',
            3: 'walking_right',
            4: 'walking_upstairs',
            5: 'walking_downstairs',
            6: 'running_forward',
            7: 'jumping',
            8: 'sitting',
            9: 'standing',
            10: 'sleeping',
            11: 'elevator_up',
            12: 'elevator_down'
        }

        # Demographic descriptors mapping
        self.demographic_descriptors = {
            'age': {
                'thresholds': [25, 35, 45],
                'labels': ['young adult', 'adult', 'middle-aged', 'mature adult']
            },
            'height': {
                'thresholds': [165, 175],  # cm
                'labels': ['shorter stature', 'average height', 'taller stature']
            },
            'weight': {
                'thresholds': [55, 70],  # kg
                'labels': ['lighter build', 'average build', 'heavier build']
            }
        }

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from .mat files.

        Returns:
            Dict mapping trial identifiers to trial data with metadata
        """
        folder_path = Path(self.config['data_source']['folder_path'])
        num_subjects = self.config['data_source'].get('num_subjects', 14)

        trials_data = {}

        for subject_num in range(1, num_subjects + 1):
            subject_dir = folder_path / f"Subject{subject_num}"

            if not subject_dir.exists():
                logger.warning(f"Directory not found for subject {subject_num}: {subject_dir}")
                continue

            mat_files = list(subject_dir.glob("*.mat"))

            for mat_file in mat_files:
                trial_data = self._load_mat_file(mat_file)
                if trial_data is not None:
                    trial_key = f"subject_{subject_num}_trial_{trial_data['trial_number']}_activity_{trial_data['activity_number']}"
                    trials_data[trial_key] = trial_data
                    logger.debug(f"Loaded {mat_file.name}: {len(trial_data['sensor_readings'])} samples")

        logger.info(f"Loaded {len(trials_data)} trials from {num_subjects} subjects")
        return trials_data

    def _load_mat_file(self, file_path: Path) -> Dict:
        """Load data from .mat file and extract metadata including demographics."""
        try:
            data = scipy.io.loadmat(str(file_path))
            sensor_readings = data['sensor_readings']

            # Downsample: take every 3rd sample (~100Hz to ~33Hz)
            sensor_readings = sensor_readings[::3]

            # Extract activity information (handle both field name variants)
            if 'activity_number' in data:
                activity_number = int(data['activity_number'][0])
            elif 'activity_numbr' in data:
                activity_number = int(data['activity_numbr'][0])
            else:
                raise KeyError("Neither 'activity_number' nor 'activity_numbr' found in data")

            activity = data['activity'][0]
            subject_number = int(data['subject'][0])
            trial_number = int(data['trial'][0])

            # Extract demographic information
            age = self._extract_numeric_value(data.get('age'), remove_suffix='')
            height = self._extract_numeric_value(data.get('height'), remove_suffix='cm')
            weight = self._extract_numeric_value(data.get('weight'), remove_suffix='kg')

            # Convert to descriptive labels
            age_descriptor = self._get_demographic_descriptor(age, 'age')
            height_descriptor = self._get_demographic_descriptor(height, 'height')
            weight_descriptor = self._get_demographic_descriptor(weight, 'weight')

            return {
                'sensor_readings': sensor_readings,
                'activity_number': activity_number,
                'activity_name': activity,
                'subject_number': subject_number,
                'trial_number': trial_number,
                'age': age,
                'height': height,
                'weight': weight,
                'age_descriptor': age_descriptor,
                'height_descriptor': height_descriptor,
                'weight_descriptor': weight_descriptor
            }
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def _extract_string_value(self, field_data):
        """Helper function to extract string value from numpy array."""
        if field_data is not None and len(field_data) > 0:
            return str(field_data[0])
        return None

    def _extract_numeric_value(self, field_data, remove_suffix=''):
        """Extract numeric value from field data."""
        if field_data is None:
            return None

        value_str = self._extract_string_value(field_data)
        if value_str:
            try:
                # Remove suffix if specified
                if remove_suffix:
                    value_str = value_str.replace(remove_suffix, '').strip()
                return float(value_str)
            except ValueError:
                return None
        return None

    def _get_demographic_descriptor(self, value, descriptor_type):
        """Convert numeric demographic values to descriptive labels."""
        if value is None or np.isnan(value):
            return f"unknown {descriptor_type}"

        thresholds = self.demographic_descriptors[descriptor_type]['thresholds']
        labels = self.demographic_descriptors[descriptor_type]['labels']

        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return labels[i]
        return labels[-1]

    def preprocess(self, output_dir: str) -> str:
        """
        USC-HAD specific preprocessing.

        Steps:
        1. Load raw data from all subjects
        2. Split into train/test by subject
        3. Compute normalization statistics from training data
        4. Apply normalization (zscore/minmax100/none/revin)
        5. Create sliding windows
        6. Save windows as CSV files with metadata

        Args:
            output_dir: Directory to save preprocessed windows

        Returns:
            Path to saved windows directory
        """
        logger.info("="*60)
        logger.info("USC-HAD PREPROCESSING")
        logger.info("="*60)
        logger.info("Dataset: USC Human Activity Dataset")
        logger.info("Sensors: Accelerometer (3-axis) + Gyroscope (3-axis)")
        logger.info(f"Downsampled to: {self.config['preprocessing']['sampling_rate']} Hz")
        logger.info(f"Window size: {self.config['preprocessing']['window_size']} samples")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        norm_method = self.config['preprocessing'].get('normalization', 'zscore')

        # Step 1: Load data
        logger.info("Step 1: Loading raw data from all subjects...")
        trials_data = self.load_raw_data()

        # Step 2: Split into train/test by subject
        logger.info("Step 2: Splitting into train/test by subject...")
        test_subjects = self.config['preprocessing'].get('test_subjects', [13, 14])
        train_trials, test_trials = self._split_by_subjects(trials_data, test_subjects)

        # Print demographic summary
        self._print_demographic_summary(train_trials, test_trials)

        # Step 3: Compute normalization statistics
        logger.info(f"Step 3: Computing normalization statistics (method: {norm_method})...")
        norm_stats = self._compute_normalization_stats(train_trials, norm_method)

        # Step 4: Apply normalization
        logger.info("Step 4: Applying normalization...")
        if norm_method in ['zscore', 'minmax100', 'none']:
            train_trials = self._apply_global_normalization(train_trials, norm_stats, norm_method)
            test_trials = self._apply_global_normalization(test_trials, norm_stats, norm_method)

        # Step 5: Create windows
        logger.info("Step 5: Creating sliding windows...")
        train_test_dir = output_path / 'train-test-splits'
        train_dir = train_test_dir / 'train'
        test_dir = train_test_dir / 'test'

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        total_train = self._create_and_save_windows(train_trials, train_dir, is_test=False, norm_method=norm_method)
        total_test = self._create_and_save_windows(test_trials, test_dir, is_test=True, norm_method=norm_method)

        # Save summary
        self._save_summary(train_dir, test_dir, output_path, test_subjects, norm_method)

        logger.info("")
        logger.info(f"✓ Total training windows: {total_train}")
        logger.info(f"✓ Total test windows: {total_test}")
        logger.info(f"✓ Output: {train_test_dir}")

        return str(train_test_dir)

    def _split_by_subjects(self, trials_data: Dict, test_subjects: list):
        """Split trials into train/test based on subject IDs."""
        train_trials = []
        test_trials = []

        for trial_key, trial_data in trials_data.items():
            if trial_data['subject_number'] in test_subjects:
                test_trials.append(trial_data)
            else:
                train_trials.append(trial_data)

        logger.info(f"  Train trials: {len(train_trials)} (subjects: all except {test_subjects})")
        logger.info(f"  Test trials: {len(test_trials)} (subjects: {test_subjects})")

        return train_trials, test_trials

    def _compute_normalization_stats(self, train_trials: list, norm_method: str):
        """Compute normalization statistics from training data."""
        if norm_method == 'none' or norm_method == 'revin':
            return None

        # Concatenate all training sensor readings
        train_concat = np.concatenate([t['sensor_readings'] for t in train_trials], axis=0)

        if norm_method == 'zscore':
            mean = np.mean(train_concat, axis=0)
            std = np.std(train_concat, axis=0, ddof=0)
            std = np.where(std < self.EPS, 1.0, std)
            logger.info(f"  Z-score normalization: mean={mean.mean():.3f}, std={std.mean():.3f}")
            return {"mean": mean, "std": std}

        elif norm_method == 'minmax100':
            mn = np.min(train_concat, axis=0)
            mx = np.max(train_concat, axis=0)
            mx = np.where((mx - mn) < self.EPS, mn + 1.0, mx)
            logger.info(f"  Min-max normalization: min={mn.mean():.3f}, max={mx.mean():.3f}")
            return {"min": mn, "max": mx}

        return None

    def _apply_global_normalization(self, trials: list, norm_stats, norm_method: str):
        """Apply global normalization to trials."""
        normalized_trials = []

        for trial in tqdm(trials, desc=f"  Normalizing ({norm_method})"):
            trial_copy = trial.copy()

            if norm_method == 'zscore':
                trial_copy['sensor_readings'] = (trial['sensor_readings'] - norm_stats['mean']) / norm_stats['std']
            elif norm_method == 'minmax100':
                rng = norm_stats['max'] - norm_stats['min']
                scaled = (trial['sensor_readings'] - norm_stats['min']) / rng
                trial_copy['sensor_readings'] = np.rint(scaled * 100).astype(int)
            elif norm_method == 'none':
                pass  # No normalization

            normalized_trials.append(trial_copy)

        return normalized_trials

    def _create_and_save_windows(self, trials: list, output_dir: Path, is_test: bool, norm_method: str) -> int:
        """Create sliding windows and save as CSV files."""
        window_size = self.config['preprocessing']['window_size']
        step_size = self.config['preprocessing'].get('step_size', window_size)  # Default non-overlapping

        total_windows = 0

        # Create subject directories
        subject_nums = set([t['subject_number'] for t in trials])
        for subject_num in subject_nums:
            (output_dir / f"Subject{subject_num}").mkdir(parents=True, exist_ok=True)

        for trial in tqdm(trials, desc=f"  Creating {'test' if is_test else 'train'} windows"):
            activity_labels = np.full(len(trial['sensor_readings']), trial['activity_number'])

            if is_test:
                windows, labels = self._create_windows_testing(trial['sensor_readings'], activity_labels, window_size)
            else:
                windows, labels = self._create_windows_training(trial['sensor_readings'], activity_labels, window_size, step_size)

            for window_idx, (window, label) in enumerate(zip(windows, labels)):
                # Apply ReVIN normalization if specified
                if norm_method == 'revin':
                    window = self._apply_revin_norm(window)

                self._save_window_csv(window, trial, window_idx, output_dir, is_test)
                total_windows += 1

        return total_windows

    def _create_windows_training(self, sensor_data, activity_labels, window_size, step_size):
        """Create overlapping windows for training data."""
        windows = []
        labels = []

        for start_idx in range(0, len(sensor_data) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = sensor_data[start_idx:end_idx]
            window_labels = activity_labels[start_idx:end_idx]

            if len(window) == window_size:
                # Use most frequent label in window
                label_counts = Counter(window_labels)
                most_frequent_label = label_counts.most_common(1)[0][0]

                windows.append(window)
                labels.append(most_frequent_label)

        return windows, labels

    def _create_windows_testing(self, sensor_data, activity_labels, window_size):
        """Create non-overlapping windows for test data with padding for mixed labels."""
        windows = []
        labels = []

        start_idx = 0
        while start_idx < len(sensor_data):
            end_idx = start_idx + window_size

            if end_idx <= len(sensor_data):
                window = sensor_data[start_idx:end_idx]
                window_labels = activity_labels[start_idx:end_idx]

                unique_labels = np.unique(window_labels)

                if len(unique_labels) == 1:
                    # All samples have same label
                    windows.append(window)
                    labels.append(unique_labels[0])
                    start_idx = end_idx
                else:
                    # Mixed labels - find first change
                    first_label = window_labels[0]
                    change_idx = None

                    for i in range(1, len(window_labels)):
                        if window_labels[i] != first_label:
                            change_idx = start_idx + i
                            break

                    if change_idx is not None:
                        partial_window = sensor_data[start_idx:change_idx]

                        if len(partial_window) > 0:
                            # Pad by repeating last sample
                            last_sample = partial_window[-1]
                            padding_needed = window_size - len(partial_window)

                            if padding_needed > 0:
                                padding = np.tile(last_sample, (padding_needed, 1))
                                window = np.vstack([partial_window, padding])
                            else:
                                window = partial_window

                            windows.append(window)
                            labels.append(first_label)
                            start_idx = change_idx
                        else:
                            start_idx = end_idx
            else:
                break

        return windows, labels

    def _apply_revin_norm(self, window):
        """Apply ReVIN normalization per window."""
        mu = window.mean(axis=0, keepdims=True)
        sd = window.std(axis=0, keepdims=True)
        sd = np.where(sd < self.EPS, 1.0, sd)
        return (window - mu) / sd

    def _save_window_csv(self, window, trial_info, window_idx, output_dir, is_test):
        """Save a single window as CSV file with metadata.
        Uses PAMAP2-style structure: split/activity_name/filename.csv
        """
        columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        df = pd.DataFrame(window, columns=columns)

        # Add metadata
        df['activity_name'] = trial_info['activity_name']
        df['activity_number'] = trial_info['activity_number']
        df['subject_number'] = trial_info['subject_number']
        df['age_descriptor'] = trial_info['age_descriptor']
        df['height_descriptor'] = trial_info['height_descriptor']
        df['weight_descriptor'] = trial_info['weight_descriptor']
        df['age'] = trial_info['age']
        df['height'] = trial_info['height']
        df['weight'] = trial_info['weight']

        # Create activity-based directory
        activity_name = trial_info['activity_name'].replace(' ', '_')
        activity_dir = output_dir / activity_name
        activity_dir.mkdir(parents=True, exist_ok=True)

        # New filename format: subject{id}_window{idx}_activity{num}_{name}.csv
        filename = f"subject{trial_info['subject_number']}_window{window_idx}_activity{trial_info['activity_number']}_{activity_name}.csv"
        file_path = activity_dir / filename

        df.to_csv(file_path, index=False)

    def _print_demographic_summary(self, train_trials, test_trials):
        """Print summary of demographic descriptors."""
        logger.info("")
        logger.info("Demographic Summary:")
        logger.info("-" * 30)

        all_trials = train_trials + test_trials
        age_counts = Counter([t['age_descriptor'] for t in all_trials])
        height_counts = Counter([t['height_descriptor'] for t in all_trials])
        weight_counts = Counter([t['weight_descriptor'] for t in all_trials])

        logger.info("Age distribution:")
        for age_desc, count in age_counts.items():
            logger.info(f"  {age_desc}: {count} trials")

        logger.info("Height distribution:")
        for height_desc, count in height_counts.items():
            logger.info(f"  {height_desc}: {count} trials")

        logger.info("Weight distribution:")
        for weight_desc, count in weight_counts.items():
            logger.info(f"  {weight_desc}: {count} trials")
        logger.info("")

    def _save_summary(self, train_dir: Path, test_dir: Path, output_path: Path, test_subjects: list, norm_method: str):
        """Save preprocessing summary."""
        summary_file = output_path / 'preprocessing_summary.txt'

        train_files = list(train_dir.rglob('*.csv'))
        test_files = list(test_dir.rglob('*.csv'))

        with open(summary_file, 'w') as f:
            f.write("USC-HAD Preprocessing Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total windows: {len(train_files) + len(test_files)}\n")
            f.write(f"Train windows: {len(train_files)}\n")
            f.write(f"Test windows: {len(test_files)}\n\n")

            window_size = self.config['preprocessing']['window_size']
            step_size = self.config['preprocessing'].get('step_size', window_size)
            sampling_rate = self.config['preprocessing']['sampling_rate']

            f.write(f"Window size: {window_size} samples ({window_size/sampling_rate:.1f} seconds at {sampling_rate}Hz)\n")
            f.write(f"Step size: {step_size} samples\n")
            f.write(f"Sampling rate: {sampling_rate} Hz (downsampled from ~100Hz)\n")
            f.write(f"Normalization: {norm_method}\n\n")
            f.write(f"Test subjects: {test_subjects}\n")
            f.write(f"Train subjects: all others (1-14 except {test_subjects})\n")

        logger.info(f"  Saved summary to {summary_file}")

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        USC-HAD specific feature extraction.
        Delegates to USCHADFeatureExtractor in features.py

        Args:
            windows_dir: Path to preprocessed windows directory
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved features file
        """
        from .features import USCHADFeatureExtractor

        extractor = USCHADFeatureExtractor(self.config)
        return extractor.extract_features(windows_dir, output_dir)
