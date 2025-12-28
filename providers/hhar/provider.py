"""
HHAR Dataset Provider - Loads data from the Heterogeneity Dataset for Human Activity Recognition.
Handles smartwatch sensor data (accelerometer and gyroscope).

Preprocessing for HHAR:
- Load and merge accelerometer and gyroscope data
- Fixed window size: 200 samples (2 seconds at 100Hz)
- 50% overlap (step=100)
- User-based train/test split
- Saves to CSV files for RAG/LLM classification
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import logging
from tqdm import tqdm
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from dataset_provider import DatasetProvider

logger = logging.getLogger(__name__)


class HHARProvider(DatasetProvider):
    """
    Dataset provider for HHAR (Heterogeneity Dataset for Human Activity Recognition).
    Loads smartwatch sensor data from Watch_accelerometer.csv and Watch_gyroscope.csv
    """

    def __init__(self, config_path: str):
        """Initialize HHAR provider."""
        super().__init__(config_path)

        # Activity labels for HHAR (excluding null)
        self.activity_labels = [
            'bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'
        ]

        # Map activity names to numeric IDs
        self.activity_to_id = {activity: idx for idx, activity in enumerate(self.activity_labels)}

        # Users in the dataset
        self.users = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and merge accelerometer and gyroscope data from CSV files.

        Returns:
            Dict with single key 'merged' containing merged sensor data
        """
        data_dir = self.config['data_source']['data_dir']

        acc_file = Path(data_dir) / "Watch_accelerometer.csv"
        gyro_file = Path(data_dir) / "Watch_gyroscope.csv"

        logger.info(f"Loading accelerometer data from: {acc_file}")
        acc_data = pd.read_csv(acc_file)

        logger.info(f"Loading gyroscope data from: {gyro_file}")
        gyro_data = pd.read_csv(gyro_file)

        # Merge the data
        logger.info("Merging accelerometer and gyroscope data...")
        merged_data = self._preprocess_and_merge_data(acc_data, gyro_data)

        return {'merged': merged_data}

    def _preprocess_and_merge_data(self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess and merge accelerometer and gyroscope data.

        Steps:
        1. Filter out null activities
        2. Keep only valid activities
        3. Merge on Creation_Time, User, and Device
        4. Handle missing values

        Args:
            acc_data: Accelerometer DataFrame
            gyro_data: Gyroscope DataFrame

        Returns:
            Merged DataFrame
        """
        logger.info("Preprocessing accelerometer data...")
        # Filter accelerometer data
        acc_filtered = acc_data[
            (acc_data['gt'] != 'null') &
            (acc_data['gt'].isin(self.activity_labels))
        ].copy()

        # Rename columns to distinguish accelerometer data
        acc_filtered = acc_filtered.rename(columns={
            'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'
        })

        logger.info("Preprocessing gyroscope data...")
        # Filter gyroscope data
        gyro_filtered = gyro_data[
            (gyro_data['gt'] != 'null') &
            (gyro_data['gt'].isin(self.activity_labels))
        ].copy()

        # Rename columns to distinguish gyroscope data
        gyro_filtered = gyro_filtered.rename(columns={
            'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'
        })

        logger.info("Merging sensor data...")
        # Merge on Creation_Time, User, and Device
        merged_data = pd.merge(
            acc_filtered[['Creation_Time', 'User', 'Device', 'gt', 'acc_x', 'acc_y', 'acc_z']],
            gyro_filtered[['Creation_Time', 'User', 'Device', 'gt', 'gyro_x', 'gyro_y', 'gyro_z']],
            on=['Creation_Time', 'User', 'Device', 'gt'],
            how='inner'  # Only keep rows where both sensors have data
        )

        logger.info(f"  Original accelerometer samples: {len(acc_filtered)}")
        logger.info(f"  Original gyroscope samples: {len(gyro_filtered)}")
        logger.info(f"  Merged samples: {len(merged_data)}")

        # Sort by user, device, and timestamp
        merged_data = merged_data.sort_values(['User', 'Device', 'Creation_Time'])

        # Drop any remaining missing values
        initial_len = len(merged_data)
        merged_data = merged_data.dropna()
        final_len = len(merged_data)

        if initial_len != final_len:
            logger.info(f"  Dropped {initial_len - final_len} rows with missing values")

        # Add numeric activity ID
        merged_data['activity_id'] = merged_data['gt'].map(self.activity_to_id)

        return merged_data

    def preprocess(self, output_dir: str) -> str:
        """
        HHAR specific preprocessing.

        Steps:
        1. Load and merge accelerometer and gyroscope data
        2. Split users into train/test sets
        3. Create sliding windows (200 samples, 50% overlap)
        4. Save windows as CSV files

        Args:
            output_dir: Directory to save preprocessed windows

        Returns:
            Path to saved windows directory
        """
        logger.info("="*60)
        logger.info("HHAR PREPROCESSING")
        logger.info("="*60)
        logger.info("Dataset: Heterogeneity Dataset for Human Activity Recognition")
        logger.info("Sensors: Accelerometer and Gyroscope from smartwatches")
        logger.info("Window size: 200 samples (2 seconds at 100Hz)")
        logger.info("Overlap: 50% (step=100)")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load and merge data
        logger.info("Step 1: Loading and merging sensor data...")
        data = self.load_raw_data()
        merged_data = data['merged']

        # Step 2: Split users into train/test
        logger.info("Step 2: Splitting users into train/test...")
        test_ratio = self.config['preprocessing'].get('test_ratio', 0.2)
        test_user = self.config['preprocessing'].get('test_user', None)

        if test_user:
            logger.info(f"  Using manual test user: {test_user}")
            train_users = [user for user in self.users if user != test_user]
            test_users = [test_user]
        else:
            train_users, test_users = self._split_users_train_test(merged_data, test_ratio)

        # Step 3: Create sliding windows
        logger.info("Step 3: Creating sliding windows...")
        window_size = self.config['preprocessing']['window_size']
        overlap = self.config['preprocessing'].get('overlap', 0.5)
        step_size = int(window_size * (1 - overlap))

        logger.info(f"  Window size: {window_size}, Step size: {step_size}")

        # Get unique devices
        devices = merged_data['Device'].unique()
        logger.info(f"  Found devices: {list(devices)}")

        # Create windows for each user-device combination
        all_windows = []

        for user in tqdm(self.users, desc="Processing users"):
            for device in devices:
                # Create windows for this user-device combination
                windows = self._create_sliding_windows(
                    merged_data, user, device, window_size, step_size
                )

                if windows:
                    # Determine split
                    split = 'test' if user in test_users else 'train'

                    # Save windows
                    self._save_windows_as_csv(windows, user, device, split, output_path)

                    # Track windows for split information
                    for window_data, window_idx, activity_name, activity_id in windows:
                        all_windows.append({
                            'user': user,
                            'device': device,
                            'window_idx': window_idx,
                            'activity_name': activity_name,
                            'activity_id': activity_id,
                            'split': split,
                            'filename': f"user_{user}_device_{device}_window{window_idx:04d}_activity{activity_id}_{activity_name}.csv"
                        })

        # Save split information
        split_info = pd.DataFrame(all_windows)
        split_info.to_csv(output_path / 'split_info.csv', index=False)

        # Save summary
        self._save_summary(split_info, output_path)

        logger.info("")
        logger.info(f"✓ Total windows: {len(all_windows)}")
        logger.info(f"✓ Train windows: {len(split_info[split_info['split'] == 'train'])}")
        logger.info(f"✓ Test windows: {len(split_info[split_info['split'] == 'test'])}")
        logger.info(f"✓ Output: {output_path}")

        return str(output_path)

    def _split_users_train_test(self, merged_data: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
        """
        Split users into train and test sets with balanced data distribution.

        Args:
            merged_data: The merged sensor data to analyze user distributions
            test_ratio: Ratio of users for test set (0.2 = 20%)

        Returns:
            Tuple of (train_users, test_users)
        """
        # Analyze data per user
        user_counts = merged_data.groupby('User').size()
        logger.info("  User data distribution:")
        for user in self.users:
            count = user_counts.get(user, 0)
            logger.info(f"    User {user}: {count} samples")

        # Sort users by amount of data (descending)
        users_by_data = [(user, user_counts.get(user, 0)) for user in self.users]
        users_by_data.sort(key=lambda x: x[1], reverse=True)

        # Select users with substantial data for test set
        min_samples_for_test = 50000  # Minimum samples to be eligible for test set
        eligible_users = [user for user, count in users_by_data if count >= min_samples_for_test]

        if len(eligible_users) == 0:
            logger.warning("  No users have enough data for test set, using all users")
            eligible_users = self.users

        logger.info(f"  Eligible users for test set (≥{min_samples_for_test} samples): {eligible_users}")

        # Select test users from eligible ones
        random.shuffle(eligible_users)
        num_test_users = max(1, int(len(eligible_users) * test_ratio))
        test_users = eligible_users[:num_test_users]
        train_users = [user for user in self.users if user not in test_users]

        logger.info(f"  Selected test users ({num_test_users}): {test_users}")
        logger.info(f"  Train users ({len(train_users)}): {train_users}")

        return train_users, test_users

    def _create_sliding_windows(self, data: pd.DataFrame, user: str, device: str,
                                window_size: int, step_size: int) -> List[Tuple[pd.DataFrame, int, str, int]]:
        """
        Create sliding windows from the merged data for a specific user and device.

        Args:
            data: Merged sensor data
            user: User ID
            device: Device name
            window_size: Window size in samples
            step_size: Step size in samples

        Returns:
            List of tuples: (window_data, window_index, activity_name, activity_id)
        """
        windows = []

        # Filter data for this user and device
        user_device_data = data[(data['User'] == user) & (data['Device'] == device)].copy()

        if len(user_device_data) == 0:
            return windows

        # Group by activity to maintain activity consistency within windows
        for activity, activity_group in user_device_data.groupby('gt'):
            if len(activity_group) < window_size:
                logger.debug(f"Skipping activity {activity} for user {user}, device {device}: insufficient data ({len(activity_group)} samples)")
                continue

            # Reset index for this activity group
            activity_group = activity_group.reset_index(drop=True)

            # Create sliding windows within each activity segment
            for start_idx in range(0, len(activity_group) - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window_data = activity_group.iloc[start_idx:end_idx].copy()

                # Ensure the window has consistent activity
                unique_activities = window_data['gt'].unique()
                if len(unique_activities) == 1:
                    window_index = len(windows)
                    activity_id = self.activity_to_id[activity]
                    windows.append((window_data, window_index, activity, activity_id))

        return windows

    def _save_windows_as_csv(self, windows: List[Tuple[pd.DataFrame, int, str, int]],
                            user: str, device: str, split: str, output_dir: Path):
        """
        Save each window as a separate CSV file.

        Args:
            windows: List of window tuples
            user: User ID
            device: Device name
            split: 'train' or 'test'
            output_dir: Output directory
        """
        user_device_dir = output_dir / split / f"user_{user}" / f"device_{device}"
        user_device_dir.mkdir(parents=True, exist_ok=True)

        window_size = self.config['preprocessing']['window_size']
        overlap = self.config['preprocessing'].get('overlap', 0.5)
        step_size = int(window_size * (1 - overlap))

        for window_data, window_idx, activity_name, activity_id in windows:
            # Create filename
            filename = f"user_{user}_device_{device}_window{window_idx:04d}_activity{activity_id}_{activity_name}.csv"
            filepath = user_device_dir / filename

            # Add metadata columns
            window_data_copy = window_data.copy()
            window_data_copy['window_index'] = window_idx
            window_data_copy['activity_name'] = activity_name
            window_data_copy['window_size'] = window_size
            window_data_copy['step_size'] = step_size
            window_data_copy['overlap'] = overlap

            # Save to CSV
            window_data_copy.to_csv(filepath, index=False)

        if windows:
            logger.debug(f"  Saved {len(windows)} windows for user {user}, device {device}")

    def _save_summary(self, split_info: pd.DataFrame, output_path: Path):
        """Save preprocessing summary."""
        summary_file = output_path / 'preprocessing_summary.txt'

        train_info = split_info[split_info['split'] == 'train']
        test_info = split_info[split_info['split'] == 'test']

        with open(summary_file, 'w') as f:
            f.write("HHAR Preprocessing Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total windows: {len(split_info)}\n")
            f.write(f"Train windows: {len(train_info)}\n")
            f.write(f"Test windows: {len(test_info)}\n\n")

            window_size = self.config['preprocessing']['window_size']
            overlap = self.config['preprocessing'].get('overlap', 0.5)
            step_size = int(window_size * (1 - overlap))

            f.write(f"Window size: {window_size} samples (2 seconds at 100Hz)\n")
            f.write(f"Step size: {step_size} samples ({int(overlap*100)}% overlap)\n")
            f.write(f"Sampling rate: 100 Hz\n\n")

            f.write("Train windows per activity:\n")
            train_counts = train_info.groupby('activity_name').size()
            for activity, count in sorted(train_counts.items()):
                f.write(f"  {activity}: {count}\n")

            f.write("\nTest windows per activity:\n")
            test_counts = test_info.groupby('activity_name').size()
            for activity, count in sorted(test_counts.items()):
                f.write(f"  {activity}: {count}\n")

        logger.info(f"  Saved summary to {summary_file}")

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        HHAR specific feature extraction.
        Delegates to HHARFeatureExtractor in features.py

        Args:
            windows_dir: Path to preprocessed windows directory
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved features file
        """
        from .features import HHARFeatureExtractor

        extractor = HHARFeatureExtractor(self.config)
        return extractor.extract_features(windows_dir, output_dir)
