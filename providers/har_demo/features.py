"""
HAR Demo Feature Extraction
Extracts statistical features from preprocessed windows with temporal segmentation.
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


class HARDemoFeatureExtractor:
    """
    Feature extractor for HAR Demo dataset.
    Knows how to extract features from HAR Demo's column structure:
    - accel_x, accel_y, accel_z
    - gyro_x, gyro_y, gyro_z
    - mag_x, mag_y, mag_z
    """

    def __init__(self, config: Dict):
        """
        Initialize feature extractor.

        Args:
            config: Dataset configuration dict
        """
        self.config = config
        feature_config = config.get('features', {})

        self.statistics = feature_config.get('statistics', ['mean', 'std', 'min', 'max', 'median', 'p25', 'p75'])
        self.per_axis = feature_config.get('per_axis', True)

        # HAR Demo sensor columns
        self.sensor_columns = ['accel', 'gyro', 'mag']

        self.utils = FeatureExtractorUtils()

    def _load_windows_from_csv(self, windows_dir: str) -> List[Dict]:
        """
        Load windows from organized CSV directory structure.

        Expected structure: csv_windows/subject1/walking/subject1_window_0_walking.csv

        Args:
            windows_dir: Path to csv_windows directory

        Returns:
            List of window dictionaries
        """
        windows_path = Path(windows_dir)
        windows = []

        # Walk through subject/activity folders
        for subject_dir in sorted(windows_path.iterdir()):
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            for activity_dir in sorted(subject_dir.iterdir()):
                if not activity_dir.is_dir():
                    continue

                activity = activity_dir.name

                # Load all CSV files in this activity folder
                for csv_file in sorted(activity_dir.glob("*.csv")):
                    # Parse filename: subject1_window_0_walking.csv
                    filename = csv_file.stem
                    parts = filename.split('_')

                    # Extract window_id (it's after "window_")
                    window_idx = parts.index('window') + 1
                    window_id = int(parts[window_idx])

                    # Load CSV data
                    df = pd.read_csv(csv_file)

                    windows.append({
                        'window_id': window_id,
                        'activity': activity,
                        'subject_id': subject_id,
                        'data': df
                    })

        return windows

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        Extract features from all windows.

        Args:
            windows_dir: Path to preprocessed windows CSV directory
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved features file
        """
        logger.info("HAR Demo Feature Extraction")
        logger.info("Sensors: accel_x/y/z, gyro_x/y/z, mag_x/y/z")
        logger.info("Temporal segmentation: whole, start, mid, end")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load windows from CSV files
        windows = self._load_windows_from_csv(windows_dir)

        logger.info(f"Loaded {len(windows)} windows")

        # Process windows
        all_features = []
        descriptions_dir = output_path / 'descriptions'
        descriptions_dir.mkdir(exist_ok=True)

        for window in tqdm(windows, desc="Extracting features"):
            window_id = window['window_id']
            activity = window['activity']
            df = window['data']

            # Extract features with temporal segmentation
            features, description = self._extract_segmented_features(df, activity)

            all_features.append({
                'window_id': window_id,
                'activity': activity,
                'features': features
            })

            # Save description
            desc_file = descriptions_dir / f"window_{window_id}_activity_{activity}_stats.txt"
            with open(desc_file, 'w') as f:
                f.write(description)

        # Save features
        features_file = output_path / 'features.pkl'
        with open(features_file, 'wb') as f:
            pickle.dump(all_features, f)

        logger.info(f"Saved {len(all_features)} feature vectors to {features_file}")
        logger.info(f"Saved {len(all_features)} descriptions to {descriptions_dir}")

        # Save summary
        self._save_summary(all_features, output_path)

        return str(features_file)

    def _extract_segmented_features(self, df: pd.DataFrame, activity: str) -> Tuple[np.ndarray, str]:
        """
        Extract features with temporal segmentation.

        Args:
            df: Window DataFrame
            activity: Activity label

        Returns:
            Tuple of (feature_vector, description_text)
        """
        # Split into temporal segments
        segments = self.utils.split_temporal_segments(df)

        # Sensor metadata: (prefix, full_name, unit)
        sensor_metadata = {
            'accel': ('Acceleration', 'm/s²'),
            'gyro': ('Gyroscope', 'rad/s'),
            'mag': ('Magnetometer', 'μT')
        }

        # Segment name mapping
        segment_names = {
            'whole': 'Whole Segment',
            'start': 'Start Segment',
            'middle': 'Mid Segment',
            'end': 'End Segment'
        }

        all_features = []
        description_parts = []

        for segment_key in ['whole', 'start', 'middle', 'end']:
            segment_df = segments[segment_key]
            segment_name = segment_names[segment_key]
            description_parts.append(f"[{segment_name}]")
            segment_features = []

            # Process each sensor
            for prefix in self.sensor_columns:
                sensor_name, unit = sensor_metadata[prefix]
                axes = ['x', 'y', 'z']

                # Per-axis features
                if self.per_axis:
                    for axis_idx, axis in enumerate(axes, start=1):
                        col_name = f'{prefix}_{axis}'
                        if col_name in segment_df.columns:
                            stats_dict = self.utils.compute_stats(segment_df[col_name], self.statistics)
                            segment_features.extend(stats_dict.values())

                            stats_str = ', '.join([f"{k}={v:.3f}" for k, v in stats_dict.items()])
                            description_parts.append(f"  {sensor_name} (axis {axis_idx}, {unit}): {stats_str}")


            all_features.extend(segment_features)
            description_parts.append("")  # Empty line after each segment

        description_text = "\n".join(description_parts)
        return np.array(all_features), description_text

    def _save_summary(self, all_features: List[Dict], output_path: Path):
        """Save feature extraction summary."""
        summary_file = output_path / 'feature_summary.txt'

        activity_counts = {}
        for item in all_features:
            activity = item['activity']
            activity_counts[activity] = activity_counts.get(activity, 0) + 1

        feature_dim = len(all_features[0]['features']) if all_features else 0

        with open(summary_file, 'w') as f:
            f.write("HAR Demo Feature Extraction Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total windows: {len(all_features)}\n")
            f.write(f"Feature dimension: {feature_dim}\n")
            f.write(f"Temporal segments: whole, start, mid, end\n")
            f.write(f"Sensors: accel, gyro, mag\n\n")
            f.write("Windows per activity:\n")
            for activity, count in sorted(activity_counts.items()):
                f.write(f"  {activity}: {count}\n")

        logger.info(f"Saved summary to {summary_file}")
