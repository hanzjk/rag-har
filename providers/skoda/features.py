"""
Skoda Feature Extraction
Extracts statistical features from preprocessed windows with temporal segmentation.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm

from ..common.feature_utils import FeatureExtractorUtils

logger = logging.getLogger(__name__)


class SkodaFeatureExtractor:
    """
    Feature extractor for Skoda dataset.
    Knows how to extract features from Skoda's column structure:
    - 10 left arm sensors: left_sensor{3,17,19,20,23,25,26,28,30,31}_acc_{x,y,z}_calib
    - 10 right arm sensors: right_sensor{1,2,14,16,18,21,22,24,27,29}_acc_{x,y,z}_calib
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

        # Skoda sensor IDs
        self.left_sensor_ids = [3, 17, 19, 20, 23, 25, 26, 28, 30, 31]
        self.right_sensor_ids = [1, 2, 14, 16, 18, 21, 22, 24, 27, 29]

        self.utils = FeatureExtractorUtils()

    def _load_windows_from_csv(self, windows_dir: str, split_name: str) -> List[Dict]:
        """
        Load windows from CSV directory structure.

        Expected structure: train_test_splits/train/*.csv or train_test_splits/test/*.csv

        Args:
            windows_dir: Path to train_test_splits directory
            split_name: 'train' or 'test'

        Returns:
            List of window dictionaries
        """
        windows_path = Path(windows_dir) / split_name
        windows = []

        if not windows_path.exists():
            logger.warning(f"Directory not found: {windows_path}")
            return windows

        # Load all CSV files
        for csv_file in sorted(windows_path.glob("*.csv")):
            # Parse filename: labels_X.X_window_Y.csv
            filename = csv_file.stem
            parts = filename.split('_window_')

            if len(parts) == 2:
                label = parts[0]  # e.g., "labels_1.0"
                window_id = int(parts[1])

                # Load CSV data
                df = pd.read_csv(csv_file)

                windows.append({
                    'window_id': window_id,
                    'label': label,
                    'split': split_name,
                    'data': df,
                    'filename': csv_file.name
                })

        logger.info(f"Loaded {len(windows)} {split_name} windows")
        return windows

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        Extract features from all windows (train and test).

        Args:
            windows_dir: Path to preprocessed windows directory (train_test_splits)
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved descriptions directory
        """
        logger.info("="*60)
        logger.info("SKODA FEATURE EXTRACTION")
        logger.info("="*60)
        logger.info("Sensors: 20 accelerometers (10 left arm, 10 right arm)")
        logger.info("Temporal segmentation: whole, start, mid, end")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create descriptions directories
        train_desc_dir = output_path / 'train_descriptions'
        test_desc_dir = output_path / 'test_descriptions'
        train_desc_dir.mkdir(exist_ok=True)
        test_desc_dir.mkdir(exist_ok=True)

        # Process train windows
        logger.info("Processing training windows...")
        train_windows = self._load_windows_from_csv(windows_dir, 'train')
        self._process_windows(train_windows, train_desc_dir)

        # Process test windows
        logger.info("Processing test windows...")
        test_windows = self._load_windows_from_csv(windows_dir, 'test')
        self._process_windows(test_windows, test_desc_dir)

        logger.info("")
        logger.info(f"✓ Train descriptions: {len(train_windows)} files in {train_desc_dir}")
        logger.info(f"✓ Test descriptions: {len(test_windows)} files in {test_desc_dir}")

        return str(output_path)

    def _process_windows(self, windows: List[Dict], descriptions_dir: Path):
        """
        Process windows and save their descriptions.

        Args:
            windows: List of window dictionaries
            descriptions_dir: Directory to save description files
        """
        for window in tqdm(windows, desc="Extracting features", leave=False):
            window_id = window['window_id']
            label = window['label']
            df = window['data']
            filename = window['filename']

            # Extract features with temporal segmentation and generate description
            description = self._generate_description(df, label)

            # Save description using same naming as original: activity_FILENAME_stats.txt
            # Extract base name from CSV filename
            base_name = filename.replace('.csv', '')
            desc_file = descriptions_dir / f"activity_{base_name}_stats.txt"

            with open(desc_file, 'w') as f:
                f.write(description)

    def _generate_description(self, df: pd.DataFrame, label: str) -> str:
        """
        Generate human-readable description for all sensor segments.

        Args:
            df: Window DataFrame
            label: Activity label

        Returns:
            Description text
        """
        # Split into temporal segments
        segments = self.utils.split_temporal_segments(df)

        # Segment name mapping
        segment_names = {
            'whole': 'Whole Segment',
            'start': 'Start Segment',
            'middle': 'Mid Segment',
            'end': 'End Segment'
        }

        description_parts = []

        for segment_key in ['whole', 'start', 'middle', 'end']:
            segment_df = segments[segment_key]
            segment_name = segment_names[segment_key]
            description_parts.append(f"[{segment_name}]")

            # Process each arm
            for arm, sensor_ids in [('left', self.left_sensor_ids), ('right', self.right_sensor_ids)]:
                for sensor_id in sensor_ids:
                    sensor_label = f"{arm.capitalize()} sensor {sensor_id}"
                    description_parts.append(sensor_label)

                    # Process each axis
                    for axis_num, axis in enumerate(['x', 'y', 'z'], 1):
                        col_name = f'{arm}_sensor{sensor_id}_acc_{axis}_calib'

                        if col_name in segment_df.columns:
                            stats_dict = self.utils.compute_stats(segment_df[col_name], self.statistics)
                            stats_str = ', '.join([f"{k}={v:.3f}" for k, v in stats_dict.items()])
                            description_parts.append(f"Axis {axis_num}: {stats_str}")
                        else:
                            description_parts.append(f"Axis {axis_num}: no data")

                    description_parts.append("")  # Empty line after each sensor

            description_parts.append("")  # Empty line after each segment

        return "\n".join(description_parts)
