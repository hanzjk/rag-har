"""
GOTOV Feature Extraction
Extracts statistical features from preprocessed windows with temporal segmentation.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm

from ..common.feature_utils import FeatureExtractorUtils

logger = logging.getLogger(__name__)


class GOTOVFeatureExtractor:
    """
    Feature extractor for GOTOV dataset.
    Knows how to extract features from GOTOV's column structure:
    - ankle_x, ankle_y, ankle_z
    - wrist_x, wrist_y, wrist_z
    - chest_x, chest_y, chest_z
    """

    def __init__(self, config: Dict):
        """
        Initialize feature extractor.

        Args:
            config: Dataset configuration dict
        """
        self.config = config
        feature_config = config.get('features', {})

        self.statistics = feature_config.get('statistics',
            ['mean', 'std', 'min', 'max', 'median', 'p25', 'p75', 'peak_to_peak', 'dominant_freq'])

        # GOTOV sensors (3 body locations)
        self.sensor_locations = ['ankle', 'wrist', 'chest']

        self.utils = FeatureExtractorUtils()

    def _load_windows_from_csv(self, windows_dir: str, split_name: str) -> List[Dict]:
        """
        Load windows from CSV directory structure.

        Expected structure: train_test_splits/train/*.csv or train_test_splits/test/*.csv

        Args:
            windows_dir: Path to train_test_splits directory
            split_name: 'train', 'test', or 'validation'

        Returns:
            List of window dictionaries
        """
        windows_path = Path(windows_dir) / split_name
        windows = []

        if not windows_path.exists():
            logger.warning(f"Directory not found: {windows_path}")
            return windows

        # Load all CSV files recursively (in activity subfolders)
        for csv_file in sorted(windows_path.rglob("subject*.csv")):
            # Load CSV data
            df = pd.read_csv(csv_file)

            windows.append({
                'filename': csv_file.name,
                'split': split_name,
                'data': df
            })

        logger.info(f"Loaded {len(windows)} {split_name} windows")
        return windows

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        Extract features from all windows (train, test, and validation).

        Args:
            windows_dir: Path to preprocessed windows directory (train_test_splits)
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved descriptions directory
        """
        logger.info("="*60)
        logger.info("GOTOV FEATURE EXTRACTION")
        logger.info("="*60)
        logger.info("Sensors: Ankle, Wrist, Chest (3-axis accelerometers)")
        logger.info("Temporal segmentation: whole, start, mid, end")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create descriptions directories
        train_desc_dir = output_path / 'train_descriptions'
        test_desc_dir = output_path / 'test_descriptions'
        val_desc_dir = output_path / 'validation_descriptions'

        train_desc_dir.mkdir(exist_ok=True)
        test_desc_dir.mkdir(exist_ok=True)
        val_desc_dir.mkdir(exist_ok=True)

        # Process train windows
        logger.info("Processing training windows...")
        train_windows = self._load_windows_from_csv(windows_dir, 'train')
        self._process_windows(train_windows, train_desc_dir)

        # Process test windows
        logger.info("Processing test windows...")
        test_windows = self._load_windows_from_csv(windows_dir, 'test')
        self._process_windows(test_windows, test_desc_dir)

        # Process validation windows
        logger.info("Processing validation windows...")
        val_windows = self._load_windows_from_csv(windows_dir, 'validation')
        self._process_windows(val_windows, val_desc_dir)

        logger.info("")
        logger.info(f"✓ Train descriptions: {len(train_windows)} files in {train_desc_dir}")
        logger.info(f"✓ Test descriptions: {len(test_windows)} files in {test_desc_dir}")
        logger.info(f"✓ Validation descriptions: {len(val_windows)} files in {val_desc_dir}")

        return str(output_path)

    def _process_windows(self, windows: List[Dict], descriptions_dir: Path):
        """
        Process windows and save their descriptions.

        Args:
            windows: List of window dictionaries
            descriptions_dir: Directory to save description files
        """
        for window in tqdm(windows, desc="Extracting features", leave=False):
            df = window['data']
            filename = window['filename']

            # Extract features with temporal segmentation and generate description
            description = self._generate_description(df)

            # Parse filename: subject{id}_window{num}_activity_{name}.csv
            # Extract window number and activity name
            parts = filename.replace('.csv', '').split('_', 3)
            if len(parts) >= 4:
                window_num = parts[1].replace('window', '')
                activity_name = parts[3]  # Everything after "activity_"

                # Output format: window_{num}_activity_{name}_stats.txt
                desc_file = descriptions_dir / f"window_{window_num}_activity_{activity_name}_stats.txt"
            else:
                # Fallback for unexpected format
                desc_file = descriptions_dir / f"{filename.replace('.csv', '')}_stats.txt"

            with open(desc_file, 'w') as f:
                f.write(description)

    def _generate_description(self, df: pd.DataFrame) -> str:
        """
        Generate human-readable description for all sensor segments.

        Args:
            df: Window DataFrame

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

        axis_names = ['X', 'Y', 'Z']
        description_parts = []

        for segment_key in ['whole', 'start', 'middle', 'end']:
            segment_df = segments[segment_key]
            segment_name = segment_names[segment_key]
            description_parts.append(f"[{segment_name}]")

            # Process each sensor location
            for sensor in self.sensor_locations:
                sensor_label = f"{sensor.capitalize()} Sensor"
                description_parts.append(sensor_label)

                # Process each axis
                for axis in axis_names:
                    col_name = f'{sensor}_{axis.lower()}'

                    if col_name in segment_df.columns:
                        stats_dict = self.utils.compute_stats(segment_df[col_name], self.statistics)
                        stats_str = ', '.join([f"{k}={v:.3f}" for k, v in stats_dict.items()])
                        description_parts.append(f"Axis {axis}: {stats_str}")
                    else:
                        description_parts.append(f"Axis {axis}: no data")

                # Empty line after each sensor
                description_parts.append("")

            # Empty line after each segment (except last)
            if segment_key != 'end':
                description_parts.append("")

        return "\n".join(description_parts)
