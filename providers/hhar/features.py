"""
HHAR Feature Extraction
Extracts statistical features from preprocessed windows with temporal segmentation.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm

from ..common.feature_utils import FeatureExtractorUtils

logger = logging.getLogger(__name__)


class HHARFeatureExtractor:
    """
    Feature extractor for HHAR dataset.
    Knows how to extract features from HHAR's column structure:
    - acc_x, acc_y, acc_z (accelerometer)
    - gyro_x, gyro_y, gyro_z (gyroscope)
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

        # HHAR sensors
        self.sensor_prefixes = ['acc', 'gyro']

        self.utils = FeatureExtractorUtils()

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        Extract features from all windows (train and test).

        Args:
            windows_dir: Path to preprocessed windows directory
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved descriptions directory
        """
        logger.info("="*60)
        logger.info("HHAR FEATURE EXTRACTION")
        logger.info("="*60)
        logger.info("Sensors: Accelerometer and Gyroscope")
        logger.info("Temporal segmentation: whole, start, mid, end")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create descriptions directory
        descriptions_dir = output_path / 'descriptions'
        descriptions_dir.mkdir(exist_ok=True)

        # Find all window CSV files (excluding split_info.csv)
        windows_path = Path(windows_dir)
        all_windows = []

        for csv_file in windows_path.rglob("*.csv"):
            if csv_file.name != 'split_info.csv':
                all_windows.append(csv_file)

        logger.info(f"Found {len(all_windows)} window files")

        # Process all windows
        self._process_windows(all_windows, windows_dir, descriptions_dir)

        # Copy split_info.csv if it exists
        split_info_src = windows_path / 'split_info.csv'
        split_info_dst = descriptions_dir / 'split_info.csv'

        if split_info_src.exists():
            import shutil
            shutil.copy2(split_info_src, split_info_dst)
            logger.info(f"Copied split_info.csv to {descriptions_dir}")

        logger.info("")
        logger.info(f"✓ Descriptions: {len(all_windows)} files in {descriptions_dir}")

        return str(descriptions_dir)

    def _process_windows(self, all_windows: List[Path], data_root: Path, out_root: Path):
        """
        Process all window CSV files and generate statistical descriptions.

        Args:
            all_windows: List of window CSV file paths
            data_root: Root directory containing window CSV files
            out_root: Output directory for statistical descriptions
        """
        for file_path in tqdm(all_windows, desc="Processing windows"):
            try:
                # Parse filename to extract metadata
                filename = file_path.name
                # Example: user_a_device_gear_1_window0001_activity0_bike.csv

                # Remove extension and split by underscore
                parts = filename.replace('.csv', '').split('_')

                # Extract information
                user = parts[1]  # 'a'
                device = '_'.join(parts[3:5])  # 'gear_1' or 'lgwatch_1'
                window_name = parts[5].replace('window', '')  # '0001'
                activity_id = parts[6].replace('activity', '')  # '0'
                activity_label = '_'.join(parts[7:]) if len(parts) > 7 else ''  # 'bike'

                # Load window data
                df = pd.read_csv(file_path)

                # Check if required columns exist
                required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing required columns in {filename}")
                    continue

                # Check minimum rows
                if len(df) < 3:
                    logger.warning(f"Too few rows ({len(df)}) in {filename}")
                    continue

                # Generate description
                description = self._generate_description(df)

                # Prepare output directory and filename
                # Maintain the same directory structure as input
                rel_path = file_path.relative_to(data_root)
                rel_dir = rel_path.parent

                out_dir = out_root / rel_dir
                out_dir.mkdir(parents=True, exist_ok=True)

                # Create output filename
                out_file = out_dir / f"user_{user}_device_{device}_activity{activity_id}_{activity_label}_window{window_name}_stat.txt"

                # Save description
                with open(out_file, "w") as f:
                    f.write(description)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

    def _generate_description(self, df: pd.DataFrame) -> str:
        """
        Generate human-readable description of sensor statistics.

        Args:
            df: Window DataFrame

        Returns:
            Description text
        """
        # Split into temporal segments
        segments = self.utils.split_temporal_segments(df)

        # Sensor metadata: (prefix, full_name, unit)
        sensor_info = [
            ("acc", "Accelerometer", "m/s²"),
            ("gyro", "Gyroscope", "rad/s"),
        ]

        axes = ["X", "Y", "Z"]
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
            description_parts.append("[Watch Sensor Data]")

            # Process each sensor type
            for prefix, sensor_name, sensor_unit in sensor_info:
                # Process each axis
                for axis in axes:
                    col_name = f'{prefix}_{axis.lower()}'

                    if col_name in segment_df.columns:
                        stats_dict = self.utils.compute_stats(segment_df[col_name], self.statistics)
                        stats_str = ', '.join([f"{k}={v:.3f}" for k, v in stats_dict.items()])
                        description_parts.append(
                            f"  {sensor_name} ({axis} axis, {sensor_unit}): {stats_str}"
                        )
                    else:
                        description_parts.append(f"  {sensor_name} ({axis} axis, {sensor_unit}): no data")

            # Add blank line after each segment except the last one
            if segment_key != 'end':
                description_parts.append("")

        return "\n".join(description_parts)
