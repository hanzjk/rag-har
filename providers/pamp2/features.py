"""
PAMAP2 Feature Extraction - Extracts statistical features from preprocessed windows.
Handles 3 IMU sensors (hand, chest, ankle) with accelerometer, gyroscope, and magnetometer data.
"""

import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Add parent and common directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.feature_utils import FeatureExtractorUtils

logger = logging.getLogger(__name__)


class PAMAP2FeatureExtractor:
    """
    Feature extractor for PAMAP2 dataset.
    Extracts temporal segment features from 3 IMU sensors.
    """

    def __init__(self, config: Dict):
        """Initialize PAMAP2 feature extractor."""
        self.config = config
        self.dataset_name = config['dataset_name']
        self.feature_config = config.get('features', {})
        self.sampling_rate = config['preprocessing'].get('sampling_rate', 100)

        # Initialize feature utility
        self.feature_utils = FeatureExtractorUtils()

        # Sensor locations
        self.sensor_locations = ['hand', 'chest', 'ankle']

        # Sensor types per location
        self.sensor_types = ['acc16', 'gyro', 'mag']

        logger.info(f"Initialized PAMAP2 feature extractor")

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        Extract features from preprocessed windows.

        Args:
            windows_dir: Path to preprocessed windows directory
            output_dir: Directory to save features and descriptions

        Returns:
            Path to saved descriptions directory
        """
        logger.info("="*60)
        logger.info("PAMAP2 FEATURE EXTRACTION")
        logger.info("="*60)
        logger.info(f"Windows directory: {windows_dir}")
        logger.info("")

        windows_path = Path(windows_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all window CSV files
        all_files = sorted(windows_path.glob('**/*.csv'))

        logger.info(f"Found {len(all_files)} windows")
        logger.info("")

        # Create output directory
        desc_dir = output_path / 'descriptions'
        desc_dir.mkdir(parents=True, exist_ok=True)

        # Process all windows
        logger.info("Processing windows...")
        for file_path in tqdm(all_files, desc="Extracting features", unit="file"):
            try:
                # Load window data
                df = pd.read_csv(file_path)

                # Create temporal segments
                df_whole, df_start, df_mid, df_end = self._create_temporal_segments(df)

                # Generate description
                description = self._generate_description(df_whole, df_start, df_mid, df_end, df)

                # Save description
                subject_id = df['subject_id'].iloc[0] if 'subject_id' in df.columns else 'unknown'

                # Generate output filename
                activity_name = df['activity_name'].iloc[0] if 'activity_name' in df.columns else 'unknown'
                window_idx = df['window_index'].iloc[0] if 'window_index' in df.columns else '0'
                activity_id = df['activity_id'].iloc[0] if 'activity_id' in df.columns else '0'

                safe_activity_name = activity_name.replace(" ", "_").replace("/", "_")
                out_filename = f"window_{int(window_idx)}_activity_{safe_activity_name}_stats.txt"
                out_file = desc_dir / out_filename

                # Write description
                with open(out_file, 'w') as f:
                    f.write(description)

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                continue

        logger.info("")
        logger.info(f"✓ Feature extraction complete")
        logger.info(f"✓ Output: {desc_dir}")

        return str(desc_dir)

    def _create_temporal_segments(self, df: pd.DataFrame):
        """Create temporal segments from window."""
        n_rows = len(df)
        segment_size = n_rows // 3

        # Get all sensor columns (excluding metadata)
        sensor_cols = []
        for location in self.sensor_locations:
            for sensor_type in self.sensor_types:
                for axis in ['x', 'y', 'z']:
                    col = f'{location}_{sensor_type}_{axis}'
                    if col in df.columns:
                        sensor_cols.append(col)

        df_whole = df[sensor_cols] if sensor_cols else df
        df_start = df[sensor_cols].iloc[:segment_size] if sensor_cols else df.iloc[:segment_size]
        df_mid = df[sensor_cols].iloc[segment_size:2*segment_size] if sensor_cols else df.iloc[segment_size:2*segment_size]
        df_end = df[sensor_cols].iloc[2*segment_size:] if sensor_cols else df.iloc[2*segment_size:]

        return df_whole, df_start, df_mid, df_end

    def _generate_description(self, df_whole, df_start, df_mid, df_end, df_full):
        """Generate description with temporal segmentation."""
        segments = {
            'Whole Segment': df_whole,
            'Start Segment': df_start,
            'Mid Segment': df_mid,
            'End Segment': df_end
        }

        description_parts = []

        for seg_name, seg_df in segments.items():
            description_parts.append(f"[{seg_name}]")

            if seg_df.empty:
                description_parts.append("  No sensor data available")
                description_parts.append("")
                continue

            # Process each sensor location
            for location in self.sensor_locations:
                location_desc = self._describe_location(seg_df, location)
                if location_desc:
                    description_parts.append(f"[{location.capitalize()} Sensor]")
                    description_parts.append(location_desc)

            description_parts.append("")

        return "\n".join(description_parts).strip()

    def _describe_location(self, seg_df, location):
        """Describe all sensors at a given location."""
        lines = []

        # Sensor metadata
        sensor_info = {
            'acc16': ('Acceleration', 'm/s²'),
            'gyro': ('Gyroscope', 'rad/s'),
            'mag': ('Magnetometer', 'μT')
        }

        axes = ['x', 'y', 'z']
        stat_names = ['mean', 'std', 'min', 'max', 'median', 'p25', 'p75']

        # Process each sensor type
        for sensor_type in self.sensor_types:
            sensor_name, sensor_unit = sensor_info[sensor_type]

            # Process each axis
            for axis_idx, axis in enumerate(axes, start=1):
                col = f'{location}_{sensor_type}_{axis}'

                if col not in seg_df.columns:
                    continue

                data = seg_df[col].dropna()
                if len(data) == 0:
                    continue

                # Compute statistics
                stats = self.feature_utils.compute_stats(
                    data,
                    self.feature_config.get('statistics', ['mean', 'std', 'min', 'max', 'median', 'p25', 'p75'])
                )

                # Format: Acceleration (axis 1, m/s²): mean=..., std=..., ...
                stat_str = ", ".join(f"{name}={stats[name]:.3f}" for name in stat_names if name in stats)
                lines.append(f"  {sensor_name} (axis {axis_idx}, {sensor_unit}): {stat_str}")

        return "\n".join(lines) if lines else None
