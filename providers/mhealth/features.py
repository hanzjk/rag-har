"""
MHEALTH Feature Extraction
Extracts statistical features from preprocessed windows with temporal segmentation.
Supports optional PCA features for dimensionality reduction.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..common.feature_utils import FeatureExtractorUtils

logger = logging.getLogger(__name__)


class MHEALTHFeatureExtractor:
    """
    Feature extractor for MHEALTH dataset.
    Knows how to extract features from MHEALTH's column structure:
    - Chest: acc_{x,y,z}
    - Ankle (left): acc_{x,y,z}, gyro_{x,y,z}, mag_{x,y,z}
    - Arm (right lower): acc_{x,y,z}, gyro_{x,y,z}, mag_{x,y,z}
    - ECG: ecg_lead1, ecg_lead2 (optional, can exclude)
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
        self.include_pca = feature_config.get('include_pca', False)
        self.pca_components = feature_config.get('pca_components', 3)

        # MHEALTH sensor locations
        self.sensor_locations = {
            'chest': ['chest_acc_x', 'chest_acc_y', 'chest_acc_z'],
            'ankle': ['ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
                     'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
                     'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z'],
            'arm': ['arm_acc_x', 'arm_acc_y', 'arm_acc_z',
                   'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
                   'arm_mag_x', 'arm_mag_y', 'arm_mag_z']
        }

        self.utils = FeatureExtractorUtils()

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
        logger.info("MHEALTH FEATURE EXTRACTION")
        logger.info("="*60)
        logger.info("Sensors: Chest (acc), Ankle (acc+gyro+mag), Arm (acc+gyro+mag)")
        logger.info("Temporal segmentation: whole, start, mid, end")
        if self.include_pca:
            logger.info(f"PCA enabled: {self.pca_components} components")
        logger.info("")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create descriptions directory
        descriptions_dir = output_path / 'descriptions'
        descriptions_dir.mkdir(exist_ok=True)

        # Find all window CSV files in train_test_splits directory
        # New format: subject*_window*_activity*.csv
        windows_path = Path(windows_dir)
        all_windows = []

        for csv_file in windows_path.rglob("subject*_window*.csv"):
            all_windows.append(csv_file)

        logger.info(f"Found {len(all_windows)} window files")

        # Process all windows
        self._process_windows(all_windows, windows_dir, descriptions_dir)

        logger.info("")
        logger.info(f"✓ Descriptions: {len(all_windows)} files in {descriptions_dir}")

        return str(descriptions_dir)

    def _process_windows(self, all_windows: List[Path], data_root: Path, out_root: Path):
        """
        Process all window CSV files and generate statistical descriptions.
        Uses PAMAP2-style output format.

        Args:
            all_windows: List of window CSV file paths
            data_root: Root directory containing window CSV files
            out_root: Output directory for statistical descriptions
        """
        for file_path in tqdm(all_windows, desc="Processing windows"):
            try:
                # New filename format: subject{id}_window{num}_activity{id}_{name}.csv
                # Extract metadata from filename
                filename = file_path.stem

                # Parse: subject{id}_window{num}_activity{id}_{name}
                # Split by '_' with limit to preserve activity name with underscores
                parts = filename.split('_', 3)  # Split into max 4 parts
                if len(parts) < 4:
                    logger.warning(f"Unexpected filename format: {filename}")
                    continue

                # parts[0] = 'subject{id}'
                # parts[1] = 'window{num}'
                # parts[2] = 'activity{id}'
                # parts[3] = '{activity_name}' (may contain underscores)

                window_num = parts[1].replace('window', '')
                activity_name = parts[3]  # Activity name with all underscores preserved

                # Load window data
                df = pd.read_csv(file_path)

                # Generate description
                description = self._generate_description(df)

                # Create output filename: window_{num}_activity_{name}_stats.txt
                out_filename = f"window_{window_num}_activity_{activity_name}_stats.txt"
                out_file = out_root / out_filename

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

            # Process Chest sensor
            description_parts.append("[Chest Sensor]")
            self._describe_sensor(segment_df, 'chest', description_parts)

            # Process Ankle sensor
            description_parts.append("[Ankle Sensor]")
            self._describe_sensor(segment_df, 'ankle', description_parts)

            # Process Arm sensor
            description_parts.append("[Arm Sensor]")
            self._describe_sensor(segment_df, 'arm', description_parts)

            # Add blank line after each segment except the last one
            if segment_key != 'end':
                description_parts.append("")

        return "\n".join(description_parts)

    def _describe_sensor(self, segment_df: pd.DataFrame, sensor_name: str, description_parts: List[str]):
        """
        Describe a sensor's statistics and add to description parts.

        Args:
            segment_df: Segment DataFrame
            sensor_name: Sensor location name ('chest', 'ankle', 'arm')
            description_parts: List to append descriptions to
        """
        sensor_cols = self.sensor_locations[sensor_name]

        # Metadata for sensor types
        sensor_metadata = {
            'acc': ('Acceleration', 'm/s²'),
            'gyro': ('Gyroscope', 'deg/s'),
            'mag': ('Magnetometer', 'local')
        }

        # Group columns by sensor type
        sensor_types = {}
        for col in sensor_cols:
            # Extract sensor type (acc, gyro, mag)
            if 'acc' in col:
                sensor_type = 'acc'
            elif 'gyro' in col:
                sensor_type = 'gyro'
            elif 'mag' in col:
                sensor_type = 'mag'
            else:
                continue

            if sensor_type not in sensor_types:
                sensor_types[sensor_type] = []
            sensor_types[sensor_type].append(col)

        # Describe each sensor type
        axis_num = 1
        for sensor_type in ['acc', 'gyro', 'mag']:
            if sensor_type not in sensor_types:
                continue

            sensor_desc, unit = sensor_metadata[sensor_type]
            cols = sensor_types[sensor_type]

            for col in cols:
                if col in segment_df.columns:
                    stats_dict = self.utils.compute_stats(segment_df[col], self.statistics)
                    stats_str = ', '.join([f"{k}={v:.3f}" for k, v in stats_dict.items()])
                    description_parts.append(f"  {sensor_desc} (axis {axis_num}, {unit}): {stats_str}")
                else:
                    description_parts.append(f"  {sensor_desc} (axis {axis_num}, {unit}): no data")
                axis_num += 1

    def _compute_global_pca(self, window_data: pd.DataFrame) -> List[float]:
        """
        Compute global PCA features across all sensor dimensions.

        Args:
            window_data: DataFrame containing all sensor data

        Returns:
            List of statistical features computed on each PCA component
        """
        # Define all sensor columns (excluding ECG and label)
        all_sensor_cols = []
        for sensor_cols in self.sensor_locations.values():
            all_sensor_cols.extend(sensor_cols)

        # Get available sensor columns
        available_cols = [col for col in all_sensor_cols if col in window_data.columns]

        if len(available_cols) == 0:
            return [0.0] * (self.pca_components * 7)

        # Extract sensor data
        sensor_data = window_data[available_cols].values

        # Handle case where we have insufficient data
        if sensor_data.shape[0] < 2 or sensor_data.shape[1] < 2:
            return [0.0] * (self.pca_components * 7)

        try:
            # Standardize the data
            scaler = StandardScaler()
            sensor_data_scaled = scaler.fit_transform(sensor_data)

            # Apply PCA
            n_components_actual = min(self.pca_components, sensor_data.shape[1], sensor_data.shape[0])
            pca = PCA(n_components=n_components_actual)
            pca_transformed = pca.fit_transform(sensor_data_scaled)

            # Pad with zeros if we have fewer components than requested
            if n_components_actual < self.pca_components:
                padding = np.zeros((pca_transformed.shape[0], self.pca_components - n_components_actual))
                pca_transformed = np.hstack([pca_transformed, padding])

            # Compute statistics on each PCA component
            pca_stats = []
            for i in range(self.pca_components):
                component_data = pca_transformed[:, i]
                stats_dict = self.utils.compute_stats(pd.Series(component_data), self.statistics)
                pca_stats.extend(stats_dict.values())

            return pca_stats

        except Exception as e:
            logger.warning(f"Error computing PCA: {e}")
            return [0.0] * (self.pca_components * 7)
