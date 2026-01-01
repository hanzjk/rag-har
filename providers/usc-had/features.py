"""
USC-HAD Feature Extraction - Extracts statistical features from preprocessed windows.
Supports multiple feature extraction methods: default, comprehensive, and structured.
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


class USCHADFeatureExtractor:
    """
    Feature extractor for USC-HAD dataset.
    Extracts temporal segment features with demographic metadata support.
    """

    def __init__(self, config: Dict):
        """Initialize USC-HAD feature extractor."""
        self.config = config
        self.dataset_name = config['dataset_name']
        self.feature_config = config.get('features', {})
        self.method = self.feature_config.get('method', 'default')
        self.sampling_rate = config['preprocessing'].get('sampling_rate', 33)

        # Initialize feature utility
        self.feature_utils = FeatureExtractorUtils()

        # Sensor columns
        self.sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

        logger.info(f"Initialized USC-HAD feature extractor (method: {self.method})")

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
        logger.info("USC-HAD FEATURE EXTRACTION")
        logger.info("="*60)
        logger.info(f"Method: {self.method}")
        logger.info(f"Windows directory: {windows_dir}")
        logger.info("")

        windows_path = Path(windows_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all window CSV files
        train_files = list(windows_path.glob('train/**/*.csv'))
        test_files = list(windows_path.glob('test/**/*.csv'))

        all_files = train_files + test_files

        logger.info(f"Found {len(all_files)} windows")
        logger.info(f"  Training: {len(train_files)}")
        logger.info(f"  Test: {len(test_files)}")
        logger.info("")

        # Create output directories
        train_desc_dir = output_path / 'train_descriptions'
        test_desc_dir = output_path / 'test_descriptions'
        train_desc_dir.mkdir(parents=True, exist_ok=True)
        test_desc_dir.mkdir(parents=True, exist_ok=True)

        # Process all windows
        logger.info("Processing windows...")
        for file_path in tqdm(all_files, desc="Extracting features", unit="file"):
            try:
                # Load window data
                df = pd.read_csv(file_path)

                # Verify required columns
                missing_cols = [col for col in self.sensor_columns if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns {missing_cols} in {file_path.name}")
                    continue

                # Create temporal segments
                df_whole, df_start, df_mid, df_end = self._create_temporal_segments(df)

                # Generate description based on method
                if self.method == 'default':
                    description = self._generate_default_description(df_whole, df_start, df_mid, df_end, df)
                elif self.method == 'comprehensive':
                    description = self._generate_comprehensive_description(df_whole, df_start, df_mid, df_end, df)
                elif self.method == 'structured':
                    description = self._generate_structured_description(df_whole, df_start, df_mid, df_end, df)
                else:
                    logger.error(f"Unknown method: {self.method}")
                    continue

                # Determine output directory
                if file_path in train_files:
                    base_output_dir = train_desc_dir
                else:
                    base_output_dir = test_desc_dir

                # Parse filename: subject{id}_window{idx}_activity{num}_{name}.csv
                filename = file_path.stem
                parts = filename.split('_', 3)
                if len(parts) >= 4:
                    window_idx = parts[1].replace('window', '')
                    activity_name = parts[3]  # Everything after "activity{num}_"

                    # Output format: window_{idx}_activity_{name}_stats.txt
                    out_filename = f"window_{window_idx}_activity_{activity_name}_stats.txt"
                else:
                    # Fallback
                    out_filename = f"{filename}_stats.txt"

                out_file = base_output_dir / out_filename

                # Write description
                with open(out_file, 'w') as f:
                    f.write(description)

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                continue

        logger.info("")
        logger.info(f"✓ Feature extraction complete")
        logger.info(f"✓ Output: {output_path}")

        return str(output_path)

    def _create_temporal_segments(self, df: pd.DataFrame):
        """Create temporal segments from window."""
        n_rows = len(df)
        segment_size = n_rows // 3

        df_whole = df[self.sensor_columns]
        df_start = df[self.sensor_columns].iloc[:segment_size]
        df_mid = df[self.sensor_columns].iloc[segment_size:2*segment_size]
        df_end = df[self.sensor_columns].iloc[2*segment_size:]

        return df_whole, df_start, df_mid, df_end

    def _generate_default_description(self, df_whole, df_start, df_mid, df_end, df_full):
        """Generate description using default method with temporal segmentation."""
        segments = {
            'Whole Segment': df_whole,
            'Start Segment': df_start,
            'Mid Segment': df_mid,
            'End Segment': df_end
        }

        description_parts = []

        # Add demographic information header
        age_desc = df_full['age_descriptor'].iloc[0] if 'age_descriptor' in df_full.columns else 'unknown'
        height_desc = df_full['height_descriptor'].iloc[0] if 'height_descriptor' in df_full.columns else 'unknown'
        weight_desc = df_full['weight_descriptor'].iloc[0] if 'weight_descriptor' in df_full.columns else 'unknown'

        description_parts.append(f"Participant: {age_desc}, {height_desc}, {weight_desc}")
        description_parts.append("")

        for seg_name, seg_df in segments.items():
            description_parts.append(f"[{seg_name}]")

            if seg_df.empty:
                description_parts.append("  No sensor data available")
                description_parts.append("")
                continue

            # Accelerometer
            acc_desc = self._describe_accelerometer(seg_df)
            description_parts.append(acc_desc)

            # Gyroscope
            gyro_desc = self._describe_gyroscope(seg_df)
            description_parts.append(gyro_desc)

            description_parts.append("")

        return "\n".join(description_parts).strip()

    def _describe_accelerometer(self, seg_df):
        """Describe accelerometer features."""
        lines = ["Accelerometer:"]

        # Per-axis features (mean, std, min, max, rms)
        for axis in ['x', 'y', 'z']:
            col = f'acc_{axis}'
            if col not in seg_df.columns:
                continue

            data = seg_df[col].dropna()
            if len(data) == 0:
                continue

            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            rms_val = np.sqrt(np.mean(data**2))

            lines.append(f"  {axis.upper()}-axis: mean={mean_val:.3f}, std={std_val:.3f}, "
                        f"min={min_val:.3f}, max={max_val:.3f}, rms={rms_val:.3f}")

        # Orientation-invariant features
        if all(f'acc_{axis}' in seg_df.columns for axis in ['x', 'y', 'z']):
            x_arr = seg_df['acc_x'].dropna().values
            y_arr = seg_df['acc_y'].dropna().values
            z_arr = seg_df['acc_z'].dropna().values

            min_len = min(len(x_arr), len(y_arr), len(z_arr))
            if min_len > 0:
                x_arr = x_arr[:min_len]
                y_arr = y_arr[:min_len]
                z_arr = z_arr[:min_len]

                magnitude = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
                magnitude_mean = np.mean(magnitude)
                magnitude_std = np.std(magnitude)
                sma = (np.sum(np.abs(x_arr)) + np.sum(np.abs(y_arr)) + np.sum(np.abs(z_arr))) / len(x_arr)

                # Jerk RMS
                if len(magnitude) > 1:
                    jerk = np.diff(magnitude)
                    jerk_rms = np.sqrt(np.mean(jerk**2))
                else:
                    jerk_rms = 0.0

                # Dominant frequency
                if len(magnitude) > 1:
                    from scipy.fft import fft
                    fft_magnitude = np.abs(fft(magnitude))
                    freqs = np.fft.fftfreq(len(magnitude))
                    dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
                    dominant_frequency = np.abs(freqs[dominant_freq_idx])
                else:
                    dominant_frequency = 0.0

                lines.append(f"  Orientation-invariant: magnitude_mean={magnitude_mean:.3f}, "
                           f"magnitude_std={magnitude_std:.3f}, sma={sma:.3f}, "
                           f"jerk_rms={jerk_rms:.3f}, dominant_freq={dominant_frequency:.3f}")

        return "\n".join(lines)

    def _describe_gyroscope(self, seg_df):
        """Describe gyroscope features."""
        lines = ["Gyroscope:"]

        # Per-axis features (mean, std, min, max, rms)
        for axis in ['x', 'y', 'z']:
            col = f'gyro_{axis}'
            if col not in seg_df.columns:
                continue

            data = seg_df[col].dropna()
            if len(data) == 0:
                continue

            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            rms_val = np.sqrt(np.mean(data**2))

            lines.append(f"  {axis.upper()}-axis: mean={mean_val:.3f}, std={std_val:.3f}, "
                        f"min={min_val:.3f}, max={max_val:.3f}, rms={rms_val:.3f}")

        # Orientation-invariant features
        if all(f'gyro_{axis}' in seg_df.columns for axis in ['x', 'y', 'z']):
            x_arr = seg_df['gyro_x'].dropna().values
            y_arr = seg_df['gyro_y'].dropna().values
            z_arr = seg_df['gyro_z'].dropna().values

            min_len = min(len(x_arr), len(y_arr), len(z_arr))
            if min_len > 0:
                x_arr = x_arr[:min_len]
                y_arr = y_arr[:min_len]
                z_arr = z_arr[:min_len]

                magnitude = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
                magnitude_mean = np.mean(magnitude)
                magnitude_std = np.std(magnitude)

                # Spectral entropy
                if len(magnitude) > 1:
                    from scipy.fft import fft
                    fft_magnitude = np.abs(fft(magnitude))
                    psd = fft_magnitude**2
                    psd_norm = psd / np.sum(psd)
                    psd_norm = psd_norm[psd_norm > 0]
                    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm)) if len(psd_norm) > 0 else 0.0
                else:
                    spectral_entropy = 0.0

                # Energy in low frequency band
                if len(magnitude) > 1:
                    from scipy.fft import fft
                    fft_magnitude = np.abs(fft(magnitude))
                    freqs = np.fft.fftfreq(len(magnitude))
                    low_freq_mask = (np.abs(freqs) >= 0.1) & (np.abs(freqs) <= 0.3)
                    energy_low_band = np.sum(fft_magnitude[low_freq_mask]**2)
                else:
                    energy_low_band = 0.0

                # Cross-axis correlation
                if len(x_arr) > 1:
                    corr_xy = np.corrcoef(x_arr, y_arr)[0, 1] if not (np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0])) else 0.0
                    corr_yz = np.corrcoef(y_arr, z_arr)[0, 1] if not (np.all(y_arr == y_arr[0]) or np.all(z_arr == z_arr[0])) else 0.0
                    corr_xz = np.corrcoef(x_arr, z_arr)[0, 1] if not (np.all(x_arr == x_arr[0]) or np.all(z_arr == z_arr[0])) else 0.0
                    cross_axis_corr = np.mean([corr_xy, corr_yz, corr_xz])
                else:
                    cross_axis_corr = 0.0

                lines.append(f"  Orientation-invariant: magnitude_mean={magnitude_mean:.3f}, "
                           f"magnitude_std={magnitude_std:.3f}, spectral_entropy={spectral_entropy:.3f}, "
                           f"energy_low_band={energy_low_band:.3f}, cross_axis_corr={cross_axis_corr:.3f}")

        return "\n".join(lines)

    def _generate_comprehensive_description(self, df_whole, df_start, df_mid, df_end, df_full):
        """Generate comprehensive description with extensive features."""
        # This method uses FeatureExtractorUtils for comprehensive stats
        segments = {
            'Whole Segment': df_whole,
            'Start Segment': df_start,
            'Mid Segment': df_mid,
            'End Segment': df_end
        }

        description_parts = []

        # Add demographic information
        age_desc = df_full['age_descriptor'].iloc[0] if 'age_descriptor' in df_full.columns else 'unknown'
        height_desc = df_full['height_descriptor'].iloc[0] if 'height_descriptor' in df_full.columns else 'unknown'
        weight_desc = df_full['weight_descriptor'].iloc[0] if 'weight_descriptor' in df_full.columns else 'unknown'

        description_parts.append(f"Participant: {age_desc}, {height_desc}, {weight_desc}")
        description_parts.append("")

        for seg_name, seg_df in segments.items():
            description_parts.append(f"[{seg_name}]")

            if seg_df.empty:
                description_parts.append("  No sensor data available")
                description_parts.append("")
                continue

            # Use FeatureExtractorUtils for comprehensive stats
            for sensor_group in ['acc', 'gyro']:
                axes = ['x', 'y', 'z']
                sensor_cols = [f'{sensor_group}_{axis}' for axis in axes]

                if all(col in seg_df.columns for col in sensor_cols):
                    stats_dict = {}
                    for col in sensor_cols:
                        col_stats = self.feature_utils.compute_statistics(
                            seg_df[col],
                            self.feature_config.get('statistics', ['mean', 'std', 'min', 'max', 'median'])
                        )
                        for stat_name, stat_val in col_stats.items():
                            stats_dict[f'{col}_{stat_name}'] = stat_val

                    # Format output
                    description_parts.append(f"{sensor_group.upper()}:")
                    for key, val in list(stats_dict.items())[:15]:  # Limit output
                        description_parts.append(f"  {key}={val:.3f}")

            description_parts.append("")

        return "\n".join(description_parts).strip()

    def _generate_structured_description(self, df_whole, df_start, df_mid, df_end, df_full):
        """
        Generate structured description with categorical bins.
        This is the USC-HAD specific format with motion categorization.
        """
        from .structured_features import generate_structured_description

        # Generate structured description using the specialized module
        description = generate_structured_description(df_whole, df_start, df_mid, df_end, self.sampling_rate)

        # Prepend demographic information
        age_desc = df_full['age_descriptor'].iloc[0] if 'age_descriptor' in df_full.columns else 'unknown'
        height_desc = df_full['height_descriptor'].iloc[0] if 'height_descriptor' in df_full.columns else 'unknown'
        weight_desc = df_full['weight_descriptor'].iloc[0] if 'weight_descriptor' in df_full.columns else 'unknown'

        header = f"Participant: {age_desc}, {height_desc}, {weight_desc}\n\n"

        return header + description
