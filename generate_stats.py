"""
Feature Extraction - Dataset-agnostic statistical feature generation.
Processes windows from preprocessing pipeline and generates comprehensive feature vectors.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from tqdm import tqdm
import logging

from dataset_provider import get_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_corrcoef(x_arr, y_arr):
    """Safely compute correlation coefficient, handling edge cases."""
    if np.std(x_arr) < 1e-8 or np.std(y_arr) < 1e-8:
        return 0.0
    return np.corrcoef(x_arr, y_arr)[0, 1]


def safe_skew(x_arr):
    """Safely compute skewness, handling edge cases."""
    if len(x_arr) > 2 and np.std(x_arr) > 1e-8:
        return stats.skew(x_arr)
    return 0.0


def safe_kurtosis(x_arr):
    """Safely compute kurtosis, handling edge cases."""
    if len(x_arr) > 3 and np.std(x_arr) > 1e-8:
        return stats.kurtosis(x_arr)
    return 0.0


class FeatureExtractor:
    """Dataset-agnostic feature extractor for HAR windows."""

    def __init__(self, dataset_provider):
        """
        Initialize feature extractor with dataset provider.

        Args:
            dataset_provider: DatasetProvider instance
        """
        self.provider = dataset_provider
        self.config = dataset_provider.get_feature_config()
        self.sampling_rate = dataset_provider.get_sampling_rate()

        # Get sensor column prefixes (dataset-specific)
        # List of column prefixes to extract features from
        # e.g., ['accel', 'gyro', 'mag'] for single-sensor datasets
        # or ['ankle_accel', 'wrist_gyro', 'chest_mag', ...] for multi-sensor datasets
        self.sensor_columns = self.config.get('sensor_columns', ['accel', 'gyro', 'mag'])

        # Get which statistics to compute
        self.statistics = self.config.get('statistics', ['mean', 'std', 'min', 'max', 'median'])

        # Feature options
        self.compute_magnitude = self.config.get('magnitude', True)
        self.per_axis = self.config.get('per_axis', True)

        logger.info(f"Feature extractor initialized:")
        logger.info(f"  Sensor columns: {self.sensor_columns}")
        logger.info(f"  Statistics: {self.statistics}")
        logger.info(f"  Magnitude: {self.compute_magnitude}, Per-axis: {self.per_axis}")

    def compute_stats(self, x: pd.Series, stat_names: List[str]) -> Dict[str, float]:
        """
        Compute specified statistical features for a sensor channel.

        Args:
            x: Sensor data series
            stat_names: List of statistics to compute

        Returns:
            Dict mapping statistic name to value
        """
        # Handle NaN values
        x_clean = x.dropna()
        if len(x_clean) == 0:
            return {name: 0.0 for name in stat_names}

        x_arr = np.array(x_clean)
        stats_dict = {}

        for stat_name in stat_names:
            if stat_name == 'mean':
                stats_dict['mean'] = np.mean(x_arr)
            elif stat_name == 'std':
                stats_dict['std'] = np.std(x_arr)
            elif stat_name == 'min':
                stats_dict['min'] = np.min(x_arr)
            elif stat_name == 'max':
                stats_dict['max'] = np.max(x_arr)
            elif stat_name == 'median':
                stats_dict['median'] = np.median(x_arr)
            elif stat_name == 'p25':
                stats_dict['p25'] = np.percentile(x_arr, 25)
            elif stat_name == 'p75':
                stats_dict['p75'] = np.percentile(x_arr, 75)
            elif stat_name == 'variance':
                stats_dict['variance'] = np.var(x_arr)
            elif stat_name == 'range':
                stats_dict['range'] = np.ptp(x_arr)
            elif stat_name == 'skewness':
                stats_dict['skewness'] = safe_skew(x_arr)
            elif stat_name == 'kurtosis':
                stats_dict['kurtosis'] = safe_kurtosis(x_arr)
            elif stat_name == 'rms':
                stats_dict['rms'] = np.sqrt(np.mean(x_arr ** 2))
            elif stat_name == 'energy':
                stats_dict['energy'] = np.sum(x_arr ** 2)
            elif stat_name == 'zero_crossings':
                stats_dict['zero_crossings'] = np.sum(np.diff(np.sign(x_arr)) != 0)
            elif stat_name == 'dominant_freq':
                stats_dict['dominant_freq'] = self._compute_dominant_frequency(x_arr)
            else:
                logger.warning(f"Unknown statistic: {stat_name}")
                stats_dict[stat_name] = 0.0

        return stats_dict

    def _compute_dominant_frequency(self, x_arr: np.ndarray) -> float:
        """Compute dominant frequency using FFT."""
        if len(x_arr) <= 1:
            return 0.0

        fft_vals = np.fft.fft(x_arr)
        fft_freqs = np.fft.fftfreq(len(x_arr), 1.0 / self.sampling_rate)

        # Get magnitude spectrum (ignore DC component)
        magnitude_spectrum = np.abs(fft_vals[1:len(x_arr)//2])
        if len(magnitude_spectrum) == 0:
            return 0.0

        dominant_freq_idx = np.argmax(magnitude_spectrum) + 1
        return abs(fft_freqs[dominant_freq_idx])

    def extract_window_features(self, window: Dict) -> Tuple[np.ndarray, str]:
        """
        Extract feature vector from a single window.

        Args:
            window: Window dict with 'data' (DataFrame) and 'activity' (str)

        Returns:
            Tuple of (feature_vector, description_text)
        """
        df = window['data']
        activity = window['activity']

        feature_vector = []
        description_parts = []

        # Iterate through all sensor column prefixes
        for prefix in self.sensor_columns:
            axes = ['x', 'y', 'z']

            # Per-axis features
            if self.per_axis:
                for axis in axes:
                    col_name = f'{prefix}_{axis}'
                    if col_name in df.columns:
                        stats_dict = self.compute_stats(df[col_name], self.statistics)

                        # Add to feature vector
                        feature_vector.extend(stats_dict.values())

                        # Add to description
                        stats_str = ', '.join([f"{k}={v:.3f}" for k, v in stats_dict.items()])
                        description_parts.append(f"{prefix.upper()} {axis.upper()}: {stats_str}")

            # Magnitude features
            if self.compute_magnitude:
                x_col = f'{prefix}_x'
                y_col = f'{prefix}_y'
                z_col = f'{prefix}_z'

                if all(col in df.columns for col in [x_col, y_col, z_col]):
                    magnitude = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)
                    stats_dict = self.compute_stats(magnitude, self.statistics)

                    # Add to feature vector
                    feature_vector.extend(stats_dict.values())

                    # Add to description
                    stats_str = ', '.join([f"{k}={v:.3f}" for k, v in stats_dict.items()])
                    description_parts.append(f"{prefix.upper()} Magnitude: {stats_str}")

        description_text = f"Activity: {activity}\n" + "\n".join(description_parts)

        return np.array(feature_vector), description_text

    def extract_segmented_features(self, window: Dict) -> Tuple[np.ndarray, str]:
        """
        Extract features from temporal segments (start, middle, end) of window.
        This provides more temporal context than single window features.

        Args:
            window: Window dict with 'data' (DataFrame) and 'activity' (str)

        Returns:
            Tuple of (feature_vector, description_text)
        """
        df = window['data']
        activity = window['activity']

        # Split into 3 equal segments
        n_rows = len(df)
        segment_size = n_rows // 3

        segments = {
            'whole': df,
            'start': df[:segment_size],
            'middle': df[segment_size:2*segment_size],
            'end': df[2*segment_size:]
        }

        all_features = []
        description_parts = [f"Activity: {activity}\n"]

        for segment_name, segment_df in segments.items():
            description_parts.append(f"\n[{segment_name.capitalize()} Segment]")

            segment_features = []

            # Iterate through all sensor column prefixes
            for prefix in self.sensor_columns:
                axes = ['x', 'y', 'z']

                # Per-axis features
                if self.per_axis:
                    for axis in axes:
                        col_name = f'{prefix}_{axis}'
                        if col_name in segment_df.columns:
                            stats_dict = self.compute_stats(segment_df[col_name], self.statistics)
                            segment_features.extend(stats_dict.values())

                            stats_str = ', '.join([f"{k}={v:.3f}" for k, v in stats_dict.items()])
                            description_parts.append(f"  {prefix.upper()} {axis.upper()}: {stats_str}")

                # Magnitude features
                if self.compute_magnitude:
                    x_col = f'{prefix}_x'
                    y_col = f'{prefix}_y'
                    z_col = f'{prefix}_z'

                    if all(col in segment_df.columns for col in [x_col, y_col, z_col]):
                        magnitude = np.sqrt(
                            segment_df[x_col]**2 +
                            segment_df[y_col]**2 +
                            segment_df[z_col]**2
                        )
                        stats_dict = self.compute_stats(magnitude, self.statistics)
                        segment_features.extend(stats_dict.values())

                        stats_str = ', '.join([f"{k}={v:.3f}" for k, v in stats_dict.items()])
                        description_parts.append(f"  {prefix.upper()} Magnitude: {stats_str}")

            all_features.extend(segment_features)

        description_text = "\n".join(description_parts)
        return np.array(all_features), description_text

    def process_all_windows(self, windows_file: str, output_dir: str, use_segmentation: bool = False):
        """
        Process all windows and generate feature vectors and descriptions.

        Args:
            windows_file: Path to windows.pkl file
            output_dir: Directory to save features and descriptions
            use_segmentation: If True, use temporal segmentation (start/mid/end)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load windows
        logger.info(f"Loading windows from {windows_file}")
        with open(windows_file, 'rb') as f:
            windows = pickle.load(f)

        logger.info(f"Processing {len(windows)} windows...")

        feature_vectors = []
        activities = []
        window_ids = []
        descriptions_dir = output_path / 'descriptions'
        descriptions_dir.mkdir(exist_ok=True)

        for window in tqdm(windows, desc="Extracting features"):
            try:
                if use_segmentation:
                    feature_vec, description = self.extract_segmented_features(window)
                else:
                    feature_vec, description = self.extract_window_features(window)

                feature_vectors.append(feature_vec)
                activities.append(window['activity'])
                window_ids.append(window['window_id'])

                # Save description
                desc_file = descriptions_dir / f"window_{window['window_id']}_activity_{window['activity']}_stats.txt"
                with open(desc_file, 'w') as f:
                    f.write(description)

            except Exception as e:
                logger.error(f"Error processing window {window['window_id']}: {e}")
                continue

        # Convert to numpy array
        feature_matrix = np.array(feature_vectors)

        logger.info(f"Extracted features: {feature_matrix.shape}")
        logger.info(f"Feature vector size: {feature_matrix.shape[1]}")

        # Save features
        features_file = output_path / 'features.pkl'
        with open(features_file, 'wb') as f:
            pickle.dump({
                'features': feature_matrix,
                'activities': activities,
                'window_ids': window_ids,
                'feature_dim': feature_matrix.shape[1]
            }, f)

        logger.info(f"Saved features to {features_file}")

        # Save summary
        summary_file = output_path / 'feature_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Feature Extraction Summary\n")
            f.write(f"=========================\n\n")
            f.write(f"Total windows: {len(windows)}\n")
            f.write(f"Feature vector dimension: {feature_matrix.shape[1]}\n")
            f.write(f"Sensors used: {', '.join(self.sensors)}\n")
            f.write(f"Statistics computed: {', '.join(self.statistics)}\n")
            f.write(f"Per-axis features: {self.per_axis}\n")
            f.write(f"Magnitude features: {self.compute_magnitude}\n")
            f.write(f"Temporal segmentation: {use_segmentation}\n\n")
            f.write(f"Features per activity:\n")

            activity_counts = {}
            for activity in activities:
                activity_counts[activity] = activity_counts.get(activity, 0) + 1

            for activity, count in sorted(activity_counts.items()):
                f.write(f"  {activity}: {count} windows\n")

        logger.info(f"Saved summary to {summary_file}")

        return str(features_file)


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from preprocessed windows (calls dataset-specific implementation)"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to dataset configuration YAML file'
    )

    args = parser.parse_args()

    # Get dataset provider
    logger.info(f"Loading dataset configuration from {args.config}")
    provider = get_provider(args.config)

    # Automatic paths based on dataset name
    dataset_name = provider.dataset_name
    train_windows_dir = f"output/{dataset_name}/train-test-splits/train"
    test_windows_dir = f"output/{dataset_name}/train-test-splits/test"
    train_output_dir = f"output/{dataset_name}/features/train"
    test_output_dir = f"output/{dataset_name}/features/test"

    logger.info("=" * 60)
    logger.info("FEATURE EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info("")

    # Extract features for TRAIN set
    logger.info("Extracting features for TRAIN set...")
    logger.info(f"Train windows directory: {train_windows_dir}")
    logger.info(f"Train output directory: {train_output_dir}")
    train_features_file = provider.extract_features(train_windows_dir, train_output_dir)

    logger.info("")
    logger.info("Extracting features for TEST set...")
    logger.info(f"Test windows directory: {test_windows_dir}")
    logger.info(f"Test output directory: {test_output_dir}")
    test_features_file = provider.extract_features(test_windows_dir, test_output_dir)

    logger.info("")
    logger.info(f"✓ Feature extraction complete!")
    logger.info(f"  Train features: {train_features_file}")
    logger.info(f"  Test features: {test_features_file}")


if __name__ == '__main__':
    main()
