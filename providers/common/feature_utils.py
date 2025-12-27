"""
Common utilities for feature extraction.
Providers can use these utilities in their extract_features() implementation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List


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


class FeatureExtractorUtils:
    """
    Common utilities for extracting statistical features from sensor data.
    Providers can use this class to avoid reimplementing common functionality.
    """

    @staticmethod
    def compute_stats(x: pd.Series, stat_names: List[str]) -> Dict[str, float]:
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
            if stat_name == "mean":
                stats_dict["mean"] = np.mean(x_arr)
            elif stat_name == "std":
                stats_dict["std"] = np.std(x_arr)
            elif stat_name == "min":
                stats_dict["min"] = np.min(x_arr)
            elif stat_name == "max":
                stats_dict["max"] = np.max(x_arr)
            elif stat_name == "median":
                stats_dict["median"] = np.median(x_arr)
            elif stat_name == "p25":
                stats_dict["p25"] = np.percentile(x_arr, 25)
            elif stat_name == "p75":
                stats_dict["p75"] = np.percentile(x_arr, 75)
            elif stat_name == "variance" or stat_name == "var":
                stats_dict["variance"] = np.var(x_arr)
            elif stat_name == "range":
                stats_dict["range"] = np.max(x_arr) - np.min(x_arr)
            elif stat_name == "skewness" or stat_name == "skew":
                stats_dict["skewness"] = safe_skew(x_arr)
            elif stat_name == "kurtosis" or stat_name == "kurt":
                stats_dict["kurtosis"] = safe_kurtosis(x_arr)
            elif stat_name == "rms":
                stats_dict["rms"] = np.sqrt(np.mean(x_arr**2))
            elif stat_name == "energy":
                stats_dict["energy"] = np.sum(x_arr**2)
            elif stat_name == "zero_crossings":
                stats_dict["zero_crossings"] = np.sum(np.diff(np.sign(x_arr)) != 0)
            else:
                stats_dict[stat_name] = 0.0

        return stats_dict

    @staticmethod
    def split_temporal_segments(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split a window into temporal segments.

        Args:
            df: Window DataFrame

        Returns:
            Dict with 'whole', 'start', 'middle', 'end' segments
        """
        n_rows = len(df)
        segment_size = n_rows // 3

        return {
            "whole": df,
            "start": df[:segment_size],
            "middle": df[segment_size : 2 * segment_size],
            "end": df[2 * segment_size :],
        }
