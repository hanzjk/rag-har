"""
Dataset Provider - Abstract base class for handling different HAR datasets.
Provides a unified interface for loading, preprocessing, and accessing sensor data.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetProvider(ABC):
    """
    Abstract base class for dataset providers.
    Each dataset implementation should inherit from this class.
    """

    def __init__(self, config_path: str):
        """
        Initialize dataset provider with configuration file.

        Args:
            config_path: Path to dataset configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.dataset_name = self.config["dataset_name"]

        logger.info(f"Initialized {self.dataset_name} dataset provider")

    def _load_config(self) -> Dict:
        """Load dataset configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    @abstractmethod
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from source.

        Returns:
            Dict mapping activity labels to DataFrames
        """
        pass

    def get_standardized_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and standardize data to common format.

        Returns:
            Dict mapping activity labels to standardized DataFrames
            with columns: timestamp, accel_x, accel_y, accel_z,
                         gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, activity
        """
        raw_data = self.load_raw_data()
        standardized_data = {}

        for activity, df in raw_data.items():
            standardized_df = self._standardize_columns(df, activity)
            standardized_data[activity] = standardized_df

        logger.info(f"Standardized {len(standardized_data)} activity datasets")
        return standardized_data

    def _standardize_columns(self, df: pd.DataFrame, activity: str) -> pd.DataFrame:
        """
        Rename columns to standard format based on config mapping.

        Args:
            df: Original DataFrame
            activity: Activity label

        Returns:
            DataFrame with standardized column names
        """
        column_mapping = self.config["columns"]

        # Create reverse mapping (dataset_column -> standard_name)
        # Handle case where config already uses standard names
        rename_dict = {}
        for standard_name, dataset_column in column_mapping.items():
            if dataset_column in df.columns:
                rename_dict[dataset_column] = standard_name

        standardized_df = df.rename(columns=rename_dict)

        # Ensure all required columns exist
        required_columns = [
            "timestamp",
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
        ]

        missing_columns = set(required_columns) - set(standardized_df.columns)
        if missing_columns:
            logger.warning(f"Missing columns for {activity}: {missing_columns}")

        # Ensure activity column exists
        if "activity" not in standardized_df.columns:
            standardized_df["activity"] = activity

        return standardized_df[required_columns + ["activity"]]

    def get_window_config(self) -> Dict:
        """Get window configuration for segmentation."""
        return self.config["preprocessing"]

    def get_feature_config(self) -> Dict:
        """Get feature extraction configuration."""
        return self.config["features"]

    def get_activities(self) -> List[str]:
        """Get list of activity labels."""
        return self.config["data_source"]["activities"]

    def get_sampling_rate(self) -> int:
        """Get sampling rate in Hz."""
        return self.config["preprocessing"]["sampling_rate"]

    @abstractmethod
    def preprocess(self, output_dir: str) -> str:
        """
        Dataset-specific preprocessing implementation.
        Each provider implements its own preprocessing logic.

        Args:
            output_dir: Directory to save preprocessed windows

        Returns:
            Path to saved windows file

        Note:
            Preprocessing is dataset-specific and may include:
            - Custom filtering/smoothing
            - Resampling to target frequency
            - Sensor-specific noise removal
            - Dataset-specific normalization
            - Custom windowing strategies
        """
        pass

    @abstractmethod
    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        """
        Dataset-specific feature extraction implementation.
        Each provider knows how to extract features from its own column structure.

        Args:
            windows_dir: Path to preprocessed windows CSV directory
            output_dir: Directory to save extracted features and descriptions

        Returns:
            Path to saved features file

        Note:
            Feature extraction is dataset-specific because different datasets have:
            - Different sensor naming (accel_x vs ankle_accel_x vs hand_sensor_x)
            - Different sensors available (some have mag, some don't)
            - Different sensor locations (ankle, wrist, chest vs single location)
            - Different feature requirements

            Providers can use common/feature_utils.py for reusable functionality.
        """
        pass


def get_provider(config_path: str) -> DatasetProvider:
    """
    Factory function to get appropriate dataset provider.
    Dynamically imports provider classes from providers/ directory.

    Args:
        config_path: Path to dataset configuration file

    Returns:
        Initialized DatasetProvider instance
    """
    # Load config to determine provider type
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset_name"]

    # Map dataset names to provider module and class names
    provider_registry = {
        "har_demo": ("providers.har_demo.provider", "HARDemoProvider"),
        "gotov": ("providers.gotov.provider", "GOTOVProvider"),
        "skoda": ("providers.skoda.provider", "SkodaProvider"),
        "hhar": ("providers.hhar.provider", "HHARProvider"),
        "mhealth": ("providers.mhealth.provider", "MHEALTHProvider"),
        "usc-had": ("providers.usc-had.provider", "USCHADProvider"),
        "pamap2": ("providers.pamp2.provider", "PAMAP2Provider"),
        # Add more datasets here:
        # "my_dataset": ("providers.my_dataset.provider", "MyDatasetProvider"),
    }

    if dataset_name not in provider_registry:
        logger.error(f"No provider registered for dataset: {dataset_name}")
        logger.error(f"Available datasets: {list(provider_registry.keys())}")
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Dynamically import the provider class
    module_name, class_name = provider_registry[dataset_name]
    try:
        import importlib
        module = importlib.import_module(module_name)
        provider_class = getattr(module, class_name)
        logger.info(f"Loaded provider: {class_name} from {module_name}")
        return provider_class(config_path)
    except ImportError as e:
        logger.error(f"Failed to import provider module {module_name}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Provider class {class_name} not found in {module_name}: {e}")
        raise
