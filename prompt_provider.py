"""
Prompt Provider - Provides dataset-specific system prompts for LLM classification.
Allows customization of classification instructions based on dataset characteristics.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)


class PromptProvider:
    """
    Provides system and user prompts for RAG-based classification.
    Configurable per dataset to handle domain-specific requirements.
    """

    def __init__(self, config: dict):
        """
        Initialize prompt provider with dataset configuration.

        Args:
            config: Dataset configuration dict containing prompts section
        """
        self.config = config
        self.dataset_name = config.get("dataset_name", "unknown")
        self.prompts_config = config.get("prompts", {})

        # Load prompt templates from config or use defaults
        self.system_template = self.prompts_config.get(
            "system_prompt",
            "Use semantic similarity to compare the candidate statistics with the retrieved samples and output the activity label that maximizes similarity; respond with only the class label from {classes} and nothing else.",
        )

        self.user_template = self.prompts_config.get(
            "user_prompt",
            """You are given summary statistics for sensor data across temporal segments for labeled samples and one unlabeled candidate.

--- CANDIDATE ---
Time Series:
{candidate_series}

--- LABELED SAMPLES ---
{retrieved_data}""",
        )

        logger.info(f"Initialized prompt provider for dataset: {self.dataset_name}")

    def get_system_prompt(self, valid_labels: List[str]) -> str:
        """
        Generate system prompt for classification.

        Args:
            valid_labels: List of valid activity labels

        Returns:
            Formatted system prompt
        """
        classes_str = str(valid_labels)
        return self.system_template.format(classes=classes_str)

    def get_user_prompt(self, candidate_series: str, retrieved_data: str) -> str:
        """
        Generate user prompt with candidate and retrieved samples.

        Args:
            candidate_series: Formatted candidate time series data
            retrieved_data: Formatted retrieved sample data

        Returns:
            Formatted user prompt
        """
        return self.user_template.format(
            candidate_series=candidate_series, retrieved_data=retrieved_data
        )


def get_prompt_provider(config: dict) -> PromptProvider:
    """
    Factory function to get prompt provider from config.

    Args:
        config: Dataset configuration dict

    Returns:
        PromptProvider instance
    """
    return PromptProvider(config)
