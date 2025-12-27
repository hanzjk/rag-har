"""
Preprocessing Pipeline Wrapper.
Calls dataset-specific preprocessing implementation.

Each dataset provider implements its own preprocess() method.
This allows complete customization per dataset while maintaining a unified interface.
"""

import argparse
import logging
from dataset_provider import get_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Preprocessing pipeline wrapper.
    Delegates to dataset-specific preprocessing implementation.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess HAR dataset (calls dataset-specific implementation)"
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

    # Automatic output directory based on dataset name
    output_dir = f"output/{provider.dataset_name}"

    logger.info("")
    logger.info("="*80)
    logger.info(f"Dataset: {provider.dataset_name}")
    logger.info(f"Provider: {provider.__class__.__name__}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)
    logger.info("")
    logger.info("NOTE: Preprocessing is dataset-specific.")
    logger.info("Each dataset can implement completely custom preprocessing logic.")
    logger.info("")

    # Call dataset-specific preprocessing
    windows_file = provider.preprocess(output_dir)

    logger.info("")
    logger.info("="*80)
    logger.info(f"✓ Preprocessing complete!")
    logger.info(f"✓ Windows saved to: {windows_file}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
