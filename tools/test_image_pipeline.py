#!/usr/bin/env python3
"""
Test script for the image processing pipeline.

This script tests the new process_image function from the indexing pipeline module
using the existing test datasets. It validates that the pipeline correctly:
1. Loads and processes images
2. Generates valid histograms
3. Integrates with the existing histogram generation system

Usage:
    # Activate virtual environment first
    venv311\Scripts\activate
    
    # Run the test
    python tools/test_image_pipeline.py
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.indexing.pipeline import process_image, validate_processed_image
from chromatica.utils.config import TOTAL_BINS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_image(image_path: str) -> bool:
    """
    Test processing of a single image.
    
    Args:
        image_path: Path to the image file to test.
    
    Returns:
        bool: True if processing succeeds, False otherwise.
    """
    try:
        logger.info(f"Testing image: {image_path}")
        
        # Process the image
        start_time = time.time()
        histogram = process_image(image_path)
        processing_time = time.time() - start_time
        
        # Validate the histogram
        validate_processed_image(histogram, image_path)
        
        # Log success
        logger.info(f"✓ Successfully processed {image_path}")
        logger.info(f"  - Processing time: {processing_time:.3f}s")
        logger.info(f"  - Histogram shape: {histogram.shape}")
        logger.info(f"  - Histogram sum: {histogram.sum():.6f}")
        logger.info(f"  - Non-zero bins: {np.count_nonzero(histogram)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to process {image_path}: {str(e)}")
        return False


def test_dataset(dataset_path: str) -> tuple[int, int]:
    """
    Test processing of all images in a dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory.
    
    Returns:
        tuple: (success_count, total_count)
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return 0, 0
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in dataset_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.warning(f"No image files found in {dataset_path}")
        return 0, 0
    
    logger.info(f"Found {len(image_files)} images in {dataset_path}")
    
    # Process each image
    success_count = 0
    total_count = len(image_files)
    
    for image_file in image_files:
        if test_single_image(str(image_file)):
            success_count += 1
    
    return success_count, total_count


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("Testing Image Processing Pipeline")
    logger.info("=" * 60)
    
    # Test datasets
    datasets = [
        "datasets/test-dataset-20",
        "datasets/test-dataset-50"
    ]
    
    total_success = 0
    total_images = 0
    
    for dataset in datasets:
        logger.info(f"\nTesting dataset: {dataset}")
        logger.info("-" * 40)
        
        success, count = test_dataset(dataset)
        total_success += success
        total_images += count
        
        success_rate = (success / count * 100) if count > 0 else 0
        logger.info(f"Dataset results: {success}/{count} ({success_rate:.1f}%)")
    
    # Overall results
    logger.info("\n" + "=" * 60)
    logger.info("OVERALL RESULTS")
    logger.info("=" * 60)
    
    overall_success_rate = (total_success / total_images * 100) if total_images > 0 else 0
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Failed: {total_images - total_success}")
    logger.info(f"Success rate: {overall_success_rate:.1f}%")
    
    if total_success == total_images:
        logger.info("🎉 All tests passed! The image processing pipeline is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    # Import numpy here to avoid import issues
    import numpy as np
    
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
