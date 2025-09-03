#!/usr/bin/env python3
"""
Test script for the Sinkhorn reranking system.

This script demonstrates the reranking functionality by:
1. Loading test images and generating histograms
2. Creating synthetic query histograms
3. Running the reranking pipeline
4. Displaying results and performance metrics

Usage:
    python tools/test_reranking.py [--dataset test-dataset-20] [--num-candidates 10]
"""

import argparse
import logging
import time
import numpy as np
from pathlib import Path
import sys
import cv2
from skimage import color

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.core.histogram import build_histogram
from chromatica.core.rerank import rerank_candidates, validate_reranking_system
from chromatica.utils.config import TOTAL_BINS, RERANK_K, MAX_IMAGE_DIMENSION
from chromatica.utils.config import validate_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_convert_image(image_path: Path) -> np.ndarray:
    """
    Load an image and convert it to Lab color space pixels.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: Lab pixels array of shape (N, 3)
    """
    logger.debug(f"Loading image: {image_path}")
    
    # Load image using OpenCV
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB (OpenCV loads in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image if necessary (maintain aspect ratio)
    height, width = image_rgb.shape[:2]
    if max(height, width) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_rgb = cv2.resize(
            image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    # Convert to Lab color space using scikit-image
    # Note: skimage.color.rgb2lab expects RGB in [0, 1] range
    image_rgb_normalized = image_rgb.astype(np.float32) / 255.0
    image_lab = color.rgb2lab(image_rgb_normalized, illuminant="D65")
    
    # Reshape to (N, 3) array of Lab pixels
    lab_pixels = image_lab.reshape(-1, 3)
    
    logger.debug(f"Converted image to Lab: {lab_pixels.shape[0]} pixels")
    return lab_pixels


def create_synthetic_query_histogram() -> np.ndarray:
    """
    Create a synthetic query histogram for testing.
    
    This creates a histogram that represents a query for "reddish" colors,
    with most mass concentrated in the red regions of the Lab color space.
    
    Returns:
        np.ndarray: Normalized histogram representing the query
    """
    logger.info("Creating synthetic query histogram for 'reddish' colors...")
    
    # Create a histogram with mass concentrated in red regions
    # In Lab space, red corresponds to positive a* values
    query_hist = np.zeros(TOTAL_BINS, dtype=np.float64)
    
    # Add some random variation to make it more realistic
    np.random.seed(42)
    random_weights = np.random.random(TOTAL_BINS) * 0.1
    
    # Concentrate mass in the middle L* range and positive a* range
    for i in range(TOTAL_BINS):
        # Convert linear index to 3D coordinates
        l_idx = i // (12 * 12)
        a_idx = (i % (12 * 12)) // 12
        b_idx = i % 12
        
        # Higher weight for middle lightness and positive a* (red)
        l_weight = 1.0 if 2 <= l_idx <= 5 else 0.1
        a_weight = 1.0 if a_idx >= 6 else 0.1
        b_weight = 1.0  # Neutral for blue-yellow
        
        query_hist[i] = l_weight * a_weight * b_weight + random_weights[i]
    
    # Normalize to create a probability distribution
    query_hist = query_hist / query_hist.sum()
    
    logger.info(f"Query histogram created: sum={query_hist.sum():.6f}, "
                f"max={query_hist.max():.6f}, min={query_hist.min():.6f}")
    
    return query_hist


def load_test_histograms(dataset_path: Path, max_images: int = 20) -> tuple:
    """
    Load test images and generate histograms.
    
    Args:
        dataset_path: Path to the test dataset
        max_images: Maximum number of images to process
    
    Returns:
        tuple: (histograms, image_paths, image_ids)
    """
    logger.info(f"Loading test images from {dataset_path}")
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(dataset_path.glob(f"*{ext}"))
        image_files.extend(dataset_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        raise ValueError(f"No image files found in {dataset_path}")
    
    # Limit the number of images for testing
    image_files = image_files[:max_images]
    logger.info(f"Found {len(image_files)} images, processing {len(image_files)}")
    
    histograms = []
    image_paths = []
    image_ids = []
    
    for i, image_path in enumerate(image_files):
        try:
            logger.debug(f"Processing {image_path.name} ({i+1}/{len(image_files)})")
            
            # Load image and convert to Lab pixels
            lab_pixels = load_and_convert_image(image_path)
            
            # Generate histogram using the existing pipeline
            hist = build_histogram(lab_pixels)
            
            # Validate histogram
            if hist.shape != (TOTAL_BINS,) or not np.isclose(hist.sum(), 1.0, atol=1e-6):
                logger.warning(f"Invalid histogram for {image_path.name}, skipping")
                continue
            
            histograms.append(hist)
            image_paths.append(str(image_path))
            image_ids.append(image_path.stem)
            
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(histograms)} images")
    return histograms, image_paths, image_ids


def run_reranking_demo(query_hist: np.ndarray, candidate_hists: list, 
                       candidate_ids: list, num_candidates: int = 10) -> None:
    """
    Run the reranking demonstration.
    
    Args:
        query_hist: Query histogram
        candidate_hists: List of candidate histograms
        candidate_ids: List of candidate IDs
        num_candidates: Number of top candidates to display
    """
    logger.info("Running reranking demonstration...")
    
    if not candidate_hists:
        logger.error("No candidate histograms available")
        return
    
    # Time the reranking process
    start_time = time.time()
    
    # Run reranking
    results = rerank_candidates(
        query_hist=query_hist,
        candidate_hists=candidate_hists,
        candidate_ids=candidate_ids,
        max_candidates=num_candidates
    )
    
    end_time = time.time()
    rerank_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Display results
    logger.info(f"\n{'='*60}")
    logger.info("RERANKING RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Query: Synthetic 'reddish' color histogram")
    logger.info(f"Candidates processed: {len(candidate_hists)}")
    logger.info(f"Reranking time: {rerank_time:.2f} ms")
    logger.info(f"Average time per candidate: {rerank_time/len(candidate_hists):.2f} ms")
    
    logger.info(f"\nTop {min(num_candidates, len(results))} Results:")
    logger.info(f"{'Rank':<4} {'ID':<20} {'Distance':<12} {'Similarity':<12}")
    logger.info(f"{'-'*60}")
    
    for result in results[:num_candidates]:
        # Convert distance to similarity score (lower distance = higher similarity)
        max_possible_distance = 100.0  # Approximate maximum for our cost matrix
        similarity = max(0.0, 1.0 - (result.distance / max_possible_distance))
        
        logger.info(f"{result.rank:<4} {result.candidate_id:<20} "
                   f"{result.distance:<12.6f} {similarity:<12.2%}")
    
    # Performance analysis
    if len(results) > 1:
        distances = [r.distance for r in results]
        logger.info(f"\nPerformance Analysis:")
        logger.info(f"  - Distance range: [{min(distances):.6f}, {max(distances):.6f}]")
        logger.info(f"  - Mean distance: {np.mean(distances):.6f}")
        logger.info(f"  - Median distance: {np.median(distances):.6f}")
        logger.info(f"  - Standard deviation: {np.std(distances):.6f}")


def main():
    """Main function for the reranking test script."""
    parser = argparse.ArgumentParser(description="Test the Sinkhorn reranking system")
    parser.add_argument(
        "--dataset", 
        default="test-dataset-20",
        help="Test dataset to use (default: test-dataset-20)"
    )
    parser.add_argument(
        "--num-candidates", 
        type=int, 
        default=10,
        help="Number of top candidates to display (default: 10)"
    )
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only run validation tests, skip demo"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate configuration
        logger.info("Validating configuration...")
        validate_config()
        
        # Validate reranking system
        logger.info("Validating reranking system...")
        if not validate_reranking_system():
            logger.error("Reranking system validation failed!")
            return 1
        
        if args.validate_only:
            logger.info("Validation completed successfully!")
            return 0
        
        # Load test dataset
        dataset_path = Path("datasets") / args.dataset
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return 1
        
        # Load histograms
        histograms, image_paths, image_ids = load_test_histograms(
            dataset_path, max_images=min(50, args.num_candidates * 3)
        )
        
        if len(histograms) < 2:
            logger.error("Need at least 2 histograms for reranking demo")
            return 1
        
        # Create query histogram
        query_hist = create_synthetic_query_histogram()
        
        # Run reranking demo
        run_reranking_demo(query_hist, histograms, image_ids, args.num_candidates)
        
        logger.info("\n✅ Reranking demonstration completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Reranking test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
