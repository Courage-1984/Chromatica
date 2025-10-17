"""
Image processing pipeline for the Chromatica color search engine.

This module provides the main orchestration function for processing individual images
through the complete preprocessing pipeline. It handles image loading, resizing,
color space conversion, and histogram generation in a single, streamlined function.

The pipeline follows the algorithmic specifications from the critical instructions:
- Images are downscaled to a maximum dimension of 256px
- Color conversion follows the path: BGR -> RGB -> CIE Lab (D65 illuminant)
- Histograms are generated using the existing build_histogram function
- All operations are optimized for performance and memory efficiency

This module serves as the primary interface for the offline indexing pipeline and
can also be used for real-time image processing in the search API.
"""

import cv2
import numpy as np
import os
from skimage import color
from typing import Tuple, Optional
import logging
from pathlib import Path

from ..core.histogram import build_histogram
from ..utils.config import MAX_IMAGE_DIMENSION

# Set up logging for this module
logger = logging.getLogger(__name__)


def process_image(image_path: str) -> np.ndarray:
    """
    Process a single image through the complete preprocessing pipeline.

    This function orchestrates the full image processing workflow:
    1. Loads the image using OpenCV
    2. Resizes it to maintain maximum dimension of 256px
    3. Converts from BGR to RGB, then to CIE Lab color space
    4. Generates a normalized histogram using the build_histogram function
    5. Returns the final histogram ready for indexing or search

    The function follows the algorithmic specifications from the critical instructions,
    ensuring that all images are processed consistently for the color search engine.

    Args:
        image_path: Path to the image file to process. Supports common formats
                   (JPEG, PNG, BMP, etc.) that OpenCV can read.

    Returns:
        np.ndarray: A normalized histogram of shape (1152,) representing the
                   color distribution in the image. The histogram is L1-normalized
                   and ready for use with the FAISS index or distance calculations.

    Raises:
        FileNotFoundError: If the image file does not exist at the specified path.
        ValueError: If the image cannot be loaded or is corrupted.
        RuntimeError: If any step in the processing pipeline fails.

    Example:
        >>> # Process an image and get its histogram
        >>> histogram = process_image("path/to/image.jpg")
        >>> print(f"Histogram shape: {histogram.shape}")
        >>> print(f"Histogram sum: {histogram.sum():.6f}")
        Histogram shape: (1152,)
        Histogram sum: 1.000000

    Note:
        The function automatically handles different image formats and aspect ratios.
        Images are resized while maintaining their aspect ratio, with the longest
        side limited to 256 pixels for consistent processing speed and memory usage.
    """
    # Input validation
    if not image_path:
        raise ValueError("image_path cannot be empty")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")

    # Safe logging for Unicode filenames
    try:
        logger.debug(f"Processing image: {image_path}")
    except UnicodeEncodeError:
        logger.debug(f"Processing image: [Unicode filename] {image_path.name}")

    try:
        # Step 1: Load image using OpenCV with Unicode support
        logger.debug("Loading image with OpenCV")
        
        # Handle Unicode filenames on Windows
        image_path_str = str(image_path)
        if os.name == 'nt':  # Windows
            # Use numpy to read the file and then decode with OpenCV
            import numpy as np
            try:
                # Read file as bytes
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                # Convert to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                
                # Decode image with OpenCV
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as unicode_error:
                logger.warning(f"Unicode path handling failed, trying direct path: {unicode_error}")
                # Fallback to direct path
                image = cv2.imread(image_path_str)
        else:
            # On Unix-like systems, direct path should work
            image = cv2.imread(image_path_str)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        logger.debug(f"Loaded image with shape: {image.shape}")

        # Step 2: Resize image (max side 256px)
        resized_image = _resize_image(image)
        logger.debug(f"Resized image to shape: {resized_image.shape}")

        # Step 3: Convert BGR to RGB, then to CIE Lab
        lab_image = _convert_to_lab(resized_image)
        logger.debug(f"Converted to Lab with shape: {lab_image.shape}")

        # Step 4: Extract pixel data and build histogram
        lab_pixels = lab_image.reshape(-1, 3)
        logger.debug(f"Extracted {lab_pixels.shape[0]} pixels for histogram generation")

        histogram = build_histogram(lab_pixels)
        logger.debug(f"Generated histogram with shape: {histogram.shape}")

        # Step 5: Return the normalized histogram
        try:
            logger.info(f"Successfully processed image: {image_path}")
        except UnicodeEncodeError:
            logger.info(f"Successfully processed image: [Unicode filename] {image_path.name}")
        return histogram

    except Exception as e:
        try:
            logger.error(f"Failed to process image {image_path}: {str(e)}")
        except UnicodeEncodeError:
            logger.error(f"Failed to process image [Unicode filename] {image_path.name}: {str(e)}")
        raise RuntimeError(f"Image processing failed for {image_path}: {str(e)}") from e


def _resize_image(image: np.ndarray) -> np.ndarray:
    """
    Resize image to maintain maximum dimension of 256px while preserving aspect ratio.

    Args:
        image: Input image as a NumPy array with shape (height, width, channels).

    Returns:
        np.ndarray: Resized image with the same number of channels.
    """
    height, width = image.shape[:2]

    # Calculate scaling factor to maintain aspect ratio
    scale_factor = min(MAX_IMAGE_DIMENSION / max(height, width), 1.0)

    if scale_factor < 1.0:
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Use INTER_AREA for downscaling (better quality for images smaller than original)
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        logger.debug(f"Resized from {height}x{width} to {new_height}x{new_width}")
        return resized
    else:
        logger.debug("Image already within size limits, no resizing needed")
        return image


def _convert_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convert image from BGR (OpenCV format) to CIE Lab color space.

    This function performs the color space conversion following the critical
    instructions specification:
    1. Converts BGR to RGB (OpenCV loads images in BGR format)
    2. Converts RGB to CIE Lab using D65 illuminant (scikit-image default)

    Args:
        image: Input image as a NumPy array in BGR format with shape (height, width, 3).

    Returns:
        np.ndarray: Image in CIE Lab color space with shape (height, width, 3).
                   Values are in the ranges: L* [0, 100], a* [-86, 98], b* [-108, 95].

    Note:
        The conversion uses scikit-image's color module which automatically
        handles the sRGB to CIE Lab conversion with D65 illuminant as specified
        in the critical instructions.
    """
    # Convert BGR to RGB (OpenCV loads images in BGR format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB to CIE Lab using scikit-image
    # skimage.color.rgb2lab uses D65 illuminant by default
    lab_image = color.rgb2lab(rgb_image)

    logger.debug(
        f"Converted BGR->RGB->Lab, Lab range: L*[{lab_image[:,:,0].min():.1f}, {lab_image[:,:,0].max():.1f}], "
        f"a*[{lab_image[:,:,1].min():.1f}, {lab_image[:,:,1].max():.1f}], "
        f"b*[{lab_image[:,:,2].min():.1f}, {lab_image[:,:,2].max():.1f}]"
    )

    return lab_image


def validate_processed_image(histogram: np.ndarray, image_path: str) -> bool:
    """
    Validate that a processed image histogram meets all requirements.

    This function performs comprehensive validation of the generated histogram
    to ensure it meets the specifications for the color search engine.

    Args:
        histogram: The histogram array to validate.
        image_path: Path to the original image for logging purposes.

    Returns:
        bool: True if the histogram is valid, False otherwise.

    Raises:
        ValueError: If the histogram fails validation with detailed error message.
    """
    # Check shape
    if histogram.shape != (1152,):
        raise ValueError(
            f"Histogram for {image_path} has incorrect shape: {histogram.shape}, "
            f"expected (1152,)"
        )

    # Check normalization (should sum to 1.0)
    histogram_sum = histogram.sum()
    if not np.isclose(histogram_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Histogram for {image_path} is not properly normalized: "
            f"sum = {histogram_sum:.6f}, expected 1.0"
        )

    # Check for negative values
    if np.any(histogram < 0):
        raise ValueError(
            f"Histogram for {image_path} contains negative values: "
            f"min = {histogram.min():.6f}"
        )

    # Check for NaN or infinite values
    if not np.all(np.isfinite(histogram)):
        raise ValueError(f"Histogram for {image_path} contains NaN or infinite values")

    logger.debug(f"Histogram validation passed for {image_path}")
    return True
