# Image Processing Pipeline

## Chromatica Color Search Engine

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Image Processing Workflow](#image-processing-workflow)
5. [Usage Examples](#usage-examples)
6. [Performance Characteristics](#performance-characteristics)
7. [Error Handling](#error-handling)
8. [Testing and Validation](#testing-and-validation)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)

---

## Overview

The Image Processing Pipeline is the foundational component of the Chromatica color search engine, responsible for converting raw image files into normalized color histograms suitable for similarity search. This pipeline implements the algorithmic specifications from the critical instructions document, ensuring robust and efficient image processing.

### Key Features

- **Multi-format Support**: Handles JPEG, PNG, BMP, and TIFF images
- **Automatic Resizing**: Downsamples images to 256px max dimension for efficiency
- **Color Space Conversion**: Converts from sRGB to CIE Lab (D65 illuminant)
- **Tri-linear Soft Assignment**: Creates robust histograms using 8x12x12 binning grid
- **Vectorized Processing**: Optimized NumPy operations for high performance
- **Comprehensive Validation**: Input validation and error handling throughout

### Technology Stack

- **OpenCV**: Image loading and resizing operations
- **scikit-image**: sRGB to CIE Lab color space conversion
- **NumPy**: Vectorized histogram generation and numerical operations
- **Python 3.10+**: Modern Python with type hints and comprehensive error handling

---

## Architecture

### System Design

The Image Processing Pipeline follows a modular, layered architecture designed for scalability and maintainability:

```
┌─────────────────────────────────────────────────────────────┐
│                    Image Input Layer                        │
├─────────────────────────────────────────────────────────────┤
│                 Image Loading & Validation                  │
├─────────────────────────────────────────────────────────────┤
│                Color Space Conversion                       │
├─────────────────────────────────────────────────────────────┤
│                Histogram Generation                         │
├─────────────────────────────────────────────────────────────┤
│                 Validation & Output                         │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Image Input**: Raw image files in various formats (JPEG, PNG, BMP, TIFF)
2. **Loading & Validation**: Image loading with format validation and error handling
3. **Preprocessing**: Automatic resizing to 256px max dimension for efficiency
4. **Color Conversion**: BGR to RGB conversion, then sRGB to CIE Lab
5. **Histogram Generation**: Tri-linear soft assignment with 8x12x12 binning grid
6. **Normalization**: L1 normalization to create probability distributions
7. **Output**: 1,152-dimensional normalized histogram ready for indexing

### Component Responsibilities

#### Image Loading (`cv2.imread`)

- **Format Support**: Handles multiple image formats with automatic detection
- **Error Handling**: Graceful failure for corrupted or unsupported files
- **Memory Management**: Efficient loading with automatic cleanup

#### Color Space Conversion (`skimage.color.rgb2lab`)

- **sRGB Standard**: Assumes sRGB color profile for consistent results
- **CIE Lab Conversion**: Converts to perceptually uniform Lab space
- **D65 Illuminant**: Uses standard D65 white point for color accuracy

#### Histogram Generation (`build_histogram`)

- **Tri-linear Soft Assignment**: Distributes pixel counts across neighboring bins
- **Vectorized Operations**: Efficient NumPy-based processing
- **Normalization**: L1 normalization for probability distribution

---

## Core Components

### 1. Image Loading and Preprocessing

The pipeline begins with robust image loading and preprocessing to ensure consistent input quality:

```python
import cv2
from skimage import color
import numpy as np

def load_and_preprocess_image(image_path: str, max_dimension: int = 256) -> np.ndarray:
    """
    Load and preprocess an image for histogram generation.
    
    Args:
        image_path: Path to the image file
        max_dimension: Maximum dimension for resizing (default: 256)
    
    Returns:
        Preprocessed image in CIE Lab color space
        
    Raises:
        ValueError: If image cannot be loaded or processed
    """
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB (OpenCV loads in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if necessary while maintaining aspect ratio
    height, width = image_rgb.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        image_rgb = cv2.resize(image_rgb, (new_width, new_height))
    
    # Convert to CIE Lab color space
    image_lab = color.rgb2lab(image_rgb)
    
    return image_lab
```

### 2. Histogram Generation

The core histogram generation function implements tri-linear soft assignment:

```python
from chromatica.core.histogram import build_histogram

def generate_image_histogram(image_path: str) -> np.ndarray:
    """
    Generate a normalized color histogram from an image file.
    
    This function implements the complete pipeline from image loading
    to histogram generation, following the specifications in the
    critical instructions document.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Normalized histogram of shape (1152,) representing the
        color distribution in CIE Lab space
        
    Raises:
        ValueError: If image processing fails
        RuntimeError: If histogram generation fails
    """
    try:
        # Load and preprocess image
        image_lab = load_and_preprocess_image(image_path)
        
        # Reshape to (N, 3) array for histogram generation
        lab_pixels = image_lab.reshape(-1, 3)
        
        # Generate normalized histogram
        histogram = build_histogram(lab_pixels)
        
        return histogram
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate histogram for {image_path}: {e}")
```

### 3. Batch Processing

For efficient processing of large image collections:

```python
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def process_image_directory(
    directory_path: str,
    supported_extensions: set = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
) -> Dict[str, np.ndarray]:
    """
    Process all images in a directory and generate histograms.
    
    Args:
        directory_path: Path to directory containing images
        supported_extensions: Set of supported image file extensions
        
    Returns:
        Dictionary mapping image filenames to their histograms
        
    Raises:
        ValueError: If directory doesn't exist or is empty
    """
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    results = {}
    failed_images = []
    
    # Process each image file
    for image_file in directory.iterdir():
        if image_file.suffix.lower() in supported_extensions:
            try:
                logger.info(f"Processing image: {image_file.name}")
                histogram = generate_image_histogram(str(image_file))
                results[image_file.name] = histogram
                
            except Exception as e:
                logger.error(f"Failed to process {image_file.name}: {e}")
                failed_images.append(image_file.name)
    
    # Log processing summary
    logger.info(f"Successfully processed {len(results)} images")
    if failed_images:
        logger.warning(f"Failed to process {len(failed_images)} images: {failed_images}")
    
    return results
```

---

## Image Processing Workflow

### Step-by-Step Process

The complete image processing workflow follows these sequential steps:

#### 1. Image Input Validation

```python
def validate_image_input(image_path: str) -> bool:
    """Validate that an image file can be processed."""
    
    # Check file exists
    if not Path(image_path).exists():
        return False
    
    # Check file size (reasonable limits)
    file_size = Path(image_path).stat().st_size
    if file_size < 100 or file_size > 100 * 1024 * 1024:  # 100B to 100MB
        return False
    
    # Check file extension
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    if Path(image_path).suffix.lower() not in supported_extensions:
        return False
    
    return True
```

#### 2. Image Loading and Format Detection

```python
def load_image_with_validation(image_path: str) -> np.ndarray:
    """Load image with comprehensive validation."""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    
    # Check image dimensions
    if image.size == 0:
        raise ValueError(f"Image is empty: {image_path}")
    
    # Check for valid color channels
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must have 3 color channels, got shape: {image.shape}")
    
    return image
```

#### 3. Color Space Conversion Pipeline

```python
def convert_to_lab_color_space(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to CIE Lab color space."""
    
    # Ensure RGB format (OpenCV loads as BGR)
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"Image must be uint8, got {image_rgb.dtype}")
    
    # Convert to float [0, 1] range for scikit-image
    image_rgb_float = image_rgb.astype(np.float64) / 255.0
    
    # Convert to CIE Lab
    image_lab = color.rgb2lab(image_rgb_float)
    
    # Validate Lab values are within expected ranges
    l_range = [0.0, 100.0]
    a_range = [-86.0, 98.0]
    b_range = [-108.0, 95.0]
    
    if (np.any(image_lab[:, :, 0] < l_range[0]) or 
        np.any(image_lab[:, :, 0] > l_range[1]) or
        np.any(image_lab[:, :, 1] < a_range[0]) or 
        np.any(image_lab[:, :, 1] > a_range[1]) or
        np.any(image_lab[:, :, 2] < b_range[0]) or 
        np.any(image_lab[:, :, 2] > b_range[1])):
        
        logger.warning(f"Lab values outside expected ranges in {image_path}")
    
    return image_lab
```

#### 4. Histogram Generation and Validation

```python
def generate_and_validate_histogram(lab_pixels: np.ndarray) -> np.ndarray:
    """Generate histogram and validate output quality."""
    
    # Generate histogram
    histogram = build_histogram(lab_pixels)
    
    # Validation checks
    if histogram.shape != (1152,):
        raise ValueError(f"Histogram must have 1152 dimensions, got {histogram.shape}")
    
    if not np.isfinite(histogram).all():
        raise ValueError("Histogram contains non-finite values")
    
    if histogram.sum() == 0:
        raise ValueError("Histogram sums to zero")
    
    # Check normalization (should sum to 1.0)
    histogram_sum = histogram.sum()
    if abs(histogram_sum - 1.0) > 1e-6:
        logger.warning(f"Histogram not properly normalized: sum = {histogram_sum}")
        # Renormalize
        histogram = histogram / histogram_sum
    
    return histogram
```

### Complete Pipeline Integration

```python
def process_single_image(image_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Process a single image through the complete pipeline.
    
    Returns:
        Tuple of (histogram, metadata)
    """
    metadata = {
        'file_path': image_path,
        'file_size': Path(image_path).stat().st_size,
        'processing_time': 0.0,
        'errors': []
    }
    
    start_time = time.time()
    
    try:
        # Validate input
        if not validate_image_input(image_path):
            raise ValueError(f"Invalid image input: {image_path}")
        
        # Load and validate image
        image = load_image_with_validation(image_path)
        metadata['original_dimensions'] = image.shape[:2]
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if necessary
        height, width = image_rgb.shape[:2]
        if max(height, width) > 256:
            scale = 256 / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            metadata['resized_dimensions'] = (new_height, new_width)
            metadata['resize_scale'] = scale
        
        # Convert to Lab color space
        image_lab = convert_to_lab_color_space(image_rgb)
        
        # Generate histogram
        lab_pixels = image_lab.reshape(-1, 3)
        histogram = generate_and_validate_histogram(lab_pixels)
        
        # Update metadata
        metadata['histogram_shape'] = histogram.shape
        metadata['histogram_sum'] = float(histogram.sum())
        metadata['non_zero_bins'] = int(np.count_nonzero(histogram))
        
        return histogram, metadata
        
    except Exception as e:
        metadata['errors'].append(str(e))
        raise RuntimeError(f"Pipeline failed for {image_path}: {e}")
    
    finally:
        metadata['processing_time'] = time.time() - start_time
```

---

## Usage Examples

### Basic Usage

#### 1. Single Image Processing

```python
from chromatica.core.histogram import build_histogram
import cv2
from skimage import color

# Load and process a single image
image_path = "path/to/image.jpg"

# Load image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to Lab
image_lab = color.rgb2lab(image_rgb)

# Generate histogram
lab_pixels = image_lab.reshape(-1, 3)
histogram = build_histogram(lab_pixels)

print(f"Histogram shape: {histogram.shape}")
print(f"Sum: {histogram.sum():.6f}")
print(f"Non-zero bins: {np.count_nonzero(histogram)}/1152")
```

#### 2. Batch Processing with Error Handling

```python
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_dataset(dataset_path: str) -> Dict[str, np.ndarray]:
    """Process entire dataset with comprehensive error handling."""
    
    results = {}
    failed_images = []
    
    # Get all image files
    image_files = list(Path(dataset_path).glob("*.jpg")) + \
                  list(Path(dataset_path).glob("*.png"))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        try:
            logger.info(f"Processing {image_file.name}")
            histogram, metadata = process_single_image(str(image_file))
            results[image_file.name] = histogram
            
            # Log processing details
            logger.debug(f"Processed {image_file.name}: "
                        f"shape={metadata['histogram_shape']}, "
                        f"time={metadata['processing_time']:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to process {image_file.name}: {e}")
            failed_images.append(image_file.name)
    
    # Summary
    logger.info(f"Successfully processed {len(results)}/{len(image_files)} images")
    if failed_images:
        logger.warning(f"Failed images: {failed_images}")
    
    return results

# Usage
dataset_path = "datasets/test-dataset-50"
histograms = process_dataset(dataset_path)
```

#### 3. Integration with Testing Framework

```python
from tools.test_histogram_generation import run_histogram_tests

# Run comprehensive tests on a dataset
test_results = run_histogram_tests(
    dataset_path="datasets/test-dataset-200",
    output_dir="test_output",
    report_types=["validation", "performance", "quality"]
)

print(f"Test completed: {test_results['total_images']} images processed")
print(f"Success rate: {test_results['success_rate']:.1f}%")
print(f"Average processing time: {test_results['avg_time']:.3f}s")
```

---

## Performance Characteristics

### Processing Speed

The image processing pipeline is optimized for speed while maintaining quality:

#### Performance Metrics

- **Single Image Processing**: ~200ms average per image (256px max dimension)
- **Batch Processing**: ~5 images/second on development hardware
- **Memory Usage**: ~4.6KB per histogram (1152 × 4 bytes)
- **CPU Utilization**: Efficient vectorized operations using NumPy

#### Performance Optimization

```python
def optimize_pipeline_performance():
    """Configure pipeline for maximum performance."""
    
    # Use optimized OpenCV builds
    cv2.setUseOptimized(True)
    
    # Configure NumPy for optimal performance
    np.set_printoptions(precision=6, suppress=True)
    
    # Use float32 for memory efficiency (sufficient precision for histograms)
    # The build_histogram function already uses float32 internally
    
    # Consider using multiprocessing for batch operations
    # (implemented in the testing framework)
```

### Memory Management

#### Memory Usage Breakdown

- **Image Loading**: ~1-10MB per image (depending on resolution)
- **Lab Conversion**: ~2-20MB per image (float64 precision)
- **Histogram Generation**: ~4.6KB per final histogram
- **Batch Processing**: Linear scaling with number of images

#### Memory Optimization Strategies

```python
def memory_optimized_batch_processing(image_paths: List[str]) -> Dict[str, np.ndarray]:
    """
    Process images in batches to manage memory usage.
    
    This approach processes images in small batches to avoid
    memory issues with large datasets.
    """
    
    batch_size = 10  # Process 10 images at a time
    results = {}
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Process batch
        for image_path in batch_paths:
            try:
                histogram, metadata = process_single_image(image_path)
                results[Path(image_path).name] = histogram
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
        
        # Clear memory after each batch
        import gc
        gc.collect()
    
    return results
```

---

## Error Handling

### Comprehensive Error Management

The pipeline implements robust error handling at every stage:

#### Input Validation Errors

```python
class ImageProcessingError(Exception):
    """Base exception for image processing errors."""
    pass

class InvalidImageError(ImageProcessingError):
    """Raised when an image cannot be loaded or is invalid."""
    pass

class ColorConversionError(ImageProcessingError):
    """Raised when color space conversion fails."""
    pass

class HistogramGenerationError(ImageProcessingError):
    """Raised when histogram generation fails."""
    pass
```

#### Error Recovery Strategies

```python
def robust_image_processing(image_path: str, 
                           fallback_strategies: bool = True) -> np.ndarray:
    """
    Process image with fallback strategies for error recovery.
    
    Args:
        image_path: Path to image file
        fallback_strategies: Whether to attempt fallback processing
        
    Returns:
        Generated histogram
        
    Raises:
        ImageProcessingError: If all processing strategies fail
    """
    
    strategies = [
        # Primary strategy: standard processing
        lambda: process_single_image(image_path)[0],
        
        # Fallback 1: try with different resize method
        lambda: process_with_alternative_resize(image_path),
        
        # Fallback 2: try with different color conversion
        lambda: process_with_alternative_conversion(image_path),
        
        # Fallback 3: try with reduced quality
        lambda: process_with_reduced_quality(image_path)
    ]
    
    if not fallback_strategies:
        strategies = strategies[:1]  # Only primary strategy
    
    last_error = None
    
    for i, strategy in enumerate(strategies):
        try:
            return strategy()
        except Exception as e:
            last_error = e
            logger.warning(f"Strategy {i+1} failed: {e}")
            continue
    
    # All strategies failed
    raise ImageProcessingError(f"All processing strategies failed for {image_path}. "
                             f"Last error: {last_error}")
```

---

## Testing and Validation

### Comprehensive Testing Framework

The image processing pipeline includes a robust testing framework:

#### Test Types

1. **Unit Tests**: Individual function testing
2. **Integration Tests**: End-to-end pipeline testing
3. **Performance Tests**: Speed and memory usage validation
4. **Quality Tests**: Histogram quality metrics
5. **Error Handling Tests**: Exception and edge case testing
6. **Validation Tests**: Output format and range validation

#### Running Tests

```python
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_image_pipeline.py -v

# Run with coverage
python -m pytest tests/ --cov=src/chromatica --cov-report=html
```

#### Test Dataset Usage

```python
from tools.test_histogram_generation import run_histogram_tests

# Test with different dataset sizes
datasets = [
    "datasets/test-dataset-20",    # Quick development testing
    "datasets/test-dataset-50",    # Small-scale validation
    "datasets/test-dataset-200",   # Medium-scale testing
    "datasets/test-dataset-5000"   # Production-scale testing
]

for dataset in datasets:
    print(f"\nTesting dataset: {dataset}")
    results = run_histogram_tests(
        dataset_path=dataset,
        output_dir=f"test_output/{Path(dataset).name}",
        report_types=["validation", "performance", "quality"]
    )
    
    print(f"Success rate: {results['success_rate']:.1f}%")
    print(f"Average time: {results['avg_time']:.3f}s")
```

---

## Conclusion

The Image Processing Pipeline is a robust, efficient, and well-tested component that forms the foundation of the Chromatica color search engine. It successfully implements the algorithmic specifications from the critical instructions document, providing:

- **Reliable Image Processing**: Comprehensive error handling and validation
- **High Performance**: Vectorized operations and optimized algorithms
- **Quality Assurance**: Extensive testing and quality metrics
- **Flexibility**: Support for various image formats and processing strategies
- **Scalability**: Efficient batch processing and memory management

The pipeline is production-ready and serves as the first stage in the complete color search workflow, converting raw images into normalized histograms that are then indexed and searched using the FAISS HNSW index and Sinkhorn-EMD reranking system.

For more information about related components, see:
- [FAISS and DuckDB Wrappers](faiss_duckdb_guide.md)
- [Histogram Generation Guide](histogram_generation_guide.md)
- [Two-Stage Search Architecture](two_stage_search_architecture.md)
