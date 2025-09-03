# Test Image Pipeline Tool

## Overview

The `test_image_pipeline.py` tool is a comprehensive testing script for the image processing pipeline in Chromatica. It tests the new `process_image` function from the indexing pipeline module using existing test datasets, validating that the pipeline correctly loads and processes images, generates valid histograms, and integrates with the existing histogram generation system.

## Purpose

This tool is designed to:

- Test the image processing pipeline functionality
- Validate image loading and processing operations
- Test histogram generation and validation
- Verify integration with existing histogram systems
- Provide comprehensive testing of the image processing workflow
- Serve as a quality assurance framework for the pipeline

## Features

- **Pipeline Testing**: Comprehensive testing of image processing workflow
- **Image Validation**: Verify image loading and processing operations
- **Histogram Testing**: Validate histogram generation and quality
- **Integration Testing**: Test system integration and compatibility
- **Performance Testing**: Measure processing speed and efficiency
- **Error Handling**: Test edge cases and failure modes

## Usage

### Basic Usage

```bash
# Activate virtual environment first
venv311\Scripts\activate

# Run all tests
python tools/test_image_pipeline.py

# Run from project root directory
cd /path/to/Chromatica
python tools/test_image_pipeline.py
```

### What the Tool Tests

The tool runs comprehensive tests on:

1. **Image Processing**: Image loading, conversion, and processing
2. **Histogram Generation**: Color histogram creation and validation
3. **Pipeline Integration**: End-to-end workflow validation
4. **Performance**: Processing speed and efficiency metrics
5. **Error Handling**: Invalid input handling and edge cases

## Prerequisites

### Environment Setup

```bash
# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify test datasets
ls datasets/test-dataset-20/
```

### Required Components

- Test datasets in `datasets/` directory
- All Chromatica modules installed
- Virtual environment activated
- Sufficient memory for image processing

## Core Test Functions

### 1. Single Image Testing

```python
def test_single_image(image_path: str) -> bool:
    """Test processing of a single image."""

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
```

**What it tests:**

- Individual image processing functionality
- Processing time measurement
- Histogram validation and quality checks
- Error handling for individual images
- Success/failure reporting

### 2. Dataset Testing

```python
def test_dataset(dataset_path: str) -> tuple[int, int]:
    """Test processing of all images in a dataset directory."""

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

    for i, image_file in enumerate(image_files):
        logger.info(f"Processing {i+1}/{total_count}: {image_file.name}")

        if test_single_image(str(image_file)):
            success_count += 1
        else:
            logger.warning(f"Failed to process {image_file.name}")

    # Report results
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    logger.info(f"Dataset processing complete: {success_count}/{total_count} successful ({success_rate:.1f}%)")

    return success_count, total_count
```

**What it tests:**

- Batch image processing functionality
- Multiple image format support
- Success rate calculation and reporting
- Progress tracking and logging
- Dataset-level validation

### 3. Pipeline Validation

```python
def validate_pipeline_integration():
    """Validate that the pipeline integrates correctly with existing systems."""

    logger.info("Validating pipeline integration...")

    # Test with a known good image
    test_image_path = "datasets/test-dataset-20/test.jpg"

    if not os.path.exists(test_image_path):
        logger.warning(f"Test image not found: {test_image_path}")
        return False

    try:
        # Process image through pipeline
        histogram = process_image(test_image_path)

        # Validate histogram properties
        if histogram.shape != (TOTAL_BINS,):
            logger.error(f"Invalid histogram shape: {histogram.shape}, expected {TOTAL_BINS}")
            return False

        if not np.isclose(histogram.sum(), 1.0, atol=1e-6):
            logger.error(f"Histogram not normalized: sum = {histogram.sum()}")
            return False

        if np.any(histogram < 0):
            logger.error("Histogram contains negative values")
            return False

        # Test histogram quality metrics
        non_zero_bins = np.count_nonzero(histogram)
        if non_zero_bins < 10:
            logger.warning(f"Very sparse histogram: only {non_zero_bins} non-zero bins")

        # Calculate entropy
        entropy = -np.sum(histogram * np.log(histogram + 1e-10))
        logger.info(f"Histogram entropy: {entropy:.4f}")

        if entropy < 1.0:
            logger.warning("Low entropy histogram - may indicate poor color distribution")

        logger.info("✅ Pipeline integration validation passed")
        return True

    except Exception as e:
        logger.error(f"Pipeline integration validation failed: {e}")
        return False
```

**What it validates:**

- Pipeline integration with existing systems
- Histogram quality and properties
- Color distribution analysis
- Entropy calculation and assessment
- System compatibility verification

### 4. Performance Testing

```python
def benchmark_pipeline_performance(dataset_path: str = "datasets/test-dataset-20"):
    """Benchmark the performance of the image processing pipeline."""

    logger.info(f"Benchmarking pipeline performance with {dataset_path}")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return {}

    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        logger.warning("No image files found for benchmarking")
        return {}

    # Limit to first 10 images for benchmarking
    test_images = image_files[:10]
    logger.info(f"Benchmarking with {len(test_images)} images")

    # Performance metrics
    processing_times = []
    histogram_shapes = []
    success_count = 0

    for i, image_file in enumerate(test_images):
        try:
            logger.info(f"Benchmarking {i+1}/{len(test_images)}: {image_file.name}")

            # Measure processing time
            start_time = time.time()
            histogram = process_image(str(image_file))
            processing_time = time.time() - start_time

            # Record metrics
            processing_times.append(processing_time)
            histogram_shapes.append(histogram.shape)
            success_count += 1

            logger.info(f"  Processing time: {processing_time:.3f}s")
            logger.info(f"  Histogram shape: {histogram.shape}")

        except Exception as e:
            logger.error(f"  Failed to process {image_file.name}: {e}")

    # Calculate performance statistics
    if processing_times:
        avg_time = np.mean(processing_times)
        min_time = np.min(processing_times)
        max_time = np.max(processing_times)
        std_time = np.std(processing_times)

        logger.info("\nPerformance Statistics:")
        logger.info("=" * 40)
        logger.info(f"Successful processing: {success_count}/{len(test_images)}")
        logger.info(f"Average processing time: {avg_time:.3f}s")
        logger.info(f"Time range: {min_time:.3f}s - {max_time:.3f}s")
        logger.info(f"Standard deviation: {std_time:.3f}s")
        logger.info(f"Images per second: {1/avg_time:.2f}")

        # Performance thresholds
        if avg_time < 0.5:
            logger.info("✅ Excellent performance")
        elif avg_time < 1.0:
            logger.info("✅ Good performance")
        elif avg_time < 2.0:
            logger.info("⚠️  Acceptable performance")
        else:
            logger.warning("❌ Performance below acceptable threshold")

        return {
            "success_count": success_count,
            "total_count": len(test_images),
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_time": std_time,
            "images_per_second": 1/avg_time
        }

    return {}
```

**What it measures:**

- Processing time per image
- Performance consistency and variability
- Throughput metrics (images per second)
- Performance thresholds and assessment
- Success rate and reliability

### 5. Error Handling Testing

```python
def test_error_handling():
    """Test error handling for various failure scenarios."""

    logger.info("Testing error handling...")

    # Test cases that should fail
    error_test_cases = [
        {
            "name": "Non-existent file",
            "path": "non_existent_image.jpg",
            "expected_error": True
        },
        {
            "name": "Invalid file extension",
            "path": "test.txt",
            "expected_error": True
        },
        {
            "name": "Corrupted image file",
            "path": "corrupted_image.jpg",
            "expected_error": True
        },
        {
            "name": "Empty file",
            "path": "empty_file.jpg",
            "expected_error": True
        }
    ]

    error_handling_works = True

    for test_case in error_test_cases:
        logger.info(f"Testing: {test_case['name']}")

        try:
            # Attempt to process the file
            histogram = process_image(test_case["path"])

            if test_case["expected_error"]:
                logger.warning(f"  Expected error but got histogram: {histogram.shape}")
                error_handling_works = False
            else:
                logger.info(f"  ✅ Correctly processed valid file")

        except Exception as e:
            if test_case["expected_error"]:
                logger.info(f"  ✅ Correctly caught error: {type(e).__name__}")
            else:
                logger.error(f"  ❌ Unexpected error: {e}")
                error_handling_works = False

    if error_handling_works:
        logger.info("✅ Error handling test passed")
    else:
        logger.warning("⚠️  Error handling test had issues")

    return error_handling_works
```

**What it tests:**

- Invalid file path handling
- Unsupported file format handling
- Corrupted file handling
- Empty file handling
- Error propagation and catching

## Complete Test Suite

### Running All Tests

```python
def run_complete_pipeline_tests():
    """Run the complete image pipeline test suite."""

    logger.info("🚀 Chromatica Image Pipeline Test Suite")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_results = {}

    try:
        # Test pipeline integration
        logger.info("\n🔗 Testing Pipeline Integration")
        logger.info("-" * 40)

        integration_success = validate_pipeline_integration()
        test_results["integration"] = integration_success

        # Test single image processing
        logger.info("\n🖼️  Testing Single Image Processing")
        logger.info("-" * 40)

        test_image_path = "datasets/test-dataset-20/test.jpg"
        if os.path.exists(test_image_path):
            single_image_success = test_single_image(test_image_path)
            test_results["single_image"] = single_image_success
        else:
            logger.warning(f"Test image not found: {test_image_path}")
            test_results["single_image"] = False

        # Test dataset processing
        logger.info("\n📁 Testing Dataset Processing")
        logger.info("-" * 40)

        success_count, total_count = test_dataset("datasets/test-dataset-20")
        dataset_success = success_count > 0 and success_count == total_count
        test_results["dataset_processing"] = dataset_success
        test_results["dataset_stats"] = {"success": success_count, "total": total_count}

        # Test error handling
        logger.info("\n⚠️  Testing Error Handling")
        logger.info("-" * 40)

        error_handling_success = test_error_handling()
        test_results["error_handling"] = error_handling_success

        # Performance benchmarking
        logger.info("\n⚡ Performance Benchmarking")
        logger.info("-" * 40)

        performance_results = benchmark_pipeline_performance()
        test_results["performance"] = performance_results

        # Overall success assessment
        critical_tests = ["integration", "single_image", "dataset_processing", "error_handling"]
        passed_tests = sum(test_results.get(test, False) for test in critical_tests)
        total_tests = len(critical_tests)

        test_results["overall_success"] = passed_tests == total_tests

        # Summary
        logger.info("\n📊 Test Summary")
        logger.info("=" * 60)

        for test_name in critical_tests:
            status = "✅ PASS" if test_results.get(test_name, False) else "❌ FAIL"
            logger.info(f"  {test_name.upper()}: {status}")

        if test_results["overall_success"]:
            logger.info(f"\n🎉 All critical tests passed! ({passed_tests}/{total_tests})")
        else:
            logger.info(f"\n⚠️  {total_tests - passed_tests} critical test(s) failed")

        return test_results

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        test_results["overall_success"] = False
        return test_results
```

## Integration Examples

### With CI/CD Pipeline

```python
#!/usr/bin/env python3
"""
CI/CD integration script for image pipeline testing.
"""

import sys
from tools.test_image_pipeline import run_complete_pipeline_tests

def main():
    """Run tests and exit with appropriate code for CI/CD."""

    try:
        results = run_complete_pipeline_tests()

        if results.get("overall_success", False):
            print("✅ CI/CD: All image pipeline tests passed")
            sys.exit(0)  # Success
        else:
            print("❌ CI/CD: Image pipeline tests failed")
            sys.exit(1)  # Failure

    except Exception as e:
        print(f"❌ CI/CD: Test suite error: {e}")
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()
```

### With Development Workflow

```python
def development_testing():
    """Run image pipeline tests during development."""

    print("Running image pipeline development tests...")
    results = run_complete_pipeline_tests()

    if results.get("overall_success"):
        print("✅ Development tests passed - ready for commit")
    else:
        print("❌ Development tests failed - fix issues before commit")

    return results
```

## Best Practices

### Test Development

1. **Comprehensive Coverage**: Test all major functionality areas
2. **Real Data**: Use actual image datasets for realistic testing
3. **Performance Metrics**: Measure processing efficiency consistently
4. **Error Scenarios**: Include edge cases and failure modes
5. **Integration Testing**: Verify system compatibility

### Test Execution

1. **Environment Setup**: Ensure virtual environment is activated
2. **Dataset Availability**: Verify test datasets are accessible
3. **Resource Management**: Monitor memory usage during testing
4. **Clean State**: Start with fresh test data when possible
5. **Result Validation**: Verify test results are meaningful

## Troubleshooting

### Common Issues

1. **Import Errors**: Check virtual environment and Python path
2. **Dataset Issues**: Verify test datasets exist and are accessible
3. **Memory Issues**: Reduce dataset size for memory-constrained environments
4. **Performance Issues**: Check system resources and configuration

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
validate_pipeline_integration()
test_single_image("datasets/test-dataset-20/test.jpg")
benchmark_pipeline_performance("datasets/test-dataset-20")
```

## Dependencies

### Required Packages

- `numpy`: Numerical operations
- `opencv-python`: Image loading and processing
- `scikit-image`: Color space conversion
- `chromatica`: Chromatica core modules

### Installation

```bash
# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The test image pipeline tool provides comprehensive validation of the Chromatica image processing pipeline and serves as both a development tool and a quality assurance framework for the image processing workflow.
