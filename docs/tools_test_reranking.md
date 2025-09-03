# Test Reranking Tool

## Overview

The `test_reranking.py` tool is a comprehensive testing script for the Sinkhorn reranking system in Chromatica. It demonstrates the reranking functionality by loading test images, generating histograms, creating synthetic query histograms, running the reranking pipeline, and displaying results with performance metrics.

## Purpose

This tool is designed to:

- Test the Sinkhorn-EMD reranking functionality
- Validate reranking system integration
- Measure reranking performance and accuracy
- Test with synthetic and real image data
- Provide comprehensive reranking validation
- Serve as a quality assurance framework

## Features

- **Reranking Testing**: Comprehensive Sinkhorn-EMD reranking validation
- **Image Processing**: Real image loading and histogram generation
- **Synthetic Queries**: Test with controlled query histograms
- **Performance Benchmarking**: Measure reranking speed and efficiency
- **Result Analysis**: Compare initial and reranked results
- **System Validation**: Verify reranking system integrity

## Usage

### Basic Usage

```bash
# Activate virtual environment first
venv311\Scripts\activate

# Run with default settings
python tools/test_reranking.py

# Run with custom dataset
python tools/test_reranking.py --dataset test-dataset-50

# Run with custom number of candidates
python tools/test_reranking.py --num-candidates 20
```

### Command Line Options

| Option             | Description                    | Default           |
| ------------------ | ------------------------------ | ----------------- |
| `--dataset`        | Test dataset directory name    | `test-dataset-20` |
| `--num-candidates` | Number of candidates to rerank | `10`              |
| `--verbose`        | Enable detailed logging        | False             |

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

### 1. Image Loading and Conversion

```python
def load_and_convert_image(image_path: Path) -> np.ndarray:
    """Load an image and convert it to Lab color space pixels."""

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
```

**What it does:**

- Loads images using OpenCV
- Converts BGR to RGB color format
- Resizes large images to manageable dimensions
- Converts RGB to Lab color space (D65 illuminant)
- Returns Lab pixel array for histogram generation

### 2. Synthetic Query Histogram Creation

```python
def create_synthetic_query_histogram() -> np.ndarray:
    """Create a synthetic query histogram for testing."""

    logger.info("Creating synthetic query histogram for 'reddish' colors...")

    # Create a histogram with mass concentrated in red regions
    # In Lab space, red corresponds to positive a* values
    query_hist = np.zeros(TOTAL_BINS, dtype=np.float64)

    # Add some random variation to make it more realistic
    np.random.seed(42)

    # Focus on red regions (positive a* values)
    # This is a simplified approach - in practice, you'd use the actual bin mapping
    red_bins = np.random.choice(TOTAL_BINS, size=100, replace=False)
    query_hist[red_bins] = np.random.uniform(0.1, 1.0, size=100)

    # Normalize the histogram
    query_hist = query_hist / query_hist.sum()

    logger.info(f"Created synthetic query histogram with {np.count_nonzero(query_hist)} non-zero bins")
    return query_hist
```

**What it does:**

- Creates controlled test histograms for specific color queries
- Focuses on particular color regions (e.g., reddish colors)
- Adds realistic variation to test data
- Ensures proper histogram normalization
- Provides reproducible test scenarios

### 3. Reranking Pipeline Testing

```python
def test_reranking_pipeline(dataset_path: str, num_candidates: int = 10):
    """Test the complete reranking pipeline."""

    logger.info(f"Testing reranking pipeline with {dataset_path}")
    logger.info(f"Number of candidates: {num_candidates}")

    # Load and process test images
    dataset_dir = Path(f"datasets/{dataset_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    # Get list of image files
    image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
    if not image_files:
        raise ValueError(f"No image files found in {dataset_dir}")

    logger.info(f"Found {len(image_files)} images in dataset")

    # Process images and generate histograms
    histograms = []
    image_paths = []

    for image_file in image_files[:min(20, len(image_files))]:  # Limit to 20 images for testing
        try:
            # Load and convert image
            lab_pixels = load_and_convert_image(image_file)

            # Generate histogram
            histogram = build_histogram(lab_pixels)

            # Validate histogram
            if histogram.shape == (TOTAL_BINS,) and np.isclose(histogram.sum(), 1.0, atol=1e-6):
                histograms.append(histogram)
                image_paths.append(str(image_file))
                logger.debug(f"Processed {image_file.name}")
            else:
                logger.warning(f"Invalid histogram for {image_file.name}, skipping")

        except Exception as e:
            logger.error(f"Failed to process {image_file.name}: {e}")
            continue

    if not histograms:
        raise RuntimeError("No valid histograms generated from test dataset")

    logger.info(f"Successfully processed {len(histograms)} images")

    # Create synthetic query
    query_histogram = create_synthetic_query_histogram()

    # Test reranking
    logger.info("Testing reranking functionality...")

    try:
        # Get initial candidates (simple similarity)
        initial_scores = []
        for hist in histograms:
            # Simple cosine similarity
            similarity = np.dot(query_histogram, hist) / (np.linalg.norm(query_histogram) * np.linalg.norm(hist))
            initial_scores.append(similarity)

        # Sort by initial scores
        initial_indices = np.argsort(initial_scores)[::-1]
        initial_candidates = [histograms[i] for i in initial_indices[:num_candidates]]

        logger.info(f"Initial top {num_candidates} candidates selected")

        # Apply reranking
        start_time = time.time()
        reranked_candidates = rerank_candidates(query_histogram, initial_candidates)
        rerank_time = time.time() - start_time

        logger.info(f"Reranking completed in {rerank_time*1000:.2f} ms")

        # Display results
        display_reranking_results(
            image_paths, initial_indices, initial_scores,
            reranked_candidates, num_candidates
        )

        return {
            "initial_candidates": initial_candidates,
            "reranked_candidates": reranked_candidates,
            "rerank_time_ms": rerank_time * 1000,
            "num_candidates": num_candidates
        }

    except Exception as e:
        logger.error(f"Reranking pipeline failed: {e}")
        raise
```

**What it tests:**

- Complete reranking workflow
- Image loading and histogram generation
- Initial candidate selection
- Sinkhorn-EMD reranking execution
- Performance timing and result analysis

### 4. Result Display and Analysis

```python
def display_reranking_results(
    image_paths: list,
    initial_indices: np.ndarray,
    initial_scores: list,
    reranked_candidates: list,
    num_candidates: int
):
    """Display and analyze reranking results."""

    logger.info("\n" + "="*60)
    logger.info("RERANKING RESULTS ANALYSIS")
    logger.info("="*60)

    # Display initial ranking
    logger.info("\nInitial Ranking (Cosine Similarity):")
    logger.info("-" * 40)

    for i in range(min(num_candidates, len(initial_indices))):
        idx = initial_indices[i]
        score = initial_scores[idx]
        image_name = Path(image_paths[idx]).name
        logger.info(f"{i+1:2d}. {image_name:<20} Score: {score:.4f}")

    # Display reranked results
    logger.info("\nReranked Results (Sinkhorn-EMD):")
    logger.info("-" * 40)

    for i, candidate in enumerate(reranked_candidates):
        if i < len(image_paths):
            image_name = Path(image_paths[i]).name
            logger.info(f"{i+1:2d}. {image_name:<20}")

    # Analyze ranking changes
    logger.info("\nRanking Analysis:")
    logger.info("-" * 40)

    # Check if top result changed
    initial_top = Path(image_paths[initial_indices[0]]).name if len(initial_indices) > 0 else "N/A"
    reranked_top = Path(image_paths[0]).name if len(image_paths) > 0 else "N/A"

    if initial_top == reranked_top:
        logger.info(f"✅ Top result maintained: {initial_top}")
    else:
        logger.info(f"🔄 Top result changed: {initial_top} → {reranked_top}")

    # Calculate ranking stability
    if len(initial_indices) >= num_candidates:
        initial_top_k = set(initial_indices[:num_candidates])
        reranked_top_k = set(range(num_candidates))

        overlap = len(initial_top_k.intersection(reranked_top_k))
        stability = overlap / num_candidates

        logger.info(f"Ranking stability: {overlap}/{num_candidates} ({stability:.1%})")

        if stability < 0.5:
            logger.info("⚠️  Significant ranking changes detected")
        elif stability < 0.8:
            logger.info("ℹ️  Moderate ranking changes detected")
        else:
            logger.info("✅ High ranking stability maintained")
```

**What it analyzes:**

- Initial vs. reranked rankings
- Ranking stability and changes
- Performance metrics
- Result quality assessment

### 5. Performance Benchmarking

```python
def benchmark_reranking_performance(dataset_path: str, num_candidates: int = 10):
    """Benchmark reranking performance characteristics."""

    logger.info(f"Benchmarking reranking performance with {dataset_path}")

    # Run multiple reranking tests
    test_results = []

    for test_run in range(5):  # 5 test runs
        logger.info(f"\nTest run {test_run + 1}/5")

        try:
            # Create different synthetic queries for variety
            np.random.seed(42 + test_run)
            query_histogram = create_synthetic_query_histogram()

            # Process subset of images for benchmarking
            dataset_dir = Path(f"datasets/{dataset_path}")
            image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))

            histograms = []
            for image_file in image_files[:15]:  # Use 15 images for benchmarking
                try:
                    lab_pixels = load_and_convert_image(image_file)
                    histogram = build_histogram(lab_pixels)
                    if histogram.shape == (TOTAL_BINS,) and np.isclose(histogram.sum(), 1.0, atol=1e-6):
                        histograms.append(histogram)
                except Exception as e:
                    continue

            if len(histograms) < num_candidates:
                logger.warning(f"Not enough histograms for test run {test_run + 1}")
                continue

            # Select random candidates
            candidate_indices = np.random.choice(len(histograms), num_candidates, replace=False)
            candidates = [histograms[i] for i in candidate_indices]

            # Benchmark reranking
            start_time = time.time()
            reranked_candidates = rerank_candidates(query_histogram, candidates)
            rerank_time = time.time() - start_time

            test_results.append({
                "run": test_run + 1,
                "num_candidates": num_candidates,
                "rerank_time_ms": rerank_time * 1000,
                "success": True
            })

            logger.info(f"  Reranking time: {rerank_time*1000:.2f} ms")

        except Exception as e:
            logger.error(f"  Test run {test_run + 1} failed: {e}")
            test_results.append({
                "run": test_run + 1,
                "num_candidates": num_candidates,
                "rerank_time_ms": 0,
                "success": False,
                "error": str(e)
            })

    # Analyze benchmark results
    if test_results:
        successful_runs = [r for r in test_results if r["success"]]

        if successful_runs:
            times = [r["rerank_time_ms"] for r in successful_runs]

            logger.info("\n" + "="*50)
            logger.info("PERFORMANCE BENCHMARK RESULTS")
            logger.info("="*50)

            logger.info(f"Successful runs: {len(successful_runs)}/{len(test_results)}")
            logger.info(f"Average reranking time: {np.mean(times):.2f} ms")
            logger.info(f"Time range: {np.min(times):.2f} - {np.max(times):.2f} ms")
            logger.info(f"Standard deviation: {np.std(times):.2f} ms")
            logger.info(f"Time per candidate: {np.mean(times) / num_candidates:.2f} ms/candidate")

            return {
                "successful_runs": len(successful_runs),
                "total_runs": len(test_results),
                "avg_time_ms": np.mean(times),
                "min_time_ms": np.min(times),
                "max_time_ms": np.max(times),
                "std_time_ms": np.std(times)
            }

    return {}
```

**What it measures:**

- Reranking execution time
- Performance consistency across runs
- Scalability with candidate count
- System reliability and error rates

## Complete Test Suite

### Running All Tests

```python
def run_complete_reranking_tests(dataset_path: str = "test-dataset-20", num_candidates: int = 10):
    """Run the complete reranking test suite."""

    logger.info("🚀 Chromatica Reranking System Test Suite")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Candidates: {num_candidates}")
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_results = {}

    try:
        # Test reranking pipeline
        logger.info("\n🔍 Testing Reranking Pipeline")
        logger.info("-" * 40)

        pipeline_results = test_reranking_pipeline(dataset_path, num_candidates)
        test_results["pipeline"] = pipeline_results

        # Performance benchmarking
        logger.info("\n⚡ Performance Benchmarking")
        logger.info("-" * 40)

        performance_results = benchmark_reranking_performance(dataset_path, num_candidates)
        test_results["performance"] = performance_results

        # System validation
        logger.info("\n✅ System Validation")
        logger.info("-" * 40)

        try:
            system_valid = validate_reranking_system()
            test_results["system_validation"] = system_valid
            logger.info("✅ Reranking system validation passed")
        except Exception as e:
            logger.error(f"❌ System validation failed: {e}")
            test_results["system_validation"] = False

        # Overall success assessment
        test_results["overall_success"] = (
            "pipeline" in test_results and
            "performance" in test_results and
            test_results.get("system_validation", False)
        )

        # Summary
        logger.info("\n📊 Test Summary")
        logger.info("=" * 60)

        if test_results["overall_success"]:
            logger.info("🎉 All reranking tests passed!")
        else:
            logger.info("⚠️  Some reranking tests failed")

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
CI/CD integration script for reranking testing.
"""

import sys
from tools.test_reranking import run_complete_reranking_tests

def main():
    """Run tests and exit with appropriate code for CI/CD."""

    try:
        results = run_complete_reranking_tests(
            dataset_path="test-dataset-20",
            num_candidates=10
        )

        if results.get("overall_success", False):
            print("✅ CI/CD: All reranking tests passed")
            sys.exit(0)  # Success
        else:
            print("❌ CI/CD: Reranking tests failed")
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
    """Run reranking tests during development."""

    print("Running reranking development tests...")
    results = run_complete_reranking_tests(
        dataset_path="test-dataset-20",
        num_candidates=5  # Smaller test for development
    )

    if results.get("overall_success"):
        print("✅ Development tests passed - ready for commit")
    else:
        print("❌ Development tests failed - fix issues before commit")

    return results
```

## Best Practices

### Test Development

1. **Comprehensive Coverage**: Test all reranking components
2. **Real Data**: Use actual image datasets for realistic testing
3. **Performance Metrics**: Measure timing and efficiency consistently
4. **Error Scenarios**: Include edge cases and failure modes
5. **Validation**: Verify reranking quality and accuracy

### Test Execution

1. **Environment Setup**: Ensure virtual environment is activated
2. **Dataset Availability**: Verify test datasets are accessible
3. **Resource Management**: Monitor memory usage during testing
4. **Clean State**: Start with fresh test data when possible
5. **Result Analysis**: Verify reranking improvements are meaningful

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce dataset size or candidate count
2. **Import Errors**: Check virtual environment and Python path
3. **Dataset Issues**: Verify test datasets exist and are accessible
4. **Performance Issues**: Check system resources and configuration

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
test_reranking_pipeline("test-dataset-20", 5)
benchmark_reranking_performance("test-dataset-20", 5)
```

## Dependencies

### Required Packages

- `numpy`: Numerical operations
- `opencv-python`: Image loading and processing
- `scikit-image`: Color space conversion
- `POT`: Python Optimal Transport for Sinkhorn-EMD

### Installation

```bash
# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The test reranking tool provides comprehensive validation of the Chromatica Sinkhorn-EMD reranking system and serves as both a development tool and a quality assurance framework for the reranking pipeline.
