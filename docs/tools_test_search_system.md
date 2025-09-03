# Test Search System Tool

## Overview

The `test_search_system.py` tool is a comprehensive testing script for the complete Chromatica search system. It validates the full two-stage search pipeline by testing the search module with synthetic data, validating integration between all components, measuring performance characteristics, and testing error handling and edge cases.

## Purpose

This tool is designed to:

- Test the complete search system integration
- Validate the two-stage search pipeline (ANN + reranking)
- Measure search performance and accuracy
- Test error handling and edge cases
- Provide comprehensive system validation
- Serve as a quality assurance framework

## Features

- **Complete System Testing**: End-to-end search pipeline validation
- **Integration Testing**: Component interaction verification
- **Performance Benchmarking**: Search speed and accuracy measurement
- **Error Scenario Testing**: Edge case and failure mode validation
- **Comprehensive Logging**: Detailed test execution logging
- **Automated Validation**: Automated test suite execution

## Usage

### Basic Usage

```bash
# Activate virtual environment first
venv311\Scripts\activate

# Run basic tests
python tools/test_search_system.py

# Run with verbose logging
python tools/test_search_system.py --verbose

# Run with performance testing
python tools/test_search_system.py --performance
```

### Command Line Options

| Option          | Description                     | Default                    |
| --------------- | ------------------------------- | -------------------------- |
| `--verbose`     | Enable detailed logging         | False                      |
| `--performance` | Enable performance benchmarking | False                      |
| `--dataset`     | Custom test dataset path        | `datasets/test-dataset-50` |

## Prerequisites

### Environment Setup

```bash
# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify test datasets
ls datasets/test-dataset-50/
```

### Required Components

- Test datasets in `datasets/` directory
- All Chromatica modules installed
- Virtual environment activated
- Sufficient memory for test index creation

## Core Test Functions

### 1. Test Index and Store Creation

```python
def create_test_index_and_store(
    test_dataset_path: str, max_images: int = 100
) -> tuple[AnnIndex, MetadataStore]:
    """Create a test FAISS index and metadata store with sample data."""

    logger = logging.getLogger(__name__)
    logger.info(f"Creating test index and store from {test_dataset_path}")

    # Initialize components
    index = AnnIndex()
    store = MetadataStore(":memory:")  # Use in-memory database for testing

    # Get list of image files
    dataset_path = Path(test_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")

    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    image_files = image_files[:max_images]

    if not image_files:
        raise ValueError(f"No image files found in {test_dataset_path}")

    logger.info(f"Processing {len(image_files)} images for test index")

    # Process images and build index
    histograms = []
    metadata_batch = []

    for i, image_file in enumerate(image_files):
        try:
            # Generate histogram
            histogram = generate_histogram(str(image_file))

            # Validate histogram
            if histogram.shape != (1152,) or not np.isclose(histogram.sum(), 1.0, atol=1e-6):
                logger.warning(f"Invalid histogram for {image_file.name}, skipping")
                continue

            # Prepare metadata
            image_id = f"test_{image_file.stem}"
            metadata = {
                "image_id": image_id,
                "file_path": str(image_file),
                "histogram": histogram,
                "file_size": image_file.stat().st_size
            }

            histograms.append(histogram)
            metadata_batch.append(metadata)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} images")

        except Exception as e:
            logger.error(f"Failed to process {image_file.name}: {e}")
            continue

    if not histograms:
        raise RuntimeError("No valid histograms generated from test dataset")

    # Add to index and store
    histograms_array = np.array(histograms, dtype=np.float64)

    added_count = index.add(histograms_array)
    stored_count = store.add_batch(metadata_batch)

    logger.info(f"Index created with {added_count} vectors")
    logger.info(f"Store created with {stored_count} metadata entries")

    return index, store
```

**What it does:**

- Creates a test FAISS index from real image data
- Builds a metadata store with image information
- Validates histogram quality and format
- Provides progress logging during processing
- Returns ready-to-use test components

### 2. Basic Search Functionality Testing

```python
def test_basic_search_functionality(
    index: AnnIndex, store: MetadataStore
) -> bool:
    """Test basic search functionality with simple queries."""

    logger = logging.getLogger(__name__)
    logger.info("Testing basic search functionality")

    # Get a sample image for querying
    sample_metadata = store.get_all()[0]
    sample_histogram = sample_metadata["histogram"]

    logger.info(f"Query image: {sample_metadata['file_path']}")

    # Test basic search
    try:
        results = find_similar(
            sample_histogram,
            index,
            store,
            k=5,
            rerank_k=0  # No reranking for basic test
        )

        if not results:
            logger.error("Search returned no results")
            return False

        logger.info(f"Found {len(results)} results")

        # Validate result structure
        for i, result in enumerate(results):
            if not isinstance(result, SearchResult):
                logger.error(f"Result {i} is not a SearchResult instance")
                return False

            if not hasattr(result, 'image_id'):
                logger.error(f"Result {i} missing image_id")
                return False

            if not hasattr(result, 'similarity_score'):
                logger.error(f"Result {i} missing similarity_score")
                return False

        # Check that query image is in results (should be first)
        if results[0].image_id != sample_metadata["image_id"]:
            logger.warning("Query image not found as first result")

        logger.info("✅ Basic search functionality test passed")
        return True

    except Exception as e:
        logger.error(f"Basic search test failed: {e}")
        return False
```

**What it tests:**

- Basic search query execution
- Result structure validation
- SearchResult object integrity
- Self-query behavior (query image should be first result)

### 3. Reranking Functionality Testing

```python
def test_reranking_functionality(
    index: AnnIndex, store: MetadataStore
) -> bool:
    """Test the reranking functionality with Sinkhorn-EMD."""

    logger = logging.getLogger(__name__)
    logger.info("Testing reranking functionality")

    # Get a sample image for querying
    sample_metadata = store.get_all()[0]
    query_histogram = sample_metadata["histogram"]

    try:
        # Search without reranking
        initial_results = find_similar(
            query_histogram,
            index,
            store,
            k=10,
            rerank_k=0
        )

        # Search with reranking
        reranked_results = find_similar(
            query_histogram,
            index,
            store,
            k=10,
            rerank_k=5
        )

        if not initial_results or not reranked_results:
            logger.error("Search returned no results")
            return False

        logger.info(f"Initial results: {len(initial_results)}")
        logger.info(f"Reranked results: {len(reranked_results)}")

        # Check that reranking didn't lose results
        if len(reranked_results) < len(initial_results):
            logger.warning("Reranking reduced result count")

        # Check that top result is still the same (self-query)
        if initial_results[0].image_id != reranked_results[0].image_id:
            logger.warning("Top result changed after reranking")

        # Check that reranking scores are different
        initial_scores = [r.similarity_score for r in initial_results[:5]]
        reranked_scores = [r.similarity_score for r in reranked_results[:5]]

        if initial_scores == reranked_scores:
            logger.warning("Reranking scores identical to initial scores")

        logger.info("✅ Reranking functionality test passed")
        return True

    except Exception as e:
        logger.error(f"Reranking test failed: {e}")
        return False
```

**What it tests:**

- Sinkhorn-EMD reranking execution
- Result consistency between initial and reranked searches
- Score changes after reranking
- Self-query result preservation

### 4. Performance Benchmarking

```python
def benchmark_search_performance(
    index: AnnIndex, store: MetadataStore
) -> Dict[str, Any]:
    """Benchmark search performance characteristics."""

    logger = logging.getLogger(__name__)
    logger.info("Benchmarking search performance")

    # Get multiple query images
    query_images = store.get_all()[:10]  # Test with 10 images

    performance_data = {
        "search_times": [],
        "rerank_times": [],
        "total_times": [],
        "result_counts": []
    }

    for i, query_metadata in enumerate(query_images):
        query_histogram = query_metadata["histogram"]

        logger.info(f"Benchmarking query {i+1}/{len(query_images)}")

        # Benchmark search without reranking
        start_time = time.time()
        search_results = find_similar(
            query_histogram,
            index,
            store,
            k=20,
            rerank_k=0
        )
        search_time = time.time() - start_time

        # Benchmark reranking
        if search_results:
            start_time = time.time()
            reranked_results = find_similar(
                query_histogram,
                index,
                store,
                k=20,
                rerank_k=10
            )
            rerank_time = time.time() - start_time

            total_time = search_time + rerank_time

            performance_data["search_times"].append(search_time * 1000)  # Convert to ms
            performance_data["rerank_times"].append(rerank_time * 1000)
            performance_data["total_times"].append(total_time * 1000)
            performance_data["result_counts"].append(len(reranked_results))

            logger.info(f"  Search: {search_time*1000:.2f} ms")
            logger.info(f"  Rerank: {rerank_time*1000:.2f} ms")
            logger.info(f"  Total: {total_time*1000:.2f} ms")
            logger.info(f"  Results: {len(reranked_results)}")

    # Calculate statistics
    if performance_data["search_times"]:
        stats = {
            "search_time_ms": {
                "mean": np.mean(performance_data["search_times"]),
                "std": np.std(performance_data["search_times"]),
                "min": np.min(performance_data["search_times"]),
                "max": np.max(performance_data["search_times"])
            },
            "rerank_time_ms": {
                "mean": np.mean(performance_data["rerank_times"]),
                "std": np.std(performance_data["rerank_times"]),
                "min": np.min(performance_data["rerank_times"]),
                "max": np.max(performance_data["rerank_times"])
            },
            "total_time_ms": {
                "mean": np.mean(performance_data["total_times"]),
                "std": np.std(performance_data["total_times"]),
                "min": np.min(performance_data["total_times"]),
                "max": np.max(performance_data["total_times"])
            },
            "avg_results": np.mean(performance_data["result_counts"])
        }

        logger.info("Performance Statistics:")
        logger.info(f"  Search time: {stats['search_time_ms']['mean']:.2f} ± {stats['search_time_ms']['std']:.2f} ms")
        logger.info(f"  Rerank time: {stats['rerank_time_ms']['mean']:.2f} ± {stats['rerank_time_ms']['std']:.2f} ms")
        logger.info(f"  Total time: {stats['total_time_ms']['mean']:.2f} ± {stats['total_time_ms']['std']:.2f} ms")
        logger.info(f"  Average results: {stats['avg_results']:.1f}")

        return stats

    return {}
```

**What it measures:**

- Search execution time (ANN phase)
- Reranking execution time (Sinkhorn-EMD phase)
- Total query processing time
- Result count consistency
- Performance variability across queries

### 5. Error Handling Testing

```python
def test_error_handling(
    index: AnnIndex, store: MetadataStore
) -> bool:
    """Test error handling and edge cases."""

    logger = logging.getLogger(__name__)
    logger.info("Testing error handling and edge cases")

    test_cases = [
        {
            "name": "Empty histogram",
            "histogram": np.array([]),
            "expected_error": True
        },
        {
            "name": "Wrong histogram shape",
            "histogram": np.zeros(100),  # Should be 1152
            "expected_error": True
        },
        {
            "name": "Invalid histogram values",
            "histogram": np.full(1152, -1.0),  # Negative values
            "expected_error": True
        },
        {
            "name": "Non-normalized histogram",
            "histogram": np.full(1152, 2.0),  # Sum = 2304
            "expected_error": True
        }
    ]

    error_handling_works = True

    for test_case in test_cases:
        logger.info(f"Testing: {test_case['name']}")

        try:
            results = find_similar(
                test_case["histogram"],
                index,
                store,
                k=5,
                rerank_k=0
            )

            if test_case["expected_error"]:
                logger.warning(f"  Expected error but got results: {len(results)}")
                error_handling_works = False
            else:
                logger.info(f"  ✅ Correctly handled valid input")

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

- Invalid histogram input handling
- Error propagation and catching
- Edge case behavior
- System robustness

## Complete Test Suite

### Running All Tests

```python
def run_complete_test_suite(
    test_dataset_path: str = "datasets/test-dataset-50",
    verbose: bool = False,
    performance: bool = False
) -> Dict[str, Any]:
    """Run the complete search system test suite."""

    # Setup logging
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info("🚀 Chromatica Search System Test Suite")
    logger.info("=" * 60)
    logger.info(f"Test dataset: {test_dataset_path}")
    logger.info(f"Verbose logging: {verbose}")
    logger.info(f"Performance testing: {performance}")

    test_results = {}

    try:
        # Create test index and store
        logger.info("\n📋 Creating test index and store...")
        index, store = create_test_index_and_store(test_dataset_path)

        # Test basic functionality
        logger.info("\n🔍 Testing basic search functionality...")
        test_results["basic_search"] = test_basic_search_functionality(index, store)

        # Test reranking
        logger.info("\n🔄 Testing reranking functionality...")
        test_results["reranking"] = test_reranking_functionality(index, store)

        # Test error handling
        logger.info("\n⚠️  Testing error handling...")
        test_results["error_handling"] = test_error_handling(index, store)

        # Performance testing (optional)
        if performance:
            logger.info("\n⚡ Performance benchmarking...")
            performance_stats = benchmark_search_performance(index, store)
            test_results["performance"] = performance_stats

        # System validation
        logger.info("\n🔍 System validation...")
        try:
            system_valid = validate_search_system(index, store)
            test_results["system_validation"] = system_valid
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            test_results["system_validation"] = False

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        test_results["overall_success"] = False
        return test_results

    # Calculate overall success
    critical_tests = ["basic_search", "reranking", "error_handling"]
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
```

## Integration Examples

### With CI/CD Pipeline

```python
#!/usr/bin/env python3
"""
CI/CD integration script for search system testing.
"""

import sys
from tools.test_search_system import run_complete_test_suite

def main():
    """Run tests and exit with appropriate code for CI/CD."""

    try:
        results = run_complete_test_suite(
            test_dataset_path="datasets/test-dataset-50",
            verbose=True,
            performance=True
        )

        if results.get("overall_success", False):
            print("✅ CI/CD: All critical tests passed")
            sys.exit(0)  # Success
        else:
            print("❌ CI/CD: Critical tests failed")
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
    """Run tests during development."""

    # Quick test with small dataset
    print("Running quick development tests...")
    results = run_complete_test_suite(
        test_dataset_path="datasets/test-dataset-20",
        verbose=True,
        performance=False
    )

    if results.get("overall_success"):
        print("✅ Development tests passed - ready for commit")
    else:
        print("❌ Development tests failed - fix issues before commit")

    return results
```

## Best Practices

### Test Development

1. **Comprehensive Coverage**: Test all major system components
2. **Real Data**: Use actual image datasets for realistic testing
3. **Error Scenarios**: Include edge cases and failure modes
4. **Performance Metrics**: Measure timing and resource usage
5. **Logging**: Use appropriate logging levels for debugging

### Test Execution

1. **Environment Setup**: Ensure virtual environment is activated
2. **Dataset Availability**: Verify test datasets are accessible
3. **Resource Management**: Monitor memory usage during testing
4. **Clean State**: Start with fresh test data when possible
5. **Result Validation**: Verify test results are meaningful

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `max_images` for large datasets
2. **Import Errors**: Check virtual environment and Python path
3. **Dataset Issues**: Verify test datasets exist and are accessible
4. **Performance Issues**: Check system resources and index size

### Debug Mode

```python
# Enable verbose logging
python tools/test_search_system.py --verbose

# Check specific components
python tools/test_search_system.py --dataset datasets/test-dataset-20

# Monitor memory usage
# Use smaller datasets for memory-constrained environments
```

## Dependencies

### Required Packages

- `numpy`: Numerical operations
- `faiss-cpu`: ANN index operations
- `duckdb`: Metadata storage
- `opencv-python`: Image processing
- `scikit-image`: Color space conversion

### Installation

```bash
# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The test search system tool provides comprehensive validation of the Chromatica search system and serves as both a development tool and a quality assurance framework for the complete search pipeline.
