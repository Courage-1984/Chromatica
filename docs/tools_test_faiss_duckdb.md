# Test FAISS-DuckDB Tool

## Overview

The `test_faiss_duckdb.py` tool is a comprehensive testing script for the FAISS and DuckDB wrapper classes in Chromatica. It tests the `AnnIndex` and `MetadataStore` classes to ensure they work correctly with the existing histogram generation pipeline by creating sample histograms, indexing them in FAISS, storing metadata in DuckDB, and performing test searches.

## Purpose

This tool is designed to:

- Test FAISS index functionality and configuration
- Validate DuckDB metadata storage operations
- Test integration between histogram generation and indexing
- Verify search functionality with indexed data
- Provide comprehensive testing of the indexing pipeline
- Serve as a quality assurance framework for the storage system

## Features

- **FAISS Testing**: Comprehensive testing of ANN index functionality
- **DuckDB Testing**: Validation of metadata storage operations
- **Integration Testing**: End-to-end pipeline validation
- **Search Testing**: Query execution and result validation
- **Performance Testing**: Indexing and search performance measurement
- **Error Handling**: Edge case and failure mode testing

## Usage

### Basic Usage

```bash
# Activate virtual environment first
venv311\Scripts\activate

# Run all tests
python tools/test_faiss_duckdb.py

# Run from project root directory
cd /path/to/Chromatica
python tools/test_faiss_duckdb.py
```

### What the Tool Tests

The tool runs comprehensive tests on:

1. **FAISS Index**: Index creation, vector addition, and search functionality
2. **DuckDB Storage**: Metadata storage, retrieval, and management
3. **Integration**: Complete pipeline from histogram to indexed search
4. **Performance**: Indexing speed and search efficiency
5. **Error Handling**: Invalid input handling and edge cases

## Prerequisites

### Environment Setup

```bash
# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify Chromatica modules
python -c "import chromatica.indexing.store; print('Modules available')"
```

### Required Components

- All Chromatica modules installed
- Virtual environment activated
- FAISS and DuckDB dependencies available
- Sufficient memory for index operations

## Core Test Functions

### 1. Sample Histogram Generation

```python
def create_sample_histograms(n_samples: int = 10) -> np.ndarray:
    """Create sample histograms for testing."""

    logger.info(f"Creating {n_samples} sample histograms...")

    # Create random histograms with different characteristics
    histograms = []

    for i in range(n_samples):
        # Create a random histogram with some structure
        if i % 3 == 0:
            # Bright images (high L* values)
            hist = np.random.exponential(0.1, TOTAL_BINS)
            hist[: TOTAL_BINS // 2] *= 0.1  # Suppress dark regions
        elif i % 3 == 1:
            # Colorful images (high a* and b* values)
            hist = np.random.exponential(0.1, TOTAL_BINS)
            hist[TOTAL_BINS // 2 :] *= 2.0  # Enhance color regions
        else:
            # Balanced images
            hist = np.random.exponential(0.1, TOTAL_BINS)

        # Normalize to sum to 1.0
        hist = hist / hist.sum()
        histograms.append(hist)

    histograms_array = np.array(histograms, dtype=np.float32)
    logger.info(f"Created histograms with shape: {histograms_array.shape}")

    return histograms_array
```

**What it does:**

- Creates controlled test histograms with different characteristics
- Generates bright, colorful, and balanced image representations
- Ensures proper histogram normalization
- Provides reproducible test data for consistent testing

### 2. FAISS Index Testing

```python
def test_faiss_index(histograms: np.ndarray) -> AnnIndex:
    """Test the FAISS index functionality."""

    logger.info("Testing FAISS index functionality...")

    # Create and configure the index
    ann_index = AnnIndex(dimension=TOTAL_BINS)

    # Add histograms to the index
    added_count = ann_index.add(histograms)
    logger.info(f"Added {added_count} histograms to FAISS index")

    # Verify the index has the expected number of vectors
    total_vectors = ann_index.get_total_vectors()
    assert total_vectors == len(histograms), f"Expected {len(histograms)} vectors, got {total_vectors}"

    # Test index properties
    dimension = ann_index.get_dimension()
    assert dimension == TOTAL_BINS, f"Expected dimension {TOTAL_BINS}, got {dimension}"

    # Test search functionality
    query_vector = histograms[0]  # Use first histogram as query
    k = min(5, len(histograms))

    distances, indices = ann_index.search(query_vector, k)

    logger.info(f"Search returned {len(indices)} results")
    logger.info(f"Top result distance: {distances[0]:.6f}")
    logger.info(f"Top result index: {indices[0]}")

    # Verify search results
    assert len(distances) == k, f"Expected {k} distances, got {len(distances)}"
    assert len(indices) == k, f"Expected {k} indices, got {len(indices)}"

    # Verify that query vector is in results (self-query)
    assert indices[0] == 0, "Query vector should be first result"
    assert distances[0] == 0.0, "Query vector distance should be 0.0"

    logger.info("✅ FAISS index functionality test passed")
    return ann_index
```

**What it tests:**

- Index creation and configuration
- Vector addition and indexing
- Index properties and dimensions
- Search functionality and accuracy
- Self-query behavior validation

### 3. DuckDB Storage Testing

```python
def test_duckdb_storage(histograms: np.ndarray, image_ids: list) -> MetadataStore:
    """Test the DuckDB metadata storage functionality."""

    logger.info("Testing DuckDB metadata storage...")

    # Create metadata store
    store = MetadataStore(":memory:")  # Use in-memory database for testing

    # Prepare metadata for storage
    metadata_batch = []
    for i, (hist, image_id) in enumerate(zip(histograms, image_ids)):
        metadata = {
            "image_id": image_id,
            "file_path": f"/test/images/{image_id}.jpg",
            "histogram": hist,
            "file_size": 1024 * (i + 1),  # Simulate file sizes
            "width": 800,
            "height": 600,
            "format": "JPEG"
        }
        metadata_batch.append(metadata)

    # Add metadata to store
    stored_count = store.add_batch(metadata_batch)
    logger.info(f"Stored {stored_count} metadata entries")

    # Verify storage
    total_count = store.count()
    assert total_count == len(metadata_batch), f"Expected {len(metadata_batch)} entries, got {total_count}"

    # Test retrieval
    for i, image_id in enumerate(image_ids):
        retrieved = store.get(image_id)
        assert retrieved is not None, f"Could not retrieve metadata for {image_id}"
        assert retrieved["image_id"] == image_id, f"Image ID mismatch for {image_id}"
        assert retrieved["file_path"] == f"/test/images/{image_id}.jpg"
        assert np.array_equal(retrieved["histogram"], histograms[i])

    # Test batch retrieval
    all_metadata = store.get_all()
    assert len(all_metadata) == len(metadata_batch), f"Expected {len(metadata_batch)} entries, got {len(all_metadata)}"

    # Test search functionality
    search_results = store.search("image_id", "test_image_1")
    assert len(search_results) > 0, "Search should return results"

    logger.info("✅ DuckDB storage functionality test passed")
    return store
```

**What it tests:**

- Metadata store creation and configuration
- Batch metadata addition and storage
- Data retrieval and validation
- Search functionality
- Data integrity and consistency

### 4. Integration Testing

```python
def test_integration(histograms: np.ndarray, image_ids: list):
    """Test the complete integration between FAISS and DuckDB."""

    logger.info("Testing FAISS-DuckDB integration...")

    # Create both components
    ann_index = AnnIndex(dimension=TOTAL_BINS)
    store = MetadataStore(":memory:")

    # Add data to both systems
    added_count = ann_index.add(histograms)
    stored_count = store.add_batch([
        {
            "image_id": image_id,
            "file_path": f"/test/images/{image_id}.jpg",
            "histogram": hist,
            "file_size": 1024,
            "width": 800,
            "height": 600,
            "format": "JPEG"
        }
        for image_id, hist in zip(image_ids, histograms)
    ])

    logger.info(f"Indexed {added_count} vectors, stored {stored_count} metadata entries")

    # Verify synchronization
    assert added_count == stored_count, "Index and store should have same count"

    # Test complete search workflow
    query_vector = histograms[0]  # Use first histogram as query
    k = min(5, len(histograms))

    # Search in FAISS
    distances, indices = ann_index.search(query_vector, k)

    # Retrieve metadata for results
    results = []
    for i, (distance, index) in enumerate(zip(distances, indices)):
        if index < len(image_ids):
            image_id = image_ids[index]
            metadata = store.get(image_id)
            if metadata:
                results.append({
                    "rank": i + 1,
                    "image_id": image_id,
                    "distance": distance,
                    "metadata": metadata
                })

    logger.info(f"Integration search returned {len(results)} results")

    # Verify results
    assert len(results) > 0, "Integration search should return results"
    assert results[0]["image_id"] == image_ids[0], "First result should be query image"
    assert results[0]["distance"] == 0.0, "Query image distance should be 0.0"

    # Display results
    for result in results[:3]:
        logger.info(f"Rank {result['rank']}: {result['image_id']} (distance: {result['distance']:.6f})")

    logger.info("✅ FAISS-DuckDB integration test passed")
    return ann_index, store
```

**What it tests:**

- Complete workflow integration
- Data synchronization between systems
- End-to-end search functionality
- Result consistency and accuracy
- Metadata retrieval for search results

### 5. Performance Testing

```python
def test_performance(histograms: np.ndarray, image_ids: list):
    """Test performance characteristics of the indexing system."""

    logger.info("Testing performance characteristics...")

    # Create components
    ann_index = AnnIndex(dimension=TOTAL_BINS)
    store = MetadataStore(":memory:")

    # Measure indexing performance
    start_time = time.time()
    added_count = ann_index.add(histograms)
    index_time = time.time() - start_time

    start_time = time.time()
    stored_count = store.add_batch([
        {
            "image_id": image_id,
            "file_path": f"/test/images/{image_id}.jpg",
            "histogram": hist,
            "file_size": 1024,
            "width": 800,
            "height": 600,
            "format": "JPEG"
        }
        for image_id, hist in zip(image_ids, histograms)
    ])
    store_time = time.time() - start_time

    # Measure search performance
    query_vector = histograms[0]
    k = min(10, len(histograms))

    start_time = time.time()
    distances, indices = ann_index.search(query_vector, k)
    search_time = time.time() - start_time

    # Performance metrics
    logger.info("Performance Metrics:")
    logger.info(f"  Indexing time: {index_time*1000:.2f} ms")
    logger.info(f"  Storage time: {store_time*1000:.2f} ms")
    logger.info(f"  Search time: {search_time*1000:.2f} ms")
    logger.info(f"  Vectors per second (indexing): {len(histograms)/index_time:.1f}")
    logger.info(f"  Queries per second: {1/search_time:.1f}")

    # Performance assertions
    assert index_time < 1.0, f"Indexing too slow: {index_time:.2f}s"
    assert store_time < 1.0, f"Storage too slow: {store_time:.2f}s"
    assert search_time < 0.1, f"Search too slow: {search_time:.3f}s"

    logger.info("✅ Performance test passed")
    return {
        "index_time": index_time,
        "store_time": store_time,
        "search_time": search_time,
        "vectors_per_second": len(histograms)/index_time,
        "queries_per_second": 1/search_time
    }
```

**What it measures:**

- Indexing speed and efficiency
- Storage operation performance
- Search query response time
- System throughput metrics
- Performance benchmarks and thresholds

## Complete Test Suite

### Running All Tests

```python
def run_complete_test_suite():
    """Run the complete FAISS-DuckDB test suite."""

    logger.info("🚀 Chromatica FAISS-DuckDB Test Suite")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_results = {}

    try:
        # Create test data
        logger.info("\n📊 Creating Test Data")
        logger.info("-" * 40)

        n_samples = 20
        histograms = create_sample_histograms(n_samples)
        image_ids = [f"test_image_{i+1}" for i in range(n_samples)]

        # Test FAISS index
        logger.info("\n🔍 Testing FAISS Index")
        logger.info("-" * 40)

        ann_index = test_faiss_index(histograms)
        test_results["faiss_index"] = True

        # Test DuckDB storage
        logger.info("\n💾 Testing DuckDB Storage")
        logger.info("-" * 40)

        store = test_duckdb_storage(histograms, image_ids)
        test_results["duckdb_storage"] = True

        # Test integration
        logger.info("\n🔗 Testing Integration")
        logger.info("-" * 40)

        integration_results = test_integration(histograms, image_ids)
        test_results["integration"] = True

        # Performance testing
        logger.info("\n⚡ Performance Testing")
        logger.info("-" * 40)

        performance_results = test_performance(histograms, image_ids)
        test_results["performance"] = performance_results

        # Overall success assessment
        test_results["overall_success"] = all([
            test_results.get("faiss_index", False),
            test_results.get("duckdb_storage", False),
            test_results.get("integration", False)
        ])

        # Summary
        logger.info("\n📊 Test Summary")
        logger.info("=" * 60)

        if test_results["overall_success"]:
            logger.info("🎉 All FAISS-DuckDB tests passed!")
        else:
            logger.info("⚠️  Some tests failed")

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
CI/CD integration script for FAISS-DuckDB testing.
"""

import sys
from tools.test_faiss_duckdb import run_complete_test_suite

def main():
    """Run tests and exit with appropriate code for CI/CD."""

    try:
        results = run_complete_test_suite()

        if results.get("overall_success", False):
            print("✅ CI/CD: All FAISS-DuckDB tests passed")
            sys.exit(0)  # Success
        else:
            print("❌ CI/CD: FAISS-DuckDB tests failed")
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
    """Run FAISS-DuckDB tests during development."""

    print("Running FAISS-DuckDB development tests...")
    results = run_complete_test_suite()

    if results.get("overall_success"):
        print("✅ Development tests passed - ready for commit")
    else:
        print("❌ Development tests failed - fix issues before commit")

    return results
```

## Best Practices

### Test Development

1. **Comprehensive Coverage**: Test all major functionality areas
2. **Real Data**: Use realistic histogram data for testing
3. **Performance Metrics**: Measure timing and efficiency consistently
4. **Error Scenarios**: Include edge cases and failure modes
5. **Integration Testing**: Verify component interaction

### Test Execution

1. **Environment Setup**: Ensure virtual environment is activated
2. **Dependencies**: Verify all required packages are installed
3. **Clean State**: Start with fresh test data when possible
4. **Result Validation**: Verify test results are meaningful
5. **Performance Monitoring**: Track system performance consistently

## Troubleshooting

### Common Issues

1. **Import Errors**: Check virtual environment and Python path
2. **Memory Issues**: Reduce test data size for memory-constrained environments
3. **Performance Issues**: Check system resources and configuration
4. **Integration Issues**: Verify component compatibility

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
histograms = create_sample_histograms(5)
ann_index = test_faiss_index(histograms)
store = test_duckdb_storage(histograms, ["test1", "test2", "test3", "test4", "test5"])
```

## Dependencies

### Required Packages

- `numpy`: Numerical operations
- `faiss-cpu`: FAISS ANN index operations
- `duckdb`: DuckDB database operations
- `chromatica`: Chromatica core modules

### Installation

```bash
# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The test FAISS-DuckDB tool provides comprehensive validation of the Chromatica indexing and storage system and serves as both a development tool and a quality assurance framework for the FAISS-DuckDB integration.
