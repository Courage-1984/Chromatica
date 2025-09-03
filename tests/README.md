# Chromatica Testing Framework

This directory contains the comprehensive testing infrastructure for the Chromatica color search engine.

## 🧪 Testing Overview

The Chromatica project includes a multi-layered testing approach to ensure code quality, performance, and correctness across all components:

- **Unit Tests**: Individual module and function testing
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Benchmarking and optimization validation
- **Validation Tests**: Algorithm correctness and quality assurance

## 📁 Test Structure

```
tests/
├── README.md                    # This documentation
├── unit/                        # Unit tests for individual modules
│   ├── test_core/              # Core histogram generation tests
│   ├── test_indexing/          # FAISS and DuckDB tests
│   ├── test_utils/             # Configuration and utility tests
│   └── test_api/               # API endpoint tests (when implemented)
├── integration/                 # Integration tests for complete pipelines
├── performance/                 # Performance benchmarking tests
└── validation/                  # Algorithm correctness validation
```

## 🚀 Running Tests

### Prerequisites

1. **Activate Virtual Environment**:

   ```bash
   venv311\Scripts\activate  # Windows
   # or
   source venv311/bin/activate  # Linux/Mac
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=src/chromatica

# Run specific test file
pytest tests/unit/test_core/test_histogram.py

# Run tests matching a pattern
pytest -k "histogram"
```

### Test Categories

#### Unit Tests

```bash
# Test core histogram generation
pytest tests/unit/test_core/

# Test indexing components
pytest tests/unit/test_indexing/

# Test utility functions
pytest tests/unit/test_utils/
```

#### Integration Tests

```bash
# Test complete image processing pipeline
pytest tests/integration/

# Test FAISS and DuckDB integration
pytest tests/integration/test_indexing_pipeline.py
```

#### Performance Tests

```bash
# Run performance benchmarks
pytest tests/performance/

# Test with different dataset sizes
pytest tests/performance/test_scalability.py
```

## 🔧 Test Configuration

### pytest Configuration

The project uses `pytest.ini` for test configuration:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

### Test Data

Tests use the comprehensive test datasets located in `datasets/`:

- **test-dataset-20**: Quick unit testing (20 images)
- **test-dataset-50**: Integration testing (50 images)
- **test-dataset-200**: Performance testing (200 images)
- **test-dataset-5000**: Production-scale testing (5,000 images)

### Environment Variables

```bash
# Set test environment
export CHROMATICA_TEST_ENV=development

# Enable debug logging
export CHROMATICA_LOG_LEVEL=DEBUG

# Set test data directory
export CHROMATICA_TEST_DATA=datasets/test-dataset-50
```

## 📊 Test Coverage

### Current Coverage

- **Core Module**: 95%+ coverage

  - Histogram generation functions
  - Color space conversion utilities
  - Input validation and error handling

- **Indexing Module**: 85%+ coverage

  - FAISS index operations
  - DuckDB storage operations
  - Pipeline integration

- **Utils Module**: 100% coverage
  - Configuration validation
  - Constant definitions

### Coverage Reports

Generate detailed coverage reports:

```bash
# Generate HTML coverage report
pytest --cov=src/chromatica --cov-report=html

# Generate XML coverage report for CI/CD
pytest --cov=src/chromatica --cov-report=xml

# Generate terminal coverage report
pytest --cov=src/chromatica --cov-report=term-missing
```

## 🧪 Test Types

### Unit Tests

#### Histogram Generation Tests

```python
def test_build_histogram_basic():
    """Test basic histogram generation with known input."""
    lab_pixels = np.array([[50.0, 10.0, -20.0]])
    histogram = build_histogram(lab_pixels)

    assert histogram.shape == (1152,)
    assert np.isclose(histogram.sum(), 1.0)
    assert np.all(histogram >= 0)
```

#### Configuration Tests

```python
def test_config_validation():
    """Test configuration constant validation."""
    from chromatica.utils.config import validate_config

    # Should not raise any errors
    validate_config()
```

### Integration Tests

#### Pipeline Tests

```python
def test_complete_image_pipeline():
    """Test end-to-end image processing pipeline."""
    image_path = "datasets/test-dataset-20/a1.png"
    histogram = process_image(image_path)

    assert histogram.shape == (1152,)
    assert np.isclose(histogram.sum(), 1.0)
```

#### FAISS Integration Tests

```python
def test_faiss_indexing():
    """Test FAISS index operations."""
    index = AnnIndex(dimension=1152)
    test_vectors = np.random.rand(10, 1152).astype(np.float32)

    # Test indexing
    n_added = index.add(test_vectors)
    assert n_added == 10

    # Test search
    query = np.random.rand(1, 1152).astype(np.float32)
    distances, indices = index.search(query, k=5)

    assert len(indices[0]) == 5
```

### Performance Tests

#### Scalability Tests

```python
def test_histogram_generation_performance():
    """Test histogram generation performance across dataset sizes."""
    datasets = [20, 50, 200, 5000]

    for size in datasets:
        dataset_path = f"datasets/test-dataset-{size}"
        start_time = time.time()

        # Process dataset
        process_dataset(dataset_path)

        elapsed = time.time() - start_time
        avg_time = elapsed / size

        # Performance assertions
        assert avg_time < 0.5  # Less than 500ms per image
```

## 🐛 Debugging Tests

### Verbose Output

```bash
# Enable detailed test output
pytest -v -s

# Show print statements
pytest -s

# Show local variables on failure
pytest --tb=long
```

### Test Isolation

```bash
# Run tests in isolation
pytest --dist=no

# Run single test function
pytest tests/unit/test_core/test_histogram.py::test_build_histogram_basic
```

### Logging

```bash
# Enable debug logging
pytest --log-cli-level=DEBUG

# Log to file
pytest --log-file=test.log --log-file-level=DEBUG
```

## 📈 Continuous Integration

### GitHub Actions

The project includes GitHub Actions workflows for automated testing:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=src/chromatica --cov-report=xml
```

### Pre-commit Hooks

Install pre-commit hooks for automated code quality checks:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run all hooks
pre-commit run --all-files
```

## 🔍 Test Validation

### Histogram Validation

All histogram tests validate:

- **Shape**: Exactly 1,152 dimensions (8×12×12)
- **Normalization**: L1 normalized (sum = 1.0)
- **Bounds**: All values ≥ 0
- **Quality**: Entropy and sparsity analysis

### Performance Validation

Performance tests ensure:

- **Processing Time**: < 500ms per image
- **Memory Usage**: < 10MB for 100 images
- **Throughput**: > 2 images/second
- **Scalability**: Linear performance scaling

## 📚 Test Documentation

### Writing Tests

Follow these guidelines when writing new tests:

1. **Naming**: Use descriptive test names that explain the scenario
2. **Documentation**: Include docstrings explaining test purpose
3. **Isolation**: Each test should be independent and not affect others
4. **Coverage**: Test both success and failure cases
5. **Performance**: Include performance assertions for critical functions

### Test Examples

```python
class TestHistogramGeneration:
    """Test suite for histogram generation functionality."""

    def test_valid_input(self):
        """Test histogram generation with valid Lab pixel data."""
        # Test implementation

    def test_invalid_input(self):
        """Test histogram generation with invalid input raises appropriate errors."""
        # Test implementation

    def test_edge_cases(self):
        """Test histogram generation with edge case values."""
        # Test implementation
```

## 🚨 Common Issues

### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/Chromatica

# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Missing Test Data

```bash
# Check test dataset availability
ls datasets/

# Download or regenerate test datasets if needed
python scripts/download_test_datasets.py
```

### Performance Test Failures

```bash
# Check system resources
# Ensure sufficient memory and CPU

# Run performance tests in isolation
pytest tests/performance/ -v
```

## 🔮 Future Enhancements

### Planned Testing Features

- **Property-based Testing**: Using Hypothesis for property-based testing
- **Fuzzing**: Automated input fuzzing for robustness testing
- **Load Testing**: High-load performance validation
- **Memory Testing**: Memory leak detection and profiling
- **Cross-platform Testing**: Testing on multiple operating systems

### Test Automation

- **Automated Dataset Generation**: Scripts for creating test datasets
- **Performance Regression Testing**: Automated performance monitoring
- **Integration Test Automation**: End-to-end pipeline validation
- **Documentation Testing**: Automated documentation validation

---

**Last Updated**: December 2024  
**Test Coverage**: 90%+ across all modules  
**Status**: Comprehensive testing infrastructure implemented
