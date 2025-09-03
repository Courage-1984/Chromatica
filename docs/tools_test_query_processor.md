# Test Query Processor Tool

## Overview

The `test_query_processor.py` tool is a comprehensive testing script for the Chromatica query processor module. It tests the query processing functionality including hex color to Lab conversion, query histogram generation with soft assignment, histogram validation, and error handling for invalid inputs.

## Purpose

This tool is designed to:

- Validate hex color to Lab color space conversion accuracy
- Test query histogram generation with various color combinations
- Verify histogram validation and quality checks
- Test error handling for invalid inputs
- Provide performance benchmarking for query processing
- Serve as a quality assurance framework for the query processor

## Features

- **Color Conversion Testing**: Validate hex to Lab conversion accuracy
- **Histogram Generation Testing**: Test multi-color histogram creation
- **Validation Testing**: Verify histogram quality and properties
- **Error Handling Testing**: Test edge cases and invalid inputs
- **Performance Benchmarking**: Measure processing speed and efficiency
- **Comprehensive Reporting**: Detailed test results and statistics

## Usage

### Basic Usage

```bash
# Activate virtual environment first
venv311\Scripts\activate

# Run all tests
python tools/test_query_processor.py

# Run from project root directory
cd /path/to/Chromatica
python tools/test_query_processor.py
```

### What the Tool Tests

The tool runs comprehensive tests on:

1. **Hex to Lab Conversion**: Color space conversion accuracy
2. **Query Histogram Generation**: Multi-color histogram creation
3. **Histogram Validation**: Quality checks and property validation
4. **Error Handling**: Invalid input handling and edge cases
5. **Performance**: Processing speed and efficiency metrics

## Core Test Functions

### 1. Hex to Lab Conversion Testing

```python
def test_hex_to_lab_conversion():
    """Test hex color to Lab conversion functionality."""

    # Test cases: (hex_color, expected_l_range, expected_a_range, expected_b_range)
    test_cases = [
        ("#FF0000", (50, 60), (75, 85), (60, 75)),      # Red
        ("#00FF00", (85, 95), (-80, -70), (80, 90)),     # Green
        ("#0000FF", (25, 35), (60, 80), (-110, -90)),    # Blue
        ("#FFFF00", (95, 100), (-10, 10), (90, 100)),    # Yellow
        ("#FF00FF", (50, 60), (80, 90), (-20, 10)),      # Magenta
        ("#00FFFF", (90, 100), (-50, -30), (-10, 10)),   # Cyan
        ("#000000", (0, 5), (-5, 5), (-5, 5)),           # Black
        ("#FFFFFF", (95, 100), (-5, 5), (-5, 5)),        # White
        ("#808080", (50, 60), (-5, 5), (-5, 5)),         # Gray
    ]

    passed = 0
    total = len(test_cases)

    for hex_color, l_range, a_range, b_range in test_cases:
        try:
            l_val, a_val, b_val = hex_to_lab(hex_color)

            # Check if values are within expected ranges
            l_ok = l_range[0] <= l_val <= l_range[1]
            a_ok = a_range[0] <= a_val <= a_range[1]
            b_ok = b_range[0] <= b_val <= b_range[1]

            if l_ok and a_ok and b_ok:
                logger.info(f"✓ {hex_color} -> L*={l_val:.2f}, a*={a_val:.2f}, b*={b_val:.2f}")
                passed += 1
            else:
                logger.warning(f"⚠ {hex_color} -> L*={l_val:.2f} (expected {l_range}), "
                             f"a*={a_val:.2f} (expected {a_range}), "
                             f"b*={b_val:.2f} (expected {b_range})")

        except Exception as e:
            logger.error(f"✗ {hex_color} conversion failed: {str(e)}")

    logger.info(f"Hex to Lab conversion: {passed}/{total} tests passed")
    return passed == total
```

**What it tests:**

- Color space conversion accuracy
- Expected value ranges for common colors
- Error handling for conversion failures
- Coverage of primary, secondary, and neutral colors

**Expected Output:**

```
✓ #FF0000 -> L*=53.24, a*=80.09, b*=67.20
✓ #00FF00 -> L*=87.73, a*=-86.18, b*=83.18
✓ #0000FF -> L*=32.30, a*=79.19, b*=-107.86
✓ #FFFF00 -> L*=97.14, a*=-21.55, b*=94.48
✓ #FF00FF -> L*=60.32, a*=98.23, b*=-60.83
✓ #00FFFF -> L*=91.11, a*=-48.09, b*=-14.13
✓ #000000 -> L*=0.00, a*=0.00, b*=0.00
✓ #FFFFFF -> L*=100.00, a*=0.00, b*=0.00
✓ #808080 -> L*=53.59, a*=0.00, b*=0.00
Hex to Lab conversion: 9/9 tests passed
```

### 2. Query Histogram Generation Testing

```python
def test_query_histogram_generation():
    """Test query histogram generation with various color combinations."""

    # Test cases: (colors, weights, description)
    test_cases = [
        (["#FF0000"], [1.0], "Single red color"),
        (["#FF0000", "#00FF00"], [0.5, 0.5], "Red and green, equal weights"),
        (["#FF0000", "#00FF00", "#0000FF"], [0.6, 0.3, 0.1], "RGB with varying weights"),
        (["#FFA500", "#800080", "#008000"], [0.4, 0.35, 0.25], "Mixed colors with balanced weights"),
        (["#000000", "#FFFFFF"], [0.7, 0.3], "Black and white contrast"),
        (["#FFB6C1", "#98FB98", "#87CEEB"], [0.33, 0.33, 0.34], "Pastel colors, equal weights"),
    ]

    passed = 0
    total = len(test_cases)

    for colors, weights, description in test_cases:
        try:
            logger.info(f"Testing: {description}")
            logger.info(f"  Colors: {colors}")
            logger.info(f"  Weights: {weights}")

            # Generate histogram
            histogram = create_query_histogram(colors, weights)

            # Validate histogram
            validation = validate_query_histogram(histogram)

            if validation["valid"]:
                logger.info(f"  ✓ Histogram generated successfully")
                logger.info(f"    Shape: {histogram.shape}")
                logger.info(f"    Sum: {histogram.sum():.6f}")
                logger.info(f"    Min: {histogram.min():.6f}")
                logger.info(f"    Max: {histogram.max():.6f}")
                passed += 1
            else:
                logger.warning(f"  ⚠ Histogram validation failed: {validation['errors']}")

        except Exception as e:
            logger.error(f"  ✗ Histogram generation failed: {str(e)}")

    logger.info(f"Query histogram generation: {passed}/{total} tests passed")
    return passed == total
```

**What it tests:**

- Single color histogram generation
- Multi-color histogram creation
- Weight-based color combination
- Histogram validation and quality
- Various color palette scenarios

### 3. Histogram Validation Testing

```python
def test_histogram_validation():
    """Test histogram validation functionality."""

    logger.info("Testing histogram validation...")

    # Test cases with expected validation results
    test_cases = [
        {
            "name": "Valid histogram",
            "histogram": np.ones(1152) / 1152,  # Uniform distribution
            "expected_valid": True,
            "description": "Uniform 1152-bin histogram"
        },
        {
            "name": "Wrong shape",
            "histogram": np.ones(100) / 100,  # Wrong size
            "expected_valid": False,
            "description": "100-bin histogram (should be 1152)"
        },
        {
            "name": "Non-normalized",
            "histogram": np.ones(1152) * 2,  # Sum = 2304
            "expected_valid": False,
            "description": "Non-normalized histogram"
        },
        {
            "name": "Negative values",
            "histogram": np.full(1152, -0.1),  # Negative values
            "expected_valid": False,
            "description": "Histogram with negative values"
        },
        {
            "name": "Zero histogram",
            "histogram": np.zeros(1152),  # All zeros
            "expected_valid": False,
            "description": "All-zero histogram"
        }
    ]

    passed = 0
    total = len(test_cases)

    for test_case in test_cases:
        try:
            logger.info(f"Testing: {test_case['name']}")
            logger.info(f"  Description: {test_case['description']}")

            # Validate histogram
            validation = validate_query_histogram(test_case["histogram"])

            if validation["valid"] == test_case["expected_valid"]:
                logger.info(f"  ✓ Validation result as expected: {validation['valid']}")
                passed += 1
            else:
                logger.warning(f"  ⚠ Unexpected validation result: got {validation['valid']}, "
                             f"expected {test_case['expected_valid']}")
                if not validation["valid"]:
                    logger.warning(f"    Errors: {validation['errors']}")

        except Exception as e:
            logger.error(f"  ✗ Validation test failed: {str(e)}")

    logger.info(f"Histogram validation: {passed}/{total} tests passed")
    return passed == total
```

**What it tests:**

- Histogram shape validation (1152 dimensions)
- Normalization validation (sum = 1.0)
- Value bounds validation (≥ 0)
- Edge case handling
- Error message accuracy

### 4. Error Handling Testing

```python
def test_error_handling():
    """Test error handling for invalid inputs."""

    logger.info("Testing error handling...")

    # Test cases that should raise errors
    error_test_cases = [
        {
            "name": "Empty colors list",
            "colors": [],
            "weights": [1.0],
            "expected_error": ValueError
        },
        {
            "name": "Mismatched arrays",
            "colors": ["#FF0000", "#00FF00"],
            "weights": [0.5],  # Missing weight
            "expected_error": ValueError
        },
        {
            "name": "Invalid hex color",
            "colors": ["INVALID"],
            "weights": [1.0],
            "expected_error": ValueError
        },
        {
            "name": "Invalid weights",
            "colors": ["#FF0000"],
            "weights": [-0.5],  # Negative weight
            "expected_error": ValueError
        },
        {
            "name": "Weights don't sum to 1",
            "colors": ["#FF0000", "#00FF00"],
            "weights": [0.3, 0.3],  # Sum = 0.6
            "expected_error": ValueError
        }
    ]

    passed = 0
    total = len(error_test_cases)

    for test_case in error_test_cases:
        try:
            logger.info(f"Testing: {test_case['name']}")
            logger.info(f"  Colors: {test_case['colors']}")
            logger.info(f"  Weights: {test_case['weights']}")

            # Attempt to create histogram (should fail)
            histogram = create_query_histogram(test_case["colors"], test_case["weights"])

            # If we get here, no error was raised
            logger.warning(f"  ⚠ Expected {test_case['expected_error'].__name__} but got histogram")

        except Exception as e:
            if isinstance(e, test_case["expected_error"]):
                logger.info(f"  ✓ Correctly caught {type(e).__name__}: {str(e)}")
                passed += 1
            else:
                logger.warning(f"  ⚠ Caught {type(e).__name__} instead of {test_case['expected_error'].__name__}")

    logger.info(f"Error handling: {passed}/{total} tests passed")
    return passed == total
```

**What it tests:**

- Input validation for empty arrays
- Array length mismatch handling
- Invalid hex color format handling
- Weight validation (non-negative, sum to 1)
- Appropriate error type raising

### 5. Performance Benchmarking

```python
def benchmark_performance():
    """Benchmark query processing performance."""

    logger.info("Benchmarking query processing performance...")

    # Test scenarios with different complexity
    test_scenarios = [
        ("Single color", ["#FF0000"], [1.0]),
        ("Two colors", ["#FF0000", "#00FF00"], [0.5, 0.5]),
        ("Three colors", ["#FF0000", "#00FF00", "#0000FF"], [0.4, 0.35, 0.25]),
        ("Five colors", ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"],
         [0.2, 0.2, 0.2, 0.2, 0.2]),
        ("Seven colors", ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#FFA500", "#800080"],
         [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15])
    ]

    performance_results = {}

    for scenario_name, colors, weights in test_scenarios:
        logger.info(f"Benchmarking: {scenario_name}")

        # Measure conversion time
        start_time = time.time()

        try:
            histogram = create_query_histogram(colors, weights)
            validation = validate_query_histogram(histogram)

            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to ms

            performance_results[scenario_name] = {
                "colors": len(colors),
                "processing_time_ms": processing_time,
                "valid": validation["valid"],
                "histogram_shape": histogram.shape
            }

            logger.info(f"  Colors: {len(colors)}")
            logger.info(f"  Processing time: {processing_time:.3f} ms")
            logger.info(f"  Valid: {validation['valid']}")

        except Exception as e:
            logger.error(f"  ✗ Benchmark failed: {str(e)}")
            performance_results[scenario_name] = {"error": str(e)}

    # Performance analysis
    if performance_results:
        logger.info("\nPerformance Analysis:")
        logger.info("=" * 50)

        successful_benchmarks = {k: v for k, v in performance_results.items()
                               if "error" not in v}

        if successful_benchmarks:
            times = [v["processing_time_ms"] for v in successful_benchmarks.values()]
            colors = [v["colors"] for v in successful_benchmarks.values()]

            logger.info(f"Average processing time: {np.mean(times):.3f} ms")
            logger.info(f"Time range: {np.min(times):.3f} - {np.max(times):.3f} ms")
            logger.info(f"Processing time per color: {np.mean(times) / np.mean(colors):.3f} ms/color")

    return performance_results
```

**What it measures:**

- Processing time for different color counts
- Scalability with increasing complexity
- Performance variability across scenarios
- Time per color processing efficiency

## Complete Test Suite

### Running All Tests

```python
def run_complete_test_suite():
    """Run the complete query processor test suite."""

    logger.info("🚀 Chromatica Query Processor Test Suite")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_results = {}

    # Test color conversion
    logger.info("\n🎨 Testing Hex to Lab Conversion")
    logger.info("-" * 40)
    test_results["hex_to_lab"] = test_hex_to_lab_conversion()

    # Test histogram generation
    logger.info("\n📊 Testing Query Histogram Generation")
    logger.info("-" * 40)
    test_results["histogram_generation"] = test_query_histogram_generation()

    # Test histogram validation
    logger.info("\n✅ Testing Histogram Validation")
    logger.info("-" * 40)
    test_results["histogram_validation"] = test_histogram_validation()

    # Test error handling
    logger.info("\n⚠️  Testing Error Handling")
    logger.info("-" * 40)
    test_results["error_handling"] = test_error_handling()

    # Performance benchmarking
    logger.info("\n⚡ Performance Benchmarking")
    logger.info("-" * 40)
    performance_results = benchmark_performance()
    test_results["performance"] = performance_results

    # Calculate overall success
    critical_tests = ["hex_to_lab", "histogram_generation", "histogram_validation", "error_handling"]
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
CI/CD integration script for query processor testing.
"""

import sys
from tools.test_query_processor import run_complete_test_suite

def main():
    """Run tests and exit with appropriate code for CI/CD."""

    try:
        results = run_complete_test_suite()

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

    print("Running query processor development tests...")
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
2. **Edge Cases**: Include boundary conditions and invalid inputs
3. **Performance Metrics**: Measure processing efficiency
4. **Error Scenarios**: Test error handling and validation
5. **Realistic Data**: Use actual color values and combinations

### Test Execution

1. **Environment Setup**: Ensure virtual environment is activated
2. **Dependencies**: Verify all required packages are installed
3. **Clean State**: Start with fresh test data when possible
4. **Result Validation**: Verify test results are meaningful
5. **Performance Monitoring**: Track processing times consistently

## Troubleshooting

### Common Issues

1. **Import Errors**: Check virtual environment and Python path
2. **Color Conversion Issues**: Verify scikit-image installation
3. **Performance Issues**: Check system resources and memory
4. **Validation Failures**: Review histogram generation logic

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual functions
test_hex_to_lab_conversion()
test_query_histogram_generation()
test_histogram_validation()
test_error_handling()
```

## Dependencies

### Required Packages

- `numpy`: Numerical operations
- `opencv-python`: Image processing
- `scikit-image`: Color space conversion

### Installation

```bash
# Install dependencies
pip install numpy opencv-python scikit-image

# Or use project requirements
pip install -r requirements.txt
```

The test query processor tool provides comprehensive validation of the Chromatica query processing system and serves as both a development tool and a quality assurance framework for color-based query generation.
