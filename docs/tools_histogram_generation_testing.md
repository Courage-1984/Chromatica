# Histogram Generation Testing Tool

## Overview

The `test_histogram_generation.py` tool is a comprehensive testing and validation utility for the Chromatica color search engine's histogram generation module. This tool provides extensive testing capabilities for both single images and batch processing, ensuring the quality and performance of the color histogram generation pipeline.

## Features

- **Single Image Testing**: Process individual images with detailed analysis
- **Batch Directory Processing**: Handle multiple images efficiently
- **Automatic Validation**: Comprehensive histogram quality checks
- **Performance Benchmarking**: Timing and memory usage analysis
- **Multiple Output Formats**: JSON, CSV, and visualization options
- **Comprehensive Reporting**: 6 different report types for analysis

## Command Line Usage

### Basic Commands

```bash
# Test a single image
python tools/test_histogram_generation.py --image path/to/image.jpg

# Test a directory of images
python tools/test_histogram_generation.py --directory path/to/images/

# Test with specific options
python tools/test_histogram_generation.py --image path/to/image.jpg --output-format json --visualize
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--image` | Path to single image file | None |
| `--directory` | Path to directory of images | None |
| `--output-format` | Output format: json, csv, both | json |
| `--visualize` | Generate visualization plots | False |
| `--max-images` | Maximum images to process (batch mode) | 1000 |
| `--output-dir` | Custom output directory | auto-generated |

## Programmatic Usage

### Basic Histogram Testing

```python
from tools.test_histogram_generation import HistogramTester

# Initialize tester
tester = HistogramTester(output_format="json", visualize=True)

# Test single image
result = tester.test_single_image("path/to/image.jpg")

# Test directory
results = tester.test_directory("path/to/images/", max_images=100)
```

### Advanced Configuration

```python
# Custom output format and visualization
tester = HistogramTester(
    output_format="both",  # Generate both JSON and CSV
    visualize=True         # Create visualization plots
)

# Process with custom settings
result = tester.test_single_image(
    "image.jpg",
    save_histogram=True,   # Save histogram data
    validate=True,         # Run validation checks
    benchmark=True         # Performance benchmarking
)
```

## Output and Reports

### Generated Files

The tool creates a comprehensive output structure:

```
output_directory/
├── histograms/           # Raw histogram data (.npy files)
├── visualizations/       # Generated plots and charts
├── reports/             # Analysis reports
│   ├── summary.json     # Overall results summary
│   ├── details.json     # Detailed per-image results
│   ├── validation.csv   # Validation metrics
│   ├── performance.csv  # Performance metrics
│   ├── quality.csv      # Quality metrics
│   └── errors.csv       # Error log
└── logs/                # Processing logs
```

### Report Types

1. **Summary Report**: Overall statistics and success rates
2. **Detailed Results**: Per-image analysis and metadata
3. **Validation Report**: Histogram quality metrics
4. **Performance Report**: Timing and efficiency data
5. **Quality Report**: Entropy, sparsity, and distribution analysis
6. **Error Report**: Failed processing attempts and reasons

## Validation Metrics

### Histogram Quality Checks

- **Shape Validation**: Ensures 1152 dimensions (8×12×12 L*a*b* bins)
- **Normalization**: Verifies sum equals 1.0 (L1 normalization)
- **Bounds Check**: Confirms all values ≥ 0
- **Entropy Analysis**: Measures information content
- **Sparsity Check**: Identifies overly sparse histograms

### Performance Metrics

- **Processing Time**: Mean, standard deviation, min/max
- **Memory Usage**: Estimated memory consumption
- **Throughput**: Pixels processed per second
- **Method Comparison**: Full vs. fast histogram generation

## Visualization Features

### Generated Plots

- **Histogram Distribution**: 3D visualization of color distribution
- **Channel Analysis**: Separate L*, a*, b* channel distributions
- **Performance Charts**: Timing and memory usage graphs
- **Quality Metrics**: Entropy and sparsity distributions

### Visualization Options

```python
# Enable specific visualizations
tester = HistogramTester(visualize=True)

# Custom plot settings
tester.plot_histogram_3d(histogram, title="Custom Title")
tester.plot_channel_distributions(histogram, save_path="custom_plot.png")
```

## Error Handling

### Common Issues

- **Image Loading Failures**: Unsupported formats or corrupted files
- **Memory Errors**: Large images exceeding available memory
- **Validation Failures**: Histograms failing quality checks
- **Performance Issues**: Slow processing times

### Error Recovery

```python
# Handle errors gracefully
try:
    result = tester.test_single_image("image.jpg")
    if not result.get("success", True):
        logger.warning(f"Processing failed: {result.get('error')}")
        # Continue with next image or handle error
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Implement fallback behavior
```

## Integration Examples

### With Existing Pipeline

```python
from chromatica.core.histogram import build_histogram
from tools.test_histogram_generation import HistogramTester

# Use testing tool to validate pipeline output
def validate_pipeline_histogram(histogram, image_path):
    tester = HistogramTester()
    validation_result = tester.validate_histogram(histogram)
    
    if validation_result["valid"]:
        print(f"✅ {image_path}: Valid histogram")
        return True
    else:
        print(f"❌ {image_path}: {validation_result['errors']}")
        return False
```

### Batch Processing Integration

```python
# Integrate with indexing pipeline
def process_dataset_with_validation(dataset_path):
    tester = HistogramTester(output_format="both")
    
    # Process and validate
    results = tester.test_directory(dataset_path, max_images=1000)
    
    # Filter valid results for indexing
    valid_histograms = [
        result for result in results 
        if result.get("success") and result["validation"]["valid"]
    ]
    
    return valid_histograms
```

## Performance Considerations

### Optimization Tips

- **Batch Processing**: Process multiple images together for efficiency
- **Memory Management**: Use `max_images` to limit memory usage
- **Output Format**: Choose JSON for detailed analysis, CSV for bulk processing
- **Visualization**: Disable for production use to improve performance

### Scaling Guidelines

- **Small Datasets** (< 100 images): Use default settings
- **Medium Datasets** (100-1000 images): Enable batch processing
- **Large Datasets** (> 1000 images): Use `max_images` limit and disable visualization

## Troubleshooting

### Common Problems

1. **Import Errors**: Ensure `src/` directory is in Python path
2. **Memory Issues**: Reduce `max_images` or image dimensions
3. **Visualization Failures**: Check matplotlib backend and dependencies
4. **Performance Issues**: Verify image preprocessing and validation settings

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
tester = HistogramTester()
result = tester.test_single_image("image.jpg", debug=True)
```

## Dependencies

### Required Packages

- `opencv-python`: Image loading and processing
- `scikit-image`: Color space conversion
- `numpy`: Numerical operations
- `matplotlib`: Visualization generation
- `seaborn`: Enhanced plotting

### Installation

```bash
# Install dependencies
pip install opencv-python scikit-image numpy matplotlib seaborn

# Or use project requirements
pip install -r requirements.txt
```

## Examples

### Complete Workflow Example

```python
#!/usr/bin/env python3
"""
Complete histogram testing workflow example.
"""

from tools.test_histogram_generation import HistogramTester
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Initialize tester
    tester = HistogramTester(
        output_format="both",
        visualize=True
    )
    
    # Test single image
    print("Testing single image...")
    single_result = tester.test_single_image("datasets/test-dataset-50/test.jpg")
    
    if single_result.get("success"):
        print(f"✅ Single image test passed")
        print(f"   Output: {single_result['output_directory']}")
    
    # Test directory
    print("\nTesting directory...")
    batch_results = tester.test_directory(
        "datasets/test-dataset-50/",
        max_images=50
    )
    
    # Analyze results
    success_count = sum(1 for r in batch_results if r.get("success"))
    print(f"✅ Batch processing complete: {success_count}/{len(batch_results)} successful")

if __name__ == "__main__":
    main()
```

This tool is essential for validating the histogram generation pipeline and ensuring the quality of color representations used in the Chromatica search engine.
