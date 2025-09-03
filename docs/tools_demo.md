# Demo Tool

## Overview

The `demo.py` tool is a demonstration script that showcases the capabilities of the Histogram Generation Testing Tool without requiring command-line arguments. It provides interactive examples and serves as a learning resource for understanding how to use the testing framework programmatically.

## Purpose

This tool is designed to:
- Demonstrate the HistogramTester class functionality
- Show practical usage examples
- Provide a quick way to test the system
- Serve as a reference for developers

## Usage

### Running the Demo

```bash
# From project root directory
python tools/demo.py
```

### What the Demo Shows

The demo script runs through several scenarios:

1. **Single Image Testing**: Demonstrates processing a single image
2. **Batch Processing**: Shows directory processing capabilities
3. **Validation Metrics**: Explains histogram quality checks
4. **Output Formats**: Describes available output options

## Demo Functions

### Single Image Demo

```python
def demo_single_image():
    """Demonstrate single image testing."""
    
    # Initialize tester
    tester = HistogramTester(output_format="json", visualize=True)
    
    # Test a specific image
    image_path = "datasets/test-dataset-50/test.jpg"
    result = tester.test_single_image(image_path)
    
    # Display results
    if result.get("success", True):
        print(f"✅ Success! Histogram shape: {result['histogram']['shape']}")
        print(f"   Entropy: {result['validation']['metrics']['entropy']:.4f}")
        print(f"   Processing time: {result['performance']['mean_time_ms']:.2f} ms")
```

**What it demonstrates:**
- Basic HistogramTester initialization
- Single image processing workflow
- Result interpretation and display
- Error handling for missing files

### Batch Processing Demo

```python
def demo_batch_processing():
    """Demonstrate batch directory processing."""
    
    # Initialize tester
    tester = HistogramTester(output_format="both", visualize=False)
    
    # Show directory processing concept
    directory_path = "datasets/test-dataset-50"
    print(f"Testing directory: {directory_path}")
    print("   (Demo mode: would process all images in directory)")
    print("   Use: python tools/test_histogram_generation.py --directory datasets/test-dataset-50/")
```

**What it demonstrates:**
- Batch processing setup
- Different output format configurations
- Command-line usage instructions
- Directory path handling

### Validation Metrics Demo

```python
def demo_validation_metrics():
    """Demonstrate histogram validation metrics."""
    
    print("The tool validates histograms for:")
    print("  • Shape: Correct 1152 dimensions (8×12×12)")
    print("  • Normalization: Sum equals 1.0")
    print("  • Bounds: All values ≥ 0")
    print("  • Quality: Entropy, sparsity, distribution")
    
    print("\nPerformance metrics include:")
    print("  • Processing time (mean, std, min/max)")
    print("  • Memory usage estimation")
    print("  • Pixels processed per second")
    print("  • Comparison between full and fast methods")
```

**What it demonstrates:**
- Validation criteria explanation
- Performance metrics overview
- Quality assessment factors
- Method comparison benefits

### Output Format Demo

```python
def demo_output_formats():
    """Demonstrate output format options."""
    
    print("Available output formats:")
    print("  • JSON: Detailed results with all metadata")
    print("  • CSV: Flattened format for analysis")
    print("  • Both: Generate both formats")
    
    print("\nGenerated files:")
    print("  • Histogram data (.npy files) → histograms/ folder")
    print("  • Visualization plots → visualizations/ folder")
    print("  • Analysis reports → reports/ folder")
    print("  • Processing logs → logs/ folder")
```

**What it demonstrates:**
- Output format options
- File structure organization
- Data storage locations
- Report generation capabilities

## Integration Examples

### Using Demo as a Template

The demo script can serve as a starting point for custom implementations:

```python
#!/usr/bin/env python3
"""
Custom histogram testing implementation based on demo.py
"""

from tools.test_histogram_generation import HistogramTester
import os

def custom_histogram_analysis():
    """Custom histogram analysis workflow."""
    
    # Initialize with custom settings
    tester = HistogramTester(
        output_format="json",
        visualize=True
    )
    
    # Custom image processing logic
    custom_images = [
        "path/to/image1.jpg",
        "path/to/image2.png",
        "path/to/image3.jpg"
    ]
    
    results = []
    for image_path in custom_images:
        if os.path.exists(image_path):
            try:
                result = tester.test_single_image(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    return results

if __name__ == "__main__":
    results = custom_histogram_analysis()
    print(f"Processed {len(results)} images successfully")
```

### Extending Demo Functionality

```python
def extended_demo():
    """Extended demo with additional features."""
    
    # Initialize tester
    tester = HistogramTester(output_format="both", visualize=True)
    
    # Test multiple datasets
    datasets = [
        "datasets/test-dataset-20",
        "datasets/test-dataset-50",
        "datasets/test-dataset-200"
    ]
    
    for dataset in datasets:
        if os.path.exists(dataset):
            print(f"\n📁 Testing dataset: {dataset}")
            
            # Process with different settings
            results = tester.test_directory(
                dataset,
                max_images=50,
                save_histogram=True
            )
            
            # Analyze results
            success_count = sum(1 for r in results if r.get("success"))
            print(f"   ✅ {success_count}/{len(results)} images processed successfully")
```

## Error Handling in Demo

### Graceful Degradation

The demo includes error handling for common scenarios:

```python
# Check if test image exists
image_path = "datasets/test-dataset-50/test.jpg"
if os.path.exists(image_path):
    print(f"Testing image: {image_path}")
    result = tester.test_single_image(image_path)
    # Process result...
else:
    print(f"⚠️  Image not found: {image_path}")
    print("   Make sure you have the test dataset available")
```

### Missing Dataset Handling

```python
# Check if test directory exists
directory_path = "datasets/test-dataset-50"
if os.path.exists(directory_path):
    print(f"Testing directory: {directory_path}")
    # Process directory...
else:
    print(f"⚠️  Directory not found: {directory_path}")
    print("   Please ensure test datasets are available")
```

## Demo Output

### Expected Console Output

When running the demo, you'll see output similar to:

```
🔍 Single Image Testing Demo
==================================================
Testing image: datasets/test-dataset-50/test.jpg
✅ Success! Histogram shape: (1152,)
   Entropy: 8.2341
   Processing time: 45.67 ms
   Output directory: output_20241201_143022

📁 Batch Processing Demo
==================================================
Testing directory: datasets/test-dataset-50
   (Demo mode: would process all images in directory)
   Use: python tools/test_histogram_generation.py --directory datasets/test-dataset-50/

📊 Validation Metrics Demo
==================================================
The tool validates histograms for:
  • Shape: Correct 1152 dimensions (8×12×12)
  • Normalization: Sum equals 1.0
  • Bounds: All values ≥ 0
  • Quality: Entropy, sparsity, distribution

📤 Output Format Options
==================================================
Available output formats:
  • JSON: Detailed results with all metadata
  • CSV: Flattened format for analysis
  • Both: Generate both formats
```

## Customization

### Modifying Demo Behavior

You can customize the demo by modifying the script:

```python
# Change test image path
image_path = "your/custom/image.jpg"

# Modify output format
tester = HistogramTester(output_format="csv", visualize=False)

# Add custom validation logic
def custom_validation(result):
    """Custom validation logic."""
    if result["validation"]["metrics"]["entropy"] > 7.0:
        print("✅ High entropy image - good color distribution")
    else:
        print("⚠️  Low entropy image - limited color variety")
```

### Adding New Demo Functions

```python
def demo_custom_feature():
    """Demonstrate a custom feature."""
    print("🔧 Custom Feature Demo")
    print("=" * 50)
    
    # Your custom demonstration logic here
    print("This is where you'd show your custom functionality")
    
    # Example: Custom histogram analysis
    tester = HistogramTester()
    # ... custom implementation

# Add to main demo execution
if __name__ == "__main__":
    demo_single_image()
    demo_batch_processing()
    demo_validation_metrics()
    demo_output_formats()
    demo_custom_feature()  # Your new function
```

## Best Practices

### Demo Development

1. **Keep it Simple**: Focus on core functionality demonstration
2. **Handle Errors**: Include proper error handling for missing files
3. **Clear Output**: Use descriptive messages and formatting
4. **Modular Design**: Separate concerns into distinct functions
5. **Documentation**: Include clear comments explaining each step

### Demo Usage

1. **Run from Project Root**: Ensure proper path resolution
2. **Check Dependencies**: Verify required datasets are available
3. **Review Output**: Understand what each demo section shows
4. **Customize as Needed**: Modify for your specific use cases
5. **Use as Reference**: Reference the code for your own implementations

## Troubleshooting

### Common Demo Issues

1. **Import Errors**: Ensure `src/` directory is in Python path
2. **Missing Datasets**: Check if test datasets are available
3. **Path Issues**: Run from project root directory
4. **Dependency Issues**: Verify all required packages are installed

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run demo with verbose output
demo_single_image()
```

The demo tool serves as both a learning resource and a practical example of how to use the Histogram Generation Testing Tool effectively in your own applications.
