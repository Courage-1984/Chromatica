# Chromatica Tools

This directory contains utility tools for testing, debugging, and analyzing the Chromatica color search engine.

## Histogram Generation Testing Tool

The `test_histogram_generation.py` tool provides comprehensive testing capabilities for the histogram generation module. It can process single images or entire directories, generating histograms and providing detailed analysis, validation, and visualization.

### Features

- **Single Image Testing**: Test histogram generation on individual images
- **Batch Directory Processing**: Process entire directories of images
- **Automatic Image Processing**: Load, resize, and convert images to Lab color space
- **Histogram Validation**: Comprehensive quality checks and validation
- **Performance Benchmarking**: Measure generation time and memory usage
- **Visualization**: Generate 3D plots and 2D projections of histograms
- **Multiple Output Formats**: JSON, CSV, or both formats
- **Comprehensive Logging**: Detailed logging for debugging and analysis

### Installation

1. Ensure you have the required dependencies:

   ```bash
   pip install -r tools/requirements.txt
   ```

2. Make sure you're in the Chromatica project root directory and have activated the virtual environment:
   ```bash
   venv311\Scripts\activate  # Windows
   # or
   source venv311/bin/activate  # Linux/Mac
   ```

### Usage

#### Basic Usage

```bash
# Test a single image
python tools/test_histogram_generation.py --image path/to/image.jpg

# Test a directory of images
python tools/test_histogram_generation.py --directory path/to/images/

# Test with verbose logging
python tools/test_histogram_generation.py --image path/to/image.jpg --verbose
```

#### Advanced Options

```bash
# Disable visualization generation
python tools/test_histogram_generation.py --image path/to/image.jpg --no-visualize

# Output results in both JSON and CSV formats
python tools/test_histogram_generation.py --image path/to/image.jpg --output-format both

# Test directory with custom options
python tools/test_histogram_generation.py --directory path/to/images/ --output-format csv --no-visualize
```

### Output Structure

The tool creates organized output directories in the source directory of the images being processed:

1. **`histograms/` folder**: Contains histogram data (.npy) and visualization plots (.png)
2. **`reports/` folder**: Contains all text-based results (JSON, CSV, summary reports)

#### Example Output Directory Structure

```
source_directory/
├── image1.jpg
├── image2.png
├── histograms/
│   ├── image1_histogram.npy
│   ├── image1_histogram_analysis.png
│   ├── image2_histogram.npy
│   └── image2_histogram_analysis.png
└── reports/
    ├── batch_histogram_test_20241201_143022.json
    ├── batch_histogram_test_20241201_143022.csv
    └── summary_report_20241201_143022.json
```

### Output Formats

#### JSON Output

Contains detailed information about each image:

- Image metadata (size, dimensions, resizing info)
- Histogram data (shape, data type, file path)
- Validation results (pass/fail, errors, warnings, metrics)
- Performance benchmarks (timing, memory usage)
- File paths for generated outputs

#### CSV Output

Flattened format suitable for analysis in spreadsheet applications:

- One row per image
- Key metrics (entropy, sparsity, processing time)
- Validation status and error messages
- Performance statistics

### Validation Metrics

The tool validates histograms for:

- **Shape**: Correct 1152 dimensions (8×12×12)
- **Normalization**: Sum equals 1.0 (probability distribution)
- **Bounds**: All values ≥ 0
- **Quality**: Entropy, sparsity, and distribution analysis

### Performance Metrics

- **Processing Time**: Mean, standard deviation, min/max times
- **Memory Usage**: Estimated memory consumption
- **Throughput**: Pixels processed per second
- **Comparison**: Full vs. fast histogram generation methods

### Visualization Features

1. **3D Scatter Plot**: Shows distribution of non-zero histogram bins
2. **L* vs a* Projection**: 2D heatmap of lightness vs. green-red
3. **L* vs b* Projection**: 2D heatmap of lightness vs. blue-yellow
4. **Value Distribution**: Histogram of non-zero bin values

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### Error Handling

The tool provides comprehensive error handling:

- **Image Loading Errors**: Invalid files, unsupported formats
- **Processing Errors**: Color conversion failures, histogram generation issues
- **Validation Errors**: Incorrect histogram properties
- **File System Errors**: Permission issues, disk space problems

### Examples

#### Test a Single Test Image

```bash
# Test one of the test dataset images
python tools/test_histogram_generation.py --image datasets/test-dataset-50/test.jpg --verbose
```

#### Test the Entire Test Dataset

```bash
# Process all images in the test dataset
python tools/test_histogram_generation.py --directory datasets/test-dataset-50/ --output-format both
```

#### Quick Test Without Visualization

```bash
# Fast test for performance validation
python tools/test_histogram_generation.py --image datasets/test-dataset-50/test.jpg --no-visualize --output-format json
```

### Troubleshooting

#### Common Issues

1. **Import Errors**: Ensure you're in the project root and virtual environment is activated
2. **Missing Dependencies**: Install required packages with `pip install -r tools/requirements.txt`
3. **Image Loading Failures**: Check file paths and image format support
4. **Memory Issues**: Large images may require more memory; consider resizing

#### Debug Mode

Use the `--verbose` flag for detailed logging:

```bash
python tools/test_histogram_generation.py --image path/to/image.jpg --verbose
```

### Integration with Development Workflow

This tool is designed to integrate with the Chromatica development workflow:

- **Unit Testing**: Validate histogram generation correctness
- **Performance Testing**: Benchmark processing speed and memory usage
- **Quality Assurance**: Ensure histograms meet specifications
- **Debugging**: Identify issues in the histogram generation pipeline

### Future Enhancements

Planned improvements include:

- **Batch Comparison**: Compare histograms between similar images
- **Statistical Analysis**: Advanced statistical measures and distributions
- **Export Options**: Additional output formats (HDF5, Parquet)
- **Web Interface**: Browser-based visualization and analysis
- **Integration Tests**: End-to-end testing with FAISS and EMD components

---

For more information about the Chromatica project, see the main README.md and documentation in the `docs/` directory.
