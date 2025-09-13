# Chromatica Tools

This directory contains utility tools for testing, debugging, and analyzing the Chromatica color search engine.

## Sanity Check Script

The `run_sanity_checks.py` script (located in `scripts/`) is a comprehensive validation tool that programmatically executes the four sanity checks defined in Section F of the critical instructions document. This script serves as a critical quality assurance tool to validate that the Chromatica color search engine is working correctly.

### Features

- **Programmatic Sanity Checks**: Automatically executes all four sanity checks from Section F
- **Comprehensive Validation**: Tests monochrome, complementary, weight sensitivity, and subtle hue queries
- **Real-Time Results Display**: Shows top 5 results for each check with detailed metrics
- **Performance Monitoring**: Tracks query time, search time, and total processing time
- **Detailed Logging**: Comprehensive logging to both console and log files
- **Error Handling**: Robust error handling with clear failure reporting
- **Summary Reporting**: Generates comprehensive summary reports with success/failure counts

### Usage

```bash
# Activate virtual environment
venv311\Scripts\activate

# Run sanity checks with default settings
python scripts/run_sanity_checks.py

# Run with verbose logging
python scripts/run_sanity_checks.py --verbose

# Run with custom number of top results
python scripts/run_sanity_checks.py --top-k 10
```

### Sanity Checks

1. **Monochrome Red Query**: 100% #FF0000 should return red-dominant images
2. **Complementary Colors Query**: 50% #0000FF and 50% #FFA500 should return contrasting images
3. **Weight Sensitivity Test 1**: 90% red, 10% blue should yield red-dominant results
4. **Weight Sensitivity Test 2**: 10% red, 90% blue should yield blue-dominant results
5. **Subtle Hues Test**: Similar colors #FF0000 and #EE0000 should test fine-grained perception

For detailed documentation, see `docs/tools_sanity_checks.md`.

---

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

Contains tabular data for easy analysis:

- Image information (name, path, size, dimensions)
- Histogram properties (shape, sum, entropy, sparsity)
- Validation results (pass/fail status, error details)
- Performance metrics (processing time, memory usage)
- Output file paths and metadata

---

## Search System Testing Tool

The `test_search_system.py` tool provides comprehensive testing of the complete two-stage search pipeline. It validates the integration between all components: histogram generation, FAISS indexing, metadata storage, and Sinkhorn reranking.

### Features

- **Complete System Validation**: Tests the entire search pipeline end-to-end
- **Component Integration Testing**: Validates interaction between all modules
- **Performance Benchmarking**: Measures search performance characteristics
- **Error Handling Validation**: Tests edge cases and error conditions
- **Real Image Testing**: Uses actual images from test datasets
- **Comprehensive Logging**: Detailed logging for debugging and analysis

### Usage

```bash
# Basic system testing
python tools/test_search_system.py

# Testing with custom dataset
python tools/test_search_system.py --dataset datasets/test-dataset-50

# Performance testing
python tools/test_search_system.py --performance

# Verbose logging
python tools/test_search_system.py --verbose

# Custom image limit
python tools/test_search_system.py --max-images 100
```

### Test Coverage

The tool performs comprehensive testing across multiple phases:

1. **Phase 1**: Creating test index and metadata store
2. **Phase 2**: Validating search system components
3. **Phase 3**: Testing basic search functionality
4. **Phase 4**: Testing search with real images
5. **Phase 5**: Testing error handling and edge cases
6. **Phase 6**: Performance testing (optional)

---

## Search System Demonstration Tool

The `demo_search.py` tool showcases the complete search system functionality with interactive demonstrations and clear output formatting.

### Features

- **Interactive Demonstrations**: Shows different types of searches
- **Performance Analysis**: Displays timing and throughput metrics
- **System Validation**: Runs complete system validation
- **Clear Output Formatting**: Easy-to-read results and statistics
- **Multiple Query Types**: Synthetic and real image queries

### Usage

```bash
# Basic demonstration
python tools/demo_search.py

# Custom dataset demonstration
python tools/demo_search.py --dataset datasets/test-dataset-50

# Verbose logging
python tools/demo_search.py --verbose

# Custom image limit
python tools/demo_search.py --max-images 30
```

### Demonstration Types

1. **Demo 1**: Synthetic Query Search - Tests with generated histograms
2. **Demo 2**: Real Image Query Search - Tests with actual images
3. **Demo 3**: Performance Analysis - Measures search performance
4. **Demo 4**: System Validation - Validates complete system

---

## Other Testing Tools

### Query Processor Testing (`test_query_processor.py`)

Tests the query processing functionality for converting color queries into histogram representations.

### Reranking Testing (`test_reranking.py`)

Tests the Sinkhorn reranking system for high-fidelity distance computation.

### FAISS and DuckDB Testing (`test_faiss_duckdb.py`)

Tests the FAISS HNSW index and DuckDB metadata store integration.

### Image Pipeline Testing (`test_image_pipeline.py`)

Tests the complete image processing pipeline from loading to histogram generation.

---

## Running All Tests

To run a comprehensive test of all components:

```bash
# Activate virtual environment
venv311\Scripts\activate  # Windows
# or
source venv311/bin/activate  # Linux/Mac

# Run individual test tools
python tools/test_histogram_generation.py --directory datasets/test-dataset-20
python tools/test_search_system.py --performance
python tools/demo_search.py --dataset datasets/test-dataset-50

# Or run the comprehensive test suite
python tools/test_search_system.py --performance --verbose
```

## Advanced Visualization Tools

The Chromatica project now includes three comprehensive visualization tools that provide rich analysis capabilities for color palettes, search results, and interactive color exploration.

### Color Palette Visualizer (`visualize_color_palettes.py`)

**Purpose**: Extract and visualize dominant colors from images, analyze color distributions, and compare color palettes across multiple images.

**Features**:

- **Dominant Color Extraction**: Uses K-means clustering to identify primary colors
- **Palette Visualization**: Create color swatches with percentage distributions
- **Histogram Analysis**: Visualize CIE Lab color space distributions
- **Batch Processing**: Analyze multiple images simultaneously
- **Export Capabilities**: Save visualizations and reports

**Usage**:

```bash
# Analyze a single image
python tools/visualize_color_palettes.py --image datasets/test-dataset-20/7349035.jpg

# Compare multiple images
python tools/visualize_color_palettes.py --compare image1.jpg image2.jpg image3.jpg

# Batch analysis with save
python tools/visualize_color_palettes.py --batch datasets/test-dataset-20 --save
```

### Search Results Visualizer (`visualize_search_results.py`)

**Purpose**: Comprehensive visualization and analysis of search results, including ranking analysis, performance metrics, and result galleries.

**Features**:

- **Ranking Analysis**: Visualize search result rankings and distances
- **Performance Metrics**: Analyze search timing and performance breakdown
- **Color Similarity Mapping**: Heatmaps showing color relationships
- **Result Galleries**: Interactive display of search results
- **API Integration**: Direct querying of the Chromatica API
- **Export Capabilities**: Save all visualizations and reports

**Usage**:

```bash
# Query API and visualize results
python tools/visualize_search_results.py --api-query "FF0000" --k 10

# Load results from file
python tools/visualize_search_results.py --results search_results.json

# Save visualizations
python tools/visualize_search_results.py --api-query "FF0000,00FF00" --save
```

### Interactive Color Explorer (`color_explorer.py`)

**Purpose**: Interactive tool for exploring color combinations, generating color harmonies, and experimenting with different color schemes in real-time.

**Features**:

- **Interactive Color Picker**: Add colors by hex codes with weights
- **Color Harmony Generation**: Automatic generation of complementary, analogous, triadic, and other color schemes
- **Real-time Preview**: Live visualization of color combinations
- **API Integration**: Test color combinations with live search
- **Palette Export**: Save color palettes for later use

**Usage**:

```bash
# Start the interactive explorer
python tools/color_explorer.py

# Connect to specific API instance
python tools/color_explorer.py --api-url http://localhost:8000
```

### Installation and Dependencies

The visualization tools require additional dependencies:

```bash
pip install matplotlib seaborn requests
```

For detailed documentation on all visualization tools, see `docs/visualization_tools_guide.md`.

---

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the project root directory and virtual environment is activated
2. **Missing Dependencies**: Install required packages with `pip install -r tools/requirements.txt`
3. **Dataset Not Found**: Verify dataset paths exist and contain image files
4. **Memory Issues**: Reduce batch sizes or image limits for large datasets
5. **Matplotlib Backend Issues**: Set `export MPLBACKEND=TkAgg` for interactive tools

### Getting Help

- Check the logs in the `logs/` directory for detailed error information
- Enable verbose logging with `--verbose` flag for more detailed output
- Review the progress report in `docs/progress.md` for implementation status
- Check the troubleshooting guide in `docs/troubleshooting.md` for common solutions
- For visualization tools, see `docs/visualization_tools_guide.md` for detailed usage
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

---

## Output Cleanup Tool

The `cleanup_outputs.py` tool provides comprehensive cleanup functionality for managing output files generated during development, testing, and production operations. This tool helps maintain a clean development environment by providing selective or complete removal of generated files.

### Features

- **Selective Cleanup**: Choose specific output types to clean
- **Interactive Mode**: User-friendly interface for guided cleanup
- **Batch Operations**: Clean multiple output types simultaneously
- **Safety Features**: Confirmation prompts and dry-run mode
- **Size Reporting**: Shows disk space usage and freed space
- **Script Generation**: Create standalone cleanup scripts

### Supported Output Types

- **Histograms**: Generated histogram files (`.npy`) from dataset processing
- **Reports**: Analysis reports, logs, and documentation files
- **Logs**: Application and build log files
- **Test Index**: FAISS index and DuckDB metadata files
- **Cache**: Python bytecode cache files (`__pycache__`)
- **Temp**: Temporary files and system artifacts

### Usage

```bash
# Interactive mode - guided cleanup selection
python tools/cleanup_outputs.py

# Clean all outputs with confirmation
python tools/cleanup_outputs.py --all --confirm

# Clean specific output types
python tools/cleanup_outputs.py --histograms --reports --logs

# Dry run to preview what would be deleted
python tools/cleanup_outputs.py --all --dry-run

# Clean dataset outputs only
python tools/cleanup_outputs.py --datasets

# Create standalone cleanup script
python tools/cleanup_outputs.py --datasets --create-script
```

### Safety Features

- **Confirmation Prompts**: Interactive mode requires explicit confirmation
- **Dry Run Mode**: Preview what would be deleted without making changes
- **Error Handling**: Graceful handling of permission errors
- **Logging**: All operations logged to `logs/cleanup.log`

For detailed documentation, see `docs/tools_cleanup_outputs.md`.

---

For more information about the Chromatica project, see the main README.md and documentation in the `docs/` directory.
