# Build Index Script Documentation

## Overview

The `scripts/build_index.py` script is the main entry point for building the offline index that powers the Chromatica color search engine. This script processes a directory of images and populates both the FAISS HNSW index and DuckDB metadata store, creating the foundation for fast and accurate color-based image search.

## Features

- **Batch Processing**: Efficiently processes large image datasets in configurable batches
- **Comprehensive Logging**: Detailed progress tracking and error reporting
- **Error Handling**: Graceful degradation with detailed error messages
- **Performance Monitoring**: Timing and throughput statistics
- **Validation**: Automatic histogram validation and quality checks
- **Flexible Configuration**: Command-line options for customization

## Usage

### Basic Usage

```bash
# Activate virtual environment first
venv311\Scripts\activate

# Basic indexing with default settings
python scripts/build_index.py <image_directory>

# Example with test dataset
python scripts/build_index.py datasets/test-dataset-20
```

### Advanced Usage

```bash
# Custom output directory and batch size
python scripts/build_index.py datasets/test-dataset-5000 --output-dir ./production_index --batch-size 200

# Verbose logging for debugging
python scripts/build_index.py datasets/test-dataset-200 --verbose

# Full example with all options
python scripts/build_index.py ./data/unsplash-lite --output-dir ./index --batch-size 100 --verbose
```

## Command-Line Arguments

### Required Arguments

- `image_directory`: Path to the directory containing images to index

### Optional Arguments

- `--output-dir`: Output directory for index files (default: `./index`)
- `--batch-size`: Number of images to process in each batch (default: 100)
- `--verbose, -v`: Enable verbose logging (DEBUG level)

## Output Files

The script generates two main output files in the specified output directory:

### 1. FAISS Index (`chromatica_index.faiss`)

- **Purpose**: Stores Hellinger-transformed histograms for fast approximate nearest neighbor search
- **Format**: FAISS binary format
- **Content**: 1,152-dimensional vectors (8x12x12 Lab color space bins)
- **Usage**: First stage of the two-stage search pipeline

### 2. Metadata Database (`chromatica_metadata.db`)

- **Purpose**: Stores image metadata and raw histograms for reranking
- **Format**: DuckDB database
- **Content**: Image IDs, file paths, raw histograms, file sizes
- **Usage**: Second stage reranking with Sinkhorn-EMD

## Processing Pipeline

The script follows a comprehensive image processing pipeline:

### 1. Image Discovery

- Scans the input directory for supported image formats
- Supports: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`
- Case-insensitive file extension matching
- Sorted processing order for consistent results

### 2. Batch Processing

- Processes images in configurable batches for memory efficiency
- Default batch size: 100 images
- Configurable via `--batch-size` parameter
- Progress tracking and error handling per batch

### 3. Image Processing (Per Image)

- **Loading**: Uses OpenCV to load images in BGR format
- **Resizing**: Maintains aspect ratio, max dimension 256px
- **Color Conversion**: BGR → RGB → CIE Lab (D65 illuminant)
- **Histogram Generation**: Tri-linear soft assignment, 8x12x12 bins
- **Validation**: Ensures histogram quality and normalization

### 4. Indexing

- **FAISS Index**: Adds Hellinger-transformed histograms
- **Metadata Store**: Stores raw histograms and image metadata
- **Batch Operations**: Efficient bulk insertion for performance

### 5. Finalization

- Saves FAISS index to disk
- Closes database connections
- Generates comprehensive statistics

## Logging and Monitoring

### Log Levels

- **INFO**: General progress and statistics
- **DEBUG**: Detailed processing information (with `--verbose`)
- **WARNING**: Non-fatal issues
- **ERROR**: Processing failures

### Log Outputs

- **Console**: Real-time progress and summary information
- **File**: Detailed logs saved to `logs/build_index_<timestamp>.log`

### Performance Metrics

- Total processing time
- Average time per image
- Processing throughput (images/second)
- Success rate percentage
- Batch timing statistics

## Error Handling

### Graceful Degradation

- Individual image failures don't stop the entire process
- Detailed error logging for troubleshooting
- Batch-level error recovery
- Comprehensive error statistics

### Common Error Scenarios

- **File Not Found**: Invalid image paths
- **Corrupted Images**: Unreadable image files
- **Memory Issues**: Large images or insufficient memory
- **Permission Errors**: File access restrictions

## Performance Considerations

### Memory Usage

- Batch processing prevents memory overflow
- Configurable batch sizes for different system capabilities
- Efficient histogram storage and processing

### Processing Speed

- Typical performance: ~200ms per image
- Throughput: 4-5 images/second on modern hardware
- Optimized for the 8x12x12 binning grid

### Scalability

- Tested with datasets up to 5,000 images
- Recommended for production: 7,500+ images
- Linear scaling with dataset size

## Configuration Integration

The script integrates with the project's configuration system:

- Uses `TOTAL_BINS` (1,152) for histogram dimensions
- Applies `HNSW_M=32` for FAISS index configuration
- Follows `MAX_IMAGE_DIMENSION=256` for image resizing
- Implements all algorithmic specifications from critical instructions

## Examples

### Development Testing

```bash
# Quick test with small dataset
python scripts/build_index.py datasets/test-dataset-20 --verbose

# Medium-scale validation
python scripts/build_index.py datasets/test-dataset-200 --batch-size 50
```

### Production Deployment

```bash
# Large-scale indexing
python scripts/build_index.py datasets/test-dataset-5000 --output-dir ./production_index --batch-size 200

# Custom dataset
python scripts/build_index.py ./data/unsplash-lite --output-dir ./index --batch-size 100
```

## Integration with Search System

The generated index files are used by the search system:

### Search Pipeline Integration

1. **FAISS Index**: Powers the first-stage ANN search
2. **Metadata Database**: Enables second-stage Sinkhorn-EMD reranking
3. **Image Metadata**: Provides file paths for result display

### API Integration

- Index files are loaded by the FastAPI application
- Search endpoints use the generated indexes
- Real-time search performance depends on index quality

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Memory Issues**: Reduce batch size
3. **Permission Errors**: Check file access permissions
4. **Corrupted Images**: Review error logs for specific files

### Debug Mode

```bash
# Enable verbose logging for detailed debugging
python scripts/build_index.py <directory> --verbose
```

### Log Analysis

- Check `logs/build_index_<timestamp>.log` for detailed information
- Look for ERROR and WARNING messages
- Monitor processing times and success rates

## Best Practices

### Dataset Preparation

- Use consistent image formats (JPEG/PNG recommended)
- Ensure images are not corrupted
- Organize images in a single directory
- Use descriptive filenames for easier debugging

### Performance Optimization

- Adjust batch size based on available memory
- Use SSD storage for better I/O performance
- Monitor system resources during processing
- Process during off-peak hours for large datasets

### Quality Assurance

- Always run with verbose logging for initial tests
- Validate index files after generation
- Test search functionality with generated indexes
- Monitor success rates and error patterns

## Dependencies

The script requires the following components:

- **Core Modules**: `chromatica.indexing.pipeline`, `chromatica.indexing.store`
- **Configuration**: `chromatica.utils.config`
- **External Libraries**: `numpy`, `faiss-cpu`, `duckdb`, `opencv-python`, `scikit-image`

## Future Enhancements

Potential improvements for future versions:

- **Parallel Processing**: Multi-threaded image processing
- **Resume Capability**: Continue interrupted indexing
- **Progress Bars**: Visual progress indicators
- **Configuration Files**: YAML/JSON configuration support
- **Cloud Integration**: S3/GCS dataset support

## Related Documentation

- [Image Processing Pipeline](image_processing_pipeline.md)
- [FAISS and DuckDB Integration](faiss_duckdb_integration.md)
- [Configuration Management](configuration.md)
- [Troubleshooting Guide](troubleshooting.md)
