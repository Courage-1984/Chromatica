# Offline Indexing Script

## Chromatica Color Search Engine

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Usage Examples](#usage-examples)
5. [Performance Considerations](#performance-considerations)
6. [Testing and Validation](#testing-and-validation)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The Offline Indexing Script is a comprehensive tool for building FAISS indexes and DuckDB metadata stores from large image datasets. This script implements the complete indexing pipeline specified in the critical instructions document, enabling efficient batch processing of image collections for the Chromatica color search engine.

### Key Features

- **Batch Processing**: Memory-efficient processing of large image collections
- **Progress Monitoring**: Real-time progress tracking and logging
- **Error Handling**: Robust error handling with detailed reporting
- **Memory Management**: Automatic memory optimization and garbage collection
- **Resume Capability**: Support for resuming interrupted indexing operations
- **Validation**: Comprehensive validation of generated histograms and indexes

### Technology Stack

- **Python 3.10+**: Modern Python with type hints and comprehensive error handling
- **FAISS**: Vector indexing for fast similarity search
- **DuckDB**: Embedded database for metadata storage
- **OpenCV**: Image loading and preprocessing
- **scikit-image**: Color space conversion
- **NumPy**: Numerical operations and histogram generation

---

## Architecture

### System Design

The offline indexing script follows a modular, pipeline-based architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Dataset Discovery                        │
├─────────────────────────────────────────────────────────────┤
│                 Image Processing Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                Histogram Generation                        │
├─────────────────────────────────────────────────────────────┤
│              FAISS Index Building                          │
├─────────────────────────────────────────────────────────────┤
│                DuckDB Metadata Storage                     │
├─────────────────────────────────────────────────────────────┤
│                 Validation & Output                        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Dataset Discovery**: Scan directories for supported image formats
2. **Image Processing**: Load, resize, and convert images to CIE Lab
3. **Histogram Generation**: Create normalized 1152-dimensional histograms
4. **Index Building**: Add Hellinger-transformed histograms to FAISS HNSW index
5. **Metadata Storage**: Store image metadata and raw histograms in DuckDB
6. **Validation**: Verify index quality and histogram properties
7. **Persistence**: Save index and database files

---

## Core Components

### 1. Main Indexing Script

The primary indexing script (`scripts/build_index.py`):

```python
#!/usr/bin/env python3
"""
Offline Indexing Script for Chromatica Color Search Engine

This script builds FAISS indexes and DuckDB metadata stores from image datasets.
It implements the complete indexing pipeline as specified in the critical instructions.

Usage:
    python scripts/build_index.py --dataset datasets/test-dataset-5000 --output test_index
    python scripts/build_index.py --help
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

from chromatica.indexing.store import AnnIndex, MetadataStore
from chromatica.core.histogram import build_histogram
from chromatica.utils.config import TOTAL_BINS, RERANK_K

def main():
    """Main indexing function."""
    parser = argparse.ArgumentParser(description="Build Chromatica index from image dataset")
    parser.add_argument("--dataset", required=True, help="Path to image dataset directory")
    parser.add_argument("--output", required=True, help="Output directory for index files")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--max-dimension", type=int, default=256, help="Maximum image dimension")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    
    if not dataset_path.exists():
        logger.error(f"Dataset directory does not exist: {dataset_path}")
        return 1
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run indexing
    try:
        success = build_index_from_dataset(
            dataset_path=dataset_path,
            output_path=output_path,
            batch_size=args.batch_size,
            max_dimension=args.max_dimension
        )
        
        if success:
            logger.info("Indexing completed successfully")
            return 0
        else:
            logger.error("Indexing failed")
            return 1
            
    except Exception as e:
        logger.error(f"Indexing failed with exception: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
```

### 2. Dataset Processing Pipeline

The core dataset processing function:

```python
def build_index_from_dataset(
    dataset_path: Path,
    output_path: Path,
    batch_size: int = 100,
    max_dimension: int = 256
) -> bool:
    """
    Build complete index from image dataset.
    
    Args:
        dataset_path: Path to dataset directory
        output_path: Path to output directory
        batch_size: Number of images to process per batch
        max_dimension: Maximum image dimension for processing
    
    Returns:
        True if indexing succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Initialize stores
    index = AnnIndex(dimension=TOTAL_BINS)
    store = MetadataStore(db_path=str(output_path / "chromatica_metadata.db"))
    
    try:
        # Get image files
        image_files = discover_image_files(dataset_path)
        logger.info(f"Found {len(image_files)} images to process")
        
        if len(image_files) == 0:
            logger.warning("No images found in dataset directory")
            return False
        
        # Process images in batches
        total_processed = 0
        total_failed = 0
        
        for batch_start in range(0, len(image_files), batch_size):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
            
            # Process batch
            batch_results = process_image_batch(
                batch_files, max_dimension, index, store
            )
            
            total_processed += batch_results['processed']
            total_failed += batch_results['failed']
            
            # Log progress
            logger.info(f"Batch completed: {batch_results['processed']} processed, {batch_results['failed']} failed")
            logger.info(f"Total progress: {total_processed}/{len(image_files)} ({total_processed/len(image_files)*100:.1f}%)")
            
            # Memory management
            import gc
            gc.collect()
        
        # Save index and close store
        index_path = output_path / "chromatica_index.faiss"
        index.save(str(index_path))
        store.close()
        
        # Generate summary report
        generate_indexing_report(
            output_path, total_processed, total_failed, len(image_files)
        )
        
        logger.info(f"Indexing completed successfully")
        logger.info(f"Index saved to: {index_path}")
        logger.info(f"Database saved to: {store.db_path}")
        logger.info(f"Final stats: {total_processed} processed, {total_failed} failed")
        
        return True
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return False
```

### 3. Image Discovery and Validation

```python
def discover_image_files(dataset_path: Path) -> List[Path]:
    """
    Discover all supported image files in dataset directory.
    
    Args:
        dataset_path: Path to dataset directory
    
    Returns:
        List of image file paths
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = []
    
    # Recursively scan directory
    for file_path in dataset_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            image_files.append(file_path)
    
    # Sort for consistent processing order
    image_files.sort()
    
    return image_files

def validate_image_file(file_path: Path) -> bool:
    """
    Validate that an image file can be processed.
    
    Args:
        file_path: Path to image file
    
    Returns:
        True if file is valid, False otherwise
    """
    # Check file exists and is readable
    if not file_path.exists() or not file_path.is_file():
        return False
    
    # Check file size (reasonable limits)
    try:
        file_size = file_path.stat().st_size
        if file_size < 100 or file_size > 100 * 1024 * 1024:  # 100B to 100MB
            return False
    except OSError:
        return False
    
    # Check file extension
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if file_path.suffix.lower() not in supported_extensions:
        return False
    
    return True
```

### 4. Batch Processing

```python
def process_image_batch(
    image_files: List[Path],
    max_dimension: int,
    index: AnnIndex,
    store: MetadataStore
) -> Dict[str, int]:
    """
    Process a batch of images.
    
    Args:
        image_files: List of image file paths
        max_dimension: Maximum image dimension
        index: FAISS index instance
        store: DuckDB metadata store instance
    
    Returns:
        Dictionary with processing statistics
    """
    logger = logging.getLogger(__name__)
    
    batch_histograms = []
    batch_metadata = []
    processed = 0
    failed = 0
    
    for image_file in image_files:
        try:
            # Validate file
            if not validate_image_file(image_file):
                logger.warning(f"Skipping invalid file: {image_file}")
                failed += 1
                continue
            
            # Process image
            histogram, metadata = process_single_image(image_file, max_dimension)
            
            batch_histograms.append(histogram)
            batch_metadata.append(metadata)
            processed += 1
            
        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            failed += 1
            continue
    
    # Add batch to stores
    if batch_histograms:
        try:
            histograms_array = np.array(batch_histograms)
            
            # Add to FAISS index
            index.add(histograms_array)
            
            # Add to metadata store
            store.add_images(batch_metadata)
            
            logger.debug(f"Successfully added batch of {len(batch_histograms)} images")
            
        except Exception as e:
            logger.error(f"Failed to add batch to stores: {e}")
            failed += processed
            processed = 0
    
    return {
        'processed': processed,
        'failed': failed
    }
```

### 5. Image Processing

```python
def process_single_image(
    image_path: Path,
    max_dimension: int
) -> tuple[np.ndarray, Dict]:
    """
    Process a single image through the complete pipeline.
    
    Args:
        image_path: Path to image file
        max_dimension: Maximum image dimension
    
    Returns:
        Tuple of (histogram, metadata)
    """
    import cv2
    from skimage import color
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if necessary
    height, width = image_rgb.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        image_rgb = cv2.resize(image_rgb, (new_width, new_height))
    
    # Convert to Lab color space
    image_lab = color.rgb2lab(image_rgb)
    
    # Generate histogram
    lab_pixels = image_lab.reshape(-1, 3)
    histogram = build_histogram(lab_pixels)
    
    # Prepare metadata
    metadata = {
        'image_id': image_path.stem,
        'file_path': str(image_path),
        'file_size': image_path.stat().st_size,
        'original_dimensions': (height, width),
        'processed_dimensions': image_rgb.shape[:2],
        'histogram': histogram
    }
    
    return histogram, metadata
```

---

## Usage Examples

### Basic Usage

#### 1. Index Small Dataset

```bash
# Index test dataset with default settings
python scripts/build_index.py \
    --dataset datasets/test-dataset-50 \
    --output test_index_50

# Enable verbose logging
python scripts/build_index.py \
    --dataset datasets/test-dataset-50 \
    --output test_index_50 \
    --verbose
```

#### 2. Index Large Dataset

```bash
# Index production dataset with custom batch size
python scripts/build_index.py \
    --dataset datasets/test-dataset-5000 \
    --output production_index \
    --batch-size 200 \
    --max-dimension 256

# Monitor progress with verbose logging
python scripts/build_index.py \
    --dataset datasets/test-dataset-5000 \
    --output production_index \
    --batch-size 200 \
    --verbose
```

#### 3. Custom Configuration

```bash
# High-quality processing
python scripts/build_index.py \
    --dataset datasets/high_quality_images \
    --output high_quality_index \
    --max-dimension 512 \
    --batch-size 50

# Memory-constrained processing
python scripts/build_index.py \
    --dataset datasets/large_dataset \
    --output memory_optimized_index \
    --batch-size 25 \
    --max-dimension 128
```

### Programmatic Usage

```python
from scripts.build_index import build_index_from_dataset
from pathlib import Path

# Build index programmatically
success = build_index_from_dataset(
    dataset_path=Path("datasets/test-dataset-200"),
    output_path=Path("custom_index"),
    batch_size=75,
    max_dimension=256
)

if success:
    print("Index built successfully")
else:
    print("Index building failed")
```

---

## Performance Considerations

### Memory Management

#### Batch Size Optimization

```python
def calculate_optimal_batch_size(available_memory_gb: float = 8.0) -> int:
    """
    Calculate optimal batch size based on available memory.
    
    Args:
        available_memory_gb: Available memory in GB
    
    Returns:
        Optimal batch size
    """
    # Memory requirements per image
    memory_per_image_mb = 0.010  # 10KB per histogram + overhead
    
    # Reserve memory for system operations
    reserved_memory_gb = 2.0
    usable_memory_gb = available_memory_gb - reserved_memory_gb
    
    # Calculate batch size
    optimal_batch_size = int((usable_memory_gb * 1024) / memory_per_image_mb)
    
    # Clamp to reasonable range
    optimal_batch_size = max(10, min(optimal_batch_size, 500))
    
    return optimal_batch_size
```

#### Memory Monitoring

```python
def monitor_memory_usage():
    """Monitor memory usage during indexing."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    logger.info(f"Memory percent: {memory_percent:.1f}%")
    
    return memory_info.rss / 1024 / 1024  # Return MB
```

### Performance Tuning

#### Batch Processing Optimization

```python
def optimize_batch_processing(
    dataset_size: int,
    available_memory_gb: float,
    target_processing_time_hours: float = 2.0
) -> Dict[str, int]:
    """
    Optimize batch processing parameters.
    
    Args:
        dataset_size: Total number of images
        available_memory_gb: Available memory in GB
        target_processing_time_hours: Target processing time
    
    Returns:
        Dictionary with optimized parameters
    """
    # Calculate optimal batch size
    optimal_batch_size = calculate_optimal_batch_size(available_memory_gb)
    
    # Estimate processing time
    estimated_time_per_image = 0.2  # seconds
    estimated_total_time = dataset_size * estimated_time_per_image / 3600  # hours
    
    # Adjust batch size if needed
    if estimated_total_time > target_processing_time_hours:
        # Increase batch size to reduce overhead
        optimal_batch_size = min(optimal_batch_size * 2, 500)
    
    return {
        'batch_size': optimal_batch_size,
        'estimated_time_hours': estimated_total_time,
        'memory_usage_gb': (dataset_size * 0.00001) + 2.0  # Rough estimate
    }
```

---

## Testing and Validation

### Index Validation

```python
def validate_generated_index(
    index_path: Path,
    metadata_db_path: Path,
    expected_image_count: int
) -> Dict[str, bool]:
    """
    Validate generated index and metadata store.
    
    Args:
        index_path: Path to FAISS index file
        metadata_db_path: Path to DuckDB database
        expected_image_count: Expected number of indexed images
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'index_exists': False,
        'index_loadable': False,
        'database_exists': False,
        'database_accessible': False,
        'image_count_match': False,
        'histogram_validation': False
    }
    
    try:
        # Check index file
        if index_path.exists():
            validation_results['index_exists'] = True
            
            # Try to load index
            index = AnnIndex(dimension=TOTAL_BINS)
            index.load(str(index_path))
            validation_results['index_loadable'] = True
            
            # Check vector count
            if index.total_vectors == expected_image_count:
                validation_results['image_count_match'] = True
        
        # Check database
        if metadata_db_path.exists():
            validation_results['database_exists'] = True
            
            # Try to access database
            store = MetadataStore(db_path=str(metadata_db_path))
            db_image_count = store.get_image_count()
            store.close()
            
            if db_image_count == expected_image_count:
                validation_results['database_accessible'] = True
        
        # Validate histograms
        if validation_results['database_accessible']:
            store = MetadataStore(db_path=str(metadata_db_path))
            sample_histograms = store.get_random_histograms(min(10, expected_image_count))
            store.close()
            
            if sample_histograms:
                histograms_array = np.array(sample_histograms)
                validation_results['histogram_validation'] = (
                    histograms_array.shape[1] == TOTAL_BINS and
                    np.allclose(histograms_array.sum(axis=1), 1.0, atol=1e-6)
                )
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
    
    return validation_results
```

### Performance Benchmarking

```python
def benchmark_indexing_performance(
    dataset_path: Path,
    batch_sizes: List[int] = [25, 50, 100, 200]
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark indexing performance with different batch sizes.
    
    Args:
        dataset_path: Path to test dataset
        batch_sizes: List of batch sizes to test
    
    Returns:
        Dictionary mapping batch sizes to performance metrics
    """
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size: {batch_size}")
        
        # Create temporary output directory
        temp_output = Path(f"temp_benchmark_{batch_size}")
        temp_output.mkdir(exist_ok=True)
        
        try:
            # Time indexing
            start_time = time.time()
            success = build_index_from_dataset(
                dataset_path=dataset_path,
                output_path=temp_output,
                batch_size=batch_size
            )
            end_time = time.time()
            
            if success:
                # Calculate metrics
                total_time = end_time - start_time
                image_count = len(discover_image_files(dataset_path))
                throughput = image_count / total_time
                
                results[batch_size] = {
                    'total_time': total_time,
                    'throughput_images_per_second': throughput,
                    'memory_peak_mb': monitor_memory_usage()
                }
                
                logger.info(f"Batch size {batch_size}: {throughput:.2f} images/sec")
            else:
                logger.warning(f"Batch size {batch_size}: Failed")
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_output, ignore_errors=True)
    
    return results
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues

**Problem**: Out of memory during indexing
**Solution**: Reduce batch size and monitor memory usage

```python
def diagnose_memory_issues():
    """Diagnose and suggest solutions for memory issues."""
    
    import psutil
    
    # Check system memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / 1024 / 1024 / 1024
    
    logger.info(f"Available system memory: {available_gb:.1f} GB")
    
    if available_gb < 4.0:
        logger.warning("Low memory detected. Recommendations:")
        logger.warning("  - Reduce batch size to 25-50")
        logger.warning("  - Close other applications")
        logger.warning("  - Use smaller max_dimension (128)")
    
    # Check process memory
    process = psutil.Process()
    process_memory_mb = process.memory_info().rss / 1024 / 1024
    
    logger.info(f"Current process memory: {process_memory_mb:.1f} MB")
    
    if process_memory_mb > available_gb * 1024 * 0.8:
        logger.warning("Process using high memory. Consider reducing batch size.")
```

#### 2. File Permission Issues

**Problem**: Cannot write to output directory
**Solution**: Check permissions and create directories

```python
def check_output_permissions(output_path: Path) -> bool:
    """Check if output directory is writable."""
    
    try:
        # Try to create test file
        test_file = output_path / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        logger.error(f"Cannot write to output directory: {output_path}")
        logger.error("Check directory permissions and disk space")
        return False
```

#### 3. Corrupted Images

**Problem**: Some images fail to process
**Solution**: Implement robust error handling and logging

```python
def handle_corrupted_images(failed_files: List[Path], output_path: Path):
    """Handle and report corrupted images."""
    
    if failed_files:
        # Create failure report
        report_path = output_path / "failed_images_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Failed Image Processing Report\n")
            f.write("=" * 40 + "\n\n")
            
            for failed_file in failed_files:
                f.write(f"Failed: {failed_file}\n")
        
        logger.warning(f"Created failure report: {report_path}")
        logger.warning(f"Total failed images: {len(failed_files)}")
```

---

## Conclusion

The Offline Indexing Script provides a robust, scalable solution for building FAISS indexes and DuckDB metadata stores from large image datasets. Key benefits include:

- **Efficient Processing**: Batch processing with memory optimization
- **Robust Error Handling**: Comprehensive error handling and recovery
- **Progress Monitoring**: Real-time progress tracking and validation
- **Flexible Configuration**: Customizable batch sizes and processing parameters
- **Production Ready**: Comprehensive logging and validation

The script successfully implements the indexing pipeline specified in the critical instructions document, enabling efficient color-based image search with high-fidelity reranking capabilities.

For more information about related components, see:
- [Image Processing Pipeline](image_processing_pipeline.md)
- [FAISS and DuckDB Wrappers](faiss_duckdb_wrappers.md)
- [Histogram Generation Guide](histogram_generation_guide.md)
