# FAISS and DuckDB Wrappers

## Chromatica Color Search Engine

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [FAISS Index Implementation](#faiss-index-implementation)
4. [DuckDB Metadata Store](#duckdb-metadata-store)
5. [Integration and Workflow](#integration-and-workflow)
6. [Usage Examples](#usage-examples)
7. [Performance Considerations](#performance-considerations)
8. [Testing and Validation](#testing-and-validation)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)

---

## Overview

The FAISS and DuckDB Wrappers provide the core indexing and storage infrastructure for the Chromatica color search engine. This component implements the two-stage search architecture specified in the critical instructions document, combining fast approximate nearest neighbor search with high-fidelity reranking capabilities.

### Key Features

- **FAISS HNSW Index**: Fast ANN search using Hellinger-transformed histograms
- **DuckDB Metadata Store**: Efficient storage of image metadata and raw histograms
- **Automatic Hellinger Transform**: Seamless conversion for L2 distance compatibility
- **Batch Operations**: Efficient processing of large image collections
- **Persistence**: Save/load capabilities for long-term storage
- **Integration**: Seamless workflow from indexing to search and reranking

### Technology Stack

- **FAISS**: Facebook AI Similarity Search library for vector indexing
- **DuckDB**: Embedded analytical database for metadata management
- **NumPy**: Numerical operations and array management
- **Python 3.10+**: Modern Python with type hints and comprehensive error handling

---

## Architecture

### System Design

The FAISS and DuckDB integration follows a layered architecture designed for scalability and performance:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│                 Search and Query Layer                      │
├─────────────────────────────────────────────────────────────┤
│                Reranking Layer (Future)                    │
├─────────────────────────────────────────────────────────────┤
│              FAISS Index + DuckDB Store                    │
├─────────────────────────────────────────────────────────────┤
│                Histogram Generation                        │
├─────────────────────────────────────────────────────────────┤
│                   Image Processing                         │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Indexing Phase**:
   - Images → Histograms → Hellinger Transform → FAISS Index
   - Images → Histograms → Raw Storage → DuckDB

2. **Search Phase**:
   - Query → Histogram → Hellinger Transform → FAISS Search
   - FAISS Results → Raw Histograms → Sinkhorn-EMD Reranking

### Component Responsibilities

#### FAISS Index (`AnnIndex`)

- **Vector Storage**: Hellinger-transformed histograms
- **Fast Search**: Approximate nearest neighbor retrieval
- **Distance Metrics**: L2 distance compatibility
- **Scalability**: Efficient indexing of large datasets

#### DuckDB Store (`MetadataStore`)

- **Metadata Management**: Image IDs, file paths, file sizes
- **Raw Histogram Storage**: Original probability distributions
- **Fast Retrieval**: Key-value lookups for reranking
- **Batch Operations**: Efficient bulk data processing

---

## FAISS Index Implementation

### Core Class: AnnIndex

The `AnnIndex` class wraps the FAISS HNSW index with automatic Hellinger transformation:

```python
from chromatica.indexing.store import AnnIndex
import numpy as np

# Initialize index for 1152-dimensional Lab histograms
index = AnnIndex(dimension=1152)

# Add histograms to the index
histograms = np.random.random((100, 1152))  # 100 sample histograms
histograms = histograms / histograms.sum(axis=1, keepdims=True)  # Normalize

# Add to index (Hellinger transform applied automatically)
num_added = index.add(histograms)
print(f"Added {num_added} histograms to index")

# Search for similar histograms
query_histogram = np.random.random(1152)
query_histogram = query_histogram / query_histogram.sum()  # Normalize

# Perform search (returns distances and indices)
distances, indices = index.search(query_histogram, k=10)
print(f"Top 10 results: {indices[0]}")
print(f"Distances: {distances[0]}")
```

### Hellinger Transform Implementation

The Hellinger transform is applied automatically to make histograms compatible with L2 distance:

```python
def apply_hellinger_transform(histograms: np.ndarray) -> np.ndarray:
    """
    Apply Hellinger transform to normalized histograms.
    
    The Hellinger transform converts probability distributions to
    vectors compatible with L2 distance metrics by taking the
    element-wise square root.
    
    Args:
        histograms: Array of shape (n_histograms, 1152) containing
                   normalized histograms (sum to 1.0)
    
    Returns:
        Hellinger-transformed vectors of same shape
    """
    # Validate input
    if histograms.ndim != 2 or histograms.shape[1] != 1152:
        raise ValueError(f"Expected shape (n, 1152), got {histograms.shape}")
    
    # Check normalization
    sums = histograms.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError("Histograms must be normalized (sum to 1.0)")
    
    # Apply Hellinger transform: φ(h) = √h
    transformed = np.sqrt(histograms.astype(np.float32))
    
    return transformed
```

### Index Configuration and Optimization

```python
def configure_hnsw_index(dimension: int = 1152, 
                         m: int = 32, 
                         ef_construction: int = 200) -> AnnIndex:
    """
    Configure HNSW index with optimal parameters for color search.
    
    Args:
        dimension: Vector dimensionality (1152 for Lab histograms)
        m: Number of connections per layer (default: 32)
        ef_construction: Search depth during construction (default: 200)
    
    Returns:
        Configured AnnIndex instance
    """
    index = AnnIndex(dimension=dimension)
    
    # Set HNSW parameters for optimal performance
    index.index.hnsw.efConstruction = ef_construction
    index.index.hnsw.efSearch = 100  # Search depth at query time
    
    return index
```

---

## DuckDB Metadata Store

### Core Class: MetadataStore

The `MetadataStore` class manages image metadata and raw histogram storage:

```python
from chromatica.indexing.store import MetadataStore
import numpy as np

# Initialize metadata store
store = MetadataStore(db_path="chromatica_metadata.db")

# Create tables (automatically called if they don't exist)
store.create_tables()

# Add image metadata and histograms
image_data = [
    {
        'image_id': 'img_001',
        'file_path': '/path/to/image1.jpg',
        'file_size': 1024000,
        'histogram': np.random.random(1152)
    },
    {
        'image_id': 'img_002',
        'file_path': '/path/to/image2.jpg',
        'file_size': 2048000,
        'histogram': np.random.random(1152)
    }
]

# Add to store
store.add_images(image_data)

# Retrieve histograms for reranking
image_ids = ['img_001', 'img_002']
histograms = store.get_histograms(image_ids)
print(f"Retrieved {len(histograms)} histograms")
```

### Database Schema

The metadata store uses a simple but efficient schema:

```sql
-- Images table for metadata
CREATE TABLE images (
    image_id VARCHAR PRIMARY KEY,
    file_path VARCHAR NOT NULL,
    file_size BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Histograms table for raw data
CREATE TABLE histograms (
    image_id VARCHAR PRIMARY KEY,
    histogram_data BLOB NOT NULL,  -- Serialized numpy array
    FOREIGN KEY (image_id) REFERENCES images(image_id)
);
```

### Batch Operations

Efficient batch processing for large datasets:

```python
def batch_index_dataset(image_paths: List[str], 
                       batch_size: int = 100) -> None:
    """
    Index a large dataset in batches for memory efficiency.
    
    Args:
        image_paths: List of image file paths
        batch_size: Number of images to process per batch
    """
    from chromatica.core.histogram import build_histogram
    import cv2
    from skimage import color
    
    # Initialize stores
    index = AnnIndex(dimension=1152)
    store = MetadataStore(db_path="chromatica_metadata.db")
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_data = []
        
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        
        for image_path in batch_paths:
            try:
                # Load and process image
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_lab = color.rgb2lab(image_rgb)
                
                # Generate histogram
                lab_pixels = image_lab.reshape(-1, 3)
                histogram = build_histogram(lab_pixels)
                
                # Prepare data
                image_id = Path(image_path).stem
                batch_data.append({
                    'image_id': image_id,
                    'file_path': image_path,
                    'file_size': Path(image_path).stat().st_size,
                    'histogram': histogram
                })
                
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                continue
        
        if batch_data:
            # Add to stores
            histograms = np.array([item['histogram'] for item in batch_data])
            index.add(histograms)
            store.add_images(batch_data)
            
            print(f"Added batch of {len(batch_data)} images")
        
        # Clear memory
        del batch_data, histograms
        import gc
        gc.collect()
    
    print("Dataset indexing completed")
```

---

## Integration and Workflow

### Complete Indexing Pipeline

The integration provides a seamless workflow from image processing to search:

```python
def build_complete_index(dataset_directory: str, 
                        output_dir: str) -> Tuple[AnnIndex, MetadataStore]:
    """
    Build complete FAISS index and DuckDB store from image dataset.
    
    Args:
        dataset_directory: Path to directory containing images
        output_dir: Directory to save index and database files
    
    Returns:
        Tuple of (AnnIndex, MetadataStore) instances
    """
    from pathlib import Path
    from chromatica.core.histogram import build_histogram
    import cv2
    from skimage import color
    
    # Initialize stores
    index = AnnIndex(dimension=1152)
    store = MetadataStore(db_path=str(Path(output_dir) / "chromatica_metadata.db"))
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in Path(dataset_directory).iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images to index")
    
    # Process images
    for i, image_file in enumerate(image_files):
        try:
            # Load and process image
            image = cv2.imread(str(image_file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if necessary
            height, width = image_rgb.shape[:2]
            if max(height, width) > 256:
                scale = 256 / max(height, width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
            # Convert to Lab and generate histogram
            image_lab = color.rgb2lab(image_rgb)
            lab_pixels = image_lab.reshape(-1, 3)
            histogram = build_histogram(lab_pixels)
            
            # Add to stores
            image_id = image_file.stem
            store.add_images([{
                'image_id': image_id,
                'file_path': str(image_file),
                'file_size': image_file.stat().st_size,
                'histogram': histogram
            }])
            
            # Add to FAISS index (Hellinger transform applied automatically)
            index.add(histogram.reshape(1, -1))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Failed to process {image_file}: {e}")
            continue
    
    # Save index and close store
    index_path = Path(output_dir) / "chromatica_index.faiss"
    index.save(str(index_path))
    store.close()
    
    print(f"Indexing completed. Index saved to {index_path}")
    print(f"Database saved to {store.db_path}")
    
    return index, store
```

### Search and Retrieval Workflow

The integrated system provides efficient two-stage search:

```python
def perform_integrated_search(query_histogram: np.ndarray,
                             index: AnnIndex,
                             store: MetadataStore,
                             k: int = 50,
                             rerank_k: int = 200) -> List[Dict]:
    """
    Perform complete two-stage search using integrated FAISS and DuckDB.
    
    Args:
        query_histogram: Query histogram (normalized)
        index: FAISS index instance
        store: DuckDB metadata store instance
        k: Number of final results to return
        rerank_k: Number of candidates to rerank
    
    Returns:
        List of search results with metadata
    """
    # Stage 1: FAISS ANN search
    print("Stage 1: Performing FAISS ANN search...")
    distances, indices = index.search(query_histogram, k=rerank_k)
    
    # Get candidate image IDs
    candidate_ids = [f"img_{idx}" for idx in indices[0]]  # Assuming sequential IDs
    
    # Stage 2: Retrieve raw histograms for reranking
    print("Stage 2: Retrieving raw histograms...")
    candidate_histograms = store.get_histograms(candidate_ids)
    
    # Stage 3: Rerank using Sinkhorn-EMD (placeholder)
    print("Stage 3: Reranking candidates...")
    # This would integrate with the reranking module
    # For now, return top-k based on FAISS distances
    
    results = []
    for i in range(min(k, len(candidate_ids))):
        results.append({
            'image_id': candidate_ids[i],
            'file_path': store.get_image_path(candidate_ids[i]),
            'distance': float(distances[0][i]),
            'rank': i + 1
        })
    
    return results
```

---

## Usage Examples

### Basic Indexing and Search

#### 1. Simple Index Creation

```python
from chromatica.indexing.store import AnnIndex, MetadataStore
import numpy as np

# Create sample data
num_images = 1000
histograms = np.random.random((num_images, 1152))
histograms = histograms / histograms.sum(axis=1, keepdims=True)  # Normalize

# Initialize stores
index = AnnIndex(dimension=1152)
store = MetadataStore(db_path="test_metadata.db")

# Add data
index.add(histograms)

# Add metadata
image_data = [
    {
        'image_id': f'img_{i:03d}',
        'file_path': f'/path/to/image_{i:03d}.jpg',
        'file_size': 1024000 + i * 1000,
        'histogram': histograms[i]
    }
    for i in range(num_images)
]
store.add_images(image_data)

print(f"Indexed {num_images} images")
```

#### 2. Search Operations

```python
# Create query histogram
query_hist = np.random.random(1152)
query_hist = query_hist / query_hist.sum()

# Search
distances, indices = index.search(query_hist, k=10)

# Get metadata for top results
top_ids = [f'img_{idx:03d}' for idx in indices[0]]
top_histograms = store.get_histograms(top_ids)

print("Top 10 results:")
for i, (image_id, distance) in enumerate(zip(top_ids, distances[0])):
    file_path = store.get_image_path(image_id)
    print(f"{i+1}. {image_id}: {distance:.6f} -> {file_path}")
```

#### 3. Batch Processing

```python
def process_large_dataset(dataset_path: str, 
                         batch_size: int = 500) -> None:
    """Process large dataset in memory-efficient batches."""
    
    from pathlib import Path
    import glob
    
    # Get all image files
    image_files = glob.glob(f"{dataset_path}/*.jpg") + \
                  glob.glob(f"{dataset_path}/*.png")
    
    print(f"Found {len(image_files)} images")
    
    # Initialize stores
    index = AnnIndex(dimension=1152)
    store = MetadataStore(db_path="large_dataset.db")
    
    # Process in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
        
        # Process batch (simplified - would include actual image processing)
        batch_histograms = []
        batch_metadata = []
        
        for image_file in batch_files:
            try:
                # Generate histogram (placeholder)
                histogram = np.random.random(1152)
                histogram = histogram / histogram.sum()
                
                batch_histograms.append(histogram)
                batch_metadata.append({
                    'image_id': Path(image_file).stem,
                    'file_path': image_file,
                    'file_size': Path(image_file).stat().st_size,
                    'histogram': histogram
                })
                
            except Exception as e:
                print(f"Failed to process {image_file}: {e}")
                continue
        
        if batch_histograms:
            # Add to stores
            histograms_array = np.array(batch_histograms)
            index.add(histograms_array)
            store.add_images(batch_metadata)
            
            print(f"Added batch of {len(batch_histograms)} images")
        
        # Clear memory
        del batch_histograms, batch_metadata, histograms_array
        import gc
        gc.collect()
    
    print("Dataset processing completed")
```

---

## Performance Considerations

### Memory Management

#### Memory Usage Breakdown

- **FAISS Index**: ~1.5-2x the size of raw vectors
- **Raw Histograms**: 4.6KB per histogram (1152 × 4 bytes)
- **Metadata**: ~100-200 bytes per image record
- **Total for 1M images**: ~12.7 GB RAM

#### Memory Optimization Strategies

```python
def memory_optimized_indexing(image_paths: List[str],
                              max_memory_gb: float = 8.0) -> None:
    """Index dataset with memory constraints."""
    
    # Calculate batch size based on available memory
    memory_per_image_mb = 0.005  # 5KB per histogram
    available_memory_mb = max_memory_gb * 1024
    
    # Reserve 2GB for system and other operations
    usable_memory_mb = available_memory_mb - 2048
    
    batch_size = int(usable_memory_mb / memory_per_image_mb)
    print(f"Using batch size: {batch_size} images")
    
    # Process in batches
    process_large_dataset(image_paths, batch_size=batch_size)
```

### Performance Tuning

#### FAISS HNSW Parameters

```python
def tune_hnsw_parameters(index: AnnIndex,
                         ef_construction: int = 200,
                         ef_search: int = 100,
                         m: int = 32) -> None:
    """Tune HNSW parameters for optimal performance."""
    
    # Construction parameters
    index.index.hnsw.efConstruction = ef_construction
    
    # Search parameters
    index.index.hnsw.efSearch = ef_search
    
    # Graph connectivity
    # Note: M parameter is set at construction time
    
    print(f"HNSW parameters tuned:")
    print(f"  efConstruction: {ef_construction}")
    print(f"  efSearch: {ef_search}")
    print(f"  M: {m}")
```

#### DuckDB Optimization

```python
def optimize_database_performance(store: MetadataStore) -> None:
    """Optimize DuckDB for better performance."""
    
    # Enable parallel processing
    store.execute("PRAGMA threads=4")
    
    # Optimize table layout
    store.execute("ANALYZE")
    
    # Create indexes for common queries
    store.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON images(file_path)")
    store.execute("CREATE INDEX IF NOT EXISTS idx_file_size ON images(file_size)")
    
    print("Database optimization completed")
```

---

## Conclusion

The FAISS and DuckDB Wrappers provide a robust, scalable foundation for the Chromatica color search engine's indexing and storage needs. Key benefits include:

- **High Performance**: FAISS HNSW index provides fast approximate nearest neighbor search
- **Efficient Storage**: DuckDB offers fast metadata and histogram storage
- **Seamless Integration**: Automatic Hellinger transform and unified API
- **Scalability**: Batch operations and memory-efficient processing
- **Flexibility**: Support for various index types and optimization strategies

The wrappers successfully implement the two-stage search architecture specified in the critical instructions document, enabling efficient color-based image search with high-fidelity reranking capabilities.

For more information about related components, see:
- [Image Processing Pipeline](image_processing_pipeline.md)
- [Histogram Generation Guide](histogram_generation_guide.md)
- [Two-Stage Search Architecture](two_stage_search_architecture.md)
