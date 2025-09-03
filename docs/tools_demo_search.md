# Demo Search Tool

## Overview

The `demo_search.py` tool is a comprehensive demonstration script that showcases the complete two-stage search pipeline of the Chromatica color search engine. It demonstrates the entire workflow from index creation to search execution and reranking, providing a practical example of how the system works end-to-end.

## Purpose

This tool is designed to:
- Demonstrate the complete search pipeline functionality
- Show how to create and populate search indices
- Illustrate different types of search queries
- Demonstrate reranking capabilities
- Provide performance benchmarking examples
- Serve as a reference implementation

## Features

- **Complete Pipeline Demo**: End-to-end search system demonstration
- **Index Creation**: Automatic index building from test datasets
- **Multiple Search Types**: Query by image, histogram, and color values
- **Reranking Showcase**: Sinkhorn-EMD reranking demonstration
- **Performance Analysis**: Timing and accuracy metrics
- **Interactive Examples**: Step-by-step workflow demonstration

## Usage

### Basic Usage

```bash
# Run with default settings
python tools/demo_search.py

# Run with custom dataset
python tools/demo_search.py --dataset datasets/test-dataset-50

# Enable verbose logging
python tools/demo_search.py --verbose
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Path to test dataset directory | `datasets/test-dataset-20` |
| `--verbose` | Enable detailed logging | False |

## Core Components

### Demo Index Creation

The tool creates a demonstration index from test images:

```python
def create_demo_index(dataset_path: str, max_images: int = 20) -> tuple[AnnIndex, MetadataStore]:
    """Create a demo index with sample images."""
    
    # Initialize components
    index = AnnIndex()
    store = MetadataStore(":memory:")  # Use in-memory database for demo
    
    # Process images and build index
    histograms = []
    metadata_batch = []
    
    for image_file in image_files:
        # Generate histogram
        histogram = generate_histogram(str(image_file))
        
        # Validate histogram
        if histogram.shape != (1152,) or not np.isclose(histogram.sum(), 1.0, atol=1e-6):
            continue
        
        # Prepare metadata
        metadata = {
            "image_id": f"demo_{image_file.stem}",
            "file_path": str(image_file),
            "histogram": histogram,
            "file_size": image_file.stat().st_size
        }
        
        histograms.append(histogram)
        metadata_batch.append(metadata)
    
    # Add to index and store
    histograms_array = np.array(histograms, dtype=np.float64)
    added_count = index.add(histograms_array)
    stored_count = store.add_batch(metadata_batch)
    
    return index, store
```

**What it demonstrates:**
- Index initialization and configuration
- Batch histogram generation and validation
- Metadata preparation and storage
- Integration between FAISS index and DuckDB store

### Search Pipeline Demonstration

The tool demonstrates the complete search workflow:

```python
def demonstrate_search_pipeline(index: AnnIndex, store: MetadataStore) -> None:
    """Demonstrate the complete search pipeline."""
    
    print("\n🔍 Demonstrating Search Pipeline")
    print("=" * 50)
    
    # Get a sample image for querying
    sample_metadata = store.get_all()[0]
    sample_histogram = sample_metadata["histogram"]
    
    print(f"Query image: {sample_metadata['file_path']}")
    print(f"Histogram shape: {sample_histogram.shape}")
    
    # Perform search
    results = find_similar(
        sample_histogram,
        index,
        store,
        k=5,
        rerank_k=3
    )
    
    # Display results
    print(f"\nFound {len(results)} similar images:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.image_id} (score: {result.similarity_score:.4f})")
```

**What it demonstrates:**
- Query preparation and execution
- Two-stage search (ANN + reranking)
- Result interpretation and display
- Performance metrics collection

## Search Types Demonstrated

### 1. Image-to-Image Search

```python
def demo_image_search(index: AnnIndex, store: MetadataStore) -> None:
    """Demonstrate image-to-image search."""
    
    print("\n🖼️  Image-to-Image Search Demo")
    print("=" * 40)
    
    # Get sample image
    sample_metadata = store.get_all()[0]
    sample_histogram = sample_metadata["histogram"]
    
    # Perform search
    results = find_similar(
        sample_histogram,
        index,
        store,
        k=10,
        rerank_k=5
    )
    
    # Analyze results
    print(f"Query: {sample_metadata['file_path']}")
    print(f"Results: {len(results)} similar images found")
    
    for i, result in enumerate(results[:5]):
        print(f"  {i+1}. {result.image_id} (similarity: {result.similarity_score:.4f})")
```

### 2. Histogram-to-Image Search

```python
def demo_histogram_search(index: AnnIndex, store: MetadataStore) -> None:
    """Demonstrate histogram-to-image search."""
    
    print("\n📊 Histogram-to-Image Search Demo")
    print("=" * 40)
    
    # Create synthetic histogram (e.g., blue-dominated)
    synthetic_histogram = create_synthetic_histogram("blue")
    
    # Perform search
    results = find_similar(
        synthetic_histogram,
        index,
        store,
        k=8,
        rerank_k=4
    )
    
    print(f"Query: Synthetic blue histogram")
    print(f"Results: {len(results)} similar images found")
    
    # Show color characteristics
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. {result.image_id} (score: {result.similarity_score:.4f})")
```

### 3. Color Value Search

```python
def demo_color_search(index: AnnIndex, store: MetadataStore) -> None:
    """Demonstrate color value-based search."""
    
    print("\n🎨 Color Value Search Demo")
    print("=" * 40)
    
    # Search for specific color characteristics
    color_queries = [
        ("red", [50, 80, 50]),      # Medium brightness, high red
        ("green", [70, -80, 80]),   # Light, high green
        ("blue", [40, 50, -80])     # Dark, high blue
    ]
    
    for color_name, lab_values in color_queries:
        print(f"\nSearching for {color_name} (L*={lab_values[0]}, a*={lab_values[1]}, b*={lab_values[2]})")
        
        # Create histogram from color values
        color_histogram = create_color_histogram(lab_values)
        
        # Perform search
        results = find_similar(
            color_histogram,
            index,
            store,
            k=5,
            rerank_k=3
        )
        
        print(f"  Found {len(results)} similar images")
        for i, result in enumerate(results[:3]):
            print(f"    {i+1}. {result.image_id} (score: {result.similarity_score:.4f})")
```

## Reranking Demonstration

### Sinkhorn-EMD Reranking

The tool showcases the advanced reranking capabilities:

```python
def demonstrate_reranking(index: AnnIndex, store: MetadataStore) -> None:
    """Demonstrate reranking capabilities."""
    
    print("\n🔄 Reranking Demonstration")
    print("=" * 40)
    
    # Get sample query
    sample_metadata = store.get_all()[0]
    query_histogram = sample_metadata["histogram"]
    
    # Compare with and without reranking
    print("1. Initial ANN search results:")
    initial_results = find_similar(
        query_histogram,
        index,
        store,
        k=10,
        rerank_k=0  # No reranking
    )
    
    for i, result in enumerate(initial_results[:5]):
        print(f"   {i+1}. {result.image_id} (ANN score: {result.similarity_score:.4f})")
    
    print("\n2. After Sinkhorn-EMD reranking:")
    reranked_results = find_similar(
        query_histogram,
        index,
        store,
        k=10,
        rerank_k=5  # Rerank top 5
    )
    
    for i, result in enumerate(reranked_results[:5]):
        print(f"   {i+1}. {result.image_id} (reranked score: {result.similarity_score:.4f})")
    
    # Show improvement
    if len(initial_results) >= 2 and len(reranked_results) >= 2:
        initial_top = initial_results[1].image_id
        reranked_top = reranked_results[1].image_id
        
        if initial_top != reranked_top:
            print(f"\n✅ Reranking improved results: {initial_top} → {reranked_top}")
        else:
            print(f"\nℹ️  Reranking maintained top result: {initial_top}")
```

## Performance Analysis

### Benchmarking

The tool includes performance benchmarking:

```python
def benchmark_search_performance(index: AnnIndex, store: MetadataStore) -> None:
    """Benchmark search performance."""
    
    print("\n⚡ Performance Benchmarking")
    print("=" * 40)
    
    # Get multiple query images
    query_images = store.get_all()[:5]
    
    # Benchmark search times
    search_times = []
    rerank_times = []
    
    for query_metadata in query_images:
        query_histogram = query_metadata["histogram"]
        
        # Time search without reranking
        start_time = time.time()
        results = find_similar(
            query_histogram,
            index,
            store,
            k=20,
            rerank_k=0
        )
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        # Time reranking
        if len(results) > 0:
            start_time = time.time()
            reranked_results = find_similar(
                query_histogram,
                index,
                store,
                k=20,
                rerank_k=10
            )
            rerank_time = time.time() - start_time
            rerank_times.append(rerank_time)
    
    # Calculate statistics
    avg_search_time = np.mean(search_times) * 1000  # Convert to ms
    avg_rerank_time = np.mean(rerank_times) * 1000
    
    print(f"Average search time: {avg_search_time:.2f} ms")
    print(f"Average rerank time: {avg_rerank_time:.2f} ms")
    print(f"Total time per query: {avg_search_time + avg_rerank_time:.2f} ms")
```

## System Validation

### Search System Validation

The tool validates the complete search system:

```python
def validate_search_system(index: AnnIndex, store: MetadataStore) -> None:
    """Validate the search system functionality."""
    
    print("\n🔍 System Validation")
    print("=" * 40)
    
    # Check index integrity
    index_size = index.size()
    store_size = store.count()
    
    print(f"Index size: {index_size} vectors")
    print(f"Store size: {store_size} metadata entries")
    
    if index_size == store_size:
        print("✅ Index and store are synchronized")
    else:
        print("⚠️  Index and store sizes don't match")
    
    # Test basic search functionality
    if index_size > 0:
        sample_metadata = store.get_all()[0]
        query_histogram = sample_metadata["histogram"]
        
        try:
            results = find_similar(
                query_histogram,
                index,
                store,
                k=min(5, index_size),
                rerank_k=0
            )
            
            if len(results) > 0:
                print("✅ Basic search functionality working")
                print(f"   Found {len(results)} results for sample query")
            else:
                print("⚠️  Search returned no results")
                
        except Exception as e:
            print(f"❌ Search functionality error: {e}")
    else:
        print("⚠️  Cannot test search - index is empty")
```

## Integration Examples

### Using Demo as a Template

The demo can serve as a starting point for custom implementations:

```python
#!/usr/bin/env python3
"""
Custom search implementation based on demo_search.py
"""

from tools.demo_search import create_demo_index, demonstrate_search_pipeline
from chromatica.search import find_similar

def custom_search_workflow():
    """Custom search workflow."""
    
    # Create index from custom dataset
    index, store = create_demo_index("path/to/custom/dataset", max_images=100)
    
    # Custom search logic
    custom_query = create_custom_query_histogram()
    
    # Perform search with custom parameters
    results = find_similar(
        custom_query,
        index,
        store,
        k=15,
        rerank_k=8
    )
    
    # Custom result processing
    return process_custom_results(results)

def create_custom_query_histogram():
    """Create a custom query histogram."""
    # Your custom histogram generation logic
    pass

def process_custom_results(results):
    """Process search results in custom way."""
    # Your custom result processing logic
    pass
```

### Extending Demo Functionality

```python
def extended_search_demo():
    """Extended demo with additional features."""
    
    # Create index
    index, store = create_demo_index("datasets/test-dataset-50")
    
    # Additional demonstrations
    demonstrate_advanced_queries(index, store)
    demonstrate_batch_search(index, store)
    demonstrate_search_analytics(index, store)

def demonstrate_advanced_queries(index, store):
    """Demonstrate advanced query types."""
    print("\n🚀 Advanced Query Types")
    print("=" * 40)
    
    # Range queries
    # Fuzzy matching
    # Multi-image queries
    # etc.
```

## Error Handling

### Robust Error Handling

The demo includes comprehensive error handling:

```python
def safe_search_demo():
    """Safe search demonstration with error handling."""
    
    try:
        # Create index
        index, store = create_demo_index("datasets/test-dataset-50")
        
        # Perform searches
        demonstrate_search_pipeline(index, store)
        
    except FileNotFoundError as e:
        print(f"❌ Dataset not found: {e}")
        print("   Please ensure test datasets are available")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please check dependencies and virtual environment")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("   Check logs for detailed information")
```

## Best Practices

### Demo Development

1. **Comprehensive Coverage**: Demonstrate all major features
2. **Error Handling**: Include robust error handling
3. **Performance Metrics**: Show timing and efficiency data
4. **Clear Output**: Use descriptive messages and formatting
5. **Modular Design**: Separate concerns into distinct functions

### Demo Usage

1. **Virtual Environment**: Always activate `venv311\Scripts\activate`
2. **Dependencies**: Ensure all packages are installed
3. **Datasets**: Verify test datasets are available
4. **Path Resolution**: Run from project root directory
5. **Logging**: Use `--verbose` for detailed debugging

## Troubleshooting

### Common Issues

1. **Import Errors**: Check virtual environment and Python path
2. **Dataset Issues**: Verify test datasets exist and are accessible
3. **Memory Issues**: Reduce `max_images` for large datasets
4. **Performance Issues**: Check system resources and index size

### Debug Mode

```python
# Enable verbose logging
python tools/demo_search.py --verbose

# Check system status
python tools/demo_search.py --dataset datasets/test-dataset-20
```

## Dependencies

### Required Packages

- `numpy`: Numerical operations
- `faiss-cpu`: ANN index operations
- `duckdb`: Metadata storage
- `opencv-python`: Image processing
- `scikit-image`: Color space conversion

### Installation

```bash
# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The demo search tool provides a comprehensive demonstration of the Chromatica search system's capabilities and serves as both a learning resource and a reference implementation for developers.
