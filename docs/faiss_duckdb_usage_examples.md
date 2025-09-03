# FAISS and DuckDB Usage Examples

## Chromatica Color Search Engine

---

## Usage Examples

### Overview

This section provides comprehensive usage examples for the FAISS and DuckDB integration, from basic setup to advanced scenarios and integration patterns.

### Basic Setup and Initialization

#### 1. Simple Index and Store Creation

```python
from chromatica.indexing.store import AnnIndex, MetadataStore
from chromatica.utils.config import TOTAL_BINS
import numpy as np

# Initialize FAISS index
ann_index = AnnIndex(dimension=TOTAL_BINS)
print(f"Created FAISS index with dimension {TOTAL_BINS}")

# Initialize DuckDB store (in-memory for testing)
metadata_store = MetadataStore(db_path=":memory:")
print("Created in-memory DuckDB store")

# Verify both components are ready
print(f"FAISS index vectors: {ann_index.get_total_vectors()}")
print(f"DuckDB images: {metadata_store.get_image_count()}")
```

#### 2. Basic Histogram Indexing

```python
def index_single_image(image_path: str,
                      ann_index: AnnIndex,
                      metadata_store: MetadataStore) -> str:
    """
    Index a single image using the FAISS-DuckDB pipeline.

    Args:
        image_path: Path to the image file
        ann_index: FAISS index instance
        metadata_store: DuckDB metadata store instance

    Returns:
        Generated image ID
    """
    # Generate histogram (assuming process_image function exists)
    histogram = process_image(image_path)

    # Create unique image ID
    image_id = f"img_{hash(image_path) % 1000000:06d}"

    # Add to FAISS index (Hellinger transform applied automatically)
    ann_index.add(histogram.reshape(1, -1))

    # Store metadata and raw histogram in DuckDB
    metadata_record = {
        'image_id': image_id,
        'file_path': image_path,
        'histogram': histogram,
        'file_size': os.path.getsize(image_path)
    }

    metadata_store.add_batch([metadata_record])

    print(f"Indexed image {image_id}: {image_path}")
    return image_id

# Example usage
image_path = "path/to/image.jpg"
image_id = index_single_image(image_path, ann_index, metadata_store)
```

#### 3. Batch Image Indexing

```python
def index_image_batch(image_paths: List[str],
                     ann_index: AnnIndex,
                     metadata_store: MetadataStore,
                     batch_size: int = 50) -> int:
    """
    Index multiple images in batches for optimal performance.

    Args:
        image_paths: List of image file paths
        ann_index: FAISS index instance
        metadata_store: DuckDB metadata store instance
        batch_size: Number of images to process in each batch

    Returns:
        Total number of successfully indexed images
    """
    total_indexed = 0

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(image_paths) + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} images)")

        # Process batch
        batch_histograms = []
        batch_metadata = []

        for image_path in batch_paths:
            try:
                histogram = process_image(image_path)
                image_id = f"img_{hash(image_path) % 1000000:06d}"

                batch_histograms.append(histogram.reshape(1, -1))
                batch_metadata.append({
                    'image_id': image_id,
                    'file_path': image_path,
                    'histogram': histogram,
                    'file_size': os.path.getsize(image_path)
                })

            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                continue

        if batch_histograms:
            # Batch operations
            histograms_array = np.vstack(batch_histograms)
            ann_index.add(histograms_array)
            metadata_store.add_batch(batch_metadata)

            total_indexed += len(batch_histograms)
            print(f"Batch {batch_num} completed: {len(batch_histograms)} images indexed")

    return total_indexed

# Example usage
image_paths = [
    "path/to/image1.jpg",
    "path/to/image2.png",
    "path/to/image3.jpg",
    # ... more images
]

total_indexed = index_image_batch(image_paths, ann_index, metadata_store, batch_size=25)
print(f"Total images indexed: {total_indexed}")
```

### Search Operations

#### 1. Basic Similarity Search

```python
def search_similar_images(query_histogram: np.ndarray,
                         ann_index: AnnIndex,
                         metadata_store: MetadataStore,
                         k: int = 20) -> List[Dict[str, Any]]:
    """
    Perform basic similarity search using FAISS and DuckDB.

    Args:
        query_histogram: Query image histogram
        ann_index: FAISS index instance
        metadata_store: DuckDB metadata store instance
        k: Number of similar images to retrieve

    Returns:
        List of similar images with metadata
    """
    # Stage 1: FAISS ANN Search
    print(f"Stage 1: FAISS ANN search for {k} candidates")
    distances, indices = ann_index.search(query_histogram, k)

    # Stage 2: Retrieve metadata and histograms
    print(f"Stage 2: Retrieving metadata for {len(indices[0])} candidates")

    # Convert FAISS indices to image IDs (simplified mapping)
    candidate_ids = [f"img_{idx:06d}" for idx in indices[0]]

    # Get raw histograms for reranking
    candidate_histograms = metadata_store.get_histograms_by_ids(candidate_ids)

    # Compile results
    results = []
    for i, (image_id, histogram) in enumerate(candidate_histograms.items()):
        results.append({
            'image_id': image_id,
            'rank': i + 1,
            'faiss_distance': float(distances[0][i]),
            'histogram': histogram,
            'similarity_score': 1.0 / (1.0 + float(distances[0][i]))  # Convert distance to similarity
        })

    # Sort by similarity score (descending)
    results.sort(key=lambda x: x['similarity_score'], reverse=True)

    return results

# Example usage
query_image_path = "path/to/query_image.jpg"
query_histogram = process_image(query_image_path)

similar_images = search_similar_images(query_histogram, ann_index, metadata_store, k=10)

print(f"Found {len(similar_images)} similar images:")
for result in similar_images:
    print(f"  {result['rank']}. {result['image_id']} (similarity: {result['similarity_score']:.4f})")
```

#### 2. Advanced Search with Filtering

```python
def search_with_filters(query_histogram: np.ndarray,
                       ann_index: AnnIndex,
                       metadata_store: MetadataStore,
                       k: int = 100,
                       min_file_size: int = 0,
                       max_file_size: int = None,
                       file_extensions: List[str] = None) -> List[Dict[str, Any]]:
    """
    Perform similarity search with additional filtering criteria.

    Args:
        query_histogram: Query image histogram
        ann_index: FAISS index instance
        metadata_store: DuckDB metadata store instance
        k: Number of candidates to retrieve from FAISS
        min_file_size: Minimum file size in bytes
        max_file_size: Maximum file size in bytes
        file_extensions: Allowed file extensions (e.g., ['.jpg', '.png'])

    Returns:
        Filtered list of similar images
    """
    # Stage 1: FAISS search for more candidates (we'll filter later)
    search_k = min(k * 3, ann_index.get_total_vectors())  # Get more candidates for filtering
    distances, indices = ann_index.search(query_histogram, search_k)

    # Stage 2: Retrieve metadata for filtering
    candidate_ids = [f"img_{idx:06d}" for idx in indices[0]]
    candidate_histograms = metadata_store.get_histograms_by_ids(candidate_ids)

    # Apply filters
    filtered_results = []
    for i, (image_id, histogram) in enumerate(candidate_histograms.items()):
        # Get additional metadata for filtering
        metadata = metadata_store.get_metadata_by_id(image_id)

        # File size filter
        if metadata['file_size'] < min_file_size:
            continue
        if max_file_size and metadata['file_size'] > max_file_size:
            continue

        # File extension filter
        if file_extensions:
            file_ext = os.path.splitext(metadata['file_path'])[1].lower()
            if file_ext not in file_extensions:
                continue

        # Add to filtered results
        filtered_results.append({
            'image_id': image_id,
            'rank': len(filtered_results) + 1,
            'faiss_distance': float(distances[0][i]),
            'histogram': histogram,
            'file_path': metadata['file_path'],
            'file_size': metadata['file_size'],
            'similarity_score': 1.0 / (1.0 + float(distances[0][i]))
        })

        # Stop if we have enough results
        if len(filtered_results) >= k:
            break

    # Sort by similarity score
    filtered_results.sort(key=lambda x: x['similarity_score'], reverse=True)

    return filtered_results

# Example usage with filters
similar_images = search_with_filters(
    query_histogram=query_histogram,
    ann_index=ann_index,
    metadata_store=metadata_store,
    k=10,
    min_file_size=1024,  # At least 1KB
    max_file_size=10*1024*1024,  # At most 10MB
    file_extensions=['.jpg', '.png']  # Only JPG and PNG files
)

print(f"Found {len(similar_images)} similar images after filtering:")
for result in similar_images:
    print(f"  {result['rank']}. {result['image_id']} "
          f"({result['file_path']}, {result['file_size']} bytes)")
```

### Integration with Image Processing Pipeline

#### 1. Complete End-to-End Workflow

```python
def complete_image_search_workflow(query_image_path: str,
                                 image_directory: str,
                                 output_dir: str = "search_results") -> Dict[str, Any]:
    """
    Complete workflow: index images and perform similarity search.

    This function demonstrates the full pipeline:
    1. Index all images in a directory
    2. Perform similarity search
    3. Save results and visualizations
    """
    import os
    from pathlib import Path

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    ann_index = AnnIndex(dimension=TOTAL_BINS)
    metadata_store = MetadataStore(db_path=":memory:")

    # Step 1: Index all images in directory
    print("Step 1: Indexing images...")
    image_paths = list(Path(image_directory).glob("*.jpg")) + \
                  list(Path(image_directory).glob("*.png"))

    total_indexed = index_image_batch(
        [str(p) for p in image_paths],
        ann_index,
        metadata_store,
        batch_size=20
    )

    print(f"Indexed {total_indexed} images")

    # Step 2: Process query image
    print("Step 2: Processing query image...")
    query_histogram = process_image(query_image_path)

    # Step 3: Perform similarity search
    print("Step 3: Performing similarity search...")
    similar_images = search_similar_images(
        query_histogram,
        ann_index,
        metadata_store,
        k=min(10, total_indexed)
    )

    # Step 4: Save results
    print("Step 4: Saving results...")
    results_file = os.path.join(output_dir, "search_results.json")

    # Prepare results for JSON serialization
    serializable_results = []
    for result in similar_images:
        serializable_results.append({
            'image_id': result['image_id'],
            'rank': result['rank'],
            'similarity_score': float(result['similarity_score']),
            'file_path': result.get('file_path', ''),
            'file_size': result.get('file_size', 0)
        })

    with open(results_file, 'w') as f:
        json.dump({
            'query_image': query_image_path,
            'total_indexed': total_indexed,
            'search_results': serializable_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"Results saved to {results_file}")

    # Step 5: Generate visualizations (if matplotlib is available)
    try:
        generate_search_visualization(query_image_path, similar_images, output_dir)
        print("Visualizations generated")
    except ImportError:
        print("Matplotlib not available, skipping visualizations")

    return {
        'total_indexed': total_indexed,
        'search_results': similar_images,
        'results_file': results_file
    }

# Example usage
results = complete_image_search_workflow(
    query_image_path="path/to/query.jpg",
    image_directory="path/to/image_collection",
    output_dir="my_search_results"
)

print(f"Workflow completed successfully!")
print(f"Indexed {results['total_indexed']} images")
print(f"Found {len(results['search_results'])} similar images")
```

#### 2. Interactive Search Interface

```python
def interactive_search_interface(ann_index: AnnIndex,
                               metadata_store: MetadataStore) -> None:
    """
    Interactive command-line interface for image search.

    This function provides a simple interactive way to:
    - Search for similar images
    - View search results
    - Explore the indexed dataset
    """
    print("=== Chromatica Interactive Search Interface ===")
    print("Commands:")
    print("  search <image_path> [k] - Search for similar images")
    print("  info                    - Show index information")
    print("  quit                    - Exit the interface")
    print()

    while True:
        try:
            command = input("chromatica> ").strip().split()

            if not command:
                continue

            if command[0] == "quit":
                print("Goodbye!")
                break

            elif command[0] == "info":
                print(f"FAISS index: {ann_index.get_total_vectors()} vectors")
                print(f"DuckDB store: {metadata_store.get_image_count()} images")
                print(f"Index dimension: {ann_index.dimension}")

            elif command[0] == "search":
                if len(command) < 2:
                    print("Usage: search <image_path> [k]")
                    continue

                image_path = command[1]
                k = int(command[2]) if len(command) > 2 else 10

                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                print(f"Searching for images similar to {image_path}...")

                # Process query image
                query_histogram = process_image(image_path)

                # Perform search
                similar_images = search_similar_images(
                    query_histogram, ann_index, metadata_store, k
                )

                # Display results
                print(f"\nFound {len(similar_images)} similar images:")
                for result in similar_images:
                    print(f"  {result['rank']:2d}. {result['image_id']} "
                          f"(similarity: {result['similarity_score']:.4f})")

            else:
                print(f"Unknown command: {command[0]}")
                print("Available commands: search, info, quit")

        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")

# Example usage
# interactive_search_interface(ann_index, metadata_store)
```

### Performance Monitoring and Optimization

#### 1. Search Performance Profiling

```python
import time
import statistics
from typing import List, Tuple

class SearchProfiler:
    """Profile and monitor search performance."""

    def __init__(self):
        self.search_times = []
        self.ann_times = []
        self.retrieval_times = []
        self.candidate_counts = []

    def profile_search(self, search_func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Profile a search operation and return results with timing information.

        Args:
            search_func: Function to profile
            *args, **kwargs: Arguments for the search function

        Returns:
            Tuple of (search_results, timing_info)
        """
        # Profile overall search time
        start_time = time.perf_counter()
        results = search_func(*args, **kwargs)
        total_time = time.perf_counter() - start_time

        # Profile individual components (if available)
        ann_time = getattr(results, 'ann_time', 0)
        retrieval_time = getattr(results, 'retrieval_time', 0)

        # Record metrics
        self.search_times.append(total_time)
        self.ann_times.append(ann_time)
        self.retrieval_times.append(retrieval_time)
        self.candidate_counts.append(len(results))

        timing_info = {
            'total_time': total_time,
            'ann_time': ann_time,
            'retrieval_time': retrieval_time,
            'candidate_count': len(results)
        }

        return results, timing_info

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.search_times:
            return {"error": "No search operations recorded"}

        return {
            'total_searches': len(self.search_times),
            'timing': {
                'total_time': {
                    'mean': statistics.mean(self.search_times),
                    'median': statistics.median(self.search_times),
                    'min': min(self.search_times),
                    'max': max(self.search_times),
                    'std': statistics.stdev(self.search_times) if len(self.search_times) > 1 else 0
                },
                'ann_time': {
                    'mean': statistics.mean(self.ann_times) if self.ann_times else 0,
                    'median': statistics.median(self.ann_times) if self.ann_times else 0
                },
                'retrieval_time': {
                    'mean': statistics.mean(self.retrieval_times) if self.retrieval_times else 0,
                    'median': statistics.median(self.retrieval_times) if self.retrieval_times else 0
                }
            },
            'candidates': {
                'mean': statistics.mean(self.candidate_counts),
                'min': min(self.candidate_counts),
                'max': max(self.candidate_counts)
            },
            'throughput': {
                'searches_per_second': 1.0 / statistics.mean(self.search_times),
                'candidates_per_second': statistics.mean(self.candidate_counts) / statistics.mean(self.search_times)
            }
        }

    def print_performance_report(self):
        """Print a formatted performance report."""
        summary = self.get_performance_summary()

        if 'error' in summary:
            print(f"Error: {summary['error']}")
            return

        print("=== Search Performance Report ===")
        print(f"Total searches: {summary['total_searches']}")
        print()

        print("Timing (seconds):")
        timing = summary['timing']
        print(f"  Total time: {timing['total_time']['mean']:.4f} ± {timing['total_time']['std']:.4f}")
        print(f"  ANN search:  {timing['ann_time']['mean']:.4f}")
        print(f"  Retrieval:   {timing['retrieval_time']['mean']:.4f}")
        print()

        print("Throughput:")
        throughput = summary['throughput']
        print(f"  Searches/sec: {throughput['searches_per_second']:.2f}")
        print(f"  Candidates/sec: {throughput['candidates_per_second']:.2f}")
        print()

        print("Candidates:")
        candidates = summary['candidates']
        print(f"  Average: {candidates['mean']:.1f}")
        print(f"  Range: {candidates['min']} - {candidates['max']}")

# Example usage
profiler = SearchProfiler()

# Profile multiple searches
for i in range(5):
    results, timing = profiler.profile_search(
        search_similar_images,
        query_histogram,
        ann_index,
        metadata_store,
        k=20
    )
    print(f"Search {i+1}: {timing['total_time']:.4f}s")

# Print performance report
profiler.print_performance_report()
```

#### 2. Memory Usage Monitoring

```python
import psutil
import os
from typing import Dict, Any

class MemoryMonitor:
    """Monitor memory usage during operations."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.memory_snapshots = []

    def take_snapshot(self, operation: str) -> Dict[str, Any]:
        """
        Take a memory snapshot for a specific operation.

        Args:
            operation: Description of the operation being monitored

        Returns:
            Memory usage information
        """
        memory_info = self.process.memory_info()

        snapshot = {
            'operation': operation,
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': self.process.memory_percent()
        }

        self.memory_snapshots.append(snapshot)
        return snapshot

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_snapshots:
            return {"error": "No memory snapshots recorded"}

        rss_values = [s['rss_mb'] for s in self.memory_snapshots]
        vms_values = [s['vms_mb'] for s in self.memory_snapshots]

        return {
            'total_snapshots': len(self.memory_snapshots),
            'rss_memory': {
                'current_mb': rss_values[-1],
                'peak_mb': max(rss_values),
                'average_mb': statistics.mean(rss_values),
                'growth_mb': rss_values[-1] - rss_values[0]
            },
            'virtual_memory': {
                'current_mb': vms_values[-1],
                'peak_mb': max(vms_values),
                'average_mb': statistics.mean(vms_values)
            },
            'operations': [s['operation'] for s in self.memory_snapshots]
        }

    def print_memory_report(self):
        """Print a formatted memory usage report."""
        summary = self.get_memory_summary()

        if 'error' in summary:
            print(f"Error: {summary['error']}")
            return

        print("=== Memory Usage Report ===")
        print(f"Total snapshots: {summary['total_snapshots']}")
        print()

        print("Resident Set Size (RSS):")
        rss = summary['rss_memory']
        print(f"  Current: {rss['current_mb']:.1f} MB")
        print(f"  Peak: {rss['peak_mb']:.1f} MB")
        print(f"  Average: {rss['average_mb']:.1f} MB")
        print(f"  Growth: {rss['growth_mb']:+.1f} MB")
        print()

        print("Virtual Memory:")
        vms = summary['virtual_memory']
        print(f"  Current: {vms['current_mb']:.1f} MB")
        print(f"  Peak: {vms['peak_mb']:.1f} MB")
        print(f"  Average: {vms['average_mb']:.1f} MB")
        print()

        print("Operations monitored:")
        for op in summary['operations']:
            print(f"  - {op}")

# Example usage
memory_monitor = MemoryMonitor()

# Monitor memory during indexing
memory_monitor.take_snapshot("Before indexing")
index_image_batch(image_paths, ann_index, metadata_store, batch_size=50)
memory_monitor.take_snapshot("After indexing")

# Monitor memory during search
memory_monitor.take_snapshot("Before search")
results = search_similar_images(query_histogram, ann_index, metadata_store, k=20)
memory_monitor.take_snapshot("After search")

# Print memory report
memory_monitor.print_memory_report()
```

### Advanced Integration Patterns

#### 1. Persistent Storage and Recovery

```python
def save_search_index(ann_index: AnnIndex,
                     metadata_store: MetadataStore,
                     base_path: str) -> None:
    """
    Save the complete search index to disk for persistence.

    Args:
        ann_index: FAISS index instance
        metadata_store: DuckDB metadata store instance
        base_path: Base directory for saving files
    """
    import os
    from datetime import datetime

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_path, f"search_index_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Save FAISS index
    faiss_path = os.path.join(save_dir, "faiss_index.bin")
    ann_index.save(faiss_path)
    print(f"FAISS index saved to {faiss_path}")

    # Save DuckDB database
    db_path = os.path.join(save_dir, "metadata.db")
    metadata_store.save_database(db_path)
    print(f"DuckDB database saved to {db_path}")

    # Save configuration and metadata
    config_path = os.path.join(save_dir, "index_config.json")
    config = {
        'timestamp': timestamp,
        'faiss_index_path': faiss_path,
        'duckdb_path': db_path,
        'total_vectors': ann_index.get_total_vectors(),
        'total_images': metadata_store.get_image_count(),
        'dimension': ann_index.dimension
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {config_path}")
    print(f"Complete index saved to {save_dir}")

def load_search_index(base_path: str) -> Tuple[AnnIndex, MetadataStore]:
    """
    Load a previously saved search index from disk.

    Args:
        base_path: Directory containing the saved index

    Returns:
        Tuple of (AnnIndex, MetadataStore) instances
    """
    # Find the most recent index
    index_dirs = [d for d in os.listdir(base_path) if d.startswith("search_index_")]
    if not index_dirs:
        raise FileNotFoundError(f"No search index found in {base_path}")

    # Sort by timestamp and get the most recent
    latest_dir = sorted(index_dirs)[-1]
    index_dir = os.path.join(base_path, latest_dir)

    # Load configuration
    config_path = os.path.join(index_dir, "index_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loading search index from {index_dir}")
    print(f"Index created: {config['timestamp']}")
    print(f"Total vectors: {config['total_vectors']}")
    print(f"Total images: {config['total_images']}")

    # Load FAISS index
    ann_index = AnnIndex(dimension=config['dimension'])
    ann_index.load(config['faiss_index_path'])

    # Load DuckDB database
    metadata_store = MetadataStore(db_path=config['duckdb_path'])

    return ann_index, metadata_store

# Example usage
# Save index
save_search_index(ann_index, metadata_store, "saved_indexes")

# Load index later
ann_index, metadata_store = load_search_index("saved_indexes")
```

#### 2. Incremental Indexing

```python
class IncrementalIndexer:
    """Manage incremental updates to the search index."""

    def __init__(self, ann_index: AnnIndex, metadata_store: MetadataStore):
        self.ann_index = ann_index
        self.metadata_store = metadata_store
        self.indexed_files = set()
        self.load_indexed_files()

    def load_indexed_files(self):
        """Load list of already indexed files."""
        # This is a simplified version - in practice, you'd store this in DuckDB
        # or maintain a separate index file
        pass

    def add_new_images(self, image_paths: List[str], batch_size: int = 50) -> int:
        """
        Add only new images to the index.

        Args:
            image_paths: List of image paths to check and index
            batch_size: Batch size for processing

        Returns:
            Number of newly indexed images
        """
        new_images = [path for path in image_paths if path not in self.indexed_files]

        if not new_images:
            print("No new images to index")
            return 0

        print(f"Found {len(new_images)} new images to index")

        # Index new images
        total_indexed = index_image_batch(new_images, self.ann_index, self.metadata_store, batch_size)

        # Update indexed files set
        self.indexed_files.update(new_images)

        print(f"Successfully indexed {total_indexed} new images")
        return total_indexed

    def remove_images(self, image_paths: List[str]) -> int:
        """
        Remove images from the index.

        Note: This is a simplified implementation. In practice, you'd need to:
        1. Remove from FAISS index (requires rebuilding)
        2. Remove from DuckDB store
        3. Update the indexed files set

        Args:
            image_paths: List of image paths to remove

        Returns:
            Number of removed images
        """
        removed_count = 0

        for image_path in image_paths:
            if image_path in self.indexed_files:
                # Remove from DuckDB (simplified)
                image_id = f"img_{hash(image_path) % 1000000:06d}"
                # metadata_store.remove_image(image_id)  # Implement this method

                self.indexed_files.remove(image_path)
                removed_count += 1

        print(f"Removed {removed_count} images from index")
        print("Note: FAISS index requires rebuilding for complete removal")

        return removed_count

    def get_index_status(self) -> Dict[str, Any]:
        """Get current index status."""
        return {
            'total_indexed_files': len(self.indexed_files),
            'faiss_vectors': self.ann_index.get_total_vectors(),
            'duckdb_images': self.metadata_store.get_image_count(),
            'indexed_files': list(self.indexed_files)
        }

# Example usage
incremental_indexer = IncrementalIndexer(ann_index, metadata_store)

# Add new images
new_image_paths = [
    "path/to/new_image1.jpg",
    "path/to/new_image2.png"
]

newly_indexed = incremental_indexer.add_new_images(new_image_paths)
print(f"Added {newly_indexed} new images to index")

# Check status
status = incremental_indexer.get_index_status()
print(f"Index status: {status}")
```

### Error Handling and Debugging

#### 1. Comprehensive Error Handling

```python
def robust_search_with_fallback(query_histogram: np.ndarray,
                               ann_index: AnnIndex,
                               metadata_store: MetadataStore,
                               k: int = 20) -> List[Dict[str, Any]]:
    """
    Perform search with comprehensive error handling and fallback mechanisms.

    Args:
        query_histogram: Query image histogram
        ann_index: FAISS index instance
        metadata_store: DuckDB metadata store instance
        k: Number of similar images to retrieve

    Returns:
        Search results with error information
    """
    try:
        # Stage 1: FAISS Search
        try:
            distances, indices = ann_index.search(query_histogram, k)
            ann_success = True
            ann_error = None
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            ann_success = False
            ann_error = str(e)
            # Fallback: return empty results
            return []

        # Stage 2: Histogram Retrieval
        try:
            candidate_ids = [f"img_{idx:06d}" for idx in indices[0]]
            candidate_histograms = metadata_store.get_histograms_by_ids(candidate_ids)
            retrieval_success = True
            retrieval_error = None
        except Exception as e:
            logger.error(f"Histogram retrieval failed: {e}")
            retrieval_success = False
            retrieval_error = str(e)
            # Fallback: return FAISS results without histograms
            return [{'image_id': f"img_{idx:06d}", 'rank': i+1, 'faiss_distance': float(distances[0][i])}
                   for i, idx in enumerate(indices[0])]

        # Success: return complete results
        results = []
        for i, (image_id, histogram) in enumerate(candidate_histograms.items()):
            results.append({
                'image_id': image_id,
                'rank': i + 1,
                'faiss_distance': float(distances[0][i]),
                'histogram': histogram,
                'similarity_score': 1.0 / (1.0 + float(distances[0][i]))
            })

        # Add operation status
        for result in results:
            result['operation_status'] = {
                'ann_search': ann_success,
                'retrieval': retrieval_success,
                'ann_error': ann_error,
                'retrieval_error': retrieval_error
            }

        return results

    except Exception as e:
        logger.error(f"Unexpected error in search pipeline: {e}")
        return []

# Example usage with error handling
try:
    results = robust_search_with_fallback(query_histogram, ann_index, metadata_store, k=10)

    if results:
        print(f"Search successful: {len(results)} results")

        # Check for partial failures
        for result in results:
            status = result['operation_status']
            if not status['ann_search']:
                print(f"Warning: ANN search failed for {result['image_id']}")
            if not status['retrieval']:
                print(f"Warning: Histogram retrieval failed for {result['image_id']}")
    else:
        print("Search failed - no results returned")

except Exception as e:
    print(f"Search pipeline failed: {e}")
```

#### 2. Debug Mode and Logging

```python
import logging
from typing import Optional

class DebugSearchInterface:
    """Enhanced search interface with debugging capabilities."""

    def __init__(self, ann_index: AnnIndex, metadata_store: MetadataStore,
                 debug_level: str = "INFO"):
        self.ann_index = ann_index
        self.metadata_store = metadata_store

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, debug_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Debug counters
        self.search_count = 0
        self.error_count = 0
        self.performance_metrics = []

    def debug_search(self, query_histogram: np.ndarray, k: int = 20,
                    verbose: bool = False) -> Dict[str, Any]:
        """
        Perform search with detailed debugging information.

        Args:
            query_histogram: Query image histogram
            k: Number of similar images to retrieve
            verbose: Enable verbose debugging output

        Returns:
            Dictionary containing results and debug information
        """
        self.search_count += 1
        start_time = time.perf_counter()

        debug_info = {
            'search_id': self.search_count,
            'timestamp': time.time(),
            'query_shape': query_histogram.shape,
            'query_sum': float(query_histogram.sum()),
            'k_requested': k,
            'stages': {}
        }

        try:
            # Stage 1: FAISS Search
            self.logger.info(f"Starting FAISS search (k={k})")
            stage1_start = time.perf_counter()

            distances, indices = self.ann_index.search(query_histogram, k)

            stage1_time = time.perf_counter() - stage1_start
            debug_info['stages']['faiss_search'] = {
                'success': True,
                'time_seconds': stage1_time,
                'candidates_found': len(indices[0]),
                'distance_range': [float(distances[0].min()), float(distances[0].max())],
                'indices_range': [int(indices[0].min()), int(indices[0].max())]
            }

            self.logger.info(f"FAISS search completed in {stage1_time:.4f}s")

            # Stage 2: Histogram Retrieval
            self.logger.info("Starting histogram retrieval")
            stage2_start = time.perf_counter()

            candidate_ids = [f"img_{idx:06d}" for idx in indices[0]]
            candidate_histograms = self.metadata_store.get_histograms_by_ids(candidate_ids)

            stage2_time = time.perf_counter() - stage2_start
            debug_info['stages']['histogram_retrieval'] = {
                'success': True,
                'time_seconds': stage2_time,
                'requested_ids': len(candidate_ids),
                'retrieved_histograms': len(candidate_histograms),
                'retrieval_rate': len(candidate_histograms) / len(candidate_ids)
            }

            self.logger.info(f"Histogram retrieval completed in {stage2_time:.4f}s")

            # Compile results
            results = []
            for i, (image_id, histogram) in enumerate(candidate_histograms.items()):
                results.append({
                    'image_id': image_id,
                    'rank': i + 1,
                    'faiss_distance': float(distances[0][i]),
                    'histogram': histogram,
                    'similarity_score': 1.0 / (1.0 + float(distances[0][i]))
                })

            # Sort by similarity score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)

            # Record performance metrics
            total_time = time.perf_counter() - start_time
            self.performance_metrics.append({
                'search_id': self.search_count,
                'total_time': total_time,
                'faiss_time': stage1_time,
                'retrieval_time': stage2_time,
                'candidates_found': len(results)
            })

            debug_info['overall'] = {
                'success': True,
                'total_time_seconds': total_time,
                'results_count': len(results),
                'performance_breakdown': {
                    'faiss_percentage': (stage1_time / total_time) * 100,
                    'retrieval_percentage': (stage2_time / total_time) * 100
                }
            }

            self.logger.info(f"Search completed successfully in {total_time:.4f}s")

            return {
                'results': results,
                'debug_info': debug_info
            }

        except Exception as e:
            self.error_count += 1
            total_time = time.perf_counter() - start_time

            debug_info['overall'] = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'total_time_seconds': total_time
            }

            self.logger.error(f"Search failed after {total_time:.4f}s: {e}")

            return {
                'results': [],
                'debug_info': debug_info
            }

    def get_debug_summary(self) -> Dict[str, Any]:
        """Get comprehensive debugging summary."""
        return {
            'search_statistics': {
                'total_searches': self.search_count,
                'successful_searches': len([m for m in self.performance_metrics if m]),
                'error_count': self.error_count,
                'success_rate': (self.search_count - self.error_count) / max(1, self.search_count)
            },
            'performance_analysis': {
                'total_searches': len(self.performance_metrics),
                'average_total_time': statistics.mean([m['total_time'] for m in self.performance_metrics]) if self.performance_metrics else 0,
                'average_faiss_time': statistics.mean([m['faiss_time'] for m in self.performance_metrics]) if self.performance_metrics else 0,
                'average_retrieval_time': statistics.mean([m['retrieval_time'] for m in self.performance_metrics]) if self.performance_metrics else 0,
                'average_candidates': statistics.mean([m['candidates_found'] for m in self.performance_metrics]) if self.performance_metrics else 0
            },
            'system_status': {
                'faiss_vectors': self.ann_index.get_total_vectors(),
                'duckdb_images': self.metadata_store.get_image_count(),
                'faiss_dimension': self.ann_index.dimension
            }
        }

    def print_debug_report(self):
        """Print a formatted debugging report."""
        summary = self.get_debug_summary()

        print("=== Debug Search Report ===")
        print()

        print("Search Statistics:")
        stats = summary['search_statistics']
        print(f"  Total searches: {stats['total_searches']}")
        print(f"  Successful: {stats['successful_searches']}")
        print(f"  Errors: {stats['error_count']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print()

        print("Performance Analysis:")
        perf = summary['performance_analysis']
        print(f"  Average total time: {perf['average_total_time']:.4f}s")
        print(f"  Average FAISS time: {perf['average_faiss_time']:.4f}s")
        print(f"  Average retrieval time: {perf['average_retrieval_time']:.4f}s")
        print(f"  Average candidates: {perf['average_candidates']:.1f}")
        print()

        print("System Status:")
        sys_status = summary['system_status']
        print(f"  FAISS vectors: {sys_status['faiss_vectors']}")
        print(f"  DuckDB images: {sys_status['duckdb_images']}")
        print(f"  FAISS dimension: {sys_status['faiss_dimension']}")

# Example usage
debug_interface = DebugSearchInterface(ann_index, metadata_store, debug_level="DEBUG")

# Perform debug search
search_result = debug_interface.debug_search(query_histogram, k=15, verbose=True)

# Print debug report
debug_interface.print_debug_report()
```

This comprehensive usage examples section demonstrates:

1. **Basic Setup**: Simple initialization and configuration
2. **Core Operations**: Indexing, searching, and batch processing
3. **Integration Patterns**: End-to-end workflows and interactive interfaces
4. **Performance Monitoring**: Profiling, metrics, and optimization
5. **Advanced Features**: Persistent storage, incremental indexing, and error handling
6. **Debugging Tools**: Comprehensive logging and debugging interfaces

Each example includes practical code that can be adapted and extended for different use cases in the Chromatica color search engine.
