#!/usr/bin/env python3
"""
Demonstration script for the Chromatica search system.

This script showcases the complete two-stage search pipeline by:
1. Creating a test index with sample images
2. Performing searches with different query types
3. Demonstrating the reranking capabilities
4. Showing performance characteristics

Usage:
    python tools/demo_search.py [--dataset path] [--verbose]

Requirements:
    - Virtual environment activated (venv311\Scripts\activate)
    - All dependencies installed (pip install -r requirements.txt)
    - Test datasets available in datasets/ directory
"""

import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.core.histogram import generate_histogram
from chromatica.indexing.store import AnnIndex, MetadataStore
from chromatica.search import find_similar, validate_search_system, SearchResult


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_demo_index(dataset_path: str, max_images: int = 20) -> tuple[AnnIndex, MetadataStore]:
    """Create a demo index with sample images."""
    print(f"🔧 Creating demo index from {dataset_path}")
    print(f"   Processing up to {max_images} images...")
    
    # Initialize components
    index = AnnIndex()
    store = MetadataStore(":memory:")  # Use in-memory database for demo
    
    # Get list of image files
    dataset_path = Path(dataset_path)
    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    image_files = image_files[:max_images]
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {dataset_path}")
    
    # Process images and build index
    histograms = []
    metadata_batch = []
    
    for i, image_file in enumerate(image_files):
        try:
            # Generate histogram
            histogram = generate_histogram(str(image_file))
            
            # Validate histogram
            if histogram.shape != (1152,) or not np.isclose(histogram.sum(), 1.0, atol=1e-6):
                print(f"   ⚠️  Skipping {image_file.name} (invalid histogram)")
                continue
            
            # Prepare metadata
            image_id = f"demo_{image_file.stem}"
            metadata = {
                "image_id": image_id,
                "file_path": str(image_file),
                "histogram": histogram,
                "file_size": image_file.stat().st_size
            }
            
            histograms.append(histogram)
            metadata_batch.append(metadata)
            
            print(f"   ✅ Processed {image_file.name}")
                
        except Exception as e:
            print(f"   ❌ Failed to process {image_file.name}: {e}")
            continue
    
    if not histograms:
        raise RuntimeError("No valid histograms generated from demo dataset")
    
    # Add to index and store
    histograms_array = np.array(histograms, dtype=np.float64)
    
    added_count = index.add(histograms_array)
    stored_count = store.add_batch(metadata_batch)
    
    print(f"   📊 Index created: {added_count} vectors in FAISS, {stored_count} records in store")
    print()
    
    return index, store


def demo_synthetic_search(index: AnnIndex, store: MetadataStore):
    """Demonstrate search with synthetic query histograms."""
    print("🔍 Demo 1: Synthetic Query Search")
    print("=" * 50)
    
    # Create synthetic query histogram
    np.random.seed(42)
    query_hist = np.random.random(1152).astype(np.float64)
    query_hist = query_hist / query_hist.sum()
    
    print(f"Query histogram: shape={query_hist.shape}, sum={query_hist.sum():.6f}")
    print()
    
    # Perform search
    start_time = time.time()
    results = find_similar(query_hist, index, store, k=10, max_rerank=5)
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.3f}s")
    print(f"Results returned: {len(results)}")
    print()
    
    # Display results
    print("Top 5 Results:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Image ID':<20} {'Distance':<12} {'ANN Score':<12} {'File Path':<30}")
    print("-" * 80)
    
    for result in results[:5]:
        file_path = Path(result.file_path).name if result.file_path != "unknown" else "unknown"
        print(f"{result.rank:<4} {result.image_id:<20} {result.distance:<12.6f} {result.ann_score:<12.6f} {file_path:<30}")
    
    print()


def demo_real_image_search(index: AnnIndex, store: MetadataStore, dataset_path: str):
    """Demonstrate search with a real image from the dataset."""
    print("🔍 Demo 2: Real Image Query Search")
    print("=" * 50)
    
    # Find a test image
    dataset_path = Path(dataset_path)
    test_images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png")
    
    if not test_images:
        print("   ⚠️  No test images found for real image search")
        return
    
    test_image = test_images[0]
    print(f"Query image: {test_image.name}")
    
    # Generate histogram from real image
    query_hist = generate_histogram(str(test_image))
    print(f"Query histogram: shape={query_hist.shape}, sum={query_hist.sum():.6f}")
    print()
    
    # Perform search
    start_time = time.time()
    results = find_similar(query_hist, index, store, k=15, max_rerank=10)
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.3f}s")
    print(f"Results returned: {len(results)}")
    print()
    
    # Display results
    print("Top 10 Results:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Image ID':<20} {'Distance':<12} {'ANN Score':<12} {'File Path':<40}")
    print("-" * 100)
    
    for result in results[:10]:
        file_path = Path(result.file_path).name if result.file_path != "unknown" else "unknown"
        print(f"{result.rank:<4} {result.image_id:<20} {result.distance:<12.6f} {result.ann_score:<12.6f} {file_path:<40}")
    
    # Check if query image is in top results
    query_image_id = f"demo_{test_image.stem}"
    found_query = any(r.image_id == query_image_id for r in results[:5])
    
    if found_query:
        print(f"\n✅ Query image '{test_image.name}' found in top 5 results!")
    else:
        print(f"\n⚠️  Query image '{test_image.name}' not in top 5 results")
    
    print()


def demo_performance_analysis(index: AnnIndex, store: MetadataStore):
    """Demonstrate performance characteristics."""
    print("📊 Demo 3: Performance Analysis")
    print("=" * 50)
    
    # Run multiple searches to measure performance
    num_queries = 5
    total_times = []
    
    print(f"Running {num_queries} test queries...")
    
    for i in range(num_queries):
        # Create synthetic query
        np.random.seed(42 + i)
        query_hist = np.random.random(1152).astype(np.float64)
        query_hist = query_hist / query_hist.sum()
        
        # Time the search
        start_time = time.time()
        results = find_similar(query_hist, index, store, k=20, max_rerank=15)
        total_time = time.time() - start_time
        
        total_times.append(total_time)
        print(f"   Query {i+1}: {len(results)} results in {total_time:.3f}s")
    
    # Calculate statistics
    avg_time = np.mean(total_times)
    min_time = np.min(total_times)
    max_time = np.max(total_times)
    
    print()
    print("Performance Summary:")
    print(f"   Average search time: {avg_time:.3f}s")
    print(f"   Time range: [{min_time:.3f}s, {max_time:.3f}s]")
    print(f"   Throughput: {1/avg_time:.1f} searches/second")
    
    # Performance assessment
    if avg_time < 1.0:
        print("   🚀 Performance: Excellent (< 1s average)")
    elif avg_time < 2.0:
        print("   ✅ Performance: Good (1-2s average)")
    else:
        print("   ⚠️  Performance: Could be optimized (> 2s average)")
    
    print()


def demo_system_validation(index: AnnIndex, store: MetadataStore):
    """Demonstrate system validation."""
    print("🔧 Demo 4: System Validation")
    print("=" * 50)
    
    print("Running complete system validation...")
    
    # Validate the search system
    is_valid = validate_search_system(index, store)
    
    if is_valid:
        print("✅ Search system validation PASSED")
        print("   All components are working correctly together")
    else:
        print("❌ Search system validation FAILED")
        print("   Some components may have issues")
    
    print()


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Demonstrate Chromatica search system")
    parser.add_argument("--dataset", "-d", default="datasets/test-dataset-20", 
                       help="Path to test dataset directory")
    parser.add_argument("--max-images", "-m", type=int, default=20,
                       help="Maximum number of images to process for demo")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("🔍 Chromatica Search System Demonstration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Max images: {args.max_images}")
    print("=" * 60)
    print()
    
    try:
        # Create demo index
        index, store = create_demo_index(args.dataset, args.max_images)
        
        # Run demonstrations
        demo_synthetic_search(index, store)
        demo_real_image_search(index, store, args.dataset)
        demo_performance_analysis(index, store)
        demo_system_validation(index, store)
        
        # Cleanup
        store.close()
        
        print("🎉 Demonstration completed successfully!")
        print()
        print("Next steps:")
        print("   - Try different datasets with --dataset option")
        print("   - Adjust max images with --max-images option")
        print("   - Enable verbose logging with --verbose option")
        print("   - Run comprehensive tests with tools/test_search_system.py")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
