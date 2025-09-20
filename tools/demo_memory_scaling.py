#!/usr/bin/env python3
"""
Memory Scaling Demonstration Script

This script demonstrates the memory benefits of switching from IndexHNSWFlat
to IndexIVFPQ for the Chromatica color search engine.

Usage:
    python tools/demo_memory_scaling.py

The script will:
1. Create both HNSW and IVFPQ indexes
2. Compare memory usage estimates
3. Demonstrate the training requirement for IVFPQ
4. Show compression ratios and scaling benefits
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.indexing.store import AnnIndex
from chromatica.utils.config import TOTAL_BINS, IVFPQ_M, IVFPQ_NBITS


def create_sample_data(n_vectors: int = 1000) -> np.ndarray:
    """Create sample histogram data for testing."""
    print(f"Creating {n_vectors} sample histograms...")

    # Generate random normalized histograms
    histograms = np.random.rand(n_vectors, TOTAL_BINS).astype(np.float32)

    # Normalize each histogram to sum to 1.0
    histograms = histograms / histograms.sum(axis=1, keepdims=True)

    print(
        f"✓ Generated {histograms.shape[0]} histograms of dimension {histograms.shape[1]}"
    )
    return histograms


def demonstrate_memory_scaling():
    """Demonstrate memory scaling benefits of IndexIVFPQ."""
    print("=" * 60)
    print("🚀 CHROMATICA MEMORY SCALING DEMONSTRATION")
    print("=" * 60)

    # Create sample data
    n_vectors = 1000
    histograms = create_sample_data(n_vectors)

    print("\n📊 MEMORY USAGE COMPARISON")
    print("-" * 40)

    # Calculate HNSW memory usage (theoretical)
    hnsw_memory_per_vector = TOTAL_BINS * 4  # 4 bytes per float32
    hnsw_total_memory = n_vectors * hnsw_memory_per_vector

    print(f"HNSW Index (theoretical):")
    print(f"  • Memory per vector: {hnsw_memory_per_vector:,} bytes")
    print(
        f"  • Total memory ({n_vectors:,} vectors): {hnsw_total_memory:,} bytes ({hnsw_total_memory/1024/1024:.1f} MB)"
    )

    # Create IVFPQ index and get memory estimates
    print(f"\nIndexIVFPQ (actual):")
    ivfpq_index = AnnIndex()
    memory_info = ivfpq_index.get_memory_usage_estimate()

    print(f"  • Memory per vector: {memory_info['memory_per_vector']:.1f} bytes")
    print(
        f"  • Total memory ({n_vectors:,} vectors): {memory_info['total_memory'] * n_vectors:,.0f} bytes ({memory_info['total_memory'] * n_vectors/1024/1024:.1f} MB)"
    )
    print(f"  • Compression ratio: {memory_info['compression_ratio']:.1f}x")

    # Calculate scaling benefits
    print(f"\n🎯 SCALING BENEFITS")
    print("-" * 40)

    scales = [1000, 10000, 100000, 1000000, 10000000]

    print(f"{'Scale':<10} {'HNSW Memory':<15} {'IVFPQ Memory':<15} {'Savings':<15}")
    print("-" * 60)

    for scale in scales:
        hnsw_mem = scale * hnsw_memory_per_vector
        ivfpq_mem = scale * memory_info["memory_per_vector"]
        savings = hnsw_mem - ivfpq_mem

        print(
            f"{scale:,}     {hnsw_mem/1024/1024/1024:.1f} GB      {ivfpq_mem/1024/1024:.1f} MB      {savings/1024/1024/1024:.1f} GB"
        )

    print(f"\n🔧 INDEXIVFPQ TRAINING DEMONSTRATION")
    print("-" * 40)

    # Demonstrate training requirement
    print("1. Creating IndexIVFPQ...")
    index = AnnIndex()
    print(f"   ✓ Index created with M={IVFPQ_M}, nbits={IVFPQ_NBITS}")
    print(f"   ✓ Training status: {index.is_trained}")

    # Try to add vectors without training (should fail)
    print("\n2. Attempting to add vectors without training...")
    try:
        index.add(histograms[:10])
        print("   ❌ ERROR: This should have failed!")
    except ValueError as e:
        print(f"   ✓ Expected error: {e}")

    # Train the index
    print("\n3. Training IndexIVFPQ...")
    training_size = min(200, len(histograms) // 5)  # Use 20% for training
    training_data = histograms[:training_size]

    print(f"   • Using {training_size} vectors for training")
    print(f"   • Training vectors shape: {training_data.shape}")

    index.train(training_data)
    print(f"   ✓ Training completed! Status: {index.is_trained}")

    # Now add vectors successfully
    print("\n4. Adding vectors to trained index...")
    added_count = index.add(histograms)
    print(f"   ✓ Successfully added {added_count} vectors")
    print(f"   ✓ Total vectors in index: {index.get_total_vectors()}")

    # Demonstrate search
    print("\n5. Testing search functionality...")
    query = histograms[0].reshape(1, -1)
    distances, indices = index.search(query, k=5)

    print(f"   ✓ Search completed successfully")
    print(f"   ✓ Retrieved {len(indices[0])} nearest neighbors")
    print(f"   ✓ Distance range: {distances.min():.4f} to {distances.max():.4f}")

    print(f"\n📈 FINAL MEMORY STATISTICS")
    print("-" * 40)

    final_memory_info = index.get_memory_usage_estimate()
    print(f"Total vectors indexed: {final_memory_info['total_vectors']:,}")
    print(f"Memory per vector: {final_memory_info['memory_per_vector']:.1f} bytes")
    print(
        f"Total memory used: {final_memory_info['total_memory']:,.0f} bytes ({final_memory_info['total_memory']/1024:.1f} KB)"
    )
    print(f"Compression ratio: {final_memory_info['compression_ratio']:.1f}x")

    # Calculate what HNSW would have used
    hnsw_equivalent = (
        final_memory_info["total_vectors"] * final_memory_info["full_vector_memory"]
    )
    print(
        f"HNSW equivalent memory: {hnsw_equivalent:,.0f} bytes ({hnsw_equivalent/1024/1024:.1f} MB)"
    )
    print(
        f"Memory saved: {hnsw_equivalent - final_memory_info['total_memory']:,.0f} bytes ({(hnsw_equivalent - final_memory_info['total_memory'])/1024/1024:.1f} MB)"
    )

    print(f"\n✅ MEMORY SCALING DEMONSTRATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demonstrate_memory_scaling()
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
