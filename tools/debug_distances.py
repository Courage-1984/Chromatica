#!/usr/bin/env python3
"""
Debug script to investigate Sinkhorn-EMD distance calculations.

This script helps debug why all search results are showing distance = 0.0.
"""

import sys
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.core.query import create_query_histogram
from chromatica.core.rerank import compute_sinkhorn_distance
from chromatica.indexing.store import MetadataStore


def debug_distances():
    """Debug the distance calculation issue."""

    try:
        print("🔍 Debugging Sinkhorn-EMD Distance Calculations")
        print("=" * 60)
    except UnicodeEncodeError:
        print("Debugging Sinkhorn-EMD Distance Calculations")
        print("=" * 60)

    # Test 1: Create a query histogram
    print("\n1. Creating test query histogram...")
    try:
        colors = ["FF0000"]  # Red
        weights = [1.0]
        query_hist = create_query_histogram(colors, weights)
        try:
            print(f"   ✅ Query histogram created: shape={query_hist.shape}")
            print(f"   📊 Sum: {query_hist.sum():.6f}")
            print(f"   📊 Min: {query_hist.min():.6f}, Max: {query_hist.max():.6f}")
            print(f"   📊 Non-zero bins: {np.count_nonzero(query_hist)}")
        except UnicodeEncodeError:
            print(f"   Query histogram created: shape={query_hist.shape}")
            print(f"   Sum: {query_hist.sum():.6f}")
            print(f"   Min: {query_hist.min():.6f}, Max: {query_hist.max():.6f}")
            print(f"   Non-zero bins: {np.count_nonzero(query_hist)}")
    except Exception as e:
        try:
            print(f"   ❌ Failed to create query histogram: {e}")
        except UnicodeEncodeError:
            print(f"   Failed to create query histogram: {e}")
        return

    # Test 2: Get some histograms from the metadata store
    print("\n2. Retrieving histograms from metadata store...")
    try:
        store = MetadataStore("test_index/chromatica_metadata.db")
        all_histograms = store.get_all_histograms()
        try:
            print(f"   ✅ Retrieved {len(all_histograms)} histograms")
        except UnicodeEncodeError:
            print(f"   Retrieved {len(all_histograms)} histograms")

        if not all_histograms:
            try:
                print("   ❌ No histograms found in store")
            except UnicodeEncodeError:
                print("   No histograms found in store")
            return

        # Get first few histograms for testing
        test_histograms = list(all_histograms.values())[:3]
        test_ids = list(all_histograms.keys())[:3]

        try:
            print(f"   📊 Testing with histograms: {test_ids}")
        except UnicodeEncodeError:
            print(f"   Testing with histograms: {test_ids}")

    except Exception as e:
        try:
            print(f"   ❌ Failed to retrieve histograms: {e}")
        except UnicodeEncodeError:
            print(f"   Failed to retrieve histograms: {e}")
        return

    # Test 3: Compute distances
    print("\n3. Computing Sinkhorn-EMD distances...")
    try:
        for i, (hist_id, histogram) in enumerate(zip(test_ids, test_histograms)):
            try:
                print(f"\n   📊 Histogram {i+1}: {hist_id}")
            except UnicodeEncodeError:
                print(f"\n   Histogram {i+1}: {hist_id}")
            print(f"      Shape: {histogram.shape}")
            print(f"      Sum: {histogram.sum():.6f}")
            print(f"      Min: {histogram.min():.6f}, Max: {histogram.max():.6f}")
            print(f"      Non-zero bins: {np.count_nonzero(histogram)}")

            # Check if histogram is identical to query
            if np.array_equal(query_hist, histogram):
                try:
                    print(f"      ⚠️  IDENTICAL to query histogram!")
                except UnicodeEncodeError:
                    print(f"      WARNING: IDENTICAL to query histogram!")
            elif np.allclose(query_hist, histogram, atol=1e-10):
                try:
                    print(f"      ⚠️  VERY SIMILAR to query histogram (within 1e-10)")
                except UnicodeEncodeError:
                    print(
                        f"      WARNING: VERY SIMILAR to query histogram (within 1e-10)"
                    )
            else:
                try:
                    print(f"      ✅ Different from query histogram")
                except UnicodeEncodeError:
                    print(f"      Different from query histogram")

            # Compute distance
            try:
                distance = compute_sinkhorn_distance(query_hist, histogram)
                try:
                    print(f"      🎯 Sinkhorn distance: {distance:.10f}")
                except UnicodeEncodeError:
                    print(f"      Sinkhorn distance: {distance:.10f}")

                if distance == 0.0:
                    try:
                        print(f"      ⚠️  Distance is exactly 0.0!")
                    except UnicodeEncodeError:
                        print(f"      WARNING: Distance is exactly 0.0!")
                elif distance < 1e-10:
                    try:
                        print(f"      ⚠️  Distance is very small: {distance:.2e}")
                    except UnicodeEncodeError:
                        print(f"      WARNING: Distance is very small: {distance:.2e}")
                else:
                    try:
                        print(f"      ✅ Normal distance value")
                    except UnicodeEncodeError:
                        print(f"      Normal distance value")

            except Exception as e:
                try:
                    print(f"      ❌ Distance computation failed: {e}")
                except UnicodeEncodeError:
                    print(f"      Distance computation failed: {e}")

    except Exception as e:
        try:
            print(f"   ❌ Failed to compute distances: {e}")
        except UnicodeEncodeError:
            print(f"   Failed to compute distances: {e}")
        return

    # Test 4: Check if histograms are all identical
    print("\n4. Checking histogram uniqueness...")
    try:
        unique_histograms = set()
        for hist_id, histogram in all_histograms.items():
            # Convert to tuple for hashing
            hist_tuple = tuple(histogram.flatten())
            unique_histograms.add(hist_tuple)

        try:
            print(f"   📊 Total histograms: {len(all_histograms)}")
            print(f"   📊 Unique histograms: {len(unique_histograms)}")
        except UnicodeEncodeError:
            print(f"   Total histograms: {len(all_histograms)}")
            print(f"   Unique histograms: {len(unique_histograms)}")

        if len(unique_histograms) == 1:
            try:
                print("   ⚠️  ALL HISTOGRAMS ARE IDENTICAL!")
                print("   🔍 This explains why all distances are 0.0")
            except UnicodeEncodeError:
                print("   WARNING: ALL HISTOGRAMS ARE IDENTICAL!")
                print("   This explains why all distances are 0.0")
        elif len(unique_histograms) < len(all_histograms) * 0.1:
            try:
                print("   ⚠️  Most histograms are identical or very similar")
            except UnicodeEncodeError:
                print("   WARNING: Most histograms are identical or very similar")
        else:
            try:
                print("   ✅ Histograms appear to be unique")
            except UnicodeEncodeError:
                print("   Histograms appear to be unique")

    except Exception as e:
        try:
            print(f"   ❌ Failed to check uniqueness: {e}")
        except UnicodeEncodeError:
            print(f"   Failed to check uniqueness: {e}")
        return

    print("\n" + "=" * 60)
    try:
        print("🎯 Debug Summary:")
    except UnicodeEncodeError:
        print("Debug Summary:")
    print("   - Check if all histograms are identical")
    print("   - Verify histogram generation during indexing")
    print("   - Check if Sinkhorn-EMD is working correctly")
    print("   - Verify query histogram creation")


if __name__ == "__main__":
    debug_distances()
