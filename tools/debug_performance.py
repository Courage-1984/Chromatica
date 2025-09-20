#!/usr/bin/env python3
"""
Debug performance issues in the Chromatica search system.
"""

import time
import requests
import json
from pathlib import Path


def check_index_info():
    """Check basic index information."""
    print("🔍 Index Information Check")
    print("=" * 30)

    index_dir = Path("index")
    if not index_dir.exists():
        print("❌ Index directory not found")
        return

    faiss_file = index_dir / "chromatica_index.faiss"
    db_file = index_dir / "chromatica_metadata.db"

    if faiss_file.exists():
        size_mb = faiss_file.stat().st_size / (1024 * 1024)
        print(f"✅ FAISS Index: {size_mb:.2f} MB")
    else:
        print("❌ FAISS index not found")

    if db_file.exists():
        size_mb = db_file.stat().st_size / (1024 * 1024)
        print(f"✅ DuckDB Store: {size_mb:.2f} MB")

        # Try to get image count
        try:
            import duckdb

            conn = duckdb.connect(str(db_file))
            result = conn.execute("SELECT COUNT(*) FROM image_metadata").fetchone()
            print(f"✅ Total Images: {result[0]}")
            conn.close()
        except Exception as e:
            print(f"⚠️  Could not read image count: {e}")
    else:
        print("❌ DuckDB store not found")


def test_simple_search():
    """Test a simple search to understand performance."""
    print("\n🔍 Simple Search Test")
    print("=" * 25)

    query = {
        "colors": "FF0000,00FF00",
        "weights": "0.5,0.5",
        "k": 3,  # Small k for faster results
    }

    print(f"Query: {query}")

    start_time = time.time()
    try:
        response = requests.get(
            "http://localhost:8000/search", params=query, timeout=60
        )
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            total_time = (end_time - start_time) * 1000

            print(f"✅ Success: {total_time:.1f}ms")
            print(f"📊 Results: {data['results_count']} images")

            metadata = data.get("metadata", {})
            print(f"⚡ API Time: {metadata.get('total_time_ms', 0)}ms")
            print(f"🔍 ANN Time: {metadata.get('ann_time_ms', 0)}ms")
            print(f"🎯 Rerank Time: {metadata.get('rerank_time_ms', 0)}ms")
            print(f"📦 Index Size: {metadata.get('index_size', 0)}")

            # Show first result
            if data.get("results"):
                first_result = data["results"][0]
                print(f"\n📋 First Result:")
                print(f"  Image ID: {first_result['image_id']}")
                print(f"  Distance: {first_result['distance']:.2f}")
                print(f"  File Path: {first_result['file_path']}")
        else:
            print(f"❌ Failed: Status {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def test_different_k_values():
    """Test different k values to understand scaling."""
    print("\n📊 K Value Scaling Test")
    print("=" * 30)

    base_query = {"colors": "FF0000,00FF00", "weights": "0.5,0.5"}

    k_values = [1, 3, 5, 10, 20]

    for k in k_values:
        query = {**base_query, "k": k}
        print(f"\nTesting k={k}:")

        start_time = time.time()
        try:
            response = requests.get(
                "http://localhost:8000/search", params=query, timeout=60
            )
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                total_time = (end_time - start_time) * 1000
                metadata = data.get("metadata", {})

                print(
                    f"  ✅ {total_time:.1f}ms total, {metadata.get('total_time_ms', 0)}ms API"
                )
                print(f"  📊 {data['results_count']} results")
            else:
                print(f"  ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")


if __name__ == "__main__":
    try:
        check_index_info()
        test_simple_search()
        test_different_k_values()
        print("\n✅ Performance debug completed!")
    except KeyboardInterrupt:
        print("\n⏹️  Debug interrupted by user")
    except Exception as e:
        print(f"\n❌ Debug failed: {str(e)}")
