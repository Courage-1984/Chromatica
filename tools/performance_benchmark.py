#!/usr/bin/env python3
"""
Performance benchmark script for Chromatica Color Search Engine.

This script tests the API performance with various query types and measures
response times, success rates, and system metrics.
"""

import requests
import time
import statistics
import json
from typing import List, Dict, Any


def run_performance_benchmark() -> None:
    """Run comprehensive performance benchmark tests."""
    print("🚀 Chromatica Performance Benchmark")
    print("=" * 50)

    # Test different query types
    queries = [
        {"colors": "FF0000,00FF00", "weights": "0.5,0.5", "k": 5},
        {"colors": "0000FF,FFFF00", "weights": "0.7,0.3", "k": 10},
        {"colors": "FF00FF,00FFFF", "weights": "0.6,0.4", "k": 8},
        {"colors": "FF8000,8000FF", "weights": "0.8,0.2", "k": 6},
        {"colors": "FF0000,00FF00,0000FF", "weights": "0.4,0.3,0.3", "k": 7},
        {"colors": "FF6B6B,4ECDC4,45B7D1", "weights": "0.5,0.3,0.2", "k": 12},
        {"colors": "A8E6CF,FFD93D,FF6B6B", "weights": "0.4,0.4,0.2", "k": 9},
        {"colors": "6C5CE7,FD79A8,FDCB6E", "weights": "0.3,0.4,0.3", "k": 15},
    ]

    response_times = []
    api_times = []
    ann_times = []
    rerank_times = []
    success_count = 0
    total_results = 0

    for i, query in enumerate(queries, 1):
        print(f"Test {i}: {query['colors']} with k={query['k']}")

        start_time = time.time()
        try:
            response = requests.get(
                "http://localhost:8000/search", params=query, timeout=30
            )
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                success_count += 1

                # Extract API timing metrics
                metadata = data.get("metadata", {})
                api_times.append(metadata.get("total_time_ms", 0))
                ann_times.append(metadata.get("ann_time_ms", 0))
                rerank_times.append(metadata.get("rerank_time_ms", 0))
                total_results += data.get("results_count", 0)

                print(f"  ✅ Success: {response_time:.1f}ms")
                print(f"  📊 Results: {data['results_count']} images")
                print(f"  ⚡ API Time: {metadata.get('total_time_ms', 0)}ms")
                print(f"  🔍 ANN Time: {metadata.get('ann_time_ms', 0)}ms")
                print(f"  🎯 Rerank Time: {metadata.get('rerank_time_ms', 0)}ms")
            else:
                print(f"  ❌ Failed: Status {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

    # Performance Summary
    print("\n📈 Performance Summary:")
    print(
        f"  Success Rate: {success_count}/{len(queries)} ({success_count/len(queries)*100:.1f}%)"
    )
    print(f"  Total Results: {total_results} images")

    if response_times:
        print(f"\n⏱️  Response Times:")
        print(f"  Average: {statistics.mean(response_times):.1f}ms")
        print(f"  Min: {min(response_times):.1f}ms")
        print(f"  Max: {max(response_times):.1f}ms")
        print(f"  Median: {statistics.median(response_times):.1f}ms")
        print(f"  Std Dev: {statistics.stdev(response_times):.1f}ms")

    if api_times:
        print(f"\n🔧 API Performance:")
        print(f"  Average API Time: {statistics.mean(api_times):.1f}ms")
        print(f"  Average ANN Time: {statistics.mean(ann_times):.1f}ms")
        print(f"  Average Rerank Time: {statistics.mean(rerank_times):.1f}ms")

        # Performance targets from critical instructions
        target_total = 450  # P95 < 450ms
        target_ann = 150  # ANN < 150ms
        target_rerank = 300  # Rerank < 300ms

        print(f"\n🎯 Performance Targets:")
        print(f"  Total Time Target: <{target_total}ms")
        print(f"  ANN Time Target: <{target_ann}ms")
        print(f"  Rerank Time Target: <{target_rerank}ms")

        avg_total = statistics.mean(api_times)
        avg_ann = statistics.mean(ann_times)
        avg_rerank = statistics.mean(rerank_times)

        print(f"\n📊 Target Compliance:")
        print(
            f"  Total Time: {'✅' if avg_total < target_total else '❌'} {avg_total:.1f}ms"
        )
        print(f"  ANN Time: {'✅' if avg_ann < target_ann else '❌'} {avg_ann:.1f}ms")
        print(
            f"  Rerank Time: {'✅' if avg_rerank < target_rerank else '❌'} {avg_rerank:.1f}ms"
        )


def test_concurrent_requests() -> None:
    """Test concurrent request handling."""
    print("\n🔄 Concurrent Request Test")
    print("=" * 30)

    import concurrent.futures
    import threading

    def make_request(query_id: int) -> Dict[str, Any]:
        """Make a single request and return timing data."""
        query = {"colors": "FF0000,00FF00", "weights": "0.5,0.5", "k": 5}
        start_time = time.time()
        try:
            response = requests.get(
                "http://localhost:8000/search", params=query, timeout=30
            )
            end_time = time.time()
            return {
                "query_id": query_id,
                "status_code": response.status_code,
                "response_time": (end_time - start_time) * 1000,
                "success": response.status_code == 200,
            }
        except Exception as e:
            return {
                "query_id": query_id,
                "status_code": 0,
                "response_time": 0,
                "success": False,
                "error": str(e),
            }

    # Test with 5 concurrent requests
    concurrent_requests = 5
    print(f"Testing {concurrent_requests} concurrent requests...")

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrent_requests
    ) as executor:
        futures = [executor.submit(make_request, i) for i in range(concurrent_requests)]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    end_time = time.time()

    total_time = (end_time - start_time) * 1000
    successful_requests = sum(1 for r in results if r["success"])
    avg_response_time = statistics.mean(
        [r["response_time"] for r in results if r["success"]]
    )

    print(f"  Total Time: {total_time:.1f}ms")
    print(f"  Successful Requests: {successful_requests}/{concurrent_requests}")
    print(f"  Average Response Time: {avg_response_time:.1f}ms")
    print(
        f"  Throughput: {concurrent_requests / (total_time / 1000):.1f} requests/second"
    )


if __name__ == "__main__":
    try:
        run_performance_benchmark()
        test_concurrent_requests()
        print("\n✅ Performance benchmark completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
