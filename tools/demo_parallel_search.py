#!/usr/bin/env python3
"""
Parallel search demonstration tool for the Chromatica color search engine.

This script demonstrates the parallel processing capabilities by:
1. Running multiple search queries concurrently
2. Comparing performance between sequential and parallel execution
3. Showing real-time performance statistics
4. Demonstrating load testing capabilities

Usage:
    # Activate virtual environment first
    venv311\Scripts\activate

    # Run parallel search demo
    python tools/demo_parallel_search.py

    # Run with specific options
    python tools/demo_parallel_search.py --queries 10 --concurrent 5
    python tools/demo_parallel_search.py --load-test
    python tools/demo_parallel_search.py --compare
"""

import requests
import json
import time
import asyncio
import aiohttp
import argparse
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import random

# API base URL
BASE_URL = os.getenv("CHROMATICA_API_URL", "http://localhost:8000")

# Sample color palettes for testing
SAMPLE_PALETTES = [
    {"colors": "ea6a81,f6d727", "weights": "0.49,0.51", "name": "Pink & Yellow"},
    {"colors": "8bd5ca,89b0ae", "weights": "0.6,0.4", "name": "Mint & Teal"},
    {"colors": "FF0000,00FF00,0000FF", "weights": "0.4,0.3,0.3", "name": "RGB Primary"},
    {"colors": "FFFF00,FF00FF,00FFFF", "weights": "0.33,0.33,0.34", "name": "CMY Secondary"},
    {"colors": "FF6B6B,4ECDC4,45B7D1", "weights": "0.4,0.3,0.3", "name": "Coral & Blues"},
    {"colors": "A8E6CF,FFD93D,FF6B6B", "weights": "0.4,0.3,0.3", "name": "Pastel Rainbow"},
    {"colors": "2C3E50,3498DB,E74C3C", "weights": "0.3,0.4,0.3", "name": "Dark & Bright"},
    {"colors": "F39C12,E67E22,D35400", "weights": "0.4,0.3,0.3", "name": "Orange Gradient"},
    {"colors": "9B59B6,8E44AD,7D3C98", "weights": "0.4,0.3,0.3", "name": "Purple Gradient"},
    {"colors": "1ABC9C,16A085,138D75", "weights": "0.4,0.3,0.3", "name": "Green Gradient"}
]


def generate_random_queries(count: int) -> List[Dict[str, Any]]:
    """Generate random search queries for testing."""
    queries = []
    
    for i in range(count):
        # Pick a random palette
        palette = random.choice(SAMPLE_PALETTES)
        
        # Add some variation
        k = random.randint(5, 20)
        fast_mode = random.choice([True, False])
        batch_size = random.randint(3, 8)
        
        queries.append({
            "colors": palette["colors"],
            "weights": palette["weights"],
            "k": k,
            "fast_mode": fast_mode,
            "batch_size": batch_size,
            "name": f"{palette['name']} #{i+1}"
        })
    
    return queries


def make_search_request(query_params: Dict[str, Any]) -> Dict[str, Any]:
    """Make a single search request."""
    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/search", params=query_params)
        request_time = time.time() - start_time
        
        return {
            "success": response.status_code == 200,
            "time": request_time,
            "status": response.status_code,
            "query_name": query_params.get("name", "Unknown"),
            "results_count": len(response.json().get("results", [])) if response.status_code == 200 else 0
        }
    except Exception as e:
        return {
            "success": False,
            "time": 0,
            "error": str(e),
            "query_name": query_params.get("name", "Unknown"),
            "results_count": 0
        }


def run_sequential_searches(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run searches sequentially."""
    print("🔄 Running sequential searches...")
    
    start_time = time.time()
    results = []
    
    for i, query in enumerate(queries):
        print(f"   Query {i+1}/{len(queries)}: {query['name']}")
        result = make_search_request(query)
        results.append(result)
    
    total_time = time.time() - start_time
    
    return {
        "method": "Sequential",
        "total_time": total_time,
        "results": results,
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"])
    }


def run_parallel_searches(queries: List[Dict[str, Any]], max_workers: int = 5) -> Dict[str, Any]:
    """Run searches in parallel."""
    print(f"⚡ Running parallel searches with {max_workers} workers...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_search_request, query) for query in queries]
        results = [future.result() for future in as_completed(futures)]
    
    total_time = time.time() - start_time
    
    return {
        "method": "Parallel",
        "total_time": total_time,
        "results": results,
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "max_workers": max_workers
    }


async def run_async_searches(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run searches asynchronously."""
    print("🚀 Running async searches...")
    
    async def make_async_request(session, query):
        try:
            start_time = time.time()
            async with session.get(f"{BASE_URL}/search", params=query) as response:
                data = await response.json()
                request_time = time.time() - start_time
                
                return {
                    "success": response.status == 200,
                    "time": request_time,
                    "status": response.status,
                    "query_name": query.get("name", "Unknown"),
                    "results_count": len(data.get("results", [])) if response.status == 200 else 0
                }
        except Exception as e:
            return {
                "success": False,
                "time": 0,
                "error": str(e),
                "query_name": query.get("name", "Unknown"),
                "results_count": 0
            }
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_async_request(session, query) for query in queries]
        results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    return {
        "method": "Async",
        "total_time": total_time,
        "results": results,
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"])
    }


def run_parallel_api_searches(queries: List[Dict[str, Any]], max_concurrent: int = 5) -> Dict[str, Any]:
    """Run searches using the parallel API endpoint."""
    print(f"🎯 Running parallel API searches with max_concurrent={max_concurrent}...")
    
    try:
        # Prepare parallel search request
        request_data = {
            "queries": queries,
            "max_concurrent": max_concurrent
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/search/parallel", json=request_data)
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert API response to our format
            results = []
            for result in data["results"]:
                results.append({
                    "success": result["success"],
                    "time": 0,  # Individual times not available from API
                    "query_name": f"Query {len(results)+1}",
                    "results_count": result.get("results_count", 0)
                })
            
            return {
                "method": "Parallel API",
                "total_time": total_time,
                "results": results,
                "successful": data["successful_queries"],
                "failed": data["failed_queries"],
                "max_concurrent": max_concurrent
            }
        else:
            print(f"   ❌ Parallel API request failed: {response.text}")
            return {
                "method": "Parallel API",
                "total_time": total_time,
                "results": [],
                "successful": 0,
                "failed": len(queries),
                "error": f"HTTP {response.status_code}"
            }
            
    except Exception as e:
        print(f"   ❌ Parallel API request failed: {e}")
        return {
            "method": "Parallel API",
            "total_time": 0,
            "results": [],
            "successful": 0,
            "failed": len(queries),
            "error": str(e)
        }


def print_performance_comparison(results: List[Dict[str, Any]]) -> None:
    """Print performance comparison between different methods."""
    print("\n" + "="*80)
    print("📊 PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"{'Method':<15} {'Total Time':<12} {'Successful':<12} {'Failed':<10} {'Avg Time':<12} {'Speedup':<10}")
    print("-" * 80)
    
    baseline_time = None
    for result in results:
        if result["successful"] > 0:
            avg_time = result["total_time"] / result["successful"]
        else:
            avg_time = 0
        
        if baseline_time is None:
            baseline_time = result["total_time"]
            speedup = "1.00x"
        else:
            speedup = f"{baseline_time / result['total_time']:.2f}x"
        
        print(f"{result['method']:<15} {result['total_time']:.3f}s{'':<7} "
              f"{result['successful']:<12} {result['failed']:<10} "
              f"{avg_time:.3f}s{'':<7} {speedup:<10}")
    
    print("-" * 80)
    
    # Find best method
    best_result = min(results, key=lambda x: x["total_time"])
    print(f"🏆 Best performance: {best_result['method']} ({best_result['total_time']:.3f}s)")


def print_detailed_results(results: List[Dict[str, Any]]) -> None:
    """Print detailed results for each method."""
    for result in results:
        print(f"\n📋 {result['method']} Results:")
        print(f"   Total time: {result['total_time']:.3f}s")
        print(f"   Successful: {result['successful']}")
        print(f"   Failed: {result['failed']}")
        
        if result["results"]:
            times = [r["time"] for r in result["results"] if r["success"]]
            if times:
                print(f"   Min time: {min(times):.3f}s")
                print(f"   Max time: {max(times):.3f}s")
                print(f"   Avg time: {statistics.mean(times):.3f}s")
                print(f"   Std dev: {statistics.stdev(times):.3f}s" if len(times) > 1 else "   Std dev: N/A")
        
        # Show failed queries
        failed_queries = [r for r in result["results"] if not r["success"]]
        if failed_queries:
            print(f"   Failed queries: {[q['query_name'] for q in failed_queries]}")


def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics from the API."""
    try:
        response = requests.get(f"{BASE_URL}/performance/stats")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception:
        return {}


def print_api_stats() -> None:
    """Print current API performance statistics."""
    stats = get_performance_stats()
    if stats:
        print("\n📈 API Performance Statistics:")
        print(f"   Total searches: {stats.get('total_searches', 0)}")
        print(f"   Current concurrent: {stats.get('concurrent_searches', 0)}")
        print(f"   Max concurrent: {stats.get('max_concurrent_searches', 0)}")
        print(f"   Average search time: {stats.get('average_search_time', 0):.3f}s")
        
        recent_times = stats.get('recent_search_times', [])
        if recent_times:
            print(f"   Recent searches: {len(recent_times)}")
            print(f"   Recent avg time: {statistics.mean(recent_times):.3f}s")


async def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Demonstrate Chromatica parallel search capabilities")
    parser.add_argument("--queries", type=int, default=8, help="Number of queries to run (default: 8)")
    parser.add_argument("--concurrent", type=int, default=4, help="Number of concurrent workers (default: 4)")
    parser.add_argument("--compare", action="store_true", help="Compare all methods")
    parser.add_argument("--load-test", action="store_true", help="Run load test with many queries")
    parser.add_argument("--async-only", action="store_true", help="Run only async method")
    parser.add_argument("--parallel-only", action="store_true", help="Run only parallel method")
    
    args = parser.parse_args()
    
    print("🚀 Chromatica Parallel Search Demonstration")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding properly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running")
        print("\nTo start the server:")
        print("1. Activate virtual environment: venv311\\Scripts\\activate")
        print("2. Run: uvicorn src.chromatica.api.main:app --reload")
        return
    
    print("✅ API server is running")
    
    # Generate test queries
    if args.load_test:
        query_count = 50
        concurrent_workers = 20
        print(f"🔥 Load test mode: {query_count} queries with {concurrent_workers} workers")
    else:
        query_count = args.queries
        concurrent_workers = args.concurrent
        print(f"📝 Running {query_count} queries with {concurrent_workers} workers")
    
    queries = generate_random_queries(query_count)
    print(f"   Generated {len(queries)} test queries")
    
    # Show initial API stats
    print_api_stats()
    
    results = []
    
    if args.async_only:
        # Run only async method
        result = await run_async_searches(queries)
        results.append(result)
    elif args.parallel_only:
        # Run only parallel method
        result = run_parallel_searches(queries, concurrent_workers)
        results.append(result)
    elif args.compare:
        # Run all methods for comparison
        print("\n🔄 Running all methods for comparison...")
        
        # Sequential
        result = run_sequential_searches(queries)
        results.append(result)
        
        # Parallel
        result = run_parallel_searches(queries, concurrent_workers)
        results.append(result)
        
        # Async
        result = await run_async_searches(queries)
        results.append(result)
        
        # Parallel API
        result = run_parallel_api_searches(queries, concurrent_workers)
        results.append(result)
    else:
        # Run parallel method by default
        result = run_parallel_searches(queries, concurrent_workers)
        results.append(result)
    
    # Print results
    print_detailed_results(results)
    
    if len(results) > 1:
        print_performance_comparison(results)
    
    # Show final API stats
    print_api_stats()
    
    print("\n✅ Demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())

