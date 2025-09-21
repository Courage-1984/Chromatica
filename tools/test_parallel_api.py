#!/usr/bin/env python3
"""
Parallel API testing tool for the Chromatica color search engine.

This script tests the parallel processing capabilities of the API:
1. Parallel search endpoint testing
2. Performance monitoring
3. Concurrent request handling
4. Load testing with multiple simultaneous requests

Usage:
    # Activate virtual environment first
    venv311\Scripts\activate

    # Run parallel tests
    python tools/test_parallel_api.py

    # Run specific test types
    python tools/test_parallel_api.py --parallel-search
    python tools/test_parallel_api.py --load-test
    python tools/test_parallel_api.py --performance
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

# API base URL
BASE_URL = os.getenv("CHROMATICA_API_URL", "http://localhost:8000")


def test_parallel_search_endpoint() -> bool:
    """Test the parallel search endpoint with multiple queries."""
    print("🔍 Testing parallel search endpoint...")
    
    # Test queries for parallel processing
    test_queries = [
        {
            "colors": "ea6a81,f6d727",
            "weights": "0.49,0.51",
            "k": 10,
            "fast_mode": False,
            "batch_size": 5
        },
        {
            "colors": "FF0000",
            "weights": "1.0",
            "k": 5,
            "fast_mode": True,
            "batch_size": 3
        },
        {
            "colors": "00FF00,0000FF",
            "weights": "0.6,0.4",
            "k": 8,
            "fast_mode": False,
            "batch_size": 4
        },
        {
            "colors": "FFFF00,FF00FF,00FFFF",
            "weights": "0.4,0.3,0.3",
            "k": 12,
            "fast_mode": True,
            "batch_size": 6
        }
    ]
    
    try:
        # Prepare parallel search request
        request_data = {
            "queries": test_queries,
            "max_concurrent": 4
        }
        
        print(f"   Sending {len(test_queries)} queries with max_concurrent=4")
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/search/parallel", json=request_data)
        request_time = time.time() - start_time
        
        print(f"   Status: {response.status_code}")
        print(f"   Request time: {request_time:.3f}s")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"   Total queries: {data['total_queries']}")
            print(f"   Successful: {data['successful_queries']}")
            print(f"   Failed: {data['failed_queries']}")
            print(f"   Total time: {data['total_time_ms']}ms")
            
            # Validate results
            if data['successful_queries'] == len(test_queries):
                print("   ✅ All queries processed successfully")
                return True
            else:
                print(f"   ❌ Only {data['successful_queries']}/{len(test_queries)} queries succeeded")
                return False
        else:
            print(f"   ❌ Request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_performance_stats() -> bool:
    """Test the performance statistics endpoint."""
    print("🔍 Testing performance statistics endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/performance/stats")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"   Total searches: {data['total_searches']}")
            print(f"   Concurrent searches: {data['concurrent_searches']}")
            print(f"   Max concurrent: {data['max_concurrent_searches']}")
            print(f"   Average search time: {data['average_search_time']:.3f}s")
            print(f"   Recent search times: {len(data['recent_search_times'])}")
            
            print("   ✅ Performance stats retrieved successfully")
            return True
        else:
            print(f"   ❌ Request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_concurrent_requests() -> bool:
    """Test concurrent individual search requests."""
    print("🔍 Testing concurrent individual requests...")
    
    # Test queries
    test_queries = [
        {"colors": "FF0000", "weights": "1.0", "k": 5},
        {"colors": "00FF00", "weights": "1.0", "k": 5},
        {"colors": "0000FF", "weights": "1.0", "k": 5},
        {"colors": "FFFF00", "weights": "1.0", "k": 5},
        {"colors": "FF00FF", "weights": "1.0", "k": 5}
    ]
    
    def make_search_request(query_params):
        """Make a single search request."""
        try:
            start_time = time.time()
            response = requests.get(f"{BASE_URL}/search", params=query_params)
            request_time = time.time() - start_time
            
            return {
                "success": response.status_code == 200,
                "time": request_time,
                "status": response.status_code
            }
        except Exception as e:
            return {
                "success": False,
                "time": 0,
                "error": str(e)
            }
    
    try:
        # Execute requests concurrently
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_search_request, query) for query in test_queries]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r["success"])
        times = [r["time"] for r in results if r["success"]]
        
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Successful requests: {successful}/{len(test_queries)}")
        print(f"   Average request time: {statistics.mean(times):.3f}s" if times else "   No successful requests")
        print(f"   Min request time: {min(times):.3f}s" if times else "   No successful requests")
        print(f"   Max request time: {max(times):.3f}s" if times else "   No successful requests")
        
        if successful == len(test_queries):
            print("   ✅ All concurrent requests succeeded")
            return True
        else:
            print(f"   ❌ Only {successful}/{len(test_queries)} requests succeeded")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


async def test_async_requests() -> bool:
    """Test async HTTP requests using aiohttp."""
    print("🔍 Testing async HTTP requests...")
    
    # Test queries
    test_queries = [
        {"colors": "ea6a81", "weights": "1.0", "k": 3},
        {"colors": "f6d727", "weights": "1.0", "k": 3},
        {"colors": "8bd5ca", "weights": "1.0", "k": 3}
    ]
    
    async def make_async_request(session, query_params):
        """Make an async search request."""
        try:
            start_time = time.time()
            async with session.get(f"{BASE_URL}/search", params=query_params) as response:
                await response.text()  # Consume response
                request_time = time.time() - start_time
                
                return {
                    "success": response.status == 200,
                    "time": request_time,
                    "status": response.status
                }
        except Exception as e:
            return {
                "success": False,
                "time": 0,
                "error": str(e)
            }
    
    try:
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [make_async_request(session, query) for query in test_queries]
            results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r["success"])
        times = [r["time"] for r in results if r["success"]]
        
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Successful requests: {successful}/{len(test_queries)}")
        print(f"   Average request time: {statistics.mean(times):.3f}s" if times else "   No successful requests")
        
        if successful == len(test_queries):
            print("   ✅ All async requests succeeded")
            return True
        else:
            print(f"   ❌ Only {successful}/{len(test_queries)} requests succeeded")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_load_performance() -> bool:
    """Test load performance with many concurrent requests."""
    print("🔍 Testing load performance...")
    
    # Generate many test queries
    test_queries = []
    colors = ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF"]
    
    for i in range(20):  # 20 concurrent requests
        color = colors[i % len(colors)]
        test_queries.append({
            "colors": color,
            "weights": "1.0",
            "k": 5
        })
    
    def make_load_request(query_params, request_id):
        """Make a load test request."""
        try:
            start_time = time.time()
            response = requests.get(f"{BASE_URL}/search", params=query_params)
            request_time = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": response.status_code == 200,
                "time": request_time,
                "status": response.status_code
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "time": 0,
                "error": str(e)
            }
    
    try:
        print(f"   Sending {len(test_queries)} concurrent requests...")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(make_load_request, query, i) 
                for i, query in enumerate(test_queries)
            ]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r["success"])
        times = [r["time"] for r in results if r["success"]]
        
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Successful requests: {successful}/{len(test_queries)}")
        print(f"   Requests per second: {len(test_queries)/total_time:.2f}")
        print(f"   Average request time: {statistics.mean(times):.3f}s" if times else "   No successful requests")
        print(f"   Min request time: {min(times):.3f}s" if times else "   No successful requests")
        print(f"   Max request time: {max(times):.3f}s" if times else "   No successful requests")
        
        success_rate = successful / len(test_queries) * 100
        print(f"   Success rate: {success_rate:.1f}%")
        
        if success_rate >= 90:  # 90% success rate threshold
            print("   ✅ Load test passed")
            return True
        else:
            print("   ❌ Load test failed - success rate too low")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Chromatica parallel API capabilities")
    parser.add_argument("--parallel-search", action="store_true", help="Test only parallel search endpoint")
    parser.add_argument("--performance", action="store_true", help="Test only performance stats")
    parser.add_argument("--concurrent", action="store_true", help="Test only concurrent requests")
    parser.add_argument("--async", action="store_true", help="Test only async requests")
    parser.add_argument("--load-test", action="store_true", help="Test only load performance")
    
    args = parser.parse_args()
    
    print("🚀 Chromatica Parallel API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    print("Checking if API server is running...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print("⚠️  API server responded but health check failed")
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running")
        print("\nTo start the server:")
        print("1. Activate virtual environment: venv311\\Scripts\\activate")
        print("2. Run: uvicorn src.chromatica.api.main:app --reload")
        print("3. Then run this test script again")
        return
    
    print()
    
    # Run tests based on arguments
    if args.parallel_search:
        test_parallel_search_endpoint()
    elif args.performance:
        test_performance_stats()
    elif args.concurrent:
        test_concurrent_requests()
    elif args.async:
        await test_async_requests()
    elif args.load_test:
        test_load_performance()
    else:
        # Run all tests
        print("Running all parallel processing tests...\n")
        
        parallel_ok = test_parallel_search_endpoint()
        print()
        
        performance_ok = test_performance_stats()
        print()
        
        concurrent_ok = test_concurrent_requests()
        print()
        
        async_ok = await test_async_requests()
        print()
        
        load_ok = test_load_performance()
        print()
        
        # Summary
        print("=" * 60)
        print("Test Summary:")
        print(f"  Parallel search endpoint: {'✅ PASS' if parallel_ok else '❌ FAIL'}")
        print(f"  Performance stats: {'✅ PASS' if performance_ok else '❌ FAIL'}")
        print(f"  Concurrent requests: {'✅ PASS' if concurrent_ok else '❌ FAIL'}")
        print(f"  Async requests: {'✅ PASS' if async_ok else '❌ FAIL'}")
        print(f"  Load performance: {'✅ PASS' if load_ok else '❌ FAIL'}")
        
        all_passed = parallel_ok and performance_ok and concurrent_ok and async_ok and load_ok
        print(f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())

