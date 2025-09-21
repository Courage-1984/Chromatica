#!/usr/bin/env python3
"""
Test script for the Chromatica FastAPI endpoint.

This script tests the API endpoints to ensure they're working correctly:
1. Health check endpoint
2. Root endpoint
3. Search endpoint with various query parameters

Usage:
    # Activate virtual environment first
    venv311\Scripts\activate

    # Run the test
    python tools/test_api.py

    # Or test specific endpoints
    python tools/test_api.py --health
    python tools/test_api.py --search
"""

import requests
import json
import time
import argparse
import asyncio
import aiohttp
import os
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# API base URL
BASE_URL = os.getenv("CHROMATICA_API_URL", "http://localhost:8000")


def test_health_endpoint() -> bool:
    """Test the health check endpoint."""
    print("🔍 Testing health endpoint...")

    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Message: {data.get('message')}")

            if "data" in data:
                print(
                    f"   Images in store: {data['data'].get('images_in_store', 'N/A')}"
                )
                print(
                    f"   Vectors in index: {data['data'].get('vectors_in_index', 'N/A')}"
                )

            return True
        else:
            print(f"   Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to API server")
        print(
            "   Make sure the server is running with: uvicorn src.chromatica.api.main:app --reload"
        )
        return False
    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_root_endpoint() -> bool:
    """Test the root endpoint."""
    print("🔍 Testing root endpoint...")

    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Message: {data.get('message')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Status: {data.get('status')}")

            endpoints = data.get("endpoints", {})
            print(f"   Available endpoints: {list(endpoints.keys())}")

            return True
        else:
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_search_endpoint() -> bool:
    """Test the search endpoint with sample queries."""
    print("🔍 Testing search endpoint...")

    # Test queries
    test_queries = [
        {
            "colors": "ea6a81,f6d727",
            "weights": "0.49,0.51",
            "k": 10,
            "description": "Pink and yellow with balanced weights",
        },
        {
            "colors": "FF0000",
            "weights": "1.0",
            "k": 5,
            "description": "Pure red, single color",
        },
        {
            "colors": "00FF00,0000FF,FFFF00",
            "weights": "0.4,0.3,0.3",
            "k": 15,
            "description": "Green, blue, yellow with varied weights",
        },
    ]

    success_count = 0

    for i, query in enumerate(test_queries, 1):
        print(f"   Test {i}: {query['description']}")
        print(f"     Colors: {query['colors']}")
        print(f"     Weights: {query['weights']}")
        print(f"     k: {query['k']}")

        try:
            # Build query parameters
            params = {
                "colors": query["colors"],
                "weights": query["weights"],
                "k": query["k"],
            }

            # Make request
            start_time = time.time()
            response = requests.get(f"{BASE_URL}/search", params=params)
            request_time = time.time() - start_time

            print(f"     Status: {response.status_code}")
            print(f"     Request time: {request_time:.3f}s")

            if response.status_code == 200:
                data = response.json()

                # Validate response structure
                required_fields = [
                    "query_id",
                    "query",
                    "results_count",
                    "results",
                    "metadata",
                ]
                if all(field in data for field in required_fields):
                    print(f"     Query ID: {data['query_id']}")
                    print(f"     Results count: {data['results_count']}")
                    print(f"     Total time: {data['metadata']['total_time_ms']}ms")

                    if data["results"]:
                        print(
                            f"     Top result distance: {data['results'][0]['distance']:.6f}"
                        )

                    success_count += 1
                    print("     ✅ Test passed")
                else:
                    print("     ❌ Test failed: Invalid response structure")
                    print(
                        f"     Missing fields: {[f for f in required_fields if f not in data]}"
                    )
            else:
                print(f"     ❌ Test failed: {response.text}")

        except Exception as e:
            print(f"     ❌ Test failed: {e}")

        print()  # Empty line for readability

    print(f"   Search endpoint tests: {success_count}/{len(test_queries)} passed")
    return success_count == len(test_queries)


def test_invalid_queries() -> bool:
    """Test the search endpoint with invalid queries to ensure proper error handling."""
    print("🔍 Testing invalid query handling...")

    invalid_queries = [
        {
            "params": {"colors": "", "weights": "1.0"},
            "expected_status": 400,
            "description": "Empty colors",
        },
        {
            "params": {"colors": "invalid", "weights": "1.0"},
            "expected_status": 400,
            "description": "Invalid hex color",
        },
        {
            "params": {"colors": "FF0000", "weights": "0.5,0.6"},
            "expected_status": 400,
            "description": "Mismatched colors and weights",
        },
        {
            "params": {"colors": "FF0000", "weights": "-1.0"},
            "expected_status": 400,
            "description": "Negative weight",
        },
    ]

    success_count = 0

    for i, query in enumerate(invalid_queries, 1):
        print(f"   Test {i}: {query['description']}")

        try:
            response = requests.get(f"{BASE_URL}/search", params=query["params"])

            if response.status_code == query["expected_status"]:
                print(f"     ✅ Correctly rejected with status {response.status_code}")
                success_count += 1
            else:
                print(
                    f"     ❌ Expected status {query['expected_status']}, got {response.status_code}"
                )
                print(f"     Response: {response.text}")

        except Exception as e:
            print(f"     ❌ Test failed: {e}")

    print(f"   Invalid query tests: {success_count}/{len(invalid_queries)} passed")
    return success_count == len(invalid_queries)


def test_parallel_requests() -> bool:
    """Test multiple concurrent search requests."""
    print("🔍 Testing parallel search requests...")
    
    # Test queries
    test_queries = [
        {"colors": "FF0000", "weights": "1.0", "k": 5},
        {"colors": "00FF00", "weights": "1.0", "k": 5},
        {"colors": "0000FF", "weights": "1.0", "k": 5},
        {"colors": "FFFF00", "weights": "1.0", "k": 5},
        {"colors": "FF00FF", "weights": "1.0", "k": 5}
    ]
    
    def make_request(query_params):
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
            return {"success": False, "time": 0, "error": str(e)}
    
    try:
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, query) for query in test_queries]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r["success"])
        
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Successful: {successful}/{len(test_queries)}")
        print(f"   Average time per request: {total_time/len(test_queries):.3f}s")
        
        return successful == len(test_queries)
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_performance_stats() -> bool:
    """Test performance statistics endpoint."""
    print("🔍 Testing performance statistics...")
    
    try:
        response = requests.get(f"{BASE_URL}/performance/stats")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Total searches: {data['total_searches']}")
            print(f"   Concurrent searches: {data['concurrent_searches']}")
            print(f"   Average search time: {data['average_search_time']:.3f}s")
            return True
        else:
            print(f"   ❌ Request failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Chromatica API endpoints")
    parser.add_argument(
        "--health", action="store_true", help="Test only health endpoint"
    )
    parser.add_argument(
        "--search", action="store_true", help="Test only search endpoint"
    )
    parser.add_argument(
        "--invalid", action="store_true", help="Test only invalid query handling"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Test parallel request handling"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Test performance statistics"
    )

    args = parser.parse_args()

    print("🚀 Chromatica API Test Suite")
    print("=" * 50)

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
    if args.health:
        test_health_endpoint()
    elif args.search:
        test_search_endpoint()
    elif args.invalid:
        test_invalid_queries()
    elif args.parallel:
        test_parallel_requests()
    elif args.performance:
        test_performance_stats()
    else:
        # Run all tests
        print("Running all tests...\n")

        health_ok = test_health_endpoint()
        print()

        root_ok = test_root_endpoint()
        print()

        search_ok = test_search_endpoint()
        print()

        invalid_ok = test_invalid_queries()
        print()

        parallel_ok = test_parallel_requests()
        print()

        performance_ok = test_performance_stats()
        print()

        # Summary
        print("=" * 50)
        print("Test Summary:")
        print(f"  Health endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
        print(f"  Root endpoint: {'✅ PASS' if root_ok else '❌ FAIL'}")
        print(f"  Search endpoint: {'✅ PASS' if search_ok else '❌ FAIL'}")
        print(f"  Invalid query handling: {'✅ PASS' if invalid_ok else '❌ FAIL'}")
        print(f"  Parallel requests: {'✅ PASS' if parallel_ok else '❌ FAIL'}")
        print(f"  Performance stats: {'✅ PASS' if performance_ok else '❌ FAIL'}")

        all_passed = health_ok and root_ok and search_ok and invalid_ok and parallel_ok and performance_ok
        print(
            f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
        )


if __name__ == "__main__":
    main()
