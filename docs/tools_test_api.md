# Test API Tool

## Overview

The `test_api.py` tool is a comprehensive testing script for the Chromatica FastAPI endpoint. It provides automated testing capabilities to ensure the API endpoints are working correctly, including health checks, root endpoint validation, and search functionality testing with various query parameters.

## Purpose

This tool is designed to:
- Test API endpoint availability and responsiveness
- Validate search endpoint functionality with different query types
- Verify API response formats and data structures
- Provide performance benchmarking for API calls
- Serve as a testing framework for API development

## Features

- **Endpoint Testing**: Comprehensive testing of all API endpoints
- **Search Validation**: Test search functionality with various parameters
- **Response Validation**: Verify API response formats and content
- **Performance Testing**: Measure API response times
- **Error Handling**: Test error scenarios and edge cases
- **Automated Testing**: Run complete test suites automatically

## Usage

### Basic Usage

```bash
# Activate virtual environment first
venv311\Scripts\activate

# Run all tests
python tools/test_api.py

# Test specific endpoints
python tools/test_api.py --health
python tools/test_api.py --search
python tools/test_api.py --root
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--health` | Test only health endpoint | False |
| `--search` | Test only search endpoint | False |
| `--root` | Test only root endpoint | False |
| `--all` | Test all endpoints (default) | True |

## Prerequisites

### API Server Setup

Before running the tests, ensure the FastAPI server is running:

```bash
# Start the API server
uvicorn src.chromatica.api.main:app --reload

# Or use the build script
python scripts/build_index.py
```

### Virtual Environment

Always activate the virtual environment:

```bash
# Windows
venv311\Scripts\activate

# Verify activation
python --version
pip list
```

## Core Test Functions

### 1. Health Endpoint Testing

```python
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
                print(f"   Images in store: {data['data'].get('images_in_store', 'N/A')}")
                print(f"   Vectors in index: {data['data'].get('vectors_in_index', 'N/A')}")

            return True
        else:
            print(f"   Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to API server")
        print("   Make sure the server is running with: uvicorn src.chromatica.api.main:app --reload")
        return False
    except Exception as e:
        print(f"   Error: {e}")
        return False
```

**What it tests:**
- API server connectivity
- Health endpoint response format
- System status information
- Store and index statistics

**Expected Response:**
```json
{
  "status": "healthy",
  "message": "System is operational",
  "data": {
    "images_in_store": 1000,
    "vectors_in_index": 1000,
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

### 2. Root Endpoint Testing

```python
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
```

**What it tests:**
- API root endpoint availability
- Version information
- Available endpoint listing
- Basic API information

**Expected Response:**
```json
{
  "message": "Welcome to Chromatica Color Search Engine",
  "version": "1.0.0",
  "status": "operational",
  "endpoints": {
    "health": "/health",
    "search": "/search",
    "docs": "/docs"
  }
}
```

### 3. Search Endpoint Testing

```python
def test_search_endpoint() -> bool:
    """Test the search endpoint with sample queries."""
    print("🔍 Testing search endpoint...")

    # Test queries
    test_queries = [
        {
            "name": "Basic Color Search",
            "params": {
                "colors": ["#FF0000", "#00FF00"],
                "weights": [0.6, 0.4],
                "k": 5
            }
        },
        {
            "name": "Single Color Search",
            "params": {
                "colors": ["#0000FF"],
                "weights": [1.0],
                "k": 10
            }
        },
        {
            "name": "High K Search",
            "params": {
                "colors": ["#FFA500", "#800080"],
                "weights": [0.7, 0.3],
                "k": 20
            }
        }
    ]

    success_count = 0
    
    for query in test_queries:
        print(f"\n   Testing: {query['name']}")
        
        try:
            response = requests.get(f"{BASE_URL}/search", params=query['params'])
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                if validate_search_response(data):
                    print(f"     ✅ Success: {len(data.get('results', []))} results")
                    success_count += 1
                else:
                    print(f"     ❌ Invalid response format")
            else:
                print(f"     ❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"     ❌ Exception: {e}")
    
    print(f"\n   Search tests: {success_count}/{len(test_queries)} passed")
    return success_count == len(test_queries)
```

**What it tests:**
- Search endpoint functionality
- Query parameter handling
- Response format validation
- Multiple query scenarios
- Error handling

## Response Validation

### Search Response Validation

```python
def validate_search_response(data: Dict[str, Any]) -> bool:
    """Validate the structure of a search response."""
    
    # Check required fields
    required_fields = ['query', 'results', 'metadata']
    for field in required_fields:
        if field not in data:
            print(f"     Missing required field: {field}")
            return False
    
    # Validate query information
    query = data['query']
    if not isinstance(query, dict):
        print("     Query field must be a dictionary")
        return False
    
    # Validate results
    results = data['results']
    if not isinstance(results, list):
        print("     Results field must be a list")
        return False
    
    # Validate each result
    for i, result in enumerate(results):
        if not validate_search_result(result):
            print(f"     Invalid result at index {i}")
            return False
    
    # Validate metadata
    metadata = data['metadata']
    if not isinstance(metadata, dict):
        print("     Metadata field must be a dictionary")
        return False
    
    return True

def validate_search_result(result: Dict[str, Any]) -> bool:
    """Validate a single search result."""
    
    required_fields = ['image_id', 'similarity_score', 'file_path']
    for field in required_fields:
        if field not in result:
            print(f"       Missing required field in result: {field}")
            return False
    
    # Validate data types
    if not isinstance(result['image_id'], str):
        print("       image_id must be a string")
        return False
    
    if not isinstance(result['similarity_score'], (int, float)):
        print("       similarity_score must be a number")
        return False
    
    if not isinstance(result['file_path'], str):
        print("       file_path must be a string")
        return False
    
    return True
```

## Performance Testing

### Response Time Benchmarking

```python
def benchmark_api_performance() -> None:
    """Benchmark API response times."""
    
    print("\n⚡ Performance Benchmarking")
    print("=" * 40)
    
    # Test endpoints multiple times
    endpoints = [
        ("/health", "Health Check"),
        ("/", "Root Endpoint"),
        ("/search?colors=%23FF0000&weights=1.0&k=5", "Search Endpoint")
    ]
    
    results = {}
    
    for endpoint, name in endpoints:
        times = []
        
        print(f"\nBenchmarking {name}:")
        
        for i in range(5):  # 5 test runs
            try:
                start_time = time.time()
                response = requests.get(f"{BASE_URL}{endpoint}")
                end_time = time.time()
                
                if response.status_code == 200:
                    response_time = (end_time - start_time) * 1000  # Convert to ms
                    times.append(response_time)
                    print(f"  Run {i+1}: {response_time:.2f} ms")
                else:
                    print(f"  Run {i+1}: Failed (Status: {response.status_code})")
                    
            except Exception as e:
                print(f"  Run {i+1}: Error - {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"  Average: {avg_time:.2f} ms")
            print(f"  Range: {min_time:.2f} - {max_time:.2f} ms")
            
            results[name] = {
                "average": avg_time,
                "min": min_time,
                "max": max_time,
                "runs": len(times)
            }
    
    return results
```

## Error Scenario Testing

### Connection Error Testing

```python
def test_connection_errors() -> None:
    """Test API connection error scenarios."""
    
    print("\n🔌 Connection Error Testing")
    print("=" * 40)
    
    # Test with invalid URL
    invalid_urls = [
        "http://localhost:9999",  # Wrong port
        "http://invalid-host:8000",  # Invalid hostname
        "http://localhost:8000/nonexistent"  # Invalid endpoint
    ]
    
    for url in invalid_urls:
        print(f"\nTesting invalid URL: {url}")
        
        try:
            response = requests.get(f"{url}/health", timeout=5)
            print(f"  Unexpected success: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("  ✅ Correctly handled connection error")
        except requests.exceptions.Timeout:
            print("  ✅ Correctly handled timeout")
        except Exception as e:
            print(f"  ⚠️  Unexpected error: {e}")
```

### Invalid Parameter Testing

```python
def test_invalid_parameters() -> None:
    """Test search endpoint with invalid parameters."""
    
    print("\n⚠️  Invalid Parameter Testing")
    print("=" * 40)
    
    invalid_queries = [
        {
            "name": "Missing Colors",
            "params": {"weights": [1.0], "k": 5}
        },
        {
            "name": "Mismatched Arrays",
            "params": {"colors": ["#FF0000"], "weights": [0.5, 0.5], "k": 5}
        },
        {
            "name": "Invalid K Value",
            "params": {"colors": ["#FF0000"], "weights": [1.0], "k": -1}
        },
        {
            "name": "Invalid Hex Color",
            "params": {"colors": ["INVALID"], "weights": [1.0], "k": 5}
        }
    ]
    
    for query in invalid_queries:
        print(f"\nTesting: {query['name']}")
        
        try:
            response = requests.get(f"{BASE_URL}/search", params=query['params'])
            
            if response.status_code == 400:
                print("  ✅ Correctly returned 400 Bad Request")
                error_data = response.json()
                if 'detail' in error_data:
                    print(f"     Error: {error_data['detail']}")
            else:
                print(f"  ⚠️  Unexpected status: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
```

## Complete Test Suite

### Running All Tests

```python
def run_complete_test_suite() -> Dict[str, bool]:
    """Run the complete API test suite."""
    
    print("🚀 Chromatica API Test Suite")
    print("=" * 50)
    print(f"Base URL: {BASE_URL}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test basic endpoints
    print("\n📋 Basic Endpoint Tests")
    print("-" * 30)
    
    results['health'] = test_health_endpoint()
    results['root'] = test_root_endpoint()
    
    # Test search functionality
    print("\n🔍 Search Functionality Tests")
    print("-" * 30)
    
    results['search'] = test_search_endpoint()
    
    # Test error scenarios
    print("\n⚠️  Error Scenario Tests")
    print("-" * 30)
    
    test_connection_errors()
    test_invalid_parameters()
    
    # Performance testing
    print("\n⚡ Performance Tests")
    print("-" * 30)
    
    performance_results = benchmark_api_performance()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Basic Tests: {passed}/{total} passed")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.upper()}: {status}")
    
    if passed == total:
        print(f"\n🎉 All tests passed! API is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check API configuration.")
    
    return results
```

## Integration Examples

### With CI/CD Pipeline

```python
#!/usr/bin/env python3
"""
CI/CD integration script for API testing.
"""

import sys
from tools.test_api import run_complete_test_suite

def main():
    """Run tests and exit with appropriate code for CI/CD."""
    
    try:
        results = run_complete_test_suite()
        
        # Check if all critical tests passed
        critical_tests = ['health', 'root', 'search']
        all_critical_passed = all(results.get(test, False) for test in critical_tests)
        
        if all_critical_passed:
            print("✅ CI/CD: All critical tests passed")
            sys.exit(0)  # Success
        else:
            print("❌ CI/CD: Critical tests failed")
            sys.exit(1)  # Failure
            
    except Exception as e:
        print(f"❌ CI/CD: Test suite error: {e}")
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()
```

### With Monitoring System

```python
def monitor_api_health():
    """Continuous API health monitoring."""
    
    import time
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    while True:
        try:
            # Test health endpoint
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"API Health: {data.get('status')} - {data.get('message')}")
                
                # Check system metrics
                if 'data' in data:
                    images = data['data'].get('images_in_store', 0)
                    vectors = data['data'].get('vectors_in_index', 0)
                    logger.info(f"System Status: {images} images, {vectors} vectors")
            else:
                logger.warning(f"API Health Check Failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"API Health Check Error: {e}")
        
        # Wait before next check
        time.sleep(60)  # Check every minute
```

## Best Practices

### Test Development

1. **Comprehensive Coverage**: Test all endpoints and scenarios
2. **Error Handling**: Include error scenario testing
3. **Performance Metrics**: Measure response times consistently
4. **Data Validation**: Verify response format and content
5. **Automation**: Make tests runnable without manual intervention

### Test Execution

1. **Environment Setup**: Ensure virtual environment is activated
2. **Server Running**: Verify API server is accessible
3. **Dependencies**: Check all required packages are installed
4. **Clean State**: Start with fresh test data when possible
5. **Logging**: Use appropriate logging levels for debugging

## Troubleshooting

### Common Issues

1. **Connection Errors**: Check if API server is running
2. **Import Errors**: Verify virtual environment and Python path
3. **Timeout Issues**: Adjust timeout values for slow responses
4. **Response Format**: Check API response structure changes

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual endpoints
test_health_endpoint()
test_root_endpoint()
test_search_endpoint()

# Check specific error scenarios
test_connection_errors()
test_invalid_parameters()
```

## Dependencies

### Required Packages

- `requests`: HTTP client for API testing
- `json`: JSON data handling
- `time`: Timing and performance measurement
- `argparse`: Command-line argument parsing

### Installation

```bash
# Install dependencies
pip install requests

# Or use project requirements
pip install -r requirements.txt
```

The test API tool provides comprehensive testing capabilities for the Chromatica FastAPI endpoints and serves as both a development tool and a quality assurance framework for the API system.
