# Chromatica Parallel Processing Guide

## Overview

The Chromatica color search engine has been enhanced with comprehensive parallel processing capabilities to support high-throughput, concurrent operations. This guide covers all parallel processing features, performance optimizations, and usage patterns.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Parallel Processing Features](#parallel-processing-features)
3. [API Endpoints](#api-endpoints)
4. [Performance Monitoring](#performance-monitoring)
5. [Testing Tools](#testing-tools)
6. [Usage Examples](#usage-examples)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

## Architecture Overview

### Thread Pool Management

The API uses a configurable thread pool for CPU-intensive operations:

- **Default Workers**: `min(32, (os.cpu_count() or 1) + 4)`
- **Thread Naming**: `chromatica-{worker_id}`
- **Lifecycle**: Initialized on startup, shutdown on application exit
- **Resource Management**: Automatic cleanup and proper shutdown

### Async/Await Support

All search operations are fully asynchronous:

- **Non-blocking I/O**: Database and index operations run in thread pool
- **Concurrent Requests**: Multiple searches can run simultaneously
- **Resource Isolation**: Each request is processed independently

### Performance Monitoring

Real-time performance tracking includes:

- **Search Counters**: Total searches, concurrent searches, max concurrent
- **Timing Metrics**: Average search time, recent search times
- **Thread Safety**: All statistics updates are thread-safe

## Parallel Processing Features

### 1. Concurrent Search Requests

Multiple individual search requests can be processed simultaneously:

```python
# Multiple concurrent requests
import asyncio
import aiohttp

async def concurrent_searches():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for query in queries:
            task = session.get(f"{BASE_URL}/search", params=query)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
```

### 2. Parallel Search API

Dedicated endpoint for batch parallel processing:

```python
# Parallel search request
request_data = {
    "queries": [
        {"colors": "FF0000", "weights": "1.0", "k": 10},
        {"colors": "00FF00", "weights": "1.0", "k": 10},
        {"colors": "0000FF", "weights": "1.0", "k": 10}
    ],
    "max_concurrent": 5
}

response = requests.post(f"{BASE_URL}/search/parallel", json=request_data)
```

### 3. Performance Statistics

Real-time monitoring of system performance:

```python
# Get performance stats
response = requests.get(f"{BASE_URL}/performance/stats")
stats = response.json()

print(f"Total searches: {stats['total_searches']}")
print(f"Concurrent searches: {stats['concurrent_searches']}")
print(f"Average search time: {stats['average_search_time']:.3f}s")
```

## API Endpoints

### Standard Search Endpoint

**GET** `/search`

Enhanced with async processing and performance tracking.

**Parameters:**
- `colors`: Comma-separated hex color codes
- `weights`: Comma-separated weights
- `k`: Number of results (1-200)
- `fast_mode`: Use fast approximate reranking
- `batch_size`: Batch size for reranking (1-50)

**Response:**
```json
{
  "query_id": "uuid",
  "query": {
    "colors": ["FF0000"],
    "weights": [1.0]
  },
  "results_count": 10,
  "results": [...],
  "metadata": {
    "ann_time_ms": 45,
    "rerank_time_ms": 120,
    "total_time_ms": 165,
    "index_size": 5000
  }
}
```

### Parallel Search Endpoint

**POST** `/search/parallel`

Process multiple search queries concurrently.

**Request Body:**
```json
{
  "queries": [
    {
      "colors": "FF0000",
      "weights": "1.0",
      "k": 10,
      "fast_mode": false,
      "batch_size": 5
    }
  ],
  "max_concurrent": 10
}
```

**Response:**
```json
{
  "total_queries": 3,
  "successful_queries": 3,
  "failed_queries": 0,
  "total_time_ms": 450,
  "results": [
    {
      "query_id": "parallel_query_0_abc123",
      "success": true,
      "results": [...],
      "results_count": 10
    }
  ]
}
```

### Performance Statistics Endpoint

**GET** `/performance/stats`

Get real-time performance metrics.

**Response:**
```json
{
  "total_searches": 1250,
  "concurrent_searches": 3,
  "max_concurrent_searches": 15,
  "average_search_time": 0.245,
  "recent_search_times": [0.234, 0.256, 0.198, ...]
}
```

## Performance Monitoring

### Metrics Tracked

1. **Search Counters**
   - Total searches performed
   - Current concurrent searches
   - Maximum concurrent searches reached

2. **Timing Metrics**
   - Average search time (rolling average)
   - Recent search times (last 1000 searches)
   - Individual search timing

3. **System Health**
   - Thread pool utilization
   - Memory usage patterns
   - Error rates

### Monitoring Tools

#### Real-time Statistics

```python
import requests
import time

def monitor_performance(duration=60):
    """Monitor performance for specified duration."""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        response = requests.get(f"{BASE_URL}/performance/stats")
        stats = response.json()
        
        print(f"Concurrent: {stats['concurrent_searches']}, "
              f"Avg time: {stats['average_search_time']:.3f}s")
        
        time.sleep(5)
```

#### Performance Comparison

```python
def compare_methods(queries):
    """Compare sequential vs parallel performance."""
    # Sequential
    start = time.time()
    for query in queries:
        requests.get(f"{BASE_URL}/search", params=query)
    sequential_time = time.time() - start
    
    # Parallel
    start = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(requests.get, f"{BASE_URL}/search", params=q) 
                  for q in queries]
        [f.result() for f in futures]
    parallel_time = time.time() - start
    
    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")
```

## Testing Tools

### 1. Parallel API Test Suite

**File:** `tools/test_parallel_api.py`

Comprehensive testing of parallel processing capabilities:

```bash
# Run all parallel tests
python tools/test_parallel_api.py

# Test specific components
python tools/test_parallel_api.py --parallel-search
python tools/test_parallel_api.py --load-test
python tools/test_parallel_api.py --performance
```

**Test Categories:**
- Parallel search endpoint testing
- Concurrent request handling
- Performance monitoring
- Load testing (20+ concurrent requests)
- Async request processing

### 2. Enhanced API Test Suite

**File:** `tools/test_api.py`

Updated with parallel testing capabilities:

```bash
# Run all tests including parallel
python tools/test_api.py

# Test parallel features only
python tools/test_api.py --parallel
python tools/test_api.py --performance
```

### 3. Parallel Search Demo

**File:** `tools/demo_parallel_search.py`

Interactive demonstration of parallel capabilities:

```bash
# Run demonstration
python tools/demo_parallel_search.py

# Compare all methods
python tools/demo_parallel_search.py --compare

# Load test
python tools/demo_parallel_search.py --load-test

# Custom parameters
python tools/demo_parallel_search.py --queries 20 --concurrent 8
```

## Usage Examples

### Basic Parallel Processing

```python
import requests
from concurrent.futures import ThreadPoolExecutor

def search_parallel(queries, max_workers=5):
    """Run multiple searches in parallel."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for query in queries:
            future = executor.submit(requests.get, f"{BASE_URL}/search", params=query)
            futures.append(future)
        
        results = [future.result() for future in futures]
        return results

# Usage
queries = [
    {"colors": "FF0000", "weights": "1.0", "k": 10},
    {"colors": "00FF00", "weights": "1.0", "k": 10},
    {"colors": "0000FF", "weights": "1.0", "k": 10}
]

results = search_parallel(queries, max_workers=3)
```

### Async Processing

```python
import asyncio
import aiohttp

async def search_async(queries):
    """Run searches asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for query in queries:
            task = session.get(f"{BASE_URL}/search", params=query)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

# Usage
queries = [{"colors": "FF0000", "weights": "1.0", "k": 10}]
results = await search_async(queries)
```

### Batch Parallel Processing

```python
import requests

def batch_parallel_search(queries, max_concurrent=10):
    """Use the parallel API endpoint for batch processing."""
    request_data = {
        "queries": queries,
        "max_concurrent": max_concurrent
    }
    
    response = requests.post(f"{BASE_URL}/search/parallel", json=request_data)
    return response.json()

# Usage
queries = [
    {"colors": "FF0000", "weights": "1.0", "k": 10, "fast_mode": False},
    {"colors": "00FF00", "weights": "1.0", "k": 10, "fast_mode": True},
    {"colors": "0000FF", "weights": "1.0", "k": 10, "fast_mode": False}
]

results = batch_parallel_search(queries, max_concurrent=5)
print(f"Processed {results['total_queries']} queries in {results['total_time_ms']}ms")
```

### Performance Monitoring

```python
import requests
import time

def monitor_performance():
    """Monitor API performance in real-time."""
    while True:
        try:
            response = requests.get(f"{BASE_URL}/performance/stats")
            stats = response.json()
            
            print(f"Total: {stats['total_searches']}, "
                  f"Concurrent: {stats['concurrent_searches']}, "
                  f"Avg time: {stats['average_search_time']:.3f}s")
            
            time.sleep(10)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

# Usage
monitor_performance()
```

## Performance Tuning

### Thread Pool Configuration

The thread pool is automatically configured based on system resources:

```python
# In src/chromatica/api/main.py
max_workers = min(32, (os.cpu_count() or 1) + 4)
thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="chromatica")
```

**Tuning Guidelines:**
- **CPU-bound tasks**: Use `os.cpu_count()` workers
- **I/O-bound tasks**: Use `os.cpu_count() * 2-4` workers
- **Memory constraints**: Reduce workers if memory usage is high
- **Network latency**: Increase workers for high-latency scenarios

### Concurrency Limits

**Parallel Search API:**
- Maximum concurrent queries: 50
- Recommended: 5-20 depending on system resources

**Individual Requests:**
- No hard limit (limited by thread pool)
- Recommended: 10-50 concurrent requests

### Memory Management

**Histogram Caching:**
- Cache size: 1000 histograms (configurable)
- LRU eviction policy
- Thread-safe operations

**Performance Stats:**
- Rolling window: 1000 recent search times
- Automatic cleanup of old data

## Troubleshooting

### Common Issues

#### 1. High Memory Usage

**Symptoms:**
- Memory usage increases over time
- System becomes slow
- Out of memory errors

**Solutions:**
- Reduce thread pool size
- Decrease histogram cache size
- Monitor memory usage patterns

#### 2. Poor Performance

**Symptoms:**
- Slow search times
- High CPU usage
- Timeout errors

**Solutions:**
- Check thread pool utilization
- Monitor concurrent request count
- Optimize query parameters
- Use fast_mode for approximate results

#### 3. Connection Errors

**Symptoms:**
- Connection refused errors
- Timeout errors
- Incomplete responses

**Solutions:**
- Check server status
- Verify API endpoint availability
- Increase timeout values
- Check network connectivity

### Debugging Tools

#### 1. Performance Statistics

```python
# Check current performance
response = requests.get(f"{BASE_URL}/performance/stats")
stats = response.json()
print(json.dumps(stats, indent=2))
```

#### 2. Health Check

```python
# Check system health
response = requests.get(f"{BASE_URL}/health")
health = response.json()
print(f"Status: {health['status']}")
print(f"Components: {health['components']}")
```

#### 3. Load Testing

```bash
# Run load test
python tools/demo_parallel_search.py --load-test

# Test with specific parameters
python tools/demo_parallel_search.py --queries 50 --concurrent 20
```

### Performance Optimization

#### 1. Query Optimization

- Use `fast_mode=True` for approximate results
- Optimize `batch_size` for reranking
- Limit result count (`k`) when possible

#### 2. Concurrency Tuning

- Start with 5-10 concurrent requests
- Monitor performance statistics
- Adjust based on system resources

#### 3. Resource Management

- Monitor memory usage
- Check thread pool utilization
- Optimize cache sizes

## Best Practices

### 1. Request Patterns

- **Batch similar queries** using the parallel API
- **Use appropriate concurrency levels** (5-20 for most cases)
- **Monitor performance** and adjust accordingly

### 2. Error Handling

- **Implement retry logic** for failed requests
- **Handle timeouts** gracefully
- **Log errors** for debugging

### 3. Performance Monitoring

- **Track key metrics** (response times, success rates)
- **Set up alerts** for performance degradation
- **Regular load testing** to validate performance

### 4. Resource Management

- **Monitor memory usage** patterns
- **Clean up resources** properly
- **Use connection pooling** for high-throughput scenarios

## Conclusion

The Chromatica parallel processing implementation provides significant performance improvements for concurrent search operations. By following the guidelines in this document, you can effectively utilize these capabilities while maintaining system stability and optimal performance.

For additional support or questions, refer to the troubleshooting section or consult the API documentation.

