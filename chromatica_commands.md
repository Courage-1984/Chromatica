# 🚀 Chromatica Commands Reference

Quick reference for all essential commands and usage examples for the Chromatica color search engine.

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Index Building](#index-building)
- [API Server](#api-server)
- [Parallel Processing](#parallel-processing)
- [Testing & Validation](#testing--validation)
- [Development Tools](#development-tools)
- [Data Management](#data-management)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Environment Setup

### Virtual Environment

```bash
# Create virtual environment (Python 3.11)
python -m venv venv311

# Activate virtual environment
# Windows:
venv311\Scripts\activate

# Unix/macOS:
source venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Upgrade pip (if needed)
python -m pip install --upgrade pip
```

### Configuration Validation

```bash
# Validate project configuration
python -c "from src.chromatica.utils.config import validate_config; validate_config()"
```

---

## 🏗️ Index Building

### Basic Index Building

```bash
# Build index from small test dataset
# Note: IndexIVFPQ requires training step - handled automatically by build script
python scripts/build_index.py datasets/test-dataset-20

# Build index from medium dataset
python scripts/build_index.py datasets/test-dataset-200

# Build index from large dataset
python scripts/build_index.py datasets/test-dataset-5000
```

### IndexIVFPQ Parameters

```bash
# Check current IndexIVFPQ configuration
python -c "
from src.chromatica.utils.config import IVFPQ_NLIST, IVFPQ_M, IVFPQ_NBITS, IVFPQ_NPROBE
print(f'nlist (Voronoi cells): {IVFPQ_NLIST}')
print(f'M (subquantizers): {IVFPQ_M}')
print(f'nbits (bits per subquantizer): {IVFPQ_NBITS}')
print(f'nprobe (clusters to probe): {IVFPQ_NPROBE}')
print(f'Memory per vector: {IVFPQ_M * IVFPQ_NBITS / 8} bytes')
print(f'Compression ratio: {(1152 * 4) / (IVFPQ_M * IVFPQ_NBITS / 8):.1f}x')
"
```

### Memory Scaling Demonstration

```bash
# Demonstrate IndexIVFPQ memory benefits
python tools/demo_memory_scaling.py

# This script shows:
# - Memory usage comparison (HNSW vs IVFPQ)
# - Compression ratios and scaling benefits
# - Training requirement demonstration
# - Search functionality validation
```

### Evaluation and Testing

```bash
# Run comprehensive evaluation with test queries
python scripts/evaluate.py

# Run evaluation with custom query file
python scripts/evaluate.py --queries datasets/test-queries.json

# Run evaluation with ground truth for quality metrics
python scripts/evaluate.py --queries datasets/test-queries.json --ground-truth datasets/ground-truth.json

# Create sample test queries
python scripts/evaluate.py --create-sample-queries

# Run evaluation with custom parameters
python scripts/evaluate.py --k 20 --log-level DEBUG

# Help and all options
python scripts/evaluate.py --help
```

### Performance Metrics

```bash
# The evaluation harness measures:
# - P95 latency (target: <450ms)
# - Mean/median latency
# - Precision@10 and Recall@10 (with ground truth)
# - Memory usage during evaluation
# - Search quality metrics

# Results are saved to:
# - Console output with formatted results
# - logs/evaluation.log for detailed logs
# - logs/evaluation_results.json for raw data
```

### Advanced Index Building

```bash
# Custom output directory and batch size
python scripts/build_index.py datasets/test-dataset-5000 --output-dir ./production_index --batch-size 200

# Verbose logging for debugging
python scripts/build_index.py datasets/test-dataset-200 --verbose

# Help and all options
python scripts/build_index.py --help
```

### Index Management

```bash
# Check index files exist
ls -la index/
# Should show: chromatica_index.faiss, chromatica_metadata.db
```

---

## ⚡ Parallel Processing

### Parallel Search API

```bash
# Test parallel search endpoint
curl -X POST "http://localhost:8000/search/parallel" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {"colors": "FF0000", "weights": "1.0", "k": 10},
      {"colors": "00FF00", "weights": "1.0", "k": 10},
      {"colors": "0000FF", "weights": "1.0", "k": 10}
    ],
    "max_concurrent": 5
  }'

# Get performance statistics
curl "http://localhost:8000/performance/stats"
```

### Parallel Testing Tools

```bash
# Test parallel API capabilities
python tools/test_parallel_api.py

# Test specific parallel features
python tools/test_parallel_api.py --parallel-search
python tools/test_parallel_api.py --load-test
python tools/test_parallel_api.py --performance

# Run parallel search demonstration
python tools/demo_parallel_search.py

# Compare all methods (sequential, parallel, async, parallel API)
python tools/demo_parallel_search.py --compare

# Load testing with many queries
python tools/demo_parallel_search.py --load-test

# Custom parameters
python tools/demo_parallel_search.py --queries 20 --concurrent 8
```

### Enhanced API Testing

```bash
# Test all API features including parallel processing
python tools/test_api.py

# Test parallel request handling
python tools/test_api.py --parallel

# Test performance monitoring
python tools/test_api.py --performance

# Test specific endpoints
python tools/test_api.py --health
python tools/test_api.py --search
```

### Performance Monitoring

```bash
# Monitor performance in real-time
python -c "
import requests
import time
while True:
    try:
        response = requests.get('http://localhost:8000/performance/stats')
        stats = response.json()
        print(f'Total: {stats[\"total_searches\"]}, Concurrent: {stats[\"concurrent_searches\"]}, Avg: {stats[\"average_search_time\"]:.3f}s')
        time.sleep(5)
    except KeyboardInterrupt:
        break
"

# Check system health
curl "http://localhost:8000/health"
```

### Concurrent Request Examples

```bash
# Multiple concurrent individual requests
for i in {1..5}; do
  curl "http://localhost:8000/search?colors=FF0000&weights=1.0&k=5" &
done
wait

# Load testing with many requests
for i in {1..20}; do
  curl "http://localhost:8000/search?colors=00FF00&weights=1.0&k=10" &
done
wait
```

### Python Parallel Examples

```python
# Concurrent requests using ThreadPoolExecutor
import requests
from concurrent.futures import ThreadPoolExecutor

def search_request(query):
    return requests.get("http://localhost:8000/search", params=query)

queries = [
    {"colors": "FF0000", "weights": "1.0", "k": 10},
    {"colors": "00FF00", "weights": "1.0", "k": 10},
    {"colors": "0000FF", "weights": "1.0", "k": 10}
]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(search_request, queries))
```

```python
# Async requests using aiohttp
import asyncio
import aiohttp

async def async_search(session, query):
    async with session.get("http://localhost:8000/search", params=query) as response:
        return await response.json()

async def main():
    queries = [
        {"colors": "FF0000", "weights": "1.0", "k": 10},
        {"colors": "00FF00", "weights": "1.0", "k": 10}
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [async_search(session, query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results

# Run async searches
results = asyncio.run(main())
```

### Performance Tuning

```bash
# Monitor system resources during parallel processing
# Windows:
tasklist /fi "imagename eq python.exe"

# Unix/macOS:
ps aux | grep python

# Check memory usage
# Windows:
wmic process where name="python.exe" get processid,workingsetsize

# Unix/macOS:
ps -o pid,vsz,rss,comm -p $(pgrep python)
```

---

# Remove old index files
rm -rf index/
# or on Windows: rmdir /s index
```

---

## 🌐 API Server

### Local Development

```bash
# Start API server (development mode)
python -m src.chromatica.api.main

# Start with auto-reload
uvicorn src.chromatica.api.main:app --reload

# Start on specific port
uvicorn src.chromatica.api.main:app --port 8080
```

### Production Deployment

```bash
# Start production server
uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000

# With workers (production)
uvicorn src.chromatica.api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### External Access (ngrok)

```bash
# Install ngrok (if not already installed)
# Download from: https://ngrok.com/download

# Authenticate ngrok
ngrok authtoken YOUR_AUTH_TOKEN_HERE

# Start API server
python -m src.chromatica.api.main

# In another terminal, expose with ngrok
ngrok http 8000

# Use the provided URL (e.g., https://abc123.ngrok.io)
```

---

## 🧪 Testing & Validation

### System Validation

```bash
# Run comprehensive sanity checks
python scripts/run_sanity_checks.py

# Test histogram generation
python tools/test_histogram_generation.py

# Test saliency weighting functionality
python tools/test_saliency_weighting.py

# Test FAISS and DuckDB components
python tools/test_faiss_duckdb.py

# Test complete search system
python tools/test_search_system.py
```

### Performance Evaluation

```bash
# Run comprehensive evaluation with test queries
python scripts/evaluate.py

# Run evaluation with ground truth for quality metrics
python scripts/evaluate.py --queries datasets/test-queries.json --ground-truth datasets/ground-truth.json

# Create sample test queries
python scripts/evaluate.py --create-sample-queries

# Run evaluation with custom parameters
python scripts/evaluate.py --k 20 --log-level DEBUG

# Help and all options
python scripts/evaluate.py --help
```

### Sinkhorn Reranking System

```bash
# Validate reranking system
python -c "from src.chromatica.core.rerank import validate_reranking_system; print('Validation:', validate_reranking_system())"

# Test cost matrix generation
python -c "from src.chromatica.core.rerank import build_cost_matrix; cm = build_cost_matrix(); print(f'Cost matrix shape: {cm.shape}, Memory: {cm.nbytes/1024/1024:.1f}MB')"

# Test Sinkhorn distance computation
python -c "
import numpy as np
from src.chromatica.core.rerank import compute_sinkhorn_distance
h1 = np.random.random(1152); h1 = h1 / h1.sum()
h2 = np.random.random(1152); h2 = h2 / h2.sum()
dist = compute_sinkhorn_distance(h1, h2)
print(f'Sinkhorn distance: {dist:.6f}')
"

# Test candidate reranking
python -c "
import numpy as np
from src.chromatica.core.rerank import rerank_candidates
query = np.random.random(1152); query = query / query.sum()
candidates = [np.random.random(1152) for _ in range(5)]
candidates = [h / h.sum() for h in candidates]
ids = [f'img_{i}' for i in range(5)]
results = rerank_candidates(query, candidates, ids)
for r in results[:3]: print(f'Rank {r.rank}: {r.candidate_id} (dist: {r.distance:.6f})')
"
```

### API Testing

```bash
# Test API endpoints
python tools/test_api.py

# Test with curl (when server is running)
curl "http://localhost:8000/search?colors=FF0000,00FF00&weights=0.5,0.5&k=10"

# Test health endpoint
curl "http://localhost:8000/health"
```

### Performance Testing

```bash
# Test with different dataset sizes
python tools/test_histogram_generation.py --dataset datasets/test-dataset-20
python tools/test_histogram_generation.py --dataset datasets/test-dataset-200
python tools/test_histogram_generation.py --dataset datasets/test-dataset-5000
```

---

## 🛠️ Development Tools

### Visualization Tools

```bash
# Demo visualization features
python tools/demo_visualization.py

# Test color palette visualization
python tools/visualize_color_palettes.py

# Test search results visualization
python tools/visualize_search_results.py
```

### Query Processing

```bash
# Demo query processor
python tools/demo_query_processor.py

# Test search functionality
python tools/demo_search.py
```

### Debugging Tools

```bash
# Debug distance calculations
python tools/debug_distances.py

# Color explorer tool
python tools/color_explorer.py
```

---

## 📊 Data Management

### Dataset Operations

```bash
# List available datasets
ls datasets/

# Check dataset contents
ls datasets/test-dataset-20/
ls datasets/test-dataset-200/
ls datasets/test-dataset-5000/

# Count images in dataset
find datasets/test-dataset-5000 -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l
```

### Cleanup Operations

```bash
# Clean up output files
python tools/cleanup_outputs.py

# Clean specific output types
python tools/cleanup_outputs.py --histograms
python tools/cleanup_outputs.py --reports
python tools/cleanup_outputs.py --logs

# Clean all outputs (with confirmation)
python tools/cleanup_outputs.py --all --confirm

# Interactive cleanup
python tools/cleanup_outputs.py --interactive
```

### Log Management

```bash
# View recent logs
tail -f logs/build_index_*.log

# Clean old log files
python tools/cleanup_outputs.py --logs

# Check log directory
ls -la logs/
```

---

## 🔍 Troubleshooting

### Common Issues

```bash
# Check virtual environment is activated
which python
# Should show: /path/to/Chromatica/venv311/bin/python

# Verify dependencies
pip list | grep -E "(faiss|duckdb|opencv|scikit)"

# Check Python version
python --version
# Should be 3.10 or higher

# Validate configuration
python -c "from src.chromatica.utils.config import validate_config; validate_config()"
```

### Debug Mode

```bash
# Run with verbose logging
python scripts/build_index.py datasets/test-dataset-20 --verbose

# Test individual components
python -c "from src.chromatica.core.histogram import build_histogram; print('Histogram module OK')"
python -c "from src.chromatica.indexing.store import AnnIndex; print('Store module OK')"
```

### Performance Monitoring

```bash
# Monitor system resources during indexing
# On Windows: Task Manager
# On Unix: htop or top

# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Monitor disk space
df -h  # Unix
dir     # Windows
```

---

## 🌐 Web Interface

### Access Points

```bash
# Local development
http://localhost:8000/

# API documentation
http://localhost:8000/docs

# Health check
http://localhost:8000/health

# Search endpoint
http://localhost:8000/search?colors=FF0000,00FF00&weights=0.5,0.5&k=10
```

### Browser Testing

```bash
# Test in different browsers
# Chrome: http://localhost:8000/
# Firefox: http://localhost:8000/
# Safari: http://localhost:8000/

# Test responsive design
# Resize browser window to test mobile layout
```

---

## 📈 Performance Benchmarks

### Expected Performance

```bash
# Index building times (approximate)
# 20 images: 5-10 seconds
# 200 images: 1-2 minutes
# 5000 images: 10-20 minutes

# Search response times
# Single query: 150-500ms
# Batch queries: 1-5 seconds

# Memory usage
# Runtime: 100-500MB
# Index storage: 100-500MB
```

### Optimization Commands

```bash
# Optimize batch size for your system
python scripts/build_index.py datasets/test-dataset-200 --batch-size 50  # Smaller batches
python scripts/build_index.py datasets/test-dataset-200 --batch-size 200 # Larger batches

# Monitor performance
python scripts/build_index.py datasets/test-dataset-200 --verbose
```

---

## 🚨 Emergency Commands

### Reset Environment

```bash
# Deactivate and recreate virtual environment
deactivate
rm -rf venv311
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
```

### Clean Slate

```bash
# Remove all generated files
python tools/cleanup_outputs.py --all --confirm
rm -rf index/
rm -rf test_index/

# Rebuild everything
python scripts/build_index.py datasets/test-dataset-20
```

### Quick Health Check

```bash
# One-command system check
python scripts/run_sanity_checks.py && echo "✅ System OK" || echo "❌ System Issues"
```

---

## 📚 Additional Resources

### Documentation

- **Complete Guide**: `docs/complete_usage_guide.md`
- **API Reference**: `docs/api_reference.md`
- **Troubleshooting**: `docs/troubleshooting.md`
- **Project Architecture**: `docs/project_architecture.md`
- **Sinkhorn Reranking**: `docs/sinkhorn_reranking_logic.md`

### Help Commands

```bash
# Get help for any script
python scripts/build_index.py --help
python tools/cleanup_outputs.py --help
python scripts/run_sanity_checks.py --help

# Check script status
python -c "import sys; print('Python:', sys.version)"
python -c "import src.chromatica; print('Chromatica modules loaded successfully')"
```

---

**💡 Tip**: Bookmark this file for quick reference during development!

**🔄 Last Updated**: December 2024  
**📝 Maintained By**: Chromatica Development Team
