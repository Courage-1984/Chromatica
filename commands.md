# 🚀 Chromatica Commands Reference

Quick reference for all essential commands and usage examples for the Chromatica color search engine.

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Index Building](#index-building)
- [API Server](#api-server)
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
python scripts/build_index.py datasets/test-dataset-20

# Build index from medium dataset
python scripts/build_index.py datasets/test-dataset-200

# Build index from large dataset
python scripts/build_index.py datasets/test-dataset-5000
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

# Test FAISS and DuckDB components
python tools/test_faiss_duckdb.py

# Test complete search system
python tools/test_search_system.py
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