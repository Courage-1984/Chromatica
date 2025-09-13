# 🔧 Chromatica Comprehensive Troubleshooting Guide

This guide addresses all common issues and provides step-by-step solutions for the Chromatica color search engine.

## 📋 Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Setup & Installation Issues](#setup--installation-issues)
3. [Index Building Problems](#index-building-problems)
4. [API Server Issues](#api-server-issues)
5. [Web Interface Problems](#web-interface-problems)
6. [Search & Visualization Errors](#search--visualization-errors)
7. [Performance Issues](#performance-issues)
8. [Common Error Messages](#common-error-messages)
9. [Debug Mode](#debug-mode)
10. [Getting Help](#getting-help)

## 🚨 Quick Diagnosis

### **System Status Check**

```bash
# 1. Check if virtual environment is activated
echo $VIRTUAL_ENV  # Should show path to venv311

# 2. Check if index files exist
ls -la test_index/
# Should show: chromatica_index.faiss, chromatica_metadata.db

# 3. Check if server is running
curl http://localhost:8000/api/info
# Should return: {"status": "ready", ...}

# 4. Check web interface
# Open: http://localhost:8000/
# Status should show: "✅ System Ready"
```

### **Common Symptoms & Solutions**

| Symptom                             | Likely Cause           | Quick Fix                                   |
| ----------------------------------- | ---------------------- | ------------------------------------------- |
| "Search components not initialized" | Index not built        | Build index with `build_index.py`           |
| "Failed to generate visualization"  | Matplotlib/PIL issues  | Check package installation                  |
| "ImportError: relative import"      | Wrong execution method | Use `python -m src.chromatica.api.main`     |
| "Port already in use"               | Another server running | Use different port or kill existing process |
| "System Not Ready"                  | Index files missing    | Rebuild index or check file paths           |

## 🔧 Setup & Installation Issues

### **Issue: Python Version Problems**

```bash
# Check Python version
python --version

# If < 3.10, install Python 3.10+
# Windows: Download from python.org
# macOS: brew install python@3.10
# Linux: sudo apt install python3.10
```

### **Issue: Virtual Environment Not Working**

```bash
# Windows - Create new environment
python -m venv venv311
venv311\Scripts\activate

# macOS/Linux - Create new environment
python3 -m venv venv311
source venv311/bin/activate

# Verify activation
which python  # Should point to venv311/bin/python
python --version  # Should show Python 3.10+
```

### **Issue: Package Installation Failures**

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install packages one by one if bulk install fails
pip install numpy
pip install opencv-python
pip install scikit-image
pip install faiss-cpu
pip install duckdb
pip install fastapi
pip install uvicorn
pip install matplotlib
pip install pillow

# Or try with specific versions
pip install -r requirements.txt --no-cache-dir
```

### **Issue: Import Errors After Installation**

```bash
# Check if packages are in the right environment
pip list | grep -E "(numpy|opencv|skimage|faiss|duckdb|fastapi)"

# If empty, reactivate virtual environment
deactivate
venv311\Scripts\activate  # Windows
# source venv311/bin/activate  # macOS/Linux

# Reinstall packages
pip install -r requirements.txt
```

## 🏗️ Index Building Problems

### **Issue: "unrecognized arguments: --input-dir"**

```bash
# ❌ WRONG - This won't work
python scripts/build_index.py --input-dir datasets/test-dataset-20

# ✅ CORRECT - Use positional argument
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index
```

### **Issue: "one of the arguments --image/-i --directory/-d is required"**

```bash
# ❌ WRONG - This won't work
python tools/test_histogram_generation.py --input-dir datasets/test-dataset-20

# ✅ CORRECT - Use --directory or -d
python tools/test_histogram_generation.py --directory datasets/test-dataset-20
```

### **Issue: Index Building Fails with Memory Errors**

```bash
# For large datasets, use smaller ones first
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index

# If successful, try larger dataset
python scripts/build_index.py datasets/test-dataset-50 --output-dir test_index

# Monitor memory usage
# Windows: Task Manager
# macOS: Activity Monitor
# Linux: htop or top
```

### **Issue: "No images found in directory"**

```bash
# Check if directory exists and contains images
ls -la datasets/test-dataset-20/

# Check supported image formats
find datasets/test-dataset-20/ -name "*.jpg" -o -name "*.png" -o -name "*.jpeg"

# Verify image files are not corrupted
python -c "
from PIL import Image
import os
for file in os.listdir('datasets/test-dataset-20/'):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        try:
            img = Image.open(f'datasets/test-dataset-20/{file}')
            print(f'✅ {file}: {img.size}')
        except Exception as e:
            print(f'❌ {file}: {e}')
"
```

### **Issue: Index Files Not Created**

```bash
# Check if build script completed successfully
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index

# Look for these files:
ls -la test_index/
# Should show:
# -rw-r--r-- 1 user user 1234567 date chromatica_index.faiss
# -rw-r--r-- 1 user user 123456 date chromatica_metadata.db

# Check file sizes (should not be 0 bytes)
du -h test_index/*
```

## 🖥️ API Server Issues

### **Issue: "ImportError: attempted relative import with no known parent package"**

```bash
# ❌ WRONG - This causes import errors
python src/chromatica/api/main.py

# ✅ CORRECT - Run as module
python -m src.chromatica.api.main

# Alternative methods
uvicorn src.chromatica.api.main:app --reload
# or
cd src && python -m chromatica.api.main
```

### **Issue: Server Won't Start**

```bash
# Check if port 8000 is in use
# Windows
netstat -an | findstr :8000

# macOS/Linux
lsof -i :8000

# Kill existing process if needed
# Windows
taskkill /PID <PID> /F

# macOS/Linux
kill -9 <PID>

# Use different port
uvicorn src.chromatica.api.main:app --port 8001
```

### **Issue: "Search components not initialized"**

```bash
# This means the index files are missing or corrupted
# Solution: Rebuild the index

# 1. Stop the server (Ctrl+C)
# 2. Build the index
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index

# 3. Verify files exist
ls -la test_index/

# 4. Restart the server
python -m src.chromatica.api.main
```

### **Issue: Server Starts But API Endpoints Fail**

```bash
# Check server logs for specific errors
# Look for lines starting with ERROR or WARNING

# Test basic connectivity
curl http://localhost:8000/

# Test API info endpoint
curl http://localhost:8000/api/info

# Check if index files are accessible
python -c "
import os
print(f'Index exists: {os.path.exists(\"test_index/chromatica_index.faiss\")}')
print(f'DB exists: {os.path.exists(\"test_index/chromatica_metadata.db\")}')
"
```

## 🌐 Web Interface Problems

### **Issue: "System Not Ready" Status**

```bash
# This means the search components aren't initialized
# Check the browser console for errors (F12 → Console)

# Verify index files exist
ls -la test_index/

# Check API status
curl http://localhost:8000/api/info

# If status is not "ready", rebuild the index
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index
```

### **Issue: "Cannot Connect" Status**

```bash
# This means the API server isn't running or accessible

# 1. Check if server is running
curl http://localhost:8000/api/info

# 2. If server isn't running, start it
python -m src.chromatica.api.main

# 3. Check firewall settings
# Windows: Windows Defender Firewall
# macOS: System Preferences → Security & Privacy → Firewall
# Linux: ufw or iptables

# 4. Try different host binding
uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000
```

### **Issue: Search Button Disabled**

```bash
# This happens when the system status check fails

# 1. Check browser console for JavaScript errors
# 2. Verify API endpoints are working
curl http://localhost:8000/api/info

# 3. Check if CORS is blocking requests
# Look for CORS errors in browser console

# 4. Try refreshing the page
# 5. Clear browser cache and cookies
```

### **Issue: "Failed to generate visualization"**

```bash
# This usually means matplotlib or PIL issues

# 1. Check package installation
python -c "
try:
    import matplotlib.pyplot as plt
    print('✅ matplotlib working')
except ImportError as e:
    print(f'❌ matplotlib error: {e}')
"

python -c "
try:
    from PIL import Image
    print('✅ PIL working')
except ImportError as e:
    print(f'❌ PIL error: {e}')
"

# 2. Reinstall packages if needed
pip install matplotlib pillow --force-reinstall

# 3. Check matplotlib backend
python -c "
import matplotlib
print(f'Backend: {matplotlib.get_backend()}')
# Should show 'Agg' for server use
"
```

### **Issue: "Failed to generate collage"**

```bash
# This usually means image loading or processing issues

# 1. Check if result images exist
ls -la datasets/test-dataset-20/

# 2. Test image loading manually
python -c "
from PIL import Image
import os
for file in os.listdir('datasets/test-dataset-20/')[:3]:
    if file.lower().endswith(('.jpg', '.png')):
        try:
            img = Image.open(f'datasets/test-dataset-20/{file}')
            print(f'✅ {file}: {img.size}')
        except Exception as e:
            print(f'❌ {file}: {e}')
"

# 3. Check file permissions
ls -la datasets/test-dataset-20/

# 4. Verify image paths in metadata
python -c "
import duckdb
conn = duckdb.connect('test_index/chromatica_metadata.db')
result = conn.execute('SELECT image_id, file_path FROM images LIMIT 5').fetchall()
for row in result:
    print(f'{row[0]}: {row[1]}')
conn.close()
"
```

## 🔍 Search & Visualization Errors

### **Issue: "Search failed: can't access property 'total_time_ms', data.metadata is undefined"**

```bash
# This JavaScript error occurs when the search response is malformed

# 1. Check the actual API response
curl "http://localhost:8000/search?colors=FF0000&weights=1.0&k=5"

# 2. Look for server-side errors in the terminal running the API

# 3. Check if the search components are properly initialized
curl http://localhost:8000/api/info

# 4. Verify the index files are valid
ls -la test_index/
du -h test_index/*

# 5. Test the search functionality directly
python -c "
import sys
sys.path.insert(0, 'src')
from chromatica.search import find_similar
try:
    result = find_similar(['FF0000'], [1.0], k=5)
    print(f'✅ Search working: {len(result[\"results\"])} results')
except Exception as e:
    print(f'❌ Search error: {e}')
"
```

### **Issue: "503 Service Unavailable"**

```bash
# This means the search system is not ready

# 1. Check server logs for the specific error
# Look for lines containing "Search components not initialized"

# 2. Verify index files exist and are accessible
ls -la test_index/
python -c "
import os
print(f'Index readable: {os.access(\"test_index/chromatica_index.faiss\", os.R_OK)}')
print(f'DB readable: {os.access(\"test_index/chromatica_metadata.db\", os.R_OK)}')
"

# 3. Rebuild the index if files are missing or corrupted
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index

# 4. Restart the server
# Stop with Ctrl+C, then restart
python -m src.chromatica.api.main
```

### **Issue: Search Returns No Results**

```bash
# This can happen for several reasons

# 1. Check if the query histogram is valid
python -c "
import sys
sys.path.insert(0, 'src')
from chromatica.core.query import create_query_histogram, validate_query_histogram
try:
    hist = create_query_histogram(['FF0000'], [1.0])
    valid = validate_query_histogram(hist)
    print(f'✅ Query histogram valid: {valid}')
    print(f'Histogram shape: {hist.shape}')
    print(f'Sum: {hist.sum():.6f}')
except Exception as e:
    print(f'❌ Query error: {e}')
"

# 2. Check if the index contains data
python -c "
import duckdb
conn = duckdb.connect('test_index/chromatica_metadata.db')
count = conn.execute('SELECT COUNT(*) FROM images').fetchone()[0]
print(f'Images in index: {count}')
conn.close()
"

# 3. Test with a simple color that should exist
# Try searching for a very common color like white or black
curl "http://localhost:8000/search?colors=FFFFFF&weights=1.0&k=5"
```

## ⚡ Performance Issues

### **Issue: Slow Index Building**

```bash
# Index building can be slow for large datasets

# 1. Use smaller datasets for development
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index

# 2. Monitor system resources
# Windows: Task Manager → Performance
# macOS: Activity Monitor
# Linux: htop

# 3. Check if other processes are consuming resources
# Close unnecessary applications

# 4. Consider using SSD storage for faster I/O
```

### **Issue: Slow Search Response**

```bash
# Search should be fast (< 500ms)

# 1. Check search timing in the response
curl "http://localhost:8000/search?colors=FF0000&weights=1.0&k=5"

# 2. Monitor server performance
# Look for high CPU or memory usage

# 3. Check if the FAISS index is optimized
python -c "
import faiss
index = faiss.read_index('test_index/chromatica_index.faiss')
print(f'Index type: {type(index)}')
print(f'Index size: {index.ntotal}')
print(f'Index dimension: {index.d}')
"

# 4. Reduce result count (k) for faster response
curl "http://localhost:8000/search?colors=FF0000&weights=1.0&k=3"
```

### **Issue: Memory Usage Too High**

```bash
# Monitor memory usage during operations

# 1. Check current memory usage
# Windows: Task Manager → Memory
# macOS: Activity Monitor → Memory
# Linux: free -h

# 2. Use smaller datasets for development
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index

# 3. Close other applications to free memory

# 4. Check for memory leaks
# Restart the server periodically during development
```

## ❌ Common Error Messages

### **"ModuleNotFoundError: No module named 'chromatica'"**

```bash
# Solution: Add src to Python path or run from correct location
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# or
python -m src.chromatica.api.main
```

### **"Permission denied"**

```bash
# Solution: Check file permissions
ls -la test_index/
chmod 644 test_index/*  # Linux/macOS
# Windows: Right-click → Properties → Security
```

### **"Address already in use"**

```bash
# Solution: Kill existing process or use different port
lsof -i :8000  # Find process using port 8000
kill -9 <PID>  # Kill the process
# or
uvicorn src.chromatica.api.main:app --port 8001
```

### **"No such file or directory"**

```bash
# Solution: Check if files exist and paths are correct
pwd  # Current directory
ls -la  # List files
find . -name "*.faiss"  # Find FAISS index files
```

## 🐛 Debug Mode

### **Enable Verbose Logging**

```bash
# Set environment variables for debug mode
export LOG_LEVEL=DEBUG
export CHROMATICA_DEBUG=true

# Windows
set LOG_LEVEL=DEBUG
set CHROMATICA_DEBUG=true

# Start server with debug logging
python -m src.chromatica.api.main
```

### **Check Logs**

```bash
# View log files
tail -f logs/*.log

# Check specific log files
cat logs/sanity_checks.log
cat logs/build_index_*.log
```

### **Test Individual Components**

```bash
# Test histogram generation
python tools/test_histogram_generation.py --directory datasets/test-dataset-20

# Test FAISS and DuckDB
python tools/test_faiss_duckdb.py

# Test search system
python tools/test_search_system.py

# Test visualization
python tools/demo_visualization.py
```

### **Interactive Debugging**

```bash
# Start Python with project modules available
cd src
python -i -c "
import chromatica.core.histogram as hist
import chromatica.core.query as query
import chromatica.indexing.faiss_wrapper as faiss_wrap
print('Modules loaded successfully')
"

# Test functions interactively
>>> hist.generate_histogram('datasets/test-dataset-20/7348262.jpg')
>>> query.create_query_histogram(['FF0000'], [1.0])
```

## 🆘 Getting Help

### **Self-Service Troubleshooting**

1. **Check this guide** for your specific error
2. **Run sanity checks**: `python scripts/run_sanity_checks.py`
3. **Check logs**: Look in `logs/` directory
4. **Test components**: Use tools in `tools/` directory
5. **Verify setup**: Follow the complete usage guide

### **Information to Collect**

When seeking help, provide:

- **Error message**: Exact text of the error
- **Command used**: What you typed to cause the error
- **System info**: OS, Python version, virtual environment status
- **Current state**: What you've already tried
- **Logs**: Relevant log file contents

### **When to Seek Help**

- **System won't start** after following all troubleshooting steps
- **Unexpected errors** not covered in this guide
- **Performance issues** that can't be resolved
- **Feature requests** or enhancement ideas

---

## 🎯 Quick Fix Checklist

If you're having issues, try this sequence:

```bash
# 1. Verify environment
venv311\Scripts\activate  # Windows
# source venv311/bin/activate  # macOS/Linux
python --version  # Should show 3.10+

# 2. Check index files
ls -la test_index/
# Should show: chromatica_index.faiss, chromatica_metadata.db

# 3. If missing, rebuild index
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index

# 4. Start server
python -m src.chromatica.api.main

# 5. Test API
curl http://localhost:8000/api/info
# Should return: {"status": "ready", ...}

# 6. Open web interface
# Navigate to: http://localhost:8000/
# Status should show: "✅ System Ready"
```

**Most issues can be resolved by following this checklist!** 🎨✨

---

_For additional help, see the complete usage guide and project documentation._
