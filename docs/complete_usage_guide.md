# 🎨 Chromatica Color Search Engine - Complete Usage Guide

This comprehensive guide covers everything you need to know to set up, run, and use the Chromatica color search engine project.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Initial Project Setup](#initial-project-setup)
4. [Building the Search Index](#building-the-search-index)
5. [Running the API Server](#running-the-api-server)
6. [Using the Web Interface](#using-the-web-interface)
7. [API Usage Examples](#api-usage-examples)
8. [Testing & Validation](#testing--validation)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)
11. [Development Workflow](#development-workflow)

## 🎯 Project Overview

Chromatica is a sophisticated color-based image search engine that uses:

- **CIE Lab color space** for perceptual accuracy
- **8x12x12 histogram binning** (1,152 dimensions)
- **FAISS HNSW index** for fast approximate search
- **Sinkhorn-EMD reranking** for high-quality results
- **Weighted multi-color queries** with customizable importance
- **Visual query representations** and result collages

## 🔧 Prerequisites & Setup

### **System Requirements**

- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.10 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: 2GB+ free space for datasets and indices

### **Required Software**

```bash
# Python 3.10+ (check with: python --version)
# Git (for cloning the repository)
# A modern web browser (Chrome, Firefox, Safari, Edge)
```

## 🚀 Initial Project Setup

### **Step 1: Clone and Navigate**

```bash
# Clone the repository
git clone <your-repo-url>
cd Chromatica

# Verify the structure
ls -la
```

### **Step 2: Create Virtual Environment**

```bash
# Windows
python -m venv venv311
venv311\Scripts\activate

# macOS/Linux
python3 -m venv venv311
source venv311/bin/activate

# Verify activation
python --version  # Should show Python 3.10+
which python     # Should point to venv311
```

### **Step 3: Install Dependencies**

```bash
# Ensure pip is up to date
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify key packages
python -c "import opencv-python, skimage, numpy, faiss, duckdb, fastapi, matplotlib, pillow; print('All packages installed successfully!')"
```

### **Step 4: Verify Dataset Structure**

```bash
# Check available datasets
ls -la datasets/

# Expected structure:
# datasets/
# ├── test-dataset-20/     # 20 images for quick testing
# ├── test-dataset-50/     # 50 images for validation
# ├── test-dataset-200/    # 200 images for performance testing
# └── test-dataset-5000/   # 5000 images for production testing
```

## 🏗️ Building the Search Index

### **Step 1: Choose Your Dataset**

```bash
# For development and testing (recommended to start)
DATASET_PATH="datasets/test-dataset-20"

# For validation
# DATASET_PATH="datasets/test-dataset-50"

# For performance testing
# DATASET_PATH="datasets/test-dataset-200"

# For production-scale testing
# DATASET_PATH="datasets/test-dataset-5000"
```

### **Step 2: Build the Index**

```bash
# Ensure virtual environment is activated
venv311\Scripts\activate  # Windows
# source venv311/bin/activate  # macOS/Linux

# Build the search index
python scripts/build_index.py $DATASET_PATH --output-dir test_index
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index
python scripts/build_index.py datasets/test-dataset-200 --output-dir test_index

# Expected output:
# ✅ Successfully built FAISS index with 20 images
# ✅ Successfully built DuckDB metadata store
# 📁 Index files saved to: test_index/
```

### **Step 3: Verify Index Creation**

```bash
# Check index files
ls -la test_index/

# Expected files:
# test_index/
# ├── chromatica_index.faiss      # FAISS HNSW index
# └── chromatica_metadata.db      # DuckDB metadata store
```

## 🖥️ Running the API Server

### **Step 1: Start the Server**

```bash
# Ensure virtual environment is activated
venv311\Scripts\activate  # Windows
# source venv311/bin/activate  # macOS/Linux

# Start the FastAPI server
python -m src.chromatica.api.main

# Alternative method (if the above doesn't work)
uvicorn src.chromatica.api.main:app --reload --host 0.0.0.0 --port 8000
```

### **Step 2: Verify Server Status**

```bash
# Check if server is running
curl http://localhost:8000/api/info

# Expected response:
# {
#   "status": "ready",
#   "message": "Search engine is ready",
#   "index_size": 20,
#   "endpoints": ["/search", "/visualize/query", "/visualize/results"]
# }
```

### **Step 3: Access the Web Interface**

- **Open your browser** and navigate to: `http://localhost:8000/`
- **Check the status indicator** - it should show "✅ System Ready"
- **Verify the search button** is enabled

## Enhanced Web Interface

The web interface at `http://localhost:8000/` now includes enhanced features for displaying search results:

### Image Display in Results

- **Top Results Section**: Each search result now displays the actual image alongside the metadata
- **Image Serving**: Images are served through the `/image/{image_id}` endpoint
- **Responsive Layout**: Results are displayed in a responsive grid layout optimized for image viewing
- **Enhanced Styling**: Result cards feature hover effects and improved visual presentation

### New API Endpoints

#### GET /image/{image_id}

Serves actual image files by their ID.

**Parameters:**

- `image_id` (path): Unique identifier for the image

**Response:**

- Returns the actual image file with appropriate content type
- Supports JPEG, PNG, GIF, and WebP formats

**Example:**

```bash
# Get image with ID "7349806"
curl http://localhost:8000/image/7349806
```

### Updated Search Response

The search response now includes `file_path` information for each result, enabling the frontend to display images:

```json
{
  "query_id": "uuid",
  "query": {
    "colors": ["ea6a81", "f6d727"],
    "weights": [0.49, 0.51]
  },
  "results_count": 2,
  "results": [
    {
      "image_id": "7349806",
      "distance": 0.1234,
      "dominant_colors": ["#placeholder"],
      "file_path": "datasets/test-dataset-20/7349806.jpg"
    }
  ],
  "metadata": {
    "ann_time_ms": 45,
    "rerank_time_ms": 156,
    "total_time_ms": 201
  }
}
```

## 🌐 Using the Web Interface

### **Step 1: Basic Color Selection**

1. **Choose your first color** using the color picker
2. **Adjust the weight** using the slider (0-100%)
3. **Add more colors** by clicking "+ Add Another Color"
4. **Remove colors** using the "Remove" button (if more than one)

### **Step 2: Perform a Search**

1. **Set your colors and weights** (e.g., Red: 60%, Blue: 40%)
2. **Click "🔍 Search Images"**
3. **Wait for results** - the interface will show loading states
4. **View the generated visualizations**:
   - **Query Visualization**: Shows your color query with weights
   - **Results Collage**: Shows the found images in a grid

### **Step 3: Interpret Results**

- **Distance scores**: Lower values = more similar to your query
- **Image IDs**: Reference the original dataset images
- **Performance metrics**: Shows search and reranking times

## 🔌 API Usage Examples

### **Basic Search Query**

```bash
# Simple single-color search
curl "http://localhost:8000/search?colors=FF0000&weights=1.0&k=5"

# Multi-color weighted search
curl "http://localhost:8000/search?colors=FF0000,0000FF&weights=0.7,0.3&k=10"
```

### **Query Visualization**

```bash
# Generate query visualization
curl "http://localhost:8000/visualize/query?colors=FF0000,00FF00&weights=0.6,0.4" \
  --output query_viz.png
```

### **Results Collage**

```bash
# Generate results collage
curl "http://localhost:8000/visualize/results?colors=FF0000,00FF00&weights=0.6,0.4&k=10" \
  --output results_collage.png
```

### **API Information**

```bash
# Check system status
curl "http://localhost:8000/api/info"

# View API documentation
# Open: http://localhost:8000/docs
```

## 🧪 Testing & Validation

### **Step 1: Run Histogram Generation Tests**

```bash
# Test histogram generation with your dataset
python tools/test_histogram_generation.py --directory datasets/test-dataset-20

# Expected output: 6 different report types showing validation results
```

### **Step 2: Test the Complete Pipeline**

```bash
# Test the entire search system
python tools/test_search_system.py

# Test specific components
python tools/test_faiss_duckdb.py
python tools/test_query_processor.py
python tools/test_reranking.py
```

### **Step 3: Run Visualization Demo**

```bash
# Test all visualization features
python tools/demo_visualization.py

# Expected output: Generated demo images and performance metrics
```

### **Step 4: Sanity Checks**

```bash
# Run comprehensive sanity checks
python scripts/run_sanity_checks.py

# Check logs
tail -f logs/sanity_checks.log
```

## 🔍 Troubleshooting

### **Common Issues & Solutions**

#### **Issue: "Search components not initialized"**

```bash
# Solution: Build the index
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index

# Verify files exist
ls -la test_index/
```

#### **Issue: Import errors when running scripts**

```bash
# Solution: Always activate virtual environment first
venv311\Scripts\activate  # Windows
# source venv311/bin/activate  # macOS/Linux

# Then run your command
python scripts/build_index.py ...
```

#### **Issue: "Failed to generate visualization"**

```bash
# Check matplotlib installation
pip install matplotlib pillow

# Verify image paths are accessible
python -c "from PIL import Image; print('PIL working')"
```
