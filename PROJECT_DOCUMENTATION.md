# 📚 Chromatica Project Documentation

Complete guide to the Chromatica color search engine project structure, core files, commands, and essential information.

---

## 1. Project Structure

### Directory Tree

```
Chromatica/
├── src/
│   └── chromatica/
│       ├── api/                    # FastAPI web application and endpoints
│       │   ├── main.py             # Main FastAPI app and REST endpoints
│       │   └── static/             # Web interface static files
│       │       ├── index.html      # Main web UI
│       │       ├── css/            # Stylesheets (Catppuccin Mocha theme)
│       │       ├── js/             # JavaScript frontend code
│       │       │   ├── main.js     # Main frontend logic
│       │       │   └── modules/    # 3D visualization modules (10 modules)
│       │       └── fonts/          # Custom fonts (JetBrains Mono Nerd Font)
│       ├── core/                   # Core algorithms and color processing
│       │   ├── histogram.py        # Histogram generation with tri-linear soft assignment
│       │   ├── query.py            # Query processing and color conversion
│       │   └── rerank.py           # Sinkhorn-EMD reranking implementation
│       ├── indexing/               # FAISS index and DuckDB store management
│       │   ├── store.py            # FAISS HNSW index and DuckDB metadata store
│       │   └── pipeline.py         # Image processing pipeline
│       ├── search.py               # Two-stage search pipeline orchestration
│       ├── utils/                  # Configuration and utility functions
│       │   ├── config.py           # Centralized configuration constants
│       │   └── color_names.py      # Color name utilities
│       └── visualization/          # Visualization utilities
│           └── query_viz.py        # Query visualization helpers
├── scripts/                        # Build and maintenance scripts
│   ├── build_index.py              # Main index building script
│   ├── build_covers_index.py       # Album cover indexing script
│   ├── chunked_indexing.py         # Chunked batch indexing
│   ├── evaluate.py                 # Evaluation and benchmarking
│   └── run_sanity_checks.py        # System validation checks
├── tools/                          # Testing and development tools
│   ├── test_histogram_generation.py    # Histogram testing
│   ├── test_api.py                 # API endpoint testing
│   ├── test_faiss_duckdb.py        # Index and database testing
│   ├── test_search_system.py       # End-to-end search testing
│   ├── cleanup_outputs.py          # Output cleanup utility
│   ├── performance_benchmark.py    # Performance benchmarking
│   └── [25+ additional testing tools]
├── datasets/                       # Test datasets and queries
│   ├── test-dataset-20/            # Quick testing (20 images)
│   ├── test-dataset-50/            # Small validation (50 images)
│   ├── test-dataset-200/           # Medium testing (200 images)
│   ├── test-dataset-5000/          # Production-scale (5,000 images)
│   └── [query files]
├── docs/                           # Comprehensive documentation
│   ├── .cursor/                    # Cursor IDE rules and critical instructions
│   ├── api/                        # API documentation
│   ├── guides/                     # User and developer guides
│   ├── modules/                    # Module-specific documentation
│   └── [60+ documentation files]
├── tests/                          # Unit and integration tests
├── logs/                           # Application logs
├── venv311/                        # Python 3.11 virtual environment
├── requirements.txt                # Python dependencies
├── COMMANDS.md                     # Command reference (detailed)
├── README.md                       # Project overview
└── PROJECT_DOCUMENTATION.md        # This file
```

### Structure Explanation

**`src/chromatica/`** - Main source code organized into logical modules:
- **`core/`**: Core algorithms (histogram generation, query processing, reranking)
- **`indexing/`**: FAISS index management and DuckDB metadata storage
- **`api/`**: FastAPI web application with REST endpoints and static web interface
- **`utils/`**: Shared utilities and configuration management
- **`visualization/`**: Visualization helper functions

**`scripts/`** - Build and maintenance scripts for indexing, evaluation, and system checks

**`tools/`** - Comprehensive testing, debugging, and development tools (25+ utilities)

**`datasets/`** - Test datasets of varying sizes for development and validation

**`docs/`** - Extensive documentation including API references, guides, and troubleshooting

---

## 2. Core Files

### Core Algorithm Files

#### `src/chromatica/core/histogram.py`
**Purpose**: Generates normalized histograms from images in CIE Lab color space.

**Key Functions**:
- `build_histogram(image_path: str) -> np.ndarray`: Main histogram generation with tri-linear soft assignment
- `build_histogram_fast(image_path: str) -> np.ndarray`: Fast alternative for prototyping
- **Binning Grid**: 8×12×12 = 1,152 dimensions (L*, a*, b*)
- **Normalization**: L1 normalization to create probability distributions
- **Color Space**: CIE Lab (D65 illuminant) for perceptual uniformity

**Dependencies**: `opencv-python`, `scikit-image`, `numpy`

#### `src/chromatica/core/query.py`
**Purpose**: Processes color queries and converts them to histogram format.

**Key Functions**:
- `process_query(colors: List[str], weights: List[float]) -> np.ndarray`: Converts hex colors to query histogram
- Handles color space conversion (hex → RGB → Lab)
- Creates "softened" query histograms for search

#### `src/chromatica/core/rerank.py`
**Purpose**: High-fidelity reranking using Sinkhorn-approximated Earth Mover's Distance.

**Key Functions**:
- `compute_sinkhorn_distance(query_hist, candidate_hist, cost_matrix) -> float`: Sinkhorn-EMD calculation
- `get_cost_matrix() -> np.ndarray`: Pre-computed squared Euclidean distance between bin centers
- `get_hue_mixture_mask()`: Gaussian mixture mask for multi-color queries
- Various masking functions for color adherence (chroma, lightness, hue proximity)

**Dependencies**: `POT` (Python Optimal Transport library)

### Indexing and Storage Files

#### `src/chromatica/indexing/store.py`
**Purpose**: Manages FAISS HNSW index and DuckDB metadata store.

**Key Classes**:
- **`AnnIndex`**: FAISS HNSW index management
  - `create_index()`: Creates IndexHNSWFlat with M=32
  - `add_vectors()`: Adds Hellinger-transformed histograms
  - `search()`: Performs ANN search (returns top-K candidates)
  
- **`MetadataStore`**: DuckDB metadata and raw histogram storage
  - `add_image()`: Stores image metadata and raw histogram
  - `get_histograms()`: Batch retrieval of raw histograms for reranking
  - Thread-safe with thread-local connections

**Dependencies**: `faiss-cpu`, `duckdb`

#### `src/chromatica/indexing/pipeline.py`
**Purpose**: Image processing pipeline for batch indexing.

**Key Functions**:
- Processes images: resize → RGB → Lab conversion → histogram generation
- Handles batch processing and error recovery
- Performance monitoring and logging

### Search Orchestration

#### `src/chromatica/search.py`
**Purpose**: Orchestrates the two-stage search pipeline.

**Key Functions**:
- `find_similar(colors, weights, k, fast_mode) -> List[Dict]`: Main search function
  - Stage 1: ANN search using FAISS HNSW (top-200 candidates)
  - Stage 2: Sinkhorn-EMD reranking on raw histograms
  - Hue presence prefiltering and per-color enforcement
  - Comprehensive logging and performance tracking

### API and Web Interface

#### `src/chromatica/api/main.py`
**Purpose**: FastAPI web application with REST endpoints.

**Key Endpoints**:
- `GET /search`: Main search endpoint with color and weight parameters
- `POST /api/extract-colors`: Image upload and color extraction
- `GET /image/{image_id}`: Serves image files
- `GET /api/info`: System status and metadata
- `/visualize/*`: Visualization endpoints

**Features**:
- Parallel processing support with ThreadPoolExecutor
- Performance statistics tracking
- CORS support for external access
- Comprehensive error handling and logging

**Dependencies**: `fastapi`, `uvicorn`, `python-multipart`

#### `src/chromatica/api/static/index.html`
**Purpose**: Main web interface with interactive color search UI.

**Features**:
- Catppuccin Mocha theme with custom typography
- Color format support (HEX, RGB, HSL, HSV, CMYK)
- Image upload and color extraction
- Color scheme generation
- 10 Interactive 3D visualizations
- Testing and development tools

#### `src/chromatica/api/static/js/main.js`
**Purpose**: Frontend JavaScript for web interface.

**Key Features**:
- Color query management (add, remove, randomize colors)
- Format conversion functions
- Dynamic module loading for 3D visualizations
- Search orchestration and results display
- Color scheme generation and export

**3D Visualization Modules** (in `js/modules/`):
1. `colorSpaceNavigator.js` - Navigate images in 3D Lab space
2. `histogramCloud.js` - 3D histogram bar charts
3. `similarityLandscape.js` - Similarity-based 3D positioning
4. `rerankingAnimation.js` - Animated two-stage search
5. `imageGlobe.js` - Images mapped onto 3D sphere
6. `connectionsGraph.js` - Similarity graph visualization
7. `otTransport3D.js` - Optimal Transport visualization
8. `hnswGraphExplorer.js` - FAISS HNSW graph explorer
9. `colorDensityVolume.js` - Color space density volume
10. `imageThumbnails3D.js` - Image thumbnails in 3D space

### Configuration

#### `src/chromatica/utils/config.py`
**Purpose**: Centralized configuration constants.

**Key Constants**:
- `BIN_SIZES`: [8, 12, 12] - Histogram binning grid dimensions
- `COLOR_RANGES`: Lab value ranges for each axis
- `FAISS_HNSW_M`: 32 - HNSW graph connection parameter
- `RERANK_K`: 300 - Number of candidates for reranking
- `SINKHORN_EPSILON`: 0.05 - Sinkhorn regularization parameter
- `MAX_SEARCH_RESULTS`: 50 - Maximum results returned
- Various tuning parameters for color adherence

---

## 3. Commands

### Environment Setup

#### Virtual Environment Activation

```powershell
# Windows PowerShell
venv311\Scripts\activate

# Windows Command Prompt
venv311\Scripts\activate.bat

# Linux/Mac
source venv311/bin/activate
```

#### Install Dependencies

```powershell
# Install all dependencies
pip install -r requirements.txt

# Upgrade pip first (if needed)
python -m pip install --upgrade pip

# Install specific package
pip install fastapi uvicorn
```

### Index Building

#### Build Index from Images

```powershell
# Basic usage - build index from dataset directory
python scripts/build_index.py --dataset datasets/test-dataset-200

# Specify output directory
python scripts/build_index.py --dataset datasets/test-dataset-200 --output _my_index_db

# Use existing FAISS index and only update DuckDB
python scripts/build_index.py --dataset datasets/test-dataset-200 --faiss-index _existing_db/chromatica_index.faiss

# Build with custom batch size
python scripts/build_index.py --dataset datasets/test-dataset-200 --batch-size 50

# Resume from checkpoint
python scripts/build_index.py --dataset datasets/test-dataset-200 --resume

# Enable verbose logging
python scripts/build_index.py --dataset datasets/test-dataset-200 --verbose
```

**Flags**:
- `--dataset`: Path to image dataset directory (required)
- `--output`: Output directory for index files (default: `_small_covers_db`)
- `--faiss-index`: Path to existing FAISS index file
- `--batch-size`: Number of images to process per batch (default: 10)
- `--resume`: Resume from last checkpoint
- `--verbose`: Enable verbose logging

#### Build Album Covers Index

```powershell
# Build index from album covers
python scripts/build_covers_index.py --dataset _covers01_100k/images --output _covers01_100k_db

# Process with parallel workers
python scripts/build_covers_index.py --dataset _covers01_100k/images --workers 4

# Specify image format
python scripts/build_covers_index.py --dataset _covers01_100k/images --image-format jpg
```

#### Chunked Indexing (Large Datasets)

```powershell
# Process dataset in chunks
python scripts/chunked_indexing.py --dataset large_dataset --chunk-size 10000 --output large_index_db
```

**Flags**:
- `--chunk-size`: Number of images per chunk (default: 10000)
- `--workers`: Number of parallel workers

### API Server

#### Start Development Server

```powershell
# Basic server start
uvicorn src.chromatica.api.main:app --reload

# Specify host and port
uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode (no reload)
uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000

# Multiple workers (production)
uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Flags**:
- `--reload`: Enable auto-reload on code changes (development)
- `--host`: Host address (default: 127.0.0.1)
- `--port`: Port number (default: 8000)
- `--workers`: Number of worker processes (production)

#### Test API Endpoint

```powershell
# Quick search test
curl "http://localhost:8000/search?colors=FF0000,00FF00&weights=0.5,0.5&k=10"

# Fast mode search
curl "http://localhost:8000/search?colors=FF0000&weights=1.0&k=20&fast_mode=true"

# Check API status
curl "http://localhost:8000/api/info"
```

### Testing and Validation

#### Test Histogram Generation

```powershell
# Test single image
python tools/test_histogram_generation.py --image datasets/test-dataset-20/image1.jpg

# Test entire dataset
python tools/test_histogram_generation.py --dataset datasets/test-dataset-200

# Generate detailed report
python tools/test_histogram_generation.py --dataset datasets/test-dataset-200 --report detailed

# Output to specific directory
python tools/test_histogram_generation.py --dataset datasets/test-dataset-200 --output-dir reports/
```

**Flags**:
- `--image`: Single image file path
- `--dataset`: Dataset directory path
- `--report`: Report type (basic, detailed, summary)
- `--output-dir`: Output directory for reports

#### Test API Endpoints

```powershell
# Run all API tests
python tools/test_api.py

# Test specific endpoint
python tools/test_api.py --endpoint search

# Verbose output
python tools/test_api.py --verbose

# Custom base URL
python tools/test_api.py --base-url http://localhost:8000
```

#### Test Search System

```powershell
# End-to-end search test
python tools/test_search_system.py --dataset datasets/test-dataset-200

# Test with specific query
python tools/test_search_system.py --colors FF0000,00FF00 --weights 0.5,0.5

# Performance benchmark
python tools/test_search_system.py --benchmark --iterations 10
```

#### Performance Benchmark

```powershell
# Run benchmark suite
python tools/performance_benchmark.py

# Custom test queries
python tools/performance_benchmark.py --queries test-queries.json

# Output results to file
python tools/performance_benchmark.py --output benchmark_results.json
```

### System Validation

#### Run Sanity Checks

```powershell
# Run all sanity checks
python scripts/run_sanity_checks.py

# Check specific component
python scripts/run_sanity_checks.py --check index

# Verbose output
python scripts/run_sanity_checks.py --verbose
```

**Flags**:
- `--check`: Specific component to check (index, api, database)
- `--verbose`: Enable verbose logging

#### Validate Configuration

```powershell
# Validate project configuration
python -c "from src.chromatica.utils.config import validate_config; validate_config()"
```

### Data Management

#### Cleanup Outputs

```powershell
# Interactive cleanup
python tools/cleanup_outputs.py

# Clean specific types
python tools/cleanup_outputs.py --types histograms reports logs

# Dry run (preview only)
python tools/cleanup_outputs.py --dry-run

# Force cleanup (no prompts)
python tools/cleanup_outputs.py --force --types cache temp
```

**Flags**:
- `--types`: Types to clean (histograms, reports, logs, test_index, cache, temp)
- `--dry-run`: Preview what would be deleted
- `--force`: Skip confirmation prompts
- `--interactive`: Interactive mode (default)

### Evaluation

#### Run Evaluation

```powershell
# Run evaluation on test queries
python scripts/evaluate.py --queries datasets/test-queries.json

# Output evaluation results
python scripts/evaluate.py --queries datasets/test-queries.json --output eval_results.json

# Custom metrics
python scripts/evaluate.py --queries datasets/test-queries.json --metrics precision recall
```

---

## 4. Important to Note

### Project Overview

**Chromatica** is a production-ready, two-stage color-based image search engine that retrieves images whose dominant palettes best match weighted, multi-color queries. The system combines fast approximate nearest neighbor search with high-fidelity reranking to deliver accurate, perceptually meaningful results.

### Key Technical Decisions

1. **CIE Lab Color Space**: Chosen for perceptual uniformity, avoiding RGB non-uniformity and HSV hue wraparound issues. This ensures that numerical distance closely approximates human-perceived color difference.

2. **8×12×12 Binning Grid**: Creates a 1,152-dimensional histogram representation that balances granularity with computational efficiency. The tri-linear soft assignment ensures robustness against minor color shifts.

3. **Two-Stage Architecture**: 
   - **Stage 1 (ANN)**: Fast candidate retrieval using FAISS HNSW index with Hellinger-transformed histograms (top-200 candidates)
   - **Stage 2 (Reranking)**: High-fidelity Sinkhorn-EMD reranking on raw histograms for perceptually accurate distance calculations

4. **Sinkhorn-EMD**: Uses entropy-regularized Earth Mover's Distance approximation that correctly models the "work" required to transform one color palette into another, accounting for both color differences and their relative weights.

5. **FAISS HNSW Index**: `IndexHNSWFlat` with M=32 provides excellent performance-to-accuracy ratio without requiring a training phase, making it ideal for dynamic indexing.

6. **DuckDB Storage**: Efficient storage for image metadata and raw histograms, with thread-safe access patterns for concurrent operations.

### Performance Characteristics

- **Histogram Generation**: ~200ms per image target
- **Total Search Latency (P95)**: <450ms target
- **ANN Search**: <150ms
- **Reranking**: <300ms
- **Index Size**: Supports 100K+ images efficiently

### Current Implementation Status

✅ **Completed**:
- Core histogram generation pipeline with tri-linear soft assignment
- FAISS HNSW index and DuckDB metadata store
- Two-stage search pipeline with comprehensive tuning parameters
- FastAPI web application with REST endpoints
- Advanced web interface with Catppuccin Mocha theme
- 10 Interactive 3D visualizations using Three.js
- Comprehensive testing and validation tools
- Parallel processing support for concurrent searches
- Color format support (HEX, RGB, HSL, HSV, CMYK)
- Image upload and color extraction
- Color scheme generation (Monochromatic, Complementary, Analogous, Triadic)
- Extensive documentation (60+ documentation files)

🔄 **Ongoing**:
- Performance optimization and tuning
- Production deployment preparation
- Large-scale evaluation and benchmarking

### Critical Configuration

All key constants are centralized in `src/chromatica/utils/config.py`. Important parameters:
- `RERANK_K`: 300 (candidates for reranking)
- `MAX_SEARCH_RESULTS`: 50 (maximum results returned)
- `FAISS_HNSW_M`: 32 (HNSW graph connections)
- `SINKHORN_EPSILON`: 0.05 (Sinkhorn regularization)
- Various color adherence parameters (hue sigma, chroma gain, etc.)

### Development Environment

- **Virtual Environment**: `venv311` (Python 3.11)
- **Activation**: `venv311\Scripts\activate` (Windows) or `source venv311/bin/activate` (Linux/Mac)
- **Always activate** virtual environment before running commands

### Testing Strategy

The project includes comprehensive testing across multiple levels:
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: System-wide testing
3. **Performance Tests**: Benchmarking and latency monitoring
4. **Validation Tests**: Histogram correctness, index integrity
5. **End-to-End Tests**: Complete search pipeline validation

### Documentation Structure

- **`docs/.cursor/critical_instructions.md`**: Single source of truth for project specifications
- **`docs/`**: Comprehensive documentation (API, guides, troubleshooting)
- **`COMMANDS.md`**: Detailed command reference
- **`PROJECT_DOCUMENTATION.md`**: This file - project overview and structure

### Best Practices

1. **Always activate virtual environment** before running any commands
2. **Use test datasets** (test-dataset-20, -50, -200, -5000) for development
3. **Run sanity checks** before major operations
4. **Monitor performance** using built-in statistics
5. **Check logs** in `logs/` directory for troubleshooting
6. **Refer to critical_instructions.md** for architectural decisions

### Important File Locations

- **Index Files**: `_small_covers_db/` or custom output directory
- **Logs**: `logs/` directory
- **Test Outputs**: `reports/` directory
- **Configuration**: `src/chromatica/utils/config.py`
- **Web Interface**: `src/chromatica/api/static/index.html`
- **Documentation**: `docs/` directory

### Project Goals

Chromatica aims to provide:
- **Accurate Color Matching**: Perceptually meaningful results that match user intent
- **High Performance**: Sub-500ms search latency for real-time use
- **Scalability**: Efficient indexing and search for 100K+ image collections
- **Usability**: Intuitive web interface with comprehensive visualization tools
- **Production Readiness**: Robust error handling, logging, and validation

---

**Last Updated**: 2025-01-31  
**Project Version**: Week 3 (In Progress)  
**Python Version**: 3.11+  
**License**: See LICENSE file (if applicable)

