# Chromatica

A high-performance, two-stage color-based image search engine built with Python, FAISS, and DuckDB.

## 🎯 Overview

Chromatica is a production-ready color search engine that retrieves images whose dominant palettes best match weighted, multi-color queries. The system uses a sophisticated two-stage approach combining fast approximate nearest neighbor search with high-fidelity reranking using Earth Mover's Distance.

## ✨ Key Features

- **CIE Lab Color Space**: Perceptually uniform color representation avoiding RGB non-uniformity and HSV hue wraparound issues
- **Advanced Histogram Generation**: 8×12×12 binning grid (1,152 dimensions) with tri-linear soft assignment for robustness
- **Two-Stage Search Pipeline**: Fast FAISS HNSW index for candidate retrieval + Sinkhorn-EMD reranking for accuracy
- **Production Ready**: Comprehensive testing, validation, and performance optimization
- **Comprehensive Tooling**: Testing tools, visualization, and analysis capabilities

## 🏗️ Architecture

```
[Image Input] → [Preprocessing Pipeline] → [Histogram Generation] → [ANN Index (FAISS HNSW)]
      ↑                      ↑
      |                      |
[Query Input] → [Query Processor] → [ANN Search] → [Candidate Reranking] → [Final Results]
```

## 🚀 Current Status

### ✅ Completed (Week 1 & 2)

- **Core Histogram Generation**: Fully implemented with tri-linear soft assignment
- **Image Processing Pipeline**: Complete preprocessing with Lab color space conversion
- **Configuration Management**: Centralized constants and parameters
- **Testing Infrastructure**: Comprehensive testing tools with validation and visualization
- **FAISS HNSW Index**: Vector similarity search implementation
- **DuckDB Metadata Store**: Efficient storage and retrieval system
- **Two-Stage Search Pipeline**: Complete ANN search + Sinkhorn-EMD reranking
- **FastAPI Web API**: REST endpoints for search functionality
- **Advanced Visualization Tools**: 6 comprehensive tools with expandable panels
- **Output Cleanup Tool**: Comprehensive utility for managing generated files and maintaining clean development environment
- **Web Interface**: Catppuccin Mocha theme with custom typography
- **Documentation**: Detailed guides and progress tracking
- **Cursor Rules System**: Modern `.cursor/rules` structure with comprehensive project guidance

### 🔄 In Progress (Week 3)

- **Performance Optimization**: Latency and throughput improvements
- **Production Deployment**: Scaling and monitoring
- **Large-Scale Testing**: 10,000+ image validation

## 🛠️ Technology Stack

- **Core**: Python 3.10+ with type hints
- **Image Processing**: OpenCV, scikit-image
- **Color Science**: CIE Lab color space (D65 illuminant)
- **Vector Search**: FAISS HNSW index
- **Database**: DuckDB for metadata and raw histograms
- **Optimal Transport**: POT library for Sinkhorn-EMD
- **Web Framework**: FastAPI
- **Testing**: Comprehensive validation and benchmarking tools

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Chromatica

# Create and activate virtual environment
python -m venv venv311
venv311\Scripts\activate  # Windows
# or
source venv311/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Testing

### Quick Test

```bash
# Test histogram generation on a single image
python tools/test_histogram_generation.py --image datasets/test-dataset-20/a1.png
```

### Comprehensive Testing

```bash
# Test entire dataset with visualization
python tools/test_histogram_generation.py --directory datasets/test-dataset-50/ --output-format both
```

### Test Datasets

- **test-dataset-20**: 20 images for rapid development and debugging
- **test-dataset-50**: 50 images for validation and testing
- **test-dataset-200**: 200 images for performance testing
- **test-dataset-5000**: 5,000 images for production-scale validation

## 🎨 Advanced Visualization Tools

Chromatica includes a comprehensive suite of **6 Advanced Visualization Tools** that provide powerful analysis and visualization capabilities:

### 🎨 Color Palette Analyzer

- Image color analysis with K-means clustering
- Multiple output formats (PNG, PDF, JSON, CSV)
- Performance benchmarking and validation

### 📊 Search Results Analyzer

- Search result visualization and analysis
- Multiple visualization styles (charts, heatmaps, 3D projections)
- Performance metrics and ranking analysis

### 🔍 Interactive Color Explorer

- Color harmony generation (complementary, analogous, triadic)
- Real-time preview and live search integration
- Palette export and scheme saving

### 📈 Histogram Analysis Tool

- Histogram validation and visualization
- Performance benchmarking and timing analysis
- Multiple visualization types and comprehensive reporting

### 🎯 Distance Debugger Tool

- Sinkhorn-EMD debugging and analysis
- Multiple test types and dataset support
- Comprehensive debugging options and analysis reports

### 🎭 Query Visualizer Tool

- Color query visualization with multiple styles
- Various layout options and customization
- Accessibility features and color harmony analysis

### ✨ Key Features

- **Expandable Tool Panels**: Full configuration options for each tool
- **Real Quick Test Execution**: Actual tool execution with specialized datasets
- **Consistent Interface**: Three-button design (Run Tool, Info, Quick Test)
- **Professional Quality**: Production-ready implementation with comprehensive error handling

**Quick Test Datasets**: Each tool includes specialized datasets in `datasets/quick-test/` for immediate testing and validation.

## 🧹 Output Cleanup Tool

The Chromatica project includes a comprehensive output cleanup tool (`tools/cleanup_outputs.py`) for managing generated files and maintaining a clean development environment.

### Key Features

- **Selective Cleanup**: Choose specific output types (histograms, reports, logs, test_index, cache, temp)
- **Interactive Mode**: User-friendly interface with guided selection
- **Safety Features**: Confirmation prompts, dry-run mode, and comprehensive error handling
- **Size Reporting**: Shows disk space usage and freed space
- **Script Generation**: Create standalone cleanup scripts for specific operations

### Usage Examples

```bash
# Interactive mode - guided cleanup selection
python tools/cleanup_outputs.py

# Clean specific output types
python tools/cleanup_outputs.py --logs --reports --histograms

# Clean all outputs with confirmation
python tools/cleanup_outputs.py --all --confirm

# Preview what would be deleted (safe)
python tools/cleanup_outputs.py --datasets --dry-run

# Clean dataset outputs (histograms + reports)
python tools/cleanup_outputs.py --datasets
```

### Current Project Status

The cleanup tool can manage:

- **21 histogram files** (13.8 MB)
- **67 report files** (1.2 MB)
- **3 log files** (119.8 KB)
- **3 test index files** (124.1 MB)
- **7,101 cache files** (146.0 MB)

**Total**: ~285 MB of generated files that can be cleaned up as needed.

For detailed documentation, see [`docs/tools_cleanup_outputs.md`](docs/tools_cleanup_outputs.md).

## 📚 Documentation

- **[Project Plan](docs/.cursor/critical_instructions.md)**: Comprehensive technical specifications
- **[Progress Report](docs/progress.md)**: Current implementation status
- **[Histogram Guide](docs/histogram_generation_guide.md)**: Detailed histogram generation documentation
- **[FAISS & DuckDB Guide](docs/faiss_duckdb_guide.md)**: Vector indexing and storage implementation
- **[Cleanup Tool Guide](docs/tools_cleanup_outputs.md)**: Comprehensive output cleanup utility documentation
- **[Cleanup Troubleshooting](docs/troubleshooting_cleanup_tool.md)**: Cleanup tool troubleshooting guide
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions

## 🔧 Development

### Project Structure

```
src/chromatica/
├── core/           # Histogram generation and color processing
├── indexing/       # FAISS index and DuckDB storage
├── api/            # FastAPI web endpoints
└── utils/          # Configuration and utilities

tools/              # Testing and development tools
datasets/           # Test datasets for validation
docs/               # Comprehensive documentation
```

### Key Modules

- **`histogram.py`**: Core histogram generation with tri-linear soft assignment
- **`pipeline.py`**: Complete image processing pipeline
- **`store.py`**: FAISS and DuckDB integration
- **`config.py`**: Centralized configuration constants

## 📊 Performance

- **Histogram Generation**: ~200ms per image
- **Validation Success Rate**: 100%
- **Memory Efficiency**: ~4.6KB per histogram
- **Target Latency**: P95 < 450ms for complete search pipeline

## 🤝 Contributing

1. Follow the project structure and coding standards
2. Ensure all code includes comprehensive docstrings and validation
3. Run tests before submitting changes
4. Update documentation as needed

## 🤖 Cursor Rules System

Chromatica uses a comprehensive Cursor rules system for AI-assisted development:

### Modern Rules Structure

- **`.cursor/rules/`**: MDC format rules with metadata and scoping
- **Always Applied**: Core project rules that ensure consistency
- **Auto-Attached**: Rules that automatically apply based on file patterns
- **Agent-Requested**: Rules available for AI to include when needed
- **Manual**: Rules for explicit invocation with `@ruleName`

### Rule Categories

- **Core Rules**: Project overview, technology stack, and critical instructions
- **Module Rules**: Scoped rules for core, API, indexing, and tools modules
- **Specialized Rules**: Documentation, testing, web interface, and workflow standards
- **Alternative Format**: `AGENTS.md` for simple markdown instructions

### Key Benefits

- **Better Organization**: Rules organized by scope and purpose
- **Improved Performance**: Selective rule application for efficiency
- **Enhanced Functionality**: More flexible and powerful rule system
- **Easier Maintenance**: Easier to maintain and update specific rule sets

For detailed information, see [Cursor Rules Guide](docs/cursor_rules_guide.md).

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **Google Gemini**: Core algorithmic approach and system design
- **FAISS**: High-performance vector similarity search
- **DuckDB**: Efficient analytical database
- **OpenCV & scikit-image**: Image processing and color science

---

**Project Status**: Week 1 Complete ✅ | Week 2 Complete ✅ | Week 3 In Progress 🔄 | Production Target: Q1 2025 🎯
