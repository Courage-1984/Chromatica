# 🏗️ Chromatica Project Architecture

This document provides a comprehensive overview of the Chromatica color search engine's architecture, system design, and component interactions.

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [File Structure](#file-structure)
7. [API Design](#api-design)
8. [Performance Characteristics](#performance-characteristics)
9. [Security Considerations](#security-considerations)
10. [Scalability Design](#scalability-design)

## 🎯 System Overview

Chromatica is a **two-stage color-based image search engine** that combines:

1. **Fast Approximate Search** using FAISS HNSW index
2. **High-Quality Reranking** using Sinkhorn-EMD distance
3. **Visual Query Interface** with real-time feedback
4. **Comprehensive Result Visualization** including collages

### **Core Capabilities**
- **Multi-color queries** with customizable weights
- **Perceptually accurate** color matching using CIE Lab space
- **Real-time search** with sub-second response times
- **Rich visualizations** for queries and results
- **Scalable architecture** supporting thousands of images

## 🏛️ Architecture Principles

### **Design Philosophy**
- **Performance First**: Optimized for speed and efficiency
- **Accuracy Matters**: High-quality results through sophisticated algorithms
- **User Experience**: Intuitive interfaces and visual feedback
- **Maintainability**: Clean, documented, and testable code
- **Scalability**: Designed to grow with data and user demands

### **Architectural Patterns**
- **Layered Architecture**: Clear separation of concerns
- **Pipeline Processing**: Sequential data transformation
- **Factory Pattern**: Configurable component creation
- **Strategy Pattern**: Pluggable algorithms and metrics
- **Observer Pattern**: Event-driven processing and logging

## 🔧 Component Architecture

### **1. Core Processing Layer**

```
┌─────────────────────────────────────────────────────────────┐
│                    Core Processing Layer                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Histogram       │  │ Query           │  │ Color       │ │
│  │ Generation      │  │ Processing      │  │ Conversion  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### **Histogram Generation (`src/chromatica/core/histogram.py`)**
- **Purpose**: Convert images to fixed-length color histograms
- **Algorithm**: Tri-linear soft assignment in CIE Lab space
- **Output**: 1,152-dimensional normalized vectors
- **Performance**: ~200ms per image

#### **Query Processing (`src/chromatica/core/query.py`)**
- **Purpose**: Convert user queries to searchable histograms
- **Features**: Multi-color support with weights
- **Validation**: Input sanitization and histogram verification
- **Output**: Normalized query histograms

#### **Color Conversion (`src/chromatica/core/color.py`)**
- **Purpose**: Handle color space transformations
- **Conversions**: RGB ↔ CIE Lab, Hex ↔ RGB
- **Standards**: D65 illuminant, sRGB color space
- **Accuracy**: Perceptually uniform color representation

### **2. Search Engine Layer**

```
┌─────────────────────────────────────────────────────────────┐
│                    Search Engine Layer                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ FAISS Index     │  │ DuckDB          │  │ Search      │ │
│  │ (ANN Search)    │  │ Metadata Store  │  │ Orchestrator│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### **FAISS Index (`src/chromatica/indexing/faiss_wrapper.py`)**
- **Type**: HNSW (Hierarchical Navigable Small World)
- **Parameters**: M=32, efConstruction=200, efSearch=100
- **Distance**: L2 (Euclidean) on Hellinger-transformed histograms
- **Performance**: Sub-millisecond search times

#### **DuckDB Store (`src/chromatica/indexing/duckdb_wrapper.py`)**
- **Purpose**: Store image metadata and raw histograms
- **Schema**: Image ID, file path, histogram data, metadata
- **Operations**: Batch insert, efficient retrieval, full-text search
- **Performance**: Fast queries with minimal memory overhead

#### **Search Orchestrator (`src/chromatica/search.py`)**
- **Purpose**: Coordinate the two-stage search process
- **Stage 1**: FAISS ANN search for candidate selection
- **Stage 2**: Sinkhorn-EMD reranking for result ordering
- **Output**: Ranked results with distance scores

### **3. Visualization Layer**

```
┌─────────────────────────────────────────────────────────────┐
│                    Visualization Layer                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Query           │  │ Results         │  │ Web         │ │
│  │ Visualizer      │  │ Collage Builder │  │ Interface   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### **Query Visualizer (`src/chromatica/visualization/query_viz.py`)**
- **Features**: Weighted color bars, palette wheels, summary grids
- **Output**: PNG images with comprehensive query representation
- **Customization**: Configurable dimensions and layouts
- **Performance**: ~50-200ms generation time

#### **Results Collage Builder (`src/chromatica/visualization/query_viz.py`)**
- **Features**: Grid layouts, distance annotations, smart sizing
- **Layout**: Configurable images per row and output dimensions
- **Annotations**: Distance scores overlaid on images
- **Performance**: ~100-500ms for 10 images

#### **Web Interface (`src/chromatica/api/static/index.html`)**
- **Features**: Color pickers, weight sliders, real-time preview
- **Responsiveness**: Works on desktop and mobile devices
- **Integration**: Seamless API communication and error handling
- **User Experience**: Intuitive color selection and search

### **4. API Layer**

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ FastAPI         │  │ Search          │  │ Visualization│ │
│  │ Application     │  │ Endpoints       │  │ Endpoints   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### **FastAPI Application (`src/chromatica/api/main.py`)**
- **Framework**: FastAPI with automatic OpenAPI generation
- **Features**: Async request handling, automatic validation
- **Documentation**: Interactive API docs at `/docs`
- **Performance**: High-throughput request processing

#### **Search Endpoints**
- **`GET /search`**: Main search functionality
- **Parameters**: `colors`, `weights`, `k` (result count)
- **Response**: JSON with results, metadata, and timing
- **Validation**: Input sanitization and error handling

#### **Visualization Endpoints**
- **`GET /visualize/query`**: Generate query visualizations
- **`GET /visualize/results`**: Generate result collages
- **Response**: PNG images for direct display
- **Caching**: Efficient image generation and delivery

## 🔄 Data Flow

### **1. Index Building Flow**

```
Images → Histogram Generation → FAISS Index + DuckDB Store
  ↓              ↓                    ↓
File System → CIE Lab Conversion → Searchable Database
```

#### **Detailed Steps**
1. **Image Loading**: Read images from filesystem
2. **Resizing**: Normalize to consistent dimensions
3. **Color Conversion**: RGB → CIE Lab transformation
4. **Histogram Generation**: Tri-linear soft assignment
5. **Hellinger Transform**: Apply square root for L2 compatibility
6. **Index Building**: FAISS HNSW construction
7. **Metadata Storage**: DuckDB insertion with file paths

### **2. Search Query Flow**

```
User Query → Query Processing → ANN Search → Reranking → Results
    ↓              ↓              ↓           ↓         ↓
Color Input → Histogram → FAISS Index → Sinkhorn-EMD → Ranked Output
```

#### **Detailed Steps**
1. **Query Input**: Parse colors and weights from request
2. **Histogram Creation**: Generate query histogram
3. **ANN Search**: Find approximate nearest neighbors
4. **Metadata Retrieval**: Get image paths and raw histograms
5. **Reranking**: Apply Sinkhorn-EMD for accurate ordering
6. **Result Formatting**: Structure response with metadata
7. **Visualization**: Generate query and result images

### **3. Visualization Flow**

```
Search Results → Image Loading → Collage Assembly → PNG Output
      ↓              ↓              ↓            ↓
Distance Data → File Paths → Grid Layout → Web Display
```

#### **Detailed Steps**
1. **Result Processing**: Extract image paths and distances
2. **Image Loading**: Read and resize result images
3. **Layout Calculation**: Determine grid dimensions
4. **Collage Assembly**: Place images with annotations
5. **Image Generation**: Convert to PNG format
6. **Web Delivery**: Serve to browser interface

## 🛠️ Technology Stack

### **Core Technologies**

#### **Python Ecosystem**
- **Python 3.10+**: Modern language features and performance
- **NumPy**: Efficient numerical operations and array handling
- **OpenCV**: Image loading, resizing, and processing
- **scikit-image**: Color space conversions and transformations

#### **Machine Learning & Search**
- **FAISS**: High-performance similarity search and clustering
- **POT**: Python Optimal Transport for Sinkhorn-EMD
- **scikit-learn**: Additional ML utilities and preprocessing

#### **Data Storage**
- **DuckDB**: Fast analytical database for metadata
- **SQLite**: Alternative storage backend (if needed)
- **JSON**: Configuration and metadata serialization

#### **Web & API**
- **FastAPI**: Modern, fast web framework for APIs
- **Uvicorn**: ASGI server for production deployment
- **HTML5/CSS3/JavaScript**: Modern web interface
- **Pillow (PIL)**: Image processing and manipulation

#### **Visualization**
- **Matplotlib**: Scientific plotting and chart generation
- **Seaborn**: Statistical data visualization
- **Pillow**: Image creation and manipulation

### **Development Tools**
- **Type Hints**: Static type checking and documentation
- **Logging**: Comprehensive logging and debugging
- **Testing**: pytest framework with comprehensive coverage
- **Documentation**: Google-style docstrings and markdown

## 📁 File Structure

### **Source Code Organization**

```
src/chromatica/
├── __init__.py                 # Package initialization
├── api/                        # Web API and interface
│   ├── __init__.py
│   ├── main.py                # FastAPI application
│   └── static/                # Web interface files
│       └── index.html         # Interactive color picker
├── core/                       # Core processing logic
│   ├── __init__.py
│   ├── histogram.py           # Histogram generation
│   ├── query.py               # Query processing
│   └── color.py               # Color conversions
├── indexing/                   # Search index management
│   ├── __init__.py
│   ├── faiss_wrapper.py       # FAISS index operations
│   └── duckdb_wrapper.py      # Metadata storage
├── visualization/              # Visualization components
│   ├── __init__.py
│   └── query_viz.py           # Query and result visualization
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── config.py              # Configuration management
└── search.py                   # Main search orchestration
```

### **Supporting Directories**

```
├── tools/                      # Testing and demonstration tools
│   ├── test_*.py              # Component testing scripts
│   ├── demo_*.py              # Feature demonstration scripts
│   └── README.md              # Tools documentation
├── scripts/                    # Production and utility scripts
│   ├── build_index.py         # Index building script
│   └── run_sanity_checks.py   # System validation
├── datasets/                   # Test and validation datasets
│   ├── test-dataset-20/       # Quick testing (20 images)
│   ├── test-dataset-50/       # Validation (50 images)
│   ├── test-dataset-200/      # Performance (200 images)
│   └── test-dataset-5000/     # Production scale (5000 images)
├── docs/                       # Comprehensive documentation
│   ├── README.md              # Project overview
│   ├── progress.md            # Development progress
│   ├── troubleshooting.md     # Common issues and solutions
│   └── *.md                   # Component-specific documentation
└── test_index/                 # Generated search indices
    ├── chromatica_index.faiss # FAISS HNSW index
    └── chromatica_metadata.db # DuckDB metadata store
```

## 🌐 API Design

### **RESTful Endpoints**

#### **Core Search**
```
GET /search
Parameters:
  - colors: Comma-separated hex codes (e.g., "FF0000,00FF00")
  - weights: Comma-separated weights (e.g., "0.7,0.3")
  - k: Number of results (default: 10)

Response:
{
  "query_id": "uuid",
  "query": {"colors": ["FF0000"], "weights": [1.0]},
  "results_count": 5,
  "results": [...],
  "metadata": {"total_time_ms": 150, "ann_time_ms": 50, "rerank_time_ms": 100}
}
```

#### **Query Visualization**
```
GET /visualize/query
Parameters:
  - colors: Comma-separated hex codes
  - weights: Comma-separated weights

Response: PNG image showing query visualization
```

#### **Results Collage**
```
GET /visualize/results
Parameters:
  - colors: Comma-separated hex codes
  - weights: Comma-separated weights
  - k: Number of results to include

Response: PNG image showing results collage
```

#### **System Information**
```
GET /api/info
Response:
{
  "status": "ready",
  "message": "Search engine is ready",
  "index_size": 20,
  "endpoints": ["/search", "/visualize/query", "/visualize/results"]
}
```

### **Error Handling**
- **HTTP Status Codes**: Proper RESTful error responses
- **Error Messages**: Descriptive and actionable error information
- **Validation**: Input sanitization and parameter validation
- **Logging**: Comprehensive error logging for debugging

## 📊 Performance Characteristics

### **Timing Benchmarks**

#### **Index Building**
- **20 images**: 2-5 seconds
- **50 images**: 5-10 seconds
- **200 images**: 20-40 seconds
- **5000 images**: 5-15 minutes

#### **Search Operations**
- **Query processing**: 1-5ms
- **ANN search**: 10-50ms
- **Reranking**: 100-300ms
- **Total search time**: 150-500ms

#### **Visualization**
- **Query visualization**: 50-200ms
- **Results collage**: 100-500ms (depends on image count)
- **Image loading**: 10-50ms per image

### **Memory Usage**
- **Histogram storage**: ~1KB per image
- **FAISS index**: ~100-500MB for 5000 images
- **DuckDB metadata**: ~10-50MB for 5000 images
- **Runtime memory**: 100-500MB depending on dataset size

### **Scalability Limits**
- **Current design**: Up to 10,000 images
- **Memory constraint**: 8GB RAM recommended
- **Storage constraint**: 5GB+ for large datasets
- **Performance degradation**: Linear scaling with dataset size

## 🔒 Security Considerations

### **Input Validation**
- **Color codes**: Hex format validation
- **Weights**: Numeric range validation (0-1)
- **File paths**: Path traversal prevention
- **Query parameters**: Size and format limits

### **Access Control**
- **API endpoints**: Public read access
- **File system**: Restricted to dataset directories
- **Index files**: Read-only access during runtime
- **Logging**: No sensitive data exposure

### **Data Protection**
- **Image content**: No content analysis beyond color
- **User queries**: Temporary storage only
- **Metadata**: No personal information collection
- **Logs**: Sanitized error messages

## 🚀 Scalability Design

### **Horizontal Scaling**
- **Load balancing**: Multiple API instances
- **Shared storage**: Centralized index and metadata
- **Caching**: Redis for query results
- **CDN**: Static file delivery optimization

### **Vertical Scaling**
- **Memory optimization**: Efficient data structures
- **CPU utilization**: Parallel processing where possible
- **Storage optimization**: Compressed index formats
- **Network optimization**: Efficient serialization

### **Performance Optimization**
- **Batch processing**: Efficient bulk operations
- **Lazy loading**: On-demand data retrieval
- **Connection pooling**: Database connection management
- **Async processing**: Non-blocking I/O operations

### **Future Enhancements**
- **Distributed indexing**: Multi-node index building
- **Streaming search**: Real-time result streaming
- **Advanced caching**: Intelligent query result caching
- **Machine learning**: Query optimization and personalization

---

## 🎯 Architecture Summary

Chromatica's architecture is designed around **performance**, **accuracy**, and **usability**:

- **🔧 Core Layer**: Efficient histogram generation and color processing
- **🔍 Search Layer**: Fast ANN search with high-quality reranking
- **🎨 Visualization Layer**: Rich visual feedback and result presentation
- **🌐 API Layer**: RESTful interface with comprehensive documentation

The system achieves **sub-second search times** while maintaining **perceptual accuracy** through sophisticated color science and machine learning algorithms. The **modular design** allows for easy extension and optimization, while the **comprehensive testing** ensures reliability and performance.

---

*For implementation details, see the individual component documentation and source code.*
