# Chromatica Source Code

This directory contains the core source code for the Chromatica color search engine.

## 📁 Directory Structure

```
src/
├── chromatica/           # Main application package
│   ├── __init__.py      # Package initialization
│   ├── core/            # Core histogram generation and color processing
│   ├── indexing/        # FAISS index and DuckDB storage implementation
│   ├── api/             # FastAPI web API endpoints (in development)
│   └── utils/           # Configuration and utility functions
└── __init__.py          # Root package initialization
```

## 🧠 Core Modules

### `chromatica/core/` - Core Color Processing

The core module contains the fundamental algorithms for color histogram generation and image processing.

**Key Components:**

- **`histogram.py`**: Implements tri-linear soft assignment histogram generation
  - `build_histogram()`: Main histogram generation function with 8×12×12 binning
  - `build_histogram_fast()`: Optimized version for rapid prototyping
  - `get_bin_centers()`: Utility for calculating bin center coordinates
  - `get_bin_grid()`: Generates the complete 3D binning grid

**Features:**

- CIE Lab color space processing (D65 illuminant)
- 1,152-dimensional histogram vectors (8×12×12 L*a*b\* bins)
- Tri-linear soft assignment for robust color representation
- L1 normalization for probability distribution
- Comprehensive input validation and error handling

### `chromatica/indexing/` - Vector Storage and Retrieval

The indexing module provides the infrastructure for storing and searching color histograms.

**Key Components:**

- **`store.py`**: FAISS and DuckDB integration
  - `AnnIndex`: FAISS HNSW index wrapper with Hellinger transform
  - `MetadataStore`: DuckDB-based metadata and raw histogram storage
  - Batch operations for efficient indexing and retrieval
- **`pipeline.py`**: Complete image processing pipeline
  - `process_image()`: End-to-end image processing function
  - Automatic resizing, color conversion, and histogram generation

**Features:**

- FAISS HNSW index with M=32 for fast approximate search
- Automatic Hellinger transformation for L2 distance compatibility
- DuckDB for efficient metadata storage and retrieval
- Batch processing capabilities for large datasets

### `chromatica/utils/` - Configuration and Utilities

The utils module provides centralized configuration and helper functions.

**Key Components:**

- **`config.py`**: Global configuration constants
  - Color space binning parameters (L_BINS=8, A_BINS=12, B_BINS=12)
  - Search and reranking parameters (RERANK_K=200)
  - Performance tuning constants (HNSW_M=32, SINKHORN_EPSILON=0.1)
  - Image processing limits (MAX_IMAGE_DIMENSION=256)

**Features:**

- Centralized configuration management
- Validation functions for configuration consistency
- Constants derived from algorithmic specifications

### `chromatica/api/` - Web API (In Development)

The API module will provide REST endpoints for the color search functionality.

**Planned Components:**

- FastAPI application with search endpoints
- Request/response models for color queries
- Integration with the indexing and search pipeline

## 🔧 Technical Implementation

### Color Processing Pipeline

1. **Image Loading**: OpenCV for loading and resizing images
2. **Color Conversion**: sRGB → CIE Lab (D65 illuminant) using scikit-image
3. **Histogram Generation**: Tri-linear soft assignment with 8×12×12 binning
4. **Normalization**: L1 normalization for probability distribution
5. **Indexing**: Hellinger transform for FAISS compatibility

### Performance Characteristics

- **Histogram Generation**: ~200ms per image
- **Memory Usage**: ~4.6KB per histogram
- **Validation Success Rate**: 100%
- **Processing Throughput**: ~5 images/second

### Data Flow

```
Image → Resize (max 256px) → BGR→RGB→Lab → Histogram → Hellinger Transform → FAISS Index
                                                      ↓
                                              Raw Histogram → DuckDB Storage
```

## 🧪 Testing and Validation

### Unit Testing

- Comprehensive validation of histogram generation
- Edge case testing for color values and image formats
- Performance benchmarking and memory usage analysis

### Integration Testing

- End-to-end pipeline validation
- FAISS index performance testing
- DuckDB storage and retrieval validation

### Test Coverage

- Input validation and error handling
- Color space conversion accuracy
- Histogram normalization and quality metrics
- Performance under various image sizes and formats

## 🚀 Development Status

### ✅ Completed

- Core histogram generation module
- Image processing pipeline
- Configuration management
- Comprehensive testing infrastructure

### 🔄 In Progress

- FAISS HNSW index integration
- DuckDB metadata store implementation
- Performance optimization and validation

### 📋 Planned

- FastAPI web endpoints
- Search query processing
- Reranking pipeline implementation
- Production deployment preparation

## 📚 Usage Examples

### Basic Histogram Generation

```python
from chromatica.core.histogram import build_histogram
from chromatica.indexing.pipeline import process_image

# Process an image through the complete pipeline
histogram = process_image("path/to/image.jpg")
print(f"Histogram shape: {histogram.shape}")  # (1152,)
print(f"Sum: {histogram.sum():.6f}")         # 1.000000
```

### Configuration Access

```python
from chromatica.utils.config import TOTAL_BINS, L_BINS, A_BINS, B_BINS

print(f"Total bins: {TOTAL_BINS}")           # 1152
print(f"L* bins: {L_BINS}")                  # 8
print(f"a* bins: {A_BINS}")                  # 12
print(f"b* bins: {B_BINS}")                  # 12
```

## 🔍 Code Quality Standards

- **Type Hints**: All functions include comprehensive type annotations
- **Documentation**: Google-style docstrings with mathematical explanations
- **Error Handling**: Comprehensive validation and descriptive error messages
- **Performance**: Vectorized operations and memory-efficient processing
- **Testing**: Extensive test coverage with edge case validation

## 📖 Related Documentation

- **[Project Plan](docs/.cursor/critical_instructions.md)**: Technical specifications
- **[Progress Report](docs/progress.md)**: Implementation status
- **[Histogram Guide](docs/histogram_generation_guide.md)**: Detailed algorithm documentation
- **[FAISS & DuckDB Guide](docs/faiss_duckdb_guide.md)**: Storage and retrieval implementation

---

**Last Updated**: December 2024  
**Status**: Core modules complete, indexing in progress
