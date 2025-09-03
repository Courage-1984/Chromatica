# Test Datasets Documentation

## Overview

This directory contains comprehensive test datasets for the Chromatica color search engine project. These datasets are designed to support development, testing, and validation across all phases of the project.

## Dataset Structure

| Dataset               | Size                   | Purpose                   | Use Case                                      | Status         |
| --------------------- | ---------------------- | ------------------------- | --------------------------------------------- | -------------- |
| **test-dataset-20**   | 20 images              | Quick development testing | Debugging, unit testing, rapid iteration      | ✅ Complete    |
| **test-dataset-50**   | 50 images              | Small-scale validation    | Feature validation, basic performance testing | ✅ Complete    |
| **test-dataset-200**  | 200 images             | Medium-scale testing      | Performance validation, integration testing   | ✅ Complete    |
| **test-dataset-5000** | 5,000 images (current) | Production-scale testing  | Stress testing, evaluation, ablation studies  | 🔄 In Progress |

## Dataset Details

### test-dataset-20

- **Purpose**: Rapid development and debugging
- **Use Cases**:
  - Quick feature testing
  - Unit test validation
  - Debugging histogram generation
  - Performance baseline measurement
- **Processing Time**: ~4 seconds total (~200ms per image)
- **Memory Usage**: ~92KB total histograms

### test-dataset-50

- **Purpose**: Small-scale validation and testing
- **Use Cases**:
  - Feature validation
  - Basic performance testing
  - Integration testing
  - Quality assurance
- **Processing Time**: ~10 seconds total (~200ms per image)
- **Memory Usage**: ~230KB total histograms

### test-dataset-200

- **Purpose**: Medium-scale testing and performance validation
- **Use Cases**:
  - Performance benchmarking
  - Integration testing
  - Memory usage validation
  - Throughput testing
- **Processing Time**: ~40 seconds total (~200ms per image)
- **Memory Usage**: ~920KB total histograms

### test-dataset-5000

- **Purpose**: Production-scale testing and evaluation
- **Current Status**: 5,000 images (66.7% of target)
- **Target Size**: **7,500 images** (recommended)
- **Use Cases**:
  - Stress testing FAISS HNSW index
  - Memory management validation
  - Statistical significance for evaluation
  - Production pipeline validation
  - Comprehensive ablation studies
- **Processing Time**: ~17 minutes total (~200ms per image)
- **Memory Usage**: ~23MB total histograms

## Dataset Expansion Recommendation

### Why 7,500 Images?

The recommended expansion of test-dataset-999 to 7,500 images provides:

1. **Performance Validation**: Test FAISS HNSW index with realistic workloads
2. **Memory Testing**: Validate memory management under production conditions
3. **Statistical Significance**: Enable meaningful evaluation metrics
4. **Stress Testing**: Identify bottlenecks and performance limits
5. **Production Readiness**: Validate complete pipeline at scale

### Memory Requirements

Based on the project's technical specifications:

| Dataset Size                  | Raw Histograms | FAISS Index | Total RAM |
| ----------------------------- | -------------- | ----------- | --------- |
| 5,000 images (current)        | ~23MB          | ~40MB       | ~63MB     |
| 7,500 images (target)         | ~34.4MB        | ~60.2MB     | ~94.6MB   |
| 25,000 images (Unsplash Lite) | ~114.7MB       | ~200.7MB    | ~315.4MB  |

### Implementation Priority

**High Priority**: Expand test-dataset-999 to 7,500 images

- Enables comprehensive production-scale testing
- Supports Week 2-3 FAISS and DuckDB implementation
- Provides foundation for evaluation and ablation studies

## Usage Guidelines

### Development Phase

- **Use**: test-dataset-20
- **Purpose**: Rapid iteration and debugging
- **When**: Implementing new features, fixing bugs

### Validation Phase

- **Use**: test-dataset-50 and test-dataset-200
- **Purpose**: Feature validation and performance testing
- **When**: Testing new implementations, benchmarking

### Production Testing

- **Use**: test-dataset-999 (expanded)
- **Purpose**: Comprehensive evaluation and stress testing
- **When**: Final validation, performance optimization

### Final Validation

- **Use**: Unsplash Lite (25k) and COCO (5k)
- **Purpose**: Production validation and evaluation
- **When**: Final testing, performance reporting

## Technical Specifications

### Image Requirements

- **Formats**: JPG, PNG
- **Max Dimension**: 256px (handled by histogram generation)
- **Color Space**: sRGB (converted to CIE Lab)
- **Quality**: High-quality images with diverse color palettes

### Histogram Specifications

- **Dimensions**: 1,152 (8×12×12 L*a*b\* bins)
- **Normalization**: L1 normalization (sum = 1.0)
- **Data Type**: float32
- **Size per Image**: ~4.6KB

### Processing Pipeline

1. **Image Loading**: OpenCV for loading and resizing
2. **Color Conversion**: sRGB → CIE Lab (D65 illuminant)
3. **Histogram Generation**: Tri-linear soft assignment
4. **Validation**: Shape, normalization, bounds checking
5. **Storage**: Raw histograms for reranking, Hellinger-transformed for FAISS

## Testing and Validation

### Histogram Testing Tool

The project includes a comprehensive testing tool (`tools/test_histogram_generation.py`) that:

- **Processes**: Single images or entire datasets
- **Generates**: 6 different report types
- **Validates**: Histogram specifications and quality
- **Outputs**: Organized histograms/ and reports/ directories

### Validation Criteria

All histograms must pass:

- **Shape**: Exactly 1,152 dimensions
- **Normalization**: L1 normalized (sum = 1.0)
- **Bounds**: All values ≥ 0
- **Quality**: Entropy and sparsity analysis

### Performance Targets

- **Processing Time**: ~200ms per image
- **Validation Success**: 100%
- **Memory Efficiency**: ~4.6KB per histogram
- **Throughput**: ~5 images/second

## File Organization

```
datasets/
├── test-dataset-20/          # 20 images for development
├── test-dataset-50/          # 50 images for validation
├── test-dataset-200/         # 200 images for testing
├── test-dataset-5000/        # 5,000 images (expand to 7,500)
└── README.md                 # This documentation
```

## Next Steps

### Immediate Actions

1. **Expand test-dataset-5000** to 7,500 images (only 2,500 more needed!)
2. **Validate all datasets** using histogram testing tool
3. **Document dataset characteristics** and metadata
4. **Prepare for FAISS index testing** (Week 2)

### Future Considerations

1. **Automated dataset management** scripts
2. **Dataset versioning** and change tracking
3. **Performance benchmarking** across dataset sizes
4. **Quality metrics** and dataset health monitoring

## Contributing

When adding new images to test datasets:

1. **Ensure diversity** in color palettes and content
2. **Maintain quality** (high-resolution, clear images)
3. **Follow naming conventions** (consistent file naming)
4. **Update documentation** when dataset sizes change
5. **Validate histograms** after additions

## References

- **Project Plan**: `docs/.cursor/critical_instructions.md`
- **Testing Tool**: `tools/test_histogram_generation.py`
- **Configuration**: `src/chromatica/utils/config.py`
- **Histogram Module**: `src/chromatica/core/histogram.py`

---

_Last Updated: December 2024_
_Project: Chromatica Color Search Engine_
