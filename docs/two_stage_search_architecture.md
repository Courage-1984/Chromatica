# Two-Stage Search Pipeline Architecture

## Overview

This document explains the foundation and architecture for implementing the complete two-stage search pipeline with Sinkhorn-EMD reranking in the Chromatica color search engine. The system balances speed and accuracy by using FAISS for fast approximate search followed by precise reranking using Earth Mover's Distance.

---

## 🏗️ Foundation Components

### **Stage 1: Fast ANN Search (FAISS HNSW)**

The first stage provides rapid candidate retrieval using Facebook's FAISS library:

#### **AnnIndex Class** (`src/chromatica/indexing/store.py`)

- **Wrapper**: Manages `faiss.IndexHNSWFlat` with M=32 neighbors as specified in configuration
- **Hellinger Transform**: Automatically applies φ(h) = √h transformation to histograms before indexing
- **Vector Management**: Tracks total vectors and provides search functionality
- **Persistence**: Save/load index to/from disk for long-term storage
- **Performance**: Uses float32 for optimal FAISS performance

#### **Technical Implementation Details**

```python
# FAISS index configuration
index = faiss.IndexHNSWFlat(dimension=1152, M=HNSW_M)  # HNSW_M = 32

# Hellinger transform for L2 compatibility
def hellinger_transform(histogram: np.ndarray) -> np.ndarray:
    """Apply Hellinger transform: φ(h) = √h"""
    return np.sqrt(histogram)

# Search method returns (distances, indices) tuples
distances, indices = index.search(query_vector, k=RERANK_K)  # RERANK_K = 200
```

#### **Why Hellinger Transform?**

- **L1 vs L2 Compatibility**: FAISS works optimally with L2 distance, but histograms are L1-normalized
- **Similarity Preservation**: The transform preserves similarity relationships between histograms
- **Mathematical Foundation**: √h maintains the relative importance of color distributions
- **Performance**: Enables efficient FAISS indexing without information loss

### **Stage 2: Precise Reranking (Sinkhorn-EMD)**

The second stage provides accurate distance calculation using Earth Mover's Distance:

#### **MetadataStore Class** (`src/chromatica/indexing/store.py`)

- **DuckDB Integration**: Manages database connection and schema
- **Raw Histogram Storage**: Stores original histograms (not Hellinger-transformed) for accurate reranking
- **Fast Retrieval**: `get_histograms_by_ids()` for candidate reranking
- **Batch Operations**: Efficient handling of multiple candidates simultaneously

#### **Database Schema**

```sql
CREATE TABLE images (
    image_id VARCHAR PRIMARY KEY,
    file_path VARCHAR NOT NULL,
    histogram JSON NOT NULL,  -- Raw 1152-dimensional histogram
    file_size BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast lookups
CREATE INDEX idx_file_path ON images(file_path);
```

---

## 🔄 Two-Stage Pipeline Flow

### **Complete Search Workflow**

```
1. Query Image Processing
   ↓
2. Histogram Generation (CIE Lab, 8×12×12 bins)
   ↓
3. Hellinger Transform: φ(h) = √h
   ↓
4. FAISS HNSW Search → Top K Candidates (k=200)
   ↓
5. Raw Histogram Retrieval from DuckDB
   ↓
6. Sinkhorn-EMD Reranking
   ↓
7. Final Ranked Results
```

### **Data Flow Between Stages**

```
Stage 1 (FAISS):
Input:  Raw histogram (L1-normalized)
Process: Hellinger transform → L2-compatible vector
Output: Top K candidate indices and distances

Stage 2 (Sinkhorn-EMD):
Input:  Raw histograms of candidates
Process: Earth Mover's Distance calculation
Output: Precise similarity scores

Final Ranking:
Combine: ANN scores + EMD distances
Result:  Optimally ranked image list
```

---

## 🎯 Why This Architecture?

### **Speed vs. Accuracy Trade-off**

| Stage       | Method       | Speed        | Accuracy    | Purpose                  |
| ----------- | ------------ | ------------ | ----------- | ------------------------ |
| **Stage 1** | FAISS HNSW   | Milliseconds | Approximate | Fast candidate filtering |
| **Stage 2** | Sinkhorn-EMD | Seconds      | Precise     | Accurate final ranking   |

### **Performance Characteristics**

- **FAISS HNSW**:

  - Search time: O(log n) where n = number of indexed images
  - Memory usage: Optimized for large-scale datasets
  - Scalability: Handles millions of images efficiently

- **Sinkhorn-EMD**:
  - Reranking time: O(k × m²) where k = candidates, m = histogram bins
  - Accuracy: Provides true Earth Mover's Distance approximation
  - Quality: Maintains color distribution similarity relationships

---

## 🚀 Implementation Status

### **✅ Completed Components**

1. **Histogram Generation Pipeline**

   - Core algorithm with tri-linear soft assignment
   - CIE Lab color space conversion (D65 illuminant)
   - 8×12×12 binning grid (1,152 dimensions)

2. **Image Processing Pipeline**

   - Complete preprocessing workflow
   - OpenCV loading, smart resizing, color conversion
   - Integration with histogram generation

3. **FAISS Index Wrapper**

   - HNSW index with M=32 neighbors
   - Automatic Hellinger transform
   - Save/load functionality

4. **DuckDB Metadata Store**
   - Efficient batch operations
   - Raw histogram storage
   - Fast retrieval by indices

### **🔄 Ready for Implementation**

1. **Cost Matrix Pre-computation**

   ```python
   # Pre-compute M matrix for EMD calculations
   # M[i,j] = squared Euclidean distance between Lab bin centers
   M = np.zeros((1152, 1152))
   for i in range(1152):
       for j in range(1152):
           M[i,j] = ||bin_center_i - bin_center_j||²
   ```

2. **Sinkhorn-EMD Integration**

   ```python
   # Using POT library for Sinkhorn approximation
   from POT import sinkhorn2

   def sinkhorn_emd(h1, h2, M, epsilon=0.1):
       """Compute Sinkhorn-approximated EMD"""
       return sinkhorn2(h1, h2, M, epsilon)
   ```

---

## 📊 Performance Expectations

### **Search Performance**

| Dataset Size | FAISS Search | EMD Reranking | Total Time |
| ------------ | ------------ | ------------- | ---------- |
| 1K images    | ~1ms         | ~100ms        | ~101ms     |
| 10K images   | ~2ms         | ~100ms        | ~102ms     |
| 100K images  | ~3ms         | ~100ms        | ~103ms     |
| 1M images    | ~5ms         | ~100ms        | ~105ms     |

### **Memory Usage**

- **FAISS Index**: ~4.6MB per 1K images (1152 × 4 bytes × 1000)
- **DuckDB Storage**: ~9.2MB per 1K images (histograms + metadata)
- **Total**: ~14MB per 1K images for complete system

---

## 🔧 Configuration Parameters

### **Key Constants** (`src/chromatica/utils/config.py`)

```python
# FAISS HNSW parameters
HNSW_M = 32                    # Number of neighbors in HNSW graph
RERANK_K = 200                 # Number of candidates for reranking

# Sinkhorn-EMD parameters
SINKHORN_EPSILON = 0.1         # Regularization parameter

# Histogram parameters
L_BINS = 8                     # Lightness bins
A_BINS = 12                    # Green-red bins
B_BINS = 12                    # Blue-yellow bins
TOTAL_BINS = 1152              # Total dimensions (8×12×12)
```

---

## 📝 Conclusion

The foundation for the two-stage search pipeline is now complete and production-ready. The architecture provides:

1. **Fast Candidate Retrieval**: FAISS HNSW with Hellinger transform
2. **Accurate Reranking**: Sinkhorn-EMD with raw histogram preservation
3. **Efficient Data Management**: DuckDB for metadata and histogram storage
4. **Scalable Performance**: Logarithmic search time, linear memory growth
5. **Future-Ready Design**: Seamless integration with upcoming components

The system is ready for implementing the complete search pipeline, with all necessary infrastructure in place for production-scale color image search.

---

_Last Updated: [Current Date]_  
_Next Review: [Next Week]_  
_Document Version: 1.0_
