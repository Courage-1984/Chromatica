---

## Usage Examples

### Overview

This section provides comprehensive usage examples for all aspects of the histogram generation system, from basic usage to advanced scenarios and integration patterns.

### Basic Histogram Generation

#### 1. Simple Histogram Generation

```python
from chromatica.core.histogram import build_histogram
import numpy as np

# Create sample Lab pixels (L*, a*, b*)
lab_pixels = np.array([
    [50.0, 10.0, -20.0],   # Medium gray with slight green tint
    [80.0, -5.0, 15.0],    # Light gray with slight blue tint
    [30.0, 20.0, -10.0],   # Dark gray with slight red tint
    [90.0, 0.0, 0.0],      # Very light neutral gray
    [10.0, 0.0, 0.0]       # Very dark neutral gray
])

# Generate histogram
histogram = build_histogram(lab_pixels)

print(f"Histogram shape: {histogram.shape}")
print(f"Sum: {histogram.sum():.6f}")
print(f"Min value: {histogram.min():.6f}")
print(f"Max value: {histogram.max():.6f}")
print(f"Non-zero bins: {np.count_nonzero(histogram)}/1152")
```

#### 2. Loading and Converting Images

```python
import cv2
from skimage import color
from chromatica.core.histogram import build_histogram

def image_to_histogram(image_path):
    """Convert an image file to a color histogram."""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if too large (optional)
    max_dim = 256
    height, width = image_rgb.shape[:2]
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        image_rgb = cv2.resize(image_rgb, (new_width, new_height))
    
    # Convert to Lab color space
    image_lab = color.rgb2lab(image_rgb)
    
    # Reshape to (N, 3) array
    lab_pixels = image_lab.reshape(-1, 3)
    
    # Generate histogram
    histogram = build_histogram(lab_pixels)
    
    return histogram, image_lab.shape[:2]

# Example usage
try:
    histogram, original_size = image_to_histogram("path/to/image.jpg")
    print(f"Original size: {original_size}")
    print(f"Histogram generated successfully: {histogram.shape}")
except Exception as e:
    print(f"Error: {e}")
```

#### 3. Batch Processing Multiple Images

```python
import os
from pathlib import Path
from chromatica.core.histogram import build_histogram

def process_image_directory(directory_path):
    """Process all images in a directory and generate histograms."""
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    results = {}
    
    for image_file in Path(directory_path).iterdir():
        if image_file.suffix.lower() in image_extensions:
            try:
                print(f"Processing: {image_file.name}")
                
                # Convert image to histogram
                histogram, size = image_to_histogram(str(image_file))
                
                results[image_file.name] = {
                    'histogram': histogram,
                    'size': size,
                    'success': True
                }
                
                print(f"  ✓ Success: {size} → {histogram.shape}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[image_file.name] = {
                    'error': str(e),
                    'success': False
                }
    
    return results

# Example usage
image_dir = "./images/"
results = process_image_directory(image_dir)

print(f"\nProcessing complete:")
print(f"  Total images: {len(results)}")
print(f"  Successful: {sum(1 for r in results.values() if r['success'])}")
print(f"  Failed: {sum(1 for r in results.values() if not r['success'])}")
```

### Advanced Histogram Analysis

#### 1. Histogram Comparison and Similarity

```python
import numpy as np
from chromatica.core.histogram import build_histogram

def histogram_similarity(hist1, hist2, method='cosine'):
    """Calculate similarity between two histograms."""
    
    if method == 'cosine':
        # Cosine similarity (0-1, higher = more similar)
        dot_product = np.dot(hist1, hist2)
        norm1 = np.linalg.norm(hist1)
        norm2 = np.linalg.norm(hist2)
        return dot_product / (norm1 * norm2)
    
    elif method == 'euclidean':
        # Euclidean distance (0-∞, lower = more similar)
        return np.linalg.norm(hist1 - hist2)
    
    elif method == 'manhattan':
        # Manhattan distance (0-∞, lower = more similar)
        return np.sum(np.abs(hist1 - hist2))
    
    elif method == 'hellinger':
        # Hellinger distance (0-2, lower = more similar)
        sqrt_hist1 = np.sqrt(hist1)
        sqrt_hist2 = np.sqrt(hist2)
        return np.linalg.norm(sqrt_hist1 - sqrt_hist2)
    
    else:
        raise ValueError(f"Unknown method: {method}")

def find_similar_images(query_histogram, image_histograms, method='cosine', top_k=5):
    """Find most similar images to a query image."""
    
    similarities = []
    
    for image_name, data in image_histograms.items():
        if data['success']:
            if method == 'cosine':
                sim = histogram_similarity(query_histogram, data['histogram'], method)
            else:
                # For distance metrics, convert to similarity (1 / (1 + distance))
                dist = histogram_similarity(query_histogram, data['histogram'], method)
                sim = 1 / (1 + dist)
            
            similarities.append((image_name, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

# Example usage
# Assuming we have processed images from previous example
query_image = "query.jpg"
query_hist, _ = image_to_histogram(query_image)

similar_images = find_similar_images(query_hist, results, method='cosine', top_k=5)

print(f"Most similar images to {query_image}:")
for i, (image_name, similarity) in enumerate(similar_images, 1):
    print(f"  {i}. {image_name}: {similarity:.4f}")
```

#### 2. Histogram Clustering and Analysis

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def analyze_histogram_clusters(histograms, n_clusters=3):
    """Cluster histograms to find color themes."""
    
    # Extract histogram data
    histogram_data = []
    image_names = []
    
    for image_name, data in histograms.items():
        if data['success']:
            histogram_data.append(data['histogram'])
            image_names.append(image_name)
    
    histogram_matrix = np.array(histogram_data)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    histograms_2d = pca.fit_transform(histogram_matrix)
    
    # Cluster histograms
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(histogram_matrix)
    
    # Visualize clusters
    plt.figure(figsize=(10, 8))
    
    for i in range(n_clusters):
        cluster_points = histograms_2d[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   label=f'Cluster {i}', alpha=0.7)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Histogram Clusters (PCA projection)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Analyze clusters
    cluster_analysis = {}
    for i in range(n_clusters):
        cluster_images = [image_names[j] for j in range(len(image_names)) 
                         if cluster_labels[j] == i]
        cluster_histograms = histogram_matrix[cluster_labels == i]
        
        # Calculate cluster centroid
        centroid = np.mean(cluster_histograms, axis=0)
        
        # Calculate average entropy and sparsity
        entropies = []
        sparsities = []
        for hist in cluster_histograms:
            non_zero = hist[hist > 0]
            if len(non_zero) > 0:
                entropy = -np.sum(non_zero * np.log2(non_zero))
                sparsity = 1.0 - (np.count_nonzero(hist) / 1152)
                entropies.append(entropy)
                sparsities.append(sparsity)
        
        cluster_analysis[f'cluster_{i}'] = {
            'image_count': len(cluster_images),
            'images': cluster_images,
            'centroid_entropy': -np.sum(centroid[centroid > 0] * np.log2(centroid[centroid > 0])),
            'avg_entropy': np.mean(entropies),
            'avg_sparsity': np.mean(sparsities)
        }
    
    return cluster_analysis

# Example usage
cluster_results = analyze_histogram_clusters(results, n_clusters=3)

print("Cluster Analysis Results:")
for cluster_name, cluster_data in cluster_results.items():
    print(f"\n{cluster_name.upper()}:")
    print(f"  Image count: {cluster_data['image_count']}")
    print(f"  Average entropy: {cluster_data['avg_entropy']:.3f}")
    print(f"  Average sparsity: {cluster_data['avg_sparsity']:.3f}")
    print(f"  Images: {', '.join(cluster_data['images'][:5])}")
    if len(cluster_data['images']) > 5:
        print(f"  ... and {len(cluster_data['images']) - 5} more")
```

### Testing Tool Usage Examples

#### 1. Basic Testing Tool Usage

```python
from tools.test_histogram_generation import HistogramTester

# Initialize tester with default settings
tester = HistogramTester()

# Test single image
result = tester.test_single_image("path/to/image.jpg")

if result.get('success', False):
    print(f"✓ Histogram generated successfully")
    print(f"  Shape: {result['histogram']['shape']}")
    print(f"  Entropy: {result['validation']['metrics']['entropy']:.4f}")
    print(f"  Processing time: {result['performance']['mean_time_ms']:.2f} ms")
else:
    print(f"✗ Failed: {result.get('error', 'Unknown error')}")
```

#### 2. Custom Testing Configuration

```python
# Create tester with specific configuration
tester = HistogramTester(
    output_format='both',    # Generate both JSON and CSV
    visualize=True           # Create visualization plots
)

# Test directory with custom settings
results = tester.test_directory("./test_images/")

# Analyze results
successful = [r for r in results if r.get('success', False)]
failed = [r for r in results if not r.get('success', False)]

print(f"Testing Results:")
print(f"  Total images: {len(results)}")
print(f"  Successful: {len(successful)}")
print(f"  Failed: {len(failed)}")

if failed:
    print(f"\nFailed images:")
    for item in failed:
        print(f"  - {item['image_path']}: {item.get('error', 'Unknown error')}")
```

#### 3. Performance Analysis

```python
def analyze_performance_trends(results):
    """Analyze performance trends across different image types."""
    
    # Group by image dimensions
    performance_by_size = {}
    
    for result in results:
        if result.get('success', False):
            size = result['image_info']['original_size']
            pixels = result['image_info']['lab_pixels']
            time_ms = result['performance']['mean_time_ms']
            
            # Categorize by total pixels
            if pixels <= 10000:
                category = "Small (≤10K pixels)"
            elif pixels <= 50000:
                category = "Medium (10K-50K pixels)"
            else:
                category = "Large (>50K pixels)"
            
            if category not in performance_by_size:
                performance_by_size[category] = []
            
            performance_by_size[category].append({
                'pixels': pixels,
                'time_ms': time_ms,
                'throughput': result['performance']['pixels_per_second']
            })
    
    # Analyze each category
    print("Performance Analysis by Image Size:")
    print("=" * 50)
    
    for category, data in performance_by_size.items():
        times = [item['time_ms'] for item in data]
        throughputs = [item['throughput'] for item in data]
        pixels_list = [item['pixels'] for item in data]
        
        print(f"\n{category}:")
        print(f"  Image count: {len(data)}")
        print(f"  Avg processing time: {np.mean(times):.2f} ms")
        print(f"  Avg throughput: {np.mean(throughputs):.0f} pixels/sec")
        print(f"  Time range: {np.min(times):.2f} - {np.max(times):.2f} ms")
        
        # Check for performance outliers
        mean_time = np.mean(times)
        std_time = np.std(times)
        outliers = [item for item in data if abs(item['time_ms'] - mean_time) > 2 * std_time]
        
        if outliers:
            print(f"  Performance outliers: {len(outliers)} images")
            for outlier in outliers:
                print(f"    - {outlier['pixels']} pixels: {outlier['time_ms']:.2f} ms")

# Example usage
analyze_performance_trends(results)
```

### Integration Examples

#### 1. Integration with Data Pipeline

```python
import pandas as pd
from pathlib import Path

def create_histogram_dataset(image_directory, output_file):
    """Create a dataset of image histograms for machine learning."""
    
    # Process all images
    tester = HistogramTester(output_format='csv', visualize=False)
    results = tester.test_directory(image_directory)
    
    # Convert to DataFrame
    df_data = []
    for result in results:
        if result.get('success', False):
            # Extract histogram data
            histogram = np.load(result['histogram']['data_path'])
            
            # Create row with histogram features
            row = {
                'image_path': result['image_path'],
                'image_name': Path(result['image_path']).name,
                'width': result['image_info']['original_size'][0],
                'height': result['image_info']['original_size'][1],
                'total_pixels': result['image_info']['lab_pixels'],
                'entropy': result['validation']['metrics']['entropy'],
                'sparsity': result['validation']['metrics']['sparsity'],
                'processing_time_ms': result['performance']['mean_time_ms']
            }
            
            # Add histogram bins as features
            for i, value in enumerate(histogram):
                row[f'bin_{i:04d}'] = value
            
            df_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Save to file
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(f"Shape: {df.shape}")
    
    return df

# Example usage
dataset = create_histogram_dataset("./images/", "histogram_dataset.csv")
```

#### 2. Real-time Histogram Generation

```python
import cv2
import numpy as np
from chromatica.core.histogram import build_histogram
import time

def real_time_histogram_generation(video_source=0):
    """Generate histograms from video stream in real-time."""
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Real-time histogram generation started. Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for performance
            frame_small = cv2.resize(frame_rgb, (128, 128))
            
            # Convert to Lab
            frame_lab = color.rgb2lab(frame_small)
            lab_pixels = frame_lab.reshape(-1, 3)
            
            # Generate histogram
            histogram = build_histogram(lab_pixels)
            
            # Calculate metrics
            entropy = -np.sum(histogram[histogram > 0] * np.log2(histogram[histogram > 0]))
            sparsity = 1.0 - (np.count_nonzero(histogram) / 1152)
            
            # Display metrics on frame
            cv2.putText(frame, f"Entropy: {entropy:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Sparsity: {sparsity:.3f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Real-time Histogram Generation', frame)
            
            frame_count += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate performance
        end_time = time.time()
        total_time = end_time - start_time
        fps = frame_count / total_time
        
        print(f"\nPerformance Summary:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average FPS: {fps:.2f}")

# Example usage (uncomment to run)
# real_time_histogram_generation()
```

### Error Handling and Debugging

#### 1. Comprehensive Error Handling

```python
def robust_histogram_generation(image_path, fallback_method=True):
    """Generate histogram with comprehensive error handling."""
    
    try:
        # Try primary method
        histogram, size = image_to_histogram(image_path)
        return histogram, size, "primary", None
        
    except Exception as e:
        if not fallback_method:
            raise
        
        print(f"Primary method failed: {e}")
        print("Attempting fallback method...")
        
        try:
            # Fallback: load with PIL and convert
            from PIL import Image
            import numpy as np
            
            # Load image
            pil_image = Image.open(image_path)
            pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Convert to Lab
            image_lab = color.rgb2lab(image_array)
            lab_pixels = image_lab.reshape(-1, 3)
            
            # Generate histogram
            histogram = build_histogram(lab_pixels)
            
            return histogram, image_lab.shape[:2], "fallback", None
            
        except Exception as fallback_error:
            error_msg = f"Both methods failed. Primary: {e}, Fallback: {fallback_error}"
            return None, None, "failed", error_msg

# Example usage
histogram, size, method, error = robust_histogram_generation("problematic_image.jpg")

if histogram is not None:
    print(f"✓ Histogram generated using {method} method")
    print(f"  Size: {size}")
else:
    print(f"✗ Failed: {error}")
```

#### 2. Debugging Histogram Issues

```python
def debug_histogram_issues(histogram, image_path):
    """Debug common histogram generation issues."""
    
    issues = []
    
    # Check shape
    if histogram.shape != (1152,):
        issues.append(f"Shape mismatch: expected (1152,), got {histogram.shape}")
    
    # Check normalization
    sum_val = histogram.sum()
    if not np.isclose(sum_val, 1.0, atol=1e-6):
        issues.append(f"Normalization issue: sum = {sum_val:.6f}, expected 1.0")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(histogram)):
        issues.append("Found NaN values in histogram")
    
    if np.any(np.isinf(histogram)):
        issues.append("Found infinite values in histogram")
    
    # Check bounds
    if np.any(histogram < 0):
        issues.append("Found negative values in histogram")
    
    if np.any(histogram > 1.0):
        issues.append("Found values > 1.0 in histogram")
    
    # Check for all-zero histogram
    if np.all(histogram == 0):
        issues.append("Histogram is all zeros - possible conversion error")
    
    # Check for extremely sparse histogram
    non_zero_count = np.count_nonzero(histogram)
    if non_zero_count < 10:
        issues.append(f"Extremely sparse histogram: only {non_zero_count} non-zero bins")
    
    # Check for dominant single bin
    max_bin_value = np.max(histogram)
    if max_bin_value > 0.5:
        issues.append(f"Dominant bin detected: {max_bin_value:.3f} > 0.5")
    
    if issues:
        print(f"Debugging {image_path}:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"✓ {image_path}: No issues detected")
    
    return issues

# Example usage
debug_histogram_issues(histogram, "test_image.jpg")
```

### Best Practices Summary

1. **Always validate input data** - Check image loading and Lab conversion
2. **Use appropriate error handling** - Implement fallback methods for robustness
3. **Monitor performance** - Track processing time and memory usage
4. **Validate output** - Check histogram properties and quality metrics
5. **Use batch processing** - Process multiple images efficiently
6. **Implement caching** - Save and reuse histograms when possible
7. **Monitor quality** - Track entropy, sparsity, and other quality metrics
8. **Document exceptions** - Log errors for debugging and improvement

---

## Advanced Features

### Overview

This section covers advanced features and capabilities of the histogram generation system, including optimization techniques, custom configurations, and integration patterns.

### Performance Optimization

#### 1. Memory-Efficient Processing

```python
def memory_efficient_batch_processing(image_paths, batch_size=10):
    """Process images in batches to manage memory usage."""

    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")

        batch_results = []
        for image_path in batch_paths:
            try:
                histogram, size = image_to_histogram(image_path)
                batch_results.append({
                    'image_path': image_path,
                    'histogram': histogram,
                    'size': size,
                    'success': True
                })
            except Exception as e:
                batch_results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })

        results.extend(batch_results)

        # Clear batch data to free memory
        del batch_results

        # Optional: force garbage collection
        import gc
        gc.collect()

    return results

# Example usage
image_paths = [f"image_{i}.jpg" for i in range(100)]
results = memory_efficient_batch_processing(image_paths, batch_size=20)
```

#### 2. Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def parallel_histogram_generation(image_paths, max_workers=None):
    """Generate histograms using parallel processing."""

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(image_to_histogram, path): path
            for path in image_paths
        }

        # Collect results as they complete
        for future in as_completed(future_to_path):
            image_path = future_to_path[future]

            try:
                histogram, size = future.result()
                results[image_path] = {
                    'histogram': histogram,
                    'size': size,
                    'success': True
                }
                print(f"✓ Completed: {image_path}")

            except Exception as e:
                results[image_path] = {
                    'error': str(e),
                    'success': False
                }
                print(f"✗ Failed: {image_path} - {e}")

    return results

# Example usage
image_paths = [f"image_{i}.jpg" for i in range(50)]
results = parallel_histogram_generation(image_paths, max_workers=4)
```

#### 3. Caching and Persistence

```python
import pickle
import hashlib
from pathlib import Path

class HistogramCache:
    """Cache for storing and retrieving generated histograms."""

    def __init__(self, cache_dir="./histogram_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, image_path):
        """Generate cache key based on image path and modification time."""
        stat = Path(image_path).stat()
        key_data = f"{image_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, image_path):
        """Retrieve histogram from cache if available."""
        cache_key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Verify the cached image still exists and hasn't changed
                if (Path(image_path).exists() and
                    Path(image_path).stat().st_mtime == cached_data['mtime']):
                    return cached_data['histogram']

            except Exception as e:
                print(f"Cache read error: {e}")

        return None

    def set(self, image_path, histogram):
        """Store histogram in cache."""
        cache_key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            cached_data = {
                'histogram': histogram,
                'mtime': Path(image_path).stat().st_mtime,
                'timestamp': time.time()
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)

        except Exception as e:
            print(f"Cache write error: {e}")

    def clear(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("Cache cleared")

# Example usage
cache = HistogramCache()

def cached_histogram_generation(image_path):
    """Generate histogram with caching."""

    # Check cache first
    cached_histogram = cache.get(image_path)
    if cached_histogram is not None:
        print(f"Cache hit: {image_path}")
        return cached_histogram

    # Generate new histogram
    print(f"Cache miss: {image_path}")
    histogram, size = image_to_histogram(image_path)

    # Store in cache
    cache.set(image_path, histogram)

    return histogram
```

### Custom Configurations

#### 1. Custom Binning Schemes

```python
def create_custom_binning_scheme(l_bins=8, a_bins=12, b_bins=12):
    """Create custom binning scheme for histogram generation."""

    # Define Lab ranges
    l_range = (0, 100)
    a_range = (-86, 98)
    b_range = (-108, 95)

    # Calculate bin sizes
    l_bin_size = (l_range[1] - l_range[0]) / l_bins
    a_bin_size = (a_range[1] - a_range[0]) / a_bins
    b_bin_size = (b_range[1] - b_range[0]) / b_bins

    # Create bin centers
    l_centers = np.linspace(l_range[0] + l_bin_size/2, l_range[1] - l_bin_size/2, l_bins)
    a_centers = np.linspace(a_range[0] + a_bin_size/2, a_range[1] - a_bin_size/2, a_bins)
    b_centers = np.linspace(b_range[0] + b_bin_size/2, b_range[1] - b_bin_size/2, b_bins)

    return {
        'l_bins': l_bins,
        'a_bins': a_bins,
        'b_bins': b_bins,
        'total_bins': l_bins * a_bins * b_bins,
        'l_centers': l_centers,
        'a_centers': a_centers,
        'b_centers': b_centers,
        'l_bin_size': l_bin_size,
        'a_bin_size': a_bin_size,
        'b_bin_size': b_bin_size,
        'ranges': (l_range, a_range, b_range)
    }

def build_custom_histogram(lab_pixels, binning_scheme):
    """Build histogram using custom binning scheme."""

    l_bins = binning_scheme['l_bins']
    a_bins = binning_scheme['a_bins']
    b_bins = binning_scheme['b_bins']
    total_bins = binning_scheme['total_bins']

    # Initialize histogram
    histogram = np.zeros((l_bins, a_bins, b_bins))

    # Get bin centers and sizes
    l_centers = binning_scheme['l_centers']
    a_centers = binning_scheme['a_centers']
    b_centers = binning_scheme['b_centers']

    l_bin_size = binning_scheme['l_bin_size']
    a_bin_size = binning_scheme['a_bin_size']
    b_bin_size = binning_scheme['b_bin_size']

    # Process each pixel
    for pixel in lab_pixels:
        l, a, b = pixel

        # Find bin indices
        l_idx = int((l - 0) / l_bin_size)
        a_idx = int((a - (-86)) / a_bin_size)
        b_idx = int((b - (-108)) / b_bin_size)

        # Clamp indices to valid range
        l_idx = max(0, min(l_bins - 1, l_idx))
        a_idx = max(0, min(a_bins - 1, a_idx))
        b_idx = max(0, min(b_bins - 1, b_idx))

        # Increment bin count
        histogram[l_idx, a_idx, b_idx] += 1

    # Flatten and normalize
    histogram_flat = histogram.flatten()
    histogram_flat = histogram_flat / histogram_flat.sum()

    return histogram_flat

# Example usage
custom_scheme = create_custom_binning_scheme(l_bins=16, a_bins=16, b_bins=16)
print(f"Custom scheme: {custom_scheme['total_bins']} bins")

# Use custom scheme
lab_pixels = np.array([[50.0, 10.0, -20.0], [80.0, -5.0, 15.0]])
custom_histogram = build_custom_histogram(lab_pixels, custom_scheme)
print(f"Custom histogram shape: {custom_histogram.shape}")
```

#### 2. Adaptive Binning

```python
def adaptive_histogram_binning(lab_pixels, target_bins=1152):
    """Create adaptive binning based on data distribution."""

    # Analyze data distribution
    l_vals = lab_pixels[:, 0]
    a_vals = lab_pixels[:, 1]
    b_vals = lab_pixels[:, 2]

    # Calculate optimal bin distribution
    l_std = np.std(l_vals)
    a_std = np.std(a_vals)
    b_std = np.std(b_vals)

    # Allocate bins proportionally to standard deviation
    total_std = l_std + a_std + b_std
    l_bins = max(4, int(target_bins * (l_std / total_std)))
    a_bins = max(4, int(target_bins * (a_std / total_std)))
    b_bins = max(4, target_bins - l_bins - a_bins)

    # Ensure total bins is close to target
    while l_bins * a_bins * b_bins > target_bins * 1.2:
        if l_bins > 4:
            l_bins -= 1
        elif a_bins > 4:
            a_bins -= 1
        elif b_bins > 4:
            b_bins -= 1

    print(f"Adaptive binning: L={l_bins}, a={a_bins}, b={b_bins}, total={l_bins*a_bins*b_bins}")

    # Create binning scheme
    return create_custom_binning_scheme(l_bins, a_bins, b_bins)

# Example usage
adaptive_scheme = adaptive_histogram_binning(lab_pixels, target_bins=1000)
adaptive_histogram = build_custom_histogram(lab_pixels, adaptive_scheme)
```

### Advanced Analysis Techniques

#### 1. Histogram Segmentation

```python
def segment_histogram_by_color(histogram, l_bins=8, a_bins=12, b_bins=12):
    """Segment histogram into color regions."""

    # Reshape to 3D
    hist_3d = histogram.reshape(l_bins, a_bins, b_bins)

    # Define color regions
    color_regions = {
        'red': (slice(None), slice(a_bins//2, None), slice(None)),
        'green': (slice(None), slice(None, a_bins//2), slice(None)),
        'blue': (slice(None), slice(None), slice(None, b_bins//2)),
        'yellow': (slice(None), slice(None), slice(b_bins//2, None)),
        'light': (slice(l_bins//2, None), slice(None), slice(None)),
        'dark': (slice(None, l_bins//2), slice(None), slice(None))
    }

    # Calculate region statistics
    region_stats = {}

    for region_name, region_slice in color_regions.items():
        region_hist = hist_3d[region_slice]
        region_sum = region_hist.sum()

        if region_sum > 0:
            region_entropy = -np.sum(region_hist * np.log2(region_hist + 1e-10))
            region_sparsity = 1.0 - (np.count_nonzero(region_hist) / region_hist.size)
        else:
            region_entropy = 0
            region_sparsity = 1.0

        region_stats[region_name] = {
            'sum': region_sum,
            'entropy': region_entropy,
            'sparsity': region_sparsity,
            'percentage': region_sum * 100
        }

    return region_stats

# Example usage
region_analysis = segment_histogram_by_color(histogram)
print("Color Region Analysis:")
for region, stats in region_analysis.items():
    print(f"  {region}: {stats['percentage']:.1f}%, entropy: {stats['entropy']:.3f}")
```

#### 2. Temporal Histogram Analysis

```python
def analyze_temporal_changes(histogram_sequence, window_size=5):
    """Analyze changes in histograms over time."""

    if len(histogram_sequence) < window_size:
        return None

    changes = []

    for i in range(len(histogram_sequence) - window_size + 1):
        window = histogram_sequence[i:i + window_size]

        # Calculate average histogram for window
        avg_hist = np.mean(window, axis=0)

        # Calculate variance within window
        var_hist = np.var(window, axis=0)

        # Calculate stability metric
        stability = 1.0 / (1.0 + np.mean(var_hist))

        changes.append({
            'window_start': i,
            'window_end': i + window_size - 1,
            'average_histogram': avg_hist,
            'variance': var_hist,
            'stability': stability,
            'change_magnitude': np.mean(var_hist)
        })

    return changes

# Example usage (for video or time-series data)
# histogram_sequence = [hist1, hist2, hist3, ...]
# temporal_analysis = analyze_temporal_changes(histogram_sequence, window_size=10)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Image Loading Problems

**Problem**: Images fail to load or convert

```python
# Error: "Could not load image"
# Solution: Check file path and format support
import cv2

def robust_image_loading(image_path):
    """Robust image loading with multiple fallbacks."""

    # Try OpenCV first
    image = cv2.imread(image_path)
    if image is not None:
        return image

    # Try PIL as fallback
    try:
        from PIL import Image
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert('RGB')
        return np.array(pil_image)
    except Exception as e:
        print(f"PIL fallback failed: {e}")

    # Try scikit-image
    try:
        from skimage import io
        return io.imread(image_path)
    except Exception as e:
        print(f"scikit-image fallback failed: {e}")

    raise ValueError(f"Could not load image with any method: {image_path}")
```

**Problem**: Memory errors with large images

```python
# Solution: Implement progressive loading
def progressive_image_processing(image_path, max_pixels=1000000):
    """Process large images progressively to avoid memory issues."""

    # Get image dimensions first
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    total_pixels = height * width

    if total_pixels > max_pixels:
        # Calculate scale factor
        scale = np.sqrt(max_pixels / total_pixels)
        new_height = int(height * scale)
        new_width = int(width * scale)

        print(f"Resizing {height}x{width} to {new_height}x{new_width}")
        image = cv2.resize(image, (new_width, new_height))

    return image
```

#### 2. Histogram Generation Issues

**Problem**: Histogram shape is incorrect

```python
# Error: Expected shape (1152,), got (1153,)
# Solution: Check binning configuration

def verify_histogram_shape(histogram, expected_shape=(1152,)):
    """Verify histogram has correct shape."""

    if histogram.shape != expected_shape:
        print(f"Shape mismatch: expected {expected_shape}, got {histogram.shape}")

        # Check if it's a multiple of expected
        if histogram.size % expected_shape[0] == 0:
            print("Attempting to reshape...")
            try:
                # Try to reshape to expected dimensions
                reshaped = histogram.reshape(expected_shape)
                print("Successfully reshaped")
                return reshaped
            except ValueError:
                print("Reshape failed")

        return None

    return histogram
```

**Problem**: Histogram normalization fails

```python
# Error: Histogram sum is not 1.0
# Solution: Check for numerical issues

def fix_histogram_normalization(histogram, tolerance=1e-6):
    """Fix histogram normalization issues."""

    current_sum = histogram.sum()

    if np.isclose(current_sum, 1.0, atol=tolerance):
        return histogram  # Already normalized

    if current_sum == 0:
        print("Warning: Histogram is all zeros")
        return histogram

    # Normalize
    normalized = histogram / current_sum

    # Verify
    new_sum = normalized.sum()
    if not np.isclose(new_sum, 1.0, atol=tolerance):
        print(f"Warning: Normalization still off: {new_sum}")

    return normalized
```

#### 3. Performance Issues

**Problem**: Processing is too slow

```python
# Solution: Profile and optimize

import cProfile
import pstats

def profile_histogram_generation(image_path):
    """Profile histogram generation performance."""

    profiler = cProfile.Profile()
    profiler.enable()

    # Run histogram generation
    histogram, size = image_to_histogram(image_path)

    profiler.disable()

    # Print statistics
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

    return histogram, size

# Example usage
histogram, size = profile_histogram_generation("large_image.jpg")
```

**Problem**: Memory usage is too high

```python
# Solution: Monitor memory usage

import psutil
import os

def monitor_memory_usage():
    """Monitor memory usage during processing."""

    process = psutil.Process(os.getpid())

    def get_memory_info():
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }

    return get_memory_info

# Example usage
memory_monitor = monitor_memory_usage()

print("Before processing:", memory_monitor())
histogram = build_histogram(lab_pixels)
print("After processing:", memory_monitor())
```

#### 4. Validation Failures

**Problem**: Histograms fail validation

```python
# Solution: Comprehensive debugging

def debug_validation_failure(histogram, image_path):
    """Debug histogram validation failures."""

    print(f"Debugging histogram for: {image_path}")
    print(f"Shape: {histogram.shape}")
    print(f"Data type: {histogram.dtype}")
    print(f"Memory usage: {histogram.nbytes / 1024:.2f} KB")

    # Check for common issues
    issues = []

    # Check shape
    if histogram.shape != (1152,):
        issues.append(f"Shape mismatch: {histogram.shape}")

    # Check for NaN/Inf
    if np.any(np.isnan(histogram)):
        issues.append("Contains NaN values")

    if np.any(np.isinf(histogram)):
        issues.append("Contains infinite values")

    # Check bounds
    if np.any(histogram < 0):
        issues.append("Contains negative values")

    if np.any(histogram > 1.0):
        issues.append("Contains values > 1.0")

    # Check normalization
    sum_val = histogram.sum()
    if not np.isclose(sum_val, 1.0, atol=1e-6):
        issues.append(f"Sum = {sum_val:.6f}, expected 1.0")

    # Check sparsity
    non_zero_count = np.count_nonzero(histogram)
    if non_zero_count == 0:
        issues.append("All bins are zero")
    elif non_zero_count < 10:
        issues.append(f"Very sparse: only {non_zero_count} non-zero bins")

    # Print issues
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No obvious issues found")

    return issues

# Example usage
issues = debug_validation_failure(histogram, "problematic_image.jpg")
```

### Performance Optimization Tips

1. **Use appropriate image sizes**: Resize large images before processing
2. **Batch processing**: Process multiple images together when possible
3. **Memory management**: Clear variables and use garbage collection
4. **Parallel processing**: Use multiprocessing for CPU-intensive tasks
5. **Caching**: Cache results for repeated operations
6. **Vectorization**: Use NumPy operations instead of loops

### Debugging Checklist

- [ ] Check image file path and format
- [ ] Verify image loading and conversion
- [ ] Check Lab color space conversion
- [ ] Verify histogram shape and dimensions
- [ ] Check normalization and bounds
- [ ] Monitor memory usage
- [ ] Profile performance bottlenecks
- [ ] Check for numerical precision issues
- [ ] Verify output file paths and permissions

---

## Conclusion

### Summary

This guide has provided comprehensive documentation for the Histogram Generation system in the Chromatica color search engine. The system offers:

- **Robust histogram generation** using tri-linear soft assignment
- **Comprehensive testing and validation** tools
- **Multiple output formats** and detailed analysis reports
- **Advanced features** for optimization and customization
- **Extensive usage examples** for various scenarios
- **Troubleshooting guidance** for common issues

### Key Benefits

1. **Accuracy**: Tri-linear soft assignment provides robust color representation
2. **Performance**: Vectorized operations and optimized algorithms
3. **Flexibility**: Multiple output formats and customization options
4. **Reliability**: Comprehensive validation and error handling
5. **Scalability**: Efficient batch processing and parallel execution
6. **Usability**: Clear command-line interface and programmatic API

### Getting Started

1. **Install dependencies**: Ensure all required packages are installed
2. **Test basic functionality**: Run the testing tool on sample images
3. **Explore features**: Experiment with different output formats and options
4. **Customize as needed**: Modify configurations for your specific use case
5. **Integrate into workflow**: Use the API in your applications

### Next Steps

- **Performance tuning**: Optimize for your specific hardware and requirements
- **Custom analysis**: Implement additional validation and analysis methods
- **Integration**: Connect with other components of your color search system
- **Monitoring**: Set up automated testing and validation pipelines
- **Documentation**: Extend this guide with your specific use cases

### Support and Resources

- **Code repository**: Check the source code for implementation details
- **Issue tracking**: Report bugs and request features through the project repository
- **Community**: Connect with other users and developers
- **Updates**: Stay informed about new features and improvements

The histogram generation system provides a solid foundation for color-based image search and analysis. With proper understanding and usage, it can significantly enhance your color processing capabilities and enable sophisticated color-based applications.

---

_This document was generated for the Chromatica Color Search Engine project. For the latest updates and additional information, please refer to the project repository and documentation._
