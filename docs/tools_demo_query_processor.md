# Demo Query Processor Tool

## Overview

The `demo_query_processor.py` tool is a demonstration script that showcases the key functionality of the Chromatica query processor module. It provides practical examples of how to use the query processor for color-based image search applications, including color conversion, histogram generation, and validation.

## Purpose

This tool is designed to:
- Demonstrate hex to Lab color conversion functionality
- Show query histogram generation with different color combinations
- Illustrate histogram validation and quality checks
- Provide performance benchmarking examples
- Serve as a learning resource for developers

## Features

- **Color Conversion Demo**: Hex to Lab color space conversion
- **Histogram Generation**: Multi-color query histogram creation
- **Validation Examples**: Histogram quality and property validation
- **Performance Testing**: Timing and efficiency benchmarking
- **Practical Examples**: Real-world color combination scenarios

## Usage

### Basic Usage

```bash
# Run the complete demo
python tools/demo_query_processor.py

# Run from project root directory
cd /path/to/Chromatica
python tools/demo_query_processor.py
```

### What the Demo Shows

The demo script runs through several key scenarios:

1. **Hex to Lab Conversion**: Converting hex colors to Lab color space
2. **Query Histogram Generation**: Creating histograms from color combinations
3. **Histogram Validation**: Checking histogram properties and quality
4. **Performance Benchmarking**: Timing conversion and generation operations

## Core Demo Functions

### 1. Hex to Lab Color Conversion

```python
def demo_hex_to_lab():
    """Demonstrate hex to Lab color conversion."""
    
    # Test colors representing different color families
    test_colors = [
        ("#FF0000", "Pure Red"),
        ("#00FF00", "Pure Green"),
        ("#0000FF", "Pure Blue"),
        ("#FFFF00", "Pure Yellow"),
        ("#FF00FF", "Pure Magenta"),
        ("#00FFFF", "Pure Cyan"),
        ("#000000", "Pure Black"),
        ("#FFFFFF", "Pure White"),
        ("#808080", "Medium Gray"),
        ("#FFA500", "Orange"),
        ("#800080", "Purple"),
        ("#008000", "Dark Green"),
    ]
    
    print(f"{'Hex Color':<10} {'Color Name':<15} {'L*':<8} {'a*':<8} {'b*':<8}")
    print("-" * 60)
    
    for hex_color, color_name in test_colors:
        try:
            l_val, a_val, b_val = hex_to_lab(hex_color)
            print(f"{hex_color:<10} {color_name:<15} {l_val:<8.2f} {a_val:<8.2f} {b_val:<8.2f}")
        except Exception as e:
            print(f"{hex_color:<10} {color_name:<15} ERROR: {str(e)}")
```

**What it demonstrates:**
- Color space conversion from hex to Lab
- Handling of various color types (primary, secondary, neutral)
- Error handling for invalid color values
- Tabular output formatting for easy reading

**Expected Output:**
```
============================================================
HEX TO LAB COLOR CONVERSION DEMO
============================================================
Hex Color Color Name      L*      a*      b*      
------------------------------------------------------------
#FF0000   Pure Red       53.24   80.09   67.20   
#00FF00   Pure Green     87.73   -86.18  83.18   
#0000FF   Pure Blue      32.30   79.19   -107.86 
#FFFF00   Pure Yellow    97.14   -21.55  94.48   
#FF00FF   Pure Magenta   60.32   98.23   -60.83  
#00FFFF   Pure Cyan      91.11   -48.09  -14.13  
#000000   Pure Black     0.00    0.00    0.00    
#FFFFFF   Pure White     100.00  0.00    0.00    
#808080   Medium Gray    53.59   0.00    0.00    
#FFA500   Orange         74.93   23.93   78.95   
#800080   Purple         25.42   47.36   -64.88  
#008000   Dark Green     46.23   -51.70  49.70   
```

### 2. Query Histogram Generation

```python
def demo_query_histograms():
    """Demonstrate query histogram generation."""
    
    # Test different color combinations and weights
    test_queries = [
        {
            "name": "Warm Colors",
            "colors": ["#FF0000", "#FFA500", "#FFFF00"],
            "weights": [0.5, 0.3, 0.2],
        },
        {
            "name": "Cool Colors",
            "colors": ["#0000FF", "#00FFFF", "#800080"],
            "weights": [0.4, 0.4, 0.2],
        },
        {
            "name": "Earth Tones",
            "colors": ["#8B4513", "#A0522D", "#CD853F", "#D2B48C"],
            "weights": [0.3, 0.3, 0.2, 0.2],
        },
        {
            "name": "High Contrast",
            "colors": ["#000000", "#FFFFFF", "#FF0000"],
            "weights": [0.4, 0.4, 0.2],
        },
        {
            "name": "Pastel Palette",
            "colors": ["#FFB6C1", "#98FB98", "#87CEEB", "#DDA0DD"],
            "weights": [0.25, 0.25, 0.25, 0.25],
        },
    ]
    
    for query in test_queries:
        print(f"\n{query['name']}:")
        print(f"  Colors: {', '.join(query['colors'])}")
        print(f"  Weights: {query['weights']}")
        
        try:
            # Generate histogram
            histogram = create_query_histogram(query['colors'], query['weights'])
            
            # Validate histogram
            validation = validate_query_histogram(histogram)
            
            print(f"  Histogram shape: {histogram.shape}")
            print(f"  Sum: {histogram.sum():.6f}")
            print(f"  Valid: {validation['valid']}")
            
            if not validation['valid']:
                print(f"  Errors: {validation['errors']}")
                
        except Exception as e:
            print(f"  Error: {e}")
```

**What it demonstrates:**
- Multi-color histogram generation
- Weight-based color combination
- Histogram validation and quality checks
- Error handling for invalid inputs

### 3. Histogram Validation

```python
def demo_histogram_validation():
    """Demonstrate histogram validation functionality."""
    
    print("=" * 60)
    print("HISTOGRAM VALIDATION DEMO")
    print("=" * 60)
    
    # Test various histogram scenarios
    test_cases = [
        {
            "name": "Valid Histogram",
            "colors": ["#FF0000", "#00FF00"],
            "weights": [0.6, 0.4],
        },
        {
            "name": "Single Color",
            "colors": ["#0000FF"],
            "weights": [1.0],
        },
        {
            "name": "Equal Weights",
            "colors": ["#FF0000", "#00FF00", "#0000FF"],
            "weights": [0.33, 0.33, 0.34],
        },
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        
        try:
            # Generate histogram
            histogram = create_query_histogram(test_case['colors'], test_case['weights'])
            
            # Validate
            validation = validate_query_histogram(histogram)
            
            print(f"  Shape: {histogram.shape}")
            print(f"  Sum: {histogram.sum():.6f}")
            print(f"  Min: {histogram.min():.6f}")
            print(f"  Max: {histogram.max():.6f}")
            print(f"  Valid: {validation['valid']}")
            
            if validation['valid']:
                print(f"  ✅ Validation passed")
            else:
                print(f"  ❌ Validation failed: {validation['errors']}")
                
        except Exception as e:
            print(f"  Error: {e}")
```

**What it demonstrates:**
- Histogram property validation
- Quality metrics calculation
- Error identification and reporting
- Success/failure status display

### 4. Performance Benchmarking

```python
def demo_performance():
    """Demonstrate performance characteristics."""
    
    print("=" * 60)
    print("PERFORMANCE BENCHMARKING DEMO")
    print("=" * 60)
    
    # Test different color combinations
    test_scenarios = [
        ("Single Color", ["#FF0000"], [1.0]),
        ("Two Colors", ["#FF0000", "#00FF00"], [0.5, 0.5]),
        ("Three Colors", ["#FF0000", "#00FF00", "#0000FF"], [0.33, 0.33, 0.34]),
        ("Five Colors", ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"], 
         [0.2, 0.2, 0.2, 0.2, 0.2]),
    ]
    
    print(f"{'Scenario':<15} {'Colors':<8} {'Time (ms)':<12} {'Valid':<8}")
    print("-" * 60)
    
    for scenario_name, colors, weights in test_scenarios:
        # Benchmark conversion time
        start_time = time.time()
        
        try:
            histogram = create_query_histogram(colors, weights)
            validation = validate_query_histogram(histogram)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            
            print(f"{scenario_name:<15} {len(colors):<8} {processing_time:<12.3f} {str(validation['valid']):<8}")
            
        except Exception as e:
            print(f"{scenario_name:<15} {len(colors):<8} ERROR: {str(e)}")
    
    print()
```

**What it demonstrates:**
- Performance timing for different scenarios
- Scalability with color count
- Processing efficiency metrics
- Error handling during benchmarking

## Advanced Usage Examples

### Custom Color Palette Creation

```python
def create_custom_palette():
    """Create a custom color palette histogram."""
    
    # Define a custom color scheme
    custom_colors = {
        "primary": ["#FF0000", "#00FF00", "#0000FF"],
        "secondary": ["#FFFF00", "#FF00FF", "#00FFFF"],
        "accent": ["#FFA500", "#800080", "#008000"]
    }
    
    # Create weighted combinations
    primary_histogram = create_query_histogram(
        custom_colors["primary"], 
        [0.4, 0.35, 0.25]
    )
    
    secondary_histogram = create_query_histogram(
        custom_colors["secondary"], 
        [0.33, 0.33, 0.34]
    )
    
    accent_histogram = create_query_histogram(
        custom_colors["accent"], 
        [0.4, 0.3, 0.3]
    )
    
    # Combine histograms (simple average)
    combined_histogram = (primary_histogram + secondary_histogram + accent_histogram) / 3
    
    # Validate combined result
    validation = validate_query_histogram(combined_histogram)
    
    return combined_histogram, validation
```

### Batch Color Processing

```python
def process_color_batch():
    """Process multiple color combinations in batch."""
    
    color_batches = [
        (["#FF0000", "#00FF00"], [0.7, 0.3]),
        (["#0000FF", "#FFFF00"], [0.6, 0.4]),
        (["#FF00FF", "#00FFFF"], [0.5, 0.5]),
        (["#FFA500", "#800080"], [0.8, 0.2]),
    ]
    
    results = []
    
    for colors, weights in color_batches:
        try:
            histogram = create_query_histogram(colors, weights)
            validation = validate_query_histogram(histogram)
            
            results.append({
                "colors": colors,
                "weights": weights,
                "histogram": histogram,
                "valid": validation["valid"],
                "errors": validation.get("errors", [])
            })
            
        except Exception as e:
            results.append({
                "colors": colors,
                "weights": weights,
                "error": str(e)
            })
    
    return results
```

## Integration Examples

### With Search System

```python
def integrate_with_search():
    """Integrate query processor with search system."""
    
    from chromatica.search import find_similar
    from chromatica.indexing.store import AnnIndex, MetadataStore
    
    # Create query histogram
    query_colors = ["#FF0000", "#00FF00", "#0000FF"]
    query_weights = [0.4, 0.35, 0.25]
    
    query_histogram = create_query_histogram(query_colors, query_weights)
    
    # Validate query
    validation = validate_query_histogram(query_histogram)
    if not validation["valid"]:
        raise ValueError(f"Invalid query histogram: {validation['errors']}")
    
    # Perform search
    index = AnnIndex()  # Your search index
    store = MetadataStore("path/to/store.db")  # Your metadata store
    
    results = find_similar(
        query_histogram,
        index,
        store,
        k=10,
        rerank_k=5
    )
    
    return results
```

### With Histogram Testing Tool

```python
def integrate_with_testing():
    """Integrate with histogram testing tool."""
    
    from tools.test_histogram_generation import HistogramTester
    
    # Create query histogram
    query_histogram = create_query_histogram(
        ["#FF0000", "#00FF00", "#0000FF"],
        [0.4, 0.35, 0.25]
    )
    
    # Test with histogram testing tool
    tester = HistogramTester(output_format="json", visualize=True)
    
    # Validate using testing tool
    validation_result = tester.validate_histogram(query_histogram)
    
    if validation_result["valid"]:
        print("✅ Query histogram is valid")
        print(f"   Entropy: {validation_result['metrics']['entropy']:.4f}")
        print(f"   Sparsity: {validation_result['metrics']['sparsity']:.4f}")
    else:
        print("❌ Query histogram validation failed")
        print(f"   Errors: {validation_result['errors']}")
```

## Error Handling

### Common Error Scenarios

```python
def handle_common_errors():
    """Handle common error scenarios."""
    
    # Invalid hex color
    try:
        l_val, a_val, b_val = hex_to_lab("#INVALID")
    except ValueError as e:
        print(f"Invalid hex color: {e}")
        # Handle gracefully - use default color or skip
    
    # Mismatched colors and weights
    try:
        histogram = create_query_histogram(
            ["#FF0000", "#00FF00"], 
            [0.5]  # Missing weight
        )
    except ValueError as e:
        print(f"Weight mismatch: {e}")
        # Handle gracefully - use equal weights or skip
    
    # Invalid histogram
    try:
        validation = validate_query_histogram(None)
    except TypeError as e:
        print(f"Invalid histogram: {e}")
        # Handle gracefully - return error status
```

### Robust Error Handling

```python
def robust_color_processing(colors, weights):
    """Robust color processing with error handling."""
    
    try:
        # Validate inputs
        if not colors or not weights:
            raise ValueError("Colors and weights must be provided")
        
        if len(colors) != len(weights):
            raise ValueError("Colors and weights must have same length")
        
        if not all(0 <= w <= 1 for w in weights):
            raise ValueError("Weights must be between 0 and 1")
        
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError("Weights must sum to 1.0")
        
        # Process colors
        histogram = create_query_histogram(colors, weights)
        validation = validate_query_histogram(histogram)
        
        return {
            "success": True,
            "histogram": histogram,
            "validation": validation
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "histogram": None,
            "validation": None
        }
```

## Best Practices

### Color Selection

1. **Use Semantic Colors**: Choose colors that represent the concept you're searching for
2. **Balance Weights**: Distribute weights based on importance
3. **Consider Color Theory**: Use complementary or analogous colors for better results
4. **Test Combinations**: Validate color combinations before use

### Performance Optimization

1. **Batch Processing**: Process multiple color combinations together
2. **Cache Results**: Store frequently used histograms
3. **Limit Color Count**: Use reasonable number of colors (3-7 typically)
4. **Validate Early**: Check histogram validity before expensive operations

### Error Handling

1. **Input Validation**: Validate colors and weights before processing
2. **Graceful Degradation**: Handle errors without crashing
3. **User Feedback**: Provide clear error messages
4. **Fallback Options**: Offer alternatives when processing fails

## Troubleshooting

### Common Issues

1. **Invalid Hex Colors**: Ensure hex colors are in #RRGGBB format
2. **Weight Mismatches**: Verify colors and weights arrays have same length
3. **Normalization Issues**: Check that weights sum to 1.0
4. **Memory Issues**: Limit color combinations for large datasets

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual functions
try:
    l_val, a_val, b_val = hex_to_lab("#FF0000")
    print(f"Red: L*={l_val}, a*={a_val}, b*={b_val}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

## Dependencies

### Required Packages

- `numpy`: Numerical operations
- `opencv-python`: Image processing
- `scikit-image`: Color space conversion

### Installation

```bash
# Install dependencies
pip install numpy opencv-python scikit-image

# Or use project requirements
pip install -r requirements.txt
```

The demo query processor tool provides comprehensive examples of how to use the Chromatica query processing system effectively and serves as both a learning resource and a reference implementation for developers working with color-based image search applications.
