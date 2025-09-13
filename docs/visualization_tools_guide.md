# Chromatica Visualization Tools Guide

This guide provides comprehensive documentation for all visualization tools available in the Chromatica color search engine. These tools enhance the user experience by providing rich visual analysis, interactive exploration, and comprehensive reporting capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Color Palette Visualizer](#color-palette-visualizer)
3. [Search Results Visualizer](#search-results-visualizer)
4. [Interactive Color Explorer](#interactive-color-explorer)
5. [Installation and Dependencies](#installation-and-dependencies)
6. [Usage Examples](#usage-examples)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Overview

The Chromatica visualization tools provide three main categories of functionality:

- **Color Analysis**: Extract, analyze, and visualize color palettes from images
- **Search Results**: Comprehensive visualization and analysis of search results
- **Interactive Exploration**: Real-time color experimentation and harmony generation

All tools integrate seamlessly with the Chromatica API and can work with both local files and live API queries.

## Color Palette Visualizer

**File**: `tools/visualize_color_palettes.py`

### Purpose
Extract and visualize dominant colors from images, analyze color distributions, and compare color palettes across multiple images.

### Features
- **Dominant Color Extraction**: Uses K-means clustering to identify primary colors
- **Palette Visualization**: Create color swatches with percentage distributions
- **Histogram Analysis**: Visualize CIE Lab color space distributions
- **Batch Processing**: Analyze multiple images simultaneously
- **Export Capabilities**: Save visualizations and reports

### Usage Examples

#### Single Image Analysis
```bash
# Analyze a single image
python tools/visualize_color_palettes.py --image datasets/test-dataset-20/7349035.jpg

# Save visualizations to files
python tools/visualize_color_palettes.py --image datasets/test-dataset-20/7349035.jpg --save
```

#### Palette Comparison
```bash
# Compare multiple images
python tools/visualize_color_palettes.py --compare image1.jpg image2.jpg image3.jpg

# Save comparison visualizations
python tools/visualize_color_palettes.py --compare image1.jpg image2.jpg --save
```

#### Batch Analysis
```bash
# Analyze all images in a directory
python tools/visualize_color_palettes.py --batch datasets/test-dataset-20

# Save all reports
python tools/visualize_color_palettes.py --batch datasets/test-dataset-20 --save
```

### Output Types
1. **Color Swatches**: Visual representation of dominant colors with percentages
2. **Distribution Charts**: Pie charts showing color proportions
3. **Histogram Visualizations**: 3D projections of CIE Lab color distributions
4. **Comprehensive Reports**: Multi-panel analysis combining all visualizations

## Search Results Visualizer

**File**: `tools/visualize_search_results.py`

### Purpose
Comprehensive visualization and analysis of search results, including ranking analysis, performance metrics, and result galleries.

### Features
- **Ranking Analysis**: Visualize search result rankings and distances
- **Performance Metrics**: Analyze search timing and performance breakdown
- **Color Similarity Mapping**: Heatmaps showing color relationships
- **Result Galleries**: Interactive display of search results
- **API Integration**: Direct querying of the Chromatica API
- **Export Capabilities**: Save all visualizations and reports

### Usage Examples

#### API Query Visualization
```bash
# Query API and visualize results
python tools/visualize_search_results.py --api-query "FF0000" --k 10

# Save all visualizations
python tools/visualize_search_results.py --api-query "FF0000,00FF00" --weights "0.7,0.3" --k 15 --save
```

#### File-based Analysis
```bash
# Load results from JSON file
python tools/visualize_search_results.py --results search_results.json

# Save visualizations
python tools/visualize_search_results.py --results search_results.json --save
```

#### Result Comparison
```bash
# Compare multiple result files
python tools/visualize_search_results.py --compare query1.json query2.json query3.json
```

### Output Types
1. **Ranking Visualizations**: Bar charts and histograms of result rankings
2. **Performance Analysis**: Pie charts and bar charts of timing breakdown
3. **Color Similarity Heatmaps**: Matrix visualizations of color relationships
4. **Result Galleries**: Grid layouts of search results with metadata
5. **Comprehensive Reports**: Multi-panel dashboards combining all analyses

## Interactive Color Explorer

**File**: `tools/color_explorer.py`

### Purpose
Interactive tool for exploring color combinations, generating color harmonies, and experimenting with different color schemes in real-time.

### Features
- **Interactive Color Picker**: Add colors by hex codes with weights
- **Color Harmony Generation**: Automatic generation of complementary, analogous, triadic, and other color schemes
- **Real-time Preview**: Live visualization of color combinations
- **API Integration**: Test color combinations with live search
- **Palette Export**: Save color palettes for later use

### Usage Examples

#### Basic Usage
```bash
# Start the interactive explorer
python tools/color_explorer.py

# Connect to specific API instance
python tools/color_explorer.py --api-url http://localhost:8000
```

#### Color Harmony Types
- **Complementary**: Opposite colors on the color wheel
- **Analogous**: Adjacent colors on the color wheel
- **Triadic**: Three colors equally spaced on the color wheel
- **Split-Complementary**: Base color plus two colors adjacent to its complement
- **Tetradic**: Four colors forming a rectangle on the color wheel
- **Monochromatic**: Variations of a single color

### Interactive Features
1. **Color Input**: Add colors using hex codes (#FF0000)
2. **Weight Adjustment**: Set relative importance of each color
3. **Harmony Buttons**: One-click generation of color schemes
4. **Live Preview**: Real-time visualization of current palette
5. **API Search**: Test palettes against the Chromatica database
6. **Export Functionality**: Save palettes as JSON files

## Installation and Dependencies

### Required Packages
```bash
pip install matplotlib seaborn numpy opencv-python scikit-learn requests
```

### Optional Dependencies
- **Jupyter Notebooks**: For interactive development
- **IPython**: Enhanced interactive features

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Display**: Support for matplotlib backend (TkAgg, Qt5Agg, etc.)

## Usage Examples

### Complete Workflow Example

#### 1. Analyze Image Colors
```bash
# Extract dominant colors from an image
python tools/visualize_color_palettes.py --image sample.jpg --save
```

#### 2. Search with Colors
```bash
# Use extracted colors for search
python tools/visualize_search_results.py --api-query "FF0000,00FF00" --k 20 --save
```

#### 3. Interactive Exploration
```bash
# Explore color combinations
python tools/color_explorer.py
```

#### 4. Compare Results
```bash
# Compare multiple search queries
python tools/visualize_search_results.py --compare results1.json results2.json --save
```

### Advanced Usage Patterns

#### Batch Processing Pipeline
```bash
# Process multiple images
for img in datasets/*.jpg; do
    python tools/visualize_color_palettes.py --image "$img" --save
done
```

#### API Performance Monitoring
```bash
# Monitor search performance over time
python tools/visualize_search_results.py --api-query "FF0000" --k 100 --save
```

#### Color Scheme Development
```bash
# Start with base color
python tools/color_explorer.py
# Add base color, apply harmony rules, export palette
```

## Advanced Features

### Custom Color Spaces
All tools support the CIE Lab color space as specified in the Chromatica architecture:
- **L* (Lightness)**: 8 bins over range [0, 100]
- **a* (Green-Red)**: 12 bins over range [-86, 98]
- **b* (Blue-Yellow)**: 12 bins over range [-108, 95]

### Histogram Analysis
Advanced histogram visualizations include:
- **3D Projections**: L* vs a*, L* vs b*, a* vs b* views
- **Distribution Analysis**: Overall histogram patterns
- **Bin Analysis**: Individual bin contributions

### Performance Optimization
- **Lazy Loading**: Histograms loaded only when needed
- **Caching**: Cost matrices and computed values cached
- **Batch Processing**: Efficient handling of multiple images

### Export Formats
- **PNG**: High-resolution visualizations (300 DPI)
- **JSON**: Structured data for further analysis
- **CSV**: Tabular data for spreadsheet applications

## Troubleshooting

### Common Issues

#### Matplotlib Backend Errors
```bash
# Set backend explicitly
export MPLBACKEND=TkAgg
python tools/color_explorer.py
```

#### Memory Issues with Large Datasets
```bash
# Limit batch size
python tools/visualize_color_palettes.py --batch large_dataset --num-colors 5
```

#### API Connection Issues
```bash
# Check API status
curl http://localhost:8000/api/info

# Use different API URL
python tools/visualize_search_results.py --api-url http://other-server:8000
```

#### Display Issues
```bash
# Use non-interactive backend
export MPLBACKEND=Agg
python tools/visualize_color_palettes.py --image sample.jpg --save
```

### Performance Tips

1. **Limit Color Count**: Use `--num-colors 5` for faster processing
2. **Batch Processing**: Process multiple images together
3. **Save Results**: Use `--save` flag to avoid regenerating visualizations
4. **API Caching**: Results are cached for repeated queries

### Debug Mode
```bash
# Enable verbose logging
export CHROMATICA_LOG_LEVEL=DEBUG
python tools/visualize_color_palettes.py --image sample.jpg
```

## Integration with Chromatica

### API Endpoints
All visualization tools integrate with Chromatica API endpoints:
- **Search**: `/search` for color-based image retrieval
- **Info**: `/api/info` for system status
- **Images**: `/image/{image_id}` for image serving

### Data Flow
1. **Input**: Images, color queries, or result files
2. **Processing**: Color extraction, histogram generation, analysis
3. **Visualization**: Charts, graphs, and interactive displays
4. **Output**: Saved files, API queries, or real-time display

### Extensibility
The visualization tools are designed for easy extension:
- **New Chart Types**: Add custom matplotlib visualizations
- **Additional Metrics**: Implement new analysis algorithms
- **Export Formats**: Support for additional file types
- **API Integration**: Connect to external color analysis services

## Conclusion

The Chromatica visualization tools provide a comprehensive suite for color analysis, search result visualization, and interactive color exploration. These tools enhance the user experience by making complex color relationships and search results accessible and understandable.

For additional support or feature requests, refer to the main Chromatica documentation or create issues in the project repository.
