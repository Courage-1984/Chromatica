# Chromatica Jupyter Notebooks

This directory contains Jupyter notebooks for experimentation, analysis, visualization, and development of the Chromatica color search engine.

## 📁 Notebooks Overview

The notebooks directory provides an interactive environment for:

- **Algorithm Development**: Prototyping and testing new approaches
- **Data Analysis**: Exploring datasets and understanding data characteristics
- **Visualization**: Creating charts, graphs, and interactive visualizations
- **Performance Analysis**: Benchmarking and optimization studies
- **Research & Development**: Experimenting with new ideas and techniques

## 🚀 Available Notebooks

### Core Algorithm Notebooks

#### Histogram Generation Analysis

- **`01_histogram_generation_analysis.ipynb`**: Deep dive into histogram generation algorithms
- **`02_color_space_exploration.ipynb`**: Analysis of different color spaces and their properties
- **`03_binning_strategies.ipynb`**: Comparison of various binning approaches

**Features:**

- Interactive histogram visualization
- Color space conversion analysis
- Performance benchmarking
- Quality metrics assessment

#### Search Algorithm Development

- **`04_search_algorithm_prototyping.ipynb`**: Development and testing of search algorithms
- **`05_distance_metrics_comparison.ipynb`**: Analysis of different distance metrics
- **`06_reranking_strategies.ipynb`**: Development of reranking approaches

**Features:**

- Algorithm performance comparison
- Parameter tuning and optimization
- Accuracy vs. speed trade-offs
- Interactive parameter exploration

### Data Analysis Notebooks

#### Dataset Exploration

- **`07_dataset_characteristics.ipynb`**: Comprehensive dataset analysis
- **`08_image_quality_assessment.ipynb`**: Quality metrics and outlier detection
- **`09_color_distribution_analysis.ipynb`**: Statistical analysis of color distributions

**Features:**

- Dataset statistics and summaries
- Quality metric distributions
- Outlier detection and analysis
- Interactive data exploration

#### Performance Analysis

- **`10_performance_benchmarking.ipynb`**: Comprehensive performance analysis
- **`11_scalability_studies.ipynb`**: Performance across different dataset sizes
- **`12_memory_usage_analysis.ipynb`**: Memory profiling and optimization

**Features:**

- Performance trend analysis
- Scalability curves
- Memory usage profiling
- Optimization recommendations

### Visualization Notebooks

#### Interactive Visualizations

- **`13_histogram_visualization.ipynb`**: 3D and 2D histogram plots
- **`14_color_space_visualization.ipynb`**: Interactive color space exploration
- **`15_search_results_visualization.ipynb`**: Search result presentation

**Features:**

- Interactive 3D plots
- Dynamic parameter adjustment
- Real-time visualization updates
- Export capabilities

#### Analysis Dashboards

- **`16_performance_dashboard.ipynb`**: Comprehensive performance monitoring
- **`17_quality_metrics_dashboard.ipynb`**: Quality assessment overview
- **`18_system_health_dashboard.ipynb`**: System status and monitoring

**Features:**

- Real-time data updates
- Interactive filtering
- Export and reporting
- Alert generation

## 🔧 Setup and Installation

### Prerequisites

1. **Jupyter Environment**:

   ```bash
   # Install Jupyter
   pip install jupyter notebook jupyterlab

   # Install additional dependencies
   pip install ipywidgets plotly bokeh
   ```

2. **Activate Virtual Environment**:

   ```bash
   venv311\Scripts\activate  # Windows
   # or
   source venv311/bin/activate  # Linux/Mac
   ```

3. **Install Notebook Dependencies**:
   ```bash
   pip install -r notebooks/requirements.txt
   ```

### Launching Notebooks

```bash
# Start Jupyter Notebook
jupyter notebook

# Start JupyterLab (recommended)
jupyter lab

# Start with specific port
jupyter lab --port 8888

# Start with specific directory
jupyter lab --notebook-dir notebooks/
```

## 📊 Notebook Features

### Interactive Widgets

Notebooks include interactive widgets for parameter tuning:

```python
import ipywidgets as widgets
from IPython.display import display

# Create interactive sliders
l_bins = widgets.IntSlider(value=8, min=4, max=16, description='L* Bins:')
a_bins = widgets.IntSlider(value=12, min=6, max=24, description='a* Bins:')
b_bins = widgets.IntSlider(value=12, min=6, max=24, description='b* Bins:')

# Display widgets
display(l_bins, a_bins, b_bins)

# Use values in analysis
def update_analysis(l, a, b):
    total_bins = l * a * b
    print(f"Total bins: {total_bins}")

# Connect widgets to function
widgets.interactive(update_analysis, l=l_bins, a=a_bins, b=b_bins)
```

### Advanced Visualizations

#### Plotly Interactive Charts

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Create interactive 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=histogram_data[:, 0],
    y=histogram_data[:, 1],
    z=histogram_data[:, 2],
    mode='markers',
    marker=dict(size=2, color=histogram_values, colorscale='Viridis')
)])

fig.update_layout(title='3D Histogram Visualization')
fig.show()
```

#### Bokeh Interactive Plots

```python
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import column, row

output_notebook()

# Create interactive plot
p = figure(width=800, height=400, title='Histogram Quality Metrics')
p.scatter(entropy_values, sparsity_values, size=8, alpha=0.6)
p.xaxis.axis_label = 'Entropy'
p.yaxis.axis_label = 'Sparsity'

show(p)
```

### Data Processing Integration

Notebooks integrate with the main Chromatica modules:

```python
# Import Chromatica modules
import sys
sys.path.append('../src')

from chromatica.core.histogram import build_histogram
from chromatica.indexing.pipeline import process_image
from chromatica.utils.config import TOTAL_BINS, L_BINS, A_BINS, B_BINS

# Use in analysis
image_path = '../datasets/test-dataset-20/a1.png'
histogram = process_image(image_path)

print(f"Histogram shape: {histogram.shape}")
print(f"Total bins: {TOTAL_BINS}")
print(f"L* bins: {L_BINS}, a* bins: {A_BINS}, b* bins: {B_BINS}")
```

## 📈 Analysis Workflows

### Performance Analysis Workflow

1. **Data Collection**: Gather performance metrics from different dataset sizes
2. **Statistical Analysis**: Calculate means, standard deviations, and confidence intervals
3. **Trend Analysis**: Identify performance patterns and bottlenecks
4. **Optimization**: Test different parameters and configurations
5. **Reporting**: Generate comprehensive performance reports

### Quality Assessment Workflow

1. **Metric Calculation**: Compute quality metrics for all histograms
2. **Distribution Analysis**: Analyze metric distributions and identify outliers
3. **Correlation Analysis**: Study relationships between different metrics
4. **Threshold Setting**: Establish quality thresholds for production use
5. **Monitoring Setup**: Create automated quality monitoring systems

### Algorithm Development Workflow

1. **Prototyping**: Implement and test new algorithms in notebooks
2. **Benchmarking**: Compare performance against existing implementations
3. **Parameter Tuning**: Optimize algorithm parameters
4. **Validation**: Test with different datasets and edge cases
5. **Integration**: Move successful algorithms to production code

## 🔍 Notebook Organization

### Directory Structure

```
notebooks/
├── README.md                           # This documentation
├── requirements.txt                    # Notebook-specific dependencies
├── core/                              # Core algorithm notebooks
│   ├── 01_histogram_generation_analysis.ipynb
│   ├── 02_color_space_exploration.ipynb
│   └── 03_binning_strategies.ipynb
├── analysis/                          # Data analysis notebooks
│   ├── 07_dataset_characteristics.ipynb
│   ├── 08_image_quality_assessment.ipynb
│   └── 09_color_distribution_analysis.ipynb
├── performance/                       # Performance analysis notebooks
│   ├── 10_performance_benchmarking.ipynb
│   ├── 11_scalability_studies.ipynb
│   └── 12_memory_usage_analysis.ipynb
├── visualization/                     # Visualization notebooks
│   ├── 13_histogram_visualization.ipynb
│   ├── 14_color_space_visualization.ipynb
│   └── 15_search_results_visualization.ipynb
├── dashboards/                        # Dashboard notebooks
│   ├── 16_performance_dashboard.ipynb
│   ├── 17_quality_metrics_dashboard.ipynb
│   └── 18_system_health_dashboard.ipynb
└── templates/                         # Notebook templates
    ├── algorithm_prototype_template.ipynb
    ├── analysis_template.ipynb
    └── visualization_template.ipynb
```

### Naming Convention

Notebooks follow a consistent naming convention:

- **Prefix**: Numbered prefix for logical ordering
- **Category**: Descriptive category name
- **Purpose**: Clear description of notebook purpose
- **Extension**: `.ipynb` for Jupyter notebooks

Example: `01_histogram_generation_analysis.ipynb`

## 🧪 Testing and Validation

### Notebook Testing

```bash
# Test notebook execution
pytest --nbval notebooks/

# Test specific notebook
pytest --nbval notebooks/core/01_histogram_generation_analysis.ipynb

# Test with coverage
pytest --nbval --cov=notebooks notebooks/
```

### Output Validation

Notebooks include validation checks:

```python
# Validate histogram generation
def validate_histogram(histogram):
    """Validate histogram properties."""
    assert histogram.shape == (1152,), f"Expected shape (1152,), got {histogram.shape}"
    assert np.isclose(histogram.sum(), 1.0), f"Sum should be 1.0, got {histogram.sum()}"
    assert np.all(histogram >= 0), "All values should be non-negative"
    return True

# Use in analysis
histogram = build_histogram(lab_pixels)
validation_result = validate_histogram(histogram)
print(f"Histogram validation: {'PASSED' if validation_result else 'FAILED'}")
```

## 📊 Output and Export

### Export Options

Notebooks support multiple export formats:

```python
# Export to HTML
jupyter nbconvert --to html notebooks/core/01_histogram_generation_analysis.ipynb

# Export to PDF
jupyter nbconvert --to pdf notebooks/core/01_histogram_generation_analysis.ipynb

# Export to Python script
jupyter nbconvert --to python notebooks/core/01_histogram_generation_analysis.ipynb

# Export to slides
jupyter nbconvert --to slides notebooks/core/01_histogram_generation_analysis.ipynb
```

### Report Generation

```python
# Generate comprehensive reports
def generate_report(notebook_path, output_format='html'):
    """Generate report from notebook."""
    import subprocess

    cmd = [
        'jupyter', 'nbconvert',
        '--to', output_format,
        '--output-dir', 'reports/',
        notebook_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

# Generate reports for all notebooks
notebooks = [
    'notebooks/core/01_histogram_generation_analysis.ipynb',
    'notebooks/analysis/07_dataset_characteristics.ipynb',
    'notebooks/performance/10_performance_benchmarking.ipynb'
]

for notebook in notebooks:
    success = generate_report(notebook, 'html')
    print(f"Report generation for {notebook}: {'SUCCESS' if success else 'FAILED'}")
```

## 🚨 Common Issues

### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/Chromatica

# Activate virtual environment
venv311\Scripts\activate

# Install notebook dependencies
pip install -r notebooks/requirements.txt
```

### Kernel Issues

```bash
# List available kernels
jupyter kernelspec list

# Install kernel for virtual environment
python -m ipykernel install --user --name=chromatica-venv --display-name="Chromatica (venv311)"
```

### Memory Issues

```bash
# Start Jupyter with memory limits
jupyter lab --NotebookApp.max_buffer_size=1000000000

# Use memory-efficient processing
# Reduce batch sizes in analysis
# Use streaming for large datasets
```

## 📈 Performance Optimization

### Notebook Optimization

```python
# Use efficient data structures
import numpy as np
import pandas as pd

# Process data in chunks
def process_large_dataset(dataset_path, chunk_size=1000):
    """Process large datasets in chunks."""
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        # Process chunk
        yield process_chunk(chunk)

# Use parallel processing
from concurrent.futures import ProcessPoolExecutor

def parallel_processing(data, func, max_workers=4):
    """Process data in parallel."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, data))
    return results
```

### Memory Management

```python
# Clear variables to free memory
import gc

def memory_efficient_analysis():
    """Memory-efficient analysis workflow."""
    # Process data
    results = process_data()

    # Clear intermediate variables
    del intermediate_data
    gc.collect()

    return results
```

## 🔮 Future Enhancements

### Planned Notebooks

- **Machine Learning Integration**: Notebooks for ML-based color analysis
- **Real-time Analysis**: Live data streaming and analysis
- **Collaborative Analysis**: Multi-user notebook collaboration
- **Automated Reporting**: Scheduled report generation and distribution
- **Integration Testing**: End-to-end system testing notebooks

### Notebook Improvements

- **Template System**: Standardized notebook templates for common tasks
- **Automated Execution**: Scheduled notebook execution and reporting
- **Version Control**: Git integration for notebook versioning
- **Dependency Management**: Automated dependency resolution
- **Cloud Integration**: Cloud-based notebook execution and storage

---

**Last Updated**: December 2024  
**Notebook Count**: 18+ specialized notebooks  
**Status**: Comprehensive notebook infrastructure implemented
