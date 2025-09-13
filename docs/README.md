# 📚 Chromatica Documentation Hub

Welcome to the comprehensive documentation for the Chromatica color search engine project. This hub provides access to all project documentation, guides, and resources.

## 🎯 Quick Start

**New to Chromatica?** Start here:

1. **[Complete Usage Guide](complete_usage_guide.md)** - Step-by-step setup and usage instructions
2. **[Project Architecture](project_architecture.md)** - System design and component overview
3. **[API Reference](api_reference.md)** - Complete API documentation and examples

## 📋 Documentation Index

### **🚀 Getting Started**

- **[Complete Usage Guide](complete_usage_guide.md)** - Comprehensive setup and usage guide
- **[Project Architecture](project_architecture.md)** - System architecture and design principles
- **[README](../README.md)** - Project overview and quick start

### **🔧 Core Components**

- **[Histogram Generation Guide](histogram_generation_guide.md)** - Color histogram generation details
- **[Image Processing Pipeline](image_processing_pipeline.md)** - Image processing workflow
- **[Two-Stage Search Architecture](two_stage_search_architecture.md)** - Search system design
- **[Query Processor](query_processor.md)** - Query handling and processing
- **[Sinkhorn Reranking Logic](sinkhorn_reranking_logic.md)** - Advanced reranking algorithms

### **🌐 API & Interface**

- **[API Reference](api_reference.md)** - Complete API documentation
- **[FastAPI Endpoint](fastapi_endpoint.md)** - Web API implementation details
- **[Visualization Features](visualization_features.md)** - Query and result visualization
- **[Web Interface Usage](visualization_features.md#web-interface-usage)** - Interactive color picker guide
- **[Catppuccin Mocha Theme](catppuccin_mocha_theme.md)** - Web interface theming documentation
- **[Theme Quick Reference](catppuccin_mocha_quick_reference.md)** - Developer theme guide
- **[Font Setup Guide](font_setup_guide.md)** - Custom font configuration

### **🏗️ Infrastructure**

- **[FAISS & DuckDB Guide](faiss_duckdb_guide.md)** - Search index and database setup
- **[FAISS & DuckDB Integration](faiss_duckdb_integration.md)** - Integration patterns
- **[FAISS & DuckDB Wrappers](faiss_duckdb_wrappers.md)** - Component wrappers
- **[FAISS & DuckDB Usage Examples](faiss_duckdb_usage_examples.md)** - Practical examples
- **[Build Index Script](scripts_build_index.md)** - Comprehensive index building documentation
- **[Offline Indexing Script](offline_indexing_script.md)** - Index building process

### **🧪 Testing & Development**

- **[Tools Overview](../tools/README.md)** - Testing and demonstration tools
- **[Histogram Generation Testing](tools_histogram_generation_testing.md)** - Histogram validation
- **[FAISS & DuckDB Testing](tools_test_faiss_duckdb.md)** - Search component testing
- **[Search System Testing](tools_test_search_system.md)** - End-to-end testing
- **[API Testing](tools_test_api.md)** - Web API validation
- **[Visualization Demo](tools_demo.md)** - Feature demonstration

### **🔍 Troubleshooting & Support**

- **[Comprehensive Troubleshooting](troubleshooting_comprehensive.md)** - Complete troubleshooting guide
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Sanity Checks](tools_sanity_checks.md)** - System validation tools

### **📊 Progress & Planning**

- **[Progress Report](progress.md)** - Current development status
- **[Critical Instructions](../.cursor/critical_instructions.md)** - Core project specifications

### **⚙️ Development Configuration**

- **[Cursor Rules Migration Guide](cursor_rules_migration_guide.md)** - Migration to new Cursor Rules system
- **[Cursor Rules Setup Summary](cursor_rules_setup_summary.md)** - Quick reference for new setup
- **[Cursor Rules Guide](cursor_rules_guide.md)** - Cursor Rules implementation guide
- **[Cursor Rules Summary](cursor_rules_summary.md)** - Overview of Cursor Rules system

## 🎨 What is Chromatica?

Chromatica is a **sophisticated color-based image search engine** that combines:

- **🔬 Advanced Color Science**: CIE Lab color space with 8x12x12 histogram binning
- **⚡ High Performance**: FAISS HNSW index for sub-second search times
- **🎯 Accuracy**: Sinkhorn-EMD reranking for perceptually accurate results
- **🎨 Rich Visualizations**: Query representations and result collages
- **🌐 Modern Interface**: Interactive web interface with real-time feedback

### **Key Features**

- **Multi-color queries** with customizable weights
- **Real-time search** with comprehensive metadata
- **Visual query representation** showing color distributions
- **Results collage** with distance annotations
- **RESTful API** with full documentation
- **Interactive web interface** for easy exploration

## 🚀 Quick Setup

For experienced users, here's the minimal setup:

```bash
# 1. Setup
cd Chromatica
venv311\Scripts\activate
pip install -r requirements.txt

# 2. Build index
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index

# 3. Start server
python -m src.chromatica.api.main

# 4. Open browser
# Navigate to: http://localhost:8000/
```

**For detailed setup instructions, see the [Complete Usage Guide](complete_usage_guide.md).**

## 🔧 System Requirements

- **Python**: 3.10 or higher
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Storage**: 2GB+ free space
- **OS**: Windows 10/11, macOS, or Linux

## 📚 Learning Path

### **Beginner Level**

1. Read the [Complete Usage Guide](complete_usage_guide.md)
2. Set up the project following the step-by-step instructions
3. Try the interactive web interface
4. Experiment with different color combinations

### **Intermediate Level**

1. Study the [Project Architecture](project_architecture.md)
2. Understand the [Two-Stage Search Logic](two_stage_search_logic.md)
3. Explore the [API Reference](api_reference.md)
4. Test with different datasets and configurations

### **Advanced Level**

1. Dive into the [Core Component Documentation](histogram_generation_guide.md)
2. Study the [Sinkhorn Reranking Logic](sinkhorn_reranking_logic.md)
3. Examine the [FAISS & DuckDB Implementation](faiss_duckdb_guide.md)
4. Contribute to the project or extend functionality

## 🛠️ Development Tools

### **Testing Tools**

- **`tools/test_histogram_generation.py`** - Validate histogram generation
- **`tools/test_faiss_duckdb.py`** - Test search components
- **`tools/test_search_system.py`** - End-to-end system testing
- **`tools/demo_visualization.py`** - Visualization feature demonstration

### **Utility Scripts**

- **`scripts/build_index.py`** - Build search indices (comprehensive offline indexing)
- **`scripts/run_sanity_checks.py`** - System validation

### **API Testing**

- **`tools/test_api.py`** - Web API validation
- **Interactive docs**: `http://localhost:8000/docs`

## 🔍 Troubleshooting

### **Common Issues**

- **"Search components not initialized"** → [Build the index](complete_usage_guide.md#building-the-search-index)
- **"Failed to generate visualization"** → [Check package installation](troubleshooting_comprehensive.md#issue-failed-to-generate-visualization)
- **Import errors** → [Activate virtual environment](troubleshooting_comprehensive.md#issue-virtual-environment-not-working)

### **Getting Help**

1. **Check the [Comprehensive Troubleshooting Guide](troubleshooting_comprehensive.md)**
2. **Run sanity checks**: `python scripts/run_sanity_checks.py`
3. **Review logs** in the `logs/` directory
4. **Test components** using tools in the `tools/` directory

## 📈 Performance Benchmarks

### **Expected Performance**

- **Index Building**: 2-5 seconds (20 images), 5-15 minutes (5000 images)
- **Search Response**: 150-500ms total time
- **Query Visualization**: 50-200ms generation time
- **Results Collage**: 100-500ms for 10 images

### **Scalability**

- **Current Design**: Up to 10,000 images
- **Memory Usage**: 100-500MB runtime, 100-500MB index storage
- **Response Time**: Linear scaling with dataset size

## 🌟 Use Cases

### **Design & Creative**

- **Interior Design**: Find images matching room color schemes
- **Brand Identity**: Search for brand color combinations
- **Art Projects**: Discover color palettes for creative work

### **Research & Analysis**

- **Color Trends**: Analyze popular color combinations
- **Palette Creation**: Build harmonious color schemes
- **Contrast Testing**: Find high-contrast color pairs

### **Educational**

- **Color Theory**: Learn about color relationships
- **Visual Design**: Understand color weight and balance
- **Art History**: Explore color usage across different styles

## 🔮 Future Enhancements

### **Planned Features**

- **Distributed indexing** for multi-node deployments
- **Advanced caching** for improved performance
- **Machine learning** for query optimization
- **Real-time streaming** for live search results

### **Extensibility**

- **Plugin architecture** for custom algorithms
- **API versioning** for backward compatibility
- **Custom metrics** for specialized use cases
- **Integration APIs** for third-party services

## 📞 Support & Community

### **Documentation Issues**

- Check the [troubleshooting guides](troubleshooting_comprehensive.md)
- Review the [complete usage guide](complete_usage_guide.md)
- Test with the [validation tools](../tools/README.md)

### **Feature Requests**

- Review the [project architecture](project_architecture.md)
- Check the [progress report](progress.md)
- Consider contributing to the project

### **Technical Questions**

- Study the [component documentation](histogram_generation_guide.md)
- Examine the [API reference](api_reference.md)
- Test with the [development tools](../tools/README.md)

---

## 🎯 Documentation Goals

This documentation hub aims to provide:

- **📖 Comprehensive Coverage**: All aspects of the project documented
- **🚀 Quick Start**: Easy setup for new users
- **🔧 Deep Dive**: Technical details for developers
- **🛠️ Practical Examples**: Real-world usage scenarios
- **🔍 Troubleshooting**: Solutions for common issues
- **📈 Performance**: Benchmarks and optimization guidance

---

**🎨 Welcome to Chromatica!** Start your color search journey with the [Complete Usage Guide](complete_usage_guide.md) and explore the rich world of color-based image search.

_For the latest updates and development status, see the [Progress Report](progress.md)._
