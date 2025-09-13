# Chromatica Color Search Engine - Agent Instructions

## Project Overview

Chromatica is a production-ready, two-stage color search engine that retrieves images whose dominant palettes best match weighted, multi-color queries. The system uses CIE Lab color space, FAISS HNSW indexing, and Sinkhorn-EMD reranking for high-fidelity color-based image search.

## Critical Instructions Compliance

**MANDATORY**: Always consult `docs/.cursor/critical_instructions.md` before making any architectural decisions or implementing new features. This document is the single source of truth for the project.

Before generating any code or making suggestions, state: "Consulting critical_instructions.md..."

## Implementation Status

### ✅ Completed (Week 1 & 2)

- Core histogram generation pipeline (`src/chromatica/core/histogram.py`)
- Configuration management (`src/chromatica/utils/config.py`)
- FAISS HNSW index and DuckDB metadata store (`src/chromatica/indexing/store.py`)
- Web interface with Catppuccin Mocha theme and custom typography
- Advanced Visualization Tools (6 tools with expandable panels)
- Comprehensive testing infrastructure

### 🔄 Current Focus

- Performance optimization and production deployment
- API endpoint refinement and error handling
- Comprehensive evaluation and benchmarking

## Technology Stack

### Core Libraries

- **opencv-python**: Image loading and resizing
- **scikit-image**: sRGB to CIE Lab color conversion
- **numpy**: Numerical operations and vectorized histogram generation
- **faiss-cpu**: ANN index with IndexHNSWFlat (M=32)
- **POT**: Sinkhorn-EMD reranking stage
- **DuckDB**: Metadata and raw histogram storage
- **FastAPI**: Web API framework

### Web Interface

- **Catppuccin Mocha theme**: 25-color palette with CSS variables
- **JetBrains Mono Nerd Font Mono**: Primary typography
- **Segoe UI Emoji/Symbol**: Emoji and symbol rendering
- **WCAG compliance**: Accessibility standards

## Algorithmic Specifications

### Color Processing

- Color space: **CIE Lab (D65 illuminant)**
- Binning grid: **8x12x12 (L* a* b\*)** = 1,152 dimensions
- Assignment: **Tri-linear soft assignment** for robustness
- Normalization: **L1 normalization** to create probability distributions

### Search Pipeline

1. **ANN Search**: Hellinger-transformed histograms in FAISS HNSW index
2. **Reranking**: Sinkhorn-EMD on raw histograms (K=200 candidates)
3. **Cost Matrix**: Pre-computed squared Euclidean distance between bin centers

### Performance Targets

- Histogram generation: ~200ms per image
- Total search latency (P95): <450ms
- ANN search: <150ms
- Reranking: <300ms

## Development Environment

### Virtual Environment

- **Location**: `venv311` (Python 3.11)
- **Activation**: `venv311\Scripts\activate` (Windows)
- **Always activate** before running any project commands, tests, or scripts

### Test Datasets

- **test-dataset-20**: Quick development testing (20 images)
- **test-dataset-50**: Small-scale validation (50 images)
- **test-dataset-200**: Medium-scale testing (200 images)
- **test-dataset-5000**: Production-scale testing (5,000 images, expand to 7,500)

## Code Quality Standards

### Python Standards

- **Python 3.10+** with comprehensive type hints
- **PEP 8** style guide compliance
- **Google-style docstrings** for all functions, classes, and modules
- **Comprehensive input validation** and error handling
- **Performance monitoring** built into all components

### Documentation Requirements

- **MANDATORY**: Documentation updates for ALL code changes
- **Comprehensive coverage**: Purpose, features, usage, troubleshooting
- **Practical examples**: Code snippets and integration patterns
- **Quality standards**: Accuracy, completeness, clarity, consistency

## Integration Rules

### New Components

- Must integrate seamlessly with existing histogram generation pipeline
- Follow established project structure and naming conventions
- Include proper `__init__.py` files and import statements
- Add dependencies to `requirements.txt` with version constraints

### Testing Requirements

- Use existing test datasets for validation
- All histograms must pass validation (1152 dimensions, L1 normalization)
- Performance targets: ~200ms per image, 100% validation success rate
- Use testing tools for validating new implementations

## Web Interface Standards

### Visualization Tools

- **6 Advanced Tools**: Color Palette Analyzer, Search Results Analyzer, Interactive Color Explorer, Histogram Analysis Tool, Distance Debugger Tool, Query Visualizer Tool
- **Expandable panels** with comprehensive configuration options
- **Quick Test functionality** using actual datasets from `datasets/quick-test/`
- **Real execution** with dynamic results, not hardcoded placeholders

### Theme Implementation

- **Catppuccin Mocha**: Complete 25-color palette implementation
- **CSS Variables**: Centralized color management
- **Responsive Design**: Mobile-optimized with accessibility compliance
- **Typography**: JetBrains Mono Nerd Font Mono + Segoe UI fonts

## Error Handling and Logging

### Logging Standards

- Use standard `logging` module in all scripts and API endpoints
- Log key events, errors, and performance metrics
- Appropriate levels: DEBUG, INFO, WARNING, ERROR
- Descriptive and actionable error messages

### Troubleshooting

- Systematic use of logs and debugging tools
- Document resolutions in `docs/troubleshooting.md`
- Include command to run scripts in file headers or instructions

## Documentation Lifecycle

### Mandatory Updates

- **New Features**: Every new feature, module, class, or function
- **Bug Fixes**: All bug fixes and error resolutions
- **Enhancements**: Performance improvements and optimizations
- **API Changes**: Endpoint modifications and model updates
- **Configuration Changes**: New constants and environment variables

### Documentation Structure

```
docs/
├── api/                    # API endpoint documentation
├── guides/                 # User and developer guides
├── modules/                # Module-specific documentation
├── tools/                  # Tool documentation
├── troubleshooting/        # Problem resolution guides
├── progress.md            # Project progress tracking
└── README.md              # Project overview
```

## Success Metrics

### Quality Indicators

- **Completeness**: 100% of functionality documented
- **Accuracy**: 0% documentation errors or outdated information
- **Performance**: Meet all specified latency and throughput targets
- **Validation**: 100% success rate for histogram validation
- **Integration**: Seamless integration with existing components

### Compliance Requirements

- **Mandatory**: No code changes without documentation updates
- **Timing**: Documentation updated before or simultaneously with code
- **Quality**: All documentation meets established standards
- **Testing**: All examples and procedures verified and tested

## Tool Usage Guidelines

### Available Tools

- **filesystem**: File and directory manipulations
- **brave-search/gitmcp-docs**: Research external information and documentation
- **gitmvp**: Repository state understanding
- **context7**: Library documentation and API understanding
- **firecrawl**: Complex web scraping tasks

### Development Commands

- **Virtual Environment**: Always activate `venv311` before running commands
- **Testing**: Use `pytest tests/ -v` for running tests
- **Index Building**: Use `python scripts/build_index.py`
- **API Testing**: Use `python tools/test_api.py`

## Project Structure

```
src/chromatica/
├── core/                   # Core algorithms and data structures
├── indexing/              # FAISS index and DuckDB store management
├── api/                   # FastAPI application and web interface
├── utils/                 # Configuration and utility functions
└── visualization/         # Visualization tools and utilities

tools/                     # Testing and development tools
scripts/                   # Build and maintenance scripts
docs/                      # Comprehensive documentation
datasets/                  # Test datasets and quick test data
```

## Key Files and Modules

### Core Implementation

- `src/chromatica/core/histogram.py`: Histogram generation with tri-linear soft assignment
- `src/chromatica/core/query.py`: Query processing and color conversion
- `src/chromatica/core/rerank.py`: Sinkhorn-EMD reranking implementation

### Indexing System

- `src/chromatica/indexing/store.py`: FAISS HNSW index and DuckDB metadata store
- `src/chromatica/indexing/pipeline.py`: Image processing pipeline

### API and Web Interface

- `src/chromatica/api/main.py`: FastAPI application with search endpoints
- `src/chromatica/api/static/index.html`: Web interface with visualization tools

### Configuration and Utilities

- `src/chromatica/utils/config.py`: Global constants and configuration
- `requirements.txt`: Project dependencies with version constraints

## Development Workflow

### Before Starting Work

1. Activate virtual environment: `venv311\Scripts\activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest tests/ -v`
4. Check configuration: `python -c "from src.chromatica.utils.config import validate_config; validate_config()"`

### During Development

1. Write tests for new functionality
2. Implement feature with proper documentation
3. Run tests to ensure everything works
4. Update documentation for all changes
5. Test with appropriate test datasets

### Before Committing

1. Run all tests: `pytest tests/ -v`
2. Check code quality: `flake8 src/ tests/ tools/`
3. Verify documentation is updated
4. Test with different test datasets
5. Ensure performance targets are met

## Common Commands

### Development

```bash
# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_histogram.py::TestHistogramGeneration -v

# Build index
python scripts/build_index.py

# Test API
python tools/test_api.py

# Run histogram testing
python tools/test_histogram_generation.py
```

### API Development

```bash
# Start API server
uvicorn src.chromatica.api.main:app --reload

# Test API endpoints
curl "http://localhost:8000/search?colors=FF0000,00FF00&weights=0.5,0.5&k=10"

# Run API tests
python tools/test_api.py
```

### Index Management

```bash
# Build index from test dataset
python scripts/build_index.py --dataset datasets/test-dataset-200

# Run sanity checks
python scripts/run_sanity_checks.py

# Test index functionality
python tools/test_faiss_duckdb.py
```

## Troubleshooting

### Common Issues

- **Virtual Environment**: Always activate venv311 before running commands
- **Dependencies**: Ensure all dependencies are installed and up to date
- **Configuration**: Validate configuration with `validate_config()`
- **Test Datasets**: Use appropriate test datasets for development and testing

### Getting Help

- Check `docs/troubleshooting.md` for common issues and solutions
- Review logs in `logs/` directory for error information
- Use testing tools to validate functionality
- Consult `docs/.cursor/critical_instructions.md` for project specifications

## Performance Monitoring

### Key Metrics

- **Histogram Generation**: ~200ms per image target
- **Search Latency**: <450ms total (P95)
- **Memory Usage**: Monitor memory usage for large datasets
- **Error Rates**: Track and minimize error rates

### Monitoring Tools

- Use built-in performance monitoring in all components
- Run performance tests regularly
- Monitor logs for performance issues
- Use testing tools for performance validation

## Security Considerations

### Development Security

- Validate all inputs thoroughly
- Use secure coding practices
- Regular dependency updates for security patches
- Don't expose sensitive information in error messages

### Production Security

- Implement appropriate access controls
- Protect sensitive data
- Use secure communication protocols
- Monitor for security issues

## Maintenance and Updates

### Regular Maintenance

- Update dependencies regularly
- Monitor performance and optimize as needed
- Update documentation for all changes
- Run comprehensive tests regularly

### Version Control

- Use proper git practices
- Write clear commit messages
- Use feature branches for development
- Review code before merging

## Success Criteria

### Technical Success

- All tests pass consistently
- Performance targets are met
- Documentation is complete and accurate
- Code quality standards are maintained

### Project Success

- System is production-ready
- All features work as specified
- User experience is excellent
- System is maintainable and extensible
