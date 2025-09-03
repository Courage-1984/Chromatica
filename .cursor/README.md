# Chromatica Cursor Configuration

This directory contains Cursor-specific configuration and documentation for the Chromatica color search engine project.

## 📁 Cursor Configuration Overview

The `.cursor` directory provides:

- **AI Assistant Configuration**: Rules and guidelines for AI development assistance
- **Project Context**: Critical project specifications and development guidelines
- **Cursor-Specific Settings**: IDE configuration and AI behavior rules
- **Development Workflow**: AI-assisted development patterns and best practices

## 🚀 Key Files

### `critical_instructions.md`

The primary project plan and technical specifications document that serves as the single source of truth for the Chromatica project.

**Contents:**

- Executive summary and system design
- Algorithm specifications and implementation details
- Technology stack and architectural decisions
- Development phases and milestones
- Performance targets and evaluation criteria

**Usage:**

- **AI Assistant**: Must consult this document before generating any code or making suggestions
- **Developers**: Primary reference for understanding project requirements
- **Architects**: Source of truth for system design decisions

## 🔧 Cursor AI Assistant Rules

### Behavior Guidelines

The AI assistant must follow these rules when working on the Chromatica project:

1. **Pre-Response Directive**: Always state "Consulting critical_instructions.md..." before responding
2. **Strict Adherence**: Follow the consolidated plan without improvisation on core architectural components
3. **Reference Requirements**: When implementing features, refer to relevant sections of critical_instructions.md
4. **Clarification First**: If details are missing, ask for clarification rather than making assumptions

### Technology Stack Compliance

The AI must ensure all code follows the specified technology stack:

- **Core**: Python 3.10+ with type hints
- **Image Processing**: OpenCV, scikit-image
- **Color Science**: CIE Lab color space (D65 illuminant)
- **Vector Search**: FAISS HNSW index (IndexHNSWFlat)
- **Database**: DuckDB for metadata and raw histograms
- **Optimal Transport**: POT library for Sinkhorn-EMD
- **Web Framework**: FastAPI

### Algorithmic Specifications

All implementations must follow the exact specifications:

- **Color Space**: CIE Lab (D65 illuminant)
- **Histogram Binning**: 8×12×12 (L* a* b\*) for 1,152 dimensions
- **Soft Assignment**: Tri-linear soft assignment for histogram generation
- **Normalization**: L1 normalization (sum = 1.0)
- **Hellinger Transform**: Element-wise square root for FAISS compatibility
- **Reranking**: Sinkhorn-approximated Earth Mover's Distance (EMD)

## 📚 Development Workflow

### AI-Assisted Development Patterns

#### Code Generation

```python
# AI must follow these patterns:
# 1. Consult critical_instructions.md for specifications
# 2. Use exact constants from config.py
# 3. Implement comprehensive validation
# 4. Include Google-style docstrings
# 5. Follow PEP 8 style guidelines
```

#### Testing and Validation

```python
# AI must ensure:
# 1. All functions have comprehensive input validation
# 2. Error messages are descriptive and actionable
# 3. Performance monitoring is built into components
# 4. Logging is implemented at appropriate levels
# 5. All code integrates with existing modules
```

### File Organization Rules

The AI must maintain the established project structure:

```
src/chromatica/
├── core/           # Core histogram generation and color processing
├── indexing/       # FAISS index and DuckDB storage implementation
├── api/            # FastAPI web API endpoints
└── utils/          # Configuration and utility functions

tools/              # Testing and development tools
datasets/           # Test datasets for validation
docs/               # Comprehensive documentation
```

## 🔍 AI Assistant Capabilities

### What the AI Can Do

1. **Code Generation**: Generate production-ready code following project specifications
2. **Documentation**: Create and update comprehensive documentation
3. **Testing**: Implement comprehensive testing infrastructure
4. **Integration**: Connect components following architectural guidelines
5. **Optimization**: Performance tuning and memory optimization
6. **Troubleshooting**: Debug issues and provide solutions

### What the AI Must Not Do

1. **Architectural Changes**: Modify core system design without consulting critical_instructions.md
2. **Technology Substitution**: Replace specified technologies with alternatives
3. **Algorithm Modification**: Change algorithmic specifications without approval
4. **Performance Compromise**: Implement solutions that don't meet performance targets

## 📊 Project Status Integration

### Current Development Phase

The AI must be aware of the current project status:

- **Week 1**: ✅ COMPLETED - Core histogram generation pipeline
- **Week 2**: 🔄 IN PROGRESS - FAISS HNSW index and DuckDB metadata store
- **Week 3+**: 📋 PLANNED - FastAPI web API and search functionality

### Implementation Priorities

1. **Immediate**: Complete FAISS and DuckDB integration
2. **Short-term**: Implement search and reranking pipeline
3. **Medium-term**: Develop FastAPI web endpoints
4. **Long-term**: Production deployment and optimization

## 🧪 Testing and Validation

### AI-Generated Code Requirements

All AI-generated code must:

1. **Pass Validation**: Meet histogram specifications (1152 dimensions, L1 normalization)
2. **Performance Targets**: Achieve ~200ms per image for histogram generation
3. **Memory Efficiency**: Use ~4.6KB per histogram
4. **Integration**: Work seamlessly with existing modules
5. **Testing**: Include comprehensive unit and integration tests

### Quality Assurance

The AI must ensure:

- **Type Safety**: Comprehensive type hints and validation
- **Error Handling**: Graceful error handling with informative messages
- **Documentation**: Clear docstrings and usage examples
- **Performance**: Optimized algorithms and memory usage
- **Maintainability**: Clean, readable, and well-structured code

## 🔧 Cursor IDE Integration

### AI Assistant Configuration

The AI assistant is configured to:

1. **Understand Context**: Read and comprehend the entire project structure
2. **Follow Rules**: Strictly adhere to critical_instructions.md specifications
3. **Generate Code**: Produce production-ready, tested code
4. **Maintain Consistency**: Follow established patterns and conventions
5. **Provide Guidance**: Offer explanations and best practices

### Development Workflow

1. **Consultation**: AI consults critical_instructions.md for specifications
2. **Implementation**: Generates code following exact requirements
3. **Validation**: Ensures code meets quality and performance standards
4. **Integration**: Connects with existing modules and systems
5. **Documentation**: Updates relevant documentation and guides

## 🚨 Common AI Assistant Issues

### When AI Gets Confused

1. **Missing Context**: AI may not have access to critical_instructions.md
2. **Unclear Requirements**: Specifications may be ambiguous or incomplete
3. **Integration Complexity**: Multiple components may have conflicting requirements
4. **Performance Constraints**: AI may not understand performance implications

### Resolution Strategies

1. **Re-read Instructions**: AI should re-consult critical_instructions.md
2. **Ask for Clarification**: Request specific details when requirements are unclear
3. **Check Integration**: Verify compatibility with existing components
4. **Validate Performance**: Ensure solutions meet performance targets

## 🔮 Future Enhancements

### AI Assistant Improvements

1. **Context Awareness**: Better understanding of project evolution and status
2. **Performance Optimization**: Automated performance analysis and suggestions
3. **Integration Testing**: Automated testing of AI-generated components
4. **Documentation Sync**: Automatic documentation updates for code changes

### Project Evolution

1. **Phase Transitions**: AI must adapt to new development phases
2. **Technology Updates**: Handle new requirements and technology changes
3. **Scale Considerations**: Adapt solutions for production-scale deployment
4. **Performance Tuning**: Optimize solutions based on real-world usage

## 📞 Getting Help

### AI Assistant Issues

- **Confusion**: Re-consult critical_instructions.md for clarity
- **Integration Problems**: Check existing module interfaces and requirements
- **Performance Issues**: Validate against specified performance targets
- **Quality Concerns**: Ensure code meets established quality standards

### Project Guidance

- **Architecture Questions**: Refer to critical_instructions.md and architecture documents
- **Implementation Details**: Check existing code and documentation
- **Testing Requirements**: Follow established testing patterns and validation
- **Performance Targets**: Consult performance specifications and benchmarks

---

**Last Updated**: December 2024  
**AI Assistant Version**: Cursor AI v1.0  
**Project Phase**: Week 2 - FAISS and DuckDB Integration  
**Status**: AI Assistant fully configured and operational
