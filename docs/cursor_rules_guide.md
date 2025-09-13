# Cursor Rules Guide for Chromatica Project

## Overview

This guide documents the comprehensive Cursor rules system implemented for the Chromatica color search engine project. The rules system has been migrated from the legacy `.cursorrules` format to the modern `.cursor/rules` directory structure with MDC (Markdown with metadata) format files.

## Rules Architecture

### Directory Structure

```
.cursor/
└── rules/
    ├── chromatica-core.mdc           # Core project rules (always applied)
    ├── core-module.mdc               # Core module development rules
    ├── api-module.mdc                # API module development rules
    ├── indexing-module.mdc           # Indexing module development rules
    ├── tools-module.mdc              # Tools and scripts development rules
    ├── documentation-standards.mdc   # Documentation standards and lifecycle
    ├── testing-standards.mdc         # Testing standards and validation
    ├── web-interface.mdc             # Web interface development standards
    ├── development-workflow.mdc      # Development environment and workflow
    └── legacy-migration.mdc          # Migration guide from legacy format
```

### Alternative Format

- **AGENTS.md**: Simple markdown alternative for straightforward use cases

## Rule Types and Application

### Always Applied Rules

- **chromatica-core.mdc**: Core project rules that are always included in model context
- **Purpose**: Ensures consistent adherence to project standards and critical instructions
- **Scope**: Applies to all files and contexts

### Auto-Attached Rules

- **core-module.mdc**: Automatically applied when working with `src/chromatica/core/**/*`
- **api-module.mdc**: Automatically applied when working with `src/chromatica/api/**/*`
- **indexing-module.mdc**: Automatically applied when working with `src/chromatica/indexing/**/*`
- **tools-module.mdc**: Automatically applied when working with `tools/**/*` or `scripts/**/*`

### Agent-Requested Rules

- **documentation-standards.mdc**: Available for AI to include when documentation work is needed
- **testing-standards.mdc**: Available for AI to include when testing work is needed
- **web-interface.mdc**: Available for AI to include when web interface work is needed
- **development-workflow.mdc**: Available for AI to include when development workflow guidance is needed

### Manual Rules

- **legacy-migration.mdc**: Only included when explicitly mentioned using `@legacy-migration`

## Rule Content Overview

### Core Project Rules (chromatica-core.mdc)

- **Critical Instructions Compliance**: Mandatory reference to `docs/.cursor/critical_instructions.md`
- **Project Overview**: Complete project status and implementation details
- **Technology Stack**: Required libraries and frameworks
- **Algorithmic Specifications**: Core algorithm requirements and performance targets
- **Development Environment**: Virtual environment setup and test datasets
- **Code Quality Standards**: Python standards, documentation requirements, integration rules
- **Web Interface Standards**: Visualization tools and theme implementation
- **Error Handling**: Logging standards and troubleshooting procedures
- **Documentation Lifecycle**: Mandatory documentation updates for all changes
- **Success Metrics**: Quality indicators and compliance requirements

### Module-Specific Rules

#### Core Module (core-module.mdc)

- **Histogram Generation**: Tri-linear soft assignment and vectorized operations
- **Query Processing**: Hex color to Lab conversion and weighted histogram generation
- **Reranking**: Sinkhorn-EMD implementation using POT library
- **Performance Requirements**: ~200ms per image, <300ms for reranking
- **Integration Points**: Configuration dependencies and external libraries
- **Testing Strategy**: Unit tests, integration tests, and validation tests

#### API Module (api-module.mdc)

- **FastAPI Implementation**: Endpoint specifications and response formats
- **Web Interface**: Catppuccin Mocha theme and typography system
- **Visualization Tools**: 6 advanced tools with expandable panels
- **Performance Targets**: <450ms total latency, <500ms API response time
- **Security Considerations**: Input validation and data protection
- **Testing Requirements**: API testing, web interface testing, and tool testing

#### Indexing Module (indexing-module.mdc)

- **FAISS Index Management**: HNSW index with Hellinger transform
- **DuckDB Metadata Store**: Schema design and batch operations
- **Processing Pipeline**: End-to-end image processing workflow
- **Performance Requirements**: <150ms ANN search, efficient batch processing
- **Error Handling**: Index errors, database errors, and pipeline errors
- **Testing Requirements**: Unit tests, integration tests, and performance tests

#### Tools Module (tools-module.mdc)

- **Testing Tools**: Comprehensive testing and validation utilities
- **Demonstration Tools**: Interactive demonstration and visualization tools
- **Build Scripts**: Index building and system validation scripts
- **Performance Requirements**: Efficient execution and resource utilization
- **Integration Points**: Core module integration and configuration integration
- **Maintenance Guidelines**: Tool updates, performance monitoring, and documentation

### Specialized Rules

#### Documentation Standards (documentation-standards.mdc)

- **Mandatory Requirements**: Documentation updates for ALL project changes
- **Quality Standards**: Comprehensive coverage, accuracy, clarity, consistency
- **Documentation Types**: API documentation, user guides, developer guides
- **File Organization**: Structured documentation directory layout
- **Update Workflow**: Pre-implementation, during implementation, post-implementation
- **Maintenance Schedule**: Regular maintenance tasks and documentation debt management
- **Success Metrics**: Quality indicators and compliance requirements

#### Testing Standards (testing-standards.mdc)

- **Test-Driven Development**: Write tests first, maintain high coverage
- **Testing Infrastructure**: Test datasets and testing tools
- **Unit Testing**: Test structure, coverage, and examples
- **Integration Testing**: End-to-end testing and API integration testing
- **Performance Testing**: Performance targets and benchmarking
- **Validation Testing**: Algorithmic validation and data validation
- **Testing Workflow**: Pre-commit testing, CI/CD integration, release testing

#### Web Interface (web-interface.mdc)

- **Theme Implementation**: Catppuccin Mocha theme with 25-color palette
- **Typography System**: JetBrains Mono Nerd Font Mono with fallback strategy
- **Responsive Design**: Mobile-first approach with consistent breakpoints
- **Accessibility Standards**: WCAG compliance and accessibility features
- **Advanced Visualization Tools**: 6 tools with expandable panels and real execution
- **JavaScript Standards**: Modern JavaScript with proper error handling
- **CSS Standards**: BEM methodology and performance optimization

#### Development Workflow (development-workflow.mdc)

- **Virtual Environment**: Python 3.11 with venv311 activation
- **Dependency Management**: Requirements file structure and installation
- **Development Tools**: Code quality tools and IDE configuration
- **Development Process**: Feature branch workflow and code quality checks
- **Logging and Monitoring**: Structured logging and performance monitoring
- **Testing Workflow**: Test execution and continuous integration
- **Build and Deployment**: Build process and production deployment

## Usage Guidelines

### For Developers

#### Automatic Rule Application

Most rules are automatically applied based on file patterns:

- Working on `src/chromatica/core/` files → core-module.mdc rules apply
- Working on `src/chromatica/api/` files → api-module.mdc rules apply
- Working on `tools/` or `scripts/` files → tools-module.mdc rules apply

#### Manual Rule Invocation

Use `@ruleName` to manually invoke specific rules:

- `@documentation-standards` for documentation guidance
- `@testing-standards` for testing guidance
- `@web-interface` for web development guidance
- `@development-workflow` for development process guidance

#### Rule Discovery

Check `.cursor/rules/` directory for available rules and their descriptions.

### For AI Assistant

#### Rule Selection

The AI assistant can automatically select relevant rules based on:

- Current file context and directory
- Type of work being performed
- Specific requirements or questions

#### Context Awareness

Rules provide better context for AI assistance by:

- Understanding project structure and requirements
- Knowing implementation status and constraints
- Providing specific guidance for different types of work

#### Scoped Assistance

More targeted assistance based on:

- Current file/context
- Type of development work
- Specific module or component being worked on

## Migration from Legacy Format

### Benefits of New Format

- **Better Organization**: Rules are better organized and categorized
- **Improved Performance**: More efficient rule application and processing
- **Enhanced Functionality**: More flexible and powerful rule system
- **Easier Maintenance**: Easier to maintain and update rules

### Migration Status

- **✅ Completed**: All legacy rules have been migrated to new format
- **✅ Organized**: Rules are properly organized by scope and purpose
- **✅ Enhanced**: Rules have been enhanced with better metadata and functionality
- **✅ Tested**: New rule system has been tested and validated

### Legacy File Status

- **`.cursorrules`**: Deprecated but maintained for reference
- **New Format**: All rules now in `.cursor/rules/` directory
- **Improved Organization**: Better scoping and organization of rules
- **Enhanced Functionality**: Better rule application and management

## Rule Maintenance

### Regular Updates

- **Project Evolution**: Update rules as project evolves
- **New Requirements**: Add new rules for new requirements
- **Rule Optimization**: Optimize rules for better performance
- **Documentation Updates**: Keep rule documentation current

### Rule Validation

- **Effectiveness**: Monitor rule effectiveness and impact
- **Usage Patterns**: Track rule usage patterns
- **Performance Impact**: Monitor performance impact of rules
- **User Feedback**: Collect feedback on rule usefulness

### Best Practices

- **Focused Scope**: Keep rules focused and specific
- **Clear Purpose**: Make rule purpose clear and understandable
- **Appropriate Length**: Keep rules concise but comprehensive
- **Regular Updates**: Update rules regularly to stay current

## Troubleshooting

### Common Issues

- **Rule Not Applying**: Check glob patterns and rule metadata
- **Rule Conflicts**: Resolve conflicts between overlapping rules
- **Performance Issues**: Optimize rules for better performance
- **Context Overload**: Reduce rule context when needed

### Resolution Steps

1. **Check Rule Metadata**: Verify rule metadata and glob patterns
2. **Test Rule Application**: Test rule application in different contexts
3. **Monitor Performance**: Monitor performance impact of rules
4. **Update Rules**: Update rules based on issues and feedback

## Future Enhancements

### Planned Improvements

- **Rule Templates**: Create templates for common rule types
- **Rule Analytics**: Add analytics for rule usage and effectiveness
- **Rule Sharing**: Enable sharing of rules between projects
- **Rule Versioning**: Add versioning for rule evolution

### Advanced Features

- **Dynamic Rules**: Rules that adapt based on project state
- **Rule Dependencies**: Rules that depend on other rules
- **Rule Inheritance**: Rules that inherit from parent rules
- **Rule Composition**: Ability to compose rules from smaller parts

## Conclusion

The new Cursor rules system provides significant improvements in organization, functionality, and performance for the Chromatica project. The system ensures consistent adherence to project standards while providing targeted assistance for different types of development work.

### Key Benefits

- **Better Organization**: Rules are better organized and categorized
- **Improved Performance**: More efficient rule application and processing
- **Enhanced Functionality**: More flexible and powerful rule system
- **Easier Maintenance**: Easier to maintain and update rules

### Next Steps

- **Monitor Usage**: Monitor rule usage and effectiveness
- **Gather Feedback**: Collect feedback from developers and AI assistant
- **Iterate and Improve**: Continuously improve rules based on usage patterns
- **Document Best Practices**: Document best practices for rule creation and usage
