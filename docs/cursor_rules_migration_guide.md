# Cursor Rules Migration Guide

## Overview

This document provides a comprehensive guide to the migration from the legacy `.cursorrules` format to the new Cursor Rules system. The Chromatica project has been updated to use the modern Cursor Rules format with proper MDC (Markdown with metadata) files and an alternative AGENTS.md approach.

## Migration Summary

### What Changed

1. **New Project Rules Structure**: Migrated from `.cursorrules` to `.cursor/rules/*.mdc` files
2. **MDC Format**: Using the new Markdown with metadata format for better organization
3. **AGENTS.md Alternative**: Created a simple markdown alternative for lightweight use
4. **Updated .cursorignore**: Modified to include new rule files and exclude legacy files
5. **Comprehensive Documentation**: Added detailed documentation for the new system

### What Stayed the Same

- **Core Project Rules**: All essential project rules and guidelines preserved
- **Critical Instructions**: Still references `docs/.cursor/critical_instructions.md` as source of truth
- **Technology Stack**: All technology requirements and specifications maintained
- **Development Workflow**: All development practices and standards preserved

## New Cursor Rules Structure

### Project Rules Directory

```
.cursor/rules/
├── chromatica-project.mdc          # Core project rules (Always Applied)
├── algorithmic-specifications.mdc  # Algorithm and implementation specs
├── api-endpoints.mdc              # API endpoint specifications
├── testing-validation.mdc         # Testing standards and validation
├── development-workflow.mdc       # Development environment and workflow
├── documentation-standards.mdc    # Documentation requirements
├── indexing-module.mdc           # FAISS and DuckDB implementation
├── tools-module.mdc              # Tools and utilities
├── web-interface.mdc             # Web interface and visualization tools
└── legacy-migration.mdc          # Migration notes and legacy support
```

### Rule Types and Application

#### Always Applied Rules

- **chromatica-project.mdc**: Core project rules that always apply
- **critical_instructions.md**: Referenced as single source of truth

#### Auto-Attached Rules

- **algorithmic-specifications.mdc**: Applied when working with core algorithms
- **api-endpoints.mdc**: Applied when working with API code
- **testing-validation.mdc**: Applied when working with tests
- **development-workflow.mdc**: Applied when working with development files

#### Manual Rules

- **documentation-standards.mdc**: Available for AI to include when relevant
- **tools-module.mdc**: Available for tool development
- **web-interface.mdc**: Available for web interface work

## AGENTS.md Alternative

### Simple Markdown Approach

The `AGENTS.md` file provides a lightweight alternative to the structured Project Rules system. It contains all essential project information in a single, readable markdown file.

### When to Use AGENTS.md

- **Simple Projects**: When you prefer a single file approach
- **Quick Setup**: For rapid project initialization
- **Readability**: When you want all rules in one place
- **Legacy Support**: For teams familiar with simple markdown rules

### AGENTS.md Contents

- Project overview and status
- Technology stack requirements
- Algorithmic specifications
- Development environment setup
- Code quality standards
- Integration rules
- Testing requirements
- Documentation lifecycle
- Common commands and workflows

## Migration Benefits

### Improved Organization

- **Modular Rules**: Rules are organized by topic and scope
- **Better Scoping**: Rules apply only when relevant files are being worked on
- **Easier Maintenance**: Individual rule files are easier to update and maintain
- **Clear Separation**: Different types of rules are clearly separated

### Enhanced Functionality

- **Auto-Attachment**: Rules automatically apply based on file patterns
- **Manual Control**: AI can choose to include relevant rules
- **Metadata Support**: Rich metadata for better rule management
- **Version Control**: Better version control with individual rule files

### Better Developer Experience

- **Focused Context**: Only relevant rules are included in AI context
- **Faster Loading**: Smaller rule files load faster
- **Easier Debugging**: Easier to identify which rules are affecting behavior
- **Flexible Configuration**: Easy to enable/disable specific rule sets

## Usage Guidelines

### For Developers

1. **Use Project Rules**: Prefer the new `.cursor/rules/*.mdc` system for comprehensive control
2. **Use AGENTS.md**: Use the simple markdown approach for quick setup or simple projects
3. **Don't Mix**: Don't use both systems simultaneously to avoid conflicts
4. **Update Rules**: Keep rules updated as the project evolves

### For AI Assistants

1. **Check Rule Type**: Understand whether rules are always applied, auto-attached, or manual
2. **Respect Scoping**: Only apply rules when working with relevant files
3. **Follow Metadata**: Use rule metadata to understand when and how to apply rules
4. **Maintain Consistency**: Ensure consistent behavior across different rule types

## Configuration Examples

### Creating a New Rule

```markdown
---
description: Description of what this rule covers
globs: ["src/chromatica/core/**/*"] # Files this rule applies to
alwaysApply: false # Whether to always include this rule
---

# Rule Title

Rule content goes here...
```

### Auto-Attached Rule Example

```markdown
---
description: Core algorithm specifications
globs: ["src/chromatica/core/**/*", "src/chromatica/indexing/**/*"]
alwaysApply: false
---

# Algorithmic Specifications

This rule applies when working with core algorithms...
```

### Always Applied Rule Example

```markdown
---
description: Core project rules
globs: []
alwaysApply: true
---

# Core Project Rules

These rules always apply to the project...
```

## Best Practices

### Rule Organization

1. **Logical Grouping**: Group related rules together
2. **Clear Naming**: Use descriptive names for rule files
3. **Appropriate Scoping**: Use globs to target relevant files
4. **Consistent Format**: Maintain consistent formatting across rules

### Rule Content

1. **Clear Instructions**: Write clear, actionable instructions
2. **Specific Examples**: Include specific examples where helpful
3. **Context Information**: Provide necessary context for understanding
4. **Regular Updates**: Keep rules updated with project changes

### Performance Considerations

1. **Relevant Rules**: Only include rules that are actually needed
2. **Efficient Globs**: Use efficient glob patterns for file matching
3. **Appropriate Size**: Keep individual rule files reasonably sized
4. **Regular Cleanup**: Remove outdated or unused rules

## Troubleshooting

### Common Issues

#### Rules Not Applying

- **Check Globs**: Verify glob patterns match the files you're working with
- **Check Metadata**: Ensure `alwaysApply` is set correctly
- **Check File Location**: Ensure rule files are in the correct directory

#### Conflicting Rules

- **Review Content**: Check for conflicting instructions between rules
- **Check Priority**: Understand which rules take precedence
- **Consolidate**: Merge or remove conflicting rules

#### Performance Issues

- **Check Rule Size**: Large rule files can slow down AI responses
- **Check Globs**: Overly broad glob patterns can cause unnecessary rule loading
- **Optimize Content**: Remove unnecessary content from rules

### Debugging Tips

1. **Check Active Rules**: Use Cursor's rule management interface to see active rules
2. **Test Rule Application**: Test rules with different file types to ensure proper scoping
3. **Review AI Behavior**: Monitor AI behavior to ensure rules are being followed
4. **Update as Needed**: Update rules based on observed behavior and project needs

## Migration Checklist

### Pre-Migration

- [ ] Review existing `.cursorrules` content
- [ ] Identify rule categories and scoping needs
- [ ] Plan new rule file structure
- [ ] Backup existing configuration

### During Migration

- [ ] Create new `.cursor/rules/` directory structure
- [ ] Convert rules to MDC format with proper metadata
- [ ] Create AGENTS.md alternative
- [ ] Update `.cursorignore` file
- [ ] Test new rule system

### Post-Migration

- [ ] Verify all rules are working correctly
- [ ] Test AI behavior with new rules
- [ ] Update team documentation
- [ ] Remove or archive legacy `.cursorrules` file
- [ ] Monitor and adjust rules as needed

## Future Considerations

### Rule Evolution

- **Regular Reviews**: Schedule regular reviews of rule effectiveness
- **User Feedback**: Collect feedback on rule clarity and usefulness
- **Project Changes**: Update rules as project requirements change
- **Best Practices**: Incorporate new best practices as they emerge

### Advanced Features

- **Conditional Rules**: Use more sophisticated rule conditions
- **Rule Dependencies**: Implement rule dependencies where needed
- **Dynamic Rules**: Consider dynamic rule generation for complex scenarios
- **Integration**: Integrate with other development tools and workflows

## Conclusion

The migration to the new Cursor Rules system provides better organization, improved functionality, and enhanced developer experience. The modular approach allows for more targeted and efficient rule application, while the AGENTS.md alternative provides a simple option for teams that prefer a single-file approach.

By following the guidelines and best practices outlined in this document, teams can effectively leverage the new Cursor Rules system to improve their development workflow and AI assistant interactions.

## References

- [Cursor Rules Documentation](https://docs.cursor.com/en/context/rules)
- [Chromatica Critical Instructions](docs/.cursor/critical_instructions.md)
- [Project Documentation](docs/README.md)
- [Legacy .cursorrules](.cursorrules) (for reference)
