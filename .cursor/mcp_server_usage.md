# MCP Server Usage Guide

## Overview

This document provides comprehensive guidance on using Model Context Protocol (MCP) servers and their tools within the Chromatica color search engine project. MCP servers enable AI assistants to interact with external systems, databases, and services through standardized interfaces.

## What are MCP Servers?

MCP (Model Context Protocol) servers are external services that provide specialized functionality to AI assistants. They enable:

- **File system operations**: Reading, writing, and managing files
- **Web search and research**: Finding current information and documentation
- **Code repository access**: Reading and analyzing GitHub repositories
- **Library documentation**: Accessing up-to-date API documentation
- **Web scraping**: Extracting content from websites
- **Database operations**: Managing and querying data

## MCP Servers in Chromatica

The Chromatica project uses several MCP servers to enhance development capabilities:

1. **filesystem** - File and directory management
2. **brave-search** - Web search and research
3. **firecrawl** - Advanced web scraping
4. **gitmcp-docs** - GitHub repository documentation access
5. **context7** - Library documentation and API reference
6. **gitmvp** - GitHub repository analysis and file access

## When to Use MCP Servers

Based on the Chromatica project rules and critical instructions:

- **Use `filesystem`** for all file and directory manipulations
- **Use `brave-search`** for researching external information, library documentation, or error solutions
- **Use `gitmvp`** when needing to understand the current state of the repository
- **Use `context7`** when encountering a new library to get its documentation
- **Use `firecrawl`** for complex web scraping tasks during research
- **Use `gitmcp-docs`** for accessing GitHub repository documentation

---

## 1. Filesystem MCP Server

### Overview

The filesystem MCP server provides comprehensive file and directory management capabilities. It's the primary tool for file operations in the Chromatica project.

### When to Use

- **File and directory manipulations** (as specified in project rules)
- **Reading and writing project files**
- **Creating directory structures**
- **Managing project assets and outputs**

### Key Functions

#### File Reading

```python
# Read entire file as text
mcp_filesystem_read_text_file(path="src/chromatica/core/histogram.py")

# Read file with specific line ranges
mcp_filesystem_read_text_file(path="requirements.txt", head=20)

# Read media files (images, audio)
mcp_filesystem_read_media_file(path="datasets/test-dataset-20/test.jpg")
```

#### File Writing and Editing

```python
# Create new file or overwrite existing
mcp_filesystem_write_file(path="new_file.py", content="# New Python file")

# Edit existing file with line-based changes
mcp_filesystem_edit_file(path="existing_file.py", edits=[
    {"oldText": "old content", "newText": "new content"}
])
```

#### Directory Operations

```python
# List directory contents
mcp_filesystem_list_directory(path="src/chromatica/")

# Create directories
mcp_filesystem_create_directory(path="new_module/")

# Get directory tree structure
mcp_filesystem_directory_tree(path="src/")
```

#### File Management

```python
# Move/rename files
mcp_filesystem_move_file(source="old_name.py", destination="new_name.py")

# Search for files
mcp_filesystem_search_files(path="src/", pattern="*.py")

# Get file information
mcp_filesystem_get_file_info(path="requirements.txt")
```

### Usage Examples from Chromatica Project

#### Creating Project Structure

```python
# Create core module directory
mcp_filesystem_create_directory(path="src/chromatica/core/")

# Create configuration file
mcp_filesystem_write_file(
    path="src/chromatica/utils/config.py",
    content="# Configuration constants for Chromatica"
)
```

#### Reading Project Files

```python
# Read histogram module
mcp_filesystem_read_text_file(path="src/chromatica/core/histogram.py")

# Read requirements
mcp_filesystem_read_text_file(path="requirements.txt")
```

#### Managing Test Outputs

```python
# Create output directories
mcp_filesystem_create_directory(path="histograms/")
mcp_filesystem_create_directory(path="reports/")

# Save test results
mcp_filesystem_write_file(
    path="reports/test_results.json",
    content=json.dumps(results, indent=2)
)
```

### Best Practices

1. **Always specify absolute paths** when possible
2. **Use appropriate file reading functions** (text vs. media)
3. **Create directories before writing files** to avoid errors
4. **Handle file operations systematically** for project consistency
5. **Use search functions** to find files when exact paths are unknown

---

## 2. Brave Search MCP Server

### Overview

The Brave Search MCP server provides web search capabilities for research, documentation lookup, and finding current information. It's essential for staying updated with the latest libraries and solving technical problems.

### When to Use

- **Researching external information** (as specified in project rules)
- **Finding library documentation** and API references
- **Solving error solutions** and troubleshooting
- **Getting current information** about libraries and tools
- **Researching best practices** for implementation

### Key Functions

#### Web Search

```python
# General web search
mcp_brave_search_brave_web_search(query="FAISS HNSW index performance", count=10)

# News search
mcp_brave_search_brave_news_search(query="Python image processing libraries 2024", count=5)

# Local search (for business/place information)
mcp_brave_search_brave_local_search(query="Python meetups near me", count=10)
```

#### Image Search

```python
# Search for images
mcp_brave_search_brave_image_search(searchTerm="color histogram visualization", count=3)

# Find diagrams and charts
mcp_brave_search_brave_image_search(searchTerm="CIE Lab color space diagram", count=2)
```

#### Video Search

```python
# Search for video content
mcp_brave_search_brave_video_search(query="FAISS tutorial Python", count=5)
```

### Usage Examples from Chromatica Project

#### Researching Library Versions

```python
# Find latest stable versions of required libraries
mcp_brave_search_brave_web_search(
    query="opencv-python latest stable version 2024",
    count=5
)

# Research FAISS performance characteristics
mcp_brave_search_brave_web_search(
    query="FAISS HNSW vs IVF performance comparison",
    count=8
)
```

#### Finding Documentation

```python
# Search for POT library documentation
mcp_brave_search_brave_web_search(
    query="Python Optimal Transport POT library documentation",
    count=5
)

# Find DuckDB Python examples
mcp_brave_search_brave_web_search(
    query="DuckDB Python examples batch operations",
    count=6
)
```

#### Troubleshooting Issues

```python
# Search for error solutions
mcp_brave_search_brave_web_search(
    query="scikit-image rgb2lab D65 illuminant error",
    count=5
)

# Find performance optimization tips
mcp_brave_search_brave_web_search(
    query="NumPy histogram generation optimization tips",
    count=7
)
```

### Best Practices

1. **Use specific, targeted queries** for better results
2. **Specify count parameter** to control result volume
3. **Combine with other MCP servers** for comprehensive research
4. **Verify information** from multiple sources when possible
5. **Use news search** for time-sensitive information
6. **Include year/version** in queries for current information

---

## 3. Firecrawl MCP Server

### Overview

The Firecrawl MCP server provides advanced web scraping capabilities for extracting content from websites, mapping site structures, and performing complex data extraction tasks. It's ideal for research that requires detailed content analysis.

### When to Use

- **Complex web scraping tasks** during research (as specified in project rules)
- **Extracting specific content** from documentation websites
- **Mapping website structures** for comprehensive analysis
- **Structured data extraction** from web pages
- **Batch processing** of multiple web pages

### Key Functions

#### Single Page Scraping

```python
# Scrape a single webpage
mcp_firecrawl_firecrawl_scrape(
    url="https://faiss.ai/",
    formats=["markdown"],
    onlyMainContent=True
)

# Scrape with specific content focus
mcp_firecrawl_firecrawl_scrape(
    url="https://docs.python.org/3/library/",
    formats=["markdown"],
    includeTags=["h1", "h2", "h3", "p", "code"]
)
```

#### Website Mapping

```python
# Discover all URLs on a website
mcp_firecrawl_firecrawl_map(
    url="https://faiss.ai/",
    limit=100,
    includeSubdomains=False
)

# Search for specific content within a site
mcp_firecrawl_firecrawl_map(
    url="https://faiss.ai/",
    search="HNSW index",
    limit=50
)
```

#### Batch Scraping

```python
# Start a crawl job
mcp_firecrawl_firecrawl_crawl(
    url="https://faiss.ai/docs/*",
    maxDiscoveryDepth=3,
    limit=100,
    scrapeOptions={
        "formats": ["markdown"],
        "onlyMainContent": True
    }
)

# Check crawl status
mcp_firecrawl_firecrawl_check_crawl_status(id="crawl_job_id")
```

#### Structured Data Extraction

```python
# Extract specific information using LLM
mcp_firecrawl_firecrawl_extract(
    urls=["https://faiss.ai/", "https://github.com/facebookresearch/faiss"],
    prompt="Extract library features, performance characteristics, and API information",
    schema={
        "type": "object",
        "properties": {
            "features": {"type": "array", "items": {"type": "string"}},
            "performance": {"type": "string"},
            "api_info": {"type": "string"}
        }
    }
)
```

### Usage Examples from Chromatica Project

#### Researching FAISS Documentation

```python
# Map FAISS documentation structure
mcp_firecrawl_firecrawl_map(
    url="https://faiss.ai/",
    limit=200,
    includeSubdomains=False
)

# Scrape specific FAISS pages
mcp_firecrawl_firecrawl_scrape(
    url="https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSW.html",
    formats=["markdown"],
    onlyMainContent=True
)
```

#### Extracting Library Information

```python
# Extract information about multiple libraries
mcp_firecrawl_firecrawl_extract(
    urls=[
        "https://pypi.org/project/faiss-cpu/",
        "https://pypi.org/project/POT/",
        "https://pypi.org/project/duckdb/"
    ],
    prompt="Extract version information, dependencies, and license details",
    schema={
        "type": "object",
        "properties": {
            "library_name": {"type": "string"},
            "latest_version": {"type": "string"},
            "dependencies": {"type": "array", "items": {"type": "string"}},
            "license": {"type": "string"}
        }
    }
)
```

#### Researching Implementation Examples

```python
# Scrape tutorial websites
mcp_firecrawl_firecrawl_scrape(
    url="https://github.com/facebookresearch/faiss/wiki/Getting-started",
    formats=["markdown"],
    onlyMainContent=True
)

# Map documentation sites
mcp_firecrawl_firecrawl_map(
    url="https://duckdb.org/docs/",
    limit=100,
    search="Python examples"
)
```

### Best Practices

1. **Use specific URLs** when you know the exact page
2. **Set appropriate limits** to avoid overwhelming results
3. **Use structured extraction** for consistent data format
4. **Combine with search** to find relevant URLs first
5. **Respect website rate limits** by using appropriate delays
6. **Use markdown format** for better content readability
7. **Focus on main content** to avoid navigation clutter

---

## 4. GitMCP-Docs MCP Server

### Overview

The GitMCP-Docs MCP server provides access to GitHub repository documentation, enabling you to read and analyze code, documentation, and project structures directly from GitHub repositories.

### When to Use

- **Accessing GitHub repository documentation** (as specified in project rules)
- **Reading source code** from open-source projects
- **Analyzing project structures** and implementations
- **Finding examples** and usage patterns
- **Understanding library internals**

### Key Functions

#### Repository Documentation Access

```python
# Fetch documentation for a specific repository
mcp_gitmcp-docs_fetch_generic_documentation(
    owner="facebook",
    repo="faiss"
)

# Search for specific content in repository documentation
mcp_gitmcp-docs_search_generic_documentation(
    owner="facebook",
    repo="faiss",
    query="HNSW index implementation"
)
```

#### Code Search

```python
# Search for specific code patterns
mcp_gitmcp-docs_search_generic_code(
    owner="facebook",
    repo="faiss",
    query="IndexHNSWFlat class definition"
)

# Get file tree structure
mcp_gitmcp-docs_get_file_tree(
    owner="facebook",
    repo="faiss",
    branch="main"
)
```

#### URL Content Fetching

```python
# Fetch content from absolute URLs found in documentation
mcp_gitmcp-docs_fetch_generic_url_content(
    url="https://github.com/facebookresearch/faiss/blob/main/INSTALL.md"
)
```

### Usage Examples from Chromatica Project

#### Researching FAISS Implementation

```python
# Get FAISS repository documentation
mcp_gitmcp-docs_fetch_generic_documentation(
    owner="facebookresearch",
    repo="faiss"
)

# Search for HNSW-specific code
mcp_gitmcp-docs_search_generic_code(
    owner="facebookresearch",
    repo="faiss",
    query="IndexHNSWFlat implementation"
)
```

#### Understanding Library Architecture

```python
# Get repository structure
mcp_gitmcp-docs_get_file_tree(
    owner="facebookresearch",
    repo="faiss",
    branch="main"
)

# Search for Python bindings
mcp_gitmcp-docs_search_generic_code(
    owner="facebookresearch",
    repo="faiss",
    query="Python wrapper IndexHNSW"
)
```

#### Finding Implementation Examples

```python
# Search for usage examples
mcp_gitmcp-docs_search_generic_documentation(
    owner="facebookresearch",
    repo="faiss",
    query="Python examples HNSW index"
)
```

### Best Practices

1. **Use specific owner/repo combinations** for targeted access
2. **Combine search with documentation fetch** for comprehensive understanding
3. **Use file tree** to understand project structure before diving into code
4. **Search for specific terms** rather than general queries
5. **Follow up with URL content** for detailed information

---

## 5. Context7 MCP Server

### Overview

The Context7 MCP server provides access to up-to-date library documentation and API references, enabling you to understand new libraries and their capabilities without leaving your development environment.

### When to Use

- **When encountering a new library** to get its documentation (as specified in project rules)
- **Understanding library APIs** and usage patterns
- **Finding function signatures** and parameter details
- **Learning about library features** and capabilities
- **Getting current documentation** for libraries

### Key Functions

#### Library Resolution

```python
# Resolve library name to Context7-compatible ID
mcp_context7_resolve-library-id(
    libraryName="faiss"
)

# Resolve specific library
mcp_context7_resolve-library-id(
    libraryName="opencv-python"
)
```

#### Documentation Retrieval

```python
# Get comprehensive library documentation
mcp_context7_get-library-docs(
    context7CompatibleLibraryID="/facebook/faiss",
    topic="indexing",
    tokens=10000
)

# Get specific topic documentation
mcp_context7_get-library-docs(
    context7CompatibleLibraryID="/opencv/opencv",
    topic="image processing",
    tokens=5000
)
```

### Usage Examples from Chromatica Project

#### Understanding FAISS Library

```python
# Resolve FAISS library
mcp_context7_resolve-library-id(libraryName="faiss")

# Get FAISS documentation
mcp_context7_get-library-docs(
    context7CompatibleLibraryID="/facebook/faiss",
    topic="HNSW index",
    tokens=8000
)
```

#### Learning OpenCV Capabilities

```python
# Resolve OpenCV library
mcp_context7_resolve-library-id(libraryName="opencv-python")

# Get image processing documentation
mcp_context7_get-library-docs(
    context7CompatibleLibraryID="/opencv/opencv",
    topic="color space conversion",
    tokens=6000
)
```

#### Understanding NumPy Functions

```python
# Resolve NumPy library
mcp_context7_resolve-library-id(libraryName="numpy")

# Get histogram functions documentation
mcp_context7_get-library-docs(
    context7CompatibleLibraryID="/numpy/numpy",
    topic="histogram functions",
    tokens=4000
)
```

### Best Practices

1. **Always resolve library ID first** before getting documentation
2. **Use specific topics** to focus documentation retrieval
3. **Set appropriate token limits** based on detail needed
4. **Combine with other MCP servers** for comprehensive research
5. **Use for initial library exploration** before diving into code

---

## 6. GitMVP MCP Server

### Overview

The GitMVP MCP server provides comprehensive GitHub repository analysis capabilities, enabling you to understand repository structures, read files, and analyze code directly from GitHub.

### When to Use

- **When needing to understand the current state of the repository** (as specified in project rules)
- **Analyzing repository structures** and file organization
- **Reading specific files** from GitHub repositories
- **Understanding code implementations** and patterns
- **Estimating project scope** and complexity

### Key Functions

#### Repository Search

```python
# Search for repositories by query
mcp_gitmvp_search_repositories(
    query="color search engine Python",
    per_page=20,
    sort="stars"
)

# Search for specific libraries
mcp_gitmvp_search_repositories(
    query="language:python faiss",
    per_page=10,
    sort="updated"
)
```

#### Repository Structure Analysis

```python
# Get file tree structure
mcp_gitmvp_get_file_tree(
    owner="facebookresearch",
    repo="faiss",
    branch="main",
    format="tree"
)

# Get detailed file metadata
mcp_gitmvp_get_file_tree(
    owner="facebookresearch",
    repo="faiss",
    branch="main",
    format="json"
)
```

#### File Reading and Analysis

```python
# Read specific files
mcp_gitmvp_read_repository(
    owner="facebookresearch",
    repo="faiss",
    path="python/faiss/__init__.py"
)

# Read multiple files
mcp_gitmvp_read_repository(
    owner="facebookresearch",
    repo="faiss",
    path=["python/faiss/__init__.py", "python/faiss/index.py"]
)
```

#### Project Scope Estimation

```python
# Estimate token count for repository analysis
mcp_gitmvp_get_estimated_tokens(
    owner="facebookresearch",
    repo="faiss",
    max_files=100
)
```

### Usage Examples from Chromatica Project

#### Understanding FAISS Repository

```python
# Get FAISS repository structure
mcp_gitmvp_get_file_tree(
    owner="facebookresearch",
    repo="faiss",
    branch="main",
    format="tree"
)

# Read Python wrapper implementation
mcp_gitmvp_read_repository(
    owner="facebookresearch",
    repo="faiss",
    path="python/faiss/IndexHNSW.py"
)
```

#### Researching Similar Projects

```python
# Search for color search engine projects
mcp_gitmvp_search_repositories(
    query="color histogram search engine",
    per_page=15,
    sort="stars"
)

# Analyze promising repositories
mcp_gitmvp_get_file_tree(
    owner="promising_owner",
    repo="promising_repo",
    format="tree"
)
```

#### Learning Implementation Patterns

```python
# Read multiple implementation files
mcp_gitmvp_read_repository(
    owner="facebookresearch",
    repo="faiss",
    path=[
        "python/faiss/__init__.py",
        "python/faiss/index.py",
        "python/faiss/swigfaiss.py"
    ]
)
```

### Best Practices

1. **Use specific search queries** to find relevant repositories
2. **Start with file tree** to understand repository structure
3. **Read key files** to understand implementation patterns
4. **Estimate scope** before deep-diving into large repositories
5. **Combine with other MCP servers** for comprehensive analysis

---

## Integration and Workflow Examples

### Complete Research Workflow

#### 1. Initial Library Research

```python
# Step 1: Search for library information
mcp_brave_search_brave_web_search(
    query="FAISS HNSW index Python performance",
    count=10
)

# Step 2: Get library documentation
mcp_context7_resolve-library-id(libraryName="faiss")
mcp_context7_get-library-docs(
    context7CompatibleLibraryID="/facebook/faiss",
    topic="HNSW index",
    tokens=8000
)

# Step 3: Analyze repository structure
mcp_gitmvp_get_file_tree(
    owner="facebookresearch",
    repo="faiss",
    format="tree"
)
```

#### 2. Implementation Research

```python
# Step 1: Map documentation website
mcp_firecrawl_firecrawl_map(
    url="https://faiss.ai/",
    limit=100,
    search="HNSW implementation"
)

# Step 2: Scrape specific pages
mcp_firecrawl_firecrawl_scrape(
    url="https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSW.html",
    formats=["markdown"]
)

# Step 3: Read source code
mcp_gitmvp_read_repository(
    owner="facebookresearch",
    repo="faiss",
    path="python/faiss/IndexHNSW.py"
)
```

#### 3. Project Implementation

```python
# Step 1: Create project structure
mcp_filesystem_create_directory(path="src/chromatica/indexing/")

# Step 2: Write implementation files
mcp_filesystem_write_file(
    path="src/chromatica/indexing/store.py",
    content="# FAISS HNSW index implementation"
)

# Step 3: Update project files
mcp_filesystem_edit_file(
    path="requirements.txt",
    edits=[{"oldText": "# Add new dependency", "newText": "faiss-cpu>=1.7.4"}]
)
```

### Best Practices for Integration

1. **Start with search** to get overview information
2. **Use documentation servers** for API understanding
3. **Analyze repositories** for implementation patterns
4. **Scrape detailed content** for comprehensive understanding
5. **Implement systematically** using filesystem tools
6. **Document decisions** and research findings

---

## Conclusion

This MCP Server Usage Guide provides comprehensive coverage of all the MCP servers available in the Chromatica project. By following the guidelines and examples provided, you can:

- **Efficiently research** libraries and technologies
- **Understand implementations** from open-source projects
- **Stay current** with latest developments
- **Implement features** systematically and correctly
- **Maintain code quality** through proper research and validation

### Key Takeaways

1. **Use the right tool for the job** - Each MCP server has specific strengths
2. **Follow project rules** - Always adhere to Chromatica project guidelines
3. **Combine tools effectively** - Use multiple servers for comprehensive research
4. **Document your process** - Keep track of research findings and decisions
5. **Validate implementations** - Use testing tools to ensure correctness

### Next Steps

- **Practice using each MCP server** with the examples provided
- **Develop your own workflows** based on project needs
- **Contribute to the guide** with additional examples and best practices
- **Stay updated** with new MCP server capabilities

For more information about the Chromatica project and its development guidelines, refer to the `docs/.cursor/critical_instructions.md` file and the `.cursorrules` configuration.
