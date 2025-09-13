# Troubleshooting Guide

This document provides solutions to common issues encountered while using the Chromatica color search engine.

---

## Common Issues and Solutions

### ❌ Search System Not Ready - "Search components are not initialized"

**Symptoms**:

- API returns 503 error with message "Search system is not available"
- Web interface shows "❌ System Not Ready" status
- Error message: "Search components are not initialized"

**Root Cause**:
The FAISS index and DuckDB metadata store failed to load during API startup, usually due to:

1. Index files not found or corrupted
2. Incorrect file paths
3. Permission issues
4. Code bugs in the initialization process

**Solutions**:

#### Solution 1: Rebuild the Index

If the index files are missing or corrupted:

```bash
# Activate virtual environment
venv311\Scripts\activate

# Rebuild index with test dataset
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index
```

#### Solution 2: Check File Paths

Ensure the index files exist in the expected location:

```bash
# Check if index files exist
ls test_index/
# Should show: chromatica_index.faiss, chromatica_metadata.db
```

#### Solution 3: Verify File Permissions

Ensure the API process has read access to the index files:

```bash
# Check file permissions
dir test_index\
```

#### Solution 4: Check API Logs

Look for initialization errors in the API startup logs:

```bash
# Start API with verbose logging
python -m uvicorn src.chromatica.api.main:app --reload --log-level debug
```

**Prevention**:

- Always run `build_index.py` after making changes to the indexing pipeline
- Use the test datasets for development and validation
- Monitor API startup logs for initialization errors
- Ensure all dependencies are properly installed in the virtual environment

**Status**: ✅ RESOLVED - Fixed initialization bug in API startup code

### ❌ Font Files Not Loading (404 Errors)

**Symptoms**:

- Browser console shows 404 errors for font files
- Web interface uses fallback fonts instead of custom fonts
- Error messages: "GET /fonts/[filename] HTTP/1.1" 404 Not Found

**Root Cause**:
The web interface was trying to access fonts at `/fonts/` but FastAPI serves static files at `/static/`. This caused a path mismatch between:

- CSS font references: `fonts/[filename]`
- Actual static file serving: `/static/fonts/[filename]`

**Solutions**:

#### Solution 1: Update Font Paths (IMPLEMENTED)

All font paths in the CSS have been updated to use absolute paths starting with `/static/fonts/`:

```css
@font-face {
  font-family: "JetBrainsMono Nerd Font Mono";
  src: url("/static/fonts/JetBrainsMonoNerdFontMono-Regular.ttf") format("truetype");
  font-weight: 400;
  font-style: normal;
}
```

#### Solution 2: Verify Font File Structure

Ensure font files are in the correct directory:

```bash
# Check font directory structure
ls src/chromatica/api/static/fonts/
# Should show all required font files
```

#### Solution 3: Test Font Loading

After applying the fix, verify fonts load correctly:

1. Open browser dev tools (F12)
2. Go to Network tab
3. Refresh the page
4. Look for successful font requests (200 OK)

**Prevention**:

- Always use absolute paths starting with `/static/` for static file references
- Test font loading after making changes to static file serving
- Monitor browser console for 404 errors during development

**Status**: ✅ RESOLVED - Updated all font paths to use correct `/static/fonts/` URLs

### ❌ Favicon Not Found (404 Error)

**Symptoms**:

- Browser console shows 404 error for favicon.ico
- Error message: "GET /favicon.ico HTTP/1.1" 404 Not Found
- Browser tab shows default icon instead of custom favicon

**Root Cause**:
The web interface was missing a favicon file and had no favicon links in the HTML head section. Browsers automatically request `/favicon.ico` by default.

**Solutions**:

#### Solution 1: Add Comprehensive Favicon Support (IMPLEMENTED)

Added multiple favicon links using inline SVG data URIs with Catppuccin Mocha theme colors:

```html
<link
  rel="icon"
  href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' fill='%23cba6f7'/><circle cx='30' cy='30' r='15' fill='%23f5c2e7'/><circle cx='70' cy='30' r='15' fill='%23f2cdcd'/><circle cx='30' cy='70' r='15' fill='%23a6e3a1'/><circle cx='70' cy='70' r='15' fill='%23f9e2af'/></svg>"
/>
<link rel="icon" type="image/svg+xml" href="..." />
<link rel="shortcut icon" href="..." />
<link rel="apple-touch-icon" href="..." />
```

#### Solution 2: Verify Favicon Display

After applying the fix, verify favicon works correctly:

1. Refresh the web page
2. Check browser tab for custom favicon
3. Look for successful favicon loading in Network tab
4. No more 404 errors for favicon.ico

**Prevention**:

- Always include comprehensive favicon support for all devices
- Use inline SVG data URIs for immediate favicon display
- Test favicon loading across different browsers and devices
- Monitor browser console for favicon-related errors

**Status**: ✅ RESOLVED - Added comprehensive favicon support with Catppuccin Mocha theme colors

### ❌ Results Collage Generation Failed

**Symptoms**:

- Error message: "Failed to generate results collage"
- Results collage endpoint returns 500 error
- Web interface shows collage generation failure

**Root Cause**:
The results collage generation was failing due to:

1. Missing `get_image_info` method in MetadataStore class
2. Inability to retrieve image file paths for collage generation
3. Incomplete image data handling in search results

**Solutions**:

#### Solution 1: Check API Logs

Look for specific error messages in the API logs:

```bash
# Start API with verbose logging
python -m uvicorn src.chromatica.api.main:app --reload --log-level debug
```

#### Solution 2: Verify Image Paths

Ensure that image files referenced in the database actually exist:

```bash
# Check if image files exist
dir datasets\test-dataset-20\
```

#### Solution 3: Rebuild Index

If image paths are incorrect, rebuild the index:

```bash
# Activate virtual environment
venv311\Scripts\activate

# Rebuild index
python scripts/build_index.py datasets/test-dataset-20 --output-dir test_index
```

**Prevention**:

- Always use absolute paths when building the index
- Ensure image files exist before indexing
- Monitor API logs for collage generation errors
- Test visualization endpoints after index rebuilds

**Status**: ✅ RESOLVED - Added missing `get_image_info` method and enhanced image data handling
