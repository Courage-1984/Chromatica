# Chromatica JavaScript Architecture

## Overview
The JavaScript codebase has been modularized to improve maintainability, readability, and code organization. Each module has a specific responsibility and is imported into the main entry point (`main.js`).

## Directory Structure
```
static/js/
├── main.js              # Main entry point
└── modules/
    ├── colorManagement.js  # Color handling and palette management
    ├── uiUtils.js         # UI utility functions
    ├── search.js          # Search functionality
    ├── results.js         # Search results handling
    ├── visualization.js   # Visualization features
    └── collage.js        # Collage generation and handling
```

## Module Descriptions

### main.js
The main entry point that imports all modules and sets up the global scope. It:
- Imports all required modules
- Exposes necessary functions to the global scope
- Initializes event listeners
- Manages the initial application state

### colorManagement.js
Handles all color-related functionality:
- Color addition/removal
- Color weight management
- Color palette visualization
- Color input event handling
- Global color state management

### uiUtils.js
Contains utility functions for UI interactions:
- Error message display
- Success message display
- Modal handling
- Image detail display
- Common UI helper functions

### search.js
Manages search functionality:
- Search execution
- Search parameter building
- Results processing
- Search statistics
- Error handling
- Search state management

### results.js
Handles search result display and interaction:
- Result grid generation
- Result card creation
- Result visualization
- Result metadata display
- Result interaction handlers

### visualization.js
Contains visualization-related functionality:
- Query visualization
- Results preview
- Histogram generation
- Visualization updates
- Data representation helpers

### collage.js
Manages collage creation and handling:
- Collage generation
- Layout management
- Collage downloading
- Image arrangement
- Collage customization

## Usage
The modules use ES6 module syntax. To use them, import the main.js file with type="module":

```html
<script type="module" src="/static/js/main.js"></script>
```

### Server Configuration
To serve ES modules correctly, you need to configure your server to use the proper MIME type. Here's how to do it for different servers:

#### FastAPI (Current Setup)
In your FastAPI application, ensure static files are served with the correct MIME type:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

# Optional: Explicitly set MIME type for .js files
@app.get("/static/js/{file_path:path}")
async def serve_js(file_path: str):
    return FileResponse(
        f"static/js/{file_path}",
        media_type="application/javascript"
    )
```

#### Nginx
Add this to your nginx configuration:

```nginx
types {
    application/javascript  js mjs;
}

# Or in the location block:
location ~ \.js$ {
    default_type application/javascript;
}
```

#### Apache
Add this to your .htaccess file or Apache configuration:

```apache
AddType application/javascript .js
AddType application/javascript .mjs
```

#### Express.js
For Express.js servers:

```javascript
express.static('static', {
    setHeaders: (res, path) => {
        if (path.endsWith('.js')) {
            res.set('Content-Type', 'application/javascript');
        }
    }
});
```

You can verify the MIME type is correct by:
1. Opening Chrome DevTools (F12)
2. Going to the Network tab
3. Clicking on any .js file
4. Checking that the "Content-Type" header is "application/javascript"

## Global Functions
The following functions are exposed to the global scope through `main.js`:

### Color Management
- `window.addColor()`: Add a new color input
- `window.removeColor()`: Remove a color input
- `window.updateColors()`: Update color array
- `window.updateWeights()`: Update weight array
- `window.updateColorPalette()`: Update color palette visualization

### UI Utils
- `window.showError()`: Display error message
- `window.showSuccess()`: Display success message
- `window.showImageInModal()`: Show image in modal

### Search
- `window.performSearch()`: Execute search
- `window.updateSearchResults()`: Update results display
- `window.updateVisualization()`: Update visualization
- `window.generateHistogramVisualization()`: Generate histograms

### Collage
- `window.generateResultsCollage()`: Generate results collage
- `window.downloadCollage()`: Download generated collage

## State Management
Global state is managed through window properties:
- `window.colors`: Array of current colors
- `window.weights`: Array of current weights
- `window.lastSearchResults`: Latest search results

## Event Handling
Event listeners are initialized in `main.js` for:
- Color palette updates
- Modal interactions
- Result card hover effects
- Image interactions

## Dependencies
- html2canvas.js (for collage generation and download)
- Three.js (for 3D visualizations)

## Best Practices
1. Avoid direct DOM manipulation in modules when possible
2. Use the provided UI utility functions for consistent messaging
3. Maintain proper error handling throughout
4. Keep modules focused on their specific responsibilities
5. Use ES6 module syntax for imports/exports
6. Document any new functions or changes

## Notes
- The codebase uses the Catppuccin Mocha color theme variables
- UI components follow a consistent style guide
- Error handling includes both user feedback and console logging
- Visualization features are designed to be extensible
