# 🏛️ Chromatica Refactoring - Critical Instructions

## 1. The Golden Rule: No Code Deletion

**THIS IS THE MOST IMPORTANT RULE. DO NOT BREAK IT.**

Your primary objective is **refactoring**, not rewriting. You must **NEVER** remove, omit, truncate, replace, or delete *any* existing, working code from the project.

Your task is to **MOVE** code from large, monolithic files into new, smaller, modular files and update the imports/exports to ensure 100% of the original functionality is preserved.

**Definition of "Move":**
1.  **COPY** the target code (a function, a class, an HTML block) from the source file.
2.  **PASTE** it into the new destination file.
3.  **ADD** the necessary `export` statement in the new file (for JS) or ensure it's importable (for Python).
4.  **ADD** the necessary `import` statement in the original file (or the new entry point, like `app.js` or `main.py`).
5.  **DELETE** the original code from the source file *only after* you have confirmed it is correctly imported and working.

## 2. The Refactoring Mandate

The goal is to make the codebase modular and manageable for future AI-assisted development. Adhere strictly to the 3-Phase Plan.

* **Python:** Use `APIRouter` to split endpoints. Move utilities and models to separate files.
* **JavaScript:** Use ES6 modules (`import`/`export`). Split logic by feature (UI, API, utils).
* **HTML:** Use the provided JavaScript-based `loadHTMLComponents` function to split HTML into templates.

## 3. The 3-Phase Plan & Progress Tracker

You must follow this plan step-by-step. Do not skip phases. Mark tasks as complete only when they are fully verified.

### Phase 1: Python (FastAPI) Modularization

-   [ ] **Utils:** Create `src/chromatica/utils/logging_config.py` and move `setup_logging`.
-   [ ] **Utils:** Create `src/chromatica/utils/color_utils.py` and move `extract_dominant_colors` and `extract_dominant_colors_with_weights`.
-   [ ] **Models:** Create `src/chromatica/api/models.py` and move all Pydantic `BaseModel` classes from `main.py`.
-   [ ] **Router:** Create `src/chromatica/api/routers/search.py` and move `search_images`, `parallel_search`, and `perform_search_async`.
-   [ ] **Router:** Create `src/chromatica/api/routers/image.py` and move `get_image` and `extract_colors_from_image`.
-   [ ] **Router:** Create `src/chromatica/api/routers/system.py` and move `health_check`, `api_info`, `get_performance_stats`, and `restart_server`.
-   [ ] **Router:** Create `src/chromatica/api/routers/visualize.py` and move `visualize_query` and `visualize_results`.
-   [ ] **Router:** Create `src/chromatica/api/routers/dev.py` and move `execute_command`.
-   [ ] **Main App:** Refactor `src/chromatica/api/main.py` to be a minimal app creator that imports and includes all routers.
-   [ ] **Verification:** Run the FastAPI server and test all endpoints to confirm 100% functionality.

### Phase 2: JavaScript (ES6) Modularization

-   [ ] **HTML Update:** Change `<script src="...main.js" ...>` to `<script type="module" src="/static/js/app.js"></script>` in `index.html`.
-   [ ] **Module:** Create `js/utils.js` and move all helper/color conversion functions.
-   [ ] **Module:** Create `js/api.js` and move all `fetch` calls.
-   [ ] **Module:** Create `js/ui/state.js` and move global state variables (`colors`, `weights`, `lastSearchResults`, etc.).
-   [ ] **Module:** Create `js/ui/colorPicker.js` and move color input panel logic (`addColor`, `updateColorPalette`, `rollDice`, etc.).
-   [ ] **Module:** Create `js/ui/colorSuggestions.js` and move palette/suggestion logic.
-   [ ] **Module:** Create `js/ui/results.js` and move search result display logic (`updateSearchResults`, `showImageDetails`, etc.).
-   [ ] **Module:** Create `js/ui/visualization2D.js` and move 2D chart/collage logic.
-   [ ] **Module:** Create `js/ui/stats.js` and move `updateSearchStats` and `playSearchCompleteSound`.
-   [ ] **Module:** Create `js/ui/modals.js` and move all modal (upload, info, etc.) logic.
-   [ ] **Module:** Create `js/ui/devTools.js` and move dev tool UI functions.
-   [ ] **Module:** Create `js/3d/visualizationManager.js` and move 3D module loader and wrappers.
-   [ ] **Entry Point:** Create `js/app.js`. Import all functions from new modules and attach necessary event handlers to the `window` object.
-   [ ] **Cleanup:** Delete `js/main.js` (after verification).
-   [ ] **Verification:** Load the web page and test *all* UI functionality (adding colors, searching, modals, 3D viz) to confirm 100% functionality.

### Phase 3: HTML (Component) Modularization

-   [ ] **Loader:** Create `js/ui/htmlLoader.js` with the `loadHTMLComponents` function.
-   [ ] **Update `app.js`:** Import and call `loadHTMLComponents` on `DOMContentLoaded`.
-   [ ] **Templates:** Create `static/templates/modals/` directory.
-   [ ] **Move Modals:** Move HTML for each modal from `index.html` to its own file in `static/templates/modals/`.
-   [ ] **Templates:** Create `static/templates/toolPanels/` directory.
-   [ ] **Move Panels:** Move HTML for all 6 tool panels from `index.html` to their own files.
-   [ ] **Templates:** Create `static/templates/sections/` directory.
-   [ ] **Move Sections:** Move HTML for `testingTools` and `visualization3D` sections to their own files.
-   [ ] **Refactor `index.html`:** Replace all moved HTML blocks with their corresponding `<div data-include="..."></div>` tags.
-   [ ] **Verification:** Load the web page and confirm all modals, tool panels, and sections load and function correctly.

## 4. General Rules

* **Verify After Each Step:** After moving a file or a block of code, immediately test the application. Do not batch large, unverified changes.
* **Imports are Key:** Pay close attention to relative imports (e.g., `from ..utils import ...` vs `from .models import ...`).
* **Use Git:** Commit your changes after *each* successful step (e.g., "Refactor: Move Pydantic models to api/models.py").


