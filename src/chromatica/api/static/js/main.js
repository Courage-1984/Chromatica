// Chromatica Color Search Engine - Shared JavaScript Functions

console.log('Loading Chromatica JavaScript...');

// Global variables
window.colors = ['#FF0000'];
window.weights = [100];

// Explicitly expose functions to global scope
window.addColor = function () {
    console.log('addColor called');
    // Call the helper function to add a new row with default values
    addColorRow('#00FF00', 100);
    window.updateColorSuggestions();
};

// Update colors, weights, and color palette
window.updateColors = function () {
    console.log('Updating colors');
    // Try multiple selectors to find all color pickers (including hidden ones)
    let colorPickers = document.querySelectorAll('input.color-picker[type="color"]');
    if (colorPickers.length === 0) {
        colorPickers = document.querySelectorAll('.color-picker');
    }
    if (colorPickers.length === 0) {
        // Fallback: find all color inputs
        colorPickers = document.querySelectorAll('input[type="color"]');
    }
    window.colors = Array.from(colorPickers).map(picker => {
        const value = picker.value;
        // Ensure hex format with #
        if (value && value !== '') {
            return value.startsWith('#') ? value : '#' + value;
        }
        return null;
    }).filter(color => color && color !== '#' && color !== '#null');
    console.log('Colors updated:', window.colors);
    console.log('Found', colorPickers.length, 'color pickers,', window.colors.length, 'valid colors');
};

window.updateWeights = function () {
    console.log('Updating weights');
    const weightSliders = document.querySelectorAll('.weight-slider');
    window.weights = Array.from(weightSliders).map(slider => parseInt(slider.value));
    console.log('Weights updated:', window.weights);
};

window.updateColorPalette = function () {
    console.log('Updating color palette');
    window.updateColors();
    window.updateWeights();

    // Update color suggestions whenever the palette changes
    window.updateColorSuggestions();

    // Create a color palette visualization if there's a container for it
    const paletteContainer = document.getElementById('colorPalette');
    if (!paletteContainer) return;

    // Clear existing content
    paletteContainer.innerHTML = '';

    // Create color bars with relative widths according to weights
    const totalWeight = window.weights.reduce((sum, w) => sum + w, 0) || 1;
    const colorBar = document.createElement('div');
    colorBar.style.display = 'flex';
    colorBar.style.height = '40px';
    colorBar.style.width = '100%';
    colorBar.style.borderRadius = '8px';
    colorBar.style.overflow = 'hidden';
    colorBar.style.marginBottom = '10px';

    window.colors.forEach((color, index) => {
        const width = (window.weights[index] / totalWeight) * 100;
        const colorSegment = document.createElement('div');
        colorSegment.style.backgroundColor = color;
        colorSegment.style.width = `${width}%`;
        colorSegment.style.height = '100%';
        colorBar.appendChild(colorSegment);
    });

    paletteContainer.appendChild(colorBar);

    // Add individual color swatches with percentage labels
    const swatchesContainer = document.createElement('div');
    swatchesContainer.style.display = 'flex';
    swatchesContainer.style.flexWrap = 'wrap';
    swatchesContainer.style.gap = '10px';
    swatchesContainer.style.justifyContent = 'center';

    window.colors.forEach((color, index) => {
        const percentage = Math.round((window.weights[index] / totalWeight) * 100);

        const swatch = document.createElement('div');
        swatch.style.display = 'flex';
        swatch.style.flexDirection = 'column';
        swatch.style.alignItems = 'center';

        const colorBox = document.createElement('div');
        colorBox.style.width = '30px';
        colorBox.style.height = '30px';
        colorBox.style.backgroundColor = color;
        colorBox.style.border = '1px solid var(--surface2)';
        colorBox.style.borderRadius = '4px';
        colorBox.style.marginBottom = '5px';

        const percentageText = document.createElement('span');
        percentageText.textContent = `${percentage}%`;
        percentageText.style.fontSize = '12px';
        percentageText.style.color = 'var(--text)';

        swatch.appendChild(colorBox);
        swatch.appendChild(percentageText);
        swatchesContainer.appendChild(swatch);
    });

    paletteContainer.appendChild(swatchesContainer);
};

// Function to remove a color
window.removeColor = function (buttonElement) {
    const colorRow = buttonElement.closest('.color-row');
    if (!colorRow) return;

    // Only allow removal if there are more than one color
    const colorRows = document.querySelectorAll('.color-row');
    if (colorRows.length <= 1) {
        window.showError('Error', 'You must have at least one color.');
        return;
    }

    colorRow.remove();
    window.updateColorPalette();
};

// Add error/success helper functions
window.showError = function (title, message) {
    console.error(`${title}: ${message}`);
    const error = document.getElementById('error');
    if (error) {
        error.innerHTML = `<h3>${title}</h3><p>${message}</p>`;
        error.style.display = 'block';
    }
};

window.showSuccess = function (title, message) {
    console.log(`${title}: ${message}`);
    const success = document.getElementById('success');
    if (success) {
        success.innerHTML = `<h3>${title}</h3><p>${message}</p>`;
        success.style.display = 'block';
    }
};

// Update the visualization function to implement your requests
window.updateVisualization = function (data) {
    console.log('Updating visualization:', data);

    // Update Query Visualization
    const queryViz = document.getElementById('query-viz');
    if (queryViz) {
        // Add padding to match the Histogram Analysis section
        queryViz.style.padding = "20px";

        const queryVizContent = document.createElement('div');
        queryVizContent.id = 'query-viz-content';
        queryViz.innerHTML = ''; // Clear existing content
        queryViz.appendChild(queryVizContent);

        const colors = data.query?.colors || [];
        const weights = data.query?.weights || [];

        // Create HTML for query visualization
        let queryHTML = '';

        if (colors.length > 0) {
            // Create color circles visualization
            queryHTML = `
                <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-bottom: 20px;">
            `;

            colors.forEach((color, index) => {
                const weightPercent = Math.round(weights[index] * 100);
                const size = 60 + (weightPercent * 0.6); // Size based on weight

                queryHTML += `
                    <div style="text-align: center;">
                        <div style="
                            width: ${size}px; 
                            height: ${size}px; 
                            border-radius: 50%; 
                            background-color: #${color};
                            margin: 0 auto 10px;
                            border: 2px solid var(--surface2);
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                        "></div>
                        <div style="color: var(--text); font-size: 14px; font-weight: 600; font-family: 'JetBrainsMono Nerd Font Mono', monospace;">
                            #${color}
                        </div>
                        <div style="color: var(--subtext1); font-size: 12px;">
                            ${weightPercent}%
                        </div>
                    </div>
                `;
            });

            queryHTML += `</div>`;

            // Add color bar visualization
            queryHTML += `
                <div style="height: 40px; border-radius: 8px; overflow: hidden; margin: 20px 0; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
            `;

            let currentPosition = 0;
            colors.forEach((color, index) => {
                const width = weights[index] * 100;
                queryHTML += `
                    <div style="
                        height: 100%; 
                        width: ${width}%; 
                        background-color: #${color}; 
                        display: inline-block;
                        float: left;
                    "></div>
                `;
                currentPosition += width;
            });

            queryHTML += `</div>`;

            // Add LAB color space visualization
            queryHTML += `
                <div style="margin-top: 20px;">
                    <h4 style="color: var(--text); margin-bottom: 10px; font-family: 'JetBrainsMono Nerd Font Mono', monospace; text-align: center;">
                        LAB Color Space Visualization
                    </h4>
                    <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 15px;">
            `;

            // Convert hex colors to LAB and create visualization
            colors.forEach((color, index) => {
                // Calculate approximate LAB values (simplified conversion)
                // In a real implementation, you would use a proper RGB to LAB conversion
                const r = parseInt(color.substring(0, 2), 16) / 255;
                const g = parseInt(color.substring(2, 4), 16) / 255;
                const b = parseInt(color.substring(4, 6), 16) / 255;

                // Simplified RGB to LAB conversion (approximation)
                const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                const a = (r - g) * 0.5;  // Simplified a* calculation
                const bb = (g - b) * 0.5; // Simplified b* calculation

                const l_scaled = Math.round(l * 100);
                const a_scaled = Math.round(a * 200 - 100);
                const b_scaled = Math.round(bb * 200 - 100);

                queryHTML += `
                    <div style="text-align: center; background: var(--surface0); padding: 10px; border-radius: 8px; width: 100px;">
                        <div style="
                            width: 50px;
                            height: 50px;
                            border-radius: 50%;
                            background-color: #${color};
                            margin: 0 auto 10px;
                            border: 2px solid var(--surface2);
                        "></div>
                        <div style="font-family: 'JetBrainsMono Nerd Font Mono', monospace; color: var(--text); font-size: 12px;">
                            <div>L: ${l_scaled}</div>
                            <div>a: ${a_scaled}</div>
                            <div>b: ${b_scaled}</div>
                        </div>
                    </div>
                `;
            });

            queryHTML += `
                    </div>
                </div>
            `;

            // Add LAB values if available from server
            if (data.query?.lab_values) {
                queryHTML += `
                    <div style="margin-top: 20px; text-align: center;">
                        <h4 style="color: var(--text); margin-bottom: 10px; font-family: 'JetBrainsMono Nerd Font Mono', monospace;">Precise LAB Color Values</h4>
                        <div style="display: flex; justify-content: center; gap: 20px;">
                `;

                data.query.lab_values.forEach((lab, index) => {
                    queryHTML += `
                        <div style="text-align: center;">
                            <div style="color: var(--text); font-size: 12px; font-family: 'JetBrainsMono Nerd Font Mono', monospace;">
                                L: ${lab.L?.toFixed(2) || 'N/A'}<br>
                                a: ${lab.a?.toFixed(2) || 'N/A'}<br>
                                b: ${lab.b?.toFixed(2) || 'N/A'}
                            </div>
                        </div>
                    `;
                });

                queryHTML += `</div></div>`;
            }

            // Add 3D LAB color space representation (simplified visualization)
            queryHTML += `
                <div style="margin-top: 20px; text-align: center;">
                    <h4 style="color: var(--text); margin-bottom: 10px; font-family: 'JetBrainsMono Nerd Font Mono', monospace;">
                        3D LAB Space Position
                    </h4>
                    <div style="
                        position: relative;
                        width: 200px;
                        height: 200px;
                        margin: 0 auto;
                        border: 1px solid var(--surface2);
                        border-radius: 8px;
                        background: var(--surface0);
                        overflow: hidden;
                    ">
                        <!-- Simplified LAB 3D space axes -->
                        <div style="position: absolute; width: 1px; height: 100%; background: rgba(255,255,255,0.3); left: 50%; top: 0;"></div>
                        <div style="position: absolute; width: 100%; height: 1px; background: rgba(255,255,255,0.3); left: 0; top: 50%;"></div>
            `;

            // Position dots representing colors in 2D space (a simplified projection of LAB space)
            colors.forEach((color, index) => {
                // Simple positioning based on approx LAB values
                const r = parseInt(color.substring(0, 2), 16) / 255;
                const g = parseInt(color.substring(2, 4), 16) / 255;
                const b = parseInt(color.substring(4, 6), 16) / 255;

                // Simplified positioning
                const x = 50 + (r - g) * 50; // a* approximation (red-green)
                const y = 50 + (g - b) * 50; // b* approximation (yellow-blue)

                const dotSize = 10 + weights[index] * 20;

                queryHTML += `
                    <div style="
                        position: absolute;
                        width: ${dotSize}px;
                        height: ${dotSize}px;
                        border-radius: 50%;
                        background-color: #${color};
                        border: 2px solid var(--surface2);
                        left: ${x}%;
                        top: ${y}%;
                        transform: translate(-50%, -50%);
                        z-index: 2;
                    "></div>
                `;
            });

            queryHTML += `
                        <div style="position: absolute; bottom: 5px; right: 5px; font-size: 10px; color: var(--subtext1);">a*-b* plane</div>
                    </div>
                </div>
            `;

            queryVizContent.innerHTML = queryHTML;
        }
    }

    // Update Results Preview as a clean 3x3 grid without overlays
    const resultsViz = document.getElementById('results-viz');
    if (resultsViz && data.results && data.results.length > 0) {
        // Add padding to match the Histogram Analysis section
        resultsViz.style.padding = "25px";
        resultsViz.style.background = "var(--surface0)";
        resultsViz.style.borderRadius = "12px";
        resultsViz.style.marginBottom = "20px";

        // Create a grid of result thumbnails (3x3)
        let resultsHTML = `
            <h4 style="color: var(--text); margin-bottom: 20px; font-family: 'JetBrainsMono Nerd Font Mono', monospace; text-align: center;">
                Top 9 Color Matches
            </h4>
            <div id="previewGrid" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 20px;">
        `;

        // Take the top 9 results (for 3x3 grid)
        const topResults = data.results.slice(0, 9);

        topResults.forEach(result => {
            const imgSrc = result.image_url || `/image/${encodeURIComponent(result.image_id)}`;
            const distance = typeof result.distance === 'number' ? result.distance.toFixed(3) : 'N/A';

            resultsHTML += `
                <div class="preview-item" style="
                    position: relative;
                    aspect-ratio: 1 / 1;
                    overflow: hidden;
                    border-radius: 8px;
                    border: 1px solid var(--surface2);
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                    cursor: pointer;
                    background-color: transparent !important;
                " onclick="window.showImageInModal('${imgSrc}')">
                    <img 
                        src="${imgSrc}" 
                        style="width: 100%; height: 100%; object-fit: cover; object-position: center; display: block;" 
                        crossOrigin="anonymous"
                    />
                    <div style="
                        position: absolute;
                        bottom: 0;
                        background: rgba(0, 0, 0, 0.7);
                        color: #000000;
                        padding: 3px 5px;
                        font-size: 10px;
                        text-align: center;
                        font-family: 'JetBrainsMono Nerd Font Mono', monospace;
                    ">
                        d: ${distance}
                    </div>
                </div>
            `;
        });

        resultsHTML += `</div>`;

        // Add buttons: view more link and download button
        resultsHTML += `
            <div style="display: flex; justify-content: center; gap: 15px; margin-top: 20px;">
                <a href="#resultsSection" style="
                    display: inline-block;
                    background: var(--blue);
                    color: #000000;
                    padding: 8px 16px;
                    border-radius: 8px;
                    text-decoration: none;
                    font-family: 'JetBrainsMono Nerd Font Mono', monospace;
                    font-weight: 500;
                    font-size: 14px;
                ">View All Results</a>
                
                <button onclick="window.downloadPreviewGrid()" style="
                    display: inline-block;
                    background: var(--green);
                    color: #000000;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 8px;
                    text-decoration: none;
                    font-family: 'JetBrainsMono Nerd Font Mono', monospace;
                    font-weight: 500;
                    font-size: 14px;
                    cursor: pointer;
                ">Download Grid</button>
            </div>
        `;

        resultsViz.innerHTML = resultsHTML;
    }

    // Update histograms if implemented
    const queryHistogram = document.getElementById('query-histogram');
    const resultHistogram = document.getElementById('result-histogram');

    if (queryHistogram && data.query?.histogram) {
        // Simple visual representation of histogram
        queryHistogram.innerHTML = createSimpleHistogramVisualization(data.query.histogram, 'Query');
    }

    if (resultHistogram && data.results?.[0]?.histogram) {
        // Simple visual representation of first result histogram
        resultHistogram.innerHTML = createSimpleHistogramVisualization(data.results[0].histogram, 'Top Result');
    }

    // Update search stats
    updateSearchStats(data);

    // Show the visualization section and adjust layout
    const visualizationSection = document.getElementById('visualizationSection');
    if (visualizationSection) {
        visualizationSection.style.display = 'block';

        // Update the grid layout to be 50/50
        const vizGrid = visualizationSection.querySelector('.viz-grid');
        if (vizGrid) {
            vizGrid.style.display = 'grid';
            vizGrid.style.gridTemplateColumns = '1fr 1fr';
            vizGrid.style.gap = '20px';
        }
    }
};

// Function to play sound on search completion
function playSearchCompleteSound() {
    const audio = new Audio('/static/ding_sound.mp3');
    audio.volume = 0.5; // Set volume to 50%
    audio.play().catch(error => {
        console.warn('Could not play search complete sound:', error);
    });
};

// Create simple histogram visualization function
function createSimpleHistogramVisualization(histogram, title) {
    if (!histogram || !Array.isArray(histogram)) {
        return `<p style="text-align: center; color: var(--subtext0);">No histogram data available</p>`;
    }

    // Determine how many bars to show (max 32)
    const barCount = Math.min(32, histogram.length);
    const step = Math.ceil(histogram.length / barCount);

    let html = `<h5 style="text-align: center; color: var(--text); margin-bottom: 15px;">${title} Histogram</h5>`;
    html += `<div style="display: flex; align-items: flex-end; height: 150px; gap: 2px;">`;

    for (let i = 0; i < barCount; i++) {
        const index = i * step;
        const value = histogram[index] || 0;
        const height = Math.max(5, value * 150); // Ensure minimum height of 5px

        // Calculate hue based on position (0-360)
        const hue = Math.round((i / barCount) * 360);

        html += `
            <div style="
                flex-grow: 1;
                height: ${height}px;
                background: hsl(${hue}, 70%, 60%);
                border-radius: 3px 3px 0 0;
            "></div>
        `;
    }

    html += `</div>`;

    return html;
}

// Create a color distribution histogram visualization
function createColorDistributionHistogram(colors, options = {}) {
    if (!Array.isArray(colors) || colors.length === 0) {
        return '<div class="histogram-container">No color data available.</div>';
    }

    // Count occurrences of each color
    const colorCounts = {};
    colors.forEach(hex => {
        const color = hex.toLowerCase();
        colorCounts[color] = (colorCounts[color] || 0) + 1;
    });

    // Calculate total count for percentage calculation
    const totalCount = Object.values(colorCounts).reduce((sum, count) => sum + count, 0);

    // Sort colors by count descending, then by color value
    const sortedColors = Object.keys(colorCounts).sort((a, b) => {
        const countDiff = colorCounts[b] - colorCounts[a];
        return countDiff !== 0 ? countDiff : a.localeCompare(b);
    });

    // Get max count for scaling
    const maxCount = Math.max(...Object.values(colorCounts));

    // Build histogram bars
    let barsHtml = '';
    sortedColors.forEach(color => {
        const count = colorCounts[color];
        const percentage = ((count / totalCount) * 100).toFixed(1);
        const heightPercent = ((count / maxCount) * 100).toFixed(1);

        barsHtml += `
            <div class="histogram-bar" style="
                flex: 1;
                height: ${heightPercent}%;
                background: ${color};
                min-width: 4px;
                border-radius: 2px 2px 0 0;
                position: relative;
                transition: transform 0.2s ease;
            " title="${color} (${count} occurrences, ${percentage}%)"></div>
        `;
    });

    // Create the histogram container with bars
    return `
        <div class="histogram-container" style="
            padding: 10px;
            background: var(--surface0);
            border-radius: 8px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div class="histogram-bars" style="
                display: flex;
                align-items: flex-end;
                height: 100px;
                gap: 2px;
                background: var(--base);
                border-radius: 4px;
                padding: 4px;
            ">
                ${barsHtml}
            </div>
        </div>
    `;
}

// Color conversion helper function for histogram
function hexToHsl(hexColor) {
    if (!hexColor) return null;

    // Remove # if present and convert to RGB
    const hex = hexColor.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16) / 255;
    const g = parseInt(hex.substring(2, 4), 16) / 255;
    const b = parseInt(hex.substring(4, 6), 16) / 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;

    if (max === min) {
        h = s = 0; // achromatic
    } else {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }

    return { h: h * 360, s: s * 100, l: l * 100 };
}

// Fix the histogram visualization function to use the defined function
window.generateHistogramVisualization = function () {
    console.log('Generating histogram visualization');

    // Check if we have search results
    if (!window.lastSearchResults || !Array.isArray(window.lastSearchResults) || window.lastSearchResults.length === 0) {
        window.showError('No Search Results', 'Please perform a search first to generate histogram visualizations.');
        return;
    }

    console.log('lastSearchResults:', window.lastSearchResults);

    const queryHistogram = document.getElementById('query-histogram');
    const resultHistogram = document.getElementById('result-histogram');

    console.log('DOM elements:', { queryHistogram, resultHistogram });
    console.log('Results exists:', window.lastSearchResults.length > 0);

    // Create color distribution histogram from dominant colors
    function createColorDistributionHistogram(colors) {
        // Create 32 bins for hue values (0-360 degrees)
        const histogram = new Array(32).fill(0);

        colors.forEach(color => {
            const hsl = hexToHsl(color);
            if (!hsl) return;
            // Map hue (0-360) to bin index (0-31)
            const binIndex = Math.floor((hsl.h / 360) * 32);
            // Weight by saturation and lightness to give more impact to vivid colors
            const weight = (hsl.s / 100) * (1 - Math.abs((hsl.l - 50) / 100));
            histogram[binIndex] += weight;
        });

        // Normalize histogram
        const sum = histogram.reduce((a, b) => a + b, 0);
        return histogram.map(v => v / (sum || 1));
    }

    if (queryHistogram) {
        const currentColors = window.colors || [];
        if (currentColors.length > 0) {
            const queryColorHist = createColorDistributionHistogram(currentColors);
            queryHistogram.innerHTML = createSimpleHistogramVisualization(queryColorHist, 'Query Colors');
            queryHistogram.style.display = 'block';
        } else {
            queryHistogram.innerHTML = '<p style="text-align: center; color: var(--subtext0);">No query colors available</p>';
            queryHistogram.style.display = 'block';
        }
    }

    if (resultHistogram) {
        const firstResult = window.lastSearchResults[0];
        if (firstResult?.dominant_colors && firstResult.dominant_colors.length > 0) {
            // Create histogram from dominant colors
            const resultColorHist = createColorDistributionHistogram(firstResult.dominant_colors);
            resultHistogram.innerHTML = createSimpleHistogramVisualization(resultColorHist, 'Result Colors');
            resultHistogram.style.display = 'block';
        } else {
            resultHistogram.innerHTML = '<p style="text-align: center; color: var(--subtext0);">No dominant colors available</p>';
            resultHistogram.style.display = 'block';
        }
    }

    // Show the histogram section if it exists
    const histogramSection = document.querySelector('.histogram-section');
    if (histogramSection) {
        histogramSection.style.display = 'block';
    }

    window.showSuccess('Histogram Visualization', 'Generated histogram visualization successfully.');
};

// Compare histograms between query and result
window.compareHistograms = function () {
    console.log('Comparing histograms - Start');

    // Get current search results and colors
    if (!window.lastSearchResults || !window.lastSearchResults.length) {
        window.showError('Comparison Error', 'No search results available');
        return;
    }
    console.log('Last search results:', window.lastSearchResults);

    // Get current colors from color inputs
    const colors = [];
    const colorInputs = document.querySelectorAll('.color-picker');
    colorInputs.forEach(input => colors.push(input.value));
    console.log('Current colors:', colors);

    if (!colors.length) {
        window.showError('Comparison Error', 'No query colors selected');
        return;
    }

    // Find the histogram section
    const histogramSection = document.querySelector('.histogram-section');
    if (!histogramSection) {
        window.showError('Comparison Error', 'Histogram section not found');
        return;
    }
    console.log('Found histogram section:', histogramSection);

    // Get query histogram element
    const queryHistogram = document.getElementById('query-histogram');
    if (!queryHistogram) {
        window.showError('Comparison Error', 'Query histogram element not found');
        return;
    }
    console.log('Query histogram element:', queryHistogram);

    // Generate query histogram
    console.log('Current colors for query histogram:', colors);
    console.log('Generating query histogram from colors');
    const queryHistogramHtml = createColorDistributionHistogram(colors);
    queryHistogram.innerHTML = queryHistogramHtml;
    console.log('Query color histogram generated:', queryHistogramHtml);

    // Get result histogram element
    const resultHistogram = document.getElementById('result-histogram');
    if (!resultHistogram) {
        window.showError('Comparison Error', 'Result histogram element not found');
        return;
    }
    console.log('Result histogram element:', resultHistogram);

    // Get first search result's dominant colors
    const firstResult = window.lastSearchResults[0];
    console.log('First search result:', firstResult);

    if (!firstResult || !firstResult.dominant_colors) {
        window.showError('Comparison Error', 'No dominant colors found in search result');
        return;
    }

    // Generate result histogram
    console.log('Generating result histogram from dominant colors:', firstResult.dominant_colors);
    const resultHistogramHtml = createColorDistributionHistogram(firstResult.dominant_colors);
    resultHistogram.innerHTML = resultHistogramHtml;
    console.log('Result color histogram generated:', resultHistogramHtml);

    // Get comparison container
    const comparisonContainer = document.getElementById('histogram-comparison');
    if (!comparisonContainer) {
        window.showError('Comparison Error', 'Comparison container not found');
        return;
    }
    console.log('Found comparison container');

    try {
        // Generate comparison data
        const histogramData = {
            queryData: colors,
            resultData: firstResult.dominant_colors
        };
        console.log('Generated histogram data:', histogramData);

        // Calculate color similarity metrics
        const metrics = calculateColorSimilarityMetrics(histogramData.queryData, histogramData.resultData);

        // Create comparison visualization
        const comparisonHtml = `
            <div class="comparison-metrics" style="
                margin: 20px 0;
                padding: 15px;
                background: var(--surface0);
                border-radius: 8px;
                border: 1px solid var(--surface2);
            ">
                <h4 style="margin: 0 0 10px 0; color: var(--text);">Color Distribution Analysis</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div class="metric-card">
                        <div class="metric-title">Color Count</div>
                        <div class="metric-value">Query: ${histogramData.queryData.length} / Result: ${histogramData.resultData.length}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Average Distance</div>
                        <div class="metric-value">${metrics.averageDistance.toFixed(2)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Similarity Score</div>
                        <div class="metric-value">${(metrics.similarityScore * 100).toFixed(1)}%</div>
                    </div>
                </div>
                <div style="margin-top: 15px; font-size: 14px; color: var(--subtext0);">
                    <strong>Analysis:</strong> ${metrics.analysis}
                </div>
            </div>
        `;

        comparisonContainer.innerHTML = comparisonHtml;
        window.showSuccess('Comparison Complete', 'Histogram comparison generated successfully');
    } catch (err) {
        console.error('Error during histogram comparison:', err);
        window.showError('Comparison Error', 'Error generating histogram comparison: ' + err.message);
    }
};

// Calculate color similarity metrics between two sets of colors
function calculateColorSimilarityMetrics(queryColors, resultColors) {
    // Convert colors to Lab color space for better comparison
    const queryLab = queryColors.map(color => {
        const rgb = hexToRgb(color);
        return rgbToLab(rgb.r, rgb.g, rgb.b);
    });

    const resultLab = resultColors.map(color => {
        const rgb = hexToRgb(color);
        return rgbToLab(rgb.r, rgb.g, rgb.b);
    });

    // Calculate average color distance
    let totalDistance = 0;
    let minDistance = Infinity;
    let maxDistance = 0;

    queryLab.forEach(queryColor => {
        resultLab.forEach(resultColor => {
            const distance = calculateDeltaE(queryColor, resultColor);
            totalDistance += distance;
            minDistance = Math.min(minDistance, distance);
            maxDistance = Math.max(maxDistance, distance);
        });
    });

    const averageDistance = totalDistance / (queryLab.length * resultLab.length);
    const similarityScore = Math.max(0, 1 - (averageDistance / 100));

    // Generate analysis text
    let analysis = '';
    if (similarityScore > 0.8) {
        analysis = 'Very high color similarity between query and result.';
    } else if (similarityScore > 0.6) {
        analysis = 'Good color similarity with some variations in distribution.';
    } else if (similarityScore > 0.4) {
        analysis = 'Moderate color similarity with significant differences.';
    } else {
        analysis = 'Low color similarity, suggesting distinct color distributions.';
    }

    return {
        averageDistance,
        similarityScore,
        minDistance,
        maxDistance,
        analysis
    };
}

// Generate a collage of search results
window.generateResultsCollage = function () {
    console.log('Generating results collage');

    // Check if we have search results
    if (!window.lastSearchResults || !Array.isArray(window.lastSearchResults) || window.lastSearchResults.length === 0) {
        window.showError('Collage Error', 'No search results available. Please perform a search first.');
        return;
    }

    // Find the collage section
    const collageSection = document.querySelector('.collage-section');
    if (!collageSection) {
        window.showError('Collage Error', 'Collage section not found in the DOM.');
        return;
    }

    // Preserve previously selected tile count if it exists
    const previousTileCount = (document.getElementById('tileCount')?.value) || window.currentCollageTileCount || '4';
    window.currentCollageTileCount = previousTileCount; // cache globally

    // Clear the entire collage section and rebuild it
    collageSection.innerHTML = '';

    // Recreate the header
    const header = document.createElement('h3');
    header.style.color = 'var(--text)';
    header.style.margin = '0 0 15px 0';
    header.style.fontFamily = "'JetBrainsMono Nerd Font Mono', monospace";
    header.textContent = '🖼️ Results Collage';
    collageSection.appendChild(header);

    // Recreate the controls
    const controls = document.createElement('div');
    controls.style.display = 'flex';
    controls.style.gap = '15px';
    controls.style.alignItems = 'center';
    controls.style.marginBottom = '20px';
    controls.style.flexWrap = 'wrap';

    // Tile count selector
    const tileCountGroup = document.createElement('div');
    tileCountGroup.style.display = 'flex';
    tileCountGroup.style.alignItems = 'center';
    tileCountGroup.style.gap = '10px';

    const tileCountLabel = document.createElement('label');
    tileCountLabel.setAttribute('for', 'tileCount');
    tileCountLabel.style.color = 'var(--subtext1)';
    tileCountLabel.style.fontSize = '14px';
    tileCountLabel.style.fontWeight = '500';
    tileCountLabel.textContent = 'Tiles per row:';
    tileCountGroup.appendChild(tileCountLabel);

    const tileCountSelect = document.createElement('select');
    tileCountSelect.id = 'tileCount';
    tileCountSelect.style.background = 'var(--base)';
    tileCountSelect.style.color = 'var(--text)';
    tileCountSelect.style.border = '1px solid var(--surface2)';
    tileCountSelect.style.borderRadius = '6px';
    tileCountSelect.style.padding = '8px 12px';
    tileCountSelect.style.fontSize = '14px';

    const optionsData = [
        { value: '3', text: '3x3 (9 tiles)' },
        { value: '4', text: '4x4 (16 tiles)' },
        { value: '5', text: '5x5 (25 tiles)' },
        { value: '6', text: '6x6 (36 tiles)' },
        { value: '8', text: '8x8 (64 tiles)' },
        { value: '10', text: '10x10 (100 tiles)' }
    ];

    optionsData.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.text;
        if (opt.value === previousTileCount) option.selected = true;
        tileCountSelect.appendChild(option);
    });

    tileCountSelect.addEventListener('change', () => {
        window.currentCollageTileCount = tileCountSelect.value; // update cached value
    });

    tileCountGroup.appendChild(tileCountSelect);
    controls.appendChild(tileCountGroup);

    // Generate Collage button
    const generateButton = document.createElement('button');
    generateButton.onclick = () => window.generateResultsCollage();
    generateButton.style.background = 'var(--mauve)';
    generateButton.style.color = '#000000';
    generateButton.style.border = 'none';
    generateButton.style.padding = '10px 20px';
    generateButton.style.borderRadius = '6px';
    generateButton.style.cursor = 'pointer';
    generateButton.style.fontSize = '14px';
    generateButton.style.fontWeight = '500';
    generateButton.style.transition = 'all 0.2s';
    generateButton.textContent = '🎨 Generate Collage';
    controls.appendChild(generateButton);

    // Download button
    const downloadButton = document.createElement('button');
    downloadButton.id = 'downloadCollageBtn';
    downloadButton.onclick = () => window.downloadCollage();
    downloadButton.style.background = 'var(--green)';
    downloadButton.style.color = '#000000';
    downloadButton.style.border = 'none';
    downloadButton.style.padding = '10px 20px';
    downloadButton.style.borderRadius = '6px';
    downloadButton.style.cursor = 'pointer';
    downloadButton.style.fontSize = '14px';
    downloadButton.style.fontWeight = '500';
    downloadButton.style.transition = 'all 0.2s';
    downloadButton.disabled = true;
    downloadButton.style.opacity = '0.5';
    downloadButton.textContent = 'Download';
    controls.appendChild(downloadButton);

    collageSection.appendChild(controls);

    // Create a new collage container
    const collageContainer = document.createElement('div');
    collageContainer.id = 'collageContainer';
    collageContainer.style.textAlign = 'center';
    collageContainer.style.minHeight = '200px';
    collageContainer.style.display = 'flex';
    collageContainer.style.alignItems = 'center';
    collageContainer.style.justifyContent = 'center';
    collageContainer.style.background = 'var(--base)';
    collageContainer.style.borderRadius = '8px';
    collageContainer.style.border = '1px solid var(--surface2)';
    collageSection.appendChild(collageContainer);

    // Get tile count from selector
    const selectedTileCount = parseInt(tileCountSelect.value);
    const resultCount = Math.min(window.lastSearchResults.length, selectedTileCount * selectedTileCount);
    const cols = selectedTileCount;
    const rows = Math.ceil(resultCount / cols);

    // Create a title for the collage
    const collageTitle = document.createElement('h4');
    collageTitle.textContent = `Color Search Results Collage (${resultCount} images)`;
    collageTitle.style.textAlign = 'center';
    collageTitle.style.margin = '0 0 15px 0';
    collageTitle.style.color = 'var(--text)';
    collageTitle.style.fontFamily = "'JetBrainsMono Nerd Font Mono', monospace";

    collageContainer.appendChild(collageTitle);

    // Create the collage grid
    const collageGrid = document.createElement('div');
    collageGrid.id = 'collageGrid';
    collageGrid.style.display = 'grid';
    collageGrid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    collageGrid.style.gap = '8px';
    collageGrid.style.padding = '15px';
    collageGrid.style.background = '#313244';
    collageGrid.style.borderRadius = '12px';
    collageGrid.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)';

    const results = window.lastSearchResults.slice(0, resultCount);

    results.forEach((result, index) => {
        const imgSrc = result.image_url || `/image/${encodeURIComponent(result.image_id)}`;
        const distance = typeof result.distance === 'number' ? result.distance.toFixed(3) : 'N/A';
        const imgContainer = document.createElement('div');
        imgContainer.style.position = 'relative';
        imgContainer.style.aspectRatio = '1 / 1';
        imgContainer.style.overflow = 'hidden';
        imgContainer.style.borderRadius = '8px';
        imgContainer.style.cursor = 'pointer';
        const img = document.createElement('img');
        img.src = imgSrc; img.style.width = '100%'; img.style.height = '100%'; img.style.objectFit = 'cover'; img.style.objectPosition = 'center'; img.style.transition = 'transform 0.3s ease'; img.crossOrigin = 'anonymous';
        const overlay = document.createElement('div');
        overlay.style.position = 'absolute'; overlay.style.inset = '0'; overlay.style.background = 'rgba(0,0,0,0.6)'; overlay.style.opacity = '0'; overlay.style.transition = 'opacity 0.3s ease'; overlay.style.display = 'flex'; overlay.style.alignItems = 'center'; overlay.style.justifyContent = 'center';
        if (index < 3) { const rankColors = ['var(--yellow)', 'var(--subtext0)', 'var(--peach)']; const rankEmoji = ['🥇', '🥈', '🥉']; const badge = document.createElement('div'); badge.style.position = 'absolute'; badge.style.top = '5px'; badge.style.left = '5px'; badge.style.backgroundColor = rankColors[index]; badge.style.color = '#000'; badge.style.padding = '3px 6px'; badge.style.borderRadius = '20px'; badge.style.fontSize = '12px'; badge.style.fontWeight = 'bold'; badge.style.zIndex = '2'; badge.innerHTML = `${rankEmoji[index]}${index + 1}`; imgContainer.appendChild(badge); }
        const distanceIndicator = document.createElement('div');
        distanceIndicator.style.position = 'absolute'; distanceIndicator.style.bottom = '0'; distanceIndicator.style.left = '0'; distanceIndicator.style.right = '0'; distanceIndicator.style.padding = '4px'; distanceIndicator.style.backgroundColor = 'var(--yellow)'; distanceIndicator.style.color = '#000000'; distanceIndicator.style.fontSize = '10px'; distanceIndicator.style.textAlign = 'center'; distanceIndicator.textContent = `d: ${distance}`;
        imgContainer.addEventListener('mouseover', () => { img.style.transform = 'scale(1.05)'; overlay.style.opacity = '0.3'; });
        imgContainer.addEventListener('mouseout', () => { img.style.transform = 'scale(1)'; overlay.style.opacity = '0'; });
        imgContainer.addEventListener('click', () => window.showImageInModal(imgSrc));
        imgContainer.onclick = function () {
            showImageDetails(result);
        };
        imgContainer.appendChild(img); imgContainer.appendChild(overlay); imgContainer.appendChild(distanceIndicator); collageGrid.appendChild(imgContainer);
    });

    collageContainer.appendChild(collageGrid);

    // Enable the download button
    downloadButton.disabled = false; downloadButton.style.opacity = '1';

    window.showSuccess('Collage Generated', `Results collage generated with ${resultCount} images.`);
};

// Function to download the generated collage
window.downloadCollage = function () {
    const collageGrid = document.getElementById('collageGrid');
    if (!collageGrid) {
        window.showError('Download Failed', 'No collage found. Please generate a collage first.');
        return;
    }

    // Use html2canvas to capture the collage
    html2canvas(collageGrid, {
        backgroundColor: '#313244', // Using hex instead of var(--surface0)
        scale: 2, // Higher quality
        logging: true, // Enable logging for debugging
        allowTaint: true,
        useCORS: true,
        imageTimeout: 15000, // Increase timeout for image loading
        onclone: function (clonedDoc) {
            console.log('HTML2Canvas cloned document for collage:', clonedDoc);
        }
    }).then(canvas => {
        // Add watermark text
        const ctx = canvas.getContext('2d');
        ctx.save();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.font = '16px JetBrainsMono Nerd Font Mono';
        ctx.textAlign = 'center';
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(-Math.PI / 8);
        ctx.fillText('Chromatica', 0, 0);
        ctx.restore();

        const link = document.createElement('a');
        link.download = `chromatica-collage-${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        window.showSuccess('Download Complete', 'Collage image downloaded successfully.');
    }).catch(err => {
        window.showError('Download Failed', err.message);
    });
};

// Update search statistics function to properly handle search time
function updateSearchStats(data) {
    // Get all statistics elements
    const totalResults = document.getElementById('totalResults');
    const searchTime = document.getElementById('searchTime');
    const avgDistance = document.getElementById('avgDistance');
    const searchMode = document.getElementById('searchMode');
    const indexSize = document.getElementById('indexSize');
    const searchStats = document.getElementById('searchStats');

    if (!searchStats) return;

    // Calculate statistics from data
    const resultCount = data.results?.length || 0;

    // Handle search time from multiple possible locations
    let timeMs = 0;
    if (data.search_time !== undefined) {
        timeMs = data.search_time;
    } else if (data.metadata?.search_time !== undefined) {
        timeMs = data.metadata.search_time;
    } else if (data.time !== undefined) {
        timeMs = data.time;
    } else if (data.elapsed_time !== undefined) {
        timeMs = data.elapsed_time;
    }

    // Calculate average distance
    let avgDist = 0;
    if (data.results && data.results.length > 0) {
        const validDistances = data.results
            .map(r => r.distance)
            .filter(d => typeof d === 'number');

        if (validDistances.length > 0) {
            avgDist = validDistances.reduce((sum, dist) => sum + dist, 0) / validDistances.length;
        }
    }

    // Update UI elements
    if (totalResults) totalResults.textContent = resultCount;
    if (searchTime) searchTime.textContent = `${timeMs.toFixed(2)}ms`;
    if (avgDistance) avgDistance.textContent = avgDist.toFixed(3);
    if (searchMode) {
        // Check multiple possible locations for fast_mode flag
        const isFastMode = data.metadata?.fast_mode || data.fast_mode || false;
        searchMode.textContent = isFastMode ? 'Fast' : 'Standard';
        console.log('Search mode:', isFastMode ? 'Fast' : 'Standard');
    }
    if (indexSize) indexSize.textContent = data.metadata?.index_size?.toLocaleString() || 'N/A';

    // Show stats section
    searchStats.style.display = 'block';
}

// Update performSearch to track timing
window.performSearch = async function () {
    console.log('performSearch called');
    const searchBtn = document.getElementById('searchBtn');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('resultsSection');
    const visualizationSection = document.getElementById('visualizationSection');

    if (!searchBtn || !loading) {
        console.error('Required elements not found');
        return;
    }

    // Track search start time
    const searchStartTime = performance.now();

    // Update UI state
    searchBtn.disabled = true;
    searchBtn.textContent = '⏳ Searching...';
    loading.style.display = 'block';

    // Scroll to center the loading indicator
    if (loading) {
        setTimeout(() => {
            const loadingRect = loading.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            const scrollPosition = window.scrollY + loadingRect.top - (windowHeight / 2) + (loadingRect.height / 2);
            window.scrollTo({ top: scrollPosition, behavior: 'smooth' });
        }, 100);
    }

    try {
        // Get color inputs
        const colorRows = document.querySelectorAll('.color-row');
        if (colorRows.length === 0) {
            throw new Error('Please add at least one color before searching');
        }

        const colors = [];
        const weights = [];
        let totalWeight = 0;

        colorRows.forEach((row, index) => {
            // Try multiple selectors to find the hidden color picker
            let picker = row.querySelector('input[data-color-picker="true"]');
            if (!picker) {
                picker = row.querySelector('input.color-picker[type="color"]');
            }
            if (!picker) {
                picker = row.querySelector('.color-picker[type="color"]');
            }
            if (!picker) {
                picker = row.querySelector('input.color-picker');
            }
            if (!picker) {
                picker = row.querySelector('.color-picker');
            }
            if (!picker) {
                // Fallback: find any color input in the row
                picker = row.querySelector('input[type="color"]');
            }
            const slider = row.querySelector('.weight-slider');

            if (picker && slider) {
                let color = picker.value;
                console.log(`[Search] Row ${index}: Found picker with value:`, color, 'picker id:', picker.id);
                // Ensure we have the color value and remove # if present
                if (color && color !== '') {
                    if (color.startsWith('#')) {
                        color = color.substring(1); // Remove # from hex color
                    }
                    const weight = parseInt(slider.value) || 100;
                    colors.push(color);
                    weights.push(weight);
                    totalWeight += weight;
                    console.log(`[Search] Row ${index}: Added color ${color} with weight ${weight}`);
                } else {
                    console.warn(`[Search] Row ${index}: Color picker found but has no value:`, picker, 'picker.value:', picker.value);
                }
            } else {
                console.warn(`[Search] Row ${index}: Missing picker or slider:`, { 
                    picker: !!picker, 
                    slider: !!slider,
                    availableInputs: row.querySelectorAll('input').length
                });
            }
        });

        // Normalize weights to sum to 1
        const normalizedWeights = weights.map(w => (w / totalWeight).toFixed(3));

        // Get performance options
        const fastMode = document.getElementById('fastMode')?.checked || false;
        const batchSize = parseInt(document.getElementById('batchSize')?.value || '10');
        const resultCount = parseInt(document.getElementById('resultCount')?.value || '10');

        // Log search configuration for debugging
        console.log('Search configuration:', {
            resultCount,
            fastMode,
            batchSize,
            colors: colors.length,
            weights: normalizedWeights.length
        });

        const searchParams = new URLSearchParams();
        // Add parameters one by one for better control
        searchParams.append('colors', colors.join(','));
        searchParams.append('weights', normalizedWeights.join(','));
        searchParams.append('k', resultCount.toString());
        searchParams.append('n_results', resultCount.toString());
        searchParams.append('fast_mode', fastMode.toString());
        searchParams.append('batch_size', batchSize.toString());

        // Log the final URL parameters
        console.log('Search URL parameters:', searchParams.toString());

        // Try to fetch search results
        const response = await fetch(`/search?${searchParams.toString()}`);
        if (!response.ok) {
            if (response.status === 503) {
                throw new Error('The search server is currently unavailable. Try clicking the "Restart Server" button in the top right corner and then search again.');
            }
            throw new Error(`Search failed with status: ${response.status}. ${await response.text()}`);
        }

        const data = await response.json();

        // Add fast_mode to the response data for the stats display
        data.fast_mode = fastMode;

        // Calculate search time if not provided by server
        const searchEndTime = performance.now();
        const clientSearchTime = searchEndTime - searchStartTime;

        // Add timing information to data if not present
        if (!data.search_time && !data.metadata?.search_time) {
            data.search_time = clientSearchTime;
        }

        // Store the search results for later use by histogram functions
        window.lastSearchResults = data.results;

        // Debug: Log first result to check dominant_colors
        if (data.results && data.results.length > 0) {
            console.log('[Search] First result:', {
                image_id: data.results[0].image_id,
                dominant_colors: data.results[0].dominant_colors,
                dominant_colors_type: typeof data.results[0].dominant_colors,
                is_array: Array.isArray(data.results[0].dominant_colors),
                length: data.results[0].dominant_colors ? data.results[0].dominant_colors.length : 0
            });
        }

        // Update UI with results
        if (resultsSection) resultsSection.style.display = 'block';
        if (visualizationSection) visualizationSection.style.display = 'block';
        loading.style.display = 'none';
        searchBtn.disabled = false;
        searchBtn.textContent = '🔍 Search Images';

        window.showSuccess('Search Complete', `Found ${data.results.length} matching images`);
        window.updateSearchResults(data);
        window.updateVisualization(data);

        // Scroll to Query Visualization h2 (almost at top)
        setTimeout(() => {
            const visualizationSection = document.getElementById('visualizationSection');
            if (visualizationSection) {
                const queryVizH2 = visualizationSection.querySelector('h2');
                if (queryVizH2 && queryVizH2.textContent.includes('Query Visualization')) {
                    const h2Rect = queryVizH2.getBoundingClientRect();
                    const scrollPosition = window.scrollY + h2Rect.top - 100; // 100px from top
                    window.scrollTo({ top: Math.max(0, scrollPosition), behavior: 'smooth' });
                }
            }
        }, 100);

    } catch (err) {
        console.error('Search failed:', err);
        window.showError('Search Failed', err.message);

        // Clear results on error
        const resultsGrid = document.getElementById('resultsGrid');
        if (resultsGrid) {
            resultsGrid.innerHTML = '';
        }
    } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = '🔍 Search Images';
        loading.style.display = 'none';
    }
};

// Function to show image in a modal popup
window.showImageInModal = function (imageSrc) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');

    if (modal && modalImage) {
        modalImage.src = imageSrc;
        modal.style.display = 'flex';
    }
};

// Function to update search results in the UI
window.updateSearchResults = function (data) {
    // Play the sound when results are ready
    playSearchCompleteSound();

    const resultsGrid = document.getElementById('resultsGrid');
    if (!resultsGrid) return;

    resultsGrid.innerHTML = '';
    console.log('Processing search results:', data.results); // Debug log

    data.results.forEach((result, index) => {
        console.log(`Result ${index + 1}:`, {
            image_id: result.image_id,
            image_url: result.image_url,
            file_path: result.file_path,
            distance: result.distance
        }); // Debug log for each result

        const imgSrc = result.image_url || `/image/${encodeURIComponent(result.image_id)}`;
        console.log(`Using imgSrc: ${imgSrc}`); // Debug log for image source
        const distance = typeof result.distance === 'number' ? result.distance.toFixed(6) : 'N/A';

        console.log('Result dominant colors:', result.dominant_colors); // Debug log

        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';

        // Add rank number as a badge
        const rankBadge = document.createElement('div');
        rankBadge.className = 'rank-badge';
        rankBadge.textContent = `#${index + 1}`;
        resultCard.appendChild(rankBadge);

        // Image container with hover overlay
        const imageContainer = document.createElement('div');
        imageContainer.className = 'result-image-container';
        imageContainer.style.position = 'relative';
        imageContainer.style.overflow = 'hidden';
        imageContainer.style.borderRadius = '8px';
        imageContainer.style.cursor = 'pointer';

        const img = document.createElement('img');
        img.src = imgSrc;
        img.alt = `Result ${index + 1}`;
        img.className = 'result-image';
        img.style.transition = 'transform 0.3s ease';
        imageContainer.appendChild(img);

        // Create hover overlay
        const overlay = document.createElement('div');
        overlay.className = 'image-overlay';
        overlay.style.position = 'absolute';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.right = '0';
        overlay.style.bottom = '0';
        overlay.style.background = 'var(--blue)';
        overlay.style.display = 'flex';
        overlay.style.alignItems = 'center';
        overlay.style.justifyContent = 'center';
        overlay.style.opacity = '0';
        overlay.style.transition = 'opacity 0.3s ease';
        overlay.style.color = '#000000';
        overlay.style.fontSize = '14px';
        overlay.style.fontWeight = 'bold';
        overlay.textContent = 'Click to View';
        imageContainer.appendChild(overlay);

        // Add hover effects
        imageContainer.addEventListener('mouseenter', () => {
            overlay.style.opacity = '1';
            img.style.transform = 'scale(1.05)';
        });

        imageContainer.addEventListener('mouseleave', () => {
            overlay.style.opacity = '0';
            img.style.transform = 'scale(1)';
        });

        // Add click to view full resolution
        imageContainer.addEventListener('click', () => {
            window.showFullResolutionImage(imgSrc);
        });

        resultCard.appendChild(imageContainer);

        // Info section
        const infoSection = document.createElement('div');
        infoSection.className = 'result-info';

        // Image ID
        const idElement = document.createElement('p');
        idElement.innerHTML = `<strong>ID:</strong> <span class="image-id">${result.image_id}</span>`;
        infoSection.appendChild(idElement);

        // Distance score
        const distanceElement = document.createElement('p');
        distanceElement.innerHTML = `<strong>Distance:</strong> <span class="distance-score">${distance}</span>`;
        infoSection.appendChild(distanceElement);

        // Dominant colors section
        if (result.dominant_colors && Array.isArray(result.dominant_colors)) {
            const colorsContainer = document.createElement('div');
            colorsContainer.className = 'dominant-colors';
            const colorsTitle = document.createElement('p');
            colorsTitle.innerHTML = '<strong>Dominant Colors:</strong>';
            colorsContainer.appendChild(colorsTitle);

            const colorSwatches = document.createElement('div');
            colorSwatches.className = 'color-swatches';

            result.dominant_colors.forEach(color => {
                const swatch = document.createElement('div');
                swatch.className = 'color-swatch';
                swatch.style.backgroundColor = color;
                swatch.title = `Click to copy: ${color}`;

                const colorLabel = document.createElement('span');
                colorLabel.className = 'color-label';
                colorLabel.textContent = color;
                swatch.appendChild(colorLabel);

                // Add color name below swatch
                const colorName = document.createElement('div');
                colorName.className = 'color-name';
                colorName.textContent = getColorName(color);
                colorName.style.fontSize = '11px';
                colorName.style.color = 'var(--subtext0)';
                colorName.style.textAlign = 'center';
                colorName.style.marginTop = '2px';
                swatch.appendChild(colorName);

                swatch.addEventListener('click', () => {
                    navigator.clipboard.writeText(color)
                        .then(() => {
                            swatch.setAttribute('data-copied', 'true');
                            setTimeout(() => swatch.removeAttribute('data-copied'), 1500);
                        });
                });
                colorSwatches.appendChild(swatch);
            });

            colorsContainer.appendChild(colorSwatches);
            infoSection.appendChild(colorsContainer);
        }

        // Buttons container
        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'result-buttons';

        // Download button
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'action-btn download-btn';
        downloadBtn.innerHTML = 'Download';
        downloadBtn.onclick = async () => {
            try {
                // Fetch the image as a blob to force download instead of opening
                const response = await fetch(imgSrc);
                const blob = await response.blob();
                const blobUrl = URL.createObjectURL(blob);
                
                // Create download link with blob URL
                const link = document.createElement('a');
                link.href = blobUrl;
                link.download = result.image_id || 'image';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                // Clean up blob URL after a short delay
                setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
                
                window.showSuccess('Download', 'Image download started');
            } catch (err) {
                console.error('Download error:', err);
                window.showError('Download Failed', 'Could not download image. Please try again.');
            }
        };
        buttonsContainer.appendChild(downloadBtn);

        // Copy Info button
        const copyBtn = document.createElement('button');
        copyBtn.className = 'action-btn copy-btn';
        copyBtn.innerHTML = 'Copy Info';
        copyBtn.onclick = () => {
            const info = {
                imageId: result.image_id,
                distance: distance,
                dominantColors: result.dominant_colors || [],
                url: imgSrc
            };
            navigator.clipboard.writeText(JSON.stringify(info, null, 2))
                .then(() => {
                    copyBtn.innerHTML = '✓ Copied!';
                    setTimeout(() => copyBtn.innerHTML = 'Copy Info', 1500);
                });
        };
        buttonsContainer.appendChild(copyBtn);

        // Details button
        const detailsBtn = document.createElement('button');
        detailsBtn.className = 'action-btn details-btn';
        detailsBtn.innerHTML = 'Details';
        detailsBtn.onclick = () => showImageDetails(result, index + 1);
        buttonsContainer.appendChild(detailsBtn);

        infoSection.appendChild(buttonsContainer);
        resultCard.appendChild(infoSection);
        resultsGrid.appendChild(resultCard);
    });

    // Show the results section
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
};

// Show full resolution image in a modal
window.showFullResolutionImage = function (imageSrc) {
    // Create or get the full resolution modal
    let fullResModal = document.getElementById('fullResolutionModal');
    if (!fullResModal) {
        fullResModal = document.createElement('div');
        fullResModal.id = 'fullResolutionModal';
        fullResModal.style.cssText = `
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            cursor: pointer;
        `;
        document.body.appendChild(fullResModal);
    }

    const modalContent = `
        <div style="position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
            <img src="${imageSrc}" alt="Full resolution image" style="max-width: 95%; max-height: 95%; object-fit: contain; border-radius: 8px;">
            <button style="position: absolute; top: 20px; right: 30px; background: var(--yellow); color: #000000; border: none; font-size: 30px; cursor: pointer; border-radius: 50%; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center;">&times;</button>
        </div>
    `;

    fullResModal.innerHTML = modalContent;
    fullResModal.style.display = 'block';

    // Close handlers
    const closeBtn = fullResModal.querySelector('button');
    closeBtn.onclick = (e) => {
        e.stopPropagation();
        fullResModal.style.display = 'none';
    };

    fullResModal.onclick = () => {
        fullResModal.style.display = 'none';
    };

    // Prevent image click from closing modal
    const img = fullResModal.querySelector('img');
    img.onclick = (e) => e.stopPropagation();
};

// Show detailed information about an image result
function showImageDetails(result, rank) {
    const modal = document.getElementById('detailsModal');
    if (!modal) {
        console.error('Details modal not found');
        return;
    }

    // Calculate additional metrics
    const distance = typeof result.distance === 'number' ? result.distance : 0;
    const matchScore = ((1 - distance) * 100).toFixed(2);
    const imageSrc = result.image_url || `/image/${encodeURIComponent(result.image_id)}`;
    const colorCount = result.dominant_colors ? result.dominant_colors.length : 0;

    // Determine match quality
    let matchQuality = 'Poor';
    let qualityColor = '#f38ba8'; // red
    if (distance < 0.2) {
        matchQuality = 'Excellent';
        qualityColor = '#a6e3a1'; // green
    } else if (distance < 0.4) {
        matchQuality = 'Very Good';
        qualityColor = '#94e2d5'; // teal
    } else if (distance < 0.6) {
        matchQuality = 'Good';
        qualityColor = '#f9e2af'; // yellow
    } else if (distance < 0.8) {
        matchQuality = 'Fair';
        qualityColor = '#fab387'; // peach
    }

    const modalContent = `
        <div class="modal-header" style="display: flex; justify-content: space-between; align-items: center; padding: 20px; border-bottom: 1px solid var(--surface2); position: relative;">
            <h2 style="margin: 0; color: var(--text); font-family: 'JetBrainsMono Nerd Font Mono', monospace;">Image Details - Rank #${rank}</h2>
            <button class="close" onclick="window.closeDetailsModal()" style="background: var(--yellow) !important; border: none !important; font-size: 32px !important; color: #000000 !important; cursor: pointer !important; padding: 0 !important; width: 40px !important; height: 40px !important; border-radius: 50% !important; display: flex !important; align-items: center !important; justify-content: center !important; position: absolute !important; top: 15px !important; right: 20px !important; z-index: 1001 !important; transition: all 0.2s !important; font-weight: bold !important;">&times;</button>
        </div>
        <div class="modal-body" style="padding: 20px; max-height: 60vh; overflow-y: auto;">
            <div class="details-section" style="margin-bottom: 25px;">
                <img src="${imageSrc}" alt="Result image" style="width: 100%; max-height: 300px; object-fit: contain; border-radius: 8px; background: var(--surface0);" onclick="window.showFullResolutionImage('${imageSrc}')" title="Click to view full resolution">
            </div>
            
            <div class="details-section" style="margin-bottom: 25px;">
                <h3 style="color: var(--blue); margin: 0 0 15px 0; font-size: 16px; border-bottom: 2px solid var(--blue); padding-bottom: 5px;">📄 Image Information</h3>
                <div class="details-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Image ID</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold; font-family: monospace;">${result.image_id || 'N/A'}</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Rank Position</span>
                        <span class="detail-value" style="color: var(--mauve); font-weight: bold; font-size: 18px;">#${rank}</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px; grid-column: 1 / -1;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">File Path</span>
                        <span class="detail-value" style="color: var(--text); font-family: monospace; word-break: break-all;">${result.filepath || result.image_id || 'N/A'}</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Distance Score</span>
                        <span class="detail-value" style="color: var(--red); font-weight: bold; font-family: monospace;">${distance.toFixed(6)}</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Image Format</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold;">JPG/PNG</span>
                    </div>
                </div>
            </div>

            <div class="details-section" style="margin-bottom: 25px;">
                <h3 style="color: var(--green); margin: 0 0 15px 0; font-size: 16px; border-bottom: 2px solid var(--green); padding-bottom: 5px;">🎨 Color Analysis</h3>
                <div style="background: var(--surface0); padding: 15px; border-radius: 8px;">
                    <div style="margin-bottom: 12px;">
                        <span style="font-size: 14px; color: var(--subtext1); font-weight: bold;">Dominant Colors (${colorCount} colors detected):</span>
                    </div>
                    <div class="modal-color-swatches" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
                        ${result.dominant_colors ? result.dominant_colors.map((color, index) => `
                            <div class="modal-color-swatch" style="display: flex; flex-direction: column; align-items: center; min-width: 80px;">
                                <div style="width: 50px; height: 50px; background-color: ${color}; border-radius: 8px; border: 2px solid var(--surface2); margin-bottom: 5px; cursor: pointer;" onclick="navigator.clipboard.writeText('${color}')" title="Click to copy ${color}"></div>
                                <span class="color-hex" style="font-size: 10px; color: var(--subtext1); font-family: monospace;">${color}</span>
                                <span class="color-name" style="font-size: 10px; color: var(--subtext0); text-align: center; max-width: 80px; word-wrap: break-word;">${getColorName(color)}</span>
                            </div>
                        `).join('') : '<span style="color: var(--subtext0);">No dominant colors available</span>'}
                    </div>
                    <div class="color-stats" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px;">
                        <div style="color: var(--subtext1);">
                            <strong>Color Diversity:</strong> ${colorCount > 4 ? 'High' : colorCount > 2 ? 'Medium' : 'Low'}
                        </div>
                        <div style="color: var(--subtext1);">
                            <strong>Color Temperature:</strong> ${result.dominant_colors && result.dominant_colors.length > 0 ? 'Mixed' : 'Unknown'}
                        </div>
                    </div>
                </div>
            </div>

            <div class="details-section" style="margin-bottom: 25px;">
                <h3 style="color: var(--yellow); margin: 0 0 15px 0; font-size: 16px; border-bottom: 2px solid var(--yellow); padding-bottom: 5px;">⚡ Processing Status</h3>
                <div class="details-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Index Status</span>
                        <span class="detail-value" style="color: var(--green); font-weight: bold;">✓ Indexed</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Processing Method</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold;">FAISS + EMD</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Algorithm</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold;">Sinkhorn-EMD</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Vector Dimensions</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold;">512D</span>
                    </div>
                </div>
            </div>

            <div class="details-section" style="margin-bottom: 25px;">
                <h3 style="color: var(--mauve); margin: 0 0 15px 0; font-size: 16px; border-bottom: 2px solid var(--mauve); padding-bottom: 5px;">📊 Search Relevance</h3>
                <div class="details-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Color Match Score</span>
                        <span class="detail-value" style="color: var(--green); font-weight: bold; font-size: 18px;">${matchScore}%</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Match Quality</span>
                        <span class="detail-value" style="color: ${qualityColor}; font-weight: bold;">${matchQuality}</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Confidence Level</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold;">${distance < 0.3 ? 'Very High' : distance < 0.6 ? 'High' : 'Medium'}</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Search Method</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold;">Two-Stage</span>
                    </div>
                </div>
            </div>

            <div class="details-section" style="margin-bottom: 25px;">
                <h3 style="color: var(--teal); margin: 0 0 15px 0; font-size: 16px; border-bottom: 2px solid var(--teal); padding-bottom: 5px;">🔧 Technical Details</h3>
                <div class="details-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Histogram Bins</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold;">64 bins</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Color Space</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold;">LAB</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Distance Metric</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold;">Earth Mover's</span>
                    </div>
                    <div class="detail-item" style="background: var(--surface0); padding: 10px; border-radius: 6px;">
                        <span class="detail-label" style="display: block; font-size: 12px; color: var(--subtext0); margin-bottom: 4px;">Index Type</span>
                        <span class="detail-value" style="color: var(--text); font-weight: bold;">HNSW</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Set content in the modal content div, not the modal itself
    const modalContentDiv = document.getElementById('imageDetailsContent');
    if (modalContentDiv) {
        modalContentDiv.innerHTML = modalContent;
    }

    // Show the modal with proper styling
    modal.style.cssText = `
        display: flex !important;
        position: fixed !important;
        z-index: 1000 !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        transform: none !important;
        background-color: rgba(0, 0, 0, 0.85) !important;
        align-items: center !important;
        justify-content: center !important;
        overflow-y: auto !important;
        margin: 0 !important;
        padding: 0 !important;
    `;

    // Ensure modal-content has proper styling
    const modalContentEl = modal.querySelector('.modal-content');
    if (modalContentEl) {
        modalContentEl.style.cssText = `
            background: var(--base) !important;
            border: 2px solid var(--surface1) !important;
            border-radius: 12px !important;
            width: 90% !important;
            max-width: 800px !important;
            max-height: 90vh !important;
            margin: 0 auto !important;
            padding: 0 !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
            display: flex !important;
            flex-direction: column !important;
    `;
    }

    // Add event listener to close button - find the dynamically created one
    const closeBtn = modalContentDiv ? modalContentDiv.querySelector('.close') : modal.querySelector('.close');
    if (closeBtn) {
        closeBtn.onclick = function (e) {
            e.stopPropagation();
            window.closeDetailsModal();
        };
    }

    // Close modal when clicking outside the modal-content
    modal.onclick = function (event) {
        if (event.target === modal) {
            window.closeDetailsModal();
        }
    };

    // Prevent modal-content clicks from closing the modal
    if (modalContentEl) {
        modalContentEl.onclick = function (e) {
            e.stopPropagation();
        };
    }
}

// Function to download preview grid as image
window.downloadPreviewGrid = function () {
    const previewGrid = document.getElementById('previewGrid');
    if (!previewGrid) {
        window.showError('Download Failed', 'Could not find preview grid');
        return;
    }

    // Add a title to the downloadable grid
    const title = document.createElement('h4');
    title.textContent = 'Chromatica Search Results';
    title.style.textAlign = 'center';
    title.style.margin = '0 0 15px 0';
    title.style.color = '#cdd6f4'; // Using hex instead of var(--text)
    title.style.fontFamily = "'JetBrainsMono Nerd Font Mono', monospace";

    // Create a temporary container with the exact same styling as the collage grid
    const tempContainer = document.createElement('div');
    tempContainer.style.display = 'flex';
    tempContainer.style.flexDirection = 'column';
    tempContainer.style.padding = '20px';
    tempContainer.style.background = '#313244'; // Using hex instead of var(--surface0)
    tempContainer.style.borderRadius = '12px';
    tempContainer.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)';
    tempContainer.style.width = '800px'; // Fixed width to ensure consistency
    tempContainer.style.position = 'fixed';
    tempContainer.style.top = '-9999px';
    tempContainer.style.left = '-9999px';

    // Add timestamp
    const timestamp = document.createElement('div');
    timestamp.textContent = new Date().toLocaleString();
    timestamp.style.textAlign = 'center';
    timestamp.style.fontSize = '12px';
    timestamp.style.color = '#a6adc8'; // Using hex instead of var(--subtext0)
    timestamp.style.marginBottom = '15px';

    // Create the grid container
    const gridContainer = document.createElement('div');
    gridContainer.style.display = 'grid';
    gridContainer.style.gridTemplateColumns = 'repeat(3, 1fr)';
    gridContainer.style.gap = '8px';

    tempContainer.appendChild(title);
    tempContainer.appendChild(timestamp);
    tempContainer.appendChild(gridContainer);
    document.body.appendChild(tempContainer);

    // Load and clone all images first
    const imageLoadPromises = Array.from(previewGrid.children).map(item => {
        return new Promise((resolve) => {
            const clone = item.cloneNode(true);
            const originalImg = item.querySelector('img');
            const clonedImg = clone.querySelector('img');

            if (originalImg && clonedImg) {
                // Create a new image to properly load with CORS
                const newImg = new Image();
                newImg.crossOrigin = 'anonymous';
                newImg.onload = () => {
                    // Once loaded, update the cloned image
                    clonedImg.src = newImg.src;
                    // Ensure the image size is maintained
                    clonedImg.style.width = '100%';
                    clonedImg.style.height = '100%';
                    clonedImg.style.objectFit = 'cover';
                    resolve(clone);
                };
                newImg.onerror = () => {
                    // If image fails to load, still resolve but maybe with a placeholder
                    clonedImg.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="%23313244"/><text x="50" y="50" text-anchor="middle" fill="%23cdd6f4">No Image</text></svg>';
                    resolve(clone);
                };
                newImg.src = originalImg.src;
            } else {
                resolve(clone);
            }
        });
    });

    // Wait for all images to load then create the canvas
    Promise.all(imageLoadPromises).then(clones => {
        // Add all clones to the grid
        clones.forEach(clone => gridContainer.appendChild(clone));

        // Use html2canvas with specific options for better image handling
        html2canvas(tempContainer, {
            backgroundColor: '#313244',
            scale: 2,
            logging: false,
            allowTaint: true,
            useCORS: true,
            imageTimeout: 15000,
            onclone: function (clonedDoc) {
                // Additional styling fixes in the cloned document if needed
                const clonedImages = clonedDoc.querySelectorAll('img');
                clonedImages.forEach(img => {
                    img.style.width = '100%';
                    img.style.height = '100%';
                    img.style.objectFit = 'cover';
                });
            }
        }).then(canvas => {
            // Create download link
            const link = document.createElement('a');
            link.download = `chromatica-results-${new Date().toISOString().slice(0, 10)}.png`;
            link.href = canvas.toDataURL('image/png');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Clean up
            document.body.removeChild(tempContainer);
            window.showSuccess('Download Complete', 'Results grid image downloaded successfully');
        }).catch(err => {
            // Clean up on error
            document.body.removeChild(tempContainer);
            window.showError('Download Failed', err.message);
        });
    });
};

// Restart the server function
window.restartServer = function () {
    console.log('Restarting Chromatica server...');

    // Show a confirmation dialog
    if (!confirm('Are you sure you want to restart the Chromatica server? This will interrupt any ongoing searches.')) {
        return;
    }

    // Update the restart button
    const restartButton = document.getElementById('restartButton');
    if (restartButton) {
        restartButton.innerHTML = '⏳ Restarting...';
        restartButton.style.backgroundColor = 'var(--overlay0)';
        restartButton.disabled = true;
    }

    // Show a loading message
    window.showSuccess('Restarting Server', 'Restarting the Chromatica server. Please wait...');

    // Send a request to the restart endpoint
    fetch('/restart', {
        method: 'POST'
    })
        .then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error(`Server restart failed with status: ${response.status}`);
        })
        .then(data => {
            console.log('Server restart response:', data);
            window.showSuccess('Server Restarted', 'Chromatica server has been restarted successfully.');

            // Wait 2 seconds and then reload the page
            setTimeout(() => {
                window.location.reload();
            }, 2000);
        })
        .catch(error => {
            console.error('Server restart error:', error);
            window.showError('Restart Failed', error.message);

            // Reset the restart button
            if (restartButton) {
                restartButton.innerHTML = '🔄 Restart Server';
                restartButton.style.backgroundColor = 'var(--red)';
                restartButton.disabled = false;
            }
        });
};

// ============================================================================
// COLOR ENHANCEMENTS: Color Names and Schemes (Step 1-8 Implementation)
// ============================================================================

// Global variables for color enhancements
window.colorNames = [];
window.colorNamesLoaded = false;

// Load color names from colornames.json
async function loadColorNames() {
    if (window.colorNamesLoaded) {
        console.log('Color names already loaded, colorNames length:', window.colorNames?.length);
        return Promise.resolve();
    }

    console.log('Current window.colorNames:', window.colorNames);

    try {
        console.log('Loading color names from colornames.json...');
        const response = await fetch('/static/colornames.json');
        console.log('Color names fetch response:', response.status, response.statusText);

        if (!response.ok) {
            throw new Error(`Failed to load color names: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log('Color names data loaded:', data ? 'success' : 'empty');

        if (!data || !Array.isArray(data) || data.length === 0) {
            throw new Error('Color names data is empty or invalid');
        }

        window.colorNames = data;
        window.colorNamesLoaded = true;
        console.log(`Successfully loaded ${window.colorNames.length} color names`);

        // Try getting a sample color name to verify functionality
        const sampleColor = '#FF0000';
        const sampleName = getColorName(sampleColor);
        console.log(`Sample color test - ${sampleColor}: "${sampleName}"`);

        return Promise.resolve();
    } catch (error) {
        console.error('Error loading color names:', error);
        window.colorNames = [];
        window.colorNamesLoaded = false;
        return Promise.reject(error);
    }
}

// Get color name from hex value with better matching
function getColorName(hexColor) {
    if (!window.colorNamesLoaded || !Array.isArray(window.colorNames) || window.colorNames.length === 0) {
        console.log('Color names not available, returning hex code');
        return hexColor;
    }

    // Handle invalid input
    if (!hexColor || typeof hexColor !== 'string') {
        console.warn('Invalid color input:', hexColor);
        return 'Invalid color';
    }

    // Remove # if present and ensure uppercase
    const cleanHex = hexColor.replace('#', '').toUpperCase();

    // Validate hex format
    if (!/^[0-9A-F]{6}$/.test(cleanHex)) {
        console.warn('Invalid hex format:', hexColor);
        return hexColor;
    }

    // Find exact match first
    const exactMatch = window.colorNames.find(color =>
        color.hex.replace('#', '').toUpperCase() === cleanHex
    );

    if (exactMatch) {
        console.log(`Exact match found for ${hexColor}: "${exactMatch.name}"`);
        return exactMatch.name;
    }

    // Convert target color to Lab color space for better matching
    const targetRgb = hexToRgb(cleanHex);
    if (!targetRgb) {
        console.warn('Could not convert hex to RGB:', hexColor);
        return hexColor;
    }

    const targetLab = rgbToLab(targetRgb.r, targetRgb.g, targetRgb.b);

    // Find closest color using Delta E (CIE76)
    let closestColor = null;
    let minDeltaE = Infinity;

    for (const color of window.colorNames) {
        const rgb = hexToRgb(color.hex.replace('#', ''));
        if (!rgb) continue;

        const lab = rgbToLab(rgb.r, rgb.g, rgb.b);
        const distance = calculateDeltaE(targetLab, lab);

        if (distance < minDeltaE) {
            minDeltaE = distance;
            closestColor = color;
        }
    }

    if (closestColor) {
        console.log(`Closest match for ${hexColor}: "${closestColor.name}" (ΔE: ${minDeltaE.toFixed(2)})`);
        return closestColor.name;
    }

    console.warn('No suitable color name found for:', hexColor);
    return hexColor;
}

// Helper function to convert hex to RGB
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

// Helper function to convert RGB to hex
function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

// Convert RGB to Lab color space
function rgbToLab(r, g, b) {
    // First convert RGB to XYZ
    let x, y, z;
    r = r / 255;
    g = g / 255;
    b = b / 255;

    // Convert RGB to linear RGB (remove gamma correction)
    r = r > 0.04045 ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = g > 0.04045 ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = b > 0.04045 ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    // Convert to XYZ using D65 illuminant
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) * 100;
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) * 100;
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) * 100;

    // Convert XYZ to Lab
    // Using D65 reference white
    x = x / 95.047;
    y = y / 100.000;
    z = z / 108.883;

    x = x > 0.008856 ? Math.pow(x, 1 / 3) : (7.787 * x) + 16 / 116;
    y = y > 0.008856 ? Math.pow(y, 1 / 3) : (7.787 * y) + 16 / 116;
    z = z > 0.008856 ? Math.pow(z, 1 / 3) : (7.787 * z) + 16 / 116;

    return {
        l: (116 * y) - 16,
        a: 500 * (x - y),
        b: 200 * (y - z)
    };
}

// Calculate color difference using Delta E (CIE76)
function calculateDeltaE(lab1, lab2) {
    return Math.sqrt(
        Math.pow(lab2.l - lab1.l, 2) +
        Math.pow(lab2.a - lab1.a, 2) +
        Math.pow(lab2.b - lab1.b, 2)
    );
}

// Helper function to convert hex to HSL
function hexToHsl(hex) {
    const rgb = hexToRgb(hex);
    if (!rgb) return null;

    let { r, g, b } = rgb;
    r /= 255;
    g /= 255;
    b /= 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;

    if (max === min) {
        h = s = 0;
    } else {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }

    return { h: h * 360, s: s * 100, l: l * 100 };
}

// Helper function to convert HSL to hex
function hslToHex(h, s, l) {
    h /= 360;
    s /= 100;
    l /= 100;

    const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
    };

    let r, g, b;

    if (s === 0) {
        r = g = b = l;
    } else {
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }

    return rgbToHex(
        Math.round(r * 255),
        Math.round(g * 255),
        Math.round(b * 255)
    );
}

// Generate complementary color
function getComplementaryColor(hex) {
    // Convert hex to HSL
    const hsl = hexToHsl(hex);

    // Add 180 degrees to hue for complementary color
    hsl.h = (hsl.h + 180) % 360;

    // Convert back to hex
    return hslToHex(hsl.h, hsl.s, hsl.l);
}

// Generate monochromatic color scheme
function generateMonochromaticScheme(baseColor) {
    const hsl = hexToHsl(baseColor);
    const colors = [];

    // Generate variations by adjusting lightness and saturation
    colors.push(baseColor); // Original color
    colors.push(hslToHex(hsl.h, hsl.s, Math.max(0, hsl.l - 0.3))); // Darker
    colors.push(hslToHex(hsl.h, Math.min(100, hsl.s + 0.2), hsl.l)); // More saturated
    colors.push(hslToHex(hsl.h, hsl.s, Math.min(1, hsl.l + 0.3))); // Lighter
    colors.push(hslToHex(hsl.h, Math.max(0, hsl.s - 0.2), hsl.l)); // Less saturated

    return colors;
}

// Generate analogous color scheme
function generateAnalogousScheme(baseColor) {
    const hsl = hexToHsl(baseColor);
    const colors = [];

    // Generate analogous colors (30 degrees apart)
    colors.push(baseColor); // Original color
    colors.push(hslToHex((hsl.h - 30 + 360) % 360, hsl.s, hsl.l));
    colors.push(hslToHex((hsl.h + 30) % 360, hsl.s, hsl.l));

    return colors;
}

// Generate triadic color scheme
function generateTriadicScheme(baseColor) {
    const hsl = hexToHsl(baseColor);
    const colors = [];

    // Generate triadic colors (120 degrees apart)
    colors.push(baseColor); // Original color
    colors.push(hslToHex((hsl.h + 120) % 360, hsl.s, hsl.l));
    colors.push(hslToHex((hsl.h + 240) % 360, hsl.s, hsl.l));

    return colors;
}

// Generate color schemes from search results (Step 1)
window.generateColorSchemesFromResults = function () {
    console.log('Generating color schemes from search results');

    if (!window.lastSearchResults || !Array.isArray(window.lastSearchResults) || window.lastSearchResults.length === 0) {
        window.showError('Scheme Generation Error', 'No search results available. Please perform a search first.');
        return;
    }

    // Find or create the color schemes section
    let schemesSection = document.getElementById('colorSchemesSection');
    const resultsSection = document.getElementById('resultsSection');

    if (!schemesSection) {
        // Create new color schemes section if it doesn't exist
        schemesSection = document.createElement('div');
        schemesSection.id = 'colorSchemesSection';
        schemesSection.style.marginTop = '30px';
        schemesSection.style.padding = '20px';
        schemesSection.style.background = 'var(--surface1)';
        schemesSection.style.borderRadius = '12px';
        schemesSection.style.border = '1px solid var(--surface2)';
    } else {
        // Clear existing content
        schemesSection.innerHTML = '';
        // Remove it from its current position
        schemesSection.remove();
    }

    // Insert after results section
    if (resultsSection && resultsSection.parentNode) {
        resultsSection.parentNode.insertBefore(schemesSection, resultsSection.nextSibling);
    }

    // Create header
    const header = document.createElement('h3');
    header.textContent = '🎨 Generated Color Schemes from Results';
    header.style.color = 'var(--text)';
    header.style.margin = '0 0 20px 0';
    header.style.fontFamily = "'JetBrainsMono Nerd Font Mono', monospace";
    schemesSection.appendChild(header);

    // Extract dominant colors from top results
    const topResults = window.lastSearchResults.slice(0, 5);
    const allColors = [];

    topResults.forEach((result, idx) => {
        console.log(`[Generate Schemes] Result ${idx + 1}:`, {
            image_id: result.image_id,
            dominant_colors: result.dominant_colors,
            dominant_colors_type: typeof result.dominant_colors,
            is_array: Array.isArray(result.dominant_colors),
            length: result.dominant_colors ? (Array.isArray(result.dominant_colors) ? result.dominant_colors.length : 0) : 0
        });
        if (result.dominant_colors && Array.isArray(result.dominant_colors) && result.dominant_colors.length > 0) {
            allColors.push(...result.dominant_colors);
        }
    });

    console.log(`[Generate Schemes] Total colors collected: ${allColors.length}`);

    if (allColors.length === 0) {
        schemesSection.innerHTML = '<p style="color: var(--subtext1);">No dominant colors found in search results.</p>';
        return;
    }

    // Generate different color schemes
    const schemes = generateColorSchemes(allColors);

    // Create schemes container
    const schemesContainer = document.createElement('div');
    schemesContainer.style.display = 'grid';
    schemesContainer.style.gridTemplateColumns = 'repeat(auto-fit, minmax(300px, 1fr))';
    schemesContainer.style.gap = '20px';

    schemes.forEach(scheme => {
        const schemeCard = createSchemeCard(scheme);
        schemesContainer.appendChild(schemeCard);
    });

    schemesSection.appendChild(schemesContainer);

    // Show success message
    window.showSuccess('Color Schemes Generated', `Created ${schemes.length} color schemes from search results.`);

    // Scroll to center the colorSchemesSection
    setTimeout(() => {
        if (schemesSection) {
            const sectionRect = schemesSection.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            const scrollPosition = window.scrollY + sectionRect.top - (windowHeight / 2) + (sectionRect.height / 2);
            window.scrollTo({ top: scrollPosition, behavior: 'smooth' });
        }
    }, 100);
};

// Generate different types of color schemes
function generateColorSchemes(colors) {
    const schemes = [];

    // Get unique colors (remove duplicates)
    const uniqueColors = [...new Set(colors)];

    // Take first 3-5 colors for base schemes
    const baseColors = uniqueColors.slice(0, 5);

    if (baseColors.length > 0) {
        // Ensure all color arrays are valid before adding to schemes
        const ensureColorArray = (colors) => Array.isArray(colors) ? colors : [];

        // Monochromatic scheme based on first color
        schemes.push({
            name: 'Monochromatic',
            type: 'monochromatic',
            colors: ensureColorArray(generateMonochromaticScheme(baseColors[0])),
            description: 'Different shades and tints of the dominant color'
        });

        // Complementary scheme
        if (baseColors.length >= 2) {
            schemes.push({
                name: 'Complementary',
                type: 'complementary',
                colors: [baseColors[0], getComplementaryColor(baseColors[0])],
                description: 'Two colors opposite on the color wheel'
            });
        }

        // Analogous scheme
        if (baseColors.length >= 3) {
            schemes.push({
                name: 'Analogous',
                type: 'analogous',
                colors: ensureColorArray(generateAnalogousScheme(baseColors[0])),
                description: 'Colors adjacent on the color wheel'
            });
        }

        // Triadic scheme
        schemes.push({
            name: 'Triadic',
            type: 'triadic',
            colors: ensureColorArray(generateTriadicScheme(baseColors[0])),
            description: 'Three colors evenly spaced on the color wheel'
        });

        // Result-based palette (actual dominant colors)
        schemes.push({
            name: 'Result Palette',
            type: 'result-based',
            colors: baseColors,
            description: 'Dominant colors from your search results'
        });
    }

    return schemes;
}

// Create a visual card for a color scheme
function createSchemeCard(scheme) {
    const card = document.createElement('div');
    card.style.background = 'var(--surface0)';
    card.style.padding = '15px';
    card.style.borderRadius = '8px';
    card.style.border = '1px solid var(--surface2)';

    // Scheme header
    const header = document.createElement('div');
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.marginBottom = '10px';

    const title = document.createElement('h4');
    title.textContent = scheme.name;
    title.style.margin = '0';
    title.style.color = 'var(--text)';
    title.style.fontSize = '16px';

    const useBtn = document.createElement('button');
    useBtn.textContent = 'Use Scheme';
    useBtn.style.background = 'var(--blue)';
    useBtn.style.color = '#000000';
    useBtn.style.border = 'none';
    useBtn.style.padding = '6px 12px';
    useBtn.style.borderRadius = '4px';
    useBtn.style.cursor = 'pointer';
    useBtn.style.fontSize = '12px';
    useBtn.onclick = () => applyColorScheme(scheme);

    header.appendChild(title);
    header.appendChild(useBtn);
    card.appendChild(header);

    // Description
    const description = document.createElement('p');
    description.textContent = scheme.description;
    description.style.margin = '0 0 15px 0';
    description.style.color = 'var(--subtext1)';
    description.style.fontSize = '14px';
    card.appendChild(description);

    // Color swatches
    const swatchesContainer = document.createElement('div');
    swatchesContainer.style.display = 'flex';
    swatchesContainer.style.gap = '8px';
    swatchesContainer.style.flexWrap = 'wrap';

    scheme.colors.forEach(color => {
        const swatch = document.createElement('div');
        swatch.className = 'color-palette-swatch';
        swatch.style.width = '40px';
        swatch.style.height = '40px';
        swatch.style.backgroundColor = color;
        swatch.style.borderRadius = '6px';
        swatch.style.border = '2px solid var(--surface2)';
        swatch.style.cursor = 'pointer';
        swatch.title = `${color} - ${getColorName(color)}`;

        // Add color name below swatch
        const colorName = document.createElement('div');
        colorName.className = 'color-name';
        colorName.textContent = getColorName(color);
        colorName.style.fontSize = '11px';
        colorName.style.color = 'var(--subtext0)';
        colorName.style.textAlign = 'center';
        colorName.style.marginTop = '2px';
        swatch.appendChild(colorName);

        swatch.onclick = () => {
            navigator.clipboard.writeText(color).then(() => {
                swatch.style.transform = 'scale(1.1)';
                setTimeout(() => swatch.style.transform = 'scale(1)', 200);
            });
        };

        swatchesContainer.appendChild(swatch);
    });

    card.appendChild(swatchesContainer);

    // Color names
    const namesContainer = document.createElement('div');
    namesContainer.style.marginTop = '10px';
    namesContainer.style.fontSize = '12px';
    namesContainer.style.color = 'var(--subtext0)';

    const colorNames = scheme.colors.map(color => getColorName(color));
    namesContainer.textContent = colorNames.join(' • ');
    card.appendChild(namesContainer);

    return card;
}

// Apply a color scheme to the current search
function applyColorScheme(scheme) {
    console.log('Applying color scheme:', scheme.name);

    // Clear existing color inputs
    const colorInputs = document.getElementById('colorInputs');
    if (!colorInputs) return;

    colorInputs.innerHTML = '';

    // Add colors from scheme
    scheme.colors.forEach((color, index) => {
        const weight = index === 0 ? 100 : Math.max(20, 100 - (index * 15)); // Decreasing weights
        addColorRow(color, weight);
    });

    // Update the color palette
    if (window.updateColorPalette) {
        window.updateColorPalette();
    }

    window.showSuccess('Scheme Applied', `Applied "${scheme.name}" color scheme with ${scheme.colors.length} colors.`);
}

// Helper function to add a color row
function addColorRow(color, weight = 100) {
    console.log('Adding color row for:', color);
    const colorInputs = document.getElementById('colorInputs');
    if (!colorInputs) {
        console.error('Color inputs container not found');
        return;
    }

    // Make sure color names are loaded
    if (!window.colorNamesLoaded) {
        console.log('Color names not loaded, loading now...');
        loadColorNames();
    }

    // Get color name and validate it
    const resolvedColorName = getColorName(color);
    console.log(`Color name for ${color}: "${resolvedColorName}"`);

    const colorRow = document.createElement('div');
    colorRow.className = 'color-row';
    colorRow.style.display = 'flex';
    colorRow.style.alignItems = 'center';
    colorRow.style.gap = '10px';
    colorRow.style.padding = '10px';
    colorRow.style.backgroundColor = 'var(--surface0)';
    colorRow.style.borderRadius = '6px';
    colorRow.style.marginBottom = '8px';

    // Create elements individually for better control
    const colorPicker = document.createElement('input');
    colorPicker.type = 'color';
    colorPicker.className = 'color-picker';
    colorPicker.value = color;
    colorPicker.id = 'color-picker-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    // Ensure it's accessible even when hidden
    colorPicker.setAttribute('data-color-picker', 'true');
    // We'll handle the event listener in a unified way below
    // Not adding event listener here to avoid duplicates

    // Color format dropdown
    const formatSelect = document.createElement('select');
    formatSelect.className = 'color-format-select';
    formatSelect.style.cssText = 'padding: 6px; border-radius: 4px; border: 1px solid var(--surface2); background: var(--surface0); color: var(--text); font-size: 13px;';
    ['HEX', 'RGB', 'HSL', 'HSV', 'CMYK'].forEach(format => {
        const option = document.createElement('option');
        option.value = format;
        option.textContent = format;
        formatSelect.appendChild(option);
    });
    formatSelect.value = 'HEX';

    // Color format input - increased width for long color formats
    const formatInput = document.createElement('input');
    formatInput.type = 'text';
    formatInput.className = 'color-format-input';
    formatInput.placeholder = '#FF0000';
    formatInput.value = color;
    formatInput.style.cssText = 'padding: 8px; border-radius: 6px; border: 1px solid var(--surface2); background: var(--surface0); color: var(--text); font-size: 13px; width: 250px; min-width: 250px; flex: 1;';

    // Randomize button
    const randomizeBtn = document.createElement('button');
    randomizeBtn.className = 'randomize-color-btn';
    randomizeBtn.textContent = '🔀';
    randomizeBtn.title = 'Randomize this color';
    randomizeBtn.style.cssText = 'background: var(--lavender); color: #000000; border: none; padding: 6px 10px; border-radius: 6px; cursor: pointer; font-size: 16px;';
    randomizeBtn.onclick = () => window.randomizeColorRow(randomizeBtn);

    // Update format input when color picker changes
    const updateFormatInput = () => {
        const currentFormat = formatSelect.value;
        const hexColor = colorPicker.value;
        formatInput.value = convertColorToFormat(hexColor, currentFormat);
    };

    // Update color picker when format input changes
    formatInput.addEventListener('input', () => {
        try {
            const format = formatSelect.value;
            const hex = convertFormatToHex(formatInput.value, format);
            if (hex) {
                colorPicker.value = hex;
                window.handleColorPickerChange({ target: colorPicker });
            }
        } catch (e) {
            console.warn('Invalid color format:', e);
        }
    });

    // Update format input when format dropdown changes
    formatSelect.addEventListener('change', updateFormatInput);

    // Update format input when color picker changes
    colorPicker.addEventListener('input', updateFormatInput);
    colorPicker.addEventListener('change', updateFormatInput);

    const weightSlider = document.createElement('input');
    weightSlider.type = 'range';
    weightSlider.className = 'weight-slider';
    weightSlider.min = '0';
    weightSlider.max = '100';
    weightSlider.value = weight;
    weightSlider.step = '1';

    const weightValue = document.createElement('span');
    weightValue.className = 'weight-value';
    weightValue.textContent = `${weight}%`;

    const colorName = document.createElement('span');
    colorName.className = 'color-name';
    colorName.style.fontSize = '12px';
    colorName.style.color = 'var(--subtext0)';
    colorName.style.marginLeft = '10px';
    colorName.style.fontStyle = 'italic';
    colorName.style.display = 'inline-block';
    colorName.textContent = resolvedColorName;

    const removeButton = document.createElement('button');
    removeButton.className = 'remove-btn';
    removeButton.textContent = 'Remove';
    removeButton.onclick = () => window.removeColor(removeButton);

    // Restructure layout: Create input container with label above and complementary below
    const inputContainer = document.createElement('div');
    inputContainer.style.display = 'flex';
    inputContainer.style.flexDirection = 'column';
    inputContainer.style.gap = '6px';
    inputContainer.style.marginRight = '10px';

    // Color name above input
    const colorNameLabel = document.createElement('div');
    colorNameLabel.style.fontSize = '11px';
    colorNameLabel.style.color = 'var(--subtext1)';
    colorNameLabel.style.fontWeight = '500';
    colorNameLabel.style.marginBottom = '2px';
    colorNameLabel.textContent = resolvedColorName;

    // Input wrapper
    const inputWrapper = document.createElement('div');
    inputWrapper.style.display = 'flex';
    inputWrapper.style.gap = '6px';
    inputWrapper.style.alignItems = 'center';

    // Hide default color picker and use custom simple picker
    // Keep it accessible for queries but visually hidden
    colorPicker.style.display = 'none';
    colorPicker.style.position = 'absolute';
    colorPicker.style.visibility = 'hidden';
    colorPicker.style.width = '0';
    colorPicker.style.height = '0';
    colorPicker.style.opacity = '0';
    colorPicker.style.pointerEvents = 'none';

    // Create simple color picker button
    const simpleColorPicker = document.createElement('button');
    simpleColorPicker.type = 'button';
    simpleColorPicker.className = 'simple-color-picker';
    simpleColorPicker.style.cssText = 'width: 50px; height: 40px; border: 2px solid var(--surface2); border-radius: 6px; cursor: pointer; background: ' + color + '; padding: 0; position: relative; overflow: hidden; flex-shrink: 0;';
    simpleColorPicker.title = 'Click to pick color';
    
    // Create simple color picker modal
    const createSimpleColorPicker = () => {
        const pickerModal = document.createElement('div');
        pickerModal.id = 'simpleColorPickerModal_' + Date.now();
        pickerModal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.7); z-index: 2000; display: flex; justify-content: center; align-items: center;';
        
        const pickerContent = document.createElement('div');
        pickerContent.style.cssText = 'background: var(--base); padding: 20px; border-radius: 12px; border: 2px solid var(--surface1); max-width: 400px; width: 90%;';
        
        const pickerCanvas = document.createElement('canvas');
        pickerCanvas.width = 300;
        pickerCanvas.height = 300;
        pickerCanvas.style.cssText = 'width: 100%; height: auto; border-radius: 8px; cursor: crosshair; border: 2px solid var(--surface2);';
        
        const lightnessSlider = document.createElement('input');
        lightnessSlider.type = 'range';
        lightnessSlider.min = '0';
        lightnessSlider.max = '100';
        lightnessSlider.value = '50';
        lightnessSlider.style.cssText = 'width: 100%; margin: 15px 0;';
        
        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'Close';
        closeBtn.style.cssText = 'background: var(--blue); color: #000000; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; width: 100%; margin-top: 10px;';
        closeBtn.onclick = () => pickerModal.remove();
        
        // Draw simple color wheel
        const ctx = pickerCanvas.getContext('2d');
        const centerX = pickerCanvas.width / 2;
        const centerY = pickerCanvas.height / 2;
        const radius = 140;
        
        // Helper function to convert HSL to RGB
        const hslToRgb = (h, s, l) => {
            let r, g, b;
            if (s === 0) {
                r = g = b = l;
            } else {
                const hue2rgb = (p, q, t) => {
                    if (t < 0) t += 1;
                    if (t > 1) t -= 1;
                    if (t < 1/6) return p + (q - p) * 6 * t;
                    if (t < 1/2) return q;
                    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                    return p;
                };
                const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
                const p = 2 * l - q;
                r = hue2rgb(p, q, h + 1/3);
                g = hue2rgb(p, q, h);
                b = hue2rgb(p, q, h - 1/3);
            }
            return {
                r: Math.round(r * 255),
                g: Math.round(g * 255),
                b: Math.round(b * 255)
            };
        };
        
        const drawColorWheel = (lightness) => {
            for (let angle = 0; angle < 360; angle += 2) {
                const hue = angle;
                for (let r = 0; r < radius; r += 3) {
                    const saturation = (r / radius) * 100;
                    const x = centerX + Math.cos(angle * Math.PI / 180) * r;
                    const y = centerY + Math.sin(angle * Math.PI / 180) * r;
                    const rgb = hslToRgb(hue / 360, saturation / 100, lightness / 100);
                    ctx.fillStyle = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
                    ctx.fillRect(x, y, 3, 3);
                }
            }
        };
        
        drawColorWheel(50);
        
        // Floating color display - shows current color while dragging
        const floatingColorDisplay = document.createElement('div');
        floatingColorDisplay.id = 'floatingColorDisplay_' + Date.now();
        floatingColorDisplay.style.cssText = 'position: fixed; top: 20px; left: 20px; background: var(--base); border: 2px solid var(--mauve); border-radius: 12px; padding: 15px; z-index: 2100; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5); display: none; min-width: 200px;';
        
        const colorSwatchDisplay = document.createElement('div');
        colorSwatchDisplay.style.cssText = 'width: 60px; height: 60px; border-radius: 8px; border: 2px solid var(--surface2); margin: 0 auto 10px;';
        
        const colorHexDisplay = document.createElement('div');
        colorHexDisplay.style.cssText = 'text-align: center; font-family: monospace; font-size: 18px; font-weight: bold; color: var(--text); margin-bottom: 5px;';
        
        const colorNameDisplay = document.createElement('div');
        colorNameDisplay.style.cssText = 'text-align: center; font-size: 12px; color: var(--subtext1); font-weight: 500;';
        
        floatingColorDisplay.appendChild(colorSwatchDisplay);
        floatingColorDisplay.appendChild(colorHexDisplay);
        floatingColorDisplay.appendChild(colorNameDisplay);
        document.body.appendChild(floatingColorDisplay);
        
        lightnessSlider.addEventListener('input', (e) => {
            ctx.clearRect(0, 0, pickerCanvas.width, pickerCanvas.height);
            drawColorWheel(e.target.value);
            // Update floating display if visible
            if (floatingColorDisplay.style.display === 'block' && isDragging) {
                // Recalculate color with new lightness
                const lastEvent = { clientX: 0, clientY: 0 }; // Will be updated on next mousemove
                // For now, just redraw - display will update on next mouse move
            }
        });
        
        // Helper to update color from canvas coordinates
        const updateColorFromCanvas = (e) => {
            const rect = pickerCanvas.getBoundingClientRect();
            let x = e.clientX - rect.left - centerX;
            let y = e.clientY - rect.top - centerY;
            
            // Clamp coordinates to canvas bounds for continuous dragging outside
            const distance = Math.sqrt(x * x + y * y);
            if (distance > radius) {
                // Clamp to edge of circle
                const angle = Math.atan2(y, x);
                x = Math.cos(angle) * radius;
                y = Math.sin(angle) * radius;
            }
            
            const clampedDistance = Math.sqrt(x * x + y * y);
            const angle = (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
            const hue = angle;
            const saturation = (clampedDistance / radius) * 100;
            const lightness = parseFloat(lightnessSlider.value);
            
            const rgb = hslToRgb(hue / 360, saturation / 100, lightness / 100);
            const hex = rgbToHex(rgb.r, rgb.g, rgb.b);
            
            // Update floating display with current color
            colorSwatchDisplay.style.backgroundColor = hex;
            colorHexDisplay.textContent = hex;
            colorNameDisplay.textContent = getColorName(hex);
            floatingColorDisplay.style.display = 'block';
            
            // Update actual color picker (but don't trigger full update while dragging)
            colorPicker.value = hex;
            formatInput.value = convertColorToFormat(hex, formatSelect.value);
            simpleColorPicker.style.background = hex;
            
            // Trigger updateSimplePicker if available to update color name and complementary
            // We need to get the colorRow from the colorPicker's parent
            const colorRow = colorPicker.closest('.color-row');
            if (colorRow && colorRow._updateSimplePicker) {
                // Call updateSimplePicker to update color name and complementary color
                try {
                    colorRow._updateSimplePicker();
                } catch (e) {
                    console.warn('[Modal Color Picker] Error calling updateSimplePicker:', e);
                }
            } else {
                console.warn('[Modal Color Picker] ColorRow or _updateSimplePicker not found', {
                    colorRow: !!colorRow,
                    hasUpdateFunction: colorRow ? !!colorRow._updateSimplePicker : false
                });
            }
        };
        
        let isDragging = false;
        
        // Global mouse move handler (defined before use)
        const handleGlobalMouseMove = (e) => {
            if (isDragging) {
                updateColorFromCanvas(e);
            }
        };
        
        pickerCanvas.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDragging = true;
            floatingColorDisplay.style.display = 'block';
            updateColorFromCanvas(e);
            // Add global listener for dragging outside canvas
            document.addEventListener('mousemove', handleGlobalMouseMove);
        });
        
        pickerCanvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                updateColorFromCanvas(e);
            }
        });
        
        pickerCanvas.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                document.removeEventListener('mousemove', handleGlobalMouseMove);
                
                // Final update: trigger all update functions
                const colorRow = colorPicker.closest('.color-row');
                if (colorRow && colorRow._updateSimplePicker) {
                    colorRow._updateSimplePicker();
                }
                
                // Manually trigger input and change events to ensure all listeners fire
                const inputEvent = new Event('input', { bubbles: true });
                const changeEvent = new Event('change', { bubbles: true });
                colorPicker.dispatchEvent(inputEvent);
                colorPicker.dispatchEvent(changeEvent);
                
                // Final update with full handler trigger
                if (window.handleColorPickerChange) {
                    window.handleColorPickerChange({ target: colorPicker });
                }
                
                floatingColorDisplay.style.display = 'none';
                floatingColorDisplay.remove();
                pickerModal.remove();
            }
        });
        
        // Global mouseup to handle release outside canvas
        const handleGlobalMouseUp = () => {
            if (isDragging) {
                isDragging = false;
                document.removeEventListener('mousemove', handleGlobalMouseMove);
                document.removeEventListener('mouseup', handleGlobalMouseUp);
                
                // Final update: trigger all update functions
                const colorRow = colorPicker.closest('.color-row');
                if (colorRow && colorRow._updateSimplePicker) {
                    colorRow._updateSimplePicker();
                }
                
                // Manually trigger input and change events to ensure all listeners fire
                const inputEvent = new Event('input', { bubbles: true });
                const changeEvent = new Event('change', { bubbles: true });
                colorPicker.dispatchEvent(inputEvent);
                colorPicker.dispatchEvent(changeEvent);
                
                // Final update with full handler trigger
                if (window.handleColorPickerChange) {
                    window.handleColorPickerChange({ target: colorPicker });
                }
                
                floatingColorDisplay.style.display = 'none';
                floatingColorDisplay.remove();
                pickerModal.remove();
            }
        };
        document.addEventListener('mouseup', handleGlobalMouseUp);
        
        pickerCanvas.addEventListener('mouseleave', () => {
            // Don't stop dragging - allow dragging outside canvas
            // Display will continue to update via global mousemove
        });
        
        // Also support click (without drag)
        pickerCanvas.addEventListener('click', (e) => {
            if (!isDragging) {
                updateColorFromCanvas(e);
                
                // Trigger all update functions
                const colorRow = colorPicker.closest('.color-row');
                if (colorRow && colorRow._updateSimplePicker) {
                    colorRow._updateSimplePicker();
                }
                
                // Manually trigger input and change events to ensure all listeners fire
                const inputEvent = new Event('input', { bubbles: true });
                const changeEvent = new Event('change', { bubbles: true });
                colorPicker.dispatchEvent(inputEvent);
                colorPicker.dispatchEvent(changeEvent);
                
                // Trigger full update
                if (window.handleColorPickerChange) {
                    window.handleColorPickerChange({ target: colorPicker });
                }
                
                floatingColorDisplay.style.display = 'none';
                floatingColorDisplay.remove();
                pickerModal.remove();
            }
        });
        
        // Update floating display when lightness changes
        lightnessSlider.addEventListener('input', (e) => {
            if (isDragging) {
                // Get current mouse position to update color
                // For now, just redraw wheel - the display will update on next mouse move
                ctx.clearRect(0, 0, pickerCanvas.width, pickerCanvas.height);
                drawColorWheel(e.target.value);
            }
        });
        
        // Close button should also hide floating display
        closeBtn.onclick = () => {
            floatingColorDisplay.style.display = 'none';
            floatingColorDisplay.remove();
            pickerModal.remove();
        };
        
        // Click outside modal to close
        pickerModal.addEventListener('click', (e) => {
            if (e.target === pickerModal) {
                document.removeEventListener('mousemove', handleGlobalMouseMove);
                floatingColorDisplay.style.display = 'none';
                floatingColorDisplay.remove();
                pickerModal.remove();
            }
        });
        
        // Clean up global listener when modal closes
        const originalCloseBtnOnclick = closeBtn.onclick;
        closeBtn.onclick = () => {
            document.removeEventListener('mousemove', handleGlobalMouseMove);
            floatingColorDisplay.style.display = 'none';
            floatingColorDisplay.remove();
            pickerModal.remove();
        };
        
        pickerContent.appendChild(pickerCanvas);
        pickerContent.appendChild(lightnessSlider);
        pickerContent.appendChild(closeBtn);
        pickerModal.appendChild(pickerContent);
        document.body.appendChild(pickerModal);
    };
    
    simpleColorPicker.onclick = createSimpleColorPicker;
    
    // Complementary color below input
    let complementaryColor = getComplementaryColor(color);
    let complementaryName = getColorName(complementaryColor);
    
    const complementaryContainer = document.createElement('div');
    complementaryContainer.className = 'complementary-color';
    complementaryContainer.style.display = 'flex';
    complementaryContainer.style.alignItems = 'center';
    complementaryContainer.style.gap = '6px';
    complementaryContainer.style.marginTop = '4px';
    complementaryContainer.style.cursor = 'pointer';
    complementaryContainer.title = 'Click to use complementary color';

    const complementarySwatch = document.createElement('div');
    complementarySwatch.style.width = '20px';
    complementarySwatch.style.height = '20px';
    complementarySwatch.style.backgroundColor = complementaryColor;
    complementarySwatch.style.border = '1px solid var(--surface2)';
    complementarySwatch.style.borderRadius = '4px';
    complementarySwatch.style.flexShrink = '0';

    const complementaryInfo = document.createElement('span');
    complementaryInfo.style.fontSize = '10px';
    complementaryInfo.style.color = 'var(--subtext0)';
    complementaryInfo.textContent = `Complementary: ${complementaryName}`;

    complementaryContainer.appendChild(complementarySwatch);
    complementaryContainer.appendChild(complementaryInfo);
    
    // Update complementary color when main color changes
    const updateComplementary = () => {
        complementaryColor = getComplementaryColor(colorPicker.value);
        complementaryName = getColorName(complementaryColor);
        complementarySwatch.style.backgroundColor = complementaryColor;
        complementaryInfo.textContent = `Complementary: ${complementaryName}`;
        complementaryContainer.title = 'Click to use complementary color: ' + complementaryColor;
    };
    
    // Update simple picker, color name, format input, and complementary when color changes
    const updateSimplePicker = () => {
        const currentColor = colorPicker.value;
        console.log('[updateSimplePicker] Updating for color:', currentColor);
        
        // Update simple color picker button
        simpleColorPicker.style.background = currentColor;
        
        // Update format input
        updateFormatInput();
        
        // Update color name label
        const newColorName = getColorName(currentColor);
        colorNameLabel.textContent = newColorName;
        console.log('[updateSimplePicker] Updated color name to:', newColorName);
        
        // Update complementary color
        updateComplementary();
        
        // Update global colors array
        if (window.updateColors) {
            window.updateColors();
        }
    };
    
    // Store updateSimplePicker reference on the colorRow for randomizeColorRow to use
    colorRow._updateSimplePicker = updateSimplePicker;
    
    // Listen to color picker changes (both input and change events)
    // Use capture phase to ensure we catch events even if stopPropagation is called
    colorPicker.addEventListener('input', updateSimplePicker, true);
    colorPicker.addEventListener('change', updateSimplePicker, true);
    
    // Click complementary to use it
    complementaryContainer.onclick = () => {
        colorPicker.value = complementaryColor;
        formatInput.value = convertColorToFormat(complementaryColor, formatSelect.value);
        simpleColorPicker.style.background = complementaryColor;
        updateComplementary(); // Update to get new complementary of the complementary
        if (window.handleColorPickerChange) {
            window.handleColorPickerChange({ target: colorPicker });
        }
    };

    // Build input container
    inputContainer.appendChild(colorNameLabel);
    inputWrapper.appendChild(simpleColorPicker);
    inputWrapper.appendChild(formatSelect);
    inputWrapper.appendChild(formatInput);
    inputContainer.appendChild(inputWrapper);
    inputContainer.appendChild(complementaryContainer);

    // IMPORTANT: Append the hidden colorPicker to the colorRow so it's in the DOM and findable
    // It's hidden but needs to be accessible for queries and event listeners
    colorRow.appendChild(colorPicker);

    // Append elements to row
    colorRow.appendChild(inputContainer);
    colorRow.appendChild(randomizeBtn);
    colorRow.appendChild(weightSlider);
    colorRow.appendChild(weightValue);
    colorRow.appendChild(removeButton);

    colorInputs.appendChild(colorRow);

    // Add event listeners to new elements
    // Use the actual colorPicker variable we created, not querySelector
    const newColorPicker = colorPicker; // Use the variable directly since we just created it
    const newWeightSlider = colorRow.querySelector('.weight-slider');
    const weightDisplay = colorRow.querySelector('.weight-value');

    if (newColorPicker) {
        console.log('[addColorRow] Attaching event listeners to color picker:', newColorPicker.id || 'no-id');
        // Attach the unified handler
        if (window.handleColorPickerChange) {
            newColorPicker.addEventListener('change', window.handleColorPickerChange);
            newColorPicker.addEventListener('input', window.handleColorPickerChange); // For live updates
        }
    } else {
        console.warn('[addColorRow] Color picker not found for event listeners!');
    }

    if (newWeightSlider && weightDisplay) {
        newWeightSlider.addEventListener('input', () => {
            weightDisplay.textContent = `${newWeightSlider.value}%`;
            window.updateColorPalette(); // This will also update colors and suggestions
        });
    }
}

// Generate color suggestions based on current selection
window.generateColorSuggestions = function () {
    console.log('Generating color suggestions');
    const currentColors = window.colors;
    if (!currentColors || currentColors.length === 0) {
        return [];
    }

    const suggestions = new Set();

    // Consider all colors for generating suggestions, not just the last one
    currentColors.forEach(color => {
        // Add complementary color for each existing color
        suggestions.add(getComplementaryColor(color));

        // Add colors from analogous scheme
        const analogousScheme = generateAnalogousScheme(color);
        if (analogousScheme && analogousScheme.colors) {
            analogousScheme.colors.forEach(c => suggestions.add(c));
        }

        // Add some colors from triadic scheme
        if (currentColors.length <= 4) {
            const triadicScheme = generateTriadicScheme(color);
            if (triadicScheme && triadicScheme.colors) {
                triadicScheme.colors.forEach(c => suggestions.add(c));
            }
        }

        // Add some monochromatic variations
        generateMonochromaticScheme(color).slice(0, 2).forEach(c => suggestions.add(c));
    });

    // Remove colors that are already selected
    currentColors.forEach(color => suggestions.delete(color));

    // Convert to array and limit to 5 suggestions
    return Array.from(suggestions).slice(0, 5);
};

// Update the suggestion display
window.updateColorSuggestions = function () {
    console.log('Updating color suggestions display');
    const container = document.getElementById('colorSuggestions');
    if (!container) {
        console.error('Color suggestions container not found');
        return;
    }

    // Ensure colors array is up to date by reading from the DOM
    window.updateColors();

    // Log the colors being used for suggestions
    console.log('Generating suggestions based on colors:', window.colors);

    // Generate suggestions
    const suggestions = window.generateColorSuggestions();
    container.innerHTML = '';

    if (suggestions.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'flex';
    const label = document.createElement('span');
    label.className = 'suggestion-label';
    label.textContent = 'Suggested colors : ';
    container.appendChild(label);

    suggestions.forEach(color => {
        const suggestionBtn = document.createElement('button');
        suggestionBtn.className = 'color-suggestion';
        suggestionBtn.style.backgroundColor = color;
        suggestionBtn.title = `Add ${getColorName(color)}`;

        // Add a tooltip with the color name
        const colorName = document.createElement('span');
        colorName.className = 'color-name-tooltip';
        colorName.textContent = getColorName(color);
        suggestionBtn.appendChild(colorName);

        suggestionBtn.addEventListener('click', () => {
            addColorRow(color);
            window.updateColorPalette();
            window.updateColorSuggestions();
        });

        container.appendChild(suggestionBtn);
    });
};

// Store the currently generated palette
window.generatedPalette = null;

// ============================================================================
// 3D VISUALIZATION MODULES - Dynamic Loading
// ============================================================================

/**
 * Dynamically load visualization modules and ensure they're available.
 * Modules are loaded from /static/js/modules/ and attach functions to window.
 */

// Track which modules are loaded
const visualizationModules = {
    colorSpaceNavigator: false,
    histogramCloud: false,
    similarityLandscape: false,
    rerankingAnimation: false,
    imageGlobe: false,
    connectionsGraph: false
};

/**
 * Load a single visualization module script
 */
function loadVisualizationModule(moduleName) {
    return new Promise((resolve, reject) => {
        // Check if already loaded
        if (visualizationModules[moduleName]) {
            resolve();
            return;
        }

        // Check if script already exists in DOM
        const existingScript = document.querySelector(`script[data-viz-module="${moduleName}"]`);
        if (existingScript) {
            visualizationModules[moduleName] = true;
            resolve();
            return;
        }

        // Create and load script
        const script = document.createElement('script');
        script.src = `/static/js/modules/${moduleName}.js`;
        script.type = 'text/javascript';
        script.setAttribute('data-viz-module', moduleName);

        script.onload = () => {
            visualizationModules[moduleName] = true;
            console.log(`✓ Visualization module loaded: ${moduleName}`);
            resolve();
        };

        script.onerror = () => {
            console.error(`✗ Failed to load visualization module: ${moduleName}`);
            reject(new Error(`Failed to load ${moduleName}`));
        };

        document.head.appendChild(script);
    });
}

/**
 * Load all visualization modules when page loads
 */
async function loadAllVisualizationModules() {
    const modules = [
        'colorSpaceNavigator',
        'histogramCloud',
        'similarityLandscape',
        'rerankingAnimation',
        'imageGlobe',
        'connectionsGraph',
        'otTransport3D',
        'hnswGraphExplorer',
        'colorDensityVolume',
        'imageThumbnails3D'
    ];

    try {
        console.log('Loading 3D visualization modules...');
        await Promise.all(modules.map(module => loadVisualizationModule(module)));
        console.log('✓ All visualization modules loaded successfully');
    } catch (error) {
        console.error('Error loading visualization modules:', error);
    }
}

// Load modules when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadAllVisualizationModules);
} else {
    loadAllVisualizationModules();
}

/**
 * Function wrappers that ensure modules are loaded before executing.
 * These act as proxies until the actual module functions are loaded.
 * Once loaded, the modules define the actual functions on window.
 */

// Generate function wrappers
// These ensure modules are loaded before execution.
// When a module loads, it overwrites the corresponding window function with its implementation.
// After loadVisualizationModule completes, the function on window is the module's version.
window.generateColorSpaceNavigator = async function () {
    await loadVisualizationModule('colorSpaceNavigator');
    // Scroll to Interactive 3D Visualizations h2 (almost at top)
    scrollTo3DVisualizationsHeader();
    // Module has loaded and overwritten this function, so call it
    // Store reference before calling to avoid recursion
    const func = window.generateColorSpaceNavigator;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.generateHistogramCloud = async function () {
    await loadVisualizationModule('histogramCloud');
    scrollTo3DVisualizationsHeader();
    const func = window.generateHistogramCloud;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.generateSimilarityLandscape = async function () {
    await loadVisualizationModule('similarityLandscape');
    scrollTo3DVisualizationsHeader();
    const func = window.generateSimilarityLandscape;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.generateRerankingAnimation = async function () {
    await loadVisualizationModule('rerankingAnimation');
    scrollTo3DVisualizationsHeader();
    const func = window.generateRerankingAnimation;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

// ============================================================================
// ADVANCED FEATURES HELPER FUNCTIONS
// ============================================================================

/**
 * Show/hide advanced filters panel
 */
window.showAdvancedFilters = function() {
    const panel = document.getElementById('advancedFilters');
    if (panel) {
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }
};

window.toggleAdvancedFilters = window.showAdvancedFilters;

/**
 * Show color wheel modal
 */
window.showColorWheel = function() {
    const modal = document.getElementById('colorWheelModal') || createColorWheelModal();
    modal.style.display = 'block';
    setTimeout(() => {
        const container = document.getElementById('colorWheelContainer');
        if (container && window.createColorWheel) {
            window.createColorWheel('colorWheelContainer');
        }
    }, 100);
};

/**
 * Create color wheel modal
 */
function createColorWheelModal() {
    const modal = document.createElement('div');
    modal.id = 'colorWheelModal';
    modal.style.cssText = 'display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); overflow-y: auto;';
    
    const content = document.createElement('div');
    content.style.cssText = 'background: var(--base); margin: 5% auto; padding: 30px; border: 1px solid var(--surface2); width: 90%; max-width: 600px; border-radius: 12px; max-height: 90vh; overflow-y: auto;';
    
    content.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h3 style="margin: 0; color: var(--text);">🎨 Interactive Color Wheel (HSL/HSV)</h3>
            <button onclick="document.getElementById('colorWheelModal').style.display='none'" 
                style="background: var(--red); color: #000000; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer;">Close</button>
        </div>
        <div id="colorWheelContainer" style="text-align: center;"></div>
    `;
    
    modal.appendChild(content);
    document.body.appendChild(modal);
    return modal;
}

/**
 * Show export options modal
 */
window.showExportOptions = function() {
    const modal = document.getElementById('exportOptionsModal') || createExportOptionsModal();
    modal.style.display = 'block';
};

/**
 * Create export options modal
 */
function createExportOptionsModal() {
    const modal = document.createElement('div');
    modal.id = 'exportOptionsModal';
    modal.style.cssText = 'display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7);';
    
    const content = document.createElement('div');
    content.style.cssText = 'background: var(--base); margin: 5% auto; padding: 30px; border: 1px solid var(--surface2); width: 90%; max-width: 700px; border-radius: 12px; max-height: 90vh; overflow-y: auto;';
    
    content.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h3 style="margin: 0; color: var(--text);">📤 Export Palette</h3>
            <button onclick="document.getElementById('exportOptionsModal').style.display='none'" 
                style="background: var(--red); color: #000000; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer;">Close</button>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <button onclick="exportPalette('css'); document.getElementById('exportOptionsModal').style.display='none';" 
                style="background: var(--blue); color: #000000; border: none; padding: 20px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: bold;">
                CSS Variables
            </button>
            <button onclick="exportPalette('scss'); document.getElementById('exportOptionsModal').style.display='none';" 
                style="background: var(--mauve); color: #000000; border: none; padding: 20px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: bold;">
                SCSS/SASS
            </button>
            <button onclick="exportPalette('json'); document.getElementById('exportOptionsModal').style.display='none';" 
                style="background: var(--green); color: #000000; border: none; padding: 20px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: bold;">
                JSON
            </button>
            <button onclick="exportPalette('ase'); document.getElementById('exportOptionsModal').style.display='none';" 
                style="background: var(--red); color: #000000; border: none; padding: 20px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: bold;">
                Adobe Swatch (.ase)
            </button>
            <button onclick="exportPalette('sketch'); document.getElementById('exportOptionsModal').style.display='none';" 
                style="background: var(--yellow); color: var(--base); border: none; padding: 20px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: bold;">
                Sketch
            </button>
            <button onclick="exportPalette('figma'); document.getElementById('exportOptionsModal').style.display='none';" 
                style="background: var(--teal); color: #000000; border: none; padding: 20px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: bold;">
                Figma
            </button>
        </div>
        <div style="margin-top: 20px; padding: 15px; background: var(--surface0); border-radius: 8px;">
            <h4 style="color: var(--text); margin: 0 0 10px 0;">Export Grid/Palette Image</h4>
            <p style="color: var(--subtext1); font-size: 13px; margin: 0 0 15px 0;">Export search results or palette as an image</p>
            <button onclick="exportSearchResultsAsImage()" 
                style="background: var(--lavender); color: #000000; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: bold; margin-right: 10px;">
                Export Search Results Grid
            </button>
            <button onclick="exportPaletteAsImage()" 
                style="background: var(--sapphire); color: #000000; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: bold;">
                Export Palette Image
            </button>
        </div>
    `;
    
    modal.appendChild(content);
    document.body.appendChild(modal);
    return modal;
}

/**
 * Show gradient options modal
 */
window.showGradientOptions = function() {
    const modal = document.getElementById('gradientOptionsModal') || createGradientOptionsModal();
    modal.style.display = 'block';
};

/**
 * Create gradient options modal
 */
function createGradientOptionsModal() {
    const modal = document.createElement('div');
    modal.id = 'gradientOptionsModal';
    modal.style.cssText = 'display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7);';
    
    const content = document.createElement('div');
    content.style.cssText = 'background: var(--base); margin: 5% auto; padding: 30px; border: 1px solid var(--surface2); width: 90%; max-width: 600px; border-radius: 12px; max-height: 90vh; overflow-y: auto;';
    
    content.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h3 style="margin: 0; color: var(--text);">🌈 Generate Gradient</h3>
            <button onclick="document.getElementById('gradientOptionsModal').style.display='none'" 
                style="background: var(--red); color: #000000; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer;">Close</button>
        </div>
        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 8px; color: var(--subtext1); font-size: 14px;">Gradient Type:</label>
            <select id="gradientType" style="width: 100%; padding: 10px; border-radius: 6px; border: 1px solid var(--surface2); background: var(--surface0); color: var(--text);">
                <option value="css">CSS String</option>
                <option value="linear">Linear Image</option>
                <option value="radial">Radial Image</option>
            </select>
        </div>
        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 8px; color: var(--subtext1); font-size: 14px;">Width: <span id="gradientWidthValue">800</span>px</label>
            <input type="range" id="gradientWidth" min="200" max="2000" step="50" value="800" style="width: 100%;"
                oninput="document.getElementById('gradientWidthValue').textContent = this.value;">
        </div>
        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 8px; color: var(--subtext1); font-size: 14px;">Height: <span id="gradientHeightValue">200</span>px</label>
            <input type="range" id="gradientHeight" min="100" max="1000" step="50" value="200" style="width: 100%;"
                oninput="document.getElementById('gradientHeightValue').textContent = this.value;">
        </div>
        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 8px; color: var(--subtext1); font-size: 14px;">Direction:</label>
            <select id="gradientDirection" style="width: 100%; padding: 10px; border-radius: 6px; border: 1px solid var(--surface2); background: var(--surface0); color: var(--text);">
                <option value="horizontal">Horizontal</option>
                <option value="vertical">Vertical</option>
                <option value="diagonal">Diagonal</option>
            </select>
        </div>
        <button onclick="generateGradientFromModal()" 
            style="width: 100%; background: var(--teal); color: #000000; border: none; padding: 15px; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold;">
            Generate Gradient
        </button>
    `;
    
    modal.appendChild(content);
    document.body.appendChild(modal);
    return modal;
}

/**
 * Generate gradient from modal inputs
 */
window.generateGradientFromModal = function() {
    const type = document.getElementById('gradientType').value;
    const width = parseInt(document.getElementById('gradientWidth').value);
    const height = parseInt(document.getElementById('gradientHeight').value);
    const direction = document.getElementById('gradientDirection').value;
    
    if (window.generateGradient) {
        window.generateGradient(type, width, height, direction);
    }
    document.getElementById('gradientOptionsModal').style.display = 'none';
};

/**
 * Load preset palette
 */
window.loadPreset = function(name, colors) {
    // Clear current colors
    const colorInputs = document.getElementById('colorInputs');
    if (colorInputs) {
        colorInputs.innerHTML = '';
    }
    
    // Add preset colors
    colors.forEach((color, index) => {
        const weight = Math.floor(100 / colors.length);
        if (window.addColorRow) {
            window.addColorRow(`#${color}`, weight);
        }
    });
    
    if (window.updateColorPalette) {
        window.updateColorPalette();
    }
    
    showSuccess('Preset Loaded', `${name} palette loaded successfully!`);
};

/**
 * Export search results as image
 */
window.exportSearchResultsAsImage = async function() {
    try {
        const resultsGrid = document.getElementById('resultsGrid');
        if (!resultsGrid || resultsGrid.children.length === 0) {
            showError('Export Error', 'No search results to export');
            return;
        }
        
        if (typeof html2canvas === 'undefined') {
            showError('Export Error', 'html2canvas library not loaded');
            return;
        }
        
        const canvas = await html2canvas(resultsGrid, {
            backgroundColor: 'var(--base)',
            scale: 2,
            logging: false
        });
        
        const dataUrl = canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = dataUrl;
        a.download = `search-results-${Date.now()}.png`;
        a.click();
        
        showSuccess('Export Success', 'Search results exported as image');
    } catch (error) {
        console.error('Export error:', error);
        showError('Export Error', error.message);
    }
};

/**
 * Export palette as image
 */
window.exportPaletteAsImage = async function() {
    try {
        const colors = window.colors || [];
        if (colors.length === 0) {
            showError('Export Error', 'No colors to export');
            return;
        }
        
        const canvas = document.createElement('canvas');
        canvas.width = 800;
        canvas.height = 200;
        const ctx = canvas.getContext('2d');
        
        const totalWeight = window.weights ? window.weights.reduce((a, b) => a + b, 0) : colors.length * 100;
        
        let x = 0;
        colors.forEach((color, index) => {
            const weight = window.weights ? window.weights[index] : 100;
            const width = (weight / totalWeight) * canvas.width;
            
            ctx.fillStyle = color;
            ctx.fillRect(x, 0, width, canvas.height);
            
            x += width;
        });
        
        const dataUrl = canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = dataUrl;
        a.download = `palette-${Date.now()}.png`;
        a.click();
        
        showSuccess('Export Success', 'Palette exported as image');
    } catch (error) {
        console.error('Export error:', error);
        showError('Export Error', error.message);
    }
};

window.generateImageGlobe = async function () {
    await loadVisualizationModule('imageGlobe');
    scrollTo3DVisualizationsHeader();
    const func = window.generateImageGlobe;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.generateConnectionsGraph = async function () {
    await loadVisualizationModule('connectionsGraph');
    scrollTo3DVisualizationsHeader();
    const func = window.generateConnectionsGraph;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

// Export function wrappers
window.exportColorSpaceData = async function () {
    await loadVisualizationModule('colorSpaceNavigator');
    const func = window.exportColorSpaceData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.exportHistogramData = async function () {
    await loadVisualizationModule('histogramCloud');
    const func = window.exportHistogramData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.exportSimilarityData = async function () {
    await loadVisualizationModule('similarityLandscape');
    const func = window.exportSimilarityData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.exportAnimationData = async function () {
    await loadVisualizationModule('rerankingAnimation');
    const func = window.exportAnimationData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.exportGlobeData = async function () {
    await loadVisualizationModule('imageGlobe');
    const func = window.exportGlobeData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.exportGraphData = async function () {
    await loadVisualizationModule('connectionsGraph');
    const func = window.exportGraphData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

// New visualizations wrappers
window.generateOTTransport3D = async function () { await loadVisualizationModule('otTransport3D'); scrollTo3DVisualizationsHeader(); const func = window.generateOTTransport3D; if (func && func !== arguments.callee) { return func.apply(this, arguments); } };
window.exportOTTransportData = async function () { await loadVisualizationModule('otTransport3D'); const func = window.exportOTTransportData; if (func && func !== arguments.callee) { return func.apply(this, arguments); } };

window.generateHNSWExplorer = async function () { await loadVisualizationModule('hnswGraphExplorer'); scrollTo3DVisualizationsHeader(); const func = window.generateHNSWExplorer; if (func && func !== arguments.callee) { return func.apply(this, arguments); } };
window.exportHNSWData = async function () { await loadVisualizationModule('hnswGraphExplorer'); const func = window.exportHNSWData; if (func && func !== arguments.callee) { return func.apply(this, arguments); } };

window.generateColorDensityVolume = async function () { await loadVisualizationModule('colorDensityVolume'); scrollTo3DVisualizationsHeader(); const func = window.generateColorDensityVolume; if (func && func !== arguments.callee) { return func.apply(this, arguments); } };
window.exportColorDensityData = async function () { await loadVisualizationModule('colorDensityVolume'); const func = window.exportColorDensityData; if (func && func !== arguments.callee) { return func.apply(this, arguments); } };

window.generateImageThumbnails3D = async function () { await loadVisualizationModule('imageThumbnails3D'); scrollTo3DVisualizationsHeader(); const func = window.generateImageThumbnails3D; if (func && func !== arguments.callee) { return func.apply(this, arguments); } };
window.exportThumbnailsData = async function () { await loadVisualizationModule('imageThumbnails3D'); const func = window.exportThumbnailsData; if (func && func !== arguments.callee) { return func.apply(this, arguments); } };

// ============================================================================
// 3D VISUALIZATION SHARED UTILITIES
// ============================================================================

/**
 * Scroll to center the 3D visualization container vertically in the browser window
 */
function scrollTo3DVisualizationsHeader() {
    const container = document.getElementById('visualization3d-container');
    if (container) {
        setTimeout(() => {
            const containerRect = container.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            // Calculate scroll position to center the container vertically
            const scrollPosition = window.scrollY + containerRect.top - (windowHeight / 2) + (containerRect.height / 2);
            window.scrollTo({ top: Math.max(0, scrollPosition), behavior: 'smooth' });
        }, 100);
    }
}

// Global state for current visualization
window.current3DVisualization = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    animationId: null,
    isPaused: false
};

/**
 * Clear any existing 3D visualization
 */
window.clear3DVisualization = function () {
    console.log('[3D Visualization] Clearing previous visualization...');

    // Stop any running animations
    if (window.current3DVisualization.animationId) {
        cancelAnimationFrame(window.current3DVisualization.animationId);
        window.current3DVisualization.animationId = null;
    }

    // Dispose of Three.js resources
    if (window.current3DVisualization.scene) {
        // Remove all objects from scene
        while (window.current3DVisualization.scene.children.length > 0) {
            const child = window.current3DVisualization.scene.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
            window.current3DVisualization.scene.remove(child);
        }
    }

    // Remove renderer from DOM
    const container = document.getElementById('visualization3d-container');
    if (container && window.current3DVisualization.renderer) {
        const canvas = window.current3DVisualization.renderer.domElement;
        if (canvas && canvas.parentNode === container) {
            container.removeChild(canvas);
        }
    }

    // Dispose renderer
    if (window.current3DVisualization.renderer) {
        window.current3DVisualization.renderer.dispose();
    }

    // Clear references
    window.current3DVisualization.scene = null;
    window.current3DVisualization.camera = null;
    window.current3DVisualization.renderer = null;
    window.current3DVisualization.controls = null;
    window.current3DVisualization.isPaused = false;

    // Show placeholder
    const placeholder = document.getElementById('3d-placeholder');
    const loading = document.getElementById('3d-loading');
    if (placeholder) placeholder.style.display = 'block';
    if (loading) loading.style.display = 'none';

    console.log('[3D Visualization] Previous visualization cleared');
};

/**
 * Show loading indicator
 */
window.show3DLoading = function () {
    const loading = document.getElementById('3d-loading');
    const placeholder = document.getElementById('3d-placeholder');
    if (loading) loading.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
};

/**
 * Hide loading indicator
 */
window.hide3DLoading = function () {
    const loading = document.getElementById('3d-loading');
    const placeholder = document.getElementById('3d-placeholder');
    if (loading) loading.style.display = 'none';
    if (placeholder) placeholder.style.display = 'none';
};

/**
 * Get container for 3D visualization
 */
window.get3DContainer = function () {
    return document.getElementById('visualization3d-container');
};

// ============================================================================
// RANDOM VALUE GENERATORS
// ============================================================================

/**
 * Generate random hex color
 */
function randomHexColor() {
    return Math.floor(Math.random() * 16777215).toString(16).toUpperCase().padStart(6, '0');
}

/**
 * Generate random colors string
 */
window.randomSimilarityColors = function () {
    const numColors = Math.floor(Math.random() * 3) + 1; // 1-3 colors
    const colors = Array.from({ length: numColors }, () => randomHexColor());
    document.getElementById('similarityColors').value = colors.join(',');
    console.log('[Random] Generated similarity colors:', colors.join(','));
};

window.randomAnimationColors = function () {
    const numColors = Math.floor(Math.random() * 3) + 1;
    const colors = Array.from({ length: numColors }, () => randomHexColor());
    document.getElementById('animationColors').value = colors.join(',');
    console.log('[Random] Generated animation colors:', colors.join(','));
};

/**
 * Generate random weights string
 */
window.randomSimilarityWeights = function () {
    const numWeights = document.getElementById('similarityColors').value.split(',').length || 1;
    const weights = Array.from({ length: numWeights }, () => (Math.random() * 0.8 + 0.1).toFixed(2));
    const sum = weights.reduce((a, b) => parseFloat(a) + parseFloat(b), 0);
    const normalized = weights.map(w => (parseFloat(w) / sum).toFixed(3));
    document.getElementById('similarityWeights').value = normalized.join(',');
    console.log('[Random] Generated similarity weights:', normalized.join(','));
};

window.randomAnimationWeights = function () {
    const numWeights = document.getElementById('animationColors').value.split(',').length || 1;
    const weights = Array.from({ length: numWeights }, () => (Math.random() * 0.8 + 0.1).toFixed(2));
    const sum = weights.reduce((a, b) => parseFloat(a) + parseFloat(b), 0);
    const normalized = weights.map(w => (parseFloat(w) / sum).toFixed(3));
    document.getElementById('animationWeights').value = normalized.join(',');
    console.log('[Random] Generated animation weights:', normalized.join(','));
};

/**
 * Generate random image ID (fetches from search results)
 */
window.randomHistogramImageId = async function () {
    try {
        console.log('[Random] Fetching random image ID...');
        const response = await fetch(`/search?colors=808080&weights=1.0&k=50&fast_mode=true`);
        if (!response.ok) throw new Error('Failed to fetch images');
        const data = await response.json();
        if (data.results && data.results.length > 0) {
            const randomResult = data.results[Math.floor(Math.random() * data.results.length)];
            document.getElementById('histogramImageId').value = randomResult.image_id;
            console.log('[Random] Generated histogram image ID:', randomResult.image_id);
        } else {
            throw new Error('No results available');
        }
    } catch (error) {
        console.error('[Random] Error fetching random image ID:', error);
        alert('Failed to fetch random image ID. Please enter one manually.');
    }
};

// Random for OT Transport 3D
window.randomOTColors = function () {
    const numColors = Math.floor(Math.random() * 3) + 1;
    const colors = Array.from({ length: numColors }, () => randomHexColor());
    const el = document.getElementById('otColors'); if (el) el.value = colors.join(',');
    console.log('[Random] Generated OT colors:', colors.join(','));
};
window.randomOTWeights = function () {
    const el = document.getElementById('otColors');
    const num = el?.value ? el.value.split(',').length : 1;
    const ws = Array.from({ length: num }, () => (Math.random() * 0.8 + 0.1).toFixed(2));
    const sum = ws.reduce((a, b) => parseFloat(a) + parseFloat(b), 0);
    const norm = ws.map(w => (parseFloat(w) / sum).toFixed(3));
    const we = document.getElementById('otWeights'); if (we) we.value = norm.join(',');
    console.log('[Random] Generated OT weights:', norm.join(','));
};

// Random for HNSW Explorer
window.randomHNSWColors = function () {
    const numColors = Math.floor(Math.random() * 3) + 1;
    const colors = Array.from({ length: numColors }, () => randomHexColor());
    const el = document.getElementById('hnswColors'); if (el) el.value = colors.join(',');
    console.log('[Random] Generated HNSW colors:', colors.join(','));
};
window.randomHNSWWeights = function () {
    const el = document.getElementById('hnswColors');
    const num = el?.value ? el.value.split(',').length : 1;
    const ws = Array.from({ length: num }, () => (Math.random() * 0.8 + 0.1).toFixed(2));
    const sum = ws.reduce((a, b) => parseFloat(a) + parseFloat(b), 0);
    const norm = ws.map(w => (parseFloat(w) / sum).toFixed(3));
    const we = document.getElementById('hnswWeights'); if (we) we.value = norm.join(',');
    console.log('[Random] Generated HNSW weights:', norm.join(','));
};

// ============================================================================
// 3D VISUALIZATION CONTROLS
// ============================================================================

window.toggle3DVisualization = function () {
    window.current3DVisualization.isPaused = !window.current3DVisualization.isPaused;
    console.log('[3D Controls] Visualization', window.current3DVisualization.isPaused ? 'paused' : 'resumed');
};

window.reset3DVisualization = function () {
    if (window.current3DVisualization.camera && window.current3DVisualization.controls) {
        window.current3DVisualization.camera.position.set(150, 150, 150);
        window.current3DVisualization.camera.lookAt(0, 0, 0);
        // Reset target for OrbitControls
        if (window.current3DVisualization.controls.target) {
            window.current3DVisualization.controls.target.set(0, 0, 0);
        }
        window.current3DVisualization.controls.update();
        console.log('[3D Controls] View reset');
    }
};

window.turnOff3DVisualization = function () {
    console.log('[3D Controls] Turning off visualization');
    window.clear3DVisualization();
};

// Show the suggest palette modal
window.showSuggestPaletteModal = function () {
    const modal = document.getElementById('suggestPaletteModal');
    if (modal) {
        // Update current colors display
        window.updateColors();
        const currentColors = window.colors || [];
        const currentColorsDisplay = document.getElementById('currentColorsDisplay');
        
        if (currentColorsDisplay) {
            currentColorsDisplay.innerHTML = '';
            if (currentColors.length === 0) {
                currentColorsDisplay.innerHTML = '<span style="color: var(--subtext0); font-style: italic;">No colors selected</span>';
            } else {
                currentColors.forEach(color => {
                    const colorSwatch = document.createElement('div');
                    colorSwatch.style.width = '40px';
                    colorSwatch.style.height = '40px';
                    colorSwatch.style.backgroundColor = color;
                    colorSwatch.style.borderRadius = '6px';
                    colorSwatch.style.border = '2px solid var(--surface2)';
                    colorSwatch.style.cursor = 'pointer';
                    colorSwatch.style.position = 'relative';
                    colorSwatch.title = color;
                    
                    // Add hex label on hover
                    const hexLabel = document.createElement('div');
                    hexLabel.style.position = 'absolute';
                    hexLabel.style.bottom = '-25px';
                    hexLabel.style.left = '50%';
                    hexLabel.style.transform = 'translateX(-50%)';
                    hexLabel.style.background = 'var(--yellow)';
                    hexLabel.style.color = '#000000';
                    hexLabel.style.padding = '2px 6px';
                    hexLabel.style.borderRadius = '4px';
                    hexLabel.style.fontSize = '10px';
                    hexLabel.style.whiteSpace = 'nowrap';
                    hexLabel.style.display = 'none';
                    hexLabel.textContent = color;
                    colorSwatch.appendChild(hexLabel);
                    
                    colorSwatch.addEventListener('mouseenter', () => {
                        hexLabel.style.display = 'block';
                    });
                    colorSwatch.addEventListener('mouseleave', () => {
                        hexLabel.style.display = 'none';
                    });
                    
                    currentColorsDisplay.appendChild(colorSwatch);
                });
            }
        }
        
        // Reset any previously generated palette
        const generatedPaletteDiv = document.getElementById('generatedPalette');
        if (generatedPaletteDiv) {
            generatedPaletteDiv.style.display = 'none';
        }
        window.generatedPalette = null;

        // Add hover effects to scheme modes
        const schemeModes = modal.querySelectorAll('.scheme-mode');
        schemeModes.forEach(mode => {
            mode.addEventListener('mouseenter', () => {
                mode.style.transform = 'translateY(-2px) scale(1.02)';
                mode.style.borderColor = 'var(--mauve)';
                mode.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.2)';
            });
            mode.addEventListener('mouseleave', () => {
                mode.style.transform = 'translateY(0) scale(1)';
                mode.style.borderColor = 'var(--surface2)';
                mode.style.boxShadow = 'none';
            });
        });

        modal.style.display = 'block';
    }
};

// Generate a new color palette based on the selected scheme
window.generatePalette = function (scheme) {
    console.log('Generating palette for scheme:', scheme);

    // Get the currently selected colors - use ALL colors, not just first
    window.updateColors();
    const currentColors = window.colors || [];
    
    // Use the primary (first) color as the base, but consider all colors for correlation
    let baseColor = currentColors[0] || '#FF0000';
    if (baseColor.startsWith('#')) {
        baseColor = baseColor.substring(1);
    }

    let newColors = [];
    switch (scheme) {
        case 'monochromatic':
            newColors = generateMonochromaticScheme(baseColor);
            // If user has multiple colors, add variations that correlate with them
            if (currentColors.length > 1) {
                newColors = enhancePaletteWithCorrelation(newColors, currentColors, 'monochromatic');
            }
            break;
        case 'complementary':
            newColors = [baseColor, getComplementaryColor(baseColor)];
            // If user has multiple colors, try to match their complementary relationships
            if (currentColors.length > 1) {
                newColors = enhancePaletteWithCorrelation(newColors, currentColors, 'complementary');
            }
            break;
        case 'analogous':
            newColors = generateAnalogousScheme(baseColor);
            // Correlate with user's existing analogous colors
            if (currentColors.length > 1) {
                newColors = enhancePaletteWithCorrelation(newColors, currentColors, 'analogous');
            }
            break;
        case 'triadic':
            newColors = generateTriadicScheme(baseColor);
            // Correlate with user's triadic colors if they exist
            if (currentColors.length > 1) {
                newColors = enhancePaletteWithCorrelation(newColors, currentColors, 'triadic');
            }
            break;
        case 'split-complementary':
            // Generate split complementary colors
            const complementary = getComplementaryColor(baseColor);
            const hsl = hexToHsl(complementary);
            newColors = [
                baseColor,
                hslToHex((hsl.h + 30) % 360, hsl.s, hsl.l),
                hslToHex((hsl.h - 30 + 360) % 360, hsl.s, hsl.l)
            ];
            // Correlate with user's colors
            if (currentColors.length > 1) {
                newColors = enhancePaletteWithCorrelation(newColors, currentColors, 'split-complementary');
            }
            break;
        case 'random':
            // Generate a random harmonious palette, but try to correlate with user colors
            if (currentColors.length > 0) {
                // Use the average hue of user's colors
                const avgHsl = getAverageHsl(currentColors);
                const randomHue = avgHsl ? avgHsl.h : Math.floor(Math.random() * 360);
                newColors = [
                    hslToHex(randomHue, 70, 50),
                    hslToHex((randomHue + 120) % 360, 70, 50),
                    hslToHex((randomHue + 240) % 360, 70, 50),
                    hslToHex(randomHue, 85, 35),
                    hslToHex(randomHue, 60, 65)
                ];
            } else {
                const randomHue = Math.floor(Math.random() * 360);
                newColors = [
                    hslToHex(randomHue, 70, 50),
                    hslToHex((randomHue + 120) % 360, 70, 50),
                    hslToHex((randomHue + 240) % 360, 70, 50),
                    hslToHex(randomHue, 85, 35),
                    hslToHex(randomHue, 60, 65)
                ];
            }
            break;
    }

    // Ensure all colors have # prefix
    newColors = newColors.map(color => {
        if (!color.startsWith('#')) return '#' + color;
        return color;
    });

    // Store the generated palette
    window.generatedPalette = newColors;

    // Show the preview
    const previewDiv = document.getElementById('palettePreview');
    const generatedPaletteDiv = document.getElementById('generatedPalette');
    const paletteCount = document.getElementById('paletteCount');

    if (previewDiv && generatedPaletteDiv) {
        // Clear previous preview
        previewDiv.innerHTML = '';

        // Update palette count
        if (paletteCount) {
            paletteCount.textContent = `${newColors.length} colors`;
        }

        // Create swatches for each color
        newColors.forEach((color, index) => {
            const swatch = document.createElement('div');
            swatch.style.flex = '1';
            swatch.style.minWidth = '80px';
            swatch.style.height = '80px';
            swatch.style.borderRadius = '8px';
            swatch.style.backgroundColor = color;
            swatch.style.position = 'relative';
            swatch.style.border = '2px solid var(--surface2)';
            swatch.style.cursor = 'pointer';
            swatch.style.transition = 'all 0.2s';
            swatch.title = `${color} - ${getColorName(color)}`;

            // Add color info overlay
            const colorInfo = document.createElement('div');
            colorInfo.style.position = 'absolute';
            colorInfo.style.bottom = '0';
            colorInfo.style.left = '0';
            colorInfo.style.right = '0';
            colorInfo.style.background = 'var(--yellow)';
            colorInfo.style.color = '#000000';
            colorInfo.style.padding = '6px';
            colorInfo.style.fontSize = '11px';
            colorInfo.style.textAlign = 'center';
            colorInfo.style.borderRadius = '0 0 6px 6px';
            colorInfo.style.fontFamily = 'monospace';
            colorInfo.textContent = color;

            swatch.appendChild(colorInfo);

            // Hover effect
            swatch.addEventListener('mouseenter', () => {
                swatch.style.transform = 'translateY(-4px) scale(1.05)';
                swatch.style.boxShadow = '0 6px 16px rgba(0, 0, 0, 0.3)';
                swatch.style.borderColor = 'var(--mauve)';
            });
            swatch.addEventListener('mouseleave', () => {
                swatch.style.transform = 'translateY(0) scale(1)';
                swatch.style.boxShadow = 'none';
                swatch.style.borderColor = 'var(--surface2)';
            });

            // Click to copy
            swatch.addEventListener('click', () => {
                navigator.clipboard.writeText(color);
                swatch.style.transform = 'scale(1.1)';
                setTimeout(() => {
                    swatch.style.transform = '';
                }, 200);
                // Show temporary feedback
                const feedback = document.createElement('div');
                feedback.textContent = '✓ Copied!';
                feedback.style.position = 'absolute';
                feedback.style.top = '50%';
                feedback.style.left = '50%';
                feedback.style.transform = 'translate(-50%, -50%)';
                feedback.style.background = 'var(--green)';
                feedback.style.color = '#000000';
                feedback.style.padding = '4px 8px';
                feedback.style.borderRadius = '4px';
                feedback.style.fontSize = '12px';
                feedback.style.fontWeight = 'bold';
                feedback.style.pointerEvents = 'none';
                swatch.appendChild(feedback);
                setTimeout(() => feedback.remove(), 1000);
            });

            previewDiv.appendChild(swatch);
        });

        // Show the generated palette section
        generatedPaletteDiv.style.display = 'block';
    }
};

// Helper function to get average HSL from multiple colors
function getAverageHsl(colors) {
    if (!colors || colors.length === 0) return null;
    
    const hslColors = colors.map(c => hexToHsl(c)).filter(h => h !== null);
    if (hslColors.length === 0) return null;
    
    // Calculate average hue (handling circular nature)
    let hueSum = 0;
    let sSum = 0;
    let lSum = 0;
    
    hslColors.forEach(hsl => {
        hueSum += hsl.h;
        sSum += hsl.s;
        lSum += hsl.l;
    });
    
    return {
        h: Math.round(hueSum / hslColors.length),
        s: Math.round(sSum / hslColors.length),
        l: Math.round(lSum / hslColors.length)
    };
}

// Enhance palette by correlating with user's existing colors
function enhancePaletteWithCorrelation(palette, userColors, schemeType) {
    if (!userColors || userColors.length <= 1) return palette;
    
    // Convert to arrays with # prefix
    const normalizedPalette = palette.map(c => c.startsWith('#') ? c : '#' + c);
    const normalizedUserColors = userColors.map(c => c.startsWith('#') ? c : '#' + c);
    
    // For schemes with specific relationships, try to maintain those relationships
    // while adjusting colors to be closer to user's existing palette
    
    const userHsl = normalizedUserColors.map(c => hexToHsl(c));
    const baseHsl = userHsl[0];
    
    // Adjust palette colors to be more correlated with user colors
    // by finding the closest user color and adjusting saturation/lightness to match
    return normalizedPalette.map(paletteColor => {
        const paletteHsl = hexToHsl(paletteColor);
        if (!paletteHsl) return paletteColor;
        
        // Find closest user color by hue distance
        let closestUserHsl = baseHsl;
        let minHueDistance = 360;
        
        userHsl.forEach(userH => {
            if (!userH) return;
            const hueDist = Math.min(
                Math.abs(paletteHsl.h - userH.h),
                360 - Math.abs(paletteHsl.h - userH.h)
            );
            if (hueDist < minHueDistance) {
                minHueDistance = hueDist;
                closestUserHsl = userH;
            }
        });
        
        // Blend saturation and lightness toward user's color characteristics
        // Keep the hue from the generated palette to maintain the scheme relationship
        const blendedS = paletteHsl.s * 0.7 + closestUserHsl.s * 0.3;
        const blendedL = paletteHsl.l * 0.7 + closestUserHsl.l * 0.3;
        
        return hslToHex(paletteHsl.h, blendedS, blendedL);
    });
}

// Apply the generated palette
window.applyGeneratedPalette = function () {
    if (!window.generatedPalette || !Array.isArray(window.generatedPalette)) {
        window.showError('Palette Error', 'No palette has been generated yet.');
        return;
    }

    // Clear existing colors
    const colorInputs = document.getElementById('colorInputs');
    if (colorInputs) {
        colorInputs.innerHTML = '';

        // Add each color from the generated palette
        window.generatedPalette.forEach((color, index) => {
            if (!color.startsWith('#')) color = '#' + color;
            // Set decreasing weights for subsequent colors
            const weight = Math.max(20, 100 - (index * 15));
            addColorRow(color, weight);
        });

        // Update the palette
        window.updateColorPalette();

        // Hide the modal
        const modal = document.getElementById('suggestPaletteModal');
        if (modal) {
            modal.style.display = 'none';
        }

        window.showSuccess('Palette Applied', `Applied palette with ${window.generatedPalette.length} colors`);
    }
};

// Add hover effects for scheme mode selection

// Function to show tool information
window.showToolInfo = function (toolId) {
    console.log('Showing info for tool:', toolId);

    const modal = document.getElementById('toolModal');
    const modalContent = document.getElementById('toolModalContent');

    if (!modal || !modalContent) {
        console.error('Tool modal not found');
        return;
    }

    // Set content based on tool ID
    let content = '';

    switch (toolId) {
        case 'color-palette':
            content = `
                <h2 style="color: var(--text); margin-top: 0;">🎨 Color Palette Analyzer</h2>
                <p style="color: var(--subtext1); line-height: 1.6;">Comprehensive analysis and visualization of color palettes extracted from images, including dominant colors, distribution patterns, and palette comparison.</p>
                
                <h3 style="color: var(--mauve); margin-top: 24px; margin-bottom: 12px;">Key Features</h3>
                <ul style="color: var(--subtext1); line-height: 1.8;">
                    <li><strong style="color: var(--text);">Extract dominant colors</strong> using K-means clustering</li>
                    <li><strong style="color: var(--text);">Generate color swatches</strong> and distribution charts</li>
                    <li><strong style="color: var(--text);">Compare palettes</strong> between multiple images</li>
                    <li><strong style="color: var(--text);">L*a*b* color space projections</strong> for accurate analysis</li>
                    <li><strong style="color: var(--text);">Export palette visualizations</strong> in multiple formats</li>
                </ul>
                
                <h3 style="color: var(--mauve); margin-top: 24px; margin-bottom: 12px;">How to Use</h3>
                <ol style="color: var(--subtext1); line-height: 1.8;">
                    <li>Upload an image or select a directory of images</li>
                    <li>Configure analysis parameters (number of colors, color space, clustering method)</li>
                    <li>Select output options and format</li>
                    <li>Run the analysis to generate palette visualizations</li>
                    <li>Export results for further use</li>
                </ol>
                
                <p style="color: var(--subtext0); margin-top: 20px; font-style: italic;">This tool uses advanced color clustering algorithms to identify the most representative colors in your images.</p>
            `;
            break;

        case 'search-results':
            content = `
                <h2 style="color: var(--text); margin-top: 0;">📊 Search Results Analyzer</h2>
                <p style="color: var(--subtext1); line-height: 1.6;">Advanced visualization and analysis of search results, including ranking analysis, distance distribution, color similarity mapping, and performance metrics.</p>
                
                <h3 style="color: var(--blue); margin-top: 24px; margin-bottom: 12px;">Key Features</h3>
                <ul style="color: var(--subtext1); line-height: 1.8;">
                    <li><strong style="color: var(--text);">Visualize search result rankings</strong> and distances</li>
                    <li><strong style="color: var(--text);">Analyze color similarity patterns</strong> across results</li>
                    <li><strong style="color: var(--text);">Generate performance reports</strong> and heatmaps</li>
                    <li><strong style="color: var(--text);">Create interactive result galleries</strong></li>
                    <li><strong style="color: var(--text);">Export visualizations</strong> for analysis</li>
                </ul>
                
                <h3 style="color: var(--blue); margin-top: 24px; margin-bottom: 12px;">How to Use</h3>
                <ol style="color: var(--subtext1); line-height: 1.8;">
                    <li>Enter a color query or use your current search</li>
                    <li>Configure analysis type and visualization style</li>
                    <li>Select metadata and output options</li>
                    <li>Run the analysis to generate comprehensive reports</li>
                    <li>Explore interactive visualizations and export results</li>
                </ol>
                
                <p style="color: var(--subtext0); margin-top: 20px; font-style: italic;">This tool provides deep insights into how Chromatica's search algorithm ranks and retrieves results.</p>
            `;
            break;

        case 'color-explorer':
            content = `
                <h2 style="color: var(--text); margin-top: 0;">🔍 Interactive Color Explorer</h2>
                <p style="color: var(--subtext1); line-height: 1.6;">Interactive interface for exploring color combinations, analyzing color relationships, and experimenting with different color schemes and harmonies.</p>
                
                <h3 style="color: var(--green); margin-top: 24px; margin-bottom: 12px;">Key Features</h3>
                <ul style="color: var(--subtext1); line-height: 1.8;">
                    <li><strong style="color: var(--text);">Interactive color picker</strong> and palette builder</li>
                    <li><strong style="color: var(--text);">Color harmony analysis</strong> (complementary, analogous, triadic)</li>
                    <li><strong style="color: var(--text);">Real-time color preview</strong> and scheme generation</li>
                    <li><strong style="color: var(--text);">Integration with Chromatica API</strong> for live search</li>
                    <li><strong style="color: var(--text);">Export color palettes</strong> and schemes</li>
                </ul>
                
                <h3 style="color: var(--green); margin-top: 24px; margin-bottom: 12px;">How to Use</h3>
                <ol style="color: var(--subtext1); line-height: 1.8;">
                    <li>Select a base color using the color picker</li>
                    <li>Choose a harmony type to generate related colors</li>
                    <li>Adjust saturation and brightness ranges</li>
                    <li>Fine-tune the generated color scheme</li>
                    <li>Apply to search or export the palette</li>
                </ol>
                
                <p style="color: var(--subtext0); margin-top: 20px; font-style: italic;">Perfect for discovering harmonious color combinations based on established color theory principles.</p>
            `;
            break;

        case 'histogram-analysis':
            content = `
                <h2 style="color: var(--text); margin-top: 0;">📈 Histogram Analysis Tool</h2>
                <p style="color: var(--subtext1); line-height: 1.6;">Comprehensive testing and visualization of histogram generation, including validation, performance benchmarking, and distribution analysis.</p>
                
                <h3 style="color: var(--peach); margin-top: 24px; margin-bottom: 12px;">Key Features</h3>
                <ul style="color: var(--subtext1); line-height: 1.8;">
                    <li><strong style="color: var(--text);">Single image and batch directory processing</strong></li>
                    <li><strong style="color: var(--text);">Histogram validation</strong> and quality checks</li>
                    <li><strong style="color: var(--text);">Performance benchmarking</strong> and timing analysis</li>
                    <li><strong style="color: var(--text);">Visualization of histogram distributions</strong></li>
                    <li><strong style="color: var(--text);">Multiple output formats</strong> (JSON, CSV, visualizations)</li>
                </ul>
                
                <h3 style="color: var(--peach); margin-top: 24px; margin-bottom: 12px;">Technical Details</h3>
                <p style="color: var(--subtext1); line-height: 1.6;">Chromatica uses <strong style="color: var(--text);">8x12x12 binning grid (1,152 dimensions)</strong> in CIE Lab color space with tri-linear soft assignment. Histograms are L1-normalized and transformed using Hellinger distance for ANN search.</p>
                
                <h3 style="color: var(--peach); margin-top: 24px; margin-bottom: 12px;">How to Use</h3>
                <ol style="color: var(--subtext1); line-height: 1.8;">
                    <li>Select input source (single image, directory, or pre-computed histogram)</li>
                    <li>Configure validation level and performance testing options</li>
                    <li>Choose visualization options</li>
                    <li>Run the analysis and review comprehensive reports</li>
                    <li>Export results in your preferred format</li>
                </ol>
            `;
            break;

        case 'distance-debugger':
            content = `
                <h2 style="color: var(--text); margin-top: 0;">🐛 Distance Debugger Tool</h2>
                <p style="color: var(--subtext1); line-height: 1.6;">Debug and analyze Sinkhorn-EMD distance calculations, identify numerical stability issues, and validate distance metrics.</p>
                
                <h3 style="color: var(--red); margin-top: 24px; margin-bottom: 12px;">Key Features</h3>
                <ul style="color: var(--subtext1); line-height: 1.8;">
                    <li><strong style="color: var(--text);">Debug Sinkhorn-EMD distance calculations</strong></li>
                    <li><strong style="color: var(--text);">Identify numerical stability issues</strong></li>
                    <li><strong style="color: var(--text);">Validate distance metrics</strong> and fallbacks</li>
                    <li><strong style="color: var(--text);">Compare different distance calculation methods</strong></li>
                    <li><strong style="color: var(--text);">Generate debugging reports</strong></li>
                </ul>
                
                <h3 style="color: var(--red); margin-top: 24px; margin-bottom: 12px;">Technical Background</h3>
                <p style="color: var(--subtext1); line-height: 1.6;">The Earth Mover's Distance (EMD) is approximated using the <strong style="color: var(--text);">Sinkhorn algorithm</strong>, providing a balance of accuracy and performance. This tool helps visualize and debug those calculations, including convergence analysis and numerical stability checks.</p>
                
                <h3 style="color: var(--red); margin-top: 24px; margin-bottom: 12px;">How to Use</h3>
                <ol style="color: var(--subtext1); line-height: 1.8;">
                    <li>Select test type and dataset for debugging</li>
                    <li>Configure Sinkhorn parameters (epsilon, max iterations)</li>
                    <li>Choose fallback strategy for failed calculations</li>
                    <li>Run the debugger and analyze results</li>
                    <li>Review detailed debug logs and visualizations</li>
                </ol>
            `;
            break;

        case 'query-visualizer':
            content = `
                <h2 style="color: var(--text); margin-top: 0;">🎭 Query Visualizer Tool</h2>
                <p style="color: var(--subtext1); line-height: 1.6;">Create visual representations of color queries with weighted color bars, color palettes, and comprehensive query summaries.</p>
                
                <h3 style="color: var(--pink); margin-top: 24px; margin-bottom: 12px;">Key Features</h3>
                <ul style="color: var(--subtext1); line-height: 1.8;">
                    <li><strong style="color: var(--text);">Generate weighted color bars</strong> representing query colors</li>
                    <li><strong style="color: var(--text);">Create color palette visualizations</strong> with weights</li>
                    <li><strong style="color: var(--text);">Build comprehensive query summary images</strong></li>
                    <li><strong style="color: var(--text);">Export query visualizations</strong> in multiple formats</li>
                    <li><strong style="color: var(--text);">Customizable color representations</strong> and layouts</li>
                </ul>
                
                <h3 style="color: var(--pink); margin-top: 24px; margin-bottom: 12px;">How to Use</h3>
                <ol style="color: var(--subtext1); line-height: 1.8;">
                    <li>Enter color query and weights (or use current search)</li>
                    <li>Choose visualization style and layout</li>
                    <li>Configure size, background, and additional features</li>
                    <li>Generate the visualization</li>
                    <li>Export in your preferred format (PNG, SVG, PDF, HTML)</li>
                </ol>
                
                <p style="color: var(--subtext0); margin-top: 20px; font-style: italic;">Perfect for creating visual summaries of your color search queries for presentations and documentation.</p>
            `;
            break;

        case 'color-analysis':
            content = `
                <h2 style="color: var(--text); margin-top: 0;">🎨 Color Analysis Tool</h2>
                <p style="color: var(--subtext1); line-height: 1.6;">Analyze color relationships, harmony scores, and palette characteristics with comprehensive metrics and visual feedback.</p>
                
                <h3 style="color: var(--mauve); margin-top: 24px; margin-bottom: 12px;">Key Features</h3>
                <ul style="color: var(--subtext1); line-height: 1.8;">
                    <li><strong style="color: var(--text);">Color harmony score calculation</strong> - Evaluates how well colors work together</li>
                    <li><strong style="color: var(--text);">Color relationship analysis</strong> - Identifies complementary, analogous, triadic relationships</li>
                    <li><strong style="color: var(--text);">Complementary color detection</strong></li>
                    <li><strong style="color: var(--text);">Contrast ratio evaluation</strong> for accessibility compliance</li>
                    <li><strong style="color: var(--text);">Accessibility compliance checks</strong></li>
                    <li><strong style="color: var(--text);">Color scheme suggestions</strong></li>
                </ul>
                
                <h3 style="color: var(--mauve); margin-top: 24px; margin-bottom: 12px;">How to Use</h3>
                <ol style="color: var(--subtext1); line-height: 1.8;">
                    <li>Enter hex colors separated by commas or spaces</li>
                    <li>Select analysis mode (basic, advanced, comprehensive)</li>
                    <li>Configure analysis options and harmony rules</li>
                    <li>Run the analysis to generate results</li>
                    <li>Review harmony scores, relationships, and suggestions</li>
                </ol>
                
                <p style="color: var(--subtext0); margin-top: 20px; font-style: italic;">This tool provides deep insights into color theory and helps create harmonious color combinations.</p>
            `;
            break;

        default:
            content = `<h2 style="color: var(--text);">Tool Information</h2><p style="color: var(--subtext1);">No specific information available for this tool.</p>`;
    }

    modalContent.innerHTML = content;
    modal.style.display = 'block';
};

// Function to close the tool modal
window.closeToolModal = function () {
    const modal = document.getElementById('toolModal');
    if (modal) {
        modal.style.display = 'none';
    }
};

// Function to close the details modal
window.closeDetailsModal = function () {
    const modal = document.getElementById('detailsModal');
    if (modal) {
        modal.style.display = 'none';
    }
};

// Toggle tool panel visibility
window.toggleToolPanel = function (panelId) {
    const panel = document.getElementById(panelId);
    if (panel) {
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }
};


document.addEventListener('DOMContentLoaded', function () {
    // Initialize with default color if no rows exist
    const colorInputs = document.getElementById('colorInputs');
    if (colorInputs && colorInputs.children.length === 0) {
        addColorRow('#FF0000', 100);
    }
    
    // Add hover effects to scheme modes
    const schemeModes = document.querySelectorAll('.scheme-mode');
    schemeModes.forEach(mode => {
        mode.addEventListener('mouseover', () => {
            mode.style.transform = 'scale(1.02)';
            mode.style.borderColor = 'var(--mauve)';
            mode.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)';
        });
        mode.addEventListener('mouseout', () => {
            mode.style.transform = 'scale(1)';
            mode.style.borderColor = 'var(--surface2)';
            mode.style.boxShadow = 'none';
        });
        mode.addEventListener('click', () => {
            // Add a brief "active" effect
            mode.style.transform = 'scale(0.98)';
            setTimeout(() => mode.style.transform = 'scale(1)', 100);
        });
    });

    // Set up event handlers for initial color row(s) - these should be handled by addColorRow now
    const initialColorRows = document.querySelectorAll('.color-row');
    initialColorRows.forEach(colorRow => {
        const colorPicker = colorRow.querySelector('.color-picker');
        const formatSelect = colorRow.querySelector('.color-format-select');
        const formatInput = colorRow.querySelector('.color-format-input');

        if (colorPicker && formatSelect && formatInput) {
            // Update format input when color picker changes
            const updateFormatInput = () => {
                const currentFormat = formatSelect.value;
                const hexColor = colorPicker.value;
                formatInput.value = convertColorToFormat(hexColor, currentFormat);
            };

            // Update color picker when format input changes
            formatInput.addEventListener('input', () => {
                try {
                    const format = formatSelect.value;
                    const hex = convertFormatToHex(formatInput.value, format);
                    if (hex) {
                        colorPicker.value = hex;
                        if (window.handleColorPickerChange) {
                            window.handleColorPickerChange({ target: colorPicker });
                        }
                    }
                } catch (e) {
                    console.warn('Invalid color format:', e);
                }
            });

            // Update format input when format select changes
            formatSelect.addEventListener('change', updateFormatInput);

            // Update format input when color picker changes
            colorPicker.addEventListener('input', updateFormatInput);
            colorPicker.addEventListener('change', updateFormatInput);

            // Initial update
            updateFormatInput();
        }
    });
});


// Add this in the window.showImageInModal function:
window.showImageInModal = function (imageSrc) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const closeButton = modal.querySelector('.image-close');

    if (modal && modalImage) {
        modalImage.src = imageSrc;
        modal.style.display = 'flex';

        // Add click handlers for closing
        closeButton.onclick = () => {
            modal.style.display = 'none';
        };

        // Close on clicking outside the image
        modal.onclick = (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        };

        // Prevent image click from closing modal
        modalImage.onclick = (e) => {
            e.stopPropagation();
        };
    }
};

// ============================================================================
// TESTING & DEVELOPMENT TOOLS
// ============================================================================

/**
 * Run a search test
 */
window.runSearchTest = async function () {
    console.log('[Search Test] Starting search test...');

    const resultsSection = document.getElementById('quickTestResultsSection');
    const resultsContent = document.getElementById('quickTestResultsContent');

    if (resultsSection && resultsContent) {
        resultsSection.style.display = 'block';
        resultsContent.innerHTML = '<div style="text-align: center; padding: 20px;"><p>⏳ Running search test...</p></div>';
        // Scroll to results section
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    try {
        // Test with a simple color query
        const testColors = 'FF0000,00FF00,0000FF';
        const testWeights = '0.4,0.3,0.3';
        const testK = 10;

        const startTime = performance.now();
        const response = await fetch(`/search?colors=${testColors}&weights=${testWeights}&k=${testK}&fast_mode=false`);
        const endTime = performance.now();

        if (!response.ok) {
            throw new Error(`Search failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        const duration = (endTime - startTime).toFixed(2);

        let html = `
            <div style="padding: 20px;">
                <h3 style="color: var(--text); margin: 0 0 15px 0;">🔍 Search Test Results</h3>
                <div style="background: var(--surface1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <p style="margin: 5px 0; color: var(--text);"><strong>Status:</strong> <span style="color: var(--green);">✓ Success</span></p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Duration:</strong> ${duration}ms</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Results Found:</strong> ${data.results_count || 0}</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Index Size:</strong> ${data.metadata?.index_size || 0}</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>ANN Time:</strong> ${data.metadata?.ann_time_ms || 0}ms</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Rerank Time:</strong> ${data.metadata?.rerank_time_ms || 0}ms</p>
                </div>
        `;

        if (data.results && data.results.length > 0) {
            html += `<h4 style="color: var(--text); margin: 15px 0 10px 0;">Top Results:</h4>`;
            html += `<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px;">`;
            data.results.slice(0, 6).forEach((result, idx) => {
                const imgUrl = result.image_url || `/image/${encodeURIComponent(result.image_id)}`;
                html += `
                    <div style="background: var(--surface0); padding: 10px; border-radius: 8px; text-align: center;">
                        <img src="${imgUrl}" alt="Result ${idx + 1}" style="width: 100%; height: 100px; object-fit: cover; border-radius: 4px; margin-bottom: 5px;" onerror="this.style.display='none'">
                        <p style="margin: 0; font-size: 12px; color: var(--subtext1);">Distance: ${result.distance?.toFixed(4) || 'N/A'}</p>
                    </div>
                `;
            });
            html += `</div>`;
        }

        html += `</div>`;

        if (resultsContent) {
            resultsContent.innerHTML = html;
        }

        // Scroll to results section after results are displayed
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        console.log('[Search Test] ✓ Test completed successfully');
    } catch (error) {
        console.error('[Search Test] ✗ Test failed:', error);
        if (resultsContent) {
            resultsContent.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                    <h3 style="color: var(--red); margin: 0 0 10px 0;">❌ Search Test Failed</h3>
                    <p style="color: var(--subtext1);">${error.message}</p>
                </div>
            `;
        }
    }
};

/**
 * Run performance benchmark
 */
window.runPerformanceBenchmark = async function () {
    console.log('[Performance Benchmark] Starting benchmark...');

    const resultsSection = document.getElementById('quickTestResultsSection');
    const resultsContent = document.getElementById('quickTestResultsContent');

    if (resultsSection && resultsContent) {
        resultsSection.style.display = 'block';
        resultsContent.innerHTML = '<div style="text-align: center; padding: 20px;"><p>⏳ Running performance benchmark...</p><p style="font-size: 12px; color: var(--subtext0);">This may take a few seconds...</p></div>';
        // Scroll to results section
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    try {
        const testQueries = [
            { colors: 'FF0000', weights: '1.0', name: 'Red' },
            { colors: '00FF00', weights: '1.0', name: 'Green' },
            { colors: '0000FF', weights: '1.0', name: 'Blue' },
            { colors: 'FF0000,00FF00', weights: '0.5,0.5', name: 'Red+Green' },
            { colors: '0000FF,FFFF00', weights: '0.5,0.5', name: 'Blue+Yellow' }
        ];

        const results = [];

        for (const query of testQueries) {
            const startTime = performance.now();
            try {
                const response = await fetch(`/search?colors=${query.colors}&weights=${query.weights}&k=10&fast_mode=false`);
                const endTime = performance.now();

                if (response.ok) {
                    const data = await response.json();
                    results.push({
                        name: query.name,
                        duration: endTime - startTime,
                        success: true,
                        resultsCount: data.results_count || 0,
                        annTime: data.metadata?.ann_time_ms || 0,
                        rerankTime: data.metadata?.rerank_time_ms || 0
                    });
                } else {
                    results.push({
                        name: query.name,
                        duration: endTime - startTime,
                        success: false,
                        error: `HTTP ${response.status}`
                    });
                }
            } catch (error) {
                results.push({
                    name: query.name,
                    duration: 0,
                    success: false,
                    error: error.message
                });
            }
        }

        const avgDuration = results.filter(r => r.success).reduce((sum, r) => sum + r.duration, 0) / results.filter(r => r.success).length;
        const avgANNTime = results.filter(r => r.success).reduce((sum, r) => sum + (r.annTime || 0), 0) / results.filter(r => r.success).length;
        const avgRerankTime = results.filter(r => r.success).reduce((sum, r) => sum + (r.rerankTime || 0), 0) / results.filter(r => r.success).length;

        let html = `
            <div style="padding: 20px;">
                <h3 style="color: var(--text); margin: 0 0 15px 0;">⚡ Performance Benchmark Results</h3>
                <div style="background: var(--surface1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h4 style="color: var(--text); margin: 0 0 10px 0;">Summary</h4>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Tests Completed:</strong> ${results.filter(r => r.success).length}/${results.length}</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Average Duration:</strong> ${avgDuration.toFixed(2)}ms</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Average ANN Time:</strong> ${avgANNTime.toFixed(2)}ms</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Average Rerank Time:</strong> ${avgRerankTime.toFixed(2)}ms</p>
                </div>
                <h4 style="color: var(--text); margin: 15px 0 10px 0;">Individual Results:</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <thead>
                        <tr style="background: var(--surface1);">
                            <th style="text-align: left; padding: 8px; color: var(--text);">Query</th>
                            <th style="text-align: right; padding: 8px; color: var(--text);">Duration (ms)</th>
                            <th style="text-align: right; padding: 8px; color: var(--text);">ANN (ms)</th>
                            <th style="text-align: right; padding: 8px; color: var(--text);">Rerank (ms)</th>
                            <th style="text-align: center; padding: 8px; color: var(--text);">Status</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        results.forEach(result => {
            const statusColor = result.success ? 'var(--green)' : 'var(--red)';
            const statusText = result.success ? '✓' : '✗';
            html += `
                <tr style="border-top: 1px solid var(--surface2);">
                    <td style="padding: 8px; color: var(--text);">${result.name}</td>
                    <td style="padding: 8px; text-align: right; color: var(--text);">${result.duration.toFixed(2)}</td>
                    <td style="padding: 8px; text-align: right; color: var(--text);">${result.annTime || 'N/A'}</td>
                    <td style="padding: 8px; text-align: right; color: var(--text);">${result.rerankTime || 'N/A'}</td>
                    <td style="padding: 8px; text-align: center; color: ${statusColor};">${statusText}</td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;

        if (resultsContent) {
            resultsContent.innerHTML = html;
        }

        // Scroll to results section after results are displayed
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        console.log('[Performance Benchmark] ✓ Benchmark completed');
    } catch (error) {
        console.error('[Performance Benchmark] ✗ Benchmark failed:', error);
        if (resultsContent) {
            resultsContent.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                    <h3 style="color: var(--red); margin: 0 0 10px 0;">❌ Benchmark Failed</h3>
                    <p style="color: var(--subtext1);">${error.message}</p>
                </div>
            `;
        }
    }
};

/**
 * Run histogram test
 */
window.runHistogramTest = async function () {
    console.log('[Histogram Test] Starting histogram test...');

    const resultsSection = document.getElementById('quickTestResultsSection');
    const resultsContent = document.getElementById('quickTestResultsContent');

    if (resultsSection && resultsContent) {
        resultsSection.style.display = 'block';
        resultsContent.innerHTML = '<div style="text-align: center; padding: 20px;"><p>⏳ Running histogram test...</p></div>';
        // Scroll to results section
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    try {
        // Test histogram generation via a search query
        const testColors = '808080';
        const testWeights = '1.0';

        const response = await fetch(`/search?colors=${testColors}&weights=${testWeights}&k=1&fast_mode=false`);

        if (!response.ok) {
            throw new Error(`Test failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();

        let html = `
            <div style="padding: 20px;">
                <h3 style="color: var(--text); margin: 0 0 15px 0;">📊 Histogram Test Results</h3>
                <div style="background: var(--surface1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <p style="margin: 5px 0; color: var(--text);"><strong>Status:</strong> <span style="color: var(--green);">✓ Success</span></p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Query Histogram:</strong> Created successfully</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Search Results:</strong> ${data.results_count || 0} found</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Index Size:</strong> ${data.metadata?.index_size || 0} images</p>
                </div>
                <p style="color: var(--subtext1); font-size: 14px;">Histogram generation is functioning correctly. Query histogram was created and used successfully in search.</p>
            </div>
        `;

        if (resultsContent) {
            resultsContent.innerHTML = html;
        }

        // Scroll to results section after results are displayed
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        console.log('[Histogram Test] ✓ Test completed successfully');
    } catch (error) {
        console.error('[Histogram Test] ✗ Test failed:', error);
        if (resultsContent) {
            resultsContent.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                    <h3 style="color: var(--red); margin: 0 0 10px 0;">❌ Histogram Test Failed</h3>
                    <p style="color: var(--subtext1);">${error.message}</p>
                </div>
            `;
        }
    }
};

/**
 * Validate system
 */
window.validateSystem = async function () {
    console.log('[System Validation] Starting validation...');

    const resultsSection = document.getElementById('quickTestResultsSection');
    const resultsContent = document.getElementById('quickTestResultsContent');

    if (resultsSection && resultsContent) {
        resultsSection.style.display = 'block';
        resultsContent.innerHTML = '<div style="text-align: center; padding: 20px;"><p>⏳ Validating system...</p></div>';
        // Scroll to results section
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    try {
        const infoResponse = await fetch('/api/info');
        const infoData = await infoResponse.ok ? await infoResponse.json() : null;

        const testResponse = await fetch('/search?colors=FF0000&weights=1.0&k=1&fast_mode=false');
        const testData = testResponse.ok ? await testResponse.json() : null;

        let html = `
            <div style="padding: 20px;">
                <h3 style="color: var(--text); margin: 0 0 15px 0;">✅ System Validation Results</h3>
                <div style="background: var(--surface1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h4 style="color: var(--text); margin: 0 0 10px 0;">Component Status</h4>
        `;

        const checks = [
            { name: 'API Endpoint', status: infoResponse.ok, details: infoData?.status || 'Unknown' },
            { name: 'Search Functionality', status: testResponse.ok, details: testData ? `${testData.results_count} results` : 'Failed' },
            { name: 'Index Access', status: testData && testData.metadata?.index_size > 0, details: testData?.metadata?.index_size || 0 },
        ];

        checks.forEach(check => {
            const statusColor = check.status ? 'var(--green)' : 'var(--red)';
            const statusText = check.status ? '✓' : '✗';
            html += `
                <div style="margin: 8px 0; padding: 8px; background: var(--surface0); border-radius: 4px;">
                    <span style="color: ${statusColor}; font-weight: bold;">${statusText}</span>
                    <span style="color: var(--text); margin-left: 10px;"><strong>${check.name}:</strong> ${check.details}</span>
                </div>
            `;
        });

        const allPassed = checks.every(c => c.status);

        html += `
                </div>
                <div style="background: ${allPassed ? 'var(--green)' : 'var(--red)'}; padding: 15px; border-radius: 8px; text-align: center;">
                    <h4 style="color: #000000; margin: 0;">${allPassed ? '✓ System Validation Passed' : '✗ System Validation Failed'}</h4>
                </div>
            </div>
        `;

        if (resultsContent) {
            resultsContent.innerHTML = html;
        }

        // Scroll to results section after results are displayed
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        console.log('[System Validation] ✓ Validation completed');
    } catch (error) {
        console.error('[System Validation] ✗ Validation failed:', error);
        if (resultsContent) {
            resultsContent.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                    <h3 style="color: var(--red); margin: 0 0 10px 0;">❌ Validation Failed</h3>
                    <p style="color: var(--subtext1);">${error.message}</p>
                </div>
            `;
        }
    }
};

/**
 * Show system status
 */
window.showSystemStatus = async function () {
    console.log('[System Status] Fetching status...');

    const resultsSection = document.getElementById('quickTestResultsSection');
    const resultsContent = document.getElementById('quickTestResultsContent');

    if (resultsSection && resultsContent) {
        resultsSection.style.display = 'block';
        resultsContent.innerHTML = '<div style="text-align: center; padding: 20px;"><p>⏳ Fetching system status...</p></div>';
        // Scroll to results section
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    try {
        const response = await fetch('/api/info');
        const data = response.ok ? await response.json() : null;

        if (!data) {
            throw new Error('Failed to fetch system status');
        }

        // Also try to get index size
        let indexSize = 0;
        try {
            const searchResponse = await fetch('/search?colors=808080&weights=1.0&k=1&fast_mode=true');
            if (searchResponse.ok) {
                const searchData = await searchResponse.json();
                indexSize = searchData.metadata?.index_size || 0;
            }
        } catch (e) {
            console.warn('Could not fetch index size:', e);
        }

        let html = `
            <div style="padding: 20px;">
                <h3 style="color: var(--text); margin: 0 0 15px 0;">📊 System Status</h3>
                <div style="background: var(--surface1); padding: 15px; border-radius: 8px;">
                    <p style="margin: 5px 0; color: var(--text);"><strong>Status:</strong> <span style="color: ${data.status === 'ready' ? 'var(--green)' : 'var(--yellow)'};">${data.status || 'Unknown'}</span></p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Version:</strong> ${data.version || 'Unknown'}</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Index Size:</strong> ${indexSize.toLocaleString()} images</p>
                    <p style="margin: 5px 0; color: var(--text);"><strong>Message:</strong> ${data.message || 'N/A'}</p>
                    <h4 style="color: var(--text); margin: 15px 0 10px 0;">Available Endpoints:</h4>
                    <ul style="color: var(--subtext1); margin: 0; padding-left: 20px;">
                        ${data.endpoints ? Object.entries(data.endpoints).map(([key, value]) => `<li>${value}</li>`).join('') : '<li>No endpoints listed</li>'}
                    </ul>
                </div>
            </div>
        `;

        if (resultsContent) {
            resultsContent.innerHTML = html;
        }

        // Scroll to results section after results are displayed
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        console.log('[System Status] ✓ Status fetched successfully');
    } catch (error) {
        console.error('[System Status] ✗ Failed to fetch status:', error);
        if (resultsContent) {
            resultsContent.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                    <h3 style="color: var(--red); margin: 0 0 10px 0;">❌ Failed to Fetch Status</h3>
                    <p style="color: var(--subtext1);">${error.message}</p>
                </div>
            `;
        }
    }
};

/**
 * Run diagnostics
 */
window.runDiagnostics = async function () {
    console.log('[Diagnostics] Running diagnostics...');

    const resultsSection = document.getElementById('quickTestResultsSection');
    const resultsContent = document.getElementById('quickTestResultsContent');

    if (resultsSection && resultsContent) {
        resultsSection.style.display = 'block';
        resultsContent.innerHTML = '<div style="text-align: center; padding: 20px;"><p>⏳ Running diagnostics...</p><p style="font-size: 12px; color: var(--subtext0);">This may take a few seconds...</p></div>';
        // Scroll to results section
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    try {
        const diagnostics = [];

        // Test 1: API availability
        try {
            const infoResponse = await fetch('/api/info');
            diagnostics.push({
                name: 'API Endpoint',
                status: infoResponse.ok,
                details: infoResponse.ok ? 'Available' : `HTTP ${infoResponse.status}`
            });
        } catch (e) {
            diagnostics.push({ name: 'API Endpoint', status: false, details: e.message });
        }

        // Test 2: Search functionality
        try {
            const searchResponse = await fetch('/search?colors=FF0000&weights=1.0&k=5&fast_mode=false');
            const searchData = searchResponse.ok ? await searchResponse.json() : null;
            diagnostics.push({
                name: 'Search Functionality',
                status: searchResponse.ok && searchData?.results_count > 0,
                details: searchData ? `${searchData.results_count} results, ${searchData.metadata?.total_time_ms || 0}ms` : 'Failed'
            });
        } catch (e) {
            diagnostics.push({ name: 'Search Functionality', status: false, details: e.message });
        }

        // Test 3: Fast mode
        try {
            const fastResponse = await fetch('/search?colors=00FF00&weights=1.0&k=5&fast_mode=true');
            const fastData = fastResponse.ok ? await fastResponse.json() : null;
            diagnostics.push({
                name: 'Fast Mode Search',
                status: fastResponse.ok && fastData?.results_count > 0,
                details: fastData ? `${fastData.results_count} results, ${fastData.metadata?.total_time_ms || 0}ms` : 'Failed'
            });
        } catch (e) {
            diagnostics.push({ name: 'Fast Mode Search', status: false, details: e.message });
        }

        // Test 4: Image serving
        try {
            // First get an image ID from search
            const testResponse = await fetch('/search?colors=808080&weights=1.0&k=1&fast_mode=true');
            if (testResponse.ok) {
                const testData = await testResponse.json();
                if (testData.results && testData.results.length > 0) {
                    const imageId = testData.results[0].image_id;
                    const imageResponse = await fetch(`/image/${encodeURIComponent(imageId)}`);
                    diagnostics.push({
                        name: 'Image Serving',
                        status: imageResponse.ok,
                        details: imageResponse.ok ? 'Available' : `HTTP ${imageResponse.status}`
                    });
                } else {
                    diagnostics.push({ name: 'Image Serving', status: false, details: 'No images in index' });
                }
            } else {
                diagnostics.push({ name: 'Image Serving', status: false, details: 'Cannot test - search failed' });
            }
        } catch (e) {
            diagnostics.push({ name: 'Image Serving', status: false, details: e.message });
        }

        let html = `
            <div style="padding: 20px;">
                <h3 style="color: var(--text); margin: 0 0 15px 0;">🔧 Diagnostics Results</h3>
        `;

        diagnostics.forEach(diag => {
            const statusColor = diag.status ? 'var(--green)' : 'var(--red)';
            const statusText = diag.status ? '✓' : '✗';
            html += `
                <div style="background: var(--surface1); padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="color: ${statusColor}; font-size: 20px; font-weight: bold;">${statusText}</span>
                        <div style="flex: 1;">
                            <p style="margin: 0; color: var(--text); font-weight: bold;">${diag.name}</p>
                            <p style="margin: 5px 0 0 0; color: var(--subtext1); font-size: 13px;">${diag.details}</p>
                        </div>
                    </div>
                </div>
            `;
        });

        const allPassed = diagnostics.every(d => d.status);

        html += `
                <div style="background: ${allPassed ? 'var(--green)' : 'var(--yellow)'}; padding: 15px; border-radius: 8px; text-align: center; margin-top: 15px;">
                    <h4 style="color: #000000; margin: 0;">${allPassed ? '✓ All Diagnostics Passed' : '⚠ Some Issues Detected'}</h4>
                </div>
            </div>
        `;

        if (resultsContent) {
            resultsContent.innerHTML = html;
        }

        // Scroll to results section after results are displayed
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        console.log('[Diagnostics] ✓ Diagnostics completed');
    } catch (error) {
        console.error('[Diagnostics] ✗ Diagnostics failed:', error);
        if (resultsContent) {
            resultsContent.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                    <h3 style="color: var(--red); margin: 0 0 10px 0;">❌ Diagnostics Failed</h3>
                    <p style="color: var(--subtext1);">${error.message}</p>
                </div>
            `;
        }
    }
};

/**
 * Export logs
 */
window.exportLogs = async function () {
    console.log('[Export Logs] Exporting logs...');

    try {
        // Get log directory info
        const logInfo = {
            timestamp: new Date().toISOString(),
            message: 'Log export functionality',
            note: 'Log files are stored in the logs/ directory on the server. For direct access, check the server logs directory.',
            logFiles: [
                'chromatica_api_*.log - API server logs',
                'chromatica_search_*.log - Search operation logs',
                'Other component-specific logs may also be available'
            ]
        };

        // Create a downloadable JSON file with log info
        const blob = new Blob([JSON.stringify(logInfo, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chromatica_logs_info_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);

        // Show success message
        const resultsSection = document.getElementById('quickTestResultsSection');
        const resultsContent = document.getElementById('quickTestResultsContent');

        if (resultsSection && resultsContent) {
            resultsSection.style.display = 'block';
            resultsContent.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                    <h3 style="color: var(--green); margin: 0 0 15px 0;">✓ Log Info Exported</h3>
                    <p style="color: var(--text); margin-bottom: 15px;">A JSON file with log information has been downloaded.</p>
                    <div style="background: var(--surface1); padding: 15px; border-radius: 8px; text-align: left; max-width: 600px; margin: 0 auto;">
                        <p style="color: var(--subtext1); margin: 5px 0; font-size: 14px;"><strong>Note:</strong> Direct log file export requires server-side access. Log files are located in the <code>logs/</code> directory on the server.</p>
                        <p style="color: var(--subtext1); margin: 5px 0; font-size: 14px;">For server log files, check:</p>
                        <ul style="color: var(--subtext1); font-size: 13px; margin: 10px 0;">
                            <li>logs/chromatica_api_*.log</li>
                            <li>logs/chromatica_search_*.log</li>
                        </ul>
                    </div>
                </div>
            `;
            // Scroll to results section
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        console.log('[Export Logs] ✓ Log info exported');
    } catch (error) {
        console.error('[Export Logs] ✗ Export failed:', error);
        alert(`Failed to export logs: ${error.message}`);
    }
};

// ============================================================================
// COLOR FORMAT CONVERSION FUNCTIONS
// ============================================================================

/**
 * Convert hex color to HSV format
 */
function hexToHsv(hex) {
    const rgb = hexToRgb(hex);
    if (!rgb) return null;

    let { r, g, b } = rgb;
    r /= 255;
    g /= 255;
    b /= 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;

    let h = 0;
    if (delta !== 0) {
        if (max === r) {
            h = ((g - b) / delta) % 6;
        } else if (max === g) {
            h = (b - r) / delta + 2;
        } else {
            h = (r - g) / delta + 4;
        }
    }
    h = Math.round(h * 60);
    if (h < 0) h += 360;

    const s = max === 0 ? 0 : Math.round((delta / max) * 100);
    const v = Math.round(max * 100);

    return { h, s, v };
}

/**
 * Convert HSV to hex color
 */
function hsvToHex(h, s, v) {
    h = h % 360;
    if (h < 0) h += 360;
    s = s / 100;
    v = v / 100;

    const c = v * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = v - c;

    let r = 0, g = 0, b = 0;

    if (h >= 0 && h < 60) {
        r = c; g = x; b = 0;
    } else if (h >= 60 && h < 120) {
        r = x; g = c; b = 0;
    } else if (h >= 120 && h < 180) {
        r = 0; g = c; b = x;
    } else if (h >= 180 && h < 240) {
        r = 0; g = x; b = c;
    } else if (h >= 240 && h < 300) {
        r = x; g = 0; b = c;
    } else if (h >= 300 && h < 360) {
        r = c; g = 0; b = x;
    }

    r = Math.round((r + m) * 255);
    g = Math.round((g + m) * 255);
    b = Math.round((b + m) * 255);

    return rgbToHex(r, g, b);
}

/**
 * Convert hex color to CMYK format
 */
function hexToCmyk(hex) {
    const rgb = hexToRgb(hex);
    if (!rgb) return null;

    let { r, g, b } = rgb;
    r /= 255;
    g /= 255;
    b /= 255;

    const k = 1 - Math.max(r, g, b);
    const c = k === 1 ? 0 : (1 - r - k) / (1 - k);
    const m = k === 1 ? 0 : (1 - g - k) / (1 - k);
    const y = k === 1 ? 0 : (1 - b - k) / (1 - k);

    return {
        c: Math.round(c * 100),
        m: Math.round(m * 100),
        y: Math.round(y * 100),
        k: Math.round(k * 100)
    };
}

/**
 * Convert CMYK to hex color
 */
function cmykToHex(c, m, y, k) {
    c = c / 100;
    m = m / 100;
    y = y / 100;
    k = k / 100;

    const r = Math.round(255 * (1 - c) * (1 - k));
    const g = Math.round(255 * (1 - m) * (1 - k));
    const b = Math.round(255 * (1 - y) * (1 - k));

    return rgbToHex(r, g, b);
}

/**
 * Convert color to specified format string
 */
function convertColorToFormat(hex, format) {
    if (!hex) return '';

    switch (format) {
        case 'HEX':
            return hex;
        case 'RGB':
            const rgb = hexToRgb(hex);
            return rgb ? `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})` : '';
        case 'HSL':
            const hsl = hexToHsl(hex);
            return hsl ? `hsl(${Math.round(hsl.h)}°, ${Math.round(hsl.s)}%, ${Math.round(hsl.l)}%)` : '';
        case 'HSV':
            const hsv = hexToHsv(hex);
            return hsv ? `hsv(${hsv.h}°, ${hsv.s}%, ${hsv.v}%)` : '';
        case 'CMYK':
            const cmyk = hexToCmyk(hex);
            return cmyk ? `cmyk(${cmyk.c}%, ${cmyk.m}%, ${cmyk.y}%, ${cmyk.k}%)` : '';
        default:
            return hex;
    }
}

/**
 * Convert color format string to hex
 */
function convertFormatToHex(colorString, format) {
    if (!colorString || !format) return null;

    colorString = colorString.trim();

    try {
        switch (format) {
            case 'HEX':
                // Handle with or without #
                if (colorString.startsWith('#')) {
                    return colorString.length === 7 ? colorString : null;
                } else {
                    return colorString.length === 6 ? '#' + colorString : null;
                }

            case 'RGB':
                // Parse rgb(r, g, b) or r, g, b
                const rgbMatch = colorString.match(/\d+/g);
                if (rgbMatch && rgbMatch.length === 3) {
                    const [r, g, b] = rgbMatch.map(Number);
                    if (r >= 0 && r <= 255 && g >= 0 && g <= 255 && b >= 0 && b <= 255) {
                        return rgbToHex(r, g, b);
                    }
                }
                return null;

            case 'HSL':
                // Parse hsl(h, s%, l%) or h, s, l
                const hslMatch = colorString.match(/(\d+(?:\.\d+)?)/g);
                if (hslMatch && hslMatch.length >= 3) {
                    const h = parseFloat(hslMatch[0]);
                    const s = parseFloat(hslMatch[1]);
                    const l = parseFloat(hslMatch[2]);
                    if (h >= 0 && h <= 360 && s >= 0 && s <= 100 && l >= 0 && l <= 100) {
                        return hslToHex(h, s, l);
                    }
                }
                return null;

            case 'HSV':
                // Parse hsv(h, s%, v%) or h, s, v
                const hsvMatch = colorString.match(/(\d+(?:\.\d+)?)/g);
                if (hsvMatch && hsvMatch.length >= 3) {
                    const h = parseFloat(hsvMatch[0]);
                    const s = parseFloat(hsvMatch[1]);
                    const v = parseFloat(hsvMatch[2]);
                    if (h >= 0 && h <= 360 && s >= 0 && s <= 100 && v >= 0 && v <= 100) {
                        return hsvToHex(h, s, v);
                    }
                }
                return null;

            case 'CMYK':
                // Parse cmyk(c%, m%, y%, k%) or c, m, y, k
                const cmykMatch = colorString.match(/(\d+(?:\.\d+)?)/g);
                if (cmykMatch && cmykMatch.length >= 4) {
                    const c = parseFloat(cmykMatch[0]);
                    const m = parseFloat(cmykMatch[1]);
                    const y = parseFloat(cmykMatch[2]);
                    const k = parseFloat(cmykMatch[3]);
                    if (c >= 0 && c <= 100 && m >= 0 && m <= 100 && y >= 0 && y <= 100 && k >= 0 && k <= 100) {
                        return cmykToHex(c, m, y, k);
                    }
                }
                return null;

            default:
                return null;
        }
    } catch (e) {
        console.warn('Error converting color format:', e);
        return null;
    }
}

// ============================================================================
// RANDOMIZE COLOR ROW FUNCTION
// ============================================================================

/**
 * Randomize the color in a specific row
 */
window.randomizeColorRow = function (buttonElement) {
    console.log('[Randomize] Button clicked, buttonElement:', buttonElement);
    const colorRow = buttonElement.closest('.color-row');
    if (!colorRow) {
        console.warn('[Randomize] Color row not found');
        return;
    }
    console.log('[Randomize] Found color row:', colorRow);

    // Try multiple ways to find the color picker (it's hidden with display: none)
    let colorPicker = colorRow.querySelector('input.color-picker[type="color"]');
    if (!colorPicker) {
        colorPicker = colorRow.querySelector('.color-picker');
    }
    if (!colorPicker) {
        // Try finding by checking all input elements
        const allInputs = colorRow.querySelectorAll('input');
        colorPicker = Array.from(allInputs).find(input => input.type === 'color');
    }
    
    if (!colorPicker) {
        console.warn('[Randomize] Color picker not found. Available elements:', colorRow.querySelectorAll('input'));
        return;
    }
    console.log('[Randomize] Found color picker:', colorPicker);

    // Generate random hex color
    const randomHex = '#' + randomHexColor();
    console.log('[Randomize] Generated random color:', randomHex);
    
    // Set color picker value
    colorPicker.value = randomHex;

    // Try to use the stored updateSimplePicker function if available
    if (colorRow._updateSimplePicker) {
        console.log('[Randomize] Using stored updateSimplePicker function');
        colorRow._updateSimplePicker();
    } else {
        // Fallback: manual updates
        console.log('[Randomize] Using fallback manual updates');
        
        // Update format input
        const formatSelect = colorRow.querySelector('.color-format-select');
        const formatInput = colorRow.querySelector('.color-format-input');
        if (formatSelect && formatInput) {
            formatInput.value = convertColorToFormat(randomHex, formatSelect.value);
            console.log('[Randomize] Updated format input:', formatInput.value);
        }

        // Update simple color picker button
        const simpleColorPicker = colorRow.querySelector('.simple-color-picker');
        if (simpleColorPicker) {
            simpleColorPicker.style.background = randomHex;
            console.log('[Randomize] Updated simple color picker');
        }

        // Update color name label (it's in the inputContainer - first div child)
        const inputContainer = colorRow.querySelector('div[style*="flex-direction: column"]');
        if (inputContainer) {
            const colorNameLabel = inputContainer.querySelector('div:first-child');
            if (colorNameLabel) {
                const newColorName = getColorName(randomHex);
                colorNameLabel.textContent = newColorName;
                console.log('[Randomize] Updated color name label:', newColorName);
            }
        }

        // Update complementary color
        const complementarySwatch = colorRow.querySelector('.complementary-color > div:first-child');
        const complementaryInfo = colorRow.querySelector('.complementary-color span');
        if (complementarySwatch && complementaryInfo) {
            const complementaryColor = getComplementaryColor(randomHex);
            const complementaryName = getColorName(complementaryColor);
            complementarySwatch.style.backgroundColor = complementaryColor;
            complementaryInfo.textContent = `Complementary: ${complementaryName}`;
            console.log('[Randomize] Updated complementary color:', complementaryColor);
        }
    }

    // Manually trigger input and change events to ensure all listeners fire
    const inputEvent = new Event('input', { bubbles: true });
    const changeEvent = new Event('change', { bubbles: true });
    colorPicker.dispatchEvent(inputEvent);
    colorPicker.dispatchEvent(changeEvent);

    // Trigger color picker change handler
    if (window.handleColorPickerChange) {
        window.handleColorPickerChange({ target: colorPicker });
    }

    // Update color palette
    if (window.updateColorPalette) {
        window.updateColorPalette();
    }

    console.log('[Randomize] Changed color row to:', randomHex);
};

// ============================================================================
// ROLL DICE FUNCTION
// ============================================================================

/**
 * Roll Dice: Generate random number of colors with random weights
 */
window.rollDice = function () {
    console.log('[Roll Dice] Generating random color query...');

    // Generate random number of colors (2-5, not too many)
    const numColors = Math.floor(Math.random() * 4) + 2; // 2-5 colors

    // Generate random colors
    const colors = [];
    for (let i = 0; i < numColors; i++) {
        colors.push('#' + randomHexColor());
    }

    // Generate random weights (percentages)
    const weights = [];
    for (let i = 0; i < numColors; i++) {
        weights.push(Math.floor(Math.random() * 80) + 10); // 10-90%
    }

    // Normalize weights to sum to 100
    const totalWeight = weights.reduce((sum, w) => sum + w, 0);
    const normalizedWeights = weights.map(w => Math.round((w / totalWeight) * 100));

    // Clear existing color inputs
    const colorInputs = document.getElementById('colorInputs');
    if (!colorInputs) return;
    colorInputs.innerHTML = '';

    // Add new colors with weights
    colors.forEach((color, index) => {
        addColorRow(color, normalizedWeights[index]);
    });

    // Update color palette
    window.updateColorPalette();

    console.log('[Roll Dice] Generated query with', numColors, 'colors:', colors, 'weights:', normalizedWeights);
    window.showSuccess('🎲 Roll Dice', `Generated random query with ${numColors} colors`);
};

// ============================================================================
// IMAGE UPLOAD AND COLOR EXTRACTION FUNCTIONS
// ============================================================================

/**
 * Show image upload modal
 */
window.showImageUploadModal = function () {
    const modal = document.getElementById('imageUploadModal');
    if (modal) {
        // Reset modal state
        const preview = document.getElementById('imagePreview');
        const previewImg = document.getElementById('uploadedImagePreview');
        const extractionStatus = document.getElementById('extractionStatus');
        const fileInput = document.getElementById('imageFileInput');

        if (preview) preview.style.display = 'none';
        if (previewImg) previewImg.src = '';
        if (extractionStatus) extractionStatus.style.display = 'none';
        if (fileInput) fileInput.value = '';

        // Ensure modal is properly centered and displayed
        modal.style.display = 'flex';
        modal.style.position = 'absolute';
        modal.style.top = '50%';
        modal.style.left = '50%';
        modal.style.transform = 'translate(-50%, -50%)';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.zIndex = '9999';
        modal.style.flexDirection = 'column';
        modal.style.justifyContent = 'center';
        modal.style.alignItems = 'center';
        modal.style.padding = '20px';
        modal.style.boxSizing = 'border-box';

        console.log('[Image Upload] Modal opened');
    }
};

/**
 * Close image upload modal
 */
window.closeImageUploadModal = function () {
    const modal = document.getElementById('imageUploadModal');
    if (modal) {
        modal.style.display = 'none';
        console.log('[Image Upload] Modal closed');
    }
};

/**
 * Handle drag over event
 */
window.handleDragOver = function (event) {
    event.preventDefault();
    event.stopPropagation();
    const uploadArea = document.getElementById('imageUploadArea');
    if (uploadArea) {
        uploadArea.style.borderColor = 'var(--teal)';
        uploadArea.style.backgroundColor = 'var(--surface1)';
    }
};

/**
 * Handle drag leave event
 */
window.handleDragLeave = function (event) {
    event.preventDefault();
    event.stopPropagation();
    const uploadArea = document.getElementById('imageUploadArea');
    if (uploadArea) {
        uploadArea.style.borderColor = 'var(--surface2)';
        uploadArea.style.backgroundColor = 'var(--surface0)';
    }
};

/**
 * Handle drop event
 */
window.handleDrop = function (event) {
    event.preventDefault();
    event.stopPropagation();

    const uploadArea = document.getElementById('imageUploadArea');
    if (uploadArea) {
        uploadArea.style.borderColor = 'var(--surface2)';
        uploadArea.style.backgroundColor = 'var(--surface0)';
    }

    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
        handleImageFile(files[0]);
    }
};

/**
 * Handle file select event
 */
window.handleFileSelect = function (event) {
    const files = event.target.files;
    if (files && files.length > 0) {
        handleImageFile(files[0]);
    }
};

/**
 * Handle image file selection
 */
function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        window.showError('Invalid File', 'Please select an image file (JPG, PNG, GIF, WebP)');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = function (e) {
        const preview = document.getElementById('imagePreview');
        const previewImg = document.getElementById('uploadedImagePreview');

        if (preview && previewImg) {
            previewImg.src = e.target.result;
            preview.style.display = 'block';
        }

        // Store file for extraction
        window.selectedImageFile = file;
        console.log('[Image Upload] File selected:', file.name, 'Size:', (file.size / 1024).toFixed(2), 'KB');
    };

    reader.readAsDataURL(file);
}

/**
 * Extract colors from uploaded image
 */
window.extractColorsFromImage = async function () {
    const file = window.selectedImageFile;
    if (!file) {
        window.showError('No Image', 'Please select an image file first');
        return;
    }

    const numColors = parseInt(document.getElementById('numColorsToExtract')?.value) || 5;
    if (numColors < 1 || numColors > 10) {
        window.showError('Invalid Number', 'Number of colors must be between 1 and 10');
        return;
    }

    const extractionStatus = document.getElementById('extractionStatus');
    if (extractionStatus) {
        extractionStatus.style.display = 'block';
    }

    try {
        console.log('[Image Upload] Extracting', numColors, 'colors from image...');

        // Create FormData for file upload
        const formData = new FormData();
        formData.append('image', file);
        formData.append('num_colors', numColors);

        // Upload and extract colors
        const response = await fetch('/extract_colors', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to extract colors' }));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();

        if (!data.colors || !Array.isArray(data.colors) || data.colors.length === 0) {
            throw new Error('No colors extracted from image');
        }

        console.log('[Image Upload] Extracted colors:', data.colors);
        console.log('[Image Upload] Color weights:', data.weights);

        // Clear existing color inputs
        const colorInputs = document.getElementById('colorInputs');
        if (!colorInputs) return;
        colorInputs.innerHTML = '';

        // Add extracted colors with their weights
        data.colors.forEach((color, index) => {
            // Ensure color is in hex format
            let hexColor = color;
            if (!color.startsWith('#')) {
                hexColor = '#' + color;
            }

            // Convert weight percentage to slider value (0-100)
            const weight = data.weights && data.weights[index] ? Math.round(data.weights[index] * 100) : 100;
            addColorRow(hexColor, weight);
        });

        // Update color palette
        window.updateColorPalette();

        // Close modal
        window.closeImageUploadModal();

        // Hide extraction status
        if (extractionStatus) {
            extractionStatus.style.display = 'none';
        }

        window.showSuccess('🎨 Colors Extracted', `Extracted ${data.colors.length} colors from image`);

    } catch (error) {
        console.error('[Image Upload] Error extracting colors:', error);
        window.showError('Extraction Failed', error.message || 'Failed to extract colors from image');

        if (extractionStatus) {
            extractionStatus.style.display = 'none';
        }
    }
};

