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
    const colorPickers = document.querySelectorAll('.color-picker');
    window.colors = Array.from(colorPickers).map(picker => picker.value);
    console.log('Colors updated:', window.colors);
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
                        color: white;
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
                    color: white;
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
                    color: white;
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
    generateButton.style.color = 'white';
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
    downloadButton.style.color = 'white';
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
        distanceIndicator.style.position = 'absolute'; distanceIndicator.style.bottom = '0'; distanceIndicator.style.left = '0'; distanceIndicator.style.right = '0'; distanceIndicator.style.padding = '4px'; distanceIndicator.style.backgroundColor = 'rgba(0,0,0,0.7)'; distanceIndicator.style.color = 'white'; distanceIndicator.style.fontSize = '10px'; distanceIndicator.style.textAlign = 'center'; distanceIndicator.textContent = `d: ${distance}`;
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

    try {
        // Get color inputs
        const colorRows = document.querySelectorAll('.color-row');
        if (colorRows.length === 0) {
            throw new Error('Please add at least one color before searching');
        }

        const colors = [];
        const weights = [];
        let totalWeight = 0;

        colorRows.forEach(row => {
            const picker = row.querySelector('.color-picker');
            const slider = row.querySelector('.weight-slider');

            if (picker && slider) {
                const color = picker.value.substring(1); // Remove # from hex color
                const weight = parseInt(slider.value);
                colors.push(color);
                weights.push(weight);
                totalWeight += weight;
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

        // Update UI with results
        if (resultsSection) resultsSection.style.display = 'block';
        if (visualizationSection) visualizationSection.style.display = 'block';
        loading.style.display = 'none';
        searchBtn.disabled = false;
        searchBtn.textContent = '🔍 Search Images';

        window.showSuccess('Search Complete', `Found ${data.results.length} matching images`);
        window.updateSearchResults(data);
        window.updateVisualization(data);

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
        overlay.style.background = 'rgba(0, 0, 0, 0.7)';
        overlay.style.display = 'flex';
        overlay.style.alignItems = 'center';
        overlay.style.justifyContent = 'center';
        overlay.style.opacity = '0';
        overlay.style.transition = 'opacity 0.3s ease';
        overlay.style.color = 'white';
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
        downloadBtn.onclick = () => {
            const link = document.createElement('a');
            link.href = imgSrc;
            link.download = result.image_id || 'image';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
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
            <button style="position: absolute; top: 20px; right: 30px; background: rgba(0, 0, 0, 0.7); color: white; border: none; font-size: 30px; cursor: pointer; border-radius: 50%; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center;">&times;</button>
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
        <div class="modal-header" style="display: flex; justify-content: space-between; align-items: center; padding: 20px; border-bottom: 1px solid var(--surface2);">
            <h2 style="margin: 0; color: var(--text); font-family: 'JetBrainsMono Nerd Font Mono', monospace;">Image Details - Rank #${rank}</h2>
            <button class="close" style="background: none; border: none; font-size: 24px; color: var(--text); cursor: pointer; padding: 0; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;">&times;</button>
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

    modal.innerHTML = modalContent;

    // Apply 70% size and center the modal
    modal.style.cssText = `
        display: block;
        position: fixed;
        z-index: 1000;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 70%;
        max-width: 800px;
        max-height: 85vh;
        background: var(--base);
        border: 1px solid var(--surface2);
        border-radius: 12px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        overflow: hidden;
    `;

    // Add event listener to close button
    const closeBtn = modal.querySelector('.close');
    if (closeBtn) {
        closeBtn.onclick = function () {
            modal.style.display = 'none';
        };
    }

    // Close modal when clicking outside
    const closeModal = (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    };

    // Remove previous event listeners and add new one
    modal.onclick = closeModal;
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

    topResults.forEach(result => {
        if (result.dominant_colors && Array.isArray(result.dominant_colors)) {
            allColors.push(...result.dominant_colors);
        }
    });

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
    useBtn.style.color = 'white';
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
    // We'll handle the event listener in a unified way below
    // Not adding event listener here to avoid duplicates

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

    // Append all elements to the color row
    // Create info container for extra color details
    const infoContainer = document.createElement('div');
    infoContainer.className = 'color-info-container';
    infoContainer.style.display = 'flex';
    infoContainer.style.flexDirection = 'column';
    infoContainer.style.marginLeft = '10px';

    // Add color name with enhanced styling
    colorName.style.fontWeight = 'bold';
    infoContainer.appendChild(colorName);

    // Add RGB values
    const rgb = hexToRgb(color);
    const rgbInfo = document.createElement('span');
    rgbInfo.className = 'color-rgb-info';
    rgbInfo.style.fontSize = '11px';
    rgbInfo.style.color = 'var(--subtext0)';
    rgbInfo.textContent = `RGB: ${rgb.r}, ${rgb.g}, ${rgb.b}`;
    infoContainer.appendChild(rgbInfo);

    // Add HSL values
    const hsl = hexToHsl(color);
    const hslInfo = document.createElement('span');
    hslInfo.className = 'color-hsl-info';
    hslInfo.style.fontSize = '11px';
    hslInfo.style.color = 'var(--subtext0)';
    hslInfo.textContent = `HSL: ${Math.round(hsl.h)}°, ${Math.round(hsl.s * 100)}%, ${Math.round(hsl.l * 100)}%`;
    infoContainer.appendChild(hslInfo);

    // Add complementary color info
    const complementaryColor = getComplementaryColor(color);
    const complementaryName = getColorName(complementaryColor);

    const complementaryContainer = document.createElement('div');
    complementaryContainer.className = 'complementary-color';
    complementaryContainer.style.display = 'flex';
    complementaryContainer.style.alignItems = 'center';
    complementaryContainer.style.marginTop = '4px';

    const complementarySwatch = document.createElement('div');
    complementarySwatch.style.width = '15px';
    complementarySwatch.style.height = '15px';
    complementarySwatch.style.backgroundColor = complementaryColor;
    complementarySwatch.style.border = '1px solid var(--surface2)';
    complementarySwatch.style.borderRadius = '3px';
    complementarySwatch.style.marginRight = '6px';

    const complementaryInfo = document.createElement('span');
    complementaryInfo.style.fontSize = '11px';
    complementaryInfo.style.color = 'var(--subtext0)';
    complementaryInfo.textContent = `Complementary: ${complementaryName} (${complementaryColor})`;

    complementaryContainer.appendChild(complementarySwatch);
    complementaryContainer.appendChild(complementaryInfo);
    infoContainer.appendChild(complementaryContainer);

    colorRow.appendChild(colorPicker);
    colorRow.appendChild(weightSlider);
    colorRow.appendChild(weightValue);
    colorRow.appendChild(removeButton);
    colorRow.appendChild(infoContainer);

    // Add styles for color name
    const colorNameElement = colorRow.querySelector('.color-name');
    if (colorNameElement) {
        colorNameElement.style.fontSize = '12px';
        colorNameElement.style.color = 'var(--subtext0)';
        colorNameElement.style.marginLeft = '10px';
        colorNameElement.style.fontStyle = 'italic';
        colorNameElement.style.display = 'inline-block'; // Ensure visibility
    }

    colorInputs.appendChild(colorRow);

    // Add event listeners to new elements
    const newColorPicker = colorRow.querySelector('.color-picker');
    const newWeightSlider = colorRow.querySelector('.weight-slider');
    const weightDisplay = colorRow.querySelector('.weight-value');
    const colorNameDisplay = colorRow.querySelector('.color-name');

    if (newColorPicker) {
        // Attach the unified handler
        newColorPicker.addEventListener('change', window.handleColorPickerChange);
        newColorPicker.addEventListener('input', window.handleColorPickerChange); // For live updates
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
window.generateColorSpaceNavigator = async function() {
    await loadVisualizationModule('colorSpaceNavigator');
    // Module has loaded and overwritten this function, so call it
    // Store reference before calling to avoid recursion
    const func = window.generateColorSpaceNavigator;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.generateHistogramCloud = async function() {
    await loadVisualizationModule('histogramCloud');
    const func = window.generateHistogramCloud;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.generateSimilarityLandscape = async function() {
    await loadVisualizationModule('similarityLandscape');
    const func = window.generateSimilarityLandscape;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.generateRerankingAnimation = async function() {
    await loadVisualizationModule('rerankingAnimation');
    const func = window.generateRerankingAnimation;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.generateImageGlobe = async function() {
    await loadVisualizationModule('imageGlobe');
    const func = window.generateImageGlobe;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.generateConnectionsGraph = async function() {
    await loadVisualizationModule('connectionsGraph');
    const func = window.generateConnectionsGraph;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

// Export function wrappers
window.exportColorSpaceData = async function() {
    await loadVisualizationModule('colorSpaceNavigator');
    const func = window.exportColorSpaceData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.exportHistogramData = async function() {
    await loadVisualizationModule('histogramCloud');
    const func = window.exportHistogramData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.exportSimilarityData = async function() {
    await loadVisualizationModule('similarityLandscape');
    const func = window.exportSimilarityData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.exportAnimationData = async function() {
    await loadVisualizationModule('rerankingAnimation');
    const func = window.exportAnimationData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.exportGlobeData = async function() {
    await loadVisualizationModule('imageGlobe');
    const func = window.exportGlobeData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

window.exportGraphData = async function() {
    await loadVisualizationModule('connectionsGraph');
    const func = window.exportGraphData;
    if (func && func !== arguments.callee) {
        return func.apply(this, arguments);
    }
};

// New visualizations wrappers
window.generateOTTransport3D = async function(){ await loadVisualizationModule('otTransport3D'); const func = window.generateOTTransport3D; if(func && func !== arguments.callee){ return func.apply(this, arguments);} };
window.exportOTTransportData = async function(){ await loadVisualizationModule('otTransport3D'); const func = window.exportOTTransportData; if(func && func !== arguments.callee){ return func.apply(this, arguments);} };

window.generateHNSWExplorer = async function(){ await loadVisualizationModule('hnswGraphExplorer'); const func = window.generateHNSWExplorer; if(func && func !== arguments.callee){ return func.apply(this, arguments);} };
window.exportHNSWData = async function(){ await loadVisualizationModule('hnswGraphExplorer'); const func = window.exportHNSWData; if(func && func !== arguments.callee){ return func.apply(this, arguments);} };

window.generateColorDensityVolume = async function(){ await loadVisualizationModule('colorDensityVolume'); const func = window.generateColorDensityVolume; if(func && func !== arguments.callee){ return func.apply(this, arguments);} };
window.exportColorDensityData = async function(){ await loadVisualizationModule('colorDensityVolume'); const func = window.exportColorDensityData; if(func && func !== arguments.callee){ return func.apply(this, arguments);} };

window.generateImageThumbnails3D = async function(){ await loadVisualizationModule('imageThumbnails3D'); const func = window.generateImageThumbnails3D; if(func && func !== arguments.callee){ return func.apply(this, arguments);} };
window.exportThumbnailsData = async function(){ await loadVisualizationModule('imageThumbnails3D'); const func = window.exportThumbnailsData; if(func && func !== arguments.callee){ return func.apply(this, arguments);} };

// ============================================================================
// 3D VISUALIZATION SHARED UTILITIES
// ============================================================================

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
window.clear3DVisualization = function() {
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
window.show3DLoading = function() {
    const loading = document.getElementById('3d-loading');
    const placeholder = document.getElementById('3d-placeholder');
    if (loading) loading.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
};

/**
 * Hide loading indicator
 */
window.hide3DLoading = function() {
    const loading = document.getElementById('3d-loading');
    const placeholder = document.getElementById('3d-placeholder');
    if (loading) loading.style.display = 'none';
    if (placeholder) placeholder.style.display = 'none';
};

/**
 * Get container for 3D visualization
 */
window.get3DContainer = function() {
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
window.randomSimilarityColors = function() {
    const numColors = Math.floor(Math.random() * 3) + 1; // 1-3 colors
    const colors = Array.from({ length: numColors }, () => randomHexColor());
    document.getElementById('similarityColors').value = colors.join(',');
    console.log('[Random] Generated similarity colors:', colors.join(','));
};

window.randomAnimationColors = function() {
    const numColors = Math.floor(Math.random() * 3) + 1;
    const colors = Array.from({ length: numColors }, () => randomHexColor());
    document.getElementById('animationColors').value = colors.join(',');
    console.log('[Random] Generated animation colors:', colors.join(','));
};

/**
 * Generate random weights string
 */
window.randomSimilarityWeights = function() {
    const numWeights = document.getElementById('similarityColors').value.split(',').length || 1;
    const weights = Array.from({ length: numWeights }, () => (Math.random() * 0.8 + 0.1).toFixed(2));
    const sum = weights.reduce((a, b) => parseFloat(a) + parseFloat(b), 0);
    const normalized = weights.map(w => (parseFloat(w) / sum).toFixed(3));
    document.getElementById('similarityWeights').value = normalized.join(',');
    console.log('[Random] Generated similarity weights:', normalized.join(','));
};

window.randomAnimationWeights = function() {
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
window.randomHistogramImageId = async function() {
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
window.randomOTColors = function() {
    const numColors = Math.floor(Math.random() * 3) + 1;
    const colors = Array.from({ length: numColors }, () => randomHexColor());
    const el = document.getElementById('otColors'); if (el) el.value = colors.join(',');
    console.log('[Random] Generated OT colors:', colors.join(','));
};
window.randomOTWeights = function() {
    const el = document.getElementById('otColors');
    const num = el?.value ? el.value.split(',').length : 1;
    const ws = Array.from({ length: num }, () => (Math.random() * 0.8 + 0.1).toFixed(2));
    const sum = ws.reduce((a,b)=>parseFloat(a)+parseFloat(b),0);
    const norm = ws.map(w => (parseFloat(w)/sum).toFixed(3));
    const we = document.getElementById('otWeights'); if (we) we.value = norm.join(',');
    console.log('[Random] Generated OT weights:', norm.join(','));
};

// Random for HNSW Explorer
window.randomHNSWColors = function() {
    const numColors = Math.floor(Math.random() * 3) + 1;
    const colors = Array.from({ length: numColors }, () => randomHexColor());
    const el = document.getElementById('hnswColors'); if (el) el.value = colors.join(',');
    console.log('[Random] Generated HNSW colors:', colors.join(','));
};
window.randomHNSWWeights = function() {
    const el = document.getElementById('hnswColors');
    const num = el?.value ? el.value.split(',').length : 1;
    const ws = Array.from({ length: num }, () => (Math.random() * 0.8 + 0.1).toFixed(2));
    const sum = ws.reduce((a,b)=>parseFloat(a)+parseFloat(b),0);
    const norm = ws.map(w => (parseFloat(w)/sum).toFixed(3));
    const we = document.getElementById('hnswWeights'); if (we) we.value = norm.join(',');
    console.log('[Random] Generated HNSW weights:', norm.join(','));
};

// ============================================================================
// 3D VISUALIZATION CONTROLS
// ============================================================================

window.toggle3DVisualization = function() {
    window.current3DVisualization.isPaused = !window.current3DVisualization.isPaused;
    console.log('[3D Controls] Visualization', window.current3DVisualization.isPaused ? 'paused' : 'resumed');
};

window.reset3DVisualization = function() {
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

window.turnOff3DVisualization = function() {
    console.log('[3D Controls] Turning off visualization');
    window.clear3DVisualization();
};

// Show the suggest palette modal
window.showSuggestPaletteModal = function () {
    const modal = document.getElementById('suggestPaletteModal');
    if (modal) {
        // Reset any previously generated palette
        const generatedPaletteDiv = document.getElementById('generatedPalette');
        if (generatedPaletteDiv) {
            generatedPaletteDiv.style.display = 'none';
        }
        window.generatedPalette = null;

        // Add hover effects to scheme modes
        const schemeModes = modal.querySelectorAll('.scheme-mode');
        schemeModes.forEach(mode => {
            mode.addEventListener('mouseover', () => {
                mode.style.transform = 'scale(1.02)';
                mode.style.borderColor = 'var(--mauve)';
            });
            mode.addEventListener('mouseout', () => {
                mode.style.transform = 'scale(1)';
                mode.style.borderColor = 'var(--surface2)';
            });
        });

        modal.style.display = 'block';
    }
};

// Generate a new color palette based on the selected scheme
window.generatePalette = function (scheme) {
    console.log('Generating palette for scheme:', scheme);

    // Get the currently selected colors
    const currentColors = window.colors || [];
    let baseColor = currentColors[0] || '#FF0000';
    if (baseColor.startsWith('#')) {
        baseColor = baseColor.substring(1);
    }

    let newColors = [];
    switch (scheme) {
        case 'monochromatic':
            newColors = generateMonochromaticScheme(baseColor);
            break;
        case 'complementary':
            newColors = [baseColor, getComplementaryColor(baseColor)];
            break;
        case 'analogous':
            newColors = generateAnalogousScheme(baseColor);
            break;
        case 'triadic':
            newColors = generateTriadicScheme(baseColor);
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
            break;
        case 'random':
            // Generate a random harmonious palette
            const randomHue = Math.floor(Math.random() * 360);
            newColors = [
                hslToHex(randomHue, 70, 50),
                hslToHex((randomHue + 120) % 360, 70, 50),
                hslToHex((randomHue + 240) % 360, 70, 50),
                hslToHex(randomHue, 85, 35),
                hslToHex(randomHue, 60, 65)
            ];
            break;
    }

    // Store the generated palette
    window.generatedPalette = newColors;

    // Show the preview
    const previewDiv = document.getElementById('palettePreview');
    const generatedPaletteDiv = document.getElementById('generatedPalette');

    if (previewDiv && generatedPaletteDiv) {
        // Clear previous preview
        previewDiv.innerHTML = '';

        // Create swatches for each color
        newColors.forEach(color => {
            if (!color.startsWith('#')) color = '#' + color;

            const swatch = document.createElement('div');
            swatch.style.flex = '1';
            swatch.style.height = '60px';
            swatch.style.borderRadius = '6px';
            swatch.style.backgroundColor = color;
            swatch.style.position = 'relative';
            swatch.style.border = '2px solid var(--surface2)';
            swatch.style.cursor = 'pointer';
            swatch.title = `${color} - ${getColorName(color)}`;

            // Add color name and hex
            const colorInfo = document.createElement('div');
            colorInfo.style.position = 'absolute';
            colorInfo.style.bottom = '0';
            colorInfo.style.left = '0';
            colorInfo.style.right = '0';
            colorInfo.style.background = 'rgba(0, 0, 0, 0.7)';
            colorInfo.style.color = 'white';
            colorInfo.style.padding = '4px';
            colorInfo.style.fontSize = '10px';
            colorInfo.style.textAlign = 'center';
            colorInfo.style.borderRadius = '0 0 4px 4px';
            colorInfo.textContent = color;

            swatch.appendChild(colorInfo);

            // Click to copy
            swatch.addEventListener('click', () => {
                navigator.clipboard.writeText(color);
                swatch.style.transform = 'scale(1.05)';
                setTimeout(() => swatch.style.transform = 'scale(1)', 200);
            });

            previewDiv.appendChild(swatch);
        });

        // Show the generated palette section
        generatedPaletteDiv.style.display = 'block';
    }
};

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
// Function to run Color Analysis tool
window.runColorAnalysis = function () {
    console.log('Running Color Analysis tool');
    const panel = document.getElementById('colorAnalysisPanel');
    if (panel) {
        panel.style.display = 'block';

        // Initialize the color input with current colors if available
        const colorAnalysisInput = document.getElementById('colorAnalysisInput');
        if (colorAnalysisInput && window.colors && window.colors.length > 0) {
            colorAnalysisInput.value = window.colors.join(', ');
        }
    } else {
        window.showError('Tool Error', 'Color Analysis panel not found');
    }
};

// Function to run a quick test of the Color Analysis tool
window.quickTestColorAnalysis = function () {
    console.log('Running quick test of Color Analysis tool');

    // Set some sample colors for the test
    const testColors = ['#F5C2E7', '#CBA6F7', '#F2CDCD', '#A6E3A1', '#F9E2AF'];

    // Set up the color analysis input
    const colorAnalysisInput = document.getElementById('colorAnalysisInput');
    if (colorAnalysisInput) {
        colorAnalysisInput.value = testColors.join(', ');
    }

    // Display the panel if it's not already visible
    const panel = document.getElementById('colorAnalysisPanel');
    if (panel) {
        panel.style.display = 'block';
    }

    // Execute the analysis with sample data
    executeColorAnalysis();
};

// Function to execute color analysis
function executeColorAnalysis() {
    console.log('Executing color analysis');

    // Get input values
    const colorInput = document.getElementById('colorAnalysisInput')?.value;
    if (!colorInput) {
        window.showError('Analysis Error', 'Please enter at least one color');
        return;
    }

    // Parse color input (expecting hex colors separated by commas or spaces)
    const colors = colorInput.split(/[\s,]+/).filter(c => c).map(c => {
        // Ensure colors start with # 
        return c.startsWith('#') ? c : `#${c}`;
    });

    if (colors.length === 0) {
        window.showError('Analysis Error', 'No valid colors found');
        return;
    }

    // Get analysis options (if elements exist)
    const analyzeHarmony = document.getElementById('analyzeHarmony')?.checked ?? true;
    const analyzeRelationships = document.getElementById('analyzeRelationships')?.checked ?? true;
    const analyzeContrast = document.getElementById('analyzeContrast')?.checked ?? true;

    // Analyze colors
    const results = analyzeColors(colors, {
        harmony: analyzeHarmony,
        relationships: analyzeRelationships,
        contrast: analyzeContrast
    });

    // Display results
    displayColorAnalysisResults(results);
}

// Function to analyze colors
function analyzeColors(colors, options) {
    const results = {
        colors: colors,
        colorNames: colors.map(c => getColorName(c)),
        harmonyScore: 0,
        relationships: [],
        contrastScores: [],
        suggestions: []
    };

    // Calculate harmony score (simple implementation)
    if (options.harmony) {
        const hslColors = colors.map(c => hexToHsl(c));

        // Check hue distances for harmony
        let totalHarmony = 0;
        let pairs = 0;

        for (let i = 0; i < hslColors.length; i++) {
            for (let j = i + 1; j < hslColors.length; j++) {
                const hue1 = hslColors[i].h;
                const hue2 = hslColors[j].h;

                // Calculate smallest distance between hues (0-180 degrees)
                const hueDiff = Math.min(
                    Math.abs(hue1 - hue2),
                    360 - Math.abs(hue1 - hue2)
                );

                // Different harmony rules
                // - Complementary: ~180 degrees
                // - Analogous: <30 degrees
                // - Triadic: ~120 degrees

                // Calculate harmony contribution (higher = more harmonious)
                const complementaryScore = 1 - Math.abs(hueDiff - 180) / 180;
                const analogousScore = 1 - Math.min(hueDiff, 60) / 60;
                const triadicScore = 1 - Math.abs(hueDiff - 120) / 120;

                // Take the highest harmony score
                const harmonyContribution = Math.max(
                    complementaryScore,
                    analogousScore,
                    triadicScore
                );

                totalHarmony += harmonyContribution;
                pairs++;

                // Add relationship
                let relationship = "Mixed";
                if (complementaryScore > 0.8) relationship = "Complementary";
                else if (analogousScore > 0.8) relationship = "Analogous";
                else if (triadicScore > 0.8) relationship = "Triadic";

                results.relationships.push({
                    color1: colors[i],
                    color2: colors[j],
                    relationship: relationship,
                    harmonyScore: harmonyContribution
                });
            }
        }

        // Calculate overall harmony (0-100%)
        results.harmonyScore = pairs > 0 ? (totalHarmony / pairs) * 100 : 100;
    }

    // Calculate contrast ratios
    if (options.contrast && colors.length >= 2) {
        for (let i = 0; i < colors.length; i++) {
            for (let j = i + 1; j < colors.length; j++) {
                // Simple luminance calculation for contrast (not fully WCAG compliant)
                const rgb1 = hexToRgb(colors[i]);
                const rgb2 = hexToRgb(colors[j]);

                // Simplified relative luminance calculation
                const lum1 = 0.2126 * rgb1.r / 255 + 0.7152 * rgb1.g / 255 + 0.0722 * rgb1.b / 255;
                const lum2 = 0.2126 * rgb2.r / 255 + 0.7152 * rgb2.g / 255 + 0.0722 * rgb2.b / 255;

                // Calculate contrast ratio
                const ratio = (Math.max(lum1, lum2) + 0.05) / (Math.min(lum1, lum2) + 0.05);

                results.contrastScores.push({
                    color1: colors[i],
                    color2: colors[j],
                    ratio: ratio.toFixed(2),
                    isAccessible: ratio >= 4.5
                });
            }
        }
    }

    // Generate suggestions based on harmony analysis
    const baseColor = colors[0];

    // Add complementary color suggestion
    results.suggestions.push({
        type: "Complementary",
        colors: [baseColor, getComplementaryColor(baseColor)]
    });

    // Add analogous suggestion
    results.suggestions.push({
        type: "Analogous",
        colors: generateAnalogousScheme(baseColor)
    });

    // Add triadic suggestion
    results.suggestions.push({
        type: "Triadic",
        colors: generateTriadicScheme(baseColor)
    });

    return results;
}

// Function to display color analysis results
function displayColorAnalysisResults(results) {
    const resultsContainer = document.getElementById('colorAnalysisResultsContent');
    const resultsArea = document.getElementById('colorAnalysisResults');

    if (!resultsContainer || !resultsArea) {
        window.showError('Display Error', 'Results container not found');
        return;
    }

    // Show results area
    resultsArea.style.display = 'block';

    // Build HTML for results
    let html = `
        <div style="margin-bottom: 20px;">
            <h3 style="color: var(--text); margin-bottom: 10px;">Color Palette Analysis</h3>
            
            <!-- Color Swatches -->
            <div style="display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap;">
                ${results.colors.map((color, idx) => `
                    <div style="text-align: center;">
                        <div style="
                            width: 60px;
                            height: 60px;
                            background-color: ${color};
                            border-radius: 8px;
                            margin-bottom: 5px;
                            border: 2px solid var(--surface2);
                        "></div>
                        <div style="font-size: 12px; color: var(--text);">${color}</div>
                        <div style="font-size: 11px; color: var(--subtext0);">${results.colorNames[idx]}</div>
                    </div>
                `).join('')}
            </div>
            
            <!-- Harmony Score -->
            <div style="background: var(--surface0); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="color: var(--text); margin: 0 0 10px 0;">Harmony Score</h4>
                <div style="position: relative; height: 20px; background: var(--surface2); border-radius: 10px; overflow: hidden;">
                    <div style="position: absolute; top: 0; left: 0; height: 100%; width: ${results.harmonyScore.toFixed(1)}%; background: var(--green); border-radius: 10px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span style="font-size: 12px; color: var(--subtext1);">0%</span>
                    <span style="font-size: 12px; font-weight: bold; color: var(--text);">${results.harmonyScore.toFixed(1)}%</span>
                    <span style="font-size: 12px; color: var(--subtext1);">100%</span>
                </div>
                <div style="text-align: center; margin-top: 10px; font-size: 14px; color: var(--text);">
                    ${results.harmonyScore > 80 ? 'Very Harmonious' :
            results.harmonyScore > 60 ? 'Good Harmony' :
                results.harmonyScore > 40 ? 'Moderate Harmony' : 'Low Harmony'}
                </div>
            </div>
    `;

    // Color Relationships
    if (results.relationships.length > 0) {
        html += `
            <div style="background: var(--surface0); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="color: var(--text); margin: 0 0 10px 0;">Color Relationships</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <thead>
                        <tr>
                            <th style="text-align: left; padding: 5px; color: var(--subtext0);">Colors</th>
                            <th style="text-align: left; padding: 5px; color: var(--subtext0);">Relationship</th>
                            <th style="text-align: right; padding: 5px; color: var(--subtext0);">Harmony</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        results.relationships.forEach(rel => {
            html += `
                <tr>
                    <td style="padding: 5px;">
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <div style="width: 15px; height: 15px; background-color: ${rel.color1}; border-radius: 3px;"></div>
                            <div style="width: 15px; height: 15px; background-color: ${rel.color2}; border-radius: 3px;"></div>
                        </div>
                    </td>
                    <td style="padding: 5px; color: var(--text);">${rel.relationship}</td>
                    <td style="padding: 5px; text-align: right; color: var(--text);">${(rel.harmonyScore * 100).toFixed(0)}%</td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;
    }

    // Contrast Analysis
    if (results.contrastScores.length > 0) {
        html += `
            <div style="background: var(--surface0); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="color: var(--text); margin: 0 0 10px 0;">Contrast Analysis</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <thead>
                        <tr>
                            <th style="text-align: left; padding: 5px; color: var(--subtext0);">Colors</th>
                            <th style="text-align: center; padding: 5px; color: var(--subtext0);">Ratio</th>
                            <th style="text-align: right; padding: 5px; color: var(--subtext0);">Accessibility</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        results.contrastScores.forEach(score => {
            html += `
                <tr>
                    <td style="padding: 5px;">
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <div style="width: 15px; height: 15px; background-color: ${score.color1}; border-radius: 3px;"></div>
                            <div style="width: 15px; height: 15px; background-color: ${score.color2}; border-radius: 3px;"></div>
                        </div>
                    </td>
                    <td style="padding: 5px; text-align: center; color: var(--text);">${score.ratio}:1</td>
                    <td style="padding: 5px; text-align: right; color: ${score.isAccessible ? 'var(--green)' : 'var(--red)'};">
                        ${score.isAccessible ? '✓ WCAG AA' : '✗ Failed'}
                    </td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;
    }

    // Suggestions
    html += `
        <div style="background: var(--surface0); padding: 15px; border-radius: 8px;">
            <h4 style="color: var(--text); margin: 0 0 15px 0;">Color Scheme Suggestions</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
    `;

    results.suggestions.forEach(suggestion => {
        html += `
            <div style="background: var(--surface1); padding: 10px; border-radius: 6px;">
                <h5 style="margin: 0 0 10px 0; color: var(--text);">${suggestion.type}</h5>
                <div style="height: 30px; display: flex; border-radius: 4px; overflow: hidden; margin-bottom: 10px;">
                    ${suggestion.colors.map(color => `
                        <div style="flex: 1; background-color: ${color};"></div>
                    `).join('')}
                </div>
                <button onclick="applySuggestion('${suggestion.colors.join(',')}')" style="
                    background: var(--blue);
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    width: 100%;
                ">Apply</button>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;

    // Set the HTML content
    resultsContainer.innerHTML = html;
}

// Function to apply a suggested color scheme
window.applySuggestion = function (colorsStr) {
    const colors = colorsStr.split(',');
    if (colors.length === 0) return;

    // Clear existing color inputs
    const colorInputs = document.getElementById('colorInputs');
    if (!colorInputs) return;
    colorInputs.innerHTML = '';

    // Add colors from suggestion
    colors.forEach((color, index) => {
        const weight = index === 0 ? 100 : Math.max(20, 100 - (index * 15));
        addColorRow(color, weight);
    });

    // Update color palette
    window.updateColorPalette();

    window.showSuccess('Colors Applied', `Applied ${colors.length} colors from suggestion`);
};

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
        case 'color-analysis':
            content = `
                <h2>Color Analysis Tool</h2>
                <p>The Color Analysis tool helps you analyze relationships, harmony scores, and palette characteristics of colors with comprehensive metrics and visual feedback.</p>
                
                <h3>Key Features</h3>
                <ul>
                    <li><strong>Color harmony score calculation</strong> - Evaluates how well colors work together based on color theory principles</li>
                    <li><strong>Color relationship analysis</strong> - Identifies complementary, analogous, triadic, and other relationships</li>
                    <li><strong>Contrast ratio evaluation</strong> - Calculates contrast ratios between colors for accessibility compliance</li>
                    <li><strong>Color scheme suggestions</strong> - Generates harmonious color scheme alternatives</li>
                </ul>
                
                <h3>How to Use</h3>
                <ol>
                    <li>Enter hex colors separated by commas or spaces</li>
                    <li>Select the analysis options you need</li>
                    <li>Click "Run Analysis" to generate results</li>
                    <li>Review the harmony scores, relationships, and suggestions</li>
                    <li>Apply suggested color schemes with a single click</li>
                </ol>
                
                <p>This tool is part of Chromatica's comprehensive color analysis and visualization toolkit.</p>
            `;
            break;

        case 'query-visualizer':
            content = `
                <h2>Query Visualizer Tool</h2>
                <p>The Query Visualizer tool creates visual representations of color queries with weighted color bars, color palettes, and comprehensive query summaries.</p>
                
                <h3>Key Features</h3>
                <ul>
                    <li><strong>Generate weighted color bars</strong> representing query colors</li>
                    <li><strong>Create color palette visualizations</strong> with weights</li>
                    <li><strong>Build comprehensive query summary images</strong></li>
                    <li><strong>Export query visualizations</strong> in various formats</li>
                    <li><strong>Customizable color representations</strong></li>
                </ul>
                
                <h3>How to Use</h3>
                <ol>
                    <li>Enter your query colors or use the current search colors</li>
                    <li>Adjust visualization options as needed</li>
                    <li>Generate the visualization</li>
                    <li>Export or save the results</li>
                </ol>
            `;
            break;

        case 'histogram-analysis':
            content = `
                <h2>Histogram Analysis Tool</h2>
                <p>The Histogram Analysis tool provides comprehensive testing and visualization of histogram generation, including validation, performance benchmarking, and distribution analysis.</p>
                
                <h3>Key Features</h3>
                <ul>
                    <li><strong>Visual histogram representation</strong> of color distribution</li>
                    <li><strong>Histogram comparison</strong> between query and results</li>
                    <li><strong>Performance metrics</strong> for histogram generation</li>
                    <li><strong>Distribution analysis</strong> of color histograms</li>
                    <li><strong>Visual validation</strong> of histogram accuracy</li>
                </ul>
                
                <h3>Technical Details</h3>
                <p>Chromatica uses 512-bin LAB color histograms with Hellinger transformation for primary indexing and Earth Mover's Distance for refined ranking.</p>
            `;
            break;

        case 'distance-debugger':
            content = `
                <h2>Distance Debugger Tool</h2>
                <p>The Distance Debugger tool helps analyze Sinkhorn-EMD distance calculations, identify numerical stability issues, and validate distance metrics.</p>
                
                <h3>Key Features</h3>
                <ul>
                    <li><strong>Debug Sinkhorn-EMD calculations</strong></li>
                    <li><strong>Analyze numerical stability</strong> of distance metrics</li>
                    <li><strong>Compare different distance metrics</strong></li>
                    <li><strong>Visualize distance calculations</strong></li>
                    <li><strong>Performance benchmarking</strong> of distance calculations</li>
                </ul>
                
                <h3>Technical Background</h3>
                <p>The Earth Mover's Distance (EMD) is approximated using the Sinkhorn algorithm, providing a balance of accuracy and performance. This tool helps visualize and debug those calculations.</p>
            `;
            break;

        case 'color-explorer':
            content = `
                <h2>Interactive Color Explorer Tool</h2>
                <p>The Interactive Color Explorer provides a powerful interface for exploring color combinations, analyzing color relationships, and experimenting with different color schemes and harmonies.</p>
                
                <h3>Key Features</h3>
                <ul>
                    <li><strong>Interactive color wheel</strong> for exploring harmonious combinations</li>
                    <li><strong>Color scheme generation</strong> based on color theory principles</li>
                    <li><strong>Real-time color relationship analysis</strong></li>
                    <li><strong>Color accessibility testing</strong></li>
                    <li><strong>Export color schemes</strong> in various formats</li>
                </ul>
                
                <h3>How to Use</h3>
                <ol>
                    <li>Select a base color on the color wheel</li>
                    <li>Choose a harmony rule to generate related colors</li>
                    <li>Fine-tune the generated colors</li>
                    <li>Apply the colors to your search or export the scheme</li>
                </ol>
            `;
            break;

        default:
            content = `<h2>Tool Information</h2><p>No specific information available for this tool.</p>`;
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

document.addEventListener('DOMContentLoaded', function () {
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

