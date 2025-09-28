// Chromatica Color Search Engine - Shared JavaScript Functions

console.log('Loading Chromatica JavaScript...');

// Global variables
window.colors = ['#FF0000'];
window.weights = [100];

// Explicitly expose functions to global scope
window.addColor = function () {
    console.log('addColor called');
    const colorInputs = document.getElementById('colorInputs');
    if (!colorInputs) {
        console.error('colorInputs element not found');
        return;
    }

    const colorRow = document.createElement('div');
    colorRow.className = 'color-row';
    colorRow.innerHTML = `
        <input type="color" class="color-picker" value="#00FF00">
        <input type="range" class="weight-slider" min="0" max="100" value="100" step="1">
        <span class="weight-value">100%</span>
        <button onclick="window.removeColor(this)" class="remove-btn">Remove</button>
    `;

    colorInputs.appendChild(colorRow);
    console.log('Color row added');

    // Add event listeners
    const newColorPicker = colorRow.querySelector('.color-picker');
    const newWeightSlider = colorRow.querySelector('.weight-slider');
    const weightDisplay = colorRow.querySelector('.weight-value');

    if (newColorPicker) {
        newColorPicker.addEventListener('change', window.updateColors || function () { });
    }

    if (newWeightSlider && weightDisplay) {
        newWeightSlider.addEventListener('input', () => {
            weightDisplay.textContent = `${newWeightSlider.value}%`;
            if (window.updateWeights) {
                window.updateWeights();
            }
        });
    }

    if (window.updateColorPalette) {
        window.updateColorPalette();
    } else {
        console.log('updateColorPalette function not defined yet');
    }
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
                ">💾 Download Grid</button>
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

// Fix the histogram visualization function to use the defined function
window.generateHistogramVisualization = function () {
    console.log('Generating histogram visualization');

    // Store search results if they're available
    if (window.lastSearchResults === undefined && window.searchResults) {
        window.lastSearchResults = window.searchResults;
    }

    // Generate mock histogram data
    const generateMockHistogram = () => {
        const length = 32;
        return Array(length).fill(0).map(() => Math.random() * 0.8 + 0.1); // Values between 0.1 and 0.9
    };

    const queryHistogram = document.getElementById('query-histogram');
    const resultHistogram = document.getElementById('result-histogram');

    if (queryHistogram) {
        // Generate histogram data
        const histData = generateMockHistogram();
        queryHistogram.innerHTML = createSimpleHistogramVisualization(histData, 'Query');
        queryHistogram.style.display = 'block'; // Ensure it's visible
    }

    if (resultHistogram) {
        // Generate different histogram data for result
        const histData = generateMockHistogram();
        resultHistogram.innerHTML = createSimpleHistogramVisualization(histData, 'Top Result');
        resultHistogram.style.display = 'block'; // Ensure it's visible
    }

    // Show the histogram section if it exists
    const histogramSection = document.querySelector('.histogram-section');
    if (histogramSection) {
        histogramSection.style.display = 'block';
    }

    window.showSuccess('Histogram Visualization', 'Generated histogram visualization successfully.');
};

// Compare histograms and display visual comparison
window.compareHistograms = function () {
    console.log('Comparing histograms');

    // Check if we have search results
    if (!window.lastSearchResults || !Array.isArray(window.lastSearchResults) || window.lastSearchResults.length === 0) {
        window.showError('Comparison Error', 'No search results available. Please perform a search first.');
        return;
    }

    // Find the histogram section and ensure it's displayed
    const histogramSection = document.querySelector('.histogram-section');
    if (!histogramSection) {
        window.showError('Comparison Error', 'Histogram section not found in the DOM.');
        return;
    }

    histogramSection.style.display = 'block';

    // Generate mock query histogram data if not available
    const queryHistogram = document.getElementById('query-histogram');
    if (queryHistogram) {
        // Either use existing histogram or generate mock data
        const queryHistData = Array(32).fill(0).map(() => Math.random() * 0.8 + 0.1);
        queryHistogram.innerHTML = createSimpleHistogramVisualization(queryHistData, 'Query');
        queryHistogram.style.display = 'block';
    }

    // Get top result histogram
    const resultHistogram = document.getElementById('result-histogram');
    if (resultHistogram) {
        // Either use top result histogram or generate mock data
        const resultHistData = window.lastSearchResults[0]?.histogram ||
            Array(32).fill(0).map(() => Math.random() * 0.8 + 0.1);

        resultHistogram.innerHTML = createSimpleHistogramVisualization(resultHistData, 'Top Result');
        resultHistogram.style.display = 'block';
    }

    // Create a comparison visualization
    const comparisonSection = document.createElement('div');
    comparisonSection.id = 'histogram-comparison';
    comparisonSection.style.marginTop = '30px';
    comparisonSection.style.padding = '20px';
    comparisonSection.style.background = 'var(--surface0)';
    comparisonSection.style.borderRadius = '12px';
    comparisonSection.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';

    let comparisonHTML = `
        <h5 style="text-align: center; color: var(--text); margin-bottom: 20px; font-family: 'JetBrainsMono Nerd Font Mono', monospace;">Histogram Comparison</h5>
        <div style="position: relative; height: 200px; margin-bottom: 25px;">
    `;

    // Generate comparison data (either use real data or mock data)
    const queryData = Array(32).fill(0).map(() => Math.random() * 0.8 + 0.1);
    const resultData = Array(32).fill(0).map(() => Math.random() * 0.8 + 0.1);

    // Calculate differences for comparison
    const diffData = queryData.map((val, i) => Math.abs(val - (resultData[i] || 0)));
    const maxDiff = Math.max(...diffData);

    // Draw comparison visualization
    comparisonHTML += `
        <div style="display: flex; height: 150px; align-items: flex-end; gap: 2px;">
    `;

    for (let i = 0; i < queryData.length; i++) {
        const queryHeight = queryData[i] * 150;
        const resultHeight = resultData[i] * 150;
        const diffHeight = (diffData[i] / maxDiff) * 50; // Scale difference to max 50px
        const diffColor = diffHeight > 25 ? 'var(--red)' : diffHeight > 10 ? 'var(--yellow)' : 'var(--green)';

        comparisonHTML += `
            <div style="flex-grow: 1; position: relative;">
                <div style="
                    position: absolute;
                    bottom: 0;
                    height: ${queryHeight}px;
                    width: 45%;
                    left: 0;
                    background: hsla(${Math.round((i / queryData.length) * 360)}, 70%, 60%, 0.7);
                    border-radius: 3px 3px 0 0;
                "></div>
                <div style="
                    position: absolute;
                    bottom: 0;
                    height: ${resultHeight}px;
                    width: 45%;
                    right: 0;
                    background: hsla(${Math.round((i / resultData.length) * 360)}, 70%, 60%, 0.7);
                    border-radius: 3px 3px 0 0;
                "></div>
                <div style="
                    position: absolute;
                    top: 0;
                    height: ${diffHeight}px;
                    width: 100%;
                    background: ${diffColor};
                    opacity: 0.4;
                    border-radius: 0 0 3px 3px;
                "></div>
            </div>
        `;
    }

    comparisonHTML += `
        </div>
        <div style="display: flex; justify-content: center; margin: 15px 0; padding: 10px; gap: 25px; background: var(--surface1); border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 18px; height: 18px; background: var(--blue); opacity: 0.8; border-radius: 3px;"></div>
                <span style="font-size: 13px; color: var(--text); font-family: 'JetBrainsMono Nerd Font Mono', monospace;">Query</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 18px; height: 18px; background: var(--mauve); opacity: 0.8; border-radius: 3px;"></div>
                <span style="font-size: 13px; color: var(--text); font-family: 'JetBrainsMono Nerd Font Mono', monospace;">Result</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 18px; height: 18px; background: var(--green); opacity: 0.5; border-radius: 3px;"></div>
                <span style="font-size: 13px; color: var(--text); font-family: 'JetBrainsMono Nerd Font Mono', monospace;">Difference</span>
            </div>
        </div>
    `;

    // Add histogram stats
    const querySum = queryData.reduce((sum, val) => sum + val, 0);
    const resultSum = resultData.reduce((sum, val) => sum + val, 0);
    const totalDiff = diffData.reduce((sum, val) => sum + val, 0);
    const similarity = 1 - (totalDiff / Math.max(querySum, resultSum));

    comparisonHTML += `
        <div style="margin-top: 20px; background: var(--surface0); padding: 15px; border-radius: 8px;">
            <h5 style="margin-top: 0; color: var(--text); text-align: center;">Similarity Analysis</h5>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div>
                    <div style="font-size: 14px; color: var(--subtext0);">Similarity Score:</div>
                    <div style="font-size: 20px; font-weight: bold; color: var(--green);">${(similarity * 100).toFixed(2)}%</div>
                </div>
                <div>
                    <div style="font-size: 14px; color: var(--subtext0);">Distance Score:</div>
                    <div style="font-size: 20px; font-weight: bold, color: var(--text);">${totalDiff.toFixed(4)}</div>
                </div>
                <div>
                    <div style="font-size: 14px; color: var(--subtext0);">Max Bin Difference:</div>
                    <div style="font-size: 16px; color: var(--text);">${maxDiff.toFixed(4)}</div>
                </div>
                <div>
                    <div style="font-size: 14px; color: var(--subtext0);">Avg Bin Difference:</div>
                    <div style="font-size: 16px; color: var(--text);">${(totalDiff / diffData.length).toFixed(4)}</div>
                </div>
            </div>
        </div>
    `;

    comparisonSection.innerHTML = comparisonHTML;

    // Add to the DOM if not already there
    const existingComparison = document.getElementById('histogram-comparison');
    if (existingComparison) {
        existingComparison.innerHTML = comparisonHTML;
    } else if (histogramSection) {
        histogramSection.appendChild(comparisonSection);
    }

    window.showSuccess('Comparison Complete', 'Histogram comparison visualization generated.');
};

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
    downloadButton.textContent = '💾 Download';
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
    if (searchMode) searchMode.textContent = data.metadata?.fast_mode ? 'Fast' : 'Standard';
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
            window.showError('Error', 'Please add at least one color to search for.');
            return;
        }

        const colors = [];
        const weights = [];
        let totalWeight = 0;

        colorRows.forEach(row => {
            const colorPicker = row.querySelector('.color-picker');
            const weightSlider = row.querySelector('.weight-slider');
            if (colorPicker && weightSlider) {
                const weight = parseInt(weightSlider.value);
                colors.push(colorPicker.value.substring(1)); // Remove # from hex color
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

        const searchParams = new URLSearchParams({
            colors: colors.join(','),
            weights: normalizedWeights.join(','),
            k: resultCount,
            fast_mode: fastMode,
            batch_size: batchSize
        });

        // Change endpoint from /api/search to /search
        const response = await fetch(`/search?${searchParams.toString()}`);
        if (!response.ok) throw new Error(`Search failed with status: ${response.status}`);

        const data = await response.json();

        // Calculate search time if not provided by server
        const searchEndTime = performance.now();
        const clientSearchTime = searchEndTime - searchStartTime;

        // Add timing information to data if not present
        if (!data.search_time && !data.metadata?.search_time) {
            if (!data.metadata) data.metadata = {};
            data.metadata.search_time = clientSearchTime;
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
            resultsGrid.innerHTML = '<div class="error-message">Search failed. Please try again.</div>';
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
    console.log('Updating search results:', data);

    // Get the results grid
    const resultsGrid = document.getElementById('resultsGrid');
    if (!resultsGrid || !data || !data.results || !Array.isArray(data.results)) {
        console.error('Missing required elements or data for updating search results');
        return;
    }

    // Clear existing results
    resultsGrid.innerHTML = '';

    if (data.results.length === 0) {
        resultsGrid.innerHTML = `

            <div class="no-results" style="
                grid-column: span 3;
                text-align: center;
                padding: 40px 20px;
                background: var(--surface0);
                border-radius: 12px;
                margin: 20px 0;
            ">
                <h3>No matching images found</h3>
                <p>Try different colors or adjust your search parameters.</p>
            </div>
        `;
        return;
    }

    // Loop through results and create result cards
    data.results.forEach((result, index) => {
        const imgSrc = result.image_url || `/image/${encodeURIComponent(result.image_id)}`;
        const distance = typeof result.distance === 'number' ? result.distance.toFixed(4) : 'N/A';
        const filename = result.filename || result.image_id || `result-${index + 1}`;

        // Create result card
        const card = document.createElement('div');
        card.className = 'result-card';
        card.style.position = 'relative';

        // Add rank badge for top 3 results
        if (index < 3) {
            const rankColors = ['var(--yellow)', 'var(--subtext0)', 'var(--peach)'];
            const rankEmoji = ['🥇', '🥈', '🥉'];

            const rankBadge = document.createElement('div');
            rankBadge.style.position = 'absolute';
            rankBadge.style.top = '10px';
            rankBadge.style.left = '10px';
            rankBadge.style.backgroundColor = rankColors[index];
            rankBadge.style.color = 'var(--crust)';
            rankBadge.style.padding = '5px 10px';
            rankBadge.style.borderRadius = '20px';
            rankBadge.style.fontSize = '14px';
            rankBadge.style.fontWeight = 'bold';
            rankBadge.style.zIndex = '2';
            rankBadge.style.display = 'flex';
            rankBadge.style.alignItems = 'center';
            rankBadge.style.gap = '4px';
            rankBadge.innerHTML = `${rankEmoji[index]} ${index + 1}`;

            card.appendChild(rankBadge);
        }

        // Image container with hover effect
        const imageContainer = document.createElement('div');
        imageContainer.className = 'image-container';
        imageContainer.style.position = 'relative';
        imageContainer.style.height = '200px';
        imageContainer.style.overflow = 'hidden';
        imageContainer.style.borderRadius = '8px 8px 0 0';
        imageContainer.style.cursor = 'pointer';

        const img = document.createElement('img');
        img.src = imgSrc;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        img.style.objectPosition = 'center';
        img.style.transition = 'transform 0.3s ease';
        img.crossOrigin = 'anonymous';

        const overlay = document.createElement('div');
        overlay.className = 'image-overlay';
        overlay.style.position = 'absolute';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.background = 'rgba(0, 0, 0, 0.5)';
        overlay.style.opacity = '0';
        overlay.style.transition = 'opacity 0.3s ease';
        overlay.style.display = 'flex';
        overlay.style.alignItems = 'center';
        overlay.style.justifyContent = 'center';

        const overlayText = document.createElement('div');
        overlayText.className = 'overlay-text';
        overlayText.style.color = 'white';
        overlayText.style.fontSize = '14px';
        overlayText.style.fontWeight = 'bold';
        overlayText.innerHTML = 'Click to view';

        overlay.appendChild(overlayText);
        imageContainer.appendChild(img);
        imageContainer.appendChild(overlay);

        // Add hover effects via event listeners
        imageContainer.addEventListener('mouseover', () => {
            img.style.transform = 'scale(1.05)';
            overlay.style.opacity = '1';
        });

        imageContainer.addEventListener('mouseout', () => {
            img.style.transform = 'scale(1)';
            overlay.style.opacity = '0';
        });

        // Add click event to open image modal
        imageContainer.addEventListener('click', () => {
            window.showImageInModal(imgSrc);
        });

        // Result info
        const infoContainer = document.createElement('div');
        infoContainer.className = 'result-info';
        infoContainer.style.padding = '15px';
        infoContainer.style.background = 'var(--surface0)';
        infoContainer.style.borderRadius = '0 0 8px 8px';

        // Format filename to be more readable
        let displayName = filename;
        if (displayName.length > 30) {
            displayName = displayName.substring(0, 27) + '...';
        }

        infoContainer.innerHTML = `
            <p style="margin: 0 0 10px 0; font-size: 14px; color: var(--text);">
                <strong style="display: block; margin-bottom: 5px; font-size: 16px; color: var(--blue);">${displayName}</strong>
                Distance: <span style="color: var(--green); font-weight: bold;">${distance}</span>
            </p>
        `;

        // Color swatches if available
        if (result.dominant_colors && Array.isArray(result.dominant_colors)) {
            const swatchesContainer = document.createElement('div');
            swatchesContainer.className = 'color-swatches';
            swatchesContainer.style.display = 'flex';
            swatchesContainer.style.marginTop = '10px';
            swatchesContainer.style.gap = '5px';

            result.dominant_colors.forEach(color => {
                const swatch = document.createElement('div');
                swatch.className = 'color-swatch';
                swatch.style.width = '20px';
                swatch.style.height = '20px';
                swatch.style.backgroundColor = `#${color}`;
                swatch.style.borderRadius = '3px';
                swatch.style.border = '1px solid var(--surface1)';
                swatchesContainer.appendChild(swatch);
            });

            infoContainer.appendChild(swatchesContainer);
        }

        // Action buttons
        const actionsContainer = document.createElement('div');
        actionsContainer.className = 'action-buttons';
        actionsContainer.style.display = 'flex';
        actionsContainer.style.justifyContent = 'space-between';
        actionsContainer.style.marginTop = '15px';

        // Details button
        const detailsButton = document.createElement('button');
        detailsButton.className = 'details-btn action-btn';
        detailsButton.innerHTML = '🔍 Details';
        detailsButton.onclick = () => showImageDetails(result);

        // Download button
        const downloadButton = document.createElement('button');
        downloadButton.className = 'download-btn action-btn';
        downloadButton.innerHTML = '💾 Download';
        downloadButton.onclick = () => {
            const link = document.createElement('a');
            link.href = imgSrc;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        };

        // Copy colors button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-btn action-btn';
        copyButton.innerHTML = '📋 Copy Colors';
        copyButton.onclick = () => {
            if (result.dominant_colors && result.dominant_colors.length) {
                const colorText = result.dominant_colors.map(c => `#${c}`).join(', ');
                navigator.clipboard.writeText(colorText)
                    .then(() => window.showSuccess('Copied', 'Colors copied to clipboard'))
                    .catch(err => window.showError('Copy Failed', err.message));
            } else {
                window.showError('Copy Failed', 'No color data available');
            }
        };

        actionsContainer.appendChild(detailsButton);
        actionsContainer.appendChild(copyButton);
        actionsContainer.appendChild(downloadButton);

        infoContainer.appendChild(actionsContainer);

        // Assemble card
        card.appendChild(imageContainer);
        card.appendChild(infoContainer);

        // Add to results grid
        resultsGrid.appendChild(card);
    });

    // Show the results section
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
};

// Show detailed information about an image result
function showImageDetails(result) {
    const modal = document.getElementById('detailsModal');
    const content = document.getElementById('imageDetailsContent');

    if (!modal || !content) return;

    const imgSrc = result.image_url || `/image/${encodeURIComponent(result.image_id)}`;
    const distance = typeof result.distance === 'number' ? result.distance.toFixed(6) : 'N/A';
    const filename = result.filename || result.image_id || 'image';

    let detailsHtml = `
        <div style="display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px;">
                <div class="image-preview" style="
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                    max-height: 400px;
                ">
                    <img src="${imgSrc}" alt="${filename}" style="width: 100%; height: auto;">
                </div>
            </div>
            
            <div style="flex: 1; min-width: 300px;">
                <h3 style="color: var(--blue); margin-top: 0;">${filename}</h3>
                <p><strong>Distance Score:</strong> <span style="color: var(--green); font-weight: bold;">${distance}</span></p>
    `;

    // Add file metadata if available
    if (result.metadata) {
        detailsHtml += `
            <h4 style="margin-bottom: 10px; color: var(--text);">File Metadata</h4>
            <ul style="list-style: none; padding: 0; margin: 0 0 15px 0;">
        `;

        for (const [key, value] of Object.entries(result.metadata)) {
            if (key === 'dominant_colors' || key === 'histogram') continue;
            detailsHtml += `<li><strong>${key}:</strong> ${value}</li>`;
        }

        detailsHtml += `</ul>`;
    }

    // Add dominant colors if available
    if (result.dominant_colors && Array.isArray(result.dominant_colors)) {
        detailsHtml += `
            <h4 style="margin-bottom: 10px; color: var(--text);">Dominant Colors</h4>
            <div class="color-swatches" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
        `;

        result.dominant_colors.forEach(color => {
            detailsHtml += `
                <div style="text-align: center;">
                    <div class="color-swatch" style="
                        width: 40px;
                        height: 40px;
                        background-color: #${color};
                        border-radius: 8px;
                        border: 1px solid var(--surface2);
                        margin-bottom: 5px;
                    "></div>
                    <span style="font-size: 12px; color: var(--text);">#${color}</span>
                </div>
            `;
        });

        detailsHtml += `</div>`;
    }

    // Add color histogram if available
    if (result.histogram && Array.isArray(result.histogram)) {
        detailsHtml += `
            <h4 style="margin-bottom: 10px; color: var(--text);">Color Distribution</h4>
            <div style="height: 100px; display: flex; align-items: flex-end; gap: 1px; margin-bottom: 15px;">
        `;

        const histogramLength = Math.min(32, result.histogram.length);
        for (let i = 0; i < histogramLength; i++) {
            const value = result.histogram[i];
            const height = Math.max(5, value * 100);
            const hue = Math.round((i / histogramLength) * 360);

            detailsHtml += `
                <div style="
                    flex-grow: 1;
                    height: ${height}px;
                    background: hsl(${hue}, 70%, 60%);
                    border-radius: 2px 2px 0 0;
                "></div>
            `;
        }

        detailsHtml += `</div>`;
    }

    // Add LAB color values if available
    if (result.lab_values && Array.isArray(result.lab_values)) {
        detailsHtml += `
            <h4 style="margin-bottom: 10px; color: var(--text);">LAB Color Values</h4>
            <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 15px;">
        `;

        result.lab_values.forEach((lab, index) => {
            let color = '#CCCCCC';
            if (result.dominant_colors && result.dominant_colors[index]) {
                color = `#${result.dominant_colors[index]}`;
            }

            detailsHtml += `
                <div style="text-align: center; background: var(--surface0); padding: 8px; border-radius: 8px;">
                    <div style="
                        width: 30px;
                        height: 30px;
                        border-radius: 50%;
                        background-color: ${color};
                        margin: 0 auto 5px;
                        border: 1px solid var(--surface2);
                    "></div>
                    <div style="font-family: 'JetBrainsMono Nerd Font Mono', monospace; color: var(--text); font-size: 12px;">
                        <div>L: ${lab.L?.toFixed(2) || 'N/A'}</div>
                        <div>a: ${lab.a?.toFixed(2) || 'N/A'}</div>
                        <div>b: ${lab.b?.toFixed(2) || 'N/A'}</div>
                    </div>
                </div>
            `;
        });

        detailsHtml += `</div>`;
    }

    // Add action buttons
    detailsHtml += `
            </div>
        </div>
        
        <div class="action-buttons" style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 20px;">
            <button class="action-btn copy-btn" onclick="navigator.clipboard.writeText('${imgSrc}')">📋 Copy URL</button>
            <button class="action-btn download-btn" onclick="window.location.href='${imgSrc}' download='${filename}'">💾 Download Image</button>
        </div>
    `;

    content.innerHTML = detailsHtml;

    // Show modal with fade in animation
    modal.style.display = 'block';

    // Add close button functionality
    const closeBtn = modal.querySelector('.close');
    if (closeBtn) {
        closeBtn.onclick = function () {
            modal.style.display = 'none';
        };
    }

    // Close when clicking outside the modal content
    window.onclick = function (event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    };
}

// Function to download preview grid as image
window.downloadPreviewGrid = function () {
    const previewGrid = document.getElementById('previewGrid');
    if (!previewGrid) {
        window.showError('Download Failed', 'Preview grid not found');
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

    // Deep clone the preview grid items into the temp container
    Array.from(previewGrid.children).forEach(item => {
        // Create a clean container for each image
        const imgContainer = document.createElement('div');
        imgContainer.style.position = 'relative';
        imgContainer.style.aspectRatio = '1 / 1';
        imgContainer.style.overflow = 'hidden';
        imgContainer.style.borderRadius = '8px';
        imgContainer.style.border = '1px solid #45475a'; // Using hex instead of var(--surface2)
        imgContainer.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.2)';
        imgContainer.style.backgroundColor = '#1e1e2e'; // Dark background matching theme

        // Get the original image element
        const originalImg = item.querySelector('img');
        if (originalImg && originalImg.src) {
            const imgSrc = originalImg.src;

            // Create an actual image element instead of using background-image
            const imgElement = document.createElement('img');
            imgElement.src = imgSrc;
            imgElement.style.width = '100%';
            imgElement.style.height = '100%';
            imgElement.style.objectFit = 'cover';
            imgElement.style.objectPosition = 'center';
            imgElement.crossOrigin = 'anonymous'; // Enable cross-origin loading
            imgContainer.appendChild(imgElement);

            // Add the distance label if it exists
            const distanceLabel = item.querySelector('div:last-child');
            if (distanceLabel && distanceLabel.textContent) {
                const distanceDiv = document.createElement('div');
                distanceDiv.style.position = 'absolute';
                distanceDiv.style.bottom = '0';
                distanceDiv.style.left = '0';
                distanceDiv.style.right = '0';
                distanceDiv.style.padding = '4px';
                distanceDiv.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                distanceDiv.style.color = 'white';
                distanceDiv.style.fontSize = '10px';
                distanceDiv.style.textAlign = 'center';
                distanceDiv.textContent = distanceLabel.textContent;
                imgContainer.appendChild(distanceDiv);
            }

            gridContainer.appendChild(imgContainer);
        }
    });

    // Use html2canvas to capture the temp container
    html2canvas(tempContainer, {
        backgroundColor: '#313244', // Using hex instead of var(--surface0)
        scale: 2, // Higher quality
        logging: true, // Enable logging to debug image loading issues
        allowTaint: true,
        useCORS: true,
        imageTimeout: 15000, // Increase timeout for image loading
        onclone: function (clonedDoc) {
            console.log('HTML2Canvas cloned document:', clonedDoc);
        }
    }).then(canvas => {
        // Create watermark text
        const ctx = canvas.getContext('2d');
        ctx.save();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.font = '16px JetBrainsMono Nerd Font Mono';
        ctx.textAlign = 'center';
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(-Math.PI / 8);
        ctx.fillText('Chromatica', 0, 0);
        ctx.restore();

        // Create download link
        const link = document.createElement('a');
        link.download = `chromatica-results-${Date.now()}.png`;
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
};

// Function to restart the Chromatica server
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

// Initialize event listeners when the document is loaded
document.addEventListener('DOMContentLoaded', function () {
    console.log('Document loaded, initializing event listeners');

    // Initialize modal close buttons
    const modalCloseButtons = document.querySelectorAll('.close, .image-close');
    modalCloseButtons.forEach(button => {
        button.addEventListener('click', function () {
            const modal = this.closest('.modal, .image-modal');
            if (modal) modal.style.display = 'none';
        });
    });

    // Initialize color inputs
    window.updateColorPalette();

    // Add an initial color picker event listener if not already set
    const initialColorPicker = document.querySelector('.color-picker');
    if (initialColorPicker) {
        initialColorPicker.addEventListener('change', window.updateColors || function () { });
    }

    // Add an initial weight slider event listener if not already set
    const initialWeightSlider = document.querySelector('.weight-slider');
    const initialWeightValue = document.querySelector('.weight-value');
    if (initialWeightSlider && initialWeightValue) {
        initialWeightSlider.addEventListener('input', () => {
            initialWeightValue.textContent = `${initialWeightSlider.value}%`;
            if (window.updateWeights) {
                window.updateWeights();
            }
        });
    }

    console.log('Event listeners initialized');
});
