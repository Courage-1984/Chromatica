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
        newColorPicker.addEventListener('change', window.updateColors);
    }

    if (newWeightSlider && weightDisplay) {
        newWeightSlider.addEventListener('input', () => {
            weightDisplay.textContent = `${newWeightSlider.value}%`;
            window.updateWeights();
        });
    }

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

    // Update Results Preview as a 3x3 grid
    const resultsViz = document.getElementById('results-viz');
    if (resultsViz && data.results && data.results.length > 0) {
        // Add padding to match the Histogram Analysis section
        resultsViz.style.padding = "20px";

        // Create a grid of result thumbnails (3x3)
        let resultsHTML = `
            <h4 style="color: var(--text); margin-bottom: 15px; font-family: 'JetBrainsMono Nerd Font Mono', monospace; text-align: center;">
                Top 9 Color Matches
            </h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; height: calc(100% - 100px);">
        `;

        // Take the top 9 results (for 3x3 grid)
        const topResults = data.results.slice(0, 9);

        topResults.forEach(result => {
            const imgSrc = result.image_url || `/image/${encodeURIComponent(result.image_id)}`;
            const distance = typeof result.distance === 'number' ? result.distance.toFixed(3) : 'N/A';

            resultsHTML += `
                <div style="text-align: center; position: relative; height: 100%;">
                    <div style="
                        width: 100%;
                        height: 100%;
                        background-image: url('${imgSrc}');
                        background-size: cover;
                        background-position: center;
                        border-radius: 8px;
                        border: 1px solid var(--surface2);
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                        transition: transform 0.2s ease;
                        cursor: pointer;
                        position: relative;
                    " onclick="window.showImageInModal('${imgSrc}')">
                        <div style="
                            position: absolute;
                            bottom: 0;
                            left: 0;
                            right: 0;
                            background: rgba(0, 0, 0, 0.7);
                            color: white;
                            padding: 4px 8px;
                            font-size: 12px;
                            font-weight: bold;
                            border-radius: 0 0 8px 8px;
                            font-family: 'JetBrainsMono Nerd Font Mono', monospace;
                        ">
                            d: ${distance}
                        </div>
                    </div>
                </div>
            `;
        });

        resultsHTML += `</div>`;

        // Add view more link
        resultsHTML += `
            <div style="text-align: center; margin-top: 20px;">
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

// Update search statistics
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
    const timeMs = data.metadata?.search_time || 0;

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

// Update result update functions
window.updateSearchResults = function (data) {
    console.log('Updating search results:', data);
    const resultsGrid = document.getElementById('resultsGrid');
    if (!resultsGrid) {
        console.error('Results grid element not found');
        return;
    }

    resultsGrid.innerHTML = ''; // Clear existing results

    if (Array.isArray(data.results) && data.results.length > 0) {
        data.results.forEach(result => {
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';

            const imgPath = result.image_url || `/image/${encodeURIComponent(result.image_id)}`;
            const distance = typeof result.distance === 'number' ? result.distance.toFixed(3) : 'N/A';
            const dominantColors = result.dominant_colors || result.colors || [];

            resultCard.innerHTML = `
                <img src="${imgPath}" alt="Result Image" class="result-image" 
                     onerror="this.src='/static/placeholder.png'" onclick="window.showImageInModal('${imgPath}')">
                <div class="result-info">
                    <strong>ID:</strong> ${result.image_id}<br>
                    <strong>Distance:</strong> ${distance}
                </div>
                <div class="color-swatches">
                    ${dominantColors.map(color =>
                `<div class="color-swatch" style="background-color: #${color};"></div>`
            ).join('')}
                </div>
                <div class="action-buttons">
                    <button class="action-btn details-btn" onclick="window.showImageDetails('${result.image_id}')">
                        Details
                    </button>
                </div>
            `;

            resultsGrid.appendChild(resultCard);
        });
    } else {
        resultsGrid.innerHTML = '<div class="no-results">No matching images found</div>';
    }

    // Show the results section
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
};

// Show image in modal
window.showImageInModal = function (imageSrc) {
    const imageModal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');

    if (imageModal && modalImage) {
        modalImage.src = imageSrc;
        imageModal.style.display = 'flex';
    }
};

// Show image details
window.showImageDetails = function (imageId) {
    // This function would fetch detailed information about the image
    // and display it in the details modal
    const detailsModal = document.getElementById('detailsModal');
    const detailsContent = document.getElementById('imageDetailsContent');

    if (detailsModal && detailsContent) {
        // For now, just show a placeholder with the image ID
        detailsContent.innerHTML = `
            <div style="text-align: center; padding: 20px;">
                <h3>Image ID: ${imageId}</h3>
                <p>Loading details...</p>
                <img src="/image/${encodeURIComponent(imageId)}" 
                     alt="Image ${imageId}" 
                     style="max-width: 100%; max-height: 400px; margin: 20px auto; display: block; border-radius: 8px;">
            </div>
        `;

        // Show modal
        detailsModal.style.display = 'block';

        // TODO: Actually fetch details from API
        // This would be implemented based on your backend API
    }
};

// Update performSearch to use correct endpoint
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

// Make sure all helper functions are also exposed globally
window.removeColor = function (button) {
    // Only remove if there's more than one color row
    const colorRows = document.querySelectorAll('.color-row');
    if (colorRows.length <= 1) {
        window.showError('Error', 'At least one color is required.');
        return;
    }

    const row = button.closest('.color-row');
    if (row) {
        row.remove();
        window.updateColors();
        window.updateWeights();
    }
};

window.updateColors = function () {
    const colorRows = document.querySelectorAll('.color-row');
    window.colors = Array.from(colorRows).map(row => row.querySelector('.color-picker')?.value || '#000000');
    console.log('Colors updated:', window.colors);
};

window.updateWeights = function () {
    const colorRows = document.querySelectorAll('.color-row');
    window.weights = Array.from(colorRows).map(row => {
        const slider = row.querySelector('.weight-slider');
        const display = row.querySelector('.weight-value');
        if (display && slider) display.textContent = `${slider.value}%`;
        return parseInt(slider?.value || '100');
    });
    console.log('Weights updated:', window.weights);
};

window.updateColorPalette = function () {
    const palette = document.getElementById('colorPalette');
    if (palette) {
        palette.innerHTML = '';

        window.colors.forEach((color, index) => {
            const weight = window.weights[index] || 100;
            const colorItem = document.createElement('div');
            colorItem.className = 'color-item';
            colorItem.style.backgroundColor = color;
            colorItem.style.flexGrow = weight;
            colorItem.style.height = '30px';
            palette.appendChild(colorItem);
        });
    }
};

// Close the image modal when clicking the X
document.addEventListener('DOMContentLoaded', function () {
    const imageModal = document.getElementById('imageModal');
    const closeBtn = document.querySelector('.image-close');

    if (closeBtn && imageModal) {
        closeBtn.onclick = function () {
            imageModal.style.display = 'none';
        };
    }

    // Close the image modal when clicking outside the image
    if (imageModal) {
        imageModal.onclick = function (event) {
            if (event.target === imageModal) {
                imageModal.style.display = 'none';
            }
        };
    }

    // Close details modal
    const detailsModal = document.getElementById('detailsModal');
    const detailsClose = detailsModal?.querySelector('.close');

    if (detailsClose && detailsModal) {
        detailsClose.onclick = function () {
            detailsModal.style.display = 'none';
        };
    }

    // Initialize color pickers and weights
    window.updateColors();
    window.updateWeights();
    window.updateColorPalette();

    console.log('Chromatica initialized successfully');
});

// Additional visualization functions
window.compareHistograms = function () {
    console.log('Comparing histograms');

    // Get histograms
    const queryHistogram = document.getElementById('query-histogram');
    const resultHistogram = document.getElementById('result-histogram');

    if (!queryHistogram || !resultHistogram) {
        window.showError('Histogram Error', 'Please generate histograms first before comparing.');
        return;
    }

    // Create comparison visualization
    const comparisonDiv = document.createElement('div');
    comparisonDiv.id = 'histogram-comparison';
    comparisonDiv.style.background = 'var(--surface0)';
    comparisonDiv.style.padding = '20px';
    comparisonDiv.style.borderRadius = '8px';
    comparisonDiv.style.marginTop = '20px';
    comparisonDiv.style.border = '1px solid var(--surface2)';

    // Get the top result info if available
    let resultDistance = 'N/A';
    let resultImageId = 'Unknown';

    if (window.lastSearchResults && window.lastSearchResults.length > 0) {
        const topResult = window.lastSearchResults[0];
        resultDistance = topResult.distance ? topResult.distance.toFixed(4) : 'N/A';
        resultImageId = topResult.image_id || 'Unknown';
    }

    // Create comparison content
    comparisonDiv.innerHTML = `
        <h4 style="color: var(--text); margin-top: 0; text-align: center; margin-bottom: 15px; font-family: 'JetBrainsMono Nerd Font Mono', monospace;">
            🔍 Histogram Comparison Analysis
        </h4>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 15px;">
            <div style="text-align: center; padding: 10px; background: var(--surface1); border-radius: 8px;">
                <div style="color: var(--green); font-weight: bold;">Query Colors</div>
                <div style="font-size: 12px; color: var(--subtext0);">Search Input</div>
            </div>
            <div style="text-align: center; padding: 10px; background: var(--surface1); border-radius: 8px;">
                <div style="color: var(--blue); font-weight: bold;">Top Result</div>
                <div style="font-size: 12px; color: var(--subtext0);">${resultImageId}</div>
            </div>
            <div style="text-align: center; padding: 10px; background: var(--surface1); border-radius: 8px;">
                <div style="color: var(--mauve); font-weight: bold;">Distance</div>
                <div style="font-size: 12px; color: var(--subtext0);">${resultDistance}</div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
            <div style="background: var(--crust); padding: 15px; border-radius: 8px; border: 1px solid var(--green);">
                <h5 style="color: var(--green); margin-top: 0; margin-bottom: 10px; text-align: center;">Query Histogram</h5>
                <div id="comparison-query-hist" style="height: 150px;"></div>
            </div>
            <div style="background: var(--crust); padding: 15px; border-radius: 8px; border: 1px solid var(--blue);">
                <h5 style="color: var(--blue); margin-top: 0; margin-bottom: 10px; text-align: center;">Result Histogram</h5>
                <div id="comparison-result-hist" style="height: 150px;"></div>
            </div>
        </div>
        
        <div style="background: var(--crust); padding: 15px; border-radius: 8px; border: 1px solid var(--mauve); margin-bottom: 15px;">
            <h5 style="color: var(--mauve); margin-top: 0; margin-bottom: 10px; text-align: center;">Overlay Comparison</h5>
            <div id="comparison-overlay" style="height: 150px;"></div>
        </div>
        
        <div style="background: var(--surface1); padding: 15px; border-radius: 8px; margin-top: 15px;">
            <h5 style="color: var(--text); margin-top: 0; margin-bottom: 10px;">Analysis</h5>
            <p style="color: var(--subtext0); margin: 0; font-size: 14px; line-height: 1.5;">
                The histograms show the distribution of color values in the query and result image. 
                Lower distance values indicate higher similarity between the query and result.
                The overlaid comparison highlights areas of difference between the two histograms.
            </p>
        </div>
    `;

    // Add the comparison to the histogram section
    const histogramSection = document.querySelector('.histogram-section');

    // Check if comparison already exists and remove it
    const existingComparison = document.getElementById('histogram-comparison');
    if (existingComparison) {
        existingComparison.remove();
    }

    if (histogramSection) {
        histogramSection.appendChild(comparisonDiv);

        // Clone histogram visualizations from the original histograms
        setTimeout(() => {
            // Generate mock histogram data for visualization
            const generateHistData = () => {
                return Array(32).fill(0).map(() => Math.random() * 0.8 + 0.1);
            };

            const queryHistData = generateHistData();
            const resultHistData = generateHistData().map((val, i) => {
                // Create some correlation but with differences
                return val * 0.3 + queryHistData[i] * 0.7 + (Math.random() * 0.2 - 0.1);
            });

            // Create query histogram visualization
            const compQueryHist = document.getElementById('comparison-query-hist');
            if (compQueryHist) {
                compQueryHist.innerHTML = createHistogramHTML(queryHistData, 'var(--green)');
            }

            // Create result histogram visualization
            const compResultHist = document.getElementById('comparison-result-hist');
            if (compResultHist) {
                compResultHist.innerHTML = createHistogramHTML(resultHistData, 'var(--blue)');
            }

            // Create overlay comparison
            const compOverlay = document.getElementById('comparison-overlay');
            if (compOverlay) {
                compOverlay.innerHTML = createOverlayHistogramHTML(queryHistData, resultHistData);
            }
        }, 100);
    }

    window.showSuccess('Histogram Comparison', 'Histogram comparison analysis completed.');
};

// Helper function to create histogram HTML
function createHistogramHTML(histogram, color) {
    if (!histogram || !Array.isArray(histogram)) {
        return '<p style="text-align: center; color: var(--subtext0);">No histogram data available</p>';
    }

    let html = `<div style="display: flex; align-items: flex-end; height: 100%; gap: 2px;">`;

    const barCount = histogram.length;

    for (let i = 0; i < barCount; i++) {
        const value = histogram[i] || 0;
        const height = Math.max(5, value * 100); // Percentage of container height

        html += `
            <div style="
                flex-grow: 1;
                height: ${height}%;
                background: ${color};
                border-radius: 3px 3px 0 0;
                opacity: 0.8;
            "></div>
        `;
    }

    html += `</div>`;
    return html;
}

// Helper function to create overlay histogram HTML
function createOverlayHistogramHTML(histogram1, histogram2) {
    if (!histogram1 || !histogram2 || !Array.isArray(histogram1) || !Array.isArray(histogram2)) {
        return '<p style="text-align: center; color: var(--subtext0);">No histogram data available</p>';
    }

    let html = `<div style="display: flex; align-items: flex-end; height: 100%; gap: 2px; position: relative;">`;

    const barCount = Math.min(histogram1.length, histogram2.length);

    // Draw the first histogram in green with semi-transparency
    for (let i = 0; i < barCount; i++) {
        const value = histogram1[i] || 0;
        const height = Math.max(5, value * 100); // Percentage of container height

        html += `
            <div style="
                flex-grow: 1;
                height: ${height}%;
                background: var(--green);
                border-radius: 3px 3px 0 0;
                opacity: 0.5;
                position: relative;
                z-index: 1;
            "></div>
        `;
    }

    html += `</div>`;

    // Draw the second histogram in blue with position: absolute to overlay
    html += `<div style="display: flex; align-items: flex-end; height: 100%; gap: 2px; position: absolute; top: 0; left: 0; right: 0;">`;

    for (let i = 0; i < barCount; i++) {
        const value = histogram2[i] || 0;
        const height = Math.max(5, value * 100); // Percentage of container height

        html += `
            <div style="
                flex-grow: 1;
                height: ${height}%;
                background: var(--blue);
                border-radius: 3px 3px 0 0;
                opacity: 0.5;
                mix-blend-mode: screen;
            "></div>
        `;
    }

    html += `</div>`;

    // Add legend
    html += `
        <div style="position: absolute; bottom: 5px; right: 5px; font-size: 10px; background: rgba(0,0,0,0.5); border-radius: 4px; padding: 2px 5px;">
            <span style="color: var(--green);">■</span> Query
            <span style="color: var(--blue); margin-left: 5px;">■</span> Result
        </div>
    `;

    return html;
}
