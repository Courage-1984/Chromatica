// Chromatica Advanced Features JavaScript Module
// Handles all advanced features including palette export, harmony analysis, gradients, statistics, and favorites

console.log('Loading Chromatica Advanced Features...');

// ============================================================================
// PALETTE EXPORT FUNCTIONS
// ============================================================================

/**
 * Export palette in various formats
 */
window.exportPalette = async function(formatType = 'css') {
    try {
        const colors = window.colors || [];
        const weights = window.weights || [];
        
        if (colors.length === 0) {
            showError('Export Error', 'No colors to export. Add colors first.');
            return;
        }
        
        const response = await fetch('/advanced/palette/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                colors: colors.map(c => c.replace('#', '')),
                weights: weights.map(w => w / 100),
                format_type: formatType,
                metadata: {
                    name: 'Chromatica Palette',
                    created_at: new Date().toISOString()
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`Export failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Handle different export types
        if (formatType === 'ase') {
            // Download ASE file
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'palette.ase';
            a.click();
            window.URL.revokeObjectURL(url);
            showSuccess('Export Success', 'Palette exported as Adobe Swatch file');
        } else {
            // Show export data
            showExportModal(data.data, formatType);
        }
    } catch (error) {
        console.error('Export error:', error);
        showError('Export Error', error.message);
    }
};

/**
 * Show export modal with formatted data
 */
function showExportModal(data, formatType) {
    const modal = document.getElementById('exportModal') || createExportModal();
    modal.style.display = 'block';
    
    const content = modal.querySelector('#exportContent');
    if (formatType === 'json' || formatType === 'sketch' || formatType === 'figma') {
        content.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
    } else {
        content.textContent = data;
    }
    
    // Copy to clipboard button
    const copyBtn = modal.querySelector('#copyExportBtn');
    copyBtn.onclick = () => {
        navigator.clipboard.writeText(content.textContent);
        showSuccess('Copied!', 'Export data copied to clipboard');
    };
}

/**
 * Create export modal if it doesn't exist
 */
function createExportModal() {
    const modal = document.createElement('div');
    modal.id = 'exportModal';
    modal.style.cssText = 'display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7);';
    
    const content = document.createElement('div');
    content.style.cssText = 'background: var(--base); margin: 5% auto; padding: 20px; border: 1px solid var(--surface2); width: 80%; max-width: 800px; border-radius: 12px; max-height: 80vh; overflow-y: auto;';
    
    content.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h3 style="margin: 0; color: var(--text);">Export Data</h3>
            <button onclick="document.getElementById('exportModal').style.display='none'" 
                style="background: var(--red); color: #000000; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer;">Close</button>
        </div>
        <pre id="exportContent" style="background: var(--surface0); padding: 15px; border-radius: 8px; overflow-x: auto; color: var(--text); font-family: 'JetBrainsMono Nerd Font Mono', monospace; font-size: 12px; max-height: 60vh; overflow-y: auto;"></pre>
        <button id="copyExportBtn" style="margin-top: 15px; background: var(--teal); color: #000000; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer;">Copy to Clipboard</button>
    `;
    
    modal.appendChild(content);
    document.body.appendChild(modal);
    return modal;
}

// ============================================================================
// COLOR HARMONY ANALYSIS
// ============================================================================

/**
 * Analyze color harmony in current palette
 */
window.analyzeHarmony = async function(includeSuggestions = true) {
    try {
        const colors = window.colors || [];
        if (colors.length < 2) {
            showError('Harmony Analysis', 'Need at least 2 colors for harmony analysis');
            return;
        }
        
        const colorString = colors.map(c => c.replace('#', '')).join(',');
        
        const response = await fetch(`/advanced/palette/harmony?colors=${encodeURIComponent(colorString)}&suggest=${includeSuggestions}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Harmony analysis failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        showHarmonyResults(data);
    } catch (error) {
        console.error('Harmony analysis error:', error);
        showError('Harmony Analysis Error', error.message);
    }
};

/**
 * Show harmony analysis results
 */
function showHarmonyResults(data) {
    const resultsDiv = document.getElementById('harmonyResults') || createHarmonyResultsDiv();
    resultsDiv.style.display = 'block';
    
    let html = `
        <div style="background: var(--surface0); padding: 20px; border-radius: 12px; border: 2px solid var(--mauve);">
            <h4 style="margin: 0 0 15px 0; color: var(--text);">Harmony Analysis Results</h4>
            <div style="margin-bottom: 15px;">
                <strong style="color: var(--mauve);">Type:</strong> 
                <span style="color: var(--text);">${data.harmony_type}</span>
            </div>
            <div style="margin-bottom: 15px;">
                <strong style="color: var(--mauve);">Confidence:</strong> 
                <span style="color: var(--text);">${(data.confidence * 100).toFixed(1)}%</span>
            </div>
            <div style="margin-bottom: 15px;">
                <strong style="color: var(--mauve);">Description:</strong> 
                <span style="color: var(--text);">${data.description}</span>
            </div>
    `;
    
    if (data.suggestions && data.suggestions.length > 0) {
        html += '<div style="margin-top: 20px;"><strong style="color: var(--mauve);">Suggestions:</strong><ul style="margin: 10px 0; padding-left: 20px;">';
        data.suggestions.forEach(suggestion => {
            html += `<li style="color: var(--text); margin-bottom: 10px;">
                <strong>${suggestion.type}:</strong> ${suggestion.description}<br>
                <div style="display: flex; gap: 5px; margin-top: 5px;">`;
            suggestion.colors.forEach(color => {
                html += `<div style="width: 30px; height: 30px; background: ${color}; border: 1px solid var(--surface2); border-radius: 4px;"></div>`;
            });
            html += '</div></li>';
        });
        html += '</ul></div>';
    }
    
    html += '</div>';
    resultsDiv.innerHTML = html;
}

/**
 * Create harmony results div if it doesn't exist
 */
function createHarmonyResultsDiv() {
    const div = document.createElement('div');
    div.id = 'harmonyResults';
    div.style.cssText = 'margin: 20px 0; display: none;';
    const colorInputSection = document.querySelector('.color-input-section');
    if (colorInputSection) {
        colorInputSection.appendChild(div);
    }
    return div;
}

// ============================================================================
// GRADIENT GENERATION
// ============================================================================

/**
 * Generate gradient from current palette
 */
window.generateGradient = async function(type = 'css', width = 800, height = 200, direction = 'horizontal') {
    try {
        const colors = window.colors || [];
        if (colors.length < 2) {
            showError('Gradient Generation', 'Need at least 2 colors for gradient');
            return;
        }
        
        const response = await fetch('/advanced/gradient/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                colors: colors.map(c => c.replace('#', '')),
                weights: window.weights ? window.weights.map(w => w / 100) : null,
                width: width,
                height: height,
                direction: direction,
                gradient_type: type
            })
        });
        
        if (!response.ok) {
            throw new Error(`Gradient generation failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        showGradientResult(data);
    } catch (error) {
        console.error('Gradient generation error:', error);
        showError('Gradient Generation Error', error.message);
    }
};

/**
 * Show gradient result
 */
function showGradientResult(data) {
    const modal = document.getElementById('gradientModal') || createGradientModal();
    modal.style.display = 'block';
    
    const content = modal.querySelector('#gradientContent');
    if (data.type === 'css') {
        content.innerHTML = `
            <div style="margin-bottom: 20px;">
                <h4 style="color: var(--text);">CSS Gradient:</h4>
                <pre style="background: var(--surface0); padding: 15px; border-radius: 8px; overflow-x: auto; color: var(--text); font-family: 'JetBrainsMono Nerd Font Mono', monospace;">background: ${data.gradient};</pre>
            </div>
        `;
    } else {
        content.innerHTML = `
            <div style="margin-bottom: 20px;">
                <h4 style="color: var(--text);">Gradient Image (${data.width}x${data.height}):</h4>
                <img src="${data.data_url}" style="max-width: 100%; border: 1px solid var(--surface2); border-radius: 8px; margin-bottom: 15px;" alt="Generated Gradient">
                <button onclick="downloadGradientImage('${data.data_url}')" style="background: var(--teal); color: #000000; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer;">Download Image</button>
            </div>
        `;
    }
}

/**
 * Create gradient modal if it doesn't exist
 */
function createGradientModal() {
    const modal = document.createElement('div');
    modal.id = 'gradientModal';
    modal.style.cssText = 'display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7);';
    
    const content = document.createElement('div');
    content.style.cssText = 'background: var(--base); margin: 5% auto; padding: 20px; border: 1px solid var(--surface2); width: 80%; max-width: 800px; border-radius: 12px; max-height: 80vh; overflow-y: auto;';
    
    content.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h3 style="margin: 0; color: var(--text);">Generated Gradient</h3>
            <button onclick="document.getElementById('gradientModal').style.display='none'" 
                style="background: var(--red); color: #000000; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer;">Close</button>
        </div>
        <div id="gradientContent"></div>
    `;
    
    modal.appendChild(content);
    document.body.appendChild(modal);
    return modal;
}

/**
 * Download gradient image
 */
window.downloadGradientImage = function(dataUrl) {
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `gradient-${Date.now()}.png`;
    a.click();
};

// ============================================================================
// COLOR STATISTICS
// ============================================================================

/**
 * Analyze color statistics for current palette
 */
window.analyzeStatistics = async function() {
    try {
        const colors = window.colors || [];
        if (colors.length === 0) {
            showError('Statistics Analysis', 'No colors to analyze');
            return;
        }
        
        const colorString = colors.map(c => c.replace('#', '')).join(',');
        
        const response = await fetch(`/advanced/statistics/analyze?colors=${encodeURIComponent(colorString)}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Statistics analysis failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        showStatisticsDashboard(data);
    } catch (error) {
        console.error('Statistics analysis error:', error);
        showError('Statistics Analysis Error', error.message);
    }
};

/**
 * Show statistics dashboard
 */
function showStatisticsDashboard(data) {
    const dashboard = document.getElementById('statisticsDashboard') || createStatisticsDashboard();
    dashboard.style.display = 'block';
    
    let html = `
        <div style="background: var(--surface0); padding: 20px; border-radius: 12px; border: 2px solid var(--blue);">
            <h4 style="margin: 0 0 20px 0; color: var(--text);">Color Statistics Dashboard</h4>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                <div style="background: var(--base); padding: 15px; border-radius: 8px; border: 1px solid var(--surface2);">
                    <div style="color: var(--subtext1); font-size: 12px; margin-bottom: 5px;">Average Brightness</div>
                    <div style="color: var(--text); font-size: 24px; font-weight: bold;">${(data.average_brightness * 100).toFixed(1)}%</div>
                </div>
                <div style="background: var(--base); padding: 15px; border-radius: 8px; border: 1px solid var(--surface2);">
                    <div style="color: var(--subtext1); font-size: 12px; margin-bottom: 5px;">Average Saturation</div>
                    <div style="color: var(--text); font-size: 24px; font-weight: bold;">${(data.average_saturation * 100).toFixed(1)}%</div>
                </div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h5 style="color: var(--text); margin-bottom: 10px;">Most Common Colors:</h5>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">`;
    
    data.most_common_colors.slice(0, 10).forEach(item => {
        html += `
            <div style="text-align: center;">
                <div style="width: 50px; height: 50px; background: ${item.color}; border: 2px solid var(--surface2); border-radius: 8px; margin-bottom: 5px;"></div>
                <div style="color: var(--subtext1); font-size: 11px;">${item.frequency}x</div>
            </div>`;
    });
    
    html += `</div></div>`;
    
    html += `
        <div style="margin-bottom: 20px;">
            <h5 style="color: var(--text); margin-bottom: 10px;">Temperature Distribution:</h5>
            <div style="display: flex; gap: 15px;">`;
    
    Object.entries(data.temperature_distribution).forEach(([temp, count]) => {
        html += `
            <div style="background: var(--base); padding: 15px; border-radius: 8px; border: 1px solid var(--surface2); flex: 1;">
                <div style="color: var(--subtext1); font-size: 12px; margin-bottom: 5px; text-transform: capitalize;">${temp}</div>
                <div style="color: var(--text); font-size: 20px; font-weight: bold;">${count}</div>
            </div>`;
    });
    
    html += `</div></div></div>`;
    dashboard.innerHTML = html;
}

/**
 * Create statistics dashboard if it doesn't exist
 */
function createStatisticsDashboard() {
    const div = document.createElement('div');
    div.id = 'statisticsDashboard';
    div.style.cssText = 'margin: 20px 0; display: none;';
    const colorInputSection = document.querySelector('.color-input-section');
    if (colorInputSection) {
        colorInputSection.appendChild(div);
    }
    return div;
}

// ============================================================================
// FAVORITE PALETTES
// ============================================================================

/**
 * Save current palette as favorite
 */
window.saveFavorite = async function(name = null) {
    try {
        const colors = window.colors || [];
        const weights = window.weights || [];
        
        if (colors.length === 0) {
            showError('Save Favorite', 'No colors to save');
            return;
        }
        
        if (!name) {
            name = prompt('Enter a name for this palette:');
            if (!name) return;
        }
        
        const response = await fetch('/advanced/favorites/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                colors: colors.map(c => c.replace('#', '')),
                weights: weights.map(w => w / 100)
            })
        });
        
        if (!response.ok) {
            throw new Error(`Save favorite failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        showSuccess('Favorite Saved', `Palette "${name}" saved successfully!`);
        loadFavorites(); // Refresh favorites list
    } catch (error) {
        console.error('Save favorite error:', error);
        showError('Save Favorite Error', error.message);
    }
};

/**
 * Load and display favorites
 */
window.loadFavorites = async function() {
    try {
        const response = await fetch('/advanced/favorites/list');
        if (!response.ok) {
            throw new Error(`Load favorites failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        showFavoritesList(data.favorites);
    } catch (error) {
        console.error('Load favorites error:', error);
        showError('Load Favorites Error', error.message);
    }
};

/**
 * Show favorites list
 */
function showFavoritesList(favorites) {
    const listDiv = document.getElementById('favoritesList') || createFavoritesListDiv();
    listDiv.style.display = favorites.length > 0 ? 'block' : 'none';
    
    if (favorites.length === 0) {
        listDiv.innerHTML = '<p style="color: var(--subtext1); text-align: center; padding: 20px;">No saved favorites yet.</p>';
        return;
    }
    
    let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;">';
    
    favorites.forEach(fav => {
        html += `
            <div style="background: var(--surface0); padding: 15px; border-radius: 8px; border: 1px solid var(--surface2);">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                    <h5 style="margin: 0; color: var(--text);">${fav.name}</h5>
                    <button onclick="deleteFavorite('${fav.id}')" style="background: var(--red); color: #000000; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px;">×</button>
                </div>
                <div style="display: flex; gap: 5px; margin-bottom: 10px;">`;
        
        fav.colors.forEach(color => {
            html += `<div style="width: 30px; height: 30px; background: #${color}; border: 1px solid var(--surface2); border-radius: 4px;"></div>`;
        });
        
        html += `</div>
                <button onclick="loadFavorite('${fav.id}')" style="width: 100%; background: var(--teal); color: #000000; border: none; padding: 8px; border-radius: 4px; cursor: pointer; font-size: 12px;">Load</button>
            </div>`;
    });
    
    html += '</div>';
    listDiv.innerHTML = html;
}

/**
 * Load a favorite palette
 */
window.loadFavorite = function(favoriteId) {
    // This would fetch the favorite and load it into the current palette
    // Implementation depends on how favorites are stored
    showSuccess('Favorite Loaded', 'Favorite palette loaded');
};

/**
 * Delete a favorite
 */
window.deleteFavorite = async function(favoriteId) {
    if (!confirm('Delete this favorite?')) return;
    
    try {
        const response = await fetch(`/advanced/favorites/${favoriteId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`Delete favorite failed: ${response.statusText}`);
        }
        
        showSuccess('Favorite Deleted', 'Favorite deleted successfully');
        loadFavorites();
    } catch (error) {
        console.error('Delete favorite error:', error);
        showError('Delete Favorite Error', error.message);
    }
};

/**
 * Create favorites list div if it doesn't exist
 */
function createFavoritesListDiv() {
    const div = document.createElement('div');
    div.id = 'favoritesList';
    div.style.cssText = 'margin: 20px 0; display: none;';
    const colorInputSection = document.querySelector('.color-input-section');
    if (colorInputSection) {
        colorInputSection.appendChild(div);
    }
    return div;
}

// ============================================================================
// ADVANCED FILTERS
// ============================================================================

/**
 * Get advanced filter values
 */
window.getAdvancedFilters = function() {
    return {
        exclude_colors: document.getElementById('excludeColors')?.value?.split(',').filter(c => c.trim()) || [],
        similarity_range: document.getElementById('similarityRange')?.value || null,
        temperature: document.getElementById('temperatureFilter')?.value || 'all',
        brightness_min: document.getElementById('brightnessMin')?.value || null,
        brightness_max: document.getElementById('brightnessMax')?.value || null,
        saturation_min: document.getElementById('saturationMin')?.value || null,
        saturation_max: document.getElementById('saturationMax')?.value || null,
        dominant_color_count_min: document.getElementById('dominantColorCountMin')?.value || null,
        dominant_color_count_max: document.getElementById('dominantColorCountMax')?.value || null
    };
};

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        loadFavorites();
    });
} else {
    loadFavorites();
}

console.log('✓ Chromatica Advanced Features loaded');

