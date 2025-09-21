// Chromatica Color Search Engine - Shared JavaScript Functions

// Global variables
let colors = ['#FF0000'];
let weights = [100];

// Utility Functions
function showSuccess(title, message) {
    const successDiv = document.getElementById('success');
    if (successDiv) {
        successDiv.innerHTML = 
            <h3></h3>
            <p></p>
        ;
        successDiv.style.display = 'block';
        setTimeout(() => {
            successDiv.style.display = 'none';
        }, 10000);
    }
}

function showError(title, message) {
    const errorDiv = document.getElementById('error');
    if (errorDiv) {
        errorDiv.innerHTML = 
            <h3></h3>
            <p></p>
        ;
        errorDiv.style.display = 'block';
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 10000);
    }
}

// Modal Functions
function showModal(title, content) {
    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modalTitle');
    const modalContent = document.getElementById('modalContent');
    
    if (modal && modalTitle && modalContent) {
        modalTitle.textContent = title;
        modalContent.innerHTML = content;
        modal.style.display = 'block';
    }
}

function hideModal() {
    const modal = document.getElementById('modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Clipboard Functions
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showSuccess('Copied!', 'Text copied to clipboard');
    }).catch(err => {
        showError('Copy Failed', 'Could not copy to clipboard');
    });
}

// System Status Functions
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Status check failed:', error);
        return { status: 'error', message: 'Could not connect to server' };
    }
}

// Server Restart Function
async function restartServer() {
    if (confirm('Are you sure you want to restart the server? This will temporarily interrupt service.')) {
        try {
            const response = await fetch('/api/restart', { method: 'POST' });
            if (response.ok) {
                showSuccess('Server Restart', 'Server restart initiated. Please wait...');
                setTimeout(() => {
                    window.location.reload();
                }, 5000);
            } else {
                showError('Restart Failed', 'Could not restart server');
            }
        } catch (error) {
            showError('Restart Failed', 'Server restart failed: ' + error.message);
        }
    }
}

// Section Navigation Functions
function showSection(sectionName) {
    // Hide all sections
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.classList.remove('active');
    });
    
    // Show selected section
    const targetSection = document.getElementById(sectionName + '-section');
    if (targetSection) {
        targetSection.classList.add('active');
    }
    
    // Update navigation buttons
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    const activeBtn = document.querySelector([onclick="showSection('')"]);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    // Load section content
    loadSectionContent(sectionName);
}

// Load section content dynamically
async function loadSectionContent(sectionName) {
    const contentDiv = document.getElementById(sectionName + '-content');
    if (contentDiv) {
        try {
            const response = await fetch(/static/sections/.html);
            const content = await response.text();
            contentDiv.innerHTML = content;
            
            // Initialize section-specific functionality
            initializeSection(sectionName);
        } catch (error) {
            console.error(Failed to load  section:, error);
            contentDiv.innerHTML = <p>Error loading  section</p>;
        }
    }
}

// Initialize section-specific functionality
function initializeSection(sectionName) {
    switch(sectionName) {
        case 'search':
            initializeSearchSection();
            break;
        case 'visualization':
            initializeVisualizationSection();
            break;
        case 'tools':
            initializeToolsSection();
            break;
        case 'testing':
            initializeTestingSection();
            break;
        case '3d':
            initialize3DSection();
            break;
    }
}

// Initialize search section
function initializeSearchSection() {
    // Color picker functionality
    const colorPicker = document.getElementById('colorPicker');
    const colorDisplay = document.getElementById('colorDisplay');
    const weightSlider = document.getElementById('weightSlider');
    const weightValue = document.getElementById('weightValue');
    
    if (colorPicker && colorDisplay) {
        colorPicker.addEventListener('change', function() {
            colorDisplay.style.backgroundColor = this.value;
        });
    }
    
    if (weightSlider && weightValue) {
        weightSlider.addEventListener('input', function() {
            weightValue.textContent = this.value;
        });
    }
    
    // Add color button
    const addColorBtn = document.getElementById('addColorBtn');
    if (addColorBtn) {
        addColorBtn.addEventListener('click', addColor);
    }
    
    // Clear colors button
    const clearColorsBtn = document.getElementById('clearColorsBtn');
    if (clearColorsBtn) {
        clearColorsBtn.addEventListener('click', clearColors);
    }
    
    // Search button
    const searchBtn = document.getElementById('searchBtn');
    if (searchBtn) {
        searchBtn.addEventListener('click', performSearch);
    }
}

// Initialize other sections (placeholder functions)
function initializeVisualizationSection() {
    console.log('Initializing visualization section');
}

function initializeToolsSection() {
    console.log('Initializing tools section');
}

function initializeTestingSection() {
    console.log('Initializing testing section');
}

function initialize3DSection() {
    console.log('Initializing 3D section');
}

// Color management functions
function addColor() {
    const colorPicker = document.getElementById('colorPicker');
    const weightSlider = document.getElementById('weightSlider');
    
    if (colorPicker && weightSlider) {
        const color = colorPicker.value;
        const weight = parseInt(weightSlider.value);
        
        colors.push(color);
        weights.push(weight);
        
        updateColorPalette();
    }
}

function removeColor(index) {
    colors.splice(index, 1);
    weights.splice(index, 1);
    updateColorPalette();
}

function clearColors() {
    colors = ['#FF0000'];
    weights = [100];
    updateColorPalette();
}

function updateColorPalette() {
    const palette = document.getElementById('colorPalette');
    if (palette) {
        palette.innerHTML = '';
        
        colors.forEach((color, index) => {
            const colorItem = document.createElement('div');
            colorItem.className = 'color-item';
            colorItem.style.backgroundColor = color;
            colorItem.innerHTML = 
                <span class="color-weight">%</span>
                <button class="remove-color" onclick="removeColor()">×</button>
            ;
            palette.appendChild(colorItem);
        });
    }
}

// Tool panel toggle function
function toggleToolPanel(panelId) {
    const panel = document.getElementById(panelId);
    const toggle = document.querySelector([onclick="toggleToolPanel('')"]);
    
    if (panel && toggle) {
        if (panel.style.display === 'none' || panel.style.display === '') {
            panel.style.display = 'block';
            toggle.textContent = '-';
        } else {
            panel.style.display = 'none';
            toggle.textContent = '+';
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Chromatica Color Search Engine loaded');
    
    // Load the default section (search)
    showSection('search');
    
    // Initialize color palette
    updateColorPalette();
});
