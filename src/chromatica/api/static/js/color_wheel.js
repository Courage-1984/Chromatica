// Chromatica Interactive Color Wheel (HSL/HSV)
// Provides interactive color picking functionality

console.log('Loading Chromatica Color Wheel...');

/**
 * Create an interactive color wheel
 */
window.createColorWheel = function(containerId = 'colorWheelContainer') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error('Color wheel container not found:', containerId);
        return;
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    canvas.style.cssText = 'cursor: crosshair; border-radius: 50%; border: 3px solid var(--surface2); box-shadow: 0 4px 12px rgba(0,0,0,0.3);';
    
    const ctx = canvas.getContext('2d');
    
    // Draw color wheel
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 180;
    
    // Draw HSL color wheel
    for (let angle = 0; angle < 360; angle += 0.5) {
        const hue = angle;
        for (let r = 0; r < radius; r += 2) {
            const saturation = r / radius * 100;
            const lightness = 50; // Fixed lightness for color wheel
            const x = centerX + Math.cos(angle * Math.PI / 180) * r;
            const y = centerY + Math.sin(angle * Math.PI / 180) * r;
            
            const color = hslToRgb(hue / 360, saturation / 100, lightness / 100);
            ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
            ctx.fillRect(x, y, 2, 2);
        }
    }
    
    // Lightness slider
    const lightnessCanvas = document.createElement('canvas');
    lightnessCanvas.width = 400;
    lightnessCanvas.height = 40;
    lightnessCanvas.style.cssText = 'border-radius: 8px; border: 2px solid var(--surface2); margin-top: 20px;';
    
    const lightnessCtx = lightnessCanvas.getContext('2d');
    const gradient = lightnessCtx.createLinearGradient(0, 0, lightnessCanvas.width, 0);
    for (let i = 0; i <= 100; i += 10) {
        const color = hslToRgb(0, 0, i / 100);
        gradient.addColorStop(i / 100, `rgb(${color.r}, ${color.g}, ${color.b})`);
    }
    lightnessCtx.fillStyle = gradient;
    lightnessCtx.fillRect(0, 0, lightnessCanvas.width, lightnessCanvas.height);
    
    // Current color display
    const colorDisplay = document.createElement('div');
    colorDisplay.id = 'colorWheelDisplay';
    colorDisplay.style.cssText = 'width: 100px; height: 100px; border-radius: 12px; border: 3px solid var(--surface2); margin: 20px auto; background: #FF0000; box-shadow: 0 4px 12px rgba(0,0,0,0.3);';
    
    // Color info display
    const colorInfo = document.createElement('div');
    colorInfo.id = 'colorWheelInfo';
    colorInfo.style.cssText = 'text-align: center; color: var(--text); font-family: "JetBrainsMono Nerd Font Mono", monospace; margin: 10px 0;';
    colorInfo.innerHTML = '<div>HSL: <span id="hslValue">0, 100%, 50%</span></div><div>HEX: <span id="hexValue">#FF0000</span></div>';
    
    // Add button
    const addButton = document.createElement('button');
    addButton.textContent = 'Add Color to Palette';
    addButton.style.cssText = 'width: 100%; background: var(--teal); color: #000000; border: none; padding: 12px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: bold; margin-top: 15px;';
    
    let currentHue = 0;
    let currentSaturation = 100;
    let currentLightness = 50;
    let isDraggingWheel = false;
    let isDraggingLightness = false;
    
    // Helper function to update color from wheel coordinates
    function updateColorFromWheel(e, targetCanvas) {
        const rect = targetCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left - centerX;
        const y = e.clientY - rect.top - centerY;
        const distance = Math.sqrt(x * x + y * y);
        
        // Always update hue and saturation, clamping distance to radius if outside
        const clampedDistance = Math.min(distance, radius);
        const angle = (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
        currentHue = angle;
        currentSaturation = (clampedDistance / radius) * 100;
        updateColorDisplay();
    }
    
    // Helper function to update lightness from slider coordinates
    function updateLightnessFromSlider(e, targetCanvas) {
        const rect = targetCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        // Clamp x to slider bounds even when dragging outside
        const clampedX = Math.max(0, Math.min(targetCanvas.width, x));
        currentLightness = (clampedX / targetCanvas.width) * 100;
        updateColorDisplay();
    }
    
    // Color wheel mouse down - start dragging
    canvas.addEventListener('mousedown', (e) => {
        e.preventDefault(); // Prevent text selection
        isDraggingWheel = true;
        updateColorFromWheel(e, canvas);
        canvas.style.cursor = 'grabbing';
    });
    
    // Color wheel mouse move - update while dragging
    canvas.addEventListener('mousemove', (e) => {
        if (isDraggingWheel) {
            updateColorFromWheel(e, canvas);
        }
    });
    
    // Global mouse move - handle dragging even outside canvas
    const handleGlobalMouseMove = (e) => {
        if (isDraggingWheel) {
            updateColorFromWheel(e, canvas);
        }
        if (isDraggingLightness) {
            updateLightnessFromSlider(e, lightnessCanvas);
        }
    };
    
    // Color wheel mouse up - stop dragging
    canvas.addEventListener('mouseup', () => {
        isDraggingWheel = false;
        canvas.style.cursor = 'crosshair';
    });
    
    // Color wheel mouse leave - keep dragging if mouse leaves (but don't stop)
    canvas.addEventListener('mouseleave', () => {
        // Don't stop dragging - allow continuous drag outside canvas
        // Only stop if mouse button is released (handled by global mouseup)
    });
    
    // Color wheel click handler (for compatibility)
    canvas.addEventListener('click', (e) => {
        updateColorFromWheel(e, canvas);
    });
    
    // Lightness slider mouse down - start dragging
    lightnessCanvas.addEventListener('mousedown', (e) => {
        e.preventDefault(); // Prevent text selection
        isDraggingLightness = true;
        updateLightnessFromSlider(e, lightnessCanvas);
        lightnessCanvas.style.cursor = 'grabbing';
    });
    
    // Lightness slider mouse move - update while dragging
    lightnessCanvas.addEventListener('mousemove', (e) => {
        if (isDraggingLightness) {
            updateLightnessFromSlider(e, lightnessCanvas);
        }
    });
    
    // Lightness slider mouse up - stop dragging
    lightnessCanvas.addEventListener('mouseup', () => {
        isDraggingLightness = false;
        lightnessCanvas.style.cursor = 'pointer';
    });
    
    // Lightness slider mouse leave - keep dragging if mouse leaves (but don't stop)
    lightnessCanvas.addEventListener('mouseleave', () => {
        // Don't stop dragging - allow continuous drag outside slider
        // Only stop if mouse button is released (handled by global mouseup)
    });
    
    // Lightness slider click handler (for compatibility)
    lightnessCanvas.addEventListener('click', (e) => {
        updateLightnessFromSlider(e, lightnessCanvas);
    });
    
    // Global mouse move - handle dragging even outside canvas/slider
    document.addEventListener('mousemove', handleGlobalMouseMove);
    
    // Global mouse up to handle dragging when mouse leaves element
    document.addEventListener('mouseup', () => {
        isDraggingWheel = false;
        isDraggingLightness = false;
        canvas.style.cursor = 'crosshair';
        lightnessCanvas.style.cursor = 'pointer';
    });
    
    // Update color display
    function updateColorDisplay() {
        const color = hslToRgb(currentHue / 360, currentSaturation / 100, currentLightness / 100);
        const hex = rgbToHex(color.r, color.g, color.b);
        
        colorDisplay.style.background = hex;
        document.getElementById('hslValue').textContent = `${Math.round(currentHue)}, ${Math.round(currentSaturation)}%, ${Math.round(currentLightness)}%`;
        document.getElementById('hexValue').textContent = hex;
    }
    
    // Add color button handler
    addButton.addEventListener('click', () => {
        const color = hslToRgb(currentHue / 360, currentSaturation / 100, currentLightness / 100);
        const hex = rgbToHex(color.r, color.g, color.b);
        addColorFromWheel(hex);
    });
    
    // Assemble UI
    container.innerHTML = '';
    container.appendChild(canvas);
    container.appendChild(colorDisplay);
    container.appendChild(colorInfo);
    container.appendChild(lightnessCanvas);
    container.appendChild(addButton);
    
    updateColorDisplay();
};

/**
 * HSL to RGB conversion
 */
function hslToRgb(h, s, l) {
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
}

/**
 * RGB to Hex conversion
 */
function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(x => {
        const hex = x.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('').toUpperCase();
}

/**
 * Add color from wheel to palette
 */
function addColorFromWheel(hex) {
    if (window.addColorRow) {
        window.addColorRow(hex, 100);
        showSuccess('Color Added', `Color ${hex} added to palette`);
    } else {
        showError('Add Color', 'Cannot add color - palette system not loaded');
    }
}

console.log('✓ Chromatica Color Wheel loaded');

