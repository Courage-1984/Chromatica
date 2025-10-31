/**
 * Color Space Navigator - 3D visualization of indexed images in CIE Lab color space
 * 
 * Fetches image data from the Chromatica API and visualizes them as points
 * positioned by their dominant colors in 3D Lab space.
 */

(function() {
    'use strict';

    let scene, camera, renderer, controls;
    let points = [];
    let currentData = null;

    /**
     * Fetch image data from the API for visualization
     */
    async function fetchImageData(limit = 500) {
        try {
            // Get a sample of images from the index
            // For now, we'll use search with a neutral query to get diverse results
            const response = await fetch(`/api/info`);
            if (!response.ok) throw new Error('Failed to fetch API info');
            
            // Since we don't have a direct "get all images" endpoint, we'll need to
            // use a workaround - perform a broad search to get sample images
            // In production, you might want to add a dedicated endpoint
            const searchResponse = await fetch(`/search?colors=808080&weights=1.0&k=${limit}&fast_mode=true`);
            if (!searchResponse.ok) throw new Error('Failed to fetch image data');
            
            const data = await searchResponse.json();
            return data.results || [];
        } catch (error) {
            console.error('Error fetching image data:', error);
            return [];
        }
    }

    /**
     * Convert hex color to Lab color space (approximation)
     */
    function hexToLab(hex) {
        // Remove # if present
        hex = hex.replace('#', '');
        const r = parseInt(hex.substring(0, 2), 16) / 255;
        const g = parseInt(hex.substring(2, 4), 16) / 255;
        const b = parseInt(hex.substring(4, 6), 16) / 255;

        // Convert RGB to Lab (simplified - using sRGB to XYZ to Lab)
        // This is a simplified conversion - for production, use a proper color library
        const [X, Y, Z] = rgbToXyz(r, g, b);
        const [L, a, b_] = xyzToLab(X, Y, Z);
        
        return { L, a, b: b_ };
    }

    function rgbToXyz(r, g, b) {
        // Apply gamma correction
        r = r > 0.04045 ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
        g = g > 0.04045 ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
        b = b > 0.04045 ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

        // Convert to XYZ (sRGB D65)
        const X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
        const Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
        const Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
        
        return [X, Y, Z];
    }

    function xyzToLab(X, Y, Z) {
        // D65 white point
        const Xn = 0.95047, Yn = 1.00000, Zn = 1.08883;
        
        const fx = f(X / Xn);
        const fy = f(Y / Yn);
        const fz = f(Z / Zn);
        
        const L = 116 * fy - 16;
        const a = 500 * (fx - fy);
        const b_ = 200 * (fy - fz);
        
        return [L, a, b_];
    }

    function f(t) {
        const delta = 6 / 29;
        if (t > delta * delta * delta) {
            return Math.pow(t, 1 / 3);
        }
        return t / (3 * delta * delta) + 4 / 29;
    }

    /**
     * Initialize Three.js scene
     */
    function initScene() {
        console.log('[Color Space Navigator] Initializing scene...');
        
        // Clear any existing visualization
        window.clear3DVisualization();
        
        // Show loading
        window.show3DLoading();
        
        const container = window.get3DContainer();
        if (!container) {
            console.error('[Color Space Navigator] Container not found');
            return false;
        }
        
        // Clear container
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) {
            existingCanvas.remove();
        }
        
        // Scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1e1e2e); // Catppuccin Mocha base

        // Camera
        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(150, 150, 150);
        camera.lookAt(0, 0, 0);

        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Controls
        if (typeof THREE.OrbitControls !== 'undefined') {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
        }

        // Store references globally
        window.current3DVisualization.scene = scene;
        window.current3DVisualization.camera = camera;
        window.current3DVisualization.renderer = renderer;
        window.current3DVisualization.controls = controls;

        // Add axes helper (if enabled)
        const showAxes = document.getElementById('show3DAxes')?.checked !== false;
        if (showAxes) {
            const axesHelper = new THREE.AxesHelper(100);
            scene.add(axesHelper);
            console.log('[Color Space Navigator] Axes enabled');
        }

        // Add grid
        const gridHelper = new THREE.GridHelper(200, 20, 0x45475a, 0x313244);
        scene.add(gridHelper);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        scene.add(directionalLight);
        
        return true;
    }

    /**
     * Create points from image data
     */
    function createPoints(imageData) {
        // Clear existing points
        points.forEach(point => scene.remove(point));
        points = [];

        if (!imageData || imageData.length === 0) {
            console.warn('No image data to visualize');
            return;
        }

        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const sizes = [];

        imageData.forEach((result, index) => {
            // Get dominant color (first color from dominant_colors if available)
            let colorHex = '#808080'; // default gray
            if (result.dominant_colors && result.dominant_colors.length > 0) {
                colorHex = result.dominant_colors[0].hex || result.dominant_colors[0];
            }

            // Convert to Lab
            const lab = hexToLab(colorHex);

            // Map Lab to 3D space (L: 0-100, a: -86 to 98, b: -108 to 95)
            const x = lab.a;  // a* axis
            const y = lab.L;  // L* axis (vertical)
            const z = lab.b;  // b* axis

            positions.push(x, y, z);

            // Convert hex to RGB for Three.js
            const rgb = hexToRgb(colorHex);
            colors.push(rgb.r / 255, rgb.g / 255, rgb.b / 255);

            // Size based on distance (closer = larger)
            const size = result.distance ? Math.max(1, 5 - result.distance * 2) : 2;
            sizes.push(size);
        });

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

        const material = new THREE.PointsMaterial({
            size: 5,
            vertexColors: true,
            sizeAttenuation: true,
        });

        const pointsMesh = new THREE.Points(geometry, material);
        scene.add(pointsMesh);
        points.push(pointsMesh);
    }

    function hexToRgb(hex) {
        hex = hex.replace('#', '');
        return {
            r: parseInt(hex.substring(0, 2), 16),
            g: parseInt(hex.substring(2, 4), 16),
            b: parseInt(hex.substring(4, 6), 16)
        };
    }

    /**
     * Animation loop
     */
    function animate() {
        if (!window.current3DVisualization.isPaused) {
            window.current3DVisualization.animationId = requestAnimationFrame(animate);
            if (controls) controls.update();
            if (renderer && scene && camera) {
                renderer.render(scene, camera);
            }
        }
    }

    /**
     * Main function to generate the Color Space Navigator
     */
    window.generateColorSpaceNavigator = async function() {
        console.log('[Color Space Navigator] ===== Starting generation =====');
        
        try {
            // Clear previous and show loading
            if (!initScene()) {
                throw new Error('Failed to initialize scene');
            }

            console.log('[Color Space Navigator] Fetching image data...');
            // Fetch data
            const imageData = await fetchImageData(500);
            currentData = imageData;
            console.log(`[Color Space Navigator] Fetched ${imageData.length} images`);

            if (imageData.length === 0) {
                window.hide3DLoading();
                alert('No image data available. Please ensure the index is loaded.');
                return;
            }

            // Create visualization
            console.log('[Color Space Navigator] Creating points visualization...');
            createPoints(imageData);
            console.log(`[Color Space Navigator] Created ${points.length} point mesh(es)`);

            // Hide loading
            window.hide3DLoading();

            // Start animation
            const enableAnimations = document.getElementById('enable3DAnimations')?.checked !== false;
            if (enableAnimations) {
                console.log('[Color Space Navigator] Starting animation loop...');
                animate();
            } else {
                console.log('[Color Space Navigator] Animations disabled, rendering static view');
                if (controls) controls.update();
                if (renderer && scene && camera) {
                    renderer.render(scene, camera);
                }
            }

            console.log(`[Color Space Navigator] ===== Generation complete with ${imageData.length} points =====`);
        } catch (error) {
            console.error('[Color Space Navigator] Error:', error);
            window.hide3DLoading();
            alert(`Failed to generate Color Space Navigator: ${error.message}`);
        }
    };

    /**
     * Export data as JSON
     */
    window.exportColorSpaceData = function() {
        if (!currentData || currentData.length === 0) {
            alert('No data to export. Please generate the navigator first.');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            total_images: currentData.length,
            images: currentData.map(img => ({
                image_id: img.image_id,
                image_url: img.image_url,
                distance: img.distance,
                dominant_colors: img.dominant_colors
            }))
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `color_space_navigator_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);

        console.log('Color Space Navigator data exported');
    };

})();
