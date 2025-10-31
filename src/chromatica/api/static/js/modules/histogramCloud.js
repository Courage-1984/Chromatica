/**
 * Histogram Cloud - Visualize image histograms as 3D bar charts in Lab color cube
 * 
 * Fetches histogram data for a specific image and visualizes it as bars
 * positioned in 3D space according to Lab bin centers.
 */

(function() {
    'use strict';

    let scene, camera, renderer, controls;
    let bars = [];
    let currentHistogram = null;
    let currentImageId = null;

    /**
     * Fetch histogram data for an image
     */
    async function fetchHistogram(imageId) {
        try {
            // Since we don't have a direct histogram endpoint, we'll need to
            // use a search result that contains the image, or create an endpoint
            // For now, we'll use the search API and find the matching result
            const response = await fetch(`/search?colors=808080&weights=1.0&k=1000&fast_mode=true`);
            if (!response.ok) throw new Error('Failed to fetch search results');
            
            const data = await response.json();
            const result = data.results.find(r => r.image_id === imageId);
            
            if (!result) {
                throw new Error(`Image ${imageId} not found in search results`);
            }

            // For histogram, we'd ideally fetch raw histogram data from an endpoint
            // For now, we'll approximate from dominant colors
            // In production, add: GET /api/histogram/{image_id}
            return {
                image_id: imageId,
                dominant_colors: result.dominant_colors || [],
                image_url: result.image_url
            };
        } catch (error) {
            console.error('Error fetching histogram:', error);
            throw error;
        }
    }

    /**
     * Initialize Three.js scene
     */
    function initScene() {
        console.log('[Histogram Cloud] Initializing scene...');
        
        // Clear any existing visualization
        window.clear3DVisualization();
        
        // Show loading
        window.show3DLoading();
        
        const container = window.get3DContainer();
        if (!container) {
            console.error('[Histogram Cloud] Container not found');
            return false;
        }
        
        // Clear container
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) {
            existingCanvas.remove();
        }
        
        // Scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1e1e2e);

        // Camera
        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 2000);
        camera.position.set(100, 100, 100);
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

        // Add axes
        const showAxes = document.getElementById('show3DAxes')?.checked !== false;
        if (showAxes) {
            const axesHelper = new THREE.AxesHelper(50);
            scene.add(axesHelper);
            console.log('[Histogram Cloud] Axes enabled');
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
     * Create histogram bars from dominant colors
     */
    function createHistogramBars(histogramData) {
        // Clear existing bars
        bars.forEach(bar => scene.remove(bar));
        bars = [];

        if (!histogramData || !histogramData.dominant_colors) {
            console.warn('No histogram data available');
            return;
        }

        const colors = Array.isArray(histogramData.dominant_colors) ? histogramData.dominant_colors : [];
        // Compute a safe max proportion to avoid division by zero
        let maxProportion = Math.max(...colors.map(c => (typeof c.p === 'number' ? c.p : 0)));
        const hasValidP = colors.some(c => typeof c.p === 'number' && isFinite(c.p) && c.p > 0);
        if (!isFinite(maxProportion) || maxProportion <= 0) {
            // Fallback if proportions are missing or zero; we'll normalize later
            maxProportion = 1;
        }

        colors.forEach((colorData, index) => {
            const rawHex = (colorData && typeof colorData === 'object') ? colorData.hex : colorData;
            const hex = (typeof rawHex === 'string') ? rawHex.replace('#','').toUpperCase() : null;
            if (!hex || !/^[0-9A-F]{6}$/.test(hex)) {
                console.warn('[Histogram Cloud] Skipping invalid hex color:', rawHex);
                return;
            }
            const proportion = (typeof colorData.p === 'number' && isFinite(colorData.p))
                ? colorData.p
                : (1 / Math.max(1, colors.length));

            // Convert hex to Lab (simplified)
            const lab = hexToLab('#' + hex);

            // Map Lab to 3D position
            // Clamp Lab to safe ranges
            const x = isFinite(lab.a) ? Math.max(-128, Math.min(128, lab.a)) : 0;
            const y = isFinite(lab.L) ? Math.max(0, Math.min(100, lab.L)) : 0;
            const z = isFinite(lab.b) ? Math.max(-128, Math.min(128, lab.b)) : 0;

            // Bar height based on proportion; guard against zero/NaN
            let height = hasValidP ? (proportion / Math.max(1e-6, maxProportion)) * 50 : 30;
            if (!isFinite(height) || height <= 0) height = 10;
            height = Math.max(0.1, height);

            // Create bar geometry
            const geometry = new THREE.BoxGeometry(2, height, 2);
            const rgb = hexToRgb('#' + hex);
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(rgb.r / 255, rgb.g / 255, rgb.b / 255),
                emissive: new THREE.Color(rgb.r / 510, rgb.g / 510, rgb.b / 510),
                specular: new THREE.Color(0x111111),
                shininess: 30
            });

            const bar = new THREE.Mesh(geometry, material);
            // Ensure finite positions
            const px = isFinite(x) ? x : 0;
            const py = isFinite(y) ? y : 0;
            const pz = isFinite(z) ? z : 0;
            bar.position.set(px, py + height / 2, pz);
            bar.userData = { colorData, imageId: histogramData.image_id };

            scene.add(bar);
            bars.push(bar);
        });
    }

    function hexToLab(hex) {
        hex = hex.replace('#', '');
        const r = parseInt(hex.substring(0, 2), 16) / 255;
        const g = parseInt(hex.substring(2, 4), 16) / 255;
        const b = parseInt(hex.substring(4, 6), 16) / 255;

        const [X, Y, Z] = rgbToXyz(r, g, b);
        const [L, a, b_] = xyzToLab(X, Y, Z);
        
        return { L, a, b: b_ };
    }

    function rgbToXyz(r, g, b) {
        r = r > 0.04045 ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
        g = g > 0.04045 ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
        b = b > 0.04045 ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

        const X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
        const Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
        const Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
        
        return [X, Y, Z];
    }

    function xyzToLab(X, Y, Z) {
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
     * Main function
     */
    window.generateHistogramCloud = async function() {
        const imageId = document.getElementById('histogramImageId')?.value;
        if (!imageId) {
            alert('Please enter an Image ID');
            return;
        }

        console.log('[Histogram Cloud] ===== Starting generation =====');
        console.log(`[Histogram Cloud] Image ID: ${imageId}`);
        
        try {
            if (!initScene()) {
                throw new Error('Failed to initialize scene');
            }

            console.log('[Histogram Cloud] Fetching histogram data...');
            const histogramData = await fetchHistogram(imageId);
            currentHistogram = histogramData;
            currentImageId = imageId;
            console.log(`[Histogram Cloud] Fetched histogram data for image ${imageId}`);

            console.log('[Histogram Cloud] Creating histogram bars...');
            createHistogramBars(histogramData);
            console.log(`[Histogram Cloud] Created ${bars.length} bars`);

            window.hide3DLoading();

            const enableAnimations = document.getElementById('enable3DAnimations')?.checked !== false;
            if (enableAnimations) {
                console.log('[Histogram Cloud] Starting animation loop...');
                animate();
            } else {
                console.log('[Histogram Cloud] Animations disabled, rendering static view');
                if (controls) controls.update();
                if (renderer && scene && camera) {
                    renderer.render(scene, camera);
                }
            }

            console.log('[Histogram Cloud] ===== Generation complete =====');
        } catch (error) {
            console.error('[Histogram Cloud] Error:', error);
            window.hide3DLoading();
            alert(`Failed to generate Histogram Cloud: ${error.message}`);
        }
    };

    /**
     * Export data
     */
    window.exportHistogramData = function() {
        if (!currentHistogram) {
            alert('No histogram data to export. Please generate the cloud first.');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            image_id: currentImageId,
            histogram: currentHistogram
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `histogram_${currentImageId}_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);

        console.log('Histogram data exported');
    };

})();
