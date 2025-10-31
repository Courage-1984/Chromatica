/**
 * Similarity Landscape - Explore search results in 3D using dimensionality reduction
 * 
 * Performs a search query and visualizes results using 3D positioning based on
 * similarity/distance relationships.
 */

(function() {
    'use strict';

    let scene, camera, renderer, controls;
    let resultPoints = [];
    let currentResults = null;
    let currentQuery = null;

    /**
     * Perform search and get results
     */
    async function performSearch(colors, weights) {
        try {
            const params = new URLSearchParams({
                colors: colors,
                weights: weights,
                k: '50',
                fast_mode: 'false'
            });

            const response = await fetch(`/search?${params.toString()}`);
            if (!response.ok) throw new Error('Search failed');
            
            const data = await response.json();
            return data.results || [];
        } catch (error) {
            console.error('Error performing search:', error);
            throw error;
        }
    }

    /**
     * Position results in 3D space based on similarity/distance
     * Similar results cluster together, creating a "landscape" of similarity
     */
    function positionResults(results) {
        if (!results || results.length === 0) return [];
        
        // Normalize distances to [0, 1] range
        const distances = results.map(r => r.distance || 1.0);
        const minDist = Math.min(...distances);
        const maxDist = Math.max(...distances);
        const distRange = maxDist - minDist || 1.0;
        
        // Extract dominant colors for positioning
        const positions = results.map((result, index) => {
            const normalizedDist = distRange > 0 ? (result.distance - minDist) / distRange : 0.5;
            
            // Get dominant color for positioning in Lab space
            let colorHex = '#808080';
            if (result.dominant_colors && result.dominant_colors.length > 0) {
                const firstColor = result.dominant_colors[0];
                if (typeof firstColor === 'string') {
                    colorHex = firstColor.startsWith('#') ? firstColor : '#' + firstColor;
                } else if (firstColor && typeof firstColor === 'object') {
                    colorHex = firstColor.hex || (firstColor.color ? (firstColor.color.startsWith('#') ? firstColor.color : '#' + firstColor.color) : '#808080');
                }
            }
            
            // Convert to Lab for positioning
            const lab = hexToLab(colorHex);
            
            // Position in 3D space: use Lab coordinates scaled + distance for height
            // Similar colors cluster together, distance determines height
            const x = lab.a * 0.8; // Scale a* component
            const y = normalizedDist * 80; // Height based on distance (farther = lower)
            const z = lab.b * 0.8; // Scale b* component
            
            return {
                x: isFinite(x) ? x : (Math.random() - 0.5) * 50,
                y: isFinite(y) ? y : normalizedDist * 80,
                z: isFinite(z) ? z : (Math.random() - 0.5) * 50,
                result: result,
                colorHex: colorHex
            };
        });
        
        return positions;
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
        return [
            r * 0.4124564 + g * 0.3575761 + b * 0.1804375,
            r * 0.2126729 + g * 0.7151522 + b * 0.0721750,
            r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        ];
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

    /**
     * Initialize Three.js scene
     */
    function initScene() {
        console.log('[Similarity Landscape] Initializing scene...');
        
        window.clear3DVisualization();
        window.show3DLoading();
        
        const container = window.get3DContainer();
        if (!container) {
            console.error('[Similarity Landscape] Container not found');
            return false;
        }
        
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) existingCanvas.remove();
        
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1e1e2e);

        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(150, 150, 150);
        camera.lookAt(0, 0, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        if (typeof THREE.OrbitControls !== 'undefined') {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
        }

        window.current3DVisualization.scene = scene;
        window.current3DVisualization.camera = camera;
        window.current3DVisualization.renderer = renderer;
        window.current3DVisualization.controls = controls;

        const showAxes = document.getElementById('show3DAxes')?.checked !== false;
        if (showAxes) {
            const axesHelper = new THREE.AxesHelper(100);
            scene.add(axesHelper);
            console.log('[Similarity Landscape] Axes enabled');
        }

        const gridHelper = new THREE.GridHelper(200, 20, 0x45475a, 0x313244);
        scene.add(gridHelper);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        scene.add(directionalLight);
        
        return true;
    }

    /**
     * Create visualization from search results
     */
    function createVisualization(results) {
        // Clear existing
        resultPoints.forEach(point => scene.remove(point));
        resultPoints = [];

        if (!results || results.length === 0) return;

        const positions = positionResults(results);

        // Create separate spheres for each result (no connections)
        positions.forEach((pos, index) => {
            const result = pos.result;
            const distance = result.distance || 1.0;

            // Use color from positioning calculation
            const colorHex = pos.colorHex || '#808080';
            const rgb = hexToRgb(colorHex);
            
            // Size based on distance (closer = larger, but reasonable range)
            const normalizedDist = distance / (Math.max(...results.map(r => r.distance || 1.0)) || 1.0);
            const size = Math.max(2, Math.min(8, 10 - normalizedDist * 6));

            const geometry = new THREE.SphereGeometry(size, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(rgb.r / 255, rgb.g / 255, rgb.b / 255),
                emissive: new THREE.Color(rgb.r / 510, rgb.g / 510, rgb.b / 510),
                emissiveIntensity: 0.3,
                transparent: true,
                opacity: 0.85
            });

            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(pos.x, pos.y, pos.z);
            sphere.userData = { result, index, distance };

            // Add as separate sphere (no connections - this is a landscape, not a path)
            scene.add(sphere);
            resultPoints.push(sphere);
        });
        
        console.log(`[Similarity Landscape] Created ${resultPoints.length} separate spheres positioned by similarity`);
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
    window.generateSimilarityLandscape = async function() {
        const colors = document.getElementById('similarityColors')?.value;
        const weights = document.getElementById('similarityWeights')?.value;
        
        if (!colors || !weights) {
            alert('Please enter colors and weights');
            return;
        }

        console.log('[Similarity Landscape] ===== Starting generation =====');
        console.log(`[Similarity Landscape] Colors: ${colors}, Weights: ${weights}`);
        
        try {
            if (!initScene()) {
                throw new Error('Failed to initialize scene');
            }

            console.log('[Similarity Landscape] Performing search...');
            const results = await performSearch(colors, weights);
            currentResults = results;
            currentQuery = { colors, weights };
            console.log(`[Similarity Landscape] Found ${results.length} results`);

            if (results.length === 0) {
                window.hide3DLoading();
                alert('No results found for this query.');
                return;
            }

            console.log('[Similarity Landscape] Creating visualization...');
            createVisualization(results);
            window.hide3DLoading();

            const enableAnimations = document.getElementById('enable3DAnimations')?.checked !== false;
            if (enableAnimations) {
                console.log('[Similarity Landscape] Starting animation loop...');
                animate();
            } else {
                console.log('[Similarity Landscape] Animations disabled, rendering static view');
                if (controls) controls.update();
                if (renderer && scene && camera) {
                    renderer.render(scene, camera);
                }
            }

            console.log(`[Similarity Landscape] ===== Generation complete with ${results.length} results =====`);
        } catch (error) {
            console.error('[Similarity Landscape] Error:', error);
            window.hide3DLoading();
            alert(`Failed to generate Similarity Landscape: ${error.message}`);
        }
    };

    /**
     * Export data
     */
    window.exportSimilarityData = function() {
        if (!currentResults) {
            alert('No data to export. Please generate the landscape first.');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            query: currentQuery,
            results_count: currentResults.length,
            results: currentResults.map(r => ({
                image_id: r.image_id,
                distance: r.distance,
                dominant_colors: r.dominant_colors
            }))
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `similarity_landscape_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);

        console.log('Similarity Landscape data exported');
    };

})();
