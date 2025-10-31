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
     * Simple 3D positioning based on distance from query
     */
    function positionResults(results) {
        // Use distance to create a landscape
        // Closer results are higher and closer to center
        return results.map((result, index) => {
            const distance = result.distance || 1.0;
            const angle = (index / results.length) * Math.PI * 2;
            
            // Position in 3D space: spiral outward based on distance
            const radius = distance * 30;
            const height = (1 - distance) * 100; // Closer = higher
            
            return {
                x: Math.cos(angle) * radius,
                y: height,
                z: Math.sin(angle) * radius,
                result: result
            };
        });
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

        // Create spheres for each result
        positions.forEach((pos, index) => {
            const result = pos.result;
            const distance = result.distance || 1.0;

            // Get color from dominant colors or default
            let colorHex = '#808080';
            if (result.dominant_colors && result.dominant_colors.length > 0) {
                colorHex = result.dominant_colors[0].hex || result.dominant_colors[0];
            }

            const rgb = hexToRgb(colorHex);
            const size = Math.max(2, 10 - distance * 5);

            const geometry = new THREE.SphereGeometry(size, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(rgb.r / 255, rgb.g / 255, rgb.b / 255),
                emissive: new THREE.Color(rgb.r / 510, rgb.g / 510, rgb.b / 510),
                transparent: true,
                opacity: 0.8
            });

            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(pos.x, pos.y, pos.z);
            sphere.userData = { result, index };

            // Add label (text sprite would be better, but simple approach)
            scene.add(sphere);
            resultPoints.push(sphere);
        });
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
