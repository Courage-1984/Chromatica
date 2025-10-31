/**
 * Reranking Animation - Watch the two-stage search pipeline in action
 * 
 * Visualizes the search process: FAISS ANN search followed by Sinkhorn-EMD reranking
 * Shows candidates moving from initial positions to final ranked positions.
 */

(function() {
    'use strict';

    let scene, camera, renderer, controls;
    let annCandidates = [];
    let rerankedResults = [];
    let animationFrame = null;
    let isAnimating = false;

    /**
     * Perform search to get both ANN and final results
     */
    async function performAnimatedSearch(colors, weights) {
        try {
            const params = new URLSearchParams({
                colors: colors,
                weights: weights,
                k: '20',
                fast_mode: 'false' // Must be false to show reranking
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
     * Initialize Three.js scene
     */
    function initScene() {
        console.log('[Reranking Animation] Initializing scene...');
        
        window.clear3DVisualization();
        window.show3DLoading();
        
        const container = window.get3DContainer();
        if (!container) {
            console.error('[Reranking Animation] Container not found');
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
            console.log('[Reranking Animation] Axes enabled');
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
     * Create initial ANN candidate positions (scattered)
     */
    function createANNPositions(results) {
        return results.map((result, index) => {
            const angle = (index / results.length) * Math.PI * 2;
            const radius = 50 + Math.random() * 30;
            return {
                x: Math.cos(angle) * radius,
                y: Math.random() * 50,
                z: Math.sin(angle) * radius,
                result: result
            };
        });
    }

    /**
     * Create final reranked positions (organized by distance)
     */
    function createRerankedPositions(results) {
        return results.map((result, index) => {
            const distance = result.distance || 1.0;
            const x = (index - results.length / 2) * 10;
            const y = (1 - distance) * 100; // Better matches higher
            const z = distance * 20;
            return {
                x: x,
                y: y,
                z: z,
                result: result
            };
        });
    }

    /**
     * Create visualization objects
     */
    function createVisualization(annPositions, rerankedPositions) {
        // Clear existing
        annCandidates.forEach(obj => scene.remove(obj));
        rerankedResults.forEach(obj => scene.remove(obj));
        annCandidates = [];
        rerankedResults = [];

        // Create spheres for ANN candidates (initial positions)
        annPositions.forEach((pos, index) => {
            const result = pos.result;
            let colorHex = '#808080';
            if (result.dominant_colors && result.dominant_colors.length > 0) {
                colorHex = result.dominant_colors[0].hex || result.dominant_colors[0];
            }

            const rgb = hexToRgb(colorHex);
            const geometry = new THREE.SphereGeometry(3, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(rgb.r / 255, rgb.g / 255, rgb.b / 255),
                transparent: true,
                opacity: 0.5
            });

            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(pos.x, pos.y, pos.z);
            sphere.userData = { position: pos, targetPosition: rerankedPositions[index], result };
            
            scene.add(sphere);
            annCandidates.push(sphere);
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
     * Animate transition from ANN to reranked positions
     */
    let animationProgress = 0;
    function animateTransition() {
        if (!isAnimating) return;

        animationProgress += 0.02;
        if (animationProgress >= 1.0) {
            animationProgress = 1.0;
            isAnimating = false;
        }

        // Interpolate positions
        annCandidates.forEach((sphere, index) => {
            const start = sphere.userData.position;
            const target = sphere.userData.targetPosition;
            
            const easeProgress = easeInOutCubic(animationProgress);
            
            sphere.position.x = start.x + (target.x - start.x) * easeProgress;
            sphere.position.y = start.y + (target.y - start.y) * easeProgress;
            sphere.position.z = start.z + (target.z - start.z) * easeProgress;

            // Update opacity
            sphere.material.opacity = 0.5 + easeProgress * 0.5;
            sphere.material.needsUpdate = true;
        });

        if (controls) controls.update();
        if (renderer && scene && camera) {
            renderer.render(scene, camera);
        }

        if (isAnimating) {
            animationFrame = requestAnimationFrame(animateTransition);
        }
    }

    function easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    /**
     * Animation loop
     */
    function animate() {
        if (!window.current3DVisualization.isPaused && !isAnimating) {
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
    window.generateRerankingAnimation = async function() {
        const colors = document.getElementById('animationColors')?.value;
        const weights = document.getElementById('animationWeights')?.value;
        
        if (!colors || !weights) {
            alert('Please enter colors and weights');
            return;
        }

        console.log('[Reranking Animation] ===== Starting generation =====');
        console.log(`[Reranking Animation] Colors: ${colors}, Weights: ${weights}`);
        
        try {
            if (!initScene()) {
                throw new Error('Failed to initialize scene');
            }

            console.log('[Reranking Animation] Performing search...');
            const results = await performAnimatedSearch(colors, weights);
            console.log(`[Reranking Animation] Found ${results.length} results`);

            if (results.length === 0) {
                window.hide3DLoading();
                alert('No results found for this query.');
                return;
            }

            console.log('[Reranking Animation] Creating visualization...');
            const annPositions = createANNPositions(results);
            const rerankedPositions = createRerankedPositions(results);
            createVisualization(annPositions, rerankedPositions);
            console.log(`[Reranking Animation] Created ${annCandidates.length} candidate spheres`);

            window.hide3DLoading();

            // Start static animation first
            const enableAnimations = document.getElementById('enable3DAnimations')?.checked !== false;
            if (enableAnimations) {
                console.log('[Reranking Animation] Starting animation loop...');
                animate();

                // After a delay, start the transition animation
                setTimeout(() => {
                    console.log('[Reranking Animation] Starting transition animation...');
                    isAnimating = true;
                    animationProgress = 0;
                    animateTransition();
                }, 1000);
            } else {
                console.log('[Reranking Animation] Animations disabled, rendering static view');
                if (controls) controls.update();
                if (renderer && scene && camera) {
                    renderer.render(scene, camera);
                }
            }

            console.log('[Reranking Animation] ===== Generation complete =====');
        } catch (error) {
            console.error('[Reranking Animation] Error:', error);
            window.hide3DLoading();
            alert(`Failed to generate Reranking Animation: ${error.message}`);
        }
    };

    /**
     * Export data
     */
    window.exportAnimationData = function() {
        if (annCandidates.length === 0) {
            alert('No animation data to export. Please generate the animation first.');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            query: {
                colors: document.getElementById('animationColors')?.value,
                weights: document.getElementById('animationWeights')?.value
            },
            ann_candidates: annCandidates.map(obj => ({
                position: obj.userData.position,
                result: obj.userData.result
            })),
            reranked_results: annCandidates.map(obj => ({
                position: obj.userData.targetPosition,
                result: obj.userData.result
            }))
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `reranking_animation_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);

        console.log('Animation data exported');
    };

})();
