/**
 * Image Thumbnails 3D - Visualize all indexed images as thumbnails in Lab color space
 * 
 * Displays small thumbnail images positioned in 3D space by their dominant colors,
 * creating a visual index browser in CIE Lab color space.
 */

(function() {
    'use strict';

    let scene, camera, renderer, controls;
    let thumbnails = [];
    let currentData = null;
    let textureLoader = null;

    /**
     * Generate random hex color
     */
    function randomHexColor() {
        return Math.floor(Math.random() * 16777215).toString(16).toUpperCase().padStart(6, '0');
    }

    /**
     * Fetch image data from the API using multiple random color queries for diversity
     */
    async function fetchImageData(limit = 300) {
        try {
            console.log(`[Thumbnails 3D] Fetching ${limit} diverse images using multiple random queries...`);
            
            // Calculate how many queries we need (API limit is 50 per request)
            const maxPerQuery = 50; // API MAX_SEARCH_RESULTS
            const numQueries = Math.ceil(limit / maxPerQuery);
            const perQuery = Math.min(maxPerQuery, limit);
            
            console.log(`[Thumbnails 3D] Will make ${numQueries} queries for ${perQuery} images each`);
            
            // Collect all unique results
            const allResults = new Map(); // Use Map to deduplicate by image_id
            const seenImageIds = new Set();
            
            // Make multiple queries with random colors for diversity
            const queries = [];
            for (let i = 0; i < numQueries; i++) {
                // Generate random colors for diversity
                const randomColors = [];
                const numColors = Math.floor(Math.random() * 2) + 1; // 1-2 colors
                for (let j = 0; j < numColors; j++) {
                    randomColors.push(randomHexColor());
                }
                const colors = randomColors.join(',');
                const weights = Array(numColors).fill(1.0 / numColors).map(w => w.toFixed(3)).join(',');
                
                queries.push(
                    fetch(`/search?colors=${colors}&weights=${weights}&k=${perQuery}&fast_mode=true`)
                        .then(res => {
                            if (!res.ok) {
                                console.warn(`[Thumbnails 3D] Query ${i+1} failed: ${res.status}`);
                                return { results: [] };
                            }
                            return res.json();
                        })
                        .then(data => {
                            if (data.results) {
                                data.results.forEach(result => {
                                    // Deduplicate by image_id
                                    if (!seenImageIds.has(result.image_id)) {
                                        seenImageIds.add(result.image_id);
                                        allResults.set(result.image_id, result);
                                    }
                                });
                            }
                            return data.results?.length || 0;
                        })
                        .catch(error => {
                            console.warn(`[Thumbnails 3D] Query ${i+1} error:`, error);
                            return 0;
                        })
                );
            }
            
            // Wait for all queries to complete
            const results = await Promise.all(queries);
            const totalFetched = Array.from(allResults.values()).length;
            console.log(`[Thumbnails 3D] Fetched ${totalFetched} unique images from ${numQueries} queries`);
            
            // Convert to array and limit to requested amount
            let finalResults = Array.from(allResults.values());
            
            // If we have more than requested, randomly sample
            if (finalResults.length > limit) {
                console.log(`[Thumbnails 3D] Sampling ${limit} images from ${finalResults.length} total`);
                // Shuffle and take first N
                for (let i = finalResults.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [finalResults[i], finalResults[j]] = [finalResults[j], finalResults[i]];
                }
                finalResults = finalResults.slice(0, limit);
            }
            
            console.log(`[Thumbnails 3D] Returning ${finalResults.length} images`);
            return finalResults;
        } catch (error) {
            console.error('[Thumbnails 3D] Error fetching image data:', error);
            throw error;
        }
    }

    /**
     * Initialize Three.js scene
     */
    function initScene() {
        console.log('[Thumbnails 3D] Initializing scene...');
        
        window.clear3DVisualization();
        window.show3DLoading();
        
        const container = window.get3DContainer();
        if (!container) {
            console.error('[Thumbnails 3D] Container not found');
            return false;
        }
        
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) existingCanvas.remove();
        
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1e1e2e);

        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 2000);
        camera.position.set(200, 200, 200);
        camera.lookAt(0, 50, 0);

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
        }

        const gridHelper = new THREE.GridHelper(200, 20, 0x45475a, 0x313244);
        scene.add(gridHelper);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        scene.add(directionalLight);
        
        textureLoader = new THREE.TextureLoader();
        
        return true;
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
     * Load image as texture and create thumbnail plane
     */
    function loadThumbnail(imageResult, position, index) {
        return new Promise((resolve, reject) => {
            // Get image URL
            const imageUrl = imageResult.image_url || `/image/${encodeURIComponent(imageResult.image_id)}`;
            
            // Load texture
            textureLoader.load(
                imageUrl,
                (texture) => {
                    // Create plane geometry for thumbnail
                    const width = 8; // thumbnail size
                    const height = 8;
                    const geometry = new THREE.PlaneGeometry(width, height);
                    
                    // Create material with texture
                    const material = new THREE.MeshBasicMaterial({
                        map: texture,
                        side: THREE.DoubleSide,
                        transparent: true,
                        opacity: 0.9
                    });
                    
                    // Create mesh
                    const plane = new THREE.Mesh(geometry, material);
                    plane.position.copy(position);
                    
                    // Store reference for billboard update
                    plane.userData = {
                        result: imageResult,
                        index: index,
                        isBillboard: true
                    };
                    
                    scene.add(plane);
                    thumbnails.push(plane);
                    
                    resolve(plane);
                },
                undefined,
                (error) => {
                    console.warn(`[Thumbnails 3D] Failed to load thumbnail for ${imageResult.image_id}:`, error);
                    reject(error);
                }
            );
        });
    }

    /**
     * Create thumbnails positioned by dominant color
     */
    async function createThumbnails(results) {
        // Clear existing thumbnails
        thumbnails.forEach(thumb => scene.remove(thumb));
        thumbnails = [];
        
        if (!results || results.length === 0) {
            console.warn('[Thumbnails 3D] No results to display');
            return;
        }

        console.log(`[Thumbnails 3D] Creating ${results.length} thumbnails...`);
        
        const loadPromises = [];
        let loadedCount = 0;
        
        results.forEach((result, index) => {
            // Get dominant color
            let colorHex = '#808080';
            const domColors = Array.isArray(result.dominant_colors) ? result.dominant_colors : [];
            if (domColors.length > 0) {
                const firstColor = domColors[0];
                if (typeof firstColor === 'string') {
                    colorHex = firstColor.startsWith('#') ? firstColor : '#' + firstColor;
                } else if (firstColor && typeof firstColor === 'object') {
                    colorHex = firstColor.hex || (firstColor.color ? (firstColor.color.startsWith('#') ? firstColor.color : '#' + firstColor.color) : '#808080');
                }
            }
            
            // Convert to Lab and position
            const lab = hexToLab(colorHex);
            const position = new THREE.Vector3(
                lab.a * 1.2,  // Scale a* component
                lab.L,         // L* as height
                lab.b * 1.2   // Scale b* component
            );
            
            // Load thumbnail
            const loadPromise = loadThumbnail(result, position, index)
                .then(() => {
                    loadedCount++;
                    if (loadedCount % 50 === 0) {
                        console.log(`[Thumbnails 3D] Loaded ${loadedCount}/${results.length} thumbnails...`);
                    }
                })
                .catch(() => {
                    // Skip failed loads
                });
            
            loadPromises.push(loadPromise);
        });
        
        // Wait for all thumbnails to load
        await Promise.allSettled(loadPromises);
        console.log(`[Thumbnails 3D] Successfully loaded ${thumbnails.length} thumbnails`);
    }

    /**
     * Update billboard orientation (make thumbnails face camera)
     */
    function updateBillboards() {
        if (!camera || !scene || thumbnails.length === 0) return;
        
        thumbnails.forEach(thumb => {
            if (thumb.userData && thumb.userData.isBillboard) {
                // Make plane face camera (billboard effect)
                // Calculate direction from plane to camera
                const direction = new THREE.Vector3()
                    .subVectors(camera.position, thumb.position)
                    .normalize();
                
                // Set plane's rotation to face camera
                thumb.lookAt(thumb.position.clone().add(direction));
            }
        });
    }

    /**
     * Animation loop
     */
    function animate() {
        if (!window.current3DVisualization.isPaused) {
            window.current3DVisualization.animationId = requestAnimationFrame(animate);
            if (controls) controls.update();
            
            // Update billboard orientation
            updateBillboards();
            
            if (renderer && scene && camera) {
                renderer.render(scene, camera);
            }
        }
    }

    /**
     * Main function
     */
    window.generateImageThumbnails3D = async function() {
        const limit = parseInt(document.getElementById('thumbnailLimit')?.value) || 300;
        
        console.log('[Thumbnails 3D] ===== Starting generation =====');
        console.log(`[Thumbnails 3D] Limit: ${limit} images`);
        
        try {
            if (!initScene()) {
                throw new Error('Failed to initialize scene');
            }

            console.log('[Thumbnails 3D] Fetching image data...');
            const results = await fetchImageData(limit);
            currentData = results;
            
            if (results.length === 0) {
                window.hide3DLoading();
                alert('No images found');
                return;
            }

            console.log(`[Thumbnails 3D] Creating thumbnails for ${results.length} images...`);
            await createThumbnails(results);
            
            window.hide3DLoading();

            const enableAnimations = document.getElementById('enable3DAnimations')?.checked !== false;
            if (enableAnimations) {
                console.log('[Thumbnails 3D] Starting animation loop...');
                animate();
            } else {
                updateBillboards();
                if (controls) controls.update();
                if (renderer && scene && camera) {
                    renderer.render(scene, camera);
                }
            }

            console.log(`[Thumbnails 3D] ===== Generation complete with ${thumbnails.length} thumbnails =====`);
        } catch (error) {
            console.error('[Thumbnails 3D] Error:', error);
            window.hide3DLoading();
            alert(`Failed to generate thumbnails: ${error.message}`);
        }
    };

    /**
     * Export data
     */
    window.exportThumbnailsData = function() {
        if (!currentData) {
            alert('No data to export. Please generate thumbnails first.');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            thumbnail_count: thumbnails.length,
            results: currentData.map(r => ({
                image_id: r.image_id,
                dominant_colors: r.dominant_colors,
                image_url: r.image_url
            }))
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `image_thumbnails_3d_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);

        console.log('Thumbnails data exported');
    };

})();
