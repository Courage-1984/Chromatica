/**
 * Image Globe - Map images onto a 3D sphere positioned by dominant colors
 * 
 * Fetches image data and positions them on a sphere based on their color properties.
 */

(function() {
    'use strict';

    let scene, camera, renderer, controls;
    let globe = null;
    let imageObjects = [];
    let currentData = null;

    /**
     * Fetch image data
     */
    async function fetchImageData(limit = 200) {
        try {
            const response = await fetch(`/search?colors=808080&weights=1.0&k=${limit}&fast_mode=true`);
            if (!response.ok) throw new Error('Failed to fetch image data');
            
            const data = await response.json();
            return data.results || [];
        } catch (error) {
            console.error('Error fetching image data:', error);
            return [];
        }
    }

    /**
     * Convert color to spherical coordinates
     */
    function colorToSpherical(hex) {
        const lab = hexToLab(hex);
        
        // Map Lab to spherical coordinates
        // L* (0-100) -> theta (0 to PI)
        // a* and b* -> phi (0 to 2*PI) based on hue
        const theta = (lab.L / 100) * Math.PI; // Latitude
        const hue = Math.atan2(lab.b, lab.a);
        const phi = hue + Math.PI; // Longitude (normalized to 0-2π)
        
        return { theta, phi, lab };
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

    /**
     * Initialize Three.js scene
     */
    function initScene() {
        console.log('[Image Globe] Initializing scene...');
        
        window.clear3DVisualization();
        window.show3DLoading();
        
        const container = window.get3DContainer();
        if (!container) {
            console.error('[Image Globe] Container not found');
            return false;
        }
        
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) existingCanvas.remove();
        
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1e1e2e);

        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 2000);
        camera.position.set(0, 0, 150);
        camera.lookAt(0, 0, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        if (typeof THREE.OrbitControls !== 'undefined') {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.enableZoom = true;
            controls.enablePan = false;
        }

        window.current3DVisualization.scene = scene;
        window.current3DVisualization.camera = camera;
        window.current3DVisualization.renderer = renderer;
        window.current3DVisualization.controls = controls;

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        // Add point light
        const pointLight = new THREE.PointLight(0xffffff, 1);
        pointLight.position.set(100, 100, 100);
        scene.add(pointLight);
        
        return true;
    }

    /**
     * Create sphere with image points
     */
    function createGlobe(imageData) {
        // Clear existing
        if (globe) scene.remove(globe);
        imageObjects.forEach(obj => scene.remove(obj));
        imageObjects = [];

        // Create base sphere
        const sphereGeometry = new THREE.SphereGeometry(50, 32, 32);
        const sphereMaterial = new THREE.MeshPhongMaterial({
            color: 0x313244,
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });
        globe = new THREE.Mesh(sphereGeometry, sphereMaterial);
        scene.add(globe);

        // Position images on sphere
        imageData.forEach((result) => {
            let colorHex = '#808080';
            if (result.dominant_colors && result.dominant_colors.length > 0) {
                colorHex = result.dominant_colors[0].hex || result.dominant_colors[0];
            }

            const spherical = colorToSpherical(colorHex);
            const radius = 52; // Slightly outside the sphere

            // Convert spherical to Cartesian
            const x = radius * Math.sin(spherical.theta) * Math.cos(spherical.phi);
            const y = radius * Math.cos(spherical.theta);
            const z = radius * Math.sin(spherical.theta) * Math.sin(spherical.phi);

            const rgb = hexToRgb(colorHex);
            
            // Create a small sphere to represent the image
            const geometry = new THREE.SphereGeometry(2, 8, 8);
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(rgb.r / 255, rgb.g / 255, rgb.b / 255),
                emissive: new THREE.Color(rgb.r / 510, rgb.g / 510, rgb.b / 510)
            });

            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(x, y, z);
            sphere.userData = { result, spherical };

            scene.add(sphere);
            imageObjects.push(sphere);
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
            
            // Rotate globe slowly
            if (globe) {
                globe.rotation.y += 0.001;
            }
            
            if (renderer && scene && camera) {
                renderer.render(scene, camera);
            }
        }
    }

    /**
     * Main function
     */
    window.generateImageGlobe = async function() {
        console.log('[Image Globe] ===== Starting generation =====');
        
        try {
            if (!initScene()) {
                throw new Error('Failed to initialize scene');
            }

            console.log('[Image Globe] Fetching image data...');
            const imageData = await fetchImageData(200);
            currentData = imageData;
            console.log(`[Image Globe] Fetched ${imageData.length} images`);

            if (imageData.length === 0) {
                window.hide3DLoading();
                alert('No image data available.');
                return;
            }

            console.log('[Image Globe] Creating globe visualization...');
            createGlobe(imageData);
            console.log(`[Image Globe] Created globe with ${imageObjects.length} image points`);
            
            window.hide3DLoading();

            const enableAnimations = document.getElementById('enable3DAnimations')?.checked !== false;
            if (enableAnimations) {
                console.log('[Image Globe] Starting animation loop...');
                animate();
            } else {
                console.log('[Image Globe] Animations disabled, rendering static view');
                if (controls) controls.update();
                if (renderer && scene && camera) {
                    renderer.render(scene, camera);
                }
            }

            console.log(`[Image Globe] ===== Generation complete with ${imageData.length} images =====`);
        } catch (error) {
            console.error('[Image Globe] Error:', error);
            window.hide3DLoading();
            alert(`Failed to generate Image Globe: ${error.message}`);
        }
    };

    /**
     * Export data
     */
    window.exportGlobeData = function() {
        if (!currentData || currentData.length === 0) {
            alert('No data to export. Please generate the globe first.');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            total_images: currentData.length,
            images: currentData.map(img => ({
                image_id: img.image_id,
                image_url: img.image_url,
                dominant_colors: img.dominant_colors
            }))
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `image_globe_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);

        console.log('Image Globe data exported');
    };

})();
