(function() {
    'use strict';

    let scene, camera, renderer, controls;
    let arrows = [];
    let currentData = null;

    async function performSearch(colors, weights) {
        const params = new URLSearchParams({ colors, weights, k: '20', fast_mode: 'false' });
        const res = await fetch(`/search?${params.toString()}`);
        if (!res.ok) throw new Error('Search failed');
        return res.json();
    }

    function initScene() {
        console.log('[OT Transport 3D] Initializing scene...');
        window.clear3DVisualization();
        window.show3DLoading();
        const container = window.get3DContainer();
        if (!container) return false;
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) existingCanvas.remove();

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1e1e2e);

        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 2000);
        camera.position.set(130, 130, 130);
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
        if (showAxes) scene.add(new THREE.AxesHelper(100));

        const gridHelper = new THREE.GridHelper(200, 20, 0x45475a, 0x313244);
        scene.add(gridHelper);

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dir = new THREE.DirectionalLight(0xffffff, 0.8);
        dir.position.set(50, 50, 50);
        scene.add(dir);
        return true;
    }

    function hexToRgb(hex) {
        hex = hex.replace('#','');
        return { r: parseInt(hex.slice(0,2),16), g: parseInt(hex.slice(2,4),16), b: parseInt(hex.slice(4,6),16) };
    }

    function hexToLab(hex) {
        hex = hex.replace('#','');
        const r = parseInt(hex.substring(0, 2), 16) / 255;
        const g = parseInt(hex.substring(2, 4), 16) / 255;
        const b = parseInt(hex.substring(4, 6), 16) / 255;
        const [X, Y, Z] = rgbToXyz(r, g, b);
        const [L, a, b_] = xyzToLab(X, Y, Z);
        return { L, a, b: b_ };
    }
    function rgbToXyz(r,g,b){r=r>0.04045?Math.pow((r+0.055)/1.055,2.4):r/12.92;g=g>0.04045?Math.pow((g+0.055)/1.055,2.4):g/12.92;b=b>0.04045?Math.pow((b+0.055)/1.055,2.4):b/12.92;return [r*0.4124564+g*0.3575761+b*0.1804375,r*0.2126729+g*0.7151522+b*0.0721750,r*0.0193339+g*0.1191920+b*0.9503041];}
    function xyzToLab(X,Y,Z){const Xn=0.95047,Yn=1,Zn=1.08883;const fx=f(X/Xn),fy=f(Y/Yn),fz=f(Z/Zn);return [116*fy-16,500*(fx-fy),200*(fy-fz)];}
    function f(t){const d=6/29;return t>d*d*d?Math.pow(t,1/3):t/(3*d*d)+4/29;}

    function createArrows(queryColors, candidate) {
        // Show optimal transport from query colors to candidate's dominant colors
        arrows.forEach(a => scene.remove(a));
        arrows = [];

        const candColors = Array.isArray(candidate.dominant_colors) ? candidate.dominant_colors : [];
        // Take top-5 dominant colors for better visualization
        const topColors = candColors.slice(0, 5);

        // Create spheres for query colors
        queryColors.forEach((qHex, qIdx) => {
            const qLab = hexToLab(qHex);
            // Scale Lab space: a* and b* range from -128 to +127, L* from 0-100
            // Normalize to visible 3D space: scale a* and b* by ~1.5, L* by 1
            const qPos = new THREE.Vector3(qLab.a * 1.2, qLab.L, qLab.b * 1.2);
            const qRgb = hexToRgb(qHex);
            
            // Create a colored sphere at query position
            const qGeo = new THREE.SphereGeometry(3, 16, 16);
            const qMat = new THREE.MeshPhongMaterial({ 
                color: new THREE.Color(qRgb.r/255, qRgb.g/255, qRgb.b/255),
                emissive: new THREE.Color(qRgb.r/255, qRgb.g/255, qRgb.b/255),
                emissiveIntensity: 0.3
            });
            const qSphere = new THREE.Mesh(qGeo, qMat);
            qSphere.position.copy(qPos);
            scene.add(qSphere);
            arrows.push(qSphere); // Track for cleanup
        });

        // Create transport arrows from query colors to candidate colors
        queryColors.forEach((qHex, qIdx) => {
            const qLab = hexToLab(qHex);
            const qPos = new THREE.Vector3(qLab.a * 1.2, qLab.L, qLab.b * 1.2);
            const qRgb = hexToRgb(qHex);
            const qColor = new THREE.Color(qRgb.r/255, qRgb.g/255, qRgb.b/255);

            topColors.forEach((tc, cIdx) => {
                // Handle both string format ["#FF0000"] and object format [{hex:"#FF0000",p:0.5}]
                let tHex = null;
                let mass = 1.0 / topColors.length; // default equal distribution
                
                if (typeof tc === 'string') {
                    tHex = tc.startsWith('#') ? tc : '#' + tc;
                } else if (tc && typeof tc === 'object') {
                    tHex = tc.hex || (tc.color ? (tc.color.startsWith('#') ? tc.color : '#' + tc.color) : null);
                    mass = typeof tc.p === 'number' ? tc.p : (typeof tc.weight === 'number' ? tc.weight : mass);
                }
                
                if (!tHex) return;
                
                const cLab = hexToLab(tHex);
                const cPos = new THREE.Vector3(cLab.a * 1.2, cLab.L, cLab.b * 1.2);
                const cRgb = hexToRgb(tHex);

                // Create a colored sphere at candidate position
                const cGeo = new THREE.SphereGeometry(2, 16, 16);
                const cMat = new THREE.MeshPhongMaterial({ 
                    color: new THREE.Color(cRgb.r/255, cRgb.g/255, cRgb.b/255),
                    emissive: new THREE.Color(cRgb.r/255, cRgb.g/255, cRgb.b/255),
                    emissiveIntensity: 0.2
                });
                const cSphere = new THREE.Mesh(cGeo, cMat);
                cSphere.position.copy(cPos);
                scene.add(cSphere);
                arrows.push(cSphere);

                // Create transport arrow - thickness represents mass
                const dir = new THREE.Vector3().subVectors(cPos, qPos);
                const len = dir.length();
                if (len < 0.1) return; // Skip if too close
                dir.normalize();

                // Create arrow - thickness/opacity represents transported mass
                const arrowOpacity = 0.5 + mass * 0.5;
                const arrowColor = qColor.clone();
                arrowColor.multiplyScalar(0.8); // Slightly dimmer for better visibility
                
                // Use ArrowHelper with adjusted parameters
                const arrow = new THREE.ArrowHelper(
                    dir,
                    qPos,
                    len,
                    arrowColor,
                    len * 0.15, // head length
                    Math.max(0.5, mass * 2) // head width (thickness)
                );
                
                // Adjust arrow line thickness by scaling
                if (arrow.line) {
                    arrow.line.scale.set(1, 1, Math.max(0.3, mass * 1.5));
                }
                
                // Set material opacity
                if (arrow.cone) {
                    arrow.cone.material = arrow.cone.material.clone();
                    arrow.cone.material.transparent = true;
                    arrow.cone.material.opacity = arrowOpacity;
                }
                if (arrow.line) {
                    arrow.line.material = arrow.line.material.clone();
                    arrow.line.material.transparent = true;
                    arrow.line.material.opacity = arrowOpacity;
                }
                
                scene.add(arrow);
                arrows.push(arrow);
            });
        });
        
        console.log(`[OT Transport 3D] Created ${queryColors.length} query spheres and ${topColors.length} candidate spheres with transport arrows`);
    }

    function animate() {
        if (!window.current3DVisualization.isPaused) {
            window.current3DVisualization.animationId = requestAnimationFrame(animate);
            if (controls) controls.update();
            if (renderer && scene && camera) renderer.render(scene, camera);
        }
    }

    window.generateOTTransport3D = async function() {
        const colors = document.getElementById('otColors')?.value;
        const weights = document.getElementById('otWeights')?.value;
        if (!colors || !weights) { alert('Please enter colors and weights'); return; }
        console.log('[OT Transport 3D] ===== Starting generation =====');
        try {
            if (!initScene()) throw new Error('Failed to initialize scene');
            console.log('[OT Transport 3D] Performing search...');
            const data = await performSearch(colors, weights);
            currentData = data;
            if (!data.results?.length) { window.hide3DLoading(); alert('No results'); return; }

            const qColors = colors.split(',').map(c => '#' + c.trim());
            console.log('[OT Transport 3D] Creating transport arrows to top result...');
            createArrows(qColors, data.results[0]);
            window.hide3DLoading();

            const enable = document.getElementById('enable3DAnimations')?.checked !== false;
            if (enable) animate(); else renderer.render(scene, camera);
            console.log('[OT Transport 3D] ===== Generation complete =====');
        } catch (e) {
            console.error('[OT Transport 3D] Error:', e); window.hide3DLoading(); alert(e.message);
        }
    };

    window.exportOTTransportData = function() {
        if (!currentData) { alert('No data to export'); return; }
        const blob = new Blob([JSON.stringify(currentData, null, 2)], {type:'application/json'});
        const url = URL.createObjectURL(blob); const a = document.createElement('a');
        a.href = url; a.download = `ot_transport_${Date.now()}.json`; a.click(); URL.revokeObjectURL(url);
    };
})();


