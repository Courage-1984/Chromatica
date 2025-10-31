(function() {
    'use strict';

    let scene, camera, renderer, controls;
    let nodes = [], links = [];
    let currentResults = null;

    async function performSearch(colors, weights) {
        // Request more results for better graph visualization
        const params = new URLSearchParams({ colors, weights, k: '50', fast_mode: 'false' });
        const res = await fetch(`/search?${params.toString()}`);
        if (!res.ok) throw new Error('Search failed');
        const data = await res.json();
        console.log(`[HNSW Explorer] Search returned ${data.results?.length || 0} results`);
        return data;
    }

    function initScene() {
        console.log('[HNSW Explorer] Initializing scene...');
        window.clear3DVisualization();
        window.show3DLoading();
        const container = window.get3DContainer();
        if (!container) return false;
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) existingCanvas.remove();

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1e1e2e);

        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 2000);
        camera.position.set(140, 140, 140);
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

    function hexToRgb(hex){hex=hex.replace('#','');return{r:parseInt(hex.slice(0,2),16),g:parseInt(hex.slice(2,4),16),b:parseInt(hex.slice(4,6),16)}}
    function hexToLab(hex){hex=hex.replace('#','');const r=parseInt(hex.substring(0,2),16)/255,g=parseInt(hex.substring(2,4),16)/255,b=parseInt(hex.substring(4,6),16)/255;const[X,Y,Z]=rgbToXyz(r,g,b);const[L,a,b_]=xyzToLab(X,Y,Z);return{L,a,b:b_}}
    function rgbToXyz(r,g,b){r=r>0.04045?Math.pow((r+0.055)/1.055,2.4):r/12.92;g=g>0.04045?Math.pow((g+0.055)/1.055,2.4):g/12.92;b=b>0.04045?Math.pow((b+0.055)/1.055,2.4):b/12.92;return[r*0.4124564+g*0.3575761+b*0.1804375,r*0.2126729+g*0.7151522+b*0.0721750,r*0.0193339+g*0.1191920+b*0.9503041]}
    function xyzToLab(X,Y,Z){const Xn=0.95047,Yn=1,Zn=1.08883;const fx=f(X/Xn),fy=f(Y/Yn),fz=f(Z/Zn);return[116*fy-16,500*(fx-fy),200*(fy-fz)]}
    function f(t){const d=6/29;return t>d*d*d?Math.pow(t,1/3):t/(3*d*d)+4/29}

    function buildGraph(results) {
        nodes.forEach(n=>scene.remove(n)); links.forEach(l=>scene.remove(l)); nodes=[]; links=[];
        if (!results || results.length === 0) {
            console.warn('[HNSW Explorer] No results to visualize');
            return;
        }
        
        const spheres = [];
        let nodeCount = 0;
        
        // Process up to 50 results for better visualization
        const maxResults = Math.min(50, results.length);
        const resultsToShow = results.slice(0, maxResults);
        
        console.log(`[HNSW Explorer] Creating graph with ${resultsToShow.length} nodes from ${results.length} results`);
        
        resultsToShow.forEach((r, idx) => {
            // Handle both string format ["#FF0000"] and object format [{hex:"#FF0000",p:0.5}]
            let hex = '#808080'; // default gray
            const domColors = Array.isArray(r.dominant_colors) ? r.dominant_colors : [];
            if (domColors.length > 0) {
                const firstColor = domColors[0];
                if (typeof firstColor === 'string') {
                    hex = firstColor.startsWith('#') ? firstColor : '#' + firstColor;
                } else if (firstColor && typeof firstColor === 'object') {
                    hex = firstColor.hex || (firstColor.color ? (firstColor.color.startsWith('#') ? firstColor.color : '#' + firstColor.color) : '#808080');
                }
            }
            
            const lab = hexToLab(hex);
            // Scale Lab space properly: a* and b* range from -128 to +127, L* from 0-100
            // Scale to make it visible in 3D space
            const pos = new THREE.Vector3(lab.a * 1.2, lab.L, lab.b * 1.2);
            const rgb = hexToRgb(hex);
            
            // Create colored sphere
            const geo = new THREE.SphereGeometry(2.5, 12, 12);
            const mat = new THREE.MeshPhongMaterial({ 
                color: new THREE.Color(rgb.r/255, rgb.g/255, rgb.b/255),
                emissive: new THREE.Color(rgb.r/255, rgb.g/255, rgb.b/255),
                emissiveIntensity: 0.3
            });
            const sphere = new THREE.Mesh(geo, mat);
            sphere.position.copy(pos);
            sphere.userData = { result: r, idx, imageId: r.image_id };
            
            scene.add(sphere);
            nodes.push(sphere);
            spheres.push({ pos, idx, result: r });
            nodeCount++;
        });
        
        console.log(`[HNSW Explorer] Created ${nodeCount} nodes`);
        
        // Connect near neighbors based on Lab distance (simulating HNSW graph structure)
        const maxLinks = Math.min(6, Math.max(2, Math.floor(spheres.length / 10))); // Adaptive based on graph size
        const threshold = 50; // Larger threshold to ensure connections
        let linkCount = 0;
        
        spheres.forEach((s, i)=>{
            let neighbors = [];
            // Find all neighbors within threshold
            spheres.forEach((t, j)=>{
                if (i !== j) {
                    const d = s.pos.distanceTo(t.pos);
                    if (d < threshold) {
                        neighbors.push({j, d});
                    }
                }
            });
            // Sort by distance and take top-k nearest
            neighbors.sort((a, b) => a.d - b.d);
            const topNeighbors = neighbors.slice(0, maxLinks);
            
            topNeighbors.forEach(n => {
                const src = s.pos;
                const dst = spheres[n.j].pos;
                const distance = n.d;
                
                // Create edge with opacity based on distance (closer = more opaque)
                const opacity = Math.max(0.2, Math.min(0.7, 1.0 - (distance / threshold)));
                const geo = new THREE.BufferGeometry().setFromPoints([src, dst]);
                const mat = new THREE.LineBasicMaterial({ 
                    color: 0x89b4fa, 
                    transparent: true, 
                    opacity: opacity,
                    linewidth: 1
                });
                const line = new THREE.Line(geo, mat);
                scene.add(line);
                links.push(line);
                linkCount++;
            });
        });
        
        console.log(`[HNSW Explorer] Created ${linkCount} edges connecting ${nodeCount} nodes`);
        
        if (nodeCount === 0) {
            console.error('[HNSW Explorer] No nodes were created - check results data');
        }
        if (linkCount === 0 && nodeCount > 1) {
            console.warn('[HNSW Explorer] No edges created - nodes may be too far apart or threshold too small');
        }
    }

    function animate(){ if(!window.current3DVisualization.isPaused){ window.current3DVisualization.animationId=requestAnimationFrame(animate); if(controls) controls.update(); if(renderer&&scene&&camera) renderer.render(scene,camera);} }

    window.generateHNSWExplorer = async function() {
        const colors = document.getElementById('hnswColors')?.value;
        const weights = document.getElementById('hnswWeights')?.value;
        if (!colors || !weights) { alert('Please enter colors and weights'); return; }
        console.log('[HNSW Explorer] ===== Starting generation =====');
        console.log(`[HNSW Explorer] Query colors: ${colors}, weights: ${weights}`);
        try {
            if (!initScene()) throw new Error('Failed to initialize scene');
            console.log('[HNSW Explorer] Performing search...');
            const data = await performSearch(colors, weights);
            currentResults = data.results || [];
            console.log(`[HNSW Explorer] Got ${currentResults.length} search results`);
            if (!currentResults.length) { 
                window.hide3DLoading(); 
                alert('No results found'); 
                return; 
            }
            // Log sample of first result for debugging
            if (currentResults.length > 0) {
                console.log('[HNSW Explorer] First result sample:', {
                    image_id: currentResults[0].image_id,
                    dominant_colors: currentResults[0].dominant_colors,
                    dominant_colors_type: typeof currentResults[0].dominant_colors,
                    dominant_colors_length: Array.isArray(currentResults[0].dominant_colors) ? currentResults[0].dominant_colors.length : 'not array'
                });
            }
            console.log(`[HNSW Explorer] Building graph with ${currentResults.length} results...`);
            buildGraph(currentResults);
            window.hide3DLoading();
            const enable = document.getElementById('enable3DAnimations')?.checked !== false; 
            if (enable) animate(); 
            else renderer.render(scene, camera);
            console.log(`[HNSW Explorer] Graph rendered: ${nodes.length} nodes, ${links.length} edges`);
            console.log('[HNSW Explorer] ===== Generation complete =====');
        } catch(e){ 
            console.error('[HNSW Explorer] Error:', e); 
            console.error('[HNSW Explorer] Stack:', e.stack);
            window.hide3DLoading(); 
            alert(`Error: ${e.message}`); 
        }
    };

    window.exportHNSWData = function(){ if(!currentResults){ alert('No data to export'); return;} const blob = new Blob([JSON.stringify(currentResults,null,2)],{type:'application/json'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=`hnsw_explorer_${Date.now()}.json`; a.click(); URL.revokeObjectURL(url); };
})();


