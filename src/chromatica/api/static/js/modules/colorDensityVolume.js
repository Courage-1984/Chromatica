(function() {
    'use strict';

    let scene, camera, renderer, controls;
    let voxels = [];
    let currentAggregate = null;

    async function fetchSample(limit=300){
        // Prefer most recent real search results if available
        if (Array.isArray(window.lastSearchResults) && window.lastSearchResults.length) {
            return { results: window.lastSearchResults };
        }
        // Try with EMD (fast_mode=false) to encourage rich payloads
        let res = await fetch(`/search?colors=808080&weights=1.0&k=${limit}&fast_mode=false`);
        if (res.ok) {
            const data = await res.json();
            if (Array.isArray(data.results) && data.results.some(r => Array.isArray(r.dominant_colors) && r.dominant_colors.length)) {
                return data;
            }
        }
        // Second attempt with a different neutral
        res = await fetch(`/search?colors=7F7F7F&weights=1.0&k=${limit}&fast_mode=false`);
        if (res.ok) {
            return res.json();
        }
        throw new Error('Failed to fetch sample');
    }

    async function ensureDominantColors(results, maxEstimate=50) {
        // If results lack dominant_colors, estimate a simple dominant color client-side
        const lacking = results.filter(r => !Array.isArray(r.dominant_colors) || r.dominant_colors.length === 0).slice(0, maxEstimate);
        console.log(`[Density Volume] Found ${lacking.length} results lacking dominant_colors`);
        if (!lacking.length) return results;
        let estimated = 0;
        const estimateColor = async (r) => {
            try {
                const src = r.image_url || `/image/${encodeURIComponent(r.image_id)}`;
                const img = await loadImage(src);
                const hex = averageImageHex(img, 16);
                // API expects List[str], so use string format
                r.dominant_colors = [hex];
                estimated++;
            } catch (e) {
                console.warn(`[Density Volume] Failed to estimate color for ${r.image_id}:`, e);
            }
            return r;
        };
        await Promise.all(lacking.map(estimateColor));
        console.log(`[Density Volume] Estimated dominant colors for ${estimated} images`);
        return results;
    }

    function loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = src;
        });
    }

    function averageImageHex(img, step=16) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const w = canvas.width = Math.min(256, img.naturalWidth || img.width);
        const h = canvas.height = Math.min(256, img.naturalHeight || img.height);
        ctx.drawImage(img, 0, 0, w, h);
        const data = ctx.getImageData(0, 0, w, h).data;
        let r=0,g=0,b=0,c=0;
        const sx = Math.max(1, Math.floor(w/step));
        const sy = Math.max(1, Math.floor(h/step));
        for (let y=0;y<h;y+=sy){
            for (let x=0;x<w;x+=sx){
                const i = (y*w + x)*4; r+=data[i]; g+=data[i+1]; b+=data[i+2]; c++;
            }
        }
        r=Math.round(r/Math.max(1,c)); g=Math.round(g/Math.max(1,c)); b=Math.round(b/Math.max(1,c));
        const toHex = v => v.toString(16).toUpperCase().padStart(2,'0');
        return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
    }

    function initScene(){
        console.log('[Density Volume] Initializing scene...');
        window.clear3DVisualization(); window.show3DLoading();
        const container = window.get3DContainer(); if(!container) return false;
        const existingCanvas = container.querySelector('canvas'); if(existingCanvas) existingCanvas.remove();

        scene = new THREE.Scene(); scene.background = new THREE.Color(0x1e1e2e);
        camera = new THREE.PerspectiveCamera(75, container.clientWidth/container.clientHeight, 0.1, 3000);
        camera.position.set(160, 160, 160); camera.lookAt(0,0,0);
        renderer = new THREE.WebGLRenderer({antialias:true}); renderer.setSize(container.clientWidth, container.clientHeight); renderer.setPixelRatio(window.devicePixelRatio); container.appendChild(renderer.domElement);
        if(typeof THREE.OrbitControls!=='undefined'){ controls=new THREE.OrbitControls(camera, renderer.domElement); controls.enableDamping=true; controls.dampingFactor=0.05; }
        window.current3DVisualization.scene=scene; window.current3DVisualization.camera=camera; window.current3DVisualization.renderer=renderer; window.current3DVisualization.controls=controls;
        const showAxes = document.getElementById('show3DAxes')?.checked !== false; if(showAxes) scene.add(new THREE.AxesHelper(100));
        scene.add(new THREE.GridHelper(200,20,0x45475a,0x313244));
        scene.add(new THREE.AmbientLight(0xffffff,0.6)); const dir=new THREE.DirectionalLight(0xffffff,0.8); dir.position.set(50,50,50); scene.add(dir);
        return true;
    }

    function hexToLab(hex){hex=hex.replace('#','');const r=parseInt(hex.substring(0,2),16)/255,g=parseInt(hex.substring(2,4),16)/255,b=parseInt(hex.substring(4,6),16)/255;const[X,Y,Z]=rgbToXyz(r,g,b);const[L,a,b_]=xyzToLab(X,Y,Z);return{L,a,b:b_}}
    function rgbToXyz(r,g,b){r=r>0.04045?Math.pow((r+0.055)/1.055,2.4):r/12.92;g=g>0.04045?Math.pow((g+0.055)/1.055,2.4):g/12.92;b=b>0.04045?Math.pow((b+0.055)/1.055,2.4):b/12.92;return[r*0.4124564+g*0.3575761+b*0.1804375,r*0.2126729+g*0.7151522+b*0.0721750,r*0.0193339+g*0.1191920+b*0.9503041]}
    function xyzToLab(X,Y,Z){const Xn=0.95047,Yn=1,Zn=1.08883;const fx=f(X/Xn),fy=f(Y/Yn),fz=f(Z/Zn);return[116*fy-16,500*(fx-fy),200*(fy-fz)]}
    function f(t){const d=6/29;return t>d*d*d?Math.pow(t,1/3):t/(3*d*d)+4/29}
    // Convert Lab back to RGB for display
    function labToXyz(L,a,b){const fy=(L+16)/116;const fx=a/500+fy;const fz=fy-b/200;const Xn=0.95047,Yn=1,Zn=1.08883;const x=xnInv(fx)*Xn;const y=xnInv(fy)*Yn;const z=xnInv(fz)*Zn;return[x,y,z]}
    function xnInv(t){const d=6/29;if(t>d)return Math.pow(t,3);return 3*d*d*(t-4/29)}
    function xyzToRgb(X,Y,Z){let r=X*3.2404542+Y*-1.5371385+Z*-0.4985314;let g=X*-0.9692660+Y*1.8760108+Z*0.0415560;let b=X*0.0556434+Y*-0.2040259+Z*1.0572252;r=r>0.0031308?1.055*Math.pow(r,1/2.4)-0.055:r*12.92;g=g>0.0031308?1.055*Math.pow(g,1/2.4)-0.055:g*12.92;b=b>0.0031308?1.055*Math.pow(b,1/2.4)-0.055:b*12.92;return[Math.max(0,Math.min(1,r)),Math.max(0,Math.min(1,g)),Math.max(0,Math.min(1,b))]}

    function buildDensity(results){
        voxels.forEach(v=>scene.remove(v)); voxels=[];
        // Aggregate dominant colors into coarse bins to approximate volume
        const bins = new Map();
        const Q = 12; // coarse quantization per axis
        let totalColors = 0;
        let processedColors = 0;
        results.forEach((r, idx)=>{
            const cols = Array.isArray(r.dominant_colors)? r.dominant_colors: [];
            totalColors += cols.length;
            // Normalize: handle both string format ["#FF0000"] and object format [{hex:"#FF0000",p:0.5}]
            cols.forEach((c, cIdx)=>{
                let hex = null;
                let weight = 0.3; // default weight
                // Handle string format: "#FF0000" or "FF0000"
                if (typeof c === 'string') {
                    hex = c.startsWith('#') ? c : '#' + c;
                    weight = 1.0 / cols.length; // equal weight if strings
                }
                // Handle object format: {hex: "#FF0000", p: 0.5}
                else if (c && typeof c === 'object') {
                    hex = c.hex || (c.color ? (c.color.startsWith('#') ? c.color : '#' + c.color) : null);
                    weight = typeof c.p === 'number' ? c.p : (typeof c.weight === 'number' ? c.weight : 0.3);
                }
                if (!hex) {
                    console.warn(`[Density Volume] Result ${idx}, color ${cIdx} has no valid hex:`, c);
                    return;
                }
                // Normalize hex format
                hex = hex.replace('#', '').toUpperCase();
                if (hex.length !== 6) {
                    console.warn(`[Density Volume] Invalid hex length for "${hex}"`);
                    return;
                }
                const hexFull = '#' + hex;
                const lab = hexToLab(hexFull);
                // Normalize to index 0..Q-1
                const iL = Math.max(0, Math.min(Q-1, Math.floor((lab.L/100)*(Q))));
                const ia = Math.max(0, Math.min(Q-1, Math.floor(((lab.a+128)/256)*(Q))));
                const ib = Math.max(0, Math.min(Q-1, Math.floor(((lab.b+128)/256)*(Q))));
                const key = `${iL},${ia},${ib}`;
                bins.set(key, (bins.get(key)||0) + weight);
                processedColors++;
            });
        });
        console.log(`[Density Volume] Processed ${processedColors} colors from ${totalColors} total across ${results.length} results`);
        currentAggregate = { Q, bins: Array.from(bins.entries()) };
        if (currentAggregate.bins.length === 0) {
            console.warn('[Density Volume] No bins aggregated. Debug info:');
            console.warn('- Results count:', results.length);
            if (results.length > 0) {
                console.warn('- First result:', results[0]);
                console.warn('- First result dominant_colors:', results[0].dominant_colors);
            }
        }
        // Render voxels - color each by its actual Lab position
        bins.forEach((val, key)=>{
            const [iL, ia, ib] = key.split(',').map(Number);
            // Convert back to Lab coordinates for this bin center
            const L = (iL / (Q-1)) * 100;
            const a = (ia / (Q-1)) * 256 - 128;
            const b = (ib / (Q-1)) * 256 - 128;
            // Position in 3D space
            const x = (ia/(Q-1))*256 - 128;
            const y = (iL/(Q-1))*100;
            const z = (ib/(Q-1))*256 - 128;
            // Size based on density
            const size = 2 + Math.min(12, Math.max(0.5, val*15));
            // Convert Lab to RGB for voxel color
            const [X,Y,Z] = labToXyz(L, a, b);
            const [r, g, b_rgb] = xyzToRgb(X, Y, Z);
            const rgbHex = ((Math.round(r*255)<<16)|(Math.round(g*255)<<8)|Math.round(b_rgb*255));
            // Opacity based on density
            const opacity = Math.min(0.9, Math.max(0.3, 0.3 + val*0.6));
            const geo = new THREE.BoxGeometry(size, size, size);
            const mat = new THREE.MeshPhongMaterial({ 
                color: rgbHex, 
                transparent: true, 
                opacity: opacity,
                emissive: rgbHex,
                emissiveIntensity: 0.2
            });
            const cube = new THREE.Mesh(geo, mat); 
            cube.position.set(x,y,z);
            scene.add(cube); 
            voxels.push(cube);
        });
        if (!voxels.length) {
            const msg = document.createElement('div');
            msg.textContent = 'No density found for sample. Try a different sample or ensure dominant_colors are available.';
            msg.style.cssText = 'position:absolute;top:10px;left:10px;color:var(--yellow);font-size:12px;';
            const c = window.get3DContainer(); if (c) c.appendChild(msg);
        }
    }

    function animate(){ if(!window.current3DVisualization.isPaused){ window.current3DVisualization.animationId=requestAnimationFrame(animate); if(controls) controls.update(); if(renderer&&scene&&camera) renderer.render(scene,camera);} }

    window.generateColorDensityVolume = async function(){
        console.log('[Density Volume] ===== Starting generation =====');
        try{ if(!initScene()) throw new Error('Failed to initialize scene');
            const data = await fetchSample(300); let results = data.results||[];
            console.log(`[Density Volume] Fetched ${results.length} results`);
            if (results.length > 0) {
                console.log(`[Density Volume] First result sample:`, {
                    image_id: results[0].image_id,
                    dominant_colors: results[0].dominant_colors,
                    dominant_colors_type: typeof results[0].dominant_colors,
                    dominant_colors_length: Array.isArray(results[0].dominant_colors) ? results[0].dominant_colors.length : 'not array'
                });
            }
            // Ensure we have dominant_colors; estimate if missing
            results = await ensureDominantColors(results, 60);
            if(!results.length){ window.hide3DLoading(); alert('No data'); return; }
            console.log(`[Density Volume] Aggregating ${results.length} results`);
            buildDensity(results);
            window.hide3DLoading(); const enable = document.getElementById('enable3DAnimations')?.checked !== false; if(enable) animate(); else renderer.render(scene,camera);
            console.log(`[Density Volume] Rendered ${voxels.length} voxels`);
            console.log('[Density Volume] ===== Generation complete =====');
        }catch(e){ console.error('[Density Volume] Error:', e); window.hide3DLoading(); alert(e.message); }
    };

    window.exportColorDensityData = function(){ if(!currentAggregate){ alert('No data to export'); return;} const blob=new Blob([JSON.stringify(currentAggregate,null,2)],{type:'application/json'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=`color_density_${Date.now()}.json`; a.click(); URL.revokeObjectURL(url); };
})();


