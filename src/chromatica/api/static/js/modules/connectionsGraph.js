/**
 * Connections Graph - Visualize image similarity as a 3D graph
 * 
 * Creates a graph where nodes are images and edges represent similarity.
 * Uses search results to build connections between similar images.
 */

(function() {
    'use strict';

    let scene, camera, renderer, controls;
    let nodes = [];
    let edges = [];
    let currentGraph = null;

    /**
     * Fetch image data and build similarity graph
     */
    async function buildSimilarityGraph(limit = 50) {
        try {
            // Get diverse sample of images
            const response = await fetch(`/search?colors=808080&weights=1.0&k=${limit}&fast_mode=true`);
            if (!response.ok) throw new Error('Failed to fetch image data');
            
            const data = await response.json();
            const results = data.results || [];

            // Build graph: nodes are images, edges connect similar ones
            const graph = {
                nodes: results.map((result, index) => ({
                    id: result.image_id,
                    result: result,
                    index: index
                })),
                edges: []
            };

            // Create edges between images with similar dominant colors
            for (let i = 0; i < graph.nodes.length; i++) {
                for (let j = i + 1; j < Math.min(i + 10, graph.nodes.length); j++) {
                    const node1 = graph.nodes[i];
                    const node2 = graph.nodes[j];

                    // Calculate similarity based on color distance
                    const similarity = calculateColorSimilarity(
                        node1.result.dominant_colors,
                        node2.result.dominant_colors
                    );

                    // Add edge if similar enough
                    if (similarity > 0.3) {
                        graph.edges.push({
                            source: i,
                            target: j,
                            similarity: similarity
                        });
                    }
                }
            }

            return graph;
        } catch (error) {
            console.error('Error building similarity graph:', error);
            return { nodes: [], edges: [] };
        }
    }

    /**
     * Calculate color similarity between two sets of dominant colors
     */
    function calculateColorSimilarity(colors1, colors2) {
        if (!colors1 || !colors2 || colors1.length === 0 || colors2.length === 0) {
            return 0;
        }

        let totalSimilarity = 0;
        let comparisons = 0;

        colors1.forEach(c1 => {
            const hex1 = c1.hex || c1;
            colors2.forEach(c2 => {
                const hex2 = c2.hex || c2;
                const similarity = hexColorSimilarity(hex1, hex2);
                totalSimilarity += similarity;
                comparisons++;
            });
        });

        return comparisons > 0 ? totalSimilarity / comparisons : 0;
    }

    /**
     * Simple hex color similarity (0-1)
     */
    function hexColorSimilarity(hex1, hex2) {
        const rgb1 = hexToRgb(hex1);
        const rgb2 = hexToRgb(hex2);

        const distance = Math.sqrt(
            Math.pow(rgb1.r - rgb2.r, 2) +
            Math.pow(rgb1.g - rgb2.g, 2) +
            Math.pow(rgb1.b - rgb2.b, 2)
        );

        // Normalize to 0-1 (max distance is ~441 for RGB)
        return 1 - (distance / 441);
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
     * Initialize Three.js scene
     */
    function initScene() {
        console.log('[Connections Graph] Initializing scene...');
        
        window.clear3DVisualization();
        window.show3DLoading();
        
        const container = window.get3DContainer();
        if (!container) {
            console.error('[Connections Graph] Container not found');
            return false;
        }
        
        const existingCanvas = container.querySelector('canvas');
        if (existingCanvas) existingCanvas.remove();
        
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1e1e2e);

        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(100, 100, 100);
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
            console.log('[Connections Graph] Axes enabled');
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
     * Position nodes using force-directed layout simulation
     */
    function positionNodes(graph) {
        const nodeCount = graph.nodes.length;
        const positions = [];

        // Simple spring layout: distribute nodes in 3D space
        for (let i = 0; i < nodeCount; i++) {
            const angle = (i / nodeCount) * Math.PI * 2;
            const height = (i % 10) * 10 - 45;
            const radius = 30 + (i % 5) * 10;
            
            positions.push({
                x: Math.cos(angle) * radius,
                y: height,
                z: Math.sin(angle) * radius
            });
        }

        return positions;
    }

    /**
     * Create graph visualization
     */
    function createGraph(graph) {
        // Clear existing
        nodes.forEach(node => scene.remove(node));
        edges.forEach(edge => scene.remove(edge));
        nodes = [];
        edges = [];

        if (!graph || graph.nodes.length === 0) return;

        const positions = positionNodes(graph);

        // Create nodes (spheres)
        graph.nodes.forEach((node, index) => {
            const pos = positions[index];
            const result = node.result;

            let colorHex = '#808080';
            if (result.dominant_colors && result.dominant_colors.length > 0) {
                colorHex = result.dominant_colors[0].hex || result.dominant_colors[0];
            }

            const rgb = hexToRgb(colorHex);
            const geometry = new THREE.SphereGeometry(3, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(rgb.r / 255, rgb.g / 255, rgb.b / 255),
                emissive: new THREE.Color(rgb.r / 510, rgb.g / 510, rgb.b / 510)
            });

            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(pos.x, pos.y, pos.z);
            sphere.userData = { node, position: pos };

            scene.add(sphere);
            nodes.push(sphere);
        });

        // Create edges (lines)
        graph.edges.forEach(edge => {
            const sourceNode = nodes[edge.source];
            const targetNode = nodes[edge.target];
            
            if (!sourceNode || !targetNode) return;

            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(
                    sourceNode.position.x,
                    sourceNode.position.y,
                    sourceNode.position.z
                ),
                new THREE.Vector3(
                    targetNode.position.x,
                    targetNode.position.y,
                    targetNode.position.z
                )
            ]);

            const material = new THREE.LineBasicMaterial({
                color: 0x89b4fa,
                transparent: true,
                opacity: edge.similarity * 0.5
            });

            const line = new THREE.Line(geometry, material);
            line.userData = { edge };

            scene.add(line);
            edges.push(line);
        });
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
    window.generateConnectionsGraph = async function() {
        console.log('[Connections Graph] ===== Starting generation =====');
        
        try {
            if (!initScene()) {
                throw new Error('Failed to initialize scene');
            }

            console.log('[Connections Graph] Building similarity graph...');
            const graph = await buildSimilarityGraph(50);
            currentGraph = graph;
            console.log(`[Connections Graph] Built graph with ${graph.nodes.length} nodes, ${graph.edges.length} edges`);

            if (graph.nodes.length === 0) {
                window.hide3DLoading();
                alert('No data available to build graph.');
                return;
            }

            console.log('[Connections Graph] Creating graph visualization...');
            createGraph(graph);
            console.log(`[Connections Graph] Created ${nodes.length} nodes and ${edges.length} edges`);
            
            window.hide3DLoading();

            const enableAnimations = document.getElementById('enable3DAnimations')?.checked !== false;
            if (enableAnimations) {
                console.log('[Connections Graph] Starting animation loop...');
                animate();
            } else {
                console.log('[Connections Graph] Animations disabled, rendering static view');
                if (controls) controls.update();
                if (renderer && scene && camera) {
                    renderer.render(scene, camera);
                }
            }

            console.log(`[Connections Graph] ===== Generation complete =====`);
        } catch (error) {
            console.error('[Connections Graph] Error:', error);
            window.hide3DLoading();
            alert(`Failed to generate Connections Graph: ${error.message}`);
        }
    };

    /**
     * Export data
     */
    window.exportGraphData = function() {
        if (!currentGraph || currentGraph.nodes.length === 0) {
            alert('No graph data to export. Please generate the graph first.');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            nodes_count: currentGraph.nodes.length,
            edges_count: currentGraph.edges.length,
            nodes: currentGraph.nodes.map(n => ({
                id: n.id,
                image_id: n.result.image_id,
                dominant_colors: n.result.dominant_colors
            })),
            edges: currentGraph.edges.map(e => ({
                source: e.source,
                target: e.target,
                similarity: e.similarity
            }))
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `connections_graph_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);

        console.log('Connections Graph data exported');
    };

})();
