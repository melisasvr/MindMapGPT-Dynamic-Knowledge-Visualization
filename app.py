"""
MindMapGPT Flask Web Application
Integrates the Python backend with the web frontend
"""

from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import json
import io
import base64
from typing import Dict, List, Tuple, Optional
import logging
import os
import tempfile
from dataclasses import dataclass, asdict
import random
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Core MindMapGPT Classes (Simplified Mock) ---
# This is a simplified version of the core logic to make the app self-contained.
# In a real-world scenario, this would be in a separate file.

@dataclass
class MindMapNode:
    """Represents a single node in the mind map"""
    id: str
    text: str
    category: str
    importance_score: float
    size: int = 20
    color: str = "#4A90E2"

@dataclass
class MindMapEdge:
    """Represents a connection between nodes"""
    source: str
    target: str
    weight: float
    relationship_type: str = "related"

class MindMapGenerator:
    """Generates a mock mind map structure from text."""
    def generate_mindmap(self, text: str, num_concepts: int = 15) -> Tuple[List[Dict], List[Dict]]:
        """
        Generates a mock mind map with a star-like structure.
        The center node is the first concept, and others branch off of it.
        """
        # Define a list of mock concepts for demonstration
        mock_concepts = [
            "Artificial Intelligence", "Machine Learning", "Deep Learning",
            "Neural Networks", "Data Science", "Computer Vision",
            "Natural Language Processing", "Reinforcement Learning",
            "Large Language Models", "Product Development",
            "Customer Experience", "Market Analysis", "Algorithm Development",
            "Software Engineering", "Business Sectors"
        ]
        
        # Shuffle the list and trim to the requested number of concepts
        concepts = mock_concepts[:min(num_concepts, len(mock_concepts))]
        
        nodes = []
        edges = []

        # Create the central node (if it exists)
        if concepts:
            center_node = MindMapNode(
                id="node_0",
                text=concepts[0],
                category="general",
                importance_score=1.0,
                size=30,
                color="#E25822"
            )
            nodes.append(asdict(center_node))

        # Create other nodes and link them to the center node
        for i in range(1, len(concepts)):
            node = MindMapNode(
                id=f"node_{i}",
                text=concepts[i],
                category=random.choice(["technology", "business", "science"]),
                importance_score=1.0 - (i * 0.05),
                size=random.randint(15, 25),
                color=random.choice(["#4A90E2", "#7ED321", "#9013FE"])
            )
            nodes.append(asdict(node))
            
            # Link to the center node
            edge = MindMapEdge(
                source="node_0",
                target=f"node_{i}",
                weight=random.random() * 0.5 + 0.5,
                relationship_type="related"
            )
            edges.append(asdict(edge))

        return nodes, edges

class MindMapExporter:
    """Exports mind map data to various formats."""
    @staticmethod
    def to_json(nodes: List[Dict], edges: List[Dict]) -> str:
        return json.dumps({"nodes": nodes, "edges": edges}, indent=2)

    @staticmethod
    def to_graphml(nodes: List[Dict], edges: List[Dict]) -> str:
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'], **node)
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Convert to GraphML string (simplified)
        return "\n".join(nx.generate_graphml(G))

# --- Flask Application Setup ---
app = Flask(__name__)
CORS(app)

# Initialize the mind map generator and exporter
mindmap_generator = MindMapGenerator()
mindmap_exporter = MindMapExporter()

# --- Flask Routes ---

@app.route("/")
def index():
    """Renders the main HTML page for the application."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/generate", methods=["POST"])
def generate_mindmap_api():
    """
    API endpoint to generate a mind map from input text.
    """
    data = request.json
    text_input = data.get("text")
    num_concepts = data.get("num_concepts", 15)
    
    if not text_input:
        return jsonify({"error": "No text provided"}), 400

    logger.info("Generating mind map for input text...")
    try:
        nodes, edges = mindmap_generator.generate_mindmap(text_input, num_concepts)
        return jsonify({"nodes": nodes, "edges": edges})
    except Exception as e:
        logger.error(f"Error generating mind map: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/export_svg", methods=["POST"])
def export_svg():
    """
    API endpoint to export the SVG data as a downloadable file.
    """
    try:
        data = request.json
        svg_content = data.get("svg")
        if not svg_content:
            return jsonify({"error": "No SVG data provided"}), 400

        # Create a temporary file to hold the SVG content
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, "mindmap.svg")

        with open(temp_file_path, "w") as f:
            f.write(svg_content)
        
        return send_file(
            temp_file_path,
            mimetype="image/svg+xml",
            as_attachment=True,
            download_name="mindmap.svg"
        )
    except Exception as e:
        logger.error(f"Error exporting SVG: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/export_png", methods=["POST"])
def export_png():
    """
    API endpoint to export a PNG image from base64 data.
    """
    try:
        data = request.json
        base64_string = data.get("image_data")
        if not base64_string:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)

        # Create a temporary file to save the PNG
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, "mindmap.png")
        
        with open(temp_file_path, "wb") as f:
            f.write(image_data)

        return send_file(
            temp_file_path,
            mimetype="image/png",
            as_attachment=True,
            download_name="mindmap.png"
        )
    except Exception as e:
        logger.error(f"Error exporting PNG: {e}")
        return jsonify({"error": "Internal server error"}), 500

# --- Embedded HTML/CSS/JavaScript Frontend ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MindMapGPT - Dynamic Knowledge Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            overflow: hidden;
        }

        .container {
            display: flex;
            height: 100vh;
        }

        .sidebar {
            width: 350px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            box-shadow: 2px 0 15px rgba(0,0,0,0.1);
            overflow-y: auto;
        }

        .main-content {
            flex: 1;
            position: relative;
            background: rgba(255, 255, 255, 0.1);
        }

        .header {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 15px 30px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }

        h1 {
            color: white;
            font-size: 24px;
            font-weight: 300;
            margin: 0;
        }

        .controls {
            margin-bottom: 20px;
        }

        .input-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }

        textarea {
            width: 100%;
            height: 150px;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            resize: vertical;
            transition: border-color 0.3s;
            font-family: inherit;
        }

        textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
        }

        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
            margin-bottom: 10px;
        }

        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .stats {
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .stat-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .export-controls {
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
        }

        .export-btn {
            margin: 5px 0;
            padding: 8px 16px;
            font-size: 12px;
        }

        #mindmap-container {
            width: 100%;
            height: calc(100vh - 80px);
            position: relative;
        }

        .node {
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .node:hover {
            stroke-width: 3px;
            filter: brightness(1.2);
        }

        .node-label {
            font-family: 'Segoe UI', sans-serif;
            font-size: 11px;
            font-weight: 600;
            text-anchor: middle;
            dominant-baseline: middle;
            fill: white;
            pointer-events: none;
        }

        .link {
            stroke: rgba(255, 255, 255, 0.6);
            stroke-width: 1.5;
            transition: all 0.3s ease;
        }

        .link:hover {
            stroke: rgba(255, 255, 255, 0.9);
            stroke-width: 2.5;
        }

        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 13px;
            pointer-events: none;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }

        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
            display: none;
        }

        .spinner {
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 3px solid white;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error-message {
            background: #ff4757;
            color: white;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 15px;
            display: none;
        }

        .success-message {
            background: #2ed573;
            color: white;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 15px;
            display: none;
        }

        .highlighted {
            stroke: #ff6b6b !important;
            stroke-width: 4px !important;
        }

        .advanced-controls {
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .slider-container {
            margin: 10px 0;
        }

        .slider {
            width: 100%;
            margin: 10px 0;
        }

        .file-upload {
            margin: 10px 0;
        }

        .file-upload input[type="file"] {
            width: 100%;
            padding: 8px;
            border: 2px dashed #ccc;
            border-radius: 6px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>🧠 MindMapGPT</h2>
            
            <div id="error-message" class="error-message"></div>
            <div id="success-message" class="success-message"></div>
            
            <div class="controls">
                <div class="input-group">
                    <label for="text-input">Input Text:</label>
                    <textarea id="text-input" placeholder="Paste your text, notes, research, or transcripts here...">Machine learning has revolutionized the way we approach data analysis and artificial intelligence. Deep learning, a subset of machine learning, uses neural networks with multiple layers to process complex patterns in data. These algorithms excel at tasks like image recognition, natural language processing, and speech synthesis. Companies across industries are implementing AI solutions to automate processes, improve decision-making, and enhance customer experiences. The field continues to evolve rapidly with advances in computer vision, reinforcement learning, and large language models.</textarea>
                </div>
                
                <div class="file-upload">
                    <label for="file-input">Or Upload File:</label>
                    <input type="file" id="file-input" accept=".txt,.md,.doc,.docx" onchange="handleFileUpload(event)">
                </div>
                
                <button onclick="generateMindMap()" id="generate-btn">
                    Generate Mind Map
                </button>
            </div>

            <div class="advanced-controls">
                <h3>Advanced Settings</h3>
                <div class="slider-container">
                    <label for="concept-count">Number of Concepts: <span id="concept-value">15</span></label>
                    <input type="range" id="concept-count" class="slider" min="5" max="30" value="15" onchange="updateConceptValue()">
                </div>
                
                <div class="input-group">
                    <label for="layout-select">Layout Algorithm:</label>
                    <select id="layout-select" style="width: 100%; padding: 8px; border: 2px solid #e0e0e0; border-radius: 6px;">
                        <option value="force">Force-directed</option>
                        <option value="circular">Circular</option>
                        <option value="hierarchical">Hierarchical</option>
                    </select>
                </div>
            </div>

            <div class="stats" id="stats">
                <h3>Mind Map Stats</h3>
                <div class="stat-item">
                    <span>Nodes:</span>
                    <span id="node-count">0</span>
                </div>
                <div class="stat-item">
                    <span>Connections:</span>
                    <span id="edge-count">0</span>
                </div>
                <div class="stat-item">
                    <span>Categories:</span>
                    <span id="category-count">0</span>
                </div>
                <div class="stat-item">
                    <span>Processing Time:</span>
                    <span id="processing-time">-</span>
                </div>
            </div>

            <div class="export-controls">
                <h3>Export & Share</h3>
                <button class="export-btn" onclick="exportJSON()">📄 Export JSON</button>
                <button class="export-btn" onclick="exportSVG()">🖼️ Export SVG</button>
                <button class="export-btn" onclick="exportPNG()">📸 Export PNG</button>
                <button class="export-btn" onclick="shareVisualization()">🔗 Share Link</button>
            </div>
        </div>

        <div class="main-content">
            <div class="header">
                <h1>Dynamic Knowledge Visualization</h1>
            </div>
            
            <div id="mindmap-container">
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <div>Generating mind map...</div>
                </div>
                <div class="tooltip" id="tooltip"></div>
            </div>
        </div>
    </div>
    
    <div id="message-box-container" style="position:fixed;top:20px;left:50%;transform:translateX(-50%);z-index:10000;display:flex;flex-direction:column;gap:10px;"></div>

    <script>
        // Global variables
        let svg, g, simulation;
        let currentNodes = [], currentLinks = [];
        let width, height;
        
        // Color scheme for categories
        const categoryColors = {
            'technology': '#4A90E2',
            'business': '#F5A623',
            'science': '#7ED321',
            'people': '#D0021B',
            'process': '#9013FE',
            'general': '#50E3C2'
        };

        // Initialize the visualization
        function initVisualization() {
            width = document.querySelector('.main-content').offsetWidth;
            height = document.querySelector('.main-content').offsetHeight;

            const container = d3.select('#mindmap-container');
            
            svg = container
                .append('svg')
                .attr('width', width)
                .attr('height', height);

            // Add zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => {
                    g.attr('transform', event.transform);
                });

            svg.call(zoom);

            // Create main group
            g = svg.append('g');

            // Initialize force simulation
            simulation = d3.forceSimulation()
                .force('link', d3.forceLink().id(d => d.id).distance(120).strength(0.5))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(d => (d.size || 20) + 5));
        }
        
        // Message Box Utility
        function showMessage(message, type = 'info') {
            const container = document.getElementById('message-box-container');
            const messageBox = document.createElement('div');
            messageBox.style.cssText = `
                padding: 10px 20px;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                background: ${type === 'error' ? '#ff4757' : (type === 'success' ? '#2ed573' : '#4a69bd')};
                opacity: 0;
                transform: translateY(-20px);
                transition: all 0.5s ease;
            `;
            messageBox.textContent = message;
            container.appendChild(messageBox);

            setTimeout(() => {
                messageBox.style.opacity = '1';
                messageBox.style.transform = 'translateY(0)';
            }, 10);
            
            setTimeout(() => {
                messageBox.style.opacity = '0';
                messageBox.style.transform = 'translateY(-20px)';
                setTimeout(() => messageBox.remove(), 500);
            }, 3000);
        }

        // Generate mind map by calling Python backend
        async function generateMindMap() {
            const textInput = document.getElementById('text-input').value.trim();
            const conceptCount = parseInt(document.getElementById('concept-count').value);
            
            if (!textInput) {
                showMessage('Please enter some text to generate a mind map.', 'error');
                return;
            }

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('generate-btn').disabled = true;
            
            const startTime = Date.now();

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: textInput,
                        num_concepts: conceptCount
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }

                const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);
                document.getElementById('processing-time').textContent = processingTime + 's';

                renderMindMap(data.nodes, data.edges);
                showMessage('Mind map generated successfully!', 'success');
                
            } catch (error) {
                console.error('Error generating mind map:', error);
                showMessage('Failed to generate mind map: ' + error.message, 'error');
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('generate-btn').disabled = false;
            }
        }

        // Render the mind map
        function renderMindMap(nodeData, linkData) {
            currentNodes = nodeData.map(d => ({
                ...d,
                x: width / 2 + (Math.random() - 0.5) * 100,
                y: height / 2 + (Math.random() - 0.5) * 100,
                color: categoryColors[d.category] || categoryColors.general
            }));
            
            currentLinks = linkData.map(d => ({...d}));

            updateVisualization();
            updateStats();
        }

        // Update the visualization
        function updateVisualization() {
            // Clear existing elements
            g.selectAll('.link').remove();
            g.selectAll('.node-group').remove();

            // Create links
            const link = g.selectAll('.link')
                .data(currentLinks)
                .enter().append('line')
                .attr('class', 'link')
                .style('stroke-width', d => Math.max(1, d.weight * 3));

            // Create node groups
            const nodeGroup = g.selectAll('.node-group')
                .data(currentNodes)
                .enter().append('g')
                .attr('class', 'node-group')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            // Add circles
            nodeGroup.append('circle')
                .attr('class', 'node')
                .attr('r', d => d.size || 20)
                .style('fill', d => d.color)
                .style('stroke', '#fff')
                .style('stroke-width', 2)
                .on('mouseover', showTooltip)
                .on('mouseout', hideTooltip);

            // Add labels
            nodeGroup.append('text')
                .attr('class', 'node-label')
                .text(d => {
                    const maxLength = Math.max(8, Math.floor((d.size || 20) * 0.8));
                    return d.text.length > maxLength ? d.text.substring(0, maxLength) + '...' : d.text;
                })
                .style('font-size', d => Math.max(9, (d.size || 20) * 0.35) + 'px');

            // Update simulation
            simulation.nodes(currentNodes).on('tick', ticked);
            simulation.force('link').links(currentLinks);
            simulation.alpha(1).restart();

            function ticked() {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                nodeGroup
                    .attr('transform', d => `translate(${d.x},${d.y})`);
            }
        }

        // Utility functions
        function updateStats() {
            const categories = new Set(currentNodes.map(d => d.category));
            document.getElementById('node-count').textContent = currentNodes.length;
            document.getElementById('edge-count').textContent = currentLinks.length;
            document.getElementById('category-count').textContent = categories.size;
        }

        function updateConceptValue() {
            const value = document.getElementById('concept-count').value;
            document.getElementById('concept-value').textContent = value;
        }

        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('text-input').value = e.target.result;
                    showMessage('File loaded successfully.', 'success');
                };
                reader.onerror = function() {
                    showMessage('Error reading file.', 'error');
                };
                reader.readAsText(file);
            }
        }

        // Drag functions
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        // Tooltip functions
        function showTooltip(event, d) {
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = `
                <strong>${d.text}</strong><br>
                Category: ${d.category}<br>
                Importance: ${(d.importance_score * 100).toFixed(1)}%<br>
                Connections: ${currentLinks.filter(l => l.source.id === d.id || l.target.id === d.id).length}
            `;
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 28) + 'px';
            tooltip.style.opacity = 1;
        }

        function hideTooltip() {
            const tooltip = document.getElementById('tooltip');
            tooltip.style.opacity = 0;
        }

        // Export functions
        function exportJSON() {
            if (currentNodes.length === 0) {
                showMessage('No mind map to export.', 'error');
                return;
            }
            const data = { nodes: currentNodes, edges: currentLinks };
            const jsonString = JSON.stringify(data, null, 2);
            const blob = new Blob([jsonString], {type: "application/json"});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "mindmap.json";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showMessage('JSON exported.', 'success');
        }

        async function exportSVG() {
            if (currentNodes.length === 0) {
                showMessage('No mind map to export.', 'error');
                return;
            }
            const svgElement = document.querySelector('#mindmap-container svg');
            const svgContent = new XMLSerializer().serializeToString(svgElement);
            
            try {
                const response = await fetch('/api/export_svg', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ svg: svgContent })
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = "mindmap.svg";
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    showMessage('SVG exported successfully!', 'success');
                } else {
                    const error = await response.json();
                    throw new Error(error.error || "Failed to export SVG.");
                }
            } catch (error) {
                console.error("Error during SVG export:", error);
                showMessage(error.message, 'error');
            }
        }

        async function exportPNG() {
            if (currentNodes.length === 0) {
                showMessage('No mind map to export.', 'error');
                return;
            }
            const svgElement = document.querySelector('#mindmap-container svg');
            const svgContent = new XMLSerializer().serializeToString(svgElement);

            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            const svgSize = svgElement.getBoundingClientRect();
            canvas.width = svgSize.width * 2;
            canvas.height = svgSize.height * 2;
            ctx.scale(2, 2);

            img.onload = async () => {
                ctx.drawImage(img, 0, 0);
                const imageDataUrl = canvas.toDataURL('image/png');
                const base64String = imageDataUrl.split(',')[1];

                try {
                    const response = await fetch('/api/export_png', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image_data: base64String })
                    });
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = "mindmap.png";
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                        showMessage('PNG exported successfully!', 'success');
                    } else {
                        const error = await response.json();
                        throw new Error(error.error || "Failed to export PNG.");
                    }
                } catch (error) {
                    console.error("Error during PNG export:", error);
                    showMessage(error.message, 'error');
                }
            };

            img.onerror = () => {
                showMessage('Error converting SVG to image.', 'error');
            };

            img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgContent)));
        }

        function shareVisualization() {
            if (currentNodes.length === 0) {
                showMessage('No mind map to share.', 'error');
                return;
            }
            const data = { nodes: currentNodes, edges: currentLinks };
            const compressedData = btoa(JSON.stringify(data));
            const shareUrl = `${window.location.origin}/?data=${encodeURIComponent(compressedData)}`;
            
            navigator.clipboard.writeText(shareUrl)
                .then(() => showMessage('Share link copied to clipboard!', 'success'))
                .catch(err => {
                    console.error('Failed to copy link:', err);
                    showMessage('Failed to copy link.', 'error');
                });
        }
        
        // Resize handler
        function handleResize() {
            width = document.querySelector('.main-content').offsetWidth;
            height = document.querySelector('.main-content').offsetHeight;
            
            if (svg) {
                svg.attr('width', width).attr('height', height);
                simulation.force('center', d3.forceCenter(width / 2, height / 2));
                simulation.alpha(0.3).restart();
            }
        }

        // Initialize on page load
        window.addEventListener('load', () => {
            initVisualization();
            generateMindMap(); // Generate sample mind map on load
        });

        window.addEventListener('resize', handleResize);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)

