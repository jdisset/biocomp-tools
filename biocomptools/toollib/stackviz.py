from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional
import json

from jinja2 import Template
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from biocomp.compute import ComputeStack


class StackVisualizerConfig(BaseModel):
    node_fill_color: str = Field(default="#aaa", description="Default fill color for nodes")
    node_stroke_color: str = Field(default="#2F4F4F", description="Stroke color for nodes")
    edge_color: str = Field(default="#666666", description="Default color for edges")
    arrow_color: str = Field(default="#666666", description="Color for edge arrows")
    tooltip_bg_color: str = Field(default="white", description="Background color for tooltips")
    tooltip_border_color: str = Field(default="#333333", description="Border color for tooltips")
    hover_colors: List[str] = Field(
        default=[
            "#d62728",
        ],
        description="Colors used for highlighting networks on hover",
    )
    layer_colors: List[str] = Field(
        default=[
            "#f7f7f7",
            "#e6f3ff",
            "#fff0e6",
            "#e6ffe6",
            "#ffe6e6",
            "#e6e6ff",
            "#fff3e6",
            "#e6fff2",
            "#ffe6f7",
            "#f2ffe6",
        ],
        description="Pastel colors used for layer backgrounds",
    )

    # Layout and Dimensions
    node_spacing: float = Field(default=15, description="Horizontal spacing between nodes")
    layer_height: float = Field(default=60, description="Height of each layer")
    layer_spacing: float = Field(default=70, description="Vertical spacing between layers")
    left_margin: float = Field(default=200, description="Left margin for layer labels")
    right_margin: float = Field(default=50, description="Right margin")
    layer_padding: float = Field(default=20, description="Padding inside layer rectangles")
    uniform_layer_width: bool = Field(
        default=True, description="If True, all layers will be as wide as the longest layer"
    )

    # Geometry Settings
    node_radius: float = Field(default=7, description="Radius of node circles")
    edge_control_offset: float = Field(
        default=40, description="Control point offset for Bezier curves"
    )
    edge_angle_cone: float = Field(
        default=0.7,
        description="Cone angle for distributing edges (in radians)",
    )
    layer_corner_radius: float = Field(default=10, description="Corner radius for layer rectangles")

    get_tooltip_text: Optional[Callable] = Field(
        default=None, description="Custom function to generate tooltip text for a node"
    )
    get_layer_color: Optional[Callable] = Field(
        default=None, description="Custom function to determine layer background color"
    )

    class Config:
        arbitrary_types_allowed = True


def generate_stack_html(
    stack: "ComputeStack", config: Optional[StackVisualizerConfig] = None
) -> str:
    """Generate an interactive HTML visualization of a compute stack using D3.js.

    Args:
        stack: ComputeStack object to visualize (works with both old and new systems)
        config: Optional StackVisualizerConfig for customizing the visualization

    Returns:
        str: HTML content with embedded D3.js visualization
    """
    if config is None:
        config = StackVisualizerConfig()

    layers_data = []
    nodes_data = []
    edges_data = []

    total_networks = len(stack.networks)
    total_nodes = stack.number_of_nodes

    # Detect if this is old or new system
    is_old_system = hasattr(stack.layers[0].nodes[0], 'compute_node_id') if stack.layers and stack.layers[0].nodes else False

    # Process each layer
    for layer_id, layer in enumerate(stack.layers):
        # Get layer type - both systems have this
        layer_type = layer.type_str() if hasattr(layer, 'type_str') else layer.f_type

        # Get signature - try multiple approaches
        signature = None
        if layer.nodes:
            node = layer.nodes[0]
            if hasattr(node, 'type_signature'):
                signature = node.type_signature
            else:
                # For new system, get from graph
                graph = stack.networks[node.network_id].compute_graph
                if hasattr(graph, 'nodes'):  # New system
                    cg_node = graph.nodes.get(node.node_id)
                    if cg_node:
                        signature = f"{cg_node.node_type}"

        layer_info = {
            "id": layer_id,
            "type": layer_type,
            "nodeCount": len(layer.nodes),
            "signature": signature,
        }
        layers_data.append(layer_info)

        # Process nodes in the layer
        for node_idx, node in enumerate(layer.nodes):
            # Get node ID - both systems have this
            if is_old_system:
                node_id = node.compute_node_id
                compute_id = node.compute_node_id
            else:
                node_id = node.node_id
                compute_id = node.node_id

            # Get signature
            node_signature = None
            if hasattr(node, 'type_signature'):
                node_signature = node.type_signature
            else:
                graph = stack.networks[node.network_id].compute_graph
                if hasattr(graph, 'nodes'):  # New system
                    cg_node = graph.nodes.get(node.node_id)
                    if cg_node:
                        node_signature = cg_node.node_type

            # Get batch order if available
            batch_order = getattr(node, 'batch_order', None)

            node_data = {
                "id": node_id,
                "layerId": layer_id,
                "networkId": node.network_id,
                "networkName": node.network_id,
                "localId": node_idx,
                "computeId": compute_id,
                "signature": node_signature,
                "batchOrder": batch_order,
            }
            nodes_data.append(node_data)

            # Process edges (connections to other nodes)
            if is_old_system:
                # Old system: use get_compute_node
                if node.get_compute_node("output_to"):
                    for out_idx, (target_id, slot) in enumerate(node.get_compute_node("output_to")):
                        if (node.network_id, target_id) in stack.node_map:
                            target_layer, target_pos = stack.node_map[(node.network_id, target_id)]
                            edge_data = {
                                "source": node_id,
                                "target": stack.layers[target_layer].nodes[target_pos].compute_node_id,
                                "sourceSlot": out_idx,
                                "targetSlot": slot,
                                "networkId": node.network_id,
                            }
                            edges_data.append(edge_data)
            else:
                # New system: use compute_graph edges
                graph = stack.networks[node.network_id].compute_graph
                outgoing_edges = graph.get_outgoing_edges(node.node_id)
                for edge in outgoing_edges:
                    if (node.network_id, edge.target_id) in stack.node_map:
                        target_layer, target_pos = stack.node_map[(node.network_id, edge.target_id)]
                        edge_data = {
                            "source": node_id,
                            "target": stack.layers[target_layer].nodes[target_pos].node_id,
                            "sourceSlot": edge.output_slot,
                            "targetSlot": edge.input_slot,
                            "networkId": node.network_id,
                        }
                        edges_data.append(edge_data)

    # Create visualization data
    viz_data = {
        "config": {
            "nodeFillColor": config.node_fill_color,
            "nodeStrokeColor": config.node_stroke_color,
            "edgeColor": config.edge_color,
            "arrowColor": config.arrow_color,
            "tooltipBgColor": config.tooltip_bg_color,
            "tooltipBorderColor": config.tooltip_border_color,
            "hoverColors": config.hover_colors,
            "layerColors": config.layer_colors,
            "nodeSpacing": config.node_spacing,
            "layerHeight": config.layer_height,
            "layerSpacing": config.layer_spacing,
            "leftMargin": config.left_margin,
            "rightMargin": config.right_margin,
            "layerPadding": config.layer_padding,
            "uniformLayerWidth": config.uniform_layer_width,
            "nodeRadius": config.node_radius,
            "edgeControlOffset": config.edge_control_offset,
            "edgeAngleCone": config.edge_angle_cone,
            "layerCornerRadius": config.layer_corner_radius,
        },
        "layers": layers_data,
        "nodes": nodes_data,
        "edges": edges_data,
        "summary": {
            "totalNetworks": total_networks,
            "totalLayers": len(stack.layers),
            "totalNodes": total_nodes,
        },
    }

    # Load and render the template
    template = Template(_HTML_TEMPLATE)
    return template.render(viz_data=json.dumps(viz_data, default=str))


def save_stackviz(stack, path, config=None):
    html_content = generate_stack_html(stack, config=config)
    with open(path, "w") as f:
        f.write(html_content)


_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Compute Stack Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <style>
        body {
            font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, Ubuntu;
            margin: 0;
            padding: 20px;
        }

        .tooltip {
            position: absolute;
            padding: 8px;
            border-radius: 4px;
            pointer-events: none;
            font-size: 12px;
            max-width: 300px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 1000;
        }

        .layer-label {
            font-size: 12px;
            dominant-baseline: middle;
        }

        .title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .description {
            font-size: 14px;
            color: #666;
            margin-bottom: 20px;
        }

        .node {
            transition: fill 0.2s;
            cursor: pointer;
        }

        .edge {
            transition: stroke 0.2s;
        }
    </style>
</head>
<body>
    <div id="visualization"></div>

    <script>
        // Visualization data
        const vizData = {{ viz_data | safe }};

        // Calculate dimensions
        const maxNodesInLayer = Math.max(...vizData.layers.map(l => l.nodeCount));
        const width = vizData.config.uniformLayerWidth
            ? vizData.config.leftMargin + maxNodesInLayer * vizData.config.nodeSpacing + vizData.config.rightMargin
            : vizData.config.leftMargin + Math.max(...vizData.layers.map(l => l.nodeCount * vizData.config.nodeSpacing)) + vizData.config.rightMargin;
        const height = vizData.layers.length * (vizData.config.layerHeight + vizData.config.layerSpacing);

        // Create SVG
        const svg = d3.select("#visualization")
            .append("svg")
            .attr("width", width)
            .attr("height", height + 100); // Extra space for title

        // Add title and description
        svg.append("text")
            .attr("class", "title")
            .attr("x", vizData.config.leftMargin)
            .attr("y", 30)
            .text(`Compute Stack Visualization`);

        svg.append("text")
            .attr("class", "description")
            .attr("x", vizData.config.leftMargin)
            .attr("y", 50)
            .text(`${vizData.summary.totalNetworks} Networks, ${vizData.summary.totalLayers} Layers, ${vizData.summary.totalNodes} Nodes`);

        // Create main group and move it down to make room for title
        const mainGroup = svg.append("g")
            .attr("transform", "translate(0, 80)");

        // Define arrow marker
        mainGroup.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 8)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", vizData.config.arrowColor);

        // Helper function to get node position
        function getNodePosition(node) {
            const layer = vizData.layers[node.layerId];
            const layerWidth = vizData.config.uniformLayerWidth
                ? maxNodesInLayer * vizData.config.nodeSpacing
                : layer.nodeCount * vizData.config.nodeSpacing;

            return {
                x: vizData.config.leftMargin + (layerWidth / (layer.nodeCount + 1)) * (node.localId + 1),
                y: node.layerId * (vizData.config.layerHeight + vizData.config.layerSpacing) + vizData.config.layerHeight / 2
            };
        }

        // Helper function to calculate edge angles
        function calculateEdgeAngles(node) {
            const outgoingEdges = vizData.edges.filter(e => e.source === node.id);
            const incomingEdges = vizData.edges.filter(e => e.target === node.id);

            // Calculate angles for outgoing edges
            outgoingEdges.forEach((edge, i) => {
                const angleDelta = vizData.config.edgeAngleCone / (outgoingEdges.length + 1);
                edge.sourceAngle = -vizData.config.edgeAngleCone/2 + angleDelta * (i + 1);
            });

            // Calculate angles for incoming edges
            incomingEdges.forEach((edge, i) => {
                const angleDelta = vizData.config.edgeAngleCone / (incomingEdges.length + 1);
                edge.targetAngle = -vizData.config.edgeAngleCone/2 + angleDelta * (i + 1);
            });
        }

        // Calculate edge angles for all nodes
        vizData.nodes.forEach(calculateEdgeAngles);

        // Create layers
        mainGroup.selectAll(".layer")
            .data(vizData.layers)
            .enter()
            .append("rect")
            .attr("class", "layer")
            .attr("x", vizData.config.leftMargin - vizData.config.layerPadding)
            .attr("y", d => d.id * (vizData.config.layerHeight + vizData.config.layerSpacing))
            .attr("width", d => vizData.config.uniformLayerWidth
                ? maxNodesInLayer * vizData.config.nodeSpacing + 2 * vizData.config.layerPadding
                : d.nodeCount * vizData.config.nodeSpacing + 2 * vizData.config.layerPadding)
            .attr("height", vizData.config.layerHeight)
            .attr("rx", vizData.config.layerCornerRadius)
            .attr("fill", (d, i) => vizData.config.layerColors[i % vizData.config.layerColors.length])
            .attr("opacity", 0.3)
            .attr("stroke", "black");

        // Add layer labels
        mainGroup.selectAll(".layer-label")
            .data(vizData.layers)
            .enter()
            .append("text")
            .attr("class", "layer-label")
            .attr("x", vizData.config.leftMargin - vizData.config.layerPadding - 10)
            .attr("y", d => d.id * (vizData.config.layerHeight + vizData.config.layerSpacing) + vizData.config.layerHeight / 2)
            .attr("text-anchor", "end")
            .each(function(d) {
                const text = d3.select(this);
                text.append("tspan")
                    .text(d.type)
                    .attr("x", vizData.config.leftMargin - vizData.config.layerPadding - 10);
                text.append("tspan")
                    .text(`(${d.nodeCount} nodes)`)
                    .attr("x", vizData.config.leftMargin - vizData.config.layerPadding - 10)
                    .attr("dy", "1.2em");
                text.append("tspan")
                    .text(d.signature)
                    .attr("x", vizData.config.leftMargin - vizData.config.layerPadding - 10)
                    .attr("dy", "1.2em")
                    .style("font-size", "10px");
            });

        // Create edges with distributed angles
        const edgeGroup = mainGroup.append("g").attr("class", "edges");
        const edges = edgeGroup.selectAll(".edge")
            .data(vizData.edges)
            .enter()
            .append("path")
            .attr("class", "edge")
            .attr("fill", "none")
            .attr("stroke", vizData.config.edgeColor)
            .attr("stroke-width", 1)
            .attr("marker-end", "url(#arrowhead)")
            .attr("d", d => {
                const source = getNodePosition(vizData.nodes.find(n => n.id === d.source));
                const target = getNodePosition(vizData.nodes.find(n => n.id === d.target));

                // Use the pre-calculated angles for a more even distribution
                const sourceAngle = d.sourceAngle || 0;
                const targetAngle = d.targetAngle || 0;

                // Calculate control points using the angles
                const dx = target.x - source.x;
                const dy = target.y - source.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                const cp1 = {
                    x: source.x + Math.sin(sourceAngle) * vizData.config.edgeControlOffset,
                    y: source.y + Math.cos(sourceAngle) * vizData.config.edgeControlOffset
                };

                const cp2 = {
                    x: target.x + Math.sin(targetAngle) * vizData.config.edgeControlOffset,
                    y: target.y - Math.cos(targetAngle) * vizData.config.edgeControlOffset
                };

                return `M${source.x},${source.y} C${cp1.x},${cp1.y} ${cp2.x},${cp2.y} ${target.x},${target.y}`;
            });

        // Create tooltip
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "tooltip")
            .style("opacity", 0)
            .style("background-color", vizData.config.tooltipBgColor)
            .style("border", `1px solid ${vizData.config.tooltipBorderColor}`);

            // Create nodes with hover effects
            const nodeGroup = mainGroup.append("g").attr("class", "nodes");
            const nodes = nodeGroup.selectAll(".node")
                .data(vizData.nodes)
                .enter()
                .append("circle")
                .attr("class", "node")
                .attr("cx", d => getNodePosition(d).x)
                .attr("cy", d => getNodePosition(d).y)
                .attr("r", vizData.config.nodeRadius)
                .attr("fill", vizData.config.nodeFillColor)
                .attr("stroke", vizData.config.nodeStrokeColor)
                .on("mouseover", function(event, d) {
                    const networkColor = vizData.config.hoverColors[d.networkId % vizData.config.hoverColors.length];

                    // Find all ancestors (nodes that lead to this node)
                    const ancestors = new Set();
                    const descendants = new Set();
                    const connectedEdges = new Set();

                    // Helper to find ancestors recursively
                    function findAncestors(nodeId) {
                        const incomingEdges = vizData.edges.filter(e => e.target === nodeId && e.networkId === d.networkId);
                        incomingEdges.forEach(e => {
                            connectedEdges.add(e);
                            if (!ancestors.has(e.source)) {
                                ancestors.add(e.source);
                                findAncestors(e.source);
                            }
                        });
                    }

                    // Helper to find descendants recursively
                    function findDescendants(nodeId) {
                        const outgoingEdges = vizData.edges.filter(e => e.source === nodeId && e.networkId === d.networkId);
                        outgoingEdges.forEach(e => {
                            connectedEdges.add(e);
                            if (!descendants.has(e.target)) {
                                descendants.add(e.target);
                                findDescendants(e.target);
                            }
                        });
                    }

                    // Find all connected nodes
                    findAncestors(d.id);
                    findDescendants(d.id);

                    // Add the current node itself
                    const connectedNodes = new Set([d.id, ...ancestors, ...descendants]);

                    // Highlight connected nodes only
                    nodeGroup.selectAll(".node")
                        .attr("fill", n => connectedNodes.has(n.id) ? networkColor : vizData.config.nodeFillColor)
                        .attr("r", n => connectedNodes.has(n.id) ? vizData.config.nodeRadius * 1.2 : vizData.config.nodeRadius)
                        .attr("opacity", n => connectedNodes.has(n.id) ? 1 : 0.2);

                    // Highlight connected edges only
                    edgeGroup.selectAll(".edge")
                        .attr("stroke", e => connectedEdges.has(e) ? networkColor : vizData.config.edgeColor)
                        .attr("stroke-width", e => connectedEdges.has(e) ? 2 : 1)
                        .attr("opacity", e => connectedEdges.has(e) ? 1 : 0.1);

                    // Update arrow markers for highlighted edges
                    const markerId = `arrowhead-${d.networkId}`;
                    if (!mainGroup.select(`#${markerId}`).size()) {
                        mainGroup.select("defs")
                            .append("marker")
                            .attr("id", markerId)
                            .attr("viewBox", "0 -5 10 10")
                            .attr("refX", 8)
                            .attr("refY", 0)
                            .attr("markerWidth", 6)
                            .attr("markerHeight", 6)
                            .attr("orient", "auto")
                            .append("path")
                            .attr("d", "M0,-5L10,0L0,5")
                            .attr("fill", networkColor);
                    }

                    // Apply marker to highlighted edges
                    edgeGroup.selectAll(".edge")
                        .attr("marker-end", e => connectedEdges.has(e) ? `url(#${markerId})` : "url(#arrowhead)");

                    // Show tooltip
                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);

                    const ancestorCount = ancestors.size;
                    const descendantCount = descendants.size;

                    const tooltipContent = `
                        <strong>Network:</strong> ${d.networkId}<br/>
                        <strong>Node ID:</strong> ${d.id}<br/>
                        <strong>Compute ID:</strong> ${d.computeId}<br/>
                        <strong>Layer:</strong> ${vizData.layers[d.layerId].type}<br/>
                        <strong>Signature:</strong> ${d.signature}<br/>
                        <strong>Batch Order:</strong> ${d.batchOrder}<br/>
                        <strong>Ancestors:</strong> ${ancestorCount}<br/>
                        <strong>Descendants:</strong> ${descendantCount}
                    `;

                    tooltip.html(tooltipContent)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                })
                .on("mouseout", function(event, d) {
                    // Reset all nodes
                    nodeGroup.selectAll(".node")
                        .attr("fill", vizData.config.nodeFillColor)
                        .attr("r", vizData.config.nodeRadius)
                        .attr("opacity", 1);

                    // Reset all edges
                    edgeGroup.selectAll(".edge")
                        .attr("stroke", vizData.config.edgeColor)
                        .attr("stroke-width", 1)
                        .attr("opacity", 1)
                        .attr("marker-end", "url(#arrowhead)");

                    // Hide tooltip
                    tooltip.transition()
                        .duration(500)
                        .style("opacity", 0);
                });

            // Add zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on("zoom", (event) => {
                    mainGroup.attr("transform", event.transform);
                });

            svg.call(zoom);

            // Center the visualization initially
            const initialScale = 0.9;
            const initialX = (width - width * initialScale) / 2;
            const initialY = 80;  // Account for the title space
            svg.call(zoom.transform, d3.zoomIdentity
                .translate(initialX, initialY)
                .scale(initialScale));
        </script>
    </body>
    </html>
"""
