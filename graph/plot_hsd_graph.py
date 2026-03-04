#!/usr/bin/env python3
"""
Plot the HSD phrase graph using Graphviz.

Generates a DAG showing how Honda Scenes Dataset scenario descriptions
are composed from temporal state, place, road type, weather, surface,
and lighting components.

Usage:
    cd /home/hiwi/VENUSS_supplementary/graph
    python plot_hsd_graph.py
"""

import json
import graphviz
from pathlib import Path

GRAPH_JSON = Path(__file__).parent / "hsd_phrase_graph.json"
OUTPUT_FILE = Path(__file__).parent / "hsd_phrase_graph"

# Color map by category
COLOR_MAP = {
    "root": "#90EE90",       # light green
    "temporal": "#87CEEB",   # sky blue
    "place": "#FFA500",      # orange
    "road": "#FFD700",       # gold
    "weather": "#DDA0DD",    # plum
    "surface": "#F08080",    # light coral
    "lighting": "#98FB98",   # pale green
    "connector": "#D3D3D3",  # light gray
    "end": "#FFFFFF",        # white
}


def load_graph():
    with open(GRAPH_JSON) as f:
        data = json.load(f)
    return data["nodes"], data["connections"]


def create_visualization(nodes, connections):
    dot = graphviz.Digraph("HSD Phrase Graph")
    dot.format = "png"
    dot.attr(rankdir="LR", dpi="300")
    dot.graph_attr.update({
        "fontname": "Arial",
        "fontsize": "14",
        "bgcolor": "white",
        "splines": "true",
        "overlap": "false",
        "nodesep": "0.15",
        "ranksep": "0.5",
    })
    dot.node_attr.update({
        "fontname": "Arial",
        "fontsize": "11",
        "shape": "box",
        "style": "filled,rounded",
        "margin": "0.12,0.06",
    })
    dot.edge_attr.update({
        "arrowsize": "0.7",
    })

    # Group nodes by category for rank alignment
    category_nodes = {}
    node_map = {}
    for node in nodes:
        node_map[node["id"]] = node
        cat = node.get("category", "other")
        category_nodes.setdefault(cat, []).append(node)

    # Add nodes
    for node in nodes:
        nid = str(node["id"])
        label = node["text"]
        cat = node.get("category", "other")
        fillcolor = COLOR_MAP.get(cat, "#FFFFFF")

        if cat == "end":
            shape = "circle"
            label = ""
            width = "0.3"
            dot.node(nid, label, shape=shape, fillcolor=fillcolor,
                     width=width, fixedsize="true")
        elif cat == "root":
            dot.node(nid, label, fillcolor=fillcolor, penwidth="2")
        elif cat == "connector":
            dot.node(nid, label, fillcolor=fillcolor, shape="ellipse",
                     fontsize="9", style="filled")
        else:
            dot.node(nid, label, fillcolor=fillcolor)

    # Use subgraphs with rank=same to align nodes in columns
    rank_groups = {
        "temporal": [1, 2, 3, 4],
        "place": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "road": [31, 32, 33, 34, 35],
        "weather": [41, 42, 43, 44, 45, 46],
        "surface": [51, 52, 53, 54],
        "lighting": [61, 62, 63],
    }
    for group_name, ids in rank_groups.items():
        with dot.subgraph() as s:
            s.attr(rank="same")
            for nid in ids:
                s.node(str(nid))

    # Add edges
    for conn in connections:
        from_id = str(conn["from"])
        to_id = str(conn["to"])
        # Lighter edges for connections to END node
        if conn["to"] == 99:
            dot.edge(from_id, to_id, color="#AAAAAA", style="dashed")
        else:
            dot.edge(from_id, to_id)

    return dot


def create_legend(dot):
    """Add a legend subgraph."""
    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Legend", fontsize="12", fontname="Arial",
                    style="rounded", color="gray")
        legend.node("leg_root", "Root", fillcolor=COLOR_MAP["root"],
                    shape="box", style="filled,rounded")
        legend.node("leg_temporal", "Temporal State", fillcolor=COLOR_MAP["temporal"],
                    shape="box", style="filled,rounded")
        legend.node("leg_place", "Place", fillcolor=COLOR_MAP["place"],
                    shape="box", style="filled,rounded")
        legend.node("leg_road", "Road Type", fillcolor=COLOR_MAP["road"],
                    shape="box", style="filled,rounded")
        legend.node("leg_weather", "Weather", fillcolor=COLOR_MAP["weather"],
                    shape="box", style="filled,rounded")
        legend.node("leg_surface", "Surface", fillcolor=COLOR_MAP["surface"],
                    shape="box", style="filled,rounded")
        legend.node("leg_lighting", "Lighting", fillcolor=COLOR_MAP["lighting"],
                    shape="box", style="filled,rounded")
        # Invisible edges to order legend items vertically
        legend.edge("leg_root", "leg_temporal", style="invis")
        legend.edge("leg_temporal", "leg_place", style="invis")
        legend.edge("leg_place", "leg_road", style="invis")
        legend.edge("leg_road", "leg_weather", style="invis")
        legend.edge("leg_weather", "leg_surface", style="invis")
        legend.edge("leg_surface", "leg_lighting", style="invis")


def main():
    nodes, connections = load_graph()
    print(f"Loaded {len(nodes)} nodes, {len(connections)} connections")

    dot = create_visualization(nodes, connections)
    # No legend — colors are explained in the webpage caption

    dot.render(str(OUTPUT_FILE), cleanup=True)
    print(f"Saved: {OUTPUT_FILE}.png")

    # Also create a compact PDF version
    dot.format = "pdf"
    dot.render(str(OUTPUT_FILE), cleanup=True)
    print(f"Saved: {OUTPUT_FILE}.pdf")


if __name__ == "__main__":
    main()
