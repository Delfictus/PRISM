#!/usr/bin/env python3
"""
Generate chromatic coloring for Nipah protein contact graph
Uses greedy algorithm since PRISM binary has hardcoded paths
"""

import numpy as np
import json
from pathlib import Path

def load_mtx(filename):
    """Load MatrixMarket format file"""
    with open(filename) as f:
        lines = f.readlines()

    # Skip comments
    idx = 0
    while lines[idx].startswith('%'):
        idx += 1

    # Read dimensions
    n, m, nnz = map(int, lines[idx].split())
    idx += 1

    # Build adjacency matrix
    adj = np.zeros((n, n))
    for i in range(idx, len(lines)):
        if lines[i].strip():
            parts = lines[i].split()
            row, col = int(parts[0])-1, int(parts[1])-1
            adj[row, col] = 1
            adj[col, row] = 1

    return adj

def greedy_coloring(adj):
    """
    Greedy graph coloring algorithm
    Returns coloring array and number of colors used
    """
    n = adj.shape[0]
    colors = np.full(n, -1)  # -1 means uncolored

    # Process vertices in order
    for vertex in range(n):
        # Find neighbors' colors
        neighbors = np.where(adj[vertex] > 0)[0]
        neighbor_colors = set(colors[neighbors[colors[neighbors] >= 0]])

        # Assign lowest available color
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[vertex] = color

    return colors, len(np.unique(colors))

def dsatur_coloring(adj):
    """
    DSatur algorithm - better than greedy
    Colors vertices with highest saturation (most colored neighbors) first
    """
    n = adj.shape[0]
    colors = np.full(n, -1)
    saturation = np.zeros(n)  # Number of different colors in neighborhood
    uncolored = set(range(n))

    # Start with vertex of highest degree
    degrees = np.sum(adj, axis=1)
    first = np.argmax(degrees)
    colors[first] = 0
    uncolored.remove(first)

    # Update saturation for neighbors
    neighbors = np.where(adj[first] > 0)[0]
    for neighbor in neighbors:
        if colors[neighbor] == -1:
            saturation[neighbor] = 1

    # Color remaining vertices
    while uncolored:
        # Choose vertex with highest saturation (ties broken by degree)
        max_sat = -1
        next_vertex = -1
        for v in uncolored:
            if saturation[v] > max_sat or (saturation[v] == max_sat and degrees[v] > degrees[next_vertex]):
                max_sat = saturation[v]
                next_vertex = v

        # Find available colors
        neighbors = np.where(adj[next_vertex] > 0)[0]
        neighbor_colors = set(colors[neighbors[colors[neighbors] >= 0]])

        # Assign lowest available color
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[next_vertex] = color
        uncolored.remove(next_vertex)

        # Update saturation for uncolored neighbors
        for neighbor in neighbors:
            if colors[neighbor] == -1:
                # Count unique colors in neighborhood
                neighbor_neighbors = np.where(adj[neighbor] > 0)[0]
                colored_neighbors = neighbor_neighbors[colors[neighbor_neighbors] >= 0]
                if len(colored_neighbors) > 0:
                    saturation[neighbor] = len(np.unique(colors[colored_neighbors]))

    return colors, len(np.unique(colors))

def main():
    # Load Nipah protein contact graph
    mtx_file = "data/nipah/2VSM.mtx"
    print(f"Loading {mtx_file}...")
    adj = load_mtx(mtx_file)
    print(f"Loaded graph: {adj.shape[0]} vertices, {int(np.sum(adj)/2)} edges")

    # Apply DSatur coloring
    print("\nApplying DSatur coloring algorithm...")
    colors, num_colors = dsatur_coloring(adj)
    print(f"Chromatic number: {num_colors}")

    # Analyze color distribution
    print("\nColor distribution:")
    for c in range(num_colors):
        count = np.sum(colors == c)
        print(f"  Color {c}: {count} vertices")

    # Save result in PRISM format
    result = {
        "graph": {
            "vertices": adj.shape[0],
            "edges": int(np.sum(adj)/2)
        },
        "solution": {
            "coloring": colors.tolist(),
            "num_colors": num_colors
        },
        "algorithm": "DSatur (Python implementation)"
    }

    output_file = "output/nipah_coloring.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved coloring to {output_file}")

    # Physics interpretation
    print("\n" + "="*60)
    print("PHYSICS INTERPRETATION")
    print("="*60)

    if num_colors <= 4:
        print(f"χ = {num_colors}: EXCELLENT complementarity")
        print("Expected Kd: < 10 nM (antibody-like affinity)")
    elif num_colors <= 6:
        print(f"χ = {num_colors}: GOOD complementarity")
        print("Expected Kd: 10-100 nM (drug-like affinity)")
    elif num_colors <= 8:
        print(f"χ = {num_colors}: MODERATE complementarity")
        print("Expected Kd: 100 nM - 1 μM (lead compound)")
    else:
        print(f"χ = {num_colors}: POOR complementarity")
        print("Expected Kd: > 1 μM (needs optimization)")

    print("\nThis chromatic structure will guide inhibitor design.")

    return colors, num_colors

if __name__ == "__main__":
    main()