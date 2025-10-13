"""
Test the volume computation for a cube inscribed in the unit sphere.
The volume should be 5 times the volume of a regular ideal tetrahedron.
"""

import numpy as np
from volume_checker import compute_volume, stereographic

# Regular ideal tetrahedron volume (Lobachevsky function value)
REGULAR_IDEAL_TET_VOLUME = 1.0149416064096536

def create_unit_cube_vertices():
    """Create vertices of a cube inscribed in the unit sphere."""
    # Cube vertices at (±1/√3, ±1/√3, ±1/√3)
    coord = 1.0 / np.sqrt(3)
    vertices = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices.append(np.array([i*coord, j*coord, k*coord]))
    return vertices

def main():
    # Create cube vertices on the sphere
    sphere_vertices = create_unit_cube_vertices()
    
    # Project to complex plane via stereographic projection
    complex_points = []
    for v in sphere_vertices:
        z = stereographic(v)
        complex_points.append(z)
        print(f"Sphere point {v} -> Complex point {z}")
    
    # Compute volume
    volume, combinatorics = compute_volume(complex_points, return_combinatorics=True)
    
    print(f"\nCube ideal polyhedron:")
    print(f"Volume: {volume}")
    print(f"Expected (5 * regular tet): {5 * REGULAR_IDEAL_TET_VOLUME}")
    print(f"Ratio: {volume / REGULAR_IDEAL_TET_VOLUME}")
    
    print(f"\nCombinatorics:")
    print(f"Vertices: {len(combinatorics['vertices'])}")
    print(f"Edges: {len(combinatorics['edges'])}")
    print(f"Faces: {len(combinatorics['faces'])}")
    
    # Check if we get 6 faces (cube faces)
    print(f"\nFace sizes:")
    for i, face in enumerate(combinatorics['faces']):
        print(f"  Face {i}: {len(face)} vertices - {face}")
    
    # Let's also check which vertices are at infinity
    print(f"\nVertices at infinity:")
    for i, v in enumerate(combinatorics['vertices']):
        if v is None:
            print(f"  Vertex {i} is at infinity")
    
    # Print edges to understand the structure
    print(f"\nEdges: {combinatorics['edges']}")
    
    # Let's check if this is actually an octahedron
    # A cube has 12 edges, an octahedron has 12 edges
    # But we have 14 edges, which is strange
    
    # Print the actual sphere coordinates to verify
    print(f"\nOriginal sphere coordinates:")
    for i, v in enumerate(sphere_vertices):
        print(f"  Vertex {i}: {v}")
    
    # Check distances between vertices
    print(f"\nChecking edge lengths on sphere:")
    for e in combinatorics['edges']:
        v1, v2 = sphere_vertices[e[0]], sphere_vertices[e[1]]
        dist = np.linalg.norm(v1 - v2)
        print(f"  Edge {e}: distance = {dist:.6f}")

if __name__ == "__main__":
    main()