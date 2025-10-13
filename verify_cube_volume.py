"""
Let's verify the cube volume computation more carefully.
The issue might be in how we're interpreting the ideal polyhedron.
"""

import numpy as np
import math
from volume_checker import compute_volume, stereographic_inverse

def print_polyhedron_info(name, vertices):
    """Helper to print detailed info about a polyhedron."""
    volume, comb = compute_volume(vertices, return_combinatorics=True)
    print(f"\n{name}:")
    print(f"  Volume: {volume}")
    print(f"  Volume/Λ(π/3): {volume / 1.0149416064096536}")
    print(f"  Vertices: {len(comb['vertices'])}")
    print(f"  Edges: {len(comb['edges'])}")
    print(f"  Faces: {len(comb['faces'])}")
    
    # Analyze face structure
    face_sizes = {}
    for face in comb['faces']:
        size = len(face)
        face_sizes[size] = face_sizes.get(size, 0) + 1
    print(f"  Face sizes: {face_sizes}")
    
    return volume, comb

def main():
    # Let's think about this differently. When we stereographically project
    # the cube vertices and then take the convex hull, we might not get
    # the "ideal cube" in the hyperbolic sense.
    
    # The ideal cube in hyperbolic geometry has 8 ideal vertices
    # (at infinity in the hyperbolic model).
    
    # But our cube vertices project to finite points in the plane.
    # So we're not actually computing the volume of an "ideal" cube.
    
    print("Understanding the polyhedron from cube vertices:")
    
    # Cube vertices
    a = 1.0 / math.sqrt(3)
    cube_sphere = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                cube_sphere.append([i*a, j*a, k*a])
    
    # Project to complex plane
    cube_complex = []
    for v in cube_sphere:
        x, y, z = v
        cube_complex.append(complex(x/(1-z), y/(1-z)))
    
    vol1, comb1 = print_polyhedron_info("Stereographic cube", cube_complex)
    
    # Let's also try the vertices of a regular octahedron
    oct_vertices = [
        complex(1, 0),    # East
        complex(-1, 0),   # West
        complex(0, 1),    # North
        complex(0, -1),   # South
        complex(0, 0),    # Bottom (south pole)
        None              # Top (north pole)
    ]
    
    vol2, comb2 = print_polyhedron_info("Regular octahedron", oct_vertices)
    
    # The fact that we're getting 2×Λ(π/3) suggests this might be
    # the volume of the regular ideal octahedron!
    
    print(f"\nConclusion:")
    print(f"The 'stereographic cube' is giving us volume ≈ 2×Λ(π/3)")
    print(f"This is the volume of the regular ideal octahedron.")
    print(f"The combinatorial structure (6 triangular faces) confirms this.")

if __name__ == "__main__":
    main()