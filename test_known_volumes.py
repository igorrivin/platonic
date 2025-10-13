"""
Test known ideal polyhedron volumes.
According to various sources, the volumes should be:
- Regular ideal tetrahedron: Λ(π/3) ≈ 1.01494
- Regular ideal octahedron: 2Λ(π/3) ≈ 2.02988
- Cube (ideal): 5Λ(π/3) ≈ 5.07471

Where Λ is the Lobachevsky function.
"""

import numpy as np
import math
from volume_checker import compute_volume

# Lobachevsky function value at π/3
LAMBDA_PI_3 = 1.0149416064096536

def test_cube_from_octahedron_vertices():
    """
    The cube can be constructed as the convex hull of 8 points that are
    the vertices of a regular octahedron plus two additional points.
    But let's try a different approach - using the actual cube vertices.
    """
    # For a cube with vertices at (±a, ±a, ±a) on the unit sphere
    # we need a = 1/sqrt(3)
    a = 1.0 / math.sqrt(3)
    
    # All 8 vertices of the cube
    vertices = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                # Convert sphere point to complex via stereographic projection
                x, y, z = i*a, j*a, k*a
                if abs(z - 1) < 1e-10:
                    vertices.append(None)  # North pole
                else:
                    vertices.append(complex(x/(1-z), y/(1-z)))
    
    return vertices

def main():
    print("Testing known ideal polyhedron volumes:\n")
    
    # Test 1: Regular ideal tetrahedron
    # The regular ideal tetrahedron has one vertex at infinity and 3 at the vertices
    # of an equilateral triangle
    tet_vertices = [
        complex(1, 0),
        complex(-0.5, math.sqrt(3)/2),
        complex(-0.5, -math.sqrt(3)/2),
        None  # infinity
    ]
    tet_volume = compute_volume(tet_vertices)
    print(f"Regular ideal tetrahedron (4 vertices on circle):")
    print(f"  Computed: {tet_volume}")
    print(f"  Expected: {LAMBDA_PI_3}")
    print(f"  Ratio: {tet_volume / LAMBDA_PI_3}\n")
    
    # Test 2: Cube
    cube_vertices = test_cube_from_octahedron_vertices()
    cube_volume, cube_comb = compute_volume(cube_vertices, return_combinatorics=True)
    print(f"Cube (8 vertices):")
    print(f"  Computed: {cube_volume}")
    print(f"  Expected: {5 * LAMBDA_PI_3}")
    print(f"  Ratio to Λ(π/3): {cube_volume / LAMBDA_PI_3}")
    print(f"  Faces: {len(cube_comb['faces'])}, Edges: {len(cube_comb['edges'])}")
    
    # Let's also try with vertices at infinity
    print(f"\nChecking which approach gives the correct cube volume...")

if __name__ == "__main__":
    main()