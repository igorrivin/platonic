"""
Test the fixed volume_checker.py with the cube example.
"""

import numpy as np
import math
from volume_checker import compute_volume

LAMBDA_PI_3 = 1.0149416064096537

def test_cube_volume():
    """Test cube volume with fixed volume_checker."""
    
    print("Testing cube volume with fixed volume_checker:\n")
    
    # Create cube vertices on unit sphere
    a = 1.0 / math.sqrt(3)
    cube_points = []
    
    print(f"Creating cube vertices (a = {a:.6f}):")
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                # Stereographic projection
                x, y, z = i*a, j*a, k*a
                if abs(z - 1) < 1e-10:
                    cube_points.append(None)
                else:
                    cube_points.append(complex(x/(1-z), y/(1-z)))
    
    # Compute volume
    volume, combinatorics = compute_volume(cube_points, return_combinatorics=True)
    
    print(f"\nResults:")
    print(f"  Volume: {volume}")
    print(f"  Volume / Λ(π/3): {volume / LAMBDA_PI_3}")
    print(f"  Expected: 5.0")
    print(f"\n  Number of triangles: {len(combinatorics['triangles'])}")
    print(f"  Number of edges: {len(combinatorics['edges'])}")
    print(f"  Number of faces: {len(combinatorics['faces'])}")
    
    # Test with other known polyhedra
    print("\n" + "="*60)
    print("Testing regular ideal tetrahedron:\n")
    
    tet_points = [
        complex(1, 0),
        complex(-0.5, math.sqrt(3)/2),
        complex(-0.5, -math.sqrt(3)/2),
        None  # infinity
    ]
    
    tet_volume = compute_volume(tet_points)
    print(f"  Volume: {tet_volume}")
    print(f"  Volume / Λ(π/3): {tet_volume / LAMBDA_PI_3}")
    print(f"  Expected: 1.0")

if __name__ == "__main__":
    test_cube_volume()