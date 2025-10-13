"""
Test cube volume with manually specified correct combinatorics.
This will help us determine if the issue is with the combinatorics or the volume formula.
"""

import numpy as np
import math
import cmath
from volume_from_sphere_points import stereographic, tetra_shape, bloch_wigner

try:
    import mpmath as mp
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False

LAMBDA_PI_3 = 1.0149416064096537

def compute_cube_volume_manual():
    """Compute cube volume with correct combinatorics."""
    
    # Cube vertices on unit sphere
    a = 1.0 / math.sqrt(3)
    
    # Label vertices by their signs
    vertices = []
    vertex_map = {}
    idx = 0
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices.append(np.array([i*a, j*a, k*a]))
                vertex_map[(i, j, k)] = idx
                idx += 1
    
    print(f"Cube vertices (a = {a:.6f}):")
    for (i, j, k), idx in vertex_map.items():
        print(f"  v{idx}: ({i:+d}, {j:+d}, {k:+d})")
    
    # Define the 6 square faces of the cube
    faces = [
        # x = -1 face
        [vertex_map[(-1, -1, -1)], vertex_map[(-1, -1, +1)], 
         vertex_map[(-1, +1, +1)], vertex_map[(-1, +1, -1)]],
        # x = +1 face
        [vertex_map[(+1, -1, -1)], vertex_map[(+1, +1, -1)], 
         vertex_map[(+1, +1, +1)], vertex_map[(+1, -1, +1)]],
        # y = -1 face
        [vertex_map[(-1, -1, -1)], vertex_map[(+1, -1, -1)], 
         vertex_map[(+1, -1, +1)], vertex_map[(-1, -1, +1)]],
        # y = +1 face
        [vertex_map[(-1, +1, -1)], vertex_map[(-1, +1, +1)], 
         vertex_map[(+1, +1, +1)], vertex_map[(+1, +1, -1)]],
        # z = -1 face
        [vertex_map[(-1, -1, -1)], vertex_map[(-1, +1, -1)], 
         vertex_map[(+1, +1, -1)], vertex_map[(+1, -1, -1)]],
        # z = +1 face
        [vertex_map[(-1, -1, +1)], vertex_map[(+1, -1, +1)], 
         vertex_map[(+1, +1, +1)], vertex_map[(-1, +1, +1)]]
    ]
    
    print(f"\nCube faces (6 squares):")
    for i, face in enumerate(faces):
        print(f"  Face {i}: {face}")
    
    # Triangulate each square face by diagonal
    triangles = []
    for face in faces:
        # Triangulate square as two triangles
        triangles.append((face[0], face[1], face[2]))
        triangles.append((face[0], face[2], face[3]))
    
    print(f"\nTriangulated into {len(triangles)} triangles")
    
    # Project vertices to complex plane
    z = [stereographic(v) for v in vertices]
    
    # Compute volume using apex = vertex 0
    apex = 0
    total_volume = 0.0
    
    print(f"\nComputing volume with apex = v{apex}:")
    for i, (a, b, c) in enumerate(triangles):
        if apex in (a, b, c):
            print(f"  Triangle {i}: ({a}, {b}, {c}) - skipped (contains apex)")
            continue
        
        shape = tetra_shape(z[apex], z[a], z[b], z[c])
        vol = abs(bloch_wigner(shape))
        total_volume += vol
        print(f"  Triangle {i}: ({a}, {b}, {c}) - volume = {vol:.6f}")
    
    print(f"\nTotal volume: {total_volume}")
    print(f"Volume / Λ(π/3): {total_volume / LAMBDA_PI_3}")
    print(f"Expected: 5.0")
    
    # Let's also check if we can decompose the cube into 5 tetrahedra
    print(f"\n\nAlternative: Decompose cube into 5 regular tetrahedra")
    print("A cube can be decomposed into 5 tetrahedra:")
    print("- 1 regular tetrahedron at the center")
    print("- 4 tetrahedra at the corners")
    
    return total_volume

if __name__ == "__main__":
    if not _HAS_MPMATH:
        print("mpmath is required for this test")
    else:
        volume = compute_cube_volume_manual()