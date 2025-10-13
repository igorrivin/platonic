"""
Debug the cube volume computation.
Let's carefully track what's happening.
"""

import numpy as np
import math
from volume_checker import compute_volume, stereographic_inverse, stereographic

# Known volumes (in units of Λ(π/3))
LAMBDA_PI_3 = 1.0149416064096537

def test_cube_volume():
    """Test the cube volume computation step by step."""
    
    # Cube vertices on unit sphere
    a = 1.0 / math.sqrt(3)
    print(f"Cube vertices on sphere (±{a:.6f}, ±{a:.6f}, ±{a:.6f}):\n")
    
    cube_sphere = []
    for i, sx in enumerate([-1, 1]):
        for j, sy in enumerate([-1, 1]):
            for k, sz in enumerate([-1, 1]):
                v = np.array([sx*a, sy*a, sz*a])
                cube_sphere.append(v)
                print(f"v{len(cube_sphere)-1}: ({sx:+d}a, {sy:+d}a, {sz:+d}a) = {v}")
    
    # Stereographic projection
    print(f"\nStereographic projections:")
    cube_complex = []
    for i, v in enumerate(cube_sphere):
        z = stereographic(v)
        cube_complex.append(z)
        print(f"v{i}: {z}")
    
    # Compute volume with details
    volume, comb = compute_volume(cube_complex, return_combinatorics=True)
    
    print(f"\nVolume computation:")
    print(f"Volume = {volume}")
    print(f"Volume / Λ(π/3) = {volume / LAMBDA_PI_3}")
    print(f"Expected ratio for cube: 5")
    print(f"Actual ratio: {volume / LAMBDA_PI_3:.6f}")
    
    # Let's check what triangles are being used
    print(f"\nTriangulated faces: {len(comb['triangles'])} triangles")
    
    # The volume is computed as sum over tetrahedra formed by
    # apex (default 0) and each triangle
    print(f"\nUsing apex = vertex 0")
    
    # According to the literature:
    # - Regular ideal tetrahedron: 1×Λ(π/3)
    # - Regular ideal octahedron: 2×Λ(π/3)  
    # - Cube: 5×Λ(π/3)
    # - Dodecahedron: 10×Λ(π/3)
    
    # We're getting 2×Λ(π/3), which suggests we might be computing
    # the octahedron volume. But we have 8 vertices (cube) not 6 (octahedron).
    
    # One possibility: The "ideal cube" might mean something different
    # than what we're computing. We're computing the volume of the
    # convex hull of 8 finite points. But the "ideal cube" in hyperbolic
    # geometry might refer to a different object.
    
    return volume, comb

if __name__ == "__main__":
    volume, comb = test_cube_volume()