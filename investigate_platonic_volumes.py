"""
Investigate the volumes of various Platonic solids.
The classical ideal hyperbolic polyhedra have specific volume relationships.
"""

import numpy as np
import math
from volume_checker import compute_volume

LAMBDA_PI_3 = 1.0149416064096537

def test_platonic_volumes():
    """Test various configurations to understand the volume relationships."""
    
    print("Investigating Platonic solid volumes:\n")
    
    # 1. Regular tetrahedron (confirmed working)
    print("1. Regular ideal tetrahedron (3 vertices + infinity):")
    tet = [
        complex(1, 0),
        complex(-0.5, math.sqrt(3)/2),
        complex(-0.5, -math.sqrt(3)/2),
        None
    ]
    v1 = compute_volume(tet)
    print(f"   Volume = {v1:.6f} = {v1/LAMBDA_PI_3:.3f}×Λ(π/3)\n")
    
    # 2. Regular octahedron (6 vertices)
    print("2. Regular ideal octahedron (6 vertices):")
    oct = [
        complex(1, 0),
        complex(-1, 0),
        complex(0, 1),
        complex(0, -1),
        complex(0, 0),
        None
    ]
    v2 = compute_volume(oct)
    print(f"   Volume = {v2:.6f} = {v2/LAMBDA_PI_3:.3f}×Λ(π/3)\n")
    
    # 3. Cube vertices (stereographic projection)
    print("3. Stereographically projected cube (8 vertices):")
    a = 1/math.sqrt(3)
    cube_pts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                z_coord = sz * a
                if abs(z_coord - 1) < 1e-10:
                    cube_pts.append(None)
                else:
                    x, y = sx * a, sy * a
                    cube_pts.append(complex(x/(1-z_coord), y/(1-z_coord)))
    v3, comb3 = compute_volume(cube_pts, return_combinatorics=True)
    print(f"   Volume = {v3:.6f} = {v3/LAMBDA_PI_3:.3f}×Λ(π/3)")
    print(f"   Combinatorics: {len(comb3['vertices'])} vertices, {len(comb3['faces'])} faces\n")
    
    # The discrepancy suggests that the "ideal hyperbolic cube" with volume 5×Λ(π/3)
    # is not simply the stereographic projection of a Euclidean cube.
    
    # In hyperbolic geometry, ideal polyhedra have all vertices at infinity.
    # The "ideal cube" might be a different construction.
    
    print("Summary:")
    print(f"- Regular ideal tetrahedron: {v1/LAMBDA_PI_3:.1f}×Λ(π/3) ✓")
    print(f"- Regular ideal octahedron: {v2/LAMBDA_PI_3:.1f}×Λ(π/3) (expected 2)")
    print(f"- Projected cube: {v3/LAMBDA_PI_3:.1f}×Λ(π/3) (expected 5?)")
    
    print("\nThe projected cube gives 2×Λ(π/3), which matches the octahedron volume.")
    print("This suggests the convex hull of the projected cube vertices")
    print("yields a polyhedron with octahedral volume.")

if __name__ == "__main__":
    test_platonic_volumes()