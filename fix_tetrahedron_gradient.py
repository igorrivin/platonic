"""
Figure out why gradient computation fails for tetrahedron.
"""

import math
import numpy as np
from scipy.spatial import ConvexHull
from pointset_to_fuchsian import (
    INF, stereographic_inverse, stereographic, cr_axby, 
    bloch_wigner, _G_of_shape, _dlog_cr_coeffs
)

# Regular ideal tetrahedron
points = [0+0j, 1+0j, INF, complex(0.5, math.sqrt(3)/2)]
var_mask = [False, False, False, True]

# Get hull
pts3 = np.vstack([stereographic_inverse(z) for z in points])
hull = ConvexHull(pts3)
z = [stereographic(p) for p in pts3]

print("Points:", points)
print("z values (after stereographic):", z)
print("\nHull simplices:", hull.simplices)

# Compute volume properly (sum over all tetrahedra formed by faces)
total_volume = 0.0
grad = np.zeros((4, 2))

print("\nComputing volume from each simplex:")
for i, (a, b, c) in enumerate(hull.simplices):
    # For each face triangle (a,b,c), we form tetrahedra with all 4 vertices
    # The 4th vertex is the one not in (a,b,c)
    all_verts = {0, 1, 2, 3}
    face_verts = {a, b, c}
    remaining = all_verts - face_verts
    if len(remaining) != 1:
        continue
    apex = list(remaining)[0]
    
    print(f"\nSimplex {i}: face ({a},{b},{c}), apex={apex}")
    
    # Skip if any vertex has None (infinity in complex plane after stereographic)
    if any(z[v] is None for v in [apex, a, b, c]):
        print("  Contains stereographic infinity (None), using cr_axby")
    
    # Shape using cr_axby which handles INF
    z0, z1, z2, z3 = z[apex], z[a], z[b], z[c]
    
    # We need to use the correct cross ratio order
    # For volume, we use CR(z2, z0; z3, z1)
    shape = cr_axby(z2, z0, z3, z1)
    vol = abs(bloch_wigner(shape))
    total_volume += vol
    
    print(f"  z0={z0}, z1={z1}, z2={z2}, z3={z3}")
    print(f"  shape = CR(z2,z0;z3,z1) = {shape}")
    print(f"  volume contribution = {vol}")
    
    # Gradient computation
    G = _G_of_shape(shape)
    Sa, Sx, Sb, Sy = _dlog_cr_coeffs(z2, z0, z3, z1)
    
    print(f"  G(shape) = {G}")
    print(f"  Coefficients: Sa={Sa}, Sx={Sx}, Sb={Sb}, Sy={Sy}")
    
    # The gradient formula maps:
    # Sa -> vertex at position 'a' in CR, which is z2, which is vertex b
    # Sx -> vertex at position 'x' in CR, which is z0, which is vertex apex
    # Sb -> vertex at position 'b' in CR, which is z3, which is vertex c
    # Sy -> vertex at position 'y' in CR, which is z1, which is vertex a

print(f"\nTotal volume: {total_volume}")
print(f"Expected: 1.0149416064096537")

# The issue is likely that with only 4 points and apex=0,
# we skip 3 out of 4 faces in the standard algorithm