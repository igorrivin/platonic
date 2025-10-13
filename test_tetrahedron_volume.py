"""Test tetrahedron volume computation"""

import math
from pointset_to_fuchsian import INF, stereographic_inverse, stereographic, tetra_shape, bloch_wigner
from scipy.spatial import ConvexHull
import numpy as np

# Regular ideal tetrahedron
points = [0+0j, 1+0j, INF, complex(0.5, math.sqrt(3)/2)]

# Get sphere points and hull
pts3 = np.vstack([stereographic_inverse(z) for z in points])
hull = ConvexHull(pts3)
z = [stereographic(p) for p in pts3]

print("Points:", points)
print("\nHull simplices:")
for i, tri in enumerate(hull.simplices):
    print(f"  {i}: {list(tri)} -> vertices {[points[j] if points[j] is not None else 'INF' for j in tri]}")

# For a tetrahedron, we should compute volume differently
# Each face (triangle) with the opposite vertex forms a tetrahedron
print("\nComputing volume for each face:")

total_volume = 0.0

# The hull of 4 points is a tetrahedron with 4 triangular faces
# For each face, we can form a tetrahedron with all 4 vertices
# But we need to be careful about orientation and avoid double counting

# Actually, for the ideal tetrahedron, we should just compute directly
# The shape is determined by the cross ratio of the 4 vertices

print(f"\nTrying different vertex orders:")
orders = [
    (0, 1, 2, 3),
    (0, 2, 1, 3),
    (0, 1, 3, 2),
    (1, 0, 2, 3),
    (1, 2, 0, 3),
    (2, 0, 3, 1)
]

for order in orders:
    if None in [z[i] for i in order]:
        print(f"  Order {order}: contains INF (stereographic(north pole) = None)")
        continue
    shape = tetra_shape(z[order[0]], z[order[1]], z[order[2]], z[order[3]])
    try:
        vol = abs(bloch_wigner(shape))
        print(f"  Order {order}: shape = {shape}, volume = {vol}")
    except:
        print(f"  Order {order}: shape = {shape}, volume = ERROR")

# The issue is z[2] is None (north pole projects to infinity)
print(f"\nz values:")
for i, zi in enumerate(z):
    print(f"  z[{i}] = {zi}")

# Expected volume
REGULAR_IDEAL_TET_VOLUME = 1.0149416064096537
print(f"\nExpected volume: {REGULAR_IDEAL_TET_VOLUME}")
print(f"Error: {abs(volume - REGULAR_IDEAL_TET_VOLUME)}")