"""Debug why volume is 0 in gradient computation"""

from pointset_to_fuchsian import INF, volume_and_gradient_from_pointset, volume_from_pointset
import math

# Test configuration
points = [0+0j, 1+0j, INF, complex(0.5, math.sqrt(3)/2)]

print("Testing volume computation:")
print(f"Points: {points}")

# Try the non-gradient volume function
try:
    vol_no_grad = volume_from_pointset(points)
    print(f"\nvolume_from_pointset: {vol_no_grad}")
except Exception as e:
    print(f"\nvolume_from_pointset error: {e}")

# Try the gradient version
try:
    var_mask = [False, False, False, True]
    vol_with_grad, grad = volume_and_gradient_from_pointset(points, variable_mask=var_mask)
    print(f"\nvolume_and_gradient_from_pointset: {vol_with_grad}")
    print(f"Gradient: {grad[3]}")
except Exception as e:
    print(f"\nvolume_and_gradient_from_pointset error: {e}")

# Let's trace through what's happening
import numpy as np
from scipy.spatial import ConvexHull

# Let's compute the actual stereographic inverses
from pointset_to_fuchsian import stereographic_inverse

pts3 = np.vstack([stereographic_inverse(z) for z in points])
print("\nActual stereographic inverses:")
for i, (z, p) in enumerate(zip(points, pts3)):
    print(f"  {i}: {z} -> {p}")

print(f"\nSphere points:\n{pts3}")

hull = ConvexHull(pts3)
print(f"\nHull simplices: {len(hull.simplices)}")
for i, tri in enumerate(hull.simplices):
    print(f"  {i}: {tri}")

# The issue might be that all tetrahedra contain the apex
apex = 0
skipped = 0
for tri in hull.simplices:
    if apex in tri:
        skipped += 1
        
print(f"\nWith apex={apex}, skipped {skipped}/{len(hull.simplices)} triangles")