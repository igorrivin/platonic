"""
Debug the Lobachevsky method for tetrahedron.
"""

import numpy as np
import math
from scipy.spatial import ConvexHull
from volume_lobachevsky import (
    rotate_to_north_pole, stereographic_from_north,
    triangle_angles, lobachevsky
)

# Create regular ideal tetrahedron on sphere
z4 = complex(0.5, math.sqrt(3)/2)

# Map to sphere
def complex_to_sphere(z):
    if z is None:
        return np.array([0., 0., 1.])
    x = 2 * z.real / (abs(z)**2 + 1)
    y = 2 * z.imag / (abs(z)**2 + 1) 
    z_coord = (abs(z)**2 - 1) / (abs(z)**2 + 1)
    return np.array([x, y, z_coord])

pts_complex = [0+0j, 1+0j, None, z4]
pts_sphere = np.array([complex_to_sphere(z) for z in pts_complex])

print("Tetrahedron vertices on sphere:")
for i, (zc, ps) in enumerate(zip(pts_complex, pts_sphere)):
    print(f"  {i}: {zc} -> {ps}")

# Get convex hull
hull = ConvexHull(pts_sphere)
print(f"\nConvex hull has {len(hull.simplices)} faces:")
for i, face in enumerate(hull.simplices):
    print(f"  Face {i}: {list(face)}")

# Now let's project from each vertex and see what we get
for proj_vertex in range(4):
    print(f"\n{'='*60}")
    print(f"Projecting from vertex {proj_vertex}")
    
    # Rotate so this vertex is at north pole
    rotated = rotate_to_north_pole(pts_sphere, proj_vertex)
    
    # Check rotation
    print(f"After rotation, vertex {proj_vertex} is at: {rotated[proj_vertex]}")
    
    # Project all vertices
    z_proj = [stereographic_from_north(p) for p in rotated]
    print(f"\nProjected vertices:")
    for i, z in enumerate(z_proj):
        print(f"  {i}: {z}")
    
    # Find which triangles are visible (don't contain proj_vertex)
    visible_triangles = []
    for face in hull.simplices:
        if proj_vertex not in face:
            visible_triangles.append(face)
    
    print(f"\nVisible triangles from vertex {proj_vertex}: {len(visible_triangles)}")
    
    # Compute volume from visible triangles
    total_volume = 0.0
    for tri in visible_triangles:
        i, j, k = tri
        if any(z_proj[v] is None for v in [i, j, k]):
            print(f"  Triangle {list(tri)}: contains infinity, skipping")
            continue
            
        zi, zj, zk = z_proj[i], z_proj[j], z_proj[k]
        angles = triangle_angles(zi, zj, zk)
        vol = sum(lobachevsky(a) for a in angles) - lobachevsky(np.pi)
        
        print(f"  Triangle {list(tri)}: angles = {[f'{a:.4f}' for a in angles]}, volume = {vol:.6f}")
        total_volume += vol
    
    print(f"\nTotal volume from vertex {proj_vertex}: {total_volume:.6f}")

print(f"\n{'='*60}")
print(f"Expected volume: {1.0149416064096537:.6f}")