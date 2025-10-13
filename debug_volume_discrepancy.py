"""
Debug the volume discrepancy - why do we get 2×Λ(π/3) instead of 5×Λ(π/3)?
"""

import numpy as np
import math
from scipy.spatial import ConvexHull
from volume_checker import compute_volume, stereographic_inverse, stereographic, tetra_shape, bloch_wigner

LAMBDA_PI_3 = 1.0149416064096537

def debug_volume_computation():
    """Debug step by step what's happening in the volume computation."""
    
    # Create cube vertices
    a = 1.0 / math.sqrt(3)
    sphere_points = []
    complex_points = []
    
    print("Cube vertices:")
    for i, sx in enumerate([-1, 1]):
        for j, sy in enumerate([-1, 1]):
            for k, sz in enumerate([-1, 1]):
                pt = np.array([sx*a, sy*a, sz*a])
                sphere_points.append(pt)
                z = stereographic(pt)
                complex_points.append(z)
                print(f"  v{len(sphere_points)-1}: ({sx:+d}, {sy:+d}, {sz:+d}) -> {z}")
    
    # Get the actual output from our volume_checker
    volume1, comb1 = compute_volume(complex_points, return_combinatorics=True)
    
    print(f"\nvolume_checker output:")
    print(f"  Volume: {volume1} = {volume1/LAMBDA_PI_3:.3f}×Λ(π/3)")
    print(f"  Faces: {len(comb1['faces'])}")
    print(f"  Face list: {comb1['faces']}")
    print(f"  Triangles: {len(comb1['triangles'])}")
    
    # Now compute with scipy ConvexHull directly
    sphere_array = np.array(sphere_points)
    hull = ConvexHull(sphere_array)
    
    print(f"\nDirect ConvexHull analysis:")
    print(f"  Simplices: {len(hull.simplices)}")
    
    # The key issue: how are faces being grouped?
    print("\nChecking face grouping in volume_checker...")
    
    # Let's manually compute volume using hull simplices
    z_points = [stereographic(p) for p in sphere_points]
    apex = 0
    manual_volume = 0.0
    
    print(f"\nManual volume computation from hull simplices (apex={apex}):")
    for i, tri in enumerate(hull.simplices):
        a, b, c = tri
        if apex in (a, b, c):
            print(f"  Triangle {i} {list(tri)}: skipped (contains apex)")
            continue
        
        shape = tetra_shape(z_points[apex], z_points[a], z_points[b], z_points[c])
        vol = abs(bloch_wigner(shape))
        manual_volume += vol
        print(f"  Triangle {i} {list(tri)}: volume = {vol:.6f}")
    
    print(f"\nManual total: {manual_volume} = {manual_volume/LAMBDA_PI_3:.3f}×Λ(π/3)")
    
    # The issue might be in the face grouping algorithm
    # Let's see what's happening with the face extraction
    
    return volume1, manual_volume

def check_face_extraction():
    """Check how faces are being extracted from the convex hull."""
    
    # Recreate what volume_checker does
    a = 1.0 / math.sqrt(3)
    points = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                points.append(complex(sx*a/(1-sz*a), sy*a/(1-sz*a)))
    
    # Inverse stereographic projection
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    hull = ConvexHull(pts3)
    
    print("\nDebugging face extraction algorithm:")
    print(f"Hull equations shape: {hull.equations.shape}")
    
    # Group simplices by face (this is the code from volume_checker)
    planes = {}
    for fi, (a, b, c, d) in enumerate(hull.equations):
        n = np.array([a, b, c], dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0:
            continue
        n = n / norm
        key = tuple(np.round(np.append(n, d/norm), 8))
        planes.setdefault(key, []).append(hull.simplices[fi])
        
        print(f"  Simplex {fi} {list(hull.simplices[fi])}: normal={n}, d={d/norm:.6f}")
    
    print(f"\nNumber of distinct planes found: {len(planes)}")
    
    # Extract faces
    print("\nExtracting faces from grouped simplices:")
    for plane_idx, (key, tris) in enumerate(planes.items()):
        print(f"\n  Plane {plane_idx}: {len(tris)} triangles")
        print(f"    Normal: {key[:3]}")
        print(f"    Triangles: {tris}")
        
        # Find boundary
        edge_count = {}
        for tri in tris:
            v = list(tri)
            for e in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]:
                edge_count[e] = edge_count.get(e, 0) + 1
        
        boundary = [e for e, cnt in edge_count.items() if cnt == 1]
        print(f"    Boundary edges: {len(boundary)}")
        
        # For a square face, we should have 4 boundary edges
        # But if triangles are grouped wrong, we might get 3

if __name__ == "__main__":
    print("Debugging volume discrepancy...\n")
    
    v1, v2 = debug_volume_computation()
    
    print("\n" + "="*60)
    check_face_extraction()