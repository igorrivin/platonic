"""
Trace through the face extraction to see where it goes wrong.
"""

import numpy as np
from scipy.spatial import ConvexHull
from volume_checker import stereographic_inverse

def trace_face_extraction():
    """Detailed trace of face extraction algorithm."""
    
    # Create cube points
    points = []
    a = 1.0 / np.sqrt(3)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                z = complex(sx*a/(1-sz*a), sy*a/(1-sz*a))
                points.append(z)
    
    # Inverse stereographic projection
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    hull = ConvexHull(pts3)
    
    print("Step 1: Group simplices by plane")
    print("-" * 50)
    
    planes = {}
    for fi, (a, b, c, d) in enumerate(hull.equations):
        n = np.array([a, b, c], dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0:
            continue
        n = n / norm
        
        # This is the key line - rounding to 8 decimal places
        key = tuple(np.round(np.append(n, d/norm), 8))
        planes.setdefault(key, []).append(hull.simplices[fi])
    
    print(f"Found {len(planes)} planes")
    
    # Process each plane to extract face
    all_faces = []
    
    for plane_idx, (plane_key, tris) in enumerate(planes.items()):
        print(f"\nProcessing plane {plane_idx}:")
        print(f"  Normal: {plane_key[:3]}")
        print(f"  Triangles in this plane: {tris}")
        
        # Build edge count
        edge_count = {}
        for tri in tris:
            v = list(tri)
            for e in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]:
                edge_count[e] = edge_count.get(e, 0) + 1
        
        print(f"  Edge counts: {edge_count}")
        
        # Find boundary edges (appear only once)
        boundary = [e for e, cnt in edge_count.items() if cnt == 1]
        print(f"  Boundary edges: {boundary}")
        
        if not boundary:
            print("  No boundary found!")
            continue
        
        # Build next-vertex map
        nxt = {}
        for u, v in boundary:
            nxt[u] = v
        
        print(f"  Next-vertex map: {nxt}")
        
        # Try to build polygon
        start = boundary[0][0]
        poly = [start]
        seen = {start}
        
        print(f"  Building polygon starting from {start}:")
        while poly[-1] in nxt and nxt[poly[-1]] not in seen:
            next_v = nxt[poly[-1]]
            poly.append(next_v)
            seen.add(next_v)
            print(f"    Added vertex {next_v}, polygon now: {poly}")
        
        if len(poly) >= 3:
            all_faces.append(poly)
            print(f"  Final face: {poly} (length {len(poly)})")
        else:
            print(f"  Face too small: {poly}")
    
    print(f"\n\nTotal faces extracted: {len(all_faces)}")
    print("Face sizes:", [len(f) for f in all_faces])
    
    # The issue might be in how scipy orders the simplices
    # Let's check if the triangles are oriented consistently
    print("\n\nChecking triangle orientations:")
    for i, simplex in enumerate(hull.simplices):
        v0, v1, v2 = pts3[simplex[0]], pts3[simplex[1]], pts3[simplex[2]]
        # Compute normal
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / np.linalg.norm(normal)
        print(f"  Triangle {i} {list(simplex)}: normal = {normal}")

if __name__ == "__main__":
    trace_face_extraction()