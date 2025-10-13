"""
Compute hyperbolic volume directly from points on the sphere,
without going through stereographic projection.
"""

from typing import List, Optional
import numpy as np
import math
import cmath

try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import mpmath as mp
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False

def stereographic(p: np.ndarray) -> complex:
    """Stereographic projection from unit sphere to complex plane."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    denom = 1.0 - z
    if abs(denom) < 1e-15:
        # North pole
        return None
    return complex(x/denom, y/denom)

def cross_ratio(z1, z2, z3, z4):
    """Cross ratio with support for None (infinity)."""
    if z1 is None:
        if z3 is None:
            return complex(1.0)
        if z4 is None:
            return complex(0.0)
        return (z2 - z4) / (z2 - z3)
    
    if z2 is None:
        if z4 is None:
            return complex(1.0)
        if z3 is None:
            return complex(0.0)
        return (z1 - z3) / (z1 - z4)
    
    if z3 is None:
        if z4 is None:
            raise ValueError("Invalid cross ratio: z3=z4=INF")
        return (z1 - z4) / (z2 - z4)
    
    if z4 is None:
        return (z1 - z3) / (z2 - z3)
    
    # All finite
    return (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3))

def tetra_shape(z0, z1, z2, z3):
    """Compute shape parameter of a tetrahedron."""
    CR = cross_ratio(z0, z1, z2, z3)
    if abs(CR.imag) < 1e-15 and CR.real < 0:
        CR += 1e-15j
    return CR

def bloch_wigner(z: complex) -> float:
    """Bloch-Wigner dilogarithm."""
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath is required for Bloch-Wigner computation")
    return float(mp.im(mp.polylog(2, z)) + mp.arg(1 - z) * mp.log(abs(z)))

def triangulate_faces(faces: List[List[int]]) -> List[tuple]:
    """Triangulate polygon faces."""
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(tuple(face))
        else:
            # Fan triangulation
            v0 = face[0]
            for i in range(1, len(face) - 1):
                triangles.append((v0, face[i], face[i + 1]))
    return triangles

def compute_volume_from_sphere_points(sphere_points: np.ndarray, apex_index: Optional[int] = None,
                                     return_combinatorics: bool = False):
    """
    Compute hyperbolic volume directly from points on the unit sphere.
    
    Parameters:
    -----------
    sphere_points : numpy array of shape (n, 3)
        Points on the unit sphere
    apex_index : int, optional
        Index of apex for volume computation. Defaults to 0.
    return_combinatorics : bool, optional
        If True, also return combinatorial structure
    
    Returns:
    --------
    float or tuple : Volume, or (volume, combinatorics) if requested
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for convex hull computation")
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath is required for volume computation")
    
    # Compute convex hull
    hull = ConvexHull(sphere_points)
    
    # Group simplices by face
    planes = {}
    for fi, (a, b, c, d) in enumerate(hull.equations):
        n = np.array([a, b, c], dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0:
            continue
        n = n / norm
        key = tuple(np.round(np.append(n, d/norm), 8))
        planes.setdefault(key, []).append(hull.simplices[fi])
    
    # Extract faces
    faces = []
    for tris in planes.values():
        edge_count = {}
        for tri in tris:
            v = list(tri)
            for e in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]:
                edge_count[e] = edge_count.get(e, 0) + 1
        
        boundary = [e for e, cnt in edge_count.items() if cnt == 1]
        if not boundary:
            continue
        
        nxt = {}
        for u, v in boundary:
            nxt[u] = v
        
        start = boundary[0][0]
        poly = [start]
        seen = {start}
        
        while poly[-1] in nxt and nxt[poly[-1]] not in seen:
            poly.append(nxt[poly[-1]])
            seen.add(poly[-1])
        
        if len(poly) >= 3:
            faces.append(poly)
    
    # Get edges
    edges = set()
    for face in faces:
        for i in range(len(face)):
            v1, v2 = face[i], face[(i + 1) % len(face)]
            edge = tuple(sorted((v1, v2)))
            edges.add(edge)
    
    # Triangulate faces
    triangles = triangulate_faces(faces)
    
    # Project to complex plane for volume computation
    z = [stereographic(p) for p in sphere_points]
    
    # Choose apex
    if apex_index is None:
        apex_index = 0
    
    # Compute volume
    total_volume = 0.0
    for (a, b, c) in triangles:
        if apex_index in (a, b, c):
            continue
        
        shape = tetra_shape(z[apex_index], z[a], z[b], z[c])
        vol_contrib = abs(bloch_wigner(shape))
        total_volume += vol_contrib
    
    if return_combinatorics:
        combinatorics = {
            'vertices': sphere_points,
            'edges': list(edges),
            'faces': faces,
            'triangles': triangles
        }
        return total_volume, combinatorics
    else:
        return total_volume


# Test with cube
if __name__ == "__main__":
    LAMBDA_PI_3 = 1.0149416064096537
    
    print("Testing cube volume directly from sphere points:\n")
    
    # Cube vertices on unit sphere
    a = 1.0 / math.sqrt(3)
    cube_points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                cube_points.append([i*a, j*a, k*a])
    
    cube_sphere = np.array(cube_points, dtype=float)
    
    # Compute volume
    volume, comb = compute_volume_from_sphere_points(cube_sphere, return_combinatorics=True)
    
    print(f"Cube vertices: ±{a:.6f} on each coordinate")
    print(f"Number of vertices: {len(cube_sphere)}")
    print(f"Number of faces: {len(comb['faces'])}")
    print(f"Number of edges: {len(comb['edges'])}")
    print(f"\nVolume: {volume}")
    print(f"Volume / Λ(π/3): {volume / LAMBDA_PI_3}")
    print(f"Expected: 5.0")
    
    # Show face structure
    print(f"\nFace sizes:")
    for i, face in enumerate(comb['faces']):
        print(f"  Face {i}: {len(face)} vertices")