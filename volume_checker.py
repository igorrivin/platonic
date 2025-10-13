"""
volume_checker.py
-----------------
Standalone function to compute the hyperbolic volume of an ideal polyhedron
from a collection of complex numbers (possibly including INF).

This is checker code for verifying volume computations.
"""

from typing import List, Union, Optional
import numpy as np
import math
import cmath

# Try importing required dependencies
try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("Warning: scipy not available. ConvexHull computation will fail.")

try:
    import mpmath as mp
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False
    print("Warning: mpmath not available. Volume computation will fail.")

# Define INF as a sentinel value
INF = None

def stereographic_inverse(z: Union[complex, None]) -> np.ndarray:
    """
    Inverse stereographic projection from complex plane to unit sphere.
    If z is None (representing infinity), return the north pole (0,0,1).
    """
    if z is None or z is INF:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    
    x, y = z.real, z.imag
    d = x*x + y*y + 1.0
    return np.array([2*x/d, 2*y/d, (x*x + y*y - 1.0)/d], dtype=float)

def stereographic(p: np.ndarray) -> Union[complex, None]:
    """
    Stereographic projection from unit sphere to complex plane.
    Returns None for the north pole.
    """
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    
    # Check if we're at the north pole
    if abs(z - 1.0) < 1e-14:
        return None
    
    denom = 1.0 - z
    if abs(denom) < 1e-15:
        # Very close to north pole
        return None
    
    return complex(x/denom, y/denom)

def cross_ratio(z1: Union[complex, None], z2: Union[complex, None], 
                z3: Union[complex, None], z4: Union[complex, None]) -> complex:
    """
    Cross ratio (z1,z2;z3,z4) with support for None representing infinity.
    """
    # Handle various infinity cases
    if z1 is None:
        if z3 is None:
            return complex(1.0)
        if z4 is None:
            return complex(0.0)
        if z2 is None:
            raise ValueError("Invalid cross ratio: z1=z2=INF")
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

def tetra_shape(z0: Union[complex, None], z1: Union[complex, None], 
                z2: Union[complex, None], z3: Union[complex, None]) -> complex:
    """
    Compute the shape parameter of a tetrahedron from its vertices.
    """
    CR = cross_ratio(z0, z1, z2, z3)
    # Avoid branch cut issues
    if abs(CR.imag) < 1e-15 and CR.real < 0:
        CR += 1e-15j
    return CR

def bloch_wigner(z: complex) -> float:
    """
    Bloch-Wigner dilogarithm function.
    """
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath is required for Bloch-Wigner computation")
    
    return float(mp.im(mp.polylog(2, z)) + mp.arg(1 - z) * mp.log(abs(z)))

def triangulate_faces(faces: List[List[int]]) -> List[tuple]:
    """
    Simple triangulation of polygon faces by connecting to the first vertex.
    """
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(tuple(face))
        else:
            # Fan triangulation from first vertex
            v0 = face[0]
            for i in range(1, len(face) - 1):
                triangles.append((v0, face[i], face[i + 1]))
    return triangles

def compute_volume(points: List[Union[complex, None]], apex_index: Optional[int] = None,
                   return_combinatorics: bool = False) -> Union[float, tuple]:
    """
    Compute the hyperbolic volume of an ideal polyhedron from a collection of
    complex numbers (possibly including None for infinity).
    
    Parameters:
    -----------
    points : List of complex numbers or None
        The vertices of the polyhedron in the complex plane. None represents infinity.
    apex_index : int, optional
        Index of the apex vertex for volume computation. Defaults to 0.
    return_combinatorics : bool, optional
        If True, also return the combinatorial structure (vertices, edges, faces).
    
    Returns:
    --------
    float or tuple : The hyperbolic volume, or (volume, combinatorics) if return_combinatorics=True.
                     combinatorics is a dict with keys:
                     - 'vertices': original input points
                     - 'edges': list of (i,j) vertex index pairs
                     - 'faces': list of lists of vertex indices for each face
                     - 'triangles': list of (i,j,k) tuples for triangulated faces
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for convex hull computation")
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath is required for volume computation")
    
    # Convert points to 3D via inverse stereographic projection
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    
    # Compute convex hull
    hull = ConvexHull(pts3)
    
    # Use simplices directly as triangles
    triangles = [tuple(simplex) for simplex in hull.simplices]
    
    # Extract edges from triangles
    edges = set()
    for tri in triangles:
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            edge = tuple(sorted((v1, v2)))
            edges.add(edge)
    
    # For combinatorics reporting, we can still try to extract faces
    # but this is only for informational purposes, not for volume computation
    faces = []
    if return_combinatorics:
        # Group simplices by face (coplanar triangles)
        planes = {}
        for fi, (a, b, c, d) in enumerate(hull.equations):
            n = np.array([a, b, c], dtype=float)
            norm = np.linalg.norm(n)
            if norm == 0:
                continue
            n = n / norm
            key = tuple(np.round(np.append(n, d/norm), 8))
            planes.setdefault(key, []).append(hull.simplices[fi])
        
        # Just report the groups as faces (don't try to reconstruct polygons)
        for plane_tris in planes.values():
            # Collect all vertices in this face
            face_vertices = set()
            for tri in plane_tris:
                face_vertices.update(tri)
            faces.append(sorted(face_vertices))
    
    # Convert back to complex coordinates (or None)
    z = [stereographic(p) for p in pts3]
    
    # Choose apex
    if apex_index is None:
        apex_index = 0
    
    # Compute volume using Bloch-Wigner
    total_volume = 0.0
    for (a, b, c) in triangles:
        if apex_index in (a, b, c):
            continue
        
        shape = tetra_shape(z[apex_index], z[a], z[b], z[c])
        vol_contrib = abs(bloch_wigner(shape))
        total_volume += vol_contrib
    
    if return_combinatorics:
        combinatorics = {
            'vertices': points,
            'edges': list(edges),
            'faces': faces,
            'triangles': triangles
        }
        return total_volume, combinatorics
    else:
        return total_volume

# Example usage
if __name__ == "__main__":
    # Test with a simple example
    # Four vertices forming a tetrahedron, one at infinity
    test_points = [
        complex(0, 0),
        complex(1, 0),
        complex(0.5, 0.866),  # sqrt(3)/2
        None  # infinity
    ]
    
    try:
        # Test without combinatorics
        volume = compute_volume(test_points)
        print(f"Volume of the ideal tetrahedron: {volume}")
        
        # Test with combinatorics
        volume, comb = compute_volume(test_points, return_combinatorics=True)
        print(f"\nVolume with combinatorics: {volume}")
        print(f"Number of vertices: {len(comb['vertices'])}")
        print(f"Number of edges: {len(comb['edges'])}")
        print(f"Number of faces: {len(comb['faces'])}")
        print(f"Edges: {comb['edges']}")
        print(f"Faces: {comb['faces']}")
    except Exception as e:
        print(f"Error computing volume: {e}")