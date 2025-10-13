
"""
pointset_to_fuchsian.py
-----------------------
Pipeline:
  (a) Start from a finite point set P in the complex plane (R^2).
  (b) Back-project to the unit sphere S^2 via inverse stereographic projection.
  (c) Compute the convex hull in R^3 to get an ideal polyhedron combinatorics.
  (d) Triangulate faces, compute complex shear–bend parameters via cross-ratios,
      feed shears (real parts) to the Penner–Rivin algorithm to get the Fuchsian group.
  (e) Since we get cross-ratios, we also recover exterior dihedral angles and (by
      a pulling triangulation into ideal tetrahedra) the hyperbolic volume.

Optional extras:
  - Triangular mesh export + midpoint subdivision + Klein→Poincaré conversion
  - Offscreen PNG renderer (pyrender) with lights, transparency, and a shaderless soap-film iridescence

Requirements:
  - numpy
  - scipy (for 3D convex hull) -- optional but strongly recommended
  - mpmath (for Bloch–Wigner dilogarithm in volume calculation)
  - trimesh (optional, for mesh packaging)
  - pyrender, imageio (optional, for offscreen PNG rendering)
  - rivin_holonomy.py in the same directory (the Penner–Rivin engine)
"""

import cmath
import math
from typing import List, Tuple, Dict, Any, Optional

try:
    import numpy as np
except Exception as e:
    raise ImportError("pointset_to_fuchsian requires numpy.") from e

# Optional dependency for convex hull
try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Optional dependency for volume
try:
    import mpmath as mp
    _HAS_MPMATH = True
except Exception:
    _HAS_MPMATH = False

# Local import; expects rivin_holonomy.py available
from rivin_holonomy import Triangulation, generators_from_triangulation, matmul, matR, matL, matI



# ---------- Symbolic infinity & robust cross-ratios ----------

# Sentinel for the point at infinity in \widehat{\mathbb C}
INF = None

def cr_axby(a, x, b, y):
    """
    Möbius-invariant cross ratio CR(a,x; b,y) = M(a)/M(b), with M(z)=(z-x)/(z-y),
    allowing any of a,x,b,y to be INF (None). Returns a complex number.
    """
    # Degenerate: both edge endpoints at INF is not expected
    if x is INF and y is INF:
        raise ValueError("Degenerate configuration: x=y=INF in cr_axby")
    # Cases where either x or y is INF
    if x is INF:
        # M(z) ~ 1/(z - y)
        if a is INF or b is INF:
            raise ValueError("Unexpected INF pattern (a or b is INF while x is INF)")
        return (b - y) / (a - y)
    if y is INF:
        # M(z) ~ (z - x)
        if a is INF or b is INF:
            raise ValueError("Unexpected INF pattern (a or b is INF while y is INF)")
        return (a - x) / (b - x)
    # Now x,y finite
    if a is INF and b is INF:
        # M(a)=M(b)=1
        return 1.0 + 0.0j
    if a is INF:
        # 1 / ((b-x)/(b-y))
        return (b - y) / (b - x)
    if b is INF:
        return (a - x) / (a - y)
    # All finite
    return ((a - x)*(b - y)) / ((a - y)*(b - x))

def shear_bend_from_four_safe(zx, za, zy, zb):
    """
    Safe complex shear-bend using cr_axby, with tiny nudge off the negative real axis
    to avoid branch cut issues for the principal log.
    """
    import cmath
    CR = cr_axby(za, zx, zb, zy)
    if abs(CR.imag) < 1e-15 and CR.real < 0:
        CR += 1e-15j
    return cmath.log(CR)

def cross_ratio_INF(z1, z2, z3, z4):
    """
    Cross ratio [z1, z2; z3, z4] with any argument possibly INF.
    Reduces to finite formulas by taking limits.
    """
    # Handle one INF among the four (typical in our use)
    if z1 is INF:
        # ((z1 - z3)/(z1 - z4)) -> 1, so CR = (z2 - z4)/(z2 - z3)
        if z2 is INF:
            raise ValueError("Unexpected: z1=z2=INF in cross_ratio_INF")
        return (z2 - z4) / (z2 - z3)
    if z2 is INF:
        # CR = (z1 - z3)/(z1 - z4)
        if z1 is INF:
            raise ValueError("Unexpected: z1=z2=INF in cross_ratio_INF")
        return (z1 - z3) / (z1 - z4)
    if z3 is INF:
        # CR = (z2 - z4)/(z1 - z4)
        return (z2 - z4) / (z1 - z4)
    if z4 is INF:
        # CR = (z1 - z3)/(z2 - z3)
        return (z1 - z3) / (z2 - z3)
    # All finite
    return ((z1 - z3)*(z2 - z4)) / ((z1 - z4)*(z2 - z3))

# ---------- Geometry helpers ----------

def stereographic_inverse(z):
    """Inverse stereographic projection from C∪{INF} to S^2 ⊂ R^3 (north-pole projection)."""
    if z is INF:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    x, y = z.real, z.imag
    d = x*x + y*y + 1.0
    return np.array([2*x/d, 2*y/d, (x*x + y*y - 1.0)/d], dtype=float)

def stereographic(p: np.ndarray):
    """Stereographic projection from S^2\{(0,0,1)} to C, returning INF for (near) north pole."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    denom = 1.0 - z
    if abs(denom) < 1e-12:
        return INF  # exact (or near) north pole
    return complex(x/denom, y/denom)

def cross_ratio(z1: complex, z2: complex, z3: complex, z4: complex) -> complex:
    """Möbius-invariant cross ratio [z1, z2; z3, z4] = (z1 - z3)(z2 - z4)/((z1 - z4)(z2 - z3))."""
    return (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3))

def shear_bend_from_four(zx: complex, za: complex, zy: complex, zb: complex) -> complex:
    """
    Complex shear-bend for oriented edge x->y with adjacent third vertices a (left) and b (right):
      tau = log( cross_ratio(za, zx, zb, zy) ).
    Re tau = shear; Im tau = exterior dihedral (bending) across the edge (0 for diagonals within a face).
    """
    CR = cross_ratio(za, zx, zb, zy)
    if abs(CR.imag) < 1e-15 and CR.real < 0:
        CR += 1e-15j
    return cmath.log(CR)

def _M_ratio_any(a, b, x, y):
    """
    Compute M(a)/M(b) with M(z) = (z - x)/(z - y), allowing any of a,b,x,y to be infinity (represented by None).
    Cases:
      - If neither x nor y is inf: M(inf) = 1, so a=inf -> 1, b=inf -> 1.
      - If x is inf: M(z) ~ 1/(z - y) up to scale; ratio = (1/(a-y)) / (1/(b-y)) = (b - y)/(a - y).
      - If y is inf: M(z) ~ (z - x) up to scale; ratio = (a - x)/(b - x).
    """
    a_inf = (a is None); b_inf = (b is None); x_inf = (x is None); y_inf = (y is None)
    # Degenerate: both x and y cannot be infinity
    if x_inf and y_inf:
        raise ValueError("Both endpoints (x,y) are infinity.")
    # Handle x or y infinite
    if x_inf and not y_inf:
        # M(z) = 1/(z - y) up to scale
        if a_inf and b_inf:
            return 1.0  # (0)/(0) in scaled sense -> limit 1
        if a_inf:
            return 0.0
        if b_inf:
            return float('inf')
        return (b - y)/(a - y)
    if y_inf and not x_inf:
        # M(z) = (z - x) up to scale
        if a_inf and b_inf:
            return 1.0
        if a_inf:
            return float('inf')
        if b_inf:
            return 0.0
        return (a - x)/(b - x)
    # Neither x nor y is inf
    def M(u):
        if u is None:
            return 1.0  # limit (u->inf) (u - x)/(u - y) -> 1
        return (u - x)/(u - y)
    Ma = M(a); Mb = M(b)
    # Guard Mb==0
    if Mb == 0:
        return float('inf')
    return Ma / Mb

def _as_complex_or_none(idx, z_complex, pts3, north_eps=1e-12):
    """
    Return complex coordinate for vertex idx, or None if it's (numerically) the north pole (∞).
    """
    if pts3[idx][2] > 1.0 - north_eps:
        return None  # treat as ∞
    return z_complex[idx]

# ---------- Convex hull and face structure ----------

def convex_hull_from_points_on_sphere(pts3: np.ndarray) -> Dict[str, Any]:
    """
    Compute convex hull in 3D.
    Returns a dict with vertices (Nx3 array), faces (list of lists of vertex indices in CCW order),
    and edges (set of unordered pairs).
    """
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy not found. Install scipy to compute the 3D convex hull "
                           "(pip install scipy). Alternatively, provide faces yourself.")
    hull = ConvexHull(pts3)
    # Group coplanar triangles into polygonal faces using plane equations
    planes = {}
    for fi, (a,b,c,d) in enumerate(hull.equations):
        n = np.array([a,b,c], dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0: continue
        n = n / norm
        key = tuple(np.round(np.append(n, d/norm), 8))
        planes.setdefault(key, []).append(hull.simplices[fi])
    faces = []
    for tris in planes.values():
        edge_count = {}
        for tri in tris:
            v = list(tri)
            edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]
            for e in edges:
                edge_count[e] = edge_count.get(e, 0) + 1
        boundary = [e for e, cnt in edge_count.items() if cnt == 1]
        nxt = {}
        for u,v in boundary: nxt[u] = v
        if not boundary: continue
        start = boundary[0][0]
        poly = [start]
        seen = set([start])
        while poly[-1] in nxt and nxt[poly[-1]] not in seen:
            poly.append(nxt[poly[-1]]); seen.add(poly[-1])
        if len(poly) >= 3:
            faces.append(poly)
    edge_set = set()
    for f in faces:
        m = len(f)
        for i in range(m):
            a = f[i]; b = f[(i+1)%m]
            edge_set.add(tuple(sorted((a,b))))
    return {"vertices": pts3, "faces": faces, "edges": edge_set}

# ---------- Triangulation of faces ----------

def triangulate_faces_with_face_id(faces: List[List[int]]) -> Tuple[List[Tuple[int,int,int]], List[int]]:
    """Fan triangulate each face from its first vertex. Return triangles and a parallel face-id list."""
    tris = []
    tri_face_id = []
    for f_idx, f in enumerate(faces):
        if len(f) == 3:
            tris.append(tuple(f)); tri_face_id.append(f_idx)
        else:
            a0 = f[0]
            for i in range(1, len(f)-1):
                tri = (a0, f[i], f[i+1])
                tris.append(tri); tri_face_id.append(f_idx)
    return tris, tri_face_id

# ---------- Build Penner–Rivin data from hull ----------

def build_penner_rivin_from_hull(pts3: np.ndarray,
                                 faces: List[List[int]]) -> Tuple[Triangulation, Dict[int, float], Dict[str, Any]]:
    """
    From vertices on S^2 (as Nx3), and polygonal faces (vertex-index cycles), build:
      - a Triangulation object (for the ideal triangulation of the surface),
      - a shear dict Z[eid] (real parts of shear-bends),
      - an info dict with maps: 'edge_id', 'opposites', 'triangles', 'faces', 'stereo', 'tri_face_id'.
    """
    N = pts3.shape[0]
    z = np.array([stereographic(pts3[i]) for i in range(N)], dtype=complex)
    tris, tri_face_id = triangulate_faces_with_face_id(faces)

    def canonical_edge(u,v): return (u,v) if u < v else (v,u)
    Tverts = [tuple(t) for t in tris]

    oriented_to = {}
    edge_id = {}
    opp = {}
    edge_occ = {}
    for ti, tri in enumerate(Tverts):
        for si in range(3):
            u, v = tri[si%3], tri[(si+1)%3]
            key = canonical_edge(u,v)
            edge_occ.setdefault(key, []).append((ti, si))

    for key, occ in edge_occ.items():
        if len(occ) != 2:
            continue
        (t1,s1), (t2,s2) = occ
        if key not in edge_id:
            edge_id[key] = len(edge_id)
        eid = edge_id[key]
        oriented_to[(t1, s1)] = (t2, s2, eid)
        oriented_to[(t2, s2)] = (t1, s1, eid)
        a = Tverts[t1][(s1+2)%3]
        b = Tverts[t2][(s2+2)%3]
        opp[eid] = (a, b)

    orientation = {}
    for key, eid in edge_id.items():
        (t1,s1), (t2,s2) = edge_occ[key]
        orientation[eid] = ((t1, s1), (t2, s2))

    Adj = {}
    for (t,s), (u,su,eid) in oriented_to.items():
        Adj[(t,s)] = (u,su,eid)
    Order = {t: [0,1,2] for t in range(len(Tverts))}

    # Shears
    Z = {}
    # Precompute which indices behave as ∞ in this chart
    # (We still keep their complex values for non-infinite vertices.)
    for key, eid in edge_id.items():
        (t1,s1), (t2,s2) = edge_occ[key]
        tri1 = Tverts[t1]; tri2 = Tverts[t2]
        u, v = tri1[s1%3], tri1[(s1+1)%3]
        a = tri1[(s1+2)%3]
        b = tri2[(s2+2)%3]
        # Complex or None for ∞
        zx = _as_complex_or_none(u, z, pts3)
        zy = _as_complex_or_none(v, z, pts3)
        za = _as_complex_or_none(a, z, pts3)
        zb = _as_complex_or_none(b, z, pts3)
        R = _M_ratio_any(za, zb, zx, zy)  # cross-ratio = M(a)/M(b)
        R = complex(R)
        if abs(R.imag) < 1e-15 and R.real < 0:
            R += 1e-15j
        tau = cmath.log(R)
        Z[eid] = float(tau.real)

    T = Triangulation(F=len(Tverts), adjacency=Adj, order=Order, orientation=orientation)
    info = {
        "edge_id": edge_id,
        "opposites": opp,
        "triangles": Tverts,
        "faces": faces,
        "stereo": z,
        "pts3": pts3,
        "tri_face_id": tri_face_id,
    }
    return T, Z, info

# ---------- High-level pipeline ----------

def group_from_pointset(points: List[complex], add_north_pole: bool = False) -> Dict[str, Any]:
    """Main entry: planar points -> hull -> shears -> holonomy."""
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    if add_north_pole:
        pts3 = np.vstack([pts3, np.array([[0.0, 0.0, 1.0]])])
    hull = convex_hull_from_points_on_sphere(pts3)
    faces = hull["faces"]
    T, Z, info = build_penner_rivin_from_hull(pts3, faces)
    gens = generators_from_triangulation(T, Z, root=0)
    return {
        "Gamma_generators": [M for (_,_,_,M) in gens],
        "T": T,
        "Z": Z,
        "info": info,
        "faces": faces,
    }

# ---------- Arithmeticity helpers ----------

def trace(M):
    return M[0][0] + M[1][1]

def invariant_trace_field_signature(generators: List[List[List[float]]]) -> Dict[str, Any]:
    """Heuristic signature for k(Γ): compute traces of squares of generators and a few products."""
    vals = []
    # squares
    for M in generators:
        M2 = [[M[0][0]*M[0][0] + M[0][1]*M[1][0], M[0][0]*M[0][1] + M[0][1]*M[1][1]],
              [M[1][0]*M[0][0] + M[1][1]*M[1][0], M[1][0]*M[0][1] + M[1][1]*M[1][1]]]
        vals.append(trace(M2))
    # pairwise products (limited)
    for i in range(min(5, len(generators))):
        for j in range(i+1, min(5, len(generators))):
            A = generators[i]; B = generators[j]
            AB = [[A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
                  [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]]
            AB2 = [[AB[0][0]*AB[0][0] + AB[0][1]*AB[1][0], AB[0][0]*AB[0][1] + AB[0][1]*AB[1][1]],
                   [AB[1][0]*AB[0][0] + AB[1][1]*AB[1][0], AB[1][0]*AB[0][1] + AB[1][1]*AB[1][1]]]
            vals.append(trace(AB2))
    def near_rational(x, tol=1e-9):
        r = round(x)
        if abs(x - r) < tol:
            return True, float(r)
        return False, x
    rational_flags = [near_rational(v) for v in vals]
    all_int = all(flag and float(val).is_integer() for flag, val in rational_flags)
    return {"traces_sq": vals, "all_integers": all_int, "rationality": rational_flags}

# ---------- Dihedral angles and volume ----------

def edge_dihedrals(info: Dict[str,Any]) -> Dict[Tuple[int,int], float]:
    """Exterior dihedral angles (radians) on hull edges as |Im tau|; 0 on face diagonals."""
    z = info["stereo"]
    triangles = info["triangles"]
    tri_face_id = info["tri_face_id"]
    # reconstruct occurrences on edges
    dihedral = {}
    # Build all undirected edges from triangles
    from collections import defaultdict
    edge_occ = defaultdict(list)
    for t_idx, tri in enumerate(triangles):
        for s in range(3):
            a,b = tri[s%3], tri[(s+1)%3]
            key = (a,b) if a<b else (b,a)
            edge_occ[key].append((t_idx, s))
    for (u,v), occ in edge_occ.items():
        if len(occ) != 2:
            continue
        (t1,s1),(t2,s2) = occ
        if tri_face_id[t1] == tri_face_id[t2]:
            dihedral[(u,v)] = 0.0; continue
        tri1 = triangles[t1]; tri2 = triangles[t2]
        x,y = tri1[s1%3], tri1[(s1+1)%3]
        a = tri1[(s1+2)%3]; b = tri2[(s2+2)%3]
        tau = shear_bend_from_four(z[x], z[a], z[y], z[b])
        dihedral[(u,v)] = abs(float(tau.imag))
    return dihedral

def bloch_wigner(z: complex) -> float:
    """Bloch–Wigner dilogarithm D(z) = Im Li2(z) + arg(1 - z) log |z|."""
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath is required for volume computation (pip install mpmath).")
    return float(mp.im(mp.polylog(2, z)) + mp.arg(1 - z)*mp.log(abs(z)))

def tetra_shape(z0: complex, z1: complex, z2: complex, z3: complex) -> complex:
    """Shape parameter for ideal tetrahedron (z0,z1,z2,z3): cross-ratio [z0,z1; z2,z3]."""
    CR = cross_ratio(z0, z1, z2, z3)
    if abs(CR.imag) < 1e-15 and CR.real < 0:
        CR += 1e-15j
    return CR

def volume_from_hull(pts3: np.ndarray, faces: List[List[int]], apex: Optional[int]=None) -> float:
    """Hyperbolic volume via a pulling triangulation from 'apex'."""
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath is required for volume computation (pip install mpmath).")
    z = [stereographic(p) for p in pts3]
    tris, _ = triangulate_faces_with_face_id(faces)
    if apex is None:
        apex = 0
    total = 0.0
    for (a,b,c) in tris:
        if apex in (a,b,c): continue
        shp = tetra_shape(z[apex], z[a], z[b], z[c])
        total += abs(bloch_wigner(shp))
    return total

def volume_from_pointset(points: List[complex], apex_index: Optional[int]=None, add_north_pole: bool=False) -> float:
    """From planar points to hull to volume; optionally add the north pole before the hull (pyramids)."""
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    if add_north_pole:
        pts3 = np.vstack([pts3, np.array([[0.0,0.0,1.0]])])
    hull = convex_hull_from_points_on_sphere(pts3)
    faces = hull["faces"]
    return volume_from_hull(pts3, faces, apex=apex_index)

# ---------- Meshing, subdivision, and model conversion ----------

# Optional trimesh dependency for mesh packaging
try:
    import trimesh as _trimesh
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False

def klein_to_poincare(V: np.ndarray) -> np.ndarray:
    """Convert points V (n x 3) in the Klein ball to the Poincaré ball."""
    V = np.asarray(V, dtype=float)
    r2 = np.sum(V*V, axis=1)
    s = np.sqrt(np.maximum(0.0, 1.0 - r2))
    denom = (1.0 + s)[:, None]
    mask = (np.abs(r2 - 1.0) < 1e-14)
    denom[mask] = 1.0
    P = V / denom
    return P

def triangulated_mesh_from_hull(pts3: np.ndarray,
                                faces: List[List[int]],
                                refinement: int = 0,
                                model: str = "klein",
                                as_trimesh: bool = False):
    """Triangular mesh: fan-triangulate faces, refine by midpoints, optionally convert to Poincaré."""
    V = np.asarray(pts3, dtype=float).copy()
    tris, _ = triangulate_faces_with_face_id(faces)
    F = np.array(tris, dtype=int)
    def refine_once(V, F):
        Vlist = V.tolist()
        edge_mid = {}
        def midpoint(i, j):
            a, b = min(i, j), max(i, j)
            key = (a, b)
            if key in edge_mid:
                return edge_mid[key]
            m = 0.5*(V[a] + V[b])
            idx = len(Vlist)
            Vlist.append(m.tolist())
            edge_mid[key] = idx
            return idx
        newF = []
        for (i, j, k) in F:
            ij = midpoint(i, j); jk = midpoint(j, k); ki = midpoint(k, i)
            newF.extend([(i, ij, ki), (ij, j, jk), (ki, jk, k), (ij, jk, ki)])
        return np.array(Vlist, dtype=float), np.array(newF, dtype=int)
    for _ in range(max(0, refinement)):
        V, F = refine_once(V, F)
    if model.lower().startswith("poin"):
        V = klein_to_poincare(V)
    if as_trimesh and _HAS_TRIMESH:
        return _trimesh.Trimesh(vertices=V, faces=F, process=False)
    return V, F

def export_mesh_obj(filename: str, V: np.ndarray, F: np.ndarray):
    """Minimal OBJ exporter for (V, F)."""
    with open(filename, "w", encoding="utf-8") as f:
        for x, y, z in V:
            f.write(f"v {x:.10f} {y:.10f} {z:.10f}\n")
        for a, b, c in F:
            f.write(f"f {a+1} {b+1} {c+1}\n")
    return filename

def hull_to_mesh(points: List[complex],
                 add_north_pole: bool = False,
                 refinement: int = 0,
                 model: str = "poincare",
                 as_trimesh: bool = True):
    """High-level: planar points -> hull -> triangular mesh -> (optional) Poincaré conversion."""
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    if add_north_pole:
        pts3 = np.vstack([pts3, np.array([[0.0, 0.0, 1.0]])])
    hull = convex_hull_from_points_on_sphere(pts3)
    faces = hull["faces"]
    return triangulated_mesh_from_hull(pts3, faces, refinement=refinement, model=model, as_trimesh=as_trimesh)

# ---------- Optional viewer / renderer (pyrender) ----------

try:
    import pyrender as _pyrender
    _HAS_PYRENDER = True
except Exception:
    _HAS_PYRENDER = False

def _lambda_to_rgb_nm(lam_nm: float) -> Tuple[float, float, float]:
    """Approximate sRGB for a visible wavelength lam_nm in [380, 700]."""
    lam = lam_nm
    if lam < 380 or lam > 700: return (0.0, 0.0, 0.0)
    if lam < 440:
        t = (lam - 380) / (440 - 380); r, g, b = (-(t), 0.0, 1.0)
    elif lam < 490:
        t = (lam - 440) / (490 - 440); r, g, b = (0.0, t, 1.0)
    elif lam < 510:
        t = (lam - 490) / (510 - 490); r, g, b = (0.0, 1.0, 1.0 - t)
    elif lam < 580:
        t = (lam - 510) / (580 - 510); r, g, b = (t, 1.0, 0.0)
    elif lam < 645:
        t = (lam - 580) / (645 - 580); r, g, b = (1.0, 1.0 - t, 0.0)
    else:
        t = (lam - 645) / (700 - 645); r, g, b = (1.0, 0.0, 0.0)
    if lam < 420:
        factor = 0.3 + 0.7*(lam - 380)/(420 - 380)
    elif lam > 645:
        factor = 0.3 + 0.7*(700 - lam)/(700 - 645)
    else:
        factor = 1.0
    gamma = 0.8
    def correct(c):
        c = max(0.0, min(1.0, c * factor))
        return c ** gamma
    return (correct(r), correct(g), correct(b))

def _soap_iridescence_vertex_colors(mesh: "_trimesh.Trimesh",
                                    camera_pos: np.ndarray,
                                    n_film: float = 1.33,
                                    n_env: float = 1.0,
                                    t0_nm: float = 500.0,
                                    t_amp_nm: float = 300.0,
                                    swirl_scale: float = 5.0,
                                    wavelengths_nm: Tuple[float,float,float] = (450.0, 530.0, 610.0),
                                    alpha: float = 0.35) -> np.ndarray:
    """Per-vertex RGBA colors simulating thin-film iridescence (CPU approximation)."""
    V = mesh.vertices
    N = mesh.vertex_normals
    if len(N) == 0:
        mesh.rezero(); mesh.remove_duplicate_faces()
        N = mesh.vertex_normals
    cam = np.asarray(camera_pos, dtype=float).reshape(3)
    V_to_cam = cam[None, :] - V
    def _norm_rows(A):
        L = np.linalg.norm(A, axis=1) + 1e-12
        return A / L[:, None], L
    Vdir, _ = _norm_rows(V_to_cam)
    Nn, _ = _norm_rows(N)
    cos_theta = np.abs(np.sum(Nn * Vdir, axis=1))
    x, y, z = V[:,0], V[:,1], V[:,2]
    phi = np.arctan2(y, x)
    theta = np.arccos(np.clip(z/np.maximum(1e-12, np.linalg.norm(V, axis=1)), -1, 1))
    t_nm = t0_nm + t_amp_nm * (np.sin(swirl_scale*phi) * np.cos(0.5*swirl_scale*theta))
    lambdas = np.array(list(wavelengths_nm), dtype=float)
    r12 = ((n_env - n_film)/(n_env + n_film))**2
    r23 = r12
    sqrt_term = math.sqrt(max(0.0, r12 * r23))
    C = np.zeros((V.shape[0], 3), dtype=float)
    for lam_nm in lambdas:
        delta = (4.0*math.pi*n_film/lam_nm) * t_nm * cos_theta
        R = r12 + r23 + 2.0*sqrt_term*np.cos(delta)
        R = np.clip(R, 0.0, 1.0)
        rgb = np.array(_lambda_to_rgb_nm(float(lam_nm)))[None, :]
        C += (R[:, None]) * rgb
    C = np.clip(C / (C.max(axis=1, keepdims=True) + 1e-12), 0.0, 1.0)
    A = np.full((V.shape[0], 1), np.clip(alpha, 0.0, 1.0), dtype=float)
    rgba = np.concatenate([C, A], axis=1)
    rgba_u8 = (rgba * 255.0 + 0.5).astype(np.uint8)
    return rgba_u8

def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0,0,1.0])) -> np.ndarray:
    """Right-handed look-at 4x4 pose matrix."""
    eye = np.asarray(eye, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    up = np.asarray(up, dtype=float).reshape(3)
    z = eye - target; z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.vstack([x, y, z]).T; T[:3, 3] = eye
    return T

def render_snapshot(points: List[complex],
                    add_north_pole: bool = False,
                    refinement: int = 1,
                    model: str = "poincare",
                    width: int = 1280,
                    height: int = 960,
                    background: Tuple[float,float,float,float] = (1.0, 1.0, 1.0, 1.0),
                    camera_pos: Optional[Tuple[float,float,float]] = None,
                    target: Optional[Tuple[float,float,float]] = None,
                    transparency_alpha: float = 0.35,
                    light_setup: str = "three_point",
                    soap_iridescence: bool = True,
                    # ---- style knobs ----
                    base_color_rgba: Tuple[float,float,float,float] = (0.6,0.8,1.0,0.35),
                    metallic: float = 0.05,
                    roughness: float = 0.05,
                    n_film: float = 1.33,
                    n_env: float = 1.0,
                    t0_nm: float = 500.0,
                    t_amp_nm: float = 300.0,
                    swirl_scale: float = 5.0,
                    wavelengths_nm: Tuple[float,float,float] = (450.0, 530.0, 610.0),
                    # lights
                    key_intensity: float = 4.0,
                    fill_intensity: float = 1.5,
                    rim_intensity: float = 2.5,
                    ring_n: int = 8,
                    ring_radius: float = 4.5,
                    ring_z: float = 2.0,
                    ring_intensity: float = 2.0,
                    outfile: str = "render.png") -> str:
    """
    Render a PNG snapshot of the convex hull mesh with lights and optional soap-film iridescence.
    Requires: trimesh + pyrender.
    """
    if not (_HAS_TRIMESH and _HAS_PYRENDER):
        raise RuntimeError("render_snapshot requires trimesh and pyrender (pip install trimesh pyrender).")
    mesh = hull_to_mesh(points, add_north_pole=add_north_pole, refinement=refinement, model=model, as_trimesh=True)
    if camera_pos is None: camera_pos = (2.7, 2.1, 1.6)
    if target is None: target = (0.0, 0.0, 0.0)
    if soap_iridescence:
        vcols = _soap_iridescence_vertex_colors(mesh, np.array(camera_pos),
                                                n_film=n_film, n_env=n_env,
                                                t0_nm=t0_nm, t_amp_nm=t_amp_nm,
                                                swirl_scale=swirl_scale,
                                                wavelengths_nm=wavelengths_nm,
                                                alpha=transparency_alpha)
        mesh.visual.vertex_colors = vcols
        material = _pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, max(0.05, transparency_alpha)],
            metallicFactor=float(metallic), roughnessFactor=float(roughness),
            alphaMode="BLEND", doubleSided=True
        )
    else:
        mesh.visual.vertex_colors = (np.array(base_color_rgba)[None, :] * 255.0 + 0.5).astype(np.uint8)
        material = _pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[float(base_color_rgba[0]), float(base_color_rgba[1]), float(base_color_rgba[2]), max(0.05, float(base_color_rgba[3]))],
            metallicFactor=float(metallic), roughnessFactor=float(roughness),
            alphaMode="BLEND", doubleSided=True
        )
    pm = _pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene = _pyrender.Scene(bg_color=np.array(background, dtype=float))
    scene.add(pm)
    cam = _pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    cam_pose = _look_at(np.array(camera_pos), np.array(target))
    scene.add(cam, pose=cam_pose)
    def add_three_point(scene):
        light_key = _pyrender.DirectionalLight(color=np.ones(3), intensity=float(key_intensity))
        light_fill = _pyrender.DirectionalLight(color=np.ones(3), intensity=float(fill_intensity))
        light_rim = _pyrender.DirectionalLight(color=np.ones(3), intensity=float(rim_intensity))
        scene.add(light_key, pose=_look_at(np.array([3, 2, 2]), np.zeros(3)))
        scene.add(light_fill, pose=_look_at(np.array([-2, 3, 1]), np.zeros(3)))
        scene.add(light_rim, pose=_look_at(np.array([-3, -2, 2.5]), np.zeros(3)))
    def add_ring(scene, n=8, radius=4.5, z=2.0, intensity=2.0):
        for k in range(int(n)):
            ang = 2*np.pi*k/n
            pos = np.array([radius*np.cos(ang), radius*np.sin(ang), z])
            scene.add(_pyrender.PointLight(color=np.ones(3), intensity=float(intensity)),
                      pose=_look_at(pos, np.zeros(3)))
    if light_setup == "ring":
        add_ring(scene, n=ring_n, radius=ring_radius, z=ring_z, intensity=ring_intensity)
    else:
        add_three_point(scene)
    r = _pyrender.OffscreenRenderer(viewport_width=int(width), viewport_height=int(height))
    color, _ = r.render(scene, flags=_pyrender.RenderFlags.RGBA)
    import imageio
    imageio.imwrite(outfile, color)
    r.delete()
    return outfile
