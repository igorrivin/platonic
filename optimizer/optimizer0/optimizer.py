"""
optimizer.py
------------
L-BFGS optimizer for maximizing the hyperbolic volume of an ideal polyhedron determined by
planar boundary points (complex numbers in \widehat{C}, with INF=None for infinity).

Features:
- Anchors: fix the first three vertices to (0, 1, INF) by default; any subset can be frozen via a mask.
- Dynamic vs fixed combinatorics:
  * dynamic=True: at each evaluation, rebuild the spherical convex hull and compute the (exact) gradient
                  for the current combinatorics (piecewise-smooth objective).
  * dynamic=False: freeze the hull faces from the initial configuration and use the analytic gradient
                   on that fixed triangulation (fastest within a combinatorics cell).
- Optional soft barriers:
  * separation barrier to keep vertices apart: sum_{i<j} -log(|z_i - z_j| + eps)
  * L2 regularizer on coordinates to keep them bounded if desired.

Returns:
- optimized complex points (with anchors preserved),
- final volume, number of iterations, SciPy status, and a small history.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
import numpy as np

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from pointset_to_fuchsian import (
    INF, stereographic_inverse, convex_hull_from_points_on_sphere,
    volume_and_gradient_from_hull, volume_and_gradient_from_pointset
)

@dataclass
class OptimizeOptions:
    dynamic: bool = True                  # recompute hull each evaluation
    add_north_pole: bool = False          # append north pole to points before hull
    apex_index: Optional[int] = None      # apex for pulling triangulation (None -> 0)
    barrier_weight: float = 1e-3          # separation barrier weight
    barrier_eps: float = 1e-6             # epsilon inside -log(r+eps)
    l2_weight: float = 0.0                # optional L2 regularizer weight on free coords
    maxiter: int = 500
    ftol: float = 1e-9
    gtol: float = 1e-7
    history: bool = True
    verbose: bool = True

@dataclass
class OptimizeResult:
    points_opt: List[complex]
    volume: float
    niter: int
    status: int
    message: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False

def _pack_free(free: np.ndarray) -> np.ndarray:
    return np.concatenate([free.real, free.imag])

def _unpack_free(x: np.ndarray) -> np.ndarray:
    M = x.size // 2
    re = x[:M]; im = x[M:]
    return re + 1j*im

def _apply_mask(points: List[complex], x_free: np.ndarray, var_idx: List[int]) -> List[complex]:
    z = list(points)
    free = _unpack_free(x_free)
    for k, idx in enumerate(var_idx):
        z[idx] = free[k]
    return z

def _barrier_penalty_and_grad(z: List[complex], var_idx: List[int], w: float, eps: float) -> Tuple[float, np.ndarray]:
    """Compute separation barrier penalty and gradient wrt free vars. Skip any pair with INF."""
    M = len(var_idx)
    g = np.zeros((M, 2), dtype=float)
    if w <= 0.0 or M == 0:
        return 0.0, g
    # map from global index -> local free index
    pos = {idx: k for k, idx in enumerate(var_idx)}
    val = 0.0
    finite_idx = [i for i in range(len(z)) if z[i] is not INF]
    # naive O(N^2) pairs; for thousands of points that's fine for a barrier (can be thinned if needed)
    for i in finite_idx:
        zi = z[i]
        for j in finite_idx:
            if j <= i: continue
            zj = z[j]
            ui = zi - zj
            r = abs(ui)
            rr = r + eps
            val += -math.log(rr)
            # gradient only if both i,j are free
            if i in pos or j in pos:
                # d/dr (-log(r+eps)) = -1/(r+eps)
                if r > 0.0:
                    coeff = -1.0 / rr / r
                    dx = ui.real; dy = ui.imag
                    gi = np.array([coeff * dx, coeff * dy])
                    gj = -gi
                    if i in pos:
                        g[pos[i]] += gi
                    if j in pos:
                        g[pos[j]] += gj
    return w*val, w*g

def _l2_penalty_and_grad(x_free: np.ndarray, w: float) -> Tuple[float, np.ndarray]:
    if w <= 0.0:
        return 0.0, np.zeros_like(_unpack_free(x_free).view(float))
    # x_free packs real and imag separately; L2 on complex entries => sum |z|^2 = sum re^2+im^2
    p = 0.5 * w * float(np.dot(x_free, x_free))
    g = w * x_free
    return p, g.reshape(-1,)

def optimize_vertices_lbfgs(points_init: List[complex],
                            var_mask: Optional[List[bool]] = None,
                            options: OptimizeOptions = OptimizeOptions()) -> OptimizeResult:
    """
    Maximize hyperbolic volume starting from 'points_init' (complex numbers and INF). L-BFGS-B.
    'var_mask': list of booleans (len N), True for variables to optimize. By default:
        anchors = (0,1,INF) at indices (0,1,2) are frozen; the rest are free.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("optimize_vertices_lbfgs requires SciPy (pip install scipy).")

    N = len(points_init)
    z0 = list(points_init)
    # Default var_mask: freeze first 3, optimize the rest
    if var_mask is None:
        var_mask = [False, False, False] + [True]*(N-3)
    var_idx = [i for i, m in enumerate(var_mask) if m]

    # Initial free vector
    free_init = np.array([z0[i] for i in var_idx], dtype=complex)
    x0 = _pack_free(free_init)

    hist: List[Dict[str, Any]] = []

    # Precompute faces if fixed combinatorics
    faces_fixed = None
    pts3_fixed = None
    if not options.dynamic:
        pts3_fixed = np.vstack([stereographic_inverse(z) for z in z0])
        if options.add_north_pole:
            pts3_fixed = np.vstack([pts3_fixed, np.array([[0.0, 0.0, 1.0]], dtype=float)])
        hull = convex_hull_from_points_on_sphere(pts3_fixed)
        faces_fixed = hull["faces"]

    def f_and_g(x):
        z = _apply_mask(z0, x, var_idx)

        # Volume + gradient
        if options.dynamic:
            V, G = volume_and_gradient_from_pointset(z, apex_index=options.apex_index,
                                                     add_north_pole=options.add_north_pole,
                                                     variable_mask=[i in var_idx for i in range(N)])
        else:
            # fixed faces
            if pts3_fixed is None or faces_fixed is None:
                raise RuntimeError("Fixed-hull mode missing precomputed faces.")
            V, G_full = volume_and_gradient_from_hull(pts3_fixed, faces_fixed, apex=options.apex_index,
                                                      variable_mask=[i in var_idx for i in range(N)])
            # recompute G for the free subset (G_full is N x 2)
            G = G_full

        # Extract gradient for free variables
        G_free = np.array([G[i] for i in var_idx], dtype=float)   # shape (M, 2)
        grad_vec = np.concatenate([G_free[:,0], G_free[:,1]])

        # Barriers
        b_val, b_grad_mat = _barrier_penalty_and_grad(z, var_idx, options.barrier_weight, options.barrier_eps)
        b_grad = np.concatenate([b_grad_mat[:,0], b_grad_mat[:,1]])

        # L2 regularizer
        p_val, p_grad = _l2_penalty_and_grad(x, options.l2_weight)

        # We maximize V => minimize (-V + penalties)
        f = -V + b_val + p_val
        g = -grad_vec + b_grad + p_grad

        if options.history:
            hist.append({"V": V, "f": f, "barrier": b_val, "l2": p_val})

        return f, g

    # Callback for logging
    def cb(xk):
        if options.verbose and hist:
            h = hist[-1]
            print(f"iter {len(hist)}: V={h['V']:.10f}, f={h['f']:.6e}, barrier={h['barrier']:.3e}")

    res = minimize(lambda x: f_and_g(x), x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': options.maxiter, 'ftol': options.ftol, 'gtol': options.gtol, 'disp': options.verbose},
                   callback=cb)

    # Build final points
    z_opt = _apply_mask(z0, res.x, var_idx)
    # Final volume (evaluate once more for reporting)
    if options.dynamic:
        V_final, _ = volume_and_gradient_from_pointset(z_opt, apex_index=options.apex_index,
                                                       add_north_pole=options.add_north_pole,
                                                       variable_mask=[False]*N)
    else:
        V_final, _ = volume_and_gradient_from_hull(pts3_fixed, faces_fixed, apex=options.apex_index,
                                                   variable_mask=[False]*N)

    return OptimizeResult(points_opt=z_opt, volume=V_final, niter=res.nit,
                          status=res.status, message=res.message,
                          history=hist, success=res.success)
