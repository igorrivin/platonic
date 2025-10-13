"""
volume_with_gradients.py
------------------------
Compute hyperbolic volume and its analytic gradients with respect to point positions.
Based on the mathematical formulation described, with proper handling of infinity.
"""

import numpy as np
import math
import cmath
from typing import List, Union, Optional, Tuple

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

# Sentinel for infinity
INF = None

def stereographic_inverse(z: Union[complex, None]) -> np.ndarray:
    """Inverse stereographic projection from complex plane to unit sphere."""
    if z is None or z is INF:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    
    x, y = z.real, z.imag
    d = x*x + y*y + 1.0
    return np.array([2*x/d, 2*y/d, (x*x + y*y - 1.0)/d], dtype=float)

def cross_ratio_safe(a, x, b, y):
    """
    Möbius-invariant cross ratio CR(a,x; b,y) = M(a)/M(b), M(z)=(z-x)/(z-y),
    allowing any of a,x,b,y to be INF (None).
    """
    # Handle cases where either x or y is INF first
    if x is INF and y is INF:
        raise ValueError("Degenerate configuration: x=y=INF")
    if x is INF:
        # M(z) ~ 1/(z - y)
        if a is INF or b is INF:
            raise ValueError("Unexpected INF pattern with x=INF.")
        return (b - y) / (a - y)
    if y is INF:
        # M(z) ~ (z - x)
        if a is INF or b is INF:
            raise ValueError("Unexpected INF pattern with y=INF.")
        return (a - x) / (b - x)
    
    # Now x,y are finite
    if a is INF and b is INF:
        return 1.0 + 0.0j
    if a is INF:
        return (b - y) / (b - x)
    if b is INF:
        return (a - x) / (a - y)
    
    # All finite
    return ((a - x)*(b - y)) / ((a - y)*(b - x))

def tetra_shape_safe(z0, z1, z2, z3):
    """Compute shape parameter with INF support."""
    CR = cross_ratio_safe(z0, z1, z2, z3)
    # Avoid branch cut on negative real axis
    if abs(CR.imag) < 1e-15 and CR.real < 0:
        CR += 1e-15j
    return CR

def bloch_wigner(z: complex) -> float:
    """Bloch-Wigner dilogarithm D(z)."""
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath required for Bloch-Wigner")
    return float(mp.im(mp.polylog(2, z)) + mp.arg(1 - z) * mp.log(abs(z)))

def bloch_wigner_gradient_factor(z: complex) -> complex:
    """
    Compute G(z) such that dV = Im(G(z) * dz) for volume V = D(z).
    
    G(z) = [-s_α + s_β]/z - s_β/(z-1) + s_γ/(1-z)
    where s_α = log|2sin(arg z)|, etc.
    """
    eps = 1e-15
    
    # Compute s_α = log|2sin(arg z)| using branch-agnostic formula
    if abs(z) < eps:
        s_alpha = -np.inf  # Will contribute 0 due to division by z
    else:
        s_alpha = math.log(2 * abs(z.imag) / abs(z) + eps)
    
    # Compute s_β = log|2sin(arg(1-1/z))|
    if abs(z) < eps:
        s_beta = 0.0
    else:
        w = 1 - 1/z
        if abs(w) < eps:
            s_beta = -np.inf
        else:
            s_beta = math.log(2 * abs(w.imag) / abs(w) + eps)
    
    # Compute s_γ = log|2sin(arg(1-z))|
    w = 1 - z
    if abs(w) < eps:
        s_gamma = -np.inf
    else:
        s_gamma = math.log(2 * abs(w.imag) / abs(w) + eps)
    
    # Compute G(z)
    G = complex(0)
    
    if abs(z) > eps:
        G += (-s_alpha + s_beta) / z
    
    if abs(z - 1) > eps:
        G -= s_beta / (z - 1)
    
    if abs(1 - z) > eps:
        G += s_gamma / (1 - z)
    
    return G

def dlog_cross_ratio_coeffs(a, x, b, y):
    """
    Compute coefficients for d(log CR) = sum of c_v * dv/v for v in {a,x,b,y}.
    Returns dict mapping position to coefficient.
    
    d(log z) = (da-dx)/(a-x) + (db-dy)/(b-y) - (da-dy)/(a-y) - (db-dx)/(b-x)
    """
    coeffs = {}
    
    # Handle INF cases carefully
    eps = 1e-15
    
    # Coefficient for da
    if a is not INF:
        c_a = complex(0)
        if x is not INF and abs(a - x) > eps:
            c_a += 1 / (a - x)
        if y is not INF and abs(a - y) > eps:
            c_a -= 1 / (a - y)
        coeffs[id(a)] = coeffs.get(id(a), 0) + c_a
    
    # Coefficient for dx
    if x is not INF:
        c_x = complex(0)
        if a is not INF and abs(a - x) > eps:
            c_x -= 1 / (a - x)
        if b is not INF and abs(b - x) > eps:
            c_x -= 1 / (b - x)
        coeffs[id(x)] = coeffs.get(id(x), 0) + c_x
    
    # Coefficient for db
    if b is not INF:
        c_b = complex(0)
        if y is not INF and abs(b - y) > eps:
            c_b += 1 / (b - y)
        if x is not INF and abs(b - x) > eps:
            c_b -= 1 / (b - x)
        coeffs[id(b)] = coeffs.get(id(b), 0) + c_b
    
    # Coefficient for dy
    if y is not INF:
        c_y = complex(0)
        if b is not INF and abs(b - y) > eps:
            c_y -= 1 / (b - y)
        if a is not INF and abs(a - y) > eps:
            c_y += 1 / (a - y)
        coeffs[id(y)] = coeffs.get(id(y), 0) + c_y
    
    return coeffs

def volume_and_gradient_from_hull(pts3: np.ndarray, hull, 
                                  apex_index: Optional[int] = None,
                                  variable_mask: Optional[List[bool]] = None) -> Tuple[float, np.ndarray]:
    """
    Compute volume and gradients from precomputed hull.
    
    Parameters:
    -----------
    pts3 : array of shape (n, 3)
        Points on the unit sphere
    hull : ConvexHull object
        Precomputed convex hull
    apex_index : int, optional
        Index of apex for volume computation
    variable_mask : list of bool, optional
        Which points are variables (can move). If None, all points are variables.
    
    Returns:
    --------
    volume : float
    gradients : array of shape (n, 2) where gradients[i] = [dV/dx_i, dV/dy_i]
    """
    n_points = len(pts3)
    
    if variable_mask is None:
        variable_mask = [True] * n_points
    
    # Project to complex plane
    z_points = []
    for p in pts3:
        z = None if abs(p[2] - 1.0) < 1e-14 else complex(p[0]/(1-p[2]), p[1]/(1-p[2]))
        z_points.append(z)
    
    if apex_index is None:
        apex_index = 0
    
    # Initialize gradients
    gradients = np.zeros((n_points, 2), dtype=float)
    
    # Use simplices directly
    triangles = hull.simplices
    
    # Compute volume and accumulate gradients
    total_volume = 0.0
    
    for tri in triangles:
        a, b, c = tri
        if apex_index in (a, b, c):
            continue
        
        # Get vertex positions
        z0 = z_points[apex_index]
        z1 = z_points[a]
        z2 = z_points[b]
        z3 = z_points[c]
        
        # Compute shape and volume contribution
        shape = tetra_shape_safe(z0, z1, z2, z3)
        vol_contrib = bloch_wigner(shape)
        total_volume += abs(vol_contrib)
        
        # Skip gradient if volume is too small
        if abs(vol_contrib) < 1e-15:
            continue
        
        # Get gradient factor G(z)
        G = bloch_wigner_gradient_factor(shape)
        
        # Get d(log CR) coefficients
        coeffs = dlog_cross_ratio_coeffs(z0, z1, z2, z3)
        
        # Map vertex indices to positions
        vertex_map = {
            id(z0): apex_index,
            id(z1): a,
            id(z2): b,
            id(z3): c
        }
        
        # Accumulate gradient contributions
        sign = 1.0 if vol_contrib > 0 else -1.0
        
        for z_id, coeff in coeffs.items():
            if z_id not in vertex_map:
                continue
            
            idx = vertex_map[z_id]
            if not variable_mask[idx] or z_points[idx] is INF:
                continue
            
            # Weight for this vertex: W_v = G(z) * z * coeff
            W = sign * G * shape * coeff
            
            # dV = Im(W * dv) = Im(W) * d(Re v) + Re(W) * d(Im v)
            gradients[idx, 0] += W.imag  # dV/dx
            gradients[idx, 1] += W.real  # dV/dy
    
    return total_volume, gradients

def volume_and_gradient_from_pointset(points: List[Union[complex, None]], 
                                      apex_index: Optional[int] = None,
                                      add_north_pole: bool = False,
                                      variable_mask: Optional[List[bool]] = None) -> Tuple[float, np.ndarray]:
    """
    Compute volume and gradients from complex points.
    
    This is a wrapper that:
    1. Converts points to sphere via inverse stereographic projection
    2. Computes convex hull
    3. Calls volume_and_gradient_from_hull
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy required for convex hull")
    
    # Convert to sphere points
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    
    if add_north_pole:
        pts3 = np.vstack([pts3, np.array([[0.0, 0.0, 1.0]])])
        if variable_mask is not None:
            variable_mask = variable_mask + [False]
    
    # Compute hull
    hull = ConvexHull(pts3)
    
    # Compute volume and gradients
    return volume_and_gradient_from_hull(pts3, hull, apex_index, variable_mask)