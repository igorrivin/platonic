"""
volume_lobachevsky.py
---------------------
Compute hyperbolic volume using stereographic projection and Lobachevsky function.

Algorithm:
1. Rotate polyhedron so one vertex is at north pole
2. Stereographically project from north pole to complex plane
3. For each triangle, compute angles and sum Lobachevsky functions
4. Total volume is sum over all triangles

This approach is gradient-friendly and works for any triangulated polyhedron.
"""

import numpy as np
import cmath
import math
from typing import List, Tuple, Optional

try:
    import mpmath as mp
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False

def rotate_to_north_pole(pts3: np.ndarray, vertex_idx: int) -> np.ndarray:
    """
    Rotate points on sphere so that vertex_idx is at north pole (0,0,1).
    
    Parameters:
    -----------
    pts3 : array of shape (n, 3)
        Points on unit sphere
    vertex_idx : int
        Index of vertex to rotate to north pole
        
    Returns:
    --------
    rotated : array of shape (n, 3)
        Rotated points
    """
    p = pts3[vertex_idx]
    x, y, z = p
    
    # If already at north pole, no rotation needed
    if abs(z - 1.0) < 1e-14:
        return pts3.copy()
    
    # Compute rotation axis (perpendicular to p and north pole)
    # axis = p × (0,0,1) = (-y, x, 0)
    axis = np.array([-y, x, 0.0])
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-14:
        # p is at south pole, rotate around x-axis
        axis = np.array([1.0, 0.0, 0.0])
        angle = np.pi
    else:
        axis = axis / axis_norm
        # Angle between p and north pole
        angle = np.arccos(np.clip(z, -1.0, 1.0))
    
    # Create rotation matrix using Rodrigues formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Apply rotation to all points
    rotated = np.dot(pts3, R.T)
    
    return rotated

def stereographic_from_north(p: np.ndarray) -> Optional[complex]:
    """
    Stereographic projection from north pole.
    Points are projected onto z=0 plane.
    North pole itself maps to infinity (None).
    """
    x, y, z = p
    if abs(z - 1.0) < 1e-14:
        return None  # North pole maps to infinity
    
    # Project from north pole: (x,y,z) -> (x/(1-z), y/(1-z))
    denom = 1.0 - z
    return complex(x/denom, y/denom)

def triangle_angles(z1: complex, z2: complex, z3: complex) -> Tuple[float, float, float]:
    """
    Compute the three angles of a triangle with vertices z1, z2, z3.
    
    Returns:
    --------
    (alpha, beta, gamma) : angles at vertices z1, z2, z3 respectively
    """
    # Vectors from each vertex to the other two
    v21 = z2 - z1
    v31 = z3 - z1
    v12 = z1 - z2
    v32 = z3 - z2
    v13 = z1 - z3
    v23 = z2 - z3
    
    # Angle at z1
    alpha = abs(cmath.phase(v31) - cmath.phase(v21))
    if alpha > np.pi:
        alpha = 2*np.pi - alpha
    
    # Angle at z2  
    beta = abs(cmath.phase(v32) - cmath.phase(v12))
    if beta > np.pi:
        beta = 2*np.pi - beta
    
    # Angle at z3
    gamma = abs(cmath.phase(v13) - cmath.phase(v23))
    if gamma > np.pi:
        gamma = 2*np.pi - gamma
    
    return alpha, beta, gamma

def lobachevsky(x: float) -> float:
    """
    Lobachevsky function Λ(x) = -∫_0^x log|2sin(t)| dt
    
    Using mpmath for accuracy.
    """
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath required for Lobachevsky function")
    
    # For small x, use series expansion to avoid numerical issues
    if abs(x) < 1e-10:
        return 0.0
    
    # Use mpmath's implementation
    # Λ(x) = Im[Li_2(e^{2ix})] where Li_2 is the dilogarithm
    return float(mp.im(mp.polylog(2, mp.exp(2j * x))))

def volume_from_angles(angles: List[float]) -> float:
    """
    Compute volume contribution from triangle angles.
    Volume = Λ(α) + Λ(β) + Λ(γ) - Λ(π)
    """
    if not angles:
        return 0.0
    
    volume = sum(lobachevsky(angle) for angle in angles)
    volume -= lobachevsky(np.pi)
    
    return volume

def volume_and_gradient_lobachevsky(pts3: np.ndarray, triangles: List[Tuple[int, int, int]], 
                                   north_pole_idx: int = 0,
                                   variable_mask: Optional[List[bool]] = None) -> Tuple[float, np.ndarray]:
    """
    Compute volume and gradients using Lobachevsky function method.
    
    Parameters:
    -----------
    pts3 : array of shape (n, 3)
        Points on unit sphere
    triangles : list of (i, j, k) tuples
        Triangle faces of the polyhedron
    north_pole_idx : int
        Index of vertex to use as north pole for projection
    variable_mask : list of bool, optional
        Which vertices are variables
        
    Returns:
    --------
    volume : float
        Total volume
    gradients : array of shape (n, 2)
        Gradients with respect to stereographic coordinates
    """
    n = len(pts3)
    
    # Rotate so north_pole_idx is at north pole
    rotated = rotate_to_north_pole(pts3, north_pole_idx)
    
    # Stereographic projection from north pole
    z_coords = []
    for p in rotated:
        z = stereographic_from_north(p)
        z_coords.append(z)
    
    # Initialize
    total_volume = 0.0
    gradients = np.zeros((n, 2), dtype=float)
    
    # Process each triangle
    for tri_idx, (i, j, k) in enumerate(triangles):
        # Skip if any vertex is at infinity (north pole)
        if any(z_coords[v] is None for v in [i, j, k]):
            continue
        
        zi, zj, zk = z_coords[i], z_coords[j], z_coords[k]
        
        # Compute angles
        alpha_i, alpha_j, alpha_k = triangle_angles(zi, zj, zk)
        
        # Volume contribution
        vol = lobachevsky(alpha_i) + lobachevsky(alpha_j) + lobachevsky(alpha_k) - lobachevsky(np.pi)
        total_volume += vol
        
        # Gradient computation
        # dΛ/dx = -log|2sin(x)|
        # We need d(angle)/d(z) for each angle and vertex
        
        # Gradient of angle with respect to vertex positions
        # angle = arccos((v1·v2)/(|v1||v2|)) where v1, v2 are edge vectors
        
        # For angle at zi:
        # alpha_i = angle between (zj-zi) and (zk-zi)
        v1 = zj - zi
        v2 = zk - zi
        
        # d(alpha_i)/d(zi) involves both edge vectors
        # This is complex but computable
        
        # Simplified gradient using finite differences for now
        # TODO: Implement analytic gradient
        
        if variable_mask is None or variable_mask[i]:
            # Finite difference for zi
            eps = 1e-8
            
            # Perturb x
            zi_px = zi + eps
            alpha_px, _, _ = triangle_angles(zi_px, zj, zk)
            d_alpha_dx = (alpha_px - alpha_i) / eps
            gradients[i, 0] += -math.log(abs(2*math.sin(alpha_i))) * d_alpha_dx
            
            # Perturb y
            zi_py = zi + 1j*eps
            alpha_py, _, _ = triangle_angles(zi_py, zj, zk)
            d_alpha_dy = (alpha_py - alpha_i) / eps
            gradients[i, 1] += -math.log(abs(2*math.sin(alpha_i))) * d_alpha_dy
    
    return total_volume, gradients

def volume_from_complex_points(points: List[complex], triangles: List[Tuple[int, int, int]]) -> float:
    """
    Compute volume from points already in complex plane.
    Assumes one point is at infinity (north pole).
    """
    total_volume = 0.0
    
    for i, j, k in triangles:
        # Skip if any vertex is None (infinity)
        if points[i] is None or points[j] is None or points[k] is None:
            continue
            
        # Compute angles
        angles = triangle_angles(points[i], points[j], points[k])
        
        # Add volume contribution
        vol = sum(lobachevsky(angle) for angle in angles) - lobachevsky(np.pi)
        total_volume += vol
    
    return total_volume