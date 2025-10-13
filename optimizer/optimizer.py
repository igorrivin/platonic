"""
optimizer.py
------------
L-BFGS optimizer for maximizing the hyperbolic volume of ideal polyhedra.
Uses analytic gradients and handles infinity properly.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Optional, Tuple, Dict, Any
import time

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from volume_with_gradients import (
    INF, volume_and_gradient_from_pointset, 
    volume_and_gradient_from_hull, 
    stereographic_inverse, ConvexHull
)

@dataclass
class OptimizeConfig:
    """Configuration for the optimizer."""
    # Optimization settings
    method: str = 'L-BFGS-B'  # Optimization method
    maxiter: int = 500         # Maximum iterations
    ftol: float = 1e-9         # Function tolerance
    gtol: float = 1e-7         # Gradient tolerance
    
    # Problem settings
    dynamic_hull: bool = True   # Recompute hull at each step
    apex_index: Optional[int] = None  # Apex for volume computation
    add_north_pole: bool = False      # Add north pole to points
    
    # Regularization
    barrier_weight: float = 0.0   # Separation barrier weight
    barrier_eps: float = 1e-6     # Epsilon for log barrier
    l2_weight: float = 0.0        # L2 regularization weight
    
    # Output
    verbose: bool = True          # Print progress
    history: bool = True          # Keep optimization history

@dataclass
class OptimizeResult:
    """Result of optimization."""
    points_opt: List[Union[complex, None]]  # Optimized points
    volume_opt: float                        # Final volume
    success: bool                            # Optimization successful?
    niter: int                              # Number of iterations
    nfev: int                               # Number of function evaluations
    message: str                            # Termination message
    history: List[Dict[str, Any]] = field(default_factory=list)
    initial_volume: float = 0.0             # Initial volume for comparison

def pack_variables(points: List[Union[complex, None]], 
                   var_indices: List[int]) -> np.ndarray:
    """Pack variable points into a real array for optimizer."""
    values = []
    for idx in var_indices:
        z = points[idx]
        if z is None or z is INF:
            raise ValueError(f"Variable point {idx} cannot be infinity")
        values.append(z.real)
        values.append(z.imag)
    return np.array(values)

def unpack_variables(x: np.ndarray, n_vars: int) -> List[complex]:
    """Unpack real array back to complex points."""
    points = []
    for i in range(n_vars):
        re = x[2*i]
        im = x[2*i + 1]
        points.append(complex(re, im))
    return points

def apply_variables(base_points: List[Union[complex, None]], 
                    var_values: List[complex],
                    var_indices: List[int]) -> List[Union[complex, None]]:
    """Apply variable values to base points."""
    points = base_points.copy()
    for i, idx in enumerate(var_indices):
        points[idx] = var_values[i]
    return points

def separation_barrier(points: List[Union[complex, None]], 
                      weight: float, eps: float) -> float:
    """Compute separation barrier: -weight * sum log(|zi - zj| + eps)."""
    if weight <= 0:
        return 0.0
    
    total = 0.0
    n = len(points)
    
    for i in range(n):
        if points[i] is None or points[i] is INF:
            continue
        for j in range(i + 1, n):
            if points[j] is None or points[j] is INF:
                continue
            
            dist = abs(points[i] - points[j])
            total += -weight * np.log(dist + eps)
    
    return total

def separation_barrier_gradient(points: List[Union[complex, None]], 
                               var_indices: List[int],
                               weight: float, eps: float) -> np.ndarray:
    """Compute gradient of separation barrier."""
    n_vars = len(var_indices)
    grad = np.zeros(2 * n_vars)
    
    if weight <= 0:
        return grad
    
    # Map from global index to variable index
    var_map = {idx: i for i, idx in enumerate(var_indices)}
    
    n = len(points)
    for i in range(n):
        if points[i] is None or points[i] is INF:
            continue
        
        for j in range(n):
            if i == j or points[j] is None or points[j] is INF:
                continue
            
            diff = points[i] - points[j]
            dist = abs(diff)
            
            if dist > 0 and i in var_map:
                # d/dzi(-log|zi-zj|) = -(zi-zj)/|zi-zj|^2 / (1 + eps/|zi-zj|)
                factor = -weight / (dist * (dist + eps))
                var_idx = var_map[i]
                grad[2*var_idx] += factor * diff.real
                grad[2*var_idx + 1] += factor * diff.imag
    
    return grad

def optimize_volume(initial_points: List[Union[complex, None]],
                    var_indices: Optional[List[int]] = None,
                    config: Optional[OptimizeConfig] = None) -> OptimizeResult:
    """
    Optimize the hyperbolic volume by moving variable points.
    
    Parameters:
    -----------
    initial_points : list of complex or None
        Initial point configuration (None represents infinity)
    var_indices : list of int, optional
        Indices of points that can move. If None, all finite points are variables.
    config : OptimizeConfig, optional
        Optimization configuration
    
    Returns:
    --------
    OptimizeResult with optimized points and diagnostics
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy required for optimization")
    
    if config is None:
        config = OptimizeConfig()
    
    # Determine variable indices
    if var_indices is None:
        var_indices = [i for i, p in enumerate(initial_points) 
                      if p is not None and p is not INF]
    
    n_vars = len(var_indices)
    if n_vars == 0:
        raise ValueError("No variable points to optimize")
    
    # Create variable mask
    n_points = len(initial_points)
    var_mask = [False] * n_points
    for idx in var_indices:
        var_mask[idx] = True
    
    # Pre-compute hull if not dynamic
    fixed_hull = None
    fixed_pts3 = None
    if not config.dynamic_hull:
        pts3 = np.vstack([stereographic_inverse(z) for z in initial_points])
        if config.add_north_pole:
            pts3 = np.vstack([pts3, np.array([[0.0, 0.0, 1.0]])])
        fixed_hull = ConvexHull(pts3)
        fixed_pts3 = pts3
    
    # Optimization history
    history = []
    iteration_count = 0
    
    def objective_and_gradient(x):
        nonlocal iteration_count
        iteration_count += 1
        
        # Unpack variables
        var_values = unpack_variables(x, n_vars)
        points = apply_variables(initial_points, var_values, var_indices)
        
        # Compute volume and gradient
        if config.dynamic_hull:
            vol, grad_full = volume_and_gradient_from_pointset(
                points, config.apex_index, config.add_north_pole, var_mask
            )
        else:
            # Update only variable points in pts3
            pts3 = fixed_pts3.copy()
            for i, idx in enumerate(var_indices):
                pts3[idx] = stereographic_inverse(var_values[i])
            
            vol, grad_full = volume_and_gradient_from_hull(
                pts3, fixed_hull, config.apex_index, var_mask
            )
        
        # Extract gradients for variables only
        grad = np.zeros(2 * n_vars)
        for i, idx in enumerate(var_indices):
            grad[2*i] = grad_full[idx, 0]
            grad[2*i + 1] = grad_full[idx, 1]
        
        # Add regularization
        reg_value = 0.0
        
        # Separation barrier
        if config.barrier_weight > 0:
            reg_value += separation_barrier(points, config.barrier_weight, config.barrier_eps)
            grad += separation_barrier_gradient(points, var_indices, 
                                               config.barrier_weight, config.barrier_eps)
        
        # L2 regularization
        if config.l2_weight > 0:
            reg_value += 0.5 * config.l2_weight * np.sum(x**2)
            grad += config.l2_weight * x
        
        # Total objective (negative because we maximize volume)
        obj = -(vol - reg_value)
        grad = -grad
        
        # Record history
        if config.history and iteration_count % 10 == 0:
            history.append({
                'iter': iteration_count,
                'volume': vol,
                'objective': -obj,
                'grad_norm': np.linalg.norm(grad)
            })
        
        if config.verbose and iteration_count % 20 == 0:
            print(f"Iter {iteration_count}: volume = {vol:.6f}, |grad| = {np.linalg.norm(grad):.3e}")
        
        return obj, grad
    
    # Initial point
    x0 = pack_variables(initial_points, var_indices)
    
    # Compute initial volume
    initial_volume, _ = volume_and_gradient_from_pointset(
        initial_points, config.apex_index, config.add_north_pole, var_mask
    )
    
    if config.verbose:
        print(f"Initial volume: {initial_volume:.6f}")
        print(f"Optimizing {n_vars} variable points...")
    
    # Run optimization
    start_time = time.time()
    
    result = minimize(
        objective_and_gradient,
        x0,
        method=config.method,
        jac=True,
        options={
            'maxiter': config.maxiter,
            'ftol': config.ftol,
            'gtol': config.gtol,
            'disp': config.verbose
        }
    )
    
    elapsed = time.time() - start_time
    
    # Extract final configuration
    var_values_opt = unpack_variables(result.x, n_vars)
    points_opt = apply_variables(initial_points, var_values_opt, var_indices)
    
    # Compute final volume
    volume_opt, _ = volume_and_gradient_from_pointset(
        points_opt, config.apex_index, config.add_north_pole, var_mask
    )
    
    if config.verbose:
        print(f"\nOptimization completed in {elapsed:.2f} seconds")
        print(f"Final volume: {volume_opt:.6f} (improvement: {volume_opt/initial_volume:.3f}x)")
    
    return OptimizeResult(
        points_opt=points_opt,
        volume_opt=volume_opt,
        success=result.success,
        niter=result.nit,
        nfev=result.nfev,
        message=result.message,
        history=history,
        initial_volume=initial_volume
    )