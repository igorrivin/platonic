"""
Test the corrected gradient implementation from pointset_to_fuchsian.py
"""

import numpy as np
import math
from pointset_to_fuchsian import INF, volume_and_gradient_from_pointset

# Known regular ideal tetrahedron volume
REGULAR_IDEAL_TET_VOLUME = 1.0149416064096537

def test_gradient_at_optimum():
    """Test that gradients are zero at the optimal tetrahedron configuration."""
    print("Testing gradients at optimal tetrahedron configuration:")
    print("=" * 60)
    
    # Optimal configuration: vertices at 0, 1, ∞, and exp(iπ/3)
    z4_opt = complex(0.5, math.sqrt(3)/2)
    points = [0+0j, 1+0j, INF, z4_opt]
    var_mask = [False, False, False, True]  # Only z4 is variable
    
    V, grad = volume_and_gradient_from_pointset(points, variable_mask=var_mask)
    
    print(f"Configuration: 0, 1, ∞, {z4_opt}")
    print(f"Volume: {V:.10f}")
    print(f"Expected: {REGULAR_IDEAL_TET_VOLUME:.10f}")
    print(f"Volume error: {abs(V - REGULAR_IDEAL_TET_VOLUME):.2e}")
    print(f"\nGradient at z4:")
    print(f"  dV/dx = {grad[3,0]:.6e}")
    print(f"  dV/dy = {grad[3,1]:.6e}")
    print(f"  |grad| = {np.linalg.norm(grad[3]):.6e}")
    
    # Test the other optimal position
    print("\n" + "-"*40)
    z4_opt2 = complex(0.5, -math.sqrt(3)/2)
    points2 = [0+0j, 1+0j, INF, z4_opt2]
    
    V2, grad2 = volume_and_gradient_from_pointset(points2, variable_mask=var_mask)
    
    print(f"Alternative configuration: 0, 1, ∞, {z4_opt2}")
    print(f"Volume: {V2:.10f}")
    print(f"Gradient at z4:")
    print(f"  dV/dx = {grad2[3,0]:.6e}")
    print(f"  dV/dy = {grad2[3,1]:.6e}")
    print(f"  |grad| = {np.linalg.norm(grad2[3]):.6e}")

def test_finite_differences():
    """Compare analytic gradients with finite differences."""
    print("\n\nFinite difference verification:")
    print("=" * 60)
    
    # Test at a non-optimal point
    points = [0+0j, 1+0j, INF, 0.3+0.4j]
    var_mask = [False, False, False, True]
    
    V, grad = volume_and_gradient_from_pointset(points, variable_mask=var_mask)
    
    print(f"Test point: z4 = {points[3]}")
    print(f"Volume: {V:.6f}")
    print(f"Analytic gradient: dV/dx = {grad[3,0]:.6f}, dV/dy = {grad[3,1]:.6f}")
    
    # Finite differences
    eps = 1e-8
    
    # Perturb x
    points_px = points.copy()
    points_px[3] = points[3] + eps
    Vx, _ = volume_and_gradient_from_pointset(points_px, variable_mask=var_mask)
    fd_x = (Vx - V) / eps
    
    # Perturb y
    points_py = points.copy()
    points_py[3] = points[3] + 1j*eps
    Vy, _ = volume_and_gradient_from_pointset(points_py, variable_mask=var_mask)
    fd_y = (Vy - V) / eps
    
    print(f"\nFinite differences (eps={eps}):")
    print(f"  dV/dx = {fd_x:.6f} (error: {abs(fd_x - grad[3,0]):.2e})")
    print(f"  dV/dy = {fd_y:.6f} (error: {abs(fd_y - grad[3,1]):.2e})")

def test_optimization():
    """Test if optimizer can now find the optimal tetrahedron."""
    print("\n\nOptimization test:")
    print("=" * 60)
    
    from scipy.optimize import minimize
    
    # Start from random point
    z0 = 0.3 + 0.4j
    points_init = [0+0j, 1+0j, INF, z0]
    var_mask = [False, False, False, True]
    
    def objective_and_grad(xy):
        z = xy[0] + 1j*xy[1]
        points = [0+0j, 1+0j, INF, z]
        V, G = volume_and_gradient_from_pointset(points, variable_mask=var_mask)
        return -V, -np.array([G[3,0], G[3,1]])  # Minimize negative volume
    
    x0 = np.array([z0.real, z0.imag])
    
    print(f"Starting from: z4 = {z0}")
    V_init, _ = volume_and_gradient_from_pointset(points_init, variable_mask=var_mask)
    print(f"Initial volume: {V_init:.6f}")
    
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    z_opt = result.x[0] + 1j*result.x[1]
    V_opt = -result.fun
    
    print(f"\nOptimization result:")
    print(f"  Optimal z4: {z_opt}")
    print(f"  Final volume: {V_opt:.10f}")
    print(f"  Expected: {REGULAR_IDEAL_TET_VOLUME:.10f}")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.nit}")
    
    # Check which optimal position we found
    z4_expected1 = complex(0.5, math.sqrt(3)/2)
    z4_expected2 = complex(0.5, -math.sqrt(3)/2)
    dist1 = abs(z_opt - z4_expected1)
    dist2 = abs(z_opt - z4_expected2)
    
    if dist1 < 0.01:
        print(f"  Found expected position 1 (distance: {dist1:.6f})")
    elif dist2 < 0.01:
        print(f"  Found expected position 2 (distance: {dist2:.6f})")
    else:
        print(f"  WARNING: Not at expected position!")
        print(f"  Distance to position 1: {dist1:.6f}")
        print(f"  Distance to position 2: {dist2:.6f}")

if __name__ == "__main__":
    test_gradient_at_optimum()
    test_finite_differences()
    test_optimization()