"""
test_optimizer.py
-----------------
Test the volume optimizer with various examples.
"""

import numpy as np
import math
from optimizer import optimize_volume, OptimizeConfig, INF
from volume_with_gradients import volume_and_gradient_from_pointset

def test_fixed_anchors():
    """Test optimization with fixed (0, 1, ∞) anchors."""
    print("=" * 60)
    print("Test 1: Optimize with fixed anchors (0, 1, ∞)")
    print("=" * 60)
    
    # Initial configuration: 3 fixed anchors + 5 random points
    np.random.seed(42)
    n_free = 5
    
    # Generate random initial points in a reasonable range
    free_points = []
    for i in range(n_free):
        r = 0.3 + 0.4 * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        z = r * complex(np.cos(theta), np.sin(theta))
        free_points.append(z)
    
    # Full point set: anchors + free points
    initial_points = [0+0j, 1+0j, INF] + free_points
    
    # Variable indices (only the free points)
    var_indices = list(range(3, 3 + n_free))
    
    print(f"Initial points:")
    for i, p in enumerate(initial_points):
        fixed = "FIXED" if i < 3 else "VAR"
        print(f"  {i}: {p} [{fixed}]")
    
    # Compute initial volume
    var_mask = [False, False, False] + [True] * n_free
    initial_vol, _ = volume_and_gradient_from_pointset(initial_points, variable_mask=var_mask)
    print(f"\nInitial volume: {initial_vol:.6f}")
    
    # Optimize
    config = OptimizeConfig(
        maxiter=200,
        verbose=True,
        barrier_weight=1e-3,  # Small separation barrier
        dynamic_hull=True
    )
    
    result = optimize_volume(initial_points, var_indices, config)
    
    print(f"\nOptimized points:")
    for i, p in enumerate(result.points_opt):
        fixed = "FIXED" if i < 3 else "VAR"
        print(f"  {i}: {p} [{fixed}]")
    
    print(f"\nVolume improvement: {result.volume_opt / initial_vol:.3f}x")
    
    return result

def test_hexagon_pyramid():
    """Test with hexagon + north pole (should be near-optimal already)."""
    print("\n" + "=" * 60)
    print("Test 2: Hexagon pyramid (should be near-optimal)")
    print("=" * 60)
    
    # Regular hexagon centered at origin
    w = complex(math.cos(math.pi/3), math.sin(math.pi/3))
    hex_points = [w**i for i in range(6)]
    
    # Use first 3 as anchors, optimize the rest
    initial_points = hex_points
    var_indices = [3, 4, 5]  # Only optimize half the hexagon
    
    print(f"Initial hexagon vertices:")
    for i, p in enumerate(initial_points):
        fixed = "FIXED" if i < 3 else "VAR"
        print(f"  {i}: {p:.4f} [{fixed}]")
    
    # Compute initial volume (with north pole added)
    var_mask = [i in var_indices for i in range(6)]
    initial_vol, _ = volume_and_gradient_from_pointset(
        initial_points, add_north_pole=True, variable_mask=var_mask
    )
    print(f"\nInitial volume (with north pole): {initial_vol:.6f}")
    
    # Optimize
    config = OptimizeConfig(
        maxiter=100,
        verbose=True,
        add_north_pole=True,
        dynamic_hull=False  # Fix combinatorics for this symmetric case
    )
    
    result = optimize_volume(initial_points, var_indices, config)
    
    print(f"\nOptimized vertices:")
    for i, p in enumerate(result.points_opt):
        fixed = "FIXED" if i < 3 else "VAR"
        print(f"  {i}: {p:.4f} [{fixed}]")
    
    print(f"\nVolume improvement: {result.volume_opt / initial_vol:.6f}")
    print("(Should be close to 1.0 since hexagon is already optimal)")
    
    return result

def test_gradient_correctness():
    """Verify gradients using finite differences."""
    print("\n" + "=" * 60)
    print("Test 3: Gradient verification")
    print("=" * 60)
    
    # Simple configuration
    points = [0+0j, 1+0j, 0.5+0.5j, -0.3+0.8j]
    var_mask = [False, False, True, True]
    
    # Compute analytic gradient
    vol, grad = volume_and_gradient_from_pointset(points, variable_mask=var_mask)
    
    print(f"Volume: {vol:.6f}")
    print(f"Analytic gradients:")
    for i, mask in enumerate(var_mask):
        if mask:
            print(f"  Point {i}: dV/dx = {grad[i,0]:.6f}, dV/dy = {grad[i,1]:.6f}")
    
    # Verify with finite differences
    print(f"\nFinite difference verification (eps = 1e-6):")
    eps = 1e-6
    
    for i, mask in enumerate(var_mask):
        if not mask:
            continue
        
        # Perturb x
        points_p = points.copy()
        points_p[i] = points[i] + eps
        vol_p, _ = volume_and_gradient_from_pointset(points_p, variable_mask=var_mask)
        fd_x = (vol_p - vol) / eps
        
        # Perturb y
        points_p = points.copy()
        points_p[i] = points[i] + 1j * eps
        vol_p, _ = volume_and_gradient_from_pointset(points_p, variable_mask=var_mask)
        fd_y = (vol_p - vol) / eps
        
        print(f"  Point {i}: dV/dx = {fd_x:.6f} (err: {abs(fd_x - grad[i,0]):.2e}), " +
              f"dV/dy = {fd_y:.6f} (err: {abs(fd_y - grad[i,1]):.2e})")

if __name__ == "__main__":
    # Run tests
    result1 = test_fixed_anchors()
    result2 = test_hexagon_pyramid()
    test_gradient_correctness()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    
    # Show optimization history for first test
    if result1.history:
        print(f"\nOptimization history (Test 1):")
        print("Iter  Volume      |grad|")
        print("-" * 30)
        for h in result1.history:
            print(f"{h['iter']:4d}  {h['volume']:.6f}  {h['grad_norm']:.3e}")