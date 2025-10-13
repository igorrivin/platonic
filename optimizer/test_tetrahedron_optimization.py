"""
test_tetrahedron_optimization.py
---------------------------------
Test that the optimizer finds the regular ideal tetrahedron.
The regular ideal tetrahedron has vertices at 0, 1, ∞, and a fourth point
that maximizes volume. The optimal fourth point should form an equilateral
triangle with 0 and 1 in the complex plane.
"""

import numpy as np
import math
import cmath
from optimizer import optimize_volume, OptimizeConfig, INF
from volume_with_gradients import volume_and_gradient_from_pointset

# Known volume of regular ideal tetrahedron
REGULAR_IDEAL_TET_VOLUME = 1.0149416064096537

def test_find_regular_tetrahedron():
    """Test optimization starting from random fourth point."""
    print("=" * 60)
    print("Test: Find regular ideal tetrahedron")
    print("=" * 60)
    
    # The optimal fourth point should be at one of two positions
    # that form an equilateral triangle with 0 and 1
    optimal_z4_1 = complex(0.5, math.sqrt(3)/2)  # exp(i*π/3)
    optimal_z4_2 = complex(0.5, -math.sqrt(3)/2) # exp(-i*π/3)
    
    print(f"Known optimal positions for 4th vertex:")
    print(f"  Option 1: {optimal_z4_1}")
    print(f"  Option 2: {optimal_z4_2}")
    print(f"Expected volume: {REGULAR_IDEAL_TET_VOLUME}")
    
    # Try several random starting points
    np.random.seed(123)
    n_trials = 5
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}:")
        print("-" * 40)
        
        # Random initial position for 4th point
        r = 0.3 + 0.7 * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        z4_init = r * complex(np.cos(theta), np.sin(theta))
        
        # Points: 0, 1, ∞ (fixed) + fourth point (variable)
        initial_points = [0+0j, 1+0j, INF, z4_init]
        var_indices = [3]  # Only the 4th point is variable
        
        print(f"Initial 4th point: {z4_init}")
        
        # Compute initial volume
        var_mask = [False, False, False, True]
        initial_vol, _ = volume_and_gradient_from_pointset(initial_points, variable_mask=var_mask)
        print(f"Initial volume: {initial_vol:.6f}")
        
        # Optimize with tight tolerances
        config = OptimizeConfig(
            maxiter=100,
            verbose=False,
            ftol=1e-12,
            gtol=1e-10,
            barrier_weight=0.0,  # No barrier needed for single point
            dynamic_hull=False   # Combinatorics won't change for tetrahedron
        )
        
        result = optimize_volume(initial_points, var_indices, config)
        
        z4_opt = result.points_opt[3]
        print(f"Optimized 4th point: {z4_opt}")
        print(f"Final volume: {result.volume_opt:.10f}")
        print(f"Expected:     {REGULAR_IDEAL_TET_VOLUME:.10f}")
        print(f"Relative error: {abs(result.volume_opt - REGULAR_IDEAL_TET_VOLUME) / REGULAR_IDEAL_TET_VOLUME:.2e}")
        
        # Check which optimal position we found
        dist1 = abs(z4_opt - optimal_z4_1)
        dist2 = abs(z4_opt - optimal_z4_2)
        
        if dist1 < 0.01:
            print(f"Found optimal position 1 (distance: {dist1:.6f})")
        elif dist2 < 0.01:
            print(f"Found optimal position 2 (distance: {dist2:.6f})")
        else:
            print(f"Warning: Not at expected position! Distances: {dist1:.6f}, {dist2:.6f}")
        
        # Verify it's on the unit circle centered at 0.5
        center = 0.5 + 0j
        radius = abs(z4_opt - center)
        expected_radius = math.sqrt(3) / 2
        print(f"Distance from (0.5, 0): {radius:.6f} (expected: {expected_radius:.6f})")

def test_gradient_at_optimum():
    """Verify gradient is zero at the optimal configuration."""
    print("\n" + "=" * 60)
    print("Test: Gradient at optimal configuration")
    print("=" * 60)
    
    # Optimal configuration
    z4_opt = complex(0.5, math.sqrt(3)/2)
    points = [0+0j, 1+0j, INF, z4_opt]
    var_mask = [False, False, False, True]
    
    vol, grad = volume_and_gradient_from_pointset(points, variable_mask=var_mask)
    
    print(f"Volume at optimum: {vol:.10f}")
    print(f"Expected:          {REGULAR_IDEAL_TET_VOLUME:.10f}")
    print(f"Gradient at optimum: dV/dx = {grad[3,0]:.2e}, dV/dy = {grad[3,1]:.2e}")
    print(f"Gradient norm: {np.linalg.norm(grad[3]):.2e}")
    
    # Also test the other optimal position
    z4_opt2 = complex(0.5, -math.sqrt(3)/2)
    points2 = [0+0j, 1+0j, INF, z4_opt2]
    
    vol2, grad2 = volume_and_gradient_from_pointset(points2, variable_mask=var_mask)
    
    print(f"\nAlternative optimal position:")
    print(f"Volume: {vol2:.10f}")
    print(f"Gradient: dV/dx = {grad2[3,0]:.2e}, dV/dy = {grad2[3,1]:.2e}")
    print(f"Gradient norm: {np.linalg.norm(grad2[3]):.2e}")

def test_symmetry():
    """Test that the two optimal positions have the same volume."""
    print("\n" + "=" * 60)
    print("Test: Symmetry of optimal positions")
    print("=" * 60)
    
    z4_1 = complex(0.5, math.sqrt(3)/2)
    z4_2 = complex(0.5, -math.sqrt(3)/2)
    
    points1 = [0+0j, 1+0j, INF, z4_1]
    points2 = [0+0j, 1+0j, INF, z4_2]
    
    var_mask = [False, False, False, True]
    
    vol1, _ = volume_and_gradient_from_pointset(points1, variable_mask=var_mask)
    vol2, _ = volume_and_gradient_from_pointset(points2, variable_mask=var_mask)
    
    print(f"Volume with z4 = {z4_1}: {vol1:.10f}")
    print(f"Volume with z4 = {z4_2}: {vol2:.10f}")
    print(f"Difference: {abs(vol1 - vol2):.2e}")
    print(f"Both should equal: {REGULAR_IDEAL_TET_VOLUME:.10f}")

if __name__ == "__main__":
    test_find_regular_tetrahedron()
    test_gradient_at_optimum()
    test_symmetry()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)