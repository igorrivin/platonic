#!/usr/bin/env python3
"""
Unit test: 4 points (tetrahedron) under the (0,1,INF) gauge.

We optimize over the single free vertex z \in C \ {0,1} to maximize
the ideal tetrahedron volume. The maximum is at the regular shape,
with z ≈ e^{iπ/3} and volume v3 ≈ 1.0149416064.

If SciPy is available, we use L-BFGS-B on the 2D variable (Re z, Im z).
Otherwise we fall back to a simple gradient-ascent with backtracking.
"""

import os, sys, math, cmath
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from pointset_to_fuchsian import INF, volume_and_gradient_from_pointset

def volume_and_grad_for_z(z):
    # Anchors fixed at 0, 1, INF; the fourth point is z
    points = [0+0j, 1+0j, INF, z]
    var_mask = [False, False, False, True]
    V, G = volume_and_gradient_from_pointset(points, variable_mask=var_mask)
    # Gradient for z only (dx, dy)
    gx, gy = G[3, 0], G[3, 1]
    return V, np.array([gx, gy], dtype=float)

def dihedral_angles(z):
    # The three dihedral angles are Arg(z), Arg(1 - 1/z), Arg(1 - z) (mod π)
    a = abs(cmath.phase(z))
    b = abs(cmath.phase(1 - 1/z))
    c = abs(cmath.phase(1 - z))
    return a, b, c

def optimize_tetrahedron(z0=0.3 + 0.4j, maxiter=300):
    if _HAS_SCIPY:
        def fg(xy):
            z = xy[0] + 1j*xy[1]
            V, g = volume_and_grad_for_z(z)
            return -V, -g  # maximize V
        res = minimize(lambda xy: fg(xy), np.array([z0.real, z0.imag]), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-9, 'disp': True})
        z_opt = res.x[0] + 1j*res.x[1]
        V_opt, _ = volume_and_grad_for_z(z_opt)
        return z_opt, V_opt, res.nit, res.success
    else:
        # Simple gradient ascent with backtracking
        z = complex(z0)
        alpha = 0.5
        for it in range(maxiter):
            V, g = volume_and_grad_for_z(z)
            step = alpha
            improved = False
            for _ in range(20):
                z_new = (z.real + step*g[0]) + 1j*(z.imag + step*g[1])
                # avoid z near 0 or 1
                if abs(z_new) < 1e-6 or abs(z_new - 1) < 1e-6:
                    step *= 0.5
                    continue
                V_new, _ = volume_and_grad_for_z(z_new)
                if V_new > V:
                    z = z_new
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
        V_fin, _ = volume_and_grad_for_z(z)
        return z, V_fin, it+1, True

def main():
    print("Optimizing single tetrahedron from initial z0...")
    z0 = 0.3 + 0.4j
    z_opt, V_opt, iters, ok = optimize_tetrahedron(z0=z0)
    print("\n--- Results ---")
    print("z* ≈ {:.10f} {:+.10f}i".format(z_opt.real, z_opt.imag))
    print("Volume ≈ {:.10f}".format(V_opt))
    a, b, c = dihedral_angles(z_opt)
    print("Dihedral angles (radians): a≈{:.10f}, b≈{:.10f}, c≈{:.10f}".format(a, b, c))
    print("Should be close to π/3 ≈ {:.10f}".format(math.pi/3))
    print("Iterations:", iters, "| success:", ok)

    # Reference values
    v3 = 1.014941606409653625  # regular ideal tetrahedron volume
    print("\nReference volume v3 ≈", v3)

if __name__ == "__main__":
    main()
