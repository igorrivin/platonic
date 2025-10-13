#!/usr/bin/env python3
"""
Example: maximize hyperbolic volume with L-BFGS under the (0,1,INF) gauge.
- Starts from random free vertices
- Optionally uses dynamic hull recomputation
- Prints final volume and saves a PNG render (if pyrender is available)
"""
import os, sys, math, numpy as np
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointset_to_fuchsian import INF, render_snapshot
from optimizer import optimize_vertices_lbfgs, OptimizeOptions

def main():
    # Anchors (0, 1, INF)
    anchors = [0+0j, 1+0j, INF]

    # Random initial free points (e.g., 60 points in a disk)
    rng = np.random.default_rng(0)
    M = 60
    r = 0.7 * np.sqrt(rng.random(M))
    theta = 2*np.pi*rng.random(M)
    free_init = list(r*np.cos(theta) + 1j*r*np.sin(theta))

    points0 = anchors + free_init
    var_mask = [False, False, False] + [True]*M

    opts = OptimizeOptions(dynamic=True, barrier_weight=1e-3, barrier_eps=1e-6,
                           l2_weight=0.0, maxiter=200, verbose=True)

    res = optimize_vertices_lbfgs(points0, var_mask=var_mask, options=opts)
    print("\n--- Optimization summary ---")
    print("success:", res.success, "| status:", res.status, "| iters:", res.niter)
    print("final volume ≈", res.volume)
    # Try rendering (may require trimesh+pyrender)
    try:
        png = render_snapshot(res.points_opt, refinement=1, model="poincare",
                              soap_iridescence=True, outfile="optimized_polyhedron.png")
        print("Saved render to:", png)
    except Exception as e:
        print("(Render skipped:", e, ")")

if __name__ == "__main__":
    main()
