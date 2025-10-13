#!/usr/bin/env python3
"""
Example: off-center hexagon (z_i = 1 + ω^i, ω = e^{πi/3}) with a north-pole apex (hexagonal pyramid).

- Builds the convex hull on S^2
- Computes shears → holonomy generators
- Prints an invariant trace field signature (heuristic)
- Computes dihedral angles on hull edges
- Computes hyperbolic volume via a pulling triangulation (Bloch–Wigner)
"""

import os, sys, math
import numpy as np

# Make sure the repository root is on sys.path when run as: python examples/off_center_hexagon.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointset_to_fuchsian import (
    stereographic_inverse, convex_hull_from_points_on_sphere, build_penner_rivin_from_hull,
    invariant_trace_field_signature, edge_dihedrals, volume_from_hull
)
from rivin_holonomy import generators_from_triangulation

def main():
    w = complex(math.cos(math.pi/3), math.sin(math.pi/3))
    pts2 = [1 + (w**i) for i in range(6)]

    # Back-project to S^2 and add the north pole apex
    pts3 = np.vstack([stereographic_inverse(z) for z in pts2])
    north = np.array([[0.0, 0.0, 1.0]])
    pts3 = np.vstack([pts3, north])  # apex index = len(pts2) = 6

    hull = convex_hull_from_points_on_sphere(pts3)
    faces = hull["faces"]

    # Build triangulation and shears; compute holonomy generators
    T, Z, info = build_penner_rivin_from_hull(pts3, faces)
    gens = generators_from_triangulation(T, Z, root=0)
    Gamma = [M for (_,_,_,M) in gens]

    print(f"# triangles: {T.F}")
    print(f"# generators: {len(Gamma)}")
    print(f"# triangulation edges (with shear): {len(Z)}")

    # Invariant trace field signature (heuristic)
    sig = invariant_trace_field_signature(Gamma)
    print("All tr(g^2) integers? ", sig["all_integers"])
    print("First few tr(g^2):   ", [round(x, 8) for x in sig["traces_sq"][:8]])

    # Dihedral angles on hull edges
    dih = edge_dihedrals(info)  # dict {edge: angle}
    if dih:
        vals = list(dih.values())
        print(f"Dihedrals: min={min(vals):.6f}, max={max(vals):.6f}, mean={sum(vals)/len(vals):.6f} (radians)")
    else:
        print("No nonzero dihedral angles detected (flat hull?).")

    # Volume via Bloch–Wigner (pulling triangulation from apex)
    V = volume_from_hull(pts3, faces, apex=len(pts2))
    print(f"Hyperbolic volume ≈ {V:.10f}")

if __name__ == "__main__":
    main()
