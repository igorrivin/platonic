#!/usr/bin/env python3
"""
Off-center hexagon -> hexagonal pyramid:
- convex hull on S^2,
- shear-bend via cross-ratios -> Penner-Rivin holonomy,
- invariant trace field signature,
- dihedral angles on hull edges,
- hyperbolic volume (Bloch-Wigner).
"""
import math, numpy as np
from pointset_to_fuchsian import (
    stereographic_inverse, convex_hull_from_points_on_sphere, build_penner_rivin_from_hull,
    invariant_trace_field_signature, volume_from_pointset, edge_dihedrals
)
from rivin_holonomy import generators_from_triangulation

def main():
    # Hexagon points: z_i = 1 + w^i, w = exp(pi i / 3)
    w = complex(math.cos(math.pi/3), math.sin(math.pi/3))
    points = [1 + (w**i) for i in range(6)]
    # Build sphere points and add the north pole to form a hexagonal pyramid
    pts3 = np.vstack([stereographic_inverse(z) for z in points])
    north = np.array([[0.0, 0.0, 1.0]])
    pts3 = np.vstack([pts3, north])
    # Convex hull faces
    hull = convex_hull_from_points_on_sphere(pts3)
    faces = hull["faces"]
    print(f"#vertices = {pts3.shape[0]}, #faces = {len(faces)}")
    # Build triangulation + shears and holonomy
    T, Z, info = build_penner_rivin_from_hull(pts3, faces)
    gens = generators_from_triangulation(T, Z, root=0)
    Gamma_gens = [M for (_,_,_,M) in gens]
    print(f"#generators (non-tree dual edges) = {len(Gamma_gens)}")
    # Invariant trace field (heuristic)
    sig = invariant_trace_field_signature(Gamma_gens)
    print("All tr(g^2) integers?", sig["all_integers"])
    print("Sample tr(g^2):", [round(x,6) for x in sig["traces_sq"][:8]])
    # Dihedral angles (radians) on hull edges
    dih = edge_dihedrals(info)
    sample = list(dih.items())[:8]
    print("Sample dihedrals (edge -> angle):")
    for e, ang in sample:
        print(" ", e, "->", ang)
    # Hyperbolic volume via Bloch-Wigner (pulling triangulation)
    V = volume_from_pointset(points, add_north_pole=True)
    print("Hexagonal pyramid volume ~", V)

if __name__ == "__main__":
    main()
