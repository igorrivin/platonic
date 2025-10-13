# Arithmetic of Platonic Surfaces — Code & Cookbook

This repository accompanies the note **“Arithmetic of Platonic Surfaces”** and provides a compact, practical pipeline that turns a **planar point set** into:

- an **ideal polyhedron** (inverse stereographic projection + convex hull),
- a **pleated surface** with **complex shear–bend** parameters on edges,
- a **uniformizing Fuchsian group** (Penner–Rivin algorithm),
- its **invariant trace field** (from traces of squares),
- the **hyperbolic volume** of the polyhedron (Bloch–Wigner),
- and **visualization**: triangular mesh export, subdivision, Klein→Poincaré mapping, and a PNG **renderer** with lights, transparency, and soap‑film iridescence.

Install the basics:
```bash
pip install numpy scipy mpmath
# for viewer:
pip install trimesh pyrender imageio
```

## Files

- `rivin_holonomy.py` – Penner–Rivin engine: ideal triangulation + shears → holonomy matrices.
- `pointset_to_fuchsian.py` – Stereographic → convex hull → shears → holonomy → dihedrals → volume → mesh → renderer.
- `arithmetic_of_platonic_surfaces_full.tex` – LaTeX note (self‑contained) with the theory + worked example.
- `examples/` – Small drivers.

## Mathematics in one page

For an oriented edge \(x\to y\) with adjacent vertices \(a\) (left) and \(b\) (right),
\[
\tau_{xy}=\log\!\left(\frac{(a-x)(b-y)}{(a-y)(b-x)}\right)=\log\frac{M(a)}{M(b)},\ \ M(z)=\frac{z-x}{z-y}.
\]
- \(\Re\tau\) is the **shear**; \(\Im\tau\) is the **exterior dihedral**.
- Holonomy uses fixed \(R,L\) and edge matrices \(X(Z)\) with \(Z=\Re\tau\).
- \(k(\Gamma)=\mathbb{Q}(\{\mathrm{tr}(g^2)\})\). For cusped lattices \(k(\Gamma)=\mathbb Q \iff\) modular commensurability (**arithmetic**).
- Volume via a pulling triangulation and the Bloch–Wigner \(D(z)\).

## Quickstart

```python
from pointset_to_fuchsian import group_from_pointset, invariant_trace_field_signature, volume_from_pointset

points = [0+0j, 1+0j, 0.7+0.4j, -0.3+0.9j, -0.8-0.1j, 0.1-0.5j]
out = group_from_pointset(points)
sig = invariant_trace_field_signature(out["Gamma_generators"])
print("All tr(g^2) integers? ", sig["all_integers"])

V = volume_from_pointset(points)
print("Volume ≈", V)
```

## Off‑center hexagon (hexagonal pyramid)

Let \(\omega=e^{\pi i/3}\), \(z_i=1+\omega^i\). Add the north pole to make a pyramid:
```python
import math
from pointset_to_fuchsian import volume_from_pointset, render_snapshot

w = complex(math.cos(math.pi/3), math.sin(math.pi/3))
points = [1 + (w**i) for i in range(6)]

print("Volume ≈", volume_from_pointset(points, add_north_pole=True))

render_snapshot(points, add_north_pole=True, refinement=2, model="poincare",
                transparency_alpha=0.35, light_setup="three_point",
                soap_iridescence=True, outfile="hex_pyramid_iridescent.png")
```

## Mesh export & Poincaré visualization

```python
from pointset_to_fuchsian import hull_to_mesh, export_mesh_obj
import math

w = complex(math.cos(math.pi/3), math.sin(math.pi/3))
points = [1 + (w**i) for i in range(6)]

V, F = hull_to_mesh(points, add_north_pole=True, refinement=2, model="poincare", as_trimesh=False)
export_mesh_obj("hex_pyramid.obj", V, F)
```

## Viewer (lights, transparency, iridescence)

```python
from pointset_to_fuchsian import render_snapshot
import math

w = complex(math.cos(math.pi/3), math.sin(math.pi/3))
points = [1 + (w**i) for i in range(6)]

png = render_snapshot(points, add_north_pole=True, refinement=2, model="poincare",
                      width=1400, height=1050,
                      camera_pos=(2.7,2.1,1.6), target=(0,0,0),
                      transparency_alpha=0.35,
                      light_setup="three_point",
                      soap_iridescence=True,
                      outfile="hex_pyramid_iridescent.png")
print("Saved:", png)
```

### Style knobs

- **Iridescence physics**: `n_film`, `n_env`, `t0_nm`, `t_amp_nm`, `swirl_scale`, `wavelengths_nm`.
- **Material**: `transparency_alpha`, `base_rgba`, `metallic`, `roughness`.
- **Camera**: `camera_pos`, `target`, `fov_y_deg`.
- **Lights**: `light_setup="three_point"` with `key_intensity`/`fill_intensity`/`rim_intensity`, or `light_setup="ring"` with `ring_n`/`ring_radius`/`ring_z`/`ring_intensity`.
- **Mesh smoothing**: `refinement` (midpoint subdivision per round).

## Citations
Epstein–Marden (1986); Bonahon (1996); Maclachlan–Reid (2003); Takeuchi (1977); Rivin (2005).

License: MIT.


---

## Examples

Run from the repo root:

```bash
python examples/off_center_hexagon.py         # compute holonomy, field signature, dihedrals, volume
python examples/render_hex_pyramid.py         # render two PNGs with different style settings
```

Outputs:
- `off_center_hexagon.py` prints a list of `tr(g^2)`, shows a field signature, dihedral stats, and the volume.
- `render_hex_pyramid.py` saves `hex_pyramid_glass.png` and `hex_pyramid_iridescent.png` (requires `trimesh`, `pyrender`, `imageio`).
