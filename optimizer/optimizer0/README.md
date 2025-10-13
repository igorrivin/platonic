
# Arithmetic of Platonic Surfaces — Code & Cookbook

This repository accompanies the note **“Arithmetic of Platonic Surfaces”** and provides a compact, practical pipeline that turns a **planar point set** into:

- an **ideal polyhedron** (via inverse stereographic projection + convex hull),
- a **pleated surface** with **complex shear–bend** parameters on edges,
- a **uniformizing Fuchsian group** (Penner–Rivin algorithm),
- its **invariant trace field** (from traces of squares),
- the **exterior dihedral angles** on hull edges,
- and the **hyperbolic volume** of the polyhedron (via a pulling triangulation + Bloch–Wigner dilog).

It also includes **mesh export**, **midpoint subdivision**, **Klein→Poincaré** mapping, and an **offscreen PNG renderer** with lights, transparency, and a shaderless **soap‑film iridescence** effect.

## Install

```bash
pip install numpy scipy mpmath
# for mesh + renderer:
pip install trimesh pyrender imageio
```

## Files

- `rivin_holonomy.py` — Penner–Rivin engine: ideal triangulation + shears → holonomy matrices.
- `pointset_to_fuchsian.py` — Planar‑points pipeline (stereographic → convex hull → shears → holonomy),
  plus:
  - `edge_dihedrals(info)` — exterior dihedral angles (radians) on hull edges,
  - `volume_from_pointset(points, add_north_pole=False)` — hyperbolic volume,
  - mesh tools: `hull_to_mesh`, `klein_to_poincare`, `export_mesh_obj`, and
  - `render_snapshot(...)` — PNG renderer (lights, transparency, iridescence, style knobs).
- `arithmetic_of_platonic_surfaces_full.tex` — the self‑contained LaTeX note with theory + examples.

## Mathematics (one page)

For an oriented edge \(x\to y\) with adjacent vertices \(a\) (left) and \(b\) (right), the **complex shear–bend** is
\[
\tau_{xy} = \log\!\left(\frac{(a-x)(b-y)}{(a-y)(b-x)}\right) = \log\frac{M(a)}{M(b)},\quad M(z)=\frac{z-x}{z-y}.
\]
- \(\Re\tau\) is the **shear** (Penner \(Z\)-coordinate).  
- \(\Im\tau\) is the **exterior dihedral** (bending) angle.

The Penner–Rivin holonomy uses the blocks
\[
R=\begin{pmatrix}1&1\\-1&0\end{pmatrix},\;
L=\begin{pmatrix}0&-1\\1&1\end{pmatrix},\;
X(Z)=\begin{pmatrix}0&-e^{Z/2}\\ e^{-Z/2}&0\end{pmatrix}.
\]
The **invariant trace field** is \(k(\Gamma)=\mathbb Q(\{\mathrm{tr}(g^2)\})\). For cusped lattices, arithmetic \(\iff\) \(k(\Gamma)=\mathbb Q\).

For **volume**, triangulate the ideal polyhedron into ideal tetrahedra (pulling from a vertex) and sum \(D(z)\) (Bloch–Wigner) for each shape \(z\).

## Quickstart

```python
from pointset_to_fuchsian import group_from_pointset, invariant_trace_field_signature, volume_from_pointset

# generic points
points = [0+0j, 1+0j, 0.7+0.4j, -0.3+0.9j, -0.8-0.1j, 0.1-0.5j]

out = group_from_pointset(points)
Gamma_gens = out["Gamma_generators"]
sig = invariant_trace_field_signature(Gamma_gens)
print("All tr(g^2) integers? ", sig["all_integers"])

V = volume_from_pointset(points)  # may be 0 if the hull is planar
print("Volume ≈", V)
```

> If your points lie on a circle (planar hull): pass `add_north_pole=True` to `volume_from_pointset` to make a pyramid. To compute the group for that pyramid in 3D, append a very far point to the planar set (to approximate the north pole) or construct the hull directly in 3D and call the lower‑level builder.

## Mesh export, subdivision, and Poincaré visualization

```python
from pointset_to_fuchsian import hull_to_mesh, export_mesh_obj
import math

w = complex(math.cos(math.pi/3), math.sin(math.pi/3))
points = [1 + (w**i) for i in range(6)]  # off-center hexagon

# Build a hexagonal pyramid mesh (add apex), refine twice, convert to Poincaré
V, F = hull_to_mesh(points, add_north_pole=True, refinement=2, model="poincare", as_trimesh=False)
export_mesh_obj("hex_pyramid_poincare.obj", V, F)
```

## Viewer & PNG renderer (lights, transparency, iridescence)

```bash
pip install trimesh pyrender imageio
```

```python
from pointset_to_fuchsian import render_snapshot
import math

w = complex(math.cos(math.pi/3), math.sin(math.pi/3))
points = [1 + (w**i) for i in range(6)]

png = render_snapshot(points,
                      add_north_pole=True,     # makes a hexagonal pyramid
                      refinement=2,
                      model="poincare",
                      width=1400, height=1050,
                      background=(1,1,1,1),
                      camera_pos=(2.7,2.1,1.6), target=(0,0,0),
                      transparency_alpha=0.35,
                      light_setup="three_point",  # or "ring"
                      soap_iridescence=True,      # thin-film color effect
                      # ---- style knobs ----
                      base_color_rgba=(0.6,0.8,1.0,0.35),
                      metallic=0.05, roughness=0.05,
                      n_film=1.33, n_env=1.0,
                      t0_nm=500.0, t_amp_nm=300.0, swirl_scale=5.0,
                      wavelengths_nm=(450.0, 530.0, 610.0),
                      key_intensity=4.0, fill_intensity=1.5, rim_intensity=2.5,
                      ring_n=8, ring_radius=4.5, ring_z=2.0, ring_intensity=2.0,
                      outfile="hex_pyramid_iridescent.png")
print("Saved:", png)
```

### Style knobs (summary)

- **Material:** `base_color_rgba`, `metallic`, `roughness`, `transparency_alpha`.
- **Iridescence (thin film):** `soap_iridescence`, `n_film`, `n_env`, `t0_nm`, `t_amp_nm`, `swirl_scale`, `wavelengths_nm`.
- **Lights:** `light_setup` = `"three_point"` or `"ring"`; adjust `key_intensity`, `fill_intensity`, `rim_intensity`, or `ring_n`, `ring_radius`, `ring_z`, `ring_intensity`.
- **Camera / framing:** `camera_pos`, `target`, `width`, `height`, `background`.

## License

MIT. Please cite the note if you use this in a paper.


---

## Optimization (L‑BFGS) — maximize volume in the (0,1,∞) gauge

Use `optimizer.py` to maximize the Bloch–Wigner volume by moving the **non‑anchor** vertices
while keeping the first three fixed at \((0,1,\infty)\). Gradient is **analytic** and supports the `INF`
sentinel; with `dynamic=True` the spherical hull is recomputed each evaluation (piecewise‑smooth objective).

```python
from optimizer import optimize_vertices_lbfgs, OptimizeOptions
from pointset_to_fuchsian import INF

anchors = [0+0j, 1+0j, INF]
free_init = [...]  # list of complex numbers for the other vertices
points0 = anchors + free_init
var_mask = [False, False, False] + [True]*len(free_init)

opts = OptimizeOptions(dynamic=True, barrier_weight=1e-3, maxiter=200)
res = optimize_vertices_lbfgs(points0, var_mask=var_mask, options=opts)

print("Final volume:", res.volume)
```
Run the included example:

```bash
python examples/optimize_lbfgs_random.py
```
