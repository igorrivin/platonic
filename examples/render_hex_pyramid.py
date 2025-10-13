#!/usr/bin/env python3
"""
Render a hexagonal pyramid (off-center hexagon + north pole) with lights/transparency/iridescence.
Saves PNGs demonstrating different style knobs.
"""
import os, sys, math

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointset_to_fuchsian import render_snapshot

def points_off_center_hexagon():
    w = complex(math.cos(math.pi/3), math.sin(math.pi/3))
    return [1 + (w**i) for i in range(6)]

def main():
    points = points_off_center_hexagon()

    # Clean glassy look
    render_snapshot(points,
                    add_north_pole=True, refinement=2, model="poincare",
                    soap_iridescence=False,
                    base_color=(0.7, 0.85, 1.0, 0.25), metallic=0.05, roughness=0.04,
                    light_setup="three_point",
                    key_intensity=4.0, fill_intensity=1.5, rim_intensity=2.5,
                    camera_pos=(2.7, 2.1, 1.6), target=(0, 0, 0),
                    width=1400, height=1050,
                    outfile="hex_pyramid_glass.png")

    # Soap-film iridescence with ring lights
    render_snapshot(points,
                    add_north_pole=True, refinement=2, model="poincare",
                    soap_iridescence=True,
                    n_film=1.37, t0_nm=650.0, t_amp_nm=450.0, swirl_scale=9.0,
                    thickness_mode='stripes', wavelengths_nm=(420.0, 520.0, 660.0),
                    light_setup="ring", ring_n=12, ring_radius=5.0, ring_z=2.5, ring_intensity=1.2,
                    metallic=0.08, roughness=0.05,
                    camera_pos=(2.5, 2.0, 1.4), target=(0, 0, 0),
                    width=1600, height=1200,
                    outfile="hex_pyramid_iridescent.png")

if __name__ == "__main__":
    main()
