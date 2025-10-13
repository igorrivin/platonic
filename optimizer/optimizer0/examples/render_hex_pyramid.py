#!/usr/bin/env python3
"""
Render a hexagonal pyramid (off-center hexagon + north pole) with lights/transparency/iridescence.
Saves a PNG in the working directory.
"""
import math
from pointset_to_fuchsian import render_snapshot

def main():
    w = complex(math.cos(math.pi/3), math.sin(math.pi/3))
    points = [1 + (w**i) for i in range(6)]
    png = render_snapshot(points,
                          add_north_pole=True,
                          refinement=2,
                          model="poincare",
                          width=1600, height=1200,
                          background=(1,1,1,1),
                          camera_pos=(2.6,2.0,1.5), target=(0,0,0),
                          transparency_alpha=0.33,
                          light_setup="ring",
                          soap_iridescence=True,
                          # style knobs
                          base_color=(1.0, 1.0, 1.0, 0.25),
                          metallic=0.1, roughness=0.05, double_sided=True,
                          n_film=1.37, t0_nm=650, t_amp_nm=450, swirl_scale=9.0,
                          thickness_mode='stripes', wavelengths_nm=(420.0, 520.0, 660.0),
                          key_intensity=5.0, fill_intensity=2.0, rim_intensity=3.0,
                          ring_n=12, ring_radius=5.0, ring_z=2.5, ring_intensity=1.2,
                          outfile="hex_pyramid_iridescent.png")
    print("Saved:", png)

if __name__ == "__main__":
    main()
