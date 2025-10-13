"""Test volume_checker.py with tetrahedron"""

import math
from volume_checker import compute_volume

# Regular ideal tetrahedron
points = [0+0j, 1+0j, None, complex(0.5, math.sqrt(3)/2)]

volume = compute_volume(points)
print(f"Volume from volume_checker: {volume}")

REGULAR_IDEAL_TET_VOLUME = 1.0149416064096537
print(f"Expected: {REGULAR_IDEAL_TET_VOLUME}")
print(f"Error: {abs(volume - REGULAR_IDEAL_TET_VOLUME)}")