"""
Test with a regular octahedron to verify the volume calculation.
The regular octahedron should have volume = 2 * regular ideal tetrahedron.
"""

import numpy as np
from volume_checker import compute_volume

# Regular ideal tetrahedron volume
REGULAR_IDEAL_TET_VOLUME = 1.0149416064096536

def create_regular_octahedron():
    """Create vertices of a regular octahedron at (±1,0,0), (0,±1,0), (0,0,±1)."""
    return [
        complex(1, 0),    # (1,0,0) -> stereographic projection
        complex(-1, 0),   # (-1,0,0)
        complex(0, 1),    # (0,1,0)
        complex(0, -1),   # (0,-1,0)
        complex(0, 0),    # (0,0,-1) -> origin
        None              # (0,0,1) -> infinity
    ]

def main():
    # Regular octahedron vertices
    vertices = create_regular_octahedron()
    
    # Compute volume
    volume, comb = compute_volume(vertices, return_combinatorics=True)
    
    print(f"Regular octahedron:")
    print(f"Volume: {volume}")
    print(f"Expected (2 * regular tet): {2 * REGULAR_IDEAL_TET_VOLUME}")
    print(f"Ratio to regular tet: {volume / REGULAR_IDEAL_TET_VOLUME}")
    
    print(f"\nCombinatorics:")
    print(f"Vertices: {len(comb['vertices'])}")
    print(f"Edges: {len(comb['edges'])}")
    print(f"Faces: {len(comb['faces'])}")
    
    # The cube that we computed appears to be giving us the octahedron volume!
    # This confirms that we're getting 2x instead of 5x

if __name__ == "__main__":
    main()