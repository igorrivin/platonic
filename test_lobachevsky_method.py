"""
Test the Lobachevsky method for volume computation.
"""

import numpy as np
import math
from scipy.spatial import ConvexHull
from volume_lobachevsky import (
    volume_from_complex_points, rotate_to_north_pole, 
    stereographic_from_north, volume_and_gradient_lobachevsky,
    triangle_angles, lobachevsky
)

# For comparison
REGULAR_IDEAL_TET_VOLUME = 1.0149416064096537

def test_regular_tetrahedron():
    """Test with regular ideal tetrahedron."""
    print("Testing regular ideal tetrahedron")
    print("=" * 60)
    
    # Regular tetrahedron: vertices at 0, 1, ∞, exp(iπ/3)
    # In complex plane (with ∞ at north pole)
    z4 = complex(0.5, math.sqrt(3)/2)
    points = [0+0j, 1+0j, None, z4]  # None represents infinity
    
    # The tetrahedron has 4 faces (triangles)
    # When projecting from vertex 2 (infinity), we get a triangle with vertices 0, 1, z4
    triangles_from_inf = [(0, 1, 3)]  # Only one triangle visible from infinity
    
    # But we need all 4 faces of the tetrahedron
    # Faces are: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    # But with vertex 2 at infinity, we only see triangles not containing it
    
    # Actually, let's work with sphere coordinates first
    # Map to sphere
    pts_sphere = []
    pts_sphere.append(np.array([0., 0., -1.]))  # 0 -> south pole
    pts_sphere.append(np.array([1., 0., 0.]))   # 1 -> (1,0,0)
    pts_sphere.append(np.array([0., 0., 1.]))   # ∞ -> north pole
    
    # For z4 = 0.5 + i*sqrt(3)/2
    x = 2 * z4.real / (abs(z4)**2 + 1)
    y = 2 * z4.imag / (abs(z4)**2 + 1)
    z = (abs(z4)**2 - 1) / (abs(z4)**2 + 1)
    pts_sphere.append(np.array([x, y, z]))
    
    pts3 = np.array(pts_sphere)
    
    # Get convex hull to find all triangular faces
    hull = ConvexHull(pts3)
    print(f"Hull has {len(hull.simplices)} faces")
    
    # Test volume computation
    volume, _ = volume_and_gradient_lobachevsky(pts3, hull.simplices, north_pole_idx=2)
    
    print(f"\nComputed volume: {volume}")
    print(f"Expected volume: {REGULAR_IDEAL_TET_VOLUME}")
    print(f"Error: {abs(volume - REGULAR_IDEAL_TET_VOLUME)}")
    
def test_triangle_volume():
    """Test volume of a single triangle."""
    print("\n\nTesting single triangle volume")
    print("=" * 60)
    
    # For the ideal triangle with vertices 0, 1, exp(iπ/3)
    z1 = 0+0j
    z2 = 1+0j
    z3 = complex(0.5, math.sqrt(3)/2)
    
    angles = triangle_angles(z1, z2, z3)
    print(f"Triangle vertices: {z1}, {z2}, {z3}")
    print(f"Angles: {angles}")
    print(f"Sum of angles: {sum(angles)} (should equal π)")
    
    # For ideal triangle, all angles should be π/3
    expected_angle = math.pi / 3
    print(f"Expected angles: {expected_angle} each")
    print(f"Angle errors: {[abs(a - expected_angle) for a in angles]}")
    
    # Volume of single triangle
    vol = sum(lobachevsky(angle) for angle in angles) - lobachevsky(math.pi)
    print(f"\nTriangle volume: {vol}")
    
    # The volume of the tetrahedron should be related to this
    # For regular tetrahedron projecting from one vertex, we see the opposite face
    # which is an equilateral triangle
    
def test_north_pole_rotation():
    """Test rotation to north pole."""
    print("\n\nTesting rotation to north pole")
    print("=" * 60)
    
    # Test with a point not at north pole
    p = np.array([1., 0., 0.])  # Point on equator
    pts = np.array([p, [0., 1., 0.], [0., 0., -1.]])
    
    rotated = rotate_to_north_pole(pts, 0)
    print(f"Original point: {p}")
    print(f"After rotation: {rotated[0]}")
    print(f"Should be close to [0, 0, 1]: {np.allclose(rotated[0], [0, 0, 1])}")
    
def test_stereographic_projection():
    """Test stereographic projection from north pole."""
    print("\n\nTesting stereographic projection")
    print("=" * 60)
    
    # Test some known projections
    test_points = [
        (np.array([0., 0., 1.]), None, "North pole -> infinity"),
        (np.array([0., 0., -1.]), 0+0j, "South pole -> origin"),
        (np.array([1., 0., 0.]), 1+0j, "(1,0,0) -> 1"),
        (np.array([0., 1., 0.]), 0+1j, "(0,1,0) -> i"),
    ]
    
    for p, expected, desc in test_points:
        result = stereographic_from_north(p)
        print(f"{desc}")
        print(f"  Got: {result}, Expected: {expected}")
        if expected is None:
            print(f"  Correct: {result is None}")
        else:
            print(f"  Correct: {abs(result - expected) < 1e-10}")

if __name__ == "__main__":
    test_north_pole_rotation()
    test_stereographic_projection()
    test_triangle_volume()
    test_regular_tetrahedron()