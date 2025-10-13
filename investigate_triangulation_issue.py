"""
Investigate why different triangulations give different volumes.
The volume should be independent of face triangulation!
"""

import numpy as np
import math
import cmath
from volume_from_sphere_points import stereographic, tetra_shape, bloch_wigner

LAMBDA_PI_3 = 1.0149416064096537

def test_triangulation_independence():
    """Test that volume is independent of face triangulation."""
    
    # Cube vertices on unit sphere
    a = 1.0 / math.sqrt(3)
    
    # Create vertices
    vertices = []
    vertex_map = {}
    idx = 0
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices.append(np.array([i*a, j*a, k*a]))
                vertex_map[(i, j, k)] = idx
                idx += 1
    
    # Project to complex plane
    z = [stereographic(v) for v in vertices]
    
    # Define one square face: the top face (z = +1)
    top_face = [
        vertex_map[(-1, -1, +1)],  # v1
        vertex_map[(+1, -1, +1)],  # v5
        vertex_map[(+1, +1, +1)],  # v7
        vertex_map[(-1, +1, +1)]   # v3
    ]
    
    print("Testing triangulation independence on top face of cube")
    print(f"Square face vertices: {top_face}")
    
    # Method 1: Triangulate using diagonal (1,5,7) and (1,7,3)
    triangulation1 = [
        (top_face[0], top_face[1], top_face[2]),  # (1, 5, 7)
        (top_face[0], top_face[2], top_face[3])   # (1, 7, 3)
    ]
    
    # Method 2: Triangulate using other diagonal (5,7,3) and (5,3,1)
    triangulation2 = [
        (top_face[1], top_face[2], top_face[3]),  # (5, 7, 3)
        (top_face[1], top_face[3], top_face[0])   # (5, 3, 1)
    ]
    
    # Use different apexes to see what happens
    apex = 0  # vertex at (-1, -1, -1)
    
    print(f"\nUsing apex = v{apex}")
    
    # Compute volume for triangulation 1
    vol1 = 0.0
    print("\nTriangulation 1:")
    for tri in triangulation1:
        shape = tetra_shape(z[apex], z[tri[0]], z[tri[1]], z[tri[2]])
        v = abs(bloch_wigner(shape))
        vol1 += v
        print(f"  Tetrahedron ({apex}, {tri[0]}, {tri[1]}, {tri[2]}): volume = {v:.6f}")
    
    # Compute volume for triangulation 2
    vol2 = 0.0
    print("\nTriangulation 2:")
    for tri in triangulation2:
        shape = tetra_shape(z[apex], z[tri[0]], z[tri[1]], z[tri[2]])
        v = abs(bloch_wigner(shape))
        vol2 += v
        print(f"  Tetrahedron ({apex}, {tri[0]}, {tri[1]}, {tri[2]}): volume = {v:.6f}")
    
    print(f"\nVolume from triangulation 1: {vol1:.6f}")
    print(f"Volume from triangulation 2: {vol2:.6f}")
    print(f"Difference: {abs(vol1 - vol2):.6f}")
    
    # Now let's check what happens with the problematic triangulation
    # that includes a diagonal across the face
    print("\n" + "="*60)
    print("Now testing the 'problematic' triangulation that ConvexHull creates")
    
    # This is what we saw in the output: the face was split into triangles
    # that don't form a proper triangulation of the square
    # Let's see what triangles ConvexHull might create
    
    # From the output, we had faces like [6, 7, 5] which suggests
    # ConvexHull is creating triangles that span across different cube faces!
    
    print("\nThe issue might be that ConvexHull is not respecting face boundaries")
    print("and creating triangles that cut across the polyhedron interior!")
    
    return vol1, vol2

def analyze_convex_hull_triangulation():
    """Analyze what ConvexHull is actually doing."""
    from scipy.spatial import ConvexHull
    
    # Cube vertices
    a = 1.0 / math.sqrt(3)
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([i*a, j*a, k*a])
    
    points = np.array(points)
    hull = ConvexHull(points)
    
    print("\n" + "="*60)
    print("Analyzing ConvexHull output:")
    print(f"Number of simplices: {len(hull.simplices)}")
    print(f"Number of vertices: {len(hull.vertices)}")
    
    print("\nSimplices (triangles):")
    for i, simplex in enumerate(hull.simplices):
        print(f"  {i}: {list(simplex)}")
    
    # Check if simplices are coplanar with cube faces
    print("\nChecking which simplices belong to which cube face:")
    
    # For a cube, faces are defined by one coordinate being constant
    for i, simplex in enumerate(hull.simplices):
        v0, v1, v2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Check if x, y, or z is constant
        tol = 1e-10
        if abs(v0[0] - v1[0]) < tol and abs(v1[0] - v2[0]) < tol:
            print(f"  Simplex {i}: x = {v0[0]:.3f} face")
        elif abs(v0[1] - v1[1]) < tol and abs(v1[1] - v2[1]) < tol:
            print(f"  Simplex {i}: y = {v0[1]:.3f} face")
        elif abs(v0[2] - v1[2]) < tol and abs(v1[2] - v2[2]) < tol:
            print(f"  Simplex {i}: z = {v0[2]:.3f} face")
        else:
            print(f"  Simplex {i}: NOT ON A CUBE FACE - this is the problem!")

if __name__ == "__main__":
    print("Testing triangulation independence...\n")
    vol1, vol2 = test_triangulation_independence()
    
    if abs(vol1 - vol2) < 1e-10:
        print("\n✓ Triangulation gives same volume - as expected!")
    else:
        print("\n✗ Different triangulations give different volumes - this is wrong!")
    
    analyze_convex_hull_triangulation()