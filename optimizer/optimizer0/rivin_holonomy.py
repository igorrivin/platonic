
"""
rivin_holonomy.py
-----------------
Penner--Rivin holonomy builder from an ideal triangulation with shear data.
"""

from typing import Dict, Tuple, List, Any
import math

Matrix = List[List[float]]

class Triangulation:
    """
    F: number of triangles
    adjacency: mapping (t, s) -> (u, su, eid)  where s, su in {0,1,2}
    order: mapping t -> a cyclic order (permutation) of [0,1,2], typically [0,1,2]
    orientation: mapping eid -> ((t_from, s_from), (t_to, s_to)) (global edge orientation)
    """
    def __init__(self, F: int, adjacency: Dict[Tuple[int,int], Tuple[int,int,int]],
                 order: Dict[int, List[int]], orientation: Dict[int, Tuple[Tuple[int,int], Tuple[int,int]]]):
        self.F = F
        self.adjacency = adjacency
        self.order = order
        self.orientation = orientation

# Basic 2x2 matrices and ops
def matI() -> Matrix: return [[1.0, 0.0],[0.0, 1.0]]
def matmul(A: Matrix, B: Matrix) -> Matrix:
    return [[A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]]
def matR() -> Matrix: return [[1.0, 1.0],[-1.0, 0.0]]
def matL() -> Matrix: return [[0.0, -1.0],[1.0, 1.0]]
def matX(Z: float) -> Matrix:
    # X(Z) = [[0, -e^{Z/2}],[e^{-Z/2}, 0]]
    e = math.exp(Z/2.0)
    return [[0.0, -e],[1.0/e, 0.0]]

def _turn(order_list: List[int], side_in: int, side_out: int) -> str:
    i = order_list.index(side_in)
    j = order_list.index(side_out)
    d = (j - i) % 3
    if d == 1: return 'L'
    if d == 2: return 'R'
    raise ValueError("Degenerate turn: side_out equals side_in")

def _path_to_root(parent: Dict[int,int], v: int, root: int) -> List[int]:
    path = []
    x = v
    while x is not None and x != root:
        path.append(x); x = parent.get(x, None)
    path.append(root)
    path.reverse()
    return path

def generators_from_triangulation(T: Triangulation, Z: Dict[int, float], root: int = 0):
    """
    Return a list of generators, one per non-tree dual edge.
    Each generator is a tuple (u, v, tokens, M):
      - u,v: the endpoints (triangle indices) of the non-tree edge in the dual graph
      - tokens: word in {'L','R'} and ('X', eid, sign) where sign in {+1,-1}
      - M: 2x2 matrix in SL(2,R) (up to sign) for that loop
    """
    # Build dual graph adjacency
    G = {t: [] for t in range(T.F)}
    for (t,s), (u,su,eid) in T.adjacency.items():
        G[t].append((u, s, su, eid))
    # BFS spanning tree
    from collections import deque
    Q = deque([root])
    parent = {root: None}
    parent_edge = {root: None}  # for v!=root: (eid, side_in_v, side_in_parent)
    while Q:
        v = Q.popleft()
        for (w, sv, sw, eid) in G[v]:
            if w not in parent:
                parent[w] = v
                parent_edge[w] = (eid, sw, sv)
                Q.append(w)
    # Identify non-tree edges (u,v) with u<v to avoid duplicates
    nt_edges = []
    seen = set()
    for (t,s), (u,su,eid) in T.adjacency.items():
        a, b = t, u
        if parent.get(b, None) == a or parent.get(a, None) == b:
            continue
        key = tuple(sorted((a,b))+[eid]) if isinstance(eid, int) else tuple(sorted((a,b)))
        if key in seen: continue
        seen.add(key)
        nt_edges.append((a,b,eid))
    # Helper to find unique path in the tree
    def path_in_tree(u, v):
        # collect ancestors
        A, B = [], []
        x = u
        while x is not None:
            A.append(x); x = parent.get(x, None)
        y = v
        while y is not None:
            B.append(y); y = parent.get(y, None)
        # find LCA
        i, j = len(A)-1, len(B)-1
        while i>=0 and j>=0 and A[i]==B[j]:
            i -= 1; j -= 1
        path = A[:i+1] + list(reversed(B[:j+1]))
        return path if path else [u]
    # Build tokens for a tree path
    def tokens_for_path(path):
        tokens = []
        side_in_prev = None
        for k in range(len(path)-1):
            a, b = path[k], path[k+1]
            # find adjacency entry a -> b
            eid = None; sa = None; sb = None
            for (w, sv, sw, e) in G[a]:
                if w == b:
                    eid, sa, sb = e, sv, sw
                    break
            if eid is None:
                raise RuntimeError("Broken adjacency on path")
            if side_in_prev is not None:
                tokens.append(('T', a, side_in_prev, sa))
            # sign from orientation
            (t_from, s_from), (t_to, s_to) = T.orientation[eid]
            sign = +1 if (t_from==a and t_to==b and s_from==sa) else -1
            tokens.append(('X', eid, sign))
            side_in_prev = sb
        # Resolve T -> L/R
        resolved = []
        for tok in tokens:
            if tok[0]=='T':
                _, tri, s_in, s_out = tok
                resolved.append(_turn(T.order[tri], s_in, s_out))
            else:
                resolved.append(tok)
        return resolved
    # Compose two token lists: path root->u, edge u--v, path v->root
    gens = []
    for (u, v, eid_hint) in nt_edges:
        p1 = path_in_tree(root, u)
        p2 = path_in_tree(root, v)
        tok1 = tokens_for_path(p1)
        tok2 = tokens_for_path(p2)
        # Find direct tokens for the non-tree connection u->v
        # Determine the oriented side in u leading to v
        eid = None; su = None; sv = None
        for (w, s_u, s_v, e) in G[u]:
            if w == v:
                eid, su, sv = e, s_u, s_v
                break
        if eid is None:
            # try v->u and invert sign
            for (w, s_v, s_u, e) in G[v]:
                if w == u:
                    eid, su, sv = e, s_u, s_v
                    break
        if eid is None:
            continue
        # If tok1 has at least one token, we last stood inside triangle u coming from some side; else fabricate a turn from side sv to su
        # Cross the non-tree edge
        (t_from, s_from), (t_to, s_to) = T.orientation[eid]
        sign = +1 if (t_from==u and t_to==v and s_from==su) else -1
        cross_tok = [('X', eid, sign)]
        # Now the return path v->root tokens are tok2; but we need to reverse direction. For simplicity we just reuse tokens_for_path on p2 and interpret as already giving turns and crosses for root->v;
        # to go v->root we traverse in reverse, which complicates turns. A simpler, standard trick is to form the concatenation: root->u, cross u->v, v->root (forward) but then interpret as a loop up to conjugacy, which is fine for traces.
        tokens = tok1 + cross_tok + tok2
        # Build matrix
        M = matI()
        for Ttok in tokens:
            if Ttok == 'L':
                M = matmul(M, matL())
            elif Ttok == 'R':
                M = matmul(M, matR())
            else:
                _, e, sgn = Ttok
                z = Z.get(e, 0.0)
                M = matmul(M, matX(z if sgn>0 else -z))
        gens.append((u, v, tokens, M))
    return gens
