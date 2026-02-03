"""
Initial UB/LB program for Euclidean TSP (pure stdlib).

Exports:
    from typing import List, Tuple, Dict, Any
    Coord = Tuple[float, float]

    def solve_tsp_bounds(coords: List[Coord]) -> Dict[str, Any]:
        return {"tour": [...], "lower_bound": float, "witness": {"pi": [...]}}

Design:
  • UB: multi-start Nearest-Neighbor + 2-opt (candidate-driven), deterministic seed.
  • LB: Held–Karp Lagrangian (1-tree) via subgradient on node potentials pi.
        Recomputed by evaluator ⇒ acts as a certificate.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any
import math, random, time

Coord = Tuple[float, float]
EPS = 1e-9

# ---------------- distances / tour utilities ----------------

def _dist(a: Coord, b: Coord) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _tour_len(tour: List[int], coords: List[Coord]) -> float:
    n = len(tour); s = 0.0
    for i in range(n):
        s += _dist(coords[tour[i]], coords[tour[(i+1)%n]])
    return s

def _valid_perm(tour: List[int], n: int) -> bool:
    return isinstance(tour, list) and len(tour)==n and set(tour)==set(range(n))

def _pos(tour: List[int]) -> List[int]:
    p = [0]*len(tour)
    for i,v in enumerate(tour): p[v]=i
    return p

def _two_opt(tour: List[int], coords: List[Coord], time_limit: float, t0: float) -> None:
    """Candidate-driven 2-opt (nearest-k)."""
    n = len(tour)
    if n < 4: return
    # build k-NN for candidates (k=20 or n-1)
    k = min(20, n-1)
    nn = [[] for _ in range(n)]
    for i in range(n):
        djs = [( _dist(coords[i], coords[j]), j) for j in range(n) if j!=i]
        djs.sort(); nn[i] = [j for _,j in djs[:k]]

    improved = True
    while improved and (time.time()-t0 < time_limit):
        improved = False
        pos = _pos(tour)
        for ai in range(n):
            if time.time()-t0 >= time_limit: break
            a = tour[ai]; b = tour[(ai+1)%n]
            dab = _dist(coords[a], coords[b])
            for c in nn[a]:
                ci = pos[c]; d = tour[(ci+1)%n]
                if c in (a,b) or d in (a,b): continue
                if ci in (ai, (ai-1)%n, (ai+1)%n): continue
                delta = _dist(coords[a], coords[c]) + _dist(coords[b], coords[d]) - dab - _dist(coords[c], coords[d])
                if delta < -1e-12:
                    i, j = min(ai+1, ci), max(ai+1, ci)
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
                    break
            if improved: break

def _nn_tour(coords: List[Coord], start: int) -> List[int]:
    n = len(coords)
    if n<=1: return list(range(n))
    unv = set(range(n)); unv.remove(start)
    t = [start]; cur = start
    while unv:
        nxt = min(unv, key=lambda j: _dist(coords[cur], coords[j]))
        t.append(nxt); unv.remove(nxt); cur = nxt
    return t

def _multistart_2opt(coords: List[Coord], seed: int = 42, starts: int = 6, time_limit: float = 0.6) -> List[int]:
    rng = random.Random(seed)
    n = len(coords)
    best = None; bestL = float("inf")
    t0 = time.time()
    starts = min(starts, max(1, n))
    chosen = list(range(n))
    rng.shuffle(chosen); chosen = chosen[:starts]
    for s in chosen:
        if time.time()-t0 >= time_limit: break
        t = _nn_tour(coords, s)
        _two_opt(t, coords, time_limit, t0)
        L = _tour_len(t, coords)
        if L < bestL:
            best, bestL = t, L
    return best if best is not None else list(range(n))

# ---------------- HK 1-tree subgradient (LB certificate) ----------------

def _aug_cost(coords: List[Coord], i: int, j: int, pi: List[float]) -> float:
    return _dist(coords[i], coords[j]) + pi[i] + pi[j]

def _mst_cost_and_degrees_on_nodes(coords: List[Coord], nodes: List[int], pi: List[float]) -> Tuple[float, List[int]]:
    m = len(nodes)
    if m <= 1:
        return 0.0, [0]*m
    in_mst = [False]*m
    key = [float('inf')]*m
    parent = [-1]*m
    key[0] = 0.0
    for _ in range(m):
        u=-1; best=float('inf')
        for i in range(m):
            if not in_mst[i] and key[i]<best:
                best=key[i]; u=i
        if u==-1: break
        in_mst[u]=True
        u_node = nodes[u]
        for v_i, v_node in enumerate(nodes):
            if not in_mst[v_i] and v_i!=u:
                w = _aug_cost(coords, u_node, v_node, pi)
                if w < key[v_i]:
                    key[v_i]=w; parent[v_i]=u
    cost=0.0; deg=[0]*m
    for v in range(1,m):
        p = parent[v]
        if p!=-1:
            a=nodes[v]; b=nodes[p]
            cost += _aug_cost(coords, a, b, pi)
            deg[v]+=1; deg[p]+=1
    return cost, deg

def _one_tree_bound(coords: List[Coord], pi: List[float], root: int = 0) -> Tuple[float, List[int]]:
    n = len(coords)
    pi = (pi + [0.0]*n)[:n]
    others = [i for i in range(n) if i != root]
    mst_cost, deg_local = _mst_cost_and_degrees_on_nodes(coords, others, pi)
    # two cheapest from root
    best1=(float('inf'),-1); best2=(float('inf'),-1)
    for j in others:
        w = _aug_cost(coords, root, j, pi)
        if w < best1[0]:
            best2 = best1; best1 = (w,j)
        elif w < best2[0] and j != best1[1]:
            best2 = (w,j)
    if best2[1] == -1:
        best2 = best1
    aug_cost = mst_cost + best1[0] + best2[0]
    LB = aug_cost - 2.0*sum(pi)
    # degrees
    deg = [0]*n
    for idx, v in enumerate(others): deg[v] += deg_local[idx]
    if best1[1] != -1: deg[root]+=1; deg[best1[1]]+=1
    if best2[1] != -1: deg[root]+=1; deg[best2[1]]+=1
    return LB, deg

def _hk_subgradient_lb(coords: List[Coord], ub_hint: float, iters: int = 200, seed: int = 42) -> Tuple[float, List[float]]:
    """Held–Karp via subgradient on node potentials pi (HK 1-tree bound)."""
    n = len(coords)
    rng = random.Random(seed)
    # scale for clamping steps
    avg_d = 0.0
    if n >= 2:
        sample_pairs = min(200, n*(n-1)//2)
        for _ in range(sample_pairs):
            i = rng.randrange(n); j = rng.randrange(n)
            if i==j: continue
            avg_d += _dist(coords[i], coords[j])
        avg_d /= max(1, sample_pairs)
    scale = max(1.0, avg_d)

    pi = [0.0]*n
    best_lb = -float('inf'); best_pi = pi[:]
    alpha = 2.0
    stall = 0

    for t in range(iters):
        lb, deg = _one_tree_bound(coords, pi, root=0)
        if lb > best_lb + 1e-9:
            best_lb = lb; best_pi = pi[:]; stall = 0
        else:
            stall += 1
            if stall % 20 == 0:
                alpha *= 0.5  # shrink step if stalling

        g2 = 0.0
        subgrad = [d-2 for d in deg]
        for gi in subgrad: g2 += gi*gi
        if g2 <= EPS:
            # already a tour (deg 2 everywhere)
            break

        # Polyak-like step using ub_hint
        step = alpha * max(0.0, ub_hint - lb) / g2

        # update pi
        for i in range(n):
            pi[i] += step * subgrad[i]
            # guard against blow-up
            if pi[i] > 10*scale: pi[i] = 10*scale
            if pi[i] < -10*scale: pi[i] = -10*scale

    # final recompute
    final_lb, _ = _one_tree_bound(coords, best_pi, root=0)
    return final_lb, best_pi

# ---------------- public API ----------------

def solve_tsp_bounds(coords: List[Coord]) -> Dict[str, Any]:
    n = len(coords)
    if n == 0:
        return {"tour": [], "lower_bound": 0.0, "witness": {"pi": []}}
    if n == 1:
        return {"tour": [0], "lower_bound": 0.0, "witness": {"pi": [0.0]}}

    # UB: multi-start NN + 2-opt (~0.6s budget typical; here fixed small)
    ub_t0 = time.time()
    tour = _multistart_2opt(coords, seed=42, starts=min(6, n), time_limit=0.5)
    # safety
    if not _valid_perm(tour, n):
        tour = list(range(n))
    ub_len = _tour_len(tour, coords)

    # LB: HK 1-tree with subgradient; use UB as a hint
    lb_val, pi = _hk_subgradient_lb(coords, ub_hint=ub_len, iters=180, seed=43)

    # never claim above UB and never negative
    lb_val = max(0.0, min(lb_val, ub_len))

    return {
        "tour": tour,
        "lower_bound": float(lb_val),
        "witness": {"pi": [float(x) for x in pi]}
    }

# local smoke test
if __name__ == "__main__":
    rng = random.Random(0)
    coords = [(rng.random()*200.0, rng.random()*200.0) for _ in range(24)]
    out = solve_tsp_bounds(coords)
    print("n:", len(coords), "UB:", out["tour"] and round(sum(_dist(coords[out["tour"][i]], coords[out["tour"][(i+1)%len(coords)]]) for i in range(len(coords))), 3),
          "LB:", round(out["lower_bound"], 3),
          "gap:", round((sum(_dist(coords[out["tour"][i]], coords[out["tour"][(i+1)%len(coords)]]) for i in range(len(coords))) - out["lower_bound"]) / max(1e-9, sum(_dist(coords[out["tour"][i]], coords[out["tour"][(i+1)%len(coords)]]) for i in range(len(coords)))), 4))
