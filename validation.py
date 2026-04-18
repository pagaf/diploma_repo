import numpy as np

def greedy_decode_tour(heatmap, edges_idx, n_nodes):
    scores = heatmap.cpu().numpy()
    ei = edges_idx.cpu().numpy()

    degree = np.zeros(n_nodes, dtype=int)
    adj = [[] for _ in range(n_nodes)]
    n_edges_added = 0

    parent = np.arange(n_nodes)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb
            return True
        return False

    order = np.argsort(-scores)
    for idx in order:
        u, v = int(ei[idx, 0]), int(ei[idx, 1])
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        if n_edges_added < n_nodes - 1 and find(u) == find(v):
            continue

        degree[u] += 1
        degree[v] += 1
        adj[u].append(v)
        adj[v].append(u)
        union(u, v)
        n_edges_added += 1
        if n_edges_added == n_nodes:
            break

    if n_edges_added < n_nodes:
        return None

    try:
        tour = [0]
        visited = {0}
        for _ in range(n_nodes - 1):
            nxt = next((nb for nb in adj[tour[-1]] if nb not in visited), None)
            if nxt is None:
                return None
            tour.append(nxt)
            visited.add(nxt)
        return tour
    except Exception:
        return None

def compute_tour_length(coords, tour):
    total = 0.0
    n = len(tour)
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        total += float(np.linalg.norm(coords[a] - coords[b]))
    return total

def tour_gap(pred_length, opt_length):
    return (pred_length / opt_length - 1.0) * 100.0
