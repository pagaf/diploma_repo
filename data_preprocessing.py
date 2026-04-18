import numpy as np
import torch
from collections import deque

def get_greedy_tsp_path(coords):
    n = len(coords)
    visited = np.zeros(n, dtype=bool)
    path = np.empty(n, dtype=np.int64)
    path[0] = 0
    visited[0] = True
    for i in range(1, n):
        curr = path[i - 1]
        dists = np.linalg.norm(coords - coords[curr], axis=-1)
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        path[i] = nxt
        visited[nxt] = True
    return path

def get_bfs_graph_path(coords, k_neighbors=5):
    n = len(coords)
    dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    visited = np.zeros(n, dtype=bool)
    path = []
    queue = deque([0])
    visited[0] = True
    while queue:
        curr = queue.popleft()
        path.append(curr)
        neighbors = np.argsort(dist[curr])[1:k_neighbors + 1]
        for nxt in neighbors:
            if not visited[nxt]:
                visited[nxt] = True
                queue.append(nxt)
    for i in range(n):
        if not visited[i]:
            path.append(i)
    return np.array(path, dtype=np.int64)

def prepare_graph(coords, tour, k=20, scan_mode='tsp'):
    coords_norm = coords - coords.min(axis=0, keepdims=True)
    coords_norm = coords_norm / max(float(coords_norm.max()), 1e-9)

    if scan_mode == 'tsp':
        sort_idx = get_greedy_tsp_path(coords_norm)
    elif scan_mode == 'bfs':
        sort_idx = get_bfs_graph_path(coords_norm)
    else:
        sort_idx = np.argsort(coords_norm[:, 0])

    inv_sort_idx = np.empty_like(sort_idx)
    inv_sort_idx[sort_idx] = np.arange(len(sort_idx))
    sorted_coords = coords_norm[sort_idx]

    n = len(sorted_coords)
    dist = np.linalg.norm(sorted_coords[:, None] - sorted_coords[None, :], axis=-1)

    tour_edges = set()
    if tour is not None:
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            a_new, b_new = inv_sort_idx[a], inv_sort_idx[b]
            tour_edges.add((a_new, b_new))
            tour_edges.add((b_new, a_new))

    edges, labels = [], []
    for i in range(n):
        neighbors = np.argsort(dist[i])[1:k + 1]
        for j in neighbors:
            edges.append([i, int(j)])
            labels.append(1.0 if (i, int(j)) in tour_edges else 0.0)

    edges_idx = np.array(edges, dtype=np.int64)
    labels_arr = np.array(labels, dtype=np.float32)

    src, dst = edges_idx[:, 0], edges_idx[:, 1]
    d = dist[src, dst].astype(np.float32)
    D_PE = 16
    f = np.exp(-np.log(10000.0) * np.arange(D_PE // 2) / (D_PE // 2))
    pe = np.concatenate([np.sin(d[:, None] * f), np.cos(d[:, None] * f)], axis=-1)
    edges_feat = np.concatenate([d[:, None], pe], axis=-1)

    return (
        torch.tensor(sorted_coords, dtype=torch.float32),
        torch.tensor(edges_idx, dtype=torch.long),
        torch.tensor(edges_feat, dtype=torch.float32),
        torch.tensor(labels_arr, dtype=torch.float32)
    )
