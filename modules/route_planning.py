import heapq
import math
import numpy as np

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _step_factor(s_a, s_b, h_a, h_b, lam, mu):
    slope = 0.5 * (s_a + s_b)
    haz = 0.5 * (h_a + h_b)
    return 1.0 + lam * slope / 45.0 + mu * haz / 3.0     # always >= 1


def astar(slope, hazard, start, goal, dx, dy, lam=3.0, mu=2.0, max_slope=30.0):
    """Least-cost path on the slope grid. Returns a list of (row, col) or None."""
    rows, cols = slope.shape
    blocked = ~np.isfinite(slope) | (slope >= max_slope)
    if blocked[start]:
        raise ValueError("Start cell is blocked (too steep or no data).")
    if blocked[goal]:
        raise ValueError("Destination cell is blocked (too steep or no data).")

    haz = np.nan_to_num(hazard, nan=0.0)
    slp = np.nan_to_num(slope, nan=0.0)

    def h(cell):   # admissible: every step costs at least its length
        return math.hypot((cell[0] - goal[0]) * dy, (cell[1] - goal[1]) * dx)

    g = np.full((rows, cols), np.inf)
    parent = {}
    g[start] = 0.0
    heap = [(h(start), start)]

    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        gc = g[cur]
        for dr, dc in DIRECTIONS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                continue
            if dr and dc and (blocked[cur[0] + dr, cur[1]] or blocked[cur[0], cur[1] + dc]):
                continue
            nb = (nr, nc)
            step = math.hypot(dr * dy, dc * dx) * _step_factor(
                slp[cur], slp[nb], haz[cur], haz[nb], lam, mu)
            ng = gc + step
            if ng < g[nb]:
                g[nb] = ng
                parent[nb] = cur
                heapq.heappush(heap, (ng + h(nb), nb))
    return None


def straight_line(start, goal):
    n = int(max(abs(goal[0] - start[0]), abs(goal[1] - start[1]))) + 1
    rr = np.linspace(start[0], goal[0], n).round().astype(int)
    cc = np.linspace(start[1], goal[1], n).round().astype(int)
    return list(zip(rr.tolist(), cc.tolist()))


def path_metrics(path, terrain, slope, hazard, dx, dy, lam=3.0, mu=2.0, max_slope=30.0):
    dist = cost = ascent = 0.0
    haz = np.nan_to_num(hazard, nan=0.0)
    slp = np.nan_to_num(slope, nan=0.0)
    for a, b in zip(path[:-1], path[1:]):
        seg = math.hypot((b[0] - a[0]) * dy, (b[1] - a[1]) * dx)
        dist += seg
        cost += seg * _step_factor(slp[a], slp[b], haz[a], haz[b], lam, mu)
        dz = terrain[b] - terrain[a]
        if np.isfinite(dz) and dz > 0:
            ascent += dz
    s = np.array([slope[p] for p in path], dtype=float)
    return {
        "Distance (km)": dist / 1000.0,
        "Mean slope (°)": float(np.nanmean(s)),
        "Max slope (°)": float(np.nanmax(s)),
        "Total ascent (m)": ascent,
        "Traversal cost (km-equiv.)": cost / 1000.0,
    }