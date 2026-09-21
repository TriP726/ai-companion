"""Deterministic force-directed layout in three dimensions.

Same family as the 2D layout in graph_service (Fruchterman–Reingold style
repulsion + springs) with a z axis. Deterministic: initial positions are
seeded by a sha256 of each node id rather than process-global randomness,
so the graph does not rearrange itself between launches.

Qt-free on purpose — the renderer pulls plain dicts out of this.

Note for future editors: the packaging guard scans raw source text with a
regex, and lines that read like "from a ..." in prose trip it. Start
wrapped docstring lines with the preposition's object ("by a ..."), never
with "from a"/"import a".
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Optional

import numpy as np

Vec3 = tuple[float, float, float]

# RMS radius the layout is normalized to. Everything downstream (node
# world radius, camera fit distance) is sized against this.
TARGET_SPREAD = 350.0


def _seed_for(node_id: str) -> int:
    digest = hashlib.sha256(node_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _initial_positions(node_ids: list[str]) -> np.ndarray:
    """One deterministic point per node on a rough sphere shell."""
    pts = np.zeros((len(node_ids), 3))
    for i, node_id in enumerate(node_ids):
        rng = np.random.default_rng(_seed_for(node_id))
        v = rng.normal(size=3)
        n = np.linalg.norm(v)
        if n < 1e-9:
            v = np.array([1.0, 0.3, 0.7])
            n = np.linalg.norm(v)
        pts[i] = v / n * TARGET_SPREAD * (0.55 + 0.45 * rng.random())
    return pts


def _normalize(positions: np.ndarray) -> np.ndarray:
    """Recentre at the origin and scale so the RMS radius is TARGET_SPREAD.

    Keeps the camera-fit distance (and the tests) independent of node count.
    """
    if len(positions) == 0:
        return positions
    positions = positions - positions.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum(positions**2, axis=1))))
    if rms < 1e-6:  # all nodes stacked (happens for n == 1)
        if len(positions) == 1:
            return positions
        idx = np.arange(len(positions), dtype=float)
        positions = np.stack(
            [idx - idx.mean(), (idx * 1.7) % 3 - 1.5, (idx * 2.3) % 5 - 2.5],
            axis=1,
        )
        rms = float(np.sqrt(np.mean(np.sum(positions**2, axis=1))))
    return positions * (TARGET_SPREAD / rms)


def force_layout_3d(
    node_ids: Iterable[str],
    edges: Iterable[tuple[str, str]],
    *,
    iterations: int = 260,
    seed_positions: Optional[dict[str, Vec3]] = None,
) -> dict[str, Vec3]:
    """Spring/repulsion layout in 3D.

    Args:
        node_ids: nodes to place.
        edges: (source_id, target_id) pairs; unknown ids are ignored.
        iterations: more = rounder; 260 converges well into the hundreds
            of nodes on a desktop CPU and stays under ~50 ms.
        seed_positions: keep a layout stable across incremental updates —
            known ids start from these coordinates instead of their hashed
            starting points.

    Returns {node_id: (x, y, z)} centred on the origin with RMS radius
    TARGET_SPREAD. One node -> {(0, 0, 0)}; zero nodes -> {}.
    """
    ids = list(dict.fromkeys(node_ids))  # dedupe, keep order
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: (0.0, 0.0, 0.0)}

    idx = {node_id: i for i, node_id in enumerate(ids)}
    pairs = [
        (idx[a], idx[b])
        for a, b in ((s, t) for s, t in edges if s in idx and t in idx)
        if a != b
    ]

    pos = _initial_positions(ids)
    for node_id, p in (seed_positions or {}).items():
        if node_id in idx:
            pos[idx[node_id]] = np.asarray(p, dtype=float)

    # Ideal spring length scales with the cube root of the per-node volume.
    volume = (2.0 * TARGET_SPREAD) ** 3
    k = 0.9 * (volume / n) ** (1.0 / 3.0)

    t_start, t_end = 0.16 * TARGET_SPREAD, 0.02 * TARGET_SPREAD
    for step in range(iterations):
        cooling = t_start * (t_end / t_start) ** (step / max(iterations - 1, 1))
        disp = np.zeros((n, 3))

        # Repulsion: every pair pushes apart (O(n^2); fine for graph scale).
        delta = pos[:, None, :] - pos[None, :, :]  # (n, n, 3)
        dist = np.linalg.norm(delta, axis=-1) + 1e-6
        np.fill_diagonal(dist, 1e9)  # a node does not repel itself
        force = (k * k / dist**2)[:, :, None] * (delta / dist[:, :, None])
        disp += force.sum(axis=1)

        # Springs along edges.
        for a, b in pairs:
            d = pos[b] - pos[a]
            length = float(np.linalg.norm(d)) + 1e-6
            pull = (d / length) * (length * length / k)
            disp[a] += pull
            disp[b] -= pull

        lengths = np.linalg.norm(disp, axis=1)
        scale = np.minimum(lengths, cooling) / np.maximum(lengths, 1e-9)
        pos += disp * scale[:, None]
        pos -= pos.mean(axis=0)  # keep centred while iterating

    pos = _normalize(pos)
    return {node_id: tuple(float(c) for c in pos[i]) for i, node_id in enumerate(ids)}
