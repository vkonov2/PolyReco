"""Posthoc diagnostics of a saved minima cloud against finite template edges.

No alignment, point generation or physical edge tracking happens here.  Window
indices identify overlapping source measurements, never persistent edge tracks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _array(value: object, name: str, columns: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _indices(value: object, name: str, count: int, upper: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (count,):
        raise ValueError(f"{name} length does not match points")
    if not np.all(np.isfinite(result)) or not np.all(result == np.floor(result)):
        raise ValueError(f"{name} must contain finite integer indices")
    if np.any(result < 0) or np.any(result >= upper):
        raise ValueError(f"{name} contains an out-of-range index")
    return result.astype(np.int64)


def _numbers(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _summary(values: object) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"count": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def nearest_finite_edge_distances(
    points: np.ndarray,
    vertices: np.ndarray,
    edges: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return distances, edge IDs and clipped segment parameters, without lines."""
    pts = _array(points, "points", 3)
    verts = _array(vertices, "vertices", 3)
    raw_edges = _array(edges, "edges", 2)
    if raw_edges.shape[0] == 0 or not np.all(raw_edges == np.floor(raw_edges)):
        raise ValueError("edges must contain at least one integer vertex pair")
    if np.any(raw_edges < 0) or np.any(raw_edges >= len(verts)):
        raise ValueError("edges contains an out-of-range vertex")
    edge_indices = raw_edges.astype(np.int64)
    starts = verts[edge_indices[:, 0]]
    vectors = verts[edge_indices[:, 1]] - starts
    squared_lengths = np.einsum("ei,ei->e", vectors, vectors)
    if np.any(squared_lengths <= 0.0):
        raise ValueError("edges contains a zero-length segment")
    if int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    distances = np.empty(len(pts), dtype=float)
    ids = np.empty(len(pts), dtype=np.int64)
    parameters = np.empty(len(pts), dtype=float)
    for offset in range(0, len(pts), int(chunk_size)):
        chunk = pts[offset : offset + int(chunk_size)]
        delta = chunk[:, None, :] - starts[None, :, :]
        t = np.clip(np.einsum("nei,ei->ne", delta, vectors) / squared_lengths, 0.0, 1.0)
        residual = delta - t[:, :, None] * vectors[None, :, :]
        squared = np.einsum("nei,nei->ne", residual, residual)
        best = np.argmin(squared, axis=1)
        rows = np.arange(len(chunk))
        distances[offset : offset + len(chunk)] = np.sqrt(squared[rows, best])
        ids[offset : offset + len(chunk)] = best
        parameters[offset : offset + len(chunk)] = t[rows, best]
    return distances, ids, parameters


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for left, right in sorted(intervals):
        left, right = max(0.0, float(left)), min(1.0, float(right))
        if right < left:
            continue
        if merged and left <= merged[-1][1] + 1e-12:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def _coverage(points: np.ndarray, start: np.ndarray, vector: np.ndarray, tolerance: float) -> dict[str, Any]:
    """Union of edge pieces inside tolerance balls; a gap is never interpolated."""
    length_squared = float(vector @ vector)
    intervals: list[tuple[float, float]] = []
    if len(points):
        delta = points - start
        t = delta @ vector / length_squared
        perpendicular = delta - t[:, None] * vector
        slack = tolerance * tolerance - np.einsum("ni,ni->n", perpendicular, perpendicular)
        for center, value in zip(t, slack):
            if value >= 0:
                radius = float(np.sqrt(value / length_squared))
                intervals.append((float(center - radius), float(center + radius)))
    merged = _merge_intervals(intervals)
    gaps: list[list[float]] = []
    last = 0.0
    for left, right in merged:
        if left > last:
            gaps.append([last, left])
        last = right
    if last < 1.0:
        gaps.append([last, 1.0])
    fraction = float(sum(right - left for left, right in merged))
    maximum_gap = max((right - left for left, right in gaps), default=0.0)
    return {
        "coverage_fraction": fraction,
        "coverage_intervals_t": merged,
        "gaps_t": gaps,
        "max_gap_fraction": float(maximum_gap),
        "covered_length": fraction * float(np.sqrt(length_squared)),
    }


def _field(half: object, key: str, default: object = None) -> object:
    return half.get(key, default) if isinstance(half, Mapping) else getattr(half, key, default)


def _prepare_halves(half_contours: list[object]) -> dict[str, Any]:
    normals: list[np.ndarray] = []
    outwards: list[np.ndarray] = []
    source_indices: list[int] = []
    ranges: list[tuple[float, float]] = []
    for index, half in enumerate(half_contours):
        normal = np.asarray(_field(half, "normal_calc", _field(half, "normal")), dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError(f"half_contours[{index}].normal must be a finite 3-vector")
        norm = float(np.linalg.norm(normal))
        if norm <= 0:
            raise ValueError(f"half_contours[{index}].normal is zero")
        normal = normal / norm
        if abs(float(normal[2])) > 1e-7:
            raise ValueError("same-Z angular visibility requires horizontal projection normals")
        half_id = _field(half, "half_id")
        if half_id not in (0, 1):
            raise ValueError(f"half_contours[{index}].half_id must be 0 or 1")
        source_index = _field(half, "source_index")
        if source_index is None or int(source_index) != source_index:
            raise ValueError(f"half_contours[{index}].source_index must be an integer")
        polyline = _array(_field(half, "points_calc", _field(half, "points")), "half contour points", 3)
        if len(polyline) < 2:
            ranges.append((float("inf"), float("-inf")))
        else:
            ranges.append((float(np.min(polyline[:, 2])), float(np.max(polyline[:, 2]))))
        lateral = np.array([-normal[1], normal[0]], dtype=float)
        normals.append(normal)
        outwards.append(lateral * (-1 if half_id == 0 else 1))
        source_indices.append(int(source_index))
    return {
        "normals": np.asarray(normals, dtype=float).reshape((-1, 3)),
        "outwards": np.asarray(outwards, dtype=float).reshape((-1, 2)),
        "source_indices": np.asarray(source_indices, dtype=np.int64),
        "z_ranges": np.asarray(ranges, dtype=float).reshape((-1, 2)),
    }


def _window_information(
    normals: np.ndarray, source_indices: np.ndarray, selected: np.ndarray
) -> tuple[float, float, int, float]:
    directions = normals[selected]
    if len(directions) < 2:
        return float("inf"), 0.0, int(np.unique(source_indices[selected]).size), 0.0
    information = np.eye(3) * len(directions) - directions.T @ directions
    eigenvalues = np.linalg.eigvalsh(information)
    weakest = max(0.0, float(eigenvalues[0]))
    condition = float(eigenvalues[-1] / weakest) if weakest > 1e-12 else float("inf")
    angles = np.sort(np.mod(np.arctan2(directions[:, 1], directions[:, 0]), np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + np.pi])
    aperture = float(np.degrees(np.pi - np.max(gaps)))
    return condition, weakest, int(np.unique(source_indices[selected]).size), aperture


def build_edge_diagnostics(
    *,
    points: np.ndarray,
    vertices: np.ndarray,
    edges: np.ndarray,
    surface_distances: np.ndarray,
    z_indices: np.ndarray,
    z_levels: np.ndarray,
    cyclic_indices: np.ndarray,
    rms_values: np.ndarray,
    window_size: int,
    half_contours: list[object],
    edge_tolerance_fraction: float = 0.001,
    ambiguity_tolerance_fraction: float = 0.00025,
    chunk_size: int = 2048,
    max_condition_number: float = 1e8,
) -> dict[str, Any]:
    """Describe geometric associations, coverage and angular observability.

    A unique ``edge`` label requires a close same-Z intersection, separation
    from alternatives, a bounded condition number and a finite small weak-axis proxy.  ``ambiguous`` may
    have only one geometric candidate if the source window is underdetermined.
    The proxy is a conditioning diagnostic, explicitly not calibrated error.
    """
    pts = _array(points, "points", 3)
    verts = _array(vertices, "vertices", 3)
    finite_distances, finite_ids, finite_t = nearest_finite_edge_distances(
        pts, verts, edges, chunk_size=chunk_size
    )
    edge_indices = np.asarray(edges, dtype=np.int64)
    if len(np.unique(np.sort(edge_indices, axis=1), axis=0)) != len(edge_indices):
        raise ValueError("edges contains duplicate unordered vertex pairs")
    levels = np.asarray(z_levels, dtype=float)
    if levels.ndim != 1 or not np.all(np.isfinite(levels)) or len(np.unique(levels)) != len(levels):
        raise ValueError("z_levels must be a finite vector of distinct coordinates")
    zis = _indices(z_indices, "z_indices", len(pts), len(levels))
    halves = _prepare_halves(half_contours)
    half_count = len(half_contours)
    cyclic = _indices(cyclic_indices, "cyclic_indices", len(pts), half_count)
    if int(window_size) != window_size or not 2 <= window_size <= half_count:
        raise ValueError("window_size must be an integer between 2 and the half-contour count")
    surface = np.asarray(surface_distances, dtype=float)
    rms = np.asarray(rms_values, dtype=float)
    if surface.shape != (len(pts),) or not np.all(np.isfinite(surface)) or np.any(surface < 0):
        raise ValueError("surface_distances must be finite nonnegative values, one per point")
    if rms.shape != (len(pts),) or np.any(np.isinf(rms)) or np.any(rms[np.isfinite(rms)] < 0):
        raise ValueError("rms_values must be nonnegative or NaN, one per point")
    for name, value in (("edge_tolerance_fraction", edge_tolerance_fraction), ("ambiguity_tolerance_fraction", ambiguity_tolerance_fraction)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(max_condition_number) or max_condition_number < 1.0:
        raise ValueError("max_condition_number must be finite and at least 1")
    diagonal = float(np.linalg.norm(np.max(verts, axis=0) - np.min(verts, axis=0)))
    tolerance = diagonal * float(edge_tolerance_fraction)
    ambiguity_tolerance = diagonal * float(ambiguity_tolerance_fraction)
    z_tolerance = max(1e-9, diagonal * 1e-7)
    support_tie_tolerance = max(1e-8, diagonal * 1e-9)
    if len(pts) and np.max(np.abs(pts[:, 2] - levels[zis])) > z_tolerance:
        raise ValueError("point Z does not agree with z_levels[z_indices] within z_tolerance")
    starts = verts[edge_indices[:, 0]]
    vectors = verts[edge_indices[:, 1]] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    horizontal = np.abs(vectors[:, 2]) <= z_tolerance
    edge_count = len(edge_indices)
    point_count = len(pts)
    same_ids = np.full(point_count, -1, dtype=np.int64)
    same_distances = np.full(point_count, np.inf)
    same_t = np.full(point_count, np.nan)
    second_ids = np.full(point_count, -1, dtype=np.int64)
    second_distances = np.full(point_count, np.inf)
    assigned = np.full(point_count, -1, dtype=np.int64)
    candidates: list[list[int]] = [[] for _ in pts]
    states = np.where(surface <= tolerance, "face", "outlier").astype(object)
    reasons = np.where(surface <= tolerance, "surface_close_no_same_z_edge", "far_from_surface_and_edge").astype(object)
    condition = np.full(point_count, np.inf)
    proxy = np.full(point_count, np.nan)
    uncertainty_status = np.full(point_count, "underdetermined", dtype=object)
    window_source_counts = np.zeros(point_count, dtype=np.int64)
    window_line_counts = np.zeros(point_count, dtype=np.int64)
    window_apertures = np.zeros(point_count)
    support_source_sets: list[set[int]] = [set() for _ in pts]
    eligible_levels: list[list[int]] = [[] for _ in range(edge_count)]
    ideal_level_rows: list[list[dict[str, Any]]] = [[] for _ in range(edge_count)]
    near_points: list[list[int]] = [[] for _ in range(edge_count)]
    candidate_points: list[list[int]] = [[] for _ in range(edge_count)]
    unique_points: list[list[int]] = [[] for _ in range(edge_count)]
    candidate_residuals: list[list[float]] = [[] for _ in range(edge_count)]
    unique_residuals: list[list[float]] = [[] for _ in range(edge_count)]
    observed_availability = (
        (levels[:, None] >= halves["z_ranges"][None, :, 0] - z_tolerance)
        & (levels[:, None] <= halves["z_ranges"][None, :, 1] + z_tolerance)
    )
    for zi, z in enumerate(levels):
        raw_t = np.divide(z - starts[:, 2], vectors[:, 2], out=np.zeros(edge_count), where=~horizontal)
        eligible = np.where(horizontal, np.abs(z - starts[:, 2]) <= z_tolerance,
                            (z >= np.minimum(starts[:, 2], starts[:, 2] + vectors[:, 2]) - z_tolerance)
                            & (z <= np.maximum(starts[:, 2], starts[:, 2] + vectors[:, 2]) + z_tolerance))
        active_edges = np.flatnonzero(eligible)
        point_ids = np.flatnonzero(zis == zi)
        available = observed_availability[zi]
        window_cache: dict[int, tuple[np.ndarray, float, float, int, float]] = {}
        for point_id in point_ids:
            start_index = int(cyclic[point_id])
            if start_index not in window_cache:
                selected = (start_index + np.arange(int(window_size))) % half_count
                selected = selected[available[selected]]
                cn, weakest, distinct, aperture = _window_information(halves["normals"], halves["source_indices"], selected)
                window_cache[start_index] = selected, cn, weakest, distinct, aperture
            selected, cn, weakest, distinct, aperture = window_cache[start_index]
            condition[point_id] = cn
            window_source_counts[point_id] = distinct
            window_line_counts[point_id] = len(selected)
            window_apertures[point_id] = aperture
            support_source_sets[point_id] = set(int(v) for v in halves["source_indices"][selected])
            if np.isfinite(cn) and distinct >= 2:
                if cn > max_condition_number:
                    # Zero residual is possible even when nearly parallel lines
                    # barely constrain position.  Keep this guard independent of RMS.
                    uncertainty_status[point_id] = "ill_conditioned"
                elif np.isfinite(rms[point_id]):
                    proxy[point_id] = float(rms[point_id] * np.sqrt(len(selected) / weakest))
                    uncertainty_status[point_id] = "finite_proxy" if proxy[point_id] <= tolerance else "large_proxy"
                else:
                    uncertainty_status[point_id] = "missing_rms"
        if not len(active_edges):
            reasons[point_ids] = "no_edge_at_this_z"
            continue
        q = starts[active_edges] + np.clip(raw_t[active_edges], 0.0, 1.0)[:, None] * vectors[active_edges]
        local_horizontal = horizontal[active_edges]
        endpoint_b = q.copy()
        q[local_horizontal] = starts[active_edges[local_horizontal]]
        endpoint_b[local_horizontal] = starts[active_edges[local_horizontal]] + vectors[active_edges[local_horizontal]]
        support_a = q[:, :2] @ halves["outwards"].T
        support_b = endpoint_b[:, :2] @ halves["outwards"].T
        support_maximum = np.max(np.maximum(support_a, support_b), axis=0)
        visible_a = support_a >= support_maximum[None, :] - support_tie_tolerance
        visible_b = support_b >= support_maximum[None, :] - support_tie_tolerance
        visible = (visible_a & visible_b) & available[None, :]
        endpoint_visible = (visible_a | visible_b) & available[None, :]
        for local_id, edge_id in enumerate(active_edges):
            eligible_levels[edge_id].append(int(zi))
            line_visible = visible[local_id]
            # All lines of one cyclic window must support this same section element.
            padded = np.r_[line_visible, line_visible[: int(window_size) - 1]]
            counts = np.convolve(padded.astype(int), np.ones(int(window_size), dtype=int), mode="valid")[:half_count]
            full_starts = np.flatnonzero(counts == window_size)
            ideal_level_rows[edge_id].append({
                "z_index": int(zi),
                "visible_half_count": int(np.count_nonzero(line_visible)),
                "visible_source_views": sorted(set(int(v) for v in halves["source_indices"][line_visible])),
                "full_window_count": int(len(full_starts)),
                "endpoint_visible_half_count": int(np.count_nonzero(endpoint_visible[local_id])),
            })
        for offset in range(0, len(point_ids), int(chunk_size)):
            ids = point_ids[offset : offset + int(chunk_size)]
            delta = pts[ids, None, :] - q[None, :, :]
            parameters = np.broadcast_to(np.clip(raw_t[active_edges], 0.0, 1.0), (len(ids), len(active_edges))).copy()
            if np.any(local_horizontal):
                hv = vectors[active_edges[local_horizontal]]
                ht = np.clip(np.einsum("nei,ei->ne", delta[:, local_horizontal], hv) / (lengths[active_edges[local_horizontal]] ** 2), 0, 1)
                parameters[:, local_horizontal] = ht
                delta[:, local_horizontal] -= ht[:, :, None] * hv[None, :, :]
            residuals = np.linalg.norm(delta, axis=2)
            order = np.argsort(residuals, axis=1, kind="stable")[:, :2]
            for local_point, point_id in enumerate(ids):
                first = int(order[local_point, 0])
                best_distance = float(residuals[local_point, first])
                same_ids[point_id] = int(active_edges[first])
                same_distances[point_id] = best_distance
                same_t[point_id] = float(parameters[local_point, first])
                if len(active_edges) > 1:
                    second = int(order[local_point, 1])
                    second_ids[point_id] = int(active_edges[second])
                    second_distances[point_id] = float(residuals[local_point, second])
                close = residuals[local_point] <= tolerance
                for edge_id in active_edges[close]:
                    near_points[edge_id].append(int(point_id))
                if best_distance > tolerance:
                    continue
                expansion = ambiguity_tolerance + (2.0 * proxy[point_id] if np.isfinite(proxy[point_id]) else 0.0)
                plausible = close & (residuals[local_point] <= best_distance + expansion)
                candidate_ids = active_edges[plausible].astype(int).tolist()
                candidates[point_id] = candidate_ids
                good = uncertainty_status[point_id] == "finite_proxy"
                unique = len(candidate_ids) == 1 and good
                states[point_id] = "edge" if unique else "ambiguous"
                reasons[point_id] = "unique_close_edge" if unique else ("multiple_close_edges" if len(candidate_ids) > 1 else str(uncertainty_status[point_id]))
                if unique:
                    assigned[point_id] = candidate_ids[0]
                for local_edge in np.flatnonzero(plausible):
                    edge_id = int(active_edges[local_edge])
                    candidate_points[edge_id].append(int(point_id))
                    candidate_residuals[edge_id].append(float(residuals[local_point, local_edge]))
                    if unique:
                        unique_points[edge_id].append(int(point_id))
                        unique_residuals[edge_id].append(float(residuals[local_point, local_edge]))
    edge_rows: list[dict[str, Any]] = []
    for edge_id in range(edge_count):
        candidate_ids = candidate_points[edge_id]
        unique_ids = unique_points[edge_id]
        visible_levels = [row["z_index"] for row in ideal_level_rows[edge_id] if row["visible_half_count"]]
        window_levels = [row["z_index"] for row in ideal_level_rows[edge_id] if row["full_window_count"]]
        visible_views = sorted({view for row in ideal_level_rows[edge_id] for view in row["visible_source_views"]})
        unique_z = sorted(set(int(v) for v in zis[unique_ids]))
        candidate_z = sorted(set(int(v) for v in zis[candidate_ids]))
        unique_views = sorted({view for point_id in unique_ids for view in support_source_sets[point_id]})
        candidate_views = sorted({view for point_id in candidate_ids for view in support_source_sets[point_id]})
        if unique_ids:
            unsupported_reason = None
        elif not eligible_levels[edge_id]:
            unsupported_reason = "no_sampled_z_intersection"
        elif candidate_ids:
            unsupported_reason = "only_ambiguous_points"
        elif near_points[edge_id]:
            unsupported_reason = "near_points_prefer_other_edges"
        elif not visible_levels:
            unsupported_reason = "no_ideal_angular_visibility"
        elif not window_levels:
            unsupported_reason = "no_full_window_visibility"
        else:
            unsupported_reason = "no_close_minima_points"
        row = {
            "edge_id": edge_id,
            "vertices": edge_indices[edge_id].astype(int).tolist(),
            "length": float(lengths[edge_id]),
            "z_span": float(abs(vectors[edge_id, 2])),
            "horizontal": bool(horizontal[edge_id]),
            "short_relative_to_tolerance": bool(lengths[edge_id] <= 2.0 * tolerance),
            "geometric_z_indices": eligible_levels[edge_id],
            "geometric_z_count": len(eligible_levels[edge_id]),
            "ideal_visible_z_indices": visible_levels,
            "ideal_visible_z_count": len(visible_levels),
            "ideal_window_z_indices": window_levels,
            "ideal_window_z_count": len(window_levels),
            "ideal_visible_source_views": visible_views,
            "ideal_visible_source_view_count": len(visible_views),
            "ideal_by_z": ideal_level_rows[edge_id],
            "geometrically_near_point_count": len(near_points[edge_id]),
            "candidate_point_count": len(candidate_ids),
            "unique_point_count": len(unique_ids),
            "ambiguous_point_count": len(candidate_ids) - len(unique_ids),
            "unique_z_indices": unique_z,
            "candidate_z_indices": candidate_z,
            "unique_z_count": len(unique_z),
            "candidate_z_count": len(candidate_z),
            "unique_source_view_count": len(unique_views),
            "candidate_source_view_count": len(candidate_views),
            "unique_source_views": unique_views,
            "candidate_source_views": candidate_views,
            "sampling_level_fraction": len(unique_z) / len(eligible_levels[edge_id]) if eligible_levels[edge_id] else None,
            "residual_summary": _summary(unique_residuals[edge_id]),
            "candidate_residual_summary": _summary(candidate_residuals[edge_id]),
            "unsupported_reason": unsupported_reason,
        }
        row.update(_coverage(pts[unique_ids], starts[edge_id], vectors[edge_id], tolerance))
        row.update({f"candidate_{key}": value for key, value in _coverage(pts[candidate_ids], starts[edge_id], vectors[edge_id], tolerance).items()})
        edge_rows.append(row)
    total_length = float(np.sum(lengths))
    counts = {name: int(np.count_nonzero(states == name)) for name in ("edge", "ambiguous", "face", "outlier")}
    return {
        "schema_version": 1,
        "semantics": {
            "purpose": "posthoc InitialModel diagnostic; no alignment, no production selection and no point regeneration",
            "edge_ids": "zero-based row of the supplied unique finite edge pairs; no triangulation diagonals",
            "same_z": "nonhorizontal edge intersects declared Z; on-level horizontal edge remains a complete finite segment",
            "classification": "edge=one close candidate with condition number <= max_condition_number and finite small conditioning proxy; ambiguous=multiple candidates or insufficient precision; face=surface-close without close same-Z edge; outlier=neither",
            "candidate_edges": "same-Z residual <= edge_tolerance and <= best residual + ambiguity_tolerance + 2 * weak_axis_proxy; if proxy unavailable use geometric ambiguity_tolerance but never label unique edge",
            "uncertainty": "uncalibrated diagnostic RMS * sqrt(valid line count / lambda_min(sum(I-ddT))); correlated windows and model mismatch prevent a probabilistic interpretation; zero RMS does not establish zero uncertainty",
            "conditioning_guard": "condition number above max_condition_number forces ill_conditioned and null proxy independently of RMS, including zero RMS; default 1e8 caps information-matrix spectral amplification at 1e8 (weak-axis ratio 1e4); this is a diagnostic numerical safeguard, not a calibrated uncertainty bound or production geometry threshold",
            "coverage": "union of exact finite-edge intersections with edge_tolerance balls around unique points; candidate coverage includes ambiguous alternatives; gaps and endpoint tails retained; no interpolation or max-min span",
            "visibility": "ideal same-Z support extrema along actual observed signed lateral directions, including all support ties; support is an angular eligibility test, not an observed ownership or nonconvex-silhouette proof",
            "horizontal_visibility": "visible counts require BOTH segment endpoints tied at support; endpoint-only directions counted separately and cannot localize the segment interior",
            "full_window_visibility": "all window_size cyclic half-contours available at Z and tied to the same ideal section element; a necessary ownership diagnostic, not enough conditioning",
            "source_views": "union of original source_index across available lines in supporting overlapping windows; two halves count once; no claim of independent samples, angular spacing or physical edge tracking",
            "z_sampling": "declared original grid and indices retained; distinct Z are not independent measurements when interpolated from contours",
            "thresholds": "diagnostic scale fractions, not approved product accuracy or an inferred noise floor",
        },
        "tolerances": {
            "bbox_diagonal": diagonal,
            "edge_tolerance_fraction": float(edge_tolerance_fraction),
            "edge_tolerance": tolerance,
            "ambiguity_tolerance_fraction": float(ambiguity_tolerance_fraction),
            "ambiguity_tolerance": ambiguity_tolerance,
            "z_tolerance": z_tolerance,
            "support_tie_tolerance": support_tie_tolerance,
            "window_size": int(window_size),
            "max_condition_number": float(max_condition_number),
        },
        "points": {
            "nearest_edge_ids": finite_ids.tolist(),
            "nearest_edge_distances": finite_distances.tolist(),
            "nearest_edge_t": finite_t.tolist(),
            "same_z_edge_ids": same_ids.tolist(),
            "same_z_distances": _numbers(same_distances),
            "same_z_t": _numbers(same_t),
            "second_same_z_edge_ids": second_ids.tolist(),
            "second_same_z_distances": _numbers(second_distances),
            "assigned_edge_ids": assigned.tolist(),
            "candidate_edge_ids": candidates,
            "states": states.tolist(),
            "reasons": reasons.tolist(),
            "condition_numbers": _numbers(condition),
            "uncertainty_proxy": _numbers(proxy),
            "uncertainty_status": uncertainty_status.tolist(),
            "window_distinct_source_views": window_source_counts.tolist(),
            "window_valid_line_count": window_line_counts.tolist(),
            "window_angular_aperture_degrees": window_apertures.tolist(),
        },
        "edges": edge_rows,
        "summary": {
            "point_count": point_count,
            "edge_count": edge_count,
            "state_counts": counts,
            "nearest_finite_edge_distances": _summary(finite_distances),
            "same_z_distances": _summary(same_distances),
            "condition_numbers": _summary(condition),
            "uncertainty_proxy": _summary(proxy),
            "uncertainty_status_counts": {str(value): int(np.count_nonzero(uncertainty_status == value)) for value in np.unique(uncertainty_status)},
            "edges_with_unique_support": sum(bool(row["unique_point_count"]) for row in edge_rows),
            "edges_with_candidate_support": sum(bool(row["candidate_point_count"]) for row in edge_rows),
            "length_weighted_coverage": sum(row["covered_length"] for row in edge_rows) / total_length,
            "length_weighted_candidate_coverage": sum(row["candidate_covered_length"] for row in edge_rows) / total_length,
            "unsupported_reason_counts": {str(reason): sum(row["unsupported_reason"] == reason for row in edge_rows) for reason in sorted({row["unsupported_reason"] for row in edge_rows if row["unsupported_reason"] is not None})},
        },
    }
