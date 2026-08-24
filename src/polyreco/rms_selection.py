from __future__ import annotations

import time

import numpy as np
from highspy import Highs, kHighsInf

from polyreco.rms_mesh import (
    candidate_halfspace,
    candidate_plane,
    convex_hull_indices,
    highs_add_row,
    highs_change_row_bounds,
    plane_basis,
    reconstruct_polyhedron_from_halfspaces,
    solve_halfspace_lp,
)

EPS = 1e-9

__all__ = [
    "LpActivityEngine",
    "annotate_candidate_pool_neutral",
    "annotate_w2_support_diverse_scores",
    "as_json_float",
    "as_json_point",
    "candidate_hull_bounds",
    "candidate_hull_centroid",
    "candidate_hull_points",
    "candidate_hull_polygon_2d",
    "candidate_margin_against_vertices",
    "candidate_outside_mask",
    "candidate_plane_relation_score",
    "candidate_rank_key",
    "candidate_track_id",
    "candidate_z_intervals",
    "cumulative_metrics_from_mask",
    "cumulative_outside_metrics",
    "finite_interval",
    "finite_float",
    "hull_bounds_overlap_ratio",
    "interval_overlap",
    "lp_constraints_for_candidates",
    "lp_face_activity_details",
    "polygon_area_2d",
    "polygon_signed_distances_2d",
    "prepare_z_level_slices",
    "select_compatible_candidates",
    "support_diverse_candidate_key",
]


def as_json_float(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def as_json_point(p: np.ndarray) -> list[float | None]:
    return [as_json_float(float(v)) for v in p]


def finite_float(value: object, default: float) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def candidate_rank_key(candidate: dict[str, object]) -> tuple[float, float, float, float]:
    plane_rms = finite_float(candidate.get("plane_rms"), float("inf"))
    levels = max(1, int(candidate.get("levels") or 0))
    z_span = finite_float(candidate.get("z_max"), 0.0) - finite_float(candidate.get("z_min"), 0.0)
    condition_p95 = finite_float(candidate.get("condition_p95"), 1.0)
    condition_term = max(np.log10(max(condition_p95, 1.0)) - 3.0, 0.0) * 0.00002
    hull_area = finite_float(candidate.get("hull_area"), 0.0)
    support_bonus = -0.00001 * float(levels) - 0.00002 * max(float(z_span), 0.0)
    hull_bonus = -0.000002 * min(max(hull_area, 0.0), 20.0)
    return (plane_rms + condition_term + support_bonus + hull_bonus, condition_p95, -float(levels), -max(float(z_span), 0.0))


def candidate_hull_centroid(candidate: dict[str, object]) -> np.ndarray | None:
    hull = np.array(candidate.get("hull", []), dtype=float)
    if hull.ndim != 2 or hull.shape[0] == 0 or hull.shape[1] != 3 or not np.all(np.isfinite(hull)):
        point = np.array(candidate.get("plane_centroid", []), dtype=float)
        if point.shape == (3,) and np.all(np.isfinite(point)):
            return point
        return None
    return np.mean(hull, axis=0)


def candidate_plane_relation_score(a: dict[str, object], b: dict[str, object]) -> tuple[float, float, float]:
    pa = candidate_plane(a)
    pb = candidate_plane(b)
    if pa is None or pb is None:
        return float("inf"), float("inf"), float("inf")
    p_a, n_a = pa
    p_b, n_b = pb
    angle = float(np.degrees(np.arccos(np.clip(abs(float(n_a @ n_b)), -1.0, 1.0))))
    offset = abs(float(n_a @ p_a) - float(n_b @ p_b))
    ca = candidate_hull_centroid(a)
    cb = candidate_hull_centroid(b)
    centroid = float(np.linalg.norm(ca - cb)) if ca is not None and cb is not None else float("inf")
    return angle, offset, centroid


def candidate_hull_points(candidate: dict[str, object]) -> np.ndarray:
    hull = np.array(candidate.get("hull") or [], dtype=float)
    if hull.ndim == 2 and hull.shape[1] == 3 and hull.shape[0] > 0:
        return hull
    centroid = candidate_hull_centroid(candidate)
    if centroid is not None:
        return np.array([centroid], dtype=float)
    plane = candidate_plane(candidate)
    if plane is not None:
        return np.array([plane[0]], dtype=float)
    return np.zeros((0, 3), dtype=float)


def finite_interval(values: list[object] | np.ndarray) -> tuple[float | None, float | None]:
    arr = np.array(values, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        arr = arr[:, 2]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(np.min(arr)), float(np.max(arr))


def candidate_z_intervals(candidate: dict[str, object]) -> dict[str, object]:
    hull = candidate_hull_points(candidate)
    hull_min, hull_max = finite_interval(hull)
    support_points = np.array(candidate.get("sample_points") or candidate.get("fit_points") or [], dtype=float)
    support_min, support_max = finite_interval(support_points)
    z_idx_min, z_idx_max = finite_interval(candidate.get("z_indices") or [])
    track_min = finite_float(candidate.get("z_min"), float("nan"))
    track_max = finite_float(candidate.get("z_max"), float("nan"))
    if not np.isfinite(track_min) or not np.isfinite(track_max):
        track_min = hull_min if hull_min is not None else support_min
        track_max = hull_max if hull_max is not None else support_max
    centroid = candidate_hull_centroid(candidate)
    return {
        "centroid_z": as_json_float(float(centroid[2])) if centroid is not None and np.isfinite(float(centroid[2])) else None,
        "hull_z_min": as_json_float(float(hull_min)) if hull_min is not None else None,
        "hull_z_max": as_json_float(float(hull_max)) if hull_max is not None else None,
        "support_z_min": as_json_float(float(support_min)) if support_min is not None else None,
        "support_z_max": as_json_float(float(support_max)) if support_max is not None else None,
        "track_z_min": as_json_float(float(track_min)) if np.isfinite(track_min) else None,
        "track_z_max": as_json_float(float(track_max)) if np.isfinite(track_max) else None,
        "z_index_min": as_json_float(float(z_idx_min)) if z_idx_min is not None else None,
        "z_index_max": as_json_float(float(z_idx_max)) if z_idx_max is not None else None,
    }


def interval_overlap(a0: object, a1: object, b0: object, b1: object) -> tuple[float, float]:
    x0 = finite_float(a0, float("nan"))
    x1 = finite_float(a1, float("nan"))
    y0 = finite_float(b0, float("nan"))
    y1 = finite_float(b1, float("nan"))
    if not all(np.isfinite(v) for v in (x0, x1, y0, y1)):
        return 0.0, 0.0
    lo, hi = min(x0, x1), max(x0, x1)
    blo, bhi = min(y0, y1), max(y0, y1)
    overlap = max(0.0, min(hi, bhi) - max(lo, blo))
    span = max(hi - lo, EPS)
    return float(overlap), float(overlap / span)


def candidate_hull_bounds(candidate: dict[str, object]) -> tuple[np.ndarray, np.ndarray] | None:
    hull = np.array(candidate.get("hull") or [], dtype=float)
    if hull.ndim != 2 or hull.shape[0] == 0 or hull.shape[1] != 3 or not np.all(np.isfinite(hull)):
        point = np.array(candidate.get("plane_centroid", []), dtype=float)
        if point.shape == (3,) and np.all(np.isfinite(point)):
            hull = point.reshape((1, 3))
        else:
            return None
    return np.min(hull, axis=0), np.max(hull, axis=0)


def hull_bounds_overlap_ratio(a: dict[str, object], b: dict[str, object]) -> float:
    bounds_a = candidate_hull_bounds(a)
    bounds_b = candidate_hull_bounds(b)
    if bounds_a is None or bounds_b is None:
        return 0.0
    amin, amax = bounds_a
    bmin, bmax = bounds_b
    inter = np.maximum(0.0, np.minimum(amax, bmax) - np.maximum(amin, bmin))
    av = float(np.prod(np.maximum(amax - amin, 1e-6)))
    bv = float(np.prod(np.maximum(bmax - bmin, 1e-6)))
    iv = float(np.prod(np.maximum(inter, 0.0)))
    return float(iv / max(min(av, bv), EPS))


def polygon_signed_distances_2d(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[0] == 0 or polygon.ndim != 2 or polygon.shape[0] < 3:
        return np.full(points.shape[0], float("inf"), dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        crosses = (yi > y) != (yj > y)
        x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        inside ^= crosses & (x < x_cross)
        j = i

    edge_dist = np.full(points.shape[0], float("inf"), dtype=float)
    for i in range(polygon.shape[0]):
        a = polygon[i]
        b = polygon[(i + 1) % polygon.shape[0]]
        ab = b - a
        denom = float(ab @ ab)
        if denom <= EPS:
            dist = np.linalg.norm(points - a[None, :], axis=1)
        else:
            t = np.clip(((points - a[None, :]) @ ab) / denom, 0.0, 1.0)
            projection = a[None, :] + t[:, None] * ab[None, :]
            dist = np.linalg.norm(points - projection, axis=1)
        edge_dist = np.minimum(edge_dist, dist)
    return np.where(inside, -edge_dist, edge_dist)


def candidate_hull_polygon_2d(candidate: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    plane = candidate_plane(candidate)
    hull = np.array(candidate.get("hull", []), dtype=float)
    if plane is None or hull.ndim != 2 or hull.shape[0] < 3 or hull.shape[1] != 3:
        return None
    if not np.all(np.isfinite(hull)):
        return None
    plane_point, normal = plane
    u, v = plane_basis(normal)
    coords = np.column_stack([(hull - plane_point) @ u, (hull - plane_point) @ v])
    hull_indices = convex_hull_indices(coords)
    if len(hull_indices) < 3:
        return None
    polygon = coords[np.array(hull_indices, dtype=int)]
    area = polygon_area_2d(polygon)
    if not np.isfinite(area) or area <= EPS:
        area = finite_float(candidate.get("hull_area"), 0.0)
    return polygon, u, v, max(float(area), EPS)


def finite_local_support_stats(
    candidate: dict[str, object],
    points: np.ndarray,
    z_indices: np.ndarray,
    *,
    plane_distance: float,
    hull_margin: float,
    relative_hull_margin: float,
) -> dict[str, object]:
    plane = candidate_plane(candidate)
    hull_data = candidate_hull_polygon_2d(candidate)
    if plane is None or hull_data is None or points.size == 0:
        return {
            "finite_support_count": 0,
            "finite_support_density": None,
            "finite_support_z_levels": 0,
            "finite_support_z_coverage": None,
            "finite_support_track_coverage": None,
            "finite_support_residual_median": None,
            "finite_support_residual_p95": None,
            "finite_support_hull_distance_median": None,
            "finite_support_purity": None,
            "finite_support_balance": None,
            "finite_support_empty_z_bins": None,
            "finite_support_margin": None,
        }
    plane_point, normal = plane
    polygon, u, v, hull_area = hull_data
    margin = float(hull_margin) + float(relative_hull_margin) * float(np.sqrt(max(hull_area, EPS)))

    signed = (points - plane_point) @ normal
    residual = np.abs(signed)
    coords = np.column_stack([(points - plane_point) @ u, (points - plane_point) @ v])
    hull_signed_dist = polygon_signed_distances_2d(coords, polygon)
    in_hull_prism = hull_signed_dist <= margin

    candidate_z = np.array(candidate.get("z_indices", []), dtype=int)
    candidate_z_set = {int(z) for z in candidate_z if int(z) >= 0}
    if candidate_z_set and z_indices.size == points.shape[0]:
        z_min_idx = min(candidate_z_set)
        z_max_idx = max(candidate_z_set)
        in_track_span = (z_indices >= z_min_idx) & (z_indices <= z_max_idx)
        exact_track_z = np.array([int(z) in candidate_z_set for z in z_indices], dtype=bool)
    else:
        z_min = finite_float(candidate.get("z_min"), float("-inf"))
        z_max = finite_float(candidate.get("z_max"), float("inf"))
        in_track_span = (points[:, 2] >= z_min - 0.03) & (points[:, 2] <= z_max + 0.03)
        exact_track_z = in_track_span

    near_plane = residual <= float(plane_distance)
    support_mask = near_plane & in_hull_prism & in_track_span
    support_count = int(np.sum(support_mask))
    support_z_values = sorted({int(z) for z in z_indices[support_mask]}) if z_indices.size == points.shape[0] else []
    support_z_levels = int(len(support_z_values))
    span_z_total = int(max(1, int(np.max(z_indices[in_track_span]) - np.min(z_indices[in_track_span]) + 1))) if np.any(in_track_span) and z_indices.size == points.shape[0] else None
    track_z_total = int(max(1, len(candidate_z_set))) if candidate_z_set else None

    purity_denom_mask = in_hull_prism & in_track_span
    purity_denom = int(np.sum(purity_denom_mask))
    purity = float(np.sum(near_plane & purity_denom_mask)) / float(purity_denom) if purity_denom > 0 else None

    empty_bins: int | None = None
    balance: float | None = None
    if candidate_z_set:
        bins = min(6, max(1, len(candidate_z_set)))
        z0 = min(candidate_z_set)
        z1 = max(candidate_z_set)
        if z1 == z0:
            counts = np.array([support_count], dtype=float)
        else:
            counts = np.zeros(bins, dtype=float)
            for z in support_z_values:
                bi = min(bins - 1, int((z - z0) * bins / float(z1 - z0 + 1)))
                counts[bi] += 1.0
        empty_bins = int(np.sum(counts <= 0.0))
        balance = float(np.min(counts) / max(float(np.mean(counts)), 1.0)) if counts.size else None

    supported_exact_z = int(len({int(z) for z in z_indices[support_mask & exact_track_z]})) if z_indices.size == points.shape[0] else support_z_levels
    residual_support = residual[support_mask]
    hull_dist_support = np.maximum(hull_signed_dist[support_mask], 0.0)
    return {
        "finite_support_count": support_count,
        "finite_support_density": as_json_float(float(support_count) / max(float(hull_area), EPS)),
        "finite_support_z_levels": support_z_levels,
        "finite_support_z_coverage": as_json_float(float(support_z_levels) / float(span_z_total)) if span_z_total else None,
        "finite_support_track_coverage": as_json_float(float(supported_exact_z) / float(track_z_total)) if track_z_total else None,
        "finite_support_residual_median": as_json_float(float(np.median(residual_support))) if residual_support.size else None,
        "finite_support_residual_p95": as_json_float(float(np.percentile(residual_support, 95))) if residual_support.size else None,
        "finite_support_hull_distance_median": as_json_float(float(np.median(hull_dist_support))) if hull_dist_support.size else None,
        "finite_support_purity": as_json_float(float(purity)) if purity is not None else None,
        "finite_support_balance": as_json_float(float(balance)) if balance is not None else None,
        "finite_support_empty_z_bins": empty_bins,
        "finite_support_margin": as_json_float(float(margin)),
    }


def candidate_track_id(candidate: dict[str, object]) -> int:
    track_id = candidate.get("track_id")
    if track_id is not None:
        return int(track_id)
    source_track_id = candidate.get("source_track_id")
    return int(source_track_id) if source_track_id is not None else -1


def novelty_candidate_key(candidate: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        finite_float(candidate.get("w2_novelty_score"), float("inf")),
        finite_float(candidate.get("plane_rms"), float("inf")),
        -finite_float(candidate.get("finite_support_density"), 0.0),
        -int(candidate.get("levels") or 0),
    )


def robust_feature_scale(values: list[float], fallback: float = 1.0) -> tuple[float, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return 0.0, float(fallback)
    med = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.std(arr))
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(fallback)
    return med, scale


def annotate_w2_support_diverse_scores(candidates: list[dict[str, object]], core_candidates: list[dict[str, object]], core_reconstructed: dict[str, object]) -> None:
    core_vertices = np.array(core_reconstructed.get("vertices") or [], dtype=float)
    core_planes = [candidate_plane(c) for c in core_candidates]
    core_planes = [p for p in core_planes if p is not None]
    margins: list[float] = []
    nearest_angles: list[float] = []
    nearest_distances: list[float] = []
    for candidate in candidates:
        plane = candidate_plane(candidate)
        margin = candidate_margin_against_vertices(candidate, core_vertices)
        candidate["_support_diverse_margin"] = float(margin or 0.0)
        margins.append(float(margin or 0.0))
        if plane is None:
            candidate["_support_diverse_nearest_angle"] = 0.0
            candidate["_support_diverse_nearest_distance"] = 0.0
            continue
        point, normal = plane
        best_angle = 180.0
        best_distance = float("inf")
        for core_point, core_normal in core_planes:
            angle = float(np.degrees(np.arccos(np.clip(abs(float(normal @ core_normal)), -1.0, 1.0))))
            distance = abs(float(normal @ (point - core_point)))
            if angle + 10.0 * distance < best_angle + 10.0 * best_distance:
                best_angle = angle
                best_distance = distance
        candidate["_support_diverse_nearest_angle"] = best_angle
        candidate["_support_diverse_nearest_distance"] = best_distance
        nearest_angles.append(best_angle)
        nearest_distances.append(best_distance)

    rms_med, rms_scale = robust_feature_scale([finite_float(c.get("plane_rms"), float("nan")) for c in candidates], 0.001)
    residual_med, residual_scale = robust_feature_scale([finite_float(c.get("finite_support_residual_p95"), float("nan")) for c in candidates], 0.02)
    density_med, density_scale = robust_feature_scale([np.log1p(max(finite_float(c.get("finite_support_density"), 0.0), 0.0)) for c in candidates], 1.0)
    levels_med, levels_scale = robust_feature_scale([float(c.get("levels") or 0) for c in candidates], 50.0)
    zspan_med, zspan_scale = robust_feature_scale([finite_float(c.get("z_span"), float("nan")) for c in candidates], 0.5)
    margin_gate = max(float(np.percentile(np.array(margins, dtype=float), 20)) if margins else 0.03, 0.03)
    margin_soft_high = max(float(np.percentile(np.array(margins, dtype=float), 75)) if margins else 0.08, margin_gate * 2.0)

    for candidate in candidates:
        rms = finite_float(candidate.get("plane_rms"), rms_med + rms_scale)
        residual = finite_float(candidate.get("finite_support_residual_p95"), residual_med + residual_scale)
        density = np.log1p(max(finite_float(candidate.get("finite_support_density"), 0.0), 0.0))
        purity = max(0.0, min(1.0, finite_float(candidate.get("finite_support_purity"), 0.0)))
        coverage = max(0.0, min(1.0, finite_float(candidate.get("finite_support_track_coverage"), 0.0)))
        levels = float(candidate.get("levels") or 0)
        zspan = finite_float(candidate.get("z_span"), 0.0)
        margin = float(candidate.get("_support_diverse_margin") or 0.0)
        angle = float(candidate.get("_support_diverse_nearest_angle") or 0.0)
        distance = float(candidate.get("_support_diverse_nearest_distance") or 0.0)
        rms_z = (rms - rms_med) / rms_scale
        residual_z = (residual - residual_med) / residual_scale
        density_z = (density - density_med) / density_scale
        levels_z = (levels - levels_med) / levels_scale
        zspan_z = (zspan - zspan_med) / zspan_scale
        activity_penalty = 0.0
        if margin < margin_gate:
            activity_penalty += (margin_gate - margin) / max(margin_gate, EPS)
        if margin > margin_soft_high:
            activity_penalty += (margin - margin_soft_high) / max(margin_soft_high, EPS)
        duplicate_penalty = 0.0
        if angle < 0.75 and distance < 0.02:
            duplicate_penalty += 1.5
        novelty_gate_bonus = -0.25 if (angle >= 1.0 or distance >= 0.025) else 0.0
        score = (
            1.4 * rms_z
            + 1.2 * residual_z
            - 0.9 * density_z
            - 0.8 * purity
            - 0.7 * coverage
            - 0.25 * levels_z
            - 0.25 * zspan_z
            + 1.3 * activity_penalty
            + duplicate_penalty
            + novelty_gate_bonus
        )
        candidate["w2_support_diverse_score"] = as_json_float(float(score))
        candidate["w2_support_diverse_components"] = {
            "plane_rms_z": as_json_float(float(rms_z)),
            "support_residual_z": as_json_float(float(residual_z)),
            "support_density_z": as_json_float(float(density_z)),
            "purity": as_json_float(float(purity)),
            "coverage": as_json_float(float(coverage)),
            "levels_z": as_json_float(float(levels_z)),
            "zspan_z": as_json_float(float(zspan_z)),
            "activity_margin": as_json_float(float(margin)),
            "activity_penalty": as_json_float(float(activity_penalty)),
            "nearest_core_angle": as_json_float(float(angle)),
            "nearest_core_plane_distance": as_json_float(float(distance)),
            "duplicate_penalty": as_json_float(float(duplicate_penalty)),
        }


def support_diverse_candidate_key(candidate: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        finite_float(candidate.get("w2_support_diverse_score"), float("inf")),
        finite_float(candidate.get("plane_rms"), float("inf")),
        finite_float(candidate.get("finite_support_residual_p95"), float("inf")),
        -int(candidate.get("levels") or 0),
    )


def candidate_margin_against_vertices(candidate: dict[str, object], vertices: np.ndarray) -> float | None:
    plane = candidate_plane(candidate)
    if plane is None or vertices.size == 0:
        return None
    point, normal = plane
    return float(np.max((vertices - point[None, :]) @ normal))


def polygon_area_2d(poly: np.ndarray) -> float:
    if poly.ndim != 2 or poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def lp_constraints_for_candidates(candidates: list[dict[str, object]]) -> list[tuple[np.ndarray, float]]:
    constraints: list[tuple[np.ndarray, float]] = []
    for candidate in candidates:
        halfspace = candidate_halfspace(candidate)
        if halfspace is not None:
            constraints.append(halfspace)
    return constraints


def observed_cloud_stats(
    candidate: dict[str, object],
    points: np.ndarray,
    z_indices: np.ndarray,
    *,
    point_tol: float,
    local_support_distance: float,
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
) -> dict[str, object]:
    plane = candidate_plane(candidate)
    finite_support = finite_local_support_stats(
        candidate,
        points,
        z_indices,
        plane_distance=float(finite_support_plane_distance),
        hull_margin=float(finite_support_hull_margin),
        relative_hull_margin=float(finite_support_relative_hull_margin),
    )
    if plane is None or points.size == 0:
        return {
            "observed_outside_count": 0,
            "observed_outside_fraction": None,
            "observed_outside_max": None,
            "observed_outside_p95": None,
            "observed_outside_p99": None,
            "observed_max_level_outside_fraction": None,
            "observed_points_near_plane": 0,
            "observed_local_support": 0,
            "observed_support_gap_global": None,
            "observed_support_gap_local": None,
            "observed_support_near_count": 0,
            "observed_support_z_coverage": None,
            **finite_support,
        }
    plane_point, normal = plane
    signed = (points - plane_point) @ normal
    support_values = points @ normal
    plane_offset = float(normal @ plane_point)
    robust_support = float(np.percentile(support_values, 99.5)) if support_values.size else float("nan")
    observed_support_gap_global = plane_offset - robust_support
    outside = signed > float(point_tol)
    positive = signed[outside]
    level_fracs: list[float] = []
    per_level_rows: list[dict[str, object]] = []
    if z_indices.size:
        for zi in sorted(set(int(z) for z in z_indices)):
            mask = z_indices == zi
            total = int(np.sum(mask))
            if total <= 0:
                continue
            frac = float(np.sum(outside[mask])) / float(total)
            level_fracs.append(frac)
            if frac > 0.0:
                per_level_rows.append({"z_index": int(zi), "outside_fraction": as_json_float(frac), "points": total})
    hull = np.array(candidate.get("hull", []), dtype=float)
    z_min = finite_float(candidate.get("z_min"), float("-inf"))
    z_max = finite_float(candidate.get("z_max"), float("inf"))
    near_plane = np.abs(signed) <= float(local_support_distance)
    in_z = (points[:, 2] >= z_min - 0.03) & (points[:, 2] <= z_max + 0.03)
    local_support = near_plane & in_z
    if hull.ndim == 2 and hull.shape[0] > 0:
        center = np.mean(hull, axis=0)
        radius = max(float(np.max(np.linalg.norm(hull - center, axis=1))), 0.2)
        local_support &= np.linalg.norm(points - center, axis=1) <= radius + float(local_support_distance) * 4.0
    local_gap = float("nan")
    local_near_count = 0
    local_z_coverage = None
    hull_data = candidate_hull_polygon_2d(candidate)
    if hull_data is not None and points.size:
        polygon, u, v, hull_area = hull_data
        coords = np.column_stack([(points - plane_point) @ u, (points - plane_point) @ v])
        hull_dist = polygon_signed_distances_2d(coords, polygon)
        margin = max(0.08, 0.03 * float(np.sqrt(max(hull_area, EPS))))
        local_mask = (hull_dist <= margin) & in_z
        local_near_count = int(np.sum(local_mask))
        if local_near_count > 0:
            local_robust = float(np.percentile(support_values[local_mask], 99.0))
            local_gap = plane_offset - local_robust
            if z_indices.size == points.shape[0]:
                local_z_coverage = float(len(set(int(z) for z in z_indices[local_mask]))) / max(1.0, float(int(candidate.get("levels") or 1)))
    return {
        "observed_outside_count": int(np.sum(outside)),
        "observed_outside_fraction": as_json_float(float(np.mean(outside))),
        "observed_outside_max": as_json_float(float(np.max(positive)) if positive.size else 0.0),
        "observed_outside_p95": as_json_float(float(np.percentile(np.maximum(signed, 0.0), 95))),
        "observed_outside_p99": as_json_float(float(np.percentile(np.maximum(signed, 0.0), 99))),
        "observed_max_level_outside_fraction": as_json_float(float(max(level_fracs)) if level_fracs else 0.0),
        "observed_level_outside": per_level_rows[:20],
        "observed_points_near_plane": int(np.sum(near_plane)),
        "observed_local_support": int(np.sum(local_support)),
        "observed_spherical_local_support": int(np.sum(local_support)),
        "observed_support_gap_global": as_json_float(float(observed_support_gap_global)),
        "observed_support_gap_local": as_json_float(float(local_gap)),
        "observed_support_near_count": int(local_near_count),
        "observed_support_z_coverage": as_json_float(float(local_z_coverage)) if local_z_coverage is not None else None,
        **finite_support,
    }


def compatible_candidate_key(candidate: dict[str, object]) -> tuple[float, float, float, float, tuple[float, float, float, float]]:
    outside_frac = finite_float(candidate.get("observed_outside_fraction"), 1.0)
    level_frac = finite_float(candidate.get("observed_max_level_outside_fraction"), 1.0)
    local_support = -float(int(candidate.get("observed_local_support") or 0))
    cluster_size = -float(int(candidate.get("dedupe_cluster_size") or 1))
    return (outside_frac, level_frac, local_support, cluster_size, candidate_rank_key(candidate))


def annotate_candidate_pool_neutral(
    candidates: list[dict[str, object]],
    *,
    core_candidates: list[dict[str, object]],
    core_reconstructed: dict[str, object],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
) -> list[dict[str, object]]:
    annotated = [dict(candidate) for candidate in candidates]
    local_support_distance = max(float(point_tol) * 2.0, 0.05)
    for candidate in annotated:
        candidate.update(
            observed_cloud_stats(
                candidate,
                trusted_points,
                trusted_z_indices,
                point_tol=float(point_tol),
                local_support_distance=local_support_distance,
                finite_support_plane_distance=float(finite_support_plane_distance),
                finite_support_hull_margin=float(finite_support_hull_margin),
                finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
            )
        )
    annotate_w2_support_diverse_scores(annotated, core_candidates, core_reconstructed)
    return annotated


def valley_quality_rejection_reason(
    candidate: dict[str, object],
    *,
    min_confidence_median: float,
    min_confidence_p10: float,
    min_two_side_fraction: float,
    max_one_side_fraction: float,
    max_distance_p95: float,
    max_track_smoothness: float,
    max_support_gap_global: float,
    max_support_gap_local: float,
    min_local_support: int,
    min_scale_persistence: int,
    min_track_levels: int,
    max_plane_rms: float,
    min_hull_area: float,
    min_finite_support_count: int,
    min_finite_track_coverage: float,
) -> str | None:
    low_counts = candidate.get("low_source_counts") or {}
    if int(low_counts.get("valley", 0) or 0) <= 0:
        return None
    conf_med = finite_float(candidate.get("valley_confidence_median"), 0.0)
    conf_p10 = finite_float(candidate.get("valley_confidence_p10"), 0.0)
    if conf_med < float(min_confidence_median) or conf_p10 < float(min_confidence_p10):
        return "low_valley_confidence"
    two_side = finite_float(candidate.get("valley_two_side_fraction"), 0.0)
    one_side = finite_float(candidate.get("valley_one_side_fraction"), 0.0)
    if two_side < float(min_two_side_fraction) or one_side > float(max_one_side_fraction):
        return "insufficient_bilateral_support"
    distance_p95 = finite_float(candidate.get("valley_distance_p95"), 0.0)
    if distance_p95 > float(max_distance_p95):
        return "excessive_valley_distance"
    left_smooth = finite_float(candidate.get("valley_left_track_smoothness"), 0.0)
    right_smooth = finite_float(candidate.get("valley_right_track_smoothness"), 0.0)
    if max(left_smooth, right_smooth) > float(max_track_smoothness):
        return "unstable_valley_tracks"
    windows = candidate.get("dedupe_cluster_windows", []) or []
    if len(windows) < int(min_scale_persistence):
        return "insufficient_scale_persistence"
    if int(candidate.get("levels") or 0) < int(min_track_levels):
        return "unstable_valley_tracks"
    if finite_float(candidate.get("plane_rms"), float("inf")) > float(max_plane_rms):
        return "unstable_valley_tracks"
    if finite_float(candidate.get("hull_area"), 0.0) < float(min_hull_area):
        return "insufficient_local_support"
    gap_global = finite_float(candidate.get("observed_support_gap_global"), 0.0)
    gap_local = finite_float(candidate.get("observed_support_gap_local"), 0.0)
    if gap_global > float(max_support_gap_global) or gap_local > float(max_support_gap_local):
        return "outward_support_gap"
    if int(candidate.get("observed_support_near_count") or 0) < int(min_local_support):
        return "insufficient_local_support"
    if int(candidate.get("finite_support_count") or 0) < int(min_finite_support_count):
        return "insufficient_local_support"
    if finite_float(candidate.get("finite_support_track_coverage"), 0.0) < float(min_finite_track_coverage):
        return "insufficient_local_support"
    return None


def cumulative_local_candidate_key(candidate: dict[str, object]) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    tuple[float, float, float, float],
]:
    track_coverage = finite_float(candidate.get("finite_support_track_coverage"), 0.0)
    z_coverage = finite_float(candidate.get("finite_support_z_coverage"), 0.0)
    purity = finite_float(candidate.get("finite_support_purity"), 0.0)
    residual_p95 = finite_float(candidate.get("finite_support_residual_p95"), float("inf"))
    density = finite_float(candidate.get("finite_support_density"), 0.0)
    balance = finite_float(candidate.get("finite_support_balance"), 0.0)
    persistence = float(len(candidate.get("dedupe_cluster_windows", []) or []))
    return (
        -track_coverage,
        -z_coverage,
        -purity,
        residual_p95,
        -density,
        -balance,
        -persistence,
        candidate_rank_key(candidate),
    )


def cluster_debug_rows(clusters: list[list[dict[str, object]]], limit: int = 50) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cluster in clusters[: max(0, int(limit))]:
        if not cluster:
            continue
        rows.append(
            {
                "cluster_id": int(cluster[0].get("dedupe_cluster_id") or 0),
                "size": int(len(cluster)),
                "windows": [int(w) for w in cluster[0].get("dedupe_cluster_windows", [])],
                "candidates": [
                    {
                        "track_id": int(candidate.get("track_id") or 0),
                        "window": candidate.get("window"),
                        "local_rank": int(candidate.get("cluster_local_rank") or 0),
                        "plane_rms": candidate.get("plane_rms"),
                        "levels": candidate.get("levels"),
                        "z_min": candidate.get("z_min"),
                        "z_max": candidate.get("z_max"),
                        "condition_p95": candidate.get("condition_p95"),
                        "observed_outside_fraction": candidate.get("observed_outside_fraction"),
                        "observed_max_level_outside_fraction": candidate.get("observed_max_level_outside_fraction"),
                        "observed_local_support": candidate.get("observed_local_support"),
                        "finite_support_count": candidate.get("finite_support_count"),
                        "finite_support_track_coverage": candidate.get("finite_support_track_coverage"),
                        "finite_support_z_coverage": candidate.get("finite_support_z_coverage"),
                        "finite_support_purity": candidate.get("finite_support_purity"),
                        "finite_support_residual_p95": candidate.get("finite_support_residual_p95"),
                        "finite_support_density": candidate.get("finite_support_density"),
                    }
                    for candidate in cluster[:8]
                ],
            }
        )
    return rows


def cumulative_outside_metrics(
    candidates: list[dict[str, object]],
    points: np.ndarray,
    z_indices: np.ndarray,
    *,
    point_tol: float,
) -> dict[str, object]:
    if points.size == 0 or not candidates:
        return {
            "cumulative_outside_fraction": 0.0,
            "cumulative_max_level_outside_fraction": 0.0,
            "cumulative_lost_z_levels": 0,
        }
    outside_any = np.zeros(points.shape[0], dtype=bool)
    for candidate in candidates:
        plane = candidate_plane(candidate)
        if plane is None:
            continue
        plane_point, normal = plane
        outside_any |= ((points - plane_point) @ normal) > float(point_tol)
    max_level = 0.0
    lost = 0
    for zi in sorted(set(int(z) for z in z_indices)):
        mask = z_indices == zi
        total = int(np.sum(mask))
        if total <= 0:
            continue
        frac = float(np.sum(outside_any[mask])) / float(total)
        max_level = max(max_level, frac)
        if frac >= 1.0:
            lost += 1
    return {
        "cumulative_outside_fraction": float(np.mean(outside_any)),
        "cumulative_max_level_outside_fraction": float(max_level),
        "cumulative_lost_z_levels": int(lost),
    }


def candidate_outside_mask(
    candidate: dict[str, object],
    points: np.ndarray,
    *,
    point_tol: float,
) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    plane = candidate_plane(candidate)
    if plane is None:
        return np.zeros(points.shape[0], dtype=bool)
    plane_point, normal = plane
    return ((points - plane_point) @ normal) > float(point_tol)


def prepare_z_level_slices(z_indices: np.ndarray) -> list[np.ndarray]:
    if z_indices.size == 0:
        return []
    return [np.flatnonzero(z_indices == zi) for zi in sorted(set(int(z) for z in z_indices))]


def cumulative_metrics_from_mask(mask: np.ndarray, z_slices: list[np.ndarray]) -> dict[str, object]:
    if mask.size == 0:
        return {
            "cumulative_outside_fraction": 0.0,
            "cumulative_max_level_outside_fraction": 0.0,
            "cumulative_lost_z_levels": 0,
        }
    max_level = 0.0
    lost = 0
    for idx in z_slices:
        total = int(idx.size)
        if total <= 0:
            continue
        frac = float(np.sum(mask[idx])) / float(total)
        max_level = max(max_level, frac)
        if frac >= 1.0:
            lost += 1
    return {
        "cumulative_outside_fraction": float(np.mean(mask)),
        "cumulative_max_level_outside_fraction": float(max_level),
        "cumulative_lost_z_levels": int(lost),
    }


def normalized_lp_plane_signature(constraint: tuple[np.ndarray, float] | None) -> tuple[str, str, str, str] | None:
    if constraint is None:
        return None
    normal, offset = constraint
    arr = np.array(normal, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= EPS:
        return None
    arr = arr / norm
    off = float(offset) / norm
    return (float(arr[0]).hex(), float(arr[1]).hex(), float(arr[2]).hex(), float(off).hex())


class ReusableHalfspaceLpModel:
    def __init__(
        self,
        candidates: list[dict[str, object]],
        *,
        inside_tol: float,
        profile: "LpActivityEngine",
    ) -> None:
        self.inside_tol = float(inside_tol)
        self.profile = profile
        self.highs = Highs()
        self.highs.setOptionValue("output_flag", False)
        for _ in range(3):
            self.highs.addVar(-kHighsInf, kHighsInf)
        self.row_by_id: dict[int, int] = {}
        self.ub_by_id: dict[int, float] = {}
        build_started = time.perf_counter()
        for candidate in candidates:
            cid = int(candidate_track_id(candidate))
            if cid in self.row_by_id:
                continue
            halfspace = candidate_halfspace(candidate)
            if halfspace is None:
                continue
            normal, offset = halfspace
            row_index = len(self.row_by_id)
            ub = float(offset) + float(inside_tol)
            highs_add_row(
                self.highs,
                -kHighsInf,
                ub,
                [0, 1, 2],
                [float(normal[0]), float(normal[1]), float(normal[2])],
            )
            self.row_by_id[cid] = int(row_index)
            self.ub_by_id[cid] = float(ub)
        self.active_ids: set[int] = set(self.row_by_id)
        profile.add("reusable_model_build_seconds", time.perf_counter() - build_started)
        profile.add("reusable_model_rows", len(self.row_by_id))

    def solve(
        self,
        constraint_ids: tuple[int, ...],
        objective: tuple[np.ndarray, float],
        *,
        include_point: bool,
    ) -> tuple[bool, float | None, str, np.ndarray | None]:
        requested = set(int(v) for v in constraint_ids if int(v) in self.row_by_id)
        update_started = time.perf_counter()
        for cid in sorted(self.active_ids - requested):
            highs_change_row_bounds(self.highs, self.row_by_id[cid], -kHighsInf, kHighsInf)
        for cid in sorted(requested - self.active_ids):
            highs_change_row_bounds(self.highs, self.row_by_id[cid], -kHighsInf, self.ub_by_id[cid])
        self.active_ids = requested
        self.profile.add("row_bound_update_seconds", time.perf_counter() - update_started)
        self.profile.add("row_bound_update_calls", 1)

        normal, offset = objective
        objective_started = time.perf_counter()
        for i in range(3):
            self.highs.changeColCost(i, -float(normal[i]))
        self.profile.add("objective_update_seconds", time.perf_counter() - objective_started)
        self.profile.add("objective_update_calls", 1)

        solve_started = time.perf_counter()
        self.highs.run()
        self.profile.add("solve_seconds", time.perf_counter() - solve_started)
        self.profile.add("solve_halfspace_lp_calls", 1)
        extraction_started = time.perf_counter()
        status = self.highs.modelStatusToString(self.highs.getModelStatus())
        feasible = status in {"Optimal", "Objective bound", "Objective target"}
        optimum: float | None = None
        point: np.ndarray | None = None
        if feasible and status == "Optimal":
            optimum = -float(self.highs.getObjectiveValue()) - float(offset)
            if include_point:
                solution = self.highs.getSolution()
                point = np.array(solution.col_value, dtype=float)
        self.profile.add("solve_result_extraction_seconds", time.perf_counter() - extraction_started)
        self.profile.add("solve_result_extraction_count", 1)
        return bool(feasible), optimum, status, point


class LpActivityEngine:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.stage = "unknown"
        self.cache: dict[tuple[object, ...], dict[str, object]] = {}
        self.constraints_cache: dict[tuple[int, ...], list[tuple[np.ndarray, float]]] = {}
        self.reusable_models: dict[str, ReusableHalfspaceLpModel] = {}
        self.constraint_signatures: set[tuple[int, ...]] = set()
        self.objective_signatures: set[tuple[str, str, str, str]] = set()
        self.full_signatures: set[tuple[object, ...]] = set()
        self.stats: dict[str, float | int] = {
            "lp_face_activity_details_calls": 0,
            "solve_halfspace_lp_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cacheable_calls": 0,
            "duplicate_cacheable_calls": 0,
            "model_construction_seconds": 0.0,
            "solve_seconds": 0.0,
            "constraint_build_seconds": 0.0,
            "constraint_state_key_seconds": 0.0,
            "residual_summary_seconds": 0.0,
            "residual_summary_count": 0,
            "solve_result_extraction_seconds": 0.0,
            "solve_result_extraction_count": 0,
            "cache_lookup_seconds": 0.0,
            "reusable_model_build_seconds": 0.0,
            "reusable_model_rows": 0,
            "row_bound_update_seconds": 0.0,
            "row_bound_update_calls": 0,
            "objective_update_seconds": 0.0,
            "objective_update_calls": 0,
        }
        self.stage_stats: dict[str, dict[str, float | int]] = {}

    def _stage_stats(self) -> dict[str, float | int]:
        return self.stage_stats.setdefault(
            str(self.stage),
            {
                "lp_face_activity_details_calls": 0,
                "solve_halfspace_lp_calls": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "model_construction_seconds": 0.0,
                "solve_seconds": 0.0,
                "constraint_build_seconds": 0.0,
                "constraint_state_key_seconds": 0.0,
                "residual_summary_seconds": 0.0,
                "residual_summary_count": 0,
                "solve_result_extraction_seconds": 0.0,
                "solve_result_extraction_count": 0,
                "reusable_model_build_seconds": 0.0,
                "reusable_model_rows": 0,
                "row_bound_update_seconds": 0.0,
                "row_bound_update_calls": 0,
                "objective_update_seconds": 0.0,
                "objective_update_calls": 0,
            },
        )

    def add(self, key: str, value: float | int) -> None:
        if isinstance(value, int):
            self.stats[key] = int(self.stats.get(key, 0)) + int(value)
            stage = self._stage_stats()
            stage[key] = int(stage.get(key, 0)) + int(value)
        else:
            self.stats[key] = float(self.stats.get(key, 0.0)) + float(value)
            stage = self._stage_stats()
            stage[key] = float(stage.get(key, 0.0)) + float(value)

    def cache_key(
        self,
        base_candidates: list[dict[str, object]] | None,
        target_candidate: dict[str, object],
        *,
        inside_tol: float,
        face_activity_slack: float,
        include_point: bool,
        base_candidate_ids: tuple[int, ...] | None = None,
    ) -> tuple[object, ...] | None:
        objective = normalized_lp_plane_signature(candidate_halfspace(target_candidate))
        if objective is None:
            return None
        key_started = time.perf_counter()
        if base_candidate_ids is None:
            constraint_ids = tuple(sorted(int(candidate_track_id(candidate)) for candidate in (base_candidates or [])))
        else:
            constraint_ids = tuple(sorted(int(v) for v in base_candidate_ids))
        self.add("constraint_state_key_seconds", time.perf_counter() - key_started)
        self.constraint_signatures.add(constraint_ids)
        self.objective_signatures.add(objective)
        key = (
            constraint_ids,
            objective,
            float(inside_tol).hex(),
            float(face_activity_slack).hex(),
            bool(include_point),
        )
        if key in self.full_signatures:
            self.add("duplicate_cacheable_calls", 1)
        self.full_signatures.add(key)
        return key

    def constraints_for(
        self,
        constraint_ids: tuple[int, ...],
        base_candidates: list[dict[str, object]],
    ) -> list[tuple[np.ndarray, float]]:
        cached = self.constraints_cache.get(constraint_ids)
        if cached is not None:
            return cached
        constraints = lp_constraints_for_candidates(base_candidates)
        self.constraints_cache[constraint_ids] = constraints
        return constraints

    def set_reusable_universe(
        self,
        stage: str,
        candidates: list[dict[str, object]],
        *,
        inside_tol: float,
    ) -> None:
        self.stage = str(stage)
        self.reusable_models[str(stage)] = ReusableHalfspaceLpModel(
            candidates,
            inside_tol=float(inside_tol),
            profile=self,
        )

    def reusable_model(self) -> ReusableHalfspaceLpModel | None:
        return self.reusable_models.get(str(self.stage))

    def summary(self) -> dict[str, object]:
        return {
            "engine": "reused-highs-with-exact-cache",
            "enabled": bool(self.enabled),
            "lp_face_activity_details_calls": int(self.stats.get("lp_face_activity_details_calls", 0)),
            "solve_halfspace_lp_calls": int(self.stats.get("solve_halfspace_lp_calls", 0)),
            "cache_hits": int(self.stats.get("cache_hits", 0)),
            "cache_misses": int(self.stats.get("cache_misses", 0)),
            "cacheable_calls": int(self.stats.get("cacheable_calls", 0)),
            "duplicate_cacheable_calls": int(self.stats.get("duplicate_cacheable_calls", 0)),
            "unique_constraint_set_signatures": int(len(self.constraint_signatures)),
            "unique_objective_signatures": int(len(self.objective_signatures)),
            "unique_constraint_objective_signatures": int(len(self.full_signatures)),
            "cached_constraint_matrices": int(len(self.constraints_cache)),
            "model_construction_seconds": as_json_float(float(self.stats.get("model_construction_seconds", 0.0))),
            "solve_seconds": as_json_float(float(self.stats.get("solve_seconds", 0.0))),
            "constraint_build_seconds": as_json_float(float(self.stats.get("constraint_build_seconds", 0.0))),
            "constraint_state_key_seconds": as_json_float(float(self.stats.get("constraint_state_key_seconds", 0.0))),
            "residual_summary_seconds": as_json_float(float(self.stats.get("residual_summary_seconds", 0.0))),
            "residual_summary_count": int(self.stats.get("residual_summary_count", 0)),
            "solve_result_extraction_seconds": as_json_float(float(self.stats.get("solve_result_extraction_seconds", 0.0))),
            "solve_result_extraction_count": int(self.stats.get("solve_result_extraction_count", 0)),
            "cache_lookup_seconds": as_json_float(float(self.stats.get("cache_lookup_seconds", 0.0))),
            "reusable_model_build_seconds": as_json_float(float(self.stats.get("reusable_model_build_seconds", 0.0))),
            "reusable_model_rows": int(self.stats.get("reusable_model_rows", 0)),
            "row_bound_update_seconds": as_json_float(float(self.stats.get("row_bound_update_seconds", 0.0))),
            "row_bound_update_calls": int(self.stats.get("row_bound_update_calls", 0)),
            "objective_update_seconds": as_json_float(float(self.stats.get("objective_update_seconds", 0.0))),
            "objective_update_calls": int(self.stats.get("objective_update_calls", 0)),
            "stage_stats": {
                key: {
                    sub_key: (as_json_float(float(value)) if isinstance(value, float) else int(value))
                    for sub_key, value in row.items()
                }
                for key, row in self.stage_stats.items()
            },
        }


def lp_face_activity_details(
    base_candidates: list[dict[str, object]] | None,
    target_candidate: dict[str, object],
    *,
    inside_tol: float,
    face_activity_slack: float,
    include_point: bool = False,
    lp_engine: LpActivityEngine | None = None,
    base_candidate_ids: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if lp_engine is not None:
        lp_engine.add("lp_face_activity_details_calls", 1)
    target_constraint = candidate_halfspace(target_candidate)
    if target_constraint is None:
        return {
            "feasible": False,
            "active": False,
            "solver_status": "invalid_candidate",
            "raw_lp_optimum": None,
            "face_activity_slack": as_json_float(float(face_activity_slack)),
            "effective_face_margin": None,
            "optimum_point": None,
            "residual_summary": None,
        }

    cache_key: tuple[object, ...] | None = None
    if lp_engine is not None and lp_engine.enabled:
        lookup_started = time.perf_counter()
        cache_key = lp_engine.cache_key(
            base_candidates,
            target_candidate,
            inside_tol=float(inside_tol),
            face_activity_slack=float(face_activity_slack),
            include_point=bool(include_point),
            base_candidate_ids=base_candidate_ids,
        )
        lp_engine.add("cache_lookup_seconds", time.perf_counter() - lookup_started)
        if cache_key is not None:
            lp_engine.add("cacheable_calls", 1)
            cached = lp_engine.cache.get(cache_key)
            if cached is not None:
                lp_engine.add("cache_hits", 1)
                return dict(cached)
            lp_engine.add("cache_misses", 1)

    reusable = lp_engine.reusable_model() if lp_engine is not None else None
    constraints: list[tuple[np.ndarray, float]] = []
    needs_constraints = bool(include_point or reusable is None or cache_key is None)
    if needs_constraints:
        constraints_started = time.perf_counter()
        if lp_engine is not None and lp_engine.enabled and cache_key is not None:
            constraints = lp_engine.constraints_for(cache_key[0], base_candidates or [])  # type: ignore[arg-type]
        else:
            constraints = lp_constraints_for_candidates(base_candidates or [])
        if lp_engine is not None:
            lp_engine.add("constraint_build_seconds", time.perf_counter() - constraints_started)
    if not include_point:
        if reusable is not None and cache_key is not None:
            feasible, optimum, status, _ = reusable.solve(
                cache_key[0],  # type: ignore[arg-type]
                target_constraint,
                include_point=False,
            )
        else:
            model_before = float(lp_engine.stats.get("model_construction_seconds", 0.0)) if lp_engine is not None else 0.0
            solve_before = float(lp_engine.stats.get("solve_seconds", 0.0)) if lp_engine is not None else 0.0
            feasible, optimum, status = solve_halfspace_lp(
                constraints,
                inside_tol=float(inside_tol),
                objective=target_constraint,
                profile=lp_engine.stats if lp_engine is not None else None,
            )
            if lp_engine is not None:
                stage = lp_engine._stage_stats()
                stage["solve_halfspace_lp_calls"] = int(stage.get("solve_halfspace_lp_calls", 0)) + 1
                stage["model_construction_seconds"] = float(stage.get("model_construction_seconds", 0.0)) + (
                    float(lp_engine.stats.get("model_construction_seconds", 0.0)) - model_before
                )
                stage["solve_seconds"] = float(stage.get("solve_seconds", 0.0)) + (
                    float(lp_engine.stats.get("solve_seconds", 0.0)) - solve_before
                )
        effective = float(optimum) - float(face_activity_slack) if optimum is not None else None
        result = {
            "feasible": bool(feasible),
            "active": bool(feasible and optimum is not None and effective is not None and effective > 0.0),
            "solver_status": status,
            "raw_lp_optimum": as_json_float(float(optimum)) if optimum is not None else None,
            "face_activity_slack": as_json_float(float(face_activity_slack)),
            "effective_face_margin": as_json_float(float(effective)) if effective is not None else None,
            "optimum_point": None,
            "residual_summary": None,
        }
        if cache_key is not None and lp_engine is not None and lp_engine.enabled:
            lp_engine.cache[cache_key] = dict(result)
        return result

    normal, offset = target_constraint
    if reusable is not None and cache_key is not None:
        feasible, optimum, status, point = reusable.solve(
            cache_key[0],  # type: ignore[arg-type]
            target_constraint,
            include_point=True,
        )
    else:
        model_started = time.perf_counter()
        highs = Highs()
        highs.setOptionValue("output_flag", False)
        for _ in range(3):
            highs.addVar(-kHighsInf, kHighsInf)
        for i in range(3):
            highs.changeColCost(i, -float(normal[i]))
        for constraint_normal, constraint_offset in constraints:
            highs_add_row(
                highs,
                -kHighsInf,
                float(constraint_offset) + float(inside_tol),
                [0, 1, 2],
                [float(constraint_normal[0]), float(constraint_normal[1]), float(constraint_normal[2])],
            )
        if lp_engine is not None:
            lp_engine.add("model_construction_seconds", time.perf_counter() - model_started)
            lp_engine.add("solve_halfspace_lp_calls", 1)
        solve_started = time.perf_counter()
        highs.run()
        if lp_engine is not None:
            lp_engine.add("solve_seconds", time.perf_counter() - solve_started)
        status = highs.modelStatusToString(highs.getModelStatus())
        feasible = status in {"Optimal", "Objective bound", "Objective target"}
        optimum = None
        point = None
        if feasible and status == "Optimal":
            optimum = -float(highs.getObjectiveValue()) - float(offset)
            solution = highs.getSolution()
            point = np.array(solution.col_value, dtype=float)
    residual_summary: dict[str, object] | None = None
    if feasible and status == "Optimal":
        residual_started = time.perf_counter()
        residuals: list[tuple[float, int]] = []
        for candidate in base_candidates or []:
            candidate_constraint = candidate_halfspace(candidate)
            if candidate_constraint is None:
                continue
            constraint_normal, constraint_offset = candidate_constraint
            slack = float(constraint_offset) + float(inside_tol) - float(constraint_normal @ point)
            residuals.append((slack, candidate_track_id(candidate)))
        residuals.sort(key=lambda item: item[0])
        min_slack = residuals[0][0] if residuals else None
        residual_summary = {
            "min_slack": as_json_float(float(min_slack)) if min_slack is not None else None,
            "max_violation": as_json_float(float(max(0.0, -float(min_slack)))) if min_slack is not None else None,
            "active_constraints_sample": [int(tid) for _, tid in residuals[:12]],
        }
        if lp_engine is not None:
            lp_engine.add("residual_summary_seconds", time.perf_counter() - residual_started)
            lp_engine.add("residual_summary_count", 1)
    effective = float(optimum) - float(face_activity_slack) if optimum is not None else None
    result = {
        "feasible": bool(feasible),
        "active": bool(feasible and optimum is not None and effective is not None and effective > 0.0),
        "solver_status": status,
        "raw_lp_optimum": as_json_float(float(optimum)) if optimum is not None else None,
        "face_activity_slack": as_json_float(float(face_activity_slack)),
        "effective_face_margin": as_json_float(float(effective)) if effective is not None else None,
        "optimum_point": as_json_point(point) if point is not None else None,
        "residual_summary": residual_summary,
    }
    if cache_key is not None and lp_engine is not None and lp_engine.enabled:
        lp_engine.cache[cache_key] = dict(result)
    return result


def lp_candidate_activity(
    accepted: list[dict[str, object]],
    candidate: dict[str, object],
    bbox_planes: list[dict[str, object]],
    *,
    inside_tol: float,
    activity_tol: float,
) -> dict[str, object]:
    candidate_constraint = candidate_halfspace(candidate)
    if candidate_constraint is None:
        return {"feasible": False, "active": False, "redundant": False, "status": "invalid_candidate"}

    constraints: list[tuple[np.ndarray, float]] = []
    for item in accepted + bbox_planes:
        halfspace = candidate_halfspace(item)
        if halfspace is not None:
            constraints.append(halfspace)

    feasible_before, optimum, status_before = solve_halfspace_lp(
        constraints,
        inside_tol=float(inside_tol),
        objective=candidate_constraint,
    )
    if not feasible_before:
        return {
            "feasible": False,
            "active": False,
            "redundant": False,
            "status": status_before,
            "lp_optimum": None,
        }
    active = optimum is None or float(optimum) > float(activity_tol)

    feasible_after, _, status_after = solve_halfspace_lp(
        constraints + [candidate_constraint],
        inside_tol=float(inside_tol),
        objective=None,
    )
    return {
        "feasible": bool(feasible_after),
        "active": bool(active),
        "redundant": bool(not active),
        "status": status_after,
        "status_before": status_before,
        "lp_optimum": as_json_float(float(optimum)) if optimum is not None else None,
    }


def bbox_trial_candidates(points: np.ndarray, *, margin: float) -> list[dict[str, object]]:
    if points.size == 0:
        return []
    mn = np.min(points, axis=0) - float(margin)
    mx = np.max(points, axis=0) + float(margin)
    out: list[dict[str, object]] = []
    specs = [
        (np.array([1.0, 0.0, 0.0]), np.array([mx[0], 0.0, 0.0])),
        (np.array([-1.0, 0.0, 0.0]), np.array([mn[0], 0.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, mx[1], 0.0])),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, mn[1], 0.0])),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, mx[2]])),
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, mn[2]])),
    ]
    for i, (normal, point) in enumerate(specs):
        out.append(
            {
                "track_id": -1000 - i,
                "plane_normal": as_json_point(normal),
                "plane_centroid": as_json_point(point),
                "hull": [],
                "hull_area": 1.0,
                "candidate_score": 0.0,
                "is_trial_bbox_plane": True,
            }
        )
    return out


def select_compatible_candidates(
    clusters: list[list[dict[str, object]]],
    *,
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    mode: str,
    point_tol: float,
    max_global_outside_frac: float,
    max_level_outside_frac: float,
    bbox_margin: float,
    min_local_support: int,
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
    max_alternatives_per_cluster: int,
    max_candidates: int,
    outside_mode: str,
    activity_check: str,
    activity_tol: float,
    valley_candidate_mode: str,
    valley_min_confidence_median: float,
    valley_min_confidence_p10: float,
    valley_min_two_side_fraction: float,
    valley_max_one_side_fraction: float,
    valley_max_distance_p95: float,
    valley_max_track_smoothness: float,
    valley_max_support_gap_global: float,
    valley_max_support_gap_local: float,
    valley_min_support_near_count: int,
    valley_min_scale_persistence: int,
    valley_min_track_levels: int,
    valley_max_plane_rms: float,
    valley_min_hull_area: float,
    valley_min_finite_support_count: int,
    valley_min_finite_track_coverage: float,
    initial_accepted: list[dict[str, object]] | None = None,
    max_additions: int | None = None,
    initial_label: str = "initial",
    addition_selection_mode: str = "rank",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    selection_started = time.perf_counter()
    annotated_clusters: list[list[dict[str, object]]] = []
    rejection_counts: dict[str, int] = {}
    representatives_rank: list[dict[str, object]] = []
    representatives_compatible: list[dict[str, object]] = []
    local_support_distance = max(float(point_tol) * 2.0, 0.05)
    z_slices = prepare_z_level_slices(trusted_z_indices)
    outside_masks: list[np.ndarray] = []
    timing = {
        "annotate_seconds": 0.0,
        "outside_seconds": 0.0,
        "activity_seconds": 0.0,
    }
    counters = {
        "trial_checks": 0,
        "geometry_reconstructions": 0,
        "lp_activity_checks": 0,
        "lp_solves": 0,
        "estimated_geometry_plane_triples": 0,
    }

    annotate_started = time.perf_counter()
    for cluster in clusters:
        annotated: list[dict[str, object]] = []
        for candidate in cluster:
            out = dict(candidate)
            out.update(
                observed_cloud_stats(
                    out,
                    trusted_points,
                    trusted_z_indices,
                    point_tol=float(point_tol),
                    local_support_distance=local_support_distance,
                    finite_support_plane_distance=float(finite_support_plane_distance),
                    finite_support_hull_margin=float(finite_support_hull_margin),
                    finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
                )
            )
            outside_masks.append(
                candidate_outside_mask(
                    out,
                    trusted_points,
                    point_tol=float(point_tol),
                )
            )
            out["_outside_mask_index"] = int(len(outside_masks) - 1)
            annotated.append(out)
        annotated.sort(key=candidate_rank_key)
        representatives_rank.append(dict(annotated[0]))
        if str(addition_selection_mode) == "support-diverse":
            compatible_sorted = sorted(annotated, key=support_diverse_candidate_key)
        else:
            compatible_sorted = sorted(annotated, key=compatible_candidate_key)
        chosen = dict(compatible_sorted[0])
        chosen["compatible_selection_reason"] = "best_observed_compatibility"
        chosen["rank_representative_track_id"] = annotated[0].get("track_id")
        chosen["rank_representative_window"] = annotated[0].get("window")
        representatives_compatible.append(chosen)
        for alt in compatible_sorted[1:]:
            rejection_counts["cluster_alternative"] = rejection_counts.get("cluster_alternative", 0) + 1
        annotated_clusters.append(compatible_sorted)
    timing["annotate_seconds"] = time.perf_counter() - annotate_started

    if mode == "rank":
        selected = representatives_rank
        for candidate in selected:
            candidate.pop("_outside_mask_index", None)
            candidate["selection_status"] = "accepted"
            candidate["selection_reason"] = "rank_representative"
        return selected[:max_candidates], {
            "candidate_selection_mode": "rank",
            "selection_rejection_counts": rejection_counts,
            "selected_representatives": int(min(len(selected), max_candidates)),
            "accepted_halfspaces": int(min(len(selected), max_candidates)),
            "cluster_details": cluster_debug_rows(annotated_clusters),
        }

    if mode == "compatible":
        selected = representatives_compatible
        for candidate in selected:
            candidate.pop("_outside_mask_index", None)
            candidate["selection_status"] = "accepted"
            candidate["selection_reason"] = "compatible_representative"
        return selected[:max_candidates], {
            "candidate_selection_mode": "compatible",
            "selection_rejection_counts": rejection_counts,
            "selected_representatives": int(min(len(selected), max_candidates)),
            "accepted_halfspaces": int(min(len(selected), max_candidates)),
            "cluster_details": cluster_debug_rows(annotated_clusters),
        }

    accepted: list[dict[str, object]] = []
    initial_count = 0
    if initial_accepted:
        for item in initial_accepted:
            seeded = dict(item)
            seeded.pop("_outside_mask_index", None)
            seeded["selection_status"] = "accepted"
            seeded["selection_reason"] = str(initial_label)
            seeded.setdefault("candidate_origin", str(initial_label))
            accepted.append(seeded)
        initial_count = len(accepted)
    bbox_planes = bbox_trial_candidates(trusted_points, margin=float(bbox_margin))
    if str(outside_mode) == "incremental" and accepted:
        cumulative_mask = np.zeros(trusted_points.shape[0], dtype=bool)
        for item in accepted:
            cumulative_mask |= candidate_outside_mask(item, trusted_points, point_tol=float(point_tol))
    else:
        cumulative_mask = np.zeros(trusted_points.shape[0], dtype=bool)
    if mode == "cumulative-local":
        cluster_queue: list[tuple[tuple[float, ...], int, list[dict[str, object]]]] = []
        for cluster_id, annotated in enumerate(annotated_clusters):
            local_sorted = sorted(annotated, key=cumulative_local_candidate_key)
            local_sorted = local_sorted[: max(1, int(max_alternatives_per_cluster))]
            cluster_queue.append((cumulative_local_candidate_key(local_sorted[0]), cluster_id, local_sorted))
        cluster_queue.sort(key=lambda item: item[0])
        iterable: list[tuple[int | None, list[dict[str, object]]]] = [
            (cluster_id, alternatives) for _, cluster_id, alternatives in cluster_queue
        ]
    else:
        if str(addition_selection_mode) == "novelty":
            sorted_reps = sorted(representatives_compatible, key=novelty_candidate_key)
        elif str(addition_selection_mode) == "support-diverse":
            sorted_reps = sorted(representatives_compatible, key=support_diverse_candidate_key)
        else:
            sorted_reps = sorted(representatives_compatible, key=compatible_candidate_key)
        iterable = [(None, [candidate]) for candidate in sorted_reps]

    alternative_attempts: list[dict[str, object]] = []
    accepted_cluster_ids: set[int] = set()

    def record_attempt(candidate: dict[str, object], reason: str) -> None:
        alternative_attempts.append(
            {
                "cluster_id": candidate.get("dedupe_cluster_id"),
                "track_id": candidate.get("track_id"),
                "window": candidate.get("window"),
                "alternative_rank": candidate.get("cumulative_local_alternative_rank"),
                "reason": reason,
            }
        )

    for cluster_id, alternatives in iterable:
        accepted_additions = len(accepted) - int(initial_count)
        if len(accepted) >= int(max_candidates) or (max_additions is not None and accepted_additions >= int(max_additions)):
            break
        if cluster_id is not None and cluster_id in accepted_cluster_ids:
            continue
        for alternative_rank, candidate_in in enumerate(alternatives):
            candidate = dict(candidate_in)
            if cluster_id is not None:
                candidate["cumulative_local_alternative_rank"] = int(alternative_rank)
                candidate["cumulative_local_alternatives_total"] = int(len(alternatives))
            if len(accepted) >= int(max_candidates):
                break
            if max_additions is not None and (len(accepted) - int(initial_count)) >= int(max_additions):
                break
            if str(valley_candidate_mode) == "quality":
                valley_rejection = valley_quality_rejection_reason(
                    candidate,
                    min_confidence_median=float(valley_min_confidence_median),
                    min_confidence_p10=float(valley_min_confidence_p10),
                    min_two_side_fraction=float(valley_min_two_side_fraction),
                    max_one_side_fraction=float(valley_max_one_side_fraction),
                    max_distance_p95=float(valley_max_distance_p95),
                    max_track_smoothness=float(valley_max_track_smoothness),
                    max_support_gap_global=float(valley_max_support_gap_global),
                    max_support_gap_local=float(valley_max_support_gap_local),
                    min_local_support=int(valley_min_support_near_count),
                    min_scale_persistence=int(valley_min_scale_persistence),
                    min_track_levels=int(valley_min_track_levels),
                    max_plane_rms=float(valley_max_plane_rms),
                    min_hull_area=float(valley_min_hull_area),
                    min_finite_support_count=int(valley_min_finite_support_count),
                    min_finite_track_coverage=float(valley_min_finite_track_coverage),
                )
                if valley_rejection is not None:
                    candidate["selection_status"] = "rejected"
                    candidate["selection_reason"] = valley_rejection
                    candidate["valley_quality_rejection_reason"] = valley_rejection
                    rejection_counts[valley_rejection] = rejection_counts.get(valley_rejection, 0) + 1
                    record_attempt(candidate, valley_rejection)
                    continue
            support_for_gate = (
                int(candidate.get("finite_support_count") or 0)
                if mode == "cumulative-local"
                else int(candidate.get("observed_local_support") or 0)
            )
            if support_for_gate < int(min_local_support):
                candidate["selection_status"] = "rejected"
                candidate["selection_reason"] = "insufficient_local_support"
                rejection_counts["insufficient_local_support"] = rejection_counts.get("insufficient_local_support", 0) + 1
                record_attempt(candidate, "insufficient_local_support")
                continue
            counters["trial_checks"] += 1
            outside_started = time.perf_counter()
            if str(outside_mode) == "incremental":
                mask_index = int(candidate.get("_outside_mask_index") or 0)
                trial_mask = cumulative_mask | outside_masks[mask_index]
                outside_metrics = cumulative_metrics_from_mask(trial_mask, z_slices)
            else:
                trial = accepted + [candidate]
                outside_metrics = cumulative_outside_metrics(
                    trial,
                    trusted_points,
                    trusted_z_indices,
                    point_tol=float(point_tol),
                )
                trial_mask = np.zeros(0, dtype=bool)
            timing["outside_seconds"] += time.perf_counter() - outside_started
            if outside_metrics["cumulative_outside_fraction"] > float(max_global_outside_frac):
                candidate["selection_status"] = "rejected"
                candidate["selection_reason"] = "observed_global_violation"
                rejection_counts["observed_global_violation"] = rejection_counts.get("observed_global_violation", 0) + 1
                record_attempt(candidate, "observed_global_violation")
                continue
            if outside_metrics["cumulative_max_level_outside_fraction"] > float(max_level_outside_frac):
                candidate["selection_status"] = "rejected"
                candidate["selection_reason"] = "observed_level_violation"
                rejection_counts["observed_level_violation"] = rejection_counts.get("observed_level_violation", 0) + 1
                record_attempt(candidate, "observed_level_violation")
                continue
            activity_started = time.perf_counter()
            if str(activity_check) == "lp":
                activity = lp_candidate_activity(
                    accepted,
                    candidate,
                    bbox_planes,
                    inside_tol=max(float(point_tol), 0.03),
                    activity_tol=float(activity_tol),
                )
                counters["lp_activity_checks"] += 1
                counters["lp_solves"] += 2
                candidate["lp_activity_status"] = activity.get("status")
                candidate["lp_activity_status_before"] = activity.get("status_before")
                candidate["lp_activity_optimum"] = activity.get("lp_optimum")
                if not bool(activity.get("feasible")):
                    candidate["selection_status"] = "rejected"
                    candidate["selection_reason"] = "intersection_empty"
                    rejection_counts["intersection_empty"] = rejection_counts.get("intersection_empty", 0) + 1
                    record_attempt(candidate, "intersection_empty")
                    timing["activity_seconds"] += time.perf_counter() - activity_started
                    continue
                if bool(activity.get("redundant")) and len(accepted) >= 6:
                    candidate["selection_status"] = "rejected"
                    candidate["selection_reason"] = "redundant"
                    rejection_counts["redundant"] = rejection_counts.get("redundant", 0) + 1
                    record_attempt(candidate, "redundant")
                    timing["activity_seconds"] += time.perf_counter() - activity_started
                    continue
            else:
                trial = accepted + [candidate]
                n_planes = len(trial) + len(bbox_planes)
                counters["estimated_geometry_plane_triples"] += (
                    n_planes * (n_planes - 1) * (n_planes - 2) // 6 if n_planes >= 3 else 0
                )
                counters["geometry_reconstructions"] += 1
                trial_rec = reconstruct_polyhedron_from_halfspaces(
                    trial + bbox_planes,
                    inside_tol=max(float(point_tol), 0.03),
                    vertex_tol=0.02,
                )
                if len(trial_rec.get("vertices", [])) == 0:
                    candidate["selection_status"] = "rejected"
                    candidate["selection_reason"] = "intersection_empty"
                    rejection_counts["intersection_empty"] = rejection_counts.get("intersection_empty", 0) + 1
                    record_attempt(candidate, "intersection_empty")
                    timing["activity_seconds"] += time.perf_counter() - activity_started
                    continue
                active_sources = {int(i) for i in trial_rec.get("face_candidate_indices", [])}
                candidate_source = len(trial) - 1
                if candidate_source not in active_sources and len(accepted) >= 6:
                    candidate["selection_status"] = "rejected"
                    candidate["selection_reason"] = "redundant"
                    rejection_counts["redundant"] = rejection_counts.get("redundant", 0) + 1
                    record_attempt(candidate, "redundant")
                    timing["activity_seconds"] += time.perf_counter() - activity_started
                    continue
            timing["activity_seconds"] += time.perf_counter() - activity_started
            candidate["selection_status"] = "accepted"
            candidate["selection_reason"] = "cumulative_local_compatible" if mode == "cumulative-local" else "cumulative_compatible"
            candidate.update({k: as_json_float(float(v)) if isinstance(v, float) else v for k, v in outside_metrics.items()})
            candidate.pop("_outside_mask_index", None)
            accepted.append(candidate)
            if str(outside_mode) == "incremental":
                cumulative_mask = trial_mask
            if cluster_id is not None:
                accepted_cluster_ids.add(int(cluster_id))
            record_attempt(candidate, "accepted")
            break

    if str(outside_mode) == "incremental":
        final_metrics = cumulative_metrics_from_mask(cumulative_mask, z_slices)
        legacy_final_metrics = cumulative_outside_metrics(
            accepted,
            trusted_points,
            trusted_z_indices,
            point_tol=float(point_tol),
        )
        outside_equivalence = {
            "global_delta": as_json_float(
                abs(float(final_metrics["cumulative_outside_fraction"]) - float(legacy_final_metrics["cumulative_outside_fraction"]))
            ),
            "max_level_delta": as_json_float(
                abs(float(final_metrics["cumulative_max_level_outside_fraction"]) - float(legacy_final_metrics["cumulative_max_level_outside_fraction"]))
            ),
            "lost_z_delta": int(final_metrics["cumulative_lost_z_levels"]) - int(legacy_final_metrics["cumulative_lost_z_levels"]),
        }
    else:
        final_metrics = cumulative_outside_metrics(
            accepted,
            trusted_points,
            trusted_z_indices,
            point_tol=float(point_tol),
        )
        outside_equivalence = None
    for cluster in annotated_clusters:
        for candidate in cluster:
            candidate.pop("_outside_mask_index", None)
    timing["total_seconds"] = time.perf_counter() - selection_started
    return accepted, {
        "candidate_selection_mode": mode,
        "w2_addition_selection_mode": str(addition_selection_mode),
        "compatibility_outside_mode": str(outside_mode),
        "compatibility_activity_check": str(activity_check),
        "compatibility_activity_tol": float(activity_tol),
        "valley_candidate_mode": str(valley_candidate_mode),
        "valley_quality_thresholds": {
            "min_confidence_median": float(valley_min_confidence_median),
            "min_confidence_p10": float(valley_min_confidence_p10),
            "min_two_side_fraction": float(valley_min_two_side_fraction),
            "max_one_side_fraction": float(valley_max_one_side_fraction),
            "max_distance_p95": float(valley_max_distance_p95),
            "max_track_smoothness": float(valley_max_track_smoothness),
            "max_support_gap_global": float(valley_max_support_gap_global),
            "max_support_gap_local": float(valley_max_support_gap_local),
            "min_support_near_count": int(valley_min_support_near_count),
            "min_scale_persistence": int(valley_min_scale_persistence),
            "min_track_levels": int(valley_min_track_levels),
            "max_plane_rms": float(valley_max_plane_rms),
            "min_hull_area": float(valley_min_hull_area),
            "min_finite_support_count": int(valley_min_finite_support_count),
            "min_finite_track_coverage": float(valley_min_finite_track_coverage),
        },
        "selection_rejection_counts": rejection_counts,
        "selected_representatives": int(len(representatives_compatible)),
        "accepted_halfspaces": int(len(accepted)),
        "initial_accepted_halfspaces": int(initial_count),
        "accepted_additions": int(max(0, len(accepted) - int(initial_count))),
        "max_additions": int(max_additions) if max_additions is not None else None,
        "cumulative_local_alternative_attempts": int(len(alternative_attempts)),
        "cumulative_local_attempt_log": alternative_attempts[:200],
        "cumulative_selection_counters": counters,
        "cumulative_selection_timing": {k: as_json_float(float(v)) for k, v in timing.items()},
        "cumulative_outside_equivalence": outside_equivalence,
        "cumulative_selection_metrics": {
            k: as_json_float(float(v)) if isinstance(v, float) else v
            for k, v in final_metrics.items()
        },
        "cluster_details": cluster_debug_rows(annotated_clusters),
    }
