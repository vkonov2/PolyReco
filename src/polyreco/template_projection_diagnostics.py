"""Reference-only checks of finite-face projections and contour preprocessing.

The triangle union is evaluated exactly on the sampled lines.  Connecting its
outer ray endpoints is a visualization, not an exact union polygon: radial
outer distances alone cannot describe holes or disconnected components.
"""

from __future__ import annotations

import numpy as np

from polyreco.rms_reprojection import (
    contour_basis,
    convex_hull_2d,
    polar_radial_error_summary,
    reprojection_directions,
)


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _array(values: np.ndarray) -> list:
    arr = np.asarray(values)
    if arr.ndim > 1:
        return [_array(row) for row in arr]
    return [_number(value) for value in arr]


def _summary(values: list | np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"count": 0, "mean_signed": None, "rms": None, "median_absolute": None, "p95_absolute": None, "max_absolute": None}
    return {
        "count": int(arr.size),
        "mean_signed": float(np.mean(arr)),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "median_absolute": float(np.median(np.abs(arr))),
        "p95_absolute": float(np.percentile(np.abs(arr), 95)),
        "max_absolute": float(np.max(np.abs(arr))),
    }


def merge_intervals(intervals: np.ndarray | list, *, tol: float = 1e-9) -> np.ndarray:
    """Union of closed intervals, including touching intervals and point hits."""
    values = np.asarray(intervals, dtype=float).reshape(-1, 2)
    values = values[np.all(np.isfinite(values), axis=1)]
    if not len(values):
        return np.empty((0, 2), dtype=float)
    values = np.sort(values, axis=1)
    values = values[np.argsort(values[:, 0], kind="stable")]
    result = [values[0].tolist()]
    for low, high in values[1:]:
        if low <= result[-1][1] + tol:
            result[-1][1] = max(result[-1][1], float(high))
        else:
            result.append([float(low), float(high)])
    return np.asarray(result, dtype=float)


def triangle_union_line_intervals(
    triangles_2d: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    rays: bool = True,
    tol: float = 1e-9,
) -> list[np.ndarray]:
    """Intersect oriented lines/rays with the union of finite 2D triangles.

    Parameters on each line satisfy ``origin + t * direction``.  The clipping
    is performed on each triangle before interval union, so neither internal
    triangulation edges nor overlapping projected faces introduce false gaps.
    Projected zero-area triangles contribute only a boundary to a closed mesh
    and are omitted; the caller reports their number separately.
    """
    tris = np.asarray(triangles_2d, dtype=float).reshape(-1, 3, 2)
    dirs = np.asarray(directions, dtype=float).reshape(-1, 2)
    centers = np.asarray(origins, dtype=float)
    if centers.shape == (2,):
        centers = np.broadcast_to(centers, dirs.shape)
    if centers.shape != dirs.shape or not np.all(np.isfinite(dirs)):
        raise ValueError("Expected one finite 2D origin per direction")
    if np.any(np.linalg.norm(dirs, axis=1) <= tol):
        raise ValueError("Line directions must be nonzero")
    edges = np.roll(tris, -1, axis=1) - tris
    areas = edges[:, 0, 0] * (-edges[:, 2, 1]) - edges[:, 0, 1] * (-edges[:, 2, 0])
    keep = np.abs(areas) > tol * np.max(np.linalg.norm(edges, axis=2), axis=1)
    tris, edges, areas = tris[keep], edges[keep], areas[keep]
    if not len(tris):
        return [np.empty((0, 2), dtype=float) for _ in dirs]
    relative = tris[None, :, :, :] - centers[:, None, None, :]
    direction_norms = np.linalg.norm(dirs, axis=1)
    along = np.sum(relative * dirs[:, None, None, :], axis=3) / (direction_norms[:, None, None] ** 2)
    side = dirs[:, None, None, 0] * relative[:, :, :, 1] - dirs[:, None, None, 1] * relative[:, :, :, 0]
    side[np.abs(side) <= tol * direction_norms[:, None, None]] = 0.0
    next_side, next_along = np.roll(side, -1, axis=2), np.roll(along, -1, axis=2)
    crossing = ((side < 0.0) & (next_side > 0.0)) | ((side > 0.0) & (next_side < 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        parameter = along - side / (next_side - side) * (next_along - along)
    # Interpolate only between finite edge endpoints.  A tolerance-relaxed
    # halfplane intersection can otherwise invent a distant point where the
    # extensions of an almost-collinear triangle's edges meet.
    candidates = np.concatenate([np.where(crossing, parameter, np.nan), np.where(side == 0.0, along, np.nan)], axis=2)
    lower = np.min(np.where(np.isfinite(candidates), candidates, np.inf), axis=2)
    upper = np.max(np.where(np.isfinite(candidates), candidates, -np.inf), axis=2)
    if rays:
        lower = np.maximum(lower, 0.0)
    valid = np.isfinite(lower) & np.isfinite(upper) & (upper >= lower - tol)
    return [
        merge_intervals(np.column_stack([low[ok], np.maximum(low[ok], high[ok])]), tol=tol)
        for low, high, ok in zip(lower, upper, valid)
    ]


def polygon_line_intervals(
    polygon: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    rays: bool = True,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Even-odd fill of an ordered simple polygon on a line, plus unique hits.

    Collinear boundary segments and vertex tangencies are retained.  An odd
    crossing count is explicitly returned as invalid rather than silently
    treating an incomplete contour as a reliable filled polygon.
    """
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    direction = np.asarray(direction, dtype=float).reshape(2)
    if not len(poly):
        return np.empty((0, 2)), np.empty(0), True
    norm_squared = float(direction @ direction)
    if norm_squared <= tol * tol:
        raise ValueError("Line direction must be nonzero")
    relative = poly - np.asarray(origin, dtype=float).reshape(2)
    along = (relative @ direction) / norm_squared
    side = direction[0] * relative[:, 1] - direction[1] * relative[:, 0]
    side[np.abs(side) <= tol * np.sqrt(norm_squared)] = 0.0
    along_next, side_next = np.roll(along, -1), np.roll(side, -1)
    crossing = ((side <= 0.0) & (side_next > 0.0)) | ((side_next <= 0.0) & (side > 0.0))
    fractions = -side[crossing] / (side_next[crossing] - side[crossing])
    crossing_values = np.sort(along[crossing] + fractions * (along_next[crossing] - along[crossing]))
    invalid = bool(len(crossing_values) % 2)
    intervals = [] if invalid else crossing_values.reshape(-1, 2).tolist()
    collinear = (side == 0.0) & (side_next == 0.0)
    intervals.extend(np.column_stack([along[collinear], along_next[collinear]]).tolist())
    tangent_values = along[side == 0.0]
    intervals.extend(np.column_stack([tangent_values, tangent_values]).tolist())
    hits = np.sort(np.concatenate([crossing_values, tangent_values]))
    if hits.size:
        hits = hits[np.r_[True, np.diff(hits) > tol]]
    merged = merge_intervals(intervals, tol=tol)
    if rays:
        merged = merged[merged[:, 1] >= -tol]
        merged = np.maximum(merged, 0.0)
        hits = np.maximum(hits[hits >= -tol], 0.0)
    return merged, hits, invalid


def _ray_profile(intervals: list[np.ndarray], tol: float) -> dict[str, np.ndarray]:
    return {
        "outer": np.asarray([row[-1, 1] if len(row) else np.nan for row in intervals]),
        "count": np.asarray([len(row) for row in intervals], dtype=int),
        "origin_outside": np.asarray([not len(row) or row[0, 0] > tol for row in intervals]),
        "gap_length": np.asarray([float(np.sum(row[1:, 0] - row[:-1, 1])) if len(row) > 1 else 0.0 for row in intervals]),
    }


def _slice_triangle_segments(vertices: np.ndarray, triangles: np.ndarray, z: float, tol: float) -> np.ndarray:
    """Return finite triangle/plane intersections; coplanar triangles use edges."""
    tris = vertices[triangles]
    next_vertices = np.roll(tris, -1, axis=1)
    low, high = tris[:, :, 2] - z, next_vertices[:, :, 2] - z
    near_low, near_high = np.abs(low) <= tol, np.abs(high) <= tol
    crossing = ((low < -tol) & (high > tol)) | ((low > tol) & (high < -tol))
    segments = []
    active = np.any(near_low | near_high | crossing, axis=1)
    for tri, nxt, near_a, near_b, crosses in zip(tris[active], next_vertices[active], near_low[active], near_high[active], crossing[active]):
        if np.all(near_a):
            segments.extend(np.stack([tri, nxt], axis=1).tolist())
            continue
        points = [p for p, close in zip(tri, near_a) if close]
        for a, b, crosses_plane in zip(tri, nxt, crosses):
            if crosses_plane:
                points.append(a + (z - a[2]) / (b[2] - a[2]) * (b - a))
        if points:
            arr = np.asarray(points)
            axis = int(np.argmax(np.ptp(arr, axis=0)))
            segments.append([arr[np.argmin(arr[:, axis])].tolist(), arr[np.argmax(arr[:, axis])].tolist()])
    return np.asarray(segments, dtype=float).reshape(-1, 2, 3)


def _polyline_at_z(points: np.ndarray, z: float, *, closed: bool, tol: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    starts = pts if closed else pts[:-1]
    ends = np.roll(pts, -1, axis=0) if closed else pts[1:]
    delta_start, delta_end = starts[:, 2] - z, ends[:, 2] - z
    near_start, near_end = np.abs(delta_start) <= tol, np.abs(delta_end) <= tol
    crossing = ((delta_start < -tol) & (delta_end > tol)) | ((delta_start > tol) & (delta_end < -tol))
    crosses = starts[crossing] + ((z - starts[crossing, 2]) / (ends[crossing, 2] - starts[crossing, 2]))[:, None] * (ends[crossing] - starts[crossing])
    return np.vstack([starts[near_start], ends[near_end], crosses])


def _unique_sorted(values: np.ndarray, tol: float) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=float))
    return values[np.r_[True, np.diff(values) > tol]] if len(values) else values


def _geometry_diagnostics(vertices: np.ndarray, faces: list, tol: float) -> dict[str, object]:
    rows = []
    for face_index, face in enumerate(faces):
        points = vertices[np.asarray(face, dtype=int)]
        if len(points) < 3:
            rows.append({"face_index": face_index, "degenerate": True})
            continue
        _, singular, axes = np.linalg.svd(points - np.mean(points, axis=0), full_matrices=False)
        normal = axes[-1]
        distances = (vertices - np.mean(points, axis=0)) @ normal
        deviation = float(np.max(np.abs((points - np.mean(points, axis=0)) @ normal)))
        rows.append({
            "face_index": face_index,
            "degenerate": bool(len(singular) < 2 or singular[1] <= tol),
            "max_planarity_error": deviation,
            "other_vertices_min_signed": float(np.min(distances)),
            "other_vertices_max_signed": float(np.max(distances)),
            "plane_straddles_vertices": bool(np.min(distances) < -tol and np.max(distances) > tol),
        })
    return {
        "face_count": len(faces),
        "vertex_count": len(vertices),
        "support_plane_tolerance": tol,
        "non_supporting_face_count": sum(bool(row.get("plane_straddles_vertices")) for row in rows),
        "degenerate_face_count": sum(bool(row["degenerate"]) for row in rows),
        "max_face_planarity_error": max((row.get("max_planarity_error", 0.0) for row in rows), default=0.0),
        "per_face": rows,
        "interpretation": "A plane straddling vertices cannot be a global containing halfspace; this check alone does not validate mesh topology.",
    }


def _calibration(contour: object, vertices: np.ndarray, observed: np.ndarray, projected: np.ndarray, u: np.ndarray, v: np.ndarray) -> dict[str, object]:
    normal = np.asarray(contour.normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    unit = normal / norm
    angle = float(contour.angle)
    clockwise_angle = float(np.arctan2(-unit[1], unit[0]))
    angle_error = float(np.arctan2(np.sin(clockwise_angle - angle), np.cos(clockwise_angle - angle)))
    alternate_error = float(np.arctan2(np.sin(-clockwise_angle - angle), np.cos(-clockwise_angle - angle)))
    plane_values = np.asarray(contour.points) @ unit
    model_span = np.ptp(projected, axis=0)
    observed_span = np.ptp(observed, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_ratio = model_span / observed_span
    return {
        "normal": normal.tolist(),
        "normal_norm_error": norm - 1.0,
        "normal_abs_z": abs(float(unit[2])),
        "lateral_axis": u.tolist(),
        "vertical_axis": v.tolist(),
        "basis_orthogonality_error": float(max(abs(u @ unit), abs(v @ unit), abs(u @ v))),
        "clockwise_angle_error_radians": _number(angle_error),
        "counterclockwise_angle_error_radians": _number(alternate_error),
        "observed_plane_rms": float(np.std(plane_values)),
        "observed_plane_offset": float(np.mean(plane_values)),
        "model_to_observed_span_ratio_uv": _array(scale_ratio),
        "model_minus_observed_bbox_midpoint_uv": ((np.min(projected, axis=0) + np.max(projected, axis=0) - np.min(observed, axis=0) - np.max(observed, axis=0)) / 2.0).tolist(),
        "model_z_range": [float(np.min(vertices[:, 2])), float(np.max(vertices[:, 2]))],
        "observed_z_range": [float(np.min(np.asarray(contour.points)[:, 2])), float(np.max(np.asarray(contour.points)[:, 2]))],
    }


def _horizontal_probes(contour: object, halves: list, z_levels: np.ndarray, slice_segments: list[np.ndarray], u: np.ndarray, tol: float) -> tuple[list[dict], dict[str, object]]:
    probes = []
    half_residuals, left_residuals, right_residuals, widths, centers = [], [], [], [], []
    for z_index, (z, segments) in enumerate(zip(z_levels, slice_segments)):
        intersections_3d = _polyline_at_z(contour.points, float(z), closed=True, tol=tol)
        observed_hits = _unique_sorted(intersections_3d @ u, tol)
        model_intervals = merge_intervals(segments @ u, tol=tol)
        observed_min = float(observed_hits[0]) if len(observed_hits) else np.nan
        observed_max = float(observed_hits[-1]) if len(observed_hits) else np.nan
        model_min = float(model_intervals[0, 0]) if len(model_intervals) else np.nan
        model_max = float(model_intervals[-1, 1]) if len(model_intervals) else np.nan
        half_rows = []
        for half in halves:
            hits_3d = _polyline_at_z(half.points, float(z), closed=False, tol=tol)
            hits = _unique_sorted(hits_3d @ u, tol)
            # Reproduce the existing point choice, which uses distance to the
            # 3D branch centroid, and compare it to the true full-contour extreme.
            if len(hits_3d):
                center = np.mean(half.points, axis=0)
                selected = hits_3d[np.argmin(np.linalg.norm(hits_3d - center, axis=1))]
                selected_u = float(selected @ u)
            else:
                selected_u = np.nan
            target = observed_min if int(half.half_id) == 0 else observed_max
            residual = selected_u - target
            half_residuals.append(residual)
            half_rows.append({
                "half_id": int(half.half_id),
                "intersection_count": len(hits),
                "selected_u": _number(selected_u),
                "expected_extreme_u": _number(target),
                "selected_minus_extreme": _number(residual),
                "missing_when_full_contour_present": bool(np.isfinite(target) and not np.isfinite(selected_u)),
                "mismatch": bool(np.isfinite(residual) and abs(residual) > tol),
            })
        left, right = model_min - observed_min, model_max - observed_max
        width = (model_max - model_min) - (observed_max - observed_min)
        center = ((model_max + model_min) - (observed_max + observed_min)) / 2.0
        left_residuals.append(left)
        right_residuals.append(right)
        widths.append(width)
        centers.append(center)
        probes.append({
            "z_index": z_index, "z": float(z),
            "observed_min": _number(observed_min), "observed_max": _number(observed_max),
            "observed_intersection_count": len(observed_hits),
            "observed_intersections": observed_hits.tolist() if len(observed_hits) > 2 else None,
            "model_min": _number(model_min), "model_max": _number(model_max),
            "model_interval_count": len(model_intervals),
            "model_intervals": model_intervals.tolist() if len(model_intervals) > 1 else None,
            "signed_left": _number(left), "signed_right": _number(right),
            "width_difference": _number(width), "midpoint_shift": _number(center),
            "halves": half_rows,
        })
    summary = {
        "probe_count": len(probes),
        "left_residual": _summary(left_residuals), "right_residual": _summary(right_residuals),
        "width_difference": _summary(widths), "midpoint_shift": _summary(centers),
        "half_selected_minus_extreme": _summary(half_residuals),
        "observed_multi_intersection_count": sum(row["observed_intersection_count"] > 2 for row in probes),
        "model_disjoint_interval_count": sum(row["model_interval_count"] > 1 for row in probes),
        "missing_observed_count": sum(row["observed_intersection_count"] == 0 for row in probes),
        "missing_model_count": sum(row["model_interval_count"] == 0 for row in probes),
        "half_branch_mismatch_count": sum(half["mismatch"] for row in probes for half in row["halves"]),
        "half_branch_missing_count": sum(half["missing_when_full_contour_present"] for row in probes for half in row["halves"]),
        "half_branch_multi_intersection_count": sum(half["intersection_count"] > 1 for row in probes for half in row["halves"]),
    }
    return probes, summary


def build_projection_diagnostics(
    *,
    vertices: np.ndarray,
    faces: list,
    triangles: np.ndarray,
    contours: list,
    z_levels: np.ndarray,
    half_contours: list,
    polar_origin: np.ndarray,
    ray_count: int = 360,
    example_z_values: tuple = (-6.0, -5.0, -2.0),
) -> dict[str, object]:
    """Build JSON-safe input diagnostics without fitting or mutating geometry."""
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=int).reshape(-1, 3)
    levels = np.asarray(z_levels, dtype=float).reshape(-1)
    origin = np.asarray(polar_origin, dtype=float).reshape(3)
    if not len(vertices) or not len(triangles) or not np.all(np.isfinite(vertices)):
        raise ValueError("Finite vertices and a nonempty finite-face triangulation are required")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("Triangle vertex index out of bounds")
    scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1.0)
    tol = max(scale * 1e-10, 1e-10)
    directions = reprojection_directions(ray_count)
    slices = [_slice_triangle_segments(vertices, triangles, float(z), tol) for z in levels]
    halves_by_index = {}
    for half in half_contours:
        halves_by_index.setdefault(int(half.source_index), []).append(half)
    views = []
    all_observed, all_finite, all_hull = [], [], []
    total_ray_counts = {name: 0 for name in ("finite_missing", "observed_missing", "finite_disjoint", "observed_disjoint", "finite_origin_outside", "observed_origin_outside", "observed_invalid_crossings")}
    for contour in contours:
        u, v = contour_basis(np.asarray(contour.normal, dtype=float))
        observed = np.column_stack([np.asarray(contour.points) @ u, np.asarray(contour.points) @ v])
        projected = np.column_stack([vertices @ u, vertices @ v])
        origin_2d = np.asarray([origin @ u, origin @ v])
        hull = convex_hull_2d(projected)
        finite_intervals = triangle_union_line_intervals(projected[triangles], origin_2d, directions, tol=tol)
        observed_results = [polygon_line_intervals(observed, origin_2d, direction, tol=tol) for direction in directions]
        observed_intervals = [row[0] for row in observed_results]
        hull_intervals = [polygon_line_intervals(hull, origin_2d, direction, tol=tol)[0] for direction in directions]
        finite = _ray_profile(finite_intervals, tol)
        obs = _ray_profile(observed_intervals, tol)
        hull_profile = _ray_profile(hull_intervals, tol)
        all_observed.append(obs["outer"])
        all_finite.append(finite["outer"])
        all_hull.append(hull_profile["outer"])
        counts = {
            "finite_missing": int(np.sum(finite["count"] == 0)),
            "observed_missing": int(np.sum(obs["count"] == 0)),
            "finite_disjoint": int(np.sum(finite["count"] > 1)),
            "observed_disjoint": int(np.sum(obs["count"] > 1)),
            "finite_origin_outside": int(np.sum(finite["origin_outside"])),
            "observed_origin_outside": int(np.sum(obs["origin_outside"])),
            "observed_invalid_crossings": sum(row[2] for row in observed_results),
        }
        for key, value in counts.items():
            total_ray_counts[key] += value
        calibration = _calibration(contour, vertices, observed, projected, u, v)
        horizontal_valid = calibration["normal_abs_z"] <= 1e-8
        if horizontal_valid:
            probes, horizontal_summary = _horizontal_probes(contour, halves_by_index.get(int(contour.index), []), levels, slices, u, tol)
        else:
            probes, horizontal_summary = [], {"probe_count": 0, "reason": "tilted_projection_normal_global_Z_is_not_image_vertical"}
        projected_triangles = projected[triangles]
        sides = projected_triangles[:, 1:] - projected_triangles[:, :1]
        areas = sides[:, 0, 0] * sides[:, 1, 1] - sides[:, 0, 1] * sides[:, 1, 0]
        views.append({
            "contour_index": int(contour.index), "angle": float(contour.angle),
            "finite_faces": polar_radial_error_summary(obs["outer"], finite["outer"]),
            "convex_hull": polar_radial_error_summary(obs["outer"], hull_profile["outer"]),
            "hull_minus_finite": polar_radial_error_summary(finite["outer"], hull_profile["outer"]),
            "ray_union": counts,
            "projected_degenerate_triangle_count": int(np.sum(np.abs(areas) <= tol * np.max(np.linalg.norm(np.roll(projected_triangles, -1, axis=1) - projected_triangles, axis=2), axis=1))),
            "calibration": calibration,
            "horizontal_valid": horizontal_valid, "horizontal_summary": horizontal_summary,
            "polar_origin_2d": origin_2d.tolist(),
            "observed_polygon_2d": observed.tolist(), "hull_polygon_2d": hull.tolist(),
            "observed_outer_radii": _array(obs["outer"]),
            "finite_outer_radii": _array(finite["outer"]), "hull_outer_radii": _array(hull_profile["outer"]),
            "finite_ray_interval_counts": finite["count"].tolist(),
            "observed_ray_interval_counts": obs["count"].tolist(),
            "ray_gap_details": [{"ray_index": index, "finite_intervals": finite_intervals[index].tolist(), "observed_intervals": observed_intervals[index].tolist()} for index in range(len(directions)) if finite["count"][index] > 1 or obs["count"][index] > 1 or finite["origin_outside"][index] or obs["origin_outside"][index]],
            "horizontal_probes": probes,
        })
    observed_all = np.concatenate(all_observed) if all_observed else np.empty(0)
    finite_all = np.concatenate(all_finite) if all_finite else np.empty(0)
    hull_all = np.concatenate(all_hull) if all_hull else np.empty(0)
    probes_all = [probe for view in views for probe in view["horizontal_probes"]]
    horizontal_summary = {name: sum(view["horizontal_summary"].get(name, 0) for view in views) for name in ("probe_count", "observed_multi_intersection_count", "model_disjoint_interval_count", "missing_observed_count", "missing_model_count", "half_branch_mismatch_count", "half_branch_missing_count", "half_branch_multi_intersection_count")}
    for source, name in (("signed_left", "left_residual"), ("signed_right", "right_residual"), ("width_difference", "width_difference"), ("midpoint_shift", "midpoint_shift")):
        horizontal_summary[name] = _summary([row[source] for row in probes_all])
    horizontal_summary["half_selected_minus_extreme"] = _summary([half["selected_minus_extreme"] for row in probes_all for half in row["halves"]])
    horizontal_summary["per_z"] = [{
        "z_index": index, "z": float(z),
        "left_residual": _summary([view["horizontal_probes"][index]["signed_left"] for view in views if view["horizontal_valid"]]),
        "right_residual": _summary([view["horizontal_probes"][index]["signed_right"] for view in views if view["horizontal_valid"]]),
        "half_branch_mismatch_count": sum(half["mismatch"] for view in views if view["horizontal_valid"] for half in view["horizontal_probes"][index]["halves"]),
    } for index, z in enumerate(levels)]
    calibration_summary = {key: _summary([view["calibration"][key] for view in views]) for key in ("normal_norm_error", "normal_abs_z", "clockwise_angle_error_radians", "counterclockwise_angle_error_radians", "observed_plane_rms")}
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "method": "finite_triangle_projection_union_intervals_on_common_polar_rays_and_global_Z_slices",
        "sign_convention": "model_minus_observed; positive right expands, negative left expands",
        "angle_convention": "Input files use clockwise angle: normal=(cos(angle), -sin(angle), 0); projection always uses the actual normal.",
        "registration": "identity; no per-view or per-Z alignment, scale correction, or parameter fitting",
        "limitations": [
            "Outer radial error does not fully measure holes, disconnected components or all concave boundary segments; interval gaps and missing rays are reported separately.",
            "Triangle unions are exact only along sampled lines. Connecting outer endpoints is not an exact silhouette polygon.",
            "Observed ordered contours are treated as simple closed polylines with even-odd fill; self-intersections are not repaired.",
            "Projected zero-area triangles are omitted from area-union rays; valid closed meshes retain those boundaries through adjacent nondegenerate triangles.",
            "No input calibration correction is inferred from small residuals or symmetric silhouettes; signed span and midpoint diagnostics are descriptive.",
        ],
        "numeric_tolerance": tol,
        "polar_origin_3d": origin.tolist(), "view_count": len(views),
        "ray_count_per_view": len(directions),
        "ray_angles_degrees": np.rad2deg(np.arctan2(directions[:, 1], directions[:, 0]) % (2.0 * np.pi)).tolist(),
        "z_levels": levels.tolist(),
        "example_z_values": [{"requested_z": float(z), "z_index": int(np.argmin(np.abs(levels - z))), "actual_z": float(levels[np.argmin(np.abs(levels - z))])} for z in example_z_values] if len(levels) else [],
        "geometry": _geometry_diagnostics(vertices, faces, max(scale * 1e-8, 1e-9)),
        "summary": {
            "finite_faces": polar_radial_error_summary(observed_all, finite_all),
            "convex_hull": polar_radial_error_summary(observed_all, hull_all),
            "hull_minus_finite": polar_radial_error_summary(finite_all, hull_all),
            "ray_union": total_ray_counts,
            "horizontal": horizontal_summary,
            "calibration": calibration_summary,
        },
        "per_view": views,
    }
