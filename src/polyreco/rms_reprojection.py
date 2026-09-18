from __future__ import annotations

import time

import numpy as np

from polyreco.rms_mesh import plane_basis
from polyreco.rms_selection import (
    as_json_float,
    finite_float,
    polygon_area_2d,
    polygon_signed_distances_2d,
)


EPS = 1e-9


def lateral_axis_for_normal(direction: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    lateral = np.cross(z_axis, direction)
    ln = np.linalg.norm(lateral)
    if ln < EPS:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        lateral = np.cross(x_axis, direction)
        ln = np.linalg.norm(lateral)
        if ln < EPS:
            raise ValueError("Cannot build lateral axis for projection direction")
    return lateral / ln


def distribution_summary(values: list[object]) -> dict[str, object]:
    arr = np.array([finite_float(v, float("nan")) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "p10": None, "median": None, "p90": None}
    p10, median, p90 = np.percentile(arr, [10, 50, 90])
    return {
        "count": int(arr.size),
        "p10": as_json_float(float(p10)),
        "median": as_json_float(float(median)),
        "p90": as_json_float(float(p90)),
    }


def point_to_face_polygon_distance(point: np.ndarray, face: dict[str, object]) -> float:
    fn = np.array(face["normal"], dtype=float)
    face_pts = np.array(face["vertices"], dtype=float)
    cache = face.get("_polygon_distance_cache")
    if isinstance(cache, dict):
        origin = np.array(cache["origin"], dtype=float)
        u = np.array(cache["u"], dtype=float)
        v = np.array(cache["v"], dtype=float)
        poly_2d = np.array(cache["poly_2d"], dtype=float)
    else:
        u, v = plane_basis(fn)
        origin = np.mean(face_pts, axis=0)
        poly_2d = np.column_stack([(face_pts - origin) @ u, (face_pts - origin) @ v])
        face["_polygon_distance_cache"] = {
            "origin": origin,
            "u": u,
            "v": v,
            "poly_2d": poly_2d,
        }
    point_2d = np.array([[(point - origin) @ u, (point - origin) @ v]], dtype=float)
    outside = float(max(0.0, polygon_signed_distances_2d(point_2d, poly_2d)[0]))
    plane_distance = abs(float((point - face_pts[0]) @ fn))
    return float(np.sqrt(plane_distance * plane_distance + outside * outside))


def surface_distance_summary(points: np.ndarray, faces: list[dict[str, object]], *, max_points: int = 32) -> dict[str, object]:
    if points.size == 0 or not faces:
        return {"median": None, "p95": None, "point_count": 0, "sampled_point_count": 0}
    sample_points = np.asarray(points, dtype=float)
    if sample_points.shape[0] > int(max_points):
        idx = np.linspace(0, sample_points.shape[0] - 1, int(max_points), dtype=int)
        sample_points = sample_points[idx]
    vals: list[float] = []
    for p in sample_points:
        vals.append(min(point_to_face_polygon_distance(np.array(p, dtype=float), face) for face in faces))
    arr = np.array(vals, dtype=float)
    return {
        "median": as_json_float(float(np.median(arr))),
        "p95": as_json_float(float(np.percentile(arr, 95))),
        "point_count": int(len(points)),
        "sampled_point_count": int(len(sample_points)),
    }


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] <= 1:
        return pts.copy()
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    unique = [pts[0]]
    for p in pts[1:]:
        if float(np.linalg.norm(p - unique[-1])) > 1e-10:
            unique.append(p)
    pts = np.array(unique, dtype=float)
    if pts.shape[0] <= 2:
        return pts

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def point_to_polygon_distance_2d(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    polygon = np.asarray(poly, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or polygon.ndim != 2 or polygon.shape[0] == 0:
        return np.full(pts.shape[0] if pts.ndim == 2 else 0, float("inf"), dtype=float)
    if polygon.shape[0] == 1:
        return np.linalg.norm(pts - polygon[0][None, :], axis=1)
    if polygon.shape[0] == 2:
        a, b = polygon
        ab = b - a
        denom = max(float(ab @ ab), EPS)
        t = np.clip(((pts - a[None, :]) @ ab) / denom, 0.0, 1.0)
        return np.linalg.norm(pts - (a[None, :] + t[:, None] * ab[None, :]), axis=1)
    return np.maximum(polygon_signed_distances_2d(pts, polygon), 0.0)


def reprojection_directions(count: int) -> np.ndarray:
    n = max(8, int(count))
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=float)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def ray_polygon_outer_radii(
    polygon: np.ndarray,
    origin: np.ndarray,
    directions: np.ndarray,
    *,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the farthest non-negative intersection for every 2D ray.

    The polygon is treated as an ordered closed polyline.  Taking the farthest
    intersection keeps the metric meaningful if an observed contour contains a
    small non-convex sampling artefact or a ray crosses the boundary more than
    once.
    """

    poly = np.asarray(polygon, dtype=float)
    center = np.asarray(origin, dtype=float).reshape(2)
    dirs = np.asarray(directions, dtype=float)
    if (
        poly.ndim != 2
        or poly.shape[1] != 2
        or dirs.ndim != 2
        or dirs.shape[1] != 2
    ):
        raise ValueError("Expected polygon Nx2, origin 2D and directions Mx2")
    poly = poly[np.all(np.isfinite(poly), axis=1)]
    if poly.shape[0] > 1 and float(np.linalg.norm(poly[0] - poly[-1])) <= tol:
        poly = poly[:-1]
    if poly.shape[0] < 3:
        return np.full(dirs.shape[0], float("nan"), dtype=float), np.zeros(dirs.shape[0], dtype=int)

    starts = poly
    edges = np.roll(poly, -1, axis=0) - starts
    rel = starts - center[None, :]

    # O + t*d = A + u*edge.  In 2D:
    # t = cross(A-O, edge) / cross(d, edge)
    # u = cross(A-O, d)    / cross(d, edge)
    denominator = (
        dirs[:, 0, None] * edges[None, :, 1]
        - dirs[:, 1, None] * edges[None, :, 0]
    )
    t_numerator = rel[:, 0] * edges[:, 1] - rel[:, 1] * edges[:, 0]
    u_numerator = (
        rel[None, :, 0] * dirs[:, None, 1]
        - rel[None, :, 1] * dirs[:, None, 0]
    )
    non_parallel = np.abs(denominator) > tol
    with np.errstate(divide="ignore", invalid="ignore"):
        ray_t = t_numerator[None, :] / denominator
        edge_u = u_numerator / denominator
    valid = (
        non_parallel
        & np.isfinite(ray_t)
        & np.isfinite(edge_u)
        & (ray_t >= -tol)
        & (edge_u >= -tol)
        & (edge_u <= 1.0 + tol)
    )
    hit_counts = np.sum(valid, axis=1, dtype=int)
    candidates = np.where(valid, np.maximum(ray_t, 0.0), -np.inf)
    radii = np.max(candidates, axis=1)
    radii[hit_counts == 0] = float("nan")
    return radii, hit_counts


def polar_radial_error_summary(observed_radii: np.ndarray, model_radii: np.ndarray) -> dict[str, object]:
    observed = np.asarray(observed_radii, dtype=float)
    model = np.asarray(model_radii, dtype=float)
    if observed.shape != model.shape:
        raise ValueError("Observed and model radial profiles must have the same shape")
    valid = np.isfinite(observed) & np.isfinite(model)
    missing_observed = int(np.sum(~np.isfinite(observed)))
    missing_model = int(np.sum(np.isfinite(observed) & ~np.isfinite(model)))
    if not np.any(valid):
        return {
            "sample_count": 0,
            "missing_observed_rays": missing_observed,
            "missing_model_rays": missing_model,
        }

    observed = observed[valid]
    signed = model[valid] - observed
    absolute = np.abs(signed)
    squared = signed * signed
    outside = signed[signed > 0.0]
    inside = -signed[signed < 0.0]
    observed_energy = float(np.sum(observed * observed))
    squared_error_sum = float(np.sum(squared))
    relative_squared_error = squared_error_sum / max(observed_energy, EPS)
    return {
        "sample_count": int(signed.size),
        "missing_observed_rays": missing_observed,
        "missing_model_rays": missing_model,
        "squared_error_sum": as_json_float(squared_error_sum),
        "mean_squared_error": as_json_float(float(np.mean(squared))),
        "root_mean_squared_error": as_json_float(float(np.sqrt(np.mean(squared)))),
        "mean_absolute_error": as_json_float(float(np.mean(absolute))),
        "median_absolute_error": as_json_float(float(np.median(absolute))),
        "p90_absolute_error": as_json_float(float(np.percentile(absolute, 90))),
        "p95_absolute_error": as_json_float(float(np.percentile(absolute, 95))),
        "p99_absolute_error": as_json_float(float(np.percentile(absolute, 99))),
        "max_absolute_error": as_json_float(float(np.max(absolute))),
        "mean_signed_error": as_json_float(float(np.mean(signed))),
        "median_signed_error": as_json_float(float(np.median(signed))),
        "model_outside_ray_fraction": as_json_float(float(np.mean(signed > 0.0))),
        "model_inside_ray_fraction": as_json_float(float(np.mean(signed < 0.0))),
        "model_outside_mean_error": (
            as_json_float(float(np.mean(outside))) if outside.size else None
        ),
        "model_outside_root_mean_squared_error": (
            as_json_float(float(np.sqrt(np.mean(outside * outside)))) if outside.size else None
        ),
        "model_inside_mean_error": (
            as_json_float(float(np.mean(inside))) if inside.size else None
        ),
        "model_inside_root_mean_squared_error": (
            as_json_float(float(np.sqrt(np.mean(inside * inside)))) if inside.size else None
        ),
        "observed_radius_mean": as_json_float(float(np.mean(observed))),
        "observed_radius_rms": as_json_float(float(np.sqrt(np.mean(observed * observed)))),
        "relative_squared_error": as_json_float(relative_squared_error),
        "relative_root_mean_squared_error": as_json_float(float(np.sqrt(relative_squared_error))),
    }


def evaluate_polyhedron_polar_reprojection(
    *,
    name: str,
    vertices_3d: np.ndarray,
    contours: list[object],
    polar_origin_3d: np.ndarray,
    ray_count: int,
    contour_sample: int | None = None,
) -> dict[str, object]:
    """Compare observed and projected silhouettes on a common polar ray grid."""

    started = time.perf_counter()
    vertices = np.asarray(vertices_3d, dtype=float)
    origin_3d = np.asarray(polar_origin_3d, dtype=float).reshape(3)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
        return {"name": name, "valid": False, "reason": "empty_vertices"}

    selected = list(contours)
    if (
        contour_sample is not None
        and int(contour_sample) > 0
        and len(selected) > int(contour_sample)
    ):
        idx = np.linspace(0, len(selected) - 1, int(contour_sample), dtype=int)
        selected = [selected[int(i)] for i in idx]
    directions = reprojection_directions(ray_count)
    rows: list[dict[str, object]] = []
    all_observed: list[np.ndarray] = []
    all_model: list[np.ndarray] = []

    for contour in selected:
        normal = np.asarray(contour.normal, dtype=float)
        observed_polygon = project_points_to_contour_2d(
            np.asarray(contour.points, dtype=float),
            normal,
        )
        projected_vertices = project_points_to_contour_2d(vertices, normal)
        model_polygon = convex_hull_2d(projected_vertices)
        origin_2d = project_points_to_contour_2d(origin_3d[None, :], normal)[0]
        observed_radii, observed_hits = ray_polygon_outer_radii(
            observed_polygon,
            origin_2d,
            directions,
        )
        model_radii, model_hits = ray_polygon_outer_radii(model_polygon, origin_2d, directions)
        summary = polar_radial_error_summary(observed_radii, model_radii)
        valid = np.isfinite(observed_radii) & np.isfinite(model_radii)
        if np.any(valid):
            all_observed.append(observed_radii[valid])
            all_model.append(model_radii[valid])
        rows.append(
            {
                "contour_index": int(contour.index),
                "contour_angle": as_json_float(float(contour.angle)),
                "polar_origin_2d": [as_json_float(float(v)) for v in origin_2d],
                "observed_polygon_points": int(observed_polygon.shape[0]),
                "model_hull_vertices": int(model_polygon.shape[0]),
                "observed_multi_hit_ray_count": int(np.sum(observed_hits > 1)),
                "model_multi_hit_ray_count": int(np.sum(model_hits > 1)),
                **summary,
            }
        )

    if not all_observed:
        return {
            "name": name,
            "valid": False,
            "reason": "no_common_ray_intersections",
            "view_count": int(len(rows)),
        }

    observed_all = np.concatenate(all_observed)
    model_all = np.concatenate(all_model)
    aggregate = polar_radial_error_summary(observed_all, model_all)
    angular_step = 2.0 * np.pi / max(8, int(ray_count))
    squared_error_sum = finite_float(aggregate.get("squared_error_sum"), 0.0)
    rows.sort(key=lambda row: finite_float(row.get("root_mean_squared_error"), -1.0), reverse=True)
    return {
        "name": name,
        "valid": True,
        "metric": "common_origin_outer_ray_radial_difference",
        "sign_convention": "model_radius_minus_observed_radius",
        "positive_error_meaning": "projected model extends outside the observed contour",
        "observed_contour_mode": "ordered_polyline_farthest_ray_intersection",
        "model_contour_mode": "convex_hull_of_projected_polyhedron_vertices",
        "polar_origin_3d": [as_json_float(float(v)) for v in origin_3d],
        "view_count": int(len(rows)),
        "ray_count_per_view": int(max(8, int(ray_count))),
        "requested_contour_sample": int(len(selected)),
        "angular_step_radians": as_json_float(angular_step),
        "angular_integrated_squared_error_sum": as_json_float(angular_step * squared_error_sum),
        **aggregate,
        "missing_observed_rays": int(sum(int(row.get("missing_observed_rays", 0)) for row in rows)),
        "missing_model_rays": int(sum(int(row.get("missing_model_rays", 0)) for row in rows)),
        "worst_views": rows[:12],
        "per_view": sorted(rows, key=lambda row: int(row["contour_index"])),
        "timing_seconds": as_json_float(time.perf_counter() - started),
    }


def contour_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = lateral_axis_for_normal(normal)
    v = np.cross(normal, u)
    vn = float(np.linalg.norm(v))
    if vn <= EPS:
        raise ValueError("Cannot build contour basis")
    return u, v / vn


def project_points_to_contour_2d(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    u, v = contour_basis(normal)
    pts = np.asarray(points, dtype=float)
    return np.column_stack([pts @ u, pts @ v])


def evaluate_polyhedron_reprojection(
    *,
    name: str,
    vertices_3d: np.ndarray,
    contours: list[object],
    direction_count: int,
    contour_sample: int | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    verts = np.asarray(vertices_3d, dtype=float)
    if verts.ndim != 2 or verts.shape[0] == 0 or verts.shape[1] != 3:
        return {"name": name, "valid": False, "reason": "empty_vertices"}
    selected = list(contours)
    if contour_sample is not None and int(contour_sample) > 0 and len(selected) > int(contour_sample):
        idx = np.linspace(0, len(selected) - 1, int(contour_sample), dtype=int)
        selected = [selected[int(i)] for i in idx]
    directions = reprojection_directions(direction_count)
    rows: list[dict[str, object]] = []
    all_abs_support: list[float] = []
    all_pos_support: list[float] = []
    all_neg_support: list[float] = []
    all_dist: list[float] = []
    ratios: list[float] = []
    for contour in selected:
        normal = np.array(contour.normal, dtype=float)
        obs2 = project_points_to_contour_2d(np.asarray(contour.points, dtype=float), normal)
        model2 = project_points_to_contour_2d(verts, normal)
        hull = convex_hull_2d(model2)
        if hull.shape[0] < 3 or obs2.shape[0] < 3:
            continue
        h_model = np.max(hull @ directions.T, axis=0)
        h_obs = np.max(obs2 @ directions.T, axis=0)
        signed = h_model - h_obs
        abs_signed = np.abs(signed)
        pos = np.maximum(signed, 0.0)
        neg = np.maximum(-signed, 0.0)
        obs_hull = convex_hull_2d(obs2)
        obs_to_model = point_to_polygon_distance_2d(obs2, hull)
        model_to_obs = point_to_polygon_distance_2d(hull, obs_hull)
        sym = np.concatenate([obs_to_model, model_to_obs]) if obs_to_model.size or model_to_obs.size else np.zeros(0)
        model_area = polygon_area_2d(hull)
        obs_area = polygon_area_2d(obs_hull)
        ratio = model_area / max(obs_area, EPS)
        all_abs_support.extend(float(v) for v in abs_signed if np.isfinite(v))
        all_pos_support.extend(float(v) for v in pos if np.isfinite(v))
        all_neg_support.extend(float(v) for v in neg if np.isfinite(v))
        all_dist.extend(float(v) for v in sym if np.isfinite(v))
        ratios.append(float(ratio))
        rows.append(
            {
                "contour_index": int(contour.index),
                "support_abs_median": as_json_float(float(np.median(abs_signed))),
                "support_abs_p95": as_json_float(float(np.percentile(abs_signed, 95))),
                "support_abs_max": as_json_float(float(np.max(abs_signed))),
                "model_outside_observed_median": as_json_float(float(np.median(pos))),
                "model_inside_observed_median": as_json_float(float(np.median(neg))),
                "symmetric_contour_distance_median": as_json_float(float(np.median(sym))) if sym.size else None,
                "symmetric_contour_distance_p95": as_json_float(float(np.percentile(sym, 95))) if sym.size else None,
                "polygon_area_ratio": as_json_float(float(ratio)),
                "model_hull_vertices": int(hull.shape[0]),
                "observed_points": int(obs2.shape[0]),
            }
        )
    rows.sort(key=lambda row: finite_float(row.get("support_abs_p95"), -1.0), reverse=True)
    return {
        "name": name,
        "valid": True,
        "view_count": int(len(rows)),
        "direction_count": int(direction_count),
        "contour_sample": int(len(selected)),
        "support_abs_median": distribution_summary(all_abs_support).get("median"),
        "support_abs_p95": distribution_summary(all_abs_support).get("p90"),
        "support_abs_max": as_json_float(float(max(all_abs_support))) if all_abs_support else None,
        "model_outside_observed_median": distribution_summary(all_pos_support).get("median"),
        "model_inside_observed_median": distribution_summary(all_neg_support).get("median"),
        "symmetric_contour_distance_median": distribution_summary(all_dist).get("median"),
        "symmetric_contour_distance_p95": distribution_summary(all_dist).get("p90"),
        "polygon_area_ratio_median": distribution_summary(ratios).get("median"),
        "polygon_area_ratio_p10": distribution_summary(ratios).get("p10"),
        "polygon_area_ratio_p90": distribution_summary(ratios).get("p90"),
        "worst_views": rows[:12],
        "timing_seconds": as_json_float(time.perf_counter() - started),
    }


__all__ = [
    "contour_basis",
    "convex_hull_2d",
    "evaluate_polyhedron_polar_reprojection",
    "evaluate_polyhedron_reprojection",
    "polar_radial_error_summary",
    "point_to_face_polygon_distance",
    "point_to_polygon_distance_2d",
    "project_points_to_contour_2d",
    "ray_polygon_outer_radii",
    "reprojection_directions",
    "surface_distance_summary",
]
