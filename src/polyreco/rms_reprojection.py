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
    "evaluate_polyhedron_reprojection",
    "point_to_face_polygon_distance",
    "point_to_polygon_distance_2d",
    "project_points_to_contour_2d",
    "reprojection_directions",
    "surface_distance_summary",
]
