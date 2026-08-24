from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyreco.rms_mesh import convex_hull_indices, plane_basis, polygon_area
from polyreco.rms_selection import as_json_float, as_json_point, finite_float
from polyreco.rms_w2 import (
    is_overlapping_collinear_duplicate_pair,
    predict_w2_line,
)


EPS = 1e-9


__all__ = [
    "CyclicRegion",
    "PeakObservation",
    "PeakTrack",
    "PlaneFit",
    "aggregate_valley_quality",
    "best_low_region_points",
    "build_face_candidates",
    "build_w2_face_candidates",
    "cyclic_distance",
    "fit_plane",
    "orient_plane_from_inside_point",
    "orient_plane_from_observed_points",
    "region_point_rows",
    "region_to_json",
    "robust_jump_smoothness",
    "track_observation_debug",
    "track_peak_observations",
    "track_points",
]


@dataclass(frozen=True)
class CyclicRegion:
    indices: tuple[int, ...]
    start: int
    end: int
    center: float
    width: int
    min_value: float
    max_value: float
    mean_value: float


@dataclass
class PeakObservation:
    z_index: int
    z_value: float
    peak: CyclicRegion
    left_low: CyclicRegion | None
    right_low: CyclicRegion | None
    left_low_distance: float | None
    right_low_distance: float | None
    left_low_source: str = "threshold"
    right_low_source: str = "threshold"
    left_crossed_soft_barrier: bool = False
    right_crossed_soft_barrier: bool = False
    left_valley_quality: dict[str, float | bool | None] | None = None
    right_valley_quality: dict[str, float | bool | None] | None = None


@dataclass
class PeakTrack:
    track_id: int
    observations: list[PeakObservation]


@dataclass(frozen=True)
class PlaneFit:
    centroid: np.ndarray
    normal: np.ndarray
    rms: float
    max_abs: float


def region_to_json(region: CyclicRegion | None) -> dict[str, object] | None:
    if region is None:
        return None
    return {
        "indices": [int(i) for i in region.indices],
        "start": int(region.start),
        "end": int(region.end),
        "center": as_json_float(float(region.center)),
        "width": int(region.width),
        "min_value": as_json_float(float(region.min_value)),
        "max_value": as_json_float(float(region.max_value)),
        "mean_value": as_json_float(float(region.mean_value)),
    }


def cyclic_distance(a: float, b: float, n: int) -> float:
    d = abs(float(a) - float(b)) % float(n)
    return float(min(d, float(n) - d))


def track_peak_observations(
    per_level: list[list[PeakObservation]],
    *,
    n_half: int,
    max_center_jump: float,
    max_z_gap: int,
) -> list[PeakTrack]:
    tracks: list[PeakTrack] = []
    active: list[PeakTrack] = []

    for observations in per_level:
        candidates: list[tuple[float, int, int]] = []
        for ti, track in enumerate(active):
            prev = track.observations[-1]
            gap = observations[0].z_index - prev.z_index if observations else 1
            if gap > max_z_gap:
                continue
            for oi, obs in enumerate(observations):
                dist = cyclic_distance(prev.peak.center, obs.peak.center, n_half)
                width_penalty = abs(prev.peak.width - obs.peak.width) * 0.15
                if dist <= max_center_jump:
                    candidates.append((dist + width_penalty, ti, oi))

        candidates.sort(key=lambda x: x[0])
        used_tracks: set[int] = set()
        used_obs: set[int] = set()
        for _, ti, oi in candidates:
            if ti in used_tracks or oi in used_obs:
                continue
            active[ti].observations.append(observations[oi])
            used_tracks.add(ti)
            used_obs.add(oi)

        for oi, obs in enumerate(observations):
            if oi in used_obs:
                continue
            track = PeakTrack(track_id=len(tracks), observations=[obs])
            tracks.append(track)
            active.append(track)

        if observations:
            current_zi = observations[0].z_index
            active = [t for t in active if current_zi - t.observations[-1].z_index < max_z_gap]

    return tracks


def fit_plane(points: np.ndarray) -> PlaneFit | None:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 3:
        return None
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1].astype(float)
    norm = float(np.linalg.norm(normal))
    if norm <= EPS:
        return None
    normal /= norm
    if normal[2] < 0.0:
        normal *= -1.0
    signed = centered @ normal
    return PlaneFit(
        centroid=centroid,
        normal=normal,
        rms=float(np.sqrt(np.mean(signed * signed))),
        max_abs=float(np.max(np.abs(signed))),
    )


def orient_plane_from_inside_point(
    plane: PlaneFit,
    inside_point: np.ndarray,
) -> np.ndarray:
    normal = plane.normal.copy()
    if float((inside_point - plane.centroid) @ normal) > 0.0:
        normal *= -1.0
    return normal


def orient_plane_from_observed_points(
    plane: PlaneFit,
    observed_points: np.ndarray,
    inside_point: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    normal = plane.normal.copy()
    pts = np.asarray(observed_points, dtype=float).reshape((-1, 3))
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 3:
        return orient_plane_from_inside_point(plane, inside_point), {
            "orientation_positive_frac": float("nan"),
            "orientation_point_count": 0,
        }

    signed = (pts - plane.centroid) @ normal
    positive = int(np.sum(signed > 0.0))
    negative = int(np.sum(signed < 0.0))
    if positive > negative:
        normal *= -1.0
        positive = negative

    return normal, {
        "orientation_positive_frac": float(positive / max(1, int(pts.shape[0]))),
        "orientation_point_count": int(pts.shape[0]),
    }


def best_low_region_points(
    region: CyclicRegion,
    *,
    z_index: int,
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    cond_values: np.ndarray | None,
    max_line_condition: float,
    max_samples: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[list[np.ndarray], list[float], int, int, int]:
    ranked: list[tuple[float, int, float | None]] = []
    rejected_by_condition = 0
    invalid_conditions = 0
    condition_total = 0
    for idx in region.indices:
        value = float(fit_rms[int(idx), int(z_index)])
        p = line_points[int(idx), int(z_index), :]
        if (
            np.isfinite(value)
            and np.all(np.isfinite(p))
            and bool(np.all(p >= bounds_min))
            and bool(np.all(p <= bounds_max))
        ):
            cond: float | None = None
            if cond_values is not None:
                condition_total += 1
                raw_cond = float(cond_values[int(idx), int(z_index)])
                if np.isfinite(raw_cond):
                    cond = raw_cond
                    if max_line_condition >= 0.0 and raw_cond > float(max_line_condition):
                        rejected_by_condition += 1
                        continue
                else:
                    invalid_conditions += 1
                    if max_line_condition >= 0.0:
                        rejected_by_condition += 1
                        continue
            ranked.append((value, int(idx), cond))
    ranked.sort(key=lambda item: item[0])
    out: list[np.ndarray] = []
    kept_conditions: list[float] = []
    for _, idx, cond in ranked[: max(1, int(max_samples))]:
        out.append(line_points[idx, int(z_index), :].astype(float))
        if cond is not None and np.isfinite(cond):
            kept_conditions.append(float(cond))
    return out, kept_conditions, rejected_by_condition, invalid_conditions, condition_total


def region_point_rows(
    region: CyclicRegion | None,
    *,
    z_index: int,
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    max_samples: int | None,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> list[dict[str, object]]:
    if region is None:
        return []
    ranked: list[tuple[float, int, np.ndarray]] = []
    for idx in region.indices:
        value = float(fit_rms[int(idx), int(z_index)])
        p = line_points[int(idx), int(z_index), :]
        if (
            np.isfinite(value)
            and np.all(np.isfinite(p))
            and bool(np.all(p >= bounds_min))
            and bool(np.all(p <= bounds_max))
        ):
            ranked.append((value, int(idx), p.astype(float)))
    ranked.sort(key=lambda item: item[0])
    if max_samples is not None:
        ranked = ranked[: max(1, int(max_samples))]
    return [
        {
            "index": int(idx),
            "rms": as_json_float(value),
            "point": as_json_point(point),
        }
        for value, idx, point in ranked
    ]


def track_observation_debug(
    track: PeakTrack,
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    *,
    low_region_samples: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for obs in track.observations:
        zi = int(obs.z_index)
        rows.append(
            {
                "z_index": zi,
                "z": as_json_float(float(obs.z_value)),
                "peak": region_to_json(obs.peak),
                "left_low": region_to_json(obs.left_low),
                "right_low": region_to_json(obs.right_low),
                "left_low_distance": as_json_float(float(obs.left_low_distance))
                if obs.left_low_distance is not None
                else None,
                "right_low_distance": as_json_float(float(obs.right_low_distance))
                if obs.right_low_distance is not None
                else None,
                "left_low_source": str(obs.left_low_source),
                "right_low_source": str(obs.right_low_source),
                "left_crossed_soft_barrier": bool(obs.left_crossed_soft_barrier),
                "right_crossed_soft_barrier": bool(obs.right_crossed_soft_barrier),
                "left_valley_quality": obs.left_valley_quality,
                "right_valley_quality": obs.right_valley_quality,
                "left_fit_points": region_point_rows(
                    obs.left_low,
                    z_index=zi,
                    line_points=line_points,
                    fit_rms=fit_rms,
                    max_samples=low_region_samples,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                ),
                "right_fit_points": region_point_rows(
                    obs.right_low,
                    z_index=zi,
                    line_points=line_points,
                    fit_rms=fit_rms,
                    max_samples=low_region_samples,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                ),
                "peak_points": region_point_rows(
                    obs.peak,
                    z_index=zi,
                    line_points=line_points,
                    fit_rms=fit_rms,
                    max_samples=None,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                ),
            }
        )
    return rows


def track_points(
    track: PeakTrack,
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    *,
    cond_values: np.ndarray | None,
    max_line_condition: float,
    low_region_samples: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], int, int, int]:
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    peak: list[np.ndarray] = []
    kept_conditions: list[float] = []
    rejected_by_condition = 0
    invalid_conditions = 0
    condition_total = 0
    for obs in track.observations:
        zi = int(obs.z_index)
        if obs.left_low is not None:
            pts, conds, rejected, invalid, total = best_low_region_points(
                obs.left_low,
                z_index=zi,
                line_points=line_points,
                fit_rms=fit_rms,
                cond_values=cond_values,
                max_line_condition=max_line_condition,
                max_samples=low_region_samples,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
            )
            left.extend(pts)
            kept_conditions.extend(conds)
            rejected_by_condition += rejected
            invalid_conditions += invalid
            condition_total += total
        if obs.right_low is not None:
            pts, conds, rejected, invalid, total = best_low_region_points(
                obs.right_low,
                z_index=zi,
                line_points=line_points,
                fit_rms=fit_rms,
                cond_values=cond_values,
                max_line_condition=max_line_condition,
                max_samples=low_region_samples,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
            )
            right.extend(pts)
            kept_conditions.extend(conds)
            rejected_by_condition += rejected
            invalid_conditions += invalid
            condition_total += total
        for idx in obs.peak.indices:
            p = line_points[int(idx), zi, :]
            if np.all(np.isfinite(p)) and bool(np.all(p >= bounds_min)) and bool(np.all(p <= bounds_max)):
                peak.append(p.astype(float))
    return (
        np.array(left, dtype=float).reshape((-1, 3)),
        np.array(right, dtype=float).reshape((-1, 3)),
        np.array(peak, dtype=float).reshape((-1, 3)),
        kept_conditions,
        int(rejected_by_condition),
        int(invalid_conditions),
        int(condition_total),
    )


def robust_jump_smoothness(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    diffs = np.abs(np.diff(np.array(values, dtype=float)))
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def aggregate_valley_quality(track: PeakTrack) -> dict[str, object]:
    confidences: list[float] = []
    distances: list[float] = []
    left_centers: list[float] = []
    right_centers: list[float] = []
    one_side = 0
    two_side = 0
    low_conf = 0
    valley_obs = 0
    for obs in track.observations:
        left_is_valley = obs.left_low_source == "valley"
        right_is_valley = obs.right_low_source == "valley"
        if left_is_valley or right_is_valley:
            valley_obs += 1
        if left_is_valley ^ right_is_valley:
            one_side += 1
        if left_is_valley and right_is_valley:
            two_side += 1
        for is_valley, quality, region, distance in (
            (left_is_valley, obs.left_valley_quality, obs.left_low, obs.left_low_distance),
            (right_is_valley, obs.right_valley_quality, obs.right_low, obs.right_low_distance),
        ):
            if not is_valley:
                continue
            if quality is not None and quality.get("valley_confidence") is not None:
                conf = float(quality["valley_confidence"])  # type: ignore[arg-type]
                confidences.append(conf)
                if conf < 0.5:
                    low_conf += 1
            if distance is not None:
                distances.append(float(distance))
            if region is not None:
                if quality is obs.left_valley_quality:
                    left_centers.append(float(region.center))
                else:
                    right_centers.append(float(region.center))
    total_obs = max(1, len(track.observations))
    confidence_arr = np.array(confidences, dtype=float)
    distance_arr = np.array(distances, dtype=float)
    left_smooth = robust_jump_smoothness(left_centers)
    right_smooth = robust_jump_smoothness(right_centers)
    bilateral_balance = 1.0 - abs(float(one_side) - float(two_side * 2)) / max(float(one_side + two_side * 2), 1.0)
    return {
        "valley_confidence_median": as_json_float(float(np.median(confidence_arr))) if confidence_arr.size else None,
        "valley_confidence_p10": as_json_float(float(np.percentile(confidence_arr, 10))) if confidence_arr.size else None,
        "valley_two_side_fraction": as_json_float(float(two_side) / float(total_obs)),
        "valley_one_side_fraction": as_json_float(float(one_side) / float(total_obs)),
        "valley_distance_median": as_json_float(float(np.median(distance_arr))) if distance_arr.size else None,
        "valley_distance_p95": as_json_float(float(np.percentile(distance_arr, 95))) if distance_arr.size else None,
        "valley_left_track_smoothness": as_json_float(float(left_smooth)) if left_smooth is not None else None,
        "valley_right_track_smoothness": as_json_float(float(right_smooth)) if right_smooth is not None else None,
        "valley_bilateral_balance": as_json_float(float(np.clip(bilateral_balance, 0.0, 1.0))),
        "valley_scale_persistence": None,
        "valley_low_confidence_observations": int(low_conf),
        "valley_observation_count": int(valley_obs),
    }


def build_face_candidates(
    tracks: list[PeakTrack],
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    *,
    window: int | None,
    track_id_offset: int,
    cond_values: np.ndarray | None,
    max_line_condition: float,
    min_track_levels: int,
    min_fit_points: int,
    max_plane_rms: float,
    inside_point: np.ndarray,
    orientation_points: np.ndarray,
    support_points: np.ndarray,
    support_tol: float,
    max_support_outside_count: int,
    low_region_samples: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for track in tracks:
        if len(track.observations) < min_track_levels:
            continue
        left, right, peak_pts, kept_conditions, condition_rejected, invalid_conditions, condition_total = track_points(
            track,
            line_points,
            fit_rms,
            cond_values=cond_values,
            max_line_condition=float(max_line_condition),
            low_region_samples=int(low_region_samples),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        fit_pts = np.vstack([left, right]) if left.size and right.size else np.zeros((0, 3), dtype=float)
        if fit_pts.shape[0] < min_fit_points:
            continue
        plane = fit_plane(fit_pts)
        if plane is None or plane.rms > max_plane_rms:
            continue
        support_normal, orientation = orient_plane_from_observed_points(
            plane,
            orientation_points,
            inside_point,
        )
        support_outside_count = 0
        support_positive_max = float("nan")
        if support_points.size:
            signed_support = (support_points - plane.centroid) @ support_normal
            support_outside_count = int(np.sum(signed_support > float(support_tol)))
            support_positive_max = float(np.max(signed_support))
            if max_support_outside_count >= 0 and support_outside_count > int(max_support_outside_count):
                continue

        u, v = plane_basis(support_normal)
        rel = fit_pts - plane.centroid
        coords = np.column_stack([rel @ u, rel @ v])
        hull_idx = convex_hull_indices(coords)
        if len(hull_idx) >= 3:
            hull_coords = coords[np.array(hull_idx, dtype=int)]
        else:
            hull_coords = coords
        hull = plane.centroid + hull_coords[:, 0:1] * u[None, :] + hull_coords[:, 1:2] * v[None, :]
        if hull.shape[0] >= 2:
            hull_diff = hull[:, None, :] - hull[None, :, :]
            hull_diameter = float(np.sqrt(np.max(np.sum(hull_diff * hull_diff, axis=2))))
        else:
            hull_diameter = 0.0
        hull_area_vec = np.zeros(3, dtype=float)
        if hull.shape[0] >= 3:
            for hi in range(hull.shape[0]):
                hull_area_vec += np.cross(hull[hi], hull[(hi + 1) % hull.shape[0]])
        hull_area = float(0.5 * np.linalg.norm(hull_area_vec))
        z_indices = [int(o.z_index) for o in track.observations]
        centers = [float(o.peak.center) for o in track.observations]
        low_distances = [
            float(d)
            for obs in track.observations
            for d in (obs.left_low_distance, obs.right_low_distance)
            if d is not None
        ]
        low_source_counts = {
            "threshold": 0,
            "threshold-cross-soft": 0,
            "valley": 0,
        }
        for obs in track.observations:
            low_source_counts[str(obs.left_low_source)] = low_source_counts.get(str(obs.left_low_source), 0) + 1
            low_source_counts[str(obs.right_low_source)] = low_source_counts.get(str(obs.right_low_source), 0) + 1
        low_source_total = max(1, sum(int(v) for v in low_source_counts.values()))
        valley_frac = float(low_source_counts.get("valley", 0)) / float(low_source_total)
        crossed_soft_frac = float(low_source_counts.get("threshold-cross-soft", 0)) / float(low_source_total)
        valley_quality = aggregate_valley_quality(track)
        condition_arr = np.array(kept_conditions, dtype=float)
        condition_median = float(np.median(condition_arr)) if condition_arr.size else float("nan")
        condition_p95 = float(np.percentile(condition_arr, 95)) if condition_arr.size else float("nan")
        condition_max = float(np.max(condition_arr)) if condition_arr.size else float("nan")
        invalid_condition_frac = (
            float(invalid_conditions) / float(condition_total)
            if condition_total > 0
            else float("nan")
        )

        source_penalty = 1.0 + 0.75 * valley_frac + 0.25 * crossed_soft_frac
        condition_penalty = 1.0
        if np.isfinite(condition_median):
            condition_penalty += min(max(np.log10(max(condition_median, 1.0)) - 3.0, 0.0), 4.0) * 0.08
        candidate_score = (
            float(plane.rms)
            * source_penalty
            * condition_penalty
            / max(float(len(track.observations)) ** 0.5, 1.0)
        )
        candidates.append(
            {
                "track_id": int(track.track_id) + int(track_id_offset),
                "source_track_id": int(track.track_id),
                "window": int(window) if window is not None else None,
                "levels": len(track.observations),
                "z_min": float(min(o.z_value for o in track.observations)),
                "z_max": float(max(o.z_value for o in track.observations)),
                "center_mean": float(np.mean(centers)),
                "center_min": float(np.min(centers)),
                "center_max": float(np.max(centers)),
                "left_points": int(left.shape[0]),
                "right_points": int(right.shape[0]),
                "peak_points": int(peak_pts.shape[0]),
                "fit_points": int(fit_pts.shape[0]),
                "low_distance_mean": as_json_float(float(np.mean(low_distances))) if low_distances else None,
                "low_distance_max": as_json_float(float(np.max(low_distances))) if low_distances else None,
                "low_source_counts": {str(k): int(v) for k, v in low_source_counts.items()},
                "low_valley_frac": as_json_float(valley_frac),
                "low_crossed_soft_frac": as_json_float(crossed_soft_frac),
                **valley_quality,
                "condition_median": as_json_float(condition_median),
                "condition_p95": as_json_float(condition_p95),
                "condition_max": as_json_float(condition_max),
                "condition_rejected_fit_points": int(condition_rejected),
                "condition_invalid_fraction": as_json_float(invalid_condition_frac),
                "plane_centroid": as_json_point(plane.centroid),
                "plane_normal": as_json_point(support_normal),
                "plane_rms": as_json_float(plane.rms),
                "plane_max_abs": as_json_float(plane.max_abs),
                "orientation_positive_frac": as_json_float(float(orientation["orientation_positive_frac"])),
                "orientation_point_count": int(orientation["orientation_point_count"]),
                "support_outside_count": int(support_outside_count),
                "support_positive_max": as_json_float(support_positive_max),
                "candidate_score": as_json_float(candidate_score),
                "hull_diameter": as_json_float(hull_diameter),
                "hull_area": as_json_float(hull_area),
                "hull": [as_json_point(p) for p in hull],
                "left_sample_points": [as_json_point(p) for p in left],
                "right_sample_points": [as_json_point(p) for p in right],
                "sample_points": [as_json_point(p) for p in fit_pts[:: max(1, fit_pts.shape[0] // 80)]],
                "peak_sample_points": [as_json_point(p) for p in peak_pts[:: max(1, max(1, peak_pts.shape[0]) // 50)]],
                "z_indices": z_indices,
                "debug_observations": track_observation_debug(
                    track,
                    line_points,
                    fit_rms,
                    low_region_samples=int(low_region_samples),
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                ),
            }
        )
    candidates.sort(
        key=lambda c: (
            float(c.get("candidate_score") or float("inf")),
            float(c.get("hull_diameter") or float("inf")),
            -int(c.get("levels") or 0),
        )
    )
    return candidates


def build_w2_face_candidates(
    clusters: list[dict[str, object]],
    z_levels: np.ndarray,
    *,
    inside_point: np.ndarray,
    orientation_points: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    min_adjacency_levels: int,
    min_adjacency_fraction: float,
    min_z_span: float,
    max_plane_rms: float,
    duplicate_pair_mode: str = "allow",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    active_by_z: dict[int, list[dict[str, object]]] = {}
    for cluster in clusters:
        z0 = finite_float(cluster.get("cluster_z_min"), float("nan"))
        z1 = finite_float(cluster.get("cluster_z_max"), float("nan"))
        for zi, z in enumerate(z_levels):
            if np.isfinite(z0) and np.isfinite(z1) and z0 <= float(z) <= z1:
                active_by_z.setdefault(int(zi), []).append(cluster)
    pair_levels: dict[tuple[int, int], list[int]] = {}
    for zi, rows in active_by_z.items():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda c: float(np.mean(c.get("member_cyclic_indices") or [c.get("cyclic_index") or 0])))
        for a, b in zip(rows, rows[1:] + rows[:1]):
            key = tuple(sorted((int(a["edge_cluster_id"]), int(b["edge_cluster_id"]))))
            pair_levels.setdefault(key, []).append(int(zi))
    by_id = {int(c["edge_cluster_id"]): c for c in clusters}
    candidates: list[dict[str, object]] = []
    rejected: dict[str, int] = {}
    for (a_id, b_id), levels in pair_levels.items():
        if len(levels) < int(min_adjacency_levels):
            rejected["insufficient_adjacency"] = rejected.get("insufficient_adjacency", 0) + 1
            continue
        z = z_levels[np.array(levels, dtype=int)]
        z_span = float(np.max(z) - np.min(z)) if z.size else 0.0
        if z_span < float(min_z_span):
            rejected["insufficient_adjacency"] = rejected.get("insufficient_adjacency", 0) + 1
            continue
        a = by_id[a_id]
        b = by_id[b_id]
        if str(duplicate_pair_mode) == "reject-overlapping-collinear" and is_overlapping_collinear_duplicate_pair(a, b):
            rejected["duplicate_overlapping_collinear_pair"] = rejected.get("duplicate_overlapping_collinear_pair", 0) + 1
            continue
        overlap = min(finite_float(a.get("cluster_z_max"), -float("inf")), finite_float(b.get("cluster_z_max"), -float("inf"))) - max(
            finite_float(a.get("cluster_z_min"), float("inf")), finite_float(b.get("cluster_z_min"), float("inf"))
        )
        denom = max(float(overlap) / max(float(np.median(np.diff(z_levels))), EPS), 1.0) if np.isfinite(overlap) else float(len(levels))
        adjacency_fraction = float(len(levels)) / float(denom)
        if adjacency_fraction < float(min_adjacency_fraction):
            rejected["insufficient_adjacency"] = rejected.get("insufficient_adjacency", 0) + 1
            continue
        pts = np.vstack([predict_w2_line(a, z), predict_w2_line(b, z)])
        if not (np.all(np.isfinite(pts)) and np.all(pts >= bounds_min[None, :]) and np.all(pts <= bounds_max[None, :])):
            rejected["poor_plane_fit"] = rejected.get("poor_plane_fit", 0) + 1
            continue
        plane = fit_plane(pts)
        if plane is None or float(plane.rms) > float(max_plane_rms):
            rejected["poor_plane_fit"] = rejected.get("poor_plane_fit", 0) + 1
            continue
        normal, orientation = orient_plane_from_observed_points(plane, orientation_points, inside_point)
        u, v = plane_basis(normal)
        rel = pts - plane.centroid
        coords = np.column_stack([rel @ u, rel @ v])
        hull_idx = convex_hull_indices(coords)
        hull = pts[np.array(hull_idx, dtype=int)] if len(hull_idx) >= 3 else pts
        hull_area = polygon_area(hull)
        diameter = float(np.max(np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2))) if hull.shape[0] >= 2 else 0.0
        candidates.append(
            {
                "track_id": 9000000 + int(len(candidates)),
                "source_track_id": None,
                "window": 2,
                "levels": int(len(levels)),
                "z_min": as_json_float(float(np.min(z))),
                "z_max": as_json_float(float(np.max(z))),
                "center_mean": as_json_float(float(np.mean([np.mean(a.get("member_cyclic_indices") or [0]), np.mean(b.get("member_cyclic_indices") or [0])]))),
                "fit_points": int(pts.shape[0]),
                "candidate_source": "w2_edge_adjacency",
                "candidate_origin": "w2_addition",
                "w2_edge_cluster_ids": [int(a_id), int(b_id)],
                "w2_adjacency_levels": int(len(levels)),
                "w2_adjacency_fraction": as_json_float(float(adjacency_fraction)),
                "w2_edge_confidence": as_json_float(float(np.mean([finite_float(a.get("cluster_confidence"), 0.0), finite_float(b.get("cluster_confidence"), 0.0)]))),
                "plane_centroid": as_json_point(plane.centroid),
                "plane_normal": as_json_point(normal),
                "plane_rms": as_json_float(float(plane.rms)),
                "plane_max_abs": as_json_float(float(plane.max_abs)),
                "orientation_positive_frac": as_json_float(float(orientation["orientation_positive_frac"])),
                "orientation_point_count": int(orientation["orientation_point_count"]),
                "candidate_score": as_json_float(float(plane.rms) / max(float(len(levels)) ** 0.5, 1.0)),
                "hull_diameter": as_json_float(float(diameter)),
                "hull_area": as_json_float(float(hull_area)),
                "hull": [as_json_point(p) for p in hull],
                "sample_points": [as_json_point(p) for p in pts[:: max(1, pts.shape[0] // 80)]],
                "z_indices": [int(v) for v in levels],
                "low_source_counts": {"w2_edge": int(2 * len(levels))},
                "low_valley_frac": 0.0,
            }
        )
    return candidates, {
        "w2_adjacency_pairs": int(len(pair_levels)),
        "w2_face_candidates": int(len(candidates)),
        "w2_duplicate_pair_mode": str(duplicate_pair_mode),
        "w2_face_rejection_counts": rejected,
    }
