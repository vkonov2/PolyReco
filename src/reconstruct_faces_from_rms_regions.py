from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from generate_full_circle_split_cached_viewer import (
    EPS,
    build_half_contours,
    parse_initial_model,
    parse_merged_contour,
    sorted_contour_files,
    top_points_vertical_axis_point,
    triangulate_faces,
)
from match_full_circle_points_to_edges import (
    compute_line_points,
    compute_line_points_multi_window,
    discover_single_model,
    relative_plotly_src,
)


SCHEMA_VERSION = 1


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


def as_json_float(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def as_json_point(p: np.ndarray) -> list[float | None]:
    return [as_json_float(float(v)) for v in p]


def parse_windows(value: str, fallback: int) -> list[int]:
    raw = [part.strip() for part in value.split(",") if part.strip()]
    if not raw:
        return [int(fallback)]
    windows = sorted({max(2, int(part)) for part in raw})
    return windows


def cyclic_distance(a: float, b: float, n: int) -> float:
    d = abs(float(a) - float(b)) % float(n)
    return float(min(d, float(n) - d))


def contiguous_regions(mask: np.ndarray, values: np.ndarray) -> list[CyclicRegion]:
    n = int(mask.size)
    if n == 0 or not np.any(mask):
        return []
    if bool(np.all(mask)):
        idx = tuple(range(n))
        finite_vals = values[np.array(idx, dtype=int)]
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        return [
            CyclicRegion(
                indices=idx,
                start=0,
                end=n - 1,
                center=(n - 1) / 2.0,
                width=n,
                min_value=float(np.min(finite_vals)) if finite_vals.size else float("nan"),
                max_value=float(np.max(finite_vals)) if finite_vals.size else float("nan"),
                mean_value=float(np.mean(finite_vals)) if finite_vals.size else float("nan"),
            )
        ]

    start = int(np.flatnonzero(~mask)[0])
    ordered = [(start + 1 + k) % n for k in range(n)]
    runs: list[list[int]] = []
    cur: list[int] = []
    for idx in ordered:
        if bool(mask[idx]):
            cur.append(int(idx))
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    out: list[CyclicRegion] = []
    for run in runs:
        vals = values[np.array(run, dtype=int)]
        vals = vals[np.isfinite(vals)]
        unwrapped = np.array(run, dtype=float)
        for i in range(1, unwrapped.size):
            if unwrapped[i] < unwrapped[i - 1]:
                unwrapped[i:] += n
        center = float(np.mean(unwrapped) % n)
        out.append(
            CyclicRegion(
                indices=tuple(int(i) for i in run),
                start=int(run[0]),
                end=int(run[-1]),
                center=center,
                width=len(run),
                min_value=float(np.min(vals)) if vals.size else float("nan"),
                max_value=float(np.max(vals)) if vals.size else float("nan"),
                mean_value=float(np.mean(vals)) if vals.size else float("nan"),
            )
        )
    return out


def level_thresholds(values: np.ndarray, peak_q: float, low_q: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return float("nan"), float("nan")
    peak_threshold = float(np.quantile(finite, np.clip(peak_q, 0.0, 1.0)))
    low_threshold = float(np.quantile(finite, np.clip(low_q, 0.0, 1.0)))
    return peak_threshold, low_threshold


def nearest_region(
    regions: list[CyclicRegion],
    center: float,
    n: int,
    side: str,
) -> tuple[CyclicRegion | None, float | None]:
    if not regions:
        return None, None
    best: tuple[float, CyclicRegion] | None = None
    for region in regions:
        if side == "left":
            dist = (float(center) - float(region.center)) % float(n)
        else:
            dist = (float(region.center) - float(center)) % float(n)
        if dist <= 0.0:
            dist = float(n)
        item = (dist, region)
        if best is None or item[0] < best[0]:
            best = item
    if best is None or best[0] >= float(n):
        return None, None
    return best[1], float(best[0])


def adaptive_windows_from_rms(
    fit_rms_by_window: np.ndarray,
    windows: list[int],
    *,
    smooth_weight: float,
    max_relative_rms: float,
    max_abs_rms_gap: float,
    large_window_bonus: float,
) -> tuple[np.ndarray, dict[str, object]]:
    n_windows = int(fit_rms_by_window.shape[0])
    n_levels = int(fit_rms_by_window.shape[2])
    quality = np.full((n_levels, n_windows), np.inf, dtype=float)
    medians = np.full((n_levels, n_windows), np.nan, dtype=float)
    invalid_penalties = np.full((n_levels, n_windows), 1.0, dtype=float)

    for wi in range(n_windows):
        for zi in range(n_levels):
            values = fit_rms_by_window[wi, :, zi].astype(float)
            finite = np.isfinite(values)
            finite_count = int(np.sum(finite))
            if finite_count < 4:
                continue
            median = float(np.median(values[finite]))
            medians[zi, wi] = median
            invalid_penalties[zi, wi] = 1.0 - finite_count / max(1, int(values.size))

    window_arr = np.array(windows, dtype=float)
    span = max(float(np.max(window_arr) - np.min(window_arr)), 1.0)
    size_norm = (window_arr - float(np.min(window_arr))) / span
    for zi in range(n_levels):
        row = medians[zi]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        best = float(np.min(row[finite]))
        allowed = max(best * max(1.0, float(max_relative_rms)), best + max(0.0, float(max_abs_rms_gap)))
        scale = max(allowed - best, best, EPS)
        for wi in range(n_windows):
            if not np.isfinite(row[wi]):
                continue
            excess = max(0.0, float(row[wi]) - allowed) / scale
            within = max(0.0, float(row[wi]) - best) / scale
            quality[zi, wi] = within + 20.0 * excess + invalid_penalties[zi, wi] - float(large_window_bonus) * size_norm[wi]

    safe_quality = quality.copy()
    finite_quality = np.isfinite(safe_quality)
    fallback = float(np.median(safe_quality[finite_quality])) if np.any(finite_quality) else 1.0
    safe_quality[~finite_quality] = fallback * 10.0 + 1.0

    dp = np.full((n_levels, n_windows), np.inf, dtype=float)
    back = np.zeros((n_levels, n_windows), dtype=int)
    dp[0, :] = safe_quality[0, :]
    for zi in range(1, n_levels):
        for wi in range(n_windows):
            transition = float(smooth_weight) * np.abs(window_arr[wi] - window_arr) / span
            prev = dp[zi - 1, :] + transition
            best_prev = int(np.argmin(prev))
            dp[zi, wi] = float(prev[best_prev] + safe_quality[zi, wi])
            back[zi, wi] = best_prev

    selected = np.zeros(n_levels, dtype=int)
    selected[-1] = int(np.argmin(dp[-1, :]))
    for zi in range(n_levels - 1, 0, -1):
        selected[zi - 1] = int(back[zi, selected[zi]])

    selected_windows = [int(windows[int(i)]) for i in selected]
    return selected, {
        "adaptive_window_min_selected": int(min(selected_windows)) if selected_windows else None,
        "adaptive_window_max_selected": int(max(selected_windows)) if selected_windows else None,
        "adaptive_window_median_selected": as_json_float(float(np.median(selected_windows))) if selected_windows else None,
        "adaptive_window_changes": int(np.sum(np.diff(selected) != 0)) if len(selected) > 1 else 0,
        "adaptive_window_sequence": selected_windows,
    }


def apply_selected_windows(
    line_points_by_window: np.ndarray,
    fit_rms_by_window: np.ndarray,
    selected_window_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_half = int(line_points_by_window.shape[1])
    n_levels = int(line_points_by_window.shape[2])
    line_points = np.full((n_half, n_levels, 3), np.nan, dtype=float)
    fit_rms = np.full((n_half, n_levels), np.nan, dtype=float)

    for zi, wi_raw in enumerate(selected_window_indices):
        wi = int(wi_raw)
        line_points[:, zi, :] = line_points_by_window[wi, :, zi, :].astype(float)
        fit_rms[:, zi] = fit_rms_by_window[wi, :, zi].astype(float)

    return line_points, fit_rms


def detect_peak_observations(
    z_levels: np.ndarray,
    fit_rms: np.ndarray,
    *,
    peak_threshold_abs: float,
    low_threshold_abs: float,
    peak_quantile: float,
    low_quantile: float,
    min_peak_width: int,
    max_peak_width: int,
    max_low_distance: float,
) -> tuple[list[list[PeakObservation]], list[dict[str, object]]]:
    n_half, n_levels = fit_rms.shape
    per_level: list[list[PeakObservation]] = []
    diagnostics: list[dict[str, object]] = []

    for zi in range(n_levels):
        values = fit_rms[:, zi].astype(float)
        finite = np.isfinite(values)
        if peak_threshold_abs >= 0.0 and low_threshold_abs >= 0.0:
            peak_threshold = float(peak_threshold_abs)
            low_threshold = float(low_threshold_abs)
        else:
            peak_threshold, low_threshold = level_thresholds(values, peak_quantile, low_quantile)
        if not np.isfinite(peak_threshold) or not np.isfinite(low_threshold):
            per_level.append([])
            diagnostics.append(
                {
                    "z_index": zi,
                    "z": float(z_levels[zi]),
                    "peak_threshold": None,
                    "low_threshold": None,
                    "peaks": 0,
                    "low_regions": 0,
                }
            )
            continue

        peak_mask = finite & (values >= peak_threshold)
        low_mask = finite & (values <= low_threshold)
        peak_regions = [
            r
            for r in contiguous_regions(peak_mask, values)
            if int(r.width) >= int(min_peak_width) and int(r.width) <= int(max_peak_width)
        ]
        low_regions = contiguous_regions(low_mask, values)

        observations: list[PeakObservation] = []
        for peak in peak_regions:
            left_low, left_dist = nearest_region(low_regions, peak.center, n_half, "left")
            right_low, right_dist = nearest_region(low_regions, peak.center, n_half, "right")
            if left_low is None or right_low is None or left_dist is None or right_dist is None:
                continue
            if max_low_distance >= 0.0 and (left_dist > max_low_distance or right_dist > max_low_distance):
                continue
            observations.append(
                PeakObservation(
                    z_index=zi,
                    z_value=float(z_levels[zi]),
                    peak=peak,
                    left_low=left_low,
                    right_low=right_low,
                    left_low_distance=float(left_dist),
                    right_low_distance=float(right_dist),
                )
            )
        per_level.append(observations)
        diagnostics.append(
            {
                "z_index": zi,
                "z": float(z_levels[zi]),
                "peak_threshold": float(peak_threshold),
                "low_threshold": float(low_threshold),
                "peaks": len(peak_regions),
                "low_regions": len(low_regions),
            }
        )
    return per_level, diagnostics


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


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / max(float(np.linalg.norm(normal)), EPS)
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(ref @ n)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(ref, n)
    u /= max(float(np.linalg.norm(u)), EPS)
    v = np.cross(n, u)
    v /= max(float(np.linalg.norm(v)), EPS)
    return u, v


def convex_hull_indices(points_2d: np.ndarray) -> list[int]:
    pts = [(float(p[0]), float(p[1]), i) for i, p in enumerate(points_2d)]
    pts.sort(key=lambda x: (x[0], x[1], x[2]))
    if len(pts) <= 2:
        return [p[2] for p in pts]

    def cross(o: tuple[float, float, int], a: tuple[float, float, int], b: tuple[float, float, int]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float, int]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)

    return [p[2] for p in lower[:-1] + upper[:-1]]


def best_low_region_points(
    region: CyclicRegion,
    *,
    z_index: int,
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    max_samples: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> list[np.ndarray]:
    ranked: list[tuple[float, int]] = []
    for idx in region.indices:
        value = float(fit_rms[int(idx), int(z_index)])
        p = line_points[int(idx), int(z_index), :]
        if (
            np.isfinite(value)
            and np.all(np.isfinite(p))
            and bool(np.all(p >= bounds_min))
            and bool(np.all(p <= bounds_max))
        ):
            ranked.append((value, int(idx)))
    ranked.sort(key=lambda item: item[0])
    out: list[np.ndarray] = []
    for _, idx in ranked[: max(1, int(max_samples))]:
        out.append(line_points[idx, int(z_index), :].astype(float))
    return out


def track_points(
    track: PeakTrack,
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    *,
    low_region_samples: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    peak: list[np.ndarray] = []
    for obs in track.observations:
        zi = int(obs.z_index)
        if obs.left_low is not None:
            left.extend(
                best_low_region_points(
                    obs.left_low,
                    z_index=zi,
                    line_points=line_points,
                    fit_rms=fit_rms,
                    max_samples=low_region_samples,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                )
            )
        if obs.right_low is not None:
            right.extend(
                best_low_region_points(
                    obs.right_low,
                    z_index=zi,
                    line_points=line_points,
                    fit_rms=fit_rms,
                    max_samples=low_region_samples,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                )
            )
        for idx in obs.peak.indices:
            p = line_points[int(idx), zi, :]
            if np.all(np.isfinite(p)) and bool(np.all(p >= bounds_min)) and bool(np.all(p <= bounds_max)):
                peak.append(p.astype(float))
    return (
        np.array(left, dtype=float).reshape((-1, 3)),
        np.array(right, dtype=float).reshape((-1, 3)),
        np.array(peak, dtype=float).reshape((-1, 3)),
    )


def build_face_candidates(
    tracks: list[PeakTrack],
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    *,
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
        left, right, peak_pts = track_points(
            track,
            line_points,
            fit_rms,
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

        candidate_score = float(plane.rms) / max(float(len(track.observations)) ** 0.5, 1.0)
        candidates.append(
            {
                "track_id": int(track.track_id),
                "window": None,
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
                "sample_points": [as_json_point(p) for p in fit_pts[:: max(1, fit_pts.shape[0] // 80)]],
                "peak_sample_points": [as_json_point(p) for p in peak_pts[:: max(1, max(1, peak_pts.shape[0]) // 50)]],
                "z_indices": z_indices,
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


def dedupe_points(points: list[np.ndarray], tol: float) -> np.ndarray:
    out: list[np.ndarray] = []
    for p in points:
        found = False
        for i, q in enumerate(out):
            if float(np.linalg.norm(p - q)) <= tol:
                out[i] = 0.5 * (q + p)
                found = True
                break
        if not found:
            out.append(p.astype(float))
    if not out:
        return np.zeros((0, 3), dtype=float)
    return np.array(out, dtype=float)


def order_face_vertices(points: np.ndarray, normal: np.ndarray) -> list[int]:
    if points.shape[0] < 3:
        return []
    centroid = np.mean(points, axis=0)
    u, v = plane_basis(normal)
    coords = np.column_stack([(points - centroid) @ u, (points - centroid) @ v])
    hull = convex_hull_indices(coords)
    if len(hull) < 3:
        return []
    ordered = points[np.array(hull, dtype=int)]
    area_vec = np.zeros(3, dtype=float)
    for i in range(ordered.shape[0]):
        area_vec += np.cross(ordered[i], ordered[(i + 1) % ordered.shape[0]])
    if float(area_vec @ normal) < 0.0:
        hull = list(reversed(hull))
    return [int(i) for i in hull]


def reconstruct_polyhedron_from_halfspaces(
    candidates: list[dict[str, object]],
    *,
    inside_tol: float,
    vertex_tol: float,
) -> dict[str, object]:
    planes: list[tuple[np.ndarray, np.ndarray, int]] = []
    for pi, candidate in enumerate(candidates):
        normal = np.array(candidate.get("plane_normal", []), dtype=float)
        point = np.array(candidate.get("plane_centroid", []), dtype=float)
        if normal.shape != (3,) or point.shape != (3,):
            continue
        if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(point)):
            continue
        norm = float(np.linalg.norm(normal))
        if norm <= EPS:
            continue
        planes.append((normal / norm, point, pi))

    if planes:
        normals_arr = np.array([p[0] for p in planes], dtype=float)
        offsets_arr = np.array([float(p[0] @ p[1]) for p in planes], dtype=float)
    else:
        normals_arr = np.zeros((0, 3), dtype=float)
        offsets_arr = np.zeros((0,), dtype=float)

    raw_vertices: list[np.ndarray] = []
    n_planes = len(planes)
    for ia in range(n_planes):
        na, pa, _ = planes[ia]
        for ib in range(ia + 1, n_planes):
            nb, pb, _ = planes[ib]
            for ic in range(ib + 1, n_planes):
                nc, pc, _ = planes[ic]
                mat = np.vstack([na, nb, nc])
                det = float(np.linalg.det(mat))
                if abs(det) <= 1e-8:
                    continue
                rhs = np.array([float(na @ pa), float(nb @ pb), float(nc @ pc)], dtype=float)
                try:
                    x = np.linalg.solve(mat, rhs)
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(x)):
                    continue
                signed = normals_arr @ x - offsets_arr
                if np.max(signed) <= float(inside_tol):
                    raw_vertices.append(x)

    vertices = dedupe_points(raw_vertices, vertex_tol)
    faces: list[list[int]] = []
    face_sources: list[int] = []
    for normal, point, candidate_index in planes:
        if vertices.shape[0] < 3:
            break
        dist = np.abs((vertices - point) @ normal)
        local_idx = np.flatnonzero(dist <= max(float(inside_tol) * 2.0, float(vertex_tol) * 4.0))
        if local_idx.size < 3:
            continue
        ordered_local = order_face_vertices(vertices[local_idx], normal)
        if len(ordered_local) < 3:
            continue
        faces.append([int(local_idx[i]) for i in ordered_local])
        face_sources.append(int(candidate_index))

    return {
        "vertices": [as_json_point(p) for p in vertices],
        "faces": faces,
        "face_candidate_indices": face_sources,
        "raw_intersections": int(len(raw_vertices)),
        "planes": int(n_planes),
        "inside_tol": float(inside_tol),
        "vertex_tol": float(vertex_tol),
    }


def polygon_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    centroid = np.mean(points, axis=0)
    area = 0.0
    for i in range(points.shape[0]):
        area += 0.5 * float(np.linalg.norm(np.cross(points[i] - centroid, points[(i + 1) % points.shape[0]] - centroid)))
    return float(area)


def filter_candidates_by_active_face_size(
    candidates: list[dict[str, object]],
    *,
    inside_tol: float,
    vertex_tol: float,
    max_area_ratio: float,
    max_extra_area: float,
    iterations: int,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    current = list(candidates)
    total_rejected = 0
    last_rejected: list[dict[str, object]] = []
    reconstructed = reconstruct_polyhedron_from_halfspaces(
        current,
        inside_tol=inside_tol,
        vertex_tol=vertex_tol,
    )
    if max_area_ratio < 0.0 and max_extra_area < 0.0:
        return current, reconstructed, {"active_face_size_rejected": 0, "active_face_size_iterations": 0}

    max_iter = max(1, int(iterations))
    for iteration in range(max_iter):
        reconstructed = reconstruct_polyhedron_from_halfspaces(
            current,
            inside_tol=inside_tol,
            vertex_tol=vertex_tol,
        )
        vertices = np.array(reconstructed.get("vertices", []), dtype=float)
        faces = reconstructed.get("faces", [])
        sources = reconstructed.get("face_candidate_indices", [])
        if vertices.size == 0 or not faces or not sources:
            break

        reject: set[int] = set()
        rejected_rows: list[dict[str, object]] = []
        for face, source_raw in zip(faces, sources):
            source = int(source_raw)
            if source < 0 or source >= len(current):
                continue
            idx = np.array(face, dtype=int)
            if idx.size < 3 or np.any(idx < 0) or np.any(idx >= vertices.shape[0]):
                continue
            face_area = polygon_area(vertices[idx])
            hull_area = float(current[source].get("hull_area") or 0.0)
            ratio = face_area / max(hull_area, EPS)
            too_large_by_ratio = max_area_ratio >= 0.0 and ratio > float(max_area_ratio)
            too_large_by_extra = max_extra_area >= 0.0 and (face_area - hull_area) > float(max_extra_area)
            if too_large_by_ratio and too_large_by_extra:
                reject.add(source)
                rejected_rows.append(
                    {
                        "candidate_index": int(source),
                        "track_id": current[source].get("track_id"),
                        "face_area": as_json_float(face_area),
                        "hull_area": as_json_float(hull_area),
                        "area_ratio": as_json_float(ratio),
                    }
                )
        if not reject:
            return current, reconstructed, {
                "active_face_size_rejected": int(total_rejected),
                "active_face_size_iterations": int(iteration),
                "active_face_size_last_rejected": last_rejected[:20],
            }
        current = [candidate for i, candidate in enumerate(current) if i not in reject]
        total_rejected += len(reject)
        last_rejected = rejected_rows

    reconstructed = reconstruct_polyhedron_from_halfspaces(
        current,
        inside_tol=inside_tol,
        vertex_tol=vertex_tol,
    )
    return current, reconstructed, {
        "active_face_size_rejected": int(total_rejected),
        "active_face_size_iterations": int(max_iter),
        "active_face_size_last_rejected": last_rejected[:20],
    }


def build_payload(
    *,
    model_name: str,
    vertices: np.ndarray,
    faces: list[list[int]],
    z_levels: np.ndarray,
    fit_rms: np.ndarray,
    line_points: np.ndarray,
    tracks: list[PeakTrack],
    diagnostics: list[dict[str, object]],
    candidates: list[dict[str, object]],
    reconstructed: dict[str, object],
    parameters: dict[str, object],
) -> dict[str, object]:
    finite = fit_rms[np.isfinite(fit_rms)]
    track_lengths = [len(t.observations) for t in tracks]
    peak_observations = int(sum(int(item.get("peaks", 0) or 0) for item in diagnostics))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "parameters": parameters,
        "summary": {
            "z_levels": int(z_levels.size),
            "n_half": int(fit_rms.shape[0]),
            "fit_rms_min": as_json_float(float(np.min(finite))) if finite.size else None,
            "fit_rms_max": as_json_float(float(np.max(finite))) if finite.size else None,
            "fit_rms_median": as_json_float(float(np.median(finite))) if finite.size else None,
            "peak_observations": peak_observations,
            "tracks": int(len(tracks)),
            "tracks_kept": int(len(candidates)),
            "reconstructed_vertices": len(reconstructed.get("vertices", [])),
            "reconstructed_faces": len(reconstructed.get("faces", [])),
            "track_levels_median": as_json_float(float(np.median(track_lengths))) if track_lengths else None,
        },
        "viewer": {
            "vertices": [as_json_point(p) for p in vertices],
            "triangles": triangulate_faces(faces).astype(int).tolist(),
            "faces": [[int(i) for i in face] for face in faces],
            "z_levels": [float(z) for z in z_levels],
            "rms_profile": [
                [as_json_float(float(v)) for v in fit_rms[:, zi]]
                for zi in range(int(z_levels.size))
            ],
            "line_points_sample": [
                as_json_point(line_points[i, zi, :])
                for zi in range(0, int(z_levels.size), max(1, int(z_levels.size) // 80))
                for i in range(0, int(fit_rms.shape[0]), max(1, int(fit_rms.shape[0]) // 80))
                if np.all(np.isfinite(line_points[i, zi, :]))
            ],
            "diagnostics": diagnostics,
        },
        "face_candidates": candidates,
        "reconstructed": reconstructed,
    }


def build_viewer_html(payload: dict[str, object], output_html: Path) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    plotly_src = relative_plotly_src(output_html)
    default_plane_limit = max(1, len(payload.get("face_candidates", [])))
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RMS face-region reconstruction</title>
  <script src="{plotly_src}"></script>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: Arial, sans-serif; color: #192230; background: #f7f8fa; }}
    #bar {{ height: 58px; box-sizing: border-box; display: flex; align-items: center; gap: 14px; padding: 8px 12px; border-bottom: 1px solid #d8dee8; background: #ffffff; }}
    #wrap {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-template-rows: repeat(2, minmax(0, 1fr)); width: 100vw; height: calc(100vh - 58px); }}
    .plot3d {{ width: 100%; min-width: 0; min-height: 0; }}
    #plotPlanes, #plotOverlay {{ border-bottom: 1px solid #d8dee8; }}
    #plotPlanes, #plotInitial {{ border-right: 1px solid #d8dee8; }}
    select, input {{ height: 30px; border: 1px solid #b8c1ce; border-radius: 4px; background: #fff; color: #192230; }}
    label {{ font-size: 12px; color: #526071; display: flex; align-items: center; gap: 6px; }}
    .title {{ font-weight: 700; }}
    .muted {{ color: #526071; font-size: 12px; }}
  </style>
</head>
<body>
  <div id="bar">
    <div class="title" id="title"></div>
    <label><input id="showPlanesInput" type="checkbox" checked> Найденные плоскости</label>
    <label><input id="showActiveFacesInput" type="checkbox" checked> Активные грани</label>
    <label>Показать плоскостей <input id="limitInput" type="number" min="1" max="400" step="1" value="{default_plane_limit}"></label>
    <span class="muted" id="status"></span>
  </div>
  <div id="wrap">
    <div id="plotPlanes" class="plot3d"></div>
    <div id="plotOverlay" class="plot3d"></div>
    <div id="plotInitial" class="plot3d"></div>
    <div id="plotReconstructed" class="plot3d"></div>
  </div>
  <script>
    const DATA = {data_json};
    const vertices = DATA.viewer.vertices || [];
    const triangles = DATA.viewer.triangles || [];
    const candidates = DATA.face_candidates || [];
    const reconstructed = DATA.reconstructed || {{}};
    const limitInput = document.getElementById('limitInput');
    const showPlanesInput = document.getElementById('showPlanesInput');
    const showActiveFacesInput = document.getElementById('showActiveFacesInput');
    const status = document.getElementById('status');
    const syncedPlots = ['plotPlanes', 'plotOverlay', 'plotInitial', 'plotReconstructed'];
    let syncingCamera = false;

    function colorFor(i, alpha = 1) {{
      const h = (i * 137.508) % 360;
      return `hsla(${{h.toFixed(1)}}, 72%, 45%, ${{alpha}})`;
    }}

    function fmt(v, d = 5) {{
      return Number.isFinite(Number(v)) ? Number(v).toFixed(d) : 'n/a';
    }}

    function initialModelTraces(opacity = 0.18) {{
      if (!triangles.length) return [];
      return [{{
        type: 'mesh3d',
        x: vertices.map(p => p[0]),
        y: vertices.map(p => p[1]),
        z: vertices.map(p => p[2]),
        i: triangles.map(t => t[0]),
        j: triangles.map(t => t[1]),
        k: triangles.map(t => t[2]),
        color: '#8b99aa',
        opacity,
        flatshading: true,
        hoverinfo: 'skip',
        name: 'InitialModel',
        showscale: false,
      }}];
    }}

    function initialWireframeTraces(color = '#111827', width = 2) {{
      const faces = DATA.viewer.faces || [];
      const traces = [];
      for (let fi = 0; fi < faces.length; fi++) {{
        const face = faces[fi];
        if (!Array.isArray(face) || face.length < 2) continue;
        const closed = face.concat([face[0]]);
        traces.push({{
          type: 'scatter3d',
          mode: 'lines',
          x: closed.map(i => vertices[i]?.[0]),
          y: closed.map(i => vertices[i]?.[1]),
          z: closed.map(i => vertices[i]?.[2]),
          line: {{ color, width }},
          hoverinfo: 'skip',
          showlegend: false,
        }});
      }}
      return traces;
    }}

    function polygonTrace(candidate, idx) {{
      const hull = candidate.hull || [];
      if (hull.length < 3) return null;
      return {{
        type: 'mesh3d',
        x: hull.map(p => p[0]),
        y: hull.map(p => p[1]),
        z: hull.map(p => p[2]),
        i: Array.from({{length: hull.length - 2}}, (_, k) => 0),
        j: Array.from({{length: hull.length - 2}}, (_, k) => k + 1),
        k: Array.from({{length: hull.length - 2}}, (_, k) => k + 2),
        color: colorFor(idx, 0.55),
        opacity: 0.46,
        flatshading: true,
        name: `face candidate ${{candidate.track_id}}`,
        text: `track=${{candidate.track_id}}<br>levels=${{candidate.levels}}<br>z=${{fmt(candidate.z_min)}}..${{fmt(candidate.z_max)}}<br>plane_rms=${{fmt(candidate.plane_rms)}}<br>fit_points=${{candidate.fit_points}}`,
        hovertemplate: '%{{text}}<extra></extra>',
        showscale: false,
      }};
    }}

    function lineTrace(candidate, idx) {{
      const hull = candidate.hull || [];
      if (hull.length < 3) return null;
      const closed = hull.concat([hull[0]]);
      return {{
        type: 'scatter3d',
        mode: 'lines',
        x: closed.map(p => p[0]),
        y: closed.map(p => p[1]),
        z: closed.map(p => p[2]),
        line: {{ color: colorFor(idx, 0.95), width: 4 }},
        hoverinfo: 'skip',
        showlegend: false,
      }};
    }}

    function reconstructedTraces(opacity = 0.34) {{
      const verts = reconstructed.vertices || [];
      const faces = reconstructed.faces || [];
      const traces = [];
      for (let fi = 0; fi < faces.length; fi++) {{
        const face = faces[fi];
        if (!Array.isArray(face) || face.length < 3) continue;
        traces.push({{
          type: 'mesh3d',
          x: face.map(i => verts[i]?.[0]),
          y: face.map(i => verts[i]?.[1]),
          z: face.map(i => verts[i]?.[2]),
          i: Array.from({{length: face.length - 2}}, () => 0),
          j: Array.from({{length: face.length - 2}}, (_, k) => k + 1),
          k: Array.from({{length: face.length - 2}}, (_, k) => k + 2),
          color: '#2f80ed',
          opacity,
          flatshading: true,
          name: 'halfspace intersection',
          text: `reconstructed face ${{fi}}<br>vertices=${{face.length}}`,
          hovertemplate: '%{{text}}<extra></extra>',
          showscale: false,
        }});
        const closed = face.concat([face[0]]);
        traces.push({{
          type: 'scatter3d',
          mode: 'lines',
          x: closed.map(i => verts[i]?.[0]),
          y: closed.map(i => verts[i]?.[1]),
          z: closed.map(i => verts[i]?.[2]),
          line: {{ color: '#1f5fbf', width: 3 }},
          hoverinfo: 'skip',
          showlegend: false,
        }});
      }}
      return traces;
    }}

    function candidateTraces(limit) {{
      const traces = [];
      candidates.slice(0, limit).forEach((candidate, idx) => {{
        const poly = polygonTrace(candidate, idx);
        const line = lineTrace(candidate, idx);
        if (poly) traces.push(poly);
        if (line) traces.push(line);
      }});
      return traces;
    }}

    function sceneLayout(title) {{
      return {{
        margin: {{ l: 0, r: 0, b: 0, t: 28 }},
        title: {{ text: title, font: {{ size: 13 }} }},
        scene: {{
          xaxis: {{ title: 'X' }},
          yaxis: {{ title: 'Y' }},
          zaxis: {{ title: 'Z' }},
          aspectmode: 'data',
        }},
        showlegend: false,
      }};
    }}

    async function render() {{
      const limit = Math.max(1, Number(limitInput.value) || 1);
      const showPlanes = Boolean(showPlanesInput.checked);
      const showActiveFaces = Boolean(showActiveFacesInput.checked);
      const initialTraces = initialModelTraces(0.28);
      const reconstructionTraces = reconstructedTraces(0.42);
      const planesTraces = (showPlanes ? candidateTraces(limit) : []).concat(showActiveFaces ? reconstructedTraces(0.16) : []);
      const overlayTraces = initialModelTraces(0.42).concat(initialWireframeTraces('#111827', 3)).concat(reconstructedTraces(0.34));
      const rv = (reconstructed.vertices || []).length;
      const rf = (reconstructed.faces || []).length;
      status.textContent = `planes=${{showPlanes ? Math.min(limit, candidates.length) : 0}}/${{candidates.length}}, intersection=${{rv}}v/${{rf}}f`;
      Plotly.react('plotPlanes', planesTraces, sceneLayout('Найденные плоскости + активные грани пересечения'), {{ responsive: true, displaylogo: false }});
      Plotly.react('plotInitial', initialTraces, sceneLayout('Исходная модель'), {{ responsive: true, displaylogo: false }});
      Plotly.react('plotReconstructed', reconstructionTraces, sceneLayout('Полученная модель'), {{ responsive: true, displaylogo: false }});
      Plotly.react('plotOverlay', overlayTraces, sceneLayout('Наложение'), {{ responsive: true, displaylogo: false }});
      window.setTimeout(bindCameraSync, 250);
    }}

    function bindCameraSync() {{
      for (const id of syncedPlots) {{
        const el = document.getElementById(id);
        if (!el || el._cameraSyncBound) continue;
        el._cameraSyncBound = true;
        el.addEventListener('plotly_relayout', ev => {{
          const detail = ev && ev.detail ? ev.detail : {{}};
          const camera = detail['scene.camera'];
          if (!camera || syncingCamera) return;
          syncingCamera = true;
          Promise.all(
            syncedPlots
              .filter(otherId => otherId !== id)
              .map(otherId => Plotly.relayout(otherId, {{ 'scene.camera': camera }}))
          ).finally(() => {{
            syncingCamera = false;
          }});
        }});
      }}
    }}

    document.getElementById('title').textContent = `${{DATA.model}}: RMS-регионы -> кандидаты граней`;
    limitInput.addEventListener('change', () => void render());
    showPlanesInput.addEventListener('change', () => void render());
    showActiveFacesInput.addEventListener('change', () => void render());
    void render();
    window.setTimeout(bindCameraSync, 1000);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental face reconstruction from thresholded RMS peak/low regions."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model", type=str, default="round")
    parser.add_argument("--max-contours", type=int, default=None)
    parser.add_argument("--z-step", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--windows",
        type=str,
        default="4,6,8,10,12,16,20,24,28",
        help="Comma-separated window sizes. Multiple windows collect face candidates at several scales.",
    )
    parser.add_argument("--peak-threshold", type=float, default=0.00055)
    parser.add_argument("--low-threshold", type=float, default=0.0002)
    parser.add_argument("--peak-quantile", type=float, default=-1.0)
    parser.add_argument("--low-quantile", type=float, default=-1.0)
    parser.add_argument("--min-peak-width", type=int, default=1)
    parser.add_argument("--max-peak-width", type=int, default=40)
    parser.add_argument("--max-low-distance", type=float, default=35.0)
    parser.add_argument("--adaptive-window-smooth", type=float, default=0.005)
    parser.add_argument("--adaptive-window-max-relative-rms", type=float, default=2.0)
    parser.add_argument("--adaptive-window-max-abs-rms-gap", type=float, default=0.00035)
    parser.add_argument("--adaptive-window-large-bonus", type=float, default=0.35)
    parser.add_argument("--max-center-jump", type=float, default=8.0)
    parser.add_argument("--max-z-gap", type=int, default=2)
    parser.add_argument("--min-track-levels", type=int, default=5)
    parser.add_argument("--min-fit-points", type=int, default=18)
    parser.add_argument("--low-region-samples", type=int, default=3)
    parser.add_argument("--max-plane-rms", type=float, default=0.08)
    parser.add_argument(
        "--orientation-mode",
        choices=("observed-points", "model-vertices", "inside-point"),
        default="observed-points",
    )
    parser.add_argument("--model-support-tol", type=float, default=0.03)
    parser.add_argument(
        "--model-support-max-outside-count",
        type=int,
        default=-1,
        help="Diagnostic filter for orientation-mode=model-vertices. Negative disables.",
    )
    parser.add_argument("--max-candidates", type=int, default=180)
    parser.add_argument("--max-active-face-hull-area-ratio", type=float, default=20.0)
    parser.add_argument("--max-active-face-extra-area", type=float, default=1.0)
    parser.add_argument("--active-face-filter-iterations", type=int, default=4)
    parser.add_argument("--intersection-inside-tol", type=float, default=0.03)
    parser.add_argument("--intersection-vertex-tol", type=float, default=0.02)
    parser.add_argument("--output-html", type=Path, default=Path("output/round_rms_face_regions.html"))
    parser.add_argument("--output-json", type=Path, default=Path("output/round_rms_face_regions.json"))
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    model_name = discover_single_model(args.data_root, args.model.strip() or None)
    model_path = args.data_root / model_name / "InitialModel"
    shadow_dir = args.data_root / model_name / "shadow"
    model = parse_initial_model(model_path)
    vertices = model.vertices
    contour_files = sorted_contour_files(shadow_dir, "merged-cont*", args.max_contours)
    contours = [parse_merged_contour(p) for p in contour_files]
    top_axis_point = top_points_vertical_axis_point(vertices)
    half_contours = build_half_contours(
        contours=contours,
        split_line_point=np.array([top_axis_point[0], top_axis_point[1], top_axis_point[2]], dtype=float),
        split_line_dir=np.array([0.0, 0.0, 1.0], dtype=float),
    )

    contour_points = np.vstack([c.points for c in contours])
    z_min = float(np.min(contour_points[:, 2]))
    z_max = float(np.max(contour_points[:, 2]))
    bounds_min = np.min(contour_points, axis=0)
    bounds_max = np.max(contour_points, axis=0)
    bounds_margin = 0.1 * max(float(np.linalg.norm(bounds_max - bounds_min)), EPS)
    point_bounds_min = bounds_min - bounds_margin
    point_bounds_max = bounds_max + bounds_margin
    z_step = max(0.0005, float(args.z_step))
    z_levels = np.arange(z_min, z_max + 0.5 * z_step, z_step, dtype=float)

    windows = parse_windows(str(args.windows), int(args.window))
    adaptive_summary: dict[str, object] = {}
    if len(windows) == 1:
        line_points, fit_rms = compute_line_points(
            half_contours,
            z_levels,
            int(windows[0]),
            use_progress=not args.no_progress,
        )
        selected_window_indices = np.zeros(int(z_levels.size), dtype=int)
        adaptive_summary = {
            "adaptive_window_min_selected": int(windows[0]),
            "adaptive_window_max_selected": int(windows[0]),
            "adaptive_window_median_selected": int(windows[0]),
            "adaptive_window_changes": 0,
            "adaptive_window_sequence": [int(windows[0]) for _ in range(int(z_levels.size))],
        }
    else:
        line_points_by_window, fit_rms_by_window, _ = compute_line_points_multi_window(
            half_contours,
            z_levels,
            windows,
            use_progress=not args.no_progress,
        )
        selected_window_indices, adaptive_summary = adaptive_windows_from_rms(
            fit_rms_by_window,
            windows,
            smooth_weight=float(args.adaptive_window_smooth),
            max_relative_rms=float(args.adaptive_window_max_relative_rms),
            max_abs_rms_gap=float(args.adaptive_window_max_abs_rms_gap),
            large_window_bonus=float(args.adaptive_window_large_bonus),
        )
        line_points, fit_rms = apply_selected_windows(
            line_points_by_window,
            fit_rms_by_window,
            selected_window_indices,
        )

    per_level, diagnostics = detect_peak_observations(
        z_levels,
        fit_rms,
        peak_threshold_abs=float(args.peak_threshold),
        low_threshold_abs=float(args.low_threshold),
        peak_quantile=float(args.peak_quantile),
        low_quantile=float(args.low_quantile),
        min_peak_width=int(args.min_peak_width),
        max_peak_width=int(args.max_peak_width),
        max_low_distance=float(args.max_low_distance),
    )
    tracks = track_peak_observations(
        per_level,
        n_half=int(fit_rms.shape[0]),
        max_center_jump=float(args.max_center_jump),
        max_z_gap=int(args.max_z_gap),
    )
    finite_mask = (
        np.all(np.isfinite(line_points), axis=2)
        & np.all(line_points >= point_bounds_min[None, None, :], axis=2)
        & np.all(line_points <= point_bounds_max[None, None, :], axis=2)
    )
    finite_points = line_points[finite_mask]
    inside_point = np.mean(finite_points, axis=0) if finite_points.size else np.mean(vertices, axis=0)
    if args.orientation_mode == "model-vertices":
        orientation_points = vertices
        support_points = vertices
    elif args.orientation_mode == "inside-point":
        orientation_points = np.zeros((0, 3), dtype=float)
        support_points = np.zeros((0, 3), dtype=float)
    else:
        orientation_step = max(1, int(finite_points.shape[0]) // 25000) if finite_points.size else 1
        orientation_points = finite_points[::orientation_step] if finite_points.size else finite_points
        support_points = np.zeros((0, 3), dtype=float)
    all_candidates = build_face_candidates(
        tracks,
        line_points,
        fit_rms,
        min_track_levels=int(args.min_track_levels),
        min_fit_points=int(args.min_fit_points),
        max_plane_rms=float(args.max_plane_rms),
        inside_point=inside_point,
        orientation_points=orientation_points,
        support_points=support_points,
        support_tol=float(args.model_support_tol),
        max_support_outside_count=int(args.model_support_max_outside_count),
        low_region_samples=int(args.low_region_samples),
        bounds_min=point_bounds_min,
        bounds_max=point_bounds_max,
    )
    max_candidates = max(1, int(args.max_candidates))
    candidates = all_candidates[:max_candidates]
    candidates, reconstructed, active_face_filter_summary = filter_candidates_by_active_face_size(
        candidates,
        inside_tol=float(args.intersection_inside_tol),
        vertex_tol=float(args.intersection_vertex_tol),
        max_area_ratio=float(args.max_active_face_hull_area_ratio),
        max_extra_area=float(args.max_active_face_extra_area),
        iterations=int(args.active_face_filter_iterations),
    )

    parameters = {
        "data_root": str(args.data_root),
        "max_contours": args.max_contours,
        "z_step": z_step,
        "window": int(args.window),
        "windows": [int(w) for w in windows],
        "window_mode": "z-adaptive" if len(windows) > 1 else "single",
        "display_window": None,
        **adaptive_summary,
        "peak_threshold": float(args.peak_threshold),
        "low_threshold": float(args.low_threshold),
        "peak_quantile": float(args.peak_quantile),
        "low_quantile": float(args.low_quantile),
        "min_peak_width": int(args.min_peak_width),
        "max_peak_width": int(args.max_peak_width),
        "max_low_distance": float(args.max_low_distance),
        "adaptive_window_smooth": float(args.adaptive_window_smooth),
        "adaptive_window_max_relative_rms": float(args.adaptive_window_max_relative_rms),
        "adaptive_window_max_abs_rms_gap": float(args.adaptive_window_max_abs_rms_gap),
        "adaptive_window_large_bonus": float(args.adaptive_window_large_bonus),
        "point_bounds_min": as_json_point(point_bounds_min),
        "point_bounds_max": as_json_point(point_bounds_max),
        "max_center_jump": float(args.max_center_jump),
        "max_z_gap": int(args.max_z_gap),
        "min_track_levels": int(args.min_track_levels),
        "min_fit_points": int(args.min_fit_points),
        "low_region_samples": int(args.low_region_samples),
        "max_plane_rms": float(args.max_plane_rms),
        "orientation_mode": str(args.orientation_mode),
        "model_support_tol": float(args.model_support_tol),
        "model_support_max_outside_count": int(args.model_support_max_outside_count),
        "max_candidates": int(args.max_candidates),
        "post_filter_candidates": int(len(candidates)),
        "max_active_face_hull_area_ratio": float(args.max_active_face_hull_area_ratio),
        "max_active_face_extra_area": float(args.max_active_face_extra_area),
        "active_face_filter_iterations": int(args.active_face_filter_iterations),
        **active_face_filter_summary,
        "raw_candidates_before_merge": int(len(all_candidates)),
        "candidates_after_merge": None,
        "inside_point": as_json_point(inside_point),
        "orientation_point_count": int(orientation_points.shape[0]),
        "intersection_inside_tol": float(args.intersection_inside_tol),
        "intersection_vertex_tol": float(args.intersection_vertex_tol),
    }
    payload = build_payload(
        model_name=model_name,
        vertices=vertices,
        faces=model.faces,
        z_levels=z_levels,
        fit_rms=fit_rms,
        line_points=line_points,
        tracks=tracks,
        diagnostics=diagnostics,
        candidates=candidates,
        reconstructed=reconstructed,
        parameters=parameters,
    )

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(build_viewer_html(payload, args.output_html), encoding="utf-8")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "model={model} z_levels={z_levels} peak_observations={peaks} tracks={tracks} candidates={candidates}".format(
            model=model_name,
            z_levels=int(z_levels.size),
            peaks=sum(int(item.get("peaks", 0) or 0) for item in diagnostics),
            tracks=len(tracks),
            candidates=len(candidates),
        )
    )
    print(
        "intersection=vertices:{vertices} faces:{faces} raw_intersections:{raw}".format(
            vertices=len(reconstructed.get("vertices", [])),
            faces=len(reconstructed.get("faces", [])),
            raw=int(reconstructed.get("raw_intersections", 0)),
        )
    )
    print(f"html={args.output_html}")
    if args.output_json is not None:
        print(f"json={args.output_json}")


if __name__ == "__main__":
    main()
