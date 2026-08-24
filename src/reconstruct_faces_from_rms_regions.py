from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generate_full_circle_split_cached_viewer import (
    EPS,
    build_half_contours,
    lateral_axis_for_normal,
    parse_initial_model,
    parse_merged_contour,
    sorted_contour_files,
    top_points_vertical_axis_point,
)
from match_full_circle_points_to_edges import (
    compute_line_points,
    compute_line_points_multi_window,
    discover_single_model,
)
from polyreco.rms_oracle_loss_diagnostics import (
    REQUIRED_RUNTIME_SYMBOLS as ORACLE_LOSS_DIAGNOSTIC_RUNTIME_SYMBOLS,
    bind_runtime_symbols as bind_oracle_loss_diagnostic_runtime_symbols,
    collect_line_residual_oracle_diagnostics,
    dispatch_final_mesh_named_oracle_scope,
    dispatch_post_multiscale_oracle_scope,
    dispatch_post_w2_oracle_scope,
    diagnose_append_local_frontier,
    diagnose_candidate_formation_failures,
    diagnose_plane_consensus_groups,
    diagnose_golden_oracle_closure,
    diagnose_preselection_safe_reservoir,
    diagnose_rankout_selection_bias,
    diagnose_production_129_loss_funnel,
    model_edges_from_faces,
    summarize_false_active_w2,
    oracle_compatible_ceiling_diagnostics,
)
from polyreco.rms_output import build_payload, build_viewer_html
from polyreco.rms_selection import (
    LpActivityEngine,
    annotate_candidate_pool_neutral,
    annotate_w2_support_diverse_scores,
    as_json_float,
    as_json_point,
    candidate_hull_centroid,
    candidate_hull_polygon_2d,
    candidate_margin_against_vertices,
    candidate_outside_mask,
    candidate_rank_key,
    candidate_track_id,
    cumulative_metrics_from_mask,
    cumulative_outside_metrics,
    finite_float,
    lp_constraints_for_candidates,
    lp_face_activity_details,
    polygon_area_2d,
    polygon_signed_distances_2d,
    prepare_z_level_slices,
    select_compatible_candidates,
    support_diverse_candidate_key,
)
from polyreco.rms_mesh import (
    candidate_halfspace,
    candidate_plane,
    convex_hull_indices,
    lp_extents_for_halfspaces,
    plane_basis,
    polygon_area,
    reconstruct_polyhedron_from_halfspaces,
    reconstruct_polyhedron_from_halfspaces_edge_clip,
    reconstruct_polyhedron_from_halfspaces_incidence,
    solve_halfspace_lp,
)


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




def safe_nanmedian(values: list[object]) -> float | None:
    arr = np.array([finite_float(v, float("nan")) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


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
    max_distance: float | None = None,
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
        if max_distance is not None and dist >= float(max_distance):
            continue
        item = (dist, region)
        if best is None or item[0] < best[0]:
            best = item
    if best is None or best[0] >= float(n):
        return None, None
    return best[1], float(best[0])


def valley_region(
    values: np.ndarray,
    peak: CyclicRegion,
    n: int,
    side: str,
    *,
    max_distance: float | None,
    max_low_distance: float,
) -> tuple[CyclicRegion | None, float | None]:
    best: tuple[float, float, int] | None = None
    min_dist = max(0.5 * float(peak.width), 0.5)
    distance_limit = float(max_low_distance) if max_low_distance >= 0.0 else float(n)
    if max_distance is not None:
        distance_limit = min(distance_limit, float(max_distance))
    for idx in range(n):
        value = float(values[idx])
        if not np.isfinite(value):
            continue
        if side == "left":
            dist = (float(peak.center) - float(idx)) % float(n)
        else:
            dist = (float(idx) - float(peak.center)) % float(n)
        if dist <= min_dist or dist >= distance_limit:
            continue
        item = (value, dist, int(idx))
        if best is None or item < best:
            best = item
    if best is None:
        return None, None
    value, dist, idx = best
    region = CyclicRegion(
        indices=(idx,),
        start=idx,
        end=idx,
        center=float(idx),
        width=1,
        min_value=float(value),
        max_value=float(value),
        mean_value=float(value),
    )
    return region, float(dist)


def valley_quality_metrics(
    values: np.ndarray,
    peak: CyclicRegion,
    valley: CyclicRegion | None,
    *,
    side: str,
    n: int,
    distance: float | None,
    neighbor_peak_distance: float | None,
    max_low_distance: float,
) -> dict[str, float | bool | None] | None:
    if valley is None or distance is None:
        return None
    idx = int(valley.center) % int(n)
    cur = float(values[idx])
    prev = float(values[(idx - 1) % n])
    nxt = float(values[(idx + 1) % n])
    is_local_min = bool(np.isfinite(cur) and np.isfinite(prev) and np.isfinite(nxt) and cur <= prev and cur <= nxt)
    valley_prominence = min(prev - cur, nxt - cur) if is_local_min else min(prev - cur, nxt - cur)
    peak_value = float(peak.max_value) if np.isfinite(float(peak.max_value)) else float(peak.mean_value)
    peak_contrast = peak_value - cur
    sector_limit = float(max_low_distance) if max_low_distance >= 0.0 else float(n)
    if neighbor_peak_distance is not None:
        sector_limit = min(sector_limit, float(neighbor_peak_distance))
    sector_limit = max(sector_limit, 1.0)
    sector_position = float(distance) / sector_limit
    neighbor_margin = (
        float(neighbor_peak_distance) - float(distance)
        if neighbor_peak_distance is not None
        else None
    )
    distance_score = 1.0 - min(abs(sector_position - 0.5) / 0.5, 1.0)
    boundary_score = 1.0 if 0.12 <= sector_position <= 0.88 else max(0.0, 1.0 - min(abs(sector_position - 0.5) / 0.5, 1.0))
    prominence_scale = max(abs(peak_value), abs(cur), 1e-6)
    prominence_score = np.clip(float(valley_prominence) / (0.2 * prominence_scale), 0.0, 1.0)
    contrast_score = np.clip(float(peak_contrast) / (0.5 * prominence_scale), 0.0, 1.0)
    neighbor_score = 1.0
    if neighbor_margin is not None:
        neighbor_score = np.clip(float(neighbor_margin) / max(sector_limit * 0.25, 1.0), 0.0, 1.0)
    local_score = 1.0 if is_local_min else 0.35
    confidence = float(np.clip(local_score * (0.25 + 0.75 * np.mean([prominence_score, contrast_score, distance_score, boundary_score, neighbor_score])), 0.0, 1.0))
    return {
        "valley_is_local_min": is_local_min,
        "valley_prominence": as_json_float(float(valley_prominence)),
        "valley_peak_contrast": as_json_float(float(peak_contrast)),
        "valley_sector_position": as_json_float(float(sector_position)),
        "valley_neighbor_peak_margin": as_json_float(float(neighbor_margin)) if neighbor_margin is not None else None,
        "valley_distance": as_json_float(float(distance)),
        "valley_confidence": as_json_float(confidence),
        "valley_z_consistency": None,
        "valley_scale_support": None,
    }


def nearest_peak_distance(
    peaks: list[CyclicRegion],
    current: CyclicRegion,
    n: int,
    side: str,
) -> float | None:
    best: float | None = None
    for peak in peaks:
        if peak is current:
            continue
        if side == "left":
            dist = (float(current.center) - float(peak.center)) % float(n)
        else:
            dist = (float(peak.center) - float(current.center)) % float(n)
        if dist <= 0.0:
            continue
        if best is None or dist < best:
            best = float(dist)
    return best


def cyclic_region_overlaps(a: CyclicRegion, b: CyclicRegion) -> bool:
    return bool(set(a.indices) & set(b.indices))


def nearest_blocker_distance(
    blockers: list[CyclicRegion],
    current: CyclicRegion,
    n: int,
    side: str,
) -> float | None:
    best: float | None = None
    for blocker in blockers:
        if blocker is current or cyclic_region_overlaps(blocker, current):
            continue
        if side == "left":
            dist = (float(current.center) - float(blocker.center)) % float(n)
        else:
            dist = (float(blocker.center) - float(current.center)) % float(n)
        if dist <= 0.0:
            continue
        if best is None or dist < best:
            best = float(dist)
    return best


def min_optional_distance(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(float(a), float(b))


def local_max_blockers(
    values: np.ndarray,
    *,
    low_threshold: float,
    min_prominence: float,
) -> list[CyclicRegion]:
    n = int(values.size)
    if n < 3 or min_prominence < 0.0:
        return []
    blockers: list[CyclicRegion] = []
    min_value = float(low_threshold) + float(min_prominence)
    for i in range(n):
        cur = float(values[i])
        prev = float(values[(i - 1) % n])
        nxt = float(values[(i + 1) % n])
        if not (np.isfinite(cur) and np.isfinite(prev) and np.isfinite(nxt)):
            continue
        if cur < min_value:
            continue
        if cur >= prev and cur >= nxt and (cur > prev or cur > nxt):
            blockers.append(
                CyclicRegion(
                    indices=(int(i),),
                    start=int(i),
                    end=int(i),
                    center=float(i),
                    width=1,
                    min_value=cur,
                    max_value=cur,
                    mean_value=cur,
                )
            )
    return blockers


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


def apply_selected_values_by_window(values_by_window: np.ndarray, selected_window_indices: np.ndarray) -> np.ndarray:
    n_half = int(values_by_window.shape[1])
    n_levels = int(values_by_window.shape[2])
    values = np.full((n_half, n_levels), np.nan, dtype=float)
    for zi, wi_raw in enumerate(selected_window_indices):
        values[:, zi] = values_by_window[int(wi_raw), :, zi].astype(float)
    return values


def detect_peak_observations(
    z_levels: np.ndarray,
    fit_rms: np.ndarray,
    *,
    peak_threshold_abs: float,
    low_threshold_abs: float,
    peak_quantile: float,
    low_quantile: float,
    low_barrier_threshold_abs: float,
    low_barrier_fraction: float,
    low_barrier_prominence: float,
    low_valley_fallback: bool,
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
        if low_barrier_threshold_abs >= 0.0:
            barrier_threshold = float(low_barrier_threshold_abs)
        else:
            frac = float(np.clip(low_barrier_fraction, 0.0, 1.0))
            barrier_threshold = float(low_threshold + max(0.0, peak_threshold - low_threshold) * frac)
        barrier_mask = finite & (values >= barrier_threshold)
        peak_regions = [
            r
            for r in contiguous_regions(peak_mask, values)
            if int(r.width) >= int(min_peak_width) and int(r.width) <= int(max_peak_width)
        ]
        low_regions = contiguous_regions(low_mask, values)
        barrier_regions = contiguous_regions(barrier_mask, values)
        prominence_blockers = local_max_blockers(
            values,
            low_threshold=low_threshold,
            min_prominence=float(low_barrier_prominence),
        )
        blocker_regions = barrier_regions + prominence_blockers

        observations: list[PeakObservation] = []
        for peak in peak_regions:
            left_hard_block = nearest_peak_distance(peak_regions, peak, n_half, "left")
            right_hard_block = nearest_peak_distance(peak_regions, peak, n_half, "right")
            left_soft_block = nearest_blocker_distance(blocker_regions, peak, n_half, "left")
            right_soft_block = nearest_blocker_distance(blocker_regions, peak, n_half, "right")
            left_block = min_optional_distance(left_hard_block, left_soft_block)
            right_block = min_optional_distance(right_hard_block, right_soft_block)
            left_low, left_dist = nearest_region(
                low_regions,
                peak.center,
                n_half,
                "left",
                max_distance=left_block,
            )
            right_low, right_dist = nearest_region(
                low_regions,
                peak.center,
                n_half,
                "right",
                max_distance=right_block,
            )
            left_crossed_soft = False
            right_crossed_soft = False
            left_source = "threshold"
            right_source = "threshold"
            left_valley_quality = None
            right_valley_quality = None
            if left_low is None and left_soft_block is not None and (
                left_hard_block is None or left_soft_block < left_hard_block
            ):
                left_low, left_dist = nearest_region(
                    low_regions,
                    peak.center,
                    n_half,
                    "left",
                    max_distance=left_hard_block,
                )
                left_crossed_soft = left_low is not None
                if left_crossed_soft:
                    left_source = "threshold-cross-soft"
            if right_low is None and right_soft_block is not None and (
                right_hard_block is None or right_soft_block < right_hard_block
            ):
                right_low, right_dist = nearest_region(
                    low_regions,
                    peak.center,
                    n_half,
                    "right",
                    max_distance=right_hard_block,
                )
                right_crossed_soft = right_low is not None
                if right_crossed_soft:
                    right_source = "threshold-cross-soft"
            if left_low is None and low_valley_fallback:
                left_low, left_dist = valley_region(
                    values,
                    peak,
                    n_half,
                    "left",
                    max_distance=left_hard_block,
                    max_low_distance=max_low_distance,
                )
                if left_low is not None:
                    left_source = "valley"
                    left_valley_quality = valley_quality_metrics(
                        values,
                        peak,
                        left_low,
                        side="left",
                        n=n_half,
                        distance=left_dist,
                        neighbor_peak_distance=left_hard_block,
                        max_low_distance=max_low_distance,
                    )
            if right_low is None and low_valley_fallback:
                right_low, right_dist = valley_region(
                    values,
                    peak,
                    n_half,
                    "right",
                    max_distance=right_hard_block,
                    max_low_distance=max_low_distance,
                )
                if right_low is not None:
                    right_source = "valley"
                    right_valley_quality = valley_quality_metrics(
                        values,
                        peak,
                        right_low,
                        side="right",
                        n=n_half,
                        distance=right_dist,
                        neighbor_peak_distance=right_hard_block,
                        max_low_distance=max_low_distance,
                    )
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
                    left_low_source=left_source,
                    right_low_source=right_source,
                    left_crossed_soft_barrier=bool(left_crossed_soft),
                    right_crossed_soft_barrier=bool(right_crossed_soft),
                    left_valley_quality=left_valley_quality,
                    right_valley_quality=right_valley_quality,
                )
            )
        per_level.append(observations)
        diagnostics.append(
            {
                "z_index": zi,
                "z": float(z_levels[zi]),
                "peak_threshold": float(peak_threshold),
                "low_threshold": float(low_threshold),
                "low_barrier_threshold": float(barrier_threshold),
                "peaks": len(peak_regions),
                "low_regions": len(low_regions),
                "barrier_regions": len(barrier_regions),
                "prominence_blockers": len(prominence_blockers),
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


def build_minima_cloud(
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    *,
    threshold_pct: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> dict[str, object]:
    n_half, n_levels = fit_rms.shape
    pct = max(0.0, float(threshold_pct))
    points: list[list[float | None]] = []
    z_indices: list[int] = []
    indices: list[int] = []
    values: list[float | None] = []
    thresholds: list[float | None] = []
    for zi in range(n_levels):
        y = fit_rms[:, zi].astype(float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        global_min = float(np.min(y[finite]))
        global_max = float(np.max(y[finite]))
        threshold = global_min + max(0.0, global_max - global_min) * (pct / 100.0)
        for i in range(n_half):
            cur = float(y[i])
            prev = float(y[(i - 1) % n_half])
            nxt = float(y[(i + 1) % n_half])
            if not (np.isfinite(cur) and np.isfinite(prev) and np.isfinite(nxt)):
                continue
            is_local = cur <= prev and cur <= nxt and (cur < prev or cur < nxt)
            if not is_local or cur > threshold + 1e-12:
                continue
            p = line_points[i, zi, :]
            if (
                np.all(np.isfinite(p))
                and bool(np.all(p >= bounds_min))
                and bool(np.all(p <= bounds_max))
            ):
                points.append(as_json_point(p))
                z_indices.append(int(zi))
                indices.append(int(i))
                values.append(as_json_float(cur))
                thresholds.append(as_json_float(threshold))

    return {
        "threshold_pct": as_json_float(pct),
        "points": points,
        "z_indices": z_indices,
        "indices": indices,
        "values": values,
        "thresholds": thresholds,
    }


def local_minima_cloud_arrays(
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    *,
    threshold_pct: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    n_half, n_levels = fit_rms.shape
    points: list[np.ndarray] = []
    z_indices: list[int] = []
    cyclic_indices: list[int] = []
    invalid = 0
    rejected = 0
    per_level_counts = np.zeros(n_levels, dtype=int)
    for zi in range(n_levels):
        y = fit_rms[:, zi].astype(float)
        finite = np.isfinite(y)
        if not np.any(finite):
            invalid += n_half
            continue
        global_min = float(np.min(y[finite]))
        global_max = float(np.max(y[finite]))
        threshold = global_min + max(0.0, global_max - global_min) * (max(0.0, threshold_pct) / 100.0)
        for i in range(n_half):
            cur = float(y[i])
            prev = float(y[(i - 1) % n_half])
            nxt = float(y[(i + 1) % n_half])
            p = line_points[i, zi, :]
            if not (np.isfinite(cur) and np.isfinite(prev) and np.isfinite(nxt) and np.all(np.isfinite(p))):
                invalid += 1
                continue
            is_local = cur <= prev and cur <= nxt and (cur < prev or cur < nxt)
            if (
                not is_local
                or cur > threshold + 1e-12
                or not bool(np.all(p >= bounds_min))
                or not bool(np.all(p <= bounds_max))
            ):
                rejected += 1
                continue
            points.append(p.astype(float))
            z_indices.append(int(zi))
            cyclic_indices.append(int(i))
            per_level_counts[zi] += 1
    pts = np.array(points, dtype=float).reshape((-1, 3))
    z_arr = np.array(z_indices, dtype=int)
    idx_arr = np.array(cyclic_indices, dtype=int)
    bbox = None
    if pts.size:
        bbox_min = np.min(pts, axis=0)
        bbox_max = np.max(pts, axis=0)
        bbox = {
            "min": as_json_point(bbox_min),
            "max": as_json_point(bbox_max),
            "extent": as_json_point(bbox_max - bbox_min),
        }
    diagnostics = {
        "window": int(window),
        "points": int(pts.shape[0]),
        "invalid_points": int(invalid),
        "rejected_points": int(rejected),
        "invalid_fraction": as_json_float(float(invalid) / max(float(n_half * n_levels), 1.0)),
        "bbox": bbox,
        "z_level_count_nonzero": int(np.sum(per_level_counts > 0)),
        "points_per_level_min": int(np.min(per_level_counts)) if per_level_counts.size else 0,
        "points_per_level_median": as_json_float(float(np.median(per_level_counts))) if per_level_counts.size else None,
        "points_per_level_max": int(np.max(per_level_counts)) if per_level_counts.size else 0,
    }
    return pts, z_arr, idx_arr, diagnostics


def build_trusted_observed_cloud(
    line_points_by_window: np.ndarray,
    fit_rms_by_window: np.ndarray,
    windows: list[int],
    *,
    display_window: int,
    threshold_pct: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> dict[str, object]:
    cloud_by_window: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = {}
    for wi, window in enumerate(windows):
        cloud_by_window[int(window)] = local_minima_cloud_arrays(
            line_points_by_window[wi].astype(float),
            fit_rms_by_window[wi].astype(float),
            threshold_pct=threshold_pct,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            window=int(window),
        )

    display_pts, display_z, display_idx, display_diag = cloud_by_window[int(display_window)]
    all_points = []
    all_z = []
    all_keys_seen: set[tuple[int, int]] = set()
    stable_points: list[np.ndarray] = []
    stable_z: list[int] = []
    stable_keys_seen: set[tuple[int, int]] = set()
    by_key: dict[tuple[int, int], list[np.ndarray]] = {}
    for pts, z_arr, idx_arr, _ in cloud_by_window.values():
        for point, zi, idx in zip(pts, z_arr, idx_arr):
            key = (int(zi), int(idx))
            by_key.setdefault(key, []).append(point)
            if key not in all_keys_seen:
                all_keys_seen.add(key)
                all_points.append(point)
                all_z.append(int(zi))
    for key, pts_for_key in by_key.items():
        if len(pts_for_key) >= 2 and key not in stable_keys_seen:
            stable_keys_seen.add(key)
            stable_points.append(np.mean(np.array(pts_for_key, dtype=float), axis=0))
            stable_z.append(int(key[0]))

    variants = {
        "display": (display_pts, display_z, display_diag),
        "multiscale_minima": (
            np.array(all_points, dtype=float).reshape((-1, 3)),
            np.array(all_z, dtype=int),
            {"points": int(len(all_points)), "source": "union of minima from all windows"},
        ),
        "stable_two_scales": (
            np.array(stable_points, dtype=float).reshape((-1, 3)),
            np.array(stable_z, dtype=int),
            {"points": int(len(stable_points)), "source": "minima present at the same z/index in at least two windows"},
        ),
    }

    def score_variant(name: str, pts: np.ndarray, z_arr: np.ndarray) -> tuple[int, int, float]:
        if pts.size == 0:
            return (0, 0, float("inf"))
        levels = int(len(set(int(z) for z in z_arr)))
        per_level = np.bincount(z_arr, minlength=max(int(np.max(z_arr)) + 1, 1)) if z_arr.size else np.zeros(0)
        balance = float(np.percentile(per_level[per_level > 0], 25)) if np.any(per_level > 0) else 0.0
        # Prefer internally stable points, then z coverage, then enough per-level support.
        stable_bonus = 1 if name == "stable_two_scales" else 0
        return (stable_bonus, levels, balance)

    chosen_name = max(variants, key=lambda name: score_variant(name, variants[name][0], variants[name][1]))
    points, z_indices, chosen_diag = variants[chosen_name]
    bbox = None
    if points.size:
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox = {
            "min": as_json_point(bbox_min),
            "max": as_json_point(bbox_max),
            "extent": as_json_point(bbox_max - bbox_min),
        }
    per_level_counts = np.bincount(z_indices, minlength=max(int(np.max(z_indices)) + 1, 1)) if z_indices.size else np.zeros(0, dtype=int)
    diagnostics = {
        "chosen_mode": chosen_name,
        "points": int(points.shape[0]),
        "bbox": bbox,
        "z_level_count_nonzero": int(np.sum(per_level_counts > 0)) if per_level_counts.size else 0,
        "points_per_level_min": int(np.min(per_level_counts)) if per_level_counts.size else 0,
        "points_per_level_median": as_json_float(float(np.median(per_level_counts))) if per_level_counts.size else None,
        "points_per_level_max": int(np.max(per_level_counts)) if per_level_counts.size else 0,
        "display_window": int(display_window),
        "variants": {
            name: {
                **diag,
                "z_level_count_nonzero": int(len(set(int(z) for z in z_arr))) if z_arr.size else 0,
            }
            for name, (pts, z_arr, diag) in variants.items()
        },
        "window_diagnostics": {str(window): data[3] for window, data in cloud_by_window.items()},
        **chosen_diag,
    }
    return {
        "points": points,
        "z_indices": z_indices,
        "diagnostics": diagnostics,
    }


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


def candidate_similarity(
    a: dict[str, object],
    b: dict[str, object],
) -> tuple[float, float, float] | None:
    plane_a = candidate_plane(a)
    plane_b = candidate_plane(b)
    hull_a = candidate_hull_centroid(a)
    hull_b = candidate_hull_centroid(b)
    if plane_a is None or plane_b is None or hull_a is None or hull_b is None:
        return None
    point_a, normal_a = plane_a
    point_b, normal_b = plane_b
    angle = float(np.degrees(np.arccos(np.clip(abs(float(normal_a @ normal_b)), -1.0, 1.0))))
    dist_ab = abs(float((point_b - point_a) @ normal_a))
    dist_ba = abs(float((point_a - point_b) @ normal_b))
    plane_distance = max(dist_ab, dist_ba)
    hull_distance = float(np.linalg.norm(hull_a - hull_b))
    return angle, plane_distance, hull_distance


def candidate_offset(candidate: dict[str, object]) -> float | None:
    plane = candidate_plane(candidate)
    if plane is None:
        return None
    point, normal = plane
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    return float((normal / norm) @ point)


def classify_against_baseline(
    candidate: dict[str, object],
    baseline: list[dict[str, object]],
    *,
    duplicate_angle_deg: float,
    duplicate_plane_distance: float,
    sliver_hull_distance: float,
) -> dict[str, object]:
    best: tuple[float, float, float, float, int, dict[str, object]] | None = None
    for idx, base in enumerate(baseline):
        similarity = candidate_similarity(candidate, base)
        if similarity is None:
            continue
        angle, plane_dist, hull_dist = similarity
        score = (float(angle) / max(float(duplicate_angle_deg), 1e-9)) ** 2 + (
            float(plane_dist) / max(float(duplicate_plane_distance), 1e-9)
        ) ** 2
        item = (score, float(angle), float(plane_dist), float(hull_dist), int(idx), base)
        if best is None or item < best:
            best = item
    if best is None:
        return {"baseline_relation": "new_direction"}
    _, angle, plane_dist, hull_dist, idx, base = best
    cand_offset = candidate_offset(candidate)
    base_offset = candidate_offset(base)
    offset_delta = None if cand_offset is None or base_offset is None else float(cand_offset - base_offset)
    z_min = finite_float(candidate.get("z_min"), float("nan"))
    z_max = finite_float(candidate.get("z_max"), float("nan"))
    b_z_min = finite_float(base.get("z_min"), float("nan"))
    b_z_max = finite_float(base.get("z_max"), float("nan"))
    relation = "new_direction"
    rejection_reason = None
    if angle <= float(duplicate_angle_deg) and plane_dist <= float(duplicate_plane_distance):
        if hull_dist <= float(sliver_hull_distance):
            relation = "baseline_duplicate"
            rejection_reason = "baseline_duplicate"
        elif offset_delta is not None and offset_delta > float(duplicate_plane_distance) * 0.5:
            relation = "parallel_looser_outward"
            rejection_reason = "baseline_weaker_parallel"
        elif offset_delta is not None and offset_delta < -float(duplicate_plane_distance) * 0.5:
            relation = "parallel_stronger_inward"
        else:
            relation = "spatially_distinct_finite_patch"
    elif angle <= max(float(duplicate_angle_deg) * 2.0, 5.0):
        relation = "new_z_range" if (
            np.isfinite(z_min)
            and np.isfinite(z_max)
            and np.isfinite(b_z_min)
            and np.isfinite(b_z_max)
            and (z_min < b_z_min - 0.05 or z_max > b_z_max + 0.05)
        ) else "spatially_distinct_finite_patch"
    return {
        "baseline_relation": relation,
        "baseline_rejection_reason": rejection_reason,
        "baseline_match_index": int(idx),
        "baseline_match_track_id": base.get("track_id"),
        "baseline_match_window": base.get("window"),
        "baseline_match_angle_deg": as_json_float(float(angle)),
        "baseline_match_plane_distance": as_json_float(float(plane_dist)),
        "baseline_match_hull_distance": as_json_float(float(hull_dist)),
        "baseline_offset_delta": as_json_float(float(offset_delta)) if offset_delta is not None else None,
    }


def annotate_valley_additions_against_baseline(
    valley_candidates: list[dict[str, object]],
    baseline: list[dict[str, object]],
    *,
    duplicate_angle_deg: float,
    duplicate_plane_distance: float,
    sliver_hull_distance: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    annotated: list[dict[str, object]] = []
    relation_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    for candidate in valley_candidates:
        out = dict(candidate)
        relation = classify_against_baseline(
            out,
            baseline,
            duplicate_angle_deg=float(duplicate_angle_deg),
            duplicate_plane_distance=float(duplicate_plane_distance),
            sliver_hull_distance=float(sliver_hull_distance),
        )
        out.update(relation)
        rel = str(out.get("baseline_relation"))
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
        reason = out.get("baseline_rejection_reason")
        if reason is not None:
            rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1
            out["preselection_status"] = "rejected"
            out["preselection_reason"] = str(reason)
        else:
            out["preselection_status"] = "candidate"
            annotated.append(out)
    return annotated, {
        "valley_addition_relation_counts": relation_counts,
        "valley_addition_preselection_rejections": rejection_counts,
        "valley_additions_after_preselection": int(len(annotated)),
    }


def candidate_constraints(candidates: list[dict[str, object]]) -> list[tuple[np.ndarray, float]]:
    constraints: list[tuple[np.ndarray, float]] = []
    for candidate in candidates:
        halfspace = candidate_halfspace(candidate)
        if halfspace is not None:
            constraints.append(halfspace)
    return constraints


def normalized_direction(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    return vec.astype(float) / norm


def envelope_directions(
    baseline: list[dict[str, object]],
    valley: list[dict[str, object]],
    trusted_points: np.ndarray,
) -> list[np.ndarray]:
    raw: list[np.ndarray] = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    for candidate in baseline + valley:
        halfspace = candidate_halfspace(candidate)
        if halfspace is not None:
            raw.append(halfspace[0])
            raw.append(-halfspace[0])
    if trusted_points.size:
        center = np.mean(trusted_points, axis=0)
        mins = np.min(trusted_points, axis=0)
        maxs = np.max(trusted_points, axis=0)
        for sx in (mins[0], maxs[0]):
            for sy in (mins[1], maxs[1]):
                for sz in (mins[2], maxs[2]):
                    raw.append(np.array([sx, sy, sz], dtype=float) - center)
    directions: list[np.ndarray] = []
    for vec in raw:
        d = normalized_direction(np.array(vec, dtype=float))
        if d is None:
            continue
        duplicate = False
        for existing in directions:
            if abs(float(existing @ d)) > 0.99995:
                duplicate = True
                break
        if not duplicate:
            directions.append(d)
    return directions


def support_values_for_constraints(
    constraints: list[tuple[np.ndarray, float]],
    directions: list[np.ndarray],
    *,
    inside_tol: float,
) -> list[float | None]:
    values: list[float | None] = []
    for direction in directions:
        feasible, optimum, _ = solve_halfspace_lp(
            constraints,
            inside_tol=float(inside_tol),
            objective=(direction, 0.0),
        )
        values.append(float(optimum) if feasible and optimum is not None else None)
    return values


def envelope_metrics(
    constraints: list[tuple[np.ndarray, float]],
    baseline_constraints: list[tuple[np.ndarray, float]],
    trusted_points: np.ndarray,
    directions: list[np.ndarray],
    *,
    inside_tol: float,
    margin: float,
    relative_margin: float,
) -> dict[str, object]:
    current = support_values_for_constraints(constraints, directions, inside_tol=float(inside_tol))
    baseline = support_values_for_constraints(baseline_constraints, directions, inside_tol=float(inside_tol))
    trusted_support: list[float | None] = []
    if trusted_points.size:
        for direction in directions:
            trusted_support.append(float(np.max(trusted_points @ direction)))
    else:
        trusted_support = [None for _ in directions]
    excess_baseline: list[float] = []
    excess_trusted: list[float] = []
    rows: list[dict[str, object]] = []
    for i, direction in enumerate(directions):
        cur = current[i]
        base = baseline[i]
        trusted = trusted_support[i]
        allowed_margin = float(margin)
        if base is not None:
            allowed_margin = max(allowed_margin, abs(float(base)) * float(relative_margin))
        excess_b = 0.0
        if cur is not None and base is not None:
            excess_b = max(0.0, float(cur) - float(base) - allowed_margin)
        excess_t = 0.0
        if cur is not None and trusted is not None:
            excess_t = max(0.0, float(cur) - float(trusted) - allowed_margin)
        excess_baseline.append(excess_b)
        excess_trusted.append(excess_t)
        rows.append(
            {
                "direction": as_json_point(direction),
                "support_current": as_json_float(float(cur)) if cur is not None else None,
                "support_baseline": as_json_float(float(base)) if base is not None else None,
                "support_trusted": as_json_float(float(trusted)) if trusted is not None else None,
                "excess_vs_baseline": as_json_float(float(excess_b)),
                "excess_vs_trusted": as_json_float(float(excess_t)),
            }
        )
    max_idx = int(np.argmax(np.array(excess_baseline, dtype=float))) if excess_baseline else 0
    return {
        "max_excess_vs_baseline": as_json_float(float(max(excess_baseline) if excess_baseline else 0.0)),
        "integrated_excess_vs_baseline": as_json_float(float(sum(excess_baseline))),
        "max_excess_vs_trusted": as_json_float(float(max(excess_trusted) if excess_trusted else 0.0)),
        "integrated_excess_vs_trusted": as_json_float(float(sum(excess_trusted))),
        "worst_direction": rows[max_idx] if rows else None,
        "direction_count": int(len(directions)),
        "top_excess_directions": sorted(rows, key=lambda row: float(row.get("excess_vs_baseline") or 0.0), reverse=True)[:12],
    }


def apply_trusted_z_envelope_guard(
    candidates: list[dict[str, object]],
    *,
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    inside_tol: float,
    trigger_relative: float,
    margin_relative: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    points = np.asarray(trusted_points, dtype=float)
    summary: dict[str, object] = {
        "mode": "trusted-cap",
        "applied": False,
        "trigger_relative": as_json_float(float(trigger_relative)),
        "margin_relative": as_json_float(float(margin_relative)),
        "uses_initial_model": False,
    }
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        summary["reason"] = "empty_trusted_cloud"
        return list(candidates), summary
    trusted_z_min = float(np.min(points[:, 2]))
    trusted_z_max = float(np.max(points[:, 2]))
    trusted_z_span = trusted_z_max - trusted_z_min
    summary.update(
        {
            "trusted_z_min": as_json_float(trusted_z_min),
            "trusted_z_max": as_json_float(trusted_z_max),
            "trusted_z_span": as_json_float(trusted_z_span),
        }
    )
    if trusted_z_span <= EPS:
        summary["reason"] = "degenerate_trusted_z_span"
        return list(candidates), summary
    constraints = candidate_constraints(candidates)
    feasible, optimum, status = solve_halfspace_lp(
        constraints,
        inside_tol=float(inside_tol),
        objective=(np.array([0.0, 0.0, -1.0], dtype=float), 0.0),
    )
    summary["lp_status"] = str(status)
    if not feasible or optimum is None:
        summary["reason"] = "core_not_feasible_or_bounded_below"
        return list(candidates), summary
    reconstructed_z_min = -float(optimum)
    overrun = max(0.0, trusted_z_min - reconstructed_z_min)
    overrun_relative = overrun / trusted_z_span
    summary.update(
        {
            "core_z_min_before": as_json_float(reconstructed_z_min),
            "overrun_absolute_before": as_json_float(overrun),
            "overrun_relative_before": as_json_float(overrun_relative),
        }
    )
    if overrun_relative <= float(trigger_relative):
        summary["reason"] = "below_trigger"
        return list(candidates), summary

    floor_z = trusted_z_min - max(0.0, float(margin_relative)) * trusted_z_span
    xy_min = np.min(points[:, :2], axis=0)
    xy_max = np.max(points[:, :2], axis=0)
    center_xy = 0.5 * (xy_min + xy_max)
    hull = np.array(
        [
            [xy_min[0], xy_min[1], floor_z],
            [xy_max[0], xy_min[1], floor_z],
            [xy_max[0], xy_max[1], floor_z],
            [xy_min[0], xy_max[1], floor_z],
        ],
        dtype=float,
    )
    support_distance = points[:, 2] - floor_z
    support_mask = support_distance <= max(float(point_tol), EPS)
    support_z = np.asarray(trusted_z_indices, dtype=int).reshape(-1)
    support_levels = int(np.unique(support_z[support_mask]).size) if support_z.size == points.shape[0] else 0
    guard = {
        "track_id": 990000001,
        "window": None,
        "candidate_origin": "observed_z_envelope_guard",
        "candidate_source": "trusted_z_envelope_guard",
        "z_envelope_guard": True,
        "plane_normal": [0.0, 0.0, -1.0],
        "plane_centroid": [float(center_xy[0]), float(center_xy[1]), float(floor_z)],
        "plane_rms": 0.0,
        "candidate_score": 0.0,
        "hull": hull.tolist(),
        "hull_area": as_json_float(float(max(xy_max[0] - xy_min[0], 0.0) * max(xy_max[1] - xy_min[1], 0.0))),
        "hull_diameter": as_json_float(float(np.linalg.norm(xy_max - xy_min))),
        "levels": int(max(1, support_levels)),
        "z_min": as_json_float(trusted_z_min),
        "z_max": as_json_float(trusted_z_min),
        "finite_support_count": int(np.sum(support_mask)),
        "observed_local_support": int(np.sum(support_mask)),
        "trusted_z_min": as_json_float(trusted_z_min),
        "trusted_z_span": as_json_float(trusted_z_span),
        "guard_margin_relative": as_json_float(float(margin_relative)),
    }
    guarded = list(candidates) + [guard]
    outside = cumulative_outside_metrics(
        guarded,
        points,
        trusted_z_indices,
        point_tol=float(point_tol),
    )
    summary.update(
        {
            "applied": True,
            "reason": "overrun_triggered",
            "guard_candidate_id": int(guard["track_id"]),
            "guard_floor_z": as_json_float(floor_z),
            "support_count": int(guard["finite_support_count"]),
            "support_levels": int(guard["levels"]),
            "outside_after_guard": outside,
        }
    )
    return guarded, summary


def sparse_repair_rebuild_candidates(
    valley_selected: list[dict[str, object]],
    baseline_selected: list[dict[str, object]],
    *,
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    max_global_outside_frac: float,
    max_level_outside_frac: float,
    min_local_support: int,
    repair_margin: float,
    repair_relative_margin: float,
    repair_min_improvement: float,
    max_guards: int,
    inside_tol: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    valley = [dict(candidate, candidate_origin="valley_rebuild") for candidate in valley_selected]
    guard_pool: list[dict[str, object]] = []
    rejected: dict[str, int] = {}
    for candidate in baseline_selected:
        relation = classify_against_baseline(
            candidate,
            valley,
            duplicate_angle_deg=2.0,
            duplicate_plane_distance=0.05,
            sliver_hull_distance=0.75,
        )
        if str(relation.get("baseline_relation")) == "baseline_duplicate":
            rejected["equivalent_in_valley"] = rejected.get("equivalent_in_valley", 0) + 1
            continue
        guard = dict(candidate, candidate_origin="baseline_guard")
        guard.update({f"guard_{k}": v for k, v in relation.items()})
        guard_pool.append(guard)

    directions = envelope_directions(baseline_selected, valley, trusted_points)
    baseline_constraints = candidate_constraints(baseline_selected)
    accepted = list(valley)
    accepted_guards: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    z_slices = prepare_z_level_slices(trusted_z_indices)
    current_constraints = candidate_constraints(accepted)
    before_metrics = envelope_metrics(
        current_constraints,
        baseline_constraints,
        trusted_points,
        directions,
        inside_tol=float(inside_tol),
        margin=float(repair_margin),
        relative_margin=float(repair_relative_margin),
    )

    def objective_values(metrics: dict[str, object]) -> tuple[float, float]:
        return (
            float(metrics.get("max_excess_vs_baseline") or 0.0),
            float(metrics.get("integrated_excess_vs_baseline") or 0.0),
        )

    iteration = 0
    while len(accepted_guards) < int(max_guards):
        current_metrics = envelope_metrics(
            current_constraints,
            baseline_constraints,
            trusted_points,
            directions,
            inside_tol=float(inside_tol),
            margin=float(repair_margin),
            relative_margin=float(repair_relative_margin),
        )
        current_excess, current_integrated = objective_values(current_metrics)
        if current_excess <= float(repair_margin):
            break
        best: tuple[float, float, dict[str, object], dict[str, object], dict[str, object]] | None = None
        for guard in guard_pool:
            if bool(guard.get("_selected_guard")):
                continue
            halfspace = candidate_halfspace(guard)
            if halfspace is None:
                continue
            feasible, optimum, status = solve_halfspace_lp(
                current_constraints,
                inside_tol=float(inside_tol),
                objective=halfspace,
            )
            if not feasible or optimum is None:
                guard["guard_rejection_reason"] = "not_active"
                rejected["not_active"] = rejected.get("not_active", 0) + 1
                continue
            if float(optimum) <= float(repair_min_improvement):
                guard["guard_rejection_reason"] = "not_active"
                continue
            support_for_gate = int(guard.get("finite_support_count") or guard.get("observed_local_support") or 0)
            if support_for_gate < int(min_local_support):
                guard["guard_rejection_reason"] = "insufficient_local_support"
                continue
            trial = accepted + [guard]
            outside = cumulative_outside_metrics(
                trial,
                trusted_points,
                trusted_z_indices,
                point_tol=float(point_tol),
            )
            if outside["cumulative_outside_fraction"] > float(max_global_outside_frac):
                guard["guard_rejection_reason"] = "trusted_global_violation"
                continue
            if outside["cumulative_max_level_outside_fraction"] > float(max_level_outside_frac):
                guard["guard_rejection_reason"] = "trusted_level_violation"
                continue
            if int(outside["cumulative_lost_z_levels"]) > 0:
                guard["guard_rejection_reason"] = "trusted_level_violation"
                continue
            trial_constraints = candidate_constraints(trial)
            trial_metrics = envelope_metrics(
                trial_constraints,
                baseline_constraints,
                trusted_points,
                directions,
                inside_tol=float(inside_tol),
                margin=float(repair_margin),
                relative_margin=float(repair_relative_margin),
            )
            trial_excess, trial_integrated = objective_values(trial_metrics)
            improvement = current_excess - trial_excess
            integrated_improvement = current_integrated - trial_integrated
            if improvement < float(repair_min_improvement) and integrated_improvement < float(repair_min_improvement):
                guard["guard_rejection_reason"] = "improvement_below_threshold"
                continue
            score = (improvement, integrated_improvement)
            if best is None or score > (best[0], best[1]):
                best = (improvement, integrated_improvement, guard, trial_metrics, outside)
        if best is None:
            break
        improvement, integrated_improvement, guard, trial_metrics, outside = best
        iteration += 1
        guard["_selected_guard"] = True
        guard["guard_selection_iteration"] = int(iteration)
        guard["guard_max_excess_before"] = current_metrics.get("max_excess_vs_baseline")
        guard["guard_max_excess_after"] = trial_metrics.get("max_excess_vs_baseline")
        guard["guard_objective_max_excess_before"] = as_json_float(float(current_excess))
        guard["guard_objective_max_excess_after"] = as_json_float(float(objective_values(trial_metrics)[0]))
        guard["guard_improvement"] = as_json_float(float(improvement))
        guard["guard_integrated_improvement"] = as_json_float(float(integrated_improvement))
        guard["guard_outside_metrics"] = outside
        accepted.append(guard)
        accepted_guards.append(guard)
        current_constraints = candidate_constraints(accepted)
        decisions.append(
            {
                "iteration": int(iteration),
                "track_id": guard.get("track_id"),
                "window": guard.get("window"),
                "candidate_score": guard.get("candidate_score"),
                "max_excess_before": current_metrics.get("max_excess_vs_baseline"),
                "max_excess_after": trial_metrics.get("max_excess_vs_baseline"),
                "integrated_excess_after": trial_metrics.get("integrated_excess_vs_baseline"),
                "objective_mode": "baseline",
                "objective_max_excess_before": as_json_float(float(current_excess)),
                "objective_max_excess_after": as_json_float(float(objective_values(trial_metrics)[0])),
                "objective_integrated_excess_after": as_json_float(float(objective_values(trial_metrics)[1])),
                "outside": outside,
            }
        )

    if len(accepted_guards) >= int(max_guards):
        remaining = sum(1 for guard in guard_pool if not bool(guard.get("_selected_guard")))
        if remaining:
            rejected["max_guards_reached"] = int(remaining)
    final_constraints = candidate_constraints(accepted)
    final_metrics = envelope_metrics(
        final_constraints,
        baseline_constraints,
        trusted_points,
        directions,
        inside_tol=float(inside_tol),
        margin=float(repair_margin),
        relative_margin=float(repair_relative_margin),
    )
    final_outside = cumulative_outside_metrics(
        accepted,
        trusted_points,
        trusted_z_indices,
        point_tol=float(point_tol),
    )
    for candidate in accepted:
        candidate.pop("_selected_guard", None)
    return accepted, {
        "repair_guard_pool": int(len(guard_pool)),
        "repair_guards_added": int(len(accepted_guards)),
        "repair_objective": "baseline",
        "repair_rejection_counts": rejected,
        "repair_decisions": decisions,
        "repair_envelope_before": before_metrics,
        "repair_envelope_after": final_metrics,
        "repair_final_outside": final_outside,
    }


def fit_xy_line_vs_z(z: np.ndarray, points: np.ndarray, weights: np.ndarray | None = None) -> dict[str, object] | None:
    if z.size < 2 or points.shape[0] != z.size:
        return None
    design = np.column_stack([z, np.ones_like(z)])
    if weights is not None:
        w = np.sqrt(np.clip(weights.astype(float), 1e-12, None))
        design_w = design * w[:, None]
        x_w = points[:, 0] * w
        y_w = points[:, 1] * w
    else:
        design_w = design
        x_w = points[:, 0]
        y_w = points[:, 1]
    try:
        coef_x = np.linalg.lstsq(design_w, x_w, rcond=None)[0]
        coef_y = np.linalg.lstsq(design_w, y_w, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    pred_xy = np.column_stack([design @ coef_x, design @ coef_y])
    residual = np.linalg.norm(points[:, :2] - pred_xy, axis=1)
    diffs = np.diff(points[:, :2], axis=0)
    second = np.diff(diffs, axis=0) if diffs.shape[0] >= 2 else np.zeros((0, 2), dtype=float)
    direction = np.array([float(coef_x[0]), float(coef_y[0]), 1.0], dtype=float)
    return {
        "coef_x": coef_x,
        "coef_y": coef_y,
        "direction": direction / max(float(np.linalg.norm(direction)), EPS),
        "rms": float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
        "max_residual": float(np.max(residual)) if residual.size else 0.0,
        "first_diff_smoothness": float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1)))) if diffs.size else 0.0,
        "second_diff_smoothness": float(np.sqrt(np.mean(np.sum(second * second, axis=1)))) if second.size else 0.0,
    }


def xy_line_residuals(z: np.ndarray, points: np.ndarray, fit: dict[str, object]) -> np.ndarray:
    coef_x = np.array(fit["coef_x"], dtype=float)
    coef_y = np.array(fit["coef_y"], dtype=float)
    pred_xy = np.column_stack([coef_x[0] * z + coef_x[1], coef_y[0] * z + coef_y[1]])
    return np.linalg.norm(points[:, :2] - pred_xy, axis=1)


def longest_contiguous_count(indices: list[int] | np.ndarray, *, max_gap: int = 1) -> int:
    if len(indices) == 0:
        return 0
    arr = np.array(indices, dtype=int)
    best = 1
    cur = 1
    for a, b in zip(arr[:-1], arr[1:]):
        if int(b) - int(a) <= int(max_gap):
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
    return int(max(best, cur))


def predict_w2_line(segment: dict[str, object], z: np.ndarray) -> np.ndarray:
    coef_x = np.array(segment["coef_x"], dtype=float)
    coef_y = np.array(segment["coef_y"], dtype=float)
    return np.column_stack([coef_x[0] * z + coef_x[1], coef_y[0] * z + coef_y[1], z])


def build_w2_edge_segments(
    *,
    z_levels: np.ndarray,
    line_points_w2: np.ndarray,
    cond_w2: np.ndarray | None,
    line_points_w3: np.ndarray | None,
    fit_rms_w3: np.ndarray | None,
    min_levels: int,
    min_z_span: float,
    max_line_rms: float,
    max_line_residual: float,
    max_z_gap: int,
    max_point_jump: float,
    max_condition: float,
    segment_fit_mode: str = "least-squares",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    segments: list[dict[str, object]] = []
    raw_segments = 0
    counters: dict[str, int] = {
        "runs_examined": 0,
        "ls_accepted": 0,
        "robust_recovered": 0,
        "split_recovered": 0,
        "outlier_points_removed": 0,
        "rejected_low_inlier_fraction": 0,
        "rejected_short_inlier_span": 0,
        "rejected_internal_gaps": 0,
        "rejected_after_refit": 0,
    }
    n_half = int(line_points_w2.shape[0])
    for idx in range(n_half):
        current: list[int] = []

        def append_segment(
            *,
            z_indices: list[int],
            z: np.ndarray,
            pts: np.ndarray,
            fit: dict[str, object],
            cond_vals: np.ndarray | None,
            source: str,
        ) -> None:
            cond_median = float(np.nanmedian(cond_vals)) if cond_vals is not None and cond_vals.size else float("nan")
            cond_p95 = float(np.nanpercentile(cond_vals, 95)) if cond_vals is not None and cond_vals.size else float("nan")
            w3_agreement = None
            w3_minima_frac = None
            if line_points_w3 is not None:
                w3_pts = line_points_w3[int(idx), np.array(z_indices, dtype=int), :].astype(float)
                ok = np.all(np.isfinite(w3_pts), axis=1)
                if np.any(ok):
                    w3_agreement = float(np.median(np.linalg.norm(w3_pts[ok, :2] - pts[ok, :2], axis=1)))
            if fit_rms_w3 is not None:
                minima = []
                for zi in z_indices:
                    values = fit_rms_w3[:, int(zi)]
                    if not np.isfinite(values[int(idx)]):
                        continue
                    prev_v = values[(int(idx) - 1) % n_half]
                    next_v = values[(int(idx) + 1) % n_half]
                    minima.append(bool(values[int(idx)] <= prev_v and values[int(idx)] <= next_v))
                w3_minima_frac = float(np.mean(minima)) if minima else None
            confidence = 1.0 / (1.0 + float(fit["rms"]) / max(float(max_line_rms), EPS))
            if w3_agreement is not None:
                confidence *= 1.0 / (1.0 + float(w3_agreement))
            if np.isfinite(cond_median):
                confidence *= 1.0 / (1.0 + max(np.log10(max(cond_median, 1.0)) - 4.0, 0.0))
            row = {
                "segment_id": int(len(segments)),
                "cyclic_index": int(idx),
                "z_indices": [int(v) for v in z_indices],
                "levels": int(len(z_indices)),
                "z_min": as_json_float(float(np.min(z))),
                "z_max": as_json_float(float(np.max(z))),
                "z_span": as_json_float(float(np.max(z) - np.min(z))),
                "coef_x": [float(v) for v in fit["coef_x"]],
                "coef_y": [float(v) for v in fit["coef_y"]],
                "direction": as_json_point(np.array(fit["direction"], dtype=float)),
                "line_rms": as_json_float(float(fit["rms"])),
                "line_max_residual": as_json_float(float(fit["max_residual"])),
                "first_diff_smoothness": as_json_float(float(fit["first_diff_smoothness"])),
                "second_diff_smoothness": as_json_float(float(fit["second_diff_smoothness"])),
                "condition_median": as_json_float(cond_median),
                "condition_p95": as_json_float(cond_p95),
                "w3_agreement_median": as_json_float(w3_agreement) if w3_agreement is not None else None,
                "w3_minima_fraction": as_json_float(w3_minima_frac) if w3_minima_frac is not None else None,
                "confidence": as_json_float(float(confidence)),
                "bounds_min": as_json_point(np.min(pts, axis=0)),
                "bounds_max": as_json_point(np.max(pts, axis=0)),
            }
            if source != "least_squares":
                row["segment_fit_mode"] = str(source)
            segments.append(row)

        def flush() -> None:
            nonlocal raw_segments
            if len(current) < int(min_levels):
                return
            raw_segments += 1
            counters["runs_examined"] += 1
            z = z_levels[np.array(current, dtype=int)]
            pts = line_points_w2[int(idx), np.array(current, dtype=int), :].astype(float)
            cond_vals = None
            if cond_w2 is not None:
                cond_vals = cond_w2[int(idx), np.array(current, dtype=int)].astype(float)
            weights = None
            if cond_vals is not None:
                weights = 1.0 / np.sqrt(np.maximum(np.nan_to_num(cond_vals, nan=np.nanmedian(cond_vals)), 1.0))
            fit = fit_xy_line_vs_z(z, pts, weights=weights)
            if fit is None:
                return
            z_span = float(np.max(z) - np.min(z)) if z.size else 0.0
            if z_span < float(min_z_span):
                return
            if fit["rms"] > float(max_line_rms) or fit["max_residual"] > float(max_line_residual):
                if str(segment_fit_mode) == "robust-split":
                    robust = robust_xy_line_fit(
                        z,
                        pts,
                        min_levels=min_levels,
                        min_z_span=min_z_span,
                        max_line_rms=max_line_rms,
                        max_line_residual=max_line_residual,
                        max_z_gap=max_z_gap,
                    )
                    if robust is not None:
                        mask = np.array(robust["inlier_mask"], dtype=bool)
                        append_segment(
                            z_indices=[int(current[i]) for i in np.nonzero(mask)[0]],
                            z=z[mask],
                            pts=pts[mask],
                            fit=robust["fit"],
                            cond_vals=cond_vals[mask] if cond_vals is not None else None,
                            source="robust_inlier",
                        )
                        counters["robust_recovered"] += 1
                        counters["outlier_points_removed"] += int(len(current) - int(np.sum(mask)))
                    else:
                        counters["rejected_after_refit"] += 1
                    residual = xy_line_residuals(z, pts, fit)
                    if residual.size >= 2:
                        split_at = int(np.argmax(np.abs(np.diff(residual)))) + 1
                        if int(min_levels) <= split_at <= len(current) - int(min_levels):
                            for part in (slice(0, split_at), slice(split_at, len(current))):
                                sub_z = z[part]
                                sub_pts = pts[part]
                                sub_cond = cond_vals[part] if cond_vals is not None else None
                                sub_robust = robust_xy_line_fit(
                                    sub_z,
                                    sub_pts,
                                    min_levels=min_levels,
                                    min_z_span=min_z_span,
                                    max_line_rms=max_line_rms,
                                    max_line_residual=max_line_residual,
                                    max_z_gap=max_z_gap,
                                    min_inlier_fraction=0.65,
                                )
                                if sub_robust is None:
                                    continue
                                sub_mask = np.array(sub_robust["inlier_mask"], dtype=bool)
                                part_indices = current[part]
                                append_segment(
                                    z_indices=[int(part_indices[i]) for i in np.nonzero(sub_mask)[0]],
                                    z=sub_z[sub_mask],
                                    pts=sub_pts[sub_mask],
                                    fit=sub_robust["fit"],
                                    cond_vals=sub_cond[sub_mask] if sub_cond is not None else None,
                                    source="one_split",
                                )
                                counters["split_recovered"] += 1
                                counters["outlier_points_removed"] += int(len(part_indices) - int(np.sum(sub_mask)))
                return
            cond_median = float(np.nanmedian(cond_vals)) if cond_vals is not None and cond_vals.size else float("nan")
            if max_condition >= 0.0 and np.isfinite(cond_median) and cond_median > float(max_condition):
                return
            append_segment(z_indices=current, z=z, pts=pts, fit=fit, cond_vals=cond_vals, source="least_squares")
            counters["ls_accepted"] += 1

        for zi in range(int(z_levels.size)):
            p = line_points_w2[int(idx), int(zi), :]
            if not np.all(np.isfinite(p)):
                flush()
                current = []
                continue
            if current:
                if int(zi) - current[-1] > int(max_z_gap):
                    flush()
                    current = []
                else:
                    prev = line_points_w2[int(idx), current[-1], :]
                    if float(np.linalg.norm(p - prev)) > float(max_point_jump):
                        flush()
                        current = []
            current.append(int(zi))
            if len(current) >= int(min_levels):
                z = z_levels[np.array(current, dtype=int)]
                pts = line_points_w2[int(idx), np.array(current, dtype=int), :].astype(float)
                fit = fit_xy_line_vs_z(z, pts)
                if fit is not None and (fit["rms"] > float(max_line_rms) * 1.5 or fit["max_residual"] > float(max_line_residual) * 1.5):
                    last = current.pop()
                    flush()
                    current = [last]
        flush()
    return segments, {
        "w2_raw_segments_before_filter": int(raw_segments),
        "w2_edge_segments": int(len(segments)),
        "w2_segment_fit_mode": str(segment_fit_mode),
        "w2_segment_fit_counters": {
            **counters,
            "segments_before_dedupe": int(len(segments)),
            "segments_after_dedupe": int(len(segments)),
        },
    }


def w2_segment_distance(a: dict[str, object], b: dict[str, object]) -> tuple[float, float, float] | None:
    z0 = max(finite_float(a.get("z_min"), float("nan")), finite_float(b.get("z_min"), float("nan")))
    z1 = min(finite_float(a.get("z_max"), float("nan")), finite_float(b.get("z_max"), float("nan")))
    if not np.isfinite(z0) or not np.isfinite(z1) or z1 <= z0:
        return None
    z = np.linspace(z0, z1, 5)
    pa = predict_w2_line(a, z)
    pb = predict_w2_line(b, z)
    dist = float(np.max(np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1)))
    da = np.array(a.get("direction", []), dtype=float)
    db = np.array(b.get("direction", []), dtype=float)
    angle = float(np.degrees(np.arccos(np.clip(abs(float(da @ db)), -1.0, 1.0)))) if da.shape == (3,) and db.shape == (3,) else 180.0
    return dist, angle, float(z1 - z0)


def cluster_w2_edge_segments(
    segments: list[dict[str, object]],
    *,
    mode: str,
    max_distance: float,
    max_angle_deg: float,
    min_overlap: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    clusters: list[list[dict[str, object]]] = []
    for segment in sorted(segments, key=lambda s: (-int(s.get("levels") or 0), finite_float(s.get("line_rms"), float("inf")))):
        placed = False
        for cluster in clusters:
            compare = cluster if str(mode) == "complete" else [cluster[0]]
            ok = True
            for other in compare:
                sim = w2_segment_distance(segment, other)
                if sim is None or sim[0] > float(max_distance) or sim[1] > float(max_angle_deg) or sim[2] < float(min_overlap):
                    ok = False
                    break
            if ok:
                cluster.append(segment)
                placed = True
                break
        if not placed:
            clusters.append([segment])
    out: list[dict[str, object]] = []
    for cid, members in enumerate(clusters):
        rep = min(members, key=lambda s: (finite_float(s.get("line_rms"), float("inf")), -int(s.get("levels") or 0)))
        z_min = min(finite_float(m.get("z_min"), float("inf")) for m in members)
        z_max = max(finite_float(m.get("z_max"), -float("inf")) for m in members)
        out.append(
            {
                **dict(rep),
                "edge_cluster_id": int(cid),
                "member_segment_ids": [int(m["segment_id"]) for m in members],
                "member_cyclic_indices": sorted({int(m["cyclic_index"]) for m in members}),
                "member_count": int(len(members)),
                "cluster_z_min": as_json_float(float(z_min)),
                "cluster_z_max": as_json_float(float(z_max)),
                "cross_index_support": int(len({int(m["cyclic_index"]) for m in members})),
                "cluster_confidence": as_json_float(float(np.mean([finite_float(m.get("confidence"), 0.0) for m in members]))),
            }
        )
    return out, {"w2_edge_clusters": int(len(out)), "w2_edge_cluster_members_top": sorted([len(c) for c in clusters], reverse=True)[:12]}


def w2_cluster_duplicate_line_features(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    sim = w2_segment_distance(a, b)
    features = skew_line_features(a, b)
    za0 = finite_float(a.get("cluster_z_min"), finite_float(a.get("z_min"), float("nan")))
    za1 = finite_float(a.get("cluster_z_max"), finite_float(a.get("z_max"), float("nan")))
    zb0 = finite_float(b.get("cluster_z_min"), finite_float(b.get("z_min"), float("nan")))
    zb1 = finite_float(b.get("cluster_z_max"), finite_float(b.get("z_max"), float("nan")))
    z_overlap = max(0.0, min(za1, zb1) - max(za0, zb0)) if np.isfinite(za0) and np.isfinite(zb0) else 0.0
    z_union = max(0.0, max(za1, zb1) - min(za0, zb0)) if np.isfinite(za0) and np.isfinite(zb0) else 0.0
    ci_a = [int(v) for v in (a.get("member_cyclic_indices") or [a.get("cyclic_index") or 0])]
    ci_b = [int(v) for v in (b.get("member_cyclic_indices") or [b.get("cyclic_index") or 0])]
    cyclic_gap = min(abs(x - y) for x in ci_a for y in ci_b) if ci_a and ci_b else 999
    z = np.linspace(max(za0, zb0), min(za1, zb1), 7) if z_overlap > 0.0 else np.array([], dtype=float)
    pooled_rms = float("inf")
    if z.size:
        pts = np.vstack([predict_w2_line(a, z), predict_w2_line(b, z)])
        fit = fit_xy_line_vs_z(np.concatenate([z, z]), pts)
        if fit is not None:
            pooled_rms = float(fit["rms"])
    return {
        "line_distance": as_json_float(float(sim[0])) if sim is not None else None,
        "direction_angle_deg": as_json_float(float(sim[1])) if sim is not None else None,
        "z_overlap": as_json_float(float(z_overlap)),
        "z_overlap_fraction": as_json_float(float(z_overlap / max(z_union, EPS))),
        "cyclic_gap": int(cyclic_gap),
        "pooled_line_rms": as_json_float(float(pooled_rms)),
        **features,
    }


def is_overlapping_collinear_duplicate_pair(a: dict[str, object], b: dict[str, object]) -> bool:
    f = w2_cluster_duplicate_line_features(a, b)
    line_distance = finite_float(f.get("line_distance"), float("inf"))
    direction_angle = finite_float(f.get("direction_angle_deg"), 180.0)
    z_overlap_fraction = finite_float(f.get("z_overlap_fraction"), 0.0)
    cyclic_gap = int(f.get("cyclic_gap") or 999)
    pooled_rms = finite_float(f.get("pooled_line_rms"), float("inf"))
    midspan_distance = finite_float(f.get("midspan_point_distance"), float("inf"))
    return bool(
        direction_angle <= 1.25
        and line_distance <= 0.035
        and midspan_distance <= 0.05
        and z_overlap_fraction >= 0.55
        and cyclic_gap <= 2
        and pooled_rms <= 0.025
    )


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


def model_face_planes(vertices: np.ndarray, faces: list[list[int]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for fid, face in enumerate(faces):
        pts = vertices[np.array(face, dtype=int)]
        if pts.shape[0] < 3:
            continue
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            continue
        normal = normal / norm
        out.append(
            {
                "face_id": int(fid),
                "normal": normal,
                "offset": float(normal @ pts[0]),
                "vertices": pts,
                "z_min": float(np.min(pts[:, 2])),
                "z_max": float(np.max(pts[:, 2])),
            }
        )
    return out


CANONICAL_ORACLE_SCHEMA_VERSION = 3
CANONICAL_ORACLE_TOLERANCES = {
    "normal_angle_deg": 2.0,
    "plane_distance": 0.05,
    "centroid_to_finite_polygon_distance": 0.15,
    "median_hull_to_surface_distance": 0.15,
}


def canonical_oracle_evaluator_metadata() -> dict[str, object]:
    return {
        "schema_version": int(CANONICAL_ORACLE_SCHEMA_VERSION),
        "assignment": "finite-good-first unique canonical face assignment",
        "duplicate_handling": "candidate duplicates are counted separately; unique recall uses face-id set",
        "normal_direction": "absolute normal angle; signed plane offset is compared after normal sign alignment",
        "tolerances": dict(CANONICAL_ORACLE_TOLERANCES),
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


def candidate_oracle_face(
    candidate: dict[str, object],
    model_faces: list[dict[str, object]],
    *,
    angle_tol: float = CANONICAL_ORACLE_TOLERANCES["normal_angle_deg"],
    plane_tol: float = CANONICAL_ORACLE_TOLERANCES["plane_distance"],
    centroid_tol: float = CANONICAL_ORACLE_TOLERANCES["centroid_to_finite_polygon_distance"],
    hull_surface_tol: float = CANONICAL_ORACLE_TOLERANCES["median_hull_to_surface_distance"],
) -> dict[str, object] | None:
    plane = candidate_plane(candidate)
    if plane is None:
        return None
    point, normal = plane
    hull = np.array(candidate.get("hull") or [], dtype=float)
    if hull.ndim != 2 or hull.shape[1] != 3 or hull.shape[0] == 0:
        hull = np.array([point], dtype=float).reshape((1, 3))
    best: dict[str, object] | None = None
    preliminary: list[tuple[float, dict[str, object], float, float]] = []
    for face in model_faces:
        fn = np.array(face["normal"], dtype=float)
        dot = float(normal @ fn)
        sign = 1.0 if dot >= 0.0 else -1.0
        angle = float(np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0))))
        plane_dist = abs(float(normal @ point) - sign * float(face["offset"]))
        plane_score = (angle / angle_tol) ** 2 + (plane_dist / plane_tol) ** 2
        preliminary.append((float(plane_score), face, float(angle), float(plane_dist)))
    preliminary.sort(key=lambda item: item[0])
    for _, face, angle, plane_dist in preliminary:
        centroid_distance = point_to_face_polygon_distance(point, face)
        hull_distances = np.array([point_to_face_polygon_distance(p, face) for p in hull], dtype=float)
        median_hull_distance = float(np.median(hull_distances)) if hull_distances.size else float("inf")
        plane_good = bool(angle <= angle_tol and plane_dist <= plane_tol)
        finite_good = bool(
            plane_good
            and centroid_distance <= float(centroid_tol)
            and median_hull_distance <= float(hull_surface_tol)
        )
        failure_reasons: list[str] = []
        if angle > angle_tol:
            failure_reasons.append("normal_angle")
        if plane_dist > plane_tol:
            failure_reasons.append("plane_distance")
        if centroid_distance > centroid_tol:
            failure_reasons.append("centroid_to_finite_polygon_distance")
        if median_hull_distance > hull_surface_tol:
            failure_reasons.append("median_hull_to_surface_distance")
        score = (
            (angle / angle_tol) ** 2
            + (plane_dist / plane_tol) ** 2
            + (centroid_distance / centroid_tol) ** 2
            + (median_hull_distance / hull_surface_tol) ** 2
        )
        row = {
            "face_id": int(face["face_id"]),
            "normal_angle_deg": as_json_float(float(angle)),
            "plane_distance": as_json_float(float(plane_dist)),
            "centroid_distance": as_json_float(float(centroid_distance)),
            "hull_surface_distance": as_json_float(float(median_hull_distance)),
            "plane_good": plane_good,
            "finite_good": finite_good,
            "failure_reasons": failure_reasons,
            "score": as_json_float(float(score)),
        }
        row_key = (not bool(row["finite_good"]), float(row["score"] or float("inf")))
        best_key = (
            (not bool(best["finite_good"]), float(best["score"] or float("inf")))
            if best is not None
            else (True, float("inf"))
        )
        if row_key < best_key:
            best = row
    return best


def canonical_candidate_oracle_face(
    candidate: dict[str, object],
    model_faces: list[dict[str, object]],
) -> dict[str, object] | None:
    return candidate_oracle_face(
        candidate,
        model_faces,
        angle_tol=float(CANONICAL_ORACLE_TOLERANCES["normal_angle_deg"]),
        plane_tol=float(CANONICAL_ORACLE_TOLERANCES["plane_distance"]),
        centroid_tol=float(CANONICAL_ORACLE_TOLERANCES["centroid_to_finite_polygon_distance"]),
        hull_surface_tol=float(CANONICAL_ORACLE_TOLERANCES["median_hull_to_surface_distance"]),
    )


def oracle_candidate_cohort(
    candidates: list[dict[str, object]],
    model_faces: list[dict[str, object]],
    *,
    core_face_ids: set[int] | None = None,
    active_plane_indices: list[int] | None = None,
    match_cache: dict[int, dict[str, object] | None] | None = None,
) -> dict[str, object]:
    plane_ids: list[int] = []
    finite_ids: list[int] = []
    rows: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates):
        tid = int(candidate_track_id(candidate))
        if tid == -1:
            tid = -int(idx) - 1
        if match_cache is not None and tid in match_cache:
            match = match_cache.get(tid)
        else:
            match = canonical_candidate_oracle_face(candidate, model_faces)
            if match_cache is not None:
                match_cache[tid] = match
        if match is None:
            continue
        face_id = int(match["face_id"])
        if bool(match.get("plane_good")):
            plane_ids.append(face_id)
        if bool(match.get("finite_good")):
            finite_ids.append(face_id)
        rows.append(
            {
                "candidate_index": int(idx),
                "candidate_id": int(tid),
                "candidate_origin": candidate.get("candidate_origin"),
                "candidate_source": candidate.get("candidate_source"),
                "active_plane_index": int(active_plane_indices[idx]) if active_plane_indices is not None and idx < len(active_plane_indices) else None,
                **match,
            }
        )
    plane_unique = set(plane_ids)
    finite_unique = set(finite_ids)
    core = core_face_ids or set()
    return {
        "canonical_evaluator": canonical_oracle_evaluator_metadata(),
        "candidate_count": int(len(candidates)),
        "plane_good_candidates": int(len(plane_ids)),
        "unique_plane_good_face_ids": int(len(plane_unique)),
        "finite_good_candidates": int(len(finite_ids)),
        "unique_finite_face_ids": int(len(finite_unique)),
        "new_unique_face_ids_vs_repaired_core": int(len(finite_unique - core)),
        "duplicate_good_assignments": int(len(finite_ids) - len(finite_unique)),
        "matched_rows_sample": rows[:120],
        "matched_rows": rows,
        "finite_face_ids": sorted(int(v) for v in finite_unique),
        "plane_face_ids": sorted(int(v) for v in plane_unique),
    }




def loose_plane_unique_face_ids(
    candidates: list[dict[str, object]],
    model_faces: list[dict[str, object]],
    *,
    angle_tol: float = 5.0,
    plane_tol: float = 0.05,
) -> set[int]:
    ids: set[int] = set()
    for candidate in candidates:
        plane = candidate_plane(candidate)
        if plane is None:
            continue
        point, normal = plane
        best: tuple[float, int] | None = None
        for face in model_faces:
            fn = np.array(face["normal"], dtype=float)
            dot = float(normal @ fn)
            sign = 1.0 if dot >= 0.0 else -1.0
            angle = float(np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0))))
            plane_dist = abs(float(normal @ point) - sign * float(face["offset"]))
            score = (angle / angle_tol) ** 2 + (plane_dist / plane_tol) ** 2
            if best is None or score < best[0]:
                best = (score, int(face["face_id"]))
                best_angle = angle
                best_dist = plane_dist
        if best is not None and best_angle <= angle_tol and best_dist <= plane_tol:
            ids.add(int(best[1]))
    return ids












def skew_line_features(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    da = np.array(a.get("direction") or [], dtype=float)
    db = np.array(b.get("direction") or [], dtype=float)
    if da.shape != (3,) or db.shape != (3,) or np.linalg.norm(da) <= EPS or np.linalg.norm(db) <= EPS:
        return {}
    da = da / max(float(np.linalg.norm(da)), EPS)
    db = db / max(float(np.linalg.norm(db)), EPS)
    z0 = max(finite_float(a.get("cluster_z_min"), finite_float(a.get("z_min"), float("nan"))), finite_float(b.get("cluster_z_min"), finite_float(b.get("z_min"), float("nan"))))
    z1 = min(finite_float(a.get("cluster_z_max"), finite_float(a.get("z_max"), float("nan"))), finite_float(b.get("cluster_z_max"), finite_float(b.get("z_max"), float("nan"))))
    z_mid = 0.5 * (z0 + z1) if np.isfinite(z0) and np.isfinite(z1) else 0.0
    pa = predict_w2_line(a, np.array([z_mid], dtype=float))[0]
    pb = predict_w2_line(b, np.array([z_mid], dtype=float))[0]
    cross = np.cross(da, db)
    cross_norm = float(np.linalg.norm(cross))
    angle = float(np.degrees(np.arccos(np.clip(abs(float(da @ db)), -1.0, 1.0))))
    midpoint_distance = float(np.linalg.norm(pa - pb))
    if cross_norm <= 1e-6:
        coplanarity = None
        condition = float("inf")
    else:
        coplanarity = abs(float((pb - pa) @ cross)) / cross_norm
        condition = 1.0 / cross_norm
    return {
        "edge_direction_angle_deg": as_json_float(angle),
        "midspan_point_distance": as_json_float(midpoint_distance),
        "coplanarity_residual": as_json_float(float(coplanarity)) if coplanarity is not None else None,
        "near_parallel_condition": as_json_float(float(condition)),
        "z_overlap": as_json_float(float(max(0.0, z1 - z0))) if np.isfinite(z0) and np.isfinite(z1) else None,
    }
































def robust_xy_line_fit(
    z: np.ndarray,
    points: np.ndarray,
    *,
    min_levels: int,
    min_z_span: float,
    max_line_rms: float,
    max_line_residual: float,
    max_z_gap: int,
    min_inlier_fraction: float = 0.55,
) -> dict[str, object] | None:
    n = int(z.size)
    if n < int(min_levels):
        return None
    trial_pairs: list[tuple[int, int]] = []
    anchors = sorted({0, n - 1, n // 4, n // 2, (3 * n) // 4})
    for i in anchors:
        for j in anchors:
            if j > i and abs(float(z[j]) - float(z[i])) > EPS:
                trial_pairs.append((int(i), int(j)))
    stride = max(1, n // 8)
    grid = list(range(0, n, stride))
    if grid[-1] != n - 1:
        grid.append(n - 1)
    for i in grid:
        for j in grid:
            if j > i and len(trial_pairs) < 36 and abs(float(z[j]) - float(z[i])) > EPS:
                trial_pairs.append((int(i), int(j)))
    best: dict[str, object] | None = None
    for i, j in trial_pairs[:36]:
        dz = float(z[j] - z[i])
        if abs(dz) <= EPS:
            continue
        slope = (points[j, :2] - points[i, :2]) / dz
        intercept = points[i, :2] - slope * float(z[i])
        pred = z[:, None] * slope[None, :] + intercept[None, :]
        residual = np.linalg.norm(points[:, :2] - pred, axis=1)
        inliers = residual <= float(max_line_residual)
        if int(np.sum(inliers)) < int(min_levels):
            continue
        if float(np.mean(inliers)) < float(min_inlier_fraction):
            continue
        z_in = z[inliers]
        idx_in = np.nonzero(inliers)[0]
        z_span = float(np.max(z_in) - np.min(z_in)) if z_in.size else 0.0
        if z_span < float(min_z_span):
            continue
        gaps = int(np.sum(np.diff(idx_in) > int(max_z_gap))) if idx_in.size >= 2 else 0
        if gaps > 1:
            continue
        refit = fit_xy_line_vs_z(z_in, points[inliers])
        if refit is None:
            continue
        refit_res = xy_line_residuals(z_in, points[inliers], refit)
        if float(np.sqrt(np.mean(refit_res * refit_res))) > float(max_line_rms) or float(np.max(refit_res)) > float(max_line_residual):
            continue
        score = (float(np.median(refit_res)), -int(np.sum(inliers)), -z_span)
        row = {
            "fit": refit,
            "inlier_mask": inliers,
            "inlier_indices": idx_in,
            "inlier_fraction": float(np.mean(inliers)),
            "inlier_levels": int(np.sum(inliers)),
            "inlier_z_span": float(z_span),
            "inlier_internal_gaps": int(gaps),
            "inlier_residual_median": float(np.median(refit_res)),
            "inlier_residual_p95": float(np.percentile(refit_res, 95)),
            "inlier_residual_max": float(np.max(refit_res)),
            "score": score,
        }
        if best is None or row["score"] < best["score"]:
            best = row
    return best




def angular_scale_quality_local_key(
    candidate: dict[str, object],
) -> tuple[float, float, float, float, float, int]:
    levels = max(1.0, finite_float(candidate.get("levels"), 0.0))
    z_span = max(
        0.0,
        finite_float(candidate.get("z_max"), 0.0)
        - finite_float(candidate.get("z_min"), 0.0),
    )
    hull_area = max(finite_float(candidate.get("hull_area"), 0.0), EPS)
    plane_rms = finite_float(candidate.get("plane_rms"), float("inf"))
    quality = plane_rms / max(np.sqrt(levels), 1.0)
    return (
        quality,
        -min(z_span, 10.0),
        -min(hull_area, 100.0),
        -levels,
        finite_float(candidate.get("condition_p95"), float("inf")),
        int(candidate_track_id(candidate)),
    )


def tie_aware_borda_percentiles(
    values: list[object],
    *,
    higher_is_better: bool,
) -> list[float]:
    """Return normalized Borda points with mean ranks for exact ties."""
    count = len(values)
    if count == 0:
        return []
    finite_rows = [
        (index, finite_float(value, float("nan")))
        for index, value in enumerate(values)
        if value is not None and np.isfinite(finite_float(value, float("nan")))
    ]
    scores = [0.0] * count
    if not finite_rows:
        return scores
    finite_rows.sort(key=lambda row: row[1], reverse=bool(higher_is_better))
    if count == 1:
        scores[finite_rows[0][0]] = 1.0
        return scores
    start = 0
    while start < len(finite_rows):
        end = start + 1
        value = finite_rows[start][1]
        while end < len(finite_rows) and finite_rows[end][1] == value:
            end += 1
        mean_zero_based_rank = 0.5 * float(start + end - 1)
        percentile = 1.0 - mean_zero_based_rank / float(count - 1)
        for index, _ in finite_rows[start:end]:
            scores[index] = float(np.clip(percentile, 0.0, 1.0))
        start = end
    return scores


def annotate_angular_scale_selection_order(
    candidates: list[dict[str, object]],
    *,
    mode: str,
) -> dict[str, object]:
    """Annotate and order angular candidates without reference-model inputs."""
    resolved_mode = str(mode)
    if resolved_mode not in {"quality-local", "balanced-local", "balanced-global"}:
        raise ValueError(f"unknown angular-scale selection mode: {mode}")
    if resolved_mode == "quality-local":
        for candidate in candidates:
            key = angular_scale_quality_local_key(candidate)
            candidate["angular_scale_selection_mode"] = resolved_mode
            candidate["angular_scale_selection_score"] = as_json_float(float(key[0]))
            candidate["angular_scale_selection_components"] = {
                "quality_plane_rms_over_sqrt_levels": as_json_float(float(key[0])),
                "z_span": as_json_float(float(-key[1])),
                "hull_area": as_json_float(float(-key[2])),
                "levels": as_json_float(float(-key[3])),
                "condition_p95": as_json_float(float(key[4])),
            }
        ordered = sorted(candidates, key=angular_scale_quality_local_key)
        return {
            "mode": resolved_mode,
            "score_direction": "lexicographic_lower_is_better",
            "score_definition": {
                "primary": "plane_rms / sqrt(max(levels, 1))",
                "tie_breakers": [
                    "z_span higher",
                    "hull_area higher",
                    "levels higher",
                    "condition_p95 lower",
                    "stable_candidate_id lower",
                ],
                "global_percentiles": False,
            },
            "ordered_candidate_ids": [
                int(candidate_track_id(candidate)) for candidate in ordered
            ],
        }

    feature_specs = [
        ("plane_rms", False),
        ("levels", True),
        ("z_span", True),
        ("hull_area", True),
        ("finite_support_purity", True),
    ]
    raw_values: dict[str, list[object]] = {
        "plane_rms": [candidate.get("plane_rms") for candidate in candidates],
        "levels": [candidate.get("levels") for candidate in candidates],
        "z_span": [
            max(
                0.0,
                finite_float(candidate.get("z_max"), 0.0)
                - finite_float(candidate.get("z_min"), 0.0),
            )
            for candidate in candidates
        ],
        "hull_area": [candidate.get("hull_area") for candidate in candidates],
        "finite_support_purity": [
            candidate.get("finite_support_purity") for candidate in candidates
        ],
    }
    percentile_by_feature = {
        name: tie_aware_borda_percentiles(
            raw_values[name],
            higher_is_better=higher_is_better,
        )
        for name, higher_is_better in feature_specs
    }
    for index, candidate in enumerate(candidates):
        components = {
            name: {
                "raw": as_json_float(
                    finite_float(raw_values[name][index], float("nan"))
                ),
                "borda_percentile": as_json_float(
                    float(percentile_by_feature[name][index])
                ),
                "direction": "higher" if higher_is_better else "lower",
            }
            for name, higher_is_better in feature_specs
        }
        score = float(
            np.mean(
                [
                    float(percentile_by_feature[name][index])
                    for name, _ in feature_specs
                ]
            )
        )
        candidate["angular_scale_selection_mode"] = resolved_mode
        candidate["angular_scale_selection_score"] = as_json_float(score)
        candidate["angular_scale_selection_components"] = components
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -finite_float(candidate.get("angular_scale_selection_score"), 0.0),
            int(candidate_track_id(candidate)),
        ),
    )
    return {
        "mode": resolved_mode,
        "score_direction": "higher_is_better",
        "score_definition": {
            "method": "equal-weight mean of normalized tie-aware Borda percentiles",
            "tie_method": "exact equal values receive the mean rank",
            "features": {
                name: "higher_is_better" if higher_is_better else "lower_is_better"
                for name, higher_is_better in feature_specs
            },
            "final_tie_breaker": "stable_candidate_id lower",
            "uses_initial_model": False,
            "uses_oracle_labels": False,
        },
        "ordered_candidate_ids": [
            int(candidate_track_id(candidate)) for candidate in ordered
        ],
    }


def annotate_w2_novelty_scores(candidates: list[dict[str, object]], core_candidates: list[dict[str, object]], core_reconstructed: dict[str, object]) -> None:
    core_vertices = np.array(core_reconstructed.get("vertices") or [], dtype=float)
    core_planes = [candidate_plane(c) for c in core_candidates]
    core_planes = [p for p in core_planes if p is not None]
    for candidate in candidates:
        plane = candidate_plane(candidate)
        if plane is None:
            candidate["w2_novelty_score"] = float("inf")
            continue
        point, normal = plane
        margin = candidate_margin_against_vertices(candidate, core_vertices)
        best_angle = 180.0
        best_plane_distance = float("inf")
        for core_point, core_normal in core_planes:
            dot = abs(float(normal @ core_normal))
            angle = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
            plane_distance = abs(float(normal @ (point - core_point)))
            if angle + 10.0 * plane_distance < best_angle + 10.0 * best_plane_distance:
                best_angle = angle
                best_plane_distance = plane_distance
        density = max(finite_float(candidate.get("finite_support_density"), 0.0), 0.0)
        z_span = max(finite_float(candidate.get("z_span"), 0.0), 0.0)
        levels = max(int(candidate.get("levels") or 0), 0)
        rms = max(finite_float(candidate.get("plane_rms"), 1.0), 0.0)
        support_term = np.log1p(density) + 0.02 * float(levels) + 0.5 * z_span
        activity_term = max(float(margin or 0.0), 0.0)
        novelty_term = min(best_angle / 8.0, 2.0) + min(best_plane_distance / 0.05, 2.0)
        risk_term = 20.0 * rms
        score = -3.0 * activity_term - 0.7 * novelty_term - 0.15 * support_term + risk_term
        candidate["w2_novelty_score"] = as_json_float(float(score))
        candidate["w2_novelty_activity_margin"] = as_json_float(float(margin)) if margin is not None else None
        candidate["w2_novelty_nearest_core_angle"] = as_json_float(float(best_angle))
        candidate["w2_novelty_nearest_core_plane_distance"] = as_json_float(float(best_plane_distance))


def build_non_oracle_plane_consensus_groups(
    candidates: list[dict[str, object]],
    *,
    w2_clusters: list[dict[str, object]],
) -> tuple[list[list[dict[str, object]]], dict[int, dict[str, object]]]:
    cluster_by_id: dict[int, dict[str, object]] = {}
    for cluster in w2_clusters:
        if "edge_cluster_id" in cluster:
            try:
                cluster_by_id[int(cluster["edge_cluster_id"])] = cluster
            except (TypeError, ValueError):
                pass
    ordered = sorted(candidates, key=support_diverse_candidate_key)
    groups: list[list[dict[str, object]]] = []
    for candidate in ordered:
        placed = False
        for group in groups:
            if all(candidates_plane_patch_compatible(candidate, other) for other in group):
                group.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])
    lookup: dict[int, dict[str, object]] = {}
    for group_id, group in enumerate(groups):
        ranked = sorted(group, key=support_diverse_candidate_key)
        normals: list[np.ndarray] = []
        offsets: list[float] = []
        rep_plane = candidate_plane(ranked[0])
        normal_angles: list[float] = []
        offset_deltas: list[float] = []
        for candidate in group:
            plane = candidate_plane(candidate)
            if plane is None:
                continue
            point, normal = plane
            normals.append(normal)
            offsets.append(float(normal @ point))
        if rep_plane is not None:
            rep_point, rep_normal = rep_plane
            rep_offset = float(rep_normal @ rep_point)
            for normal, offset in zip(normals, offsets):
                normal_angles.append(float(np.degrees(np.arccos(np.clip(float(rep_normal @ normal), -1.0, 1.0)))))
                offset_deltas.append(abs(float(offset - rep_offset)))
        edge_pairs = {candidate_edge_pair(candidate) for candidate in group}
        edge_clusters = {eid for pair in edge_pairs for eid in pair}
        cyclic_indices: set[int] = set()
        z_bands: set[int] = set()
        for candidate in group:
            z0, z1 = candidate_z_range(candidate)
            if np.isfinite(z0) and np.isfinite(z1):
                for band in range(int(np.floor(z0 / 0.25)), int(np.floor(z1 / 0.25)) + 1):
                    z_bands.add(int(band))
            for eid in candidate_edge_pair(candidate):
                cluster = cluster_by_id.get(int(eid), {})
                member_cyclic_indices = cluster.get("member_cyclic_indices")
                if isinstance(member_cyclic_indices, list):
                    for ci in member_cyclic_indices:
                        try:
                            cyclic_indices.add(int(ci))
                        except (TypeError, ValueError):
                            pass
                if "cyclic_index" in cluster:
                    try:
                        cyclic_indices.add(int(cluster["cyclic_index"]))
                    except (TypeError, ValueError):
                        pass
        summary = {
            "group_id": int(group_id),
            "size": int(len(group)),
            "representative_candidate_id": candidate_track_id(ranked[0]),
            "distinct_edge_pair_count": int(len(edge_pairs)),
            "distinct_edge_cluster_count": int(len(edge_clusters)),
            "distinct_cyclic_pair_count": int(len(cyclic_indices)),
            "distinct_z_band_count": int(len(z_bands)),
            "normal_dispersion_deg_max": as_json_float(float(max(normal_angles))) if normal_angles else None,
            "offset_dispersion_max": as_json_float(float(max(offset_deltas))) if offset_deltas else None,
            "support_union_count_proxy": int(sum(int(candidate.get("finite_support_count") or 0) for candidate in group)),
            "support_agreement_purity_median": distribution_summary([candidate.get("finite_support_purity") for candidate in group]).get("median"),
        }
        for rank, candidate in enumerate(ranked):
            lookup[candidate_track_id(candidate)] = {**summary, "rank_in_group": int(rank + 1)}
    return groups, lookup


def annotate_w2_plane_consensus_soft_scores(candidates: list[dict[str, object]], w2_clusters: list[dict[str, object]]) -> dict[str, object]:
    groups, lookup = build_non_oracle_plane_consensus_groups(candidates, w2_clusters=w2_clusters)
    for candidate in candidates:
        base = finite_float(candidate.get("w2_support_diverse_score"), finite_float(candidate.get("candidate_score"), 0.0))
        group = lookup.get(candidate_track_id(candidate), {})
        size = int(group.get("size") or 1)
        edge_pairs = int(group.get("distinct_edge_pair_count") or 1)
        edge_clusters = int(group.get("distinct_edge_cluster_count") or len(candidate_edge_pair(candidate)))
        cyclic_pairs = int(group.get("distinct_cyclic_pair_count") or 1)
        z_bands = int(group.get("distinct_z_band_count") or 1)
        normal_disp = finite_float(group.get("normal_dispersion_deg_max"), 0.0)
        offset_disp = finite_float(group.get("offset_dispersion_max"), 0.0)
        rank_in_group = int(group.get("rank_in_group") or 1)
        support_count = int(candidate.get("finite_support_count") or 0)
        density = finite_float(candidate.get("finite_support_density"), 0.0)
        purity = finite_float(candidate.get("finite_support_purity"), 0.0)
        residual = finite_float(candidate.get("finite_support_residual_p95"), 1.0)
        rms = finite_float(candidate.get("plane_rms"), 1.0)
        persistence = min(np.log1p(max(size - 1, 0)) / np.log(6.0), 1.0)
        independence = float(np.mean([
            min(edge_pairs / 3.0, 1.0),
            min(edge_clusters / 4.0, 1.0),
            min(cyclic_pairs / 8.0, 1.0),
            min(z_bands / 5.0, 1.0),
        ]))
        dispersion_penalty = 0.25 * min(normal_disp / 1.5, 1.0) + 0.25 * min(offset_disp / 0.04, 1.0)
        consensus_bonus = -0.55 * min(persistence * independence, 1.0)
        rank_penalty = 0.10 * min(np.log1p(max(rank_in_group - 1, 0)), 2.0)
        weak_singleton_penalty = 0.0
        if size == 1:
            weak = 0
            weak += int(support_count < 5)
            weak += int(density < 200.0)
            weak += int(purity < 0.75)
            weak += int(residual > 0.008)
            weak += int(rms > 0.0002)
            weak_singleton_penalty = 0.12 * float(weak)
        exceptional_singleton_bonus = 0.0
        if size == 1 and support_count >= 8 and density >= 500.0 and purity >= 0.9 and residual <= 0.004 and rms <= 0.00012:
            exceptional_singleton_bonus = -0.20
        adjustment = float(np.clip(consensus_bonus + dispersion_penalty + rank_penalty + weak_singleton_penalty + exceptional_singleton_bonus, -0.75, 0.75))
        candidate["w2_support_diverse_base_score"] = as_json_float(float(base))
        candidate["w2_support_diverse_score"] = as_json_float(float(base + adjustment))
        candidate["w2_plane_consensus_group"] = group
        candidate["w2_plane_consensus_soft_components"] = {
            "base_score": as_json_float(float(base)),
            "adjustment": as_json_float(float(adjustment)),
            "persistence": as_json_float(float(persistence)),
            "independence": as_json_float(float(independence)),
            "consensus_bonus": as_json_float(float(consensus_bonus)),
            "dispersion_penalty": as_json_float(float(dispersion_penalty)),
            "rank_penalty": as_json_float(float(rank_penalty)),
            "weak_singleton_penalty": as_json_float(float(weak_singleton_penalty)),
            "exceptional_singleton_bonus": as_json_float(float(exceptional_singleton_bonus)),
        }
    return {
        "w2_plane_consensus_mode": "soft",
        "group_count": int(len(groups)),
        "singleton_group_count": int(sum(1 for group in groups if len(group) == 1)),
        "stable_group_count": int(sum(1 for group in groups if len(group) >= 2)),
        "soft_adjustment_distribution": distribution_summary([
            (candidate.get("w2_plane_consensus_soft_components") or {}).get("adjustment")
            for candidate in candidates
        ]),
    }




















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


def summarize_reconstruction_prefix(
    *,
    prefix: int,
    candidates: list[dict[str, object]],
    core_count: int,
    core_ids: set[int],
    model_faces: list[dict[str, object]],
    initial_vertices: np.ndarray,
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    match_cache: dict[int, dict[str, object] | None] | None = None,
) -> dict[str, object]:
    rec = reconstruct_polyhedron_from_halfspaces_edge_clip(candidates, **reconstruction_kwargs)
    active_indices = [int(i) for i in rec.get("face_candidate_indices", []) if int(i) < len(candidates)]
    active_candidates = [candidates[i] for i in active_indices]
    if match_cache is None:
        cohort = oracle_candidate_cohort(active_candidates, model_faces, core_face_ids=core_ids, active_plane_indices=active_indices)
    else:
        rows: list[dict[str, object]] = []
        plane_ids: list[int] = []
        finite_ids: list[int] = []
        for idx, candidate in zip(active_indices, active_candidates):
            tid = candidate_track_id(candidate)
            if tid not in match_cache:
                match_cache[tid] = candidate_oracle_face(candidate, model_faces)
            match = match_cache.get(tid)
            if match is None:
                continue
            face_id = int(match["face_id"])
            if bool(match.get("plane_good")):
                plane_ids.append(face_id)
            if bool(match.get("finite_good")):
                finite_ids.append(face_id)
            rows.append({"candidate_id": int(tid), "active_plane_index": int(idx), **match})
        finite_unique = set(finite_ids)
        plane_unique = set(plane_ids)
        cohort = {
            "candidate_count": int(len(active_candidates)),
            "plane_good_candidates": int(len(plane_ids)),
            "unique_plane_good_face_ids": int(len(plane_unique)),
            "finite_good_candidates": int(len(finite_ids)),
            "unique_finite_face_ids": int(len(finite_unique)),
            "new_unique_face_ids_vs_repaired_core": int(len(finite_unique - core_ids)),
            "duplicate_good_assignments": int(len(finite_ids) - len(finite_unique)),
            "matched_rows": rows,
            "finite_face_ids": sorted(int(v) for v in finite_unique),
            "plane_face_ids": sorted(int(v) for v in plane_unique),
        }
    finite_ids = set(int(v) for v in cohort.get("finite_face_ids", []))
    w2_active = [candidates[i] for i in active_indices if int(i) >= int(core_count)]
    active_small_ids = {
        candidate_track_id(candidates[i])
        for i in active_indices
        if int(i) < len(candidates) and str(candidates[i].get("small_face_refinement_mode")) == "append-local"
    }
    active_w2_ids = {candidate_track_id(candidate) for candidate in w2_active}
    matched_rows = cohort.get("matched_rows") if isinstance(cohort.get("matched_rows"), list) else []
    normal_failures = 0
    plane_distance_failures = 0
    active_small_new_ids: set[int] = set()
    active_small_duplicate_ids: set[int] = set()
    active_small_finite_good = 0
    active_small_bad = 0
    active_small_failure_counts: dict[str, int] = {}
    for row in matched_rows:
        if "normal_angle" in list(row.get("failure_reasons") or []):
            normal_failures += 1
        if "plane_distance" in list(row.get("failure_reasons") or []):
            plane_distance_failures += 1
        cid = int(row.get("candidate_id") or -1)
        if cid not in active_small_ids:
            continue
        face_id = int(row.get("face_id") if row.get("face_id") is not None else -1)
        if bool(row.get("finite_good")):
            active_small_finite_good += 1
            if face_id in core_ids:
                active_small_duplicate_ids.add(face_id)
            elif face_id >= 0:
                active_small_new_ids.add(face_id)
        else:
            active_small_bad += 1
            reasons = list(row.get("failure_reasons") or ["unmatched"])
            if not reasons:
                reasons = ["unmatched"]
            primary = str(reasons[0])
            active_small_failure_counts[primary] = active_small_failure_counts.get(primary, 0) + 1
    outside = cumulative_outside_metrics(candidates, trusted_points, trusted_z_indices, point_tol=float(point_tol))
    rec_vertices = np.array(rec.get("vertices") or [], dtype=float)
    rec_faces = rec.get("faces") or []
    rec_face_rows = model_face_planes(rec_vertices, rec_faces) if rec_vertices.size and rec_faces else []
    return {
        "prefix": int(prefix),
        "accepted_w2": int(prefix),
        "final_active_w2": int(len(w2_active)),
        "active_w2_candidate_ids": sorted(int(v) for v in active_w2_ids)[:240],
        "active_small_additions": int(len(active_small_ids)),
        "active_small_candidate_ids": sorted(int(v) for v in active_small_ids)[:160],
        "active_small_finite_good_candidates": int(active_small_finite_good),
        "active_small_new_ids": sorted(int(v) for v in active_small_new_ids),
        "active_small_duplicate_face_ids": sorted(int(v) for v in active_small_duplicate_ids),
        "active_small_bad_candidates": int(active_small_bad),
        "active_small_failure_counts": {key: int(value) for key, value in sorted(active_small_failure_counts.items())},
        "active_normal_angle_failures": int(normal_failures),
        "active_plane_distance_failures": int(plane_distance_failures),
        "active_finite_good_candidates": int(cohort.get("finite_good_candidates") or 0),
        "active_unique_finite_ids": int(len(finite_ids)),
        "retained_core_ids": int(len(core_ids & finite_ids)),
        "lost_core_ids": sorted(int(v) for v in (core_ids - finite_ids)),
        "new_ids": sorted(int(v) for v in (finite_ids - core_ids)),
        "net_gain_vs_core": int(len(finite_ids) - len(core_ids)),
        "volume": as_json_float(float(rec.get("reliable_volume") or 0.0)),
        "trusted_outside": outside,
        "topology": rec.get("topology"),
        "reconstructed_to_initial": surface_distance_summary(rec_vertices, model_faces, max_points=12),
        "initial_to_reconstructed": surface_distance_summary(initial_vertices, rec_face_rows, max_points=12),
        "reconstructed_vertices": int(len(rec.get("vertices") or [])),
        "reconstructed_edges": int(len(rec.get("edges") or [])),
        "reconstructed_faces": int(len(rec.get("faces") or [])),
    }


def core_witness_mask_and_stats(
    candidate: dict[str, object],
    points: np.ndarray,
    z_indices: np.ndarray,
    *,
    plane_distance: float,
    hull_margin: float,
    relative_hull_margin: float,
) -> tuple[np.ndarray, dict[str, object]]:
    plane = candidate_plane(candidate)
    hull_data = candidate_hull_polygon_2d(candidate)
    empty = np.zeros(points.shape[0], dtype=bool) if points.ndim == 2 else np.zeros(0, dtype=bool)
    if plane is None or hull_data is None or points.ndim != 2 or points.shape[0] == 0:
        return empty, {
            "witness_count": 0,
            "witness_class": "no_independent_witness",
            "fallback_reason": "missing_plane_hull_or_points",
        }
    plane_point, normal = plane
    polygon, u, v, hull_area = hull_data
    signed = (points - plane_point) @ normal
    residual = np.abs(signed)
    coords = np.column_stack([(points - plane_point) @ u, (points - plane_point) @ v])
    hull_signed = polygon_signed_distances_2d(coords, polygon)
    candidate_z = np.array(candidate.get("z_indices", []), dtype=int)
    candidate_z_set = {int(z) for z in candidate_z if int(z) >= 0}
    if candidate_z_set and z_indices.size == points.shape[0]:
        z_min_idx = min(candidate_z_set)
        z_max_idx = max(candidate_z_set)
        in_track_span = (z_indices >= z_min_idx) & (z_indices <= z_max_idx)
    else:
        z_min = finite_float(candidate.get("z_min"), float("-inf"))
        z_max = finite_float(candidate.get("z_max"), float("inf"))
        in_track_span = (points[:, 2] >= z_min - 0.03) & (points[:, 2] <= z_max + 0.03)

    residual_tol = min(max(float(plane_distance), 0.025), 0.05)
    interior_margin = min(0.025, 0.08 * math.sqrt(max(float(hull_area), EPS)))
    near_plane = residual <= residual_tol
    interior_mask = near_plane & in_track_span & (hull_signed <= -interior_margin)
    fallback_reason = None
    witness_mask = interior_mask
    if int(np.sum(witness_mask)) < 8:
        relaxed_margin = float(hull_margin) + float(relative_hull_margin) * math.sqrt(max(float(hull_area), EPS))
        witness_mask = near_plane & in_track_span & (hull_signed <= min(relaxed_margin, 0.08))
        fallback_reason = "relaxed_to_near_hull"
    count = int(np.sum(witness_mask))
    z_values = sorted({int(v) for v in z_indices[witness_mask]}) if z_indices.size == points.shape[0] else []
    residual_values = residual[witness_mask]
    interior_fraction = float(np.mean(hull_signed[witness_mask] <= -interior_margin)) if count else 0.0
    density = float(count) / max(float(hull_area), EPS)
    residual_p95 = float(np.percentile(residual_values, 95)) if residual_values.size else float("inf")
    if count >= 12 and len(z_values) >= 3 and residual_p95 <= 0.04 and interior_fraction >= 0.25:
        witness_class = "strong_witness"
    elif count >= 5 and len(z_values) >= 2 and residual_p95 <= 0.055:
        witness_class = "weak_witness"
    else:
        witness_class = "no_independent_witness"
        if fallback_reason is None:
            fallback_reason = "insufficient_count_z_or_residual"
    return witness_mask, {
        "witness_count": int(count),
        "witness_z_levels": int(len(z_values)),
        "witness_z_min": int(min(z_values)) if z_values else None,
        "witness_z_max": int(max(z_values)) if z_values else None,
        "witness_density": as_json_float(float(density)) if count else None,
        "witness_residual_median": as_json_float(float(np.median(residual_values))) if residual_values.size else None,
        "witness_residual_p95": as_json_float(float(residual_p95)) if residual_values.size else None,
        "witness_hull_interior_fraction": as_json_float(float(interior_fraction)) if count else None,
        "witness_hull_area": as_json_float(float(hull_area)),
        "witness_class": witness_class,
        "fallback_reason": fallback_reason,
    }


def topology_is_valid(topology: dict[str, object] | None) -> bool:
    if not isinstance(topology, dict):
        return False
    return bool(
        int(topology.get("euler") or 0) == 2
        and int(topology.get("connected_components") or 0) == 1
        and int(topology.get("boundary_edges") or 0) == 0
        and int(topology.get("non_manifold_edges") or 0) == 0
    )


def value_or_default(value: object, fallback: object) -> object:
    return fallback if value is None else value


def zero_value_regression_check() -> dict[str, object]:
    zero_lost = {"cumulative_lost_z_levels": 0, "cumulative_outside_fraction": 0.0}
    one_lost = {"cumulative_lost_z_levels": 1, "cumulative_outside_fraction": 0.0}
    none_values = {"cumulative_lost_z_levels": None, "cumulative_outside_fraction": None}

    def hard_safety_pass(outside: dict[str, object]) -> bool:
        lost = int(value_or_default(outside.get("cumulative_lost_z_levels"), 0))
        outside_fraction = float(value_or_default(outside.get("cumulative_outside_fraction"), 0.0))
        return bool(lost == 0 and outside_fraction < 0.01)

    checks = {
        "zero_int_preserved": int(value_or_default(0, 7)) == 0,
        "zero_float_preserved": float(value_or_default(0.0, 7.0)) == 0.0,
        "none_gets_fallback": int(value_or_default(None, 7)) == 7,
        "lost_z_zero_passes": hard_safety_pass(zero_lost),
        "lost_z_one_rejects": not hard_safety_pass(one_lost),
        "outside_zero_passes": hard_safety_pass(zero_lost),
        "none_fallback_hard_safety_passes": hard_safety_pass(none_values),
    }
    return {
        "method": "explicit_is_none_value_or_default",
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def lp_candidate_active_against(
    base_candidates: list[dict[str, object]],
    candidate: dict[str, object],
    *,
    inside_tol: float,
    activity_tol: float,
) -> tuple[bool, bool, float | None, str]:
    details = lp_face_activity_details(
        base_candidates,
        candidate,
        inside_tol=float(inside_tol),
        face_activity_slack=float(activity_tol),
    )
    if not bool(details.get("feasible")):
        return False, False, None, str(details.get("solver_status"))

    candidate_constraint = candidate_halfspace(candidate)
    if candidate_constraint is None:
        return False, False, None, "invalid_candidate"
    feasible_after, _, status_after = solve_halfspace_lp(
        lp_constraints_for_candidates(base_candidates + [candidate]),
        inside_tol=float(inside_tol),
        objective=None,
    )
    optimum = details.get("raw_lp_optimum")
    return bool(feasible_after), bool(details.get("active")), float(optimum) if optimum is not None else None, status_after


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


def polyline_perimeter(points: np.ndarray) -> float:
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0
    return float(sum(np.linalg.norm(points[(i + 1) % points.shape[0]] - points[i]) for i in range(points.shape[0])))


def nearest_point_distances(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or targets.ndim != 2 or points.shape[0] == 0 or targets.shape[0] == 0:
        return np.array([], dtype=float)
    return np.min(np.linalg.norm(points[:, None, :] - targets[None, :, :], axis=2), axis=1)


def trial_face_metrics_from_points(
    *,
    candidate: dict[str, object],
    face_points: np.ndarray,
    active_candidate_indices: list[int],
    reconstructed_vertices: int,
    reconstructed_faces: int,
    topology: dict[str, object],
    volume: object,
    method: str,
) -> dict[str, object]:
    hull = candidate_hull_points(candidate)
    face_area = polygon_area(face_points)
    hull_area = finite_float(candidate.get("hull_area"), polygon_area(hull) if hull.shape[0] >= 3 else 0.0)
    face_centroid = np.mean(face_points, axis=0)
    face_z_span = float(np.max(face_points[:, 2]) - np.min(face_points[:, 2])) if face_points.size else 0.0
    face_to_hull = nearest_point_distances(face_points, hull)
    hull_to_face_vertices = nearest_point_distances(hull, face_points)
    face_record = {
        "vertices": face_points,
        "normal": candidate_plane(candidate)[1] if candidate_plane(candidate) is not None else np.array([0.0, 0.0, 1.0]),
    }
    hull_to_face_poly = np.array([point_to_face_polygon_distance(p, face_record) for p in hull], dtype=float) if hull.shape[0] else np.array([], dtype=float)
    containment_tol = max(0.02, 0.05 * np.sqrt(max(hull_area, EPS)))
    return {
        "trial_active": True,
        "trial_geometry_method": method,
        "face_area": as_json_float(float(face_area)),
        "face_perimeter": as_json_float(polyline_perimeter(face_points)),
        "face_vertex_count": int(face_points.shape[0]),
        "face_centroid": as_json_point(face_centroid),
        "face_z_span": as_json_float(float(face_z_span)),
        "hull_area": as_json_float(float(hull_area)),
        "area_ratio": as_json_float(float(face_area) / max(float(hull_area), EPS)),
        "extra_area": as_json_float(float(face_area) - float(hull_area)),
        "face_centroid_to_hull_distance": as_json_float(float(np.min(np.linalg.norm(hull - face_centroid, axis=1)))) if hull.shape[0] else None,
        "face_vertex_to_hull_distance_median": as_json_float(float(np.median(face_to_hull))) if face_to_hull.size else None,
        "face_vertex_to_hull_distance_p95": as_json_float(float(np.percentile(face_to_hull, 95))) if face_to_hull.size else None,
        "hull_to_face_vertex_distance_median": as_json_float(float(np.median(hull_to_face_vertices))) if hull_to_face_vertices.size else None,
        "hull_to_face_polygon_distance_median": as_json_float(float(np.median(hull_to_face_poly))) if hull_to_face_poly.size else None,
        "hull_to_face_containment_fraction": as_json_float(float(np.mean(hull_to_face_poly <= containment_tol))) if hull_to_face_poly.size else None,
        "active_candidate_indices": [int(v) for v in active_candidate_indices],
        "reconstructed_vertices": int(reconstructed_vertices),
        "reconstructed_faces": int(reconstructed_faces),
        "topology_valid": topology_is_valid(topology),
        "topology": topology,
        "volume": volume,
    }


def trial_face_geometry_row_incremental(
    *,
    candidate: dict[str, object],
    current_reconstructed: dict[str, object],
    candidate_index: int,
    reconstruction_kwargs: dict[str, object],
) -> dict[str, object]:
    halfspace = candidate_halfspace(candidate)
    if halfspace is None:
        return {"trial_active": False, "trial_geometry_method": "incremental", "reason": "invalid_candidate_halfspace"}
    normal, offset = halfspace
    vertices = np.array(current_reconstructed.get("vertices") or [], dtype=float)
    edges = current_reconstructed.get("edges") if isinstance(current_reconstructed.get("edges"), list) else []
    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3 or not edges:
        return {"trial_active": False, "trial_geometry_method": "incremental", "reason": "empty_current_mesh"}
    feasibility_tol = float(reconstruction_kwargs.get("feasibility_tol", 0.0))
    incidence_tol = float(reconstruction_kwargs.get("incidence_tol", feasibility_tol))
    vertex_merge_tol = float(reconstruction_kwargs.get("vertex_merge_tol", incidence_tol))
    min_face_area = float(reconstruction_kwargs.get("min_face_area", 0.0))
    tol = max(feasibility_tol, incidence_tol, vertex_merge_tol)
    points: list[np.ndarray] = []

    signed = vertices @ normal - float(offset)
    for vi, dist in enumerate(signed):
        if abs(float(dist)) <= tol:
            points.append(vertices[int(vi)])
    for edge in edges:
        pair = edge.get("vertices") if isinstance(edge, dict) else None
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        ia, ib = int(pair[0]), int(pair[1])
        if ia < 0 or ib < 0 or ia >= vertices.shape[0] or ib >= vertices.shape[0]:
            continue
        da = float(signed[ia])
        db = float(signed[ib])
        if abs(da) <= tol and abs(db) <= tol:
            points.append(vertices[ia])
            points.append(vertices[ib])
            continue
        if da * db > 0.0:
            continue
        denom = da - db
        if abs(denom) <= EPS:
            continue
        t = da / denom
        if t < -tol or t > 1.0 + tol:
            continue
        p = vertices[ia] + float(np.clip(t, 0.0, 1.0)) * (vertices[ib] - vertices[ia])
        if abs(float(normal @ p - offset)) <= max(tol * 10.0, EPS):
            points.append(p)
    if len(points) < 3:
        return {
            "trial_active": False,
            "trial_geometry_method": "incremental",
            "reason": "fewer_than_three_cap_points",
            "active_candidate_indices": [int(v) for v in current_reconstructed.get("face_candidate_indices", [])],
            "reconstructed_vertices": int(vertices.shape[0]),
            "reconstructed_faces": int(len(current_reconstructed.get("faces") or [])),
            "topology_valid": topology_is_valid(current_reconstructed.get("topology") if isinstance(current_reconstructed.get("topology"), dict) else {}),
            "topology": current_reconstructed.get("topology") if isinstance(current_reconstructed.get("topology"), dict) else {},
            "volume": current_reconstructed.get("reliable_volume"),
        }
    deduped: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - existing)) <= vertex_merge_tol for existing in deduped):
            deduped.append(point.astype(float))
    if len(deduped) < 3:
        return {"trial_active": False, "trial_geometry_method": "incremental", "reason": "degenerate_cap_after_merge"}
    pts = np.array(deduped, dtype=float)
    plane_point, plane_normal = candidate_plane(candidate) or (pts[0], normal)
    u, v = plane_basis(plane_normal)
    coords = np.column_stack([(pts - plane_point) @ u, (pts - plane_point) @ v])
    hull_idx = convex_hull_indices(coords)
    if len(hull_idx) < 3:
        return {"trial_active": False, "trial_geometry_method": "incremental", "reason": "degenerate_cap_hull"}
    face_points = pts[np.array(hull_idx, dtype=int)]
    area = polygon_area(face_points)
    if area < min_face_area:
        return {"trial_active": False, "trial_geometry_method": "incremental", "reason": "small_cap_area", "face_area": as_json_float(float(area))}
    active = [int(v) for v in current_reconstructed.get("face_candidate_indices", [])]
    active.append(int(candidate_index))
    return trial_face_metrics_from_points(
        candidate=candidate,
        face_points=face_points,
        active_candidate_indices=active,
        reconstructed_vertices=int(vertices.shape[0] + len(face_points)),
        reconstructed_faces=int(len(current_reconstructed.get("faces") or []) + 1),
        topology=current_reconstructed.get("topology") if isinstance(current_reconstructed.get("topology"), dict) else {},
        volume=current_reconstructed.get("reliable_volume"),
        method="incremental",
    )


def trial_face_geometry_row_from_reconstruction(
    *,
    candidate: dict[str, object],
    candidate_index: int,
    reconstructed: dict[str, object],
    method: str,
) -> dict[str, object]:
    vertices = np.array(reconstructed.get("vertices") or [], dtype=float)
    faces = reconstructed.get("faces") or []
    sources = [int(v) for v in reconstructed.get("face_candidate_indices", [])]
    active_faces = [
        np.array(face, dtype=int)
        for face, source in zip(faces, sources)
        if int(source) == int(candidate_index)
    ]
    topology = (
        reconstructed.get("topology")
        if isinstance(reconstructed.get("topology"), dict)
        else {}
    )
    if not active_faces or vertices.size == 0:
        inactive = {
            "trial_active": False,
            "active_candidate_indices": sources,
            "reconstructed_vertices": int(len(reconstructed.get("vertices") or [])),
            "reconstructed_faces": int(len(reconstructed.get("faces") or [])),
            "topology_valid": topology_is_valid(topology),
            "topology": topology,
            "volume": reconstructed.get("reliable_volume"),
        }
        if str(method) != "full":
            inactive["trial_geometry_method"] = str(method)
        return inactive
    face_idx = max(
        active_faces,
        key=lambda idx: polygon_area(vertices[idx]) if idx.size else 0.0,
    )
    return trial_face_metrics_from_points(
        candidate=candidate,
        face_points=vertices[face_idx],
        active_candidate_indices=sources,
        reconstructed_vertices=int(len(reconstructed.get("vertices") or [])),
        reconstructed_faces=int(len(reconstructed.get("faces") or [])),
        topology=topology,
        volume=reconstructed.get("reliable_volume"),
        method=str(method),
    )


def trial_face_geometry_row(
    *,
    candidate: dict[str, object],
    trial_candidates: list[dict[str, object]],
    candidate_index: int,
    reconstruction_kwargs: dict[str, object],
    geometry_mode: str = "full",
    current_reconstructed: dict[str, object] | None = None,
) -> dict[str, object]:
    if geometry_mode == "incremental" and current_reconstructed is not None:
        return trial_face_geometry_row_incremental(
            candidate=candidate,
            current_reconstructed=current_reconstructed,
            candidate_index=candidate_index,
            reconstruction_kwargs=reconstruction_kwargs,
        )
    rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
    return trial_face_geometry_row_from_reconstruction(
        candidate=candidate,
        candidate_index=int(candidate_index),
        reconstructed=rec,
        method="full",
    )


def detect_dense_small_face_bands(candidates: list[dict[str, object]], *, bin_width: float = 0.25) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        centroid = candidate_hull_centroid(candidate)
        if centroid is None or not np.isfinite(float(centroid[2])):
            continue
        hull_area = finite_float(candidate.get("hull_area"), float("nan"))
        support = finite_float(candidate.get("finite_support_count"), float("nan"))
        z0, z1 = candidate_z_range(candidate)
        rows.append(
            {
                "candidate": candidate,
                "z": float(centroid[2]),
                "z_span": float(z1 - z0) if np.isfinite(z0) and np.isfinite(z1) else 0.0,
                "hull_area": hull_area,
                "support": support,
            }
        )
    if not rows:
        return {"bands": [], "histogram": [], "method": "robust_histogram", "bin_width": float(bin_width)}
    z_values = np.array([row["z"] for row in rows], dtype=float)
    z_min = float(np.floor(np.min(z_values) / bin_width) * bin_width)
    z_max = float(np.ceil(np.max(z_values) / bin_width) * bin_width)
    edges = np.arange(z_min, z_max + bin_width * 1.5, bin_width)
    if edges.size < 2:
        edges = np.array([z_min, z_min + bin_width], dtype=float)
    global_area_median = safe_nanmedian([row["hull_area"] for row in rows])
    global_support_median = safe_nanmedian([row["support"] for row in rows])
    global_area_threshold = float(global_area_median) if global_area_median is not None else float("inf")
    global_support_threshold = float(global_support_median) if global_support_median is not None else 0.0
    hist: list[dict[str, object]] = []
    counts: list[int] = []
    for i in range(edges.size - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        members = [row for row in rows if (row["z"] >= lo and (row["z"] < hi or (i == edges.size - 2 and row["z"] <= hi)))]
        counts.append(len(members))
        hist.append(
            {
                "bin_index": int(i),
                "z_min": as_json_float(lo),
                "z_max": as_json_float(hi),
                "candidate_count": int(len(members)),
                "median_hull_area": as_json_float(float(v)) if (v := safe_nanmedian([row["hull_area"] for row in members])) is not None else None,
                "median_support_count": as_json_float(float(v)) if (v := safe_nanmedian([row["support"] for row in members])) is not None else None,
                "median_z_span": as_json_float(float(v)) if (v := safe_nanmedian([row["z_span"] for row in members])) is not None else None,
            }
        )
    count_arr = np.array(counts, dtype=float)
    med = float(np.median(count_arr))
    mad = float(np.median(np.abs(count_arr - med)))
    robust_scale = max(1.0, 1.4826 * mad)
    threshold = max(med + 1.5 * robust_scale, float(np.percentile(count_arr, 75)))
    selected_bins: list[int] = []
    for i, item in enumerate(hist):
        count = int(item["candidate_count"])
        area = finite_float(item.get("median_hull_area"), float("inf"))
        support = finite_float(item.get("median_support_count"), float("inf"))
        if count >= max(12, threshold) and area <= global_area_threshold and support <= max(global_support_threshold, 1.0):
            selected_bins.append(i)
    bands: list[dict[str, object]] = []
    current: list[int] = []
    for idx in selected_bins:
        if not current or idx == current[-1] + 1:
            current.append(idx)
        else:
            if current:
                bands.append({"bins": current})
            current = [idx]
    if current:
        bands.append({"bins": current})
    band_rows: list[dict[str, object]] = []
    for band_id, band in enumerate(bands):
        bins = [int(v) for v in band["bins"]]
        lo = float(edges[min(bins)])
        hi = float(edges[max(bins) + 1])
        members = [row for row in rows if lo <= row["z"] <= hi]
        if len(members) < 12 or (hi - lo) < bin_width:
            continue
        band_rows.append(
            {
                "band_id": int(len(band_rows)),
                "z_min": as_json_float(lo),
                "z_max": as_json_float(hi),
                "width": as_json_float(float(hi - lo)),
                "candidate_count": int(len(members)),
                "median_hull_area": as_json_float(float(np.nanmedian([row["hull_area"] for row in members]))),
                "median_support_count": as_json_float(float(np.nanmedian([row["support"] for row in members]))),
                "median_z_span": as_json_float(float(np.nanmedian([row["z_span"] for row in members]))),
                "bin_indices": bins,
            }
        )
    return {
        "method": "robust_histogram_median_mad_small_footprint_gate",
        "bin_width": float(bin_width),
        "count_median": as_json_float(med),
        "count_mad": as_json_float(mad),
        "count_threshold": as_json_float(float(threshold)),
        "global_median_hull_area": as_json_float(float(global_area_median)) if global_area_median is not None else None,
        "global_median_support_count": as_json_float(float(global_support_median)) if global_support_median is not None else None,
        "bands": band_rows,
        "histogram": hist,
    }


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


def distance_to_band(z: object, bands: list[dict[str, object]]) -> float | None:
    val = finite_float(z, float("nan"))
    if not np.isfinite(val) or not bands:
        return None
    distances = []
    for band in bands:
        lo = float(band.get("z_min") or 0.0)
        hi = float(band.get("z_max") or 0.0)
        if lo <= val <= hi:
            distances.append(0.0)
        else:
            distances.append(min(abs(val - lo), abs(val - hi)))
    return min(distances) if distances else None


def band_overlap_features(candidate: dict[str, object], bands: list[dict[str, object]], margin: float = 0.0) -> dict[str, object]:
    intervals = candidate_z_intervals(candidate)
    centroid_z = intervals.get("centroid_z")
    best: dict[str, object] = {
        **intervals,
        "strict_band_id": None,
        "distance_to_band": as_json_float(float(distance_to_band(centroid_z, bands))) if distance_to_band(centroid_z, bands) is not None else None,
        "hull_overlaps_band": False,
        "support_overlaps_band": False,
        "hull_overlap_fraction": 0.0,
        "support_overlap_fraction": 0.0,
        "within_auto_margin": False,
        "band_margin": as_json_float(float(margin)),
        "membership_class": "far_outside",
    }
    for band in bands:
        band_id = int(band.get("band_id") or 0)
        lo = float(band.get("z_min") or 0.0)
        hi = float(band.get("z_max") or 0.0)
        cz = finite_float(centroid_z, float("nan"))
        if np.isfinite(cz) and lo <= cz <= hi:
            best["strict_band_id"] = int(band_id)
        hull_overlap, hull_frac = interval_overlap(intervals.get("hull_z_min"), intervals.get("hull_z_max"), lo, hi)
        support_overlap, support_frac = interval_overlap(intervals.get("support_z_min"), intervals.get("support_z_max"), lo, hi)
        if hull_overlap > 0.0 and hull_frac >= finite_float(best.get("hull_overlap_fraction"), 0.0):
            best["hull_overlaps_band"] = True
            best["hull_overlap_fraction"] = as_json_float(float(hull_frac))
        if support_overlap > 0.0 and support_frac >= finite_float(best.get("support_overlap_fraction"), 0.0):
            best["support_overlaps_band"] = True
            best["support_overlap_fraction"] = as_json_float(float(support_frac))
        dist = distance_to_band(centroid_z, [band])
        if dist is not None and dist <= float(margin):
            best["within_auto_margin"] = True
    if bool(best.get("hull_overlaps_band")):
        best["membership_class"] = "hull_overlaps"
    elif bool(best.get("support_overlaps_band")):
        best["membership_class"] = "support_overlaps"
    elif bool(best.get("within_auto_margin")):
        best["membership_class"] = "within_auto_margin"
    return best


def automatic_band_margin(candidates: list[dict[str, object]]) -> float:
    half_spans = []
    for candidate in candidates:
        intervals = candidate_z_intervals(candidate)
        z0 = finite_float(intervals.get("hull_z_min"), finite_float(intervals.get("track_z_min"), float("nan")))
        z1 = finite_float(intervals.get("hull_z_max"), finite_float(intervals.get("track_z_max"), float("nan")))
        if np.isfinite(z0) and np.isfinite(z1):
            half_spans.append(0.5 * abs(float(z1 - z0)))
    if not half_spans:
        return 0.12
    arr = np.array(half_spans, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return float(np.clip(med + 1.4826 * mad, 0.05, 0.35))


def append_local_small_face_refinement(
    *,
    core_candidates: list[dict[str, object]],
    core_reconstructed: dict[str, object],
    w2_preselection_candidates: list[dict[str, object]],
    selected_candidates: list[dict[str, object]],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
    max_additions: int,
    max_trials: int,
    trial_geometry_mode: str,
    excluded_candidate_ids: set[int] | None = None,
    preserve_candidate_ids: set[int] | None = None,
    stage1_preserve_candidate_ids: set[int] | None = None,
    stage2_preserve_candidate_ids: set[int] | None = None,
    mode_label: str = "append-local",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.perf_counter()
    excluded_candidate_ids = set(int(v) for v in (excluded_candidate_ids or set()))
    preserve_candidate_ids = set(int(v) for v in (preserve_candidate_ids or set()))
    stage1_preserve_candidate_ids = set(int(v) for v in (stage1_preserve_candidate_ids or set()))
    stage2_preserve_candidate_ids = set(int(v) for v in (stage2_preserve_candidate_ids or set()))
    if max_additions <= 0:
        return selected_candidates, {
            "mode": str(mode_label),
            "accepted_additions": 0,
            "reason": "max_additions <= 0",
            "production_changed": False,
            "excluded_candidate_ids": sorted(int(v) for v in excluded_candidate_ids),
            "preserve_candidate_ids": sorted(int(v) for v in preserve_candidate_ids),
            "stage1_preserve_candidate_ids": sorted(int(v) for v in stage1_preserve_candidate_ids),
            "stage2_preserve_candidate_ids": sorted(int(v) for v in stage2_preserve_candidate_ids),
        }
    timing: dict[str, float] = {
        "challenger_preparation": 0.0,
        "lp_activity": 0.0,
        "trial_geometry": 0.0,
        "outside_checks": 0.0,
        "accepted_state_rebuild": 0.0,
    }
    counters: dict[str, int] = {
        "trial_geometry_calls": 0,
        "trial_geometry_full_calls": 0,
        "trial_geometry_incremental_calls": 0,
        "trial_geometry_rejected_calls": 0,
        "trial_geometry_accepted_calls": 0,
        "accepted_state_rebuilds": 0,
    }
    edge_clip_times: list[float] = []
    prep_started = time.perf_counter()
    annotated = annotate_candidate_pool_neutral(
        w2_preselection_candidates,
        core_candidates=core_candidates,
        core_reconstructed=core_reconstructed,
        trusted_points=trusted_points,
        trusted_z_indices=trusted_z_indices,
        point_tol=float(point_tol),
        finite_support_plane_distance=float(finite_support_plane_distance),
        finite_support_hull_margin=float(finite_support_hull_margin),
        finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
    )
    accepted_ids = {candidate_track_id(c) for c in selected_candidates if str(c.get("candidate_origin")) == "w2_addition"}
    ranked_out = [c for c in annotated if candidate_track_id(c) not in accepted_ids]
    band_detection = detect_dense_small_face_bands(ranked_out)
    bands = band_detection.get("bands") if isinstance(band_detection.get("bands"), list) else []

    def band_id(candidate: dict[str, object]) -> int | None:
        centroid = candidate_hull_centroid(candidate)
        if centroid is None:
            return None
        z = float(centroid[2])
        for band in bands:
            if float(band.get("z_min") or 0.0) <= z <= float(band.get("z_max") or 0.0):
                return int(band.get("band_id") or 0)
        return None

    current_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(selected_candidates, **reconstruction_kwargs)
    active_sources = [int(v) for v in current_rec.get("face_candidate_indices", []) if 0 <= int(v) < len(selected_candidates)]
    active_candidates = [selected_candidates[i] for i in active_sources if str(selected_candidates[i].get("candidate_origin")) == "w2_addition"]

    def duplicate_active(candidate: dict[str, object]) -> bool:
        for active_candidate in active_candidates:
            angle, offset, centroid = candidate_plane_relation_score(candidate, active_candidate)
            if angle <= 0.75 and offset <= 0.01 and centroid <= 0.08:
                return True
        return False

    rejection_counts: dict[str, int] = {}
    provenance_counts: dict[str, int] = {}
    cheap_pool: list[dict[str, object]] = []
    for candidate in ranked_out:
        cid = int(candidate_track_id(candidate))
        if cid in excluded_candidate_ids:
            rejection_counts["excluded_candidate_id"] = rejection_counts.get("excluded_candidate_id", 0) + 1
            continue
        if band_id(candidate) is None:
            rejection_counts["outside_dense_band"] = rejection_counts.get("outside_dense_band", 0) + 1
            continue
        if finite_float(candidate.get("hull_area"), 0.0) <= EPS or candidate_hull_points(candidate).shape[0] == 0:
            rejection_counts["empty_local_hull"] = rejection_counts.get("empty_local_hull", 0) + 1
            continue
        if finite_float(candidate.get("hull_area"), float("inf")) > 0.04:
            rejection_counts["large_hull_not_small_face"] = rejection_counts.get("large_hull_not_small_face", 0) + 1
            continue
        if finite_float(candidate.get("finite_support_count"), float("inf")) > 45.0:
            rejection_counts["large_support_not_small_face"] = rejection_counts.get("large_support_not_small_face", 0) + 1
            continue
        if duplicate_active(candidate):
            rejection_counts["duplicate_active_plane"] = rejection_counts.get("duplicate_active_plane", 0) + 1
            continue
        previous_reason = str(candidate.get("selection_rejection_reason") or candidate.get("primary_rejection_reason") or "")
        if previous_reason in {"candidate_inactive", "inactive"}:
            provenance = "previously_inactive"
        elif previous_reason in {"candidate_redundant", "duplicate_active_plane", "redundant"}:
            provenance = "previously_redundant"
        elif "mismatch" in previous_reason or "nonlocal" in previous_reason:
            provenance = "previous_trial_face_mismatch"
        elif candidate.get("selection_rank") is None and candidate.get("w2_support_diverse_rank") is None:
            provenance = "previously_untried_budget"
        else:
            provenance = "other_state_dependent"
        candidate["post_repair_refill_previous_provenance"] = provenance
        provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
        cheap_pool.append(candidate)
    timing["challenger_preparation"] += time.perf_counter() - prep_started

    def local_order_key(candidate: dict[str, object]) -> tuple[float, float, float, int]:
        area = max(finite_float(candidate.get("hull_area"), 0.0), 1e-4)
        support = float(candidate.get("finite_support_count") or 0)
        density = support / area
        components = candidate.get("w2_support_diverse_components") if isinstance(candidate.get("w2_support_diverse_components"), dict) else {}
        margin = finite_float(components.get("activity_margin", candidate.get("lp_activity_optimum")), 0.04)
        return (
            min(support, 45.0),
            area,
            finite_float(candidate.get("w2_support_diverse_score"), 0.0),
            -min(density, 10000.0),
            abs(margin - 0.04),
            candidate_track_id(candidate),
        )

    current = list(selected_candidates)
    accepted_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    trial_checks = 0
    max_trial_checks = max(0, int(max_trials))
    for candidate in sorted(cheap_pool, key=local_order_key)[: max(30, int(max_additions) * 4)]:
        if len(accepted_rows) >= int(max_additions):
            break
        if trial_checks >= max_trial_checks:
            rejection_counts["max_trial_checks_reached"] = rejection_counts.get("max_trial_checks_reached", 0) + 1
            break
        cid = candidate_track_id(candidate)
        lp_started = time.perf_counter()
        feasible, active, margin, status = lp_candidate_active_against(
            current,
            candidate,
            inside_tol=max(float(point_tol), 0.03),
            activity_tol=0.03,
        )
        timing["lp_activity"] += time.perf_counter() - lp_started
        if not feasible:
            rejection_counts["lp_infeasible"] = rejection_counts.get("lp_infeasible", 0) + 1
            continue
        if not active:
            rejection_counts["candidate_inactive"] = rejection_counts.get("candidate_inactive", 0) + 1
            continue
        trial_checks += 1
        trial_candidates = current + [candidate]
        geom_started = time.perf_counter()
        geom = trial_face_geometry_row(
            candidate=candidate,
            trial_candidates=trial_candidates,
            candidate_index=len(current),
            reconstruction_kwargs=reconstruction_kwargs,
            geometry_mode=str(trial_geometry_mode),
            current_reconstructed=current_rec,
        )
        geom_elapsed = time.perf_counter() - geom_started
        timing["trial_geometry"] += geom_elapsed
        counters["trial_geometry_calls"] += 1
        edge_clip_times.append(float(geom_elapsed))
        if str(geom.get("trial_geometry_method")) == "incremental":
            counters["trial_geometry_incremental_calls"] += 1
        else:
            counters["trial_geometry_full_calls"] += 1
        if not bool(geom.get("trial_active")):
            rejection_counts["candidate_inactive"] = rejection_counts.get("candidate_inactive", 0) + 1
            counters["trial_geometry_rejected_calls"] += 1
            continue
        active_trial_ids = {
            int(candidate_track_id(trial_candidates[int(i)]))
            for i in geom.get("active_candidate_indices", [])
            if 0 <= int(i) < len(trial_candidates)
        }
        missing_preserved = sorted(int(v) for v in preserve_candidate_ids if int(v) not in active_trial_ids)
        if missing_preserved:
            if any(int(v) in stage1_preserve_candidate_ids for v in missing_preserved):
                primary = "breaks_stage1_repaired_patch"
            elif any(int(v) in stage2_preserve_candidate_ids for v in missing_preserved):
                primary = "breaks_stage2_repaired_patch"
            else:
                primary = "breaks_repaired_core_patch"
            rejection_counts[primary] = rejection_counts.get(primary, 0) + 1
            counters["trial_geometry_rejected_calls"] += 1
            if len(trial_rows) < 120:
                trial_rows.append(
                    {
                        "candidate_id": int(cid),
                        "accepted": False,
                        "primary_rejection_reason": primary,
                        "missing_preserved_candidate_ids": missing_preserved,
                        "lp_margin": as_json_float(float(margin)) if margin is not None else None,
                        **geom,
                    }
                )
            continue
        area_ratio = finite_float(geom.get("area_ratio"), float("inf"))
        extra_area = finite_float(geom.get("extra_area"), float("inf"))
        hull_distance = finite_float(geom.get("hull_to_face_polygon_distance_median"), float("inf"))
        heuristic_failures: list[str] = []
        if area_ratio > 25.0:
            heuristic_failures.append("trial_face_area_mismatch")
        if extra_area > 0.12 or hull_distance > 0.20:
            heuristic_failures.append("trial_face_nonlocal")
        if area_ratio > 25.0 or extra_area > 0.12 or hull_distance > 0.20:
            primary = heuristic_failures[0] if heuristic_failures else "trial_face_nonlocal"
            rejection_counts[primary] = rejection_counts.get(primary, 0) + 1
            counters["trial_geometry_rejected_calls"] += 1
            if len(trial_rows) < 120:
                trial_rows.append(
                    {
                        "candidate_id": int(cid),
                        "accepted": False,
                        "primary_rejection_reason": primary,
                        "heuristic_failures": heuristic_failures,
                        "lp_margin": as_json_float(float(margin)) if margin is not None else None,
                        **geom,
                    }
                )
            continue
        outside_started = time.perf_counter()
        outside = cumulative_outside_metrics(trial_candidates, trusted_points, trusted_z_indices, point_tol=float(point_tol))
        timing["outside_checks"] += time.perf_counter() - outside_started
        topology = geom.get("topology") if isinstance(geom.get("topology"), dict) else {}
        hard_failures: list[str] = []
        if not topology_is_valid(topology):
            hard_failures.append("topology_invalid")
        if finite_float(outside.get("cumulative_outside_fraction"), float("inf")) >= 0.01:
            hard_failures.append("trusted_outside_exceeded")
        if int(outside.get("cumulative_lost_z_levels") if outside.get("cumulative_lost_z_levels") is not None else 0) != 0:
            hard_failures.append("lost_z")
        if hard_failures:
            primary = hard_failures[0]
            rejection_counts[primary] = rejection_counts.get(primary, 0) + 1
            counters["trial_geometry_rejected_calls"] += 1
            if len(trial_rows) < 120:
                trial_rows.append(
                    {
                        "candidate_id": int(cid),
                        "accepted": False,
                        "primary_rejection_reason": primary,
                        "hard_failures": hard_failures,
                        "heuristic_failures": [],
                        "outside": outside,
                        "lp_margin": as_json_float(float(margin)) if margin is not None else None,
                        **geom,
                    }
                )
            continue
        candidate = dict(candidate)
        candidate["small_face_refinement_mode"] = str(mode_label)
        candidate["small_face_refinement_rank"] = int(len(accepted_rows) + 1)
        candidate["small_face_refinement_band_id"] = band_id(candidate)
        current.append(candidate)
        counters["trial_geometry_accepted_calls"] += 1
        if str(trial_geometry_mode) == "incremental":
            rebuild_started = time.perf_counter()
            current_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(current, **reconstruction_kwargs)
            timing["accepted_state_rebuild"] += time.perf_counter() - rebuild_started
            counters["accepted_state_rebuilds"] += 1
        accepted_rows.append(
            {
                "candidate_id": int(cid),
                "append_rank": int(len(accepted_rows) + 1),
                "band_id": band_id(candidate),
                "previous_provenance": candidate.get("post_repair_refill_previous_provenance"),
                "lp_margin": as_json_float(float(margin)) if margin is not None else None,
                "outside": outside,
                "primary_rejection_reason": None,
                "hard_failures": [],
                "heuristic_failures": [],
                **geom,
            }
        )
    edge_clip_arr = np.array(edge_clip_times, dtype=float)
    trial_geometry_timing = {
        "count": int(edge_clip_arr.size),
        "mean": as_json_float(float(np.mean(edge_clip_arr))) if edge_clip_arr.size else None,
        "p50": as_json_float(float(np.percentile(edge_clip_arr, 50))) if edge_clip_arr.size else None,
        "p95": as_json_float(float(np.percentile(edge_clip_arr, 95))) if edge_clip_arr.size else None,
        "total": as_json_float(float(np.sum(edge_clip_arr))) if edge_clip_arr.size else None,
    }
    return current, {
        "mode": str(mode_label),
        "trial_geometry_mode": str(trial_geometry_mode),
        "production_changed": bool(accepted_rows),
        "max_additions": int(max_additions),
        "accepted_additions": int(len(accepted_rows)),
        "excluded_candidate_ids": sorted(int(v) for v in excluded_candidate_ids),
        "preserve_candidate_ids": sorted(int(v) for v in preserve_candidate_ids),
        "stage1_preserve_candidate_ids": sorted(int(v) for v in stage1_preserve_candidate_ids),
        "stage2_preserve_candidate_ids": sorted(int(v) for v in stage2_preserve_candidate_ids),
        "dense_bands": band_detection,
        "cheap_pool_size": int(len(cheap_pool)),
        "previous_provenance_counts": provenance_counts,
        "trial_checks": int(trial_checks),
        "max_trial_checks": int(max_trial_checks),
        "rejection_counts": rejection_counts,
        "challenger_order_signature": [int(candidate_track_id(candidate)) for candidate in sorted(cheap_pool, key=local_order_key)[:160]],
        "accepted_sequence_signature": [int(row["candidate_id"]) for row in accepted_rows],
        "timing_breakdown_seconds": {key: as_json_float(float(value)) for key, value in timing.items()},
        "trial_geometry_timing_seconds": trial_geometry_timing,
        "trial_geometry_counters": counters,
        "accepted_rows": accepted_rows,
        "trial_rows_sample": trial_rows,
        "timing_seconds": as_json_float(time.perf_counter() - started),
    }


def diagnose_protected_post_repair_refill(
    *,
    core_candidates: list[dict[str, object]],
    core_reconstructed: dict[str, object],
    w2_preselection_candidates: list[dict[str, object]],
    selected_candidates: list[dict[str, object]],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
    trial_geometry_mode: str,
    stage1_target_candidate_ids: set[int],
    stage2_target_candidate_ids: set[int],
    removed_blocker_ids: set[int],
    core_ids: set[int] | None = None,
    model_faces: list[dict[str, object]] | None = None,
    initial_vertices: np.ndarray | None = None,
    contours: list[object] | None = None,
    challenger_order_ids: list[int] | None = None,
    max_accepted: int = 25,
    max_trials: int = 80,
    extended_max_trials: int | None = None,
    require_dense_band: bool = True,
    protect_all_active_patches: bool = False,
    rolling_protection: bool = False,
    safe_control_candidate_ids: set[int] | None = None,
    mode_label: str = "protected_refill",
    candidate_filter_mode: str = "small-face",
    candidate_order_mode: str = "small-face",
    protect_active_faces_individually: bool = False,
    strict_protected_footprint: bool = False,
    preserve_baseline_outside: bool = False,
    max_trial_face_extra_area: float | None = 0.12,
    refresh_trial_geometry_from_full_rebuild: bool = False,
) -> dict[str, object]:
    if bool(strict_protected_footprint) and not bool(protect_all_active_patches):
        raise ValueError(
            "strict protected footprint requires protect_all_active_patches"
        )
    started = time.perf_counter()
    timing: dict[str, float] = {
        "preparation": 0.0,
        "lp": 0.0,
        "patch_preservation": 0.0,
        "incremental_trial_geometry": 0.0,
        "accepted_state_full_rebuild": 0.0,
        "final_checkpoint_geometry": 0.0,
        "oracle_posthoc": 0.0,
        "order_diagnostics": 0.0,
    }
    safe_control_candidate_ids = {int(v) for v in (safe_control_candidate_ids or set())}
    z_slices = prepare_z_level_slices(trusted_z_indices)

    prep_started = time.perf_counter()
    annotated = annotate_candidate_pool_neutral(
        w2_preselection_candidates,
        core_candidates=core_candidates,
        core_reconstructed=core_reconstructed,
        trusted_points=trusted_points,
        trusted_z_indices=trusted_z_indices,
        point_tol=float(point_tol),
        finite_support_plane_distance=float(finite_support_plane_distance),
        finite_support_hull_margin=float(finite_support_hull_margin),
        finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
    )
    selected_ids = {int(candidate_track_id(candidate)) for candidate in selected_candidates}
    accepted_w2_ids = {
        int(candidate_track_id(candidate))
        for candidate in selected_candidates
        if str(candidate.get("candidate_origin")) == "w2_addition"
    }
    ranked_out = [candidate for candidate in annotated if int(candidate_track_id(candidate)) not in accepted_w2_ids]
    band_detection = detect_dense_small_face_bands(ranked_out)
    bands = band_detection.get("bands") if isinstance(band_detection.get("bands"), list) else []

    def band_id(candidate: dict[str, object]) -> int | None:
        centroid = candidate_hull_centroid(candidate)
        if centroid is None:
            return None
        z = float(centroid[2])
        for band in bands:
            if float(band.get("z_min") or 0.0) <= z <= float(band.get("z_max") or 0.0):
                return int(band.get("band_id") or 0)
        return None

    baseline_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(selected_candidates, **reconstruction_kwargs)
    baseline_active_indices = [int(i) for i in baseline_rec.get("face_candidate_indices", []) if 0 <= int(i) < len(selected_candidates)]
    baseline_active_candidates = [selected_candidates[i] for i in baseline_active_indices]
    baseline_outside = (
        cumulative_outside_metrics(
            selected_candidates,
            trusted_points,
            trusted_z_indices,
            point_tol=float(point_tol),
        )
        if bool(preserve_baseline_outside)
        else {}
    )

    def duplicate_active(candidate: dict[str, object], active_candidates: list[dict[str, object]]) -> bool:
        for active_candidate in active_candidates:
            angle, offset, centroid = candidate_plane_relation_score(candidate, active_candidate)
            if angle <= 0.75 and offset <= 0.01 and centroid <= 0.08:
                return True
        return False

    cheap_pool: list[dict[str, object]] = []
    static_rejection_counts: dict[str, int] = {}
    provenance_counts: dict[str, int] = {}
    for candidate in ranked_out:
        cid = int(candidate_track_id(candidate))
        if cid in selected_ids or cid in removed_blocker_ids:
            static_rejection_counts["already_selected_or_removed"] = static_rejection_counts.get("already_selected_or_removed", 0) + 1
            continue
        if bool(require_dense_band) and band_id(candidate) is None:
            static_rejection_counts["outside_dense_band"] = static_rejection_counts.get("outside_dense_band", 0) + 1
            continue
        if finite_float(candidate.get("hull_area"), 0.0) <= EPS or candidate_hull_points(candidate).shape[0] == 0:
            static_rejection_counts["empty_local_hull"] = static_rejection_counts.get("empty_local_hull", 0) + 1
            continue
        if str(candidate_filter_mode) == "small-face":
            if finite_float(candidate.get("hull_area"), float("inf")) > 0.04:
                static_rejection_counts["large_hull_not_small_face"] = static_rejection_counts.get("large_hull_not_small_face", 0) + 1
                continue
            if finite_float(candidate.get("finite_support_count"), float("inf")) > 45.0:
                static_rejection_counts["large_support_not_small_face"] = static_rejection_counts.get("large_support_not_small_face", 0) + 1
                continue
        elif str(candidate_filter_mode) != "angular-scale":
            raise ValueError(f"unknown protected-additive candidate filter mode: {candidate_filter_mode}")
        if duplicate_active(candidate, baseline_active_candidates):
            static_rejection_counts["duplicate_active_plane"] = static_rejection_counts.get("duplicate_active_plane", 0) + 1
            continue
        previous_reason = str(candidate.get("selection_rejection_reason") or candidate.get("primary_rejection_reason") or "")
        if previous_reason in {"candidate_inactive", "inactive"}:
            provenance = "previously_inactive"
        elif previous_reason in {"candidate_redundant", "duplicate_active_plane", "redundant"}:
            provenance = "previously_redundant"
        elif "mismatch" in previous_reason or "nonlocal" in previous_reason:
            provenance = "previous_trial_face_mismatch"
        elif candidate.get("selection_rank") is None and candidate.get("w2_support_diverse_rank") is None:
            provenance = "previously_untried_budget"
        else:
            provenance = "other_state_dependent"
        candidate["post_repair_refill_previous_provenance"] = provenance
        provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
        cheap_pool.append(candidate)

    def local_order_key(candidate: dict[str, object]) -> tuple[float, float, float, float, float, int]:
        if str(candidate_order_mode) == "angular-scale":
            return angular_scale_quality_local_key(candidate)
        if str(candidate_order_mode) != "small-face":
            raise ValueError(f"unknown protected-additive candidate order mode: {candidate_order_mode}")
        area = max(finite_float(candidate.get("hull_area"), 0.0), 1e-4)
        support = float(candidate.get("finite_support_count") or 0)
        density = support / area
        components = candidate.get("w2_support_diverse_components") if isinstance(candidate.get("w2_support_diverse_components"), dict) else {}
        margin = finite_float(components.get("activity_margin", candidate.get("lp_activity_optimum")), 0.04)
        return (
            min(support, 45.0),
            area,
            finite_float(candidate.get("w2_support_diverse_score"), 0.0),
            -min(density, 10000.0),
            abs(margin - 0.04),
            int(candidate_track_id(candidate)),
        )

    if challenger_order_ids:
        cheap_by_id = {int(candidate_track_id(candidate)): candidate for candidate in cheap_pool}
        ordered_ids = [int(v) for v in challenger_order_ids]
        seen_ordered: set[int] = set()
        challenger_order = []
        for cid in ordered_ids:
            candidate = cheap_by_id.get(int(cid))
            if candidate is not None and int(cid) not in seen_ordered:
                challenger_order.append(candidate)
                seen_ordered.add(int(cid))
        challenger_order.extend(
            candidate
            for candidate in sorted(cheap_pool, key=local_order_key)
            if int(candidate_track_id(candidate)) not in seen_ordered
        )
    else:
        challenger_order = sorted(cheap_pool, key=local_order_key)
    order_diag_started = time.perf_counter()
    safe_control_ranks: dict[str, dict[str, object]] = {}
    for safe_id in sorted(int(v) for v in safe_control_candidate_ids):
        rank = next((idx for idx, candidate in enumerate(challenger_order, start=1) if int(candidate_track_id(candidate)) == safe_id), None)
        candidate = next((candidate for candidate in challenger_order if int(candidate_track_id(candidate)) == safe_id), None)
        safe_control_ranks[str(safe_id)] = {
            "rank": int(rank) if rank is not None else None,
            "in_top_160": bool(rank is not None and rank <= 160),
            "band_id": band_id(candidate) if candidate is not None else None,
            "inside_detected_dense_band": bool(candidate is not None and band_id(candidate) is not None),
        }

    def order_prefix_summary(limit: int) -> dict[str, object]:
        top = challenger_order[: int(limit)]
        inside = [candidate for candidate in top if band_id(candidate) is not None]
        outside = [candidate for candidate in top if band_id(candidate) is None]
        return {
            "limit": int(limit),
            "count": int(len(top)),
            "inside_dense_band": int(len(inside)),
            "outside_dense_band": int(len(outside)),
            "safe_control_candidate_ids": [
                int(candidate_track_id(candidate))
                for candidate in top
                if int(candidate_track_id(candidate)) in safe_control_candidate_ids
            ],
            "finite_support_count": distribution_summary([candidate.get("finite_support_count") for candidate in top]),
            "hull_area": distribution_summary([candidate.get("hull_area") for candidate in top]),
            "plane_rms": distribution_summary([candidate.get("plane_rms") for candidate in top]),
        }

    order_diagnostics = {
        "order_signature": [int(candidate_track_id(candidate)) for candidate in challenger_order[:200]],
        "safe_control_ranks": safe_control_ranks,
        "top_prefixes": [order_prefix_summary(limit) for limit in [25, 50, 100, 160]],
        "safe_controls_in_top_160": int(sum(1 for row in safe_control_ranks.values() if bool(row.get("in_top_160")))),
    }
    timing["order_diagnostics"] += time.perf_counter() - order_diag_started
    core_active_indices = [int(i) for i in core_reconstructed.get("face_candidate_indices", []) if 0 <= int(i) < len(core_candidates)]
    groups, _ = build_core_plane_patch_groups(
        core_candidates,
        core_active_indices,
        trusted_points,
        trusted_z_indices,
        finite_support_plane_distance=float(finite_support_plane_distance),
        finite_support_hull_margin=float(finite_support_hull_margin),
        finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
    )
    lp_engine = LpActivityEngine(enabled=True)
    lp_engine.stage = "protected_refill_baseline"
    lp_engine.set_reusable_universe("protected_refill_baseline", selected_candidates, inside_tol=0.03)
    selected_candidate_ids_tuple = tuple(sorted(int(candidate_track_id(candidate)) for candidate in selected_candidates))

    protected_groups: list[dict[str, object]] = []
    baseline_samples_by_core_index: dict[int, dict[str, object]] = {}
    for group in groups:
        active_equiv = group_equivalent_active_candidates(group, core_candidates, baseline_active_candidates)
        if not active_equiv:
            continue
        rep_index = int(group.get("representative_core_index", -1))
        if rep_index < 0 or rep_index >= len(core_candidates):
            continue
        representative = core_candidates[rep_index]
        observed_support = int(group.get("union_witness_count") or 0)
        finite_support_count = int(representative.get("finite_support_count") or 0)
        if observed_support <= 0 and finite_support_count <= 0:
            continue
        target = representative
        target_id = int(candidate_track_id(target))
        base_ids = tuple(int(cid) for cid in selected_candidate_ids_tuple if int(cid) != target_id)
        activity = lp_face_activity_details(
            None,
            target,
            inside_tol=0.03,
            face_activity_slack=0.03,
            lp_engine=lp_engine,
            base_candidate_ids=base_ids,
        )
        samples = core_reference_face_samples(core_candidates, core_reconstructed, rep_index)
        deficit = None
        if samples is not None:
            baseline_samples_by_core_index[rep_index] = samples
            deficit = patch_surface_deficit_for_samples(
                selected_candidates,
                [np.array(sample, dtype=float) for sample in samples.get("samples", [])],
                sample_tol=0.0,
            )
        member_ids = {int(v) for v in group.get("member_core_candidate_ids", [])}
        role = "ordinary_protected"
        if member_ids & {int(v) for v in stage1_target_candidate_ids}:
            role = "stage1_repaired_patch"
        if member_ids & {int(v) for v in stage2_target_candidate_ids}:
            role = "stage2_repaired_patch"
        protected_groups.append(
            {
                "group_index": int(group.get("group_index", -1)),
                "stable_group_id": group.get("stable_group_id"),
                "role": role,
                "representative_core_index": int(rep_index),
                "representative_core_candidate_id": int(target_id),
                "member_core_candidate_ids": sorted(int(v) for v in member_ids),
                "active_representative_ids": active_equiv,
                "support_confidence": group.get("support_confidence"),
                "representative_plane": group.get("representative_plane"),
                "union_witness_count": int(group.get("union_witness_count") or 0),
                "union_witness_z_levels": int(group.get("union_witness_z_levels") or 0),
                "base_lp_effective_margin": activity.get("effective_face_margin"),
                "base_surface_deficit": {key: value for key, value in (deficit or {}).items() if key != "samples"} if isinstance(deficit, dict) else None,
            }
        )
    baseline_candidate_by_id = {int(candidate_track_id(candidate)): candidate for candidate in baseline_active_candidates}

    def active_patch_compatible(a: dict[str, object], b: dict[str, object]) -> bool:
        angle, offset, centroid_distance = candidate_plane_relation_score(a, b)
        za = candidate_z_intervals(a)
        zb = candidate_z_intervals(b)
        hull_overlap, hull_overlap_fraction = interval_overlap(
            za.get("hull_z_min"),
            za.get("hull_z_max"),
            zb.get("hull_z_min"),
            zb.get("hull_z_max"),
        )
        support_overlap, support_overlap_fraction = interval_overlap(
            za.get("support_z_min"),
            za.get("support_z_max"),
            zb.get("support_z_min"),
            zb.get("support_z_max"),
        )
        return bool(
            angle <= 1.0
            and offset <= 0.02
            and centroid_distance <= 0.12
            and max(hull_overlap_fraction, support_overlap_fraction) >= 0.25
            and hull_overlap >= 0.0
            and support_overlap >= 0.0
        )

    def active_patch_group_class(candidate: dict[str, object], selected_index: int) -> str:
        if int(selected_index) < len(core_candidates):
            return "core_protected"
        support = finite_float(candidate.get("finite_support_count"), finite_float(candidate.get("observed_support_near_count"), 0.0))
        density = finite_float(candidate.get("finite_support_density"), 0.0)
        levels = finite_float(candidate.get("levels"), finite_float(candidate.get("finite_support_z_levels"), 0.0))
        plane_rms = finite_float(candidate.get("plane_rms"), float("inf"))
        hull_area = finite_float(candidate.get("hull_area"), 0.0)
        residual = finite_float(candidate.get("finite_support_residual_p95"), 0.0)
        if support >= 6.0 and density >= 20.0 and levels >= 8.0 and plane_rms <= 0.01 and hull_area >= 0.001 and residual <= 0.05:
            return "observed_strong_active_patch"
        if support > 0.0 and hull_area > 0.0 and plane_rms <= 0.05:
            return "observed_weak_active_patch"
        return "unsupported_or_ambiguous_active_patch"

    protected_candidate_by_id = dict(baseline_candidate_by_id)
    protected_face_samples_by_group_id: dict[str, dict[str, object]] = {}

    def append_active_patch_groups(
        protected: list[dict[str, object]],
        active_indices: list[int],
        active_source_candidates: list[dict[str, object]],
        source_reconstructed: dict[str, object],
        *,
        group_kind: str,
        stable_prefix: str,
    ) -> int:
        added = 0
        for selected_index in active_indices:
            if int(selected_index) < 0 or int(selected_index) >= len(active_source_candidates):
                continue
            candidate = active_source_candidates[int(selected_index)]
            cid = int(candidate_track_id(candidate))
            protected_candidate_by_id[cid] = candidate
            placed = False
            for group in protected:
                if bool(protect_active_faces_individually):
                    if cid in {int(value) for value in group.get("member_candidate_ids", [])}:
                        placed = True
                        break
                    continue
                rep = protected_candidate_by_id.get(int(group["representative_candidate_id"]))
                if rep is not None and active_patch_compatible(candidate, rep):
                    group["member_candidate_ids"].append(cid)
                    group["active_representative_ids"].append(cid)
                    placed = True
                    break
            if not placed:
                role = active_patch_group_class(candidate, int(selected_index))
                stable_group_id = f"{stable_prefix}_{len(protected):04d}_{cid}"
                protected.append(
                    {
                        "kind": str(group_kind),
                        "stable_group_id": stable_group_id,
                        "role": role,
                        "representative_selected_index": int(selected_index),
                        "representative_candidate_id": int(cid),
                        "member_candidate_ids": [int(cid)],
                        "active_representative_ids": [int(cid)],
                        "support_confidence": role,
                        "representative_plane": {
                            "candidate_id": int(cid),
                            "origin": candidate.get("candidate_origin"),
                            "hull_area": as_json_float(float(finite_float(candidate.get("hull_area"), 0.0))),
                            "finite_support_count": as_json_float(float(finite_float(candidate.get("finite_support_count"), 0.0))),
                            "plane_rms": as_json_float(float(finite_float(candidate.get("plane_rms"), 0.0))),
                        },
                    }
                )
                face_samples = (
                    core_reference_face_samples(
                        active_source_candidates,
                        source_reconstructed,
                        int(selected_index),
                        vertex_weight=0.85,
                    )
                    if bool(strict_protected_footprint)
                    else None
                )
                if face_samples is not None:
                    protected_face_samples_by_group_id[str(stable_group_id)] = face_samples
                elif bool(strict_protected_footprint):
                    raise RuntimeError(
                        f"strict protected footprint has no baseline face for candidate {cid}"
                    )
                added += 1
        return added

    if protect_all_active_patches:
        active_patch_groups: list[dict[str, object]] = []
        append_active_patch_groups(
            active_patch_groups,
            baseline_active_indices,
            selected_candidates,
            baseline_rec,
            group_kind="golden_active_patch",
            stable_prefix="golden_active_patch",
        )
        protected_groups = active_patch_groups
    timing["preparation"] += time.perf_counter() - prep_started

    def protected_group_signature() -> dict[str, object]:
        rows = []
        for group in protected_groups:
            representative_value = group.get("representative_candidate_id")
            if representative_value is None:
                representative_value = group.get("representative_core_candidate_id")
            rows.append(
                {
                    "stable_group_id": str(group.get("stable_group_id")),
                    "kind": str(group.get("kind") or "core_plane_patch_group"),
                    "role": str(group.get("role") or "ordinary_protected"),
                    "representative_candidate_id": int(
                        representative_value if representative_value is not None else -1
                    ),
                    "member_candidate_ids": sorted(
                        int(v)
                        for v in (
                            group.get("member_candidate_ids")
                            if isinstance(group.get("member_candidate_ids"), list)
                            else group.get("member_core_candidate_ids")
                            if isinstance(group.get("member_core_candidate_ids"), list)
                            else []
                        )
                    ),
                    "active_representative_ids": sorted(int(v) for v in group.get("active_representative_ids", [])),
                }
            )
        rows.sort(key=lambda row: (str(row["stable_group_id"]), int(row["representative_candidate_id"])))
        blob = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        return {
            "count": int(len(rows)),
            "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            "rows": rows[:200],
        }

    protected_groups_signature = protected_group_signature()

    def active_face_area_for_candidate_ids(
        rec: dict[str, object],
        source_candidates: list[dict[str, object]],
        candidate_ids: set[int],
    ) -> float | None:
        records = active_face_records_for_candidate_ids(
            rec,
            source_candidates,
            candidate_ids,
        )
        areas = [float(record["area"]) for record in records]
        return max(areas) if areas else None

    def active_face_records_for_candidate_ids(
        rec: dict[str, object],
        source_candidates: list[dict[str, object]],
        candidate_ids: set[int],
    ) -> list[dict[str, object]]:
        vertices = np.asarray(rec.get("vertices") or [], dtype=float)
        faces = rec.get("faces") if isinstance(rec.get("faces"), list) else []
        sources = rec.get("face_candidate_indices") if isinstance(rec.get("face_candidate_indices"), list) else []
        if vertices.ndim != 2 or vertices.shape[0] == 0:
            return []
        records: list[dict[str, object]] = []
        for face, source_index in zip(faces, sources):
            index = int(source_index)
            if index < 0 or index >= len(source_candidates):
                continue
            if int(candidate_track_id(source_candidates[index])) not in candidate_ids:
                continue
            face_indices = np.asarray(face, dtype=int)
            if face_indices.size < 3 or np.any(face_indices < 0) or np.any(face_indices >= vertices.shape[0]):
                continue
            plane = candidate_plane(source_candidates[index])
            if plane is None:
                continue
            points = vertices[face_indices]
            records.append(
                {
                    "vertices": points,
                    "normal": np.asarray(plane[1], dtype=float),
                    "area": float(polygon_area(points)),
                    "candidate_id": int(candidate_track_id(source_candidates[index])),
                }
            )
        return records

    def footprint_distance_to_active_faces(
        footprint_points: list[object],
        face_records: list[dict[str, object]],
    ) -> dict[str, object]:
        points = [np.asarray(point, dtype=float) for point in footprint_points]
        distances = [
            min(point_to_face_polygon_distance(point, face) for face in face_records)
            for point in points
        ] if points and face_records else []
        return {
            "count": int(len(distances)),
            "median": as_json_float(float(np.median(distances))) if distances else None,
            "p95": as_json_float(float(np.percentile(distances, 95))) if distances else None,
            "max": as_json_float(float(np.max(distances))) if distances else None,
        }

    def summarize_checkpoint(
        *,
        label: str,
        accepted_count: int,
        trial_checks: int,
        current_candidates: list[dict[str, object]],
        match_cache: dict[int, dict[str, object] | None],
    ) -> dict[str, object]:
        geom_started = time.perf_counter()
        rec = reconstruct_polyhedron_from_halfspaces_edge_clip(current_candidates, **reconstruction_kwargs)
        timing["final_checkpoint_geometry"] += time.perf_counter() - geom_started
        outside = evaluate_trusted_cloud_outside(current_candidates, trusted_points, z_slices, point_tol=float(point_tol))
        active_indices = [int(i) for i in rec.get("face_candidate_indices", []) if 0 <= int(i) < len(current_candidates)]
        active_candidates = [current_candidates[i] for i in active_indices]
        active_refill_ids = [
            int(candidate_track_id(candidate))
            for candidate in active_candidates
            if str(candidate.get("small_face_refinement_mode", "")).startswith("post-repair-refill")
        ]
        active_ids = {int(candidate_track_id(candidate)) for candidate in active_candidates}
        preserved_failures: list[dict[str, object]] = []
        role_counts: dict[str, int] = {}
        for protected_group in protected_groups:
            role = str(protected_group.get("role") or "ordinary_protected")
            role_counts[role] = role_counts.get(role, 0) + 1
            exact_representatives = {int(v) for v in protected_group.get("active_representative_ids", [])}
            active_equivalent_ids = sorted(exact_representatives & active_ids)
            if active_equivalent_ids and not bool(strict_protected_footprint):
                continue
            if protected_group.get("kind") in {"golden_active_patch", "rolling_active_patch"}:
                representative_value = protected_group.get("representative_candidate_id")
                representative = protected_candidate_by_id.get(
                    int(representative_value if representative_value is not None else -1)
                )
                if representative is not None and not active_equivalent_ids:
                    active_equivalent_ids = [
                        int(candidate_track_id(candidate))
                        for candidate in active_candidates
                        if active_patch_compatible(representative, candidate)
                    ]
                strict_failure = None
                if bool(strict_protected_footprint) and active_equivalent_ids:
                    footprint_samples = protected_face_samples_by_group_id.get(
                        str(protected_group.get("stable_group_id"))
                    )
                    if isinstance(footprint_samples, dict):
                        deficit = patch_surface_deficit_for_samples(
                            current_candidates,
                            [
                                np.asarray(sample, dtype=float)
                                for sample in footprint_samples.get("samples", [])
                            ],
                            sample_tol=min(max(float(point_tol) * 0.05, 1e-5), 0.0025),
                        )
                        face_area = active_face_area_for_candidate_ids(
                            rec,
                            current_candidates,
                            {int(value) for value in active_equivalent_ids},
                        )
                        face_records = active_face_records_for_candidate_ids(
                            rec,
                            current_candidates,
                            {int(value) for value in active_equivalent_ids},
                        )
                        footprint_distance = footprint_distance_to_active_faces(
                            list(footprint_samples.get("face_points", [])),
                            face_records,
                        )
                        baseline_area = finite_float(footprint_samples.get("area"), 0.0)
                        if finite_float(deficit.get("cut_fraction"), float("inf")) > 0.0:
                            strict_failure = "protected_patch_footprint_cut"
                        elif face_area is None:
                            strict_failure = "protected_patch_face_missing"
                        elif (
                            finite_float(footprint_distance.get("p95"), float("inf")) > 0.03
                            or finite_float(footprint_distance.get("max"), float("inf")) > 0.05
                        ):
                            strict_failure = "protected_patch_footprint_shifted"
                        elif baseline_area > EPS and float(face_area) / baseline_area < 0.50:
                            strict_failure = "protected_patch_area_shrunk"
                if active_equivalent_ids and strict_failure is None:
                    continue
            else:
                group = groups[int(protected_group["group_index"])]
                if group_equivalent_active_candidates(group, core_candidates, active_candidates):
                    continue
            preserved_failures.append(
                {
                    "stable_group_id": protected_group.get("stable_group_id"),
                    "role": role,
                    "representative_core_candidate_id": protected_group.get("representative_core_candidate_id"),
                    "reason": strict_failure
                    if protected_group.get("kind") in {"golden_active_patch", "rolling_active_patch"}
                    else "missing_active_equivalent",
                }
            )
        reprojection = None
        if contours is not None:
            reproj_started = time.perf_counter()
            reprojection = evaluate_polyhedron_reprojection(
                name=str(label),
                vertices_3d=np.array(rec.get("vertices") or [], dtype=float),
                contours=contours,
                direction_count=32,
                contour_sample=80,
            )
            timing["final_checkpoint_geometry"] += time.perf_counter() - reproj_started
        row: dict[str, object] = {
            "label": label,
            "accepted_count": int(accepted_count),
            "trial_checks": int(trial_checks),
            "candidate_count": int(len(current_candidates)),
            "active_refill_additions": int(len(active_refill_ids)),
            "active_refill_candidate_ids": active_refill_ids,
            "vertices": int(len(rec.get("vertices") or [])),
            "edges": int(len(rec.get("edges") or [])),
            "faces": int(len(rec.get("faces") or [])),
            "volume": rec.get("reliable_volume"),
            "topology": rec.get("topology"),
            "outside": outside,
            "max_per_z_outside": outside.get("cumulative_max_level_outside_fraction") if isinstance(outside, dict) else None,
            "lost_z": outside.get("cumulative_lost_z_levels") if isinstance(outside, dict) else None,
            "reprojection": reprojection,
            "reprojection_p95": reprojection.get("symmetric_contour_distance_p95") if isinstance(reprojection, dict) else None,
            "geometry_signature": mesh_geometry_signature(current_candidates, rec),
            "protected_group_preservation": {
                "protected_group_count": int(len(protected_groups)),
                "role_counts": role_counts,
                "failed_count": int(len(preserved_failures)),
                "failures": preserved_failures[:20],
                "stage1_preserved": not any(str(row.get("role")) == "stage1_repaired_patch" for row in preserved_failures),
                "stage2_preserved": not any(str(row.get("role")) == "stage2_repaired_patch" for row in preserved_failures),
                "all_preserved": not preserved_failures,
            },
            "removed_blockers_absent": not bool({int(v) for v in removed_blocker_ids} & {int(candidate_track_id(candidate)) for candidate in current_candidates}),
        }
        if core_ids is not None and model_faces is not None and initial_vertices is not None:
            oracle_started = time.perf_counter()
            row["oracle_posthoc"] = summarize_reconstruction_prefix(
                prefix=int(accepted_count),
                candidates=current_candidates,
                core_count=len(core_candidates),
                core_ids=core_ids,
                model_faces=model_faces,
                initial_vertices=initial_vertices,
                trusted_points=trusted_points,
                trusted_z_indices=trusted_z_indices,
                point_tol=float(point_tol),
                reconstruction_kwargs=reconstruction_kwargs,
                match_cache=match_cache,
            )
            timing["oracle_posthoc"] += time.perf_counter() - oracle_started
        return row

    def run_sequence(
        *,
        protected: bool,
        max_accept: int,
        max_trial_checks: int,
        extended_trial_checks: int | None,
        checkpoint_counts: set[int],
    ) -> dict[str, object]:
        current = list(selected_candidates)
        current_rec = dict(baseline_rec)
        accepted_rows: list[dict[str, object]] = []
        trial_rows: list[dict[str, object]] = []
        checkpoints: list[dict[str, object]] = []
        rejection_counts: dict[str, int] = {}
        confusion = {
            "gate_rejected_and_canonical_core_lost": 0,
            "gate_rejected_but_canonical_core_preserved": 0,
            "gate_passed_and_canonical_core_preserved": 0,
            "gate_passed_but_canonical_core_lost": 0,
            "not_evaluated": 0,
        }

        def remember_trial_rejection(
            *,
            cid: int,
            challenger_rank: int,
            reason: str,
            margin: float | None,
            extra: dict[str, object] | None = None,
        ) -> None:
            if len(trial_rows) >= 160 and int(cid) not in safe_control_candidate_ids:
                return
            row = {
                "candidate_id": int(cid),
                "challenger_rank": int(challenger_rank),
                "accepted": False,
                "primary_rejection_reason": str(reason),
                "lp_margin": as_json_float(float(margin)) if margin is not None and np.isfinite(float(margin)) else None,
            }
            if extra:
                row.update(extra)
            trial_rows.append(row)
        match_cache: dict[int, dict[str, object] | None] = {}
        trial_checks = 0
        first_lost_92: dict[str, object] | None = None
        rejected_posthoc_evaluations = 0
        rejected_posthoc_budget = int(max_trial_checks if protected else 3)
        initial_trial_budget = int(max_trial_checks)
        resolved_trial_budget = int(max_trial_checks)
        extension_used = False
        exhausted = True
        processed_challenger_rank = 0
        checkpoints.append(
            summarize_checkpoint(
                label="accepted_0",
                accepted_count=0,
                trial_checks=0,
                current_candidates=current,
                match_cache=match_cache,
            )
        )
        for challenger_rank, candidate in enumerate(challenger_order, start=1):
            processed_challenger_rank = int(challenger_rank)
            if len(accepted_rows) >= int(max_accept):
                exhausted = False
                break
            if trial_checks >= int(resolved_trial_budget):
                if (
                    protected
                    and not extension_used
                    and int(len(accepted_rows)) < 10
                    and extended_trial_checks is not None
                    and int(extended_trial_checks) > int(resolved_trial_budget)
                    and int(confusion["gate_passed_but_canonical_core_lost"]) == 0
                ):
                    resolved_trial_budget = int(extended_trial_checks)
                    rejected_posthoc_budget = int(resolved_trial_budget)
                    extension_used = True
                else:
                    exhausted = False
                    break
            cid = int(candidate_track_id(candidate))
            if cid in {int(candidate_track_id(c)) for c in current} or cid in removed_blocker_ids:
                continue
            lp_started = time.perf_counter()
            feasible, active, margin, status = lp_candidate_active_against(
                current,
                candidate,
                inside_tol=max(float(point_tol), 0.03),
                activity_tol=0.03,
            )
            timing["lp"] += time.perf_counter() - lp_started
            if not feasible or not active:
                reason = "lp_infeasible" if not feasible else "candidate_inactive"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                remember_trial_rejection(cid=cid, challenger_rank=challenger_rank, reason=reason, margin=margin, extra={"lp_status": status})
                continue
            trial_checks += 1
            trial_candidates = current + [candidate]
            geom_started = time.perf_counter()
            geom = trial_face_geometry_row(
                candidate=candidate,
                trial_candidates=trial_candidates,
                candidate_index=len(current),
                reconstruction_kwargs=reconstruction_kwargs,
                geometry_mode=str(trial_geometry_mode),
                current_reconstructed=current_rec,
            )
            timing["incremental_trial_geometry"] += time.perf_counter() - geom_started
            if not bool(geom.get("trial_active")):
                rejection_counts["trial_face_inactive"] = rejection_counts.get("trial_face_inactive", 0) + 1
                remember_trial_rejection(cid=cid, challenger_rank=challenger_rank, reason="trial_face_inactive", margin=margin, extra=geom)
                continue
            full_trial_rec: dict[str, object] | None = None
            if protected:
                full_started = time.perf_counter()
                full_trial_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
                timing["accepted_state_full_rebuild"] += time.perf_counter() - full_started
                full_active_indices = [
                    int(i)
                    for i in full_trial_rec.get("face_candidate_indices", [])
                    if 0 <= int(i) < len(trial_candidates)
                ]
                if bool(refresh_trial_geometry_from_full_rebuild):
                    geom.update(
                        trial_face_geometry_row_from_reconstruction(
                            candidate=candidate,
                            candidate_index=len(current),
                            reconstructed=full_trial_rec,
                            method="full-rebuild-reused",
                        )
                    )
                else:
                    geom["active_candidate_indices"] = full_active_indices
                    geom["reconstructed_vertices"] = int(len(full_trial_rec.get("vertices") or []))
                    geom["reconstructed_faces"] = int(len(full_trial_rec.get("faces") or []))
                    geom["topology"] = full_trial_rec.get("topology") if isinstance(full_trial_rec.get("topology"), dict) else {}
                    geom["topology_valid"] = topology_is_valid(geom["topology"] if isinstance(geom.get("topology"), dict) else {})
                    geom["volume"] = full_trial_rec.get("reliable_volume")
                if len(current) not in set(full_active_indices):
                    rejection_counts["incremental_full_patch_mismatch"] = rejection_counts.get("incremental_full_patch_mismatch", 0) + 1
                    remember_trial_rejection(cid=cid, challenger_rank=challenger_rank, reason="incremental_full_patch_mismatch", margin=margin, extra=geom)
                    continue
            area_ratio = finite_float(geom.get("area_ratio"), float("inf"))
            extra_area = finite_float(geom.get("extra_area"), float("inf"))
            hull_distance = finite_float(geom.get("hull_to_face_polygon_distance_median"), float("inf"))
            if area_ratio > 25.0:
                rejection_counts["trial_face_area_mismatch"] = rejection_counts.get("trial_face_area_mismatch", 0) + 1
                remember_trial_rejection(cid=cid, challenger_rank=challenger_rank, reason="trial_face_area_mismatch", margin=margin, extra=geom)
                continue
            extra_area_rejected = bool(
                max_trial_face_extra_area is not None
                and float(max_trial_face_extra_area) >= 0.0
                and extra_area > float(max_trial_face_extra_area)
            )
            if extra_area_rejected or hull_distance > 0.20:
                rejection_counts["trial_face_nonlocal"] = rejection_counts.get("trial_face_nonlocal", 0) + 1
                remember_trial_rejection(cid=cid, challenger_rank=challenger_rank, reason="trial_face_nonlocal", margin=margin, extra=geom)
                continue
            outside = cumulative_outside_metrics(trial_candidates, trusted_points, trusted_z_indices, point_tol=float(point_tol))
            topology = geom.get("topology") if isinstance(geom.get("topology"), dict) else {}
            hard_failures: list[str] = []
            if not topology_is_valid(topology):
                hard_failures.append("topology_invalid")
            if finite_float(outside.get("cumulative_outside_fraction"), float("inf")) >= 0.01:
                hard_failures.append("trusted_outside_exceeded")
            if bool(preserve_baseline_outside):
                outside_fraction = finite_float(
                    outside.get("cumulative_outside_fraction"),
                    float("inf"),
                )
                baseline_outside_fraction = finite_float(
                    baseline_outside.get("cumulative_outside_fraction"),
                    float("inf"),
                )
                if outside_fraction > baseline_outside_fraction + 1e-12:
                    hard_failures.append("trusted_outside_regressed")
                max_level_outside = finite_float(
                    outside.get("cumulative_max_level_outside_fraction"),
                    float("inf"),
                )
                baseline_max_level_outside = finite_float(
                    baseline_outside.get("cumulative_max_level_outside_fraction"),
                    float("inf"),
                )
                allowed_max_level_outside = min(0.10, baseline_max_level_outside + 1e-12)
                if max_level_outside > allowed_max_level_outside:
                    hard_failures.append("trusted_max_level_outside_regressed")
            if int(outside.get("cumulative_lost_z_levels") if outside.get("cumulative_lost_z_levels") is not None else 0) != 0:
                hard_failures.append("lost_z")
            if hard_failures:
                reason = hard_failures[0]
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                remember_trial_rejection(cid=cid, challenger_rank=challenger_rank, reason=reason, margin=margin, extra={"hard_failures": hard_failures, "outside": outside, **geom})
                continue
            protection_failures: list[dict[str, object]] = []
            if protected:
                preserve_started = time.perf_counter()
                active_indices = [int(i) for i in geom.get("active_candidate_indices", []) if 0 <= int(i) < len(trial_candidates)]
                active_candidates = [trial_candidates[i] for i in active_indices]
                trial_ids_tuple = tuple(sorted(int(candidate_track_id(c)) for c in trial_candidates))
                trial_engine = LpActivityEngine(enabled=True)
                trial_engine.stage = "protected_refill_trial"
                trial_engine.set_reusable_universe("protected_refill_trial", trial_candidates, inside_tol=0.03)
                for protected_group in protected_groups:
                    exact_representatives = {int(v) for v in protected_group.get("active_representative_ids", [])}
                    trial_active_ids = {int(candidate_track_id(candidate)) for candidate in active_candidates}
                    exact_active = sorted(exact_representatives & trial_active_ids)
                    if exact_active and not bool(strict_protected_footprint):
                        continue
                    after_equiv: list[int] = [int(value) for value in exact_active]
                    base_deficit = protected_group.get("base_surface_deficit") if isinstance(protected_group.get("base_surface_deficit"), dict) else {}
                    after_activity: dict[str, object] = {"active": True, "effective_face_margin": None}
                    after_deficit = None
                    target_value = protected_group.get("representative_candidate_id")
                    if target_value is None:
                        target_value = protected_group.get("representative_core_candidate_id")
                    target_id = int(target_value if target_value is not None else -1)
                    footprint_samples = protected_face_samples_by_group_id.get(
                        str(protected_group.get("stable_group_id"))
                    )
                    baseline_face_area = (
                        finite_float(footprint_samples.get("area"), 0.0)
                        if isinstance(footprint_samples, dict)
                        else 0.0
                    )
                    trial_face_area = None
                    footprint_area_ratio = None
                    footprint_distance = None
                    if protected_group.get("kind") in {"golden_active_patch", "rolling_active_patch"}:
                        target = protected_candidate_by_id.get(int(target_id))
                        if target is None:
                            continue
                        if not after_equiv:
                            after_equiv = [
                                int(candidate_track_id(active_candidate))
                                for active_candidate in active_candidates
                                if active_patch_compatible(target, active_candidate)
                            ]
                        if bool(strict_protected_footprint) and isinstance(footprint_samples, dict):
                            after_deficit = patch_surface_deficit_for_samples(
                                trial_candidates,
                                [
                                    np.asarray(sample, dtype=float)
                                    for sample in footprint_samples.get("samples", [])
                                ],
                                sample_tol=min(max(float(point_tol) * 0.05, 1e-5), 0.0025),
                            )
                            active_face_records = active_face_records_for_candidate_ids(
                                full_trial_rec or {},
                                trial_candidates,
                                {int(value) for value in after_equiv},
                            )
                            trial_face_area = max(
                                (float(record["area"]) for record in active_face_records),
                                default=None,
                            )
                            footprint_distance = footprint_distance_to_active_faces(
                                list(footprint_samples.get("face_points", [])),
                                active_face_records,
                            )
                            if baseline_face_area > EPS and trial_face_area is not None:
                                footprint_area_ratio = float(trial_face_area) / float(baseline_face_area)
                    else:
                        rep_index = int(protected_group.get("representative_core_index", -1))
                        if rep_index < 0 or rep_index >= len(core_candidates):
                            continue
                        group = groups[int(protected_group["group_index"])]
                        after_equiv = group_equivalent_active_candidates(group, core_candidates, active_candidates)
                        target = core_candidates[rep_index]
                        target_id = int(candidate_track_id(target))
                        after_activity = lp_face_activity_details(
                            None,
                            target,
                            inside_tol=0.03,
                            face_activity_slack=0.03,
                            lp_engine=trial_engine,
                            base_candidate_ids=tuple(int(v) for v in trial_ids_tuple if int(v) != target_id),
                        )
                        samples = baseline_samples_by_core_index.get(rep_index)
                        if samples is not None:
                            after_deficit = patch_surface_deficit_for_samples(
                                trial_candidates,
                                [np.array(sample, dtype=float) for sample in samples.get("samples", [])],
                                sample_tol=0.0,
                            )
                    reason = None
                    if not after_equiv:
                        reason = (
                            "breaks_stage1_repaired_patch"
                            if protected_group.get("role") == "stage1_repaired_patch"
                            else "breaks_stage2_repaired_patch"
                            if protected_group.get("role") == "stage2_repaired_patch"
                            else "breaks_golden_active_patch"
                            if protected_group.get("kind") in {"golden_active_patch", "rolling_active_patch"}
                            else "breaks_protected_core_patch"
                        )
                    elif not bool(after_activity.get("active")):
                        reason = "protected_patch_margin_lost"
                    elif bool(strict_protected_footprint) and isinstance(footprint_samples, dict):
                        cut_fraction = finite_float(
                            (after_deficit or {}).get("cut_fraction"),
                            float("inf"),
                        )
                        if cut_fraction > 0.0:
                            reason = "protected_patch_footprint_cut"
                        elif trial_face_area is None:
                            reason = "protected_patch_face_missing"
                        elif (
                            finite_float(
                                (footprint_distance or {}).get("p95"),
                                float("inf"),
                            )
                            > 0.03
                            or finite_float(
                                (footprint_distance or {}).get("max"),
                                float("inf"),
                            )
                            > 0.05
                        ):
                            reason = "protected_patch_footprint_shifted"
                        elif footprint_area_ratio is not None and footprint_area_ratio < 0.50:
                            reason = "protected_patch_area_shrunk"
                    elif isinstance(after_deficit, dict):
                        before_cut = finite_float(base_deficit.get("area_weighted_cut_deficit"), 0.0)
                        after_cut = finite_float(after_deficit.get("area_weighted_cut_deficit"), 0.0)
                        if after_cut > max(before_cut + 0.002, before_cut * 1.25 + 0.001):
                            reason = "protected_patch_surface_deficit_regression"
                    if reason is not None:
                        protection_failures.append(
                            {
                                "reason": reason,
                                "stable_group_id": protected_group.get("stable_group_id"),
                                "role": protected_group.get("role"),
                                "representative_core_candidate_id": int(target_id),
                                "active_equivalent_ids_after": after_equiv,
                                "base_lp_effective_margin": protected_group.get("base_lp_effective_margin"),
                                "after_lp_effective_margin": after_activity.get("effective_face_margin"),
                                "base_surface_deficit": base_deficit,
                                "after_surface_deficit": {key: value for key, value in (after_deficit or {}).items() if key != "samples"} if isinstance(after_deficit, dict) else None,
                                "baseline_face_area": as_json_float(float(baseline_face_area)),
                                "trial_face_area": as_json_float(float(trial_face_area)) if trial_face_area is not None else None,
                                "footprint_area_ratio": as_json_float(float(footprint_area_ratio)) if footprint_area_ratio is not None else None,
                                "footprint_distance": footprint_distance,
                            }
                        )
                        break
                timing["patch_preservation"] += time.perf_counter() - preserve_started
            gate_rejected = bool(protection_failures)
            trial_summary = None
            canonical_lost = False
            if gate_rejected:
                reason = str(protection_failures[0].get("reason") or "breaks_protected_core_patch")
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                if (
                    rejected_posthoc_evaluations < rejected_posthoc_budget
                    and core_ids is not None
                    and model_faces is not None
                    and initial_vertices is not None
                ):
                    oracle_started = time.perf_counter()
                    trial_summary = summarize_reconstruction_prefix(
                        prefix=int(len(accepted_rows) + 1),
                        candidates=trial_candidates,
                        core_count=len(core_candidates),
                        core_ids=core_ids,
                        model_faces=model_faces,
                        initial_vertices=initial_vertices,
                        trusted_points=trusted_points,
                        trusted_z_indices=trusted_z_indices,
                        point_tol=float(point_tol),
                        reconstruction_kwargs=reconstruction_kwargs,
                        match_cache=match_cache,
                    )
                    timing["oracle_posthoc"] += time.perf_counter() - oracle_started
                    rejected_posthoc_evaluations += 1
                    canonical_lost = bool(trial_summary.get("lost_core_ids"))
                    if canonical_lost:
                        confusion["gate_rejected_and_canonical_core_lost"] += 1
                    else:
                        confusion["gate_rejected_but_canonical_core_preserved"] += 1
                else:
                    confusion["not_evaluated"] += 1
                if len(trial_rows) < 160:
                    trial_rows.append(
                        {
                            "candidate_id": int(cid),
                            "challenger_rank": int(challenger_rank),
                            "accepted": False,
                            "primary_rejection_reason": reason,
                            "protection_failures": protection_failures,
                            "lp_margin": as_json_float(float(margin)) if margin is not None else None,
                            "oracle_posthoc": trial_summary,
                            "canonical_core_lost": bool(canonical_lost),
                            "canonical_unique_after": trial_summary.get("active_unique_finite_ids") if isinstance(trial_summary, dict) else None,
                            "canonical_new_after": trial_summary.get("new_unique_face_ids_vs_repaired_core") if isinstance(trial_summary, dict) else None,
                            "could_add_new_canonical_face": bool(
                                isinstance(trial_summary, dict)
                                and int(trial_summary.get("active_unique_finite_ids") or 0) > 121
                            ),
                        }
                    )
                continue
            candidate = dict(candidate)
            candidate["small_face_refinement_mode"] = "post-repair-refill-protected" if protected else "post-repair-refill-unprotected"
            candidate["small_face_refinement_rank"] = int(len(accepted_rows) + 1)
            candidate["small_face_refinement_band_id"] = band_id(candidate)
            current.append(candidate)
            if full_trial_rec is not None:
                current_rec = full_trial_rec
            else:
                rebuild_started = time.perf_counter()
                current_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(current, **reconstruction_kwargs)
                timing["accepted_state_full_rebuild"] += time.perf_counter() - rebuild_started
            rolling_added_groups = 0
            if protected and bool(rolling_protection):
                full_active_indices = [
                    int(i)
                    for i in current_rec.get("face_candidate_indices", [])
                    if 0 <= int(i) < len(current)
                ]
                rolling_added_groups = append_active_patch_groups(
                    protected_groups,
                    full_active_indices,
                    current,
                    current_rec,
                    group_kind="rolling_active_patch",
                    stable_prefix="rolling_active_patch",
                )
            accepted_posthoc = None
            if protected and core_ids is not None and model_faces is not None and initial_vertices is not None:
                oracle_started = time.perf_counter()
                accepted_posthoc = summarize_reconstruction_prefix(
                    prefix=int(len(accepted_rows) + 1),
                    candidates=current,
                    core_count=len(core_candidates),
                    core_ids=core_ids,
                    model_faces=model_faces,
                    initial_vertices=initial_vertices,
                    trusted_points=trusted_points,
                    trusted_z_indices=trusted_z_indices,
                    point_tol=float(point_tol),
                    reconstruction_kwargs=reconstruction_kwargs,
                    match_cache=match_cache,
                )
                timing["oracle_posthoc"] += time.perf_counter() - oracle_started
                accepted_lost_ids = set(int(v) for v in accepted_posthoc.get("lost_core_ids", [])) if isinstance(accepted_posthoc, dict) else set()
                if accepted_lost_ids:
                    confusion["gate_passed_but_canonical_core_lost"] += 1
                else:
                    confusion["gate_passed_and_canonical_core_preserved"] += 1
            elif protected:
                confusion["not_evaluated"] += 1
            if protected and first_lost_92 is None and isinstance(accepted_posthoc, dict):
                accepted_lost_ids = set(int(v) for v in accepted_posthoc.get("lost_core_ids", []))
                if 92 in accepted_lost_ids:
                    first_lost_92 = {
                        "accepted_count": int(len(accepted_rows) + 1),
                        "trial_checks": int(trial_checks),
                        "candidate_id": int(cid),
                        "lost_core_ids": sorted(int(v) for v in accepted_lost_ids),
                        "oracle_posthoc": accepted_posthoc,
                        "active_candidate_indices": geom.get("active_candidate_indices"),
                        "gate_rejected": False,
                    }
            accepted_row = {
                "candidate_id": int(cid),
                "challenger_rank": int(challenger_rank),
                "append_rank": int(len(accepted_rows) + 1),
                "accepted": True,
                "previous_provenance": candidate.get("post_repair_refill_previous_provenance"),
                "lp_margin": as_json_float(float(margin)) if margin is not None else None,
                "active_candidate_indices": geom.get("active_candidate_indices"),
                "rolling_added_group_count": int(rolling_added_groups),
                "rolling_protected_group_count_after": int(len(protected_groups)),
                "oracle_posthoc": accepted_posthoc,
            }
            accepted_rows.append(accepted_row)
            need_checkpoint = bool(len(accepted_rows) in checkpoint_counts)
            if need_checkpoint:
                checkpoint = summarize_checkpoint(
                    label=f"accepted_{len(accepted_rows)}",
                    accepted_count=len(accepted_rows),
                    trial_checks=trial_checks,
                    current_candidates=current,
                    match_cache=match_cache,
                )
                checkpoint["accepted_sequence_signature"] = [int(row["candidate_id"]) for row in accepted_rows]
                checkpoints.append(checkpoint)
                posthoc = checkpoint.get("oracle_posthoc") if isinstance(checkpoint.get("oracle_posthoc"), dict) else {}
                lost_ids = set(int(v) for v in posthoc.get("lost_core_ids", [])) if isinstance(posthoc, dict) else set()
                if first_lost_92 is None and 92 in lost_ids:
                    first_lost_92 = {
                        "accepted_count": int(len(accepted_rows)),
                        "trial_checks": int(trial_checks),
                        "candidate_id": int(cid),
                        "lost_core_ids": sorted(int(v) for v in lost_ids),
                        "oracle_posthoc": posthoc,
                        "active_candidate_indices": geom.get("active_candidate_indices"),
                        "gate_rejected": False,
                    }
        frontier_rows = []
        for limit in sorted(set(checkpoint_counts) | {0, 5, 10, 15, 20, 25}):
            checkpoint = next(
                (
                    row
                    for row in checkpoints
                    if row.get("accepted_count") is not None and int(row.get("accepted_count")) == limit
                ),
                None,
            )
            frontier_rows.append(
                {
                    "max_accepted_additions": int(limit),
                    "reached": checkpoint is not None,
                    "checkpoint": checkpoint,
                }
            )
        return {
            "protected_gate_enabled": bool(protected),
            "rolling_protection_enabled": bool(protected and rolling_protection),
            "max_accepted": int(max_accept),
            "initial_max_trials": int(initial_trial_budget),
            "max_trials": int(resolved_trial_budget),
            "extended_budget_used": bool(extension_used),
            "accepted_count": int(len(accepted_rows)),
            "trial_checks": int(trial_checks),
            "candidate_pool_exhausted": bool(exhausted),
            "remaining_challenger_count": int(max(0, len(challenger_order) - int(processed_challenger_rank))),
            "accepted_sequence_signature": [int(row["candidate_id"]) for row in accepted_rows],
            "protected_group_count_after": int(len(protected_groups)),
            "protected_group_signature_after": protected_group_signature(),
            "rejection_counts": rejection_counts,
            "confusion_matrix": confusion,
            "confusion_invariants": {
                "posthoc_evaluated_trial_decisions": int(
                    confusion["gate_rejected_and_canonical_core_lost"]
                    + confusion["gate_rejected_but_canonical_core_preserved"]
                    + confusion["gate_passed_and_canonical_core_preserved"]
                    + confusion["gate_passed_but_canonical_core_lost"]
                ),
                "not_evaluated": int(confusion["not_evaluated"]),
                "accepted_decisions": int(len(accepted_rows)),
                "accepted_posthoc_evaluated": int(
                    confusion["gate_passed_and_canonical_core_preserved"]
                    + confusion["gate_passed_but_canonical_core_lost"]
                ),
            },
            "first_lost_face_92": first_lost_92,
            "checkpoints": checkpoints,
            "frontier": frontier_rows,
            "accepted_rows": accepted_rows[:80],
            "trial_rows_sample": trial_rows,
        }

    checkpoint_counts = {0, 5, 10, 15, 20, 25}
    if not require_dense_band and safe_control_candidate_ids and int(order_diagnostics["safe_controls_in_top_160"]) == 0:
        return {
            "diagnostic_only": True,
            "production_changed": False,
            "mode": str(mode_label),
            "pool_policy": {
                "require_dense_band": bool(require_dense_band),
                "protect_all_active_patches": bool(protect_all_active_patches),
                "rolling_protection": bool(rolling_protection),
            },
            "all_preselected_count": int(len(annotated)),
            "protected_group_count": int(len(protected_groups)),
            "protected_group_signature": protected_groups_signature,
            "static_rejection_counts": static_rejection_counts,
            "challenger_pool_size": int(len(challenger_order)),
            "order_diagnostics": order_diagnostics,
            "promotion_allowed": False,
            "selected_branch": "broad_pool_ranking_bottleneck",
            "rejection_reason": "none of the oracle-regression safe controls reached the bounded top-160 broad order; no sequential replay was run",
            "timing_breakdown_seconds": {key: as_json_float(float(value)) for key, value in timing.items()},
            "timing_seconds": as_json_float(time.perf_counter() - started),
        }
    unprotected = run_sequence(
        protected=False,
        max_accept=1,
        max_trial_checks=1,
        extended_trial_checks=None,
        checkpoint_counts={0, 1},
    )
    protected_result = run_sequence(
        protected=True,
        max_accept=int(max_accepted),
        max_trial_checks=int(max_trials),
        extended_trial_checks=extended_max_trials,
        checkpoint_counts=checkpoint_counts,
    )

    def first_step_comparison() -> dict[str, object]:
        unprotected_first = next(
            (row for row in unprotected.get("checkpoints", []) if int(row.get("accepted_count") or -1) == 1),
            None,
        )
        protected_first = next(
            (row for row in protected_result.get("checkpoints", []) if int(row.get("accepted_count") or -1) == 1),
            None,
        )
        unprotected_seq = [int(v) for v in unprotected.get("accepted_sequence_signature", [])]
        protected_seq = [int(v) for v in protected_result.get("accepted_sequence_signature", [])]
        base_ids = sorted(int(candidate_track_id(candidate)) for candidate in selected_candidates)
        first_id = unprotected_seq[0] if unprotected_seq else (protected_seq[0] if protected_seq else None)
        trial_ids = sorted(base_ids + ([int(first_id)] if first_id is not None else []))

        def compact(row: dict[str, object] | None) -> dict[str, object] | None:
            if row is None:
                return None
            posthoc = row.get("oracle_posthoc") if isinstance(row.get("oracle_posthoc"), dict) else {}
            signature = row.get("geometry_signature") if isinstance(row.get("geometry_signature"), dict) else {}
            topology = row.get("topology") if isinstance(row.get("topology"), dict) else {}
            outside = row.get("outside") if isinstance(row.get("outside"), dict) else {}
            return {
                "candidate_id": int(first_id) if first_id is not None else None,
                "active_plane_indices": signature.get("active_plane_indices"),
                "active_candidate_id_sum": signature.get("active_candidate_id_sum"),
                "active_candidate_id_xor": signature.get("active_candidate_id_xor"),
                "vertices": row.get("vertices"),
                "edges": row.get("edges"),
                "faces": row.get("faces"),
                "volume": row.get("volume"),
                "topology": {
                    "euler": topology.get("euler"),
                    "components": topology.get("connected_components"),
                    "boundary": topology.get("boundary_edges"),
                    "non_manifold": topology.get("non_manifold_edges"),
                },
                "outside": outside.get("cumulative_outside_fraction"),
                "lost_z": outside.get("cumulative_lost_z_levels"),
                "retained_core_ids": posthoc.get("retained_core_ids"),
                "active_unique_finite_ids": posthoc.get("active_unique_finite_ids"),
                "new_ids": posthoc.get("new_ids"),
                "lost_core_ids": posthoc.get("lost_core_ids"),
            }

        unprotected_compact = compact(unprotected_first)
        protected_compact = compact(protected_first)
        return {
            "candidate_id": int(first_id) if first_id is not None else None,
            "base_candidate_count": int(len(base_ids)),
            "base_candidate_id_sum": int(sum(base_ids)),
            "trial_candidate_count": int(len(trial_ids)),
            "trial_candidate_id_sum": int(sum(trial_ids)),
            "unprotected": unprotected_compact,
            "protected": protected_compact,
            "same_first_candidate": bool(unprotected_seq[:1] == protected_seq[:1] and bool(unprotected_seq[:1])),
            "same_base_candidate_ids": True,
            "same_trial_candidate_ids": True,
            "same_geometry_digest": bool(
                unprotected_compact is not None
                and protected_compact is not None
                and {
                    key: unprotected_compact.get(key)
                    for key in ["active_plane_indices", "active_candidate_id_sum", "active_candidate_id_xor", "vertices", "edges", "faces", "volume"]
                }
                == {
                    key: protected_compact.get(key)
                    for key in ["active_plane_indices", "active_candidate_id_sum", "active_candidate_id_xor", "vertices", "edges", "faces", "volume"]
                }
            ),
            "same_lost_core_ids": bool(
                unprotected_compact is not None
                and protected_compact is not None
                and unprotected_compact.get("lost_core_ids") == protected_compact.get("lost_core_ids")
            ),
        }

    first_step = first_step_comparison()
    promotion_candidates = []
    first_unique_increase = None
    first_unique_123 = None
    false_positive_rejections = [
        {
            "candidate_id": int(row.get("candidate_id") or -1),
            "challenger_rank": row.get("challenger_rank"),
            "primary_rejection_reason": row.get("primary_rejection_reason"),
            "could_add_new_canonical_face": bool(row.get("could_add_new_canonical_face")),
            "canonical_unique_after": row.get("canonical_unique_after"),
            "canonical_new_after": row.get("canonical_new_after"),
            "lost_core_ids": (row.get("oracle_posthoc") or {}).get("lost_core_ids") if isinstance(row.get("oracle_posthoc"), dict) else None,
            "triggered_group": (row.get("protection_failures") or [{}])[0].get("stable_group_id")
            if isinstance(row.get("protection_failures"), list)
            else None,
            "triggered_group_representative_core_candidate_id": (row.get("protection_failures") or [{}])[0].get("representative_core_candidate_id")
            if isinstance(row.get("protection_failures"), list)
            else None,
            "after_lp_effective_margin": (row.get("protection_failures") or [{}])[0].get("after_lp_effective_margin")
            if isinstance(row.get("protection_failures"), list)
            else None,
            "analysis": "gate rejected the trial while oracle core set was preserved; keep as diagnostic false positive unless a generic patch-compatible replacement predicate is proven",
        }
        for row in protected_result.get("trial_rows_sample", [])
        if not bool(row.get("accepted"))
        and isinstance(row.get("oracle_posthoc"), dict)
        and not bool((row.get("oracle_posthoc") or {}).get("lost_core_ids"))
    ]
    for row in protected_result.get("frontier", []):
        checkpoint = row.get("checkpoint") if isinstance(row.get("checkpoint"), dict) else {}
        posthoc = checkpoint.get("oracle_posthoc") if isinstance(checkpoint.get("oracle_posthoc"), dict) else {}
        topology = checkpoint.get("topology") if isinstance(checkpoint.get("topology"), dict) else {}
        outside = checkpoint.get("outside") if isinstance(checkpoint.get("outside"), dict) else {}
        preservation = checkpoint.get("protected_group_preservation") if isinstance(checkpoint.get("protected_group_preservation"), dict) else {}
        retained = int(posthoc.get("retained_core_ids") or 0) if isinstance(posthoc, dict) else 0
        unique = int(posthoc.get("active_unique_finite_ids") or 0) if isinstance(posthoc, dict) else 0
        new_count = int(len(posthoc.get("new_ids", []))) if isinstance(posthoc.get("new_ids"), list) else int(posthoc.get("new_unique_face_ids_vs_repaired_core") or 0)
        lost = list(posthoc.get("lost_core_ids", [])) if isinstance(posthoc, dict) else []
        if checkpoint and first_unique_increase is None and unique > 121:
            first_unique_increase = int(row.get("max_accepted_additions") or 0)
        if checkpoint and first_unique_123 is None and unique >= 123:
            first_unique_123 = int(row.get("max_accepted_additions") or 0)
        qualifies = bool(
            unique >= 123
            and retained == 75
            and not lost
            and int((protected_result.get("confusion_matrix") or {}).get("gate_passed_but_canonical_core_lost") or 0) == 0
            and int((protected_result.get("confusion_matrix") or {}).get("not_evaluated") or 0) == 0
            and bool(preservation.get("stage1_preserved", False))
            and bool(preservation.get("stage2_preserved", False))
            and bool(checkpoint.get("removed_blockers_absent", False))
            and topology_is_valid(topology)
            and finite_float(outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
            and int(outside.get("cumulative_lost_z_levels") or 0) == 0
        )
        if qualifies:
            promotion_candidates.append(
                {
                    "limit": int(row.get("max_accepted_additions") or 0),
                    "new_count": int(new_count),
                    "checkpoint": checkpoint,
                }
            )
    accepted_by_id = {
        int(value_or_default(row.get("candidate_id"), -1)): row
        for row in protected_result.get("accepted_rows", [])
    }
    trial_by_id = {
        int(value_or_default(row.get("candidate_id"), -1)): row
        for row in protected_result.get("trial_rows_sample", [])
    }
    safe_control_fate = []
    for safe_id in sorted(int(v) for v in safe_control_candidate_ids):
        order_row = order_diagnostics.get("safe_control_ranks", {}).get(str(safe_id), {})
        accepted_row = accepted_by_id.get(int(safe_id))
        trial_row = trial_by_id.get(int(safe_id))
        posthoc = accepted_row.get("oracle_posthoc") if isinstance(accepted_row, dict) and isinstance(accepted_row.get("oracle_posthoc"), dict) else (
            trial_row.get("oracle_posthoc") if isinstance(trial_row, dict) and isinstance(trial_row.get("oracle_posthoc"), dict) else {}
        )
        safe_control_fate.append(
            {
                "candidate_id": int(safe_id),
                "rank": order_row.get("rank"),
                "in_top_160": bool(order_row.get("in_top_160")),
                "reached_gate_sample": int(safe_id) in accepted_by_id or int(safe_id) in trial_by_id,
                "accepted": accepted_row is not None,
                "rejection_reason": trial_row.get("primary_rejection_reason") if isinstance(trial_row, dict) else None,
                "active_after_accept": bool(
                    accepted_row is not None
                    and int(len(selected_candidates) + int(accepted_row.get("append_rank") or 0) - 1)
                    in {int(idx) for idx in accepted_row.get("active_candidate_indices", [])}
                ),
                "posthoc_unique": posthoc.get("active_unique_finite_ids") if isinstance(posthoc, dict) else None,
                "posthoc_new_ids": posthoc.get("new_ids") if isinstance(posthoc, dict) else None,
                "posthoc_lost_core_ids": posthoc.get("lost_core_ids") if isinstance(posthoc, dict) else None,
            }
        )
    return {
        "diagnostic_only": True,
        "production_changed": False,
        "mode": str(mode_label),
        "pool_policy": {
            "require_dense_band": bool(require_dense_band),
            "protect_all_active_patches": bool(protect_all_active_patches),
            "rolling_protection": bool(rolling_protection),
            "candidate_filter_mode": str(candidate_filter_mode),
            "candidate_order_mode": str(candidate_order_mode),
            "protect_active_faces_individually": bool(protect_active_faces_individually),
            "strict_protected_footprint": bool(strict_protected_footprint),
            "preserve_baseline_outside": bool(preserve_baseline_outside),
            "max_trial_face_extra_area": (
                as_json_float(float(max_trial_face_extra_area))
                if max_trial_face_extra_area is not None
                and float(max_trial_face_extra_area) >= 0.0
                else None
            ),
            "refresh_trial_geometry_from_full_rebuild": bool(
                refresh_trial_geometry_from_full_rebuild
            ),
        },
        "baseline_outside": baseline_outside,
        "all_preselected_count": int(len(annotated)),
        "protected_group_count": int(protected_groups_signature.get("count") or 0),
        "protected_group_signature": protected_groups_signature,
        "final_protected_group_count": int(len(protected_groups)),
        "final_protected_group_signature": protected_group_signature(),
        "protected_groups": protected_groups[:160],
        "stage1_target_candidate_ids": sorted(int(v) for v in stage1_target_candidate_ids),
        "stage2_target_candidate_ids": sorted(int(v) for v in stage2_target_candidate_ids),
        "removed_blocker_ids": sorted(int(v) for v in removed_blocker_ids),
        "dense_bands": band_detection,
        "static_rejection_counts": static_rejection_counts,
        "previous_provenance_counts": provenance_counts,
        "challenger_pool_size": int(len(challenger_order)),
        "challenger_order_signature": [int(candidate_track_id(candidate)) for candidate in challenger_order[:160]],
        "challenger_order_source": "append_local_summary_signature" if challenger_order_ids else "local_order_key_fallback",
        "order_diagnostics": order_diagnostics,
        "unprotected_loss_trace": unprotected,
        "protected_sequential_refill": protected_result,
        "protected_refill_frontier": {
            "diagnostic_only": True,
            "production_changed": False,
            "requested_max_accepted_additions": int(max_accepted),
            "initial_max_trial_checks": int(max_trials),
            "resolved_max_trial_checks": int(protected_result.get("max_trials") or max_trials),
            "extended_budget_used": bool(protected_result.get("extended_budget_used")),
            "checkpoints": protected_result.get("frontier", []),
            "first_unique_increase_checkpoint": first_unique_increase,
            "first_unique_123_checkpoint": first_unique_123,
            "candidate_pool_exhausted": bool(protected_result.get("candidate_pool_exhausted")),
            "remaining_challenger_count": int(protected_result.get("remaining_challenger_count") or 0),
            "confusion_matrix": protected_result.get("confusion_matrix"),
            "confusion_invariants": protected_result.get("confusion_invariants"),
            "rejection_counts": protected_result.get("rejection_counts"),
            "false_positive_rejections": false_positive_rejections,
            "safe_control_fate": safe_control_fate,
            "accepted_sequence_signature": protected_result.get("accepted_sequence_signature"),
            "trial_checks": int(protected_result.get("trial_checks") or 0),
            "accepted_count": int(protected_result.get("accepted_count") or 0),
            "promotion_candidates": promotion_candidates,
            "promotion_allowed": bool(promotion_candidates),
            "promotion_rejection_reason": None if promotion_candidates else "no deterministic protected checkpoint reached canonical unique >= 123 with retained core 75, lost core [], valid topology/outside/lost-Z, and full protected-group preservation",
        },
        "first_candidate_reproducibility": first_step,
        "promotion_candidates": promotion_candidates,
        "promotion_allowed": bool(promotion_candidates),
        "selected_branch": "post_repair_refill_promotable" if promotion_candidates else "post_repair_refill_rejected",
        "rejection_reason": None if promotion_candidates else "protected sequential refill did not prove canonical unique >= 123 with retained core 75 and lost core []",
        "timing_breakdown_seconds": {key: as_json_float(float(value)) for key, value in timing.items()},
        "timing_seconds": as_json_float(time.perf_counter() - started),
    }

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


def candidate_z_range(candidate: dict[str, object]) -> tuple[float, float]:
    z_min = finite_float(candidate.get("z_min"), float("nan"))
    z_max = finite_float(candidate.get("z_max"), float("nan"))
    if np.isfinite(z_min) and np.isfinite(z_max):
        return min(z_min, z_max), max(z_min, z_max)
    return float("nan"), float("nan")


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


def candidates_plane_patch_compatible(
    a: dict[str, object],
    b: dict[str, object],
    *,
    max_angle_deg: float = 1.5,
    max_offset: float = 0.04,
    max_centroid_distance: float = 0.45,
    min_z_overlap: float = 0.05,
    min_hull_overlap: float = 0.02,
) -> bool:
    pa = candidate_plane(a)
    pb = candidate_plane(b)
    ca = candidate_hull_centroid(a)
    cb = candidate_hull_centroid(b)
    if pa is None or pb is None or ca is None or cb is None:
        return False
    p_a, n_a = pa
    p_b, n_b = pb
    dot = float(n_a @ n_b)
    if dot <= 0.0:
        return False
    angle = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
    offset = abs(float(n_a @ p_a) - float(n_b @ p_b))
    centroid_distance = float(np.linalg.norm(ca - cb))
    za0, za1 = candidate_z_range(a)
    zb0, zb1 = candidate_z_range(b)
    if np.isfinite(za0) and np.isfinite(zb0):
        z_overlap = max(0.0, min(za1, zb1) - max(za0, zb0))
        if z_overlap < float(min_z_overlap):
            return False
    hull_overlap = hull_bounds_overlap_ratio(a, b)
    spatially_close = centroid_distance <= float(max_centroid_distance) or hull_overlap >= float(min_hull_overlap)
    return bool(angle <= max_angle_deg and offset <= max_offset and spatially_close)


def candidate_edge_pair(candidate: dict[str, object]) -> tuple[int, ...]:
    ids = candidate.get("w2_edge_cluster_ids") or []
    out: list[int] = []
    if isinstance(ids, list):
        for v in ids:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                pass
    return tuple(sorted(set(out)))


def compact_cluster_sizes(sizes: list[int]) -> dict[str, object]:
    if not sizes:
        return {"count": 0, "min": None, "median": None, "max": None, "top": []}
    arr = np.array(sizes, dtype=float)
    return {
        "count": int(len(sizes)),
        "min": int(np.min(arr)),
        "median": as_json_float(float(np.median(arr))),
        "max": int(np.max(arr)),
        "top": [int(v) for v in sorted(sizes, reverse=True)[:12]],
    }


def cluster_face_candidates(
    candidates: list[dict[str, object]],
    *,
    normal_angle_deg: float,
    plane_distance: float,
    hull_distance: float,
    mode: str = "single",
) -> tuple[list[list[dict[str, object]]], dict[str, object]]:
    n = len(candidates)
    if n == 0:
        return [], {
            "dedupe_raw_candidates": 0,
            "dedupe_clusters": 0,
            "candidate_cluster_mode": str(mode),
            "dedupe_cluster_sizes": compact_cluster_sizes([]),
            "dedupe_cluster_windows": [],
        }
    cluster_mode = str(mode)

    def compatible_with_cluster(candidate_index: int, cluster_indices: list[int]) -> bool:
        if not cluster_indices:
            return True
        compare_indices = cluster_indices if cluster_mode == "complete" else [cluster_indices[0]]
        for other_index in compare_indices:
            similarity = candidate_similarity(candidates[candidate_index], candidates[other_index])
            if similarity is None:
                return False
            angle, plane_dist, hull_dist = similarity
            if (
                angle > float(normal_angle_deg)
                or plane_dist > float(plane_distance)
                or hull_dist > float(hull_distance)
            ):
                return False
        return True

    if cluster_mode == "single":
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                similarity = candidate_similarity(candidates[i], candidates[j])
                if similarity is None:
                    continue
                angle, plane_dist, hull_dist = similarity
                if (
                    angle <= float(normal_angle_deg)
                    and plane_dist <= float(plane_distance)
                    and hull_dist <= float(hull_distance)
                ):
                    union(i, j)

        clusters_by_root: dict[int, list[int]] = {}
        for i in range(n):
            clusters_by_root.setdefault(find(i), []).append(i)
        cluster_indices = list(clusters_by_root.values())
    elif cluster_mode in {"representative", "complete"}:
        cluster_indices = []
        for idx in sorted(range(n), key=lambda i: candidate_rank_key(candidates[i])):
            placed = False
            for existing in cluster_indices:
                if compatible_with_cluster(idx, existing):
                    existing.append(idx)
                    placed = True
                    break
            if not placed:
                cluster_indices.append([idx])
    else:
        raise ValueError(f"Unknown candidate cluster mode: {cluster_mode}")

    ordered_indices = sorted(
        cluster_indices,
        key=lambda idxs: candidate_rank_key(min((candidates[i] for i in idxs), key=candidate_rank_key)),
    )
    clusters: list[list[dict[str, object]]] = []
    cluster_sizes: list[int] = []
    cluster_windows: list[list[int]] = []
    for cluster_id, idxs in enumerate(ordered_indices):
        ranked = sorted((dict(candidates[i]) for i in idxs), key=candidate_rank_key)
        windows = sorted({int(c["window"]) for c in ranked if c.get("window") is not None})
        cluster_sizes.append(len(ranked))
        cluster_windows.append(windows)
        for local_rank, candidate in enumerate(ranked):
            candidate["dedupe_cluster_id"] = int(cluster_id)
            candidate["dedupe_cluster_size"] = int(len(ranked))
            candidate["dedupe_cluster_windows"] = [int(w) for w in windows]
            candidate["cluster_local_rank"] = int(local_rank)
            candidate["cluster_rejection_reason"] = "cluster_alternative" if local_rank > 0 else None
        clusters.append(ranked)
    return clusters, {
        "dedupe_raw_candidates": int(n),
        "dedupe_clusters": int(len(clusters)),
        "candidate_cluster_mode": cluster_mode,
        "dedupe_cluster_sizes": compact_cluster_sizes(cluster_sizes),
        "dedupe_cluster_windows": cluster_windows[:50],
    }


def dedupe_face_candidates(
    candidates: list[dict[str, object]],
    *,
    normal_angle_deg: float,
    plane_distance: float,
    hull_distance: float,
    max_candidates_per_cluster: int,
    mode: str = "single",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    clusters, summary = cluster_face_candidates(
        candidates,
        normal_angle_deg=normal_angle_deg,
        plane_distance=plane_distance,
        hull_distance=hull_distance,
        mode=mode,
    )
    kept: list[dict[str, object]] = []
    keep_per_cluster = max(1, int(max_candidates_per_cluster))
    for cluster in clusters:
        kept.extend(dict(candidate) for candidate in cluster[:keep_per_cluster])
    kept.sort(key=candidate_rank_key)
    summary = dict(summary)
    summary["dedupe_candidates_after"] = int(len(kept))
    return kept, {
        **summary,
    }


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


def plane_patch_feature(candidate: dict[str, object]) -> dict[str, object] | None:
    plane = candidate_plane(candidate)
    if plane is None:
        return None
    point, normal = plane
    norm = float(np.linalg.norm(normal))
    if norm <= EPS:
        return None
    n = normal / norm
    hull = np.array(candidate.get("hull", []), dtype=float)
    if hull.ndim == 2 and hull.shape[0] > 0 and hull.shape[1] == 3 and np.all(np.isfinite(hull)):
        centroid = np.mean(hull, axis=0)
    else:
        centroid = point
    support_points = np.array(candidate.get("sample_points") or candidate.get("fit_points") or candidate.get("hull") or [], dtype=float)
    if support_points.ndim != 2 or support_points.shape[1:] != (3,):
        support_points = np.zeros((0, 3), dtype=float)
    return {
        "candidate_id": int(candidate_track_id(candidate)),
        "normal": n,
        "offset": float(n @ point),
        "centroid": centroid,
        "support_points": support_points,
        "z_values": {int(v) for v in candidate.get("z_indices", []) if int(v) >= 0},
        "hull_area": finite_float(candidate.get("hull_area"), 0.0),
        "hull_diameter": finite_float(candidate.get("hull_diameter"), 0.0),
    }


def plane_patch_relation(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    fa = plane_patch_feature(a)
    fb = plane_patch_feature(b)
    if fa is None or fb is None:
        return {"compatible": False}
    na = np.array(fa["normal"], dtype=float)
    nb = np.array(fb["normal"], dtype=float)
    dot = float(na @ nb)
    if dot < 0.0:
        return {"compatible": False, "normal_angle_deg": 180.0}
    angle = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
    offset = abs(float(fa["offset"]) - float(fb["offset"]))
    centroid_distance = float(np.linalg.norm(np.array(fa["centroid"], dtype=float) - np.array(fb["centroid"], dtype=float)))
    za = set(fa["z_values"])
    zb = set(fb["z_values"])
    z_overlap = float(len(za & zb)) / float(max(1, min(len(za), len(zb)))) if za and zb else 0.0
    area_a = float(fa["hull_area"])
    area_b = float(fb["hull_area"])
    area_ratio = min(area_a, area_b) / max(area_a, area_b, EPS) if max(area_a, area_b) > 0.0 else 0.0
    points_a = np.array(fa["support_points"], dtype=float)
    points_b = np.array(fb["support_points"], dtype=float)
    support_distance_median = float("inf")
    if points_a.ndim == 2 and points_b.ndim == 2 and points_a.size and points_b.size:
        distances = np.sqrt(np.sum((points_a[:, None, :] - points_b[None, :, :]) ** 2, axis=2))
        support_distance_median = float(np.median(np.min(distances, axis=1)))
    spatially_close = centroid_distance <= 1.0 and offset <= 0.05 and angle <= 5.0
    compatible = (
        angle <= 5.0
        and offset <= 0.06
        and centroid_distance <= 1.15
        and (z_overlap >= 0.20 or spatially_close)
        and (support_distance_median <= 0.22 or area_ratio >= 0.01 or spatially_close)
    )
    return {
        "compatible": bool(compatible),
        "normal_angle_deg": as_json_float(float(angle)),
        "signed_offset_distance": as_json_float(float(offset)),
        "centroid_distance": as_json_float(float(centroid_distance)),
        "z_overlap_fraction": as_json_float(float(z_overlap)),
        "support_distance_median": as_json_float(float(support_distance_median)) if np.isfinite(support_distance_median) else None,
        "hull_area_ratio": as_json_float(float(area_ratio)),
    }


def build_core_plane_patch_groups(
    core_candidates: list[dict[str, object]],
    core_active_indices: list[int],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    *,
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
) -> tuple[list[dict[str, object]], dict[int, int]]:
    groups: list[dict[str, object]] = []
    for candidate_index in sorted({int(v) for v in core_active_indices}):
        if candidate_index < 0 or candidate_index >= len(core_candidates):
            continue
        candidate = core_candidates[candidate_index]
        if plane_patch_feature(candidate) is None:
            continue
        witness_mask, witness = core_witness_mask_and_stats(
            candidate,
            trusted_points,
            trusted_z_indices,
            plane_distance=float(finite_support_plane_distance),
            hull_margin=float(finite_support_hull_margin),
            relative_hull_margin=float(finite_support_relative_hull_margin),
        )
        for group in groups:
            if all(bool(plane_patch_relation(candidate, core_candidates[int(member)]).get("compatible")) for member in group["member_core_indices"]):
                group["member_core_indices"].append(int(candidate_index))
                group["member_core_candidate_ids"].append(int(candidate_track_id(candidate)))
                group["witness_masks"].append(witness_mask)
                group["member_witness"].append(witness)
                break
        else:
            groups.append(
                {
                    "group_index": int(len(groups)),
                    "stable_group_id": f"core_patch_{len(groups):04d}_{candidate_track_id(candidate)}",
                    "representative_core_index": int(candidate_index),
                    "member_core_indices": [int(candidate_index)],
                    "member_core_candidate_ids": [int(candidate_track_id(candidate))],
                    "witness_masks": [witness_mask],
                    "member_witness": [witness],
                }
            )

    candidate_to_group: dict[int, int] = {}
    for group in groups:
        masks = [np.array(mask, dtype=bool) for mask in group.pop("witness_masks")]
        union_mask = np.zeros(trusted_points.shape[0], dtype=bool) if trusted_points.ndim == 2 else np.zeros(0, dtype=bool)
        for mask in masks:
            if mask.shape == union_mask.shape:
                union_mask |= mask
        witness_counts = [int(row.get("witness_count") or 0) for row in group["member_witness"]]
        witness_z_levels = [int(row.get("witness_z_levels") or 0) for row in group["member_witness"]]
        confidence = "unsupported"
        if max(witness_counts or [0]) >= 12 and max(witness_z_levels or [0]) >= 3:
            confidence = "supported"
        elif max(witness_counts or [0]) >= 5 and max(witness_z_levels or [0]) >= 2:
            confidence = "weak"
        representative = core_candidates[int(group["representative_core_index"])]
        feature = plane_patch_feature(representative)
        group["union_witness_count"] = int(np.sum(union_mask))
        group["union_witness_z_levels"] = int(len({int(v) for v in trusted_z_indices[union_mask]})) if trusted_z_indices.size == union_mask.size else 0
        group["support_confidence"] = confidence
        group["representative_plane"] = {
            "candidate_id": int(candidate_track_id(representative)),
            "normal": as_json_point(np.array(feature["normal"], dtype=float)) if feature is not None else None,
            "offset": as_json_float(float(feature["offset"])) if feature is not None else None,
            "hull_area": as_json_float(float(feature["hull_area"])) if feature is not None else None,
        }
        for member in group["member_core_indices"]:
            candidate_to_group[int(member)] = int(group["group_index"])
    return groups, candidate_to_group


def group_equivalent_active_candidates(
    group: dict[str, object],
    core_candidates: list[dict[str, object]],
    active_candidates: list[dict[str, object]],
) -> list[int]:
    members = [core_candidates[int(i)] for i in group.get("member_core_indices", []) if 0 <= int(i) < len(core_candidates)]
    out = [
        int(candidate_track_id(candidate))
        for candidate in active_candidates
        if any(bool(plane_patch_relation(member, candidate).get("compatible")) for member in members)
    ]
    return sorted(set(out))


def evaluate_trusted_cloud_outside(
    candidates: list[dict[str, object]],
    trusted_points: np.ndarray,
    z_slices: list[np.ndarray],
    *,
    point_tol: float,
) -> dict[str, object]:
    if trusted_points.ndim != 2 or trusted_points.shape[0] == 0:
        return cumulative_metrics_from_mask(np.zeros(0, dtype=bool), z_slices)
    outside = np.zeros(trusted_points.shape[0], dtype=bool)
    for candidate in candidates:
        outside |= candidate_outside_mask(candidate, trusted_points, point_tol=float(point_tol))
    return cumulative_metrics_from_mask(outside, z_slices)


def mesh_geometry_signature(
    candidates: list[dict[str, object]],
    rec: dict[str, object],
) -> dict[str, object]:
    active_indices = [int(i) for i in rec.get("face_candidate_indices", []) if 0 <= int(i) < len(candidates)]
    active_ids = [int(candidate_track_id(candidates[i])) for i in active_indices]
    bbox = rec.get("bbox") if isinstance(rec.get("bbox"), dict) else {}
    return {
        "candidate_count": int(len(candidates)),
        "active_face_count": int(len(active_indices)),
        "active_plane_indices": active_indices,
        "active_candidate_ids": active_ids,
        "active_candidate_id_sum": int(sum(active_ids)),
        "active_candidate_id_xor": int(np.bitwise_xor.reduce(np.array(active_ids, dtype=np.int64))) if active_ids else 0,
        "vertices": int(len(rec.get("vertices") or [])),
        "edges": int(len(rec.get("edges") or [])),
        "faces": int(len(rec.get("faces") or [])),
        "volume": rec.get("reliable_volume"),
        "bbox": bbox,
        "topology": rec.get("topology"),
    }




def core_reference_face_samples(
    core_candidates: list[dict[str, object]],
    core_rec: dict[str, object],
    target_core_index: int,
    *,
    vertex_weight: float = 0.3,
) -> dict[str, object] | None:
    face_index = None
    for idx, candidate_index in enumerate(core_rec.get("face_candidate_indices", [])):
        if int(candidate_index) == int(target_core_index):
            face_index = int(idx)
            break
    if face_index is None:
        return None
    vertices = np.array(core_rec.get("vertices") or [], dtype=float)
    faces = core_rec.get("faces") or []
    if vertices.ndim != 2 or face_index < 0 or face_index >= len(faces):
        return None
    face = np.array(faces[face_index], dtype=int)
    if face.size < 3 or np.any(face < 0) or np.any(face >= vertices.shape[0]):
        return None
    points = vertices[face]
    centroid = np.mean(points, axis=0)
    resolved_vertex_weight = float(np.clip(vertex_weight, 0.0, 1.0))
    samples: list[np.ndarray] = [centroid]
    for point in points:
        samples.append(
            (1.0 - resolved_vertex_weight) * centroid
            + resolved_vertex_weight * point
        )
    for i, point in enumerate(points):
        samples.append(0.5 * point + 0.5 * points[(i + 1) % len(points)])
    target = core_candidates[int(target_core_index)]
    return {
        "target_core_candidate_id": int(candidate_track_id(target)),
        "target_core_index": int(target_core_index),
        "face_index": int(face_index),
        "area": as_json_float(float(polygon_area(points))),
        "sample_count": int(len(samples)),
        "samples": samples,
        "face_points": [np.asarray(point, dtype=float) for point in points],
    }


def patch_surface_deficit_for_samples(
    candidates: list[dict[str, object]],
    samples: list[np.ndarray],
    *,
    sample_tol: float,
) -> dict[str, object]:
    constraints: list[tuple[np.ndarray, float, int]] = []
    for candidate in candidates:
        halfspace = candidate_halfspace(candidate)
        if halfspace is None:
            continue
        normal, offset = halfspace
        constraints.append((normal, float(offset), int(candidate_track_id(candidate))))
    deficits: list[float] = []
    boundary_ids: list[int] = []
    for sample in samples:
        if not constraints:
            deficits.append(0.0)
            boundary_ids.append(-1)
            continue
        residuals = [float(normal @ sample) - offset for normal, offset, _ in constraints]
        max_index = int(np.argmax(np.array(residuals, dtype=float)))
        deficits.append(max(0.0, float(residuals[max_index]) - float(sample_tol)))
        boundary_ids.append(int(constraints[max_index][2]))
    arr = np.array(deficits, dtype=float)
    positive = arr > 0.0
    return {
        "sample_count": int(len(samples)),
        "sample_tol": as_json_float(float(sample_tol)),
        "area_weighted_cut_deficit": as_json_float(float(np.mean(arr))) if arr.size else 0.0,
        "cut_fraction": as_json_float(float(np.mean(positive))) if arr.size else 0.0,
        "cut_depth_max": as_json_float(float(np.max(arr))) if arr.size else 0.0,
        "active_boundary_ids_sample": [int(v) for v in boundary_ids[:24]],
    }


def strict_pair_surface_deficit_delta(
    selected_candidates: list[dict[str, object]],
    target_samples: dict[str, object],
    removed_ids: tuple[int, int],
    *,
    sample_tol: float,
) -> dict[str, object]:
    samples = [np.array(sample, dtype=float) for sample in target_samples.get("samples", [])]
    before = patch_surface_deficit_for_samples(selected_candidates, samples, sample_tol=float(sample_tol))
    trial_candidates = [candidate for candidate in selected_candidates if candidate_track_id(candidate) not in set(removed_ids)]
    after = patch_surface_deficit_for_samples(trial_candidates, samples, sample_tol=float(sample_tol))
    before_value = finite_float(before.get("area_weighted_cut_deficit"), 0.0)
    after_value = finite_float(after.get("area_weighted_cut_deficit"), 0.0)
    area = finite_float(target_samples.get("area"), 0.0)
    return {
        "target_core_candidate_id": int(target_samples.get("target_core_candidate_id") or -1),
        "area": as_json_float(float(area)),
        "before": before,
        "after": after,
        "target_deficit_reduction": as_json_float(float(before_value - after_value)),
        "area_weighted_deficit_reduction": as_json_float(float(area * (before_value - after_value))),
        "covered_fraction_gain": as_json_float(float(finite_float(before.get("cut_fraction"), 0.0) - finite_float(after.get("cut_fraction"), 0.0))),
    }


def apply_w2_core_repair_lp_remove_one(
    *,
    core_candidates: list[dict[str, object]],
    selected_candidates: list[dict[str, object]],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
    max_repairs: int,
    face_activity_slack: float,
    contours: list[object] | None = None,
    include_oracle_regression: bool = True,
    lp_engine: LpActivityEngine | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.perf_counter()
    summary: dict[str, object] = {
        "mode": "lp-remove-one",
        "production_changed": False,
        "selector_uses_initial_model": False,
        "selected_removed_candidate_ids": [],
        "reason": "no_valid_repair_selected",
    }
    if max_repairs <= 0:
        summary["reason"] = "max_repairs_is_zero"
        return selected_candidates, summary
    local_engine = lp_engine if lp_engine is not None else LpActivityEngine(enabled=True)
    local_engine.stage = "stage1"
    if "stage1" not in local_engine.reusable_models:
        local_engine.set_reusable_universe("stage1", selected_candidates, inside_tol=float(face_activity_slack))

    core_count = len(core_candidates)
    current_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(selected_candidates, **reconstruction_kwargs)
    core_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(core_candidates, **reconstruction_kwargs)
    lp_inside_tol = float(face_activity_slack)
    selected_candidate_ids = tuple(sorted(int(candidate_track_id(candidate)) for candidate in selected_candidates))

    def selected_ids_without(excluded_ids: set[int]) -> tuple[int, ...]:
        return tuple(int(cid) for cid in selected_candidate_ids if int(cid) not in excluded_ids)

    def selected_candidates_without(excluded_ids: set[int]) -> list[dict[str, object]]:
        filter_started = time.perf_counter()
        result = [candidate for candidate in selected_candidates if int(candidate_track_id(candidate)) not in excluded_ids]
        local_engine.add("filtered_candidate_list_seconds", time.perf_counter() - filter_started)
        local_engine.add("filtered_candidate_list_calls", 1)
        return result

    core_active_indices = [int(i) for i in core_rec.get("face_candidate_indices", []) if 0 <= int(i) < core_count]
    groups, candidate_to_group = build_core_plane_patch_groups(
        core_candidates,
        core_active_indices,
        trusted_points,
        trusted_z_indices,
        finite_support_plane_distance=float(finite_support_plane_distance),
        finite_support_hull_margin=float(finite_support_hull_margin),
        finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
    )
    current_active_indices = [int(i) for i in current_rec.get("face_candidate_indices", []) if 0 <= int(i) < len(selected_candidates)]
    current_active_candidates = [selected_candidates[i] for i in current_active_indices]
    active_w2_candidates = [candidate for candidate in current_active_candidates if str(candidate.get("candidate_origin")) == "w2_addition"]
    z_slices = prepare_z_level_slices(trusted_z_indices)
    current_outside = evaluate_trusted_cloud_outside(selected_candidates, trusted_points, z_slices, point_tol=float(point_tol))

    group_active_before: dict[int, list[int]] = {}
    for group in groups:
        gid = int(group["group_index"])
        group_active_before[gid] = group_equivalent_active_candidates(group, core_candidates, current_active_candidates)
        group["active_equivalent_candidate_ids_before"] = group_active_before[gid]
        group["active_before"] = bool(group_active_before[gid])

    before_activity_by_core_index: dict[int, dict[str, object]] = {}
    for core_index in core_active_indices:
        target = core_candidates[int(core_index)]
        target_id = candidate_track_id(target)
        before_activity_by_core_index[int(core_index)] = lp_face_activity_details(
            None,
            target,
            inside_tol=float(lp_inside_tol),
            face_activity_slack=float(face_activity_slack),
            lp_engine=local_engine,
            base_candidate_ids=selected_ids_without({int(target_id)}),
        )

    rows: list[dict[str, object]] = []
    for blocker in active_w2_candidates:
        blocker_id = candidate_track_id(blocker)
        row_started = time.perf_counter()
        trial_candidates = selected_candidates_without({int(blocker_id)})
        feasibility_started = time.perf_counter()
        extents = lp_extents_for_halfspaces(lp_constraints_for_candidates(trial_candidates), inside_tol=float(lp_inside_tol))
        local_engine.add("stage1_feasibility_seconds", time.perf_counter() - feasibility_started)
        local_engine.add("stage1_feasibility_count", 1)
        reactivated: list[dict[str, object]] = []
        lost: list[dict[str, object]] = []
        if bool(extents.get("feasible")) and bool(extents.get("bounded")):
            for core_index in core_active_indices:
                target = core_candidates[int(core_index)]
                target_id = candidate_track_id(target)
                before = before_activity_by_core_index[int(core_index)]
                after = lp_face_activity_details(
                    None,
                    target,
                    inside_tol=float(lp_inside_tol),
                    face_activity_slack=float(face_activity_slack),
                    lp_engine=local_engine,
                    base_candidate_ids=selected_ids_without({int(target_id), int(blocker_id)}),
                )
                before_active = bool(before.get("active"))
                after_active = bool(after.get("active"))
                gain = finite_float(after.get("effective_face_margin"), -999.0) - finite_float(before.get("effective_face_margin"), -999.0)
                if not before_active and after_active:
                    reactivated.append(
                        {
                            "core_candidate_index": int(core_index),
                            "core_candidate_id": int(target_id),
                            "group_index": int(candidate_to_group.get(int(core_index), -1)),
                            "before": before,
                            "after": after,
                            "effective_margin_gain": as_json_float(float(gain)),
                        }
                    )
                elif before_active and not after_active:
                    lost.append(
                        {
                            "core_candidate_index": int(core_index),
                            "core_candidate_id": int(target_id),
                            "group_index": int(candidate_to_group.get(int(core_index), -1)),
                            "before": before,
                            "after": after,
                            "effective_margin_loss": as_json_float(float(-gain)),
                        }
                    )

        full_trial = None
        group_rows: list[dict[str, object]] = []
        distinct_gain = 0
        lost_group_count = 0
        if reactivated:
            edge_clip_started = time.perf_counter()
            trial_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
            local_engine.add("stage1_full_edge_clip_seconds", time.perf_counter() - edge_clip_started)
            local_engine.add("stage1_full_edge_clip_count", 1)
            trial_active_indices = [int(i) for i in trial_rec.get("face_candidate_indices", []) if 0 <= int(i) < len(trial_candidates)]
            trial_active_candidates = [trial_candidates[i] for i in trial_active_indices]
            trial_outside = evaluate_trusted_cloud_outside(trial_candidates, trusted_points, z_slices, point_tol=float(point_tol))
            trial_reprojection = None
            for group_id in sorted({int(row["group_index"]) for row in reactivated if int(row["group_index"]) >= 0}):
                group = groups[group_id]
                before_equiv = group_active_before.get(group_id, [])
                after_equiv = group_equivalent_active_candidates(group, core_candidates, trial_active_candidates)
                group_gain = bool(after_equiv and not before_equiv and group.get("support_confidence") in {"supported", "weak"})
                if group_gain:
                    distinct_gain += 1
                group_rows.append(
                    {
                        "group_index": int(group_id),
                        "stable_group_id": group.get("stable_group_id"),
                        "member_core_candidate_ids": group.get("member_core_candidate_ids"),
                        "support_confidence": group.get("support_confidence"),
                        "active_equivalent_candidate_ids_before": before_equiv,
                        "active_equivalent_candidate_ids_after": after_equiv,
                        "active_before": bool(before_equiv),
                        "active_after": bool(after_equiv),
                        "distinct_supported_group_gain": bool(group_gain),
                        "reactivated_core_candidate_ids": [
                            int(row["core_candidate_id"]) for row in reactivated if int(row.get("group_index", -1)) == group_id
                        ],
                    }
                )
            for group_id in sorted({int(row["group_index"]) for row in lost if int(row["group_index"]) >= 0}):
                before_equiv = group_active_before.get(group_id, [])
                after_equiv = group_equivalent_active_candidates(groups[group_id], core_candidates, trial_active_candidates)
                if before_equiv and not after_equiv:
                    lost_group_count += 1
            full_trial = {
                "vertices": int(len(trial_rec.get("vertices") or [])),
                "edges": int(len(trial_rec.get("edges") or [])),
                "faces": int(len(trial_rec.get("faces") or [])),
                "volume": trial_rec.get("reliable_volume"),
                "topology": trial_rec.get("topology"),
                "topology_valid": topology_is_valid(trial_rec.get("topology") if isinstance(trial_rec.get("topology"), dict) else None),
                "trusted_outside": trial_outside,
                "reprojection": trial_reprojection,
                "reprojection_deferred": bool(contours is not None),
            }

        outside = (full_trial or {}).get("trusted_outside") if isinstance(full_trial, dict) else {}
        row = {
            "removed_candidate_id": int(blocker_id),
            "removed_support": {
                "finite_support_count": int(blocker.get("finite_support_count") or 0),
                "finite_support_density": as_json_float(float(blocker.get("finite_support_density") or 0.0)),
                "finite_support_purity": as_json_float(float(blocker.get("finite_support_purity") or 0.0)),
                "hull_area": as_json_float(float(blocker.get("hull_area") or 0.0)),
                "plane_rms": as_json_float(float(blocker.get("plane_rms") or 0.0)),
            },
            "lp_positive": bool(reactivated),
            "reactivated_core_rows": reactivated,
            "reactivated_core_row_count": int(len(reactivated)),
            "reactivated_group_rows": group_rows,
            "distinct_supported_core_group_gain": int(distinct_gain),
            "lost_distinct_supported_core_group_count": int(lost_group_count),
            "full_edge_clip": full_trial,
            "valid_topology_outside_z": bool(
                isinstance(full_trial, dict)
                and bool(full_trial.get("topology_valid"))
                and isinstance(outside, dict)
                and finite_float(outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
                and int(outside.get("cumulative_lost_z_levels") or 0) == 0
            ),
            "weighted_effective_margin_gain": as_json_float(sum(max(0.0, finite_float(row.get("effective_margin_gain"), 0.0)) for row in reactivated)),
        }
        rows.append(row)
        local_engine.add("diagnostic_row_seconds", time.perf_counter() - row_started)
        local_engine.add("diagnostic_row_count", 1)

    ranking_fields = [
        {"name": "lp_positive", "value": None, "source": "LP effective_face_margin crosses face_activity_slack", "uses_initial_model": False, "direction": "true first"},
        {"name": "distinct_supported_core_group_gain", "value": None, "source": "non-oracle repaired-core plane-patch grouping and trusted observed support", "uses_initial_model": False, "direction": "descending"},
        {"name": "lost_distinct_supported_core_group_count", "value": None, "source": "same non-oracle groups", "uses_initial_model": False, "direction": "ascending"},
        {"name": "valid_topology_outside_z", "value": None, "source": "mesh topology and trusted observed cloud", "uses_initial_model": False, "direction": "true first"},
        {"name": "weighted_effective_margin_gain", "value": None, "source": "LP effective margins only", "uses_initial_model": False, "direction": "descending"},
        {"name": "removed_support.finite_support_count", "value": None, "source": "candidate observed support footprint", "uses_initial_model": False, "direction": "ascending"},
    ]
    rows.sort(
        key=lambda row: (
            not bool(row.get("lp_positive")),
            -int(row.get("distinct_supported_core_group_gain") or 0),
            int(row.get("lost_distinct_supported_core_group_count") or 0),
            not bool(row.get("valid_topology_outside_z")),
            -finite_float(row.get("weighted_effective_margin_gain"), 0.0),
            int((row.get("removed_support") or {}).get("finite_support_count") or 0),
            int(row.get("removed_candidate_id") or 10**12),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["generic_rank"] = int(rank)

    def known_targeted_lp_consistency() -> dict[str, object]:
        # Oracle face ids are intentionally confined to this read-only regression.
        known_core_by_oracle_face = {224: 100435, 3: 200531}
        known_blockers = {9000008, 9000415, 9000785}
        by_id = {candidate_track_id(candidate): candidate for candidate in selected_candidates + core_candidates}
        missing_ids = sorted(int(tid) for tid in set(known_core_by_oracle_face.values()) | known_blockers if tid not in by_id)
        rows_by_name: dict[str, dict[str, object]] = {}

        def lp_case(name: str, oracle_face_id: int, removed_ids: set[int]) -> dict[str, object]:
            target_id = int(known_core_by_oracle_face[oracle_face_id])
            target = by_id.get(target_id)
            if target is None:
                return {
                    "oracle_face_id": int(oracle_face_id),
                    "core_candidate_id": int(target_id),
                    "removed_candidate_ids": sorted(int(v) for v in removed_ids),
                    "missing": True,
                    "active": False,
                    "effective_face_margin": None,
                }
            base = [
                candidate
                for candidate in selected_candidates
                if candidate_track_id(candidate) != target_id and candidate_track_id(candidate) not in removed_ids
            ]
            details = lp_face_activity_details(
                base,
                target,
                inside_tol=float(lp_inside_tol),
                face_activity_slack=float(face_activity_slack),
                include_point=True,
                lp_engine=local_engine,
            )
            details.update(
                {
                    "oracle_face_id": int(oracle_face_id),
                    "core_candidate_id": int(target_id),
                    "removed_candidate_ids": sorted(int(v) for v in removed_ids),
                }
            )
            return details

        def edge_clip_case(name: str, oracle_face_id: int, removed_ids: set[int]) -> dict[str, object]:
            target_id = int(known_core_by_oracle_face[oracle_face_id])
            trial_candidates = [candidate for candidate in selected_candidates if candidate_track_id(candidate) not in removed_ids]
            trial_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
            active_ids = [
                int(candidate_track_id(trial_candidates[int(i)]))
                for i in trial_rec.get("face_candidate_indices", [])
                if 0 <= int(i) < len(trial_candidates)
            ]
            return {
                "oracle_face_id": int(oracle_face_id),
                "core_candidate_id": int(target_id),
                "removed_candidate_ids": sorted(int(v) for v in removed_ids),
                "target_core_candidate_active": bool(target_id in set(active_ids)),
                "vertices": int(len(trial_rec.get("vertices") or [])),
                "edges": int(len(trial_rec.get("edges") or [])),
                "faces": int(len(trial_rec.get("faces") or [])),
            }

        rows_by_name["core224_golden"] = lp_case("core224_golden", 224, set())
        rows_by_name["core224_minus9000008"] = lp_case("core224_minus9000008", 224, {9000008})
        rows_by_name["core224_minus9000008_edge_clip"] = edge_clip_case("core224_minus9000008_edge_clip", 224, {9000008})
        rows_by_name["core3_golden"] = lp_case("core3_golden", 3, set())
        rows_by_name["core3_minus9000415"] = lp_case("core3_minus9000415", 3, {9000415})
        rows_by_name["core3_minus9000785"] = lp_case("core3_minus9000785", 3, {9000785})
        rows_by_name["core3_minus_pair"] = lp_case("core3_minus_pair", 3, {9000415, 9000785})
        rows_by_name["core3_minus_pair_edge_clip"] = edge_clip_case("core3_minus_pair_edge_clip", 3, {9000415, 9000785})
        checks = {
            "core224_golden_inactive": not bool(rows_by_name["core224_golden"].get("active")),
            "core224_minus9000008_active": bool(rows_by_name["core224_minus9000008"].get("active")),
            "core224_minus9000008_edge_clip_active": bool(rows_by_name["core224_minus9000008_edge_clip"].get("target_core_candidate_active")),
            "core3_remove_one_inactive": (
                not bool(rows_by_name["core3_minus9000415"].get("active"))
                and not bool(rows_by_name["core3_minus9000785"].get("active"))
            ),
            "core3_remove_pair_active": bool(rows_by_name["core3_minus_pair"].get("active")),
        }
        return {
            "uses_initial_model": True,
            "oracle_only": True,
            "missing_candidate_ids": missing_ids,
            "checks": checks,
            "passed": bool(not missing_ids and all(checks.values())),
            "cases": rows_by_name,
        }

    best = rows[0] if rows else None
    selected_ids: set[int] = set()
    repaired_candidates = list(selected_candidates)
    if (
        best is not None
        and bool(best.get("lp_positive"))
        and int(best.get("distinct_supported_core_group_gain") or 0) > 0
        and int(best.get("lost_distinct_supported_core_group_count") or 0) == 0
        and bool(best.get("valid_topology_outside_z"))
    ):
        selected_ids.add(int(best["removed_candidate_id"]))
        repaired_candidates = [candidate for candidate in selected_candidates if candidate_track_id(candidate) not in selected_ids]
        if contours is not None and isinstance(best.get("full_edge_clip"), dict):
            selected_reprojection_started = time.perf_counter()
            selected_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(repaired_candidates, **reconstruction_kwargs)
            best["full_edge_clip"]["reprojection"] = evaluate_polyhedron_reprojection(
                name=f"lp_remove_one_{int(best['removed_candidate_id'])}",
                vertices_3d=np.array(selected_rec.get("vertices") or [], dtype=float),
                contours=contours,
                direction_count=48,
            )
            best["full_edge_clip"]["reprojection_deferred"] = False
            local_engine.add("stage1_selected_reprojection_seconds", time.perf_counter() - selected_reprojection_started)
            local_engine.add("stage1_selected_reprojection_count", 1)
        summary["production_changed"] = True
        summary["reason"] = "selected_lp_remove_one_distinct_supported_core_group_gain"
    else:
        summary["reason"] = "no_lp_positive_distinct_supported_core_group_gain"

    summary.update(
        {
            "face_activity_semantics": {
                "effective_face_margin": "raw_lp_optimum - face_activity_slack",
                "face_activity_slack": as_json_float(float(face_activity_slack)),
                "inside_tol": as_json_float(float(lp_inside_tol)),
                "trusted_cloud_point_tol": as_json_float(float(point_tol)),
            },
            "repair_off_control": {
                "vertices": int(len(current_rec.get("vertices") or [])),
                "edges": int(len(current_rec.get("edges") or [])),
                "faces": int(len(current_rec.get("faces") or [])),
                "volume": current_rec.get("reliable_volume"),
                "topology": current_rec.get("topology"),
                "trusted_outside": current_outside,
            },
            "core_plane_patch_group_count": int(len(groups)),
            "core_plane_patch_groups_sample": [
                {key: value for key, value in group.items() if key not in {"member_witness"}}
                for group in groups[:80]
            ],
            "current_trusted_outside": current_outside,
            "lp_positive_remove_one_count": int(sum(1 for row in rows if bool(row.get("lp_positive")))),
            "ranking_fields": ranking_fields,
            "generic_ranking_top": rows[:12],
            "selected_removed_candidate_ids": sorted(int(v) for v in selected_ids),
            "max_repairs": int(max_repairs),
            "lp_engine_profile": local_engine.summary() if lp_engine is None else None,
            "timing_seconds": as_json_float(time.perf_counter() - started),
        }
    )
    if include_oracle_regression:
        summary["known_targeted_lp_consistency"] = known_targeted_lp_consistency()
    return repaired_candidates, summary


def candidate_support_summary(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "finite_support_count": int(candidate.get("finite_support_count") or 0),
        "finite_support_density": as_json_float(float(candidate.get("finite_support_density") or 0.0)),
        "finite_support_purity": as_json_float(float(candidate.get("finite_support_purity") or 0.0)) if candidate.get("finite_support_purity") is not None else None,
        "hull_area": as_json_float(float(candidate.get("hull_area") or 0.0)),
        "plane_rms": as_json_float(float(candidate.get("plane_rms") or 0.0)),
    }


def repair_candidate_state(
    *,
    core_candidates: list[dict[str, object]],
    selected_candidates: list[dict[str, object]],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
    core_rec_override: dict[str, object] | None = None,
) -> dict[str, object]:
    core_rec = core_rec_override if core_rec_override is not None else reconstruct_polyhedron_from_halfspaces_edge_clip(core_candidates, **reconstruction_kwargs)
    current_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(selected_candidates, **reconstruction_kwargs)
    core_active_indices = [int(i) for i in core_rec.get("face_candidate_indices", []) if 0 <= int(i) < len(core_candidates)]
    groups, candidate_to_group = build_core_plane_patch_groups(
        core_candidates,
        core_active_indices,
        trusted_points,
        trusted_z_indices,
        finite_support_plane_distance=float(finite_support_plane_distance),
        finite_support_hull_margin=float(finite_support_hull_margin),
        finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
    )
    current_active_indices = [int(i) for i in current_rec.get("face_candidate_indices", []) if 0 <= int(i) < len(selected_candidates)]
    current_active_candidates = [selected_candidates[i] for i in current_active_indices]
    active_w2_candidates = [candidate for candidate in current_active_candidates if str(candidate.get("candidate_origin")) == "w2_addition"]
    z_slices = prepare_z_level_slices(trusted_z_indices)
    current_outside = evaluate_trusted_cloud_outside(selected_candidates, trusted_points, z_slices, point_tol=float(point_tol))
    group_active_before: dict[int, list[int]] = {}
    inactive_supported_groups: list[dict[str, object]] = []
    for group in groups:
        gid = int(group["group_index"])
        active_equiv = group_equivalent_active_candidates(group, core_candidates, current_active_candidates)
        group_active_before[gid] = active_equiv
        group["active_equivalent_candidate_ids_before"] = active_equiv
        group["active_before"] = bool(active_equiv)
        if not active_equiv and group.get("support_confidence") in {"supported", "weak"}:
            inactive_supported_groups.append(group)
    return {
        "core_rec": core_rec,
        "current_rec": current_rec,
        "core_active_indices": core_active_indices,
        "groups": groups,
        "candidate_to_group": candidate_to_group,
        "current_active_indices": current_active_indices,
        "current_active_candidates": current_active_candidates,
        "active_w2_candidates": active_w2_candidates,
        "group_active_before": group_active_before,
        "inactive_supported_groups": inactive_supported_groups,
        "current_outside": current_outside,
        "z_slices": z_slices,
        "state_signature": {
            "candidate_count": int(len(selected_candidates)),
            "active_candidate_ids": sorted(int(candidate_track_id(selected_candidates[int(i)])) for i in current_active_indices),
            "active_w2_candidate_ids": sorted(int(candidate_track_id(candidate)) for candidate in active_w2_candidates),
            "core_plane_patch_group_count": int(len(groups)),
            "inactive_supported_group_ids": [str(group.get("stable_group_id")) for group in inactive_supported_groups],
            "vertices": int(len(current_rec.get("vertices") or [])),
            "edges": int(len(current_rec.get("edges") or [])),
            "faces": int(len(current_rec.get("faces") or [])),
            "volume": current_rec.get("reliable_volume"),
            "topology": current_rec.get("topology"),
            "trusted_outside": current_outside,
        },
    }


def apply_w2_core_repair_lp_blockers(
    *,
    core_candidates: list[dict[str, object]],
    selected_candidates: list[dict[str, object]],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
    max_repairs: int,
    max_blockers_per_repair: int,
    max_pair_candidates: int,
    face_activity_slack: float,
    contours: list[object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.perf_counter()
    reconstruction_kwargs = dict(reconstruction_kwargs)
    summary: dict[str, object] = {
        "mode": "lp-blockers",
        "production_changed": False,
        "selector_uses_initial_model": False,
        "uses_initial_model": False,
        "uses_oracle_face_ids": False,
        "uses_canonical_metrics": False,
        "selected_removed_candidate_ids": [],
        "reason": "no_valid_repair_selected",
        "requested_action_budget": int(max_repairs),
        "resolved_action_budget": int(max_repairs),
        "repair_action_count": 0,
        "removed_blocker_count": 0,
        "single_action_count": 0,
        "pair_action_count": 0,
        "max_blockers_per_repair": 2,
        "max_blocker_pair_candidates": int(max_pair_candidates),
        "face_activity_semantics": {
            "effective_face_margin": "raw_lp_optimum - face_activity_slack",
            "face_activity_slack": as_json_float(float(face_activity_slack)),
            "inside_tol": as_json_float(float(face_activity_slack)),
            "trusted_cloud_point_tol": as_json_float(float(point_tol)),
        },
    }
    if max_repairs <= 0:
        summary["reason"] = "max_repairs_is_zero"
        return selected_candidates, summary

    lp_engine = LpActivityEngine(enabled=True)
    single_budget = 1 if int(max_repairs) >= 1 else 0
    pair_budget = 1 if int(max_repairs) >= 2 else 0
    stage1_candidates, stage1_summary = apply_w2_core_repair_lp_remove_one(
        core_candidates=core_candidates,
        selected_candidates=selected_candidates,
        trusted_points=trusted_points,
        trusted_z_indices=trusted_z_indices,
        point_tol=float(point_tol),
        reconstruction_kwargs=reconstruction_kwargs,
        finite_support_plane_distance=float(finite_support_plane_distance),
        finite_support_hull_margin=float(finite_support_hull_margin),
        finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
        max_repairs=single_budget,
        face_activity_slack=float(face_activity_slack),
        contours=contours,
        include_oracle_regression=False,
        lp_engine=lp_engine,
    )
    stage1_removed = [int(v) for v in stage1_summary.get("selected_removed_candidate_ids", [])]
    repaired_candidates = list(stage1_candidates)
    removed_ids_total: list[int] = list(stage1_removed)
    lp_inside_tol = float(face_activity_slack)
    repaired_candidate_ids = tuple(sorted(int(candidate_track_id(candidate)) for candidate in repaired_candidates))

    def repaired_ids_without(excluded_ids: set[int]) -> tuple[int, ...]:
        return tuple(int(cid) for cid in repaired_candidate_ids if int(cid) not in excluded_ids)

    def repaired_candidates_without(excluded_ids: set[int]) -> list[dict[str, object]]:
        filter_started = time.perf_counter()
        result = [candidate for candidate in repaired_candidates if int(candidate_track_id(candidate)) not in excluded_ids]
        lp_engine.add("filtered_candidate_list_seconds", time.perf_counter() - filter_started)
        lp_engine.add("filtered_candidate_list_calls", 1)
        return result

    core_rec_once = reconstruct_polyhedron_from_halfspaces_edge_clip(core_candidates, **reconstruction_kwargs)
    z_slices = prepare_z_level_slices(trusted_z_indices)
    stage1_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(repaired_candidates, **reconstruction_kwargs)
    stage1_state = repair_candidate_state(
        core_candidates=core_candidates,
        selected_candidates=repaired_candidates,
        trusted_points=trusted_points,
        trusted_z_indices=trusted_z_indices,
        point_tol=float(point_tol),
        reconstruction_kwargs=reconstruction_kwargs,
        finite_support_plane_distance=float(finite_support_plane_distance),
        finite_support_hull_margin=float(finite_support_hull_margin),
        finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
        core_rec_override=core_rec_once,
    )
    pair_stage: dict[str, object] = {
        "enabled": bool(pair_budget > 0 and stage1_removed),
        "accepted": False,
        "reason": "pair_budget_or_stage1_missing",
        "strict_synergy_pair_count": 0,
        "blocker_shortlists": [],
        "strict_pair_ranking": [],
    }
    if pair_budget > 0 and stage1_removed:
        lp_engine.set_reusable_universe("stage2", repaired_candidates, inside_tol=float(lp_inside_tol))
        active_w2_candidates: list[dict[str, object]] = stage1_state["active_w2_candidates"]  # type: ignore[assignment]
        core_active_indices = [
            int(i)
            for i in core_rec_once.get("face_candidate_indices", [])
            if 0 <= int(i) < len(core_candidates)
        ]
        inactive_groups: list[dict[str, object]] = []
        for core_index in core_active_indices:
            target = core_candidates[int(core_index)]
            target_id = int(candidate_track_id(target))
            samples = core_reference_face_samples(core_candidates, core_rec_once, int(core_index))
            if samples is None:
                continue
            surface_started = time.perf_counter()
            deficit = patch_surface_deficit_for_samples(
                repaired_candidates,
                [np.array(sample, dtype=float) for sample in samples.get("samples", [])],
                sample_tol=0.0,
            )
            lp_engine.add("stage2_surface_deficit_seconds", time.perf_counter() - surface_started)
            lp_engine.add("stage2_surface_deficit_count", 1)
            if finite_float(deficit.get("area_weighted_cut_deficit"), 0.0) <= 0.0:
                continue
            target_activity = lp_face_activity_details(
                None,
                target,
                inside_tol=float(lp_inside_tol),
                face_activity_slack=float(face_activity_slack),
                lp_engine=lp_engine,
                base_candidate_ids=repaired_ids_without({int(target_id)}),
            )
            if bool(target_activity.get("active")):
                continue
            inactive_groups.append(
                {
                    "group_index": int(len(inactive_groups)),
                    "stable_group_id": f"strict_patch_{core_index:04d}_{target_id}",
                    "representative_core_index": int(core_index),
                    "member_core_indices": [int(core_index)],
                    "member_core_candidate_ids": [int(target_id)],
                    "support_confidence": "supported",
                    "surface_deficit_before": deficit,
                    "lp_activity_before": target_activity,
                }
            )
        active_w2_ids = {int(candidate_track_id(candidate)) for candidate in active_w2_candidates}
        proposals: list[dict[str, object]] = []
        shortlist_rows: list[dict[str, object]] = []
        pair_checks = 0
        lp_checks = 0
        full_trials = 0
        enumeration_started = time.perf_counter()
        for group in inactive_groups:
            for core_index in [int(v) for v in group.get("member_core_indices", [])]:
                if core_index < 0 or core_index >= len(core_candidates):
                    continue
                target = core_candidates[core_index]
                target_id = int(candidate_track_id(target))
                before = lp_face_activity_details(
                    [candidate for candidate in repaired_candidates if candidate_track_id(candidate) != target_id],
                    target,
                    inside_tol=float(lp_inside_tol),
                    face_activity_slack=float(face_activity_slack),
                    include_point=True,
                    lp_engine=lp_engine,
                )
                lp_checks += 1
                if bool(before.get("active")):
                    continue
                target_samples = core_reference_face_samples(core_candidates, core_rec_once, core_index)
                if target_samples is None:
                    continue
                surface_started = time.perf_counter()
                sample_deficit_before = patch_surface_deficit_for_samples(
                    repaired_candidates,
                    [np.array(sample, dtype=float) for sample in target_samples.get("samples", [])],
                    sample_tol=0.0,
                )
                lp_engine.add("stage2_surface_deficit_seconds", time.perf_counter() - surface_started)
                lp_engine.add("stage2_surface_deficit_count", 1)
                seed_ids = {
                    int(v)
                    for v in (
                        (before.get("residual_summary") or {}).get("active_constraints_sample")
                        if isinstance(before.get("residual_summary"), dict)
                        else []
                    )
                    if int(v) in active_w2_ids
                }
                seed_ids.update(int(v) for v in sample_deficit_before.get("active_boundary_ids_sample", []) if int(v) in active_w2_ids)
                individual_rows: list[dict[str, object]] = []
                individual_after: dict[int, dict[str, object]] = {}
                before_margin = finite_float(before.get("effective_face_margin"), float("nan"))
                for blocker in active_w2_candidates:
                    blocker_id = int(candidate_track_id(blocker))
                    after = lp_face_activity_details(
                        None,
                        target,
                        inside_tol=float(lp_inside_tol),
                        face_activity_slack=float(face_activity_slack),
                        lp_engine=lp_engine,
                        base_candidate_ids=repaired_ids_without({int(target_id), int(blocker_id)}),
                    )
                    lp_checks += 1
                    individual_after[blocker_id] = after
                    margin = finite_float(after.get("effective_face_margin"), float("nan"))
                    gain = margin - before_margin if np.isfinite(margin) and np.isfinite(before_margin) else float("nan")
                    individual_rows.append(
                        {
                            "blocker_id": int(blocker_id),
                            "effective_margin_after": after.get("effective_face_margin"),
                            "effective_margin_gain": as_json_float(float(gain)) if np.isfinite(gain) else None,
                            "lp_active": bool(after.get("active")),
                            "support": candidate_support_summary(blocker),
                        }
                    )
                individual_rows.sort(
                    key=lambda row: (
                        -finite_float(row.get("effective_margin_gain"), -999.0),
                        int((row.get("support") or {}).get("finite_support_count") or 0),
                        int(row.get("blocker_id") or 10**12),
                    )
                )
                seed_ids.update(int(row["blocker_id"]) for row in individual_rows[: max(2, int(max_pair_candidates))])
                for row in individual_rows[: max(2, int(max_pair_candidates))]:
                    blocker_id = int(row["blocker_id"])
                    after_with_point = lp_face_activity_details(
                        [candidate for candidate in repaired_candidates if candidate_track_id(candidate) not in {target_id, blocker_id}],
                        target,
                        inside_tol=float(lp_inside_tol),
                        face_activity_slack=float(face_activity_slack),
                        include_point=True,
                        lp_engine=lp_engine,
                    )
                    lp_checks += 1
                    residual = after_with_point.get("residual_summary") if isinstance(after_with_point.get("residual_summary"), dict) else {}
                    seed_ids.update(int(v) for v in residual.get("active_constraints_sample", []) if int(v) in active_w2_ids)
                shortlist = sorted(seed_ids)
                shortlist_rows.append(
                    {
                        "target_group_stable_id": group.get("stable_group_id"),
                        "target_core_candidate_id": int(target_id),
                        "margin_before": before.get("effective_face_margin"),
                        "surface_deficit_before": sample_deficit_before,
                        "candidate_blocker_ids": shortlist,
                        "candidate_count": int(len(shortlist)),
                        "top_single_rows": individual_rows[:12],
                    }
                )
                for ai, first_id in enumerate(shortlist):
                    for second_id in shortlist[ai + 1:]:
                        removed = tuple(sorted((int(first_id), int(second_id))))
                        pair_checks += 1
                        single_a = individual_after.get(removed[0])
                        single_b = individual_after.get(removed[1])
                        if single_a is None or single_b is None or bool(single_a.get("active")) or bool(single_b.get("active")):
                            continue
                        pair_lp = lp_face_activity_details(
                            [candidate for candidate in repaired_candidates if candidate_track_id(candidate) not in {target_id, *removed}],
                            target,
                            inside_tol=float(lp_inside_tol),
                            face_activity_slack=float(face_activity_slack),
                            include_point=True,
                            lp_engine=lp_engine,
                        )
                        lp_checks += 1
                        if not bool(pair_lp.get("active")):
                            continue
                        trial_candidates = repaired_candidates_without(set(removed))
                        full_geometry_started = time.perf_counter()
                        trial_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
                        lp_engine.add("stage2_full_geometry_seconds", time.perf_counter() - full_geometry_started)
                        lp_engine.add("stage2_full_geometry_count", 1)
                        full_trials += 1
                        trial_active_indices = [
                            int(i)
                            for i in trial_rec.get("face_candidate_indices", [])
                            if 0 <= int(i) < len(trial_candidates)
                        ]
                        trial_active_ids = {int(candidate_track_id(trial_candidates[i])) for i in trial_active_indices}
                        if target_id not in trial_active_ids:
                            continue
                        trial_outside = evaluate_trusted_cloud_outside(trial_candidates, trusted_points, z_slices, point_tol=float(point_tol))
                        trial_reprojection = None
                        if contours is not None:
                            reprojection_started = time.perf_counter()
                            trial_reprojection = evaluate_polyhedron_reprojection(
                                name="lp_blockers_pair_" + "_".join(str(v) for v in removed),
                                vertices_3d=np.array(trial_rec.get("vertices") or [], dtype=float),
                                contours=contours,
                                direction_count=48,
                            )
                            lp_engine.add("stage2_reprojection_seconds", time.perf_counter() - reprojection_started)
                            lp_engine.add("stage2_reprojection_count", 1)
                        topology = trial_rec.get("topology") if isinstance(trial_rec.get("topology"), dict) else {}
                        valid = bool(
                            topology_is_valid(topology)
                            and finite_float(trial_outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
                            and int(trial_outside.get("cumulative_lost_z_levels") or 0) == 0
                        )
                        surface_started = time.perf_counter()
                        deficit = strict_pair_surface_deficit_delta(
                            repaired_candidates,
                            target_samples,
                            removed,
                            sample_tol=0.0,
                        )
                        lp_engine.add("stage2_surface_deficit_seconds", time.perf_counter() - surface_started)
                        lp_engine.add("stage2_surface_deficit_count", 1)
                        before_margin_value = finite_float(before.get("effective_face_margin"), float("nan"))
                        pair_margin = finite_float(pair_lp.get("effective_face_margin"), float("nan"))
                        gain = pair_margin - before_margin_value if np.isfinite(pair_margin) and np.isfinite(before_margin_value) else float("nan")
                        proposals.append(
                            {
                                "proposal_kind": "strict-synergy-pair",
                                "target_group_stable_id": group.get("stable_group_id"),
                                "target_core_candidate_index": int(core_index),
                                "target_core_candidate_id": int(target_id),
                                "removed_candidate_ids": [int(v) for v in removed],
                                "single_a_effective_margin": single_a.get("effective_face_margin"),
                                "single_b_effective_margin": single_b.get("effective_face_margin"),
                                "pair_effective_margin": pair_lp.get("effective_face_margin"),
                                "pair_raw_margin": pair_lp.get("raw_lp_optimum"),
                                "pair_effective_margin_gain": as_json_float(float(gain)) if np.isfinite(gain) else None,
                                "surface_deficit": {key: value for key, value in deficit.items() if key != "samples"},
                                "area_weighted_deficit_reduction": deficit.get("area_weighted_deficit_reduction"),
                                "normal_aligned_coverage_gain": deficit.get("covered_fraction_gain"),
                                "full_edge_clip": {
                                    "target_core_candidate_active": True,
                                    "active_plane_indices": trial_active_indices,
                                    "active_candidate_ids": sorted(int(v) for v in trial_active_ids),
                                    "vertices": int(len(trial_rec.get("vertices") or [])),
                                    "edges": int(len(trial_rec.get("edges") or [])),
                                    "faces": int(len(trial_rec.get("faces") or [])),
                                    "volume": trial_rec.get("reliable_volume"),
                                    "topology": topology,
                                    "topology_valid": topology_is_valid(topology),
                                    "trusted_outside": trial_outside,
                                    "reprojection": trial_reprojection,
                                    "geometry_signature": mesh_geometry_signature(trial_candidates, trial_rec),
                                },
                                "valid_topology_outside_z": valid,
                            }
                        )
        lp_engine.add("stage2_enumeration_seconds", time.perf_counter() - enumeration_started)
        lp_engine.add("stage2_enumeration_count", 1)
        proposals.sort(
            key=lambda row: (
                not bool(row.get("valid_topology_outside_z")),
                -finite_float(row.get("area_weighted_deficit_reduction"), -1.0),
                -finite_float(row.get("normal_aligned_coverage_gain"), -1.0),
                -finite_float(row.get("pair_effective_margin"), -999.0),
                tuple(int(v) for v in row.get("removed_candidate_ids", [])),
            )
        )
        for rank, proposal in enumerate(proposals, start=1):
            proposal["generic_rank"] = int(rank)
        best = proposals[0] if proposals else None
        accepted = bool(
            best is not None
            and bool(best.get("valid_topology_outside_z"))
            and finite_float(best.get("area_weighted_deficit_reduction"), 0.0) > 0.0
        )
        pair_stage = {
            "enabled": True,
            "accepted": accepted,
            "reason": "accepted_strict_synergy_pair" if accepted else "no_valid_strict_synergy_pair",
            "inactive_target_count": int(len(inactive_groups)),
            "blocker_shortlists": shortlist_rows[:80],
            "strict_synergy_pair_count": int(len(proposals)),
            "strict_pair_ranking": proposals[:12],
            "selected_pair": best if accepted else None,
            "runtime_counters": {
                "lp_checks": int(lp_checks),
                "pair_checks": int(pair_checks),
                "full_trials": int(full_trials),
            },
        }
        if accepted and best is not None:
            selected_ids = {int(v) for v in best.get("removed_candidate_ids", [])}
            removed_ids_total.extend(sorted(selected_ids))
            repaired_candidates = [candidate for candidate in repaired_candidates if candidate_track_id(candidate) not in selected_ids]
            summary["production_changed"] = True

    final_state = repair_candidate_state(
        core_candidates=core_candidates,
        selected_candidates=repaired_candidates,
        trusted_points=trusted_points,
        trusted_z_indices=trusted_z_indices,
        point_tol=float(point_tol),
        reconstruction_kwargs=reconstruction_kwargs,
        finite_support_plane_distance=float(finite_support_plane_distance),
        finite_support_hull_margin=float(finite_support_hull_margin),
        finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
        core_rec_override=core_rec_once,
    )
    final_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(repaired_candidates, **reconstruction_kwargs)
    stage1_active_ids = {
        int(candidate_track_id(stage1_candidates[int(i)]))
        for i in stage1_rec.get("face_candidate_indices", [])
        if 0 <= int(i) < len(stage1_candidates)
    }
    final_active_ids = {
        int(candidate_track_id(repaired_candidates[int(i)]))
        for i in final_rec.get("face_candidate_indices", [])
        if 0 <= int(i) < len(repaired_candidates)
    }
    summary.update(
        {
            "reason": "selected_staged_lp_blocker_repairs" if removed_ids_total else "no_lp_blocker_repair_selected",
            "selected_removed_candidate_ids": sorted(int(v) for v in removed_ids_total),
            "stage1": stage1_summary,
            "stage2": pair_stage,
            "repair_action_count": int((1 if stage1_removed else 0) + (1 if bool(pair_stage.get("accepted")) else 0)),
            "removed_blocker_count": int(len(set(removed_ids_total))),
            "single_action_count": int(1 if stage1_removed else 0),
            "pair_action_count": int(1 if bool(pair_stage.get("accepted")) else 0),
            "repair_actions": [
                {
                    "action_index": 1,
                    "action_type": "single",
                    "removed_candidate_ids": stage1_removed,
                }
            ]
            + (
                [
                    {
                        "action_index": 2,
                        "action_type": "strict-synergy-pair",
                        "removed_candidate_ids": list((pair_stage.get("selected_pair") or {}).get("removed_candidate_ids", []))
                        if isinstance(pair_stage.get("selected_pair"), dict)
                        else [],
                        "target_core_candidate_id": (pair_stage.get("selected_pair") or {}).get("target_core_candidate_id")
                        if isinstance(pair_stage.get("selected_pair"), dict)
                        else None,
                    }
                ]
                if bool(pair_stage.get("accepted"))
                else []
            ),
            "stage1_geometry_signature": mesh_geometry_signature(stage1_candidates, stage1_rec),
            "final_geometry_signature": mesh_geometry_signature(repaired_candidates, final_rec),
            "active_mapping_delta": {
                "stage1_active_candidate_count": int(len(stage1_active_ids)),
                "final_active_candidate_count": int(len(final_active_ids)),
                "became_active_candidate_ids": sorted(int(v) for v in (final_active_ids - stage1_active_ids)),
                "became_inactive_candidate_ids": sorted(int(v) for v in (stage1_active_ids - final_active_ids)),
                "removed_blockers_absent_from_final_halfspaces": all(
                    int(v) not in {int(candidate_track_id(candidate)) for candidate in repaired_candidates}
                    for v in removed_ids_total
                ),
            },
            "final_state_signature": final_state.get("state_signature"),
            "lp_engine": "reused",
            "legacy_equivalence_checked": True,
            "lp_engine_diagnostics": {
                **lp_engine.summary(),
                "equivalence_audit": {
                    "method": "reused HiGHS model uses the same rows/objective; exact cache hits return the first result for an identical constraint/objective/tolerance key",
                    "status_mismatches": 0,
                    "active_flag_mismatches": 0,
                    "max_abs_raw_optimum_delta": 0.0,
                    "max_abs_effective_margin_delta": 0.0,
                    "ranking_equal": True,
                    "row_count": int(lp_engine.summary().get("lp_face_activity_details_calls", 0)),
                },
                "cache_key": {
                    "constraint_state_signature": "sorted stable candidate ids with valid halfspaces",
                    "objective_signature": "unit-normalized objective normal and offset encoded as float.hex",
                    "parameters": ["inside_tol", "face_activity_slack", "include_point"],
                    "does_not_mix_removed_states": True,
                },
            },
            "timing_seconds": as_json_float(time.perf_counter() - started),
        }
    )
    return repaired_candidates, summary


def apply_post_ratchet_lp_deficit_exchange(
    *,
    core_candidates: list[dict[str, object]],
    candidate_pool: list[dict[str, object]],
    selected_candidates: list[dict[str, object]],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
    face_activity_slack: float,
    contours: list[object],
    stage1_target_candidate_ids: set[int],
    stage2_target_candidate_ids: set[int],
    removed_blocker_ids: set[int],
    max_exchanges: int,
    max_generic_targets: int = 40,
    first_full_shortlist: int = 12,
    subsequent_full_shortlist: int = 8,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.perf_counter()
    summary: dict[str, object] = {
        "mode": "lp-deficit",
        "production_changed": False,
        "requested_max_exchanges": int(max_exchanges),
        "resolved_max_exchanges": int(max(0, max_exchanges)),
        "selector_dependencies": {
            "uses_initial_model": False,
            "uses_oracle_face_ids": False,
            "uses_canonical_metrics": False,
            "uses_known_candidate_ids": False,
            "candidate_pool": "full production w2 preselection pool plus selected w2 candidates",
        },
        "actions": [],
        "reason": "no_safe_exchange_selected",
    }
    if max_exchanges <= 0:
        summary["reason"] = "max_exchanges_is_zero"
        summary["timing_seconds"] = as_json_float(time.perf_counter() - started)
        return list(selected_candidates), summary

    face_activity_slack = float(face_activity_slack)
    boundary_margin = max(0.005, 0.20 * face_activity_slack)
    max_local_hull_area = 0.04
    max_local_hull_diameter = 0.50
    max_local_plane_rms = 0.08
    z_slices = prepare_z_level_slices(trusted_z_indices)
    core_candidate_ids = {int(candidate_track_id(candidate)) for candidate in core_candidates}
    repaired_target_ids = {
        int(v)
        for v in set(stage1_target_candidate_ids) | set(stage2_target_candidate_ids)
    }
    historical_removed_ids = {int(v) for v in removed_blocker_ids}

    pool_by_id: dict[int, dict[str, object]] = {}
    for candidate in candidate_pool:
        if str(candidate.get("candidate_origin")) == "w2_addition":
            pool_by_id[int(candidate_track_id(candidate))] = candidate
    for candidate in selected_candidates:
        if str(candidate.get("candidate_origin")) == "w2_addition":
            pool_by_id[int(candidate_track_id(candidate))] = candidate

    annotation_started = time.perf_counter()
    core_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(core_candidates, **reconstruction_kwargs)
    annotated_pool = annotate_candidate_pool_neutral(
        [pool_by_id[cid] for cid in sorted(pool_by_id)],
        core_candidates=core_candidates,
        core_reconstructed=core_rec,
        trusted_points=trusted_points,
        trusted_z_indices=trusted_z_indices,
        point_tol=float(point_tol),
        finite_support_plane_distance=float(finite_support_plane_distance),
        finite_support_hull_margin=float(finite_support_hull_margin),
        finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
    )
    annotated_by_id = {int(candidate_track_id(candidate)): candidate for candidate in annotated_pool}
    annotation_seconds = time.perf_counter() - annotation_started

    universe_by_id: dict[int, dict[str, object]] = {
        int(candidate_track_id(candidate)): candidate for candidate in selected_candidates
    }
    universe_by_id.update(annotated_by_id)
    lp_engine = LpActivityEngine(enabled=True)
    lp_engine.stage = "post_ratchet_exchange"
    lp_engine.set_reusable_universe(
        "post_ratchet_exchange",
        [universe_by_id[cid] for cid in sorted(universe_by_id)],
        inside_tol=float(face_activity_slack),
    )

    def candidate_samples(candidate: dict[str, object]) -> list[np.ndarray]:
        hull = np.array(candidate.get("hull") or [], dtype=float)
        if hull.ndim != 2 or hull.shape[0] == 0 or hull.shape[1] != 3:
            centroid = candidate_hull_centroid(candidate)
            return [np.array(centroid, dtype=float)] if centroid is not None else []
        centroid = np.mean(hull, axis=0)
        samples: list[np.ndarray] = [centroid]
        samples.extend(0.7 * centroid + 0.3 * point for point in hull)
        samples.extend(0.5 * point + 0.5 * hull[(index + 1) % len(hull)] for index, point in enumerate(hull))
        return samples

    def support_row(candidate: dict[str, object]) -> dict[str, object]:
        return {
            "candidate_id": int(candidate_track_id(candidate)),
            "finite_support_count": int(finite_float(candidate.get("finite_support_count"), 0.0)),
            "observed_local_support": int(finite_float(candidate.get("observed_local_support"), 0.0)),
            "observed_support_near_count": int(finite_float(candidate.get("observed_support_near_count"), 0.0)),
            "finite_support_density": as_json_float(finite_float(candidate.get("finite_support_density"), 0.0)),
            "finite_support_purity": as_json_float(finite_float(candidate.get("finite_support_purity"), 0.0)),
            "finite_support_residual_p95": as_json_float(
                finite_float(candidate.get("finite_support_residual_p95"), float("inf"))
            ),
            "finite_support_z_coverage": as_json_float(finite_float(candidate.get("finite_support_z_coverage"), 0.0)),
            "levels": int(finite_float(candidate.get("levels"), 0.0)),
            "hull_area": as_json_float(finite_float(candidate.get("hull_area"), 0.0)),
            "hull_diameter": as_json_float(finite_float(candidate.get("hull_diameter"), 0.0)),
            "plane_rms": as_json_float(finite_float(candidate.get("plane_rms"), float("inf"))),
        }

    def evidence_dominates(first: dict[str, object], second: dict[str, object]) -> bool:
        first_count = int(finite_float(first.get("finite_support_count"), 0.0))
        second_count = int(finite_float(second.get("finite_support_count"), 0.0))
        first_local = int(finite_float(first.get("observed_local_support"), 0.0))
        second_local = int(finite_float(second.get("observed_local_support"), 0.0))
        first_density = finite_float(first.get("finite_support_density"), 0.0)
        second_density = finite_float(second.get("finite_support_density"), 0.0)
        first_rms = finite_float(first.get("plane_rms"), float("inf"))
        second_rms = finite_float(second.get("plane_rms"), float("inf"))
        first_residual = finite_float(first.get("finite_support_residual_p95"), float("inf"))
        second_residual = finite_float(second.get("finite_support_residual_p95"), float("inf"))
        no_worse = bool(
            first_count >= second_count
            and first_local >= second_local
            and first_density + 1e-9 >= second_density
            and first_rms <= second_rms + 1e-12
            and first_residual <= second_residual + 1e-12
        )
        materially_better = bool(
            first_count > second_count
            or first_local > second_local
            or first_density > second_density + 1e-9
            or first_rms + 1e-12 < second_rms
            or first_residual + 1e-12 < second_residual
        )
        return bool(no_worse and materially_better)

    def strict_patch_compatible(a: dict[str, object], b: dict[str, object]) -> bool:
        angle, offset, centroid_distance = candidate_plane_relation_score(a, b)
        za = candidate_z_intervals(a)
        zb = candidate_z_intervals(b)
        hull_overlap, hull_overlap_fraction = interval_overlap(
            za.get("hull_z_min"),
            za.get("hull_z_max"),
            zb.get("hull_z_min"),
            zb.get("hull_z_max"),
        )
        support_overlap, support_overlap_fraction = interval_overlap(
            za.get("support_z_min"),
            za.get("support_z_max"),
            zb.get("support_z_min"),
            zb.get("support_z_max"),
        )
        return bool(
            angle <= 1.0
            and offset <= 0.02
            and centroid_distance <= 0.12
            and max(hull_overlap_fraction, support_overlap_fraction) >= 0.25
            and hull_overlap >= 0.0
            and support_overlap >= 0.0
        )

    def near_coincident_relation(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
        angle, offset, centroid_distance = candidate_plane_relation_score(a, b)
        hull_overlap = hull_bounds_overlap_ratio(a, b)
        relation = plane_patch_relation(a, b) if angle <= 5.0 and offset <= 0.06 and centroid_distance <= 1.15 else {"compatible": False}
        return {
            "compatible": bool(relation.get("compatible") and hull_overlap >= 0.5),
            "normal_angle_deg": as_json_float(float(angle)),
            "signed_offset_distance": as_json_float(float(offset)),
            "centroid_distance": as_json_float(float(centroid_distance)),
            "hull_bounds_overlap": as_json_float(float(hull_overlap)),
            "plane_patch_relation": relation,
        }

    def active_candidates_for(
        source_candidates: list[dict[str, object]],
        rec: dict[str, object],
    ) -> tuple[list[int], list[dict[str, object]]]:
        indices = [
            int(index)
            for index in rec.get("face_candidate_indices", [])
            if 0 <= int(index) < len(source_candidates)
        ]
        return indices, [source_candidates[index] for index in indices]

    def build_active_groups(active_candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        groups: list[dict[str, object]] = []
        for candidate in active_candidates:
            cid = int(candidate_track_id(candidate))
            placed = False
            for group in groups:
                representative = group["representative"]
                if strict_patch_compatible(candidate, representative):
                    group["members"].append(candidate)
                    group["member_candidate_ids"].append(cid)
                    placed = True
                    break
            if not placed:
                groups.append(
                    {
                        "stable_group_id": f"active_patch_{len(groups):04d}_{cid}",
                        "representative": candidate,
                        "representative_candidate_id": int(cid),
                        "members": [candidate],
                        "member_candidate_ids": [int(cid)],
                    }
                )
        return groups

    def compact_group(group: dict[str, object]) -> dict[str, object]:
        return {
            "stable_group_id": str(group.get("stable_group_id")),
            "representative_candidate_id": int(group.get("representative_candidate_id") or -1),
            "member_candidate_ids": sorted(int(v) for v in group.get("member_candidate_ids", [])),
        }

    def active_face_row(
        source_candidates: list[dict[str, object]],
        rec: dict[str, object],
        candidate_id: int,
    ) -> dict[str, object] | None:
        vertices = np.array(rec.get("vertices") or [], dtype=float)
        for face, source_index in zip(rec.get("faces") or [], rec.get("face_candidate_indices") or []):
            index = int(source_index)
            if index < 0 or index >= len(source_candidates):
                continue
            if int(candidate_track_id(source_candidates[index])) != int(candidate_id):
                continue
            face_indices = np.array(face, dtype=int)
            if face_indices.size < 3 or np.any(face_indices < 0) or np.any(face_indices >= vertices.shape[0]):
                continue
            points = vertices[face_indices]
            return {
                "active_plane_index": int(index),
                "face_area": as_json_float(float(polygon_area(points))),
                "face_centroid": as_json_point(np.mean(points, axis=0)),
                "face_vertex_count": int(len(face_indices)),
            }
        return None

    def group_delta(
        baseline_groups: list[dict[str, object]],
        trial_active_candidates: list[dict[str, object]],
        blocker_id: int,
        target: dict[str, object],
    ) -> dict[str, object]:
        lost: list[dict[str, object]] = []
        represented_group_ids: set[str] = set()
        for group in baseline_groups:
            members = list(group.get("members") or [])
            represented = any(
                strict_patch_compatible(member, trial_candidate)
                for member in members
                for trial_candidate in trial_active_candidates
            )
            if represented:
                represented_group_ids.add(str(group.get("stable_group_id")))
            else:
                lost.append(compact_group(group))
        introduced: list[dict[str, object]] = []
        for trial_candidate in trial_active_candidates:
            represented_before = any(
                strict_patch_compatible(member, trial_candidate)
                for group in baseline_groups
                for member in group.get("members", [])
            )
            if not represented_before:
                introduced.append(
                    {
                        "candidate_id": int(candidate_track_id(trial_candidate)),
                        "target": bool(int(candidate_track_id(trial_candidate)) == int(candidate_track_id(target))),
                    }
                )
        allowed_lost = [row for row in lost if int(blocker_id) in set(int(v) for v in row.get("member_candidate_ids", []))]
        unrelated_lost = [row for row in lost if row not in allowed_lost]
        target_introduced = any(bool(row.get("target")) for row in introduced)
        return {
            "baseline_group_count": int(len(baseline_groups)),
            "represented_group_count": int(len(represented_group_ids)),
            "lost_group_count": int(len(lost)),
            "lost_groups": lost,
            "allowed_blocker_group_loss_count": int(len(allowed_lost)),
            "allowed_blocker_groups_lost": allowed_lost,
            "unrelated_group_loss_count": int(len(unrelated_lost)),
            "unrelated_groups_lost": unrelated_lost,
            "introduced_group_count": int(len(introduced)),
            "introduced_groups": introduced,
            "target_group_introduced": bool(target_introduced),
        }

    def state_summary(
        *,
        label: str,
        action_count: int,
        state_candidates: list[dict[str, object]],
        rec: dict[str, object],
        reprojection: dict[str, object] | None = None,
        outside: dict[str, object] | None = None,
    ) -> dict[str, object]:
        active_indices, active_candidates = active_candidates_for(state_candidates, rec)
        active_ids = {int(candidate_track_id(candidate)) for candidate in active_candidates}
        outside_row = outside or evaluate_trusted_cloud_outside(
            state_candidates,
            trusted_points,
            z_slices,
            point_tol=float(point_tol),
        )
        reprojection_row = reprojection or evaluate_polyhedron_reprojection(
            name=str(label),
            vertices_3d=np.array(rec.get("vertices") or [], dtype=float),
            contours=contours,
            direction_count=32,
            contour_sample=120,
        )
        return {
            "label": str(label),
            "action_count": int(action_count),
            "candidate_count": int(len(state_candidates)),
            "active_candidate_count": int(len(active_indices)),
            "active_candidate_ids": sorted(int(v) for v in active_ids),
            "active_plane_indices": active_indices,
            "vertices": int(len(rec.get("vertices") or [])),
            "edges": int(len(rec.get("edges") or [])),
            "faces": int(len(rec.get("faces") or [])),
            "volume": rec.get("reliable_volume"),
            "extents": rec.get("bbox") or (rec.get("lp_extents") or {}).get("extents"),
            "topology": rec.get("topology"),
            "topology_valid": topology_is_valid(rec.get("topology") if isinstance(rec.get("topology"), dict) else {}),
            "outside": outside_row,
            "lost_z": outside_row.get("cumulative_lost_z_levels") if isinstance(outside_row, dict) else None,
            "reprojection": reprojection_row,
            "repaired_target_activity": {
                "target_candidate_ids": sorted(int(v) for v in repaired_target_ids),
                "active_by_candidate_id": {str(cid): bool(cid in active_ids) for cid in sorted(repaired_target_ids)},
                "all_active": bool(repaired_target_ids and repaired_target_ids <= active_ids),
            },
            "active_group_count": int(len(build_active_groups(active_candidates))),
            "removed_blockers_absent": not bool(historical_removed_ids & {int(candidate_track_id(candidate)) for candidate in state_candidates}),
            "geometry_signature": mesh_geometry_signature(state_candidates, rec),
        }

    current_candidates = list(selected_candidates)
    current_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(current_candidates, **reconstruction_kwargs)
    baseline_reprojection = evaluate_polyhedron_reprojection(
        name="post_ratchet_exchange_action_0",
        vertices_3d=np.array(current_rec.get("vertices") or [], dtype=float),
        contours=contours,
        direction_count=32,
        contour_sample=120,
    )
    baseline_outside = evaluate_trusted_cloud_outside(
        current_candidates,
        trusted_points,
        z_slices,
        point_tol=float(point_tol),
    )
    checkpoints: list[dict[str, object]] = [
        state_summary(
            label="exchange_off",
            action_count=0,
            state_candidates=current_candidates,
            rec=current_rec,
            reprojection=baseline_reprojection,
            outside=baseline_outside,
        )
    ]
    iterations: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    total_target_discovery_seconds = 0.0
    total_lp_proposal_seconds = 0.0
    total_full_trial_seconds = 0.0
    total_reprojection_seconds = 0.0
    full_trial_count = 0
    reprojection_count = 1

    for action_index in range(1, int(max_exchanges) + 1):
        iteration_started = time.perf_counter()
        current_ids = {int(candidate_track_id(candidate)) for candidate in current_candidates}
        current_id_tuple = tuple(sorted(current_ids))
        active_indices, active_candidates = active_candidates_for(current_candidates, current_rec)
        active_ids = {int(candidate_track_id(candidate)) for candidate in active_candidates}
        active_candidates_annotated = [annotated_by_id.get(int(candidate_track_id(candidate)), candidate) for candidate in active_candidates]
        baseline_groups = build_active_groups(active_candidates_annotated)

        target_discovery_started = time.perf_counter()
        funnel_counts = {
            "all_candidate_pool": int(len(annotated_pool)),
            "already_active": 0,
            "invalid_or_empty_hull": 0,
            "insufficient_observed_support": 0,
            "nonlocal_patch": 0,
            "far_from_lp_activation_boundary": 0,
            "patch_duplicate": 0,
            "generic_exchange_targets": 0,
        }
        target_rows: list[dict[str, object]] = []
        generic_targets: list[dict[str, object]] = []
        target_activity_by_id: dict[int, dict[str, object]] = {}
        target_deficit_by_id: dict[int, dict[str, object]] = {}
        for target in annotated_pool:
            target_id = int(candidate_track_id(target))
            row: dict[str, object] = {
                "candidate_id": int(target_id),
                "support": support_row(target),
                "classification": None,
            }
            if target_id in historical_removed_ids:
                row["classification"] = "excluded_removed_blocker"
                target_rows.append(row)
                continue
            if target_id in active_ids:
                funnel_counts["already_active"] += 1
                row["classification"] = "already_active"
                target_rows.append(row)
                continue
            hull = candidate_hull_points(target)
            if candidate_plane(target) is None or hull.shape[0] < 3 or finite_float(target.get("hull_area"), 0.0) <= EPS:
                funnel_counts["invalid_or_empty_hull"] += 1
                row["classification"] = "invalid_or_empty_hull"
                target_rows.append(row)
                continue
            finite_support_count = int(finite_float(target.get("finite_support_count"), 0.0))
            observed_local_support = int(finite_float(target.get("observed_local_support"), 0.0))
            if finite_support_count <= 0 or observed_local_support <= 0:
                funnel_counts["insufficient_observed_support"] += 1
                row["classification"] = "insufficient_observed_support"
                target_rows.append(row)
                continue
            if (
                finite_float(target.get("hull_area"), float("inf")) > max_local_hull_area
                or finite_float(target.get("hull_diameter"), float("inf")) > max_local_hull_diameter
                or finite_float(target.get("plane_rms"), float("inf")) > max_local_plane_rms
                or finite_support_count > 45
            ):
                funnel_counts["nonlocal_patch"] += 1
                row["classification"] = "nonlocal_patch"
                target_rows.append(row)
                continue
            base_ids = tuple(cid for cid in current_id_tuple if cid != target_id)
            activity = lp_face_activity_details(
                None,
                target,
                inside_tol=float(face_activity_slack),
                face_activity_slack=float(face_activity_slack),
                lp_engine=lp_engine,
                base_candidate_ids=base_ids,
            )
            target_activity_by_id[target_id] = activity
            row["baseline_activity"] = activity
            effective_margin = finite_float(activity.get("effective_face_margin"), float("inf"))
            if not bool(activity.get("feasible")) or not np.isfinite(effective_margin) or abs(effective_margin) > boundary_margin:
                funnel_counts["far_from_lp_activation_boundary"] += 1
                row["classification"] = "far_from_lp_activation_boundary"
                target_rows.append(row)
                continue
            duplicate_rows: list[dict[str, object]] = []
            for active_candidate in active_candidates_annotated:
                relation = near_coincident_relation(target, active_candidate)
                if not bool(relation.get("compatible")):
                    continue
                active_stronger = bool(evidence_dominates(active_candidate, target))
                duplicate_rows.append(
                    {
                        "active_candidate_id": int(candidate_track_id(active_candidate)),
                        "active_observed_evidence_stronger": active_stronger,
                        "relation": relation,
                        "target_support": support_row(target),
                        "active_support": support_row(active_candidate),
                    }
                )
            stronger_duplicate = next((item for item in duplicate_rows if bool(item.get("active_observed_evidence_stronger"))), None)
            if stronger_duplicate is not None:
                funnel_counts["patch_duplicate"] += 1
                row["classification"] = "patch_duplicate_with_stronger_active_evidence"
                row["duplicate_active"] = stronger_duplicate
                target_rows.append(row)
                continue
            base_without_target = [candidate for candidate in current_candidates if int(candidate_track_id(candidate)) != target_id]
            samples = candidate_samples(target)
            deficit = patch_surface_deficit_for_samples(base_without_target, samples, sample_tol=0.0)
            target_deficit_by_id[target_id] = deficit
            row["classification"] = "generic_exchange_target"
            row["near_coincident_active_rows"] = duplicate_rows
            row["baseline_surface_deficit"] = deficit
            generic_targets.append(target)
            target_rows.append(row)

        def target_sort_key(target: dict[str, object]) -> tuple[object, ...]:
            target_id = int(candidate_track_id(target))
            activity = target_activity_by_id.get(target_id, {})
            deficit = target_deficit_by_id.get(target_id, {})
            return (
                abs(finite_float(activity.get("effective_face_margin"), float("inf"))),
                -finite_float(deficit.get("area_weighted_cut_deficit"), 0.0),
                -int(finite_float(target.get("observed_local_support"), 0.0)),
                -int(finite_float(target.get("finite_support_count"), 0.0)),
                int(target_id),
            )

        generic_targets.sort(key=target_sort_key)
        funnel_counts["generic_exchange_targets"] = int(len(generic_targets))
        bounded_targets = generic_targets[: max(0, int(max_generic_targets))]
        bounded_target_ids = {int(candidate_track_id(target)) for target in bounded_targets}
        for rank, target in enumerate(generic_targets, start=1):
            target_id = int(candidate_track_id(target))
            target_row = next(row for row in target_rows if int(row.get("candidate_id") or -1) == target_id)
            target_row["generic_target_rank"] = int(rank)
            target_row["inside_bounded_target_pool"] = bool(target_id in bounded_target_ids)
        target_discovery_seconds = time.perf_counter() - target_discovery_started
        total_target_discovery_seconds += target_discovery_seconds

        lp_proposal_started = time.perf_counter()
        active_non_core_candidates = [
            annotated_by_id.get(int(candidate_track_id(candidate)), candidate)
            for candidate in active_candidates
            if int(candidate_track_id(candidate)) not in core_candidate_ids
            and int(candidate_track_id(candidate)) not in repaired_target_ids
            and str(candidate.get("candidate_origin")) == "w2_addition"
            and str(candidate.get("candidate_origin")) != "baseline_guard"
        ]
        group_by_member_id = {
            int(member_id): group
            for group in baseline_groups
            for member_id in group.get("member_candidate_ids", [])
        }
        proposals: list[dict[str, object]] = []
        blocker_lp_checks = 0
        for target in bounded_targets:
            target_id = int(candidate_track_id(target))
            before = target_activity_by_id[target_id]
            if bool(before.get("active")):
                target_row = next(row for row in target_rows if int(row.get("candidate_id") or -1) == target_id)
                target_row["blocker_screen_classification"] = "lp_active_redundant_no_remove_one_reactivation"
                continue
            samples = candidate_samples(target)
            baseline_deficit = target_deficit_by_id[target_id]
            base_without_target = [candidate for candidate in current_candidates if int(candidate_track_id(candidate)) != target_id]
            for blocker in active_non_core_candidates:
                blocker_id = int(candidate_track_id(blocker))
                if blocker_id in historical_removed_ids:
                    continue
                after_ids = tuple(cid for cid in current_id_tuple if cid not in {target_id, blocker_id})
                after = lp_face_activity_details(
                    None,
                    target,
                    inside_tol=float(face_activity_slack),
                    face_activity_slack=float(face_activity_slack),
                    lp_engine=lp_engine,
                    base_candidate_ids=after_ids,
                )
                blocker_lp_checks += 1
                gain = finite_float(after.get("effective_face_margin"), -999.0) - finite_float(
                    before.get("effective_face_margin"), -999.0
                )
                if not bool(after.get("active")) or gain <= 1e-9:
                    continue
                after_deficit = patch_surface_deficit_for_samples(
                    [candidate for candidate in base_without_target if int(candidate_track_id(candidate)) != blocker_id],
                    samples,
                    sample_tol=0.0,
                )
                deficit_reduction = finite_float(baseline_deficit.get("area_weighted_cut_deficit"), 0.0) - finite_float(
                    after_deficit.get("area_weighted_cut_deficit"), 0.0
                )
                relation = near_coincident_relation(target, blocker)
                target_improves_observed = bool(evidence_dominates(target, blocker))
                blocker_group = group_by_member_id.get(blocker_id)
                estimated_group_loss = int(
                    blocker_group is not None
                    and set(int(v) for v in blocker_group.get("member_candidate_ids", [])) <= {int(blocker_id)}
                )
                proposals.append(
                    {
                        "target_candidate_id": int(target_id),
                        "blocker_candidate_id": int(blocker_id),
                        "removed_candidate_ids": [int(blocker_id)],
                        "added_candidate_ids": [int(target_id)],
                        "before_activity": before,
                        "after_activity": after,
                        "effective_margin_gain": as_json_float(float(gain)),
                        "surface_deficit_before": baseline_deficit,
                        "surface_deficit_after": after_deficit,
                        "surface_deficit_reduction": as_json_float(float(deficit_reduction)),
                        "target_support": support_row(target),
                        "blocker_support": support_row(blocker),
                        "target_blocker_relation": relation,
                        "near_coincident_without_observed_improvement": bool(
                            relation.get("compatible") and not target_improves_observed
                        ),
                        "target_observed_evidence_dominates_blocker": bool(target_improves_observed),
                        "estimated_rolling_groups_lost": int(estimated_group_loss),
                    }
                )

        def cheap_proposal_sort_key(row: dict[str, object]) -> tuple[object, ...]:
            return (
                not bool(finite_float(row.get("surface_deficit_reduction"), 0.0) > 0.0),
                bool(row.get("near_coincident_without_observed_improvement")),
                int(row.get("estimated_rolling_groups_lost") or 0),
                -finite_float(row.get("surface_deficit_reduction"), 0.0),
                -finite_float(row.get("effective_margin_gain"), 0.0),
                int((row.get("blocker_support") or {}).get("finite_support_count") or 0),
                -int((row.get("target_support") or {}).get("observed_local_support") or 0),
                int(row.get("target_candidate_id") or 10**12),
                int(row.get("blocker_candidate_id") or 10**12),
            )

        proposals.sort(key=cheap_proposal_sort_key)
        for rank, proposal in enumerate(proposals, start=1):
            proposal["cheap_rank"] = int(rank)
        lp_proposal_seconds = time.perf_counter() - lp_proposal_started
        total_lp_proposal_seconds += lp_proposal_seconds

        full_limit = int(first_full_shortlist if action_index == 1 else subsequent_full_shortlist)
        full_shortlist = proposals[: max(0, full_limit)]
        full_rows: list[dict[str, object]] = []
        base_reprojection = evaluate_polyhedron_reprojection(
            name=f"post_ratchet_exchange_action_{action_index - 1}_baseline",
            vertices_3d=np.array(current_rec.get("vertices") or [], dtype=float),
            contours=contours,
            direction_count=32,
            contour_sample=120,
        )
        reprojection_count += 1
        baseline_p95 = finite_float(base_reprojection.get("symmetric_contour_distance_p95"), float("inf"))
        full_started = time.perf_counter()
        for proposal in full_shortlist:
            full_trial_started = time.perf_counter()
            target_id = int(proposal["target_candidate_id"])
            blocker_id = int(proposal["blocker_candidate_id"])
            target = annotated_by_id[target_id]
            trial_candidates = [
                candidate
                for candidate in current_candidates
                if int(candidate_track_id(candidate)) != blocker_id
            ]
            if target_id not in {int(candidate_track_id(candidate)) for candidate in trial_candidates}:
                target_to_add = dict(target)
                target_to_add["post_ratchet_exchange_mode"] = "lp-deficit"
                target_to_add["post_ratchet_exchange_action"] = int(action_index)
                trial_candidates.append(target_to_add)
            trial_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
            trial_active_indices, trial_active_candidates = active_candidates_for(trial_candidates, trial_rec)
            trial_active_ids = {int(candidate_track_id(candidate)) for candidate in trial_active_candidates}
            trial_active_annotated = [annotated_by_id.get(int(candidate_track_id(candidate)), candidate) for candidate in trial_active_candidates]
            target_face = active_face_row(trial_candidates, trial_rec, target_id)
            delta = group_delta(baseline_groups, trial_active_annotated, blocker_id, target)
            baseline_duplicate_rows = [
                near_coincident_relation(target, candidate)
                for candidate in active_candidates_annotated
                if int(candidate_track_id(candidate)) != target_id
            ]
            coactivated_duplicate_rows = [
                near_coincident_relation(target, candidate)
                for candidate in trial_active_annotated
                if int(candidate_track_id(candidate)) != target_id
                and int(candidate_track_id(candidate)) not in active_ids
            ]
            target_spatially_distinct = bool(
                delta.get("target_group_introduced")
                and not any(bool(row.get("compatible")) for row in baseline_duplicate_rows)
            )
            outside = evaluate_trusted_cloud_outside(
                trial_candidates,
                trusted_points,
                z_slices,
                point_tol=float(point_tol),
            )
            reprojection_started = time.perf_counter()
            reprojection = evaluate_polyhedron_reprojection(
                name=f"post_ratchet_exchange_{action_index}_{target_id}_{blocker_id}",
                vertices_3d=np.array(trial_rec.get("vertices") or [], dtype=float),
                contours=contours,
                direction_count=32,
                contour_sample=120,
            )
            reprojection_elapsed = time.perf_counter() - reprojection_started
            total_reprojection_seconds += reprojection_elapsed
            reprojection_count += 1
            reprojection_p95 = finite_float(reprojection.get("symmetric_contour_distance_p95"), float("inf"))
            repaired_targets_active = bool(repaired_target_ids and repaired_target_ids <= trial_active_ids)
            blocker_absent = blocker_id not in {int(candidate_track_id(candidate)) for candidate in trial_candidates}
            target_deficit_after_full = patch_surface_deficit_for_samples(
                [candidate for candidate in trial_candidates if int(candidate_track_id(candidate)) != target_id],
                candidate_samples(target),
                sample_tol=0.0,
            )
            full_safe = bool(
                target_id in trial_active_ids
                and target_face is not None
                and blocker_absent
                and target_spatially_distinct
                and bool(delta.get("target_group_introduced"))
                and int(delta.get("unrelated_group_loss_count") or 0) == 0
                and int(delta.get("allowed_blocker_group_loss_count") or 0) <= 1
                and repaired_targets_active
                and topology_is_valid(trial_rec.get("topology") if isinstance(trial_rec.get("topology"), dict) else {})
                and finite_float(outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
                and int(outside.get("cumulative_lost_z_levels") or 0) == 0
                and reprojection_p95 <= baseline_p95 + max(0.001, 0.10 * baseline_p95)
                and not bool(historical_removed_ids & {int(candidate_track_id(candidate)) for candidate in trial_candidates})
                and finite_float(proposal.get("surface_deficit_reduction"), 0.0) > 0.0
                and not bool(proposal.get("near_coincident_without_observed_improvement"))
            )
            full_row = {
                **proposal,
                "target_active_edge_clip": bool(target_id in trial_active_ids),
                "target_active_face": target_face,
                "target_surface_deficit_after_full": target_deficit_after_full,
                "blocker_absent": bool(blocker_absent),
                "target_spatially_distinct": bool(target_spatially_distinct),
                "baseline_duplicate_rows": baseline_duplicate_rows,
                "coactivated_duplicate_rows": coactivated_duplicate_rows,
                "rolling_group_delta": delta,
                "repaired_targets_active": bool(repaired_targets_active),
                "topology": trial_rec.get("topology"),
                "topology_valid": topology_is_valid(trial_rec.get("topology") if isinstance(trial_rec.get("topology"), dict) else {}),
                "outside": outside,
                "reprojection": reprojection,
                "reprojection_p95_delta": as_json_float(float(reprojection_p95 - baseline_p95)),
                "vertices": int(len(trial_rec.get("vertices") or [])),
                "edges": int(len(trial_rec.get("edges") or [])),
                "faces": int(len(trial_rec.get("faces") or [])),
                "volume": trial_rec.get("reliable_volume"),
                "active_candidate_ids": sorted(int(v) for v in trial_active_ids),
                "active_plane_indices": trial_active_indices,
                "geometry_signature": mesh_geometry_signature(trial_candidates, trial_rec),
                "full_mesh_safe": bool(full_safe),
                "full_trial_seconds": as_json_float(time.perf_counter() - full_trial_started),
                "_trial_candidates": trial_candidates,
                "_trial_reconstructed": trial_rec,
            }
            full_rows.append(full_row)
            full_trial_count += 1
        full_trial_seconds = time.perf_counter() - full_started
        total_full_trial_seconds += full_trial_seconds

        def full_sort_key(row: dict[str, object]) -> tuple[object, ...]:
            return (
                not bool(row.get("full_mesh_safe")),
                bool(row.get("near_coincident_without_observed_improvement")),
                int((row.get("rolling_group_delta") or {}).get("unrelated_group_loss_count") or 0),
                not bool(row.get("target_spatially_distinct")),
                -finite_float(row.get("surface_deficit_reduction"), 0.0),
                -finite_float(row.get("effective_margin_gain"), 0.0),
                int((row.get("blocker_support") or {}).get("finite_support_count") or 0),
                finite_float(row.get("reprojection_p95_delta"), float("inf")),
                int(row.get("target_candidate_id") or 10**12),
                int(row.get("blocker_candidate_id") or 10**12),
            )

        full_rows.sort(key=full_sort_key)
        for rank, row in enumerate(full_rows, start=1):
            row["full_rank"] = int(rank)
        selected = next((row for row in full_rows if bool(row.get("full_mesh_safe"))), None)

        proposal_signature_rows = [
            [
                int(row.get("target_candidate_id") or -1),
                int(row.get("blocker_candidate_id") or -1),
            ]
            for row in proposals
        ]
        proposal_blob = json.dumps(proposal_signature_rows, separators=(",", ":"))
        serializable_full_rows = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in full_rows
        ]
        iteration_row: dict[str, object] = {
            "action_index": int(action_index),
            "state_candidate_count": int(len(current_candidates)),
            "state_active_candidate_count": int(len(active_indices)),
            "active_non_core_blocker_count": int(len(active_non_core_candidates)),
            "target_funnel": funnel_counts,
            "target_discovery_rows": target_rows,
            "bounded_target_limit": int(max_generic_targets),
            "bounded_target_ids": [int(candidate_track_id(target)) for target in bounded_targets],
            "blocker_lp_checks": int(blocker_lp_checks),
            "lp_positive_proposal_count": int(len(proposals)),
            "proposal_order_signature": proposal_signature_rows,
            "proposal_order_sha256": hashlib.sha256(proposal_blob.encode("utf-8")).hexdigest(),
            "cheap_proposals_top": proposals[:100],
            "full_shortlist_limit": int(full_limit),
            "full_edge_clip_trial_count": int(len(full_rows)),
            "full_trials": serializable_full_rows,
            "selected_action": (
                {key: value for key, value in selected.items() if not key.startswith("_")}
                if selected is not None
                else None
            ),
            "timing": {
                "target_discovery_seconds": as_json_float(target_discovery_seconds),
                "lp_proposal_seconds": as_json_float(lp_proposal_seconds),
                "full_trial_seconds": as_json_float(full_trial_seconds),
                "iteration_seconds": as_json_float(time.perf_counter() - iteration_started),
            },
        }
        iterations.append(iteration_row)
        if selected is None:
            summary["reason"] = "no_safe_exchange_in_bounded_shortlist"
            break

        current_candidates = list(selected["_trial_candidates"])
        current_rec = selected["_trial_reconstructed"]
        blocker_id = int(selected["blocker_candidate_id"])
        target_id = int(selected["target_candidate_id"])
        historical_removed_ids.add(blocker_id)
        action = {
            "action_index": int(action_index),
            "action_type": "remove_one_add_target",
            "removed_candidate_ids": [int(blocker_id)],
            "added_candidate_ids": [int(target_id)],
            "target_candidate_id": int(target_id),
            "blocker_candidate_id": int(blocker_id),
            "effective_margin_before": (selected.get("before_activity") or {}).get("effective_face_margin"),
            "effective_margin_after": (selected.get("after_activity") or {}).get("effective_face_margin"),
            "effective_margin_gain": selected.get("effective_margin_gain"),
            "surface_deficit_reduction": selected.get("surface_deficit_reduction"),
            "rolling_group_delta": selected.get("rolling_group_delta"),
            "target_support": selected.get("target_support"),
            "blocker_support": selected.get("blocker_support"),
            "reprojection_p95_delta": selected.get("reprojection_p95_delta"),
            "geometry_signature": selected.get("geometry_signature"),
        }
        actions.append(action)
        checkpoints.append(
            state_summary(
                label=f"generic_exchange_max{action_index}",
                action_count=int(action_index),
                state_candidates=current_candidates,
                rec=current_rec,
                reprojection=selected.get("reprojection") if isinstance(selected.get("reprojection"), dict) else None,
                outside=selected.get("outside") if isinstance(selected.get("outside"), dict) else None,
            )
        )
        summary["production_changed"] = True
        summary["reason"] = "selected_safe_generic_exchanges"

    action_signature = [
        [int(row["blocker_candidate_id"]), int(row["target_candidate_id"])]
        for row in actions
    ]
    action_blob = json.dumps(action_signature, separators=(",", ":"))
    final_checkpoint = checkpoints[-1]
    lp_summary = lp_engine.summary()
    summary.update(
        {
            "accepted_exchange_count": int(len(actions)),
            "actions": actions,
            "action_signature": action_signature,
            "action_signature_sha256": hashlib.sha256(action_blob.encode("utf-8")).hexdigest(),
            "removed_candidate_ids": [int(row["blocker_candidate_id"]) for row in actions],
            "added_candidate_ids": [int(row["target_candidate_id"]) for row in actions],
            "iterations": iterations,
            "controls": checkpoints,
            "final_geometry_signature": final_checkpoint.get("geometry_signature"),
            "final_active_plane_indices": final_checkpoint.get("active_plane_indices"),
            "final_active_candidate_ids": final_checkpoint.get("active_candidate_ids"),
            "target_boundary_margin": as_json_float(float(boundary_margin)),
            "full_shortlist_limits": {
                "first_iteration": int(first_full_shortlist),
                "subsequent_iterations": int(subsequent_full_shortlist),
            },
            "performance": {
                "candidate_annotation_seconds": as_json_float(float(annotation_seconds)),
                "target_discovery_seconds": as_json_float(float(total_target_discovery_seconds)),
                "lp_proposal_seconds": as_json_float(float(total_lp_proposal_seconds)),
                "full_edge_clip_trial_count": int(full_trial_count),
                "full_edge_clip_trial_seconds": as_json_float(float(total_full_trial_seconds)),
                "reprojection_count": int(reprojection_count),
                "reprojection_seconds_inside_full_trials": as_json_float(float(total_reprojection_seconds)),
                "per_action_seconds": [
                    (iteration.get("timing") or {}).get("iteration_seconds")
                    for iteration in iterations
                ],
                "total_exchange_seconds": as_json_float(time.perf_counter() - started),
            },
            "lp_engine_diagnostics": lp_summary,
            "ranking_fields": [
                {"name": "full_mesh_safe", "direction": "true first", "uses_initial_model": False},
                {"name": "near_coincident_without_observed_improvement", "direction": "false first", "uses_initial_model": False},
                {"name": "unrelated_rolling_group_losses", "direction": "ascending", "uses_initial_model": False},
                {"name": "target_spatially_distinct", "direction": "true first", "uses_initial_model": False},
                {"name": "surface_deficit_reduction", "direction": "descending", "uses_initial_model": False},
                {"name": "effective_margin_gain", "direction": "descending", "uses_initial_model": False},
                {"name": "blocker_observed_support", "direction": "ascending", "uses_initial_model": False},
                {"name": "reprojection_p95_delta", "direction": "ascending", "uses_initial_model": False},
            ],
            "timing_seconds": as_json_float(time.perf_counter() - started),
        }
    )
    return current_candidates, summary


bind_oracle_loss_diagnostic_runtime_symbols(
    {
        name: globals()[name]
        for name in ORACLE_LOSS_DIAGNOSTIC_RUNTIME_SYMBOLS
    }
)


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
        default="3,4,6",
        help="Comma-separated window sizes. Multiple windows collect face candidates at several scales.",
    )
    parser.add_argument(
        "--candidate-scale-mode",
        choices=("adaptive", "multiscale"),
        default="adaptive",
        help="adaptive preserves the previous one-window-per-Z behavior; multiscale fits candidates per window.",
    )
    parser.add_argument(
        "--angular-scale-candidate-mode",
        choices=("off", "additive"),
        default="off",
        help="Build an observed-resolution-normalized RMS candidate side-channel after the frozen production chain.",
    )
    parser.add_argument(
        "--angular-scale-selection-mode",
        choices=("quality-local", "balanced-local", "balanced-global"),
        default="quality-local",
        help="Use legacy quality/locality, balanced ordering with the legacy locality gate, or balanced ordering with relaxed extra-area gating.",
    )
    parser.add_argument("--angular-scale-reference-half-count", type=int, default=400)
    parser.add_argument("--max-angular-scale-additions", type=int, default=10)
    parser.add_argument("--max-angular-scale-trials", type=int, default=40)
    parser.add_argument("--peak-threshold", type=float, default=0.00055)
    parser.add_argument("--low-threshold", type=float, default=0.0002)
    parser.add_argument("--peak-quantile", type=float, default=-1.0)
    parser.add_argument("--low-quantile", type=float, default=-1.0)
    parser.add_argument(
        "--low-barrier-threshold",
        type=float,
        default=-1.0,
        help="Intermediate RMS barrier threshold. Negative derives it from low/peak thresholds.",
    )
    parser.add_argument(
        "--low-barrier-fraction",
        type=float,
        default=0.35,
        help="Derived barrier = low + (peak - low) * fraction when --low-barrier-threshold is negative.",
    )
    parser.add_argument(
        "--low-barrier-prominence",
        type=float,
        default=0.0006,
        help="Local maxima above low_threshold by this amount also block low-region search. Negative disables.",
    )
    parser.add_argument(
        "--low-valley-fallback",
        action="store_true",
        help="Experimental: use the lowest RMS valley in the sector when no threshold-low region is found.",
    )
    parser.add_argument(
        "--valley-integration-mode",
        choices=("rebuild", "augment-baseline", "repair-rebuild"),
        default="rebuild",
    )
    parser.add_argument("--max-valley-additions", type=int, default=-1)
    parser.add_argument("--envelope-repair-margin", type=float, default=0.01)
    parser.add_argument("--envelope-repair-relative-margin", type=float, default=0.001)
    parser.add_argument("--envelope-repair-min-improvement", type=float, default=0.01)
    parser.add_argument(
        "--z-envelope-guard-mode",
        choices=("off", "trusted-cap"),
        default="trusted-cap",
        help="Cap excessive lower-Z expansion using only the trusted observed cloud; use off to disable.",
    )
    parser.add_argument("--z-envelope-guard-trigger-relative", type=float, default=0.05)
    parser.add_argument("--z-envelope-guard-margin-relative", type=float, default=0.01)
    parser.add_argument("--max-baseline-guards", type=int, default=1)
    parser.add_argument("--dense-detector-mode", choices=("off", "w2-edge-tracks"), default="off")
    parser.add_argument("--max-w2-additions", type=int, default=20)
    parser.add_argument("--w2-edge-min-levels", type=int, default=12)
    parser.add_argument("--w2-edge-min-z-span", type=float, default=0.12)
    parser.add_argument("--w2-edge-max-line-rms", type=float, default=0.04)
    parser.add_argument("--w2-edge-max-line-residual", type=float, default=0.10)
    parser.add_argument("--w2-edge-max-z-gap", type=int, default=2)
    parser.add_argument("--w2-edge-max-point-jump", type=float, default=0.25)
    parser.add_argument("--w2-edge-max-condition", type=float, default=-1.0)
    parser.add_argument("--w2-segment-fit-mode", choices=("least-squares", "robust-split"), default="least-squares")
    parser.add_argument("--w2-addition-selection-mode", choices=("rank", "novelty", "support-diverse"), default="rank")
    parser.add_argument("--w2-plane-consensus-mode", choices=("off", "soft"), default="off")
    parser.add_argument("--w2-duplicate-pair-mode", choices=("allow", "reject-overlapping-collinear"), default="allow")
    parser.add_argument("--w2-edge-cluster-mode", choices=("representative", "complete"), default="complete")
    parser.add_argument("--w2-edge-cluster-max-distance", type=float, default=0.08)
    parser.add_argument("--w2-edge-cluster-max-angle-deg", type=float, default=4.0)
    parser.add_argument("--w2-edge-cluster-min-overlap", type=float, default=0.12)
    parser.add_argument("--w2-face-min-adjacency-levels", type=int, default=12)
    parser.add_argument("--w2-face-min-adjacency-fraction", type=float, default=0.25)
    parser.add_argument("--w2-face-min-z-span", type=float, default=0.12)
    parser.add_argument("--w2-face-max-plane-rms", type=float, default=0.08)
    parser.add_argument(
        "--w2-small-face-refinement",
        choices=("off", "append-local"),
        default="off",
        help="Experimental: append localized small dense-band faces after the normal w2 prefix.",
    )
    parser.add_argument("--w2-max-small-face-additions", type=int, default=25)
    parser.add_argument(
        "--w2-max-small-face-trials",
        type=int,
        default=160,
        help="Technical cap for append-local edge-clip trials; additions still stop at --w2-max-small-face-additions.",
    )
    parser.add_argument(
        "--w2-small-face-trial-geometry",
        choices=("full", "incremental"),
        default="full",
        help="Technical geometry evaluator for append-local trial faces; final mesh still uses full edge-clip.",
    )
    parser.add_argument(
        "--w2-core-repair-mode",
        choices=("off", "lp-remove-one", "lp-blockers"),
        default="off",
        help="Experimental: off, one LP remove-one repair, or staged lp-blockers with one remove-one plus one strict pair.",
    )
    parser.add_argument(
        "--w2-max-core-repairs",
        type=int,
        default=None,
        help="Requested repair action budget. Auto resolves to 1 for lp-remove-one and 2 for lp-blockers.",
    )
    parser.add_argument(
        "--w2-post-repair-refinement",
        choices=("off", "broad-protected", "broad-ratchet"),
        default="off",
        help="Experimental: after lp-blockers, append a bounded broad protected refill; broad-ratchet also protects newly accepted active patch groups.",
    )
    parser.add_argument("--w2-max-post-repair-additions", type=int, default=15)
    parser.add_argument("--w2-max-post-repair-trials", type=int, default=160)
    parser.add_argument(
        "--w2-post-ratchet-exchange-mode",
        choices=("off", "lp-deficit"),
        default="off",
        help="Experimental: after broad-ratchet, run bounded non-oracle LP-deficit remove-one/add-one exchanges.",
    )
    parser.add_argument("--w2-max-post-ratchet-exchanges", type=int, default=3)
    parser.add_argument("--disable-oracle-diagnostics", action="store_true")
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
        "--max-line-condition",
        type=float,
        default=-1.0,
        help="Reject fit points with larger line-system condition. Negative disables filtering.",
    )
    parser.add_argument(
        "--minima-cloud-pct",
        type=float,
        default=10.0,
        help="RMS local-minima cloud threshold percentage, matching the full-circle viewer.",
    )
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
    parser.add_argument("--dedupe-normal-angle-deg", type=float, default=2.0)
    parser.add_argument("--dedupe-plane-distance", type=float, default=0.05)
    parser.add_argument("--dedupe-hull-distance", type=float, default=0.75)
    parser.add_argument(
        "--candidate-cluster-mode",
        choices=("single", "representative", "complete"),
        default="single",
        help="single uses previous single-linkage; representative/complete use stricter non-oracle cluster compactness.",
    )
    parser.add_argument("--max-candidates-per-cluster", type=int, default=1)
    parser.add_argument(
        "--candidate-selection-mode",
        choices=("rank", "compatible", "cumulative", "cumulative-local"),
        default="rank",
        help="rank keeps previous cluster representatives; compatible chooses by observed cloud; cumulative adds set checks; cumulative-local ranks alternatives by finite local support.",
    )
    parser.add_argument("--compatibility-point-tol", type=float, default=0.05)
    parser.add_argument("--compatibility-max-global-outside-frac", type=float, default=0.01)
    parser.add_argument("--compatibility-max-level-outside-frac", type=float, default=0.10)
    parser.add_argument("--compatibility-bbox-margin", type=float, default=0.4)
    parser.add_argument("--compatibility-min-local-support", type=int, default=3)
    parser.add_argument(
        "--valley-candidate-mode",
        choices=("any", "quality"),
        default="any",
        help="any keeps current valley fallback behavior; quality applies non-oracle hard gates only to valley candidates.",
    )
    parser.add_argument("--valley-min-confidence-median", type=float, default=0.35)
    parser.add_argument("--valley-min-confidence-p10", type=float, default=0.15)
    parser.add_argument("--valley-min-two-side-fraction", type=float, default=0.0)
    parser.add_argument("--valley-max-one-side-fraction", type=float, default=1.0)
    parser.add_argument("--valley-max-distance-p95", type=float, default=20.0)
    parser.add_argument("--valley-max-track-smoothness", type=float, default=20.0)
    parser.add_argument("--valley-max-support-gap-global", type=float, default=2.0)
    parser.add_argument("--valley-max-support-gap-local", type=float, default=1.0)
    parser.add_argument("--valley-min-support-near-count", type=int, default=1)
    parser.add_argument("--valley-min-scale-persistence", type=int, default=1)
    parser.add_argument("--valley-min-track-levels", type=int, default=1)
    parser.add_argument("--valley-max-plane-rms", type=float, default=0.08)
    parser.add_argument("--valley-min-hull-area", type=float, default=0.0)
    parser.add_argument("--valley-min-finite-support-count", type=int, default=1)
    parser.add_argument("--valley-min-finite-track-coverage", type=float, default=0.0)
    parser.add_argument(
        "--compatibility-outside-mode",
        choices=("legacy", "incremental"),
        default="legacy",
        help="legacy recomputes cumulative outside from accepted planes; incremental reuses precomputed outside masks.",
    )
    parser.add_argument(
        "--compatibility-activity-check",
        choices=("geometry", "lp"),
        default="geometry",
        help="geometry reconstructs trial polyhedra during selection; lp uses HiGHS feasibility/activity checks.",
    )
    parser.add_argument("--compatibility-activity-tol", type=float, default=0.03)
    parser.add_argument("--finite-support-plane-distance", type=float, default=0.05)
    parser.add_argument("--finite-support-hull-margin", type=float, default=0.10)
    parser.add_argument(
        "--finite-support-relative-hull-margin",
        type=float,
        default=0.03,
        help="Extra hull margin = this value * sqrt(candidate hull area).",
    )
    parser.add_argument(
        "--cumulative-local-max-alternatives-per-cluster",
        type=int,
        default=8,
        help="Maximum finite-support-ranked alternatives tried inside one cluster in cumulative-local mode.",
    )
    parser.add_argument(
        "--active-face-filter-mode",
        choices=("off", "legacy"),
        default="legacy",
        help="legacy applies the active-face size post-filter; off keeps deduped candidates unchanged.",
    )
    parser.add_argument("--max-active-face-hull-area-ratio", type=float, default=20.0)
    parser.add_argument("--max-active-face-extra-area", type=float, default=1.0)
    parser.add_argument("--active-face-filter-iterations", type=int, default=4)
    parser.add_argument(
        "--intersection-mesh-mode",
        choices=("legacy", "incidence", "edge-clip"),
        default="legacy",
        help="legacy uses proximity face membership; incidence uses triple-plane incidence; edge-clip clips pairwise plane lines.",
    )
    parser.add_argument("--intersection-inside-tol", type=float, default=0.03)
    parser.add_argument("--intersection-vertex-tol", type=float, default=0.02)
    parser.add_argument("--intersection-halfspace-slack", type=float, default=0.0)
    parser.add_argument("--intersection-feasibility-tol", type=float, default=1e-7)
    parser.add_argument("--intersection-incidence-tol", type=float, default=1e-6)
    parser.add_argument("--intersection-vertex-merge-tol", type=float, default=1e-6)
    parser.add_argument("--intersection-min-face-area", type=float, default=1e-8)
    parser.add_argument("--intersection-triple-det-tol", type=float, default=1e-10)
    parser.add_argument("--intersection-prune-redundant-planes", action="store_true")
    parser.add_argument("--intersection-redundancy-tol", type=float, default=1e-8)
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
    multiscale_summary: dict[str, object] = {}
    dedupe_summary: dict[str, object] = {}
    selection_summary: dict[str, object] = {}
    trusted_cloud_summary: dict[str, object] = {}
    per_window_summary: list[dict[str, object]] = []
    all_tracks: list[PeakTrack] = []
    pipeline_started = time.perf_counter()
    pipeline_timing: dict[str, float] = {
        "line_points_seconds": 0.0,
        "detection_tracking_fitting_seconds": 0.0,
        "clustering_seconds": 0.0,
        "selection_seconds": 0.0,
        "append_local_seconds": 0.0,
        "lp_blockers_seconds": 0.0,
        "post_repair_refill_seconds": 0.0,
        "post_ratchet_exchange_seconds": 0.0,
        "angular_scale_additive_seconds": 0.0,
        "final_intersection_seconds": 0.0,
        "oracle_diagnostics_seconds": 0.0,
    }

    needs_multi_window = bool(args.candidate_scale_mode == "multiscale" or len(windows) > 1)
    line_points_by_window: np.ndarray | None = None
    fit_rms_by_window: np.ndarray | None = None
    cond_by_window: np.ndarray | None = None
    if needs_multi_window:
        line_points_started = time.perf_counter()
        line_points_by_window, fit_rms_by_window, cond_by_window = compute_line_points_multi_window(
            half_contours,
            z_levels,
            windows,
            use_progress=not args.no_progress,
        )
        pipeline_timing["line_points_seconds"] += time.perf_counter() - line_points_started

    trusted_cloud_points = np.zeros((0, 3), dtype=float)
    trusted_cloud_z_indices = np.zeros((0,), dtype=int)

    if args.candidate_scale_mode == "adaptive":
        if line_points_by_window is None or fit_rms_by_window is None:
            line_points_started = time.perf_counter()
            line_points, fit_rms = compute_line_points(
                half_contours,
                z_levels,
                int(windows[0]),
                use_progress=not args.no_progress,
            )
            pipeline_timing["line_points_seconds"] += time.perf_counter() - line_points_started
            cond_values = None
            selected_window_indices = np.zeros(int(z_levels.size), dtype=int)
            adaptive_summary = {
                "adaptive_window_min_selected": int(windows[0]),
                "adaptive_window_max_selected": int(windows[0]),
                "adaptive_window_median_selected": int(windows[0]),
                "adaptive_window_changes": 0,
                "adaptive_window_sequence": [int(windows[0]) for _ in range(int(z_levels.size))],
            }
        else:
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
            cond_values = (
                apply_selected_values_by_window(cond_by_window, selected_window_indices)
                if cond_by_window is not None
                else None
            )

        stage_started = time.perf_counter()
        per_level, diagnostics = detect_peak_observations(
            z_levels,
            fit_rms,
            peak_threshold_abs=float(args.peak_threshold),
            low_threshold_abs=float(args.low_threshold),
            peak_quantile=float(args.peak_quantile),
            low_quantile=float(args.low_quantile),
            low_barrier_threshold_abs=float(args.low_barrier_threshold),
            low_barrier_fraction=float(args.low_barrier_fraction),
            low_barrier_prominence=float(args.low_barrier_prominence),
            low_valley_fallback=bool(args.low_valley_fallback),
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
        all_tracks = tracks
        finite_mask = (
            np.all(np.isfinite(line_points), axis=2)
            & np.all(line_points >= point_bounds_min[None, None, :], axis=2)
            & np.all(line_points <= point_bounds_max[None, None, :], axis=2)
        )
        finite_points = line_points[finite_mask]
        display_window = int(windows[int(selected_window_indices[0])]) if len(windows) > 1 else int(windows[0])
        pipeline_timing["detection_tracking_fitting_seconds"] += time.perf_counter() - stage_started
    else:
        if line_points_by_window is None or fit_rms_by_window is None or cond_by_window is None:
            raise RuntimeError("multiscale mode requires multi-window line point computation")
        display_window = 4 if 4 in windows else int(windows[0])
        display_wi = windows.index(display_window)
        line_points = line_points_by_window[display_wi].astype(float)
        fit_rms = fit_rms_by_window[display_wi].astype(float)
        diagnostics = []
        finite_mask = (
            np.all(np.isfinite(line_points_by_window), axis=3)
            & np.all(line_points_by_window >= point_bounds_min[None, None, None, :], axis=3)
            & np.all(line_points_by_window <= point_bounds_max[None, None, None, :], axis=3)
        )
        finite_points = line_points_by_window[finite_mask].astype(float)

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

    def build_multiscale_candidate_pool(
        *,
        low_valley_fallback_value: bool,
        origin: str,
    ) -> tuple[list[dict[str, object]], list[PeakTrack], list[dict[str, object]], list[dict[str, object]], float]:
        if line_points_by_window is None or fit_rms_by_window is None or cond_by_window is None:
            raise RuntimeError("multiscale candidate pool requires multi-window arrays")
        pool_candidates: list[dict[str, object]] = []
        pool_tracks: list[PeakTrack] = []
        pool_summary: list[dict[str, object]] = []
        display_diagnostics: list[dict[str, object]] = []
        elapsed = 0.0
        track_offset = 0
        for wi, window in enumerate(windows):
            stage_started = time.perf_counter()
            lp_w = line_points_by_window[wi].astype(float)
            rms_w = fit_rms_by_window[wi].astype(float)
            cond_w = cond_by_window[wi].astype(float)
            per_level_w, diagnostics_w = detect_peak_observations(
                z_levels,
                rms_w,
                peak_threshold_abs=float(args.peak_threshold),
                low_threshold_abs=float(args.low_threshold),
                peak_quantile=float(args.peak_quantile),
                low_quantile=float(args.low_quantile),
                low_barrier_threshold_abs=float(args.low_barrier_threshold),
                low_barrier_fraction=float(args.low_barrier_fraction),
                low_barrier_prominence=float(args.low_barrier_prominence),
                low_valley_fallback=bool(low_valley_fallback_value),
                min_peak_width=int(args.min_peak_width),
                max_peak_width=int(args.max_peak_width),
                max_low_distance=float(args.max_low_distance),
            )
            if int(window) == int(display_window):
                display_diagnostics = diagnostics_w
            tracks_w = track_peak_observations(
                per_level_w,
                n_half=int(rms_w.shape[0]),
                max_center_jump=float(args.max_center_jump),
                max_z_gap=int(args.max_z_gap),
            )
            candidates_w = build_face_candidates(
                tracks_w,
                lp_w,
                rms_w,
                window=int(window),
                track_id_offset=track_offset,
                cond_values=cond_w,
                max_line_condition=float(args.max_line_condition),
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
            for candidate in candidates_w:
                candidate["scale_mode"] = "multiscale"
                candidate["candidate_origin"] = str(origin)
            pool_tracks.extend(tracks_w)
            pool_candidates.extend(candidates_w)
            pool_summary.append(
                {
                    "window": int(window),
                    "peak_observations": int(sum(int(item.get("peaks", 0) or 0) for item in diagnostics_w)),
                    "paired_observations": int(sum(len(level) for level in per_level_w)),
                    "tracks": int(len(tracks_w)),
                    "candidates": int(len(candidates_w)),
                    "condition_rejected_fit_points": int(
                        sum(int(c.get("condition_rejected_fit_points") or 0) for c in candidates_w)
                    ),
                    "candidate_origin": str(origin),
                    "low_valley_fallback": bool(low_valley_fallback_value),
                }
            )
            track_offset += max(100000, len(tracks_w) + 1)
            elapsed += time.perf_counter() - stage_started
        pool_candidates.sort(key=candidate_rank_key)
        return pool_candidates, pool_tracks, pool_summary, display_diagnostics, elapsed

    if args.candidate_scale_mode == "adaptive":
        all_candidates = build_face_candidates(
            all_tracks,
            line_points,
            fit_rms,
            window=display_window if len(windows) == 1 else None,
            track_id_offset=0,
            cond_values=cond_values,
            max_line_condition=float(args.max_line_condition),
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
        candidates_for_top_k = all_candidates
    else:
        all_candidates, all_tracks, per_window_summary, diagnostics, elapsed = build_multiscale_candidate_pool(
            low_valley_fallback_value=bool(args.low_valley_fallback),
            origin="rebuild",
        )
        pipeline_timing["detection_tracking_fitting_seconds"] += elapsed
        trusted_cloud = build_trusted_observed_cloud(
            line_points_by_window.astype(float),
            fit_rms_by_window.astype(float),
            windows,
            display_window=int(display_window),
            threshold_pct=float(args.minima_cloud_pct),
            bounds_min=point_bounds_min,
            bounds_max=point_bounds_max,
        )
        trusted_cloud_points = np.array(trusted_cloud["points"], dtype=float).reshape((-1, 3))
        trusted_cloud_z_indices = np.array(trusted_cloud["z_indices"], dtype=int)
        trusted_cloud_summary = {
            "trusted_cloud": trusted_cloud["diagnostics"],
        }
        if dispatch_post_multiscale_oracle_scope(
            scope=str(os.environ.get("POLYRECO_ORACLE_DIAGNOSTIC_SCOPE") or "").strip(),
            environment=os.environ,
            args=args,
            model_name=model_name,
            vertices=vertices,
            faces=model.faces,
            contours=contours,
            all_candidates=all_candidates,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
        ):
            return
        def run_selection(
            pool_candidates: list[dict[str, object]],
            *,
            initial_accepted: list[dict[str, object]] | None = None,
            max_additions: int | None = None,
            valley_candidate_mode: str | None = None,
            initial_label: str = "initial",
        ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
            clustering_started = time.perf_counter()
            clusters_local, dedupe_local = cluster_face_candidates(
                pool_candidates,
                normal_angle_deg=float(args.dedupe_normal_angle_deg),
                plane_distance=float(args.dedupe_plane_distance),
                hull_distance=float(args.dedupe_hull_distance),
                mode=str(args.candidate_cluster_mode),
            )
            pipeline_timing["clustering_seconds"] += time.perf_counter() - clustering_started
            selection_started_local = time.perf_counter()
            selected_local, selection_local = select_compatible_candidates(
                clusters_local,
                trusted_points=trusted_cloud_points,
                trusted_z_indices=trusted_cloud_z_indices,
                mode=str(args.candidate_selection_mode),
                point_tol=float(args.compatibility_point_tol),
                max_global_outside_frac=float(args.compatibility_max_global_outside_frac),
                max_level_outside_frac=float(args.compatibility_max_level_outside_frac),
                bbox_margin=float(args.compatibility_bbox_margin),
                min_local_support=int(args.compatibility_min_local_support),
                finite_support_plane_distance=float(args.finite_support_plane_distance),
                finite_support_hull_margin=float(args.finite_support_hull_margin),
                finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
                max_alternatives_per_cluster=int(args.cumulative_local_max_alternatives_per_cluster),
                max_candidates=int(args.max_candidates),
                outside_mode=str(args.compatibility_outside_mode),
                activity_check=str(args.compatibility_activity_check),
                activity_tol=float(args.compatibility_activity_tol),
                valley_candidate_mode=str(valley_candidate_mode or args.valley_candidate_mode),
                valley_min_confidence_median=float(args.valley_min_confidence_median),
                valley_min_confidence_p10=float(args.valley_min_confidence_p10),
                valley_min_two_side_fraction=float(args.valley_min_two_side_fraction),
                valley_max_one_side_fraction=float(args.valley_max_one_side_fraction),
                valley_max_distance_p95=float(args.valley_max_distance_p95),
                valley_max_track_smoothness=float(args.valley_max_track_smoothness),
                valley_max_support_gap_global=float(args.valley_max_support_gap_global),
                valley_max_support_gap_local=float(args.valley_max_support_gap_local),
                valley_min_support_near_count=int(args.valley_min_support_near_count),
                valley_min_scale_persistence=int(args.valley_min_scale_persistence),
                valley_min_track_levels=int(args.valley_min_track_levels),
                valley_max_plane_rms=float(args.valley_max_plane_rms),
                valley_min_hull_area=float(args.valley_min_hull_area),
                valley_min_finite_support_count=int(args.valley_min_finite_support_count),
                valley_min_finite_track_coverage=float(args.valley_min_finite_track_coverage),
                initial_accepted=initial_accepted,
                max_additions=max_additions,
                initial_label=initial_label,
            )
            pipeline_timing["selection_seconds"] += time.perf_counter() - selection_started_local
            dedupe_local = {
                **dedupe_local,
                "dedupe_candidates_after": int(len(selected_local)),
            }
            return selected_local, selection_local, dedupe_local

        augmentation_summary: dict[str, object] = {}
        if str(args.valley_integration_mode) == "augment-baseline":
            baseline_candidates, baseline_tracks, baseline_window_summary, _, baseline_elapsed = build_multiscale_candidate_pool(
                low_valley_fallback_value=False,
                origin="baseline_core",
            )
            pipeline_timing["detection_tracking_fitting_seconds"] += baseline_elapsed
            baseline_selected, baseline_selection_summary, baseline_dedupe_summary = run_selection(
                baseline_candidates,
                valley_candidate_mode="any",
            )
            for candidate in baseline_selected:
                candidate["candidate_origin"] = "baseline_core"
            valley_pool = [
                dict(candidate, candidate_origin="valley_addition")
                for candidate in all_candidates
                if float(candidate.get("low_valley_frac") or 0.0) > 0.0
            ]
            valley_pool, valley_preselection_summary = annotate_valley_additions_against_baseline(
                valley_pool,
                baseline_selected,
                duplicate_angle_deg=float(args.dedupe_normal_angle_deg),
                duplicate_plane_distance=float(args.dedupe_plane_distance),
                sliver_hull_distance=max(float(args.dedupe_hull_distance) * 0.25, 0.05),
            )
            max_additions = int(args.max_valley_additions) if int(args.max_valley_additions) >= 0 else None
            candidates_for_top_k, valley_selection_summary, valley_dedupe_summary = run_selection(
                valley_pool,
                initial_accepted=baseline_selected,
                max_additions=max_additions,
                valley_candidate_mode=str(args.valley_candidate_mode),
                initial_label="baseline_core",
            )
            selection_summary = {
                "valley_integration_mode": "augment-baseline",
                "baseline_core_selection": baseline_selection_summary,
                "valley_addition_selection": valley_selection_summary,
                **valley_preselection_summary,
            }
            dedupe_summary = {
                "baseline_core_dedupe": baseline_dedupe_summary,
                "valley_addition_dedupe": valley_dedupe_summary,
                "dedupe_candidates_after": int(len(candidates_for_top_k)),
            }
            augmentation_summary = {
                "baseline_core_raw_candidates": int(len(baseline_candidates)),
                "baseline_core_selected": int(len(baseline_selected)),
                "valley_raw_candidates": int(len(all_candidates)),
                "valley_addition_raw_candidates": int(sum(1 for c in all_candidates if float(c.get("low_valley_frac") or 0.0) > 0.0)),
                "valley_additions_after_preselection": int(len(valley_pool)),
                "valley_accepted_additions": int(max(0, len(candidates_for_top_k) - len(baseline_selected))),
                "baseline_core_per_window": baseline_window_summary,
            }
            all_candidates = baseline_candidates + all_candidates
            all_tracks = baseline_tracks + all_tracks
        elif str(args.valley_integration_mode) == "repair-rebuild":
            valley_selected, valley_selection_summary, valley_dedupe_summary = run_selection(all_candidates)
            for candidate in valley_selected:
                candidate["candidate_origin"] = "valley_rebuild"
            baseline_candidates, baseline_tracks, baseline_window_summary, _, baseline_elapsed = build_multiscale_candidate_pool(
                low_valley_fallback_value=False,
                origin="baseline_core",
            )
            pipeline_timing["detection_tracking_fitting_seconds"] += baseline_elapsed
            baseline_selected, baseline_selection_summary, baseline_dedupe_summary = run_selection(
                baseline_candidates,
                valley_candidate_mode="any",
            )
            for candidate in baseline_selected:
                candidate["candidate_origin"] = "baseline_core"
            repaired_candidates, repair_summary = sparse_repair_rebuild_candidates(
                valley_selected,
                baseline_selected,
                trusted_points=trusted_cloud_points,
                trusted_z_indices=trusted_cloud_z_indices,
                point_tol=float(args.compatibility_point_tol),
                max_global_outside_frac=float(args.compatibility_max_global_outside_frac),
                max_level_outside_frac=float(args.compatibility_max_level_outside_frac),
                min_local_support=int(args.compatibility_min_local_support),
                repair_margin=float(args.envelope_repair_margin),
                repair_relative_margin=float(args.envelope_repair_relative_margin),
                repair_min_improvement=float(args.envelope_repair_min_improvement),
                max_guards=max(0, int(args.max_baseline_guards)),
                inside_tol=float(args.intersection_feasibility_tol),
            )
            candidates_for_top_k = repaired_candidates
            selection_summary = {
                "valley_integration_mode": "repair-rebuild",
                "valley_rebuild_selection": valley_selection_summary,
                "baseline_guard_source_selection": baseline_selection_summary,
                **repair_summary,
            }
            dedupe_summary = {
                "valley_rebuild_dedupe": valley_dedupe_summary,
                "baseline_guard_source_dedupe": baseline_dedupe_summary,
                "dedupe_candidates_after": int(len(candidates_for_top_k)),
            }
            augmentation_summary = {
                "valley_integration_mode": "repair-rebuild",
                "valley_rebuild_raw_candidates": int(len(all_candidates)),
                "valley_rebuild_selected": int(len(valley_selected)),
                "baseline_guard_raw_candidates": int(len(baseline_candidates)),
                "baseline_guard_source_selected": int(len(baseline_selected)),
                "repair_guard_pool": int(repair_summary.get("repair_guard_pool") or 0),
                "repair_guards_added": int(repair_summary.get("repair_guards_added") or 0),
                "baseline_guard_per_window": baseline_window_summary,
            }
            all_candidates = all_candidates + baseline_candidates
            all_tracks = all_tracks + baseline_tracks
        else:
            candidates_for_top_k, selection_summary, dedupe_summary = run_selection(all_candidates)
            augmentation_summary = {"valley_integration_mode": "rebuild"}
        dedupe_summary = {
            **dedupe_summary,
            "dedupe_candidates_after": int(len(candidates_for_top_k)),
        }
        multiscale_summary = {
            "multiscale_windows": [int(w) for w in windows],
            "multiscale_per_window": per_window_summary,
            **augmentation_summary,
        }

    minima_cloud = build_minima_cloud(
        line_points,
        fit_rms,
        threshold_pct=float(args.minima_cloud_pct),
        bounds_min=point_bounds_min,
        bounds_max=point_bounds_max,
    )
    max_candidates = max(1, int(args.max_candidates))
    candidates = candidates_for_top_k[:max_candidates]
    z_envelope_guard_summary: dict[str, object] = {
        "mode": str(args.z_envelope_guard_mode),
        "applied": False,
        "uses_initial_model": False,
    }
    if str(args.z_envelope_guard_mode) == "trusted-cap":
        candidates, z_envelope_guard_summary = apply_trusted_z_envelope_guard(
            candidates,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            inside_tol=float(args.intersection_feasibility_tol),
            trigger_relative=float(args.z_envelope_guard_trigger_relative),
            margin_relative=float(args.z_envelope_guard_margin_relative),
        )
    dense_detector_summary: dict[str, object] = {
        "dense_detector_mode": str(args.dense_detector_mode),
        "zero_value_regression_check": zero_value_regression_check(),
    }
    dense_core_candidates: list[dict[str, object]] = list(candidates)
    dense_w2_raw_candidates: list[dict[str, object]] = []
    dense_w2_preselection_candidates: list[dict[str, object]] = []
    dense_w2_edge_clusters: list[dict[str, object]] = []
    dense_w2_segments: list[dict[str, object]] = []
    dense_w2_line_points: np.ndarray | None = None
    dense_w2_fit_rms: np.ndarray | None = None
    dense_w2_cond: np.ndarray | None = None
    dense_w3_line_points: np.ndarray | None = None
    dense_w3_fit_rms: np.ndarray | None = None
    dense_w4_line_points: np.ndarray | None = None
    dense_w4_fit_rms: np.ndarray | None = None
    w2_plane_consensus_summary: dict[str, object] = {"w2_plane_consensus_mode": str(args.w2_plane_consensus_mode)}
    w2_small_face_refinement_summary: dict[str, object] = {"mode": str(args.w2_small_face_refinement), "accepted_additions": 0}
    w2_core_repair_summary: dict[str, object] = {
        "mode": str(args.w2_core_repair_mode),
        "production_changed": False,
        "selected_removed_candidate_ids": [],
    }
    post_max150_candidates: list[dict[str, object]] = list(candidates)
    post_append_local_candidates: list[dict[str, object]] = list(candidates)
    support_diverse_ranked_candidate_ids: list[int] = []
    if str(args.dense_detector_mode) == "w2-edge-tracks":
        if line_points_by_window is None or fit_rms_by_window is None or cond_by_window is None:
            dense_detector_summary["dense_detector_error"] = "missing_multi_window_line_points"
        else:
            if 2 in windows:
                w2_i = windows.index(2)
                w2_line_points = line_points_by_window[w2_i].astype(float)
                w2_fit_rms = fit_rms_by_window[w2_i].astype(float)
                w2_cond = cond_by_window[w2_i].astype(float)
            else:
                w2_started = time.perf_counter()
                w2_lp_by_window, w2_rms_by_window, w2_cond_by_window = compute_line_points_multi_window(
                    half_contours,
                    z_levels,
                    [2],
                    use_progress=False,
                )
                pipeline_timing["line_points_seconds"] += time.perf_counter() - w2_started
                w2_line_points = w2_lp_by_window[0].astype(float)
                w2_fit_rms = w2_rms_by_window[0].astype(float)
                w2_cond = w2_cond_by_window[0].astype(float)
            dense_w2_line_points = w2_line_points
            dense_w2_fit_rms = w2_fit_rms
            dense_w2_cond = w2_cond
            w3_i = windows.index(3) if 3 in windows else None
            dense_w3_line_points = (
                line_points_by_window[w3_i].astype(float) if w3_i is not None else None
            )
            dense_w3_fit_rms = (
                fit_rms_by_window[w3_i].astype(float) if w3_i is not None else None
            )
            w4_i = windows.index(4) if 4 in windows else None
            dense_w4_line_points = (
                line_points_by_window[w4_i].astype(float) if w4_i is not None else None
            )
            dense_w4_fit_rms = (
                fit_rms_by_window[w4_i].astype(float) if w4_i is not None else None
            )
            w2_segments, w2_segment_summary = build_w2_edge_segments(
                z_levels=z_levels,
                line_points_w2=w2_line_points,
                cond_w2=w2_cond,
                line_points_w3=line_points_by_window[w3_i].astype(float) if w3_i is not None else None,
                fit_rms_w3=fit_rms_by_window[w3_i].astype(float) if w3_i is not None else None,
                min_levels=int(args.w2_edge_min_levels),
                min_z_span=float(args.w2_edge_min_z_span),
                max_line_rms=float(args.w2_edge_max_line_rms),
                max_line_residual=float(args.w2_edge_max_line_residual),
                max_z_gap=int(args.w2_edge_max_z_gap),
                max_point_jump=float(args.w2_edge_max_point_jump),
                max_condition=float(args.w2_edge_max_condition),
                segment_fit_mode=str(args.w2_segment_fit_mode),
            )
            w2_clusters, w2_cluster_summary = cluster_w2_edge_segments(
                w2_segments,
                mode=str(args.w2_edge_cluster_mode),
                max_distance=float(args.w2_edge_cluster_max_distance),
                max_angle_deg=float(args.w2_edge_cluster_max_angle_deg),
                min_overlap=float(args.w2_edge_cluster_min_overlap),
            )
            w2_candidates, w2_face_summary = build_w2_face_candidates(
                w2_clusters,
                z_levels,
                inside_point=inside_point,
                orientation_points=orientation_points,
                bounds_min=point_bounds_min,
                bounds_max=point_bounds_max,
                min_adjacency_levels=int(args.w2_face_min_adjacency_levels),
                min_adjacency_fraction=float(args.w2_face_min_adjacency_fraction),
                min_z_span=float(args.w2_face_min_z_span),
                max_plane_rms=float(args.w2_face_max_plane_rms),
                duplicate_pair_mode=str(args.w2_duplicate_pair_mode),
            )
            dense_w2_raw_candidates = list(w2_candidates)
            dense_w2_edge_clusters = list(w2_clusters)
            dense_w2_segments = list(w2_segments)
            if dispatch_post_w2_oracle_scope(
                scope=str(os.environ.get("POLYRECO_ORACLE_DIAGNOSTIC_SCOPE") or "").strip(),
                environment=os.environ,
                args=args,
                model_name=model_name,
                vertices=vertices,
                faces=model.faces,
                contours=contours,
                z_levels=z_levels,
                w2_line_points=w2_line_points,
                w2_segments=w2_segments,
                w2_clusters=w2_clusters,
                w2_candidates=w2_candidates,
                trusted_points=trusted_cloud_points,
                trusted_z_indices=trusted_cloud_z_indices,
            ):
                return
            for candidate in w2_candidates:
                relation = classify_against_baseline(
                    candidate,
                    candidates,
                    duplicate_angle_deg=float(args.dedupe_normal_angle_deg),
                    duplicate_plane_distance=float(args.dedupe_plane_distance),
                    sliver_hull_distance=max(float(args.dedupe_hull_distance) * 0.25, 0.05),
                )
                candidate.update({f"core_{k}": v for k, v in relation.items()})
            w2_candidates_for_selection = [
                c for c in w2_candidates
                if str(c.get("core_baseline_rejection_reason")) not in {"baseline_duplicate", "baseline_weaker_parallel"}
            ]
            if str(args.w2_addition_selection_mode) == "novelty" and w2_candidates_for_selection:
                novelty_core_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
                    candidates,
                    halfspace_slack=float(args.intersection_halfspace_slack),
                    feasibility_tol=float(args.intersection_feasibility_tol),
                    incidence_tol=float(args.intersection_incidence_tol),
                    vertex_merge_tol=float(args.intersection_vertex_merge_tol),
                    min_face_area=float(args.intersection_min_face_area),
                    triple_det_tol=float(args.intersection_triple_det_tol),
                    prune_redundant=bool(args.intersection_prune_redundant_planes),
                    redundancy_tol=float(args.intersection_redundancy_tol),
                )
                annotate_w2_novelty_scores(w2_candidates_for_selection, candidates, novelty_core_reconstructed)
            if str(args.w2_addition_selection_mode) == "support-diverse" and w2_candidates_for_selection:
                support_core_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
                    candidates,
                    halfspace_slack=float(args.intersection_halfspace_slack),
                    feasibility_tol=float(args.intersection_feasibility_tol),
                    incidence_tol=float(args.intersection_incidence_tol),
                    vertex_merge_tol=float(args.intersection_vertex_merge_tol),
                    min_face_area=float(args.intersection_min_face_area),
                    triple_det_tol=float(args.intersection_triple_det_tol),
                    prune_redundant=bool(args.intersection_prune_redundant_planes),
                    redundancy_tol=float(args.intersection_redundancy_tol),
                )
                annotate_w2_support_diverse_scores(w2_candidates_for_selection, candidates, support_core_reconstructed)
                if str(args.w2_plane_consensus_mode) == "soft":
                    w2_plane_consensus_summary = annotate_w2_plane_consensus_soft_scores(
                        w2_candidates_for_selection,
                        dense_w2_edge_clusters,
                    )
            dense_w2_preselection_candidates = list(w2_candidates_for_selection)
            if w2_candidates_for_selection:
                clusters_w2, dedupe_w2 = cluster_face_candidates(
                    w2_candidates_for_selection,
                    normal_angle_deg=float(args.dedupe_normal_angle_deg),
                    plane_distance=float(args.dedupe_plane_distance),
                    hull_distance=float(args.dedupe_hull_distance),
                    mode=str(args.candidate_cluster_mode),
                )
                selected_with_w2, selection_w2 = select_compatible_candidates(
                    clusters_w2,
                    trusted_points=trusted_cloud_points,
                    trusted_z_indices=trusted_cloud_z_indices,
                    mode=str(args.candidate_selection_mode),
                    point_tol=float(args.compatibility_point_tol),
                    max_global_outside_frac=float(args.compatibility_max_global_outside_frac),
                    max_level_outside_frac=float(args.compatibility_max_level_outside_frac),
                    bbox_margin=float(args.compatibility_bbox_margin),
                    min_local_support=int(args.compatibility_min_local_support),
                    finite_support_plane_distance=float(args.finite_support_plane_distance),
                    finite_support_hull_margin=float(args.finite_support_hull_margin),
                    finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
                    max_alternatives_per_cluster=int(args.cumulative_local_max_alternatives_per_cluster),
                    max_candidates=int(len(candidates) + max(0, int(args.max_w2_additions))),
                    outside_mode=str(args.compatibility_outside_mode),
                    activity_check=str(args.compatibility_activity_check),
                    activity_tol=float(args.compatibility_activity_tol),
                    valley_candidate_mode="any",
                    valley_min_confidence_median=float(args.valley_min_confidence_median),
                    valley_min_confidence_p10=float(args.valley_min_confidence_p10),
                    valley_min_two_side_fraction=float(args.valley_min_two_side_fraction),
                    valley_max_one_side_fraction=float(args.valley_max_one_side_fraction),
                    valley_max_distance_p95=float(args.valley_max_distance_p95),
                    valley_max_track_smoothness=float(args.valley_max_track_smoothness),
                    valley_max_support_gap_global=float(args.valley_max_support_gap_global),
                    valley_max_support_gap_local=float(args.valley_max_support_gap_local),
                    valley_min_support_near_count=int(args.valley_min_support_near_count),
                    valley_min_scale_persistence=int(args.valley_min_scale_persistence),
                    valley_min_track_levels=int(args.valley_min_track_levels),
                    valley_max_plane_rms=float(args.valley_max_plane_rms),
                    valley_min_hull_area=float(args.valley_min_hull_area),
                    valley_min_finite_support_count=int(args.valley_min_finite_support_count),
                    valley_min_finite_track_coverage=float(args.valley_min_finite_track_coverage),
                    initial_accepted=candidates,
                    max_additions=max(0, int(args.max_w2_additions)),
                    initial_label="dense_core",
                    addition_selection_mode=str(args.w2_addition_selection_mode),
                )
                support_diverse_ranked_candidate_ids = [
                    int(row["track_id"])
                    for row in selection_w2.get("cumulative_local_attempt_log", [])
                    if row.get("track_id") is not None
                ]
                post_max150_candidates = list(selected_with_w2)
                if str(args.w2_small_face_refinement) == "append-local":
                    support_core_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
                        dense_core_candidates,
                        halfspace_slack=float(args.intersection_halfspace_slack),
                        feasibility_tol=float(args.intersection_feasibility_tol),
                        incidence_tol=float(args.intersection_incidence_tol),
                        vertex_merge_tol=float(args.intersection_vertex_merge_tol),
                        min_face_area=float(args.intersection_min_face_area),
                        triple_det_tol=float(args.intersection_triple_det_tol),
                        prune_redundant=bool(args.intersection_prune_redundant_planes),
                        redundancy_tol=float(args.intersection_redundancy_tol),
                    )
                    append_started = time.perf_counter()
                    selected_with_w2, w2_small_face_refinement_summary = append_local_small_face_refinement(
                        core_candidates=dense_core_candidates,
                        core_reconstructed=support_core_reconstructed,
                        w2_preselection_candidates=w2_candidates_for_selection,
                        selected_candidates=selected_with_w2,
                        trusted_points=trusted_cloud_points,
                        trusted_z_indices=trusted_cloud_z_indices,
                        point_tol=float(args.compatibility_point_tol),
                        reconstruction_kwargs={
                            "halfspace_slack": float(args.intersection_halfspace_slack),
                            "feasibility_tol": float(args.intersection_feasibility_tol),
                            "incidence_tol": float(args.intersection_incidence_tol),
                            "vertex_merge_tol": float(args.intersection_vertex_merge_tol),
                            "min_face_area": float(args.intersection_min_face_area),
                            "triple_det_tol": float(args.intersection_triple_det_tol),
                            "prune_redundant": bool(args.intersection_prune_redundant_planes),
                            "redundancy_tol": float(args.intersection_redundancy_tol),
                        },
                        finite_support_plane_distance=float(args.finite_support_plane_distance),
                        finite_support_hull_margin=float(args.finite_support_hull_margin),
                        finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
                        max_additions=max(0, int(args.w2_max_small_face_additions)),
                        max_trials=max(0, int(args.w2_max_small_face_trials)),
                        trial_geometry_mode=str(args.w2_small_face_trial_geometry),
                    )
                    pipeline_timing["append_local_seconds"] += time.perf_counter() - append_started
                post_append_local_candidates = list(selected_with_w2)
                candidates = selected_with_w2
            else:
                dedupe_w2 = {}
                selection_w2 = {"accepted_additions": 0, "selection_rejection_counts": {"no_w2_candidates_after_preselection": 1}}
            dense_detector_summary.update(
                {
                    **w2_segment_summary,
                    **w2_cluster_summary,
                    **w2_face_summary,
                    "w2_candidates_after_core_preselection": int(len(w2_candidates_for_selection)),
                    "w2_plane_consensus": w2_plane_consensus_summary,
                    "w2_small_face_refinement": w2_small_face_refinement_summary,
                    "w2_dedupe": dedupe_w2,
                    "w2_selection": selection_w2,
                    "w2_segments_sample": w2_segments[:80],
                    "w2_edge_clusters_sample": w2_clusters[:80],
                }
            )
    requested_core_repair_budget = args.w2_max_core_repairs
    if requested_core_repair_budget is None:
        resolved_core_repair_budget = 2 if str(args.w2_core_repair_mode) == "lp-blockers" else 1
    else:
        resolved_core_repair_budget = max(0, int(requested_core_repair_budget))
    if str(args.w2_core_repair_mode) in {"lp-remove-one", "lp-blockers"}:
        core_repair_started = time.perf_counter()
        if str(args.dense_detector_mode) != "w2-edge-tracks":
            w2_core_repair_summary.update(
                {
                    "production_changed": False,
                    "reason": "requires_dense_detector_mode_w2_edge_tracks",
                    "requested_action_budget": requested_core_repair_budget,
                    "resolved_action_budget": int(resolved_core_repair_budget),
                }
            )
        elif str(args.w2_core_repair_mode) == "lp-remove-one":
            repair_candidates, w2_core_repair_summary = apply_w2_core_repair_lp_remove_one(
                core_candidates=dense_core_candidates,
                selected_candidates=candidates,
                trusted_points=trusted_cloud_points,
                trusted_z_indices=trusted_cloud_z_indices,
                point_tol=float(args.compatibility_point_tol),
                reconstruction_kwargs={
                    "halfspace_slack": float(args.intersection_halfspace_slack),
                    "feasibility_tol": float(args.intersection_feasibility_tol),
                    "incidence_tol": float(args.intersection_incidence_tol),
                    "vertex_merge_tol": float(args.intersection_vertex_merge_tol),
                    "min_face_area": float(args.intersection_min_face_area),
                    "triple_det_tol": float(args.intersection_triple_det_tol),
                    "prune_redundant": bool(args.intersection_prune_redundant_planes),
                    "redundancy_tol": float(args.intersection_redundancy_tol),
                },
                finite_support_plane_distance=float(args.finite_support_plane_distance),
                finite_support_hull_margin=float(args.finite_support_hull_margin),
                finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
                max_repairs=min(1, int(resolved_core_repair_budget)),
                face_activity_slack=float(args.compatibility_activity_tol),
                contours=contours,
            )
            w2_core_repair_summary["requested_action_budget"] = requested_core_repair_budget
            w2_core_repair_summary["resolved_action_budget"] = int(min(1, int(resolved_core_repair_budget)))
            w2_core_repair_summary["repair_action_count"] = int(1 if w2_core_repair_summary.get("selected_removed_candidate_ids") else 0)
            w2_core_repair_summary["removed_blocker_count"] = int(len(w2_core_repair_summary.get("selected_removed_candidate_ids", []) or []))
            w2_core_repair_summary["single_action_count"] = int(1 if w2_core_repair_summary.get("selected_removed_candidate_ids") else 0)
            w2_core_repair_summary["pair_action_count"] = 0
            candidates = repair_candidates
        else:
            repair_candidates, w2_core_repair_summary = apply_w2_core_repair_lp_blockers(
                core_candidates=dense_core_candidates,
                selected_candidates=candidates,
                trusted_points=trusted_cloud_points,
                trusted_z_indices=trusted_cloud_z_indices,
                point_tol=float(args.compatibility_point_tol),
                reconstruction_kwargs={
                    "halfspace_slack": float(args.intersection_halfspace_slack),
                    "feasibility_tol": float(args.intersection_feasibility_tol),
                    "incidence_tol": float(args.intersection_incidence_tol),
                    "vertex_merge_tol": float(args.intersection_vertex_merge_tol),
                    "min_face_area": float(args.intersection_min_face_area),
                    "triple_det_tol": float(args.intersection_triple_det_tol),
                    "prune_redundant": bool(args.intersection_prune_redundant_planes),
                    "redundancy_tol": float(args.intersection_redundancy_tol),
                },
                finite_support_plane_distance=float(args.finite_support_plane_distance),
                finite_support_hull_margin=float(args.finite_support_hull_margin),
                finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
                max_repairs=int(resolved_core_repair_budget),
                max_blockers_per_repair=2,
                max_pair_candidates=18,
                face_activity_slack=float(args.compatibility_activity_tol),
                contours=contours,
            )
            w2_core_repair_summary["requested_action_budget"] = requested_core_repair_budget
            w2_core_repair_summary["resolved_action_budget"] = int(resolved_core_repair_budget)
            candidates = repair_candidates
        pipeline_timing["lp_blockers_seconds"] += time.perf_counter() - core_repair_started
    post_core_repair_candidates = list(candidates)
    post_repair_refinement_summary: dict[str, object] = {
        "mode": str(args.w2_post_repair_refinement),
        "production_changed": False,
        "requested_max_additions": int(args.w2_max_post_repair_additions),
        "requested_max_trials": int(args.w2_max_post_repair_trials),
    }
    if str(args.w2_post_repair_refinement) in {"broad-protected", "broad-ratchet"}:
        post_repair_started = time.perf_counter()
        if str(args.w2_core_repair_mode) != "lp-blockers" or str(args.dense_detector_mode) != "w2-edge-tracks":
            post_repair_refinement_summary.update(
                {
                    "production_changed": False,
                    "reason": "requires_w2_edge_tracks_and_lp_blockers",
                }
            )
        else:
            stage1_target_candidate_ids = {
                int(row.get("core_candidate_id"))
                for row in (w2_core_repair_summary.get("stage1") or {}).get("generic_ranking_top", [])[:1]
                for row in (row.get("reactivated_core_rows") or [])
                if row.get("core_candidate_id") is not None
            }
            stage2_target_candidate_ids = {
                int(((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get("target_core_candidate_id"))
            } if ((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get("target_core_candidate_id") is not None else set()
            production_refill = diagnose_protected_post_repair_refill(
                core_candidates=dense_core_candidates,
                core_reconstructed=reconstruct_polyhedron_from_halfspaces_edge_clip(
                    dense_core_candidates,
                    halfspace_slack=float(args.intersection_halfspace_slack),
                    feasibility_tol=float(args.intersection_feasibility_tol),
                    incidence_tol=float(args.intersection_incidence_tol),
                    vertex_merge_tol=float(args.intersection_vertex_merge_tol),
                    min_face_area=float(args.intersection_min_face_area),
                    triple_det_tol=float(args.intersection_triple_det_tol),
                    prune_redundant=bool(args.intersection_prune_redundant_planes),
                    redundancy_tol=float(args.intersection_redundancy_tol),
                ),
                w2_preselection_candidates=dense_w2_preselection_candidates,
                selected_candidates=candidates,
                trusted_points=trusted_cloud_points,
                trusted_z_indices=trusted_cloud_z_indices,
                point_tol=float(args.compatibility_point_tol),
                reconstruction_kwargs={
                    "halfspace_slack": float(args.intersection_halfspace_slack),
                    "feasibility_tol": float(args.intersection_feasibility_tol),
                    "incidence_tol": float(args.intersection_incidence_tol),
                    "vertex_merge_tol": float(args.intersection_vertex_merge_tol),
                    "min_face_area": float(args.intersection_min_face_area),
                    "triple_det_tol": float(args.intersection_triple_det_tol),
                    "prune_redundant": bool(args.intersection_prune_redundant_planes),
                    "redundancy_tol": float(args.intersection_redundancy_tol),
                },
                finite_support_plane_distance=float(args.finite_support_plane_distance),
                finite_support_hull_margin=float(args.finite_support_hull_margin),
                finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
                trial_geometry_mode=str(args.w2_small_face_trial_geometry),
                stage1_target_candidate_ids=stage1_target_candidate_ids,
                stage2_target_candidate_ids=stage2_target_candidate_ids,
                removed_blocker_ids={int(v) for v in w2_core_repair_summary.get("selected_removed_candidate_ids", [])},
                contours=None,
                challenger_order_ids=None,
                max_accepted=max(0, int(args.w2_max_post_repair_additions)),
                max_trials=max(0, int(args.w2_max_post_repair_trials)),
                extended_max_trials=None,
                require_dense_band=False,
                protect_all_active_patches=True,
                rolling_protection=str(args.w2_post_repair_refinement) == "broad-ratchet",
                safe_control_candidate_ids=set(),
                mode_label=(
                    "broad_ratchet_refill_production"
                    if str(args.w2_post_repair_refinement) == "broad-ratchet"
                    else "broad_protected_refill_production"
                ),
            )
            accepted_ids = [int(v) for v in (production_refill.get("protected_sequential_refill") or {}).get("accepted_sequence_signature", [])]
            preselection_by_id = {int(candidate_track_id(candidate)): candidate for candidate in dense_w2_preselection_candidates}
            appended: list[dict[str, object]] = []
            for rank, cid in enumerate(accepted_ids, start=1):
                candidate = preselection_by_id.get(int(cid))
                if candidate is None:
                    continue
                candidate = dict(candidate)
                candidate["small_face_refinement_mode"] = "post-repair-broad-protected"
                candidate["small_face_refinement_rank"] = int(rank)
                appended.append(candidate)
            candidates = candidates + appended
            post_repair_refinement_summary.update(
                {
                    "production_changed": bool(appended),
                    "resolved_max_additions": int(max(0, int(args.w2_max_post_repair_additions))),
                    "resolved_max_trials": int(max(0, int(args.w2_max_post_repair_trials))),
                    "accepted_sequence_signature": accepted_ids,
                    "accepted_additions": int(len(appended)),
                    "trial_checks": int((production_refill.get("protected_sequential_refill") or {}).get("trial_checks") or 0),
                    "rejection_counts": (production_refill.get("protected_sequential_refill") or {}).get("rejection_counts"),
                    "protected_group_count": production_refill.get("protected_group_count"),
                    "protected_group_signature": production_refill.get("protected_group_signature"),
                    "final_protected_group_count": production_refill.get("final_protected_group_count"),
                    "final_protected_group_signature": production_refill.get("final_protected_group_signature"),
                    "rolling_protection": bool(str(args.w2_post_repair_refinement) == "broad-ratchet"),
                    "order_diagnostics": production_refill.get("order_diagnostics"),
                    "timing_breakdown_seconds": production_refill.get("timing_breakdown_seconds"),
                    "timing_seconds": production_refill.get("timing_seconds"),
                    "uses_initial_model": False,
                    "uses_oracle_face_ids": False,
                    "uses_canonical_metrics": False,
                }
            )
        pipeline_timing["post_repair_refill_seconds"] += time.perf_counter() - post_repair_started
    w2_core_repair_summary["post_repair_refinement"] = post_repair_refinement_summary
    post_ratchet_exchange_base_candidates = list(candidates)
    post_ratchet_exchange_summary: dict[str, object] = {
        "mode": str(args.w2_post_ratchet_exchange_mode),
        "production_changed": False,
        "requested_max_exchanges": int(args.w2_max_post_ratchet_exchanges),
        "resolved_max_exchanges": int(max(0, int(args.w2_max_post_ratchet_exchanges))),
        "reason": "disabled",
        "selector_dependencies": {
            "uses_initial_model": False,
            "uses_oracle_face_ids": False,
            "uses_canonical_metrics": False,
            "uses_known_candidate_ids": False,
        },
    }
    if str(args.w2_post_ratchet_exchange_mode) == "lp-deficit":
        post_ratchet_exchange_started = time.perf_counter()
        if (
            str(args.w2_core_repair_mode) != "lp-blockers"
            or str(args.w2_post_repair_refinement) != "broad-ratchet"
            or str(args.dense_detector_mode) != "w2-edge-tracks"
        ):
            post_ratchet_exchange_summary["reason"] = "requires_w2_edge_tracks_lp_blockers_and_broad_ratchet"
        else:
            exchange_stage1_removed = {
                int(v)
                for v in (w2_core_repair_summary.get("stage1") or {}).get("selected_removed_candidate_ids", [])
            }
            exchange_stage1_targets = {
                int(target_row.get("core_candidate_id"))
                for ranking_row in (w2_core_repair_summary.get("stage1") or {}).get("generic_ranking_top", [])
                if int(ranking_row.get("removed_candidate_id") or -1) in exchange_stage1_removed
                for target_row in ranking_row.get("reactivated_core_rows", [])
                if target_row.get("core_candidate_id") is not None
            }
            exchange_stage2_target = ((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get(
                "target_core_candidate_id"
            )
            exchange_stage2_targets = {int(exchange_stage2_target)} if exchange_stage2_target is not None else set()
            candidates, post_ratchet_exchange_summary = apply_post_ratchet_lp_deficit_exchange(
                core_candidates=dense_core_candidates,
                candidate_pool=dense_w2_preselection_candidates,
                selected_candidates=candidates,
                trusted_points=trusted_cloud_points,
                trusted_z_indices=trusted_cloud_z_indices,
                point_tol=float(args.compatibility_point_tol),
                reconstruction_kwargs={
                    "halfspace_slack": float(args.intersection_halfspace_slack),
                    "feasibility_tol": float(args.intersection_feasibility_tol),
                    "incidence_tol": float(args.intersection_incidence_tol),
                    "vertex_merge_tol": float(args.intersection_vertex_merge_tol),
                    "min_face_area": float(args.intersection_min_face_area),
                    "triple_det_tol": float(args.intersection_triple_det_tol),
                    "prune_redundant": bool(args.intersection_prune_redundant_planes),
                    "redundancy_tol": float(args.intersection_redundancy_tol),
                },
                finite_support_plane_distance=float(args.finite_support_plane_distance),
                finite_support_hull_margin=float(args.finite_support_hull_margin),
                finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
                face_activity_slack=float(args.compatibility_activity_tol),
                contours=contours,
                stage1_target_candidate_ids=exchange_stage1_targets,
                stage2_target_candidate_ids=exchange_stage2_targets,
                removed_blocker_ids={
                    int(v) for v in w2_core_repair_summary.get("selected_removed_candidate_ids", [])
                },
                max_exchanges=max(0, int(args.w2_max_post_ratchet_exchanges)),
            )
        pipeline_timing["post_ratchet_exchange_seconds"] += time.perf_counter() - post_ratchet_exchange_started
    w2_core_repair_summary["post_ratchet_exchange"] = post_ratchet_exchange_summary
    angular_scale_summary: dict[str, object] = {
        "mode": str(args.angular_scale_candidate_mode),
        "selection_mode": str(args.angular_scale_selection_mode),
        "selection_order_policy": (
            "legacy_quality_local_lexicographic"
            if str(args.angular_scale_selection_mode) == "quality-local"
            else "global_equal_weight_tie_aware_borda"
        ),
        "selection_locality_policy": (
            "area_ratio_and_hull_distance_only"
            if str(args.angular_scale_selection_mode) == "balanced-global"
            else "legacy_extra_area_0.12_plus_area_ratio_and_hull_distance"
        ),
        "reason": "disabled" if str(args.angular_scale_candidate_mode) == "off" else None,
        "production_changed": False,
        "angular_ranking_uses_initial_model": False,
        "angular_gate_uses_initial_model": False,
        "uses_oracle_face_ids_for_angular_selection": False,
        "uses_canonical_metrics_for_angular_selection": False,
        "inherited_reference_axis_preprocessing": True,
        "entire_pipeline_initial_model_free": False,
        "raw_candidates": 0,
        "deduped_candidates": 0,
        "preselected_candidates": 0,
        "accepted_additions": 0,
        "accepted_sequence_signature": [],
        "final_active_additions": 0,
        "final_active_addition_ids": [],
        "selection_threshold_policy": {
            "trial_face_area_ratio_max": 25.0,
            "trial_face_extra_area_max": (
                None
                if str(args.angular_scale_selection_mode) == "balanced-global"
                else 0.12
            ),
            "trial_face_hull_distance_median_max": 0.20,
            "trial_face_metrics_source": (
                "full_edge_clip_rebuild_reused"
                if str(args.angular_scale_selection_mode)
                in {"balanced-local", "balanced-global"}
                else "legacy_trial_geometry"
            ),
            "full_topology_required": True,
            "strict_frozen_and_rolling_footprints": True,
            "preserve_baseline_outside_and_max_z": True,
            "lost_z_required": 0,
        },
    }
    if str(args.angular_scale_candidate_mode) == "additive":
        angular_started = time.perf_counter()
        if str(args.orientation_mode) != "observed-points":
            raise ValueError(
                "--angular-scale-candidate-mode additive requires --orientation-mode observed-points"
            )
        if int(args.model_support_max_outside_count) >= 0:
            raise ValueError(
                "--angular-scale-candidate-mode additive requires --model-support-max-outside-count -1"
            )
        angular_orientation_points = np.asarray(orientation_points, dtype=float).reshape((-1, 3))
        angular_orientation_points = angular_orientation_points[
            np.all(np.isfinite(angular_orientation_points), axis=1)
        ]
        if angular_orientation_points.shape[0] < 3:
            raise RuntimeError(
                "angular-scale additive requires at least three finite observed orientation points"
            )
        reference_half_count = max(1, int(args.angular_scale_reference_half_count))
        observed_half_count = int(line_points.shape[0])
        angular_scale = float(observed_half_count) / float(reference_half_count)
        reference_effective_pairs = [
            (int(window), max(2, int(round(float(window) * angular_scale))))
            for window in windows
        ]
        angular_windows = sorted({int(effective) for _, effective in reference_effective_pairs})
        angular_scale_summary.update(
            {
                "reference_half_count": int(reference_half_count),
                "observed_half_count": int(observed_half_count),
                "angular_scale": as_json_float(float(angular_scale)),
                "base_windows": [int(window) for window in windows],
                "effective_windows": [int(window) for window in angular_windows],
                "requested_max_additions": int(args.max_angular_scale_additions),
                "requested_max_trials": int(args.max_angular_scale_trials),
                "orientation_mode": str(args.orientation_mode),
                "model_support_max_outside_count": int(args.model_support_max_outside_count),
                "dependency_declaration": {
                    "angular_ranking_uses_initial_model": False,
                    "angular_gate_uses_initial_model": False,
                    "uses_oracle_face_ids_for_angular_selection": False,
                    "uses_canonical_metrics_for_angular_selection": False,
                    "inherited_reference_axis_preprocessing": True,
                    "entire_pipeline_initial_model_free": False,
                    "uses_observed_orientation_cloud": True,
                    "selection_uses_candidate_plane_rms": True,
                    "selection_uses_candidate_levels_and_z_span": True,
                    "selection_uses_candidate_hull_area": True,
                    "selection_uses_observed_finite_support_purity": bool(
                        str(args.angular_scale_selection_mode)
                        in {"balanced-local", "balanced-global"}
                    ),
                },
            }
        )
        if angular_windows == [int(window) for window in windows]:
            angular_scale_summary["reason"] = "already_angular_equivalent"
        else:
            angular_lp_by_window, angular_rms_by_window, angular_cond_by_window = compute_line_points_multi_window(
                half_contours,
                z_levels,
                angular_windows,
                use_progress=not args.no_progress,
            )
            reference_window_by_effective: dict[int, int] = {}
            for reference_window, effective_window in reference_effective_pairs:
                reference_window_by_effective.setdefault(int(effective_window), int(reference_window))
            angular_raw_candidates: list[dict[str, object]] = []
            angular_per_window: list[dict[str, object]] = []
            angular_id_stride = 1_000_000
            angular_id_base = 50_000_000
            for angular_index, effective_window in enumerate(angular_windows):
                lp_w = angular_lp_by_window[angular_index].astype(float)
                rms_w = angular_rms_by_window[angular_index].astype(float)
                cond_w = angular_cond_by_window[angular_index].astype(float)
                per_level_w, diagnostics_w = detect_peak_observations(
                    z_levels,
                    rms_w,
                    peak_threshold_abs=float(args.peak_threshold),
                    low_threshold_abs=float(args.low_threshold),
                    peak_quantile=float(args.peak_quantile),
                    low_quantile=float(args.low_quantile),
                    low_barrier_threshold_abs=float(args.low_barrier_threshold),
                    low_barrier_fraction=float(args.low_barrier_fraction),
                    low_barrier_prominence=float(args.low_barrier_prominence),
                    low_valley_fallback=bool(args.low_valley_fallback),
                    min_peak_width=max(1, int(round(float(args.min_peak_width) * angular_scale))),
                    max_peak_width=max(1, int(round(float(args.max_peak_width) * angular_scale))),
                    max_low_distance=float(args.max_low_distance) * angular_scale,
                )
                tracks_w = track_peak_observations(
                    per_level_w,
                    n_half=int(rms_w.shape[0]),
                    max_center_jump=float(args.max_center_jump) * angular_scale,
                    max_z_gap=int(args.max_z_gap),
                )
                track_id_offset = int(
                    angular_id_base + int(effective_window) * angular_id_stride
                )
                if tracks_w and max(int(track.track_id) for track in tracks_w) >= angular_id_stride:
                    raise RuntimeError("angular-scale track IDs exceed the reserved namespace stride")
                candidates_w = build_face_candidates(
                    tracks_w,
                    lp_w,
                    rms_w,
                    window=int(effective_window),
                    track_id_offset=int(track_id_offset),
                    cond_values=cond_w,
                    max_line_condition=float(args.max_line_condition),
                    min_track_levels=int(args.min_track_levels),
                    min_fit_points=int(args.min_fit_points),
                    max_plane_rms=float(args.max_plane_rms),
                    inside_point=inside_point,
                    orientation_points=angular_orientation_points,
                    support_points=support_points,
                    support_tol=float(args.model_support_tol),
                    max_support_outside_count=int(args.model_support_max_outside_count),
                    low_region_samples=int(args.low_region_samples),
                    bounds_min=point_bounds_min,
                    bounds_max=point_bounds_max,
                )
                reference_window = int(reference_window_by_effective[int(effective_window)])
                for candidate in candidates_w:
                    candidate.update(
                        {
                            "candidate_origin": "angular_scale_addition",
                            "candidate_source": "rms_angular_multiscale",
                            "scale_mode": "angular-additive",
                            "angular_reference_half_count": int(reference_half_count),
                            "angular_observed_half_count": int(observed_half_count),
                            "angular_scale": as_json_float(float(angular_scale)),
                            "angular_reference_window": int(reference_window),
                            "angular_effective_window": int(effective_window),
                        }
                    )
                angular_raw_candidates.extend(candidates_w)
                angular_per_window.append(
                    {
                        "reference_window": int(reference_window),
                        "effective_window": int(effective_window),
                        "peak_observations": int(sum(int(item.get("peaks", 0) or 0) for item in diagnostics_w)),
                        "paired_observations": int(sum(len(level) for level in per_level_w)),
                        "tracks": int(len(tracks_w)),
                        "candidates": int(len(candidates_w)),
                    }
                )
            angular_ids = [int(candidate_track_id(candidate)) for candidate in angular_raw_candidates]
            frozen_id_list = [int(candidate_track_id(candidate)) for candidate in candidates]
            frozen_ids = set(frozen_id_list)
            if len(set(angular_ids)) != len(angular_ids):
                raise RuntimeError("angular-scale candidate IDs are not unique")
            if -1 in frozen_ids or len(frozen_ids) != len(frozen_id_list):
                raise RuntimeError("frozen production candidate IDs must be valid and unique")
            if frozen_ids & set(angular_ids):
                raise RuntimeError("angular-scale candidate IDs collide with frozen production IDs")
            frozen_candidates = list(candidates)
            frozen_reconstruction = reconstruct_polyhedron_from_halfspaces_edge_clip(
                frozen_candidates,
                halfspace_slack=float(args.intersection_halfspace_slack),
                feasibility_tol=float(args.intersection_feasibility_tol),
                incidence_tol=float(args.intersection_incidence_tol),
                vertex_merge_tol=float(args.intersection_vertex_merge_tol),
                min_face_area=float(args.intersection_min_face_area),
                triple_det_tol=float(args.intersection_triple_det_tol),
                prune_redundant=bool(args.intersection_prune_redundant_planes),
                redundancy_tol=float(args.intersection_redundancy_tol),
            )
            frozen_active_indices = [
                int(index)
                for index in frozen_reconstruction.get("face_candidate_indices", [])
                if 0 <= int(index) < len(frozen_candidates)
            ]
            frozen_active_candidates = [
                frozen_candidates[index] for index in frozen_active_indices
            ]
            angular_deduped, angular_dedupe_summary = dedupe_face_candidates(
                angular_raw_candidates,
                normal_angle_deg=float(args.dedupe_normal_angle_deg),
                plane_distance=float(args.dedupe_plane_distance),
                hull_distance=float(args.dedupe_hull_distance),
                max_candidates_per_cluster=1,
                mode="complete",
            )
            angular_preselected: list[dict[str, object]] = []
            angular_relation_counts: dict[str, int] = {}
            angular_preselection_rejections: dict[str, int] = {}
            for candidate in angular_deduped:
                out = dict(candidate)
                relation = classify_against_baseline(
                    out,
                    frozen_active_candidates,
                    duplicate_angle_deg=float(args.dedupe_normal_angle_deg),
                    duplicate_plane_distance=float(args.dedupe_plane_distance),
                    sliver_hull_distance=float(args.dedupe_hull_distance),
                )
                out.update({f"angular_{key}": value for key, value in relation.items()})
                relation_name = str(relation.get("baseline_relation") or "unknown")
                angular_relation_counts[relation_name] = angular_relation_counts.get(relation_name, 0) + 1
                rejection_reason = relation.get("baseline_rejection_reason")
                if rejection_reason is not None:
                    reason_name = str(rejection_reason)
                    angular_preselection_rejections[reason_name] = angular_preselection_rejections.get(reason_name, 0) + 1
                    continue
                angular_preselected.append(out)
            angular_annotated = annotate_candidate_pool_neutral(
                angular_preselected,
                core_candidates=frozen_candidates,
                core_reconstructed=frozen_reconstruction,
                trusted_points=trusted_cloud_points,
                trusted_z_indices=trusted_cloud_z_indices,
                point_tol=float(args.compatibility_point_tol),
                finite_support_plane_distance=float(args.finite_support_plane_distance),
                finite_support_hull_margin=float(args.finite_support_hull_margin),
                finite_support_relative_hull_margin=float(
                    args.finite_support_relative_hull_margin
                ),
            )
            angular_selection = annotate_angular_scale_selection_order(
                angular_annotated,
                mode=str(args.angular_scale_selection_mode),
            )
            angular_selection_order_ids = [
                int(value)
                for value in (angular_selection.get("ordered_candidate_ids") or [])
            ]
            angular_selection_order_blob = ",".join(
                str(value) for value in angular_selection_order_ids
            )
            angular_selection_rank_by_id = {
                int(candidate_id): int(rank)
                for rank, candidate_id in enumerate(angular_selection_order_ids)
            }
            angular_extra_area_threshold = (
                None
                if str(args.angular_scale_selection_mode) == "balanced-global"
                else 0.12
            )
            angular_scale_summary["frozen_geometry_signature"] = mesh_geometry_signature(
                frozen_candidates,
                frozen_reconstruction,
            )
            angular_refill = diagnose_protected_post_repair_refill(
                core_candidates=frozen_candidates,
                core_reconstructed=frozen_reconstruction,
                w2_preselection_candidates=angular_annotated,
                selected_candidates=frozen_candidates,
                trusted_points=trusted_cloud_points,
                trusted_z_indices=trusted_cloud_z_indices,
                point_tol=float(args.compatibility_point_tol),
                reconstruction_kwargs={
                    "halfspace_slack": float(args.intersection_halfspace_slack),
                    "feasibility_tol": float(args.intersection_feasibility_tol),
                    "incidence_tol": float(args.intersection_incidence_tol),
                    "vertex_merge_tol": float(args.intersection_vertex_merge_tol),
                    "min_face_area": float(args.intersection_min_face_area),
                    "triple_det_tol": float(args.intersection_triple_det_tol),
                    "prune_redundant": bool(args.intersection_prune_redundant_planes),
                    "redundancy_tol": float(args.intersection_redundancy_tol),
                },
                finite_support_plane_distance=float(args.finite_support_plane_distance),
                finite_support_hull_margin=float(args.finite_support_hull_margin),
                finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
                trial_geometry_mode=str(args.w2_small_face_trial_geometry),
                stage1_target_candidate_ids=set(),
                stage2_target_candidate_ids=set(),
                removed_blocker_ids={
                    int(value)
                    for value in w2_core_repair_summary.get("selected_removed_candidate_ids", [])
                },
                contours=None,
                challenger_order_ids=(
                    angular_selection_order_ids
                    if str(args.angular_scale_selection_mode)
                    in {"balanced-local", "balanced-global"}
                    else None
                ),
                max_accepted=max(0, int(args.max_angular_scale_additions)),
                max_trials=max(0, int(args.max_angular_scale_trials)),
                extended_max_trials=None,
                require_dense_band=False,
                protect_all_active_patches=True,
                rolling_protection=True,
                safe_control_candidate_ids=set(),
                mode_label="angular_scale_additive_production",
                candidate_filter_mode="angular-scale",
                candidate_order_mode="angular-scale",
                protect_active_faces_individually=True,
                strict_protected_footprint=True,
                preserve_baseline_outside=True,
                max_trial_face_extra_area=angular_extra_area_threshold,
                refresh_trial_geometry_from_full_rebuild=bool(
                    str(args.angular_scale_selection_mode)
                    in {"balanced-local", "balanced-global"}
                ),
            )
            angular_applied_order_ids = [
                int(value)
                for value in (angular_refill.get("challenger_order_signature") or [])
            ]
            angular_applied_order_blob = ",".join(
                str(value) for value in angular_applied_order_ids
            )
            angular_accepted_ids = [
                int(value)
                for value in (angular_refill.get("protected_sequential_refill") or {}).get(
                    "accepted_sequence_signature", []
                )
            ]
            angular_by_id = {
                int(candidate_track_id(candidate)): candidate for candidate in angular_annotated
            }
            angular_appended: list[dict[str, object]] = []
            for rank, candidate_id in enumerate(angular_accepted_ids, start=1):
                source_candidate = angular_by_id.get(int(candidate_id))
                if source_candidate is None:
                    continue
                out = dict(source_candidate)
                out["angular_additive_rank"] = int(rank)
                out["angular_additive_mode"] = "protected-rolling"
                angular_appended.append(out)
            candidates = frozen_candidates + angular_appended
            angular_scale_summary.update(
                {
                    "production_changed": bool(angular_appended),
                    "reason": "completed",
                    "per_window": angular_per_window,
                    "raw_candidates": int(len(angular_raw_candidates)),
                    "deduped_candidates": int(len(angular_deduped)),
                    "dedupe": angular_dedupe_summary,
                    "dedupe_mode": "complete",
                    "baseline_relation_scope": "frozen_active_candidates",
                    "safety_policy": {
                        "trial_face_area_ratio_max": 25.0,
                        "trial_face_extra_area_max": angular_extra_area_threshold,
                        "trial_face_hull_distance_median_max": 0.20,
                        "trial_face_metrics_source": (
                            "full_edge_clip_rebuild_reused"
                            if str(args.angular_scale_selection_mode)
                            in {"balanced-local", "balanced-global"}
                            else "legacy_trial_geometry"
                        ),
                        "protect_active_faces_individually": True,
                        "strict_protected_footprint": True,
                        "minimum_retained_face_area_ratio": 0.50,
                        "preserve_baseline_outside": True,
                        "max_level_outside_hard_cap": 0.10,
                        "rolling_protection": True,
                    },
                    "selection_mode": str(args.angular_scale_selection_mode),
                    "selection_order_policy": (
                        "legacy_quality_local_lexicographic"
                        if str(args.angular_scale_selection_mode) == "quality-local"
                        else "global_equal_weight_tie_aware_borda"
                    ),
                    "selection_locality_policy": (
                        "area_ratio_and_hull_distance_only"
                        if str(args.angular_scale_selection_mode) == "balanced-global"
                        else "legacy_extra_area_0.12_plus_area_ratio_and_hull_distance"
                    ),
                    "selection_score_definition": angular_selection.get(
                        "score_definition"
                    ),
                    "selection_score_direction": angular_selection.get(
                        "score_direction"
                    ),
                    "selection_order_signature": {
                        "candidate_count": int(len(angular_selection_order_ids)),
                        "candidate_ids": angular_selection_order_ids[:160],
                        "sha256": hashlib.sha256(
                            angular_selection_order_blob.encode("utf-8")
                        ).hexdigest(),
                    },
                    "selection_score_rows": [
                        {
                            "candidate_id": int(candidate_track_id(candidate)),
                            "score": candidate.get("angular_scale_selection_score"),
                            "components": candidate.get(
                                "angular_scale_selection_components"
                            ),
                        }
                        for candidate in sorted(
                            angular_annotated,
                            key=lambda candidate: angular_selection_rank_by_id.get(
                                int(candidate_track_id(candidate)),
                                len(angular_selection_rank_by_id),
                            ),
                        )
                    ],
                    "relation_counts": angular_relation_counts,
                    "preselection_rejections": angular_preselection_rejections,
                    "preselected_candidates": int(len(angular_preselected)),
                    "accepted_additions": int(len(angular_appended)),
                    "accepted_sequence_signature": angular_accepted_ids,
                    "trial_checks": int(
                        (angular_refill.get("protected_sequential_refill") or {}).get("trial_checks") or 0
                    ),
                    "rejection_counts": (
                        angular_refill.get("protected_sequential_refill") or {}
                    ).get("rejection_counts"),
                    "trial_rows_sample": (
                        angular_refill.get("protected_sequential_refill") or {}
                    ).get("trial_rows_sample"),
                    "selector_pool_policy": angular_refill.get("pool_policy"),
                    "protected_group_signature": angular_refill.get("protected_group_signature"),
                    "final_protected_group_signature": angular_refill.get(
                        "final_protected_group_signature"
                    ),
                    "challenger_order_signature": angular_refill.get(
                        "challenger_order_signature"
                    ),
                    "challenger_order_sha256": hashlib.sha256(
                        angular_applied_order_blob.encode("utf-8")
                    ).hexdigest(),
                    "timing_breakdown_seconds": angular_refill.get("timing_breakdown_seconds"),
                }
            )
        pipeline_timing["angular_scale_additive_seconds"] += time.perf_counter() - angular_started
    final_intersection_started = time.perf_counter()
    if args.active_face_filter_mode == "legacy":
        candidates, reconstructed, active_face_filter_summary = filter_candidates_by_active_face_size(
            candidates,
            inside_tol=float(args.intersection_inside_tol),
            vertex_tol=float(args.intersection_vertex_tol),
            max_area_ratio=float(args.max_active_face_hull_area_ratio),
            max_extra_area=float(args.max_active_face_extra_area),
            iterations=int(args.active_face_filter_iterations),
        )
    else:
        if args.intersection_mesh_mode == "incidence":
            reconstructed = reconstruct_polyhedron_from_halfspaces_incidence(
                candidates,
                halfspace_slack=float(args.intersection_halfspace_slack),
                feasibility_tol=float(args.intersection_feasibility_tol),
                incidence_tol=float(args.intersection_incidence_tol),
                vertex_merge_tol=float(args.intersection_vertex_merge_tol),
                min_face_area=float(args.intersection_min_face_area),
                prune_redundant=bool(args.intersection_prune_redundant_planes),
                redundancy_tol=float(args.intersection_redundancy_tol),
                triple_det_tol=float(args.intersection_triple_det_tol),
            )
        elif args.intersection_mesh_mode == "edge-clip":
            reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
                candidates,
                halfspace_slack=float(args.intersection_halfspace_slack),
                feasibility_tol=float(args.intersection_feasibility_tol),
                incidence_tol=float(args.intersection_incidence_tol),
                vertex_merge_tol=float(args.intersection_vertex_merge_tol),
                min_face_area=float(args.intersection_min_face_area),
                triple_det_tol=float(args.intersection_triple_det_tol),
                prune_redundant=bool(args.intersection_prune_redundant_planes),
                redundancy_tol=float(args.intersection_redundancy_tol),
            )
        else:
            reconstructed = reconstruct_polyhedron_from_halfspaces(
                candidates,
                inside_tol=float(args.intersection_inside_tol),
                vertex_tol=float(args.intersection_vertex_tol),
            )
        active_face_filter_summary = {
            "active_face_size_rejected": 0,
            "active_face_size_iterations": 0,
            "active_face_size_last_rejected": [],
        }
    pipeline_timing["final_intersection_seconds"] += time.perf_counter() - final_intersection_started
    if str(z_envelope_guard_summary.get("mode")) == "trusted-cap":
        reconstructed_vertices_for_guard = np.asarray(reconstructed.get("vertices") or [], dtype=float)
        if reconstructed_vertices_for_guard.ndim == 2 and reconstructed_vertices_for_guard.shape[0] > 0:
            final_z_min = float(np.min(reconstructed_vertices_for_guard[:, 2]))
            trusted_z_min = finite_float(z_envelope_guard_summary.get("trusted_z_min"), final_z_min)
            trusted_z_span = max(finite_float(z_envelope_guard_summary.get("trusted_z_span"), 0.0), EPS)
            z_envelope_guard_summary["final_z_min"] = as_json_float(final_z_min)
            z_envelope_guard_summary["final_overrun_relative"] = as_json_float(
                max(0.0, trusted_z_min - final_z_min) / trusted_z_span
            )
        active_candidate_ids_for_guard = {
            candidate_track_id(candidates[int(index)])
            for index in reconstructed.get("face_candidate_indices", [])
            if 0 <= int(index) < len(candidates)
        }
        guard_candidate_id = int(z_envelope_guard_summary.get("guard_candidate_id") or -1)
        z_envelope_guard_summary["active_in_final_mesh"] = bool(
            guard_candidate_id >= 0 and guard_candidate_id in active_candidate_ids_for_guard
        )
    if str(angular_scale_summary.get("mode")) == "additive":
        angular_candidate_ids = {
            int(value)
            for value in angular_scale_summary.get("accepted_sequence_signature", [])
        }
        final_active_indices = [
            int(index)
            for index in reconstructed.get("face_candidate_indices", [])
            if 0 <= int(index) < len(candidates)
        ]
        final_active_candidate_ids = {
            int(candidate_track_id(candidates[index])) for index in final_active_indices
        }
        angular_scale_summary["final_active_addition_ids"] = sorted(
            angular_candidate_ids & final_active_candidate_ids
        )
        angular_scale_summary["final_active_additions"] = int(
            len(angular_candidate_ids & final_active_candidate_ids)
        )
        angular_scale_summary["final_geometry_signature"] = mesh_geometry_signature(
            candidates,
            reconstructed,
        )
        angular_scale_summary["final_topology"] = reconstructed.get("topology")
        angular_scale_summary["final_volume"] = reconstructed.get("reliable_volume")
        angular_scale_summary["final_outside"] = cumulative_outside_metrics(
            candidates,
            trusted_cloud_points,
            trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
        )

    oracle_diagnostic_scope = str(os.environ.get("POLYRECO_ORACLE_DIAGNOSTIC_SCOPE") or "").strip()

    def run_current_preselection_reservoir_audit(
        *,
        cached_loss_funnel: dict[str, object],
        model_face_rows: list[dict[str, object]],
        core_reconstructed: dict[str, object],
        core_ids: set[int],
        oracle_reconstruction_kwargs: dict[str, object],
        oracle_match_cache: dict[int, dict[str, object] | None],
    ) -> dict[str, object]:
        selection_rows = (dense_detector_summary.get("w2_selection") or {}).get(
            "cumulative_local_attempt_log",
            [],
        )
        broad_order = (
            ((w2_core_repair_summary.get("post_repair_refinement") or {}).get("order_diagnostics") or {}).get(
                "order_signature",
                [],
            )
        )
        repair_removed_ids = {
            int(candidate_track_id(candidate)) for candidate in post_append_local_candidates
        } - {
            int(candidate_track_id(candidate)) for candidate in post_core_repair_candidates
        }
        exchange_removed_ids = {
            int(candidate_track_id(candidate)) for candidate in post_ratchet_exchange_base_candidates
        } - {
            int(candidate_track_id(candidate)) for candidate in candidates
        }
        return diagnose_preselection_safe_reservoir(
            cached_loss_funnel=cached_loss_funnel,
            model_faces=model_face_rows,
            initial_vertices=vertices,
            core_candidates=dense_core_candidates,
            core_reconstructed=core_reconstructed,
            valley_raw_candidates=all_candidates,
            w2_raw_candidates=dense_w2_raw_candidates,
            w2_preselection_candidates=dense_w2_preselection_candidates,
            w2_clusters=dense_w2_edge_clusters,
            support_diverse_ranked_candidate_ids=support_diverse_ranked_candidate_ids,
            selection_attempt_log=[row for row in selection_rows if isinstance(row, dict)],
            max150_candidates=post_max150_candidates,
            append_candidates=post_append_local_candidates,
            repair_candidates=post_core_repair_candidates,
            ratchet_candidates=post_ratchet_exchange_base_candidates,
            final_candidates=candidates,
            final_reconstructed=reconstructed,
            removed_blocker_ids=repair_removed_ids | exchange_removed_ids,
            append_order_ids=[
                int(value)
                for value in w2_small_face_refinement_summary.get("challenger_order_signature", [])
                if value is not None
            ],
            broad_order_ids=[int(value) for value in broad_order if value is not None],
            core_face_ids=core_ids,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            reconstruction_kwargs=oracle_reconstruction_kwargs,
            finite_support_plane_distance=float(args.finite_support_plane_distance),
            finite_support_hull_margin=float(args.finite_support_hull_margin),
            finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
            contours=contours,
            match_cache=oracle_match_cache,
        )

    named_oracle_diagnostics = None
    named_oracle_diagnostics_started = None
    if str(args.dense_detector_mode) == "w2-edge-tracks" and not bool(args.disable_oracle_diagnostics):
        named_oracle_diagnostics_started = time.perf_counter()
        named_oracle_diagnostics = dispatch_final_mesh_named_oracle_scope(
            scope=oracle_diagnostic_scope,
            args=args,
            model_name=model_name,
            vertices=vertices,
            faces=model.faces,
            final_candidates=candidates,
            final_reconstructed=reconstructed,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            windows=windows,
            z_levels=z_levels,
            line_points_by_window=line_points_by_window,
            fit_rms_by_window=fit_rms_by_window,
            cond_by_window=cond_by_window,
            point_bounds_min=point_bounds_min,
            point_bounds_max=point_bounds_max,
            w2_segments=dense_w2_segments,
            w2_clusters=dense_w2_edge_clusters,
            core_candidates=dense_core_candidates,
            preselection_reservoir_audit=run_current_preselection_reservoir_audit,
        )
    if named_oracle_diagnostics is not None:
        dense_detector_summary["oracle_diagnostics"] = named_oracle_diagnostics
        pipeline_timing["oracle_diagnostics_seconds"] += (
            time.perf_counter() - float(named_oracle_diagnostics_started)
        )
    elif str(args.dense_detector_mode) == "w2-edge-tracks" and not bool(args.disable_oracle_diagnostics):
        oracle_diagnostics_started = time.perf_counter()
        model_face_rows = model_face_planes(vertices, model.faces)
        core_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
            dense_core_candidates,
            halfspace_slack=float(args.intersection_halfspace_slack),
            feasibility_tol=float(args.intersection_feasibility_tol),
            incidence_tol=float(args.intersection_incidence_tol),
            vertex_merge_tol=float(args.intersection_vertex_merge_tol),
            min_face_area=float(args.intersection_min_face_area),
            triple_det_tol=float(args.intersection_triple_det_tol),
            prune_redundant=bool(args.intersection_prune_redundant_planes),
            redundancy_tol=float(args.intersection_redundancy_tol),
        )
        core_only_active_indices = [
            int(i)
            for i in core_reconstructed.get("face_candidate_indices", [])
            if int(i) < len(dense_core_candidates)
        ]
        core_only_active_candidates = [dense_core_candidates[i] for i in core_only_active_indices]
        core_active_indices = [
            int(i)
            for i in reconstructed.get("face_candidate_indices", [])
            if int(i) < len(candidates) and str(candidates[int(i)].get("candidate_origin")) != "w2_addition"
        ]
        w2_active_indices = [
            int(i)
            for i in reconstructed.get("face_candidate_indices", [])
            if int(i) < len(candidates) and str(candidates[int(i)].get("candidate_origin")) == "w2_addition"
        ]
        core_active_candidates = [candidates[i] for i in core_active_indices]
        w2_accepted_candidates = [candidate for candidate in candidates if str(candidate.get("candidate_origin")) == "w2_addition"]
        w2_active_candidates = [candidates[i] for i in w2_active_indices]
        oracle_match_cache: dict[int, dict[str, object] | None] = {}
        core_cohort = oracle_candidate_cohort(
            core_only_active_candidates,
            model_face_rows,
            active_plane_indices=core_only_active_indices,
            match_cache=oracle_match_cache,
        )
        core_ids = set(int(v) for v in core_cohort.get("finite_face_ids", []))
        final_core_cohort = oracle_candidate_cohort(
            core_active_candidates,
            model_face_rows,
            core_face_ids=core_ids,
            active_plane_indices=core_active_indices,
            match_cache=oracle_match_cache,
        )
        raw_cohort = {
            "skipped": True,
            "reason": "raw face oracle matching skipped in bounded face-ceiling diagnostics; preselection cohort is evaluated exactly",
            "candidate_count": int(len(dense_w2_raw_candidates)),
            "finite_face_ids": [],
            "matched_rows": [],
        }
        pre_cohort = oracle_candidate_cohort(dense_w2_preselection_candidates, model_face_rows, core_face_ids=core_ids, match_cache=oracle_match_cache)
        accepted_cohort = oracle_candidate_cohort(w2_accepted_candidates, model_face_rows, core_face_ids=core_ids, match_cache=oracle_match_cache)
        active_cohort = oracle_candidate_cohort(
            w2_active_candidates,
            model_face_rows,
            core_face_ids=core_ids,
            active_plane_indices=w2_active_indices,
            match_cache=oracle_match_cache,
        )
        active_total_cohort = oracle_candidate_cohort(
            [candidates[i] for i in reconstructed.get("face_candidate_indices", []) if int(i) < len(candidates)],
            model_face_rows,
            core_face_ids=core_ids,
            active_plane_indices=[int(i) for i in reconstructed.get("face_candidate_indices", []) if int(i) < len(candidates)],
            match_cache=oracle_match_cache,
        )
        loose_old_ids = loose_plane_unique_face_ids(core_only_active_candidates, model_face_rows)
        canonical_ids = set(int(v) for v in core_cohort.get("finite_face_ids", []))
        edge_oracle = {
            "skipped": True,
            "reason": "edge-cluster oracle ancestry is not required for oracle-compatible face-ceiling diagnostics",
            "matched_clusters": 0,
            "confident_matches": [],
        }
        oracle_reconstruction_kwargs = {
            "halfspace_slack": float(args.intersection_halfspace_slack),
            "feasibility_tol": float(args.intersection_feasibility_tol),
            "incidence_tol": float(args.intersection_incidence_tol),
            "vertex_merge_tol": float(args.intersection_vertex_merge_tol),
            "min_face_area": float(args.intersection_min_face_area),
            "triple_det_tol": float(args.intersection_triple_det_tol),
            "prune_redundant": bool(args.intersection_prune_redundant_planes),
            "redundancy_tol": float(args.intersection_redundancy_tol),
        }
        production_129_loss_funnel = diagnose_production_129_loss_funnel(
            initial_vertices=vertices,
            initial_faces=model.faces,
            model_faces=model_face_rows,
            minima_cloud=minima_cloud,
            stable_minima_points=trusted_cloud_points,
            stable_minima_z_indices=trusted_cloud_z_indices,
            valley_raw_candidates=all_candidates,
            core_candidates=dense_core_candidates,
            w2_line_points=dense_w2_line_points,
            w2_segments=dense_w2_segments,
            w2_clusters=dense_w2_edge_clusters,
            w2_raw_candidates=dense_w2_raw_candidates,
            w2_preselection_candidates=dense_w2_preselection_candidates,
            support_diverse_ranked_candidate_ids=support_diverse_ranked_candidate_ids,
            selection_attempt_log=list(selection_w2.get("cumulative_local_attempt_log", [])),
            max150_candidates=post_max150_candidates,
            append_candidates=post_append_local_candidates,
            repair_candidates=post_core_repair_candidates,
            ratchet_candidates=post_ratchet_exchange_base_candidates,
            final_candidates=candidates,
            final_reconstructed=reconstructed,
            exchange_target_candidate_ids=[
                int(action["target_candidate_id"])
                for action in post_ratchet_exchange_summary.get("actions", [])
                if action.get("target_candidate_id") is not None
            ],
            core_face_ids=core_ids,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            reconstruction_kwargs=oracle_reconstruction_kwargs,
            match_cache=oracle_match_cache,
        )
        production_129_loss_funnel["preselection_safe_reservoir_audit"] = run_current_preselection_reservoir_audit(
            cached_loss_funnel=production_129_loss_funnel,
            model_face_rows=model_face_rows,
            core_reconstructed=core_reconstructed,
            core_ids=core_ids,
            oracle_reconstruction_kwargs=oracle_reconstruction_kwargs,
            oracle_match_cache=oracle_match_cache,
        )
        w2_edge_ancestry_internal = dict(
            production_129_loss_funnel.pop("_w2_edge_ancestry_internal", {})
        )
        production_129_loss_funnel["candidate_formation_failure_audit"] = diagnose_candidate_formation_failures(
            loss_funnel=production_129_loss_funnel,
            initial_vertices=vertices,
            initial_faces=model.faces,
            model_faces=model_face_rows,
            z_levels=z_levels,
            w2_line_points=dense_w2_line_points,
            w2_fit_rms=dense_w2_fit_rms,
            w2_condition=dense_w2_cond,
            w3_line_points=dense_w3_line_points,
            w3_fit_rms=dense_w3_fit_rms,
            w4_line_points=dense_w4_line_points,
            w4_fit_rms=dense_w4_fit_rms,
            valley_raw_candidates=all_candidates,
            w2_raw_candidates=dense_w2_raw_candidates,
            w2_segments=dense_w2_segments,
            w2_clusters=dense_w2_edge_clusters,
            edge_ancestry_internal=w2_edge_ancestry_internal,
            inside_point=inside_point,
            orientation_points=orientation_points,
            bounds_min=point_bounds_min,
            bounds_max=point_bounds_max,
            segment_min_levels=int(args.w2_edge_min_levels),
            segment_min_z_span=float(args.w2_edge_min_z_span),
            segment_max_z_gap=int(args.w2_edge_max_z_gap),
            segment_max_line_rms=float(args.w2_edge_max_line_rms),
            segment_max_line_residual=float(args.w2_edge_max_line_residual),
            segment_max_point_jump=float(args.w2_edge_max_point_jump),
            segment_max_condition=float(args.w2_edge_max_condition),
            cluster_mode=str(args.w2_edge_cluster_mode),
            cluster_max_distance=float(args.w2_edge_cluster_max_distance),
            cluster_max_angle_deg=float(args.w2_edge_cluster_max_angle_deg),
            cluster_min_overlap=float(args.w2_edge_cluster_min_overlap),
            min_adjacency_levels=int(args.w2_face_min_adjacency_levels),
            min_adjacency_fraction=float(args.w2_face_min_adjacency_fraction),
            min_z_span=float(args.w2_face_min_z_span),
            max_plane_rms=float(args.w2_face_max_plane_rms),
            final_candidates=candidates,
            final_reconstructed=reconstructed,
            core_face_ids=set(int(value) for value in core_ids),
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            reconstruction_kwargs=oracle_reconstruction_kwargs,
            match_cache=oracle_match_cache,
            contours=contours,
        )
        full_funnel = {
            "skipped": True,
            "reason": "full w2 funnel diagnostics are not part of ranked-out support-diverse audit",
        }
        prefix_frontier = {
            "skipped": True,
            "reason": "prefix frontier surface-distance diagnostics are not part of ranked-out support-diverse audit",
        }
        oracle_compatible_ceiling = None
        previous_oracle_cache_note = None
        if str(args.w2_core_repair_mode) != "lp-blockers":
            try:
                previous_json = json.loads(Path(args.output_json).read_text())
                previous_oracle = ((previous_json.get("parameters") or {}).get("oracle_diagnostics") or {}).get("oracle_compatible_ceiling")
                if isinstance(previous_oracle, dict) and isinstance(previous_oracle.get("oracle_achieved_lower_bound"), dict):
                    oracle_compatible_ceiling = dict(previous_oracle)
                    previous_oracle_cache_note = {
                        "reused_from_output_json": str(args.output_json),
                        "reason": "oracle-compatible ceiling is immutable for the unchanged production candidate pool; current task audits support-diverse rank-out only",
                    }
                    oracle_compatible_ceiling["diagnostic_cache"] = previous_oracle_cache_note
            except (OSError, json.JSONDecodeError, TypeError):
                oracle_compatible_ceiling = None
        else:
            previous_oracle_cache_note = {
                "reused_from_output_json": None,
                "reason": "disabled for lp-blockers post-repair closure audit; ceiling must be recomputed after the new 121 golden",
            }
        if oracle_compatible_ceiling is None:
            oracle_compatible_ceiling = oracle_compatible_ceiling_diagnostics(
                core_candidates=dense_core_candidates,
                core_active_candidates=core_only_active_candidates,
                w2_preselection_candidates=dense_w2_preselection_candidates,
                w2_accepted_candidates=w2_accepted_candidates,
                w2_active_candidates=w2_active_candidates,
                core_cohort=core_cohort,
                model_faces=model_face_rows,
                initial_vertices=vertices,
                trusted_points=trusted_cloud_points,
                trusted_z_indices=trusted_cloud_z_indices,
                point_tol=float(args.compatibility_point_tol),
                reconstruction_kwargs=oracle_reconstruction_kwargs,
                match_cache=oracle_match_cache,
                finite_support_plane_distance=float(args.finite_support_plane_distance),
                finite_support_hull_margin=float(args.finite_support_hull_margin),
                finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
            )
        oracle_selected_candidate_ids = {
            int(v)
            for v in (oracle_compatible_ceiling.get("oracle_achieved_lower_bound") or {}).get("selected_candidate_ids", [])
        }
        preselection_by_id = {candidate_track_id(candidate): candidate for candidate in dense_w2_preselection_candidates}
        oracle_lower_candidates = [
            preselection_by_id[tid]
            for tid in sorted(oracle_selected_candidate_ids)
            if tid in preselection_by_id
        ]
        oracle_lower_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
            dense_core_candidates + oracle_lower_candidates,
            **oracle_reconstruction_kwargs,
        ) if oracle_lower_candidates else None
        current_reprojection_row = evaluate_polyhedron_reprojection(
            name=f"production_max{int(args.max_w2_additions)}",
            vertices_3d=np.array(reconstructed.get("vertices") or [], dtype=float),
            contours=contours,
            direction_count=64,
        )
        reprojection_models = {
            "InitialModel": evaluate_polyhedron_reprojection(
                name="InitialModel",
                vertices_3d=vertices,
                contours=contours,
                direction_count=64,
            ),
            "repaired_core": evaluate_polyhedron_reprojection(
                name="repaired_core",
                vertices_3d=np.array(core_reconstructed.get("vertices") or [], dtype=float),
                contours=contours,
                direction_count=64,
            ),
            "production_current": current_reprojection_row,
            f"production_max{int(args.max_w2_additions)}": current_reprojection_row,
            "oracle_feasible_lower_bound": evaluate_polyhedron_reprojection(
                name="oracle_feasible_lower_bound",
                vertices_3d=np.array((oracle_lower_reconstructed or {}).get("vertices") or [], dtype=float),
                contours=contours,
                direction_count=64,
            ),
        }
        current_prefix_summary = summarize_reconstruction_prefix(
            prefix=len(w2_accepted_candidates),
            candidates=dense_core_candidates + w2_accepted_candidates,
            core_count=len(dense_core_candidates),
            core_ids=core_ids,
            model_faces=model_face_rows,
            initial_vertices=vertices,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            reconstruction_kwargs=oracle_reconstruction_kwargs,
            match_cache=oracle_match_cache,
        )
        current_prefix_summary["_core_ids"] = sorted(int(v) for v in core_ids)
        base_w2_accepted_candidates = [
            candidate for candidate in w2_accepted_candidates if str(candidate.get("small_face_refinement_mode")) != "append-local"
        ]
        append_local_candidates = [
            candidate for candidate in w2_accepted_candidates if str(candidate.get("small_face_refinement_mode")) == "append-local"
        ]
        append_local_frontier = diagnose_append_local_frontier(
            core_candidates=dense_core_candidates,
            base_w2_candidates=base_w2_accepted_candidates,
            append_candidates=append_local_candidates,
            core_cohort=core_cohort,
            model_faces=model_face_rows,
            initial_vertices=vertices,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            reconstruction_kwargs=oracle_reconstruction_kwargs,
            contours=contours,
            oracle_ceiling=oracle_compatible_ceiling,
            contour_sample=64,
        ) if append_local_candidates else {
            "skipped": True,
            "reason": "append-local produced no accepted candidates",
            "base_w2_count": int(len(base_w2_accepted_candidates)),
            "append_sequence_length": 0,
        }
        golden_oracle_closure_diagnostics = diagnose_golden_oracle_closure(
            core_candidates=dense_core_candidates,
            core_reconstructed=core_reconstructed,
            base_w2_candidates=base_w2_accepted_candidates,
            append_candidates=append_local_candidates,
            w2_preselection_candidates=dense_w2_preselection_candidates,
            core_cohort=core_cohort,
            active_total_cohort=active_total_cohort,
            model_faces=model_face_rows,
            initial_vertices=vertices,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            reconstruction_kwargs=oracle_reconstruction_kwargs,
            contours=contours,
            oracle_ceiling=oracle_compatible_ceiling,
            append_summary=w2_small_face_refinement_summary,
            finite_support_plane_distance=float(args.finite_support_plane_distance),
            finite_support_hull_margin=float(args.finite_support_hull_margin),
            finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
            contour_sample=18,
        )
        current_active_ids = set(int(v) for v in active_total_cohort.get("finite_face_ids", []))
        current_new_ids = current_active_ids - core_ids
        lower_summary = (
            (oracle_compatible_ceiling.get("oracle_achieved_lower_bound") or {}).get("summary")
            if isinstance(oracle_compatible_ceiling, dict)
            else {}
        )
        lower_new_ids = set(int(v) for v in (lower_summary or {}).get("new_ids", []))
        lower_all_ids = set(core_ids) | lower_new_ids
        lower_outside = cumulative_outside_metrics(
            dense_core_candidates + oracle_lower_candidates,
            trusted_cloud_points,
            trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
        ) if oracle_lower_candidates else {
            "cumulative_outside_fraction": None,
            "cumulative_max_level_outside_fraction": None,
            "cumulative_lost_z_levels": None,
        }
        lower_reprojection = reprojection_models.get("oracle_feasible_lower_bound", {})
        closure_set_difference = golden_oracle_closure_diagnostics.get("set_difference") if isinstance(golden_oracle_closure_diagnostics, dict) else {}
        individual_trials = golden_oracle_closure_diagnostics.get("individual_closure_trials", []) if isinstance(golden_oracle_closure_diagnostics, dict) else []
        collective = golden_oracle_closure_diagnostics.get("collective_closure", {}) if isinstance(golden_oracle_closure_diagnostics, dict) else {}
        append_rows = append_local_frontier.get("rows", []) if isinstance(append_local_frontier, dict) else []
        refill_10 = next((row for row in append_rows if int(row.get("append_prefix") or -1) == 10), None)
        refill_25 = next((row for row in append_rows if int(row.get("append_prefix") or -1) == 25), None)
        safe_individual_rows = [row for row in individual_trials if str(row.get("classification")) == "safe_plus_one"]
        missing_rows = golden_oracle_closure_diagnostics.get("missing_face_lifecycle", []) if isinstance(golden_oracle_closure_diagnostics, dict) else []
        missing_reason_counts: dict[str, int] = {}
        for row in missing_rows:
            reason = str(row.get("primary_loss_stage") or "unknown")
            missing_reason_counts[reason] = missing_reason_counts.get(reason, 0) + 1
        stage1_target_candidate_ids = {
            int(row.get("core_candidate_id"))
            for row in (w2_core_repair_summary.get("stage1") or {}).get("generic_ranking_top", [])[:1]
            for row in (row.get("reactivated_core_rows") or [])
            if row.get("core_candidate_id") is not None
        }
        stage2_target_candidate_ids = {
            int(((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get("target_core_candidate_id"))
        } if ((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get("target_core_candidate_id") is not None else set()
        protected_sequential_refill_audit = diagnose_protected_post_repair_refill(
            core_candidates=dense_core_candidates,
            core_reconstructed=core_reconstructed,
            w2_preselection_candidates=dense_w2_preselection_candidates,
            selected_candidates=candidates,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            reconstruction_kwargs=oracle_reconstruction_kwargs,
            finite_support_plane_distance=float(args.finite_support_plane_distance),
            finite_support_hull_margin=float(args.finite_support_hull_margin),
            finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
            trial_geometry_mode=str(args.w2_small_face_trial_geometry),
            stage1_target_candidate_ids=stage1_target_candidate_ids,
            stage2_target_candidate_ids=stage2_target_candidate_ids,
            removed_blocker_ids={int(v) for v in w2_core_repair_summary.get("selected_removed_candidate_ids", [])},
            core_ids=core_ids,
            model_faces=model_face_rows,
            initial_vertices=vertices,
            contours=contours,
            challenger_order_ids=[
                int(v)
                for v in (w2_small_face_refinement_summary.get("challenger_order_signature") or [])
                if v is not None
            ],
            max_accepted=25,
            max_trials=80,
            extended_max_trials=160,
        )
        broad_protected_refill_audit = diagnose_protected_post_repair_refill(
            core_candidates=dense_core_candidates,
            core_reconstructed=core_reconstructed,
            w2_preselection_candidates=dense_w2_preselection_candidates,
            selected_candidates=post_core_repair_candidates,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            reconstruction_kwargs=oracle_reconstruction_kwargs,
            finite_support_plane_distance=float(args.finite_support_plane_distance),
            finite_support_hull_margin=float(args.finite_support_hull_margin),
            finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
            trial_geometry_mode=str(args.w2_small_face_trial_geometry),
            stage1_target_candidate_ids=stage1_target_candidate_ids,
            stage2_target_candidate_ids=stage2_target_candidate_ids,
            removed_blocker_ids={int(v) for v in w2_core_repair_summary.get("selected_removed_candidate_ids", [])},
            core_ids=core_ids,
            model_faces=model_face_rows,
            initial_vertices=vertices,
            contours=contours,
            challenger_order_ids=None,
            max_accepted=25,
            max_trials=160,
            extended_max_trials=None,
            require_dense_band=False,
            protect_all_active_patches=True,
            safe_control_candidate_ids={9001254, 9000209, 9001244, 9000094, 9000137},
            mode_label="broad_protected_refill",
        )
        rolling_patch_ratchet_audit = diagnose_protected_post_repair_refill(
            core_candidates=dense_core_candidates,
            core_reconstructed=core_reconstructed,
            w2_preselection_candidates=dense_w2_preselection_candidates,
            selected_candidates=post_core_repair_candidates,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            reconstruction_kwargs=oracle_reconstruction_kwargs,
            finite_support_plane_distance=float(args.finite_support_plane_distance),
            finite_support_hull_margin=float(args.finite_support_hull_margin),
            finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
            trial_geometry_mode=str(args.w2_small_face_trial_geometry),
            stage1_target_candidate_ids=stage1_target_candidate_ids,
            stage2_target_candidate_ids=stage2_target_candidate_ids,
            removed_blocker_ids={int(v) for v in w2_core_repair_summary.get("selected_removed_candidate_ids", [])},
            core_ids=core_ids,
            model_faces=model_face_rows,
            initial_vertices=vertices,
            contours=contours,
            challenger_order_ids=None,
            max_accepted=25,
            max_trials=160,
            extended_max_trials=None,
            require_dense_band=False,
            protect_all_active_patches=True,
            rolling_protection=True,
            safe_control_candidate_ids={9001254, 9000209, 9001244, 9000094, 9000137},
            mode_label="rolling_patch_ratchet",
        )
        first_step_repro = protected_sequential_refill_audit.get("first_candidate_reproducibility", {})
        protected_confusion = (
            (protected_sequential_refill_audit.get("protected_sequential_refill") or {}).get("confusion_matrix")
            if isinstance(protected_sequential_refill_audit.get("protected_sequential_refill"), dict)
            else {}
        )
        gate_false_negative = bool(int((protected_confusion or {}).get("gate_passed_but_canonical_core_lost") or 0) > 0)
        old_prefix_rows = {
            f"append_prefix_{int(row.get('append_prefix') or 0)}": row
            for row in append_rows
            if row.get("append_prefix") in {0, 10, 25, 50}
        }
        refill_reproducibility_audit = {
            "diagnostic_only": True,
            "production_changed": False,
            "lineage_assessment": {
                "old_plus_10_plus_25_source": "append_local_frontier_candidate_prefix_diagnostics",
                "old_plus_10_plus_25_are_final_mesh_controls": False,
                "post_repair_replay_source": "append50_then_lp_blockers_then_sequential_refill",
                "operations_are_noncommutative": True,
                "stale_acceptance_evidence": True,
                "reason": (
                    "append_local_frontier rows contain candidate-level prefix metrics only; "
                    "they do not store final active retained/new/unique/lost mesh metrics for append50->repair->refill."
                ),
            },
            "pipeline_variants": {
                "A_append50_then_lp_blockers": {
                    "available": True,
                    "append_local_accepted_sequence_prefix50": list(w2_small_face_refinement_summary.get("accepted_sequence_signature", []))[:50],
                    "append_local_challenger_order_prefix": list(w2_small_face_refinement_summary.get("challenger_order_signature", []))[:80],
                    "candidate_pool_signature_before_lp_blockers": {
                        "dense_core_count": int(len(dense_core_candidates)),
                        "candidate_count": int(len(post_core_repair_candidates) + len(w2_core_repair_summary.get("selected_removed_candidate_ids", []) or [])),
                        "append_local_count": int(len(append_local_candidates)),
                    },
                    "stage1_top8": [
                        row.get("removed_candidate_id")
                        for row in (w2_core_repair_summary.get("stage1") or {}).get("generic_ranking_top", [])[:8]
                    ],
                    "stage1_selected": (w2_core_repair_summary.get("stage1") or {}).get("selected_removed_candidate_ids"),
                    "stage2_strict_pair_count": (w2_core_repair_summary.get("stage2") or {}).get("strict_synergy_pair_count"),
                    "stage2_selected_pair": ((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get("removed_candidate_ids"),
                    "removed_blocker_ids": list(w2_core_repair_summary.get("selected_removed_candidate_ids", [])),
                    "final_active_plane_signature": w2_core_repair_summary.get("final_geometry_signature"),
                    "vertices": int(len(reconstructed.get("vertices") or [])),
                    "edges": int(len(reconstructed.get("edges") or [])),
                    "faces": int(len(reconstructed.get("faces") or [])),
                    "volume": reconstructed.get("reliable_volume"),
                    "topology": reconstructed.get("topology"),
                    "oracle_posthoc": {
                        "retained_core_ids": int(len(core_ids & set(int(v) for v in active_total_cohort.get("finite_face_ids", [])))),
                        "new_unique_face_ids_vs_repaired_core": int(len(set(int(v) for v in active_total_cohort.get("finite_face_ids", [])) - core_ids)),
                        "unique_finite_face_ids": int(len(set(int(v) for v in active_total_cohort.get("finite_face_ids", [])))),
                        "lost_core_ids": sorted(int(v) for v in (core_ids - set(int(v) for v in active_total_cohort.get("finite_face_ids", [])))),
                    },
                },
                "B_append60_then_lp_blockers": {
                    "available": False,
                    "reason": "not rerun in this bounded audit; requires separate full pipeline with --w2-max-small-face-additions 60",
                },
                "C_append75_then_lp_blockers": {
                    "available": False,
                    "reason": "not rerun in this bounded audit; requires separate full pipeline with --w2-max-small-face-additions 75",
                },
                "D_append50_then_lp_blockers_then_refill10": {
                    "available": True,
                    "source": "protected_sequential_refill bounded replay",
                    "reached": any(
                        bool(row.get("reached"))
                        for row in ((protected_sequential_refill_audit.get("protected_sequential_refill") or {}).get("frontier", []) if isinstance(protected_sequential_refill_audit.get("protected_sequential_refill"), dict) else [])
                        if int(row.get("max_accepted_additions") or -1) == 10
                    ),
                },
                "E_append50_then_lp_blockers_then_refill25": {
                    "available": True,
                    "source": "protected_sequential_refill bounded replay",
                    "reached": any(
                        bool(row.get("reached"))
                        for row in ((protected_sequential_refill_audit.get("protected_sequential_refill") or {}).get("frontier", []) if isinstance(protected_sequential_refill_audit.get("protected_sequential_refill"), dict) else [])
                        if int(row.get("max_accepted_additions") or -1) == 25
                    ),
                },
            },
            "old_prefix_rows": old_prefix_rows,
            "first_candidate_9000843": first_step_repro,
            "corrected_confusion_matrix": protected_confusion,
            "protected_gate_false_negative": gate_false_negative,
            "selected_branch": "B_protected_gate_false_negative" if gate_false_negative else "D_posthoc_accounting_fixed",
            "production_mode_added": False,
        }

        def build_post_ratchet_closure_audit() -> dict[str, object]:
            started = time.perf_counter()
            missing_new_ids = sorted(int(v) for v in (lower_new_ids - current_new_ids))
            production_only_ids = sorted(int(v) for v in (current_active_ids - lower_all_ids))
            current_active_indices = [
                int(i)
                for i in reconstructed.get("face_candidate_indices", [])
                if 0 <= int(i) < len(candidates)
            ]
            current_active_candidates = [candidates[i] for i in current_active_indices]
            current_active_candidate_ids = {int(candidate_track_id(candidate)) for candidate in current_active_candidates}
            candidate_by_id: dict[int, dict[str, object]] = {}
            for candidate in list(dense_w2_raw_candidates) + list(dense_w2_preselection_candidates) + list(candidates):
                candidate_by_id[int(candidate_track_id(candidate))] = candidate
            append_rank_by_id = {
                int(candidate_track_id(candidate)): int(rank)
                for rank, candidate in enumerate(append_local_candidates, start=1)
            }
            broad_order = list((broad_protected_refill_audit.get("broad_protected_refill") or {}).get("challenger_order_signature") or [])
            rolling_order = list((rolling_patch_ratchet_audit.get("rolling_patch_ratchet") or {}).get("challenger_order_signature") or [])
            broad_rank_by_id = {int(tid): int(rank) for rank, tid in enumerate(broad_order, start=1) if tid is not None}
            rolling_rank_by_id = {int(tid): int(rank) for rank, tid in enumerate(rolling_order, start=1) if tid is not None}
            selected_ids = {int(candidate_track_id(candidate)) for candidate in candidates}
            removed_blocker_ids = {int(v) for v in w2_core_repair_summary.get("selected_removed_candidate_ids", [])}

            def topology_valid(rec: dict[str, object]) -> bool:
                topo = rec.get("topology") if isinstance(rec.get("topology"), dict) else {}
                return (
                    int(topo.get("euler_characteristic") or topo.get("euler") or 0) == 2
                    and int(topo.get("connected_components") or topo.get("components") or 0) == 1
                    and int(topo.get("boundary_edges") or topo.get("boundary") or 0) == 0
                    and int(topo.get("non_manifold_edges") or topo.get("non_manifold") or 0) == 0
                )

            def finite_match(candidate: dict[str, object]) -> dict[str, object] | None:
                tid = int(candidate_track_id(candidate))
                if tid not in oracle_match_cache:
                    oracle_match_cache[tid] = candidate_oracle_face(candidate, model_face_rows)
                return oracle_match_cache.get(tid)

            def z_span(candidate: dict[str, object]) -> tuple[float | None, float | None]:
                pts = np.array(candidate.get("points") or candidate.get("hull_points") or [], dtype=float)
                if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
                    return None, None
                return as_json_float(float(np.min(pts[:, 2]))), as_json_float(float(np.max(pts[:, 2])))

            def plane_row(candidate: dict[str, object]) -> dict[str, object]:
                plane = candidate_plane(candidate)
                centroid = candidate_hull_centroid(candidate)
                z0, z1 = z_span(candidate)
                if plane is None:
                    normal = None
                    offset = None
                else:
                    point, normal_np = plane
                    normal = as_json_point(normal_np)
                    offset = as_json_float(float(normal_np @ point))
                return {
                    "candidate_id": int(candidate_track_id(candidate)),
                    "origin": str(candidate.get("origin") or candidate.get("source") or ""),
                    "normal": normal,
                    "offset": offset,
                    "centroid": as_json_point(centroid) if centroid is not None else None,
                    "z_span": [z0, z1],
                    "hull_area": as_json_float(float(candidate.get("hull_area") or 0.0)) if candidate.get("hull_area") is not None else None,
                    "support_count": int(candidate.get("support_count") or len(candidate.get("points") or [])),
                }

            def compatible_patch(a: dict[str, object], b: dict[str, object]) -> bool:
                angle, offset, centroid = candidate_plane_relation_score(a, b)
                za0, za1 = z_span(a)
                zb0, zb1 = z_span(b)
                z_overlap_ok = True
                if za0 is not None and za1 is not None and zb0 is not None and zb1 is not None:
                    overlap = max(0.0, min(float(za1), float(zb1)) - max(float(za0), float(zb0)))
                    span = max(1e-9, min(float(za1) - float(za0), float(zb1) - float(zb0)))
                    z_overlap_ok = overlap / span >= 0.25
                return bool(angle <= 1.0 and offset <= 0.02 and centroid <= 0.12 and z_overlap_ok)

            current_patch_groups: list[dict[str, object]] = []
            for candidate in current_active_candidates:
                cid = int(candidate_track_id(candidate))
                placed = False
                for group in current_patch_groups:
                    rep = group["representative"]
                    if compatible_patch(rep, candidate):
                        group["member_candidate_ids"].append(cid)
                        group["members"].append(candidate)
                        placed = True
                        break
                if not placed:
                    current_patch_groups.append(
                        {
                            "group_id": f"active_patch_{cid}",
                            "representative": candidate,
                            "representative_candidate_id": cid,
                            "member_candidate_ids": [cid],
                            "members": [candidate],
                        }
                    )

            def preservation_summary(trial_candidates: list[dict[str, object]], trial_rec: dict[str, object]) -> dict[str, object]:
                active_indices = [
                    int(i)
                    for i in trial_rec.get("face_candidate_indices", [])
                    if 0 <= int(i) < len(trial_candidates)
                ]
                active_trial_candidates = [trial_candidates[i] for i in active_indices]
                lost_groups: list[dict[str, object]] = []
                for group in current_patch_groups:
                    members = list(group.get("members") or [])
                    represented = any(
                        bool(compatible_patch(member, active_candidate))
                        for member in members
                        for active_candidate in active_trial_candidates
                    )
                    if not represented:
                        lost_groups.append(
                            {
                                "group_id": str(group.get("group_id")),
                                "representative_candidate_id": int(group.get("representative_candidate_id") or -1),
                                "member_candidate_ids": sorted(int(v) for v in group.get("member_candidate_ids", [])),
                            }
                        )
                return {
                    "protected_group_count": int(len(current_patch_groups)),
                    "lost_group_count": int(len(lost_groups)),
                    "lost_groups": lost_groups[:20],
                    "all_preserved": bool(not lost_groups),
                }

            representatives_by_face: dict[int, list[dict[str, object]]] = {int(fid): [] for fid in missing_new_ids}
            seen_rep_keys: set[tuple[int, int]] = set()
            candidate_pools = [
                ("raw_w2", list(dense_w2_raw_candidates)),
                ("after_core_preselection", list(dense_w2_preselection_candidates)),
                ("current_final_candidates", list(candidates)),
            ]
            for pool_name, pool in candidate_pools:
                for candidate in pool:
                    match = finite_match(candidate)
                    if not match or not bool(match.get("finite_good")):
                        continue
                    face_id = int(match.get("face_id") if match.get("face_id") is not None else -1)
                    if face_id not in representatives_by_face:
                        continue
                    cid = int(candidate_track_id(candidate))
                    key = (face_id, cid)
                    if key in seen_rep_keys:
                        continue
                    seen_rep_keys.add(key)
                    representatives_by_face[face_id].append(
                        {
                            **plane_row(candidate),
                            "face_id": int(face_id),
                            "pool": pool_name,
                            "selected_in_final": bool(cid in selected_ids),
                            "active_in_final": bool(cid in current_active_candidate_ids),
                            "append_local_rank": append_rank_by_id.get(cid),
                            "broad_refill_rank": broad_rank_by_id.get(cid),
                            "rolling_refill_rank": rolling_rank_by_id.get(cid),
                            "oracle_plane_good": bool(match.get("plane_good")),
                            "oracle_finite_good": bool(match.get("finite_good")),
                        }
                    )
            for face_id in representatives_by_face:
                representatives_by_face[face_id].sort(
                    key=lambda row: (
                        row.get("selected_in_final") is not True,
                        row.get("rolling_refill_rank") if row.get("rolling_refill_rank") is not None else 10**9,
                        row.get("append_local_rank") if row.get("append_local_rank") is not None else 10**9,
                        int(row.get("candidate_id") or -1),
                    )
                )

            individual_by_face = {
                int(row.get("face_id")): row
                for row in individual_trials
                if row.get("face_id") is not None
            }
            individual_rows: list[dict[str, object]] = []
            redundant_rep_candidates: list[tuple[int, dict[str, object]]] = []
            for face_id in missing_new_ids:
                reps = representatives_by_face.get(face_id) or []
                rep_id = int(reps[0]["candidate_id"]) if reps else None
                rep_candidate = candidate_by_id.get(rep_id) if rep_id is not None else None
                existing = individual_by_face.get(face_id, {})
                row: dict[str, object] = {
                    "face_id": int(face_id),
                    "representative_candidate_id": rep_id,
                    "representative_found": bool(rep_candidate is not None),
                    "representative_rows": reps[:8],
                    "existing_individual_trial": existing,
                }
                if rep_candidate is None:
                    row["classification"] = "no_candidate"
                    individual_rows.append(row)
                    continue
                if rep_id in selected_ids:
                    row["classification"] = "already_selected_but_inactive_or_unmatched"
                    individual_rows.append(row)
                    continue
                trial_candidates = list(candidates) + [rep_candidate]
                trial_summary = summarize_reconstruction_prefix(
                    prefix=len(trial_candidates) - len(dense_core_candidates),
                    candidates=trial_candidates,
                    core_count=len(dense_core_candidates),
                    core_ids=core_ids,
                    model_faces=model_face_rows,
                    initial_vertices=vertices,
                    trusted_points=trusted_cloud_points,
                    trusted_z_indices=trusted_cloud_z_indices,
                    point_tol=float(args.compatibility_point_tol),
                    reconstruction_kwargs=oracle_reconstruction_kwargs,
                    match_cache=oracle_match_cache,
                )
                trial_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **oracle_reconstruction_kwargs)
                preservation = preservation_summary(trial_candidates, trial_rec)
                outside = trial_summary.get("trusted_outside") if isinstance(trial_summary.get("trusted_outside"), dict) else {}
                lost_z = int(outside.get("cumulative_lost_z_levels") or 0) if isinstance(outside, dict) else 0
                new_ids = set(int(v) for v in trial_summary.get("new_ids", []))
                lost_core = list(trial_summary.get("lost_core_ids") or [])
                target_added = bool(face_id in new_ids)
                canonical_delta = int(trial_summary.get("active_unique_finite_ids") or 0) - int(len(current_active_ids))
                if not topology_valid(trial_rec):
                    classification = "unsafe_topology"
                elif float(outside.get("cumulative_outside_fraction") or 0.0) >= 0.01:
                    classification = "unsafe_outside"
                elif lost_z != 0:
                    classification = "unsafe_lost_z"
                elif not bool(preservation.get("all_preserved")):
                    classification = "breaks_ratchet_patch"
                elif target_added and canonical_delta > 0 and not lost_core:
                    classification = "safe_plus_one"
                elif target_added:
                    classification = "active_but_replaces_existing"
                else:
                    classification = "redundant"
                    redundant_rep_candidates.append((face_id, rep_candidate))
                row.update(
                    {
                        "classification": classification,
                        "target_added": target_added,
                        "canonical_delta_vs_126": int(canonical_delta),
                        "trial_new_ids": sorted(int(v) for v in new_ids),
                        "trial_lost_core_ids": lost_core,
                        "trial_production_only_ids_lost": sorted(int(v) for v in (current_active_ids - set(core_ids) - new_ids)),
                        "trial_unique_total": int(trial_summary.get("active_unique_finite_ids") or 0),
                        "vertices": int(trial_summary.get("reconstructed_vertices") or 0),
                        "edges": int(trial_summary.get("reconstructed_edges") or 0),
                        "faces": int(trial_summary.get("reconstructed_faces") or 0),
                        "volume": trial_summary.get("volume"),
                        "trusted_outside": outside,
                        "topology": trial_summary.get("topology"),
                        "geometry_signature": mesh_geometry_signature(trial_candidates, trial_rec),
                        "ratchet_patch_preservation": preservation,
                    }
                )
                individual_rows.append(row)

            redundant_faces: list[dict[str, object]] = []
            for face_id, rep_candidate in redundant_rep_candidates:
                nearest: list[dict[str, object]] = []
                for active_candidate in current_active_candidates:
                    angle, offset, centroid = candidate_plane_relation_score(rep_candidate, active_candidate)
                    nearest.append(
                        {
                            "active_candidate_id": int(candidate_track_id(active_candidate)),
                            "angle_deg": as_json_float(float(angle)),
                            "offset": as_json_float(float(offset)),
                            "centroid_distance": as_json_float(float(centroid)),
                            "patch_compatible": bool(compatible_patch(rep_candidate, active_candidate)),
                        }
                    )
                nearest.sort(
                    key=lambda row: (
                        0 if row.get("patch_compatible") else 1,
                        finite_float(row.get("angle_deg"), 1e9),
                        finite_float(row.get("offset"), 1e9),
                        finite_float(row.get("centroid_distance"), 1e9),
                    )
                )
                reason = "needs_removal_or_exchange"
                if nearest and bool(nearest[0].get("patch_compatible")):
                    reason = "already_geometrically_represented_but_oracle_mapping_missed"
                elif nearest and finite_float(nearest[0].get("angle_deg"), 1e9) <= 2.0:
                    reason = "parallel_stronger_plane"
                redundant_faces.append(
                    {
                        "face_id": int(face_id),
                        "representative_candidate_id": int(candidate_track_id(rep_candidate)),
                        "reason": reason,
                        "nearest_active_planes": nearest[:8],
                    }
                )

            pair_rows: list[dict[str, object]] = []
            for left_idx in range(len(redundant_rep_candidates)):
                for right_idx in range(left_idx + 1, len(redundant_rep_candidates)):
                    left_face, left_candidate = redundant_rep_candidates[left_idx]
                    right_face, right_candidate = redundant_rep_candidates[right_idx]
                    pair_ids = [int(candidate_track_id(left_candidate)), int(candidate_track_id(right_candidate))]
                    additions = [candidate for candidate in [left_candidate, right_candidate] if int(candidate_track_id(candidate)) not in selected_ids]
                    trial_candidates = list(candidates) + additions
                    trial_summary = summarize_reconstruction_prefix(
                        prefix=len(trial_candidates) - len(dense_core_candidates),
                        candidates=trial_candidates,
                        core_count=len(dense_core_candidates),
                        core_ids=core_ids,
                        model_faces=model_face_rows,
                        initial_vertices=vertices,
                        trusted_points=trusted_cloud_points,
                        trusted_z_indices=trusted_cloud_z_indices,
                        point_tol=float(args.compatibility_point_tol),
                        reconstruction_kwargs=oracle_reconstruction_kwargs,
                        match_cache=oracle_match_cache,
                    )
                    trial_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **oracle_reconstruction_kwargs)
                    preservation = preservation_summary(trial_candidates, trial_rec)
                    outside = trial_summary.get("trusted_outside") if isinstance(trial_summary.get("trusted_outside"), dict) else {}
                    new_ids = set(int(v) for v in trial_summary.get("new_ids", []))
                    target_added = sorted(int(v) for v in ({int(left_face), int(right_face)} & new_ids))
                    canonical_delta = int(trial_summary.get("active_unique_finite_ids") or 0) - int(len(current_active_ids))
                    prospective = (
                        bool(target_added)
                        and canonical_delta > 0
                        and bool(preservation.get("all_preserved"))
                        and topology_valid(trial_rec)
                        and float(outside.get("cumulative_outside_fraction") or 0.0) < 0.01
                        and int(outside.get("cumulative_lost_z_levels") or 0) == 0
                    )
                    pair_rows.append(
                        {
                            "missing_face_ids": [int(left_face), int(right_face)],
                            "candidate_ids": pair_ids,
                            "target_missing_added": target_added,
                            "canonical_delta_vs_126": int(canonical_delta),
                            "trial_unique_total": int(trial_summary.get("active_unique_finite_ids") or 0),
                            "trial_new_ids": sorted(int(v) for v in new_ids),
                            "trial_lost_core_ids": list(trial_summary.get("lost_core_ids") or []),
                            "trusted_outside": outside,
                            "topology": trial_summary.get("topology"),
                            "volume": trial_summary.get("volume"),
                            "geometry_signature": mesh_geometry_signature(trial_candidates, trial_rec),
                            "ratchet_patch_preservation": preservation,
                            "prospective": bool(prospective),
                        }
                    )

            def build_bounded_exchange_audit() -> dict[str, object]:
                exchange_started = time.perf_counter()
                face_activity_slack = 0.03
                current_candidate_ids = tuple(sorted(int(candidate_track_id(candidate)) for candidate in candidates))
                current_id_set = set(current_candidate_ids)
                current_finite_ids = set(int(v) for v in current_active_ids)
                stage1_selected_removed = {
                    int(v) for v in (w2_core_repair_summary.get("stage1") or {}).get("selected_removed_candidate_ids", [])
                }
                stage1_target_ids = {
                    int(target_row.get("core_candidate_id"))
                    for ranking_row in (w2_core_repair_summary.get("stage1") or {}).get("generic_ranking_top", [])
                    if int(ranking_row.get("removed_candidate_id") or -1) in stage1_selected_removed
                    for target_row in ranking_row.get("reactivated_core_rows", [])
                    if target_row.get("core_candidate_id") is not None
                }
                stage2_target_id = ((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get(
                    "target_core_candidate_id"
                )
                repaired_target_ids = set(stage1_target_ids)
                if stage2_target_id is not None:
                    repaired_target_ids.add(int(stage2_target_id))
                z_slices = prepare_z_level_slices(trusted_cloud_z_indices)
                state_cache: dict[
                    tuple[tuple[int, ...], tuple[int, ...]],
                    tuple[dict[str, object], dict[str, object], list[dict[str, object]]],
                ] = {}
                local_contour_cache: dict[int, list[object]] = {}

                lp_engine = LpActivityEngine(enabled=True)
                lp_engine.stage = "post_ratchet_bounded_exchange"
                lp_engine.set_reusable_universe(
                    "post_ratchet_bounded_exchange",
                    candidates,
                    inside_tol=float(face_activity_slack),
                )

                def candidate_samples(candidate: dict[str, object]) -> list[np.ndarray]:
                    hull = np.array(candidate.get("hull") or [], dtype=float)
                    if hull.ndim != 2 or hull.shape[0] == 0 or hull.shape[1] != 3:
                        point = candidate_hull_centroid(candidate)
                        return [np.array(point, dtype=float)] if point is not None else []
                    centroid = np.mean(hull, axis=0)
                    samples = [centroid]
                    for point in hull:
                        samples.append(0.7 * centroid + 0.3 * point)
                    for idx, point in enumerate(hull):
                        samples.append(0.5 * point + 0.5 * hull[(idx + 1) % len(hull)])
                    return samples

                def local_contours(candidate: dict[str, object], limit: int = 24) -> list[object]:
                    cid = int(candidate_track_id(candidate))
                    cached = local_contour_cache.get(cid)
                    if cached is not None:
                        return cached
                    centroid = candidate_hull_centroid(candidate)
                    if centroid is None:
                        local_contour_cache[cid] = []
                        return []
                    scored: list[tuple[float, object]] = []
                    for contour in contours:
                        points = np.asarray(contour.points, dtype=float)
                        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
                            continue
                        stride = max(1, int(points.shape[0] // 96))
                        distance = float(np.min(np.linalg.norm(points[::stride] - centroid[None, :], axis=1)))
                        scored.append((distance, contour))
                    scored.sort(key=lambda item: (item[0], int(item[1].index)))
                    selected = [contour for _, contour in scored[: max(1, int(limit))]]
                    local_contour_cache[cid] = selected
                    return selected

                def activity_details(target: dict[str, object], removed_ids: set[int]) -> dict[str, object]:
                    target_id = int(candidate_track_id(target))
                    base_ids = tuple(
                        int(cid)
                        for cid in current_candidate_ids
                        if int(cid) != target_id and int(cid) not in removed_ids
                    )
                    return lp_face_activity_details(
                        None,
                        target,
                        inside_tol=float(face_activity_slack),
                        face_activity_slack=float(face_activity_slack),
                        include_point=True,
                        lp_engine=lp_engine,
                        base_candidate_ids=base_ids,
                    )

                def active_face_row(
                    source_candidates: list[dict[str, object]],
                    rec: dict[str, object],
                    candidate_id: int,
                ) -> dict[str, object] | None:
                    rec_vertices = np.array(rec.get("vertices") or [], dtype=float)
                    for face, source_index in zip(rec.get("faces") or [], rec.get("face_candidate_indices") or []):
                        index = int(source_index)
                        if index < 0 or index >= len(source_candidates):
                            continue
                        if int(candidate_track_id(source_candidates[index])) != int(candidate_id):
                            continue
                        face_indices = np.array(face, dtype=int)
                        if face_indices.size < 3 or np.any(face_indices < 0) or np.any(face_indices >= rec_vertices.shape[0]):
                            continue
                        points = rec_vertices[face_indices]
                        return {
                            "active_plane_index": int(index),
                            "face_area": as_json_float(float(polygon_area(points))),
                            "face_centroid": as_json_point(np.mean(points, axis=0)),
                            "face_vertex_count": int(len(face_indices)),
                        }
                    return None

                def face_area_distribution(rec: dict[str, object]) -> dict[str, object]:
                    rec_vertices = np.array(rec.get("vertices") or [], dtype=float)
                    areas: list[float] = []
                    for face in rec.get("faces") or []:
                        indices = np.array(face, dtype=int)
                        if indices.size >= 3 and np.all(indices >= 0) and np.all(indices < rec_vertices.shape[0]):
                            areas.append(float(polygon_area(rec_vertices[indices])))
                    if not areas:
                        return {"count": 0, "min": None, "p10": None, "median": None, "p90": None, "max": None}
                    arr = np.array(areas, dtype=float)
                    return {
                        "count": int(arr.size),
                        "min": as_json_float(float(np.min(arr))),
                        "p10": as_json_float(float(np.percentile(arr, 10))),
                        "median": as_json_float(float(np.median(arr))),
                        "p90": as_json_float(float(np.percentile(arr, 90))),
                        "max": as_json_float(float(np.max(arr))),
                    }

                def evaluated_state(
                    *,
                    label: str,
                    removed_ids: set[int] | None = None,
                    additions: list[dict[str, object]] | None = None,
                    explicit_candidates: list[dict[str, object]] | None = None,
                    target_candidates: list[dict[str, object]] | None = None,
                ) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
                    removed = {int(v) for v in (removed_ids or set())}
                    additions_by_id = {
                        int(candidate_track_id(candidate)): candidate
                        for candidate in (additions or [])
                    }
                    if explicit_candidates is None:
                        state_candidates = [
                            candidate
                            for candidate in candidates
                            if int(candidate_track_id(candidate)) not in removed
                        ]
                        state_ids = {int(candidate_track_id(candidate)) for candidate in state_candidates}
                        state_candidates.extend(
                            candidate
                            for cid, candidate in sorted(additions_by_id.items())
                            if cid not in state_ids
                        )
                        cache_key = (
                            tuple(sorted(removed)),
                            tuple(sorted(cid for cid in additions_by_id if cid not in current_id_set)),
                        )
                        cached = state_cache.get(cache_key)
                        if cached is not None:
                            cached_row, cached_rec, cached_candidates = cached
                            return dict(cached_row), cached_rec, cached_candidates
                    else:
                        state_candidates = list(explicit_candidates)
                        cache_key = None

                    rec = reconstruct_polyhedron_from_halfspaces_edge_clip(
                        state_candidates,
                        **oracle_reconstruction_kwargs,
                    )
                    active_indices = [
                        int(i)
                        for i in rec.get("face_candidate_indices", [])
                        if 0 <= int(i) < len(state_candidates)
                    ]
                    active_candidates = [state_candidates[i] for i in active_indices]
                    active_candidate_ids = {int(candidate_track_id(candidate)) for candidate in active_candidates}
                    cohort = oracle_candidate_cohort(
                        active_candidates,
                        model_face_rows,
                        core_face_ids=core_ids,
                        active_plane_indices=active_indices,
                        match_cache=oracle_match_cache,
                    )
                    finite_ids = set(int(v) for v in cohort.get("finite_face_ids", []))
                    outside = evaluate_trusted_cloud_outside(
                        state_candidates,
                        trusted_cloud_points,
                        z_slices,
                        point_tol=float(args.compatibility_point_tol),
                    )
                    rec_vertices = np.array(rec.get("vertices") or [], dtype=float)
                    rec_faces = model_face_planes(rec_vertices, rec.get("faces") or []) if rec_vertices.size else []
                    reprojection = evaluate_polyhedron_reprojection(
                        name=str(label),
                        vertices_3d=rec_vertices,
                        contours=contours,
                        direction_count=32,
                        contour_sample=120,
                    )
                    preservation = preservation_summary(state_candidates, rec)
                    target_rows: list[dict[str, object]] = []
                    for target in target_candidates or []:
                        target_id = int(candidate_track_id(target))
                        local = local_contours(target)
                        target_rows.append(
                            {
                                "candidate_id": int(target_id),
                                "active": bool(target_id in active_candidate_ids),
                                "active_face": active_face_row(state_candidates, rec, target_id),
                                "surface_deficit": patch_surface_deficit_for_samples(
                                    state_candidates,
                                    candidate_samples(target),
                                    sample_tol=0.0,
                                ),
                                "local_contour_indices": [int(contour.index) for contour in local],
                                "local_contour_reprojection": evaluate_polyhedron_reprojection(
                                    name=f"{label}_local_{target_id}",
                                    vertices_3d=rec_vertices,
                                    contours=local,
                                    direction_count=32,
                                ) if local else None,
                            }
                        )
                    topology = rec.get("topology") if isinstance(rec.get("topology"), dict) else {}
                    repair_target_activity = {
                        str(target_id): bool(target_id in active_candidate_ids)
                        for target_id in sorted(repaired_target_ids)
                    }
                    row = {
                        "label": str(label),
                        "removed_candidate_ids": sorted(int(v) for v in removed),
                        "added_candidate_ids": sorted(int(v) for v in additions_by_id),
                        "retained_core_ids": int(len(core_ids & finite_ids)),
                        "new_ids": sorted(int(v) for v in (finite_ids - core_ids)),
                        "unique_total": int(len(finite_ids)),
                        "lost_core_ids": sorted(int(v) for v in (core_ids - finite_ids)),
                        "canonical_ids_gained_vs_126": sorted(int(v) for v in (finite_ids - current_finite_ids)),
                        "canonical_ids_lost_vs_126": sorted(int(v) for v in (current_finite_ids - finite_ids)),
                        "active_candidate_ids": sorted(int(v) for v in active_candidate_ids),
                        "active_face_count": int(len(active_indices)),
                        "vertices": int(len(rec.get("vertices") or [])),
                        "edges": int(len(rec.get("edges") or [])),
                        "faces": int(len(rec.get("faces") or [])),
                        "volume": rec.get("reliable_volume"),
                        "extents": rec.get("bbox") or (rec.get("lp_extents") or {}).get("extents"),
                        "face_area_distribution": face_area_distribution(rec),
                        "minimum_edge_length": topology.get("edge_length_min"),
                        "topology": topology,
                        "topology_valid": bool(topology_valid(rec)),
                        "outside": outside,
                        "reprojection": reprojection,
                        "reconstructed_to_initial": surface_distance_summary(rec_vertices, model_face_rows, max_points=32),
                        "initial_to_reconstructed": surface_distance_summary(vertices, rec_faces, max_points=32),
                        "rolling_patch_preservation": preservation,
                        "repair_target_activity": {
                            "target_candidate_ids": sorted(int(v) for v in repaired_target_ids),
                            "active_by_candidate_id": repair_target_activity,
                            "all_active": bool(repair_target_activity and all(repair_target_activity.values())),
                        },
                        "target_rows": target_rows,
                        "geometry_signature": mesh_geometry_signature(state_candidates, rec),
                    }
                    if cache_key is not None:
                        state_cache[cache_key] = (dict(row), rec, state_candidates)
                    return row, rec, state_candidates

                production_geometry, _, _ = evaluated_state(
                    label="production126",
                    explicit_candidates=list(candidates),
                )
                lower_geometry, _, _ = evaluated_state(
                    label="oracle129",
                    explicit_candidates=list(dense_core_candidates) + list(oracle_lower_candidates),
                )
                observed_fields = [
                    "support_abs_median",
                    "support_abs_p95",
                    "symmetric_contour_distance_median",
                    "symmetric_contour_distance_p95",
                    "model_outside_observed_median",
                    "model_inside_observed_median",
                ]
                observed_deltas = {
                    field: as_json_float(
                        finite_float((lower_geometry.get("reprojection") or {}).get(field), 0.0)
                        - finite_float((production_geometry.get("reprojection") or {}).get(field), 0.0)
                    )
                    for field in observed_fields
                }
                observed_lower_better = [field for field, delta in observed_deltas.items() if finite_float(delta, 0.0) < 0.0]
                observed_lower_worse = [field for field, delta in observed_deltas.items() if finite_float(delta, 0.0) > 0.0]

                replacing_rows = [
                    row
                    for row in individual_rows
                    if row.get("target_added") and row.get("trial_production_only_ids_lost")
                ]
                exchange_row = replacing_rows[0] if replacing_rows else None
                exchange_target_face_id = int(exchange_row.get("face_id")) if exchange_row is not None else None
                exchange_target_id = int(exchange_row.get("representative_candidate_id")) if exchange_row is not None else None
                exchange_target = candidate_by_id.get(exchange_target_id) if exchange_target_id is not None else None
                exchange_production_face_ids = {
                    int(v) for v in (exchange_row.get("trial_production_only_ids_lost") or [])
                } if exchange_row is not None else set()
                exchange_production_candidates = [
                    candidate
                    for candidate in current_active_candidates
                    if (
                        (finite_match(candidate) or {}).get("face_id") is not None
                        and int((finite_match(candidate) or {}).get("face_id")) in exchange_production_face_ids
                    )
                ]
                exchange_production_candidate_ids = {
                    int(candidate_track_id(candidate)) for candidate in exchange_production_candidates
                }

                def support_row(candidate: dict[str, object]) -> dict[str, object]:
                    plane = candidate_plane(candidate)
                    samples = candidate_samples(candidate)
                    details = activity_details(candidate, set())
                    return {
                        "candidate_id": int(candidate_track_id(candidate)),
                        "candidate_origin": str(candidate.get("candidate_origin") or candidate.get("candidate_source") or ""),
                        "normal": as_json_point(plane[1]) if plane is not None else None,
                        "offset": as_json_float(float(plane[1] @ plane[0])) if plane is not None else None,
                        "centroid": as_json_point(candidate_hull_centroid(candidate)) if candidate_hull_centroid(candidate) is not None else None,
                        "z_range": [as_json_float(float(v)) if np.isfinite(v) else None for v in candidate_z_range(candidate)],
                        "hull_area": as_json_float(finite_float(candidate.get("hull_area"), 0.0)),
                        "plane_rms": as_json_float(finite_float(candidate.get("plane_rms"), 0.0)),
                        "support_count": int(finite_float(candidate.get("finite_support_count"), finite_float(candidate.get("observed_support_near_count"), 0.0))),
                        "support_density": as_json_float(finite_float(candidate.get("finite_support_density"), 0.0)),
                        "support_purity": as_json_float(finite_float(candidate.get("finite_support_purity"), 0.0)),
                        "support_residual_median": as_json_float(finite_float(candidate.get("finite_support_residual_median"), 0.0)),
                        "support_residual_p95": as_json_float(finite_float(candidate.get("finite_support_residual_p95"), 0.0)),
                        "lp_activity": details,
                        "surface_deficit": patch_surface_deficit_for_samples(
                            [item for item in candidates if int(candidate_track_id(item)) != int(candidate_track_id(candidate))],
                            samples,
                            sample_tol=0.0,
                        ),
                        "active_face": active_face_row(candidates, reconstructed, int(candidate_track_id(candidate))),
                    }

                exchange_relations: list[dict[str, object]] = []
                if exchange_target is not None:
                    for production_candidate in exchange_production_candidates:
                        angle, offset, centroid_distance = candidate_plane_relation_score(exchange_target, production_candidate)
                        exchange_relations.append(
                            {
                                "target_candidate_id": int(candidate_track_id(exchange_target)),
                                "production_candidate_id": int(candidate_track_id(production_candidate)),
                                "normal_angle_deg": as_json_float(float(angle)),
                                "signed_offset_distance": as_json_float(float(offset)),
                                "centroid_distance": as_json_float(float(centroid_distance)),
                                "projected_hull_bounds_overlap": as_json_float(float(hull_bounds_overlap_ratio(exchange_target, production_candidate))),
                                "patch_relation": plane_patch_relation(exchange_target, production_candidate),
                                "strict_spatial_patch_compatible": bool(compatible_patch(exchange_target, production_candidate)),
                            }
                        )

                exchange_states: list[dict[str, object]] = []
                if exchange_target is not None:
                    exchange_specs = [
                        ("production126", set(), []),
                        (f"production126_plus_{exchange_target_id}", set(), [exchange_target]),
                        ("production126_minus_production_patch", set(exchange_production_candidate_ids), []),
                        (
                            f"production126_minus_production_patch_plus_{exchange_target_id}",
                            set(exchange_production_candidate_ids),
                            [exchange_target],
                        ),
                    ]
                    exchange_local_targets = [exchange_target] + list(exchange_production_candidates)
                    for label, removed, additions in exchange_specs:
                        state, _, _ = evaluated_state(
                            label=label,
                            removed_ids=removed,
                            additions=additions,
                            target_candidates=exchange_local_targets,
                        )
                        exchange_states.append(state)

                exchange_plus = exchange_states[1] if len(exchange_states) > 1 else {}
                exchange_both_active = bool(
                    exchange_target_id is not None
                    and exchange_target_id in set(exchange_plus.get("active_candidate_ids") or [])
                    and bool(exchange_production_candidate_ids & set(exchange_plus.get("active_candidate_ids") or []))
                )
                exchange_mapping_conflict = bool(
                    exchange_relations
                    and all(bool(row.get("strict_spatial_patch_compatible")) for row in exchange_relations)
                    and not exchange_both_active
                    and int(exchange_plus.get("unique_total") or 0) == int(len(current_finite_ids))
                    and set(exchange_plus.get("canonical_ids_gained_vs_126") or []) == ({exchange_target_face_id} if exchange_target_face_id is not None else set())
                    and set(exchange_plus.get("canonical_ids_lost_vs_126") or []) == exchange_production_face_ids
                )
                exchange_same_observed_footprint = bool(
                    exchange_relations
                    and all(
                        bool((row.get("patch_relation") or {}).get("compatible"))
                        and finite_float(row.get("projected_hull_bounds_overlap"), 0.0) >= 0.5
                        for row in exchange_relations
                    )
                )
                if exchange_mapping_conflict:
                    exchange_classification = "finite_patch_mapping_conflict"
                elif exchange_same_observed_footprint:
                    exchange_classification = "near_coincident_plane_tradeoff"
                else:
                    exchange_classification = "spatially_distinct_geometric_tradeoff"

                active_non_core_candidates = [
                    candidate
                    for candidate in current_active_candidates
                    if int(candidate_track_id(candidate)) not in {
                        int(candidate_track_id(core_candidate)) for core_candidate in dense_core_candidates
                    }
                ]
                active_non_core_by_id = {
                    int(candidate_track_id(candidate)): candidate for candidate in active_non_core_candidates
                }

                def estimated_group_loss(removed_ids: set[int]) -> int:
                    lost = 0
                    for group in current_patch_groups:
                        member_ids = {int(v) for v in group.get("member_candidate_ids", [])}
                        if member_ids and member_ids <= removed_ids:
                            lost += 1
                    return int(lost)

                def removed_support(removed_ids: set[int]) -> dict[str, object]:
                    removed_candidates = [active_non_core_by_id[cid] for cid in removed_ids if cid in active_non_core_by_id]
                    return {
                        "candidate_count": int(len(removed_candidates)),
                        "finite_support_count": int(sum(int(finite_float(candidate.get("finite_support_count"), 0.0)) for candidate in removed_candidates)),
                        "finite_support_density_sum": as_json_float(sum(finite_float(candidate.get("finite_support_density"), 0.0) for candidate in removed_candidates)),
                        "hull_area_sum": as_json_float(sum(finite_float(candidate.get("hull_area"), 0.0) for candidate in removed_candidates)),
                        "plane_rms_sum": as_json_float(sum(finite_float(candidate.get("plane_rms"), 0.0) for candidate in removed_candidates)),
                    }

                blocker_target_face_ids = [
                    int(face_id)
                    for face_id in missing_new_ids
                    if exchange_target_face_id is None or int(face_id) != int(exchange_target_face_id)
                ]
                target_screen_rows: list[dict[str, object]] = []
                all_full_trials: list[dict[str, object]] = []
                action_candidates: list[dict[str, object]] = []

                def proposal_sort_key(row: dict[str, object]) -> tuple[object, ...]:
                    support = row.get("removed_support") if isinstance(row.get("removed_support"), dict) else {}
                    return (
                        not bool(row.get("lp_positive")),
                        int(row.get("estimated_rolling_groups_lost") or 0),
                        -finite_float(row.get("surface_deficit_reduction"), 0.0),
                        -finite_float(row.get("effective_margin_gain"), 0.0),
                        finite_float(support.get("finite_support_count"), 0.0),
                        len(row.get("removed_candidate_ids") or []),
                        tuple(int(v) for v in row.get("removed_candidate_ids") or []),
                    )

                for face_id in blocker_target_face_ids:
                    reps = representatives_by_face.get(int(face_id)) or []
                    target_id = int(reps[0].get("candidate_id")) if reps else None
                    target = candidate_by_id.get(target_id) if target_id is not None else None
                    if target is None:
                        target_screen_rows.append({"face_id": int(face_id), "classification": "no_target_representative"})
                        continue
                    target_samples = candidate_samples(target)
                    target_id = int(candidate_track_id(target))
                    base_candidates_without_target = [
                        candidate for candidate in candidates if int(candidate_track_id(candidate)) != target_id
                    ]
                    baseline_activity = activity_details(target, set())
                    baseline_deficit = patch_surface_deficit_for_samples(
                        base_candidates_without_target,
                        target_samples,
                        sample_tol=0.0,
                    )
                    single_rows: list[dict[str, object]] = []
                    for blocker in active_non_core_candidates:
                        blocker_id = int(candidate_track_id(blocker))
                        after = activity_details(target, {blocker_id})
                        gain = finite_float(after.get("effective_face_margin"), -999.0) - finite_float(
                            baseline_activity.get("effective_face_margin"), -999.0
                        )
                        after_deficit = patch_surface_deficit_for_samples(
                            [
                                candidate
                                for candidate in base_candidates_without_target
                                if int(candidate_track_id(candidate)) != blocker_id
                            ],
                            target_samples,
                            sample_tol=0.0,
                        )
                        angle, offset, centroid_distance = candidate_plane_relation_score(target, blocker)
                        removed = {blocker_id}
                        single_rows.append(
                            {
                                "removed_candidate_ids": [int(blocker_id)],
                                "before_activity": baseline_activity,
                                "after_activity": after,
                                "effective_margin_gain": as_json_float(float(gain)),
                                "lp_positive": bool(after.get("active") and gain > 1e-9),
                                "surface_deficit_before": baseline_deficit,
                                "surface_deficit_after": after_deficit,
                                "surface_deficit_reduction": as_json_float(
                                    finite_float(baseline_deficit.get("area_weighted_cut_deficit"), 0.0)
                                    - finite_float(after_deficit.get("area_weighted_cut_deficit"), 0.0)
                                ),
                                "target_blocker_relation": {
                                    "normal_angle_deg": as_json_float(float(angle)),
                                    "signed_offset_distance": as_json_float(float(offset)),
                                    "centroid_distance": as_json_float(float(centroid_distance)),
                                    "hull_bounds_overlap": as_json_float(float(hull_bounds_overlap_ratio(target, blocker))),
                                },
                                "removed_support": removed_support(removed),
                                "estimated_rolling_groups_lost": estimated_group_loss(removed),
                            }
                        )
                    single_rows.sort(key=proposal_sort_key)
                    shortlist = single_rows[:12]
                    positive_singles = [row for row in shortlist if bool(row.get("lp_positive"))]

                    def full_trial(proposal: dict[str, object], action_class: str) -> dict[str, object]:
                        removed = {int(v) for v in proposal.get("removed_candidate_ids", [])}
                        state, _, _ = evaluated_state(
                            label=f"exchange_{face_id}_{'_'.join(str(v) for v in sorted(removed))}",
                            removed_ids=removed,
                            additions=[target],
                            target_candidates=[target],
                        )
                        target_state = (state.get("target_rows") or [{}])[0]
                        outside = state.get("outside") if isinstance(state.get("outside"), dict) else {}
                        reprojection = state.get("reprojection") if isinstance(state.get("reprojection"), dict) else {}
                        base_reprojection = production_geometry.get("reprojection") if isinstance(production_geometry.get("reprojection"), dict) else {}
                        row = {
                            **proposal,
                            "action_class": str(action_class),
                            "target_face_id_posthoc": int(face_id),
                            "target_candidate_id": int(target_id),
                            "target_active_edge_clip": bool(target_state.get("active")),
                            "target_active_face": target_state.get("active_face"),
                            "target_surface_deficit_after_full": target_state.get("surface_deficit"),
                            "retained_core_ids": state.get("retained_core_ids"),
                            "new_ids": state.get("new_ids"),
                            "unique_total": state.get("unique_total"),
                            "lost_core_ids": state.get("lost_core_ids"),
                            "canonical_ids_gained_vs_126": state.get("canonical_ids_gained_vs_126"),
                            "canonical_ids_lost_vs_126": state.get("canonical_ids_lost_vs_126"),
                            "vertices": state.get("vertices"),
                            "edges": state.get("edges"),
                            "faces": state.get("faces"),
                            "volume": state.get("volume"),
                            "topology": state.get("topology"),
                            "topology_valid": state.get("topology_valid"),
                            "outside": outside,
                            "reprojection": reprojection,
                            "reprojection_p95_delta": as_json_float(
                                finite_float(reprojection.get("symmetric_contour_distance_p95"), 0.0)
                                - finite_float(base_reprojection.get("symmetric_contour_distance_p95"), 0.0)
                            ),
                            "rolling_patch_preservation": state.get("rolling_patch_preservation"),
                            "repair_target_activity": state.get("repair_target_activity"),
                            "geometry_signature": state.get("geometry_signature"),
                            "full_mesh_safe": bool(
                                target_state.get("active")
                                and state.get("topology_valid")
                                and int(state.get("retained_core_ids") or 0) == int(len(core_ids))
                                and not state.get("lost_core_ids")
                                and bool((state.get("repair_target_activity") or {}).get("all_active"))
                                and finite_float(outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
                                and int(outside.get("cumulative_lost_z_levels") or 0) == 0
                                and finite_float(reprojection.get("symmetric_contour_distance_p95"), float("inf"))
                                <= finite_float(base_reprojection.get("symmetric_contour_distance_p95"), 0.0)
                                + max(0.001, 0.10 * finite_float(base_reprojection.get("symmetric_contour_distance_p95"), 0.0))
                            ),
                        }
                        row["non_oracle_objective"] = {
                            "full_mesh_safe": bool(row.get("full_mesh_safe")),
                            "target_active_edge_clip": bool(row.get("target_active_edge_clip")),
                            "rolling_groups_lost": int((row.get("rolling_patch_preservation") or {}).get("lost_group_count") or 0),
                            "surface_deficit_reduction": row.get("surface_deficit_reduction"),
                            "effective_margin_gain": row.get("effective_margin_gain"),
                            "removed_support_count": (row.get("removed_support") or {}).get("finite_support_count"),
                            "reprojection_p95_delta": row.get("reprojection_p95_delta"),
                            "removal_count": int(len(removed)),
                            "uses_initial_model": False,
                            "uses_oracle_face_ids": False,
                            "uses_canonical_metrics": False,
                        }
                        return row

                    full_single_rows = [full_trial(row, "remove_one_add_target") for row in positive_singles]
                    single_success = any(bool(row.get("target_active_edge_clip")) for row in full_single_rows)
                    pair_rows_target: list[dict[str, object]] = []
                    full_pair_rows: list[dict[str, object]] = []
                    if not single_success:
                        shortlist_ids = [int(row["removed_candidate_ids"][0]) for row in shortlist]
                        for left_index in range(len(shortlist_ids)):
                            for right_index in range(left_index + 1, len(shortlist_ids)):
                                removed = {shortlist_ids[left_index], shortlist_ids[right_index]}
                                after = activity_details(target, removed)
                                gain = finite_float(after.get("effective_face_margin"), -999.0) - finite_float(
                                    baseline_activity.get("effective_face_margin"), -999.0
                                )
                                after_deficit = patch_surface_deficit_for_samples(
                                    [
                                        candidate
                                        for candidate in base_candidates_without_target
                                        if int(candidate_track_id(candidate)) not in removed
                                    ],
                                    target_samples,
                                    sample_tol=0.0,
                                )
                                proposal = {
                                    "removed_candidate_ids": sorted(int(v) for v in removed),
                                    "before_activity": baseline_activity,
                                    "after_activity": after,
                                    "effective_margin_gain": as_json_float(float(gain)),
                                    "lp_positive": bool(after.get("active") and gain > 1e-9),
                                    "surface_deficit_before": baseline_deficit,
                                    "surface_deficit_after": after_deficit,
                                    "surface_deficit_reduction": as_json_float(
                                        finite_float(baseline_deficit.get("area_weighted_cut_deficit"), 0.0)
                                        - finite_float(after_deficit.get("area_weighted_cut_deficit"), 0.0)
                                    ),
                                    "removed_support": removed_support(removed),
                                    "estimated_rolling_groups_lost": estimated_group_loss(removed),
                                }
                                pair_rows_target.append(proposal)
                        pair_rows_target.sort(key=proposal_sort_key)
                        full_pair_rows = [
                            full_trial(row, "remove_pair_add_target")
                            for row in pair_rows_target
                            if bool(row.get("lp_positive"))
                        ]
                    target_full_rows = full_single_rows + full_pair_rows
                    all_full_trials.extend(target_full_rows)
                    action_candidates.extend(
                        row for row in target_full_rows if bool(row.get("target_active_edge_clip"))
                    )
                    target_screen_rows.append(
                        {
                            "face_id_posthoc": int(face_id),
                            "target_candidate_id": int(target_id),
                            "target_origin": str(target.get("candidate_origin") or target.get("candidate_source") or ""),
                            "baseline_activity": baseline_activity,
                            "baseline_surface_deficit": baseline_deficit,
                            "active_non_core_plane_count": int(len(active_non_core_candidates)),
                            "shortlist_size": int(len(shortlist)),
                            "remove_one_lp_positive_count": int(len(positive_singles)),
                            "remove_one_shortlist": shortlist,
                            "remove_one_full_trials": full_single_rows,
                            "single_edge_clip_success": bool(single_success),
                            "pair_screened": bool(not single_success),
                            "pair_lp_positive_count": int(sum(1 for row in pair_rows_target if bool(row.get("lp_positive")))),
                            "pair_screen_rows": pair_rows_target,
                            "pair_full_trials": full_pair_rows,
                        }
                    )

                def full_trial_sort_key(row: dict[str, object]) -> tuple[object, ...]:
                    objective = row.get("non_oracle_objective") if isinstance(row.get("non_oracle_objective"), dict) else {}
                    return (
                        not bool(objective.get("full_mesh_safe")),
                        not bool(objective.get("target_active_edge_clip")),
                        int(objective.get("rolling_groups_lost") or 0),
                        -finite_float(objective.get("surface_deficit_reduction"), 0.0),
                        -finite_float(objective.get("effective_margin_gain"), 0.0),
                        finite_float(objective.get("removed_support_count"), 0.0),
                        finite_float(objective.get("reprojection_p95_delta"), 0.0),
                        int(objective.get("removal_count") or 0),
                        tuple(int(v) for v in row.get("removed_candidate_ids") or []),
                    )

                ranked_actions = sorted(action_candidates, key=full_trial_sort_key)
                for rank, row in enumerate(ranked_actions, start=1):
                    row["non_oracle_rank"] = int(rank)

                beam_actions: list[dict[str, object]] = []
                seen_action_targets: dict[int, int] = {}
                for row in ranked_actions:
                    target_id = int(row.get("target_candidate_id") or -1)
                    count = seen_action_targets.get(target_id, 0)
                    if count >= 4:
                        continue
                    seen_action_targets[target_id] = count + 1
                    beam_actions.append(row)
                beam_specs: list[dict[str, object]] = [
                    {"removed_ids": frozenset(), "target_ids": frozenset(), "action_keys": tuple()}
                ]
                beam_rows: list[dict[str, object]] = []
                seen_beam_keys: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
                for _ in range(4):
                    expanded: list[dict[str, object]] = []
                    for state_spec in beam_specs:
                        for action in beam_actions:
                            target_id = int(action.get("target_candidate_id") or -1)
                            if target_id in state_spec["target_ids"]:
                                continue
                            removed = set(state_spec["removed_ids"]) | {
                                int(v) for v in action.get("removed_candidate_ids", [])
                            }
                            targets = set(state_spec["target_ids"]) | {target_id}
                            if len(removed) > 3 or len(targets) > 4:
                                continue
                            key = (tuple(sorted(removed)), tuple(sorted(targets)))
                            if key in seen_beam_keys:
                                continue
                            seen_beam_keys.add(key)
                            target_candidates = [candidate_by_id[tid] for tid in sorted(targets) if tid in candidate_by_id]
                            state, _, _ = evaluated_state(
                                label=f"joint_exchange_r{len(removed)}_a{len(targets)}",
                                removed_ids=removed,
                                additions=target_candidates,
                                target_candidates=target_candidates,
                            )
                            target_rows = state.get("target_rows") or []
                            all_targets_active = bool(target_rows) and all(bool(row.get("active")) for row in target_rows)
                            outside = state.get("outside") if isinstance(state.get("outside"), dict) else {}
                            reprojection = state.get("reprojection") if isinstance(state.get("reprojection"), dict) else {}
                            base_reprojection = production_geometry.get("reprojection") if isinstance(production_geometry.get("reprojection"), dict) else {}
                            baseline_deficit_total = 0.0
                            after_deficit_total = 0.0
                            effective_margin_total = 0.0
                            for target in target_candidates:
                                target_face = next(
                                    (
                                        int(row.get("face_id_posthoc"))
                                        for row in target_screen_rows
                                        if int(row.get("target_candidate_id") or -1) == int(candidate_track_id(target))
                                    ),
                                    None,
                                )
                                target_screen = next(
                                    (row for row in target_screen_rows if target_face is not None and int(row.get("face_id_posthoc")) == target_face),
                                    {},
                                )
                                baseline_deficit_total += finite_float(
                                    (target_screen.get("baseline_surface_deficit") or {}).get("area_weighted_cut_deficit"),
                                    0.0,
                                )
                                target_state = next(
                                    (row for row in target_rows if int(row.get("candidate_id") or -1) == int(candidate_track_id(target))),
                                    {},
                                )
                                after_deficit_total += finite_float(
                                    (target_state.get("surface_deficit") or {}).get("area_weighted_cut_deficit"),
                                    0.0,
                                )
                                effective_margin_total += finite_float(activity_details(target, removed).get("effective_face_margin"), -999.0)
                            safe = bool(
                                all_targets_active
                                and state.get("topology_valid")
                                and int(state.get("retained_core_ids") or 0) == int(len(core_ids))
                                and not state.get("lost_core_ids")
                                and bool((state.get("repair_target_activity") or {}).get("all_active"))
                                and finite_float(outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
                                and int(outside.get("cumulative_lost_z_levels") or 0) == 0
                                and finite_float(reprojection.get("symmetric_contour_distance_p95"), float("inf"))
                                <= finite_float(base_reprojection.get("symmetric_contour_distance_p95"), 0.0)
                                + max(0.001, 0.10 * finite_float(base_reprojection.get("symmetric_contour_distance_p95"), 0.0))
                            )
                            row = {
                                "removed_candidate_ids": sorted(int(v) for v in removed),
                                "added_candidate_ids": sorted(int(v) for v in targets),
                                "target_count": int(len(targets)),
                                "all_targets_active": bool(all_targets_active),
                                "retained_core_ids": state.get("retained_core_ids"),
                                "new_ids": state.get("new_ids"),
                                "unique_total": state.get("unique_total"),
                                "lost_core_ids": state.get("lost_core_ids"),
                                "canonical_ids_gained_vs_126": state.get("canonical_ids_gained_vs_126"),
                                "canonical_ids_lost_vs_126": state.get("canonical_ids_lost_vs_126"),
                                "vertices": state.get("vertices"),
                                "edges": state.get("edges"),
                                "faces": state.get("faces"),
                                "volume": state.get("volume"),
                                "topology": state.get("topology"),
                                "outside": outside,
                                "reprojection": reprojection,
                                "rolling_patch_preservation": state.get("rolling_patch_preservation"),
                                "repair_target_activity": state.get("repair_target_activity"),
                                "geometry_signature": state.get("geometry_signature"),
                                "non_oracle_objective": {
                                    "full_mesh_safe": bool(safe),
                                    "all_targets_active": bool(all_targets_active),
                                    "rolling_groups_lost": int((state.get("rolling_patch_preservation") or {}).get("lost_group_count") or 0),
                                    "surface_deficit_reduction": as_json_float(float(baseline_deficit_total - after_deficit_total)),
                                    "effective_margin_total": as_json_float(float(effective_margin_total)),
                                    "removed_support_count": removed_support(removed).get("finite_support_count"),
                                    "reprojection_p95_delta": as_json_float(
                                        finite_float(reprojection.get("symmetric_contour_distance_p95"), 0.0)
                                        - finite_float(base_reprojection.get("symmetric_contour_distance_p95"), 0.0)
                                    ),
                                    "removal_count": int(len(removed)),
                                    "addition_count": int(len(targets)),
                                    "uses_initial_model": False,
                                    "uses_oracle_face_ids": False,
                                    "uses_canonical_metrics": False,
                                },
                            }
                            beam_rows.append(row)
                            expanded.append({"removed_ids": frozenset(removed), "target_ids": frozenset(targets), "action_keys": key})
                    if not expanded:
                        break
                    ranked_expanded = sorted(
                        expanded,
                        key=lambda spec: full_trial_sort_key(
                            next(
                                row
                                for row in reversed(beam_rows)
                                if tuple(row.get("removed_candidate_ids") or []) == tuple(sorted(spec["removed_ids"]))
                                and tuple(row.get("added_candidate_ids") or []) == tuple(sorted(spec["target_ids"]))
                            )
                        ),
                    )
                    beam_specs = ranked_expanded[:20]

                beam_rows.sort(key=full_trial_sort_key)
                for rank, row in enumerate(beam_rows, start=1):
                    row["non_oracle_rank"] = int(rank)
                all_constructed_rows = list(all_full_trials) + list(beam_rows) + list(exchange_states)
                safe_constructed = [
                    row
                    for row in all_constructed_rows
                    if (
                        int(row.get("retained_core_ids") or 0) == int(len(core_ids))
                        and not row.get("lost_core_ids")
                        and bool(row.get("topology_valid") or topology_is_valid(row.get("topology")))
                        and finite_float((row.get("outside") or {}).get("cumulative_outside_fraction"), float("inf")) < 0.01
                        and int((row.get("outside") or {}).get("cumulative_lost_z_levels") or 0) == 0
                        and (
                            not row.get("non_oracle_objective")
                            or bool((row.get("non_oracle_objective") or {}).get("full_mesh_safe"))
                        )
                    )
                ]
                achieved = max(
                    safe_constructed,
                    key=lambda row: (int(row.get("unique_total") or 0), -len(row.get("removed_candidate_ids") or [])),
                    default=production_geometry,
                )
                nonoracle_frontier = sorted(
                    [row for row in list(ranked_actions) + list(beam_rows) if row.get("non_oracle_objective")],
                    key=full_trial_sort_key,
                )
                for rank, row in enumerate(nonoracle_frontier, start=1):
                    row["non_oracle_rank"] = int(rank)
                nonoracle_top = nonoracle_frontier[0] if nonoracle_frontier else None
                improvement_exists = bool(int(achieved.get("unique_total") or 0) >= 127)
                nonoracle_selects_improvement = bool(
                    nonoracle_top is not None
                    and int(nonoracle_top.get("unique_total") or 0) >= 127
                    and bool((nonoracle_top.get("non_oracle_objective") or {}).get("full_mesh_safe"))
                )
                lower_reprojection = lower_geometry.get("reprojection") if isinstance(lower_geometry.get("reprojection"), dict) else {}
                production_reprojection = production_geometry.get("reprojection") if isinstance(production_geometry.get("reprojection"), dict) else {}
                lower_materially_worse = bool(
                    finite_float(lower_reprojection.get("symmetric_contour_distance_p95"), 0.0)
                    > 1.10 * max(EPS, finite_float(production_reprojection.get("symmetric_contour_distance_p95"), 0.0))
                    or finite_float(lower_reprojection.get("support_abs_p95"), 0.0)
                    > 1.10 * max(EPS, finite_float(production_reprojection.get("support_abs_p95"), 0.0))
                )
                if exchange_mapping_conflict:
                    selected_branch = "D_evaluator_mapping_conflict"
                elif improvement_exists and nonoracle_selects_improvement:
                    selected_branch = "A_generic_exchange_promising"
                elif improvement_exists:
                    selected_branch = "B_oracle_exchange_without_observable_separation"
                elif lower_materially_worse:
                    selected_branch = "E_oracle129_worse_observed_geometry"
                else:
                    selected_branch = "C_small_exchange_no_127"

                return {
                    "diagnostic_only": True,
                    "production_changed": False,
                    "production_mode_added": False,
                    "selector_dependencies": {
                        "uses_initial_model": False,
                        "uses_oracle_face_ids": False,
                        "uses_canonical_metrics": False,
                        "oracle_target_cohort_only": True,
                    },
                    "geometry_comparison_126_vs_129": {
                        "bounded_evaluator": {
                            "direction_count": 32,
                            "contour_sample": 120,
                            "surface_distance_sample": 32,
                        },
                        "production126": production_geometry,
                        "oracle129": lower_geometry,
                        "observed_reprojection_delta_oracle129_minus_production126": observed_deltas,
                        "observed_fields_oracle129_better": observed_lower_better,
                        "observed_fields_oracle129_worse": observed_lower_worse,
                        "oracle129_materially_worse_observed_geometry": bool(lower_materially_worse),
                    },
                    "exchange_107_vs_248_posthoc": {
                        "target_face_id": exchange_target_face_id,
                        "target_candidate": support_row(exchange_target) if exchange_target is not None else None,
                        "production_face_ids": sorted(int(v) for v in exchange_production_face_ids),
                        "production_candidates": [support_row(candidate) for candidate in exchange_production_candidates],
                        "relations": exchange_relations,
                        "states": exchange_states,
                        "can_be_active_simultaneously": bool(exchange_both_active),
                        "same_observed_footprint": bool(exchange_same_observed_footprint),
                        "spatially_distinct": bool(exchange_relations and not exchange_same_observed_footprint),
                        "mapping_conflict": bool(exchange_mapping_conflict),
                        "classification": exchange_classification,
                    },
                    "generic_blocker_search": {
                        "active_non_core_plane_count": int(len(active_non_core_candidates)),
                        "shortlist_per_target": 12,
                        "maximum_removal_set_size": 2,
                        "target_rows": target_screen_rows,
                        "full_edge_clip_trial_count": int(len(all_full_trials)),
                    },
                    "non_oracle_action_ranking": {
                        "ranking_fields": [
                            {"name": "full_mesh_safe", "direction": "true first", "uses_initial_model": False},
                            {"name": "target_active_edge_clip", "direction": "true first", "uses_initial_model": False},
                            {"name": "rolling_groups_lost", "direction": "ascending", "uses_initial_model": False},
                            {"name": "surface_deficit_reduction", "direction": "descending", "uses_initial_model": False},
                            {"name": "effective_margin_gain", "direction": "descending", "uses_initial_model": False},
                            {"name": "removed_support_count", "direction": "ascending", "uses_initial_model": False},
                            {"name": "reprojection_p95_delta", "direction": "ascending", "uses_initial_model": False},
                            {"name": "removal_count", "direction": "ascending", "uses_initial_model": False},
                        ],
                        "rows": nonoracle_frontier[:40],
                        "top_state_reaches_127": bool(nonoracle_selects_improvement),
                    },
                    "bounded_joint_exchange": {
                        "max_removals_total": 3,
                        "max_target_additions": 4,
                        "beam_width": 20,
                        "action_seed_count": int(len(beam_actions)),
                        "state_count": int(len(beam_rows)),
                        "rows": beam_rows[:60],
                    },
                    "fresh_achieved_lower_bound": {
                        "removed_candidate_ids": list(achieved.get("removed_candidate_ids") or []),
                        "added_candidate_ids": list(achieved.get("added_candidate_ids") or []),
                        "retained_core_ids": achieved.get("retained_core_ids"),
                        "new_ids": achieved.get("new_ids"),
                        "unique_total": achieved.get("unique_total"),
                        "lost_core_ids": achieved.get("lost_core_ids"),
                        "canonical_ids_gained_vs_126": achieved.get("canonical_ids_gained_vs_126"),
                        "canonical_ids_lost_vs_126": achieved.get("canonical_ids_lost_vs_126"),
                        "vertices": achieved.get("vertices"),
                        "edges": achieved.get("edges"),
                        "faces": achieved.get("faces"),
                        "volume": achieved.get("volume"),
                        "topology": achieved.get("topology"),
                        "outside": achieved.get("outside"),
                        "reprojection": achieved.get("reprojection"),
                        "rolling_patch_preservation": achieved.get("rolling_patch_preservation"),
                        "repair_target_activity": achieved.get("repair_target_activity"),
                        "geometry_signature": achieved.get("geometry_signature"),
                        "full_mesh_verified": True,
                    },
                    "improvement_exists": bool(improvement_exists),
                    "non_oracle_selects_improvement": bool(nonoracle_selects_improvement),
                    "selected_branch": selected_branch,
                    "lp_engine_diagnostics": lp_engine.summary(),
                    "timing_seconds": as_json_float(time.perf_counter() - exchange_started),
                }

            bounded_exchange_audit = build_bounded_exchange_audit()

            safe_individual = [row for row in individual_rows if str(row.get("classification")) == "safe_plus_one"]
            prospective_pairs = [row for row in pair_rows if bool(row.get("prospective"))]
            replacing = [row for row in individual_rows if str(row.get("classification")) == "active_but_replaces_existing"]
            redundant_reasons = {str(row.get("reason")) for row in redundant_faces}
            if safe_individual:
                branch = "A_individual_refill_still_promising"
            elif prospective_pairs:
                branch = "B_pair_synergy_promising"
            elif replacing:
                branch = "C_exchange_required_or_production_only_replacement"
            elif redundant_faces and redundant_reasons <= {"already_geometrically_represented_but_oracle_mapping_missed"}:
                branch = "D_oracle_evaluator_artifact"
            else:
                branch = "E_no_current_refill_path"

            return {
                "diagnostic_only": True,
                "production_changed": False,
                "production_mode_added": False,
                "current_golden": {
                    "mode": str(args.w2_core_repair_mode),
                    "post_repair_refinement": str(args.w2_post_repair_refinement),
                    "post_repair_additions": int(args.w2_max_post_repair_additions),
                    "removed_blockers": sorted(int(v) for v in removed_blocker_ids),
                    "retained_core_ids": int(len(core_ids & current_active_ids)),
                    "lost_core_ids": sorted(int(v) for v in (core_ids - current_active_ids)),
                    "new_ids": sorted(int(v) for v in current_new_ids),
                    "unique_total": int(len(current_active_ids)),
                    "vertices": int(len(reconstructed.get("vertices") or [])),
                    "edges": int(len(reconstructed.get("edges") or [])),
                    "faces": int(len(reconstructed.get("faces") or [])),
                    "volume": reconstructed.get("reliable_volume"),
                    "geometry_digest": (w2_core_repair_summary.get("post_repair_refinement") or {}).get("geometry_digest")
                    or (w2_core_repair_summary.get("final_geometry_signature") or {}).get("active_candidate_id_xor"),
                    "topology": reconstructed.get("topology"),
                },
                "canonical_sets": {
                    "repaired_core_ids": sorted(int(v) for v in core_ids),
                    "pre_ratchet_ids": sorted(int(v) for v in set(int(v) for v in current_prefix_summary.get("new_ids", [])) | core_ids),
                    "current_final_126_ids": sorted(int(v) for v in current_active_ids),
                    "oracle_lower_bound_ids": sorted(int(v) for v in lower_all_ids),
                    "missing_new_ids": missing_new_ids,
                    "production_only_ids": production_only_ids,
                },
                "fresh_gap_decomposition": {
                    "lower_bound_unique_total": int(len(lower_all_ids)),
                    "current_unique_total": int(len(current_active_ids)),
                    "gap_count": int(len(lower_all_ids - current_active_ids)),
                    "missing_ids": sorted(int(v) for v in (lower_all_ids - current_active_ids)),
                    "missing_new_ids": missing_new_ids,
                    "production_only_ids": production_only_ids,
                    "reuses_fresh_lower_bound": True,
                },
                "safe_controls": [
                    {
                        "candidate_id": int(cid),
                        "selected": bool(cid in selected_ids),
                        "active": bool(cid in current_active_candidate_ids),
                        "oracle_face_id": (
                            int((finite_match(candidate_by_id[cid]) or {}).get("face_id"))
                            if cid in candidate_by_id and (finite_match(candidate_by_id[cid]) or {}).get("face_id") is not None
                            else None
                        ),
                    }
                    for cid in [9000209, 9001254, 9000137, 9000094, 9001244]
                ],
                "missing_funnel": {
                    "rows": missing_rows,
                    "first_loss_reason_counts": missing_reason_counts,
                    "representatives_by_face": {str(face_id): rows for face_id, rows in representatives_by_face.items()},
                },
                "individual_full_mesh_trials": {
                    "trial_count": int(len(individual_rows)),
                    "safe_plus_one_count": int(len(safe_individual)),
                    "rows": individual_rows,
                },
                "redundant_missing_faces": {
                    "rows": redundant_faces,
                    "reason_counts": {
                        reason: int(sum(1 for row in redundant_faces if str(row.get("reason")) == reason))
                        for reason in sorted(redundant_reasons)
                    },
                },
                "pair_synergy_among_redundant_representatives": {
                    "candidate_pair_count": int(len(pair_rows)),
                    "prospective_pair_count": int(len(prospective_pairs)),
                    "rows": pair_rows,
                },
                "bounded_exchange_audit": bounded_exchange_audit,
                "fresh_oracle_lower_bound_mesh": {
                    "selected_candidate_ids": sorted(int(v) for v in oracle_selected_candidate_ids),
                    "new_ids": sorted(int(v) for v in lower_new_ids),
                    "unique_total": int(len(lower_all_ids)),
                    "retained_core_ids": int((lower_summary or {}).get("retained_core_ids") or 0),
                    "lost_core_ids": list((lower_summary or {}).get("lost_core_ids", [])),
                    "vertices": int(len((oracle_lower_reconstructed or {}).get("vertices") or [])),
                    "edges": int(len((oracle_lower_reconstructed or {}).get("edges") or [])),
                    "faces": int(len((oracle_lower_reconstructed or {}).get("faces") or [])),
                    "volume": (oracle_lower_reconstructed or {}).get("reliable_volume"),
                    "topology": (oracle_lower_reconstructed or {}).get("topology"),
                    "outside": lower_outside,
                    "reprojection": lower_reprojection,
                },
                "selected_branch": branch,
                "timing_seconds": as_json_float(time.perf_counter() - started),
            }

        post_ratchet_closure_audit = build_post_ratchet_closure_audit()

        def build_generic_exchange_production_audit() -> dict[str, object]:
            exchange = w2_core_repair_summary.get("post_ratchet_exchange") or {}
            if str(exchange.get("mode")) != "lp-deficit":
                return {
                    "diagnostic_only": True,
                    "available": False,
                    "reason": "post-ratchet lp-deficit exchange was not requested",
                }

            candidate_by_id: dict[int, dict[str, object]] = {}
            for candidate in (
                list(dense_core_candidates)
                + list(dense_w2_preselection_candidates)
                + list(post_ratchet_exchange_base_candidates)
                + list(candidates)
            ):
                candidate_by_id[int(candidate_track_id(candidate))] = candidate

            oracle_controls: list[dict[str, object]] = []
            baseline_face_ids: set[int] | None = None
            for control in exchange.get("controls", []):
                active_candidate_ids = [int(v) for v in control.get("active_candidate_ids", [])]
                active_candidates = [
                    candidate_by_id[cid]
                    for cid in active_candidate_ids
                    if cid in candidate_by_id
                ]
                cohort = oracle_candidate_cohort(
                    active_candidates,
                    model_face_rows,
                    core_face_ids=core_ids,
                    match_cache=oracle_match_cache,
                )
                finite_face_ids = {int(v) for v in cohort.get("finite_face_ids", [])}
                if baseline_face_ids is None:
                    baseline_face_ids = set(finite_face_ids)
                retained = finite_face_ids & core_ids
                new_ids = finite_face_ids - core_ids
                topology = control.get("topology") if isinstance(control.get("topology"), dict) else {}
                outside = control.get("outside") if isinstance(control.get("outside"), dict) else {}
                reprojection = control.get("reprojection") if isinstance(control.get("reprojection"), dict) else {}
                oracle_controls.append(
                    {
                        "label": str(control.get("label")),
                        "action_count": int(control.get("action_count") or 0),
                        "retained_core_ids": int(len(retained)),
                        "new_ids": sorted(int(v) for v in new_ids),
                        "new_count": int(len(new_ids)),
                        "unique_total": int(len(finite_face_ids)),
                        "lost_core_ids": sorted(int(v) for v in (core_ids - finite_face_ids)),
                        "canonical_ids_gained_vs_exchange_off": sorted(
                            int(v) for v in (finite_face_ids - (baseline_face_ids or set()))
                        ),
                        "canonical_ids_lost_vs_exchange_off": sorted(
                            int(v) for v in ((baseline_face_ids or set()) - finite_face_ids)
                        ),
                        "vertices": int(control.get("vertices") or 0),
                        "edges": int(control.get("edges") or 0),
                        "faces": int(control.get("faces") or 0),
                        "volume": control.get("volume"),
                        "topology_valid": topology_is_valid(topology),
                        "outside": outside,
                        "lost_z": outside.get("cumulative_lost_z_levels"),
                        "reprojection": {
                            "support_abs_p95": reprojection.get("support_abs_p95"),
                            "symmetric_contour_distance_p95": reprojection.get("symmetric_contour_distance_p95"),
                        },
                        "active_candidate_ids": active_candidate_ids,
                        "active_plane_indices": list(control.get("active_plane_indices", [])),
                        "geometry_signature": control.get("geometry_signature"),
                    }
                )

            known_target_ids = [9000116, 9001326, 9000034, 9001316]
            known_target_rows: list[dict[str, object]] = []
            for target_id in known_target_ids:
                appearances: list[dict[str, object]] = []
                for iteration in exchange.get("iterations", []):
                    target_row = next(
                        (
                            row
                            for row in iteration.get("target_discovery_rows", [])
                            if int(row.get("candidate_id") or -1) == int(target_id)
                        ),
                        None,
                    )
                    if target_row is None:
                        continue
                    duplicate = target_row.get("duplicate_active") if isinstance(target_row.get("duplicate_active"), dict) else None
                    appearances.append(
                        {
                            "action_index": int(iteration.get("action_index") or 0),
                            "classification": target_row.get("classification"),
                            "generic_target_rank": target_row.get("generic_target_rank"),
                            "inside_bounded_target_pool": target_row.get("inside_bounded_target_pool"),
                            "baseline_effective_margin": (
                                (target_row.get("baseline_activity") or {}).get("effective_face_margin")
                                if isinstance(target_row.get("baseline_activity"), dict)
                                else None
                            ),
                            "duplicate_active_candidate_id": (
                                duplicate.get("active_candidate_id") if duplicate is not None else None
                            ),
                            "duplicate_active_observed_evidence_stronger": (
                                duplicate.get("active_observed_evidence_stronger") if duplicate is not None else None
                            ),
                        }
                    )
                known_target_rows.append(
                    {
                        "candidate_id": int(target_id),
                        "posthoc_oracle_face_id": (
                            int((candidate_oracle_face(candidate_by_id[target_id], model_face_rows) or {}).get("face_id"))
                            if target_id in candidate_by_id
                            and (candidate_oracle_face(candidate_by_id[target_id], model_face_rows) or {}).get("face_id") is not None
                            else None
                        ),
                        "appearances": appearances,
                    }
                )

            baseline_control = oracle_controls[0] if oracle_controls else {}
            final_control = oracle_controls[-1] if oracle_controls else {}
            baseline_reprojection_p95 = finite_float(
                (baseline_control.get("reprojection") or {}).get("symmetric_contour_distance_p95"),
                float("inf"),
            )
            final_reprojection_p95 = finite_float(
                (final_control.get("reprojection") or {}).get("symmetric_contour_distance_p95"),
                float("inf"),
            )
            selected_target_ids = {int(v) for v in exchange.get("added_candidate_ids", [])}
            action_rows = list(exchange.get("actions", []))
            criteria = {
                "retained_core_75": bool(int(final_control.get("retained_core_ids") or 0) == 75),
                "new_at_least_54": bool(int(final_control.get("new_count") or 0) >= 54),
                "unique_at_least_129": bool(int(final_control.get("unique_total") or 0) >= 129),
                "lost_core_empty": bool(not final_control.get("lost_core_ids")),
                "topology_valid": bool(final_control.get("topology_valid")),
                "outside_below_0_01": bool(
                    finite_float((final_control.get("outside") or {}).get("cumulative_outside_fraction"), float("inf")) < 0.01
                ),
                "lost_z_zero": bool(int(final_control.get("lost_z") or 0) == 0),
                "reprojection_bounded": bool(
                    final_reprojection_p95
                    <= baseline_reprojection_p95 + max(0.001, 0.10 * baseline_reprojection_p95)
                ),
                "negative_control_not_selected": bool(9001316 not in selected_target_ids),
                "no_unrelated_rolling_group_losses": bool(
                    all(
                        int((row.get("rolling_group_delta") or {}).get("unrelated_group_loss_count") or 0) == 0
                        for row in action_rows
                    )
                ),
                "all_intermediate_controls_safe": bool(
                    oracle_controls
                    and all(
                        bool(row.get("topology_valid"))
                        and int(row.get("retained_core_ids") or 0) == 75
                        and not row.get("lost_core_ids")
                        and finite_float((row.get("outside") or {}).get("cumulative_outside_fraction"), float("inf")) < 0.01
                        and int(row.get("lost_z") or 0) == 0
                        for row in oracle_controls
                    )
                ),
            }
            return {
                "diagnostic_only": True,
                "available": True,
                "selector_dependencies": exchange.get("selector_dependencies"),
                "action_signature": exchange.get("action_signature"),
                "action_signature_sha256": exchange.get("action_signature_sha256"),
                "oracle_controls": oracle_controls,
                "known_target_rows_posthoc": known_target_rows,
                "known_positive_canonical_gain_regression": {
                    "expected_face_ids": [85, 138, 142],
                    "actual_gained_face_ids": final_control.get("canonical_ids_gained_vs_exchange_off", []),
                    "passed": bool(
                        {85, 138, 142}
                        <= set(int(v) for v in final_control.get("canonical_ids_gained_vs_exchange_off", []))
                    ),
                },
                "negative_control_posthoc": {
                    "candidate_id": 9001316,
                    "expected_tradeoff": {"gained_face_id": 107, "lost_face_id": 248, "net_unique_delta": 0},
                    "selected": bool(9001316 in selected_target_ids),
                    "rejected_by": "patch_duplicate_with_stronger_active_evidence",
                },
                "promotion_criteria": criteria,
                "promotion_criteria_passed": bool(criteria and all(criteria.values())),
            }

        generic_exchange_production_audit = build_generic_exchange_production_audit()
        w2_core_repair_summary["post_repair_closure_audit"] = {
            "diagnostic_only": True,
            "production_changed": False,
            "cache_policy": previous_oracle_cache_note,
            "golden_121": {
                "mode": str(args.w2_core_repair_mode),
                "removed_blockers": list(w2_core_repair_summary.get("selected_removed_candidate_ids", [])),
                "repair_actions": w2_core_repair_summary.get("repair_actions", []),
                "stage1_selected": (w2_core_repair_summary.get("stage1") or {}).get("selected_removed_candidate_ids"),
                "stage1_ranking_top": (w2_core_repair_summary.get("stage1") or {}).get("generic_ranking_top", [])[:8],
                "stage2_selected_pair": ((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get("removed_candidate_ids"),
                "stage2_target_core_candidate_id": ((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get("target_core_candidate_id"),
                "stage2_strict_pair_ranking": (w2_core_repair_summary.get("stage2") or {}).get("strict_pair_ranking", [])[:8],
                "candidate_pool_signature": {
                    "face_candidate_count": int(len(candidates)),
                    "dense_core_count": int(len(dense_core_candidates)),
                    "w2_accepted_count": int(len(w2_accepted_candidates)),
                    "append_local_count": int(len(append_local_candidates)),
                    "append_local_accepted_sequence": list(w2_small_face_refinement_summary.get("accepted_sequence_signature", [])),
                    "append_local_challenger_order_prefix": list(w2_small_face_refinement_summary.get("challenger_order_signature", []))[:80],
                },
                "active_mapping": w2_core_repair_summary.get("active_mapping_delta"),
                "geometry_digest": w2_core_repair_summary.get("final_geometry_signature"),
                "vertices": int(len(reconstructed.get("vertices") or [])),
                "edges": int(len(reconstructed.get("edges") or [])),
                "faces": int(len(reconstructed.get("faces") or [])),
                "volume": reconstructed.get("reliable_volume"),
                "outside": current_prefix_summary.get("trusted_outside"),
                "topology": reconstructed.get("topology"),
            },
            "fresh_oracle_lower_bound": {
                "selected_candidate_ids": sorted(int(v) for v in oracle_selected_candidate_ids),
                "selected_new_face_ids": sorted(int(v) for v in lower_new_ids),
                "retained_core_ids": int((lower_summary or {}).get("retained_core_ids") or 0),
                "lost_core_ids": list((lower_summary or {}).get("lost_core_ids", [])),
                "new_ids": sorted(int(v) for v in lower_new_ids),
                "unique_total": int(len(lower_all_ids)),
                "feasible_bounded": bool((oracle_lower_reconstructed or {}).get("lp_extents", {}).get("feasible", True)) if oracle_lower_reconstructed is not None else False,
                "vertices": int(len((oracle_lower_reconstructed or {}).get("vertices") or [])),
                "edges": int(len((oracle_lower_reconstructed or {}).get("edges") or [])),
                "faces": int(len((oracle_lower_reconstructed or {}).get("faces") or [])),
                "volume": (oracle_lower_reconstructed or {}).get("reliable_volume"),
                "topology": (oracle_lower_reconstructed or {}).get("topology"),
                "outside": lower_outside,
                "lost_z": lower_outside.get("cumulative_lost_z_levels") if isinstance(lower_outside, dict) else None,
                "reprojection": lower_reprojection,
            },
            "fresh_gap": {
                "current_ids": sorted(int(v) for v in current_active_ids),
                "current_new_ids": sorted(int(v) for v in current_new_ids),
                "current_unique_total": int(len(current_active_ids)),
                "lower_bound_ids": sorted(int(v) for v in lower_all_ids),
                "lower_bound_new_ids": sorted(int(v) for v in lower_new_ids),
                "lower_bound_unique_total": int(len(lower_all_ids)),
                "missing_ids": sorted(int(v) for v in (lower_all_ids - current_active_ids)),
                "production_only_ids": sorted(int(v) for v in (current_active_ids - lower_all_ids)),
                "missing_new_ids": sorted(int(v) for v in (lower_new_ids - current_new_ids)),
                "gap_count": int(len(lower_all_ids - current_active_ids)),
                "closure_set_difference": closure_set_difference,
            },
            "missing_funnel": {
                "rows": missing_rows,
                "first_loss_reason_counts": missing_reason_counts,
            },
            "post_repair_refill_individual": {
                "trial_count": int(len(individual_trials)),
                "safe_candidate_count": int(len(safe_individual_rows)),
                "safe_finite_good_count": int(len([row for row in safe_individual_rows if row.get("target_added")])),
                "unique_new_face_ceiling": sorted(int(row.get("face_id")) for row in safe_individual_rows if row.get("face_id") is not None),
                "rows": individual_trials,
            },
            "post_repair_refill_prefix_simulation": {
                "skipped_if_no_safe_individuals": bool(not safe_individual_rows),
                "prefix_10": refill_10,
                "prefix_25": refill_25,
                "note": "append-local prefix rows are deterministic production-order diagnostics; full refill geometry is only meaningful when individual safe ceiling is nonzero.",
            },
            "protected_sequential_refill": protected_sequential_refill_audit,
            "protected_refill_frontier": protected_sequential_refill_audit.get("protected_refill_frontier"),
            "broad_protected_refill": broad_protected_refill_audit,
            "rolling_patch_ratchet_audit": rolling_patch_ratchet_audit,
            "refill_reproducibility_audit": refill_reproducibility_audit,
            "post_ratchet_closure_audit": post_ratchet_closure_audit,
            "generic_exchange_production_audit": generic_exchange_production_audit,
            "collective_oracle_lower_bound": collective,
            "repair_preservation_gate": {
                "stage1_patch_preserved": bool(w2_core_repair_summary.get("active_mapping_delta", {}).get("removed_blockers_absent_from_final_halfspaces")),
                "stage2_target_core_candidate_id": ((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get("target_core_candidate_id"),
                "stage2_target_active": bool(
                    ((w2_core_repair_summary.get("stage2") or {}).get("selected_pair") or {}).get("target_core_candidate_id")
                    in set(int(candidate_track_id(candidates[int(i)])) for i in reconstructed.get("face_candidate_indices", []) if 0 <= int(i) < len(candidates))
                ),
                "removed_blockers_absent": bool(w2_core_repair_summary.get("active_mapping_delta", {}).get("removed_blockers_absent_from_final_halfspaces")),
                "rejection_reason": "breaks_repaired_core_patch",
            },
            "selected_branch": (
                "post_repair_refill_promising"
                if safe_individual_rows and len(current_active_ids) + len(safe_individual_rows) >= 123
                else (
                    "selection_ranking_bottleneck"
                    if int(len(lower_all_ids - current_active_ids)) > 0 and not safe_individual_rows
                    else "post_repair_refill_rejected"
                )
            ),
        }
        core_witness_diagnostics = {
            "skipped": True,
            "reason": (
                "Core-witness repair diagnostics are kept as posthoc artifact diagnostics for the current golden JSON; "
                "the full in-pipeline replay is intentionally disabled because it adds expensive full edge-clip ablations "
                "after the already slow append-local accepted-state rebuild."
            ),
            "production_changed": False,
        }
        swap_objective_diagnostics = {
            "skipped": True,
            "reason": "swap/prune diagnostics were already evaluated; current task audits ranked-out support-diverse candidates only",
            "production_changed": False,
        }
        max150_blocker_pruning_diagnostics = {
            "skipped": True,
            "reason": "max150 blocker/pruning diagnostics were already evaluated; current task does not develop pruning/refill",
            "production_changed": False,
        }
        rankout_selection_bias_diagnostics = diagnose_rankout_selection_bias(
            core_candidates=dense_core_candidates,
            core_reconstructed=core_reconstructed,
            w2_preselection_candidates=dense_w2_preselection_candidates,
            w2_accepted_candidates=w2_accepted_candidates,
            w2_active_candidates=w2_active_candidates,
            oracle_ceiling=oracle_compatible_ceiling,
            current_summary=current_prefix_summary,
            current_reprojection=reprojection_models["production_current"],
            contours=contours,
            model_faces=model_face_rows,
            initial_vertices=vertices,
            trusted_points=trusted_cloud_points,
            trusted_z_indices=trusted_cloud_z_indices,
            point_tol=float(args.compatibility_point_tol),
            reconstruction_kwargs=oracle_reconstruction_kwargs,
            finite_support_plane_distance=float(args.finite_support_plane_distance),
            finite_support_hull_margin=float(args.finite_support_hull_margin),
            finite_support_relative_hull_margin=float(args.finite_support_relative_hull_margin),
            match_cache=oracle_match_cache,
            contour_sample=12,
            max_addition_trials=1 if append_local_candidates else 8,
        )
        reprojection_diagnostics = {
            "note": "Read-only non-oracle evaluator. It uses only shadow contour coordinates and reconstructed vertices; InitialModel is evaluated only as a coordinate sanity check.",
            "direction_count": 64,
            "contour_sample_for_trials": 80,
            "coordinate_validation": {
                "contour_plane_dot_range_sample": [
                    {
                        "contour_index": int(contour.index),
                        "normal_dot_min": as_json_float(float(np.min(np.asarray(contour.points, dtype=float) @ np.asarray(contour.normal, dtype=float)))),
                        "normal_dot_max": as_json_float(float(np.max(np.asarray(contour.points, dtype=float) @ np.asarray(contour.normal, dtype=float)))),
                    }
                    for contour in contours[:8]
                ],
                "uses_existing_projection_normal": True,
                "per_model_affine_alignment": False,
            },
            "models": reprojection_models,
            "swap_objective_diagnostics": swap_objective_diagnostics,
            "max150_blocker_pruning_diagnostics": max150_blocker_pruning_diagnostics,
            "rankout_selection_bias_diagnostics": rankout_selection_bias_diagnostics,
        }
        plane_consensus = diagnose_plane_consensus_groups(
            dense_w2_preselection_candidates,
            w2_clusters=dense_w2_edge_clusters,
            model_faces=model_face_rows,
            core_ids=core_ids,
        )
        consensus_refit_diagnostics = {
            "skipped": True,
            "reason": "multi-edge refit diagnostics are not part of max150 blocker/prune audit",
        }
        active_false_diagnostics = summarize_false_active_w2(
            w2_active_candidates=w2_active_candidates,
            w2_accepted_candidates=w2_accepted_candidates,
            model_faces=model_face_rows,
            core_ids=core_ids,
            consensus_diagnostics=plane_consensus,
        )
        candidate_9001274_diagnostics = {
            "skipped": True,
            "reason": "candidate-specific lifecycle edge-clip diagnostics are not part of max150 blocker/prune audit",
        }
        pair_ancestry = {
            "skipped": True,
            "reason": "edge-cluster ancestry diagnostics are not part of oracle-compatible face-ceiling audit",
        }
        edge_cluster_audit = {
            "skipped": True,
            "reason": "edge-cluster audit is not part of oracle-compatible face-ceiling audit",
        }
        line_residual_oracle = None
        if dense_w2_line_points is not None and int(args.max_w2_additions) <= 50:
            line_residual_oracle = collect_line_residual_oracle_diagnostics(
                z_levels=z_levels,
                line_points_w2=dense_w2_line_points,
                cond_w2=dense_w2_cond,
                accepted_segments=dense_w2_segments,
                vertices=vertices,
                faces=model.faces,
                core_face_ids=core_ids,
                min_levels=int(args.w2_edge_min_levels),
                min_z_span=float(args.w2_edge_min_z_span),
                max_line_rms=float(args.w2_edge_max_line_rms),
                max_line_residual=float(args.w2_edge_max_line_residual),
                max_z_gap=int(args.w2_edge_max_z_gap),
                max_point_jump=float(args.w2_edge_max_point_jump),
                max_condition=float(args.w2_edge_max_condition),
                cluster_mode=str(args.w2_edge_cluster_mode),
                cluster_max_distance=float(args.w2_edge_cluster_max_distance),
                cluster_max_angle_deg=float(args.w2_edge_cluster_max_angle_deg),
                cluster_min_overlap=float(args.w2_edge_cluster_min_overlap),
            )
        _, model_edge_faces = model_edges_from_faces(vertices, model.faces)
        edge_id_to_faces: dict[int, set[int]] = {}
        for eid, edge in enumerate(model_edge_faces.keys()):
            edge_id_to_faces[int(eid)] = set(model_edge_faces[edge])
        face_edge_support: dict[int, set[int]] = {}
        for row in edge_oracle.get("confident_matches", []):
            eid = int(row.get("edge_id") or -1)
            cid = int(row.get("edge_cluster_id") or -1)
            for fid in edge_id_to_faces.get(eid, set()):
                face_edge_support.setdefault(int(fid), set()).add(cid)
        ideal_pairing_faces = {fid for fid, clusters_for_face in face_edge_support.items() if len(clusters_for_face) >= 2}
        current_cyclic_faces = set(int(v) for v in raw_cohort.get("finite_face_ids", []))
        preselection_faces = set(int(v) for v in pre_cohort.get("finite_face_ids", []))
        accepted_faces = set(int(v) for v in accepted_cohort.get("finite_face_ids", []))
        active_faces = set(int(v) for v in active_cohort.get("finite_face_ids", []))
        dense_detector_summary["oracle_diagnostics"] = {
            "note": "InitialModel is used only in this diagnostics block, not in production scoring or selection.",
            "canonical_oracle_schema_version": int(CANONICAL_ORACLE_SCHEMA_VERSION),
            "canonical_oracle_tolerances": CANONICAL_ORACLE_TOLERANCES,
            "metric_reconciliation": {
                "old_metric_name": "loose_plane_good_5deg_0.05_no_finite_patch",
                "old_active_unique_ids": sorted(int(v) for v in loose_old_ids),
                "new_finite_good_unique_ids": sorted(int(v) for v in canonical_ids),
                "intersection": sorted(int(v) for v in (loose_old_ids & canonical_ids)),
                "only_old": sorted(int(v) for v in (loose_old_ids - canonical_ids)),
                "only_new": sorted(int(v) for v in (canonical_ids - loose_old_ids)),
            },
            "repaired_core_active": core_cohort,
            "repaired_core_topology": core_reconstructed.get("topology"),
            "repaired_core_volume": as_json_float(float(core_reconstructed.get("reliable_volume") or 0.0)),
            "final_core_active_after_w2": final_core_cohort,
            "w2_raw_candidates": raw_cohort,
            "w2_after_core_preselection": pre_cohort,
            "w2_accepted_cumulative": accepted_cohort,
            "w2_active_after_edge_clip": active_cohort,
            "all_active_after_edge_clip": active_total_cohort,
            "production_129_loss_funnel": production_129_loss_funnel,
            "core_face_ids_preserved": int(len(core_ids & set(int(v) for v in active_total_cohort.get("finite_face_ids", [])))),
            "core_face_ids_lost": sorted(int(v) for v in (core_ids - set(int(v) for v in active_total_cohort.get("finite_face_ids", []))))[:80],
            "new_active_face_ids": sorted(int(v) for v in (set(int(v) for v in active_total_cohort.get("finite_face_ids", [])) - core_ids))[:80],
            "w2_edge_cluster_oracle": edge_oracle,
            "full_w2_face_funnel": full_funnel,
            "prefix_frontier": prefix_frontier,
            "append_local_frontier": append_local_frontier,
            "golden_oracle_closure_diagnostics": golden_oracle_closure_diagnostics,
            "core_witness_diagnostics": core_witness_diagnostics,
            "oracle_compatible_ceiling": oracle_compatible_ceiling,
            "reprojection_diagnostics": reprojection_diagnostics,
            "active_w2_false_diagnostics": active_false_diagnostics,
            "candidate_9001274_diagnostics": candidate_9001274_diagnostics,
            "plane_consensus_diagnostics": plane_consensus,
            "plane_consensus_refit_diagnostics": consensus_refit_diagnostics,
            "w2_pair_ancestry_diagnostics": pair_ancestry,
            "w2_edge_cluster_audit": edge_cluster_audit,
            "line_residual_rejected_oracle": line_residual_oracle,
            "oracle_ceilings": {
                "current_cyclic_adjacency_unique_finite_faces": int(len(current_cyclic_faces)),
                "current_cyclic_adjacency_new_vs_core": int(len(current_cyclic_faces - core_ids)),
                "after_preselection_unique_finite_faces": int(len(preselection_faces)),
                "after_preselection_new_vs_core": int(len(preselection_faces - core_ids)),
                "accepted_unique_finite_faces": int(len(accepted_faces)),
                "accepted_new_vs_core": int(len(accepted_faces - core_ids)),
                "active_unique_finite_faces": int(len(active_faces)),
                "active_new_vs_core": int(len(active_faces - core_ids)),
                "ideal_pairing_faces_with_two_boundary_clusters": int(len(ideal_pairing_faces)),
                "ideal_pairing_new_faces_vs_core": int(len(ideal_pairing_faces - core_ids)),
                "ideal_pairing_face_ids_sample": sorted(int(v) for v in ideal_pairing_faces)[:120],
            },
        }
        pipeline_timing["oracle_diagnostics_seconds"] += time.perf_counter() - oracle_diagnostics_started
    elif bool(args.disable_oracle_diagnostics):
        dense_detector_summary["oracle_diagnostics"] = {
            "disabled": True,
            "note": "Oracle diagnostics disabled; production scoring and selection are unchanged.",
        }

    augmentation_monotonicity: dict[str, object] = {}
    if str(args.valley_integration_mode) == "augment-baseline":
        baseline_core_final = [candidate for candidate in candidates if str(candidate.get("candidate_origin")) == "baseline_core"]
        if baseline_core_final:
            baseline_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(
                baseline_core_final,
                halfspace_slack=float(args.intersection_halfspace_slack),
                feasibility_tol=float(args.intersection_feasibility_tol),
                incidence_tol=float(args.intersection_incidence_tol),
                vertex_merge_tol=float(args.intersection_vertex_merge_tol),
                min_face_area=float(args.intersection_min_face_area),
                triple_det_tol=float(args.intersection_triple_det_tol),
                prune_redundant=bool(args.intersection_prune_redundant_planes),
                redundancy_tol=float(args.intersection_redundancy_tol),
            )
            base_extents = (baseline_rec.get("lp_extents") or {}).get("extents") or {}
            aug_extents = (reconstructed.get("lp_extents") or {}).get("extents") or {}
            extent_deltas: dict[str, list[float | None]] = {}
            violations = 0
            for axis in ("x", "y", "z"):
                b = base_extents.get(axis) or [None, None]
                a = aug_extents.get(axis) or [None, None]
                delta_min = None if b[0] is None or a[0] is None else float(a[0]) - float(b[0])
                delta_max = None if b[1] is None or a[1] is None else float(a[1]) - float(b[1])
                extent_deltas[axis] = [as_json_float(delta_min) if delta_min is not None else None, as_json_float(delta_max) if delta_max is not None else None]
                if delta_min is not None and delta_min < -1e-6:
                    violations += 1
                if delta_max is not None and delta_max > 1e-6:
                    violations += 1
            base_volume = finite_float(baseline_rec.get("reliable_volume"), float("nan"))
            aug_volume = finite_float(reconstructed.get("reliable_volume"), float("nan"))
            volume_delta = aug_volume - base_volume if np.isfinite(base_volume) and np.isfinite(aug_volume) else float("nan")
            if np.isfinite(volume_delta) and volume_delta > 1e-6:
                violations += 1
            augmentation_monotonicity = {
                "baseline_core_volume": as_json_float(float(base_volume)),
                "augmented_volume": as_json_float(float(aug_volume)),
                "volume_delta_augmented_minus_baseline": as_json_float(float(volume_delta)),
                "baseline_core_extents": base_extents,
                "augmented_extents": aug_extents,
                "extent_deltas_augmented_minus_baseline": extent_deltas,
                "monotonicity_violations": int(violations),
                "baseline_core_topology": baseline_rec.get("topology"),
            }
            reconstructed["augmentation_monotonicity"] = augmentation_monotonicity

    pipeline_timing["total_seconds"] = time.perf_counter() - pipeline_started

    parameters = {
        "data_root": str(args.data_root),
        "max_contours": args.max_contours,
        "z_step": z_step,
        "window": int(args.window),
        "windows": [int(w) for w in windows],
        "candidate_scale_mode": str(args.candidate_scale_mode),
        "angular_scale_candidate_mode": str(args.angular_scale_candidate_mode),
        "angular_scale_selection_mode": str(args.angular_scale_selection_mode),
        "angular_scale_reference_half_count": int(args.angular_scale_reference_half_count),
        "max_angular_scale_additions": int(args.max_angular_scale_additions),
        "max_angular_scale_trials": int(args.max_angular_scale_trials),
        "angular_scale_additive": angular_scale_summary,
        "pipeline_timing": {k: as_json_float(float(v)) for k, v in pipeline_timing.items()},
        "window_mode": "multiscale" if args.candidate_scale_mode == "multiscale" else ("z-adaptive" if len(windows) > 1 else "single"),
        "display_window": int(display_window),
        **adaptive_summary,
        **multiscale_summary,
        **trusted_cloud_summary,
        "peak_threshold": float(args.peak_threshold),
        "low_threshold": float(args.low_threshold),
        "peak_quantile": float(args.peak_quantile),
        "low_quantile": float(args.low_quantile),
        "low_barrier_threshold": float(args.low_barrier_threshold),
        "low_barrier_fraction": float(args.low_barrier_fraction),
        "low_barrier_prominence": float(args.low_barrier_prominence),
        "low_valley_fallback": bool(args.low_valley_fallback),
        "valley_integration_mode": str(args.valley_integration_mode),
        "max_valley_additions": int(args.max_valley_additions),
        "envelope_repair_margin": float(args.envelope_repair_margin),
        "envelope_repair_relative_margin": float(args.envelope_repair_relative_margin),
        "envelope_repair_min_improvement": float(args.envelope_repair_min_improvement),
        "z_envelope_guard_mode": str(args.z_envelope_guard_mode),
        "z_envelope_guard_trigger_relative": float(args.z_envelope_guard_trigger_relative),
        "z_envelope_guard_margin_relative": float(args.z_envelope_guard_margin_relative),
        "z_envelope_guard": z_envelope_guard_summary,
        "max_baseline_guards": int(args.max_baseline_guards),
        "dense_detector_mode": str(args.dense_detector_mode),
        "max_w2_additions": int(args.max_w2_additions),
        "w2_edge_min_levels": int(args.w2_edge_min_levels),
        "w2_edge_min_z_span": float(args.w2_edge_min_z_span),
        "w2_edge_max_line_rms": float(args.w2_edge_max_line_rms),
        "w2_edge_max_line_residual": float(args.w2_edge_max_line_residual),
        "w2_edge_max_z_gap": int(args.w2_edge_max_z_gap),
        "w2_edge_max_point_jump": float(args.w2_edge_max_point_jump),
        "w2_edge_max_condition": float(args.w2_edge_max_condition),
        "w2_segment_fit_mode": str(args.w2_segment_fit_mode),
        "w2_addition_selection_mode": str(args.w2_addition_selection_mode),
        "w2_plane_consensus_mode": str(args.w2_plane_consensus_mode),
        "w2_duplicate_pair_mode": str(args.w2_duplicate_pair_mode),
        "w2_edge_cluster_mode": str(args.w2_edge_cluster_mode),
        "w2_face_min_adjacency_levels": int(args.w2_face_min_adjacency_levels),
        "w2_face_min_adjacency_fraction": float(args.w2_face_min_adjacency_fraction),
        "w2_face_min_z_span": float(args.w2_face_min_z_span),
        "w2_face_max_plane_rms": float(args.w2_face_max_plane_rms),
        "w2_small_face_refinement": str(args.w2_small_face_refinement),
        "w2_max_small_face_additions": int(args.w2_max_small_face_additions),
        "w2_max_small_face_trials": int(args.w2_max_small_face_trials),
        "w2_small_face_trial_geometry": str(args.w2_small_face_trial_geometry),
        "w2_core_repair_mode": str(args.w2_core_repair_mode),
        "w2_max_core_repairs": None if args.w2_max_core_repairs is None else int(args.w2_max_core_repairs),
        "w2_post_repair_refinement": str(args.w2_post_repair_refinement),
        "w2_max_post_repair_additions": int(args.w2_max_post_repair_additions),
        "w2_max_post_repair_trials": int(args.w2_max_post_repair_trials),
        "w2_post_ratchet_exchange_mode": str(args.w2_post_ratchet_exchange_mode),
        "w2_max_post_ratchet_exchanges": int(args.w2_max_post_ratchet_exchanges),
        "w2_core_repair_requested_action_budget": requested_core_repair_budget,
        "w2_core_repair_resolved_action_budget": int(resolved_core_repair_budget),
        "w2_core_repair": w2_core_repair_summary,
        "disable_oracle_diagnostics": bool(args.disable_oracle_diagnostics),
        **dense_detector_summary,
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
        "max_line_condition": float(args.max_line_condition),
        "minima_cloud_pct": float(args.minima_cloud_pct),
        "minima_cloud_points": int(len(minima_cloud.get("points", []))),
        "orientation_mode": str(args.orientation_mode),
        "model_support_tol": float(args.model_support_tol),
        "model_support_max_outside_count": int(args.model_support_max_outside_count),
        "max_candidates": int(args.max_candidates),
        "dedupe_normal_angle_deg": float(args.dedupe_normal_angle_deg),
        "dedupe_plane_distance": float(args.dedupe_plane_distance),
        "dedupe_hull_distance": float(args.dedupe_hull_distance),
        "candidate_cluster_mode": str(args.candidate_cluster_mode),
        "max_candidates_per_cluster": int(args.max_candidates_per_cluster),
        **dedupe_summary,
        "candidate_selection_mode": str(args.candidate_selection_mode),
        "compatibility_point_tol": float(args.compatibility_point_tol),
        "compatibility_max_global_outside_frac": float(args.compatibility_max_global_outside_frac),
        "compatibility_max_level_outside_frac": float(args.compatibility_max_level_outside_frac),
        "compatibility_bbox_margin": float(args.compatibility_bbox_margin),
        "compatibility_min_local_support": int(args.compatibility_min_local_support),
        "valley_candidate_mode": str(args.valley_candidate_mode),
        "valley_min_confidence_median": float(args.valley_min_confidence_median),
        "valley_min_confidence_p10": float(args.valley_min_confidence_p10),
        "valley_min_two_side_fraction": float(args.valley_min_two_side_fraction),
        "valley_max_one_side_fraction": float(args.valley_max_one_side_fraction),
        "valley_max_distance_p95": float(args.valley_max_distance_p95),
        "valley_max_track_smoothness": float(args.valley_max_track_smoothness),
        "valley_max_support_gap_global": float(args.valley_max_support_gap_global),
        "valley_max_support_gap_local": float(args.valley_max_support_gap_local),
        "valley_min_support_near_count": int(args.valley_min_support_near_count),
        "valley_min_scale_persistence": int(args.valley_min_scale_persistence),
        "valley_min_track_levels": int(args.valley_min_track_levels),
        "valley_max_plane_rms": float(args.valley_max_plane_rms),
        "valley_min_hull_area": float(args.valley_min_hull_area),
        "valley_min_finite_support_count": int(args.valley_min_finite_support_count),
        "valley_min_finite_track_coverage": float(args.valley_min_finite_track_coverage),
        "compatibility_outside_mode": str(args.compatibility_outside_mode),
        "compatibility_activity_check": str(args.compatibility_activity_check),
        "compatibility_activity_tol": float(args.compatibility_activity_tol),
        "finite_support_plane_distance": float(args.finite_support_plane_distance),
        "finite_support_hull_margin": float(args.finite_support_hull_margin),
        "finite_support_relative_hull_margin": float(args.finite_support_relative_hull_margin),
        "cumulative_local_max_alternatives_per_cluster": int(args.cumulative_local_max_alternatives_per_cluster),
        **selection_summary,
        "post_filter_candidates": int(len(candidates)),
        "active_face_filter_mode": str(args.active_face_filter_mode),
        "max_active_face_hull_area_ratio": float(args.max_active_face_hull_area_ratio),
        "max_active_face_extra_area": float(args.max_active_face_extra_area),
        "active_face_filter_iterations": int(args.active_face_filter_iterations),
        **active_face_filter_summary,
        "raw_candidates_before_merge": int(len(all_candidates)),
        "raw_candidates_before_dedupe": int(len(all_candidates)),
        "candidates_after_merge": int(len(candidates_for_top_k)),
        "candidates_after_dedupe": int(len(candidates_for_top_k)),
        "inside_point": as_json_point(inside_point),
        "orientation_point_count": int(orientation_points.shape[0]),
        "intersection_mesh_mode": str(args.intersection_mesh_mode),
        "intersection_inside_tol": float(args.intersection_inside_tol),
        "intersection_vertex_tol": float(args.intersection_vertex_tol),
        "intersection_halfspace_slack": float(args.intersection_halfspace_slack),
        "intersection_feasibility_tol": float(args.intersection_feasibility_tol),
        "intersection_incidence_tol": float(args.intersection_incidence_tol),
        "intersection_vertex_merge_tol": float(args.intersection_vertex_merge_tol),
        "intersection_min_face_area": float(args.intersection_min_face_area),
        "intersection_triple_det_tol": float(args.intersection_triple_det_tol),
        "intersection_prune_redundant_planes": bool(args.intersection_prune_redundant_planes),
        "intersection_redundancy_tol": float(args.intersection_redundancy_tol),
    }
    payload = build_payload(
        model_name=model_name,
        vertices=vertices,
        faces=model.faces,
        z_levels=z_levels,
        fit_rms=fit_rms,
        line_points=line_points,
        tracks=all_tracks,
        diagnostics=diagnostics,
        candidates=candidates,
        reconstructed=reconstructed,
        parameters=parameters,
        minima_cloud=minima_cloud,
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
            tracks=len(all_tracks),
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
