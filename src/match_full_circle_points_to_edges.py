from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from generate_full_circle_split_cached_viewer import (
    EPS,
    build_half_contours,
    closest_point_to_lines,
    half_contour_point_at_z,
    parse_initial_model,
    parse_merged_contour,
    polyhedron_center_of_mass,
    progress_iter,
    sorted_contour_files,
    top_points_vertical_axis_point,
    triangulate_faces,
    unique_edges_from_faces,
)


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SectionPoint:
    point: np.ndarray
    edge_ids: tuple[int, ...]
    edge_vertices: tuple[tuple[int, int], ...]
    t_values: tuple[float, ...]


@dataclass(frozen=True)
class SectionMatch:
    z_index: int
    z_observed: float
    z_reference: float
    shift: int
    reversed_order: bool
    rmse: float
    transform: np.ndarray
    ref_points: list[SectionPoint]


@dataclass(frozen=True)
class LevelMatchRecord:
    z_index: int
    z_observed: float
    z_reference: float
    point_indices: list[int]
    points: np.ndarray
    aligned_points: np.ndarray
    transform: np.ndarray
    section_rmse: float
    cyclic_shift: int
    reversed_order: bool
    ref_points: list[SectionPoint]
    section_scale: float


@dataclass(frozen=True)
class CandidateMatch:
    edge_id: int
    edge_vertices: tuple[int, int]
    t_on_edge: float
    reference_point: np.ndarray
    section_point_index: int
    distance_xy: float
    second_distance_xy: float | None
    distance_cost: float
    direction_cost: float
    unary_cost: float


@dataclass
class PointObservation:
    z_index: int
    z_observed: float
    z_reference: float
    point_index: int
    point: np.ndarray
    aligned_point: np.ndarray
    fit_rms: float
    section_rmse: float
    cyclic_shift: int
    reversed_order: bool
    candidates: list[CandidateMatch]
    chosen: CandidateMatch | None = None


def finite3(p: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(p)))


def as_json_float(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def as_json_point(p: np.ndarray) -> list[float | None]:
    return [as_json_float(float(v)) for v in p]


def section_points_with_edges(
    vertices: np.ndarray,
    edge_indices: np.ndarray,
    z_level: float,
    *,
    tol: float = 1e-8,
) -> list[SectionPoint]:
    raw: list[tuple[np.ndarray, int, tuple[int, int], float]] = []

    for edge_id, (a_raw, b_raw) in enumerate(edge_indices):
        a = int(a_raw)
        b = int(b_raw)
        p0 = vertices[a]
        p1 = vertices[b]
        z0 = float(p0[2])
        z1 = float(p1[2])
        dz0 = z0 - z_level
        dz1 = z1 - z_level

        if abs(dz0) <= tol and abs(dz1) <= tol:
            raw.append((p0.copy(), edge_id, (a, b), 0.0))
            raw.append((p1.copy(), edge_id, (a, b), 1.0))
            continue
        if abs(dz0) <= tol:
            raw.append((p0.copy(), edge_id, (a, b), 0.0))
            continue
        if abs(dz1) <= tol:
            raw.append((p1.copy(), edge_id, (a, b), 1.0))
            continue
        if dz0 * dz1 < 0.0:
            t = (z_level - z0) / (z1 - z0)
            raw.append((p0 + t * (p1 - p0), edge_id, (a, b), float(t)))

    grouped: list[dict[str, object]] = []
    for point, edge_id, edge_vertices, t_value in raw:
        found: dict[str, object] | None = None
        for item in grouped:
            if np.linalg.norm(point - item["point"]) <= tol:
                found = item
                break
        if found is None:
            grouped.append(
                {
                    "point": point,
                    "edge_ids": [edge_id],
                    "edge_vertices": [edge_vertices],
                    "t_values": [t_value],
                }
            )
        else:
            found["edge_ids"].append(edge_id)
            found["edge_vertices"].append(edge_vertices)
            found["t_values"].append(t_value)

    if len(grouped) < 2:
        return []

    center_xy = np.mean(np.array([item["point"][:2] for item in grouped], dtype=float), axis=0)
    grouped.sort(
        key=lambda item: float(
            np.arctan2(
                float(item["point"][1]) - float(center_xy[1]),
                float(item["point"][0]) - float(center_xy[0]),
            )
        )
    )

    return [
        SectionPoint(
            point=np.array(item["point"], dtype=float),
            edge_ids=tuple(int(x) for x in item["edge_ids"]),
            edge_vertices=tuple((int(a), int(b)) for a, b in item["edge_vertices"]),
            t_values=tuple(float(x) for x in item["t_values"]),
        )
        for item in grouped
    ]


def compute_line_points(
    half_contours: list[object],
    z_levels: np.ndarray,
    window: int,
    *,
    use_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_half = len(half_contours)
    n_levels = int(z_levels.size)
    window = max(2, min(n_half, int(window)))

    points_calc = [np.array(hc.points, dtype=float) for hc in half_contours]
    normals = np.array([hc.normal for hc in half_contours], dtype=float)

    dn = np.linalg.norm(normals, axis=1)
    d_unit = normals.copy()
    valid_dn = dn > EPS
    d_unit[valid_dn] = d_unit[valid_dn] / dn[valid_dn, None]
    d_unit[~valid_dn] = 0.0
    eye = np.eye(3, dtype=float)
    a_base = eye[None, :, :] - d_unit[:, :, None] * d_unit[:, None, :]
    a_base[~valid_dn, :, :] = 0.0

    half_points = np.full((n_half, n_levels, 3), np.nan, dtype=float)
    half_iter = progress_iter(
        range(n_half),
        total=n_half,
        desc="Interpolate half-contours on Z",
        use_tqdm=use_progress,
    )
    for hi in half_iter:
        poly = points_calc[hi]
        for zi, z in enumerate(z_levels):
            half_points[hi, zi, :] = half_contour_point_at_z(poly, float(z))

    line_points = np.full((n_half, n_levels, 3), np.nan, dtype=float)
    fit_rms = np.full((n_half, n_levels), np.nan, dtype=float)

    z_iter = progress_iter(
        range(n_levels),
        total=n_levels,
        desc="Compute line points",
        use_tqdm=use_progress,
    )
    for zi in z_iter:
        p_zi = half_points[:, zi, :]
        valid = np.all(np.isfinite(p_zi), axis=1) & valid_dn

        a_i = a_base.copy()
        a_i[~valid, :, :] = 0.0
        b_i = np.einsum("nij,nj->ni", a_base, p_zi, optimize=True)
        b_i[~valid, :] = 0.0
        c_i = valid.astype(np.int32)
        q_i = np.einsum("ni,ni->n", b_i, b_i, optimize=True)
        q_i[~valid] = 0.0

        a_ext = np.concatenate([a_i, a_i], axis=0)
        b_ext = np.concatenate([b_i, b_i], axis=0)
        c_ext = np.concatenate([c_i, c_i], axis=0)
        q_ext = np.concatenate([q_i, q_i], axis=0)

        a_pref = np.concatenate(
            [np.zeros((1, 3, 3), dtype=float), np.cumsum(a_ext, axis=0)],
            axis=0,
        )
        b_pref = np.concatenate(
            [np.zeros((1, 3), dtype=float), np.cumsum(b_ext, axis=0)],
            axis=0,
        )
        c_pref = np.concatenate(
            [np.zeros((1,), dtype=np.int32), np.cumsum(c_ext, axis=0)],
            axis=0,
        )
        q_pref = np.concatenate(
            [np.zeros((1,), dtype=float), np.cumsum(q_ext, axis=0)],
            axis=0,
        )

        starts = np.arange(n_half, dtype=int)
        ends = starts + window
        a_sum = a_pref[ends] - a_pref[starts]
        b_sum = b_pref[ends] - b_pref[starts]
        c_sum = c_pref[ends] - c_pref[starts]
        q_sum = q_pref[ends] - q_pref[starts]

        valid_ws = c_sum >= 2
        if not np.any(valid_ws):
            continue

        mats = a_sum[valid_ws]
        rhs = b_sum[valid_ws]
        dets = np.linalg.det(mats)
        solvable = np.abs(dets) > EPS
        idx_valid = np.flatnonzero(valid_ws)
        if np.any(solvable):
            mats_ok = mats[solvable]
            rhs_ok = rhs[solvable]
            inv_ok = np.linalg.inv(mats_ok)
            sol = np.einsum("nij,nj->ni", inv_ok, rhs_ok, optimize=True)
            line_points[idx_valid[solvable], zi, :] = sol

        if np.any(~solvable):
            idx_bad = idx_valid[~solvable]
            for ws_bad in idx_bad:
                idx_arr = np.array([(ws_bad + j) % n_half for j in range(window)], dtype=int)
                local_valid = valid[idx_arr]
                if int(np.sum(local_valid)) < 2:
                    continue
                line_points[ws_bad, zi, :] = closest_point_to_lines(
                    points=p_zi[idx_arr][local_valid],
                    directions=normals[idx_arr][local_valid],
                )

        valid_lp = np.all(np.isfinite(line_points[:, zi, :]), axis=1) & valid_ws
        if np.any(valid_lp):
            x = line_points[valid_lp, zi, :]
            a_loc = a_sum[valid_lp]
            b_loc = b_sum[valid_lp]
            q_loc = q_sum[valid_lp]
            c_loc = c_sum[valid_lp].astype(float)
            ax = np.einsum("nij,nj->ni", a_loc, x, optimize=True)
            x_ax = np.einsum("ni,ni->n", x, ax, optimize=True)
            b_x = np.einsum("ni,ni->n", b_loc, x, optimize=True)
            sse = np.maximum(x_ax - 2.0 * b_x + q_loc, 0.0)
            fit_rms[valid_lp, zi] = np.sqrt(sse / np.maximum(c_loc, 1.0))

    return line_points, fit_rms


def compute_line_points_multi_window(
    half_contours: list[object],
    z_levels: np.ndarray,
    windows: list[int],
    *,
    use_progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_half = len(half_contours)
    n_levels = int(z_levels.size)
    windows = [max(2, min(n_half, int(w))) for w in windows]

    points_calc = [np.array(hc.points, dtype=float) for hc in half_contours]
    normals = np.array([hc.normal for hc in half_contours], dtype=float)

    dn = np.linalg.norm(normals, axis=1)
    d_unit = normals.copy()
    valid_dn = dn > EPS
    d_unit[valid_dn] = d_unit[valid_dn] / dn[valid_dn, None]
    d_unit[~valid_dn] = 0.0
    eye = np.eye(3, dtype=float)
    a_base = eye[None, :, :] - d_unit[:, :, None] * d_unit[:, None, :]
    a_base[~valid_dn, :, :] = 0.0

    half_points = np.full((n_half, n_levels, 3), np.nan, dtype=float)
    half_iter = progress_iter(
        range(n_half),
        total=n_half,
        desc="Interpolate half-contours on Z",
        use_tqdm=use_progress,
    )
    for hi in half_iter:
        poly = points_calc[hi]
        for zi, z in enumerate(z_levels):
            half_points[hi, zi, :] = half_contour_point_at_z(poly, float(z))

    n_windows = len(windows)
    line_points = np.full((n_windows, n_half, n_levels, 3), np.nan, dtype=np.float32)
    fit_rms = np.full((n_windows, n_half, n_levels), np.nan, dtype=np.float32)
    cond_values = np.full((n_windows, n_half, n_levels), np.nan, dtype=np.float32)

    z_iter = progress_iter(
        range(n_levels),
        total=n_levels,
        desc="Compute line points for adaptive windows",
        use_tqdm=use_progress,
    )
    for zi in z_iter:
        p_zi = half_points[:, zi, :]
        valid = np.all(np.isfinite(p_zi), axis=1) & valid_dn

        a_i = a_base.copy()
        a_i[~valid, :, :] = 0.0
        b_i = np.einsum("nij,nj->ni", a_base, p_zi, optimize=True)
        b_i[~valid, :] = 0.0
        c_i = valid.astype(np.int32)
        q_i = np.einsum("ni,ni->n", b_i, b_i, optimize=True)
        q_i[~valid] = 0.0

        a_ext = np.concatenate([a_i, a_i], axis=0)
        b_ext = np.concatenate([b_i, b_i], axis=0)
        c_ext = np.concatenate([c_i, c_i], axis=0)
        q_ext = np.concatenate([q_i, q_i], axis=0)

        a_pref = np.concatenate(
            [np.zeros((1, 3, 3), dtype=float), np.cumsum(a_ext, axis=0)],
            axis=0,
        )
        b_pref = np.concatenate(
            [np.zeros((1, 3), dtype=float), np.cumsum(b_ext, axis=0)],
            axis=0,
        )
        c_pref = np.concatenate(
            [np.zeros((1,), dtype=np.int32), np.cumsum(c_ext, axis=0)],
            axis=0,
        )
        q_pref = np.concatenate(
            [np.zeros((1,), dtype=float), np.cumsum(q_ext, axis=0)],
            axis=0,
        )

        starts = np.arange(n_half, dtype=int)
        for wi, window in enumerate(windows):
            ends = starts + window
            a_sum = a_pref[ends] - a_pref[starts]
            b_sum = b_pref[ends] - b_pref[starts]
            c_sum = c_pref[ends] - c_pref[starts]
            q_sum = q_pref[ends] - q_pref[starts]

            valid_ws = c_sum >= 2
            if not np.any(valid_ws):
                continue

            mats = a_sum[valid_ws]
            rhs = b_sum[valid_ws]
            idx_valid = np.flatnonzero(valid_ws)
            dets = np.linalg.det(mats)
            solvable = np.abs(dets) > EPS

            if np.any(solvable):
                mats_ok = mats[solvable]
                rhs_ok = rhs[solvable]
                sol = np.linalg.solve(mats_ok, rhs_ok[..., None])[..., 0]
                idx_solvable = idx_valid[solvable]
                line_points[wi, idx_solvable, zi, :] = sol.astype(np.float32)

                try:
                    sv = np.linalg.svd(mats_ok, compute_uv=False)
                    cond = sv[:, 0] / np.maximum(sv[:, -1], EPS)
                    cond_values[wi, idx_solvable, zi] = cond.astype(np.float32)
                except np.linalg.LinAlgError:
                    cond_values[wi, idx_solvable, zi] = np.inf

            if np.any(~solvable):
                idx_bad = idx_valid[~solvable]
                for ws_bad in idx_bad:
                    idx_arr = np.array([(ws_bad + j) % n_half for j in range(window)], dtype=int)
                    local_valid = valid[idx_arr]
                    if int(np.sum(local_valid)) < 2:
                        continue
                    p = closest_point_to_lines(
                        points=p_zi[idx_arr][local_valid],
                        directions=normals[idx_arr][local_valid],
                    )
                    line_points[wi, ws_bad, zi, :] = p.astype(np.float32)
                    cond_values[wi, ws_bad, zi] = np.inf

            valid_lp = np.all(np.isfinite(line_points[wi, :, zi, :]), axis=1) & valid_ws
            if np.any(valid_lp):
                x = line_points[wi, valid_lp, zi, :].astype(float)
                a_loc = a_sum[valid_lp]
                b_loc = b_sum[valid_lp]
                q_loc = q_sum[valid_lp]
                c_loc = c_sum[valid_lp].astype(float)
                ax = np.einsum("nij,nj->ni", a_loc, x, optimize=True)
                x_ax = np.einsum("ni,ni->n", x, ax, optimize=True)
                b_x = np.einsum("ni,ni->n", b_loc, x, optimize=True)
                sse = np.maximum(x_ax - 2.0 * b_x + q_loc, 0.0)
                fit_rms[wi, valid_lp, zi] = np.sqrt(sse / np.maximum(c_loc, 1.0)).astype(np.float32)

    return line_points, fit_rms, cond_values


def local_minima_indices(values: np.ndarray, min_pct: float) -> list[int]:
    finite = np.isfinite(values)
    if int(np.sum(finite)) < 3:
        return []

    valid_values = values[finite]
    global_min = float(np.min(valid_values))
    global_max = float(np.max(valid_values))
    threshold = global_min + max(0.0, global_max - global_min) * (max(0.0, min_pct) / 100.0)

    out: list[int] = []
    n = int(values.size)
    for i in range(n):
        cur = float(values[i])
        if not np.isfinite(cur):
            continue
        prev = float(values[(i - 1 + n) % n])
        nxt = float(values[(i + 1) % n])
        if not np.isfinite(prev) or not np.isfinite(nxt):
            continue
        is_local = cur <= prev and cur <= nxt and (cur < prev or cur < nxt)
        if is_local and cur <= threshold + 1e-12:
            out.append(i)
    return out


def resample_cyclic(points: np.ndarray, n_out: int, shift: int, reversed_order: bool) -> np.ndarray:
    seq = points[::-1] if reversed_order else points
    n = int(seq.shape[0])
    if n == n_out:
        idx = (np.arange(n_out, dtype=int) + shift) % n
        return seq[idx]

    pos = (np.arange(n_out, dtype=float) * (n / n_out) + shift) % n
    i0 = np.floor(pos).astype(int)
    i1 = (i0 + 1) % n
    alpha = (pos - i0)[:, None]
    return (1.0 - alpha) * seq[i0] + alpha * seq[i1]


def fit_affine_transform(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    design = np.column_stack([src_xy, np.ones(src_xy.shape[0], dtype=float)])
    transform, *_ = np.linalg.lstsq(design, dst_xy, rcond=None)
    return transform


def fit_translation_transform(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    transform = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)
    transform[2, :] = np.mean(dst_xy - src_xy, axis=0)
    return transform


def apply_transform(src_xy: np.ndarray, transform: np.ndarray) -> np.ndarray:
    design = np.column_stack([src_xy, np.ones(src_xy.shape[0], dtype=float)])
    return design @ transform


def fit_transform(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    mode: Literal["affine", "translation"],
) -> np.ndarray:
    if mode == "translation" or src_xy.shape[0] < 3:
        return fit_translation_transform(src_xy, dst_xy)
    return fit_affine_transform(src_xy, dst_xy)


def best_section_match(
    obs_points: np.ndarray,
    ref_points: list[SectionPoint],
    *,
    transform_mode: Literal["affine", "translation"],
    z_index: int,
    z_observed: float,
    z_reference: float,
    allow_reverse: bool,
) -> SectionMatch | None:
    if obs_points.shape[0] < 2 or len(ref_points) < 2:
        return None

    obs_xy = obs_points[:, :2]
    ref_xy = np.array([sp.point[:2] for sp in ref_points], dtype=float)
    best: tuple[float, int, bool, np.ndarray] | None = None

    for reversed_order in ([False, True] if allow_reverse else [False]):
        for shift in range(ref_xy.shape[0]):
            dst = resample_cyclic(ref_xy, obs_xy.shape[0], shift, reversed_order)
            transform = fit_transform(obs_xy, dst, transform_mode)
            aligned = apply_transform(obs_xy, transform)
            residual = aligned - dst
            rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            if best is None or rmse < best[0]:
                best = (rmse, shift, reversed_order, transform)

    if best is None:
        return None

    rmse, shift, reversed_order, transform = best
    return SectionMatch(
        z_index=z_index,
        z_observed=z_observed,
        z_reference=z_reference,
        shift=shift,
        reversed_order=reversed_order,
        rmse=rmse,
        transform=transform,
        ref_points=ref_points,
    )


def z_reference_for_observed(
    z_observed: float,
    *,
    z_observed_min: float,
    z_observed_max: float,
    z_reference_min: float,
    z_reference_max: float,
    mode: Literal["identity", "normalized"],
) -> float:
    if mode == "identity":
        return float(z_observed)
    denom = z_observed_max - z_observed_min
    if abs(denom) <= EPS:
        return float(z_reference_min)
    u = (z_observed - z_observed_min) / denom
    return float(z_reference_min + u * (z_reference_max - z_reference_min))


def normalized_reference_index(obs_index: int, n_obs: int, n_ref: int) -> float:
    if n_obs <= 1 or n_ref <= 1:
        return 0.0
    return float(obs_index) * float(n_ref - 1) / float(n_obs - 1)


def finite_median(values: np.ndarray, default: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.median(finite))


def adaptive_window_quality(
    *,
    vertices: np.ndarray,
    edges: np.ndarray,
    z_levels: np.ndarray,
    windows: list[int],
    line_points_by_window: np.ndarray,
    fit_rms_by_window: np.ndarray,
    cond_by_window: np.ndarray,
    point_mode: Literal["minima", "all"],
    rms_min_pct: float,
    transform_mode: Literal["affine", "translation"],
    z_map: Literal["identity", "normalized"],
    allow_reverse: bool,
    fit_weight: float,
    condition_weight: float,
    match_weight: float,
    valid_weight: float,
) -> tuple[np.ndarray, dict[str, object]]:
    n_windows = len(windows)
    n_half = int(line_points_by_window.shape[1])
    n_levels = int(z_levels.size)
    quality = np.full((n_levels, n_windows), np.inf, dtype=float)
    match_rmse = np.full((n_levels, n_windows), np.nan, dtype=float)
    fit_medians = np.full((n_levels, n_windows), np.nan, dtype=float)
    cond_medians = np.full((n_levels, n_windows), np.nan, dtype=float)

    model_z_min = float(np.min(vertices[:, 2]))
    model_z_max = float(np.max(vertices[:, 2]))
    obs_z_min = float(np.min(z_levels))
    obs_z_max = float(np.max(z_levels))
    model_radius = max(float(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1).max()), 1.0)

    for zi, z_observed_raw in enumerate(z_levels):
        z_observed = float(z_observed_raw)
        z_reference = z_reference_for_observed(
            z_observed,
            z_observed_min=obs_z_min,
            z_observed_max=obs_z_max,
            z_reference_min=model_z_min,
            z_reference_max=model_z_max,
            mode=z_map,
        )
        ref = section_points_with_edges(vertices, edges, z_reference)
        scale = section_scale(ref) if len(ref) >= 2 else max(model_radius / 100.0, 1.0)

        for wi in range(n_windows):
            fit_values = fit_rms_by_window[wi, :, zi].astype(float)
            cond_values = cond_by_window[wi, :, zi].astype(float)
            finite_fit = np.isfinite(fit_values)
            finite_points = np.all(np.isfinite(line_points_by_window[wi, :, zi, :]), axis=1)
            valid_count = int(np.sum(finite_points))
            if valid_count < 2:
                continue

            fit_med = finite_median(fit_values, default=scale * 10.0)
            cond_med = finite_median(np.log10(np.clip(cond_values, 1.0, 1e8)), default=8.0)
            fit_medians[zi, wi] = fit_med
            cond_medians[zi, wi] = cond_med

            valid_penalty = 1.0 - (valid_count / max(1, n_half))
            fit_score = fit_med / max(scale, EPS)
            condition_score = cond_med / 8.0
            score = fit_weight * fit_score + condition_weight * condition_score + valid_weight * valid_penalty

            local_match_score = 2.5
            if len(ref) >= 2:
                if point_mode == "minima":
                    point_indices = local_minima_indices(fit_values, rms_min_pct)
                else:
                    point_indices = [int(i) for i in np.flatnonzero(finite_points)]
                point_indices = [int(i) for i in point_indices if finite_points[i]]
                if len(point_indices) >= 2:
                    obs = line_points_by_window[wi, point_indices, zi, :].astype(float)
                    match = best_section_match(
                        obs,
                        ref,
                        transform_mode=transform_mode,
                        z_index=zi,
                        z_observed=z_observed,
                        z_reference=z_reference,
                        allow_reverse=allow_reverse,
                    )
                    if match is not None:
                        local_match_score = match.rmse / max(scale, EPS)
                        match_rmse[zi, wi] = match.rmse
            quality[zi, wi] = score + match_weight * local_match_score

    summary = {
        "adaptive_quality_finite": int(np.sum(np.isfinite(quality))),
        "adaptive_quality_total": int(quality.size),
        "adaptive_match_rmse_median": as_json_float(finite_median(match_rmse, default=float("nan"))),
        "adaptive_fit_median": as_json_float(finite_median(fit_medians, default=float("nan"))),
        "adaptive_log10_cond_median": as_json_float(finite_median(cond_medians, default=float("nan"))),
    }
    return quality, summary


def select_z_adaptive_windows(
    quality: np.ndarray,
    windows: list[int],
    *,
    smooth_weight: float,
) -> tuple[np.ndarray, dict[str, object]]:
    n_levels, n_windows = quality.shape
    safe_quality = quality.copy()
    finite = np.isfinite(safe_quality)
    fallback = float(np.nanmedian(safe_quality[finite])) if np.any(finite) else 100.0
    safe_quality[~finite] = fallback + 25.0

    dp = np.full((n_levels, n_windows), np.inf, dtype=float)
    back = np.zeros((n_levels, n_windows), dtype=int)
    dp[0, :] = safe_quality[0, :]
    window_arr = np.array(windows, dtype=float)
    span = max(float(np.max(window_arr) - np.min(window_arr)), 1.0)

    for zi in range(1, n_levels):
        for wi in range(n_windows):
            transition = smooth_weight * np.abs(window_arr[wi] - window_arr) / span
            prev_costs = dp[zi - 1, :] + transition
            best_prev = int(np.argmin(prev_costs))
            dp[zi, wi] = float(prev_costs[best_prev] + safe_quality[zi, wi])
            back[zi, wi] = best_prev

    selected = np.zeros(n_levels, dtype=int)
    selected[-1] = int(np.argmin(dp[-1, :]))
    for zi in range(n_levels - 1, 0, -1):
        selected[zi - 1] = int(back[zi, selected[zi]])

    selected_windows = [int(windows[int(i)]) for i in selected]
    changes = int(np.sum(np.diff(selected) != 0)) if n_levels > 1 else 0
    summary = {
        "adaptive_window_min_selected": int(min(selected_windows)) if selected_windows else None,
        "adaptive_window_max_selected": int(max(selected_windows)) if selected_windows else None,
        "adaptive_window_median_selected": as_json_float(float(np.median(selected_windows))) if selected_windows else None,
        "adaptive_window_changes": changes,
        "adaptive_window_sequence": selected_windows,
    }
    return selected, summary


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


def adaptive_z_map_quality(
    *,
    vertices: np.ndarray,
    edges: np.ndarray,
    z_levels: np.ndarray,
    reference_z_levels: np.ndarray,
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    point_mode: Literal["minima", "all"],
    rms_min_pct: float,
    transform_mode: Literal["affine", "translation"],
    allow_reverse: bool,
    band: int,
    prior_weight: float,
    count_weight: float,
) -> tuple[np.ndarray, dict[str, object]]:
    n_obs = int(z_levels.size)
    n_ref = int(reference_z_levels.size)
    quality = np.full((n_obs, n_ref), np.inf, dtype=float)
    match_rmse = np.full((n_obs, n_ref), np.nan, dtype=float)
    section_counts = np.full(n_ref, 0, dtype=np.int32)
    model_radius = max(float(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1).max()), 1.0)

    ref_cache: list[list[SectionPoint]] = []
    scale_cache = np.full(n_ref, 1.0, dtype=float)
    for ri, z_ref in enumerate(reference_z_levels):
        ref = section_points_with_edges(vertices, edges, float(z_ref))
        ref_cache.append(ref)
        section_counts[ri] = len(ref)
        scale_cache[ri] = section_scale(ref) if len(ref) >= 2 else max(model_radius / 100.0, 1.0)

    for zi in range(n_obs):
        finite_points = np.all(np.isfinite(line_points[:, zi, :]), axis=1)
        if point_mode == "minima":
            point_indices = local_minima_indices(fit_rms[:, zi].astype(float), rms_min_pct)
        else:
            point_indices = [int(i) for i in np.flatnonzero(finite_points)]
        point_indices = [int(i) for i in point_indices if finite_points[i]]
        if len(point_indices) < 2:
            continue

        obs = line_points[point_indices, zi, :].astype(float)
        expected = normalized_reference_index(zi, n_obs, n_ref)
        if band > 0:
            lo = max(0, int(np.floor(expected - band)))
            hi = min(n_ref, int(np.ceil(expected + band)) + 1)
        else:
            lo = 0
            hi = n_ref

        for ri in range(lo, hi):
            ref = ref_cache[ri]
            if len(ref) < 2:
                continue
            match = best_section_match(
                obs,
                ref,
                transform_mode=transform_mode,
                z_index=zi,
                z_observed=float(z_levels[zi]),
                z_reference=float(reference_z_levels[ri]),
                allow_reverse=allow_reverse,
            )
            if match is None:
                continue

            scale = max(scale_cache[ri], EPS)
            match_score = match.rmse / scale
            prior_score = abs(float(ri) - expected) / max(float(band), 1.0) if band > 0 else 0.0
            count_score = abs(len(point_indices) - len(ref)) / max(len(point_indices), len(ref), 1)
            quality[zi, ri] = match_score + prior_weight * prior_score + count_weight * count_score
            match_rmse[zi, ri] = match.rmse

    summary = {
        "adaptive_z_quality_finite": int(np.sum(np.isfinite(quality))),
        "adaptive_z_quality_total": int(quality.size),
        "adaptive_z_match_rmse_median": as_json_float(finite_median(match_rmse, default=float("nan"))),
        "adaptive_z_section_count_min": int(np.min(section_counts)) if section_counts.size else None,
        "adaptive_z_section_count_max": int(np.max(section_counts)) if section_counts.size else None,
    }
    return quality, summary


def select_adaptive_z_map(
    quality: np.ndarray,
    reference_z_levels: np.ndarray,
    *,
    smooth_weight: float,
    max_jump: int,
    flat_penalty: float,
) -> tuple[np.ndarray, dict[str, object]]:
    n_obs, n_ref = quality.shape
    safe_quality = quality.copy()
    finite = np.isfinite(safe_quality)
    fallback = float(np.nanmedian(safe_quality[finite])) if np.any(finite) else 100.0
    safe_quality[~finite] = fallback + 25.0

    max_jump = max(1, int(max_jump))
    expected_step = float(n_ref - 1) / float(max(n_obs - 1, 1))
    step_norm = max(expected_step, 1.0)
    dp = np.full((n_obs, n_ref), np.inf, dtype=float)
    back = np.zeros((n_obs, n_ref), dtype=np.int32)
    dp[0, :] = safe_quality[0, :]

    for zi in range(1, n_obs):
        for ri in range(n_ref):
            lo = max(0, ri - max_jump)
            hi = ri + 1  # monotonic non-decreasing reference Z.
            prev_idx = np.arange(lo, hi, dtype=int)
            jump = ri - prev_idx
            transition = smooth_weight * np.abs(jump - expected_step) / step_norm
            transition = transition + flat_penalty * (jump == 0)
            prev_costs = dp[zi - 1, lo:hi] + transition
            best_local = int(np.argmin(prev_costs))
            dp[zi, ri] = float(prev_costs[best_local] + safe_quality[zi, ri])
            back[zi, ri] = int(prev_idx[best_local])

    selected = np.zeros(n_obs, dtype=np.int32)
    selected[-1] = int(np.argmin(dp[-1, :]))
    for zi in range(n_obs - 1, 0, -1):
        selected[zi - 1] = int(back[zi, selected[zi]])

    expected = np.array([normalized_reference_index(i, n_obs, n_ref) for i in range(n_obs)], dtype=float)
    deltas = selected.astype(float) - expected
    selected_z = reference_z_levels[selected]
    dz = np.diff(selected_z) if selected_z.size > 1 else np.array([], dtype=float)
    summary = {
        "adaptive_z_ref_levels": int(n_ref),
        "adaptive_z_ref_min": float(reference_z_levels[0]) if n_ref else None,
        "adaptive_z_ref_max": float(reference_z_levels[-1]) if n_ref else None,
        "adaptive_z_selected_min": as_json_float(float(np.min(selected_z))) if selected_z.size else None,
        "adaptive_z_selected_max": as_json_float(float(np.max(selected_z))) if selected_z.size else None,
        "adaptive_z_index_delta_median": as_json_float(float(np.median(deltas))) if deltas.size else None,
        "adaptive_z_index_delta_abs_median": as_json_float(float(np.median(np.abs(deltas)))) if deltas.size else None,
        "adaptive_z_index_delta_abs_max": as_json_float(float(np.max(np.abs(deltas)))) if deltas.size else None,
        "adaptive_z_changes": int(np.sum(dz > EPS)) if dz.size else 0,
        "adaptive_z_flat_steps": int(np.sum(np.abs(dz) <= EPS)) if dz.size else 0,
        "adaptive_z_selected_indices": selected.astype(int).tolist(),
    }
    return selected_z.astype(float), summary


def nearest_section_point(
    aligned_xy: np.ndarray,
    ref_points: list[SectionPoint],
) -> tuple[int, float, float, float]:
    ref_xy = np.array([sp.point[:2] for sp in ref_points], dtype=float)
    d = np.linalg.norm(ref_xy - aligned_xy[None, :], axis=1)
    order = np.argsort(d)
    best_idx = int(order[0])
    best_dist = float(d[best_idx])
    second_dist = float(d[int(order[1])]) if d.size > 1 else float("inf")
    confidence = 1.0
    if np.isfinite(second_dist) and second_dist > EPS:
        confidence = float(np.clip(1.0 - best_dist / second_dist, 0.0, 1.0))
    return best_idx, best_dist, second_dist, confidence


def section_scale(ref_points: list[SectionPoint]) -> float:
    if len(ref_points) < 2:
        return 1.0
    xy = np.array([sp.point[:2] for sp in ref_points], dtype=float)
    d = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    valid = d[np.isfinite(d) & (d > EPS)]
    if valid.size:
        return max(float(np.median(valid)), EPS)
    span = np.linalg.norm(np.max(xy, axis=0) - np.min(xy, axis=0))
    return max(float(span) / max(1, len(ref_points)), 1.0)


def build_edge_adjacency(edge_indices: np.ndarray) -> list[set[int]]:
    vertex_to_edges: dict[int, list[int]] = {}
    for edge_id, (a_raw, b_raw) in enumerate(edge_indices):
        vertex_to_edges.setdefault(int(a_raw), []).append(int(edge_id))
        vertex_to_edges.setdefault(int(b_raw), []).append(int(edge_id))

    adjacency = [set() for _ in range(edge_indices.shape[0])]
    for incident_edges in vertex_to_edges.values():
        for i in incident_edges:
            for j in incident_edges:
                if i != j:
                    adjacency[i].add(j)
    return adjacency


def edge_graph_distance(
    adjacency: list[set[int]],
    a: int,
    b: int,
    *,
    max_depth: int = 4,
) -> int:
    if a == b:
        return 0
    frontier = {a}
    visited = {a}
    for depth in range(1, max_depth + 1):
        nxt: set[int] = set()
        for cur in frontier:
            for nb in adjacency[cur]:
                if nb == b:
                    return depth
                if nb not in visited:
                    visited.add(nb)
                    nxt.add(nb)
        frontier = nxt
        if not frontier:
            break
    return max_depth + 1


def topology_transition_cost(
    adjacency: list[set[int]],
    a: int,
    b: int,
    cache: dict[tuple[int, int], float],
) -> float:
    if a == b:
        return 0.0
    key = (a, b) if a <= b else (b, a)
    if key in cache:
        return cache[key]
    dist = edge_graph_distance(adjacency, a, b)
    if dist == 1:
        cost = 0.18
    elif dist == 2:
        cost = 0.55
    elif dist == 3:
        cost = 0.95
    elif dist == 4:
        cost = 1.35
    else:
        cost = 1.75
    cache[key] = cost
    return cost


def edge_direction_unit(vertices: np.ndarray, edge_vertices: tuple[int, int]) -> np.ndarray | None:
    a, b = edge_vertices
    d = vertices[b] - vertices[a]
    dn = float(np.linalg.norm(d))
    if dn <= EPS:
        return None
    return d / dn


def direction_mismatch(observed_dir: np.ndarray | None, edge_dir: np.ndarray | None) -> float:
    if observed_dir is None or edge_dir is None:
        return 0.0
    od = float(np.linalg.norm(observed_dir))
    ed = float(np.linalg.norm(edge_dir))
    if od <= EPS or ed <= EPS:
        return 0.0
    dot = float(np.dot(observed_dir / od, edge_dir / ed))
    return float(np.clip(1.0 - abs(dot), 0.0, 1.0))


def build_level_records(
    *,
    vertices: np.ndarray,
    edges: np.ndarray,
    z_levels: np.ndarray,
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    point_mode: Literal["minima", "all"],
    rms_min_pct: float,
    transform_mode: Literal["affine", "translation"],
    z_map: Literal["identity", "normalized"],
    trim_bottom: int,
    trim_top: int,
    allow_reverse: bool,
    z_reference_levels: np.ndarray | None = None,
) -> tuple[list[LevelMatchRecord], dict[str, object]]:
    model_z_min = float(np.min(vertices[:, 2]))
    model_z_max = float(np.max(vertices[:, 2]))
    obs_z_min = float(np.min(z_levels))
    obs_z_max = float(np.max(z_levels))

    records: list[LevelMatchRecord] = []
    level_rmses: list[float] = []
    skipped_no_points = 0
    skipped_no_section = 0
    skipped_no_match = 0
    n_levels = int(z_levels.size)
    z_start = max(0, int(trim_bottom))
    z_stop = max(z_start, n_levels - max(0, int(trim_top)))

    for zi in range(z_start, z_stop):
        z_observed = float(z_levels[zi])
        if z_reference_levels is not None:
            z_reference = float(z_reference_levels[zi])
        else:
            z_reference = z_reference_for_observed(
                z_observed,
                z_observed_min=obs_z_min,
                z_observed_max=obs_z_max,
                z_reference_min=model_z_min,
                z_reference_max=model_z_max,
                mode=z_map,
            )

        ref = section_points_with_edges(vertices, edges, z_reference)
        if len(ref) < 2:
            skipped_no_section += 1
            continue

        if point_mode == "minima":
            point_indices = local_minima_indices(fit_rms[:, zi], rms_min_pct)
        else:
            point_indices = [
                int(i)
                for i in range(line_points.shape[0])
                if finite3(line_points[i, zi, :])
            ]

        if len(point_indices) < 2:
            skipped_no_points += 1
            continue

        point_indices = [int(i) for i in point_indices if finite3(line_points[i, zi, :])]
        obs = np.array([line_points[i, zi, :] for i in point_indices], dtype=float)
        if obs.shape[0] < 2:
            skipped_no_points += 1
            continue

        match = best_section_match(
            obs,
            ref,
            transform_mode=transform_mode,
            z_index=zi,
            z_observed=z_observed,
            z_reference=z_reference,
            allow_reverse=allow_reverse,
        )
        if match is None:
            skipped_no_match += 1
            continue

        level_rmses.append(match.rmse)
        aligned = apply_transform(obs[:, :2], match.transform)
        aligned_points = np.column_stack([aligned, np.full(aligned.shape[0], z_reference, dtype=float)])
        records.append(
            LevelMatchRecord(
                z_index=int(zi),
                z_observed=z_observed,
                z_reference=z_reference,
                point_indices=point_indices,
                points=obs,
                aligned_points=aligned_points,
                transform=match.transform,
                section_rmse=match.rmse,
                cyclic_shift=int(match.shift),
                reversed_order=bool(match.reversed_order),
                ref_points=ref,
                section_scale=section_scale(ref),
            )
        )

    summary = {
        "levels_total": int(n_levels),
        "levels_used_range": [int(z_start), int(max(z_start, z_stop - 1))],
        "levels_matched": int(len(records)),
        "skipped_no_points": int(skipped_no_points),
        "skipped_no_section": int(skipped_no_section),
        "skipped_no_match": int(skipped_no_match),
        "section_rmse_median": as_json_float(float(np.median(level_rmses))) if level_rmses else None,
        "section_rmse_max": as_json_float(float(np.max(level_rmses))) if level_rmses else None,
    }
    return records, summary


def observed_direction_map(records: list[LevelMatchRecord]) -> dict[tuple[int, int], np.ndarray]:
    tracks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for record in records:
        for local_idx, point_index in enumerate(record.point_indices):
            tracks.setdefault(int(point_index), []).append(
                (int(record.z_index), record.aligned_points[local_idx].astype(float))
            )

    out: dict[tuple[int, int], np.ndarray] = {}
    for point_index, items in tracks.items():
        items.sort(key=lambda item: item[0])
        for idx, (z_index, p) in enumerate(items):
            if len(items) < 2:
                continue
            if 0 < idx < len(items) - 1:
                d = items[idx + 1][1] - items[idx - 1][1]
            elif idx == 0:
                d = items[idx + 1][1] - p
            else:
                d = p - items[idx - 1][1]
            dn = float(np.linalg.norm(d))
            if dn > EPS:
                out[(int(point_index), int(z_index))] = d / dn
    return out


def candidate_matches_for_point(
    *,
    aligned_point: np.ndarray,
    ref_points: list[SectionPoint],
    vertices: np.ndarray,
    observed_dir: np.ndarray | None,
    candidate_k: int,
    scale: float,
    distance_weight: float,
    direction_weight: float,
) -> list[CandidateMatch]:
    if not ref_points:
        return []

    ref_xy = np.array([sp.point[:2] for sp in ref_points], dtype=float)
    distances = np.linalg.norm(ref_xy - aligned_point[:2][None, :], axis=1)
    order = np.argsort(distances)
    top = order[: max(1, min(int(candidate_k), len(order)))]
    second_distance = float(distances[int(order[1])]) if len(order) > 1 else None

    candidates: list[CandidateMatch] = []
    for ref_idx_raw in top:
        ref_idx = int(ref_idx_raw)
        sp = ref_points[ref_idx]
        dist = float(distances[ref_idx])
        distance_cost = dist / max(scale, EPS)
        for edge_id, edge_vertices, t_value in zip(sp.edge_ids, sp.edge_vertices, sp.t_values):
            edge_dir = edge_direction_unit(vertices, edge_vertices)
            dir_cost = direction_mismatch(observed_dir, edge_dir)
            unary = distance_weight * distance_cost + direction_weight * dir_cost
            candidates.append(
                CandidateMatch(
                    edge_id=int(edge_id),
                    edge_vertices=(int(edge_vertices[0]), int(edge_vertices[1])),
                    t_on_edge=float(t_value),
                    reference_point=sp.point.astype(float),
                    section_point_index=ref_idx,
                    distance_xy=dist,
                    second_distance_xy=second_distance,
                    distance_cost=float(distance_cost),
                    direction_cost=float(dir_cost),
                    unary_cost=float(unary),
                )
            )

    candidates.sort(key=lambda c: c.unary_cost)
    return candidates


def build_observations_with_fit_rms(
    *,
    records: list[LevelMatchRecord],
    vertices: np.ndarray,
    fit_rms: np.ndarray,
    candidate_k: int,
    distance_weight: float,
    direction_weight: float,
) -> list[PointObservation]:
    dir_map = observed_direction_map(records)
    observations: list[PointObservation] = []

    for record in records:
        for local_idx, point_index in enumerate(record.point_indices):
            observed_dir = dir_map.get((int(point_index), int(record.z_index)))
            candidates = candidate_matches_for_point(
                aligned_point=record.aligned_points[local_idx],
                ref_points=record.ref_points,
                vertices=vertices,
                observed_dir=observed_dir,
                candidate_k=candidate_k,
                scale=record.section_scale,
                distance_weight=distance_weight,
                direction_weight=direction_weight,
            )
            observations.append(
                PointObservation(
                    z_index=record.z_index,
                    z_observed=record.z_observed,
                    z_reference=record.z_reference,
                    point_index=int(point_index),
                    point=record.points[local_idx].astype(float),
                    aligned_point=record.aligned_points[local_idx].astype(float),
                    fit_rms=float(fit_rms[int(point_index), int(record.z_index)]),
                    section_rmse=record.section_rmse,
                    cyclic_shift=record.cyclic_shift,
                    reversed_order=record.reversed_order,
                    candidates=candidates,
                )
            )

    return observations


def solve_tracks_viterbi(
    observations: list[PointObservation],
    *,
    adjacency: list[set[int]],
    smooth_weight: float,
) -> dict[str, int]:
    tracks: dict[int, list[PointObservation]] = {}
    for obs in observations:
        if obs.candidates:
            tracks.setdefault(int(obs.point_index), []).append(obs)

    transition_cache: dict[tuple[int, int], float] = {}
    transitions_same = 0
    transitions_near = 0
    transitions_far = 0

    for track in tracks.values():
        track.sort(key=lambda obs: obs.z_index)
        if not track:
            continue

        dp = np.array([c.unary_cost for c in track[0].candidates], dtype=float)
        back_ptrs: list[np.ndarray] = []

        for obs_idx in range(1, len(track)):
            prev_obs = track[obs_idx - 1]
            cur_obs = track[obs_idx]
            prev_candidates = prev_obs.candidates
            cur_candidates = cur_obs.candidates
            cur_dp = np.full(len(cur_candidates), np.inf, dtype=float)
            cur_back = np.zeros(len(cur_candidates), dtype=int)
            gap = max(1, int(cur_obs.z_index) - int(prev_obs.z_index))
            gap_relax = float(np.sqrt(gap))

            for ci, cur_candidate in enumerate(cur_candidates):
                best_cost = float("inf")
                best_prev = 0
                for pi, prev_candidate in enumerate(prev_candidates):
                    trans = topology_transition_cost(
                        adjacency,
                        int(prev_candidate.edge_id),
                        int(cur_candidate.edge_id),
                        transition_cache,
                    )
                    cost = float(dp[pi]) + (smooth_weight * trans / gap_relax)
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = pi
                cur_dp[ci] = best_cost + cur_candidate.unary_cost
                cur_back[ci] = best_prev
            dp = cur_dp
            back_ptrs.append(cur_back)

        chosen_idx = int(np.argmin(dp))
        chosen_indices = [0] * len(track)
        chosen_indices[-1] = chosen_idx
        for obs_idx in range(len(track) - 1, 0, -1):
            chosen_idx = int(back_ptrs[obs_idx - 1][chosen_idx])
            chosen_indices[obs_idx - 1] = chosen_idx

        prev_edge: int | None = None
        for obs, idx in zip(track, chosen_indices):
            obs.chosen = obs.candidates[int(idx)]
            if prev_edge is not None:
                dist = edge_graph_distance(adjacency, prev_edge, obs.chosen.edge_id, max_depth=2)
                if obs.chosen.edge_id == prev_edge:
                    transitions_same += 1
                elif dist <= 1:
                    transitions_near += 1
                else:
                    transitions_far += 1
            prev_edge = obs.chosen.edge_id

    for obs in observations:
        if obs.chosen is None and obs.candidates:
            obs.chosen = obs.candidates[0]

    return {
        "tracks": int(len(tracks)),
        "transitions_same_edge": int(transitions_same),
        "transitions_adjacent_edge": int(transitions_near),
        "transitions_far_edge": int(transitions_far),
    }


def candidate_confidence(chosen: CandidateMatch, candidates: list[CandidateMatch]) -> float:
    if not candidates:
        return 0.0
    by_cost = sorted(candidates, key=lambda c: c.unary_cost)
    if len(by_cost) < 2:
        cost_margin = 1.0
    else:
        denom = max(abs(by_cost[1].unary_cost), EPS)
        cost_margin = float(np.clip((by_cost[1].unary_cost - chosen.unary_cost) / denom, 0.0, 1.0))
    if chosen.second_distance_xy is not None and chosen.second_distance_xy > EPS:
        distance_margin = float(np.clip(1.0 - chosen.distance_xy / chosen.second_distance_xy, 0.0, 1.0))
    else:
        distance_margin = 1.0
    direction_conf = float(np.clip(1.0 - chosen.direction_cost, 0.0, 1.0))
    return float(np.clip(0.45 * distance_margin + 0.35 * cost_margin + 0.20 * direction_conf, 0.0, 1.0))


def build_matches(
    *,
    vertices: np.ndarray,
    edges: np.ndarray,
    model_center: np.ndarray,
    z_levels: np.ndarray,
    line_points: np.ndarray,
    fit_rms: np.ndarray,
    point_mode: Literal["minima", "all"],
    rms_min_pct: float,
    transform_mode: Literal["affine", "translation"],
    z_map: Literal["identity", "normalized"],
    trim_bottom: int,
    trim_top: int,
    allow_reverse: bool,
    candidate_k: int,
    distance_weight: float,
    direction_weight: float,
    smooth_weight: float,
    z_reference_levels: np.ndarray | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records, summary = build_level_records(
        vertices=vertices,
        edges=edges,
        z_levels=z_levels,
        line_points=line_points,
        fit_rms=fit_rms,
        point_mode=point_mode,
        rms_min_pct=rms_min_pct,
        transform_mode=transform_mode,
        z_map=z_map,
        trim_bottom=trim_bottom,
        trim_top=trim_top,
        allow_reverse=allow_reverse,
        z_reference_levels=z_reference_levels,
    )
    observations = build_observations_with_fit_rms(
        records=records,
        vertices=vertices,
        fit_rms=fit_rms,
        candidate_k=candidate_k,
        distance_weight=distance_weight,
        direction_weight=direction_weight,
    )

    adjacency = build_edge_adjacency(edges)
    transition_summary = solve_tracks_viterbi(
        observations,
        adjacency=adjacency,
        smooth_weight=smooth_weight,
    )

    matches: list[dict[str, object]] = []
    for obs in sorted(observations, key=lambda item: (item.z_index, item.point_index)):
        if obs.chosen is None:
            continue
        chosen = obs.chosen
        confidence = candidate_confidence(chosen, obs.candidates)
        matches.append(
            {
                "z_index": int(obs.z_index),
                "z_observed": float(obs.z_observed),
                "z_reference": float(obs.z_reference),
                "point_index": int(obs.point_index),
                "point": as_json_point(obs.point),
                "point_display": as_json_point(obs.point - model_center),
                "fit_rms": as_json_float(float(obs.fit_rms)),
                "edge_id": int(chosen.edge_id),
                "edge_vertices": [int(chosen.edge_vertices[0]), int(chosen.edge_vertices[1])],
                "candidate_edge_ids": [int(c.edge_id) for c in obs.candidates],
                "t_on_edge": float(chosen.t_on_edge),
                "reference_point": as_json_point(chosen.reference_point),
                "reference_point_display": as_json_point(chosen.reference_point - model_center),
                "aligned_point": as_json_point(obs.aligned_point),
                "aligned_point_display": as_json_point(obs.aligned_point - model_center),
                "distance_xy": float(chosen.distance_xy),
                "second_distance_xy": chosen.second_distance_xy,
                "distance_cost": float(chosen.distance_cost),
                "direction_cost": float(chosen.direction_cost),
                "unary_cost": float(chosen.unary_cost),
                "confidence": confidence,
                "section_rmse": float(obs.section_rmse),
                "cyclic_shift": int(obs.cyclic_shift),
                "orientation": "reversed" if obs.reversed_order else "forward",
                "matcher": "section-candidates+direction+viterbi",
            }
        )

    distances = np.array([float(m["distance_xy"]) for m in matches], dtype=float) if matches else np.array([])
    direction_costs = (
        np.array([float(m["direction_cost"]) for m in matches], dtype=float) if matches else np.array([])
    )
    summary.update(
        {
            "matches": int(len(matches)),
            "candidate_k": int(candidate_k),
            "distance_weight": float(distance_weight),
            "direction_weight": float(direction_weight),
            "smooth_weight": float(smooth_weight),
            "distance_xy_median": as_json_float(float(np.median(distances))) if distances.size else None,
            "distance_xy_max": as_json_float(float(np.max(distances))) if distances.size else None,
            "direction_cost_median": as_json_float(float(np.median(direction_costs))) if direction_costs.size else None,
            "direction_cost_max": as_json_float(float(np.max(direction_costs))) if direction_costs.size else None,
            **transition_summary,
        }
    )
    return matches, summary


def discover_single_model(data_root: Path, model_name: str | None) -> str:
    if model_name:
        return model_name
    models = [
        p.name
        for p in sorted(data_root.iterdir())
        if p.is_dir() and (p / "InitialModel").exists() and (p / "shadow").exists()
    ]
    if not models:
        raise ValueError(f"No models found under {data_root}")
    if len(models) > 1:
        raise ValueError(f"Multiple models found: {', '.join(models)}. Pass --model.")
    return models[0]


def relative_plotly_src(output_html: Path) -> str:
    local_plotly = output_html.parent / "plotly-2.35.2.min.js"
    if local_plotly.exists():
        return local_plotly.name
    return "https://cdn.plot.ly/plotly-2.35.2.min.js"


def build_viewer_html(payload: dict[str, object], output_html: Path) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    plotly_src = relative_plotly_src(output_html)
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Point-to-edge matches</title>
  <script src="{plotly_src}"></script>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: "Trebuchet MS", sans-serif; background: #f4f7fb; color: #152238; }}
    #bar {{ box-sizing: border-box; padding: 10px 14px; border-bottom: 1px solid #dbe4ee; background: rgba(255,255,255,0.92); }}
    #plot {{ width: 100vw; height: calc(100vh - 58px); }}
    .muted {{ color: #667085; }}
  </style>
</head>
<body>
  <div id="bar">
    <strong id="title">Point-to-edge matches</strong>
    <span id="summary" class="muted"></span>
  </div>
  <div id="plot"></div>
  <script>
    const DATA = {data_json};

    function edgeColor(edgeId) {{
      const h = (Number(edgeId) * 137.508) % 360;
      return `hsl(${{h.toFixed(1)}}, 78%, 45%)`;
    }}

    function fmt(v, digits = 5) {{
      return Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : 'n/a';
    }}

    const viewer = DATA.viewer || {{}};
    const vertices = viewer.vertices_display || [];
    const triangles = viewer.triangles || [];
    const edges = viewer.edges || [];
    const matches = DATA.matches || [];
    const matchedEdges = new Set(matches.map(m => Number(m.edge_id)));
    const traces = [];

    if (triangles.length > 0) {{
      traces.push({{
        type: 'mesh3d',
        x: vertices.map(p => p[0]),
        y: vertices.map(p => p[1]),
        z: vertices.map(p => p[2]),
        i: triangles.map(t => t[0]),
        j: triangles.map(t => t[1]),
        k: triangles.map(t => t[2]),
        color: '#aab8c8',
        opacity: 0.16,
        flatshading: true,
        hoverinfo: 'skip',
        showscale: false,
        name: 'model',
      }});
    }}

    for (let edgeId = 0; edgeId < edges.length; edgeId++) {{
      const e = edges[edgeId];
      const p0 = vertices[e[0]];
      const p1 = vertices[e[1]];
      const isMatched = matchedEdges.has(edgeId);
      traces.push({{
        type: 'scatter3d',
        mode: 'lines',
        x: [p0[0], p1[0]],
        y: [p0[1], p1[1]],
        z: [p0[2], p1[2]],
        line: {{
          color: isMatched ? edgeColor(edgeId) : 'rgba(72, 84, 101, 0.18)',
          width: isMatched ? 5 : 1.5,
        }},
        opacity: isMatched ? 0.92 : 0.35,
        hovertemplate: `edge_id=${{edgeId}}<br>vertices=${{e[0]}}-${{e[1]}}<extra></extra>`,
        showlegend: false,
        name: `edge ${{edgeId}}`,
      }});
    }}

    traces.push({{
      type: 'scatter3d',
      mode: 'markers',
      x: matches.map(m => m.aligned_point_display[0]),
      y: matches.map(m => m.aligned_point_display[1]),
      z: matches.map(m => m.aligned_point_display[2]),
      marker: {{
        size: 4.5,
        color: matches.map(m => edgeColor(m.edge_id)),
        line: {{ color: 'rgba(20, 33, 61, 0.42)', width: 0.5 }},
        opacity: 0.95,
      }},
      text: matches.map(m =>
        `edge_id=${{m.edge_id}}` +
        `<br>edge_vertices=${{m.edge_vertices[0]}}-${{m.edge_vertices[1]}}` +
        `<br>point_index=${{m.point_index}}, z_index=${{m.z_index}}` +
        `<br>distance_xy=${{fmt(m.distance_xy)}}` +
        `<br>direction_cost=${{fmt(m.direction_cost, 3)}}` +
        `<br>unary_cost=${{fmt(m.unary_cost, 3)}}` +
        `<br>confidence=${{fmt(m.confidence, 3)}}` +
        `<br>fit_rms=${{fmt(m.fit_rms)}}` +
        `<br>orientation=${{m.orientation}}`
      ),
      hovertemplate: '%{{text}}<extra>matched point</extra>',
      showlegend: false,
      name: 'matched points',
    }});

    document.getElementById('title').textContent = `${{DATA.model}}: ребра и соответствующие точки`;
    document.getElementById('summary').textContent =
      ` | matches=${{DATA.summary.matches}}, point_mode=${{DATA.parameters.point_mode}}, window_mode=${{DATA.parameters.window_mode}}, transform=${{DATA.parameters.transform}}, z_map=${{DATA.parameters.z_map}}, matcher=direction+Viterbi`;

    Plotly.newPlot('plot', traces, {{
      margin: {{ l: 0, r: 0, b: 0, t: 8 }},
      scene: {{
        xaxis: {{ title: 'X' }},
        yaxis: {{ title: 'Y' }},
        zaxis: {{ title: 'Z' }},
        aspectmode: 'data',
      }},
    }}, {{ responsive: true, displaylogo: false }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Match full-circle split line-points/RMS minima to reference model edges "
            "using per-Z cross-sections and cyclic topological alignment."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--max-contours", type=int, default=None)
    parser.add_argument("--z-step", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--window-mode",
        choices=("fixed", "z-adaptive"),
        default="fixed",
        help="fixed uses --window; z-adaptive selects one window per Z level.",
    )
    parser.add_argument("--window-min", type=int, default=4)
    parser.add_argument("--window-max", type=int, default=24)
    parser.add_argument("--window-step", type=int, default=2)
    parser.add_argument(
        "--window-fit-weight",
        type=float,
        default=0.7,
        help="Adaptive-window weight for median fit RMS.",
    )
    parser.add_argument(
        "--window-condition-weight",
        type=float,
        default=0.8,
        help="Adaptive-window weight for line-system conditioning.",
    )
    parser.add_argument(
        "--window-match-weight",
        type=float,
        default=1.0,
        help="Adaptive-window weight for section-to-model matching RMSE.",
    )
    parser.add_argument(
        "--window-valid-weight",
        type=float,
        default=0.4,
        help="Adaptive-window penalty for missing/invalid line points.",
    )
    parser.add_argument(
        "--window-smooth-weight",
        type=float,
        default=1.0,
        help="Adaptive-window DP penalty for changing window between adjacent Z levels.",
    )
    parser.add_argument(
        "--point-mode",
        choices=("minima", "all"),
        default="minima",
        help="minima matches the yellow RMS-minima cloud; all matches every computed line point.",
    )
    parser.add_argument("--rms-min-pct", type=float, default=10.0)
    parser.add_argument("--trim-bottom", type=int, default=2)
    parser.add_argument("--trim-top", type=int, default=20)
    parser.add_argument(
        "--transform",
        choices=("affine", "translation"),
        default="affine",
        help="Per-Z XY transform from observed points to reference section points.",
    )
    parser.add_argument(
        "--z-map",
        choices=("identity", "normalized", "adaptive"),
        default="identity",
        help="identity uses same Z; normalized maps Z fraction; adaptive selects a monotonic reference Z per observed Z.",
    )
    parser.add_argument(
        "--z-map-ref-levels",
        type=int,
        default=0,
        help="Reference Z levels for adaptive z-map. 0 means same count as observed levels.",
    )
    parser.add_argument(
        "--z-map-band",
        type=int,
        default=60,
        help="Adaptive z-map search band in reference-level indices around normalized mapping. 0 disables the band.",
    )
    parser.add_argument(
        "--z-map-max-jump",
        type=int,
        default=5,
        help="Maximum reference-level jump between adjacent observed Z levels in adaptive z-map.",
    )
    parser.add_argument(
        "--z-map-smooth-weight",
        type=float,
        default=0.8,
        help="Adaptive z-map DP penalty for slope changes away from normalized mapping.",
    )
    parser.add_argument(
        "--z-map-flat-penalty",
        type=float,
        default=1.0,
        help="Extra adaptive z-map DP penalty for mapping adjacent observed Z levels to the same reference Z.",
    )
    parser.add_argument(
        "--z-map-prior-weight",
        type=float,
        default=0.25,
        help="Adaptive z-map penalty for moving far from normalized Z mapping.",
    )
    parser.add_argument(
        "--z-map-count-weight",
        type=float,
        default=0.05,
        help="Adaptive z-map penalty for mismatch between observed point count and section point count.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=6,
        help="How many nearest section points are kept as edge candidates before smoothing.",
    )
    parser.add_argument(
        "--distance-weight",
        type=float,
        default=1.0,
        help="Weight for aligned XY distance in the candidate unary score.",
    )
    parser.add_argument(
        "--direction-weight",
        type=float,
        default=0.8,
        help="Weight for mismatch between point trajectory direction and model edge direction.",
    )
    parser.add_argument(
        "--smooth-weight",
        type=float,
        default=1.2,
        help="Weight for Viterbi topology smoothing along Z for each point_index track.",
    )
    parser.add_argument("--no-reverse", action="store_true", help="Do not try reversed cyclic order.")
    parser.add_argument("--output-html", type=Path, default=Path("output/full_circle_point_edge_matches.html"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    use_progress = not args.no_progress
    model_name = discover_single_model(args.data_root, args.model.strip() or None)
    model_path = args.data_root / model_name / "InitialModel"
    shadow_dir = args.data_root / model_name / "shadow"

    model = parse_initial_model(model_path)
    vertices = model.vertices
    edges = unique_edges_from_faces(model.faces)
    triangles = triangulate_faces(model.faces)
    model_center = polyhedron_center_of_mass(vertices, model.faces)

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
    z_step = max(0.0005, float(args.z_step))
    z_levels = np.arange(z_min, z_max + 0.5 * z_step, z_step, dtype=float)

    adaptive_summary: dict[str, object] = {}
    selected_window_sequence: list[int] | None = None
    if args.window_mode == "z-adaptive":
        step = max(1, int(args.window_step))
        w_min = max(2, int(args.window_min))
        w_max = max(w_min, int(args.window_max))
        windows = list(range(w_min, w_max + 1, step))
        if int(args.window) not in windows:
            windows.append(int(args.window))
            windows = sorted(set(windows))
        windows = [max(2, min(len(half_contours), int(w))) for w in windows]
        windows = sorted(set(windows))

        line_points_by_window, fit_rms_by_window, cond_by_window = compute_line_points_multi_window(
            half_contours,
            z_levels,
            windows,
            use_progress=use_progress,
        )
        quality, quality_summary = adaptive_window_quality(
            vertices=vertices,
            edges=edges,
            z_levels=z_levels,
            windows=windows,
            line_points_by_window=line_points_by_window,
            fit_rms_by_window=fit_rms_by_window,
            cond_by_window=cond_by_window,
            point_mode=args.point_mode,
            rms_min_pct=float(args.rms_min_pct),
            transform_mode=args.transform,
            z_map="normalized" if args.z_map == "adaptive" else args.z_map,
            allow_reverse=not args.no_reverse,
            fit_weight=float(args.window_fit_weight),
            condition_weight=float(args.window_condition_weight),
            match_weight=float(args.window_match_weight),
            valid_weight=float(args.window_valid_weight),
        )
        selected_window_indices, selection_summary = select_z_adaptive_windows(
            quality,
            windows,
            smooth_weight=float(args.window_smooth_weight),
        )
        selected_window_sequence = [int(windows[int(i)]) for i in selected_window_indices]
        line_points, fit_rms = apply_selected_windows(
            line_points_by_window,
            fit_rms_by_window,
            selected_window_indices,
        )
        adaptive_summary = {
            "adaptive_windows": windows,
            **quality_summary,
            **selection_summary,
        }
    else:
        line_points, fit_rms = compute_line_points(
            half_contours,
            z_levels,
            int(args.window),
            use_progress=use_progress,
        )

    z_map_summary: dict[str, object] = {}
    z_reference_levels_for_match: np.ndarray | None = None
    if args.z_map == "adaptive":
        model_z_min = float(np.min(vertices[:, 2]))
        model_z_max = float(np.max(vertices[:, 2]))
        n_ref_levels = int(args.z_map_ref_levels) if int(args.z_map_ref_levels) > 1 else int(z_levels.size)
        reference_z_levels = np.linspace(model_z_min, model_z_max, n_ref_levels, dtype=float)
        z_quality, z_quality_summary = adaptive_z_map_quality(
            vertices=vertices,
            edges=edges,
            z_levels=z_levels,
            reference_z_levels=reference_z_levels,
            line_points=line_points,
            fit_rms=fit_rms,
            point_mode=args.point_mode,
            rms_min_pct=float(args.rms_min_pct),
            transform_mode=args.transform,
            allow_reverse=not args.no_reverse,
            band=max(0, int(args.z_map_band)),
            prior_weight=float(args.z_map_prior_weight),
            count_weight=float(args.z_map_count_weight),
        )
        z_reference_levels_for_match, z_selection_summary = select_adaptive_z_map(
            z_quality,
            reference_z_levels,
            smooth_weight=float(args.z_map_smooth_weight),
            max_jump=int(args.z_map_max_jump),
            flat_penalty=float(args.z_map_flat_penalty),
        )
        z_map_summary = {
            **z_quality_summary,
            **z_selection_summary,
        }

    matches, summary = build_matches(
        vertices=vertices,
        edges=edges,
        model_center=model_center,
        z_levels=z_levels,
        line_points=line_points,
        fit_rms=fit_rms,
        point_mode=args.point_mode,
        rms_min_pct=float(args.rms_min_pct),
        transform_mode=args.transform,
        z_map="normalized" if args.z_map == "adaptive" else args.z_map,
        trim_bottom=int(args.trim_bottom),
        trim_top=int(args.trim_top),
        allow_reverse=not args.no_reverse,
        candidate_k=max(1, int(args.candidate_k)),
        distance_weight=float(args.distance_weight),
        direction_weight=float(args.direction_weight),
        smooth_weight=float(args.smooth_weight),
        z_reference_levels=z_reference_levels_for_match,
    )
    summary.update(adaptive_summary)
    summary.update(z_map_summary)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "parameters": {
            "data_root": str(args.data_root),
            "max_contours": args.max_contours,
            "z_step": z_step,
            "window": int(args.window),
            "window_mode": args.window_mode,
            "window_min": int(args.window_min),
            "window_max": int(args.window_max),
            "window_step": int(args.window_step),
            "window_fit_weight": float(args.window_fit_weight),
            "window_condition_weight": float(args.window_condition_weight),
            "window_match_weight": float(args.window_match_weight),
            "window_valid_weight": float(args.window_valid_weight),
            "window_smooth_weight": float(args.window_smooth_weight),
            "point_mode": args.point_mode,
            "rms_min_pct": float(args.rms_min_pct),
            "trim_bottom": int(args.trim_bottom),
            "trim_top": int(args.trim_top),
            "transform": args.transform,
            "z_map": args.z_map,
            "z_map_ref_levels": int(args.z_map_ref_levels),
            "z_map_band": int(args.z_map_band),
            "z_map_max_jump": int(args.z_map_max_jump),
            "z_map_smooth_weight": float(args.z_map_smooth_weight),
            "z_map_flat_penalty": float(args.z_map_flat_penalty),
            "z_map_prior_weight": float(args.z_map_prior_weight),
            "z_map_count_weight": float(args.z_map_count_weight),
            "allow_reverse": not args.no_reverse,
            "candidate_k": max(1, int(args.candidate_k)),
            "distance_weight": float(args.distance_weight),
            "direction_weight": float(args.direction_weight),
            "smooth_weight": float(args.smooth_weight),
        },
        "model_center": as_json_point(model_center),
        "summary": summary,
        "selected_window_sequence": selected_window_sequence,
        "adaptive_z_reference_levels": (
            [as_json_float(float(z)) for z in z_reference_levels_for_match]
            if z_reference_levels_for_match is not None
            else None
        ),
        "viewer": {
            "vertices_display": (vertices - model_center).astype(float).round(6).tolist(),
            "triangles": triangles.astype(int).tolist(),
            "edges": edges.astype(int).tolist(),
        },
        "matches": matches,
    }

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(build_viewer_html(payload, args.output_html), encoding="utf-8")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "matched={matches} levels={levels} distance_xy_median={median} html={html}".format(
            matches=summary["matches"],
            levels=summary["levels_total"],
            median=summary["distance_xy_median"],
            html=args.output_html,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
