from __future__ import annotations

import numpy as np

from polyreco.rms_selection import as_json_float, as_json_point, finite_float


EPS = 1e-9

__all__ = [
    "build_w2_edge_segments",
    "cluster_w2_edge_segments",
    "fit_xy_line_vs_z",
    "is_overlapping_collinear_duplicate_pair",
    "longest_contiguous_count",
    "predict_w2_line",
    "robust_xy_line_fit",
    "skew_line_features",
    "w2_cluster_duplicate_line_features",
    "w2_segment_distance",
    "xy_line_residuals",
]

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

