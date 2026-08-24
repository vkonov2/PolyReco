from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np


REQUIRED_RUNTIME_SYMBOLS: tuple[str, ...] = (
    "EPS",
    "CANONICAL_ORACLE_SCHEMA_VERSION",
    "CANONICAL_ORACLE_TOLERANCES",
    "PlaneFit",
    "as_json_float",
    "as_json_point",
    "fit_plane",
    "orient_plane_from_observed_points",
    "plane_basis",
    "convex_hull_indices",
    "finite_float",
    "candidate_plane",
    "candidate_hull_centroid",
    "canonical_oracle_evaluator_metadata",
    "canonical_candidate_oracle_face",
    "polygon_signed_distances_2d",
    "fit_xy_line_vs_z",
    "xy_line_residuals",
    "longest_contiguous_count",
    "predict_w2_line",
    "build_w2_edge_segments",
    "cluster_w2_edge_segments",
    "is_overlapping_collinear_duplicate_pair",
    "build_w2_face_candidates",
    "point_to_face_polygon_distance",
    "candidate_oracle_face",
    "oracle_candidate_cohort",
    "model_face_planes",
    "robust_xy_line_fit",
    "candidate_track_id",
    "convex_hull_2d",
    "point_to_polygon_distance_2d",
    "evaluate_polyhedron_reprojection",
    "topology_is_valid",
    "value_or_default",
    "oracle_geometry_key",
    "nonoracle_candidate_quality_key",
    "candidate_plane_relation_score",
    "candidate_hull_points",
    "nearest_point_distances",
    "trial_face_geometry_row_incremental",
    "candidate_z_intervals",
    "interval_overlap",
    "distribution_summary",
    "candidate_z_range",
    "hull_bounds_overlap_ratio",
    "candidates_plane_patch_compatible",
    "candidate_outside_mask",
    "annotate_candidate_pool_neutral",
    "prepare_z_level_slices",
    "cumulative_metrics_from_mask",
    "candidate_halfspace",
    "LpActivityEngine",
    "lp_face_activity_details",
    "reconstruct_polyhedron_from_halfspaces_edge_clip",
    "polygon_area",
    "plane_patch_relation",
    "evaluate_trusted_cloud_outside",
    "mesh_geometry_signature",
    "patch_surface_deficit_for_samples",
)


def bind_runtime_symbols(mapping: dict[str, object]) -> None:
    missing = [name for name in REQUIRED_RUNTIME_SYMBOLS if name not in mapping]
    if missing:
        raise RuntimeError(
            "Missing RMS oracle loss diagnostic runtime symbols: "
            + ", ".join(sorted(missing))
        )
    module_globals = globals()
    for name in REQUIRED_RUNTIME_SYMBOLS:
        module_globals[name] = mapping[name]


def model_edges_from_faces(vertices: np.ndarray, faces: list[list[int]]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], set[int]]]:
    edge_faces: dict[tuple[int, int], set[int]] = {}
    for fid, face in enumerate(faces):
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((int(a), int(b))))
            edge_faces.setdefault(edge, set()).add(int(fid))
    return list(edge_faces.keys()), edge_faces


def point_segment_distance_with_t(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ab = b - a
    den = float(ab @ ab)
    if den <= 0.0:
        return float(np.linalg.norm(p - a)), 0.0
    t = float((p - a) @ ab / den)
    t_clamped = min(1.0, max(0.0, t))
    return float(np.linalg.norm(p - (a + t_clamped * ab))), float(t)


def oracle_edge_cluster_matches(
    clusters: list[dict[str, object]],
    vertices: np.ndarray,
    faces: list[list[int]],
) -> dict[str, object]:
    edges, _ = model_edges_from_faces(vertices, faces)
    matches: list[dict[str, object]] = []
    confident_rows: list[dict[str, object]] = []
    matched_ids: list[int] = []
    ambiguous = 0
    for cluster in clusters:
        z0 = finite_float(cluster.get("cluster_z_min"), float("nan"))
        z1 = finite_float(cluster.get("cluster_z_max"), float("nan"))
        if not np.isfinite(z0) or not np.isfinite(z1) or z1 <= z0:
            continue
        z = np.linspace(z0, z1, 7)
        pts = predict_w2_line(cluster, z)
        direction = np.array(cluster.get("direction") or [], dtype=float)
        rows: list[dict[str, object]] = []
        for eid, (a, b) in enumerate(edges):
            va = vertices[int(a)]
            vb = vertices[int(b)]
            evec = vb - va
            enorm = float(np.linalg.norm(evec))
            if enorm <= 0.0 or direction.shape != (3,):
                continue
            distances = []
            inside_count = 0
            for p in pts:
                dist, t_raw = point_segment_distance_with_t(p, va, vb)
                distances.append(dist)
                if 0.0 <= t_raw <= 1.0:
                    inside_count += 1
            angle = float(np.degrees(np.arccos(np.clip(abs(float(direction @ (evec / enorm))), -1.0, 1.0))))
            z_overlap = max(0.0, min(z1, float(max(va[2], vb[2]))) - max(z0, float(min(va[2], vb[2]))))
            row = {
                "edge_rank": 0,
                "edge_id": int(eid),
                "vertices": [int(a), int(b)],
                "median_distance": as_json_float(float(np.median(distances))),
                "max_distance": as_json_float(float(np.max(distances))),
                "angle_deg": as_json_float(float(angle)),
                "z_overlap": as_json_float(float(z_overlap)),
                "inside_fraction": as_json_float(float(inside_count) / float(len(pts))),
                "score": float(np.median(distances)) + 0.01 * angle + max(0.0, 0.5 - float(inside_count) / float(len(pts))),
            }
            rows.append(row)
        rows.sort(key=lambda row: float(row["score"]))
        for rank, row in enumerate(rows[:3]):
            row["edge_rank"] = int(rank + 1)
        top = rows[0] if rows else None
        second = rows[1] if len(rows) > 1 else None
        confident = False
        if top is not None:
            confident = (
                float(top["median_distance"] or float("inf")) <= 0.08
                and float(top["angle_deg"] or 180.0) <= 12.0
                and float(top["inside_fraction"] or 0.0) >= 0.5
            )
            if second is not None and float(second["score"]) - float(top["score"]) < 0.02:
                confident = False
                ambiguous += 1
        if confident and top is not None:
            matched_ids.append(int(top["edge_id"]))
            confident_rows.append(
                {
                    "edge_cluster_id": int(cluster["edge_cluster_id"]) if cluster.get("edge_cluster_id") is not None else -1,
                    "edge_id": int(top["edge_id"]),
                    "vertices": top.get("vertices"),
                    "median_distance": top.get("median_distance"),
                    "angle_deg": top.get("angle_deg"),
                    "inside_fraction": top.get("inside_fraction"),
                }
            )
        matches.append(
            {
                "edge_cluster_id": int(cluster["edge_cluster_id"]) if cluster.get("edge_cluster_id") is not None else -1,
                "cyclic_indices": cluster.get("member_cyclic_indices"),
                "top_matches": rows[:3],
                "confident_match": bool(confident),
                "ambiguous": bool(top is not None and not confident),
            }
        )
    unique_ids = set(matched_ids)
    return {
        "matched_clusters": int(len(matched_ids)),
        "unmatched_or_ambiguous_clusters": int(len(clusters) - len(matched_ids)),
        "ambiguous_clusters": int(ambiguous),
        "unique_oracle_edge_ids": int(len(unique_ids)),
        "duplicate_clusters_on_edges": int(len(matched_ids) - len(unique_ids)),
        "precision_confident": as_json_float(float(len(matched_ids)) / max(float(len(clusters)), 1.0)),
        "confident_matches": confident_rows,
        "cluster_match_lookup": {
            str(int(row["edge_cluster_id"])): row
            for row in matches
            if row.get("edge_cluster_id") is not None and int(row["edge_cluster_id"]) >= 0
        },
        "cluster_matches_sample": matches[:120],
    }


def edge_match_lookup(edge_oracle: dict[str, object]) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    source_rows = list((edge_oracle.get("cluster_match_lookup") or {}).values()) if isinstance(edge_oracle.get("cluster_match_lookup"), dict) else list(edge_oracle.get("cluster_matches_sample", []))
    for row in source_rows:
        cluster_id = row.get("edge_cluster_id")
        cid = int(cluster_id) if cluster_id is not None else -1
        if cid >= 0:
            top = (row.get("top_matches") or [None])[0]
            second = (row.get("top_matches") or [None, None])[1] if len(row.get("top_matches") or []) > 1 else None
            margin = None
            if isinstance(top, dict) and isinstance(second, dict):
                margin = float(second.get("score") or 0.0) - float(top.get("score") or 0.0)
            out[cid] = {
                **row,
                "top_edge_id": int(top["edge_id"]) if isinstance(top, dict) and top.get("edge_id") is not None else None,
                "top_confidence": bool(row.get("confident_match")),
                "score_margin": as_json_float(float(margin)) if margin is not None else None,
                "oracle_purity_proxy": top.get("inside_fraction") if isinstance(top, dict) else None,
                "top_distance": top.get("median_distance") if isinstance(top, dict) else None,
                "top_direction_angle": top.get("angle_deg") if isinstance(top, dict) else None,
                "top_z_overlap": top.get("z_overlap") if isinstance(top, dict) else None,
            }
    return out


def point_edge_distances_vectorized(points: np.ndarray, vertices: np.ndarray, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0 or not edges:
        return np.zeros((points.shape[0], 0), dtype=float), np.zeros((points.shape[0], 0), dtype=float)
    a = vertices[np.array([e[0] for e in edges], dtype=int)]
    b = vertices[np.array([e[1] for e in edges], dtype=int)]
    ab = b - a
    den = np.sum(ab * ab, axis=1)
    den = np.where(den <= 0.0, 1.0, den)
    rel = points[:, None, :] - a[None, :, :]
    t = np.sum(rel * ab[None, :, :], axis=2) / den[None, :]
    t_clip = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t_clip[:, :, None] * ab[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return distances, t


def edge_line_distances(points: np.ndarray, vertices: np.ndarray, edge: tuple[int, int]) -> np.ndarray:
    a = vertices[int(edge[0])]
    b = vertices[int(edge[1])]
    ab = b - a
    den = float(ab @ ab)
    if den <= 0.0:
        return np.linalg.norm(points - a[None, :], axis=1)
    t = ((points - a[None, :]) @ ab) / den
    closest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - closest, axis=1)


def edge_to_missing_faces(edge_id: int, edge_id_to_faces: dict[int, set[int]], core_face_ids: set[int]) -> list[int]:
    return sorted(int(fid) for fid in edge_id_to_faces.get(int(edge_id), set()) if int(fid) not in core_face_ids)


def oracle_run_edge_match(
    *,
    points: np.ndarray,
    fit: dict[str, object],
    vertices: np.ndarray,
    edges: list[tuple[int, int]],
    edge_id_to_faces: dict[int, set[int]],
    core_face_ids: set[int],
) -> dict[str, object]:
    if points.shape[0] == 0 or not edges:
        return {"dominant_edge_id": None, "dominant_edge_fraction": 0.0}
    distances, t_values = point_edge_distances_vectorized(points, vertices, edges)
    nearest = np.argmin(distances, axis=1)
    counts = np.bincount(nearest, minlength=len(edges))
    dominant = int(np.argmax(counts))
    dominant_fraction = float(counts[dominant]) / float(points.shape[0])
    dominant_distances = distances[:, dominant]
    dominant_t = t_values[:, dominant]
    edge = edges[dominant]
    va = vertices[int(edge[0])]
    vb = vertices[int(edge[1])]
    evec = vb - va
    enorm = float(np.linalg.norm(evec))
    raw_direction = fit.get("direction")
    direction = np.array(raw_direction if raw_direction is not None else [], dtype=float)
    angle = 180.0
    if enorm > 0.0 and direction.shape == (3,):
        angle = float(np.degrees(np.arccos(np.clip(abs(float(direction @ (evec / enorm))), -1.0, 1.0))))
    finite_inside = (dominant_t >= 0.0) & (dominant_t <= 1.0)
    line_distances = edge_line_distances(points, vertices, edge)
    missing_faces = edge_to_missing_faces(dominant, edge_id_to_faces, core_face_ids)
    confident = (
        float(np.median(dominant_distances)) <= 0.08
        and angle <= 12.0
        and float(np.mean(finite_inside)) >= 0.5
        and dominant_fraction >= 0.65
    )
    return {
        "dominant_edge_id": int(dominant),
        "dominant_edge_vertices": [int(edge[0]), int(edge[1])],
        "dominant_edge_fraction": as_json_float(dominant_fraction),
        "oracle_edge_id_count": int(np.count_nonzero(counts)),
        "dominant_edge_distance_median": as_json_float(float(np.median(dominant_distances))),
        "dominant_edge_distance_p95": as_json_float(float(np.percentile(dominant_distances, 95))),
        "angle_to_dominant_edge_deg": as_json_float(float(angle)),
        "true_edge_line_residual_median": as_json_float(float(np.median(line_distances))),
        "true_edge_line_residual_p95": as_json_float(float(np.percentile(line_distances, 95))),
        "finite_edge_inside_fraction": as_json_float(float(np.mean(finite_inside))),
        "finite_edge_t_coverage": as_json_float(float(np.max(np.clip(dominant_t, 0.0, 1.0)) - np.min(np.clip(dominant_t, 0.0, 1.0)))),
        "missing_canonical_face_ids": missing_faces,
        "belongs_to_missing_canonical_face": bool(missing_faces),
        "oracle_confident_finite_edge": bool(confident),
    }


def summarize_edge_ceiling(
    edge_ids: set[int],
    *,
    edge_id_to_faces: dict[int, set[int]],
    core_face_ids: set[int],
) -> dict[str, object]:
    face_edge_support: dict[int, set[int]] = {}
    missing_boundary_edges = 0
    for eid in edge_ids:
        faces_for_edge = edge_id_to_faces.get(int(eid), set())
        if any(int(fid) not in core_face_ids for fid in faces_for_edge):
            missing_boundary_edges += 1
        for fid in faces_for_edge:
            face_edge_support.setdefault(int(fid), set()).add(int(eid))
    faces_two = {fid for fid, support in face_edge_support.items() if len(support) >= 2}
    return {
        "unique_finite_edge_ids": int(len(edge_ids)),
        "boundary_edges_of_missing_core_faces": int(missing_boundary_edges),
        "faces_with_two_recoverable_boundary_edges": int(len(faces_two)),
        "potential_new_canonical_face_ids": int(len(faces_two - core_face_ids)),
        "face_ids_sample": sorted(int(v) for v in faces_two)[:120],
        "new_face_ids_sample": sorted(int(v) for v in (faces_two - core_face_ids))[:120],
    }


def collect_line_residual_oracle_diagnostics(
    *,
    z_levels: np.ndarray,
    line_points_w2: np.ndarray,
    cond_w2: np.ndarray | None,
    accepted_segments: list[dict[str, object]],
    vertices: np.ndarray,
    faces: list[list[int]],
    core_face_ids: set[int],
    min_levels: int,
    min_z_span: float,
    max_line_rms: float,
    max_line_residual: float,
    max_z_gap: int,
    max_point_jump: float,
    max_condition: float,
    cluster_mode: str,
    cluster_max_distance: float,
    cluster_max_angle_deg: float,
    cluster_min_overlap: float,
) -> dict[str, object]:
    edges, model_edge_faces = model_edges_from_faces(vertices, faces)
    edge_id_to_faces: dict[int, set[int]] = {int(eid): set(face_ids) for eid, face_ids in enumerate(model_edge_faces.values())}
    ls_edge_ids: set[int] = set()
    for segment in accepted_segments:
        z_indices = [int(v) for v in segment.get("z_indices") or []]
        if not z_indices:
            continue
        pts = line_points_w2[int(segment.get("cyclic_index") or 0), np.array(z_indices, dtype=int), :].astype(float)
        fit = {
            "coef_x": np.array(segment.get("coef_x") or [], dtype=float),
            "coef_y": np.array(segment.get("coef_y") or [], dtype=float),
            "direction": np.array(segment.get("direction") or [], dtype=float),
        }
        if fit["coef_x"].shape != (2,) or fit["coef_y"].shape != (2,):
            continue
        match = oracle_run_edge_match(
            points=pts,
            fit=fit,
            vertices=vertices,
            edges=edges,
            edge_id_to_faces=edge_id_to_faces,
            core_face_ids=core_face_ids,
        )
        if bool(match.get("oracle_confident_finite_edge")) and match.get("dominant_edge_id") is not None:
            ls_edge_ids.add(int(match["dominant_edge_id"]))

    examined = 0
    rejection_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    robust_edge_ids: set[int] = set()
    split_edge_ids: set[int] = set()
    perfect_edge_ids: set[int] = set()
    recovered_segments: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    belt_rows: list[dict[str, object]] = []
    n_half = int(line_points_w2.shape[0])

    def add_count(d: dict[str, int], key: str) -> None:
        d[key] = d.get(key, 0) + 1

    def make_recovered_segment(
        *,
        idx: int,
        z_indices: list[int],
        z: np.ndarray,
        pts: np.ndarray,
        fit: dict[str, object],
        source: str,
    ) -> dict[str, object]:
        residual = xy_line_residuals(z, pts, fit)
        return {
            "segment_id": int(1000000 + len(recovered_segments)),
            "cyclic_index": int(idx),
            "z_indices": [int(v) for v in z_indices],
            "levels": int(len(z_indices)),
            "z_min": as_json_float(float(np.min(z))),
            "z_max": as_json_float(float(np.max(z))),
            "z_span": as_json_float(float(np.max(z) - np.min(z))),
            "coef_x": [float(v) for v in np.array(fit["coef_x"], dtype=float)],
            "coef_y": [float(v) for v in np.array(fit["coef_y"], dtype=float)],
            "direction": as_json_point(np.array(fit["direction"], dtype=float)),
            "line_rms": as_json_float(float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0),
            "line_max_residual": as_json_float(float(np.max(residual)) if residual.size else 0.0),
            "first_diff_smoothness": as_json_float(float(fit.get("first_diff_smoothness") or 0.0)),
            "second_diff_smoothness": as_json_float(float(fit.get("second_diff_smoothness") or 0.0)),
            "condition_median": None,
            "condition_p95": None,
            "w3_agreement_median": None,
            "w3_minima_fraction": None,
            "confidence": as_json_float(1.0 / (1.0 + float(np.sqrt(np.mean(residual * residual))) / max(float(max_line_rms), EPS))),
            "bounds_min": as_json_point(np.min(pts, axis=0)),
            "bounds_max": as_json_point(np.max(pts, axis=0)),
            "diagnostic_recovery_source": source,
        }

    def classify(match: dict[str, object], robust: dict[str, object] | None, split_matches: list[dict[str, object]], residual_jump: float) -> str:
        purity = finite_float(match.get("dominant_edge_fraction"), 0.0)
        distinct = int(match.get("oracle_edge_id_count") or 0)
        median_dist = finite_float(match.get("dominant_edge_distance_median"), float("inf"))
        edge_id = match.get("dominant_edge_id")
        horizontal = False
        if edge_id is not None:
            a, b = edges[int(edge_id)]
            evec = vertices[int(b)] - vertices[int(a)]
            enorm = float(np.linalg.norm(evec))
            horizontal = bool(enorm > 0.0 and abs(float(evec[2])) / enorm < 0.12)
        if purity < 0.45 or median_dist > 0.18:
            return "false_raw_support"
        if distinct >= 3 and purity < 0.75:
            return "mixed_multiple_edges"
        if horizontal:
            return "near_horizontal_parameterization"
        if robust is not None and bool(match.get("oracle_confident_finite_edge")):
            return "straight_track_with_outliers"
        if split_matches:
            return "piecewise_linear_same_edge"
        if robust is None and purity >= 0.65 and residual_jump > float(max_line_residual):
            return "insufficient_contiguous_inliers"
        return "nonlinear_or_unstable"

    def analyze_run(idx: int, current: list[int]) -> None:
        nonlocal examined
        if len(current) < int(min_levels):
            return
        examined += 1
        z = z_levels[np.array(current, dtype=int)]
        pts = line_points_w2[int(idx), np.array(current, dtype=int), :].astype(float)
        cond_vals = cond_w2[int(idx), np.array(current, dtype=int)].astype(float) if cond_w2 is not None else None
        weights = None
        if cond_vals is not None:
            weights = 1.0 / np.sqrt(np.maximum(np.nan_to_num(cond_vals, nan=np.nanmedian(cond_vals)), 1.0))
        fit = fit_xy_line_vs_z(z, pts, weights=weights)
        if fit is None:
            add_count(rejection_counts, "fit_failed")
            return
        z_span = float(np.max(z) - np.min(z)) if z.size else 0.0
        residual = xy_line_residuals(z, pts, fit)
        reason = "accepted"
        if z_span < float(min_z_span):
            reason = "short_z_span"
        elif fit["rms"] > float(max_line_rms) or fit["max_residual"] > float(max_line_residual):
            reason = "line_residual_rejected"
        elif cond_vals is not None:
            cond_median = float(np.nanmedian(cond_vals)) if cond_vals.size else float("nan")
            if max_condition >= 0.0 and np.isfinite(cond_median) and cond_median > float(max_condition):
                reason = "condition_rejected"
        if reason != "line_residual_rejected":
            add_count(rejection_counts, reason)
            return
        add_count(rejection_counts, reason)
        residual_jump = float(np.max(np.abs(np.diff(residual)))) if residual.size >= 2 else 0.0
        match = oracle_run_edge_match(
            points=pts,
            fit=fit,
            vertices=vertices,
            edges=edges,
            edge_id_to_faces=edge_id_to_faces,
            core_face_ids=core_face_ids,
        )
        if bool(match.get("oracle_confident_finite_edge")) and match.get("dominant_edge_id") is not None:
            perfect_edge_ids.add(int(match["dominant_edge_id"]))
        robust = robust_xy_line_fit(
            z,
            pts,
            min_levels=min_levels,
            min_z_span=min_z_span,
            max_line_rms=max_line_rms,
            max_line_residual=max_line_residual,
            max_z_gap=max_z_gap,
        )
        robust_match: dict[str, object] | None = None
        if robust is not None:
            mask = np.array(robust["inlier_mask"], dtype=bool)
            recovered_segments.append(
                make_recovered_segment(
                    idx=idx,
                    z_indices=[int(current[i]) for i in np.nonzero(mask)[0]],
                    z=z[mask],
                    pts=pts[mask],
                    fit=robust["fit"],
                    source="robust_inlier",
                )
            )
            robust_match = oracle_run_edge_match(
                points=pts[mask],
                fit=robust["fit"],
                vertices=vertices,
                edges=edges,
                edge_id_to_faces=edge_id_to_faces,
                core_face_ids=core_face_ids,
            )
            if bool(robust_match.get("oracle_confident_finite_edge")) and robust_match.get("dominant_edge_id") is not None:
                robust_edge_ids.add(int(robust_match["dominant_edge_id"]))
        split_matches: list[dict[str, object]] = []
        if residual.size >= 2:
            jump_order = np.argsort(-np.abs(np.diff(residual)))[:1]
            seen_split: set[int] = set()
            for jump_i in jump_order:
                split_at = int(jump_i) + 1
                if split_at in seen_split or split_at < int(min_levels) or len(current) - split_at < int(min_levels):
                    continue
                seen_split.add(split_at)
                for part in (slice(0, split_at), slice(split_at, len(current))):
                    sub_z = z[part]
                    sub_pts = pts[part]
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
                    recovered_segments.append(
                        make_recovered_segment(
                            idx=idx,
                            z_indices=[int(part_indices[i]) for i in np.nonzero(sub_mask)[0]],
                            z=sub_z[sub_mask],
                            pts=sub_pts[sub_mask],
                            fit=sub_robust["fit"],
                            source="one_split",
                        )
                    )
                    sub_match = oracle_run_edge_match(
                        points=sub_pts[sub_mask],
                        fit=sub_robust["fit"],
                        vertices=vertices,
                        edges=edges,
                        edge_id_to_faces=edge_id_to_faces,
                        core_face_ids=core_face_ids,
                    )
                    if bool(sub_match.get("oracle_confident_finite_edge")) and sub_match.get("dominant_edge_id") is not None:
                        split_edge_ids.add(int(sub_match["dominant_edge_id"]))
                        split_matches.append(sub_match)
        category = classify(match, robust, split_matches, residual_jump)
        add_count(category_counts, category)
        row = {
            "cyclic_index": int(idx),
            "points": int(len(current)),
            "z_levels": int(len(current)),
            "z_span": as_json_float(float(z_span)),
            "z_min": as_json_float(float(np.min(z))),
            "z_max": as_json_float(float(np.max(z))),
            "longest_contiguous_z_run": int(longest_contiguous_count(current, max_gap=1)),
            "ls_residual_median": as_json_float(float(np.median(residual))),
            "ls_residual_p95": as_json_float(float(np.percentile(residual, 95))),
            "ls_residual_max": as_json_float(float(np.max(residual))),
            "max_residual_jump": as_json_float(float(residual_jump)),
            "large_internal_z_gaps": int(np.sum(np.diff(np.array(current, dtype=int)) > 1)) if len(current) >= 2 else 0,
            "condition_median": as_json_float(float(np.nanmedian(cond_vals))) if cond_vals is not None and cond_vals.size else None,
            "condition_p95": as_json_float(float(np.nanpercentile(cond_vals, 95))) if cond_vals is not None and cond_vals.size else None,
            "fitted_direction": as_json_point(np.array(fit["direction"], dtype=float)),
            "rejection_reason": reason,
            "oracle_category": category,
            "zone": "belt_z_minus_5" if float(np.min(z)) <= -5.0 <= float(np.max(z)) else "other",
            **match,
            "robust_recovered": bool(robust_match is not None and robust_match.get("oracle_confident_finite_edge")),
            "robust_inlier_fraction": as_json_float(float(robust["inlier_fraction"])) if robust is not None else None,
            "robust_inlier_levels": int(robust["inlier_levels"]) if robust is not None else 0,
            "robust_inlier_z_span": as_json_float(float(robust["inlier_z_span"])) if robust is not None else None,
            "robust_residual_median": as_json_float(float(robust["inlier_residual_median"])) if robust is not None else None,
            "robust_residual_p95": as_json_float(float(robust["inlier_residual_p95"])) if robust is not None else None,
            "split_recovered": bool(split_matches),
            "split_recovered_edge_ids": sorted({int(m["dominant_edge_id"]) for m in split_matches if m.get("dominant_edge_id") is not None}),
        }
        if len(rows) < 240:
            rows.append(row)
        if row["zone"] == "belt_z_minus_5" and len(belt_rows) < 80:
            belt_rows.append(row)

    for idx in range(n_half):
        current: list[int] = []
        for zi in range(int(z_levels.size)):
            p = line_points_w2[int(idx), int(zi), :]
            if not np.all(np.isfinite(p)):
                analyze_run(idx, current)
                current = []
                continue
            if current:
                if int(zi) - current[-1] > int(max_z_gap):
                    analyze_run(idx, current)
                    current = []
                else:
                    prev = line_points_w2[int(idx), current[-1], :]
                    if float(np.linalg.norm(p - prev)) > float(max_point_jump):
                        analyze_run(idx, current)
                        current = []
            current.append(int(zi))
            if len(current) >= int(min_levels):
                z = z_levels[np.array(current, dtype=int)]
                pts = line_points_w2[int(idx), np.array(current, dtype=int), :].astype(float)
                fit = fit_xy_line_vs_z(z, pts)
                if fit is not None and (fit["rms"] > float(max_line_rms) * 1.5 or fit["max_residual"] > float(max_line_residual) * 1.5):
                    last = current.pop()
                    analyze_run(idx, current)
                    current = [last]
        analyze_run(idx, current)

    robust_plus_split = set(robust_edge_ids) | set(split_edge_ids)
    perfect_plus_ls = set(ls_edge_ids) | set(perfect_edge_ids)
    ls_clusters, _ = cluster_w2_edge_segments(
        accepted_segments,
        mode=str(cluster_mode),
        max_distance=float(cluster_max_distance),
        max_angle_deg=float(cluster_max_angle_deg),
        min_overlap=float(cluster_min_overlap),
    )
    robust_segments_all = list(accepted_segments) + recovered_segments
    robust_clusters, _ = cluster_w2_edge_segments(
        robust_segments_all,
        mode=str(cluster_mode),
        max_distance=float(cluster_max_distance),
        max_angle_deg=float(cluster_max_angle_deg),
        min_overlap=float(cluster_min_overlap),
    )
    ls_cluster_oracle = oracle_edge_cluster_matches(ls_clusters, vertices, faces)
    robust_cluster_oracle = oracle_edge_cluster_matches(robust_clusters, vertices, faces)

    def cluster_ceiling(cluster_oracle: dict[str, object]) -> dict[str, object]:
        ids = {int(row["edge_id"]) for row in cluster_oracle.get("confident_matches", []) if row.get("edge_id") is not None}
        return summarize_edge_ceiling(ids, edge_id_to_faces=edge_id_to_faces, core_face_ids=core_face_ids)

    return {
        "runs_examined": int(examined),
        "rejection_counts": rejection_counts,
        "line_residual_rejected_count": int(rejection_counts.get("line_residual_rejected", 0)),
        "category_counts": category_counts,
        "samples": rows,
        "belt_z_minus_5_samples": belt_rows,
        "ceilings": {
            "current_ls_segment_ceiling": summarize_edge_ceiling(ls_edge_ids, edge_id_to_faces=edge_id_to_faces, core_face_ids=core_face_ids),
            "robust_inlier_segment_ceiling": summarize_edge_ceiling(set(ls_edge_ids) | set(robust_edge_ids), edge_id_to_faces=edge_id_to_faces, core_face_ids=core_face_ids),
            "one_split_robust_ceiling": summarize_edge_ceiling(set(ls_edge_ids) | robust_plus_split, edge_id_to_faces=edge_id_to_faces, core_face_ids=core_face_ids),
            "oracle_perfect_subset_ceiling": summarize_edge_ceiling(perfect_plus_ls, edge_id_to_faces=edge_id_to_faces, core_face_ids=core_face_ids),
        },
        "segment_cluster_comparison": {
            "least_squares": {
                "kept_segments": int(len(accepted_segments)),
                "edge_clusters": int(len(ls_clusters)),
                "confident_matched_clusters": int(ls_cluster_oracle.get("matched_clusters") or 0),
                "unique_matched_finite_edge_ids": int(ls_cluster_oracle.get("unique_oracle_edge_ids") or 0),
                "duplicate_clusters_per_edge": int(ls_cluster_oracle.get("duplicate_clusters_on_edges") or 0),
                "ambiguous_clusters": int(ls_cluster_oracle.get("ambiguous_clusters") or 0),
                "ideal_pairing_ceiling": cluster_ceiling(ls_cluster_oracle),
            },
            "robust_split_diagnostic": {
                "kept_segments": int(len(robust_segments_all)),
                "recovered_segments": int(len(recovered_segments)),
                "edge_clusters": int(len(robust_clusters)),
                "confident_matched_clusters": int(robust_cluster_oracle.get("matched_clusters") or 0),
                "unique_matched_finite_edge_ids": int(robust_cluster_oracle.get("unique_oracle_edge_ids") or 0),
                "duplicate_clusters_per_edge": int(robust_cluster_oracle.get("duplicate_clusters_on_edges") or 0),
                "ambiguous_clusters": int(robust_cluster_oracle.get("ambiguous_clusters") or 0),
                "ideal_pairing_ceiling": cluster_ceiling(robust_cluster_oracle),
            },
        },
        "recovered_edges": {
            "ls_edge_ids": sorted(int(v) for v in ls_edge_ids)[:300],
            "robust_edge_ids": sorted(int(v) for v in robust_edge_ids)[:300],
            "split_edge_ids": sorted(int(v) for v in split_edge_ids)[:300],
            "oracle_perfect_rejected_edge_ids": sorted(int(v) for v in perfect_edge_ids)[:300],
        },
        "z_belt_minus5": {
            "line_residual_rejected_sampled": int(sum(1 for row in rows if row.get("zone") == "belt_z_minus_5")),
            "category_counts_sampled": {
                key: int(sum(1 for row in rows if row.get("zone") == "belt_z_minus_5" and row.get("oracle_category") == key))
                for key in sorted(category_counts)
            },
        },
    }


def dispatch_post_multiscale_oracle_scope(
    *,
    scope: str,
    environment: Mapping[str, str],
    args: object,
    model_name: str,
    vertices: np.ndarray,
    faces: list[list[int]],
    contours: list[object],
    all_candidates: list[dict[str, object]],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
) -> bool:
    if scope == "current-candidate-exchange":
        baseline_json_path = environment.get("POLYRECO_CURRENT_CANDIDATE_EXCHANGE_BASELINE_JSON")
        if not baseline_json_path:
            raise RuntimeError(
                "POLYRECO_ORACLE_DIAGNOSTIC_SCOPE=current-candidate-exchange requires "
                "POLYRECO_CURRENT_CANDIDATE_EXCHANGE_BASELINE_JSON"
            )
        try:
            baseline_payload = json.loads(Path(baseline_json_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"failed to read current-candidate exchange baseline JSON {baseline_json_path!r}: {exc!r}"
            ) from exc
        baseline_candidates = [
            candidate for candidate in baseline_payload.get("face_candidates", [])
            if isinstance(candidate, dict)
        ]
        baseline_reconstructed = baseline_payload.get("reconstructed")
        if not baseline_candidates or not isinstance(baseline_reconstructed, dict):
            raise RuntimeError(
                "current-candidate exchange baseline JSON must contain face_candidates and reconstructed"
            )
        target_face_ids_by_model = {
            "pear": [107, 113, 118, 120, 122, 125, 130, 150, 196, 214],
            "cushion": [0, 3, 20, 119, 218, 256, 285],
        }
        exchange_output = Path(
            environment.get(
                "POLYRECO_CURRENT_CANDIDATE_EXCHANGE_OUTPUT_JSON",
                f"/private/tmp/{model_name}_current_candidate_exchange.json",
            )
        )
        diagnostic_payload = diagnose_current_candidate_exchange(
            model_name=model_name,
            vertices=vertices,
            model_faces=model_face_planes(vertices, faces),
            final_candidates=baseline_candidates,
            final_reconstructed=baseline_reconstructed,
            trusted_points=trusted_points,
            trusted_z_indices=trusted_z_indices,
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
            contours=contours,
            target_face_ids=target_face_ids_by_model.get(str(model_name), []),
            baseline_payload=baseline_payload,
            max_full_trials=30,
        )
        diagnostic_payload["diagnostic_output_path"] = str(exchange_output)
        diagnostic_payload["baseline_json_path"] = baseline_json_path
        diagnostic_payload["fast_path"] = "baseline_final_state_plus_in_process_trusted_cloud_before_selection"
        exchange_output.parent.mkdir(parents=True, exist_ok=True)
        exchange_output.write_text(
            json.dumps(diagnostic_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "current-candidate-exchange model={model} targets={targets} trials={trials} output={output}".format(
                model=model_name,
                targets=len(target_face_ids_by_model.get(str(model_name), [])),
                trials=int((diagnostic_payload.get("full_edge_clip_trials") or {}).get("trial_count") or 0),
                output=exchange_output,
            )
        )
        return True
    if scope == "observed-symmetry-orbit-audit":
        baseline_json_path = environment.get("POLYRECO_SYMMETRY_ORBIT_BASELINE_JSON")
        if not baseline_json_path:
            raise RuntimeError(
                "POLYRECO_ORACLE_DIAGNOSTIC_SCOPE=observed-symmetry-orbit-audit requires "
                "POLYRECO_SYMMETRY_ORBIT_BASELINE_JSON"
            )
        try:
            baseline_payload = json.loads(Path(baseline_json_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"failed to read symmetry orbit baseline JSON {baseline_json_path!r}: {exc!r}"
            ) from exc
        baseline_candidates = [
            candidate for candidate in baseline_payload.get("face_candidates", [])
            if isinstance(candidate, dict)
        ]
        baseline_reconstructed = baseline_payload.get("reconstructed")
        if not baseline_candidates or not isinstance(baseline_reconstructed, dict):
            raise RuntimeError(
                "symmetry orbit baseline JSON must contain face_candidates and reconstructed"
            )
        symmetry_output = Path(
            environment.get(
                "POLYRECO_SYMMETRY_ORBIT_OUTPUT_JSON",
                f"/private/tmp/{model_name}_symmetry_orbit_audit.json",
            )
        )
        diagnostic_payload = observed_symmetry_diagnostics(
            model_name=model_name,
            vertices=vertices,
            model_faces=model_face_planes(vertices, faces),
            final_candidates=baseline_candidates,
            candidate_pool=all_candidates,
            final_reconstructed=baseline_reconstructed,
            trusted_points=trusted_points,
            trusted_z_indices=trusted_z_indices,
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
            contours=contours,
            baseline_payload=baseline_payload,
            baseline_json_path=str(baseline_json_path),
            max_existing_trials=int(environment.get("POLYRECO_SYMMETRY_ORBIT_MAX_EXISTING_TRIALS", "15")),
            max_synthetic_trials=int(environment.get("POLYRECO_SYMMETRY_ORBIT_MAX_SYNTHETIC_TRIALS", "10")),
        )
        diagnostic_payload["diagnostic_output_path"] = str(symmetry_output)
        diagnostic_payload["fast_path"] = "baseline_final_state_plus_in_process_stable_two_scales_trusted_cloud_before_selection"
        symmetry_output.parent.mkdir(parents=True, exist_ok=True)
        symmetry_output.write_text(
            json.dumps(diagnostic_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        existing_trials = diagnostic_payload.get("existing_candidate_orbit_completion") if isinstance(diagnostic_payload.get("existing_candidate_orbit_completion"), dict) else {}
        synthetic_trials = diagnostic_payload.get("synthetic_transformed_hypotheses") if isinstance(diagnostic_payload.get("synthetic_transformed_hypotheses"), dict) else {}
        print(
            "observed-symmetry-orbit-audit model={model} existing_trials={existing} synthetic_trials={synthetic} output={output}".format(
                model=model_name,
                existing=int(existing_trials.get("trial_count") or 0),
                synthetic=int(synthetic_trials.get("trial_count") or 0),
                output=symmetry_output,
            )
        )
        return True
    if scope == "active-plane-pruning-corrected":
        baseline_json_path = environment.get("POLYRECO_ACTIVE_PLANE_PRUNING_BASELINE_JSON")
        if not baseline_json_path:
            raise RuntimeError(
                "POLYRECO_ORACLE_DIAGNOSTIC_SCOPE=active-plane-pruning-corrected requires "
                "POLYRECO_ACTIVE_PLANE_PRUNING_BASELINE_JSON"
            )
        try:
            baseline_payload = json.loads(Path(baseline_json_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"failed to read active-plane pruning baseline JSON {baseline_json_path!r}: {exc!r}"
            ) from exc
        baseline_candidates = [
            candidate for candidate in baseline_payload.get("face_candidates", [])
            if isinstance(candidate, dict)
        ]
        baseline_reconstructed = baseline_payload.get("reconstructed")
        if not baseline_candidates or not isinstance(baseline_reconstructed, dict):
            raise RuntimeError(
                "active-plane pruning baseline JSON must contain face_candidates and reconstructed"
            )
        previous_pruning_json_path = environment.get(
            "POLYRECO_ACTIVE_PLANE_PRUNING_PROPOSAL_JSON",
            f"/private/tmp/{model_name}_active_plane_pruning.json",
        )
        previous_pruning_payload: dict[str, object] | None = None
        if previous_pruning_json_path:
            try:
                previous_pruning_payload = json.loads(Path(previous_pruning_json_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous_pruning_payload = None
        pruning_output = Path(
            environment.get(
                "POLYRECO_ACTIVE_PLANE_PRUNING_OUTPUT_JSON",
                f"/private/tmp/{model_name}_active_plane_pruning_corrected.json",
            )
        )
        diagnostic_payload = diagnose_active_plane_pruning_corrected(
            model_name=model_name,
            vertices=vertices,
            model_faces=model_face_planes(vertices, faces),
            final_candidates=baseline_candidates,
            final_reconstructed=baseline_reconstructed,
            trusted_points=trusted_points,
            trusted_z_indices=trusted_z_indices,
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
            contours=contours,
            baseline_payload=baseline_payload,
            previous_pruning_payload=previous_pruning_payload,
            baseline_json_path=str(baseline_json_path),
            previous_pruning_json_path=str(previous_pruning_json_path),
            max_full_trials=int(environment.get("POLYRECO_ACTIVE_PLANE_PRUNING_MAX_TRIALS", "25")),
        )
        diagnostic_payload["diagnostic_output_path"] = str(pruning_output)
        diagnostic_payload["fast_path"] = "baseline_final_state_plus_in_process_stable_two_scales_trusted_cloud_before_selection"
        pruning_output.parent.mkdir(parents=True, exist_ok=True)
        pruning_output.write_text(
            json.dumps(diagnostic_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        trials = diagnostic_payload.get("remove_one_full_trials") if isinstance(diagnostic_payload.get("remove_one_full_trials"), dict) else {}
        print(
            "active-plane-pruning-corrected model={model} trials={trials} safe={safe} output={output}".format(
                model=model_name,
                trials=int(trials.get("trial_count") or 0),
                safe=int(trials.get("safe_count") or 0),
                output=pruning_output,
            )
        )
        return True
    return False


def dispatch_post_w2_oracle_scope(
    *,
    scope: str,
    environment: Mapping[str, str],
    args: object,
    model_name: str,
    vertices: np.ndarray,
    faces: list[list[int]],
    contours: list[object],
    z_levels: np.ndarray,
    w2_line_points: np.ndarray,
    w2_segments: list[dict[str, object]],
    w2_clusters: list[dict[str, object]],
    w2_candidates: list[dict[str, object]],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
) -> bool:
    if scope == "current-candidate-exchange":
        baseline_json_path = environment.get("POLYRECO_CURRENT_CANDIDATE_EXCHANGE_BASELINE_JSON")
        if not baseline_json_path:
            raise RuntimeError(
                "POLYRECO_ORACLE_DIAGNOSTIC_SCOPE=current-candidate-exchange requires "
                "POLYRECO_CURRENT_CANDIDATE_EXCHANGE_BASELINE_JSON for the bounded fast path"
            )
        try:
            baseline_payload = json.loads(Path(baseline_json_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"failed to read current-candidate exchange baseline JSON {baseline_json_path!r}: {exc!r}"
            ) from exc
        baseline_candidates = [
            candidate for candidate in baseline_payload.get("face_candidates", [])
            if isinstance(candidate, dict)
        ]
        baseline_reconstructed = baseline_payload.get("reconstructed")
        if not baseline_candidates or not isinstance(baseline_reconstructed, dict):
            raise RuntimeError(
                "current-candidate exchange baseline JSON must contain face_candidates and reconstructed"
            )
        target_face_ids_by_model = {
            "pear": [107, 113, 118, 120, 122, 125, 130, 150, 196, 214],
            "cushion": [0, 3, 20, 119, 218, 256, 285],
        }
        exchange_output = Path(
            environment.get(
                "POLYRECO_CURRENT_CANDIDATE_EXCHANGE_OUTPUT_JSON",
                f"/private/tmp/{model_name}_current_candidate_exchange.json",
            )
        )
        diagnostic_payload = diagnose_current_candidate_exchange(
            model_name=model_name,
            vertices=vertices,
            model_faces=model_face_planes(vertices, faces),
            final_candidates=baseline_candidates,
            final_reconstructed=baseline_reconstructed,
            trusted_points=trusted_points,
            trusted_z_indices=trusted_z_indices,
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
            contours=contours,
            target_face_ids=target_face_ids_by_model.get(str(model_name), []),
            baseline_payload=baseline_payload,
            max_full_trials=30,
        )
        diagnostic_payload["diagnostic_output_path"] = str(exchange_output)
        diagnostic_payload["baseline_json_path"] = baseline_json_path
        diagnostic_payload["fast_path"] = "baseline_final_state_plus_in_process_trusted_cloud"
        exchange_output.parent.mkdir(parents=True, exist_ok=True)
        exchange_output.write_text(
            json.dumps(diagnostic_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "current-candidate-exchange model={model} targets={targets} trials={trials} output={output}".format(
                model=model_name,
                targets=len(target_face_ids_by_model.get(str(model_name), [])),
                trials=int((diagnostic_payload.get("full_edge_clip_trials") or {}).get("trial_count") or 0),
                output=exchange_output,
            )
        )
        return True
    if scope in {"generic-cross-view-edge-pool", "generic-edge-additive-control"}:
        additive_scope = scope == "generic-edge-additive-control"
        diagnostic_output = Path(
            environment.get(
                "POLYRECO_GENERIC_EDGE_OUTPUT_JSON",
                f"/private/tmp/{model_name}_{'generic_edge_additive' if additive_scope else 'generic_cross_view_edges'}.json",
            )
        )
        baseline_payload = None
        baseline_json_path = environment.get("POLYRECO_GENERIC_EDGE_BASELINE_JSON")
        if baseline_json_path:
            try:
                baseline_payload = json.loads(Path(baseline_json_path).read_text())
            except (OSError, json.JSONDecodeError) as exc:
                baseline_payload = {"_load_error": repr(exc), "_path": str(baseline_json_path)}
        diagnostic_payload = diagnose_generic_cross_view_edge_pool(
            model_name=model_name,
            vertices=vertices,
            faces=faces,
            z_levels=z_levels,
            line_points_w2=w2_line_points,
            w2_segments=w2_segments,
            w2_clusters=w2_clusters,
            w2_candidates=w2_candidates,
            baseline_payload=baseline_payload,
            additive_mode=additive_scope,
        )
        diagnostic_payload["diagnostic_output_path"] = str(diagnostic_output)
        diagnostic_payload["baseline_json_path"] = baseline_json_path
        diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_output.write_text(
            json.dumps(diagnostic_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "{scope} model={model} primitives={primitives} hypotheses={hypotheses} output={output}".format(
                scope=scope,
                model=model_name,
                primitives=int((diagnostic_payload.get("primitive_pool") or {}).get("materialized_primitives") or 0),
                hypotheses=int((diagnostic_payload.get("consensus") or {}).get("kept_hypotheses") or 0),
                output=diagnostic_output,
            )
        )
        return True
    return False


def dispatch_final_mesh_named_oracle_scope(
    *,
    scope: str,
    args: object,
    model_name: str,
    vertices: np.ndarray,
    faces: list[list[int]],
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    windows: list[int],
    z_levels: np.ndarray,
    line_points_by_window: np.ndarray,
    fit_rms_by_window: np.ndarray,
    cond_by_window: np.ndarray,
    point_bounds_min: np.ndarray,
    point_bounds_max: np.ndarray,
    w2_segments: list[dict[str, object]],
    w2_clusters: list[dict[str, object]],
    core_candidates: list[dict[str, object]],
    preselection_reservoir_audit: Callable[..., dict[str, object]],
) -> dict[str, object] | None:
    if scope == "edge-incidence-raw-support":
        model_face_rows = model_face_planes(vertices, faces)
        return {
            "diagnostic_scope": {
                "name": "edge-incidence-raw-support",
                "production_read_only": True,
                "uses_initial_model": "posthoc edge labels/ceiling evaluation only",
                "recomputed_blocks": ["edge_incidence_raw_support"],
                "full_edge_clip_trials": "not run unless the diagnostic produces a production-ready non-oracle shortlist",
            },
            "edge_incidence_raw_support": diagnose_edge_incidence_raw_support(
                model_name=model_name,
                vertices=vertices,
                faces=faces,
                windows=windows,
                z_levels=z_levels,
                line_points_by_window=line_points_by_window,
                fit_rms_by_window=fit_rms_by_window,
                cond_by_window=cond_by_window,
                point_bounds_min=point_bounds_min,
                point_bounds_max=point_bounds_max,
                peak_threshold=float(args.peak_threshold),
                low_threshold=float(args.low_threshold),
                final_candidates=final_candidates,
                final_reconstructed=final_reconstructed,
                w2_segments=w2_segments,
                w2_clusters=w2_clusters,
                model_faces=model_face_rows,
            ),
        }

    if scope == "view-conditioned-raw-support":
        model_face_rows = model_face_planes(vertices, faces)
        return {
            "diagnostic_scope": {
                "name": "view-conditioned-raw-support",
                "production_read_only": True,
                "uses_initial_model": "posthoc labels/evaluation only",
                "recomputed_blocks": ["view_conditioned_raw_support"],
                "full_edge_clip_trials": "not run unless the diagnostic produces a production-ready non-oracle shortlist",
            },
            "view_conditioned_raw_support": diagnose_view_conditioned_raw_support(
                model_name=model_name,
                vertices=vertices,
                faces=faces,
                windows=windows,
                z_levels=z_levels,
                line_points_by_window=line_points_by_window,
                fit_rms_by_window=fit_rms_by_window,
                cond_by_window=cond_by_window,
                point_bounds_min=point_bounds_min,
                point_bounds_max=point_bounds_max,
                peak_threshold=float(args.peak_threshold),
                low_threshold=float(args.low_threshold),
                final_candidates=final_candidates,
                final_reconstructed=final_reconstructed,
                model_faces=model_face_rows,
            ),
        }

    if scope == "preselection-safe-reservoir":
        previous_json = json.loads(Path(args.output_json).read_text())
        cached_oracle = ((previous_json.get("parameters") or {}).get("oracle_diagnostics") or {})
        cached_loss_funnel = cached_oracle.get("production_129_loss_funnel")
        if not isinstance(cached_loss_funnel, dict):
            raise RuntimeError(
                "POLYRECO_ORACLE_DIAGNOSTIC_SCOPE=preselection-safe-reservoir requires an existing "
                "production_129_loss_funnel in --output-json"
            )
        model_face_rows = model_face_planes(vertices, faces)
        core_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
            core_candidates,
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
            int(index)
            for index in core_reconstructed.get("face_candidate_indices", [])
            if 0 <= int(index) < len(core_candidates)
        ]
        core_only_active_candidates = [core_candidates[index] for index in core_only_active_indices]
        oracle_match_cache: dict[int, dict[str, object] | None] = {}
        core_cohort = oracle_candidate_cohort(
            core_only_active_candidates,
            model_face_rows,
            active_plane_indices=core_only_active_indices,
            match_cache=oracle_match_cache,
        )
        core_ids = set(int(value) for value in core_cohort.get("finite_face_ids", []))
        reconstruction_kwargs = {
            "halfspace_slack": float(args.intersection_halfspace_slack),
            "feasibility_tol": float(args.intersection_feasibility_tol),
            "incidence_tol": float(args.intersection_incidence_tol),
            "vertex_merge_tol": float(args.intersection_vertex_merge_tol),
            "min_face_area": float(args.intersection_min_face_area),
            "triple_det_tol": float(args.intersection_triple_det_tol),
            "prune_redundant": bool(args.intersection_prune_redundant_planes),
            "redundancy_tol": float(args.intersection_redundancy_tol),
        }
        cached_loss_funnel["preselection_safe_reservoir_audit"] = preselection_reservoir_audit(
            cached_loss_funnel=cached_loss_funnel,
            model_face_rows=model_face_rows,
            core_reconstructed=core_reconstructed,
            core_ids=core_ids,
            oracle_reconstruction_kwargs=reconstruction_kwargs,
            oracle_match_cache=oracle_match_cache,
        )
        cached_oracle["production_129_loss_funnel"] = cached_loss_funnel
        cached_oracle["diagnostic_scope"] = {
            "name": "preselection-safe-reservoir",
            "reused_existing_oracle_diagnostics": True,
            "recomputed_blocks": [
                "production_129_loss_funnel.preselection_safe_reservoir_audit",
            ],
            "w2_short_edge_diagnostics_rerun": False,
        }
        return cached_oracle

    return None


def diagnose_production_129_loss_funnel(
    *,
    initial_vertices: np.ndarray,
    initial_faces: list[list[int]],
    model_faces: list[dict[str, object]],
    minima_cloud: dict[str, object],
    stable_minima_points: np.ndarray,
    stable_minima_z_indices: np.ndarray,
    valley_raw_candidates: list[dict[str, object]],
    core_candidates: list[dict[str, object]],
    w2_line_points: np.ndarray | None,
    w2_segments: list[dict[str, object]],
    w2_clusters: list[dict[str, object]],
    w2_raw_candidates: list[dict[str, object]],
    w2_preselection_candidates: list[dict[str, object]],
    support_diverse_ranked_candidate_ids: list[int],
    selection_attempt_log: list[dict[str, object]],
    max150_candidates: list[dict[str, object]],
    append_candidates: list[dict[str, object]],
    repair_candidates: list[dict[str, object]],
    ratchet_candidates: list[dict[str, object]],
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    exchange_target_candidate_ids: list[int],
    core_face_ids: set[int],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    match_cache: dict[int, dict[str, object] | None],
) -> dict[str, object]:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    exchange_target_ids = {int(v) for v in exchange_target_candidate_ids}

    def unique_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        seen: set[int] = set()
        for candidate in candidates:
            cid = int(candidate_track_id(candidate))
            if cid in seen:
                continue
            seen.add(cid)
            out.append(candidate)
        return out

    def oracle_match(candidate: dict[str, object]) -> dict[str, object] | None:
        cid = int(candidate_track_id(candidate))
        if cid not in match_cache:
            match_cache[cid] = candidate_oracle_face(candidate, model_faces)
        return match_cache.get(cid)

    def candidate_source(candidate: dict[str, object], entry_source_by_id: dict[int, str] | None = None) -> str:
        cid = int(candidate_track_id(candidate))
        if entry_source_by_id is not None and cid in entry_source_by_id:
            return str(entry_source_by_id[cid])
        if cid in exchange_target_ids or candidate.get("post_ratchet_exchange_action") is not None:
            return "lp-deficit exchange targets"
        mode = str(candidate.get("small_face_refinement_mode") or "")
        if mode == "append-local":
            return "append-local"
        if mode.startswith("post-repair"):
            return "broad-ratchet"
        if str(candidate.get("candidate_origin") or "") == "w2_addition":
            return "w2 edge adjacency"
        return "baseline/valley"

    def candidate_row(candidate: dict[str, object], match: dict[str, object]) -> dict[str, object]:
        return {
            "candidate_id": int(candidate_track_id(candidate)),
            "source": candidate_source(candidate),
            "normal_angle_deg": match.get("normal_angle_deg"),
            "plane_distance": match.get("plane_distance"),
            "centroid_distance": match.get("centroid_distance"),
            "hull_surface_distance": match.get("hull_surface_distance"),
            "plane_good": bool(match.get("plane_good")),
            "finite_good": bool(match.get("finite_good")),
            "plane_rms": candidate.get("plane_rms"),
            "levels": candidate.get("levels"),
            "hull_area": candidate.get("hull_area"),
            "finite_support_count": candidate.get("finite_support_count"),
        }

    def face_candidate_index(
        candidates: list[dict[str, object]],
        *,
        finite_only: bool,
    ) -> tuple[dict[int, list[dict[str, object]]], dict[int, dict[str, object]]]:
        by_face: dict[int, list[dict[str, object]]] = {}
        best: dict[int, tuple[tuple[float, float, float, float, int], dict[str, object]]] = {}
        for candidate in unique_candidates(candidates):
            match = oracle_match(candidate)
            if match is None or match.get("face_id") is None:
                continue
            if finite_only and not bool(match.get("finite_good")):
                continue
            face_id = int(match["face_id"])
            by_face.setdefault(face_id, []).append(candidate)
            key = oracle_geometry_key(match)
            if face_id not in best or key < best[face_id][0]:
                best[face_id] = (key, candidate_row(candidate, match))
        return by_face, {face_id: row for face_id, (_, row) in best.items()}

    def point_support_by_face(
        points: np.ndarray,
        z_indices: np.ndarray,
        *,
        tolerance: float,
    ) -> tuple[dict[int, int], dict[int, int]]:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
            return {}, {}
        valid = np.all(np.isfinite(pts), axis=1)
        pts = pts[valid]
        z = np.asarray(z_indices, dtype=int)
        if z.size == valid.size:
            z = z[valid]
        else:
            z = np.arange(pts.shape[0], dtype=int)
        counts: dict[int, int] = {}
        z_counts: dict[int, int] = {}
        for face in model_faces:
            face_id = int(face["face_id"])
            normal = np.array(face["normal"], dtype=float)
            face_points = np.array(face["vertices"], dtype=float)
            cache = face.get("_polygon_distance_cache")
            if isinstance(cache, dict):
                origin = np.array(cache["origin"], dtype=float)
                u = np.array(cache["u"], dtype=float)
                v = np.array(cache["v"], dtype=float)
                poly_2d = np.array(cache["poly_2d"], dtype=float)
            else:
                u, v = plane_basis(normal)
                origin = np.mean(face_points, axis=0)
                poly_2d = np.column_stack([(face_points - origin) @ u, (face_points - origin) @ v])
                face["_polygon_distance_cache"] = {"origin": origin, "u": u, "v": v, "poly_2d": poly_2d}
            point_2d = np.column_stack([(pts - origin) @ u, (pts - origin) @ v])
            outside = np.maximum(polygon_signed_distances_2d(point_2d, poly_2d), 0.0)
            plane_distance = np.abs((pts - face_points[0]) @ normal)
            near = np.sqrt(plane_distance * plane_distance + outside * outside) <= float(tolerance)
            counts[face_id] = int(np.sum(near))
            z_counts[face_id] = int(len(set(int(value) for value in z[near])))
        return counts, z_counts

    evidence_started = time.perf_counter()
    minima_points = np.array(minima_cloud.get("points") or [], dtype=float).reshape((-1, 3))
    minima_z = np.array(minima_cloud.get("z_indices") or [], dtype=int)
    raw_minima_counts, raw_minima_z_counts = point_support_by_face(minima_points, minima_z, tolerance=0.15)
    stable_minima_counts, stable_minima_z_counts = point_support_by_face(
        stable_minima_points,
        stable_minima_z_indices,
        tolerance=0.15,
    )
    timings["minima_support_seconds"] = time.perf_counter() - evidence_started

    edges, edge_faces = model_edges_from_faces(initial_vertices, initial_faces)
    edge_id_to_faces = {int(edge_id): set(int(v) for v in edge_faces[edge]) for edge_id, edge in enumerate(edges)}
    raw_edge_point_counts = np.zeros(len(edges), dtype=int)
    raw_edge_z_levels: list[set[int]] = [set() for _ in edges]
    raw_edge_cyclic_z: list[dict[int, set[int]]] = [dict() for _ in edges]
    edge_started = time.perf_counter()
    if w2_line_points is not None:
        array = np.asarray(w2_line_points, dtype=float)
        if array.ndim == 3 and array.shape[2] == 3:
            flat = array.reshape((-1, 3))
            flat_z = np.tile(np.arange(array.shape[1], dtype=int), array.shape[0])
            flat_cyclic = np.repeat(np.arange(array.shape[0], dtype=int), array.shape[1])
            valid = np.all(np.isfinite(flat), axis=1)
            flat = flat[valid]
            flat_z = flat_z[valid]
            flat_cyclic = flat_cyclic[valid]
            for start in range(0, flat.shape[0], 512):
                chunk = flat[start : start + 512]
                distances, _ = point_edge_distances_vectorized(chunk, initial_vertices, edges)
                nearest = np.argmin(distances, axis=1)
                nearest_distance = distances[np.arange(distances.shape[0]), nearest]
                accepted = nearest_distance <= 0.08
                for edge_id, z_index, cyclic_index in zip(
                    nearest[accepted],
                    flat_z[start : start + 512][accepted],
                    flat_cyclic[start : start + 512][accepted],
                ):
                    raw_edge_point_counts[int(edge_id)] += 1
                    raw_edge_z_levels[int(edge_id)].add(int(z_index))
                    raw_edge_cyclic_z[int(edge_id)].setdefault(int(cyclic_index), set()).add(int(z_index))
    raw_supported_edge_ids = {
        int(edge_id)
        for edge_id, count in enumerate(raw_edge_point_counts)
        if int(count) >= 3 and len(raw_edge_z_levels[edge_id]) >= 3
    }

    segment_edge_rows: list[dict[str, object]] = []
    segment_oracle_rows_internal: list[dict[str, object]] = []
    robust_segments_by_edge: dict[int, set[int]] = {}
    if w2_line_points is not None:
        array = np.asarray(w2_line_points, dtype=float)
        for segment in w2_segments:
            cyclic_index_value = segment.get("cyclic_index")
            cyclic_index = int(cyclic_index_value) if cyclic_index_value is not None else -1
            z_values = [int(v) for v in segment.get("z_indices", [])]
            if cyclic_index < 0 or cyclic_index >= array.shape[0] or not z_values:
                continue
            valid_z = [value for value in z_values if 0 <= value < array.shape[1]]
            points = array[cyclic_index, np.array(valid_z, dtype=int), :]
            points = points[np.all(np.isfinite(points), axis=1)]
            if points.shape[0] > 32:
                sample = np.linspace(0, points.shape[0] - 1, 32, dtype=int)
                points = points[sample]
            match = oracle_run_edge_match(
                points=points,
                fit=segment,
                vertices=initial_vertices,
                edges=edges,
                edge_id_to_faces=edge_id_to_faces,
                core_face_ids=core_face_ids,
            )
            edge_id = match.get("dominant_edge_id")
            segment_id_value = segment.get("segment_id")
            segment_id = int(segment_id_value) if segment_id_value is not None else -1
            if edge_id is not None and bool(match.get("oracle_confident_finite_edge")):
                robust_segments_by_edge.setdefault(int(edge_id), set()).add(int(segment_id))
            segment_oracle_rows_internal.append(
                {
                    "segment_id": int(segment_id),
                    "cyclic_index": int(cyclic_index),
                    "segment_fit_mode": str(segment.get("segment_fit_mode") or "least_squares"),
                    "levels": int(segment.get("levels") or len(valid_z)),
                    "z_min": segment.get("z_min"),
                    "z_max": segment.get("z_max"),
                    "z_span": segment.get("z_span"),
                    **match,
                }
            )
            segment_edge_rows.append(
                {
                    "segment_id": int(segment_id),
                    "edge_id": int(edge_id) if edge_id is not None else None,
                    "confident": bool(match.get("oracle_confident_finite_edge")),
                    "median_distance": match.get("dominant_edge_distance_median"),
                    "angle_deg": match.get("angle_to_dominant_edge_deg"),
                }
            )
    cluster_oracle = oracle_edge_cluster_matches(w2_clusters, initial_vertices, initial_faces)
    cluster_lookup = edge_match_lookup(cluster_oracle)
    clusters_by_edge: dict[int, set[int]] = {}
    for cluster_id, row in cluster_lookup.items():
        edge_id = row.get("top_edge_id")
        if edge_id is not None and bool(row.get("top_confidence")):
            clusters_by_edge.setdefault(int(edge_id), set()).add(int(cluster_id))
    timings["w2_edge_ancestry_seconds"] = time.perf_counter() - edge_started

    face_raw_edges: dict[int, set[int]] = {int(face["face_id"]): set() for face in model_faces}
    face_robust_segments: dict[int, set[int]] = {int(face["face_id"]): set() for face in model_faces}
    face_robust_edges: dict[int, set[int]] = {int(face["face_id"]): set() for face in model_faces}
    face_clusters: dict[int, set[int]] = {int(face["face_id"]): set() for face in model_faces}
    face_cluster_edges: dict[int, set[int]] = {int(face["face_id"]): set() for face in model_faces}
    for edge_id in raw_supported_edge_ids:
        for face_id in edge_id_to_faces.get(int(edge_id), set()):
            face_raw_edges[int(face_id)].add(int(edge_id))
    for edge_id, segment_ids in robust_segments_by_edge.items():
        for face_id in edge_id_to_faces.get(int(edge_id), set()):
            face_robust_edges[int(face_id)].add(int(edge_id))
            face_robust_segments[int(face_id)].update(int(v) for v in segment_ids)
    for edge_id, cluster_ids in clusters_by_edge.items():
        for face_id in edge_id_to_faces.get(int(edge_id), set()):
            face_cluster_edges[int(face_id)].add(int(edge_id))
            face_clusters[int(face_id)].update(int(v) for v in cluster_ids)

    face_hypothesis_ids: dict[int, set[int]] = {int(face["face_id"]): set() for face in model_faces}
    for candidate in w2_raw_candidates:
        pair = [int(v) for v in candidate.get("w2_edge_cluster_ids", [])]
        if len(pair) != 2:
            continue
        rows = [cluster_lookup.get(pair[0]), cluster_lookup.get(pair[1])]
        if any(row is None or not bool(row.get("top_confidence")) for row in rows):
            continue
        edge_ids = [int(row["top_edge_id"]) for row in rows if row is not None and row.get("top_edge_id") is not None]
        if len(edge_ids) != 2 or edge_ids[0] == edge_ids[1]:
            continue
        shared_faces = edge_id_to_faces.get(edge_ids[0], set()) & edge_id_to_faces.get(edge_ids[1], set())
        for face_id in shared_faces:
            face_hypothesis_ids[int(face_id)].add(int(candidate_track_id(candidate)))

    matching_started = time.perf_counter()
    raw_candidates = unique_candidates(list(valley_raw_candidates) + list(w2_raw_candidates))
    preselection_candidates = unique_candidates(list(core_candidates) + list(w2_preselection_candidates))
    ranked_lookup = {int(candidate_track_id(candidate)): candidate for candidate in w2_preselection_candidates}
    ranked_candidates = unique_candidates(
        list(core_candidates)
        + [ranked_lookup[cid] for cid in support_diverse_ranked_candidate_ids if cid in ranked_lookup]
    )
    valley_any_by_face, valley_best = face_candidate_index(valley_raw_candidates, finite_only=False)
    raw_any_by_face, raw_any_best = face_candidate_index(raw_candidates, finite_only=False)
    raw_finite_by_face, raw_finite_best = face_candidate_index(raw_candidates, finite_only=True)
    pre_finite_by_face, pre_finite_best = face_candidate_index(preselection_candidates, finite_only=True)
    ranked_finite_by_face, ranked_finite_best = face_candidate_index(ranked_candidates, finite_only=True)
    accepted_finite_by_face, accepted_finite_best = face_candidate_index(max150_candidates, finite_only=True)
    timings["candidate_oracle_matching_seconds"] = time.perf_counter() - matching_started

    stage_started = time.perf_counter()
    stage_specs = [
        ("max150", max150_candidates, None),
        ("append_local", append_candidates, None),
        ("lp_blockers", repair_candidates, None),
        ("broad_ratchet", ratchet_candidates, None),
        ("lp_deficit", final_candidates, final_reconstructed),
    ]
    stage_data: dict[str, dict[str, object]] = {}
    for name, stage_candidates, known_reconstructed in stage_specs:
        rec = known_reconstructed or reconstruct_polyhedron_from_halfspaces_edge_clip(
            stage_candidates,
            **reconstruction_kwargs,
        )
        active_indices = [
            int(index)
            for index in rec.get("face_candidate_indices", [])
            if 0 <= int(index) < len(stage_candidates)
        ]
        active_candidates = [stage_candidates[index] for index in active_indices]
        active_by_face, active_best = face_candidate_index(active_candidates, finite_only=True)
        stage_data[name] = {
            "candidates": stage_candidates,
            "reconstructed": rec,
            "active_indices": active_indices,
            "active_candidates": active_candidates,
            "active_by_face": active_by_face,
            "active_best": active_best,
            "geometry_signature": mesh_geometry_signature(stage_candidates, rec),
        }
    timings["stage_edge_clip_seconds"] = time.perf_counter() - stage_started

    final_ids = set(int(v) for v in stage_data["lp_deficit"]["active_by_face"])
    active_any_stage_ids = set().union(
        *(set(int(v) for v in stage_data[name]["active_by_face"]) for name, _, _ in stage_specs)
    )
    attempt_reason_by_id = {
        int(row["track_id"]): str(row.get("reason") or "unknown")
        for row in selection_attempt_log
        if row.get("track_id") is not None
    }

    zone_definitions = {
        "near_horizontal": "abs(oriented_normal_z) >= 0.90; evaluated first",
        "belt_z_minus_5": "face centroid z in [-5.25, -4.50)",
        "low_z": "face centroid z < -5.25",
        "middle_z": "face centroid z in [-4.50, -2.00)",
        "other": "remaining faces",
    }

    def face_zone(face: dict[str, object]) -> tuple[str, float]:
        centroid_z = float(np.mean(np.array(face["vertices"], dtype=float)[:, 2]))
        normal_z = abs(float(np.array(face["normal"], dtype=float)[2]))
        if normal_z >= 0.90:
            return "near_horizontal", centroid_z
        if -5.25 <= centroid_z < -4.50:
            return "belt_z_minus_5", centroid_z
        if centroid_z < -5.25:
            return "low_z", centroid_z
        if -4.50 <= centroid_z < -2.00:
            return "middle_z", centroid_z
        return "other", centroid_z

    def first_loss(row: dict[str, object]) -> str:
        if bool(row["final_canonical_covered"]):
            return "covered_final"
        if bool(row["raw_finite_good_candidate_exists"]):
            if not bool(row["after_core_preselection"]):
                return "lost_at_preselection"
            if not bool(row["accepted_max150"]):
                return "ranked_out"
            if int(row["active_stage_count"]) == 0:
                return "accepted_but_never_active"
            if bool(row["active_before_final"]):
                return "became_redundant_later"
            return "canonical_mapping_conflict"
        if bool(row["raw_candidate_exists"]):
            return "raw_candidate_not_finite_good"
        valley_exists = bool(row["valley_candidate_exists"])
        if int(row["raw_line_minima_support_count"]) == 0 and int(row["w2_raw_boundary_edge_count"]) == 0:
            return "no_raw_observed_support"
        if int(row["stable_multiscale_minima_z_levels"]) < 2 and int(row["w2_robust_segment_count"]) == 0:
            return "insufficient_stable_minima"
        if not valley_exists and int(row["w2_robust_segment_count"]) == 0:
            return "valley_pair_or_track_missing"
        if not valley_exists and int(row["w2_raw_boundary_edge_count"]) < 2:
            return "w2_boundary_edges_missing"
        if not valley_exists and (
            int(row["w2_robust_boundary_edge_count"]) < 2
            or int(row["w2_matched_cluster_edge_count"]) < 2
        ):
            return "w2_segmentation_or_cluster_missing"
        if not valley_exists and int(row["valid_adjacency_face_hypothesis_count"]) == 0:
            return "face_hypothesis_missing"
        return "plane_fit_rejected"

    face_rows: list[dict[str, object]] = []
    for face in sorted(model_faces, key=lambda item: int(item["face_id"])):
        face_id = int(face["face_id"])
        zone, centroid_z = face_zone(face)
        valley_candidates = valley_any_by_face.get(face_id, [])
        active_flags = {
            name: bool(face_id in stage_data[name]["active_by_face"])
            for name, _, _ in stage_specs
        }
        raw_best = raw_finite_best.get(face_id) or raw_any_best.get(face_id)
        best_candidates = {
            "raw": raw_best,
            "preselection": pre_finite_best.get(face_id),
            "support_diverse_ranking": ranked_finite_best.get(face_id),
            "accepted_max150": accepted_finite_best.get(face_id),
            "active_max150": stage_data["max150"]["active_best"].get(face_id),
            "active_append_local": stage_data["append_local"]["active_best"].get(face_id),
            "active_lp_blockers": stage_data["lp_blockers"]["active_best"].get(face_id),
            "active_broad_ratchet": stage_data["broad_ratchet"]["active_best"].get(face_id),
            "active_lp_deficit": stage_data["lp_deficit"]["active_best"].get(face_id),
        }
        accepted_ids = {
            int(candidate_track_id(candidate)) for candidate in accepted_finite_by_face.get(face_id, [])
        }
        rejection_reason = None
        if pre_finite_by_face.get(face_id) and not accepted_ids:
            pre_ids = [int(candidate_track_id(candidate)) for candidate in pre_finite_by_face[face_id]]
            rejection_reason = next(
                (attempt_reason_by_id[cid] for cid in pre_ids if cid in attempt_reason_by_id),
                "not_reached_before_max150_budget",
            )
        row: dict[str, object] = {
            "face_id": int(face_id),
            "zone": zone,
            "centroid_z": as_json_float(float(centroid_z)),
            "z_min": as_json_float(float(face["z_min"])),
            "z_max": as_json_float(float(face["z_max"])),
            "normal": as_json_point(np.array(face["normal"], dtype=float)),
            "raw_line_minima_support": bool(raw_minima_counts.get(face_id, 0) > 0),
            "raw_line_minima_support_count": int(raw_minima_counts.get(face_id, 0)),
            "raw_line_minima_z_levels": int(raw_minima_z_counts.get(face_id, 0)),
            "stable_multiscale_minima_support": bool(stable_minima_z_counts.get(face_id, 0) >= 2),
            "stable_multiscale_minima_support_count": int(stable_minima_counts.get(face_id, 0)),
            "stable_multiscale_minima_z_levels": int(stable_minima_z_counts.get(face_id, 0)),
            "valley_observations": int(sum(int(candidate.get("levels") or 0) for candidate in valley_candidates)),
            "valley_candidate_exists": bool(valley_candidates),
            "w2_raw_boundary_edge_support": bool(face_raw_edges.get(face_id)),
            "w2_raw_boundary_edge_count": int(len(face_raw_edges.get(face_id, set()))),
            "w2_robust_segment_support": bool(face_robust_segments.get(face_id)),
            "w2_robust_segment_count": int(len(face_robust_segments.get(face_id, set()))),
            "w2_robust_boundary_edge_count": int(len(face_robust_edges.get(face_id, set()))),
            "w2_matched_edge_clusters": int(len(face_clusters.get(face_id, set()))),
            "w2_matched_cluster_edge_count": int(len(face_cluster_edges.get(face_id, set()))),
            "valid_adjacency_face_hypothesis": bool(face_hypothesis_ids.get(face_id)),
            "valid_adjacency_face_hypothesis_count": int(len(face_hypothesis_ids.get(face_id, set()))),
            "raw_candidate_exists": bool(raw_any_by_face.get(face_id)),
            "raw_candidate_count": int(len(raw_any_by_face.get(face_id, []))),
            "raw_finite_good_candidate_exists": bool(raw_finite_by_face.get(face_id)),
            "raw_finite_good_candidate_count": int(len(raw_finite_by_face.get(face_id, []))),
            "after_core_preselection": bool(pre_finite_by_face.get(face_id)),
            "after_core_preselection_count": int(len(pre_finite_by_face.get(face_id, []))),
            "after_support_diverse_ranking": bool(ranked_finite_by_face.get(face_id)),
            "after_support_diverse_ranking_count": int(len(ranked_finite_by_face.get(face_id, []))),
            "accepted_max150": bool(accepted_finite_by_face.get(face_id)),
            "accepted_max150_count": int(len(accepted_finite_by_face.get(face_id, []))),
            "active_after_max150": active_flags["max150"],
            "active_after_append_local": active_flags["append_local"],
            "active_after_lp_blockers": active_flags["lp_blockers"],
            "active_after_broad_ratchet": active_flags["broad_ratchet"],
            "active_after_lp_deficit": active_flags["lp_deficit"],
            "final_canonical_covered": bool(face_id in final_ids),
            "active_stage_count": int(sum(active_flags.values())),
            "active_before_final": bool(any(active_flags[name] for name in ("max150", "append_local", "lp_blockers", "broad_ratchet"))),
            "best_candidates": best_candidates,
            "best_candidate_id": raw_best.get("candidate_id") if isinstance(raw_best, dict) else None,
            "candidate_source": raw_best.get("source") if isinstance(raw_best, dict) else None,
            "best_canonical_distances": {
                key: raw_best.get(key) if isinstance(raw_best, dict) else None
                for key in ("normal_angle_deg", "plane_distance", "centroid_distance", "hull_surface_distance")
            },
            "selection_rejection_reason": rejection_reason,
        }
        row["first_loss_category"] = first_loss(row)
        row["first_rejection_reason"] = rejection_reason or row["first_loss_category"]
        face_rows.append(row)

    category_order = [
        "covered_final",
        "no_raw_observed_support",
        "insufficient_stable_minima",
        "valley_pair_or_track_missing",
        "w2_boundary_edges_missing",
        "w2_segmentation_or_cluster_missing",
        "face_hypothesis_missing",
        "plane_fit_rejected",
        "raw_candidate_not_finite_good",
        "lost_at_preselection",
        "ranked_out",
        "accepted_but_never_active",
        "became_redundant_later",
        "canonical_mapping_conflict",
    ]
    category_counts = {
        category: int(sum(1 for row in face_rows if row["first_loss_category"] == category))
        for category in category_order
    }

    zone_rows: dict[str, object] = {}
    for zone in zone_definitions:
        rows = [row for row in face_rows if row["zone"] == zone]
        zone_rows[zone] = {
            "total_faces": int(len(rows)),
            "covered": int(sum(1 for row in rows if bool(row["final_canonical_covered"]))),
            "first_loss_categories": {
                category: int(sum(1 for row in rows if row["first_loss_category"] == category))
                for category in category_order
            },
            "raw_candidate_ceiling": int(sum(1 for row in rows if bool(row["raw_finite_good_candidate_exists"]))),
            "preselection_ceiling": int(sum(1 for row in rows if bool(row["after_core_preselection"]))),
            "final_coverage": int(sum(1 for row in rows if bool(row["final_canonical_covered"]))),
        }

    max150_ids = {int(candidate_track_id(candidate)) for candidate in max150_candidates}
    append_ids = {int(candidate_track_id(candidate)) for candidate in append_candidates}
    repair_ids = {int(candidate_track_id(candidate)) for candidate in repair_candidates}
    ratchet_ids = {int(candidate_track_id(candidate)) for candidate in ratchet_candidates}
    final_candidate_ids = {int(candidate_track_id(candidate)) for candidate in final_candidates}
    core_candidate_ids = {int(candidate_track_id(candidate)) for candidate in core_candidates}
    all_known_by_id = {
        int(candidate_track_id(candidate)): candidate
        for candidate in unique_candidates(
            list(core_candidates)
            + list(w2_preselection_candidates)
            + list(max150_candidates)
            + list(append_candidates)
            + list(ratchet_candidates)
            + list(final_candidates)
        )
    }
    entry_pools = {
        "baseline/valley": unique_candidates(list(core_candidates)),
        "w2 edge adjacency": [
            candidate
            for candidate in max150_candidates
            if int(candidate_track_id(candidate)) not in core_candidate_ids
            and int(candidate_track_id(candidate)) not in exchange_target_ids
        ],
        "append-local": [
            candidate
            for candidate in append_candidates
            if int(candidate_track_id(candidate)) not in max150_ids
            and int(candidate_track_id(candidate)) not in exchange_target_ids
        ],
        "broad-ratchet": [
            candidate
            for candidate in ratchet_candidates
            if int(candidate_track_id(candidate)) not in repair_ids
            and int(candidate_track_id(candidate)) not in exchange_target_ids
        ],
        "lp-deficit exchange targets": [
            all_known_by_id[cid]
            for cid in sorted(exchange_target_ids)
            if cid in all_known_by_id
        ],
    }
    entry_source_by_id: dict[int, str] = {}
    for source, pool in entry_pools.items():
        for candidate in pool:
            entry_source_by_id[int(candidate_track_id(candidate))] = source

    def exclusive_active_category(candidate: dict[str, object], unique_face_representatives: set[tuple[int, int]]) -> str:
        match = oracle_match(candidate)
        cid = int(candidate_track_id(candidate))
        if match is None:
            return "unsupported_or_ambiguous"
        face_id = int(match["face_id"]) if match.get("face_id") is not None else -1
        if bool(match.get("finite_good")):
            return "canonical_finite_good_unique" if (face_id, cid) in unique_face_representatives else "canonical_finite_good_duplicates"
        if bool(match.get("plane_good")):
            return "plane_good_patch_bad"
        reasons = set(str(v) for v in match.get("failure_reasons", []))
        if "normal_angle" in reasons:
            return "normal_failure"
        if "plane_distance" in reasons:
            return "plane_distance_failure"
        return "unsupported_or_ambiguous"

    final_active_candidates = list(stage_data["lp_deficit"]["active_candidates"])
    finite_active_by_face: dict[int, list[dict[str, object]]] = {}
    for candidate in final_active_candidates:
        match = oracle_match(candidate)
        if match is not None and bool(match.get("finite_good")):
            finite_active_by_face.setdefault(int(match["face_id"]), []).append(candidate)
    unique_face_representatives: set[tuple[int, int]] = set()
    for face_id, pool in finite_active_by_face.items():
        representative = min(pool, key=lambda candidate: oracle_geometry_key(oracle_match(candidate) or {}))
        unique_face_representatives.add((int(face_id), int(candidate_track_id(representative))))
    anatomy_categories = [
        "canonical_finite_good_unique",
        "canonical_finite_good_duplicates",
        "plane_good_patch_bad",
        "normal_failure",
        "plane_distance_failure",
        "unsupported_or_ambiguous",
    ]
    anatomy_by_candidate_id = {
        int(candidate_track_id(candidate)): exclusive_active_category(candidate, unique_face_representatives)
        for candidate in final_active_candidates
    }
    anatomy = {
        category: int(sum(1 for value in anatomy_by_candidate_id.values() if value == category))
        for category in anatomy_categories
    }
    anatomy["total_active_faces"] = int(len(final_active_candidates))
    anatomy["exclusive_sum"] = int(sum(int(anatomy[category]) for category in anatomy_categories))
    anatomy["overlapping_raw_failure_flags"] = {
        "normal_angle": int(sum(1 for candidate in final_active_candidates if "normal_angle" in set((oracle_match(candidate) or {}).get("failure_reasons", [])))),
        "plane_distance": int(sum(1 for candidate in final_active_candidates if "plane_distance" in set((oracle_match(candidate) or {}).get("failure_reasons", [])))),
    }

    source_id_sets: dict[str, set[int]] = {}
    source_contributions: dict[str, object] = {}
    for source, pool in entry_pools.items():
        pool = unique_candidates(pool)
        finite_matches = [
            (candidate, oracle_match(candidate))
            for candidate in pool
            if oracle_match(candidate) is not None and bool((oracle_match(candidate) or {}).get("finite_good"))
        ]
        finite_ids = {int(match["face_id"]) for _, match in finite_matches if match is not None}
        source_id_sets[source] = finite_ids
        active_pool = [
            candidate
            for candidate in final_active_candidates
            if candidate_source(candidate, entry_source_by_id) == source
        ]
        active_finite = [candidate for candidate in active_pool if bool((oracle_match(candidate) or {}).get("finite_good"))]
        active_categories = {
            category: int(sum(1 for candidate in active_pool if anatomy_by_candidate_id[int(candidate_track_id(candidate))] == category))
            for category in anatomy_categories
        }
        source_contributions[source] = {
            "candidate_count": int(len(pool)),
            "finite_good_count": int(len(finite_matches)),
            "unique_finite_ids": int(len(finite_ids)),
            "finite_face_ids": sorted(int(v) for v in finite_ids),
            "active_candidate_count": int(len(active_pool)),
            "active_finite_good": int(len(active_finite)),
            "duplicate_assignments": int(len(finite_matches) - len(finite_ids)),
            "false_active_categories": active_categories,
        }
    for source in source_contributions:
        others = set().union(*(ids for name, ids in source_id_sets.items() if name != source))
        unique_only = source_id_sets[source] - others
        source_contributions[source]["ids_unique_only_to_source"] = sorted(int(v) for v in unique_only)
        source_contributions[source]["unique_only_to_source_count"] = int(len(unique_only))
    all_entry_candidates = unique_candidates([candidate for pool in entry_pools.values() for candidate in pool])
    all_entry_finite = [candidate for candidate in all_entry_candidates if bool((oracle_match(candidate) or {}).get("finite_good"))]
    all_entry_ids = {int((oracle_match(candidate) or {})["face_id"]) for candidate in all_entry_finite}
    source_contributions["union_all_sources"] = {
        "candidate_count": int(len(all_entry_candidates)),
        "finite_good_count": int(len(all_entry_finite)),
        "unique_finite_ids": int(len(all_entry_ids)),
        "finite_face_ids": sorted(int(v) for v in all_entry_ids),
        "active_candidate_count": int(len(final_active_candidates)),
        "active_finite_good": int(sum(1 for candidate in final_active_candidates if bool((oracle_match(candidate) or {}).get("finite_good")))),
        "duplicate_assignments": int(len(all_entry_finite) - len(all_entry_ids)),
        "false_active_categories": anatomy,
    }

    grouping_started = time.perf_counter()
    raw_finite_candidates = [candidate for candidate in raw_candidates if bool((oracle_match(candidate) or {}).get("finite_good"))]
    raw_finite_candidates.sort(key=nonoracle_candidate_quality_key)
    patch_groups: list[list[dict[str, object]]] = []
    for candidate in raw_finite_candidates:
        for group in patch_groups:
            if all(candidates_plane_patch_compatible(candidate, member) for member in group):
                group.append(candidate)
                break
        else:
            patch_groups.append([candidate])
    patch_representatives = [min(group, key=nonoracle_candidate_quality_key) for group in patch_groups]
    patch_ceiling_ids = {
        int((oracle_match(candidate) or {})["face_id"])
        for candidate in patch_representatives
        if bool((oracle_match(candidate) or {}).get("finite_good"))
    }
    timings["patch_grouping_seconds"] = time.perf_counter() - grouping_started

    individual_started = time.perf_counter()
    individual_rows: list[dict[str, object]] = []
    z_slices = prepare_z_level_slices(trusted_z_indices)
    final_selected_by_id = {int(candidate_track_id(candidate)): candidate for candidate in final_candidates}
    for face_id in sorted(set(raw_finite_by_face) - final_ids):
        representative = min(raw_finite_by_face[face_id], key=nonoracle_candidate_quality_key)
        representative_id = int(candidate_track_id(representative))
        if representative_id in final_selected_by_id:
            trial_candidates = list(final_candidates)
        else:
            trial_candidates = list(final_candidates) + [representative]
        trial_rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
        trial_active_indices = [
            int(index)
            for index in trial_rec.get("face_candidate_indices", [])
            if 0 <= int(index) < len(trial_candidates)
        ]
        trial_active_candidates = [trial_candidates[index] for index in trial_active_indices]
        trial_by_face, _ = face_candidate_index(trial_active_candidates, finite_only=True)
        trial_ids = set(int(v) for v in trial_by_face)
        outside = evaluate_trusted_cloud_outside(
            trial_candidates,
            trusted_points,
            z_slices,
            point_tol=float(point_tol),
        )
        lost_final = final_ids - trial_ids
        target_active = face_id in trial_ids
        safe_geometry = bool(
            topology_is_valid(trial_rec.get("topology") if isinstance(trial_rec.get("topology"), dict) else {})
            and finite_float(outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
            and int(value_or_default(outside.get("cumulative_lost_z_levels"), 1)) == 0
        )
        if not safe_geometry or bool(core_face_ids - trial_ids):
            classification = "unsafe"
        elif target_active and not lost_final and len(trial_ids) > len(final_ids):
            classification = "safe_plus_one"
        elif target_active:
            classification = "active_but_replaces_existing"
        else:
            classification = "redundant"
        individual_rows.append(
            {
                "face_id": int(face_id),
                "representative_candidate_id": int(representative_id),
                "representative_already_selected": bool(representative_id in final_selected_by_id),
                "classification": classification,
                "target_active": bool(target_active),
                "trial_unique_finite_ids": int(len(trial_ids)),
                "canonical_ids_gained": sorted(int(v) for v in (trial_ids - final_ids)),
                "canonical_ids_lost": sorted(int(v) for v in lost_final),
                "lost_core_ids": sorted(int(v) for v in (core_face_ids - trial_ids)),
                "vertices": int(len(trial_rec.get("vertices") or [])),
                "edges": int(len(trial_rec.get("edges") or [])),
                "faces": int(len(trial_rec.get("faces") or [])),
                "volume": trial_rec.get("reliable_volume"),
                "topology": trial_rec.get("topology"),
                "trusted_outside": outside,
            }
        )
    timings["individual_edge_clip_trials_seconds"] = time.perf_counter() - individual_started
    individual_counts = {
        classification: int(sum(1 for row in individual_rows if row["classification"] == classification))
        for classification in ("safe_plus_one", "active_but_replaces_existing", "redundant", "unsafe")
    }

    def simplified_cohort(row: dict[str, object]) -> str:
        if not bool(row["raw_candidate_exists"]):
            return "no_candidate_at_all"
        if not bool(row["raw_finite_good_candidate_exists"]):
            return "candidate_only_before_robust_segmentation"
        if not bool(row["after_core_preselection"]):
            return "raw_finite_good_lost_at_preselection"
        if not bool(row["accepted_max150"]):
            return "preselected_finite_good_lost_at_ranking"
        return "accepted_finite_good_inactive_or_redundant"

    cohort_order = [
        "no_candidate_at_all",
        "candidate_only_before_robust_segmentation",
        "raw_finite_good_lost_at_preselection",
        "preselected_finite_good_lost_at_ranking",
        "accepted_finite_good_inactive_or_redundant",
    ]
    missing_rows = [row for row in face_rows if not bool(row["final_canonical_covered"])]
    for row in missing_rows:
        row["missing_face_cohort"] = simplified_cohort(row)

    def feature_summary(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "raw_minima_count": distribution_summary([row["raw_line_minima_support_count"] for row in rows]),
            "stable_minima_z_levels": distribution_summary([row["stable_multiscale_minima_z_levels"] for row in rows]),
            "w2_raw_boundary_edges": distribution_summary([row["w2_raw_boundary_edge_count"] for row in rows]),
            "w2_robust_segments": distribution_summary([row["w2_robust_segment_count"] for row in rows]),
            "w2_matched_clusters": distribution_summary([row["w2_matched_edge_clusters"] for row in rows]),
            "raw_finite_good_candidates": distribution_summary([row["raw_finite_good_candidate_count"] for row in rows]),
        }

    gain_notes = {
        "no_candidate_at_all": "upper bound for new raw observation/detection recall",
        "candidate_only_before_robust_segmentation": "upper bound for segmentation, clustering, adjacency, or finite-plane formation",
        "raw_finite_good_lost_at_preselection": "upper bound for preselection recall",
        "preselected_finite_good_lost_at_ranking": "upper bound for support-diverse/compatibility selection",
        "accepted_finite_good_inactive_or_redundant": "upper bound for activity/blocker/exchange work",
    }
    missing_cohorts: dict[str, object] = {}
    for cohort in cohort_order:
        rows = [row for row in missing_rows if row["missing_face_cohort"] == cohort]
        zones: dict[str, int] = {}
        for row in rows:
            zones[str(row["zone"])] = zones.get(str(row["zone"]), 0) + 1
        missing_cohorts[cohort] = {
            "size": int(len(rows)),
            "zones": {key: int(value) for key, value in sorted(zones.items())},
            "typical_support_fit_features": feature_summary(rows),
            "representative_examples": [
                {
                    "face_id": int(row["face_id"]),
                    "zone": row["zone"],
                    "first_loss_category": row["first_loss_category"],
                    "raw_minima": int(row["raw_line_minima_support_count"]),
                    "stable_minima_z_levels": int(row["stable_multiscale_minima_z_levels"]),
                    "w2_raw_edges": int(row["w2_raw_boundary_edge_count"]),
                    "w2_robust_segments": int(row["w2_robust_segment_count"]),
                    "w2_clusters": int(row["w2_matched_edge_clusters"]),
                    "best_candidate_id": row["best_candidate_id"],
                    "best_canonical_distances": row["best_canonical_distances"],
                }
                for row in rows[:10]
            ],
            "potential_gain_next_fix": gain_notes[cohort],
        }

    branch_buckets = {
        "A_detection_segmentation": int(sum(category_counts[name] for name in (
            "no_raw_observed_support",
            "insufficient_stable_minima",
            "valley_pair_or_track_missing",
            "w2_boundary_edges_missing",
            "w2_segmentation_or_cluster_missing",
        ))),
        "B_candidate_formation_plane_fit": int(sum(category_counts[name] for name in (
            "face_hypothesis_missing",
            "plane_fit_rejected",
            "raw_candidate_not_finite_good",
        ))),
        "C_preselection_ranking": int(category_counts["lost_at_preselection"] + category_counts["ranked_out"]),
        "D_compatibility_activity": int(category_counts["accepted_but_never_active"] + category_counts["became_redundant_later"]),
        "E_evaluator_mapping": int(category_counts["canonical_mapping_conflict"]),
    }
    branch_key = max(branch_buckets, key=lambda key: (branch_buckets[key], key))
    branch_letter = branch_key[0]
    raw_ceiling = int(len(raw_finite_by_face))
    false_active_count = int(len(final_active_candidates) - anatomy["canonical_finite_good_unique"])
    if raw_ceiling - len(final_ids) <= 5 and false_active_count > len(missing_rows):
        branch_letter = "F"
        branch_key = "F_precision_cleanup"

    final_outside = evaluate_trusted_cloud_outside(
        final_candidates,
        trusted_points,
        z_slices,
        point_tol=float(point_tol),
    )
    timings["total_seconds"] = time.perf_counter() - started
    cache_blob = json.dumps(sorted(int(v) for v in match_cache), separators=(",", ":"))
    return {
        "diagnostic_only": True,
        "uses_initial_model": True,
        "production_selector_changed": False,
        "canonical_evaluator": {
            "schema_version": int(CANONICAL_ORACLE_SCHEMA_VERSION),
            "tolerances": CANONICAL_ORACLE_TOLERANCES,
            "final_unique_finite_face_ids": int(len(final_ids)),
            "final_finite_face_ids": sorted(int(v) for v in final_ids),
            "anatomy": anatomy,
        },
        "golden_confirmation": {
            "passed": bool(
                len(final_ids) == 129
                and len(final_reconstructed.get("vertices") or []) == 616
                and len(final_reconstructed.get("edges") or []) == 924
                and len(final_reconstructed.get("faces") or []) == 310
                and topology_is_valid(final_reconstructed.get("topology") if isinstance(final_reconstructed.get("topology"), dict) else {})
                and finite_float(final_outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
                and int(value_or_default(final_outside.get("cumulative_lost_z_levels"), 1)) == 0
            ),
            "retained_core": int(len(core_face_ids & final_ids)),
            "new_vs_repaired_core": int(len(final_ids - core_face_ids)),
            "unique": int(len(final_ids)),
            "lost_core_ids": sorted(int(v) for v in (core_face_ids - final_ids)),
            "vertices": int(len(final_reconstructed.get("vertices") or [])),
            "edges": int(len(final_reconstructed.get("edges") or [])),
            "faces": int(len(final_reconstructed.get("faces") or [])),
            "volume": final_reconstructed.get("reliable_volume"),
            "topology": final_reconstructed.get("topology"),
            "outside": final_outside,
            "geometry_signature": mesh_geometry_signature(final_candidates, final_reconstructed),
        },
        "methodology": {
            "raw_line_minima_support": "finite InitialModel polygon distance <= 0.15 for local minima cloud points",
            "stable_multiscale_minima_support": "trusted multiscale minima points within finite polygon distance 0.15 on >=2 z levels",
            "valley_observations": "sum of tracked levels inherited by raw multiscale valley/baseline fitted candidates assigned to the face",
            "w2_raw_boundary_support": "raw finite W2 line points within 0.08 of a model edge, >=3 points on >=3 z levels",
            "w2_robust_segment_support": "robust-split segment oracle edge match: median distance <=0.08, angle <=12 deg, finite-edge fraction >=0.5, dominant fraction >=0.65",
            "w2_matched_clusters": "shared oracle edge-cluster matcher with the same finite-edge confidence predicate",
            "stage_presence": "canonical finite-good candidate at that stage; active stages are full edge-clip faces",
            "support_diverse_ranking": "candidates actually reached in the cumulative support-diverse attempt order before max150 budget",
            "first_loss": "exclusive earliest applicable category in category_order",
            "individual_trials": "one non-oracle-quality representative per missing raw finite-good face, each added independently to final129",
            "patch_grouping": "complete-compatibility non-oracle spatial plane-patch groups; no single-linkage chaining",
            "source_contributions": "exclusive production lifecycle attribution; exchange targets override their earlier entry stage",
            "ceilings_not_jointly_achievable": True,
        },
        "category_order": category_order,
        "mutually_exclusive_first_loss_funnel": {
            "counts": category_counts,
            "sum": int(sum(category_counts.values())),
            "expected_total": int(len(model_faces)),
            "covered_final": int(category_counts["covered_final"]),
            "invariants_passed": bool(sum(category_counts.values()) == len(model_faces) and category_counts["covered_final"] == 129),
        },
        "face_stage_matrix": face_rows,
        "zone_definitions": zone_definitions,
        "zone_funnel": zone_rows,
        "source_contributions": source_contributions,
        "ceilings": {
            "candidate_existence": {
                "unique_finite_face_ids": int(len(raw_finite_by_face)),
                "face_ids": sorted(int(v) for v in raw_finite_by_face),
            },
            "preselection": {
                "unique_finite_face_ids": int(len(pre_finite_by_face)),
                "face_ids": sorted(int(v) for v in pre_finite_by_face),
            },
            "individually_compatible": {
                "representative_trial_count": int(len(individual_rows)),
                "class_counts": individual_counts,
                "safe_plus_one_face_ids": sorted(int(row["face_id"]) for row in individual_rows if row["classification"] == "safe_plus_one"),
                "isolation_upper_bound": int(len(final_ids) + individual_counts["safe_plus_one"]),
                "rows": individual_rows,
            },
            "patch_group": {
                "raw_finite_candidate_count": int(len(raw_finite_candidates)),
                "nonoracle_complete_compatibility_group_count": int(len(patch_groups)),
                "one_representative_unique_finite_face_ids": int(len(patch_ceiling_ids)),
                "face_ids": sorted(int(v) for v in patch_ceiling_ids),
                "note": "one representative per independently formed non-oracle patch group; not a jointly feasible mesh bound",
            },
        },
        "missing_face_cohorts": {
            "missing_face_count": int(len(missing_rows)),
            "cohort_sum": int(sum(int(row["size"]) for row in missing_cohorts.values())),
            "cohorts": missing_cohorts,
        },
        "false_active_anatomy": {
            "overall": anatomy,
            "by_source": {
                source: row.get("false_active_categories")
                for source, row in source_contributions.items()
                if source != "union_all_sources"
            },
        },
        "stage_geometry": {
            name: data["geometry_signature"]
            for name, data in stage_data.items()
        },
        "edge_ancestry_summary": {
            "raw_supported_edge_count": int(len(raw_supported_edge_ids)),
            "robust_segment_count": int(len(w2_segments)),
            "robust_confident_segment_count": int(sum(1 for row in segment_edge_rows if bool(row["confident"]))),
            "cluster_oracle": {
                key: cluster_oracle.get(key)
                for key in (
                    "matched_clusters",
                    "unmatched_or_ambiguous_clusters",
                    "ambiguous_clusters",
                    "unique_oracle_edge_ids",
                    "duplicate_clusters_on_edges",
                    "precision_confident",
                )
            },
            "segment_matches_sample": segment_edge_rows[:120],
        },
        "branch_decision": {
            "selected": branch_letter,
            "selected_key": branch_key,
            "bucket_counts": branch_buckets,
            "precision_cleanup_context": {
                "raw_ceiling_minus_final": int(raw_ceiling - len(final_ids)),
                "false_active_count": int(false_active_count),
            },
            "production_fix_implemented": False,
        },
        "oracle_match_cache": {
            "entries": int(len(match_cache)),
            "key_signature_sha256": hashlib.sha256(cache_blob.encode("utf-8")).hexdigest(),
        },
        "timings_seconds": {key: as_json_float(float(value)) for key, value in timings.items()},
        "stage_input_signatures": {
            "ranked_candidate_ids": [int(v) for v in support_diverse_ranked_candidate_ids],
            "max150_candidate_count": int(len(max150_candidates)),
            "append_candidate_count": int(len(append_candidates)),
            "repair_candidate_count": int(len(repair_candidates)),
            "ratchet_candidate_count": int(len(ratchet_candidates)),
            "final_candidate_count": int(len(final_candidates)),
            "append_added_ids": sorted(int(v) for v in (append_ids - max150_ids)),
            "repair_removed_ids": sorted(int(v) for v in (append_ids - repair_ids)),
            "ratchet_added_ids": sorted(int(v) for v in (ratchet_ids - repair_ids)),
            "exchange_removed_ids": sorted(int(v) for v in (ratchet_ids - final_candidate_ids)),
            "exchange_added_ids": sorted(int(v) for v in (final_candidate_ids - ratchet_ids)),
        },
        "_w2_edge_ancestry_internal": {
            "raw_edge_point_counts": raw_edge_point_counts,
            "raw_edge_z_levels": raw_edge_z_levels,
            "raw_edge_cyclic_z": raw_edge_cyclic_z,
            "segment_oracle_rows": segment_oracle_rows_internal,
        },
    }


def diagnose_preselection_safe_reservoir(
    *,
    cached_loss_funnel: dict[str, object],
    model_faces: list[dict[str, object]],
    initial_vertices: np.ndarray,
    core_candidates: list[dict[str, object]],
    core_reconstructed: dict[str, object],
    valley_raw_candidates: list[dict[str, object]],
    w2_raw_candidates: list[dict[str, object]],
    w2_preselection_candidates: list[dict[str, object]],
    w2_clusters: list[dict[str, object]],
    support_diverse_ranked_candidate_ids: list[int],
    selection_attempt_log: list[dict[str, object]],
    max150_candidates: list[dict[str, object]],
    append_candidates: list[dict[str, object]],
    repair_candidates: list[dict[str, object]],
    ratchet_candidates: list[dict[str, object]],
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    removed_blocker_ids: set[int],
    append_order_ids: list[int],
    broad_order_ids: list[int],
    core_face_ids: set[int],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    finite_support_plane_distance: float,
    finite_support_hull_margin: float,
    finite_support_relative_hull_margin: float,
    contours: list[object],
    match_cache: dict[int, dict[str, object] | None],
) -> dict[str, object]:
    """Oracle-seeded, production-read-only audit of the final129 addition reservoir."""

    started = time.perf_counter()
    timings: dict[str, float] = {}
    z_slices = prepare_z_level_slices(trusted_z_indices)

    def unique_candidates(pools: list[list[dict[str, object]]]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        seen: set[int] = set()
        for pool in pools:
            for candidate in pool:
                candidate_id = int(candidate_track_id(candidate))
                if candidate_id < 0 or candidate_id in seen:
                    continue
                seen.add(candidate_id)
                out.append(candidate)
        return out

    all_formed_candidates = unique_candidates(
        [
            list(valley_raw_candidates),
            list(w2_raw_candidates),
            list(w2_preselection_candidates),
            list(core_candidates),
            list(max150_candidates),
            list(append_candidates),
            list(repair_candidates),
            list(ratchet_candidates),
            list(final_candidates),
        ]
    )
    candidate_by_id = {int(candidate_track_id(candidate)): candidate for candidate in all_formed_candidates}

    def oracle_match(candidate: dict[str, object]) -> dict[str, object] | None:
        candidate_id = int(candidate_track_id(candidate))
        if candidate_id not in match_cache:
            match_cache[candidate_id] = candidate_oracle_face(candidate, model_faces)
        return match_cache.get(candidate_id)

    def active_indices(candidates: list[dict[str, object]], rec: dict[str, object]) -> list[int]:
        return [
            int(index)
            for index in rec.get("face_candidate_indices", [])
            if 0 <= int(index) < len(candidates)
        ]

    def active_candidates(candidates: list[dict[str, object]], rec: dict[str, object]) -> list[dict[str, object]]:
        return [candidates[index] for index in active_indices(candidates, rec)]

    def canonical_ids(candidates: list[dict[str, object]], rec: dict[str, object]) -> set[int]:
        result: set[int] = set()
        for candidate in active_candidates(candidates, rec):
            match = oracle_match(candidate)
            if match is not None and bool(match.get("finite_good")) and match.get("face_id") is not None:
                result.add(int(match["face_id"]))
        return result

    reconstruction_cache: dict[tuple[int, ...], dict[str, object]] = {}

    def reconstruction(candidates: list[dict[str, object]]) -> dict[str, object]:
        key = tuple(int(candidate_track_id(candidate)) for candidate in candidates)
        cached = reconstruction_cache.get(key)
        if cached is not None:
            return cached
        rec = reconstruct_polyhedron_from_halfspaces_edge_clip(candidates, **reconstruction_kwargs)
        reconstruction_cache[key] = rec
        return rec

    final_key = tuple(int(candidate_track_id(candidate)) for candidate in final_candidates)
    reconstruction_cache[final_key] = final_reconstructed
    final_active_candidates = active_candidates(final_candidates, final_reconstructed)
    final_active_ids = {int(candidate_track_id(candidate)) for candidate in final_active_candidates}
    final_selected_ids = {int(candidate_track_id(candidate)) for candidate in final_candidates}
    final_canonical_ids = canonical_ids(final_candidates, final_reconstructed)
    final_outside = evaluate_trusted_cloud_outside(
        final_candidates,
        trusted_points,
        z_slices,
        point_tol=float(point_tol),
    )

    def build_patch_groups(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        groups: list[list[dict[str, object]]] = []
        for candidate in sorted(candidates, key=nonoracle_candidate_quality_key):
            for group in groups:
                if all(candidates_plane_patch_compatible(candidate, member) for member in group):
                    group.append(candidate)
                    break
            else:
                groups.append([candidate])
        return [
            {
                "group_id": f"active_patch_{index:04d}_{candidate_track_id(group[0])}",
                "members": group,
                "member_candidate_ids": sorted(int(candidate_track_id(member)) for member in group),
                "representative_candidate_id": int(candidate_track_id(group[0])),
            }
            for index, group in enumerate(groups)
        ]

    baseline_patch_groups = build_patch_groups(final_active_candidates)

    def group_public_row(group: dict[str, object]) -> dict[str, object]:
        return {
            "group_id": str(group.get("group_id")),
            "representative_candidate_id": int(group.get("representative_candidate_id") or -1),
            "member_candidate_ids": [int(value) for value in group.get("member_candidate_ids", [])],
        }

    def preservation_summary(
        protected_groups: list[dict[str, object]],
        trial_active: list[dict[str, object]],
    ) -> dict[str, object]:
        lost: list[dict[str, object]] = []
        for group in protected_groups:
            members = list(group.get("members") or [])
            represented = any(
                candidates_plane_patch_compatible(member, candidate)
                for member in members
                for candidate in trial_active
            )
            if not represented:
                lost.append(group_public_row(group))
        return {
            "protected_group_count": int(len(protected_groups)),
            "lost_group_count": int(len(lost)),
            "all_preserved": bool(not lost),
            "lost_groups": lost[:30],
        }

    def ratchet_groups(
        protected_groups: list[dict[str, object]],
        state_active: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        out = [dict(group, members=list(group.get("members") or [])) for group in protected_groups]
        for candidate in sorted(state_active, key=nonoracle_candidate_quality_key):
            if any(
                candidates_plane_patch_compatible(candidate, member)
                for group in out
                for member in list(group.get("members") or [])
            ):
                continue
            group_index = len(out)
            out.append(
                {
                    "group_id": f"rolling_patch_{group_index:04d}_{candidate_track_id(candidate)}",
                    "members": [candidate],
                    "member_candidate_ids": [int(candidate_track_id(candidate))],
                    "representative_candidate_id": int(candidate_track_id(candidate)),
                }
            )
        return out

    def candidate_face_geometry(
        candidates: list[dict[str, object]],
        rec: dict[str, object],
        candidate_id: int,
    ) -> dict[str, object]:
        vertices = np.asarray(rec.get("vertices") or [], dtype=float).reshape((-1, 3))
        for face, source_index in zip(rec.get("faces") or [], rec.get("face_candidate_indices") or []):
            index = int(source_index)
            if index < 0 or index >= len(candidates):
                continue
            candidate = candidates[index]
            if int(candidate_track_id(candidate)) != int(candidate_id):
                continue
            face_indices = np.asarray(face, dtype=int)
            if face_indices.size < 3 or np.any(face_indices < 0) or np.any(face_indices >= vertices.shape[0]):
                continue
            points = vertices[face_indices]
            face_area = float(polygon_area(points))
            hull = candidate_hull_points(candidate)
            hull_area = max(finite_float(candidate.get("hull_area"), 0.0), EPS)
            containment = None
            distance_p95 = None
            plane = candidate_plane(candidate)
            if plane is not None and hull.shape[0] >= 3:
                origin, normal = plane
                u, v = plane_basis(normal)
                face_polygon = convex_hull_2d(np.column_stack([(points - origin) @ u, (points - origin) @ v]))
                hull_polygon = convex_hull_2d(np.column_stack([(hull - origin) @ u, (hull - origin) @ v]))
                if face_polygon.shape[0] >= 3 and hull_polygon.shape[0] >= 3:
                    hull_to_face = point_to_polygon_distance_2d(hull_polygon, face_polygon)
                    containment = float(np.mean(hull_to_face <= 0.02)) if hull_to_face.size else None
                    distance_p95 = float(np.percentile(hull_to_face, 95)) if hull_to_face.size else None
            return {
                "active": True,
                "candidate_index": int(index),
                "face_area": as_json_float(face_area),
                "face_centroid": as_json_point(np.mean(points, axis=0)),
                "face_vertex_count": int(face_indices.size),
                "face_hull_area_ratio": as_json_float(face_area / hull_area),
                "face_extra_area": as_json_float(max(0.0, face_area - hull_area)),
                "hull_to_face_containment_fraction": as_json_float(containment) if containment is not None else None,
                "hull_to_face_distance_p95": as_json_float(distance_p95) if distance_p95 is not None else None,
            }
        return {"active": False}

    def candidate_samples(candidate: dict[str, object]) -> list[np.ndarray]:
        hull = candidate_hull_points(candidate)
        if hull.shape[0] == 0:
            centroid = candidate_hull_centroid(candidate)
            return [centroid] if centroid is not None else []
        centroid = np.mean(hull, axis=0)
        samples = [centroid]
        samples.extend(0.7 * centroid + 0.3 * point for point in hull)
        samples.extend(0.5 * point + 0.5 * hull[(index + 1) % len(hull)] for index, point in enumerate(hull))
        return [np.asarray(sample, dtype=float) for sample in samples]

    baseline_active_planes = [candidate_plane(candidate) for candidate in final_active_candidates]
    baseline_active_planes = [plane for plane in baseline_active_planes if plane is not None]
    trusted_array = np.asarray(trusted_points, dtype=float).reshape((-1, 3))
    if baseline_active_planes and trusted_array.shape[0]:
        baseline_normals = np.asarray([plane[1] for plane in baseline_active_planes], dtype=float)
        baseline_offsets = np.asarray([float(plane[1] @ plane[0]) for plane in baseline_active_planes], dtype=float)
        baseline_plane_distance = np.min(
            np.abs(trusted_array @ baseline_normals.T - baseline_offsets[None, :]),
            axis=1,
        )
    else:
        baseline_plane_distance = np.full(trusted_array.shape[0], float("inf"), dtype=float)

    def local_surface_features(candidate: dict[str, object], trial_candidates: list[dict[str, object]]) -> dict[str, object]:
        samples = candidate_samples(candidate)
        before = patch_surface_deficit_for_samples(final_candidates, samples, sample_tol=0.01)
        after = patch_surface_deficit_for_samples(trial_candidates, samples, sample_tol=0.01)
        before_value = finite_float(before.get("area_weighted_cut_deficit"), 0.0)
        after_value = finite_float(after.get("area_weighted_cut_deficit"), 0.0)
        plane = candidate_plane(candidate)
        local_reduction = 0.0
        unsupported_reduction = 0.0
        local_count = 0
        if plane is not None and trusted_array.shape[0]:
            point, normal = plane
            hull = candidate_hull_points(candidate)
            if hull.shape[0] >= 3:
                u, v = plane_basis(normal)
                hull_polygon = convex_hull_2d(np.column_stack([(hull - point) @ u, (hull - point) @ v]))
                projected = np.column_stack([(trusted_array - point) @ u, (trusted_array - point) @ v])
                hull_distance = point_to_polygon_distance_2d(projected, hull_polygon)
                plane_distance = np.abs((trusted_array - point) @ normal)
                z_min, z_max = candidate_z_range(candidate)
                z_mask = np.ones(trusted_array.shape[0], dtype=bool)
                if np.isfinite(z_min) and np.isfinite(z_max):
                    z_mask = (trusted_array[:, 2] >= z_min - 0.08) & (trusted_array[:, 2] <= z_max + 0.08)
                local_indices = np.flatnonzero(z_mask & (plane_distance <= 0.12) & (hull_distance <= 0.18))
                if local_indices.size > 320:
                    order = np.argsort(plane_distance[local_indices] + hull_distance[local_indices])
                    local_indices = local_indices[order[:320]]
                local_count = int(local_indices.size)
                if local_indices.size:
                    candidate_distance = np.sqrt(
                        plane_distance[local_indices] ** 2 + hull_distance[local_indices] ** 2
                    )
                    distance_before = baseline_plane_distance[local_indices]
                    distance_after = np.minimum(distance_before, candidate_distance)
                    local_reduction = float(np.median(distance_before - distance_after))
                    unsupported_reduction = float(
                        np.mean(distance_before > 0.05) - np.mean(distance_after > 0.05)
                    )
        return {
            "surface_deficit_before": before,
            "surface_deficit_after": after,
            "surface_deficit_reduction": as_json_float(before_value - after_value),
            "local_contour_support_count": int(local_count),
            "local_contour_support_delta": as_json_float(local_reduction),
            "unsupported_fraction_reduction": as_json_float(unsupported_reduction),
        }

    cohort_container = (cached_loss_funnel.get("ceilings") or {}).get("individually_compatible")
    cohort_rows_cached = list((cohort_container or {}).get("rows") or []) if isinstance(cohort_container, dict) else []
    cohort_rows_cached = [row for row in cohort_rows_cached if isinstance(row, dict)]
    cohort_by_candidate_id = {
        int(row["representative_candidate_id"]): row
        for row in cohort_rows_cached
        if row.get("representative_candidate_id") is not None
    }
    safe_seed_rows = [row for row in cohort_rows_cached if str(row.get("classification")) == "safe_plus_one"]
    safe_seed_ids = {int(row["representative_candidate_id"]) for row in safe_seed_rows}

    stage_specs = [
        ("max150", max150_candidates),
        ("append_local", append_candidates),
        ("lp_blockers", repair_candidates),
        ("broad_ratchet", ratchet_candidates),
        ("production129", final_candidates),
    ]
    stage_membership = {
        name: {int(candidate_track_id(candidate)) for candidate in stage_candidates}
        for name, stage_candidates in stage_specs
    }
    stage_active: dict[str, set[int]] = {}
    stage_geometry: dict[str, object] = {}
    stage_started = time.perf_counter()
    for name, stage_candidates in stage_specs:
        rec = final_reconstructed if name == "production129" else reconstruction(stage_candidates)
        stage_active[name] = {
            int(candidate_track_id(stage_candidates[index]))
            for index in active_indices(stage_candidates, rec)
        }
        stage_geometry[name] = mesh_geometry_signature(stage_candidates, rec)
    timings["stage_history_edge_clip_seconds"] = time.perf_counter() - stage_started

    support_rank = {int(candidate_id): rank for rank, candidate_id in enumerate(support_diverse_ranked_candidate_ids, start=1)}
    append_rank = {int(candidate_id): rank for rank, candidate_id in enumerate(append_order_ids, start=1)}
    broad_rank = {int(candidate_id): rank for rank, candidate_id in enumerate(broad_order_ids, start=1)}
    attempt_reason: dict[int, str] = {}
    for row in selection_attempt_log:
        if row.get("track_id") is not None:
            attempt_reason[int(row["track_id"])] = str(row.get("reason") or "unknown")
    w2_raw_ids = {int(candidate_track_id(candidate)) for candidate in w2_raw_candidates}
    w2_preselection_ids = {int(candidate_track_id(candidate)) for candidate in w2_preselection_candidates}
    valley_raw_ids = {int(candidate_track_id(candidate)) for candidate in valley_raw_candidates}

    def stage_history(candidate: dict[str, object], oracle_face_id: int | None) -> dict[str, object]:
        candidate_id = int(candidate_track_id(candidate))
        if candidate_id in w2_raw_ids and candidate_id not in w2_preselection_ids:
            absence_reason = str(candidate.get("core_baseline_rejection_reason") or "core_preselection_rejected")
        elif candidate_id in w2_preselection_ids and candidate_id not in stage_membership["max150"]:
            absence_reason = attempt_reason.get(candidate_id, "ranked_out_or_budget_exhausted")
        elif candidate_id in stage_membership["max150"] and candidate_id not in stage_active["max150"]:
            absence_reason = "accepted_but_inactive"
        elif candidate_id in final_selected_ids and candidate_id not in final_active_ids:
            absence_reason = "accepted_but_redundant"
        elif candidate_id not in final_selected_ids:
            absence_reason = "not_selected_before_final129"
        else:
            absence_reason = "active_in_final129"
        return {
            "oracle_face_id_posthoc": int(oracle_face_id) if oracle_face_id is not None else None,
            "candidate_id": int(candidate_id),
            "source": str(candidate.get("candidate_source") or candidate.get("candidate_origin") or "unknown"),
            "origin": str(candidate.get("candidate_origin") or "unknown"),
            "window": candidate.get("window"),
            "scale": candidate.get("scale_mode"),
            "cluster_id": candidate.get("dedupe_cluster_id"),
            "source_track_id": candidate.get("source_track_id"),
            "w2_edge_cluster_ids": [int(value) for value in candidate.get("w2_edge_cluster_ids", [])],
            "support_diverse_rank": support_rank.get(candidate_id),
            "append_local_rank": append_rank.get(candidate_id),
            "broad_ratchet_rank": broad_rank.get(candidate_id),
            "selection_rejection_reason": attempt_reason.get(candidate_id) or candidate.get("selection_reason"),
            "static_gate_rejection_reason": (
                candidate.get("core_baseline_rejection_reason")
                or candidate.get("valley_quality_rejection_reason")
                or candidate.get("cluster_rejection_reason")
            ),
            "stage_selected": {name: bool(candidate_id in stage_membership[name]) for name, _ in stage_specs},
            "stage_active": {name: bool(candidate_id in stage_active[name]) for name, _ in stage_specs},
            "final129_absence_reason": absence_reason,
        }

    active_core_ids = {
        int(candidate_track_id(core_candidates[index]))
        for index in active_indices(core_candidates, core_reconstructed)
    }
    annotated_started = time.perf_counter()
    formed_source_pool = unique_candidates(
        [list(valley_raw_candidates), list(w2_raw_candidates), list(w2_preselection_candidates)]
    )
    annotated_pool = annotate_candidate_pool_neutral(
        formed_source_pool,
        core_candidates=core_candidates,
        core_reconstructed=core_reconstructed,
        trusted_points=trusted_points,
        trusted_z_indices=trusted_z_indices,
        point_tol=float(point_tol),
        finite_support_plane_distance=float(finite_support_plane_distance),
        finite_support_hull_margin=float(finite_support_hull_margin),
        finite_support_relative_hull_margin=float(finite_support_relative_hull_margin),
    )
    for candidate in annotated_pool:
        candidate_by_id[int(candidate_track_id(candidate))] = candidate
    timings["generic_pool_annotation_seconds"] = time.perf_counter() - annotated_started

    pool_funnel = {
        "formed_existing_candidates": int(len(annotated_pool)),
        "already_selected": 0,
        "removed_blocker": 0,
        "hard_protected_core": 0,
        "invalid_plane_or_hull": 0,
        "no_observed_support": 0,
        "eligible_before_patch_dedupe": 0,
    }
    eligible_pool: list[dict[str, object]] = []
    for candidate in annotated_pool:
        candidate_id = int(candidate_track_id(candidate))
        if candidate_id in final_selected_ids:
            pool_funnel["already_selected"] += 1
            continue
        if candidate_id in removed_blocker_ids:
            pool_funnel["removed_blocker"] += 1
            continue
        if candidate_id in active_core_ids:
            pool_funnel["hard_protected_core"] += 1
            continue
        plane = candidate_plane(candidate)
        hull = candidate_hull_points(candidate)
        if plane is None or hull.shape[0] < 3 or finite_float(candidate.get("hull_area"), 0.0) <= 0.0:
            pool_funnel["invalid_plane_or_hull"] += 1
            continue
        support = max(
            int(candidate.get("finite_support_count") or 0),
            int(candidate.get("observed_local_support") or 0),
            int(candidate.get("observed_support_near_count") or 0),
            int(candidate.get("fit_points") or 0),
            int(candidate.get("levels") or 0),
        )
        if support <= 0:
            pool_funnel["no_observed_support"] += 1
            continue
        eligible_pool.append(candidate)
    pool_funnel["eligible_before_patch_dedupe"] = int(len(eligible_pool))

    generic_groups: list[list[dict[str, object]]] = []
    for candidate in sorted(eligible_pool, key=nonoracle_candidate_quality_key):
        for group in generic_groups:
            if all(candidates_plane_patch_compatible(candidate, member) for member in group):
                group.append(candidate)
                break
        else:
            generic_groups.append([candidate])
    generic_representatives = [min(group, key=nonoracle_candidate_quality_key) for group in generic_groups]
    pool_funnel["complete_compatibility_patch_groups"] = int(len(generic_groups))
    pool_funnel["generic_representatives"] = int(len(generic_representatives))

    consensus_pool = list(eligible_pool)

    def consensus_features(candidate: dict[str, object]) -> dict[str, object]:
        compatible = [
            other
            for other in consensus_pool
            if int(candidate_track_id(other)) != int(candidate_track_id(candidate))
            and candidates_plane_patch_compatible(candidate, other)
        ]
        source_tokens = {
            (
                str(other.get("candidate_origin") or other.get("candidate_source") or "unknown"),
                int(other.get("window") or -1),
                tuple(int(value) for value in other.get("w2_edge_cluster_ids", [])),
            )
            for other in compatible
        }
        return {
            "compatible_candidate_count": int(len(compatible)),
            "independent_plane_consensus": int(len(source_tokens)),
            "compatible_candidate_ids_sample": sorted(int(candidate_track_id(other)) for other in compatible)[:20],
        }

    lp_engine = LpActivityEngine(enabled=True)
    lp_universe = unique_candidates([list(final_candidates), list(generic_representatives)])
    lp_engine.set_reusable_universe("preselection_safe_reservoir", lp_universe, inside_tol=0.03)
    base_candidate_ids = tuple(int(candidate_track_id(candidate)) for candidate in final_candidates)

    def lp_features(candidate: dict[str, object]) -> dict[str, object]:
        return lp_face_activity_details(
            None,
            candidate,
            inside_tol=0.03,
            face_activity_slack=0.03,
            include_point=False,
            lp_engine=lp_engine,
            base_candidate_ids=base_candidate_ids,
        )

    trial_cache: dict[int, dict[str, object]] = {}
    baseline_reprojection = evaluate_polyhedron_reprojection(
        name="production129",
        vertices_3d=np.asarray(final_reconstructed.get("vertices") or [], dtype=float),
        contours=contours,
        direction_count=24,
        contour_sample=24,
    )

    def evaluate_individual(candidate: dict[str, object], *, include_reprojection: bool) -> dict[str, object]:
        candidate_id = int(candidate_track_id(candidate))
        cached = trial_cache.get(candidate_id)
        if cached is not None and (not include_reprojection or cached.get("reprojection") is not None):
            return cached
        trial_candidates = list(final_candidates)
        if candidate_id not in final_selected_ids:
            trial_candidates.append(candidate)
        rec = reconstruction(trial_candidates)
        trial_active = active_candidates(trial_candidates, rec)
        active_ids = {int(candidate_track_id(item)) for item in trial_active}
        trial_canonical = canonical_ids(trial_candidates, rec)
        outside = evaluate_trusted_cloud_outside(
            trial_candidates,
            trusted_points,
            z_slices,
            point_tol=float(point_tol),
        )
        preservation = preservation_summary(baseline_patch_groups, trial_active)
        face_geometry = candidate_face_geometry(trial_candidates, rec, candidate_id)
        target_active = bool(candidate_id in active_ids and face_geometry.get("active"))
        nonoracle_safe = bool(
            target_active
            and preservation.get("all_preserved")
            and topology_is_valid(rec.get("topology") if isinstance(rec.get("topology"), dict) else {})
            and finite_float(outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
            and int(value_or_default(outside.get("cumulative_lost_z_levels"), 1)) == 0
        )
        gained = trial_canonical - final_canonical_ids
        lost = final_canonical_ids - trial_canonical
        lost_core = core_face_ids - trial_canonical
        if not nonoracle_safe or lost_core:
            classification = "unsafe"
        elif gained and not lost and len(trial_canonical) > len(final_canonical_ids):
            classification = "safe_plus_one"
        elif target_active and (lost or len(trial_canonical) <= len(final_canonical_ids)):
            classification = "active_but_replaces_existing"
        else:
            classification = "redundant"
        surface = local_surface_features(candidate, trial_candidates)
        reprojection = None
        if include_reprojection:
            reprojection = evaluate_polyhedron_reprojection(
                name=f"plus_{candidate_id}",
                vertices_3d=np.asarray(rec.get("vertices") or [], dtype=float),
                contours=contours,
                direction_count=24,
                contour_sample=24,
            )
        result = {
            "candidate_id": int(candidate_id),
            "classification": classification,
            "target_candidate_active": bool(target_active),
            "canonical_unique": int(len(trial_canonical)),
            "canonical_ids_gained": sorted(int(value) for value in gained),
            "canonical_ids_lost": sorted(int(value) for value in lost),
            "retained_core": int(len(core_face_ids & trial_canonical)),
            "lost_core_ids": sorted(int(value) for value in lost_core),
            "rolling_groups_preserved": bool(preservation.get("all_preserved")),
            "patch_preservation": preservation,
            "face_geometry": face_geometry,
            "lp_activity": lp_features(candidate),
            "surface": surface,
            "vertices": int(len(rec.get("vertices") or [])),
            "edges": int(len(rec.get("edges") or [])),
            "faces": int(len(rec.get("faces") or [])),
            "volume": rec.get("reliable_volume"),
            "volume_delta": as_json_float(
                finite_float(rec.get("reliable_volume"), 0.0)
                - finite_float(final_reconstructed.get("reliable_volume"), 0.0)
            ),
            "topology": rec.get("topology"),
            "trusted_outside": outside,
            "geometry_signature": mesh_geometry_signature(trial_candidates, rec),
            "reprojection": reprojection,
            "reprojection_delta": {
                key: as_json_float(
                    finite_float((reprojection or {}).get(key), 0.0)
                    - finite_float(baseline_reprojection.get(key), 0.0)
                )
                if reprojection is not None
                else None
                for key in (
                    "support_abs_median",
                    "support_abs_p95",
                    "symmetric_contour_distance_median",
                    "symmetric_contour_distance_p95",
                )
            },
        }
        trial_cache[candidate_id] = result
        return result

    individual_started = time.perf_counter()
    fresh_safe_rows: list[dict[str, object]] = []
    for seed in sorted(safe_seed_rows, key=lambda row: int(row.get("face_id") or -1)):
        candidate_id = int(seed.get("representative_candidate_id") or -1)
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            fresh_safe_rows.append(
                {
                    "oracle_face_id_posthoc": seed.get("face_id"),
                    "candidate_id": int(candidate_id),
                    "reproduced": False,
                    "reason": "representative_candidate_not_found_in_current_pool",
                }
            )
            continue
        fresh = evaluate_individual(candidate, include_reprojection=True)
        fresh_safe_rows.append(
            {
                "oracle_face_id_posthoc": int(seed.get("face_id") or -1),
                "candidate_id": int(candidate_id),
                "cached_classification": seed.get("classification"),
                "fresh": fresh,
                "reproduced": bool(
                    fresh.get("classification") == "safe_plus_one"
                    and int(seed.get("face_id") or -1) in set(int(value) for value in fresh.get("canonical_ids_gained", []))
                    and not fresh.get("canonical_ids_lost")
                    and not fresh.get("lost_core_ids")
                ),
            }
        )
    timings["fresh_safe_individual_seconds"] = time.perf_counter() - individual_started

    feature_directions = {
        "lp_effective_margin": "higher",
        "trial_face_area": "higher",
        "hull_area": "higher",
        "face_hull_area_ratio": "closer_to_one",
        "face_extra_area": "lower",
        "hull_to_face_distance_p95": "lower",
        "finite_support_count": "higher",
        "finite_support_density": "higher",
        "finite_support_residual_p95": "lower",
        "finite_support_purity": "higher",
        "finite_support_balance": "higher",
        "finite_support_track_coverage": "higher",
        "finite_support_z_coverage": "higher",
        "plane_rms": "lower",
        "condition_p95": "lower",
        "observed_support_gap_local": "lower",
        "surface_deficit_reduction": "higher",
        "patch_novelty": "higher",
        "independent_plane_consensus": "higher",
        "local_contour_support_delta": "higher",
    }

    def feature_row(candidate: dict[str, object], trial: dict[str, object]) -> dict[str, object]:
        candidate_id = int(candidate_track_id(candidate))
        nearest_relations = [
            candidate_plane_relation_score(candidate, active)
            for active in final_active_candidates
        ]
        nearest = min(nearest_relations, default=(float("inf"), float("inf"), float("inf")))
        patch_novel = not any(candidates_plane_patch_compatible(candidate, active) for active in final_active_candidates)
        face = trial.get("face_geometry") if isinstance(trial.get("face_geometry"), dict) else {}
        lp = trial.get("lp_activity") if isinstance(trial.get("lp_activity"), dict) else {}
        surface = trial.get("surface") if isinstance(trial.get("surface"), dict) else {}
        consensus = consensus_features(candidate)
        return {
            "candidate_id": int(candidate_id),
            "oracle_face_id_posthoc": (cohort_by_candidate_id.get(candidate_id) or {}).get("face_id"),
            "oracle_classification_posthoc": (cohort_by_candidate_id.get(candidate_id) or {}).get("classification"),
            "fresh_classification_posthoc": trial.get("classification"),
            "lp_active": bool(lp.get("active")),
            "lp_effective_margin": lp.get("effective_face_margin"),
            "trial_face_active": bool(face.get("active")),
            "trial_face_area": face.get("face_area"),
            "hull_area": candidate.get("hull_area"),
            "face_hull_area_ratio": face.get("face_hull_area_ratio"),
            "face_extra_area": face.get("face_extra_area"),
            "hull_to_face_distance_p95": face.get("hull_to_face_distance_p95"),
            "hull_to_face_containment_fraction": face.get("hull_to_face_containment_fraction"),
            "finite_support_count": candidate.get("finite_support_count"),
            "finite_support_density": candidate.get("finite_support_density"),
            "finite_support_residual_p95": candidate.get("finite_support_residual_p95"),
            "finite_support_purity": candidate.get("finite_support_purity"),
            "finite_support_balance": candidate.get("finite_support_balance"),
            "finite_support_track_coverage": candidate.get("finite_support_track_coverage"),
            "finite_support_z_coverage": candidate.get("finite_support_z_coverage"),
            "plane_rms": candidate.get("plane_rms"),
            "condition_p95": candidate.get("condition_p95"),
            "observed_support_gap_local": candidate.get("observed_support_gap_local"),
            "surface_deficit_reduction": surface.get("surface_deficit_reduction"),
            "local_contour_support_delta": surface.get("local_contour_support_delta"),
            "unsupported_fraction_reduction": surface.get("unsupported_fraction_reduction"),
            "patch_novelty": int(patch_novel),
            "nearest_active_angle_deg": as_json_float(nearest[0]),
            "nearest_active_offset": as_json_float(nearest[1]),
            "nearest_active_centroid_distance": as_json_float(nearest[2]),
            **consensus,
            "rolling_groups_preserved": bool(trial.get("rolling_groups_preserved")),
        }

    cohort_feature_started = time.perf_counter()
    cohort_feature_rows: list[dict[str, object]] = []
    for cached_row in cohort_rows_cached:
        candidate_id = int(cached_row.get("representative_candidate_id") or -1)
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            continue
        trial = evaluate_individual(candidate, include_reprojection=candidate_id in safe_seed_ids)
        cohort_feature_rows.append(feature_row(candidate, trial))
    timings["bounded_51_feature_trials_seconds"] = time.perf_counter() - cohort_feature_started

    def feature_value(row: dict[str, object], name: str, direction: str) -> float | None:
        value = row.get(name)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number):
            return None
        if direction == "lower":
            return -number
        if direction == "closer_to_one":
            return -abs(number - 1.0)
        return number

    def auc_for(rows: list[dict[str, object]], feature_name: str, direction: str) -> dict[str, object]:
        positives = [
            value
            for row in rows
            if str(row.get("fresh_classification_posthoc")) == "safe_plus_one"
            for value in [feature_value(row, feature_name, direction)]
            if value is not None
        ]
        negatives = [
            value
            for row in rows
            if str(row.get("fresh_classification_posthoc")) in {"redundant", "active_but_replaces_existing"}
            for value in [feature_value(row, feature_name, direction)]
            if value is not None
        ]
        wins = sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0 for positive in positives for negative in negatives)
        auc = wins / max(1, len(positives) * len(negatives))
        return {
            "direction": direction,
            "safe_count": int(len(positives)),
            "control_count": int(len(negatives)),
            "auc": as_json_float(float(auc)) if positives and negatives else None,
            "safe_distribution": distribution_summary(positives),
            "control_distribution": distribution_summary(negatives),
        }

    feature_separation = {
        name: auc_for(cohort_feature_rows, name, direction)
        for name, direction in feature_directions.items()
    }

    def validator_flags(row: dict[str, object]) -> dict[str, bool]:
        activity_local = bool(
            row.get("lp_active")
            and row.get("trial_face_active")
            and finite_float(row.get("face_hull_area_ratio"), float("inf")) <= 20.0
            and finite_float(row.get("face_extra_area"), float("inf")) <= 1.0
        )
        novelty = bool(row.get("trial_face_active") and int(row.get("patch_novelty") or 0) > 0)
        surface = bool(
            finite_float(row.get("surface_deficit_reduction"), 0.0) > 1e-6
            or finite_float(row.get("local_contour_support_delta"), 0.0) > 1e-6
        )
        consensus = bool(
            int(row.get("independent_plane_consensus") or 0) >= 2
            and int(row.get("finite_support_count") or 0) >= 3
            and finite_float(row.get("finite_support_residual_p95"), float("inf")) <= 0.08
            and finite_float(row.get("finite_support_track_coverage"), 0.0) > 0.0
        )
        evidence = int(activity_local) + int(novelty) + int(surface) + int(consensus)
        combined = bool(activity_local and novelty and evidence >= 3 and row.get("rolling_groups_preserved"))
        return {
            "A_activity_local_face": activity_local,
            "B_patch_novelty": novelty,
            "C_surface_deficit": surface,
            "D_consensus_support": consensus,
            "E_combined": combined,
        }

    for row in cohort_feature_rows:
        row["validator_flags"] = validator_flags(row)
        row["independent_evidence_family_count"] = int(sum(validator_flags(row).values()) - int(validator_flags(row)["E_combined"]))

    def validator_summary(name: str) -> dict[str, object]:
        accepted = [row for row in cohort_feature_rows if bool((row.get("validator_flags") or {}).get(name))]
        safe = [row for row in accepted if str(row.get("fresh_classification_posthoc")) == "safe_plus_one"]
        redundant = [row for row in accepted if str(row.get("fresh_classification_posthoc")) == "redundant"]
        replacement = [row for row in accepted if str(row.get("fresh_classification_posthoc")) == "active_but_replaces_existing"]
        all_safe = [row for row in cohort_feature_rows if str(row.get("fresh_classification_posthoc")) == "safe_plus_one"]
        damaged = [row for row in accepted if not bool(row.get("rolling_groups_preserved"))]
        return {
            "accepted_count": int(len(accepted)),
            "safe_count": int(len(safe)),
            "safe_precision": as_json_float(len(safe) / max(1, len(accepted))),
            "safe_recall": as_json_float(len(safe) / max(1, len(all_safe))),
            "redundant_count": int(len(redundant)),
            "replacement_count": int(len(replacement)),
            "safe_face_ids_posthoc": sorted(int(row["oracle_face_id_posthoc"]) for row in safe if row.get("oracle_face_id_posthoc") is not None),
            "accepted_candidate_ids": [int(row["candidate_id"]) for row in accepted],
            "current_controls_damaged": int(len(damaged)),
        }

    validator_summaries = {
        name: validator_summary(name)
        for name in (
            "A_activity_local_face",
            "B_patch_novelty",
            "C_surface_deficit",
            "D_consensus_support",
            "E_combined",
        )
    }

    def quality_key(row: dict[str, object]) -> tuple[object, ...]:
        flags = row.get("validator_flags") if isinstance(row.get("validator_flags"), dict) else {}
        evidence_count = int(row.get("independent_evidence_family_count") or 0)
        return (
            0 if bool(flags.get("E_combined")) else 1,
            -evidence_count,
            0 if bool(row.get("lp_active")) else 1,
            0 if bool(row.get("trial_face_active")) else 1,
            -int(row.get("patch_novelty") or 0),
            -finite_float(row.get("surface_deficit_reduction"), 0.0),
            -finite_float(row.get("local_contour_support_delta"), 0.0),
            -int(row.get("independent_plane_consensus") or 0),
            -finite_float(row.get("lp_effective_margin"), -1e9),
            finite_float(row.get("finite_support_residual_p95"), float("inf")),
            finite_float(row.get("plane_rms"), float("inf")),
            int(row.get("candidate_id") or -1),
        )

    cohort_ranked = sorted(cohort_feature_rows, key=quality_key)
    precision_recall_frontier: list[dict[str, object]] = []
    total_safe = max(1, sum(1 for row in cohort_ranked if str(row.get("fresh_classification_posthoc")) == "safe_plus_one"))
    for prefix in (1, 3, 5, 10, 13, 20, 30, 40, len(cohort_ranked)):
        if prefix <= 0 or prefix > len(cohort_ranked):
            continue
        rows = cohort_ranked[:prefix]
        safe_count = sum(1 for row in rows if str(row.get("fresh_classification_posthoc")) == "safe_plus_one")
        precision_recall_frontier.append(
            {
                "prefix": int(prefix),
                "safe_count": int(safe_count),
                "precision": as_json_float(safe_count / max(1, prefix)),
                "recall": as_json_float(safe_count / total_safe),
                "candidate_ids": [int(row["candidate_id"]) for row in rows],
            }
        )

    generic_lp_started = time.perf_counter()
    generic_feature_rows: list[dict[str, object]] = []
    for candidate in generic_representatives:
        candidate_id = int(candidate_track_id(candidate))
        lp = lp_features(candidate)
        consensus = consensus_features(candidate)
        samples = candidate_samples(candidate)
        deficit = patch_surface_deficit_for_samples(final_candidates, samples, sample_tol=0.01)
        patch_novel = not any(candidates_plane_patch_compatible(candidate, active) for active in final_active_candidates)
        row = {
            "candidate_id": int(candidate_id),
            "candidate": candidate,
            "lp_active": bool(lp.get("active")),
            "lp_effective_margin": lp.get("effective_face_margin"),
            "trial_face_active": None,
            "trial_face_area": None,
            "hull_area": candidate.get("hull_area"),
            "face_hull_area_ratio": None,
            "face_extra_area": None,
            "hull_to_face_distance_p95": None,
            "finite_support_count": candidate.get("finite_support_count"),
            "finite_support_density": candidate.get("finite_support_density"),
            "finite_support_residual_p95": candidate.get("finite_support_residual_p95"),
            "finite_support_purity": candidate.get("finite_support_purity"),
            "finite_support_balance": candidate.get("finite_support_balance"),
            "finite_support_track_coverage": candidate.get("finite_support_track_coverage"),
            "finite_support_z_coverage": candidate.get("finite_support_z_coverage"),
            "plane_rms": candidate.get("plane_rms"),
            "condition_p95": candidate.get("condition_p95"),
            "observed_support_gap_local": candidate.get("observed_support_gap_local"),
            "surface_deficit_reduction": deficit.get("area_weighted_cut_deficit"),
            "local_contour_support_delta": 0.0,
            "patch_novelty": int(patch_novel),
            **consensus,
            "rolling_groups_preserved": True,
            "oracle_face_id_posthoc": (cohort_by_candidate_id.get(candidate_id) or {}).get("face_id"),
            "oracle_classification_posthoc": (cohort_by_candidate_id.get(candidate_id) or {}).get("classification"),
        }
        provisional_flags = {
            "A_activity_local_face": bool(row["lp_active"]),
            "B_patch_novelty": bool(row["patch_novelty"]),
            "C_surface_deficit": finite_float(row["surface_deficit_reduction"], 0.0) > 1e-6,
            "D_consensus_support": bool(
                int(row.get("independent_plane_consensus") or 0) >= 2
                and int(row.get("finite_support_count") or 0) >= 3
                and finite_float(row.get("finite_support_residual_p95"), float("inf")) <= 0.08
                and finite_float(row.get("finite_support_track_coverage"), 0.0) > 0.0
            ),
        }
        row["validator_flags"] = provisional_flags
        row["independent_evidence_family_count"] = int(sum(provisional_flags.values()))
        generic_feature_rows.append(row)
    generic_feature_rows.sort(key=quality_key)
    timings["generic_pool_lp_and_feature_seconds"] = time.perf_counter() - generic_lp_started
    generic_rank_by_id = {int(row["candidate_id"]): rank for rank, row in enumerate(generic_feature_rows, start=1)}
    current_order_candidates = sorted(
        [candidate_by_id[candidate_id] for candidate_id in safe_seed_ids if candidate_id in candidate_by_id],
        key=nonoracle_candidate_quality_key,
    )
    quality_order_candidates = [
        candidate_by_id[int(row["candidate_id"])]
        for row in cohort_ranked
        if int(row["candidate_id"]) in safe_seed_ids and int(row["candidate_id"]) in candidate_by_id
    ]

    def evaluate_state(
        current_candidates: list[dict[str, object]],
        candidate: dict[str, object],
        protected_groups: list[dict[str, object]],
    ) -> dict[str, object]:
        candidate_id = int(candidate_track_id(candidate))
        current_ids = {int(candidate_track_id(item)) for item in current_candidates}
        trial_candidates = list(current_candidates) if candidate_id in current_ids else list(current_candidates) + [candidate]
        rec = reconstruction(trial_candidates)
        trial_active_candidates = active_candidates(trial_candidates, rec)
        trial_active_ids = {int(candidate_track_id(item)) for item in trial_active_candidates}
        trial_canonical = canonical_ids(trial_candidates, rec)
        current_rec = reconstruction(current_candidates)
        current_canonical = canonical_ids(current_candidates, current_rec)
        outside = evaluate_trusted_cloud_outside(
            trial_candidates,
            trusted_points,
            z_slices,
            point_tol=float(point_tol),
        )
        preservation = preservation_summary(protected_groups, trial_active_candidates)
        face_geometry = candidate_face_geometry(trial_candidates, rec, candidate_id)
        accepted_nonoracle = bool(
            candidate_id in trial_active_ids
            and face_geometry.get("active")
            and preservation.get("all_preserved")
            and topology_is_valid(rec.get("topology") if isinstance(rec.get("topology"), dict) else {})
            and finite_float(outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
            and int(value_or_default(outside.get("cumulative_lost_z_levels"), 1)) == 0
        )
        reason = "accepted" if accepted_nonoracle else (
            "candidate_inactive"
            if candidate_id not in trial_active_ids or not face_geometry.get("active")
            else "breaks_rolling_patch"
            if not preservation.get("all_preserved")
            else "invalid_topology"
            if not topology_is_valid(rec.get("topology") if isinstance(rec.get("topology"), dict) else {})
            else "trusted_outside_or_lost_z"
        )
        return {
            "candidate_id": int(candidate_id),
            "accepted_nonoracle": bool(accepted_nonoracle),
            "reason": reason,
            "trial_candidates": trial_candidates,
            "reconstructed": rec,
            "active_candidates": trial_active_candidates,
            "canonical_ids": trial_canonical,
            "canonical_ids_gained": sorted(int(value) for value in (trial_canonical - current_canonical)),
            "canonical_ids_lost": sorted(int(value) for value in (current_canonical - trial_canonical)),
            "canonical_delta": int(len(trial_canonical) - len(current_canonical)),
            "outside": outside,
            "patch_preservation": preservation,
            "face_geometry": face_geometry,
        }

    def collective_sequence(
        name: str,
        ordered: list[dict[str, object]],
        *,
        oracle_greedy: bool,
    ) -> dict[str, object]:
        current = list(final_candidates)
        protected = [dict(group, members=list(group.get("members") or [])) for group in baseline_patch_groups]
        remaining = list(ordered)
        rows: list[dict[str, object]] = []
        first_conflict = None
        while remaining:
            chosen_index = 0
            chosen_trial = None
            if oracle_greedy:
                scored: list[tuple[tuple[object, ...], int, dict[str, object]]] = []
                for index, candidate in enumerate(remaining):
                    trial = evaluate_state(current, candidate, protected)
                    score = (
                        0 if bool(trial.get("accepted_nonoracle")) else 1,
                        -int(trial.get("canonical_delta") or 0),
                        len(trial.get("canonical_ids_lost") or []),
                        nonoracle_candidate_quality_key(candidate),
                    )
                    scored.append((score, index, trial))
                scored.sort(key=lambda value: value[0])
                _, chosen_index, chosen_trial = scored[0]
            candidate = remaining.pop(chosen_index)
            trial = chosen_trial or evaluate_state(current, candidate, protected)
            accepted = bool(trial.get("accepted_nonoracle"))
            if oracle_greedy and trial.get("canonical_ids_lost"):
                accepted = False
                trial["reason"] = "oracle_greedy_rejects_canonical_loss"
            rows.append(
                {
                    "candidate_id": int(candidate_track_id(candidate)),
                    "accepted": bool(accepted),
                    "reason": trial.get("reason"),
                    "canonical_delta_posthoc": int(trial.get("canonical_delta") or 0),
                    "canonical_ids_gained_posthoc": trial.get("canonical_ids_gained"),
                    "canonical_ids_lost_posthoc": trial.get("canonical_ids_lost"),
                    "unique_after_posthoc": int(len(trial.get("canonical_ids") or [])),
                    "protected_group_count_before": int(len(protected)),
                    "patch_preservation": trial.get("patch_preservation"),
                }
            )
            if not accepted:
                if first_conflict is None:
                    first_conflict = rows[-1]
                continue
            current = list(trial["trial_candidates"])
            protected = ratchet_groups(protected, list(trial["active_candidates"]))
        rec = reconstruction(current)
        final_ids = canonical_ids(current, rec)
        outside = evaluate_trusted_cloud_outside(current, trusted_points, z_slices, point_tol=float(point_tol))
        return {
            "name": name,
            "oracle_ordering": bool(oracle_greedy),
            "selection_uses_oracle": bool(oracle_greedy),
            "input_order": [int(candidate_track_id(candidate)) for candidate in ordered],
            "accepted_candidate_ids": [int(row["candidate_id"]) for row in rows if bool(row.get("accepted"))],
            "rows": rows,
            "first_conflict": first_conflict,
            "safe_seed_face_ids_active_posthoc": sorted(
                int(row.get("face_id"))
                for row in safe_seed_rows
                if row.get("face_id") is not None and int(row["face_id"]) in final_ids
            ),
            "retained_core": int(len(core_face_ids & final_ids)),
            "lost_core_ids": sorted(int(value) for value in (core_face_ids - final_ids)),
            "new_ids": sorted(int(value) for value in (final_ids - core_face_ids)),
            "unique": int(len(final_ids)),
            "net_unique_gain": int(len(final_ids) - len(final_canonical_ids)),
            "canonical_ids_gained_vs_129": sorted(int(value) for value in (final_ids - final_canonical_ids)),
            "canonical_ids_lost_vs_129": sorted(int(value) for value in (final_canonical_ids - final_ids)),
            "vertices": int(len(rec.get("vertices") or [])),
            "edges": int(len(rec.get("edges") or [])),
            "faces": int(len(rec.get("faces") or [])),
            "volume": rec.get("reliable_volume"),
            "topology": rec.get("topology"),
            "trusted_outside": outside,
            "rolling_protected_group_count": int(len(protected)),
            "geometry_signature": mesh_geometry_signature(current, rec),
        }

    collective_started = time.perf_counter()
    collective_existing = collective_sequence(
        "A_existing_support_diverse_order",
        current_order_candidates,
        oracle_greedy=False,
    )
    collective_quality = collective_sequence(
        "B_best_quality_nonoracle_order",
        quality_order_candidates,
        oracle_greedy=False,
    )
    collective_oracle = collective_sequence(
        "C_bounded_oracle_greedy_lower_bound",
        current_order_candidates,
        oracle_greedy=True,
    )
    timings["collective_sequences_seconds"] = time.perf_counter() - collective_started

    safe_baseline_rate = sum(
        1 for row in cohort_feature_rows if str(row.get("fresh_classification_posthoc")) == "safe_plus_one"
    ) / max(1, len(cohort_feature_rows))
    best_validator = max(
        validator_summaries.items(),
        key=lambda item: (
            finite_float(item[1].get("safe_precision"), 0.0),
            finite_float(item[1].get("safe_recall"), 0.0),
        ),
    )
    separation_reasonable = bool(
        finite_float(best_validator[1].get("safe_precision"), 0.0) >= max(0.50, safe_baseline_rate + 0.20)
        and finite_float(best_validator[1].get("safe_recall"), 0.0) >= 0.30
    )

    sequential_result: dict[str, object]
    if separation_reasonable:
        sequential_started = time.perf_counter()
        current = list(final_candidates)
        protected = [dict(group, members=list(group.get("members") or [])) for group in baseline_patch_groups]
        decisions: list[dict[str, object]] = []
        accepted_ids: list[int] = []
        full_trials = 0
        for generic_row in generic_feature_rows:
            if len(accepted_ids) >= 13 or full_trials >= 40:
                break
            flags = generic_row.get("validator_flags") if isinstance(generic_row.get("validator_flags"), dict) else {}
            if not bool(generic_row.get("lp_active")):
                continue
            if int(generic_row.get("independent_evidence_family_count") or 0) < 2:
                continue
            if not bool(flags.get("B_patch_novelty")):
                continue
            candidate = generic_row["candidate"]
            current_lp = lp_face_activity_details(
                current,
                candidate,
                inside_tol=0.03,
                face_activity_slack=0.03,
                include_point=False,
            )
            if not bool(current_lp.get("active")):
                decisions.append(
                    {
                        "candidate_id": int(generic_row["candidate_id"]),
                        "accepted": False,
                        "reason": "lp_inactive_in_current_state",
                    }
                )
                continue
            full_trials += 1
            trial = evaluate_state(current, candidate, protected)
            accepted = bool(trial.get("accepted_nonoracle"))
            decisions.append(
                {
                    "candidate_id": int(generic_row["candidate_id"]),
                    "generic_rank": int(generic_rank_by_id[int(generic_row["candidate_id"])]),
                    "accepted": bool(accepted),
                    "reason": trial.get("reason"),
                    "canonical_delta_posthoc": int(trial.get("canonical_delta") or 0),
                    "canonical_ids_gained_posthoc": trial.get("canonical_ids_gained"),
                    "canonical_ids_lost_posthoc": trial.get("canonical_ids_lost"),
                    "unique_after_posthoc": int(len(trial.get("canonical_ids") or [])),
                    "patch_preservation": trial.get("patch_preservation"),
                }
            )
            if not accepted:
                continue
            current = list(trial["trial_candidates"])
            accepted_ids.append(int(generic_row["candidate_id"]))
            protected = ratchet_groups(protected, list(trial["active_candidates"]))
        rec = reconstruction(current)
        result_ids = canonical_ids(current, rec)
        outside = evaluate_trusted_cloud_outside(current, trusted_points, z_slices, point_tol=float(point_tol))
        reprojection = evaluate_polyhedron_reprojection(
            name="generic_preselection_reservoir",
            vertices_3d=np.asarray(rec.get("vertices") or [], dtype=float),
            contours=contours,
            direction_count=32,
            contour_sample=32,
        )
        sequential_result = {
            "executed": True,
            "selector_uses_oracle": False,
            "max_accepted": 13,
            "max_full_trials": 40,
            "full_trials": int(full_trials),
            "accepted_candidate_ids": accepted_ids,
            "decisions": decisions,
            "retained_core": int(len(core_face_ids & result_ids)),
            "lost_core_ids": sorted(int(value) for value in (core_face_ids - result_ids)),
            "new_ids": sorted(int(value) for value in (result_ids - core_face_ids)),
            "unique": int(len(result_ids)),
            "net_unique_gain": int(len(result_ids) - len(final_canonical_ids)),
            "safe_seed_ids_gained_posthoc": sorted(
                int(row["face_id"])
                for row in safe_seed_rows
                if row.get("face_id") is not None
                and int(row["face_id"]) in result_ids - final_canonical_ids
            ),
            "other_ids_gained_posthoc": sorted(
                int(value)
                for value in (result_ids - final_canonical_ids)
                if int(value) not in {int(row.get("face_id") or -1) for row in safe_seed_rows}
            ),
            "canonical_ids_lost_posthoc": sorted(int(value) for value in (final_canonical_ids - result_ids)),
            "vertices": int(len(rec.get("vertices") or [])),
            "edges": int(len(rec.get("edges") or [])),
            "faces": int(len(rec.get("faces") or [])),
            "volume": rec.get("reliable_volume"),
            "topology": rec.get("topology"),
            "trusted_outside": outside,
            "rolling_protected_group_count": int(len(protected)),
            "reprojection": reprojection,
            "reprojection_delta": {
                key: as_json_float(finite_float(reprojection.get(key), 0.0) - finite_float(baseline_reprojection.get(key), 0.0))
                for key in (
                    "support_abs_median",
                    "support_abs_p95",
                    "symmetric_contour_distance_median",
                    "symmetric_contour_distance_p95",
                )
            },
            "geometry_signature": mesh_geometry_signature(current, rec),
        }
        timings["generic_sequential_seconds"] = time.perf_counter() - sequential_started
    else:
        sequential_result = {
            "executed": False,
            "reason": "no simple non-oracle validator reached the predeclared precision/recall separation gate",
            "separation_gate": {
                "minimum_precision": as_json_float(max(0.50, safe_baseline_rate + 0.20)),
                "minimum_recall": 0.30,
                "best_validator": best_validator[0],
                "best_validator_summary": best_validator[1],
            },
        }

    fresh_mismatches = [row for row in fresh_safe_rows if not bool(row.get("reproduced"))]
    collective_best_gain = max(
        int(collective_existing.get("net_unique_gain") or 0),
        int(collective_quality.get("net_unique_gain") or 0),
        int(collective_oracle.get("net_unique_gain") or 0),
    )
    generic_gain = int(sequential_result.get("net_unique_gain") or 0) if sequential_result.get("executed") else 0
    generic_safe = bool(
        sequential_result.get("executed")
        and generic_gain >= 5
        and int(sequential_result.get("retained_core") or 0) == len(core_face_ids)
        and not sequential_result.get("canonical_ids_lost_posthoc")
        and topology_is_valid(sequential_result.get("topology") if isinstance(sequential_result.get("topology"), dict) else {})
        and finite_float((sequential_result.get("trusted_outside") or {}).get("cumulative_outside_fraction"), float("inf")) < 0.01
        and int(value_or_default((sequential_result.get("trusted_outside") or {}).get("cumulative_lost_z_levels"), 1)) == 0
    )
    if fresh_mismatches:
        branch = "E_old_safe_ceiling_not_reproduced"
    elif generic_safe:
        branch = "A_generic_protected_refill_promising"
    elif collective_best_gain < 5:
        branch = "C_individual_safety_not_collectively_transferable"
    elif not separation_reasonable or generic_gain < 5:
        branch = "B_collective_ceiling_without_nonoracle_separation"
    else:
        branch = "D_false_active_planes_require_exchange"

    safe_history_rows = []
    for seed in sorted(safe_seed_rows, key=lambda row: int(row.get("face_id") or -1)):
        candidate_id = int(seed.get("representative_candidate_id") or -1)
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            safe_history_rows.append(
                {
                    "oracle_face_id_posthoc": seed.get("face_id"),
                    "candidate_id": candidate_id,
                    "missing_candidate": True,
                }
            )
        else:
            row = stage_history(candidate, int(seed.get("face_id")) if seed.get("face_id") is not None else None)
            row["generic_pool_rank"] = generic_rank_by_id.get(candidate_id)
            row["current_support_diverse_quality_rank_within_51"] = next(
                (index for index, value in enumerate(sorted(cohort_feature_rows, key=lambda item: nonoracle_candidate_quality_key(candidate_by_id[int(item["candidate_id"])])), start=1) if int(value["candidate_id"]) == candidate_id),
                None,
            )
            row["best_quality_rank_within_51"] = next(
                (index for index, value in enumerate(cohort_ranked, start=1) if int(value["candidate_id"]) == candidate_id),
                None,
            )
            safe_history_rows.append(row)

    public_generic_rows = []
    for rank, row in enumerate(generic_feature_rows, start=1):
        public = {key: value for key, value in row.items() if key != "candidate"}
        public["generic_rank"] = int(rank)
        public_generic_rows.append(public)

    timings["total_seconds"] = time.perf_counter() - started
    golden_geometry = mesh_geometry_signature(final_candidates, final_reconstructed)
    cached_golden = cached_loss_funnel.get("golden_confirmation") if isinstance(cached_loss_funnel.get("golden_confirmation"), dict) else {}
    return {
        "diagnostic_only": True,
        "production_changed": False,
        "uses_initial_model": True,
        "oracle_usage": {
            "cohort_seed": "cached 51 representative oracle classifications",
            "individual_and_collective_evaluation": "canonical posthoc only",
            "generic_pool_formation": "no InitialModel, canonical ID, or oracle label",
            "generic_ranking": "no InitialModel, canonical ID, or oracle label",
            "generic_sequential_acceptance": "LP, full edge-clip, rolling patch, trusted cloud, and topology only",
        },
        "golden_confirmation": {
            "retained": int(len(core_face_ids & final_canonical_ids)),
            "new": int(len(final_canonical_ids - core_face_ids)),
            "unique": int(len(final_canonical_ids)),
            "lost_core_ids": sorted(int(value) for value in (core_face_ids - final_canonical_ids)),
            "vertices": int(len(final_reconstructed.get("vertices") or [])),
            "edges": int(len(final_reconstructed.get("edges") or [])),
            "faces": int(len(final_reconstructed.get("faces") or [])),
            "volume": final_reconstructed.get("reliable_volume"),
            "topology": final_reconstructed.get("topology"),
            "trusted_outside": final_outside,
            "geometry_signature": golden_geometry,
            "matches_cached_129": bool(
                len(final_canonical_ids) == 129
                and int(len(final_reconstructed.get("vertices") or [])) == int(cached_golden.get("vertices") or 616)
                and int(len(final_reconstructed.get("edges") or [])) == int(cached_golden.get("edges") or 924)
                and int(len(final_reconstructed.get("faces") or [])) == int(cached_golden.get("faces") or 310)
                and abs(finite_float(final_reconstructed.get("reliable_volume"), 0.0) - 88.83279097892814) <= 1e-9
            ),
        },
        "cohort": {
            "cached_representative_count": int(len(cohort_rows_cached)),
            "cached_class_counts": (cohort_container or {}).get("class_counts") if isinstance(cohort_container, dict) else None,
            "safe_plus_one_seed_count": int(len(safe_seed_rows)),
            "safe_face_ids_posthoc": sorted(int(row["face_id"]) for row in safe_seed_rows if row.get("face_id") is not None),
            "safe_representative_history": safe_history_rows,
        },
        "fresh_individual_controls": {
            "rows": fresh_safe_rows,
            "reproduced_count": int(sum(1 for row in fresh_safe_rows if bool(row.get("reproduced")))),
            "mismatch_count": int(len(fresh_mismatches)),
            "mismatches": fresh_mismatches,
        },
        "collective_lower_bounds": {
            "A_existing_nonoracle_order": collective_existing,
            "B_best_quality_nonoracle_order": collective_quality,
            "C_bounded_oracle_order": collective_oracle,
            "achieved_full_mesh_lower_bound": int(max(
                int(collective_existing.get("unique") or 0),
                int(collective_quality.get("unique") or 0),
                int(collective_oracle.get("unique") or 0),
            )),
        },
        "generic_pool": {
            "definition": [
                "existing formed valley/W2 candidate",
                "not selected in final129",
                "not a removed blocker",
                "not an active immutable-core plane",
                "valid plane and local hull",
                "has observed support",
            ],
            "small_face_caps_are_features_not_gates": True,
            "funnel": pool_funnel,
            "safe_representative_ranks": [
                {
                    "oracle_face_id_posthoc": row.get("oracle_face_id_posthoc"),
                    "candidate_id": int(row["candidate_id"]),
                    "generic_rank": generic_rank_by_id.get(int(row["candidate_id"])),
                }
                for row in cohort_feature_rows
                if str(row.get("fresh_classification_posthoc")) == "safe_plus_one"
            ],
            "ranking_top40": public_generic_rows[:40],
        },
        "safe_vs_controls": {
            "feature_directions_fixed_before_labels": feature_directions,
            "feature_separation": feature_separation,
            "validator_summaries": validator_summaries,
            "safe_baseline_rate": as_json_float(safe_baseline_rate),
            "quality_ranking_top51": [
                {key: value for key, value in row.items() if key != "candidate"}
                for row in cohort_ranked
            ],
            "precision_recall_frontier": precision_recall_frontier,
            "separation_reasonable": bool(separation_reasonable),
            "best_validator": {"name": best_validator[0], **best_validator[1]},
        },
        "generic_sequential_diagnostic": sequential_result,
        "stage_geometry": stage_geometry,
        "lp_engine": lp_engine.summary(),
        "reconstruction_cache": {
            "unique_full_edge_clip_states": int(len(reconstruction_cache)),
        },
        "selected_branch": branch,
        "production_mode_added": False,
        "production_cli_changed": False,
        "timings_seconds": {key: as_json_float(value) for key, value in timings.items()},
    }


def diagnose_w2_adjacency_failures(
    *,
    failure_rows: list[dict[str, object]],
    initial_vertices: np.ndarray,
    initial_faces: list[list[int]],
    model_faces: list[dict[str, object]],
    z_levels: np.ndarray,
    w2_line_points: np.ndarray | None,
    w2_raw_candidates: list[dict[str, object]],
    w2_clusters: list[dict[str, object]],
    inside_point: np.ndarray,
    orientation_points: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    min_adjacency_levels: int,
    min_adjacency_fraction: float,
    min_z_span: float,
    max_plane_rms: float,
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    core_face_ids: set[int],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
) -> dict[str, object]:
    """Oracle-only audit of alternate adjacency graphs over fixed W2 clusters."""

    started = time.perf_counter()
    timings: dict[str, float] = {}
    target_rows = [
        row for row in failure_rows
        if str(row.get("root_cause")) == "wrong_edge_pair_or_adjacency"
    ]
    target_face_ids = sorted({int(row["target_face_id"]) for row in target_rows})
    if w2_line_points is None or not w2_clusters or not target_rows:
        return {
            "diagnostic_only": True,
            "uses_initial_model": True,
            "production_selector_changed": False,
            "skipped": True,
            "reason": "W2 cluster arrays or wrong-adjacency cohort are unavailable",
            "target_face_ids": target_face_ids,
        }

    cluster_by_id = {
        int(cluster["edge_cluster_id"]): cluster
        for cluster in w2_clusters
        if cluster.get("edge_cluster_id") is not None
    }
    face_by_id = {int(face["face_id"]): face for face in model_faces}
    edges, edge_face_lookup = model_edges_from_faces(initial_vertices, initial_faces)
    edge_id_by_vertices = {tuple(sorted(edge)): int(edge_id) for edge_id, edge in enumerate(edges)}
    face_boundary_edges: dict[int, list[int]] = {}
    edge_incident_faces: dict[int, list[int]] = {}
    for edge_id, edge in enumerate(edges):
        edge_incident_faces[int(edge_id)] = sorted(
            int(value) for value in edge_face_lookup.get(tuple(sorted(edge)), set())
        )
    for face_id, face_indices in enumerate(initial_faces):
        boundary: list[int] = []
        for index, vertex_id in enumerate(face_indices):
            pair = tuple(sorted((int(vertex_id), int(face_indices[(index + 1) % len(face_indices)]))))
            edge_id = edge_id_by_vertices.get(pair)
            if edge_id is not None:
                boundary.append(int(edge_id))
        face_boundary_edges[int(face_id)] = boundary

    oracle_started = time.perf_counter()
    edge_oracle = oracle_edge_cluster_matches(w2_clusters, initial_vertices, initial_faces)
    cluster_oracle = edge_match_lookup(edge_oracle)
    timings["oracle_cluster_matching_seconds"] = time.perf_counter() - oracle_started

    n_half = int(w2_line_points.shape[0])
    z_step = float(np.median(np.diff(z_levels))) if z_levels.size >= 2 else 1.0

    def cluster_center_index(cluster: dict[str, object]) -> float:
        values = [float(value) for value in (cluster.get("member_cyclic_indices") or [cluster.get("cyclic_index") or 0])]
        return float(np.mean(values)) if values else 0.0

    cluster_center = {cid: cluster_center_index(cluster) for cid, cluster in cluster_by_id.items()}
    active_by_level: dict[int, list[int]] = {}
    for zi, z_value in enumerate(z_levels):
        active = [
            int(cid)
            for cid, cluster in cluster_by_id.items()
            if finite_float(cluster.get("cluster_z_min"), float("nan")) <= float(z_value)
            <= finite_float(cluster.get("cluster_z_max"), float("nan"))
        ]
        active_by_level[int(zi)] = sorted(active, key=lambda cid: (cluster_center[cid], cid))

    Pair = tuple[int, int]

    def pair_key(a_id: int, b_id: int) -> Pair:
        return (int(min(a_id, b_id)), int(max(a_id, b_id)))

    def add_pair_level(mapping: dict[Pair, list[int]], a_id: int, b_id: int, zi: int) -> None:
        if int(a_id) == int(b_id):
            return
        mapping.setdefault(pair_key(a_id, b_id), []).append(int(zi))

    current_pairs: dict[Pair, list[int]] = {}
    bounded_pairs: dict[Pair, list[int]] = {}
    mutual_pairs: dict[Pair, list[int]] = {}
    duplicate_skip_pairs: dict[Pair, list[int]] = {}
    duplicate_cache: dict[Pair, bool] = {}

    def duplicate_pair(a_id: int, b_id: int) -> bool:
        key = pair_key(a_id, b_id)
        if key not in duplicate_cache:
            duplicate_cache[key] = bool(
                is_overlapping_collinear_duplicate_pair(cluster_by_id[key[0]], cluster_by_id[key[1]])
            )
        return bool(duplicate_cache[key])

    graph_started = time.perf_counter()
    bounded_k = 4
    for zi, rows in active_by_level.items():
        count = len(rows)
        if count < 2:
            continue
        for index, a_id in enumerate(rows):
            b_id = rows[(index + 1) % count]
            add_pair_level(current_pairs, a_id, b_id, zi)
            add_pair_level(duplicate_skip_pairs, a_id, b_id, zi)
            for gap in range(1, min(bounded_k, count - 1) + 1):
                add_pair_level(bounded_pairs, a_id, rows[(index + gap) % count], zi)
            for gap in range(2, min(bounded_k, count - 1) + 1):
                b_id = rows[(index + gap) % count]
                between = [rows[(index + step) % count] for step in range(1, gap)]
                if between and all(duplicate_pair(mid, a_id) or duplicate_pair(mid, b_id) for mid in between):
                    add_pair_level(duplicate_skip_pairs, a_id, b_id, zi)
        points = np.array(
            [predict_w2_line(cluster_by_id[cid], np.array([z_levels[zi]], dtype=float))[0, :2] for cid in rows],
            dtype=float,
        )
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distances, float("inf"))
        nearest = np.argmin(distances, axis=1)
        for index, other in enumerate(nearest):
            if int(nearest[int(other)]) == int(index):
                add_pair_level(mutual_pairs, rows[index], rows[int(other)], zi)
    for mapping in (current_pairs, bounded_pairs, mutual_pairs, duplicate_skip_pairs):
        for pair in list(mapping):
            mapping[pair] = sorted(set(int(value) for value in mapping[pair]))
    timings["adjacency_graph_construction_seconds"] = time.perf_counter() - graph_started

    def all_common_levels(pair: Pair) -> list[int]:
        a = cluster_by_id[pair[0]]
        b = cluster_by_id[pair[1]]
        z0 = max(
            finite_float(a.get("cluster_z_min"), float("inf")),
            finite_float(b.get("cluster_z_min"), float("inf")),
        )
        z1 = min(
            finite_float(a.get("cluster_z_max"), -float("inf")),
            finite_float(b.get("cluster_z_max"), -float("inf")),
        )
        if not np.isfinite(z0) or not np.isfinite(z1) or z1 < z0:
            return []
        return [int(zi) for zi, value in enumerate(z_levels) if z0 <= float(value) <= z1]

    def circular_between_indices(start: int, end: int) -> list[int]:
        forward = [(start + step) % n_half for step in range(1, (end - start) % n_half)]
        backward = [(start - step) % n_half for step in range(1, (start - end) % n_half)]
        return forward if len(forward) <= len(backward) else backward

    feature_cache: dict[tuple[Pair, tuple[int, ...]], dict[str, object]] = {}

    def pair_features(pair: Pair, levels: list[int]) -> dict[str, object]:
        key = (pair, tuple(sorted(set(int(value) for value in levels))))
        if key in feature_cache:
            return feature_cache[key]
        a = cluster_by_id[pair[0]]
        b = cluster_by_id[pair[1]]
        ordered = list(key[1])
        common = all_common_levels(pair)
        z_values = z_levels[np.array(ordered, dtype=int)] if ordered else np.zeros(0, dtype=float)
        pa = predict_w2_line(a, z_values) if z_values.size else np.zeros((0, 3), dtype=float)
        pb = predict_w2_line(b, z_values) if z_values.size else np.zeros((0, 3), dtype=float)
        separation = np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1) if z_values.size else np.zeros(0, dtype=float)
        separation_median = float(np.median(separation)) if separation.size else float("inf")
        separation_p95 = float(np.percentile(separation, 95)) if separation.size else float("inf")
        separation_delta = np.abs(np.diff(separation)) if separation.size >= 2 else np.zeros(0, dtype=float)
        separation_smoothness = (
            float(np.percentile(separation_delta, 95)) / max(separation_median, 0.01)
            if separation_delta.size else 0.0
        )
        radial_a = pa[:, :2] - np.asarray(inside_point[:2], dtype=float)[None, :] if pa.size else np.zeros((0, 2))
        radial_b = pb[:, :2] - np.asarray(inside_point[:2], dtype=float)[None, :] if pb.size else np.zeros((0, 2))
        signed_order = radial_a[:, 0] * radial_b[:, 1] - radial_a[:, 1] * radial_b[:, 0]
        signed_order = signed_order[np.abs(signed_order) > 1e-9]
        order_flips = int(np.sum(np.sign(signed_order[1:]) != np.sign(signed_order[:-1]))) if signed_order.size >= 2 else 0
        order_stable_fraction = (
            float(max(np.sum(signed_order > 0.0), np.sum(signed_order < 0.0)) / signed_order.size)
            if signed_order.size else 0.0
        )
        direction_a = np.array(a.get("direction") or [], dtype=float)
        direction_b = np.array(b.get("direction") or [], dtype=float)
        direction_angle = 180.0
        line_cross_norm = 0.0
        if direction_a.shape == (3,) and direction_b.shape == (3,):
            direction_angle = float(
                np.degrees(np.arccos(np.clip(abs(float(direction_a @ direction_b)), -1.0, 1.0)))
            )
            line_cross_norm = float(np.linalg.norm(np.cross(direction_a, direction_b)))
        points = np.vstack([pa, pb]) if z_values.size else np.zeros((0, 3), dtype=float)
        plane = fit_plane(points) if points.shape[0] >= 3 else None
        plane_stability_angles: list[float] = []
        plane_stability_offsets: list[float] = []
        if plane is not None and len(ordered) >= 6:
            for chunk in np.array_split(np.array(ordered, dtype=int), 3):
                if chunk.size < 2:
                    continue
                chunk_z = z_levels[chunk]
                chunk_points = np.vstack([predict_w2_line(a, chunk_z), predict_w2_line(b, chunk_z)])
                chunk_plane = fit_plane(chunk_points)
                if chunk_plane is None:
                    continue
                dot = float(plane.normal @ chunk_plane.normal)
                sign = 1.0 if dot >= 0.0 else -1.0
                plane_stability_angles.append(
                    float(np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0))))
                )
                plane_stability_offsets.append(
                    abs(float(plane.normal @ plane.centroid) - sign * float(chunk_plane.normal @ chunk_plane.centroid))
                )
        line_derived_residual_p95 = float("inf")
        if points.shape[0] >= 3 and line_cross_norm > 1e-6:
            line_normal = np.cross(direction_a, direction_b) / line_cross_norm
            line_center = np.mean(points, axis=0)
            line_derived_residual_p95 = float(np.percentile(np.abs((points - line_center) @ line_normal), 95))

        sampled_levels = ordered
        if len(sampled_levels) > 11:
            sampled_levels = [sampled_levels[int(index)] for index in np.linspace(0, len(sampled_levels) - 1, 11, dtype=int)]
        between_counts: list[int] = []
        duplicate_between_counts: list[int] = []
        strip_covered = 0
        center_a = int(round(cluster_center[pair[0]])) % n_half
        center_b = int(round(cluster_center[pair[1]])) % n_half
        cyclic_gap = min((center_b - center_a) % n_half, (center_a - center_b) % n_half)
        arc_indices = circular_between_indices(center_a, center_b)
        for zi in sampled_levels:
            rows = active_by_level.get(int(zi), [])
            if pair[0] in rows and pair[1] in rows:
                ia = rows.index(pair[0])
                ib = rows.index(pair[1])
                forward = [rows[(ia + step) % len(rows)] for step in range(1, (ib - ia) % len(rows))]
                backward = [rows[(ia - step) % len(rows)] for step in range(1, (ia - ib) % len(rows))]
                between = forward if len(forward) <= len(backward) else backward
                between_counts.append(int(len(between)))
                duplicate_between_counts.append(
                    int(sum(duplicate_pair(mid, pair[0]) or duplicate_pair(mid, pair[1]) for mid in between))
                )
            if not arc_indices:
                continue
            observed = np.asarray(w2_line_points[np.array(arc_indices, dtype=int), int(zi), :2], dtype=float)
            observed = observed[np.all(np.isfinite(observed), axis=1)]
            if observed.shape[0] == 0:
                continue
            p0 = predict_w2_line(a, np.array([z_levels[int(zi)]], dtype=float))[0, :2]
            p1 = predict_w2_line(b, np.array([z_levels[int(zi)]], dtype=float))[0, :2]
            segment = p1 - p0
            denom = float(segment @ segment)
            if denom <= EPS:
                continue
            t = np.clip(((observed - p0) @ segment) / denom, 0.0, 1.0)
            distance = np.linalg.norm(observed - (p0 + t[:, None] * segment[None, :]), axis=1)
            width = max(0.025, min(0.15, 0.25 * float(np.linalg.norm(segment))))
            if np.any(distance <= width):
                strip_covered += 1
        strip_fraction = float(strip_covered / max(1, len(sampled_levels)))
        features = {
            "cyclic_gap": int(cyclic_gap),
            "clusters_between_median": as_json_float(float(np.median(between_counts))) if between_counts else None,
            "duplicate_hypotheses_between_median": as_json_float(float(np.median(duplicate_between_counts))) if duplicate_between_counts else 0.0,
            "common_z_level_count": int(len(common)),
            "common_z_coverage": as_json_float(float(len(ordered) / max(1, len(common)))),
            "z_span": as_json_float(float(np.ptp(z_values))) if z_values.size else 0.0,
            "track_separation_median": as_json_float(separation_median),
            "track_separation_p95": as_json_float(separation_p95),
            "separation_smoothness": as_json_float(separation_smoothness),
            "order_flip_count": int(order_flips),
            "order_stable_fraction": as_json_float(order_stable_fraction),
            "direction_angle_deg": as_json_float(direction_angle),
            "line_residual_median": as_json_float(
                float(np.median([finite_float(a.get("line_rms"), float("inf")), finite_float(b.get("line_rms"), float("inf"))]))
            ),
            "pair_plane_rms": as_json_float(float(plane.rms)) if plane is not None else None,
            "plane_normal_stability_deg": as_json_float(max(plane_stability_angles)) if plane_stability_angles else None,
            "plane_offset_stability": as_json_float(max(plane_stability_offsets)) if plane_stability_offsets else None,
            "local_observed_strip_support": as_json_float(strip_fraction),
            "spatial_mutuality": as_json_float(
                float(len(set(ordered) & set(mutual_pairs.get(pair, []))) / max(1, len(ordered)))
            ),
            "duplicate_skip_confidence": as_json_float(
                float(len(set(ordered) & set(duplicate_skip_pairs.get(pair, []))) / max(1, len(ordered)))
            ),
            "line_cross_norm": as_json_float(line_cross_norm),
            "line_derived_residual_p95": as_json_float(line_derived_residual_p95),
            "cluster_member_count_min": int(min(int(a.get("member_count") or 1), int(b.get("member_count") or 1))),
            "cluster_cross_index_support_min": int(
                min(int(a.get("cross_index_support") or 1), int(b.get("cross_index_support") or 1))
            ),
            "cluster_confidence_mean": as_json_float(
                float(np.mean([finite_float(a.get("cluster_confidence"), 0.0), finite_float(b.get("cluster_confidence"), 0.0)]))
            ),
        }
        feature_cache[key] = features
        return features

    def target_metrics(candidate: dict[str, object], face_id: int) -> dict[str, object]:
        plane = candidate_plane(candidate)
        face = face_by_id.get(int(face_id))
        if plane is None or face is None:
            return {"face_id": int(face_id), "finite_good": False, "failure_reasons": ["invalid_candidate"]}
        point, normal = plane
        face_normal = np.array(face["normal"], dtype=float)
        dot = float(normal @ face_normal)
        sign = 1.0 if dot >= 0.0 else -1.0
        angle = float(np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0))))
        plane_distance = abs(float(normal @ point) - sign * float(face["offset"]))
        centroid_distance = point_to_face_polygon_distance(point, face)
        hull = np.array(candidate.get("hull") or [point], dtype=float).reshape((-1, 3))
        hull_distance = float(np.median([point_to_face_polygon_distance(value, face) for value in hull]))
        failures: list[str] = []
        if angle > CANONICAL_ORACLE_TOLERANCES["normal_angle_deg"]:
            failures.append("normal_angle")
        if plane_distance > CANONICAL_ORACLE_TOLERANCES["plane_distance"]:
            failures.append("plane_distance")
        if centroid_distance > CANONICAL_ORACLE_TOLERANCES["centroid_to_finite_polygon_distance"]:
            failures.append("centroid_to_finite_polygon_distance")
        if hull_distance > CANONICAL_ORACLE_TOLERANCES["median_hull_to_surface_distance"]:
            failures.append("median_hull_to_surface_distance")
        return {
            "face_id": int(face_id),
            "normal_angle_deg": as_json_float(angle),
            "plane_distance": as_json_float(plane_distance),
            "centroid_distance": as_json_float(centroid_distance),
            "hull_surface_distance": as_json_float(hull_distance),
            "plane_good": bool(not any(value in failures for value in ("normal_angle", "plane_distance"))),
            "finite_good": bool(not failures),
            "failure_reasons": failures,
        }

    def build_pair_candidate(
        pair: Pair,
        levels: list[int],
        *,
        track_id: int,
        fit_mode: str,
    ) -> tuple[dict[str, object] | None, str | None]:
        ordered = sorted(set(int(value) for value in levels if 0 <= int(value) < len(z_levels)))
        if len(ordered) < int(min_adjacency_levels):
            return None, "insufficient_shared_z"
        z = z_levels[np.array(ordered, dtype=int)]
        z_span = float(np.ptp(z)) if z.size else 0.0
        if z_span < float(min_z_span):
            return None, "insufficient_shared_z"
        a = cluster_by_id[pair[0]]
        b = cluster_by_id[pair[1]]
        overlap = min(
            finite_float(a.get("cluster_z_max"), -float("inf")),
            finite_float(b.get("cluster_z_max"), -float("inf")),
        ) - max(
            finite_float(a.get("cluster_z_min"), float("inf")),
            finite_float(b.get("cluster_z_min"), float("inf")),
        )
        denominator = max(float(overlap) / max(z_step, EPS), 1.0) if np.isfinite(overlap) else float(len(ordered))
        adjacency_fraction = float(len(ordered)) / max(denominator, 1.0)
        if adjacency_fraction < float(min_adjacency_fraction):
            return None, "insufficient_shared_z"
        pa = predict_w2_line(a, z)
        pb = predict_w2_line(b, z)
        points = np.vstack([pa, pb])
        if not (
            np.all(np.isfinite(points))
            and np.all(points >= bounds_min[None, :])
            and np.all(points <= bounds_max[None, :])
        ):
            return None, "outside_observed_bounds"
        plane = fit_plane(points)
        if fit_mode == "line-derived":
            direction_a = np.array(a.get("direction") or [], dtype=float)
            direction_b = np.array(b.get("direction") or [], dtype=float)
            if direction_a.shape != (3,) or direction_b.shape != (3,):
                return None, "line_derived_degenerate"
            normal = np.cross(direction_a, direction_b)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-6:
                return None, "line_derived_degenerate"
            normal /= norm
            centroid = np.mean(points, axis=0)
            signed = (points - centroid) @ normal
            plane = PlaneFit(
                centroid=centroid,
                normal=normal,
                rms=float(np.sqrt(np.mean(signed * signed))),
                max_abs=float(np.max(np.abs(signed))),
            )
        if plane is None or float(plane.rms) > float(max_plane_rms):
            return None, "poor_plane_fit"
        normal, orientation = orient_plane_from_observed_points(plane, orientation_points, inside_point)
        signed = (points - plane.centroid) @ normal
        projected = points - signed[:, None] * normal[None, :] if fit_mode == "line-derived" else points
        u, v = plane_basis(normal)
        coordinates = np.column_stack([(projected - plane.centroid) @ u, (projected - plane.centroid) @ v])
        hull_indices = convex_hull_indices(coordinates)
        if len(hull_indices) < 3:
            return None, "nonfinite_hull"
        hull = projected[np.array(hull_indices, dtype=int)]
        diameter = float(np.max(np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2)))
        return {
            "track_id": int(track_id),
            "source_track_id": None,
            "window": 2,
            "levels": int(len(ordered)),
            "z_min": as_json_float(float(np.min(z))),
            "z_max": as_json_float(float(np.max(z))),
            "center_mean": as_json_float(float(np.mean([cluster_center[pair[0]], cluster_center[pair[1]]]))),
            "fit_points": int(points.shape[0]),
            "candidate_source": "w2_edge_adjacency_diagnostic",
            "candidate_origin": "w2_addition",
            "diagnostic_fit_mode": str(fit_mode),
            "w2_edge_cluster_ids": [int(pair[0]), int(pair[1])],
            "w2_adjacency_levels": int(len(ordered)),
            "w2_adjacency_fraction": as_json_float(adjacency_fraction),
            "w2_edge_confidence": as_json_float(
                float(np.mean([finite_float(a.get("cluster_confidence"), 0.0), finite_float(b.get("cluster_confidence"), 0.0)]))
            ),
            "plane_centroid": as_json_point(plane.centroid),
            "plane_normal": as_json_point(normal),
            "plane_rms": as_json_float(float(plane.rms)),
            "plane_max_abs": as_json_float(float(plane.max_abs)),
            "orientation_positive_frac": as_json_float(float(orientation["orientation_positive_frac"])),
            "orientation_point_count": int(orientation["orientation_point_count"]),
            "candidate_score": as_json_float(float(plane.rms) / max(float(len(ordered)) ** 0.5, 1.0)),
            "hull_diameter": as_json_float(diameter),
            "hull_area": as_json_float(float(polygon_area(hull))),
            "hull": [as_json_point(point) for point in hull],
            "sample_points": [as_json_point(point) for point in points[:: max(1, points.shape[0] // 80)]],
            "z_indices": ordered,
            "low_source_counts": {"w2_edge": int(2 * len(ordered))},
            "low_valley_frac": 0.0,
        }, None

    def nonoracle_geometry_signature(candidate: dict[str, object]) -> tuple[object, ...]:
        plane = candidate_plane(candidate)
        if plane is None:
            return (int(candidate_track_id(candidate)),)
        point, normal = plane
        normal = normal.copy()
        first = next((value for value in normal if abs(float(value)) > 1e-10), 1.0)
        if float(first) < 0.0:
            normal *= -1.0
        offset = float(normal @ point)
        hull = np.array(candidate.get("hull") or [point], dtype=float).reshape((-1, 3))
        centroid = np.mean(hull, axis=0)
        return (
            *(round(float(value), 4) for value in normal),
            round(offset, 3),
            *(round(float(value), 2) for value in centroid),
        )

    def nonoracle_pair_score(features: dict[str, object], candidate: dict[str, object]) -> float:
        return float(
            finite_float(candidate.get("plane_rms"), 1.0)
            + 0.02 * finite_float(features.get("separation_smoothness"), 10.0)
            + 0.02 * (1.0 - finite_float(features.get("common_z_coverage"), 0.0))
            + 0.02 * (1.0 - finite_float(features.get("local_observed_strip_support"), 0.0))
            + 0.002 * finite_float(features.get("plane_normal_stability_deg"), 90.0)
            + 0.05 * float(int(features.get("order_flip_count") or 0) > 0)
        )

    stable_strip_thresholds = {
        "minimum_order_stable_fraction": 0.90,
        "maximum_order_flips": 0,
        "maximum_separation_smoothness": 0.35,
        "maximum_plane_normal_stability_deg": 10.0,
        "minimum_local_observed_strip_support": 0.40,
    }

    def stable_strip_pass(features: dict[str, object]) -> bool:
        return bool(
            finite_float(features.get("order_stable_fraction"), 0.0) >= 0.90
            and int(features.get("order_flip_count") or 0) == 0
            and finite_float(features.get("separation_smoothness"), float("inf")) <= 0.35
            and finite_float(features.get("plane_normal_stability_deg"), float("inf")) <= 10.0
            and finite_float(features.get("local_observed_strip_support"), 0.0) >= 0.40
        )

    mode_maps: dict[str, dict[Pair, list[int]]] = {
        "A_current_cyclic_immediate": current_pairs,
        "B_bounded_cyclic_k4": bounded_pairs,
        "C_mutual_spatial": mutual_pairs,
        "E_duplicate_skipping": duplicate_skip_pairs,
    }
    stable_source: dict[Pair, list[int]] = {}
    for mapping in (bounded_pairs, mutual_pairs, duplicate_skip_pairs):
        for pair, levels in mapping.items():
            stable_source.setdefault(pair, []).extend(int(value) for value in levels)
    for pair in list(stable_source):
        stable_source[pair] = sorted(set(stable_source[pair]))
    mode_maps["D_stable_strip"] = stable_source

    def bounded_pair_rows(mapping: dict[Pair, list[int]], *, control: bool) -> list[tuple[Pair, list[int]]]:
        rows = sorted(mapping.items(), key=lambda item: (-len(item[1]), item[0]))
        if control:
            return rows
        selected: list[tuple[Pair, list[int]]] = []
        degree: dict[int, int] = {}
        for pair, levels in rows:
            if degree.get(pair[0], 0) >= 12 or degree.get(pair[1], 0) >= 12:
                continue
            selected.append((pair, levels))
            degree[pair[0]] = degree.get(pair[0], 0) + 1
            degree[pair[1]] = degree.get(pair[1], 0) + 1
            if len(selected) >= 6000:
                break
        return selected

    current_match_cache: dict[int, dict[str, object] | None] = {}

    def current_candidate_match(candidate: dict[str, object]) -> dict[str, object] | None:
        candidate_id = int(candidate_track_id(candidate))
        if candidate_id not in current_match_cache:
            current_match_cache[candidate_id] = candidate_oracle_face(candidate, model_faces)
        return current_match_cache[candidate_id]

    current_finite_ids = {
        int(match["face_id"])
        for candidate in w2_raw_candidates
        for match in [current_candidate_match(candidate)]
        if match is not None and bool(match.get("finite_good")) and match.get("face_id") is not None
    }

    finite_current_by_face: dict[int, dict[str, object]] = {}
    for candidate in w2_raw_candidates:
        match = current_candidate_match(candidate)
        if match is None or not bool(match.get("finite_good")) or match.get("face_id") is None:
            continue
        face_id = int(match["face_id"])
        previous = finite_current_by_face.get(face_id)
        if previous is None or finite_float(candidate.get("candidate_score"), float("inf")) < finite_float(previous.get("candidate_score"), float("inf")):
            finite_current_by_face[face_id] = candidate

    def control_distance(candidate: dict[str, object]) -> float:
        levels = max(1.0, float(candidate.get("levels") or 1))
        span = max(1e-4, finite_float(candidate.get("z_max"), 0.0) - finite_float(candidate.get("z_min"), 0.0))
        return min(
            (
                abs(math.log(levels / max(1.0, float(row.get("levels") or 1))))
                + abs(math.log(span / max(1e-4, finite_float(row.get("z_span"), 0.0))))
            )
            for row in target_rows
        )

    control_candidates = sorted(
        finite_current_by_face.values(),
        key=lambda candidate: (control_distance(candidate), int(candidate_track_id(candidate))),
    )[: max(21, len(target_rows))]
    control_pairs = {
        pair_key(*[int(value) for value in candidate.get("w2_edge_cluster_ids", [])]): candidate
        for candidate in control_candidates
        if len(candidate.get("w2_edge_cluster_ids", [])) == 2
    }
    control_face_ids = {
        int(match["face_id"])
        for candidate in control_candidates
        for match in [current_candidate_match(candidate)]
        if match is not None and match.get("face_id") is not None
    }

    mode_started = time.perf_counter()
    mode_internal_rows: dict[str, list[dict[str, object]]] = {}
    mode_summaries: dict[str, dict[str, object]] = {}
    mode_pair_presence: dict[str, set[Pair]] = {}
    mode_candidate_face_ids: dict[str, set[int]] = {}
    mode_index_lookup = {name: index for index, name in enumerate(mode_maps)}
    for mode_name, mapping in mode_maps.items():
        control_mode = mode_name == "A_current_cyclic_immediate"
        bounded_rows = bounded_pair_rows(mapping, control=control_mode)
        funnel = {
            "raw_alternative_pairs": int(len(mapping)),
            "bounded_pairs": int(len(bounded_rows)),
            "shared_z_valid": 0,
            "order_stable": 0,
            "strip_valid": 0,
            "plane_fit_valid": 0,
            "nonoracle_deduped_candidates": 0,
        }
        generated: list[dict[str, object]] = []
        seen_geometry: set[tuple[object, ...]] = set()
        pre_dedupe_pair_presence: set[Pair] = set()
        rejection_counts: dict[str, int] = {}
        for ordinal, (pair, levels) in enumerate(bounded_rows):
            features = pair_features(pair, levels)
            enough = bool(
                len(levels) >= int(min_adjacency_levels)
                and finite_float(features.get("z_span"), 0.0) >= float(min_z_span)
                and finite_float(features.get("common_z_coverage"), 0.0) >= float(min_adjacency_fraction)
            )
            if not enough:
                rejection_counts["insufficient_shared_z"] = rejection_counts.get("insufficient_shared_z", 0) + 1
                continue
            funnel["shared_z_valid"] += 1
            order_ok = bool(
                finite_float(features.get("order_stable_fraction"), 0.0) >= 0.90
                and int(features.get("order_flip_count") or 0) == 0
            )
            strip_ok = finite_float(features.get("local_observed_strip_support"), 0.0) >= 0.40
            funnel["order_stable"] += int(order_ok)
            funnel["strip_valid"] += int(strip_ok)
            if mode_name == "D_stable_strip" and not stable_strip_pass(features):
                rejection_counts["stable_strip_gate"] = rejection_counts.get("stable_strip_gate", 0) + 1
                continue
            candidate, reason = build_pair_candidate(
                pair,
                levels,
                track_id=9700000 + 10000 * mode_index_lookup[mode_name] + ordinal,
                fit_mode="current",
            )
            if candidate is None:
                rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1
                continue
            funnel["plane_fit_valid"] += 1
            pre_dedupe_pair_presence.add(pair)
            signature = nonoracle_geometry_signature(candidate)
            if not control_mode and signature in seen_geometry:
                rejection_counts["nonoracle_geometry_duplicate"] = rejection_counts.get("nonoracle_geometry_duplicate", 0) + 1
                continue
            seen_geometry.add(signature)
            oracle_match = candidate_oracle_face(candidate, model_faces)
            finite_good = bool(oracle_match is not None and oracle_match.get("finite_good"))
            face_id = int(oracle_match["face_id"]) if oracle_match is not None and oracle_match.get("face_id") is not None else None
            generated.append(
                {
                    "pair": pair,
                    "levels": list(levels),
                    "features": features,
                    "candidate": candidate,
                    "oracle_face_id": face_id,
                    "finite_good": finite_good,
                    "nonoracle_score": nonoracle_pair_score(features, candidate),
                }
            )
        funnel["nonoracle_deduped_candidates"] = int(len(generated))
        finite_rows = [row for row in generated if bool(row["finite_good"])]
        finite_ids = {int(row["oracle_face_id"]) for row in finite_rows if row.get("oracle_face_id") is not None}
        recovered_ids = sorted(int(value) for value in finite_ids & set(target_face_ids))
        control_faces_preserved = {
            int(row["oracle_face_id"])
            for row in finite_rows
            if row.get("oracle_face_id") is not None
        } & control_face_ids
        control_pair_present = sum(1 for pair in control_pairs if pair in pre_dedupe_pair_presence)
        mode_internal_rows[mode_name] = generated
        mode_pair_presence[mode_name] = pre_dedupe_pair_presence
        mode_candidate_face_ids[mode_name] = finite_ids
        mode_summaries[mode_name] = {
            "candidate_count": int(len(generated)),
            "finite_good_count": int(len(finite_rows)),
            "precision": as_json_float(float(len(finite_rows) / max(1, len(generated)))),
            "unique_finite_face_ids": int(len(finite_ids)),
            "new_ids_vs_current_w2_adjacency": sorted(int(value) for value in finite_ids - current_finite_ids),
            "recovered_target_count": int(len(recovered_ids)),
            "recovered_target_face_ids": recovered_ids,
            "duplicate_assignments": int(len(finite_rows) - len(finite_ids)),
            "matched_current_good_controls": int(len(control_pairs)),
            "current_good_control_pairs_present": int(control_pair_present),
            "current_good_control_pairs_rejected": int(len(control_pairs) - control_pair_present),
            "current_good_control_faces_preserved": int(len(control_faces_preserved)),
            "current_good_control_faces_lost": sorted(int(value) for value in control_face_ids - control_faces_preserved),
            "wrong_current_pairs_passing": int(
                sum(
                    pair_key(*[int(value) for value in row.get("w2_edge_cluster_ids", [])]) in pre_dedupe_pair_presence
                    for row in target_rows
                    if len(row.get("w2_edge_cluster_ids", [])) == 2
                )
            ),
            "funnel": funnel,
            "rejection_counts": {key: int(value) for key, value in sorted(rejection_counts.items())},
        }
    timings["alternative_mode_evaluation_seconds"] = time.perf_counter() - mode_started

    availability_started = time.perf_counter()
    availability_rows: list[dict[str, object]] = []
    anatomy_rows: list[dict[str, object]] = []
    for target_row in target_rows:
        face_id = int(target_row["target_face_id"])
        boundary_edges = set(face_boundary_edges.get(face_id, []))
        boundary_clusters: dict[int, list[int]] = {edge_id: [] for edge_id in sorted(boundary_edges)}
        for cluster_id, oracle_row in cluster_oracle.items():
            edge_id = oracle_row.get("top_edge_id")
            if bool(oracle_row.get("top_confidence")) and edge_id is not None and int(edge_id) in boundary_edges:
                boundary_clusters[int(edge_id)].append(int(cluster_id))
        populated_edges = [edge_id for edge_id, values in boundary_clusters.items() if values]
        correct_pairs: list[tuple[Pair, int, int]] = []
        for left_index, left_edge in enumerate(populated_edges):
            for right_edge in populated_edges[left_index + 1:]:
                for a_id in boundary_clusters[left_edge]:
                    for b_id in boundary_clusters[right_edge]:
                        correct_pairs.append((pair_key(a_id, b_id), int(left_edge), int(right_edge)))
        correct_pairs = sorted(
            correct_pairs,
            key=lambda item: (
                -len(all_common_levels(item[0])),
                -finite_float(cluster_by_id[item[0][0]].get("cluster_confidence"), 0.0)
                - finite_float(cluster_by_id[item[0][1]].get("cluster_confidence"), 0.0),
                item[0],
            ),
        )
        enough_count = 0
        current_valid_count = 0
        line_valid_count = 0
        correct_pair_details: list[dict[str, object]] = []
        for pair_index, (pair, left_edge, right_edge) in enumerate(correct_pairs[:240]):
            levels = all_common_levels(pair)
            features = pair_features(pair, levels)
            enough = bool(
                len(levels) >= int(min_adjacency_levels)
                and finite_float(features.get("z_span"), 0.0) >= float(min_z_span)
            )
            enough_count += int(enough)
            current_candidate = None
            current_metrics = None
            line_candidate = None
            line_metrics = None
            if enough:
                current_candidate, _ = build_pair_candidate(
                    pair,
                    levels,
                    track_id=9600000 + 1000 * face_id + 2 * pair_index,
                    fit_mode="current",
                )
                line_candidate, _ = build_pair_candidate(
                    pair,
                    levels,
                    track_id=9600001 + 1000 * face_id + 2 * pair_index,
                    fit_mode="line-derived",
                )
                if current_candidate is not None:
                    current_metrics = target_metrics(current_candidate, face_id)
                    current_valid_count += int(bool(current_metrics.get("finite_good")))
                if line_candidate is not None:
                    line_metrics = target_metrics(line_candidate, face_id)
                    line_valid_count += int(bool(line_metrics.get("finite_good")))
            correct_pair_details.append(
                {
                    "cluster_ids": [int(pair[0]), int(pair[1])],
                    "boundary_edge_ids": [int(left_edge), int(right_edge)],
                    "enough_shared_z": bool(enough),
                    "features": features,
                    "current_fit": current_metrics,
                    "line_derived_fit": line_metrics,
                }
            )
        availability = {
            "target_face_id": int(face_id),
            "boundary_edge_ids": sorted(int(value) for value in boundary_edges),
            "clusters_per_boundary_edge": {
                str(edge_id): sorted(int(value) for value in cluster_ids)
                for edge_id, cluster_ids in boundary_clusters.items()
            },
            "boundary_edges_with_clusters": int(len(populated_edges)),
            "correct_boundary_pair_exists": bool(correct_pairs),
            "correct_pair_count": int(len(correct_pairs)),
            "correct_pair_with_enough_shared_z": bool(enough_count > 0),
            "correct_pairs_with_enough_shared_z": int(enough_count),
            "correct_pair_valid_current_plane": bool(current_valid_count > 0),
            "correct_pairs_valid_current_plane": int(current_valid_count),
            "correct_pair_valid_line_derived_plane": bool(line_valid_count > 0),
            "correct_pairs_valid_line_derived_plane": int(line_valid_count),
            "best_correct_pairs": correct_pair_details[:8],
        }
        availability_rows.append(availability)

        current_pair_ids = [int(value) for value in target_row.get("w2_edge_cluster_ids", [])]
        current_cluster_rows: list[dict[str, object]] = []
        current_target_edges: list[int] = []
        current_confident_edges: list[int] = []
        for cluster_id in current_pair_ids:
            oracle_row = cluster_oracle.get(cluster_id, {})
            edge_id = oracle_row.get("top_edge_id")
            if bool(oracle_row.get("top_confidence")) and edge_id is not None:
                current_confident_edges.append(int(edge_id))
                if int(edge_id) in boundary_edges:
                    current_target_edges.append(int(edge_id))
            cluster = cluster_by_id.get(cluster_id, {})
            current_cluster_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "oracle_edge_id": int(edge_id) if edge_id is not None else None,
                    "oracle_confident": bool(oracle_row.get("top_confidence")),
                    "oracle_score_margin": oracle_row.get("score_margin"),
                    "oracle_edge_incident_faces": edge_incident_faces.get(int(edge_id), []) if edge_id is not None else [],
                    "cyclic_indices": [int(value) for value in (cluster.get("member_cyclic_indices") or [])],
                    "z_min": cluster.get("cluster_z_min"),
                    "z_max": cluster.get("cluster_z_max"),
                }
            )
        ancestry = dict(target_row.get("fit_point_ancestry") or {})
        if len(populated_edges) == 0:
            classification = "both_boundary_clusters_missing"
        elif len(populated_edges) == 1:
            classification = "one_boundary_cluster_missing"
        elif len(current_target_edges) == 2 and len(set(current_target_edges)) == 1:
            classification = "duplicate_clusters_same_model_edge"
        elif len(set(current_target_edges)) >= 2:
            classification = "correct_boundary_pair_but_bad_pair_fit"
        elif (
            len(current_confident_edges) == 2
            and not (set(edge_incident_faces.get(current_confident_edges[0], [])) & set(edge_incident_faces.get(current_confident_edges[1], [])))
        ):
            classification = "clusters_match_edges_of_different_faces"
        elif (
            finite_float(ancestry.get("target_face_point_fraction"), 0.0) >= 0.60
            and finite_float(ancestry.get("target_boundary_fraction"), 0.0) < 0.45
        ):
            classification = "correct_face_but_nonboundary_edges"
        elif bool(correct_pairs):
            classification = "correct_pair_exists_but_current_adjacency_missed"
        else:
            classification = "both_clusters_wrong_or_unmatched"
        current_pair = pair_key(*current_pair_ids) if len(current_pair_ids) == 2 else None
        anatomy_rows.append(
            {
                "target_face_id": int(face_id),
                "current_candidate_id": int(target_row["candidate_id"]),
                "classification": classification,
                "current_pair_cluster_ids": current_pair_ids,
                "current_clusters": current_cluster_rows,
                "target_face_boundary_edge_ids": sorted(int(value) for value in boundary_edges),
                "current_pair_features": pair_features(current_pair, list(target_row.get("z_indices") or current_pairs.get(current_pair, []))) if current_pair is not None else None,
                "current_plane": {
                    "centroid": (target_row.get("fit_alternatives", {}).get("current_least_squares", {}).get("candidate_geometry") or {}).get("plane_centroid"),
                    "normal": (target_row.get("fit_alternatives", {}).get("current_least_squares", {}).get("candidate_geometry") or {}).get("plane_normal"),
                    "hull_area": target_row.get("hull_area"),
                    "plane_rms": target_row.get("plane_rms"),
                },
                "availability": {
                    key: value for key, value in availability.items()
                    if key not in {"clusters_per_boundary_edge", "best_correct_pairs"}
                },
            }
        )
    timings["availability_and_anatomy_seconds"] = time.perf_counter() - availability_started

    def count_values(values: list[str]) -> dict[str, int]:
        output: dict[str, int] = {}
        for value in values:
            output[str(value)] = output.get(str(value), 0) + 1
        return {key: int(value) for key, value in sorted(output.items())}

    anatomy_counts = count_values([str(row["classification"]) for row in anatomy_rows])
    anatomy_categories = [
        "both_clusters_wrong_or_unmatched",
        "duplicate_clusters_same_model_edge",
        "clusters_match_edges_of_different_faces",
        "correct_face_but_nonboundary_edges",
        "correct_boundary_pair_but_bad_pair_fit",
        "correct_pair_exists_but_current_adjacency_missed",
        "one_boundary_cluster_missing",
        "both_boundary_clusters_missing",
    ]
    anatomy_counts = {key: int(anatomy_counts.get(key, 0)) for key in anatomy_categories}

    current_wrong_feature_rows = [
        row["current_pair_features"] for row in anatomy_rows if isinstance(row.get("current_pair_features"), dict)
    ]
    control_feature_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for candidate in control_candidates:
        pair_values = [int(value) for value in candidate.get("w2_edge_cluster_ids", [])]
        if len(pair_values) != 2:
            continue
        pair = pair_key(*pair_values)
        levels = list(candidate.get("z_indices") or current_pairs.get(pair, []))
        features = pair_features(pair, levels)
        control_feature_rows.append(features)
        match = current_candidate_match(candidate)
        control_rows.append(
            {
                "candidate_id": int(candidate_track_id(candidate)),
                "face_id": int(match["face_id"]) if match is not None and match.get("face_id") is not None else None,
                "cluster_ids": [int(pair[0]), int(pair[1])],
                "levels": int(candidate.get("levels") or 0),
                "z_span": as_json_float(
                    max(0.0, finite_float(candidate.get("z_max"), 0.0) - finite_float(candidate.get("z_min"), 0.0))
                ),
                "features": features,
            }
        )

    feature_names = [
        "cyclic_gap",
        "clusters_between_median",
        "duplicate_hypotheses_between_median",
        "common_z_level_count",
        "common_z_coverage",
        "track_separation_median",
        "track_separation_p95",
        "separation_smoothness",
        "order_flip_count",
        "order_stable_fraction",
        "direction_angle_deg",
        "line_residual_median",
        "pair_plane_rms",
        "plane_normal_stability_deg",
        "local_observed_strip_support",
        "spatial_mutuality",
        "duplicate_skip_confidence",
        "line_derived_residual_p95",
        "cluster_member_count_min",
        "cluster_cross_index_support_min",
        "cluster_confidence_mean",
    ]
    current_comparison = {
        feature: {
            "wrong": distribution_summary([row.get(feature) for row in current_wrong_feature_rows]),
            "matched_good_controls": distribution_summary([row.get(feature) for row in control_feature_rows]),
        }
        for feature in feature_names
    }

    availability_by_face = {int(row["target_face_id"]): row for row in availability_rows}
    hypotheses = {
        "duplicate_cluster_inserted_between_true_neighbors": int(
            sum(
                finite_float((row.get("current_pair_features") or {}).get("duplicate_hypotheses_between_median"), 0.0) > 0.0
                for row in anatomy_rows
                if bool(availability_by_face[int(row["target_face_id"])] ["correct_boundary_pair_exists"])
            )
        ),
        "immediate_cyclic_neighbor_not_spatial_neighbor": int(
            sum(finite_float((row.get("current_pair_features") or {}).get("spatial_mutuality"), 0.0) < 0.5 for row in anatomy_rows)
        ),
        "cyclic_order_changes_over_z": int(
            sum(int((row.get("current_pair_features") or {}).get("order_flip_count") or 0) > 0 for row in anatomy_rows)
        ),
        "shared_z_too_short": int(
            sum(
                int((row.get("current_pair_features") or {}).get("common_z_level_count") or 0) < int(min_adjacency_levels)
                or finite_float((row.get("current_pair_features") or {}).get("z_span"), 0.0) < float(min_z_span)
                for row in anatomy_rows
            )
        ),
        "one_cluster_mixes_physical_edges": int(
            sum(
                any(
                    not bool(cluster.get("oracle_confident"))
                    or finite_float(cluster.get("oracle_score_margin"), 0.0) < 0.02
                    for cluster in row.get("current_clusters", [])
                )
                for row in anatomy_rows
            )
        ),
    }

    separation_started = time.perf_counter()
    labeled_rows = [row for values in mode_internal_rows.values() for row in values]
    feature_directions = {
        "spatial_mutuality": "higher",
        "separation_smoothness": "lower",
        "common_z_coverage": "higher",
        "line_derived_residual_p95": "lower",
        "local_observed_strip_support": "higher",
        "duplicate_skip_confidence": "higher",
    }

    def auc(rows: list[tuple[float, bool]], direction: str) -> float | None:
        positives = [value for value, label in rows if label and np.isfinite(value)]
        negatives = [value for value, label in rows if not label and np.isfinite(value)]
        if not positives or not negatives:
            return None
        wins = 0.0
        for positive in positives:
            for negative in negatives:
                if positive == negative:
                    wins += 0.5
                elif (direction == "higher" and positive > negative) or (direction == "lower" and positive < negative):
                    wins += 1.0
        return float(wins / (len(positives) * len(negatives)))

    separation: dict[str, object] = {}
    for feature, direction in feature_directions.items():
        values = [
            (finite_float(row["features"].get(feature), float("nan")), bool(row["finite_good"]))
            for row in labeled_rows
        ]
        finite_values = np.array([value for value, _ in values if np.isfinite(value)], dtype=float)
        frontier: list[dict[str, object]] = []
        if finite_values.size:
            for quantile in (0.10, 0.25, 0.50, 0.75, 0.90):
                threshold = float(np.quantile(finite_values, quantile))
                selected = [
                    row for row in labeled_rows
                    if np.isfinite(finite_float(row["features"].get(feature), float("nan")))
                    and (
                        finite_float(row["features"].get(feature), float("nan")) >= threshold
                        if direction == "higher"
                        else finite_float(row["features"].get(feature), float("nan")) <= threshold
                    )
                ]
                selected_good = [row for row in selected if bool(row["finite_good"])]
                selected_targets = {
                    int(row["oracle_face_id"])
                    for row in selected_good
                    if row.get("oracle_face_id") is not None and int(row["oracle_face_id"]) in set(target_face_ids)
                }
                frontier.append(
                    {
                        "quantile": as_json_float(quantile),
                        "threshold": as_json_float(threshold),
                        "selected": int(len(selected)),
                        "precision": as_json_float(float(len(selected_good) / max(1, len(selected)))),
                        "target_recovered": int(len(selected_targets)),
                    }
                )
        feature_auc = auc(values, direction)
        separation[feature] = {
            "direction": direction,
            "auc": as_json_float(feature_auc) if feature_auc is not None else None,
            "finite_good": distribution_summary(
                [value for value, label in values if label and np.isfinite(value)]
            ),
            "wrong": distribution_summary(
                [value for value, label in values if not label and np.isfinite(value)]
            ),
            "threshold_frontier": frontier,
        }
    timings["nonoracle_separation_seconds"] = time.perf_counter() - separation_started

    baseline_active_indices = [
        int(index)
        for index in final_reconstructed.get("face_candidate_indices", [])
        if 0 <= int(index) < len(final_candidates)
    ]
    baseline_active_candidates = [final_candidates[index] for index in baseline_active_indices]
    baseline_finite_ids = {
        int(match["face_id"])
        for candidate in baseline_active_candidates
        for match in [candidate_oracle_face(candidate, model_faces)]
        if match is not None and bool(match.get("finite_good")) and match.get("face_id") is not None
    }

    def strict_patch_compatible(a: dict[str, object], b: dict[str, object]) -> bool:
        angle, offset, centroid_distance = candidate_plane_relation_score(a, b)
        za = candidate_z_intervals(a)
        zb = candidate_z_intervals(b)
        _, hull_fraction = interval_overlap(
            za.get("hull_z_min"), za.get("hull_z_max"), zb.get("hull_z_min"), zb.get("hull_z_max")
        )
        _, support_fraction = interval_overlap(
            za.get("support_z_min"), za.get("support_z_max"), zb.get("support_z_min"), zb.get("support_z_max")
        )
        return bool(
            angle <= 1.0
            and offset <= 0.02
            and centroid_distance <= 0.12
            and max(hull_fraction, support_fraction) >= 0.25
        )

    baseline_groups: list[list[dict[str, object]]] = []
    for candidate in baseline_active_candidates:
        for group in baseline_groups:
            if all(strict_patch_compatible(candidate, member) for member in group):
                group.append(candidate)
                break
        else:
            baseline_groups.append([candidate])

    representatives_by_face: dict[int, dict[str, object]] = {}
    for mode_name, rows in mode_internal_rows.items():
        if mode_name == "A_current_cyclic_immediate":
            continue
        for row in rows:
            face_id = row.get("oracle_face_id")
            if not bool(row.get("finite_good")) or face_id is None or int(face_id) not in set(target_face_ids):
                continue
            previous = representatives_by_face.get(int(face_id))
            if previous is None or (
                finite_float(row.get("nonoracle_score"), float("inf")), mode_name, row["pair"]
            ) < (
                finite_float(previous.get("nonoracle_score"), float("inf")), str(previous["mode"]), previous["pair"]
            ):
                representatives_by_face[int(face_id)] = {**row, "mode": mode_name}
    representatives = sorted(
        representatives_by_face.values(),
        key=lambda row: (finite_float(row.get("nonoracle_score"), float("inf")), str(row["mode"]), row["pair"]),
    )[:12]

    full_started = time.perf_counter()
    full_trials: list[dict[str, object]] = []
    z_slices = [
        np.flatnonzero(trusted_z_indices == int(value))
        for value in sorted(set(int(value) for value in trusted_z_indices))
    ] if trusted_z_indices.size else []
    for trial_index, row in enumerate(representatives):
        alternative = dict(row["candidate"])
        alternative["track_id"] = 9900000 + int(trial_index)
        trial_candidates = list(final_candidates) + [alternative]
        trial_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
            trial_candidates,
            **reconstruction_kwargs,
        )
        trial_active_indices = [
            int(index)
            for index in trial_reconstructed.get("face_candidate_indices", [])
            if 0 <= int(index) < len(trial_candidates)
        ]
        trial_active_candidates = [trial_candidates[index] for index in trial_active_indices]
        trial_ids = {
            int(match["face_id"])
            for candidate in trial_active_candidates
            for match in [candidate_oracle_face(candidate, model_faces)]
            if match is not None and bool(match.get("finite_good")) and match.get("face_id") is not None
        }
        lost_groups: list[list[int]] = []
        for group in baseline_groups:
            if not any(
                strict_patch_compatible(member, trial_candidate)
                for member in group
                for trial_candidate in trial_active_candidates
            ):
                lost_groups.append(sorted(int(candidate_track_id(member)) for member in group))
        outside = evaluate_trusted_cloud_outside(
            trial_candidates,
            trusted_points,
            z_slices,
            point_tol=float(point_tol),
        )
        full_trials.append(
            {
                "candidate_level_target_face_id": int(row["oracle_face_id"]),
                "adjacency_mode": str(row["mode"]),
                "cluster_ids": [int(value) for value in row["pair"]],
                "nonoracle_score": as_json_float(finite_float(row.get("nonoracle_score"), float("inf"))),
                "candidate_active": bool(len(trial_candidates) - 1 in trial_active_indices),
                "final_target_active": bool(int(row["oracle_face_id"]) in trial_ids),
                "canonical_ids_gained": sorted(int(value) for value in trial_ids - baseline_finite_ids),
                "canonical_ids_lost": sorted(int(value) for value in baseline_finite_ids - trial_ids),
                "retained": int(len(trial_ids & set(core_face_ids))),
                "new": int(len(trial_ids - set(core_face_ids))),
                "unique": int(len(trial_ids)),
                "rolling_groups_lost": int(len(lost_groups)),
                "lost_rolling_group_candidate_ids": lost_groups,
                "vertices": int(len(trial_reconstructed.get("vertices", []))),
                "edges": int(len(trial_reconstructed.get("edges", []))),
                "faces": int(len(trial_reconstructed.get("faces", []))),
                "volume": trial_reconstructed.get("reliable_volume"),
                "topology": trial_reconstructed.get("topology"),
                "topology_valid": topology_is_valid(
                    trial_reconstructed.get("topology")
                    if isinstance(trial_reconstructed.get("topology"), dict)
                    else None
                ),
                "outside": outside,
            }
        )
    timings["bounded_full_edge_clip_seconds"] = time.perf_counter() - full_started

    correct_exists = sum(bool(row["correct_boundary_pair_exists"]) for row in availability_rows)
    enough_exists = sum(bool(row["correct_pair_with_enough_shared_z"]) for row in availability_rows)
    current_valid = sum(bool(row["correct_pair_valid_current_plane"]) for row in availability_rows)
    alternative_valid = sum(bool(row["correct_pair_valid_line_derived_plane"]) for row in availability_rows)
    best_mode_name = max(
        mode_summaries,
        key=lambda name: (
            int(mode_summaries[name]["recovered_target_count"]),
            -int(mode_summaries[name]["current_good_control_pairs_rejected"]),
            -int(mode_summaries[name]["candidate_count"]),
            name,
        ),
    )
    best_mode = mode_summaries[best_mode_name]
    confirmed_plus_one = sum(
        bool(row.get("canonical_ids_gained"))
        and not bool(row.get("canonical_ids_lost"))
        and int(row.get("rolling_groups_lost") or 0) == 0
        and bool(row.get("topology_valid"))
        for row in full_trials
    )
    duplicate_failures = anatomy_counts.get("duplicate_clusters_same_model_edge", 0)
    missing_boundary = anatomy_counts.get("one_boundary_cluster_missing", 0) + anatomy_counts.get("both_boundary_clusters_missing", 0)
    if missing_boundary >= math.ceil(0.5 * len(target_rows)) or correct_exists < 10:
        branch = "C_cluster_formation"
        reason = "Most target faces do not have two distinct confidently matched boundary clusters in the fixed pool."
    elif alternative_valid >= current_valid + 5 and current_valid < 10:
        branch = "D_pair_plane_representation"
        reason = "Correct pairs exist, but line-derived construction has a materially higher finite-patch ceiling than the current pair fit."
    elif duplicate_failures >= 7 and int(mode_summaries["E_duplicate_skipping"]["recovered_target_count"]) >= 10:
        branch = "B_duplicate_consolidation"
        reason = "Duplicate clusters dominate current failures and duplicate-skipping recovers a material target ceiling."
    elif (
        int(best_mode["recovered_target_count"]) >= 10
        and int(best_mode["current_good_control_pairs_rejected"]) <= 3
        and confirmed_plus_one >= 2
    ):
        branch = "A_spatial_adjacency"
        reason = "A bounded non-oracle adjacency graph recovers a material candidate ceiling with low control damage and final-mesh confirmations."
    else:
        branch = "E_separation_absent"
        reason = "Oracle-correct alternatives do not yet have a low-damage production-feature separation with enough final-mesh confirmations."

    timings["total_seconds"] = time.perf_counter() - started
    return {
        "diagnostic_only": True,
        "uses_initial_model": True,
        "production_selector_changed": False,
        "cohort": {
            "expected": 21,
            "actual": int(len(target_rows)),
            "target_face_ids": target_face_ids,
            "source": "candidate_formation_failure_audit rows classified wrong_edge_pair_or_adjacency",
        },
        "oracle_cluster_matching": {
            key: value for key, value in edge_oracle.items()
            if key not in {"cluster_match_lookup", "cluster_matches_sample"}
        },
        "oracle_anatomy": {
            "classification_counts": anatomy_counts,
            "classification_sum": int(sum(anatomy_counts.values())),
            "rows": anatomy_rows,
        },
        "availability_ceiling": {
            "correct_boundary_pair_exists": int(correct_exists),
            "correct_pair_with_enough_shared_z": int(enough_exists),
            "correct_pair_produces_valid_current_plane": int(current_valid),
            "correct_pair_produces_valid_line_derived_plane": int(alternative_valid),
            "rows": availability_rows,
        },
        "current_adjacency_analysis": {
            "production_rule": "sort active clusters by mean cyclic index independently at each Z and pair immediate cyclic neighbors",
            "wrong_vs_matched_good_controls": current_comparison,
            "hypothesis_counts": hypotheses,
            "matched_good_control_count": int(len(control_rows)),
            "matched_good_controls": control_rows,
        },
        "alternative_adjacency_graphs": {
            "cluster_generation_changed": False,
            "bounded_k": int(bounded_k),
            "max_alternatives_per_cluster": 12,
            "max_pairs_per_mode": 6000,
            "stable_strip_thresholds": stable_strip_thresholds,
            "nonoracle_geometry_dedupe": "oriented normal rounded 1e-4, signed offset 1e-3, finite hull centroid 1e-2",
            "modes": mode_summaries,
        },
        "nonoracle_separation": {
            "label_use": "oracle labels are evaluation-only; graph construction, features, gates, and score are non-oracle",
            "score": "plane_rms + separation/common-Z/strip/stability/order penalties",
            "features": separation,
        },
        "bounded_full_edge_clip_verification": {
            "baseline_unique_finite_ids": int(len(baseline_finite_ids)),
            "baseline_finite_face_ids": sorted(int(value) for value in baseline_finite_ids),
            "one_representative_per_recovered_face": True,
            "representative_selection": "lowest non-oracle pair score within each posthoc recovered face; max 12",
            "trial_count": int(len(full_trials)),
            "confirmed_plus_one_without_loss": int(confirmed_plus_one),
            "rows": full_trials,
        },
        "branch_decision": {
            "selected": branch,
            "reason": reason,
            "best_adjacency_mode": best_mode_name,
            "best_mode_recovered_targets": int(best_mode["recovered_target_count"]),
            "best_mode_control_pair_rejections": int(best_mode["current_good_control_pairs_rejected"]),
            "final_mesh_confirmed_plus_one": int(confirmed_plus_one),
            "production_fix_implemented": False,
        },
        "thresholds": {
            "oracle_edge_match_distance": 0.08,
            "oracle_edge_match_angle_deg": 12.0,
            "oracle_edge_match_inside_fraction": 0.5,
            "production_min_adjacency_levels": int(min_adjacency_levels),
            "production_min_adjacency_fraction": as_json_float(float(min_adjacency_fraction)),
            "production_min_z_span": as_json_float(float(min_z_span)),
            "production_max_plane_rms": as_json_float(float(max_plane_rms)),
        },
        "timings_seconds": {key: as_json_float(float(value)) for key, value in timings.items()},
    }


def diagnose_w2_cluster_formation_failures(
    *,
    failure_rows: list[dict[str, object]],
    initial_vertices: np.ndarray,
    initial_faces: list[list[int]],
    model_faces: list[dict[str, object]],
    z_levels: np.ndarray,
    w2_line_points: np.ndarray | None,
    w2_segments: list[dict[str, object]],
    w2_clusters: list[dict[str, object]],
    w2_raw_candidates: list[dict[str, object]],
    edge_ancestry_internal: dict[str, object],
    segment_min_levels: int,
    segment_min_z_span: float,
    segment_max_z_gap: int,
    cluster_mode: str,
    cluster_max_distance: float,
    cluster_max_angle_deg: float,
    cluster_min_overlap: float,
    face_min_adjacency_levels: int,
    face_min_adjacency_fraction: float,
    face_min_z_span: float,
    face_max_plane_rms: float,
    inside_point: np.ndarray,
    orientation_points: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    core_face_ids: set[int],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
) -> dict[str, object]:
    """Oracle-only segment and cluster audit for the fixed W2 signal."""

    started = time.perf_counter()
    timings: dict[str, float] = {}
    target_rows = [
        row for row in failure_rows
        if str(row.get("root_cause")) == "wrong_edge_pair_or_adjacency"
    ]
    target_face_ids = sorted({int(row["target_face_id"]) for row in target_rows})
    if w2_line_points is None or not target_rows or not w2_segments:
        return {
            "diagnostic_only": True,
            "uses_initial_model": True,
            "production_selector_changed": False,
            "skipped": True,
            "reason": "fixed W2 line points, segments, or target cohort are unavailable",
            "target_face_ids": target_face_ids,
        }

    edges, edge_face_lookup = model_edges_from_faces(initial_vertices, initial_faces)
    edge_id_to_faces = {
        int(edge_id): set(int(value) for value in edge_face_lookup.get(tuple(sorted(edge)), set()))
        for edge_id, edge in enumerate(edges)
    }
    edge_id_by_vertices = {tuple(sorted(edge)): int(edge_id) for edge_id, edge in enumerate(edges)}
    face_boundary_edges: dict[int, list[int]] = {}
    for face_id, face_indices in enumerate(initial_faces):
        boundary: list[int] = []
        for index, vertex_id in enumerate(face_indices):
            key = tuple(sorted((int(vertex_id), int(face_indices[(index + 1) % len(face_indices)]))))
            edge_id = edge_id_by_vertices.get(key)
            if edge_id is not None:
                boundary.append(int(edge_id))
        face_boundary_edges[int(face_id)] = boundary
    required_edge_ids = sorted(
        {
            int(edge_id)
            for face_id in target_face_ids
            for edge_id in face_boundary_edges.get(int(face_id), [])
        }
    )
    required_edge_set = set(required_edge_ids)
    segment_by_id = {
        int(segment["segment_id"]): segment
        for segment in w2_segments
        if segment.get("segment_id") is not None
    }
    segment_oracle_by_id = {
        int(row["segment_id"]): row
        for row in edge_ancestry_internal.get("segment_oracle_rows", [])
        if row.get("segment_id") is not None
    }
    raw_counts = np.asarray(edge_ancestry_internal.get("raw_edge_point_counts", []), dtype=int)
    raw_z_levels = list(edge_ancestry_internal.get("raw_edge_z_levels", []))
    raw_cyclic_z = list(edge_ancestry_internal.get("raw_edge_cyclic_z", []))
    n_half = int(np.asarray(w2_line_points).shape[0])
    z_step = float(np.median(np.diff(z_levels))) if z_levels.size >= 2 else 1.0

    def segment_source(segment: dict[str, object]) -> str:
        return str(segment.get("segment_fit_mode") or "least_squares")

    def segment_oracle_label(segment_id: int) -> dict[str, object]:
        row = dict(segment_oracle_by_id.get(int(segment_id)) or {})
        edge_id = row.get("dominant_edge_id")
        dominant_fraction = finite_float(row.get("dominant_edge_fraction"), 0.0)
        edge_count = int(row.get("oracle_edge_id_count") or 0)
        mixed = bool(edge_count >= 2 and dominant_fraction < 0.85)
        confident = bool(row.get("oracle_confident_finite_edge")) and not mixed
        return {
            **row,
            "edge_id": int(edge_id) if edge_id is not None else None,
            "confident": confident,
            "mixed": mixed,
        }

    def circular_index_gap(a: int, b: int) -> int:
        direct = abs(int(a) - int(b))
        return int(min(direct, max(0, n_half - direct)))

    def circular_index_span(values: list[int]) -> int:
        unique = sorted({int(value) % max(1, n_half) for value in values})
        if len(unique) <= 1:
            return 0
        gaps = [
            int((unique[(index + 1) % len(unique)] - unique[index]) % n_half)
            for index in range(len(unique))
        ]
        return int(n_half - max(gaps))

    def source_counts(members: list[dict[str, object]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for member in members:
            source = segment_source(member)
            counts[source] = counts.get(source, 0) + 1
        return {key: int(value) for key, value in sorted(counts.items())}

    def materialize_groups(groups: list[list[dict[str, object]]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        ordered_groups = [list(group) for group in groups if group]
        for cluster_id, members in enumerate(ordered_groups):
            rep = min(
                members,
                key=lambda segment: (
                    finite_float(segment.get("line_rms"), float("inf")),
                    -int(segment.get("levels") or 0),
                    int(segment.get("segment_id") or -1),
                ),
            )
            rows.append(
                {
                    **dict(rep),
                    "edge_cluster_id": int(cluster_id),
                    "member_segment_ids": sorted(int(member["segment_id"]) for member in members),
                    "member_cyclic_indices": sorted({int(member["cyclic_index"]) for member in members}),
                    "member_count": int(len(members)),
                    "cluster_z_min": as_json_float(
                        min(finite_float(member.get("z_min"), float("inf")) for member in members)
                    ),
                    "cluster_z_max": as_json_float(
                        max(finite_float(member.get("z_max"), -float("inf")) for member in members)
                    ),
                    "cross_index_support": int(len({int(member["cyclic_index"]) for member in members})),
                    "cluster_confidence": as_json_float(
                        float(np.mean([finite_float(member.get("confidence"), 0.0) for member in members]))
                    ),
                    "diagnostic_source_modes": source_counts(members),
                }
            )
        return rows

    def groups_from_clusters(clusters: list[dict[str, object]]) -> list[list[dict[str, object]]]:
        groups: list[list[dict[str, object]]] = []
        for cluster in clusters:
            members = [
                segment_by_id[int(segment_id)]
                for segment_id in cluster.get("member_segment_ids", [])
                if int(segment_id) in segment_by_id
            ]
            if members:
                groups.append(members)
        return groups

    def extended_segment_relation(
        a: dict[str, object],
        b: dict[str, object],
        *,
        allow_small_gap: bool,
    ) -> tuple[float, float, float, int] | None:
        za0 = finite_float(a.get("z_min"), float("nan"))
        za1 = finite_float(a.get("z_max"), float("nan"))
        zb0 = finite_float(b.get("z_min"), float("nan"))
        zb1 = finite_float(b.get("z_max"), float("nan"))
        if not all(np.isfinite(value) for value in (za0, za1, zb0, zb1)):
            return None
        overlap = min(za1, zb1) - max(za0, zb0)
        gap = max(0.0, max(za0, zb0) - min(za1, zb1))
        if overlap > 0.0:
            z = np.linspace(max(za0, zb0), min(za1, zb1), 5)
        elif allow_small_gap and gap <= max(2.0 * z_step, float(cluster_min_overlap)):
            z = np.array([(max(za0, zb0) + min(za1, zb1)) * 0.5], dtype=float)
        else:
            return None
        pa = predict_w2_line(a, z)
        pb = predict_w2_line(b, z)
        distance = float(np.max(np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1)))
        direction_a = np.array(a.get("direction") or [], dtype=float)
        direction_b = np.array(b.get("direction") or [], dtype=float)
        angle = 180.0
        if direction_a.shape == (3,) and direction_b.shape == (3,):
            angle = float(
                np.degrees(np.arccos(np.clip(abs(float(direction_a @ direction_b)), -1.0, 1.0)))
            )
        cyclic_gap = circular_index_gap(int(a.get("cyclic_index") or 0), int(b.get("cyclic_index") or 0))
        return distance, angle, float(overlap), int(cyclic_gap)

    def pairwise_cluster_geometry(members: list[dict[str, object]]) -> dict[str, object]:
        distances: list[float] = []
        angles: list[float] = []
        overlaps: list[float] = []
        graph_edges: list[dict[str, object]] = []
        for left_index, left in enumerate(members):
            for right in members[left_index + 1:]:
                relation = extended_segment_relation(left, right, allow_small_gap=True)
                if relation is None:
                    continue
                distance, angle, overlap, cyclic_gap = relation
                distances.append(float(distance))
                angles.append(float(angle))
                overlaps.append(float(max(0.0, overlap)))
                graph_edges.append(
                    {
                        "segment_ids": [int(left["segment_id"]), int(right["segment_id"])],
                        "distance": as_json_float(distance),
                        "angle_deg": as_json_float(angle),
                        "z_overlap": as_json_float(max(0.0, overlap)),
                        "cyclic_gap": int(cyclic_gap),
                    }
                )
        rep = min(
            members,
            key=lambda segment: (
                finite_float(segment.get("line_rms"), float("inf")),
                -int(segment.get("levels") or 0),
                int(segment.get("segment_id") or -1),
            ),
        )
        rep_distances: list[float] = []
        for member in members:
            if member is rep:
                continue
            relation = extended_segment_relation(rep, member, allow_small_gap=True)
            if relation is not None:
                rep_distances.append(float(relation[0]))
        nearest_links: list[float] = []
        for member in members:
            values = []
            for other in members:
                if other is member:
                    continue
                relation = extended_segment_relation(member, other, allow_small_gap=True)
                if relation is not None:
                    values.append(float(relation[0]))
            if values:
                nearest_links.append(min(values))
        sorted_rep = sorted(rep_distances)
        rep_gaps = np.diff(np.array(sorted_rep, dtype=float)) if len(sorted_rep) >= 2 else np.zeros(0)
        bimodality = (
            float(np.max(rep_gaps)) / max(float(np.median(sorted_rep)), 0.01)
            if rep_gaps.size
            else 0.0
        )
        complete_diameter = max(distances) if distances else 0.0
        chain_ratio = complete_diameter / max(float(np.median(nearest_links)), 0.01) if nearest_links else 0.0
        overlap_density = float(
            sum(value >= float(cluster_min_overlap) for value in overlaps) / max(1, len(overlaps))
        )
        z_indices = sorted(
            {
                int(value)
                for member in members
                for value in member.get("z_indices", [])
            }
        )
        z_gap_count = int(np.sum(np.diff(np.array(z_indices, dtype=int)) > int(segment_max_z_gap))) if len(z_indices) >= 2 else 0
        pooled_z: list[float] = []
        pooled_points: list[np.ndarray] = []
        pooled_member_ids: list[int] = []
        for member_index, member in enumerate(members):
            z0 = finite_float(member.get("z_min"), float("nan"))
            z1 = finite_float(member.get("z_max"), float("nan"))
            if not np.isfinite(z0) or not np.isfinite(z1):
                continue
            sample_z = np.linspace(z0, z1, min(3, max(2, int(member.get("levels") or 2))))
            points = predict_w2_line(member, sample_z)
            pooled_z.extend(float(value) for value in sample_z)
            pooled_points.extend(point for point in points)
            pooled_member_ids.extend([int(member_index)] * len(sample_z))
        loo_angle = 0.0
        loo_offset = 0.0
        if len(members) >= 3 and pooled_z:
            z_array = np.array(pooled_z, dtype=float)
            point_array = np.array(pooled_points, dtype=float)
            member_array = np.array(pooled_member_ids, dtype=int)
            full_fit = fit_xy_line_vs_z(z_array, point_array)
            if full_fit is not None:
                direction = np.array(full_fit["direction"], dtype=float)
                reference_z = float(np.median(z_array))
                reference_point = np.array(
                    [
                        float(full_fit["coef_x"][0]) * reference_z + float(full_fit["coef_x"][1]),
                        float(full_fit["coef_y"][0]) * reference_z + float(full_fit["coef_y"][1]),
                    ],
                    dtype=float,
                )
                indices = list(range(len(members)))
                if len(indices) > 12:
                    indices = [indices[int(index)] for index in np.linspace(0, len(indices) - 1, 12, dtype=int)]
                for member_index in indices:
                    mask = member_array != int(member_index)
                    if int(np.sum(mask)) < 4:
                        continue
                    fit = fit_xy_line_vs_z(z_array[mask], point_array[mask])
                    if fit is None:
                        continue
                    other_direction = np.array(fit["direction"], dtype=float)
                    loo_angle = max(
                        loo_angle,
                        float(
                            np.degrees(
                                np.arccos(
                                    np.clip(abs(float(direction @ other_direction)), -1.0, 1.0)
                                )
                            )
                        ),
                    )
                    other_point = np.array(
                        [
                            float(fit["coef_x"][0]) * reference_z + float(fit["coef_x"][1]),
                            float(fit["coef_y"][0]) * reference_z + float(fit["coef_y"][1]),
                        ],
                        dtype=float,
                    )
                    loo_offset = max(loo_offset, float(np.linalg.norm(reference_point - other_point)))
        feature_row = {
            "pair_count": int(len(graph_edges)),
            "pair_distance_min": as_json_float(min(distances)) if distances else None,
            "pair_distance_median": as_json_float(float(np.median(distances))) if distances else None,
            "complete_diameter": as_json_float(complete_diameter),
            "direction_spread_deg": as_json_float(max(angles)) if angles else 0.0,
            "representative_residual_median": as_json_float(float(np.median(rep_distances))) if rep_distances else 0.0,
            "representative_residual_max": as_json_float(max(rep_distances)) if rep_distances else 0.0,
            "z_overlap_graph_density": as_json_float(overlap_density),
            "cyclic_index_span": int(circular_index_span([int(member.get("cyclic_index") or 0) for member in members])),
            "z_gap_count": int(z_gap_count),
            "leave_one_segment_out_angle_deg": as_json_float(loo_angle),
            "leave_one_segment_out_position": as_json_float(loo_offset),
            "residual_bimodality_score": as_json_float(bimodality),
            "single_linkage_chain_ratio": as_json_float(chain_ratio),
            "flow_continuity": as_json_float(float(len(nearest_links) / max(1, len(members)))),
            "segment_support_levels": int(sum(int(member.get("levels") or 0) for member in members)),
            "cluster_z_span": as_json_float(
                max(finite_float(member.get("z_max"), 0.0) for member in members)
                - min(finite_float(member.get("z_min"), 0.0) for member in members)
            ),
            "source_mode_count": int(len(source_counts(members))),
            "source_modes": source_counts(members),
        }
        feature_row["impurity_composite"] = as_json_float(
            finite_float(feature_row.get("complete_diameter"), 0.0) / max(float(cluster_max_distance), EPS)
            + finite_float(feature_row.get("direction_spread_deg"), 0.0) / max(float(cluster_max_angle_deg), EPS)
            + finite_float(feature_row.get("representative_residual_max"), 0.0) / max(float(cluster_max_distance), EPS)
            + 0.25 * finite_float(feature_row.get("residual_bimodality_score"), 0.0)
            + 0.25 * finite_float(feature_row.get("leave_one_segment_out_position"), 0.0) / max(float(cluster_max_distance), EPS)
        )
        feature_row["_graph_edges"] = graph_edges
        return feature_row

    def cluster_oracle_label(cluster: dict[str, object], members: list[dict[str, object]]) -> dict[str, object]:
        labels = [segment_oracle_label(int(member["segment_id"])) for member in members]
        pure_edge_ids = {
            int(label["edge_id"])
            for label in labels
            if bool(label.get("confident")) and label.get("edge_id") is not None
        }
        mixed_segments = [label for label in labels if bool(label.get("mixed"))]
        rep_id_value = cluster.get("segment_id")
        rep_id = int(rep_id_value) if rep_id_value is not None else -1
        rep_label = segment_oracle_label(rep_id)
        confident_fraction = float(sum(bool(label.get("confident")) for label in labels) / max(1, len(labels)))
        pure = bool(
            len(pure_edge_ids) == 1
            and not mixed_segments
            and confident_fraction >= 0.75
            and bool(rep_label.get("confident"))
            and int(rep_label.get("edge_id")) in pure_edge_ids
        )
        mixed = bool(len(pure_edge_ids) >= 2 or mixed_segments)
        edge_id = next(iter(pure_edge_ids)) if pure else None
        return {
            "classification": "confident_pure" if pure else "mixed" if mixed else "unmatched_or_ambiguous",
            "edge_id": int(edge_id) if edge_id is not None else None,
            "pure_member_edge_ids": sorted(int(value) for value in pure_edge_ids),
            "confident_member_fraction": as_json_float(confident_fraction),
            "mixed_segment_ids": sorted(
                int(label["segment_id"])
                for label in mixed_segments
                if label.get("segment_id") is not None
            ),
            "representative_segment_id": int(rep_id),
            "representative_edge_id": rep_label.get("edge_id"),
            "representative_confident": bool(rep_label.get("confident")),
        }

    def evaluate_cluster_mode(
        name: str,
        clusters: list[dict[str, object]],
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        mode_started = time.perf_counter()
        rows: list[dict[str, object]] = []
        for cluster in clusters:
            members = [
                segment_by_id[int(segment_id)]
                for segment_id in cluster.get("member_segment_ids", [])
                if int(segment_id) in segment_by_id
            ]
            if not members:
                continue
            features = pairwise_cluster_geometry(members)
            graph_edges = features.pop("_graph_edges")
            oracle = cluster_oracle_label(cluster, members)
            rows.append(
                {
                    "cluster_id": int(cluster["edge_cluster_id"]),
                    "member_segment_ids": sorted(int(member["segment_id"]) for member in members),
                    "member_count": int(len(members)),
                    "member_cyclic_indices": sorted({int(member["cyclic_index"]) for member in members}),
                    "features": features,
                    "oracle": oracle,
                    "_graph_edges": graph_edges,
                    "_cluster": cluster,
                }
            )
        pure_rows = [row for row in rows if row["oracle"]["classification"] == "confident_pure"]
        mixed_rows = [row for row in rows if row["oracle"]["classification"] == "mixed"]
        unmatched_rows = [row for row in rows if row["oracle"]["classification"] == "unmatched_or_ambiguous"]
        pure_edge_ids = {
            int(row["oracle"]["edge_id"])
            for row in pure_rows
            if row["oracle"].get("edge_id") is not None
        }
        duplicate_count = int(len(pure_rows) - len(pure_edge_ids))
        summary = {
            "cluster_count": int(len(rows)),
            "confident_pure_matches": int(len(pure_rows)),
            "mixed_clusters": int(len(mixed_rows)),
            "unmatched_clusters": int(len(unmatched_rows)),
            "duplicate_clusters": int(duplicate_count),
            "unique_finite_model_edge_ids": int(len(pure_edge_ids)),
            "required_boundary_edges_recovered": int(len(pure_edge_ids & required_edge_set)),
            "required_boundary_edge_ids_recovered": sorted(int(value) for value in pure_edge_ids & required_edge_set),
            "required_boundary_edges_lost": int(len(required_edge_set - pure_edge_ids)),
            "required_boundary_edge_ids_lost": sorted(int(value) for value in required_edge_set - pure_edge_ids),
            "precision_confident": as_json_float(float(len(pure_rows) / max(1, len(rows)))),
            "required_boundary_edge_recall": as_json_float(
                float(len(pure_edge_ids & required_edge_set) / max(1, len(required_edge_set)))
            ),
            "duplicate_rate": as_json_float(float(duplicate_count / max(1, len(pure_rows)))),
            "timing_seconds": as_json_float(time.perf_counter() - mode_started),
        }
        return summary, rows

    clustering_started = time.perf_counter()
    current_groups = groups_from_clusters(w2_clusters)
    current_clusters = materialize_groups(current_groups)
    complete_clusters_raw, _ = cluster_w2_edge_segments(
        w2_segments,
        mode="complete",
        max_distance=float(cluster_max_distance),
        max_angle_deg=float(cluster_max_angle_deg),
        min_overlap=float(cluster_min_overlap),
    )
    complete_groups = groups_from_clusters(complete_clusters_raw)
    complete_clusters = materialize_groups(complete_groups)
    representative_clusters_raw, _ = cluster_w2_edge_segments(
        w2_segments,
        mode="representative",
        max_distance=float(cluster_max_distance),
        max_angle_deg=float(cluster_max_angle_deg),
        min_overlap=float(cluster_min_overlap),
    )
    representative_groups = groups_from_clusters(representative_clusters_raw)
    representative_clusters = materialize_groups(representative_groups)

    segments_by_cyclic: dict[int, list[dict[str, object]]] = {}
    for segment in w2_segments:
        segments_by_cyclic.setdefault(int(segment.get("cyclic_index") or 0), []).append(segment)
    neighbor_candidates: dict[int, list[tuple[float, int]]] = {
        int(segment["segment_id"]): [] for segment in w2_segments
    }
    for segment in w2_segments:
        segment_id = int(segment["segment_id"])
        cyclic_index = int(segment.get("cyclic_index") or 0)
        candidate_rows: list[dict[str, object]] = []
        for delta in (-2, -1, 0, 1, 2):
            candidate_rows.extend(segments_by_cyclic.get((cyclic_index + delta) % n_half, []))
        for other in candidate_rows:
            other_id = int(other["segment_id"])
            if other_id <= segment_id:
                continue
            relation = extended_segment_relation(segment, other, allow_small_gap=True)
            if relation is None:
                continue
            distance, angle, overlap, cyclic_gap = relation
            if (
                distance > float(cluster_max_distance)
                or angle > float(cluster_max_angle_deg)
                or cyclic_gap > 2
            ):
                continue
            continuity_penalty = max(0.0, float(cluster_min_overlap) - max(0.0, overlap)) / max(float(cluster_min_overlap), EPS)
            score = (
                distance / max(float(cluster_max_distance), EPS)
                + angle / max(float(cluster_max_angle_deg), EPS)
                + 0.25 * float(cyclic_gap)
                + continuity_penalty
            )
            neighbor_candidates[segment_id].append((float(score), int(other_id)))
            neighbor_candidates[other_id].append((float(score), int(segment_id)))
    top_neighbors = {
        segment_id: {
            int(other_id)
            for _, other_id in sorted(values, key=lambda row: (row[0], row[1]))[:2]
        }
        for segment_id, values in neighbor_candidates.items()
    }
    parent = {int(segment["segment_id"]): int(segment["segment_id"]) for segment in w2_segments}

    def find(segment_id: int) -> int:
        while parent[int(segment_id)] != int(segment_id):
            parent[int(segment_id)] = parent[parent[int(segment_id)]]
            segment_id = parent[int(segment_id)]
        return int(segment_id)

    def union(a_id: int, b_id: int) -> None:
        left = find(a_id)
        right = find(b_id)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for segment_id, values in top_neighbors.items():
        for other_id in values:
            if int(segment_id) in top_neighbors.get(int(other_id), set()):
                union(int(segment_id), int(other_id))
    flow_components: dict[int, list[dict[str, object]]] = {}
    for segment in w2_segments:
        flow_components.setdefault(find(int(segment["segment_id"])), []).append(segment)
    flow_groups: list[list[dict[str, object]]] = []
    for component in flow_components.values():
        partitions: list[list[dict[str, object]]] = []
        for segment in sorted(component, key=lambda row: (finite_float(row.get("z_min"), 0.0), int(row["segment_id"]))):
            best_index = None
            best_score = float("inf")
            for index, partition in enumerate(partitions):
                representative = min(
                    partition,
                    key=lambda row: (
                        finite_float(row.get("line_rms"), float("inf")),
                        -int(row.get("levels") or 0),
                        int(row["segment_id"]),
                    ),
                )
                relation = extended_segment_relation(segment, representative, allow_small_gap=True)
                if relation is None:
                    continue
                distance, angle, _, cyclic_gap = relation
                if (
                    distance <= 1.25 * float(cluster_max_distance)
                    and angle <= 1.25 * float(cluster_max_angle_deg)
                    and cyclic_gap <= 2
                ):
                    score = distance / max(float(cluster_max_distance), EPS) + angle / max(float(cluster_max_angle_deg), EPS)
                    if score < best_score:
                        best_score = float(score)
                        best_index = int(index)
            if best_index is None:
                partitions.append([segment])
            else:
                partitions[best_index].append(segment)
        flow_groups.extend(partitions)
    flow_clusters = materialize_groups(flow_groups)

    def post_split_group(members: list[dict[str, object]]) -> list[list[dict[str, object]]]:
        if len(members) < 3:
            return [members]
        pair_rows: list[tuple[float, dict[str, object], dict[str, object]]] = []
        for left_index, left in enumerate(members):
            for right in members[left_index + 1:]:
                relation = extended_segment_relation(left, right, allow_small_gap=True)
                if relation is None:
                    continue
                distance, angle, _, _ = relation
                normalized = (
                    distance / max(float(cluster_max_distance), EPS)
                    + angle / max(float(cluster_max_angle_deg), EPS)
                )
                pair_rows.append((float(normalized), left, right))
        if not pair_rows:
            return [members]
        parent_score, medoid_a, medoid_b = max(pair_rows, key=lambda row: row[0])
        if parent_score < 1.25:
            return [members]
        left_group: list[dict[str, object]] = []
        right_group: list[dict[str, object]] = []
        for member in members:
            scores = []
            for medoid in (medoid_a, medoid_b):
                relation = extended_segment_relation(member, medoid, allow_small_gap=True)
                if relation is None:
                    scores.append(float("inf"))
                else:
                    scores.append(
                        relation[0] / max(float(cluster_max_distance), EPS)
                        + relation[1] / max(float(cluster_max_angle_deg), EPS)
                    )
            (left_group if scores[0] <= scores[1] else right_group).append(member)
        if not left_group or not right_group:
            return [members]

        def complete_score(group: list[dict[str, object]]) -> float:
            values = []
            for left_index, left in enumerate(group):
                for right in group[left_index + 1:]:
                    relation = extended_segment_relation(left, right, allow_small_gap=True)
                    if relation is not None:
                        values.append(
                            relation[0] / max(float(cluster_max_distance), EPS)
                            + relation[1] / max(float(cluster_max_angle_deg), EPS)
                        )
            return max(values) if values else 0.0

        child_score = max(complete_score(left_group), complete_score(right_group))
        if child_score > 0.75 * parent_score:
            return [members]
        return [left_group, right_group]

    post_split_groups = [
        group
        for current_group in current_groups
        for group in post_split_group(current_group)
    ]
    post_split_clusters = materialize_groups(post_split_groups)
    timings["cluster_mode_construction_seconds"] = time.perf_counter() - clustering_started

    mode_clusters = {
        "A_current_complete": current_clusters,
        "B_complete_linkage_bounded": complete_clusters,
        "C_representative_link": representative_clusters,
        "D_spatial_flow": flow_clusters,
        "E_post_cluster_split": post_split_clusters,
    }
    mode_internal_rows: dict[str, list[dict[str, object]]] = {}
    mode_summaries: dict[str, dict[str, object]] = {}
    for mode_name, clusters in mode_clusters.items():
        summary, rows = evaluate_cluster_mode(mode_name, clusters)
        mode_summaries[mode_name] = summary
        mode_internal_rows[mode_name] = rows
    current_pure_edges = {
        int(row["oracle"]["edge_id"])
        for row in mode_internal_rows["A_current_complete"]
        if row["oracle"]["classification"] == "confident_pure"
        and row["oracle"].get("edge_id") is not None
    }
    control_edge_ids = current_pure_edges - required_edge_set
    for mode_name, rows in mode_internal_rows.items():
        pure_edges = {
            int(row["oracle"]["edge_id"])
            for row in rows
            if row["oracle"]["classification"] == "confident_pure"
            and row["oracle"].get("edge_id") is not None
        }
        mode_summaries[mode_name]["current_good_boundary_controls"] = int(len(control_edge_ids))
        mode_summaries[mode_name]["current_good_boundary_controls_damaged"] = int(len(control_edge_ids - pure_edges))
        mode_summaries[mode_name]["current_good_boundary_control_edge_ids_lost"] = sorted(
            int(value) for value in control_edge_ids - pure_edges
        )
        mode_summaries[mode_name]["cluster_signature"] = hashlib.sha256(
            json.dumps(
                sorted(tuple(int(value) for value in row["member_segment_ids"]) for row in rows),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    mode_summaries["B_complete_linkage_bounded"]["identical_to_current"] = bool(
        mode_summaries["B_complete_linkage_bounded"]["cluster_signature"]
        == mode_summaries["A_current_complete"]["cluster_signature"]
    )

    current_cluster_by_id = {
        int(row["cluster_id"]): row for row in mode_internal_rows["A_current_complete"]
    }
    segment_cluster_membership: dict[int, list[int]] = {}
    for row in mode_internal_rows["A_current_complete"]:
        for segment_id in row["member_segment_ids"]:
            segment_cluster_membership.setdefault(int(segment_id), []).append(int(row["cluster_id"]))
    adjacency_cluster_ids = {
        int(value)
        for candidate in w2_raw_candidates
        for value in candidate.get("w2_edge_cluster_ids", [])
    }

    funnel_started = time.perf_counter()
    first_loss_order = [
        "no_raw_support",
        "short_Z_span",
        "fixed_index_fragmentation",
        "segment_fit_rejected",
        "segment_itself_mixed",
        "lost_during_cluster_formation",
        "cluster_mixed_or_impure",
        "cluster_duplicate_only",
        "cluster_match_failed",
        "available_confident_cluster",
    ]
    first_loss_counts = {key: 0 for key in first_loss_order}
    boundary_rows: list[dict[str, object]] = []

    def fixed_runs(values: set[int]) -> list[list[int]]:
        ordered = sorted(int(value) for value in values)
        if not ordered:
            return []
        runs = [[ordered[0]]]
        for value in ordered[1:]:
            if int(value) - runs[-1][-1] <= int(segment_max_z_gap):
                runs[-1].append(int(value))
            else:
                runs.append([int(value)])
        return runs

    for edge_id in required_edge_ids:
        count = int(raw_counts[edge_id]) if edge_id < raw_counts.size else 0
        edge_z = set(int(value) for value in raw_z_levels[edge_id]) if edge_id < len(raw_z_levels) else set()
        cyclic_mapping = raw_cyclic_z[edge_id] if edge_id < len(raw_cyclic_z) else {}
        cyclic_mapping = {
            int(cyclic_index): set(int(value) for value in values)
            for cyclic_index, values in dict(cyclic_mapping).items()
        }
        fixed_index_runs = [
            {
                "cyclic_index": int(cyclic_index),
                "levels": int(len(run)),
                "z_min": as_json_float(float(z_levels[min(run)])),
                "z_max": as_json_float(float(z_levels[max(run)])),
                "z_span": as_json_float(float(z_levels[max(run)] - z_levels[min(run)])),
            }
            for cyclic_index, values in sorted(cyclic_mapping.items())
            for run in fixed_runs(values)
        ]
        qualifying_runs = [
            row for row in fixed_index_runs
            if int(row["levels"]) >= int(segment_min_levels)
            and finite_float(row.get("z_span"), 0.0) >= float(segment_min_z_span)
        ]
        edge_segment_rows = [
            row for row in segment_oracle_by_id.values()
            if row.get("dominant_edge_id") is not None and int(row["dominant_edge_id"]) == int(edge_id)
        ]
        pure_segment_rows = [
            row
            for row in edge_segment_rows
            if bool(row.get("oracle_confident_finite_edge"))
            and not (
                int(row.get("oracle_edge_id_count") or 0) >= 2
                and finite_float(row.get("dominant_edge_fraction"), 0.0) < 0.85
            )
        ]
        mixed_segment_rows = [
            row for row in edge_segment_rows
            if int(row.get("oracle_edge_id_count") or 0) >= 2
            and finite_float(row.get("dominant_edge_fraction"), 0.0) < 0.85
        ]
        source_summary = {
            source: int(sum(str(row.get("segment_fit_mode") or "least_squares") == source for row in edge_segment_rows))
            for source in ("least_squares", "robust_inlier", "one_split")
        }
        segment_ids = {
            int(row["segment_id"])
            for row in edge_segment_rows
            if row.get("segment_id") is not None
        }
        cluster_ids = sorted(
            {
                int(cluster_id)
                for segment_id in segment_ids
                for cluster_id in segment_cluster_membership.get(int(segment_id), [])
            }
        )
        pure_cluster_ids = [
            cluster_id
            for cluster_id in cluster_ids
            if current_cluster_by_id[cluster_id]["oracle"]["classification"] == "confident_pure"
            and current_cluster_by_id[cluster_id]["oracle"].get("edge_id") == int(edge_id)
        ]
        mixed_cluster_ids = [
            cluster_id
            for cluster_id in cluster_ids
            if current_cluster_by_id[cluster_id]["oracle"]["classification"] == "mixed"
        ]
        unmatched_cluster_ids = [
            cluster_id
            for cluster_id in cluster_ids
            if current_cluster_by_id[cluster_id]["oracle"]["classification"] == "unmatched_or_ambiguous"
        ]
        used_cluster_ids = sorted(int(value) for value in set(pure_cluster_ids) & adjacency_cluster_ids)
        raw_span = (
            float(z_levels[max(edge_z)] - z_levels[min(edge_z)])
            if edge_z
            else 0.0
        )
        if count == 0:
            first_loss = "no_raw_support"
        elif len(edge_z) < int(segment_min_levels) or raw_span < float(segment_min_z_span):
            first_loss = "short_Z_span"
        elif not qualifying_runs:
            first_loss = "fixed_index_fragmentation"
        elif not edge_segment_rows:
            first_loss = "segment_fit_rejected"
        elif not pure_segment_rows and mixed_segment_rows:
            first_loss = "segment_itself_mixed"
        elif pure_segment_rows and not cluster_ids:
            first_loss = "lost_during_cluster_formation"
        elif cluster_ids and not pure_cluster_ids and mixed_cluster_ids:
            first_loss = "cluster_mixed_or_impure"
        elif pure_cluster_ids and not used_cluster_ids and len(pure_cluster_ids) >= 2:
            first_loss = "cluster_duplicate_only"
        elif not pure_cluster_ids:
            first_loss = "cluster_match_failed"
        else:
            first_loss = "available_confident_cluster"
        first_loss_counts[first_loss] += 1
        boundary_rows.append(
            {
                "edge_id": int(edge_id),
                "incident_target_face_ids": sorted(
                    int(face_id)
                    for face_id in target_face_ids
                    if int(edge_id) in face_boundary_edges.get(int(face_id), [])
                ),
                "raw_w2_line_point_support": int(count),
                "raw_z_levels": int(len(edge_z)),
                "longest_contiguous_z_run": int(longest_contiguous_count(sorted(edge_z), max_gap=1)),
                "raw_z_span": as_json_float(raw_span),
                "cyclic_index_count": int(len(cyclic_mapping)),
                "cyclic_index_span": int(circular_index_span(list(cyclic_mapping))),
                "fixed_index_runs": int(len(fixed_index_runs)),
                "qualifying_fixed_index_runs": int(len(qualifying_runs)),
                "fixed_index_run_sample": fixed_index_runs[:12],
                "least_squares_segments": int(source_summary["least_squares"]),
                "robust_segments": int(source_summary["robust_inlier"]),
                "one_split_segments": int(source_summary["one_split"]),
                "segments_after_filters": int(len(edge_segment_rows)),
                "confident_pure_segments": int(len(pure_segment_rows)),
                "mixed_segments": int(len(mixed_segment_rows)),
                "cluster_membership_ids": cluster_ids,
                "confident_cluster_ids": pure_cluster_ids,
                "mixed_cluster_ids": mixed_cluster_ids,
                "ambiguous_or_unmatched_cluster_ids": unmatched_cluster_ids,
                "used_in_adjacency_pair_cluster_ids": used_cluster_ids,
                "first_loss": first_loss,
            }
        )
    timings["boundary_edge_funnel_seconds"] = time.perf_counter() - funnel_started

    mixed_started = time.perf_counter()
    mixed_case_order = [
        "mixed_segment_before_clustering",
        "pure_segments_from_different_edges_merged",
        "single_linkage_chaining",
        "intersecting_edges_merged",
        "duplicate_hypotheses_same_edge",
        "ambiguous_because_short_overlap",
        "pure_or_other",
    ]
    mixed_case_counts = {key: 0 for key in mixed_case_order}
    mixed_case_rows: list[dict[str, object]] = []
    pure_edge_cluster_counts: dict[int, int] = {}
    for row in mode_internal_rows["A_current_complete"]:
        oracle = row["oracle"]
        if oracle["classification"] == "confident_pure" and oracle.get("edge_id") is not None:
            pure_edge_cluster_counts[int(oracle["edge_id"])] = pure_edge_cluster_counts.get(int(oracle["edge_id"]), 0) + 1
    for row in mode_internal_rows["A_current_complete"]:
        oracle = row["oracle"]
        graph_edges = list(row.get("_graph_edges") or [])
        weak_links = []
        for edge_row in graph_edges:
            left_id, right_id = [int(value) for value in edge_row["segment_ids"]]
            left_label = segment_oracle_label(left_id)
            right_label = segment_oracle_label(right_id)
            if (
                left_label.get("edge_id") is not None
                and right_label.get("edge_id") is not None
                and int(left_label["edge_id"]) != int(right_label["edge_id"])
            ):
                weak_links.append(edge_row)
        pure_member_edges = [int(value) for value in oracle.get("pure_member_edge_ids", [])]
        intersecting = any(
            bool(set(edges[left]) & set(edges[right]))
            for left_index, left in enumerate(pure_member_edges)
            for right in pure_member_edges[left_index + 1:]
        )
        if oracle["classification"] == "mixed" and oracle.get("mixed_segment_ids"):
            category = "mixed_segment_before_clustering"
        elif oracle["classification"] == "mixed" and len(pure_member_edges) >= 2:
            all_pairs_compatible = all(
                finite_float(edge_row.get("distance"), float("inf")) <= float(cluster_max_distance)
                and finite_float(edge_row.get("angle_deg"), float("inf")) <= float(cluster_max_angle_deg)
                and finite_float(edge_row.get("z_overlap"), 0.0) >= float(cluster_min_overlap)
                for edge_row in graph_edges
            )
            if not all_pairs_compatible:
                category = "single_linkage_chaining"
            elif intersecting:
                category = "intersecting_edges_merged"
            else:
                category = "pure_segments_from_different_edges_merged"
        elif (
            oracle["classification"] == "confident_pure"
            and oracle.get("edge_id") is not None
            and pure_edge_cluster_counts.get(int(oracle["edge_id"]), 0) >= 2
        ):
            category = "duplicate_hypotheses_same_edge"
        elif (
            oracle["classification"] == "unmatched_or_ambiguous"
            and finite_float(row["features"].get("cluster_z_span"), 0.0) < float(cluster_min_overlap)
        ):
            category = "ambiguous_because_short_overlap"
        else:
            category = "pure_or_other"
        mixed_case_counts[category] += 1
        if category != "pure_or_other":
            mixed_case_rows.append(
                {
                    "cluster_id": int(row["cluster_id"]),
                    "category": category,
                    "member_segment_ids": row["member_segment_ids"],
                    "member_oracle_edge_ids": pure_member_edges,
                    "mixed_segment_ids": oracle.get("mixed_segment_ids", []),
                    "features": row["features"],
                    "weak_links_between_oracle_edges": sorted(
                        weak_links,
                        key=lambda value: (
                            -finite_float(value.get("distance"), 0.0),
                            -finite_float(value.get("angle_deg"), 0.0),
                        ),
                    )[:10],
                }
            )
    timings["segment_cluster_purity_seconds"] = time.perf_counter() - mixed_started

    separation_started = time.perf_counter()

    def auc(values: list[tuple[float, bool]], direction: str) -> float | None:
        positives = [value for value, label in values if label and np.isfinite(value)]
        negatives = [value for value, label in values if not label and np.isfinite(value)]
        if not positives or not negatives:
            return None
        wins = 0.0
        for positive in positives:
            for negative in negatives:
                if positive == negative:
                    wins += 0.5
                elif (direction == "higher" and positive > negative) or (direction == "lower" and positive < negative):
                    wins += 1.0
        return float(wins / (len(positives) * len(negatives)))

    feature_directions = {
        "complete_diameter": "higher",
        "direction_spread_deg": "higher",
        "representative_residual_max": "higher",
        "z_overlap_graph_density": "lower",
        "cyclic_index_span": "higher",
        "z_gap_count": "higher",
        "leave_one_segment_out_angle_deg": "higher",
        "leave_one_segment_out_position": "higher",
        "residual_bimodality_score": "higher",
        "single_linkage_chain_ratio": "higher",
        "flow_continuity": "lower",
        "source_mode_count": "higher",
        "impurity_composite": "higher",
    }
    current_labeled_rows = [
        row for row in mode_internal_rows["A_current_complete"]
        if row["oracle"]["classification"] in {"confident_pure", "mixed"}
    ]
    cluster_separation: dict[str, object] = {}
    for feature, direction in feature_directions.items():
        values = [
            (
                finite_float(row["features"].get(feature), float("nan")),
                row["oracle"]["classification"] == "mixed",
            )
            for row in current_labeled_rows
        ]
        finite_values = np.array([value for value, _ in values if np.isfinite(value)], dtype=float)
        frontier = []
        if finite_values.size:
            for quantile in (0.25, 0.50, 0.75, 0.90):
                threshold = float(np.quantile(finite_values, quantile))
                selected = [
                    (value, label)
                    for value, label in values
                    if np.isfinite(value)
                    and (value >= threshold if direction == "higher" else value <= threshold)
                ]
                frontier.append(
                    {
                        "quantile": as_json_float(quantile),
                        "threshold": as_json_float(threshold),
                        "selected": int(len(selected)),
                        "mixed_precision": as_json_float(
                            float(sum(label for _, label in selected) / max(1, len(selected)))
                        ),
                        "mixed_recall": as_json_float(
                            float(
                                sum(label for _, label in selected)
                                / max(1, sum(label for _, label in values))
                            )
                        ),
                    }
                )
        feature_auc = auc(values, direction)
        cluster_separation[feature] = {
            "direction": direction,
            "auc": as_json_float(feature_auc) if feature_auc is not None else None,
            "pure": distribution_summary([value for value, label in values if not label and np.isfinite(value)]),
            "mixed": distribution_summary([value for value, label in values if label and np.isfinite(value)]),
            "threshold_frontier": frontier,
        }
    timings["cluster_nonoracle_separation_seconds"] = time.perf_counter() - separation_started

    downstream_started = time.perf_counter()

    def cluster_center_index(cluster: dict[str, object]) -> float:
        values = [
            float(value)
            for value in (cluster.get("member_cyclic_indices") or [cluster.get("cyclic_index") or 0])
        ]
        return float(np.mean(values)) if values else 0.0

    def diagnostic_candidate_signature(candidate: dict[str, object]) -> tuple[object, ...]:
        plane = candidate_plane(candidate)
        if plane is None:
            return (int(candidate_track_id(candidate)),)
        point, normal = plane
        normal = np.array(normal, dtype=float)
        first = next((float(value) for value in normal if abs(float(value)) > 1e-10), 1.0)
        if first < 0.0:
            normal *= -1.0
        offset = float(normal @ point)
        hull = np.array(candidate.get("hull") or [point], dtype=float).reshape((-1, 3))
        centroid = np.mean(hull, axis=0)
        return (
            *(round(float(value), 4) for value in normal),
            round(offset, 3),
            *(round(float(value), 2) for value in centroid),
        )

    def exact_candidate_signature(candidate: dict[str, object]) -> tuple[object, ...]:
        plane = candidate_plane(candidate)
        if plane is None:
            return (int(candidate_track_id(candidate)),)
        point, normal = plane
        normal = np.array(normal, dtype=float)
        first = next((float(value) for value in normal if abs(float(value)) > 1e-12), 1.0)
        if first < 0.0:
            normal *= -1.0
        offset = float(normal @ point)
        hull = np.array(candidate.get("hull") or [point], dtype=float).reshape((-1, 3))
        return (
            *(round(float(value), 12) for value in normal),
            round(offset, 12),
            *(round(float(value), 10) for value in hull.reshape(-1)),
        )

    candidate_oracle_cache: dict[tuple[object, ...], dict[str, object] | None] = {}

    def downstream_pair_features(
        left: dict[str, object],
        right: dict[str, object],
        levels: list[int],
    ) -> dict[str, object]:
        ordered = sorted(set(int(value) for value in levels))
        z = z_levels[np.array(ordered, dtype=int)] if ordered else np.zeros(0, dtype=float)
        left_points = predict_w2_line(left, z) if z.size else np.zeros((0, 3), dtype=float)
        right_points = predict_w2_line(right, z) if z.size else np.zeros((0, 3), dtype=float)
        separation = (
            np.linalg.norm(left_points[:, :2] - right_points[:, :2], axis=1)
            if z.size
            else np.zeros(0, dtype=float)
        )
        separation_median = float(np.median(separation)) if separation.size else float("inf")
        separation_delta = np.abs(np.diff(separation)) if separation.size >= 2 else np.zeros(0, dtype=float)
        separation_smoothness = (
            float(np.percentile(separation_delta, 95)) / max(separation_median, 0.01)
            if separation_delta.size
            else 0.0
        )
        radial_left = (
            left_points[:, :2] - np.asarray(inside_point[:2], dtype=float)[None, :]
            if left_points.size
            else np.zeros((0, 2), dtype=float)
        )
        radial_right = (
            right_points[:, :2] - np.asarray(inside_point[:2], dtype=float)[None, :]
            if right_points.size
            else np.zeros((0, 2), dtype=float)
        )
        signed_order = radial_left[:, 0] * radial_right[:, 1] - radial_left[:, 1] * radial_right[:, 0]
        signed_order = signed_order[np.abs(signed_order) > 1e-9]
        order_flips = (
            int(np.sum(np.sign(signed_order[1:]) != np.sign(signed_order[:-1])))
            if signed_order.size >= 2
            else 0
        )
        order_stable_fraction = (
            float(max(np.sum(signed_order > 0.0), np.sum(signed_order < 0.0)) / signed_order.size)
            if signed_order.size
            else 0.0
        )
        points = np.vstack([left_points, right_points]) if z.size else np.zeros((0, 3), dtype=float)
        plane = fit_plane(points) if points.shape[0] >= 3 else None
        plane_angles: list[float] = []
        if plane is not None and len(ordered) >= 6:
            for chunk in np.array_split(np.array(ordered, dtype=int), 3):
                if chunk.size < 2:
                    continue
                chunk_z = z_levels[chunk]
                chunk_points = np.vstack(
                    [predict_w2_line(left, chunk_z), predict_w2_line(right, chunk_z)]
                )
                chunk_plane = fit_plane(chunk_points)
                if chunk_plane is None:
                    continue
                plane_angles.append(
                    float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    abs(float(plane.normal @ chunk_plane.normal)),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                )
        sampled_levels = ordered
        if len(sampled_levels) > 11:
            sampled_levels = [
                sampled_levels[int(index)]
                for index in np.linspace(0, len(sampled_levels) - 1, 11, dtype=int)
            ]
        left_center = int(round(cluster_center_index(left))) % n_half
        right_center = int(round(cluster_center_index(right))) % n_half
        forward = [
            (left_center + step) % n_half
            for step in range(1, (right_center - left_center) % n_half)
        ]
        backward = [
            (left_center - step) % n_half
            for step in range(1, (left_center - right_center) % n_half)
        ]
        arc_indices = forward if len(forward) <= len(backward) else backward
        strip_covered = 0
        for z_index in sampled_levels:
            if not arc_indices:
                continue
            observed = np.asarray(
                w2_line_points[np.array(arc_indices, dtype=int), int(z_index), :2],
                dtype=float,
            )
            observed = observed[np.all(np.isfinite(observed), axis=1)]
            if observed.shape[0] == 0:
                continue
            p0 = predict_w2_line(left, np.array([z_levels[int(z_index)]], dtype=float))[0, :2]
            p1 = predict_w2_line(right, np.array([z_levels[int(z_index)]], dtype=float))[0, :2]
            segment = p1 - p0
            denominator = float(segment @ segment)
            if denominator <= EPS:
                continue
            t = np.clip(((observed - p0) @ segment) / denominator, 0.0, 1.0)
            distance = np.linalg.norm(observed - (p0 + t[:, None] * segment[None, :]), axis=1)
            width = max(0.025, min(0.15, 0.25 * float(np.linalg.norm(segment))))
            strip_covered += int(np.any(distance <= width))
        strip_support = float(strip_covered / max(1, len(sampled_levels)))
        return {
            "common_z_levels": int(len(ordered)),
            "z_span": as_json_float(float(np.ptp(z))) if z.size else 0.0,
            "separation_median": as_json_float(separation_median),
            "separation_smoothness": as_json_float(separation_smoothness),
            "order_flip_count": int(order_flips),
            "order_stable_fraction": as_json_float(order_stable_fraction),
            "plane_normal_stability_deg": as_json_float(max(plane_angles)) if plane_angles else None,
            "local_observed_strip_support": as_json_float(strip_support),
            "_z": z,
            "_points": points,
        }

    def bounded_downstream_rows(
        mode_name: str,
        clusters: list[dict[str, object]],
        mode_index: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        active_by_z: dict[int, list[dict[str, object]]] = {}
        for cluster in clusters:
            z0 = finite_float(cluster.get("cluster_z_min"), float("nan"))
            z1 = finite_float(cluster.get("cluster_z_max"), float("nan"))
            if not np.isfinite(z0) or not np.isfinite(z1):
                continue
            for z_index, z_value in enumerate(z_levels):
                if z0 <= float(z_value) <= z1:
                    active_by_z.setdefault(int(z_index), []).append(cluster)
        pair_levels: dict[tuple[int, int], list[int]] = {}
        for z_index, rows in active_by_z.items():
            ordered = sorted(
                rows,
                key=lambda cluster: (
                    cluster_center_index(cluster),
                    int(cluster["edge_cluster_id"]),
                ),
            )
            count = len(ordered)
            if count < 2:
                continue
            for index, left in enumerate(ordered):
                for gap in range(1, min(4, count - 1) + 1):
                    right = ordered[(index + gap) % count]
                    pair = tuple(
                        sorted(
                            (
                                int(left["edge_cluster_id"]),
                                int(right["edge_cluster_id"]),
                            )
                        )
                    )
                    if pair[0] != pair[1]:
                        pair_levels.setdefault(pair, []).append(int(z_index))
        for pair in list(pair_levels):
            pair_levels[pair] = sorted(set(int(value) for value in pair_levels[pair]))
        cluster_by_id = {int(cluster["edge_cluster_id"]): cluster for cluster in clusters}
        sorted_pairs = sorted(
            pair_levels.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        bounded_pairs: list[tuple[tuple[int, int], list[int]]] = []
        degree: dict[int, int] = {}
        for pair, levels in sorted_pairs:
            if degree.get(pair[0], 0) >= 12 or degree.get(pair[1], 0) >= 12:
                continue
            bounded_pairs.append((pair, levels))
            degree[pair[0]] = degree.get(pair[0], 0) + 1
            degree[pair[1]] = degree.get(pair[1], 0) + 1
            if len(bounded_pairs) >= 6000:
                break
        rows: list[dict[str, object]] = []
        rejection_counts: dict[str, int] = {}
        seen_geometry: set[tuple[object, ...]] = set()
        for ordinal, (pair, levels) in enumerate(bounded_pairs):
            if len(levels) < int(face_min_adjacency_levels):
                rejection_counts["insufficient_shared_z"] = rejection_counts.get("insufficient_shared_z", 0) + 1
                continue
            z = z_levels[np.array(levels, dtype=int)]
            z_span = float(np.ptp(z)) if z.size else 0.0
            if z_span < float(face_min_z_span):
                rejection_counts["insufficient_shared_z"] = rejection_counts.get("insufficient_shared_z", 0) + 1
                continue
            left = cluster_by_id[pair[0]]
            right = cluster_by_id[pair[1]]
            overlap = min(
                finite_float(left.get("cluster_z_max"), -float("inf")),
                finite_float(right.get("cluster_z_max"), -float("inf")),
            ) - max(
                finite_float(left.get("cluster_z_min"), float("inf")),
                finite_float(right.get("cluster_z_min"), float("inf")),
            )
            denominator = (
                max(float(overlap) / max(z_step, EPS), 1.0)
                if np.isfinite(overlap)
                else float(len(levels))
            )
            adjacency_fraction = float(len(levels)) / max(denominator, 1.0)
            if adjacency_fraction < float(face_min_adjacency_fraction):
                rejection_counts["insufficient_shared_z"] = rejection_counts.get("insufficient_shared_z", 0) + 1
                continue
            features = downstream_pair_features(left, right, levels)
            points = np.asarray(features.pop("_points"), dtype=float)
            features.pop("_z", None)
            if not (
                points.shape[0] >= 3
                and np.all(np.isfinite(points))
                and np.all(points >= bounds_min[None, :])
                and np.all(points <= bounds_max[None, :])
            ):
                rejection_counts["outside_observed_bounds"] = rejection_counts.get("outside_observed_bounds", 0) + 1
                continue
            plane = fit_plane(points)
            if plane is None or float(plane.rms) > float(face_max_plane_rms):
                rejection_counts["poor_plane_fit"] = rejection_counts.get("poor_plane_fit", 0) + 1
                continue
            normal, orientation = orient_plane_from_observed_points(
                plane,
                orientation_points,
                inside_point,
            )
            u, v = plane_basis(normal)
            coordinates = np.column_stack(
                [
                    (points - plane.centroid) @ u,
                    (points - plane.centroid) @ v,
                ]
            )
            hull_indices = convex_hull_indices(coordinates)
            if len(hull_indices) < 3:
                rejection_counts["nonfinite_hull"] = rejection_counts.get("nonfinite_hull", 0) + 1
                continue
            hull = points[np.array(hull_indices, dtype=int)]
            diameter = float(
                np.max(np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2))
            )
            candidate = {
                "track_id": int(9800000 + 10000 * int(mode_index) + int(ordinal)),
                "source_track_id": None,
                "window": 2,
                "levels": int(len(levels)),
                "z_min": as_json_float(float(np.min(z))),
                "z_max": as_json_float(float(np.max(z))),
                "center_mean": as_json_float(
                    float(np.mean([cluster_center_index(left), cluster_center_index(right)]))
                ),
                "fit_points": int(points.shape[0]),
                "candidate_source": "w2_cluster_formation_diagnostic",
                "candidate_origin": "w2_addition",
                "diagnostic_cluster_mode": str(mode_name),
                "w2_edge_cluster_ids": [int(pair[0]), int(pair[1])],
                "w2_adjacency_levels": int(len(levels)),
                "w2_adjacency_fraction": as_json_float(adjacency_fraction),
                "w2_edge_confidence": as_json_float(
                    float(
                        np.mean(
                            [
                                finite_float(left.get("cluster_confidence"), 0.0),
                                finite_float(right.get("cluster_confidence"), 0.0),
                            ]
                        )
                    )
                ),
                "plane_centroid": as_json_point(plane.centroid),
                "plane_normal": as_json_point(normal),
                "plane_rms": as_json_float(float(plane.rms)),
                "plane_max_abs": as_json_float(float(plane.max_abs)),
                "orientation_positive_frac": as_json_float(
                    float(orientation["orientation_positive_frac"])
                ),
                "orientation_point_count": int(orientation["orientation_point_count"]),
                "candidate_score": as_json_float(
                    float(plane.rms) / max(float(len(levels)) ** 0.5, 1.0)
                ),
                "hull_diameter": as_json_float(diameter),
                "hull_area": as_json_float(float(polygon_area(hull))),
                "hull": [as_json_point(point) for point in hull],
                "sample_points": [
                    as_json_point(point)
                    for point in points[:: max(1, points.shape[0] // 80)]
                ],
                "z_indices": [int(value) for value in levels],
                "low_source_counts": {"w2_edge": int(points.shape[0])},
                "low_valley_frac": 0.0,
            }
            geometry_signature = diagnostic_candidate_signature(candidate)
            if geometry_signature in seen_geometry:
                rejection_counts["geometry_duplicate"] = rejection_counts.get("geometry_duplicate", 0) + 1
                continue
            seen_geometry.add(geometry_signature)
            oracle_signature = exact_candidate_signature(candidate)
            if oracle_signature not in candidate_oracle_cache:
                candidate_oracle_cache[oracle_signature] = candidate_oracle_face(
                    candidate,
                    model_faces,
                )
            oracle = candidate_oracle_cache[oracle_signature]
            face_id = (
                int(oracle["face_id"])
                if oracle is not None and oracle.get("face_id") is not None
                else None
            )
            finite_good = bool(oracle is not None and oracle.get("finite_good"))
            stable_strip = bool(
                finite_float(features.get("order_stable_fraction"), 0.0) >= 0.90
                and int(features.get("order_flip_count") or 0) == 0
                and finite_float(features.get("separation_smoothness"), float("inf")) <= 0.35
                and finite_float(features.get("plane_normal_stability_deg"), float("inf")) <= 10.0
                and finite_float(features.get("local_observed_strip_support"), 0.0) >= 0.40
            )
            nonoracle_score = float(
                finite_float(candidate.get("plane_rms"), 1.0)
                + 0.02 * finite_float(features.get("separation_smoothness"), 10.0)
                + 0.02 * (1.0 - finite_float(features.get("local_observed_strip_support"), 0.0))
                + 0.002 * finite_float(features.get("plane_normal_stability_deg"), 90.0)
                + 0.05 * float(int(features.get("order_flip_count") or 0) > 0)
            )
            rows.append(
                {
                    "pair": pair,
                    "features": features,
                    "candidate": candidate,
                    "oracle_face_id": face_id,
                    "finite_good": finite_good,
                    "stable_strip": stable_strip,
                    "nonoracle_score": as_json_float(nonoracle_score),
                }
            )
        return rows, {
            "raw_pairs": int(len(pair_levels)),
            "bounded_pairs": int(len(bounded_pairs)),
            "candidate_count": int(len(rows)),
            "rejection_counts": {
                key: int(value) for key, value in sorted(rejection_counts.items())
            },
        }

    baseline_required_edges = int(
        mode_summaries["A_current_complete"]["required_boundary_edges_recovered"]
    )
    downstream_selected_modes = ["A_current_complete"]
    for mode_name in (
        "C_representative_link",
        "D_spatial_flow",
        "E_post_cluster_split",
    ):
        summary = mode_summaries[mode_name]
        recall = int(summary["required_boundary_edges_recovered"])
        control_damage = int(summary["current_good_boundary_controls_damaged"])
        if recall > baseline_required_edges or (
            recall >= baseline_required_edges
            and control_damage <= max(5, int(math.ceil(0.05 * max(1, len(control_edge_ids)))))
        ):
            downstream_selected_modes.append(mode_name)
    downstream_mode_rows: dict[str, list[dict[str, object]]] = {}
    downstream_funnels: dict[str, dict[str, object]] = {}
    for mode_index, mode_name in enumerate(downstream_selected_modes):
        rows, funnel = bounded_downstream_rows(
            mode_name,
            mode_clusters[mode_name],
            mode_index,
        )
        downstream_mode_rows[mode_name] = rows
        downstream_funnels[mode_name] = funnel
    if bool(mode_summaries["B_complete_linkage_bounded"]["identical_to_current"]):
        downstream_mode_rows["B_complete_linkage_bounded"] = downstream_mode_rows["A_current_complete"]
        downstream_funnels["B_complete_linkage_bounded"] = {
            **downstream_funnels["A_current_complete"],
            "reused_identical_control_rows": True,
        }
    downstream_selected_modes_with_control = sorted(downstream_mode_rows)
    control_face_ids = {
        int(row["oracle_face_id"])
        for row in downstream_mode_rows["A_current_complete"]
        if bool(row.get("finite_good"))
        and row.get("oracle_face_id") is not None
        and int(row["oracle_face_id"]) not in set(target_face_ids)
    }

    def summarize_downstream(
        rows: list[dict[str, object]],
        *,
        stable_only: bool,
    ) -> dict[str, object]:
        selected = [row for row in rows if not stable_only or bool(row.get("stable_strip"))]
        finite_rows = [row for row in selected if bool(row.get("finite_good"))]
        finite_ids = {
            int(row["oracle_face_id"])
            for row in finite_rows
            if row.get("oracle_face_id") is not None
        }
        recovered = sorted(int(value) for value in finite_ids & set(target_face_ids))
        return {
            "candidate_count": int(len(selected)),
            "finite_good_count": int(len(finite_rows)),
            "precision": as_json_float(float(len(finite_rows) / max(1, len(selected)))),
            "unique_finite_face_ids": int(len(finite_ids)),
            "recovered_target_count": int(len(recovered)),
            "recovered_target_face_ids": recovered,
            "current_control_face_count": int(len(control_face_ids)),
            "current_control_face_ids_lost": sorted(int(value) for value in control_face_ids - finite_ids),
            "current_control_faces_lost": int(len(control_face_ids - finite_ids)),
            "duplicate_assignments": int(len(finite_rows) - len(finite_ids)),
        }

    downstream_summaries: dict[str, dict[str, object]] = {}
    variant_internal_rows: dict[str, list[dict[str, object]]] = {}
    for mode_name in downstream_selected_modes_with_control:
        rows = downstream_mode_rows[mode_name]
        bounded_name = f"{mode_name}:bounded_k4"
        stable_name = f"{mode_name}:stable_strip"
        downstream_summaries[bounded_name] = {
            **summarize_downstream(rows, stable_only=False),
            "funnel": downstream_funnels[mode_name],
        }
        downstream_summaries[stable_name] = {
            **summarize_downstream(rows, stable_only=True),
            "funnel": {
                **downstream_funnels[mode_name],
                "stable_strip_pass": int(sum(bool(row.get("stable_strip")) for row in rows)),
            },
        }
        variant_internal_rows[bounded_name] = rows
        variant_internal_rows[stable_name] = [
            row for row in rows if bool(row.get("stable_strip"))
        ]
    timings["downstream_candidate_evaluation_seconds"] = time.perf_counter() - downstream_started

    candidate_separation_started = time.perf_counter()
    candidate_feature_directions = {
        "plane_rms": "lower",
        "separation_smoothness": "lower",
        "order_stable_fraction": "higher",
        "plane_normal_stability_deg": "lower",
        "local_observed_strip_support": "higher",
    }
    candidate_labeled_rows = [
        row
        for mode_name, rows in downstream_mode_rows.items()
        if mode_name != "B_complete_linkage_bounded"
        for row in rows
    ]
    candidate_separation: dict[str, object] = {}
    for feature, direction in candidate_feature_directions.items():
        values = []
        for row in candidate_labeled_rows:
            value = (
                finite_float(row["candidate"].get("plane_rms"), float("nan"))
                if feature == "plane_rms"
                else finite_float(row["features"].get(feature), float("nan"))
            )
            values.append((value, bool(row.get("finite_good"))))
        feature_auc = auc(values, direction)
        candidate_separation[feature] = {
            "direction": direction,
            "auc": as_json_float(feature_auc) if feature_auc is not None else None,
            "finite_good": distribution_summary(
                [value for value, label in values if label and np.isfinite(value)]
            ),
            "wrong": distribution_summary(
                [value for value, label in values if not label and np.isfinite(value)]
            ),
        }
    timings["candidate_nonoracle_separation_seconds"] = time.perf_counter() - candidate_separation_started

    downstream_variant_priority = {
        name: index for index, name in enumerate(downstream_summaries)
    }
    best_variant_name = max(
        downstream_summaries,
        key=lambda name: (
            int(downstream_summaries[name]["recovered_target_count"]),
            -int(downstream_summaries[name]["current_control_faces_lost"]),
            -int(downstream_summaries[name]["candidate_count"]),
            -int(downstream_variant_priority[name]),
        ),
    )
    representatives_by_face: dict[int, dict[str, object]] = {}
    for row in variant_internal_rows[best_variant_name]:
        face_id = row.get("oracle_face_id")
        if (
            not bool(row.get("finite_good"))
            or face_id is None
            or int(face_id) not in set(target_face_ids)
        ):
            continue
        previous = representatives_by_face.get(int(face_id))
        if previous is None or (
            finite_float(row.get("nonoracle_score"), float("inf")),
            row["pair"],
        ) < (
            finite_float(previous.get("nonoracle_score"), float("inf")),
            previous["pair"],
        ):
            representatives_by_face[int(face_id)] = row
    representatives = sorted(
        representatives_by_face.values(),
        key=lambda row: (
            finite_float(row.get("nonoracle_score"), float("inf")),
            row["pair"],
        ),
    )[:12]

    full_trial_started = time.perf_counter()
    baseline_active_indices = [
        int(index)
        for index in final_reconstructed.get("face_candidate_indices", [])
        if 0 <= int(index) < len(final_candidates)
    ]
    baseline_active_candidates = [final_candidates[index] for index in baseline_active_indices]
    baseline_finite_ids = {
        int(match["face_id"])
        for candidate in baseline_active_candidates
        for match in [candidate_oracle_face(candidate, model_faces)]
        if match is not None
        and bool(match.get("finite_good"))
        and match.get("face_id") is not None
    }

    def strict_patch_compatible(
        left: dict[str, object],
        right: dict[str, object],
    ) -> bool:
        angle, offset, centroid_distance = candidate_plane_relation_score(left, right)
        left_z = candidate_z_intervals(left)
        right_z = candidate_z_intervals(right)
        _, hull_fraction = interval_overlap(
            left_z.get("hull_z_min"),
            left_z.get("hull_z_max"),
            right_z.get("hull_z_min"),
            right_z.get("hull_z_max"),
        )
        _, support_fraction = interval_overlap(
            left_z.get("support_z_min"),
            left_z.get("support_z_max"),
            right_z.get("support_z_min"),
            right_z.get("support_z_max"),
        )
        return bool(
            angle <= 1.0
            and offset <= 0.02
            and centroid_distance <= 0.12
            and max(hull_fraction, support_fraction) >= 0.25
        )

    baseline_groups: list[list[dict[str, object]]] = []
    for candidate in baseline_active_candidates:
        for group in baseline_groups:
            if all(strict_patch_compatible(candidate, member) for member in group):
                group.append(candidate)
                break
        else:
            baseline_groups.append([candidate])
    z_slices = [
        np.flatnonzero(trusted_z_indices == int(value))
        for value in sorted(set(int(value) for value in trusted_z_indices))
    ] if trusted_z_indices.size else []
    full_trials: list[dict[str, object]] = []
    for trial_index, row in enumerate(representatives):
        alternative = dict(row["candidate"])
        alternative["track_id"] = int(9950000 + trial_index)
        trial_candidates = list(final_candidates) + [alternative]
        trial_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
            trial_candidates,
            **reconstruction_kwargs,
        )
        trial_active_indices = [
            int(index)
            for index in trial_reconstructed.get("face_candidate_indices", [])
            if 0 <= int(index) < len(trial_candidates)
        ]
        trial_active_candidates = [trial_candidates[index] for index in trial_active_indices]
        trial_ids = {
            int(match["face_id"])
            for candidate in trial_active_candidates
            for match in [candidate_oracle_face(candidate, model_faces)]
            if match is not None
            and bool(match.get("finite_good"))
            and match.get("face_id") is not None
        }
        lost_groups = [
            sorted(int(candidate_track_id(member)) for member in group)
            for group in baseline_groups
            if not any(
                strict_patch_compatible(member, trial_candidate)
                for member in group
                for trial_candidate in trial_active_candidates
            )
        ]
        gained = sorted(int(value) for value in trial_ids - baseline_finite_ids)
        lost = sorted(int(value) for value in baseline_finite_ids - trial_ids)
        target_face_id = int(row["oracle_face_id"])
        candidate_active = bool(len(trial_candidates) - 1 in trial_active_indices)
        target_active = bool(target_face_id in trial_ids)
        if not candidate_active:
            classification = "inactive"
        elif lost_groups:
            classification = "rolling_group_loss"
        elif gained and lost:
            classification = "replacement"
        elif gained and target_active and not lost:
            classification = "safe_plus_one"
        elif gained and not target_active:
            classification = "canonical_mapping_mismatch"
        else:
            classification = "redundant"
        outside = evaluate_trusted_cloud_outside(
            trial_candidates,
            trusted_points,
            z_slices,
            point_tol=float(point_tol),
        )
        full_trials.append(
            {
                "candidate_level_target_face_id": int(target_face_id),
                "cluster_mode_variant": str(best_variant_name),
                "cluster_ids": [int(value) for value in row["pair"]],
                "nonoracle_score": row.get("nonoracle_score"),
                "classification": classification,
                "candidate_active": candidate_active,
                "final_target_active": target_active,
                "canonical_ids_gained": gained,
                "canonical_ids_lost": lost,
                "retained": int(len(trial_ids & set(core_face_ids))),
                "new": int(len(trial_ids - set(core_face_ids))),
                "unique": int(len(trial_ids)),
                "rolling_groups_lost": int(len(lost_groups)),
                "lost_rolling_group_candidate_ids": lost_groups,
                "vertices": int(len(trial_reconstructed.get("vertices", []))),
                "edges": int(len(trial_reconstructed.get("edges", []))),
                "faces": int(len(trial_reconstructed.get("faces", []))),
                "volume": trial_reconstructed.get("reliable_volume"),
                "topology": trial_reconstructed.get("topology"),
                "topology_valid": topology_is_valid(
                    trial_reconstructed.get("topology")
                    if isinstance(trial_reconstructed.get("topology"), dict)
                    else None
                ),
                "outside": outside,
            }
        )
    timings["bounded_full_edge_clip_seconds"] = time.perf_counter() - full_trial_started

    classification_counts = {
        category: int(sum(row["classification"] == category for row in full_trials))
        for category in (
            "safe_plus_one",
            "replacement",
            "redundant",
            "inactive",
            "rolling_group_loss",
            "canonical_mapping_mismatch",
        )
    }
    current_required_recall = int(
        mode_summaries["A_current_complete"]["required_boundary_edges_recovered"]
    )
    cluster_mode_priority = {
        "A_current_complete": 0,
        "B_complete_linkage_bounded": 1,
        "C_representative_link": 2,
        "D_spatial_flow": 3,
        "E_post_cluster_split": 4,
    }
    best_cluster_mode_name = max(
        mode_summaries,
        key=lambda name: (
            int(mode_summaries[name]["required_boundary_edges_recovered"]),
            -int(mode_summaries[name]["current_good_boundary_controls_damaged"]),
            -int(mode_summaries[name]["cluster_count"]),
            -int(cluster_mode_priority[name]),
        ),
    )
    best_cluster_summary = mode_summaries[best_cluster_mode_name]
    edge_recall_gain = (
        int(best_cluster_summary["required_boundary_edges_recovered"])
        - current_required_recall
    )
    unavailable_edges = int(
        len(required_edge_ids)
        - mode_summaries["A_current_complete"]["required_boundary_edges_recovered"]
    )
    raw_loss_count = int(
        first_loss_counts["no_raw_support"]
        + first_loss_counts["short_Z_span"]
        + first_loss_counts["fixed_index_fragmentation"]
    )
    segment_loss_count = int(
        first_loss_counts["segment_fit_rejected"]
        + first_loss_counts["segment_itself_mixed"]
    )
    max_cluster_auc = max(
        (
            finite_float(row.get("auc"), 0.0)
            for row in cluster_separation.values()
        ),
        default=0.0,
    )
    best_recovered_faces = int(
        downstream_summaries[best_variant_name]["recovered_target_count"]
    )
    safe_plus_one = int(classification_counts["safe_plus_one"])
    if raw_loss_count >= max(1, math.ceil(0.5 * max(1, unavailable_edges))):
        selected_branch = "D_raw_W2_signal"
        branch_reason = "Most unavailable required edges already lack a contiguous fixed-index raw W2 support run."
    elif segment_loss_count >= max(1, math.ceil(0.5 * max(1, unavailable_edges))):
        selected_branch = "C_segment_formation"
        branch_reason = "Raw support exists, but required edges are predominantly lost or mixed before a pure accepted segment is formed."
    elif best_cluster_mode_name == "D_spatial_flow" and edge_recall_gain > 0:
        selected_branch = "B_spatial_flow_tracking"
        branch_reason = "Spatial-flow paths have the highest required-edge recall with bounded control damage."
    elif (
        best_cluster_mode_name
        in {"B_complete_linkage_bounded", "C_representative_link", "E_post_cluster_split"}
        and edge_recall_gain > 0
    ):
        if max_cluster_auc < 0.70:
            selected_branch = "E_separation_absent"
            branch_reason = "Oracle-evaluated splitting improves recall, but non-oracle impurity features do not provide a stable separator."
        else:
            selected_branch = "A_cluster_splitting"
            branch_reason = "A bounded non-oracle split/link mode improves required-edge recall and downstream recovery."
    elif best_recovered_faces >= 10 and safe_plus_one >= 2 and edge_recall_gain > 0:
        selected_branch = "A_cluster_splitting"
        branch_reason = "Alternative clustering improves the downstream face ceiling with final-mesh confirmations."
    else:
        selected_branch = "C_segment_formation"
        branch_reason = "Changing cluster grouping does not materially raise required-edge recall; the remaining loss is earlier than clustering."

    stage_segment_counts = {
        "least_squares": int(sum(segment_source(segment) == "least_squares" for segment in w2_segments)),
        "robust_inlier": int(sum(segment_source(segment) == "robust_inlier" for segment in w2_segments)),
        "one_split": int(sum(segment_source(segment) == "one_split" for segment in w2_segments)),
    }
    confident_segment_rows = [
        row
        for row in segment_oracle_by_id.values()
        if bool(row.get("oracle_confident_finite_edge"))
        and not (
            int(row.get("oracle_edge_id_count") or 0) >= 2
            and finite_float(row.get("dominant_edge_fraction"), 0.0) < 0.85
        )
    ]
    mixed_segment_rows = [
        row for row in segment_oracle_by_id.values()
        if int(row.get("oracle_edge_id_count") or 0) >= 2
        and finite_float(row.get("dominant_edge_fraction"), 0.0) < 0.85
    ]
    ambiguous_segment_count = int(
        len(w2_segments) - len(confident_segment_rows) - len(mixed_segment_rows)
    )
    timings["total_seconds"] = time.perf_counter() - started
    return {
        "diagnostic_only": True,
        "uses_initial_model": True,
        "production_selector_changed": False,
        "cohort": {
            "problem_face_count": int(len(target_face_ids)),
            "problem_face_ids": target_face_ids,
            "required_boundary_edge_count": int(len(required_edge_ids)),
            "required_boundary_edge_ids": required_edge_ids,
        },
        "production_cluster_semantics": {
            "requested_mode": str(cluster_mode),
            "actual_control_mode": "complete-linkage" if str(cluster_mode) == "complete" else str(cluster_mode),
            "single_linkage_used": False,
            "max_distance": as_json_float(float(cluster_max_distance)),
            "max_angle_deg": as_json_float(float(cluster_max_angle_deg)),
            "min_overlap": as_json_float(float(cluster_min_overlap)),
        },
        "boundary_edge_stage_funnel": {
            "first_loss_order": first_loss_order,
            "first_loss_counts": first_loss_counts,
            "sum": int(sum(first_loss_counts.values())),
            "expected": int(len(required_edge_ids)),
            "invariant_passed": bool(sum(first_loss_counts.values()) == len(required_edge_ids)),
            "rows": boundary_rows,
        },
        "segment_purity": {
            "segment_count": int(len(w2_segments)),
            "source_counts": stage_segment_counts,
            "confident_pure_segment_count": int(len(confident_segment_rows)),
            "mixed_segment_count": int(len(mixed_segment_rows)),
            "unmatched_or_ambiguous_segment_count": int(ambiguous_segment_count),
            "exclusive_sum": int(
                len(confident_segment_rows)
                + len(mixed_segment_rows)
                + ambiguous_segment_count
            ),
            "invariant_passed": bool(
                len(confident_segment_rows)
                + len(mixed_segment_rows)
                + ambiguous_segment_count
                == len(w2_segments)
            ),
            "required_edge_ids_with_confident_segments": sorted(
                {
                    int(row["dominant_edge_id"])
                    for row in confident_segment_rows
                    if row.get("dominant_edge_id") is not None
                    and int(row["dominant_edge_id"]) in required_edge_set
                }
            ),
            "mixed_segment_sample": [
                {
                    key: row.get(key)
                    for key in (
                        "segment_id",
                        "cyclic_index",
                        "segment_fit_mode",
                        "levels",
                        "z_span",
                        "dominant_edge_id",
                        "dominant_edge_fraction",
                        "oracle_edge_id_count",
                        "dominant_edge_distance_median",
                        "angle_to_dominant_edge_deg",
                    )
                }
                for row in mixed_segment_rows[:120]
            ],
        },
        "cluster_purity": {
            "current_mode": "A_current_complete",
            "mixed_case_counts": mixed_case_counts,
            "mixed_case_sum": int(sum(mixed_case_counts.values())),
            "current_cluster_count": int(len(mode_internal_rows["A_current_complete"])),
            "single_linkage_chaining_possible_in_current_mode": False,
            "rows": mixed_case_rows[:160],
        },
        "cluster_modes": mode_summaries,
        "cluster_nonoracle_separation": {
            "label_use": "oracle segment/edge IDs are evaluation-only; all feature values and alternative clustering are non-oracle",
            "cluster_features": cluster_separation,
            "downstream_candidate_features": candidate_separation,
            "maximum_cluster_feature_auc": as_json_float(max_cluster_auc),
        },
        "downstream_face_ceiling": {
            "bounded_k": 4,
            "max_pairs_per_mode": 6000,
            "max_pairs_per_cluster": 12,
            "selected_cluster_modes": downstream_selected_modes_with_control,
            "mode_selection": "control plus modes with required-edge recall gain, or equal recall with <=5%/5 control-edge damage",
            "current_control_face_ids": sorted(int(value) for value in control_face_ids),
            "variants": downstream_summaries,
            "best_variant": str(best_variant_name),
        },
        "bounded_full_edge_clip_verification": {
            "baseline_unique_finite_ids": int(len(baseline_finite_ids)),
            "selected_variant": str(best_variant_name),
            "one_representative_per_recovered_face": True,
            "trial_limit": 12,
            "trial_count": int(len(full_trials)),
            "classification_counts": classification_counts,
            "rows": full_trials,
        },
        "branch_decision": {
            "selected": selected_branch,
            "reason": branch_reason,
            "current_required_boundary_edges": int(current_required_recall),
            "best_cluster_mode": str(best_cluster_mode_name),
            "best_required_boundary_edges": int(
                best_cluster_summary["required_boundary_edges_recovered"]
            ),
            "required_boundary_edge_recall_gain": int(edge_recall_gain),
            "best_downstream_variant": str(best_variant_name),
            "best_downstream_recovered_faces": int(best_recovered_faces),
            "safe_final_mesh_plus_one": int(safe_plus_one),
            "maximum_cluster_feature_auc": as_json_float(max_cluster_auc),
            "production_fix_implemented": False,
        },
        "timings_seconds": {key: as_json_float(float(value)) for key, value in timings.items()},
    }


def diagnose_w2_raw_signal_failures(
    *,
    cluster_audit: dict[str, object],
    initial_vertices: np.ndarray,
    initial_faces: list[list[int]],
    model_faces: list[dict[str, object]],
    z_levels: np.ndarray,
    w2_line_points: np.ndarray | None,
    w2_fit_rms: np.ndarray | None,
    w2_condition: np.ndarray | None,
    w3_line_points: np.ndarray | None,
    w3_fit_rms: np.ndarray | None,
    w4_line_points: np.ndarray | None,
    w4_fit_rms: np.ndarray | None,
    valley_raw_candidates: list[dict[str, object]],
    w2_segments: list[dict[str, object]],
    w2_clusters: list[dict[str, object]],
    w2_raw_candidates: list[dict[str, object]],
    edge_ancestry_internal: dict[str, object],
    segment_min_levels: int,
    segment_min_z_span: float,
    segment_max_z_gap: int,
    segment_max_line_rms: float,
    segment_max_line_residual: float,
    segment_max_point_jump: float,
    segment_max_condition: float,
    cluster_mode: str,
    cluster_max_distance: float,
    cluster_max_angle_deg: float,
    cluster_min_overlap: float,
    face_min_adjacency_levels: int,
    face_min_adjacency_fraction: float,
    face_min_z_span: float,
    face_max_plane_rms: float,
    inside_point: np.ndarray,
    orientation_points: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    core_face_ids: set[int],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    contours: list[object],
) -> dict[str, object]:
    """Oracle-only raw W2 recoverability audit over immutable production arrays."""

    started = time.perf_counter()
    timings: dict[str, float] = {}
    if w2_line_points is None or w2_fit_rms is None:
        return {
            "diagnostic_only": True,
            "uses_initial_model": True,
            "production_selector_changed": False,
            "skipped": True,
            "reason": "W2 line points or W2 RMS values are unavailable",
        }

    line_points = np.asarray(w2_line_points, dtype=float)
    fit_rms = np.asarray(w2_fit_rms, dtype=float)
    condition = (
        np.asarray(w2_condition, dtype=float)
        if w2_condition is not None
        else np.full(fit_rms.shape, np.nan, dtype=float)
    )
    w3_points = (
        np.asarray(w3_line_points, dtype=float)
        if w3_line_points is not None
        else None
    )
    w3_rms = (
        np.asarray(w3_fit_rms, dtype=float)
        if w3_fit_rms is not None
        else None
    )
    w4_points = (
        np.asarray(w4_line_points, dtype=float)
        if w4_line_points is not None
        else None
    )
    w4_rms = (
        np.asarray(w4_fit_rms, dtype=float)
        if w4_fit_rms is not None
        else None
    )
    n_half, n_levels = int(line_points.shape[0]), int(line_points.shape[1])
    edges, edge_face_lookup = model_edges_from_faces(initial_vertices, initial_faces)
    edge_id_to_faces = {
        int(edge_id): set(
            int(value)
            for value in edge_face_lookup.get(tuple(sorted(edge)), set())
        )
        for edge_id, edge in enumerate(edges)
    }
    funnel_rows = list((cluster_audit.get("boundary_edge_stage_funnel") or {}).get("rows") or [])
    short_edge_ids = sorted(
        int(row["edge_id"])
        for row in funnel_rows
        if str(row.get("first_loss")) == "short_Z_span"
    )
    mixed_required_edge_ids = sorted(
        int(row["edge_id"])
        for row in funnel_rows
        if str(row.get("first_loss")) == "segment_itself_mixed"
    )
    available_required_edge_ids = sorted(
        int(row["edge_id"])
        for row in funnel_rows
        if str(row.get("first_loss")) == "available_confident_cluster"
    )
    required_edge_ids = sorted(
        int(value)
        for value in (cluster_audit.get("cohort") or {}).get("required_boundary_edge_ids", [])
    )
    target_face_ids = sorted(
        int(value)
        for value in (cluster_audit.get("cohort") or {}).get("problem_face_ids", [])
    )
    immutable_short_cohort_matches = bool(len(short_edge_ids) == 49)

    def canonical_direction(raw: np.ndarray) -> np.ndarray:
        direction = np.asarray(raw, dtype=float).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm <= EPS:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        direction = direction / norm
        first = next(
            (float(value) for value in direction if abs(float(value)) > 1e-12),
            1.0,
        )
        return -direction if first < 0.0 else direction

    def angle_between(left: np.ndarray, right: np.ndarray) -> float:
        a = canonical_direction(left)
        b = canonical_direction(right)
        return float(
            np.degrees(
                np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))
            )
        )

    def orthogonal_line_fit(points: np.ndarray) -> dict[str, object] | None:
        pts = np.asarray(points, dtype=float).reshape((-1, 3))
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] < 2:
            return None
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        try:
            _, singular, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        direction = canonical_direction(vh[0])
        projection = centered @ direction
        residual = np.linalg.norm(
            centered - projection[:, None] * direction[None, :],
            axis=1,
        )
        eigenvalues = np.zeros(3, dtype=float)
        eigenvalues[: min(3, singular.size)] = (
            singular[:3] * singular[:3] / max(1, pts.shape[0] - 1)
        )
        endpoints = np.vstack(
            [
                centroid + float(np.min(projection)) * direction,
                centroid + float(np.max(projection)) * direction,
            ]
        )
        return {
            "centroid": centroid,
            "direction": direction,
            "endpoints": endpoints,
            "projection": projection,
            "residuals": residual,
            "rms": float(np.sqrt(np.mean(residual * residual))),
            "max_residual": float(np.max(residual)),
            "eigenvalues": eigenvalues,
            "primary_secondary_ratio": float(
                eigenvalues[0] / max(eigenvalues[1], 1e-12)
            ),
            "finite_length": float(np.ptp(projection)),
        }

    def observed_local_direction(
        point_keys: list[tuple[int, int]],
        points: np.ndarray,
    ) -> np.ndarray | None:
        direction_matrix = np.zeros((3, 3), dtype=float)
        contributions = 0
        key_to_point = {
            (int(cyclic), int(z_index)): np.asarray(point, dtype=float)
            for (cyclic, z_index), point in zip(point_keys, points)
        }
        for (cyclic, z_index), point in zip(point_keys, points):
            left = line_points[(int(cyclic) - 1) % n_half, int(z_index)]
            right = line_points[(int(cyclic) + 1) % n_half, int(z_index)]
            if np.all(np.isfinite(left)) and np.all(np.isfinite(right)):
                vector = right - left
                norm = float(np.linalg.norm(vector))
                if norm > EPS:
                    unit = vector / norm
                    direction_matrix += np.outer(unit, unit)
                    contributions += 1
            for delta in (-1, 1):
                neighbor = key_to_point.get((int(cyclic), int(z_index) + delta))
                if neighbor is None:
                    continue
                vector = neighbor - point
                norm = float(np.linalg.norm(vector))
                if norm > EPS:
                    unit = vector / norm
                    direction_matrix += np.outer(unit, unit)
                    contributions += 1
        if contributions == 0:
            return None
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(direction_matrix)
        except np.linalg.LinAlgError:
            return None
        return canonical_direction(eigenvectors[:, int(np.argmax(eigenvalues))])

    def fitted_endpoints(
        points: np.ndarray,
        centroid: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        projection = (points - centroid[None, :]) @ direction
        return np.vstack(
            [
                centroid + float(np.min(projection)) * direction,
                centroid + float(np.max(projection)) * direction,
            ]
        )

    def endpoint_pair_distance(left: np.ndarray, right: np.ndarray) -> float:
        direct = max(
            float(np.linalg.norm(left[0] - right[0])),
            float(np.linalg.norm(left[1] - right[1])),
        )
        reverse = max(
            float(np.linalg.norm(left[0] - right[1])),
            float(np.linalg.norm(left[1] - right[0])),
        )
        return float(min(direct, reverse))

    def fit_representation(
        *,
        points: np.ndarray,
        point_keys: list[tuple[int, int]],
        mode: str,
        compute_stability: bool = True,
    ) -> dict[str, object] | None:
        pts = np.asarray(points, dtype=float).reshape((-1, 3))
        if pts.shape[0] < 2 or not np.all(np.isfinite(pts)):
            return None
        z = pts[:, 2]
        if str(mode) == "current_x_y_of_z":
            fit = fit_xy_line_vs_z(z, pts)
            if fit is None:
                return None
            direction = canonical_direction(np.asarray(fit["direction"], dtype=float))
            centroid = np.mean(pts, axis=0)
            endpoints = np.vstack(
                [
                    predict_w2_line(fit, np.array([float(np.min(z))]))[0],
                    predict_w2_line(fit, np.array([float(np.max(z))]))[0],
                ]
            )
            predicted = predict_w2_line(fit, z)
            residual = np.linalg.norm(pts - predicted, axis=1)
            design = np.column_stack([z, np.ones_like(z)])
            centered_design = np.column_stack([z - float(np.mean(z)), np.ones_like(z)])
            condition_raw = float(np.linalg.cond(design))
            condition_centered = float(np.linalg.cond(centered_design))
            tls = orthogonal_line_fit(pts)
            eigen_ratio = finite_float(
                (tls or {}).get("primary_secondary_ratio"),
                0.0,
            )
        else:
            tls = orthogonal_line_fit(pts)
            if tls is None:
                return None
            centroid = np.asarray(tls["centroid"], dtype=float)
            direction = np.asarray(tls["direction"], dtype=float)
            if str(mode) == "locally_regularized_short_span":
                local_direction = observed_local_direction(point_keys, pts)
                if local_direction is not None:
                    direction = local_direction
            direction = canonical_direction(direction)
            endpoints = fitted_endpoints(pts, centroid, direction)
            projection = (pts - centroid[None, :]) @ direction
            residual = np.linalg.norm(
                pts - (centroid[None, :] + projection[:, None] * direction[None, :]),
                axis=1,
            )
            condition_raw = finite_float(tls.get("primary_secondary_ratio"), 0.0)
            condition_centered = condition_raw
            eigen_ratio = finite_float(tls.get("primary_secondary_ratio"), 0.0)
        direction_stability = 0.0
        endpoint_stability = 0.0
        if compute_stability and pts.shape[0] >= 3:
            leave_indices = list(range(pts.shape[0]))
            if len(leave_indices) > 10:
                leave_indices = [
                    int(value)
                    for value in np.linspace(0, pts.shape[0] - 1, 10, dtype=int)
                ]
            for leave_index in leave_indices:
                mask = np.ones(pts.shape[0], dtype=bool)
                mask[int(leave_index)] = False
                sub_keys = [key for index, key in enumerate(point_keys) if mask[index]]
                sub = fit_representation(
                    points=pts[mask],
                    point_keys=sub_keys,
                    mode=mode,
                    compute_stability=False,
                )
                if sub is None:
                    continue
                direction_stability = max(
                    direction_stability,
                    angle_between(direction, np.asarray(sub["direction"], dtype=float)),
                )
                endpoint_stability = max(
                    endpoint_stability,
                    endpoint_pair_distance(endpoints, np.asarray(sub["endpoints"], dtype=float)),
                )
        return {
            "mode": str(mode),
            "centroid": centroid,
            "direction": direction,
            "endpoints": endpoints,
            "orthogonal_residual_rms": float(np.sqrt(np.mean(residual * residual))),
            "orthogonal_residual_p95": float(np.percentile(residual, 95)),
            "orthogonal_residual_max": float(np.max(residual)),
            "condition": float(condition_raw),
            "centered_condition": float(condition_centered),
            "eigenvalue_primary_secondary_ratio": float(eigen_ratio),
            "finite_segment_length": float(np.linalg.norm(endpoints[1] - endpoints[0])),
            "direction_stability_deg": float(direction_stability),
            "endpoint_stability": float(endpoint_stability),
        }

    def point_keys_for_segment(segment: dict[str, object]) -> list[tuple[int, int]]:
        cyclic_value = segment.get("cyclic_index")
        if cyclic_value is None:
            return []
        cyclic = int(cyclic_value)
        return [
            (cyclic, int(z_index))
            for z_index in segment.get("z_indices", [])
            if 0 <= int(z_index) < n_levels
            and np.all(np.isfinite(line_points[cyclic, int(z_index)]))
        ]

    def points_for_keys(keys: list[tuple[int, int]]) -> np.ndarray:
        if not keys:
            return np.zeros((0, 3), dtype=float)
        return np.array(
            [line_points[int(cyclic), int(z_index)] for cyclic, z_index in keys],
            dtype=float,
        ).reshape((-1, 3))

    def hypothesis_scale_features(keys: list[tuple[int, int]]) -> dict[str, object]:
        agreements: list[float] = []
        w3_minima: list[bool] = []
        for cyclic, z_index in keys:
            if w3_points is not None:
                point = w3_points[int(cyclic), int(z_index)]
                if np.all(np.isfinite(point)):
                    agreements.append(
                        float(
                            np.linalg.norm(
                                point[:2] - line_points[int(cyclic), int(z_index), :2]
                            )
                        )
                    )
            if w3_rms is not None:
                value = w3_rms[int(cyclic), int(z_index)]
                if np.isfinite(value):
                    previous = w3_rms[(int(cyclic) - 1) % n_half, int(z_index)]
                    following = w3_rms[(int(cyclic) + 1) % n_half, int(z_index)]
                    w3_minima.append(bool(value <= previous and value <= following))
        return {
            "w2_w3_agreement_median": (
                as_json_float(float(np.median(agreements))) if agreements else None
            ),
            "scale_persistence_fraction": (
                as_json_float(float(np.mean(w3_minima))) if w3_minima else None
            ),
        }

    hypothesis_counter = 0

    def make_hypothesis(
        *,
        keys: list[tuple[int, int]],
        representation: str,
        source: str,
        fit_mode: str,
        extra: dict[str, object] | None = None,
        compute_stability: bool = False,
    ) -> dict[str, object] | None:
        nonlocal hypothesis_counter
        unique_keys = sorted(set((int(cyclic), int(z_index)) for cyclic, z_index in keys))
        points = points_for_keys(unique_keys)
        fit = fit_representation(
            points=points,
            point_keys=unique_keys,
            mode=fit_mode,
            compute_stability=bool(compute_stability),
        )
        if fit is None:
            return None
        stability_method = "leave_one_observation_out" if compute_stability else "two_interleaved_subsets"
        if not compute_stability and points.shape[0] >= 4:
            subset_fits = []
            for subset_indices in (
                np.arange(0, points.shape[0], 2, dtype=int),
                np.arange(1, points.shape[0], 2, dtype=int),
            ):
                if subset_indices.size < 2:
                    continue
                subset = fit_representation(
                    points=points[subset_indices],
                    point_keys=[unique_keys[int(index)] for index in subset_indices],
                    mode=fit_mode,
                    compute_stability=False,
                )
                if subset is not None:
                    subset_fits.append(subset)
            if len(subset_fits) == 2:
                fit["direction_stability_deg"] = angle_between(
                    np.asarray(subset_fits[0]["direction"], dtype=float),
                    np.asarray(subset_fits[1]["direction"], dtype=float),
                )
                fit["endpoint_stability"] = endpoint_pair_distance(
                    np.asarray(subset_fits[0]["endpoints"], dtype=float),
                    np.asarray(subset_fits[1]["endpoints"], dtype=float),
                )
        z_indices = sorted({int(z_index) for _, z_index in unique_keys})
        cyclic_indices = sorted({int(cyclic) for cyclic, _ in unique_keys})
        scale = hypothesis_scale_features(unique_keys)
        row = {
            "hypothesis_id": int(hypothesis_counter),
            "representation": str(representation),
            "extent_semantics": (
                "infinite_tls_line"
                if str(representation) == "orthogonal_3d_tls"
                else "observed_projection_interval"
            ),
            "source": str(source),
            "point_keys": [[int(cyclic), int(z_index)] for cyclic, z_index in unique_keys],
            "point_count": int(len(unique_keys)),
            "levels": int(len(z_indices)),
            "z_indices": z_indices,
            "z_min": as_json_float(float(np.min(points[:, 2]))),
            "z_max": as_json_float(float(np.max(points[:, 2]))),
            "z_span": as_json_float(float(np.ptp(points[:, 2]))),
            "cyclic_indices": cyclic_indices,
            "cyclic_span": int(circular_span(cyclic_indices, n_half)),
            "centroid": as_json_point(np.asarray(fit["centroid"], dtype=float)),
            "direction": as_json_point(np.asarray(fit["direction"], dtype=float)),
            "endpoints": [
                as_json_point(point)
                for point in np.asarray(fit["endpoints"], dtype=float)
            ],
            "orthogonal_residual_rms": as_json_float(
                finite_float(fit.get("orthogonal_residual_rms"), 0.0)
            ),
            "orthogonal_residual_p95": as_json_float(
                finite_float(fit.get("orthogonal_residual_p95"), 0.0)
            ),
            "orthogonal_residual_max": as_json_float(
                finite_float(fit.get("orthogonal_residual_max"), 0.0)
            ),
            "condition": as_json_float(finite_float(fit.get("condition"), 0.0)),
            "centered_condition": as_json_float(
                finite_float(fit.get("centered_condition"), 0.0)
            ),
            "eigenvalue_primary_secondary_ratio": as_json_float(
                finite_float(fit.get("eigenvalue_primary_secondary_ratio"), 0.0)
            ),
            "finite_segment_length": as_json_float(
                finite_float(fit.get("finite_segment_length"), 0.0)
            ),
            "direction_stability_deg": as_json_float(
                finite_float(fit.get("direction_stability_deg"), 0.0)
            ),
            "endpoint_stability": as_json_float(
                finite_float(fit.get("endpoint_stability"), 0.0)
            ),
            "stability_method": str(stability_method),
            **scale,
            **dict(extra or {}),
            "_points": points,
        }
        hypothesis_counter += 1
        return row

    def circular_span(values: list[int], modulus: int) -> int:
        unique = sorted({int(value) % max(1, int(modulus)) for value in values})
        if len(unique) <= 1:
            return 0
        gaps = [
            int((unique[(index + 1) % len(unique)] - unique[index]) % modulus)
            for index in range(len(unique))
        ]
        return int(modulus - max(gaps))

    def fixed_runs(values: list[int], max_gap: int) -> list[list[int]]:
        ordered = sorted(set(int(value) for value in values))
        if not ordered:
            return []
        runs = [[ordered[0]]]
        for value in ordered[1:]:
            if int(value) - int(runs[-1][-1]) <= int(max_gap):
                runs[-1].append(int(value))
            else:
                runs.append([int(value)])
        return runs

    assignment_started = time.perf_counter()
    observations_by_edge: list[list[dict[str, object]]] = [
        [] for _ in edges
    ]
    valid_indices = np.argwhere(np.all(np.isfinite(line_points), axis=2))
    valid_points = line_points[valid_indices[:, 0], valid_indices[:, 1]]
    for start_index in range(0, valid_points.shape[0], 512):
        chunk = valid_points[start_index : start_index + 512]
        chunk_indices = valid_indices[start_index : start_index + 512]
        distances, _ = point_edge_distances_vectorized(
            chunk,
            initial_vertices,
            edges,
        )
        nearest = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(distances.shape[0]), nearest]
        for point, indices, edge_id, distance in zip(
            chunk,
            chunk_indices,
            nearest,
            nearest_distance,
        ):
            if float(distance) > 0.08:
                continue
            cyclic, z_index = int(indices[0]), int(indices[1])
            observations_by_edge[int(edge_id)].append(
                {
                    "cyclic_index": cyclic,
                    "z_index": z_index,
                    "point": np.asarray(point, dtype=float),
                    "distance": float(distance),
                    "condition": finite_float(condition[cyclic, z_index], float("nan")),
                    "w2_rms": finite_float(fit_rms[cyclic, z_index], float("nan")),
                }
            )
    timings["raw_point_oracle_assignment_seconds"] = time.perf_counter() - assignment_started

    def edge_observed_runs(edge_id: int) -> list[list[tuple[int, int]]]:
        by_cyclic: dict[int, list[int]] = {}
        for observation in observations_by_edge[int(edge_id)]:
            by_cyclic.setdefault(int(observation["cyclic_index"]), []).append(
                int(observation["z_index"])
            )
        return [
            [(int(cyclic), int(z_index)) for z_index in run]
            for cyclic, values in sorted(by_cyclic.items())
            for run in fixed_runs(values, int(segment_max_z_gap))
        ]

    def best_edge_run(edge_id: int) -> list[tuple[int, int]]:
        runs = edge_observed_runs(int(edge_id))
        if not runs:
            return []
        return max(
            runs,
            key=lambda keys: (
                len(keys),
                float(
                    z_levels[max(z_index for _, z_index in keys)]
                    - z_levels[min(z_index for _, z_index in keys)]
                ),
                -int(keys[0][0]),
            ),
        )

    def target_edge_metrics(
        fit: dict[str, object],
        points: np.ndarray,
        edge_id: int,
    ) -> dict[str, object]:
        edge = edges[int(edge_id)]
        start = initial_vertices[int(edge[0])]
        end = initial_vertices[int(edge[1])]
        direction = end - start
        length = float(np.linalg.norm(direction))
        fit_direction = np.asarray(fit["direction"], dtype=float)
        distances, t_values = point_edge_distances_vectorized(
            points,
            initial_vertices,
            [edge],
        )
        return {
            "target_edge_angle_deg": as_json_float(
                angle_between(fit_direction, direction)
            ) if length > EPS else None,
            "target_edge_distance_median": as_json_float(
                float(np.median(distances[:, 0]))
            ),
            "target_edge_distance_p95": as_json_float(
                float(np.percentile(distances[:, 0], 95))
            ),
            "target_edge_inside_fraction": as_json_float(
                float(np.mean((t_values[:, 0] >= 0.0) & (t_values[:, 0] <= 1.0)))
            ),
            "target_edge_t_coverage": as_json_float(
                float(
                    np.ptp(np.clip(t_values[:, 0], 0.0, 1.0))
                )
            ),
        }

    anatomy_started = time.perf_counter()
    anatomy_categories = [
        "physically_short_or_horizontal_edge",
        "true_span_sufficient_but_observed_levels_missing",
        "support_exists_but_current_min_span_rejects",
        "support_fragmented_by_detection",
        "support_contaminated_or_false",
    ]
    anatomy_counts = {category: 0 for category in anatomy_categories}
    anatomy_rows: list[dict[str, object]] = []
    for edge_id in short_edge_ids:
        edge = edges[int(edge_id)]
        start = initial_vertices[int(edge[0])]
        end = initial_vertices[int(edge[1])]
        edge_vector = end - start
        true_length = float(np.linalg.norm(edge_vector))
        true_z_span = float(abs(edge_vector[2]))
        true_xy_span = float(np.linalg.norm(edge_vector[:2]))
        true_horizontal_angle = (
            float(np.degrees(np.arcsin(np.clip(true_z_span / true_length, 0.0, 1.0))))
            if true_length > EPS
            else 0.0
        )
        true_level_count = int(
            np.sum(
                (z_levels >= min(float(start[2]), float(end[2])) - EPS)
                & (z_levels <= max(float(start[2]), float(end[2])) + EPS)
            )
        )
        observations = observations_by_edge[int(edge_id)]
        observed_points = np.array(
            [row["point"] for row in observations],
            dtype=float,
        ).reshape((-1, 3))
        observed_z = sorted({int(row["z_index"]) for row in observations})
        observed_cyclic = sorted({int(row["cyclic_index"]) for row in observations})
        runs = edge_observed_runs(int(edge_id))
        best_run = best_edge_run(int(edge_id))
        best_run_points = points_for_keys(best_run)
        tls = orthogonal_line_fit(observed_points)
        best_tls = orthogonal_line_fit(best_run_points)
        points_per_level = [
            int(sum(int(row["z_index"]) == int(z_index) for row in observations))
            for z_index in observed_z
        ]
        gaps = [
            int(right - left)
            for left, right in zip(observed_z[:-1], observed_z[1:])
            if int(right - left) > 1
        ]
        observed_span = (
            float(z_levels[max(observed_z)] - z_levels[min(observed_z)])
            if observed_z else 0.0
        )
        contaminated = bool(
            best_tls is None
            or angle_between(np.asarray(best_tls["direction"], dtype=float), edge_vector) > 20.0
            or finite_float((best_tls or {}).get("rms"), float("inf")) > 0.08
        )
        if true_z_span < float(segment_min_z_span) or true_level_count < int(segment_min_levels):
            category = "physically_short_or_horizontal_edge"
        elif contaminated:
            category = "support_contaminated_or_false"
        elif len(observed_z) >= int(segment_min_levels) and observed_span < float(segment_min_z_span):
            category = "support_exists_but_current_min_span_rejects"
        elif len(observed_z) >= int(segment_min_levels) and max((len(run) for run in runs), default=0) < int(segment_min_levels):
            category = "support_fragmented_by_detection"
        else:
            category = "true_span_sufficient_but_observed_levels_missing"
        anatomy_counts[category] += 1
        anatomy_rows.append(
            {
                "edge_id": int(edge_id),
                "incident_target_face_ids": next(
                    (
                        list(row.get("incident_target_face_ids") or [])
                        for row in funnel_rows
                        if row.get("edge_id") is not None
                        and int(row["edge_id"]) == int(edge_id)
                    ),
                    [],
                ),
                "true_edge_vertices": [int(edge[0]), int(edge[1])],
                "true_edge_length": as_json_float(true_length),
                "true_z_span": as_json_float(true_z_span),
                "true_xy_span": as_json_float(true_xy_span),
                "angle_to_horizontal_plane_deg": as_json_float(true_horizontal_angle),
                "model_z_level_intersections": int(true_level_count),
                "raw_support_count": int(len(observations)),
                "supported_z_levels": int(len(observed_z)),
                "longest_contiguous_run": int(max((len(run) for run in runs), default=0)),
                "observed_z_span": as_json_float(observed_span),
                "observed_xy_span": as_json_float(
                    float(np.linalg.norm(np.ptp(observed_points[:, :2], axis=0)))
                ) if observed_points.size else 0.0,
                "observed_3d_span": as_json_float(
                    finite_float((tls or {}).get("finite_length"), 0.0)
                ),
                "cyclic_index_span": int(circular_span(observed_cyclic, n_half)),
                "points_per_level": {
                    "min": int(min(points_per_level)) if points_per_level else 0,
                    "median": as_json_float(float(np.median(points_per_level))) if points_per_level else None,
                    "max": int(max(points_per_level)) if points_per_level else 0,
                },
                "gap_pattern": gaps,
                "condition_median": as_json_float(
                    float(np.nanmedian([row["condition"] for row in observations]))
                ) if observations else None,
                "condition_p95": as_json_float(
                    float(np.nanpercentile([row["condition"] for row in observations], 95))
                ) if observations else None,
                "best_run_levels": int(len(best_run)),
                "best_run_z_span": as_json_float(
                    float(np.ptp(best_run_points[:, 2]))
                ) if best_run_points.size else 0.0,
                "best_run_orthogonal_rms": as_json_float(
                    finite_float((best_tls or {}).get("rms"), float("nan"))
                ),
                "best_run_target_angle_deg": as_json_float(
                    angle_between(np.asarray(best_tls["direction"], dtype=float), edge_vector)
                ) if best_tls is not None else None,
                "current_thresholds": {
                    "min_levels": int(segment_min_levels),
                    "min_z_span": as_json_float(float(segment_min_z_span)),
                    "max_line_rms": as_json_float(float(segment_max_line_rms)),
                    "max_line_residual": as_json_float(float(segment_max_line_residual)),
                    "max_z_gap": int(segment_max_z_gap),
                },
                "current_rejection_reason": "short_Z_span",
                "anatomy_category": category,
            }
        )
    timings["short_z_anatomy_seconds"] = time.perf_counter() - anatomy_started

    segment_oracle_rows = list(edge_ancestry_internal.get("segment_oracle_rows", []))
    ordinary_control_edge_ids = sorted(
        {
            int(row["dominant_edge_id"])
            for row in segment_oracle_rows
            if row.get("dominant_edge_id") is not None
            and bool(row.get("oracle_confident_finite_edge"))
            and not (
                int(row.get("oracle_edge_id_count") or 0) >= 2
                and finite_float(row.get("dominant_edge_fraction"), 0.0) < 0.85
            )
            and int(row["dominant_edge_id"]) not in set(required_edge_ids)
        }
    )[:40]

    representation_started = time.perf_counter()
    representation_modes = [
        "current_x_y_of_z",
        "orthogonal_3d_tls",
        "finite_3d_segment",
        "locally_regularized_short_span",
    ]
    representation_rows: list[dict[str, object]] = []
    representation_edge_sets = {
        "short_z": short_edge_ids,
        "available_required": available_required_edge_ids,
        "ordinary_matched": ordinary_control_edge_ids,
    }
    for cohort_name, edge_ids in representation_edge_sets.items():
        for edge_id in edge_ids:
            observations = observations_by_edge[int(edge_id)]
            if not observations:
                continue
            best_run = best_edge_run(int(edge_id))
            densest_levels = sorted(
                {
                    int(row["z_index"])
                    for row in observations
                },
                key=lambda z_index: (
                    -sum(int(row["z_index"]) == int(z_index) for row in observations),
                    int(z_index),
                ),
            )[:3]
            local_keys = sorted(
                {
                    (int(row["cyclic_index"]), int(row["z_index"]))
                    for row in observations
                    if int(row["z_index"]) in set(densest_levels)
                }
            )
            for mode in representation_modes:
                keys = local_keys if mode == "locally_regularized_short_span" else best_run
                if len(keys) < 2:
                    continue
                points = points_for_keys(keys)
                fit = fit_representation(points=points, point_keys=keys, mode=mode)
                if fit is None:
                    continue
                representation_rows.append(
                    {
                        "cohort": str(cohort_name),
                        "edge_id": int(edge_id),
                        "representation": str(mode),
                        "point_count": int(points.shape[0]),
                        "levels": int(len({z_index for _, z_index in keys})),
                        "z_span": as_json_float(float(np.ptp(points[:, 2]))),
                        "orthogonal_residual_rms": as_json_float(
                            finite_float(fit.get("orthogonal_residual_rms"), 0.0)
                        ),
                        "orthogonal_residual_p95": as_json_float(
                            finite_float(fit.get("orthogonal_residual_p95"), 0.0)
                        ),
                        "condition": as_json_float(finite_float(fit.get("condition"), 0.0)),
                        "centered_condition": as_json_float(
                            finite_float(fit.get("centered_condition"), 0.0)
                        ),
                        "eigenvalue_primary_secondary_ratio": as_json_float(
                            finite_float(fit.get("eigenvalue_primary_secondary_ratio"), 0.0)
                        ),
                        "finite_segment_length": as_json_float(
                            finite_float(fit.get("finite_segment_length"), 0.0)
                        ),
                        "direction_stability_deg": as_json_float(
                            finite_float(fit.get("direction_stability_deg"), 0.0)
                        ),
                        "endpoint_stability": as_json_float(
                            finite_float(fit.get("endpoint_stability"), 0.0)
                        ),
                        **target_edge_metrics(fit, points, int(edge_id)),
                        **hypothesis_scale_features(keys),
                    }
                )

    def summarize_representation(mode: str, cohort_name: str) -> dict[str, object]:
        rows = [
            row for row in representation_rows
            if str(row["representation"]) == str(mode)
            and str(row["cohort"]) == str(cohort_name)
        ]
        matched = [
            row for row in rows
            if finite_float(row.get("target_edge_angle_deg"), 180.0) <= 12.0
            and finite_float(row.get("target_edge_distance_median"), float("inf")) <= 0.08
            and finite_float(row.get("target_edge_inside_fraction"), 0.0) >= 0.5
        ]
        return {
            "edge_fits": int(len(rows)),
            "target_edge_matches": int(len(matched)),
            "target_edge_match_rate": as_json_float(float(len(matched) / max(1, len(rows)))),
            "orthogonal_residual_rms": distribution_summary(
                [row.get("orthogonal_residual_rms") for row in rows]
            ),
            "condition": distribution_summary([row.get("condition") for row in rows]),
            "direction_stability_deg": distribution_summary(
                [row.get("direction_stability_deg") for row in rows]
            ),
            "endpoint_stability": distribution_summary(
                [row.get("endpoint_stability") for row in rows]
            ),
            "target_edge_angle_deg": distribution_summary(
                [row.get("target_edge_angle_deg") for row in rows]
            ),
            "target_edge_distance_median": distribution_summary(
                [row.get("target_edge_distance_median") for row in rows]
            ),
        }

    representation_summary = {
        mode: {
            cohort_name: summarize_representation(mode, cohort_name)
            for cohort_name in representation_edge_sets
        }
        for mode in representation_modes
    }
    for mode in representation_modes:
        controls = [
            row
            for row in representation_rows
            if str(row["representation"]) == str(mode)
            and str(row["cohort"]) in {"available_required", "ordinary_matched"}
        ]
        representation_summary[mode]["current_good_controls_lost"] = int(
            sum(
                finite_float(row.get("target_edge_angle_deg"), 180.0) > 12.0
                or finite_float(row.get("target_edge_distance_median"), float("inf")) > 0.08
                or finite_float(row.get("target_edge_inside_fraction"), 0.0) < 0.5
                for row in controls
            )
        )
    timings["representation_comparison_seconds"] = time.perf_counter() - representation_started

    oracle_edge_cache: dict[tuple[object, ...], dict[str, object]] = {}

    def oracle_label(
        *,
        points: np.ndarray,
        direction: np.ndarray,
        point_keys: list[tuple[int, int]],
    ) -> dict[str, object]:
        signature = (
            tuple(sorted((int(cyclic), int(z_index)) for cyclic, z_index in point_keys)),
            tuple(round(float(value), 9) for value in canonical_direction(direction)),
        )
        if signature not in oracle_edge_cache:
            match = oracle_run_edge_match(
                points=np.asarray(points, dtype=float).reshape((-1, 3)),
                fit={"direction": canonical_direction(direction)},
                vertices=initial_vertices,
                edges=edges,
                edge_id_to_faces=edge_id_to_faces,
                core_face_ids=core_face_ids,
            )
            mixed = bool(
                int(match.get("oracle_edge_id_count") or 0) >= 2
                and finite_float(match.get("dominant_edge_fraction"), 0.0) < 0.85
            )
            pure = bool(match.get("oracle_confident_finite_edge")) and not mixed
            oracle_edge_cache[signature] = {
                **match,
                "classification": (
                    "confident_pure"
                    if pure
                    else "mixed"
                    if mixed
                    else "unmatched_or_ambiguous"
                ),
            }
        return oracle_edge_cache[signature]

    def label_hypothesis(row: dict[str, object]) -> dict[str, object]:
        keys = [
            (int(value[0]), int(value[1]))
            for value in row.get("point_keys", [])
        ]
        match = oracle_label(
            points=np.asarray(row.get("_points"), dtype=float),
            direction=np.asarray(row.get("direction"), dtype=float),
            point_keys=keys,
        )
        row["_oracle"] = match
        return match

    baseline_pure_edge_ids = {
        int(row["dominant_edge_id"])
        for row in segment_oracle_rows
        if row.get("dominant_edge_id") is not None
        and bool(row.get("oracle_confident_finite_edge"))
        and not (
            int(row.get("oracle_edge_id_count") or 0) >= 2
            and finite_float(row.get("dominant_edge_fraction"), 0.0) < 0.85
        )
    }
    baseline_control_edge_ids = baseline_pure_edge_ids - set(required_edge_ids)

    candidate_oracle_cache: dict[tuple[object, ...], dict[str, object] | None] = {}

    def candidate_geometry_signature(candidate: dict[str, object]) -> tuple[object, ...]:
        plane = candidate_plane(candidate)
        if plane is None:
            return (int(candidate_track_id(candidate)),)
        point, normal = plane
        normal = canonical_direction(np.asarray(normal, dtype=float))
        offset = float(normal @ np.asarray(point, dtype=float))
        hull = np.asarray(candidate.get("hull") or [point], dtype=float).reshape((-1, 3))
        centroid = np.mean(hull, axis=0)
        return (
            *(round(float(value), 6) for value in normal),
            round(offset, 5),
            *(round(float(value), 4) for value in centroid),
        )

    def candidate_match(candidate: dict[str, object]) -> dict[str, object] | None:
        signature = candidate_geometry_signature(candidate)
        if signature not in candidate_oracle_cache:
            candidate_oracle_cache[signature] = candidate_oracle_face(
                candidate,
                model_faces,
            )
        return candidate_oracle_cache[signature]

    current_raw_face_ids = {
        int(match["face_id"])
        for candidate in w2_raw_candidates
        for match in [candidate_match(candidate)]
        if match is not None
        and bool(match.get("finite_good"))
        and match.get("face_id") is not None
    }
    current_control_face_ids = current_raw_face_ids - set(target_face_ids)

    def segment_label_rows(segments: list[dict[str, object]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for segment in segments:
            keys = point_keys_for_segment(segment)
            if len(keys) < 2:
                continue
            points = points_for_keys(keys)
            match = oracle_label(
                points=points,
                direction=np.asarray(segment.get("direction"), dtype=float),
                point_keys=keys,
            )
            rows.append(
                {
                    "segment_id": int(segment.get("segment_id") or 0),
                    "classification": str(match["classification"]),
                    "edge_id": match.get("dominant_edge_id"),
                    "levels": int(segment.get("levels") or len(keys)),
                    "z_span": segment.get("z_span"),
                }
            )
        return rows

    def evaluate_segment_clusters(
        segments: list[dict[str, object]],
        clusters: list[dict[str, object]],
        labels: list[dict[str, object]],
    ) -> dict[str, object]:
        label_by_id = {int(row["segment_id"]): row for row in labels}
        pure_segments = [row for row in labels if row["classification"] == "confident_pure"]
        mixed_segments = [row for row in labels if row["classification"] == "mixed"]
        unmatched_segments = [
            row for row in labels if row["classification"] == "unmatched_or_ambiguous"
        ]
        cluster_counts = {
            "confident_pure": 0,
            "mixed": 0,
            "unmatched_or_ambiguous": 0,
        }
        pure_cluster_edge_ids: list[int] = []
        for cluster in clusters:
            member_labels = [
                label_by_id[int(segment_id)]
                for segment_id in cluster.get("member_segment_ids", [])
                if int(segment_id) in label_by_id
            ]
            pure_ids = {
                int(row["edge_id"])
                for row in member_labels
                if row["classification"] == "confident_pure"
                and row.get("edge_id") is not None
            }
            has_mixed = any(row["classification"] == "mixed" for row in member_labels)
            confident_fraction = float(
                sum(row["classification"] == "confident_pure" for row in member_labels)
                / max(1, len(member_labels))
            )
            if len(pure_ids) == 1 and not has_mixed and confident_fraction >= 0.75:
                classification = "confident_pure"
                pure_cluster_edge_ids.append(next(iter(pure_ids)))
            elif len(pure_ids) >= 2 or has_mixed:
                classification = "mixed"
            else:
                classification = "unmatched_or_ambiguous"
            cluster_counts[classification] += 1
        pure_edge_set = set(int(value) for value in pure_cluster_edge_ids)
        return {
            "segment_count": int(len(labels)),
            "segment_confident_pure": int(len(pure_segments)),
            "segment_mixed": int(len(mixed_segments)),
            "segment_unmatched": int(len(unmatched_segments)),
            "segment_mixed_rate": as_json_float(float(len(mixed_segments) / max(1, len(labels)))),
            "cluster_count": int(len(clusters)),
            "cluster_confident_pure": int(cluster_counts["confident_pure"]),
            "cluster_mixed": int(cluster_counts["mixed"]),
            "cluster_unmatched": int(cluster_counts["unmatched_or_ambiguous"]),
            "unique_model_edge_ids": int(len(pure_edge_set)),
            "required_boundary_edges_recovered": int(
                len(pure_edge_set & set(required_edge_ids))
            ),
            "required_boundary_edge_ids_recovered": sorted(
                int(value) for value in pure_edge_set & set(required_edge_ids)
            ),
            "current_good_controls_lost": int(
                len(baseline_control_edge_ids - pure_edge_set)
            ),
            "current_good_control_edge_ids_lost": sorted(
                int(value) for value in baseline_control_edge_ids - pure_edge_set
            ),
            "duplicate_hypotheses": int(
                len(pure_cluster_edge_ids) - len(pure_edge_set)
            ),
            "precision_confident_clusters": as_json_float(
                float(cluster_counts["confident_pure"] / max(1, len(clusters)))
            ),
        }

    sensitivity_started = time.perf_counter()
    sensitivity_configs = [
        {
            "name": "control_12_levels_0p12",
            "min_levels": int(segment_min_levels),
            "min_z_span": float(segment_min_z_span),
            "max_line_rms": float(segment_max_line_rms),
            "max_line_residual": float(segment_max_line_residual),
            "max_z_gap": int(segment_max_z_gap),
        },
        {
            "name": "short_8_levels_0p07",
            "min_levels": 8,
            "min_z_span": 0.07,
            "max_line_rms": float(segment_max_line_rms),
            "max_line_residual": float(segment_max_line_residual),
            "max_z_gap": int(segment_max_z_gap),
        },
        {
            "name": "short_4_levels_0p03",
            "min_levels": 4,
            "min_z_span": 0.03,
            "max_line_rms": float(segment_max_line_rms),
            "max_line_residual": float(segment_max_line_residual),
            "max_z_gap": int(segment_max_z_gap),
        },
        {
            "name": "short_4_levels_contiguous",
            "min_levels": 4,
            "min_z_span": 0.03,
            "max_line_rms": float(segment_max_line_rms),
            "max_line_residual": float(segment_max_line_residual),
            "max_z_gap": 1,
        },
        {
            "name": "short_4_levels_relaxed_residual",
            "min_levels": 4,
            "min_z_span": 0.03,
            "max_line_rms": max(float(segment_max_line_rms), 0.06),
            "max_line_residual": max(float(segment_max_line_residual), 0.15),
            "max_z_gap": int(segment_max_z_gap),
        },
    ]
    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_segments: dict[str, list[dict[str, object]]] = {}
    sensitivity_clusters: dict[str, list[dict[str, object]]] = {}
    sensitivity_candidates: dict[str, list[dict[str, object]]] = {}
    for config in sensitivity_configs:
        config_started = time.perf_counter()
        segments, segment_summary = build_w2_edge_segments(
            z_levels=z_levels,
            line_points_w2=line_points,
            cond_w2=condition,
            line_points_w3=w3_points,
            fit_rms_w3=w3_rms,
            min_levels=int(config["min_levels"]),
            min_z_span=float(config["min_z_span"]),
            max_line_rms=float(config["max_line_rms"]),
            max_line_residual=float(config["max_line_residual"]),
            max_z_gap=int(config["max_z_gap"]),
            max_point_jump=float(segment_max_point_jump),
            max_condition=float(segment_max_condition),
            segment_fit_mode="robust-split",
        )
        clusters, cluster_summary = cluster_w2_edge_segments(
            segments,
            mode=str(cluster_mode),
            max_distance=float(cluster_max_distance),
            max_angle_deg=float(cluster_max_angle_deg),
            min_overlap=float(cluster_min_overlap),
        )
        labels = segment_label_rows(segments)
        evaluation = evaluate_segment_clusters(segments, clusters, labels)
        candidates, face_summary = build_w2_face_candidates(
            clusters,
            z_levels,
            inside_point=inside_point,
            orientation_points=orientation_points,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            min_adjacency_levels=int(face_min_adjacency_levels),
            min_adjacency_fraction=float(face_min_adjacency_fraction),
            min_z_span=float(face_min_z_span),
            max_plane_rms=float(face_max_plane_rms),
        )
        finite_candidates = [
            candidate
            for candidate in candidates
            if bool((candidate_match(candidate) or {}).get("finite_good"))
        ]
        finite_ids = {
            int((candidate_match(candidate) or {})["face_id"])
            for candidate in finite_candidates
            if (candidate_match(candidate) or {}).get("face_id") is not None
        }
        sensitivity_rows.append(
            {
                **config,
                **evaluation,
                "face_candidates": int(len(candidates)),
                "finite_good_face_candidates": int(len(finite_candidates)),
                "finite_good_face_precision": as_json_float(
                    float(len(finite_candidates) / max(1, len(candidates)))
                ),
                "unique_finite_face_ids": int(len(finite_ids)),
                "recovered_adjacency_cohort_count": int(
                    len(finite_ids & set(target_face_ids))
                ),
                "recovered_adjacency_cohort_ids": sorted(
                    int(value) for value in finite_ids & set(target_face_ids)
                ),
                "current_control_faces_lost": int(
                    len(current_control_face_ids - finite_ids)
                ),
                "segment_builder": segment_summary,
                "cluster_builder": cluster_summary,
                "face_builder": face_summary,
                "timing_seconds": as_json_float(time.perf_counter() - config_started),
            }
        )
        sensitivity_segments[str(config["name"])] = segments
        sensitivity_clusters[str(config["name"])] = clusters
        sensitivity_candidates[str(config["name"])] = candidates
    control_lost_edge_ids = set(
        int(value)
        for value in (
            sensitivity_rows[0].get("current_good_control_edge_ids_lost", [])
            if sensitivity_rows
            else []
        )
    )
    for row in sensitivity_rows:
        lost_edge_ids = set(
            int(value)
            for value in row.get("current_good_control_edge_ids_lost", [])
        )
        additional_lost = sorted(int(value) for value in lost_edge_ids - control_lost_edge_ids)
        row["additional_current_good_controls_lost_vs_control"] = int(
            len(additional_lost)
        )
        row["additional_current_good_control_edge_ids_lost_vs_control"] = additional_lost
    timings["threshold_sensitivity_seconds"] = time.perf_counter() - sensitivity_started

    raw_runs_started = time.perf_counter()

    def all_observed_runs() -> list[list[tuple[int, int]]]:
        output: list[list[tuple[int, int]]] = []
        for cyclic in range(n_half):
            current: list[tuple[int, int]] = []
            for z_index in range(n_levels):
                point = line_points[int(cyclic), int(z_index)]
                if not np.all(np.isfinite(point)):
                    if current:
                        output.append(current)
                    current = []
                    continue
                if current:
                    previous_z = int(current[-1][1])
                    previous = line_points[int(cyclic), previous_z]
                    if (
                        int(z_index) - previous_z > int(segment_max_z_gap)
                        or float(np.linalg.norm(point - previous)) > float(segment_max_point_jump)
                    ):
                        output.append(current)
                        current = []
                current.append((int(cyclic), int(z_index)))
            if current:
                output.append(current)
        return output

    observed_runs = all_observed_runs()

    def recursive_tls_parts(
        keys: list[tuple[int, int]],
        *,
        depth: int = 0,
    ) -> list[list[tuple[int, int]]]:
        if len(keys) < 2:
            return []
        points = points_for_keys(keys)
        fit = orthogonal_line_fit(points)
        if fit is None:
            return []
        residual = np.asarray(fit["residuals"], dtype=float)
        if (
            finite_float(fit.get("rms"), float("inf")) <= float(segment_max_line_rms)
            and finite_float(fit.get("max_residual"), float("inf")) <= float(segment_max_line_residual)
        ):
            return [keys]
        if depth >= 3 or len(keys) < 4:
            return []
        residual_delta = np.abs(np.diff(residual))
        split_at = int(np.argmax(residual_delta)) + 1 if residual_delta.size else len(keys) // 2
        split_at = max(2, min(len(keys) - 2, split_at))
        return recursive_tls_parts(keys[:split_at], depth=depth + 1) + recursive_tls_parts(
            keys[split_at:],
            depth=depth + 1,
        )

    tls_parts = [
        part
        for run in observed_runs
        for part in recursive_tls_parts(run)
        if len(part) >= 2
    ]
    orthogonal_hypotheses: list[dict[str, object]] = []
    finite_hypotheses: list[dict[str, object]] = []
    local_hypotheses: list[dict[str, object]] = []
    for keys in tls_parts:
        orthogonal = make_hypothesis(
            keys=keys,
            representation="orthogonal_3d_tls",
            source="fixed_index_tls_partition",
            fit_mode="orthogonal_3d_tls",
        )
        finite = make_hypothesis(
            keys=keys,
            representation="finite_3d_segment",
            source="fixed_index_tls_partition",
            fit_mode="finite_3d_segment",
        )
        if orthogonal is not None:
            orthogonal_hypotheses.append(orthogonal)
        if finite is not None:
            finite_hypotheses.append(finite)
    for run in observed_runs:
        if not 2 <= len(run) <= 4:
            continue
        local = make_hypothesis(
            keys=run,
            representation="locally_regularized_short_span",
            source="fixed_index_few_level_run",
            fit_mode="locally_regularized_short_span",
        )
        if local is not None:
            local_hypotheses.append(local)
    timings["orthogonal_and_local_hypotheses_seconds"] = time.perf_counter() - raw_runs_started

    few_level_started = time.perf_counter()
    few_level_hypotheses: list[dict[str, object]] = []
    few_signatures: set[tuple[object, ...]] = set()
    hypotheses_by_z: dict[int, list[dict[str, object]]] = {}
    for z_index in range(n_levels):
        values = fit_rms[:, int(z_index)]
        finite = np.isfinite(values)
        minima: list[tuple[float, int]] = []
        for cyclic in np.flatnonzero(finite):
            previous = values[(int(cyclic) - 1) % n_half]
            following = values[(int(cyclic) + 1) % n_half]
            if not (values[int(cyclic)] <= previous and values[int(cyclic)] <= following):
                continue
            neighborhood = values[
                np.array(
                    [(int(cyclic) + delta) % n_half for delta in (-3, -2, -1, 1, 2, 3)],
                    dtype=int,
                )
            ]
            neighborhood = neighborhood[np.isfinite(neighborhood)]
            prominence = (
                float(np.median(neighborhood) - values[int(cyclic)])
                if neighborhood.size
                else 0.0
            )
            minima.append((float(prominence), int(cyclic)))
        for prominence, cyclic in sorted(minima, key=lambda row: (-row[0], row[1]))[:16]:
            keys = []
            for delta in (-2, -1, 0, 1, 2):
                index = (int(cyclic) + delta) % n_half
                point = line_points[index, int(z_index)]
                if not np.all(np.isfinite(point)):
                    continue
                if w3_points is not None:
                    w3_point = w3_points[index, int(z_index)]
                    if np.all(np.isfinite(w3_point)) and float(np.linalg.norm(w3_point[:2] - point[:2])) > 0.12:
                        continue
                keys.append((int(index), int(z_index)))
            if len(keys) < 2:
                continue
            hypothesis = make_hypothesis(
                keys=keys,
                representation="few_level_spatial_detector",
                source="w2_spatial_change_point",
                fit_mode="locally_regularized_short_span",
                extra={
                    "spatial_change_prominence": as_json_float(float(prominence)),
                    "cyclic_persistence": as_json_float(float(len(keys) / 5.0)),
                },
            )
            if hypothesis is None:
                continue
            if finite_float(hypothesis.get("orthogonal_residual_rms"), float("inf")) > 0.06:
                continue
            if not 0.01 <= finite_float(hypothesis.get("finite_segment_length"), 0.0) <= 1.5:
                continue
            direction = canonical_direction(np.asarray(hypothesis["direction"], dtype=float))
            centroid = np.asarray(hypothesis["centroid"], dtype=float)
            signature = (
                int(z_index),
                *(int(round(float(value) / 0.04)) for value in centroid[:2]),
                *(int(round(abs(float(value)) / 0.10)) for value in direction),
            )
            if signature in few_signatures:
                continue
            few_signatures.add(signature)
            hypotheses_by_z.setdefault(int(z_index), []).append(hypothesis)
            few_level_hypotheses.append(hypothesis)
    for z_index, rows in hypotheses_by_z.items():
        neighbor_rows = [
            row
            for delta in (-2, -1, 1, 2)
            for row in hypotheses_by_z.get(int(z_index) + delta, [])
        ]
        for row in rows:
            centroid = np.asarray(row["centroid"], dtype=float)
            direction = np.asarray(row["direction"], dtype=float)
            agreements = sum(
                float(np.linalg.norm(centroid - np.asarray(other["centroid"], dtype=float))) <= 0.12
                and angle_between(direction, np.asarray(other["direction"], dtype=float)) <= 20.0
                for other in neighbor_rows
            )
            row["neighbor_level_agreement"] = int(agreements)
    timings["few_level_spatial_detector_seconds"] = time.perf_counter() - few_level_started

    mixed_started = time.perf_counter()
    segment_by_id = {
        int(segment["segment_id"]): segment
        for segment in w2_segments
        if segment.get("segment_id") is not None
    }
    target_mixed_rows = [
        row
        for row in segment_oracle_rows
        if row.get("segment_id") is not None
        and row.get("dominant_edge_id") is not None
        and int(row["dominant_edge_id"]) in set(mixed_required_edge_ids)
        and int(row.get("oracle_edge_id_count") or 0) >= 2
        and finite_float(row.get("dominant_edge_fraction"), 0.0) < 0.85
        and int(row["segment_id"]) in segment_by_id
    ]

    def bounded_split_positions(count: int) -> list[int]:
        if count < 4:
            return []
        return sorted(
            {
                max(2, min(count - 2, int(value)))
                for value in np.linspace(2, count - 2, min(18, count - 3), dtype=int)
            }
        )

    def split_score(points: np.ndarray, split_at: int) -> float:
        parent = orthogonal_line_fit(points)
        left = orthogonal_line_fit(points[:split_at])
        right = orthogonal_line_fit(points[split_at:])
        if parent is None or left is None or right is None:
            return -float("inf")
        parent_sse = finite_float(parent.get("rms"), 0.0) ** 2 * points.shape[0]
        child_sse = (
            finite_float(left.get("rms"), 0.0) ** 2 * split_at
            + finite_float(right.get("rms"), 0.0) ** 2 * (points.shape[0] - split_at)
        )
        angle = angle_between(
            np.asarray(left["direction"], dtype=float),
            np.asarray(right["direction"], dtype=float),
        )
        return float((parent_sse - child_sse) / max(parent_sse, 1e-12) + angle / 90.0)

    def split_segment_keys(
        keys: list[tuple[int, int]],
        mode: str,
    ) -> tuple[list[list[tuple[int, int]]], dict[str, object]]:
        points = points_for_keys(keys)
        if len(keys) < 4:
            return [keys], {"split": False, "reason": "too_few_points"}
        parent_current = fit_xy_line_vs_z(points[:, 2], points)
        parent_tls = orthogonal_line_fit(points)
        residual = (
            xy_line_residuals(points[:, 2], points, parent_current)
            if parent_current is not None
            else np.zeros(len(keys), dtype=float)
        )
        step_distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if str(mode) == "residual_change_point":
            changes = np.abs(np.diff(residual))
            split_at = int(np.argmax(changes)) + 1 if changes.size else len(keys) // 2
        elif str(mode) == "direction_intercept_change":
            positions = bounded_split_positions(len(keys))
            split_at = max(positions, key=lambda value: split_score(points, value)) if positions else len(keys) // 2
        elif str(mode) == "spatial_continuity":
            split_at = int(np.argmax(step_distance)) + 1 if step_distance.size else len(keys) // 2
        else:
            components = [keys]
            while len(components) < 3:
                choices = []
                for index, component in enumerate(components):
                    component_points = points_for_keys(component)
                    for position in bounded_split_positions(len(component)):
                        choices.append(
                            (split_score(component_points, position), index, position)
                        )
                if not choices:
                    break
                score, component_index, position = max(choices)
                if score < 0.20:
                    break
                component = components.pop(int(component_index))
                components.extend([component[: int(position)], component[int(position) :]])
            return components, {
                "split": bool(len(components) > 1),
                "component_count": int(len(components)),
                "parent_tls_rms": as_json_float(finite_float((parent_tls or {}).get("rms"), float("nan"))),
            }
        split_at = max(2, min(len(keys) - 2, int(split_at)))
        score = split_score(points, split_at)
        if score < 0.08:
            return [keys], {
                "split": False,
                "reason": "insufficient_observed_improvement",
                "score": as_json_float(score),
            }
        return [keys[:split_at], keys[split_at:]], {
            "split": True,
            "split_at": int(split_at),
            "score": as_json_float(score),
            "residual_change": as_json_float(
                float(np.max(np.abs(np.diff(residual)))) if residual.size >= 2 else 0.0
            ),
            "spatial_step_change": as_json_float(
                float(np.max(step_distance)) if step_distance.size else 0.0
            ),
            "parent_tls_rms": as_json_float(finite_float((parent_tls or {}).get("rms"), float("nan"))),
        }

    mixed_split_modes = [
        "residual_change_point",
        "direction_intercept_change",
        "spatial_continuity",
        "multi_model_robust",
    ]
    mixed_split_hypotheses: dict[str, list[dict[str, object]]] = {
        mode: [] for mode in mixed_split_modes
    }
    mixed_anatomy_rows: list[dict[str, object]] = []
    for row in target_mixed_rows:
        segment_id = int(row["segment_id"])
        segment = segment_by_id[segment_id]
        keys = point_keys_for_segment(segment)
        points = points_for_keys(keys)
        current_fit = fit_xy_line_vs_z(points[:, 2], points) if len(keys) >= 2 else None
        residual = (
            xy_line_residuals(points[:, 2], points, current_fit)
            if current_fit is not None
            else np.zeros(0, dtype=float)
        )
        step_distance = np.linalg.norm(np.diff(points, axis=0), axis=1) if points.shape[0] >= 2 else np.zeros(0)
        split_results: dict[str, object] = {}
        for mode in mixed_split_modes:
            components, split_diagnostic = split_segment_keys(keys, mode)
            child_rows = []
            for component_index, component in enumerate(components):
                hypothesis = make_hypothesis(
                    keys=component,
                    representation="mixed_run_split",
                    source=str(mode),
                    fit_mode="orthogonal_3d_tls",
                    extra={
                        "parent_segment_id": int(segment_id),
                        "component_index": int(component_index),
                    },
                )
                if hypothesis is None:
                    continue
                match = label_hypothesis(hypothesis)
                mixed_split_hypotheses[mode].append(hypothesis)
                child_rows.append(
                    {
                        "component_index": int(component_index),
                        "point_count": int(hypothesis["point_count"]),
                        "classification": str(match["classification"]),
                        "edge_id": match.get("dominant_edge_id"),
                        "dominant_edge_fraction": match.get("dominant_edge_fraction"),
                        "angle_deg": match.get("angle_to_dominant_edge_deg"),
                    }
                )
            split_results[mode] = {
                **split_diagnostic,
                "children": child_rows,
            }
        mixed_anatomy_rows.append(
            {
                "segment_id": int(segment_id),
                "dominant_required_edge_id": int(row["dominant_edge_id"]),
                "physical_edge_count": int(row.get("oracle_edge_id_count") or 0),
                "dominant_edge_fraction": row.get("dominant_edge_fraction"),
                "cyclic_index": int(segment.get("cyclic_index") or 0),
                "cyclic_index_changes": 0,
                "levels": int(segment.get("levels") or len(keys)),
                "source_mode": str(segment.get("segment_fit_mode") or "least_squares"),
                "residual_change_max": as_json_float(
                    float(np.max(np.abs(np.diff(residual)))) if residual.size >= 2 else 0.0
                ),
                "spatial_step_change_max": as_json_float(
                    float(np.max(step_distance)) if step_distance.size else 0.0
                ),
                "residual_bimodality": as_json_float(
                    float(np.ptp(np.sort(residual)[-2:])) if residual.size >= 2 else 0.0
                ),
                "split_modes": split_results,
            }
        )

    def summarize_hypotheses(rows: list[dict[str, object]]) -> dict[str, object]:
        matches = [label_hypothesis(row) for row in rows]
        pure = [match for match in matches if match["classification"] == "confident_pure"]
        mixed = [match for match in matches if match["classification"] == "mixed"]
        unmatched = [
            match for match in matches if match["classification"] == "unmatched_or_ambiguous"
        ]
        pure_ids = {
            int(match["dominant_edge_id"])
            for match in pure
            if match.get("dominant_edge_id") is not None
        }
        return {
            "hypotheses": int(len(rows)),
            "confident_pure": int(len(pure)),
            "mixed": int(len(mixed)),
            "unmatched": int(len(unmatched)),
            "precision": as_json_float(float(len(pure) / max(1, len(rows)))),
            "unique_model_edge_ids": int(len(pure_ids)),
            "required_boundary_edges_recovered": int(
                len(pure_ids & set(required_edge_ids))
            ),
            "required_boundary_edge_ids_recovered": sorted(
                int(value) for value in pure_ids & set(required_edge_ids)
            ),
            "short_z_edges_recovered": int(len(pure_ids & set(short_edge_ids))),
            "short_z_edge_ids_recovered": sorted(
                int(value) for value in pure_ids & set(short_edge_ids)
            ),
            "mixed_required_edges_recovered": int(
                len(pure_ids & set(mixed_required_edge_ids))
            ),
            "mixed_required_edge_ids_recovered": sorted(
                int(value) for value in pure_ids & set(mixed_required_edge_ids)
            ),
            "duplicate_hypotheses": int(len(pure) - len(pure_ids)),
            "current_good_controls_lost": int(
                len(baseline_control_edge_ids - pure_ids)
            ),
        }

    representation_hypothesis_sets = {
        "orthogonal_3d_tls": orthogonal_hypotheses,
        "finite_3d_segment": finite_hypotheses,
        "locally_regularized_short_span": local_hypotheses,
        "few_level_spatial_detector": few_level_hypotheses,
        **{
            f"mixed_split_{mode}": rows
            for mode, rows in mixed_split_hypotheses.items()
        },
    }
    representation_hypothesis_summaries = {
        name: summarize_hypotheses(rows)
        for name, rows in representation_hypothesis_sets.items()
    }
    for summary in representation_hypothesis_summaries.values():
        not_reproduced = int(summary.pop("current_good_controls_lost", 0))
        summary["baseline_control_edges_reproduced"] = int(
            max(0, len(baseline_control_edge_ids) - not_reproduced)
        )
        summary["baseline_control_edges_not_reproduced_by_alternative_alone"] = int(
            not_reproduced
        )
        summary["current_good_controls_lost"] = 0
        summary["control_semantics"] = (
            "The raw alternative is additive to immutable current W2 clusters; lack of "
            "reproduction by the alternative alone is not a production control loss."
        )
    timings["mixed_segment_split_seconds"] = time.perf_counter() - mixed_started

    downstream_started = time.perf_counter()

    def hypothesis_nonoracle_score(row: dict[str, object]) -> float:
        w3_agreement = row.get("w2_w3_agreement_median")
        scale_persistence = row.get("scale_persistence_fraction")
        neighbor_agreement = finite_float(row.get("neighbor_level_agreement"), 0.0)
        return float(
            finite_float(row.get("orthogonal_residual_rms"), 1.0)
            + 0.002 * finite_float(row.get("direction_stability_deg"), 90.0)
            + 0.10 * finite_float(row.get("endpoint_stability"), 1.0)
            + (0.05 if w3_agreement is None else 0.20 * finite_float(w3_agreement, 1.0))
            + (0.03 if scale_persistence is None else 0.03 * (1.0 - finite_float(scale_persistence, 0.0)))
            - 0.002 * min(10.0, neighbor_agreement)
        )

    def hypothesis_geometry_key(row: dict[str, object]) -> tuple[object, ...]:
        centroid = np.asarray(row.get("centroid"), dtype=float)
        direction = canonical_direction(np.asarray(row.get("direction"), dtype=float))
        endpoints = np.asarray(row.get("endpoints"), dtype=float).reshape((-1, 3))
        midpoint = np.mean(endpoints, axis=0) if endpoints.size else centroid
        return (
            *(int(round(float(value) / 0.035)) for value in midpoint),
            *(int(round(abs(float(value)) / 0.025)) for value in direction),
            int(round(finite_float(row.get("finite_segment_length"), 0.0) / 0.04)),
        )

    def bounded_hypotheses(rows: list[dict[str, object]], limit: int = 600) -> list[dict[str, object]]:
        selected: list[dict[str, object]] = []
        seen: set[tuple[object, ...]] = set()
        for row in sorted(
            rows,
            key=lambda value: (
                hypothesis_nonoracle_score(value),
                -int(value.get("levels") or 0),
                int(value.get("hypothesis_id") or 0),
            ),
        ):
            signature = hypothesis_geometry_key(row)
            if signature in seen:
                continue
            seen.add(signature)
            selected.append(row)
            if len(selected) >= int(limit):
                break
        return selected

    def cluster_center_index(cluster: dict[str, object]) -> float:
        values = [
            float(value)
            for value in (
                cluster.get("member_cyclic_indices")
                or cluster.get("cyclic_indices")
                or [cluster.get("cyclic_index") or 0]
            )
        ]
        return float(np.mean(values)) if values else 0.0

    def cluster_level_indices(cluster: dict[str, object]) -> list[int]:
        explicit = sorted(
            {
                int(value)
                for value in cluster.get("z_indices", [])
                if 0 <= int(value) < n_levels
            }
        )
        if bool(cluster.get("_finite_hypothesis")):
            return explicit
        z_min = finite_float(
            cluster.get("cluster_z_min"),
            finite_float(cluster.get("z_min"), float("nan")),
        )
        z_max = finite_float(
            cluster.get("cluster_z_max"),
            finite_float(cluster.get("z_max"), float("nan")),
        )
        if np.isfinite(z_min) and np.isfinite(z_max):
            return [
                int(index)
                for index, value in enumerate(z_levels)
                if z_min - EPS <= float(value) <= z_max + EPS
            ]
        return explicit

    def cluster_support_points(
        cluster: dict[str, object],
        levels: list[int],
        *,
        finite_extent: bool,
    ) -> np.ndarray:
        if bool(cluster.get("_finite_hypothesis")):
            if finite_extent:
                points = np.asarray(cluster.get("_points"), dtype=float).reshape((-1, 3))
                return points[np.all(np.isfinite(points), axis=1)]
            direction = np.asarray(cluster.get("direction"), dtype=float)
            centroid = np.asarray(cluster.get("centroid"), dtype=float)
            if direction.shape != (3,) or centroid.shape != (3,) or abs(float(direction[2])) <= 1e-8:
                return np.zeros((0, 3), dtype=float)
            shared_levels = sorted(set(int(value) for value in levels))
            if not shared_levels:
                return np.zeros((0, 3), dtype=float)
            z_values = z_levels[np.array(shared_levels, dtype=int)]
            parameters = (z_values - float(centroid[2])) / float(direction[2])
            return centroid[None, :] + parameters[:, None] * direction[None, :]
        support_levels = cluster_level_indices(cluster) if finite_extent else sorted(set(levels))
        if len(support_levels) > 18:
            support_levels = [
                support_levels[int(index)]
                for index in np.linspace(0, len(support_levels) - 1, 18, dtype=int)
            ]
        if not support_levels:
            return np.zeros((0, 3), dtype=float)
        return predict_w2_line(
            cluster,
            z_levels[np.array(support_levels, dtype=int)],
        )

    def make_finite_cluster(
        row: dict[str, object],
        cluster_id: int,
    ) -> dict[str, object]:
        points = np.asarray(row.get("_points"), dtype=float).reshape((-1, 3))
        z_indices = sorted({int(value) for value in row.get("z_indices", [])})
        return {
            "edge_cluster_id": int(cluster_id),
            "member_segment_ids": [int(row.get("hypothesis_id") or -1)],
            "member_cyclic_indices": sorted(
                int(value) for value in row.get("cyclic_indices", [])
            ),
            "cyclic_indices": sorted(int(value) for value in row.get("cyclic_indices", [])),
            "z_indices": z_indices,
            "z_min": row.get("z_min"),
            "z_max": row.get("z_max"),
            "cluster_z_min": row.get("z_min"),
            "cluster_z_max": row.get("z_max"),
            "cluster_confidence": as_json_float(
                1.0 / (1.0 + 20.0 * hypothesis_nonoracle_score(row))
            ),
            "centroid": row.get("centroid"),
            "direction": row.get("direction"),
            "_finite_hypothesis": True,
            "_hypothesis": row,
            "_points": points,
        }

    def make_diagnostic_face_candidate(
        *,
        points: np.ndarray,
        left: dict[str, object],
        right: dict[str, object],
        levels: list[int],
        mode_name: str,
        mode_index: int,
        ordinal: int,
        finite_extent: bool,
    ) -> dict[str, object] | None:
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        points = points[np.all(np.isfinite(points), axis=1)]
        if (
            points.shape[0] < 3
            or np.any(points < bounds_min[None, :])
            or np.any(points > bounds_max[None, :])
        ):
            return None
        plane = fit_plane(points)
        if plane is None or float(plane.rms) > float(face_max_plane_rms):
            return None
        normal, orientation = orient_plane_from_observed_points(
            plane,
            orientation_points,
            inside_point,
        )
        u, v = plane_basis(normal)
        relative = points - plane.centroid
        coordinates = np.column_stack([relative @ u, relative @ v])
        hull_indices = convex_hull_indices(coordinates)
        if len(hull_indices) < 3:
            return None
        hull = points[np.array(hull_indices, dtype=int)]
        hull_area = float(polygon_area(hull))
        if hull_area <= 1e-8:
            return None
        diameter = float(
            np.max(np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2))
        )
        z_values = points[:, 2]
        return {
            "track_id": int(9700000 + 10000 * int(mode_index) + int(ordinal)),
            "source_track_id": None,
            "window": 2,
            "levels": int(len(set(int(value) for value in levels))),
            "z_min": as_json_float(float(np.min(z_values))),
            "z_max": as_json_float(float(np.max(z_values))),
            "center_mean": as_json_float(
                float(np.mean([cluster_center_index(left), cluster_center_index(right)]))
            ),
            "fit_points": int(points.shape[0]),
            "candidate_source": "w2_raw_signal_diagnostic",
            "candidate_origin": "w2_addition",
            "diagnostic_raw_signal_mode": str(mode_name),
            "diagnostic_finite_extent": bool(finite_extent),
            "w2_edge_cluster_ids": [
                int(left["edge_cluster_id"]),
                int(right["edge_cluster_id"]),
            ],
            "w2_adjacency_levels": int(len(set(int(value) for value in levels))),
            "w2_adjacency_fraction": 1.0,
            "w2_edge_confidence": as_json_float(
                float(
                    np.mean(
                        [
                            finite_float(left.get("cluster_confidence"), 0.0),
                            finite_float(right.get("cluster_confidence"), 0.0),
                        ]
                    )
                )
            ),
            "plane_centroid": as_json_point(plane.centroid),
            "plane_normal": as_json_point(normal),
            "plane_rms": as_json_float(float(plane.rms)),
            "plane_max_abs": as_json_float(float(plane.max_abs)),
            "orientation_positive_frac": as_json_float(
                float(orientation["orientation_positive_frac"])
            ),
            "orientation_point_count": int(orientation["orientation_point_count"]),
            "candidate_score": as_json_float(
                float(plane.rms) / max(float(points.shape[0]) ** 0.5, 1.0)
            ),
            "hull_diameter": as_json_float(diameter),
            "hull_area": as_json_float(hull_area),
            "hull": [as_json_point(point) for point in hull],
            "sample_points": [
                as_json_point(point)
                for point in points[:: max(1, points.shape[0] // 80)]
            ],
            "z_indices": sorted(set(int(value) for value in levels)),
            "low_source_counts": {"w2_edge": int(points.shape[0])},
            "low_valley_frac": 0.0,
        }

    def bounded_k4_face_rows(
        *,
        mode_name: str,
        mode_index: int,
        clusters: list[dict[str, object]],
        alternative_cluster_ids: set[int],
        finite_extent: bool,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        active_by_z: dict[int, list[dict[str, object]]] = {}
        for cluster in clusters:
            for z_index in cluster_level_indices(cluster):
                active_by_z.setdefault(int(z_index), []).append(cluster)
        pair_levels: dict[tuple[int, int], list[int]] = {}
        for z_index, rows in active_by_z.items():
            ordered = sorted(
                rows,
                key=lambda row: (cluster_center_index(row), int(row["edge_cluster_id"])),
            )
            count = len(ordered)
            if count < 2:
                continue
            for index, left in enumerate(ordered):
                for gap in range(1, min(4, count - 1) + 1):
                    right = ordered[(index + gap) % count]
                    pair = tuple(
                        sorted(
                            (int(left["edge_cluster_id"]), int(right["edge_cluster_id"]))
                        )
                    )
                    if pair[0] == pair[1]:
                        continue
                    if alternative_cluster_ids and not (
                        pair[0] in alternative_cluster_ids
                        or pair[1] in alternative_cluster_ids
                    ):
                        continue
                    pair_levels.setdefault(pair, []).append(int(z_index))
        by_id = {int(cluster["edge_cluster_id"]): cluster for cluster in clusters}
        sorted_pairs = sorted(
            (
                (pair, sorted(set(levels)))
                for pair, levels in pair_levels.items()
            ),
            key=lambda item: (-len(item[1]), item[0]),
        )
        degree: dict[int, int] = {}
        bounded_pairs: list[tuple[tuple[int, int], list[int]]] = []
        for pair, levels in sorted_pairs:
            if degree.get(pair[0], 0) >= 12 or degree.get(pair[1], 0) >= 12:
                continue
            bounded_pairs.append((pair, levels))
            degree[pair[0]] = degree.get(pair[0], 0) + 1
            degree[pair[1]] = degree.get(pair[1], 0) + 1
            if len(bounded_pairs) >= 4000:
                break
        rows: list[dict[str, object]] = []
        rejections: dict[str, int] = {}
        seen_candidates: set[tuple[object, ...]] = set()
        for ordinal, (pair, levels) in enumerate(bounded_pairs):
            if not finite_extent:
                if len(levels) < int(face_min_adjacency_levels):
                    rejections["insufficient_shared_z"] = rejections.get("insufficient_shared_z", 0) + 1
                    continue
                shared_z = z_levels[np.array(levels, dtype=int)]
                if shared_z.size == 0 or float(np.ptp(shared_z)) < float(face_min_z_span):
                    rejections["insufficient_shared_z"] = rejections.get("insufficient_shared_z", 0) + 1
                    continue
            left, right = by_id[pair[0]], by_id[pair[1]]
            left_points = cluster_support_points(left, levels, finite_extent=finite_extent)
            right_points = cluster_support_points(right, levels, finite_extent=finite_extent)
            points = np.vstack([left_points, right_points])
            candidate = make_diagnostic_face_candidate(
                points=points,
                left=left,
                right=right,
                levels=levels,
                mode_name=mode_name,
                mode_index=mode_index,
                ordinal=ordinal,
                finite_extent=finite_extent,
            )
            if candidate is None:
                rejections["poor_or_degenerate_plane"] = rejections.get("poor_or_degenerate_plane", 0) + 1
                continue
            signature = candidate_geometry_signature(candidate)
            if signature in seen_candidates:
                rejections["geometry_duplicate"] = rejections.get("geometry_duplicate", 0) + 1
                continue
            seen_candidates.add(signature)
            match = candidate_match(candidate)
            hypotheses = [
                cluster.get("_hypothesis")
                for cluster in (left, right)
                if isinstance(cluster.get("_hypothesis"), dict)
            ]
            source_score = float(
                np.mean([hypothesis_nonoracle_score(row) for row in hypotheses])
            ) if hypotheses else 0.0
            nonoracle_score = float(
                finite_float(candidate.get("plane_rms"), 1.0)
                + 0.25 * source_score
                + 0.002 * abs(cluster_center_index(left) - cluster_center_index(right))
            )
            rows.append(
                {
                    "pair": [int(pair[0]), int(pair[1])],
                    "candidate": candidate,
                    "finite_good": bool(match is not None and match.get("finite_good")),
                    "oracle_face_id": (
                        int(match["face_id"])
                        if match is not None and match.get("face_id") is not None
                        else None
                    ),
                    "nonoracle_score": as_json_float(nonoracle_score),
                    "adjacency_levels": int(len(levels)),
                    "finite_extent": bool(finite_extent),
                    "source_hypothesis_ids": [
                        int(row.get("hypothesis_id") or 0) for row in hypotheses
                    ],
                }
            )
        return rows, {
            "bounded_k": 4,
            "raw_pair_count": int(len(pair_levels)),
            "bounded_pair_count": int(len(bounded_pairs)),
            "candidate_count": int(len(rows)),
            "finite_extent_support_used": bool(finite_extent),
            "rejection_counts": {
                key: int(value) for key, value in sorted(rejections.items())
            },
        }

    downstream_mode_rows: dict[str, list[dict[str, object]]] = {}
    downstream_mode_funnels: dict[str, dict[str, object]] = {}
    downstream_selected_hypotheses: dict[str, list[dict[str, object]]] = {}
    mode_index = 0
    for config in sensitivity_configs:
        mode_name = f"threshold:{config['name']}"
        source_clusters = sensitivity_clusters[str(config["name"])]
        clusters = [
            {**dict(cluster), "edge_cluster_id": int(index)}
            for index, cluster in enumerate(source_clusters)
        ]
        rows, funnel = bounded_k4_face_rows(
            mode_name=mode_name,
            mode_index=mode_index,
            clusters=clusters,
            alternative_cluster_ids=set(),
            finite_extent=False,
        )
        downstream_mode_rows[mode_name] = rows
        downstream_mode_funnels[mode_name] = funnel
        mode_index += 1

    line_mode_names = [
        "orthogonal_3d_tls",
        "finite_3d_segment",
        "locally_regularized_short_span",
    ]
    selected_line_modes = sorted(
        line_mode_names,
        key=lambda name: (
            -int(representation_hypothesis_summaries[name]["short_z_edges_recovered"]),
            -finite_float(representation_hypothesis_summaries[name]["precision"], 0.0),
            int(representation_hypothesis_summaries[name]["hypotheses"]),
            name,
        ),
    )[:2]
    mixed_mode_names = [f"mixed_split_{mode}" for mode in mixed_split_modes]
    selected_mixed_mode = min(
        mixed_mode_names,
        key=lambda name: (
            -int(representation_hypothesis_summaries[name]["mixed_required_edges_recovered"]),
            -finite_float(representation_hypothesis_summaries[name]["precision"], 0.0),
            int(representation_hypothesis_summaries[name]["hypotheses"]),
            name,
        ),
    )
    custom_downstream_modes = list(dict.fromkeys(
        selected_line_modes
        + ["few_level_spatial_detector", selected_mixed_mode]
    ))
    for mode_name in custom_downstream_modes:
        selected = bounded_hypotheses(representation_hypothesis_sets[mode_name])
        downstream_selected_hypotheses[mode_name] = selected
        current_clusters = [
            {
                **dict(cluster),
                "edge_cluster_id": int(index),
                "_finite_hypothesis": False,
            }
            for index, cluster in enumerate(w2_clusters)
        ]
        first_alternative_id = len(current_clusters)
        alternative_clusters = [
            make_finite_cluster(row, first_alternative_id + index)
            for index, row in enumerate(selected)
        ]
        alternative_ids = {
            int(cluster["edge_cluster_id"])
            for cluster in alternative_clusters
        }
        use_finite_extent = bool(mode_name != "orthogonal_3d_tls")
        rows, funnel = bounded_k4_face_rows(
            mode_name=mode_name,
            mode_index=mode_index,
            clusters=current_clusters + alternative_clusters,
            alternative_cluster_ids=alternative_ids,
            finite_extent=use_finite_extent,
        )
        downstream_mode_rows[mode_name] = rows
        downstream_mode_funnels[mode_name] = {
            **funnel,
            "raw_hypothesis_count": int(
                len(representation_hypothesis_sets[mode_name])
            ),
            "nonoracle_bounded_hypothesis_count": int(len(selected)),
            "hypothesis_order_uses_oracle": False,
            "extent_semantics": (
                "infinite_tls_line_with_current_shared_z_gates"
                if not use_finite_extent
                else "observed_bounded_segment_extent"
            ),
        }
        mode_index += 1

    current_k4_mode = "threshold:control_12_levels_0p12"
    current_k4_finite_ids = {
        int(row["oracle_face_id"])
        for row in downstream_mode_rows.get(current_k4_mode, [])
        if bool(row.get("finite_good"))
        and row.get("oracle_face_id") is not None
    }
    current_k4_control_face_ids = current_k4_finite_ids - set(target_face_ids)

    def summarize_downstream_mode(
        mode_name: str,
        rows: list[dict[str, object]],
    ) -> dict[str, object]:
        finite_rows = [row for row in rows if bool(row.get("finite_good"))]
        finite_ids = {
            int(row["oracle_face_id"])
            for row in finite_rows
            if row.get("oracle_face_id") is not None
        }
        additive = not str(mode_name).startswith("threshold:")
        effective_ids = finite_ids | current_k4_finite_ids if additive else finite_ids
        recovered_targets = sorted(int(value) for value in finite_ids & set(target_face_ids))
        new_ids = sorted(int(value) for value in finite_ids - current_raw_face_ids)
        new_rows = [
            row
            for row in finite_rows
            if row.get("oracle_face_id") is not None
            and int(row["oracle_face_id"]) not in current_raw_face_ids
        ]
        return {
            "candidate_count": int(len(rows)),
            "finite_good_candidate_count": int(len(finite_rows)),
            "finite_good_precision": as_json_float(
                float(len(finite_rows) / max(1, len(rows)))
            ),
            "unique_finite_face_ids": int(len(finite_ids)),
            "new_finite_good_candidate_count_vs_current": int(len(new_rows)),
            "new_unique_face_count_vs_current": int(len(new_ids)),
            "new_unique_face_ids_vs_current": new_ids,
            "recovered_adjacency_cohort_count": int(len(recovered_targets)),
            "recovered_adjacency_cohort_ids": recovered_targets,
            "current_control_faces_lost": int(
                len(current_k4_control_face_ids - effective_ids)
            ),
            "current_control_face_ids_lost": sorted(
                int(value) for value in current_k4_control_face_ids - effective_ids
            ),
            "duplicate_finite_good_assignments": int(
                len(finite_rows) - len(finite_ids)
            ),
            "additive_to_current_clusters": bool(additive),
            "funnel": downstream_mode_funnels[mode_name],
        }

    downstream_summaries = {
        mode_name: summarize_downstream_mode(mode_name, rows)
        for mode_name, rows in downstream_mode_rows.items()
    }
    timings["downstream_candidate_evaluation_seconds"] = (
        time.perf_counter() - downstream_started
    )

    separation_started = time.perf_counter()

    def binary_auc(
        values: list[tuple[float, bool]],
        direction: str,
    ) -> float | None:
        positives = [value for value, label in values if label and np.isfinite(value)]
        negatives = [value for value, label in values if not label and np.isfinite(value)]
        if not positives or not negatives:
            return None
        wins = 0.0
        for positive in positives:
            for negative in negatives:
                if positive == negative:
                    wins += 0.5
                elif (
                    direction == "higher" and positive > negative
                ) or (
                    direction == "lower" and positive < negative
                ):
                    wins += 1.0
        return float(wins / (len(positives) * len(negatives)))

    separation_feature_directions = {
        "orthogonal_residual_rms": "lower",
        "eigenvalue_primary_secondary_ratio": "higher",
        "finite_segment_length": "higher",
        "direction_stability_deg": "lower",
        "endpoint_stability": "lower",
        "scale_persistence_fraction": "higher",
        "spatial_change_prominence": "higher",
        "cyclic_persistence": "higher",
        "neighbor_level_agreement": "higher",
    }
    separation_rows = [
        row
        for rows in representation_hypothesis_sets.values()
        for row in rows
        if int(row.get("levels") or 0) <= 6
        or finite_float(row.get("z_span"), float("inf")) < float(segment_min_z_span)
    ]
    hypothesis_separation: dict[str, object] = {}
    for feature_name, direction in separation_feature_directions.items():
        values: list[tuple[float, bool]] = []
        for row in separation_rows:
            match = label_hypothesis(row)
            value = finite_float(row.get(feature_name), float("nan"))
            values.append((value, str(match["classification"]) == "confident_pure"))
        feature_auc = binary_auc(values, direction)
        orientation_neutral_auc = (
            max(float(feature_auc), 1.0 - float(feature_auc))
            if feature_auc is not None
            else None
        )
        empirical_direction = (
            direction
            if feature_auc is None or feature_auc >= 0.5
            else ("higher" if direction == "lower" else "lower")
        )
        hypothesis_separation[feature_name] = {
            "prespecified_direction": str(direction),
            "auc": as_json_float(feature_auc) if feature_auc is not None else None,
            "orientation_neutral_auc": (
                as_json_float(orientation_neutral_auc)
                if orientation_neutral_auc is not None
                else None
            ),
            "empirical_direction_diagnostic_only": str(empirical_direction),
            "correct": distribution_summary(
                [value for value, label in values if label and np.isfinite(value)]
            ),
            "false_or_mixed": distribution_summary(
                [value for value, label in values if not label and np.isfinite(value)]
            ),
            "evaluated": int(sum(np.isfinite(value) for value, _ in values)),
        }
    max_hypothesis_auc = max(
        (
            finite_float(row.get("auc"), 0.0)
            for row in hypothesis_separation.values()
        ),
        default=0.0,
    )
    max_orientation_neutral_auc = max(
        (
            finite_float(row.get("orientation_neutral_auc"), 0.0)
            for row in hypothesis_separation.values()
        ),
        default=0.0,
    )
    timings["nonoracle_separation_seconds"] = time.perf_counter() - separation_started

    full_started = time.perf_counter()
    baseline_active_indices = [
        int(index)
        for index in final_reconstructed.get("face_candidate_indices", [])
        if 0 <= int(index) < len(final_candidates)
    ]
    baseline_active_candidates = [
        final_candidates[index] for index in baseline_active_indices
    ]
    baseline_finite_ids = {
        int(match["face_id"])
        for candidate in baseline_active_candidates
        for match in [candidate_match(candidate)]
        if match is not None
        and bool(match.get("finite_good"))
        and match.get("face_id") is not None
    }

    def strict_patch_compatible(
        left: dict[str, object],
        right: dict[str, object],
    ) -> bool:
        angle, offset, centroid_distance = candidate_plane_relation_score(left, right)
        left_z = candidate_z_intervals(left)
        right_z = candidate_z_intervals(right)
        _, hull_fraction = interval_overlap(
            left_z.get("hull_z_min"),
            left_z.get("hull_z_max"),
            right_z.get("hull_z_min"),
            right_z.get("hull_z_max"),
        )
        _, support_fraction = interval_overlap(
            left_z.get("support_z_min"),
            left_z.get("support_z_max"),
            right_z.get("support_z_min"),
            right_z.get("support_z_max"),
        )
        return bool(
            angle <= 1.0
            and offset <= 0.02
            and centroid_distance <= 0.12
            and max(hull_fraction, support_fraction) >= 0.25
        )

    baseline_groups: list[list[dict[str, object]]] = []
    for candidate in baseline_active_candidates:
        for group in baseline_groups:
            if all(strict_patch_compatible(candidate, member) for member in group):
                group.append(candidate)
                break
        else:
            baseline_groups.append([candidate])

    representative_by_face: dict[int, dict[str, object]] = {}
    for mode_name, rows in downstream_mode_rows.items():
        for row in rows:
            face_id = row.get("oracle_face_id")
            if (
                not bool(row.get("finite_good"))
                or face_id is None
                or int(face_id) not in set(target_face_ids)
                or int(face_id) in baseline_finite_ids
            ):
                continue
            candidate_row = {**row, "mode_name": str(mode_name)}
            previous = representative_by_face.get(int(face_id))
            if previous is None or (
                finite_float(candidate_row.get("nonoracle_score"), float("inf")),
                str(candidate_row["mode_name"]),
                candidate_row.get("pair"),
            ) < (
                finite_float(previous.get("nonoracle_score"), float("inf")),
                str(previous["mode_name"]),
                previous.get("pair"),
            ):
                representative_by_face[int(face_id)] = candidate_row
    full_trial_representatives = sorted(
        representative_by_face.values(),
        key=lambda row: (
            finite_float(row.get("nonoracle_score"), float("inf")),
            str(row["mode_name"]),
            row.get("pair"),
        ),
    )[:15]
    z_slices = prepare_z_level_slices(trusted_z_indices)
    full_trial_rows: list[dict[str, object]] = []
    for trial_index, row in enumerate(full_trial_representatives):
        candidate = dict(row["candidate"])
        candidate["track_id"] = int(9960000 + trial_index)
        trial_candidates = list(final_candidates) + [candidate]
        trial_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
            trial_candidates,
            **reconstruction_kwargs,
        )
        active_indices = [
            int(index)
            for index in trial_reconstructed.get("face_candidate_indices", [])
            if 0 <= int(index) < len(trial_candidates)
        ]
        active_candidates = [trial_candidates[index] for index in active_indices]
        trial_finite_ids = {
            int(match["face_id"])
            for active_candidate in active_candidates
            for match in [candidate_match(active_candidate)]
            if match is not None
            and bool(match.get("finite_good"))
            and match.get("face_id") is not None
        }
        lost_groups = [
            sorted(int(candidate_track_id(member)) for member in group)
            for group in baseline_groups
            if not any(
                strict_patch_compatible(member, active_candidate)
                for member in group
                for active_candidate in active_candidates
            )
        ]
        gained_ids = sorted(int(value) for value in trial_finite_ids - baseline_finite_ids)
        lost_ids = sorted(int(value) for value in baseline_finite_ids - trial_finite_ids)
        target_face_id = int(row["oracle_face_id"])
        candidate_active = bool(len(trial_candidates) - 1 in active_indices)
        target_active = bool(target_face_id in trial_finite_ids)
        if not candidate_active:
            classification = "inactive"
        elif lost_groups:
            classification = "rolling_group_loss"
        elif gained_ids and lost_ids:
            classification = "replacement"
        elif gained_ids and target_active and not lost_ids:
            classification = "safe_plus_one"
        elif gained_ids and not target_active:
            classification = "canonical_mapping_mismatch"
        else:
            classification = "redundant"
        outside = evaluate_trusted_cloud_outside(
            trial_candidates,
            trusted_points,
            z_slices,
            point_tol=float(point_tol),
        )
        topology = (
            trial_reconstructed.get("topology")
            if isinstance(trial_reconstructed.get("topology"), dict)
            else {}
        )
        safety_pass = bool(
            topology_is_valid(topology)
            and finite_float(outside.get("cumulative_outside_fraction"), 1.0) < 0.01
            and int(outside.get("cumulative_lost_z_levels") or 0) == 0
        )
        full_trial_rows.append(
            {
                "candidate_level_target_face_id": int(target_face_id),
                "raw_signal_mode": str(row["mode_name"]),
                "pair": [int(value) for value in row.get("pair", [])],
                "nonoracle_score": row.get("nonoracle_score"),
                "classification": str(classification),
                "candidate_active": bool(candidate_active),
                "final_target_active": bool(target_active),
                "canonical_ids_gained": gained_ids,
                "canonical_ids_lost": lost_ids,
                "retained": int(len(trial_finite_ids & set(core_face_ids))),
                "new": int(len(trial_finite_ids - set(core_face_ids))),
                "unique": int(len(trial_finite_ids)),
                "rolling_groups_lost": int(len(lost_groups)),
                "lost_rolling_group_candidate_ids": lost_groups,
                "vertices": int(len(trial_reconstructed.get("vertices", []))),
                "edges": int(len(trial_reconstructed.get("edges", []))),
                "faces": int(len(trial_reconstructed.get("faces", []))),
                "volume": trial_reconstructed.get("reliable_volume"),
                "topology": topology,
                "topology_valid": bool(topology_is_valid(topology)),
                "outside": outside,
                "safety_pass": bool(safety_pass),
            }
        )
    full_classification_counts = {
        category: int(
            sum(row["classification"] == category for row in full_trial_rows)
        )
        for category in (
            "safe_plus_one",
            "replacement",
            "redundant",
            "inactive",
            "rolling_group_loss",
            "canonical_mapping_mismatch",
        )
    }
    safe_plus_one_by_mode = {
        mode_name: int(
            sum(
                row["raw_signal_mode"] == mode_name
                and row["classification"] == "safe_plus_one"
                and bool(row.get("safety_pass"))
                for row in full_trial_rows
            )
        )
        for mode_name in downstream_mode_rows
    }
    timings["bounded_full_edge_clip_seconds"] = time.perf_counter() - full_started

    independent_validation_started = time.perf_counter()

    def run_short_edge_independent_validation() -> dict[str, object]:
        fixed_thresholds = {
            "cross_scale": {
                "minimum_supporting_scales": 2,
                "maximum_position_spread": 0.12,
                "maximum_direction_spread_deg": 25.0,
                "minimum_finite_overlap_fraction": 0.10,
            },
            "transition": {
                "minimum_prominence": 0.35,
                "minimum_neighbor_consistency": 0.40,
                "minimum_supporting_level_pairs": 1,
            },
            "plane_consensus": {
                "minimum_independent_sources": 2,
                "maximum_normal_spread_deg": 3.0,
                "maximum_offset_spread": 0.05,
                "minimum_patch_overlap": 0.10,
            },
            "intersection_anchor": {
                "minimum_anchor_planes": 1,
                "maximum_intersection_distance": 0.08,
                "maximum_direction_error_deg": 15.0,
                "minimum_anchored_overlap": 0.10,
                "minimum_neighbor_plane_support": 1,
            },
            "marginal_contour": {
                "minimum_local_support_points": 3,
                "minimum_local_deficit_reduction": 0.002,
                "minimum_unsupported_fraction_reduction": 0.02,
                "maximum_trusted_cut_fraction": 0.01,
                "trial_must_be_active": True,
            },
            "combined": {"minimum_independent_signal_families": 2},
        }
        direction_registry = {
            "number_of_supporting_scales": "higher",
            "cross_scale_position_spread": "lower",
            "cross_scale_direction_spread": "lower",
            "finite_overlap_fraction": "higher",
            "independent_segment_count": "higher",
            "transition_prominence": "higher",
            "above_below_position_change": "higher",
            "tangent_change": "higher",
            "neighbor_consistency": "higher",
            "number_of_supporting_level_pairs": "higher",
            "independent_plane_sources": "higher",
            "independent_candidate_count": "higher",
            "plane_normal_spread": "lower",
            "plane_offset_spread": "lower",
            "patch_overlap": "higher",
            "anchor_plane_count": "higher",
            "best_intersection_distance": "lower",
            "intersection_direction_error": "lower",
            "anchored_overlap": "higher",
            "neighbor_plane_support": "higher",
            "local_contour_deficit_reduction": "higher",
            "unsupported_fraction_reduction": "higher",
            "trusted_cut_fraction": "lower",
        }

        def point_segment_distances(points: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
            pts = np.asarray(points, dtype=float).reshape((-1, 3))
            ends = np.asarray(endpoints, dtype=float).reshape((2, 3))
            delta = ends[1] - ends[0]
            denom = max(float(delta @ delta), EPS)
            parameters = np.clip(((pts - ends[0]) @ delta) / denom, 0.0, 1.0)
            return np.linalg.norm(pts - (ends[0] + parameters[:, None] * delta), axis=1)

        scale_inputs: list[tuple[int, np.ndarray, np.ndarray | None]] = [
            (2, line_points, fit_rms),
        ]
        if w3_points is not None:
            scale_inputs.append((3, w3_points, w3_rms))
        if w4_points is not None:
            scale_inputs.append((4, w4_points, w4_rms))
        scale_level_clouds: dict[int, list[np.ndarray]] = {}
        scale_level_indices: dict[int, list[np.ndarray]] = {}
        for scale, points_array, _ in scale_inputs:
            scale_level_clouds[scale] = []
            scale_level_indices[scale] = []
            for z_index in range(min(n_levels, int(points_array.shape[1]))):
                mask = np.all(np.isfinite(points_array[:, z_index, :]), axis=1)
                scale_level_clouds[scale].append(points_array[mask, z_index, :].astype(float))
                scale_level_indices[scale].append(np.flatnonzero(mask).astype(int))

        hypothesis_independent_cache: dict[int, dict[str, object]] = {}

        def cross_scale_features(row: dict[str, object]) -> dict[str, object]:
            hypothesis_id = int(row.get("hypothesis_id") or -1)
            if hypothesis_id in hypothesis_independent_cache:
                return dict(hypothesis_independent_cache[hypothesis_id]["cross_scale"])
            endpoints = np.asarray(row.get("endpoints"), dtype=float).reshape((2, 3))
            direction = canonical_direction(np.asarray(row.get("direction"), dtype=float))
            source_points = np.asarray(row.get("_points"), dtype=float).reshape((-1, 3))
            source_keys = [
                (int(value[0]), int(value[1]))
                for value in row.get("point_keys", [])
            ]
            if len(source_keys) > 18:
                sample_indices = np.linspace(0, len(source_keys) - 1, 18, dtype=int)
                source_keys = [source_keys[int(index)] for index in sample_indices]
                source_points = source_points[sample_indices]
            supporting = [2]
            position_spreads: list[float] = []
            direction_spreads: list[float] = []
            overlaps: list[float] = [1.0]
            per_scale: list[dict[str, object]] = []
            target_projection = (endpoints - endpoints[0]) @ direction
            target_min = float(np.min(target_projection))
            target_max = float(np.max(target_projection))
            target_length = max(target_max - target_min, 1e-6)
            for scale, _, _ in scale_inputs[1:]:
                matches: list[np.ndarray] = []
                match_distances: list[float] = []
                for source_point, (_, z_index) in zip(source_points, source_keys):
                    best_point = None
                    best_distance = float("inf")
                    for nearby_z in range(max(0, z_index - 1), min(n_levels, z_index + 2)):
                        cloud = scale_level_clouds[scale][nearby_z]
                        if cloud.shape[0] == 0:
                            continue
                        distances = np.linalg.norm(cloud - source_point[None, :], axis=1)
                        local_index = int(np.argmin(distances))
                        if float(distances[local_index]) < best_distance:
                            best_distance = float(distances[local_index])
                            best_point = cloud[local_index]
                    if best_point is not None and best_distance <= 0.18:
                        matches.append(np.asarray(best_point, dtype=float))
                        match_distances.append(float(best_distance))
                matched = np.array(matches, dtype=float).reshape((-1, 3)) if matches else np.zeros((0, 3), dtype=float)
                fit = orthogonal_line_fit(matched) if matched.shape[0] >= 2 else None
                match_fraction = float(matched.shape[0] / max(1, len(source_keys)))
                position = float(np.median(match_distances)) if match_distances else float("inf")
                angle = (
                    angle_between(direction, np.asarray(fit["direction"], dtype=float))
                    if fit is not None
                    else float("inf")
                )
                overlap = 0.0
                if fit is not None:
                    projection = (matched - endpoints[0]) @ direction
                    overlap_length = max(
                        0.0,
                        min(target_max, float(np.max(projection)))
                        - max(target_min, float(np.min(projection))),
                    )
                    overlap = float(overlap_length / target_length)
                confirmed = bool(
                    match_fraction >= 0.40
                    and position <= 0.12
                    and angle <= 25.0
                    and overlap >= 0.10
                )
                if confirmed:
                    supporting.append(int(scale))
                    position_spreads.append(float(position))
                    direction_spreads.append(float(angle))
                    overlaps.append(float(overlap))
                per_scale.append(
                    {
                        "scale": int(scale),
                        "matched_points": int(matched.shape[0]),
                        "match_fraction": as_json_float(match_fraction),
                        "position_spread": as_json_float(position),
                        "direction_spread_deg": as_json_float(angle),
                        "finite_overlap_fraction": as_json_float(overlap),
                        "confirmed": bool(confirmed),
                    }
                )
            result = {
                "number_of_supporting_scales": int(len(supporting)),
                "supporting_scales": supporting,
                "cross_scale_position_spread": as_json_float(
                    max(position_spreads) if position_spreads else float("inf")
                ),
                "cross_scale_direction_spread": as_json_float(
                    max(direction_spreads) if direction_spreads else float("inf")
                ),
                "finite_overlap_fraction": as_json_float(min(overlaps)),
                "independent_segment_count": int(len(supporting)),
                "per_scale": per_scale,
            }
            return result

        def local_level_feature(
            points_array: np.ndarray,
            rms_array: np.ndarray | None,
            z_index: int,
            endpoints: np.ndarray,
        ) -> dict[str, object] | None:
            if z_index < 0 or z_index >= min(n_levels, int(points_array.shape[1])):
                return None
            cloud = scale_level_clouds[2][z_index]
            cyclic_indices = scale_level_indices[2][z_index]
            if cloud.shape[0] == 0:
                return None
            distances = point_segment_distances(cloud, endpoints)
            local = int(np.argmin(distances))
            cyclic = int(cyclic_indices[local])
            point = cloud[local]
            tangent = None
            left = points_array[(cyclic - 1) % n_half, z_index]
            right = points_array[(cyclic + 1) % n_half, z_index]
            if np.all(np.isfinite(left)) and np.all(np.isfinite(right)):
                vector = right - left
                if float(np.linalg.norm(vector)) > EPS:
                    tangent = canonical_direction(vector)
            rms_value = None
            if rms_array is not None and np.isfinite(rms_array[cyclic, z_index]):
                rms_value = float(rms_array[cyclic, z_index])
            return {
                "point": point,
                "distance": float(distances[local]),
                "cyclic_index": int(cyclic),
                "tangent": tangent,
                "rms": rms_value,
            }

        def transition_features(row: dict[str, object]) -> dict[str, object]:
            endpoints = np.asarray(row.get("endpoints"), dtype=float).reshape((2, 3))
            levels = sorted({int(value) for value in row.get("z_indices", [])})
            if not levels:
                return {
                    "transition_prominence": 0.0,
                    "above_below_position_change": 0.0,
                    "tangent_change": 0.0,
                    "neighbor_consistency": 0.0,
                    "number_of_supporting_level_pairs": 0,
                    "above_below_symmetry": 0.0,
                    "level_pairs": [],
                }
            boundary_pairs = [
                (levels[0], levels[0] - 1, "below"),
                (levels[-1], levels[-1] + 1, "above"),
            ]
            pair_rows: list[dict[str, object]] = []
            for inside_index, outside_index, side in boundary_pairs:
                inside = local_level_feature(line_points, fit_rms, inside_index, endpoints)
                outside = local_level_feature(line_points, fit_rms, outside_index, endpoints)
                if inside is None:
                    continue
                inside_distance = finite_float(inside.get("distance"), float("inf"))
                outside_distance = finite_float(
                    (outside or {}).get("distance"),
                    float("inf"),
                )
                position_change = (
                    float(np.linalg.norm(np.asarray(inside["point"]) - np.asarray(outside["point"])))
                    if outside is not None
                    else 0.24
                )
                tangent_change = 0.0
                if inside.get("tangent") is not None and (outside or {}).get("tangent") is not None:
                    tangent_change = angle_between(
                        np.asarray(inside["tangent"], dtype=float),
                        np.asarray(outside["tangent"], dtype=float),
                    )
                rms_change = 0.0
                inside_rms = inside.get("rms")
                outside_rms = (outside or {}).get("rms")
                if inside_rms is not None and outside_rms is not None:
                    rms_change = abs(
                        math.log10(max(float(inside_rms), 1e-12))
                        - math.log10(max(float(outside_rms), 1e-12))
                    )
                birth_death = bool(inside_distance <= 0.12 and outside_distance > 0.18)
                prominence = max(
                    1.0 if birth_death else 0.0,
                    max(0.0, outside_distance - inside_distance) / 0.12,
                    position_change / 0.20,
                    tangent_change / 45.0,
                    rms_change,
                )
                if outside is None:
                    neighbor_consistency = 1.0 if inside_distance <= 0.12 else 0.0
                else:
                    cyclic_delta = abs(
                        int(inside["cyclic_index"]) - int(outside["cyclic_index"])
                    )
                    cyclic_delta = min(cyclic_delta, n_half - cyclic_delta)
                    neighbor_consistency = max(0.0, 1.0 - cyclic_delta / 10.0)
                supporting = bool(
                    inside_distance <= 0.12
                    and prominence >= 0.35
                    and neighbor_consistency >= 0.40
                )
                pair_rows.append(
                    {
                        "side": str(side),
                        "inside_z_index": int(inside_index),
                        "outside_z_index": int(outside_index),
                        "inside_distance": as_json_float(inside_distance),
                        "outside_distance": as_json_float(outside_distance),
                        "birth_or_death": bool(birth_death),
                        "position_change": as_json_float(position_change),
                        "tangent_change_deg": as_json_float(tangent_change),
                        "rms_log_change": as_json_float(rms_change),
                        "prominence": as_json_float(prominence),
                        "neighbor_consistency": as_json_float(neighbor_consistency),
                        "supporting": bool(supporting),
                    }
                )
            prominences = [finite_float(value.get("prominence"), 0.0) for value in pair_rows]
            positive_prominences = [value for value in prominences if value > 0.0]
            symmetry = (
                min(positive_prominences) / max(positive_prominences)
                if len(positive_prominences) >= 2
                else 0.0
            )
            return {
                "transition_prominence": as_json_float(max(prominences, default=0.0)),
                "above_below_position_change": as_json_float(
                    max((finite_float(value.get("position_change"), 0.0) for value in pair_rows), default=0.0)
                ),
                "tangent_change": as_json_float(
                    max((finite_float(value.get("tangent_change_deg"), 0.0) for value in pair_rows), default=0.0)
                ),
                "neighbor_consistency": as_json_float(
                    max((finite_float(value.get("neighbor_consistency"), 0.0) for value in pair_rows), default=0.0)
                ),
                "number_of_supporting_level_pairs": int(
                    sum(bool(value.get("supporting")) for value in pair_rows)
                ),
                "above_below_symmetry": as_json_float(symmetry),
                "level_pairs": pair_rows,
            }

        all_mixed_hypotheses = [
            row
            for mode in mixed_split_modes
            for row in mixed_split_hypotheses[mode]
        ]
        selected_mixed_hypotheses = list(
            downstream_selected_hypotheses.get(selected_mixed_mode, [])
        )
        hypothesis_by_id = {
            int(row.get("hypothesis_id") or -1): row
            for row in all_mixed_hypotheses
        }
        for row in all_mixed_hypotheses:
            hypothesis_id = int(row.get("hypothesis_id") or -1)
            hypothesis_independent_cache[hypothesis_id] = {
                "cross_scale": cross_scale_features(row),
                "transition": transition_features(row),
            }

        segment_by_id_local = {
            int(segment.get("segment_id") or -1): segment
            for segment in w2_segments
        }
        cluster_by_id_local = {
            int(cluster.get("edge_cluster_id") or -1): cluster
            for cluster in w2_clusters
        }
        target_parent_segments = {
            int(row.get("parent_segment_id") or -1)
            for row in all_mixed_hypotheses
        }
        control_hypothesis_by_cluster: dict[int, dict[str, object]] = {}

        def control_hypothesis(cluster_id: int) -> dict[str, object] | None:
            if int(cluster_id) in control_hypothesis_by_cluster:
                return control_hypothesis_by_cluster[int(cluster_id)]
            cluster = cluster_by_id_local.get(int(cluster_id))
            if cluster is None:
                return None
            member_ids = [int(value) for value in cluster.get("member_segment_ids", [])]
            if any(value in target_parent_segments for value in member_ids):
                return None
            keys = sorted(
                {
                    key
                    for segment_id in member_ids
                    for key in point_keys_for_segment(segment_by_id_local.get(segment_id, {}))
                }
            )
            if len(keys) < max(4, int(segment_min_levels)):
                return None
            hypothesis = make_hypothesis(
                keys=keys,
                representation="production_w2_cluster_control",
                source="calibration_current_w2",
                fit_mode="orthogonal_3d_tls",
            )
            if hypothesis is None:
                return None
            if finite_float(hypothesis.get("z_span"), 0.0) < float(segment_min_z_span):
                return None
            hypothesis_id = int(hypothesis.get("hypothesis_id") or -1)
            hypothesis_independent_cache[hypothesis_id] = {
                "cross_scale": cross_scale_features(hypothesis),
                "transition": transition_features(hypothesis),
            }
            control_hypothesis_by_cluster[int(cluster_id)] = hypothesis
            return hypothesis

        active_w2_control_candidates: list[tuple[dict[str, object], list[dict[str, object]]]] = []
        for candidate in baseline_active_candidates:
            cluster_ids = [int(value) for value in candidate.get("w2_edge_cluster_ids", [])]
            if not cluster_ids:
                continue
            hypotheses = [
                hypothesis
                for cluster_id in cluster_ids
                for hypothesis in [control_hypothesis(cluster_id)]
                if hypothesis is not None
            ]
            if hypotheses:
                active_w2_control_candidates.append((candidate, hypotheses))

        def projected_patch_overlap(
            candidate: dict[str, object],
            other: dict[str, object],
        ) -> float:
            plane = candidate_plane(candidate)
            if plane is None:
                return 0.0
            point, normal = plane
            candidate_hull = candidate_hull_points(candidate)
            other_hull = candidate_hull_points(other)
            if candidate_hull.shape[0] == 0 or other_hull.shape[0] == 0:
                return 0.0
            u, v = plane_basis(normal)
            candidate_2d = convex_hull_2d(
                np.column_stack([(candidate_hull - point) @ u, (candidate_hull - point) @ v])
            )
            other_2d = convex_hull_2d(
                np.column_stack([(other_hull - point) @ u, (other_hull - point) @ v])
            )
            if candidate_2d.shape[0] < 3 or other_2d.shape[0] < 3:
                nearest = nearest_point_distances(candidate_hull, other_hull)
                return float(np.mean(nearest <= 0.10)) if nearest.size else 0.0
            candidate_inside = point_to_polygon_distance_2d(candidate_2d, other_2d) <= 0.10
            other_inside = point_to_polygon_distance_2d(other_2d, candidate_2d) <= 0.10
            return float(0.5 * (np.mean(candidate_inside) + np.mean(other_inside)))

        active_ids = {int(candidate_track_id(value)) for value in baseline_active_candidates}
        plane_source_entries: list[dict[str, object]] = []
        seen_source_keys: set[tuple[object, ...]] = set()

        def add_plane_sources(
            rows: list[dict[str, object]],
            source: str,
            *,
            limit: int,
        ) -> None:
            ordered = sorted(
                rows,
                key=lambda candidate: (
                    finite_float(candidate.get("candidate_score"), float("inf")),
                    finite_float(candidate.get("plane_rms"), float("inf")),
                    int(candidate_track_id(candidate)),
                ),
            )[: int(limit)]
            for candidate in ordered:
                signature = candidate_geometry_signature(candidate)
                if signature in seen_source_keys:
                    continue
                seen_source_keys.add(signature)
                plane = candidate_plane(candidate)
                centroid = candidate_hull_centroid(candidate)
                if plane is None or centroid is None:
                    continue
                point, normal = plane
                plane_source_entries.append(
                    {
                        "source": str(source),
                        "candidate": candidate,
                        "candidate_id": int(candidate_track_id(candidate)),
                        "point": point,
                        "normal": normal,
                        "offset": float(normal @ point),
                        "centroid": centroid,
                    }
                )

        add_plane_sources(baseline_active_candidates, "active_production", limit=400)
        add_plane_sources(
            [
                candidate
                for candidate in w2_raw_candidates
                if int(candidate_track_id(candidate)) not in active_ids
            ],
            "current_w2_adjacency",
            limit=600,
        )
        add_plane_sources(
            [
                candidate
                for candidate in valley_raw_candidates
                if str(candidate.get("candidate_origin") or "") != "w2_addition"
                and int(candidate_track_id(candidate)) not in active_ids
            ],
            "baseline_valley",
            limit=600,
        )
        line_derived_candidates = [
            dict(row["candidate"])
            for mode_name, rows in downstream_mode_rows.items()
            if mode_name in set(selected_line_modes + ["few_level_spatial_detector"])
            for row in rows
        ]
        add_plane_sources(line_derived_candidates, "robust_line_derived", limit=600)
        source_normals = np.array(
            [np.asarray(entry["normal"], dtype=float) for entry in plane_source_entries],
            dtype=float,
        ).reshape((-1, 3))
        source_offsets = np.array(
            [float(entry["offset"]) for entry in plane_source_entries],
            dtype=float,
        )
        source_centroids = np.array(
            [np.asarray(entry["centroid"], dtype=float) for entry in plane_source_entries],
            dtype=float,
        ).reshape((-1, 3))

        def plane_consensus_features(candidate: dict[str, object]) -> dict[str, object]:
            plane = candidate_plane(candidate)
            centroid = candidate_hull_centroid(candidate)
            if plane is None or centroid is None:
                return {
                    "independent_plane_sources": 0,
                    "independent_candidate_count": 0,
                    "plane_normal_spread": None,
                    "plane_offset_spread": None,
                    "patch_overlap": 0.0,
                    "matching_sources": [],
                }
            point, normal = plane
            offset = float(normal @ point)
            matches: list[dict[str, object]] = []
            if source_normals.shape[0]:
                dots = source_normals @ normal
                normal_mask = np.abs(dots) >= math.cos(math.radians(3.0))
                aligned_offsets = source_offsets * np.where(dots >= 0.0, 1.0, -1.0)
                offset_deltas = np.abs(aligned_offsets - offset)
                centroid_distances = np.linalg.norm(source_centroids - centroid[None, :], axis=1)
                compatible_indices = np.flatnonzero(
                    normal_mask
                    & (offset_deltas <= 0.05)
                    & (centroid_distances <= 0.65)
                )
            else:
                dots = np.zeros(0, dtype=float)
                offset_deltas = np.zeros(0, dtype=float)
                compatible_indices = np.zeros(0, dtype=int)
            for source_index in compatible_indices:
                entry = plane_source_entries[int(source_index)]
                angle = float(
                    np.degrees(
                        np.arccos(np.clip(abs(float(dots[int(source_index)])), -1.0, 1.0))
                    )
                )
                offset_delta = float(offset_deltas[int(source_index)])
                overlap = projected_patch_overlap(candidate, entry["candidate"])
                if overlap < 0.10:
                    continue
                candidate_z = candidate_z_range(candidate)
                other_z = candidate_z_range(entry["candidate"])
                if all(np.isfinite(value) for value in (*candidate_z, *other_z)):
                    z_overlap = max(0.0, min(candidate_z[1], other_z[1]) - max(candidate_z[0], other_z[0]))
                    if z_overlap <= 0.0:
                        continue
                matches.append(
                    {
                        "source": str(entry["source"]),
                        "candidate_id": int(entry["candidate_id"]),
                        "normal_delta_deg": as_json_float(angle),
                        "offset_delta": as_json_float(offset_delta),
                        "patch_overlap": as_json_float(overlap),
                    }
                )
            sources = sorted({str(value["source"]) for value in matches})
            return {
                "independent_plane_sources": int(len(sources)),
                "independent_candidate_count": int(len(matches)),
                "plane_normal_spread": as_json_float(
                    max((finite_float(value.get("normal_delta_deg"), 0.0) for value in matches), default=float("inf"))
                ),
                "plane_offset_spread": as_json_float(
                    max((finite_float(value.get("offset_delta"), 0.0) for value in matches), default=float("inf"))
                ),
                "patch_overlap": as_json_float(
                    max((finite_float(value.get("patch_overlap"), 0.0) for value in matches), default=0.0)
                ),
                "matching_sources": sources,
                "matches": sorted(
                    matches,
                    key=lambda value: (
                        -finite_float(value.get("patch_overlap"), 0.0),
                        finite_float(value.get("normal_delta_deg"), float("inf")),
                        int(value.get("candidate_id") or 0),
                    ),
                )[:12],
            }

        anchor_entries_all = [
            entry
            for entry in plane_source_entries
            if str(entry["source"]) in {"active_production", "robust_line_derived"}
        ]
        anchor_entries = [
            entry
            for entry in anchor_entries_all
            if str(entry["source"]) == "active_production"
        ] + [
            entry
            for entry in anchor_entries_all
            if str(entry["source"]) == "robust_line_derived"
        ][:200]
        anchor_normals = np.array(
            [np.asarray(entry["normal"], dtype=float) for entry in anchor_entries],
            dtype=float,
        ).reshape((-1, 3))

        def line_interval_overlap(
            endpoints: np.ndarray,
            anchor_hull: np.ndarray,
            line_point: np.ndarray,
            line_direction: np.ndarray,
        ) -> float:
            segment_projection = (endpoints - line_point) @ line_direction
            segment_min = float(np.min(segment_projection))
            segment_max = float(np.max(segment_projection))
            segment_length = max(segment_max - segment_min, 1e-6)
            anchor_projection = (anchor_hull - line_point) @ line_direction
            if anchor_projection.size == 0:
                return 0.0
            overlap = max(
                0.0,
                min(segment_max, float(np.max(anchor_projection)))
                - max(segment_min, float(np.min(anchor_projection))),
            )
            return float(overlap / segment_length)

        def anchor_features(
            candidate: dict[str, object],
            source_hypotheses: list[dict[str, object]],
        ) -> dict[str, object]:
            plane = candidate_plane(candidate)
            if plane is None or not source_hypotheses:
                return {
                    "anchor_plane_count": 0,
                    "best_intersection_distance": None,
                    "intersection_direction_error": None,
                    "anchored_overlap": 0.0,
                    "neighbor_plane_support": 0,
                    "anchors": [],
                }
            point, normal = plane
            offset = float(normal @ point)
            anchors: list[dict[str, object]] = []
            for hypothesis in source_hypotheses:
                endpoints = np.asarray(hypothesis.get("endpoints"), dtype=float).reshape((2, 3))
                hypothesis_direction = canonical_direction(np.asarray(hypothesis.get("direction"), dtype=float))
                cross_directions = np.cross(normal[None, :], anchor_normals)
                cross_norms = np.linalg.norm(cross_directions, axis=1)
                normalized_directions = np.zeros_like(cross_directions)
                valid_cross = cross_norms > 1e-5
                normalized_directions[valid_cross] = (
                    cross_directions[valid_cross] / cross_norms[valid_cross, None]
                )
                direction_match = (
                    np.abs(normalized_directions @ hypothesis_direction)
                    >= math.cos(math.radians(15.0))
                )
                anchor_indices = np.flatnonzero(valid_cross & direction_match)
                for anchor_index in anchor_indices:
                    entry = anchor_entries[int(anchor_index)]
                    anchor_normal = np.asarray(entry["normal"], dtype=float)
                    direction = canonical_direction(normalized_directions[int(anchor_index)])
                    matrix = np.vstack([normal, anchor_normal])
                    gram = matrix @ matrix.T
                    try:
                        weights = np.linalg.solve(
                            gram,
                            np.array([offset, float(entry["offset"])], dtype=float),
                        )
                    except np.linalg.LinAlgError:
                        continue
                    line_point = matrix.T @ weights
                    distances = np.linalg.norm(
                        np.cross(endpoints - line_point[None, :], direction[None, :]),
                        axis=1,
                    )
                    distance = float(np.median(distances))
                    direction_error = angle_between(hypothesis_direction, direction)
                    if distance > 0.08 or direction_error > 15.0:
                        continue
                    overlap = line_interval_overlap(
                        endpoints,
                        candidate_hull_points(entry["candidate"]),
                        line_point,
                        direction,
                    )
                    if overlap < 0.10:
                        continue
                    support = int(
                        len(entry["candidate"].get("sample_points") or []) > 0
                        or finite_float(entry["candidate"].get("observed_local_support"), 0.0) > 0.0
                        or finite_float(entry["candidate"].get("fit_points"), 0.0) >= 3.0
                    )
                    if support <= 0:
                        continue
                    anchors.append(
                        {
                            "source": str(entry["source"]),
                            "candidate_id": int(entry["candidate_id"]),
                            "source_hypothesis_id": int(hypothesis.get("hypothesis_id") or -1),
                            "intersection_distance": as_json_float(distance),
                            "direction_error_deg": as_json_float(direction_error),
                            "anchored_overlap": as_json_float(overlap),
                            "observed_support": int(support),
                        }
                    )
            unique_anchors: dict[tuple[str, int], dict[str, object]] = {}
            for anchor in anchors:
                key = (str(anchor["source"]), int(anchor["candidate_id"]))
                previous = unique_anchors.get(key)
                if previous is None or (
                    finite_float(anchor.get("intersection_distance"), float("inf")),
                    finite_float(anchor.get("direction_error_deg"), float("inf")),
                ) < (
                    finite_float(previous.get("intersection_distance"), float("inf")),
                    finite_float(previous.get("direction_error_deg"), float("inf")),
                ):
                    unique_anchors[key] = anchor
            rows = sorted(
                unique_anchors.values(),
                key=lambda value: (
                    finite_float(value.get("intersection_distance"), float("inf")),
                    finite_float(value.get("direction_error_deg"), float("inf")),
                    -finite_float(value.get("anchored_overlap"), 0.0),
                    int(value.get("candidate_id") or 0),
                ),
            )
            return {
                "anchor_plane_count": int(len(rows)),
                "best_intersection_distance": (
                    rows[0].get("intersection_distance") if rows else None
                ),
                "intersection_direction_error": (
                    min(
                        (finite_float(value.get("direction_error_deg"), float("inf")) for value in rows),
                        default=None,
                    )
                ),
                "anchored_overlap": as_json_float(
                    max((finite_float(value.get("anchored_overlap"), 0.0) for value in rows), default=0.0)
                ),
                "neighbor_plane_support": int(sum(int(value.get("observed_support") or 0) for value in rows)),
                "anchors": rows[:12],
            }

        trusted = np.asarray(trusted_points, dtype=float).reshape((-1, 3))
        active_plane_rows = [candidate_plane(candidate) for candidate in baseline_active_candidates]
        active_plane_rows = [value for value in active_plane_rows if value is not None]
        if active_plane_rows and trusted.shape[0]:
            active_normals = np.array([value[1] for value in active_plane_rows], dtype=float)
            active_offsets = np.array([float(value[1] @ value[0]) for value in active_plane_rows], dtype=float)
            baseline_plane_distance = np.min(
                np.abs(trusted @ active_normals.T - active_offsets[None, :]),
                axis=1,
            )
        else:
            baseline_plane_distance = np.full(trusted.shape[0], float("inf"), dtype=float)

        def marginal_contour_features(candidate: dict[str, object]) -> dict[str, object]:
            plane = candidate_plane(candidate)
            halfspace = candidate_halfspace(candidate)
            if plane is None or halfspace is None or trusted.shape[0] == 0:
                return {
                    "trial_active": False,
                    "local_support_points": 0,
                    "local_contour_support_deficit_before": None,
                    "local_contour_support_deficit_after": None,
                    "local_contour_deficit_reduction": 0.0,
                    "unsupported_fraction_reduction": 0.0,
                    "trusted_cut_fraction": 1.0,
                }
            point, normal = plane
            hull = candidate_hull_points(candidate)
            u, v = plane_basis(normal)
            hull_2d = convex_hull_2d(
                np.column_stack([(hull - point) @ u, (hull - point) @ v])
            )
            plane_distance = np.abs((trusted - point) @ normal)
            projected = np.column_stack([(trusted - point) @ u, (trusted - point) @ v])
            hull_distance = point_to_polygon_distance_2d(projected, hull_2d)
            z_min, z_max = candidate_z_range(candidate)
            z_mask = np.ones(trusted.shape[0], dtype=bool)
            if np.isfinite(z_min) and np.isfinite(z_max):
                z_mask = (trusted[:, 2] >= z_min - 0.08) & (trusted[:, 2] <= z_max + 0.08)
            local_mask = z_mask & (plane_distance <= 0.12) & (hull_distance <= 0.18)
            local_indices = np.flatnonzero(local_mask)
            if local_indices.size > 320:
                order = np.argsort(plane_distance[local_indices] + hull_distance[local_indices])
                local_indices = local_indices[order[:320]]
            candidate_distance = np.sqrt(
                plane_distance[local_indices] ** 2 + hull_distance[local_indices] ** 2
            ) if local_indices.size else np.zeros(0, dtype=float)
            before = baseline_plane_distance[local_indices] if local_indices.size else np.zeros(0, dtype=float)
            after = np.minimum(before, candidate_distance) if local_indices.size else np.zeros(0, dtype=float)
            reduction = before - after
            unsupported_before = before > 0.05
            unsupported_after = after > 0.05
            unsupported_fraction_reduction = (
                float(np.mean(unsupported_before) - np.mean(unsupported_after))
                if local_indices.size
                else 0.0
            )
            hs_normal, hs_offset = halfspace
            trusted_cut = trusted @ hs_normal - float(hs_offset) > float(point_tol)
            trial_geometry = trial_face_geometry_row_incremental(
                candidate=candidate,
                current_reconstructed=final_reconstructed,
                candidate_index=len(final_candidates),
                reconstruction_kwargs=reconstruction_kwargs,
            )
            return {
                "trial_active": bool(trial_geometry.get("trial_active")),
                "trial_reason": trial_geometry.get("reason"),
                "trial_face_area": trial_geometry.get("face_area"),
                "trial_hull_containment_fraction": trial_geometry.get("hull_to_face_containment_fraction"),
                "local_support_points": int(local_indices.size),
                "local_contour_support_deficit_before": (
                    as_json_float(float(np.median(before))) if before.size else None
                ),
                "local_contour_support_deficit_after": (
                    as_json_float(float(np.median(after))) if after.size else None
                ),
                "local_contour_deficit_reduction": (
                    as_json_float(float(np.median(reduction))) if reduction.size else 0.0
                ),
                "unsupported_fraction_reduction": as_json_float(unsupported_fraction_reduction),
                "observed_points_to_trial_face_p95": (
                    as_json_float(float(np.percentile(candidate_distance, 95)))
                    if candidate_distance.size
                    else None
                ),
                "trusted_cut_fraction": as_json_float(float(np.mean(trusted_cut))),
            }

        def aggregate_hypothesis_features(
            hypotheses: list[dict[str, object]],
        ) -> tuple[dict[str, object], dict[str, object]]:
            evidence = [
                hypothesis_independent_cache[int(row.get("hypothesis_id") or -1)]
                for row in hypotheses
                if int(row.get("hypothesis_id") or -1) in hypothesis_independent_cache
            ]
            scales = [dict(value["cross_scale"]) for value in evidence]
            transitions = [dict(value["transition"]) for value in evidence]
            best_scale = max(
                scales,
                key=lambda value: (
                    int(value.get("number_of_supporting_scales") or 0),
                    finite_float(value.get("finite_overlap_fraction"), 0.0),
                    -finite_float(value.get("cross_scale_position_spread"), float("inf")),
                ),
                default={},
            )
            best_transition = max(
                transitions,
                key=lambda value: (
                    int(value.get("number_of_supporting_level_pairs") or 0),
                    finite_float(value.get("transition_prominence"), 0.0),
                    finite_float(value.get("neighbor_consistency"), 0.0),
                ),
                default={},
            )
            return dict(best_scale), dict(best_transition)

        def evidence_row(
            *,
            row_id: int,
            candidate: dict[str, object],
            hypotheses: list[dict[str, object]],
            source_hypothesis_ids: list[int],
            pair: list[int],
        ) -> dict[str, object]:
            scale, transition = aggregate_hypothesis_features(hypotheses)
            consensus = plane_consensus_features(candidate)
            anchor = anchor_features(candidate, hypotheses)
            marginal = marginal_contour_features(candidate)
            cross_pass = bool(
                int(scale.get("number_of_supporting_scales") or 0) >= 2
                and finite_float(scale.get("cross_scale_position_spread"), float("inf")) <= 0.12
                and finite_float(scale.get("cross_scale_direction_spread"), float("inf")) <= 25.0
                and finite_float(scale.get("finite_overlap_fraction"), 0.0) >= 0.10
            )
            transition_pass = bool(
                finite_float(transition.get("transition_prominence"), 0.0) >= 0.35
                and finite_float(transition.get("neighbor_consistency"), 0.0) >= 0.40
                and int(transition.get("number_of_supporting_level_pairs") or 0) >= 1
            )
            consensus_pass = bool(
                int(consensus.get("independent_plane_sources") or 0) >= 2
                and finite_float(consensus.get("plane_normal_spread"), float("inf")) <= 3.0
                and finite_float(consensus.get("plane_offset_spread"), float("inf")) <= 0.05
                and finite_float(consensus.get("patch_overlap"), 0.0) >= 0.10
            )
            anchor_pass = bool(
                int(anchor.get("anchor_plane_count") or 0) >= 1
                and finite_float(anchor.get("best_intersection_distance"), float("inf")) <= 0.08
                and finite_float(anchor.get("intersection_direction_error"), float("inf")) <= 15.0
                and finite_float(anchor.get("anchored_overlap"), 0.0) >= 0.10
                and int(anchor.get("neighbor_plane_support") or 0) >= 1
            )
            marginal_pass = bool(
                bool(marginal.get("trial_active"))
                and int(marginal.get("local_support_points") or 0) >= 3
                and finite_float(marginal.get("local_contour_deficit_reduction"), 0.0) >= 0.002
                and finite_float(marginal.get("unsupported_fraction_reduction"), 0.0) >= 0.02
                and finite_float(marginal.get("trusted_cut_fraction"), 1.0) <= 0.01
            )
            passes = {
                "A_cross_scale": cross_pass,
                "B_transition": transition_pass,
                "C_plane_consensus": consensus_pass,
                "D_intersection_anchor": anchor_pass,
                "E_marginal_contour": marginal_pass,
            }
            evidence_count = int(sum(passes.values()))
            return {
                "row_id": int(row_id),
                "candidate_id": int(candidate_track_id(candidate)),
                "pair": [int(value) for value in pair],
                "source_hypothesis_ids": [int(value) for value in source_hypothesis_ids],
                "source_hypothesis_count": int(len(hypotheses)),
                "cross_scale": scale,
                "transition": transition,
                "plane_consensus": consensus,
                "intersection_anchor": anchor,
                "marginal_contour": marginal,
                "signal_family_passes": passes,
                "combined_evidence_count": int(evidence_count),
                "combined_evidence_pass": bool(evidence_count >= 2),
                "_candidate": candidate,
            }

        target_mode_rows = list(downstream_mode_rows.get(selected_mixed_mode, []))
        target_preoracle_rows: list[dict[str, object]] = []
        for row_index, target_row in enumerate(target_mode_rows):
            source_ids = [int(value) for value in target_row.get("source_hypothesis_ids", [])]
            source_hypotheses = [
                hypothesis_by_id[value]
                for value in source_ids
                if value in hypothesis_by_id
            ]
            target_preoracle_rows.append(
                evidence_row(
                    row_id=row_index,
                    candidate=dict(target_row["candidate"]),
                    hypotheses=source_hypotheses,
                    source_hypothesis_ids=source_ids,
                    pair=[int(value) for value in target_row.get("pair", [])],
                )
            )

        control_preoracle_rows: list[dict[str, object]] = []
        for row_index, (candidate, hypotheses) in enumerate(active_w2_control_candidates):
            control_preoracle_rows.append(
                evidence_row(
                    row_id=row_index,
                    candidate=candidate,
                    hypotheses=hypotheses,
                    source_hypothesis_ids=[int(value.get("hypothesis_id") or -1) for value in hypotheses],
                    pair=[int(value) for value in candidate.get("w2_edge_cluster_ids", [])],
                )
            )

        negative_rows: list[dict[str, object]] = []
        negative_reason_counts: dict[str, int] = {}
        for row in target_preoracle_rows:
            marginal = dict(row.get("marginal_contour") or {})
            reasons: list[str] = []
            if not bool(marginal.get("trial_active")):
                reasons.append("trial_inactive")
            if int(marginal.get("local_support_points") or 0) == 0:
                reasons.append("no_spatial_support")
            if finite_float(marginal.get("trusted_cut_fraction"), 0.0) > 0.10:
                reasons.append("cuts_trusted_points")
            scale = dict(row.get("cross_scale") or {})
            if (
                finite_float(scale.get("cross_scale_position_spread"), 0.0) > 0.25
                or finite_float(scale.get("cross_scale_direction_spread"), 0.0) > 45.0
            ):
                reasons.append("grossly_unstable_across_scales")
            if reasons:
                value = {**row, "negative_reasons": reasons}
                negative_rows.append(value)
                for reason in reasons:
                    negative_reason_counts[reason] = negative_reason_counts.get(reason, 0) + 1

        validator_order = [
            "A_cross_scale",
            "B_transition",
            "C_plane_consensus",
            "D_intersection_anchor",
            "E_marginal_contour",
            "F_combined_evidence",
        ]

        def validator_pass(row: dict[str, object], name: str) -> bool:
            if name == "F_combined_evidence":
                return bool(row.get("combined_evidence_pass"))
            return bool((row.get("signal_family_passes") or {}).get(name))

        calibration_rows: list[dict[str, object]] = []
        for priority, validator_name in enumerate(validator_order):
            positive_accept = int(sum(validator_pass(row, validator_name) for row in control_preoracle_rows))
            negative_accept = int(sum(validator_pass(row, validator_name) for row in negative_rows))
            target_accept = int(sum(validator_pass(row, validator_name) for row in target_preoracle_rows))
            positive_rate = float(positive_accept / max(1, len(control_preoracle_rows)))
            negative_rate = float(negative_accept / max(1, len(negative_rows)))
            explosion_penalty = max(0.0, target_accept - 200.0) / max(1.0, float(len(target_preoracle_rows)))
            calibration_score = positive_rate - negative_rate - 0.25 * explosion_penalty
            calibration_rows.append(
                {
                    "name": str(validator_name),
                    "priority": int(priority),
                    "calibration_positive_accepted": int(positive_accept),
                    "calibration_positive_total": int(len(control_preoracle_rows)),
                    "calibration_positive_acceptance_rate": as_json_float(positive_rate),
                    "calibration_negative_accepted": int(negative_accept),
                    "calibration_negative_total": int(len(negative_rows)),
                    "calibration_negative_acceptance_rate": as_json_float(negative_rate),
                    "target_candidate_rows_accepted": int(target_accept),
                    "calibration_score": as_json_float(calibration_score),
                }
            )
        selected_validator_row = max(
            calibration_rows,
            key=lambda value: (
                finite_float(value.get("calibration_score"), -float("inf")),
                finite_float(value.get("calibration_positive_acceptance_rate"), 0.0),
                -int(value.get("target_candidate_rows_accepted") or 0),
                -int(value.get("priority") or 0),
            ),
            default={"name": "F_combined_evidence"},
        )
        selected_validator = str(selected_validator_row.get("name") or "F_combined_evidence")

        known_safe_ids = {
            int(row["candidate_level_target_face_id"])
            for row in full_trial_rows
            if str(row.get("classification")) == "safe_plus_one"
            and bool(row.get("safety_pass"))
        }
        validator_evaluations: list[dict[str, object]] = []
        for calibration in calibration_rows:
            validator_name = str(calibration["name"])
            accepted_indices = [
                index
                for index, row in enumerate(target_preoracle_rows)
                if validator_pass(row, validator_name)
            ]
            accepted_source_ids = {
                int(value)
                for index in accepted_indices
                for value in target_preoracle_rows[index].get("source_hypothesis_ids", [])
            }
            accepted_hypothesis_rows = [
                hypothesis_by_id[value]
                for value in sorted(accepted_source_ids)
                if value in hypothesis_by_id
            ]
            accepted_hypothesis_matches = [
                label_hypothesis(value) for value in accepted_hypothesis_rows
            ]
            pure_hypothesis_matches = [
                value
                for value in accepted_hypothesis_matches
                if str(value.get("classification")) == "confident_pure"
            ]
            pure_edge_ids = {
                int(value["dominant_edge_id"])
                for value in pure_hypothesis_matches
                if value.get("dominant_edge_id") is not None
            }
            accepted_oracle_rows = [target_mode_rows[index] for index in accepted_indices]
            finite_rows = [row for row in accepted_oracle_rows if bool(row.get("finite_good"))]
            finite_ids = {
                int(row["oracle_face_id"])
                for row in finite_rows
                if row.get("oracle_face_id") is not None
            }
            validator_evaluations.append(
                {
                    **calibration,
                    "accepted_hypotheses": int(len(accepted_source_ids)),
                    "confident_pure_hypotheses_posthoc": int(len(pure_hypothesis_matches)),
                    "hypothesis_precision_posthoc": as_json_float(
                        float(len(pure_hypothesis_matches) / max(1, len(accepted_hypothesis_rows)))
                    ),
                    "precision": as_json_float(
                        float(len(pure_hypothesis_matches) / max(1, len(accepted_hypothesis_rows)))
                    ),
                    "unique_confident_edge_ids_posthoc": sorted(int(value) for value in pure_edge_ids),
                    "accepted_candidate_rows": int(len(accepted_indices)),
                    "finite_good_posthoc": int(len(finite_rows)),
                    "candidate_finite_good_precision_posthoc": as_json_float(
                        float(len(finite_rows) / max(1, len(accepted_indices)))
                    ),
                    "unique_recovered_ids": sorted(int(value) for value in finite_ids),
                    "unique_recovered_id_count": int(len(finite_ids)),
                    "known_safe_ids_recovered": sorted(int(value) for value in finite_ids & known_safe_ids),
                    "known_safe_id_recall": as_json_float(float(len(finite_ids & known_safe_ids) / max(1, len(known_safe_ids)))),
                    "false_hypotheses": int(len(accepted_indices) - len(finite_rows)),
                    "current_control_positives_rejected": int(
                        len(control_preoracle_rows)
                        - int(calibration.get("calibration_positive_accepted") or 0)
                    ),
                    "candidate_explosion": int(len(accepted_indices)),
                    "selected_by_nonoracle_calibration": bool(validator_name == selected_validator),
                }
            )

        combined_frontier: list[dict[str, object]] = []
        for minimum_count in range(1, 6):
            accepted_indices = [
                index
                for index, row in enumerate(target_preoracle_rows)
                if int(row.get("combined_evidence_count") or 0) >= minimum_count
            ]
            oracle_rows = [target_mode_rows[index] for index in accepted_indices]
            finite_rows = [row for row in oracle_rows if bool(row.get("finite_good"))]
            finite_ids = {
                int(row["oracle_face_id"])
                for row in finite_rows
                if row.get("oracle_face_id") is not None
            }
            combined_frontier.append(
                {
                    "minimum_evidence_families": int(minimum_count),
                    "accepted_candidate_rows": int(len(accepted_indices)),
                    "finite_good_posthoc": int(len(finite_rows)),
                    "precision": as_json_float(float(len(finite_rows) / max(1, len(accepted_indices)))),
                    "unique_recovered_id_count": int(len(finite_ids)),
                    "known_safe_ids_recovered": sorted(int(value) for value in finite_ids & known_safe_ids),
                    "known_safe_id_recall": as_json_float(float(len(finite_ids & known_safe_ids) / max(1, len(known_safe_ids)))),
                }
            )

        signal_scores = {
            "cross_scale": lambda row: (
                float(int((row.get("cross_scale") or {}).get("number_of_supporting_scales") or 0))
                + finite_float((row.get("cross_scale") or {}).get("finite_overlap_fraction"), 0.0)
                - finite_float((row.get("cross_scale") or {}).get("cross_scale_position_spread"), 1.0)
                - finite_float((row.get("cross_scale") or {}).get("cross_scale_direction_spread"), 90.0) / 90.0
            ),
            "transition": lambda row: (
                finite_float((row.get("transition") or {}).get("transition_prominence"), 0.0)
                + finite_float((row.get("transition") or {}).get("neighbor_consistency"), 0.0)
                + float(int((row.get("transition") or {}).get("number_of_supporting_level_pairs") or 0))
            ),
            "plane_consensus": lambda row: (
                float(int((row.get("plane_consensus") or {}).get("independent_plane_sources") or 0))
                + finite_float((row.get("plane_consensus") or {}).get("patch_overlap"), 0.0)
            ),
            "intersection_anchor": lambda row: (
                float(int((row.get("intersection_anchor") or {}).get("anchor_plane_count") or 0))
                + finite_float((row.get("intersection_anchor") or {}).get("anchored_overlap"), 0.0)
                - finite_float((row.get("intersection_anchor") or {}).get("best_intersection_distance"), 1.0)
            ),
            "marginal_contour": lambda row: (
                finite_float((row.get("marginal_contour") or {}).get("local_contour_deficit_reduction"), 0.0)
                + finite_float((row.get("marginal_contour") or {}).get("unsupported_fraction_reduction"), 0.0)
                - finite_float((row.get("marginal_contour") or {}).get("trusted_cut_fraction"), 1.0)
            ),
        }
        signal_family_rows: dict[str, dict[str, object]] = {}
        oracle_labels = [bool(row.get("finite_good")) for row in target_mode_rows]
        for family, scorer in signal_scores.items():
            values = [float(scorer(row)) for row in target_preoracle_rows]
            auc = binary_auc(list(zip(values, oracle_labels)), "higher")
            pass_name = {
                "cross_scale": "A_cross_scale",
                "transition": "B_transition",
                "plane_consensus": "C_plane_consensus",
                "intersection_anchor": "D_intersection_anchor",
                "marginal_contour": "E_marginal_contour",
            }[family]
            signal_family_rows[family] = {
                "score_direction": "higher",
                "direction_pre_registered": True,
                "auc": as_json_float(auc) if auc is not None else None,
                "target_pass_count": int(sum(validator_pass(row, pass_name) for row in target_preoracle_rows)),
                "calibration_positive_pass_count": int(sum(validator_pass(row, pass_name) for row in control_preoracle_rows)),
                "calibration_negative_pass_count": int(sum(validator_pass(row, pass_name) for row in negative_rows)),
                "finite_good_score_distribution_posthoc": distribution_summary(
                    [value for value, label in zip(values, oracle_labels) if label]
                ),
                "false_score_distribution_posthoc": distribution_summary(
                    [value for value, label in zip(values, oracle_labels) if not label]
                ),
            }
            if family in {"cross_scale", "transition"}:
                hypothesis_values: list[float] = []
                hypothesis_labels: list[bool] = []
                for hypothesis in selected_mixed_hypotheses:
                    hypothesis_id = int(hypothesis.get("hypothesis_id") or -1)
                    evidence = hypothesis_independent_cache[hypothesis_id]
                    if family == "cross_scale":
                        scale = dict(evidence["cross_scale"])
                        score = (
                            float(int(scale.get("number_of_supporting_scales") or 0))
                            + finite_float(scale.get("finite_overlap_fraction"), 0.0)
                            - finite_float(scale.get("cross_scale_position_spread"), 1.0)
                            - finite_float(scale.get("cross_scale_direction_spread"), 90.0) / 90.0
                        )
                    else:
                        transition = dict(evidence["transition"])
                        score = (
                            finite_float(transition.get("transition_prominence"), 0.0)
                            + finite_float(transition.get("neighbor_consistency"), 0.0)
                            + float(int(transition.get("number_of_supporting_level_pairs") or 0))
                        )
                    hypothesis_values.append(float(score))
                    hypothesis_labels.append(
                        str(label_hypothesis(hypothesis).get("classification")) == "confident_pure"
                    )
                hypothesis_auc = binary_auc(
                    list(zip(hypothesis_values, hypothesis_labels)),
                    "higher",
                )
                signal_family_rows[family]["hypothesis_level_auc"] = (
                    as_json_float(hypothesis_auc) if hypothesis_auc is not None else None
                )

        selected_preoracle = [
            row
            for row in target_preoracle_rows
            if validator_pass(row, selected_validator)
        ]
        selected_preoracle.sort(
            key=lambda row: (
                -int(row.get("combined_evidence_count") or 0),
                -finite_float((row.get("marginal_contour") or {}).get("local_contour_deficit_reduction"), 0.0),
                -int((row.get("plane_consensus") or {}).get("independent_plane_sources") or 0),
                -int((row.get("intersection_anchor") or {}).get("anchor_plane_count") or 0),
                int(row.get("row_id") or 0),
            )
        )
        selected_full_rows: list[dict[str, object]] = []
        seen_trial_geometry: set[tuple[object, ...]] = set()
        for row in selected_preoracle:
            candidate = row["_candidate"]
            signature = candidate_geometry_signature(candidate)
            if signature in seen_trial_geometry:
                continue
            seen_trial_geometry.add(signature)
            selected_full_rows.append(row)
            if len(selected_full_rows) >= 20:
                break

        baseline_reprojection = evaluate_polyhedron_reprojection(
            name="production129_independent_validator_baseline",
            vertices_3d=np.asarray(final_reconstructed.get("vertices") or [], dtype=float),
            contours=contours,
            direction_count=72,
        )
        full_rows: list[dict[str, object]] = []
        for trial_index, evidence in enumerate(selected_full_rows):
            target_row = target_mode_rows[int(evidence["row_id"])]
            candidate = dict(evidence["_candidate"])
            candidate["track_id"] = int(9970000 + trial_index)
            trial_candidates = list(final_candidates) + [candidate]
            trial_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
                trial_candidates,
                **reconstruction_kwargs,
            )
            active_indices = [
                int(index)
                for index in trial_reconstructed.get("face_candidate_indices", [])
                if 0 <= int(index) < len(trial_candidates)
            ]
            active_candidates = [trial_candidates[index] for index in active_indices]
            candidate_active = bool(len(trial_candidates) - 1 in active_indices)
            trial_finite_ids = {
                int(match["face_id"])
                for active_candidate in active_candidates
                for match in [candidate_match(active_candidate)]
                if match is not None
                and bool(match.get("finite_good"))
                and match.get("face_id") is not None
            }
            lost_groups = [
                sorted(int(candidate_track_id(member)) for member in group)
                for group in baseline_groups
                if not any(
                    strict_patch_compatible(member, active_candidate)
                    for member in group
                    for active_candidate in active_candidates
                )
            ]
            gained_ids = sorted(int(value) for value in trial_finite_ids - baseline_finite_ids)
            lost_ids = sorted(int(value) for value in baseline_finite_ids - trial_finite_ids)
            posthoc_face_id = target_row.get("oracle_face_id")
            target_active = bool(
                posthoc_face_id is not None and int(posthoc_face_id) in trial_finite_ids
            )
            if not candidate_active:
                classification = "inactive"
            elif lost_groups:
                classification = "rolling_group_loss"
            elif gained_ids and lost_ids:
                classification = "replacement"
            elif gained_ids and target_active and not lost_ids:
                classification = "safe_plus_one"
            elif gained_ids and not target_active:
                classification = "mapping_mismatch"
            else:
                classification = "redundant"
            outside = evaluate_trusted_cloud_outside(
                trial_candidates,
                trusted_points,
                z_slices,
                point_tol=float(point_tol),
            )
            topology = (
                trial_reconstructed.get("topology")
                if isinstance(trial_reconstructed.get("topology"), dict)
                else {}
            )
            reprojection = evaluate_polyhedron_reprojection(
                name=f"independent_validator_trial_{trial_index}",
                vertices_3d=np.asarray(trial_reconstructed.get("vertices") or [], dtype=float),
                contours=contours,
                direction_count=72,
            )
            full_rows.append(
                {
                    "rank": int(trial_index + 1),
                    "candidate_id": int(candidate_track_id(evidence["_candidate"])),
                    "pair": evidence.get("pair"),
                    "source_hypothesis_ids": evidence.get("source_hypothesis_ids"),
                    "combined_evidence_count": int(evidence.get("combined_evidence_count") or 0),
                    "signal_family_passes": evidence.get("signal_family_passes"),
                    "candidate_level_target_face_id_posthoc": (
                        int(posthoc_face_id) if posthoc_face_id is not None else None
                    ),
                    "finite_good_posthoc": bool(target_row.get("finite_good")),
                    "classification": str(classification),
                    "candidate_active": bool(candidate_active),
                    "target_active": bool(target_active),
                    "canonical_ids_gained": gained_ids,
                    "canonical_ids_lost": lost_ids,
                    "known_safe_id_gained": bool(set(gained_ids) & known_safe_ids),
                    "rolling_groups_lost": int(len(lost_groups)),
                    "vertices": int(len(trial_reconstructed.get("vertices", []))),
                    "edges": int(len(trial_reconstructed.get("edges", []))),
                    "faces": int(len(trial_reconstructed.get("faces", []))),
                    "volume": trial_reconstructed.get("reliable_volume"),
                    "topology": topology,
                    "topology_valid": bool(topology_is_valid(topology)),
                    "outside": outside,
                    "reprojection_support_p95": reprojection.get("support_abs_p95"),
                    "reprojection_support_p95_delta": as_json_float(
                        finite_float(reprojection.get("support_abs_p95"), 0.0)
                        - finite_float(baseline_reprojection.get("support_abs_p95"), 0.0)
                    ),
                    "reprojection_symmetric_p95": reprojection.get("symmetric_contour_distance_p95"),
                    "safety_pass": bool(
                        topology_is_valid(topology)
                        and finite_float(outside.get("cumulative_outside_fraction"), 1.0) < 0.01
                        and int(outside.get("cumulative_lost_z_levels") or 0) == 0
                    ),
                }
            )
        full_counts = {
            category: int(sum(str(row.get("classification")) == category for row in full_rows))
            for category in (
                "safe_plus_one",
                "replacement",
                "redundant",
                "inactive",
                "rolling_group_loss",
                "mapping_mismatch",
            )
        }
        selected_evaluation = next(
            (
                row
                for row in validator_evaluations
                if str(row.get("name")) == selected_validator
            ),
            {},
        )
        prospective = bool(
            len(selected_evaluation.get("known_safe_ids_recovered") or []) >= 8
            and finite_float(selected_evaluation.get("hypothesis_precision_posthoc"), 0.0) >= 0.20
            and int(selected_evaluation.get("current_control_positives_rejected") or 0)
            <= max(2, int(math.ceil(0.10 * max(1, len(control_preoracle_rows)))))
            and int(selected_evaluation.get("candidate_explosion") or 0) <= 200
        )
        branch_for_validator = {
            "A_cross_scale": "A_cross_scale_or_transition_validator",
            "B_transition": "A_cross_scale_or_transition_validator",
            "C_plane_consensus": "B_plane_consensus_or_anchor_validator",
            "D_intersection_anchor": "B_plane_consensus_or_anchor_validator",
            "E_marginal_contour": "C_marginal_contour_validator",
            "F_combined_evidence": "D_combined_evidence",
        }
        selected_branch = (
            branch_for_validator.get(selected_validator, "D_combined_evidence")
            if prospective
            else "E_independent_separation_absent"
        )

        def public_evidence(value: dict[str, object]) -> dict[str, object]:
            return {
                key: item
                for key, item in value.items()
                if not str(key).startswith("_")
            }

        hypothesis_mode_summaries: dict[str, dict[str, object]] = {}
        for mode in mixed_split_modes:
            rows = mixed_split_hypotheses[mode]
            cross_pass = 0
            transition_pass = 0
            for row in rows:
                evidence = hypothesis_independent_cache[int(row.get("hypothesis_id") or -1)]
                scale = dict(evidence["cross_scale"])
                transition = dict(evidence["transition"])
                cross_pass += int(
                    int(scale.get("number_of_supporting_scales") or 0) >= 2
                    and finite_float(scale.get("cross_scale_position_spread"), float("inf")) <= 0.12
                    and finite_float(scale.get("cross_scale_direction_spread"), float("inf")) <= 25.0
                    and finite_float(scale.get("finite_overlap_fraction"), 0.0) >= 0.10
                )
                transition_pass += int(
                    finite_float(transition.get("transition_prominence"), 0.0) >= 0.35
                    and finite_float(transition.get("neighbor_consistency"), 0.0) >= 0.40
                    and int(transition.get("number_of_supporting_level_pairs") or 0) >= 1
                )
            hypothesis_mode_summaries[mode] = {
                "hypotheses": int(len(rows)),
                "cross_scale_pass": int(cross_pass),
                "transition_pass": int(transition_pass),
            }

        return {
            "diagnostic_only": True,
            "production_selector_changed": False,
            "uses_initial_model": True,
            "leakage_audit": {
                "feature_direction_chosen_before_oracle_labels": True,
                "threshold_source": (
                    "fixed observed-geometry tolerances registered in code before attaching "
                    "finite-good, canonical face, or full-mesh labels"
                ),
                "calibration_cohort": {
                    "positive_definition": (
                        "active current production W2 candidates with stable observed cluster support, "
                        "excluding mixed-split parent segments"
                    ),
                    "positive_count": int(len(control_preoracle_rows)),
                    "negative_definition": (
                        "target trials that are inactive, have no local observed support, cut more than "
                        "10% trusted points, or are grossly unstable across scales"
                    ),
                    "negative_count": int(len(negative_rows)),
                    "negative_reason_counts": {
                        key: int(value) for key, value in sorted(negative_reason_counts.items())
                    },
                },
                "target_cohort": {
                    "all_mixed_split_hypotheses": int(len(all_mixed_hypotheses)),
                    "selected_immutable_mode": str(selected_mixed_mode),
                    "selected_hypotheses": int(len(selected_mixed_hypotheses)),
                    "selected_candidate_rows": int(len(target_mode_rows)),
                    "selection_uses_oracle": False,
                },
                "oracle_used_only_for_posthoc": True,
                "oracle_fields_forbidden_from_features_and_order": [
                    "InitialModel geometry",
                    "canonical face IDs",
                    "finite_good",
                    "retained/new/unique",
                    "full-mesh outcome labels",
                ],
                "feature_directions": direction_registry,
            },
            "pre_registered_thresholds": fixed_thresholds,
            "cross_scale": {
                "available_scales": [int(value[0]) for value in scale_inputs],
                "matching_uses_exact_cyclic_index": False,
                "matching_fields": ["3D proximity", "direction", "finite overlap", "Z proximity"],
                "mode_summaries": hypothesis_mode_summaries,
            },
            "signal_families": signal_family_rows,
            "plane_source_pool": {
                "candidate_count": int(len(plane_source_entries)),
                "source_counts": {
                    source: int(sum(str(value["source"]) == source for value in plane_source_entries))
                    for source in sorted({str(value["source"]) for value in plane_source_entries})
                },
                "same_split_variants_counted_as_independent": False,
            },
            "calibration": {
                "validator_selection_objective": (
                    "maximize positive acceptance minus negative acceptance, then prefer bounded target pool; "
                    "no oracle term"
                ),
                "validator_rows_preoracle": calibration_rows,
                "selected_validator": str(selected_validator),
            },
            "funnel_counts": {
                "all_mixed_split_hypotheses": int(len(all_mixed_hypotheses)),
                "selected_mode_hypotheses": int(len(selected_mixed_hypotheses)),
                "target_candidate_rows": int(len(target_mode_rows)),
                "calibration_positive_rows": int(len(control_preoracle_rows)),
                "calibration_negative_rows": int(len(negative_rows)),
                "selected_validator_candidate_rows": int(len(selected_preoracle)),
                "deduplicated_full_trial_rows": int(len(full_rows)),
            },
            "validator_evaluation_posthoc": validator_evaluations,
            "combined_evidence_pr_frontier_posthoc": combined_frontier,
            "known_safe_control": {
                "source": "previous bounded full-edge-clip safe_plus_one outcomes",
                "ids": sorted(int(value) for value in known_safe_ids),
                "count": int(len(known_safe_ids)),
                "used_for_feature_or_threshold_selection": False,
            },
            "bounded_full_edge_clip_verification": {
                "selected_validator": str(selected_validator),
                "trial_limit": 20,
                "trial_count": int(len(full_rows)),
                "classification_counts": full_counts,
                "safe_plus_one_precision": as_json_float(
                    float(full_counts["safe_plus_one"] / max(1, len(full_rows)))
                ),
                "baseline_reprojection": baseline_reprojection,
                "rows": full_rows,
            },
            "target_candidate_rows": [public_evidence(row) for row in target_preoracle_rows],
            "calibration_positive_rows": [public_evidence(row) for row in control_preoracle_rows],
            "branch_decision": {
                "selected": str(selected_branch),
                "best_nonoracle_validator": str(selected_validator),
                "production_perspective_passed": bool(prospective),
                "orientation": {
                    "minimum_known_safe_ids": 8,
                    "minimum_hypothesis_precision": 0.20,
                    "reference_unfiltered_mixed_hypothesis_precision": 0.09523809523809523,
                    "maximum_control_positive_loss_fraction": 0.10,
                    "maximum_candidate_rows": 200,
                },
                "production_fix_implemented": False,
            },
        }

    short_edge_independent_validation_audit = run_short_edge_independent_validation()
    independent_validation_elapsed = time.perf_counter() - independent_validation_started
    timings["short_edge_independent_validation_seconds"] = independent_validation_elapsed
    short_edge_independent_validation_audit["timings_seconds"] = {
        "total": as_json_float(independent_validation_elapsed),
    }

    baseline_required_edge_ids = set(
        int(value)
        for value in sensitivity_rows[0].get(
            "required_boundary_edge_ids_recovered", []
        )
    ) if sensitivity_rows else set()
    control_raw_edge_precision = finite_float(
        sensitivity_rows[0].get("precision_confident_clusters"),
        0.0,
    ) if sensitivity_rows else 0.0
    minimum_raw_edge_precision = float(
        max(0.25, 0.75 * control_raw_edge_precision)
    )

    def branch_evidence_row(
        *,
        branch: str,
        mode_name: str,
        recovered_edge_ids: set[int],
        downstream_mode_name: str,
        precision: float,
        control_damage: int,
    ) -> dict[str, object]:
        downstream = downstream_summaries.get(downstream_mode_name, {})
        edge_gain_ids = sorted(int(value) for value in recovered_edge_ids - baseline_required_edge_ids)
        safe_mesh = int(safe_plus_one_by_mode.get(downstream_mode_name, 0))
        new_finite_candidates = int(
            downstream.get("new_finite_good_candidate_count_vs_current") or 0
        )
        downstream_targets = int(
            downstream.get("recovered_adjacency_cohort_count") or 0
        )
        downstream_control_damage = int(
            downstream.get("current_control_faces_lost") or 0
        )
        allowed_control_damage = max(
            5,
            int(math.ceil(0.05 * max(1, len(baseline_control_edge_ids)))),
        )
        controlled_damage = bool(
            int(control_damage) <= allowed_control_damage
            and downstream_control_damage <= 5
        )
        controlled_noise = bool(float(precision) >= minimum_raw_edge_precision)
        acceptance_passed = bool(
            len(edge_gain_ids) >= 5
            and new_finite_candidates >= 10
            and safe_mesh >= 2
            and controlled_damage
            and controlled_noise
            and max_hypothesis_auc >= 0.65
        )
        return {
            "branch": str(branch),
            "mode": str(mode_name),
            "downstream_mode": str(downstream_mode_name),
            "new_required_boundary_edge_count": int(len(edge_gain_ids)),
            "new_required_boundary_edge_ids": edge_gain_ids,
            "raw_edge_precision": as_json_float(float(precision)),
            "minimum_raw_edge_precision": as_json_float(minimum_raw_edge_precision),
            "additional_current_good_edge_controls_lost_vs_control": int(
                control_damage
            ),
            "downstream_control_faces_lost": int(downstream_control_damage),
            "new_finite_good_face_candidates": int(new_finite_candidates),
            "downstream_target_faces_recovered": int(downstream_targets),
            "safe_final_mesh_plus_one": int(safe_mesh),
            "maximum_nonoracle_auc": as_json_float(max_hypothesis_auc),
            "controlled_damage": bool(controlled_damage),
            "controlled_noise": bool(controlled_noise),
            "acceptance_orientation_passed": bool(acceptance_passed),
        }

    branch_evidence: list[dict[str, object]] = []
    for sensitivity_row in sensitivity_rows[1:]:
        recovered = set(
            int(value)
            for value in sensitivity_row.get(
                "required_boundary_edge_ids_recovered", []
            )
        )
        branch_evidence.append(
            branch_evidence_row(
                branch="A_adaptive_short_span_thresholds",
                mode_name=str(sensitivity_row["name"]),
                downstream_mode_name=f"threshold:{sensitivity_row['name']}",
                recovered_edge_ids=recovered,
                precision=finite_float(
                    sensitivity_row.get("precision_confident_clusters"), 0.0
                ),
                control_damage=int(
                    sensitivity_row.get(
                        "additional_current_good_controls_lost_vs_control"
                    ) or 0
                ),
            )
        )
    for mode_name in selected_line_modes:
        summary = representation_hypothesis_summaries[mode_name]
        branch_evidence.append(
            branch_evidence_row(
                branch="B_orthogonal_finite_segment_representation",
                mode_name=mode_name,
                downstream_mode_name=mode_name,
                recovered_edge_ids=set(
                    int(value)
                    for value in summary.get(
                        "required_boundary_edge_ids_recovered", []
                    )
                ),
                precision=finite_float(summary.get("precision"), 0.0),
                control_damage=int(summary.get("current_good_controls_lost") or 0),
            )
        )
    few_summary = representation_hypothesis_summaries[
        "few_level_spatial_detector"
    ]
    branch_evidence.append(
        branch_evidence_row(
            branch="C_few_level_spatial_detector",
            mode_name="few_level_spatial_detector",
            downstream_mode_name="few_level_spatial_detector",
            recovered_edge_ids=set(
                int(value)
                for value in few_summary.get(
                    "required_boundary_edge_ids_recovered", []
                )
            ),
            precision=finite_float(few_summary.get("precision"), 0.0),
            control_damage=int(few_summary.get("current_good_controls_lost") or 0),
        )
    )
    mixed_summary = representation_hypothesis_summaries[selected_mixed_mode]
    branch_evidence.append(
        branch_evidence_row(
            branch="D_mixed_run_splitting",
            mode_name=selected_mixed_mode,
            downstream_mode_name=selected_mixed_mode,
            recovered_edge_ids=set(
                int(value)
                for value in mixed_summary.get(
                    "required_boundary_edge_ids_recovered", []
                )
            ),
            precision=finite_float(mixed_summary.get("precision"), 0.0),
            control_damage=int(mixed_summary.get("current_good_controls_lost") or 0),
        )
    )
    ranked_branch_evidence = sorted(
        branch_evidence,
        key=lambda row: (
            -int(bool(row["acceptance_orientation_passed"])),
            -int(row["safe_final_mesh_plus_one"]),
            -int(row["downstream_target_faces_recovered"]),
            -int(row["new_finite_good_face_candidates"]),
            -int(row["new_required_boundary_edge_count"]),
            -finite_float(row.get("raw_edge_precision"), 0.0),
            str(row["branch"]),
            str(row["mode"]),
        ),
    )
    strongest_evidence = ranked_branch_evidence[0] if ranked_branch_evidence else None
    if strongest_evidence is not None and bool(
        strongest_evidence["acceptance_orientation_passed"]
    ):
        selected_branch = str(strongest_evidence["branch"])
        branch_reason = (
            "The observed-only mode improves required-edge recall, produces at least ten "
            "new finite-good candidates, confirms multiple safe final-mesh faces, and has "
            "a measurable non-oracle separator."
        )
    else:
        selected_branch = "E_raw_observed_signal_insufficient"
        branch_reason = (
            "No observed-only alternative simultaneously meets the bounded edge-recall, "
            "downstream face, final-mesh, control-damage, noise, and separation checks."
        )

    baseline_outside = evaluate_trusted_cloud_outside(
        final_candidates,
        trusted_points,
        z_slices,
        point_tol=float(point_tol),
    )
    baseline_topology = (
        final_reconstructed.get("topology")
        if isinstance(final_reconstructed.get("topology"), dict)
        else {}
    )
    baseline_signature = mesh_geometry_signature(
        final_candidates,
        final_reconstructed,
    )
    baseline_retained = int(len(baseline_finite_ids & set(core_face_ids)))
    baseline_new = int(len(baseline_finite_ids - set(core_face_ids)))
    baseline_volume = finite_float(
        final_reconstructed.get("reliable_volume"),
        float("nan"),
    )
    production_regression_passed = bool(
        baseline_retained == 75
        and baseline_new == 54
        and len(baseline_finite_ids) == 129
        and int(len(final_reconstructed.get("vertices", []))) == 616
        and int(len(final_reconstructed.get("edges", []))) == 924
        and int(len(final_reconstructed.get("faces", []))) == 310
        and np.isfinite(baseline_volume)
        and abs(baseline_volume - 88.83279097892814) <= 1e-9
        and topology_is_valid(baseline_topology)
        and finite_float(
            baseline_outside.get("cumulative_outside_fraction"), 1.0
        ) < 0.01
        and int(baseline_outside.get("cumulative_lost_z_levels") or 0) == 0
    )

    def public_hypothesis(row: dict[str, object]) -> dict[str, object]:
        return {
            key: value
            for key, value in row.items()
            if not str(key).startswith("_")
        }

    hypothesis_samples = {
        mode_name: [
            public_hypothesis(row)
            for row in bounded_hypotheses(rows, limit=40)
        ]
        for mode_name, rows in representation_hypothesis_sets.items()
    }
    timings["total_seconds"] = time.perf_counter() - started
    return {
        "diagnostic_only": True,
        "uses_initial_model": True,
        "production_selector_changed": False,
        "oracle_use": (
            "InitialModel edge/face IDs label cohorts and outcomes only; every fit, split, "
            "hypothesis order, K=4 pair, and full-trial candidate is observed-signal driven."
        ),
        "cohort": {
            "required_boundary_edge_count": int(len(required_edge_ids)),
            "required_boundary_edge_ids": required_edge_ids,
            "short_z_edge_count": int(len(short_edge_ids)),
            "short_z_edge_ids": short_edge_ids,
            "immutable_short_z_expected_count": 49,
            "immutable_short_z_regression_passed": bool(
                immutable_short_cohort_matches
            ),
            "mixed_segment_edge_count": int(len(mixed_required_edge_ids)),
            "mixed_segment_edge_ids": mixed_required_edge_ids,
            "available_required_edge_count": int(len(available_required_edge_ids)),
            "available_required_edge_ids": available_required_edge_ids,
            "problem_face_count": int(len(target_face_ids)),
            "problem_face_ids": target_face_ids,
        },
        "current_line_parameterization": {
            "form": "independent weighted least-squares x(z), y(z)",
            "privileged_axis": "Z",
            "general_orthogonal_3d_fit": False,
            "finite_endpoint_extent": False,
        },
        "short_z_anatomy": {
            "category_order": anatomy_categories,
            "category_counts": anatomy_counts,
            "sum": int(sum(anatomy_counts.values())),
            "expected": int(len(short_edge_ids)),
            "invariant_passed": bool(sum(anatomy_counts.values()) == len(short_edge_ids)),
            "physically_unrepresentable_as_long_z_track": int(
                anatomy_counts["physically_short_or_horizontal_edge"]
            ),
            "rows": anatomy_rows,
        },
        "line_representation_comparison": {
            "fit_point_selection": (
                "Oracle IDs choose the short/available/control cohorts only; every fitted "
                "direction and endpoint uses observed W2/W3 values without oracle geometry."
            ),
            "summary": representation_summary,
            "rows": representation_rows,
        },
        "threshold_sensitivity": {
            "configuration_count": int(len(sensitivity_rows)),
            "small_controlled_grid": True,
            "segment_fit_mode": "robust-split",
            "rows": sensitivity_rows,
            "control_rebuild": {
                "segments_equal": bool(
                    len(sensitivity_segments.get("control_12_levels_0p12", []))
                    == len(w2_segments)
                ),
                "clusters_equal": bool(
                    len(sensitivity_clusters.get("control_12_levels_0p12", []))
                    == len(w2_clusters)
                ),
                "face_candidates_equal": bool(
                    len(sensitivity_candidates.get("control_12_levels_0p12", []))
                    == len(w2_raw_candidates)
                ),
            },
        },
        "raw_edge_hypotheses": {
            "representation_summaries": representation_hypothesis_summaries,
            "ordinary_matched_control_edge_count": int(len(ordinary_control_edge_ids)),
            "baseline_confident_edge_count": int(len(baseline_pure_edge_ids)),
            "baseline_nonrequired_control_edge_count": int(
                len(baseline_control_edge_ids)
            ),
            "sample_limit_per_mode": 40,
            "samples": hypothesis_samples,
        },
        "few_level_spatial_detector": {
            "observed_only": True,
            "w2_local_minimum_candidates_per_level_limit": 16,
            "neighbor_level_radius": 2,
            "w3_confirmation_available": bool(w3_points is not None),
            "summary": representation_hypothesis_summaries[
                "few_level_spatial_detector"
            ],
        },
        "mixed_segment_anatomy": {
            "required_edge_count": int(len(mixed_required_edge_ids)),
            "mixed_source_segment_count": int(len(target_mixed_rows)),
            "rows": mixed_anatomy_rows,
            "split_modes": {
                mode: representation_hypothesis_summaries[f"mixed_split_{mode}"]
                for mode in mixed_split_modes
            },
            "selected_for_downstream": str(selected_mixed_mode),
        },
        "downstream_face_ceiling": {
            "bounded_k": 4,
            "current_shared_z_gate_for_threshold_modes": True,
            "finite_extent_plane_fit_for_general_3d_modes": True,
            "current_k4_control_finite_face_ids": sorted(
                int(value) for value in current_k4_finite_ids
            ),
            "selected_general_3d_modes": custom_downstream_modes,
            "selection_uses_oracle": False,
            "variants": downstream_summaries,
        },
        "bounded_full_edge_clip_verification": {
            "baseline_unique_finite_ids": int(len(baseline_finite_ids)),
            "one_representative_per_recovered_face": True,
            "trial_limit": 15,
            "trial_count": int(len(full_trial_rows)),
            "classification_counts": full_classification_counts,
            "rows": full_trial_rows,
        },
        "short_edge_independent_validation_audit": short_edge_independent_validation_audit,
        "nonoracle_separation": {
            "correct_label": "oracle confident finite edge, evaluation only",
            "false_label": "mixed or unmatched/ambiguous, evaluation only",
            "hypothesis_order_uses_oracle": False,
            "evaluated_short_or_few_level_hypotheses": int(len(separation_rows)),
            "features": hypothesis_separation,
            "maximum_prespecified_direction_auc": as_json_float(
                max_hypothesis_auc
            ),
            "maximum_orientation_neutral_auc_diagnostic_only": as_json_float(
                max_orientation_neutral_auc
            ),
        },
        "branch_decision": {
            "selected": str(selected_branch),
            "reason": str(branch_reason),
            "acceptance_orientation": {
                "minimum_new_required_edges": 5,
                "minimum_new_finite_good_face_candidates": 10,
                "minimum_safe_final_mesh_plus_one": 2,
                "minimum_raw_edge_precision": as_json_float(
                    minimum_raw_edge_precision
                ),
                "minimum_nonoracle_auc": 0.65,
                "maximum_additional_raw_edge_control_losses": max(
                    5,
                    int(math.ceil(0.05 * max(1, len(baseline_control_edge_ids)))),
                ),
                "maximum_downstream_control_face_losses": 5,
            },
            "ranked_evidence": ranked_branch_evidence,
            "strongest_observed_evidence": strongest_evidence,
            "production_fix_implemented": False,
        },
        "production129_regression": {
            "passed": bool(production_regression_passed),
            "retained": int(baseline_retained),
            "new": int(baseline_new),
            "unique": int(len(baseline_finite_ids)),
            "vertices": int(len(final_reconstructed.get("vertices", []))),
            "edges": int(len(final_reconstructed.get("edges", []))),
            "faces": int(len(final_reconstructed.get("faces", []))),
            "volume": final_reconstructed.get("reliable_volume"),
            "topology": baseline_topology,
            "outside": baseline_outside,
            "geometry_signature": baseline_signature,
        },
        "timings_seconds": {
            **{
                key: as_json_float(float(value))
                for key, value in timings.items()
            },
        },
    }


def diagnose_candidate_formation_failures(
    *,
    loss_funnel: dict[str, object],
    initial_vertices: np.ndarray,
    initial_faces: list[list[int]],
    model_faces: list[dict[str, object]],
    z_levels: np.ndarray,
    w2_line_points: np.ndarray | None,
    w2_fit_rms: np.ndarray | None,
    w2_condition: np.ndarray | None,
    w3_line_points: np.ndarray | None,
    w3_fit_rms: np.ndarray | None,
    w4_line_points: np.ndarray | None,
    w4_fit_rms: np.ndarray | None,
    valley_raw_candidates: list[dict[str, object]],
    w2_raw_candidates: list[dict[str, object]],
    w2_segments: list[dict[str, object]],
    w2_clusters: list[dict[str, object]],
    edge_ancestry_internal: dict[str, object],
    inside_point: np.ndarray,
    orientation_points: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    segment_min_levels: int,
    segment_min_z_span: float,
    segment_max_z_gap: int,
    segment_max_line_rms: float,
    segment_max_line_residual: float,
    segment_max_point_jump: float,
    segment_max_condition: float,
    cluster_mode: str,
    cluster_max_distance: float,
    cluster_max_angle_deg: float,
    cluster_min_overlap: float,
    min_adjacency_levels: int,
    min_adjacency_fraction: float,
    min_z_span: float,
    max_plane_rms: float,
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    core_face_ids: set[int],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    match_cache: dict[int, dict[str, object] | None],
    contours: list[object],
) -> dict[str, object]:
    """Oracle-only audit of raw candidates whose best finite patch match fails."""

    started = time.perf_counter()
    timings: dict[str, float] = {}
    mode_timings: dict[str, float] = {}
    point_match_tolerance = 0.08
    method_order = [
        "current_least_squares",
        "robust_trimmed_plane",
        "level_balanced_robust",
        "track_side_balanced",
        "sub_z_consensus",
        "line_derived_plane",
        "robust_hull_reconstruction",
    ]
    cluster_by_id = {
        int(cluster["edge_cluster_id"]): cluster
        for cluster in w2_clusters
        if cluster.get("edge_cluster_id") is not None
    }
    face_by_id = {int(face["face_id"]): face for face in model_faces}
    face_column_by_id = {int(face["face_id"]): index for index, face in enumerate(model_faces)}
    face_rows = {
        int(row["face_id"]): row
        for row in loss_funnel.get("face_stage_matrix", [])
        if row.get("face_id") is not None
    }
    cohort_face_ids = sorted(
        face_id
        for face_id, row in face_rows.items()
        if str(row.get("first_loss_category")) == "raw_candidate_not_finite_good"
    )

    def unique_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        seen: set[int] = set()
        for candidate in rows:
            candidate_id = int(candidate_track_id(candidate))
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            output.append(candidate)
        return output

    raw_candidates = unique_candidates(list(valley_raw_candidates) + list(w2_raw_candidates))

    def current_match(candidate: dict[str, object]) -> dict[str, object] | None:
        candidate_id = int(candidate_track_id(candidate))
        if candidate_id not in match_cache:
            match_cache[candidate_id] = candidate_oracle_face(candidate, model_faces)
        return match_cache.get(candidate_id)

    def source_name(candidate: dict[str, object]) -> str:
        mode = str(candidate.get("small_face_refinement_mode") or "")
        if mode == "append-local" or mode.startswith("post-repair"):
            return "append/broad candidates"
        if str(candidate.get("candidate_origin") or "") == "w2_addition":
            return "W2 adjacency"
        if candidate.get("window") is not None or candidate.get("source_track_id") is not None:
            return "baseline/valley"
        return "other"

    candidate_by_id = {int(candidate_track_id(candidate)): candidate for candidate in raw_candidates}
    raw_candidates_by_face: dict[int, list[tuple[tuple[float, float, float, float, int], dict[str, object]]]] = {}
    for candidate in raw_candidates:
        match = current_match(candidate)
        if match is None or match.get("face_id") is None:
            continue
        raw_candidates_by_face.setdefault(int(match["face_id"]), []).append((oracle_geometry_key(match), candidate))
    best_failure_candidates: list[tuple[int, dict[str, object]]] = []
    missing_best_candidate_rows: list[dict[str, object]] = []
    for face_id in cohort_face_ids:
        options = sorted(
            raw_candidates_by_face.get(face_id, []),
            key=lambda item: (item[0], int(candidate_track_id(item[1]))),
        )
        if options:
            best_failure_candidates.append((face_id, options[0][1]))
            continue
        row = face_rows.get(face_id, {})
        candidate_id = row.get("best_candidate_id")
        candidate = candidate_by_id.get(int(candidate_id)) if candidate_id is not None else None
        if candidate is not None:
            best_failure_candidates.append((face_id, candidate))
        else:
            missing_best_candidate_rows.append({"face_id": int(face_id), "best_candidate_id": candidate_id})

    edges, edge_face_lookup = model_edges_from_faces(initial_vertices, initial_faces)
    edge_id_by_vertices = {tuple(sorted(edge)): int(edge_id) for edge_id, edge in enumerate(edges)}
    face_boundary_edges: dict[int, list[int]] = {}
    face_neighbors: dict[int, set[int]] = {}
    for face_id, face_indices in enumerate(initial_faces):
        boundary: list[int] = []
        neighbors: set[int] = set()
        for index, vertex_id in enumerate(face_indices):
            pair = tuple(sorted((int(vertex_id), int(face_indices[(index + 1) % len(face_indices)]))))
            edge_id = edge_id_by_vertices.get(pair)
            if edge_id is None:
                continue
            boundary.append(int(edge_id))
            neighbors.update(int(v) for v in edge_face_lookup.get(pair, set()) if int(v) != int(face_id))
        face_boundary_edges[int(face_id)] = boundary
        face_neighbors[int(face_id)] = neighbors

    def extract_fit_points(candidate: dict[str, object]) -> dict[str, object]:
        points: list[list[float]] = []
        sides: list[str] = []
        z_keys: list[int] = []
        origins: list[str] = []
        debug_rows = list(candidate.get("debug_observations") or [])
        for observation in debug_rows:
            z_index = int(observation.get("z_index") or 0)
            for key, side in (("left_fit_points", "left"), ("right_fit_points", "right")):
                for point_row in observation.get(key, []) or []:
                    point = np.array(point_row.get("point") or [], dtype=float)
                    if point.shape == (3,) and np.all(np.isfinite(point)):
                        points.append([float(v) for v in point])
                        sides.append(side)
                        z_keys.append(z_index)
                        origins.append("observed_low_region")
        pair = [int(v) for v in candidate.get("w2_edge_cluster_ids", [])]
        candidate_z_indices = [int(v) for v in candidate.get("z_indices", [])]
        if not points and len(pair) == 2 and candidate_z_indices:
            valid_indices = [value for value in candidate_z_indices if 0 <= value < len(z_levels)]
            if valid_indices and pair[0] in cluster_by_id and pair[1] in cluster_by_id:
                z = np.asarray(z_levels[np.array(valid_indices, dtype=int)], dtype=float)
                for side, cluster_id in (("left", pair[0]), ("right", pair[1])):
                    predicted = predict_w2_line(cluster_by_id[cluster_id], z)
                    for point, z_index in zip(predicted, valid_indices):
                        if point.shape == (3,) and np.all(np.isfinite(point)):
                            points.append([float(v) for v in point])
                            sides.append(side)
                            z_keys.append(int(z_index))
                            origins.append("w2_cluster_line_prediction")
        if not points:
            for key, side in (("left_sample_points", "left"), ("right_sample_points", "right")):
                for raw_point in candidate.get(key, []) or []:
                    point = np.array(raw_point, dtype=float)
                    if point.shape == (3,) and np.all(np.isfinite(point)):
                        points.append([float(v) for v in point])
                        sides.append(side)
                        z_keys.append(int(round(float(point[2]) * 1000000.0)))
                        origins.append("candidate_side_sample")
        if not points:
            for raw_point in candidate.get("sample_points", []) or []:
                point = np.array(raw_point, dtype=float)
                if point.shape == (3,) and np.all(np.isfinite(point)):
                    points.append([float(v) for v in point])
                    sides.append("unlabeled")
                    z_keys.append(int(round(float(point[2]) * 1000000.0)))
                    origins.append("candidate_sample")
        array = np.array(points, dtype=float).reshape((-1, 3)) if points else np.zeros((0, 3), dtype=float)
        return {
            "points": array,
            "sides": np.array(sides, dtype=object),
            "z_keys": np.array(z_keys, dtype=int),
            "origins": origins,
        }

    def face_distance_matrix(points: np.ndarray) -> np.ndarray:
        distances = np.full((points.shape[0], len(model_faces)), float("inf"), dtype=float)
        for column, face in enumerate(model_faces):
            normal = np.array(face["normal"], dtype=float)
            face_points = np.array(face["vertices"], dtype=float)
            cache = face.get("_polygon_distance_cache")
            if isinstance(cache, dict):
                origin = np.array(cache["origin"], dtype=float)
                u = np.array(cache["u"], dtype=float)
                v = np.array(cache["v"], dtype=float)
                polygon = np.array(cache["poly_2d"], dtype=float)
            else:
                u, v = plane_basis(normal)
                origin = np.mean(face_points, axis=0)
                polygon = np.column_stack([(face_points - origin) @ u, (face_points - origin) @ v])
                face["_polygon_distance_cache"] = {
                    "origin": origin,
                    "u": u,
                    "v": v,
                    "poly_2d": polygon,
                }
            projected = np.column_stack([(points - origin) @ u, (points - origin) @ v])
            outside = np.maximum(polygon_signed_distances_2d(projected, polygon), 0.0)
            plane_distance = np.abs((points - face_points[0]) @ normal)
            distances[:, column] = np.sqrt(plane_distance * plane_distance + outside * outside)
        return distances

    def fraction_summary(categories: np.ndarray, mask: np.ndarray) -> dict[str, object]:
        count = int(np.sum(mask))
        if count == 0:
            return {"count": 0, "target_purity": None, "target_boundary_fraction": None}
        selected = categories[mask]
        target = int(np.sum((selected == "target_boundary") | (selected == "target_face")))
        boundary = int(np.sum(selected == "target_boundary"))
        return {
            "count": count,
            "target_purity": as_json_float(float(target / count)),
            "target_boundary_fraction": as_json_float(float(boundary / count)),
        }

    def point_ancestry(face_id: int, extracted: dict[str, object]) -> dict[str, object]:
        points = np.asarray(extracted["points"], dtype=float)
        sides = np.asarray(extracted["sides"], dtype=object)
        z_keys = np.asarray(extracted["z_keys"], dtype=int)
        if points.shape[0] == 0:
            return {
                "point_count": 0,
                "target_face_point_fraction": None,
                "target_boundary_fraction": None,
                "neighbor_fraction": None,
                "unrelated_fraction": None,
                "unmatched_fraction": None,
                "number_of_oracle_faces_in_points": 0,
                "left_right_purity": {},
                "per_z_purity": [],
                "two_distinct_target_boundary_edges": False,
            }
        distances = face_distance_matrix(points)
        nearest_columns = np.argmin(distances, axis=1)
        nearest_face_ids = np.array([int(model_faces[index]["face_id"]) for index in nearest_columns], dtype=int)
        nearest_distances = distances[np.arange(points.shape[0]), nearest_columns]
        boundary_ids = face_boundary_edges.get(int(face_id), [])
        if boundary_ids:
            boundary_distances, _ = point_edge_distances_vectorized(
                points,
                initial_vertices,
                [edges[index] for index in boundary_ids],
            )
            nearest_boundary_local = np.argmin(boundary_distances, axis=1)
            nearest_boundary_distance = boundary_distances[np.arange(points.shape[0]), nearest_boundary_local]
            nearest_boundary_ids = np.array([boundary_ids[int(index)] for index in nearest_boundary_local], dtype=int)
        else:
            nearest_boundary_distance = np.full(points.shape[0], float("inf"), dtype=float)
            nearest_boundary_ids = np.full(points.shape[0], -1, dtype=int)
        categories = np.full(points.shape[0], "unmatched", dtype=object)
        categories[nearest_distances <= point_match_tolerance] = "unrelated"
        neighbor_mask = np.isin(nearest_face_ids, list(face_neighbors.get(int(face_id), set()))) & (
            nearest_distances <= point_match_tolerance
        )
        categories[neighbor_mask] = "neighbor"
        target_column = face_column_by_id.get(int(face_id))
        target_face_mask = (
            distances[:, target_column] <= point_match_tolerance
            if target_column is not None
            else np.zeros(points.shape[0], dtype=bool)
        )
        categories[target_face_mask] = "target_face"
        target_boundary_mask = nearest_boundary_distance <= point_match_tolerance
        categories[target_boundary_mask] = "target_boundary"
        category_counts = {
            name: int(np.sum(categories == name))
            for name in ("target_boundary", "target_face", "neighbor", "unrelated", "unmatched")
        }
        count = max(1, int(points.shape[0]))
        matched_faces = nearest_face_ids[nearest_distances <= point_match_tolerance]
        dominant_faces: list[dict[str, int]] = []
        if matched_faces.size:
            unique_faces, face_counts = np.unique(matched_faces, return_counts=True)
            dominant_faces = [
                {"face_id": int(fid), "count": int(face_count)}
                for fid, face_count in sorted(
                    zip(unique_faces, face_counts),
                    key=lambda item: (-int(item[1]), int(item[0])),
                )[:8]
            ]
        side_rows = {
            side: fraction_summary(categories, sides == side)
            for side in sorted(set(str(value) for value in sides))
        }
        dominant_boundary_by_side: dict[str, int | None] = {}
        for side in sorted(set(str(value) for value in sides)):
            mask = (sides == side) & target_boundary_mask
            if not np.any(mask):
                dominant_boundary_by_side[side] = None
                continue
            ids, counts = np.unique(nearest_boundary_ids[mask], return_counts=True)
            dominant_boundary_by_side[side] = int(ids[int(np.argmax(counts))])
        distinct_side_edges = {
            int(value)
            for value in dominant_boundary_by_side.values()
            if value is not None and int(value) >= 0
        }
        per_z_rows: list[dict[str, object]] = []
        for z_key in sorted(set(int(value) for value in z_keys)):
            mask = z_keys == int(z_key)
            row = fraction_summary(categories, mask)
            row["z_key"] = int(z_key)
            per_z_rows.append(row)
        return {
            "point_count": int(points.shape[0]),
            "point_match_tolerance": float(point_match_tolerance),
            "category_counts": category_counts,
            "target_face_point_fraction": as_json_float(
                float((category_counts["target_boundary"] + category_counts["target_face"]) / count)
            ),
            "target_boundary_fraction": as_json_float(float(category_counts["target_boundary"] / count)),
            "neighbor_fraction": as_json_float(float(category_counts["neighbor"] / count)),
            "unrelated_fraction": as_json_float(float(category_counts["unrelated"] / count)),
            "unmatched_fraction": as_json_float(float(category_counts["unmatched"] / count)),
            "number_of_oracle_faces_in_points": int(len(set(int(value) for value in matched_faces))),
            "dominant_oracle_faces": dominant_faces,
            "target_boundary_edge_ids": [int(value) for value in boundary_ids],
            "dominant_target_boundary_edge_by_side": dominant_boundary_by_side,
            "two_distinct_target_boundary_edges": bool(len(distinct_side_edges) >= 2),
            "left_right_purity": side_rows,
            "per_z_purity": per_z_rows,
        }

    def oriented_relation(
        plane_a: tuple[np.ndarray, np.ndarray] | None,
        plane_b: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[float, float]:
        if plane_a is None or plane_b is None:
            return float("nan"), float("nan")
        point_a, normal_a = plane_a
        point_b, normal_b = plane_b
        dot = float(normal_a @ normal_b)
        sign = 1.0 if dot >= 0.0 else -1.0
        angle = float(np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0))))
        offset = abs(float(normal_a @ point_a) - sign * float(normal_b @ point_b))
        return angle, offset

    def weighted_plane(points: np.ndarray, weights: np.ndarray, reference_normal: np.ndarray) -> PlaneFit | None:
        valid = np.all(np.isfinite(points), axis=1) & np.isfinite(weights) & (weights > 0.0)
        pts = points[valid]
        w = weights[valid]
        if pts.shape[0] < 3 or float(np.sum(w)) <= EPS:
            return None
        w = w / float(np.sum(w))
        centroid = np.sum(pts * w[:, None], axis=0)
        centered = pts - centroid
        covariance = (centered * w[:, None]).T @ centered
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            return None
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        normal /= max(float(np.linalg.norm(normal)), EPS)
        if float(normal @ reference_normal) < 0.0:
            normal *= -1.0
        signed = centered @ normal
        return PlaneFit(
            centroid=centroid,
            normal=normal,
            rms=float(np.sqrt(np.sum(w * signed * signed))),
            max_abs=float(np.max(np.abs(signed))),
        )

    def robust_weighted_plane(
        points: np.ndarray,
        base_weights: np.ndarray,
        reference_normal: np.ndarray,
    ) -> tuple[PlaneFit | None, np.ndarray, dict[str, object]]:
        if points.shape[0] < 3:
            return None, np.zeros(points.shape[0], dtype=bool), {"iterations": 0, "inlier_fraction": 0.0}
        weights = np.asarray(base_weights, dtype=float).copy()
        weights[~np.isfinite(weights) | (weights < 0.0)] = 0.0
        if float(np.sum(weights)) <= EPS:
            weights[:] = 1.0
        robust = np.ones(points.shape[0], dtype=float)
        plane: PlaneFit | None = None
        cutoff = float("nan")
        iterations = 0
        for iteration in range(6):
            plane = weighted_plane(points, weights * robust, reference_normal)
            if plane is None:
                break
            residual = np.abs((points - plane.centroid) @ plane.normal)
            median = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median)))
            cutoff = max(4.685 * max(1.4826 * mad, 1e-9), 0.002)
            scaled = residual / cutoff
            next_robust = np.where(scaled < 1.0, (1.0 - scaled * scaled) ** 2, 0.0)
            minimum = max(3, int(math.ceil(0.45 * points.shape[0])))
            if int(np.sum(next_robust > 0.05)) < minimum:
                keep = np.argsort(residual)[:minimum]
                next_robust = np.zeros(points.shape[0], dtype=float)
                next_robust[keep] = 1.0
            iterations = iteration + 1
            if np.max(np.abs(next_robust - robust)) <= 1e-5:
                robust = next_robust
                break
            robust = next_robust
        plane = weighted_plane(points, weights * robust, reference_normal)
        inliers = robust > 0.05
        return plane, inliers, {
            "iterations": int(iterations),
            "inlier_fraction": as_json_float(float(np.mean(inliers))) if inliers.size else 0.0,
            "residual_cutoff": as_json_float(cutoff),
        }

    def plane_candidate(
        base: dict[str, object],
        plane: PlaneFit,
        hull_points: np.ndarray,
        *,
        fit_point_count: int,
    ) -> dict[str, object] | None:
        if hull_points.shape[0] < 3:
            return None
        reference = np.array(base.get("plane_normal") or plane.normal, dtype=float)
        normal = plane.normal.copy()
        if reference.shape == (3,) and float(reference @ normal) < 0.0:
            normal *= -1.0
        signed = (hull_points - plane.centroid) @ normal
        projected = hull_points - signed[:, None] * normal[None, :]
        u, v = plane_basis(normal)
        coordinates = np.column_stack([(projected - plane.centroid) @ u, (projected - plane.centroid) @ v])
        hull_indices = convex_hull_indices(coordinates)
        if len(hull_indices) < 3:
            return None
        hull = projected[np.array(hull_indices, dtype=int)]
        output = dict(base)
        output.update(
            {
                "plane_centroid": as_json_point(plane.centroid),
                "plane_normal": as_json_point(normal),
                "plane_rms": as_json_float(float(plane.rms)),
                "plane_max_abs": as_json_float(float(plane.max_abs)),
                "fit_points": int(fit_point_count),
                "hull": [as_json_point(point) for point in hull],
                "hull_area": as_json_float(float(polygon_area(hull))),
                "hull_diameter": as_json_float(
                    float(np.max(np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2)))
                ),
                "sample_points": [
                    as_json_point(point)
                    for point in hull_points[:: max(1, hull_points.shape[0] // 80)]
                ],
            }
        )
        return output

    def direct_face_metrics(candidate: dict[str, object], face_id: int) -> dict[str, object]:
        plane = candidate_plane(candidate)
        face = face_by_id.get(int(face_id))
        if plane is None or face is None:
            return {
                "face_id": int(face_id),
                "normal_angle_deg": None,
                "plane_distance": None,
                "centroid_distance": None,
                "hull_surface_distance": None,
                "plane_good": False,
                "finite_good": False,
                "failure_reasons": ["invalid_candidate"],
            }
        point, normal = plane
        face_normal = np.array(face["normal"], dtype=float)
        dot = float(normal @ face_normal)
        sign = 1.0 if dot >= 0.0 else -1.0
        angle = float(np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0))))
        plane_distance = abs(float(normal @ point) - sign * float(face["offset"]))
        centroid_distance = point_to_face_polygon_distance(point, face)
        hull = np.array(candidate.get("hull") or [point], dtype=float).reshape((-1, 3))
        hull_distances = np.array([point_to_face_polygon_distance(value, face) for value in hull], dtype=float)
        hull_distance = float(np.median(hull_distances)) if hull_distances.size else float("inf")
        failures: list[str] = []
        if angle > CANONICAL_ORACLE_TOLERANCES["normal_angle_deg"]:
            failures.append("normal_angle")
        if plane_distance > CANONICAL_ORACLE_TOLERANCES["plane_distance"]:
            failures.append("plane_distance")
        if centroid_distance > CANONICAL_ORACLE_TOLERANCES["centroid_to_finite_polygon_distance"]:
            failures.append("centroid_to_finite_polygon_distance")
        if hull_distance > CANONICAL_ORACLE_TOLERANCES["median_hull_to_surface_distance"]:
            failures.append("median_hull_to_surface_distance")
        plane_good = not any(value in failures for value in ("normal_angle", "plane_distance"))
        return {
            "face_id": int(face_id),
            "normal_angle_deg": as_json_float(angle),
            "plane_distance": as_json_float(plane_distance),
            "centroid_distance": as_json_float(centroid_distance),
            "hull_surface_distance": as_json_float(hull_distance),
            "plane_good": bool(plane_good),
            "finite_good": bool(not failures),
            "failure_reasons": failures,
        }

    def primary_failure(metrics: dict[str, object]) -> str:
        failures = set(str(value) for value in metrics.get("failure_reasons", []))
        normal = "normal_angle"
        offset = "plane_distance"
        centroid = "centroid_to_finite_polygon_distance"
        hull = "median_hull_to_surface_distance"
        if failures == {normal}:
            return "normal_only"
        if failures == {offset}:
            return "plane_distance_only"
        if failures == {centroid}:
            return "finite_centroid_only"
        if failures == {hull}:
            return "finite_hull_only"
        if failures == {normal, offset}:
            return "normal_plus_offset"
        if failures and normal not in failures and offset not in failures:
            return "plane_good_patch_bad"
        return "multiple_failures"

    def span_condition(points: np.ndarray) -> tuple[float, list[float]]:
        if points.shape[0] < 3:
            return float("inf"), []
        singular = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
        values = [float(value) for value in singular]
        condition = float(values[0] / max(values[1], EPS)) if len(values) >= 2 else float("inf")
        return condition, values

    def base_weights_by_group(groups: np.ndarray) -> np.ndarray:
        weights = np.zeros(groups.shape[0], dtype=float)
        for value in set(groups.tolist()):
            mask = groups == value
            weights[mask] = 1.0 / max(1, int(np.sum(mask)))
        return weights

    def alternative_candidates(
        candidate: dict[str, object],
        extracted: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        points = np.asarray(extracted["points"], dtype=float)
        sides = np.asarray(extracted["sides"], dtype=object)
        z_keys = np.asarray(extracted["z_keys"], dtype=int)
        current_plane = candidate_plane(candidate)
        if points.shape[0] < 3 or current_plane is None:
            return {"current_least_squares": {"candidate": dict(candidate), "inlier_mask": np.ones(points.shape[0], dtype=bool)}}
        _, reference_normal = current_plane
        output: dict[str, dict[str, object]] = {
            "current_least_squares": {
                "candidate": dict(candidate),
                "inlier_mask": np.ones(points.shape[0], dtype=bool),
                "details": {"inlier_fraction": 1.0},
            }
        }

        def add_weighted(name: str, weights: np.ndarray) -> None:
            method_started = time.perf_counter()
            plane, inliers, details = robust_weighted_plane(points, weights, reference_normal)
            if plane is not None and int(np.sum(inliers)) >= 3:
                rebuilt = plane_candidate(
                    candidate,
                    plane,
                    points[inliers],
                    fit_point_count=int(points.shape[0]),
                )
                if rebuilt is not None:
                    output[name] = {"candidate": rebuilt, "inlier_mask": inliers, "details": details}
            mode_timings[name] = mode_timings.get(name, 0.0) + (time.perf_counter() - method_started)

        add_weighted("robust_trimmed_plane", np.ones(points.shape[0], dtype=float))
        add_weighted("level_balanced_robust", base_weights_by_group(z_keys))
        add_weighted("track_side_balanced", base_weights_by_group(sides))

        sub_z_started = time.perf_counter()
        unique_z = np.array(sorted(set(int(value) for value in z_keys)), dtype=int)
        if unique_z.size >= 6:
            index_chunks = [chunk for chunk in np.array_split(unique_z, 3) if chunk.size >= 2]
            subset_rows: list[tuple[PlaneFit, np.ndarray]] = []
            for chunk in index_chunks:
                mask = np.isin(z_keys, chunk)
                if int(np.sum(mask)) < 3:
                    continue
                fit = weighted_plane(points[mask], base_weights_by_group(z_keys[mask]), reference_normal)
                if fit is not None:
                    subset_rows.append((fit, mask))
            if subset_rows:
                groups: list[list[int]] = []
                for index, (fit, _) in enumerate(subset_rows):
                    compatible: list[int] = []
                    for other_index, (other_fit, _) in enumerate(subset_rows):
                        angle, offset = oriented_relation(
                            (fit.centroid, fit.normal),
                            (other_fit.centroid, other_fit.normal),
                        )
                        if angle <= 5.0 and offset <= 0.10:
                            compatible.append(other_index)
                    groups.append(compatible)
                best_index = min(
                    range(len(subset_rows)),
                    key=lambda index: (-len(groups[index]), index),
                )
                consensus_mask = np.zeros(points.shape[0], dtype=bool)
                for index in groups[best_index]:
                    consensus_mask |= subset_rows[index][1]
                if int(np.sum(consensus_mask)) >= 3:
                    plane, local_inliers, details = robust_weighted_plane(
                        points[consensus_mask],
                        base_weights_by_group(z_keys[consensus_mask]),
                        reference_normal,
                    )
                    if plane is not None:
                        global_inliers = np.zeros(points.shape[0], dtype=bool)
                        global_indices = np.flatnonzero(consensus_mask)
                        global_inliers[global_indices[local_inliers]] = True
                        rebuilt = plane_candidate(
                            candidate,
                            plane,
                            points[global_inliers],
                            fit_point_count=int(points.shape[0]),
                        )
                        if rebuilt is not None:
                            output["sub_z_consensus"] = {
                                "candidate": rebuilt,
                                "inlier_mask": global_inliers,
                                "details": {
                                    **details,
                                    "consensus_group_size": int(len(groups[best_index])),
                                    "subset_fit_count": int(len(subset_rows)),
                                },
                            }
        mode_timings["sub_z_consensus"] = mode_timings.get("sub_z_consensus", 0.0) + (
            time.perf_counter() - sub_z_started
        )

        line_started = time.perf_counter()
        pair = [int(value) for value in candidate.get("w2_edge_cluster_ids", [])]
        if len(pair) == 2 and pair[0] in cluster_by_id and pair[1] in cluster_by_id:
            direction_a = np.array(cluster_by_id[pair[0]].get("direction") or [], dtype=float)
            direction_b = np.array(cluster_by_id[pair[1]].get("direction") or [], dtype=float)
            if direction_a.shape == (3,) and direction_b.shape == (3,):
                normal = np.cross(direction_a, direction_b)
                norm = float(np.linalg.norm(normal))
                if norm > 1e-6:
                    normal /= norm
                    if float(normal @ reference_normal) < 0.0:
                        normal *= -1.0
                    centroid = np.mean(points, axis=0)
                    signed = (points - centroid) @ normal
                    plane = PlaneFit(
                        centroid=centroid,
                        normal=normal,
                        rms=float(np.sqrt(np.mean(signed * signed))),
                        max_abs=float(np.max(np.abs(signed))),
                    )
                    rebuilt = plane_candidate(
                        candidate,
                        plane,
                        points,
                        fit_point_count=int(points.shape[0]),
                    )
                    if rebuilt is not None:
                        output["line_derived_plane"] = {
                            "candidate": rebuilt,
                            "inlier_mask": np.ones(points.shape[0], dtype=bool),
                            "details": {"line_cross_norm": as_json_float(norm), "inlier_fraction": 1.0},
                        }
        mode_timings["line_derived_plane"] = mode_timings.get("line_derived_plane", 0.0) + (
            time.perf_counter() - line_started
        )

        hull_started = time.perf_counter()
        robust_row = output.get("robust_trimmed_plane")
        if robust_row is not None:
            inliers = np.asarray(robust_row["inlier_mask"], dtype=bool)
            current_point, current_normal = current_plane
            residual = (points[inliers] - current_point) @ current_normal
            projected = points[inliers] - residual[:, None] * current_normal[None, :]
            current_residual = (points - current_point) @ current_normal
            fixed_plane = PlaneFit(
                centroid=current_point,
                normal=current_normal,
                rms=float(np.sqrt(np.mean(current_residual * current_residual))),
                max_abs=float(np.max(np.abs(current_residual))),
            )
            rebuilt = plane_candidate(
                candidate,
                fixed_plane,
                projected,
                fit_point_count=int(points.shape[0]),
            )
            if rebuilt is not None:
                output["robust_hull_reconstruction"] = {
                    "candidate": rebuilt,
                    "inlier_mask": inliers,
                    "details": dict(robust_row.get("details") or {}),
                }
        mode_timings["robust_hull_reconstruction"] = mode_timings.get(
            "robust_hull_reconstruction", 0.0
        ) + (time.perf_counter() - hull_started)
        return output

    def fit_stability(
        method_candidate: dict[str, object],
        extracted: dict[str, object],
        inlier_mask: np.ndarray,
        details: dict[str, object],
    ) -> dict[str, object]:
        points = np.asarray(extracted["points"], dtype=float)
        sides = np.asarray(extracted["sides"], dtype=object)
        z_keys = np.asarray(extracted["z_keys"], dtype=int)
        plane = candidate_plane(method_candidate)
        if plane is None or points.shape[0] < 3:
            return {"quality_score": None, "stable_nonoracle": False}
        point, normal = plane
        residual = np.abs((points - point) @ normal)
        signed = (points - point) @ normal
        diameter = float(np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2))) if points.shape[0] >= 2 else 0.0
        scale = max(diameter, 0.1)
        subset_angles: list[float] = []
        subset_offsets: list[float] = []
        unique_z = np.array(sorted(set(int(value) for value in z_keys)), dtype=int)
        subset_planes: list[PlaneFit] = []
        for chunk in np.array_split(unique_z, min(3, max(1, int(unique_z.size)))):
            mask = np.isin(z_keys, chunk)
            if int(np.sum(mask)) < 3:
                continue
            subset = fit_plane(points[mask])
            if subset is None:
                continue
            subset_planes.append(subset)
            angle, offset = oriented_relation(plane, (subset.centroid, subset.normal))
            subset_angles.append(float(angle))
            subset_offsets.append(float(offset))
        loo_angles: list[float] = []
        loo_offsets: list[float] = []
        if unique_z.size >= 3:
            sampled_z = unique_z[np.linspace(0, unique_z.size - 1, min(12, unique_z.size), dtype=int)]
            for z_key in sampled_z:
                mask = z_keys != int(z_key)
                if int(np.sum(mask)) < 3:
                    continue
                leave_fit = fit_plane(points[mask])
                if leave_fit is None:
                    continue
                angle, offset = oriented_relation(plane, (leave_fit.centroid, leave_fit.normal))
                loo_angles.append(float(angle))
                loo_offsets.append(float(offset))
        side_signed_medians: list[float] = []
        side_abs_medians: list[float] = []
        for side in sorted(set(str(value) for value in sides)):
            mask = sides == side
            if np.any(mask):
                side_signed_medians.append(float(np.median(signed[mask])))
                side_abs_medians.append(float(np.median(residual[mask])))
        side_signed_gap = (
            float(max(side_signed_medians) - min(side_signed_medians))
            if len(side_signed_medians) >= 2
            else float("nan")
        )
        side_residual_gap = (
            float(max(side_abs_medians) - min(side_abs_medians))
            if len(side_abs_medians) >= 2
            else float("nan")
        )
        balanced_plane, _, _ = robust_weighted_plane(
            points,
            base_weights_by_group(sides),
            normal,
        )
        balanced_angle, balanced_offset = oriented_relation(
            plane,
            (balanced_plane.centroid, balanced_plane.normal) if balanced_plane is not None else None,
        )
        condition, singular_values = span_condition(points)
        normal_stability = max(subset_angles) if subset_angles else float("nan")
        offset_stability = max(subset_offsets) if subset_offsets else float("nan")
        loo_angle = max(loo_angles) if loo_angles else float("nan")
        loo_offset = max(loo_offsets) if loo_offsets else float("nan")
        inlier_fraction = float(np.mean(inlier_mask)) if inlier_mask.size else 0.0

        def finite_or(value: float, fallback: float) -> float:
            return float(value) if np.isfinite(value) else float(fallback)

        quality_score = (
            finite_or(float(np.percentile(residual, 95)) / scale, 10.0)
            + 0.20 * finite_or(normal_stability / 10.0, 1.0)
            + 0.20 * finite_or(offset_stability / scale, 1.0)
            + 0.15 * finite_or(loo_angle / 10.0, 1.0)
            + 0.15 * finite_or(loo_offset / scale, 1.0)
            + 0.10 * finite_or(side_signed_gap / scale, 1.0)
            + 0.10 * finite_or(balanced_angle / 10.0, 1.0)
            + 0.10 * finite_or(balanced_offset / scale, 1.0)
            + 0.20 * (1.0 - inlier_fraction)
        )
        stable = bool(
            inlier_fraction >= 0.55
            and finite_or(normal_stability, 0.0) <= 10.0
            and finite_or(offset_stability, 0.0) <= 0.15
            and finite_or(loo_angle, 0.0) <= 10.0
            and finite_or(loo_offset, 0.0) <= 0.15
        )
        return {
            "inlier_fraction": as_json_float(inlier_fraction),
            "robust_residual_median": as_json_float(float(np.median(residual))),
            "robust_residual_p95": as_json_float(float(np.percentile(residual, 95))),
            "normal_stability_across_z_deg": as_json_float(normal_stability),
            "offset_stability_across_z": as_json_float(offset_stability),
            "leave_one_level_out_normal_deg": as_json_float(loo_angle),
            "leave_one_level_out_offset": as_json_float(loo_offset),
            "left_right_signed_residual_gap": as_json_float(side_signed_gap),
            "left_right_abs_residual_gap": as_json_float(side_residual_gap),
            "track_balanced_normal_agreement_deg": as_json_float(balanced_angle),
            "track_balanced_offset_agreement": as_json_float(balanced_offset),
            "fit_span_condition": as_json_float(condition),
            "fit_singular_values": [as_json_float(value) for value in singular_values],
            "support_density": method_candidate.get("finite_support_density"),
            "hull_locality": method_candidate.get("finite_support_hull_distance_median"),
            "consensus_group_size": int(details.get("consensus_group_size") or len(subset_planes)),
            "quality_score": as_json_float(float(quality_score)),
            "stable_nonoracle": stable,
        }

    def evaluate_candidate(
        target_face_id: int,
        candidate: dict[str, object],
        *,
        include_ancestry: bool,
    ) -> dict[str, object]:
        extracted = extract_fit_points(candidate)
        points = np.asarray(extracted["points"], dtype=float)
        alternatives = alternative_candidates(candidate, extracted)
        method_rows: dict[str, dict[str, object]] = {}
        for method in method_order:
            method_row = alternatives.get(method)
            if method_row is None:
                continue
            method_candidate = method_row["candidate"]
            metrics = direct_face_metrics(method_candidate, int(target_face_id))
            features = fit_stability(
                method_candidate,
                extracted,
                np.asarray(method_row.get("inlier_mask"), dtype=bool),
                dict(method_row.get("details") or {}),
            )
            method_rows[method] = {
                "oracle_metrics": metrics,
                "primary_failure": None if bool(metrics.get("finite_good")) else primary_failure(metrics),
                "nonoracle_features": features,
                "candidate_geometry": {
                    "plane_centroid": method_candidate.get("plane_centroid"),
                    "plane_normal": method_candidate.get("plane_normal"),
                    "plane_rms": method_candidate.get("plane_rms"),
                    "plane_max_abs": method_candidate.get("plane_max_abs"),
                    "hull_area": method_candidate.get("hull_area"),
                    "hull_diameter": method_candidate.get("hull_diameter"),
                    "hull_vertex_count": int(len(method_candidate.get("hull") or [])),
                },
                "details": method_row.get("details") or {},
                "_candidate": method_candidate,
            }
        current_metrics = method_rows.get("current_least_squares", {}).get("oracle_metrics") or direct_face_metrics(
            candidate, int(target_face_id)
        )
        condition, singular_values = span_condition(points)
        output = {
            "target_face_id": int(target_face_id),
            "candidate_id": int(candidate_track_id(candidate)),
            "candidate_source": source_name(candidate),
            "candidate_origin": candidate.get("candidate_origin"),
            "window": candidate.get("window"),
            "scale_mode": candidate.get("scale_mode"),
            "track_id": candidate.get("track_id"),
            "source_track_id": candidate.get("source_track_id"),
            "w2_edge_cluster_ids": candidate.get("w2_edge_cluster_ids") or [],
            "levels": int(candidate.get("levels") or 0),
            "z_min": candidate.get("z_min"),
            "z_max": candidate.get("z_max"),
            "z_span": as_json_float(
                max(0.0, finite_float(candidate.get("z_max"), 0.0) - finite_float(candidate.get("z_min"), 0.0))
            ),
            "declared_fit_point_count": int(candidate.get("fit_points") or 0),
            "reconstructed_fit_point_count": int(points.shape[0]),
            "fit_point_origins": sorted(set(str(value) for value in extracted.get("origins", []))),
            "plane_rms": candidate.get("plane_rms"),
            "plane_max_abs": candidate.get("plane_max_abs"),
            "condition_median": candidate.get("condition_median"),
            "condition_p95": candidate.get("condition_p95"),
            "fit_span_condition": as_json_float(condition),
            "fit_singular_values": [as_json_float(value) for value in singular_values],
            "hull_area": candidate.get("hull_area"),
            "hull_diameter": candidate.get("hull_diameter"),
            "support_metrics": {
                key: candidate.get(key)
                for key in (
                    "finite_support_count",
                    "finite_support_density",
                    "finite_support_purity",
                    "finite_support_residual_median",
                    "finite_support_residual_p95",
                    "finite_support_balance",
                    "finite_support_track_coverage",
                    "observed_local_support",
                    "observed_support_near_count",
                    "observed_support_gap_local",
                    "w2_adjacency_levels",
                    "w2_adjacency_fraction",
                    "w2_edge_confidence",
                )
            },
            "current_oracle_metrics": current_metrics,
            "primary_failure": primary_failure(current_metrics),
            "secondary_failures": list(current_metrics.get("failure_reasons") or []),
            "fit_alternatives": method_rows,
        }
        if include_ancestry:
            output["fit_point_ancestry"] = point_ancestry(int(target_face_id), extracted)
        return output

    evaluation_started = time.perf_counter()
    failure_rows = [
        evaluate_candidate(face_id, candidate, include_ancestry=True)
        for face_id, candidate in best_failure_candidates
    ]
    timings["failure_candidate_evaluation_seconds"] = time.perf_counter() - evaluation_started

    def per_z_purity_range(ancestry: dict[str, object]) -> float:
        values = [
            finite_float(row.get("target_purity"), float("nan"))
            for row in ancestry.get("per_z_purity", [])
        ]
        values = [value for value in values if np.isfinite(value)]
        return float(max(values) - min(values)) if values else 0.0

    def root_cause(row: dict[str, object]) -> str:
        ancestry = dict(row.get("fit_point_ancestry") or {})
        purity = finite_float(ancestry.get("target_face_point_fraction"), 0.0)
        boundary = finite_float(ancestry.get("target_boundary_fraction"), 0.0)
        neighbor = finite_float(ancestry.get("neighbor_fraction"), 0.0)
        unrelated = finite_float(ancestry.get("unrelated_fraction"), 0.0)
        unmatched = finite_float(ancestry.get("unmatched_fraction"), 0.0)
        face_count = int(ancestry.get("number_of_oracle_faces_in_points") or 0)
        levels = int(row.get("levels") or 0)
        z_span = finite_float(row.get("z_span"), 0.0)
        condition = finite_float(row.get("fit_span_condition"), float("inf"))
        current = dict(row.get("current_oracle_metrics") or {})
        recovered_by_robust = any(
            bool((row.get("fit_alternatives", {}).get(method, {}).get("oracle_metrics") or {}).get("finite_good"))
            for method in ("robust_trimmed_plane", "level_balanced_robust", "track_side_balanced", "sub_z_consensus")
        )
        if bool(current.get("plane_good")) and not bool(current.get("finite_good")):
            return "correct_plane_wrong_hull"
        if row.get("candidate_source") == "W2 adjacency" and (
            boundary < 0.45 or not bool(ancestry.get("two_distinct_target_boundary_edges"))
        ):
            return "wrong_edge_pair_or_adjacency"
        if purity < 0.10 and unrelated + unmatched >= 0.75:
            return "false_raw_candidate"
        if unrelated >= 0.20 or (neighbor + unrelated >= 0.40 and face_count >= 3):
            return "mixed_multiple_face_patches"
        if row.get("candidate_source") == "baseline/valley" and per_z_purity_range(ancestry) >= 0.60:
            return "fragmented_track_combination"
        singular = row.get("fit_singular_values") or []
        second_span = finite_float(singular[1], 0.0) if len(singular) >= 2 else 0.0
        if z_span < 0.12 or levels < 4 or second_span < 0.04:
            return "insufficient_geometric_span"
        if purity >= 0.80 and boundary >= 0.80:
            return "ill_conditioned_but_correct"
        if purity >= 0.60 and condition >= 25.0:
            return "ill_conditioned_but_correct"
        if purity >= 0.60 and recovered_by_robust:
            return "correct_points_with_outliers"
        if face_count >= 3 and purity < 0.60:
            return "mixed_multiple_face_patches"
        return "false_raw_candidate"

    for row in failure_rows:
        row["root_cause"] = root_cause(row)

    w2_adjacency_failure_audit = diagnose_w2_adjacency_failures(
        failure_rows=failure_rows,
        initial_vertices=initial_vertices,
        initial_faces=initial_faces,
        model_faces=model_faces,
        z_levels=z_levels,
        w2_line_points=w2_line_points,
        w2_raw_candidates=w2_raw_candidates,
        w2_clusters=w2_clusters,
        inside_point=inside_point,
        orientation_points=orientation_points,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        min_adjacency_levels=int(min_adjacency_levels),
        min_adjacency_fraction=float(min_adjacency_fraction),
        min_z_span=float(min_z_span),
        max_plane_rms=float(max_plane_rms),
        final_candidates=final_candidates,
        final_reconstructed=final_reconstructed,
        core_face_ids=set(int(value) for value in core_face_ids),
        trusted_points=trusted_points,
        trusted_z_indices=trusted_z_indices,
        point_tol=float(point_tol),
        reconstruction_kwargs=reconstruction_kwargs,
    )
    w2_cluster_formation_audit = diagnose_w2_cluster_formation_failures(
        failure_rows=failure_rows,
        initial_vertices=initial_vertices,
        initial_faces=initial_faces,
        model_faces=model_faces,
        z_levels=z_levels,
        w2_line_points=w2_line_points,
        w2_segments=w2_segments,
        w2_clusters=w2_clusters,
        w2_raw_candidates=w2_raw_candidates,
        edge_ancestry_internal=edge_ancestry_internal,
        segment_min_levels=int(segment_min_levels),
        segment_min_z_span=float(segment_min_z_span),
        segment_max_z_gap=int(segment_max_z_gap),
        cluster_mode=str(cluster_mode),
        cluster_max_distance=float(cluster_max_distance),
        cluster_max_angle_deg=float(cluster_max_angle_deg),
        cluster_min_overlap=float(cluster_min_overlap),
        face_min_adjacency_levels=int(min_adjacency_levels),
        face_min_adjacency_fraction=float(min_adjacency_fraction),
        face_min_z_span=float(min_z_span),
        face_max_plane_rms=float(max_plane_rms),
        inside_point=inside_point,
        orientation_points=orientation_points,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        final_candidates=final_candidates,
        final_reconstructed=final_reconstructed,
        core_face_ids=set(int(value) for value in core_face_ids),
        trusted_points=trusted_points,
        trusted_z_indices=trusted_z_indices,
        point_tol=float(point_tol),
        reconstruction_kwargs=reconstruction_kwargs,
    )
    w2_cluster_formation_audit["w2_raw_signal_audit"] = diagnose_w2_raw_signal_failures(
        cluster_audit=w2_cluster_formation_audit,
        initial_vertices=initial_vertices,
        initial_faces=initial_faces,
        model_faces=model_faces,
        z_levels=z_levels,
        w2_line_points=w2_line_points,
        w2_fit_rms=w2_fit_rms,
        w2_condition=w2_condition,
        w3_line_points=w3_line_points,
        w3_fit_rms=w3_fit_rms,
        w4_line_points=w4_line_points,
        w4_fit_rms=w4_fit_rms,
        valley_raw_candidates=valley_raw_candidates,
        w2_segments=w2_segments,
        w2_clusters=w2_clusters,
        w2_raw_candidates=w2_raw_candidates,
        edge_ancestry_internal=edge_ancestry_internal,
        segment_min_levels=int(segment_min_levels),
        segment_min_z_span=float(segment_min_z_span),
        segment_max_z_gap=int(segment_max_z_gap),
        segment_max_line_rms=float(segment_max_line_rms),
        segment_max_line_residual=float(segment_max_line_residual),
        segment_max_point_jump=float(segment_max_point_jump),
        segment_max_condition=float(segment_max_condition),
        cluster_mode=str(cluster_mode),
        cluster_max_distance=float(cluster_max_distance),
        cluster_max_angle_deg=float(cluster_max_angle_deg),
        cluster_min_overlap=float(cluster_min_overlap),
        face_min_adjacency_levels=int(min_adjacency_levels),
        face_min_adjacency_fraction=float(min_adjacency_fraction),
        face_min_z_span=float(min_z_span),
        face_max_plane_rms=float(max_plane_rms),
        inside_point=inside_point,
        orientation_points=orientation_points,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        final_candidates=final_candidates,
        final_reconstructed=final_reconstructed,
        core_face_ids=set(int(value) for value in core_face_ids),
        trusted_points=trusted_points,
        trusted_z_indices=trusted_z_indices,
        point_tol=float(point_tol),
        reconstruction_kwargs=reconstruction_kwargs,
        contours=contours,
    )
    w2_adjacency_failure_audit["w2_cluster_formation_audit"] = w2_cluster_formation_audit

    control_selection_started = time.perf_counter()
    finite_control_by_source_face: dict[tuple[str, int], tuple[tuple[float, float, float, float, int], dict[str, object], dict[str, object]]] = {}
    for candidate in raw_candidates:
        match = current_match(candidate)
        if match is None or not bool(match.get("finite_good")) or match.get("face_id") is None:
            continue
        face_id = int(match["face_id"])
        extracted = extract_fit_points(candidate)
        if np.asarray(extracted["points"]).shape[0] < 3:
            continue
        key = (source_name(candidate), face_id)
        quality = oracle_geometry_key(match)
        previous = finite_control_by_source_face.get(key)
        if previous is None or quality < previous[0]:
            finite_control_by_source_face[key] = (quality, candidate, match)
    finite_control_pool = [
        (face_id, candidate, match)
        for (source, face_id), (_, candidate, match) in finite_control_by_source_face.items()
    ]

    def match_distance(candidate: dict[str, object], failure_row: dict[str, object]) -> float:
        source_penalty = 0.0 if source_name(candidate) == failure_row.get("candidate_source") else 3.0
        candidate_levels = max(1.0, float(candidate.get("levels") or 1))
        failure_levels = max(1.0, float(failure_row.get("levels") or 1))
        candidate_span = max(
            1e-4,
            finite_float(candidate.get("z_max"), 0.0) - finite_float(candidate.get("z_min"), 0.0),
        )
        failure_span = max(1e-4, finite_float(failure_row.get("z_span"), 0.0))
        candidate_points = max(1.0, float(candidate.get("fit_points") or 1))
        failure_points = max(1.0, float(failure_row.get("declared_fit_point_count") or 1))
        return float(
            source_penalty
            + abs(math.log(candidate_levels / failure_levels))
            + abs(math.log(candidate_span / failure_span))
            + 0.5 * abs(math.log(candidate_points / failure_points))
        )

    requested_control_count = 30
    failure_source_counts: dict[str, int] = {}
    for row in failure_rows:
        source = str(row["candidate_source"])
        failure_source_counts[source] = failure_source_counts.get(source, 0) + 1
    source_quotas: dict[str, int] = {}
    quota_fractions: list[tuple[float, str]] = []
    assigned_quota = 0
    for source, source_count in sorted(failure_source_counts.items()):
        exact = float(requested_control_count * source_count) / max(1.0, float(len(failure_rows)))
        quota = int(math.floor(exact))
        source_quotas[source] = quota
        assigned_quota += quota
        quota_fractions.append((exact - quota, source))
    for _, source in sorted(quota_fractions, key=lambda item: (-item[0], item[1])):
        if assigned_quota >= requested_control_count:
            break
        source_quotas[source] += 1
        assigned_quota += 1
    matched_controls: list[tuple[int, dict[str, object], dict[str, object]]] = []
    used_control_faces: set[int] = set()
    for source, quota in sorted(source_quotas.items()):
        source_failures = [row for row in failure_rows if row["candidate_source"] == source]
        source_options = sorted(
            [item for item in finite_control_pool if source_name(item[1]) == source],
            key=lambda item: (
                min((match_distance(item[1], row) for row in source_failures), default=float("inf")),
                int(candidate_track_id(item[1])),
            ),
        )
        for item in source_options:
            if len([row for row in matched_controls if source_name(row[1]) == source]) >= int(quota):
                break
            if int(item[0]) in used_control_faces:
                continue
            matched_controls.append(item)
            used_control_faces.add(int(item[0]))
    if len(matched_controls) < requested_control_count:
        remaining_options = sorted(
            finite_control_pool,
            key=lambda item: (
                min((match_distance(item[1], row) for row in failure_rows), default=float("inf")),
                int(candidate_track_id(item[1])),
            ),
        )
        for item in remaining_options:
            if len(matched_controls) >= requested_control_count:
                break
            if int(item[0]) in used_control_faces:
                continue
            matched_controls.append(item)
            used_control_faces.add(int(item[0]))
    control_rows = [
        evaluate_candidate(face_id, candidate, include_ancestry=False)
        for face_id, candidate, _ in matched_controls
    ]
    timings["matched_good_control_seconds"] = time.perf_counter() - control_selection_started

    active_indices = [
        int(index)
        for index in final_reconstructed.get("face_candidate_indices", [])
        if 0 <= int(index) < len(final_candidates)
    ]
    false_normal_candidates: list[tuple[int, dict[str, object]]] = []
    for index in active_indices:
        candidate = final_candidates[index]
        match = current_match(candidate)
        if match is None or match.get("face_id") is None:
            continue
        if "normal_angle" not in list(match.get("failure_reasons") or []):
            continue
        if np.asarray(extract_fit_points(candidate)["points"]).shape[0] < 3:
            continue
        false_normal_candidates.append((int(match["face_id"]), candidate))
        if len(false_normal_candidates) >= 20:
            break
    false_normal_rows = [
        evaluate_candidate(face_id, candidate, include_ancestry=False)
        for face_id, candidate in false_normal_candidates
    ]

    def strip_candidates(row: dict[str, object]) -> dict[str, object]:
        output = dict(row)
        alternatives: dict[str, object] = {}
        for method, method_row in row.get("fit_alternatives", {}).items():
            cleaned = dict(method_row)
            cleaned.pop("_candidate", None)
            alternatives[str(method)] = cleaned
        output["fit_alternatives"] = alternatives
        return output

    def counts(values: list[str]) -> dict[str, int]:
        output: dict[str, int] = {}
        for value in values:
            output[str(value)] = output.get(str(value), 0) + 1
        return {key: int(value) for key, value in sorted(output.items())}

    primary_counts = counts([str(row["primary_failure"]) for row in failure_rows])
    secondary_counts = counts(
        [str(value) for row in failure_rows for value in row.get("secondary_failures", [])]
    )
    root_counts = counts([str(row["root_cause"]) for row in failure_rows])

    method_summaries: dict[str, dict[str, object]] = {}
    for method in method_order:
        failure_method_rows = [
            row["fit_alternatives"][method]
            for row in failure_rows
            if method in row.get("fit_alternatives", {})
        ]
        recovered_ids = sorted(
            int(row["target_face_id"])
            for row in failure_rows
            if bool(
                (
                    row.get("fit_alternatives", {}).get(method, {}).get("oracle_metrics")
                    or {}
                ).get("finite_good")
            )
        )
        control_method_rows = [
            row["fit_alternatives"][method]
            for row in control_rows
            if method in row.get("fit_alternatives", {})
        ]
        damaged_controls = [
            int(row["candidate_id"])
            for row in control_rows
            if method in row.get("fit_alternatives", {})
            and not bool(row["fit_alternatives"][method]["oracle_metrics"].get("finite_good"))
        ]
        current_control = {
            int(row["candidate_id"]): row["fit_alternatives"]["current_least_squares"]["oracle_metrics"]
            for row in control_rows
            if "current_least_squares" in row.get("fit_alternatives", {})
        }
        normal_improved = 0
        offset_improved = 0
        hull_improved = 0
        unstable_hulls = 0
        for row in control_rows:
            alternative = row.get("fit_alternatives", {}).get(method)
            if alternative is None:
                continue
            before = current_control.get(int(row["candidate_id"]), {})
            after = alternative["oracle_metrics"]
            normal_improved += int(
                finite_float(after.get("normal_angle_deg"), float("inf"))
                < finite_float(before.get("normal_angle_deg"), float("inf"))
            )
            offset_improved += int(
                finite_float(after.get("plane_distance"), float("inf"))
                < finite_float(before.get("plane_distance"), float("inf"))
            )
            hull_improved += int(
                finite_float(after.get("hull_surface_distance"), float("inf"))
                < finite_float(before.get("hull_surface_distance"), float("inf"))
            )
            before_area = finite_float(
                row["fit_alternatives"]["current_least_squares"]["candidate_geometry"].get("hull_area"),
                0.0,
            )
            after_area = finite_float(alternative["candidate_geometry"].get("hull_area"), 0.0)
            ratio = after_area / max(before_area, EPS)
            unstable_hulls += int(ratio < 0.25 or ratio > 4.0)
        remaining = counts(
            [
                str(method_row.get("primary_failure"))
                for method_row in failure_method_rows
                if not bool(method_row["oracle_metrics"].get("finite_good"))
            ]
        )
        false_normal_recovered = sum(
            1
            for row in false_normal_rows
            if bool(
                (
                    row.get("fit_alternatives", {}).get(method, {}).get("oracle_metrics")
                    or {}
                ).get("finite_good")
            )
        )
        method_summaries[method] = {
            "failure_candidate_available": int(len(failure_method_rows)),
            "recovered_candidates": int(len(recovered_ids)),
            "recovered_unique_face_ids": int(len(set(recovered_ids))),
            "recovered_face_ids": recovered_ids,
            "normal_failures_remaining": int(
                sum(value for key, value in remaining.items() if key in {"normal_only", "normal_plus_offset", "multiple_failures"})
            ),
            "offset_failures_remaining": int(
                sum(value for key, value in remaining.items() if key in {"plane_distance_only", "normal_plus_offset", "multiple_failures"})
            ),
            "patch_failures_remaining": int(
                sum(value for key, value in remaining.items() if key in {"finite_centroid_only", "finite_hull_only", "plane_good_patch_bad", "multiple_failures"})
            ),
            "remaining_primary_failures": remaining,
            "good_controls_available": int(len(control_method_rows)),
            "good_controls_preserved": int(len(control_method_rows) - len(damaged_controls)),
            "damaged_good_controls": int(len(damaged_controls)),
            "damaged_good_control_candidate_ids": damaged_controls,
            "good_control_normal_improved": int(normal_improved),
            "good_control_offset_improved": int(offset_improved),
            "good_control_hull_improved": int(hull_improved),
            "unstable_good_control_hulls": int(unstable_hulls),
            "false_active_normal_sample_available": int(
                sum(1 for row in false_normal_rows if method in row.get("fit_alternatives", {}))
            ),
            "false_active_normal_sample_recovered": int(false_normal_recovered),
        }

    oracle_best_rows: list[dict[str, object]] = []
    for row in failure_rows:
        recovered = [
            method
            for method in method_order[1:]
            if bool(
                (
                    row.get("fit_alternatives", {}).get(method, {}).get("oracle_metrics")
                    or {}
                ).get("finite_good")
            )
        ]
        if recovered:
            oracle_best_rows.append(
                {
                    "target_face_id": int(row["target_face_id"]),
                    "candidate_id": int(row["candidate_id"]),
                    "recovering_methods": recovered,
                }
            )

    def group_summary(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "candidate_count": int(len(rows)),
            "failure_categories": counts([str(row["primary_failure"]) for row in rows]),
            "root_causes": counts([str(row["root_cause"]) for row in rows]),
            "target_purity": distribution_summary(
                [row.get("fit_point_ancestry", {}).get("target_face_point_fraction") for row in rows]
            ),
            "levels": distribution_summary([row.get("levels") for row in rows]),
            "z_span": distribution_summary([row.get("z_span") for row in rows]),
            "plane_rms": distribution_summary([row.get("plane_rms") for row in rows]),
            "condition": distribution_summary(
                [
                    row.get("condition_p95")
                    if row.get("condition_p95") is not None
                    else row.get("fit_span_condition")
                    for row in rows
                ]
            ),
            "theoretical_recoverable_count": int(
                sum(1 for row in rows if any(int(item["target_face_id"]) == int(row["target_face_id"]) for item in oracle_best_rows))
            ),
        }

    source_names = ["baseline/valley", "W2 adjacency", "append/broad candidates", "other"]
    source_summary = {
        source: group_summary([row for row in failure_rows if row.get("candidate_source") == source])
        for source in source_names
    }
    zone_names = ["belt_z_minus_5", "middle_z", "near_horizontal", "other"]
    zone_summary = {
        zone: group_summary(
            [
                row
                for row in failure_rows
                if str(face_rows.get(int(row["target_face_id"]), {}).get("zone") or "other") == zone
            ]
        )
        for zone in zone_names
    }

    feature_directions = {
        "inlier_fraction": "higher",
        "robust_residual_median": "lower",
        "robust_residual_p95": "lower",
        "normal_stability_across_z_deg": "lower",
        "offset_stability_across_z": "lower",
        "leave_one_level_out_normal_deg": "lower",
        "leave_one_level_out_offset": "lower",
        "left_right_signed_residual_gap": "lower",
        "track_balanced_normal_agreement_deg": "lower",
        "track_balanced_offset_agreement": "lower",
        "fit_span_condition": "lower",
        "support_density": "higher",
        "hull_locality": "lower",
        "consensus_group_size": "higher",
        "quality_score": "lower",
    }

    def auc(values: list[tuple[float, bool]], direction: str) -> float | None:
        positive = [value for value, label in values if label and np.isfinite(value)]
        negative = [value for value, label in values if not label and np.isfinite(value)]
        if not positive or not negative:
            return None
        wins = 0.0
        for left in positive:
            for right in negative:
                if left == right:
                    wins += 0.5
                elif (direction == "lower" and left < right) or (direction == "higher" and left > right):
                    wins += 1.0
        return float(wins / (len(positive) * len(negative)))

    feature_separation: dict[str, object] = {}
    for method in method_order[1:]:
        method_rows = [
            row["fit_alternatives"][method]
            for row in failure_rows
            if method in row.get("fit_alternatives", {})
        ]
        features: dict[str, object] = {}
        for feature, direction in feature_directions.items():
            labeled = [
                (
                    finite_float(row["nonoracle_features"].get(feature), float("nan")),
                    bool(row["oracle_metrics"].get("finite_good")),
                )
                for row in method_rows
            ]
            feature_auc = auc(labeled, direction)
            features[feature] = {
                "direction": direction,
                "auc": as_json_float(float(feature_auc)) if feature_auc is not None else None,
                "recovered": distribution_summary(
                    [value for value, label in labeled if label]
                ),
                "unrecovered": distribution_summary(
                    [value for value, label in labeled if not label]
                ),
            }
        feature_separation[method] = features

    def select_nonoracle_method(row: dict[str, object]) -> str:
        alternatives = row.get("fit_alternatives", {})
        current = alternatives.get("current_least_squares")
        if current is None:
            return "current_least_squares"
        current_score = finite_float(current["nonoracle_features"].get("quality_score"), float("inf"))
        candidates: list[tuple[float, int, str]] = []
        for index, method in enumerate(method_order[1:], start=1):
            method_row = alternatives.get(method)
            if method_row is None or not bool(method_row["nonoracle_features"].get("stable_nonoracle")):
                continue
            score = finite_float(method_row["nonoracle_features"].get("quality_score"), float("inf"))
            residual = finite_float(method_row["nonoracle_features"].get("robust_residual_p95"), float("inf"))
            current_residual = finite_float(current["nonoracle_features"].get("robust_residual_p95"), float("inf"))
            improves_score = score <= current_score * 0.95
            improves_residual = residual <= current_residual * 0.80
            if improves_score or improves_residual:
                candidates.append((score, index, method))
        return min(candidates)[2] if candidates else "current_least_squares"

    selector_failure_rows: list[dict[str, object]] = []
    for row in failure_rows:
        method = select_nonoracle_method(row)
        method_row = row["fit_alternatives"][method]
        selector_failure_rows.append(
            {
                "target_face_id": int(row["target_face_id"]),
                "candidate_id": int(row["candidate_id"]),
                "selected_method": method,
                "target_recovered": bool(method_row["oracle_metrics"].get("finite_good")),
                "quality_score": method_row["nonoracle_features"].get("quality_score"),
            }
        )
    selector_control_rows: list[dict[str, object]] = []
    for row in control_rows:
        method = select_nonoracle_method(row)
        method_row = row["fit_alternatives"][method]
        selector_control_rows.append(
            {
                "target_face_id": int(row["target_face_id"]),
                "candidate_id": int(row["candidate_id"]),
                "selected_method": method,
                "target_preserved": bool(method_row["oracle_metrics"].get("finite_good")),
            }
        )
    selector_recovered = sum(1 for row in selector_failure_rows if bool(row["target_recovered"]))
    selector_damaged = sum(1 for row in selector_control_rows if not bool(row["target_preserved"]))

    best_method = max(
        method_order[1:],
        key=lambda method: (
            int(method_summaries[method]["recovered_unique_face_ids"]),
            -int(method_summaries[method]["damaged_good_controls"]),
            -method_order.index(method),
        ),
    )
    best_recovered = int(method_summaries[best_method]["recovered_unique_face_ids"])
    best_damaged = int(method_summaries[best_method]["damaged_good_controls"])
    root_group_counts = {
        "B_candidate_splitting_grouping": int(
            root_counts.get("mixed_multiple_face_patches", 0)
            + root_counts.get("fragmented_track_combination", 0)
        ),
        "C_w2_adjacency": int(root_counts.get("wrong_edge_pair_or_adjacency", 0)),
        "D_finite_hull": int(root_counts.get("correct_plane_wrong_hull", 0)),
        "E_conditioning_span": int(
            root_counts.get("insufficient_geometric_span", 0)
            + root_counts.get("ill_conditioned_but_correct", 0)
        ),
    }
    if best_recovered >= 10 and best_damaged <= 2 and selector_recovered >= max(5, int(math.ceil(0.5 * best_recovered))) and selector_damaged <= 2:
        branch = "A"
        branch_key = "A_robust_fitter"
        branch_reason = "A common observed-point robust/weighted fit has material oracle ceiling and a low-damage non-oracle selection policy."
    else:
        strongest_root = max(root_group_counts, key=lambda key: (root_group_counts[key], key))
        if root_group_counts[strongest_root] > max(3, int(0.25 * len(failure_rows))):
            branch = strongest_root[0]
            branch_key = strongest_root
            branch_reason = "The largest fixed-threshold root-cause cohort determines the next candidate-formation investigation."
        else:
            branch = "F"
            branch_key = "F_separation_absent"
            branch_reason = "Oracle alternatives recover candidates, but the fixed non-oracle stability policy does not separate them safely from controls."

    baseline_final_ids: set[int] = set()
    for index in active_indices:
        match = current_match(final_candidates[index])
        if match is not None and bool(match.get("finite_good")) and match.get("face_id") is not None:
            baseline_final_ids.add(int(match["face_id"]))
    full_trial_started = time.perf_counter()
    full_trial_rows: list[dict[str, object]] = []
    selected_recoveries = [
        row
        for row in selector_failure_rows
        if bool(row["target_recovered"]) and row["selected_method"] != "current_least_squares"
    ][:5]
    failure_lookup = {int(row["candidate_id"]): row for row in failure_rows}
    for selected in selected_recoveries:
        candidate_id = int(selected["candidate_id"])
        source_row = failure_lookup[candidate_id]
        method = str(selected["selected_method"])
        replacement = source_row["fit_alternatives"][method]["_candidate"]
        trial_candidates = [
            candidate
            for candidate in final_candidates
            if int(candidate_track_id(candidate)) != candidate_id
        ] + [replacement]
        trial_reconstructed = reconstruct_polyhedron_from_halfspaces_edge_clip(
            trial_candidates,
            **reconstruction_kwargs,
        )
        trial_active_indices = [
            int(index)
            for index in trial_reconstructed.get("face_candidate_indices", [])
            if 0 <= int(index) < len(trial_candidates)
        ]
        trial_ids: set[int] = set()
        for index in trial_active_indices:
            match = candidate_oracle_face(trial_candidates[index], model_faces)
            if match is not None and bool(match.get("finite_good")) and match.get("face_id") is not None:
                trial_ids.add(int(match["face_id"]))
        replacement_index = len(trial_candidates) - 1
        full_trial_rows.append(
            {
                "target_face_id": int(selected["target_face_id"]),
                "candidate_id": candidate_id,
                "method": method,
                "replacement_active": bool(replacement_index in trial_active_indices),
                "target_final_active": bool(int(selected["target_face_id"]) in trial_ids),
                "unique_finite_face_ids": int(len(trial_ids)),
                "canonical_ids_gained": sorted(int(value) for value in trial_ids - baseline_final_ids),
                "canonical_ids_lost": sorted(int(value) for value in baseline_final_ids - trial_ids),
                "vertices": int(len(trial_reconstructed.get("vertices", []))),
                "edges": int(len(trial_reconstructed.get("edges", []))),
                "faces": int(len(trial_reconstructed.get("faces", []))),
                "volume": trial_reconstructed.get("reliable_volume"),
                "topology": trial_reconstructed.get("topology"),
            }
        )
    timings["bounded_full_edge_clip_seconds"] = time.perf_counter() - full_trial_started

    timings["total_seconds"] = time.perf_counter() - started
    return {
        "diagnostic_only": True,
        "uses_initial_model": True,
        "production_selector_changed": False,
        "canonical_evaluator": {
            "schema_version": int(CANONICAL_ORACLE_SCHEMA_VERSION),
            "tolerances": CANONICAL_ORACLE_TOLERANCES,
        },
        "cohort": {
            "expected_raw_candidate_not_finite_good": 51,
            "face_ids": cohort_face_ids,
            "selected_best_candidate_count": int(len(failure_rows)),
            "missing_best_candidate_rows": missing_best_candidate_rows,
            "selection": "oracle-best current raw candidate per missing face; diagnostics only",
        },
        "failure_anatomy": {
            "primary_failure_counts": primary_counts,
            "secondary_failure_counts": secondary_counts,
            "root_cause_counts": root_counts,
            "rows": [strip_candidates(row) for row in failure_rows],
        },
        "w2_adjacency_failure_audit": w2_adjacency_failure_audit,
        "point_ancestry_method": {
            "distance_tolerance": float(point_match_tolerance),
            "category_precedence": [
                "target_boundary",
                "target_face",
                "neighbor",
                "unrelated",
                "unmatched",
            ],
            "oracle_only": True,
        },
        "source_summary": source_summary,
        "zone_summary": zone_summary,
        "fit_modes": method_summaries,
        "oracle_best_across_alternatives": {
            "upper_bound_only": True,
            "recovered_unique_face_ids": int(len(oracle_best_rows)),
            "rows": oracle_best_rows,
            "not_a_production_or_joint_mesh_result": True,
        },
        "good_control_cohort": {
            "requested": int(requested_control_count),
            "selected": int(len(control_rows)),
            "selection": "finite-good raw candidates matched by source, levels, Z-span, and fit-point count",
            "failure_source_counts": failure_source_counts,
            "resolved_source_quotas": source_quotas,
            "rows": [strip_candidates(row) for row in control_rows],
        },
        "false_active_normal_control": {
            "bounded_sample_size": int(len(false_normal_rows)),
            "rows": [strip_candidates(row) for row in false_normal_rows],
        },
        "nonoracle_separation": {
            "features": feature_separation,
            "offline_selector_policy": {
                "uses_oracle": False,
                "requires_stable_features": True,
                "switch_rule": "alternative quality improves >=5% or residual p95 improves >=20%",
                "failure_targets_recovered": int(selector_recovered),
                "good_controls_damaged": int(selector_damaged),
                "failure_rows": selector_failure_rows,
                "control_rows": selector_control_rows,
            },
        },
        "bounded_full_edge_clip_sample": {
            "trial_count": int(len(full_trial_rows)),
            "rows": full_trial_rows,
        },
        "branch_decision": {
            "selected": branch,
            "selected_key": branch_key,
            "reason": branch_reason,
            "best_single_fit_mode": best_method,
            "best_single_fit_recovered": best_recovered,
            "best_single_fit_damaged_controls": best_damaged,
            "root_group_counts": root_group_counts,
            "production_fix_implemented": False,
        },
        "thresholds": {
            "point_match_tolerance": float(point_match_tolerance),
            "robust_tukey_constant": 4.685,
            "robust_min_residual_cutoff": 0.002,
            "robust_min_inlier_fraction": 0.45,
            "nonoracle_stable_min_inlier_fraction": 0.55,
            "nonoracle_max_normal_stability_deg": 10.0,
            "nonoracle_max_offset_stability": 0.15,
        },
        "timings_seconds": {
            **{key: as_json_float(float(value)) for key, value in timings.items()},
            "fit_modes": {key: as_json_float(float(value)) for key, value in mode_timings.items()},
        },
    }


def combined_outside_mask(
    candidates: list[dict[str, object]],
    trusted_points: np.ndarray,
    *,
    point_tol: float,
) -> np.ndarray:
    points = np.asarray(trusted_points, dtype=float).reshape((-1, 3)) if np.asarray(trusted_points).size else np.zeros((0, 3), dtype=float)
    outside = np.zeros(points.shape[0], dtype=bool)
    for candidate in candidates:
        outside |= candidate_outside_mask(candidate, points, point_tol=float(point_tol))
    return outside


def synthetic_halfspace_removal_monotonicity_regression() -> dict[str, object]:
    candidates: list[dict[str, object]] = [
        {"track_id": 1, "plane_centroid": [1.0, 0.0, 0.0], "plane_normal": [1.0, 0.0, 0.0]},
        {"track_id": 2, "plane_centroid": [-1.0, 0.0, 0.0], "plane_normal": [-1.0, 0.0, 0.0]},
        {"track_id": 3, "plane_centroid": [0.0, 1.0, 0.0], "plane_normal": [0.0, 1.0, 0.0]},
        {"track_id": 4, "plane_centroid": [0.0, -1.0, 0.0], "plane_normal": [0.0, -1.0, 0.0]},
        {"track_id": 5, "plane_centroid": [0.0, 0.0, 1.0], "plane_normal": [0.0, 0.0, 1.0]},
        {"track_id": 6, "plane_centroid": [0.0, 0.0, -1.0], "plane_normal": [0.0, 0.0, -1.0]},
    ]
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [-1.2, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, -1.2, 0.0],
            [0.0, 0.0, 1.2],
            [0.0, 0.0, -1.2],
            [1.2, 1.2, 0.0],
        ],
        dtype=float,
    )
    z_indices = np.array([0, 0, 0, 1, 1, 2, 2, 1], dtype=int)
    z_slices = prepare_z_level_slices(z_indices)
    before = combined_outside_mask(candidates, points, point_tol=0.0)
    rows: list[dict[str, object]] = []
    for remove_index, candidate in enumerate(candidates):
        after_candidates = [item for index, item in enumerate(candidates) if index != remove_index]
        after = combined_outside_mask(after_candidates, points, point_tol=0.0)
        rows.append(
            {
                "removed_candidate_id": int(candidate_track_id(candidate)),
                "before": cumulative_metrics_from_mask(before, z_slices),
                "after": cumulative_metrics_from_mask(after, z_slices),
                "newly_outside_count": int(np.sum(after & ~before)),
                "newly_inside_count": int(np.sum(before & ~after)),
                "passed": bool(not np.any(after & ~before)),
            }
        )
    return {
        "name": "synthetic_axis_aligned_cube_remove_one_halfspace",
        "invariant": "outside_mask_after_removal is subset of outside_mask_before_removal",
        "point_count": int(points.shape[0]),
        "rows": rows,
        "passed": bool(all(bool(row["passed"]) for row in rows)),
    }


def diagnose_active_plane_pruning_corrected(
    *,
    model_name: str,
    vertices: np.ndarray,
    model_faces: list[dict[str, object]],
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    contours: list[object],
    baseline_payload: dict[str, object] | None = None,
    previous_pruning_payload: dict[str, object] | None = None,
    baseline_json_path: str | None = None,
    previous_pruning_json_path: str | None = None,
    max_full_trials: int = 25,
) -> dict[str, object]:
    started = time.perf_counter()
    z_slices = prepare_z_level_slices(trusted_z_indices)
    active_indices = [
        int(index)
        for index in final_reconstructed.get("face_candidate_indices", [])
        if 0 <= int(index) < len(final_candidates)
    ]
    active_candidates = [final_candidates[index] for index in active_indices]
    candidate_by_id = {int(candidate_track_id(candidate)): candidate for candidate in final_candidates}
    candidate_index_by_id = {int(candidate_track_id(candidate)): int(index) for index, candidate in enumerate(final_candidates)}
    active_candidate_ids = {int(candidate_track_id(candidate)) for candidate in active_candidates}
    match_cache: dict[int, dict[str, object] | None] = {}

    def canonical_match(candidate: dict[str, object]) -> dict[str, object] | None:
        candidate_id = int(candidate_track_id(candidate))
        if candidate_id not in match_cache:
            match_cache[candidate_id] = canonical_candidate_oracle_face(candidate, model_faces)
        return match_cache[candidate_id]

    def canonical_ids(candidates: list[dict[str, object]], rec: dict[str, object]) -> set[int]:
        ids: set[int] = set()
        for raw_index in rec.get("face_candidate_indices", []) or []:
            index = int(raw_index)
            if index < 0 or index >= len(candidates):
                continue
            match = canonical_match(candidates[index])
            if match is not None and bool(match.get("finite_good")) and match.get("face_id") is not None:
                ids.add(int(match["face_id"]))
        return ids

    def active_patch_groups(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        groups: list[list[dict[str, object]]] = []
        for candidate in sorted(candidates, key=nonoracle_candidate_quality_key):
            for group in groups:
                if all(candidates_plane_patch_compatible(candidate, member) for member in group):
                    group.append(candidate)
                    break
            else:
                groups.append([candidate])
        rows: list[dict[str, object]] = []
        for index, group in enumerate(groups):
            ids = sorted(int(candidate_track_id(member)) for member in group)
            rows.append(
                {
                    "group_id": f"protected_active_patch_{index:04d}_{ids[0] if ids else -1}",
                    "member_candidate_ids": ids,
                    "representative_candidate_id": int(ids[0]) if ids else -1,
                    "_members": group,
                }
            )
        return rows

    def public_group(group: dict[str, object]) -> dict[str, object]:
        return {
            "group_id": str(group.get("group_id")),
            "representative_candidate_id": int(group.get("representative_candidate_id") or -1),
            "member_candidate_ids": [int(v) for v in group.get("member_candidate_ids", [])],
        }

    protected_groups = active_patch_groups(active_candidates)
    protected_group_by_candidate: dict[int, int] = {}
    for group_index, group in enumerate(protected_groups):
        for candidate_id in group.get("member_candidate_ids", []) or []:
            protected_group_by_candidate[int(candidate_id)] = int(group_index)

    def protected_patch_status(trial_active: list[dict[str, object]], removed_candidate_id: int) -> dict[str, object]:
        target_group_index = protected_group_by_candidate.get(int(removed_candidate_id))
        unrelated_lost: list[dict[str, object]] = []
        target_status: dict[str, object] = {
            "target_group_index": target_group_index,
            "target_group_lost": None,
            "target_group_semantics": "removed candidate has no protected active patch group",
        }
        for group_index, group in enumerate(protected_groups):
            members = list(group.get("_members") or [])
            represented = any(
                candidates_plane_patch_compatible(member, trial_candidate)
                for member in members
                for trial_candidate in trial_active
            )
            if group_index == target_group_index:
                target_status = {
                    "target_group_index": int(group_index),
                    "target_group": public_group(group),
                    "target_group_lost": bool(not represented),
                    "target_group_semantics": (
                        "target removal group is recorded separately; it is not an unrelated protected-patch hard failure"
                    ),
                }
            elif not represented:
                unrelated_lost.append(public_group(group))
        return {
            "protected_group_count": int(len(protected_groups)),
            "unrelated_lost_group_count": int(len(unrelated_lost)),
            "unrelated_lost_groups": unrelated_lost[:25],
            "unrelated_groups_preserved": bool(not unrelated_lost),
            "target_group": target_status,
        }

    def bbox_extent_delta_fraction(a: dict[str, object] | None, b: dict[str, object] | None) -> float | None:
        if not isinstance(a, dict) or not isinstance(b, dict):
            return None
        try:
            amin = np.asarray(a.get("min"), dtype=float)
            amax = np.asarray(a.get("max"), dtype=float)
            bmin = np.asarray(b.get("min"), dtype=float)
            bmax = np.asarray(b.get("max"), dtype=float)
        except (TypeError, ValueError):
            return None
        if amin.shape != (3,) or amax.shape != (3,) or bmin.shape != (3,) or bmax.shape != (3,):
            return None
        base = np.maximum(np.abs(amax - amin), EPS)
        return float(np.max(np.abs((bmax - bmin) - (amax - amin)) / base))

    def trusted_cloud_signature(points: np.ndarray, z_indices: np.ndarray) -> dict[str, object]:
        arr = np.asarray(points, dtype=float).reshape((-1, 3)) if np.asarray(points).size else np.zeros((0, 3), dtype=float)
        z_arr = np.asarray(z_indices, dtype=int).reshape((-1,))
        rounded = np.round(arr, 8)
        digest = hashlib.sha256(rounded.tobytes() + z_arr.tobytes()).hexdigest()
        return {
            "point_count": int(arr.shape[0]),
            "z_index_count": int(z_arr.size),
            "unique_z_levels": int(len(set(int(v) for v in z_arr))) if z_arr.size else 0,
            "sha256_round8_xyz_plus_z_index": digest,
        }

    baseline_mask = combined_outside_mask(final_candidates, trusted_points, point_tol=float(point_tol))
    baseline_outside = cumulative_metrics_from_mask(baseline_mask, z_slices)
    baseline_canonical_ids = canonical_ids(final_candidates, final_reconstructed)
    baseline_reprojection = evaluate_polyhedron_reprojection(
        name=f"active_plane_pruning_corrected_control_{model_name}",
        vertices_3d=np.asarray(final_reconstructed.get("vertices") or [], dtype=float),
        contours=contours,
        direction_count=24,
        contour_sample=48,
    )
    baseline_volume = finite_float(final_reconstructed.get("reliable_volume"), float("nan"))
    baseline_bbox = final_reconstructed.get("bbox") if isinstance(final_reconstructed.get("bbox"), dict) else None

    proposal_rows: list[dict[str, object]] = []
    if isinstance(previous_pruning_payload, dict):
        pool = previous_pruning_payload.get("proposal_pool")
        if isinstance(pool, dict) and isinstance(pool.get("rows"), list):
            proposal_rows = [row for row in pool.get("rows", []) if isinstance(row, dict)]
    proposal_candidate_ids = [
        int(row.get("candidate_id"))
        for row in proposal_rows
        if row.get("candidate_id") is not None and int(row.get("candidate_id")) in candidate_by_id
    ]
    if not proposal_candidate_ids:
        proposal_candidate_ids = [
            int(candidate_track_id(candidate))
            for candidate in sorted(active_candidates, key=nonoracle_candidate_quality_key)
        ]
    proposal_candidate_ids = list(dict.fromkeys(proposal_candidate_ids))

    pointwise_rows: list[dict[str, object]] = []
    for candidate_id in proposal_candidate_ids[:3]:
        after_candidates = [
            candidate for candidate in final_candidates
            if int(candidate_track_id(candidate)) != int(candidate_id)
        ]
        after_mask = combined_outside_mask(after_candidates, trusted_points, point_tol=float(point_tol))
        newly_outside = np.flatnonzero(after_mask & ~baseline_mask)
        newly_inside = np.flatnonzero(baseline_mask & ~after_mask)
        per_z_deltas: list[dict[str, object]] = []
        for z_order, idx in enumerate(z_slices):
            before_count = int(np.sum(baseline_mask[idx]))
            after_count = int(np.sum(after_mask[idx]))
            if before_count != after_count:
                per_z_deltas.append(
                    {
                        "z_slice_order": int(z_order),
                        "trusted_points": int(idx.size),
                        "outside_before": before_count,
                        "outside_after": after_count,
                        "delta_after_minus_before": int(after_count - before_count),
                    }
                )
        pointwise_rows.append(
            {
                "removed_candidate_id": int(candidate_id),
                "baseline": cumulative_metrics_from_mask(baseline_mask, z_slices),
                "after_removal": cumulative_metrics_from_mask(after_mask, z_slices),
                "newly_outside_count": int(newly_outside.size),
                "newly_inside_count": int(newly_inside.size),
                "newly_outside_indices_sample": [int(v) for v in newly_outside[:20]],
                "newly_inside_indices_sample": [int(v) for v in newly_inside[:20]],
                "per_z_deltas_sample": per_z_deltas[:25],
                "passed": bool(newly_outside.size == 0),
            }
        )

    synthetic_regression = synthetic_halfspace_removal_monotonicity_regression()
    monotonicity_passed = bool(
        bool(synthetic_regression.get("passed"))
        and all(bool(row.get("passed")) for row in pointwise_rows)
    )

    outside_metric_map = {
        "frozen_control.containment.trusted_points_outside_halfspaces": {
            "definition": "OR over candidate_outside_mask(candidate, production stable_two_scales trusted cloud)",
            "monotonic_under_halfspace_removal": True,
            "json_path": "frozen_control.containment.trusted_points_outside_halfspaces",
        },
        "frozen_control.containment.max_per_z_trusted_points_outside_halfspaces": {
            "definition": "maximum per-Z fraction of the same OR outside mask",
            "monotonic_under_halfspace_removal": True,
            "json_path": "frozen_control.containment.max_per_z_trusted_points_outside_halfspaces",
        },
        "frozen_control.containment.lost_z_trusted_points_outside_halfspaces": {
            "definition": "count of Z slices where every trusted point is outside at least one halfspace",
            "monotonic_under_halfspace_removal": True,
            "json_path": "frozen_control.containment.lost_z_trusted_points_outside_halfspaces",
        },
        "remove_one_full_trials.rows[].containment_delta": {
            "definition": "same mask metrics recomputed after one active candidate is removed",
            "monotonic_under_halfspace_removal": True,
            "json_path": "remove_one_full_trials.rows[].containment_delta",
        },
        "remove_one_full_trials.rows[].expansion_delta": {
            "definition": "volume, bbox, and reprojection shape expansion diagnostics after removal",
            "monotonic_under_halfspace_removal": False,
            "json_path": "remove_one_full_trials.rows[].expansion_delta",
        },
        "previous_active_plane_pruning.display_minima_outside": {
            "definition": "legacy audit artifact field from viewer.minima_cloud/display cloud, not the production stable_two_scales trusted cloud",
            "monotonic_under_halfspace_removal": "only within that different point set; not comparable to this audit path",
            "json_path": "previous artifact display_minima_outside/trusted_final_outside fields",
        },
    }

    if not monotonicity_passed:
        return {
            "diagnostic_only": True,
            "production_changed": False,
            "scope": "active-plane-pruning-corrected",
            "model": str(model_name),
            "canonical_evaluator": canonical_oracle_evaluator_metadata(),
            "outside_metric_map": outside_metric_map,
            "synthetic_halfspace_regression": synthetic_regression,
            "pointwise_monotonicity_regression": {
                "proposal_count": int(len(pointwise_rows)),
                "rows": pointwise_rows,
                "passed": False,
            },
            "selected_branch": "F_metric_or_mapping_bug",
            "acceptance": {
                "production_promotion_allowed": False,
                "reason": "pointwise halfspace-removal monotonicity failed; pruning trials intentionally skipped",
            },
            "timing_seconds": {"total": as_json_float(time.perf_counter() - started)},
        }

    def classify_trial(
        *,
        topology_valid: bool,
        unrelated_loss: bool,
        canonical_lost: set[int],
        containment_newly_outside: int,
        lost_z_after: int,
        volume_delta_fraction: float | None,
        bbox_delta_fraction: float | None,
        reprojection_delta_fraction: float | None,
        target_group_lost: bool | None,
    ) -> str:
        if not topology_valid:
            return "K_topology_failure"
        if containment_newly_outside > 0:
            return "F_containment_monotonicity_failure"
        if lost_z_after > int(baseline_outside.get("cumulative_lost_z_levels") or 0):
            return "G_lost_Z_regression"
        if unrelated_loss:
            return "E_unrelated_protected_patch_loss"
        if canonical_lost:
            return "D_canonical_patch_loss"
        volume_bad = volume_delta_fraction is not None and abs(float(volume_delta_fraction)) > 0.02
        bbox_bad = bbox_delta_fraction is not None and float(bbox_delta_fraction) > 0.02
        reprojection_bad = reprojection_delta_fraction is not None and float(reprojection_delta_fraction) > 0.10
        if volume_bad or bbox_bad or reprojection_bad:
            return "H_expansion_or_reprojection_regression"
        if bool(target_group_lost):
            return "B_safe_target_duplicate_removed"
        return "A_safe_redundant_plane_removed"

    full_rows: list[dict[str, object]] = []
    for proposal_rank, candidate_id in enumerate(proposal_candidate_ids[: max(0, int(max_full_trials))]):
        trial_candidates = [
            candidate for candidate in final_candidates
            if int(candidate_track_id(candidate)) != int(candidate_id)
        ]
        rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
        trial_active_indices = [
            int(index)
            for index in rec.get("face_candidate_indices", []) or []
            if 0 <= int(index) < len(trial_candidates)
        ]
        trial_active = [trial_candidates[index] for index in trial_active_indices]
        trial_mask = combined_outside_mask(trial_candidates, trusted_points, point_tol=float(point_tol))
        newly_outside = int(np.sum(trial_mask & ~baseline_mask))
        newly_inside = int(np.sum(baseline_mask & ~trial_mask))
        outside_after = cumulative_metrics_from_mask(trial_mask, z_slices)
        trial_canonical_ids = canonical_ids(trial_candidates, rec)
        gained = trial_canonical_ids - baseline_canonical_ids
        lost = baseline_canonical_ids - trial_canonical_ids
        topology_valid = topology_is_valid(rec.get("topology") if isinstance(rec.get("topology"), dict) else None)
        preservation = protected_patch_status(trial_active, int(candidate_id))
        volume = finite_float(rec.get("reliable_volume"), float("nan"))
        volume_delta_fraction = (
            float((volume - baseline_volume) / max(abs(baseline_volume), EPS))
            if np.isfinite(volume) and np.isfinite(baseline_volume)
            else None
        )
        bbox_delta = bbox_extent_delta_fraction(baseline_bbox, rec.get("bbox") if isinstance(rec.get("bbox"), dict) else None)
        reprojection = evaluate_polyhedron_reprojection(
            name=f"active_plane_pruning_corrected_{model_name}_{candidate_id}",
            vertices_3d=np.asarray(rec.get("vertices") or [], dtype=float),
            contours=contours,
            direction_count=24,
            contour_sample=48,
        )
        base_reproj_p95 = finite_float(baseline_reprojection.get("symmetric_contour_distance_p95"), float("nan"))
        trial_reproj_p95 = finite_float(reprojection.get("symmetric_contour_distance_p95"), float("nan"))
        reprojection_delta_fraction = (
            float((trial_reproj_p95 - base_reproj_p95) / max(abs(base_reproj_p95), EPS))
            if np.isfinite(base_reproj_p95) and np.isfinite(trial_reproj_p95)
            else None
        )
        target_group = preservation.get("target_group") if isinstance(preservation.get("target_group"), dict) else {}
        classification = classify_trial(
            topology_valid=bool(topology_valid),
            unrelated_loss=not bool(preservation.get("unrelated_groups_preserved")),
            canonical_lost=lost,
            containment_newly_outside=int(newly_outside),
            lost_z_after=int(outside_after.get("cumulative_lost_z_levels") or 0),
            volume_delta_fraction=volume_delta_fraction,
            bbox_delta_fraction=bbox_delta,
            reprojection_delta_fraction=reprojection_delta_fraction,
            target_group_lost=target_group.get("target_group_lost") if isinstance(target_group, dict) else None,
        )
        full_rows.append(
            {
                "removed_candidate_id": int(candidate_id),
                "removed_candidate_index": int(candidate_index_by_id.get(int(candidate_id), -1)),
                "proposal_rank": int(proposal_rank),
                "classification": classification,
                "safe": bool(classification in {"A_safe_redundant_plane_removed", "B_safe_target_duplicate_removed"}),
                "containment_delta": {
                    "trusted_points_outside_halfspaces_before": baseline_outside,
                    "trusted_points_outside_halfspaces_after": outside_after,
                    "newly_outside_count": int(newly_outside),
                    "newly_inside_count": int(newly_inside),
                },
                "expansion_delta": {
                    "volume_delta_fraction": as_json_float(float(volume_delta_fraction)) if volume_delta_fraction is not None else None,
                    "bbox_extent_delta_fraction_max": as_json_float(float(bbox_delta)) if bbox_delta is not None else None,
                    "reprojection_p95_delta_fraction": as_json_float(float(reprojection_delta_fraction)) if reprojection_delta_fraction is not None else None,
                },
                "canonical_ids_gained": sorted(int(v) for v in gained),
                "canonical_ids_lost": sorted(int(v) for v in lost),
                "canonical_unique": int(len(trial_canonical_ids)),
                "active_count_delta": int(len(trial_active_indices) - len(active_indices)),
                "vertices": int(len(rec.get("vertices") or [])),
                "edges": int(len(rec.get("edges") or [])),
                "faces": int(len(rec.get("faces") or [])),
                "topology": rec.get("topology"),
                "topology_valid": bool(topology_valid),
                "volume": rec.get("reliable_volume"),
                "trusted_final_outside": outside_after,
                "max_per_z_outside": outside_after.get("cumulative_max_level_outside_fraction"),
                "lost_z": outside_after.get("cumulative_lost_z_levels"),
                "protected_patch_groups": preservation,
                "reprojection": reprojection,
                "geometry_digest": mesh_geometry_signature(trial_candidates, rec),
                "proposal": next((row for row in proposal_rows if int(row.get("candidate_id") or -1) == int(candidate_id)), None),
            }
        )

    safe_rows = [row for row in full_rows if bool(row.get("safe"))]
    sequential_rows: list[dict[str, object]] = []
    if len(safe_rows) >= 3:
        removed: set[int] = set()
        state_candidates = list(final_candidates)
        for step, row in enumerate(safe_rows[:10], start=1):
            removed.add(int(row["removed_candidate_id"]))
            state_candidates = [
                candidate for candidate in final_candidates
                if int(candidate_track_id(candidate)) not in removed
            ]
            rec = reconstruct_polyhedron_from_halfspaces_edge_clip(state_candidates, **reconstruction_kwargs)
            ids = canonical_ids(state_candidates, rec)
            mask = combined_outside_mask(state_candidates, trusted_points, point_tol=float(point_tol))
            outside = cumulative_metrics_from_mask(mask, z_slices)
            sequential_rows.append(
                {
                    "step": int(step),
                    "removed_candidate_ids": sorted(int(v) for v in removed),
                    "canonical_unique": int(len(ids)),
                    "canonical_ids_gained": sorted(int(v) for v in ids - baseline_canonical_ids),
                    "canonical_ids_lost": sorted(int(v) for v in baseline_canonical_ids - ids),
                    "vertices": int(len(rec.get("vertices") or [])),
                    "edges": int(len(rec.get("edges") or [])),
                    "faces": int(len(rec.get("faces") or [])),
                    "topology": rec.get("topology"),
                    "volume": rec.get("reliable_volume"),
                    "trusted_final_outside": outside,
                    "geometry_digest": mesh_geometry_signature(state_candidates, rec),
                }
            )

    classification_counts = {
        key: int(sum(1 for row in full_rows if str(row.get("classification")) == key))
        for key in sorted({str(row.get("classification")) for row in full_rows})
    }
    if safe_rows and len(sequential_rows) >= 3:
        selected_branch = "A_corrected_safe_pruning_frontier_exists"
    elif any(str(row.get("classification")) == "H_expansion_or_reprojection_regression" for row in full_rows):
        selected_branch = "C_containment_safe_but_shape_expansion_regresses"
    elif any(str(row.get("classification")) == "D_canonical_patch_loss" for row in full_rows):
        selected_branch = "D_removal_loses_canonical_surface"
    elif any(str(row.get("classification")) == "E_unrelated_protected_patch_loss" for row in full_rows):
        selected_branch = "E_removal_loses_unrelated_observed_patch"
    elif full_rows:
        selected_branch = "B_candidates_are_redundant_but_no_promotable_frontier"
    else:
        selected_branch = "F_no_valid_remove_one_proposals"

    baseline_rec = baseline_payload.get("reconstructed") if isinstance(baseline_payload, dict) else None
    baseline_candidates = baseline_payload.get("face_candidates") if isinstance(baseline_payload, dict) else None
    return {
        "diagnostic_only": True,
        "production_changed": False,
        "production_mode_added": False,
        "scope": "active-plane-pruning-corrected",
        "model": str(model_name),
        "canonical_evaluator": canonical_oracle_evaluator_metadata(),
        "outside_metric_map": outside_metric_map,
        "synthetic_halfspace_regression": synthetic_regression,
        "pointwise_monotonicity_regression": {
            "proposal_count": int(len(pointwise_rows)),
            "rows": pointwise_rows,
            "passed": bool(monotonicity_passed),
        },
        "protected_group_semantics": {
            "target_removal_group_is_hard_fail": False,
            "unrelated_protected_group_loss_is_hard_fail": True,
            "target_group_recorded_under": "remove_one_full_trials.rows[].protected_patch_groups.target_group",
        },
        "frozen_control": {
            "baseline_json_path": baseline_json_path,
            "previous_pruning_json_path": previous_pruning_json_path,
            "artifact_parity": {
                "candidate_ids_match": (
                    [int(candidate_track_id(candidate)) for candidate in baseline_candidates]
                    == [int(candidate_track_id(candidate)) for candidate in final_candidates]
                    if isinstance(baseline_candidates, list) else None
                ),
                "face_candidate_indices_match": (
                    [int(v) for v in baseline_rec.get("face_candidate_indices", [])]
                    == [int(v) for v in final_reconstructed.get("face_candidate_indices", [])]
                    if isinstance(baseline_rec, dict) else None
                ),
                "vertices_match": (
                    int(len(baseline_rec.get("vertices") or [])) == int(len(final_reconstructed.get("vertices") or []))
                    if isinstance(baseline_rec, dict) else None
                ),
                "faces_match": (
                    int(len(baseline_rec.get("faces") or [])) == int(len(final_reconstructed.get("faces") or []))
                    if isinstance(baseline_rec, dict) else None
                ),
            },
            "candidate_count": int(len(final_candidates)),
            "active_candidate_count": int(len(active_candidates)),
            "active_candidate_ids": sorted(int(v) for v in active_candidate_ids),
            "face_candidate_indices": active_indices,
            "vertices": int(len(final_reconstructed.get("vertices") or [])),
            "edges": int(len(final_reconstructed.get("edges") or [])),
            "faces": int(len(final_reconstructed.get("faces") or [])),
            "topology": final_reconstructed.get("topology"),
            "topology_valid": topology_is_valid(final_reconstructed.get("topology") if isinstance(final_reconstructed.get("topology"), dict) else None),
            "volume": final_reconstructed.get("reliable_volume"),
            "bbox": final_reconstructed.get("bbox"),
            "geometry_digest": mesh_geometry_signature(final_candidates, final_reconstructed),
            "canonical_unique_ids": sorted(int(v) for v in baseline_canonical_ids),
            "canonical_unique": int(len(baseline_canonical_ids)),
            "trusted_cloud": trusted_cloud_signature(trusted_points, trusted_z_indices),
            "containment": {
                "trusted_points_outside_halfspaces": baseline_outside,
                "max_per_z_trusted_points_outside_halfspaces": baseline_outside.get("cumulative_max_level_outside_fraction"),
                "lost_z_trusted_points_outside_halfspaces": baseline_outside.get("cumulative_lost_z_levels"),
            },
            "reprojection": baseline_reprojection,
        },
        "proposal_source": {
            "loaded_previous_pruning_order": bool(proposal_rows),
            "proposal_count": int(len(proposal_candidate_ids)),
            "proposal_candidate_ids_sample": [int(v) for v in proposal_candidate_ids[:80]],
            "order_semantics": "same non-oracle proposal order from previous audit when available; fallback is active candidate observed quality order",
        },
        "remove_one_full_trials": {
            "max_trials": int(max_full_trials),
            "trial_count": int(len(full_rows)),
            "classification_counts": classification_counts,
            "safe_count": int(len(safe_rows)),
            "rows": full_rows,
        },
        "safe_sequential_pruning_frontier": {
            "executed": bool(sequential_rows),
            "reason": None if sequential_rows else "skipped: fewer than 3 safe remove-one rows",
            "rows": sequential_rows,
        },
        "selected_branch": selected_branch,
        "acceptance": {
            "production_promotion_allowed": False,
            "reason": "diagnostic-only corrected pruning scope; no production selector/default changed",
        },
        "timing_seconds": {"total": as_json_float(time.perf_counter() - started)},
    }


def observed_symmetry_diagnostics(
    *,
    model_name: str,
    vertices: np.ndarray,
    model_faces: list[dict[str, object]],
    final_candidates: list[dict[str, object]],
    candidate_pool: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    contours: list[object],
    baseline_payload: dict[str, object] | None = None,
    baseline_json_path: str | None = None,
    max_existing_trials: int = 15,
    max_synthetic_trials: int = 10,
) -> dict[str, object]:
    started = time.perf_counter()
    points = np.asarray(trusted_points, dtype=float).reshape((-1, 3)) if np.asarray(trusted_points).size else np.zeros((0, 3), dtype=float)
    z_indices = np.asarray(trusted_z_indices, dtype=int).reshape((-1,))
    model_faces = list(model_faces)
    z_slices = prepare_z_level_slices(z_indices)
    active_indices = [
        int(index)
        for index in final_reconstructed.get("face_candidate_indices", []) or []
        if 0 <= int(index) < len(final_candidates)
    ]
    active_candidates = [final_candidates[index] for index in active_indices]
    active_ids = {int(candidate_track_id(candidate)) for candidate in active_candidates}
    final_candidate_ids = {int(candidate_track_id(candidate)) for candidate in final_candidates}
    pool_by_id = {int(candidate_track_id(candidate)): candidate for candidate in candidate_pool}
    for candidate in final_candidates:
        pool_by_id.setdefault(int(candidate_track_id(candidate)), candidate)
    candidate_pool = list(pool_by_id.values())
    match_cache: dict[int, dict[str, object] | None] = {}

    def canonical_match(candidate: dict[str, object]) -> dict[str, object] | None:
        candidate_id = int(candidate_track_id(candidate))
        if candidate_id not in match_cache:
            match_cache[candidate_id] = canonical_candidate_oracle_face(candidate, model_faces)
        return match_cache[candidate_id]

    def canonical_ids(candidates: list[dict[str, object]], rec: dict[str, object]) -> set[int]:
        ids: set[int] = set()
        for raw_index in rec.get("face_candidate_indices", []) or []:
            index = int(raw_index)
            if index < 0 or index >= len(candidates):
                continue
            match = canonical_match(candidates[index])
            if match is not None and bool(match.get("finite_good")) and match.get("face_id") is not None:
                ids.add(int(match["face_id"]))
        return ids

    def bbox_diag() -> float:
        if points.shape[0] == 0:
            return 1.0
        ext = np.max(points, axis=0) - np.min(points, axis=0)
        return max(float(np.linalg.norm(ext)), EPS)

    diag = bbox_diag()

    def point_spacing() -> float:
        if points.shape[0] < 2:
            return diag * 0.01
        sample_count = min(900, points.shape[0])
        idx = np.linspace(0, points.shape[0] - 1, sample_count, dtype=int)
        sample = points[idx]
        vals: list[float] = []
        for start in range(0, sample.shape[0], 150):
            block = sample[start:start + 150]
            dist = np.sqrt(np.sum((block[:, None, :] - sample[None, :, :]) ** 2, axis=2))
            dist[dist <= EPS] = np.inf
            vals.extend(float(v) for v in np.min(dist, axis=1) if np.isfinite(v))
        return float(np.median(vals)) if vals else diag * 0.01

    spacing = max(point_spacing(), diag * 1e-5)

    center_rows: list[dict[str, object]] = []
    centers: list[np.ndarray] = []
    z_values: list[int] = []
    for z_order, idx in enumerate(z_slices):
        if idx.size < 4:
            continue
        pts = points[idx]
        xy = pts[:, :2]
        center = np.median(xy, axis=0)
        centers.append(center)
        z_values.append(int(z_order))
        if len(center_rows) < 80:
            center_rows.append(
                {
                    "z_slice_order": int(z_order),
                    "point_count": int(idx.size),
                    "robust_xy_center": [as_json_float(float(center[0])), as_json_float(float(center[1]))],
                }
            )
    global_center_xy = np.median(np.array(centers, dtype=float), axis=0) if centers else np.zeros(2, dtype=float)
    center_arr = np.array(centers, dtype=float) if centers else np.zeros((0, 2), dtype=float)
    center_drift = np.linalg.norm(center_arr - global_center_xy[None, :], axis=1) if center_arr.size else np.zeros(0)
    bbox_min = np.min(points, axis=0) if points.size else np.zeros(3, dtype=float)
    bbox_max = np.max(points, axis=0) if points.size else np.zeros(3, dtype=float)
    per_z_counts = [int(idx.size) for idx in z_slices]
    median_per_z = float(np.median([v for v in per_z_counts if v > 0])) if any(v > 0 for v in per_z_counts) else 1.0
    angular_step = float(2.0 * math.pi / max(8.0, median_per_z))

    def matrix_rotation(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([[c, -s], [s, c]], dtype=float)

    def matrix_reflection(axis_angle: float) -> np.ndarray:
        u = np.array([math.cos(axis_angle), math.sin(axis_angle)], dtype=float)
        return 2.0 * np.outer(u, u) - np.eye(2, dtype=float)

    def transform_points_xy(pts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        out = np.array(pts, dtype=float, copy=True)
        out[:, :2] = (out[:, :2] - global_center_xy[None, :]) @ matrix.T + global_center_xy[None, :]
        return out

    def deterministic_sample_indices(n: int, limit: int) -> np.ndarray:
        if n <= limit:
            return np.arange(n, dtype=int)
        return np.linspace(0, n - 1, int(limit), dtype=int)

    sample_idx = deterministic_sample_indices(points.shape[0], 1800)
    points_sample = points[sample_idx] if sample_idx.size else np.zeros((0, 3), dtype=float)
    z_sample = z_indices[sample_idx] if sample_idx.size and z_indices.size == points.shape[0] else np.zeros((0,), dtype=int)
    buckets: dict[int, np.ndarray] = {}
    for z_value in sorted(set(int(v) for v in z_indices)) if z_indices.size else []:
        idx = np.flatnonzero(z_indices == z_value)
        if idx.size > 320:
            idx = idx[deterministic_sample_indices(idx.size, 320)]
        buckets[int(z_value)] = points[idx]

    def nearest_same_z_distances(transformed: np.ndarray, z_sample_local: np.ndarray) -> np.ndarray:
        vals: list[float] = []
        for point, z_value in zip(transformed, z_sample_local):
            bucket = buckets.get(int(z_value))
            if bucket is None or bucket.size == 0:
                continue
            dist = np.sqrt(np.sum((bucket - point[None, :]) ** 2, axis=1))
            vals.append(float(np.min(dist)))
        return np.array(vals, dtype=float)

    def support_function_delta(matrix: np.ndarray) -> dict[str, object]:
        if points_sample.shape[0] == 0:
            return {"median": None, "p95": None}
        transformed = transform_points_xy(points_sample, matrix)
        angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        dirs = np.column_stack([np.cos(angles), np.sin(angles)])
        a = np.max((points_sample[:, :2] - global_center_xy[None, :]) @ dirs.T, axis=0)
        b = np.max((transformed[:, :2] - global_center_xy[None, :]) @ dirs.T, axis=0)
        delta = np.abs(a - b) / max(diag, EPS)
        return {
            "median": as_json_float(float(np.median(delta))),
            "p95": as_json_float(float(np.percentile(delta, 95))),
            "max": as_json_float(float(np.max(delta))),
        }

    def score_transform(name: str, matrix: np.ndarray, *, order: int, kind: str, angle: float) -> dict[str, object]:
        transformed = transform_points_xy(points_sample, matrix) if points_sample.size else np.zeros((0, 3), dtype=float)
        d_ab = nearest_same_z_distances(transformed, z_sample)
        inverse = matrix.T
        inv_transformed = transform_points_xy(points_sample, inverse) if points_sample.size else np.zeros((0, 3), dtype=float)
        d_ba = nearest_same_z_distances(inv_transformed, z_sample)
        distances = np.concatenate([d_ab, d_ba]) if d_ab.size or d_ba.size else np.zeros(0, dtype=float)
        counterpart_tol = max(4.0 * spacing, 0.006 * diag)
        p95_norm = float(np.percentile(distances, 95) / max(diag, EPS)) if distances.size else float("inf")
        spacing_p95 = float(np.percentile(distances, 95) / max(spacing, EPS)) if distances.size else float("inf")
        accepted = bool(distances.size and p95_norm <= max(0.012, 5.0 * spacing / max(diag, EPS)) and spacing_p95 <= 8.0)
        weak = bool(distances.size and not accepted and p95_norm <= max(0.030, 12.0 * spacing / max(diag, EPS)))
        per_zone: dict[str, float | None] = {}
        if distances.size and z_sample.size == points_sample.shape[0]:
            z_min = float(np.min(points_sample[:, 2]))
            z_max = float(np.max(points_sample[:, 2]))
            cuts = [z_min, z_min + (z_max - z_min) / 3.0, z_min + 2.0 * (z_max - z_min) / 3.0, z_max + EPS]
            for label, lo, hi in zip(["lower", "middle", "upper"], cuts[:-1], cuts[1:]):
                zone_mask = (points_sample[:, 2] >= lo) & (points_sample[:, 2] < hi)
                if not np.any(zone_mask):
                    per_zone[label] = None
                    continue
                dz = nearest_same_z_distances(transform_points_xy(points_sample[zone_mask], matrix), z_sample[zone_mask])
                per_zone[label] = as_json_float(float(np.percentile(dz, 95) / max(diag, EPS))) if dz.size else None
        return {
            "transform_id": name,
            "kind": kind,
            "order": int(order),
            "angle_rad": as_json_float(float(angle)),
            "matrix_2d": [[as_json_float(float(v)) for v in row] for row in matrix],
            "determinant": as_json_float(float(np.linalg.det(matrix))),
            "sampled_points": int(points_sample.shape[0]),
            "distance_median": as_json_float(float(np.median(distances))) if distances.size else None,
            "distance_p95": as_json_float(float(np.percentile(distances, 95))) if distances.size else None,
            "distance_p99": as_json_float(float(np.percentile(distances, 99))) if distances.size else None,
            "normalized_median": as_json_float(float(np.median(distances) / max(diag, EPS))) if distances.size else None,
            "normalized_p95": as_json_float(p95_norm) if np.isfinite(p95_norm) else None,
            "spacing_normalized_p95": as_json_float(spacing_p95) if np.isfinite(spacing_p95) else None,
            "fraction_with_counterpart": as_json_float(float(np.mean(distances <= counterpart_tol))) if distances.size else None,
            "per_zone_p95_normalized": per_zone,
            "support_function_delta": support_function_delta(matrix),
            "confidence": "accepted" if accepted else ("weak" if weak else "rejected"),
        }

    transforms: list[tuple[str, np.ndarray, int, str, float]] = [("identity", np.eye(2, dtype=float), 1, "rotation", 0.0)]
    for order in [2, 4, 8]:
        for k in range(1, order):
            transforms.append((f"rot{order}_{k}", matrix_rotation(2.0 * math.pi * k / order), order, "rotation", 2.0 * math.pi * k / order))
    axis_steps = max(8, min(32, int(math.ceil(math.pi / max(angular_step, 1e-3)))))
    reflection_candidates = [
        (float(angle), matrix_reflection(float(angle)))
        for angle in np.linspace(0.0, math.pi, axis_steps, endpoint=False)
    ]
    reflection_scores = [
        score_transform(f"reflection_axis_{idx:02d}", mat, order=2, kind="reflection", angle=angle)
        for idx, (angle, mat) in enumerate(reflection_candidates)
    ]
    reflection_scores.sort(key=lambda row: finite_float(row.get("normalized_p95"), float("inf")))
    best_reflection = reflection_scores[0] if reflection_scores else None
    if best_reflection is not None:
        best_axis = finite_float(best_reflection.get("angle_rad"), 0.0)
        for delta in np.linspace(-angular_step, angular_step, 9):
            angle = float((best_axis + float(delta)) % math.pi)
            transforms.append((f"reflection_refined_{len(transforms)}", matrix_reflection(angle), 2, "reflection", angle))
    score_rows = [score_transform(name, mat, order=order, kind=kind, angle=angle) for name, mat, order, kind, angle in transforms]
    score_rows.extend(reflection_scores[:4])
    unique_by_matrix: dict[str, dict[str, object]] = {}
    for row in score_rows:
        key = json.dumps(row.get("matrix_2d"), sort_keys=True)
        if key not in unique_by_matrix or finite_float(row.get("normalized_p95"), float("inf")) < finite_float(unique_by_matrix[key].get("normalized_p95"), float("inf")):
            unique_by_matrix[key] = row
    score_rows = sorted(unique_by_matrix.values(), key=lambda row: (finite_float(row.get("normalized_p95"), float("inf")), str(row.get("transform_id"))))
    accepted_transforms = [row for row in score_rows if row.get("confidence") == "accepted" and row.get("transform_id") != "identity"]
    weak_transforms = [row for row in score_rows if row.get("confidence") == "weak" and row.get("transform_id") != "identity"]
    selected_transform_rows = accepted_transforms or weak_transforms[:1]

    matrix_by_id: dict[str, np.ndarray] = {}
    for row in score_rows:
        matrix_by_id[str(row["transform_id"])] = np.array(row["matrix_2d"], dtype=float)

    def transform_candidate(candidate: dict[str, object], transform_id: str) -> dict[str, object] | None:
        matrix = matrix_by_id.get(transform_id)
        plane = candidate_plane(candidate)
        if matrix is None or plane is None:
            return None
        point, normal = plane
        mat3 = np.eye(3, dtype=float)
        mat3[:2, :2] = matrix
        point_out = point.copy()
        point_out[:2] = (point[:2] - global_center_xy) @ matrix.T + global_center_xy
        normal_out = mat3 @ normal
        normal_out = normal_out / max(float(np.linalg.norm(normal_out)), EPS)
        hull = candidate_hull_points(candidate)
        hull_out = transform_points_xy(hull, matrix) if hull.size else np.array([point_out], dtype=float)
        out = dict(candidate)
        out["track_id"] = int(candidate_track_id(candidate)) * 1000003 + abs(hash(transform_id)) % 100000
        out["source_track_id"] = int(candidate_track_id(candidate))
        out["candidate_origin"] = "diagnostic_symmetry_transform"
        out["symmetry_transform_id"] = str(transform_id)
        out["plane_centroid"] = as_json_point(point_out)
        out["plane_normal"] = as_json_point(normal_out)
        out["hull"] = [as_json_point(point) for point in hull_out]
        out["hull_area"] = as_json_float(polygon_area(hull_out)) if hull_out.shape[0] >= 3 else candidate.get("hull_area")
        out["hull_diameter"] = as_json_float(float(np.max(np.linalg.norm(hull_out[:, None, :] - hull_out[None, :, :], axis=2)))) if hull_out.shape[0] >= 2 else candidate.get("hull_diameter")
        return out

    def observed_support(candidate: dict[str, object]) -> dict[str, object]:
        plane = candidate_plane(candidate)
        hull = candidate_hull_points(candidate)
        if plane is None or points.shape[0] == 0 or hull.shape[0] == 0:
            return {"support_count": 0, "z_count": 0, "stable": False}
        point, normal = plane
        plane_dist = np.abs((points - point[None, :]) @ normal)
        u, v = plane_basis(normal)
        origin = np.mean(hull, axis=0)
        hull2 = np.column_stack([(hull - origin[None, :]) @ u, (hull - origin[None, :]) @ v])
        pts2 = np.column_stack([(points - origin[None, :]) @ u, (points - origin[None, :]) @ v])
        poly_dist = point_to_polygon_distance_2d(pts2, hull2)
        near = (plane_dist <= max(3.0 * spacing, 0.004 * diag, 0.025)) & (poly_dist <= max(6.0 * spacing, 0.010 * diag, 0.060))
        idx = np.flatnonzero(near)
        z_count = int(len(set(int(v) for v in z_indices[idx]))) if idx.size and z_indices.size == points.shape[0] else 0
        if idx.size >= 3:
            local2 = pts2[idx]
            span = np.ptp(local2, axis=0)
        else:
            span = np.zeros(2, dtype=float)
        return {
            "support_count": int(idx.size),
            "z_count": int(z_count),
            "plane_distance_median": as_json_float(float(np.median(plane_dist[idx]))) if idx.size else None,
            "plane_distance_p95": as_json_float(float(np.percentile(plane_dist[idx], 95))) if idx.size else None,
            "span_u": as_json_float(float(span[0])),
            "span_v": as_json_float(float(span[1])),
            "stable": bool(idx.size >= 12 and z_count >= 3 and min(float(span[0]), float(span[1])) >= max(2.0 * spacing, 0.002 * diag)),
            "_indices": idx,
        }

    def best_equivalent(candidate: dict[str, object]) -> dict[str, object] | None:
        best: tuple[tuple[float, float, float, int], dict[str, object], dict[str, object]] | None = None
        for pool_candidate in candidate_pool:
            relation = plane_patch_relation(candidate, pool_candidate)
            if not bool(relation.get("compatible")):
                continue
            key = (
                finite_float(relation.get("normal_angle_deg"), float("inf")),
                finite_float(relation.get("signed_offset_distance"), float("inf")),
                finite_float(relation.get("centroid_distance"), float("inf")),
                int(candidate_track_id(pool_candidate)),
            )
            if best is None or key < best[0]:
                best = (key, pool_candidate, relation)
        if best is None:
            return None
        candidate_id = int(candidate_track_id(best[1]))
        return {
            "candidate_id": candidate_id,
            "active": bool(candidate_id in active_ids),
            "accepted_final_candidate": bool(candidate_id in final_candidate_ids),
            "relation": best[2],
            "_candidate": best[1],
        }

    baseline_canonical_ids = canonical_ids(final_candidates, final_reconstructed)
    protected_groups: list[list[dict[str, object]]] = []
    for candidate in sorted(active_candidates, key=nonoracle_candidate_quality_key):
        for group in protected_groups:
            if all(candidates_plane_patch_compatible(candidate, member) for member in group):
                group.append(candidate)
                break
        else:
            protected_groups.append([candidate])

    def protected_preserved(trial_active: list[dict[str, object]]) -> dict[str, object]:
        lost: list[list[int]] = []
        for group in protected_groups:
            if any(candidates_plane_patch_compatible(member, candidate) for member in group for candidate in trial_active):
                continue
            lost.append([int(candidate_track_id(member)) for member in group])
        return {
            "protected_group_count": int(len(protected_groups)),
            "unrelated_lost_group_count": int(len(lost)),
            "unrelated_lost_groups_sample": lost[:20],
            "all_preserved": bool(not lost),
        }

    orbit_rows: list[dict[str, object]] = []
    existing_trial_specs: list[dict[str, object]] = []
    synthetic_specs: list[dict[str, object]] = []
    for transform_row in selected_transform_rows:
        transform_id = str(transform_row["transform_id"])
        for active_candidate in active_candidates:
            transformed = transform_candidate(active_candidate, transform_id)
            if transformed is None:
                continue
            support = observed_support(transformed)
            eq = best_equivalent(transformed)
            active_source_id = int(candidate_track_id(active_candidate))
            match_src = canonical_match(active_candidate)
            match_transformed = canonical_candidate_oracle_face(transformed, model_faces)
            status = "unsupported_transformed_member"
            if eq is not None and bool(eq["active"]):
                status = "active_near_equivalent_member"
            elif eq is not None and bool(eq["accepted_final_candidate"]):
                status = "candidate_exists_but_inactive"
            elif eq is not None:
                status = "missing_candidate_member_existing_pool"
            elif bool(support.get("stable")):
                status = "missing_candidate_member_with_observed_support"
            orbit_row = {
                "transform_id": transform_id,
                "source_active_candidate_id": active_source_id,
                "source_active_canonical": match_src,
                "transformed_oracle_posthoc": match_transformed,
                "member_status": status,
                "observed_support": {k: v for k, v in support.items() if not str(k).startswith("_")},
                "equivalent_candidate": {k: v for k, v in (eq or {}).items() if not str(k).startswith("_")},
            }
            orbit_rows.append(orbit_row)
            if eq is not None and not bool(eq["active"]):
                existing_candidate = eq.get("_candidate")
                if isinstance(existing_candidate, dict):
                    existing_trial_specs.append(
                        {
                            "kind": "existing_candidate",
                            "transform_id": transform_id,
                            "source_active_candidate_id": active_source_id,
                            "candidate": existing_candidate,
                            "rank_key": (
                                finite_float((eq.get("relation") or {}).get("normal_angle_deg"), float("inf")),
                                finite_float((eq.get("relation") or {}).get("signed_offset_distance"), float("inf")),
                                -int(support.get("support_count") or 0),
                                int(candidate_track_id(existing_candidate)),
                            ),
                            "orbit_row": orbit_row,
                        }
                    )
            elif eq is None and bool(support.get("stable")):
                idx = support.get("_indices")
                if isinstance(idx, np.ndarray) and idx.size >= 8:
                    local_points = points[idx]
                    refit = fit_plane(local_points)
                    seed_plane = candidate_plane(transformed)
                    if refit is not None and seed_plane is not None:
                        _, seed_normal = seed_plane
                        normal = refit.normal if float(refit.normal @ seed_normal) >= 0.0 else -refit.normal
                        signed = (local_points - refit.centroid[None, :]) @ normal
                        inliers = local_points[np.abs(signed) <= max(3.0 * spacing, 0.006 * diag, 0.035)]
                        if inliers.shape[0] >= 8:
                            u, v = plane_basis(normal)
                            origin = np.mean(inliers, axis=0)
                            pts2 = np.column_stack([(inliers - origin[None, :]) @ u, (inliers - origin[None, :]) @ v])
                            hull_idx = convex_hull_indices(pts2)
                            hull = inliers[np.array(hull_idx, dtype=int)] if hull_idx else inliers[:1]
                            synth = {
                                "track_id": int(910000000 + len(synthetic_specs)),
                                "candidate_origin": "diagnostic_symmetry_local_refit",
                                "candidate_source": "observed_symmetry_orbit",
                                "symmetry_transform_id": transform_id,
                                "source_track_id": active_source_id,
                                "plane_centroid": as_json_point(refit.centroid),
                                "plane_normal": as_json_point(normal),
                                "plane_rms": as_json_float(float(refit.rms)),
                                "plane_max_abs": as_json_float(float(refit.max_abs)),
                                "hull": [as_json_point(point) for point in hull],
                                "hull_area": as_json_float(polygon_area(hull)),
                                "hull_diameter": as_json_float(float(np.max(np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2)))) if hull.shape[0] >= 2 else 0.0,
                                "levels": int(len(set(int(v) for v in z_indices[idx]))) if z_indices.size == points.shape[0] else 0,
                                "z_indices": sorted(int(v) for v in set(z_indices[idx])) if z_indices.size == points.shape[0] else [],
                                "symmetry_refit_support_count": int(inliers.shape[0]),
                                "symmetry_refit_seed_movement": as_json_float(float(np.linalg.norm(refit.centroid - seed_plane[0]))),
                            }
                            synthetic_specs.append(
                                {
                                    "kind": "synthetic_refit",
                                    "transform_id": transform_id,
                                    "source_active_candidate_id": active_source_id,
                                    "candidate": synth,
                                    "rank_key": (
                                        finite_float(synth.get("plane_rms"), float("inf")),
                                        -int(synth.get("symmetry_refit_support_count") or 0),
                                        int(synth.get("track_id") or 0),
                                    ),
                                    "orbit_row": orbit_row,
                                }
                            )

    existing_trial_specs.sort(key=lambda row: row["rank_key"])
    synthetic_specs.sort(key=lambda row: row["rank_key"])

    def run_trial(spec: dict[str, object], trial_index: int) -> dict[str, object]:
        candidate = spec["candidate"]
        assert isinstance(candidate, dict)
        candidate_id = int(candidate_track_id(candidate))
        trial_candidates = list(final_candidates)
        if candidate_id not in {int(candidate_track_id(item)) for item in trial_candidates}:
            trial_candidates.append(candidate)
        rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
        active_after_indices = [
            int(index)
            for index in rec.get("face_candidate_indices", []) or []
            if 0 <= int(index) < len(trial_candidates)
        ]
        active_after = [trial_candidates[index] for index in active_after_indices]
        active_after_ids = {int(candidate_track_id(candidate)) for candidate in active_after}
        ids = canonical_ids(trial_candidates, rec)
        gained = ids - baseline_canonical_ids
        lost = baseline_canonical_ids - ids
        outside = evaluate_trusted_cloud_outside(trial_candidates, points, z_slices, point_tol=float(point_tol))
        reprojection = evaluate_polyhedron_reprojection(
            name=f"symmetry_orbit_{model_name}_{candidate_id}",
            vertices_3d=np.asarray(rec.get("vertices") or [], dtype=float),
            contours=contours,
            direction_count=24,
            contour_sample=48,
        )
        preservation = protected_preserved(active_after)
        topology_valid = topology_is_valid(rec.get("topology") if isinstance(rec.get("topology"), dict) else None)
        candidate_active = bool(candidate_id in active_after_ids)
        if not bool(topology_valid):
            classification = "geometry_regression"
        elif not bool(preservation.get("all_preserved")):
            classification = "protected_patch_loss"
        elif finite_float(outside.get("cumulative_outside_fraction"), float("inf")) > finite_float(baseline_outside.get("cumulative_outside_fraction"), 0.0) + 1e-9:
            classification = "geometry_regression"
        elif int(outside.get("cumulative_lost_z_levels") or 0) != 0:
            classification = "geometry_regression"
        elif not candidate_active:
            classification = "inactive"
        elif lost:
            classification = "symmetry_swap_neutral" if gained and len(ids) == len(baseline_canonical_ids) else "geometry_regression"
        elif gained:
            classification = "safe_orbit_completion_multi_gain" if len(gained) > 1 else "safe_symmetry_plus_one"
        elif str(spec.get("kind")) == "synthetic_refit" and not gained:
            classification = "unsupported_prior_hallucination"
        else:
            classification = "redundant"
        return {
            "trial_index": int(trial_index),
            "kind": str(spec.get("kind")),
            "transform_id": str(spec.get("transform_id")),
            "source_active_candidate_id": int(spec.get("source_active_candidate_id") or -1),
            "candidate_id": int(candidate_id),
            "candidate_final_active": bool(candidate_active),
            "classification": classification,
            "safe": bool(classification in {"safe_symmetry_plus_one", "safe_orbit_completion_multi_gain"}),
            "canonical_ids_gained": sorted(int(v) for v in gained),
            "canonical_ids_lost": sorted(int(v) for v in lost),
            "canonical_unique": int(len(ids)),
            "active_count_delta": int(len(active_after_indices) - len(active_indices)),
            "vertices": int(len(rec.get("vertices") or [])),
            "edges": int(len(rec.get("edges") or [])),
            "faces": int(len(rec.get("faces") or [])),
            "topology": rec.get("topology"),
            "topology_valid": bool(topology_valid),
            "volume": rec.get("reliable_volume"),
            "trusted_final_outside": outside,
            "lost_z": outside.get("cumulative_lost_z_levels"),
            "protected_patch_groups": preservation,
            "reprojection": reprojection,
            "geometry_digest": mesh_geometry_signature(trial_candidates, rec),
            "orbit_evidence": {k: v for k, v in (spec.get("orbit_row") or {}).items() if not str(k).startswith("_")},
        }

    baseline_outside = evaluate_trusted_cloud_outside(final_candidates, points, z_slices, point_tol=float(point_tol))
    existing_trials = [run_trial(spec, index) for index, spec in enumerate(existing_trial_specs[: max(0, int(max_existing_trials))])]
    existing_safe_count = int(sum(1 for row in existing_trials if bool(row.get("safe"))))
    synthetic_trials: list[dict[str, object]] = []
    strong_symmetry = bool(any(row.get("confidence") == "accepted" for row in selected_transform_rows))
    if strong_symmetry and existing_safe_count == 0:
        synthetic_trials = [
            run_trial(spec, index)
            for index, spec in enumerate(synthetic_specs[: max(0, int(max_synthetic_trials))])
        ]

    all_safe = [row for row in existing_trials + synthetic_trials if bool(row.get("safe"))]
    collective_rows: list[dict[str, object]] = []
    if all_safe:
        trial_candidates = list(final_candidates)
        applied: list[int] = []
        for step, row in enumerate(all_safe[:3], start=1):
            cid = int(row["candidate_id"])
            candidate = next((spec["candidate"] for spec in existing_trial_specs + synthetic_specs if int(candidate_track_id(spec["candidate"])) == cid), None)
            if isinstance(candidate, dict) and cid not in {int(candidate_track_id(item)) for item in trial_candidates}:
                trial_candidates.append(candidate)
            applied.append(cid)
            rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
            ids = canonical_ids(trial_candidates, rec)
            outside = evaluate_trusted_cloud_outside(trial_candidates, points, z_slices, point_tol=float(point_tol))
            collective_rows.append(
                {
                    "step": int(step),
                    "applied_candidate_ids": [int(v) for v in applied],
                    "canonical_unique": int(len(ids)),
                    "canonical_ids_gained": sorted(int(v) for v in ids - baseline_canonical_ids),
                    "canonical_ids_lost": sorted(int(v) for v in baseline_canonical_ids - ids),
                    "vertices": int(len(rec.get("vertices") or [])),
                    "edges": int(len(rec.get("edges") or [])),
                    "faces": int(len(rec.get("faces") or [])),
                    "topology": rec.get("topology"),
                    "volume": rec.get("reliable_volume"),
                    "trusted_final_outside": outside,
                    "geometry_digest": mesh_geometry_signature(trial_candidates, rec),
                }
            )

    orbit_status_counts = {
        key: int(sum(1 for row in orbit_rows if row.get("member_status") == key))
        for key in sorted({str(row.get("member_status")) for row in orbit_rows})
    }
    missing_oracle_faces = sorted(
        int((row.get("transformed_oracle_posthoc") or {}).get("face_id"))
        for row in orbit_rows
        if isinstance(row.get("transformed_oracle_posthoc"), dict)
        and bool((row.get("transformed_oracle_posthoc") or {}).get("finite_good"))
        and int((row.get("transformed_oracle_posthoc") or {}).get("face_id")) not in baseline_canonical_ids
    )
    if not selected_transform_rows or selected_transform_rows[0].get("confidence") == "rejected":
        branch = "A_observed_symmetry_insufficient"
    elif not existing_trial_specs and not synthetic_specs:
        branch = "B_symmetry_strong_but_orbit_mates_absent"
    elif synthetic_specs and not existing_trial_specs:
        branch = "C_transformed_mates_require_observed_refit"
    elif any(row.get("classification") == "protected_patch_loss" for row in existing_trials + synthetic_trials):
        branch = "D_orbit_candidates_halfspace_incompatible"
    elif all_safe:
        branch = "E_safe_diagnostic_ceiling_exists_but_selector_unpromoted"
    else:
        branch = "G_current_pipeline_local_optimum_for_available_data"

    baseline_rec = baseline_payload.get("reconstructed") if isinstance(baseline_payload, dict) else None
    baseline_candidates = baseline_payload.get("face_candidates") if isinstance(baseline_payload, dict) else None
    return {
        "diagnostic_only": True,
        "production_changed": False,
        "production_mode_added": False,
        "scope": "observed-symmetry-orbit-audit",
        "model": str(model_name),
        "baseline_json_path": baseline_json_path,
        "canonical_evaluator": canonical_oracle_evaluator_metadata(),
        "observed_coordinate_normalization": {
            "global_center_axis_xy": [as_json_float(float(global_center_xy[0])), as_json_float(float(global_center_xy[1]))],
            "bbox": {"min": as_json_point(bbox_min), "max": as_json_point(bbox_max), "extent": as_json_point(bbox_max - bbox_min)},
            "bbox_diagonal": as_json_float(float(diag)),
            "z_range": [as_json_float(float(bbox_min[2])), as_json_float(float(bbox_max[2]))],
            "median_point_spacing": as_json_float(float(spacing)),
            "angular_sampling_step_rad": as_json_float(float(angular_step)),
            "center_drift_median": as_json_float(float(np.median(center_drift))) if center_drift.size else None,
            "center_drift_p95": as_json_float(float(np.percentile(center_drift, 95))) if center_drift.size else None,
            "per_z_center_rows_sample": center_rows,
        },
        "symmetry_detection": {
            "generation_uses_initial_model": False,
            "generation_uses_model_name": False,
            "candidate_transform_count": int(len(score_rows)),
            "selected_transform_ids": [str(row.get("transform_id")) for row in selected_transform_rows],
            "selected_confidence": [str(row.get("confidence")) for row in selected_transform_rows],
            "rows": score_rows[:24],
            "regression_controls": {
                "inverse_same_score_checked_by_bidirectional_metric": True,
                "composition_rotations_predefined_orders": [1, 2, 4, 8],
                "group_closure_numeric_tolerance": as_json_float(float(max(8.0 * spacing / max(diag, EPS), 1e-6))),
                "deterministic_sample": True,
            },
        },
        "frozen_control": {
            "artifact_parity": {
                "candidate_ids_match": (
                    [int(candidate_track_id(candidate)) for candidate in baseline_candidates]
                    == [int(candidate_track_id(candidate)) for candidate in final_candidates]
                    if isinstance(baseline_candidates, list) else None
                ),
                "face_candidate_indices_match": (
                    [int(v) for v in baseline_rec.get("face_candidate_indices", [])]
                    == [int(v) for v in final_reconstructed.get("face_candidate_indices", [])]
                    if isinstance(baseline_rec, dict) else None
                ),
            },
            "candidate_count": int(len(final_candidates)),
            "candidate_pool_count": int(len(candidate_pool)),
            "active_candidate_count": int(len(active_candidates)),
            "canonical_unique_ids": sorted(int(v) for v in baseline_canonical_ids),
            "canonical_unique": int(len(baseline_canonical_ids)),
            "trusted_final_outside": baseline_outside,
            "geometry_digest": mesh_geometry_signature(final_candidates, final_reconstructed),
        },
        "plane_patch_orbit_audit": {
            "orbit_row_count": int(len(orbit_rows)),
            "status_counts": orbit_status_counts,
            "rows_sample": orbit_rows[:180],
        },
        "posthoc_oracle_orbit_ceiling": {
            "baseline_unique": int(len(baseline_canonical_ids)),
            "missing_symmetry_mate_face_ids": sorted(set(missing_oracle_faces)),
            "missing_symmetry_mate_count": int(len(set(missing_oracle_faces))),
            "existing_candidate_trial_ceiling_safe_count": int(existing_safe_count),
            "synthetic_trial_ceiling_safe_count": int(sum(1 for row in synthetic_trials if bool(row.get("safe")))),
            "category_counts": {
                "A_no_symmetry_relation": 0 if selected_transform_rows else int(len(model_faces) - len(baseline_canonical_ids)),
                "B_symmetry_mate_already_active": int(orbit_status_counts.get("active_near_equivalent_member", 0)),
                "C_mate_exists_in_candidate_pool_but_inactive": int(orbit_status_counts.get("candidate_exists_but_inactive", 0) + orbit_status_counts.get("missing_candidate_member_existing_pool", 0)),
                "D_transformed_mate_has_observed_support_but_candidate_missing": int(orbit_status_counts.get("missing_candidate_member_with_observed_support", 0)),
                "E_transformed_mate_unsupported_by_observations": int(orbit_status_counts.get("unsupported_transformed_member", 0)),
                "F_transform_geometrically_inaccurate": int(sum(1 for row in selected_transform_rows if row.get("confidence") != "accepted")),
            },
        },
        "existing_candidate_orbit_completion": {
            "trial_limit": int(max_existing_trials),
            "trial_count": int(len(existing_trials)),
            "classification_counts": {
                key: int(sum(1 for row in existing_trials if row.get("classification") == key))
                for key in sorted({str(row.get("classification")) for row in existing_trials})
            },
            "rows": existing_trials,
        },
        "synthetic_transformed_hypotheses": {
            "executed": bool(synthetic_trials),
            "reason": None if synthetic_trials else "skipped unless observed symmetry is accepted and existing-candidate safe ceiling is zero",
            "dependency_declaration": [
                {"field": "seed transform", "uses_initial_model": False, "uses_oracle_face_id": False, "uses_canonical_metric": False, "uses_model_name": False, "available_when_oracle_disabled": True},
                {"field": "local refit support", "uses_initial_model": False, "uses_oracle_face_id": False, "uses_canonical_metric": False, "uses_model_name": False, "available_when_oracle_disabled": True},
                {"field": "candidate order", "uses_initial_model": False, "uses_oracle_face_id": False, "uses_canonical_metric": False, "uses_model_name": False, "available_when_oracle_disabled": True},
            ],
            "trial_limit": int(max_synthetic_trials),
            "trial_count": int(len(synthetic_trials)),
            "classification_counts": {
                key: int(sum(1 for row in synthetic_trials if row.get("classification") == key))
                for key in sorted({str(row.get("classification")) for row in synthetic_trials})
            },
            "rows": synthetic_trials,
        },
        "orbit_collective_lower_bound": {
            "executed": bool(collective_rows),
            "max_actions": 3,
            "rows": collective_rows,
        },
        "non_oracle_leakage_audit": {
            "production_eligible_fields": [
                {"field": "observed_coordinate_normalization", "uses_initial_model": False, "uses_oracle_face_id": False, "uses_canonical_metric": False, "uses_model_name": False, "available_when_oracle_disabled": True},
                {"field": "symmetry_detection.rows", "uses_initial_model": False, "uses_oracle_face_id": False, "uses_canonical_metric": False, "uses_model_name": False, "available_when_oracle_disabled": True},
                {"field": "plane_patch_orbit_audit transform/equivalence before posthoc labels", "uses_initial_model": False, "uses_oracle_face_id": False, "uses_canonical_metric": False, "uses_model_name": False, "available_when_oracle_disabled": True},
                {"field": "existing/synthetic candidate order", "uses_initial_model": False, "uses_oracle_face_id": False, "uses_canonical_metric": False, "uses_model_name": False, "available_when_oracle_disabled": True},
            ],
            "oracle_no_oracle_expected_parity": "diagnostic scope is read-only; production geometry is loaded from frozen baseline and no selector/default changes are made",
        },
        "selected_branch": branch,
        "acceptance": {
            "production_promotion_allowed": False,
            "reason": "diagnostic-only symmetry audit; no optional production symmetry mode added",
        },
        "timing_seconds": {"total": as_json_float(time.perf_counter() - started)},
    }


def diagnose_current_candidate_exchange(
    *,
    model_name: str,
    vertices: np.ndarray,
    model_faces: list[dict[str, object]],
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    trusted_points: np.ndarray,
    trusted_z_indices: np.ndarray,
    point_tol: float,
    reconstruction_kwargs: dict[str, object],
    contours: list[object],
    target_face_ids: list[int],
    baseline_payload: dict[str, object] | None = None,
    max_full_trials: int = 30,
) -> dict[str, object]:
    started = time.perf_counter()
    timings = {
        "freeze_seconds": 0.0,
        "target_cohort_seconds": 0.0,
        "lp_single_seconds": 0.0,
        "lp_pair_seconds": 0.0,
        "trusted_cloud_seconds": 0.0,
        "edge_clip_seconds": 0.0,
        "reprojection_seconds": 0.0,
    }
    z_slices = prepare_z_level_slices(trusted_z_indices)
    candidate_by_id = {int(candidate_track_id(candidate)): candidate for candidate in final_candidates}
    final_ids = tuple(int(candidate_track_id(candidate)) for candidate in final_candidates)
    final_id_set = set(final_ids)
    active_indices = [
        int(index)
        for index in final_reconstructed.get("face_candidate_indices", [])
        if 0 <= int(index) < len(final_candidates)
    ]
    active_candidates = [final_candidates[index] for index in active_indices]
    active_candidate_ids = {int(candidate_track_id(candidate)) for candidate in active_candidates}
    inactive_candidate_ids = sorted(int(cid) for cid in final_id_set - active_candidate_ids)
    match_cache: dict[int, dict[str, object] | None] = {}

    def canonical_match(candidate: dict[str, object]) -> dict[str, object] | None:
        candidate_id = int(candidate_track_id(candidate))
        if candidate_id not in match_cache:
            match_cache[candidate_id] = canonical_candidate_oracle_face(candidate, model_faces)
        return match_cache[candidate_id]

    def finite_face_id(candidate: dict[str, object]) -> int | None:
        match = canonical_match(candidate)
        if match is None or not bool(match.get("finite_good")) or match.get("face_id") is None:
            return None
        return int(match["face_id"])

    def active_canonical_ids(candidates: list[dict[str, object]], rec: dict[str, object]) -> set[int]:
        ids: set[int] = set()
        for index in rec.get("face_candidate_indices", []):
            idx = int(index)
            if idx < 0 or idx >= len(candidates):
                continue
            face_id = finite_face_id(candidates[idx])
            if face_id is not None:
                ids.add(int(face_id))
        return ids

    freeze_started = time.perf_counter()
    final_canonical_ids = active_canonical_ids(final_candidates, final_reconstructed)
    final_outside = evaluate_trusted_cloud_outside(
        final_candidates,
        trusted_points,
        z_slices,
        point_tol=float(point_tol),
    )

    def candidate_samples(candidate: dict[str, object]) -> list[np.ndarray]:
        hull = candidate_hull_points(candidate)
        if hull.shape[0] == 0:
            centroid = candidate_hull_centroid(candidate)
            return [np.asarray(centroid, dtype=float)] if centroid is not None else []
        centroid = np.mean(hull, axis=0)
        samples: list[np.ndarray] = [np.asarray(centroid, dtype=float)]
        samples.extend(np.asarray(0.7 * centroid + 0.3 * point, dtype=float) for point in hull)
        samples.extend(
            np.asarray(0.5 * point + 0.5 * hull[(index + 1) % len(hull)], dtype=float)
            for index, point in enumerate(hull)
        )
        return samples

    def active_patch_groups(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        groups: list[list[dict[str, object]]] = []
        for candidate in sorted(candidates, key=nonoracle_candidate_quality_key):
            for group in groups:
                if all(candidates_plane_patch_compatible(candidate, member) for member in group):
                    group.append(candidate)
                    break
            else:
                groups.append([candidate])
        return [
            {
                "group_id": f"active_patch_{index:04d}_{candidate_track_id(group[0])}",
                "members": group,
                "representative_candidate_id": int(candidate_track_id(group[0])),
                "member_candidate_ids": sorted(int(candidate_track_id(member)) for member in group),
            }
            for index, group in enumerate(groups)
        ]

    protected_groups = active_patch_groups(active_candidates)

    def public_group(group: dict[str, object]) -> dict[str, object]:
        return {
            "group_id": str(group.get("group_id")),
            "representative_candidate_id": int(group.get("representative_candidate_id") or -1),
            "member_candidate_ids": [int(v) for v in group.get("member_candidate_ids", [])],
        }

    protected_rows = [public_group(group) for group in protected_groups]
    protected_blob = json.dumps(protected_rows, sort_keys=True, separators=(",", ":"))

    trusted_sample = np.asarray(trusted_points, dtype=float).reshape((-1, 3))
    if trusted_sample.shape[0] > 512:
        trusted_sample = trusted_sample[np.linspace(0, trusted_sample.shape[0] - 1, 512).astype(int)]
    trusted_blob = json.dumps(
        {
            "count": int(np.asarray(trusted_points).reshape((-1, 3)).shape[0]) if np.asarray(trusted_points).size else 0,
            "sample": [[round(float(x), 8) for x in point] for point in trusted_sample],
            "z_count": int(np.asarray(trusted_z_indices).size),
            "z_sum": int(np.sum(trusted_z_indices)) if np.asarray(trusted_z_indices).size else 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    control = {
        "final_candidate_ids": [int(v) for v in final_ids],
        "active_plane_ids": [int(v) for v in final_reconstructed.get("active_plane_indices", [])],
        "active_candidate_ids": sorted(int(v) for v in active_candidate_ids),
        "inactive_accepted_ids": inactive_candidate_ids,
        "protected_patch_group_hash": hashlib.sha256(protected_blob.encode("utf-8")).hexdigest(),
        "protected_patch_group_count": int(len(protected_groups)),
        "protected_patch_groups_sample": protected_rows[:160],
        "trusted_cloud_count": int(np.asarray(trusted_points).reshape((-1, 3)).shape[0]) if np.asarray(trusted_points).size else 0,
        "trusted_cloud_signature": hashlib.sha256(trusted_blob.encode("utf-8")).hexdigest(),
        "trusted_point_tolerance": as_json_float(float(point_tol)),
        "trusted_final_outside": final_outside,
        "max_per_z_outside": final_outside.get("cumulative_max_level_outside_fraction"),
        "lost_z": final_outside.get("cumulative_lost_z_levels"),
        "vertices": int(len(final_reconstructed.get("vertices") or [])),
        "edges": int(len(final_reconstructed.get("edges") or [])),
        "faces": int(len(final_reconstructed.get("faces") or [])),
        "topology": final_reconstructed.get("topology"),
        "topology_valid": topology_is_valid(final_reconstructed.get("topology") if isinstance(final_reconstructed.get("topology"), dict) else None),
        "volume": final_reconstructed.get("reliable_volume"),
        "geometry_digest": mesh_geometry_signature(final_candidates, final_reconstructed),
        "canonical_unique_set": sorted(int(v) for v in final_canonical_ids),
        "canonical_unique_count": int(len(final_canonical_ids)),
    }

    def baseline_candidate_ids(payload: dict[str, object] | None) -> list[int]:
        if not isinstance(payload, dict):
            return []
        return [
            int(candidate_track_id(candidate))
            for candidate in payload.get("face_candidates", [])
            if isinstance(candidate, dict)
        ]

    baseline_rec = baseline_payload.get("reconstructed") if isinstance(baseline_payload, dict) else None
    baseline_ids = baseline_candidate_ids(baseline_payload)
    baseline_face_indices = (
        [int(v) for v in baseline_rec.get("face_candidate_indices", [])]
        if isinstance(baseline_rec, dict) and isinstance(baseline_rec.get("face_candidate_indices"), list)
        else []
    )
    control["artifact_parity"] = {
        "baseline_path_loaded": bool(isinstance(baseline_payload, dict) and not baseline_payload.get("_load_error")),
        "baseline_load_error": baseline_payload.get("_load_error") if isinstance(baseline_payload, dict) else None,
        "candidate_ids_match": bool(baseline_ids == list(final_ids)) if baseline_ids else None,
        "active_plane_indices_match": (
            bool(baseline_face_indices == [int(v) for v in final_reconstructed.get("face_candidate_indices", [])])
            if baseline_face_indices else None
        ),
        "vertices_match": (
            int(len((baseline_rec or {}).get("vertices") or [])) == int(len(final_reconstructed.get("vertices") or []))
            if isinstance(baseline_rec, dict) else None
        ),
        "faces_match": (
            int(len((baseline_rec or {}).get("faces") or [])) == int(len(final_reconstructed.get("faces") or []))
            if isinstance(baseline_rec, dict) else None
        ),
        "volume_match": (
            abs(
                finite_float((baseline_rec or {}).get("reliable_volume"), float("nan"))
                - finite_float(final_reconstructed.get("reliable_volume"), float("nan"))
            )
            <= 1e-9
            if isinstance(baseline_rec, dict) else None
        ),
    }
    timings["freeze_seconds"] = time.perf_counter() - freeze_started

    target_started = time.perf_counter()

    def nonoracle_candidate_public(candidate: dict[str, object]) -> dict[str, object]:
        z_min, z_max = candidate_z_range(candidate)
        return {
            "candidate_id": int(candidate_track_id(candidate)),
            "candidate_origin": candidate.get("candidate_origin"),
            "candidate_source": candidate.get("candidate_source"),
            "window": candidate.get("window"),
            "levels": int(candidate.get("levels") or len(candidate.get("z_indices", []) or [])),
            "z_span": as_json_float(max(0.0, z_max - z_min)) if np.isfinite(z_min) and np.isfinite(z_max) else None,
            "hull_area": as_json_float(finite_float(candidate.get("hull_area"), 0.0)),
            "hull_diameter": as_json_float(finite_float(candidate.get("hull_diameter"), 0.0)),
            "plane_rms": as_json_float(finite_float(candidate.get("plane_rms"), float("nan"))),
            "candidate_score": as_json_float(finite_float(candidate.get("candidate_score"), float("nan"))),
            "finite_support_count": int(candidate.get("finite_support_count") or candidate.get("observed_support_near_count") or 0),
            "finite_support_density": as_json_float(finite_float(candidate.get("finite_support_density"), float("nan"))),
            "w2_edge_cluster_ids": [int(v) for v in candidate.get("w2_edge_cluster_ids", [])],
        }

    targets: list[dict[str, object]] = []
    alternatives_by_face: dict[int, list[dict[str, object]]] = {}
    for candidate in final_candidates:
        face_id = finite_face_id(candidate)
        if face_id is None or face_id not in set(target_face_ids):
            continue
        alternatives_by_face.setdefault(int(face_id), []).append(candidate)
    for face_id in target_face_ids:
        alternatives = alternatives_by_face.get(int(face_id), [])
        alternatives_by_oracle = sorted(
            alternatives,
            key=lambda candidate: (
                finite_float((canonical_match(candidate) or {}).get("score"), float("inf")),
                int(candidate_track_id(candidate)),
            ),
        )
        alternatives_by_nonoracle = sorted(alternatives, key=nonoracle_candidate_quality_key)
        target = alternatives_by_oracle[0] if alternatives_by_oracle else None
        current_best = alternatives_by_nonoracle[0] if alternatives_by_nonoracle else None
        targets.append(
            {
                "face_id": int(face_id),
                "oracle_best_candidate_id": int(candidate_track_id(target)) if target is not None else None,
                "current_best_nonoracle_candidate_id": int(candidate_track_id(current_best)) if current_best is not None else None,
                "accepted_alternative_count": int(len(alternatives)),
                "alternatives": [
                    {
                        **nonoracle_candidate_public(candidate),
                        "active_final": bool(int(candidate_track_id(candidate)) in active_candidate_ids),
                        "oracle_posthoc": canonical_match(candidate),
                    }
                    for candidate in alternatives_by_nonoracle[:3]
                ],
                "_target_candidate": target,
                "dependency_flags": {
                    "target_face_id_source": "oracle_seeded_diagnostic_ceiling",
                    "oracle_best_uses_initial_model": True,
                    "current_best_nonoracle_sort_uses_initial_model": False,
                    "production_eligible": False,
                },
            }
        )
    timings["target_cohort_seconds"] = time.perf_counter() - target_started

    lp_engine = LpActivityEngine(enabled=True)
    lp_engine.set_reusable_universe(
        "current_candidate_exchange",
        final_candidates,
        inside_tol=0.03,
    )

    def base_ids_without(removed_ids: set[int], target_id: int) -> tuple[int, ...]:
        return tuple(
            int(cid)
            for cid in final_ids
            if int(cid) != int(target_id) and int(cid) not in removed_ids
        )

    def activity(target: dict[str, object], removed_ids: set[int], *, include_point: bool) -> dict[str, object]:
        target_id = int(candidate_track_id(target))
        return lp_face_activity_details(
            None,
            target,
            inside_tol=0.03,
            face_activity_slack=0.03,
            include_point=bool(include_point),
            lp_engine=lp_engine,
            base_candidate_ids=base_ids_without(removed_ids, target_id),
        )

    def candidate_overlap_features(target: dict[str, object], blocker: dict[str, object]) -> dict[str, object]:
        angle, offset, centroid_distance = candidate_plane_relation_score(target, blocker)
        tz0, tz1 = candidate_z_range(target)
        bz0, bz1 = candidate_z_range(blocker)
        if np.isfinite(tz0) and np.isfinite(bz0):
            z_overlap = max(0.0, min(tz1, bz1) - max(tz0, bz0))
            z_base = max(EPS, min(tz1 - tz0, bz1 - bz0))
            z_fraction = z_overlap / z_base
        else:
            z_overlap = 0.0
            z_fraction = 0.0
        relation = plane_patch_relation(target, blocker)
        return {
            "normal_angle_deg": as_json_float(angle) if np.isfinite(angle) else None,
            "signed_offset": as_json_float(offset) if np.isfinite(offset) else None,
            "centroid_distance": as_json_float(centroid_distance) if np.isfinite(centroid_distance) else None,
            "finite_hull_overlap": as_json_float(float(hull_bounds_overlap_ratio(target, blocker))),
            "observed_footprint_overlap": relation.get("support_distance_median"),
            "z_overlap": as_json_float(float(z_overlap)),
            "z_overlap_fraction": as_json_float(float(z_fraction)),
            "patch_compatible": bool(candidates_plane_patch_compatible(target, blocker)),
        }

    def blocker_rank_key(row: dict[str, object]) -> tuple[object, ...]:
        after = row.get("after") if isinstance(row.get("after"), dict) else {}
        overlap = row.get("overlap") if isinstance(row.get("overlap"), dict) else {}
        blocker = row.get("blocker_observed") if isinstance(row.get("blocker_observed"), dict) else {}
        return (
            not bool(after.get("active")),
            -finite_float(after.get("effective_face_margin"), -float("inf")),
            finite_float(overlap.get("normal_angle_deg"), float("inf")),
            finite_float(overlap.get("signed_offset"), float("inf")),
            -finite_float(overlap.get("finite_hull_overlap"), 0.0),
            finite_float(blocker.get("candidate_score"), float("inf")),
            int(row.get("blocker_candidate_id") or 0),
        )

    single_started = time.perf_counter()
    blocker_funnel_rows: list[dict[str, object]] = []
    full_trial_specs: list[dict[str, object]] = []
    for target_row in targets:
        target = target_row.get("_target_candidate")
        if not isinstance(target, dict):
            target_row["absence_reason"] = "no_accepted_finite_good_candidate_for_seeded_face"
            continue
        target_id = int(candidate_track_id(target))
        before = activity(target, set(), include_point=True)
        target_row["lp_activity_final"] = before
        scored_blockers: list[dict[str, object]] = []
        tight_ids = set()
        residual = before.get("residual_summary") if isinstance(before.get("residual_summary"), dict) else {}
        for cid in residual.get("active_constraints_sample", []) or []:
            if int(cid) in active_candidate_ids and int(cid) != target_id:
                tight_ids.add(int(cid))
        for blocker in active_candidates:
            blocker_id = int(candidate_track_id(blocker))
            if blocker_id == target_id:
                continue
            overlap = candidate_overlap_features(target, blocker)
            prefilter_score = (
                (0 if blocker_id in tight_ids else 1),
                finite_float(overlap.get("normal_angle_deg"), float("inf")),
                finite_float(overlap.get("signed_offset"), float("inf")),
                -finite_float(overlap.get("finite_hull_overlap"), 0.0),
                finite_float(overlap.get("centroid_distance"), float("inf")),
                blocker_id,
            )
            scored_blockers.append(
                {
                    "blocker_candidate_id": int(blocker_id),
                    "prefilter_score": [as_json_float(float(v)) if isinstance(v, float) else int(v) for v in prefilter_score],
                    "is_tight_at_target_lp_optimum": bool(blocker_id in tight_ids),
                    "overlap": overlap,
                    "blocker_observed": nonoracle_candidate_public(blocker),
                    "_blocker": blocker,
                }
            )
        scored_blockers.sort(key=lambda row: tuple(row["prefilter_score"]))
        single_rows: list[dict[str, object]] = []
        for row in scored_blockers[:12]:
            blocker = row["_blocker"]
            blocker_id = int(row["blocker_candidate_id"])
            after = activity(target, {blocker_id}, include_point=True)
            samples = candidate_samples(target)
            final_deficit = patch_surface_deficit_for_samples(final_candidates, samples, sample_tol=0.01)
            trial_candidates_for_deficit = [candidate for candidate in final_candidates if int(candidate_track_id(candidate)) != blocker_id]
            after_deficit = patch_surface_deficit_for_samples(trial_candidates_for_deficit, samples, sample_tol=0.01)
            out = {
                **{key: value for key, value in row.items() if not str(key).startswith("_")},
                "before": before,
                "after": after,
                "target_candidate_id": int(target_id),
                "target_face_id": int(target_row["face_id"]),
                "target_activity_gain": as_json_float(
                    finite_float(after.get("effective_face_margin"), -999.0)
                    - finite_float(before.get("effective_face_margin"), -999.0)
                ),
                "surface_deficit_before": final_deficit,
                "surface_deficit_after_remove": after_deficit,
                "surface_deficit_delta": as_json_float(
                    finite_float(final_deficit.get("area_weighted_cut_deficit"), 0.0)
                    - finite_float(after_deficit.get("area_weighted_cut_deficit"), 0.0)
                ),
            }
            single_rows.append(out)
        single_rows.sort(key=blocker_rank_key)
        positive = [row for row in single_rows if bool((row.get("after") or {}).get("active"))]
        target_row["single_blocker_screen"] = {
            "screened_active_blockers": int(len(single_rows)),
            "lp_positive": int(len(positive)),
            "rows": single_rows[:8],
        }
        for row in positive[:3]:
            full_trial_specs.append(
                {
                    "kind": "single",
                    "target_face_id": int(target_row["face_id"]),
                    "target_candidate_id": int(target_id),
                    "removed_candidate_ids": [int(row["blocker_candidate_id"])],
                    "lp_row": row,
                    "_target": target,
                }
            )
        blocker_funnel_rows.append(
            {
                "target_face_id": int(target_row["face_id"]),
                "target_candidate_id": int(target_id),
                "final_active": bool(target_id in active_candidate_ids),
                "single_lp_positive": int(len(positive)),
                "top_single_rows": single_rows[:5],
            }
        )
    timings["lp_single_seconds"] = time.perf_counter() - single_started

    pair_started = time.perf_counter()
    for target_row in targets:
        target = target_row.get("_target_candidate")
        if not isinstance(target, dict):
            continue
        if any(int(spec["target_face_id"]) == int(target_row["face_id"]) and spec["kind"] == "single" for spec in full_trial_specs):
            continue
        screen = target_row.get("single_blocker_screen") if isinstance(target_row.get("single_blocker_screen"), dict) else {}
        single_rows = [row for row in screen.get("rows", []) if isinstance(row, dict)]
        candidates_for_pair = [
            row for row in single_rows[:4]
            if not bool((row.get("after") or {}).get("active"))
        ]
        pair_rows: list[dict[str, object]] = []
        for left_index, left in enumerate(candidates_for_pair):
            for right in candidates_for_pair[left_index + 1:]:
                removed = {int(left["blocker_candidate_id"]), int(right["blocker_candidate_id"])}
                after = activity(target, removed, include_point=False)
                if not bool(after.get("active")):
                    continue
                outside_started = time.perf_counter()
                trial_candidates = [
                    candidate for candidate in final_candidates
                    if int(candidate_track_id(candidate)) not in removed
                ]
                preliminary_outside = evaluate_trusted_cloud_outside(
                    trial_candidates,
                    trusted_points,
                    z_slices,
                    point_tol=float(point_tol),
                )
                timings["trusted_cloud_seconds"] += time.perf_counter() - outside_started
                prelim_pass = bool(
                    finite_float(preliminary_outside.get("cumulative_outside_fraction"), float("inf")) < 0.01
                    and int(value_or_default(preliminary_outside.get("cumulative_lost_z_levels"), 1)) == 0
                )
                pair_rows.append(
                    {
                        "target_face_id": int(target_row["face_id"]),
                        "target_candidate_id": int(candidate_track_id(target)),
                        "removed_candidate_ids": sorted(int(v) for v in removed),
                        "strict_synergy": True,
                        "after": after,
                        "preliminary_trusted_outside": preliminary_outside,
                        "preliminary_gate_passed": bool(prelim_pass),
                    }
                )
        target_row["strict_pair_blocker_screen"] = {
            "candidate_pair_count": int(len(candidates_for_pair) * (len(candidates_for_pair) - 1) / 2),
            "lp_positive_strict_pairs": int(len(pair_rows)),
            "rows": pair_rows[:6],
        }
        for row in pair_rows[:3]:
            if bool(row.get("preliminary_gate_passed")):
                full_trial_specs.append(
                    {
                        "kind": "strict_pair",
                        "target_face_id": int(row["target_face_id"]),
                        "target_candidate_id": int(row["target_candidate_id"]),
                        "removed_candidate_ids": [int(v) for v in row["removed_candidate_ids"]],
                        "lp_row": row,
                        "_target": target,
                    }
                )
    timings["lp_pair_seconds"] = time.perf_counter() - pair_started

    def preserved_patches(trial_active: list[dict[str, object]]) -> dict[str, object]:
        lost: list[dict[str, object]] = []
        for group in protected_groups:
            members = list(group.get("members") or [])
            if any(
                candidates_plane_patch_compatible(member, candidate)
                for member in members
                for candidate in trial_active
            ):
                continue
            lost.append(public_group(group))
        return {
            "protected_group_count": int(len(protected_groups)),
            "lost_group_count": int(len(lost)),
            "all_preserved": bool(not lost),
            "lost_groups": lost[:20],
        }

    def trial_candidates_for(removed_ids: set[int], target: dict[str, object]) -> list[dict[str, object]]:
        target_id = int(candidate_track_id(target))
        trial = [candidate for candidate in final_candidates if int(candidate_track_id(candidate)) not in removed_ids]
        if target_id not in {int(candidate_track_id(candidate)) for candidate in trial}:
            trial.append(target)
        return trial

    full_started = time.perf_counter()
    full_trials: list[dict[str, object]] = []
    for spec in full_trial_specs[: max(0, int(max_full_trials))]:
        target = spec["_target"]
        target_id = int(spec["target_candidate_id"])
        removed = {int(v) for v in spec["removed_candidate_ids"]}
        trial_candidates = trial_candidates_for(removed, target)
        rec = reconstruct_polyhedron_from_halfspaces_edge_clip(trial_candidates, **reconstruction_kwargs)
        trial_active_indices = [
            int(index)
            for index in rec.get("face_candidate_indices", [])
            if 0 <= int(index) < len(trial_candidates)
        ]
        trial_active = [trial_candidates[index] for index in trial_active_indices]
        trial_active_ids = {int(candidate_track_id(candidate)) for candidate in trial_active}
        trial_canonical_ids = active_canonical_ids(trial_candidates, rec)
        edge_clip_elapsed = time.perf_counter() - full_started - timings["edge_clip_seconds"]
        timings["edge_clip_seconds"] += edge_clip_elapsed
        outside_started = time.perf_counter()
        outside = evaluate_trusted_cloud_outside(
            trial_candidates,
            trusted_points,
            z_slices,
            point_tol=float(point_tol),
        )
        timings["trusted_cloud_seconds"] += time.perf_counter() - outside_started
        reproj_started = time.perf_counter()
        reprojection = evaluate_polyhedron_reprojection(
            name=f"current_candidate_exchange_{model_name}_{target_id}_{'_'.join(str(v) for v in sorted(removed))}",
            vertices_3d=np.asarray(rec.get("vertices") or [], dtype=float),
            contours=contours,
            direction_count=24,
            contour_sample=48,
        )
        timings["reprojection_seconds"] += time.perf_counter() - reproj_started
        preservation = preserved_patches(trial_active)
        gained = trial_canonical_ids - final_canonical_ids
        lost = final_canonical_ids - trial_canonical_ids
        target_active = bool(target_id in trial_active_ids and int(spec["target_face_id"]) in trial_canonical_ids)
        topology_valid = topology_is_valid(rec.get("topology") if isinstance(rec.get("topology"), dict) else None)
        outside_frac = finite_float(outside.get("cumulative_outside_fraction"), float("inf"))
        lost_z = int(value_or_default(outside.get("cumulative_lost_z_levels"), 1))
        reprojection_worse = bool(
            finite_float(reprojection.get("symmetric_contour_distance_p95"), 0.0)
            > 1.10 * max(EPS, finite_float(control.get("baseline_reprojection_p95"), 0.0))
            if control.get("baseline_reprojection_p95") is not None
            else False
        )
        if not topology_valid:
            classification = "G_topology_failure"
        elif outside_frac >= 0.01:
            classification = "H_outside_failure"
        elif lost_z != 0:
            classification = "I_lost_Z_failure"
        elif not bool(preservation.get("all_preserved")):
            classification = "F_breaks_protected_patch"
        elif not target_active:
            classification = "E_target_still_inactive"
        elif gained and not lost and len(trial_canonical_ids) > len(final_canonical_ids) and not reprojection_worse:
            classification = "B_safe_net_multi_gain" if len(gained) > 1 else "A_safe_net_plus_one"
        elif target_active and gained and lost and len(trial_canonical_ids) == len(final_canonical_ids):
            classification = "C_canonical_neutral_swap"
        elif target_active and lost:
            classification = "D_target_active_but_other_face_lost"
        else:
            classification = "J_equivalent_geometry_only"
        full_trials.append(
            {
                "trial_index": int(len(full_trials)),
                "kind": str(spec["kind"]),
                "target_face_id": int(spec["target_face_id"]),
                "target_candidate_id": int(target_id),
                "removed_candidate_ids": sorted(int(v) for v in removed),
                "target_active_final": bool(target_active),
                "removed_blockers_active_after": sorted(int(v) for v in (trial_active_ids & removed)),
                "active_candidate_ids_delta_gained": sorted(int(v) for v in (trial_active_ids - active_candidate_ids)),
                "active_candidate_ids_delta_lost": sorted(int(v) for v in (active_candidate_ids - trial_active_ids)),
                "canonical_ids_gained": sorted(int(v) for v in gained),
                "canonical_ids_lost": sorted(int(v) for v in lost),
                "canonical_unique": int(len(trial_canonical_ids)),
                "retained_original_ids": int(len(trial_canonical_ids & final_canonical_ids)),
                "classification": classification,
                "vertices": int(len(rec.get("vertices") or [])),
                "edges": int(len(rec.get("edges") or [])),
                "faces": int(len(rec.get("faces") or [])),
                "topology": rec.get("topology"),
                "topology_valid": bool(topology_valid),
                "volume": rec.get("reliable_volume"),
                "trusted_final_outside": outside,
                "max_per_z_outside": outside.get("cumulative_max_level_outside_fraction"),
                "lost_z": outside.get("cumulative_lost_z_levels"),
                "protected_patch_groups": preservation,
                "reprojection": reprojection,
                "geometry_digest": mesh_geometry_signature(trial_candidates, rec),
                "lp_proposal": {k: v for k, v in spec.get("lp_row", {}).items() if not str(k).startswith("_")},
                "observed_exchange_evidence": {
                    "target": nonoracle_candidate_public(target),
                    "removed_blockers": [nonoracle_candidate_public(candidate_by_id[v]) for v in sorted(removed) if v in candidate_by_id],
                    "target_surface_deficit": {
                        "before": patch_surface_deficit_for_samples(final_candidates, candidate_samples(target), sample_tol=0.01),
                        "after": patch_surface_deficit_for_samples(trial_candidates, candidate_samples(target), sample_tol=0.01),
                    },
                },
            }
        )
    timings["edge_clip_seconds"] = time.perf_counter() - full_started - timings["reprojection_seconds"] - timings["trusted_cloud_seconds"]

    safe_trials = [
        row for row in full_trials
        if str(row.get("classification")) in {"A_safe_net_plus_one", "B_safe_net_multi_gain"}
    ]
    collective_rows: list[dict[str, object]] = []
    state_candidates = list(final_candidates)
    state_ids = set(final_canonical_ids)
    state_removed: set[int] = set()
    for action in safe_trials[:5]:
        target = candidate_by_id.get(int(action["target_candidate_id"]))
        if target is None:
            continue
        state_removed |= {int(v) for v in action.get("removed_candidate_ids", [])}
        state_candidates = trial_candidates_for(state_removed, target)
        rec = reconstruct_polyhedron_from_halfspaces_edge_clip(state_candidates, **reconstruction_kwargs)
        ids = active_canonical_ids(state_candidates, rec)
        outside = evaluate_trusted_cloud_outside(state_candidates, trusted_points, z_slices, point_tol=float(point_tol))
        state_ids = ids
        collective_rows.append(
            {
                "step": int(len(collective_rows) + 1),
                "applied_target_candidate_id": int(action["target_candidate_id"]),
                "removed_candidate_ids": sorted(int(v) for v in state_removed),
                "gained_ids_vs_control": sorted(int(v) for v in (ids - final_canonical_ids)),
                "lost_ids_vs_control": sorted(int(v) for v in (final_canonical_ids - ids)),
                "total_unique": int(len(ids)),
                "area_weighted_recall": as_json_float(
                    sum(
                        polygon_area(np.asarray(face["vertices"], dtype=float))
                        for face in model_faces
                        if int(face["face_id"]) in ids
                    )
                    / max(
                        EPS,
                        sum(polygon_area(np.asarray(face["vertices"], dtype=float)) for face in model_faces),
                    )
                ),
                "vertices": int(len(rec.get("vertices") or [])),
                "edges": int(len(rec.get("edges") or [])),
                "faces": int(len(rec.get("faces") or [])),
                "topology": rec.get("topology"),
                "volume": rec.get("reliable_volume"),
                "trusted_final_outside": outside,
                "lost_z": outside.get("cumulative_lost_z_levels"),
            }
        )

    generic_selector = {
        "executed": False,
        "reason": "skipped: oracle-seeded safe net-gain exchanges below the required +3 total unique gate",
        "gate_safe_net_gain_ids": sorted(int(v) for row in safe_trials for v in row.get("canonical_ids_gained", [])),
        "policy_count": 0,
        "dependency_declaration": [
            {"field": "LP margin gain", "uses_initial_model": False, "uses_oracle_match": False, "uses_canonical_id": False, "available_when_oracle_disabled": True},
            {"field": "protected patch safety", "uses_initial_model": False, "uses_oracle_match": False, "uses_canonical_id": False, "available_when_oracle_disabled": True},
            {"field": "surface-deficit reduction", "uses_initial_model": False, "uses_oracle_match": False, "uses_canonical_id": False, "available_when_oracle_disabled": True},
            {"field": "target/blocker observed quality delta", "uses_initial_model": False, "uses_oracle_match": False, "uses_canonical_id": False, "available_when_oracle_disabled": True},
        ],
    }

    safe_gain_ids = {int(v) for row in safe_trials for v in row.get("canonical_ids_gained", [])}
    if any(str(row.get("classification")) == "A_safe_net_plus_one" for row in full_trials):
        branch = "A_safe_single_exchanges_exist"
    elif any(str(row.get("classification")) == "B_safe_net_multi_gain" for row in full_trials):
        branch = "B_strict_pair_or_multi_gain_exchanges_exist"
    elif any(str(row.get("classification")) == "C_canonical_neutral_swap" for row in full_trials):
        branch = "C_only_canonical_neutral_swaps"
    elif any(str(row.get("classification")) == "D_target_active_but_other_face_lost" for row in full_trials):
        branch = "D_blockers_exist_but_removal_loses_real_patches"
    elif any(str(row.get("classification")) == "F_breaks_protected_patch" for row in full_trials):
        branch = "D_blockers_exist_but_removal_loses_real_patches"
    elif any(str(row.get("classification")) == "E_target_still_inactive" for row in full_trials):
        branch = "E_targets_geometrically_redundant_after_exchange"
    elif safe_gain_ids and int(len(safe_gain_ids)) < 3:
        branch = "F_ceiling_exists_but_generic_selector_absent"
    else:
        branch = "E_targets_geometrically_redundant_after_exchange"

    public_targets = []
    for row in targets:
        public = {key: value for key, value in row.items() if not str(key).startswith("_")}
        public_targets.append(public)
    classification_counts = {
        key: int(sum(1 for row in full_trials if str(row.get("classification")) == key))
        for key in [
            "A_safe_net_plus_one",
            "B_safe_net_multi_gain",
            "C_canonical_neutral_swap",
            "D_target_active_but_other_face_lost",
            "E_target_still_inactive",
            "F_breaks_protected_patch",
            "G_topology_failure",
            "H_outside_failure",
            "I_lost_Z_failure",
            "J_equivalent_geometry_only",
        ]
    }
    timings["total_scope_seconds"] = time.perf_counter() - started
    return {
        "diagnostic_only": True,
        "production_changed": False,
        "production_mode_added": False,
        "scope": "current-candidate-exchange",
        "oracle_use": "target face cohort and posthoc canonical labels only; LP blocker ranking and full trial safety use current halfspaces, trusted cloud, protected patches, and observed candidate fields",
        "canonical_evaluator": canonical_oracle_evaluator_metadata(),
        "frozen_control": control,
        "target_cohort": public_targets,
        "target_blocker_lp_funnel": blocker_funnel_rows,
        "full_edge_clip_trials": {
            "max_trials": int(max_full_trials),
            "trial_count": int(len(full_trials)),
            "classification_counts": classification_counts,
            "rows": full_trials,
        },
        "collective_achieved_lower_bound": {
            "executed": bool(collective_rows),
            "max_actions": 5,
            "beam_width": 1,
            "rows": collective_rows,
            "achieved": collective_rows[-1] if collective_rows else {
                "total_unique": int(len(state_ids)),
                "gained_ids_vs_control": [],
                "lost_ids_vs_control": [],
            },
        },
        "generic_selector_controls": generic_selector,
        "selected_branch": branch,
        "acceptance": {
            "oracle_seeded_ceiling_new_ids": int(len(safe_gain_ids)),
            "generic_selector_required": bool(len(safe_gain_ids) >= 3),
            "production_promotion_allowed": False,
            "reason": "diagnostic scope only; no generic non-oracle selector was promoted",
        },
        "lp_engine_diagnostics": lp_engine.summary(),
        "timing_seconds": {key: as_json_float(float(value)) for key, value in timings.items()},
    }


def _auc_binary(values: list[tuple[float, bool]], *, higher_is_better: bool) -> float | None:
    clean = [(float(score), bool(label)) for score, label in values if np.isfinite(float(score))]
    pos = sum(1 for _, label in clean if label)
    neg = len(clean) - pos
    if pos == 0 or neg == 0:
        return None
    if not higher_is_better:
        clean = [(-score, label) for score, label in clean]
    wins = 0.0
    for p_score, p_label in clean:
        if not p_label:
            continue
        for n_score, n_label in clean:
            if n_label:
                continue
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return float(wins / max(pos * neg, EPS))


def _pr_auc_binary(values: list[tuple[float, bool]], *, higher_is_better: bool) -> float | None:
    clean = [(float(score), bool(label)) for score, label in values if np.isfinite(float(score))]
    positives = sum(1 for _, label in clean if label)
    if positives == 0:
        return None
    clean.sort(key=lambda item: item[0], reverse=higher_is_better)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, label in clean:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / max(tp + fp, 1)
        area += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return float(area)


def _view_interval_indices(center: int, width: int, count: int) -> np.ndarray:
    half = int(width) // 2
    return np.array([(int(center) + delta) % int(count) for delta in range(-half, int(width) - half)], dtype=int)


def _candidate_from_observed_points(points: np.ndarray, *, source: str) -> dict[str, object] | None:
    fit = fit_plane(points)
    if fit is None:
        return None
    u, v = plane_basis(fit.normal)
    local = np.column_stack([(points - fit.centroid) @ u, (points - fit.centroid) @ v])
    hull = points
    if local.shape[0] >= 3:
        hull_indices = convex_hull_indices(local)
        if len(hull_indices) >= 3:
            hull = points[np.array(hull_indices, dtype=int)]
    return {
        "track_id": -1,
        "candidate_origin": "oracle_raw_signal_diagnostic",
        "candidate_source": str(source),
        "plane_centroid": as_json_point(fit.centroid),
        "normal": as_json_point(fit.normal),
        "plane_rms": as_json_float(float(fit.rms)),
        "plane_max_abs": as_json_float(float(fit.max_abs)),
        "hull": [as_json_point(p) for p in hull],
        "hull_area": as_json_float(polygon_area(hull)),
        "support_points": int(points.shape[0]),
    }


def diagnose_view_conditioned_raw_support(
    *,
    model_name: str,
    vertices: np.ndarray,
    faces: list[list[int]],
    windows: list[int],
    z_levels: np.ndarray,
    line_points_by_window: np.ndarray | None,
    fit_rms_by_window: np.ndarray | None,
    cond_by_window: np.ndarray | None,
    point_bounds_min: np.ndarray,
    point_bounds_max: np.ndarray,
    peak_threshold: float,
    low_threshold: float,
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    model_faces: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    if line_points_by_window is None or fit_rms_by_window is None:
        return {
            "schema_version": 1,
            "model": str(model_name),
            "production_changed": False,
            "skipped": True,
            "reason": "multi-window raw line_points are not available in memory",
        }
    lp = np.asarray(line_points_by_window, dtype=float)
    rms = np.asarray(fit_rms_by_window, dtype=float)
    cond = np.asarray(cond_by_window, dtype=float) if cond_by_window is not None else None
    if lp.ndim != 4 or lp.shape[-1] != 3:
        return {
            "schema_version": 1,
            "model": str(model_name),
            "production_changed": False,
            "skipped": True,
            "reason": "unexpected line_points_by_window shape",
            "shape": list(lp.shape),
        }

    model_face_rows = model_faces or model_face_planes(vertices, faces)
    nw, n_view, n_z, _ = lp.shape
    flat_points = lp.reshape((-1, 3))
    flat_rms = rms.reshape((-1,)) if rms.shape[:3] == lp.shape[:3] else np.full(flat_points.shape[0], float("nan"))
    flat_cond = cond.reshape((-1,)) if cond is not None and cond.shape[:3] == lp.shape[:3] else np.full(flat_points.shape[0], float("nan"))
    finite = (
        np.all(np.isfinite(flat_points), axis=1)
        & np.all(flat_points >= point_bounds_min[None, :], axis=1)
        & np.all(flat_points <= point_bounds_max[None, :], axis=1)
    )
    finite_indices = np.flatnonzero(finite)
    finite_points = flat_points[finite_indices]
    bbox_diag = float(np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))) if vertices.size else 1.0
    z_step = float(np.median(np.diff(z_levels))) if z_levels.size > 1 else 0.01
    plane_tol = max(0.0125, 0.004 * bbox_diag, abs(z_step) * 1.25)
    finite_tol = max(0.025, 0.006 * bbox_diag, abs(z_step) * 1.5)
    fit_span_tol = max(0.03, 0.008 * bbox_diag)
    total_flat = int(flat_points.shape[0])
    assignment_counts = np.zeros(total_flat, dtype=np.uint16)
    best_distance = np.full(total_flat, float("inf"), dtype=float)
    best_face = np.full(total_flat, -1, dtype=np.int32)
    face_support_indices: dict[int, np.ndarray] = {}
    face_support_distances: dict[int, np.ndarray] = {}
    face_competing_min_distances: dict[int, np.ndarray] = {}

    face_infos: list[dict[str, object]] = []
    total_area = 0.0
    for face in model_face_rows:
        fid = int(face["face_id"])
        face_pts = np.array(face["vertices"], dtype=float)
        normal = np.array(face["normal"], dtype=float)
        origin = np.mean(face_pts, axis=0)
        u, v = plane_basis(normal)
        poly_2d = np.column_stack([(face_pts - origin) @ u, (face_pts - origin) @ v])
        poly_min = np.min(poly_2d, axis=0)
        poly_max = np.max(poly_2d, axis=0)
        area = polygon_area(face_pts)
        total_area += float(area)
        face_infos.append({
            "face_id": fid,
            "normal": normal,
            "origin": origin,
            "u": u,
            "v": v,
            "poly_2d": poly_2d,
            "poly_span": np.maximum(poly_max - poly_min, fit_span_tol),
            "area": float(area),
            "z_span": float(max(0.0, float(face["z_max"]) - float(face["z_min"]))),
            "z_mid": float(0.5 * (float(face["z_min"]) + float(face["z_max"]))),
            "normal_abs_z": abs(float(normal[2])),
            "bbox_min": np.min(face_pts, axis=0) - (plane_tol + finite_tol + abs(z_step)),
            "bbox_max": np.max(face_pts, axis=0) + (plane_tol + finite_tol + abs(z_step)),
        })

    for info in face_infos:
        fid = int(info["face_id"])
        if finite_indices.size == 0:
            face_support_indices[fid] = np.zeros(0, dtype=int)
            face_support_distances[fid] = np.zeros(0, dtype=float)
            continue
        bbox_min = np.array(info["bbox_min"], dtype=float)
        bbox_max = np.array(info["bbox_max"], dtype=float)
        bbox_mask = np.all((finite_points >= bbox_min[None, :]) & (finite_points <= bbox_max[None, :]), axis=1)
        local_finite = finite_indices[bbox_mask]
        if local_finite.size == 0:
            face_support_indices[fid] = np.zeros(0, dtype=int)
            face_support_distances[fid] = np.zeros(0, dtype=float)
            continue
        pts = flat_points[local_finite]
        normal = np.array(info["normal"], dtype=float)
        origin = np.array(info["origin"], dtype=float)
        residual = np.abs((pts - origin[None, :]) @ normal)
        plane_mask = residual <= plane_tol
        if not np.any(plane_mask):
            face_support_indices[fid] = np.zeros(0, dtype=int)
            face_support_distances[fid] = np.zeros(0, dtype=float)
            continue
        local_finite = local_finite[plane_mask]
        pts = pts[plane_mask]
        residual = residual[plane_mask]
        u = np.array(info["u"], dtype=float)
        v = np.array(info["v"], dtype=float)
        uv = np.column_stack([(pts - origin[None, :]) @ u, (pts - origin[None, :]) @ v])
        outside = np.maximum(0.0, polygon_signed_distances_2d(uv, np.array(info["poly_2d"], dtype=float)))
        support_mask = outside <= finite_tol
        support_indices = local_finite[support_mask]
        support_dist = np.sqrt(residual[support_mask] ** 2 + outside[support_mask] ** 2)
        face_support_indices[fid] = support_indices.astype(int)
        face_support_distances[fid] = support_dist.astype(float)
        assignment_counts[support_indices] = np.minimum(assignment_counts[support_indices] + 1, np.iinfo(np.uint16).max)
        better = support_dist < best_distance[support_indices]
        if np.any(better):
            update_idx = support_indices[better]
            best_distance[update_idx] = support_dist[better]
            best_face[update_idx] = int(fid)

    for info in face_infos:
        fid = int(info["face_id"])
        support = face_support_indices.get(fid, np.zeros(0, dtype=int))
        if support.size == 0:
            face_competing_min_distances[fid] = np.zeros(0, dtype=float)
            continue
        own = face_support_distances.get(fid, np.zeros(0, dtype=float))
        face_competing_min_distances[fid] = (best_distance[support] - own).astype(float)

    active_indices = [
        int(i)
        for i in final_reconstructed.get("face_candidate_indices", [])
        if 0 <= int(i) < len(final_candidates)
    ]
    active_candidates = [final_candidates[i] for i in active_indices]
    final_cohort = oracle_candidate_cohort(active_candidates, model_face_rows, active_plane_indices=active_indices)
    final_ids = set(int(v) for v in final_cohort.get("finite_face_ids", []))

    z_min = float(np.min(vertices[:, 2])) if vertices.size else 0.0
    z_max = float(np.max(vertices[:, 2])) if vertices.size else 1.0
    lower_cut = z_min + (z_max - z_min) / 3.0
    areas = np.array([float(info["area"]) for info in face_infos], dtype=float)
    small_area_cut = float(np.percentile(areas, 10)) if areas.size else 0.0
    categories = [
        "within_single_view",
        "across_Z_pooling",
        "across_view_pooling",
        "track_or_cluster_merge",
        "finite_footprint_overlap",
        "insufficient_points_or_span",
        "clean_fit_ready_support",
    ]
    category_counts = {name: 0 for name in categories}
    ceiling_ids: dict[str, set[int]] = {
        "full_raw_support_existence": set(),
        "best_single_view_finite_support": set(),
        "best_contiguous_view_interval_fit": set(),
        "cross_view_consensus_fit": set(),
        "final_active_mesh": set(final_ids),
    }
    rows: list[dict[str, object]] = []
    score_rows: list[tuple[float, bool]] = []
    feature_rows: list[dict[str, object]] = []

    def evaluate_fit(points: np.ndarray, source: str) -> dict[str, object]:
        candidate = _candidate_from_observed_points(points, source=source)
        if candidate is None:
            return {"fit_ready": False, "finite_good": False, "plane_good": False}
        match = candidate_oracle_face(candidate, model_face_rows)
        return {
            "fit_ready": True,
            "candidate": candidate,
            "match": match,
            "finite_good": bool((match or {}).get("finite_good")),
            "plane_good": bool((match or {}).get("plane_good")),
            "face_id": int((match or {}).get("face_id", -1)),
            "plane_rms": candidate.get("plane_rms"),
            "hull_area": candidate.get("hull_area"),
        }

    for info in face_infos:
        fid = int(info["face_id"])
        support = face_support_indices.get(fid, np.zeros(0, dtype=int))
        raw_count = int(support.size)
        if raw_count:
            ceiling_ids["full_raw_support_existence"].add(fid)
        wi = (support // (n_view * n_z)).astype(int) if raw_count else np.zeros(0, dtype=int)
        rem = (support % (n_view * n_z)).astype(int) if raw_count else np.zeros(0, dtype=int)
        view_idx = (rem // n_z).astype(int) if raw_count else np.zeros(0, dtype=int)
        z_idx = (rem % n_z).astype(int) if raw_count else np.zeros(0, dtype=int)
        pts = flat_points[support] if raw_count else np.zeros((0, 3), dtype=float)
        unique_views = sorted(int(v) for v in np.unique(view_idx)) if raw_count else []
        unique_z = sorted(int(v) for v in np.unique(z_idx)) if raw_count else []
        unique_windows = sorted(int(windows[int(v)]) for v in np.unique(wi)) if raw_count else []
        pure_mask = (best_face[support] == fid) & (assignment_counts[support] == 1) if raw_count else np.zeros(0, dtype=bool)
        purity = float(np.mean(pure_mask)) if raw_count else 0.0
        overlap_frac = float(np.mean(assignment_counts[support] > 1)) if raw_count else 0.0
        if raw_count:
            origin = np.array(info["origin"], dtype=float)
            u = np.array(info["u"], dtype=float)
            v = np.array(info["v"], dtype=float)
            uv = np.column_stack([(pts - origin[None, :]) @ u, (pts - origin[None, :]) @ v])
            span = np.ptp(uv, axis=0) if uv.shape[0] else np.zeros(2, dtype=float)
            norm_span = np.minimum(span / np.array(info["poly_span"], dtype=float), 1.0)
            coverage_2d = float(np.min(norm_span))
        else:
            uv = np.zeros((0, 2), dtype=float)
            norm_span = np.zeros(2, dtype=float)
            coverage_2d = 0.0
        z_span_obs = float(np.ptp(z_levels[z_idx])) if z_idx.size > 1 else 0.0
        angular_span = float(len(unique_views) / max(n_view, 1))
        raw_rms = flat_rms[support] if raw_count else np.zeros(0, dtype=float)
        raw_cond = flat_cond[support] if raw_count else np.zeros(0, dtype=float)
        minima_count = 0
        low_count = 0
        peak_count = 0
        if raw_count and rms.shape[:3] == lp.shape[:3]:
            prev = rms[wi, (view_idx - 1) % n_view, z_idx]
            cur = rms[wi, view_idx, z_idx]
            nxt = rms[wi, (view_idx + 1) % n_view, z_idx]
            minima_count = int(np.sum(np.isfinite(cur) & (cur <= prev) & (cur <= nxt)))
            low_count = int(np.sum(np.isfinite(cur) & (cur <= float(low_threshold))))
            peak_count = int(np.sum(np.isfinite(cur) & (cur >= float(peak_threshold))))

        best_view: dict[str, object] = {"view_index": None, "point_count": 0, "purity": None, "coverage_2d": None}
        contiguous_fits: dict[str, dict[str, object]] = {}
        if raw_count:
            view_counts = np.bincount(view_idx, minlength=n_view)
            for width in (1, 3, 5):
                best_interval: dict[str, object] | None = None
                best_points = np.zeros((0, 3), dtype=float)
                for candidate_center in np.flatnonzero(view_counts > 0):
                    interval_views = _view_interval_indices(int(candidate_center), int(width), int(n_view))
                    mask = np.isin(view_idx, interval_views)
                    count = int(np.sum(mask))
                    if count == 0:
                        continue
                    interval_pure = float(np.mean(pure_mask[mask])) if count else 0.0
                    interval_pts = pts[mask]
                    interval_uv = uv[mask] if uv.shape[0] else np.zeros((0, 2), dtype=float)
                    interval_span = np.ptp(interval_uv, axis=0) if interval_uv.shape[0] else np.zeros(2, dtype=float)
                    interval_cov = float(np.min(np.minimum(interval_span / np.array(info["poly_span"], dtype=float), 1.0)))
                    key = (interval_pure >= 0.65, interval_cov, count)
                    old_key = (
                        bool(best_interval and float(best_interval.get("purity") or 0.0) >= 0.65),
                        float((best_interval or {}).get("coverage_2d") or 0.0),
                        int((best_interval or {}).get("point_count") or 0),
                    )
                    if best_interval is None or key > old_key:
                        best_interval = {
                            "center_view": int(candidate_center),
                            "width": int(width),
                            "point_count": count,
                            "purity": as_json_float(interval_pure),
                            "coverage_2d": as_json_float(interval_cov),
                            "z_levels": int(len(set(int(v) for v in z_idx[mask]))),
                        }
                        best_points = interval_pts
                if best_interval is None:
                    best_interval = {"center_view": None, "width": int(width), "point_count": 0, "purity": None, "coverage_2d": None, "z_levels": 0}
                fit_eval = evaluate_fit(best_points, f"view_interval_{width}") if best_points.shape[0] >= 4 else {"fit_ready": False, "finite_good": False, "plane_good": False}
                best_interval["fit_ready"] = bool(fit_eval.get("fit_ready"))
                best_interval["finite_good"] = bool(fit_eval.get("finite_good") and int(fit_eval.get("face_id", -1)) == fid)
                best_interval["plane_good"] = bool(fit_eval.get("plane_good") and int(fit_eval.get("face_id", -1)) == fid)
                best_interval["plane_rms"] = fit_eval.get("plane_rms")
                contiguous_fits[str(width)] = best_interval
                if width == 1:
                    best_view = dict(best_interval)
            if bool(contiguous_fits.get("1", {}).get("finite_good")):
                ceiling_ids["best_single_view_finite_support"].add(fid)
            if bool(contiguous_fits.get("3", {}).get("finite_good")) or bool(contiguous_fits.get("5", {}).get("finite_good")):
                ceiling_ids["best_contiguous_view_interval_fit"].add(fid)

        consensus_good = False
        consensus_plane_variation = None
        per_view_candidates: list[dict[str, object]] = []
        if raw_count:
            for view in unique_views:
                mask = view_idx == int(view)
                if int(np.sum(mask)) < 4 or float(np.mean(pure_mask[mask])) < 0.65:
                    continue
                candidate = _candidate_from_observed_points(pts[mask], source="per_view_consensus")
                if candidate is not None:
                    match = candidate_oracle_face(candidate, model_face_rows)
                    if bool((match or {}).get("plane_good")) and int((match or {}).get("face_id", -1)) == fid:
                        per_view_candidates.append(candidate)
            if len(per_view_candidates) >= 2:
                normals = np.array([np.array(c["normal"], dtype=float) for c in per_view_candidates], dtype=float)
                ref = normals[0]
                normals = np.array([n if float(n @ ref) >= 0.0 else -n for n in normals], dtype=float)
                mean_n = np.mean(normals, axis=0)
                mean_n /= max(float(np.linalg.norm(mean_n)), EPS)
                offsets = np.array([
                    float(np.array(c["normal"], dtype=float) @ np.array(c["plane_centroid"], dtype=float))
                    for c in per_view_candidates
                ], dtype=float)
                dots = np.clip(np.abs(normals @ mean_n), -1.0, 1.0)
                angle_p95 = float(np.percentile(np.degrees(np.arccos(dots)), 95))
                offset_span = float(np.ptp(offsets)) if offsets.size else float("inf")
                consensus_plane_variation = max(angle_p95 / 2.0, offset_span / max(plane_tol, EPS))
                if angle_p95 <= 2.0 and offset_span <= plane_tol:
                    consensus_points = pts[pure_mask] if int(np.sum(pure_mask)) >= 4 else pts
                    consensus_eval = evaluate_fit(consensus_points, "cross_view_consensus")
                    consensus_good = bool(consensus_eval.get("finite_good") and int(consensus_eval.get("face_id", -1)) == fid)
            if consensus_good:
                ceiling_ids["cross_view_consensus_fit"].add(fid)

        fit_ready = bool(consensus_good or contiguous_fits.get("3", {}).get("finite_good") or contiguous_fits.get("5", {}).get("finite_good"))
        full_mixed = bool(raw_count > 0 and purity < 0.65)
        best_single_clean = bool((best_view.get("purity") or 0.0) >= 0.75 and (best_view.get("coverage_2d") or 0.0) >= 0.08)
        if raw_count < 4 or coverage_2d < 0.04:
            category = "insufficient_points_or_span"
        elif overlap_frac >= 0.35:
            category = "finite_footprint_overlap"
        elif full_mixed and not best_single_clean:
            category = "within_single_view"
        elif full_mixed and best_single_clean and len(unique_views) > 1:
            category = "across_view_pooling"
        elif full_mixed and z_span_obs > max(3.0 * abs(z_step), 0.03):
            category = "across_Z_pooling"
        elif fit_ready and fid not in final_ids:
            category = "track_or_cluster_merge"
        elif fit_ready:
            category = "clean_fit_ready_support"
        else:
            category = "insufficient_points_or_span"
        category_counts[category] += 1

        condition_penalty = min(float(np.nanmedian(raw_cond)) / 1e6, 1.0) if raw_cond.size and np.any(np.isfinite(raw_cond)) else 0.0
        score = (
            min(len(unique_views), 8) / 8.0
            + 1.5 * min(coverage_2d, 1.0)
            + 0.5 * min(raw_count / max(float(nw * n_z), 1.0), 1.0)
            - 1.0 * max(0.0, 1.0 - purity)
            - 0.5 * condition_penalty
        )
        label = bool(consensus_good or contiguous_fits.get("3", {}).get("finite_good") or contiguous_fits.get("5", {}).get("finite_good"))
        score_rows.append((float(score), label))
        feature_rows.append({
            "score": float(score),
            "label": label,
            "raw_count": raw_count,
            "independent_views": len(unique_views),
            "coverage_2d": coverage_2d,
            "purity": purity,
        })
        competing_margin = face_competing_min_distances.get(fid, np.zeros(0, dtype=float))
        rows.append({
            "face_id": fid,
            "raw_count": raw_count,
            "windows": unique_windows,
            "z_levels": int(len(unique_z)),
            "z_span": as_json_float(z_span_obs),
            "view_count": int(len(unique_views)),
            "angular_span_fraction": as_json_float(angular_span),
            "purity": as_json_float(purity),
            "overlap_fraction": as_json_float(overlap_frac),
            "coverage_u": as_json_float(float(norm_span[0])),
            "coverage_v": as_json_float(float(norm_span[1])),
            "coverage_2d": as_json_float(coverage_2d),
            "rms_median": as_json_float(float(np.nanmedian(raw_rms))) if raw_rms.size and np.any(np.isfinite(raw_rms)) else None,
            "rms_p95": as_json_float(float(np.nanpercentile(raw_rms[np.isfinite(raw_rms)], 95))) if raw_rms.size and np.any(np.isfinite(raw_rms)) else None,
            "condition_median": as_json_float(float(np.nanmedian(raw_cond))) if raw_cond.size and np.any(np.isfinite(raw_cond)) else None,
            "condition_p95": as_json_float(float(np.nanpercentile(raw_cond[np.isfinite(raw_cond)], 95))) if raw_cond.size and np.any(np.isfinite(raw_cond)) else None,
            "classification_counts": {"minimum": minima_count, "low": low_count, "peak": peak_count},
            "best_single_view": best_view,
            "contiguous_view_intervals": contiguous_fits,
            "cross_view_consensus": {
                "per_view_plane_good_hypotheses": int(len(per_view_candidates)),
                "finite_good": bool(consensus_good),
                "plane_variation_score": as_json_float(float(consensus_plane_variation)) if consensus_plane_variation is not None else None,
            },
            "competing_face_margin_median": as_json_float(float(np.median(competing_margin))) if competing_margin.size else None,
            "category": category,
            "final_active_finite_good": bool(fid in final_ids),
            "area": as_json_float(float(info["area"])),
            "area_fraction": as_json_float(float(info["area"]) / max(total_area, EPS)),
            "reference_z_span": as_json_float(float(info["z_span"])),
            "normal_abs_z": as_json_float(float(info["normal_abs_z"])),
            "near_vertical": bool(float(info["normal_abs_z"]) < 0.25),
            "lower_z": bool(float(info["z_mid"]) <= lower_cut),
            "small_area_decile": bool(float(info["area"]) <= small_area_cut),
            "non_oracle_score_fixed_direction": as_json_float(float(score)),
        })

    def ceiling_summary(ids: set[int]) -> dict[str, object]:
        selected = [row for row in rows if int(row["face_id"]) in ids]
        return {
            "unique_face_ids": int(len(ids)),
            "count_recall": as_json_float(len(ids) / max(len(face_infos), 1)),
            "area_weighted_recall": as_json_float(sum(float(row.get("area") or 0.0) for row in selected) / max(total_area, EPS)),
            "near_vertical_recall": as_json_float(sum(1 for row in selected if bool(row["near_vertical"])) / max(sum(1 for row in rows if bool(row["near_vertical"])), 1)),
            "lower_z_recall": as_json_float(sum(1 for row in selected if bool(row["lower_z"])) / max(sum(1 for row in rows if bool(row["lower_z"])), 1)),
            "small_area_decile_recall": as_json_float(sum(1 for row in selected if bool(row["small_area_decile"])) / max(sum(1 for row in rows if bool(row["small_area_decile"])), 1)),
            "face_ids_sample": sorted(int(v) for v in ids)[:80],
        }

    digest = hashlib.sha256()
    digest.update(str(model_name).encode("utf-8"))
    digest.update(np.array(windows, dtype=np.int64).tobytes())
    digest.update(np.array(lp.shape, dtype=np.int64).tobytes())
    finite_rms = flat_rms[np.isfinite(flat_rms)]
    digest.update(np.array([
        float(np.median(finite_rms)) if finite_rms.size else float("nan"),
        float(np.percentile(finite_rms, 95)) if finite_rms.size else float("nan"),
        float(finite_indices.size),
    ], dtype=np.float64).tobytes())

    sorted_features = sorted(feature_rows, key=lambda row: float(row["score"]), reverse=True)
    frontier = []
    for top_k in (20, 50):
        subset = sorted_features[: min(top_k, len(sorted_features))]
        if subset:
            frontier.append({
                "top_k": int(top_k),
                "precision": as_json_float(sum(1 for row in subset if bool(row["label"])) / len(subset)),
                "unique_face_ids": int(sum(1 for row in subset if bool(row["label"]))),
            })
    recoverable_ids = ceiling_ids["cross_view_consensus_fit"] | ceiling_ids["best_contiguous_view_interval_fit"]
    shortlist_precision = None
    if sorted_features:
        shortlist = sorted_features[: min(20, len(sorted_features))]
        shortlist_precision = sum(1 for row in shortlist if bool(row["label"])) / max(len(shortlist), 1)
    dominant_category = max(category_counts.items(), key=lambda item: item[1])[0] if category_counts else "insufficient_points_or_span"
    selected_branch = "E" if not recoverable_ids else "D"
    if dominant_category == "across_view_pooling":
        selected_branch = "A"
    elif dominant_category == "within_single_view":
        selected_branch = "B"
    elif dominant_category == "clean_fit_ready_support" and len(recoverable_ids - final_ids) == 0:
        selected_branch = "C"

    sample_rows = sorted(
        rows,
        key=lambda row: (
            not bool(row["small_area_decile"]),
            not bool(row["near_vertical"]),
            bool(row["final_active_finite_good"]),
            -int(row["raw_count"]),
        ),
    )[:80]
    roc_auc = _auc_binary(score_rows, higher_is_better=True)
    pr_auc = _pr_auc_binary(score_rows, higher_is_better=True)
    return {
        "schema_version": 1,
        "model": str(model_name),
        "scope": "view-conditioned-raw-support",
        "production_changed": False,
        "initial_model_usage": "posthoc labels/evaluation only; no production candidate score/order/gate uses InitialModel",
        "raw_provenance": {
            "windows": [int(v) for v in windows],
            "line_points_shape": [int(v) for v in lp.shape],
            "z_level_count": int(n_z),
            "view_count": int(n_view),
            "finite_raw_observations": int(finite_indices.size),
            "total_raw_slots": int(total_flat),
            "view_angle_model": "cyclic half-contour index mapped to 2*pi*i/view_count",
            "raw_signal_signature": digest.hexdigest()[:24],
            "line_direction_provenance": "not serialized by compute_line_points_multi_window; W2 segment direction remains available only after segmentation",
        },
        "tolerances": {
            "plane_distance": as_json_float(float(plane_tol)),
            "finite_polygon_distance": as_json_float(float(finite_tol)),
            "fit_span_tol": as_json_float(float(fit_span_tol)),
        },
        "mixing_category_counts": category_counts,
        "mixing_category_total": int(sum(category_counts.values())),
        "ceiling": {name: ceiling_summary(ids) for name, ids in ceiling_ids.items()},
        "representation_comparison": {
            "current_aggregation": ceiling_summary(ceiling_ids["final_active_mesh"]),
            "view_conditioned_tls": ceiling_summary(ceiling_ids["best_contiguous_view_interval_fit"]),
            "per_view_hypotheses_plane_consensus": ceiling_summary(ceiling_ids["cross_view_consensus_fit"]),
            "view_conditioned_mixed_split": {
                "skipped": True,
                "reason": "diagnostic split is evaluated only through separable view-conditioned modes; no oracle IDs are used for a production split score",
            },
        },
        "non_oracle_separation": {
            "fixed_direction_score": {
                "higher_is_better": True,
                "formula": "independent views + finite 2D span + saturated support - impurity - bounded condition penalty",
                "roc_auc": as_json_float(float(roc_auc)) if roc_auc is not None else None,
                "pr_auc": as_json_float(float(pr_auc)) if pr_auc is not None else None,
                "frontier": frontier,
                "shortlist_precision_top20": as_json_float(float(shortlist_precision)) if shortlist_precision is not None else None,
            },
            "feature_distributions": {
                "good": {
                    "support_count": distribution_summary([row["raw_count"] for row in rows if int(row["face_id"]) in recoverable_ids]),
                    "independent_views": distribution_summary([row["view_count"] for row in rows if int(row["face_id"]) in recoverable_ids]),
                    "coverage_2d": distribution_summary([row["coverage_2d"] for row in rows if int(row["face_id"]) in recoverable_ids]),
                    "purity": distribution_summary([row["purity"] for row in rows if int(row["face_id"]) in recoverable_ids]),
                },
                "bad": {
                    "support_count": distribution_summary([row["raw_count"] for row in rows if int(row["face_id"]) not in recoverable_ids]),
                    "independent_views": distribution_summary([row["view_count"] for row in rows if int(row["face_id"]) not in recoverable_ids]),
                    "coverage_2d": distribution_summary([row["coverage_2d"] for row in rows if int(row["face_id"]) not in recoverable_ids]),
                    "purity": distribution_summary([row["purity"] for row in rows if int(row["face_id"]) not in recoverable_ids]),
                },
            },
        },
        "full_edge_clip_trials": {
            "attempted": False,
            "reason": "no production-ready non-oracle shortlist is promoted by the diagnostic-only ceiling; candidate-level recovery is not counted as mesh recovery",
            "max_trials_per_model": 20,
        },
        "branch_decision": {
            "selected": selected_branch,
            "legend": {
                "A": "mixing mainly across views; view-conditioned detector is promising",
                "B": "mixing already inside single view; raw detector must change first",
                "C": "clean support exists but plane/hull formation is the likely bottleneck",
                "D": "clean hypotheses exist but non-oracle separation is insufficient",
                "E": "full raw signal still lacks fit-ready support",
            },
            "production_mode_added": False,
        },
        "bounded_face_rows_sample": sample_rows,
        "timing_seconds": as_json_float(time.perf_counter() - started),
    }


def _edge_segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ab = b - a
    den = float(ab @ ab)
    if den <= EPS:
        return np.linalg.norm(points - a[None, :], axis=1), np.zeros(points.shape[0], dtype=float)
    t = ((points - a[None, :]) @ ab) / den
    closest = a[None, :] + np.clip(t, 0.0, 1.0)[:, None] * ab[None, :]
    return np.linalg.norm(points - closest, axis=1), t.astype(float)


def _line_candidate_from_points(points: np.ndarray, *, source: str) -> dict[str, object] | None:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 2:
        return None
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid[None, :]
    try:
        _, s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = vh[0].astype(float)
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        return None
    direction /= norm
    coord = centered @ direction
    residual = np.linalg.norm(centered - coord[:, None] * direction[None, :], axis=1)
    return {
        "source": str(source),
        "point_count": int(pts.shape[0]),
        "centroid": as_json_point(centroid),
        "direction": as_json_point(direction),
        "rms": as_json_float(float(np.sqrt(np.mean(residual * residual)))),
        "p95": as_json_float(float(np.percentile(residual, 95))),
        "segment_min": as_json_point(centroid + float(np.min(coord)) * direction),
        "segment_max": as_json_point(centroid + float(np.max(coord)) * direction),
        "segment_span": as_json_float(float(np.max(coord) - np.min(coord))),
        "condition": as_json_float(float(s[0] / max(s[1], EPS))) if s.size > 1 else None,
    }


def diagnose_edge_incidence_raw_support(
    *,
    model_name: str,
    vertices: np.ndarray,
    faces: list[list[int]],
    windows: list[int],
    z_levels: np.ndarray,
    line_points_by_window: np.ndarray | None,
    fit_rms_by_window: np.ndarray | None,
    cond_by_window: np.ndarray | None,
    point_bounds_min: np.ndarray,
    point_bounds_max: np.ndarray,
    peak_threshold: float,
    low_threshold: float,
    final_candidates: list[dict[str, object]],
    final_reconstructed: dict[str, object],
    w2_segments: list[dict[str, object]],
    w2_clusters: list[dict[str, object]],
    model_faces: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    if line_points_by_window is None or fit_rms_by_window is None:
        return {
            "schema_version": 1,
            "model": str(model_name),
            "production_changed": False,
            "skipped": True,
            "reason": "multi-window raw line_points are not available in memory",
        }
    lp = np.asarray(line_points_by_window, dtype=float)
    rms = np.asarray(fit_rms_by_window, dtype=float)
    cond = np.asarray(cond_by_window, dtype=float) if cond_by_window is not None else None
    if lp.ndim != 4 or lp.shape[-1] != 3:
        return {
            "schema_version": 1,
            "model": str(model_name),
            "production_changed": False,
            "skipped": True,
            "reason": "unexpected line_points_by_window shape",
            "shape": list(lp.shape),
        }

    model_face_rows = model_faces or model_face_planes(vertices, faces)
    reference_edges, edge_faces_map = model_edges_from_faces(vertices, faces)
    face_to_edges: dict[int, list[int]] = {}
    for eid, edge in enumerate(reference_edges):
        for fid in edge_faces_map.get(edge, set()):
            face_to_edges.setdefault(int(fid), []).append(int(eid))

    nw, n_view, n_z, _ = lp.shape
    flat_points = lp.reshape((-1, 3))
    flat_rms = rms.reshape((-1,)) if rms.shape[:3] == lp.shape[:3] else np.full(flat_points.shape[0], float("nan"))
    flat_cond = cond.reshape((-1,)) if cond is not None and cond.shape[:3] == lp.shape[:3] else np.full(flat_points.shape[0], float("nan"))
    finite = (
        np.all(np.isfinite(flat_points), axis=1)
        & np.all(flat_points >= point_bounds_min[None, :], axis=1)
        & np.all(flat_points <= point_bounds_max[None, :], axis=1)
    )
    finite_indices = np.flatnonzero(finite)
    finite_points = flat_points[finite_indices]
    bbox_diag = float(np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))) if vertices.size else 1.0
    z_step = float(np.median(np.diff(z_levels))) if z_levels.size > 1 else 0.01
    edge_tol = max(0.02, 0.004 * bbox_diag, abs(z_step) * 1.5)
    line_residual_tol = max(0.035, 0.006 * bbox_diag, abs(z_step) * 2.0)
    total_flat = int(flat_points.shape[0])
    edge_assignment_counts = np.zeros(total_flat, dtype=np.uint16)
    best_distance = np.full(total_flat, float("inf"), dtype=float)
    second_distance = np.full(total_flat, float("inf"), dtype=float)
    best_edge = np.full(total_flat, -1, dtype=np.int32)
    edge_support_indices: dict[int, np.ndarray] = {}
    edge_support_t: dict[int, np.ndarray] = {}

    z_min = float(np.min(vertices[:, 2])) if vertices.size else 0.0
    z_max = float(np.max(vertices[:, 2])) if vertices.size else 1.0
    lower_cut = z_min + (z_max - z_min) / 3.0
    upper_cut = z_min + 2.0 * (z_max - z_min) / 3.0
    edge_infos: list[dict[str, object]] = []
    for eid, (ia, ib) in enumerate(reference_edges):
        a = vertices[int(ia)]
        b = vertices[int(ib)]
        vec = b - a
        length = float(np.linalg.norm(vec))
        direction = vec / max(length, EPS)
        adjacent = sorted(int(fid) for fid in edge_faces_map.get((int(ia), int(ib)), edge_faces_map.get((int(ib), int(ia)), set())))
        normals = [np.array(model_face_rows[fid]["normal"], dtype=float) for fid in adjacent if 0 <= fid < len(model_face_rows)]
        if len(normals) >= 2:
            dihedral = float(np.degrees(np.arccos(np.clip(float(normals[0] @ normals[1]), -1.0, 1.0))))
        else:
            dihedral = None
        z_mid = float(0.5 * (a[2] + b[2]))
        zone = "lower" if z_mid <= lower_cut else ("upper" if z_mid >= upper_cut else "middle")
        edge_infos.append({
            "edge_id": int(eid),
            "vertex_ids": [int(ia), int(ib)],
            "a": a,
            "b": b,
            "length": length,
            "direction": direction,
            "z_span": float(abs(float(a[2]) - float(b[2]))),
            "z_mid": z_mid,
            "z_zone": zone,
            "adjacent_faces": adjacent,
            "dihedral_angle": dihedral,
            "bbox_min": np.minimum(a, b) - edge_tol,
            "bbox_max": np.maximum(a, b) + edge_tol,
        })

    for info in edge_infos:
        eid = int(info["edge_id"])
        if finite_indices.size == 0:
            edge_support_indices[eid] = np.zeros(0, dtype=int)
            edge_support_t[eid] = np.zeros(0, dtype=float)
            continue
        bbox_mask = np.all(
            (finite_points >= np.array(info["bbox_min"], dtype=float)[None, :])
            & (finite_points <= np.array(info["bbox_max"], dtype=float)[None, :]),
            axis=1,
        )
        local = finite_indices[bbox_mask]
        if local.size == 0:
            edge_support_indices[eid] = np.zeros(0, dtype=int)
            edge_support_t[eid] = np.zeros(0, dtype=float)
            continue
        distances, t = _edge_segment_distances(flat_points[local], np.array(info["a"], dtype=float), np.array(info["b"], dtype=float))
        mask = (distances <= edge_tol) & (t >= -0.08) & (t <= 1.08)
        support = local[mask]
        support_dist = distances[mask]
        support_t = t[mask]
        edge_support_indices[eid] = support.astype(int)
        edge_support_t[eid] = support_t.astype(float)
        edge_assignment_counts[support] = np.minimum(edge_assignment_counts[support] + 1, np.iinfo(np.uint16).max)
        better = support_dist < best_distance[support]
        if np.any(better):
            update = support[better]
            second_distance[update] = best_distance[update]
            best_distance[update] = support_dist[better]
            best_edge[update] = eid
        worse = ~better
        if np.any(worse):
            update = support[worse]
            second_distance[update] = np.minimum(second_distance[update], support_dist[worse])

    active_indices = [
        int(i)
        for i in final_reconstructed.get("face_candidate_indices", [])
        if 0 <= int(i) < len(final_candidates)
    ]
    active_candidates = [final_candidates[i] for i in active_indices]
    pool_cohort = oracle_candidate_cohort(final_candidates, model_face_rows)
    active_cohort = oracle_candidate_cohort(active_candidates, model_face_rows, active_plane_indices=active_indices)
    pool_face_ids = set(int(v) for v in pool_cohort.get("finite_face_ids", []))
    active_face_ids = set(int(v) for v in active_cohort.get("finite_face_ids", []))

    def adjacent_has(edge_id: int, face_ids: set[int]) -> bool:
        return any(int(fid) in face_ids for fid in edge_infos[int(edge_id)]["adjacent_faces"])

    cluster_edge_matches: dict[int, list[int]] = {int(info["edge_id"]): [] for info in edge_infos}
    segment_edge_matches: dict[int, list[int]] = {int(info["edge_id"]): [] for info in edge_infos}
    duplicate_cluster_edges: set[int] = set()
    for segment in w2_segments:
        z0 = finite_float(segment.get("z_min"), float("nan"))
        z1 = finite_float(segment.get("z_max"), float("nan"))
        if not np.isfinite(z0) or not np.isfinite(z1):
            continue
        z = np.linspace(z0, z1, 5)
        pts = predict_w2_line(segment, z)
        direction = np.array(segment.get("direction") or [], dtype=float)
        best: tuple[float, int] | None = None
        for info in edge_infos:
            distances, t = _edge_segment_distances(pts, np.array(info["a"], dtype=float), np.array(info["b"], dtype=float))
            finite_frac = float(np.mean((t >= -0.05) & (t <= 1.05)))
            if finite_frac < 0.6:
                continue
            angle = 0.0
            if direction.shape == (3,) and np.linalg.norm(direction) > EPS:
                angle = float(np.degrees(np.arccos(np.clip(abs(float(direction @ np.array(info["direction"], dtype=float))), -1.0, 1.0))))
            score = float(np.median(distances)) + 0.01 * angle + max(0.0, 0.8 - finite_frac)
            if best is None or score < best[0]:
                best = (score, int(info["edge_id"]))
        if best is not None and best[0] <= max(edge_tol * 2.0, 0.08):
            segment_edge_matches[best[1]].append(int(segment.get("segment_id", -1)))
    for cluster in w2_clusters:
        z0 = finite_float(cluster.get("cluster_z_min"), finite_float(cluster.get("z_min"), float("nan")))
        z1 = finite_float(cluster.get("cluster_z_max"), finite_float(cluster.get("z_max"), float("nan")))
        if not np.isfinite(z0) or not np.isfinite(z1):
            continue
        z = np.linspace(z0, z1, 5)
        pts = predict_w2_line(cluster, z)
        direction = np.array(cluster.get("direction") or [], dtype=float)
        best: tuple[float, int] | None = None
        for info in edge_infos:
            distances, t = _edge_segment_distances(pts, np.array(info["a"], dtype=float), np.array(info["b"], dtype=float))
            finite_frac = float(np.mean((t >= -0.05) & (t <= 1.05)))
            if finite_frac < 0.6:
                continue
            angle = 0.0
            if direction.shape == (3,) and np.linalg.norm(direction) > EPS:
                angle = float(np.degrees(np.arccos(np.clip(abs(float(direction @ np.array(info["direction"], dtype=float))), -1.0, 1.0))))
            score = float(np.median(distances)) + 0.01 * angle + max(0.0, 0.8 - finite_frac)
            if best is None or score < best[0]:
                best = (score, int(info["edge_id"]))
        if best is not None and best[0] <= max(edge_tol * 2.0, 0.08):
            cid = int(cluster.get("edge_cluster_id", cluster.get("cluster_id", -1)))
            cluster_edge_matches[best[1]].append(cid)
    for eid, clusters_for_edge in cluster_edge_matches.items():
        if len(set(clusters_for_edge)) > 1:
            duplicate_cluster_edges.add(int(eid))

    edge_categories = [
        "no_raw_observed_support",
        "raw_support_ambiguous",
        "too_few_independent_points",
        "short_Z_but_cross_view_support_exists",
        "fixed_index_fragmentation",
        "segment_fit_rejected",
        "mixed_multiple_physical_edges",
        "edge_cluster_missing",
        "duplicate_clusters_same_edge",
        "confident_edge_cluster_available",
        "represented_in_final_face_candidate",
        "represented_in_final_active_mesh",
    ]
    category_counts = {name: 0 for name in edge_categories}
    edge_rows: list[dict[str, object]] = []
    score_rows: list[tuple[float, bool]] = []
    length_values = np.array([float(info["length"]) for info in edge_infos], dtype=float)
    zspan_values = np.array([float(info["z_span"]) for info in edge_infos], dtype=float)
    dihedral_values = np.array([
        finite_float(info.get("dihedral_angle"), float("nan")) for info in edge_infos
    ], dtype=float)

    for info in edge_infos:
        eid = int(info["edge_id"])
        support = edge_support_indices.get(eid, np.zeros(0, dtype=int))
        raw_count = int(support.size)
        wi = (support // (n_view * n_z)).astype(int) if raw_count else np.zeros(0, dtype=int)
        rem = (support % (n_view * n_z)).astype(int) if raw_count else np.zeros(0, dtype=int)
        view_idx = (rem // n_z).astype(int) if raw_count else np.zeros(0, dtype=int)
        z_idx = (rem % n_z).astype(int) if raw_count else np.zeros(0, dtype=int)
        pts = flat_points[support] if raw_count else np.zeros((0, 3), dtype=float)
        unique_views = sorted(int(v) for v in np.unique(view_idx)) if raw_count else []
        unique_z = sorted(int(v) for v in np.unique(z_idx)) if raw_count else []
        unique_indices = sorted(int(v) for v in np.unique(rem // n_z)) if raw_count else []
        ambiguity = float(np.mean(edge_assignment_counts[support] > 1)) if raw_count else 0.0
        margins = second_distance[support] - best_distance[support] if raw_count else np.zeros(0, dtype=float)
        margin_median = float(np.nanmedian(margins)) if margins.size else None
        t_values = edge_support_t.get(eid, np.zeros(0, dtype=float))
        t_span = float(np.max(np.clip(t_values, 0.0, 1.0)) - np.min(np.clip(t_values, 0.0, 1.0))) if t_values.size else 0.0
        z_span_obs = float(np.ptp(z_levels[z_idx])) if z_idx.size > 1 else 0.0
        line_eval = _line_candidate_from_points(pts, source="raw_edge_support") if pts.shape[0] >= 2 else None
        line_rms = finite_float((line_eval or {}).get("rms"), float("inf")) if line_eval else float("inf")
        condition_median = (
            float(np.nanmedian(flat_cond[support]))
            if raw_count and np.any(np.isfinite(flat_cond[support]))
            else None
        )
        cross_view_short = bool(len(unique_views) >= 3 and raw_count >= 6 and t_span >= 0.15 and float(info["z_span"]) <= max(3.0 * abs(z_step), 0.03))
        segment_available = bool(segment_edge_matches.get(eid))
        cluster_available = bool(cluster_edge_matches.get(eid))
        in_pool = adjacent_has(eid, pool_face_ids)
        in_active = adjacent_has(eid, active_face_ids)
        if in_active:
            category = "represented_in_final_active_mesh"
        elif in_pool:
            category = "represented_in_final_face_candidate"
        elif cluster_available and eid in duplicate_cluster_edges:
            category = "duplicate_clusters_same_edge"
        elif cluster_available:
            category = "confident_edge_cluster_available"
        elif segment_available:
            category = "edge_cluster_missing"
        elif raw_count == 0:
            category = "no_raw_observed_support"
        elif ambiguity >= 0.5:
            category = "raw_support_ambiguous"
        elif raw_count < 4 or len(unique_views) < 2 or t_span < 0.08:
            category = "too_few_independent_points"
        elif cross_view_short:
            category = "short_Z_but_cross_view_support_exists"
        elif len(unique_indices) > max(6, len(unique_views) * 2) and len(unique_z) >= 3:
            category = "fixed_index_fragmentation"
        elif line_rms > line_residual_tol:
            category = "segment_fit_rejected"
        else:
            category = "mixed_multiple_physical_edges"
        category_counts[category] += 1
        score_components = {
            "independent_view_score": float(min(len(unique_views), 8) / 8.0),
            "finite_segment_span_score": float(min(t_span, 1.0)),
            "saturated_support_score": float(min(raw_count / 80.0, 1.0)),
            "second_edge_margin_score": float(min(max(0.0, margin_median or 0.0) / max(edge_tol, EPS), 1.0)),
            "line_residual_penalty": float(min(line_rms / max(line_residual_tol, EPS), 1.0)),
            "condition_penalty": float(0.5 * min(float(condition_median) / 1e6, 1.0)) if condition_median is not None else 0.0,
        }
        score = (
            score_components["independent_view_score"]
            + score_components["finite_segment_span_score"]
            + score_components["saturated_support_score"]
            + score_components["second_edge_margin_score"]
            - score_components["line_residual_penalty"]
            - score_components["condition_penalty"]
        )
        label = bool(cluster_available or (segment_available and line_rms <= line_residual_tol and t_span >= 0.15))
        score_rows.append((float(score), label))
        direction_abs_z = abs(float(np.array(info["direction"], dtype=float)[2]))
        edge_rows.append({
            "edge_id": eid,
            "vertex_ids": info["vertex_ids"],
            "length": as_json_float(float(info["length"])),
            "direction": as_json_point(np.array(info["direction"], dtype=float)),
            "z_span": as_json_float(float(info["z_span"])),
            "z_zone": info["z_zone"],
            "sampled_z_levels": int(max(1, round(float(info["z_span"]) / max(abs(z_step), EPS))) + 1),
            "potential_view_visibility": int(len(unique_views)),
            "adjacent_face_ids": info["adjacent_faces"],
            "dihedral_angle": as_json_float(float(info["dihedral_angle"])) if info["dihedral_angle"] is not None else None,
            "raw_support_count": raw_count,
            "view_count": int(len(unique_views)),
            "z_level_count": int(len(unique_z)),
            "observed_z_span": as_json_float(z_span_obs),
            "segment_t_span": as_json_float(t_span),
            "assignment_ambiguity_fraction": as_json_float(ambiguity),
            "second_edge_margin_median": as_json_float(float(margin_median)) if margin_median is not None else None,
            "line_rms": as_json_float(float(line_rms)),
            "condition_median": as_json_float(float(condition_median)) if condition_median is not None else None,
            "segment_hypothesis_available": bool(segment_available),
            "confident_cluster_available": bool(cluster_available),
            "duplicate_clusters_same_edge": bool(eid in duplicate_cluster_edges),
            "represented_in_final_face_candidate": bool(in_pool),
            "represented_in_final_active_mesh": bool(in_active),
            "category": category,
            "near_horizontal_edge": bool(direction_abs_z < 0.25),
            "near_vertical_edge": bool(direction_abs_z > 0.75),
            "length_decile": int(np.searchsorted(np.percentile(length_values, np.arange(10, 100, 10)), float(info["length"]), side="right")) if length_values.size else 0,
            "z_span_decile": int(np.searchsorted(np.percentile(zspan_values, np.arange(10, 100, 10)), float(info["z_span"]), side="right")) if zspan_values.size else 0,
            "dihedral_decile": int(np.searchsorted(np.percentile(dihedral_values[np.isfinite(dihedral_values)], np.arange(10, 100, 10)), finite_float(info.get("dihedral_angle"), 0.0), side="right")) if np.any(np.isfinite(dihedral_values)) else 0,
            "non_oracle_score_fixed_direction": as_json_float(float(score)),
            "non_oracle_score_components": {key: as_json_float(value) for key, value in score_components.items()},
        })

    edge_support_ids = {int(row["edge_id"]) for row in edge_rows if int(row["raw_support_count"]) > 0}
    usable_segment_ids = {int(row["edge_id"]) for row in edge_rows if bool(row["segment_hypothesis_available"])}
    confident_cluster_ids = {int(row["edge_id"]) for row in edge_rows if bool(row["confident_cluster_available"])}
    noncollinear_pair_ids: set[int] = set()
    plane_pair_ids: set[int] = set()
    finite_hull_ids: set[int] = set()
    face_rows: list[dict[str, object]] = []
    for face in model_face_rows:
        fid = int(face["face_id"])
        boundary_edges = sorted(face_to_edges.get(fid, []))
        supported = [eid for eid in boundary_edges if eid in edge_support_ids]
        usable = [eid for eid in boundary_edges if eid in usable_segment_ids]
        clustered = [eid for eid in boundary_edges if eid in confident_cluster_ids]
        noncollinear = False
        if len(usable) >= 2:
            for i, ea in enumerate(usable):
                da = np.array(edge_infos[ea]["direction"], dtype=float)
                for eb in usable[i + 1:]:
                    db = np.array(edge_infos[eb]["direction"], dtype=float)
                    if float(np.linalg.norm(np.cross(da, db))) >= 0.08:
                        noncollinear = True
                        break
                if noncollinear:
                    break
        if noncollinear:
            noncollinear_pair_ids.add(fid)
        plane_good = bool(noncollinear and fid in pool_face_ids)
        finite_good = bool(fid in pool_face_ids)
        if plane_good:
            plane_pair_ids.add(fid)
        if finite_good:
            finite_hull_ids.add(fid)
        first_loss = "final_active_mesh" if fid in active_face_ids else (
            "current_candidate_pool" if fid in pool_face_ids else (
                "finite_hull_or_candidate_missing" if noncollinear else (
                    "noncollinear_boundary_pair_missing" if len(usable) >= 2 else (
                        "usable_segment_missing" if len(supported) >= 2 else (
                            "second_supported_boundary_edge_missing" if len(supported) == 1 else "no_supported_boundary_edge"
                        )
                    )
                )
            )
        )
        face_rows.append({
            "face_id": fid,
            "boundary_edge_count": int(len(boundary_edges)),
            "supported_boundary_edges": int(len(supported)),
            "usable_segment_edges": int(len(usable)),
            "confident_cluster_edges": int(len(clustered)),
            "two_distinct_supported_edges": bool(len(supported) >= 2),
            "two_noncollinear_edge_hypotheses": bool(noncollinear),
            "plane_fit_good_from_boundary_pair": bool(plane_good),
            "finite_hull_good": bool(finite_good),
            "current_candidate_pool": bool(fid in pool_face_ids),
            "final_active_mesh": bool(fid in active_face_ids),
            "first_loss": first_loss,
            "first_loss_edge_id": next((eid for eid in boundary_edges if edge_rows[eid]["category"] not in {"represented_in_final_active_mesh", "represented_in_final_face_candidate", "confident_edge_cluster_available"}), None),
        })

    def edge_ceiling(ids: set[int]) -> dict[str, object]:
        return {
            "unique_edge_ids": int(len(ids)),
            "edge_recall": as_json_float(len(ids) / max(len(edge_infos), 1)),
            "edge_ids_sample": sorted(int(v) for v in ids)[:100],
        }

    def face_ceiling(predicate: str) -> dict[str, object]:
        ids = {int(row["face_id"]) for row in face_rows if bool(row.get(predicate))}
        return {
            "unique_face_ids": int(len(ids)),
            "count_recall": as_json_float(len(ids) / max(len(model_face_rows), 1)),
            "face_ids_sample": sorted(int(v) for v in ids)[:100],
        }

    score_sorted = sorted(score_rows, key=lambda item: item[0], reverse=True)
    frontier = []
    for top_k in (20, 50):
        subset = score_sorted[: min(top_k, len(score_sorted))]
        if subset:
            frontier.append({
                "top_k": int(top_k),
                "precision": as_json_float(sum(1 for _, label in subset if label) / len(subset)),
                "positive_edges": int(sum(1 for _, label in subset if label)),
            })

    ranked_edge_rows = sorted(edge_rows, key=lambda row: finite_float(row.get("non_oracle_score_fixed_direction"), -float("inf")), reverse=True)

    def score_band_anatomy(start: int, end: int) -> dict[str, object]:
        band = ranked_edge_rows[start:end]
        if not band:
            return {
                "rank_start": int(start + 1),
                "rank_end": int(end),
                "row_count": 0,
            }
        component_keys = [
            "independent_view_score",
            "finite_segment_span_score",
            "saturated_support_score",
            "second_edge_margin_score",
            "line_residual_penalty",
            "condition_penalty",
        ]
        components: dict[str, object] = {}
        for key in component_keys:
            components[key] = distribution_summary(
                [
                    (row.get("non_oracle_score_components") or {}).get(key)
                    for row in band
                    if isinstance(row.get("non_oracle_score_components"), dict)
                ]
            )
        labels = [
            bool(row.get("confident_cluster_available"))
            or (
                bool(row.get("segment_hypothesis_available"))
                and finite_float(row.get("line_rms"), float("inf")) <= line_residual_tol
                and finite_float(row.get("segment_t_span"), 0.0) >= 0.15
            )
            for row in band
        ]
        return {
            "rank_start": int(start + 1),
            "rank_end": int(min(end, len(ranked_edge_rows))),
            "row_count": int(len(band)),
            "precision_label": as_json_float(float(sum(labels)) / float(max(len(labels), 1))),
            "score": distribution_summary([row.get("non_oracle_score_fixed_direction") for row in band]),
            "raw_support_count": distribution_summary([row.get("raw_support_count") for row in band]),
            "view_count": distribution_summary([row.get("view_count") for row in band]),
            "z_level_count": distribution_summary([row.get("z_level_count") for row in band]),
            "segment_t_span": distribution_summary([row.get("segment_t_span") for row in band]),
            "line_rms": distribution_summary([row.get("line_rms") for row in band]),
            "condition_median": distribution_summary([row.get("condition_median") for row in band]),
            "length": distribution_summary([row.get("length") for row in band]),
            "z_span": distribution_summary([row.get("z_span") for row in band]),
            "assignment_ambiguity_fraction": distribution_summary([row.get("assignment_ambiguity_fraction") for row in band]),
            "duplicate_cluster_rows": int(sum(1 for row in band if bool(row.get("duplicate_clusters_same_edge")))),
            "near_vertical_rows": int(sum(1 for row in band if bool(row.get("near_vertical_edge")))),
            "component_distributions": components,
        }

    two_face_edge_like = 0
    vertex_like = 0
    disconnected_or_wide = 0
    true_ambiguous = 0
    for idx in finite_indices:
        count = int(edge_assignment_counts[int(idx)])
        if count == 1:
            eid = int(best_edge[int(idx)])
            if eid >= 0 and len(edge_infos[eid]["adjacent_faces"]) == 2:
                two_face_edge_like += 1
        elif count >= 3:
            vertex_like += 1
        elif count == 2:
            true_ambiguous += 1
        elif count == 0:
            disconnected_or_wide += 1

    digest = hashlib.sha256()
    digest.update(str(model_name).encode("utf-8"))
    digest.update(np.array(windows, dtype=np.int64).tobytes())
    digest.update(np.array(lp.shape, dtype=np.int64).tobytes())
    digest.update(np.array([len(reference_edges), finite_indices.size, edge_tol], dtype=np.float64).tobytes())
    segment_supported_ids = {int(row["edge_id"]) for row in edge_rows if bool(row["segment_hypothesis_available"])}
    category_total = int(sum(category_counts.values()))
    current_w2_segment_summary = edge_ceiling(segment_supported_ids)
    current_w2_cluster_summary = edge_ceiling(confident_cluster_ids)
    cross_view_short_rows = {int(row["edge_id"]) for row in edge_rows if row["category"] == "short_Z_but_cross_view_support_exists"}
    generic_acceptance_pass = bool(
        len(confident_cluster_ids) >= len(segment_supported_ids)
        and len((confident_cluster_ids | segment_supported_ids) - segment_supported_ids) >= 10
    )
    return {
        "schema_version": 2,
        "model": str(model_name),
        "scope": "edge-incidence-raw-support",
        "production_changed": False,
        "initial_model_usage": "posthoc edge labels/ceiling evaluation only",
        "canonical_evaluator": canonical_oracle_evaluator_metadata(),
        "metric_definitions": {
            "raw_supported_edge": "a reference finite edge has at least one full raw in-memory line_point within finite-segment distance/t tolerance; oracle edge is used only for posthoc labeling",
            "usable_segment": "an existing generic W2/fixed-index segment matches that finite reference edge after posthoc edge matching",
            "confident_cluster": "an existing generic W2 edge cluster matches that finite reference edge with finite distance, direction, and ambiguity gates",
            "represented_in_final_candidate": "at least one adjacent canonical finite face is present in the final candidate pool before active clipping",
            "final_active_mesh_edge": "at least one adjacent canonical finite face is active in the final edge-clip mesh; this is face-incidence derived, so it can be much larger than the number of explicit edge clusters",
        },
        "outside_metric_paths": {
            "cumulative_selection_outside": {
                "json_path": "parameters.<selection/repair stage>.outside or trusted_outside",
                "semantics": "stage-local accumulator used while adding or testing candidates; not directly comparable across stages",
            },
            "repair_final_outside": {
                "json_path": "parameters.repair_final_outside.cumulative_outside_fraction",
                "max_per_z_path": "parameters.repair_final_outside.cumulative_max_level_outside_fraction",
                "lost_z_path": "parameters.repair_final_outside.cumulative_lost_z_levels",
            },
            "trusted_final_outside": {
                "json_path": "parameters.trusted_outside.cumulative_outside_fraction when present, otherwise the final stage-specific trusted_outside block",
                "semantics": "trusted observed-cloud outside after a concrete candidate set is evaluated",
            },
            "max_per_z_outside": {
                "json_path": "*.cumulative_max_level_outside_fraction",
                "semantics": "maximum per-Z trusted outside fraction for the same outside block",
            },
            "lost_z": {
                "json_path": "*.cumulative_lost_z_levels",
                "semantics": "number of trusted Z slices with no surviving inside support for the same outside block",
            },
        },
        "raw_provenance": {
            "windows": [int(v) for v in windows],
            "line_points_shape": [int(v) for v in lp.shape],
            "z_level_count": int(n_z),
            "view_count": int(n_view),
            "finite_raw_observations": int(finite_indices.size),
            "total_raw_slots": int(total_flat),
            "raw_signal_signature": digest.hexdigest()[:24],
            "line_direction_provenance": "compute_line_points_multi_window does not expose per-observation local line direction; W2 segment/cluster direction is evaluated after segmentation",
        },
        "reference_edges": {
            "edge_count": int(len(edge_infos)),
            "length": distribution_summary([row["length"] for row in edge_rows]),
            "z_span": distribution_summary([row["z_span"] for row in edge_rows]),
            "dihedral_angle": distribution_summary([row["dihedral_angle"] for row in edge_rows]),
        },
        "tolerances": {
            "finite_segment_distance": as_json_float(float(edge_tol)),
            "line_residual": as_json_float(float(line_residual_tol)),
        },
        "finite_edge_raw_support_ceiling": {
            "raw_supported": edge_ceiling(edge_support_ids),
            "usable_segment_hypothesis": current_w2_segment_summary,
            "confident_cluster": current_w2_cluster_summary,
            "represented_in_final_face_candidate": edge_ceiling({int(row["edge_id"]) for row in edge_rows if bool(row["represented_in_final_face_candidate"])}),
            "represented_in_final_active_mesh": edge_ceiling({int(row["edge_id"]) for row in edge_rows if bool(row["represented_in_final_active_mesh"])}),
        },
        "edge_loss_funnel": {
            "category_counts": category_counts,
            "category_total": category_total,
            "accounting": {
                "strictly_exclusive": True,
                "reference_edge_count": int(len(edge_infos)),
                "category_total": category_total,
                "sum_equals_reference_edge_count": bool(category_total == len(edge_infos)),
                "missing_or_overlapping_rows": int(category_total - len(edge_infos)),
            },
            "by_z_zone": summarize_rows_by_category([{"category": row["z_zone"]} for row in edge_rows]),
            "by_direction": {
                "near_horizontal": int(sum(1 for row in edge_rows if bool(row["near_horizontal_edge"]))),
                "near_vertical": int(sum(1 for row in edge_rows if bool(row["near_vertical_edge"]))),
            },
        },
        "boundary_edge_face_ceiling": {
            "at_least_one_supported_boundary_edge": face_ceiling("supported_boundary_edges"),
            "at_least_two_distinct_supported_boundary_edges": face_ceiling("two_distinct_supported_edges"),
            "at_least_two_noncollinear_edge_hypotheses": face_ceiling("two_noncollinear_edge_hypotheses"),
            "plane_fit_good_from_boundary_pair": face_ceiling("plane_fit_good_from_boundary_pair"),
            "finite_hull_good": face_ceiling("finite_hull_good"),
            "current_candidate_pool": face_ceiling("current_candidate_pool"),
            "final_active_mesh": face_ceiling("final_active_mesh"),
            "first_loss_counts": summarize_rows_by_category([{"category": row["first_loss"]} for row in face_rows]),
        },
        "previous_footprint_overlap_decomposition": {
            "two_adjacent_faces_via_single_boundary_edge_points": int(two_face_edge_like),
            "three_or_more_edges_near_vertex_points": int(vertex_like),
            "two_edge_ambiguous_points": int(true_ambiguous),
            "not_explained_by_finite_edge_tolerance_points": int(disconnected_or_wide),
            "normal_case_note": "single boundary-edge points are expected to belong to two adjacent face polygons and should not be treated as face-mixing failure",
        },
        "representation_comparison": {
            "current_w2_fixed_index_segments": {
                **current_w2_segment_summary,
                "hypothesis_count": int(len(w2_segments)),
                "generation_uses_initial_model": False,
                "posthoc_evaluation_uses_initial_model": True,
            },
            "cross_view_short_edge_tls": {
                **edge_ceiling(cross_view_short_rows),
                "hypothesis_count": int(len(cross_view_short_rows)),
                "generation_uses_initial_model": True,
                "production_eligible": False,
                "reason": "this row is an oracle-reference-edge ceiling over matched raw support, not a materialized generic hypothesis pool",
            },
            "cross_view_edge_consensus_clusters": {
                **current_w2_cluster_summary,
                "hypothesis_count": int(len(w2_clusters)),
                "generation_uses_initial_model": False,
                "posthoc_evaluation_uses_initial_model": True,
            },
            "mixed_segment_split": {
                "skipped": True,
                "reason": "no production split mode was created; this audit only separates spatially plausible edge-incidence evidence posthoc",
            },
        },
        "score_dependency_audit": {
            "row_type": "oracle_reference_edge_rows",
            "production_eligible": False,
            "interpretation": "AUC/top-K below are diagnostic separation ceilings because each row is a reference edge created after oracle finite-edge matching.",
            "components": [
                {
                    "field_name": "independent_view_score",
                    "source_stage": "raw observations after oracle reference-edge matching",
                    "uses_initial_model": True,
                    "uses_reference_edge": True,
                    "uses_oracle_match": True,
                    "available_when_oracle_disabled": False,
                },
                {
                    "field_name": "finite_segment_span_score",
                    "source_stage": "projection parameter on oracle reference finite segment",
                    "uses_initial_model": True,
                    "uses_reference_edge": True,
                    "uses_oracle_match": True,
                    "available_when_oracle_disabled": False,
                },
                {
                    "field_name": "saturated_support_score",
                    "source_stage": "raw observations counted after oracle reference-edge matching",
                    "uses_initial_model": True,
                    "uses_reference_edge": True,
                    "uses_oracle_match": True,
                    "available_when_oracle_disabled": False,
                },
                {
                    "field_name": "second_edge_margin_score",
                    "source_stage": "nearest and second-nearest oracle reference finite edge distances",
                    "uses_initial_model": True,
                    "uses_reference_edge": True,
                    "uses_oracle_match": True,
                    "available_when_oracle_disabled": False,
                },
                {
                    "field_name": "line_residual_penalty",
                    "source_stage": "TLS fit to oracle-matched raw support",
                    "uses_initial_model": True,
                    "uses_reference_edge": True,
                    "uses_oracle_match": True,
                    "available_when_oracle_disabled": False,
                },
                {
                    "field_name": "condition_penalty",
                    "source_stage": "raw observation condition values restricted to oracle-matched support",
                    "uses_initial_model": True,
                    "uses_reference_edge": True,
                    "uses_oracle_match": True,
                    "available_when_oracle_disabled": False,
                },
            ],
            "production_eligible_score_required_flags": {
                "uses_initial_model": False,
                "uses_reference_edge": False,
                "uses_oracle_match": False,
                "available_when_oracle_disabled": True,
            },
        },
        "non_oracle_separation": {
            "fixed_direction_score": {
                "higher_is_better": True,
                "formula": "independent views + finite t-span + saturated support + edge margin - line residual - bounded condition penalty",
                "roc_auc": as_json_float(_auc_binary(score_rows, higher_is_better=True)) if _auc_binary(score_rows, higher_is_better=True) is not None else None,
                "pr_auc": as_json_float(_pr_auc_binary(score_rows, higher_is_better=True)) if _pr_auc_binary(score_rows, higher_is_better=True) is not None else None,
                "frontier": frontier,
                "top_band_anatomy": {
                    "ranks_1_20": score_band_anatomy(0, 20),
                    "ranks_21_50": score_band_anatomy(20, 50),
                    "ranks_51_100": score_band_anatomy(50, 100),
                    "predeclared_direction_note": "higher views/span/support/margin and lower residual/condition are better; signs were not inverted after oracle labels",
                },
            },
            "edge_cluster_precision_recall": {
                "confident_cluster_edges": int(len(confident_cluster_ids)),
                "raw_supported_edges": int(len(edge_support_ids)),
                "recall_vs_raw_supported": as_json_float(len(confident_cluster_ids) / max(len(edge_support_ids), 1)),
                "duplicate_cluster_edges": int(len(duplicate_cluster_edges)),
            },
            "generic_pool_acceptance_for_face_pairing": {
                "passed": generic_acceptance_pass,
                "reason": "current generic cluster pool does not add enough unique usable edge IDs beyond W2 segment support; oracle-reference rows are not eligible for production face pairing",
                "minimum_additional_boundary_edge_ids_required": 10,
                "precision_requirement": 0.5,
            },
        },
        "generic_face_adjacency": {
            "attempted": False,
            "reason": "gated off because no production-eligible generic edge pool improved unique usable edge IDs; face pairing would be oracle-tuned if continued",
            "current_cyclic_adjacency": {
                "candidate_face_ids": int(len(pool_face_ids)),
                "final_active_face_ids": int(len(active_face_ids)),
            },
            "bounded_spatial_k_adjacency": {
                "attempted": False,
                "reject_reason": "no accepted generic edge shortlist",
            },
            "mutual_adjacency": {
                "attempted": False,
                "reject_reason": "no accepted generic edge shortlist",
            },
            "mandatory_rejects": [
                "duplicate_same_edge",
                "overlapping_collinear_duplicate",
                "insufficient_plane_span",
                "no_shared_spatial_patch",
                "unstable_pair_plane",
                "excessive_pair_distance",
                "insufficient_independent_support",
            ],
        },
        "diagnostic_face_candidates_from_edge_incidence": {
            "attempted": False,
            "reason": "edge-first ceiling is recorded, but no production-eligible generic edge shortlist passed the gate for edge-pair candidate formation",
            "candidate_finite_good_precision": None,
            "unique_face_ids": 0,
        },
        "full_edge_clip_trials": {
            "attempted": False,
            "reason": "no non-oracle edge-incidence shortlist was promoted to candidate-level trials",
            "max_trials_per_model": 20,
        },
        "branch_decision": {
            "selected": "B" if category_counts["short_Z_but_cross_view_support_exists"] + category_counts["segment_fit_rejected"] > 0 else "C",
            "legend": {
                "A": "boundary edges absent in raw signal",
                "B": "raw edges exist, but segment formation loses short edges",
                "C": "edge clusters exist, but face adjacency/pairing is wrong",
                "D": "plane is good, finite hull is wrong",
                "E": "edge-first hypotheses exist, but non-oracle separation is insufficient",
                "F": "observable edge incidence is genuinely insufficient",
            },
            "production_mode_added": False,
        },
        "edge_rows_sample": sorted(edge_rows, key=lambda row: (not bool(row["near_vertical_edge"]), -int(row["raw_support_count"])))[:120],
        "face_rows_sample": face_rows[:160],
        "timing_seconds": as_json_float(time.perf_counter() - started),
    }


def _generic_edge_segment_points(segment: dict[str, object], z_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_indices = [
        int(value)
        for value in (segment.get("z_indices") or [])
        if 0 <= int(value) < int(z_levels.size)
    ]
    if z_indices:
        z = z_levels[np.array(z_indices, dtype=int)]
    else:
        z_min = finite_float(segment.get("z_min"), float("nan"))
        z_max = finite_float(segment.get("z_max"), float("nan"))
        if not np.isfinite(z_min) or not np.isfinite(z_max):
            return np.zeros((0,), dtype=float), np.zeros((0, 3), dtype=float)
        count = max(2, int(segment.get("levels") or 2))
        z = np.linspace(z_min, z_max, min(count, 32), dtype=float)
    points = predict_w2_line(segment, z)
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(z)
    return z[finite], points[finite]


def _tls_line_from_points(points: np.ndarray) -> dict[str, object] | None:
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 2:
        return None
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid[None, :]
    try:
        _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = vh[0].astype(float)
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        return None
    direction = direction / norm
    if tuple(float(v) for v in direction) < tuple(float(-v) for v in direction):
        direction = -direction
    coord = centered @ direction
    residual = np.linalg.norm(centered - coord[:, None] * direction[None, :], axis=1)
    order = np.argsort(coord)
    return {
        "centroid": centroid,
        "direction": direction,
        "coord": coord,
        "segment_min": centroid + float(np.min(coord)) * direction,
        "segment_max": centroid + float(np.max(coord)) * direction,
        "segment_length": float(np.max(coord) - np.min(coord)),
        "residual": residual,
        "residual_median": float(np.median(residual)) if residual.size else float("inf"),
        "residual_p95": float(np.percentile(residual, 95)) if residual.size else float("inf"),
        "condition": float(singular[0] / max(singular[1], EPS)) if singular.size > 1 else float("inf"),
        "ordered_points": pts[order],
    }


def _segment_sample_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> tuple[float, float]:
    t = np.linspace(0.0, 1.0, 7)
    a_pts = a0[None, :] + t[:, None] * (a1 - a0)[None, :]
    b_pts = b0[None, :] + t[:, None] * (b1 - b0)[None, :]
    da, _ = _edge_segment_distances(a_pts, b0, b1)
    db, _ = _edge_segment_distances(b_pts, a0, a1)
    return float(max(np.median(da), np.median(db))), float(max(np.max(da), np.max(db)))


def _generic_edge_match(
    hypothesis: dict[str, object],
    edge_infos: list[dict[str, object]],
    *,
    distance_tol: float,
    angle_tol: float,
) -> dict[str, object]:
    endpoints = np.asarray(hypothesis.get("endpoints") or [], dtype=float).reshape((-1, 3))
    direction = np.asarray(hypothesis.get("direction") or [], dtype=float)
    if endpoints.shape[0] < 2 or direction.shape != (3,):
        return {"classification": "unmatched", "edge_id": None, "confidence": False}
    samples = endpoints[0][None, :] + np.linspace(0.0, 1.0, 7)[:, None] * (endpoints[-1] - endpoints[0])[None, :]
    margin = max(float(distance_tol) * 4.0, 0.08)
    hmin = np.min(samples, axis=0) - margin
    hmax = np.max(samples, axis=0) + margin
    candidate_edges = [
        info
        for info in edge_infos
        if np.all(np.asarray(info.get("bbox_max"), dtype=float) >= hmin)
        and np.all(np.asarray(info.get("bbox_min"), dtype=float) <= hmax)
    ]
    if not candidate_edges:
        candidate_edges = edge_infos
    if len(candidate_edges) > 160:
        centroid = np.mean(samples, axis=0)
        candidate_edges = sorted(
            candidate_edges,
            key=lambda info: (
                float(
                    np.linalg.norm(
                        centroid
                        - np.clip(
                            centroid,
                            np.asarray(info.get("bbox_min"), dtype=float),
                            np.asarray(info.get("bbox_max"), dtype=float),
                        )
                    )
                ),
                int(info["edge_id"]),
            ),
        )[:160]
    rows: list[dict[str, object]] = []
    for info in candidate_edges:
        edge_dir = np.asarray(info["direction"], dtype=float)
        distances, t = _edge_segment_distances(samples, np.asarray(info["a"], dtype=float), np.asarray(info["b"], dtype=float))
        angle = float(np.degrees(np.arccos(np.clip(abs(float(direction @ edge_dir)), -1.0, 1.0))))
        inside_fraction = float(np.mean((t >= -0.05) & (t <= 1.05)))
        median_distance = float(np.median(distances))
        score = median_distance / max(distance_tol, EPS) + angle / max(angle_tol, EPS) + max(0.0, 0.6 - inside_fraction)
        rows.append(
            {
                "edge_id": int(info["edge_id"]),
                "median_distance": as_json_float(median_distance),
                "max_distance": as_json_float(float(np.max(distances))),
                "angle_deg": as_json_float(angle),
                "inside_fraction": as_json_float(inside_fraction),
                "score": as_json_float(score),
            }
        )
    rows.sort(key=lambda row: (finite_float(row.get("score"), float("inf")), int(row["edge_id"])))
    top = rows[0] if rows else None
    second = rows[1] if len(rows) > 1 else None
    if top is None:
        return {"classification": "unmatched", "edge_id": None, "confidence": False}
    margin = (
        finite_float(second.get("score"), float("inf")) - finite_float(top.get("score"), float("inf"))
        if second is not None
        else float("inf")
    )
    confident = bool(
        finite_float(top.get("median_distance"), float("inf")) <= distance_tol
        and finite_float(top.get("angle_deg"), 180.0) <= angle_tol
        and finite_float(top.get("inside_fraction"), 0.0) >= 0.55
        and margin >= 0.15
    )
    return {
        "classification": "confident" if confident else ("ambiguous" if margin < 0.15 else "unmatched"),
        "edge_id": int(top["edge_id"]) if confident else None,
        "confidence": confident,
        "top_matches": rows[:3],
        "score_margin": as_json_float(float(margin)),
    }


def diagnose_generic_cross_view_edge_pool(
    *,
    model_name: str,
    vertices: np.ndarray,
    faces: list[list[int]],
    z_levels: np.ndarray,
    line_points_w2: np.ndarray,
    w2_segments: list[dict[str, object]],
    w2_clusters: list[dict[str, object]],
    w2_candidates: list[dict[str, object]],
    baseline_payload: dict[str, object] | None,
    additive_mode: bool = False,
) -> dict[str, object]:
    started = time.perf_counter()
    reference_edges, edge_faces_map = model_edges_from_faces(vertices, faces)
    model_face_rows = model_face_planes(vertices, faces)
    edge_infos: list[dict[str, object]] = []
    z_min = float(np.min(vertices[:, 2])) if vertices.size else 0.0
    z_max = float(np.max(vertices[:, 2])) if vertices.size else 1.0
    lower_cut = z_min + (z_max - z_min) / 3.0
    for eid, (ia, ib) in enumerate(reference_edges):
        a = vertices[int(ia)]
        b = vertices[int(ib)]
        vec = b - a
        length = float(np.linalg.norm(vec))
        direction = vec / max(length, EPS)
        adjacent = sorted(int(fid) for fid in edge_faces_map.get((int(ia), int(ib)), edge_faces_map.get((int(ib), int(ia)), set())))
        z_mid = float(0.5 * (a[2] + b[2]))
        edge_infos.append(
            {
                "edge_id": int(eid),
                "a": a,
                "b": b,
                "bbox_min": np.minimum(a, b),
                "bbox_max": np.maximum(a, b),
                "length": length,
                "direction": direction,
                "z_span": float(abs(float(a[2]) - float(b[2]))),
                "lower_z": bool(z_mid <= lower_cut),
                "adjacent_faces": adjacent,
            }
        )
    finite_lp = np.asarray(line_points_w2, dtype=float).reshape((-1, 3))
    finite_lp = finite_lp[np.all(np.isfinite(finite_lp), axis=1)]
    spacing_values: list[float] = []
    if np.asarray(line_points_w2).ndim == 3:
        arr = np.asarray(line_points_w2, dtype=float)
        for zi in np.linspace(0, arr.shape[1] - 1, min(arr.shape[1], 32), dtype=int):
            pts = arr[:, int(zi), :]
            ok = np.all(np.isfinite(pts), axis=1)
            pts = pts[ok]
            if pts.shape[0] >= 3:
                step = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
                spacing_values.extend(float(v) for v in step[np.isfinite(step) & (step > EPS)])
    median_spacing = float(np.median(spacing_values)) if spacing_values else 0.02
    z_step = float(np.median(np.diff(z_levels))) if z_levels.size > 1 else 0.01
    primitive_lengths = [finite_float(segment.get("z_span"), 0.0) for segment in w2_segments]
    median_primitive_length = float(np.median([v for v in primitive_lengths if v > EPS])) if any(v > EPS for v in primitive_lengths) else max(0.1, 4.0 * abs(z_step))
    residual_values = [finite_float(segment.get("line_rms"), float("nan")) for segment in w2_segments]
    residual_clean = np.asarray([v for v in residual_values if np.isfinite(v)], dtype=float)
    residual_median = float(np.median(residual_clean)) if residual_clean.size else 0.01
    residual_mad = float(np.median(np.abs(residual_clean - residual_median))) if residual_clean.size else residual_median
    n_half = int(line_points_w2.shape[0]) if np.asarray(line_points_w2).ndim >= 2 else 1
    angular_step_deg = float(360.0 / max(n_half, 1))
    thresholds = {
        "median_nearest_neighbor_point_spacing": as_json_float(median_spacing),
        "z_step": as_json_float(abs(z_step)),
        "median_primitive_z_span": as_json_float(median_primitive_length),
        "robust_residual_median": as_json_float(residual_median),
        "robust_residual_mad": as_json_float(residual_mad),
        "angular_view_step_deg": as_json_float(angular_step_deg),
        "neighbor_k": 6,
        "max_pair_segment_distance": as_json_float(max(3.0 * median_spacing, 2.0 * abs(z_step), residual_median + 3.0 * residual_mad, 0.035)),
        "max_direction_angle_deg": as_json_float(min(12.0, max(3.0, 6.0 * angular_step_deg))),
        "min_projected_overlap_fraction": 0.12,
        "max_projected_gap": as_json_float(max(4.0 * median_spacing, 2.0 * abs(z_step), 0.04)),
        "min_independent_views": 1,
        "min_independent_z": 2,
        "caps": {
            "max_direction_angle_deg": 12.0,
            "max_pair_segment_distance_floor": 0.035,
            "reason": "shared caps prevent degenerate over-tight thresholds on very dense contours; values are model-independent",
        },
    }
    primitive_materialization_cap = 3000
    sorted_source_segments = sorted(
        w2_segments,
        key=lambda row: (
            finite_float(row.get("line_rms"), float("inf")),
            -int(row.get("levels") or 0),
            int(row.get("segment_id") or 0),
        ),
    )
    source_segments = sorted_source_segments[:primitive_materialization_cap]

    def materialize_primitives_from_segments(segments: list[dict[str, object]]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for segment in segments:
            z, pts = _generic_edge_segment_points(segment, z_levels)
            tls = _tls_line_from_points(pts)
            if tls is None:
                continue
            cyclic_index = int(segment.get("cyclic_index") or 0)
            source_mode = str(segment.get("segment_fit_mode") or "least_squares")
            z_indices = [int(value) for value in (segment.get("z_indices") or [])]
            point_ids = [f"w2:{cyclic_index}:{int(value)}" for value in z_indices[:80]]
            gid_blob = json.dumps(
                {
                    "segment_id": int(segment.get("segment_id") or -1),
                    "cyclic_index": cyclic_index,
                    "z_indices": z_indices,
                    "source": source_mode,
                },
                separators=(",", ":"),
            )
            direction = np.asarray(tls["direction"], dtype=float)
            out.append(
                {
                    "generic_id": "prim_" + hashlib.sha256(gid_blob.encode("utf-8")).hexdigest()[:16],
                    "source_segment_id": int(segment.get("segment_id") or -1),
                    "source_mode": source_mode,
                    "cyclic_index": cyclic_index,
                    "cyclic_indices": [cyclic_index],
                    "contributing_observation_ids_sample": point_ids,
                    "contributing_observation_count": int(len(z_indices) if z_indices else pts.shape[0]),
                    "endpoints": [as_json_point(np.asarray(tls["segment_min"], dtype=float)), as_json_point(np.asarray(tls["segment_max"], dtype=float))],
                    "centroid": as_json_point(np.asarray(tls["centroid"], dtype=float)),
                    "direction": as_json_point(direction),
                    "segment_length": as_json_float(float(tls["segment_length"])),
                    "z_span": as_json_float(float(np.ptp(z)) if z.size else 0.0),
                    "view_angular_span_deg": as_json_float(0.0),
                    "independent_view_count": 1,
                    "independent_z_count": int(len(set(z_indices)) if z_indices else z.size),
                    "line_residual_median": as_json_float(float(tls["residual_median"])),
                    "line_residual_p95": as_json_float(float(tls["residual_p95"])),
                    "direction_dispersion_deg": as_json_float(0.0),
                    "condition": as_json_float(float(tls["condition"])),
                    "cyclic_index_provenance": {"window": 2, "cyclic_index": cyclic_index},
                    "_points": pts,
                    "_coord": np.asarray(tls["coord"], dtype=float),
                }
            )
        return out

    primitives: list[dict[str, object]] = materialize_primitives_from_segments(source_segments)

    all_primitives: list[dict[str, object]] | None = None
    if additive_mode:
        all_primitives = materialize_primitives_from_segments(sorted_source_segments)

    def source_mode_counts(rows: list[dict[str, object]]) -> dict[str, int]:
        return {
            source: int(sum(str(row.get("source_mode")) == source for row in rows))
            for source in sorted({str(row.get("source_mode")) for row in rows})
        }

    def distribution_block(values: list[float]) -> dict[str, object]:
        clean = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
        if clean.size == 0:
            return {"count": 0}
        return {
            "count": int(clean.size),
            "min": as_json_float(float(np.min(clean))),
            "median": as_json_float(float(np.median(clean))),
            "p75": as_json_float(float(np.percentile(clean, 75))),
            "p95": as_json_float(float(np.percentile(clean, 95))),
            "max": as_json_float(float(np.max(clean))),
        }

    def primitive_distribution(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "segment_length": distribution_block([finite_float(row.get("segment_length"), float("nan")) for row in rows]),
            "z_span": distribution_block([finite_float(row.get("z_span"), float("nan")) for row in rows]),
            "independent_z_count": distribution_block([finite_float(row.get("independent_z_count"), float("nan")) for row in rows]),
            "line_residual_p95": distribution_block([finite_float(row.get("line_residual_p95"), float("nan")) for row in rows]),
        }

    def primitive_match_ids(rows: list[dict[str, object]], *, cap: int) -> set[int]:
        ids: set[int] = set()
        ordered = sorted(
            rows,
            key=lambda row: (
                finite_float(row.get("line_residual_p95"), float("inf")),
                -int(row.get("independent_z_count") or 0),
                -finite_float(row.get("segment_length"), 0.0),
                str(row.get("generic_id")),
            ),
        )[:cap]
        for row in ordered:
            match = _generic_edge_match(
                row,
                edge_infos,
                distance_tol=max(2.5 * median_spacing, 2.0 * abs(z_step), residual_median + 3.0 * residual_mad, 0.04),
                angle_tol=finite_float(thresholds["max_direction_angle_deg"], 12.0),
            )
            if bool(match.get("confidence")) and match.get("edge_id") is not None:
                ids.add(int(match["edge_id"]))
        return ids

    consensus_primitive_cap = 600 if additive_mode else 1200
    consensus_primitives = sorted(
        primitives,
        key=lambda row: (
            finite_float(row.get("line_residual_p95"), float("inf")),
            -int(row.get("independent_z_count") or 0),
            -finite_float(row.get("segment_length"), 0.0),
            str(row.get("generic_id")),
        ),
    )[:consensus_primitive_cap]
    primitive_by_id = {str(row["generic_id"]): row for row in consensus_primitives}

    def primitive_pair_features(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
        a_end = np.asarray(a["endpoints"], dtype=float)
        b_end = np.asarray(b["endpoints"], dtype=float)
        da = np.asarray(a["direction"], dtype=float)
        db = np.asarray(b["direction"], dtype=float)
        direction_angle = float(np.degrees(np.arccos(np.clip(abs(float(da @ db)), -1.0, 1.0))))
        spatial_median, spatial_max = _segment_sample_distance(a_end[0], a_end[1], b_end[0], b_end[1])
        axis = da if float(da @ db) >= 0.0 else -da
        origin = 0.5 * (np.mean(a_end, axis=0) + np.mean(b_end, axis=0))
        ac = (a_end - origin[None, :]) @ axis
        bc = (b_end - origin[None, :]) @ axis
        overlap = max(0.0, min(float(np.max(ac)), float(np.max(bc))) - max(float(np.min(ac)), float(np.min(bc))))
        union = max(float(np.max(ac)), float(np.max(bc))) - min(float(np.min(ac)), float(np.min(bc)))
        gap = max(0.0, max(float(np.min(ac)), float(np.min(bc))) - min(float(np.max(ac)), float(np.max(bc))))
        za0 = finite_float(a.get("z_span"), 0.0)
        zb0 = finite_float(b.get("z_span"), 0.0)
        independent = bool(set(a.get("cyclic_indices") or []) != set(b.get("cyclic_indices") or []))
        return {
            "spatial_median": spatial_median,
            "spatial_max": spatial_max,
            "direction_angle_deg": direction_angle,
            "projected_overlap_fraction": overlap / max(union, EPS),
            "projected_gap": gap,
            "independent_provenance": independent,
            "z_span_min": min(za0, zb0),
        }

    def compatible(a: dict[str, object], b: dict[str, object]) -> bool:
        f = primitive_pair_features(a, b)
        return bool(
            f["spatial_median"] <= finite_float(thresholds["max_pair_segment_distance"], 0.0)
            and f["direction_angle_deg"] <= finite_float(thresholds["max_direction_angle_deg"], 0.0)
            and (
                f["projected_overlap_fraction"] >= float(thresholds["min_projected_overlap_fraction"])
                or f["projected_gap"] <= finite_float(thresholds["max_projected_gap"], 0.0)
            )
            and f["z_span_min"] >= abs(z_step)
        )

    neighbor_candidates: dict[str, list[tuple[float, str]]] = {str(row["generic_id"]): [] for row in consensus_primitives}
    cell_size = max(finite_float(thresholds["max_pair_segment_distance"], 0.05) * 4.0, median_spacing * 6.0, abs(z_step) * 4.0, 0.05)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index, row in enumerate(consensus_primitives):
        centroid = np.asarray(row.get("centroid"), dtype=float)
        if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            continue
        cell = tuple(int(np.floor(float(value) / cell_size)) for value in centroid)
        grid.setdefault(cell, []).append(int(index))
    candidate_pairs: set[tuple[int, int]] = set()
    for index, row in enumerate(consensus_primitives):
        centroid = np.asarray(row.get("centroid"), dtype=float)
        if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            continue
        cell = tuple(int(np.floor(float(value) / cell_size)) for value in centroid)
        local_indices: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    local_indices.extend(grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), []))
        nearest = sorted(
            {
                int(other)
                for other in local_indices
                if int(other) != int(index)
            },
            key=lambda other: (
                float(np.linalg.norm(np.asarray(consensus_primitives[other].get("centroid"), dtype=float) - centroid)),
                int(other),
            ),
        )[: max(int(thresholds["neighbor_k"]) * 4, 12)]
        for other in nearest:
            candidate_pairs.add(tuple(sorted((int(index), int(other)))))
    for i, j in sorted(candidate_pairs):
        a = consensus_primitives[int(i)]
        b = consensus_primitives[int(j)]
        if not compatible(a, b):
            continue
        f = primitive_pair_features(a, b)
        score = (
            f["spatial_median"] / max(finite_float(thresholds["max_pair_segment_distance"], EPS), EPS)
            + f["direction_angle_deg"] / max(finite_float(thresholds["max_direction_angle_deg"], EPS), EPS)
            + max(0.0, float(thresholds["min_projected_overlap_fraction"]) - f["projected_overlap_fraction"])
            + f["projected_gap"] / max(finite_float(thresholds["max_projected_gap"], EPS), EPS)
        )
        neighbor_candidates[str(a["generic_id"])].append((float(score), str(b["generic_id"])))
        neighbor_candidates[str(b["generic_id"])].append((float(score), str(a["generic_id"])))
    top_neighbors = {
        key: {other for _, other in sorted(values, key=lambda item: (item[0], item[1]))[: int(thresholds["neighbor_k"])]}
        for key, values in neighbor_candidates.items()
    }
    mutual_edges: dict[str, set[str]] = {str(row["generic_id"]): set() for row in consensus_primitives}
    for key, values in top_neighbors.items():
        for other in values:
            if key in top_neighbors.get(other, set()):
                mutual_edges[key].add(other)
                mutual_edges[other].add(key)
    visited: set[str] = set()
    components: list[list[str]] = []
    for row in consensus_primitives:
        gid = str(row["generic_id"])
        if gid in visited:
            continue
        stack = [gid]
        visited.add(gid)
        comp: list[str] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in mutual_edges.get(cur, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    complete_groups: list[list[str]] = []
    for comp in components:
        groups: list[list[str]] = []
        for gid in sorted(comp, key=lambda value: (-int(primitive_by_id[value].get("independent_z_count") or 0), value)):
            placed = False
            for group in groups:
                if all(compatible(primitive_by_id[gid], primitive_by_id[other]) for other in group):
                    group.append(gid)
                    placed = True
                    break
            if not placed:
                groups.append([gid])
        complete_groups.extend(groups)

    hypotheses: list[dict[str, object]] = []
    rejection_counts: dict[str, int] = {}
    for group_index, group in enumerate(complete_groups):
        members = [primitive_by_id[gid] for gid in group]
        points = np.vstack([np.asarray(member["_points"], dtype=float) for member in members])
        tls = _tls_line_from_points(points)
        if tls is None:
            rejection_counts["tls_failed"] = rejection_counts.get("tls_failed", 0) + 1
            continue
        views = sorted({int(v) for member in members for v in (member.get("cyclic_indices") or [])})
        z_counts = [int(member.get("independent_z_count") or 0) for member in members]
        if len(views) < int(thresholds["min_independent_views"]) or sum(z_counts) < int(thresholds["min_independent_z"]):
            rejection_counts["insufficient_independent_support"] = rejection_counts.get("insufficient_independent_support", 0) + 1
            continue
        dirs = [np.asarray(member["direction"], dtype=float) for member in members]
        ref_dir = np.asarray(tls["direction"], dtype=float)
        direction_angles = [
            float(np.degrees(np.arccos(np.clip(abs(float(ref_dir @ d)), -1.0, 1.0))))
            for d in dirs
            if d.shape == (3,)
        ]
        endpoint_coords = []
        for member in members:
            end = np.asarray(member["endpoints"], dtype=float)
            endpoint_coords.extend(float(v) for v in ((end - np.asarray(tls["centroid"], dtype=float)[None, :]) @ ref_dir))
        endpoint_stability = float(np.std(endpoint_coords) / max(float(tls["segment_length"]), EPS)) if endpoint_coords else float("inf")
        condition = finite_float(tls["condition"], float("inf"))
        score_components = {
            "independent_view_score": min(len(views), 6) / 6.0,
            "independent_z_score": min(sum(z_counts), 20) / 20.0,
            "finite_length_score": min(float(tls["segment_length"]) / max(4.0 * median_spacing, EPS), 1.0),
            "tls_residual_penalty": min(float(tls["residual_p95"]) / max(residual_median + 3.0 * residual_mad, EPS), 1.0),
            "direction_dispersion_penalty": min((max(direction_angles) if direction_angles else 0.0) / max(finite_float(thresholds["max_direction_angle_deg"], EPS), EPS), 1.0),
            "cross_view_recurrence_score": min(max(0, len(members) - 1), 5) / 5.0,
            "endpoint_stability_penalty": min(endpoint_stability, 1.0),
            "condition_penalty": min(max(np.log10(max(condition, 1.0)) - 5.0, 0.0) / 3.0, 1.0),
            "duplicate_overlap_penalty": 0.0,
        }
        score = (
            score_components["independent_view_score"]
            + score_components["independent_z_score"]
            + score_components["finite_length_score"]
            + score_components["cross_view_recurrence_score"]
            - score_components["tls_residual_penalty"]
            - score_components["direction_dispersion_penalty"]
            - score_components["endpoint_stability_penalty"]
            - score_components["condition_penalty"]
            - score_components["duplicate_overlap_penalty"]
        )
        blob = json.dumps(sorted(group), separators=(",", ":"))
        hypotheses.append(
            {
                "generic_edge_hypothesis_id": "edge_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
                "source_mode": "mutual-spatial-complete",
                "member_primitive_ids": sorted(group),
                "member_segment_ids": sorted(int(member["source_segment_id"]) for member in members),
                "contributing_observation_count": int(sum(int(member.get("contributing_observation_count") or 0) for member in members)),
                "endpoints": [as_json_point(np.asarray(tls["segment_min"], dtype=float)), as_json_point(np.asarray(tls["segment_max"], dtype=float))],
                "centroid": as_json_point(np.asarray(tls["centroid"], dtype=float)),
                "direction": as_json_point(ref_dir),
                "segment_length": as_json_float(float(tls["segment_length"])),
                "z_span": as_json_float(float(np.ptp(points[:, 2])) if points.size else 0.0),
                "view_angular_span_deg": as_json_float((max(views) - min(views)) * angular_step_deg if views else 0.0),
                "independent_view_count": int(len(views)),
                "independent_z_count": int(sum(z_counts)),
                "line_residual_median": as_json_float(float(tls["residual_median"])),
                "line_residual_p95": as_json_float(float(tls["residual_p95"])),
                "direction_dispersion_deg": as_json_float(max(direction_angles) if direction_angles else 0.0),
                "endpoint_stability": as_json_float(endpoint_stability),
                "condition": as_json_float(condition),
                "score": as_json_float(float(score)),
                "score_components": {key: as_json_float(float(value)) for key, value in score_components.items()},
                "provenance_signature": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24],
            }
        )
    hypotheses.sort(key=lambda row: (-finite_float(row.get("score"), -float("inf")), str(row["generic_edge_hypothesis_id"])))
    posthoc_evaluation_cap = 160 if additive_mode else 300
    current_w2_evaluation_cap = 300 if additive_mode else 500
    evaluated_hypotheses = hypotheses[:posthoc_evaluation_cap]
    evaluated_w2_clusters = sorted(
        w2_clusters,
        key=lambda row: (
            -finite_float(row.get("cluster_confidence"), 0.0),
            -int(row.get("levels") or 0),
            int(row.get("edge_cluster_id") or 0),
        ),
    )[:current_w2_evaluation_cap]
    match_tol = max(2.5 * median_spacing, 2.0 * abs(z_step), residual_median + 3.0 * residual_mad, 0.04)
    current_matches = [
        _generic_edge_match(
            {
                "endpoints": [
                    as_json_point(predict_w2_line(cluster, np.array([finite_float(cluster.get("cluster_z_min"), finite_float(cluster.get("z_min"), 0.0))], dtype=float))[0]),
                    as_json_point(predict_w2_line(cluster, np.array([finite_float(cluster.get("cluster_z_max"), finite_float(cluster.get("z_max"), 0.0))], dtype=float))[0]),
                ],
                "direction": cluster.get("direction"),
            },
            edge_infos,
            distance_tol=match_tol,
            angle_tol=finite_float(thresholds["max_direction_angle_deg"], 12.0),
        )
        for cluster in evaluated_w2_clusters
    ]
    generic_matches = [
        _generic_edge_match(hypothesis, edge_infos, distance_tol=match_tol, angle_tol=finite_float(thresholds["max_direction_angle_deg"], 12.0))
        for hypothesis in evaluated_hypotheses
    ]

    face_to_edges: dict[int, set[int]] = {}
    for info in edge_infos:
        for fid in info["adjacent_faces"]:
            face_to_edges.setdefault(int(fid), set()).add(int(info["edge_id"]))
    baseline_final_faces: set[int] = set()
    baseline_candidate_ids: list[int] = []
    regression_block: dict[str, object] = {"baseline_json_available": False, "passed": None}
    if isinstance(baseline_payload, dict):
        baseline_candidates = list(baseline_payload.get("face_candidates") or [])
        active_indices = [
            int(index)
            for index in (baseline_payload.get("reconstructed") or {}).get("face_candidate_indices", [])
            if 0 <= int(index) < len(baseline_candidates)
        ]
        active_candidates = [baseline_candidates[index] for index in active_indices]
        diag = oracle_candidate_cohort(active_candidates, model_face_rows, active_plane_indices=active_indices)
        diag_ids = set(int(value) for value in diag.get("finite_face_ids", []))
        bench_ids: set[int] = set()
        bench_summary = None
        try:
            import benchmark_reconstruction_quality as bench
            reference_faces = bench.reference_face_records(vertices, faces)
            bench_summary = bench.face_level_summary(
                baseline_payload,
                reference_faces,
                bench.ABSOLUTE_CANONICAL_TOLERANCES,
                assignment_mode="finite-good-first-exhaustive",
            )
            bench_ids = set(int(value) for value in bench_summary.get("unique_finite_face_ids_list", []))
        except Exception as exc:  # pragma: no cover - diagnostic payload records the exception.
            bench_summary = {"error": repr(exc)}
        baseline_final_faces = diag_ids
        baseline_candidate_ids = [int(candidate_track_id(candidate)) for candidate in active_candidates]
        regression_block = {
            "baseline_json_available": True,
            "evaluator_inputs_digest": hashlib.sha256(
                json.dumps(
                    {
                        "candidate_ids": baseline_candidate_ids,
                        "active_indices": active_indices,
                        "reference_face_count": len(model_face_rows),
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24],
            "tolerances": dict(CANONICAL_ORACLE_TOLERANCES),
            "candidate_ids": baseline_candidate_ids[:160],
            "benchmark_unique_face_ids": sorted(int(value) for value in bench_ids),
            "diagnostic_unique_face_ids": sorted(int(value) for value in diag_ids),
            "benchmark_count": int(len(bench_ids)),
            "diagnostic_count": int(len(diag_ids)),
            "only_benchmark": sorted(int(value) for value in bench_ids - diag_ids),
            "only_diagnostic": sorted(int(value) for value in diag_ids - bench_ids),
            "passed": bool(bench_ids == diag_ids),
            "benchmark_summary_error": bench_summary.get("error") if isinstance(bench_summary, dict) else None,
        }

    current_edge_ids = {int(match["edge_id"]) for match in current_matches if bool(match.get("confidence")) and match.get("edge_id") is not None}
    generic_edge_ids = {int(match["edge_id"]) for match in generic_matches if bool(match.get("confidence")) and match.get("edge_id") is not None}
    missing_final_boundary_edges = {
        int(edge_id)
        for face_id, edge_ids in face_to_edges.items()
        if baseline_final_faces and int(face_id) not in baseline_final_faces
        for edge_id in edge_ids
    }

    def summarize_matches(
        name: str,
        rows: list[dict[str, object]],
        *,
        hypothesis_count: int,
    ) -> dict[str, object]:
        confident = [row for row in rows if bool(row.get("confidence"))]
        edge_ids = [int(row["edge_id"]) for row in confident if row.get("edge_id") is not None]
        unique_ids = set(edge_ids)
        ambiguous = [row for row in rows if str(row.get("classification")) == "ambiguous"]
        unmatched = [row for row in rows if str(row.get("classification")) == "unmatched"]
        short_edges = {int(info["edge_id"]) for info in edge_infos if float(info["length"]) <= float(np.percentile([e["length"] for e in edge_infos], 25))}
        short_z = {int(info["edge_id"]) for info in edge_infos if float(info["z_span"]) <= float(np.percentile([e["z_span"] for e in edge_infos], 25))}
        lower_z = {int(info["edge_id"]) for info in edge_infos if bool(info["lower_z"])}
        return {
            "name": name,
            "raw_hypotheses": int(hypothesis_count),
            "kept_hypotheses": int(len(rows)),
            "confidently_matched_finite_edges": int(len(confident)),
            "unique_edge_ids": int(len(unique_ids)),
            "precision": as_json_float(float(len(confident) / max(len(rows), 1))),
            "recall": as_json_float(float(len(unique_ids) / max(len(edge_infos), 1))),
            "ambiguous": int(len(ambiguous)),
            "unmatched": int(len(unmatched)),
            "duplicate_assignments": int(len(edge_ids) - len(unique_ids)),
            "duplicate_rate": as_json_float(float((len(edge_ids) - len(unique_ids)) / max(len(confident), 1))),
            "short_edge_recall": as_json_float(float(len(unique_ids & short_edges) / max(len(short_edges), 1))),
            "short_Z_recall": as_json_float(float(len(unique_ids & short_z) / max(len(short_z), 1))),
            "lower_Z_recall": as_json_float(float(len(unique_ids & lower_z) / max(len(lower_z), 1))),
            "boundary_edges_missing_final_faces": int(len(unique_ids & missing_final_boundary_edges)),
            "control_edges_preserved": int(len(current_edge_ids & unique_ids)),
            "control_edges_lost": int(len(current_edge_ids - unique_ids)),
        }

    def top_band(rows: list[dict[str, object]], matches: list[dict[str, object]], start: int, end: int) -> dict[str, object]:
        pairs = list(zip(rows, matches))[start:end]
        good = [pair for pair in pairs if bool(pair[1].get("confidence"))]
        edge_ids = [int(match["edge_id"]) for _, match in good if match.get("edge_id") is not None]
        return {
            "rank_start": int(start + 1),
            "rank_end": int(min(end, len(rows))),
            "rows": int(len(pairs)),
            "precision": as_json_float(float(len(good) / max(len(pairs), 1))),
            "unique_edge_ids": int(len(set(edge_ids))),
            "duplicate_assignments": int(len(edge_ids) - len(set(edge_ids))),
            "ambiguous": int(sum(1 for _, match in pairs if str(match.get("classification")) == "ambiguous")),
            "unmatched": int(sum(1 for _, match in pairs if str(match.get("classification")) == "unmatched")),
        }

    top_bands = {
        "top_20": top_band(evaluated_hypotheses, generic_matches, 0, 20),
        "ranks_21_50": top_band(evaluated_hypotheses, generic_matches, 20, 50),
        "top_50": top_band(evaluated_hypotheses, generic_matches, 0, 50),
        "top_100": top_band(evaluated_hypotheses, generic_matches, 0, 100),
    }
    current_summary = summarize_matches("current_w2_clusters", current_matches, hypothesis_count=len(w2_clusters))
    generic_summary = summarize_matches("mutual_spatial_complete", generic_matches, hypothesis_count=len(hypotheses))
    additive_audit = None
    if additive_mode:
        additive_audit = diagnose_generic_edge_additive_control(
            model_name=model_name,
            thresholds=thresholds,
            median_spacing=median_spacing,
            z_step=abs(z_step),
            residual_median=residual_median,
            residual_mad=residual_mad,
            edge_infos=edge_infos,
            all_primitives=all_primitives or primitives,
            capped_primitives=primitives,
            hypotheses=hypotheses,
            evaluated_hypotheses=evaluated_hypotheses,
            generic_matches=generic_matches,
            evaluated_w2_clusters=evaluated_w2_clusters,
            current_matches=current_matches,
            current_edge_ids=current_edge_ids,
            missing_final_boundary_edges=missing_final_boundary_edges,
            w2_segment_count=len(w2_segments),
            primitive_materialization_cap=primitive_materialization_cap,
            match_tol=match_tol,
        )
    additional_edges = generic_edge_ids - current_edge_ids
    edge_gate = {
        "additional_unique_usable_edge_ids_vs_current_w2": int(len(additional_edges)),
        "additional_edge_ids_sample": sorted(int(value) for value in additional_edges)[:120],
        "top50_precision": top_bands["top_50"]["precision"],
        "control_edge_recall_not_decreased": bool(len(current_edge_ids - generic_edge_ids) == 0),
        "duplicate_rate_not_more_than_2x": bool(
            finite_float(generic_summary.get("duplicate_rate"), 0.0)
            <= 2.0 * max(finite_float(current_summary.get("duplicate_rate"), 0.0), EPS)
        ),
        "dependency_declaration_non_oracle": True,
    }
    edge_gate["passed_model_local"] = bool(
        finite_float(top_bands["top_50"].get("precision"), 0.0) >= 0.5
        and bool(edge_gate["control_edge_recall_not_decreased"])
        and bool(edge_gate["duplicate_rate_not_more_than_2x"])
        and bool(edge_gate["dependency_declaration_non_oracle"])
    )
    failure = "passed"
    primitive_pool_capped = bool(len(source_segments) < len(w2_segments))
    if not primitive_pool_capped and len(primitives) <= len(w2_segments) * 0.5:
        failure = "primitive_formation_failure"
    elif len(hypotheses) <= len(primitives) * 0.25:
        failure = "consensus_merge_failure"
    elif finite_float(top_bands["top_50"].get("precision"), 0.0) < 0.5:
        failure = "non_oracle_ranking_failure"
    elif len(additional_edges) < 10:
        failure = "insufficient_new_edges"
    return {
        "schema_version": 1,
        "scope": "generic-edge-additive-control" if additive_mode else "generic-cross-view-edge-pool",
        "model": str(model_name),
        "production_changed": False,
        "initial_model_usage": "posthoc oracle evaluation only; generation/clustering/ranking use W2 observed primitives only",
        "canonical_regression": regression_block,
        "primitive_pool": {
            "source": "existing non-oracle W2 primitive segments",
            "raw_w2_segments": int(len(w2_segments)),
            "primitive_materialization_cap": int(primitive_materialization_cap),
            "source_segments_materialized": int(len(source_segments)),
            "materialized_primitives": int(len(primitives)),
            "source_mode_counts": {
                source: int(sum(str(row.get("source_mode")) == source for row in primitives))
                for source in sorted({str(row.get("source_mode")) for row in primitives})
            },
            "primitive_sample": [
                {key: value for key, value in row.items() if not key.startswith("_")}
                for row in primitives[:80]
            ],
        },
        "adaptive_thresholds": thresholds,
        "consensus": {
            "mode": "mutual-spatial-complete",
            "primitive_cap": int(consensus_primitive_cap),
            "primitives_considered": int(len(consensus_primitives)),
            "mutual_neighbor_edges": int(sum(len(values) for values in mutual_edges.values()) // 2),
            "connected_components": int(len(components)),
            "complete_linkage_groups": int(len(complete_groups)),
            "raw_hypotheses": int(len(complete_groups)),
            "kept_hypotheses": int(len(hypotheses)),
            "rejection_counts": {key: int(value) for key, value in sorted(rejection_counts.items())},
            "hypothesis_sample": [
                {key: value for key, value in row.items() if key not in {"member_primitive_ids"}}
                for row in hypotheses[:100]
            ],
        },
        "score_dependency_declaration": {
            key: {
                "uses_initial_model": False,
                "uses_reference_edge": False,
                "uses_oracle_match": False,
                "available_when_oracle_disabled": True,
            }
            for key in [
                "independent_view_score",
                "independent_z_score",
                "finite_length_score",
                "tls_residual_penalty",
                "direction_dispersion_penalty",
                "cross_view_recurrence_score",
                "endpoint_stability_penalty",
                "condition_penalty",
                "duplicate_overlap_penalty",
            ]
        },
        "posthoc_evaluation": {
            "current_w2_clusters": current_summary,
            "mutual_spatial_complete": generic_summary,
            "current_w2_evaluation_cap": int(current_w2_evaluation_cap),
            "current_w2_evaluated_clusters": int(len(evaluated_w2_clusters)),
            "posthoc_evaluation_cap": int(posthoc_evaluation_cap),
            "posthoc_evaluated_hypotheses": int(len(evaluated_hypotheses)),
            "top_bands": top_bands,
        },
        "additive_control": additive_audit,
        "edge_gate": edge_gate,
        "face_formation": {
            "attempted": False,
            "reason": "edge gate is evaluated after pear+cushion aggregate; this fast scope does not modify production candidates",
            "current_cyclic_adjacency_candidate_count": int(len(w2_candidates)),
        },
        "full_edge_clip_trials": {
            "attempted": False,
            "reason": "not run in fast edge-pool scope unless aggregate gate passes in a later controlled diagnostic",
        },
        "branch": "unresolved_at_generic_edge_hypothesis_formation" if failure != "passed" else "edge_gate_candidate",
        "failure_anatomy": failure,
        "timing_seconds": as_json_float(time.perf_counter() - started),
    }


def diagnose_generic_edge_additive_control(
    *,
    model_name: str,
    thresholds: dict[str, object],
    median_spacing: float,
    z_step: float,
    residual_median: float,
    residual_mad: float,
    edge_infos: list[dict[str, object]],
    all_primitives: list[dict[str, object]],
    capped_primitives: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    evaluated_hypotheses: list[dict[str, object]],
    generic_matches: list[dict[str, object]],
    evaluated_w2_clusters: list[dict[str, object]],
    current_matches: list[dict[str, object]],
    current_edge_ids: set[int],
    missing_final_boundary_edges: set[int],
    w2_segment_count: int,
    primitive_materialization_cap: int,
    match_tol: float,
) -> dict[str, object]:
    started = time.perf_counter()
    max_pair_distance = finite_float(thresholds.get("max_pair_segment_distance"), 0.05)
    max_angle = finite_float(thresholds.get("max_direction_angle_deg"), 12.0)
    max_gap = finite_float(thresholds.get("max_projected_gap"), 0.04)
    neighbor_k = int(thresholds.get("neighbor_k") or 6)
    cell_size = max(max_pair_distance * 4.0, median_spacing * 6.0, abs(z_step) * 4.0, 0.05)

    def distribution(values: list[float]) -> dict[str, object]:
        clean = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
        if clean.size == 0:
            return {"count": 0}
        return {
            "count": int(clean.size),
            "min": as_json_float(float(np.min(clean))),
            "median": as_json_float(float(np.median(clean))),
            "p75": as_json_float(float(np.percentile(clean, 75))),
            "p95": as_json_float(float(np.percentile(clean, 95))),
            "max": as_json_float(float(np.max(clean))),
        }

    def source_mode_counts(rows: list[dict[str, object]]) -> dict[str, int]:
        return {
            source: int(sum(str(row.get("source_mode")) == source for row in rows))
            for source in sorted({str(row.get("source_mode")) for row in rows})
        }

    def primitive_distribution(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "segment_length": distribution([finite_float(row.get("segment_length"), float("nan")) for row in rows]),
            "z_span": distribution([finite_float(row.get("z_span"), float("nan")) for row in rows]),
            "independent_z_count": distribution([finite_float(row.get("independent_z_count"), float("nan")) for row in rows]),
            "line_residual_p95": distribution([finite_float(row.get("line_residual_p95"), float("nan")) for row in rows]),
        }

    def edge_match_ids(rows: list[dict[str, object]], cap: int) -> set[int]:
        ids: set[int] = set()
        ordered = sorted(
            rows,
            key=lambda row: (
                finite_float(row.get("line_residual_p95"), float("inf")),
                -int(row.get("independent_z_count") or 0),
                -finite_float(row.get("segment_length"), 0.0),
                str(row.get("generic_id") or row.get("generic_edge_hypothesis_id")),
            ),
        )[:cap]
        for row in ordered:
            match = _generic_edge_match(row, edge_infos, distance_tol=match_tol, angle_tol=max_angle)
            if bool(match.get("confidence")) and match.get("edge_id") is not None:
                ids.add(int(match["edge_id"]))
        return ids

    def pair_features(a: dict[str, object], b: dict[str, object]) -> dict[str, float]:
        a_end = np.asarray(a.get("endpoints") or [], dtype=float).reshape((-1, 3))
        b_end = np.asarray(b.get("endpoints") or [], dtype=float).reshape((-1, 3))
        da = np.asarray(a.get("direction") or [], dtype=float)
        db = np.asarray(b.get("direction") or [], dtype=float)
        if a_end.shape[0] < 2 or b_end.shape[0] < 2 or da.shape != (3,) or db.shape != (3,):
            return {
                "direction_angle_deg": float("inf"),
                "spatial_median": float("inf"),
                "projected_overlap_fraction": 0.0,
                "projected_gap": float("inf"),
                "endpoint_distance": float("inf"),
                "centroid_distance": float("inf"),
                "provenance_overlap_fraction": 0.0,
                "z_overlap_fraction": 0.0,
            }
        direction_angle = float(np.degrees(np.arccos(np.clip(abs(float(da @ db)), -1.0, 1.0))))
        spatial_median, _ = _segment_sample_distance(a_end[0], a_end[1], b_end[0], b_end[1])
        axis = da if float(da @ db) >= 0.0 else -da
        origin = 0.5 * (np.mean(a_end, axis=0) + np.mean(b_end, axis=0))
        ac = (a_end - origin[None, :]) @ axis
        bc = (b_end - origin[None, :]) @ axis
        overlap = max(0.0, min(float(np.max(ac)), float(np.max(bc))) - max(float(np.min(ac)), float(np.min(bc))))
        union = max(float(np.max(ac)), float(np.max(bc))) - min(float(np.min(ac)), float(np.min(bc)))
        gap = max(0.0, max(float(np.min(ac)), float(np.min(bc))) - min(float(np.max(ac)), float(np.max(bc))))
        endpoint_distance = min(
            max(float(np.linalg.norm(a_end[0] - b_end[0])), float(np.linalg.norm(a_end[1] - b_end[1]))),
            max(float(np.linalg.norm(a_end[0] - b_end[1])), float(np.linalg.norm(a_end[1] - b_end[0]))),
        )
        a_members = set(str(value) for value in (a.get("member_primitive_ids") or []))
        b_members = set(str(value) for value in (b.get("member_primitive_ids") or []))
        az0, az1 = float(np.min(a_end[:, 2])), float(np.max(a_end[:, 2]))
        bz0, bz1 = float(np.min(b_end[:, 2])), float(np.max(b_end[:, 2]))
        z_overlap = max(0.0, min(az1, bz1) - max(az0, bz0))
        z_union = max(az1, bz1) - min(az0, bz0)
        return {
            "direction_angle_deg": direction_angle,
            "spatial_median": spatial_median,
            "projected_overlap_fraction": overlap / max(union, EPS),
            "projected_gap": gap,
            "endpoint_distance": endpoint_distance,
            "centroid_distance": float(np.linalg.norm(np.asarray(a.get("centroid"), dtype=float) - np.asarray(b.get("centroid"), dtype=float))),
            "provenance_overlap_fraction": len(a_members & b_members) / max(len(a_members | b_members), 1),
            "z_overlap_fraction": z_overlap / max(z_union, EPS),
        }

    def duplicate_relation(a: dict[str, object], b: dict[str, object]) -> str:
        f = pair_features(a, b)
        a_members = set(str(value) for value in (a.get("member_primitive_ids") or []))
        b_members = set(str(value) for value in (b.get("member_primitive_ids") or []))
        if a_members and a_members == b_members:
            return "exact_same_primitive_provenance"
        if f["provenance_overlap_fraction"] >= 0.95:
            return "same_observed_points"
        if f["direction_angle_deg"] <= max_angle and f["endpoint_distance"] <= max(1.5 * median_spacing, 0.03) and f["projected_overlap_fraction"] >= 0.85:
            return "same_finite_segment"
        if f["direction_angle_deg"] <= max_angle and f["spatial_median"] <= max_pair_distance and f["projected_overlap_fraction"] >= 0.45:
            a_len = finite_float(a.get("segment_length"), 0.0)
            b_len = finite_float(b.get("segment_length"), 0.0)
            return "nested_segment" if min(a_len, b_len) <= 0.75 * max(a_len, b_len) else "overlapping_collinear_segment"
        if f["direction_angle_deg"] <= max_angle and f["spatial_median"] <= 2.0 * max_pair_distance and f["projected_gap"] <= 2.0 * max_gap:
            return "fragmented_pieces_one_segment"
        if f["direction_angle_deg"] <= max_angle and f["spatial_median"] > 2.0 * max_pair_distance:
            return "near_infinite_line_spatially_distinct"
        if f["spatial_median"] <= max_pair_distance and f["direction_angle_deg"] > 30.0:
            return "intersecting_physical_edges"
        return "unrelated"

    def equivalent(a: dict[str, object], b: dict[str, object]) -> bool:
        f = pair_features(a, b)
        return bool(
            f["direction_angle_deg"] <= max_angle
            and f["spatial_median"] <= max_pair_distance
            and (
                f["projected_overlap_fraction"] >= 0.35
                or f["projected_gap"] <= max_gap
                or f["provenance_overlap_fraction"] >= 0.20
            )
            and f["endpoint_distance"] <= max(4.0 * median_spacing, 3.0 * abs(z_step), 0.08)
        )

    def quality(row: dict[str, object]) -> float:
        support = min(int(row.get("contributing_observation_count") or 0), 80) / 80.0
        views = min(int(row.get("independent_view_count") or 0), 6) / 6.0
        z_count = min(int(row.get("independent_z_count") or 0), 20) / 20.0
        length = min(finite_float(row.get("segment_length"), 0.0) / max(4.0 * median_spacing, EPS), 1.0)
        residual = min(finite_float(row.get("line_residual_p95"), float("inf")) / max(residual_median + 3.0 * residual_mad, EPS), 1.0)
        dispersion = min(finite_float(row.get("direction_dispersion_deg"), 0.0) / max(max_angle, EPS), 1.0)
        endpoint = min(finite_float(row.get("endpoint_stability"), 0.0), 1.0)
        condition = finite_float(row.get("condition"), 1.0)
        condition_penalty = min(max(np.log10(max(condition, 1.0)) - 5.0, 0.0) / 3.0, 1.0)
        return float(support + views + z_count + length - residual - dispersion - endpoint - condition_penalty)

    def build_groups(rows: list[dict[str, object]]) -> list[list[int]]:
        grid: dict[tuple[int, int, int], list[int]] = {}
        for index, row in enumerate(rows):
            centroid = np.asarray(row.get("centroid"), dtype=float)
            if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
                continue
            cell = tuple(int(np.floor(float(value) / cell_size)) for value in centroid)
            grid.setdefault(cell, []).append(index)
        edges: dict[int, set[int]] = {index: set() for index in range(len(rows))}
        for index, row in enumerate(rows):
            centroid = np.asarray(row.get("centroid"), dtype=float)
            if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
                continue
            cell = tuple(int(np.floor(float(value) / cell_size)) for value in centroid)
            nearby: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nearby.extend(grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), []))
            nearest = sorted(
                {int(other) for other in nearby if int(other) != int(index)},
                key=lambda other: (float(np.linalg.norm(np.asarray(rows[other].get("centroid"), dtype=float) - centroid)), int(other)),
            )[: max(neighbor_k * 4, 12)]
            for other in nearest:
                if equivalent(row, rows[other]):
                    edges[index].add(other)
                    edges[other].add(index)
        visited: set[int] = set()
        groups: list[list[int]] = []
        for index in range(len(rows)):
            if index in visited:
                continue
            stack = [index]
            visited.add(index)
            component: list[int] = []
            while stack:
                cur = stack.pop()
                component.append(cur)
                for nxt in edges.get(cur, set()):
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            complete_groups: list[list[int]] = []
            for member in sorted(component, key=lambda idx: (-quality(rows[idx]), str(rows[idx].get("generic_edge_hypothesis_id")))):
                placed = False
                for group in complete_groups:
                    if all(equivalent(rows[member], rows[other]) for other in group):
                        group.append(member)
                        placed = True
                        break
                if not placed:
                    complete_groups.append([member])
            groups.extend([sorted(group) for group in complete_groups])
        return groups

    def representative(rows: list[dict[str, object]], group: list[int], mode: str) -> dict[str, object]:
        members = [rows[index] for index in group]
        if mode == "lowest_balanced_tls_residual":
            return sorted(members, key=lambda row: (finite_float(row.get("line_residual_p95"), float("inf")), -quality(row), str(row.get("generic_edge_hypothesis_id"))))[0]
        if mode == "highest_bounded_observed_quality":
            return sorted(members, key=lambda row: (-quality(row), finite_float(row.get("line_residual_p95"), float("inf")), str(row.get("generic_edge_hypothesis_id"))))[0]
        def medoid_cost(row: dict[str, object]) -> float:
            return float(sum(pair_features(row, other)["spatial_median"] for other in members))
        return sorted(members, key=lambda row: (medoid_cost(row), -quality(row), str(row.get("generic_edge_hypothesis_id"))))[0]

    def representative_summary(rows: list[dict[str, object]], groups: list[list[int]], mode: str) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
        reps = [representative(rows, group, mode) for group in groups]
        row_index_by_id = {id(row): index for index, row in enumerate(rows)}
        matches = [
            consolidation_match_cache.get(row_index_by_id.get(id(rep), -1))
            or _generic_edge_match(rep, edge_infos, distance_tol=match_tol, angle_tol=max_angle)
            for rep in reps
        ]
        good_ids = [int(match["edge_id"]) for match in matches if bool(match.get("confidence")) and match.get("edge_id") is not None]
        bad_rep_good_member = 0
        for group, match in zip(groups, matches):
            if bool(match.get("confidence")):
                continue
            if any(bool((consolidation_match_cache.get(int(index)) or {}).get("confidence")) for index in group):
                bad_rep_good_member += 1
        return (
            {
                "representative": mode,
                "groups": int(len(groups)),
                "matched": int(len(good_ids)),
                "unique_edge_ids": int(len(set(good_ids))),
                "precision": as_json_float(float(len(good_ids) / max(len(groups), 1))),
                "duplicate_assignments": int(len(good_ids) - len(set(good_ids))),
                "duplicate_rate": as_json_float(float((len(good_ids) - len(set(good_ids))) / max(len(good_ids), 1))),
                "bad_representative_but_good_member": int(bad_rep_good_member),
                "current_control_overlap": int(len(set(good_ids) & current_edge_ids)),
            },
            reps,
            matches,
        )

    discarded_primitives = all_primitives[len(capped_primitives):] if len(all_primitives) >= len(capped_primitives) else []
    cap_match_cap = min(len(capped_primitives), 200)
    full_match_cap = min(len(all_primitives), 400)
    cap_primitive_ids = edge_match_ids(capped_primitives, cap_match_cap)
    full_primitive_ids = edge_match_ids(all_primitives, full_match_cap)

    current_core: list[dict[str, object]] = []
    for cluster, match in zip(evaluated_w2_clusters, current_matches):
        z0 = finite_float(cluster.get("cluster_z_min"), finite_float(cluster.get("z_min"), 0.0))
        z1 = finite_float(cluster.get("cluster_z_max"), finite_float(cluster.get("z_max"), z0))
        p0 = predict_w2_line(cluster, np.array([z0], dtype=float))[0]
        p1 = predict_w2_line(cluster, np.array([z1], dtype=float))[0]
        current_core.append(
            {
                "core_id": f"w2_core_{int(cluster.get('edge_cluster_id') or len(current_core))}",
                "edge_cluster_id": int(cluster.get("edge_cluster_id") or len(current_core)),
                "contributing_primitive_ids": [int(value) for value in (cluster.get("segment_ids") or [])[:80]],
                "endpoints": [as_json_point(p0), as_json_point(p1)],
                "direction": as_json_point(np.asarray(cluster.get("direction") if cluster.get("direction") is not None else [0.0, 0.0, 1.0], dtype=float)),
                "centroid": as_json_point(0.5 * (p0 + p1)),
                "segment_length": as_json_float(float(np.linalg.norm(p1 - p0))),
                "source": "immutable_current_w2_cluster",
                "selection_order": int(len(current_core)),
                "posthoc_edge_id": int(match["edge_id"]) if bool(match.get("confidence")) and match.get("edge_id") is not None else None,
                "match_classification": str(match.get("classification")),
            }
        )

    duplicate_counts: dict[str, int] = {}
    edge_to_indices: dict[int, list[int]] = {}
    for index, match in enumerate(generic_matches):
        if bool(match.get("confidence")) and match.get("edge_id") is not None:
            edge_to_indices.setdefault(int(match["edge_id"]), []).append(index)
    for indices in edge_to_indices.values():
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1:]:
                relation = duplicate_relation(evaluated_hypotheses[left], evaluated_hypotheses[right])
                duplicate_counts[relation] = duplicate_counts.get(relation, 0) + 1

    consolidation_rows = hypotheses[: min(len(hypotheses), 300)]
    groups = build_groups(consolidation_rows)
    consolidation_match_cache = {
        int(index): _generic_edge_match(row, edge_infos, distance_tol=match_tol, angle_tol=max_angle)
        for index, row in enumerate(consolidation_rows)
    }
    representative_modes = ["medoid_finite_segment", "lowest_balanced_tls_residual", "highest_bounded_observed_quality"]
    rep_summaries: dict[str, object] = {}
    rep_payloads: dict[str, tuple[list[dict[str, object]], list[dict[str, object]]]] = {}
    for mode in representative_modes:
        summary, reps, matches = representative_summary(consolidation_rows, groups, mode)
        rep_summaries[mode] = summary
        rep_payloads[mode] = (reps, matches)
    selected_mode = sorted(
        representative_modes,
        key=lambda mode: (
            -finite_float((rep_summaries[mode] or {}).get("unique_edge_ids"), 0.0),
            -finite_float((rep_summaries[mode] or {}).get("precision"), 0.0),
            finite_float((rep_summaries[mode] or {}).get("duplicate_rate"), 1.0),
            mode,
        ),
    )[0]
    reps, rep_matches = rep_payloads[selected_mode]

    def core_relation(rep: dict[str, object]) -> tuple[str, str | None]:
        best: tuple[float, str, dict[str, float]] | None = None
        for core in current_core:
            f = pair_features(rep, core)
            score = f["spatial_median"] / max(max_pair_distance, EPS) + f["direction_angle_deg"] / max(max_angle, EPS) + max(0.0, 0.4 - f["projected_overlap_fraction"])
            if best is None or score < best[0]:
                best = (float(score), str(core["core_id"]), f)
        if best is None:
            return "new_spatial_region", None
        _, core_id, f = best
        if f["direction_angle_deg"] <= max_angle and f["spatial_median"] <= max_pair_distance and f["projected_overlap_fraction"] >= 0.70:
            return "equivalent_to_core", core_id
        core_len = finite_float(next((core for core in current_core if core["core_id"] == core_id), {}).get("segment_length"), 0.0)
        rep_len = finite_float(rep.get("segment_length"), 0.0)
        if f["direction_angle_deg"] <= max_angle and f["spatial_median"] <= max_pair_distance and f["projected_overlap_fraction"] >= 0.35 and rep_len <= 0.85 * max(core_len, rep_len):
            return "overlapping_core_fragment", core_id
        if f["direction_angle_deg"] <= max_angle and f["spatial_median"] <= 1.5 * max_pair_distance and f["projected_overlap_fraction"] >= 0.20:
            return "extends_core_finite_span", core_id
        if f["direction_angle_deg"] <= max_angle:
            return "spatially_distinct_same_direction", core_id
        return "new_direction", core_id

    additions: list[dict[str, object]] = []
    rejection_reasons: dict[str, int] = {}
    for group_index, (rep, match) in enumerate(zip(reps, rep_matches)):
        relation, core_id = core_relation(rep)
        rejection_reason = relation if relation in {"equivalent_to_core", "overlapping_core_fragment"} else None
        if rejection_reason is not None:
            rejection_reasons[rejection_reason] = rejection_reasons.get(rejection_reason, 0) + 1
        group = groups[group_index]
        member_ids = [str(consolidation_rows[index].get("generic_edge_hypothesis_id")) for index in group]
        group_id = "gadd_" + hashlib.sha256(json.dumps(sorted(member_ids), separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        additions.append(
            {
                "additive_group_id": group_id,
                "group_size": int(len(group)),
                "member_ids": member_ids[:80],
                "representative_id": str(rep.get("generic_edge_hypothesis_id")),
                "representative_mode": selected_mode,
                "score": as_json_float(quality(rep)),
                "core_relation": relation,
                "nearest_core_id": core_id,
                "rejection_reason": rejection_reason,
                "posthoc_edge_id": int(match["edge_id"]) if bool(match.get("confidence")) and match.get("edge_id") is not None else None,
                "posthoc_classification": str(match.get("classification")),
                "is_posthoc_good": bool(match.get("confidence")),
                "endpoints": rep.get("endpoints"),
                "direction": rep.get("direction"),
                "centroid": rep.get("centroid"),
                "segment_length": rep.get("segment_length"),
                "independent_view_count": rep.get("independent_view_count"),
                "independent_z_count": rep.get("independent_z_count"),
            }
        )

    survivors = [row for row in additions if row.get("rejection_reason") is None]
    covered_spatial: set[tuple[int, int, int]] = set()
    covered_direction: set[int] = set()
    covered_z: set[int] = set()
    ordered_additions: list[dict[str, object]] = []
    remaining = list(survivors)
    while remaining:
        def marginal(row: dict[str, object]) -> tuple[float, str]:
            centroid = np.asarray(row.get("centroid"), dtype=float)
            direction = np.asarray(row.get("direction"), dtype=float)
            spatial_bin = tuple(int(np.floor(float(value) / max(cell_size, EPS))) for value in centroid) if centroid.shape == (3,) else (0, 0, 0)
            direction_bin = int(np.floor((math.atan2(float(direction[1]), float(direction[0])) + math.pi) / max(math.radians(15.0), EPS))) if direction.shape == (3,) else 0
            z_bin = int(np.floor(float(centroid[2]) / max(4.0 * abs(z_step), EPS))) if centroid.shape == (3,) else 0
            novelty = (0.45 if spatial_bin not in covered_spatial else 0.0) + (0.25 if direction_bin not in covered_direction else 0.0) + (0.20 if z_bin not in covered_z else 0.0)
            relation_bonus = {
                "new_direction": 0.35,
                "new_spatial_region": 0.30,
                "spatially_distinct_same_direction": 0.20,
                "extends_core_finite_span": 0.15,
            }.get(str(row.get("core_relation")), 0.0)
            return finite_float(row.get("score"), 0.0) + novelty + relation_bonus, str(row.get("additive_group_id"))
        remaining.sort(key=lambda row: (-marginal(row)[0], marginal(row)[1]))
        chosen = remaining.pop(0)
        ordered_additions.append(chosen)
        centroid = np.asarray(chosen.get("centroid"), dtype=float)
        direction = np.asarray(chosen.get("direction"), dtype=float)
        if centroid.shape == (3,):
            covered_spatial.add(tuple(int(np.floor(float(value) / max(cell_size, EPS))) for value in centroid))
            covered_z.add(int(np.floor(float(centroid[2]) / max(4.0 * abs(z_step), EPS))))
        if direction.shape == (3,):
            covered_direction.add(int(np.floor((math.atan2(float(direction[1]), float(direction[0])) + math.pi) / max(math.radians(15.0), EPS))))

    def prefix_summary(prefix: int) -> dict[str, object]:
        selected = ordered_additions[:prefix]
        good_ids = [int(row["posthoc_edge_id"]) for row in selected if bool(row.get("is_posthoc_good")) and row.get("posthoc_edge_id") is not None]
        unique = set(good_ids)
        new_ids = unique - current_edge_ids
        return {
            "prefix": int(prefix),
            "selected": int(len(selected)),
            "posthoc_good": int(len(good_ids)),
            "new_unique_edge_ids_vs_w2_core": int(len(new_ids)),
            "precision_additions": as_json_float(float(len(good_ids) / max(len(selected), 1))),
            "duplicate_rate_additions": as_json_float(float((len(good_ids) - len(unique)) / max(len(good_ids), 1))),
            "current_control_edge_recall": 1.0,
            "missing_face_boundary_edges_gained": int(len(new_ids & missing_final_boundary_edges)),
            "new_edge_ids_sample": sorted(int(value) for value in new_ids)[:80],
        }

    prefixes = [0, 20, 50, 100]
    prefix_frontier = [prefix_summary(min(prefix, len(ordered_additions))) for prefix in prefixes]
    best_prefix = sorted(
        prefix_frontier,
        key=lambda row: (
            -int(row.get("new_unique_edge_ids_vs_w2_core") or 0),
            -finite_float(row.get("precision_additions"), 0.0),
            finite_float(row.get("duplicate_rate_additions"), 1.0),
            int(row.get("prefix") or 0),
        ),
    )[0]
    edge_gate = {
        "current_core_recall_preserved": True,
        "best_prefix": int(best_prefix.get("prefix") or 0),
        "best_prefix_new_unique_edge_ids": int(best_prefix.get("new_unique_edge_ids_vs_w2_core") or 0),
        "best_prefix_new_unique_boundary_edge_ids": int(best_prefix.get("missing_face_boundary_edges_gained") or 0),
        "best_prefix_precision_additions": best_prefix.get("precision_additions"),
        "best_prefix_duplicate_rate_additions": best_prefix.get("duplicate_rate_additions"),
        "dependency_declaration_non_oracle": True,
        "passed_model_local": bool(
            int(best_prefix.get("missing_face_boundary_edges_gained") or 0) >= 5
            and finite_float(best_prefix.get("precision_additions"), 0.0) >= 0.5
            and finite_float(best_prefix.get("duplicate_rate_additions"), 1.0) <= 0.25
        ),
    }
    return {
        "scope": "generic-edge-additive-control",
        "model": str(model_name),
        "production_changed": False,
        "primitive_cap_audit": {
            "cap": int(primitive_materialization_cap),
            "raw_segments": int(w2_segment_count),
            "cap_primitives": int(len(capped_primitives)),
            "full_primitives": int(len(all_primitives)),
            "discarded_primitives": int(len(discarded_primitives)),
            "source_composition_before_cap": source_mode_counts(all_primitives),
            "source_composition_after_cap": source_mode_counts(capped_primitives),
            "source_composition_discarded_by_cap": source_mode_counts(discarded_primitives),
            "distributions_before_cap": primitive_distribution(all_primitives),
            "distributions_after_cap": primitive_distribution(capped_primitives),
            "distributions_discarded_by_cap": primitive_distribution(discarded_primitives),
            "posthoc_unique_edge_ids_cap_sample": int(len(cap_primitive_ids)),
            "posthoc_unique_edge_ids_full_sample": int(len(full_primitive_ids)),
            "posthoc_unique_edge_ids_lost_by_cap_sample": int(len(full_primitive_ids - cap_primitive_ids)),
            "current_control_edges_lost_by_cap_sample": int(len(current_edge_ids - cap_primitive_ids)),
            "cap_pool_match_cap": int(cap_match_cap),
            "full_pool_match_cap": int(full_match_cap),
        },
        "immutable_current_w2_core": {
            "core_clusters": int(len(current_core)),
            "posthoc_unique_edge_ids": int(len(current_edge_ids)),
            "core_sample": current_core[:80],
        },
        "duplicate_anatomy": {
            "oracle_duplicate_edge_assignments_in_top_evaluated": int(sum(max(0, len(indices) - 1) for indices in edge_to_indices.values())),
            "pair_relation_counts": {key: int(value) for key, value in sorted(duplicate_counts.items())},
        },
        "consolidation": {
            "input_hypotheses": int(len(consolidation_rows)),
            "groups": int(len(groups)),
            "group_size_distribution": distribution([float(len(group)) for group in groups]),
            "representative_comparison": rep_summaries,
            "selected_representative": selected_mode,
            "group_sample": [
                {
                    "group_id": str(additions[index].get("additive_group_id")),
                    "group_size": int(additions[index].get("group_size") or 0),
                    "representative_id": str(additions[index].get("representative_id")),
                    "core_relation": str(additions[index].get("core_relation")),
                    "rejection_reason": additions[index].get("rejection_reason"),
                }
                for index in range(min(len(additions), 80))
            ],
        },
        "dedupe_against_core": {
            "groups": int(len(additions)),
            "surviving_additions": int(len(survivors)),
            "rejection_reasons": {key: int(value) for key, value in sorted(rejection_reasons.items())},
            "relation_counts": {
                relation: int(sum(str(row.get("core_relation")) == relation for row in additions))
                for relation in sorted({str(row.get("core_relation")) for row in additions})
            },
        },
        "additive_ranking": {
            "mode": "sequential_marginal_non_oracle",
            "prefixes": prefix_frontier,
            "ordered_addition_sample": [
                {key: row.get(key) for key in ["additive_group_id", "representative_id", "core_relation", "score", "posthoc_edge_id", "posthoc_classification", "is_posthoc_good"]}
                for row in ordered_additions[:100]
            ],
        },
        "edge_gate": edge_gate,
        "union_face_adjacency": {
            "attempted": False,
            "reason": "additive edge gate must pass on pear+cushion before forming union face pairs",
        },
        "face_candidate_evaluation": {
            "attempted": False,
            "reason": "edge gate failed or not yet evaluated at aggregate level",
        },
        "full_edge_clip_trials": {
            "attempted": False,
            "reason": "not run because additive diagnostic is gated by edge frontier first",
        },
        "branch_if_gate_fails": "additive_generic_reservoir_no_safe_non_oracle_frontier",
        "timing_seconds": as_json_float(time.perf_counter() - started),
    }


def summarize_rows_by_category(rows: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for row in rows:
        cat = str(row.get("normal_failure_category") or "other")
        counts[cat] = counts.get(cat, 0) + 1
    total = max(1, len(rows))
    return {
        "total": int(len(rows)),
        "counts": {k: int(v) for k, v in sorted(counts.items())},
        "rates": {k: as_json_float(float(v / total)) for k, v in sorted(counts.items())},
    }
