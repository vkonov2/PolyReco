from __future__ import annotations

import hashlib
import json
import math
import time

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
    "model_edges_from_faces",
    "oracle_edge_cluster_matches",
    "edge_match_lookup",
    "point_edge_distances_vectorized",
    "oracle_run_edge_match",
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
    "candidates_plane_patch_compatible",
    "annotate_candidate_pool_neutral",
    "prepare_z_level_slices",
    "candidate_halfspace",
    "LpActivityEngine",
    "lp_face_activity_details",
    "reconstruct_polyhedron_from_halfspaces_edge_clip",
    "polygon_area",
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

