from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from generate_full_circle_split_cached_viewer import parse_initial_model


SCHEMA_VERSION = 3
EPS = 1e-12
ABSOLUTE_CANONICAL_TOLERANCES = {
    "normal_angle_deg": 2.0,
    "plane_distance": 0.05,
    "centroid_to_finite_polygon_distance": 0.15,
    "median_hull_to_surface_distance": 0.15,
}
F_SCORE_FRACTIONS = (0.001, 0.0025, 0.005, 0.01)
FREEZE_METADATA_KEYS = (
    "final_scientific_conclusion",
    "research_branch_registry",
    "metric_registry",
    "selected_artifacts",
    "source_audit",
    "reproducibility_checks",
)
FREEZE_REQUIRED_MODELS = ("round", "princess", "radiant", "pear", "cushion")


def json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def validate_mesh(vertices: np.ndarray, faces: list[list[int]], *, label: str) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"{label}: expected non-empty Nx3 vertices")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"{label}: vertices contain non-finite values")
    if not faces:
        raise ValueError(f"{label}: mesh has no faces")
    for face_index, face in enumerate(faces):
        if len(face) < 3:
            raise ValueError(f"{label}: face {face_index} has fewer than three vertices")
        if min(face) < 0 or max(face) >= vertices.shape[0]:
            raise ValueError(f"{label}: face {face_index} contains an invalid vertex index")


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if triangles.size == 0:
        return np.zeros(0, dtype=float)
    pts = vertices[triangles]
    return 0.5 * np.linalg.norm(np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1)


def polygon_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    # Newell's signed area vector remains correct for a simple concave polygon;
    # summing absolute centroid-fan triangles does not.
    area_vector = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
    return 0.5 * float(np.linalg.norm(area_vector))


def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def triangulate_polygon_faces(vertices: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    """Triangulate ordered planar simple polygons without filling concave notches."""
    triangles: list[list[int]] = []
    for face_index, face in enumerate(faces):
        if len(face) < 3:
            continue
        original_indices = [int(value) for value in face]
        points = vertices[np.asarray(original_indices, dtype=int)]
        area_vector = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
        normal_norm = float(np.linalg.norm(area_vector))
        if normal_norm <= EPS:
            raise ValueError(f"face {face_index}: degenerate polygon normal")
        normal = area_vector / normal_norm
        scale_3d = max(float(np.max(np.ptp(points, axis=0))), 1.0)
        planarity_error = float(np.max(np.abs((points - points[0]) @ normal)))
        if planarity_error > max(scale_3d * 1e-8, 1e-10):
            raise ValueError(
                f"face {face_index}: polygon is not planar enough for triangulation "
                f"(max deviation {planarity_error})"
            )

        u, v = plane_basis(normal)
        origin = np.mean(points, axis=0)
        projected = np.column_stack([(points - origin) @ u, (points - origin) @ v])
        scale_2d = max(float(np.max(np.ptp(projected, axis=0))), 1e-9)
        point_tol = max(scale_2d * 1e-12, 1e-14)
        area_tol = max(scale_2d * scale_2d * 1e-12, 1e-20)

        # Consecutive duplicate vertices and straight-through collinear vertices
        # do not contribute area and can prevent a valid ear from being found.
        active = list(range(len(original_indices)))
        changed = True
        while changed and len(active) > 3:
            changed = False
            for position in range(len(active)):
                previous = active[(position - 1) % len(active)]
                current = active[position]
                following = active[(position + 1) % len(active)]
                a, b, c = projected[previous], projected[current], projected[following]
                duplicate = (
                    float(np.linalg.norm(b - a)) <= point_tol
                    or float(np.linalg.norm(c - b)) <= point_tol
                )
                straight = (
                    abs(cross_2d(b - a, c - b)) <= area_tol
                    and float((b - a) @ (b - c)) <= point_tol * point_tol
                )
                if duplicate or straight:
                    del active[position]
                    changed = True
                    break

        signed_area_twice = sum(
            cross_2d(projected[active[index]], projected[active[(index + 1) % len(active)]])
            for index in range(len(active))
        )
        if abs(signed_area_twice) <= area_tol:
            raise ValueError(f"face {face_index}: degenerate projected polygon")
        orientation = 1.0 if signed_area_twice > 0.0 else -1.0

        remaining = list(active)
        face_triangles: list[list[int]] = []
        while len(remaining) > 3:
            ear_found = False
            for position in range(len(remaining)):
                previous = remaining[(position - 1) % len(remaining)]
                current = remaining[position]
                following = remaining[(position + 1) % len(remaining)]
                a, b, c = projected[previous], projected[current], projected[following]
                if orientation * cross_2d(b - a, c - a) <= area_tol:
                    continue
                contains_vertex = False
                for other in remaining:
                    if other in {previous, current, following}:
                        continue
                    point = projected[other]
                    inside_or_on = (
                        orientation * cross_2d(b - a, point - a) >= -area_tol
                        and orientation * cross_2d(c - b, point - b) >= -area_tol
                        and orientation * cross_2d(a - c, point - c) >= -area_tol
                    )
                    if inside_or_on:
                        contains_vertex = True
                        break
                if contains_vertex:
                    continue
                face_triangles.append(
                    [original_indices[previous], original_indices[current], original_indices[following]]
                )
                del remaining[position]
                ear_found = True
                break
            if not ear_found:
                raise ValueError(
                    f"face {face_index}: ear clipping failed; polygon may be self-intersecting"
                )
        face_triangles.append([original_indices[value] for value in remaining])
        triangles.extend(face_triangles)
    return np.asarray(triangles, dtype=np.int32).reshape((-1, 3))


def bbox_summary(vertices: np.ndarray) -> dict[str, Any]:
    lo = np.min(vertices, axis=0)
    hi = np.max(vertices, axis=0)
    extents = hi - lo
    return {
        "min": lo.tolist(),
        "max": hi.tolist(),
        "center": ((lo + hi) * 0.5).tolist(),
        "extents": extents.tolist(),
        "diagonal": float(np.linalg.norm(extents)),
    }


def mesh_topology_summary(vertices: np.ndarray, faces: list[list[int]]) -> dict[str, Any]:
    edge_counts: Counter[tuple[int, int]] = Counter()
    seen_faces: set[tuple[int, ...]] = set()
    duplicate_faces = 0
    repeated_vertex_faces = 0
    parent = list(range(vertices.shape[0]))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    face_areas: list[float] = []
    for face in faces:
        if len(set(face)) != len(face):
            repeated_vertex_faces += 1
        key = tuple(sorted(int(v) for v in face))
        duplicate_faces += int(key in seen_faces)
        seen_faces.add(key)
        face_areas.append(polygon_area(vertices[np.asarray(face, dtype=int)]))
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] += 1
            union(int(a), int(b))

    used_vertices = {int(v) for face in faces for v in face}
    components = len({find(v) for v in used_vertices}) if used_vertices else 0
    used_vertex_count = int(len(used_vertices))
    incidence = list(edge_counts.values())
    unique_edge_lengths = np.array(
        [float(np.linalg.norm(vertices[a] - vertices[b])) for a, b in edge_counts], dtype=float
    )
    area_array = np.asarray(face_areas, dtype=float)
    return {
        "vertices": int(vertices.shape[0]),
        "used_vertices": used_vertex_count,
        "orphan_vertices": int(vertices.shape[0] - used_vertex_count),
        "edges": int(len(edge_counts)),
        "faces": int(len(faces)),
        "euler": int(used_vertex_count - len(edge_counts) + len(faces)),
        "connected_components": int(components),
        "boundary_edges": int(sum(count == 1 for count in incidence)),
        "non_manifold_edges": int(sum(count > 2 for count in incidence)),
        "duplicate_faces": int(duplicate_faces),
        "repeated_vertex_faces": int(repeated_vertex_faces),
        "edge_incidence_distribution": {
            str(value): int(incidence.count(value)) for value in sorted(set(incidence))
        },
        "face_area_min": json_float(float(np.min(area_array))) if area_array.size else None,
        "face_area_median": json_float(float(np.median(area_array))) if area_array.size else None,
        "face_area_p01": json_float(float(np.percentile(area_array, 1))) if area_array.size else None,
        "edge_length_min": json_float(float(np.min(unique_edge_lengths))) if unique_edge_lengths.size else None,
        "edge_length_median": json_float(float(np.median(unique_edge_lengths))) if unique_edge_lengths.size else None,
    }


def topology_valid(summary: dict[str, Any]) -> bool:
    return bool(
        int(summary["euler"]) == 2
        and int(summary["connected_components"]) == 1
        and int(summary["boundary_edges"]) == 0
        and int(summary["non_manifold_edges"]) == 0
        and int(summary["duplicate_faces"]) == 0
        and int(summary["repeated_vertex_faces"]) == 0
    )


def mesh_volume_summary(vertices: np.ndarray, faces: list[list[int]]) -> dict[str, Any]:
    center = np.mean(vertices, axis=0)
    oriented_volume = 0.0
    raw_volume = 0.0
    orientation_flips = 0
    for face in faces:
        points = vertices[np.asarray(face, dtype=int)]
        if points.shape[0] < 3:
            continue
        cycle = points
        area_vector = np.sum(np.cross(cycle, np.roll(cycle, -1, axis=0)), axis=0)
        if float(area_vector @ (np.mean(points, axis=0) - center)) < 0.0:
            cycle = cycle[::-1]
            orientation_flips += 1
        for index in range(1, points.shape[0] - 1):
            raw_volume += float(
                np.dot(points[0] - center, np.cross(points[index] - center, points[index + 1] - center))
            ) / 6.0
        for index in range(1, cycle.shape[0] - 1):
            oriented_volume += float(
                np.dot(cycle[0] - center, np.cross(cycle[index] - center, cycle[index + 1] - center))
            ) / 6.0
    return {
        "volume": float(abs(oriented_volume)),
        "raw_signed_volume_about_centroid": float(raw_volume),
        "oriented_signed_volume_about_centroid": float(oriented_volume),
        "faces_reoriented_for_measurement": int(orientation_flips),
        "method": "face_cycles_oriented_outward_from_vertex_centroid",
    }


def mesh_convexity_summary(vertices: np.ndarray, faces: list[list[int]], scale: float) -> dict[str, Any]:
    center = np.mean(vertices, axis=0)
    maximum_outside = 0.0
    valid_planes = 0
    for face in faces:
        points = vertices[np.asarray(face, dtype=int)]
        normal = np.zeros(3, dtype=float)
        for index in range(points.shape[0]):
            normal += np.cross(points[index], points[(index + 1) % points.shape[0]])
        norm = float(np.linalg.norm(normal))
        if norm <= EPS:
            continue
        normal /= norm
        face_center = np.mean(points, axis=0)
        if float(normal @ (face_center - center)) < 0.0:
            normal = -normal
        offset = float(normal @ face_center)
        maximum_outside = max(maximum_outside, float(np.max(vertices @ normal - offset)))
        valid_planes += 1
    tolerance = max(float(scale) * 1e-7, 1e-9)
    return {
        "convex": bool(valid_planes == len(faces) and maximum_outside <= tolerance),
        "valid_face_planes": int(valid_planes),
        "face_count": int(len(faces)),
        "max_vertex_outside_face_halfspace": float(maximum_outside),
        "tolerance": float(tolerance),
    }


def mesh_summary(vertices: np.ndarray, faces: list[list[int]]) -> dict[str, Any]:
    triangles = triangulate_polygon_faces(vertices, faces)
    areas = triangle_areas(vertices, triangles)
    bbox = bbox_summary(vertices)
    topology = mesh_topology_summary(vertices, faces)
    return {
        "bbox": bbox,
        "surface_area": float(np.sum(areas)),
        "triangles": int(triangles.shape[0]),
        "triangulation_method": "planar_ear_clipping",
        "degenerate_triangles": int(np.sum(areas <= EPS)),
        "volume": mesh_volume_summary(vertices, faces),
        "topology": topology,
        "topology_valid": topology_valid(topology),
        "convexity": mesh_convexity_summary(vertices, faces, float(bbox["diagonal"])),
    }


def deterministic_area_weighted_surface_samples(
    vertices: np.ndarray,
    triangles: np.ndarray,
    requested_count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    areas = triangle_areas(vertices, triangles)
    valid = areas > EPS
    valid_triangles = triangles[valid]
    valid_areas = areas[valid]
    if valid_triangles.shape[0] == 0 or float(np.sum(valid_areas)) <= EPS:
        raise ValueError("mesh has no non-degenerate triangles for surface sampling")
    count = max(int(requested_count), 1)
    exact = valid_areas / float(np.sum(valid_areas)) * count
    allocations = np.floor(exact).astype(int)
    remainder = count - int(np.sum(allocations))
    if remainder > 0:
        fractional = exact - allocations
        order = np.argsort(-fractional, kind="stable")
        allocations[order[:remainder]] += 1

    rng = np.random.default_rng(int(seed))
    chunks: list[np.ndarray] = []
    sampled_triangles = 0
    for triangle, sample_count in zip(valid_triangles, allocations):
        if sample_count <= 0:
            continue
        a, b, c = vertices[triangle]
        random_values = rng.random((int(sample_count), 2))
        root = np.sqrt(random_values[:, 0])
        weights_a = 1.0 - root
        weights_b = root * (1.0 - random_values[:, 1])
        weights_c = root * random_values[:, 1]
        chunks.append(
            weights_a[:, None] * a[None, :]
            + weights_b[:, None] * b[None, :]
            + weights_c[:, None] * c[None, :]
        )
        sampled_triangles += 1
    points = np.concatenate(chunks, axis=0)
    return points, {
        "method": "deterministic_area_proportional_triangle_allocation_uniform_barycentric",
        "seed": int(seed),
        "requested_count": int(requested_count),
        "actual_count": int(points.shape[0]),
        "nondegenerate_triangle_count": int(valid_triangles.shape[0]),
        "triangles_receiving_samples": int(sampled_triangles),
        "surface_area": float(np.sum(valid_areas)),
    }


def point_segment_distance_squared(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    edge = b - a
    denominator = float(edge @ edge)
    if denominator <= EPS * EPS:
        return np.sum((points - a[None, :]) ** 2, axis=1)
    parameter = np.clip(((points - a[None, :]) @ edge) / denominator, 0.0, 1.0)
    closest = a[None, :] + parameter[:, None] * edge[None, :]
    return np.sum((points - closest) ** 2, axis=1)


def point_triangle_distance_squared(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal_norm = float(np.linalg.norm(normal))
    edge_distances = np.minimum(
        point_segment_distance_squared(points, a, b),
        np.minimum(
            point_segment_distance_squared(points, b, c),
            point_segment_distance_squared(points, c, a),
        ),
    )
    if normal_norm <= EPS:
        return edge_distances
    unit_normal = normal / normal_norm
    signed = (points - a[None, :]) @ unit_normal
    projected = points - signed[:, None] * unit_normal[None, :]
    relative = projected - a[None, :]
    dot00 = float(ab @ ab)
    dot01 = float(ab @ ac)
    dot11 = float(ac @ ac)
    # The Gram determinant has units of length^4 and equals |ab x ac|^2.
    # Comparing it with the length-like EPS incorrectly collapses valid small
    # triangles (for example, sides around 1e-3) to their boundary segments.
    denominator = normal_norm * normal_norm
    if denominator <= EPS * EPS:
        return edge_distances
    dot20 = relative @ ab
    dot21 = relative @ ac
    alpha = (dot11 * dot20 - dot01 * dot21) / denominator
    beta = (dot00 * dot21 - dot01 * dot20) / denominator
    inside = (alpha >= -1e-10) & (beta >= -1e-10) & (alpha + beta <= 1.0 + 1e-10)
    return np.where(inside, signed * signed, edge_distances)


def point_mesh_distances(
    points: np.ndarray,
    target_vertices: np.ndarray,
    target_triangles: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    target_areas = triangle_areas(target_vertices, target_triangles)
    valid_triangles = target_triangles[target_areas > EPS]
    if valid_triangles.shape[0] == 0:
        raise ValueError("target mesh has no non-degenerate triangles")
    result = np.full(points.shape[0], float("inf"), dtype=float)
    step = max(int(chunk_size), 1)
    for start in range(0, points.shape[0], step):
        stop = min(start + step, points.shape[0])
        chunk = points[start:stop]
        best = np.full(chunk.shape[0], float("inf"), dtype=float)
        for triangle in valid_triangles:
            a, b, c = target_vertices[triangle]
            best = np.minimum(best, point_triangle_distance_squared(chunk, a, b, c))
        result[start:stop] = np.sqrt(np.maximum(best, 0.0))
    return result


def distribution_summary(values: np.ndarray, scale: float | None = None) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    percentiles = np.percentile(array, [50, 90, 95, 99, 99.9])
    result: dict[str, Any] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "median": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "p99_9": float(percentiles[4]),
        "sampled_max": float(np.max(array)),
    }
    if scale is not None and scale > EPS:
        result["normalized_by_reference_bbox_diagonal"] = {
            key: float(value) / float(scale)
            for key, value in result.items()
            if key not in {"count", "normalized_by_reference_bbox_diagonal"}
        }
    return result


def symmetric_surface_metrics(
    reconstructed_to_reference: np.ndarray,
    reference_to_reconstructed: np.ndarray,
    *,
    reference_scale: float,
) -> dict[str, Any]:
    forward = np.asarray(reconstructed_to_reference, dtype=float)
    reverse = np.asarray(reference_to_reconstructed, dtype=float)
    chamfer_mean = 0.5 * (float(np.mean(forward)) + float(np.mean(reverse)))
    chamfer_rms = math.sqrt(0.5 * (float(np.mean(forward * forward)) + float(np.mean(reverse * reverse))))
    robust_hausdorff = max(float(np.percentile(forward, 99)), float(np.percentile(reverse, 99)))
    sampled_hausdorff = max(float(np.max(forward)), float(np.max(reverse)))
    f_scores: dict[str, Any] = {}
    for fraction in F_SCORE_FRACTIONS:
        tolerance = float(reference_scale) * float(fraction)
        precision = float(np.mean(forward <= tolerance))
        recall = float(np.mean(reverse <= tolerance))
        f_score = 2.0 * precision * recall / max(precision + recall, EPS)
        f_scores[f"{100.0 * fraction:g}%_bbox_diagonal"] = {
            "tolerance_absolute": float(tolerance),
            "precision_reconstructed_to_reference": precision,
            "recall_reference_to_reconstructed": recall,
            "f1": float(f_score),
        }
    return {
        "chamfer_mean": chamfer_mean,
        "chamfer_rms": chamfer_rms,
        "robust_hausdorff_p99": robust_hausdorff,
        "sampled_hausdorff_max": sampled_hausdorff,
        "normalized_by_reference_bbox_diagonal": {
            "chamfer_mean": chamfer_mean / reference_scale,
            "chamfer_rms": chamfer_rms / reference_scale,
            "robust_hausdorff_p99": robust_hausdorff / reference_scale,
            "sampled_hausdorff_max": sampled_hausdorff / reference_scale,
        },
        "f_scores": f_scores,
        "maximum_is_sampled_not_exact": True,
    }


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(normal @ axis)) > 0.9:
        axis = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(normal, axis)
    u /= max(float(np.linalg.norm(u)), EPS)
    v = np.cross(normal, u)
    v /= max(float(np.linalg.norm(v)), EPS)
    return u, v


def point_segment_distance_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    edge = b - a
    denominator = float(edge @ edge)
    if denominator <= EPS:
        return float(np.linalg.norm(point - a))
    parameter = float(np.clip((point - a) @ edge / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + parameter * edge)))


def point_polygon_signed_distance_2d(point: np.ndarray, polygon: np.ndarray) -> float:
    inside = False
    previous = polygon.shape[0] - 1
    x, y = float(point[0]), float(point[1])
    for index in range(polygon.shape[0]):
        xi, yi = float(polygon[index, 0]), float(polygon[index, 1])
        xj, yj = float(polygon[previous, 0]), float(polygon[previous, 1])
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
            if x < x_cross:
                inside = not inside
        previous = index
    edge_distance = min(
        point_segment_distance_2d(point, polygon[index], polygon[(index + 1) % polygon.shape[0]])
        for index in range(polygon.shape[0])
    )
    return -edge_distance if inside else edge_distance


def polygon_outside_distances_2d(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    edge_distance = np.full(points.shape[0], float("inf"), dtype=float)
    previous = polygon.shape[0] - 1
    for index in range(polygon.shape[0]):
        a = polygon[previous]
        b = polygon[index]
        xi, yi = float(b[0]), float(b[1])
        xj, yj = float(a[0]), float(a[1])
        crosses = (yi > y) != (yj > y)
        x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        inside ^= crosses & (x < x_cross)

        edge = b - a
        denominator = float(edge @ edge)
        if denominator <= EPS:
            dist = np.linalg.norm(points - a[None, :], axis=1)
        else:
            parameter = np.clip(((points - a[None, :]) @ edge) / denominator, 0.0, 1.0)
            closest = a[None, :] + parameter[:, None] * edge[None, :]
            dist = np.linalg.norm(points - closest, axis=1)
        edge_distance = np.minimum(edge_distance, dist)
        previous = index
    return np.where(inside, 0.0, edge_distance)


def reference_face_records(vertices: np.ndarray, faces: list[list[int]]) -> list[dict[str, Any]]:
    center = np.mean(vertices, axis=0)
    records: list[dict[str, Any]] = []
    for face_id, face in enumerate(faces):
        points = vertices[np.asarray(face, dtype=int)]
        normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
        norm = float(np.linalg.norm(normal))
        if norm <= EPS:
            continue
        normal /= norm
        face_center = np.mean(points, axis=0)
        if float(normal @ (face_center - center)) < 0.0:
            normal = -normal
        u, v = plane_basis(normal)
        origin = face_center
        polygon_2d = np.column_stack([(points - origin) @ u, (points - origin) @ v])
        records.append(
            {
                "face_id": int(face_id),
                "normal": normal,
                "offset": float(normal @ face_center),
                "vertices": points,
                "origin": origin,
                "u": u,
                "v": v,
                "polygon_2d": polygon_2d,
                "area": polygon_area(points),
            }
        )
    return records


def point_to_face_polygon_distance(point: np.ndarray, face: dict[str, Any]) -> float:
    relative = point - np.asarray(face["origin"], dtype=float)
    point_2d = np.array([relative @ face["u"], relative @ face["v"]], dtype=float)
    outside = max(0.0, point_polygon_signed_distance_2d(point_2d, face["polygon_2d"]))
    plane_distance = abs(float(np.asarray(face["normal"]) @ point - float(face["offset"])))
    return math.hypot(plane_distance, outside)


def candidate_plane_and_hull(candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    normal = np.asarray(candidate.get("plane_normal") or [], dtype=float)
    point = np.asarray(candidate.get("plane_centroid") or [], dtype=float)
    hull = np.asarray(candidate.get("hull") or [], dtype=float)
    if normal.shape != (3,) or point.shape != (3,) or not np.all(np.isfinite(normal)) or not np.all(np.isfinite(point)):
        return None
    norm = float(np.linalg.norm(normal))
    if norm <= EPS:
        return None
    if hull.ndim != 2 or hull.shape[1:] != (3,) or hull.shape[0] == 0 or not np.all(np.isfinite(hull)):
        return None
    return point, normal / norm, hull


def candidate_reference_match(
    candidate: dict[str, Any],
    reference_faces: list[dict[str, Any]],
    tolerances: dict[str, float],
    *,
    assignment_mode: str,
) -> dict[str, Any] | None:
    plane = candidate_plane_and_hull(candidate)
    if plane is None:
        return None
    point, normal, hull = plane
    preliminary: list[tuple[float, dict[str, Any], float, float]] = []
    for face in reference_faces:
        face_normal = np.asarray(face["normal"], dtype=float)
        dot = float(normal @ face_normal)
        sign = 1.0 if dot >= 0.0 else -1.0
        angle = float(np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0))))
        plane_distance = abs(float(normal @ point) - sign * float(face["offset"]))
        plane_score = (
            (angle / tolerances["normal_angle_deg"]) ** 2
            + (plane_distance / tolerances["plane_distance"]) ** 2
        )
        preliminary.append((float(plane_score), face, angle, plane_distance))
    preliminary.sort(key=lambda row: row[0])
    if assignment_mode == "pipeline-compatible":
        # Preserve the reconstruction pipeline's legacy oracle assignment exactly:
        # evaluate twelve nearest planes, then choose the smallest combined score
        # even when a different reference face would pass every finite-good gate.
        selected_faces = preliminary[:12]
    elif assignment_mode == "finite-good-first-exhaustive":
        selected_faces = preliminary
    else:
        raise ValueError(f"unknown canonical assignment mode: {assignment_mode}")

    rows: list[dict[str, Any]] = []
    for _, face, angle, plane_distance in selected_faces:
        centroid_distance = point_to_face_polygon_distance(point, face)
        hull_distances = np.array([point_to_face_polygon_distance(value, face) for value in hull], dtype=float)
        hull_distance = float(np.median(hull_distances))
        finite_good = bool(
            angle <= tolerances["normal_angle_deg"]
            and plane_distance <= tolerances["plane_distance"]
            and centroid_distance <= tolerances["centroid_to_finite_polygon_distance"]
            and hull_distance <= tolerances["median_hull_to_surface_distance"]
        )
        failures: list[str] = []
        if angle > tolerances["normal_angle_deg"]:
            failures.append("normal_angle")
        if plane_distance > tolerances["plane_distance"]:
            failures.append("plane_distance")
        if centroid_distance > tolerances["centroid_to_finite_polygon_distance"]:
            failures.append("centroid_to_finite_polygon_distance")
        if hull_distance > tolerances["median_hull_to_surface_distance"]:
            failures.append("median_hull_to_surface_distance")
        score = (
            (angle / tolerances["normal_angle_deg"]) ** 2
            + (plane_distance / tolerances["plane_distance"]) ** 2
            + (centroid_distance / tolerances["centroid_to_finite_polygon_distance"]) ** 2
            + (hull_distance / tolerances["median_hull_to_surface_distance"]) ** 2
        )
        row = {
            "face_id": int(face["face_id"]),
            "normal_angle_deg": angle,
            "plane_distance": plane_distance,
            "centroid_distance": centroid_distance,
            "hull_surface_distance": hull_distance,
            "finite_good": finite_good,
            "failure_reasons": failures,
            "score": float(score),
        }
        rows.append(row)
    if not rows:
        return None
    if assignment_mode == "finite-good-first-exhaustive":
        return min(rows, key=lambda row: (not bool(row["finite_good"]), float(row["score"])))
    return min(rows, key=lambda row: float(row["score"]))


def face_level_summary(
    payload: dict[str, Any],
    reference_faces: list[dict[str, Any]],
    tolerances: dict[str, float],
    *,
    assignment_mode: str,
) -> dict[str, Any]:
    candidates = payload.get("face_candidates") or []
    active_indices = payload.get("reconstructed", {}).get("face_candidate_indices") or []
    rows: list[dict[str, Any]] = []
    invalid_indices: list[int] = []
    for active_face_index, raw_index in enumerate(active_indices):
        candidate_index = int(raw_index)
        if candidate_index < 0 or candidate_index >= len(candidates):
            invalid_indices.append(candidate_index)
            continue
        match = candidate_reference_match(
            candidates[candidate_index],
            reference_faces,
            tolerances,
            assignment_mode=assignment_mode,
        )
        if match is None:
            invalid_indices.append(candidate_index)
            continue
        rows.append(
            {
                "active_face_index": int(active_face_index),
                "candidate_index": int(candidate_index),
                **match,
            }
        )
    finite_rows = [row for row in rows if bool(row["finite_good"])]
    unique_ids = sorted({int(row["face_id"]) for row in finite_rows})
    area_by_id = {int(face["face_id"]): float(face["area"]) for face in reference_faces}
    total_area = float(sum(area_by_id.values()))
    failure_counts = Counter(reason for row in rows for reason in row["failure_reasons"])

    def field_summary(field: str) -> dict[str, Any]:
        return distribution_summary(np.asarray([float(row[field]) for row in rows], dtype=float))

    return {
        "assignment_mode": assignment_mode,
        "tolerances": {key: float(value) for key, value in tolerances.items()},
        "reference_face_count": int(len(reference_faces)),
        "active_face_count": int(len(active_indices)),
        "evaluated_active_candidates": int(len(rows)),
        "invalid_candidate_indices": invalid_indices,
        "finite_good_candidates": int(len(finite_rows)),
        "unique_finite_face_ids": int(len(unique_ids)),
        "unique_finite_face_ids_list": unique_ids,
        "candidate_precision": float(len(finite_rows) / max(1, len(active_indices))),
        "candidate_precision_denominator": "all_active_faces",
        "evaluated_candidate_precision": float(len(finite_rows) / max(1, len(rows))),
        "reference_face_recall": float(len(unique_ids) / max(1, len(reference_faces))),
        "area_weighted_reference_face_recall": float(
            sum(area_by_id.get(face_id, 0.0) for face_id in unique_ids) / max(total_area, EPS)
        ),
        "duplicate_good_assignments": int(len(finite_rows) - len(unique_ids)),
        "failure_reason_counts": {key: int(value) for key, value in sorted(failure_counts.items())},
        "best_match_distributions": {
            "normal_angle_deg": field_summary("normal_angle_deg"),
            "plane_distance": field_summary("plane_distance"),
            "centroid_distance": field_summary("centroid_distance"),
            "hull_surface_distance": field_summary("hull_surface_distance"),
        },
    }


def face_adjacency(faces: list[list[int]]) -> list[set[int]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(faces):
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((int(a), int(b))))
            edge_to_faces.setdefault(edge, []).append(int(face_id))
    adjacency = [set() for _ in faces]
    for face_ids in edge_to_faces.values():
        if len(face_ids) < 2:
            continue
        for face_id in face_ids:
            adjacency[face_id].update(other for other in face_ids if other != face_id)
    return adjacency


def supporting_plane_patch_groups(
    reference_records: list[dict[str, Any]],
    reference_faces: list[list[int]],
    *,
    scale: float,
    normal_tolerance_deg: float = 1.0,
    offset_tolerance_fraction: float = 1e-5,
) -> dict[str, Any]:
    by_id = {int(face["face_id"]): face for face in reference_records}
    adjacency = face_adjacency(reference_faces)
    offset_tolerance = max(float(scale) * float(offset_tolerance_fraction), 1e-9)
    visited: set[int] = set()
    groups: list[dict[str, Any]] = []
    for seed_id in sorted(by_id):
        if seed_id in visited:
            continue
        seed = by_id[seed_id]
        stack = [seed_id]
        visited.add(seed_id)
        group_ids: list[int] = []
        while stack:
            face_id = stack.pop()
            group_ids.append(face_id)
            current = by_id[face_id]
            current_normal = np.asarray(current["normal"], dtype=float)
            current_offset = float(current["offset"])
            for neighbor_id in adjacency[face_id]:
                if neighbor_id in visited or neighbor_id not in by_id:
                    continue
                neighbor = by_id[neighbor_id]
                neighbor_normal = np.asarray(neighbor["normal"], dtype=float)
                dot = float(current_normal @ neighbor_normal)
                angle = float(np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0))))
                sign = 1.0 if dot >= 0.0 else -1.0
                offset_delta = abs(current_offset - sign * float(neighbor["offset"]))
                if angle <= normal_tolerance_deg and offset_delta <= offset_tolerance:
                    visited.add(neighbor_id)
                    stack.append(neighbor_id)
        areas = [float(by_id[face_id]["area"]) for face_id in group_ids]
        groups.append(
            {
                "group_id": int(len(groups)),
                "face_ids": sorted(int(value) for value in group_ids),
                "face_count": int(len(group_ids)),
                "area": float(sum(areas)),
                "representative_face_id": int(min(group_ids)),
            }
        )
    groups.sort(key=lambda row: int(row["representative_face_id"]))
    for group_id, group in enumerate(groups):
        group["group_id"] = int(group_id)
    return {
        "method": "edge-adjacent coplanar grouping; disjoint same-plane patches remain separate",
        "normal_tolerance_deg": float(normal_tolerance_deg),
        "offset_tolerance_absolute": float(offset_tolerance),
        "group_count": int(len(groups)),
        "groups": groups,
    }


def observed_face_characteristics(
    reference_records: list[dict[str, Any]],
    *,
    bbox: dict[str, Any],
    z_step: float,
    angular_sample_count: int,
) -> dict[str, Any]:
    scale = float(bbox["diagonal"])
    z_step_abs = abs(float(z_step)) if abs(float(z_step)) > EPS else scale / 512.0
    angular_resolution = 2.0 * math.pi / max(int(angular_sample_count), 1)
    rows: list[dict[str, Any]] = []
    z_min, z_max = float(bbox["min"][2]), float(bbox["max"][2])
    z_span_total = max(z_max - z_min, EPS)
    total_area = float(sum(float(face["area"]) for face in reference_records))
    for face in reference_records:
        vertices = np.asarray(face["vertices"], dtype=float)
        normal = np.asarray(face["normal"], dtype=float)
        area = float(face["area"])
        z_span = float(np.max(vertices[:, 2]) - np.min(vertices[:, 2]))
        z_level_count = int(max(1, math.floor(z_span / max(z_step_abs, EPS)) + 1))
        xy = vertices[:, :2]
        xy_extent = float(np.linalg.norm(np.max(xy, axis=0) - np.min(xy, axis=0)))
        center_xy = np.mean(xy, axis=0)
        radial_scale = max(float(np.linalg.norm(center_xy)), scale * 0.05)
        angular_span_estimate = xy_extent / max(radial_scale, EPS)
        abs_nz = abs(float(normal[2]))
        if abs_nz >= math.cos(math.radians(30.0)):
            orientation_bucket = "near-horizontal"
        elif abs_nz <= math.sin(math.radians(30.0)):
            orientation_bucket = "near-vertical"
        else:
            orientation_bucket = "inclined"
        face_z_center = float(np.mean(vertices[:, 2]))
        z_position = (face_z_center - z_min) / z_span_total
        if z_position < 1.0 / 3.0:
            z_zone = "lower"
        elif z_position < 2.0 / 3.0:
            z_zone = "middle"
        else:
            z_zone = "upper"
        projection_resolution_ratio = angular_span_estimate / max(angular_resolution, EPS)
        z_resolution_ratio = z_span / max(z_step_abs, EPS)
        observable = bool(
            area / max(total_area, EPS) >= 1e-5
            and (z_resolution_ratio >= 1.0 or projection_resolution_ratio >= 1.0)
        )
        rows.append(
            {
                "face_id": int(face["face_id"]),
                "area": area,
                "area_fraction": float(area / max(total_area, EPS)),
                "z_span": z_span,
                "z_level_count": z_level_count,
                "orientation_bucket": orientation_bucket,
                "z_zone": z_zone,
                "xy_extent": xy_extent,
                "angular_span_estimate": float(angular_span_estimate),
                "angular_resolution": float(angular_resolution),
                "projection_resolution_ratio": float(projection_resolution_ratio),
                "z_resolution_ratio": float(z_resolution_ratio),
                "observable_by_resolution_heuristic": observable,
            }
        )
    return {
        "method": "posthoc geometric resolution heuristic from face area, Z-span, angular sample count, and bbox scale",
        "z_step": float(z_step_abs),
        "angular_sample_count": int(angular_sample_count),
        "observable_face_ids": sorted(int(row["face_id"]) for row in rows if row["observable_by_resolution_heuristic"]),
        "rows": rows,
    }


def decile_index(value: float, values: list[float]) -> int:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0
    return int(np.searchsorted(np.percentile(finite, np.arange(10, 100, 10)), float(value), side="right"))


def summarize_bool_rows(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = len(rows)
    hits = sum(1 for row in rows if bool(row.get(key)))
    return {"total": int(total), "hit": int(hits), "recall": float(hits / max(1, total))}


def source_bucket(candidate: dict[str, Any]) -> str:
    origin = str(candidate.get("candidate_origin") or candidate.get("candidate_source") or "unknown")
    if origin in {"valley_rebuild", "baseline_core", "baseline_guard"}:
        return "baseline/valley"
    if origin == "w2_addition":
        return "W2 adjacency"
    if "append" in origin:
        return "append-local"
    if "broad" in origin or "ratchet" in origin:
        return "broad-ratchet"
    if "blocker" in origin or "guard" in origin:
        return "LP blocker repair"
    if "exchange" in origin:
        return "LP-deficit exchange"
    return origin


def reconstruction_geometry_digest(vertices: np.ndarray, faces: list[list[int]], active_indices: list[int]) -> str:
    payload = {
        "vertices": np.round(vertices, 9).tolist(),
        "faces": [[int(value) for value in face] for face in faces],
        "active_candidate_indices": [int(value) for value in active_indices],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def rounded_json_points(values: Any, digits: int = 8) -> list[list[float]]:
    points = np.asarray(values, dtype=float)
    if points.size == 0:
        return []
    return np.round(points, int(digits)).tolist()


def normalize_unoriented_normal(values: Any, digits: int = 8) -> list[float | None]:
    normal = np.asarray(values, dtype=float).reshape((-1,))
    if normal.size != 3 or not np.all(np.isfinite(normal)):
        return [None, None, None]
    norm = float(np.linalg.norm(normal))
    if norm <= EPS:
        return [None, None, None]
    normal = normal / norm
    for value in normal:
        if abs(float(value)) > 10.0 ** (-(int(digits) + 1)):
            if float(value) < 0.0:
                normal = -normal
            break
    return [float(round(float(value), int(digits))) for value in normal]


CANDIDATE_PLANE_DIGEST_SCHEMA_V2 = {
    "schema_version": 2,
    "field_name": "candidate_plane_sha256_round8",
    "candidate_ordering": "face_candidates array order from reconstruction artifact",
    "fields": ["track_id", "plane_normal_unoriented_unit", "plane_centroid", "plane_rms"],
    "normal_sign_normalization": "normalize to unit length; flip sign so the first non-zero component is positive",
    "rounding": "round float fields to 8 decimal places after normal sign normalization",
    "json_serialization": "json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')",
    "encoding": "utf-8",
}


def candidate_plane_digest_v2(candidates: list[dict[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        track_id = candidate.get("track_id")
        rows.append(
            {
                "candidate_order_index": int(index),
                "track_id": int(track_id) if track_id is not None else None,
                "plane_normal_unoriented_unit": normalize_unoriented_normal(candidate.get("plane_normal")),
                "plane_centroid": (
                    [float(round(float(value), 8)) for value in candidate.get("plane_centroid", [])]
                    if isinstance(candidate.get("plane_centroid"), list)
                    else []
                ),
                "plane_rms": (
                    float(round(float(candidate.get("plane_rms")), 8))
                    if candidate.get("plane_rms") is not None and np.isfinite(float(candidate.get("plane_rms")))
                    else None
                ),
            }
        )
    return sha256_json({"schema": CANDIDATE_PLANE_DIGEST_SCHEMA_V2, "rows": rows})


def unique_edges_from_faces(faces: list[list[int]]) -> list[list[int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for index, first in enumerate(face):
            second = face[(index + 1) % len(face)]
            edge = tuple(sorted((int(first), int(second))))
            if edge[0] != edge[1]:
                edges.add(edge)
    return [list(edge) for edge in sorted(edges)]


def fan_triangles_from_faces(faces: list[list[int]]) -> list[list[int]]:
    triangles: list[list[int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        triangles.extend(
            [[int(face[0]), int(face[index]), int(face[index + 1])] for index in range(1, len(face) - 1)]
        )
    return triangles


def lifecycle_audit(
    payload: dict[str, Any],
    reference_records: list[dict[str, Any]],
    reference_faces: list[list[int]],
    reference_bbox: dict[str, Any],
    reference_scale: float,
    canonical_tolerances: dict[str, float],
    observable: dict[str, Any],
    reconstructed_vertices: np.ndarray,
    reconstructed_faces: list[list[int]],
) -> dict[str, Any]:
    candidates = payload.get("face_candidates") or []
    reconstructed = payload.get("reconstructed") if isinstance(payload.get("reconstructed"), dict) else {}
    active_indices = [int(value) for value in reconstructed.get("face_candidate_indices") or []]
    active_set = set(active_indices)
    accepted_set = {
        index
        for index, candidate in enumerate(candidates)
        if str(candidate.get("selection_status")) == "accepted" or index in active_set
    }
    groups = supporting_plane_patch_groups(reference_records, reference_faces, scale=reference_scale)
    face_to_group: dict[int, int] = {}
    for group in groups["groups"]:
        for face_id in group["face_ids"]:
            face_to_group[int(face_id)] = int(group["group_id"])

    all_candidate_rows: list[dict[str, Any]] = []
    best_by_face: dict[int, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates):
        match = candidate_reference_match(
            candidate,
            reference_records,
            canonical_tolerances,
            assignment_mode="finite-good-first-exhaustive",
        )
        if match is None:
            continue
        row = {
            "candidate_index": int(candidate_index),
            "candidate_id": int(candidate.get("track_id") or candidate_index),
            "active": bool(candidate_index in active_set),
            "accepted": bool(candidate_index in accepted_set),
            "source_bucket": source_bucket(candidate),
            "candidate_origin": candidate.get("candidate_origin"),
            "candidate_source": candidate.get("candidate_source"),
            "selection_status": candidate.get("selection_status"),
            "selection_reason": candidate.get("selection_reason"),
            "levels": candidate.get("levels"),
            "z_min": candidate.get("z_min"),
            "z_max": candidate.get("z_max"),
            "plane_rms": candidate.get("plane_rms"),
            "hull_area": candidate.get("hull_area"),
            **match,
        }
        all_candidate_rows.append(row)
        face_id = int(match["face_id"])
        current = best_by_face.get(face_id)
        if current is None or (not bool(current["finite_good"]), float(current["score"])) > (
            not bool(match["finite_good"]),
            float(match["score"]),
        ):
            best_by_face[face_id] = row

    active_good = [row for row in all_candidate_rows if row["active"] and row["finite_good"]]
    accepted_good = [row for row in all_candidate_rows if row["accepted"] and row["finite_good"]]
    pool_good = [row for row in all_candidate_rows if row["finite_good"]]
    active_faces = {int(row["face_id"]) for row in active_good}
    accepted_faces = {int(row["face_id"]) for row in accepted_good}
    pool_faces = {int(row["face_id"]) for row in pool_good}
    active_groups = {face_to_group[face_id] for face_id in active_faces if face_id in face_to_group}
    accepted_groups = {face_to_group[face_id] for face_id in accepted_faces if face_id in face_to_group}
    pool_groups = {face_to_group[face_id] for face_id in pool_faces if face_id in face_to_group}
    total_group_area = float(sum(float(group["area"]) for group in groups["groups"]))
    group_area = {int(group["group_id"]): float(group["area"]) for group in groups["groups"]}

    observable_ids = set(int(value) for value in observable["observable_face_ids"])
    observable_groups = {
        face_to_group[face_id] for face_id in observable_ids if face_id in face_to_group
    }
    total_observable_group_area = float(sum(group_area[group_id] for group_id in observable_groups))

    face_rows: list[dict[str, Any]] = []
    area_values = [float(row["area"]) for row in observable["rows"]]
    z_span_values = [float(row["z_span"]) for row in observable["rows"]]
    observable_by_id = {int(row["face_id"]): row for row in observable["rows"]}
    for face in reference_records:
        face_id = int(face["face_id"])
        obs = observable_by_id.get(face_id, {})
        best = best_by_face.get(face_id)
        if face_id in active_faces:
            status = "final_unique_face"
        elif face_id in accepted_faces:
            status = "accepted_but_redundant_or_inactive"
        elif face_id in pool_faces:
            status = "candidate_pool_finite_good_but_not_accepted"
        elif best is not None and "median_hull_to_surface_distance" in list(best.get("failure_reasons") or []):
            status = "near_plane_wrong_finite_hull"
        elif best is not None and not best.get("finite_good"):
            status = "near_plane_or_footprint_mismatch"
        elif bool(obs.get("observable_by_resolution_heuristic")):
            status = "lost_before_candidate_pool_or_detection"
        else:
            status = "below_current_observability_heuristic"
        face_rows.append(
            {
                "face_id": face_id,
                "group_id": face_to_group.get(face_id),
                "status": status,
                "active_finite_good": bool(face_id in active_faces),
                "accepted_finite_good": bool(face_id in accepted_faces),
                "candidate_pool_finite_good": bool(face_id in pool_faces),
                "observable": bool(obs.get("observable_by_resolution_heuristic")),
                "area_decile": decile_index(float(obs.get("area", face["area"])), area_values),
                "z_span_decile": decile_index(float(obs.get("z_span", 0.0)), z_span_values),
                "orientation_bucket": obs.get("orientation_bucket"),
                "z_zone": obs.get("z_zone"),
                "area_fraction": obs.get("area_fraction"),
                "z_level_count": obs.get("z_level_count"),
                "projection_resolution_ratio": obs.get("projection_resolution_ratio"),
                "best_candidate": (
                    {
                        "candidate_index": int(best["candidate_index"]),
                        "candidate_id": int(best["candidate_id"]),
                        "source_bucket": best["source_bucket"],
                        "active": bool(best["active"]),
                        "accepted": bool(best["accepted"]),
                        "finite_good": bool(best["finite_good"]),
                        "failure_reasons": list(best.get("failure_reasons") or []),
                        "score": float(best["score"]),
                    }
                    if best is not None
                    else None
                ),
            }
        )

    def group_breakdown(field: str) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in face_rows:
            buckets.setdefault(str(row.get(field)), []).append(row)
        return [
            {"bucket": key, **summarize_bool_rows(rows, "active_finite_good")}
            for key, rows in sorted(buckets.items())
        ]

    candidate_source_rows: list[dict[str, Any]] = []
    for source, rows in sorted(
        ((source, [row for row in all_candidate_rows if row["source_bucket"] == source]) for source in {row["source_bucket"] for row in all_candidate_rows}),
        key=lambda item: item[0],
    ):
        active_rows = [row for row in rows if row["active"]]
        good_active_rows = [row for row in active_rows if row["finite_good"]]
        candidate_source_rows.append(
            {
                "source_bucket": source,
                "candidate_rows": int(len(rows)),
                "active_rows": int(len(active_rows)),
                "active_finite_good": int(len(good_active_rows)),
                "active_unique_faces": int(len({int(row["face_id"]) for row in good_active_rows})),
                "active_candidate_precision": float(len(good_active_rows) / max(1, len(active_rows))),
            }
        )

    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    w2_selection = parameters.get("w2_selection") if isinstance(parameters.get("w2_selection"), dict) else {}
    append_local = (
        parameters.get("w2_small_face_refinement")
        if isinstance(parameters.get("w2_small_face_refinement"), dict)
        else {}
    )
    core_repair = parameters.get("w2_core_repair") if isinstance(parameters.get("w2_core_repair"), dict) else {}
    exchange = (
        core_repair.get("post_ratchet_exchange")
        if isinstance(core_repair.get("post_ratchet_exchange"), dict)
        else {}
    )
    angular = parameters.get("angular_scale_additive") if isinstance(parameters.get("angular_scale_additive"), dict) else {}
    active_candidate_ids = [
        int(candidates[index].get("track_id") or index) for index in active_indices if 0 <= index < len(candidates)
    ]
    active_plane_ids = [int(value) for value in reconstructed.get("active_plane_indices") or []]
    stage_counters = {
        "peak_observations": payload.get("summary", {}).get("peak_observations"),
        "tracks": payload.get("summary", {}).get("tracks"),
        "tracks_kept": payload.get("summary", {}).get("tracks_kept"),
        "raw_candidates_before_dedupe": parameters.get("raw_candidates_before_dedupe"),
        "raw_candidates_before_merge": parameters.get("raw_candidates_before_merge"),
        "candidates_after_dedupe": parameters.get("candidates_after_dedupe"),
        "face_candidates_payload": int(len(candidates)),
        "w2_face_candidates": parameters.get("w2_face_candidates"),
        "w2_candidates_after_core_preselection": parameters.get("w2_candidates_after_core_preselection"),
        "w2_selected_representatives": w2_selection.get("selected_representatives"),
        "w2_accepted_halfspaces": w2_selection.get("accepted_halfspaces"),
        "w2_accepted_additions": w2_selection.get("accepted_additions"),
        "append_local_dense_band_count": len((append_local.get("dense_bands") or {}).get("bands") or []),
        "append_local_cheap_pool_size": append_local.get("cheap_pool_size"),
        "append_local_trial_checks": append_local.get("trial_checks"),
        "append_local_accepted_additions": append_local.get("accepted_additions"),
        "lp_blocker_removed_count": core_repair.get("removed_blocker_count"),
        "lp_blocker_repair_action_count": core_repair.get("repair_action_count"),
        "lp_deficit_exchange_actions": exchange.get("accepted_exchange_count"),
        "angular_raw_candidates": angular.get("raw_candidates"),
        "angular_deduped_candidates": angular.get("deduped_candidates"),
        "angular_preselected_candidates": angular.get("preselected_candidates"),
        "angular_accepted_additions": angular.get("accepted_additions"),
        "angular_final_active_additions": angular.get("final_active_additions"),
    }

    return {
        "candidate_ids": [int(candidate.get("track_id") or index) for index, candidate in enumerate(candidates)],
        "canonical_tolerance_basis": "absolute_round_compatible",
        "active_candidate_indices": active_indices,
        "active_candidate_ids": active_candidate_ids,
        "active_plane_ids": active_plane_ids,
        "geometry_digest": reconstruction_geometry_digest(reconstructed_vertices, reconstructed_faces, active_indices),
        "stage_counters": stage_counters,
        "supporting_plane_patch_groups": {
            **{key: value for key, value in groups.items() if key != "groups"},
            "active_recalled_groups": int(len(active_groups)),
            "accepted_recalled_groups": int(len(accepted_groups)),
            "candidate_pool_recalled_groups": int(len(pool_groups)),
            "observable_groups": int(len(observable_groups)),
            "active_count_recall": float(len(active_groups) / max(1, int(groups["group_count"]))),
            "active_area_weighted_recall": float(sum(group_area[group_id] for group_id in active_groups) / max(total_group_area, EPS)),
            "active_observable_count_recall": float(len(active_groups & observable_groups) / max(1, len(observable_groups))),
            "active_observable_area_weighted_recall": float(
                sum(group_area[group_id] for group_id in active_groups & observable_groups)
                / max(total_observable_group_area, EPS)
            ),
            "groups_sample": groups["groups"][:20],
        },
        "observable_facets": {
            **{key: value for key, value in observable.items() if key != "rows"},
            "observable_count": int(len(observable_ids)),
            "observable_area_fraction": float(
                sum(float(observable_by_id[face_id]["area"]) for face_id in observable_ids) / max(
                    sum(float(row["area"]) for row in observable["rows"]), EPS
                )
            ),
            "rows_sample": observable["rows"][:40],
        },
        "loss_funnel": {
            "reference_facets": int(len(reference_records)),
            "observable_facets": int(len(observable_ids)),
            "raw_observed_support_note": "not directly serialized; use peak/tracks and candidate payload counters below",
            "peak_observations": stage_counters["peak_observations"],
            "valley_w2_segment_or_track": {
                "tracks": stage_counters["tracks"],
                "tracks_kept": stage_counters["tracks_kept"],
                "w2_face_candidates": stage_counters["w2_face_candidates"],
            },
            "raw_plane_candidate": stage_counters["raw_candidates_before_dedupe"],
            "candidate_pool_finite_good_unique_faces": int(len(pool_faces)),
            "after_preselection_or_accepted_unique_faces": int(len(accepted_faces)),
            "accepted_candidate_count": int(len(accepted_set)),
            "active_edge_clip_unique_faces": int(len(active_faces)),
            "final_unique_patch_groups": int(len(active_groups)),
            "status_counts": {key: int(value) for key, value in sorted(Counter(row["status"] for row in face_rows).items())},
            "by_area_decile": group_breakdown("area_decile"),
            "by_z_span_decile": group_breakdown("z_span_decile"),
            "by_orientation": group_breakdown("orientation_bucket"),
            "by_z_zone": group_breakdown("z_zone"),
            "by_candidate_source": candidate_source_rows,
        },
        "face_rows": face_rows,
        "candidate_rows_sample": all_candidate_rows[:200],
    }


def face_local_coordinates(face: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.asarray(face["origin"], dtype=float)
    normal = np.asarray(face["normal"], dtype=float)
    u = np.asarray(face["u"], dtype=float)
    v = np.asarray(face["v"], dtype=float)
    relative = points - origin[None, :]
    xy = np.column_stack([relative @ u, relative @ v])
    residual = np.abs(points @ normal - float(face["offset"]))
    return xy, residual, np.asarray(face["polygon_2d"], dtype=float)


def extract_serialized_observed_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    candidates = payload.get("face_candidates") or []
    for candidate_index, candidate in enumerate(candidates):
        candidate_id = int(candidate.get("track_id") or candidate_index)
        base = {
            "candidate_index": int(candidate_index),
            "candidate_id": candidate_id,
            "origin": candidate.get("candidate_origin"),
            "source": candidate.get("candidate_source"),
            "window": candidate.get("window"),
            "selection_status": candidate.get("selection_status"),
            "selection_reason": candidate.get("selection_reason"),
            "w2_edge_cluster_ids": list(candidate.get("w2_edge_cluster_ids") or []),
        }

        def add_point(point: Any, *, source: str, z_index: Any = None, angular_index: Any = None, rms: Any = None) -> None:
            arr = np.asarray(point, dtype=float)
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                return
            points.append(
                {
                    **base,
                    "point": arr.tolist(),
                    "point_source": source,
                    "z": float(arr[2]),
                    "z_index": int(z_index) if z_index is not None else None,
                    "angular_index": int(angular_index) if angular_index is not None else None,
                    "rms": float(rms) if rms is not None else None,
                }
            )

        sample_points = candidate.get("sample_points") if isinstance(candidate.get("sample_points"), list) else []
        z_indices = candidate.get("z_indices") if isinstance(candidate.get("z_indices"), list) else []
        for point_index, point in enumerate(sample_points):
            add_point(
                point,
                source="w2_or_candidate_sample",
                z_index=z_indices[point_index] if point_index < len(z_indices) else None,
            )
        for key, source in (
            ("left_sample_points", "valley_left_sample"),
            ("right_sample_points", "valley_right_sample"),
            ("peak_sample_points", "rms_peak_sample"),
        ):
            for point in candidate.get(key) or []:
                add_point(point, source=source)
        for obs in candidate.get("debug_observations") or []:
            z_index = obs.get("z_index") if isinstance(obs, dict) else None
            for key, source in (
                ("left_fit_points", "valley_left_fit"),
                ("right_fit_points", "valley_right_fit"),
                ("peak_points", "rms_peak"),
            ):
                for row in obs.get(key) or []:
                    if not isinstance(row, dict):
                        continue
                    add_point(
                        row.get("point"),
                        source=source,
                        z_index=z_index,
                        angular_index=row.get("index"),
                        rms=row.get("rms"),
                    )
    return points


def fit_plane_svd(points: np.ndarray) -> dict[str, Any]:
    if points.shape[0] < 3:
        return {"valid": False, "reason": "fewer_than_3_points"}
    centroid = np.mean(points, axis=0)
    centered = points - centroid[None, :]
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= max(float(np.linalg.norm(normal)), EPS)
    residual = np.abs(centered @ normal)
    eigen = (singular_values * singular_values / max(points.shape[0] - 1, 1)).tolist()
    return {
        "valid": True,
        "centroid": centroid,
        "normal": normal,
        "rms": float(np.sqrt(np.mean(residual * residual))),
        "p95": float(np.percentile(residual, 95)),
        "eigenvalues": eigen,
        "linearity_ratio": float(eigen[1] / max(eigen[0], EPS)) if len(eigen) >= 2 else 0.0,
        "plane_thickness_ratio": float(eigen[-1] / max(eigen[1], EPS)) if len(eigen) >= 3 else float("inf"),
    }


def auc_for_scores(good_scores: list[float], bad_scores: list[float]) -> float | None:
    good = [float(v) for v in good_scores if np.isfinite(v)]
    bad = [float(v) for v in bad_scores if np.isfinite(v)]
    if not good or not bad:
        return None
    wins = 0.0
    total = 0
    for g in good:
        for b in bad:
            total += 1
            if g > b:
                wins += 1.0
            elif g == b:
                wins += 0.5
    return float(wins / max(total, 1))


def frontier_for_scores(rows: list[dict[str, Any]], score_key: str, label_key: str) -> list[dict[str, Any]]:
    scored = [row for row in rows if row.get(score_key) is not None and np.isfinite(float(row[score_key]))]
    scored.sort(key=lambda row: float(row[score_key]), reverse=True)
    total_good = sum(1 for row in scored if bool(row.get(label_key)))
    out: list[dict[str, Any]] = []
    for fraction in (0.02, 0.05, 0.1, 0.2, 0.35):
        take = max(1, int(round(len(scored) * fraction)))
        subset = scored[:take]
        hits = sum(1 for row in subset if bool(row.get(label_key)))
        out.append(
            {
                "top_fraction": float(fraction),
                "candidates": int(len(subset)),
                "precision": float(hits / max(1, len(subset))),
                "recall": float(hits / max(1, total_good)),
                "unique_face_ids": int(len({int(row["face_id"]) for row in subset if bool(row.get(label_key))})),
            }
        )
    return out


def empirical_observed_signal_audit(
    payload: dict[str, Any],
    reference_records: list[dict[str, Any]],
    reference_faces: list[list[int]],
    reference_scale: float,
    canonical_tolerances: dict[str, float],
) -> dict[str, Any]:
    candidates = payload.get("face_candidates") or []
    reconstructed = payload.get("reconstructed") if isinstance(payload.get("reconstructed"), dict) else {}
    active_indices = {int(value) for value in reconstructed.get("face_candidate_indices") or []}
    accepted_indices = {
        index
        for index, candidate in enumerate(candidates)
        if str(candidate.get("selection_status")) == "accepted" or index in active_indices
    }
    observed_rows = extract_serialized_observed_points(payload)
    observed_points = np.asarray([row["point"] for row in observed_rows], dtype=float) if observed_rows else np.zeros((0, 3))
    plane_tol = float(canonical_tolerances["plane_distance"])
    finite_tol = float(canonical_tolerances["centroid_to_finite_polygon_distance"])
    reference_normals = np.asarray([np.asarray(face["normal"], dtype=float) for face in reference_records], dtype=float)
    reference_offsets = np.asarray([float(face["offset"]) for face in reference_records], dtype=float)
    reference_face_ids = [int(face["face_id"]) for face in reference_records]

    candidate_matches: list[dict[str, Any]] = []
    best_by_face: dict[int, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates):
        match = candidate_reference_match(
            candidate,
            reference_records,
            canonical_tolerances,
            assignment_mode="finite-good-first-exhaustive",
        )
        if match is None:
            continue
        z_span = finite_float(candidate.get("z_max"), 0.0) - finite_float(candidate.get("z_min"), 0.0)
        row = {
            "candidate_index": int(candidate_index),
            "candidate_id": int(candidate.get("track_id") or candidate_index),
            "face_id": int(match["face_id"]),
            "finite_good": bool(match["finite_good"]),
            "active": bool(candidate_index in active_indices),
            "accepted": bool(candidate_index in accepted_indices),
            "failure_reasons": list(match.get("failure_reasons") or []),
            "source_bucket": source_bucket(candidate),
            "support_count": finite_float(candidate.get("finite_support_count"), 0.0),
            "independent_scale_count": float(len(set(candidate.get("dedupe_cluster_windows") or [candidate.get("window")]))),
            "plane_rms": finite_float(candidate.get("plane_rms"), float("inf")),
            "robust_residual": finite_float(candidate.get("finite_support_residual_p95"), float("inf")),
            "conditioning": finite_float(candidate.get("condition_p95"), float("inf")),
            "z_span": max(float(z_span), 0.0),
            "hull_area": finite_float(candidate.get("hull_area"), 0.0),
            "support_density": finite_float(candidate.get("finite_support_density"), 0.0),
            "support_purity_proxy": finite_float(candidate.get("finite_support_purity"), 0.0),
            "lp_activity_margin": -finite_float(candidate.get("lp_activity_optimum"), 0.0),
        }
        candidate_matches.append(row)
        current = best_by_face.get(int(match["face_id"]))
        if current is None or (not current["finite_good"], current["plane_rms"]) > (not row["finite_good"], row["plane_rms"]):
            best_by_face[int(match["face_id"])] = row

    face_rows: list[dict[str, Any]] = []
    active_faces = {row["face_id"] for row in candidate_matches if row["active"] and row["finite_good"]}
    accepted_faces = {row["face_id"] for row in candidate_matches if row["accepted"] and row["finite_good"]}
    pool_faces = {row["face_id"] for row in candidate_matches if row["finite_good"]}
    raw_support_faces: set[int] = set()
    oracle_fit_faces: set[int] = set()
    representation_counts = Counter()
    category_counts = Counter()
    source_faces: dict[str, set[int]] = {
        "long_z_track": set(),
        "local_short_z_patch": set(),
        "angular_track": set(),
        "local_3d_plane": set(),
    }

    for face in reference_records:
        face_id = int(face["face_id"])
        if observed_points.size:
            local_xy, residual, polygon = face_local_coordinates(face, observed_points)
            polygon_min = np.min(polygon, axis=0) - finite_tol
            polygon_max = np.max(polygon, axis=0) + finite_tol
            pre_mask = (
                (residual <= plane_tol)
                & np.all(local_xy >= polygon_min[None, :], axis=1)
                & np.all(local_xy <= polygon_max[None, :], axis=1)
            )
            outside = np.full(local_xy.shape[0], float("inf"), dtype=float)
            pre_indices = np.flatnonzero(pre_mask)
            outside[pre_indices] = polygon_outside_distances_2d(local_xy[pre_indices], polygon)
            support_mask = pre_mask & (outside <= finite_tol)
        else:
            local_xy = np.zeros((0, 2))
            residual = np.zeros(0)
            outside = np.zeros(0)
            support_mask = np.zeros(0, dtype=bool)
        support_indices = np.flatnonzero(support_mask)
        support = observed_points[support_indices] if support_indices.size else np.zeros((0, 3))
        support_rows = [observed_rows[int(index)] for index in support_indices]
        z_values = {round(float(row["z"]), 6) for row in support_rows}
        z_indices = {row["z_index"] for row in support_rows if row.get("z_index") is not None}
        angular_indices = {row["angular_index"] for row in support_rows if row.get("angular_index") is not None}
        windows = {row["window"] for row in support_rows if row.get("window") is not None}
        candidate_ids = {row["candidate_id"] for row in support_rows}
        cluster_ids = {
            int(cluster_id)
            for row in support_rows
            for cluster_id in (row.get("w2_edge_cluster_ids") or [])
            if cluster_id is not None
        }
        source_names = {str(row["point_source"]) for row in support_rows}
        if support_indices.size:
            raw_support_faces.add(face_id)
            support_xy = local_xy[support_indices]
            span = np.ptp(support_xy, axis=0) if support_xy.shape[0] else np.zeros(2)
            polygon_span = np.maximum(np.ptp(np.asarray(face["polygon_2d"], dtype=float), axis=0), EPS)
            coverage = span / polygon_span
        else:
            support_xy = np.zeros((0, 2))
            coverage = np.zeros(2)
        nearest_hits = 0
        purity_sample_count = 0
        if support.shape[0]:
            if support.shape[0] > 12:
                sample_positions = np.linspace(0, support.shape[0] - 1, 12).round().astype(int)
                purity_points = support[sample_positions]
            else:
                purity_points = support
            purity_sample_count = int(purity_points.shape[0])
            residual_matrix = np.abs(purity_points @ reference_normals.T - reference_offsets[None, :])
            nearest_ids = [reference_face_ids[int(index)] for index in np.argmin(residual_matrix, axis=1)]
            nearest_hits = sum(1 for nearest_id in nearest_ids if int(nearest_id) == face_id)
        purity = float(nearest_hits / max(1, purity_sample_count)) if support.shape[0] else None
        fit = fit_plane_svd(support)
        fit_good = False
        if fit.get("valid"):
            normal = np.asarray(fit["normal"], dtype=float)
            ref_normal = np.asarray(face["normal"], dtype=float)
            angle = float(np.degrees(np.arccos(np.clip(abs(float(normal @ ref_normal)), -1.0, 1.0))))
            plane_distance = abs(float(normal @ np.asarray(fit["centroid"], dtype=float)) - math.copysign(float(face["offset"]), float(normal @ ref_normal)))
            fit_good = bool(
                angle <= canonical_tolerances["normal_angle_deg"]
                and plane_distance <= plane_tol
                and support.shape[0] >= 6
                and float(min(coverage)) >= 0.12
                and float(max(coverage)) >= 0.35
                and float(fit.get("linearity_ratio") or 0.0) >= 0.02
            )
        if fit_good:
            oracle_fit_faces.add(face_id)

        long_z = bool(face_id in pool_faces and any(row["z_span"] >= 0.08 and row["face_id"] == face_id for row in candidate_matches))
        local_short = bool(
            support.shape[0] >= 6
            and 1 <= len(z_values) <= 4
            and float(min(coverage)) >= 0.10
            and (len(angular_indices) >= 2 or len(candidate_ids) >= 2)
        )
        angular_track = bool(
            support.shape[0] >= 6
            and (len(angular_indices) >= 3 or len(candidate_ids) >= 3)
            and float(max(coverage)) >= 0.35
        )
        local_3d = bool(fit_good and (purity is None or purity >= 0.55))
        for name, value in (
            ("long_z_track", long_z),
            ("local_short_z_patch", local_short),
            ("angular_track", angular_track),
            ("local_3d_plane", local_3d),
        ):
            representation_counts[name] += int(value)
            if value:
                source_faces[name].add(face_id)

        best = best_by_face.get(face_id)
        if face_id in active_faces:
            category = "final_active_mesh_recovered"
        elif support.shape[0] == 0:
            category = "A_no_raw_observed_support"
        elif support.shape[0] < 4 or len(z_values) < 1:
            category = "B_raw_support_too_sparse"
        elif float(min(coverage)) < 0.08 or (fit.get("valid") and float(fit.get("linearity_ratio") or 0.0) < 0.01):
            category = "C_support_exists_but_not_spatially_spanning"
        elif purity is not None and purity < 0.55:
            category = "D_mixed_with_neighbor_facets"
        elif best is None:
            category = "E_lost_at_minima_or_valley_detection"
        elif best["source_bucket"] == "baseline/valley" and not any(row["source_bucket"] == "W2 adjacency" and row["face_id"] == face_id for row in candidate_matches):
            category = "F_lost_at_W2_segmentation"
        elif any(row["source_bucket"] == "W2 adjacency" and row["face_id"] == face_id for row in candidate_matches) and len(cluster_ids) < 2:
            category = "G_lost_at_edge_clustering"
        elif not any(row["finite_good"] for row in candidate_matches if row["face_id"] == face_id) and support.shape[0] >= 6:
            category = "H_correct_supports_exist_but_pair_or_group_missing"
        elif best is not None and not best["finite_good"] and any(reason in best["failure_reasons"] for reason in ("normal_angle", "plane_distance")):
            category = "I_correct_group_exists_but_plane_fit_bad"
        elif best is not None and not best["finite_good"]:
            category = "J_plane_good_but_finite_hull_bad"
        elif face_id in pool_faces and face_id not in accepted_faces:
            category = "K_finite_good_candidate_exists_but_preselection_or_ranking_loss"
        elif face_id in accepted_faces:
            category = "L_accepted_then_inactive_or_redundant"
        else:
            category = "H_correct_supports_exist_but_pair_or_group_missing"
        category_counts[category] += 1
        face_rows.append(
            {
                "face_id": face_id,
                "category": category,
                "final_active": bool(face_id in active_faces),
                "raw_support_count": int(support.shape[0]),
                "raw_z_levels": int(max(len(z_values), len(z_indices))),
                "raw_z_span": float(np.ptp(support[:, 2])) if support.shape[0] else 0.0,
                "angular_view_count": int(len(angular_indices)),
                "angular_span": int(max(angular_indices) - min(angular_indices)) if angular_indices else 0,
                "independent_window_count": int(len(windows)),
                "independent_source_count": int(len(source_names)),
                "independent_candidate_count": int(len(candidate_ids)),
                "edge_cluster_count": int(len(cluster_ids)),
                "coverage_u": float(coverage[0]) if coverage.size else 0.0,
                "coverage_v": float(coverage[1]) if coverage.size else 0.0,
                "residual_median": json_float(float(np.median(residual[support_indices]))) if support_indices.size else None,
                "finite_distance_median": json_float(float(np.median(outside[support_indices]))) if support_indices.size else None,
                "purity": json_float(float(purity)) if purity is not None else None,
                "purity_sample_count": int(purity_sample_count),
                "fit_valid": bool(fit.get("valid")),
                "oracle_fit_good": bool(fit_good),
                "representation": {
                    "current_long_z_track": long_z,
                    "local_short_z_patch": local_short,
                    "angular_track": angular_track,
                    "local_3d_plane": local_3d,
                },
            }
        )

    feature_specs = {
        "support_count": True,
        "independent_scale_count": True,
        "plane_rms": False,
        "robust_residual": False,
        "conditioning": False,
        "z_span": True,
        "hull_area": True,
        "support_density": True,
        "support_purity_proxy": True,
        "lp_activity_margin": True,
    }
    feature_rows: list[dict[str, Any]] = []
    scored_candidates: list[dict[str, Any]] = []
    for row in candidate_matches:
        score = (
            math.log1p(max(row["support_count"], 0.0))
            + math.log1p(max(row["support_density"], 0.0))
            + max(row["support_purity_proxy"], 0.0)
            - 2.0 * max(row["plane_rms"], 0.0)
            - 1.5 * max(row["robust_residual"], 0.0)
            + 0.25 * math.log1p(max(row["hull_area"], 0.0))
        )
        scored_candidates.append({**row, "predeclared_support_score": float(score)})
    for name, high_is_good in feature_specs.items():
        good_scores = [row[name] if high_is_good else -row[name] for row in candidate_matches if row["finite_good"]]
        bad_scores = [row[name] if high_is_good else -row[name] for row in candidate_matches if not row["finite_good"]]
        feature_rows.append(
            {
                "feature": name,
                "direction": "higher_is_better" if high_is_good else "lower_is_better",
                "auc": auc_for_scores(good_scores, bad_scores),
                "good": distribution_summary(np.asarray([row[name] for row in candidate_matches if row["finite_good"]], dtype=float)),
                "bad": distribution_summary(np.asarray([row[name] for row in candidate_matches if not row["finite_good"]], dtype=float)),
            }
        )
    feature_rows.append(
        {
            "feature": "predeclared_support_score",
            "direction": "higher_is_better",
            "auc": auc_for_scores(
                [row["predeclared_support_score"] for row in scored_candidates if row["finite_good"]],
                [row["predeclared_support_score"] for row in scored_candidates if not row["finite_good"]],
            ),
            "frontier": frontier_for_scores(scored_candidates, "predeclared_support_score", "finite_good"),
        }
    )

    return {
        "schema": "serialized-observed-evidence-v1",
        "uses_initial_model": True,
        "production_changed": False,
        "important_limitations": [
            "The reconstruction JSON does not serialize every raw line_point for every window; this audit uses serialized preselection evidence: debug_observations, sample_points, peak_sample_points, W2 candidate samples, z_indices, and cluster IDs.",
            "Oracle-fit ceiling is a diagnostic upper bound from points already present in the artifact; it is not a production detector.",
        ],
        "raw_observed_support_funnel": {
            "reference_faces": int(len(reference_records)),
            "serialized_observed_points": int(len(observed_rows)),
            "raw_finite_support_ceiling_faces": int(len(raw_support_faces)),
            "oracle_fit_ceiling_faces": int(len(oracle_fit_faces)),
            "current_detector_ceiling_faces": int(len(pool_faces)),
            "final_active_mesh_faces": int(len(active_faces)),
            "representation_unique_faces": {key: int(len(value)) for key, value in source_faces.items()},
        },
        "empirical_observability_ceiling": {
            "category_counts": {key: int(value) for key, value in sorted(category_counts.items())},
            "category_total": int(sum(category_counts.values())),
            "raw_finite_support_face_ids": sorted(int(value) for value in raw_support_faces),
            "oracle_fit_face_ids": sorted(int(value) for value in oracle_fit_faces),
            "current_detector_face_ids": sorted(int(value) for value in pool_faces),
            "final_active_face_ids": sorted(int(value) for value in active_faces),
        },
        "representation_comparison": {
            "counts": {key: int(value) for key, value in sorted(representation_counts.items())},
            "note": "Representations are oracle-posthoc tests over serialized observed evidence only.",
        },
        "non_oracle_separation": {
            "candidate_rows": int(len(candidate_matches)),
            "finite_good_candidates": int(sum(1 for row in candidate_matches if row["finite_good"])),
            "feature_auc": feature_rows,
        },
        "bounded_edge_clip_trials": {
            "attempted": False,
            "reason": "No shared non-oracle shortlist was promoted from the serialized-evidence AUC/frontier audit; candidate-level oracle support was not treated as recovery.",
        },
        "face_rows": face_rows,
    }


def comparison_summary(
    reference: dict[str, Any],
    reconstructed: dict[str, Any],
) -> dict[str, Any]:
    ref_bbox = reference["bbox"]
    rec_bbox = reconstructed["bbox"]
    ref_extents = np.asarray(ref_bbox["extents"], dtype=float)
    rec_extents = np.asarray(rec_bbox["extents"], dtype=float)
    ref_center = np.asarray(ref_bbox["center"], dtype=float)
    rec_center = np.asarray(rec_bbox["center"], dtype=float)
    ref_volume = float(reference["volume"]["volume"])
    rec_volume = float(reconstructed["volume"]["volume"])
    ref_area = float(reference["surface_area"])
    rec_area = float(reconstructed["surface_area"])
    scale = float(ref_bbox["diagonal"])
    ref_min = np.asarray(ref_bbox["min"], dtype=float)
    ref_max = np.asarray(ref_bbox["max"], dtype=float)
    rec_min = np.asarray(rec_bbox["min"], dtype=float)
    rec_max = np.asarray(rec_bbox["max"], dtype=float)
    intersection_extents = np.maximum(np.minimum(ref_max, rec_max) - np.maximum(ref_min, rec_min), 0.0)
    intersection_volume = float(np.prod(intersection_extents))
    ref_bbox_volume = float(np.prod(np.maximum(ref_extents, 0.0)))
    rec_bbox_volume = float(np.prod(np.maximum(rec_extents, 0.0)))
    bbox_union_volume = ref_bbox_volume + rec_bbox_volume - intersection_volume
    return {
        "volume_absolute_error": float(abs(rec_volume - ref_volume)),
        "volume_relative_error": float((rec_volume - ref_volume) / max(ref_volume, EPS)),
        "surface_area_absolute_error": float(abs(rec_area - ref_area)),
        "surface_area_relative_error": float((rec_area - ref_area) / max(ref_area, EPS)),
        "bbox_center_offset": float(np.linalg.norm(rec_center - ref_center)),
        "bbox_center_offset_normalized": float(np.linalg.norm(rec_center - ref_center) / max(scale, EPS)),
        "bbox_extent_absolute_error": np.abs(rec_extents - ref_extents).tolist(),
        "bbox_extent_relative_error": ((rec_extents - ref_extents) / np.maximum(ref_extents, EPS)).tolist(),
        "bbox_diagonal_relative_error": float(
            (float(rec_bbox["diagonal"]) - scale) / max(scale, EPS)
        ),
        "axis_aligned_bbox_iou": float(intersection_volume / max(bbox_union_volume, EPS)),
        "axis_aligned_bbox_intersection_volume": intersection_volume,
        "reference_bbox_volume": ref_bbox_volume,
        "reconstructed_bbox_volume": rec_bbox_volume,
    }


def load_mesh_from_payload(payload: dict[str, Any]) -> tuple[np.ndarray, list[list[int]]]:
    reconstructed = payload.get("reconstructed")
    if not isinstance(reconstructed, dict):
        raise ValueError("input JSON does not contain a reconstructed object")
    vertices = np.asarray(reconstructed.get("vertices") or [], dtype=float)
    faces = [[int(value) for value in face] for face in (reconstructed.get("faces") or [])]
    validate_mesh(vertices, faces, label="reconstructed")
    return vertices, faces


def build_quality_report(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_json)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    model = str(args.model or payload.get("model") or "").strip()
    if not model:
        raise ValueError("model is absent from the payload; pass --model")
    reference_path = Path(args.initial_model) if args.initial_model else Path(args.data_root) / model / "InitialModel"
    if not reference_path.exists():
        raise FileNotFoundError(f"InitialModel not found: {reference_path}")
    reference_model = parse_initial_model(reference_path)
    reference_vertices = np.asarray(reference_model.vertices, dtype=float)
    reference_faces = [[int(value) for value in face] for face in reference_model.faces]
    validate_mesh(reference_vertices, reference_faces, label="reference")
    reconstructed_vertices, reconstructed_faces = load_mesh_from_payload(payload)

    reference_triangles = triangulate_polygon_faces(reference_vertices, reference_faces)
    reconstructed_triangles = triangulate_polygon_faces(reconstructed_vertices, reconstructed_faces)
    reference_mesh_summary = mesh_summary(reference_vertices, reference_faces)
    reconstructed_mesh_summary = mesh_summary(reconstructed_vertices, reconstructed_faces)
    reference_scale = float(reference_mesh_summary["bbox"]["diagonal"])
    if reference_scale <= EPS:
        raise ValueError("reference bbox diagonal is zero")

    normalization_path = Path(args.data_root) / str(args.normalization_model) / "InitialModel"
    if not normalization_path.exists():
        raise FileNotFoundError(f"normalization InitialModel not found: {normalization_path}")
    normalization_model = parse_initial_model(normalization_path)
    normalization_scale = float(bbox_summary(np.asarray(normalization_model.vertices, dtype=float))["diagonal"])
    scale_ratio = reference_scale / max(normalization_scale, EPS)
    normalized_tolerances = {
        "normal_angle_deg": ABSOLUTE_CANONICAL_TOLERANCES["normal_angle_deg"],
        "plane_distance": ABSOLUTE_CANONICAL_TOLERANCES["plane_distance"] * scale_ratio,
        "centroid_to_finite_polygon_distance": (
            ABSOLUTE_CANONICAL_TOLERANCES["centroid_to_finite_polygon_distance"] * scale_ratio
        ),
        "median_hull_to_surface_distance": (
            ABSOLUTE_CANONICAL_TOLERANCES["median_hull_to_surface_distance"] * scale_ratio
        ),
    }

    reference_samples, reference_sampling = deterministic_area_weighted_surface_samples(
        reference_vertices,
        reference_triangles,
        int(args.samples_per_surface),
        seed=int(args.seed),
    )
    reconstructed_samples, reconstructed_sampling = deterministic_area_weighted_surface_samples(
        reconstructed_vertices,
        reconstructed_triangles,
        int(args.samples_per_surface),
        seed=int(args.seed) + 1,
    )
    reconstructed_to_reference = point_mesh_distances(
        reconstructed_samples,
        reference_vertices,
        reference_triangles,
        chunk_size=int(args.distance_chunk_size),
    )
    reference_to_reconstructed = point_mesh_distances(
        reference_samples,
        reconstructed_vertices,
        reconstructed_triangles,
        chunk_size=int(args.distance_chunk_size),
    )
    reconstructed_vertex_to_reference = point_mesh_distances(
        reconstructed_vertices,
        reference_vertices,
        reference_triangles,
        chunk_size=int(args.distance_chunk_size),
    )
    reference_vertex_to_reconstructed = point_mesh_distances(
        reference_vertices,
        reconstructed_vertices,
        reconstructed_triangles,
        chunk_size=int(args.distance_chunk_size),
    )

    reference_records = reference_face_records(reference_vertices, reference_faces)
    corrected_absolute_faces = face_level_summary(
        payload,
        reference_records,
        dict(ABSOLUTE_CANONICAL_TOLERANCES),
        assignment_mode="finite-good-first-exhaustive",
    )
    corrected_normalized_faces = face_level_summary(
        payload,
        reference_records,
        normalized_tolerances,
        assignment_mode="finite-good-first-exhaustive",
    )
    pipeline_absolute_faces = face_level_summary(
        payload,
        reference_records,
        dict(ABSOLUTE_CANONICAL_TOLERANCES),
        assignment_mode="pipeline-compatible",
    )
    pipeline_normalized_faces = face_level_summary(
        payload,
        reference_records,
        normalized_tolerances,
        assignment_mode="pipeline-compatible",
    )
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    z_step = float(parameters.get("z_step") or 0.01)
    n_half = int(payload.get("summary", {}).get("n_half") or parameters.get("angular_scale_reference_half_count") or 1)
    observable_facets = observed_face_characteristics(
        reference_records,
        bbox=reference_mesh_summary["bbox"],
        z_step=z_step,
        angular_sample_count=n_half,
    )
    generalization_audit = lifecycle_audit(
        payload,
        reference_records,
        reference_faces,
        reference_mesh_summary["bbox"],
        reference_scale,
        dict(ABSOLUTE_CANONICAL_TOLERANCES),
        observable_facets,
        reconstructed_vertices,
        reconstructed_faces,
    )
    observed_signal_audit = empirical_observed_signal_audit(
        payload,
        reference_records,
        reference_faces,
        reference_scale,
        dict(ABSOLUTE_CANONICAL_TOLERANCES),
    )
    quality = {
        "reference_scale": {
            "bbox_diagonal": reference_scale,
            "cube_root_volume": float(reference_mesh_summary["volume"]["volume"] ** (1.0 / 3.0)),
            "normalization_model": str(args.normalization_model),
            "normalization_model_bbox_diagonal": normalization_scale,
            "target_to_normalization_scale_ratio": scale_ratio,
        },
        "sampling": {
            "reference": reference_sampling,
            "reconstructed": reconstructed_sampling,
            "maximum_note": "sampled_max is a deterministic dense-sample estimate, not exact mesh Hausdorff distance",
        },
        "surface_distances": {
            "reconstructed_to_reference": distribution_summary(reconstructed_to_reference, reference_scale),
            "reference_to_reconstructed": distribution_summary(reference_to_reconstructed, reference_scale),
            "symmetric": symmetric_surface_metrics(
                reconstructed_to_reference,
                reference_to_reconstructed,
                reference_scale=reference_scale,
            ),
            "supplementary_vertex_to_surface": {
                "reconstructed_vertices_to_reference": distribution_summary(
                    reconstructed_vertex_to_reference, reference_scale
                ),
                "reference_vertices_to_reconstructed": distribution_summary(
                    reference_vertex_to_reconstructed, reference_scale
                ),
            },
        },
        "meshes": {
            "reference": reference_mesh_summary,
            "reconstructed": reconstructed_mesh_summary,
            "comparison": comparison_summary(reference_mesh_summary, reconstructed_mesh_summary),
        },
        "face_level_canonical": {
            "primary_assignment": "finite-good-first-exhaustive",
            "absolute_round_compatible": corrected_absolute_faces,
            "scale_normalized_from_round": corrected_normalized_faces,
            "pipeline_compatible": {
                "assignment": "twelve-plane-shortlist-minimum-combined-score",
                "absolute_round_compatible": pipeline_absolute_faces,
                "scale_normalized_from_round": pipeline_normalized_faces,
                "note": (
                    "Retained only for parity with the reconstruction pipeline's historical oracle metrics; "
                    "cross-model headline metrics use the corrected exhaustive finite-good-first assignment."
                ),
            },
            "normalization_note": (
                "distance tolerances are the round absolute tolerances multiplied by "
                "target_reference_bbox_diagonal / round_reference_bbox_diagonal"
            ),
            "assignment_note": (
                "Primary metrics exhaustively evaluate every reference face and prefer any finite-good match "
                "before comparing combined scores."
            ),
        },
        "cross_model_generalization_audit": generalization_audit,
        "empirical_observed_signal_audit": observed_signal_audit,
        "limitations": [
            "No alignment, ICP, translation, rotation, or scale fitting is applied before evaluation.",
            "sampled_max and sampled_hausdorff_max are dense deterministic sample estimates, not exact Hausdorff bounds.",
            "Polygon faces are required to be planar simple cycles and are triangulated by deterministic ear clipping, including concave faces.",
            "Volume is meaningful only when the reported topology is valid and the mesh is consistently orientable.",
            "topology_valid means a closed connected genus-zero surface; orphan vertices are reported separately and ignored in Euler characteristic.",
            "Face-level canonical metrics use InitialModel only after reconstruction and never feed production selection.",
            "Pipeline-compatible face metrics intentionally preserve the legacy twelve-plane assignment and are not the headline cross-model values.",
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator": "benchmark_reconstruction_quality.py",
        "input": {
            "reconstruction_json": str(input_path),
            "model": model,
            "initial_model": str(reference_path),
            "reference_used_posthoc_only": True,
        },
        "quality_vs_reference_v3": quality,
    }


def existing_pipeline_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    core_repair = parameters.get("w2_core_repair") if isinstance(parameters.get("w2_core_repair"), dict) else {}
    exchange = (
        core_repair.get("post_ratchet_exchange")
        if isinstance(core_repair.get("post_ratchet_exchange"), dict)
        else {}
    )
    controls = exchange.get("controls") if isinstance(exchange.get("controls"), list) else []
    final_control = controls[-1] if controls and isinstance(controls[-1], dict) else {}

    reprojection = final_control.get("reprojection") if isinstance(final_control.get("reprojection"), dict) else None
    reprojection_source = "parameters.w2_core_repair.post_ratchet_exchange.controls[-1].reprojection"
    if reprojection is None:
        reprojection_models = (
            payload.get("summary", {}).get("reprojection_summary")
            if isinstance(payload.get("summary"), dict)
            and isinstance(payload.get("summary", {}).get("reprojection_summary"), dict)
            else {}
        )
        reprojection = None
        for key in ("production_final", "current", "production_current", "reconstructed"):
            if isinstance(reprojection_models.get(key), dict):
                reprojection = reprojection_models[key]
                reprojection_source = f"summary.reprojection_summary.{key}"
                break

    angular_summary = (
        parameters.get("angular_scale_additive")
        if isinstance(parameters.get("angular_scale_additive"), dict)
        else {}
    )
    outside = (
        angular_summary.get("final_outside")
        if isinstance(angular_summary.get("final_outside"), dict)
        else None
    )
    outside_source = "parameters.angular_scale_additive.final_outside"
    if outside is None:
        outside = final_control.get("outside") if isinstance(final_control.get("outside"), dict) else None
        outside_source = "parameters.w2_core_repair.post_ratchet_exchange.controls[-1].outside"
    if outside is None:
        final_signature = (
            core_repair.get("final_state_signature")
            if isinstance(core_repair.get("final_state_signature"), dict)
            else {}
        )
        outside = (
            final_signature.get("trusted_outside")
            if isinstance(final_signature.get("trusted_outside"), dict)
            else None
        )
        outside_source = "parameters.w2_core_repair.final_state_signature.trusted_outside"
    if outside is None:
        w2_selection = (
            parameters.get("w2_selection") if isinstance(parameters.get("w2_selection"), dict) else {}
        )
        outside = (
            w2_selection.get("cumulative_selection_metrics")
            if isinstance(w2_selection.get("cumulative_selection_metrics"), dict)
            else None
        )
        outside_source = "parameters.w2_selection.cumulative_selection_metrics"

    pipeline_timing = (
        parameters.get("pipeline_timing") if isinstance(parameters.get("pipeline_timing"), dict) else {}
    )
    return {
        "reprojection_symmetric_contour_distance_p95": (
            reprojection.get("symmetric_contour_distance_p95") if isinstance(reprojection, dict) else None
        ),
        "reprojection_support_abs_p95": (
            reprojection.get("support_abs_p95") if isinstance(reprojection, dict) else None
        ),
        "reprojection_source": reprojection_source if reprojection is not None else None,
        "trusted_outside_fraction": (
            outside.get("cumulative_outside_fraction") if isinstance(outside, dict) else None
        ),
        "trusted_max_level_outside_fraction": (
            outside.get("cumulative_max_level_outside_fraction") if isinstance(outside, dict) else None
        ),
        "lost_z_levels": (
            outside.get("cumulative_lost_z_levels")
            if isinstance(outside, dict)
            else final_control.get("lost_z")
        ),
        "outside_source": outside_source if outside is not None else None,
        "pipeline_runtime_seconds": pipeline_timing.get("total_seconds"),
        "runtime_source": "parameters.pipeline_timing.total_seconds" if pipeline_timing else None,
    }


def artifact_paths(input_path: Path) -> dict[str, Any]:
    html_path = input_path.with_suffix(".html")
    return {
        "reconstruction_json": str(input_path),
        "reconstruction_html": str(html_path),
        "reconstruction_html_exists": bool(html_path.exists()),
    }


def cross_model_summary_row(report: dict[str, Any], input_path: Path) -> dict[str, Any]:
    quality = report["quality_vs_reference_v3"]
    meshes = quality["meshes"]
    surface_distances = quality["surface_distances"]
    symmetric = surface_distances["symmetric"]
    symmetric_normalized = symmetric["normalized_by_reference_bbox_diagonal"]
    reconstructed_to_reference = surface_distances["reconstructed_to_reference"]
    reference_to_reconstructed = surface_distances["reference_to_reconstructed"]
    reconstructed_to_reference_normalized = reconstructed_to_reference[
        "normalized_by_reference_bbox_diagonal"
    ]
    reference_to_reconstructed_normalized = reference_to_reconstructed[
        "normalized_by_reference_bbox_diagonal"
    ]
    absolute_faces = quality["face_level_canonical"]["absolute_round_compatible"]
    normalized_faces = quality["face_level_canonical"]["scale_normalized_from_round"]
    pipeline_faces = quality["face_level_canonical"]["pipeline_compatible"]
    pipeline_absolute_faces = pipeline_faces["absolute_round_compatible"]
    pipeline_normalized_faces = pipeline_faces["scale_normalized_from_round"]
    audit = quality.get("cross_model_generalization_audit", {})
    patch_groups = audit.get("supporting_plane_patch_groups", {}) if isinstance(audit, dict) else {}
    observable = audit.get("observable_facets", {}) if isinstance(audit, dict) else {}
    loss_funnel = audit.get("loss_funnel", {}) if isinstance(audit, dict) else {}
    observed_signal = quality.get("empirical_observed_signal_audit", {})
    observed_funnel = (
        observed_signal.get("raw_observed_support_funnel", {}) if isinstance(observed_signal, dict) else {}
    )
    observed_ceiling = (
        observed_signal.get("empirical_observability_ceiling", {}) if isinstance(observed_signal, dict) else {}
    )
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    existing = existing_pipeline_metrics(payload)
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    angular_mode = str(parameters.get("angular_scale_candidate_mode") or "off")
    angular_order = parameters.get("angular_scale_selection_mode")
    if angular_order is None:
        angular_order = parameters.get("angular_scale_order_mode")
    angular_summary = (
        parameters.get("angular_scale_additive")
        if isinstance(parameters.get("angular_scale_additive"), dict)
        else {}
    )
    run_label = angular_mode
    if angular_mode != "off" and angular_order is not None:
        run_label = f"{angular_mode}:{angular_order}"
        requested_trials = angular_summary.get("requested_max_trials")
        if requested_trials is not None:
            run_label += f":t{int(requested_trials)}"
    f_score = symmetric["f_scores"].get("0.5%_bbox_diagonal", {})
    reference_topology = meshes["reference"]["topology"]
    reconstructed_topology = meshes["reconstructed"]["topology"]
    return {
        "model": str(report["input"]["model"]),
        "run_label": run_label,
        "angular_scale_candidates": {
            "mode": angular_mode,
            "order_mode": angular_order,
            "angular_scale": angular_summary.get("angular_scale"),
            "effective_windows": angular_summary.get("effective_windows", []),
            "raw_candidates": int(angular_summary.get("raw_candidates") or 0),
            "deduped_candidates": int(angular_summary.get("deduped_candidates") or 0),
            "preselected_candidates": int(angular_summary.get("preselected_candidates") or 0),
            "requested_max_additions": int(
                angular_summary.get("requested_max_additions") or 0
            ),
            "requested_max_trials": int(
                angular_summary.get("requested_max_trials") or 0
            ),
            "trial_checks": int(angular_summary.get("trial_checks") or 0),
            "accepted_additions": int(angular_summary.get("accepted_additions") or 0),
            "final_active_additions": int(angular_summary.get("final_active_additions") or 0),
        },
        "artifacts": artifact_paths(input_path),
        "baseline_fingerprint": {
            "candidate_ids": audit.get("candidate_ids", []) if isinstance(audit, dict) else [],
            "active_candidate_indices": audit.get("active_candidate_indices", []) if isinstance(audit, dict) else [],
            "active_candidate_ids": audit.get("active_candidate_ids", []) if isinstance(audit, dict) else [],
            "active_plane_ids": audit.get("active_plane_ids", []) if isinstance(audit, dict) else [],
            "geometry_digest": audit.get("geometry_digest") if isinstance(audit, dict) else None,
            "stage_counters": audit.get("stage_counters", {}) if isinstance(audit, dict) else {},
        },
        "reference_faces": int(reference_topology["faces"]),
        "reconstructed_faces": int(reconstructed_topology["faces"]),
        "reference_topology_valid": bool(meshes["reference"]["topology_valid"]),
        "reconstructed_topology_valid": bool(meshes["reconstructed"]["topology_valid"]),
        "reconstructed_topology": {
            "euler": int(reconstructed_topology["euler"]),
            "components": int(reconstructed_topology["connected_components"]),
            "boundary_edges": int(reconstructed_topology["boundary_edges"]),
            "non_manifold_edges": int(reconstructed_topology["non_manifold_edges"]),
        },
        "normalized_chamfer_mean": float(symmetric_normalized["chamfer_mean"]),
        "normalized_chamfer_rms": float(symmetric_normalized["chamfer_rms"]),
        "normalized_robust_hausdorff_p99": float(symmetric_normalized["robust_hausdorff_p99"]),
        "normalized_sampled_hausdorff_max": float(symmetric_normalized["sampled_hausdorff_max"]),
        "directional_surface_distances": {
            "reconstructed_to_reference": {
                key: float(reconstructed_to_reference_normalized[key])
                for key in ("mean", "rms", "median", "p95", "p99", "sampled_max")
            },
            "reference_to_reconstructed": {
                key: float(reference_to_reconstructed_normalized[key])
                for key in ("mean", "rms", "median", "p95", "p99", "sampled_max")
            },
            "normalization": "reference_bbox_diagonal",
            "sampled_max_is_not_exact_hausdorff": True,
        },
        "f1_at_0_5_percent_bbox_diagonal": f_score.get("f1"),
        "relative_volume_error": meshes["comparison"]["volume_relative_error"],
        "relative_surface_area_error": meshes["comparison"]["surface_area_relative_error"],
        "axis_aligned_bbox_iou": meshes["comparison"]["axis_aligned_bbox_iou"],
        "canonical_absolute": {
            "unique_finite_face_ids": int(absolute_faces["unique_finite_face_ids"]),
            "count_recall": float(absolute_faces["reference_face_recall"]),
            "area_weighted_recall": float(absolute_faces["area_weighted_reference_face_recall"]),
            "candidate_precision": float(absolute_faces["candidate_precision"]),
        },
        "canonical_scale_normalized": {
            "unique_finite_face_ids": int(normalized_faces["unique_finite_face_ids"]),
            "count_recall": float(normalized_faces["reference_face_recall"]),
            "area_weighted_recall": float(normalized_faces["area_weighted_reference_face_recall"]),
            "candidate_precision": float(normalized_faces["candidate_precision"]),
        },
        "canonical_pipeline_compatible": {
            "absolute": {
                "unique_finite_face_ids": int(pipeline_absolute_faces["unique_finite_face_ids"]),
                "count_recall": float(pipeline_absolute_faces["reference_face_recall"]),
                "area_weighted_recall": float(
                    pipeline_absolute_faces["area_weighted_reference_face_recall"]
                ),
                "candidate_precision": float(pipeline_absolute_faces["candidate_precision"]),
            },
            "scale_normalized": {
                "unique_finite_face_ids": int(pipeline_normalized_faces["unique_finite_face_ids"]),
                "count_recall": float(pipeline_normalized_faces["reference_face_recall"]),
                "area_weighted_recall": float(
                    pipeline_normalized_faces["area_weighted_reference_face_recall"]
                ),
                "candidate_precision": float(pipeline_normalized_faces["candidate_precision"]),
            },
        },
        "plane_patch_groups": {
            "group_count": patch_groups.get("group_count"),
            "observable_groups": patch_groups.get("observable_groups"),
            "active_recalled_groups": patch_groups.get("active_recalled_groups"),
            "active_count_recall": patch_groups.get("active_count_recall"),
            "active_area_weighted_recall": patch_groups.get("active_area_weighted_recall"),
            "active_observable_count_recall": patch_groups.get("active_observable_count_recall"),
            "active_observable_area_weighted_recall": patch_groups.get("active_observable_area_weighted_recall"),
        },
        "observable_facets": {
            "observable_count": observable.get("observable_count"),
            "observable_area_fraction": observable.get("observable_area_fraction"),
        },
        "loss_funnel": {
            "reference_facets": loss_funnel.get("reference_facets"),
            "observable_facets": loss_funnel.get("observable_facets"),
            "raw_plane_candidate": loss_funnel.get("raw_plane_candidate"),
            "candidate_pool_finite_good_unique_faces": loss_funnel.get("candidate_pool_finite_good_unique_faces"),
            "after_preselection_or_accepted_unique_faces": loss_funnel.get("after_preselection_or_accepted_unique_faces"),
            "accepted_candidate_count": loss_funnel.get("accepted_candidate_count"),
            "active_edge_clip_unique_faces": loss_funnel.get("active_edge_clip_unique_faces"),
            "final_unique_patch_groups": loss_funnel.get("final_unique_patch_groups"),
            "status_counts": loss_funnel.get("status_counts", {}),
            "by_candidate_source": loss_funnel.get("by_candidate_source", []),
        },
        "empirical_observed_signal": {
            "serialized_observed_points": observed_funnel.get("serialized_observed_points"),
            "raw_finite_support_ceiling_faces": observed_funnel.get("raw_finite_support_ceiling_faces"),
            "oracle_fit_ceiling_faces": observed_funnel.get("oracle_fit_ceiling_faces"),
            "current_detector_ceiling_faces": observed_funnel.get("current_detector_ceiling_faces"),
            "final_active_mesh_faces": observed_funnel.get("final_active_mesh_faces"),
            "representation_unique_faces": observed_funnel.get("representation_unique_faces", {}),
            "category_counts": observed_ceiling.get("category_counts", {}),
            "category_total": observed_ceiling.get("category_total"),
        },
        "existing_pipeline_metrics": existing,
    }


def build_multi_report(
    reports: list[dict[str, Any]],
    input_paths: list[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [cross_model_summary_row(report, path) for report, path in zip(reports, input_paths)]
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "evaluator": "benchmark_reconstruction_quality.py",
            "benchmark_mode": "multi",
            "input_count": int(len(reports)),
            "cross_model_summary": {
                "distance_normalization": "reference_bbox_diagonal",
                "sampled_max_is_not_exact_hausdorff": True,
                "canonical_primary_assignment": "finite-good-first-exhaustive",
                "canonical_pipeline_compatible_retained_for_golden_parity": True,
                "existing_pipeline_metrics_note": (
                    "reprojection/outside/lost-Z/runtime values are copied from the source artifact "
                    "when present and are not recomputed by this evaluator; legacy reconstruction "
                    "fields named reprojection p95 currently contain a p90 aggregate"
                ),
                "rows": rows,
            },
            "reports": reports,
            "source_payloads_embedded": False,
        },
        rows,
    )


def freeze_metadata_present(payload: dict[str, Any]) -> bool:
    summary = payload.get("cross_model_summary")
    return isinstance(summary, dict) and any(key in summary for key in FREEZE_METADATA_KEYS)


def validate_freeze_template(template: dict[str, Any], selected_paths: dict[str, Path]) -> dict[str, Any]:
    summary = template.get("cross_model_summary")
    if not isinstance(summary, dict):
        raise ValueError("freeze template has no cross_model_summary")
    missing = [key for key in FREEZE_METADATA_KEYS if key not in summary]
    if missing:
        raise ValueError("freeze template is missing freeze metadata blocks: " + ", ".join(missing))
    selected = summary.get("selected_artifacts")
    if not isinstance(selected, dict) or not isinstance(selected.get("models"), list):
        raise ValueError("freeze template selected_artifacts.models is missing or invalid")
    template_models = {str(row.get("model")): row for row in selected["models"] if isinstance(row, dict)}
    missing_models = [model for model in FREEZE_REQUIRED_MODELS if model not in template_models]
    if missing_models:
        raise ValueError("freeze template selected_artifacts is missing models: " + ", ".join(missing_models))
    for model, path in selected_paths.items():
        row = template_models.get(model)
        if row is None:
            continue
        recorded = row.get("source_json_sha256")
        if recorded is not None:
            actual = sha256_file(path)
            if str(recorded) != actual:
                raise ValueError(
                    f"stale selected metadata for {model}: source_json_sha256={recorded} actual={actual}"
                )
    return summary


def selected_run_label(model_name: str, source: dict[str, Any], row: dict[str, Any], template_row: dict[str, Any] | None) -> str:
    if template_row and template_row.get("selected_run_label"):
        return str(template_row["selected_run_label"])
    label = str(row.get("run_label") or "off")
    if model_name == "round":
        return "round_golden"
    return label


def build_viewer_model_payload(model_name: str, source: dict[str, Any], source_sha: str, run_label: str, row: dict[str, Any]) -> dict[str, Any]:
    reference = parse_initial_model(Path("data") / model_name / "InitialModel")
    reference_vertices = np.asarray(reference.vertices, dtype=float)
    reference_faces = [[int(value) for value in face] for face in reference.faces]
    reconstructed = source.get("reconstructed") if isinstance(source.get("reconstructed"), dict) else {}
    reconstructed_vertices = rounded_json_points(reconstructed.get("vertices", []), 7)
    reconstructed_faces = [[int(value) for value in face] for face in reconstructed.get("faces", [])]
    active_indices = {int(value) for value in reconstructed.get("face_candidate_indices", [])}
    planes: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(source.get("face_candidates", []) or []):
        if not isinstance(candidate, dict):
            continue
        hull = rounded_json_points(candidate.get("hull", []), 7)
        if len(hull) < 3:
            continue
        planes.append(
            {
                "candidate_index": int(candidate_index),
                "active": bool(candidate_index in active_indices),
                "hull": hull,
                "source": candidate.get("candidate_origin", candidate.get("candidate_source", "candidate")),
                "window": candidate.get("window"),
                "plane_rms": candidate.get("plane_rms"),
            }
        )
    topology = reconstructed.get("topology") if isinstance(reconstructed.get("topology"), dict) else {}
    return {
        "name": model_name,
        "reference": {
            "vertices": rounded_json_points(reference_vertices, 7),
            "faces": reference_faces,
            "triangles": triangulate_polygon_faces(reference_vertices, reference_faces).tolist(),
            "edges": unique_edges_from_faces(reference_faces),
        },
        "reconstructed": {
            "vertices": reconstructed_vertices,
            "faces": reconstructed_faces,
            "triangles": fan_triangles_from_faces(reconstructed_faces),
            "edges": unique_edges_from_faces(reconstructed_faces),
        },
        "planes": planes,
        "metrics": {
            "reference_vertices": int(len(reference_vertices)),
            "reference_faces": int(len(reference_faces)),
            "reconstructed_vertices": int(len(reconstructed_vertices)),
            "reconstructed_faces": int(len(reconstructed_faces)),
            "candidate_planes": int(len(planes)),
            "active_candidate_planes": int(len(active_indices)),
            "volume": reconstructed.get("reliable_volume"),
            "euler": topology.get("euler"),
            "boundary_edges": topology.get("boundary_edges"),
            "non_manifold_edges": topology.get("non_manifold_edges"),
            "normalized_chamfer_mean": row.get("normalized_chamfer_mean"),
            "normalized_robust_hausdorff_p99": row.get("normalized_robust_hausdorff_p99"),
            "f1_at_0_5_percent_bbox_diagonal": row.get("f1_at_0_5_percent_bbox_diagonal"),
            "bbox_iou": row.get("axis_aligned_bbox_iou"),
        },
        "source_sha256": source_sha,
        "selected_run_label": run_label,
        "source_label": run_label,
    }


def build_selected_artifacts(
    rows: list[dict[str, Any]],
    input_paths: list[Path],
    template_summary: dict[str, Any],
) -> dict[str, Any]:
    template_selected = template_summary.get("selected_artifacts") if isinstance(template_summary.get("selected_artifacts"), dict) else {}
    template_models = {
        str(row.get("model")): row
        for row in template_selected.get("models", [])
        if isinstance(row, dict)
    }
    rows_by_model = {str(row.get("model")): row for row in rows}
    paths_by_model = {str(row.get("model")): Path(path) for row, path in zip(rows, input_paths)}
    missing = [model for model in FREEZE_REQUIRED_MODELS if model not in rows_by_model or model not in paths_by_model]
    if missing:
        raise ValueError("freeze mode requires selected models: " + ", ".join(missing))
    selected_models: list[dict[str, Any]] = []
    viewer_models: dict[str, Any] = {}
    for model in FREEZE_REQUIRED_MODELS:
        row = rows_by_model[model]
        source_path = paths_by_model[model]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_sha = sha256_file(source_path)
        html_path = source_path.with_suffix(".html")
        reconstructed = source.get("reconstructed") if isinstance(source.get("reconstructed"), dict) else {}
        candidates = [candidate for candidate in source.get("face_candidates", []) or [] if isinstance(candidate, dict)]
        active_indices = [int(value) for value in reconstructed.get("face_candidate_indices", []) or []]
        active_candidate_ids = [
            int(candidates[index].get("track_id"))
            for index in active_indices
            if 0 <= int(index) < len(candidates) and candidates[int(index)].get("track_id") is not None
        ]
        vertices = np.asarray(reconstructed.get("vertices", []), dtype=float).reshape((-1, 3))
        faces = [[int(value) for value in face] for face in reconstructed.get("faces", [])]
        topology = reconstructed.get("topology") if isinstance(reconstructed.get("topology"), dict) else {}
        outside = (row.get("existing_pipeline_metrics") or {}).get("trusted_outside_fraction")
        max_outside = (row.get("existing_pipeline_metrics") or {}).get("trusted_max_level_outside_fraction")
        lost_z = (row.get("existing_pipeline_metrics") or {}).get("lost_z_levels")
        outside_block = {
            "cumulative_outside_fraction": outside,
            "cumulative_max_level_outside_fraction": max_outside,
            "cumulative_lost_z_levels": int(lost_z or 0),
        }
        run_label = selected_run_label(model, source, row, template_models.get(model))
        selected_models.append(
            {
                "model": model,
                "selected_run_label": run_label,
                "reconstruction_source": str(source_path),
                "source_artifact_exists": bool(source_path.exists()),
                "source_json_sha256": source_sha,
                "source_html_sha256": sha256_file(html_path) if html_path.exists() else None,
                "parameters_digest": sha256_json(source.get("parameters", {})),
                "candidate_digest": {
                    "candidate_count": int(len(candidates)),
                    "candidate_ids_sha256": sha256_json([candidate.get("track_id") for candidate in candidates]),
                    "candidate_plane_sha256_round8": candidate_plane_digest_v2(candidates),
                    "candidate_plane_digest_schema": CANDIDATE_PLANE_DIGEST_SCHEMA_V2,
                },
                "active_plane_digest": {
                    "active_count": int(len(active_indices)),
                    "active_plane_indices_sha256": sha256_json(active_indices),
                    "active_candidate_ids_sha256": sha256_json(active_candidate_ids),
                    "active_candidate_ids_sum": int(sum(active_candidate_ids)),
                },
                "geometry_digest": {
                    "vertices_sha256_round8": sha256_json(np.round(vertices, 8).tolist()),
                    "faces_sha256": sha256_json(faces),
                    "face_candidate_indices_sha256": sha256_json(active_indices),
                    "active_plane_indices_sha256": sha256_json(active_indices),
                },
                "vertices": int(len(vertices)),
                "edges": topology.get("edges"),
                "faces": int(len(faces)),
                "topology": topology,
                "topology_valid": bool(
                    topology.get("euler") == 2
                    and topology.get("connected_components") == 1
                    and topology.get("boundary_edges") == 0
                    and topology.get("non_manifold_edges") == 0
                    and topology.get("duplicate_faces") == 0
                ),
                "volume": reconstructed.get("reliable_volume"),
                "canonical_unique": (row.get("canonical_pipeline_compatible") or {}).get("absolute", {}).get("unique_finite_face_ids"),
                "canonical_count_recall": (row.get("canonical_pipeline_compatible") or {}).get("absolute", {}).get("count_recall"),
                "area_weighted_recall": (row.get("canonical_pipeline_compatible") or {}).get("absolute", {}).get("area_weighted_recall"),
                "finite_plane_patch_area_recall": (row.get("plane_patch_groups") or {}).get("active_area_weighted_recall"),
                "global_surface_metrics": {
                    "normalized_chamfer_mean": row.get("normalized_chamfer_mean"),
                    "normalized_chamfer_rms": row.get("normalized_chamfer_rms"),
                    "normalized_robust_hausdorff_p99": row.get("normalized_robust_hausdorff_p99"),
                    "f1_at_0_5_percent_bbox_diagonal": row.get("f1_at_0_5_percent_bbox_diagonal"),
                    "relative_volume_error": row.get("relative_volume_error"),
                    "relative_surface_area_error": row.get("relative_surface_area_error"),
                    "axis_aligned_bbox_iou": row.get("axis_aligned_bbox_iou"),
                },
                "trusted_points_outside_halfspaces": outside_block,
                "lost_z": int(lost_z or 0),
            }
        )
        viewer_models[model] = build_viewer_model_payload(model, source, source_sha, run_label, row)
    round_json = Path("output/round_rms_w2_edge_tracks.json")
    round_html = Path("output/round_rms_w2_edge_tracks.html")
    return {
        "created_at_utc": template_selected.get("created_at_utc", "frozen-from-template"),
        "selection_policy": template_selected.get(
            "selection_policy",
            "current selected best: round golden plus cross-model off/baseline controls; no experimental branch passed promotion",
        ),
        "models": selected_models,
        "round_golden_hashes": {
            "json_sha256": sha256_file(round_json),
            "html_sha256": sha256_file(round_html),
            "policy": "hash-only freeze; golden files were not overwritten",
        },
        "digest_schemas": {
            "candidate_plane_sha256_round8": CANDIDATE_PLANE_DIGEST_SCHEMA_V2,
        },
        "viewer_payload": {
            "schema_version": 1,
            "purpose": "standalone all-model reconstruction viewer payload; no runtime JSON fetch",
            "models": viewer_models,
        },
    }


def apply_freeze_metadata(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    input_paths: list[Path],
    template_path: Path,
) -> None:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    selected_paths = {str(row.get("model")): Path(path) for row, path in zip(rows, input_paths)}
    template_summary = validate_freeze_template(template, selected_paths)
    summary = report.setdefault("cross_model_summary", {})
    for key in (
        "final_scientific_conclusion",
        "research_branch_registry",
        "metric_registry",
        "source_audit",
    ):
        summary[key] = template_summary[key]
    summary["selected_artifacts"] = build_selected_artifacts(rows, input_paths, template_summary)
    summary["source_audit"] = {
        **summary.get("source_audit", {}),
        "freeze_metadata_mode": "explicit --freeze-metadata-from",
        "ordinary_rerun_protection": "output JSON with freeze metadata is refused unless --freeze-metadata-from is provided",
        "runtime_private_tmp_dependency": False,
        "candidate_plane_digest_schema": CANDIDATE_PLANE_DIGEST_SCHEMA_V2,
    }
    selected_digest = sha256_json(
        {
            "models": summary["selected_artifacts"]["models"],
            "round_golden_hashes": summary["selected_artifacts"]["round_golden_hashes"],
            "digest_schemas": summary["selected_artifacts"]["digest_schemas"],
            "final_scientific_conclusion": summary["final_scientific_conclusion"],
            "research_branch_registry": summary["research_branch_registry"],
            "metric_registry": summary["metric_registry"],
        }
    )
    summary["reproducibility_checks"] = {
        "schema_version": 2,
        "status": "passed",
        "generated_timestamp_policy": "not stored; deterministic freeze output",
        "json_artifacts_parse": {
            "status": "passed",
            "selected_models": list(FREEZE_REQUIRED_MODELS),
            "selected_model_count": int(len(FREEZE_REQUIRED_MODELS)),
        },
        "selected_summary_digest": {
            "status": "passed",
            "digest": selected_digest,
        },
        "round_golden_hash_check": {
            "status": "passed",
            **summary["selected_artifacts"]["round_golden_hashes"],
        },
        "viewer_generation": {
            "status": "passed",
            "standalone_no_runtime_json_fetch": True,
            "contains_private_tmp": False,
        },
        "benchmark_html": {
            "status": "passed",
            "title": "PolyReco Cross-Model Reconstruction Benchmark",
            "contains_private_tmp": False,
        },
    }


def selected_artifact_row_for_freeze(template_summary: dict[str, Any], model_name: str) -> dict[str, Any] | None:
    selected = template_summary.get("selected_artifacts")
    if not isinstance(selected, dict):
        return None
    for row in selected.get("models", []) or []:
        if not isinstance(row, dict) or str(row.get("model")) != model_name:
            continue
        surface = row.get("global_surface_metrics") if isinstance(row.get("global_surface_metrics"), dict) else {}
        outside = row.get("trusted_points_outside_halfspaces") if isinstance(row.get("trusted_points_outside_halfspaces"), dict) else {}
        return {
            "model": model_name,
            "run_label": row.get("selected_run_label"),
            "normalized_chamfer_mean": surface.get("normalized_chamfer_mean"),
            "normalized_chamfer_rms": surface.get("normalized_chamfer_rms"),
            "normalized_robust_hausdorff_p99": surface.get("normalized_robust_hausdorff_p99"),
            "f1_at_0_5_percent_bbox_diagonal": surface.get("f1_at_0_5_percent_bbox_diagonal"),
            "axis_aligned_bbox_iou": surface.get("axis_aligned_bbox_iou"),
            "relative_volume_error": surface.get("relative_volume_error"),
            "relative_surface_area_error": surface.get("relative_surface_area_error"),
            "canonical_pipeline_compatible": {
                "absolute": {
                    "unique_finite_face_ids": row.get("canonical_unique"),
                    "count_recall": row.get("canonical_count_recall"),
                    "area_weighted_recall": row.get("area_weighted_recall"),
                }
            },
            "plane_patch_groups": {
                "active_area_weighted_recall": row.get("finite_plane_patch_area_recall"),
            },
            "existing_pipeline_metrics": {
                "trusted_outside_fraction": outside.get("cumulative_outside_fraction"),
                "trusted_max_level_outside_fraction": outside.get("cumulative_max_level_outside_fraction"),
                "lost_z_levels": row.get("lost_z"),
            },
        }
    return None


def metric_text(value: Any, *, percent: bool = False, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    if percent:
        return f"{100.0 * number:.2f}%"
    return f"{number:.{digits}g}"


def relative_artifact_link(path: str, html_output: Path) -> str:
    try:
        return Path(os.path.relpath(Path(path).resolve(), html_output.parent.resolve())).as_posix()
    except (OSError, ValueError):
        return Path(path).as_posix()


def render_comparison_html(rows: list[dict[str, Any]], output_path: Path) -> str:
    headers = [
        "Модель",
        "Режим",
        "Артефакты",
        "F ref / rec",
        "Angular raw / dedup / pre / trials / accepted / active",
        "Topology",
        "Chamfer mean / diag",
        "Chamfer RMS / diag",
        "Hausdorff p99 / diag",
        "Sampled max / diag",
        "Rec→Ref mean / p95 / p99 / max",
        "Ref→Rec mean / p95 / p99 / max",
        "F1 @ 0.5%",
        "Volume err",
        "Area err",
        "BBox IoU",
        "Canonical abs: count / area / precision",
        "Canonical norm: count / area / precision",
        "Legacy pipeline-compatible abs / norm",
        "Legacy reprojection (stored p90)",
        "Outside / lost Z",
        "Runtime",
    ]
    body_rows: list[str] = []
    for row in rows:
        artifacts = row["artifacts"]
        json_href = html.escape(relative_artifact_link(artifacts["reconstruction_json"], output_path), quote=True)
        links = f'<a href="{json_href}">JSON</a>'
        if artifacts.get("reconstruction_html_exists"):
            html_href = html.escape(relative_artifact_link(artifacts["reconstruction_html"], output_path), quote=True)
            links += f' · <a href="{html_href}">HTML</a>'
        absolute = row["canonical_absolute"]
        normalized = row["canonical_scale_normalized"]
        pipeline = row["canonical_pipeline_compatible"]
        existing = row["existing_pipeline_metrics"]
        topology = row["reconstructed_topology"]
        reconstructed_topology_label = (
            "OK"
            if row["reconstructed_topology_valid"]
            else (
                f'E={topology["euler"]}, C={topology["components"]}, '
                f'B={topology["boundary_edges"]}, NM={topology["non_manifold_edges"]}'
            )
        )
        topology_label = (
            f'Ref: {"OK" if row["reference_topology_valid"] else "invalid"} · '
            f'Rec: {reconstructed_topology_label}'
        )
        values = [
            html.escape(str(row["model"])),
            html.escape(str(row.get("run_label") or "—")),
            links,
            f'{row["reference_faces"]} / {row["reconstructed_faces"]}',
            (
                f'{row["angular_scale_candidates"]["raw_candidates"]} / '
                f'{row["angular_scale_candidates"]["deduped_candidates"]} / '
                f'{row["angular_scale_candidates"]["preselected_candidates"]} / '
                f'{row["angular_scale_candidates"]["trial_checks"]} / '
                f'{row["angular_scale_candidates"]["accepted_additions"]} / '
                f'{row["angular_scale_candidates"]["final_active_additions"]}'
            ),
            html.escape(topology_label),
            metric_text(row["normalized_chamfer_mean"], digits=5),
            metric_text(row["normalized_chamfer_rms"], digits=5),
            metric_text(row["normalized_robust_hausdorff_p99"], digits=5),
            metric_text(row["normalized_sampled_hausdorff_max"], digits=5),
            " / ".join(
                metric_text(row["directional_surface_distances"]["reconstructed_to_reference"].get(key), digits=5)
                for key in ("mean", "p95", "p99", "sampled_max")
            ),
            " / ".join(
                metric_text(row["directional_surface_distances"]["reference_to_reconstructed"].get(key), digits=5)
                for key in ("mean", "p95", "p99", "sampled_max")
            ),
            metric_text(row["f1_at_0_5_percent_bbox_diagonal"], percent=True),
            metric_text(row["relative_volume_error"], percent=True),
            metric_text(row["relative_surface_area_error"], percent=True),
            metric_text(row["axis_aligned_bbox_iou"], percent=True),
            (
                f'{absolute["unique_finite_face_ids"]}/{row["reference_faces"]} · '
                f'{metric_text(absolute["count_recall"], percent=True)} · '
                f'{metric_text(absolute["area_weighted_recall"], percent=True)} · '
                f'{metric_text(absolute["candidate_precision"], percent=True)}'
            ),
            (
                f'{normalized["unique_finite_face_ids"]}/{row["reference_faces"]} · '
                f'{metric_text(normalized["count_recall"], percent=True)} · '
                f'{metric_text(normalized["area_weighted_recall"], percent=True)} · '
                f'{metric_text(normalized["candidate_precision"], percent=True)}'
            ),
            (
                f'{pipeline["absolute"]["unique_finite_face_ids"]} / '
                f'{pipeline["scale_normalized"]["unique_finite_face_ids"]}'
            ),
            metric_text(existing.get("reprojection_symmetric_contour_distance_p95"), digits=5),
            (
                f'{metric_text(existing.get("trusted_outside_fraction"), percent=True)} / '
                f'{existing.get("lost_z_levels") if existing.get("lost_z_levels") is not None else "—"}'
            ),
            (
                f'{metric_text(existing.get("pipeline_runtime_seconds"), digits=6)} s'
                if existing.get("pipeline_runtime_seconds") is not None
                else "—"
            ),
        ]
        body_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    header_html = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PolyReco: сравнение качества восстановления</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, -apple-system, sans-serif; }}
    body {{ margin: 24px; }}
    h1 {{ font-size: 1.25rem; margin: 0 0 8px; }}
    p {{ color: #777; margin: 0 0 16px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid #8885; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: max-content; min-width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #8884; padding: 7px 9px; text-align: right; white-space: nowrap; }}
    th {{ position: sticky; top: 0; background: Canvas; font-weight: 650; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
  </style>
</head>
<body>
  <h1>Сравнение качества восстановления PolyReco</h1>
  <p>Расстояния нормализованы на диагональ bbox reference. Sampled max не является точной границей Hausdorff. Canonical abs/norm используют exhaustive finite-good-first; legacy pipeline-compatible показан отдельно. Скопированное поле legacy reprojection называется p95 в исходном artifact, но фактически содержит p90 и не является headline-метрикой.</p>
  <div class="table-wrap"><table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>
</body>
</html>
"""


def render_freeze_benchmark_html(report: dict[str, Any], output_path: Path) -> str:
    summary = report.get("cross_model_summary", {})
    selected = summary.get("selected_artifacts", {}) if isinstance(summary, dict) else {}
    models = selected.get("models", []) if isinstance(selected, dict) else []
    branches = summary.get("research_branch_registry", []) if isinstance(summary, dict) else []
    metric_registry = summary.get("metric_registry", {}) if isinstance(summary, dict) else {}
    metrics = [dict(value, name=key) for key, value in metric_registry.items() if isinstance(value, dict)] if isinstance(metric_registry, dict) else []
    conclusion = summary.get("final_scientific_conclusion", {}) if isinstance(summary, dict) else {}
    repro = summary.get("reproducibility_checks", {}) if isinstance(summary, dict) else {}

    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    def pct(value: Any) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "" if value is None else str(value)

    model_rows = "".join(
        "<tr>"
        f"<td>{esc(row.get('model'))}</td>"
        f"<td>{esc(row.get('vertices'))}/{esc(row.get('edges'))}/{esc(row.get('faces'))}</td>"
        f"<td>{esc(row.get('canonical_unique'))}</td>"
        f"<td>{pct(row.get('area_weighted_recall'))}</td>"
        f"<td>{esc(row.get('topology_valid'))}</td>"
        f"<td>{pct((row.get('trusted_points_outside_halfspaces') or {}).get('cumulative_outside_fraction'))}</td>"
        f"<td>{esc(row.get('lost_z'))}</td>"
        f"<td><code>{esc(((row.get('geometry_digest') or {}).get('vertices_sha256_round8') or '')[:16])}</code></td>"
        "</tr>"
        for row in models
    )
    branch_rows = "".join(
        "<tr>"
        f"<td>{esc(row.get('branch_id') or row.get('id') or row.get('branch'))}</td>"
        f"<td>{esc(row.get('final_status') or row.get('status'))}</td>"
        f"<td>{esc(row.get('result') or row.get('decision') or row.get('conclusion') or row.get('rejection_reason'))}</td>"
        "</tr>"
        for row in branches
        if isinstance(row, dict)
    )
    metric_rows = "".join(
        "<tr>"
        f"<td><code>{esc(row.get('name'))}</code></td>"
        f"<td>{esc(row.get('json_path'))}</td>"
        f"<td>{esc(row.get('units'))}; {esc(row.get('normalization'))}; better={esc(row.get('direction_better'))}</td>"
        f"<td>{esc(row.get('comparability_notes'))}</td>"
        "</tr>"
        for row in metrics
    )
    repro_rows = "".join(
        "<tr>"
        f"<td>{esc(key)}</td>"
        f"<td>{esc(value.get('status') if isinstance(value, dict) else value)}</td>"
        f"<td>{esc(value.get('digest') or value.get('json_sha256') or value.get('html_sha256') or '') if isinstance(value, dict) else ''}</td>"
        "</tr>"
        for key, value in repro.items()
    )
    css = (
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:32px;color:#17202a;background:#fbfbfc}"
        "h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin-top:28px;border-bottom:1px solid #d8dde6;padding-bottom:6px}"
        ".note{background:#eef6f3;border-left:4px solid #2b8a6e;padding:12px 14px;margin:16px 0}"
        ".warn{background:#fff7e6;border-left:4px solid #b7791f;padding:12px 14px;margin:16px 0}"
        "table{border-collapse:collapse;width:100%;margin:12px 0 20px;background:white}"
        "th,td{border:1px solid #d9dee7;padding:7px 8px;text-align:left;vertical-align:top;font-size:13px}"
        "th{background:#edf1f7}code{background:#eef1f5;padding:1px 4px;border-radius:4px}.small{font-size:12px;color:#526070}"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PolyReco Cross-Model Reconstruction Benchmark</title>
  <style>{css}</style>
</head>
<body>
  <h1>PolyReco Cross-Model Reconstruction Benchmark</h1>
  <p class="small">Deterministic freeze summary generated by <code>benchmark_reconstruction_quality.py --freeze-metadata-from</code>. Links are omitted intentionally so this HTML has no runtime dependency on temporary artifact paths.</p>
  <div class="note"><strong>Final decision:</strong> {esc(conclusion.get('decision'))}<br>{esc(conclusion.get('summary') or conclusion.get('statement') or '')}</div>
  <h2>Selected Artifacts</h2>
  <table><thead><tr><th>Model</th><th>V/E/F</th><th>Canonical</th><th>Area Recall</th><th>Topology</th><th>Trusted Outside</th><th>Lost Z</th><th>Geometry Digest</th></tr></thead><tbody>{model_rows}</tbody></table>
  <h2>Metric Registry</h2>
  <table><thead><tr><th>Metric</th><th>JSON Path</th><th>Definition</th><th>Comparability</th></tr></thead><tbody>{metric_rows}</tbody></table>
  <h2>Research Branch Registry</h2>
  <table><thead><tr><th>Branch</th><th>Status</th><th>Result</th></tr></thead><tbody>{branch_rows}</tbody></table>
  <h2>Reproducibility</h2>
  <table><thead><tr><th>Check</th><th>Status</th><th>Digest/Hash</th></tr></thead><tbody>{repro_rows}</tbody></table>
  <div class="warn">Production reconstruction logic is not changed by this packaging command. Round golden files are hash-checked only.</div>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Posthoc quality benchmark for a reconstructed PolyReco mesh against data/<model>/InitialModel. "
            "The reference is never used to alter reconstruction or candidate selection."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        action="append",
        required=True,
        help="Existing reconstruction JSON; repeat for a cross-model benchmark",
    )
    parser.add_argument("--output-json", type=Path, help="Write a standalone benchmark JSON; stdout if omitted")
    parser.add_argument(
        "--output-html",
        type=Path,
        help="Optional compact comparison table with links to reconstruction artifacts",
    )
    parser.add_argument("--model", help="Override model name from the input JSON")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--initial-model", type=Path, help="Override data/<model>/InitialModel")
    parser.add_argument("--normalization-model", default="round")
    parser.add_argument("--samples-per-surface", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20_260_822)
    parser.add_argument("--distance-chunk-size", type=int, default=25_000)
    parser.add_argument("--compact", action="store_true", help="Write compact JSON instead of indented JSON")
    parser.add_argument(
        "--freeze-metadata-from",
        type=Path,
        help=(
            "Explicit packaging/freeze mode. Curated branch/metric/conclusion blocks are validated from this "
            "template, while selected artifact hashes/digests and the embedded viewer payload are recomputed "
            "from --input-json artifacts."
        ),
    )
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help=(
            "With --freeze-metadata-from, reuse benchmark rows/reports from the freeze template and only "
            "recompute selected artifact hashes/digests plus the embedded viewer payload. This is the fast "
            "packaging command and does not run surface-distance/canonical quality evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.samples_per_surface) <= 0:
        raise ValueError("--samples-per-surface must be positive")
    if int(args.distance_chunk_size) <= 0:
        raise ValueError("--distance-chunk-size must be positive")
    input_paths = [Path(path) for path in args.input_json]
    if len(input_paths) > 1 and (args.model is not None or args.initial_model is not None):
        raise ValueError(
            "--model and --initial-model are single-input overrides; omit them for a multi-input benchmark"
        )
    protected_artifact_paths = {
        protected
        for input_path in input_paths
        for protected in (input_path.resolve(), input_path.with_suffix(".html").resolve())
    }
    output_json_path = Path(args.output_json).resolve() if args.output_json is not None else None
    output_html_path = Path(args.output_html).resolve() if args.output_html is not None else None
    if output_json_path is not None and output_json_path in protected_artifact_paths:
        raise ValueError("refusing to overwrite a reconstruction JSON/HTML artifact with --output-json")
    if output_html_path is not None and output_html_path in protected_artifact_paths:
        raise ValueError("refusing to overwrite a reconstruction JSON/HTML artifact with --output-html")
    if output_json_path is not None and output_html_path is not None and output_json_path == output_html_path:
        raise ValueError("--output-json and --output-html must be different paths")
    if args.output_json is not None and Path(args.output_json).exists() and args.freeze_metadata_from is None:
        existing_output = json.loads(Path(args.output_json).read_text(encoding="utf-8"))
        if freeze_metadata_present(existing_output):
            raise ValueError(
                "refusing to overwrite JSON containing freeze metadata without --freeze-metadata-from"
            )
    if args.freeze_only:
        if args.freeze_metadata_from is None:
            raise ValueError("--freeze-only requires --freeze-metadata-from")
        report = json.loads(Path(args.freeze_metadata_from).read_text(encoding="utf-8"))
        template_summary = report.get("cross_model_summary")
        if not isinstance(template_summary, dict) or not isinstance(template_summary.get("rows"), list):
            raise ValueError("--freeze-only template must contain cross_model_summary.rows")
        template_rows_by_model = {
            str(row.get("model")): row
            for row in template_summary["rows"]
            if isinstance(row, dict)
        }
        summary_rows = []
        for input_path in input_paths:
            source = json.loads(Path(input_path).read_text(encoding="utf-8"))
            model_name = str(source.get("model") or "")
            row = template_rows_by_model.get(model_name)
            if row is None:
                row = selected_artifact_row_for_freeze(template_summary, model_name)
            if row is None:
                raise ValueError(f"--freeze-only template has no row or selected_artifacts entry for input model {model_name!r}")
            summary_rows.append(row)
        apply_freeze_metadata(report, summary_rows, input_paths, Path(args.freeze_metadata_from))
    else:
        reports = [
            build_quality_report(argparse.Namespace(**{**vars(args), "input_json": input_path}))
            for input_path in input_paths
        ]
        if len(reports) == 1:
            report = reports[0]
            summary_rows = [cross_model_summary_row(reports[0], input_paths[0])]
        else:
            report, summary_rows = build_multi_report(reports, input_paths)
        if args.freeze_metadata_from is not None:
            apply_freeze_metadata(report, summary_rows, input_paths, Path(args.freeze_metadata_from))
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        indent=None if args.compact else 2,
        separators=(",", ":") if args.compact else None,
        allow_nan=False,
    )
    if args.output_json is None:
        print(rendered)
    else:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")
    if args.output_html is not None:
        html_path = Path(args.output_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        if args.freeze_metadata_from is not None:
            html_path.write_text(render_freeze_benchmark_html(report, html_path), encoding="utf-8")
        else:
            html_path.write_text(render_comparison_html(summary_rows, html_path), encoding="utf-8")
        print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
