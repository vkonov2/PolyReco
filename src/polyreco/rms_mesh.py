from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from highspy import Highs, kHighsInf


EPS = 1e-9


def as_json_float(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def as_json_point(p: np.ndarray) -> list[float | None]:
    return [as_json_float(float(v)) for v in p]


@dataclass
class HalfspacePlane:
    normal: np.ndarray
    offset: float
    candidate_index: int
    original_plane_index: int
    retained_plane_index: int | None = None
    duplicate_of: int | None = None
    redundancy_status: str = "retained"
    redundancy_optimum: float | None = None
    redundancy_reason: str | None = None


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


def candidate_plane(candidate: dict[str, object]) -> tuple[np.ndarray, np.ndarray] | None:
    normal = np.array(candidate.get("plane_normal", []), dtype=float)
    point = np.array(candidate.get("plane_centroid", []), dtype=float)
    if normal.shape != (3,) or point.shape != (3,):
        return None
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(point)):
        return None
    norm = float(np.linalg.norm(normal))
    if norm <= EPS:
        return None
    return point, normal / norm


def highs_add_row(highs: Highs, lb: float, ub: float, cols: list[int], vals: list[float]) -> None:
    try:
        highs.addRow(float(lb), float(ub), cols, vals)
    except TypeError:
        highs.addRow(float(lb), float(ub), len(cols), cols, vals)


def highs_change_row_bounds(highs: Highs, row: int, lb: float, ub: float) -> None:
    highs.changeRowBounds(int(row), float(lb), float(ub))


def candidate_halfspace(candidate: dict[str, object]) -> tuple[np.ndarray, float] | None:
    plane = candidate_plane(candidate)
    if plane is None:
        return None
    point, normal = plane
    return normal, float(normal @ point)


def solve_halfspace_lp(
    constraints: list[tuple[np.ndarray, float]],
    *,
    inside_tol: float,
    objective: tuple[np.ndarray, float] | None,
    profile: dict[str, float | int] | None = None,
) -> tuple[bool, float | None, str]:
    started = time.perf_counter()
    highs = Highs()
    highs.setOptionValue("output_flag", False)
    for _ in range(3):
        highs.addVar(-kHighsInf, kHighsInf)
    if objective is not None:
        normal, _ = objective
        for i in range(3):
            highs.changeColCost(i, -float(normal[i]))
    for normal, offset in constraints:
        highs_add_row(
            highs,
            -kHighsInf,
            float(offset) + float(inside_tol),
            [0, 1, 2],
            [float(normal[0]), float(normal[1]), float(normal[2])],
        )
    if profile is not None:
        profile["model_construction_seconds"] = float(profile.get("model_construction_seconds", 0.0)) + (time.perf_counter() - started)
        profile["solve_halfspace_lp_calls"] = int(profile.get("solve_halfspace_lp_calls", 0)) + 1
    solve_started = time.perf_counter()
    highs.run()
    if profile is not None:
        profile["solve_seconds"] = float(profile.get("solve_seconds", 0.0)) + (time.perf_counter() - solve_started)
    status = highs.modelStatusToString(highs.getModelStatus())
    feasible = status in {"Optimal", "Objective bound", "Objective target"}
    optimum: float | None = None
    if feasible and objective is not None and status == "Optimal":
        _, offset = objective
        optimum = -float(highs.getObjectiveValue()) - float(offset)
    return feasible, optimum, status


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


def mesh_topology_summary(vertices: np.ndarray, faces: list[list[int]]) -> dict[str, object]:
    edge_counts: dict[tuple[int, int], int] = {}
    duplicate_faces = 0
    seen_faces: set[tuple[int, ...]] = set()
    repeated_vertex_faces = 0
    face_areas: list[float] = []
    edge_lengths: list[float] = []
    parent = list(range(int(vertices.shape[0])))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        if not parent:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for face in faces:
        if len(set(face)) != len(face):
            repeated_vertex_faces += 1
        key = tuple(sorted(face))
        if key in seen_faces:
            duplicate_faces += 1
        seen_faces.add(key)
        if len(face) >= 3 and vertices.size:
            face_areas.append(polygon_area(vertices[np.array(face, dtype=int)]))
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            if vertices.size:
                edge_lengths.append(float(np.linalg.norm(vertices[int(a)] - vertices[int(b)])))
            union(int(a), int(b))

    used_vertices = {int(i) for face in faces for i in face}
    components = len({find(i) for i in used_vertices}) if used_vertices and parent else 0
    incidence_counts = list(edge_counts.values())
    non_manifold = sum(1 for count in incidence_counts if count != 2)
    boundary = sum(1 for count in incidence_counts if count == 1)
    area_arr = np.array(face_areas, dtype=float)
    edge_arr = np.array(edge_lengths, dtype=float)
    return {
        "vertices": int(vertices.shape[0]),
        "edges": int(len(edge_counts)),
        "faces": int(len(faces)),
        "euler": int(vertices.shape[0] - len(edge_counts) + len(faces)),
        "connected_components": int(components),
        "boundary_edges": int(boundary),
        "non_manifold_edges": int(non_manifold),
        "duplicate_faces": int(duplicate_faces),
        "repeated_vertex_faces": int(repeated_vertex_faces),
        "edge_incidence_distribution": {str(k): int(incidence_counts.count(k)) for k in sorted(set(incidence_counts))},
        "face_area_min": as_json_float(float(np.min(area_arr))) if area_arr.size else None,
        "face_area_median": as_json_float(float(np.median(area_arr))) if area_arr.size else None,
        "edge_length_min": as_json_float(float(np.min(edge_arr))) if edge_arr.size else None,
    }


def mesh_signed_volume(vertices: np.ndarray, faces: list[list[int]], reference: np.ndarray | None = None) -> float:
    if reference is None:
        reference = np.zeros(3, dtype=float)
    volume = 0.0
    for face in faces:
        if len(face) < 3:
            continue
        pts = vertices[np.array(face, dtype=int)] - reference[None, :]
        p0 = pts[0]
        for i in range(1, pts.shape[0] - 1):
            volume += float(np.dot(p0, np.cross(pts[i], pts[i + 1]))) / 6.0
    return float(volume)


def lp_extents_for_halfspaces(
    constraints: list[tuple[np.ndarray, float]],
    *,
    inside_tol: float,
) -> dict[str, object]:
    extents: dict[str, list[float | None]] = {}
    feasible = True
    bounded = True
    for axis, name in enumerate(("x", "y", "z")):
        values: list[float | None] = []
        for sign in (-1.0, 1.0):
            normal = np.zeros(3, dtype=float)
            normal[axis] = sign
            ok, optimum, status = solve_halfspace_lp(
                constraints,
                inside_tol=float(inside_tol),
                objective=(normal, 0.0),
            )
            if not ok:
                feasible = False
                values.append(None)
                continue
            if optimum is None or not np.isfinite(float(optimum)):
                bounded = False
                values.append(None)
                continue
            values.append(float(-optimum if sign < 0.0 else optimum))
            if status != "Optimal":
                bounded = False
        extents[name] = values
    return {"feasible": bool(feasible), "bounded": bool(bounded), "extents": extents}


def normalized_candidate_halfspace(
    candidate: dict[str, object],
    *,
    halfspace_slack: float,
) -> tuple[np.ndarray, float] | None:
    halfspace = candidate_halfspace(candidate)
    if halfspace is None:
        return None
    normal, offset = halfspace
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    normal = normal.astype(float) / norm
    return normal, float(offset) / norm + float(halfspace_slack)


def prepare_halfspace_planes(
    candidates: list[dict[str, object]],
    *,
    halfspace_slack: float,
    duplicate_angle_tol: float,
    duplicate_offset_tol: float,
    prune_redundant: bool,
    redundancy_tol: float,
    feasibility_tol: float,
) -> tuple[list[HalfspacePlane], list[HalfspacePlane], dict[str, object]]:
    all_planes: list[HalfspacePlane] = []
    retained: list[HalfspacePlane] = []
    invalid_count = 0
    duplicate_count = 0
    for ci, candidate in enumerate(candidates):
        halfspace = normalized_candidate_halfspace(candidate, halfspace_slack=float(halfspace_slack))
        if halfspace is None:
            invalid_count += 1
            continue
        normal, offset = halfspace
        plane = HalfspacePlane(
            normal=normal,
            offset=offset,
            candidate_index=int(ci),
            original_plane_index=len(all_planes),
        )
        duplicate_of = None
        for existing in retained:
            dot = float(existing.normal @ normal)
            if dot < 0.0:
                continue
            angle = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
            if angle <= float(duplicate_angle_tol) and abs(existing.offset - offset) <= float(duplicate_offset_tol):
                duplicate_of = int(existing.original_plane_index)
                break
        if duplicate_of is not None:
            plane.duplicate_of = duplicate_of
            plane.redundancy_status = "duplicate"
            plane.redundancy_reason = "numerically_identical_plane"
            duplicate_count += 1
        else:
            retained.append(plane)
        all_planes.append(plane)

    redundancy_removed = 0
    if prune_redundant:
        kept: list[HalfspacePlane] = []
        constraints_all = [(plane.normal, plane.offset) for plane in retained]
        for idx, plane in enumerate(retained):
            constraints = [item for j, item in enumerate(constraints_all) if j != idx]
            feasible, optimum, status = solve_halfspace_lp(
                constraints,
                inside_tol=float(feasibility_tol),
                objective=(plane.normal, plane.offset),
            )
            plane.redundancy_optimum = as_json_float(float(optimum)) if optimum is not None else None
            if feasible and optimum is not None and float(optimum) <= float(redundancy_tol):
                plane.redundancy_status = "redundant"
                plane.redundancy_reason = "lp_max_violation_with_plane_removed_below_tol"
                redundancy_removed += 1
                continue
            if not feasible:
                plane.redundancy_status = "retained"
                plane.redundancy_reason = f"redundancy_lp_{status}"
            else:
                plane.redundancy_status = "retained"
                plane.redundancy_reason = "active_boundary"
            kept.append(plane)
        retained = kept

    for retained_index, plane in enumerate(retained):
        plane.retained_plane_index = int(retained_index)

    summary = {
        "input_candidates": int(len(candidates)),
        "invalid_planes": int(invalid_count),
        "duplicate_planes": int(duplicate_count),
        "prune_redundant": bool(prune_redundant),
        "redundancy_tol": float(redundancy_tol),
        "redundant_planes": int(redundancy_removed),
        "retained_planes": int(len(retained)),
        "plane_mapping": [
            {
                "candidate_index": int(plane.candidate_index),
                "original_plane_index": int(plane.original_plane_index),
                "retained_plane_index": int(plane.retained_plane_index) if plane.retained_plane_index is not None else None,
                "duplicate_of": int(plane.duplicate_of) if plane.duplicate_of is not None else None,
                "redundancy_status": plane.redundancy_status,
                "redundancy_optimum": plane.redundancy_optimum,
                "redundancy_reason": plane.redundancy_reason,
            }
            for plane in all_planes
        ],
    }
    return retained, all_planes, summary


def face_edges_topology(
    faces: list[list[int]],
    face_edge_ids: list[list[int]] | None,
    edge_count: int,
) -> dict[str, object]:
    if face_edge_ids is None:
        return {}
    edge_face_counts = [0 for _ in range(edge_count)]
    for row in face_edge_ids:
        for eid in row:
            if 0 <= int(eid) < edge_count:
                edge_face_counts[int(eid)] += 1
    distribution = {str(k): int(edge_face_counts.count(k)) for k in sorted(set(edge_face_counts))}
    return {
        "edge_face_incidence_distribution": distribution,
        "edge_face_boundary_edges": int(sum(1 for count in edge_face_counts if count == 1)),
        "edge_face_non_manifold_edges": int(sum(1 for count in edge_face_counts if count != 2)),
        "edge_face_unused_edges": int(sum(1 for count in edge_face_counts if count == 0)),
    }


def line_point_for_planes(a: HalfspacePlane, b: HalfspacePlane) -> np.ndarray | None:
    mat = np.vstack([a.normal, b.normal])
    gram = mat @ mat.T
    try:
        weights = np.linalg.solve(gram, np.array([a.offset, b.offset], dtype=float))
    except np.linalg.LinAlgError:
        return None
    x0 = mat.T @ weights
    if not np.all(np.isfinite(x0)):
        return None
    return x0


def merge_edge_clip_vertex(
    vertices: list[np.ndarray],
    vertex_planes: list[set[int]],
    point: np.ndarray,
    incident_planes: set[int],
    planes: list[HalfspacePlane],
    *,
    vertex_merge_tol: float,
    incidence_tol: float,
) -> tuple[int, bool]:
    best_i = None
    best_dist = float("inf")
    for i, existing in enumerate(vertices):
        dist = float(np.linalg.norm(point - existing))
        if dist < best_dist:
            best_dist = dist
            best_i = i
    if best_i is not None and best_dist <= float(vertex_merge_tol):
        union_planes = set(vertex_planes[best_i]) | set(incident_planes)
        residual = max(
            (
                abs(float(planes[pi].normal @ point - planes[pi].offset))
                for pi in union_planes
                if 0 <= int(pi) < len(planes)
            ),
            default=0.0,
        )
        if residual <= max(float(incidence_tol), float(vertex_merge_tol) * 10.0):
            vertex_planes[best_i] = union_planes
            return int(best_i), True
    vertices.append(point.astype(float))
    vertex_planes.append(set(incident_planes))
    return int(len(vertices) - 1), False


def cycle_from_plane_edges(edge_ids: list[int], edges: list[dict[str, object]]) -> tuple[list[int] | None, list[int] | None, str | None, dict[str, object]]:
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for eid in edge_ids:
        edge = edges[int(eid)]
        a, b = [int(v) for v in edge["vertices"]]
        adjacency.setdefault(a, []).append((b, int(eid)))
        adjacency.setdefault(b, []).append((a, int(eid)))
    degree_violations = {str(v): len(rows) for v, rows in adjacency.items() if len(rows) != 2}
    debug = {
        "vertices": int(len(adjacency)),
        "edges": int(len(edge_ids)),
        "degree_violations": degree_violations,
    }
    if degree_violations:
        return None, None, "face_graph_degree_violation", debug
    if not adjacency:
        return None, None, "no_face_edges", debug
    start = min(adjacency)
    cycle_vertices = [start]
    cycle_edges: list[int] = []
    prev = None
    current = start
    used_edges: set[int] = set()
    while True:
        choices = [(nb, eid) for nb, eid in adjacency[current] if eid not in used_edges]
        if not choices:
            break
        if prev is not None and len(choices) > 1:
            non_back = [(nb, eid) for nb, eid in choices if nb != prev]
            if non_back:
                choices = non_back
        nxt, eid = choices[0]
        used_edges.add(int(eid))
        cycle_edges.append(int(eid))
        prev, current = current, int(nxt)
        if current == start:
            break
        cycle_vertices.append(current)
        if len(cycle_edges) > len(edge_ids) + 1:
            return None, None, "face_cycle_walk_overflow", debug
    if current != start or len(used_edges) != len(edge_ids):
        debug["used_edges"] = int(len(used_edges))
        return None, None, "face_cycle_not_closed_or_disconnected", debug
    return cycle_vertices, cycle_edges, None, debug


def orient_face_cycle(vertices: np.ndarray, face: list[int], edge_ids: list[int], normal: np.ndarray) -> tuple[list[int], list[int], bool]:
    pts = vertices[np.array(face, dtype=int)]
    area_vec = np.zeros(3, dtype=float)
    for i in range(pts.shape[0]):
        area_vec += np.cross(pts[i], pts[(i + 1) % pts.shape[0]])
    if float(area_vec @ normal) < 0.0:
        return list(reversed(face)), list(reversed(edge_ids)), True
    return face, edge_ids, False


def reconstruct_polyhedron_from_halfspaces_edge_clip(
    candidates: list[dict[str, object]],
    *,
    halfspace_slack: float,
    feasibility_tol: float,
    incidence_tol: float,
    vertex_merge_tol: float,
    min_face_area: float,
    triple_det_tol: float,
    prune_redundant: bool,
    redundancy_tol: float,
) -> dict[str, object]:
    planes, all_planes, plane_summary = prepare_halfspace_planes(
        candidates,
        halfspace_slack=float(halfspace_slack),
        duplicate_angle_tol=1e-5,
        duplicate_offset_tol=max(float(vertex_merge_tol), 1e-10),
        prune_redundant=bool(prune_redundant),
        redundancy_tol=float(redundancy_tol),
        feasibility_tol=float(feasibility_tol),
    )
    constraints = [(plane.normal, plane.offset) for plane in planes]
    lp_extents = lp_extents_for_halfspaces(constraints, inside_tol=float(feasibility_tol))
    vertices_list: list[np.ndarray] = []
    vertex_planes: list[set[int]] = []
    edges: list[dict[str, object]] = []
    duplicate_edges = 0
    degenerate_pairs = 0
    skipped_parallel_pairs = 0
    unbounded_pairs = 0
    empty_pairs = 0
    endpoint_merges = 0
    edge_keys: dict[tuple[int, int, int, int], int] = {}
    n_planes = len(planes)
    for i in range(n_planes):
        for j in range(i + 1, n_planes):
            ni = planes[i].normal
            nj = planes[j].normal
            direction = np.cross(ni, nj)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= float(triple_det_tol):
                skipped_parallel_pairs += 1
                continue
            x0 = line_point_for_planes(planes[i], planes[j])
            if x0 is None:
                skipped_parallel_pairs += 1
                continue
            residual_pair = max(abs(float(ni @ x0 - planes[i].offset)), abs(float(nj @ x0 - planes[j].offset)))
            if residual_pair > max(float(incidence_tol), float(feasibility_tol) * 10.0):
                degenerate_pairs += 1
                continue
            direction = direction / direction_norm
            t_min = -float("inf")
            t_max = float("inf")
            t_min_plane = None
            t_max_plane = None
            infeasible = False
            for k, plane in enumerate(planes):
                if k == i or k == j:
                    continue
                coeff = float(plane.normal @ direction)
                rhs = float(plane.offset - plane.normal @ x0)
                if abs(coeff) <= float(triple_det_tol):
                    if rhs < -float(feasibility_tol):
                        infeasible = True
                        break
                    continue
                bound = rhs / coeff
                if coeff > 0.0:
                    if bound < t_max:
                        t_max = float(bound)
                        t_max_plane = int(k)
                else:
                    if bound > t_min:
                        t_min = float(bound)
                        t_min_plane = int(k)
                if t_min > t_max + float(feasibility_tol):
                    infeasible = True
                    break
            if infeasible:
                empty_pairs += 1
                continue
            if not np.isfinite(t_min) or not np.isfinite(t_max):
                unbounded_pairs += 1
                continue
            length = float(t_max - t_min)
            if length <= max(float(vertex_merge_tol), float(min_face_area)):
                degenerate_pairs += 1
                continue
            p_min = x0 + t_min * direction
            p_max = x0 + t_max * direction
            min_planes = {i, j}
            max_planes = {i, j}
            if t_min_plane is not None:
                min_planes.add(int(t_min_plane))
            if t_max_plane is not None:
                max_planes.add(int(t_max_plane))
            va, merged_a = merge_edge_clip_vertex(
                vertices_list,
                vertex_planes,
                p_min,
                min_planes,
                planes,
                vertex_merge_tol=float(vertex_merge_tol),
                incidence_tol=float(incidence_tol),
            )
            vb, merged_b = merge_edge_clip_vertex(
                vertices_list,
                vertex_planes,
                p_max,
                max_planes,
                planes,
                vertex_merge_tol=float(vertex_merge_tol),
                incidence_tol=float(incidence_tol),
            )
            endpoint_merges += int(merged_a) + int(merged_b)
            if va == vb:
                degenerate_pairs += 1
                continue
            a, b = sorted((int(va), int(vb)))
            key = (a, b, int(i), int(j))
            if key in edge_keys:
                duplicate_edges += 1
                continue
            edge_id = len(edges)
            edge_keys[key] = edge_id
            edges.append(
                {
                    "edge_id": int(edge_id),
                    "vertices": [int(va), int(vb)],
                    "plane_pair": [int(i), int(j)],
                    "candidate_pair": [int(planes[i].candidate_index), int(planes[j].candidate_index)],
                    "clip_planes": [int(t_min_plane) if t_min_plane is not None else None, int(t_max_plane) if t_max_plane is not None else None],
                    "clip_candidates": [
                        int(planes[t_min_plane].candidate_index) if t_min_plane is not None else None,
                        int(planes[t_max_plane].candidate_index) if t_max_plane is not None else None,
                    ],
                    "t_interval": [as_json_float(float(t_min)), as_json_float(float(t_max))],
                    "length": as_json_float(float(np.linalg.norm(p_max - p_min))),
                    "line_residual": as_json_float(float(residual_pair)),
                }
            )

    vertices = np.array(vertices_list, dtype=float).reshape((-1, 3))
    faces: list[list[int]] = []
    face_sources: list[int] = []
    face_edge_ids: list[list[int]] = []
    face_failures: dict[str, object] = {}
    winding_reversed = 0
    for plane_index, plane in enumerate(planes):
        local_edge_ids = [
            int(edge["edge_id"])
            for edge in edges
            if int(plane_index) in [int(v) for v in edge["plane_pair"]]
        ]
        if len(local_edge_ids) < 3:
            face_failures[str(plane_index)] = {
                "candidate_index": int(plane.candidate_index),
                "reason": "fewer_than_three_edges",
                "edge_count": int(len(local_edge_ids)),
            }
            continue
        face, cycle_edges, reason, debug = cycle_from_plane_edges(local_edge_ids, edges)
        if face is None or cycle_edges is None:
            face_failures[str(plane_index)] = {
                "candidate_index": int(plane.candidate_index),
                "reason": reason,
                "edge_count": int(len(local_edge_ids)),
                "graph": debug,
            }
            continue
        face, cycle_edges, reversed_winding = orient_face_cycle(vertices, face, cycle_edges, plane.normal)
        winding_reversed += int(reversed_winding)
        area = polygon_area(vertices[np.array(face, dtype=int)])
        if area < float(min_face_area):
            face_failures[str(plane_index)] = {
                "candidate_index": int(plane.candidate_index),
                "reason": "small_face_area",
                "area": as_json_float(float(area)),
            }
            continue
        faces.append(face)
        face_edge_ids.append(cycle_edges)
        face_sources.append(int(plane.candidate_index))

    topology = mesh_topology_summary(vertices, faces)
    topology.update(face_edges_topology(faces, face_edge_ids, len(edges)))
    ref = np.mean(vertices, axis=0) if vertices.size else np.zeros(3, dtype=float)
    volume_origin = mesh_signed_volume(vertices, faces)
    volume_ref = mesh_signed_volume(vertices, faces, reference=ref)
    bbox = None
    if vertices.size:
        bbox = {"min": as_json_point(np.min(vertices, axis=0)), "max": as_json_point(np.max(vertices, axis=0))}
    active_candidate_indices = set(face_sources)
    inactive_candidate_indices = [int(i) for i in range(len(candidates)) if i not in active_candidate_indices]
    vertex_incidence = [sorted(int(v) for v in rows) for rows in vertex_planes]
    max_vertex_residual = 0.0
    for vi, incident in enumerate(vertex_incidence):
        for pi in incident:
            max_vertex_residual = max(max_vertex_residual, abs(float(planes[pi].normal @ vertices[vi] - planes[pi].offset)))
    return {
        "vertices": [as_json_point(p) for p in vertices],
        "faces": faces,
        "face_candidate_indices": face_sources,
        "face_edge_ids": face_edge_ids,
        "edges": edges,
        "raw_intersections": int(len(edges)),
        "planes": int(n_planes),
        "inside_tol": float(feasibility_tol),
        "vertex_tol": float(vertex_merge_tol),
        "mesh_mode": "edge-clip",
        "halfspace_slack": float(halfspace_slack),
        "feasibility_tol": float(feasibility_tol),
        "incidence_tol": float(incidence_tol),
        "vertex_merge_tol": float(vertex_merge_tol),
        "min_face_area": float(min_face_area),
        "triple_det_tol": float(triple_det_tol),
        "prune_redundant": bool(prune_redundant),
        "redundancy_tol": float(redundancy_tol),
        "vertex_plane_incidence": vertex_incidence,
        "active_plane_indices": [int(i) for i, plane in enumerate(planes) if int(plane.candidate_index) in active_candidate_indices],
        "inactive_plane_indices": inactive_candidate_indices,
        "inactive_plane_reasons": {str(i): "inactive_no_face_cycle" for i in inactive_candidate_indices},
        "face_failures": face_failures,
        "topology": topology,
        "reliable_volume": as_json_float(abs(volume_ref)),
        "volume_signed_origin": as_json_float(volume_origin),
        "volume_signed_reference": as_json_float(volume_ref),
        "volume_consistency_delta": as_json_float(abs(abs(volume_origin) - abs(volume_ref))),
        "bbox": bbox,
        "lp_extents": lp_extents,
        "plane_preparation": plane_summary,
        "duplicate_edges": int(duplicate_edges),
        "endpoint_merges": int(endpoint_merges),
        "skipped_parallel_pairs": int(skipped_parallel_pairs),
        "unbounded_pairs": int(unbounded_pairs),
        "empty_pairs": int(empty_pairs),
        "degenerate_pairs": int(degenerate_pairs),
        "winding_reversed_faces": int(winding_reversed),
        "max_vertex_incidence_residual": as_json_float(float(max_vertex_residual)),
    }


def reconstruct_polyhedron_from_halfspaces_incidence(
    candidates: list[dict[str, object]],
    *,
    halfspace_slack: float,
    feasibility_tol: float,
    incidence_tol: float,
    vertex_merge_tol: float,
    min_face_area: float,
    prune_redundant: bool = False,
    redundancy_tol: float = 0.0,
    triple_det_tol: float = 1e-10,
) -> dict[str, object]:
    planes, all_planes, plane_summary = prepare_halfspace_planes(
        candidates,
        halfspace_slack=float(halfspace_slack),
        duplicate_angle_tol=1e-5,
        duplicate_offset_tol=max(float(vertex_merge_tol), 1e-8),
        prune_redundant=bool(prune_redundant),
        redundancy_tol=float(redundancy_tol),
        feasibility_tol=float(feasibility_tol),
    )

    raw_vertices: list[dict[str, object]] = []
    n_planes = len(planes)
    max_violation_kept = 0.0
    skipped_parallel = 0
    for ia in range(n_planes):
        na, oa = planes[ia].normal, planes[ia].offset
        for ib in range(ia + 1, n_planes):
            nb, ob = planes[ib].normal, planes[ib].offset
            for ic in range(ib + 1, n_planes):
                nc, oc = planes[ic].normal, planes[ic].offset
                mat = np.vstack([na, nb, nc])
                det = float(np.linalg.det(mat))
                if abs(det) <= float(triple_det_tol):
                    skipped_parallel += 1
                    continue
                rhs = np.array([oa, ob, oc], dtype=float)
                try:
                    x = np.linalg.solve(mat, rhs)
                except np.linalg.LinAlgError:
                    skipped_parallel += 1
                    continue
                if not np.all(np.isfinite(x)):
                    continue
                signed = np.array([float(plane.normal @ x - plane.offset) for plane in planes], dtype=float)
                max_violation = float(np.max(signed)) if signed.size else 0.0
                if max_violation <= float(feasibility_tol):
                    max_violation_kept = max(max_violation_kept, max(0.0, max_violation))
                    incident = {
                        int(i)
                        for i, value in enumerate(signed)
                        if abs(float(value)) <= float(incidence_tol)
                    }
                    incident.update((ia, ib, ic))
                    raw_vertices.append(
                        {
                            "point": x,
                            "incident": incident,
                            "source_triple": (ia, ib, ic),
                            "max_violation": max_violation,
                            "determinant": det,
                            "condition": float(np.linalg.cond(mat)),
                        }
                    )

    merged: list[dict[str, object]] = []
    for row in raw_vertices:
        point = np.array(row["point"], dtype=float)
        best_i = None
        best_dist = float("inf")
        for i, existing in enumerate(merged):
            dist = float(np.linalg.norm(point - np.array(existing["point"], dtype=float)))
            if dist < best_dist:
                best_dist = dist
                best_i = i
        if best_i is not None and best_dist <= float(vertex_merge_tol):
            existing = merged[best_i]
            existing["incident"].update(row["incident"])  # type: ignore[union-attr]
            if float(row["max_violation"]) < float(existing["max_violation"]):
                existing["point"] = point
                existing["max_violation"] = row["max_violation"]
                existing["condition"] = row["condition"]
                existing["determinant"] = row["determinant"]
            existing.setdefault("source_triples", []).append(row["source_triple"])
            existing.setdefault("conditions", []).append(row["condition"])
            existing.setdefault("determinants", []).append(row["determinant"])
        else:
            merged.append(
                {
                    "point": point,
                    "incident": set(row["incident"]),
                    "max_violation": row["max_violation"],
                    "condition": row["condition"],
                    "determinant": row["determinant"],
                    "source_triples": [row["source_triple"]],
                    "conditions": [row["condition"]],
                    "determinants": [row["determinant"]],
                }
            )

    vertices = np.array([row["point"] for row in merged], dtype=float).reshape((-1, 3))
    vertex_incidence = [sorted(int(i) for i in row["incident"]) for row in merged]
    vertex_source_triples = [
        [[int(v) for v in triple] for triple in row.get("source_triples", [])]
        for row in merged
    ]
    vertex_conditions = [
        [as_json_float(float(v)) for v in row.get("conditions", [])]
        for row in merged
    ]
    vertex_determinants = [
        [as_json_float(float(v)) for v in row.get("determinants", [])]
        for row in merged
    ]
    faces: list[list[int]] = []
    face_sources: list[int] = []
    inactive: dict[int, str] = {}
    face_areas: list[float] = []
    for plane_index, plane in enumerate(planes):
        normal = plane.normal
        offset = plane.offset
        candidate_index = int(plane.candidate_index)
        incident_indices = [
            vi
            for vi, incident in enumerate(vertex_incidence)
            if plane_index in incident or (vertices.size and abs(float(vertices[vi] @ normal - offset)) <= float(incidence_tol))
        ]
        incident_indices = sorted(set(incident_indices))
        if len(incident_indices) < 3:
            inactive[int(candidate_index)] = "fewer_than_three_incident_vertices"
            continue
        local_points = vertices[np.array(incident_indices, dtype=int)]
        ordered_local = order_face_vertices(local_points, normal)
        if len(ordered_local) < 3:
            inactive[int(candidate_index)] = "degenerate_polygon"
            continue
        face = [int(incident_indices[i]) for i in ordered_local]
        area = polygon_area(vertices[np.array(face, dtype=int)])
        if area < float(min_face_area):
            inactive[int(candidate_index)] = "small_face_area"
            continue
        faces.append(face)
        face_sources.append(int(candidate_index))
        face_areas.append(float(area))

    topology = mesh_topology_summary(vertices, faces)
    ref = np.mean(vertices, axis=0) if vertices.size else np.zeros(3, dtype=float)
    volume_origin = mesh_signed_volume(vertices, faces)
    volume_ref = mesh_signed_volume(vertices, faces, reference=ref)
    constraints = [(plane.normal, plane.offset) for plane in planes]
    lp_extents = lp_extents_for_halfspaces(constraints, inside_tol=float(feasibility_tol))
    bbox = None
    if vertices.size:
        bbox = {
            "min": as_json_point(np.min(vertices, axis=0)),
            "max": as_json_point(np.max(vertices, axis=0)),
        }
    active_candidate_indices = set(face_sources)
    inactive_candidate_indices = [int(i) for i in range(len(candidates)) if i not in active_candidate_indices]
    for i in inactive_candidate_indices:
        inactive.setdefault(int(i), "inactive_no_incident_face")
    return {
        "vertices": [as_json_point(p) for p in vertices],
        "faces": faces,
        "face_candidate_indices": face_sources,
        "raw_intersections": int(len(raw_vertices)),
        "planes": int(n_planes),
        "inside_tol": float(feasibility_tol),
        "vertex_tol": float(vertex_merge_tol),
        "mesh_mode": "incidence",
        "halfspace_slack": float(halfspace_slack),
        "feasibility_tol": float(feasibility_tol),
        "incidence_tol": float(incidence_tol),
        "vertex_merge_tol": float(vertex_merge_tol),
        "min_face_area": float(min_face_area),
        "vertex_plane_incidence": vertex_incidence,
        "vertex_source_triples": vertex_source_triples,
        "vertex_source_conditions": vertex_conditions,
        "vertex_source_determinants": vertex_determinants,
        "active_plane_indices": [int(i) for i, plane in enumerate(planes) if int(plane.candidate_index) in active_candidate_indices],
        "inactive_plane_indices": inactive_candidate_indices,
        "inactive_plane_reasons": {str(k): v for k, v in inactive.items()},
        "topology": topology,
        "reliable_volume": as_json_float(abs(volume_ref)),
        "volume_signed_origin": as_json_float(volume_origin),
        "volume_signed_reference": as_json_float(volume_ref),
        "volume_consistency_delta": as_json_float(abs(abs(volume_origin) - abs(volume_ref))),
        "bbox": bbox,
        "lp_extents": lp_extents,
        "plane_preparation": plane_summary,
        "duplicate_planes": int(plane_summary.get("duplicate_planes", 0)),
        "prune_redundant": bool(prune_redundant),
        "redundancy_tol": float(redundancy_tol),
        "triple_det_tol": float(triple_det_tol),
        "skipped_parallel_triples": int(skipped_parallel),
        "max_vertex_violation": as_json_float(float(max_violation_kept)),
        "face_area_min": as_json_float(float(np.min(face_areas))) if face_areas else None,
        "face_area_median": as_json_float(float(np.median(face_areas))) if face_areas else None,
    }


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

    topology = mesh_topology_summary(vertices, faces)
    ref = np.mean(vertices, axis=0) if vertices.size else np.zeros(3, dtype=float)
    volume_origin = mesh_signed_volume(vertices, faces)
    volume_ref = mesh_signed_volume(vertices, faces, reference=ref)
    return {
        "vertices": [as_json_point(p) for p in vertices],
        "faces": faces,
        "face_candidate_indices": face_sources,
        "raw_intersections": int(len(raw_vertices)),
        "planes": int(n_planes),
        "inside_tol": float(inside_tol),
        "vertex_tol": float(vertex_tol),
        "mesh_mode": "legacy",
        "topology": topology,
        "reliable_volume": as_json_float(abs(volume_ref)),
        "volume_signed_origin": as_json_float(volume_origin),
        "volume_signed_reference": as_json_float(volume_ref),
        "volume_consistency_delta": as_json_float(abs(abs(volume_origin) - abs(volume_ref))),
    }


def polygon_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    centroid = np.mean(points, axis=0)
    area = 0.0
    for i in range(points.shape[0]):
        area += 0.5 * float(np.linalg.norm(np.cross(points[i] - centroid, points[(i + 1) % points.shape[0]] - centroid)))
    return float(area)


__all__ = [
    "candidate_halfspace",
    "candidate_plane",
    "convex_hull_indices",
    "highs_add_row",
    "highs_change_row_bounds",
    "lp_extents_for_halfspaces",
    "plane_basis",
    "polygon_area",
    "reconstruct_polyhedron_from_halfspaces",
    "reconstruct_polyhedron_from_halfspaces_edge_clip",
    "reconstruct_polyhedron_from_halfspaces_incidence",
    "solve_halfspace_lp",
]
