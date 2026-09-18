from __future__ import annotations

import argparse
import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None

EPS = 1e-9
CACHE_SCHEMA_VERSION = 10


@dataclass(frozen=True)
class PolyModel:
    vertices: np.ndarray  # (N, 3)
    faces: list[list[int]]


@dataclass(frozen=True)
class ShadowContour:
    index: int
    normal: np.ndarray  # (3,)
    angle: float
    points: np.ndarray  # (M, 3)


@dataclass(frozen=True)
class HalfContour:
    source_index: int
    source_angle: float
    half_id: int
    normal: np.ndarray
    points: np.ndarray  # open polyline


def progress_iter(iterable, *, total: int | None, desc: str, use_tqdm: bool):
    if use_tqdm and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)
    return iterable


def parse_initial_model(path: Path) -> PolyModel:
    raw = path.read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in raw if ln.strip()]

    counts = None
    for ln in lines:
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
            counts = tuple(map(int, parts))
            break
    if counts is None:
        raise ValueError(f"Cannot parse counts from {path}")
    num_vertices, num_facets, _ = counts

    try:
        v_start = next(i for i, ln in enumerate(lines) if ln.lower().startswith("# vertices")) + 1
    except StopIteration as exc:
        raise ValueError(f"Cannot find vertices section in {path}") from exc

    vertices = np.zeros((num_vertices, 3), dtype=float)
    i = v_start
    parsed_vertices = 0
    while i < len(lines) and parsed_vertices < num_vertices:
        ln = lines[i]
        i += 1
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) != 4:
            continue
        vid = int(parts[0])
        vertices[vid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        parsed_vertices += 1

    if parsed_vertices != num_vertices:
        raise ValueError(f"Parsed {parsed_vertices} vertices, expected {num_vertices}")

    try:
        f_start = next(i for i, ln in enumerate(lines) if ln.lower().startswith("# facets")) + 1
    except StopIteration as exc:
        raise ValueError(f"Cannot find facets section in {path}") from exc

    faces: list[list[int]] = []
    i = f_start
    while i < len(lines) and len(faces) < num_facets:
        ln = lines[i]
        i += 1
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 6:
            continue
        num_sides = int(parts[1])
        vids: list[int] = []
        while i < len(lines) and len(vids) < num_sides:
            next_ln = lines[i]
            i += 1
            if next_ln.startswith("#"):
                continue
            vids.extend(int(tok) for tok in next_ln.split())
        faces.append(vids[:num_sides])

    if len(faces) != num_facets:
        raise ValueError(f"Parsed {len(faces)} facets, expected {num_facets}")

    return PolyModel(vertices=vertices, faces=faces)


def parse_merged_contour(path: Path) -> ShadowContour:
    idx_match = re.search(r"(\d+)$", path.name)
    if idx_match is None:
        raise ValueError(f"Cannot parse contour index from {path.name}")
    contour_idx = int(idx_match.group(1))

    normal = None
    angle = float("nan")
    points: list[list[float]] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("# norm"):
            parts = ln.split()
            normal = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
            continue
        if ln.startswith("# angle"):
            angle = float(ln.split()[-1])
            continue
        if ln.startswith("#"):
            continue
        vals = ln.split()
        if len(vals) != 3:
            raise ValueError(f"Expected 3D point in {path}, got: {ln}")
        points.append([float(vals[0]), float(vals[1]), float(vals[2])])

    if normal is None:
        raise ValueError(f"Missing '# norm' in {path}")

    pts = np.array(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError(f"Contour too short in {path}")

    n_norm = np.linalg.norm(normal)
    if n_norm < EPS:
        raise ValueError(f"Zero normal in {path}")
    normal = normal / n_norm

    return ShadowContour(index=contour_idx, normal=normal, angle=angle, points=pts)


def sorted_contour_files(shadow_dir: Path, pattern: str, max_contours: int | None) -> list[Path]:
    files = list(shadow_dir.glob(pattern))
    if not files:
        raise ValueError(f"No files matched {shadow_dir / pattern}")

    def sort_key(p: Path) -> tuple[int, str]:
        m = re.search(r"(\d+)$", p.name)
        if m is None:
            return (10**9, p.name)
        return (int(m.group(1)), p.name)

    files = sorted(files, key=sort_key)
    if max_contours is not None:
        return files[:max_contours]
    return files


def triangulate_faces(faces: list[list[int]]) -> np.ndarray:
    tris: list[list[int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        a = face[0]
        for i in range(1, len(face) - 1):
            tris.append([a, face[i], face[i + 1]])
    return np.array(tris, dtype=np.int32)


def unique_edges_from_faces(faces: list[list[int]]) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        n = len(face)
        if n < 2:
            continue
        for i in range(n):
            a = int(face[i])
            b = int(face[(i + 1) % n])
            if a == b:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            edges.add((lo, hi))
    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(sorted(edges), dtype=np.int32)


def polyhedron_center_of_mass(vertices: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    if not faces:
        return vertices.mean(axis=0)

    # Use original polygonal facets from InitialModel.
    # We orient each facet consistently relative to an inner reference point,
    # then integrate tetrahedra fan-wise inside each facet.
    ref_inside = vertices.mean(axis=0)
    sum_v6 = 0.0
    weighted = np.zeros(3, dtype=float)
    sum_w = 0.0
    weighted_abs = np.zeros(3, dtype=float)

    for face in faces:
        if len(face) < 3:
            continue
        pts = vertices[np.array(face, dtype=int)]
        if pts.shape[0] < 3:
            continue

        area_vec = np.zeros(3, dtype=float)
        for i in range(pts.shape[0]):
            area_vec += np.cross(pts[i], pts[(i + 1) % pts.shape[0]])
        face_cent = pts.mean(axis=0)
        if float(np.dot(area_vec, face_cent - ref_inside)) < 0.0:
            pts = pts[::-1].copy()

        p0 = pts[0]
        for i in range(1, pts.shape[0] - 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            v6 = float(np.dot(p0, np.cross(p1, p2)))
            sum_v6 += v6
            weighted += (p0 + p1 + p2) * v6
            w = abs(v6)
            sum_w += w
            weighted_abs += (p0 + p1 + p2) * w

    if abs(sum_v6) > EPS:
        return weighted / (4.0 * sum_v6)
    if sum_w > EPS:
        return weighted_abs / (4.0 * sum_w)
    return vertices.mean(axis=0)


def top_points_vertical_axis_point(vertices: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    if vertices.size == 0:
        return np.zeros(3, dtype=float)
    max_z = float(np.max(vertices[:, 2]))
    mask = np.isclose(vertices[:, 2], max_z, rtol=0.0, atol=atol)
    if not np.any(mask):
        idx = int(np.argmax(vertices[:, 2]))
        return vertices[idx].astype(float)
    top_pts = vertices[mask]
    return np.mean(top_pts, axis=0).astype(float)


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


def split_contour_by_arc(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = points
    if np.linalg.norm(pts[0] - pts[-1]) <= 1e-10:
        pts = pts[:-1]
    if pts.shape[0] < 4:
        raise ValueError("Contour too short for arc split")
    mid = pts.shape[0] // 2
    half_a = pts[: mid + 1].copy()
    half_b = np.vstack([pts[mid:], pts[:1]])
    return half_a, half_b


def split_contour_by_projected_line(
    points: np.ndarray,
    normal: np.ndarray,
    line_point: np.ndarray,
    line_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 3:
        raise ValueError("Contour must have at least 3 points")

    pts = points
    if np.linalg.norm(pts[0] - pts[-1]) <= 1e-10:
        pts = pts[:-1]
    if pts.shape[0] < 4:
        raise ValueError("Contour too short for robust projected split")

    plane_point = pts[0]
    plane_d = float(normal @ plane_point)
    lp = line_point - (float(normal @ line_point) - plane_d) * normal
    u_proj = line_dir - float(line_dir @ normal) * normal
    if np.linalg.norm(u_proj) <= 1e-10:
        return split_contour_by_arc(pts)
    u_proj = u_proj / np.linalg.norm(u_proj)
    side_axis = np.cross(normal, u_proj)
    side_norm = np.linalg.norm(side_axis)
    if side_norm <= 1e-10:
        return split_contour_by_arc(pts)
    side_axis = side_axis / side_norm

    expanded: list[np.ndarray] = []
    n = pts.shape[0]
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        expanded.append(p0.copy())

        d0 = float((p0 - lp) @ side_axis)
        d1 = float((p1 - lp) @ side_axis)

        if abs(d0) <= EPS and abs(d1) <= EPS:
            continue

        if d0 * d1 < 0.0:
            t = d0 / (d0 - d1)
            inter = p0 + t * (p1 - p0)
            expanded.append(inter)
        elif abs(d0) > EPS and abs(d1) <= EPS:
            expanded.append(p1.copy())

    if not expanded:
        raise ValueError("Failed to expand contour for cut")

    poly = np.array(expanded, dtype=float)
    if poly.shape[0] > 1 and np.linalg.norm(poly[0] - poly[-1]) <= 1e-10:
        poly = poly[:-1]
    if poly.shape[0] < 3:
        raise ValueError("Split contour has too few points after preprocessing")

    # Remove consecutive duplicates to stabilize cut-index detection.
    compact: list[np.ndarray] = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - compact[-1]) > 1e-10:
            compact.append(p)
    poly = np.array(compact, dtype=float)
    if poly.shape[0] < 3:
        raise ValueError("Split contour collapsed to degenerate polyline")

    cut_indices = [i for i, p in enumerate(poly) if abs(float((p - lp) @ side_axis)) <= 1e-7]
    if len(cut_indices) < 2:
        return split_contour_by_arc(pts)

    n = poly.shape[0]

    # Compress long runs of on-cut vertices to keep only run borders.
    cut_set = set(cut_indices)
    ordered = sorted(cut_set)
    runs: list[tuple[int, int]] = []
    run_start = ordered[0]
    prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((run_start, prev))
        run_start = idx
        prev = idx
    runs.append((run_start, prev))
    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        merged = (runs[-1][0], runs[0][1])
        runs = [merged] + runs[1:-1]

    candidate_indices: list[int] = []
    for a, b in runs:
        candidate_indices.append(a)
        if b != a:
            candidate_indices.append(b)
    candidate_indices = sorted(set(candidate_indices))
    if len(candidate_indices) < 2:
        candidate_indices = ordered

    def forward_path_len(i: int, j: int) -> float:
        length = 0.0
        k = i
        while k != j:
            k1 = (k + 1) % n
            length += float(np.linalg.norm(poly[k1] - poly[k]))
            k = k1
        return length

    # Choose pair that produces the most balanced split by arc lengths.
    best_pair = (candidate_indices[0], candidate_indices[1])
    best_score = -1.0
    best_len_ab = 0.0
    best_len_ba = 0.0
    for i in range(len(candidate_indices)):
        for j in range(i + 1, len(candidate_indices)):
            a = candidate_indices[i]
            b = candidate_indices[j]
            len_ab = forward_path_len(a, b)
            len_ba = forward_path_len(b, a)
            score = min(len_ab, len_ba)
            if score > best_score:
                best_score = score
                best_pair = (a, b)
                best_len_ab = len_ab
                best_len_ba = len_ba

    a_idx, b_idx = best_pair
    if a_idx > b_idx:
        a_idx, b_idx = b_idx, a_idx

    half_a = poly[a_idx : b_idx + 1]
    half_b = np.vstack([poly[b_idx:], poly[: a_idx + 1]])

    if half_a.shape[0] < 2 or half_b.shape[0] < 2:
        raise ValueError("Split produced degenerate half-contour")

    # Guard against tiny sliver split (common when cut overlaps long contour part).
    total_len = best_len_ab + best_len_ba
    if total_len > EPS:
        frac = min(best_len_ab, best_len_ba) / total_len
        if frac < 0.1:
            half_a, half_b = split_contour_by_arc(pts)

    return half_a, half_b


def build_half_contours(
    contours: list[ShadowContour],
    split_line_point: np.ndarray,
    split_line_dir: np.ndarray,
) -> list[HalfContour]:
    half0: list[HalfContour] = []
    half1: list[HalfContour] = []

    for c in contours:
        try:
            a, b = split_contour_by_projected_line(
                points=c.points,
                normal=c.normal,
                line_point=split_line_point,
                line_dir=split_line_dir,
            )
        except ValueError:
            a, b = split_contour_by_arc(c.points)

        lat = lateral_axis_for_normal(c.normal)
        a_lat = float(np.mean(a @ lat))
        b_lat = float(np.mean(b @ lat))
        if a_lat <= b_lat:
            low, high = a, b
        else:
            low, high = b, a

        half0.append(
            HalfContour(
                source_index=int(c.index),
                source_angle=float(c.angle),
                half_id=0,
                normal=c.normal,
                points=low,
            )
        )
        half1.append(
            HalfContour(
                source_index=int(c.index),
                source_angle=float(c.angle),
                half_id=1,
                normal=c.normal,
                points=high[::-1].copy(),
            )
        )

    return half0 + half1


def dedup_points(points: list[np.ndarray], tol: float = 1e-7) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for p in points:
        if not any(np.linalg.norm(p - q) <= tol for q in unique):
            unique.append(p)
    return unique


def half_contour_point_at_z(points: np.ndarray, z_level: float) -> np.ndarray:
    intersections: list[np.ndarray] = []
    n = points.shape[0]
    for i in range(n - 1):
        p0 = points[i]
        p1 = points[i + 1]
        z0 = float(p0[2])
        z1 = float(p1[2])

        dz0 = z0 - z_level
        dz1 = z1 - z_level

        if abs(dz0) <= EPS and abs(dz1) <= EPS:
            intersections.append(p0.copy())
            intersections.append(p1.copy())
            continue
        if abs(dz0) <= EPS:
            intersections.append(p0.copy())
            continue
        if abs(dz1) <= EPS:
            intersections.append(p1.copy())
            continue
        if dz0 * dz1 < 0.0:
            t = (z_level - z0) / (z1 - z0)
            intersections.append(p0 + t * (p1 - p0))

    intersections = dedup_points(intersections)
    if not intersections:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    if len(intersections) == 1:
        return intersections[0]

    center = np.mean(points, axis=0)
    dists = [float(np.linalg.norm(p - center)) for p in intersections]
    return intersections[int(np.argmin(dists))]


def closest_point_to_lines(points: np.ndarray, directions: np.ndarray) -> np.ndarray:
    a = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    eye = np.eye(3, dtype=float)

    for p, d in zip(points, directions):
        dn = np.linalg.norm(d)
        if dn < EPS:
            continue
        d = d / dn
        proj = eye - np.outer(d, d)
        a += proj
        b += proj @ p

    if np.linalg.norm(a) < EPS:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
        return x


def encode_array(arr: np.ndarray, dtype: str) -> dict[str, object]:
    np_dtype = np.dtype(dtype)
    casted = np.ascontiguousarray(arr.astype(np_dtype, copy=False))
    b64 = base64.b64encode(casted.tobytes()).decode("ascii")
    return {
        "dtype": np_dtype.str,
        "shape": list(casted.shape),
        "data_b64": b64,
    }


def build_model_cache(
    model_name: str,
    data_root: Path,
    distance_scale: float,
    max_contours: int | None,
    cut_x_mode: str,
    cut_x_offset: float,
) -> dict[str, object]:
    model_path = data_root / model_name / "InitialModel"
    shadow_dir = data_root / model_name / "shadow"

    model = parse_initial_model(model_path)
    contour_files = sorted_contour_files(shadow_dir, "merged-cont*", max_contours)
    contours = [parse_merged_contour(p) for p in contour_files]

    vertices = model.vertices
    model_center = polyhedron_center_of_mass(vertices, model.faces)
    centered_vertices = vertices - model_center
    model_radius = float(np.linalg.norm(centered_vertices, axis=1).max())

    # Split by projection of a vertical line passing through the averaged
    # highest point(s) of the model.
    top_axis_point = top_points_vertical_axis_point(vertices)
    split_line_point_world = np.array(
        [float(top_axis_point[0]), float(top_axis_point[1]), float(top_axis_point[2])],
        dtype=float,
    )
    split_line_dir_world = np.array([0.0, 0.0, 1.0], dtype=float)

    half_contours = build_half_contours(
        contours=contours,
        split_line_point=split_line_point_world,
        split_line_dir=split_line_dir_world,
    )
    tri_indices = triangulate_faces(model.faces)
    edge_indices = unique_edges_from_faces(model.faces)

    z_min = float(np.min(np.vstack([c.points for c in contours])[:, 2]))
    z_max = float(np.max(np.vstack([c.points for c in contours])[:, 2]))

    half_payload: list[dict[str, object]] = []
    for i, hc in enumerate(half_contours):
        support_max = float(np.max(centered_vertices @ hc.normal))
        push = support_max + distance_scale * model_radius
        base = hc.points - model_center
        is_right = int(hc.half_id) == 1
        shifted = base - hc.normal[None, :] * push if is_right else base + hc.normal[None, :] * push
        normal_display = -hc.normal if is_right else hc.normal
        half_payload.append(
            {
                "seq_index": int(i),
                "source_index": int(hc.source_index),
                "angle": float(hc.source_angle),
                "half_id": int(hc.half_id),
                "normal": [float(normal_display[0]), float(normal_display[1]), float(normal_display[2])],
                "normal_calc": [float(hc.normal[0]), float(hc.normal[1]), float(hc.normal[2])],
                "points_calc": hc.points.astype(np.float32).round(6).tolist(),
                "points": shifted.astype(np.float32).round(6).tolist(),
            }
        )

    return {
        "name": model_name,
        "num_vertices": int(centered_vertices.shape[0]),
        "num_faces": int(len(model.faces)),
        "model_center": model_center.astype(float).tolist(),
        "model_radius": float(model_radius),
        "z_min": z_min,
        "z_max": z_max,
        "split_line_point_display": (split_line_point_world - model_center).astype(float).tolist(),
        "split_line_dir": split_line_dir_world.astype(float).tolist(),
        "vertices": encode_array(centered_vertices, "<f4"),
        "triangles": encode_array(tri_indices, "<u4"),
        "edges": encode_array(edge_indices, "<u4"),
        "half_contours": half_payload,
    }


def build_html(cache_obj: dict[str, object], windows_cache_obj: dict[str, object] | None = None) -> str:
    cache_json = json.dumps(cache_obj, ensure_ascii=False, separators=(",", ":"))
    windows_json = (
        json.dumps(windows_cache_obj, ensure_ascii=False, separators=(",", ":"))
        if windows_cache_obj is not None
        else "null"
    )
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Full-Circle Split Contour Viewer (Dynamic)</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #14213d;
      --muted: #4f5d75;
      --line: #dbe4ee;
      --accent: #0077b6;
      --accent-2: #d62828;
    }
    html, body { margin: 0; padding: 0; background: linear-gradient(160deg, #eef5ff 0%, #f8fbff 50%, #f0f4f8 100%); color: var(--ink); font-family: \"Trebuchet MS\", \"Segoe UI\", sans-serif; }
    .app { max-width: 1450px; margin: 0 auto; padding: 14px; display: grid; gap: 12px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; box-shadow: 0 8px 24px rgba(20,33,61,0.06); }
    .field { display: grid; gap: 5px; }
    .field label { font-size: 12px; color: var(--muted); }
    .field input:not([type=\"checkbox\"]):not([type=\"radio\"]), .field select { width: 100%; }
    .checkline { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); }
    .checkline input[type=\"checkbox\"] { width: auto; margin: 0; }
    .top-row { display: grid; grid-template-columns: 1fr; padding: 10px 12px; }
    .top-main { display: grid; grid-template-columns: 1fr 260px; gap: 10px; align-items: stretch; }
    .plot { min-height: 200px; border-radius: 12px; overflow: hidden; }
    .side-controls { padding: 10px; display: grid; gap: 10px; align-content: start; }
    .bottom-main { display: grid; grid-template-columns: 1fr 260px; gap: 10px; align-items: stretch; }
    .stack { display: grid; gap: 10px; }
    .compare-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .ideal-plot-card { overflow: hidden; }
    .ideal-plot-toolbar { padding: 8px 12px; display: flex; justify-content: flex-end; border-bottom: 1px solid var(--line); }
    .ideal-plot-card .plot { border: 0; border-radius: 0; box-shadow: none; }
    .all-z-toolbar { padding: 10px 12px; display: flex; flex-wrap: wrap; align-items: center; gap: 16px; }
    .all-z-toolbar strong { margin-right: 4px; }
    .all-z-overlay-card { overflow: hidden; }
    .all-z-overlay-card .all-z-toolbar { border-bottom: 1px solid var(--line); }
    .all-z-overlay-card .plot { border: 0; border-radius: 0; box-shadow: none; }
    .section-note { padding: 10px 12px; line-height: 1.45; }
    .section-note strong { color: var(--accent); }
    .params { padding: 10px; display: grid; gap: 9px; align-content: start; }
    .z-wide { padding: 10px 12px; }
    .small { font-size: 12px; color: var(--muted); }
    @media (max-width: 1100px) {
      .top-main { grid-template-columns: 1fr; }
      .bottom-main { grid-template-columns: 1fr; }
      .compare-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"app\">
    <div class=\"panel top-row\">
      <div class=\"field\" style=\"max-width:300px;\">
        <label for=\"modelSel\">Model</label>
        <select id=\"modelSel\"></select>
        <div class=\"small\">Выберите модель огранки. Для каждого Z одна и та же метрика считается по наблюдаемым теневым контурам и по идеальным теням InitialModel.</div>
      </div>
    </div>

    <div class=\"top-main\">
      <div id=\"topPlot\" class=\"panel plot\" style=\"height:56vh;\"></div>
      <div class=\"panel side-controls\">
        <div class=\"field\">
          <label for=\"projIdx\">Half-contour index</label>
          <input id=\"projIdx\" type=\"range\" min=\"0\" max=\"0\" value=\"0\" step=\"1\" />
          <div class=\"small\">Показывает выбранный полуконтур в верхнем 3D. На численный расчет функции не влияет.</div>
        </div>
        <div class=\"field\">
          <label for=\"projMode\">Projection mode</label>
          <select id=\"projMode\"><option value=\"perspective\">Perspective</option><option value=\"orthographic\">Parallel</option></select>
          <div class=\"small\">Только способ отображения камеры: перспектива или параллельная проекция.</div>
        </div>
        <div class=\"field\">
          <label for=\"zoomAbs\">3D zoom (absolute)</label>
          <input id=\"zoomAbs\" type=\"range\" min=\"0.40\" max=\"3.00\" value=\"1.00\" step=\"0.01\" />
          <div class=\"small\" id=\"zoomAbsLabel\">1.00x</div>
          <div class=\"small\">Абсолютный масштаб обеих 3D-сцен. Не сбрасывается при изменении других контролов.</div>
        </div>
      </div>
    </div>

    <div class=\"bottom-main\">
      <div class=\"stack\">
        <div id=\"contextPlot\" class=\"panel plot\" style=\"height:62vh;\"></div>
        <div class=\"compare-grid\">
          <div id=\"fnPlot\" class=\"panel plot\" style=\"height:38vh;\"></div>
          <div class=\"panel ideal-plot-card\">
            <div class=\"ideal-plot-toolbar\">
              <label class=\"checkline\"><input id=\"overlayObservedOnIdealChk\" type=\"checkbox\" />Наложить observed-графики</label>
            </div>
            <div id=\"idealFnPlot\" class=\"plot\" style=\"height:34vh;\"></div>
          </div>
        </div>
      </div>
      <div class=\"panel params\">
        <div class=\"field\">
          <label for=\"zoomAbs2\">3D zoom (absolute)</label>
          <input id=\"zoomAbs2\" type=\"range\" min=\"0.40\" max=\"3.00\" value=\"1.00\" step=\"0.01\" />
          <div class=\"small\" id=\"zoomAbsLabel2\">1.00x</div>
          <div class=\"small\">Дублирует верхний ползунок масштаба и всегда синхронизирован с ним.</div>
        </div>
        <div class=\"field\">
          <label>Context view presets</label>
          <div style=\"display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;\">
            <button id=\"viewXBtn\" type=\"button\">View X</button>
            <button id=\"viewYBtn\" type=\"button\">View Y</button>
            <button id=\"viewZBtn\" type=\"button\">View Z</button>
          </div>
          <div class=\"small\">Быстрый поворот только второй 3D-сцены (context) к осям X/Y/Z.</div>
        </div>
        <div class=\"field\">
          <label for=\"zStepInput\">Z step (dynamic)</label>
          <input id=\"zStepInput\" type=\"number\" min=\"0.0005\" step=\"0.0005\" value=\"0.01\" />
          <div class=\"small\">Шаг по Z для дискретизации. Меньше: точнее, но медленнее. Больше: быстрее, но грубее.</div>
        </div>
        <div class=\"field\">
          <label for=\"windowInput\">Window size (dynamic)</label>
          <input id=\"windowInput\" type=\"number\" min=\"2\" step=\"1\" value=\"10\" />
          <div class=\"small\">Размер скользящего окна по контурам. Больше: сильнее сглаживание. Меньше: больше локальных колебаний.</div>
        </div>
        <div class=\"field\">
          <label for=\"rmsMinPctInput\">RMS minima threshold (%)</label>
          <input id=\"rmsMinPctInput\" type=\"number\" min=\"0\" max=\"100\" step=\"0.5\" value=\"10\" />
          <div class=\"small\">Показывать локальные минимумы RMS не выше чем global_min + (max-min)*pct/100.</div>
        </div>
        <div class=\"field\">
          <label>Слои 3D-сцены</label>
          <label class=\"checkline\"><input id=\"showRedSectionChk\" type=\"checkbox\" checked />Наблюдаемые контуры</label>
          <label class=\"checkline\"><input id=\"showIdealSectionChk\" type=\"checkbox\" checked />Идеальные контуры InitialModel</label>
          <label class=\"checkline\"><input id=\"showYellowSectionChk\" type=\"checkbox\" checked />Точки локальных минимумов RMS</label>
          <label class=\"checkline\"><input id=\"sharedYScaleChk\" type=\"checkbox\" checked />Одинаковый масштаб Y</label>
          <div class=\"small\">Управление отображением сечений на 3D-сцене.</div>
        </div>
        <div class=\"field\">
          <label for=\"trimBottom\">Trim bottom</label>
          <input id=\"trimBottom\" type=\"number\" min=\"0\" value=\"2\" />
          <div class=\"small\">Сколько нижних Z-уровней скрыть в ползунке и графике.</div>
        </div>
        <div class=\"field\">
          <label for=\"trimTop\">Trim top</label>
          <input id=\"trimTop\" type=\"number\" min=\"0\" value=\"20\" />
          <div class=\"small\">Сколько верхних Z-уровней скрыть в ползунке и графике.</div>
        </div>
        <div class=\"small\" id=\"status\"></div>
      </div>
    </div>

    <div class=\"panel z-wide\">
      <div class=\"field\">
        <label for=\"zIdx\">Z level index</label>
        <input id=\"zIdx\" type=\"range\" min=\"0\" max=\"0\" value=\"0\" step=\"1\" />
        <div class=\"small\">Текущий срез по Z, для которого строится функция расстояний.</div>
      </div>
    </div>

    <div id=\"allMinimaPlot\" class=\"panel plot\" style=\"height:80vh;\"></div>

    <div class=\"panel all-z-overlay-card\">
      <div class=\"all-z-toolbar\">
        <strong>Все Z в одной 3D-сцене:</strong>
        <label class=\"checkline\"><input id=\"showAllZModelChk\" type=\"checkbox\" checked />InitialModel</label>
        <label class=\"checkline\"><input id=\"showAllZObservedMinimaChk\" type=\"checkbox\" checked />Observed minima</label>
        <label class=\"checkline\"><input id=\"showAllZIdealMinimaChk\" type=\"checkbox\" checked />Ideal minima</label>
      </div>
      <div id=\"allMinimaOverlayPlot\" class=\"plot\" style=\"height:80vh;\"></div>
    </div>
  </div>

  <script id=\"cacheData\" type=\"application/json\">__CACHE_JSON__</script>
  <script id=\"windowsCacheData\" type=\"application/json\">__WINDOWS_CACHE_JSON__</script>
  <script>
    const EPS = 1e-9;
    if (typeof Plotly === 'undefined') {
      document.body.innerHTML = '<div style=\"padding:16px;font-family:sans-serif;color:#b00020;\">Plotly failed to load. Check internet access or provide local plotly bundle.</div>';
      throw new Error('Plotly failed to load');
    }
    const CACHE = JSON.parse(document.getElementById('cacheData').textContent);
    let PRECOMPUTED = null;
    try {
      PRECOMPUTED = JSON.parse(document.getElementById('windowsCacheData').textContent);
    } catch (_) {
      PRECOMPUTED = null;
    }
    const HAS_PRECOMPUTED = PRECOMPUTED && PRECOMPUTED.models && typeof PRECOMPUTED.models === 'object';

    function decodeArray(spec) {
      const bin = atob(spec.data_b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const buf = bytes.buffer;
      let arr;
      if (spec.dtype.includes('f4')) arr = new Float32Array(buf);
      else if (spec.dtype.includes('u4')) arr = new Uint32Array(buf);
      else throw new Error('Unsupported dtype: ' + spec.dtype);
      return { data: arr, shape: spec.shape };
    }

    function toPointList(data, shape) {
      const [n, m] = shape;
      const out = new Array(n);
      let k = 0;
      for (let i = 0; i < n; i++) {
        const row = new Array(m);
        for (let j = 0; j < m; j++) row[j] = data[k + j];
        out[i] = row;
        k += m;
      }
      return out;
    }

    function norm3(v) {
      return Math.hypot(v[0], v[1], v[2]);
    }

    function solve3x3(A, b) {
      const m = [
        [A[0][0], A[0][1], A[0][2], b[0]],
        [A[1][0], A[1][1], A[1][2], b[1]],
        [A[2][0], A[2][1], A[2][2], b[2]],
      ];

      for (let col = 0; col < 3; col++) {
        let pivot = col;
        for (let r = col + 1; r < 3; r++) {
          if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
        }
        if (Math.abs(m[pivot][col]) < 1e-12) return null;
        if (pivot !== col) {
          const tmp = m[col];
          m[col] = m[pivot];
          m[pivot] = tmp;
        }

        const div = m[col][col];
        for (let c = col; c < 4; c++) m[col][c] /= div;

        for (let r = 0; r < 3; r++) {
          if (r === col) continue;
          const f = m[r][col];
          for (let c = col; c < 4; c++) m[r][c] -= f * m[col][c];
        }
      }

      return [m[0][3], m[1][3], m[2][3]];
    }

    function closestPointToLines(points, dirs) {
      const I = [[1,0,0],[0,1,0],[0,0,1]];
      const A = [[0,0,0],[0,0,0],[0,0,0]];
      const b = [0,0,0];

      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        const d0 = dirs[i];
        const dn = norm3(d0);
        if (dn < EPS) continue;
        const d = [d0[0]/dn, d0[1]/dn, d0[2]/dn];

        const proj = [
          [I[0][0]-d[0]*d[0], I[0][1]-d[0]*d[1], I[0][2]-d[0]*d[2]],
          [I[1][0]-d[1]*d[0], I[1][1]-d[1]*d[1], I[1][2]-d[1]*d[2]],
          [I[2][0]-d[2]*d[0], I[2][1]-d[2]*d[1], I[2][2]-d[2]*d[2]],
        ];

        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) A[r][c] += proj[r][c];
          b[r] += proj[r][0]*p[0] + proj[r][1]*p[1] + proj[r][2]*p[2];
        }
      }

      const x = solve3x3(A, b);
      if (!x) return [NaN, NaN, NaN];
      return x;
    }

    function dedupPoints(points, tol = 1e-7) {
      const out = [];
      for (const p of points) {
        let dup = false;
        for (const q of out) {
          const d = Math.hypot(p[0]-q[0], p[1]-q[1], p[2]-q[2]);
          if (d <= tol) {
            dup = true;
            break;
          }
        }
        if (!dup) out.push(p);
      }
      return out;
    }

    function halfContourPointAtZ(points, zLevel) {
      const intersections = [];
      for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i];
        const p1 = points[i + 1];
        const z0 = p0[2];
        const z1 = p1[2];
        const dz0 = z0 - zLevel;
        const dz1 = z1 - zLevel;

        if (Math.abs(dz0) <= EPS && Math.abs(dz1) <= EPS) {
          intersections.push([p0[0], p0[1], p0[2]], [p1[0], p1[1], p1[2]]);
          continue;
        }
        if (Math.abs(dz0) <= EPS) {
          intersections.push([p0[0], p0[1], p0[2]]);
          continue;
        }
        if (Math.abs(dz1) <= EPS) {
          intersections.push([p1[0], p1[1], p1[2]]);
          continue;
        }
        if (dz0 * dz1 < 0) {
          const t = (zLevel - z0) / (z1 - z0);
          intersections.push([
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            p0[2] + t * (p1[2] - p0[2]),
          ]);
        }
      }

      const uniq = dedupPoints(intersections);
      if (uniq.length === 0) return [NaN, NaN, NaN];
      if (uniq.length === 1) return uniq[0];

      const center = [0,0,0];
      for (const p of points) {
        center[0] += p[0];
        center[1] += p[1];
        center[2] += p[2];
      }
      center[0] /= points.length;
      center[1] /= points.length;
      center[2] /= points.length;

      let best = uniq[0];
      let bestD = Infinity;
      for (const p of uniq) {
        const d = Math.hypot(p[0]-center[0], p[1]-center[1], p[2]-center[2]);
        if (d < bestD) {
          bestD = d;
          best = p;
        }
      }
      return best;
    }

    const state = {
      modelName: null,
      contourIdx: 0,
      projection: 'perspective',
      topCamera: null,
      contextCamera: null,
      allMinimaCameraLeft: null,
      allMinimaCameraRight: null,
      allMinimaCameraIdeal: null,
      allMinimaOverlayCamera: null,
      zoomAbs: 1.0,
      zStep: 0.01,
      window: 10,
      rmsMinPct: 10.0,
      showRedSection: true,
      showIdealSection: true,
      showYellowSection: true,
      sharedYScale: true,
      overlayObservedOnIdeal: false,
      showAllZModel: true,
      showAllZObservedMinima: true,
      showAllZIdealMinima: true,
      zIdx: 0,
      trimBottom: 2,
      trimTop: 20,
      hoveredDistIdx: null,
      computed: null,
      idealComputed: null,
      rmsMinimaCacheKey: null,
      rmsMinimaCache: null,
      idealRmsMinimaCacheKey: null,
      idealRmsMinimaCache: null,
      allMinimaCloudCacheKey: null,
      allMinimaCloudCache: null,
      idealAllMinimaCloudCacheKey: null,
      idealAllMinimaCloudCache: null,
    };

    const modelSel = document.getElementById('modelSel');
    const projIdx = document.getElementById('projIdx');
    const projMode = document.getElementById('projMode');
    const zoomAbs = document.getElementById('zoomAbs');
    const zoomAbsLabel = document.getElementById('zoomAbsLabel');
    const zoomAbs2 = document.getElementById('zoomAbs2');
    const zoomAbsLabel2 = document.getElementById('zoomAbsLabel2');
    const zStepInput = document.getElementById('zStepInput');
    const windowInput = document.getElementById('windowInput');
    const rmsMinPctInput = document.getElementById('rmsMinPctInput');
    const showRedSectionChk = document.getElementById('showRedSectionChk');
    const showIdealSectionChk = document.getElementById('showIdealSectionChk');
    const showYellowSectionChk = document.getElementById('showYellowSectionChk');
    const sharedYScaleChk = document.getElementById('sharedYScaleChk');
    const overlayObservedOnIdealChk = document.getElementById('overlayObservedOnIdealChk');
    const showAllZModelChk = document.getElementById('showAllZModelChk');
    const showAllZObservedMinimaChk = document.getElementById('showAllZObservedMinimaChk');
    const showAllZIdealMinimaChk = document.getElementById('showAllZIdealMinimaChk');
    const zIdx = document.getElementById('zIdx');
    const trimBottom = document.getElementById('trimBottom');
    const trimTop = document.getElementById('trimTop');
    const status = document.getElementById('status');
    const allMinimaPlot = document.getElementById('allMinimaPlot');
    const allMinimaOverlayPlot = document.getElementById('allMinimaOverlayPlot');
    const viewXBtn = document.getElementById('viewXBtn');
    const viewYBtn = document.getElementById('viewYBtn');
    const viewZBtn = document.getElementById('viewZBtn');

    const modelNames = Object.keys(CACHE.models);
    modelNames.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      modelSel.appendChild(opt);
    });
    state.modelName = modelNames[0] || null;
    modelSel.value = state.modelName;

    const decodedModels = {};
    const decodedPrecomputed = {};

    function ensureModelDecoded(name) {
      if (decodedModels[name]) return decodedModels[name];
      const m = CACHE.models[name];
      const v = decodeArray(m.vertices);
      const t = decodeArray(m.triangles);
      const e = decodeArray(m.edges);
      const vertices = toPointList(v.data, v.shape);
      const triangles = toPointList(t.data, t.shape);
      const edges = toPointList(e.data, e.shape);
      const decoded = { ...m, vertices, triangles, edges };
      decodedModels[name] = decoded;
      return decoded;
    }

    function precomputedModel(name) {
      if (!HAS_PRECOMPUTED) return null;
      const m = PRECOMPUTED.models ? PRECOMPUTED.models[name] : null;
      if (!m || typeof m !== 'object') return null;
      return m;
    }

    function updateControlBoundsForModel() {
      const m = ensureModelDecoded(state.modelName);
      const nHalf = Array.isArray(m.half_contours) ? m.half_contours.length : 0;
      const pm = precomputedModel(state.modelName);

      if (pm && Number.isFinite(Number(pm.window_min)) && Number.isFinite(Number(pm.window_max))) {
        const wMin = Math.max(2, Math.round(Number(pm.window_min)));
        const wMax = Math.max(wMin, Math.min(nHalf, Math.round(Number(pm.window_max))));
        windowInput.min = String(wMin);
        windowInput.max = String(wMax);
        state.window = Math.max(wMin, Math.min(wMax, Math.round(Number(state.window) || wMin)));
        windowInput.value = String(state.window);

        const pz = Number(pm.z_step);
        if (Number.isFinite(pz) && pz > 0) {
          state.zStep = pz;
          zStepInput.value = String(pz);
        }
        zStepInput.disabled = true;
      } else {
        windowInput.min = '2';
        windowInput.max = String(Math.max(2, nHalf));
        state.window = Math.max(2, Math.min(nHalf, Math.round(Number(state.window) || 2)));
        windowInput.value = String(state.window);
        zStepInput.disabled = false;
      }
    }

    function precomputedFunction(name, windowSize) {
      const pm = precomputedModel(name);
      if (!pm || !pm.functions) return null;
      const key = `w=${windowSize}`;
      const fn = pm.functions[key];
      if (!fn) return null;

      const cacheKey = `${name}|${key}`;
      if (decodedPrecomputed[cacheKey]) return decodedPrecomputed[cacheKey];

      const z = decodeArray(fn.z_levels);
      const d = decodeArray(fn.distances);
      const r = fn.fit_rms ? decodeArray(fn.fit_rms) : null;
      const lp = fn.line_points ? decodeArray(fn.line_points) : null;
      const zLevels = z.data;
      const distances = d.data;
      const fitRms = r ? r.data : null;
      const linePoints = lp ? lp.data : null;
      const nLevels = Number(z.shape && z.shape[0] ? z.shape[0] : zLevels.length);
      const nHalf = Number(d.shape && d.shape[0] ? d.shape[0] : 0);
      const out = {
        zLevels,
        distances,
        fitRms,
        linePoints,
        nLevels,
        nHalf,
        zStep: Number(fn.z_step || pm.z_step || PRECOMPUTED.z_step || state.zStep),
      };
      decodedPrecomputed[cacheKey] = out;
      return out;
    }

    function contourCamera(normal) {
      const eye = [-normal[0], -normal[1], -normal[2] + 0.2];
      const norm = Math.hypot(eye[0], eye[1], eye[2]) || 1;
      const k = 1.35 / norm;
      return { x: eye[0] * k, y: eye[1] * k, z: eye[2] * k };
    }

    function scaledEye(eye, zoomAbsValue) {
      const z = Math.max(0.40, Number(zoomAbsValue) || 1.0);
      return {
        x: eye.x / z,
        y: eye.y / z,
        z: eye.z / z,
      };
    }

    function meshTrace(m) {
      const vx = m.vertices.map(p => p[0]);
      const vy = m.vertices.map(p => p[1]);
      const vz = m.vertices.map(p => p[2]);
      return {
        type: 'mesh3d',
        x: vx, y: vy, z: vz,
        i: m.triangles.map(t => t[0]),
        j: m.triangles.map(t => t[1]),
        k: m.triangles.map(t => t[2]),
        opacity: 0.25,
        color: '#86b3d1',
        flatshading: true,
        hoverinfo: 'skip',
        showscale: false,
      };
    }

    function edgeTrace(m) {
      if (!m.edgeSegments) {
        const segX = [];
        const segY = [];
        const segZ = [];
        for (const e of m.edges) {
          const a = Number(e[0]);
          const b = Number(e[1]);
          const p0 = m.vertices[a];
          const p1 = m.vertices[b];
          segX.push(p0[0], p1[0], null);
          segY.push(p0[1], p1[1], null);
          segZ.push(p0[2], p1[2], null);
        }
        m.edgeSegments = { x: segX, y: segY, z: segZ };
      }
      return {
        type: 'scatter3d',
        mode: 'lines',
        x: m.edgeSegments.x,
        y: m.edgeSegments.y,
        z: m.edgeSegments.z,
        line: { color: '#355c7d', width: 2 },
        hoverinfo: 'skip',
        showlegend: false,
      };
    }

    function contourLineTrace(points, color, width) {
      const x = points.map(p => p[0]);
      const y = points.map(p => p[1]);
      const z = points.map(p => p[2]);
      return {
        type: 'scatter3d',
        mode: 'lines',
        x, y, z,
        line: { color, width },
        hoverinfo: 'skip',
        showlegend: false,
      };
    }

    function cutLineTrace(m) {
      const p = m.split_line_point_display;
      const d = m.split_line_dir;
      const dn = Math.hypot(d[0], d[1], d[2]) || 1;
      const du = [d[0]/dn, d[1]/dn, d[2]/dn];
      const ext = Math.max(1e-6, Number(m.model_radius || 1)) * 0.7;
      const p0 = [p[0] - du[0] * ext, p[1] - du[1] * ext, p[2] - du[2] * ext];
      const p1 = [p[0] + du[0] * ext, p[1] + du[1] * ext, p[2] + du[2] * ext];
      return {
        type: 'scatter3d',
        mode: 'lines',
        x: [p0[0], p1[0]],
        y: [p0[1], p1[1]],
        z: [p0[2], p1[2]],
        line: { color: '#ff8c42', width: 6 },
        hoverinfo: 'skip',
        showlegend: false,
      };
    }

    function currentZValue() {
      if (!state.computed || !state.computed.zLevels || state.computed.zLevels.length === 0) return 0;
      const m = ensureModelDecoded(state.modelName);
      const zi = Math.max(0, Math.min(state.computed.zLevels.length - 1, Number(state.zIdx) || 0));
      const zWorld = Number(state.computed.zLevels[zi]);
      const zCenter = Array.isArray(m.model_center) ? Number(m.model_center[2] || 0) : 0;
      return zWorld - zCenter;
    }

    function zPlaneTrace(m, zVal) {
      const xs = m.vertices.map(p => p[0]);
      const ys = m.vertices.map(p => p[1]);
      const xMin = Math.min(...xs);
      const xMax = Math.max(...xs);
      const yMin = Math.min(...ys);
      const yMax = Math.max(...ys);
      return {
        type: 'mesh3d',
        x: [xMin, xMax, xMax, xMin],
        y: [yMin, yMin, yMax, yMax],
        z: [zVal, zVal, zVal, zVal],
        i: [0, 0],
        j: [1, 2],
        k: [2, 3],
        opacity: 0.22,
        color: '#06d6a0',
        hoverinfo: 'skip',
        showscale: false,
      };
    }

    function dedup3(points, tol = 1e-7) {
      const out = [];
      for (const p of points) {
        let dup = false;
        for (const q of out) {
          if (Math.hypot(p[0]-q[0], p[1]-q[1], p[2]-q[2]) <= tol) {
            dup = true;
            break;
          }
        }
        if (!dup) out.push(p);
      }
      return out;
    }

    function dedup2(points, tol = 1e-7) {
      const out = [];
      for (const p of points) {
        let dup = false;
        for (const q of out) {
          if (Math.hypot(p[0] - q[0], p[1] - q[1]) <= tol) {
            dup = true;
            break;
          }
        }
        if (!dup) out.push(p);
      }
      return out;
    }

    function simplifyCollinear2D(poly, tol = 1e-8) {
      if (!Array.isArray(poly) || poly.length <= 3) return poly ? poly.slice() : [];
      const out = poly.slice();
      let changed = true;
      while (changed && out.length > 3) {
        changed = false;
        for (let i = 0; i < out.length; i++) {
          const a = out[(i - 1 + out.length) % out.length];
          const b = out[i];
          const c = out[(i + 1) % out.length];
          const abx = b[0] - a[0];
          const aby = b[1] - a[1];
          const bcx = c[0] - b[0];
          const bcy = c[1] - b[1];
          const cross = Math.abs(abx * bcy - aby * bcx);
          const scale = Math.max(1.0, Math.hypot(abx, aby) * Math.hypot(bcx, bcy));
          if (cross <= tol * scale) {
            out.splice(i, 1);
            changed = true;
            break;
          }
        }
      }
      return out;
    }

    function sectionPointsFromEdges(m, zVal) {
      const pts = [];
      const eps = 1e-9;
      for (const e of m.edges || []) {
        const p0 = m.vertices[e[0]];
        const p1 = m.vertices[e[1]];
        const dz0 = p0[2] - zVal;
        const dz1 = p1[2] - zVal;

        if (Math.abs(dz0) <= eps && Math.abs(dz1) <= eps) {
          pts.push([p0[0], p0[1], zVal]);
          pts.push([p1[0], p1[1], zVal]);
          continue;
        }
        if (Math.abs(dz0) <= eps) {
          pts.push([p0[0], p0[1], zVal]);
          continue;
        }
        if (Math.abs(dz1) <= eps) {
          pts.push([p1[0], p1[1], zVal]);
          continue;
        }
        if (dz0 * dz1 < 0) {
          const t = (zVal - p0[2]) / (p1[2] - p0[2]);
          pts.push([
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            zVal,
          ]);
        }
      }
      return dedup3(pts, 1e-7);
    }

    function sectionPolygonFromEdges(m, zVal) {
      const section = sectionPointsFromEdges(m, zVal);
      const pts = section.map(p => [p[0], p[1]]);

      const uniq = dedup2(pts, 1e-7);
      if (uniq.length < 3) return [];

      let cx = 0;
      let cy = 0;
      for (const p of uniq) {
        cx += p[0];
        cy += p[1];
      }
      cx /= uniq.length;
      cy /= uniq.length;

      uniq.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
      return simplifyCollinear2D(uniq, 1e-8);
    }

    function sectionTrace(m, zVal) {
      if (!m.sectionCache) m.sectionCache = {};
      const key = Number(zVal).toFixed(6);
      if (m.sectionCache[key]) return m.sectionCache[key];
      const poly2 = sectionPolygonFromEdges(m, zVal);
      const segX = [];
      const segY = [];
      const segZ = [];
      if (poly2.length >= 2) {
        for (const p of poly2) {
          segX.push(p[0]);
          segY.push(p[1]);
          segZ.push(zVal);
        }
        segX.push(poly2[0][0]);
        segY.push(poly2[0][1]);
        segZ.push(zVal);
      }

      const lineTrace = {
        type: 'scatter3d',
        mode: 'lines',
        x: segX,
        y: segY,
        z: segZ,
        line: { color: '#00a6a6', width: 8 },
        hoverinfo: 'skip',
        showlegend: false,
      };

      const sectionObj = { line: lineTrace, nGon: poly2.length };
      m.sectionCache[key] = sectionObj;
      return sectionObj;
    }

    function oppositeSideAnnotation(m, activeIdx, nGon, textPrefix, color, y) {
      if (!Number.isFinite(nGon) || nGon < 3) return [];
      let sumX = 0;
      let count = 0;
      for (const ci of activeIdx) {
        const pts = m.half_contours[ci]?.points || [];
        for (const p of pts) {
          sumX += Number(p[0] || 0);
          count += 1;
        }
      }
      const meanX = count > 0 ? sumX / count : 0;
      const onRight = meanX >= 0;
      return [{
        xref: 'paper',
        yref: 'paper',
        x: onRight ? 0.05 : 0.95,
        y,
        text: `${textPrefix}${nGon}-gon`,
        showarrow: false,
        xanchor: onRight ? 'left' : 'right',
        yanchor: 'middle',
        align: onRight ? 'left' : 'right',
        font: { color, size: 16 },
        bgcolor: 'rgba(255,255,255,0.72)',
        bordercolor: color,
        borderwidth: 1,
        borderpad: 4,
      }];
    }

    function computeFunctionsFromHalfPoints(halfPoints, halves, nLevels, window, source) {
      const nHalf = halves.length;
      const linePoints = new Float32Array(nHalf * nLevels * 3);
      linePoints.fill(NaN);
      const fitRms = new Float32Array(nHalf * nLevels);
      fitRms.fill(NaN);

      for (let ws = 0; ws < nHalf; ws++) {
        for (let zi = 0; zi < nLevels; zi++) {
          const candPoints = [];
          const candDirs = [];
          for (let j = 0; j < window; j++) {
            const idx = (ws + j) % nHalf;
            const off = (idx * nLevels + zi) * 3;
            const p = [halfPoints[off], halfPoints[off + 1], halfPoints[off + 2]];
            if (Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2])) {
              candPoints.push(p);
              candDirs.push(halves[idx].normal_calc || halves[idx].normal);
            }
          }
          if (candPoints.length < 2) continue;
          const x = closestPointToLines(candPoints, candDirs);
          const outOff = (ws * nLevels + zi) * 3;
          linePoints[outOff] = x[0];
          linePoints[outOff + 1] = x[1];
          linePoints[outOff + 2] = x[2];

          if (Number.isFinite(x[0]) && Number.isFinite(x[1]) && Number.isFinite(x[2])) {
            let sse = 0.0;
            let cnt = 0;
            for (let k = 0; k < candPoints.length; k++) {
              const p = candPoints[k];
              const d = candDirs[k];
              const dn = Math.hypot(d[0], d[1], d[2]);
              if (dn <= EPS) continue;
              const ux = d[0] / dn;
              const uy = d[1] / dn;
              const uz = d[2] / dn;
              const vx = x[0] - p[0];
              const vy = x[1] - p[1];
              const vz = x[2] - p[2];
              const dot = vx * ux + vy * uy + vz * uz;
              const px = vx - dot * ux;
              const py = vy - dot * uy;
              const pz = vz - dot * uz;
              sse += px * px + py * py + pz * pz;
              cnt += 1;
            }
            if (cnt > 0) fitRms[ws * nLevels + zi] = Math.sqrt(sse / cnt);
          }
        }
      }

      const distances = new Float32Array(nHalf * nLevels);
      distances.fill(NaN);
      for (let i = 0; i < nHalf; i++) {
        const nxt = (i + 1) % nHalf;
        for (let zi = 0; zi < nLevels; zi++) {
          const aOff = (i * nLevels + zi) * 3;
          const bOff = (nxt * nLevels + zi) * 3;
          const ax = linePoints[aOff], ay = linePoints[aOff + 1], az = linePoints[aOff + 2];
          const bx = linePoints[bOff], by = linePoints[bOff + 1], bz = linePoints[bOff + 2];
          if (
            Number.isFinite(ax) && Number.isFinite(ay) && Number.isFinite(az) &&
            Number.isFinite(bx) && Number.isFinite(by) && Number.isFinite(bz)
          ) {
            distances[i * nLevels + zi] = Math.hypot(bx - ax, by - ay, bz - az);
          }
        }
      }
      return { distances, fitRms, linePoints, source };
    }

    function idealHalfPointsForZGrid(m, zLevels) {
      const halves = m.half_contours;
      const nHalf = halves.length;
      const nLevels = zLevels.length;
      const firstZ = nLevels > 0 ? Number(zLevels[0]) : NaN;
      const lastZ = nLevels > 0 ? Number(zLevels[nLevels - 1]) : NaN;
      const zStep = nLevels > 1 ? Number(zLevels[1]) - firstZ : 0.0;
      const cacheKey = [nLevels, firstZ.toPrecision(14), lastZ.toPrecision(14), zStep.toPrecision(14)].join('|');
      if (m.idealHalfPointsCache && m.idealHalfPointsCache.key === cacheKey) {
        return m.idealHalfPointsCache.halfPoints;
      }

      const halfPoints = new Float32Array(nHalf * nLevels * 3);
      halfPoints.fill(NaN);
      const center = Array.isArray(m.model_center) ? m.model_center.map(Number) : [0, 0, 0];

      for (let zi = 0; zi < nLevels; zi++) {
        const zWorld = Number(zLevels[zi]);
        const zCentered = zWorld - center[2];
        const section = sectionPointsFromEdges(m, zCentered);
        if (section.length === 0) continue;

        for (let hi = 0; hi < nHalf; hi++) {
          const half = halves[hi];
          const normal = half.normal_calc || half.normal;
          if (Math.abs(Number(normal[2]) || 0.0) > 1e-8) {
            throw new Error('InitialModel ideal metric currently requires horizontal projection normals');
          }
          let lx = -Number(normal[1]);
          let ly = Number(normal[0]);
          const ln = Math.hypot(lx, ly);
          if (ln <= EPS) continue;
          lx /= ln;
          ly /= ln;

          let best = null;
          let bestSupport = Number(half.half_id) === 0 ? Infinity : -Infinity;
          for (const p of section) {
            const support = p[0] * lx + p[1] * ly;
            if (
              (Number(half.half_id) === 0 && support < bestSupport) ||
              (Number(half.half_id) !== 0 && support > bestSupport)
            ) {
              bestSupport = support;
              best = p;
            }
          }
          if (!best) continue;
          const off = (hi * nLevels + zi) * 3;
          halfPoints[off] = best[0] + center[0];
          halfPoints[off + 1] = best[1] + center[1];
          halfPoints[off + 2] = zWorld;
        }
      }

      m.idealHalfPointsCache = { key: cacheKey, halfPoints };
      return halfPoints;
    }

    function computeIdealFunctions(m, zLevels, window) {
      const halves = m.half_contours;
      const nHalf = halves.length;
      const nLevels = zLevels.length;
      const halfPoints = idealHalfPointsForZGrid(m, zLevels);

      const out = computeFunctionsFromHalfPoints(
        halfPoints,
        halves,
        nLevels,
        window,
        'InitialModel ideal section support',
      );
      return {
        zLevels,
        nLevels,
        nWindows: nHalf,
        halfPoints,
        ...out,
      };
    }

    function idealBoundaryTrace(m, halfIndex) {
      const computed = state.idealComputed;
      if (!computed || !computed.halfPoints) {
        return contourLineTrace([], '#00a6a6', 6);
      }
      const half = m.half_contours[halfIndex];
      const normal = half.normal_calc || half.normal;
      const center = Array.isArray(m.model_center) ? m.model_center.map(Number) : [0, 0, 0];
      const observedPoints = half.points_calc || [];
      let observedPlaneSupport = 0.0;
      let observedPlaneCount = 0;
      for (const p of observedPoints) {
        if (!finite3(p)) continue;
        observedPlaneSupport +=
          (Number(p[0]) - center[0]) * normal[0] +
          (Number(p[1]) - center[1]) * normal[1] +
          (Number(p[2]) - center[2]) * normal[2];
        observedPlaneCount += 1;
      }
      if (observedPlaneCount > 0) observedPlaneSupport /= observedPlaneCount;

      let supportMax = -Infinity;
      for (const p of m.vertices) {
        const s = p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2];
        if (s > supportMax) supportMax = s;
      }
      const push = supportMax + Number(CACHE.distance_scale || 0.18) * Number(m.model_radius || 1);
      const sign = Number(half.half_id) === 1 ? -1 : 1;
      const points = [];
      for (let zi = 0; zi < computed.nLevels; zi++) {
        const off = (halfIndex * computed.nLevels + zi) * 3;
        const px = computed.halfPoints[off];
        const py = computed.halfPoints[off + 1];
        const pz = computed.halfPoints[off + 2];
        if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;
        const bx = px - center[0];
        const by = py - center[1];
        const bz = pz - center[2];
        const depth = bx * normal[0] + by * normal[1] + bz * normal[2];
        const projectToObservedPlane = observedPlaneSupport - depth;
        const normalShift = projectToObservedPlane + sign * push;
        points.push([
          bx + normal[0] * normalShift,
          by + normal[1] * normalShift,
          bz + normal[2] * normalShift,
        ]);
      }
      return contourLineTrace(points, '#00a6a6', 6);
    }

    function clearMetricCaches() {
      state.hoveredDistIdx = null;
      state.rmsMinimaCacheKey = null;
      state.rmsMinimaCache = null;
      state.idealRmsMinimaCacheKey = null;
      state.idealRmsMinimaCache = null;
      state.allMinimaCloudCacheKey = null;
      state.allMinimaCloudCache = null;
      state.idealAllMinimaCloudCacheKey = null;
      state.idealAllMinimaCloudCache = null;
    }

    function recomputeDynamic() {
      const m = ensureModelDecoded(state.modelName);
      const halves = m.half_contours;
      const nHalf = halves.length;
      const zStep = Math.max(0.0005, Number(state.zStep) || 0.01);
      const window = Math.max(2, Math.min(nHalf, Math.round(Number(state.window) || 2)));
      state.zStep = zStep;
      state.window = window;
      zStepInput.value = String(zStep);
      windowInput.value = String(window);

      const pre = precomputedFunction(state.modelName, window);
      if (pre) {
        state.zStep = pre.zStep;
        zStepInput.value = String(pre.zStep);
        state.computed = {
          zLevels: pre.zLevels,
          nLevels: pre.nLevels,
          nWindows: pre.nHalf,
          distances: pre.distances,
          fitRms: pre.fitRms,
          linePoints: pre.linePoints || null,
          source: 'precomputed',
        };
        state.idealComputed = computeIdealFunctions(m, pre.zLevels, window);
        clearMetricCaches();
        state.zIdx = Math.max(0, Math.min(state.zIdx, pre.nLevels - 1));
        projIdx.max = String(Math.max(0, nHalf - 1));
        return;
      }

      const nLevels = Math.max(2, Math.floor((m.z_max - m.z_min) / zStep) + 1);
      const zLevels = new Float32Array(nLevels);
      for (let i = 0; i < nLevels; i++) zLevels[i] = m.z_min + i * zStep;

      const halfPoints = new Float32Array(nHalf * nLevels * 3);
      for (let hi = 0; hi < nHalf; hi++) {
        const hp = halves[hi].points_calc || halves[hi].points;
        for (let zi = 0; zi < nLevels; zi++) {
          const p = halfContourPointAtZ(hp, zLevels[zi]);
          const off = (hi * nLevels + zi) * 3;
          halfPoints[off] = p[0];
          halfPoints[off + 1] = p[1];
          halfPoints[off + 2] = p[2];
        }
      }

      const observed = computeFunctionsFromHalfPoints(
        halfPoints,
        halves,
        nLevels,
        window,
        'observed contours',
      );
      state.computed = {
        zLevels,
        nLevels,
        nWindows: nHalf,
        ...observed,
      };
      state.idealComputed = computeIdealFunctions(m, zLevels, window);
      clearMetricCaches();
      state.zIdx = Math.max(0, Math.min(state.zIdx, nLevels - 1));
      projIdx.max = String(Math.max(0, nHalf - 1));
    }

    function zSliderBounds() {
      const nLevels = state.computed.nLevels;
      const lo = Math.max(0, Number(state.trimBottom));
      const hi = Math.max(lo, nLevels - 1 - Math.max(0, Number(state.trimTop)));
      return { lo, hi };
    }

    function updateZSliderBounds() {
      const b = zSliderBounds();
      if (state.zIdx < b.lo) state.zIdx = b.lo;
      if (state.zIdx > b.hi) state.zIdx = b.hi;
      zIdx.min = String(b.lo);
      zIdx.max = String(b.hi);
      zIdx.value = String(state.zIdx);
    }

    function renderTopPlot() {
      const m = ensureModelDecoded(state.modelName);
      const contourIndex = Math.max(0, Math.min(m.half_contours.length - 1, state.contourIdx));
      const c = m.half_contours[contourIndex];
      if (!state.topCamera) {
        state.topCamera = { eye: contourCamera(c.normal), projection: { type: state.projection } };
      } else {
        state.topCamera.projection = { type: state.projection };
      }
      const data = [meshTrace(m), edgeTrace(m), cutLineTrace(m)];
      if (state.showRedSection) data.push(contourLineTrace(c.points, '#d62828', 8));
      if (state.showIdealSection) data.push(idealBoundaryTrace(m, contourIndex));
      Plotly.react('topPlot', data, {
        margin: {l: 0, r: 0, b: 0, t: 44},
        title: `${m.name} | observed red · InitialModel ideal cyan | seq=${c.seq_index} | merged-cont${String(c.source_index).padStart(3, '0')} half=${c.half_id}`,
        scene: {
          xaxis: {title: 'X'},
          yaxis: {title: 'Y'},
          zaxis: {title: 'Z'},
          camera: {
            eye: scaledEye(state.topCamera.eye, state.zoomAbs),
            projection: { type: state.projection },
          },
          uirevision: 'top-camera-lock',
          aspectmode: 'data',
        },
      }, {responsive: true, displaylogo: false});
    }

    function extractYAtZFrom(computed, zIndex, keyName) {
      const y = new Array(computed.nWindows);
      const src = computed[keyName];
      for (let i = 0; i < computed.nWindows; i++) {
        y[i] = src ? src[i * computed.nLevels + zIndex] : NaN;
      }
      return y;
    }

    function extractYAtZ(zIndex, keyName) {
      return extractYAtZFrom(state.computed, zIndex, keyName);
    }

    function finite3(p) {
      return (
        Array.isArray(p) &&
        p.length >= 3 &&
        Number.isFinite(p[0]) &&
        Number.isFinite(p[1]) &&
        Number.isFinite(p[2])
      );
    }

    function linePointsAtZ(zIndex) {
      const m = ensureModelDecoded(state.modelName);
      const nHalf = state.computed.nWindows;
      const zSafe = Math.max(0, Math.min(state.computed.nLevels - 1, Number(zIndex) || 0));
      const cx = Array.isArray(m.model_center) ? Number(m.model_center[0] || 0) : 0;
      const cy = Array.isArray(m.model_center) ? Number(m.model_center[1] || 0) : 0;
      const cz = Array.isArray(m.model_center) ? Number(m.model_center[2] || 0) : 0;
      const src = state.computed.linePoints;
      if (src) {
        const out = new Array(nHalf);
        for (let i = 0; i < nHalf; i++) {
          const off = (i * state.computed.nLevels + zSafe) * 3;
          out[i] = [src[off] - cx, src[off + 1] - cy, src[off + 2] - cz];
        }
        return out;
      }

      const halves = m.half_contours;
      const window = Math.max(2, Math.min(nHalf, Math.round(Number(state.window) || 2)));
      const zLevel = Number(state.computed.zLevels[zSafe]);
      const halfAtZ = new Array(nHalf);
      for (let hi = 0; hi < nHalf; hi++) {
        const hp = halves[hi].points_calc || halves[hi].points;
        halfAtZ[hi] = halfContourPointAtZ(hp, zLevel);
      }

      const out = new Array(nHalf);
      for (let ws = 0; ws < nHalf; ws++) {
        const candPoints = [];
        const candDirs = [];
        for (let j = 0; j < window; j++) {
          const idx = (ws + j) % nHalf;
          const p = halfAtZ[idx];
          if (finite3(p)) {
            candPoints.push(p);
            candDirs.push(halves[idx].normal_calc || halves[idx].normal);
          }
        }
        if (candPoints.length >= 2) {
          const p = closestPointToLines(candPoints, candDirs);
          out[ws] = [p[0] - cx, p[1] - cy, p[2] - cz];
        } else {
          out[ws] = [NaN, NaN, NaN];
        }
      }
      return out;
    }

    function computeRmsMinimaSelection() {
      const empty = {
        indices: [],
        values: [],
        points: [],
        globalMin: NaN,
        globalMax: NaN,
        amplitude: NaN,
        threshold: NaN,
      };
      if (!state.computed || !state.computed.fitRms) return empty;

      const yFit = extractYAtZ(state.zIdx, 'fitRms');
      const n = yFit.length;
      if (n < 3) return empty;

      let globalMin = Infinity;
      let globalMax = -Infinity;
      for (let i = 0; i < n; i++) {
        const v = yFit[i];
        if (!Number.isFinite(v)) continue;
        if (v < globalMin) globalMin = v;
        if (v > globalMax) globalMax = v;
      }
      if (!Number.isFinite(globalMin) || !Number.isFinite(globalMax)) return empty;

      const amplitude = Math.max(0, globalMax - globalMin);
      const pct = Math.max(0, Number(state.rmsMinPct) || 0);
      const threshold = globalMin + amplitude * (pct / 100.0);

      const minimaIdx = [];
      const minimaVal = [];
      for (let i = 0; i < n; i++) {
        const cur = yFit[i];
        if (!Number.isFinite(cur)) continue;
        const prev = yFit[(i - 1 + n) % n];
        const next = yFit[(i + 1) % n];
        if (!Number.isFinite(prev) || !Number.isFinite(next)) continue;
        const isLocal = cur <= prev && cur <= next && (cur < prev || cur < next);
        if (!isLocal) continue;
        if (cur <= threshold + 1e-12) {
          minimaIdx.push(i);
          minimaVal.push(cur);
        }
      }

      if (minimaIdx.length === 0) {
        return { ...empty, globalMin, globalMax, amplitude, threshold };
      }

      const allPoints = linePointsAtZ(state.zIdx);
      const minimaPoints = [];
      const outIdx = [];
      const outVal = [];
      for (let k = 0; k < minimaIdx.length; k++) {
        const i = minimaIdx[k];
        const p = allPoints[i];
        if (!finite3(p)) continue;
        outIdx.push(i);
        outVal.push(minimaVal[k]);
        minimaPoints.push([p[0], p[1], p[2]]);
      }

      return {
        indices: outIdx,
        values: outVal,
        points: minimaPoints,
        globalMin,
        globalMax,
        amplitude,
        threshold,
      };
    }

    function currentRmsMinimaSelection() {
      if (!state.computed) {
        state.rmsMinimaCacheKey = null;
        state.rmsMinimaCache = null;
        return computeRmsMinimaSelection();
      }
      const key = [
        state.modelName,
        state.computed.source || 'dynamic',
        state.window,
        state.zIdx,
        Number(state.rmsMinPct).toFixed(4),
        state.computed.nLevels,
        state.computed.nWindows,
      ].join('|');
      if (state.rmsMinimaCacheKey === key && state.rmsMinimaCache) return state.rmsMinimaCache;
      state.rmsMinimaCacheKey = key;
      state.rmsMinimaCache = computeRmsMinimaSelection();
      return state.rmsMinimaCache;
    }

    function computeIdealRmsMinimaSelection() {
      const empty = {
        indices: [],
        values: [],
        points: [],
        globalMin: NaN,
        globalMax: NaN,
        amplitude: NaN,
        threshold: NaN,
      };
      const computed = state.idealComputed;
      if (!computed || !computed.fitRms || !computed.linePoints) return empty;

      const yFit = extractYAtZFrom(computed, state.zIdx, 'fitRms');
      const n = yFit.length;
      if (n < 3) return empty;
      let globalMin = Infinity;
      let globalMax = -Infinity;
      for (const value of yFit) {
        if (!Number.isFinite(value)) continue;
        if (value < globalMin) globalMin = value;
        if (value > globalMax) globalMax = value;
      }
      if (!Number.isFinite(globalMin) || !Number.isFinite(globalMax)) return empty;
      const amplitude = Math.max(0, globalMax - globalMin);
      const threshold = globalMin + amplitude * (Math.max(0, Number(state.rmsMinPct) || 0) / 100.0);
      const m = ensureModelDecoded(state.modelName);
      const center = Array.isArray(m.model_center) ? m.model_center.map(Number) : [0, 0, 0];
      const indices = [];
      const values = [];
      const points = [];
      for (let i = 0; i < n; i++) {
        const cur = yFit[i];
        const prev = yFit[(i - 1 + n) % n];
        const next = yFit[(i + 1) % n];
        if (!Number.isFinite(cur) || !Number.isFinite(prev) || !Number.isFinite(next)) continue;
        if (!(cur <= prev && cur <= next && (cur < prev || cur < next))) continue;
        if (cur > threshold + 1e-12) continue;
        const off = (i * computed.nLevels + state.zIdx) * 3;
        const p = [
          computed.linePoints[off] - center[0],
          computed.linePoints[off + 1] - center[1],
          computed.linePoints[off + 2] - center[2],
        ];
        if (!finite3(p)) continue;
        indices.push(i);
        values.push(cur);
        points.push(p);
      }
      return { indices, values, points, globalMin, globalMax, amplitude, threshold };
    }

    function currentIdealRmsMinimaSelection() {
      const computed = state.idealComputed;
      if (!computed) return computeIdealRmsMinimaSelection();
      const key = [
        state.modelName,
        computed.source || 'ideal',
        state.window,
        state.zIdx,
        Number(state.rmsMinPct).toFixed(4),
        computed.nLevels,
        computed.nWindows,
      ].join('|');
      if (state.idealRmsMinimaCacheKey === key && state.idealRmsMinimaCache) {
        return state.idealRmsMinimaCache;
      }
      state.idealRmsMinimaCacheKey = key;
      state.idealRmsMinimaCache = computeIdealRmsMinimaSelection();
      return state.idealRmsMinimaCache;
    }

    function computeAllRmsMinimaCloud() {
      const empty = { points: [], text: [] };
      if (!state.computed || !state.computed.fitRms) return empty;
      const m = ensureModelDecoded(state.modelName);
      const fit = state.computed.fitRms;
      const lp = state.computed.linePoints || null;
      const nLevels = state.computed.nLevels;
      const nHalf = state.computed.nWindows;
      const pct = Math.max(0, Number(state.rmsMinPct) || 0);
      const cx = Array.isArray(m.model_center) ? Number(m.model_center[0] || 0) : 0;
      const cy = Array.isArray(m.model_center) ? Number(m.model_center[1] || 0) : 0;
      const cz = Array.isArray(m.model_center) ? Number(m.model_center[2] || 0) : 0;

      const points = [];
      const text = [];
      let fallbackZ = null;
      for (let zi = 0; zi < nLevels; zi++) {
        let gMin = Infinity;
        let gMax = -Infinity;
        const y = new Array(nHalf);
        for (let i = 0; i < nHalf; i++) {
          const v = fit[i * nLevels + zi];
          y[i] = v;
          if (!Number.isFinite(v)) continue;
          if (v < gMin) gMin = v;
          if (v > gMax) gMax = v;
        }
        if (!Number.isFinite(gMin) || !Number.isFinite(gMax)) continue;
        const threshold = gMin + Math.max(0, gMax - gMin) * (pct / 100.0);

        if (!lp) fallbackZ = linePointsAtZ(zi);
        for (let i = 0; i < nHalf; i++) {
          const cur = y[i];
          if (!Number.isFinite(cur)) continue;
          const prev = y[(i - 1 + nHalf) % nHalf];
          const next = y[(i + 1) % nHalf];
          if (!Number.isFinite(prev) || !Number.isFinite(next)) continue;
          const isLocal = cur <= prev && cur <= next && (cur < prev || cur < next);
          if (!isLocal || cur > threshold + 1e-12) continue;

          let p;
          if (lp) {
            const off = (i * nLevels + zi) * 3;
            p = [lp[off] - cx, lp[off + 1] - cy, lp[off + 2] - cz];
          } else {
            p = fallbackZ ? fallbackZ[i] : null;
          }
          if (!finite3(p)) continue;
          points.push(p);
          text.push(`z_idx=${zi}, i=${i}, rms=${Number(cur).toFixed(6)}`);
        }
      }
      return { points, text };
    }

    function currentAllRmsMinimaCloud() {
      if (!state.computed) {
        state.allMinimaCloudCacheKey = null;
        state.allMinimaCloudCache = null;
        return computeAllRmsMinimaCloud();
      }
      const key = [
        state.modelName,
        state.computed.source || 'dynamic',
        state.window,
        Number(state.rmsMinPct).toFixed(4),
        state.computed.nLevels,
        state.computed.nWindows,
      ].join('|');
      if (state.allMinimaCloudCacheKey === key && state.allMinimaCloudCache) return state.allMinimaCloudCache;
      state.allMinimaCloudCacheKey = key;
      state.allMinimaCloudCache = computeAllRmsMinimaCloud();
      return state.allMinimaCloudCache;
    }

    function computeAllIdealRmsMinimaCloud() {
      const empty = { points: [], text: [] };
      const computed = state.idealComputed;
      if (!computed || !computed.fitRms || !computed.linePoints) return empty;
      const m = ensureModelDecoded(state.modelName);
      const center = Array.isArray(m.model_center) ? m.model_center.map(Number) : [0, 0, 0];
      const fit = computed.fitRms;
      const lp = computed.linePoints;
      const nLevels = computed.nLevels;
      const nHalf = computed.nWindows;
      const pct = Math.max(0, Number(state.rmsMinPct) || 0);
      const points = [];
      const text = [];

      for (let zi = 0; zi < nLevels; zi++) {
        let gMin = Infinity;
        let gMax = -Infinity;
        const y = new Array(nHalf);
        for (let i = 0; i < nHalf; i++) {
          const value = fit[i * nLevels + zi];
          y[i] = value;
          if (!Number.isFinite(value)) continue;
          if (value < gMin) gMin = value;
          if (value > gMax) gMax = value;
        }
        if (!Number.isFinite(gMin) || !Number.isFinite(gMax)) continue;
        const threshold = gMin + Math.max(0, gMax - gMin) * (pct / 100.0);
        for (let i = 0; i < nHalf; i++) {
          const cur = y[i];
          const prev = y[(i - 1 + nHalf) % nHalf];
          const next = y[(i + 1) % nHalf];
          if (!Number.isFinite(cur) || !Number.isFinite(prev) || !Number.isFinite(next)) continue;
          if (!(cur <= prev && cur <= next && (cur < prev || cur < next))) continue;
          if (cur > threshold + 1e-12) continue;
          const off = (i * nLevels + zi) * 3;
          const p = [
            lp[off] - center[0],
            lp[off + 1] - center[1],
            lp[off + 2] - center[2],
          ];
          if (!finite3(p)) continue;
          points.push(p);
          text.push('ideal z_idx=' + zi + ', i=' + i + ', rms=' + Number(cur).toFixed(6));
        }
      }
      return { points, text };
    }

    function currentAllIdealRmsMinimaCloud() {
      const computed = state.idealComputed;
      if (!computed) return computeAllIdealRmsMinimaCloud();
      const key = [
        state.modelName,
        computed.source || 'ideal',
        state.window,
        Number(state.rmsMinPct).toFixed(4),
        computed.nLevels,
        computed.nWindows,
      ].join('|');
      if (state.idealAllMinimaCloudCacheKey === key && state.idealAllMinimaCloudCache) {
        return state.idealAllMinimaCloudCache;
      }
      state.idealAllMinimaCloudCacheKey = key;
      state.idealAllMinimaCloudCache = computeAllIdealRmsMinimaCloud();
      return state.idealAllMinimaCloudCache;
    }

    function minimaPointsTrace(
      points,
      indices,
      values,
      color = '#ffd166',
      outline = '#8a5a00',
      label = 'RMS minima',
    ) {
      return {
        type: 'scatter3d',
        mode: 'markers',
        x: points.map(p => p[0]),
        y: points.map(p => p[1]),
        z: points.map(p => p[2]),
        marker: { size: 5, color, line: { color: outline, width: 1.2 } },
        text: indices.map((i, k) => `i=${i}, rms=${Number(values[k]).toFixed(6)}`),
        hovertemplate: '%{text}<extra>' + label + '</extra>',
        showlegend: false,
      };
    }

    function minimaCloudTrace(points, text, color = '#1d4ed8', label = 'All RMS minima') {
      return {
        type: 'scatter3d',
        mode: 'markers',
        x: points.map(p => p[0]),
        y: points.map(p => p[1]),
        z: points.map(p => p[2]),
        marker: { size: 2.7, color, opacity: 0.95 },
        text,
        hovertemplate: '%{text}<extra>' + label + '</extra>',
        showlegend: false,
      };
    }

    function activeHalfIndices() {
      const n = state.computed.nWindows;
      const maxDi = Math.max(0, n - 1);
      const di = state.hoveredDistIdx == null ? 0 : Math.max(0, Math.min(maxDi, state.hoveredDistIdx));
      const out = [];
      for (let j = 0; j <= Number(state.window); j++) out.push((di + j) % n);
      return { di, out };
    }

    function renderContextPlot() {
      const m = ensureModelDecoded(state.modelName);
      const zVal = currentZValue();
      if (!state.contextCamera) {
        state.contextCamera = { eye: {x: 1.25, y: 1.0, z: 0.8}, projection: { type: state.projection } };
      } else {
        state.contextCamera.projection = { type: state.projection };
      }
      const active = activeHalfIndices();
      const sectionObj = sectionTrace(m, zVal);
      const minima = currentRmsMinimaSelection();
      const idealMinima = currentIdealRmsMinimaSelection();
      const data = [meshTrace(m), edgeTrace(m), cutLineTrace(m), zPlaneTrace(m, zVal)];
      if (state.showRedSection) {
        for (const ci of active.out) {
          data.push(contourLineTrace(m.half_contours[ci].points, '#ef476f', 5));
        }
      }
      if (state.showIdealSection) {
        data.push(sectionObj.line);
        for (const ci of active.out) data.push(idealBoundaryTrace(m, ci));
      }
      if (state.showYellowSection) {
        if (minima.points.length > 0) data.push(minimaPointsTrace(minima.points, minima.indices, minima.values));
        if (idealMinima.points.length > 0) {
          data.push(minimaPointsTrace(
            idealMinima.points,
            idealMinima.indices,
            idealMinima.values,
            '#00d4d4',
            '#006666',
            'InitialModel ideal minima',
          ));
        }
      }
      const annotations = [];
      if (state.showRedSection) {
        annotations.push(...oppositeSideAnnotation(
          m,
          active.out,
          sectionObj.nGon,
          '',
          '#ff2d55',
          0.90,
        ));
      }
      if (state.showIdealSection) {
        annotations.push(...oppositeSideAnnotation(
          m,
          active.out,
          sectionObj.nGon,
          'ideal ',
          '#00a6a6',
          0.82,
        ));
      }

      Plotly.react('contextPlot', data, {
        margin: {l: 0, r: 0, b: 0, t: 44},
        title: `Красные observed · бирюзовые ideal | i=${active.di} (window=${state.window})`,
        annotations,
        scene: {
          xaxis: {title: 'X'},
          yaxis: {title: 'Y'},
          zaxis: {title: 'Z'},
          camera: {
            eye: scaledEye(state.contextCamera.eye, state.zoomAbs),
            projection: { type: state.projection },
          },
          uirevision: 'context-camera-lock',
          aspectmode: 'data',
        },
      }, {responsive: true, displaylogo: false});
    }

    function renderAllMinimaPlot() {
      const m = ensureModelDecoded(state.modelName);
      if (!state.allMinimaCameraLeft) {
        state.allMinimaCameraLeft = { eye: {x: 1.25, y: 1.0, z: 0.8}, projection: { type: state.projection } };
      } else {
        state.allMinimaCameraLeft.projection = { type: state.projection };
      }
      if (!state.allMinimaCameraRight) {
        state.allMinimaCameraRight = { eye: {x: 1.25, y: 1.0, z: 0.8}, projection: { type: state.projection } };
      } else {
        state.allMinimaCameraRight.projection = { type: state.projection };
      }
      if (!state.allMinimaCameraIdeal) {
        state.allMinimaCameraIdeal = { eye: {x: 1.25, y: 1.0, z: 0.8}, projection: { type: state.projection } };
      } else {
        state.allMinimaCameraIdeal.projection = { type: state.projection };
      }

      const cloud = currentAllRmsMinimaCloud();
      const idealCloud = currentAllIdealRmsMinimaCloud();
      const data = [
        { ...meshTrace(m), scene: 'scene' },
        { ...edgeTrace(m), scene: 'scene' },
        {
          ...minimaCloudTrace(cloud.points, cloud.text, '#1d4ed8', 'Observed minima'),
          scene: 'scene2',
        },
        {
          ...minimaCloudTrace(idealCloud.points, idealCloud.text, '#00a6a6', 'InitialModel ideal minima'),
          scene: 'scene3',
        },
      ];

      Plotly.react('allMinimaPlot', data, {
        margin: {l: 0, r: 0, b: 0, t: 44},
        title: `Все Z: InitialModel | observed minima (${cloud.points.length}) | ideal minima (${idealCloud.points.length})`,
        scene: {
          domain: { x: [0.0, 0.32], y: [0.0, 1.0] },
          xaxis: {title: 'X'},
          yaxis: {title: 'Y'},
          zaxis: {title: 'Z'},
          camera: {
            eye: scaledEye(state.allMinimaCameraLeft.eye, state.zoomAbs),
            projection: { type: state.projection },
          },
          uirevision: 'all-minima-camera-left-lock',
          aspectmode: 'data',
        },
        scene2: {
          domain: { x: [0.34, 0.66], y: [0.0, 1.0] },
          xaxis: {title: 'X'},
          yaxis: {title: 'Y'},
          zaxis: {title: 'Z'},
          camera: {
            eye: scaledEye(state.allMinimaCameraRight.eye, state.zoomAbs),
            projection: { type: state.projection },
          },
          uirevision: 'all-minima-camera-right-lock',
          aspectmode: 'data',
        },
        scene3: {
          domain: { x: [0.68, 1.0], y: [0.0, 1.0] },
          xaxis: {title: 'X'},
          yaxis: {title: 'Y'},
          zaxis: {title: 'Z'},
          camera: {
            eye: scaledEye(state.allMinimaCameraIdeal.eye, state.zoomAbs),
            projection: { type: state.projection },
          },
          uirevision: 'all-minima-camera-ideal-lock',
          aspectmode: 'data',
        },
      }, {responsive: true, displaylogo: false});
    }

    function renderAllMinimaOverlayPlot() {
      const m = ensureModelDecoded(state.modelName);
      if (!state.allMinimaOverlayCamera) {
        state.allMinimaOverlayCamera = {
          eye: {x: 1.25, y: 1.0, z: 0.8},
          projection: { type: state.projection },
        };
      } else {
        state.allMinimaOverlayCamera.projection = { type: state.projection };
      }

      const cloud = currentAllRmsMinimaCloud();
      const idealCloud = currentAllIdealRmsMinimaCloud();
      const data = [
        { ...meshTrace(m), scene: 'scene', visible: state.showAllZModel },
        { ...edgeTrace(m), scene: 'scene', visible: state.showAllZModel },
        {
          ...minimaCloudTrace(cloud.points, cloud.text, '#1d4ed8', 'Observed minima'),
          scene: 'scene',
          visible: state.showAllZObservedMinima,
        },
        {
          ...minimaCloudTrace(idealCloud.points, idealCloud.text, '#00a6a6', 'InitialModel ideal minima'),
          scene: 'scene',
          visible: state.showAllZIdealMinima,
        },
      ];

      Plotly.react('allMinimaOverlayPlot', data, {
        margin: {l: 0, r: 0, b: 0, t: 44},
        title: `Все Z в одной сцене: InitialModel + observed minima (${cloud.points.length}) + ideal minima (${idealCloud.points.length})`,
        scene: {
          xaxis: {title: 'X'},
          yaxis: {title: 'Y'},
          zaxis: {title: 'Z'},
          camera: {
            eye: scaledEye(state.allMinimaOverlayCamera.eye, state.zoomAbs),
            projection: { type: state.projection },
          },
          uirevision: `all-minima-overlay-camera-${state.modelName}`,
          aspectmode: 'data',
        },
      }, {responsive: true, displaylogo: false});
    }

    function updateAllMinimaOverlayVisibility() {
      Plotly.restyle('allMinimaOverlayPlot', {visible: state.showAllZModel}, [0, 1]);
      Plotly.restyle('allMinimaOverlayPlot', {visible: state.showAllZObservedMinima}, [2]);
      Plotly.restyle('allMinimaOverlayPlot', {visible: state.showAllZIdealMinima}, [3]);
    }

    function finiteUpperRange(a, b) {
      let maxValue = 0.0;
      for (const values of [a, b]) {
        for (const value of values || []) {
          if (Number.isFinite(value) && value > maxValue) maxValue = value;
        }
      }
      return [0.0, Math.max(1e-9, maxValue * 1.04)];
    }

    function currentSharedRanges() {
      const observedDistance = extractYAtZFrom(state.computed, state.zIdx, 'distances');
      const idealDistance = extractYAtZFrom(state.idealComputed, state.zIdx, 'distances');
      const observedRms = extractYAtZFrom(state.computed, state.zIdx, 'fitRms');
      const idealRms = extractYAtZFrom(state.idealComputed, state.zIdx, 'fitRms');
      return {
        distance: finiteUpperRange(observedDistance, idealDistance),
        rms: finiteUpperRange(observedRms, idealRms),
      };
    }

    function updateMetricStatus() {
      const zVal = state.computed.zLevels[state.zIdx] ?? NaN;
      const n = state.computed.nWindows;
      const i = Math.max(0, Math.min(n - 1, state.hoveredDistIdx == null ? 0 : state.hoveredDistIdx));
      const observed = state.computed.fitRms[i * state.computed.nLevels + state.zIdx];
      const ideal = state.idealComputed.fitRms[i * state.idealComputed.nLevels + state.zIdx];
      const delta = Number.isFinite(observed) && Number.isFinite(ideal) ? observed - ideal : NaN;
      status.textContent =
        `z=${Number(zVal).toFixed(6)} | window=${state.window} | i=${i} | ` +
        `observed RMS=${Number.isFinite(observed) ? Number(observed).toFixed(6) : '—'} | ` +
        `ideal RMS=${Number.isFinite(ideal) ? Number(ideal).toFixed(6) : '—'} | ` +
        `difference=${Number.isFinite(delta) ? Number(delta).toFixed(6) : '—'}`;
    }

    function bindFnPlotMouseTracking(plotId, nPoints) {
      const gd = document.getElementById(plotId);
      gd.__nPoints = nPoints;
      if (gd.__mouseTrackingBound) return;
      gd.__mouseTrackingBound = true;

      gd.addEventListener('mousemove', (ev) => {
        if (!gd.__nPoints || gd.__nPoints < 1) return;
        const fl = gd._fullLayout;
        if (!fl || !fl._size) return;
        const rect = gd.getBoundingClientRect();
        const size = fl._size;
        const xInPlot = ev.clientX - rect.left - size.l;
        if (xInPlot < 0 || xInPlot > size.w) return;
        const frac = xInPlot / Math.max(1, size.w);
        const idx = Math.round(frac * (gd.__nPoints - 1));
        const clamped = Math.max(0, Math.min(gd.__nPoints - 1, idx));
        if (clamped === state.hoveredDistIdx) return;
        state.hoveredDistIdx = clamped;
        updateMetricStatus();
        renderContextPlot();
      });
    }

    function renderFnPlot() {
      updateZSliderBounds();
      const yDist = extractYAtZ(state.zIdx, 'distances');
      const yFit = extractYAtZ(state.zIdx, 'fitRms');
      const x = Array.from({length: yDist.length}, (_, i) => i);
      const minima = currentRmsMinimaSelection();
      const ranges = currentSharedRanges();
      updateMetricStatus();

      const traces = [
        {
          type: 'scatter',
          mode: 'lines',
          x,
          y: yDist,
          name: 'Distance',
          xaxis: 'x2',
          yaxis: 'y',
          line: { color: '#0077b6', width: 2.5 },
          connectgaps: false,
        },
      ];
      if (state.computed.fitRms) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x,
          y: yFit,
          name: 'RMS fit error',
          xaxis: 'x',
          yaxis: 'y2',
          line: { color: '#d62828', width: 2.5 },
          connectgaps: false,
        });
        if (Number.isFinite(minima.threshold) && x.length > 0) {
          traces.push({
            type: 'scatter',
            mode: 'lines',
            x: [x[0], x[x.length - 1]],
            y: [minima.threshold, minima.threshold],
            name: 'RMS minima threshold',
            xaxis: 'x',
            yaxis: 'y2',
            line: { color: '#ff9f1c', width: 1.5, dash: 'dot' },
            hoverinfo: 'skip',
          });
        }
        if (minima.indices.length > 0) {
          traces.push({
            type: 'scatter',
            mode: 'markers',
            x: minima.indices,
            y: minima.values,
            name: 'Selected local minima',
            xaxis: 'x',
            yaxis: 'y2',
            visible: state.showYellowSection ? true : 'legendonly',
            showlegend: false,
            marker: { size: 7, color: '#ffd166', line: { color: '#8a5a00', width: 1.2 } },
          });
        } else {
          traces.push({
            type: 'scatter',
            mode: 'markers',
            x: [],
            y: [],
            name: 'Selected local minima',
            xaxis: 'x',
            yaxis: 'y2',
            visible: state.showYellowSection ? true : 'legendonly',
            showlegend: false,
            marker: { size: 7, color: '#ffd166', line: { color: '#8a5a00', width: 1.2 } },
          });
        }
      }

      Plotly.react('fnPlot', traces, {
        margin: {l: 58, r: 14, b: 48, t: 44},
        title: 'Наблюдаемые теневые контуры',
        xaxis: { title: 'Window transition index i (cyclic)', anchor: 'y2' },
        xaxis2: { matches: 'x', showticklabels: false, title: '', anchor: 'y' },
        yaxis: {
          domain: [0.54, 1.0],
          title: 'Distance',
          anchor: 'x2',
          range: state.sharedYScale ? ranges.distance : undefined,
        },
        yaxis2: {
          domain: [0.0, 0.46],
          anchor: 'x',
          title: 'RMS error',
          range: state.sharedYScale ? ranges.rms : undefined,
        },
        legend: { orientation: 'h', x: 0.0, y: 1.08 },
      }, {responsive: true, displaylogo: false});

      bindFnPlotMouseTracking('fnPlot', yDist.length);
    }

    function renderIdealFnPlot() {
      const computed = state.idealComputed;
      const yDist = extractYAtZFrom(computed, state.zIdx, 'distances');
      const yFit = extractYAtZFrom(computed, state.zIdx, 'fitRms');
      const observedDist = extractYAtZFrom(state.computed, state.zIdx, 'distances');
      const observedFit = extractYAtZFrom(state.computed, state.zIdx, 'fitRms');
      const x = Array.from({length: yDist.length}, (_, i) => i);
      const minima = currentIdealRmsMinimaSelection();
      const ranges = currentSharedRanges();
      const traces = [
        {
          type: 'scatter',
          mode: 'lines',
          x,
          y: yDist,
          name: 'Ideal Distance',
          xaxis: 'x2',
          yaxis: 'y',
          line: { color: '#0077b6', width: 2.5 },
          connectgaps: false,
        },
        {
          type: 'scatter',
          mode: 'lines',
          x,
          y: yFit,
          name: 'Ideal RMS fit error',
          xaxis: 'x',
          yaxis: 'y2',
          line: { color: '#00a6a6', width: 2.5 },
          connectgaps: false,
        },
      ];
      if (state.overlayObservedOnIdeal) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x,
          y: observedDist,
          name: 'Observed Distance',
          xaxis: 'x2',
          yaxis: 'y',
          line: { color: '#d97706', width: 2.0, dash: 'dash' },
          connectgaps: false,
        });
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x,
          y: observedFit,
          name: 'Observed RMS fit error',
          xaxis: 'x',
          yaxis: 'y2',
          line: { color: '#d62828', width: 2.0, dash: 'dash' },
          connectgaps: false,
        });
      }
      if (Number.isFinite(minima.threshold) && x.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: [x[0], x[x.length - 1]],
          y: [minima.threshold, minima.threshold],
          name: 'RMS minima threshold',
          xaxis: 'x',
          yaxis: 'y2',
          line: { color: '#ff9f1c', width: 1.5, dash: 'dot' },
          hoverinfo: 'skip',
        });
      }
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: minima.indices,
        y: minima.values,
        name: 'Selected local minima',
        xaxis: 'x',
        yaxis: 'y2',
        visible: state.showYellowSection ? true : 'legendonly',
        showlegend: false,
        marker: { size: 7, color: '#00d4d4', line: { color: '#006666', width: 1.2 } },
      });

      Plotly.react('idealFnPlot', traces, {
        margin: {l: 58, r: 14, b: 48, t: 44},
        title: 'Идеальные тени InitialModel (reference-only)',
        xaxis: { title: 'Window transition index i (cyclic)', anchor: 'y2' },
        xaxis2: { matches: 'x', showticklabels: false, title: '', anchor: 'y' },
        yaxis: {
          domain: [0.54, 1.0],
          title: 'Distance',
          anchor: 'x2',
          range: state.sharedYScale ? ranges.distance : undefined,
        },
        yaxis2: {
          domain: [0.0, 0.46],
          anchor: 'x',
          title: 'RMS error',
          range: state.sharedYScale ? ranges.rms : undefined,
        },
        legend: { orientation: 'h', x: 0.0, y: 1.08 },
      }, {responsive: true, displaylogo: false});

      bindFnPlotMouseTracking('idealFnPlot', yDist.length);
    }

    function recomputeAndRender() {
      updateControlBoundsForModel();
      recomputeDynamic();
      renderFnPlot();
      renderIdealFnPlot();
      renderContextPlot();
      renderAllMinimaPlot();
      renderAllMinimaOverlayPlot();
      renderTopPlot();
    }

    modelSel.addEventListener('change', () => {
      state.modelName = modelSel.value;
      state.contourIdx = 0;
      state.topCamera = null;
      state.contextCamera = null;
      state.allMinimaCameraLeft = null;
      state.allMinimaCameraRight = null;
      state.allMinimaCameraIdeal = null;
      state.allMinimaOverlayCamera = null;
      state.zIdx = 0;
      recomputeAndRender();
    });

    projIdx.addEventListener('input', () => {
      state.contourIdx = Number(projIdx.value);
      renderTopPlot();
    });

    projMode.addEventListener('change', () => {
      state.projection = projMode.value;
      renderTopPlot();
      renderContextPlot();
      renderAllMinimaPlot();
      renderAllMinimaOverlayPlot();
    });

    zoomAbs.addEventListener('input', () => {
      state.zoomAbs = Math.max(0.40, Math.min(3.00, Number(zoomAbs.value) || 1.0));
      zoomAbsLabel.textContent = `${state.zoomAbs.toFixed(2)}x`;
      zoomAbs2.value = String(state.zoomAbs.toFixed(2));
      zoomAbsLabel2.textContent = `${state.zoomAbs.toFixed(2)}x`;
      renderTopPlot();
      renderContextPlot();
      renderAllMinimaPlot();
      renderAllMinimaOverlayPlot();
    });

    zoomAbs2.addEventListener('input', () => {
      state.zoomAbs = Math.max(0.40, Math.min(3.00, Number(zoomAbs2.value) || 1.0));
      zoomAbs.value = String(state.zoomAbs.toFixed(2));
      zoomAbsLabel.textContent = `${state.zoomAbs.toFixed(2)}x`;
      zoomAbsLabel2.textContent = `${state.zoomAbs.toFixed(2)}x`;
      renderTopPlot();
      renderContextPlot();
      renderAllMinimaPlot();
      renderAllMinimaOverlayPlot();
    });

    function setContextView(axis) {
      const d = 1.25;
      if (axis === 'x') state.contextCamera = { eye: {x: d, y: 0.0, z: 0.0}, projection: { type: state.projection } };
      if (axis === 'y') state.contextCamera = { eye: {x: 0.0, y: d, z: 0.0}, projection: { type: state.projection } };
      if (axis === 'z') state.contextCamera = { eye: {x: 0.0, y: 0.0, z: d}, projection: { type: state.projection } };
      renderContextPlot();
    }
    viewXBtn.addEventListener('click', () => setContextView('x'));
    viewYBtn.addEventListener('click', () => setContextView('y'));
    viewZBtn.addEventListener('click', () => setContextView('z'));

    zStepInput.addEventListener('change', () => {
      if (zStepInput.disabled) return;
      state.zStep = Number(zStepInput.value);
      recomputeAndRender();
    });

    windowInput.addEventListener('change', () => {
      state.window = Number(windowInput.value);
      recomputeAndRender();
    });

    rmsMinPctInput.addEventListener('change', () => {
      state.rmsMinPct = Math.max(0, Math.min(100, Number(rmsMinPctInput.value) || 0));
      rmsMinPctInput.value = String(state.rmsMinPct);
      state.rmsMinimaCacheKey = null;
      state.rmsMinimaCache = null;
      state.idealRmsMinimaCacheKey = null;
      state.idealRmsMinimaCache = null;
      state.allMinimaCloudCacheKey = null;
      state.allMinimaCloudCache = null;
      state.idealAllMinimaCloudCacheKey = null;
      state.idealAllMinimaCloudCache = null;
      renderFnPlot();
      renderIdealFnPlot();
      renderContextPlot();
      renderAllMinimaPlot();
      renderAllMinimaOverlayPlot();
    });

    showRedSectionChk.addEventListener('change', () => {
      state.showRedSection = !!showRedSectionChk.checked;
      renderTopPlot();
      renderContextPlot();
    });

    showIdealSectionChk.addEventListener('change', () => {
      state.showIdealSection = !!showIdealSectionChk.checked;
      renderTopPlot();
      renderContextPlot();
    });

    showYellowSectionChk.addEventListener('change', () => {
      state.showYellowSection = !!showYellowSectionChk.checked;
      renderFnPlot();
      renderIdealFnPlot();
      renderContextPlot();
    });

    sharedYScaleChk.addEventListener('change', () => {
      state.sharedYScale = !!sharedYScaleChk.checked;
      renderFnPlot();
      renderIdealFnPlot();
    });

    overlayObservedOnIdealChk.addEventListener('change', () => {
      state.overlayObservedOnIdeal = !!overlayObservedOnIdealChk.checked;
      renderIdealFnPlot();
    });

    showAllZModelChk.addEventListener('change', () => {
      state.showAllZModel = !!showAllZModelChk.checked;
      updateAllMinimaOverlayVisibility();
    });

    showAllZObservedMinimaChk.addEventListener('change', () => {
      state.showAllZObservedMinima = !!showAllZObservedMinimaChk.checked;
      updateAllMinimaOverlayVisibility();
    });

    showAllZIdealMinimaChk.addEventListener('change', () => {
      state.showAllZIdealMinima = !!showAllZIdealMinimaChk.checked;
      updateAllMinimaOverlayVisibility();
    });

    zIdx.addEventListener('input', () => {
      state.zIdx = Number(zIdx.value);
      renderFnPlot();
      renderIdealFnPlot();
      renderContextPlot();
    });

    trimBottom.addEventListener('change', () => {
      state.trimBottom = Math.max(0, Number(trimBottom.value) || 0);
      renderFnPlot();
      renderIdealFnPlot();
      renderContextPlot();
    });

    trimTop.addEventListener('change', () => {
      state.trimTop = Math.max(0, Number(trimTop.value) || 0);
      renderFnPlot();
      renderIdealFnPlot();
      renderContextPlot();
    });

    projMode.value = state.projection;
    zoomAbs.value = String(state.zoomAbs.toFixed(2));
    zoomAbsLabel.textContent = `${state.zoomAbs.toFixed(2)}x`;
    zoomAbs2.value = String(state.zoomAbs.toFixed(2));
    zoomAbsLabel2.textContent = `${state.zoomAbs.toFixed(2)}x`;
    rmsMinPctInput.value = String(state.rmsMinPct);
    showRedSectionChk.checked = !!state.showRedSection;
    showIdealSectionChk.checked = !!state.showIdealSection;
    showYellowSectionChk.checked = !!state.showYellowSection;
    sharedYScaleChk.checked = !!state.sharedYScale;
    overlayObservedOnIdealChk.checked = !!state.overlayObservedOnIdeal;
    showAllZModelChk.checked = !!state.showAllZModel;
    showAllZObservedMinimaChk.checked = !!state.showAllZObservedMinima;
    showAllZIdealMinimaChk.checked = !!state.showAllZIdealMinima;
    recomputeAndRender();
  </script>
</body>
</html>
""".replace("__CACHE_JSON__", cache_json).replace("__WINDOWS_CACHE_JSON__", windows_json)


def discover_models(data_root: Path, explicit_models: list[str] | None) -> list[str]:
    if explicit_models:
        return explicit_models
    out: list[str] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "InitialModel").exists() and (p / "shadow").exists():
            out.append(p.name)
    return out


def build_cache_payload(
    data_root: Path,
    models: list[str],
    max_contours: int | None,
    distance_scale: float,
    cut_x_mode: str,
    cut_x_offset: float,
    *,
    use_progress: bool,
) -> dict[str, object]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "distance_scale": float(distance_scale),
        "cut_x_mode": "model-center",
        "cut_x_offset": 0.0,
        "models": {},
    }

    model_iter = progress_iter(
        models,
        total=len(models),
        desc="Build geometry cache (models)",
        use_tqdm=use_progress,
    )
    for model_name in model_iter:
        print(f"[build] model={model_name}", flush=True)
        payload["models"][model_name] = build_model_cache(
            model_name=model_name,
            data_root=data_root,
            distance_scale=distance_scale,
            max_contours=max_contours,
            cut_x_mode=cut_x_mode,
            cut_x_offset=cut_x_offset,
        )

    return payload


def precompute_windows_cache(
    payload: dict[str, object],
    *,
    window_min: int,
    window_max: int,
    z_step: float,
    use_progress: bool,
) -> dict[str, object]:
    models_obj = payload.get("models", {})
    if not isinstance(models_obj, dict):
        raise ValueError("Invalid payload: models is not a dictionary")

    out_models: dict[str, object] = {}
    model_items = list(models_obj.items())
    model_iter = progress_iter(
        model_items,
        total=len(model_items),
        desc="Precompute windows (models)",
        use_tqdm=use_progress,
    )
    for model_name, model_data in model_iter:
        if not isinstance(model_data, dict):
            continue
        halves = model_data.get("half_contours", [])
        if not isinstance(halves, list):
            continue
        n_half = len(halves)
        if n_half < window_min:
            print(f"[skip] model={model_name}, halves={n_half} < window_min={window_min}", flush=True)
            continue

        z_min = float(model_data["z_min"])
        z_max = float(model_data["z_max"])
        z_levels = np.arange(z_min, z_max + 0.5 * z_step, z_step, dtype=float)
        n_levels = z_levels.size

        points_calc = [np.array(h.get("points_calc", h.get("points", [])), dtype=float) for h in halves]
        normals = np.array([h.get("normal_calc", h.get("normal", [0.0, 0.0, 0.0])) for h in halves], dtype=float)
        # Per-contour projection matrices: A_i = I - d_i d_i^T
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
            desc=f"{model_name}: interpolate half-contours on Z",
            use_tqdm=use_progress,
        )
        for hi in half_iter:
            poly = points_calc[hi]
            for zi, z in enumerate(z_levels):
                half_points[hi, zi, :] = half_contour_point_at_z(poly, float(z))

        max_w = min(window_max, n_half)
        print(
            f"[precompute] model={model_name}, halves={n_half}, z_levels={n_levels}, windows={window_min}..{max_w}",
            flush=True,
        )
        functions: dict[str, object] = {}
        window_iter = progress_iter(
            range(window_min, max_w + 1),
            total=max_w - window_min + 1,
            desc=f"{model_name}: windows",
            use_tqdm=use_progress,
        )
        for window in window_iter:
            line_points = np.full((n_half, n_levels, 3), np.nan, dtype=float)
            fit_rms = np.full((n_half, n_levels), np.nan, dtype=float)
            for zi in range(n_levels):
                p_zi = half_points[:, zi, :]  # (N,3)
                valid = np.all(np.isfinite(p_zi), axis=1) & valid_dn  # (N,)

                # A_i (valid only) and b_i = A_i p_i
                a_i = a_base.copy()
                a_i[~valid, :, :] = 0.0
                b_i = np.einsum("nij,nj->ni", a_base, p_zi, optimize=True)
                b_i[~valid, :] = 0.0
                c_i = valid.astype(np.int32)
                q_i = np.einsum("ni,ni->n", b_i, b_i, optimize=True)
                q_i[~valid] = 0.0

                # Cyclic sliding sums via doubled arrays + prefix sums.
                a_ext = np.concatenate([a_i, a_i], axis=0)  # (2N,3,3)
                b_ext = np.concatenate([b_i, b_i], axis=0)  # (2N,3)
                c_ext = np.concatenate([c_i, c_i], axis=0)  # (2N,)
                q_ext = np.concatenate([q_i, q_i], axis=0)  # (2N,)

                a_pref = np.concatenate(
                    [np.zeros((1, 3, 3), dtype=float), np.cumsum(a_ext, axis=0)],
                    axis=0,
                )  # (2N+1,3,3)
                b_pref = np.concatenate(
                    [np.zeros((1, 3), dtype=float), np.cumsum(b_ext, axis=0)],
                    axis=0,
                )  # (2N+1,3)
                c_pref = np.concatenate(
                    [np.zeros((1,), dtype=np.int32), np.cumsum(c_ext, axis=0)],
                    axis=0,
                )  # (2N+1,)
                q_pref = np.concatenate(
                    [np.zeros((1,), dtype=float), np.cumsum(q_ext, axis=0)],
                    axis=0,
                )  # (2N+1,)

                starts = np.arange(n_half, dtype=int)
                ends = starts + window
                a_sum = a_pref[ends] - a_pref[starts]  # (N,3,3)
                b_sum = b_pref[ends] - b_pref[starts]  # (N,3)
                c_sum = c_pref[ends] - c_pref[starts]  # (N,)
                q_sum = q_pref[ends] - q_pref[starts]  # (N,)

                valid_ws = c_sum >= 2
                if not np.any(valid_ws):
                    continue

                mats = a_sum[valid_ws]
                rhs = b_sum[valid_ws]
                dets = np.linalg.det(mats)
                solvable = np.abs(dets) > EPS
                if np.any(solvable):
                    mats_ok = mats[solvable]
                    rhs_ok = rhs[solvable]
                    inv_ok = np.linalg.inv(mats_ok)
                    sol = np.einsum("nij,nj->ni", inv_ok, rhs_ok, optimize=True)
                    idx_valid = np.flatnonzero(valid_ws)
                    idx_solvable = idx_valid[solvable]
                    line_points[idx_solvable, zi, :] = sol

                # Fallback for near-singular cases.
                if np.any(~solvable):
                    idx_valid = np.flatnonzero(valid_ws)
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
                    x = line_points[valid_lp, zi, :]  # (M,3)
                    a_loc = a_sum[valid_lp]  # (M,3,3)
                    b_loc = b_sum[valid_lp]  # (M,3)
                    q_loc = q_sum[valid_lp]  # (M,)
                    c_loc = c_sum[valid_lp].astype(float)  # (M,)
                    ax = np.einsum("nij,nj->ni", a_loc, x, optimize=True)  # (M,3)
                    x_ax = np.einsum("ni,ni->n", x, ax, optimize=True)  # (M,)
                    b_x = np.einsum("ni,ni->n", b_loc, x, optimize=True)  # (M,)
                    sse = x_ax - 2.0 * b_x + q_loc
                    sse = np.maximum(sse, 0.0)
                    fit_rms[valid_lp, zi] = np.sqrt(sse / np.maximum(c_loc, 1.0))

            distances = np.full((n_half, n_levels), np.nan, dtype=float)
            for i in range(n_half):
                nxt = (i + 1) % n_half
                dxyz = line_points[nxt, :, :] - line_points[i, :, :]
                distances[i, :] = np.linalg.norm(dxyz, axis=1)

            key = f"w={window}"
            functions[key] = {
                "window_size": int(window),
                "z_step": float(z_step),
                "z_levels": encode_array(z_levels, "<f4"),
                "distances": encode_array(distances, "<f4"),
                "fit_rms": encode_array(fit_rms, "<f4"),
            }

        out_models[model_name] = {
            "n_half_contours": int(n_half),
            "window_min": int(window_min),
            "window_max": int(max_w),
            "z_step": float(z_step),
            "functions": functions,
        }

    return {
        "source_schema_version": int(payload.get("schema_version", -1)),
        "source_generated_at_utc": payload.get("generated_at_utc"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_min": int(window_min),
        "window_max": int(window_max),
        "z_step": float(z_step),
        "models": out_models,
    }


def is_dynamic_cache_compatible(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if int(payload.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        return False
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        return False
    required = {
        "z_min",
        "z_max",
        "half_contours",
        "vertices",
        "triangles",
        "edges",
        "model_center",
        "split_line_point_display",
        "split_line_dir",
        "model_radius",
    }
    for model_obj in models.values():
        if not isinstance(model_obj, dict):
            return False
        if not required.issubset(model_obj.keys()):
            return False
    return True


def load_embedded_model_cache(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    marker = '<script id="cacheData" type="application/json">'
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Cannot find embedded cacheData in {path}")
    start += len(marker)
    end = text.find("</script>", start)
    if end < 0:
        raise ValueError(f"Cannot find closing cacheData script in {path}")
    payload = json.loads(text[start:end])
    if not is_dynamic_cache_compatible(payload):
        raise ValueError(f"Embedded cache in {path} is incompatible with schema {CACHE_SCHEMA_VERSION}")
    return payload


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build dynamic HTML viewer for full-circle one-point function. "
            "Cache stores geometry for all models; function is recalculated in browser for arbitrary z-step/window."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names. Default: all models under data-root",
    )
    parser.add_argument("--max-contours", type=int, default=None)
    parser.add_argument("--distance-scale", type=float, default=0.18)
    parser.add_argument(
        "--cut-x-mode",
        type=str,
        choices=("model-center", "bbox-mid"),
        default="model-center",
        help="How to pick vertical cut line x-coordinate",
    )
    parser.add_argument(
        "--cut-x-offset",
        type=float,
        default=0.0,
        help="Extra x offset added to selected cut line",
    )
    parser.add_argument(
        "--cache-json",
        type=Path,
        default=Path("output/full_circle_split_dynamic_models_cache.json"),
    )
    parser.add_argument(
        "--cache-html",
        type=Path,
        default=None,
        help="Reuse the embedded model cache from an existing viewer HTML without creating another large cache file.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("output/full_circle_split_dynamic_viewer.html"),
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force cache rebuild even if cache-json already exists",
    )
    parser.add_argument(
        "--precompute-window-min",
        type=int,
        default=0,
        help="If >0, precompute distances for all windows in [min, max] for all models and store into windows-cache-json",
    )
    parser.add_argument(
        "--precompute-window-max",
        type=int,
        default=0,
        help="Upper bound for precomputed windows range (used with --precompute-window-min)",
    )
    parser.add_argument(
        "--precompute-z-step",
        type=float,
        default=0.01,
        help="Z step for precomputed windows cache",
    )
    parser.add_argument(
        "--windows-cache-json",
        type=Path,
        default=Path("output/full_circle_split_windows_cache.json"),
        help="Output path for precomputed windows cache",
    )
    parser.add_argument(
        "--embed-windows-cache-in-html",
        action="store_true",
        help="Embed windows-cache-json payload into output HTML and use it for no-recompute mode",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    args = parser.parse_args()
    use_progress = not args.no_progress
    if args.cut_x_mode != "model-center" or abs(float(args.cut_x_offset)) > 0.0:
        print(
            "[warn] cut-x parameters are ignored; split line is always through model center of mass.",
            flush=True,
        )

    explicit_models = [m.strip() for m in args.models.split(",") if m.strip()] or None

    if args.cache_html is not None:
        payload = load_embedded_model_cache(args.cache_html)
        print(f"[cache] reuse embedded cache: {args.cache_html}")
    elif args.cache_json.exists() and not args.rebuild_cache and explicit_models is None:
        payload = json.loads(args.cache_json.read_text(encoding="utf-8"))
        if is_dynamic_cache_compatible(payload):
            print(f"[cache] reuse existing: {args.cache_json}")
        else:
            print(f"[cache] incompatible schema -> rebuild: {args.cache_json}")
            models = discover_models(args.data_root, explicit_models)
            if not models:
                raise ValueError("No models found")
            payload = build_cache_payload(
                data_root=args.data_root,
                models=models,
                max_contours=args.max_contours,
                distance_scale=args.distance_scale,
                cut_x_mode=args.cut_x_mode,
                cut_x_offset=args.cut_x_offset,
                use_progress=use_progress,
            )
            args.cache_json.parent.mkdir(parents=True, exist_ok=True)
            args.cache_json.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            print(f"[done] cache: {args.cache_json}")
    else:
        models = discover_models(args.data_root, explicit_models)
        if not models:
            raise ValueError("No models found")

        payload = build_cache_payload(
            data_root=args.data_root,
            models=models,
            max_contours=args.max_contours,
            distance_scale=args.distance_scale,
            cut_x_mode=args.cut_x_mode,
            cut_x_offset=args.cut_x_offset,
            use_progress=use_progress,
        )

        args.cache_json.parent.mkdir(parents=True, exist_ok=True)
        args.cache_json.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"[done] cache: {args.cache_json}")

    windows_payload: dict[str, object] | None = None
    if args.precompute_window_min > 0 and args.precompute_window_max >= args.precompute_window_min:
        if args.precompute_z_step <= 0:
            raise ValueError("--precompute-z-step must be > 0")
        print(
            "[precompute] start windows cache: "
            f"range={args.precompute_window_min}..{args.precompute_window_max}, z_step={args.precompute_z_step}",
            flush=True,
        )
        windows_payload = precompute_windows_cache(
            payload=payload,
            window_min=args.precompute_window_min,
            window_max=args.precompute_window_max,
            z_step=args.precompute_z_step,
            use_progress=use_progress,
        )
        args.windows_cache_json.parent.mkdir(parents=True, exist_ok=True)
        args.windows_cache_json.write_text(
            json.dumps(windows_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        ws_mb = args.windows_cache_json.stat().st_size / (1024 * 1024)
        print(f"[done] windows cache: {args.windows_cache_json} ({ws_mb:.2f} MB)", flush=True)

    if args.embed_windows_cache_in_html and windows_payload is None:
        if not args.windows_cache_json.exists():
            raise ValueError(
                "--embed-windows-cache-in-html requested, but windows cache file is missing. "
                "Run with --precompute-window-min/--precompute-window-max first or provide existing --windows-cache-json."
            )
        windows_payload = json.loads(args.windows_cache_json.read_text(encoding="utf-8"))

    html = build_html(payload, windows_payload if args.embed_windows_cache_in_html else None)
    atomic_write_text(args.output_html, html)

    size_mb = args.output_html.stat().st_size / (1024 * 1024)
    print(f"[done] html:  {args.output_html} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
