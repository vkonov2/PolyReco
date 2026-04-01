from __future__ import annotations

import argparse
import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EPS = 1e-9


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


def global_z_grid(contours: list[ShadowContour], z_step: float) -> np.ndarray:
    if z_step <= 0:
        raise ValueError("z_step must be > 0")
    all_points = np.vstack([c.points for c in contours])
    z_min = float(np.min(all_points[:, 2]))
    z_max = float(np.max(all_points[:, 2]))
    levels = np.arange(z_min, z_max + 0.5 * z_step, z_step, dtype=float)
    if levels.size < 2:
        raise ValueError("Z grid is too small; choose a smaller z_step")
    return levels


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


def dedup_points(points: list[np.ndarray], tol: float = 1e-7) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for p in points:
        if not any(np.linalg.norm(p - q) <= tol for q in unique):
            unique.append(p)
    return unique


def contour_points_at_z(contour: ShadowContour, z_level: float, lateral_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = contour.points
    n = pts.shape[0]
    intersections: list[np.ndarray] = []

    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        z0 = p0[2]
        z1 = p1[2]

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
    if len(intersections) < 2:
        nan_point = np.array([np.nan, np.nan, np.nan], dtype=float)
        return nan_point, nan_point

    coords = np.array([float(p @ lateral_axis) for p in intersections], dtype=float)
    left = intersections[int(np.argmin(coords))]
    right = intersections[int(np.argmax(coords))]
    return left, right


def interpolate_contour_on_z_grid(contour: ShadowContour, z_levels: np.ndarray) -> np.ndarray:
    n_levels = z_levels.size
    out = np.full((n_levels, 2, 3), np.nan, dtype=float)
    lat = lateral_axis_for_normal(contour.normal)
    for zi, z in enumerate(z_levels):
        left, right = contour_points_at_z(contour, float(z), lat)
        out[zi, 0, :] = left
        out[zi, 1, :] = right
    return out


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
        x = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
    return x


def compute_distances(contours: list[ShadowContour], z_step: float, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    n_cont = len(contours)
    if window_size > n_cont:
        raise ValueError("window_size must be <= number of contours")

    z_levels = global_z_grid(contours, z_step)
    n_levels = z_levels.size

    contour_lr = np.full((n_cont, n_levels, 2, 3), np.nan, dtype=float)
    for ci, contour in enumerate(contours):
        contour_lr[ci, :, :, :] = interpolate_contour_on_z_grid(contour, z_levels)

    n_windows = n_cont - window_size + 1
    normals = np.vstack([c.normal for c in contours])
    line_points = np.full((n_windows, n_levels, 2, 3), np.nan, dtype=float)

    for ws in range(n_windows):
        idx_arr = np.arange(ws, ws + window_size, dtype=int)
        window_normals = normals[idx_arr, :]

        for zi in range(n_levels):
            for side in (0, 1):
                cand = contour_lr[idx_arr, zi, side, :]
                valid = np.all(np.isfinite(cand), axis=1)
                if int(np.sum(valid)) < 2:
                    continue
                p_arr = cand[valid, :]
                d_arr = window_normals[valid, :]
                line_points[ws, zi, side, :] = closest_point_to_lines(p_arr, d_arr)

    distances = np.full((max(0, n_windows - 1), n_levels, 2), np.nan, dtype=float)
    for i in range(max(0, n_windows - 1)):
        dxyz = line_points[i + 1, :, :, :] - line_points[i, :, :, :]
        distances[i, :, :] = np.linalg.norm(dxyz, axis=2)

    return z_levels, distances


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
    z_steps: list[float],
    window_sizes: list[int],
) -> dict[str, object]:
    model_path = data_root / model_name / "InitialModel"
    shadow_dir = data_root / model_name / "shadow"

    model = parse_initial_model(model_path)
    contour_files = sorted_contour_files(shadow_dir, "merged-cont*", max_contours)
    contours = [parse_merged_contour(p) for p in contour_files]

    vertices = model.vertices
    model_center = vertices.mean(axis=0)
    centered_vertices = vertices - model_center
    model_radius = float(np.linalg.norm(centered_vertices, axis=1).max())

    contour_payload: list[dict[str, object]] = []
    for c in contours:
        support_max = float(np.max(centered_vertices @ c.normal))
        push = support_max + distance_scale * model_radius
        shifted = (c.points - model_center) + c.normal[None, :] * push
        contour_payload.append(
            {
                "index": int(c.index),
                "angle": float(c.angle),
                "normal": [float(c.normal[0]), float(c.normal[1]), float(c.normal[2])],
                "points": shifted.astype(np.float32).round(6).tolist(),
            }
        )

    tri_indices = triangulate_faces(model.faces)

    combos: dict[str, object] = {}
    for z_step in z_steps:
        for window in window_sizes:
            if window > len(contours):
                continue
            z_levels, distances = compute_distances(contours, z_step, window)
            key = f"z={z_step:.6f}|w={window}"
            combos[key] = {
                "z_step": float(z_step),
                "window_size": int(window),
                "n_windows": int(distances.shape[0] + 1),
                "z_levels": encode_array(z_levels, "<f4"),
                "distances": encode_array(distances, "<f4"),
            }

    return {
        "name": model_name,
        "num_vertices": int(centered_vertices.shape[0]),
        "num_faces": int(len(model.faces)),
        "vertices": encode_array(centered_vertices, "<f4"),
        "triangles": encode_array(tri_indices, "<u4"),
        "contours": contour_payload,
        "functions": combos,
    }


def build_html(cache_obj: dict[str, object]) -> str:
    cache_json = json.dumps(cache_obj, ensure_ascii=False, separators=(",", ":"))
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Combined Projection + Function Viewer</title>
  <script src="./plotly-2.35.2.min.js"></script>
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
    html, body { margin: 0; padding: 0; background: linear-gradient(160deg, #eef5ff 0%, #f8fbff 50%, #f0f4f8 100%); color: var(--ink); font-family: "Trebuchet MS", "Segoe UI", sans-serif; }
    .app { max-width: 1450px; margin: 0 auto; padding: 14px; display: grid; gap: 12px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; box-shadow: 0 8px 24px rgba(20,33,61,0.06); }
    .field { display: grid; gap: 5px; }
    .field label { font-size: 12px; color: var(--muted); }
    .field input, .field select { width: 100%; }
    .top-row { display: grid; grid-template-columns: 1fr; padding: 10px 12px; }
    .top-main { display: grid; grid-template-columns: 1fr 260px; gap: 10px; align-items: stretch; }
    .plot { min-height: 200px; border-radius: 12px; overflow: hidden; }
    .side-controls { padding: 10px; display: grid; gap: 10px; align-content: start; }
    .bottom-main { display: grid; grid-template-columns: 1fr 240px; gap: 10px; align-items: stretch; }
    .stack { display: grid; grid-template-rows: 1fr 1fr; gap: 10px; }
    .params { padding: 10px; display: grid; gap: 9px; align-content: start; }
    .z-wide { padding: 10px 12px; }
    .small { font-size: 12px; color: var(--muted); }
    @media (max-width: 1100px) {
      .top-main { grid-template-columns: 1fr; }
      .bottom-main { grid-template-columns: 1fr; }
      .stack { grid-template-rows: 42vh 42vh; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="panel top-row">
      <div class="field" style="max-width:300px;">
        <label for="modelSel">Model</label>
        <select id="modelSel"></select>
      </div>
    </div>

    <div class="top-main">
      <div id="topPlot" class="panel plot" style="height:56vh;"></div>
      <div class="panel side-controls">
        <div class="field">
          <label for="projIdx">Projection index</label>
          <input id="projIdx" type="range" min="0" max="0" value="0" step="1" />
        </div>
        <div class="field">
          <label for="projMode">Projection mode</label>
          <select id="projMode"><option value="perspective">Perspective</option><option value="orthographic">Parallel</option></select>
        </div>
      </div>
    </div>

    <div class="bottom-main">
      <div class="stack">
        <div id="contextPlot" class="panel plot" style="height:44vh;"></div>
        <div id="fnPlot" class="panel plot" style="height:44vh;"></div>
      </div>
      <div class="panel params">
        <div class="field">
          <label for="zStepSel">Z step (precomputed)</label>
          <select id="zStepSel"></select>
        </div>
        <div class="field">
          <label for="windowSel">Window size (precomputed)</label>
          <select id="windowSel"></select>
        </div>
        <div class="field">
          <label for="sideSel">Side</label>
          <select id="sideSel"><option value="0">Left</option><option value="1">Right</option></select>
        </div>
        <div class="field">
          <label for="trimBottom">Trim bottom</label>
          <input id="trimBottom" type="number" min="0" value="2" />
        </div>
        <div class="field">
          <label for="trimTop">Trim top</label>
          <input id="trimTop" type="number" min="0" value="20" />
        </div>
        <div class="small" id="status"></div>
      </div>
    </div>

    <div class="panel z-wide">
      <div class="field">
        <label for="zIdx">Z level index</label>
        <input id="zIdx" type="range" min="0" max="0" value="0" step="1" />
      </div>
    </div>
  </div>
  <script id="cacheData" type="application/json">__CACHE_JSON__</script>
  <script>
    const CACHE = JSON.parse(document.getElementById('cacheData').textContent);

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
        out[i] = [data[k], data[k + 1], data[k + 2]];
        k += m;
      }
      return out;
    }

    const state = {
      modelName: null,
      contourIdx: 0,
      projection: 'perspective',
      zStep: null,
      window: null,
      zIdx: 0,
      side: 0,
      trimBottom: 2,
      trimTop: 20,
      hoveredDistIdx: null,
    };

    const modelSel = document.getElementById('modelSel');
    const projIdx = document.getElementById('projIdx');
    const projMode = document.getElementById('projMode');
    const zStepSel = document.getElementById('zStepSel');
    const windowSel = document.getElementById('windowSel');
    const zIdx = document.getElementById('zIdx');
    const sideSel = document.getElementById('sideSel');
    const trimBottom = document.getElementById('trimBottom');
    const trimTop = document.getElementById('trimTop');
    const status = document.getElementById('status');

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

    function ensureModelDecoded(name) {
      if (decodedModels[name]) return decodedModels[name];
      const m = CACHE.models[name];
      const v = decodeArray(m.vertices);
      const t = decodeArray(m.triangles);
      const vertices = toPointList(v.data, v.shape);
      const triangles = toPointList(t.data, t.shape);

      const combos = {};
      for (const [key, combo] of Object.entries(m.functions)) {
        const zl = decodeArray(combo.z_levels);
        const ds = decodeArray(combo.distances);
        combos[key] = {
          z_step: combo.z_step,
          window_size: combo.window_size,
          n_windows: combo.n_windows,
          z_levels: zl.data,
          distances: ds.data,
          distShape: ds.shape,
        };
      }

      const decoded = { ...m, vertices, triangles, combos };
      decodedModels[name] = decoded;
      return decoded;
    }

    function comboKey(zStep, windowSize) {
      return `z=${Number(zStep).toFixed(6)}|w=${Number(windowSize)}`;
    }

    function currentCombo() {
      const m = ensureModelDecoded(state.modelName);
      return m.combos[comboKey(state.zStep, state.window)];
    }

    function updateComboSelectors() {
      const m = ensureModelDecoded(state.modelName);
      const combos = Object.values(m.combos);
      const zSteps = [...new Set(combos.map(c => Number(c.z_step).toFixed(6)))].sort((a,b)=>Number(a)-Number(b));
      zStepSel.innerHTML = '';
      zSteps.forEach(z => {
        const opt = document.createElement('option');
        opt.value = z;
        opt.textContent = z;
        zStepSel.appendChild(opt);
      });
      if (!state.zStep || !zSteps.includes(Number(state.zStep).toFixed(6))) state.zStep = Number(zSteps[0]);
      zStepSel.value = Number(state.zStep).toFixed(6);

      const windows = combos
        .filter(c => Number(c.z_step).toFixed(6) === Number(state.zStep).toFixed(6))
        .map(c => c.window_size)
        .sort((a,b)=>a-b);
      windowSel.innerHTML = '';
      windows.forEach(w => {
        const opt = document.createElement('option');
        opt.value = String(w);
        opt.textContent = String(w);
        windowSel.appendChild(opt);
      });
      if (!state.window || !windows.includes(Number(state.window))) state.window = windows[0];
      windowSel.value = String(state.window);
    }

    function updateZBounds() {
      const combo = currentCombo();
      if (!combo) return;
      const nLevels = combo.z_levels.length;
      const lo = Math.max(0, Number(state.trimBottom));
      const hi = Math.max(lo, nLevels - 1 - Math.max(0, Number(state.trimTop)));
      if (state.zIdx < lo) state.zIdx = lo;
      if (state.zIdx > hi) state.zIdx = hi;
      zIdx.min = String(lo);
      zIdx.max = String(hi);
      zIdx.value = String(state.zIdx);
    }

    function contourCamera(normal) {
      const eye = [-normal[0], -normal[1], -normal[2] + 0.2];
      const norm = Math.hypot(eye[0], eye[1], eye[2]) || 1;
      const k = 2.2 / norm;
      return { x: eye[0] * k, y: eye[1] * k, z: eye[2] * k };
    }

    function meshTraces(m) {
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
        const edgeSet = new Set();
        const segX = [];
        const segY = [];
        const segZ = [];
        for (const tri of m.triangles) {
          const triEdges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
          for (const e of triEdges) {
            const a = Math.min(e[0], e[1]);
            const b = Math.max(e[0], e[1]);
            const key = a + '_' + b;
            if (edgeSet.has(key)) continue;
            edgeSet.add(key);
            const p0 = m.vertices[a];
            const p1 = m.vertices[b];
            segX.push(p0[0], p1[0], null);
            segY.push(p0[1], p1[1], null);
            segZ.push(p0[2], p1[2], null);
          }
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
      x.push(points[0][0]); y.push(points[0][1]); z.push(points[0][2]);
      return {
        type: 'scatter3d',
        mode: 'lines',
        x, y, z,
        line: { color, width },
        hoverinfo: 'skip',
        showlegend: false,
      };
    }

    function renderTopPlot() {
      const m = ensureModelDecoded(state.modelName);
      const c = m.contours[Math.max(0, Math.min(m.contours.length - 1, state.contourIdx))];
      Plotly.react('topPlot', [
        meshTraces(m),
        edgeTrace(m),
        contourLineTrace(c.points, '#d62828', 6),
      ], {
        margin: {l: 0, r: 0, b: 0, t: 44},
        title: `${m.name} | merged-cont${String(c.index).padStart(3,'0')}`,
        scene: {
          xaxis: {title: 'X'},
          yaxis: {title: 'Y'},
          zaxis: {title: 'Z'},
          camera: { eye: contourCamera(c.normal), projection: { type: state.projection } },
          aspectmode: 'data',
        },
      }, {responsive: true, displaylogo: false});
    }

    function extractY(combo, zIndex, side) {
      const [dCount, lCount, sCount] = combo.distShape;
      const out = new Array(dCount);
      for (let di = 0; di < dCount; di++) {
        out[di] = combo.distances[di * lCount * sCount + zIndex * sCount + side];
      }
      return out;
    }

    function activeContourIndices() {
      const m = ensureModelDecoded(state.modelName);
      const combo = currentCombo();
      const maxDi = Math.max(0, combo.distShape[0] - 1);
      const di = state.hoveredDistIdx == null ? 0 : Math.max(0, Math.min(maxDi, state.hoveredDistIdx));
      const start = di;
      const end = Math.min(m.contours.length - 1, di + Number(state.window));
      const out = [];
      for (let i = start; i <= end; i++) out.push(i);
      return { di, out };
    }

    function renderContextPlot() {
      const m = ensureModelDecoded(state.modelName);
      const data = [meshTraces(m), edgeTrace(m)];
      const active = activeContourIndices();
      for (const ci of active.out) {
        data.push(contourLineTrace(m.contours[ci].points, '#ef476f', 4));
      }
      Plotly.react('contextPlot', data, {
        margin: {l: 0, r: 0, b: 0, t: 44},
        title: `Contours for hovered distance index i=${active.di} (window=${state.window}, total=${active.out.length})`,
        scene: {
          xaxis: {title: 'X'},
          yaxis: {title: 'Y'},
          zaxis: {title: 'Z'},
          camera: { eye: {x: 1.9, y: 1.5, z: 1.1}, projection: { type: state.projection } },
          aspectmode: 'data',
        },
      }, {responsive: true, displaylogo: false});
    }

    function renderFnPlot() {
      const combo = currentCombo();
      if (!combo) return;
      updateZBounds();
      const y = extractY(combo, state.zIdx, state.side);
      const x = Array.from({length: y.length}, (_, i) => i);
      const sideName = state.side === 0 ? 'left' : 'right';
      const zVal = combo.z_levels[state.zIdx] ?? NaN;
      status.textContent = `z_step=${Number(state.zStep).toFixed(6)}, window=${state.window}, side=${sideName}, z=${Number(zVal).toFixed(6)}`;

      Plotly.react('fnPlot', [{
        type: 'scatter',
        mode: 'lines',
        x, y,
        line: { color: '#0077b6', width: 2.5 },
        connectgaps: false,
      }], {
        margin: {l: 58, r: 14, b: 48, t: 44},
        title: `Distance Function | side=${sideName} | z=${Number(zVal).toFixed(6)}`,
        xaxis: { title: 'Window transition index i' },
        yaxis: { title: 'Distance' },
      }, {responsive: true, displaylogo: false});
      bindFnPlotMouseTracking(y.length);
    }

    function bindFnPlotMouseTracking(nPoints) {
      const gd = document.getElementById('fnPlot');
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
        renderContextPlot();
      });
    }

    function syncControlsFromState() {
      const m = ensureModelDecoded(state.modelName);
      projIdx.max = String(Math.max(0, m.contours.length - 1));
      projIdx.value = String(state.contourIdx);
      projMode.value = state.projection;
      sideSel.value = String(state.side);
      trimBottom.value = String(state.trimBottom);
      trimTop.value = String(state.trimTop);
    }

    function renderAll() {
      syncControlsFromState();
      renderTopPlot();
      renderFnPlot();
      renderContextPlot();
    }

    modelSel.addEventListener('change', () => {
      state.modelName = modelSel.value;
      state.contourIdx = 0;
      state.zIdx = 0;
      state.hoveredDistIdx = null;
      updateComboSelectors();
      updateZBounds();
      renderAll();
    });
    projIdx.addEventListener('input', () => { state.contourIdx = Number(projIdx.value); renderTopPlot(); });
    projMode.addEventListener('change', () => { state.projection = projMode.value; renderTopPlot(); renderContextPlot(); });
    zStepSel.addEventListener('change', () => { state.zStep = Number(zStepSel.value); updateComboSelectors(); updateZBounds(); state.hoveredDistIdx = null; renderFnPlot(); renderContextPlot(); });
    windowSel.addEventListener('change', () => { state.window = Number(windowSel.value); state.hoveredDistIdx = null; renderFnPlot(); renderContextPlot(); });
    zIdx.addEventListener('input', () => { state.zIdx = Number(zIdx.value); renderFnPlot(); });
    sideSel.addEventListener('change', () => { state.side = Number(sideSel.value); state.hoveredDistIdx = null; renderFnPlot(); renderContextPlot(); });
    trimBottom.addEventListener('change', () => { state.trimBottom = Math.max(0, Number(trimBottom.value) || 0); updateZBounds(); renderFnPlot(); });
    trimTop.addEventListener('change', () => { state.trimTop = Math.max(0, Number(trimTop.value) || 0); updateZBounds(); renderFnPlot(); });

    updateComboSelectors();
    updateZBounds();
    renderAll();
  </script>
</body>
</html>
""".replace("__CACHE_JSON__", cache_json)


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


def parse_float_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float value")
    return sorted(set(vals))


def parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer value")
    return sorted(set(vals))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build single-file HTML viewer that merges 3D projection and distance-function viewer "
            "with precomputed model cache."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names. Default: auto-discover all under data-root",
    )
    parser.add_argument("--max-contours", type=int, default=None)
    parser.add_argument("--distance-scale", type=float, default=0.18)
    parser.add_argument(
        "--z-step-options",
        type=str,
        default="0.01",
        help="Comma-separated z-step values to precompute",
    )
    parser.add_argument(
        "--window-options",
        type=str,
        default="10",
        help="Comma-separated window sizes to precompute",
    )
    parser.add_argument(
        "--cache-json",
        type=Path,
        default=Path("data/cache/combined_projection_functions_cache.json"),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("data/combined_projection_functions_viewer.html"),
    )
    args = parser.parse_args()

    explicit_models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    models = discover_models(args.data_root, explicit_models)
    if not models:
        raise ValueError("No models found")

    z_steps = parse_float_list(args.z_step_options)
    window_sizes = parse_int_list(args.window_options)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "distance_scale": float(args.distance_scale),
        "z_step_options": z_steps,
        "window_options": window_sizes,
        "models": {},
    }

    for model_name in models:
        print(f"[build] model={model_name}", flush=True)
        payload["models"][model_name] = build_model_cache(
            model_name=model_name,
            data_root=args.data_root,
            distance_scale=args.distance_scale,
            max_contours=args.max_contours,
            z_steps=z_steps,
            window_sizes=window_sizes,
        )

    args.cache_json.parent.mkdir(parents=True, exist_ok=True)
    args.cache_json.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    html = build_html(payload)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")

    size_mb = args.output_html.stat().st_size / (1024 * 1024)
    print(f"[done] cache: {args.cache_json}")
    print(f"[done] html:  {args.output_html} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
