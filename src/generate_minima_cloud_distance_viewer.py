from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_reconstruction_quality import (
    point_triangle_distance_squared,
    triangulate_polygon_faces,
    unique_edges_from_faces,
)
from generate_full_circle_split_cached_viewer import (
    build_half_contours,
    parse_initial_model,
    parse_merged_contour,
    sorted_contour_files,
    top_points_vertical_axis_point,
)


EPS = 1e-12


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_models(path: Path) -> dict[str, dict[str, Any]]:
    """Read only this viewer's embedded payload or its standalone diagnostic JSON."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
        if payload.get("artifact_type") != "polyreco_minima_diagnostics":
            raise ValueError(f"Not a minima diagnostic JSON: {path}")
        models = payload.get("models")
    else:
        marker = "const MODELS = "
        packed_marker = '<script id="packed-models" type="application/json">'
        if packed_marker in raw:
            packed = json.loads(raw.split(packed_marker, 1)[1].split("</script>", 1)[0])
            models = unpack_models(packed)
        elif marker in raw:
            models, _ = json.JSONDecoder().raw_decode(raw.split(marker, 1)[1].lstrip())
        else:
            raise ValueError(f"Not a minima distance viewer: {path}")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"Missing model payloads in {path}")
    for name, payload in models.items():
        if not isinstance(payload, dict) or payload.get("model") != name or "cloud" not in payload or "reference" not in payload:
            raise ValueError(f"Invalid model {name!r} in {path}")
    return models


def pack_value(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")


def unpack_value(value: str) -> Any:
    return json.loads(gzip.decompress(base64.b64decode(value, validate=True)))


def pack_models(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Keep every value, but let the browser decode only the selected model/mode/view."""
    packed: dict[str, Any] = {"encoding": "gzip-base64-fragments-v1", "models": {}}
    for name, payload in models.items():
        entry = {"base": pack_value({k: v for k, v in payload.items() if k not in ("edge_diagnostics", "projection_diagnostics")})}
        if "edge_diagnostics" in payload:
            entry["edges"] = pack_value(payload["edge_diagnostics"])
        if "projection_diagnostics" in payload:
            projection = payload["projection_diagnostics"]
            entry["projection"] = {
                "metadata": pack_value({k: v for k, v in projection.items() if k != "per_view"}),
                "index": [{k: view[k] for k in ("contour_index", "angle", "finite_faces")} for view in projection["per_view"]],
                "views": [pack_value(view) for view in projection["per_view"]],
            }
        packed["models"][name] = entry
    return packed


def unpack_models(packed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if packed.get("encoding") != "gzip-base64-fragments-v1":
        raise ValueError("Unsupported minima viewer encoding")
    models = {}
    for name, entry in packed["models"].items():
        payload = unpack_value(entry["base"])
        if "edges" in entry:
            payload["edge_diagnostics"] = unpack_value(entry["edges"])
        if "projection" in entry:
            projection = unpack_value(entry["projection"]["metadata"])
            projection["per_view"] = [unpack_value(view) for view in entry["projection"]["views"]]
            payload["projection_diagnostics"] = projection
        models[name] = payload
    return models


def as_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def triangulate_faces_with_ids(
    vertices: np.ndarray,
    faces: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    triangles: list[np.ndarray] = []
    face_ids: list[int] = []
    for face_id, face in enumerate(faces):
        face_triangles = triangulate_polygon_faces(vertices, [face])
        if face_triangles.size == 0:
            continue
        triangles.append(face_triangles)
        face_ids.extend([int(face_id)] * int(face_triangles.shape[0]))
    if not triangles:
        raise ValueError("InitialModel has no non-degenerate finite faces")
    return np.concatenate(triangles, axis=0), np.asarray(face_ids, dtype=np.int32)


def nearest_finite_face_distances(
    points: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_face_ids: np.ndarray,
    *,
    chunk_size: int,
    ambiguity_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_count = int(points.shape[0])
    distances = np.full(point_count, float("inf"), dtype=float)
    nearest_face_ids = np.full(point_count, -1, dtype=np.int32)
    second_distances = np.full(point_count, float("inf"), dtype=float)
    face_ids = np.unique(triangle_face_ids)
    triangles_by_face = {
        int(face_id): triangles[triangle_face_ids == int(face_id)]
        for face_id in face_ids
    }

    step = max(1, int(chunk_size))
    for start in range(0, point_count, step):
        stop = min(start + step, point_count)
        chunk = points[start:stop]
        best_squared = np.full(chunk.shape[0], float("inf"), dtype=float)
        second_squared = np.full(chunk.shape[0], float("inf"), dtype=float)
        best_face = np.full(chunk.shape[0], -1, dtype=np.int32)

        for face_id, face_triangles in triangles_by_face.items():
            face_squared = np.full(chunk.shape[0], float("inf"), dtype=float)
            for triangle in face_triangles:
                a, b, c = vertices[triangle]
                face_squared = np.minimum(
                    face_squared,
                    point_triangle_distance_squared(chunk, a, b, c),
                )
            better = face_squared < best_squared
            second_squared = np.where(
                better,
                best_squared,
                np.minimum(second_squared, face_squared),
            )
            best_squared = np.where(better, face_squared, best_squared)
            best_face = np.where(better, int(face_id), best_face)

        distances[start:stop] = np.sqrt(np.maximum(best_squared, 0.0))
        second_distances[start:stop] = np.sqrt(np.maximum(second_squared, 0.0))
        nearest_face_ids[start:stop] = best_face

    ambiguous = (second_distances - distances) <= max(float(ambiguity_tol), EPS)
    return distances, nearest_face_ids, ambiguous


def distribution_summary(values: np.ndarray, bbox_diagonal: float) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite distances were computed")
    percentiles = np.percentile(finite, [50, 75, 90, 95, 99, 99.9])
    scale = max(float(bbox_diagonal), EPS)
    return {
        "count": int(finite.size),
        "sum_squared": float(np.sum(finite * finite)),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
        "median": float(percentiles[0]),
        "p75": float(percentiles[1]),
        "p90": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "p99_9": float(percentiles[5]),
        "max": float(np.max(finite)),
        "normalized": {
            "mean": float(np.mean(finite) / scale),
            "rms": float(np.sqrt(np.mean(finite * finite)) / scale),
            "median": float(percentiles[0] / scale),
            "p95": float(percentiles[3] / scale),
            "p99": float(percentiles[4] / scale),
            "max": float(np.max(finite) / scale),
        },
        "within_absolute": {
            "0.01": float(np.mean(finite <= 0.01)),
            "0.03": float(np.mean(finite <= 0.03)),
            "0.05": float(np.mean(finite <= 0.05)),
            "0.10": float(np.mean(finite <= 0.10)),
            "0.15": float(np.mean(finite <= 0.15)),
        },
        "within_bbox_fraction": {
            "0.001": float(np.mean(finite <= 0.001 * scale)),
            "0.0025": float(np.mean(finite <= 0.0025 * scale)),
            "0.005": float(np.mean(finite <= 0.005 * scale)),
            "0.01": float(np.mean(finite <= 0.01 * scale)),
        },
    }


def z_profile(
    distances: np.ndarray,
    z_indices: np.ndarray,
    z_levels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for z_index in np.unique(z_indices):
        mask = z_indices == int(z_index)
        values = distances[mask]
        if values.size == 0:
            continue
        z_value = (
            float(z_levels[int(z_index)])
            if 0 <= int(z_index) < int(z_levels.size)
            else None
        )
        rows.append(
            {
                "z_index": int(z_index),
                "z": z_value,
                "count": int(values.size),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }
        )
    return rows


def build_payload(
    *,
    model_name: str,
    reconstruction_json: Path,
    initial_model_path: Path,
    chunk_size: int,
    edge_diagnostics: bool = False,
    edge_tolerance_fraction: float = 0.001,
    ambiguity_tolerance_fraction: float = 0.00025,
    projection_rays: int = 360,
) -> dict[str, Any]:
    reconstruction = json.loads(reconstruction_json.read_text(encoding="utf-8"))
    cloud = reconstruction.get("viewer", {}).get("minima_cloud", {})
    points = np.asarray(cloud.get("points") or [], dtype=float).reshape((-1, 3))
    if points.shape[0] == 0:
        raise ValueError(f"{reconstruction_json}: viewer.minima_cloud.points is empty")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{reconstruction_json}: minima cloud contains non-finite coordinates")

    model = parse_initial_model(initial_model_path)
    vertices = np.asarray(model.vertices, dtype=float)
    faces = [[int(index) for index in face] for face in model.faces]
    triangles, triangle_face_ids = triangulate_faces_with_ids(vertices, faces)
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    distances, nearest_face_ids, ambiguous = nearest_finite_face_distances(
        points,
        vertices,
        triangles,
        triangle_face_ids,
        chunk_size=int(chunk_size),
        ambiguity_tol=max(bbox_diagonal * 1e-9, 1e-12),
    )

    point_count = int(points.shape[0])
    z_indices = np.asarray(cloud.get("z_indices") or [-1] * point_count, dtype=int)
    cyclic_indices = np.asarray(cloud.get("indices") or [-1] * point_count, dtype=int)
    rms_values = np.asarray(
        [as_float(value) for value in (cloud.get("values") or [None] * point_count)],
        dtype=float,
    )
    thresholds = np.asarray(
        [as_float(value) for value in (cloud.get("thresholds") or [None] * point_count)],
        dtype=float,
    )
    for name, array in (
        ("z_indices", z_indices),
        ("indices", cyclic_indices),
        ("values", rms_values),
        ("thresholds", thresholds),
    ):
        if int(array.shape[0]) != point_count:
            raise ValueError(f"viewer.minima_cloud.{name} length does not match points")

    z_levels = np.asarray(reconstruction.get("viewer", {}).get("z_levels") or [], dtype=float)
    parameters = reconstruction.get("parameters") if isinstance(reconstruction.get("parameters"), dict) else {}
    trusted_cloud = parameters.get("trusted_cloud") if isinstance(parameters.get("trusted_cloud"), dict) else {}
    summary = distribution_summary(distances, bbox_diagonal)
    summary.update(
        {
            "bbox_diagonal": bbox_diagonal,
            "reference_face_count": int(len(faces)),
            "nearest_face_count": int(np.unique(nearest_face_ids[nearest_face_ids >= 0]).size),
            "ambiguous_nearest_face_fraction": float(np.mean(ambiguous)),
            "minima_threshold_pct": as_float(cloud.get("threshold_pct")),
            "display_window": parameters.get("display_window"),
            "trusted_cloud_mode": trusted_cloud.get("chosen_mode"),
            "trusted_cloud_point_count": trusted_cloud.get("points"),
        }
    )

    payload = {
        "schema_version": 1,
        "model": model_name,
        "semantics": {
            "cloud": "viewer.minima_cloud: local fit_rms minima below the configured per-Z percentage threshold",
            "distance": "exact Euclidean distance to the nearest finite InitialModel polygon after deterministic triangulation",
            "nearest_face": "zero-based InitialModel face order; ties near shared edges/vertices are marked ambiguous",
            "production_note": "This is a posthoc diagnostic. minima_cloud is the blue display cloud, not the final active-plane set.",
            "color_default": "linear scale clipped at p99; points above p99 share the hottest color but keep exact hover values",
        },
        "sources": {
            "reconstruction_json": str(reconstruction_json),
            "initial_model": str(initial_model_path),
        },
        "reference": {
            "vertices": vertices.tolist(),
            "faces": faces,
            "triangles": triangles.astype(int).tolist(),
            "edges": unique_edges_from_faces(faces),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
        },
        "cloud": {
            "points": points.tolist(),
            "distances": distances.tolist(),
            "normalized_distances": (distances / max(bbox_diagonal, EPS)).tolist(),
            "nearest_face_ids": nearest_face_ids.astype(int).tolist(),
            "ambiguous": ambiguous.astype(bool).tolist(),
            "z_indices": z_indices.astype(int).tolist(),
            "cyclic_indices": cyclic_indices.astype(int).tolist(),
            "rms_values": [float(value) if np.isfinite(value) else None for value in rms_values],
            "thresholds": [float(value) if np.isfinite(value) else None for value in thresholds],
        },
        "summary": summary,
        "z_profile": z_profile(distances, z_indices, z_levels),
    }
    if edge_diagnostics:
        from polyreco.minima_edge_diagnostics import build_edge_diagnostics
        from polyreco.template_projection_diagnostics import build_projection_diagnostics

        if reconstruction.get("model") != model_name:
            raise ValueError("Reconstruction model does not match the diagnostic model")
        if z_levels.ndim != 1 or z_levels.size == 0 or not np.all(np.isfinite(z_levels)):
            raise ValueError("A finite source viewer.z_levels grid is required")
        window = parameters.get("display_window")
        if not isinstance(window, int) or window < 2:
            raise ValueError("A fixed source parameters.display_window is required")
        contour_paths = sorted_contour_files(
            initial_model_path.parent / "shadow", "merged-cont*", parameters.get("max_contours"),
        )
        contours = [parse_merged_contour(path) for path in contour_paths]
        split_point = top_points_vertical_axis_point(vertices)
        split_direction = np.array([0.0, 0.0, 1.0], dtype=float)
        halves = build_half_contours(contours, split_point, split_direction)
        origin = np.asarray(parameters.get("inside_point", np.mean(vertices, axis=0)), dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("Invalid polar origin")
        payload["z_levels"] = z_levels.tolist()
        payload["sources"]["provenance"] = {
            "reconstruction": file_fingerprint(reconstruction_json),
            "initial_model": file_fingerprint(initial_model_path),
            "parameters_sha256": canonical_sha256(parameters),
            "minima_cloud_sha256": canonical_sha256(cloud),
            "z_levels_sha256": canonical_sha256(z_levels.tolist()),
            "contours": [file_fingerprint(path) for path in contour_paths],
            "contour_count": len(contours),
            "half_contour_count": len(halves),
            "half_contour_order": "all half_id=0 in source file order, then all half_id=1",
            "contour_indices": [int(c.index) for c in contours],
            "contour_angles_radians": [float(c.angle) for c in contours],
            "split_line_point": split_point.tolist(),
            "split_line_dir": split_direction.tolist(),
            "display_window": window,
            "max_contours": parameters.get("max_contours"),
            "z_step": parameters.get("z_step"),
            "alignment": "identity; no per-level affine or Z remapping",
            "diagnostic_only": True,
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
            "diagnostic_code": [
                file_fingerprint(path) for path in (
                    Path(__file__),
                    Path(__file__).parent / "polyreco/minima_edge_diagnostics.py",
                    Path(__file__).parent / "polyreco/template_projection_diagnostics.py",
                    Path(__file__).parent / "polyreco/rms_reprojection.py",
                    Path(__file__).parent / "generate_full_circle_split_cached_viewer.py",
                    Path(__file__).parent / "benchmark_reconstruction_quality.py",
                )
            ],
        }
        print(f"{model_name}: edge association and coverage ({len(points)} points, {len(halves)} half-contours)", flush=True)
        payload["edge_diagnostics"] = build_edge_diagnostics(
            points=points, vertices=vertices, edges=np.asarray(payload["reference"]["edges"], dtype=int),
            surface_distances=distances, z_indices=z_indices, z_levels=z_levels,
            cyclic_indices=cyclic_indices, rms_values=rms_values, window_size=window,
            half_contours=halves, edge_tolerance_fraction=edge_tolerance_fraction,
            ambiguity_tolerance_fraction=ambiguity_tolerance_fraction, chunk_size=chunk_size,
        )
        print(f"{model_name}: finite-face projection control ({len(contours)} contours, {projection_rays} rays)", flush=True)
        payload["projection_diagnostics"] = build_projection_diagnostics(
            vertices=vertices, faces=faces, triangles=triangles, contours=contours,
            z_levels=z_levels, half_contours=halves, polar_origin=origin, ray_count=projection_rays,
        )
        payload["schema_version"] = 2
    return payload


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="polyreco-renderer-sha256" content="__RENDERER_SHA256__">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PolyReco — точки, рёбра и контуры</title>
<script src="__PLOTLY_SRC__"></script>
<style>
:root{color-scheme:light;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}*{box-sizing:border-box}body{margin:0;color:#18202b;background:#eef2f6}header{padding:14px 18px;background:#fff;border-bottom:1px solid #d8dee8}.top,.controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px 16px}h1{margin:0;font-size:19px}label{display:inline-flex;align-items:center;gap:6px;font-size:12px}select,input{accent-color:#2563eb}select,button{padding:6px 8px;border:1px solid #bcc6d3;border-radius:6px;background:#fff;font:inherit;font-size:12px}button{cursor:pointer}button:hover{background:#edf4ff}.controls{margin-top:12px}.metrics{display:flex;flex-wrap:wrap;gap:7px;margin-top:11px}.metric{padding:6px 9px;border:1px solid #dbe2eb;border-radius:5px;background:#f4f7fa;font-size:12px}.note{margin-top:9px;max-width:1400px;color:#566273;font-size:12px;line-height:1.5}main{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(360px,1fr);gap:12px;padding:12px}.panel{background:#fff;border:1px solid #d8dee8;border-radius:8px;overflow:hidden}#plot3d{height:700px}#plotHistogram,#plotZ{height:344px}.wide{grid-column:1/-1;padding:14px 16px}.wide h2{font-size:16px;margin:0 0 9px}.details{font-size:13px;line-height:1.5}.table-wrap{overflow:auto;max-height:440px;margin-top:10px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #e1e7ef;white-space:nowrap}th{position:sticky;top:0;background:#f5f7fb;z-index:1}tr[data-edge]{cursor:pointer}tr[data-edge]:hover,tr.selected{background:#eaf2ff}#zSlider{width:230px}.muted{color:#607087}[hidden]{display:none!important}details{font-size:12px;margin-top:9px;line-height:1.5}code{word-break:break-all}#projectionText{white-space:pre-line}@media(max-width:1050px){main{grid-template-columns:1fr}#plot3d{height:570px}.top{align-items:flex-start}#edgeSelect{max-width:88vw}}
</style>
</head>
<body>
<header>
<div class="top"><h1 id="title">PolyReco</h1><label>Модель <select id="modelSelect"></select></label><label>Режим <select id="modeSelect"><option value="surface">Расстояние до поверхности</option><option value="edges">Точки и конечные рёбра</option><option value="projection">Проверка исходных контуров</option></select></label></div>
<div class="controls" id="pointControls"><label>Верх цветовой шкалы <select id="colorLimit"><option value="bbox0_5">0,5% диагонали</option><option value="bbox1">1% диагонали</option><option value="p95">p95</option><option value="p99" selected>p99</option><option value="max">максимум</option></select></label><label>Размер точек <input id="pointSize" type="range" min="1" max="6" step="0.5" value="2.5"></label><label><input id="showModel" type="checkbox" checked> Показывать эталон</label></div>
<div class="controls" id="edgeControls" hidden><label>Ребро <select id="edgeSelect"></select></label><label>Класс точек <select id="stateSelect"><option value="all">Все</option><option value="edge">Однозначное геометрическое соответствие</option><option value="ambiguous">Неоднозначные</option><option value="face">Близко к поверхности</option><option value="outlier">Далеко от поверхности</option></select></label><label><input id="onlyEdge" type="checkbox" checked> Ближайшее ребро или кандидат</label><button id="clearEdge">Все рёбра</button></div>
<div class="controls" id="levelControls" hidden><label>Сечение <input id="zSlider" type="range" min="0" step="1"><select id="zSelect"></select></label><button data-z="-6">Разреженное ≈ −6</button><button data-z="-5">Плотное ≈ −5</button><button data-z="-2">Верхнее ≈ −2</button><button id="bottomLevel">Нижний уровень с точками</button><label><input id="onlyLevel" type="checkbox"> Только этот Z в 3D</label></div>
<div class="controls" id="projectionControls" hidden><label>Ракурс <select id="viewSelect"></select></label><button id="worstView">Наибольшая ошибка</button></div>
<div id="metrics" class="metrics"></div><div id="note" class="note">Загрузка выбранной модели…</div>
<details><summary>Источники и смысл измерений</summary><div id="provenance"></div></details>
</header>
<main>
<section class="panel"><div id="plot3d"></div></section>
<div><section class="panel"><div id="plotHistogram"></div></section><section class="panel" style="margin-top:12px"><div id="plotZ"></div></section></div>
<section id="edgePanel" class="panel wide" hidden><h2>Покрытие конечных рёбер</h2><div id="edgeDetails" class="details"></div><div class="controls"><label>Показать <select id="tableFilter"><option value="all">Все рёбра</option><option value="unsupported">Без однозначных точек</option><option value="supported">С однозначными точками</option><option value="short">Короткие по Z / горизонтальные</option></select></label><label>Сортировать <select id="tableSort"><option value="id">По номеру</option><option value="coverage">По покрытию ↑</option><option value="gap">По максимальному пробелу ↓</option></select></label></div><div class="table-wrap"><table><thead><tr><th>Ребро · вершины</th><th>Длина / ΔZ</th><th>Уровни: доступны / видимы</th><th>Точки: однозначные / кандидаты</th><th>Поддержанные Z / ракурсы</th><th>Покрытие / с кандидатами</th><th>Макс. пробел</th><th>Причина отсутствия поддержки</th></tr></thead><tbody id="edgeTable"></tbody></table></div></section>
<section id="projectionPanel" class="panel wide" hidden><h2>Что проверено в исходных наблюдениях</h2><div id="projectionText" class="details"></div></section>
</main>
<script id="packed-models" type="application/json">__DATA__</script>
<script>
function showLoadError(error){document.getElementById('note').textContent='Не удалось загрузить viewer: '+error.message}
async function unpack(encoded){
  if(typeof DecompressionStream==='undefined')throw new Error('нужен браузер с поддержкой DecompressionStream');
  const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return JSON.parse(await new Response(stream).text());
}
async function startViewer(){
const packedNode=document.getElementById('packed-models'),PACKED=JSON.parse(packedNode.textContent).models;
packedNode.textContent='';
const labels={round:'Round',princess:'Princess',radiant:'Radiant',pear:'Pear',cushion:'Cushion'};
const names=Object.keys(PACKED), requested=decodeURIComponent(location.hash.slice(1));
const model=Object.prototype.hasOwnProperty.call(PACKED,requested)?requested:names[0];
const fragments=PACKED[model],DATA=await unpack(fragments.base),cloud=DATA.cloud,ref=DATA.reference,summary=DATA.summary;
let ed=null,pd=null,ep=null,modeRevision=0,projectionRevision=0,edgesPromise=null,projectionPromise=null;
const $=id=>document.getElementById(id), cfg={responsive:true,displaylogo:false,scrollZoom:true};
const fmt=(x,n=5)=>x==null||!Number.isFinite(Number(x))?'—':Number(x).toFixed(n);
const pct=(x,n=1)=>x==null?'—':`${(100*x).toFixed(n)}%`;
const finite=x=>x!=null&&Number.isFinite(x);
const colorScale=[[0,'#1a9850'],[.45,'#91cf60'],[.68,'#fee08b'],[.84,'#fc8d59'],[1,'#d73027']];
const stateColors={edge:'#16826f',ambiguous:'#d38b19',face:'#4c78a8',outlier:'#d44242'};
const stateLabels={edge:'Однозначное соответствие',ambiguous:'Неоднозначно',face:'Близко к поверхности',outlier:'Далеко от поверхности'};
const reasons={no_sampled_z_intersection:'Нет пересечения с сеткой Z',only_ambiguous_points:'Только неоднозначные точки',near_points_prefer_other_edges:'Близкие точки предпочитают другие рёбра',no_ideal_angular_visibility:'Нет идеальной угловой видимости',no_full_window_visibility:'Нет целого окна для этого ребра',no_close_minima_points:'Нет близких минимумов',no_geometric_levels:'Нет уровня внутри ребра',not_ideally_visible:'Не видно на доступных ракурсах',no_close_points:'Нет близких точек',ambiguous_only:'Только неоднозначные точки',horizontal:'Горизонтальное ребро',insufficient_angular_support:'Недостаточно угловых наблюдений',no_unique_support:'Нет однозначной поддержки'};
const zLevels=DATA.z_levels||DATA.z_profile.reduce((a,r)=>{a[r.z_index]=r.z;return a},[]);
let mode='surface',selectedEdge=-1,zi=closestLevel(-5),viewIndex=0;
const byZ=new Map();cloud.z_indices.forEach((z,i)=>{if(!byZ.has(z))byZ.set(z,[]);byZ.get(z).push(i)});
function addOption(sel,value,text){const o=document.createElement('option');o.value=value;o.textContent=text;sel.appendChild(o)}
function closestLevel(z){let best=0,d=Infinity;zLevels.forEach((v,i)=>{if(finite(v)&&Math.abs(v-z)<d){d=Math.abs(v-z);best=i}});return best}
function stat(a){const s=a.filter(finite).sort((a,b)=>a-b);const q=p=>{const f=(s.length-1)*p,l=Math.floor(f);return s.length?s[l]+(s[Math.ceil(f)]-s[l])*(f-l):null};return {count:s.length,median:q(.5),p95:q(.95),p99:q(.99),max:s.at(-1),rms:s.length?Math.sqrt(s.reduce((n,v)=>n+v*v,0)/s.length):null}}
function badges(rows){$('metrics').replaceChildren();rows.forEach(t=>{const x=document.createElement('span');x.className='metric';x.textContent=t;$('metrics').appendChild(x)})}
function heading(title,x,y){return {margin:{l:60,r:22,t:88,b:52},title:{text:title,font:{size:14},y:.98,yanchor:'top'},xaxis:{title:x},yaxis:{title:y},paper_bgcolor:'#fff',plot_bgcolor:'#fff',legend:{orientation:'h',y:1.02,yanchor:'bottom',font:{size:10}}}}
let edgeDistances=[],edgeStats=stat([]);
const modelMesh={type:'mesh3d',x:ref.vertices.map(p=>p[0]),y:ref.vertices.map(p=>p[1]),z:ref.vertices.map(p=>p[2]),i:ref.triangles.map(t=>t[0]),j:ref.triangles.map(t=>t[1]),k:ref.triangles.map(t=>t[2]),color:'#c9d1dc',opacity:.19,flatshading:true,name:'Эталон',hoverinfo:'skip',showscale:false};
function edgeTrace(ids,color='#718096',width=1){const x=[],y=[],z=[];ids.forEach(e=>{for(const v of ref.edges[e]){const p=ref.vertices[v];x.push(p[0]);y.push(p[1]);z.push(p[2])}x.push(null);y.push(null);z.push(null)});return {type:'scatter3d',mode:'lines',x,y,z,line:{color,width},hoverinfo:'skip',showlegend:false}}
const modelEdges=edgeTrace(ref.edges.map((_,i)=>i));
const allPoints=cloud.points.map((_,i)=>i);
function candidates(i){return ep?.candidate_edge_ids?.[i]||[]}
function relevant(i){return selectedEdge<0||candidates(i).includes(selectedEdge)||ep?.nearest_edge_ids?.[i]===selectedEdge}
function selectedPoints(){return allPoints.filter(i=>(mode!=='edges'||(($('stateSelect').value==='all'||ep.states[i]===$('stateSelect').value)&&(!$('onlyEdge').checked||relevant(i))))&&!(mode==='edges'&&$('onlyLevel').checked&&cloud.z_indices[i]!==zi))}
function scalarDistance(i){return mode==='surface'?cloud.distances[i]:edgeDistances[i]}
function maxColor(){const v=$('colorLimit').value;if(v==='bbox0_5')return .5;if(v==='bbox1')return 1;const s=mode==='surface'?summary:edgeStats;return Math.max(1e-9,100*s[v]/summary.bbox_diagonal)}
function render3d(){const ids=selectedPoints(),isEdge=mode==='edges';
const hover=ids.map(i=>[i,scalarDistance(i),isEdge?stateLabels[ep.states[i]]:cloud.nearest_face_ids[i],cloud.z_indices[i],cloud.rms_values[i],isEdge?candidates(i).join(', '):cloud.cyclic_indices[i],isEdge?ep.condition_numbers?.[i]:null,isEdge?ep.uncertainty_proxy?.[i]:null]);
const points={type:'scatter3d',mode:'markers',x:ids.map(i=>cloud.points[i][0]),y:ids.map(i=>cloud.points[i][1]),z:ids.map(i=>cloud.points[i][2]),customdata:hover,name:'Точки',marker:{size:Number($('pointSize').value),color:ids.map(i=>100*scalarDistance(i)/summary.bbox_diagonal),colorscale:colorScale,cmin:0,cmax:maxColor(),opacity:.85,colorbar:{title:'% диагонали',thickness:16,len:.65}},hovertemplate:'точка %{customdata[0]}<br>x=%{x:.5f}; y=%{y:.5f}; z=%{z:.5f}<br>расстояние=%{customdata[1]:.6f}<br>'+(isEdge?'%{customdata[2]}<br>кандидаты: %{customdata[5]}<br>обусловленность=%{customdata[6]:.3g}<br>RMS-прокси ошибки=%{customdata[7]:.4g}':'ближайшая грань=%{customdata[2]}<br>индекс окна=%{customdata[5]}')+'<br>Z index=%{customdata[3]}<br>RMS=%{customdata[4]:.4g}<extra></extra>'};
const traces=$('showModel').checked?[modelMesh,modelEdges,points]:[points];if(isEdge&&selectedEdge>=0)traces.push(edgeTrace([selectedEdge],'#793cba',8));
const layout={margin:{l:0,r:0,t:44,b:0},title:{text:isEdge?`До ближайшего конечного ребра · показано ${ids.length} точек`:'До ближайшей конечной грани',font:{size:15}},showlegend:false,uirevision:model+'3d',scene:{aspectmode:'data',xaxis:{title:'X',showbackground:false},yaxis:{title:'Y',showbackground:false},zaxis:{title:'Z',showbackground:false}}};Plotly.react('plot3d',traces,layout,cfg)}
function renderSurface(){const ds=cloud.normalized_distances.map(v=>v*100),shown=ds.filter(v=>v<=summary.normalized.p99*100);
Plotly.react('plotHistogram',[{type:'histogram',x:shown,nbinsx:80,marker:{color:'#52708f'}}],heading(`Распределение до p99 · скрыто ${ds.length-shown.length} дальних точек`,'Расстояние, % диагонали','Точек'),cfg);
Plotly.react('plotZ',[{type:'scatter',mode:'lines',x:DATA.z_profile.map(r=>r.z),y:DATA.z_profile.map(r=>100*r.median/summary.bbox_diagonal),line:{color:'#16826f'},name:'Медиана'},{type:'scatter',mode:'lines',x:DATA.z_profile.map(r=>r.z),y:DATA.z_profile.map(r=>100*r.p95/summary.bbox_diagonal),line:{color:'#d44242'},name:'p95'}],heading('Ошибка до поверхности по Z','Z','% диагонали'),cfg)}
function sectionPoints(z){const out=[];ref.edges.forEach(([ia,ib],edge)=>{const a=ref.vertices[ia],b=ref.vertices[ib],dz=b[2]-a[2];if(Math.abs(dz)<1e-9){if(Math.abs(z-a[2])<1e-7){out.push({p:a,edge});out.push({p:b,edge})}}else{const t=(z-a[2])/dz;if(t>=-1e-7&&t<=1+1e-7)out.push({p:a.map((v,k)=>v+t*(b[k]-v)),edge})}});return out}
function selectedEdgeDistance(i){const [ia,ib]=ref.edges[selectedEdge],a=ref.vertices[ia],b=ref.vertices[ib],p=cloud.points[i],d=b.map((v,k)=>v-a[k]),l2=d.reduce((s,v)=>s+v*v,0),t=Math.max(0,Math.min(1,d.reduce((s,v,k)=>s+v*(p[k]-a[k]),0)/l2));return {t,d:Math.hypot(...p.map((v,k)=>v-a[k]-t*d[k]))}}
function renderSection(){const z=zLevels[zi],refs=sectionPoints(z),ids=byZ.get(zi)||[],traces=[{type:'scatter',mode:'markers',x:refs.map(r=>r.p[0]),y:refs.map(r=>r.p[1]),marker:{size:6,color:'#374151',symbol:'x'},customdata:refs.map(r=>r.edge),name:'Эталонное сечение',hovertemplate:'ребро=%{customdata}<br>x=%{x:.5f}; y=%{y:.5f}<extra></extra>'}];
Object.keys(stateColors).forEach(s=>{const ix=ids.filter(i=>ep.states[i]===s);traces.push({type:'scatter',mode:'markers',x:ix.map(i=>cloud.points[i][0]),y:ix.map(i=>cloud.points[i][1]),marker:{size:6,color:stateColors[s]},customdata:ix,name:stateLabels[s],hovertemplate:'точка=%{customdata}<br>x=%{x:.5f}; y=%{y:.5f}<extra></extra>'})});
if(selectedEdge>=0){const s=refs.filter(r=>r.edge===selectedEdge);traces.push({type:'scatter',mode:'markers',x:s.map(r=>r.p[0]),y:s.map(r=>r.p[1]),marker:{size:14,color:'#793cba',symbol:'star'},name:'Выбранное ребро'})}
const layout=heading(`Сечение Z=${fmt(z)} · ${ids.length} точек`,'X','Y');layout.yaxis.scaleanchor='x';layout.legend.font={size:9};Plotly.react('plotHistogram',traces,layout,cfg);
if(selectedEdge>=0){const pids=allPoints.filter(relevant);const pairs=pids.map(i=>selectedEdgeDistance(i));const tolerance=ed.tolerances?.edge_distance??ed.tolerances?.edge_tolerance??.001*summary.bbox_diagonal;const data=Object.keys(stateColors).map(s=>{const ix=pids.map((v,j)=>({i:v,j})).filter(o=>ep.states[o.i]===s);return {type:'scatter',mode:'markers',x:ix.map(o=>pairs[o.j].t),y:ix.map(o=>pairs[o.j].d),customdata:ix.map(o=>o.i),marker:{size:5,color:stateColors[s]},name:stateLabels[s],hovertemplate:'t=%{x:.4f}<br>до выбранного ребра=%{y:.6f}<br>точка=%{customdata}<extra></extra>'}});const r=ed.edges[selectedEdge];data.push({type:'scatter',mode:'lines',x:[0,1],y:[tolerance,tolerance],line:{color:'#9a6a10',dash:'dash'},name:'Диагностический допуск'});for(const [lo,hi] of r.coverage_intervals_t||[])data.push({type:'scatter',mode:'lines',x:[lo,hi],y:[0,0],line:{color:'#16826f',width:7},showlegend:false,hoverinfo:'skip'});const l=heading(`Ребро ${selectedEdge}: поддержка по длине`,'t: от первой вершины к второй','Расстояние');l.xaxis.range=[0,1];l.legend.font={size:9};Plotly.react('plotZ',data,l,cfg)}else{const rows=DATA.z_profile.map(row=>{const v=(byZ.get(row.z_index)||[]).map(i=>edgeDistances[i]);return {z:row.z,...stat(v)}});Plotly.react('plotZ',[{type:'scatter',mode:'lines',x:rows.map(r=>r.z),y:rows.map(r=>r.median),name:'Медиана',line:{color:'#16826f'}},{type:'scatter',mode:'lines',x:rows.map(r=>r.z),y:rows.map(r=>r.p95),name:'p95',line:{color:'#d44242'}}],heading('До конечного ребра по Z','Z','Расстояние'),cfg)}}
function reasonText(r){return reasons[r]||r||'—'}
function renderTable(){let rows=ed.edges.slice(),f=$('tableFilter').value;rows=rows.filter(r=>f==='all'||(f==='unsupported'&&r.unique_point_count===0)||(f==='supported'&&r.unique_point_count>0)||(f==='short'&&(r.horizontal||r.geometric_z_indices.length<3)));const sort=$('tableSort').value;if(sort==='coverage')rows.sort((a,b)=>a.coverage_fraction-b.coverage_fraction||a.edge_id-b.edge_id);if(sort==='gap')rows.sort((a,b)=>b.max_gap_fraction-a.max_gap_fraction||a.edge_id-b.edge_id);const tbody=$('edgeTable');tbody.replaceChildren();rows.forEach(r=>{const tr=document.createElement('tr');tr.dataset.edge=r.edge_id;tr.className=r.edge_id===selectedEdge?'selected':'';const vals=[`${r.edge_id} · ${r.vertices.join('—')}${r.horizontal?' · горизонт.':''}`,`${fmt(r.length,4)} / ${fmt(r.z_span,4)}`,`${r.geometric_z_indices.length} / ${r.ideal_visible_z_indices.length}`,`${r.unique_point_count} / ${r.candidate_point_count}`,`${r.unique_z_count} / ${r.unique_source_view_count}`,`${pct(r.coverage_fraction)} / ${pct(r.candidate_coverage_fraction)}`,pct(r.max_gap_fraction),reasonText(r.unsupported_reason)];vals.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td)});tr.onclick=()=>selectEdge(r.edge_id);tbody.appendChild(tr)});
if(selectedEdge<0){$('edgeDetails').textContent='Выберите ребро в таблице или списке. Покрытие — объединение локальных участков ребра в пределах допуска от однозначных точек; неподдержанные промежутки не соединяются. Доступность по Z и идеальная угловая видимость считаются отдельно. Ракурсы — различные исходные контуры, участвовавшие в окнах; они не являются независимыми подтверждениями ребра.';return}
const r=ed.edges[selectedEdge];$('edgeDetails').textContent=`Ребро ${r.edge_id} (${r.vertices.join('—')}): покрыто ${pct(r.coverage_fraction)}, с неоднозначными кандидатами ${pct(r.candidate_coverage_fraction)}; максимальный пробел ${pct(r.max_gap_fraction)}. Геометрически доступно ${r.geometric_z_indices.length} уровней, есть идеальная угловая видимость на ${r.ideal_visible_z_indices.length}, целое окно — на ${r.ideal_window_z_count}; однозначно поддержано ${r.unique_z_count} уровней. ${r.unique_point_count>0&&r.ideal_window_z_count===0?'Геометрическое соответствие есть, но целого видимого окна нет: принадлежность исходных лучей этому ребру не подтверждена.':''} ${r.horizontal?'Горизонтальное ребро: оценка по длине сегмента, длинный трек по Z не требуется.':''} ${r.unsupported_reason?'Причина отсутствия поддержки: '+reasonText(r.unsupported_reason)+'.':''}`}
function selectEdge(e){selectedEdge=Number(e);$('edgeSelect').value=String(e);if(e>=0){const r=ed.edges[e],zs=r.geometric_z_indices;if(zs.length){zi=zs[Math.floor(zs.length/2)];updateZ()}}render3d();renderSection();renderTable()}
function radialPoints(origin,radii){return radii.map((r,i)=>finite(r)?[origin[0]+r*Math.cos(2*Math.PI*i/radii.length),origin[1]+r*Math.sin(2*Math.PI*i/radii.length)]:[null,null])}
function line2(points,name,color,dash){return {type:'scatter',mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),line:{color,width:2,dash:dash||'solid'},name}}
function rmseOf(o){return o?.root_mean_squared_error??o?.rms??null}
async function renderProjection(){const request=++projectionRevision,revision=modeRevision,requestedView=viewIndex,v=await unpack(fragments.projection.views[requestedView]);if(mode!=='projection'||revision!==modeRevision||request!==projectionRevision)return;const origin=v.polar_origin_2d,mesh=radialPoints(origin,v.finite_outer_radii),hull=(v.hull_polygon_2d||[]).slice(),obs=(v.observed_polygon_2d||[]).slice();if(obs.length)obs.push(obs[0]);if(hull.length)hull.push(hull[0]);if(mesh.length)mesh.push(mesh[0]);let l=heading(`Ракурс ${v.contour_index} · угол ${fmt(v.angle*180/Math.PI,2)}°`,'Поперечная координата','Высота');l.yaxis.scaleanchor='x';Plotly.react('plot3d',[line2(obs,'Исходный контур','#303b4d'),line2(hull,'Выпуклая оболочка','#cd8c26','dash'),line2(mesh,`Конечные грани · ${pd.ray_count_per_view} лучей`,'#16826f')],l,cfg);
const ang=v.observed_outer_radii.map((_,i)=>360*i/v.observed_outer_radii.length),diff=(r)=>r.map((x,i)=>finite(x)&&finite(v.observed_outer_radii[i])?x-v.observed_outer_radii[i]:null);Plotly.react('plotHistogram',[{type:'scatter',mode:'lines',x:ang,y:diff(v.finite_outer_radii),line:{color:'#16826f'},name:'Конечные грани − наблюдение'},{type:'scatter',mode:'lines',x:ang,y:diff(v.hull_outer_radii),line:{color:'#cd8c26',dash:'dash'},name:'Оболочка − наблюдение'}],heading('Радиальная невязка одного ракурса','Угол луча, °','Расстояние'),cfg);
const probes=v.horizontal_probes||[];Plotly.react('plotZ',[{type:'scatter',mode:'lines',x:probes.map(r=>r.z),y:probes.map(r=>r.signed_left),name:'Левая граница',line:{color:'#5378ae'}},{type:'scatter',mode:'lines',x:probes.map(r=>r.z),y:probes.map(r=>r.signed_right),name:'Правая граница',line:{color:'#b75676'}}],heading('Невязка горизонтальных границ','Z','Модель − наблюдение'),cfg);
const s=pd.summary,g=pd.geometry||{};badges([`Ракурсов: ${pd.per_view.length}`,`RMS конечных граней: ${fmt(rmseOf(s.finite_faces))}`,`RMS оболочки: ${fmt(rmseOf(s.convex_hull))}`,`Текущий ракурс RMS: ${fmt(rmseOf(v.finite_faces))}`,`Неопорных граней: ${g.non_supporting_face_count??'—'}`]);
const horizontal=s.horizontal||{},cal=s.calibration||{},rays=s.ray_union||{};
const current=v.calibration||{},mz=current.model_z_range||[],oz=current.observed_z_range||[];
$('projectionText').textContent=[
'Контроль использует исходные контуры и конечные грани при неизменных координатах. Граница показана по угловой сетке; соединение внешних точек лучей не является точным полигоном силуэта.',
`Различие внешних границ конечных граней и выпуклой оболочки: RMS ${fmt(rmseOf(s.hull_minus_finite),8)}.`,
`Лучей без пересечения: модель ${rays.finite_missing??0}, наблюдения ${rays.observed_missing??0}. Лучей с раздельными интервалами: модель ${rays.finite_disjoint??0}, наблюдения ${rays.observed_disjoint??0}.`,
`Проверок на горизонтальных уровнях: ${horizontal.probe_count??0}. Расхождений выбора ветви: ${horizontal.half_branch_mismatch_count??0}; пропусков ветви при наличии полного контура: ${horizontal.half_branch_missing_count??0}.`,
`RMS левой / правой границы: ${fmt(horizontal.left_residual?.rms)} / ${fmt(horizontal.right_residual?.rms)}. Средний сдвиг середины сечения: ${fmt(horizontal.midpoint_shift?.mean_signed)}.`,
`Текущий ракурс: низ эталона ${fmt(mz[0])}, низ контура ${fmt(oz[0])}; верх ${fmt(mz[1])} / ${fmt(oz[1])}.`,
`Углы файлов проверены в принятом направлении вращения: максимальное расхождение ${fmt(cal.clockwise_angle_error_radians?.max_absolute,8)} рад. Проекции вычислены по фактической нормали.`,
'Остаточная ошибка не объявляется автоматически шумом. Полные распределения, интервалы, допуски и данные всех ракурсов сохранены в JSON.'
].join('\n\n');
}
function updateZ(){$('zSlider').value=zi;$('zSelect').value=zi}
async function updateMode(){const revision=++modeRevision;mode=$('modeSelect').value;
if(mode==='edges'&&!ed){$('note').textContent='Загрузка диагностики рёбер…';ed=await (edgesPromise ||= unpack(fragments.edges));ep=ed.points;edgeDistances=ep.nearest_edge_distances;edgeStats=stat(edgeDistances);$('edgeSelect').replaceChildren();addOption($('edgeSelect'),-1,'Все рёбра');ed.edges.forEach(r=>addOption($('edgeSelect'),r.edge_id,`${r.edge_id} · ${r.vertices.join('—')} · покрыто ${pct(r.coverage_fraction)}`));}
if(mode==='projection'&&!pd){$('note').textContent='Загрузка проверки контуров…';pd=await (projectionPromise ||= unpack(fragments.projection.metadata));pd.per_view=fragments.projection.index;}
if(revision!==modeRevision)return;
const edge=mode==='edges',projection=mode==='projection';$('edgeControls').hidden=!edge;$('levelControls').hidden=!edge;$('projectionControls').hidden=!projection;$('pointControls').hidden=projection;$('edgePanel').hidden=!edge;$('projectionPanel').hidden=!projection;
if(projection){$('note').textContent='Сопоставление эталона с исходными наблюдениями. Зелёная линия — внешняя граница проекции конечных граней на лучевой сетке, пунктир — прежняя выпуклая оболочка. Положительная радиальная невязка означает, что модель выходит за наблюдаемый контур.';await renderProjection();return}
if(edge){const counts={};ep.states.forEach(s=>counts[s]=(counts[s]||0)+1);badges([`Точек: ${cloud.points.length}`,`P95 до ребра: ${fmt(edgeStats.p95)}`,...Object.keys(stateLabels).map(s=>`${stateLabels[s]}: ${counts[s]||0}`),`Рёбер с однозначной поддержкой: ${ed.edges.filter(r=>r.unique_point_count>0).length}/${ed.edges.length}`]);$('note').textContent='Это постфактум диагностика на эталоне. В 3D цвет означает расстояние до ближайшего конечного ребра; классы справа учитывают тот же Z, альтернативные рёбра и некалиброванную оценку неопределённости из RMS. «Однозначное» означает выполнение диагностических правил, а не доказанную принадлежность. Порог расстояния — '+pct(ed.tolerances?.edge_tolerance_fraction??.001,3)+' диагонали, он не является требованием точности продукта. Коррелированные окна не считаются независимыми измерениями.';render3d();renderSection();renderTable()}else{badges([`Точек: ${summary.count}`,`Среднее: ${fmt(summary.mean)}`,`RMS: ${fmt(summary.rms)}`,`Медиана: ${fmt(summary.median)}`,`p95: ${fmt(summary.p95)}`,`p99: ${fmt(summary.p99)}`,`Максимум: ${fmt(summary.max)}`,`≤ 0,05: ${pct(summary.within_absolute['0.05'])}`,`Ближайших граней: ${summary.nearest_face_count}/${summary.reference_face_count}`]);$('note').textContent=`Расстояние до конечных полигонов эталона. Шкала по умолчанию насыщается на p99; точная ошибка доступна при наведении. Облако: окно ${summary.display_window}, порог ${summary.minima_threshold_pct}% диапазона RMS на Z. Оно отличается от production trusted cloud (${summary.trusted_cloud_mode}, ${summary.trusted_cloud_point_count} точек). Близость к поверхности не означает точное положение на ребре.`;render3d();renderSurface()}}
names.forEach(n=>addOption($('modelSelect'),n,labels[n]||n));$('modelSelect').value=model;$('title').textContent=`${labels[model]||model}: точки, рёбра и контуры`;
$('modeSelect').querySelector('[value="edges"]').disabled=!fragments.edges;$('modeSelect').querySelector('[value="projection"]').disabled=!fragments.projection;
if(fragments.edges){zLevels.forEach((z,i)=>{if(finite(z))addOption($('zSelect'),i,`${i} · Z=${fmt(z)} · ${(byZ.get(i)||[]).length} точек`)});$('zSlider').max=zLevels.length-1;updateZ()}
if(fragments.projection)fragments.projection.index.forEach((v,i)=>addOption($('viewSelect'),i,`${v.contour_index} · ${fmt(v.angle*180/Math.PI,2)}° · RMS=${fmt(rmseOf(v.finite_faces))}`));
const provenance=DATA.sources.provenance;const source=document.createElement('div');source.textContent='Облако: '+DATA.sources.reconstruction_json+' · Эталон: '+DATA.sources.initial_model;$('provenance').appendChild(source);if(provenance){const p=document.createElement('div');p.textContent=`SHA256 источника: ${provenance.reconstruction.sha256}. Контуров: ${provenance.contour_count}; половин: ${provenance.half_contour_count}. Совмещение: identity. Полный manifest, параметры и семантика метрик сохранены в диагностическом JSON.`;$('provenance').appendChild(p)}else{const p=document.createElement('div');p.textContent='Для этой модели сохранён прежний режим расстояния до поверхности. Диагностика этапа 1 пока выполнена только для round.';$('provenance').appendChild(p)}
$('modelSelect').onchange=e=>{location.hash=encodeURIComponent(e.target.value);location.reload()};$('modeSelect').onchange=()=>updateMode().catch(showLoadError);
$('colorLimit').onchange=render3d;$('pointSize').oninput=render3d;$('showModel').onchange=render3d;$('edgeSelect').onchange=e=>selectEdge(e.target.value);$('clearEdge').onclick=()=>selectEdge(-1);$('stateSelect').onchange=render3d;$('onlyEdge').onchange=render3d;$('onlyLevel').onchange=render3d;
function chooseZ(v){zi=Number(v);updateZ();renderSection();if($('onlyLevel').checked)render3d()}
$('zSelect').onchange=e=>chooseZ(e.target.value);$('zSlider').onchange=e=>chooseZ(e.target.value);document.querySelectorAll('[data-z]').forEach(b=>b.onclick=()=>chooseZ(closestLevel(Number(b.dataset.z))));$('bottomLevel').onclick=()=>chooseZ(Math.min(...byZ.keys()));$('tableFilter').onchange=renderTable;$('tableSort').onchange=renderTable;$('viewSelect').onchange=e=>{viewIndex=Number(e.target.value);renderProjection().catch(showLoadError)};$('worstView').onclick=()=>{viewIndex=pd.per_view.reduce((best,v,i)=>rmseOf(v.finite_faces)>rmseOf(pd.per_view[best].finite_faces)?i:best,0);$('viewSelect').value=viewIndex;renderProjection().catch(showLoadError)};
await updateMode();
}
startViewer().catch(showLoadError);
</script>
</body>
</html>
"""


def write_html(models: dict[str, dict[str, Any]], output_html: Path, plotly_bundle: Path) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    relative_plotly = Path(os.path.relpath(plotly_bundle.resolve(), output_html.parent.resolve())).as_posix()
    embedded = json.dumps(pack_models(models), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__PLOTLY_SRC__", relative_plotly).replace("__DATA__", embedded)
    html = html.replace("__RENDERER_SHA256__", file_fingerprint(Path(__file__))["sha256"])
    atomic_write_text(output_html, html)


def parse_models(raw: str | None, fallback: str) -> list[str]:
    values = [value.strip() for value in str(raw or fallback).split(",") if value.strip()]
    return list(dict.fromkeys(values))


def parse_model_json_overrides(values: list[str] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected --model-json model=path, got {value!r}")
        model_name, raw_path = value.split("=", 1)
        model_name = model_name.strip()
        if not model_name or not raw_path.strip():
            raise ValueError(f"Expected --model-json model=path, got {value!r}")
        result[model_name] = Path(raw_path.strip())
    return result


def default_reconstruction_json(model_name: str) -> Path:
    if model_name == "round":
        golden = Path("output/round_rms_w2_edge_tracks.json")
        if golden.exists():
            return golden
    return Path(f"output/{model_name}_rms_cross_model.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect saved minima cloud against finite faces/edges and observed contours; never runs reconstruction."
    )
    parser.add_argument("--model", default="round")
    parser.add_argument(
        "--models",
        help="Comma-separated model names. For multiple models one selector is embedded in the same HTML.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--reconstruction-json", type=Path)
    parser.add_argument(
        "--model-json",
        action="append",
        help="Optional model-specific source override in model=path form.",
    )
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--output-json", type=Path, help="One standalone diagnostic JSON containing all retained model payloads.")
    parser.add_argument("--merge-existing", action="store_true", help="Retain other models from the existing output HTML when updating selected models.")
    parser.add_argument("--from-json", type=Path, help="Render an existing diagnostic JSON without recomputing geometry.")
    parser.add_argument("--edge-diagnostics", action="store_true", help="Add milestone-1 edge, coverage and finite-face projection diagnostics for selected models.")
    parser.add_argument("--edge-tolerance-fraction", type=float, default=0.001, help="Diagnostic edge distance tolerance / reference bbox diagonal; not a product accuracy target.")
    parser.add_argument("--ambiguity-tolerance-fraction", type=float, default=0.00025)
    parser.add_argument("--projection-rays", type=int, default=360)
    parser.add_argument("--plotly-bundle", type=Path, default=Path("output/plotly-2.35.2.min.js"))
    parser.add_argument("--chunk-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = parse_models(args.models, str(args.model))
    if args.reconstruction_json is not None and len(model_names) != 1:
        raise ValueError("--reconstruction-json can only be used with one model; use --model-json for multiple models")
    overrides = parse_model_json_overrides(args.model_json)
    output_html = args.output_html or (
        Path("output/all_models_minima_cloud_distance_map.html")
        if len(model_names) > 1
        else Path(f"output/{model_names[0]}_minima_cloud_distance_map.html")
    )
    if not args.plotly_bundle.exists():
        raise FileNotFoundError(args.plotly_bundle)

    if args.from_json is not None:
        if args.edge_diagnostics or args.merge_existing or args.output_json is not None:
            raise ValueError("--from-json only renders HTML; do not combine with recomputation or JSON output")
        models = read_models(args.from_json)
        if output_html.exists():
            previous = read_models(output_html)
            if set(previous) - set(models):
                raise ValueError("Refusing to discard existing models while rendering JSON")
        write_html(models, output_html, args.plotly_bundle)
        print(f"Rendered {output_html}; models={len(models)}")
        return

    if args.edge_diagnostics and args.output_json is None:
        raise ValueError("--edge-diagnostics requires --output-json for a reproducible milestone artifact")
    if args.projection_rays < 8 or args.chunk_size < 1:
        raise ValueError("--projection-rays must be >=8 and --chunk-size must be positive")
    for name in ("edge_tolerance_fraction", "ambiguity_tolerance_fraction"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")

    previous = read_models(output_html) if output_html.exists() else {}
    if set(previous) - set(model_names) and not args.merge_existing:
        raise ValueError("Refusing to discard other model payloads; use --merge-existing or an isolated output path")
    models: dict[str, dict[str, Any]] = dict(previous) if args.merge_existing else {}
    if args.output_json is not None and args.output_json.exists():
        previous_json = read_models(args.output_json)
        if set(previous_json) - (set(models) | set(model_names)):
            raise ValueError("Refusing to discard other models from the diagnostic JSON")
    if args.output_json is not None and args.output_json.resolve() == output_html.resolve():
        raise ValueError("HTML and JSON output paths must differ")

    for model_name in model_names:
        reconstruction_json = overrides.get(model_name)
        if reconstruction_json is None and args.reconstruction_json is not None:
            reconstruction_json = args.reconstruction_json
        if reconstruction_json is None:
            reconstruction_json = default_reconstruction_json(model_name)
        initial_model_path = args.data_root / model_name / "InitialModel"
        for path in (reconstruction_json, initial_model_path):
            if not path.exists():
                raise FileNotFoundError(path)
        protected = {reconstruction_json.resolve(), reconstruction_json.with_suffix(".html").resolve(), initial_model_path.resolve()}
        for output in (output_html, args.output_json):
            if output is not None and output.resolve() in protected:
                raise ValueError(f"Refusing to overwrite a source reconstruction/reference artifact: {output}")
        payload = build_payload(
            model_name=model_name,
            reconstruction_json=reconstruction_json,
            initial_model_path=initial_model_path,
            chunk_size=int(args.chunk_size),
            edge_diagnostics=bool(args.edge_diagnostics),
            edge_tolerance_fraction=float(args.edge_tolerance_fraction),
            ambiguity_tolerance_fraction=float(args.ambiguity_tolerance_fraction),
            projection_rays=int(args.projection_rays),
        )
        models[model_name] = payload
        summary = payload["summary"]
        print(
            f"{model_name}: points={summary['count']}, mean={summary['mean']:.6g}, "
            f"p95={summary['p95']:.6g}, p99={summary['p99']:.6g}, max={summary['max']:.6g}"
        )

    # Serialize before either replacement; invalid values cannot leave a partial calculation behind.
    if args.output_json is not None:
        serialized = json.dumps({"schema_version": 1, "artifact_type": "polyreco_minima_diagnostics", "models": models}, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        atomic_write_text(args.output_json, serialized)
        print(f"Wrote {args.output_json} ({args.output_json.stat().st_size} bytes)")
    write_html(models, output_html, args.plotly_bundle)
    print(f"Wrote {output_html} ({output_html.stat().st_size} bytes); models={len(models)}")


if __name__ == "__main__":
    main()
