from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_reconstruction_quality import triangulate_polygon_faces
from generate_full_circle_split_cached_viewer import parse_initial_model


DEFAULT_MODELS = ("round", "princess", "radiant", "pear", "cushion")


def rounded_points(values: Any, digits: int = 7) -> list[list[float]]:
    points = np.asarray(values, dtype=float)
    if points.size == 0:
        return []
    return np.round(points, digits).tolist()


def fan_triangles(faces: list[list[int]]) -> list[list[int]]:
    triangles: list[list[int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        triangles.extend([[int(face[0]), int(face[index]), int(face[index + 1])] for index in range(1, len(face) - 1)])
    return triangles


def unique_edges(faces: list[list[int]]) -> list[list[int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for index, first in enumerate(face):
            second = face[(index + 1) % len(face)]
            edge = tuple(sorted((int(first), int(second))))
            if edge[0] != edge[1]:
                edges.add(edge)
    return [list(edge) for edge in sorted(edges)]


def mesh_measurements(mesh: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(mesh.get("vertices", []), dtype=float)
    triangles = np.asarray(mesh.get("triangles", []), dtype=int)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        return {"volume": None, "surface_area": None, "bbox_min": None, "bbox_max": None}
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    if triangles.ndim != 2 or triangles.shape[1:] != (3,) or len(triangles) == 0:
        return {
            "volume": None,
            "surface_area": None,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        }
    first = vertices[triangles[:, 0]]
    second = vertices[triangles[:, 1]]
    third = vertices[triangles[:, 2]]
    cross = np.cross(second - first, third - first)
    surface_area = 0.5 * float(np.sum(np.linalg.norm(cross, axis=1)))
    volume = abs(float(np.sum(np.einsum("ij,ij->i", first, np.cross(second, third)))) / 6.0)
    return {
        "volume": volume,
        "surface_area": surface_area,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


def bbox_iou(first: dict[str, Any], second: dict[str, Any]) -> float | None:
    first_min = first.get("bbox_min")
    first_max = first.get("bbox_max")
    second_min = second.get("bbox_min")
    second_max = second.get("bbox_max")
    if any(value is None for value in (first_min, first_max, second_min, second_max)):
        return None
    first_extent = np.maximum(np.asarray(first_max) - np.asarray(first_min), 0.0)
    second_extent = np.maximum(np.asarray(second_max) - np.asarray(second_min), 0.0)
    intersection_extent = np.maximum(
        np.minimum(np.asarray(first_max), np.asarray(second_max))
        - np.maximum(np.asarray(first_min), np.asarray(second_min)),
        0.0,
    )
    first_volume = float(np.prod(first_extent))
    second_volume = float(np.prod(second_extent))
    intersection = float(np.prod(intersection_extent))
    union = first_volume + second_volume - intersection
    return intersection / union if union > 0.0 else None


def enrich_viewer_model_metrics(
    model: dict[str, Any],
    *,
    selected: dict[str, Any] | None,
    quality: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = model.setdefault("metrics", {})
    reference = model.get("reference", {})
    reconstructed = model.get("reconstructed", {})
    reference_measurements = mesh_measurements(reference)
    reconstructed_measurements = mesh_measurements(reconstructed)

    reference_volume = reference_measurements.get("volume")
    reconstructed_volume = metrics.get("volume")
    if reconstructed_volume is None:
        reconstructed_volume = reconstructed_measurements.get("volume")
    reference_area = reference_measurements.get("surface_area")
    reconstructed_area = reconstructed_measurements.get("surface_area")

    metrics.update(
        {
            "reference_edges": len(reference.get("edges", [])),
            "reconstructed_edges": len(reconstructed.get("edges", [])),
            "reference_volume": reference_volume,
            "reconstructed_volume": reconstructed_volume,
            "relative_volume_error": (
                float(reconstructed_volume) / float(reference_volume) - 1.0
                if reference_volume not in (None, 0.0) and reconstructed_volume is not None
                else None
            ),
            "reference_surface_area": reference_area,
            "reconstructed_surface_area": reconstructed_area,
            "relative_surface_area_error": (
                float(reconstructed_area) / float(reference_area) - 1.0
                if reference_area not in (None, 0.0) and reconstructed_area is not None
                else None
            ),
        }
    )

    global_metrics = selected.get("global_surface_metrics", {}) if selected else {}
    for key in (
        "normalized_chamfer_mean",
        "normalized_chamfer_rms",
        "normalized_robust_hausdorff_p99",
        "f1_at_0_5_percent_bbox_diagonal",
        "axis_aligned_bbox_iou",
    ):
        value = quality.get(key) if quality and key in quality else global_metrics.get(key)
        if value is not None:
            metrics[key if key != "axis_aligned_bbox_iou" else "bbox_iou"] = value
    if metrics.get("bbox_iou") is None:
        metrics["bbox_iou"] = bbox_iou(reference_measurements, reconstructed_measurements)

    canonical = quality.get("canonical_absolute", {}) if quality else {}
    if canonical:
        metrics.update(
            {
                "canonical_unique": canonical.get("unique_finite_face_ids"),
                "canonical_count_recall": canonical.get("count_recall"),
                "canonical_area_recall": canonical.get("area_weighted_recall"),
            }
        )
    elif selected:
        metrics.update(
            {
                "canonical_unique": selected.get("canonical_unique"),
                "canonical_count_recall": selected.get("canonical_count_recall"),
                "canonical_area_recall": selected.get("area_weighted_recall"),
            }
        )

    if selected:
        topology = selected.get("topology", {})
        outside = selected.get("trusted_points_outside_halfspaces", {})
        metrics.update(
            {
                "topology_valid": selected.get("topology_valid"),
                "connected_components": topology.get("connected_components"),
                "trusted_outside_fraction": outside.get("cumulative_outside_fraction"),
                "trusted_max_per_z_outside_fraction": outside.get("cumulative_max_level_outside_fraction"),
                "lost_z_levels": selected.get("lost_z", outside.get("cumulative_lost_z_levels")),
            }
        )
    return model


def load_quality_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("cross_model_summary", {}).get("rows", []):
        if row.get("run_label") == "off":
            rows[str(row.get("model"))] = row
    return rows


def parse_model_json(values: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--model-json expects model=path, got {value!r}")
        model_name, raw_path = value.split("=", 1)
        model_name = model_name.strip()
        if not model_name:
            raise ValueError(f"--model-json has an empty model name: {value!r}")
        out[model_name] = Path(raw_path.strip())
    return out


def load_embedded_viewer_models(benchmark_json: Path) -> dict[str, Any] | None:
    if not benchmark_json.exists():
        return None
    payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
    selected = payload.get("cross_model_summary", {}).get("selected_artifacts", {})
    viewer_payload = selected.get("viewer_payload") if isinstance(selected, dict) else None
    models = viewer_payload.get("models") if isinstance(viewer_payload, dict) else None
    if not isinstance(models, dict) or not models:
        return None
    missing = [model for model in DEFAULT_MODELS if model not in models]
    if missing:
        raise ValueError(
            "Embedded viewer payload is missing models: " + ", ".join(missing)
        )
    selected_models = {
        str(row.get("model")): row
        for row in selected.get("models", [])
        if isinstance(row, dict) and row.get("model") is not None
    }
    quality_rows = {
        str(row.get("model")): row
        for row in payload.get("cross_model_summary", {}).get("rows", [])
        if isinstance(row, dict) and row.get("run_label") == "off"
    }
    return {
        model: enrich_viewer_model_metrics(
            models[model],
            selected=selected_models.get(model),
            quality=quality_rows.get(model),
        )
        for model in DEFAULT_MODELS
    }


def build_model_payload(model_name: str, source_path: Path, quality: dict[str, Any] | None) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    reference = parse_initial_model(Path("data") / model_name / "InitialModel")
    reference_vertices = np.asarray(reference.vertices, dtype=float)
    reference_faces = [[int(value) for value in face] for face in reference.faces]
    reference_triangles = triangulate_polygon_faces(reference_vertices, reference_faces).tolist()

    reconstructed = source["reconstructed"]
    reconstructed_vertices = rounded_points(reconstructed.get("vertices", []))
    reconstructed_faces = [[int(value) for value in face] for face in reconstructed.get("faces", [])]
    face_candidate_indices = [int(value) for value in reconstructed.get("face_candidate_indices", [])]
    active_candidate_indices = set(face_candidate_indices)

    planes = []
    for candidate_index, candidate in enumerate(source.get("face_candidates", [])):
        hull = rounded_points(candidate.get("hull", []))
        if len(hull) < 3:
            continue
        planes.append(
            {
                "candidate_index": candidate_index,
                "active": candidate_index in active_candidate_indices,
                "hull": hull,
                "source": candidate.get("candidate_origin", candidate.get("candidate_source", "candidate")),
                "window": candidate.get("window"),
                "plane_rms": candidate.get("plane_rms"),
            }
        )

    topology = reconstructed.get("topology", {})
    metrics: dict[str, Any] = {
        "reference_vertices": len(reference_vertices),
        "reference_faces": len(reference_faces),
        "reconstructed_vertices": len(reconstructed_vertices),
        "reconstructed_faces": len(reconstructed_faces),
        "candidate_planes": len(planes),
        "active_candidate_planes": len(active_candidate_indices),
        "volume": reconstructed.get("reliable_volume"),
        "euler": topology.get("euler"),
        "boundary_edges": topology.get("boundary_edges"),
        "non_manifold_edges": topology.get("non_manifold_edges"),
    }
    if quality:
        metrics.update(
            {
                "normalized_chamfer_mean": quality.get("normalized_chamfer_mean"),
                "normalized_robust_hausdorff_p99": quality.get("normalized_robust_hausdorff_p99"),
                "f1_at_0_5_percent_bbox_diagonal": quality.get("f1_at_0_5_percent_bbox_diagonal"),
                "bbox_iou": quality.get("axis_aligned_bbox_iou"),
            }
        )
    elif model_name == "round":
        reprojection = source.get("summary", {}).get("reprojection_summary", {}).get("production_current", {})
        metrics.update(
            {
                "contour_distance_p95": reprojection.get("symmetric_contour_distance_p95"),
                "support_error_p95": reprojection.get("support_abs_p95"),
            }
        )

    model = {
        "name": model_name,
        "source": str(source_path),
        "reference": {
            "vertices": rounded_points(reference_vertices),
            "faces": reference_faces,
            "triangles": reference_triangles,
            "edges": unique_edges(reference_faces),
        },
        "reconstructed": {
            "vertices": reconstructed_vertices,
            "faces": reconstructed_faces,
            "triangles": fan_triangles(reconstructed_faces),
            "edges": unique_edges(reconstructed_faces),
        },
        "planes": planes,
        "metrics": metrics,
    }
    return enrich_viewer_model_metrics(model, selected=None, quality=quality)


def generate_html(models: dict[str, Any], plotly_javascript: str) -> str:
    embedded = json.dumps(models, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PolyReco — исходные и восстановленные модели</title>
  <script>{plotly_javascript}</script>
  <style>
    :root {{ color-scheme: light; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #18202b; background: #eef2f6; }}
    header {{ position: sticky; top: 0; z-index: 5; padding: 12px 16px; background: rgba(255,255,255,.96); border-bottom: 1px solid #d8dee8; }}
    .toolbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }}
    .toolbar strong {{ font-size: 17px; }}
    label {{ display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }}
    select, input[type=range] {{ accent-color: #2563eb; }}
    select {{ min-width: 135px; padding: 6px 9px; border: 1px solid #bcc6d3; border-radius: 6px; background: white; }}
    .metric-list {{ margin-top: 9px; display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; }}
    .metric-list span {{
      position: relative;
      padding: 4px 7px;
      background: #f3f6fa;
      border: 1px solid #dbe2eb;
      border-radius: 5px;
      cursor: help;
      outline: none;
    }}
    .metric-list span::after {{
      content: attr(data-tooltip);
      position: absolute;
      top: calc(100% + 7px);
      left: 0;
      z-index: 20;
      width: max-content;
      max-width: min(520px, 88vw);
      padding: 10px 12px;
      color: #f8fafc;
      background: #172033;
      border-radius: 6px;
      box-shadow: 0 5px 16px rgba(15, 23, 42, .22);
      font-size: 12px;
      line-height: 1.5;
      white-space: normal;
      pointer-events: none;
      opacity: 0;
      visibility: hidden;
      transform: translateY(-3px);
      transition: opacity .12s ease, transform .12s ease, visibility .12s ease;
    }}
    .metric-list span:hover::after,
    .metric-list span:focus-visible::after {{
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
    }}
    .metric-list span:focus-visible {{ box-shadow: 0 0 0 2px rgba(37, 99, 235, .35); }}
    .metric-list span:nth-last-child(-n+2)::after {{ left: auto; right: 0; }}
    #metricHint {{ margin-top: 7px; color: #667085; font-size: 11px; }}
    #advancedMetrics {{ margin-top: 8px; }}
    #advancedMetrics summary {{
      width: max-content;
      color: #36516f;
      font-size: 12px;
      cursor: pointer;
      user-select: none;
    }}
    #advancedMetrics[open] summary {{ color: #1d4f91; }}
    #advancedMetrics .metric-list {{ margin-top: 7px; }}
    main {{ padding: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .panel {{ min-height: 430px; background: white; border: 1px solid #d8dee8; border-radius: 8px; overflow: hidden; }}
    .plot {{ width: 100%; height: 430px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} .panel, .plot {{ min-height: 390px; height: 390px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="toolbar">
      <strong>PolyReco: исходные и восстановленные модели</strong>
      <label>Модель <select id="modelSelect"></select></label>
      <label><input id="showActivePatches" type="checkbox"> локальные патчи поддержки</label>
      <label><input id="showInactivePatches" type="checkbox"> неактивные патчи поддержки</label>
      <label>непрозрачность активных граней <input id="planeOpacity" type="range" min="0.10" max="0.90" step="0.05" value="0.62"></label>
    </div>
    <div id="metrics" class="metric-list"></div>
    <div id="metricHint">Наведите курсор или переведите фокус на показатель — появится подробное объяснение.</div>
    <details id="advancedMetrics">
      <summary>Дополнительные метрики</summary>
      <div id="metricsAdvanced" class="metric-list"></div>
    </details>
  </header>
  <main>
    <div class="grid">
      <section class="panel"><div id="plotReference" class="plot"></div></section>
      <section class="panel"><div id="plotReconstructed" class="plot"></div></section>
      <section class="panel"><div id="plotPlanes" class="plot"></div></section>
      <section class="panel"><div id="plotOverlay" class="plot"></div></section>
    </div>
  </main>
  <script>
    const MODELS = {embedded};
    const MODEL_LABELS = {{round:'Round', princess:'Princess', radiant:'Radiant', pear:'Pear', cushion:'Cushion'}};
    const PLOTS = ['plotReference','plotReconstructed','plotPlanes','plotOverlay'];
    const modelSelect = document.getElementById('modelSelect');
    const showActivePatches = document.getElementById('showActivePatches');
    const showInactivePatches = document.getElementById('showInactivePatches');
    const planeOpacity = document.getElementById('planeOpacity');
    let sharedCamera = null;
    let syncing = false;

    Object.keys(MODELS).forEach(name => {{ const option=document.createElement('option'); option.value=name; option.textContent=MODEL_LABELS[name] || name; modelSelect.appendChild(option); }});

    function meshTrace(mesh, color, opacity, name) {{
      const t = mesh.triangles || [];
      return {{type:'mesh3d', x:mesh.vertices.map(p=>p[0]), y:mesh.vertices.map(p=>p[1]), z:mesh.vertices.map(p=>p[2]),
        i:t.map(v=>v[0]), j:t.map(v=>v[1]), k:t.map(v=>v[2]), color, opacity, flatshading:true,
        name, hovertemplate:name+'<extra></extra>', showscale:false}};
    }}

    function edgeTrace(mesh, color, width, name) {{
      const x=[],y=[],z=[];
      (mesh.edges||[]).forEach(edge => {{ const a=mesh.vertices[edge[0]], b=mesh.vertices[edge[1]]; x.push(a[0],b[0],null); y.push(a[1],b[1],null); z.push(a[2],b[2],null); }});
      return {{type:'scatter3d',mode:'lines',x,y,z,line:{{color,width}},name,hoverinfo:'skip',showlegend:false}};
    }}

    function planesTrace(planes, active, color, opacity, name) {{
      const x=[],y=[],z=[],i=[],j=[],k=[];
      planes.filter(p=>p.active===active).forEach(plane => {{
        const base=x.length; plane.hull.forEach(point=>{{x.push(point[0]);y.push(point[1]);z.push(point[2]);}});
        for(let n=1;n<plane.hull.length-1;n++){{i.push(base);j.push(base+n);k.push(base+n+1);}}
      }});
      return {{type:'mesh3d',x,y,z,i,j,k,color,opacity,flatshading:true,name,hoverinfo:'skip',showscale:false}};
    }}

    function layout(title) {{
      return {{title:{{text:title,font:{{size:15}}}},margin:{{l:0,r:0,t:42,b:0}},showlegend:false,
        paper_bgcolor:'#fff',plot_bgcolor:'#fff',scene:{{aspectmode:'data',camera:sharedCamera||undefined,
          xaxis:{{title:'X',showbackground:false}},yaxis:{{title:'Y',showbackground:false}},zaxis:{{title:'Z',showbackground:false}}}}}};
    }}

    function fmt(value, digits=5) {{ return value == null ? '—' : Number(value).toFixed(digits); }}
    function percent(value, digits=2) {{ return value == null ? '—' : `${{(Number(value)*100).toFixed(digits)}}%`; }}
    function signedPercent(value, digits=2) {{
      if(value == null) return '—';
      const scaled=Number(value)*100;
      return `${{scaled>0?'+':''}}${{scaled.toFixed(digits)}}%`;
    }}
    function fillMetricList(targetId, items) {{
      const target=document.getElementById(targetId);
      target.replaceChildren();
      items.filter(item=>item.show!==false).forEach(item=>{{
        const metric=document.createElement('span');
        metric.textContent=item.label;
        metric.dataset.tooltip=item.help;
        metric.tabIndex=0;
        target.appendChild(metric);
      }});
    }}
    function updateMetrics(model) {{
      const m=model.metrics;
      const mainItems=[
        {{label:`исходная сетка: ${{m.reference_vertices}}V / ${{m.reference_edges}}E / ${{m.reference_faces}}F`, help:'Количество элементов эталонной InitialModel: V — вершины, E — уникальные рёбра, F — конечные грани. Это описание сложности исходной сетки, а не оценка качества. Равное число элементов от восстановления не требуется.'}},
        {{label:`восстановленная: ${{m.reconstructed_vertices}}V / ${{m.reconstructed_edges}}E / ${{m.reconstructed_faces}}F`, help:'Количество элементов финальной замкнутой edge-clip сетки после отбора, repair и refinement: V — вершины, E — рёбра, F — активные clipped-грани. Больше граней не обязательно лучше: это может означать как найденные детали, так и дублирование или фрагментацию.'}},
        {{label:`патчи поддержки: ${{m.active_candidate_planes}} активных / ${{m.candidate_planes}} всего`, help:'Локальные патчи — это выпуклые оболочки наблюдаемых точек, по которым оценивались плоскости-кандидаты. Они намеренно могут быть маленькими: патч показывает только наблюдаемую опору, а не размер будущей грани. Финальная грань получается пересечением бесконечной плоскости кандидата со всеми остальными полупространствами. Активный кандидат — тот, чья плоскость действительно образовала грань финальной модели.'}},
        {{label:`объём: ${{fmt(m.reference_volume,4)}} → ${{fmt(m.reconstructed_volume,4)}} (${{signedPercent(m.relative_volume_error)}})`, help:'Слева объём исходной InitialModel, справа объём восстановленной модели; единицы — кубические единицы координат модели. Процент равен (Vвосст / Vисх − 1)×100: плюс означает завышение, минус — занижение, идеал — 0%. Объёмы вычислены по треугольникам замкнутых сеток через сумму ориентированных тетраэдров. Одинаковый объём сам по себе не гарантирует совпадение формы: локальные ошибки могут компенсироваться.'}},
        {{label:`топология: ${{m.topology_valid===false?'✗':'✓'}} χ=${{m.euler}}, CC=${{m.connected_components ?? '—'}}, bnd=${{m.boundary_edges}}, nm=${{m.non_manifold_edges}}`, help:'χ — характеристика Эйлера V−E+F; для одной замкнутой поверхности без ручек ожидается 2. CC — число связных компонент, ожидается 1. bnd — рёбра только с одной прилегающей гранью, nm — неманифолдные рёбра с некорректным числом прилегающих граней; для корректного замкнутого полиэдра оба значения должны быть 0. Валидная топология необходима, но не доказывает геометрическую точность.'}},
      ];
      if(m.normalized_chamfer_mean!=null) mainItems.push({{label:`средняя ошибка поверхности: ${{percent(m.normalized_chamfer_mean,3)}} bbox`, help:'Симметричный Chamfer mean: на обеих поверхностях берутся детерминированные выборки точек, для каждой точки ищется ближайшая точка другой поверхности, затем направленные средние объединяются. Значение делится на диагональ axis-aligned bounding box исходной модели и показано в процентах. Меньше — лучше; например 0,120% означает среднюю ошибку около 0,0012 диагонали. Метрика хорошо отражает общую близость, но может скрывать небольшие локальные выбросы.'}});
      if(m.normalized_robust_hausdorff_p99!=null) mainItems.push({{label:`почти максимальная ошибка p99: ${{percent(m.normalized_robust_hausdorff_p99,3)}} bbox`, help:'Robust Hausdorff p99: для каждого направления между sampled-поверхностями берётся 99-й перцентиль ближайших расстояний, затем выбирается худшее из двух направлений. Значение нормировано на диагональ bbox исходной модели. Меньше — лучше. Это устойчивый показатель почти максимальной ошибки: он игнорирует самый крайний 1% выбросов и поэтому не является точным абсолютным расстоянием Хаусдорфа.'}});
      if(m.f1_at_0_5_percent_bbox_diagonal!=null) mainItems.push({{label:`совпадение поверхности F1@0,5%: ${{percent(m.f1_at_0_5_percent_bbox_diagonal,1)}}`, help:'Точка поверхности считается совпавшей, если ближайшая точка другой поверхности находится не дальше 0,5% диагонали bbox исходной модели. Precision показывает, какая доля восстановленной поверхности имеет близкий эталон; recall — какая доля эталонной поверхности покрыта восстановлением. F1 — их гармоническое среднее. Больше — лучше, 100% — идеальное sampled-совпадение. Метрика не проверяет идентичность отдельных граней и топологию.'}});
      if(m.canonical_unique!=null) mainItems.push({{label:`эталонные грани: ${{m.canonical_unique}}/${{m.reference_faces}} (${{percent(m.canonical_count_recall,1)}})`, help:'Число уникальных конечных граней InitialModel, которым удалось сопоставить хотя бы одну активную грань reconstruction. Canonical-критерии: угол нормалей ≤2°, расстояние между плоскостями ≤0,05, расстояние центроидов ≤0,15 и медианное расстояние observed hull до поверхности ≤0,15; каждый ID эталонной грани считается один раз. Больше — лучше. Это posthoc/oracle-оценка по InitialModel, она не участвует в production-отборе. Метрика по количеству особенно чувствительна к пропуску множества мелких граней.'}});
      if(m.canonical_area_recall!=null) mainItems.push({{label:`покрытие площади эталонных граней: ${{percent(m.canonical_area_recall,1)}}`, help:'Доля суммарной площади InitialModel, приходящаяся на canonical-сопоставленные конечные грани. Используются те же строгие критерии соответствия, что и в счётчике эталонных граней, но каждая грань взвешивается своей площадью. Больше — лучше. Если эта метрика заметно выше recall по количеству, значит крупные грани восстановлены хорошо, а потери сосредоточены среди мелких или узких граней.'}});
      if(m.trusted_outside_fraction!=null) mainItems.push({{label:`наблюдаемые точки снаружи: ${{percent(m.trusted_outside_fraction,3)}}`, help:'Доля точек production trusted cloud stable_two_scales, оказавшихся снаружи хотя бы одного финального полупространства с учётом compatibility tolerance. Это проверка согласованности reconstruction с наблюдаемыми теневыми данными, а не сравнение с InitialModel. Меньше — лучше, идеал — 0%. Малое значение означает, что модель почти не отрезает надёжно наблюдаемые точки, но не гарантирует полноту всех граней.'}});

      const advancedItems=[
        {{label:`площадь поверхности: ${{fmt(m.reference_surface_area,4)}} → ${{fmt(m.reconstructed_surface_area,4)}} (${{signedPercent(m.relative_surface_area_error)}})`, help:'Слева суммарная площадь треугольников показанной InitialModel, справа — показанной восстановленной сетки; единицы — квадратные единицы координат. Процент равен (Aвосст / Aисх − 1)×100: идеал 0%. Площадь вычислена непосредственно по отображаемым треугольникам. Совпадение общей площади не гарантирует совпадение формы, потому что избыток площади в одном месте может компенсировать недостаток в другом.'}},
        {{label:`Chamfer RMS: ${{percent(m.normalized_chamfer_rms,3)}} bbox`, help:'Среднеквадратическое симметричное расстояние между sampled-поверхностями, нормированное на диагональ bbox исходной модели. В отличие от обычного среднего, квадрат сильнее штрафует редкие крупные отклонения. Меньше — лучше. Это выборочная метрика, поэтому она зависит от детерминированной схемы семплирования поверхности.', show:m.normalized_chamfer_rms!=null}},
        {{label:`overlap габаритов bbox: ${{percent(m.bbox_iou,2)}}`, help:'Intersection over Union двух axis-aligned bounding boxes: объём пересечения габаритных параллелепипедов делится на объём их объединения. Больше — лучше, 100% означает одинаковые осевые габариты. Метрика быстро выявляет общий сдвиг или неверный масштаб, но не видит локальные дефекты формы и зависит от ориентации координатных осей.', show:m.bbox_iou!=null}},
        {{label:`худший уровень Z снаружи: ${{percent(m.trusted_max_per_z_outside_fraction,2)}}`, help:'Trusted cloud группируется по горизонтальным уровням Z; на каждом уровне считается доля точек снаружи финальных полупространств, после чего показывается максимальная доля среди уровней. Меньше — лучше. Эта метрика ловит локальный плохой Z-срез, который мог бы потеряться в маленькой общей доле outside.', show:m.trusted_max_per_z_outside_fraction!=null}},
        {{label:`потерянные уровни Z: ${{m.lost_z_levels ?? '—'}}`, help:'Число Z-срезов production trusted cloud, которые compatibility-проверка считает полностью потерянными финальным набором полупространств. Меньше — лучше; ожидаемое значение — 0. Нулевое значение означает отсутствие полностью отброшенных наблюдаемых уровней, но отдельные точки уровня всё ещё могут быть снаружи.', show:m.lost_z_levels!=null}},
      ];
      fillMetricList('metrics', mainItems);
      fillMetricList('metricsAdvanced', advancedItems);
    }}

    async function render() {{
      const model=MODELS[modelSelect.value]; updateMetrics(model);
      const ref=[meshTrace(model.reference,'#c9d1dc',0.82,'Исходная модель'),edgeTrace(model.reference,'#202a36',2,'Рёбра исходной')];
      const rec=[meshTrace(model.reconstructed,'#2f80ed',0.64,'Восстановленная модель'),edgeTrace(model.reconstructed,'#174d91',2,'Рёбра восстановленной')];
      const planeTraces=[
        meshTrace(model.reconstructed,'#ef8a17',Number(planeOpacity.value),'Активные clipped-грани'),
        edgeTrace(model.reconstructed,'#9a4d00',2,'Рёбра активных граней'),
      ];
      if(showActivePatches.checked) planeTraces.push(planesTrace(model.planes,true,'#9b3fdb',0.82,'Локальные патчи поддержки'));
      if(showInactivePatches.checked) planeTraces.push(planesTrace(model.planes,false,'#8f9bab',0.24,'Неактивные патчи поддержки'));
      const overlay=[meshTrace(model.reference,'#aeb7c4',0.30,'Исходная'),edgeTrace(model.reference,'#111827',2,'Исходная wireframe'),meshTrace(model.reconstructed,'#2f80ed',0.45,'Восстановленная'),edgeTrace(model.reconstructed,'#1d5ca8',2,'Восстановленная wireframe')];
      const config={{responsive:true,displaylogo:false,scrollZoom:true}};
      await Promise.all([
        Plotly.react('plotReference',ref,layout(`${{MODEL_LABELS[model.name]}} — исходная модель`),config),
        Plotly.react('plotReconstructed',rec,layout(`${{MODEL_LABELS[model.name]}} — восстановленная модель`),config),
        Plotly.react('plotPlanes',planeTraces,layout(`${{MODEL_LABELS[model.name]}} — активные грани пересечения`),config),
        Plotly.react('plotOverlay',overlay,layout(`${{MODEL_LABELS[model.name]}} — наложение`),config),
      ]);
      bindCameraSync();
    }}

    function bindCameraSync() {{
      PLOTS.forEach(id => {{
        const el=document.getElementById(id);
        if(el.__cameraBound) return; el.__cameraBound=true;
        el.on('plotly_relayout', event => {{
          const camera=event['scene.camera']; if(!camera || syncing) return;
          sharedCamera=camera; syncing=true;
          Promise.all(PLOTS.filter(other=>other!==id).map(other=>Plotly.relayout(other,{{'scene.camera':camera}}))).finally(()=>{{syncing=false;}});
        }});
      }});
    }}

    modelSelect.addEventListener('change',()=>{{sharedCamera=null;void render();}});
    showActivePatches.addEventListener('change',()=>void render());
    showInactivePatches.addEventListener('change',()=>void render());
    planeOpacity.addEventListener('input',()=>void render());
    modelSelect.value='round'; void render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one viewer for all InitialModel and reconstructed meshes.")
    parser.add_argument("--output-html", type=Path, default=Path("output/all_models_reconstruction_planes.html"))
    parser.add_argument("--benchmark-json", type=Path, default=Path("output/rms_cross_model_angular_benchmark.json"))
    parser.add_argument(
        "--model-json",
        action="append",
        help="Explicit reconstruction artifact as model=path. Used only when the benchmark JSON has no embedded viewer payload.",
    )
    args = parser.parse_args()

    models = load_embedded_viewer_models(args.benchmark_json)
    if models is None:
        model_sources = parse_model_json(args.model_json)
        if not model_sources:
            model_sources = {"round": Path("output/round_rms_w2_edge_tracks.json")}
        missing = [str(path) for path in model_sources.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing reconstruction artifacts: " + ", ".join(missing))
        quality_rows = load_quality_rows(args.benchmark_json)
        models = {
            model_name: build_model_payload(model_name, source_path, quality_rows.get(model_name))
            for model_name, source_path in model_sources.items()
        }
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    plotly_path = Path("output/plotly-2.35.2.min.js")
    if not plotly_path.exists():
        raise FileNotFoundError(f"Plotly bundle not found: {plotly_path}")
    args.output_html.write_text(generate_html(models, plotly_path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Wrote {args.output_html} ({args.output_html.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
