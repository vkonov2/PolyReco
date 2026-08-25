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
    return {model: models[model] for model in DEFAULT_MODELS}


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

    return {
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
    #metrics {{ margin-top: 9px; display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; }}
    #metrics span {{
      position: relative;
      padding: 4px 7px;
      background: #f3f6fa;
      border: 1px solid #dbe2eb;
      border-radius: 5px;
      cursor: help;
      outline: none;
    }}
    #metrics span::after {{
      content: attr(data-tooltip);
      position: absolute;
      top: calc(100% + 7px);
      left: 0;
      z-index: 20;
      width: max-content;
      max-width: min(360px, 85vw);
      padding: 8px 10px;
      color: #f8fafc;
      background: #172033;
      border-radius: 6px;
      box-shadow: 0 5px 16px rgba(15, 23, 42, .22);
      font-size: 12px;
      line-height: 1.4;
      white-space: normal;
      pointer-events: none;
      opacity: 0;
      visibility: hidden;
      transform: translateY(-3px);
      transition: opacity .12s ease, transform .12s ease, visibility .12s ease;
    }}
    #metrics span:hover::after,
    #metrics span:focus-visible::after {{
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
    }}
    #metrics span:focus-visible {{ box-shadow: 0 0 0 2px rgba(37, 99, 235, .35); }}
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
      <label><input id="activeOnly" type="checkbox" checked> только активные плоскости</label>
      <label><input id="showInactive" type="checkbox"> неактивные плоскости</label>
      <label>непрозрачность плоскостей <input id="planeOpacity" type="range" min="0.05" max="0.75" step="0.05" value="0.32"></label>
    </div>
    <div id="metrics"></div>
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
    const activeOnly = document.getElementById('activeOnly');
    const showInactive = document.getElementById('showInactive');
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
    function updateMetrics(model) {{
      const m=model.metrics;
      const items=[
        {{label:`исходная: ${{m.reference_vertices}}V / ${{m.reference_faces}}F`, help:'V — число вершин, F — число граней эталонной InitialModel. Это справочная сложность: больше или меньше само по себе не означает лучше.'}},
        {{label:`восстановленная: ${{m.reconstructed_vertices}}V / ${{m.reconstructed_faces}}F`, help:'V — число вершин, F — число граней финальной edge-clip сетки. Это сложность сетки, а не прямая оценка качества.'}},
        {{label:`плоскости: ${{m.active_candidate_planes}} активных / ${{m.candidate_planes}} всего`, help:'Активные плоскости образуют грани финального пересечения; «всего» — сохранённые кандидаты с конечным hull. Это диагностика сложности, а не оценка качества.'}},
        {{label:`объём: ${{fmt(m.volume,4)}}`, help:'Объём замкнутой восстановленной модели в кубических единицах её координат. Лучше не больше или меньше, а ближе к объёму эталона.'}},
        {{label:`Euler: ${{m.euler}}`, help:'Характеристика Эйлера V−E+F. Для одной замкнутой связной поверхности без отверстий ожидается 2.'}},
        {{label:`boundary/non-manifold: ${{m.boundary_edges}}/${{m.non_manifold_edges}}`, help:'Слева — рёбра только с одной гранью. Справа — рёбра, у которых число примыкающих граней не равно 2; оно включает boundary-рёбра. Для замкнутой manifold-модели лучше 0 / 0.'}},
      ];
      if(m.normalized_chamfer_mean!=null) items.push({{label:`Chamfer mean / bbox: ${{fmt(m.normalized_chamfer_mean,6)}}`, help:'Симметричное среднее расстояние между выборками точек двух поверхностей, делённое на диагональ bbox эталона. Меньше — лучше; 0,001 означает 0,1% диагонали bbox.'}});
      if(m.normalized_robust_hausdorff_p99!=null) items.push({{label:`robust Hausdorff p99 / bbox: ${{fmt(m.normalized_robust_hausdorff_p99,6)}}`, help:'Максимум из двух направленных 99-х перцентилей расстояний между выборками поверхностей, делённый на bbox. Меньше — лучше; это робастная выборочная метрика, а не точное расстояние Хаусдорфа.'}});
      if(m.f1_at_0_5_percent_bbox_diagonal!=null) items.push({{label:`F1@0.5% bbox: ${{fmt(m.f1_at_0_5_percent_bbox_diagonal,3)}}`, help:'Гармоническое среднее precision и recall: долей точек двух поверхностей, лежащих не дальше 0,5% диагонали bbox. Больше — лучше; 1 — идеальное совпадение.'}});
      if(m.contour_distance_p95!=null) items.push({{label:`контур p95: ${{fmt(m.contour_distance_p95,5)}}`, help:'95-й перцентиль симметричного расстояния между проекциями восстановленной модели и наблюдаемыми теневыми контурами. Меньше — лучше; сравнивать следует результаты одного evaluator.'}});
      const metrics=document.getElementById('metrics');
      metrics.replaceChildren();
      items.forEach(item=>{{
        const metric=document.createElement('span');
        metric.textContent=item.label;
        metric.dataset.tooltip=item.help;
        metric.title=item.help;
        metric.tabIndex=0;
        metrics.appendChild(metric);
      }});
    }}

    async function render() {{
      const model=MODELS[modelSelect.value]; updateMetrics(model);
      const ref=[meshTrace(model.reference,'#c9d1dc',0.82,'Исходная модель'),edgeTrace(model.reference,'#202a36',2,'Рёбра исходной')];
      const rec=[meshTrace(model.reconstructed,'#2f80ed',0.64,'Восстановленная модель'),edgeTrace(model.reconstructed,'#174d91',2,'Рёбра восстановленной')];
      const planeTraces=[meshTrace(model.reconstructed,'#2f80ed',0.20,'Восстановленная модель')];
      planeTraces.push(planesTrace(model.planes,true,'#ef8a17',Number(planeOpacity.value),'Активные плоскости'));
      if(!activeOnly.checked || showInactive.checked) planeTraces.push(planesTrace(model.planes,false,'#8f9bab',Math.min(.16,Number(planeOpacity.value)),'Неактивные плоскости'));
      const overlay=[meshTrace(model.reference,'#aeb7c4',0.30,'Исходная'),edgeTrace(model.reference,'#111827',2,'Исходная wireframe'),meshTrace(model.reconstructed,'#2f80ed',0.45,'Восстановленная'),edgeTrace(model.reconstructed,'#1d5ca8',2,'Восстановленная wireframe')];
      const config={{responsive:true,displaylogo:false,scrollZoom:true}};
      await Promise.all([
        Plotly.react('plotReference',ref,layout(`${{MODEL_LABELS[model.name]}} — исходная модель`),config),
        Plotly.react('plotReconstructed',rec,layout(`${{MODEL_LABELS[model.name]}} — восстановленная модель`),config),
        Plotly.react('plotPlanes',planeTraces,layout(`${{MODEL_LABELS[model.name]}} — найденные плоскости`),config),
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
    activeOnly.addEventListener('change',()=>void render());
    showInactive.addEventListener('change',()=>void render());
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
