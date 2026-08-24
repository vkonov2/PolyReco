from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from generate_full_circle_split_cached_viewer import triangulate_faces
from match_full_circle_points_to_edges import relative_plotly_src


SCHEMA_VERSION = 1


def as_json_float(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def as_json_point(p: np.ndarray) -> list[float | None]:
    return [as_json_float(float(v)) for v in p]


def build_payload(
    *,
    model_name: str,
    vertices: np.ndarray,
    faces: list[list[int]],
    z_levels: np.ndarray,
    fit_rms: np.ndarray,
    line_points: np.ndarray,
    tracks: list[object],
    diagnostics: list[dict[str, object]],
    candidates: list[dict[str, object]],
    reconstructed: dict[str, object],
    parameters: dict[str, object],
    minima_cloud: dict[str, object],
) -> dict[str, object]:
    finite = fit_rms[np.isfinite(fit_rms)]
    track_lengths = [len(t.observations) for t in tracks]
    peak_observations = int(sum(int(item.get("peaks", 0) or 0) for item in diagnostics))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "parameters": parameters,
        "summary": {
            "z_levels": int(z_levels.size),
            "n_half": int(fit_rms.shape[0]),
            "fit_rms_min": as_json_float(float(np.min(finite))) if finite.size else None,
            "fit_rms_max": as_json_float(float(np.max(finite))) if finite.size else None,
            "fit_rms_median": as_json_float(float(np.median(finite))) if finite.size else None,
            "peak_observations": peak_observations,
            "tracks": int(len(tracks)),
            "tracks_kept": int(len(candidates)),
            "reconstructed_vertices": len(reconstructed.get("vertices", [])),
            "reconstructed_faces": len(reconstructed.get("faces", [])),
            "track_levels_median": as_json_float(float(np.median(track_lengths))) if track_lengths else None,
            "reprojection_summary": (
                parameters.get("oracle_diagnostics", {})
                .get("reprojection_diagnostics", {})
                .get("models", {})
                if isinstance(parameters.get("oracle_diagnostics"), dict)
                else {}
            ),
        },
        "viewer": {
            "vertices": [as_json_point(p) for p in vertices],
            "triangles": triangulate_faces(faces).astype(int).tolist(),
            "faces": [[int(i) for i in face] for face in faces],
            "z_levels": [float(z) for z in z_levels],
            "rms_profile": [
                [as_json_float(float(v)) for v in fit_rms[:, zi]]
                for zi in range(int(z_levels.size))
            ],
            "line_points_sample": [
                as_json_point(line_points[i, zi, :])
                for zi in range(0, int(z_levels.size), max(1, int(z_levels.size) // 80))
                for i in range(0, int(fit_rms.shape[0]), max(1, int(fit_rms.shape[0]) // 80))
                if np.all(np.isfinite(line_points[i, zi, :]))
            ],
            "minima_cloud": minima_cloud,
            "diagnostics": diagnostics,
        },
        "face_candidates": candidates,
        "reconstructed": reconstructed,
    }


def build_viewer_html(payload: dict[str, object], output_html: Path) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    plotly_src = relative_plotly_src(output_html)
    default_plane_limit = max(1, len(payload.get("face_candidates", [])))
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RMS face-region reconstruction</title>
  <script src="{plotly_src}"></script>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: Arial, sans-serif; color: #192230; background: #f7f8fa; }}
    #bar {{ min-height: 64px; box-sizing: border-box; display: flex; align-items: center; flex-wrap: wrap; gap: 10px 14px; padding: 8px 12px; border-bottom: 1px solid #d8dee8; background: #ffffff; }}
    #wrap {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); width: 100vw; height: 58vh; min-height: 560px; }}
    #debugWrap {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr); width: 100vw; height: 38vh; min-height: 360px; border-top: 1px solid #d8dee8; }}
    .plot3d {{ width: 100%; min-width: 0; min-height: 0; }}
    .plotDebug {{ width: 100%; min-width: 0; min-height: 0; }}
    #plotPlanes {{ border-right: 1px solid #d8dee8; }}
    #plotCandidate3d {{ border-right: 1px solid #d8dee8; }}
    select, input {{ height: 30px; border: 1px solid #b8c1ce; border-radius: 4px; background: #fff; color: #192230; }}
    #candidateSelect {{ width: 260px; }}
    label {{ font-size: 12px; color: #526071; display: flex; align-items: center; gap: 6px; }}
    .title {{ font-weight: 700; }}
    .muted {{ color: #526071; font-size: 12px; }}
  </style>
</head>
<body>
  <div id="bar">
    <div class="title" id="title"></div>
    <label><input id="showPlanesInput" type="checkbox" checked> Найденные плоскости</label>
    <label><input id="showMinimaCloudInput" type="checkbox" checked> Minima cloud</label>
    <label><input id="showActiveFacesInput" type="checkbox"> Активные грани</label>
    <label><input id="syncCameraInput" type="checkbox"> Синхронная камера</label>
    <label>Показать плоскостей <input id="limitInput" type="number" min="1" max="400" step="1" value="{default_plane_limit}"></label>
    <label>Debug candidate <select id="candidateSelect"></select></label>
    <span class="muted" id="status"></span>
    <span class="muted" id="candidateStatus"></span>
  </div>
  <div id="wrap">
    <div id="plotPlanes" class="plot3d"></div>
    <div id="plotOverlay" class="plot3d"></div>
  </div>
  <div id="debugWrap">
    <div id="plotCandidate3d" class="plotDebug"></div>
    <div id="plotCandidateRms" class="plotDebug"></div>
  </div>
  <script>
    const DATA = {data_json};
    const vertices = DATA.viewer.vertices || [];
    const triangles = DATA.viewer.triangles || [];
    const candidates = DATA.face_candidates || [];
    const reconstructed = DATA.reconstructed || {{}};
    const limitInput = document.getElementById('limitInput');
    const showPlanesInput = document.getElementById('showPlanesInput');
    const showMinimaCloudInput = document.getElementById('showMinimaCloudInput');
    const showActiveFacesInput = document.getElementById('showActiveFacesInput');
    const syncCameraInput = document.getElementById('syncCameraInput');
    const candidateSelect = document.getElementById('candidateSelect');
    const status = document.getElementById('status');
    const candidateStatus = document.getElementById('candidateStatus');
    const syncedPlots = ['plotPlanes', 'plotOverlay', 'plotCandidate3d'];
    let syncingCamera = false;

    function colorFor(i, alpha = 1) {{
      const h = (i * 137.508) % 360;
      return `hsla(${{h.toFixed(1)}}, 72%, 45%, ${{alpha}})`;
    }}

    function fmt(v, d = 5) {{
      return Number.isFinite(Number(v)) ? Number(v).toFixed(d) : 'n/a';
    }}

    function pointText(row, side, obs) {{
      return `${{side}}<br>z_index=${{obs.z_index}} z=${{fmt(obs.z)}}<br>idx=${{row.index}} rms=${{fmt(row.rms, 6)}}`;
    }}

    function flattenPointRows(candidate, key) {{
      const rows = [];
      for (const obs of candidate.debug_observations || []) {{
        for (const row of obs[key] || []) {{
          if (Array.isArray(row.point) && row.point.length === 3) rows.push({{ row, obs }});
        }}
      }}
      return rows;
    }}

    function initialModelTraces(opacity = 0.18) {{
      if (!triangles.length) return [];
      return [{{
        type: 'mesh3d',
        x: vertices.map(p => p[0]),
        y: vertices.map(p => p[1]),
        z: vertices.map(p => p[2]),
        i: triangles.map(t => t[0]),
        j: triangles.map(t => t[1]),
        k: triangles.map(t => t[2]),
        color: '#8b99aa',
        opacity,
        flatshading: true,
        hoverinfo: 'skip',
        name: 'InitialModel',
        showscale: false,
      }}];
    }}

    function initialWireframeTraces(color = '#111827', width = 2) {{
      const faces = DATA.viewer.faces || [];
      const traces = [];
      for (let fi = 0; fi < faces.length; fi++) {{
        const face = faces[fi];
        if (!Array.isArray(face) || face.length < 2) continue;
        const closed = face.concat([face[0]]);
        traces.push({{
          type: 'scatter3d',
          mode: 'lines',
          x: closed.map(i => vertices[i]?.[0]),
          y: closed.map(i => vertices[i]?.[1]),
          z: closed.map(i => vertices[i]?.[2]),
          line: {{ color, width }},
          hoverinfo: 'skip',
          showlegend: false,
        }});
      }}
      return traces;
    }}

    function polygonTrace(candidate, idx) {{
      const hull = candidate.hull || [];
      if (hull.length < 3) return null;
      return {{
        type: 'mesh3d',
        x: hull.map(p => p[0]),
        y: hull.map(p => p[1]),
        z: hull.map(p => p[2]),
        i: Array.from({{length: hull.length - 2}}, (_, k) => 0),
        j: Array.from({{length: hull.length - 2}}, (_, k) => k + 1),
        k: Array.from({{length: hull.length - 2}}, (_, k) => k + 2),
        color: colorFor(idx, 0.55),
        opacity: 0.46,
        flatshading: true,
        name: `face candidate ${{candidate.track_id}}`,
        hoverinfo: 'skip',
        showscale: false,
        meta: {{ candidateIndex: idx }},
      }};
    }}

    function lineTrace(candidate, idx) {{
      const hull = candidate.hull || [];
      if (hull.length < 3) return null;
      const closed = hull.concat([hull[0]]);
      return {{
        type: 'scatter3d',
        mode: 'lines',
        x: closed.map(p => p[0]),
        y: closed.map(p => p[1]),
        z: closed.map(p => p[2]),
        line: {{ color: colorFor(idx, 0.95), width: 4 }},
        hoverinfo: 'skip',
        showlegend: false,
        meta: {{ candidateIndex: idx }},
      }};
    }}

    function reconstructedTraces(opacity = 0.34) {{
      const verts = reconstructed.vertices || [];
      const faces = reconstructed.faces || [];
      const traces = [];
      for (let fi = 0; fi < faces.length; fi++) {{
        const face = faces[fi];
        if (!Array.isArray(face) || face.length < 3) continue;
        traces.push({{
          type: 'mesh3d',
          x: face.map(i => verts[i]?.[0]),
          y: face.map(i => verts[i]?.[1]),
          z: face.map(i => verts[i]?.[2]),
          i: Array.from({{length: face.length - 2}}, () => 0),
          j: Array.from({{length: face.length - 2}}, (_, k) => k + 1),
          k: Array.from({{length: face.length - 2}}, (_, k) => k + 2),
          color: '#2f80ed',
          opacity,
          flatshading: true,
          name: 'halfspace intersection',
          text: `reconstructed face ${{fi}}<br>vertices=${{face.length}}`,
          hovertemplate: '%{{text}}<extra></extra>',
          showscale: false,
        }});
        const closed = face.concat([face[0]]);
        traces.push({{
          type: 'scatter3d',
          mode: 'lines',
          x: closed.map(i => verts[i]?.[0]),
          y: closed.map(i => verts[i]?.[1]),
          z: closed.map(i => verts[i]?.[2]),
          line: {{ color: '#1f5fbf', width: 3 }},
          hoverinfo: 'skip',
          showlegend: false,
        }});
      }}
      return traces;
    }}

    function candidateTraces(limit, selectedIdx = -1) {{
      const traces = [];
      candidates.slice(0, limit).forEach((candidate, idx) => {{
        if (idx === selectedIdx) return;
        const poly = polygonTrace(candidate, idx);
        if (poly) traces.push(poly);
      }});
      return traces;
    }}

    function minimaCloudTrace() {{
      const cloud = DATA.viewer.minima_cloud || {{}};
      const points = cloud.points || [];
      return {{
        type: 'scatter3d',
        mode: 'markers',
        x: points.map(p => p[0]),
        y: points.map(p => p[1]),
        z: points.map(p => p[2]),
        marker: {{ size: 2.4, color: '#2563eb', opacity: 0.62 }},
        name: 'minima cloud',
        hoverinfo: 'skip',
        showlegend: false,
      }};
    }}

    function selectedCandidate() {{
      const idx = Math.max(0, Number(candidateSelect.value) || 0);
      return [idx, candidates[idx] || null];
    }}

    function highlightedPolygonTrace(candidate, idx) {{
      const hull = candidate.hull || [];
      if (hull.length < 3) return null;
      return {{
        type: 'mesh3d',
        x: hull.map(p => p[0]),
        y: hull.map(p => p[1]),
        z: hull.map(p => p[2]),
        i: Array.from({{length: hull.length - 2}}, (_, k) => 0),
        j: Array.from({{length: hull.length - 2}}, (_, k) => k + 1),
        k: Array.from({{length: hull.length - 2}}, (_, k) => k + 2),
        color: '#f5c542',
        opacity: 0.82,
        flatshading: true,
        name: `SELECTED candidate ${{idx}}`,
        hoverinfo: 'skip',
        showscale: false,
        meta: {{ candidateIndex: idx }},
      }};
    }}

    function highlightedLineTrace(candidate, idx) {{
      const hull = candidate.hull || [];
      if (hull.length < 3) return null;
      const closed = hull.concat([hull[0]]);
      return {{
        type: 'scatter3d',
        mode: 'lines',
        x: closed.map(p => p[0]),
        y: closed.map(p => p[1]),
        z: closed.map(p => p[2]),
        line: {{ color: '#111827', width: 9 }},
        name: `selected outline ${{idx}}`,
        hoverinfo: 'skip',
        showlegend: false,
        meta: {{ candidateIndex: idx }},
      }};
    }}

    function selectedCandidateContextTraces(candidate, idx) {{
      if (!candidate) return [];
      const traces = [
        candidatePointTrace(candidate, 'left_fit_points', 'selected left fit', '#0b5cad', 5, false),
        candidatePointTrace(candidate, 'right_fit_points', 'selected right fit', '#b91c1c', 5, false),
        candidatePointTrace(candidate, 'peak_points', 'selected peak', '#15803d', 4, false),
      ];
      const poly = highlightedPolygonTrace(candidate, idx);
      const line = highlightedLineTrace(candidate, idx);
      if (poly) traces.push(poly);
      if (line) traces.push(line);
      return traces;
    }}

    function candidatePointTrace(candidate, key, name, color, size, enableHover = true) {{
      const rows = flattenPointRows(candidate, key);
      const trace = {{
        type: 'scatter3d',
        mode: 'markers',
        x: rows.map(item => item.row.point[0]),
        y: rows.map(item => item.row.point[1]),
        z: rows.map(item => item.row.point[2]),
        marker: {{ color, size }},
        name,
        meta: {{ candidateIndex: Number(candidateSelect.value) || 0 }},
      }};
      if (enableHover) {{
        trace.text = rows.map(item => pointText(item.row, name, item.obs));
        trace.hovertemplate = '%{{text}}<extra></extra>';
      }} else {{
        trace.hoverinfo = 'skip';
      }}
      return trace;
    }}

    function selectedCandidate3dTraces(candidate, idx) {{
      if (!candidate) return [];
      const traces = [
        candidatePointTrace(candidate, 'left_fit_points', 'left fit', '#1f77b4', 4),
        candidatePointTrace(candidate, 'right_fit_points', 'right fit', '#d62728', 4),
        candidatePointTrace(candidate, 'peak_points', 'peak samples', '#2ca02c', 3),
      ];
      const poly = polygonTrace(candidate, idx);
      const line = lineTrace(candidate, idx);
      if (poly) traces.push({{ ...poly, opacity: 0.28, name: 'fitted hull' }});
      if (line) traces.push(line);
      const c = candidate.plane_centroid || [];
      const n = candidate.plane_normal || [];
      if (c.length === 3 && n.length === 3) {{
        const scale = Math.max(Number(candidate.hull_diameter) || 1, 0.5) * 0.35;
        traces.push({{
          type: 'scatter3d',
          mode: 'lines',
          x: [c[0], c[0] + n[0] * scale],
          y: [c[1], c[1] + n[1] * scale],
          z: [c[2], c[2] + n[2] * scale],
          line: {{ color: '#111827', width: 5 }},
          name: 'normal',
          hoverinfo: 'skip',
        }});
      }}
      return traces;
    }}

    function intervalPointIndices(region) {{
      return (region && Array.isArray(region.indices)) ? region.indices : [];
    }}

    function selectedCandidateRmsTraces(candidate) {{
      if (!candidate) return [];
      const profile = DATA.viewer.rms_profile || [];
      const traces = [];
      for (const obs of candidate.debug_observations || []) {{
        const y = profile[obs.z_index] || [];
        const x = y.map((_, i) => i);
        traces.push({{
          type: 'scatter',
          mode: 'lines',
          x,
          y,
          line: {{ color: 'rgba(88, 96, 112, 0.32)', width: 1 }},
          name: `z ${{obs.z_index}}`,
          hovertemplate: `z_index=${{obs.z_index}} z=${{fmt(obs.z)}}<br>idx=%{{x}} rms=%{{y:.6f}}<extra></extra>`,
          showlegend: false,
        }});
        for (const [region, name, color] of [
          [obs.left_low, 'left low', '#1f77b4'],
          [obs.peak, 'peak', '#2ca02c'],
          [obs.right_low, 'right low', '#d62728'],
        ]) {{
          const idx = intervalPointIndices(region);
          if (!idx.length) continue;
          traces.push({{
            type: 'scatter',
            mode: 'markers',
            x: idx,
            y: idx.map(i => y[i]),
            marker: {{ color, size: name === 'peak' ? 7 : 6, symbol: name === 'peak' ? 'diamond' : 'circle' }},
            name,
            text: idx.map(i => `z_index=${{obs.z_index}} z=${{fmt(obs.z)}}<br>${{name}} idx=${{i}} rms=${{fmt(y[i], 6)}}`),
            hovertemplate: '%{{text}}<extra></extra>',
            showlegend: false,
          }});
        }}
      }}
      return traces;
    }}

    function sceneLayout(title) {{
      return {{
        margin: {{ l: 0, r: 0, b: 0, t: 28 }},
        title: {{ text: title, font: {{ size: 13 }} }},
        scene: {{
          xaxis: {{ title: 'X' }},
          yaxis: {{ title: 'Y' }},
          zaxis: {{ title: 'Z' }},
          aspectmode: 'data',
        }},
        showlegend: false,
      }};
    }}

    function xyLayout(title) {{
      return {{
        margin: {{ l: 48, r: 12, b: 42, t: 28 }},
        title: {{ text: title, font: {{ size: 13 }} }},
        xaxis: {{ title: 'cyclic index' }},
        yaxis: {{ title: 'fit_rms' }},
        showlegend: false,
      }};
    }}

    function fillCandidateSelect() {{
      candidateSelect.innerHTML = '';
      candidates.forEach((candidate, idx) => {{
        const option = document.createElement('option');
        option.value = String(idx);
        option.textContent = `${{idx}}: track ${{candidate.track_id}}, levels=${{candidate.levels}}, rms=${{fmt(candidate.plane_rms, 5)}}, out=${{candidate.support_outside_count ?? 'n/a'}}`;
        candidateSelect.appendChild(option);
      }});
    }}

    function renderDebug() {{
      const [idx, candidate] = selectedCandidate();
      if (!candidate) {{
        candidateStatus.textContent = 'candidate debug: n/a';
        Plotly.react('plotCandidate3d', [], sceneLayout('Selected candidate'), {{ responsive: true, displaylogo: false }});
        Plotly.react('plotCandidateRms', [], xyLayout('fit_rms lifecycle'), {{ responsive: true, displaylogo: false }});
        return;
      }}
      const zList = (candidate.z_indices || []).join(', ');
      candidateStatus.textContent =
        `track=${{candidate.track_id}}, levels=${{candidate.levels}}, z=${{fmt(candidate.z_min)}}..${{fmt(candidate.z_max)}}, ` +
        `center=${{fmt(candidate.center_min, 1)}}..${{fmt(candidate.center_max, 1)}}, ` +
        `fit L/R=${{candidate.left_points}}/${{candidate.right_points}}, peak=${{candidate.peak_points}}, ` +
        `rms=${{fmt(candidate.plane_rms, 6)}}, max_abs=${{fmt(candidate.plane_max_abs, 6)}}, out=${{candidate.support_outside_count ?? 'n/a'}}, z_indices=[${{zList}}]`;
      Plotly.react(
        'plotCandidate3d',
        selectedCandidate3dTraces(candidate, idx),
        sceneLayout(`Candidate ${{idx}} / track ${{candidate.track_id}}: fit points, peak samples, fitted hull`),
        {{ responsive: true, displaylogo: false }}
      );
      Plotly.react(
        'plotCandidateRms',
        selectedCandidateRmsTraces(candidate),
        xyLayout(`fit_rms lifecycle: peak/low regions for candidate ${{idx}}`),
        {{ responsive: true, displaylogo: false }}
      );
    }}

    async function render() {{
      const limit = Math.max(1, Number(limitInput.value) || 1);
      const showPlanes = Boolean(showPlanesInput.checked);
      const showMinimaCloud = Boolean(showMinimaCloudInput.checked);
      const showActiveFaces = Boolean(showActiveFacesInput.checked);
      const [selectedIdx, selected] = selectedCandidate();
      const planesTraces = (showMinimaCloud ? [minimaCloudTrace()] : [])
        .concat(showPlanes ? candidateTraces(limit, selectedIdx) : [])
        .concat(showActiveFaces ? reconstructedTraces(0.16) : [])
        .concat(selectedCandidateContextTraces(selected, selectedIdx));
      const overlayTraces = initialModelTraces(0.42)
        .concat(initialWireframeTraces('#111827', 3))
        .concat([minimaCloudTrace()])
        .concat(reconstructedTraces(0.34));
      const rv = (reconstructed.vertices || []).length;
      const rf = (reconstructed.faces || []).length;
      const cloudCount = (DATA.viewer.minima_cloud?.points || []).length;
      const reproj = DATA.summary?.reprojection_summary?.production_max100 || DATA.summary?.reprojection_summary?.current || null;
      const reprojText = reproj ? `, reproj p95=${{fmt(reproj.support_abs_p95, 4)}}` : '';
      status.textContent = `planes=${{showPlanes ? Math.min(limit, candidates.length) : 0}}/${{candidates.length}}, minima=${{showMinimaCloud ? cloudCount : 0}}, intersection=${{rv}}v/${{rf}}f${{reprojText}}`;
      Plotly.react('plotPlanes', planesTraces, sceneLayout('Найденные плоскости + minima cloud'), {{ responsive: true, displaylogo: false }});
      Plotly.react('plotOverlay', overlayTraces, sceneLayout('Наложение'), {{ responsive: true, displaylogo: false }});
      renderDebug();
      window.setTimeout(bindPlaneClick, 250);
    }}

    function bindCameraSync() {{
      for (const id of syncedPlots) {{
        const el = document.getElementById(id);
        if (!el || el._cameraSyncBound) continue;
        el._cameraSyncBound = true;
        el.addEventListener('plotly_relayout', ev => {{
          const detail = ev && ev.detail ? ev.detail : {{}};
          const camera = detail['scene.camera'];
          if (!syncCameraInput.checked || !camera || syncingCamera) return;
          syncingCamera = true;
          Promise.all(
            syncedPlots
              .filter(otherId => otherId !== id)
              .map(otherId => Plotly.relayout(otherId, {{ 'scene.camera': camera }}))
          ).finally(() => {{
            syncingCamera = false;
          }});
        }});
      }}
    }}

    function bindPlaneClick() {{
      const el = document.getElementById('plotPlanes');
      if (!el || el._planeClickBound) return;
      el._planeClickBound = true;
      el.on('plotly_click', ev => {{
        const point = ev?.points?.[0];
        const idx = point?.data?.meta?.candidateIndex;
        if (!Number.isInteger(idx) || idx < 0 || idx >= candidates.length) return;
        candidateSelect.value = String(idx);
        void render();
      }});
    }}

    document.getElementById('title').textContent = `${{DATA.model}}: RMS-регионы -> кандидаты граней`;
    fillCandidateSelect();
    limitInput.addEventListener('change', () => void render());
    showPlanesInput.addEventListener('change', () => void render());
    showMinimaCloudInput.addEventListener('change', () => void render());
    showActiveFacesInput.addEventListener('change', () => void render());
    syncCameraInput.addEventListener('change', () => {{
      bindCameraSync();
    }});
    candidateSelect.addEventListener('change', () => void render());
    void render();
  </script>
</body>
</html>
"""

