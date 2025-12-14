# src/polyfit/io.py
from __future__ import annotations
import numpy as np
from typing import List
from .model_data import Vertex, Face, PolyModel, Contour2D

def load_initial_model(path: str) -> PolyModel:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # 1) header with counts (after "# num_vertices ..." line)
    # We'll find first line that looks like "508 256 762"
    counts = None
    for ln in lines:
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
            counts = tuple(map(int, parts))
            break
    if counts is None:
        raise ValueError("Cannot find counts line (num_vertices num_facets num_edges).")
    nv, nf, _ = counts

    # 2) parse vertices section
    # find '# vertices:' then read nv lines of 'id x y z'
    try:
        v_idx = lines.index("# vertices:") + 1
    except ValueError:
        # some files have "# vertices:" preceded by comments; try fuzzy
        v_idx = next(i for i,ln in enumerate(lines) if ln.lower().startswith("# vertices")) + 1

    vertices: List[Vertex] = []
    i = v_idx
    while len(vertices) < nv and i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            i += 1
            continue
        parts = ln.split()
        if len(parts) != 4:
            break
        vid = int(parts[0])
        xyz = np.array(list(map(float, parts[1:])), dtype=float)
        vertices.append(Vertex(vid=vid, xyz0=xyz))
        i += 1
    if len(vertices) != nv:
        raise ValueError(f"Parsed {len(vertices)} vertices, expected {nv}.")

    # 3) parse facets
    # Each facet: line "fid nsides a b c d" then next line: list of vertex indices
    try:
        f_idx = next(i for i,ln in enumerate(lines) if ln.lower().startswith("# facets")) + 1
    except StopIteration:
        raise ValueError("Cannot find '# facets:' section.")

    faces: List[Face] = []
    i = f_idx
    while len(faces) < nf and i < len(lines):
        if lines[i].startswith("#"):
            i += 1
            continue
        parts = lines[i].split()
        if len(parts) < 6:
            break
        fid = int(parts[0])
        nsides = int(parts[1])
        a, b, c, d0 = map(float, parts[2:6])

        # next non-comment line contains vertex indices
        j = i + 1
        while j < len(lines) and lines[j].startswith("#"):
            j += 1
        vparts = lines[j].split()
        if len(vparts) < nsides:
            raise ValueError(f"Facet {fid}: expected {nsides} vertex ids, got {len(vparts)}.")
        vids = tuple(map(int, vparts[:nsides]))

        faces.append(Face(
            fid=fid,
            normal=np.array([a,b,c], dtype=float),
            d0=float(d0),
            vertex_ids=vids
        ))
        i = j + 1

    if len(faces) != nf:
        raise ValueError(f"Parsed {len(faces)} faces, expected {nf}.")

    return PolyModel(vertices=vertices, faces=faces)


def load_contour(path: str, cid: int) -> Contour2D:
    angle = None
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("# angle"):
                angle = float(ln.split()[-1])
                continue
            if ln.startswith("#"):
                continue
            u, v = map(float, ln.split())
            pts.append((u, v))

    if angle is None:
        raise ValueError(f"{path}: missing '# angle' line.")
    pts_arr = np.array(pts, dtype=float)
    if pts_arr.ndim != 2 or pts_arr.shape[1] != 2 or pts_arr.shape[0] < 3:
        raise ValueError(f"{path}: expected Nx2 points with N>=3.")
    return Contour2D(cid=cid, angle=angle, pts=pts_arr)


def load_contours(prefix: str, first: int, last: int) -> List[Contour2D]:
    contours = []
    for cid in range(first, last + 1):
        path = f"{prefix}{cid:03d}"
        contours.append(load_contour(path, cid=cid))
    return contours
