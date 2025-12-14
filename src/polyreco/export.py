from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np

from .model_data import PolyModel
from .solve import SolveResult


def save_solution_json(path: str, res: SolveResult) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "status": res.status,
        "obj": res.obj,
        "outer_iters": res.outer_iters,
        "total_added_rows": res.total_added_rows,
        "max_violation": res.max_violation,
        "delta_d": None if res.delta_d is None else res.delta_d.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_vertices_xyz(path: str, vertex_xyz: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(vertex_xyz.shape[0]):
            x, y, z = vertex_xyz[i]
            f.write(f"{i} {x:.9f} {y:.9f} {z:.9f}\n")


def save_updated_model(path: str, poly: PolyModel, delta_d: np.ndarray) -> None:
    """
    Writes a file in the same spirit as InitialModel but with updated face 'd'.
    (Vertices are NOT written from res.vertex_xyz here; if you want that, say so.)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    nv = len(poly.vertices)
    nf = len(poly.faces)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# POLYHEDRON:\n")
        f.write("# num_vertices   num_facets   num_edges\n")
        f.write(f"{nv} {nf} 0\n")  # edges count optional/unknown here

        f.write("# vertices:\n")
        f.write("#   id   x y z\n")
        for v in poly.vertices:
            x, y, z = map(float, v.xyz0)
            f.write(f"{v.vid} {x:.9f} {y:.9f} {z:.9f}\n")

        f.write("# facets:\n")
        f.write("#   id   num_of_sides   plane_coeff  (ax + by + cz + d = 0)\n")
        f.write("#   indices_of_vertices\n")
        for idx, face in enumerate(poly.faces):
            a, b, c = map(float, face.normal)
            d_new = float(face.d0 + delta_d[idx])
            ns = len(face.vertex_ids)
            f.write(f"{face.fid} {ns} {a:.9f} {b:.9f} {c:.9f} {d_new:.9f}\n")
            f.write(" ".join(str(v) for v in face.vertex_ids) + "\n")


def export_all(out_dir: str, poly: PolyModel, res: SolveResult) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_solution_json(os.path.join(out_dir, "solution.json"), res)

    if res.vertex_xyz is not None:
        save_vertices_xyz(os.path.join(out_dir, "vertices.xyz"), res.vertex_xyz)

    if res.delta_d is not None:
        save_updated_model(os.path.join(out_dir, "UpdatedModel"), poly, res.delta_d)
