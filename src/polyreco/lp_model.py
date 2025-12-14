from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from highspy import Highs, kHighsInf

from .model_data import PolyModel, ContourSupport
from .contour import lift_q2_to_w3


@dataclass
class VarIndex:
    delta_d: np.ndarray       # (F,) indices
    abs_delta_d: np.ndarray   # (F,) indices
    vx: np.ndarray            # (V,) indices
    vy: np.ndarray            # (V,) indices
    vz: np.ndarray            # (V,) indices
    H: Dict[Tuple[int, int], int]
    eps_p: Dict[Tuple[int, int], int]
    eps_m: Dict[Tuple[int, int], int]


@dataclass
class SupportMeta:
    # Store transposed direction matrices for fast matmul:
    # WT_by_cid[cid] has shape (3, M) and dtype float16 (contiguous)
    WT_by_cid: Dict[int, np.ndarray]


# ----------------------------
# highspy 1.12.0 helpers
# ----------------------------

def add_var(highs: Highs, lb: float, ub: float, cost: float = 0.0) -> int:
    highs.addVar(lb, ub)
    idx = highs.getNumCol() - 1
    if cost != 0.0:
        highs.changeColCost(idx, float(cost))
    return idx


def add_row(highs: Highs, lb: float, ub: float, cols: List[int], vals: List[float]) -> None:
    try:
        highs.addRow(lb, ub, cols, vals)
    except TypeError:
        highs.addRow(lb, ub, len(cols), cols, vals)


def _add_abs_constraints(highs: Highs, x_var: int, abs_var: int) -> None:
    add_row(highs, 0.0, kHighsInf, [abs_var, x_var], [1.0, -1.0])
    add_row(highs, 0.0, kHighsInf, [abs_var, x_var], [1.0, 1.0])


def add_support_row_for_vertex_components(
    highs: Highs,
    vx: int, vy: int, vz: int,
    H_var: int,
    wx: float, wy: float, wz_: float,
) -> None:
    """
    Add: wx*vx + wy*vy + wz*vz - H <= 0
    """
    add_row(highs, -kHighsInf, 0.0, [vx, vy, vz, H_var], [float(wx), float(wy), float(wz_), -1.0])


def add_support_row_for_vertex(highs: Highs, vx: int, vy: int, vz: int, H_var: int, w: np.ndarray) -> None:
    add_support_row_for_vertex_components(highs, vx, vy, vz, H_var, w[0], w[1], w[2])


# ----------------------------
# Base model builder for Cutting Plane
# ----------------------------

def build_base_model_cutting_plane(
    poly: PolyModel,
    supports: List[ContourSupport],
    *,
    w_fit: float = 1.0,
    w_reg_face: float = 1.0,
    fix_one_vertex: bool = True,
) -> Tuple[Highs, VarIndex, SupportMeta]:
    """
    Build base LP for Variant B but WITHOUT the family of constraints w·x_v <= H.
    Those will be added iteratively in cutting-plane.

    Includes:
      - vertex-on-incident-face equalities
      - H/eps variables and L1 fit constraints

    meta stores WT (3,M) float16 contiguous for fast violation scanning.
    """
    highs = Highs()

    F = len(poly.faces)
    V = len(poly.vertices)

    # Face offset vars
    delta_d = np.empty(F, dtype=int)
    abs_delta_d = np.empty(F, dtype=int)
    for f in range(F):
        delta_d[f] = add_var(highs, -kHighsInf, kHighsInf, cost=0.0)
        abs_delta_d[f] = add_var(highs, 0.0, kHighsInf, cost=float(w_reg_face))
        _add_abs_constraints(highs, int(delta_d[f]), int(abs_delta_d[f]))

    # Vertex coordinate vars
    vx = np.empty(V, dtype=int)
    vy = np.empty(V, dtype=int)
    vz = np.empty(V, dtype=int)
    for v in range(V):
        vx[v] = add_var(highs, -kHighsInf, kHighsInf, cost=0.0)
        vy[v] = add_var(highs, -kHighsInf, kHighsInf, cost=0.0)
        vz[v] = add_var(highs, -kHighsInf, kHighsInf, cost=0.0)

    # Gauge fixing (optional)
    if fix_one_vertex and V > 0:
        x0, y0, z0 = map(float, poly.vertices[0].xyz0)
        add_row(highs, x0, x0, [int(vx[0])], [1.0])
        add_row(highs, y0, y0, [int(vy[0])], [1.0])
        add_row(highs, z0, z0, [int(vz[0])], [1.0])

    # Incidence constraints
    vid_to_index = {vert.vid: i for i, vert in enumerate(poly.vertices)}
    for f_idx, face in enumerate(poly.faces):
        a, b, c = map(float, face.normal)
        d0 = float(face.d0)
        dd = int(delta_d[f_idx])
        for vid in face.vertex_ids:
            v = vid_to_index[vid]
            cols = [int(vx[v]), int(vy[v]), int(vz[v]), dd]
            vals = [a, b, c, 1.0]
            rhs = -d0
            add_row(highs, rhs, rhs, cols, vals)

    # H/eps vars and L1 fit constraints
    H: Dict[Tuple[int, int], int] = {}
    eps_p: Dict[Tuple[int, int], int] = {}
    eps_m: Dict[Tuple[int, int], int] = {}

    WT_by_cid: Dict[int, np.ndarray] = {}

    for sup in supports:
        M = sup.q2.shape[0]

        # Lift and store as float16; keep WT contiguous for fast matmul.
        W = lift_q2_to_w3(sup.angle, sup.q2).astype(np.float16)     # (M,3) float16
        WT = np.ascontiguousarray(W.T)                              # (3,M) float16 contiguous
        WT_by_cid[sup.cid] = WT

        for j in range(M):
            key = (sup.cid, j)
            H_var = add_var(highs, -kHighsInf, kHighsInf, cost=0.0)
            ep = add_var(highs, 0.0, kHighsInf, cost=float(w_fit))
            em = add_var(highs, 0.0, kHighsInf, cost=float(w_fit))
            H[key] = H_var
            eps_p[key] = ep
            eps_m[key] = em

            hdat = float(sup.h_data[j])

            # H - hdat <= eps+  -> H - eps+ <= hdat
            add_row(highs, -kHighsInf, hdat, [int(H_var), int(ep)], [1.0, -1.0])

            # hdat - H <= eps-  -> -H - eps- <= -hdat
            add_row(highs, -kHighsInf, -hdat, [int(H_var), int(em)], [-1.0, -1.0])

    var_idx = VarIndex(
        delta_d=delta_d,
        abs_delta_d=abs_delta_d,
        vx=vx, vy=vy, vz=vz,
        H=H, eps_p=eps_p, eps_m=eps_m,
    )
    meta = SupportMeta(WT_by_cid=WT_by_cid)
    return highs, var_idx, meta
