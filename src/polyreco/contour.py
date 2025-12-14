# src/polyfit/contour.py
from __future__ import annotations
import numpy as np
from .model_data import Contour2D, ContourSupport

def signed_area(pts: np.ndarray) -> float:
    # polygon area with wrap-around
    x = pts[:,0]; y = pts[:,1]
    x2 = np.roll(x, -1); y2 = np.roll(y, -1)
    return 0.5 * float(np.sum(x * y2 - y * x2))

def ensure_ccw(pts: np.ndarray) -> np.ndarray:
    # If CW, reverse. Keep all points.
    if signed_area(pts) < 0:
        return pts[::-1].copy()
    return pts

def edge_outward_normals_all_edges(pts_ccw: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    For each edge i -> i+1 (mod N), compute an outward normal q2 (2D).
    For CCW polygon, outward normal for edge tangent t = p_{i+1}-p_i is (t_y, -t_x).
    Returns:
        q2: (N,2) normals (not necessarily unit)
        edge_idx: (N,) i
    """
    N = pts_ccw.shape[0]
    p = pts_ccw
    p_next = np.roll(p, -1, axis=0)
    t = p_next - p  # (N,2)
    # outward normals for CCW:
    q = np.stack([t[:,1], -t[:,0]], axis=1)

    # handle degenerate tiny edges (duplicate points)
    norm = np.linalg.norm(q, axis=1)
    good = norm > eps
    if not np.all(good):
        # keep them (no dropping), but mark zero normals -> set to something tiny (won't help constraints)
        q[~good] = 0.0
    edge_idx = np.arange(N, dtype=int)
    return q, edge_idx

def support_values(pts: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # h(q) = max_i q·p_i
    return (pts @ q2.T).max(axis=0)

def contour_to_support(cont: Contour2D) -> ContourSupport:
    pts = ensure_ccw(cont.pts)
    q2, edge_idx = edge_outward_normals_all_edges(pts)
    h = support_values(pts, q2)
    return ContourSupport(
        cid=cont.cid,
        angle=cont.angle,
        q2=q2,
        h_data=h,
        edge_idx=edge_idx
    )

def basis_from_angle(angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given angle = angle(n, X-axis) in XY plane:
      n = (cos a, sin a, 0)
      e1 = (-sin a, cos a, 0)  (in-plane axis)
      e2 = (0,0,1)
    """
    ca = np.cos(angle); sa = np.sin(angle)
    n = np.array([ca, sa, 0.0], dtype=float)
    e1 = np.array([-sa, ca, 0.0], dtype=float)
    e2 = np.array([0.0, 0.0, 1.0], dtype=float)
    return n, e1, e2

def lift_q2_to_w3(angle: float, q2: np.ndarray) -> np.ndarray:
    """
    w = q_u * e1 + q_v * e2
    q2 has components in (u,v) basis stored in contour plane.
    """
    _, e1, e2 = basis_from_angle(angle)
    return q2[:,0:1] * e1[None,:] + q2[:,1:2] * e2[None,:]
