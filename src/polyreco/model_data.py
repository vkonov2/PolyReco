# src/polyfit/model_data.py
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple

@dataclass(frozen=True)
class Vertex:
    vid: int
    xyz0: np.ndarray  # shape (3,)

@dataclass(frozen=True)
class Face:
    fid: int
    normal: np.ndarray  # shape (3,) = (a,b,c)
    d0: float
    vertex_ids: Tuple[int, ...]  # indices of vertices on face boundary (from file)

@dataclass(frozen=True)
class PolyModel:
    vertices: List[Vertex]
    faces: List[Face]

@dataclass(frozen=True)
class Contour2D:
    cid: int
    angle: float
    pts: np.ndarray  # shape (N,2) (u,v) in given projection basis (as stored)

@dataclass(frozen=True)
class ContourSupport:
    cid: int
    angle: float
    # per “constraint direction” j
    q2: np.ndarray      # shape (M,2) 2D outward normals in contour plane
    h_data: np.ndarray  # shape (M,)   support values: max_i q2[j]·pts[i]
    # (optional) map each constraint to an edge index (i -> i+1)
    edge_idx: np.ndarray  # shape (M,) indices in original pts
