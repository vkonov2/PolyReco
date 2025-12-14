from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple, Set

import numpy as np
from highspy import Highs

from .model_data import PolyModel, ContourSupport
from .lp_model import (
    build_base_model_cutting_plane,
    add_support_row_for_vertex_components,
)


@dataclass
class SolveResult:
    status: str
    obj: float | None
    delta_d: np.ndarray | None
    vertex_xyz: np.ndarray | None   # (V,3) float32
    outer_iters: int
    total_added_rows: int
    max_violation: float


def _get_solution_vector(highs: Highs) -> np.ndarray:
    sol = highs.getSolution()
    return np.asarray(sol.col_value, dtype=float)


def _is_optimal_model_status(status_obj) -> bool:
    s = str(status_obj).lower()
    return ("optimal" in s) and ("infeasible" not in s) and ("unbounded" not in s)


def _try_get_basis(highs: Highs):
    if hasattr(highs, "getBasis"):
        try:
            return highs.getBasis()
        except Exception:
            return None
    return None


def _try_set_basis(highs: Highs, basis) -> bool:
    if basis is None:
        return False
    if hasattr(highs, "setBasis"):
        try:
            highs.setBasis(basis)
            return True
        except Exception:
            return False
    return False


def _get_runtime_seconds(highs: Highs, info) -> float | None:
    if hasattr(highs, "getRunTime"):
        try:
            return float(highs.getRunTime())
        except Exception:
            pass
    if hasattr(info, "run_time"):
        try:
            return float(info.run_time)
        except Exception:
            pass
    return None


def _get_simplex_iters(info) -> int | None:
    for name in ("simplex_iteration_count", "simplex_iteration", "simplexIterations"):
        if hasattr(info, name):
            try:
                return int(getattr(info, name))
            except Exception:
                return None
    return None


def solve_cutting_plane(
    poly: PolyModel,
    supports: List[ContourSupport],
    *,
    tol: float = 1e-6,
    max_outer_iters: int = 50,
    max_added_per_iter: int = 50_000,
    log_to_console: bool = True,
    log_file: str | None = "highs.log",
    reuse_basis: bool = True,
    progress_every_contours: int = 10,
) -> SolveResult:
    highs, var_idx, meta = build_base_model_cutting_plane(poly, supports)

    # Logging
    try:
        highs.setOptionValue("log_to_console", bool(log_to_console))
    except Exception:
        pass
    if log_file is not None:
        try:
            highs.setOptionValue("log_file", str(log_file))
        except Exception:
            pass

    print("Base model:")
    print("Rows:", highs.getNumRow())
    print("Cols:", highs.getNumCol())
    print("NNZ :", highs.getNumNz())

    added: Set[Tuple[int, int, int]] = set()
    total_added = 0

    V = len(poly.vertices)
    vx = var_idx.vx.astype(int)
    vy = var_idx.vy.astype(int)
    vz = var_idx.vz.astype(int)

    last_basis = None
    cids = list(meta.WT_by_cid.keys())
    basis_reuse_enabled = reuse_basis

    for it in range(max_outer_iters):
        if basis_reuse_enabled and last_basis is not None:
            ok = _try_set_basis(highs, last_basis)
            if not ok:
                # If basis is rejected, disable reuse to avoid repeated HiGHS errors
                basis_reuse_enabled = False

        highs.run()

        status = highs.getModelStatus()
        if not _is_optimal_model_status(status):
            return SolveResult(
                status=str(status),
                obj=None,
                delta_d=None,
                vertex_xyz=None,
                outer_iters=it,
                total_added_rows=total_added,
                max_violation=float("nan"),
            )

        if basis_reuse_enabled:
            last_basis = _try_get_basis(highs)

        info = highs.getInfo()
        obj = float(info.objective_function_value)
        x = _get_solution_vector(highs)

        # X float32; WT float16 => matmul in float32
        X = np.empty((V, 3), dtype=np.float32)
        X[:, 0] = x[vx].astype(np.float32, copy=False)
        X[:, 1] = x[vy].astype(np.float32, copy=False)
        X[:, 2] = x[vz].astype(np.float32, copy=False)

        num_added = 0
        max_violation = 0.0

        print(f"[CP] iter={it} solved LP: status={status}, obj={obj:.6g}. Scanning violations...")
        t_scan0 = time.time()

        for ci, cid in enumerate(cids):
            WT = meta.WT_by_cid[cid]  # (3,M) float16 contiguous

            # S = X @ WT : (V,M) float32
            S = X @ WT
            vmax = S.max(axis=0)
            arg = S.argmax(axis=0).astype(int)

            M = WT.shape[1]
            for j in range(M):
                H_var = int(var_idx.H[(cid, j)])
                Hv = float(x[H_var])
                viol = float(vmax[j] - Hv)
                if viol > max_violation:
                    max_violation = viol

                if viol > tol:
                    v_star = int(arg[j])
                    key3 = (cid, j, v_star)
                    if key3 in added:
                        continue

                    # No allocations: take components directly
                    wx = WT[0, j]
                    wy = WT[1, j]
                    wz_ = WT[2, j]

                    add_support_row_for_vertex_components(
                        highs,
                        int(vx[v_star]), int(vy[v_star]), int(vz[v_star]),
                        H_var,
                        wx, wy, wz_,
                    )

                    added.add(key3)
                    num_added += 1
                    total_added += 1

                    if num_added >= max_added_per_iter:
                        break

            if num_added >= max_added_per_iter:
                dt = time.time() - t_scan0
                print(
                    f"[CP] iter={it} reached max_added_per_iter={max_added_per_iter} "
                    f"at contour {ci+1}/{len(cids)} after {dt:.1f}s"
                )
                break

            if progress_every_contours > 0 and ((ci + 1) % progress_every_contours == 0 or (ci + 1) == len(cids)):
                dt = time.time() - t_scan0
                print(
                    f"[CP] iter={it} scanned {ci+1}/{len(cids)} contours "
                    f"(added={num_added}, max_violation={max_violation:.3e}) "
                    f"in {dt:.1f}s"
                )

        simplex_iters = _get_simplex_iters(info)
        rt = _get_runtime_seconds(highs, info)
        rt_str = f"{rt:.2f}s" if rt is not None else "n/a"
        it_str = str(simplex_iters) if simplex_iters is not None else "n/a"

        print(
            f"[CP] iter={it} DONE: added={num_added} total_added={total_added} "
            f"max_violation={max_violation:.3e} "
            f"rows={highs.getNumRow()} nnz={highs.getNumNz()} "
            f"simplex_iters={it_str} runtime={rt_str} "
            f"basis_reuse={'on' if basis_reuse_enabled else 'off'}"
        )

        if num_added == 0:
            delta_d = x[var_idx.delta_d.astype(int)].copy()
            return SolveResult(
                status=str(status),
                obj=obj,
                delta_d=delta_d,
                vertex_xyz=X.copy(),
                outer_iters=it + 1,
                total_added_rows=total_added,
                max_violation=max_violation,
            )

    # max iters reached
    x = _get_solution_vector(highs)
    X = np.empty((V, 3), dtype=np.float32)
    X[:, 0] = x[vx].astype(np.float32, copy=False)
    X[:, 1] = x[vy].astype(np.float32, copy=False)
    X[:, 2] = x[vz].astype(np.float32, copy=False)
    delta_d = x[var_idx.delta_d.astype(int)].copy()

    info = highs.getInfo()
    return SolveResult(
        status="MaxIters",
        obj=float(info.objective_function_value),
        delta_d=delta_d,
        vertex_xyz=X.copy(),
        outer_iters=max_outer_iters,
        total_added_rows=total_added,
        max_violation=float("nan"),
    )
