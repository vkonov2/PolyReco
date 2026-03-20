from __future__ import annotations

import argparse
import os
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from matplotlib import pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider, TextBox
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install it with: .venv/bin/pip install matplotlib"
    ) from exc

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None


EPS = 1e-9
LOG = logging.getLogger("window-line-viewer")
_WINDOW_SOLVER_CONTEXT: dict[str, np.ndarray | int] | None = None


@dataclass(frozen=True)
class ShadowContour:
    index: int
    normal: np.ndarray  # (3,)
    points: np.ndarray  # (M,3)


@dataclass
class ComputationResult:
    z_levels: np.ndarray  # (L,)
    line_points: np.ndarray  # (N,L,2,3) -> [window_start, z_idx, side(0=left,1=right), xyz]
    distances: np.ndarray  # (N-1,L,2)


def progress_iter(iterable, *, total: int | None, desc: str, use_tqdm: bool):
    if use_tqdm and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)
    return iterable


def parse_merged_contour(path: Path) -> ShadowContour:
    idx_match = re.search(r"(\d+)$", path.name)
    if idx_match is None:
        raise ValueError(f"Cannot parse contour index from {path.name}")
    contour_idx = int(idx_match.group(1))

    normal = None
    points: list[list[float]] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("# norm"):
            parts = ln.split()
            normal = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
            continue
        if ln.startswith("#"):
            continue
        vals = ln.split()
        if len(vals) != 3:
            raise ValueError(f"Expected 3D point in {path}, got: {ln}")
        points.append([float(vals[0]), float(vals[1]), float(vals[2])])

    if normal is None:
        raise ValueError(f"Missing '# norm' in {path}")

    pts = np.array(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError(f"Contour too short in {path}")

    n_norm = np.linalg.norm(normal)
    if n_norm < EPS:
        raise ValueError(f"Zero normal in {path}")
    normal = normal / n_norm

    return ShadowContour(index=contour_idx, normal=normal, points=pts)


def sorted_contour_files(shadow_dir: Path, pattern: str, max_contours: int) -> list[Path]:
    files = list(shadow_dir.glob(pattern))
    if not files:
        raise ValueError(f"No files matched {shadow_dir / pattern}")

    def sort_key(p: Path) -> tuple[int, str]:
        m = re.search(r"(\d+)$", p.name)
        if m is None:
            return (10**9, p.name)
        return (int(m.group(1)), p.name)

    files = sorted(files, key=sort_key)
    return files[:max_contours]


def global_z_grid(contours: list[ShadowContour], z_step: float) -> np.ndarray:
    if z_step <= 0:
        raise ValueError("z_step must be > 0")
    all_points = np.vstack([c.points for c in contours])
    z_min = float(np.min(all_points[:, 2]))
    z_max = float(np.max(all_points[:, 2]))
    levels = np.arange(z_min, z_max + 0.5 * z_step, z_step, dtype=float)
    if levels.size < 2:
        raise ValueError("Z grid is too small; choose a smaller z_step")
    return levels


def lateral_axis_for_normal(direction: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    lateral = np.cross(z_axis, direction)
    ln = np.linalg.norm(lateral)
    if ln < EPS:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        lateral = np.cross(x_axis, direction)
        ln = np.linalg.norm(lateral)
        if ln < EPS:
            raise ValueError("Cannot build lateral axis for projection direction")
    return lateral / ln


def dedup_points(points: list[np.ndarray], tol: float = 1e-7) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for p in points:
        if not any(np.linalg.norm(p - q) <= tol for q in unique):
            unique.append(p)
    return unique


def contour_points_at_z(contour: ShadowContour, z_level: float, lateral_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = contour.points
    n = pts.shape[0]
    intersections: list[np.ndarray] = []

    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        z0 = p0[2]
        z1 = p1[2]

        dz0 = z0 - z_level
        dz1 = z1 - z_level

        if abs(dz0) <= EPS and abs(dz1) <= EPS:
            intersections.append(p0.copy())
            intersections.append(p1.copy())
            continue

        if abs(dz0) <= EPS:
            intersections.append(p0.copy())
            continue

        if abs(dz1) <= EPS:
            intersections.append(p1.copy())
            continue

        if dz0 * dz1 < 0.0:
            t = (z_level - z0) / (z1 - z0)
            intersections.append(p0 + t * (p1 - p0))

    intersections = dedup_points(intersections)
    if len(intersections) < 2:
        nan_point = np.array([np.nan, np.nan, np.nan], dtype=float)
        return nan_point, nan_point

    coords = np.array([float(p @ lateral_axis) for p in intersections], dtype=float)
    left = intersections[int(np.argmin(coords))]
    right = intersections[int(np.argmax(coords))]
    return left, right


def interpolate_contour_on_z_grid(contour: ShadowContour, z_levels: np.ndarray) -> np.ndarray:
    n_levels = z_levels.size
    out = np.full((n_levels, 2, 3), np.nan, dtype=float)
    lat = lateral_axis_for_normal(contour.normal)
    for zi, z in enumerate(z_levels):
        left, right = contour_points_at_z(contour, float(z), lat)
        out[zi, 0, :] = left
        out[zi, 1, :] = right
    return out


def init_window_solver_context(contour_lr: np.ndarray, normals: np.ndarray, window_size: int) -> None:
    global _WINDOW_SOLVER_CONTEXT
    _WINDOW_SOLVER_CONTEXT = {
        "contour_lr": contour_lr,
        "normals": normals,
        "window_size": int(window_size),
    }


def solve_single_window_points(ws: int) -> tuple[int, np.ndarray]:
    if _WINDOW_SOLVER_CONTEXT is None:
        raise RuntimeError("Window solver context is not initialized")

    contour_lr = _WINDOW_SOLVER_CONTEXT["contour_lr"]
    normals = _WINDOW_SOLVER_CONTEXT["normals"]
    window_size = int(_WINDOW_SOLVER_CONTEXT["window_size"])
    assert isinstance(contour_lr, np.ndarray)
    assert isinstance(normals, np.ndarray)

    n_cont, n_levels = contour_lr.shape[0], contour_lr.shape[1]
    out = np.full((n_levels, 2, 3), np.nan, dtype=float)
    idx_arr = np.array([(ws + j) % n_cont for j in range(window_size)], dtype=int)
    window_normals = normals[idx_arr, :]

    for zi in range(n_levels):
        for side in (0, 1):
            cand = contour_lr[idx_arr, zi, side, :]  # (window_size,3)
            valid = np.all(np.isfinite(cand), axis=1)
            if int(np.sum(valid)) < 2:
                continue
            p_arr = cand[valid, :]
            d_arr = window_normals[valid, :]
            out[zi, side, :] = closest_point_to_lines(p_arr, d_arr)

    return ws, out


def closest_point_to_lines(points: np.ndarray, directions: np.ndarray) -> np.ndarray:
    # Solve: sum ||(I - dd^T)(x - p)||^2 -> min
    a = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    eye = np.eye(3, dtype=float)

    for p, d in zip(points, directions):
        dn = np.linalg.norm(d)
        if dn < EPS:
            continue
        d = d / dn
        proj = eye - np.outer(d, d)
        a += proj
        b += proj @ p

    if np.linalg.norm(a) < EPS:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    try:
        x = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
    return x


def compute_result(
    contours: list[ShadowContour],
    z_step: float,
    window_size: int,
    *,
    show_progress: bool = False,
    workers: int | None = None,
) -> ComputationResult:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    n_cont = len(contours)
    if window_size > n_cont:
        raise ValueError("window_size must be <= number of contours")

    z_levels = global_z_grid(contours, z_step)
    n_levels = z_levels.size
    LOG.info(
        "Compute start: contours=%d, z_levels=%d, z_step=%.6f, window=%d, workers=%s",
        n_cont,
        n_levels,
        z_step,
        window_size,
        "auto" if workers is None else workers,
    )

    # Per contour / z-level left-right points.
    # shape (N,L,2,3)
    contour_lr = np.full((n_cont, n_levels, 2, 3), np.nan, dtype=float)

    resolved_workers = workers
    if resolved_workers is None:
        resolved_workers = max(1, (os.cpu_count() or 1) - 1)
    if resolved_workers < 1:
        resolved_workers = 1

    if resolved_workers == 1:
        contour_iter = progress_iter(
            enumerate(contours),
            total=n_cont,
            desc="Interpolate contours on Z grid",
            use_tqdm=show_progress,
        )
        for ci, contour in contour_iter:
            contour_lr[ci, :, :, :] = interpolate_contour_on_z_grid(contour, z_levels)
    else:
        LOG.info("Parallel interpolation with %d workers", resolved_workers)
        try:
            with ProcessPoolExecutor(max_workers=resolved_workers) as ex:
                future_to_idx = {
                    ex.submit(interpolate_contour_on_z_grid, contours[ci], z_levels): ci
                    for ci in range(n_cont)
                }
                completed = as_completed(future_to_idx)
                completed = progress_iter(
                    completed,
                    total=n_cont,
                    desc="Interpolate contours on Z grid",
                    use_tqdm=show_progress,
                )
                for fut in completed:
                    ci = future_to_idx[fut]
                    contour_lr[ci, :, :, :] = fut.result()
        except (PermissionError, OSError) as exc:
            LOG.warning(
                "Parallel interpolation unavailable (%s). Falling back to sequential mode.",
                exc,
            )
            contour_iter = progress_iter(
                enumerate(contours),
                total=n_cont,
                desc="Interpolate contours on Z grid (fallback)",
                use_tqdm=show_progress,
            )
            for ci, contour in contour_iter:
                contour_lr[ci, :, :, :] = interpolate_contour_on_z_grid(contour, z_levels)

    # For each sliding window start build a line (as points along z-levels) for left/right separately.
    # shape (N,L,2,3)
    line_points = np.full((n_cont, n_levels, 2, 3), np.nan, dtype=float)
    normals = np.vstack([c.normal for c in contours])

    if resolved_workers == 1:
        init_window_solver_context(contour_lr, normals, window_size)
        window_iter = progress_iter(
            range(n_cont),
            total=n_cont,
            desc="Solve sliding windows",
            use_tqdm=show_progress,
        )
        for ws in window_iter:
            _, ws_points = solve_single_window_points(ws)
            line_points[ws, :, :, :] = ws_points
    else:
        LOG.info("Parallel sliding-window solve with %d workers", resolved_workers)
        try:
            with ProcessPoolExecutor(
                max_workers=resolved_workers,
                initializer=init_window_solver_context,
                initargs=(contour_lr, normals, window_size),
            ) as ex:
                future_to_idx = {ex.submit(solve_single_window_points, ws): ws for ws in range(n_cont)}
                completed = as_completed(future_to_idx)
                completed = progress_iter(
                    completed,
                    total=n_cont,
                    desc="Solve sliding windows",
                    use_tqdm=show_progress,
                )
                for fut in completed:
                    ws, ws_points = fut.result()
                    line_points[ws, :, :, :] = ws_points
        except (PermissionError, OSError) as exc:
            LOG.warning(
                "Parallel sliding-window solve unavailable (%s). Falling back to sequential mode.",
                exc,
            )
            init_window_solver_context(contour_lr, normals, window_size)
            window_iter = progress_iter(
                range(n_cont),
                total=n_cont,
                desc="Solve sliding windows (fallback)",
                use_tqdm=show_progress,
            )
            for ws in window_iter:
                _, ws_points = solve_single_window_points(ws)
                line_points[ws, :, :, :] = ws_points

    # Distances between neighboring windows, non-cyclic: 0..N-2 (199 values for N=200)
    distances = np.full((n_cont - 1, n_levels, 2), np.nan, dtype=float)
    for i in range(n_cont - 1):
        dxyz = line_points[i + 1, :, :, :] - line_points[i, :, :, :]
        distances[i, :, :] = np.linalg.norm(dxyz, axis=2)

    finite_count = int(np.isfinite(distances).sum())
    LOG.info(
        "Compute done: distances shape=%s, finite=%d/%d",
        distances.shape,
        finite_count,
        distances.size,
    )

    return ComputationResult(z_levels=z_levels, line_points=line_points, distances=distances)


def build_viewer(
    contours: list[ShadowContour],
    z_step_default: float,
    window_default: int,
    workers: int | None,
) -> None:
    params = {
        "z_step": float(z_step_default),
        "window": int(window_default),
        "side": 0,  # 0 left, 1 right
        "z_idx": 0,
    }

    result = compute_result(
        contours,
        params["z_step"],
        params["window"],
        show_progress=True,
        workers=workers,
    )

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    (fn_line,) = ax.plot([], [], color="#1f77b4", linewidth=2.0)
    info_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "#cccccc"},
    )

    def update_plot() -> None:
        z_idx = int(np.clip(params["z_idx"], 0, result.z_levels.size - 1))
        params["z_idx"] = z_idx
        side = int(params["side"])

        y = result.distances[:, z_idx, side]
        x = np.arange(y.size, dtype=int)

        fn_line.set_data(x, y)

        finite = y[np.isfinite(y)]
        if finite.size:
            y_min = float(np.min(finite))
            y_max = float(np.max(finite))
            if abs(y_max - y_min) < 1e-12:
                y_min -= 0.5
                y_max += 0.5
            else:
                pad = 0.08 * (y_max - y_min)
                y_min -= pad
                y_max += pad
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(-1.0, 1.0)

        ax.set_xlim(0, max(1, y.size - 1))
        side_name = "left" if side == 0 else "right"
        z_val = float(result.z_levels[z_idx])
        ax.set_title(
            "Distance Function Between Neighbor Sliding Windows"
            f" | side={side_name} | z={z_val:.6f} | step={params['z_step']:.5f} | window={params['window']}"
        )
        ax.set_xlabel("Window transition index i (distance between windows i and i+1)")
        ax.set_ylabel("Distance")
        ax.grid(True, alpha=0.25)

        info_text.set_text(
            f"N_contours={len(contours)}\n"
            f"N_z_levels={result.z_levels.size}\n"
            f"z_min={result.z_levels[0]:.6f}, z_max={result.z_levels[-1]:.6f}"
        )

        fig.canvas.draw_idle()

    def recompute() -> None:
        nonlocal result
        LOG.info(
            "Recompute requested: z_step=%.6f, window=%d",
            params["z_step"],
            params["window"],
        )
        result = compute_result(
            contours,
            params["z_step"],
            params["window"],
            show_progress=True,
            workers=workers,
        )
        # keep previous z-value intent, clamp by new level count
        params["z_idx"] = int(np.clip(params["z_idx"], 0, result.z_levels.size - 1))

        # Replace z slider with updated bounds
        nonlocal z_slider
        z_ax.clear()
        z_slider = Slider(
            ax=z_ax,
            label="Z level index",
            valmin=0,
            valmax=max(1, result.z_levels.size - 1),
            valinit=params["z_idx"],
            valstep=1,
        )

        def on_z(val: float) -> None:
            params["z_idx"] = int(val)
            update_plot()

        z_slider.on_changed(on_z)
        update_plot()

    # Controls
    z_ax = fig.add_axes([0.15, 0.06, 0.62, 0.03])
    z_slider = Slider(
        ax=z_ax,
        label="Z level index",
        valmin=0,
        valmax=max(1, result.z_levels.size - 1),
        valinit=0,
        valstep=1,
    )

    step_ax = fig.add_axes([0.15, 0.02, 0.62, 0.03])
    step_slider = Slider(
        ax=step_ax,
        label="Z step",
        valmin=0.002,
        valmax=0.1,
        valinit=params["z_step"],
        valstep=0.001,
    )

    side_ax = fig.add_axes([0.80, 0.72, 0.17, 0.14])
    side_radio = RadioButtons(side_ax, ("Left", "Right"), active=0)
    win_box_ax = fig.add_axes([0.80, 0.64, 0.17, 0.05])
    win_box = TextBox(win_box_ax, "Window", initial=str(params["window"]))

    def on_z(val: float) -> None:
        params["z_idx"] = int(val)
        update_plot()

    def on_side(label: str) -> None:
        params["side"] = 0 if label == "Left" else 1
        update_plot()

    def on_step(val: float) -> None:
        step = float(val)
        if abs(step - params["z_step"]) < 1e-12:
            return
        params["z_step"] = step
        recompute()

    def on_window_submit(text: str) -> None:
        raw = text.strip()
        if not raw:
            win_box.set_val(str(params["window"]))
            return
        try:
            w = int(raw)
        except ValueError:
            LOG.warning("Window value must be integer, got '%s'", raw)
            win_box.set_val(str(params["window"]))
            return
        w_min = 2
        w_max = len(contours)
        if w < w_min or w > w_max:
            LOG.warning("Window value out of range [%d, %d]: %d", w_min, w_max, w)
            win_box.set_val(str(params["window"]))
            return
        if w == params["window"]:
            return
        params["window"] = w
        recompute()

    z_slider.on_changed(on_z)
    side_radio.on_clicked(on_side)
    step_slider.on_changed(on_step)
    win_box.on_submit(on_window_submit)

    fig.subplots_adjust(left=0.06, right=0.77, top=0.94, bottom=0.16)
    update_plot()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive viewer of distance functions built from sliding-window "
            "least-squares intersections of contour projection lines."
        )
    )
    parser.add_argument(
        "--shadow-dir",
        type=Path,
        default=Path("data/shadow"),
        help="Directory with merged contours",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="merged-cont*",
        help="Glob pattern for contour files (default: merged-cont*)",
    )
    parser.add_argument(
        "--max-contours",
        type=int,
        default=200,
        help="Use first N contours after numeric sorting",
    )
    parser.add_argument(
        "--z-step",
        type=float,
        default=0.01,
        help="Global Z grid step",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Sliding window size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Workers for contour interpolation: 0=auto, 1=sequential, >1=parallel",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOG.info(
        "Loading contours from %s with pattern=%s, max=%d",
        args.shadow_dir,
        args.pattern,
        args.max_contours,
    )
    files = sorted_contour_files(args.shadow_dir, args.pattern, args.max_contours)
    load_iter = progress_iter(
        files,
        total=len(files),
        desc="Load contour files",
        use_tqdm=True,
    )
    contours = [parse_merged_contour(p) for p in load_iter]
    if len(contours) < 2:
        raise ValueError("Need at least 2 contours")

    if args.window_size > len(contours):
        raise ValueError(
            f"window-size={args.window_size} > number of loaded contours={len(contours)}"
        )

    worker_count = None if args.workers == 0 else int(args.workers)
    LOG.info("Loaded %d contours, launching viewer", len(contours))
    build_viewer(
        contours,
        z_step_default=args.z_step,
        window_default=args.window_size,
        workers=worker_count,
    )


if __name__ == "__main__":
    main()
