from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    from matplotlib import pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install it with: .venv/bin/pip install matplotlib"
    ) from exc


@dataclass(frozen=True)
class PolyModel:
    vertices: np.ndarray  # (N, 3)
    faces: list[list[int]]


@dataclass(frozen=True)
class ShadowContour:
    index: int
    normal: np.ndarray  # (3,)
    angle: float
    points: np.ndarray  # (M, 3), in plane n·x = 0


def parse_initial_model(path: Path) -> PolyModel:
    raw = path.read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in raw if ln.strip()]

    counts = None
    for ln in lines:
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
            counts = tuple(map(int, parts))
            break
    if counts is None:
        raise ValueError(f"Cannot parse counts from {path}")
    num_vertices, num_facets, _ = counts

    try:
        v_start = next(i for i, ln in enumerate(lines) if ln.lower().startswith("# vertices")) + 1
    except StopIteration as exc:
        raise ValueError(f"Cannot find vertices section in {path}") from exc

    vertices = np.zeros((num_vertices, 3), dtype=float)
    i = v_start
    parsed_vertices = 0
    while i < len(lines) and parsed_vertices < num_vertices:
        ln = lines[i]
        i += 1
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) != 4:
            continue
        vid = int(parts[0])
        vertices[vid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        parsed_vertices += 1

    if parsed_vertices != num_vertices:
        raise ValueError(f"Parsed {parsed_vertices} vertices, expected {num_vertices}")

    try:
        f_start = next(i for i, ln in enumerate(lines) if ln.lower().startswith("# facets")) + 1
    except StopIteration as exc:
        raise ValueError(f"Cannot find facets section in {path}") from exc

    faces: list[list[int]] = []
    i = f_start
    while i < len(lines) and len(faces) < num_facets:
        ln = lines[i]
        i += 1
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 6:
            continue
        num_sides = int(parts[1])
        vids: list[int] = []
        while i < len(lines) and len(vids) < num_sides:
            next_ln = lines[i]
            i += 1
            if next_ln.startswith("#"):
                continue
            vids.extend(int(tok) for tok in next_ln.split())
        faces.append(vids[:num_sides])

    if len(faces) != num_facets:
        raise ValueError(f"Parsed {len(faces)} facets, expected {num_facets}")

    return PolyModel(vertices=vertices, faces=faces)


def parse_merged_contour(path: Path) -> ShadowContour:
    idx_match = re.search(r"(\d+)$", path.name)
    if idx_match is None:
        raise ValueError(f"Cannot parse contour index from {path.name}")
    contour_idx = int(idx_match.group(1))

    normal = None
    angle = None
    points: list[list[float]] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("# norm"):
            parts = ln.split()
            normal = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
            continue
        if ln.startswith("# angle"):
            angle = float(ln.split()[-1])
            continue
        if ln.startswith("#"):
            continue
        vals = ln.split()
        if len(vals) != 3:
            raise ValueError(f"Expected 3D point in {path}, got: {ln}")
        points.append([float(vals[0]), float(vals[1]), float(vals[2])])

    if normal is None or angle is None:
        raise ValueError(f"Missing '# norm' or '# angle' in {path}")
    pts = np.array(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError(f"Contour too short in {path}")

    n_norm = np.linalg.norm(normal)
    if n_norm == 0:
        raise ValueError(f"Zero normal in {path}")
    normal = normal / n_norm

    return ShadowContour(index=contour_idx, normal=normal, angle=angle, points=pts)


def set_axes_equal(ax: plt.Axes, points_sets: list[np.ndarray]) -> None:
    stacked = np.vstack(points_sets)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius == 0.0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def camera_from_normal(n: np.ndarray) -> tuple[float, float]:
    eye = -n.copy()
    eye[2] += 0.2  # small tilt so model volume is visible
    eye = eye / np.linalg.norm(eye)
    elev = math.degrees(math.asin(float(eye[2])))
    azim = math.degrees(math.atan2(float(eye[1]), float(eye[0])))
    return elev, azim


def build_viewer(
    model_path: Path,
    shadow_dir: Path,
    distance_scale: float,
    max_contours: int | None,
) -> None:
    model = parse_initial_model(model_path)
    files = sorted(shadow_dir.glob("merged-cont*"))
    if max_contours is not None:
        files = files[:max_contours]
    if not files:
        raise ValueError(f"No files matched {shadow_dir / 'merged-cont*'}")
    contours = [parse_merged_contour(p) for p in files]

    model_center = model.vertices.mean(axis=0)
    vertices = model.vertices - model_center
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    mesh_polys = [vertices[np.array(face, dtype=int)] for face in model.faces if len(face) >= 3]
    mesh = Poly3DCollection(
        mesh_polys,
        facecolor="#86b3d1",
        edgecolor="#355c7d",
        linewidths=0.25,
        alpha=0.28,
    )
    ax.add_collection3d(mesh)

    contour_line, = ax.plot([], [], [], color="#d62828", linewidth=2.2)
    projection_mode = {"kind": "persp"}

    model_radius = float(np.linalg.norm(vertices, axis=1).max())

    def contour_display_points(cont: ShadowContour) -> np.ndarray:
        base = cont.points - model_center
        support_max = float(np.max(vertices @ cont.normal))
        push = support_max + distance_scale * model_radius
        return base + cont.normal[None, :] * push

    def redraw(idx: int) -> None:
        cont = contours[idx]
        pts = contour_display_points(cont)
        closed = np.vstack([pts, pts[0]])
        contour_line.set_data(closed[:, 0], closed[:, 1])
        contour_line.set_3d_properties(closed[:, 2])

        elev, azim = camera_from_normal(cont.normal)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(
            "Model + "
            f"merged-cont{cont.index:03d} | angle={cont.angle:.6f} rad | "
            f"mode={projection_mode['kind']} | norm={cont.normal.round(4)}"
        )
        set_axes_equal(ax, [vertices, pts])
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.03])
    slider = Slider(
        ax=slider_ax,
        label="Projection index",
        valmin=0,
        valmax=len(contours) - 1,
        valinit=0,
        valstep=1,
    )

    def on_slider(val: float) -> None:
        redraw(int(val))

    slider.on_changed(on_slider)

    mode_ax = fig.add_axes([0.83, 0.03, 0.14, 0.11])
    mode_switch = RadioButtons(mode_ax, ("Perspective", "Parallel"), active=0)

    def on_mode(label: str) -> None:
        kind = "persp" if label == "Perspective" else "ortho"
        projection_mode["kind"] = kind
        ax.set_proj_type(kind)
        redraw(int(slider.val))

    mode_switch.on_clicked(on_mode)
    ax.set_proj_type("persp")
    redraw(0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.16)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive viewer: InitialModel in center + one merged contour projection with slider."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="round",
        # default="pear",
        # default="princess",
        # default="radiant",
        # default="cushion",
        help="Model folder name inside data/ (for example: round)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory with model folders",
    )
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=0.18,
        help="How far contour plane is moved from model (fraction of model radius)",
    )
    parser.add_argument(
        "--max-contours",
        type=int,
        default=None,
        help="Optional: use only first N merged-cont* files after sorting",
    )
    args = parser.parse_args()

    model_path = args.data_root / args.model_name / "InitialModel"
    shadow_dir = args.data_root / args.model_name / "shadow"

    build_viewer(
        model_path=model_path,
        shadow_dir=shadow_dir,
        distance_scale=args.distance_scale,
        max_contours=args.max_contours,
    )


if __name__ == "__main__":
    main()
