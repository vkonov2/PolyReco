from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_full_circle_split_cached_viewer import (
    parse_initial_model,
    parse_merged_contour,
    sorted_contour_files,
)
from polyreco.rms_reprojection import evaluate_polyhedron_polar_reprojection


SCHEMA_VERSION = 1


def parse_ray_counts(raw: str, primary: int) -> list[int]:
    values = {max(8, int(primary))}
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.add(max(8, int(token)))
    return sorted(values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_summary(result: dict[str, object]) -> dict[str, object]:
    excluded = {"per_view", "worst_views", "timing_seconds"}
    return {key: value for key, value in result.items() if key not in excluded}


def comparison_summary(
    reconstructed: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    rec_sse = float(reconstructed["squared_error_sum"])
    ref_sse = float(reference["squared_error_sum"])
    rec_rmse = float(reconstructed["root_mean_squared_error"])
    ref_rmse = float(reference["root_mean_squared_error"])
    return {
        "reconstructed_to_reference_sse_ratio": rec_sse / max(ref_sse, 1e-12),
        "reconstructed_to_reference_rmse_ratio": rec_rmse / max(ref_rmse, 1e-12),
        "excess_squared_error_sum": rec_sse - ref_sse,
        "excess_root_mean_squared_error": rec_rmse - ref_rmse,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare observed shadow contours with orthographic projections of a reconstructed "
            "polyhedron on a common polar ray grid."
        )
    )
    parser.add_argument("--model", default="round")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--reconstruction-json",
        type=Path,
        default=Path("output/round_rms_w2_edge_tracks.json"),
    )
    parser.add_argument("--rays", type=int, default=720)
    parser.add_argument(
        "--sensitivity-rays",
        default="180,360,720,1440",
        help="Comma-separated polar-grid sizes included in the stability check.",
    )
    parser.add_argument("--max-contours", type=int, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/round_polar_contour_reprojection.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reconstruction_path = args.reconstruction_json.resolve()
    payload = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    artifact_model = str(payload.get("model") or "")
    if artifact_model and artifact_model != args.model:
        raise ValueError(
            f"Reconstruction artifact model is {artifact_model!r}, requested {args.model!r}"
        )

    reconstructed = payload.get("reconstructed") or {}
    reconstructed_vertices = np.asarray(reconstructed.get("vertices", []), dtype=float)
    if (
        reconstructed_vertices.ndim != 2
        or reconstructed_vertices.shape[0] == 0
        or reconstructed_vertices.shape[1] != 3
    ):
        raise ValueError("reconstructed.vertices must be a non-empty Nx3 array")

    inside_point = np.asarray((payload.get("parameters") or {}).get("inside_point", []), dtype=float)
    if inside_point.shape != (3,) or not np.all(np.isfinite(inside_point)):
        inside_point = np.mean(reconstructed_vertices, axis=0)
        origin_source = "mean_reconstructed_vertices_fallback"
    else:
        origin_source = "parameters.inside_point"

    model_dir = args.data_root / args.model
    initial_model = parse_initial_model(model_dir / "InitialModel")
    contour_files = sorted_contour_files(
        model_dir / "shadow",
        "merged-cont*",
        args.max_contours,
    )
    contours = [parse_merged_contour(path) for path in contour_files]
    ray_counts = parse_ray_counts(args.sensitivity_rays, args.rays)

    reconstructed_results: dict[int, dict[str, object]] = {}
    reference_results: dict[int, dict[str, object]] = {}
    for ray_count in ray_counts:
        reconstructed_results[ray_count] = evaluate_polyhedron_polar_reprojection(
            name="reconstructed",
            vertices_3d=reconstructed_vertices,
            contours=contours,
            polar_origin_3d=inside_point,
            ray_count=ray_count,
        )
        reference_results[ray_count] = evaluate_polyhedron_polar_reprojection(
            name="initial_model_control",
            vertices_3d=initial_model.vertices,
            contours=contours,
            polar_origin_3d=inside_point,
            ray_count=ray_count,
        )

    primary_rays = max(8, int(args.rays))
    primary_reconstructed = reconstructed_results[primary_rays]
    primary_reference = reference_results[primary_rays]
    output = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "method": {
            "name": "common-origin polar outer-radius squared error",
            "polar_origin_source": origin_source,
            "polar_origin_3d": [float(v) for v in inside_point],
            "ray_grid": "uniform angles on [0, 2*pi)",
            "ray_difference": "r_projected_model(theta) - r_observed_contour(theta)",
            "intersection_rule": "farthest non-negative boundary intersection",
            "observed_contour": "ordered merged-cont polyline without convexification",
            "projected_model_contour": "convex hull of orthographically projected vertices",
            "note": (
                "Raw squared_error_sum scales linearly with views and rays; compare RMSE or "
                "angular_integrated_squared_error_sum across different grid resolutions."
            ),
        },
        "inputs": {
            "reconstruction_json": str(args.reconstruction_json),
            "reconstruction_sha256": sha256_file(reconstruction_path),
            "initial_model": str(model_dir / "InitialModel"),
            "shadow_pattern": str(model_dir / "shadow" / "merged-cont*"),
            "contour_count": len(contours),
            "reconstructed_vertex_count": int(reconstructed_vertices.shape[0]),
            "reference_vertex_count": int(initial_model.vertices.shape[0]),
        },
        "primary_ray_count": primary_rays,
        "reconstructed_vs_observed": primary_reconstructed,
        "initial_model_vs_observed_control": primary_reference,
        "comparison": comparison_summary(primary_reconstructed, primary_reference),
        "ray_count_sensitivity": [
            {
                "ray_count": ray_count,
                "reconstructed": compact_summary(reconstructed_results[ray_count]),
                "initial_model_control": compact_summary(reference_results[ray_count]),
            }
            for ray_count in ray_counts
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rec = primary_reconstructed
    ref = primary_reference
    print(f"model={args.model} contours={len(contours)} rays={primary_rays}")
    print(
        "reconstructed: "
        f"SSE={float(rec['squared_error_sum']):.12g} "
        f"RMSE={float(rec['root_mean_squared_error']):.12g} "
        f"relative_RMSE={100.0 * float(rec['relative_root_mean_squared_error']):.6f}%"
    )
    print(
        "initial control: "
        f"SSE={float(ref['squared_error_sum']):.12g} "
        f"RMSE={float(ref['root_mean_squared_error']):.12g} "
        f"relative_RMSE={100.0 * float(ref['relative_root_mean_squared_error']):.6f}%"
    )
    print(f"output={args.output_json}")


if __name__ == "__main__":
    main()
