from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polyreco.minima_edge_diagnostics import build_edge_diagnostics, nearest_finite_edge_distances  # noqa: E402


def half_contours(normals: list[list[float]] | None = None) -> list[dict[str, object]]:
    directions = normals or [[1, 0, 0], [0, 1, 0]]
    return [
        {
            "source_index": index,
            "source_angle": float(index * 90),
            "half_id": 1,
            "normal": normal,
            "points": [[0, 0, -2], [0, 0, 2]],
        }
        for index, normal in enumerate(directions)
    ]


def build(
    seed_points: list[list[float]],
    vertices: list[list[float]] | None = None,
    edges: list[list[int]] | None = None,
    **overrides: object,
) -> dict[str, object]:
    pts = np.asarray(seed_points, dtype=float).reshape((-1, 3))
    levels, zis = np.unique(pts[:, 2], return_inverse=True)
    arguments = {
        "points": pts,
        "vertices": np.array(vertices or [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float),
        "edges": np.array(edges or [[0, 1], [2, 3]], dtype=int),
        "surface_distances": np.zeros(len(pts)),
        "z_indices": zis,
        "z_levels": levels,
        "cyclic_indices": np.zeros(len(pts), dtype=int),
        "rms_values": np.full(len(pts), 1e-6),
        "window_size": 2,
        "half_contours": half_contours(),
        "edge_tolerance_fraction": 0.01,
        "ambiguity_tolerance_fraction": 0.001,
        "chunk_size": 2,
    }
    arguments.update(overrides)
    return build_edge_diagnostics(**arguments)


class MinimaEdgeDiagnosticsTests(unittest.TestCase):
    def test_finite_segment_clips_extension(self) -> None:
        distances, ids, parameters = nearest_finite_edge_distances(
            np.array([[2.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0, 1]]),
            chunk_size=1,
        )
        np.testing.assert_allclose(distances, [np.sqrt(2.0), 1.0])
        np.testing.assert_array_equal(ids, [0, 0])
        np.testing.assert_allclose(parameters, [1.0, 0.0])

    def test_same_z_is_not_nearest_point_on_sloped_edge(self) -> None:
        result = build([[1.0, 0.0, 0.2]], [[0, 0, 0], [1, 0, 1]], [[0, 1]])
        point = result["points"]
        self.assertAlmostEqual(point["nearest_edge_distances"][0], np.sqrt(0.32))
        self.assertAlmostEqual(point["nearest_edge_t"][0], 0.6)
        self.assertAlmostEqual(point["same_z_distances"][0], 0.8)
        self.assertAlmostEqual(point["same_z_t"][0], 0.2)

    def test_horizontal_on_level_is_segment_and_off_level_ineligible(self) -> None:
        result = build([[0.4, 0.0, 0.0], [0.4, 0.0, 0.2]], [[0, 0, 0], [1, 0, 0]], [[0, 1]])
        points = result["points"]
        self.assertAlmostEqual(points["same_z_distances"][0], 0.0)
        self.assertAlmostEqual(points["same_z_t"][0], 0.4)
        self.assertEqual(points["same_z_edge_ids"][1], -1)
        self.assertIsNone(points["same_z_distances"][1])
        self.assertTrue(result["edges"][0]["horizontal"])
        self.assertEqual(result["edges"][0]["geometric_z_count"], 1)

    def test_shared_endpoint_preserves_both_edges_without_assignment(self) -> None:
        result = build([[0, 0, 0]], [[0, 0, 0], [1, 0, 1], [-1, 0, 1]], [[0, 1], [0, 2]])
        point = result["points"]
        self.assertEqual(point["states"], ["ambiguous"])
        self.assertEqual(point["candidate_edge_ids"], [[0, 1]])
        self.assertEqual(point["assigned_edge_ids"], [-1])
        self.assertEqual(point["second_same_z_distances"], [0.0])

    def test_well_conditioned_unique_point_and_face_outlier_states(self) -> None:
        result = build([[0, 0, 0.5], [0.4, 0.4, 0.5], [3, 3, 0.5]], surface_distances=np.array([0, 0, 2.0]))
        self.assertEqual(result["points"]["states"], ["edge", "face", "outlier"])
        self.assertEqual(result["points"]["assigned_edge_ids"], [0, -1, -1])

    def test_underconstrained_window_never_claims_unique_edge(self) -> None:
        result = build([[0, 0, 0.5]], half_contours=half_contours([[1, 0, 0], [1, 0, 0]]))
        point = result["points"]
        self.assertEqual(point["states"], ["ambiguous"])
        self.assertEqual(point["uncertainty_status"], ["underdetermined"])
        self.assertIsNone(point["condition_numbers"][0])
        self.assertIsNone(point["uncertainty_proxy"][0])
        self.assertEqual(point["candidate_edge_ids"], [[0]])

    def test_missing_or_large_rms_proxy_is_ambiguous(self) -> None:
        for value, expected in [(np.nan, "missing_rms"), (0.1, "large_proxy")]:
            with self.subTest(value=value):
                result = build([[0, 0, 0.5]], rms_values=np.array([value]))
                self.assertEqual(result["points"]["states"], ["ambiguous"])
                self.assertEqual(result["points"]["uncertainty_status"], [expected])

    def test_tiny_aperture_with_zero_rms_does_not_claim_unique_edge(self) -> None:
        angle = np.radians(0.0001)
        halves = half_contours([[1, 0, 0], [float(np.cos(angle)), float(np.sin(angle)), 0]])
        result = build([[0, 0, 0.5]], half_contours=halves, rms_values=np.array([0.0]))
        point = result["points"]
        self.assertGreater(point["condition_numbers"][0], 1e8)
        self.assertEqual(point["uncertainty_status"], ["ill_conditioned"])
        self.assertIsNone(point["uncertainty_proxy"][0])
        self.assertEqual(point["states"], ["ambiguous"])
        self.assertEqual(point["assigned_edge_ids"], [-1])
        self.assertEqual(point["candidate_edge_ids"], [[0]])
        self.assertEqual(result["edges"][0]["unique_point_count"], 0)
        self.assertEqual(result["edges"][0]["candidate_point_count"], 1)
        self.assertEqual(result["tolerances"]["max_condition_number"], 1e8)
        json.dumps(result, allow_nan=False)

    def test_condition_threshold_override_is_used_and_recorded(self) -> None:
        angle = np.radians(20)
        halves = half_contours([[1, 0, 0], [float(np.cos(angle)), float(np.sin(angle)), 0]])
        result = build([[0, 0, 0.5]], half_contours=halves, rms_values=np.array([0.0]), max_condition_number=10)
        self.assertEqual(result["points"]["uncertainty_status"], ["ill_conditioned"])
        self.assertEqual(result["tolerances"]["max_condition_number"], 10.0)

    def test_uncertainty_can_preserve_close_second_edge(self) -> None:
        vertices = [[0, 0, 0], [0, 0, 1], [0.01, 0, 0], [0.01, 0, 1]]
        precise = build([[0.003, 0, 0.5]], vertices, [[0, 1], [2, 3]], rms_values=np.array([0.0]))
        uncertain = build([[0.003, 0, 0.5]], vertices, [[0, 1], [2, 3]], rms_values=np.array([0.003]))
        self.assertEqual(precise["points"]["candidate_edge_ids"], [[0]])
        self.assertEqual(uncertain["points"]["candidate_edge_ids"], [[0, 1]])
        self.assertEqual(uncertain["points"]["states"], ["ambiguous"])

    def test_coverage_keeps_gaps_and_endpoint_tails(self) -> None:
        result = build([[0, 0, 0.2], [0, 0, 0.8]], [[0, 0, 0], [0, 0, 1]], [[0, 1]])
        edge = result["edges"][0]
        np.testing.assert_allclose(edge["coverage_intervals_t"], [[0.19, 0.21], [0.79, 0.81]])
        self.assertAlmostEqual(edge["coverage_fraction"], 0.04)
        self.assertAlmostEqual(edge["max_gap_fraction"], 0.58)
        self.assertEqual(len(edge["gaps_t"]), 3)
        self.assertEqual(edge["sampling_level_fraction"], 1.0)

    def test_single_point_does_not_cover_whole_horizontal_edge(self) -> None:
        result = build([[0.5, 0, 0]], [[0, 0, 0], [1, 0, 0]], [[0, 1]])
        edge = result["edges"][0]
        self.assertAlmostEqual(edge["coverage_fraction"], 0.02)
        self.assertEqual(edge["unique_z_count"], 1)
        self.assertTrue(edge["horizontal"])

    def test_overlapping_windows_count_original_views_only_once(self) -> None:
        halves = half_contours([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        halves[2]["source_index"] = 0
        halves[3]["source_index"] = 1
        result = build([[0, 0, 0.3], [0, 0, 0.7]], half_contours=halves, cyclic_indices=np.array([0, 2]))
        self.assertEqual(result["edges"][0]["unique_source_view_count"], 2)
        self.assertEqual(result["edges"][0]["unique_source_views"], [0, 1])

    def test_visibility_retains_support_ties_and_horizontal_endpoints_separate(self) -> None:
        # For +Y both endpoints of each horizontal segment tie. For -X only
        # their left endpoints tie, so that direction is endpoint-only.
        result = build(
            [[0.5, 0, 0]], [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]],
            [[0, 1], [2, 3], [0, 2], [1, 3]],
        )
        horizontal = result["edges"][0]["ideal_by_z"][0]
        self.assertEqual(horizontal["visible_half_count"], 1)
        self.assertEqual(horizontal["endpoint_visible_half_count"], 2)
        self.assertEqual(horizontal["full_window_count"], 0)
        for edge_id in (2, 3):
            self.assertGreaterEqual(result["edges"][edge_id]["ideal_by_z"][0]["visible_half_count"], 1)

    def test_window_uses_only_lines_available_at_point_level(self) -> None:
        halves = half_contours()
        halves[1]["points"] = [[0, 0, -2], [0, 0, -1]]
        result = build([[0, 0, 0.5]], half_contours=halves)
        self.assertEqual(result["points"]["window_valid_line_count"], [1])
        self.assertEqual(result["points"]["uncertainty_status"], ["underdetermined"])

    def test_empty_cloud_has_serializable_metrics(self) -> None:
        result = build([], z_levels=np.array([0.0, 0.5, 1.0]))
        self.assertEqual(result["summary"]["point_count"], 0)
        self.assertEqual(result["summary"]["length_weighted_coverage"], 0.0)
        json.dumps(result, allow_nan=False)

    def test_result_contains_no_nonfinite_json_numbers(self) -> None:
        result = build([[0, 0, 1.5]], rms_values=np.array([np.nan]))
        json.dumps(result, allow_nan=False)

    def test_malformed_inputs_fail_explicitly(self) -> None:
        bad_inputs = [
            {"points": np.array([[np.nan, 0.0, 0.5]])},
            {"edges": np.array([[0, 100]])},
            {"edges": np.array([[0.0, 1.5]])},
            {"edges": np.array([[0, 0]])},
            {"edges": np.array([[0, 1], [1, 0]])},
            {"surface_distances": np.array([np.nan])},
            {"z_indices": np.array([1])},
            {"z_levels": np.array([0.0])},
            {"cyclic_indices": np.array([2])},
            {"rms_values": np.array([-1.0])},
            {"rms_values": np.array([np.inf])},
            {"window_size": 1},
            {"chunk_size": 0},
            {"edge_tolerance_fraction": 0},
            {"ambiguity_tolerance_fraction": np.inf},
            {"max_condition_number": 0.5},
            {"max_condition_number": np.inf},
            {"half_contours": half_contours([[1, 0, 0], [1, 0, 1]])},
        ]
        for overrides in bad_inputs:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                build([[0, 0, 0.5]], **overrides)


if __name__ == "__main__":
    unittest.main()
