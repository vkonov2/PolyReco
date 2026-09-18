from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_full_circle_split_cached_viewer import (  # noqa: E402
    HalfContour,
    ShadowContour,
    build_half_contours,
    half_contour_point_at_z,
)
from polyreco.rms_reprojection import contour_basis, convex_hull_2d  # noqa: E402
from polyreco.template_projection_diagnostics import (  # noqa: E402
    build_projection_diagnostics,
    merge_intervals,
    polygon_line_intervals,
    triangle_union_line_intervals,
)


def rectangle_triangles(x0, y0, x1, y1):
    points = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return points[np.asarray([[0, 1, 2], [0, 2, 3]])]


class TriangleUnionTests(unittest.TestCase):
    def test_concave_L_keeps_notch_that_convex_hull_fills(self):
        polygon = np.asarray([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], dtype=float)
        triangles = np.concatenate([rectangle_triangles(0, 0, 3, 1), rectangle_triangles(0, 1, 1, 3)])
        origin = np.asarray([0.5, 0.5])
        direction = np.asarray([1.0, 1.0]) / np.sqrt(2)
        actual = triangle_union_line_intervals(triangles, origin, direction[None, :])[0]
        observed, _, invalid = polygon_line_intervals(polygon, origin, direction)
        hull, _, _ = polygon_line_intervals(convex_hull_2d(polygon), origin, direction)
        self.assertFalse(invalid)
        np.testing.assert_allclose(actual, [[0, np.sqrt(0.5)]], atol=1e-12)
        np.testing.assert_allclose(actual, observed, atol=1e-12)
        self.assertGreater(hull[-1, 1], actual[-1, 1] + 1)

    def test_overlapping_and_reversed_triangles_have_no_spurious_holes(self):
        triangles = np.concatenate([rectangle_triangles(-1, -1, 1, 1), rectangle_triangles(-0.5, -0.5, 1.5, 1.5)])
        triangles[::2] = triangles[::2, ::-1]
        directions = np.column_stack([np.cos(np.arange(360) * np.pi / 180), np.sin(np.arange(360) * np.pi / 180)])
        union = triangle_union_line_intervals(triangles, np.zeros(2), directions)
        self.assertTrue(all(len(row) == 1 for row in union))
        self.assertTrue(all(row[0, 0] == 0 for row in union))

    def test_disjoint_intervals_are_explicit_not_bridged(self):
        triangles = np.concatenate([rectangle_triangles(0, 0, 1, 3), rectangle_triangles(1, 0, 3, 1), rectangle_triangles(3, 0, 4, 3)])
        intervals = triangle_union_line_intervals(triangles, [0.5, 2], [[1, 0]])[0]
        np.testing.assert_allclose(intervals, [[0, 0.5], [2.5, 3.5]])
        polygon = np.asarray([[0, 0], [4, 0], [4, 3], [3, 3], [3, 1], [1, 1], [1, 3], [0, 3]], dtype=float)
        observed, hits, invalid = polygon_line_intervals(polygon, [0.5, 2], [1, 0])
        np.testing.assert_allclose(observed, intervals)
        np.testing.assert_allclose(hits, [0.5, 2.5, 3.5])
        self.assertFalse(invalid)

    def test_null_ray_and_origin_outside(self):
        triangles = rectangle_triangles(1, -1, 2, 1)
        intervals = triangle_union_line_intervals(triangles, [0, 0], [[-1, 0], [1, 0]])
        self.assertEqual(len(intervals[0]), 0)
        np.testing.assert_allclose(intervals[1], [[1, 2]])

    def test_almost_collinear_triangle_cannot_extrapolate_beyond_finite_edges(self):
        triangle = np.asarray([[[3.523157156073775, -5.072955039295], [3.5232867411026563, -5.082336767413], [3.5188730064124303, -4.76279575773]]])
        intervals = triangle_union_line_intervals(triangle, [-0.13638519, -4.63609762], [[1, 0]], tol=1.2e-9)
        self.assertEqual(len(intervals[0]), 0)

    def test_vertex_and_collinear_hits_are_deduplicated(self):
        polygon = np.asarray([[0, 0], [1, 0], [2, 0], [2, 1], [0, 1]])
        interval, hits, invalid = polygon_line_intervals(polygon, [-1, 0], [1, 0])
        np.testing.assert_allclose(interval, [[1, 3]])
        np.testing.assert_allclose(hits, [1, 2, 3])
        self.assertFalse(invalid)
        np.testing.assert_allclose(merge_intervals([[0, 1], [1, 2], [0.5, 1.5]]), [[0, 2]])


class ProjectionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.asarray([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=float)
        self.faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        self.triangles = np.asarray([[face[0], face[i], face[i + 1]] for face in self.faces for i in range(1, len(face) - 1)])
        self.points = np.asarray([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], dtype=float)

    def build(self, contour, *, half_contours=None, vertices=None):
        halves = build_half_contours([contour], np.zeros(3), np.asarray([0, 0, 1])) if half_contours is None else half_contours
        return build_projection_diagnostics(
            vertices=self.vertices if vertices is None else vertices,
            faces=self.faces, triangles=self.triangles,
            contours=[contour], z_levels=np.asarray([-0.5, 0, 0.5]),
            half_contours=halves, polar_origin=np.zeros(3), ray_count=360,
        )

    def test_identity_cube_zero_finite_error_and_branch_consistency(self):
        contour = ShadowContour(0, np.asarray([1.0, 0, 0]), 0, self.points)
        result = self.build(contour)
        self.assertAlmostEqual(result["summary"]["finite_faces"]["root_mean_squared_error"], 0, places=12)
        self.assertEqual(result["summary"]["ray_union"]["finite_disjoint"], 0)
        self.assertEqual(result["geometry"]["non_supporting_face_count"], 0)
        horizontal = result["summary"]["horizontal"]
        self.assertEqual(horizontal["half_branch_mismatch_count"], 0)
        self.assertAlmostEqual(horizontal["left_residual"]["rms"], 0)
        self.assertAlmostEqual(horizontal["right_residual"]["rms"], 0)
        self.assertEqual(horizontal["probe_count"], 3)
        json.dumps(result, allow_nan=False)

    def test_model_scale_error_keeps_signed_left_right_without_alignment(self):
        contour = ShadowContour(0, np.asarray([1.0, 0, 0]), 0, self.points)
        model = self.vertices.copy()
        model[:, 1] *= 0.75
        result = self.build(contour, vertices=model)
        horizontal = result["summary"]["horizontal"]
        self.assertAlmostEqual(horizontal["left_residual"]["mean_signed"], 0.25)
        self.assertAlmostEqual(horizontal["right_residual"]["mean_signed"], -0.25)
        self.assertAlmostEqual(horizontal["width_difference"]["mean_signed"], -0.5)
        self.assertGreater(result["summary"]["finite_faces"]["root_mean_squared_error"], 0)
        self.assertEqual(result["per_view"][0]["calibration"]["model_to_observed_span_ratio_uv"], [0.75, 1.0])

    def test_normal_sign_and_clockwise_basis(self):
        angle = np.pi / 2
        normal = np.asarray([np.cos(angle), -np.sin(angle), 0])
        u, v = contour_basis(normal)
        np.testing.assert_allclose(u, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(v, [0, 0, 1], atol=1e-12)
        points = self.points[:, 1:2] * u + self.points[:, 2:3] * v
        result = self.build(ShadowContour(0, normal, angle, points))
        self.assertAlmostEqual(result["summary"]["finite_faces"]["root_mean_squared_error"], 0, places=12)
        self.assertAlmostEqual(result["per_view"][0]["calibration"]["clockwise_angle_error_radians"], 0, places=12)
        opposite = self.build(ShadowContour(0, -normal, angle + np.pi, points))
        self.assertAlmostEqual(opposite["summary"]["finite_faces"]["root_mean_squared_error"], 0, places=12)
        self.assertAlmostEqual(opposite["per_view"][0]["calibration"]["clockwise_angle_error_radians"], 0, places=12)

    def test_existing_half_selection_mismatch_is_visible(self):
        contour = ShadowContour(0, np.asarray([1.0, 0, 0]), 0, self.points)
        # An explicitly malformed branch crosses Z=0 three times.  Existing
        # centroid-based selection returns an interior crossing rather than min.
        branch = np.asarray([[0, -1, -1], [0, -1, 1], [0, 0, -1], [0, 1, 1]], dtype=float)
        half = HalfContour(0, 0, 0, contour.normal, branch)
        result = self.build(contour, half_contours=[half])
        probe = result["per_view"][0]["horizontal_probes"][1]
        row = probe["halves"][0]
        self.assertEqual(row["intersection_count"], 3)
        self.assertTrue(row["mismatch"])
        self.assertAlmostEqual(row["selected_u"], float(half_contour_point_at_z(branch, 0)[1]))
        self.assertEqual(row["expected_extreme_u"], -1)

    def test_tilted_normal_does_not_claim_same_Z_projection_check(self):
        normal = np.asarray([1, 0, 1], dtype=float) / np.sqrt(2)
        u, v = contour_basis(normal)
        points = self.points[:, 1:2] * u + self.points[:, 2:3] * v
        result = self.build(ShadowContour(0, normal, 0, points), half_contours=[])
        self.assertFalse(result["per_view"][0]["horizontal_valid"])
        self.assertEqual(result["summary"]["horizontal"]["probe_count"], 0)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
