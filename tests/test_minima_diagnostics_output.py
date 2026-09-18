from __future__ import annotations

import base64
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import generate_minima_cloud_distance_viewer as viewer


def payload(name: str, count: int = 1) -> dict:
    return {"model": name, "cloud": {"points": [[1, 2, 3]] * count}, "reference": {"vertices": [[0, 0, 0]]},
            "summary": {"count": count, "mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}}


def fragment_models() -> dict:
    models = {name: payload(name, count) for count, name in enumerate(("round", "princess", "radiant", "pear", "cushion"), 1)}
    models["round"]["sources"] = {"note": "Эталон </script> — исходные координаты", "parameters": {"limit": None}}
    models["round"]["edge_diagnostics"] = {
        "points": {"candidate_edge_ids": [[2, 8], []], "uncertainty_proxy": [1e-16, None], "states": ["ambiguous", "face"]},
        "edges": [{"edge_id": 2, "coverage_intervals_t": [[0.1, 0.3], [0.7, 1.0]], "unsupported_reason": None}],
        "semantics": {"note": "Не заменять промежутки диапазоном min/max"},
    }
    models["round"]["projection_diagnostics"] = {
        "view_count": 2,
        "summary": {"finite_faces": {"rms": 0.125}},
        "unused_by_ui": {"keep_exactly": [True, False, None, -0.0, 1e-100]},
        "per_view": [
            {"contour_index": 17, "angle": -0.3, "finite_faces": {"rms": 0.25, "note": "</script>literal"},
             "horizontal_probes": [{"z": -5.0, "halves": [{"half_id": 0, "selected_minus_extreme": None}]}],
             "observed_polygon_2d": [[-1.25, -2.5], [2.0, 3.125]], "ray_gap_details": [{"finite_intervals": [[0, 2], [3, 4]]}]},
            {"contour_index": 4, "angle": 1.7, "finite_faces": {"rms": 0.03125},
             "horizontal_probes": [], "observed_outer_radii": [None, 2.5], "finite_outer_radii": [0.0, 2.0]},
        ],
    }
    return models


class DiagnosticOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.html = self.root / "viewer.html"
        self.output_json = self.root / "diagnostic.json"
        self.bundle = self.root / "plotly.js"
        self.bundle.write_text("// test bundle")
        self.source = self.root / "source.json"
        self.source.write_text("{}")
        model_dir = self.root / "round"
        model_dir.mkdir()
        (model_dir / "InitialModel").write_text("test")
        self.previous = {"round": payload("round"), "princess": payload("princess", 2)}
        viewer.write_html(self.previous, self.html, self.bundle)

    def run_main(self, *extra: str) -> None:
        args = ["viewer", "--model", "round", "--data-root", str(self.root),
                "--reconstruction-json", str(self.source), "--output-html", str(self.html),
                "--plotly-bundle", str(self.bundle), *extra]
        with patch.object(sys, "argv", args):
            viewer.main()

    def test_merge_retains_other_model_exactly_in_both_artifacts(self) -> None:
        updated = payload("round", 3)
        with patch.object(viewer, "build_payload", return_value=updated):
            self.run_main("--merge-existing", "--output-json", str(self.output_json))
        expected = {"round": updated, "princess": self.previous["princess"]}
        self.assertEqual(viewer.read_models(self.html), expected)
        self.assertEqual(viewer.read_models(self.output_json), expected)

    def test_subset_overwrite_is_rejected_before_computation(self) -> None:
        before = self.html.read_bytes()
        with patch.object(viewer, "build_payload") as build:
            with self.assertRaisesRegex(ValueError, "discard other model"):
                self.run_main()
        build.assert_not_called()
        self.assertEqual(before, self.html.read_bytes())

    def test_existing_non_diagnostic_json_is_never_replaced(self) -> None:
        self.output_json.write_text('{"reconstructed":{"vertices":[]}}')
        before = self.output_json.read_bytes()
        with self.assertRaisesRegex(ValueError, "Not a minima diagnostic"):
            self.run_main("--merge-existing", "--output-json", str(self.output_json))
        self.assertEqual(before, self.output_json.read_bytes())

    def test_render_from_json_never_recomputes(self) -> None:
        self.output_json.write_text(json.dumps({"artifact_type": "polyreco_minima_diagnostics", "models": self.previous}))
        with patch.object(viewer, "build_payload") as build:
            self.run_main("--from-json", str(self.output_json))
        build.assert_not_called()
        self.assertEqual(viewer.read_models(self.html), self.previous)

    def test_nonfinite_result_does_not_replace_outputs(self) -> None:
        before = self.html.read_bytes()
        invalid = payload("round")
        invalid["bad"] = float("nan")
        with patch.object(viewer, "build_payload", return_value=invalid):
            with self.assertRaises(ValueError):
                self.run_main("--merge-existing", "--output-json", str(self.output_json))
        self.assertFalse(self.output_json.exists())
        self.assertEqual(before, self.html.read_bytes())

    def test_embedded_script_closing_text_roundtrips_safely(self) -> None:
        data = payload("round")
        data["note"] = "</script><p>literal</p>"
        viewer.write_html({"round": data}, self.html, self.bundle)
        self.assertEqual(viewer.read_models(self.html), {"round": data})
        self.assertNotIn("</script><p>literal", self.html.read_text())

    def test_fragments_roundtrip_preserves_every_model_and_diagnostic_value(self) -> None:
        models = fragment_models()
        before = copy.deepcopy(models)
        packed = viewer.pack_models(models)
        self.assertEqual(viewer.unpack_models(packed), models)
        self.assertEqual(viewer.canonical_sha256(viewer.unpack_models(packed)), viewer.canonical_sha256(models))
        self.assertEqual(models, before)
        self.assertEqual(list(packed["models"]), list(models))
        round_fragments = packed["models"]["round"]
        base = viewer.unpack_value(round_fragments["base"])
        self.assertNotIn("edge_diagnostics", base)
        self.assertNotIn("projection_diagnostics", base)
        self.assertNotIn("per_view", viewer.unpack_value(round_fragments["projection"]["metadata"]))
        expected_index = [{key: row[key] for key in ("contour_index", "angle", "finite_faces")}
                          for row in models["round"]["projection_diagnostics"]["per_view"]]
        self.assertEqual(round_fragments["projection"]["index"], expected_index)
        self.assertEqual(len(round_fragments["projection"]["views"]), 2)
        for name in ("princess", "radiant", "pear", "cushion"):
            self.assertEqual(set(packed["models"][name]), {"base"})
            self.assertEqual(viewer.unpack_value(packed["models"][name]["base"]), models[name])

    def test_packed_html_readback_preserves_projection_order_and_literal_text(self) -> None:
        models = fragment_models()
        viewer.write_html(models, self.html, self.bundle)
        html = self.html.read_text()
        self.assertIn('<script id="packed-models" type="application/json">', html)
        self.assertNotIn("const MODELS = ", html)
        self.assertNotIn("</script>literal", html)
        restored = viewer.read_models(self.html)
        self.assertEqual(restored, models)
        self.assertEqual([row["contour_index"] for row in restored["round"]["projection_diagnostics"]["per_view"]], [17, 4])

    def test_fragment_packing_and_html_are_deterministic(self) -> None:
        models = fragment_models()
        first = viewer.pack_models(models)
        self.assertEqual(first, viewer.pack_models(copy.deepcopy(models)))
        # Gzip timestamps must not turn unchanged data into different artifacts.
        self.assertEqual(base64.b64decode(first["models"]["round"]["base"])[4:8], b"\0\0\0\0")
        viewer.write_html(models, self.html, self.bundle)
        before = self.html.read_bytes()
        viewer.write_html(copy.deepcopy(models), self.html, self.bundle)
        self.assertEqual(self.html.read_bytes(), before)

    def test_corrupt_base64_gzip_and_truncated_gzip_are_rejected(self) -> None:
        compressed = base64.b64decode(viewer.pack_value({"finite": [1.0, None]}))
        checksum_corrupted = bytearray(compressed)
        checksum_corrupted[-8] ^= 1
        cases = [
            "not@base64!",
            base64.b64encode(b"not a gzip stream").decode("ascii"),
            base64.b64encode(compressed[:-8]).decode("ascii"),
            base64.b64encode(checksum_corrupted).decode("ascii"),
            base64.b64encode(gzip.compress(b"not JSON", mtime=0)).decode("ascii"),
        ]
        for corrupt in cases:
            with self.subTest(value=corrupt), self.assertRaises((ValueError, OSError, EOFError)):
                viewer.unpack_value(corrupt)

    def test_html_reader_rejects_corruption_in_any_fragment(self) -> None:
        paths = [("base",), ("edges",), ("projection", "metadata"), ("projection", "views", 1)]
        for path in paths:
            packed = viewer.pack_models(fragment_models())
            target = packed["models"]["round"]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = "invalid@gzip"
            self.html.write_text('<script id="packed-models" type="application/json">' + json.dumps(packed) + "</script>")
            with self.subTest(path=path), self.assertRaises((ValueError, OSError, EOFError)):
                viewer.read_models(self.html)

    def test_unknown_fragment_encoding_is_rejected(self) -> None:
        packed = viewer.pack_models(fragment_models())
        packed["encoding"] = "unsupported-v2"
        with self.assertRaisesRegex(ValueError, "Unsupported minima viewer encoding"):
            viewer.unpack_models(packed)

    def test_legacy_uncompressed_html_remains_readable_for_merge(self) -> None:
        self.html.write_text("<script>const MODELS = " + json.dumps(self.previous) + ";</script>")
        self.assertEqual(viewer.read_models(self.html), self.previous)


if __name__ == "__main__":
    unittest.main()
