"""Meaningful geometry/configuration checks; no neural model is trained or run."""

from pathlib import Path
from dataclasses import replace
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import json
import csv

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xct_cracks.demo import generate_pair, procedural_background
from xct_cracks import Optimized_Crack_Engine as engine
from xct_cracks.dataset import validate_training_dataset


class GeometryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image, cls.mask = procedural_background(271)
        cls.result = generate_pair(cls.image, cls.mask, 271)

    def test_native_fractional_geometry_and_unchanged_exterior(self):
        coverage = self.result["coverage"]
        self.assertEqual(coverage.shape, (256, 256))
        self.assertTrue(np.isfinite(coverage).all())
        self.assertGreater(float(coverage.sum()), 0)
        self.assertGreaterEqual(float(coverage.min()), 0)
        self.assertLessEqual(float(coverage.max()), 1)
        np.testing.assert_array_equal(coverage * 16, np.rint(coverage * 16))
        np.testing.assert_array_equal(self.image[~self.mask], self.result["synthetic"][~self.mask])
        self.assertTrue(set(np.unique(self.result["validity"])).issubset({0, 1}))

    def test_control_is_an_exact_identity(self):
        result = generate_pair(self.image, self.mask, 271, "control")
        np.testing.assert_array_equal(result["synthetic"], self.image)
        self.assertFalse(result["coverage"].any())
        self.assertFalse(result["decrement"].any())
        self.assertTrue(result["validity"].all())

    def test_same_seed_reproduces_geometry_and_rendering(self):
        repeated = generate_pair(self.image, self.mask, 271)
        np.testing.assert_array_equal(repeated["coverage"], self.result["coverage"])
        np.testing.assert_array_equal(repeated["synthetic"], self.result["synthetic"])

    def test_zero_coverage_does_not_change_image(self):
        rendered, decrement, _, _ = engine.render_crack(
            self.image, self.mask, np.zeros_like(self.image), 271,
            "core", engine.GenerationConfig(), 0.1)
        np.testing.assert_array_equal(rendered, self.image)
        self.assertFalse(decrement.any())

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_pair(self.image[:128], self.mask[:128], 271)
        with self.assertRaises(ValueError):
            generate_pair(self.image * np.nan, self.mask, 271)

    def test_import_does_not_require_a_research_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, XCT_PROJECT_ROOT=temporary,
                       XCT_SOURCE_DATA="missing-data", PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            result = subprocess.run(
                [sys.executable, "-c", "from xct_cracks import Optimized_Crack_Engine as e; assert not e.SOURCE_DATA.exists()"],
                cwd=temporary, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_mps_unavailable_stops_before_neural_work(self):
        from xct_cracks.mps import require_mps
        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                require_mps()

    def test_public_neural_entry_points_reject_cpu(self):
        self.assertEqual(engine.TrainingConfig().device, "mps")
        cfg = replace(engine.TrainingConfig(), device="cpu")
        with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
            engine.train_material("mortar", "unused", "unused", cfg)
        with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
            engine.predict_native(None, self.image, "cpu", cfg)
        with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
            engine.run_epoch(None, [], "cpu", 0.5)

    def test_exact_patch_size_keeps_validation_window_in_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            tifffile.imwrite(Path(temporary) / "mask.tiff", np.ones((256, 256), np.uint8))
            row = {"id": "native256", "mask": "mask.tiff"}
            selections = engine.validation_selections(temporary, [row], 256, 4)
            self.assertTrue(selections)
            self.assertTrue(all(s["origin_yx"] == [0, 0] for s in selections))

    def test_dataset_preflight_rejects_group_leakage_before_neural_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configuration.json").write_text(json.dumps({"generation": {"raster_supersampling": 4}}))
            (root / "COMPLETE.json").write_text("{}")
            rows = [{"id": "a", "background_group": "shared", "split": "train"},
                    {"id": "b", "background_group": "shared", "split": "validation"}]
            with (root / "manifest.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_training_dataset(root)


if __name__ == "__main__":
    unittest.main()
