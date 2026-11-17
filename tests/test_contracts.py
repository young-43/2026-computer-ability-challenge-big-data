import importlib.util
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inference_reference', ROOT / 'model/predict.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FusionContracts(unittest.TestCase):
    def test_equal_weight_logits(self):
        self.assertAlmostEqual(module.fuse_probability(2.0, -2.0, False), 0.5)

    def test_fallback_uses_global(self):
        self.assertAlmostEqual(module.fuse_probability(0.0, 5.0, True), 0.5)

    def test_extreme_logits_are_finite(self):
        for value in (-1000.0, 1000.0):
            result = module.fuse_probability(value, value, False)
            self.assertTrue(math.isfinite(result))
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

    def test_nonfinite_output_rejected(self):
        with self.assertRaises(ValueError):
            module.fuse_probability(float('nan'), 0.0, False)


if __name__ == '__main__':
    unittest.main()
