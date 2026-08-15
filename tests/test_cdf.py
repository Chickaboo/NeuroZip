import unittest

from neurozip.coding import cdf_from_logits, cdf_from_probs, uniform_cdf, validate_cdf


class CDFTests(unittest.TestCase):
    def test_uniform_cdf_is_strict_and_exact(self):
        cdf = uniform_cdf(cdf_bits=16)
        validate_cdf(cdf, total=1 << 16)
        self.assertEqual(cdf[0], 0)
        self.assertEqual(cdf[-1], 1 << 16)

    def test_largest_remainder_is_deterministic(self):
        first = cdf_from_probs([0.1, 0.2, 0.7], total=32)
        second = cdf_from_probs([0.1, 0.2, 0.7], total=32)
        self.assertEqual(first, second)
        self.assertEqual(first[-1], 32)
        self.assertEqual(cdf_from_logits([0.0, 1.0, -1.0], total=32)[-1], 32)

