import random
import unittest

from neurozip.coding import RangeDecoder, RangeEncoder, uniform_cdf


class RangeCoderTests(unittest.TestCase):
    def round_trip(self, symbols, cdf):
        encoder = RangeEncoder()
        for symbol in symbols:
            encoder.encode_symbol(cdf, symbol)
        payload = encoder.finish()
        decoder = RangeDecoder(payload)
        restored = [decoder.decode_symbol(cdf) for _ in symbols]
        self.assertEqual(restored, symbols)

    def test_empty_stream(self):
        payload = RangeEncoder().finish()
        self.assertIsInstance(payload, bytes)

    def test_all_symbols(self):
        cdf = uniform_cdf(cdf_bits=16)
        self.round_trip(list(range(256)) * 2, cdf)

    def test_random_symbols(self):
        random_generator = random.Random(20260814)
        symbols = [random_generator.randrange(256) for _ in range(5000)]
        self.round_trip(symbols, uniform_cdf(cdf_bits=20))

