import hashlib
import unittest

from neurozip.codec import compress_bytes, decompress_bytes
from neurozip.file_format import HEADER, parse_container
from neurozip.predictors import AdaptiveNgramPredictor, UniformPredictor


class CodecTests(unittest.TestCase):
    def test_round_trip_edge_cases(self):
        cases = [
            b"",
            b"a",
            bytes(range(256)),
            b"line one\r\nline two\n" * 100,
            "Unicode: café — 日本語 👋\n".encode("utf-8"),
            b"\x00\xff" * 1000,
            bytes(range(256)) * 100,
        ]
        for original in cases:
            container = compress_bytes(original, UniformPredictor(), cdf_bits=16)
            restored = decompress_bytes(container, UniformPredictor())
            self.assertEqual(restored, original)
            header, payload = parse_container(container)
            self.assertEqual(header.original_size, len(original))
            self.assertEqual(header.payload_size, len(payload))
            self.assertEqual(header.checksum, hashlib.sha256(original).digest())

    def test_corruption_is_detected(self):
        original = b"NeuroZip checksum test" * 100
        container = bytearray(compress_bytes(original, UniformPredictor()))
        checksum_start = HEADER.size + len(UniformPredictor.model_id)
        container[checksum_start] ^= 0x01
        with self.assertRaises(ValueError):
            decompress_bytes(bytes(container), UniformPredictor())

    def test_header_rejects_trailing_bytes(self):
        container = compress_bytes(b"test", UniformPredictor()) + b"trailing"
        with self.assertRaises(ValueError):
            parse_container(container)

    def test_adaptive_ngram_round_trip(self):
        original = (b"to be or not to be, that is the question\n" * 50)
        container = compress_bytes(original, AdaptiveNgramPredictor(order=2), cdf_bits=16)
        restored = decompress_bytes(container, AdaptiveNgramPredictor(order=2))
        self.assertEqual(restored, original)
