import json
import tempfile
import unittest
from pathlib import Path

from neurozip.data.wikitext import deterministic_window, prepare_samples


class WikiTextSamplingTests(unittest.TestCase):
    def test_window_is_deterministic_and_byte_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wiki.train.raw"
            raw = bytes(range(256)) * 100
            path.write_bytes(raw)
            first, first_offset = deterministic_window(path, 500, seed=7, split="train")
            second, second_offset = deterministic_window(path, 500, seed=7, split="train")
            self.assertEqual(first, second)
            self.assertEqual(first_offset, second_offset)
            self.assertEqual(first, raw[first_offset : first_offset + 500])

    def test_prepare_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            (raw_dir / "wiki.train.raw").write_bytes(b"train" * 1000)
            (raw_dir / "wiki.valid.raw").write_bytes(b"valid" * 1000)
            output = root / "output"
            manifest = prepare_samples(
                output_dir=output,
                raw_dir=raw_dir,
                train_bytes=100,
                valid_bytes=50,
                seed=9,
            )
            self.assertEqual((output / "train.raw").stat().st_size, 100)
            self.assertEqual((output / "validation.raw").stat().st_size, 50)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["seed"], 9)
            self.assertEqual(manifest["encoding"].startswith("raw bytes"), True)

