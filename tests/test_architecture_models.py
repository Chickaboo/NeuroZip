import hashlib
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - dependency-free test environments.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional for the dependency-free test suite")
class ArchitectureModelTests(unittest.TestCase):
    def test_all_architectures_stream_and_round_trip(self):
        from neurozip.codec import compress_bytes, decompress_bytes
        from neurozip.models.registry import build_model, model_id_from_state
        from neurozip.predictors import NeuralPredictor

        cases = {
            "gru": dict(embedding_dim=16, hidden_size=32, num_layers=1),
            "lstm": dict(embedding_dim=16, hidden_size=28, num_layers=1),
            "transformer": dict(model_dim=32, num_layers=2, num_heads=4, ff_dim=64, context_length=16),
            "mamba": dict(model_dim=32, inner_dim=48, num_layers=2, conv_kernel=3, scan_chunk_size=4),
            "griffin": dict(model_dim=32, inner_dim=32, num_layers=2, conv_kernel=3, scan_chunk_size=4),
            "gated-deltanet": dict(model_dim=32, num_heads=4, num_layers=2, value_multiplier=1, scan_chunk_size=4),
            "gated-deltanet2": dict(model_dim=32, num_heads=4, num_layers=2, value_multiplier=2, scan_chunk_size=4),
        }
        original = b"The quick brown fox jumps over the lazy dog. " * 3
        with tempfile.TemporaryDirectory() as temporary:
            for name, config in cases.items():
                model = build_model(name, **config).eval()
                ids = torch.tensor([[256, 65, 66, 67]], dtype=torch.long)
                with torch.inference_mode():
                    full, _ = model(ids)
                    state = model.init_state(1, "cpu")
                    streamed = []
                    for token in ids[0]:
                        logits, state = model.step(token.view(1), state)
                        streamed.append(logits[0])
                    streamed = torch.stack(streamed).unsqueeze(0)
                self.assertLess(float((full - streamed).abs().max()), 1.0e-4, name)

                checkpoint_path = Path(temporary) / f"{name}.pt"
                state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save(
                    {
                        "format": "neurozip-sequence-checkpoint-v1",
                        "model_id": model_id_from_state(model.model_config, state_dict),
                        "model_config": model.model_config,
                        "train_config": {},
                        "step": 0,
                        "metrics": {},
                        "model_state_dict": state_dict,
                    },
                    checkpoint_path,
                )
                encoded = compress_bytes(
                    original,
                    NeuralPredictor.from_checkpoint(str(checkpoint_path)),
                    cdf_bits=12,
                )
                restored = decompress_bytes(
                    encoded,
                    NeuralPredictor.from_checkpoint(str(checkpoint_path)),
                )
                self.assertEqual(restored, original, name)
                self.assertEqual(hashlib.sha256(restored).digest(), hashlib.sha256(original).digest(), name)


if __name__ == "__main__":
    unittest.main()
