"""V0 byte predictors.

The uniform predictor is dependency-free and exists for coder smoke tests. The
GRU predictor imports PyTorch lazily so local format tests do not require the
training stack.
"""

from __future__ import annotations

from array import array
from typing import Any

from .coding.cdf import DEFAULT_CDF_BITS, cdf_from_probs, cdf_from_torch_logits, uniform_cdf


class UniformPredictor:
    """A deterministic baseline that assigns equal probability to each byte."""

    model_id = "uniform-byte-v1"

    def __init__(self) -> None:
        self._cdf_cache: dict[int, tuple[int, ...]] = {}

    def reset(self) -> None:
        return None

    def cdf(self, cdf_bits: int = DEFAULT_CDF_BITS) -> tuple[int, ...]:
        if cdf_bits not in self._cdf_cache:
            self._cdf_cache[cdf_bits] = uniform_cdf(cdf_bits=cdf_bits)
        return self._cdf_cache[cdf_bits]

    def update(self, symbol: int) -> None:
        if not 0 <= symbol <= 255:
            raise ValueError("byte predictor received a non-byte symbol")


class AdaptiveNgramPredictor:
    """Dependency-free adaptive byte n-gram control for Gate A.

    It starts every file with add-one counts and updates the context after each
    decoded byte. The decoder therefore reconstructs the same distribution
    without transmitting a per-file model. It is a control, not the shared
    learned V0 model.
    """

    def __init__(self, order: int = 2) -> None:
        if order < 1 or order > 2:
            raise ValueError("the dependency-free V0 control supports order 1 or 2")
        self.order = order
        self.model_id = f"adaptive-byte-ngram{order}-v1"
        self._counts: dict[tuple[int, ...], array] = {}
        self._history: list[int] = []

    def reset(self) -> None:
        self._counts = {}
        self._history = [256] * self.order

    def _current_counts(self) -> array:
        key = tuple(self._history)
        counts = self._counts.get(key)
        if counts is None:
            counts = array("I", [1]) * 256
            self._counts[key] = counts
        return counts

    def cdf(self, cdf_bits: int = DEFAULT_CDF_BITS) -> tuple[int, ...]:
        return cdf_from_probs(self._current_counts(), total=1 << cdf_bits)

    def update(self, symbol: int) -> None:
        if not 0 <= symbol <= 255:
            raise ValueError("byte predictor received a non-byte symbol")
        counts = self._current_counts()
        counts[symbol] += 1
        self._history = (self._history + [symbol])[-self.order :]


def load_gru_predictor(model_path: str, *, device: str = "cpu") -> Any:
    """Load a trained neural predictor without importing PyTorch for controls."""

    return load_model_predictor(model_path, device=device)


class NeuralPredictor:
    """Streaming adapter shared by every self-describing neural checkpoint."""

    def __init__(self, model: Any, *, model_id: str, device: str = "cpu", model_config: dict[str, Any] | None = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for a neural predictor") from exc
        self._torch = torch
        self.model = model.to(device)
        self.model.eval()
        self.model_id = model_id
        self.device = device
        self.model_config = model_config or getattr(model, "model_config", {})
        self._state: Any = None
        self._logits: Any = None
        self.reset()

    @classmethod
    def from_checkpoint(cls, path: str, *, device: str = "cpu") -> "NeuralPredictor":
        from .models.registry import load_checkpoint, model_id_from_state

        model, checkpoint = load_checkpoint(path, device=device)
        model_id = checkpoint.get("model_id") or model_id_from_state(
            checkpoint["model_config"], checkpoint["model_state_dict"]
        )
        return cls(
            model,
            model_id=model_id,
            device=device,
            model_config=checkpoint["model_config"],
        )

    def _step(self, token: int) -> None:
        input_ids = self._torch.tensor([token], dtype=self._torch.long, device=self.device)
        with self._torch.inference_mode():
            logits, self._state = self.model.step(input_ids, self._state)
        self._logits = logits[0].detach()

    def reset(self) -> None:
        self._state = self.model.init_state(1, self.device)
        self._logits = None
        self._step(int(self.model.bos_id))

    def cdf(self, cdf_bits: int = DEFAULT_CDF_BITS) -> tuple[int, ...]:
        if self._logits is None:
            raise RuntimeError("neural predictor must be reset before requesting a CDF")
        return cdf_from_torch_logits(self._logits, total=1 << cdf_bits)

    def update(self, symbol: int) -> None:
        if not 0 <= symbol < self.model.output_size:
            raise ValueError("neural predictor received a non-byte symbol")
        self._step(symbol)


def load_model_predictor(model_path: str, *, device: str = "cpu") -> NeuralPredictor:
    """Load GRU or any architecture checkpoint through one codec adapter."""

    return NeuralPredictor.from_checkpoint(model_path, device=device)
