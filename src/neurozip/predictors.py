"""V0 byte predictors.

The uniform predictor is dependency-free and exists for coder smoke tests. The
GRU predictor imports PyTorch lazily so local format tests do not require the
training stack.
"""

from __future__ import annotations

from array import array
from typing import Any

from .coding.cdf import DEFAULT_CDF_BITS, cdf_from_probs, uniform_cdf


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
    """Load a trained GRU predictor without importing PyTorch for uniform use."""

    from .models.gru import GRUPredictor

    return GRUPredictor.from_checkpoint(model_path, device=device)
