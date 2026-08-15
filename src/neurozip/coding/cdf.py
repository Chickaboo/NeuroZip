"""Deterministic conversion from probabilities/logits to integer CDFs."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


DEFAULT_CDF_BITS = 20
BYTE_ALPHABET_SIZE = 256


def validate_cdf(
    cdf: Sequence[int],
    *,
    alphabet_size: int = BYTE_ALPHABET_SIZE,
    total: int | None = None,
) -> None:
    """Validate the invariants required by the range coder."""

    if len(cdf) != alphabet_size + 1:
        raise ValueError(f"expected {alphabet_size + 1} CDF entries, got {len(cdf)}")
    if cdf[0] != 0:
        raise ValueError("CDF must start at zero")
    if total is not None and cdf[-1] != total:
        raise ValueError(f"CDF total {cdf[-1]} does not equal requested total {total}")
    previous = cdf[0]
    for value in cdf[1:]:
        if not isinstance(value, int):
            raise TypeError("CDF entries must be integers")
        if value <= previous:
            raise ValueError("CDF entries must be strictly increasing")
        previous = value


def uniform_cdf(
    *, cdf_bits: int = DEFAULT_CDF_BITS, alphabet_size: int = BYTE_ALPHABET_SIZE
) -> tuple[int, ...]:
    """Return an exactly uniform positive-count CDF."""

    total = 1 << cdf_bits
    if total < alphabet_size:
        raise ValueError("CDF total must be at least the alphabet size")
    base, remainder = divmod(total, alphabet_size)
    counts = [base + (1 if i < remainder else 0) for i in range(alphabet_size)]
    cdf = [0]
    for count in counts:
        cdf.append(cdf[-1] + count)
    result = tuple(cdf)
    validate_cdf(result, alphabet_size=alphabet_size, total=total)
    return result


def cdf_from_probs(
    probabilities: Iterable[float],
    *,
    total: int = 1 << DEFAULT_CDF_BITS,
) -> tuple[int, ...]:
    """Quantize probabilities with deterministic largest-remainder rounding.

    One count is reserved for every symbol. The remaining counts are allocated
    according to the probabilities. Ties are resolved by the lower symbol
    index, making the resulting CDF reproducible.
    """

    probs = [float(value) for value in probabilities]
    if not probs:
        raise ValueError("probability vector cannot be empty")
    if total < len(probs):
        raise ValueError("CDF total must be at least the alphabet size")
    if any(not math.isfinite(value) or value < 0.0 for value in probs):
        raise ValueError("probabilities must be finite and non-negative")
    probability_sum = math.fsum(probs)
    if probability_sum <= 0.0:
        return uniform_cdf(cdf_bits=(total.bit_length() - 1), alphabet_size=len(probs))

    remaining = total - len(probs)
    scaled = [(value / probability_sum) * remaining for value in probs]
    floors = [math.floor(value) for value in scaled]
    leftover = remaining - sum(floors)
    fractions = [value - floor for value, floor in zip(scaled, floors)]
    # Stable sorting by negative fraction then symbol index.
    order = sorted(range(len(probs)), key=lambda index: (-fractions[index], index))
    for index in order[:leftover]:
        floors[index] += 1

    counts = [1 + floor for floor in floors]
    cdf = [0]
    for count in counts:
        cdf.append(cdf[-1] + count)
    result = tuple(cdf)
    validate_cdf(result, alphabet_size=len(probs), total=total)
    return result


def cdf_from_logits(logits: Iterable[float], *, total: int = 1 << DEFAULT_CDF_BITS) -> tuple[int, ...]:
    """Convert arbitrary finite logits to a stable integer CDF."""

    values = [float(value) for value in logits]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("logits must be a non-empty finite vector")
    maximum = max(values)
    probabilities = [math.exp(value - maximum) for value in values]
    return cdf_from_probs(probabilities, total=total)


def cdf_from_torch_logits(logits, *, total: int = 1 << DEFAULT_CDF_BITS) -> tuple[int, ...]:
    """Fast deterministic CDF conversion for a one-dimensional torch tensor.

    This keeps the probability-to-CDF work in tensor operations during
    byte-by-byte GRU inference. ``stable=True`` makes equal fractional counts
    resolve by symbol order, matching :func:`cdf_from_probs`.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional training path.
        raise RuntimeError("PyTorch is required for tensor CDF conversion") from exc
    values = logits.detach().float()
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("torch logits must be a non-empty one-dimensional tensor")
    if not torch.isfinite(values).all():
        raise ValueError("torch logits must be finite")
    alphabet_size = int(values.numel())
    if total < alphabet_size:
        raise ValueError("CDF total must be at least the alphabet size")
    probabilities = torch.softmax(values, dim=0)
    remaining = total - alphabet_size
    scaled = probabilities * remaining
    floors = torch.floor(scaled).to(torch.int64)
    leftover = remaining - int(floors.sum().item())
    if leftover:
        fractions = scaled - floors.to(scaled.dtype)
        try:
            order = torch.argsort(fractions, descending=True, stable=True)
        except TypeError:  # pragma: no cover - old torch fallback.
            order = torch.argsort(fractions, descending=True)
        floors[order[:leftover]] += 1
    counts = floors + 1
    cdf = torch.cat((torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0)))
    return tuple(int(value) for value in cdf.cpu().tolist())
