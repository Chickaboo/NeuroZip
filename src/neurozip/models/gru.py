"""The V0 byte-level GRU and its inference adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:  # Keep importing the package possible on the dependency-free host.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without optional deps.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from ..coding.cdf import DEFAULT_CDF_BITS, cdf_from_torch_logits


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for the GRU path; run training in Kaggle or "
            "install the optional NeuroZip train dependencies in Toolbx"
        )
    return torch


class ByteGRU(nn.Module if nn is not None else object):
    """Causal GRU with a BOS input and 256 byte logits."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        bos_id: int = 256,
        output_size: int = 256,
    ) -> None:
        _require_torch()
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bos_id = bos_id
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size + 1, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    @property
    def model_config(self) -> dict[str, int | str]:
        return {
            "architecture": "byte-gru-v0",
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bos_id": self.bos_id,
            "output_size": self.output_size,
        }

    def forward(self, input_ids: Any, hidden: Any = None) -> tuple[Any, Any]:
        embedded = self.embedding(input_ids)
        sequence, hidden = self.gru(embedded, hidden)
        return self.output(sequence), hidden


def model_id_from_state(model_config: dict[str, Any], state_dict: dict[str, Any]) -> str:
    """Create a stable identifier from canonical config and tensor bytes."""

    torch_module = _require_torch()
    digest = hashlib.sha256()
    digest.update(json.dumps(model_config, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        # PyTorch's NumPy bridge is present in the Kaggle training image. The
        # fallback keeps the function usable with minimal torch installations.
        try:
            digest.update(tensor.numpy().tobytes())
        except RuntimeError:
            digest.update(bytes(tensor.view(torch_module.uint8).tolist()))
    return f"gru-v0-{digest.hexdigest()[:16]}"


class GRUPredictor:
    """Streaming inference adapter used by the causal entropy coder."""

    def __init__(self, model: ByteGRU, *, model_id: str, device: str = "cpu") -> None:
        torch_module = _require_torch()
        self.model = model.to(device)
        self.model.eval()
        self.model_id = model_id
        self.device = device
        self._hidden: Any = None
        self._logits: Any = None
        self.reset()

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "cpu") -> "GRUPredictor":
        torch_module = _require_torch()
        try:
            checkpoint = torch_module.load(path, map_location=device, weights_only=False)
        except TypeError:  # Older PyTorch releases do not have weights_only.
            checkpoint = torch_module.load(path, map_location=device)
        model_config = checkpoint["model_config"]
        model = ByteGRU(**{key: model_config[key] for key in (
            "embedding_dim",
            "hidden_size",
            "num_layers",
            "bos_id",
            "output_size",
        )})
        model.load_state_dict(checkpoint["model_state_dict"])
        model_id = checkpoint.get("model_id")
        if not model_id:
            model_id = model_id_from_state(model_config, model.state_dict())
        return cls(model, model_id=model_id, device=device)

    def _step(self, token: int) -> None:
        torch_module = _require_torch()
        input_ids = torch_module.tensor([[token]], dtype=torch_module.long, device=self.device)
        with torch_module.inference_mode():
            logits, self._hidden = self.model(input_ids, self._hidden)
        self._logits = logits[0, 0].detach()

    def reset(self) -> None:
        self._hidden = None
        self._logits = None
        self._step(self.model.bos_id)

    def cdf(self, cdf_bits: int = DEFAULT_CDF_BITS) -> tuple[int, ...]:
        if self._logits is None:
            raise RuntimeError("GRU predictor must be reset before requesting a CDF")
        return cdf_from_torch_logits(self._logits, total=1 << cdf_bits)

    def update(self, symbol: int) -> None:
        if not 0 <= symbol < self.model.output_size:
            raise ValueError("GRU predictor received a non-byte symbol")
        self._step(symbol)
