"""Architecture registry and checkpoint compatibility helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from .architectures import ByteGatedDeltaNet, ByteGriffin, ByteLSTM, ByteMamba, ByteTransformer
from .gru import ByteGRU, model_id_from_state as gru_model_id_from_state


ARCHITECTURE_ALIASES = {
    "gru": "byte-gru-v0",
    "lstm": "byte-lstm-v1",
    "transformer": "byte-transformer-v1",
    "mamba": "byte-mamba-lite-v1",
    "griffin": "byte-griffin-lite-v1",
    "gated-deltanet": "byte-gated-deltanet-lite-v1",
    "gated-deltanet2": "byte-gated-deltanet2-lite-v1",
}

SUPPORTED_ARCHITECTURES = tuple(ARCHITECTURE_ALIASES)


def canonical_architecture(name: str) -> str:
    return ARCHITECTURE_ALIASES.get(name, name)


def build_model(architecture: str, **config: Any):
    """Construct one of the sweep models from a short or canonical name."""

    name = canonical_architecture(architecture)
    if name == "byte-gru-v0":
        return ByteGRU(
            embedding_dim=int(config.get("embedding_dim", 256)),
            hidden_size=int(config.get("hidden_size", 512)),
            num_layers=int(config.get("num_layers", 2)),
            bos_id=int(config.get("bos_id", 256)),
            output_size=int(config.get("output_size", 256)),
        )
    if name == "byte-lstm-v1":
        return ByteLSTM(
            embedding_dim=int(config.get("embedding_dim", 256)),
            hidden_size=int(config.get("hidden_size", 448)),
            num_layers=int(config.get("num_layers", 2)),
            bos_id=int(config.get("bos_id", 256)),
            output_size=int(config.get("output_size", 256)),
        )
    if name == "byte-transformer-v1":
        return ByteTransformer(
            model_dim=int(config.get("model_dim", 256)),
            num_layers=int(config.get("num_layers", 4)),
            num_heads=int(config.get("num_heads", 8)),
            ff_dim=int(config.get("ff_dim", 896)),
            context_length=int(config.get("context_length", 2048)),
            bos_id=int(config.get("bos_id", 256)),
            output_size=int(config.get("output_size", 256)),
        )
    if name == "byte-mamba-lite-v1":
        return ByteMamba(
            model_dim=int(config.get("model_dim", 256)),
            inner_dim=int(config.get("inner_dim", 512)),
            num_layers=int(config.get("num_layers", 4)),
            conv_kernel=int(config.get("conv_kernel", 4)),
            scan_chunk_size=int(config.get("scan_chunk_size", 32)),
            bos_id=int(config.get("bos_id", 256)),
            output_size=int(config.get("output_size", 256)),
        )
    if name == "byte-griffin-lite-v1":
        return ByteGriffin(
            model_dim=int(config.get("model_dim", 256)),
            inner_dim=int(config.get("inner_dim", 384)),
            num_layers=int(config.get("num_layers", 7)),
            conv_kernel=int(config.get("conv_kernel", 4)),
            scan_chunk_size=int(config.get("scan_chunk_size", 32)),
            bos_id=int(config.get("bos_id", 256)),
            output_size=int(config.get("output_size", 256)),
        )
    if name in {"byte-gated-deltanet-lite-v1", "byte-gated-deltanet2-lite-v1"}:
        is_deltanet2 = name == "byte-gated-deltanet2-lite-v1"
        default_multiplier = 1
        return ByteGatedDeltaNet(
            model_dim=int(config.get("model_dim", 256)),
            num_heads=int(config.get("num_heads", 8)),
            num_layers=int(config.get("num_layers", 7)),
            value_multiplier=int(config.get("value_multiplier", default_multiplier)),
            scan_chunk_size=int(config.get("scan_chunk_size", 32)),
            decoupled_gates=bool(config.get("decoupled_gates", is_deltanet2)),
            bos_id=int(config.get("bos_id", 256)),
            output_size=int(config.get("output_size", 256)),
            architecture_name=name,
        )
    raise ValueError(f"unsupported NeuroZip architecture: {architecture!r}")


def model_from_config(model_config: dict[str, Any]):
    """Construct a model from the self-describing checkpoint config."""

    config = dict(model_config)
    architecture = config.pop("architecture")
    return build_model(architecture, **config)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    try:
        return tensor.numpy().tobytes()
    except RuntimeError:
        return bytes(tensor.view(torch.uint8).reshape(-1).tolist())


def model_id_from_state(model_config: dict[str, Any], state_dict: dict[str, Any]) -> str:
    """Hash architecture metadata and weights into a stream model identifier."""

    architecture = canonical_architecture(str(model_config["architecture"]))
    if architecture == "byte-gru-v0":
        # Preserve the V0 identifier algorithm so old and newly trained GRU
        # checkpoints remain interchangeable with existing streams.
        return gru_model_id_from_state(model_config, state_dict)
    digest = hashlib.sha256()
    digest.update(json.dumps(model_config, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(_tensor_bytes(tensor))
    prefix = architecture.removeprefix("byte-")
    if prefix.endswith("-v1"):
        prefix = prefix[:-3]
    prefix = prefix.replace("-", "")
    return f"{prefix}-v1-{digest.hexdigest()[:16]}"


def load_checkpoint(path: str, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model = model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


__all__ = [
    "ARCHITECTURE_ALIASES",
    "SUPPORTED_ARCHITECTURES",
    "build_model",
    "canonical_architecture",
    "load_checkpoint",
    "model_from_config",
    "model_id_from_state",
]
