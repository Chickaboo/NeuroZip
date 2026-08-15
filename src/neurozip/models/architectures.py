"""Comparable byte-level sequence architectures for NeuroZip experiments.

The models in this module deliberately share three properties:

* a 257-entry byte/BOS input vocabulary and a 256-way byte output;
* a batch-first training ``forward`` method returning ``(logits, state)``;
* a one-token ``step`` method with an explicit recurrent/cache state.

The latter is what lets every architecture use the same integer CDF and
range-coder path.  The Mamba, Griffin, and Gated-DeltaNet implementations are
small reference variants written in ordinary PyTorch.  They are useful for a
fair architecture sweep, but are intentionally not presented as optimized
reproductions of vendor CUDA kernels or the full research implementations.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _chunked_affine_scan(
    coefficient: torch.Tensor,
    input_term: torch.Tensor,
    initial: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallelize ``h[t] = coefficient[t] * h[t-1] + input_term[t]``.

    A short chunk keeps the cumulative-product form numerically bounded while
    avoiding a Python loop over every byte during training.  Streaming
    inference uses the same recurrence one token at a time.
    """

    if coefficient.shape != input_term.shape:
        raise ValueError("affine scan coefficient and input term must match")
    if coefficient.ndim != 3:
        raise ValueError("affine scan expects [batch, time, channels]")
    state = initial
    outputs: list[torch.Tensor] = []
    for start in range(0, coefficient.shape[1], chunk_size):
        stop = min(start + chunk_size, coefficient.shape[1])
        local_coefficient = coefficient[:, start:stop].clamp(0.5, 0.9995)
        local_input = input_term[:, start:stop]
        prefix = torch.cumprod(local_coefficient, dim=1)
        scaled_input = local_input / prefix.clamp_min(1.0e-6)
        local_state = prefix * (
            state.unsqueeze(1) + torch.cumsum(scaled_input, dim=1)
        )
        outputs.append(local_state)
        state = local_state[:, -1]
    return torch.cat(outputs, dim=1), state


def _sinusoidal_positions(
    positions: torch.Tensor,
    model_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create deterministic positional features without learned metadata."""

    inverse_frequency = torch.exp(
        torch.arange(0, model_dim, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / model_dim)
    )
    angles = positions.to(torch.float32).unsqueeze(1) * inverse_frequency.unsqueeze(0)
    encoding = torch.zeros(positions.numel(), model_dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype).unsqueeze(0)


class ByteLSTM(nn.Module):
    """Two-layer LSTM control with the same byte interface as the GRU."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        hidden_size: int = 448,
        num_layers: int = 2,
        bos_id: int = 256,
        output_size: int = 256,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bos_id = bos_id
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size + 1, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    @property
    def model_config(self) -> dict[str, int | str]:
        return {
            "architecture": "byte-lstm-v1",
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bos_id": self.bos_id,
            "output_size": self.output_size,
        }

    def init_state(self, batch_size: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (self.num_layers, batch_size, self.hidden_size)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )

    def forward(self, input_ids: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        embedded = self.embedding(input_ids)
        sequence, state = self.lstm(embedded, state)
        return self.output(sequence), state

    def step(self, input_ids: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        logits, state = self.forward(input_ids, state)
        return logits[:, 0], state


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, context_length: int) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.context_length = context_length
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.output = nn.Linear(model_dim, model_dim)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, time, _ = tensor.shape
        return tensor.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

    def _join(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, time, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, time, self.model_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        query, key, value = self._split(query), self._split(key), self._split(value)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        causal_mask = torch.triu(
            torch.ones(inputs.shape[1], inputs.shape[1], device=inputs.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return self.output(self._join(torch.matmul(weights, value)))

    def step(
        self,
        inputs: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        query, key, value = self._split(query), self._split(key), self._split(value)
        if cache is not None:
            key = torch.cat([cache[0], key], dim=2)
            value = torch.cat([cache[1], value], dim=2)
        if key.shape[2] > self.context_length:
            key = key[:, :, -self.context_length :]
            value = value[:, :, -self.context_length :]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        output = self.output(self._join(torch.matmul(weights, value)))
        return output, (key.detach(), value.detach())


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, context_length: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attention = CausalSelfAttention(model_dim, num_heads, context_length)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, model_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.norm1(inputs))
        return inputs + self.feed_forward(self.norm2(inputs))

    def step(
        self,
        inputs: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, cache = self.attention.step(self.norm1(inputs), cache)
        inputs = inputs + attended
        return inputs + self.feed_forward(self.norm2(inputs)), cache


class ByteTransformer(nn.Module):
    """Small decoder-only Transformer with a bounded streaming KV cache."""

    def __init__(
        self,
        *,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 896,
        context_length: int = 2048,
        bos_id: int = 256,
        output_size: int = 256,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.context_length = context_length
        self.bos_id = bos_id
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size + 1, model_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(model_dim, num_heads, ff_dim, context_length) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, output_size)

    @property
    def model_config(self) -> dict[str, int | str]:
        return {
            "architecture": "byte-transformer-v1",
            "model_dim": self.model_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "context_length": self.context_length,
            "bos_id": self.bos_id,
            "output_size": self.output_size,
        }

    def init_state(self, batch_size: int, device: str | torch.device) -> dict[str, Any]:
        del batch_size, device
        return {"caches": [None] * self.num_layers, "position": 0}

    def forward(self, input_ids: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        del state
        sequence = self.embedding(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        sequence = sequence + _sinusoidal_positions(
            positions,
            self.model_dim,
            device=input_ids.device,
            dtype=sequence.dtype,
        )
        for layer in self.layers:
            sequence = layer(sequence)
        return self.output(self.final_norm(sequence)), None

    def step(self, input_ids: torch.Tensor, state: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        caches = [None] * self.num_layers if state is None else state["caches"]
        position = 0 if state is None else int(state["position"])
        sequence = self.embedding(input_ids)
        positions = torch.arange(
            position,
            position + input_ids.shape[1],
            device=input_ids.device,
        )
        sequence = sequence + _sinusoidal_positions(
            positions,
            self.model_dim,
            device=input_ids.device,
            dtype=sequence.dtype,
        )
        next_caches: list[Any] = []
        for layer, cache in zip(self.layers, caches):
            sequence, cache = layer.step(sequence, cache)
            next_caches.append(cache)
        return self.output(self.final_norm(sequence))[:, 0], {
            "caches": next_caches,
            "position": position + input_ids.shape[1],
        }


class GriffinBlock(nn.Module):
    """A Griffin-style local convolution plus gated linear recurrence."""

    def __init__(self, model_dim: int, inner_dim: int, conv_kernel: int, scan_chunk_size: int) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.conv_kernel = conv_kernel
        self.scan_chunk_size = scan_chunk_size
        self.norm = nn.LayerNorm(model_dim)
        self.input = nn.Linear(model_dim, 2 * inner_dim)
        self.local_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            conv_kernel,
            groups=inner_dim,
        )
        self.gate = nn.Linear(inner_dim, inner_dim)
        self.output = nn.Linear(inner_dim, model_dim)

    def _conv_full(self, values: torch.Tensor) -> torch.Tensor:
        padded = F.pad(values.transpose(1, 2), (self.conv_kernel - 1, 0))
        return self.local_conv(padded).transpose(1, 2)

    def _conv_step(
        self,
        values: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = values.new_zeros(values.shape[0], self.inner_dim, self.conv_kernel - 1)
        sequence = torch.cat([state, values.unsqueeze(-1)], dim=-1)
        result = F.conv1d(
            sequence,
            self.local_conv.weight,
            self.local_conv.bias,
            groups=self.inner_dim,
        )
        next_state = sequence[:, :, 1:] if self.conv_kernel > 1 else sequence[:, :, :0]
        return result[:, :, 0], next_state

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values, gates = self.input(self.norm(inputs)).chunk(2, dim=-1)
        values = torch.tanh(self._conv_full(values))
        coefficient = 0.5 + 0.499 * torch.sigmoid(self.gate(values))
        input_term = (1.0 - coefficient) * values
        state_sequence, _ = _chunked_affine_scan(
            coefficient,
            input_term,
            values.new_zeros(values.shape[0], self.inner_dim),
            chunk_size=self.scan_chunk_size,
        )
        return inputs + self.output(F.silu(gates) * state_sequence)

    def step(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        recurrent = None if state is None else state[0]
        convolution = None if state is None else state[1]
        values, gates = self.input(self.norm(inputs)).chunk(2, dim=-1)
        values, convolution = self._conv_step(values, convolution)
        values = torch.tanh(values)
        coefficient = 0.5 + 0.499 * torch.sigmoid(self.gate(values))
        recurrent = (
            coefficient * (values.new_zeros(values.shape) if recurrent is None else recurrent)
            + (1.0 - coefficient) * values
        )
        return inputs + self.output(F.silu(gates) * recurrent), (recurrent, convolution)


class ByteGriffin(nn.Module):
    """Pure-PyTorch Griffin-Lite reference model."""

    def __init__(
        self,
        *,
        model_dim: int = 256,
        inner_dim: int = 384,
        num_layers: int = 7,
        conv_kernel: int = 4,
        scan_chunk_size: int = 32,
        bos_id: int = 256,
        output_size: int = 256,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.num_layers = num_layers
        self.conv_kernel = conv_kernel
        self.scan_chunk_size = scan_chunk_size
        self.bos_id = bos_id
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size + 1, model_dim)
        self.layers = nn.ModuleList(
            [GriffinBlock(model_dim, inner_dim, conv_kernel, scan_chunk_size) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, output_size)

    @property
    def model_config(self) -> dict[str, int | str]:
        return {
            "architecture": "byte-griffin-lite-v1",
            "model_dim": self.model_dim,
            "inner_dim": self.inner_dim,
            "num_layers": self.num_layers,
            "conv_kernel": self.conv_kernel,
            "scan_chunk_size": self.scan_chunk_size,
            "bos_id": self.bos_id,
            "output_size": self.output_size,
        }

    def init_state(self, batch_size: int, device: str | torch.device) -> list[Any]:
        del batch_size, device
        return [None] * self.num_layers

    def forward(self, input_ids: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        del state
        sequence = self.embedding(input_ids)
        for layer in self.layers:
            sequence = layer(sequence)
        return self.output(self.final_norm(sequence)), None

    def step(self, input_ids: torch.Tensor, state: list[Any] | None = None) -> tuple[torch.Tensor, list[Any]]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        states = [None] * self.num_layers if state is None else state
        sequence = self.embedding(input_ids)
        next_states: list[Any] = []
        for layer, layer_state in zip(self.layers, states):
            sequence, layer_state = layer.step(sequence[:, 0], layer_state)
            sequence = sequence.unsqueeze(1)
            next_states.append(layer_state)
        return self.output(self.final_norm(sequence))[:, 0], next_states


class MambaBlock(nn.Module):
    """Diagonal selective state-space block with a parallel training scan."""

    def __init__(self, model_dim: int, inner_dim: int, conv_kernel: int, scan_chunk_size: int) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.conv_kernel = conv_kernel
        self.scan_chunk_size = scan_chunk_size
        self.norm = nn.LayerNorm(model_dim)
        self.input = nn.Linear(model_dim, 2 * inner_dim)
        self.local_conv = nn.Conv1d(inner_dim, inner_dim, conv_kernel, groups=inner_dim)
        self.delta = nn.Linear(inner_dim, inner_dim)
        self.log_decay = nn.Parameter(torch.full((inner_dim,), -3.0))
        self.input_scale = nn.Parameter(torch.full((inner_dim,), 0.1))
        self.output = nn.Linear(inner_dim, model_dim)

    def _conv_full(self, values: torch.Tensor) -> torch.Tensor:
        padded = F.pad(values.transpose(1, 2), (self.conv_kernel - 1, 0))
        return self.local_conv(padded).transpose(1, 2)

    def _conv_step(
        self,
        values: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = values.new_zeros(values.shape[0], self.inner_dim, self.conv_kernel - 1)
        sequence = torch.cat([state, values.unsqueeze(-1)], dim=-1)
        result = F.conv1d(
            sequence,
            self.local_conv.weight,
            self.local_conv.bias,
            groups=self.inner_dim,
        )
        next_state = sequence[:, :, 1:] if self.conv_kernel > 1 else sequence[:, :, :0]
        return result[:, :, 0], next_state

    def _state_parameters(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = F.softplus(self.delta(values)) + 1.0e-3
        decay = torch.exp(-delta * torch.exp(self.log_decay).clamp(max=10.0))
        decay = decay.clamp(0.5, 0.9995)
        input_term = torch.tanh(values) * self.input_scale
        return decay, input_term

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values, gates = self.input(self.norm(inputs)).chunk(2, dim=-1)
        values = self._conv_full(values)
        coefficient, input_term = self._state_parameters(values)
        state_sequence, _ = _chunked_affine_scan(
            coefficient,
            input_term,
            values.new_zeros(values.shape[0], self.inner_dim),
            chunk_size=self.scan_chunk_size,
        )
        return inputs + self.output(F.silu(gates) * state_sequence)

    def step(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        recurrent = None if state is None else state[0]
        convolution = None if state is None else state[1]
        values, gates = self.input(self.norm(inputs)).chunk(2, dim=-1)
        values, convolution = self._conv_step(values, convolution)
        coefficient, input_term = self._state_parameters(values)
        recurrent = coefficient * (values.new_zeros(values.shape) if recurrent is None else recurrent) + input_term
        return inputs + self.output(F.silu(gates) * recurrent), (recurrent, convolution)


class ByteMamba(nn.Module):
    """Pure-PyTorch Mamba-Lite reference model over raw bytes."""

    def __init__(
        self,
        *,
        model_dim: int = 256,
        inner_dim: int = 512,
        num_layers: int = 4,
        conv_kernel: int = 4,
        scan_chunk_size: int = 32,
        bos_id: int = 256,
        output_size: int = 256,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.num_layers = num_layers
        self.conv_kernel = conv_kernel
        self.scan_chunk_size = scan_chunk_size
        self.bos_id = bos_id
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size + 1, model_dim)
        self.layers = nn.ModuleList(
            [MambaBlock(model_dim, inner_dim, conv_kernel, scan_chunk_size) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, output_size)

    @property
    def model_config(self) -> dict[str, int | str]:
        return {
            "architecture": "byte-mamba-lite-v1",
            "model_dim": self.model_dim,
            "inner_dim": self.inner_dim,
            "num_layers": self.num_layers,
            "conv_kernel": self.conv_kernel,
            "scan_chunk_size": self.scan_chunk_size,
            "bos_id": self.bos_id,
            "output_size": self.output_size,
        }

    def init_state(self, batch_size: int, device: str | torch.device) -> list[Any]:
        del batch_size, device
        return [None] * self.num_layers

    def forward(self, input_ids: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        del state
        sequence = self.embedding(input_ids)
        for layer in self.layers:
            sequence = layer(sequence)
        return self.output(self.final_norm(sequence)), None

    def step(self, input_ids: torch.Tensor, state: list[Any] | None = None) -> tuple[torch.Tensor, list[Any]]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        states = [None] * self.num_layers if state is None else state
        sequence = self.embedding(input_ids)
        next_states: list[Any] = []
        for layer, layer_state in zip(self.layers, states):
            sequence, layer_state = layer.step(sequence[:, 0], layer_state)
            sequence = sequence.unsqueeze(1)
            next_states.append(layer_state)
        return self.output(self.final_norm(sequence))[:, 0], next_states


class GatedDeltaBlock(nn.Module):
    """Diagonal Gated DeltaNet-style memory with optional erase/write gates."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        value_multiplier: int,
        scan_chunk_size: int,
        decoupled_gates: bool = False,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.value_multiplier = value_multiplier
        self.scan_chunk_size = scan_chunk_size
        self.decoupled_gates = decoupled_gates
        self.norm = nn.LayerNorm(model_dim)
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim * value_multiplier)
        if decoupled_gates:
            self.erase = nn.Linear(model_dim, model_dim)
            self.write = nn.Linear(model_dim, model_dim)
        else:
            self.beta = nn.Linear(model_dim, model_dim)
            self.gate = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim * value_multiplier, model_dim)

    def _reshape_key(self, values: torch.Tensor) -> torch.Tensor:
        batch, time, _ = values.shape
        return values.view(batch, time, self.num_heads, self.head_dim)

    def _reshape_value(self, values: torch.Tensor) -> torch.Tensor:
        batch, time, _ = values.shape
        return values.view(batch, time, self.num_heads, self.head_dim, self.value_multiplier)

    def _coefficients(
        self,
        key: torch.Tensor,
        write_gate: torch.Tensor,
        erase_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        write_gate = torch.sigmoid(write_gate)
        erase_gate = 0.5 + 0.499 * torch.sigmoid(erase_gate)
        coefficient = (erase_gate - write_gate * key.square()).clamp(0.5, 0.9995)
        return coefficient, write_gate * key

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        query = self._reshape_key(self.query(normalized))
        key = self._reshape_key(self.key(normalized))
        value = self._reshape_value(self.value(normalized))
        if self.decoupled_gates:
            write_gate = self._reshape_key(self.write(normalized))
            erase_gate = self._reshape_key(self.erase(normalized))
        else:
            write_gate = self._reshape_key(self.beta(normalized))
            erase_gate = self._reshape_key(self.gate(normalized))
        coefficient, scaled_key = self._coefficients(key, write_gate, erase_gate)
        input_term = scaled_key.unsqueeze(-1) * value
        scan_coefficient = coefficient.unsqueeze(-1).expand_as(input_term)
        initial = input_term.new_zeros(input_term.shape[0], self.num_heads, self.head_dim, self.value_multiplier)
        state_sequence, _ = _chunked_affine_scan(
            scan_coefficient.reshape(input_term.shape[0], input_term.shape[1], -1),
            input_term.reshape(input_term.shape[0], input_term.shape[1], -1),
            initial.reshape(input_term.shape[0], -1),
            chunk_size=self.scan_chunk_size,
        )
        state_sequence = state_sequence.view_as(input_term)
        attended = query.unsqueeze(-1) * state_sequence
        return inputs + self.output(attended.reshape(inputs.shape[0], inputs.shape[1], -1))

    def step(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(inputs)
        query = self._reshape_key(self.query(normalized.unsqueeze(1)))[:, 0]
        key = self._reshape_key(self.key(normalized.unsqueeze(1)))[:, 0]
        value = self.value(normalized).view(-1, self.num_heads, self.head_dim, self.value_multiplier)
        if self.decoupled_gates:
            write_gate = self._reshape_key(self.write(normalized.unsqueeze(1)))[:, 0]
            erase_gate = self._reshape_key(self.erase(normalized.unsqueeze(1)))[:, 0]
        else:
            write_gate = self._reshape_key(self.beta(normalized.unsqueeze(1)))[:, 0]
            erase_gate = self._reshape_key(self.gate(normalized.unsqueeze(1)))[:, 0]
        coefficient, scaled_key = self._coefficients(
            key.unsqueeze(1), write_gate.unsqueeze(1), erase_gate.unsqueeze(1)
        )
        coefficient = coefficient[:, 0]
        scaled_key = scaled_key[:, 0]
        if state is None:
            state = value.new_zeros(value.shape)
        state = coefficient.unsqueeze(-1) * state + scaled_key.unsqueeze(-1) * value
        attended = query.unsqueeze(-1) * state
        return inputs + self.output(attended.reshape(inputs.shape[0], -1)), state


class ByteGatedDeltaNet(nn.Module):
    """Small diagonal Gated DeltaNet/Gated DeltaNet-2 reference variants."""

    def __init__(
        self,
        *,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 7,
        value_multiplier: int = 1,
        scan_chunk_size: int = 32,
        decoupled_gates: bool = False,
        bos_id: int = 256,
        output_size: int = 256,
        architecture_name: str = "byte-gated-deltanet-lite-v1",
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.value_multiplier = value_multiplier
        self.scan_chunk_size = scan_chunk_size
        self.decoupled_gates = decoupled_gates
        self.bos_id = bos_id
        self.output_size = output_size
        self.architecture_name = architecture_name
        self.embedding = nn.Embedding(output_size + 1, model_dim)
        self.layers = nn.ModuleList(
            [
                GatedDeltaBlock(
                    model_dim,
                    num_heads,
                    value_multiplier,
                    scan_chunk_size,
                    decoupled_gates=decoupled_gates,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, output_size)

    @property
    def model_config(self) -> dict[str, int | str]:
        return {
            "architecture": self.architecture_name,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "value_multiplier": self.value_multiplier,
            "scan_chunk_size": self.scan_chunk_size,
            "decoupled_gates": self.decoupled_gates,
            "bos_id": self.bos_id,
            "output_size": self.output_size,
        }

    def init_state(self, batch_size: int, device: str | torch.device) -> list[Any]:
        del batch_size, device
        return [None] * self.num_layers

    def forward(self, input_ids: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        del state
        sequence = self.embedding(input_ids)
        for layer in self.layers:
            sequence = layer(sequence)
        return self.output(self.final_norm(sequence)), None

    def step(self, input_ids: torch.Tensor, state: list[Any] | None = None) -> tuple[torch.Tensor, list[Any]]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        states = [None] * self.num_layers if state is None else state
        sequence = self.embedding(input_ids)[:, 0]
        next_states: list[Any] = []
        for layer, layer_state in zip(self.layers, states):
            sequence, layer_state = layer.step(sequence, layer_state)
            next_states.append(layer_state)
        return self.output(self.final_norm(sequence)), next_states


__all__ = [
    "ByteGatedDeltaNet",
    "ByteGriffin",
    "ByteLSTM",
    "ByteMamba",
    "ByteTransformer",
]
