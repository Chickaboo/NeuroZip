"""Train a NeuroZip byte sequence model and export usable checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - only used outside training envs.
        raise RuntimeError("PyTorch is required to train NeuroZip V0") from exc
    return torch, F


@dataclass(frozen=True)
class DistributedContext:
    """Runtime information for single-process or DDP training."""

    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(
    requested_device: str, *, local_rank_override: int | None = None
) -> DistributedContext:
    """Initialize torch.distributed when launched by torchrun."""

    torch, _ = _require_torch()
    resolved_device = requested_device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            str(local_rank_override if local_rank_override is not None else 0),
        )
    )
    if world_size <= 1:
        return DistributedContext(
            enabled=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=resolved_device,
        )

    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is required for multi-GPU training")
    if resolved_device == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"local rank {local_rank} is not a visible CUDA device "
                f"(device count: {torch.cuda.device_count()})"
            )
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = "cpu"
        backend = "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def teardown_distributed(context: DistributedContext) -> None:
    torch, _ = _require_torch()
    if context.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def distributed_barrier(context: DistributedContext) -> None:
    if context.enabled:
        torch, _ = _require_torch()
        torch.distributed.barrier()


def distributed_mean(value: float, *, context: DistributedContext) -> float:
    """Average a scalar across ranks for one global training metric."""

    if not context.enabled:
        return value
    torch, _ = _require_torch()
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    torch.distributed.all_reduce(tensor)
    return float(tensor.item() / context.world_size)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch, _ = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_byte_tensor(path: Path):
    torch, _ = _require_torch()
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"training/validation file is empty: {path}")
    # bytearray makes the backing buffer writable, avoiding a warning in
    # torch.frombuffer while keeping the corpus at one byte per element.
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


def sample_batch(data, *, batch_size: int, sequence_length: int, generator, device: str):
    torch, _ = _require_torch()
    if len(data) < sequence_length:
        raise ValueError("corpus must contain at least sequence_length bytes")
    starts = torch.randint(
        0,
        len(data) - sequence_length + 1,
        (batch_size,),
        generator=generator,
        dtype=torch.long,
    )
    offsets = torch.arange(sequence_length, dtype=torch.long)
    targets = data[starts[:, None] + offsets].long()
    inputs = torch.empty_like(targets)
    inputs[:, 0] = 256  # BOS
    if sequence_length > 1:
        inputs[:, 1:] = targets[:, :-1]
    return inputs.to(device), targets.to(device)


def evaluate(model, data, *, sequence_length: int, max_bytes: int, device: str) -> dict[str, float]:
    torch, F = _require_torch()
    model.eval()
    total_loss = 0.0
    total_bytes = 0
    with torch.inference_mode():
        limit = min(len(data), max_bytes)
        start = 0
        while start < limit:
            end = min(start + sequence_length, limit)
            target_slice = data[start:end]
            if len(target_slice) == 0:
                break
            targets = target_slice.long().unsqueeze(0).to(device)
            inputs = torch.empty_like(targets)
            inputs[:, 0] = 256
            if len(target_slice) > 1:
                inputs[:, 1:] = targets[:, :-1]
            logits, _ = model(inputs)
            total_loss += float(
                F.cross_entropy(
                    logits.reshape(-1, model.output_size),
                    targets.reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_bytes += len(target_slice)
            start = end
    model.train()
    if total_bytes == 0:
        raise ValueError("validation corpus produced no bytes")
    nats_per_byte = total_loss / total_bytes
    return {
        "validation_loss_nats_per_byte": nats_per_byte,
        "validation_bpb": nats_per_byte / math.log(2.0),
        "validation_bytes": total_bytes,
    }


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def process_max_rss_bytes() -> int:
    """Return the process high-water RSS in bytes on Linux/macOS."""

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. Kaggle and the supported local
    # Toolbx are Linux, but keeping the branch makes the metric portable.
    return int(value * (1 if os.uname().sysname == "Darwin" else 1024))


def distributed_max(value: int, *, context: DistributedContext) -> int:
    if not context.enabled:
        return value
    torch, _ = _require_torch()
    tensor = torch.tensor(value, dtype=torch.int64, device=context.device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return int(tensor.item())


def save_checkpoint(
    path: Path,
    *,
    model,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    step: int,
    metrics: dict[str, Any],
    optimizer=None,
) -> str:
    torch, _ = _require_torch()
    from .models.registry import model_id_from_state

    state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    model_id = model_id_from_state(model_config, state_dict)
    payload: dict[str, Any] = {
        "format": (
            "neurozip-gru-checkpoint-v1"
            if model_config.get("architecture") == "byte-gru-v0"
            else "neurozip-sequence-checkpoint-v1"
        ),
        "model_id": model_id,
        "model_config": model_config,
        "train_config": train_config,
        "step": step,
        "metrics": metrics,
        "model_state_dict": state_dict,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return model_id


def train(args: argparse.Namespace) -> dict[str, Any] | None:
    """Train on one process or on all processes launched by torchrun."""

    context = setup_distributed(
        args.device, local_rank_override=getattr(args, "local_rank", None)
    )
    try:
        return _train_impl(args, context=context)
    finally:
        teardown_distributed(context)


def build_model_from_args(args: argparse.Namespace):
    """Construct a sweep model while keeping the CLI backward compatible."""

    from .models.registry import build_model

    common = {
        "bos_id": 256,
        "output_size": 256,
        "num_layers": args.num_layers,
    }
    if args.architecture in {"gru", "lstm"}:
        return build_model(
            args.architecture,
            **common,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
        )
    if args.architecture == "transformer":
        return build_model(
            args.architecture,
            **common,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            context_length=args.context_length,
        )
    if args.architecture in {"mamba", "griffin"}:
        return build_model(
            args.architecture,
            **common,
            model_dim=args.model_dim,
            inner_dim=args.inner_dim,
            conv_kernel=args.conv_kernel,
            scan_chunk_size=args.scan_chunk_size,
        )
    if args.architecture in {"gated-deltanet", "gated-deltanet2"}:
        return build_model(
            args.architecture,
            **common,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            value_multiplier=args.value_multiplier,
            scan_chunk_size=args.scan_chunk_size,
        )
    raise ValueError(f"unsupported architecture: {args.architecture}")


def _train_impl(
    args: argparse.Namespace, *, context: DistributedContext
) -> dict[str, Any] | None:
    torch, F = _require_torch()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.batch_size <= 0 or args.sequence_length <= 0:
        raise ValueError("batch size and sequence length must be positive")
    device = context.device

    seed_everything(args.seed)
    train_data = load_byte_tensor(args.train_path)
    valid_data = load_byte_tensor(args.valid_path)
    model = build_model_from_args(args).to(device)
    train_model = model
    if context.enabled:
        ddp_kwargs: dict[str, Any] = {}
        if device.startswith("cuda"):
            ddp_kwargs = {
                "device_ids": [context.local_rank],
                "output_device": context.local_rank,
            }
        train_model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1_000_003 * context.rank)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = model.model_config
    train_config = {}
    for key, value in vars(args).items():
        if key == "func":
            continue
        train_config[key] = str(value) if isinstance(value, Path) else value
    train_config["resolved_device"] = device
    train_config["distributed"] = context.enabled
    train_config["rank"] = context.rank
    train_config["world_size"] = context.world_size
    train_config["local_rank"] = context.local_rank
    train_config["effective_batch_size"] = args.batch_size * context.world_size
    train_config["train_bytes"] = len(train_data)
    train_config["validation_bytes"] = len(valid_data)
    train_config["parameter_count"] = parameter_count(model)
    train_config["architecture"] = args.architecture
    train_config["model_config"] = model_config
    if context.is_main:
        (output_dir / "run_config.json").write_text(
            json.dumps(
                {"model_config": model_config, "train_config": train_config},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    metrics_path = output_dir / "metrics.jsonl"
    best_bpb = float("inf")
    best_model_id = None
    last_metrics: dict[str, Any] | None = None
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    metrics_context = metrics_path.open("w", encoding="utf-8") if context.is_main else nullcontext()
    with metrics_context as metrics_file:
        for step in range(1, args.steps + 1):
            inputs, targets = sample_batch(
                train_data,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                generator=generator,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, _ = train_model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, model.output_size), targets.reshape(-1), reduction="mean"
            )
            loss.backward()
            if args.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            should_log = step == 1 or step % args.eval_every == 0 or step == args.steps
            if should_log:
                train_loss = distributed_mean(float(loss.detach().item()), context=context)
                if context.is_main:
                    validation = evaluate(
                        model,
                        valid_data,
                        sequence_length=args.sequence_length,
                        max_bytes=args.validation_eval_bytes,
                        device=device,
                    )
                    row = {
                        "step": step,
                        "train_loss_nats_per_byte": train_loss,
                        "train_bpb_estimate": train_loss / math.log(2.0),
                        **validation,
                        "elapsed_seconds": time.perf_counter() - started,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    last_metrics = row
                    metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                    metrics_file.flush()
                    print(json.dumps(row, sort_keys=True))

                    model_id = save_checkpoint(
                        output_dir / "last.pt",
                        model=model,
                        model_config=model_config,
                        train_config=train_config,
                        step=step,
                        metrics=row,
                        optimizer=optimizer,
                    )
                    if validation["validation_bpb"] < best_bpb:
                        best_bpb = validation["validation_bpb"]
                        best_model_id = save_checkpoint(
                            output_dir / "best.pt",
                            model=model,
                            model_config=model_config,
                            train_config=train_config,
                            step=step,
                            metrics=row,
                        )
                        print(json.dumps({"best_checkpoint": str(output_dir / "best.pt"), "model_id": best_model_id}))
                distributed_barrier(context)

    elapsed_seconds = time.perf_counter() - started
    peak_gpu_memory = 0
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        peak_gpu_memory = distributed_max(
            int(torch.cuda.max_memory_allocated(device)), context=context
        )
    peak_cpu_memory = process_max_rss_bytes()
    training_bytes = args.steps * args.batch_size * args.sequence_length * context.world_size
    summary = {
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_validation_bpb": best_bpb,
        "model_id": best_model_id,
        "parameter_count": parameter_count(model),
        "total_steps": args.steps,
        "wall_time_seconds": elapsed_seconds,
        "training_bytes": training_bytes,
        "training_bytes_per_second": training_bytes / elapsed_seconds if elapsed_seconds else None,
        "steps_per_second": args.steps / elapsed_seconds if elapsed_seconds else None,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "peak_cpu_memory_bytes": peak_cpu_memory,
        "last_metrics": last_metrics,
    }
    if context.is_main:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    distributed_barrier(context)
    return summary if context.is_main else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--valid-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=["gru", "lstm", "transformer", "mamba", "griffin", "gated-deltanet", "gated-deltanet2"],
        default="gru",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=896)
    parser.add_argument("--inner-dim", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--scan-chunk-size", type=int, default=32)
    parser.add_argument("--value-multiplier", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--validation-eval-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = train(args)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
