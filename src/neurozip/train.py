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


def _resolve_amp_mode(requested: str, device: str) -> str:
    mode = str(requested or "off").lower()
    if mode == "auto":
        mode = "fp16" if device.startswith("cuda") else "off"
    if mode not in {"off", "fp16", "bf16"}:
        raise ValueError("amp must be one of: off, auto, fp16, bf16")
    if mode == "fp16" and not device.startswith("cuda"):
        raise ValueError("fp16 AMP requires a CUDA device")
    return mode


def _autocast_context(torch, *, device: str, amp_mode: str):
    if amp_mode == "auto":
        amp_mode = "fp16" if device.startswith("cuda") else "off"
    if amp_mode == "off":
        return nullcontext()
    dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype)


def _make_grad_scaler(torch, *, device: str, amp_mode: str):
    enabled = amp_mode == "fp16" and device.startswith("cuda")
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):  # pragma: no cover - older torch fallback.
        return torch.cuda.amp.GradScaler(enabled=True)


def evaluate(
    model,
    data,
    *,
    sequence_length: int,
    max_bytes: int,
    device: str,
    amp_mode: str = "off",
) -> dict[str, float]:
    torch, F = _require_torch()
    model.eval()
    total_loss = 0.0
    total_bytes = 0
    with torch.inference_mode(), _autocast_context(torch, device=device, amp_mode=amp_mode):
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
                    logits.float().reshape(-1, model.output_size),
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
    extra_state: dict[str, Any] | None = None,
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
    if extra_state is not None:
        payload["resume_state"] = extra_state
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
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


def _load_torch_checkpoint(torch, path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch fallback.
        return torch.load(path, map_location="cpu")


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A process killed while appending should not make a later
                # resume impossible; the last complete JSON row is enough.
                continue
    return rows


def _distributed_generator_states(generator, *, context: DistributedContext):
    torch, _ = _require_torch()
    state = generator.get_state().cpu()
    if not context.enabled:
        return [state]
    states: list[Any] = [None] * context.world_size
    torch.distributed.all_gather_object(states, state)
    return states


def _optimizer_to_device(optimizer, *, device: str) -> None:
    torch, _ = _require_torch()
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _clear_training_artifacts(output_dir: Path) -> None:
    for name in ("best.pt", "last.pt", "metrics.jsonl", "summary.json", "run_config.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()
    for path in output_dir.glob(".*.pt.tmp"):
        path.unlink()


def _train_impl(
    args: argparse.Namespace, *, context: DistributedContext
) -> dict[str, Any] | None:
    torch, F = _require_torch()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.batch_size <= 0 or args.sequence_length <= 0:
        raise ValueError("batch size and sequence length must be positive")
    accumulation_steps = int(getattr(args, "gradient_accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    device = context.device
    amp_mode = _resolve_amp_mode(getattr(args, "amp", "off"), device)
    resume_requested = bool(getattr(args, "resume", False))
    restart_requested = bool(getattr(args, "restart", False))
    if resume_requested and restart_requested:
        raise ValueError("--resume and --restart are mutually exclusive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if restart_requested and context.is_main:
        _clear_training_artifacts(output_dir)
    distributed_barrier(context)

    seed_everything(args.seed)
    train_data = load_byte_tensor(args.train_path)
    valid_data = load_byte_tensor(args.valid_path)
    model = build_model_from_args(args).to(device)
    model_config = model.model_config

    resume_path: Path | None = None
    resume_checkpoint: dict[str, Any] | None = None
    if resume_requested:
        for candidate in (output_dir / "last.pt", output_dir / "best.pt"):
            if candidate.exists():
                resume_path = candidate
                try:
                    resume_checkpoint = _load_torch_checkpoint(torch, candidate)
                except Exception as exc:
                    raise RuntimeError(f"could not load resume checkpoint {candidate}: {exc}") from exc
                break
    if resume_checkpoint is not None:
        if resume_checkpoint.get("model_config") != model_config:
            raise RuntimeError(
                "resume checkpoint model configuration does not match the requested architecture"
            )
        model.load_state_dict(resume_checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = _make_grad_scaler(torch, device=device, amp_mode=amp_mode)
    resume_state = (resume_checkpoint or {}).get("resume_state") or {}
    if resume_checkpoint is not None and resume_checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device=device)
    if scaler is not None and resume_state.get("scaler_state_dict"):
        scaler.load_state_dict(resume_state["scaler_state_dict"])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1_000_003 * context.rank)
    generator_states = resume_state.get("generator_states")
    if generator_states is None and resume_state.get("generator_state") is not None:
        generator_states = [resume_state["generator_state"]]
    if generator_states:
        state_index = min(context.rank, len(generator_states) - 1)
        generator.set_state(generator_states[state_index].cpu())

    best_model_id = resume_state.get("best_model_id")
    if best_model_id is None and (output_dir / "best.pt").exists():
        try:
            best_model_id = _load_torch_checkpoint(torch, output_dir / "best.pt").get("model_id")
        except Exception:
            best_model_id = None

    train_model = model
    if context.enabled:
        ddp_kwargs: dict[str, Any] = {}
        if device.startswith("cuda"):
            ddp_kwargs = {
                "device_ids": [context.local_rank],
                "output_device": context.local_rank,
            }
        train_model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    train_config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "func":
            continue
        train_config[key] = str(value) if isinstance(value, Path) else value
    train_config["resolved_device"] = device
    train_config["distributed"] = context.enabled
    train_config["rank"] = context.rank
    train_config["world_size"] = context.world_size
    train_config["local_rank"] = context.local_rank
    train_config["batch_size_per_gpu"] = args.batch_size
    train_config["gradient_accumulation_steps"] = accumulation_steps
    train_config["effective_batch_size"] = args.batch_size * accumulation_steps * context.world_size
    train_config["amp"] = amp_mode
    train_config["train_bytes"] = len(train_data)
    train_config["validation_bytes"] = len(valid_data)
    train_config["parameter_count"] = parameter_count(model)
    train_config["architecture"] = args.architecture
    train_config["model_config"] = model_config
    train_config["training_bytes_budget"] = (
        args.steps * args.batch_size * accumulation_steps * args.sequence_length * context.world_size
    )
    if resume_path is not None:
        train_config["resumed_from"] = str(resume_path)
        train_config["resumed_from_step"] = int((resume_checkpoint or {}).get("step", 0))
    if context.is_main:
        _atomic_write_json(output_dir / "run_config.json", {"model_config": model_config, "train_config": train_config})
    distributed_barrier(context)

    metrics_path = output_dir / "metrics.jsonl"
    metric_rows = _read_metric_rows(metrics_path)
    last_metrics: dict[str, Any] | None = metric_rows[-1] if metric_rows else None
    row_best_bpb = min(
        (float(row["validation_bpb"]) for row in metric_rows if row.get("validation_bpb") is not None),
        default=float("inf"),
    )
    saved_best_bpb = resume_state.get("best_validation_bpb")
    best_bpb = float(saved_best_bpb) if saved_best_bpb is not None else row_best_bpb
    if not math.isfinite(best_bpb):
        checkpoint_metrics = (resume_checkpoint or {}).get("metrics") or {}
        checkpoint_bpb = checkpoint_metrics.get("validation_bpb")
        best_bpb = float(checkpoint_bpb) if checkpoint_bpb is not None else float("inf")
    elapsed_offset = float(resume_state.get("elapsed_seconds", 0.0))
    start_step = int((resume_checkpoint or {}).get("step", 0)) + 1
    started = time.perf_counter() - elapsed_offset
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    metrics_mode = "a" if resume_checkpoint is not None and metrics_path.exists() else "w"
    metrics_context = metrics_path.open(metrics_mode, encoding="utf-8") if context.is_main else nullcontext()
    with metrics_context as metrics_file:
        for step in range(start_step, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for micro_step in range(accumulation_steps):
                inputs, targets = sample_batch(
                    train_data,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    generator=generator,
                    device=device,
                )
                if context.enabled and micro_step < accumulation_steps - 1:
                    sync_context = train_model.no_sync()
                else:
                    sync_context = nullcontext()
                with sync_context:
                    with _autocast_context(torch, device=device, amp_mode=amp_mode):
                        logits, _ = train_model(inputs)
                        micro_loss = F.cross_entropy(
                            logits.float().reshape(-1, model.output_size),
                            targets.reshape(-1),
                            reduction="mean",
                        )
                    accumulated_loss += float(micro_loss.detach().item())
                    scaled_loss = micro_loss / accumulation_steps
                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            should_log = step == start_step or step % args.eval_every == 0 or step == args.steps
            if should_log:
                train_loss = distributed_mean(accumulated_loss / accumulation_steps, context=context)
                if context.is_main:
                    validation = evaluate(
                        model,
                        valid_data,
                        sequence_length=args.sequence_length,
                        max_bytes=args.validation_eval_bytes,
                        device=device,
                        amp_mode=amp_mode,
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
                generator_states = _distributed_generator_states(generator, context=context)
                if context.is_main:
                    assert last_metrics is not None
                    if validation["validation_bpb"] < best_bpb:
                        best_bpb = validation["validation_bpb"]
                        best_model_id = save_checkpoint(
                            output_dir / "best.pt",
                            model=model,
                            model_config=model_config,
                            train_config=train_config,
                            step=step,
                            metrics=last_metrics,
                        )
                        print(json.dumps({"best_checkpoint": str(output_dir / "best.pt"), "model_id": best_model_id}))
                    resume_payload = {
                        "best_validation_bpb": best_bpb,
                        "best_model_id": best_model_id,
                        "elapsed_seconds": last_metrics["elapsed_seconds"],
                        "generator_states": generator_states,
                        "gradient_accumulation_steps": accumulation_steps,
                        "amp": amp_mode,
                    }
                    if scaler is not None:
                        resume_payload["scaler_state_dict"] = scaler.state_dict()
                    save_checkpoint(
                        output_dir / "last.pt",
                        model=model,
                        model_config=model_config,
                        train_config=train_config,
                        step=step,
                        metrics=last_metrics,
                        optimizer=optimizer,
                        extra_state=resume_payload,
                    )
                    print(json.dumps(last_metrics, sort_keys=True))
                distributed_barrier(context)

    elapsed_seconds = time.perf_counter() - started
    peak_gpu_memory = 0
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        peak_gpu_memory = distributed_max(
            int(torch.cuda.max_memory_allocated(device)), context=context
        )
    peak_cpu_memory = process_max_rss_bytes()
    training_bytes = (
        args.steps * args.batch_size * accumulation_steps * args.sequence_length * context.world_size
    )
    summary = {
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_validation_bpb": best_bpb if math.isfinite(best_bpb) else None,
        "model_id": best_model_id,
        "parameter_count": parameter_count(model),
        "total_steps": args.steps,
        "completed_steps": args.steps if start_step <= args.steps else min(start_step - 1, args.steps),
        "wall_time_seconds": elapsed_seconds,
        "training_bytes": training_bytes,
        "training_bytes_per_second": training_bytes / elapsed_seconds if elapsed_seconds else None,
        "steps_per_second": args.steps / elapsed_seconds if elapsed_seconds else None,
        "batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": args.batch_size * accumulation_steps * context.world_size,
        "amp": amp_mode,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "peak_cpu_memory_bytes": peak_cpu_memory,
        "last_metrics": last_metrics,
    }
    if context.is_main:
        _atomic_write_json(output_dir / "summary.json", summary)
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
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="microbatches per optimizer step; effective batch is batch-size * this * world-size",
    )
    parser.add_argument(
        "--amp",
        choices=["off", "auto", "fp16", "bf16"],
        default="off",
        help="optional autocast mode; fp16 uses CUDA GradScaler",
    )
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--validation-eval-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from output-dir/last.pt, or best.pt when last.pt is unavailable",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="remove only this run's checkpoints and metrics before starting fresh",
    )
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
