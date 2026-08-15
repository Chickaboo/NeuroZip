"""Train the NeuroZip V0 byte GRU and export usable checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - only used outside training envs.
        raise RuntimeError("PyTorch is required to train NeuroZip V0") from exc
    return torch, F


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
    from .models.gru import model_id_from_state

    state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    model_id = model_id_from_state(model_config, state_dict)
    payload: dict[str, Any] = {
        "format": "neurozip-gru-checkpoint-v1",
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


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch, F = _require_torch()
    from .models.gru import ByteGRU

    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.batch_size <= 0 or args.sequence_length <= 0:
        raise ValueError("batch size and sequence length must be positive")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")

    seed_everything(args.seed)
    train_data = load_byte_tensor(args.train_path)
    valid_data = load_byte_tensor(args.valid_path)
    model = ByteGRU(
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = model.model_config
    train_config = {}
    for key, value in vars(args).items():
        if key == "func":
            continue
        train_config[key] = str(value) if isinstance(value, Path) else value
    train_config["resolved_device"] = device
    train_config["train_bytes"] = len(train_data)
    train_config["validation_bytes"] = len(valid_data)
    train_config["parameter_count"] = parameter_count(model)
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
    started = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            inputs, targets = sample_batch(
                train_data,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                generator=generator,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, model.output_size), targets.reshape(-1), reduction="mean"
            )
            loss.backward()
            if args.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            should_log = step == 1 or step % args.eval_every == 0 or step == args.steps
            if should_log:
                validation = evaluate(
                    model,
                    valid_data,
                    sequence_length=args.sequence_length,
                    max_bytes=args.validation_eval_bytes,
                    device=device,
                )
                row = {
                    "step": step,
                    "train_loss_nats_per_byte": float(loss.detach().item()),
                    "train_bpb_estimate": float(loss.detach().item() / math.log(2.0)),
                    **validation,
                    "elapsed_seconds": time.perf_counter() - started,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
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

    summary = {
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_validation_bpb": best_bpb,
        "model_id": best_model_id,
        "parameter_count": parameter_count(model),
        "total_steps": args.steps,
        "wall_time_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--valid-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--validation-eval-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.device == "auto":
        torch, _ = _require_torch()
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = train(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
