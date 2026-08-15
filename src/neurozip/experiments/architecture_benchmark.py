"""Benchmark trained NeuroZip architectures through the real codec.

This runner intentionally treats exact decoding as a hard gate.  A model with
good cross-entropy but a failed byte comparison or SHA-256 check is reported as
``FAILED`` and excluded from the trade-off recommendation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch

from ..codec import compress_bytes, decompress_bytes
from ..file_format import parse_container
from ..predictors import NeuralPredictor


def process_max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1 if os.uname().sysname == "Darwin" else 1024))


class LossTrackingPredictor:
    """Delegate coding to a real predictor while recording float-model NLL."""

    def __init__(self, predictor: NeuralPredictor) -> None:
        self.predictor = predictor
        self.model_id = predictor.model_id
        self.nll_bits = 0.0

    def reset(self) -> None:
        self.nll_bits = 0.0
        self.predictor.reset()

    def cdf(self, cdf_bits: int) -> tuple[int, ...]:
        return self.predictor.cdf(cdf_bits)

    def update(self, symbol: int) -> None:
        logits = self.predictor._logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        self.nll_bits += float((-log_probs[int(symbol)] / math.log(2.0)).item())
        self.predictor.update(symbol)


def _checkpoint_metrics(artifact_dir: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    metrics_path = artifact_dir / "metrics.jsonl"
    rows = []
    if metrics_path.exists():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    best_metrics = checkpoint.get("metrics") or {}
    last_metrics = rows[-1] if rows else {}
    return {
        "train_loss_nats_per_byte": best_metrics.get("train_loss_nats_per_byte"),
        "train_bpb": best_metrics.get("train_bpb_estimate"),
        "validation_loss_nats_per_byte": best_metrics.get("validation_loss_nats_per_byte"),
        "validation_bpb": best_metrics.get("validation_bpb"),
        "best_validation_bpb": best_metrics.get("validation_bpb"),
        "last_train_bpb": last_metrics.get("train_bpb_estimate"),
        "last_validation_bpb": last_metrics.get("validation_bpb"),
        "logged_steps": len(rows),
    }


def _state_dict_bytes(state_dict: dict[str, Any]) -> int:
    return sum(int(value.numel() * value.element_size()) for value in state_dict.values())


def _reset_gpu_peak(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _gpu_peak_bytes(device: str) -> int:
    if not device.startswith("cuda"):
        return 0
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def generate_sample(
    checkpoint_path: Path,
    *,
    device: str,
    prompt: bytes = b"The ",
    length: int = 96,
) -> dict[str, str]:
    """Greedily generate a short byte sample for a qualitative sanity check."""

    predictor = NeuralPredictor.from_checkpoint(str(checkpoint_path), device=device)
    predictor.reset()
    for symbol in prompt:
        predictor.update(symbol)
    generated = bytearray(prompt)
    for _ in range(length):
        symbol = int(torch.argmax(predictor._logits).item())
        generated.append(symbol)
        predictor.update(symbol)
    return {
        "sample_hex": bytes(generated).hex(),
        "sample_text": bytes(generated).decode("utf-8", errors="replace"),
    }


def _base_result(artifact_dir: Path, checkpoint_path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "architecture": artifact_dir.name,
        "status": "FAILED",
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "exact_round_trip": False,
        "byte_identical": False,
        "sha256_identical": False,
    }
    run_config_path = artifact_dir / "run_config.json"
    summary_path = artifact_dir / "summary.json"
    if run_config_path.exists():
        try:
            config = json.loads(run_config_path.read_text())
            result["model_config"] = config.get("model_config")
            result["train_config"] = config.get("train_config")
        except (OSError, json.JSONDecodeError) as exc:
            result["metadata_failure"] = f"run_config.json: {type(exc).__name__}: {exc}"
    if summary_path.exists():
        try:
            result.update(
                {f"training_{key}": value for key, value in json.loads(summary_path.read_text()).items()}
            )
        except (OSError, json.JSONDecodeError) as exc:
            result["metadata_failure"] = f"summary.json: {type(exc).__name__}: {exc}"
    return result


def _missing_checkpoint_result(artifact_dir: Path) -> dict[str, Any]:
    """Keep an interrupted architecture visible in the final comparison."""

    result = _base_result(artifact_dir, None)
    failure_path = artifact_dir / "failure.json"
    if failure_path.exists():
        try:
            failure = json.loads(failure_path.read_text())
            result["failure_reason"] = failure.get("failure_reason", "training did not produce best.pt")
            result["training_failure"] = failure
        except (OSError, json.JSONDecodeError):
            result["failure_reason"] = "training did not produce a readable best.pt checkpoint"
    else:
        result["failure_reason"] = "training did not produce best.pt"
    return result


def benchmark_checkpoint(
    checkpoint_path: Path,
    original: bytes,
    *,
    device: str,
    cdf_bits: int,
    output_dir: Path,
    sample_prompt: bytes,
    sample_length: int,
) -> dict[str, Any]:
    artifact_dir = checkpoint_path.parent
    result = _base_result(artifact_dir, checkpoint_path)
    try:
        predictor = NeuralPredictor.from_checkpoint(str(checkpoint_path), device=device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        result["model_id"] = predictor.model_id
        result["parameter_count"] = sum(int(value.numel()) for value in checkpoint["model_state_dict"].values())
        result["model_state_dict_bytes"] = _state_dict_bytes(checkpoint["model_state_dict"])
        result["checkpoint_bytes"] = checkpoint_path.stat().st_size
        result.update(_checkpoint_metrics(artifact_dir, checkpoint))

        tracker = LossTrackingPredictor(predictor)
        _reset_gpu_peak(device)
        started = time.perf_counter()
        container = compress_bytes(original, tracker, cdf_bits=cdf_bits)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        encode_seconds = time.perf_counter() - started
        encode_gpu_peak = _gpu_peak_bytes(device)

        decoder_predictor = NeuralPredictor.from_checkpoint(str(checkpoint_path), device=device)
        _reset_gpu_peak(device)
        started = time.perf_counter()
        restored = decompress_bytes(
            container,
            decoder_predictor,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        decode_seconds = time.perf_counter() - started
        decode_gpu_peak = _gpu_peak_bytes(device)

        header, payload = parse_container(container)
        original_sha = hashlib.sha256(original).hexdigest()
        restored_sha = hashlib.sha256(restored).hexdigest()
        byte_identical = original == restored
        sha_identical = original_sha == restored_sha
        exact = byte_identical and sha_identical
        stream_dir = output_dir / "streams"
        stream_dir.mkdir(parents=True, exist_ok=True)
        stream_path = stream_dir / f"{artifact_dir.name}.nz"
        restored_path = stream_dir / f"{artifact_dir.name}.restored"
        stream_path.write_bytes(container)
        restored_path.write_bytes(restored)

        result.update(
            {
                "status": "PASSED" if exact else "FAILED",
                "source_bytes": len(original),
                "compressed_bytes": len(container),
                "payload_bytes": len(payload),
                "format_overhead_bytes": len(container) - len(payload),
                "format_overhead_bits": (len(container) - len(payload)) * 8,
                "actual_compressed_bpb": len(container) * 8 / len(original) if original else None,
                "payload_bpb": len(payload) * 8 / len(original) if original else None,
                "predicted_model_bpb": tracker.nll_bits / len(original) if original else None,
                "compression_ratio": len(original) / len(container) if container else None,
                "encode_seconds": encode_seconds,
                "decode_seconds": decode_seconds,
                "encode_bytes_per_second": len(original) / encode_seconds if encode_seconds else None,
                "decode_bytes_per_second": len(original) / decode_seconds if decode_seconds else None,
                "encode_peak_gpu_memory_bytes": encode_gpu_peak,
                "decode_peak_gpu_memory_bytes": decode_gpu_peak,
                "peak_process_rss_bytes": process_max_rss_bytes(),
                "byte_identical": byte_identical,
                "sha256_original": original_sha,
                "sha256_restored": restored_sha,
                "sha256_identical": sha_identical,
                "exact_round_trip": exact,
                "container_model_id": header.model_id,
                "cdf_bits": header.cdf_bits,
                "stream_path": str(stream_path),
                "restored_path": str(restored_path),
                **generate_sample(
                    checkpoint_path,
                    device=device,
                    prompt=sample_prompt,
                    length=sample_length,
                ),
            }
        )
        return result
    except Exception as exc:  # Every architecture remains visible in the table.
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return result


def _harmonic_mean(left: float | None, right: float | None) -> float | None:
    if not left or not right or left <= 0 or right <= 0:
        return None
    return 2.0 / (1.0 / left + 1.0 / right)


def score_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Add a transparent compression/compute score relative to the GRU."""

    passed = [row for row in results if row.get("exact_round_trip")]
    baseline = next((row for row in passed if row["architecture"] == "gru"), None)
    if baseline:
        baseline_speed = _harmonic_mean(
            baseline.get("encode_bytes_per_second"), baseline.get("decode_bytes_per_second")
        ) or 0.0
        baseline_memory = max(
            baseline.get("training_peak_gpu_memory_bytes", 0) or 0,
            baseline.get("encode_peak_gpu_memory_bytes", 0) or 0,
            baseline.get("decode_peak_gpu_memory_bytes", 0) or 0,
            1,
        )
        baseline_model_size = max(baseline.get("checkpoint_bytes", 0) or 0, 1)
        baseline_bpb = baseline.get("actual_compressed_bpb") or 0.0
        for row in results:
            speed = _harmonic_mean(row.get("encode_bytes_per_second"), row.get("decode_bytes_per_second"))
            memory = max(
                row.get("training_peak_gpu_memory_bytes", 0) or 0,
                row.get("encode_peak_gpu_memory_bytes", 0) or 0,
                row.get("decode_peak_gpu_memory_bytes", 0) or 0,
                1,
            )
            model_size = max(row.get("checkpoint_bytes", 0) or 0, 1)
            if row.get("exact_round_trip") and speed and row.get("actual_compressed_bpb"):
                compression_factor = baseline_bpb / row["actual_compressed_bpb"]
                speed_factor = speed / baseline_speed if baseline_speed else 1.0
                memory_factor = baseline_memory / memory
                model_size_factor = baseline_model_size / model_size
                row["encode_decode_hmean_bytes_per_second"] = speed
                row["tradeoff_score"] = compression_factor * (
                    max(speed_factor, 0.0) * max(memory_factor, 0.0) * max(model_size_factor, 0.0)
                ) ** (1.0 / 3.0)
            else:
                row["tradeoff_score"] = None
    recommendation = None
    if passed:
        ranked = sorted(
            passed,
            key=lambda row: (
                row.get("tradeoff_score") is not None,
                row.get("tradeoff_score") or float("-inf"),
                -(row.get("actual_compressed_bpb") or float("inf")),
            ),
            reverse=True,
        )
        recommendation = ranked[0]["architecture"]
    return {
        "baseline": "gru",
        "passed_architectures": [row["architecture"] for row in passed],
        "failed_architectures": [row["architecture"] for row in results if not row.get("exact_round_trip")],
        "recommended_architecture": recommendation,
        "score_definition": "(GRU actual BPB / candidate actual BPB) * cube_root(speed factor * peak-memory factor * checkpoint-size factor); failed exact round trips are excluded",
    }


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else f"{value:,}"
    return str(value)


def write_comparison_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    columns = [
        "architecture",
        "status",
        "train_loss_nats_per_byte",
        "train_bpb",
        "validation_loss_nats_per_byte",
        "validation_bpb",
        "actual_compressed_bpb",
        "payload_bpb",
        "compression_ratio",
        "parameter_count",
        "checkpoint_bytes",
        "training_wall_time_seconds",
        "training_bytes_per_second",
        "encode_bytes_per_second",
        "decode_bytes_per_second",
        "training_peak_gpu_memory_bytes",
        "training_peak_cpu_memory_bytes",
        "encode_peak_gpu_memory_bytes",
        "decode_peak_gpu_memory_bytes",
        "peak_process_rss_bytes",
        "byte_identical",
        "sha256_identical",
        "exact_round_trip",
        "tradeoff_score",
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])

    lines = [
        "# NeuroZip architecture comparison",
        "",
        "## Technical summary",
        "",
        f"Recommended architecture: **{report['recommendation']['recommended_architecture'] or 'none'}**. "
        "The recommendation excludes any model that fails exact byte and SHA-256 reconstruction and balances actual full-stream BPB, "
        "encode/decode throughput, checkpoint size, and peak GPU memory relative to the GRU baseline.",
        "",
        "## Metric definitions and scope",
        "",
        "All rows use the same raw-byte representation, BOS token, WikiText-103 train/validation split, training steps, per-GPU batch size, "
        "sequence length, CDF precision, and NeuroZip integer range coder. `actual_compressed_bpb` includes the NeuroZip container header, "
        "model identifier, checksum, and coder payload; `payload_bpb` excludes the shared format overhead. The held-out input is the "
        "configured prefix of `validation.raw` and remains disjoint from the training slice. Batched validation resets at each evaluation "
        "window; the held-out codec run carries predictor state continuously from the first byte to the last, matching real compression.",
        "",
        "## Comparison table",
        "",
        "| Architecture | Status | Train loss | Train BPB | Valid loss | Valid BPB | Actual BPB | Payload BPB | Ratio | Params | Checkpoint | Train s | Train B/s | Encode B/s | Decode B/s | Peak GPU MB | Peak RSS MB | Exact | Score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in report["results"]:
        peak_gpu = max(
            row.get("training_peak_gpu_memory_bytes", 0) or 0,
            row.get("encode_peak_gpu_memory_bytes", 0) or 0,
            row.get("decode_peak_gpu_memory_bytes", 0) or 0,
        ) / (1024 * 1024)
        peak_rss = (row.get("peak_process_rss_bytes") or 0) / (1024 * 1024)
        lines.append(
            "| {architecture} | {status} | {train_loss} | {train} | {valid_loss} | {valid} | {actual} | {payload} | {ratio} | {params} | {checkpoint} | {train_s} | {train_bps} | {enc} | {dec} | {mem} | {rss} | {exact} | {score} |".format(
                architecture=row.get("architecture", "—"),
                status=row.get("status", "FAILED"),
                train_loss=_format(row.get("train_loss_nats_per_byte")),
                train=_format(row.get("train_bpb")),
                valid_loss=_format(row.get("validation_loss_nats_per_byte")),
                valid=_format(row.get("validation_bpb")),
                actual=_format(row.get("actual_compressed_bpb")),
                payload=_format(row.get("payload_bpb")),
                ratio=_format(row.get("compression_ratio")),
                params=_format(row.get("parameter_count"), 0),
                checkpoint=_format((row.get("checkpoint_bytes") or 0) / (1024 * 1024), 2) + " MB",
                train_s=_format(row.get("training_wall_time_seconds")),
                train_bps=_format(row.get("training_bytes_per_second"), 0),
                enc=_format(row.get("encode_bytes_per_second"), 0),
                dec=_format(row.get("decode_bytes_per_second"), 0),
                mem=_format(peak_gpu, 1),
                rss=_format(peak_rss, 1),
                exact=_format(row.get("exact_round_trip")),
                score=_format(row.get("tradeoff_score")),
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation and caveats",
            "",
            f"{report['recommendation']['score_definition']}.",
            "Validation BPB is a useful predictive diagnostic, but it is not the final decision metric: the table also exposes model size, "
            "training speed, codec throughput, memory, container overhead, and exact reconstruction. The Mamba, Griffin, and Gated DeltaNet "
            "rows are pure-PyTorch Lite references rather than optimized research-kernel reproductions; their speed results should be read as "
            "reference implementation measurements until an optimized kernel is added. Their batched training scans and one-token streaming "
            "steps use the same recurrence with different floating-point evaluation order; the actual streaming codec BPB and exact round-trip "
            "check are therefore the final authority for compression correctness.",
            "",
            "## Generation sanity check",
            "",
        ]
    )
    for row in report["results"]:
        sample = row.get("sample_text")
        if sample is not None:
            lines.append(f"- **{row['architecture']}**: {json.dumps(sample[:160])}")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bytes", type=int, default=262144)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cdf-bits", type=int, default=20)
    parser.add_argument("--sample-prompt", default="The ")
    parser.add_argument("--sample-length", type=int, default=96)
    parser.add_argument(
        "--expected-architectures",
        nargs="*",
        help="optional ordered architecture names; missing checkpoints are emitted as FAILED rows",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    original = args.input.read_bytes()
    if args.bytes > 0:
        original = original[: args.bytes]
    if not original:
        raise SystemExit("held-out input is empty")
    checkpoints = sorted(args.artifacts_root.glob("*/best.pt"))
    checkpoint_by_architecture = {checkpoint.parent.name: checkpoint for checkpoint in checkpoints}
    if not checkpoints and not args.expected_architectures:
        raise SystemExit(f"no architecture checkpoints found under {args.artifacts_root}")
    if args.expected_architectures:
        ordered_names = list(dict.fromkeys(args.expected_architectures))
        ordered_names.extend(
            name for name in sorted(checkpoint_by_architecture) if name not in ordered_names
        )
    else:
        ordered_names = [checkpoint.parent.name for checkpoint in checkpoints]
    results = []
    for name in ordered_names:
        checkpoint = checkpoint_by_architecture.get(name)
        if checkpoint is None:
            results.append(_missing_checkpoint_result(args.artifacts_root / name))
            continue
        results.append(
            benchmark_checkpoint(
                checkpoint,
                original,
                device=args.device,
                cdf_bits=args.cdf_bits,
                output_dir=args.output_dir,
                sample_prompt=args.sample_prompt.encode("utf-8"),
                sample_length=args.sample_length,
            )
        )
    report = {
        "input": str(args.input),
        "input_bytes": len(original),
        "device": args.device,
        "cdf_bits": args.cdf_bits,
        "expected_architectures": args.expected_architectures,
        "results": results,
        "recommendation": score_results(results),
    }
    write_comparison_report(report, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
