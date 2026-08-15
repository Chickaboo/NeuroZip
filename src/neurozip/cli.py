"""Command-line interface for NeuroZip V0."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .codec import compress_file, decompress_file
from .file_format import parse_container
from .predictors import AdaptiveNgramPredictor, UniformPredictor, load_model_predictor


def _predictor(model: str | None, predictor: str | None = None, *, device: str = "cpu"):
    if predictor == "uniform" or model == "uniform":
        return UniformPredictor()
    if predictor == "ngram2":
        return AdaptiveNgramPredictor(order=2)
    if predictor == "ngram1":
        return AdaptiveNgramPredictor(order=1)
    if predictor == "gru" and model is None:
        raise SystemExit("--predictor gru requires --model PATH")
    if model is None:
        return UniformPredictor()
    return load_model_predictor(model, device=device)


def _add_predictor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help="path to a trained .pt model checkpoint; omit for the uniform smoke-test predictor",
    )
    parser.add_argument(
        "--predictor",
        choices=["uniform", "ngram1", "ngram2", "gru"],
        help="explicit predictor selection; uniform/ngram controls are dependency-free",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device for a learned model")


def _compress(args: argparse.Namespace) -> None:
    predictor = _predictor(args.model, args.predictor, device=args.device)
    start = time.perf_counter()
    compress_file(args.input, args.output, predictor, cdf_bits=args.cdf_bits)
    elapsed = time.perf_counter() - start
    original_size = Path(args.input).stat().st_size
    compressed_size = Path(args.output).stat().st_size
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "model_id": predictor.model_id,
                "original_bytes": original_size,
                "compressed_bytes": compressed_size,
                "file_bpb": (compressed_size * 8 / original_size) if original_size else None,
                "elapsed_seconds": elapsed,
                "encode_bytes_per_second": (original_size / elapsed) if elapsed else None,
            },
            indent=2,
        )
    )


def _decompress(args: argparse.Namespace) -> None:
    container = Path(args.input).read_bytes()
    header, _ = parse_container(container)
    predictor_name = args.predictor
    if args.model is None and predictor_name is None:
        builtin_models = {
            UniformPredictor.model_id: "uniform",
            "adaptive-byte-ngram1-v1": "ngram1",
            "adaptive-byte-ngram2-v1": "ngram2",
        }
        predictor_name = builtin_models.get(header.model_id)
        if predictor_name is None:
            raise SystemExit(
                f"stream requires model {header.model_id!r}; supply --model PATH "
                "or the matching built-in --predictor"
            )
    predictor = _predictor(args.model, predictor_name, device=args.device)
    start = time.perf_counter()
    decompress_file(args.input, args.output, predictor)
    elapsed = time.perf_counter() - start
    restored_size = Path(args.output).stat().st_size
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "model_id": header.model_id,
                "restored_bytes": restored_size,
                "elapsed_seconds": elapsed,
                "decode_bytes_per_second": (restored_size / elapsed) if elapsed else None,
            },
            indent=2,
        )
    )


def _benchmark(args: argparse.Namespace) -> None:
    predictor = _predictor(args.model, args.predictor, device=args.device)
    original = Path(args.input).read_bytes()
    start = time.perf_counter()
    from .codec import compress_bytes, decompress_bytes

    container = compress_bytes(original, predictor, cdf_bits=args.cdf_bits)
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    restored = decompress_bytes(
        container, _predictor(args.model, args.predictor, device=args.device)
    )
    decode_seconds = time.perf_counter() - start
    if restored != original:
        raise SystemExit("benchmark round-trip failed")
    report = {
        "input": str(args.input),
        "model_id": predictor.model_id,
        "original_bytes": len(original),
        "compressed_bytes": len(container),
        "file_bpb": len(container) * 8 / len(original) if original else None,
        "compression_ratio": len(original) / len(container) if container else None,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "encode_bytes_per_second": len(original) / encode_seconds if encode_seconds else None,
        "decode_bytes_per_second": len(original) / decode_seconds if decode_seconds else None,
    }
    print(json.dumps(report, indent=2))


def _model_info(args: argparse.Namespace) -> None:
    from .models.registry import load_checkpoint

    model, checkpoint = load_checkpoint(args.model, device="cpu")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "model_id": checkpoint.get("model_id"),
                "model_config": model.model_config,
                "parameters": parameters,
                "checkpoint_bytes": Path(args.model).stat().st_size,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurozip")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress = subparsers.add_parser("compress", help="compress a file")
    compress.add_argument("input", type=Path)
    compress.add_argument("output", type=Path)
    compress.add_argument("--cdf-bits", type=int, default=20)
    _add_predictor_args(compress)
    compress.set_defaults(func=_compress)

    decompress = subparsers.add_parser("decompress", help="decompress a NeuroZip stream")
    decompress.add_argument("input", type=Path)
    decompress.add_argument("output", type=Path)
    _add_predictor_args(decompress)
    decompress.set_defaults(func=_decompress)

    benchmark = subparsers.add_parser("benchmark", help="round-trip and time one input")
    benchmark.add_argument("input", type=Path)
    benchmark.add_argument("--cdf-bits", type=int, default=20)
    _add_predictor_args(benchmark)
    benchmark.set_defaults(func=_benchmark)

    model_info = subparsers.add_parser("model-info", help="inspect a learned checkpoint")
    model_info.add_argument("model", type=Path)
    model_info.set_defaults(func=_model_info)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
