"""High-level lossless compression/decompression orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .coding import DEFAULT_CDF_BITS, RangeDecoder, RangeEncoder
from .file_format import build_container, parse_container, verify_checksum


class BytePredictor(Protocol):
    model_id: str

    def reset(self) -> None: ...

    def cdf(self, cdf_bits: int) -> tuple[int, ...]: ...

    def update(self, symbol: int) -> None: ...


def compress_bytes(
    original: bytes,
    predictor: BytePredictor,
    *,
    cdf_bits: int = DEFAULT_CDF_BITS,
) -> bytes:
    """Compress bytes with a causal predictor and return a V0 container."""

    predictor.reset()
    # CDF factories validate their own integer outputs. Avoid repeating the
    # 257-entry invariant scan in the hot symbol loop.
    encoder = RangeEncoder(validate_cdfs=False)
    for symbol in original:
        encoder.encode_symbol(predictor.cdf(cdf_bits), symbol)
        predictor.update(symbol)
    payload = encoder.finish()
    return build_container(
        model_id=predictor.model_id,
        cdf_bits=cdf_bits,
        original=original,
        payload=payload,
    )


def decompress_bytes(container: bytes, predictor: BytePredictor) -> bytes:
    """Decode a V0 container with the matching predictor."""

    header, payload = parse_container(container)
    if predictor.model_id != header.model_id:
        raise ValueError(
            f"model mismatch: stream requires {header.model_id!r}, "
            f"but predictor is {predictor.model_id!r}"
        )
    predictor.reset()
    decoder = RangeDecoder(payload, validate_cdfs=False)
    restored = bytearray()
    for _ in range(header.original_size):
        symbol = decoder.decode_symbol(predictor.cdf(header.cdf_bits))
        if not 0 <= symbol <= 255:
            raise ValueError(f"decoded non-byte symbol {symbol}")
        restored.append(symbol)
        predictor.update(symbol)
    result = bytes(restored)
    verify_checksum(header, result)
    return result


def compress_file(
    input_path: str | Path,
    output_path: str | Path,
    predictor: BytePredictor,
    *,
    cdf_bits: int = DEFAULT_CDF_BITS,
) -> None:
    original = Path(input_path).read_bytes()
    compressed = compress_bytes(original, predictor, cdf_bits=cdf_bits)
    Path(output_path).write_bytes(compressed)


def decompress_file(
    input_path: str | Path,
    output_path: str | Path,
    predictor: BytePredictor,
) -> None:
    container = Path(input_path).read_bytes()
    restored = decompress_bytes(container, predictor)
    Path(output_path).write_bytes(restored)
