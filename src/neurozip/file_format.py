"""Versioned NeuroZip V0 container format."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass


MAGIC = b"NZP0"
VERSION = 1
CHECKSUM_SIZE = hashlib.sha256().digest_size
HEADER = struct.Struct("<4sBBBBQQ")


@dataclass(frozen=True)
class ContainerHeader:
    version: int
    flags: int
    cdf_bits: int
    model_id: str
    original_size: int
    payload_size: int
    checksum: bytes


def build_container(
    *,
    model_id: str,
    cdf_bits: int,
    original: bytes,
    payload: bytes,
    flags: int = 0,
) -> bytes:
    """Build a complete V0 stream, including metadata and SHA-256."""

    model_bytes = model_id.encode("ascii")
    if not model_bytes or len(model_bytes) > 255:
        raise ValueError("model_id must be 1..255 ASCII bytes")
    if not 8 <= cdf_bits <= 24:
        raise ValueError("cdf_bits must be between 8 and 24")
    if not 0 <= flags <= 255:
        raise ValueError("flags must fit in one byte")
    header = HEADER.pack(
        MAGIC,
        VERSION,
        flags,
        cdf_bits,
        len(model_bytes),
        len(original),
        len(payload),
    )
    checksum = hashlib.sha256(original).digest()
    return header + model_bytes + checksum + payload


def parse_container(container: bytes) -> tuple[ContainerHeader, bytes]:
    """Parse and validate the header, returning metadata and the payload."""

    if len(container) < HEADER.size + CHECKSUM_SIZE:
        raise ValueError("compressed stream is shorter than the V0 header")
    magic, version, flags, cdf_bits, model_len, original_size, payload_size = HEADER.unpack_from(
        container
    )
    if magic != MAGIC:
        raise ValueError("invalid NeuroZip magic")
    if version != VERSION:
        raise ValueError(f"unsupported NeuroZip version: {version}")
    if not 8 <= cdf_bits <= 24:
        raise ValueError(f"unsupported CDF precision: {cdf_bits}")
    metadata_end = HEADER.size + model_len + CHECKSUM_SIZE
    expected_size = metadata_end + payload_size
    if len(container) != expected_size:
        raise ValueError(
            f"container size mismatch: header expects {expected_size} bytes, got {len(container)}"
        )
    model_start = HEADER.size
    model_end = model_start + model_len
    try:
        model_id = container[model_start:model_end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("model identifier is not ASCII") from exc
    checksum = container[model_end:metadata_end]
    payload = container[metadata_end:]
    return (
        ContainerHeader(
            version=version,
            flags=flags,
            cdf_bits=cdf_bits,
            model_id=model_id,
            original_size=original_size,
            payload_size=payload_size,
            checksum=checksum,
        ),
        payload,
    )


def verify_checksum(header: ContainerHeader, restored: bytes) -> None:
    if len(restored) != header.original_size:
        raise ValueError(
            f"decoded length mismatch: expected {header.original_size}, got {len(restored)}"
        )
    actual = hashlib.sha256(restored).digest()
    if actual != header.checksum:
        raise ValueError("decoded data failed the SHA-256 checksum")

