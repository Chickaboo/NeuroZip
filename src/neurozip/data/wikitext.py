"""Deterministic raw WikiText-103 sample preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


WIKITEXT_RAW_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
DEFAULT_TRAIN_BYTES = 50 * 1024 * 1024
DEFAULT_VALID_BYTES = 5 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "neurozip-v0/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)


def find_raw_file(root: Path, filename: str) -> Path:
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"could not find {filename} under {root}")
    return matches[0]


def ensure_raw_dataset(
    *,
    cache_dir: Path,
    archive: Path | None = None,
    raw_dir: Path | None = None,
    url: str = WIKITEXT_RAW_URL,
) -> tuple[Path, Path, dict[str, str]]:
    """Return raw train/validation paths and source metadata."""

    if raw_dir is not None:
        train_path = find_raw_file(raw_dir, "wiki.train.raw")
        valid_path = find_raw_file(raw_dir, "wiki.valid.raw")
        source = {"source": "local", "source_path": str(raw_dir)}
        return train_path, valid_path, source

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive or cache_dir / "wikitext-103-raw-v1.zip"
    if not archive_path.exists():
        print(f"downloading WikiText-103 raw archive to {archive_path}")
        download(url, archive_path)
    extracted = cache_dir / "wikitext-103-raw-v1"
    extracted.mkdir(parents=True, exist_ok=True)
    train_candidates = list(extracted.rglob("wiki.train.raw"))
    valid_candidates = list(extracted.rglob("wiki.valid.raw"))
    if not train_candidates or not valid_candidates:
        with zipfile.ZipFile(archive_path) as archive_file:
            archive_file.extractall(extracted)
    train_path = find_raw_file(extracted, "wiki.train.raw")
    valid_path = find_raw_file(extracted, "wiki.valid.raw")
    source = {
        "source": "download",
        "source_url": url,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
    }
    return train_path, valid_path, source


def deterministic_window(path: Path, target_bytes: int, *, seed: int, split: str) -> tuple[bytes, int]:
    """Select one deterministic contiguous raw-byte window.

    The bytes are copied without decoding, newline conversion, or Unicode
    normalization. A contiguous window keeps the training stream free of
    artificial separators while the seed makes the chosen offset reproducible.
    """

    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    size = path.stat().st_size
    if size <= target_bytes:
        offset = 0
        length = size
    else:
        digest = hashlib.sha256(f"{seed}:{split}".encode("utf-8")).digest()
        offset = int.from_bytes(digest[:8], "big") % (size - target_bytes + 1)
        length = target_bytes
    with path.open("rb") as handle:
        handle.seek(offset)
        sample = handle.read(length)
    if len(sample) != length:
        raise IOError(f"short read while sampling {path}")
    return sample, offset


def prepare_samples(
    *,
    output_dir: Path,
    train_bytes: int = DEFAULT_TRAIN_BYTES,
    valid_bytes: int = DEFAULT_VALID_BYTES,
    seed: int = 20260814,
    cache_dir: Path | None = None,
    archive: Path | None = None,
    raw_dir: Path | None = None,
    url: str = WIKITEXT_RAW_URL,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or output_dir / "cache"
    train_raw, valid_raw, source = ensure_raw_dataset(
        cache_dir=cache_dir, archive=archive, raw_dir=raw_dir, url=url
    )
    train_sample, train_offset = deterministic_window(
        train_raw, train_bytes, seed=seed, split="train"
    )
    valid_sample, valid_offset = deterministic_window(
        valid_raw, valid_bytes, seed=seed, split="validation"
    )
    train_output = output_dir / "train.raw"
    valid_output = output_dir / "validation.raw"
    train_output.write_bytes(train_sample)
    valid_output.write_bytes(valid_sample)
    manifest: dict[str, object] = {
        **source,
        "dataset": "WikiText-103 raw v1",
        "seed": seed,
        "encoding": "raw bytes copied from the official .raw files; no decoding or normalization",
        "train_source_file": str(train_raw),
        "validation_source_file": str(valid_raw),
        "train_source_sha256": sha256_file(train_raw),
        "validation_source_sha256": sha256_file(valid_raw),
        "train_source_size_bytes": train_raw.stat().st_size,
        "validation_source_size_bytes": valid_raw.stat().st_size,
        "train_offset": train_offset,
        "validation_offset": valid_offset,
        "train_bytes": len(train_sample),
        "validation_bytes": len(valid_sample),
        "train_sha256": hashlib.sha256(train_sample).hexdigest(),
        "validation_sha256": hashlib.sha256(valid_sample).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--url", default=WIKITEXT_RAW_URL)
    parser.add_argument("--train-bytes", type=int, default=DEFAULT_TRAIN_BYTES)
    parser.add_argument("--valid-bytes", type=int, default=DEFAULT_VALID_BYTES)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = prepare_samples(
        output_dir=args.output_dir,
        train_bytes=args.train_bytes,
        valid_bytes=args.valid_bytes,
        seed=args.seed,
        cache_dir=args.cache_dir,
        archive=args.archive,
        raw_dir=args.raw_dir,
        url=args.url,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
