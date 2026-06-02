#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spinefairbench.generator.config import GeneratorConfig
from spinefairbench.generator.prompts import DEMOGRAPHIC_PROMPTS


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_synthetic_png(path: Path, size: int = 128) -> None:
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            spine_band = 90 if abs(x - size // 2) < 10 else 0
            gradient = int(40 + 120 * (y / max(size - 1, 1)))
            row.append(min(255, gradient + spine_band))
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a SpineFairBench generator smoke test.")
    parser.add_argument("--dry-run", action="store_true", help="Create synthetic metadata without SD inference.")
    parser.add_argument("--input", type=Path, help="Safe user-supplied source image for real inference.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/spinefairbench_generator_smoke"))
    parser.add_argument("--checkpoint", type=Path, help="Local Diffusers-compatible LoRA checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.dry_run or args.input is None or args.checkpoint is None:
        source = args.output / "synthetic_test_input.png"
        write_synthetic_png(source)
        metadata = {
            "status": "dry_run_ok",
            "source_filename": source.name,
            "seed": args.seed,
            "config": GeneratorConfig(seed=args.seed, device=args.device).to_dict(),
            "prompts": DEMOGRAPHIC_PROMPTS,
            "checkpoint_required_for_real_inference": True,
            "checkpoint_released": False,
            "raw_source_radiographs_released": False,
            "note": "Synthetic dry run only; no Stable Diffusion inference executed.",
        }
        meta_path = args.output / "dry_run_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "ok", "output": str(args.output), "metadata": meta_path.name}, sort_keys=True))
        return

    from spinefairbench.generator.infer import build_parser as build_infer_parser
    from spinefairbench.generator.infer import run_inference

    infer_args = build_infer_parser().parse_args(
        [
            "--input",
            str(args.input),
            "--output",
            str(args.output),
            "--checkpoint",
            str(args.checkpoint),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
    )
    payload = run_inference(infer_args)
    print(json.dumps({"status": "ok", "metadata": payload}, sort_keys=True))


if __name__ == "__main__":
    main()
