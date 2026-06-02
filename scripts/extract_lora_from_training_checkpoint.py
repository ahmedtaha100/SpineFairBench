#!/usr/bin/env python3
"""Extract LoRA-only tensors from a SpineFairBench training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


UNSAFE_KEY_FRAGMENTS = (
    "optimizer",
    "discriminator",
    "demographic_encoder",
    "raw",
    "image",
    "patient",
    "token",
    "secret",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_lora_tensors(unet_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in sorted(unet_state.items()):
        lowered = key.lower()
        if "lora_" not in lowered:
            continue
        if any(fragment in lowered for fragment in UNSAFE_KEY_FRAGMENTS):
            raise ValueError(f"unsafe LoRA key fragment found in {key!r}")
        if not torch.is_tensor(value):
            raise TypeError(f"LoRA entry {key!r} is not a tensor")
        tensors[key] = value.detach().cpu().contiguous()
    return tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clean LoRA-only safetensors from a training checkpoint."
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw PyTorch checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Output safetensors file")
    parser.add_argument(
        "--expected-source-sha256",
        default=None,
        help="Optional expected SHA256 for the input checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output

    source_sha256 = sha256_file(input_path)
    if args.expected_source_sha256 and source_sha256 != args.expected_source_sha256:
        raise SystemExit(
            "input checkpoint SHA256 mismatch: "
            f"expected {args.expected_source_sha256}, got {source_sha256}"
        )

    checkpoint = torch.load(input_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(checkpoint)!r}")

    unet_state = checkpoint.get("unet")
    if not isinstance(unet_state, dict):
        raise KeyError("checkpoint does not contain a dict-valued 'unet' state")

    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    lora_tensors = get_lora_tensors(unet_state)
    if not lora_tensors:
        raise SystemExit("no LoRA tensors found under checkpoint['unet']")

    metadata = {
        "format": "spinefairbench_lora_safetensors",
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_size_bytes": str(input_path.stat().st_size),
        "base_model": "stable-diffusion-v1-5",
        "lora_rank": str(config.get("lora_rank", "")),
        "lora_alpha": str(config.get("lora_alpha", "")),
        "tensor_scope": "unet_lora_only",
        "raw_source_radiographs": "not_included",
        "source_masks": "not_included",
        "training_state": "not_included",
        "optimizer_state": "not_included",
        "discriminator": "not_included",
        "demographic_encoder": "not_included",
        "private_run_roots": "not_included",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_tensors, str(output_path), metadata=metadata)

    print("source_sha256", source_sha256)
    print("num_lora_tensors", len(lora_tensors))
    print("output", output_path)
    print("first_20_keys", list(lora_tensors.keys())[:20])


if __name__ == "__main__":
    main()
