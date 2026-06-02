from __future__ import annotations

import argparse
from pathlib import Path

from spinefairbench.generator.config import GeneratorConfig, load_yaml_config
from spinefairbench.generator.io import public_path_label, write_json
from spinefairbench.generator.pipeline import CounterfactualGeneratorPipeline


def run_inference(args: argparse.Namespace) -> dict[str, object]:
    config = load_yaml_config(args.config) if args.config else GeneratorConfig()
    config = GeneratorConfig.from_mapping({
        **config.to_dict(),
        "checkpoint_path": str(args.checkpoint) if args.checkpoint else config.checkpoint_path,
        "device": args.device or config.device,
    })
    pipeline = CounterfactualGeneratorPipeline(config)
    output_path = args.output / f"{args.demographic}_counterfactual.png"
    metadata = pipeline.generate(
        args.input,
        args.demographic,
        output_path,
        mask_path=args.mask,
        seed=args.seed,
    )
    payload = {
        **metadata,
        "source_filename": public_path_label(args.input),
        "checkpoint_filename": public_path_label(args.checkpoint) if args.checkpoint else None,
    }
    write_json(output_path.with_suffix(".json"), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SpineFairBench generator inference.")
    parser.add_argument("--input", required=True, type=Path, help="Source radiograph supplied by the user.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--checkpoint", type=Path, help="Local Diffusers-compatible LoRA checkpoint.")
    parser.add_argument("--config", type=Path, help="Generator YAML config.")
    parser.add_argument("--mask", type=Path, help="Optional binary TSXR spine mask image.")
    parser.add_argument(
        "--demographic",
        default="elderly_female",
        choices=["elderly_female", "elderly_male", "young_female", "young_male"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, mps, or auto.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
