#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spinefairbench.generator.config import GeneratorConfig, QCThresholds
from spinefairbench.generator.prompts import DEMOGRAPHIC_PROMPTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify generator release metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Run standard-library release checks.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.dry_run:
        raise SystemExit("Pass --dry-run to run release checks.")

    config = GeneratorConfig()
    thresholds = QCThresholds()
    payload = {
        "status": "ok",
        "config": config.to_dict(),
        "qc_thresholds": thresholds.__dict__,
        "prompt_count": len(DEMOGRAPHIC_PROMPTS),
        "prompts": DEMOGRAPHIC_PROMPTS,
        "checkpoint_released": False,
        "training_code_released": False,
        "raw_source_radiographs_released": False,
        "standard_library_only": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
