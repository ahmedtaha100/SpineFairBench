from __future__ import annotations

import csv
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spinefairbench.release.common import (
    LEADERBOARD_COLUMNS,
    ReleaseError,
    copy_file,
    create_tar_gz,
    ensure_empty_dir,
    normalize_release_tag,
    write_checksums,
    write_json,
    write_text,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse


def _resolve_artifact_dir(
    provided: str | None,
    fallback: Path,
) -> Path:
    if provided:
        return Path(provided).resolve()
    return fallback.resolve()


def _load_pairs_file(pairs_file: Path) -> list[dict[str, Any]]:
    if not pairs_file.exists():
        raise ReleaseError(f"Pairs file not found: {pairs_file}")
    with open(pairs_file, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ReleaseError(f"Pairs file must contain a list: {pairs_file}")
    return payload


def _resolve_image_path(path_value: str, pairs_file: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    rel_from_pairs = (pairs_file.parent / path_value).resolve()
    if rel_from_pairs.exists():
        return rel_from_pairs
    abs_candidate = path.resolve()
    if abs_candidate.exists():
        return abs_candidate
    raise ReleaseError(
        f"Image path from pairs.json does not exist: {path_value}"
    )


def _freeze_pairs_and_images(
    pairs: list[dict[str, Any]],
    pairs_file: Path,
    dataset_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    frozen_pairs: list[dict[str, Any]] = []
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    mapped_paths: dict[Path, str] = {}

    for pair in pairs:
        for key in ("source_path", "generated_path"):
            raw = pair.get(key)
            if not isinstance(raw, str) or not raw:
                raise ReleaseError(f"Pair entry missing {key}")
            original = _resolve_image_path(raw, pairs_file)
            if original not in mapped_paths:
                name = f"{len(mapped_paths):07d}_{original.name}"
                target = images_dir / name
                copy_file(original, target)
                mapped_paths[original] = f"images/{name}"

        frozen_pair = dict(pair)
        frozen_pair["source_path"] = mapped_paths[_resolve_image_path(str(pair["source_path"]), pairs_file)]
        frozen_pair["generated_path"] = mapped_paths[_resolve_image_path(str(pair["generated_path"]), pairs_file)]
        frozen_pairs.append(frozen_pair)

    return frozen_pairs, len(mapped_paths)


def _write_submission_schema(metrics_dir: Path, sample_pair_id: str | None = None) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SpineFairBench Submission",
        "description": "Pre-computed free-text source/counterfactual model reports for Kaggle-style SpineFairBench scoring.",
        "type": "object",
        "required": ["schema_version", "model", "scope", "results"],
        "properties": {
            "schema_version": {"const": "spinefairbench-submission-v1"},
            "model": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "version": {"type": "string"},
                    "organization": {"type": "string"},
                    "model_family": {"type": "string"},
                    "prompt_id": {"type": "string"},
                    "inference_date_utc": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "scope": {
                "enum": [
                    "common-core-1000",
                    "all-model-intersection-2166",
                    "qc-passed",
                    "toy",
                ]
            },
            "results": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "pair_id",
                        "source_report",
                        "counterfactual_report",
                    ],
                    "properties": {
                        "pair_id": {
                            "type": "string",
                            "pattern": "^(buu|vindr)_.+__.+$",
                        },
                        "source_id": {"type": "string", "minLength": 1},
                        "source_report": {"type": "string", "minLength": 1},
                        "counterfactual_report": {"type": "string", "minLength": 1},
                        "metadata": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }
    template_pair_id = sample_pair_id or "buu_0004-F-010Y0__elderly_female"
    template = {
        "schema_version": "spinefairbench-submission-v1",
        "model": {
            "name": "your-model-name",
            "version": "your-model-version",
            "organization": "your-lab-or-company",
            "model_family": "foundation-or-medical-vlm-family",
            "prompt_id": "your-prompt-or-protocol-id",
            "inference_date_utc": "2026-05-06",
        },
        "scope": "common-core-1000",
        "results": [
            {
                "pair_id": template_pair_id,
                "source_report": "Findings: ...",
                "counterfactual_report": "Findings: ...",
            }
        ],
    }
    write_json(metrics_dir / "submission_schema.json", schema)
    write_json(metrics_dir / "submission_template.json", template)


def _write_repro_scripts(release_dir: Path) -> None:
    eval_script = """from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()

    evaluate_cmd = [
        sys.executable,
        "-m",
        "spinefairbench",
        "evaluate",
        "--config",
        args.config,
        "--pairs-dir",
        args.pairs_dir,
        "--results-dir",
        args.results_dir,
    ]
    if args.models:
        evaluate_cmd.extend(["--models", *args.models])

    analyze_cmd = [
        sys.executable,
        "-m",
        "spinefairbench",
        "analyze",
        "--config",
        args.config,
        "--results-dir",
        args.results_dir,
    ]

    first = subprocess.run(evaluate_cmd, check=False)
    if first.returncode != 0:
        return first.returncode
    second = subprocess.run(analyze_cmd, check=False)
    return second.returncode


if __name__ == "__main__":
    raise SystemExit(main())
"""
    leaderboard_script = """from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-pairs", required=True)
    parser.add_argument("--submission-file", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--leaderboard-file", required=True)
    args = parser.parse_args()

    score_cmd = [
        sys.executable,
        "-m",
        "spinefairbench",
        "score-submission",
        "--benchmark-pairs",
        args.benchmark_pairs,
        "--submission-file",
        args.submission_file,
        "--output-file",
        args.score_output,
    ]
    update_cmd = [
        sys.executable,
        "-m",
        "spinefairbench",
        "update-leaderboard",
        "--leaderboard-file",
        args.leaderboard_file,
        "--score-file",
        args.score_output,
    ]

    first = subprocess.run(score_cmd, check=False)
    if first.returncode != 0:
        return first.returncode
    second = subprocess.run(update_cmd, check=False)
    return second.returncode


if __name__ == "__main__":
    raise SystemExit(main())
"""
    write_text(release_dir / "evaluation" / "run_reproduction.py", eval_script)
    write_text(release_dir / "leaderboard" / "run_scoring_and_leaderboard.py", leaderboard_script)


def _write_docs(release_dir: Path, release_tag: str) -> None:
    readme = f"""# SpineFairBench Benchmark Package {release_tag}

This package contains a frozen counterfactual dataset, benchmark metadata, scoring templates, and leaderboard scaffolding.

## Package layout
- `dataset/pairs.json`: frozen benchmark pair metadata.
- `dataset/images/`: frozen source and generated images used by `pairs.json`.
- `evaluation/`: baseline artifacts and a reproduction script.
- `metrics/submission_schema.json`: schema for leaderboard submissions.
- `metrics/submission_template.json`: starter template for submissions.
- `leaderboard/leaderboard.csv`: initial leaderboard scaffold.
- `manifests/`: checksums and release manifest.

## Basic usage
1. Install SpineFairBench.
2. Run model inference over `dataset/pairs.json`.
3. Save outputs in the submission format.
4. Score submission:
   `python -m spinefairbench score-submission --benchmark-pairs dataset/pairs.json --submission-file your_submission.json --output-file score.json`
5. Update leaderboard:
   `python -m spinefairbench update-leaderboard --leaderboard-file leaderboard/leaderboard.csv --score-file score.json`
6. Validate release package:
   `python -m spinefairbench validate-release --release-dir . --output-file validation_report.json`
"""
    repro = """# Reproducibility Guide

Use `evaluation/run_reproduction.py` to rerun evaluation and analysis on frozen pairs:

`python evaluation/run_reproduction.py --config configs/default.yaml --pairs-dir dataset --results-dir evaluation`
"""
    extension = """# Extension Guide

To extend this benchmark:
1. Keep pair IDs and image files immutable for a release version.
2. Add new model outputs in submission JSON format.
3. Score outputs with `score-submission`.
4. Append scores using `update-leaderboard`.
5. Publish new benchmark versions only when dataset membership changes.
"""
    write_text(release_dir / "README.md", readme)
    write_text(release_dir / "docs" / "REPRODUCING.md", repro)
    write_text(release_dir / "docs" / "EXTENDING.md", extension)


def package_benchmark_release(args: argparse.Namespace) -> tuple[Path, Path]:
    try:
        from spinefairbench.config.schemas import load_config
    except ModuleNotFoundError as exc:
        raise ReleaseError(
            "package_benchmark_release requires the internal source-tree config "
            "module and is not part of the reviewer-facing scoring path."
        ) from exc

    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    pairs_dir = _resolve_artifact_dir(
        getattr(args, "pairs_dir", None),
        Path(config.generation.output_dir),
    )
    evaluation_dir = _resolve_artifact_dir(
        getattr(args, "evaluation_dir", None),
        Path(config.evaluation.output_dir),
    )
    analysis_dir = _resolve_artifact_dir(
        getattr(args, "analysis_dir", None),
        Path(config.analysis.output_dir),
    )

    pairs_file = pairs_dir / "pairs.json"
    release_tag = normalize_release_tag(args.version)
    release_name = args.release_name or f"spinefairbench_benchmark_{release_tag}"

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    release_dir = output_root / release_name
    ensure_empty_dir(release_dir)

    dataset_dir = release_dir / "dataset"
    evaluation_out = release_dir / "evaluation"
    metrics_out = release_dir / "metrics"
    leaderboard_out = release_dir / "leaderboard"
    manifests_out = release_dir / "manifests"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    evaluation_out.mkdir(parents=True, exist_ok=True)
    metrics_out.mkdir(parents=True, exist_ok=True)
    leaderboard_out.mkdir(parents=True, exist_ok=True)
    manifests_out.mkdir(parents=True, exist_ok=True)

    pairs = _load_pairs_file(pairs_file)
    frozen_pairs, image_count = _freeze_pairs_and_images(pairs, pairs_file, dataset_dir)
    write_json(dataset_dir / "pairs.json", frozen_pairs)

    quality_report = pairs_dir / "quality_report.json"
    if quality_report.exists():
        copy_file(quality_report, dataset_dir / "quality_report.json")

    reference_eval = evaluation_dir / "evaluation_results.json"
    if reference_eval.exists():
        copy_file(reference_eval, evaluation_out / "reference_evaluation_results.json")

    benchmark_report = analysis_dir / "benchmark_report.json"
    run_manifest = analysis_dir / "run_manifest.json"
    if benchmark_report.exists():
        copy_file(benchmark_report, evaluation_out / "reference_benchmark_report.json")
    if run_manifest.exists():
        copy_file(run_manifest, evaluation_out / "reference_run_manifest.json")

    copy_file(config_path, manifests_out / "config_snapshot.yaml")
    sample_pair_id: str | None = None
    if frozen_pairs:
        source_id = frozen_pairs[0].get("source_id")
        if isinstance(source_id, str) and source_id:
            sample_pair_id = source_id
    _write_submission_schema(metrics_out, sample_pair_id=sample_pair_id)
    csv_buffer = io.StringIO()
    csv.writer(csv_buffer).writerow(LEADERBOARD_COLUMNS)
    write_text(leaderboard_out / "leaderboard.csv", csv_buffer.getvalue())
    _write_repro_scripts(release_dir)
    _write_docs(release_dir, release_tag)

    manifest_path = manifests_out / "release_manifest.json"
    manifest: dict[str, Any] = {
        "package_type": "benchmark",
        "name": release_name,
        "version": release_tag,
        "created_utc": datetime.now(UTC).isoformat(),
        "pairs_count": len(frozen_pairs),
        "image_count": image_count,
        "source_artifacts": {
            "config_file": config_path.name,
            "pairs_file": pairs_file.name,
            "evaluation_dir_name": evaluation_dir.name,
            "analysis_dir_name": analysis_dir.name,
        },
        "files": [],
    }
    write_json(manifest_path, manifest)
    checksums = write_checksums(release_dir, manifests_out / "checksums.sha256")
    manifest["files"] = [
        entry
        for entry in checksums
        if entry.get("path") != "manifests/release_manifest.json"
    ]
    write_json(manifest_path, manifest)
    write_checksums(release_dir, manifests_out / "checksums.sha256")

    archive_path = create_tar_gz(release_dir, output_root / f"{release_name}.tar.gz")
    return release_dir, archive_path


def run_package_benchmark(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    release_dir, archive_path = package_benchmark_release(args)
    logger.info("Benchmark package created at %s", release_dir)
    logger.info("Benchmark archive created at %s", archive_path)
