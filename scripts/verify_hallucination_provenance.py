#!/usr/bin/env python3
"""Verify saved exploratory arithmetic and compare the recorded base pairing.

Uses only the Python standard library and five SHA-256-bound source files.
Does not call a model, load checkpoints, generate images, or bootstrap intervals.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

SOURCE_HASHES = {
    "analyze.py": "5fc2eeb362ada8a4720db8db467ea65b3202f4a5c4e34289cdd66e3f5a0d3f1f",
    "metrics/hallucination.py": "fe82ca6cbad226bbb47129b3a958ae86dc2bf7e08ae7d0e73d0992a4f9fc3840",
    "metrics/diagnostic_label.py": "79650c751cb346c688d0741df7f872b6cafb9076d761f04e6ffa4961702240ea",
    "metrics/refusal_detector.py": "45830d930cbbdbf4f35582aa94e688b7590c302a4072e67f8b0cfc33be9915a3",
    "data/annotations.py": "2bed2312045293de07b8196454d2e957e291bce190f759122ad7881be6a9929c",
}
ARTIFACT_HASHES = {
    "Results/analysis/common_core_1000_summary.json": "e0eb42ebcb28b8e8aae16ae7749ae0adb810bf0c0c8f5ca2a8810d50993611f5",
    "Results/final_inputs/panels/full_pipeline_retained/evaluation_results.json": "5e3ff8d1c3cc808e2199e1c19f0b6c456437c21db2bc58df05aca0c34891b7b3",
    "Results/final_inputs/panels/baseline_only_retained/evaluation_results.json": "7d900f5e2965e9ad863e0c1a0a6e2bcd41a03c5a9bc7339efc60566172b0e67a",
    "Results/final_inputs/panels/full_pipeline_retained/pairs.json": "cb5c29f25c54a086c934c77517e8e28ae695704192f181084c5efdc5a23c58bb",
    "Results/final_inputs/panels/baseline_only_retained/pairs.json": "cb5c29f25c54a086c934c77517e8e28ae695704192f181084c5efdc5a23c58bb",
}


def _load_json(path: Path):
    # JSON bytes carry their Unicode encoding independently of the OS locale.
    return json.loads(path.read_bytes())


def verify_hashes(root: Path, expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        path = root / relative
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch: {relative}")


def load_verified_module(name: str, path: Path, expected_sha256: str):
    source_bytes = path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load verified module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # A source loader can execute a timestamp-valid .pyc without reading the
    # verified source. Compile these exact bytes instead of consulting caches.
    exec(compile(source_bytes, str(path), "exec"), module.__dict__)
    return module


def reconstruct(artifacts: Path, source: Path) -> dict:
    # Verify every executable source and scientific input before interpretation.
    verify_hashes(source, SOURCE_HASHES)
    verify_hashes(artifacts, ARTIFACT_HASHES)
    label = load_verified_module("_sfb_recorded_labels", source / "metrics/diagnostic_label.py", SOURCE_HASHES["metrics/diagnostic_label.py"])
    refusal = load_verified_module("_sfb_recorded_refusal", source / "metrics/refusal_detector.py", SOURCE_HASHES["metrics/refusal_detector.py"])
    syntax = ast.parse((source / "data/annotations.py").read_bytes())
    categories = next(
        ast.literal_eval(node.value)
        for node in syntax.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ABNORMALITY_CATEGORIES" for target in node.targets)
    )
    summary = _load_json(artifacts / "Results/analysis/common_core_1000_summary.json")

    @lru_cache(maxsize=100000)
    def predictions(text: str) -> frozenset[str]:
        return frozenset(label.extract_labels(text) - {"No finding"})

    @lru_cache(maxsize=100000)
    def is_full_refusal(text: str) -> bool:
        return refusal.classify_response(text) == refusal.ResponseClass.FULL_REFUSAL

    def lexical_score(text: str, ground: set[str]) -> float:
        predicted = predictions(text)
        return len(predicted - ground) / len(predicted) if predicted else 0.0

    result = {
        "schema_version": "1.0",
        "record_type": "saved_hallucination_arithmetic_verification",
        "scope": "Current arithmetic reconstruction from frozen reports; no model call, bootstrap, or authentication of the dirty historical runtime.",
        "recorded_base_commit": "889358c42a52f3406005f0939e1f20ad698d7be4",
        "recorded_runtime_git_dirty": True,
        "source_sha256": SOURCE_HASHES,
        "artifact_sha256": ARTIFACT_HASHES,
        "pair_id_recipe": "Join source/generated reports by model and pair_id, exclude errors and full refusals, compute mean absolute difference of unmatched-positive-prediction fractions relative to source labels.",
        "recorded_base_recipe": "Mirror analyze.py source_id-keyed last-surviving source reports and source_id::image_path-keyed generated reports, then retain generated rows with any surviving source report and source labels.",
        "models": {},
    }
    for panel, short_name in [("full_pipeline_retained", "full"), ("baseline_only_retained", "baseline")]:
        panel_path = artifacts / "Results/final_inputs/panels" / panel
        rows = _load_json(panel_path / "evaluation_results.json")
        pairs = _load_json(panel_path / "pairs.json")
        labels_by_source = {
            row["source_id"]: {categories[i] for i in row["pathology_labels"] if i in range(len(categories))} - {"No finding"}
            for row in pairs if "pathology_labels" in row
        }
        matched = {}
        base_sources = {}
        base_generated = {}
        for row in rows:
            text = row.get("response") or ""
            if row.get("error") or is_full_refusal(text):
                continue
            model, source_id, role = row["model"], row["pair_source_id"], row["image_role"]
            pair = matched.setdefault(model, {}).setdefault(row["pair_id"], {"source_id": source_id})
            if role in pair:
                raise ValueError(f"Duplicate surviving pair-role entry: {model}/{row['pair_id']}/{role}")
            pair[role] = text
            if role == "source":
                base_sources.setdefault(model, {})[source_id] = text
            elif role == "generated":
                key = f"{source_id}::{row['image_path']}"
                base_generated.setdefault(model, {})[key] = (source_id, text)

        for model, pairs_by_id in matched.items():
            diffs = []
            pair_counts = Counter()
            for pair in pairs_by_id.values():
                if "source" not in pair or "generated" not in pair:
                    continue
                source_id = pair["source_id"]
                ground = labels_by_source[source_id]
                diffs.append(abs(lexical_score(pair["source"], ground) - lexical_score(pair["generated"], ground)))
                pair_counts[source_id.split("_")[0]] += 1

            base_diffs = []
            base_counts = Counter()
            for source_id, generated in base_generated.get(model, {}).values():
                if source_id not in base_sources.get(model, {}) or source_id not in labels_by_source:
                    continue
                ground = labels_by_source[source_id]
                base_diffs.append(abs(lexical_score(base_sources[model][source_id], ground) - lexical_score(generated, ground)))
                base_counts[source_id.split("_")[0]] += 1

            saved = summary["panels"][short_name]["models"][model]["primary_secondary_stats"]["hallucination"]
            mean = math.fsum(diffs) / len(diffs)
            base_mean = math.fsum(base_diffs) / len(base_diffs)
            result["models"][model] = {
                "n_pairs": len(diffs),
                "source_dataset_pair_counts": dict(sorted(pair_counts.items())),
                "computed_mean_abs_difference": mean,
                "saved_mean_disparity": saved["mean_disparity"],
                "absolute_error": abs(mean - saved["mean_disparity"]),
                "denominator_matches": len(diffs) == saved["n_pairs"],
                "recorded_base_pairing": {
                    "n_pairs": len(base_diffs),
                    "source_dataset_pair_counts": dict(sorted(base_counts.items())),
                    "mean_abs_difference": base_mean,
                    "absolute_error_against_saved": abs(base_mean - saved["mean_disparity"]),
                    "denominator_matches_saved": len(base_diffs) == saved["n_pairs"],
                },
            }
    models = result["models"]
    result["model_count"] = len(models)
    result["all_denominators_match"] = all(row["denominator_matches"] for row in models.values())
    result["max_absolute_error"] = max(row["absolute_error"] for row in models.values())
    result["verified"] = len(models) == 9 and result["all_denominators_match"] and result["max_absolute_error"] <= 1e-12
    result["limitations"] = [
        "Arithmetic agreement is not authentication of the exact dirty April runtime source.",
        "Recorded-base pairing is a faithful code-path reconstruction using the same frozen inputs, not a model rerun or a replacement result.",
        "The population includes BUU and VinDr; it is not an independently isolated VinDr-only estimate.",
        "This lexical fraction is neither a conventional false-positive rate nor a clinically adjudicated hallucination rate.",
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path, help="Extracted bundle root containing artifacts/Results/, or the inner root containing Results/.")
    parser.add_argument("--recorded-source-root", type=Path, default=Path(__file__).resolve().parents[1] / "provenance/historical_scoring_source/spinefairbench", help="Directory containing the five SHA-bound recorded-base files (analyze.py, metrics/, data/).")
    parser.add_argument("--output", type=Path, help="Optional JSON receipt path; parent must exist.")
    args = parser.parse_args()
    try:
        artifact_root = args.artifacts
        if not (artifact_root / "Results").is_dir() and (artifact_root / "artifacts/Results").is_dir():
            artifact_root = artifact_root / "artifacts"
        result = reconstruct(artifact_root, args.recorded_source_root)
        encoded = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0 if result["verified"] else 1
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print(json.dumps({"verified": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
