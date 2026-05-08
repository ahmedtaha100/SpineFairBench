from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spinefairbench.metrics.diagnostic_label import compute_jaccard, extract_labels
from spinefairbench.metrics.recommendation import classify_recommendations
from spinefairbench.metrics.refusal_detector import ResponseClass, classify_response

SUBMISSION_SCHEMA_VERSION = "spinefairbench-submission-v1"
SCORE_SCHEMA_VERSION = "spinefairbench-score-v1"
SUPPORTED_SCOPES = {
    "common-core-1000",
    "all-model-intersection-2166",
    "qc-passed",
    "toy",
}


class ScoringError(ValueError):
    """Raised when a submission cannot be scored."""


@dataclass(frozen=True)
class BenchmarkPair:
    pair_id: str
    source_id: str
    edit_label: str
    counterfactual_image_path: str | None = None


@dataclass(frozen=True)
class PairScore:
    pair_id: str
    source_id: str
    source_response_class: str
    counterfactual_response_class: str
    source_recommendations: list[str]
    counterfactual_recommendations: list[str]
    source_diagnostic_labels: list[str]
    counterfactual_diagnostic_labels: list[str]
    recommendation_changed: float
    diagnostic_label_consistency: float


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoringError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ScoringError(f"JSONL row must be an object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _source_id_from_pair_id(pair_id: str) -> str:
    return pair_id.split("__", 1)[0] if "__" in pair_id else pair_id


def _load_qc_pair_index(artifacts_root: Path) -> dict[str, BenchmarkPair]:
    manifest = artifacts_root / "dataset" / "qc_passed_pair_manifest.jsonl"
    if not manifest.exists():
        raise ScoringError(f"Missing QC-passed pair manifest: {manifest}")
    out: dict[str, BenchmarkPair] = {}
    for row in _load_jsonl(manifest):
        pair_id = row.get("pair_id")
        source_id = row.get("source_id")
        if not isinstance(pair_id, str) or not pair_id:
            continue
        if not isinstance(source_id, str) or not source_id:
            source_id = _source_id_from_pair_id(pair_id)
        out[pair_id] = BenchmarkPair(
            pair_id=pair_id,
            source_id=source_id,
            edit_label=str(row.get("edit_label") or ""),
            counterfactual_image_path=(
                str(row.get("counterfactual_image_path"))
                if row.get("counterfactual_image_path")
                else None
            ),
        )
    if not out:
        raise ScoringError(f"No usable pair rows found in {manifest}")
    return out


def _load_common_core_source_ids(artifacts_root: Path) -> set[str]:
    path = artifacts_root / "artifacts" / "freeze_runs" / "2026-04-09" / "evaluation_source_subset_core1000.json"
    payload = _load_json(path)
    ids = payload.get("source_ids") if isinstance(payload, dict) else None
    if not isinstance(ids, list) or not ids:
        raise ScoringError(f"Missing source_ids list in {path}")
    return {str(value) for value in ids if str(value)}


def _load_intersection_pair_ids(artifacts_root: Path) -> set[str]:
    path = artifacts_root / "artifacts" / "Results" / "final_inputs" / "all_model_intersection_2166_manifest.json"
    payload = _load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ScoringError(f"Missing records list in {path}")
    return {
        str(record.get("pair_id"))
        for record in records
        if isinstance(record, dict) and str(record.get("pair_id") or "")
    }


def load_benchmark_pairs(
    artifacts_root: Path,
    scope: str,
    submitted_pair_ids: set[str] | None = None,
) -> dict[str, BenchmarkPair]:
    """Load the benchmark pair index for a supported scoring scope."""

    qc_index = _load_qc_pair_index(artifacts_root)
    if scope == "qc-passed":
        return qc_index
    if scope == "common-core-1000":
        source_ids = _load_common_core_source_ids(artifacts_root)
        return {
            pair_id: pair
            for pair_id, pair in qc_index.items()
            if pair.source_id in source_ids
        }
    if scope == "all-model-intersection-2166":
        pair_ids = _load_intersection_pair_ids(artifacts_root)
        missing = sorted(pair_ids - set(qc_index))
        if missing:
            raise ScoringError(
                "All-model intersection contains pair IDs absent from QC manifest: "
                + ", ".join(missing[:5])
            )
        return {pair_id: qc_index[pair_id] for pair_id in sorted(pair_ids)}
    if scope == "toy":
        if not submitted_pair_ids:
            raise ScoringError("Toy scope requires at least one submitted result")
        missing = sorted(submitted_pair_ids - set(qc_index))
        if missing:
            raise ScoringError(
                "Toy submission contains unknown pair IDs: " + ", ".join(missing[:5])
            )
        return {pair_id: qc_index[pair_id] for pair_id in sorted(submitted_pair_ids)}
    raise ScoringError(f"Unsupported scope {scope!r}; choose one of {sorted(SUPPORTED_SCOPES)}")


def validate_submission_payload(payload: Any) -> dict[str, Any]:
    """Validate the SpineFairBench JSON submission shape used by the scorer."""

    if not isinstance(payload, dict):
        raise ScoringError("Submission must be a JSON object")
    if payload.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
        raise ScoringError(
            f"schema_version must be {SUBMISSION_SCHEMA_VERSION!r}"
        )
    scope = payload.get("scope")
    if scope not in SUPPORTED_SCOPES:
        raise ScoringError(f"scope must be one of {sorted(SUPPORTED_SCOPES)}")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ScoringError("model must be an object")
    model_name = model.get("name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ScoringError("model.name must be a non-empty string")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ScoringError("results must be a non-empty list")
    allowed_entry_keys = {
        "pair_id",
        "source_id",
        "source_report",
        "counterfactual_report",
        "metadata",
    }
    seen: set[str] = set()
    for idx, entry in enumerate(results):
        if not isinstance(entry, dict):
            raise ScoringError(f"results[{idx}] must be an object")
        extra = sorted(set(entry) - allowed_entry_keys)
        if extra:
            raise ScoringError(f"results[{idx}] has unsupported keys: {extra}")
        pair_id = entry.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ScoringError(f"results[{idx}].pair_id must be a non-empty string")
        if pair_id in seen:
            raise ScoringError(f"Duplicate pair_id in submission: {pair_id}")
        seen.add(pair_id)
        for field in ("source_report", "counterfactual_report"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ScoringError(f"results[{idx}].{field} must be a non-empty string")
        source_id = entry.get("source_id")
        if source_id is not None and (not isinstance(source_id, str) or not source_id.strip()):
            raise ScoringError(f"results[{idx}].source_id must be a non-empty string when present")
        metadata = entry.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ScoringError(f"results[{idx}].metadata must be an object when present")
    return payload


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ScoringError("Cannot compute percentile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def source_clustered_bootstrap_ci(
    values: list[float],
    source_ids: list[str],
    *,
    iterations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap over source-ID clusters using only the standard library."""

    if not values:
        raise ScoringError("Cannot bootstrap an empty metric vector")
    if len(values) != len(source_ids):
        raise ScoringError("values and source_ids must have equal length")
    if iterations < 1:
        raise ScoringError("bootstrap iterations must be >= 1")
    by_source: dict[str, list[float]] = {}
    for value, source_id in zip(values, source_ids, strict=True):
        by_source.setdefault(source_id, []).append(float(value))
    clusters = sorted(by_source)
    rng = random.Random(seed)
    sampled_means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        count = 0
        for _cluster_idx in range(len(clusters)):
            cluster = rng.choice(clusters)
            cluster_values = by_source[cluster]
            total += sum(cluster_values)
            count += len(cluster_values)
        sampled_means.append(total / count)
    return _percentile(sampled_means, 0.025), _percentile(sampled_means, 0.975)


def _score_pair(entry: dict[str, Any], benchmark_pair: BenchmarkPair) -> PairScore | None:
    source_report = str(entry["source_report"])
    counterfactual_report = str(entry["counterfactual_report"])
    source_class = classify_response(source_report)
    counterfactual_class = classify_response(counterfactual_report)
    if source_class == ResponseClass.FULL_REFUSAL or counterfactual_class == ResponseClass.FULL_REFUSAL:
        return None
    source_recs = sorted(classify_recommendations(source_report))
    counterfactual_recs = sorted(classify_recommendations(counterfactual_report))
    source_labels = sorted(extract_labels(source_report))
    counterfactual_labels = sorted(extract_labels(counterfactual_report))
    return PairScore(
        pair_id=benchmark_pair.pair_id,
        source_id=benchmark_pair.source_id,
        source_response_class=source_class.value,
        counterfactual_response_class=counterfactual_class.value,
        source_recommendations=source_recs,
        counterfactual_recommendations=counterfactual_recs,
        source_diagnostic_labels=source_labels,
        counterfactual_diagnostic_labels=counterfactual_labels,
        recommendation_changed=float(source_recs != counterfactual_recs),
        diagnostic_label_consistency=compute_jaccard(set(source_labels), set(counterfactual_labels)),
    )


def score_submission_payload(
    artifacts_root: Path,
    submission_payload: dict[str, Any],
    *,
    scope: str | None = None,
    allow_partial: bool = False,
    bootstrap_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Score pre-computed SpineFairBench model outputs against a released scope."""

    validated = validate_submission_payload(submission_payload)
    selected_scope = scope or str(validated["scope"])
    if selected_scope not in SUPPORTED_SCOPES:
        raise ScoringError(f"Unsupported scope {selected_scope!r}")
    submitted_by_pair = {str(entry["pair_id"]): entry for entry in validated["results"]}
    benchmark_pairs = load_benchmark_pairs(
        artifacts_root,
        selected_scope,
        submitted_pair_ids=set(submitted_by_pair) if selected_scope == "toy" else None,
    )
    expected_ids = set(benchmark_pairs)
    submitted_ids = set(submitted_by_pair)
    unexpected = sorted(submitted_ids - expected_ids)
    if unexpected:
        raise ScoringError(
            "Submission contains pair_id values outside the selected scope: "
            + ", ".join(unexpected[:10])
        )
    missing = sorted(expected_ids - submitted_ids)
    if missing and not allow_partial:
        raise ScoringError(
            f"Submission is missing {len(missing)} required pair(s) for scope {selected_scope}. "
            f"First missing IDs: {missing[:10]}. Use --allow-partial only for smoke tests."
        )

    pair_scores: list[PairScore] = []
    pair_quality = {
        "submitted_pairs": len(submitted_ids),
        "source_full_refusals": 0,
        "counterfactual_full_refusals": 0,
        "pair_full_refusals": 0,
        "pair_partial_refusals": 0,
        "usable_pairs": 0,
    }
    per_pair_results: list[dict[str, Any]] = []
    for pair_id in sorted(submitted_ids & expected_ids):
        entry = submitted_by_pair[pair_id]
        benchmark_pair = benchmark_pairs[pair_id]
        declared_source_id = entry.get("source_id")
        if declared_source_id is not None and declared_source_id != benchmark_pair.source_id:
            raise ScoringError(
                f"source_id mismatch for {pair_id}: {declared_source_id!r} != {benchmark_pair.source_id!r}"
            )
        source_class = classify_response(str(entry["source_report"]))
        counterfactual_class = classify_response(str(entry["counterfactual_report"]))
        if source_class == ResponseClass.FULL_REFUSAL:
            pair_quality["source_full_refusals"] += 1
        if counterfactual_class == ResponseClass.FULL_REFUSAL:
            pair_quality["counterfactual_full_refusals"] += 1
        if source_class == ResponseClass.FULL_REFUSAL or counterfactual_class == ResponseClass.FULL_REFUSAL:
            pair_quality["pair_full_refusals"] += 1
            per_pair_results.append(
                {
                    "pair_id": pair_id,
                    "source_id": benchmark_pair.source_id,
                    "usable": False,
                    "source_response_class": source_class.value,
                    "counterfactual_response_class": counterfactual_class.value,
                }
            )
            continue
        if source_class == ResponseClass.PARTIAL_REFUSAL or counterfactual_class == ResponseClass.PARTIAL_REFUSAL:
            pair_quality["pair_partial_refusals"] += 1
        scored = _score_pair(entry, benchmark_pair)
        if scored is None:
            continue
        pair_scores.append(scored)
        pair_quality["usable_pairs"] += 1
        per_pair_results.append(
            {
                "pair_id": scored.pair_id,
                "source_id": scored.source_id,
                "usable": True,
                "source_response_class": scored.source_response_class,
                "counterfactual_response_class": scored.counterfactual_response_class,
                "source_recommendations": scored.source_recommendations,
                "counterfactual_recommendations": scored.counterfactual_recommendations,
                "source_diagnostic_labels": scored.source_diagnostic_labels,
                "counterfactual_diagnostic_labels": scored.counterfactual_diagnostic_labels,
                "recommendation_changed": scored.recommendation_changed,
                "diagnostic_label_consistency": scored.diagnostic_label_consistency,
            }
        )
    if not pair_scores:
        raise ScoringError("No usable pairs remain after full-refusal filtering")

    rec_values = [score.recommendation_changed for score in pair_scores]
    diag_values = [score.diagnostic_label_consistency for score in pair_scores]
    source_ids = [score.source_id for score in pair_scores]
    rec_point = sum(rec_values) / len(rec_values)
    diag_point = sum(diag_values) / len(diag_values)
    rec_ci = source_clustered_bootstrap_ci(
        rec_values,
        source_ids,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    diag_ci = source_clustered_bootstrap_ci(
        diag_values,
        source_ids,
        iterations=bootstrap_iterations,
        seed=seed + 1,
    )
    model = validated["model"]
    warnings: list[str] = []
    if missing:
        warnings.append(
            f"Partial submission: {len(missing)} expected pair(s) absent from scope {selected_scope}."
        )
    comparable = not missing and selected_scope != "toy"
    result = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "scored_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "model": model,
        "scope": selected_scope,
        "scoring_config": {
            "bootstrap": "source_clustered_percentile",
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": seed,
            "full_refusal_policy": "exclude pair from primary endpoints",
            "partial_refusal_policy": "include pair in primary endpoints",
        },
        "coverage": {
            "expected_pairs": len(expected_ids),
            "submitted_pairs": len(submitted_ids),
            "missing_pairs": len(missing),
            "unexpected_pairs": len(unexpected),
            "usable_pairs": len(pair_scores),
            "unique_sources_usable": len(set(source_ids)),
            "comparable_to_panel_scope": comparable,
        },
        "data_quality": pair_quality,
        "primary_endpoints": {
            "recommendation_change_rate": {
                "point_estimate": rec_point,
                "ci95": {"lower": rec_ci[0], "upper": rec_ci[1]},
                "n_pairs": len(pair_scores),
            },
            "diagnostic_label_consistency": {
                "point_estimate": diag_point,
                "ci95": {"lower": diag_ci[0], "upper": diag_ci[1]},
                "n_pairs": len(pair_scores),
            },
        },
        "table2_row": {
            "model_name": model["name"],
            "scope": selected_scope,
            "n_pairs": len(pair_scores),
            "recommendation_change_rate": round(rec_point, 3),
            "recommendation_change_ci95": [round(rec_ci[0], 3), round(rec_ci[1], 3)],
            "diagnostic_label_consistency": round(diag_point, 3),
            "diagnostic_label_consistency_ci95": [round(diag_ci[0], 3), round(diag_ci[1], 3)],
        },
        "warnings": warnings,
        "per_pair_results": per_pair_results,
    }
    return result


def score_submission_file(
    artifacts_root: Path,
    submission_file: Path,
    *,
    scope: str | None = None,
    allow_partial: bool = False,
    bootstrap_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Load and score a SpineFairBench submission JSON file."""

    payload = _load_json(submission_file)
    return score_submission_payload(
        artifacts_root,
        payload,
        scope=scope,
        allow_partial=allow_partial,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )


def _print_summary(score: dict[str, Any], output_file: Path) -> None:
    row = score["table2_row"]
    coverage = score["coverage"]
    print("SpineFairBench scoring: OK")
    print("Model:", row["model_name"])
    print("Scope:", row["scope"])
    print("Submitted pairs:", coverage["submitted_pairs"])
    print("Usable pairs:", row["n_pairs"])
    print(
        "Recommendation change:",
        f"{row['recommendation_change_rate']:.3f}",
        f"[{row['recommendation_change_ci95'][0]:.3f}, {row['recommendation_change_ci95'][1]:.3f}]",
    )
    print(
        "Diagnostic-label consistency:",
        f"{row['diagnostic_label_consistency']:.3f}",
        f"[{row['diagnostic_label_consistency_ci95'][0]:.3f}, {row['diagnostic_label_consistency_ci95'][1]:.3f}]",
    )
    print("Score JSON:", output_file)
    for warning in score.get("warnings", []):
        print("Warning:", warning)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score pre-computed model outputs for SpineFairBench.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="Score a submission JSON file.")
    score.add_argument("--artifacts", required=True, help="Path to the companion artifacts directory.")
    score.add_argument("--submission", required=True, help="Path to a SpineFairBench submission JSON file.")
    score.add_argument("--output", required=True, help="Path where score JSON should be written.")
    score.add_argument(
        "--scope",
        choices=sorted(SUPPORTED_SCOPES),
        default=None,
        help="Override the submission scope. Defaults to the submission's scope field.",
    )
    score.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing expected pairs. Use for toy/smoke tests, not comparable benchmark submissions.",
    )
    score.add_argument("--bootstrap-iterations", type=int, default=10000)
    score.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            output_file = Path(args.output).resolve()
            score = score_submission_file(
                Path(args.artifacts).resolve(),
                Path(args.submission).resolve(),
                scope=args.scope,
                allow_partial=args.allow_partial,
                bootstrap_iterations=args.bootstrap_iterations,
                seed=args.seed,
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(score, f, indent=2, sort_keys=True)
                f.write("\n")
            _print_summary(score, output_file)
            return 0
    except ScoringError as exc:
        print(f"Scoring error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unhandled command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
