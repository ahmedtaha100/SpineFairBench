from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = CODE_ROOT.parent / "artifacts"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from spinefairbench.metrics.diagnostic_label import PATHOLOGY_SYNONYMS, compute_jaccard, extract_labels
from spinefairbench.metrics.recommendation import classify_recommendations
from spinefairbench.metrics.refusal_detector import ResponseClass, classify_response
from spinefairbench.release.scoring import source_clustered_bootstrap_ci

RETAINED_TABLE2_MODELS = (
    "gpt-5.4",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "glm-4.6v",
    "kimi-k2.5",
    "gemma-4",
    "llama-4-scout",
    "qwen2.5-vl",
    "radfm",
)


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def _artifact_root(args: argparse.Namespace) -> Path:
    value = getattr(args, "artifacts", None) or getattr(args, "root", None)
    return Path(value or DEFAULT_ARTIFACT_ROOT).resolve()


def _required(root: Path, rel: str, *, kind: str = "artifact") -> Path:
    path = root / rel
    if not path.exists():
        raise SystemExit(f"Missing required {kind}: {rel}")
    return path


def _panel_paths(root: Path, panel: str) -> tuple[Path, Path, Path]:
    base = root / "artifacts" / "Results" / "final_inputs" / "panels" / panel
    return (
        _required(root, str(base.relative_to(root) / "panel_manifest.json")),
        _required(root, str(base.relative_to(root) / "pairs.json")),
        _required(root, str(base.relative_to(root) / "evaluation_results.json")),
    )


def command_inspect(args: argparse.Namespace) -> None:
    root = _artifact_root(args)
    required_artifacts = [
        "README.md",
        "DATA_LICENSE.md",
        "SHA256SUMS.txt",
        "benchmark_card.md",
        "croissant.json",
        "dataset/README.md",
        "dataset/release_manifest.json",
        "dataset/qc_metadata.jsonl",
        "dataset/qc_passed_pair_manifest.jsonl",
        "dataset/pairs.json",
        "dataset/source_metadata.json",
        "artifacts/freeze_runs/2026-04-09/canonical_definitions.json",
        "artifacts/freeze_runs/2026-04-09/evaluation_source_subset.json",
        "artifacts/freeze_runs/2026-04-09/evaluation_source_subset_core1000.json",
        "artifacts/freeze_runs/2026-04-09/exclusion_evidence/gemini_baseline_backfill_canonical_2026-04-16.json",
        "artifacts/freeze_runs/2026-04-09/exclusion_evidence/medgemma_baseline_canary_canonical.json",
        "artifacts/Results/analysis/common_core_1000_summary.json",
        "artifacts/Results/analysis/generator_evaluation_overlap.json",
        "artifacts/Results/analysis/mitigation_conditions_manifest.json",
        "artifacts/Results/analysis/mitigation_parsing_confidence.json",
        "artifacts/Results/analysis/mitigation_stage1_trace_manifest.json",
        "artifacts/Results/analysis/mitigation_summary.json",
        "artifacts/Results/final_inputs/all_model_intersection_2166_manifest.json",
        "artifacts/Results/final_inputs/final_panel_freeze_manifest.json",
        "protocol/binding_rule_2026-04-08.md",
        "protocol/generator_artifact_controls_c1_c4_registration.json",
        "validation_report_public_final_2026-04-28.json",
        "radiologist_validated_subset.json",
        "radiologist_exclusion_list.json",
    ]
    for rel in required_artifacts:
        _required(root, rel)
    if not (
        (root / "artifacts/Results/analysis/mitigation_stage1_confidence_sample_manifest.json").exists()
        or (root / "artifacts/Results/analysis/mitigation_stage1_trace_manifest.json").exists()
    ):
        raise SystemExit("Missing required artifact: Stage-1 confidence sample manifest or trace manifest")
    for panel in ("full_pipeline_retained", "baseline_only_retained", "full_pipeline_mitigation_retained"):
        base = root / "artifacts" / "Results" / "final_inputs" / "panels" / panel
        _required(root, str(base.relative_to(root) / "panel_manifest.json"))
        _required(root, str(base.relative_to(root) / "evaluation_results.json"))

    prompt_path = CODE_ROOT / "prompts" / "canonical_definitions.json"
    _required(CODE_ROOT, str(prompt_path.relative_to(CODE_ROOT)), kind="code file")
    prompt_registry = _load_json(prompt_path).get("prompt_registry", {})
    system_prompt = prompt_registry.get("system")
    primary_prompt = prompt_registry.get("primary")
    if not isinstance(system_prompt, str) or not isinstance(primary_prompt, str):
        raise SystemExit("Locked canonical prompt registry is missing system/primary prompts")
    summary = _load_json(root / "artifacts" / "Results" / "analysis" / "common_core_1000_summary.json")
    common_pairs = summary.get("common_pairs", {})
    all9 = common_pairs.get("intersection_9_all_models") or common_pairs.get("all9", {}).get("count")
    all9_manifest = _load_json(root / "artifacts" / "Results" / "final_inputs" / "all_model_intersection_2166_manifest.json")
    panel_freeze = _load_json(root / "artifacts" / "Results" / "final_inputs" / "final_panel_freeze_manifest.json")

    print("Submission package inspection: OK")
    print("Artifact root:", root)
    print("Frozen manifests: present")
    print("Common-core source manifest: present")
    print("All-model intersection count:", all9)
    print("All-model intersection manifest rows:", all9_manifest["pair_count"])
    print("Panel freeze timestamp:", panel_freeze.get("prepared_at_utc"))
    dataset_manifest = _load_json(root / "dataset" / "release_manifest.json")
    print("Released counterfactual PNGs:", dataset_manifest["counterfactual_images"]["included_png_count"])
    print("QC-passed pair manifest rows:", dataset_manifest["pair_manifests"]["qc_passed_pair_rows"])
    print("Raw source PNGs in public bundle:", dataset_manifest["source_radiographs"]["source_png_files_in_public_bundle"])
    print("Panel manifests and retained output files: present")
    print("Dropped-model exclusion evidence: present")
    print("Public radiologist-validation artifacts: present")
    print("Benchmark card and Croissant metadata: present")
    print("Locked prompt file:", prompt_path.relative_to(CODE_ROOT))
    print("Prompt excerpt:")
    print("System:", system_prompt)
    print("Primary:", primary_prompt)


def command_dataset(args: argparse.Namespace) -> None:
    root = _artifact_root(args)
    manifest = _load_json(_required(root, "dataset/release_manifest.json"))
    passed_path = _required(root, "dataset/qc_passed_pair_manifest.jsonl")
    qc_path = _required(root, "dataset/qc_metadata.jsonl")
    image_root = _required(root, "dataset/counterfactual_images")

    included_png_count = sum(1 for _ in image_root.rglob("*.png"))
    source_png_count = sum(1 for _ in image_root.rglob("source.png"))
    passed_rows = _count_lines(passed_path)
    qc_rows = _count_lines(qc_path)

    expected_pngs = manifest["counterfactual_images"]["included_png_count"]
    expected_passed = manifest["pair_manifests"]["qc_passed_pair_rows"]
    expected_qc_rows = manifest["pair_manifests"]["attempted_qc_rows"]

    print("Dataset release inspection: OK")
    print("Counterfactual image root:", image_root.relative_to(root))
    print("Released QC-passed PNGs:", included_png_count)
    print("QC-passed pair manifest rows:", passed_rows)
    print("Attempted QC metadata rows:", qc_rows)
    print("Raw source PNGs in public bundle:", source_png_count)
    print("Source radiograph policy:", manifest["source_radiographs"]["release_status"])

    if included_png_count != expected_pngs:
        raise SystemExit(f"Unexpected released PNG count: {included_png_count} != {expected_pngs}")
    if passed_rows != expected_passed:
        raise SystemExit(f"Unexpected QC-passed manifest row count: {passed_rows} != {expected_passed}")
    if qc_rows != expected_qc_rows:
        raise SystemExit(f"Unexpected QC metadata row count: {qc_rows} != {expected_qc_rows}")
    if source_png_count != 0:
        raise SystemExit("Raw source PNG files are present in the public dataset tree")


def _panel_for_model(root: Path, model: str) -> str:
    panels = ("full_pipeline_retained", "baseline_only_retained")
    available: dict[str, list[str]] = {}
    for panel in panels:
        manifest_path, _, _ = _panel_paths(root, panel)
        manifest = _load_json(manifest_path)
        models = manifest.get("models")
        if isinstance(models, list):
            available[panel] = [str(value) for value in models]
            if model in available[panel]:
                return panel
    known_models = sorted({model_name for models in available.values() for model_name in models})
    raise SystemExit(
        f"Model {model!r} is not in the retained full or baseline panels. "
        f"Available models: {', '.join(known_models)}"
    )


def _paired_records(root: Path, model: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    panel = _panel_for_model(root, model)
    _, _, eval_path = _panel_paths(root, panel)
    rows = _load_json(eval_path)
    if not isinstance(rows, list):
        raise SystemExit("evaluation_results.json must contain a list")

    sources: dict[str, dict[str, Any]] = {}
    generated: list[dict[str, Any]] = []
    for row in rows:
        if row.get("model") != model or row.get("error"):
            continue
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            continue
        if row.get("image_role") == "source":
            sources[pair_id] = row
        elif row.get("image_role") == "generated":
            generated.append(row)
    return sources, generated


def _pair_refusal_classes(source_report: str, generated_report: str) -> tuple[ResponseClass, ResponseClass]:
    return classify_response(source_report), classify_response(generated_report)


def _is_usable_endpoint_pair(source_report: str, generated_report: str) -> bool:
    source_class, generated_class = _pair_refusal_classes(source_report, generated_report)
    return (
        source_class != ResponseClass.FULL_REFUSAL
        and generated_class != ResponseClass.FULL_REFUSAL
    )


def _set_jaccard(values_a: set[str], values_b: set[str]) -> float:
    if not values_a and not values_b:
        return 1.0
    union = values_a | values_b
    return len(values_a & values_b) / len(union)


def _source_id_for_record(src: dict[str, Any], gen: dict[str, Any]) -> str:
    for row in (gen, src):
        value = row.get("pair_source_id") or row.get("source_id")
        if isinstance(value, str) and value:
            return value
    pair_id = str(gen.get("pair_id") or src.get("pair_id") or "")
    return pair_id.split("__", 1)[0] if "__" in pair_id else pair_id


def _endpoint_metrics(root: Path, model: str) -> dict[str, Any]:
    sources, generated = _paired_records(root, model)
    recommendation_matches: list[bool] = []
    recommendation_changed: list[float] = []
    recommendation_jaccards: list[float] = []
    diagnostic_jaccards: list[float] = []
    source_ids: list[str] = []
    both_empty_diagnostic = 0
    full_refusal_pairs = 0
    partial_refusal_pairs = 0

    for gen in generated:
        src = sources.get(gen["pair_id"])
        if not src:
            continue
        source_report = src.get("response", "")
        generated_report = gen.get("response", "")
        source_class, generated_class = _pair_refusal_classes(source_report, generated_report)
        if source_class == ResponseClass.FULL_REFUSAL or generated_class == ResponseClass.FULL_REFUSAL:
            full_refusal_pairs += 1
            continue
        if source_class == ResponseClass.PARTIAL_REFUSAL or generated_class == ResponseClass.PARTIAL_REFUSAL:
            partial_refusal_pairs += 1

        source_recs = classify_recommendations(source_report)
        generated_recs = classify_recommendations(generated_report)
        rec_match = source_recs == generated_recs
        recommendation_matches.append(rec_match)
        recommendation_changed.append(float(not rec_match))
        recommendation_jaccards.append(_set_jaccard(source_recs, generated_recs))

        source_labels = extract_labels(source_report)
        generated_labels = extract_labels(generated_report)
        if not source_labels and not generated_labels:
            both_empty_diagnostic += 1
        diagnostic_jaccards.append(compute_jaccard(source_labels, generated_labels))
        source_ids.append(_source_id_for_record(src, gen))

    if not recommendation_matches:
        raise SystemExit(f"No usable pairs found for model {model}")

    return {
        "recommendation_matches": recommendation_matches,
        "recommendation_changed": recommendation_changed,
        "recommendation_jaccards": recommendation_jaccards,
        "diagnostic_jaccards": diagnostic_jaccards,
        "source_ids": source_ids,
        "both_empty_diagnostic": both_empty_diagnostic,
        "full_refusal_pairs": full_refusal_pairs,
        "partial_refusal_pairs": partial_refusal_pairs,
    }


def command_parse_sample(args: argparse.Namespace) -> None:
    root = _artifact_root(args)
    sources, generated = _paired_records(root, args.model)
    for gen in generated:
        src = sources.get(gen["pair_id"])
        if src:
            source_report = src.get("response", "")
            generated_report = gen.get("response", "")
            if not _is_usable_endpoint_pair(source_report, generated_report):
                continue
            source_recs = classify_recommendations(source_report)
            generated_recs = classify_recommendations(generated_report)
            source_labels = extract_labels(source_report)
            generated_labels = extract_labels(generated_report)
            print("Model:", args.model)
            print("Pair ID:", gen["pair_id"])
            print("Source recommendation categories:", sorted(source_recs))
            print("Generated recommendation categories:", sorted(generated_recs))
            print("Source diagnostic labels:", sorted(source_labels))
            print("Generated diagnostic labels:", sorted(generated_labels))
            print("Diagnostic-label Jaccard:", f"{compute_jaccard(source_labels, generated_labels):.3f}")
            return
    raise SystemExit(f"No paired records found for model {args.model}")


def command_table2(args: argparse.Namespace) -> None:
    root = _artifact_root(args)
    metrics = _endpoint_metrics(root, args.model)
    recommendation_matches = metrics["recommendation_matches"]
    diagnostic_jaccards = metrics["diagnostic_jaccards"]

    agreement = sum(recommendation_matches) / len(recommendation_matches)
    rec_change = 1.0 - agreement
    diag_consistency = sum(diagnostic_jaccards) / len(diagnostic_jaccards)

    summary_path = _required(root, "artifacts/Results/analysis/common_core_1000_summary.json")
    summary = _load_json(summary_path)
    panel = "full" if args.model in summary["panels"]["full"]["models"] else "baseline"
    frozen_model = summary["panels"][panel]["models"][args.model]
    frozen = frozen_model["primary_secondary_stats"]
    frozen_quality = frozen_model.get("data_quality", {})
    frozen_rec_change = 1.0 - frozen["recommendation"]["agreement_rate"]
    frozen_diag = frozen["diagnostic_label"]["mean"]
    rec_ci = frozen["recommendation"]["bootstrap_ci"]
    rec_change_ci = (1.0 - rec_ci["upper"], 1.0 - rec_ci["lower"])
    diag_ci = frozen["diagnostic_label"]["bootstrap_ci"]

    print("Model:", args.model)
    print("Usable pairs:", len(recommendation_matches))
    print("Full-refusal pairs excluded:", metrics["full_refusal_pairs"])
    print("Partial-refusal pairs included:", metrics["partial_refusal_pairs"])
    if "pair_usable" in frozen_quality:
        print("Frozen summary usable pairs:", frozen_quality["pair_usable"])
    print("Recomputed recommendation change:", f"{rec_change:.6f}", f"(rounded: {rec_change:.3f})")
    print("Frozen summary recommendation change:", f"{frozen_rec_change:.6f}", f"(rounded: {frozen_rec_change:.3f})")
    print(
        "Frozen summary recommendation change 95% CI (read, not recomputed):",
        f"[{rec_change_ci[0]:.3f}, {rec_change_ci[1]:.3f}]",
    )
    print("Recomputed diagnostic consistency:", f"{diag_consistency:.6f}", f"(rounded: {diag_consistency:.3f})")
    print("Frozen summary diagnostic consistency:", f"{frozen_diag:.6f}", f"(rounded: {frozen_diag:.3f})")
    print(
        "Frozen summary diagnostic consistency 95% CI (read, not recomputed):",
        f"[{diag_ci['lower']:.3f}, {diag_ci['upper']:.3f}]",
    )
    if args.recompute_ci:
        recomputed_rec_ci = source_clustered_bootstrap_ci(
            metrics["recommendation_changed"],
            metrics["source_ids"],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        recomputed_diag_ci = source_clustered_bootstrap_ci(
            metrics["diagnostic_jaccards"],
            metrics["source_ids"],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        print(
            "Recomputed recommendation change 95% CI:",
            f"[{recomputed_rec_ci[0]:.3f}, {recomputed_rec_ci[1]:.3f}]",
        )
        print(
            "Recomputed diagnostic consistency 95% CI:",
            f"[{recomputed_diag_ci[0]:.3f}, {recomputed_diag_ci[1]:.3f}]",
        )
        print(
            "CI recomputation settings:",
            f"source_clustered_percentile iterations={args.bootstrap_iterations} seed={args.seed}",
        )
    else:
        print("CI mode: frozen summary read; pass --recompute-ci to run source-clustered bootstrap.")
    if "pair_usable" in frozen_quality and len(recommendation_matches) != int(frozen_quality["pair_usable"]):
        raise SystemExit("Recomputed usable-pair count does not match frozen summary")
    if abs(rec_change - frozen_rec_change) > 1e-12:
        raise SystemExit("Recomputed recommendation change does not match frozen summary")
    if abs(diag_consistency - frozen_diag) > 1e-12:
        raise SystemExit("Recomputed diagnostic consistency does not match frozen summary")


def command_diagnostic_scoring(args: argparse.Namespace) -> None:
    print("Frozen Table 2 diagnostic scoring path:")
    print("  spinefairbench.metrics.diagnostic_label.extract_labels()")
    print("  spinefairbench.metrics.diagnostic_label.compute_jaccard()")
    print("Released diagnostic-label registry size:", len(PATHOLOGY_SYNONYMS))
    print("Released diagnostic-label categories:", ", ".join(sorted(PATHOLOGY_SYNONYMS)))
    print("Both-empty diagnostic-label Jaccard:", f"{compute_jaccard(set(), set()):.1f}")
    print(
        "Archival tokenized helper:",
        "spinefairbench.analysis.endpoints._extract_diagnosis_tokens is not used for frozen Table 2.",
    )


def command_gap_sensitivity(args: argparse.Namespace) -> None:
    root = _artifact_root(args)
    models = [args.model] if args.model else list(RETAINED_TABLE2_MODELS)
    gap_exact_values: list[float] = []
    gap_graded_values: list[float] = []

    print("model,usable_pairs,rec_change,rec_exact_stability,rec_jaccard,diag_jaccard,gap_exact,gap_graded")
    for model in models:
        metrics = _endpoint_metrics(root, model)
        usable = len(metrics["recommendation_changed"])
        rec_change = sum(metrics["recommendation_changed"]) / usable
        rec_exact_stability = 1.0 - rec_change
        rec_jaccard = sum(metrics["recommendation_jaccards"]) / usable
        diag_jaccard = sum(metrics["diagnostic_jaccards"]) / usable
        gap_exact = diag_jaccard - rec_exact_stability
        gap_graded = diag_jaccard - rec_jaccard
        gap_exact_values.append(gap_exact)
        gap_graded_values.append(gap_graded)
        print(
            ",".join(
                [
                    model,
                    str(usable),
                    f"{rec_change:.6f}",
                    f"{rec_exact_stability:.6f}",
                    f"{rec_jaccard:.6f}",
                    f"{diag_jaccard:.6f}",
                    f"{gap_exact:.6f}",
                    f"{gap_graded:.6f}",
                ]
            )
        )

    print("median_gap_exact:", f"{statistics.median(gap_exact_values):.6f}")
    print("median_gap_graded:", f"{statistics.median(gap_graded_values):.6f}")
    print("gap_graded_negative_models:", f"{sum(value < 0 for value in gap_graded_values)}/{len(gap_graded_values)}")


def command_both_empty_diagnostic(args: argparse.Namespace) -> None:
    root = _artifact_root(args)
    models = [args.model] if args.model else list(RETAINED_TABLE2_MODELS)
    pooled_empty = 0
    pooled_usable = 0

    print("model,usable_pairs,both_empty_diagnostic_label_pairs,rate")
    for model in models:
        metrics = _endpoint_metrics(root, model)
        usable = len(metrics["diagnostic_jaccards"])
        both_empty = int(metrics["both_empty_diagnostic"])
        pooled_empty += both_empty
        pooled_usable += usable
        print(model, usable, both_empty, f"{both_empty / usable:.4%}", sep=",")

    print("pooled_both_empty:", f"{pooled_empty}/{pooled_usable}", f"({pooled_empty / pooled_usable:.4%})")


def command_radiologist(args: argparse.Namespace) -> None:
    root = _artifact_root(args)
    report = _load_json(_required(root, "validation_report_public_final_2026-04-28.json"))
    validated = _load_json(_required(root, "radiologist_validated_subset.json"))
    exclusions = _load_json(_required(root, "radiologist_exclusion_list.json"))
    public_manifest = _load_json(_required(root, "validation_public/validation_public_manifest.json"))
    detectability_path = _required(root, "validation_public/detectability_responses_public_2026-04-28.csv")
    hidden_path = _required(root, "validation_public/hidden_repeat_records_public_2026-04-28.csv")
    pair_results_path = _required(root, "validation_public/validation_pair_results_public_2026-04-28.csv")

    passing_ids = validated.get("pair_ids", [])
    excluded_pairs = exclusions.get("excluded_pairs", [])
    with open(detectability_path, newline="", encoding="utf-8") as f:
        detectability_rows = list(csv.DictReader(f))
    with open(hidden_path, newline="", encoding="utf-8") as f:
        hidden_repeat_rows = list(csv.DictReader(f))
    with open(pair_results_path, newline="", encoding="utf-8") as f:
        pair_result_rows = list(csv.DictReader(f))
    per_reviewer_rows = 0
    for rel in public_manifest["files"]["per_reviewer_files"]:
        with open(_required(root, rel), newline="", encoding="utf-8") as f:
            per_reviewer_rows += sum(1 for _ in csv.DictReader(f))

    votes: dict[str, dict[str, int]] = {}
    for row in pair_result_rows:
        pair = row["pair_id"]
        bucket = votes.setdefault(pair, {"q1_yes": 0, "q2_yes": 0, "rows": 0})
        bucket["rows"] += 1
        bucket["q1_yes"] += row["q1_clinical_plausibility"] == "Yes"
        bucket["q2_yes"] += row["q2_pathology_preservation"] == "Yes"
    majority_pass = sorted(
        pair
        for pair, counts in votes.items()
        if counts["rows"] == 3 and counts["q1_yes"] >= 2 and counts["q2_yes"] >= 2
    )
    cannot_tell = sum(row["q4_detectability"] == "Cannot tell" for row in detectability_rows)

    print("Public validation report schema:", report.get("report_schema_version"))
    print("Validated subset count:", len(passing_ids))
    print("Exclusion count:", len(excluded_pairs))
    print("Recomputed 2-of-3 majority pass:", f"{len(majority_pass)}/{len(votes)}")
    print("Detectability response rows:", len(detectability_rows))
    print("Recomputed detectability 'Cannot tell':", f"{cannot_tell}/{len(detectability_rows)}")
    print("Per-reviewer display-event rows:", per_reviewer_rows)
    print("Hidden-repeat rows:", len(hidden_repeat_rows))
    print("Expected validation result: 443 passing, 7 excluded")
    if len(passing_ids) != 443 or len(excluded_pairs) != 7:
        raise SystemExit("Unexpected public radiologist-validation counts")
    if len(detectability_rows) != 1350:
        raise SystemExit("Unexpected detectability response count")
    if per_reviewer_rows != 1380:
        raise SystemExit("Unexpected per-reviewer display-event count")
    if len(hidden_repeat_rows) != 30:
        raise SystemExit("Unexpected hidden-repeat record count")
    if len(majority_pass) != 443 or len(votes) != 450:
        raise SystemExit("Unexpected 2-of-3 radiologist majority result")
    if set(majority_pass) != set(passing_ids):
        raise SystemExit("Recomputed majority-pass IDs do not match radiologist_validated_subset.json")
    if cannot_tell != 1307:
        raise SystemExit("Unexpected 'Cannot tell' detectability count")


def _clean_markdown(text: str) -> str:
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*[-:]\s*", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def _extract_stage1_primary_diagnosis(text: str) -> str | None:
    inline = re.search(
        r"(?im)^\s*(?:\d+\.\s*)?(?:\*\*)?Primary diagnosis(?:\*\*)?\s*:\s*(.+?)\s*$",
        text,
    )
    if inline:
        value = _clean_markdown(inline.group(1))
        return value or None

    block = re.search(
        r"(?is)(?:^|\n)\s*(?:\d+\.\s*)?(?:\*\*)?Primary diagnosis(?:\*\*)?\s*:?\s*"
        r"(?:\n\s*[-*]?\s*)?"
        r"(.+?)"
        r"(?=\n\s*\d+\.\s|\n\s*(?:\*\*)?Overall severity|\n\s*(?:\*\*)?Confidence|\Z)",
        text,
    )
    if not block:
        return None
    value = _clean_markdown(block.group(1))
    return value or None


def _validation_is_clean(path: Path) -> bool:
    try:
        payload = _load_json(path)
    except Exception:
        return False
    html_flag = bool(payload.get("html_garbage_detected", payload.get("html_garbage", False)))
    return (
        payload.get("status") == "valid"
        and not bool(payload.get("prompt_echo_detected", False))
        and not html_flag
    )


def _stage1_diagnosis_parse_ok(txt_path: Path) -> bool:
    validation_path = txt_path.with_name(f"{txt_path.stem}.validation.json")
    if not validation_path.exists() or not _validation_is_clean(validation_path):
        return False
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    return _extract_stage1_primary_diagnosis(text) is not None


def command_stage1_confidence(args: argparse.Namespace) -> None:
    root = _artifact_root(args)
    summary = _load_json(_required(root, "artifacts/Results/analysis/mitigation_parsing_confidence.json"))
    sample_manifest_path = root / "artifacts/Results/analysis/mitigation_stage1_confidence_sample_manifest.json"
    if sample_manifest_path.exists():
        sample_manifest = _load_json(sample_manifest_path)
        sample_models = sample_manifest.get("models", {})
    else:
        trace_manifest = _load_json(
            _required(root, "artifacts/Results/analysis/mitigation_stage1_trace_manifest.json")
        )
        sample_models = {
            model: [Path(str(path)).name for path in payload.get("sample_files_first_200", [])]
            for model, payload in trace_manifest.get("models", {}).items()
        }
    stage1_root = _required(
        root,
        "artifacts/Results/final_inputs/panels/full_pipeline_mitigation_retained/stage1_outputs",
    )
    if not isinstance(sample_models, dict):
        raise SystemExit("Stage-1 sample manifest must contain a models object")

    print("Stage-1 parse-confidence recomputation:")
    for model in sorted(summary):
        model_dir = stage1_root / model
        sample_names = sample_models.get(model)
        if not isinstance(sample_names, list) or not sample_names:
            raise SystemExit(f"Stage-1 sample manifest is missing file list for {model}")
        sample = [model_dir / str(name) for name in sample_names]
        missing = [path.name for path in sample if not path.exists()]
        if missing:
            raise SystemExit(f"Stage-1 sample manifest references missing files for {model}: {missing[:5]}")
        pass_count = sum(_stage1_diagnosis_parse_ok(path) for path in sample)
        rate = pass_count / len(sample) if sample else 0.0
        if rate < 0.85:
            decision = "excluded_full"
        elif rate < 0.95:
            decision = "excluded_diag_only"
        else:
            decision = "included"
        frozen = summary[model]
        print(
            model,
            f"{pass_count}/{len(sample)}",
            f"({rate:.3f})",
            "decision:",
            decision,
        )
        if (
            frozen["sample_size"] != len(sample)
            or frozen["stage1_diag_pass_count"] != pass_count
            or abs(float(frozen["stage1_diag_pass_rate"]) - rate) > 1e-12
            or frozen["decision"] != decision
        ):
            raise SystemExit(f"Stage-1 parse-confidence mismatch for {model}")


def _load_mitigation_table(root: Path) -> list[dict[str, str]]:
    table_path = _required(root, "artifacts/Results/analysis/mitigation_table.csv")
    with open(table_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mitigation_row(rows: list[dict[str, str]], model: str, condition: str) -> dict[str, str]:
    for row in rows:
        if row.get("model") == model and row.get("condition") == condition:
            return row
    raise SystemExit(f"Missing mitigation-table row for {model} {condition}")


def _cell_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid mitigation-table value for {field}: {row.get(field)!r}") from exc


def command_mitigation(args: argparse.Namespace) -> None:
    root = _artifact_root(args)
    parsing = _load_json(_required(root, "artifacts/Results/analysis/mitigation_parsing_confidence.json"))
    rows = _load_mitigation_table(root)

    print("Mitigation Table 3 verification:")
    for model in ("gpt-5.4", "glm-4.6v"):
        gate = parsing.get(model, {})
        if gate.get("decision") != "included":
            raise SystemExit(f"Mitigation model {model} was not included by the Stage-1 gate")

        condition_a = _mitigation_row(rows, model, "condition_a")
        condition_b = _mitigation_row(rows, model, "condition_b")
        delta_rec = _cell_float(condition_b, "delta_rec")
        delta_diag = _cell_float(condition_b, "delta_diag")
        b_rule_pass = condition_b.get("b_rule_pass", "").strip().lower() == "true"
        if b_rule_pass or delta_rec <= 0 or delta_diag >= -0.05:
            raise SystemExit(f"Unexpected mitigation binding-rule result for {model}")

        print(
            model,
            "stage1:",
            f"{gate.get('stage1_diag_pass_count')}/{gate.get('sample_size')}",
            f"({float(gate.get('stage1_diag_pass_rate', 0.0)):.3f})",
            "condition_a rec/diag:",
            f"{_cell_float(condition_a, 'rec_change'):.3f}",
            f"{_cell_float(condition_a, 'diag_consistency'):.3f}",
            "condition_b rec/diag:",
            f"{_cell_float(condition_b, 'rec_change'):.3f}",
            f"{_cell_float(condition_b, 'diag_consistency'):.3f}",
            "delta:",
            f"{delta_rec:.3f}",
            f"{delta_diag:.3f}",
            "b_rule_pass:",
            condition_b.get("b_rule_pass"),
        )

    excluded = {
        model: payload.get("decision")
        for model, payload in parsing.items()
        if payload.get("decision") == "excluded_full"
    }
    expected_excluded = {"claude-opus-4-6", "claude-sonnet-4-6", "kimi-k2.5"}
    if set(excluded) != expected_excluded:
        raise SystemExit(f"Unexpected mitigation Stage-1 exclusions: {sorted(excluded)}")
    print("Stage-1 full exclusions:", ", ".join(sorted(excluded)))
    print("Mitigation verification: OK")


def _add_artifact_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_ROOT), help="Path to the companion artifacts folder.")
    parser.add_argument("--root", default=None, help=argparse.SUPPRESS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reviewer verification helpers for SpineFairBench submission artifacts.")
    sub = parser.add_subparsers(required=True)

    inspect = sub.add_parser("inspect")
    _add_artifact_arg(inspect)
    inspect.set_defaults(func=command_inspect)

    dataset = sub.add_parser("dataset")
    _add_artifact_arg(dataset)
    dataset.set_defaults(func=command_dataset)

    sample = sub.add_parser("parse-sample")
    _add_artifact_arg(sample)
    sample.add_argument("--model", default="gpt-5.4")
    sample.set_defaults(func=command_parse_sample)

    table2 = sub.add_parser("table2")
    _add_artifact_arg(table2)
    table2.add_argument("--model", default="gpt-5.4")
    table2.add_argument("--recompute-ci", action="store_true", help="Regenerate source-clustered bootstrap CIs from released per-pair outputs.")
    table2.add_argument("--bootstrap-iterations", type=int, default=10000)
    table2.add_argument("--seed", type=int, default=42)
    table2.set_defaults(func=command_table2)

    table3 = sub.add_parser("table3", help="Backward-compatible alias for table2.")
    _add_artifact_arg(table3)
    table3.add_argument("--model", default="gpt-5.4")
    table3.add_argument("--recompute-ci", action="store_true", help="Regenerate source-clustered bootstrap CIs from released per-pair outputs.")
    table3.add_argument("--bootstrap-iterations", type=int, default=10000)
    table3.add_argument("--seed", type=int, default=42)
    table3.set_defaults(func=command_table2)

    diagnostic = sub.add_parser("diagnostic-scoring", help="Print the frozen Table 2 diagnostic-label scoring path.")
    diagnostic.set_defaults(func=command_diagnostic_scoring)

    gap = sub.add_parser("gap-sensitivity", help="Recompute exact-vs-graded stability-gap sensitivity from released outputs.")
    _add_artifact_arg(gap)
    gap.add_argument("--model", default=None, choices=RETAINED_TABLE2_MODELS)
    gap.set_defaults(func=command_gap_sensitivity)

    empty = sub.add_parser("both-empty-diagnostic", help="Count usable pairs where both reports have no released diagnostic labels.")
    _add_artifact_arg(empty)
    empty.add_argument("--model", default=None, choices=RETAINED_TABLE2_MODELS)
    empty.set_defaults(func=command_both_empty_diagnostic)

    radiologist = sub.add_parser("radiologist")
    _add_artifact_arg(radiologist)
    radiologist.set_defaults(func=command_radiologist)

    stage1 = sub.add_parser("stage1-confidence")
    _add_artifact_arg(stage1)
    stage1.set_defaults(func=command_stage1_confidence)

    mitigation = sub.add_parser("mitigation")
    _add_artifact_arg(mitigation)
    mitigation.set_defaults(func=command_mitigation)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
