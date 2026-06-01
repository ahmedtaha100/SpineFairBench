from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional analysis dependency
    np = None

try:
    from scipy import stats
except ModuleNotFoundError:  # pragma: no cover - optional analysis dependency
    stats = None

from spinefairbench.metrics.diagnostic_label import compute_jaccard, extract_labels
from spinefairbench.metrics.recommendation import classify_recommendations
from spinefairbench.metrics.refusal_detector import ResponseClass, classify_response

CONDITION_STEPS = {
    "condition_b": "mitigation_b_stage2",
    "condition_bprime": "mitigation_bprime_stage2",
    "condition_d": "mitigation_d_stage25",
}
ALL_CONDITIONS = ("condition_a", "condition_b", "condition_bprime", "condition_d")
COMPARISON_CONDITIONS = ("condition_b", "condition_bprime", "condition_d")
CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = CODE_ROOT.parent / "artifacts" / "artifacts"
DEFAULT_COUNTERFACTUAL_PANEL = (
    DEFAULT_ARTIFACT_ROOT
    / "Results"
    / "final_inputs"
    / "panels"
    / "full_pipeline_retained"
    / "evaluation_results.json"
)
DEFAULT_MITIGATION_PANEL = (
    DEFAULT_ARTIFACT_ROOT
    / "Results"
    / "final_inputs"
    / "panels"
    / "full_pipeline_mitigation_retained"
    / "evaluation_results.json"
)
DEFAULT_COMMON_CORE = (
    DEFAULT_ARTIFACT_ROOT
    / "freeze_runs"
    / "2026-04-09"
    / "evaluation_source_subset_core1000.json"
)
DEFAULT_ANALYSIS_DIR = DEFAULT_ARTIFACT_ROOT / "Results" / "analysis"


def _require_numeric_stack() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if stats is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError(
            "spinefairbench.analysis.mitigation requires optional analysis "
            f"dependencies not installed on the reviewer path: {', '.join(missing)}"
        )


@dataclass(frozen=True)
class ParsedSide:
    source_id: str
    pair_id: str
    recommendation: set[str]
    diagnostic_labels: set[str]
    taxonomy: str


@dataclass(frozen=True)
class PairMetric:
    source_id: str
    pair_id: str
    rec_change: float
    diag_consistency: float


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _row_model(row: dict[str, Any]) -> str:
    return str(row.get("model") or row.get("requested_model") or "")


def _row_source_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_source_id") or row.get("source_id") or "")


def _row_role(row: dict[str, Any]) -> str:
    role = str(row.get("image_role") or "")
    if role == "generated":
        return "counterfactual"
    return role


def _row_text(row: dict[str, Any]) -> str:
    response = row.get("response")
    if isinstance(response, str) and response.strip():
        return response
    raw_path = row.get("raw_output_path")
    if isinstance(raw_path, str) and raw_path:
        path = Path(raw_path)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def load_common_core_source_ids(path: Path = DEFAULT_COMMON_CORE) -> set[str]:
    payload = read_json(path)
    source_ids = payload.get("source_ids", []) if isinstance(payload, dict) else payload
    return {str(source_id) for source_id in source_ids}


def _clean_markdown(text: str) -> str:
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*[-:]\s*", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def extract_stage1_primary_diagnosis(text: str) -> str | None:
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


def _validation_is_clean(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    html_flag = bool(payload.get("html_garbage_detected", payload.get("html_garbage", False)))
    return (
        payload.get("status") == "valid"
        and not bool(payload.get("prompt_echo_detected", False))
        and not html_flag
    )


def stage1_diagnosis_parse_ok(stage1_path: Path, validation_path: Path | None = None) -> bool:
    if validation_path is not None and not _validation_is_clean(validation_path):
        return False
    if not stage1_path.exists():
        return False
    text = stage1_path.read_text(encoding="utf-8", errors="replace")
    return extract_stage1_primary_diagnosis(text) is not None


def compute_stage1_parsing_confidence(
    model_stage1_dir: Path,
    *,
    sample_size: int = 200,
) -> dict[str, Any]:
    txt_files = sorted(model_stage1_dir.glob("*.txt"))
    sample = txt_files[: max(sample_size, 200)]
    if not sample:
        return {
            "sample_size": 0,
            "stage1_diag_pass_count": 0,
            "stage1_diag_pass_rate": 0.0,
            "decision": "excluded_full",
        }
    pass_count = 0
    for txt_path in sample:
        validation_path = txt_path.with_suffix(txt_path.suffix + ".validation.json")
        # stage1 files are named *.txt and validation files are *.txt.validation.json in
        # Path.with_suffix form only if the suffix is appended manually.
        if not validation_path.exists():
            validation_path = txt_path.with_name(f"{txt_path.stem}.validation.json")
        if stage1_diagnosis_parse_ok(txt_path, validation_path):
            pass_count += 1
    rate = pass_count / len(sample)
    if rate < 0.85:
        decision = "excluded_full"
    elif rate < 0.95:
        decision = "excluded_diag_only"
    else:
        decision = "included"
    return {
        "sample_size": len(sample),
        "stage1_diag_pass_count": pass_count,
        "stage1_diag_pass_rate": float(rate),
        "decision": decision,
    }


def _taxonomy_from_text(row: dict[str, Any], text: str) -> str:
    if row.get("error") or row.get("failure_code"):
        return "api_failure"
    if not text.strip():
        return "parse_failure"
    response_class = classify_response(text)
    if response_class == ResponseClass.FULL_REFUSAL:
        return "full_refusal"
    if response_class == ResponseClass.PARTIAL_REFUSAL:
        return "partial_refusal"
    return "genuine"


def _stage1_diagnostic_labels(row: dict[str, Any]) -> set[str] | None:
    path_value = row.get("stage1_output_path")
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    diagnosis = extract_stage1_primary_diagnosis(
        path.read_text(encoding="utf-8", errors="replace")
    )
    if diagnosis is None:
        return None
    return extract_labels(diagnosis)


def _condition_text(row: dict[str, Any], condition: str) -> str:
    if condition != "condition_d":
        return _row_text(row)
    rewrite_path_value = row.get("stage2_5_rewrite_path")
    if isinstance(rewrite_path_value, str) and rewrite_path_value:
        rewrite_path = Path(rewrite_path_value)
        if rewrite_path.exists():
            try:
                payload = read_json(rewrite_path)
                rewritten = payload.get("rewritten_output")
                if isinstance(rewritten, str) and rewritten.strip():
                    return rewritten
            except Exception:
                pass
    return _row_text(row)


def _parse_side(row: dict[str, Any], condition: str) -> ParsedSide:
    text = _condition_text(row, condition)
    taxonomy = _taxonomy_from_text(row, text)
    if condition == "condition_a":
        diagnostic_labels = extract_labels(text)
    else:
        labels = _stage1_diagnostic_labels(row)
        if labels is None:
            taxonomy = "parse_failure"
            diagnostic_labels = set()
        else:
            diagnostic_labels = labels
    return ParsedSide(
        source_id=_row_source_id(row),
        pair_id=str(row.get("pair_id") or ""),
        recommendation=classify_recommendations(text),
        diagnostic_labels=diagnostic_labels,
        taxonomy=taxonomy,
    )


def _build_condition_metrics(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    common_core_source_ids: set[str],
) -> tuple[dict[str, dict[str, PairMetric]], dict[str, dict[str, int]], dict[str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    step = "counterfactual" if condition == "condition_a" else CONDITION_STEPS[condition]
    for row in rows:
        if str(row.get("experiment_step")) != step:
            continue
        source_id = _row_source_id(row)
        if source_id not in common_core_source_ids:
            continue
        role = _row_role(row)
        if role not in {"source", "counterfactual"}:
            continue
        model = _row_model(row)
        pair_id = str(row.get("pair_id") or "")
        if not model or not pair_id:
            continue
        grouped[(model, pair_id, role)] = row

    metrics: dict[str, dict[str, PairMetric]] = defaultdict(dict)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_pairs = {(model, pair_id) for model, pair_id, _ in grouped}
    for model, pair_id in sorted(model_pairs):
        src_row = grouped.get((model, pair_id, "source"))
        cf_row = grouped.get((model, pair_id, "counterfactual"))
        if src_row is None or cf_row is None:
            counts[model]["missing_side"] += 1
            continue
        src = _parse_side(src_row, condition)
        cf = _parse_side(cf_row, condition)
        for side in (src, cf):
            counts[model][side.taxonomy] += 1
        excluded = {"full_refusal", "api_failure", "parse_failure"}
        if src.taxonomy in excluded or cf.taxonomy in excluded:
            counts[model]["pair_excluded"] += 1
            continue
        counts[model]["pair_usable"] += 1
        metric = PairMetric(
            source_id=src.source_id,
            pair_id=pair_id,
            rec_change=float(src.recommendation != cf.recommendation),
            diag_consistency=compute_jaccard(src.diagnostic_labels, cf.diagnostic_labels),
        )
        metrics[model][pair_id] = metric
        if len(traces[model]) < 5:
            traces[model].append(
                {
                    "condition": condition,
                    "source_id": src.source_id,
                    "pair_id": pair_id,
                    "source_recommendations": sorted(src.recommendation),
                    "counterfactual_recommendations": sorted(cf.recommendation),
                    "source_diagnostic_labels": sorted(src.diagnostic_labels),
                    "counterfactual_diagnostic_labels": sorted(cf.diagnostic_labels),
                    "rec_change": metric.rec_change,
                    "diag_consistency": metric.diag_consistency,
                    "source_row_path": src_row.get("raw_output_path", ""),
                    "counterfactual_row_path": cf_row.get("raw_output_path", ""),
                }
            )
    return dict(metrics), {m: dict(c) for m, c in counts.items()}, dict(traces)


def _clustered_mean_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    n_iter: int,
    seed: int,
) -> tuple[float, float]:
    _require_numeric_stack()
    if len(values) == 0:
        return 0.0, 0.0
    unique = np.array(sorted(set(clusters.tolist())))
    sums = np.zeros(len(unique), dtype=float)
    counts = np.zeros(len(unique), dtype=float)
    index = {cluster: i for i, cluster in enumerate(unique)}
    for value, cluster in zip(values, clusters, strict=True):
        i = index[cluster]
        sums[i] += float(value)
        counts[i] += 1.0
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(n_iter, len(unique)))
    sample_sums = sums[sampled].sum(axis=1)
    sample_counts = counts[sampled].sum(axis=1)
    means = sample_sums / sample_counts
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    _require_numeric_stack()
    if len(a) == 0:
        return 1.0
    if np.allclose(a, b):
        return 1.0
    return float(stats.wilcoxon(a, b).pvalue)


def _cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    _require_numeric_stack()
    if len(a) < 2:
        return 0.0
    diff = a - b
    std = float(np.std(diff, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(diff) / std)


def _assert_finite(payload: Any) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            _assert_finite(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_finite(value)
    elif isinstance(payload, float) and (math.isnan(payload) or math.isinf(payload)):
        raise ValueError("non-finite statistic detected")


def _condition_summary(
    metrics: list[PairMetric],
    *,
    counts: dict[str, int],
    n_iter: int,
    seed: int,
) -> dict[str, Any]:
    _require_numeric_stack()
    rec = np.array([m.rec_change for m in metrics], dtype=float)
    diag = np.array([m.diag_consistency for m in metrics], dtype=float)
    clusters = np.array([m.source_id for m in metrics])
    return {
        "rec_change": float(np.mean(rec)) if len(rec) else 0.0,
        "rec_change_ci": list(_clustered_mean_ci(rec, clusters, n_iter=n_iter, seed=seed)),
        "diag_consistency": float(np.mean(diag)) if len(diag) else 0.0,
        "diag_consistency_ci": list(
            _clustered_mean_ci(diag, clusters, n_iter=n_iter, seed=seed)
        ),
        "n_pairs": int(len(metrics)),
        "full_refusals": int(counts.get("full_refusal", 0)),
        "partial_refusals": int(counts.get("partial_refusal", 0)),
    }


def _comparison_summary(
    condition_metrics: list[PairMetric],
    a_metrics: list[PairMetric],
    *,
    counts: dict[str, int],
    n_iter: int,
    seed: int,
    include_b_rule: bool,
) -> dict[str, Any]:
    _require_numeric_stack()
    base = _condition_summary(condition_metrics, counts=counts, n_iter=n_iter, seed=seed)
    rec = np.array([m.rec_change for m in condition_metrics], dtype=float)
    rec_a = np.array([m.rec_change for m in a_metrics], dtype=float)
    diag = np.array([m.diag_consistency for m in condition_metrics], dtype=float)
    diag_a = np.array([m.diag_consistency for m in a_metrics], dtype=float)
    clusters = np.array([m.source_id for m in condition_metrics])
    delta_rec = rec - rec_a
    delta_diag = diag - diag_a
    base.update(
        {
            "delta_rec": float(np.mean(delta_rec)) if len(delta_rec) else 0.0,
            "delta_rec_ci": list(
                _clustered_mean_ci(delta_rec, clusters, n_iter=n_iter, seed=seed + 2)
            ),
            "delta_diag": float(np.mean(delta_diag)) if len(delta_diag) else 0.0,
            "delta_diag_ci": list(
                _clustered_mean_ci(delta_diag, clusters, n_iter=n_iter, seed=seed + 3)
            ),
            "wilcoxon_p_rec": _paired_wilcoxon(rec, rec_a),
            "wilcoxon_p_diag": _paired_wilcoxon(diag, diag_a),
            "cohens_dz_rec": _cohens_dz(rec, rec_a),
            "cohens_dz_diag": _cohens_dz(diag, diag_a),
            "diag_preserved_within_guardrail": bool(
                base["diag_consistency"]
                >= (float(np.mean(diag_a)) if len(diag_a) else 0.0) - 0.05
            ),
        }
    )
    if include_b_rule:
        base["b_rule_pass"] = bool(
            base["rec_change"] < (float(np.mean(rec_a)) if len(rec_a) else 0.0)
            and base["diag_preserved_within_guardrail"]
        )
    return base


def _validate_summary_schema(summary: dict[str, Any]) -> None:
    required_top = {
        "scope",
        "counterfactual_source_panel",
        "mitigation_source_panel",
        "bootstrap_iterations",
        "smoke_test_iterations",
        "smoke_test_status",
        "included_models",
        "excluded_models",
        "parsing_confidence",
        "models",
        "multiple_testing",
        "rule_summary",
    }
    missing = required_top - set(summary)
    if missing:
        raise ValueError(f"summary missing keys: {sorted(missing)}")
    for model, model_summary in summary["models"].items():
        for condition in ALL_CONDITIONS:
            if condition not in model_summary:
                raise ValueError(f"{model} missing {condition}")
        if not isinstance(model_summary["condition_b"].get("b_rule_pass"), bool):
            raise ValueError(f"{model} missing boolean b_rule_pass")
        for condition in ALL_CONDITIONS:
            n_pairs = model_summary[condition]["n_pairs"]
            if not (1000 <= n_pairs <= 4000):
                raise ValueError(f"{model} {condition} n_pairs out of plausible bounds: {n_pairs}")
    _assert_finite(summary)


def build_mitigation_summary(
    *,
    counterfactual_panel: Path = DEFAULT_COUNTERFACTUAL_PANEL,
    mitigation_panel: Path = DEFAULT_MITIGATION_PANEL,
    common_core_path: Path = DEFAULT_COMMON_CORE,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    bootstrap_iterations: int = 10000,
    smoke_test_iterations: int = 100,
    smoke_test_status: str = "not_run",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    _require_numeric_stack()
    common_core = load_common_core_source_ids(common_core_path)
    counterfactual_rows = read_json(counterfactual_panel)
    mitigation_rows = read_json(mitigation_panel)
    manifest = read_json(mitigation_panel.parent / "panel_manifest.json")
    included_models = list(manifest["included_models"])
    excluded_models = list(manifest.get("excluded_models", []))

    parsing_confidence: dict[str, Any] = {}
    stage1_base = mitigation_panel.parent / "stage1_outputs"
    for model in included_models:
        parsing_confidence[model] = compute_stage1_parsing_confidence(stage1_base / model)

    write_json(analysis_dir / "mitigation_parsing_confidence.json", parsing_confidence)
    active_models = [
        model
        for model in included_models
        if parsing_confidence[model]["decision"] != "excluded_full"
    ]
    for model in included_models:
        if parsing_confidence[model]["decision"] == "excluded_full":
            excluded_models.append(
                {"model": model, "reason": "stage1 diagnosis parsing confidence below 85%"}
            )

    condition_maps: dict[str, dict[str, dict[str, PairMetric]]] = {}
    condition_counts: dict[str, dict[str, dict[str, int]]] = {}
    condition_traces: dict[str, dict[str, list[dict[str, Any]]]] = {}
    condition_maps["condition_a"], condition_counts["condition_a"], condition_traces["condition_a"] = (
        _build_condition_metrics(
            counterfactual_rows,
            condition="condition_a",
            common_core_source_ids=common_core,
        )
    )
    for condition in COMPARISON_CONDITIONS:
        condition_maps[condition], condition_counts[condition], condition_traces[condition] = (
            _build_condition_metrics(
                mitigation_rows,
                condition=condition,
                common_core_source_ids=common_core,
            )
        )

    summary_models: dict[str, Any] = {}
    tests: dict[str, Any] = {}
    pass_models: list[str] = []
    fail_models: list[str] = []
    trace: dict[str, list[dict[str, Any]]] = {}
    family_size = len(active_models) * len(COMPARISON_CONDITIONS) * 2
    alpha_adj = 0.05 / max(family_size, 1)

    for model in active_models:
        common_pair_ids = set(condition_maps["condition_a"].get(model, {}))
        for condition in COMPARISON_CONDITIONS:
            common_pair_ids &= set(condition_maps[condition].get(model, {}))
        common_pair_ids = set(sorted(common_pair_ids))
        if not common_pair_ids:
            raise ValueError(f"{model} has empty A-vs-mitigation alignment")
        ordered_pair_ids = sorted(common_pair_ids)
        per_condition_metrics = {
            condition: [condition_maps[condition][model][pair_id] for pair_id in ordered_pair_ids]
            for condition in ALL_CONDITIONS
        }
        model_summary: dict[str, Any] = {}
        model_summary["condition_a"] = _condition_summary(
            per_condition_metrics["condition_a"],
            counts=condition_counts["condition_a"].get(model, {}),
            n_iter=bootstrap_iterations,
            seed=101,
        )
        for idx, condition in enumerate(COMPARISON_CONDITIONS):
            model_summary[condition] = _comparison_summary(
                per_condition_metrics[condition],
                per_condition_metrics["condition_a"],
                counts=condition_counts[condition].get(model, {}),
                n_iter=bootstrap_iterations,
                seed=200 + idx * 10,
                include_b_rule=condition == "condition_b",
            )
            tests[f"{model}:{condition}:recommendation"] = {
                "p_value": model_summary[condition]["wilcoxon_p_rec"],
                "alpha_adj": alpha_adj,
                "significant": bool(model_summary[condition]["wilcoxon_p_rec"] < alpha_adj),
            }
            tests[f"{model}:{condition}:diagnostic"] = {
                "p_value": model_summary[condition]["wilcoxon_p_diag"],
                "alpha_adj": alpha_adj,
                "significant": bool(model_summary[condition]["wilcoxon_p_diag"] < alpha_adj),
            }
        if model_summary["condition_b"]["b_rule_pass"]:
            pass_models.append(model)
        else:
            fail_models.append(model)
        summary_models[model] = model_summary
        trace[model] = []
        for condition in ALL_CONDITIONS:
            trace[model].extend(condition_traces.get(condition, {}).get(model, [])[:3])

    summary = {
        "scope": "common_core_1000",
        "counterfactual_source_panel": str(counterfactual_panel),
        "mitigation_source_panel": str(mitigation_panel),
        "bootstrap_iterations": int(bootstrap_iterations),
        "smoke_test_iterations": int(smoke_test_iterations),
        "smoke_test_status": smoke_test_status,
        "included_models": active_models,
        "excluded_models": excluded_models,
        "parsing_confidence": parsing_confidence,
        "models": summary_models,
        "multiple_testing": {
            "method": "bonferroni",
            "family_size": int(family_size),
            "alpha_adj": float(alpha_adj),
            "tests": tests,
        },
        "rule_summary": {
            "models_with_b_rule_pass": pass_models,
            "models_with_b_rule_fail": fail_models,
        },
    }
    return summary, condition_counts, trace


def write_mitigation_tables(summary: dict[str, Any], analysis_dir: Path = DEFAULT_ANALYSIS_DIR) -> None:
    csv_path = analysis_dir / "mitigation_table.csv"
    rows: list[dict[str, Any]] = []
    for model, model_summary in summary["models"].items():
        for condition in ALL_CONDITIONS:
            payload = model_summary[condition]
            row = {
                "model": model,
                "condition": condition,
                "n_pairs": payload["n_pairs"],
                "rec_change": payload["rec_change"],
                "rec_change_ci_low": payload["rec_change_ci"][0],
                "rec_change_ci_high": payload["rec_change_ci"][1],
                "diag_consistency": payload["diag_consistency"],
                "diag_consistency_ci_low": payload["diag_consistency_ci"][0],
                "diag_consistency_ci_high": payload["diag_consistency_ci"][1],
                "full_refusals": payload["full_refusals"],
                "partial_refusals": payload["partial_refusals"],
            }
            for optional in (
                "delta_rec",
                "delta_diag",
                "wilcoxon_p_rec",
                "wilcoxon_p_diag",
                "cohens_dz_rec",
                "cohens_dz_diag",
                "diag_preserved_within_guardrail",
                "b_rule_pass",
            ):
                row[optional] = payload.get(optional, "")
            rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Condition & $n$ & Rec. change & Diag. consistency & $\\Delta$ rec. \\\\",
        "\\midrule",
    ]
    for row in rows:
        delta = row["delta_rec"]
        delta_text = "--" if delta == "" else f"{float(delta):.3f}"
        tex_lines.append(
            f"{row['model']} & {row['condition'].replace('condition_', '').upper()} & "
            f"{row['n_pairs']} & {float(row['rec_change']):.3f} & "
            f"{float(row['diag_consistency']):.3f} & {delta_text} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (analysis_dir / "mitigation_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def run_mitigation_analysis(
    *,
    counterfactual_panel: Path = DEFAULT_COUNTERFACTUAL_PANEL,
    mitigation_panel: Path = DEFAULT_MITIGATION_PANEL,
    common_core_path: Path = DEFAULT_COMMON_CORE,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    smoke_test_iterations: int = 100,
    bootstrap_iterations: int = 10000,
) -> dict[str, Any]:
    smoke_summary, _, _ = build_mitigation_summary(
        counterfactual_panel=counterfactual_panel,
        mitigation_panel=mitigation_panel,
        common_core_path=common_core_path,
        analysis_dir=analysis_dir,
        bootstrap_iterations=smoke_test_iterations,
        smoke_test_iterations=smoke_test_iterations,
        smoke_test_status="passed",
    )
    _validate_summary_schema(smoke_summary)
    write_json(analysis_dir / "mitigation_summary_smoke.json", smoke_summary)

    final_summary, condition_counts, trace = build_mitigation_summary(
        counterfactual_panel=counterfactual_panel,
        mitigation_panel=mitigation_panel,
        common_core_path=common_core_path,
        analysis_dir=analysis_dir,
        bootstrap_iterations=bootstrap_iterations,
        smoke_test_iterations=smoke_test_iterations,
        smoke_test_status="passed",
    )
    _validate_summary_schema(final_summary)
    write_json(analysis_dir / "mitigation_summary.json", final_summary)
    write_json(analysis_dir / "mitigation_condition_counts.json", condition_counts)
    write_json(analysis_dir / "mitigation_trace_samples.json", trace)
    write_mitigation_tables(final_summary, analysis_dir=analysis_dir)
    return final_summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run SpineFairBench mitigation analysis")
    parser.add_argument("--counterfactual-panel", type=Path, default=DEFAULT_COUNTERFACTUAL_PANEL)
    parser.add_argument("--mitigation-panel", type=Path, default=DEFAULT_MITIGATION_PANEL)
    parser.add_argument("--common-core", type=Path, default=DEFAULT_COMMON_CORE)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--smoke-test-iterations", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    summary = run_mitigation_analysis(
        counterfactual_panel=args.counterfactual_panel,
        mitigation_panel=args.mitigation_panel,
        common_core_path=args.common_core,
        analysis_dir=args.analysis_dir,
        smoke_test_iterations=args.smoke_test_iterations,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(
        json.dumps(
            {
                "summary": str(args.analysis_dir / "mitigation_summary.json"),
                "included_models": summary["included_models"],
                "rule_summary": summary["rule_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
