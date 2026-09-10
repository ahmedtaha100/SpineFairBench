from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spinefairbench.analysis.report import BenchmarkReport
from spinefairbench.config.schemas import load_config
from spinefairbench.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _git_output(args: list[str]) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return output or None


def _build_run_manifest(
    config: Any,
    config_path: Path,
    results_dir: Path,
    llm_judge_requested: bool,
    llm_judge_enabled: bool,
    ground_truth_summary: dict[str, Any],
) -> dict[str, Any]:
    llm_judge_status = "disabled"
    if llm_judge_enabled:
        llm_judge_status = "enabled"
    elif llm_judge_requested:
        llm_judge_status = "requested_no_api_key"

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git_output(["status", "--porcelain"])),
        "config_path": str(config_path.resolve()),
        "results_dir": str(results_dir.resolve()),
        "training_seed": config.training.seed,
        "vlm_models": config.evaluation.vlm_models,
        "analysis": {
            "bootstrap_samples": config.analysis.bootstrap_samples,
            "significance_level": config.analysis.significance_level,
            "correction_method": config.analysis.correction_method,
            "output_dir": str(config.analysis.output_dir),
        },
        "refusal_policy": {
            "exclude_full_refusals": True,
            "include_partial_refusals": True,
        },
        "ground_truth_labels": ground_truth_summary,
        "llm_judge": {
            "requested": llm_judge_requested,
            "enabled": llm_judge_enabled,
            "status": llm_judge_status,
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }

    try:
        import torch

        manifest["runtime"]["torch_version"] = torch.__version__
        manifest["runtime"]["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            manifest["runtime"]["cuda_device_name"] = torch.cuda.get_device_name(
                0
            )
    except Exception:
        pass

    return manifest


def _summarize_ground_truth_labels(
    pairs_data: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not pairs_data:
        return {
            "pairs_total": 0,
            "pairs_with_labels_field": 0,
            "pairs_with_non_empty_labels": 0,
            "usable_for_hallucination": False,
        }

    pairs_with_labels_field = 0
    pairs_with_non_empty_labels = 0
    for pair_entry in pairs_data:
        if "pathology_labels" not in pair_entry:
            continue
        pairs_with_labels_field += 1
        pathology_indices = pair_entry.get("pathology_labels", [])
        if isinstance(pathology_indices, list) and len(pathology_indices) > 0:
            pairs_with_non_empty_labels += 1

    return {
        "pairs_total": len(pairs_data),
        "pairs_with_labels_field": pairs_with_labels_field,
        "pairs_with_non_empty_labels": pairs_with_non_empty_labels,
        "usable_for_hallucination": pairs_with_non_empty_labels > 0,
    }


_FairnessResult = tuple[
    dict[str, Any],
    dict[str, list[tuple[str, str]]],
    dict[str, list[set[str]]],
    dict[str, list[tuple[str, str]]],
]


def _compute_fairness_metrics(
    eval_results: list[dict[str, Any]],
    pairs_data: list[dict[str, Any]] | None = None,
    llm_judge_api_key: str | None = None,
) -> _FairnessResult:
    from spinefairbench.data.annotations import INDEX_TO_CATEGORY
    from spinefairbench.metrics.aggregator import aggregate_metrics
    from spinefairbench.metrics.refusal_detector import ResponseClass, classify_response

    gt_labels_by_source_id: dict[str, set[str]] = {}
    pairs_with_labels_field = 0
    pairs_with_non_empty_labels = 0
    if pairs_data:
        for pair_entry in pairs_data:
            source_id = pair_entry.get("source_id", "")
            if not source_id or "pathology_labels" not in pair_entry:
                continue
            pairs_with_labels_field += 1
            pathology_indices = pair_entry.get("pathology_labels", [])
            if isinstance(pathology_indices, list) and len(pathology_indices) > 0:
                pairs_with_non_empty_labels += 1
            labels = {
                INDEX_TO_CATEGORY[idx]
                for idx in pathology_indices
                if idx in INDEX_TO_CATEGORY
            }
            labels.discard("No finding")
            gt_labels_by_source_id[source_id] = labels
    if pairs_with_labels_field > 0 and pairs_with_non_empty_labels == 0:
        logger.warning(
            "All pathology_labels are empty across %d pairs; "
            "hallucination and per-pathology metrics will be disabled.",
            pairs_with_labels_field,
        )
        gt_labels_by_source_id = {}

    source_responses: dict[str, dict[str, str]] = {}
    generated_responses: dict[str, dict[str, str]] = {}

    for r in eval_results:
        model = r.get("model", "unknown")
        pair_id = r.get("pair_source_id", "")
        img_path = r.get("image_path", "")
        response = r.get("response", "") or ""
        role = r.get("image_role", "")

        if not pair_id or r.get("error"):
            continue

        if classify_response(response) == ResponseClass.FULL_REFUSAL:
            continue

        if role == "source" or (not role and "_source" in img_path):
            source_responses.setdefault(model, {})[pair_id] = response
        elif role == "generated" or (not role and "_source" not in img_path):
            key = f"{pair_id}::{img_path}"
            generated_responses.setdefault(model, {})[key] = response

    all_metrics: dict[str, Any] = {}
    model_report_pairs: dict[str, list[tuple[str, str]]] = {}
    model_gt_labels: dict[str, list[set[str]]] = {}
    model_hallucination_pairs: dict[str, list[tuple[str, str]]] = {}

    for model_name in source_responses:
        report_pairs: list[tuple[str, str]] = []
        model_sources = source_responses[model_name]
        model_generated = generated_responses.get(model_name, {})

        ground_truth_labels: list[set[str]] = []
        hallucination_pairs: list[tuple[str, str]] = []
        for gen_key, gen_response in model_generated.items():
            src_id = gen_key.split("::")[0]
            if src_id in model_sources:
                pair = (model_sources[src_id], gen_response)
                report_pairs.append(pair)
                if src_id in gt_labels_by_source_id:
                    ground_truth_labels.append(gt_labels_by_source_id[src_id])
                    hallucination_pairs.append(pair)

        if report_pairs:
            llm_judge_results: list[dict[str, Any]] | None = None
            if llm_judge_api_key:
                from spinefairbench.metrics.llm_judge import batch_judge

                logger.info(
                    "Running LLM judge on %d pairs for %s",
                    len(report_pairs),
                    model_name,
                )
                llm_judge_results = _run_async(
                    batch_judge(report_pairs, llm_judge_api_key)
                )
                logger.info(
                    "LLM judge complete for %s: %d results",
                    model_name,
                    len(llm_judge_results),
                )

            metrics = aggregate_metrics(
                report_pairs, llm_judge_results=llm_judge_results
            )
            if hallucination_pairs:
                from spinefairbench.metrics.hallucination import compute_hallucination_differential

                metrics["hallucination"] = compute_hallucination_differential(
                    hallucination_pairs, ground_truth_labels
                )
            all_metrics[model_name] = metrics
            model_report_pairs[model_name] = report_pairs
            model_gt_labels[model_name] = ground_truth_labels
            model_hallucination_pairs[model_name] = hallucination_pairs

    return all_metrics, model_report_pairs, model_gt_labels, model_hallucination_pairs


def _extract_per_pair_scores(
    report_pairs: list[tuple[str, str]],
    hallucination_pairs: list[tuple[str, str]] | None = None,
    ground_truth_labels: list[set[str]] | None = None,
) -> dict[str, Any]:
    from spinefairbench.metrics.confidence import count_hedging_instances
    from spinefairbench.metrics.diagnostic_label import compute_jaccard, extract_labels
    from spinefairbench.metrics.hallucination import compute_false_positive_rate
    from spinefairbench.metrics.recommendation import classify_recommendations
    from spinefairbench.metrics.severity_language import extract_severity_score

    jaccard_scores: list[float] = []
    severity_a_scores: list[float] = []
    severity_b_scores: list[float] = []
    rec_agreements: list[float] = []
    hedging_a_counts: list[float] = []
    hedging_b_counts: list[float] = []

    for report_a, report_b in report_pairs:
        labels_a = extract_labels(report_a)
        labels_b = extract_labels(report_b)
        jaccard_scores.append(compute_jaccard(labels_a, labels_b))

        severity_a_scores.append(float(extract_severity_score(report_a)))
        severity_b_scores.append(float(extract_severity_score(report_b)))

        recs_a = classify_recommendations(report_a)
        recs_b = classify_recommendations(report_b)
        rec_agreements.append(1.0 if recs_a == recs_b else 0.0)

        hedging_a_counts.append(float(count_hedging_instances(report_a)))
        hedging_b_counts.append(float(count_hedging_instances(report_b)))

    result: dict[str, Any] = {
        "diagnostic_label": {"jaccard_scores": jaccard_scores},
        "severity_language": {
            "scores_a": severity_a_scores,
            "scores_b": severity_b_scores,
        },
        "recommendation": {"agreements": rec_agreements},
        "confidence": {
            "counts_a": hedging_a_counts,
            "counts_b": hedging_b_counts,
        },
    }

    if hallucination_pairs and ground_truth_labels:
        fpr_a_values: list[float] = []
        fpr_b_values: list[float] = []
        for (report_a, report_b), gt in zip(
            hallucination_pairs,
            ground_truth_labels,
            strict=True,
        ):
            fpr_a_values.append(compute_false_positive_rate(report_a, gt))
            fpr_b_values.append(compute_false_positive_rate(report_b, gt))
        result["hallucination"] = {
            "fpr_a": fpr_a_values,
            "fpr_b": fpr_b_values,
        }

    return result


def _apply_correction(
    p_values: list[float],
    method: str,
    alpha: float,
) -> list[dict[str, Any]]:
    from spinefairbench.analysis.statistics import (
        bonferroni_correction,
        fdr_bh_correction,
        holm_correction,
    )

    if method == "holm":
        return holm_correction(p_values, alpha=alpha)
    if method == "fdr_bh":
        return fdr_bh_correction(p_values, alpha=alpha)
    return bonferroni_correction(p_values, alpha=alpha)


def _compute_statistics(
    per_pair_scores: dict[str, Any],
    bootstrap_samples: int = 1000,
    significance_level: float = 0.05,
    correction_method: str = "bonferroni",
) -> dict[str, Any]:
    import numpy as np

    from spinefairbench.analysis.statistics import (
        bootstrap_ci,
        paired_cohens_d,
        paired_wilcoxon_test,
    )

    stats_result: dict[str, Any] = {}
    all_p_values: list[float] = []
    p_value_labels: list[str] = []

    jaccard = np.array(per_pair_scores["diagnostic_label"]["jaccard_scores"])
    if len(jaccard) > 0:
        ci = bootstrap_ci(
            jaccard, n_resamples=bootstrap_samples, alpha=significance_level
        )
        ones = np.ones_like(jaccard)
        wilcoxon = paired_wilcoxon_test(jaccard, ones)
        effect = paired_cohens_d(jaccard, ones)
        all_p_values.append(wilcoxon["p_value"])
        p_value_labels.append("diagnostic_label")
        stats_result["diagnostic_label"] = {
            "mean": float(np.mean(jaccard)),
            "bootstrap_ci": {"lower": ci[0], "upper": ci[1]},
            "wilcoxon": wilcoxon,
            "cohens_d_z": effect,
            "n_pairs": len(jaccard),
        }

    sev_a = np.array(
        per_pair_scores["severity_language"]["scores_a"], dtype=float
    )
    sev_b = np.array(
        per_pair_scores["severity_language"]["scores_b"], dtype=float
    )
    if len(sev_a) > 0:
        sev_diffs = np.abs(sev_a - sev_b)
        ci = bootstrap_ci(
            sev_diffs, n_resamples=bootstrap_samples, alpha=significance_level
        )
        wilcoxon = paired_wilcoxon_test(sev_a, sev_b)
        effect = paired_cohens_d(sev_a, sev_b)
        all_p_values.append(wilcoxon["p_value"])
        p_value_labels.append("severity_language")
        stats_result["severity_language"] = {
            "mean_abs_difference": float(np.mean(sev_diffs)),
            "bootstrap_ci": {"lower": ci[0], "upper": ci[1]},
            "wilcoxon": wilcoxon,
            "cohens_d_z": effect,
            "n_pairs": len(sev_a),
        }

    rec = np.array(per_pair_scores["recommendation"]["agreements"])
    if len(rec) > 0:
        ci = bootstrap_ci(
            rec, n_resamples=bootstrap_samples, alpha=significance_level
        )
        ones = np.ones_like(rec)
        wilcoxon = paired_wilcoxon_test(rec, ones)
        effect = paired_cohens_d(rec, ones)
        all_p_values.append(wilcoxon["p_value"])
        p_value_labels.append("recommendation")
        stats_result["recommendation"] = {
            "agreement_rate": float(np.mean(rec)),
            "bootstrap_ci": {"lower": ci[0], "upper": ci[1]},
            "wilcoxon": wilcoxon,
            "cohens_d_z": effect,
            "n_pairs": len(rec),
        }

    hed_a = np.array(
        per_pair_scores["confidence"]["counts_a"], dtype=float
    )
    hed_b = np.array(
        per_pair_scores["confidence"]["counts_b"], dtype=float
    )
    if len(hed_a) > 0:
        hed_diffs = np.abs(hed_a - hed_b)
        ci = bootstrap_ci(
            hed_diffs, n_resamples=bootstrap_samples, alpha=significance_level
        )
        wilcoxon = paired_wilcoxon_test(hed_a, hed_b)
        effect = paired_cohens_d(hed_a, hed_b)
        all_p_values.append(wilcoxon["p_value"])
        p_value_labels.append("confidence")
        stats_result["confidence"] = {
            "mean_abs_difference": float(np.mean(hed_diffs)),
            "bootstrap_ci": {"lower": ci[0], "upper": ci[1]},
            "wilcoxon": wilcoxon,
            "cohens_d_z": effect,
            "n_pairs": len(hed_a),
        }

    if "hallucination" in per_pair_scores:
        fpr_a = np.array(per_pair_scores["hallucination"]["fpr_a"])
        fpr_b = np.array(per_pair_scores["hallucination"]["fpr_b"])
        if len(fpr_a) > 0:
            fpr_diffs = np.abs(fpr_a - fpr_b)
            ci = bootstrap_ci(
                fpr_diffs,
                n_resamples=bootstrap_samples,
                alpha=significance_level,
            )
            wilcoxon = paired_wilcoxon_test(fpr_a, fpr_b)
            effect = paired_cohens_d(fpr_a, fpr_b)
            all_p_values.append(wilcoxon["p_value"])
            p_value_labels.append("hallucination")
            stats_result["hallucination"] = {
                "mean_disparity": float(np.mean(fpr_diffs)),
                "bootstrap_ci": {"lower": ci[0], "upper": ci[1]},
                "wilcoxon": wilcoxon,
                "cohens_d_z": effect,
                "n_pairs": len(fpr_a),
            }

    if all_p_values:
        corrected = _apply_correction(
            all_p_values, correction_method, significance_level
        )
        stats_result["multiple_testing_correction"] = {
            "method": correction_method,
            "tests": dict(zip(p_value_labels, corrected, strict=True)),
        }

    return stats_result


def _run_statistical_analysis(
    model_report_pairs: dict[str, list[tuple[str, str]]],
    model_gt_labels: dict[str, list[set[str]]],
    model_hallucination_pairs: dict[str, list[tuple[str, str]]],
    bootstrap_samples: int = 1000,
    significance_level: float = 0.05,
    correction_method: str = "bonferroni",
) -> dict[str, Any]:
    all_statistics: dict[str, Any] = {}
    for model_name, pairs in model_report_pairs.items():
        gt_labels = model_gt_labels.get(model_name)
        h_pairs = model_hallucination_pairs.get(model_name)
        per_pair = _extract_per_pair_scores(
            pairs,
            hallucination_pairs=h_pairs if h_pairs else None,
            ground_truth_labels=gt_labels if gt_labels else None,
        )
        model_stats = _compute_statistics(
            per_pair,
            bootstrap_samples=bootstrap_samples,
            significance_level=significance_level,
            correction_method=correction_method,
        )
        all_statistics[model_name] = model_stats
        logger.info(
            "Computed statistics for %s: %d metrics with CIs",
            model_name,
            len(model_stats),
        )
    return all_statistics


def _compute_per_pathology_metrics(
    hallucination_pairs: list[tuple[str, str]],
    ground_truth_labels: list[set[str]],
) -> dict[str, dict[str, float]]:
    from spinefairbench.metrics.confidence import count_hedging_instances
    from spinefairbench.metrics.diagnostic_label import compute_jaccard, extract_labels
    from spinefairbench.metrics.recommendation import classify_recommendations
    from spinefairbench.metrics.severity_language import extract_severity_score

    pathology_pairs: dict[str, list[tuple[str, str]]] = {}
    for (report_a, report_b), labels in zip(
        hallucination_pairs, ground_truth_labels, strict=True
    ):
        for pathology in labels:
            pathology_pairs.setdefault(pathology, []).append(
                (report_a, report_b)
            )

    results: dict[str, dict[str, float]] = {}
    for pathology, pairs in sorted(pathology_pairs.items()):
        if not pairs:
            continue

        jaccard_total = 0.0
        severity_total = 0.0
        rec_total = 0.0
        confidence_total = 0.0
        n = len(pairs)

        for report_a, report_b in pairs:
            labels_a = extract_labels(report_a)
            labels_b = extract_labels(report_b)
            jaccard_total += compute_jaccard(labels_a, labels_b)

            severity_total += abs(
                float(extract_severity_score(report_a))
                - float(extract_severity_score(report_b))
            )

            recs_a = classify_recommendations(report_a)
            recs_b = classify_recommendations(report_b)
            rec_total += 1.0 if recs_a == recs_b else 0.0

            confidence_total += abs(
                float(count_hedging_instances(report_a))
                - float(count_hedging_instances(report_b))
            )

        results[pathology] = {
            "n_pairs": float(n),
            "jaccard": jaccard_total / n,
            "severity_disp": severity_total / n,
            "rec_agreement": rec_total / n,
            "confidence_disp": confidence_total / n,
        }

    return results


def run_analysis(args: argparse.Namespace) -> None:
    setup_logging()

    config_path = Path(args.config)
    config = load_config(config_path)
    ac = config.analysis

    results_dir = (
        Path(args.results_dir)
        if hasattr(args, "results_dir") and args.results_dir
        else Path(config.evaluation.output_dir)
    )

    output_dir = Path(ac.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_fmt = ac.figure_format
    fig_dpi = ac.figure_dpi

    report = BenchmarkReport(output_dir)
    report.add_config(config.model_dump())

    eval_results_path = results_dir / "evaluation_results.json"
    if eval_results_path.exists():
        with open(eval_results_path) as f:
            eval_results = json.load(f)
        logger.info("Loaded %d evaluation results", len(eval_results))
    else:
        logger.warning("No evaluation results found at %s", eval_results_path)
        eval_results = []

    pairs_data: list[dict[str, Any]] | None = None
    pairs_candidates = [
        results_dir / "pairs.json",
        results_dir.parent / "counterfactuals" / "pairs.json",
        Path(config.generation.output_dir) / "pairs.json",
    ]
    pairs_file = next((p for p in pairs_candidates if p.exists()), None)
    if pairs_file is not None:
        with open(pairs_file) as f:
            pairs_data = json.load(f)
        logger.info("Loaded %d pairs for ground truth labels", len(pairs_data))
    ground_truth_summary = _summarize_ground_truth_labels(pairs_data)
    if not ground_truth_summary["usable_for_hallucination"]:
        logger.warning(
            "Ground truth labels are not usable for hallucination metrics: %s",
            ground_truth_summary,
        )

    llm_judge_api_key: str | None = None
    llm_judge_requested = bool(hasattr(args, "llm_judge") and args.llm_judge)
    if llm_judge_requested:
        from spinefairbench.config.settings import get_settings

        settings = get_settings()
        llm_judge_api_key = settings.anthropic_api_key
        if not llm_judge_api_key:
            logger.warning(
                "LLM judge requested but no Anthropic API key configured"
            )
    llm_judge_enabled = bool(llm_judge_api_key)

    run_manifest = _build_run_manifest(
        config,
        config_path,
        results_dir,
        llm_judge_requested=llm_judge_requested,
        llm_judge_enabled=llm_judge_enabled,
        ground_truth_summary=ground_truth_summary,
    )
    report.add_manifest(run_manifest)
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2, default=str)
    logger.info("Saved run manifest to %s", manifest_path)

    if eval_results:
        from spinefairbench.analysis.tables import (
            data_quality_table,
            main_results_table,
            save_table,
        )
        from spinefairbench.metrics.refusal_detector import compute_data_quality

        data_quality = compute_data_quality(eval_results)
        report.add_data_quality(data_quality)
        quality_latex = data_quality_table(data_quality)
        quality_table_path = output_dir / "data_quality.tex"
        save_table(quality_latex, quality_table_path)
        report.add_table_paths([str(quality_table_path)])
        for model_name, counts in data_quality.items():
            logger.info(
                "Data quality for %s: responses=%d total, %d genuine, "
                "%d partial refusals, %d full refusals, %d api errors, %d usable; "
                "pairs=%d total, %d usable, %d full refusals, %d partial refusals, "
                "%d api errors, %d missing source",
                model_name,
                counts["total"],
                counts["genuine"],
                counts["partial_refusals"],
                counts["full_refusals"],
                counts["api_errors"],
                counts["usable"],
                counts["pair_total"],
                counts["pair_usable"],
                counts["pair_full_refusals"],
                counts["pair_partial_refusals"],
                counts["pair_api_errors"],
                counts["pair_missing_source"],
            )

        fairness_metrics, model_report_pairs, model_gt_labels, model_h_pairs = (
            _compute_fairness_metrics(
                eval_results, pairs_data, llm_judge_api_key
            )
        )
        report.add_metrics(fairness_metrics)
        logger.info(
            "Computed fairness metrics for %d VLMs", len(fairness_metrics)
        )
        model_pair_counts = {
            model: int(metrics.get("n_pairs", 0))
            for model, metrics in fairness_metrics.items()
        }
        if model_pair_counts:
            min_pairs = min(model_pair_counts.values())
            max_pairs = max(model_pair_counts.values())
            if min_pairs != max_pairs:
                logger.warning(
                    "Usable pair counts differ across models: %s",
                    model_pair_counts,
                )
            for model_name, pair_count in model_pair_counts.items():
                dq = data_quality.get(model_name, {})
                dq_pairs = int(dq.get("pair_usable", pair_count))
                if dq_pairs != pair_count:
                    logger.warning(
                        "Pair-count mismatch for %s: fairness=%d data_quality=%d",
                        model_name,
                        pair_count,
                        dq_pairs,
                    )

        statistics = _run_statistical_analysis(
            model_report_pairs,
            model_gt_labels,
            model_h_pairs,
            bootstrap_samples=ac.bootstrap_samples,
            significance_level=ac.significance_level,
            correction_method=ac.correction_method,
        )
        report.add_statistics(statistics)
        logger.info(
            "Computed statistical analysis for %d VLMs", len(statistics)
        )

        p_values: dict[str, dict[str, float]] = {}
        for vlm_name, vlm_stats in statistics.items():
            vlm_pvals: dict[str, float] = {}
            correction_data = vlm_stats.get(
                "multiple_testing_correction", {}
            )
            tests = correction_data.get("tests", {})
            for metric_name, test_result in tests.items():
                vlm_pvals[metric_name] = test_result.get("adjusted_p", 1.0)
            for metric_name in [
                "diagnostic_label",
                "severity_language",
                "recommendation",
                "confidence",
                "hallucination",
            ]:
                if (
                    metric_name in vlm_stats
                    and metric_name not in vlm_pvals
                ):
                    wilcoxon_data = vlm_stats[metric_name].get("wilcoxon", {})
                    vlm_pvals[metric_name] = wilcoxon_data.get(
                        "p_value", 1.0
                    )
            p_values[vlm_name] = vlm_pvals

        table_latex = main_results_table(
            fairness_metrics, p_values=p_values
        )
        table_path = output_dir / "main_results.tex"
        save_table(table_latex, table_path)
        report.add_table_paths([str(table_path)])

        categories = [
            "diagnostic_label",
            "severity_language",
            "recommendation",
            "confidence",
            "hallucination",
        ]
        disparity_keys: dict[str, tuple[str, bool]] = {
            "diagnostic_label": ("mean_jaccard", True),
            "severity_language": ("mean_abs_difference", False),
            "recommendation": ("agreement_rate", True),
            "confidence": ("mean_abs_difference", False),
            "hallucination": ("disparity", False),
        }

        from spinefairbench.analysis.visualization import radar_chart as _radar_chart

        try:
            fig_path = output_dir / f"radar_chart.{fig_fmt}"
            radar_data: dict[str, list[float]] = {}
            for vlm, m in fairness_metrics.items():
                values: list[float] = []
                for c in categories:
                    cat_data = m.get(c, {})
                    if not isinstance(cat_data, dict):
                        values.append(0.0)
                        continue
                    key, invert = disparity_keys[c]
                    raw = cat_data.get(key, 0.0)
                    val = float(raw) if raw is not None else 0.0
                    values.append(1.0 - val if invert else val)
                radar_data[vlm] = values
            _radar_chart(radar_data, categories, fig_path, dpi=fig_dpi)
            report.add_figure_paths([str(fig_path)])
        except Exception as e:
            logger.warning("Failed to create radar chart: %s", e)

        import numpy as np

        try:
            from spinefairbench.analysis.visualization import (
                forest_plot as _forest_plot,
            )
            all_effects: list[float] = []
            all_ci_lower: list[float] = []
            all_ci_upper: list[float] = []
            all_labels: list[str] = []
            _mean_keys = (
                "mean",
                "mean_abs_difference",
                "agreement_rate",
                "mean_disparity",
            )
            for vlm_name in statistics:
                vlm_stats = statistics[vlm_name]
                for metric in categories:
                    if metric not in vlm_stats:
                        continue
                    ms = vlm_stats[metric]
                    _, should_invert = disparity_keys[metric]
                    mean_key = next(
                        (k for k in _mean_keys if k in ms), None
                    )
                    if mean_key is None:
                        continue
                    ci = ms["bootstrap_ci"]
                    if should_invert:
                        all_effects.append(1.0 - ms[mean_key])
                        all_ci_lower.append(1.0 - ci["upper"])
                        all_ci_upper.append(1.0 - ci["lower"])
                    else:
                        all_effects.append(ms[mean_key])
                        all_ci_lower.append(ci["lower"])
                        all_ci_upper.append(ci["upper"])
                    all_labels.append(f"{vlm_name} | {metric}")

            if all_labels:
                fp = output_dir / f"forest_plot.{fig_fmt}"
                _forest_plot(
                    all_effects,
                    all_ci_lower,
                    all_ci_upper,
                    all_labels,
                    fp,
                    title="Disparity Magnitudes with 95% Bootstrap CIs",
                    xlabel="Disparity Magnitude",
                    dpi=fig_dpi,
                )
                report.add_figure_paths([str(fp)])
        except Exception as e:
            logger.warning("Failed to create forest plot: %s", e)

        try:
            from spinefairbench.analysis.visualization import (
                bias_heatmap as _bias_heatmap,
            )

            heatmap_data: dict[str, dict[str, float]] = {}
            for vlm, m in fairness_metrics.items():
                row: dict[str, float] = {}
                for c in categories:
                    cat_data = m.get(c, {})
                    if not isinstance(cat_data, dict):
                        row[c] = 0.0
                        continue
                    key, invert = disparity_keys[c]
                    raw = cat_data.get(key, 0.0)
                    val = float(raw) if raw is not None else 0.0
                    row[c] = 1.0 - val if invert else val
                heatmap_data[vlm] = row

            if heatmap_data:
                hmp = output_dir / f"bias_heatmap.{fig_fmt}"
                _bias_heatmap(heatmap_data, hmp, dpi=fig_dpi)
                report.add_figure_paths([str(hmp)])
        except Exception as e:
            logger.warning("Failed to create bias heatmap: %s", e)

        vlm_per_pair: dict[str, dict[str, Any]] = {}
        for vlm_name, pairs in model_report_pairs.items():
            gt = model_gt_labels.get(vlm_name)
            hp_list = model_h_pairs.get(vlm_name)
            vlm_per_pair[vlm_name] = _extract_per_pair_scores(
                pairs,
                hallucination_pairs=hp_list if hp_list else None,
                ground_truth_labels=gt if gt else None,
            )

        for metric in categories:
            try:
                from spinefairbench.analysis.visualization import (
                    metric_boxplots as _metric_boxplots,
                )

                boxplot_data: dict[str, list[float]] = {}
                for vlm_name, pp in vlm_per_pair.items():
                    if metric == "diagnostic_label":
                        boxplot_data[vlm_name] = pp["diagnostic_label"][
                            "jaccard_scores"
                        ]
                    elif metric == "severity_language":
                        sa = np.array(pp["severity_language"]["scores_a"])
                        sb = np.array(pp["severity_language"]["scores_b"])
                        boxplot_data[vlm_name] = np.abs(sa - sb).tolist()
                    elif metric == "recommendation":
                        boxplot_data[vlm_name] = pp["recommendation"][
                            "agreements"
                        ]
                    elif metric == "confidence":
                        ca = np.array(pp["confidence"]["counts_a"])
                        cb = np.array(pp["confidence"]["counts_b"])
                        boxplot_data[vlm_name] = np.abs(ca - cb).tolist()
                    elif (
                        metric == "hallucination"
                        and "hallucination" in pp
                    ):
                        fa = np.array(pp["hallucination"]["fpr_a"])
                        fb = np.array(pp["hallucination"]["fpr_b"])
                        boxplot_data[vlm_name] = np.abs(fa - fb).tolist()

                if boxplot_data and any(
                    len(v) > 0 for v in boxplot_data.values()
                ):
                    bp = output_dir / f"boxplot_{metric}.{fig_fmt}"
                    _metric_boxplots(
                        boxplot_data,
                        bp,
                        title=f"{metric.replace('_', ' ').title()} Distribution",
                        dpi=fig_dpi,
                    )
                    report.add_figure_paths([str(bp)])
            except Exception as e:
                logger.warning(
                    "Failed to create boxplot for %s: %s", metric, e
                )

        for vlm_name, pp in vlm_per_pair.items():
            try:
                from spinefairbench.analysis.visualization import (
                    paired_comparison_plot as _paired_comparison_plot,
                )

                sev_a = pp["severity_language"]["scores_a"]
                sev_b = pp["severity_language"]["scores_b"]
                if sev_a and sev_b:
                    safe = vlm_name.replace("/", "_").replace(" ", "_")
                    pcp = output_dir / f"paired_severity_{safe}.{fig_fmt}"
                    _paired_comparison_plot(
                        sev_a,
                        sev_b,
                        pcp,
                        label_a="Source",
                        label_b="Counterfactual",
                        title=f"Severity: {vlm_name}",
                        dpi=fig_dpi,
                    )
                    report.add_figure_paths([str(pcp)])
            except Exception as e:
                logger.warning(
                    "Failed to create paired severity plot for %s: %s",
                    vlm_name,
                    e,
                )

        try:
            from spinefairbench.analysis.tables import (
                per_pathology_table as _per_pathology_table,
            )

            for vlm_name in model_h_pairs:
                h_pairs_list = model_h_pairs[vlm_name]
                gt_list = model_gt_labels.get(vlm_name, [])
                if not h_pairs_list or not gt_list:
                    continue

                patho_metrics = _compute_per_pathology_metrics(
                    h_pairs_list, gt_list
                )
                if patho_metrics:
                    patho_latex = _per_pathology_table(patho_metrics)
                    safe = vlm_name.replace("/", "_").replace(" ", "_")
                    patho_path = (
                        output_dir / f"per_pathology_{safe}.tex"
                    )
                    save_table(patho_latex, patho_path)
                    report.add_table_paths([str(patho_path)])
                    logger.info(
                        "Generated per-pathology table for %s: %d categories",
                        vlm_name,
                        len(patho_metrics),
                    )

        except Exception as e:
            logger.warning(
                "Failed to create per-pathology tables: %s", e
            )

    report_path = report.save()
    logger.info("Analysis complete. Report: %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        default=False,
    )
    run_analysis(parser.parse_args())
