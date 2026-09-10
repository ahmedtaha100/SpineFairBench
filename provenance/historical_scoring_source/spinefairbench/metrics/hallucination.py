from __future__ import annotations

import logging

from spinefairbench.metrics.diagnostic_label import extract_labels

logger = logging.getLogger(__name__)


def compute_false_positive_rate(
    vlm_report: str,
    ground_truth_labels: set[str],
) -> float:
    predicted = extract_labels(vlm_report)
    predicted.discard("No finding")

    if not predicted:
        return 0.0

    false_positives = predicted - ground_truth_labels
    return len(false_positives) / len(predicted)


def compute_hallucination_differential(
    report_pairs: list[tuple[str, str]],
    ground_truth_labels_list: list[set[str]],
) -> dict[str, float]:
    if not report_pairs:
        return {"fpr_group_a": 0.0, "fpr_group_b": 0.0, "disparity": 0.0, "n_pairs": 0}

    if len(report_pairs) != len(ground_truth_labels_list):
        logger.error(
            "Length mismatch: %d report pairs vs %d ground truth label sets",
            len(report_pairs),
            len(ground_truth_labels_list),
        )
        return {"fpr_group_a": 0.0, "fpr_group_b": 0.0, "disparity": 0.0, "n_pairs": 0}

    fpr_a_values = []
    fpr_b_values = []

    for (report_a, report_b), gt_labels in zip(report_pairs, ground_truth_labels_list, strict=True):
        fpr_a = compute_false_positive_rate(report_a, gt_labels)
        fpr_b = compute_false_positive_rate(report_b, gt_labels)
        fpr_a_values.append(fpr_a)
        fpr_b_values.append(fpr_b)

    mean_fpr_a = sum(fpr_a_values) / len(fpr_a_values)
    mean_fpr_b = sum(fpr_b_values) / len(fpr_b_values)

    return {
        "fpr_group_a": mean_fpr_a,
        "fpr_group_b": mean_fpr_b,
        "disparity": abs(mean_fpr_a - mean_fpr_b),
        "n_pairs": len(report_pairs),
    }
