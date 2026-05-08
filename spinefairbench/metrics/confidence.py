from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

HEDGING_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bpossibly\b",
    r"\bpossible\b",
    r"\blikely\b",
    r"\bsuggests?\b",
    r"\bcannot exclude\b",
    r"\bcannot rule out\b",
    r"\buncertain\b",
    r"\bdifferential includes\b",
    r"\bconsider\b",
    r"\bcorrelate clinically\b",
    r"\bprobable\b",
    r"\bsuspected\b",
    r"\bapparent\b",
    r"\bquestionable\b",
    r"\bequivocal\b",
]


def count_hedging_instances(text: str) -> int:
    text_lower = text.lower()
    count = 0
    for pattern in HEDGING_PATTERNS:
        count += len(re.findall(pattern, text_lower))
    return count


def compute_confidence_disparity(
    report_pairs: list[tuple[str, str]],
) -> dict[str, float]:
    if not report_pairs:
        return {
            "mean_hedging_ratio": 1.0,
            "mean_abs_difference": 0.0,
            "n_pairs": 0,
        }

    ratios = []
    abs_diffs = []

    for report_a, report_b in report_pairs:
        count_a = count_hedging_instances(report_a)
        count_b = count_hedging_instances(report_b)

        abs_diffs.append(abs(count_a - count_b))

        if count_a == 0 and count_b == 0:
            ratios.append(1.0)
        elif min(count_a, count_b) == 0:
            ratios.append(0.0)
        else:
            ratios.append(min(count_a, count_b) / max(count_a, count_b))

    return {
        "mean_hedging_ratio": sum(ratios) / len(ratios),
        "mean_abs_difference": sum(abs_diffs) / len(abs_diffs),
        "n_pairs": len(report_pairs),
    }
