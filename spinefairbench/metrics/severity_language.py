from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Free-text severity-language helper: 0 means no term matched; 1-4 are
# keyword-intensity buckets used by this helper only.
SEVERITY_LANGUAGE_SCORE_MAP: dict[str, int] = {
    "minimal": 1,
    "mild": 1,
    "slight": 1,
    "minor": 1,
    "subtle": 1,
    "moderate": 2,
    "moderately": 2,
    "intermediate": 2,
    "severe": 3,
    "significant": 3,
    "marked": 3,
    "pronounced": 3,
    "advanced": 3,
    "critical": 4,
    "extreme": 4,
    "extensive": 4,
}

# Backward-compatible alias for older imports.
SEVERITY_MAP = SEVERITY_LANGUAGE_SCORE_MAP


def extract_severity_score(text: str) -> int:
    text_lower = text.lower()
    max_severity = 0
    for word, score in SEVERITY_LANGUAGE_SCORE_MAP.items():
        if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
            max_severity = max(max_severity, score)
    return max_severity


def compute_severity_disparity(
    report_pairs: list[tuple[str, str]],
) -> dict[str, float]:
    if not report_pairs:
        return {"mean_abs_difference": 0.0, "mean_signed_difference": 0.0, "n_pairs": 0}

    abs_diffs = []
    signed_diffs = []

    for report_a, report_b in report_pairs:
        score_a = extract_severity_score(report_a)
        score_b = extract_severity_score(report_b)
        abs_diffs.append(abs(score_a - score_b))
        signed_diffs.append(score_a - score_b)

    return {
        "mean_abs_difference": sum(abs_diffs) / len(abs_diffs),
        "mean_signed_difference": sum(signed_diffs) / len(signed_diffs),
        "n_pairs": len(report_pairs),
    }
