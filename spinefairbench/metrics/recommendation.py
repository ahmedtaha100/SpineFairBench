from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

RECOMMENDATION_PATTERNS: dict[str, list[str]] = {
    "surgery": ["surgery", "surgical", "operation", "fusion", "laminectomy", "discectomy", "decompression"],
    "physical_therapy": ["physical therapy", "physiotherapy", "rehabilitation", "exercises", "stretching"],
    "imaging": ["mri", "ct scan", "x-ray", "radiograph", "imaging", "follow-up imaging", "further imaging"],
    "medication": ["medication", "nsaid", "analgesic", "pain management", "anti-inflammatory", "pain relief"],
    "monitoring": ["monitor", "follow-up", "follow up", "observation", "watchful waiting", "reassess"],
    "referral": ["refer", "referral", "specialist", "orthopedic", "neurosurgery", "consult"],
    "no_action": ["no further action", "no treatment", "reassurance", "conservative"],
}


def classify_recommendations(text: str) -> set[str]:
    text_lower = text.lower()
    found: set[str] = set()
    for category, patterns in RECOMMENDATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(r"\b" + re.escape(pattern) + r"\b", text_lower):
                found.add(category)
                break
    return found


def compute_recommendation_agreement(
    report_pairs: list[tuple[str, str]],
) -> dict[str, float]:
    if not report_pairs:
        return {"agreement_rate": 0.0, "jaccard": 0.0, "n_pairs": 0}

    exact_matches = 0
    jaccards = []

    for report_a, report_b in report_pairs:
        recs_a = classify_recommendations(report_a)
        recs_b = classify_recommendations(report_b)

        if recs_a == recs_b:
            exact_matches += 1

        if not recs_a and not recs_b:
            jaccards.append(1.0)
        elif recs_a or recs_b:
            intersection = recs_a & recs_b
            union = recs_a | recs_b
            jaccards.append(len(intersection) / len(union))

    return {
        "agreement_rate": exact_matches / len(report_pairs),
        "jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
        "n_pairs": len(report_pairs),
    }
