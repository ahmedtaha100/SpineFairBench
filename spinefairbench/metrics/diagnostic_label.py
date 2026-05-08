from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Frozen released synonym registry. Some broad synonyms, especially the
# fracture aliases "fx", "break", and "broken", are intentionally preserved
# because changing them alters released Table 2 values.
PATHOLOGY_SYNONYMS: dict[str, list[str]] = {
    "Osteophytes": ["osteophyte", "osteophytes", "bone spur", "bone spurs", "bony spur"],
    "Foraminal stenosis": ["foraminal stenosis", "neural foraminal narrowing", "foraminal narrowing"],
    "Disc space narrowing": ["disc space narrowing", "disc height loss", "disc narrowing", "reduced disc height"],
    "Spondylolisthesis": ["spondylolisthesis", "spondylolysthesis", "vertebral slippage", "anterolisthesis", "retrolisthesis"],
    "Surgical implant": ["surgical implant", "hardware", "fixation device", "pedicle screw", "surgical hardware", "instrumentation"],
    "Vertebral collapse": ["vertebral collapse", "compression fracture", "vertebral body collapse", "collapsed vertebra"],
    "Other lesions": ["lesion", "other lesion", "abnormality"],
    "No finding": ["normal", "no finding", "no abnormality", "unremarkable", "within normal limits", "no significant finding"],
    "Posterior osteophyte": ["posterior osteophyte", "posterior bone spur", "dorsal osteophyte"],
    "Endplate sclerosis": ["endplate sclerosis", "sclerotic endplate", "endplate changes"],
    "Schmorl's node": ["schmorl", "schmorl's node", "schmorl node", "intravertebral herniation"],
    "Disc calcification": ["disc calcification", "calcified disc", "discal calcification"],
    "Fracture": ["fracture", "fx", "break", "broken"],
}

STRICT_PATHOLOGY_SYNONYMS: dict[str, list[str]] = {
    **PATHOLOGY_SYNONYMS,
    "Fracture": ["fracture"],
}


def extract_labels(text: str) -> set[str]:
    text_lower = text.lower()
    found: set[str] = set()
    for category, synonyms in PATHOLOGY_SYNONYMS.items():
        for synonym in synonyms:
            if re.search(r"\b" + re.escape(synonym) + r"\b", text_lower):
                found.add(category)
                break
    return found


def extract_labels_strict(text: str) -> set[str]:
    """Return labels with narrower non-frozen fracture synonyms.

    This helper is provided for ad hoc audits. The benchmark scorer uses
    ``extract_labels`` and the frozen synonym registry above.
    """

    text_lower = text.lower()
    found: set[str] = set()
    for category, synonyms in STRICT_PATHOLOGY_SYNONYMS.items():
        for synonym in synonyms:
            if re.search(r"\b" + re.escape(synonym) + r"\b", text_lower):
                found.add(category)
                break
    return found


def compute_jaccard(labels_a: set[str], labels_b: set[str]) -> float:
    if not labels_a and not labels_b:
        return 1.0
    intersection = labels_a & labels_b
    union = labels_a | labels_b
    return len(intersection) / len(union)


def compute_label_agreement(
    report_pairs: list[tuple[str, str]],
) -> dict[str, float]:
    if not report_pairs:
        return {"mean_jaccard": 0.0, "agreement_rate": 0.0, "n_pairs": 0}

    jaccards = []
    exact_matches = 0

    for report_a, report_b in report_pairs:
        labels_a = extract_labels(report_a)
        labels_b = extract_labels(report_b)
        j = compute_jaccard(labels_a, labels_b)
        jaccards.append(j)
        if labels_a == labels_b:
            exact_matches += 1

    mean_jaccard = sum(jaccards) / len(jaccards)
    agreement_rate = exact_matches / len(report_pairs)

    return {
        "mean_jaccard": mean_jaccard,
        "agreement_rate": agreement_rate,
        "n_pairs": len(report_pairs),
    }
