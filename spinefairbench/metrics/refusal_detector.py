from __future__ import annotations

import re
from enum import Enum
from typing import Any


class ResponseClass(Enum):
    GENUINE = "genuine"
    PARTIAL_REFUSAL = "partial_refusal"
    FULL_REFUSAL = "full_refusal"


_REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi('m| am) unable to\b",
        r"\bi can(?:'t|not)\b.*(?:analy[sz]e|interpret|examine|diagnose|review|read|assess|evaluate)\b.*(?:image|x[- ]?ray|radiograph|scan|medical)",
        r"\bcannot provide\b.*(?:diagnosis|medical|clinical|interpretation)",
        r"\bnot able to\b.*(?:analy[sz]e|interpret|examine|diagnose)",
        r"\bi('m| am) not (?:qualified|able|authorized)\b",
        r"\bconsult a (?:qualified |)(?:radiologist|healthcare|medical)\b",
        r"\bi (?:cannot|can(?:'t| not)) (?:provide|offer|give)\b.*(?:medical|clinical|diagnostic)\b",
    ]
]

_CLINICAL_CONTENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:this|the) (?:x[- ]?ray|radiograph|image) (?:shows|demonstrates|reveals|indicates)\b",
        r"\bi (?:observe|see|note|identify|detect)\b",
        r"\bfindings?:\s*[^\s]",
        r"\bprimary diagnosis:\s*(?!consult\b|see\b|follow\s*up\b|unknown\b|n/?a\b)[^\s]",
        r"\bdiagnosis:\s*(?!consult\b|see\b|follow\s*up\b|unknown\b|n/?a\b)[^\s]",
        r"\bimpression:\s*(?!consult\b|see\b|follow\s*up\b|unknown\b|n/?a\b)[^\s]",
    ]
]

_HEDGED_CLINICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\ba radiologist (?:might|would|could|may)\b",
        r"\b(?:possible|potential) (?:findings|conditions|diagnos[ei]s)\b",
        r"\bgenerally?\b.*\blook for\b",
        r"\b(?:might|could) include\b",
        r"\bin general\b.*(?:radiologist|x[- ]?ray|spine)",
    ]
]


def classify_response(text: str) -> ResponseClass:
    if not text or not text.strip():
        return ResponseClass.FULL_REFUSAL

    has_refusal = any(p.search(text) for p in _REFUSAL_PATTERNS)

    if not has_refusal:
        return ResponseClass.GENUINE

    has_direct_clinical = any(p.search(text) for p in _CLINICAL_CONTENT_PATTERNS)
    if has_direct_clinical:
        return ResponseClass.GENUINE

    has_hedged = any(p.search(text) for p in _HEDGED_CLINICAL_PATTERNS)
    if has_hedged:
        return ResponseClass.PARTIAL_REFUSAL

    return ResponseClass.FULL_REFUSAL


def is_refusal(text: str) -> bool:
    return classify_response(text) != ResponseClass.GENUINE


def compute_data_quality(
    eval_results: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    quality: dict[str, dict[str, int]] = {}
    source_records: dict[
        str,
        dict[str, tuple[bool, ResponseClass | None]],
    ] = {}
    generated_records: dict[
        str,
        dict[str, tuple[bool, ResponseClass | None]],
    ] = {}

    for r in eval_results:
        model = r.get("model", "unknown")
        if model not in quality:
            quality[model] = {
                "total": 0,
                "api_errors": 0,
                "full_refusals": 0,
                "partial_refusals": 0,
                "genuine": 0,
                "pair_total": 0,
                "pair_api_errors": 0,
                "pair_full_refusals": 0,
                "pair_partial_refusals": 0,
                "pair_missing_source": 0,
                "pair_usable": 0,
            }
            source_records[model] = {}
            generated_records[model] = {}
        quality[model]["total"] += 1

        pair_id = r.get("pair_source_id", "")
        img_path = r.get("image_path", "")
        role = r.get("image_role", "")
        has_error = bool(r.get("error"))

        cls: ResponseClass | None = None
        if has_error:
            quality[model]["api_errors"] += 1
        else:
            response = r.get("response", "") or ""
            cls = classify_response(response)
            if cls == ResponseClass.FULL_REFUSAL:
                quality[model]["full_refusals"] += 1
            elif cls == ResponseClass.PARTIAL_REFUSAL:
                quality[model]["partial_refusals"] += 1
            else:
                quality[model]["genuine"] += 1

        if pair_id:
            if role == "source" or (not role and "_source" in img_path):
                source_records[model][pair_id] = (has_error, cls)
            elif role == "generated" or (not role and "_source" not in img_path):
                key = f"{pair_id}::{img_path}"
                generated_records[model][key] = (has_error, cls)

    for model, counts in quality.items():
        usable = counts["genuine"] + counts["partial_refusals"]
        counts["usable"] = usable
        counts["excluded"] = counts["total"] - usable

        model_sources = source_records[model]
        model_generated = generated_records[model]
        for gen_key, (gen_error, gen_cls) in model_generated.items():
            counts["pair_total"] += 1
            src_id = gen_key.split("::")[0]
            src_record = model_sources.get(src_id)
            if src_record is None:
                counts["pair_missing_source"] += 1
                continue

            src_error, src_cls = src_record
            if src_error or gen_error:
                counts["pair_api_errors"] += 1
                continue

            if (
                src_cls == ResponseClass.FULL_REFUSAL
                or gen_cls == ResponseClass.FULL_REFUSAL
            ):
                counts["pair_full_refusals"] += 1
                continue

            counts["pair_usable"] += 1
            if (
                src_cls == ResponseClass.PARTIAL_REFUSAL
                or gen_cls == ResponseClass.PARTIAL_REFUSAL
            ):
                counts["pair_partial_refusals"] += 1

        counts["pair_excluded"] = counts["pair_total"] - counts["pair_usable"]

    return quality
