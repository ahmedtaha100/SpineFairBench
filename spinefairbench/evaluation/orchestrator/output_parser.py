from __future__ import annotations

import json
import logging
import re
from typing import Any

from spinefairbench.evaluation.orchestrator.stage1_json import normalize_stage1_payload
from spinefairbench.metrics.refusal_detector import ResponseClass, classify_response

logger = logging.getLogger(__name__)

EXPECTED_FIELDS = frozenset({
    "findings",
    "diagnosis",
    "severity",
    "recommendation",
    "confidence",
})
SCOUT_EXPECTED_FIELDS = frozenset({
    "findings",
    "diagnosis",
    "severity",
    "confidence",
})

MEDICAL_CONTENT_TERMS = frozenset({
    "acetabular",
    "alignment",
    "ankylosis",
    "arthroplasty",
    "cervical",
    "compression",
    "degenerative",
    "disc space",
    "dislocation",
    "dysplasia",
    "endplate",
    "femoral",
    "foramen",
    "foraminal stenosis",
    "fracture",
    "hip",
    "impression",
    "intervertebral",
    "kyphosis",
    "lordosis",
    "lumbar",
    "lucency",
    "narrowing",
    "normal",
    "observations",
    "osteophyte",
    "pars defect",
    "pedicle",
    "radiograph",
    "radiographs",
    "radiological",
    "radiographically diagnosed",
    "sacr",
    "scoliosis",
    "sclerotic",
    "severe",
    "spondyl",
    "spondyloepiphyseal",
    "spinous process",
    "stenosis",
    "thoracic",
    "unremarkable",
    "vertebr",
})

SEVERITY_TERMS = frozenset({
    "mild", "moderate", "severe", "normal", "unremarkable",
    "minimal", "significant", "marked", "advanced",
})

def parse_radiology_output(raw_output: str, *, model_name: str | None = None) -> dict[str, Any]:
    expected_fields = _expected_fields(model_name)
    result: dict[str, Any] = {
        "findings": "",
        "diagnosis": "",
        "severity": "",
        "recommendation": "",
        "confidence": "",
        "raw_length": len(raw_output),
        "fields_found": [],
        "fields_missing": [],
    }

    if not raw_output or not raw_output.strip():
        result["fields_missing"] = list(expected_fields)
        return result

    text = raw_output.strip()
    lower = text.lower()

    if model_name and "llama-4-scout" in model_name.lower() and text.startswith("{"):
        _apply_model_specific_fallbacks(result, text, model_name=model_name)
        result["fields_missing"] = [f for f in expected_fields if f not in result["fields_found"]]
        return result

    findings = _extract_section(text, [
        r"(?:1\)|1\.|findings?:?)\s*(.+?)(?=(?:^|\n)\s*(?:2\)|2\.|(?:primary )?diagnosis\b)|\Z)",
        r"(?:findings?|observations?|radiological findings)\s*[:]\s*(.+?)(?=(?:^|\n)\s*(?:diagnosis|severity)\b|\Z)",
    ])
    if findings:
        result["findings"] = findings.strip()
        _mark_field_found(result, "findings")

    diagnosis = _extract_section(text, [
        r"(?:2\)|2\.|(?:primary )?diagnosis:?)\s*(.+?)(?=(?:3\)|3\.|severity|\Z))",
        r"(?:diagnosis|impression)\s*[:]\s*(.+?)(?=(?:severity|recommendation|\Z))",
    ])
    if diagnosis:
        result["diagnosis"] = diagnosis.strip()
        _mark_field_found(result, "diagnosis")

    severity = _extract_section(text, [
        r"(?:3\)|3\.|severity(?: assessment)?:?)\s*(.+?)(?=(?:4\)|4\.|recommend|\Z))",
        r"(?:severity)\s*[:]\s*(.+?)(?=(?:recommend|confidence|next steps|\Z))",
    ])
    if severity:
        result["severity"] = severity.strip()
        _mark_field_found(result, "severity")
    elif any(term in lower for term in SEVERITY_TERMS):
        for term in SEVERITY_TERMS:
            if term in lower:
                result["severity"] = term
                _mark_field_found(result, "severity")
                break

    recommendation = _extract_section(text, [
        r"(?:4\)|4\.|recommend(?:ation|ed)?s?(?: next steps)?:?)\s*(.+?)(?=(?:5\)|5\.|confidence|\Z))",
        r"(?:next steps|management|follow[- ]?up)\s*[:]\s*(.+?)(?=(?:confidence|\Z))",
    ])
    if recommendation:
        result["recommendation"] = recommendation.strip()
        _mark_field_found(result, "recommendation")

    confidence = _extract_section(text, [
        r"(?:5\)|5\.|confidence(?: level)?:?)\s*(.+?)$",
        r"(?:confidence)\s*[:]\s*(.+?)$",
    ])
    if confidence:
        result["confidence"] = confidence.strip()
        _mark_field_found(result, "confidence")

    _apply_model_specific_fallbacks(result, text, model_name=model_name)
    result["fields_missing"] = [f for f in expected_fields if f not in result["fields_found"]]
    return result


def compute_parser_confidence(parsed: dict[str, Any], *, model_name: str | None = None) -> str:
    found = len(set(parsed.get("fields_found", [])))
    total = len(_expected_fields(model_name))
    raw_len = parsed.get("raw_length", 0)

    if raw_len < 10:
        return "failed"
    if found == total:
        return "clean"
    if found >= 1:
        return "partial"
    return "failed"


def _expected_fields(model_name: str | None) -> frozenset[str]:
    if model_name and "llama-4-scout" in model_name.lower():
        return SCOUT_EXPECTED_FIELDS
    return EXPECTED_FIELDS


def classify_refusal(raw_output: str) -> str:
    if not raw_output or not raw_output.strip():
        return "parse_failure"

    lower = raw_output.lower().strip()

    if len(lower) < 20:
        return "parse_failure"

    response_class = classify_response(raw_output)
    if response_class == ResponseClass.FULL_REFUSAL:
        return "full_refusal"
    if response_class == ResponseClass.PARTIAL_REFUSAL:
        return "partial_refusal"
    if len(lower) < 50 and not _contains_medical_content(lower):
        return "parse_failure"
    return "genuine"


def _extract_section(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
    return ""


def _contains_medical_content(lower: str) -> bool:
    return any(term in lower for term in MEDICAL_CONTENT_TERMS)


def _contains_radfm_shortform_content(lower: str) -> bool:
    return any(
        term in lower
        for term in (
            "alignment",
            "ankylosis",
            "cervical",
            "compression",
            "disc space",
            "dislocation",
            "endplate",
            "foraminal stenosis",
            "fracture",
            "kyphosis",
            "lordosis",
            "lumbar",
            "osteophyte",
            "pars defect",
            "radiographically diagnosed",
            "scoliosis",
            "spondyl",
            "stenosis",
            "thoracic",
            "vertebr",
        )
    )


def _mark_field_found(parsed: dict[str, Any], field_name: str) -> None:
    fields_found = parsed.setdefault("fields_found", [])
    if field_name not in fields_found:
        fields_found.append(field_name)


def _apply_model_specific_fallbacks(
    parsed: dict[str, Any],
    raw_text: str,
    *,
    model_name: str | None,
) -> None:
    if model_name and "llama-4-scout" in model_name.lower():
        _apply_scout_json_fallback(parsed, raw_text)
        if parsed.get("findings") or parsed.get("diagnosis"):
            return

    if not model_name or "radfm" not in model_name.lower():
        return
    if parsed.get("findings") or parsed.get("diagnosis"):
        return

    text = " ".join(raw_text.split())
    lower = text.lower()

    # RadFM often emits terse diagnosis-only strings or score-like summaries
    # without the numbered report scaffold required by the generic parser.
    if lower.startswith("6) remarks") or lower.startswith("remarks"):
        parsed["findings"] = text
        _mark_field_found(parsed, "findings")
        return

    if len(text) < 15 or not _contains_radfm_shortform_content(lower):
        return

    if ":" in text and "%" in text:
        parsed["findings"] = text
        _mark_field_found(parsed, "findings")
        return

    parsed["diagnosis"] = text
    _mark_field_found(parsed, "diagnosis")

    if not parsed["severity"]:
        for term in SEVERITY_TERMS:
            if term in lower:
                parsed["severity"] = term
                _mark_field_found(parsed, "severity")
                break


def _apply_scout_json_fallback(parsed: dict[str, Any], raw_text: str) -> None:
    text = (raw_text or "").strip()
    if not text.startswith("{"):
        return
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    payload = normalize_stage1_payload(payload, fill_defaults=False)

    findings = payload.get("findings")
    if isinstance(findings, list):
        _mark_field_found(parsed, "findings")
        finding_codes = [
            str(item.get("finding_type") or "").strip()
            for item in findings
            if isinstance(item, dict) and str(item.get("finding_type") or "").strip()
        ]
        if finding_codes:
            parsed["findings"] = ", ".join(finding_codes)

    diagnosis = str(payload.get("primary_diagnosis_code") or "").strip()
    if diagnosis:
        parsed["diagnosis"] = diagnosis
        _mark_field_found(parsed, "diagnosis")

    severity = str(payload.get("overall_severity") or "").strip()
    if severity:
        parsed["severity"] = severity
        _mark_field_found(parsed, "severity")

    confidence = str(payload.get("confidence_level") or "").strip()
    if confidence:
        parsed["confidence"] = confidence
        _mark_field_found(parsed, "confidence")
