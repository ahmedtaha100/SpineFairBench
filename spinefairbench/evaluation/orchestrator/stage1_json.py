from __future__ import annotations

from typing import Any


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_string(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key not in item:
            continue
        normalized = _string_value(item.get(key))
        if normalized:
            return normalized
    return ""


def normalize_stage1_finding(
    item: dict[str, Any],
    *,
    fill_defaults: bool,
) -> dict[str, str]:
    """Normalize a Stage-1 finding dict to canonical Scout field names."""
    raw_location = _first_string(item, "location")
    raw_level = _first_string(item, "spinal_level", "lvl")
    if not raw_level and raw_location:
        raw_level = raw_location

    spinal_level = "unspecified" if raw_level == "unknown" else raw_level
    if raw_location:
        location = "unknown" if raw_location == "unspecified" else raw_location
    elif spinal_level:
        location = "unknown" if spinal_level == "unspecified" else spinal_level
    else:
        location = ""

    normalized = {
        "location": location,
        "spinal_level": spinal_level,
        "finding_type": _first_string(item, "finding_type", "type"),
        "normality": _first_string(item, "normality", "norm"),
        "severity": _first_string(item, "severity", "sev"),
        "laterality": _first_string(item, "laterality", "lat"),
        "alignment": _first_string(item, "alignment", "alg"),
    }

    if fill_defaults:
        return {
            "location": normalized["location"] or "unknown",
            "spinal_level": normalized["spinal_level"] or "unspecified",
            "finding_type": normalized["finding_type"] or "other",
            "normality": normalized["normality"] or "abnormal",
            "severity": normalized["severity"] or "unspecified",
            "laterality": normalized["laterality"] or "unspecified",
            "alignment": normalized["alignment"] or "unspecified",
        }

    return {key: value for key, value in normalized.items() if value}


def normalize_stage1_payload(
    payload: dict[str, Any],
    *,
    fill_defaults: bool,
) -> dict[str, Any]:
    """Normalize Scout Stage-1 payload aliases while optionally preserving missing keys."""
    findings_payload = payload.get("findings")
    normalized_findings: Any
    if isinstance(findings_payload, list):
        normalized_findings = []
        for item in findings_payload:
            if not isinstance(item, dict):
                normalized_findings.append(item)
                continue
            normalized_findings.append(normalize_stage1_finding(item, fill_defaults=fill_defaults))
    elif fill_defaults:
        normalized_findings = []
    else:
        normalized_findings = findings_payload

    result: dict[str, Any] = {}
    if findings_payload is not None or fill_defaults:
        result["findings"] = normalized_findings

    primary_diagnosis_code = _first_string(payload, "primary_diagnosis_code", "dx")
    overall_severity = _first_string(payload, "overall_severity", "overall")
    confidence_level = _first_string(payload, "confidence_level", "conf")

    if fill_defaults:
        result["primary_diagnosis_code"] = primary_diagnosis_code or "other"
        result["overall_severity"] = overall_severity or "unspecified"
        result["confidence_level"] = confidence_level or "medium"
        return result

    if primary_diagnosis_code:
        result["primary_diagnosis_code"] = primary_diagnosis_code
    if overall_severity:
        result["overall_severity"] = overall_severity
    if confidence_level:
        result["confidence_level"] = confidence_level
    return result
