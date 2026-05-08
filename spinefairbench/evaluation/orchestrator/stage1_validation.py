from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from spinefairbench.evaluation.orchestrator.output_parser import (
    compute_parser_confidence,
    parse_radiology_output,
)
from spinefairbench.evaluation.orchestrator.stage1_json import normalize_stage1_payload

PROMPT_ECHO_MARKERS = (
    "for each finding, state",
    "a) the anatomical location",
    "b) the finding description",
    "c) whether it is normal or abnormal",
    "d) severity if abnormal (mild/moderate/severe)",
    "then provide:",
    "use a numbered list format",
)

HTML_GARBAGE_PATTERN = re.compile(r"</?[a-z][^>]*>|&(?:nbsp|amp|lt|gt|quot);|&#\d+;", re.IGNORECASE)
SCOUT_STAGE1_REQUIRED_KEYS = frozenset({
    "findings",
    "primary_diagnosis_code",
    "overall_severity",
    "confidence_level",
})
SCOUT_STAGE1_FINDING_KEYS = frozenset({
    "location",
    "spinal_level",
    "finding_type",
    "normality",
    "severity",
    "laterality",
    "alignment",
})
SCOUT_STAGE1_FINDING_ENUMS = {
    "location": {"cervical", "thoracic", "lumbar", "sacral", "multilevel", "unknown"},
    "spinal_level": {"cervical", "thoracic", "lumbar", "sacral", "multilevel", "unspecified"},
    "finding_type": {
        "normal_exam",
        "disc_space_narrowing",
        "foraminal_stenosis",
        "osteophyte",
        "fracture",
        "spondylolisthesis",
        "alignment_issue",
        "other",
    },
    "normality": {"normal", "abnormal"},
    "severity": {"none", "mild", "moderate", "severe", "unspecified"},
    "laterality": {"left", "right", "bilateral", "unspecified"},
    "alignment": {"present", "absent", "unspecified"},
}
SCOUT_STAGE1_TOP_LEVEL_ENUMS = {
    "primary_diagnosis_code": SCOUT_STAGE1_FINDING_ENUMS["finding_type"],
    "overall_severity": {"normal", "mild", "moderate", "severe", "unspecified"},
    "confidence_level": {"low", "medium", "high"},
}
SCOUT_TEMPLATE_ECHO_PATTERN = re.compile(
    r"findings checklist|use a numbered list format|single json object|use this exact schema|"
    r"return raw json only|no markdown, headings, bullets, numbering",
    re.IGNORECASE,
)
STAGE1_VALIDATOR_VERSION = "2026-04-13.1"
_SCOUT_STAGE1_TOP_LEVEL_ALIASES = frozenset({"dx", "overall", "conf"})
_SCOUT_STAGE1_FINDING_ALIASES = frozenset({"lvl", "type", "norm", "sev", "lat", "alg"})


def validate_stage1_output(raw_output: str, *, model_name: str | None = None) -> dict[str, Any]:
    stripped = raw_output.strip()
    lower = stripped.lower()
    if model_name and "llama-4-scout" in model_name.lower():
        return _validate_scout_stage1_output(raw_output)

    parsed = parse_radiology_output(raw_output, model_name=model_name)
    parser_confidence = compute_parser_confidence(parsed, model_name=model_name)

    prompt_echo_hits = sum(marker in lower for marker in PROMPT_ECHO_MARKERS)
    prompt_echo_detected = prompt_echo_hits >= 2
    html_garbage_detected = bool(HTML_GARBAGE_PATTERN.search(raw_output))

    reasons: list[str] = []
    if raw_output == "":
        reasons.append("blank")
    elif stripped == "":
        reasons.append("whitespace_only")
    if parser_confidence == "failed":
        reasons.append("parser_failed")
    if prompt_echo_detected:
        reasons.append("prompt_echo")
    if html_garbage_detected:
        reasons.append("html_garbage")

    return {
        "status": "valid" if not reasons else "invalid",
        "validator_version": STAGE1_VALIDATOR_VERSION,
        "reasons": reasons,
        "parser_confidence_flag": parser_confidence,
        "prompt_echo_detected": prompt_echo_detected,
        "prompt_echo_marker_hits": prompt_echo_hits,
        "html_garbage_detected": html_garbage_detected,
        "raw_length": len(raw_output),
        "non_whitespace_length": len(stripped),
        "fields_found": list(parsed.get("fields_found", [])),
        "fields_missing": list(parsed.get("fields_missing", [])),
        "validated_at_utc": datetime.now(UTC).isoformat(),
    }


def _validate_scout_stage1_output(raw_output: str) -> dict[str, Any]:
    stripped = raw_output.strip()
    reasons: list[str] = []
    prompt_echo_detected = bool(SCOUT_TEMPLATE_ECHO_PATTERN.search(raw_output))
    html_garbage_detected = bool(HTML_GARBAGE_PATTERN.search(raw_output))

    if raw_output == "":
        reasons.append("blank")
    elif stripped == "":
        reasons.append("whitespace_only")

    parsed = parse_radiology_output(raw_output, model_name="llama-4-scout")
    fields_found = list(parsed.get("fields_found", []))
    fields_missing = list(parsed.get("fields_missing", []))

    scout_payload: dict[str, Any] | None = None
    schema_errors: list[str] = []
    if stripped and not reasons:
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            reasons.append("non_json_output")
        else:
            if not isinstance(loaded, dict):
                reasons.append("non_json_output")
            else:
                schema_errors.extend(_validate_scout_stage1_raw_shape(loaded))
                scout_payload = normalize_stage1_payload(loaded, fill_defaults=False)

    if prompt_echo_detected:
        reasons.append("prompt_echo")
    if html_garbage_detected:
        reasons.append("html_garbage")

    if scout_payload is not None:
        schema_errors.extend(_validate_scout_stage1_schema(scout_payload))
        if schema_errors:
            reasons.append("scout_schema_invalid")

    parser_confidence = "clean" if scout_payload is not None and not schema_errors and not reasons else "failed"

    return {
        "status": "valid" if not reasons else "invalid",
        "validator_version": STAGE1_VALIDATOR_VERSION,
        "reasons": reasons,
        "parser_confidence_flag": parser_confidence,
        "prompt_echo_detected": prompt_echo_detected,
        "prompt_echo_marker_hits": 1 if prompt_echo_detected else 0,
        "html_garbage_detected": html_garbage_detected,
        "raw_length": len(raw_output),
        "non_whitespace_length": len(stripped),
        "fields_found": fields_found,
        "fields_missing": fields_missing,
        "scout_schema_errors": schema_errors,
        "validated_at_utc": datetime.now(UTC).isoformat(),
    }


def _validate_scout_stage1_schema(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    extra_top_level = sorted(set(payload) - SCOUT_STAGE1_REQUIRED_KEYS)
    missing_top_level = sorted(SCOUT_STAGE1_REQUIRED_KEYS - set(payload))
    if extra_top_level:
        errors.append(f"unexpected_top_level_keys:{','.join(extra_top_level)}")
    if missing_top_level:
        errors.append(f"missing_top_level_keys:{','.join(missing_top_level)}")

    findings = payload.get("findings")
    if not isinstance(findings, list):
        errors.append("findings_not_list")
        findings = []

    for key, allowed in SCOUT_STAGE1_TOP_LEVEL_ENUMS.items():
        value = payload.get(key)
        if not isinstance(value, str) or value not in allowed:
            errors.append(f"invalid_{key}")

    for idx, finding in enumerate(findings):
        if not isinstance(finding, dict):
            errors.append(f"finding_{idx}_not_object")
            continue
        extra_keys = sorted(set(finding) - SCOUT_STAGE1_FINDING_KEYS)
        missing_keys = sorted(SCOUT_STAGE1_FINDING_KEYS - set(finding))
        if extra_keys:
            errors.append(f"finding_{idx}_unexpected_keys:{','.join(extra_keys)}")
        if missing_keys:
            errors.append(f"finding_{idx}_missing_keys:{','.join(missing_keys)}")
        for key, allowed in SCOUT_STAGE1_FINDING_ENUMS.items():
            value = finding.get(key)
            if not isinstance(value, str) or value not in allowed:
                errors.append(f"finding_{idx}_invalid_{key}")
    return errors


def _validate_scout_stage1_raw_shape(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed_top_level = SCOUT_STAGE1_REQUIRED_KEYS | _SCOUT_STAGE1_TOP_LEVEL_ALIASES
    extra_top_level = sorted(set(payload) - allowed_top_level)
    if extra_top_level:
        errors.append(f"unexpected_top_level_keys:{','.join(extra_top_level)}")

    findings = payload.get("findings")
    if not isinstance(findings, list):
        return errors

    allowed_finding_keys = SCOUT_STAGE1_FINDING_KEYS | _SCOUT_STAGE1_FINDING_ALIASES
    for idx, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        extra_keys = sorted(set(finding) - allowed_finding_keys)
        if extra_keys:
            errors.append(f"finding_{idx}_unexpected_keys:{','.join(extra_keys)}")
    return errors
