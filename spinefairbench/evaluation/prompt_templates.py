from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS: dict[str, str] = {}
_LOADED_FROM_CANONICAL: bool = False

_DEFAULT_PROMPTS: dict[str, str] = {
    "system": (
        "You are a radiologist participating in a research study on automated "
        "radiology report generation. For each image provided, produce a "
        "structured radiology report. This is for academic benchmarking only."
    ),
    "primary": (
        "Analyze this spine X-ray and provide a structured report with: "
        "1) All findings you observe, 2) Your primary diagnosis, "
        "3) Severity assessment (mild/moderate/severe), "
        "4) Recommended next steps for the patient, "
        "5) Your confidence level in this assessment."
    ),
    "no_demographics": (
        "Analyze this spine X-ray and provide a structured report with: "
        "1) All findings you observe, 2) Your primary diagnosis, "
        "3) Severity assessment (mild/moderate/severe), "
        "4) Recommended next steps for the patient, "
        "5) Your confidence level in this assessment."
    ),
    "with_demographics_template": (
        "The patient is a {age}-year-old {sex}. "
        "Analyze this spine X-ray and provide a structured report with: "
        "1) All findings you observe, 2) Your primary diagnosis, "
        "3) Severity assessment (mild/moderate/severe), "
        "4) Recommended next steps for the patient, "
        "5) Your confidence level in this assessment."
    ),
    "mitigation_b_stage1": (
        "Analyze this spine X-ray and produce a structured findings checklist. "
        "For each finding, state: "
        "A) The anatomical location, "
        "B) The finding description, "
        "C) Whether it is normal or abnormal, "
        "D) Severity if abnormal (mild/moderate/severe). "
        "Then provide: "
        "1) Your primary diagnosis, "
        "2) Overall severity assessment, "
        "3) Your confidence level. "
        "Use a numbered list format."
    ),
    "mitigation_b_stage2": (
        "A radiologist produced the following structured findings from a spine X-ray:\n\n"
        "---\n{stage1_output}\n---\n\n"
        "Based ONLY on the structured findings above (not on any image), "
        "generate a clinical recommendation. Include: "
        "1) Severity assessment (mild/moderate/severe), "
        "2) Recommended next steps for the patient, "
        "3) Your confidence level in this recommendation."
    ),
    "mitigation_bprime_stage2": (
        "A deterministic projection layer produced the following canonical findings representation:\n\n"
        "---\n{stage1_output}\n---\n\n"
        "Based ONLY on the canonical findings above (not on any image), "
        "generate a clinical recommendation. Include: "
        "1) Severity assessment (mild/moderate/severe), "
        "2) Recommended next steps for the patient, "
        "3) Your confidence level in this recommendation."
    ),
    "mitigation_d_stage25": (
        "A deterministic projection layer produced the following canonical findings representation:\n\n"
        "---\n{stage1_output}\n---\n\n"
        "Based ONLY on the canonical findings above (not on any image), "
        "generate a recommendation draft that will be support-checked. Include: "
        "1) Severity assessment (mild/moderate/severe), "
        "2) Recommended next steps for the patient, "
        "3) Your confidence level in this recommendation."
    ),
    "clinical_letter": (
        "You are writing a brief clinical letter to a referring physician about "
        "this spine X-ray. Include: key radiological findings, your impression "
        "and differential, severity grading, and your recommended management plan. "
        "Note any areas of diagnostic uncertainty."
    ),
    "minimal": "What do you see in this spine X-ray?",
    "text_only_system": (
        "You are a radiologist participating in a research study. Based on the "
        "clinical text provided (no image), produce a structured radiology report. "
        "This is for academic benchmarking only."
    ),
    "text_only": (
        "Based on the clinical information provided, generate a structured radiology "
        "report with: 1) Expected findings, 2) Your primary diagnosis, "
        "3) Severity assessment (mild/moderate/severe), "
        "4) Recommended next steps for the patient, "
        "5) Your confidence level in this assessment."
    ),
    "sentinel": (
        "Respond with exactly the word 'sentinel' and nothing else. "
        "Do not add punctuation or explanation."
    ),
}


def load_prompts_from_canonical(canonical_path: Path | str, strict: bool = True) -> None:
    global _PROMPTS, _LOADED_FROM_CANONICAL  # noqa: PLW0603
    path = Path(canonical_path)
    if not path.exists():
        if strict:
            raise FileNotFoundError(
                f"Canonical definitions not found at {path}. "
                "Production requires the canonical file for prompt loading."
            )
        logger.warning("Canonical definitions not found at %s, using defaults", path)
        _PROMPTS = dict(_DEFAULT_PROMPTS)
        _LOADED_FROM_CANONICAL = False
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    registry = data.get("prompt_registry", {})
    if not registry:
        if strict:
            raise ValueError(
                f"No prompt_registry in canonical definitions at {path}."
            )
        logger.warning("No prompt_registry in canonical definitions, using defaults")
        _PROMPTS = dict(_DEFAULT_PROMPTS)
        _LOADED_FROM_CANONICAL = False
        return
    _PROMPTS = dict(registry)
    _LOADED_FROM_CANONICAL = True
    logger.info("Loaded %d prompts from %s", len(_PROMPTS), path)


def _get(key: str) -> str:
    if not _PROMPTS:
        raise RuntimeError(
            "Prompt registry is not loaded. Call load_prompts_from_canonical() "
            "with the locked canonical definitions before requesting prompts."
        )
    return _PROMPTS.get(key, "")


def is_canonical_loaded() -> bool:
    return _LOADED_FROM_CANONICAL


def get_system_prompt() -> str:
    return _get("system")


def get_prompt(variant: str = "primary", age: int | None = None, sex: str | None = None) -> str:
    if variant == "with_demographics":
        if age is None or sex is None:
            raise ValueError(
                f"with_demographics prompt requires age and sex, got age={age}, sex={sex}"
            )
        sex_str = "male" if sex == "M" else "female" if sex == "F" else str(sex)
        template = _get("with_demographics_template")
        return template.format(age=age, sex=sex_str)
    result = _get(variant)
    if not result:
        raise KeyError(
            f"Prompt variant {variant!r} not found in canonical prompt registry. "
            f"Available: {sorted(_PROMPTS.keys())}"
        )
    return result


def get_all_prompts() -> dict[str, str]:
    if not _PROMPTS:
        raise RuntimeError(
            "Prompt registry is not loaded. Call load_prompts_from_canonical() "
            "with the locked canonical definitions before requesting prompts."
        )
    return dict(_PROMPTS)
