from __future__ import annotations

DEMOGRAPHIC_PROMPTS: dict[str, str] = {
    "elderly_female": "Lumbar spine X-ray of a 75-year-old female patient",
    "elderly_male": "Lumbar spine X-ray of a 75-year-old male patient",
    "young_female": "Lumbar spine X-ray of a 25-year-old female patient",
    "young_male": "Lumbar spine X-ray of a 25-year-old male patient",
}


def get_prompt(name: str) -> str:
    try:
        return DEMOGRAPHIC_PROMPTS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(DEMOGRAPHIC_PROMPTS))
        raise ValueError(f"Unknown demographic prompt {name!r}; valid prompts: {valid}") from exc
