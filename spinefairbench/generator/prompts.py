from __future__ import annotations

DEMOGRAPHIC_PROMPTS: dict[str, str] = {
    "elderly_female": "75-year-old female lumbar spine radiograph",
    "elderly_male": "75-year-old male lumbar spine radiograph",
    "young_female": "25-year-old female lumbar spine radiograph",
    "young_male": "25-year-old male lumbar spine radiograph",
}


def get_prompt(name: str) -> str:
    try:
        return DEMOGRAPHIC_PROMPTS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(DEMOGRAPHIC_PROMPTS))
        raise ValueError(f"Unknown demographic prompt {name!r}; valid prompts: {valid}") from exc
