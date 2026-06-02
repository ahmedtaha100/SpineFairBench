from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QCThresholds:
    ssim_min: float = 0.70
    edge_preservation_min: float = 0.276
    lpips_max: float = 0.40
    edge_dilation_kernel: int = 3


@dataclass(frozen=True)
class GeneratorConfig:
    base_model: str = "runwayml/stable-diffusion-v1-5"
    lora_rank: int = 64
    lora_alpha: int = 128
    inference_steps: int = 50
    guidance_scale: float = 5.0
    strength: float = 0.15
    seed: int = 42
    image_size: int = 512
    tsxr_mask_blend: float = 0.7
    checkpoint_path: str | None = None
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "GeneratorConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: values[key] for key in values if key in known})


def load_yaml_config(path: Path) -> GeneratorConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load generator YAML configs. "
            "Install optional dependencies with: pip install -r requirements-generator.txt"
        ) from exc

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return GeneratorConfig.from_mapping(data)
