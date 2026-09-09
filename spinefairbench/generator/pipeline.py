from __future__ import annotations

from pathlib import Path
from typing import Any

from spinefairbench.generator.config import GeneratorConfig
from spinefairbench.generator.masks import blend_with_source_in_mask, load_binary_mask
from spinefairbench.generator.prompts import get_prompt


class GeneratorDependencyError(RuntimeError):
    pass


class CounterfactualGeneratorPipeline:
    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._pipe: Any = None
        self._torch: Any = None

    def load(self) -> None:
        try:
            import torch
            from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
        except ImportError as exc:
            raise GeneratorDependencyError(
                "Generator inference requires optional dependencies. "
                "Install with: pip install -r requirements-generator.txt"
            ) from exc

        if self.config.checkpoint_path is None:
            raise FileNotFoundError(
                "No LoRA checkpoint path configured. The public benchmark release does "
                "not include generator weights; provide a local Diffusers-compatible "
                "LoRA checkpoint with --checkpoint."
            )
        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")

        device = self._resolve_device(torch)
        dtype = torch.float16 if device != "cpu" else torch.float32
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.base_model,
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        metadata = {}
        if checkpoint_path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
                metadata = checkpoint.metadata() or {}
        if metadata.get("format") == "spinefairbench_lora_safetensors":
            from peft import LoraConfig
            from safetensors.torch import load_file

            # The released tensors retain PEFT's wrapper and adapter names.
            pipe.unet.add_adapter(LoraConfig(
                r=int(metadata["lora_rank"]),
                lora_alpha=int(metadata["lora_alpha"]),
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            ))
            weights = {
                key.removeprefix("base_model.model."): value
                for key, value in load_file(str(checkpoint_path)).items()
            }
            incompatible = pipe.unet.load_state_dict(weights, strict=False)
            if incompatible.unexpected_keys or any("lora_" in key for key in incompatible.missing_keys):
                raise ValueError("Released LoRA tensors do not match the base UNet")
        else:
            pipe.load_lora_weights(str(checkpoint_path.parent), weight_name=checkpoint_path.name)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        self._torch = torch

    def generate(
        self,
        source_image_path: Path,
        target_demographic: str,
        output_path: Path,
        *,
        mask_path: Path | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if self._pipe is None or self._torch is None:
            self.load()

        from PIL import Image

        prompt = get_prompt(target_demographic)
        source = Image.open(source_image_path).convert("RGB").resize(
            (self.config.image_size, self.config.image_size)
        )
        effective_seed = self.config.seed if seed is None else seed
        device = self._pipe.device.type
        generator = self._torch.Generator(device=device).manual_seed(effective_seed)

        result = self._pipe(
            prompt=prompt,
            image=source,
            strength=self.config.strength,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.inference_steps,
            generator=generator,
        )
        generated = result.images[0]

        mask_applied = False
        if mask_path is not None:
            mask = load_binary_mask(mask_path, source.size)
            generated = blend_with_source_in_mask(
                source, generated, mask, self.config.tsxr_mask_blend
            )
            mask_applied = True

        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated.save(output_path)
        return {
            "output_filename": output_path.name,
            "target_demographic": target_demographic,
            "prompt": prompt,
            "seed": effective_seed,
            "inference_steps": self.config.inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "strength": self.config.strength,
            "mask_applied": mask_applied,
            "tsxr_mask_blend": self.config.tsxr_mask_blend if mask_applied else None,
        }

    def _resolve_device(self, torch: Any) -> str:
        if self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"
