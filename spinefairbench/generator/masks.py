from __future__ import annotations

from pathlib import Path
from typing import Any


def load_binary_mask(mask_path: Path, size: tuple[int, int]) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow and numpy are required for generator mask utilities. "
            "Install optional dependencies with: pip install -r requirements-generator.txt"
        ) from exc

    mask = Image.open(mask_path).convert("L").resize(size)
    arr = np.asarray(mask, dtype="float32") / 255.0
    return (arr > 0.5).astype("float32")


def blend_with_source_in_mask(source_image: Any, generated_image: Any, mask: Any, blend: float) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow and numpy are required for generator mask utilities. "
            "Install optional dependencies with: pip install -r requirements-generator.txt"
        ) from exc

    source = source_image.convert("RGB")
    generated = generated_image.convert("RGB").resize(source.size)
    if mask.shape != (source.size[1], source.size[0]):
        mask_img = Image.fromarray((mask * 255).astype("uint8")).resize(source.size)
        mask = np.asarray(mask_img, dtype="float32") / 255.0

    source_arr = np.asarray(source, dtype="float32")
    generated_arr = np.asarray(generated, dtype="float32")
    mask_arr = mask[..., None].astype("float32")
    out = generated_arr * (1.0 - mask_arr * blend) + source_arr * (mask_arr * blend)
    return Image.fromarray(out.clip(0, 255).astype("uint8"), mode="RGB")
