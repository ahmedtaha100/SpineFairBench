from __future__ import annotations

from pathlib import Path
from typing import Any

from spinefairbench.generator.config import QCThresholds


def _require_numpy_pillow() -> tuple[Any, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow and numpy are required for generator QC. "
            "Install optional dependencies with: pip install -r requirements-generator.txt"
        ) from exc
    return np, Image


def compute_ssim(image_a: Any, image_b: Any, data_range: float = 1.0) -> float:
    np, _ = _require_numpy_pillow()
    a = image_a.astype("float64")
    b = image_b.astype("float64")
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a_sq = a.var()
    sigma_b_sq = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(numerator / denominator)


def _edge_map(image: Any) -> Any:
    np, _ = _require_numpy_pillow()
    arr = image.astype("float32")
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    threshold = max(float(mag.mean() + mag.std()), 1e-6)
    return mag > threshold


def _dilate_3x3(mask: Any) -> Any:
    np, _ = _require_numpy_pillow()
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for row in range(3):
        for col in range(3):
            out |= padded[row : row + mask.shape[0], col : col + mask.shape[1]]
    return out


def compute_edge_preservation(source: Any, generated: Any) -> float:
    source_edges = _edge_map(source)
    generated_edges = _edge_map(generated)
    if source_edges.sum() == 0:
        return 1.0
    generated_dilated = _dilate_3x3(generated_edges)
    return float((source_edges & generated_dilated).sum()) / float(source_edges.sum())


def compute_lpips_distance(source: Any, generated: Any) -> float:
    try:
        import lpips
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS requires optional generator QC dependencies. "
            "Install lpips and torch from requirements-generator.txt."
        ) from exc

    np, _ = _require_numpy_pillow()
    scorer = lpips.LPIPS(net="alex", verbose=False)
    src = torch.from_numpy(np.stack([source] * 3)).unsqueeze(0).float() * 2.0 - 1.0
    gen = torch.from_numpy(np.stack([generated] * 3)).unsqueeze(0).float() * 2.0 - 1.0
    with torch.no_grad():
        return float(scorer(src, gen).item())


def compute_qc(source_path: Path, generated_path: Path, *, include_lpips: bool = False) -> dict[str, float | None]:
    np, Image = _require_numpy_pillow()
    source = np.asarray(Image.open(source_path).convert("L"), dtype="float32") / 255.0
    generated = np.asarray(
        Image.open(generated_path).convert("L").resize((source.shape[1], source.shape[0])),
        dtype="float32",
    ) / 255.0
    result: dict[str, float | None] = {
        "ssim": compute_ssim(source, generated),
        "edge_preservation": compute_edge_preservation(source, generated),
        "lpips": None,
    }
    if include_lpips:
        result["lpips"] = compute_lpips_distance(source, generated)
    return result


def passes_qc(metrics: dict[str, float | None], thresholds: QCThresholds | None = None) -> bool:
    thresholds = thresholds or QCThresholds()
    lpips_value = metrics.get("lpips")
    lpips_ok = lpips_value is not None and float(lpips_value) <= thresholds.lpips_max
    return (
        float(metrics["ssim"] or 0.0) >= thresholds.ssim_min
        and float(metrics["edge_preservation"] or 0.0) >= thresholds.edge_preservation_min
        and lpips_ok
    )
