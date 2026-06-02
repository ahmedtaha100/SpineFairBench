# Release Notes - 2026-06-02

This release adds reviewer-facing counterfactual generator code and
documentation without changing the frozen benchmark artifacts or manuscript.

## Added

- Optional generator package under `spinefairbench/generator/`.
- Locked demographic prompt templates and SD v1.5 + LoRA inference defaults.
- Mask-blending and QC utilities, including the released SSIM, edge-preservation,
  and LPIPS thresholds.
- Generator config templates and standard-library dry-run smoke scripts.
- `requirements-generator.txt` for optional heavy dependencies.
- Anonymous HF generator checkpoint repo:
  `anon-submission7979/spinefairbench-generator`.
- Recommended LoRA-only checkpoint:
  `spinefairbench_sd15_lora.safetensors`, SHA256
  `21bbfa4ee50ff389c3a49edc9fa39f4e1f7c691fad85c34cbcce595f62b8d56c`.
- Optional archival PyTorch training checkpoint:
  `latest.pt`, SHA256
  `bc3f71631f07f3d0ca645808b0622f28da2c863b21622cab0ed65e8e44f329ff`.
- Final anonymous generator repo revision:
  `8dbf77c0a79c2de645b49e93f2cd479953508529`.

## Not Included

- Raw VinDr-SpineXR or BUU-LSPINE radiographs.
- Source masks.
- Runnable training code or raw training data.
- TSXR/TotalSegmentator binaries, provider credentials, provider logs, or
  private run roots.

The benchmark reviewer verification path remains standard-library only.
Generator execution is optional and not required for endpoint verification.
