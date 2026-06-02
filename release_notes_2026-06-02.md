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

## Not Included

- Raw VinDr-SpineXR or BUU-LSPINE radiographs.
- Generator checkpoint/LoRA weights.
- Runnable training code or raw training data.
- TSXR/TotalSegmentator binaries, provider credentials, provider logs, or
  private run roots.

The benchmark reviewer verification path remains standard-library only.
