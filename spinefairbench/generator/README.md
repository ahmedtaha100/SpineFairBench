# SpineFairBench Counterfactual Generator

This package provides reviewer-facing generator inference code, locked prompt
templates, mask-blending utilities, QC utilities, and smoke-test scripts.

It does not include raw VinDr-SpineXR or BUU-LSPINE radiographs, provider
credentials, private run roots, generator training data, or generator
checkpoint/LoRA weights.

## Released

- SD v1.5 img2img inference adapter with optional local LoRA loading.
- Locked demographic prompt templates:
  - `75-year-old female lumbar spine radiograph`
  - `75-year-old male lumbar spine radiograph`
  - `25-year-old female lumbar spine radiograph`
  - `25-year-old male lumbar spine radiograph`
- Inference defaults: 50 steps, guidance scale 5.0, strength 0.15, seed 42,
  LoRA rank 64 / alpha 128 metadata, and TSXR mask blend 0.7.
- Mask-blending utility for user-supplied binary spine masks.
- QC utility thresholds: SSIM >= 0.70, edge preservation >= 0.276 with 3x3
  dilation, LPIPS <= 0.40 when optional LPIPS dependencies are installed.
- Standard-library dry-run release verifier and smoke-test metadata writer.

## Not Released

- Generator checkpoint/LoRA weights.
- Generator training pipeline and raw training radiographs.
- TSXR/TotalSegmentator binaries or trained segmentation weights.
- Provider-client orchestration, provider logs, credentials, or private paths.

If you have an independently authorized Diffusers-compatible LoRA checkpoint,
place it outside git and pass it with `--checkpoint`.

## Install Optional Dependencies

The benchmark reviewer path remains standard-library only. Generator
dependencies are optional:

```bash
python3 -m pip install -r requirements-generator.txt
```

## Dry Run

```bash
python3 scripts/verify_generator_release.py --dry-run
python3 scripts/run_generator_smoke_test.py --dry-run --output /tmp/spinefairbench_generator_smoke
```

The dry run creates a synthetic non-clinical test image and JSON metadata. It
does not run SD inference and does not require a checkpoint.

## Real Inference

```bash
python3 -m spinefairbench.generator.infer \
  --input /path/to/user_supplied_source.png \
  --output /tmp/spinefairbench_generator \
  --checkpoint /path/to/local_lora.safetensors \
  --config spinefairbench/generator/configs/inference_sd15_lora.yaml \
  --demographic elderly_female \
  --seed 42 \
  --device cuda
```

Real inference requires a user-supplied source image and local checkpoint. Exact
reproduction of the submitted benchmark image set may also require the original
upstream radiographs, the unreleased production checkpoint, matching dependency
versions, and comparable GPU/runtime behavior.
