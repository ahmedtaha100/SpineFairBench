# SpineFairBench Counterfactual Generator

This package provides reviewer-facing generator inference code, locked prompt
templates, mask-blending utilities, QC utilities, and smoke-test scripts.

It does not include raw VinDr-SpineXR or BUU-LSPINE radiographs, provider
credentials, private run roots, generator training data, or generator
checkpoint/LoRA weights in git. Reviewer checkpoint assets are released
separately on Hugging Face at `anon-submission7979/spinefairbench-generator`.

## Released

- SD v1.5 img2img inference adapter with optional local LoRA loading.
- Locked demographic prompt templates:
  - `Lumbar spine X-ray of a 75-year-old female patient`
  - `Lumbar spine X-ray of a 75-year-old male patient`
  - `Lumbar spine X-ray of a 25-year-old female patient`
  - `Lumbar spine X-ray of a 25-year-old male patient`
- Inference defaults: 50 steps, guidance scale 5.0, strength 0.15, seed 42,
  LoRA rank 64 / alpha 128 metadata, and TSXR mask blend 0.7.
- Mask-blending utility for user-supplied binary spine masks.
- QC utility thresholds: SSIM >= 0.70, edge preservation >= 0.276 with 3x3
  dilation, LPIPS <= 0.40 when optional LPIPS dependencies are installed.
- Standard-library dry-run release verifier and smoke-test metadata writer.

## Not Released

- Generator checkpoint/LoRA weights in this git repository.
- Generator training pipeline and raw training radiographs.
- TSXR/TotalSegmentator binaries or trained segmentation weights.
- Provider-client orchestration, provider logs, credentials, or private paths.

Use the separately released LoRA-only safetensors checkpoint for reviewer
inspection or optional local inference attempts. The same HF model repo also
contains `latest.pt`, an archival full PyTorch training checkpoint that includes
training state and is not the recommended inference artifact.

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

Real inference requires a user-supplied source image and local checkpoint. The
released LoRA is sufficient for reviewer inspection and optional inference
attempts; exact reproduction of the submitted benchmark image set may also
require the original upstream radiographs, source masks, original runtime
conditions, and/or the archival `latest.pt` training checkpoint.
