# SpineFairBench Illustrative Generator

This package provides an illustrative Diffusers adapter, locked prompt templates,
mask-blending utilities, QC utilities, and smoke-test scripts. It uses stock
img2img followed by pixel-space blending. Production blended source latents at
each denoising step and applied CLIP-guided latent drift outside the mask.
This adapter does not reproduce the released benchmark images or production QC.

It does not include raw VinDr-SpineXR or BUU-LSPINE radiographs, provider
credentials, private run roots, generator training data, or generator
checkpoint/LoRA weights in git. Reviewer checkpoint assets are released
separately on Hugging Face at `anon-submission7979/spinefairbench-generator`.

## Released

- SD v1.5 img2img inference adapter with local LoRA loading. The released PEFT
  checkpoint uses its recorded rank and alpha; Diffusers-format LoRAs are also
  supported. The released checkpoint was checked on an RTX 5090: all 256 tensors
  loaded across 128 modules at alpha/rank = 2.0, followed by synthetic inference.
- Locked demographic prompt templates:
  - `Lumbar spine X-ray of a 75-year-old female patient`
  - `Lumbar spine X-ray of a 75-year-old male patient`
  - `Lumbar spine X-ray of a 25-year-old female patient`
  - `Lumbar spine X-ray of a 25-year-old male patient`
- Inference defaults: 50 steps, guidance scale 5.0, strength 0.15, seed 42,
  LoRA rank 64 / alpha 128 metadata, and TSXR mask blend 0.7.
- Mask-blending utility for user-supplied binary spine masks.
- QC utility thresholds: SSIM >= 0.70, edge preservation >= 0.276 with 3x3
  dilation, LPIPS <= 0.40. All three measurements are required to pass.
  The adapter's gradient-based edge calculation differs from production Canny
  edges; matching threshold values does not establish production parity.
- Standard-library synthetic smoke-test input and metadata writer.

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

Frozen point-estimate and checksum verification uses the standard library.
Generator dependencies are separate:

```bash
python -m pip install -r requirements-generator.txt
```

## Dry Run

```bash
python scripts/run_generator_smoke_test.py --dry-run --output generator_smoke
```

The dry run creates a synthetic non-clinical test image and JSON metadata. It
does not run SD inference and does not require a checkpoint.

## Real Inference

```bash
python -m spinefairbench.generator.infer \
  --input /path/to/user_supplied_source.png \
  --output /tmp/spinefairbench_generator \
  --checkpoint /path/to/local_lora.safetensors \
  --config spinefairbench/generator/configs/inference_sd15_lora.yaml \
  --demographic elderly_female \
  --seed 42 \
  --device cuda
```

Real inference requires a user-supplied source image and local checkpoint.
Use the fixed released images for benchmark scoring. Loading the released LoRA
or archival training checkpoint does not make this adapter production-faithful;
the production inference path, original source images, masks, and runtime are
also needed for historical regeneration.
