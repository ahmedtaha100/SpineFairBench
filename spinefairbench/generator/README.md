# SpineFairBench Illustrative Generator

This package provides an illustrative Diffusers adapter, locked prompt templates,
mask-blending utilities, and QC utilities. It uses stock
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
- Inference defaults: 50 configured scheduler steps, guidance scale 5.0, strength 0.15, seed 42,
  LoRA rank 64 / alpha 128 metadata, and TSXR mask blend 0.7.
- Mask-blending utility for user-supplied binary spine masks.
- QC utility thresholds: SSIM >= 0.70, edge preservation >= 0.276 with 3x3
  dilation, LPIPS <= 0.40. All three measurements are required to pass.
  The adapter's gradient-based edge calculation differs from production Canny
  edges; matching threshold values does not establish production parity.

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

## Download the Checkpoint

Run these commands from the code repository root. The pinned revision includes
the recommended checksum manifest and does not require the full training checkpoint.
Install the hf CLI with `python -m pip install huggingface_hub` if needed.

```sh
hf download anon-submission7979/spinefairbench-generator spinefairbench_sd15_lora.safetensors generator_config.yaml SHA256SUMS.recommended.txt --repo-type model --revision f3ae3af9564b7cae1b93ebfbc2cf2921b155d436 --local-dir generator_assets
python reviewer_verify.py checksums generator_assets/SHA256SUMS.recommended.txt
```

The LoRA SHA-256 is
`21bbfa4ee50ff389c3a49edc9fa39f4e1f7c691fad85c34cbcce595f62b8d56c`.
The downloaded generator_config.yaml records release metadata; use the repository's
inference_sd15_lora.yaml with the inference command below.

## Real Inference

```sh
python -m spinefairbench.generator.infer --input source.png --output generator_output --checkpoint generator_assets/spinefairbench_sd15_lora.safetensors --config spinefairbench/generator/configs/inference_sd15_lora.yaml --demographic elderly_female --seed 42 --device cuda
```

Real inference requires a user-supplied source image and local checkpoint.
Use the fixed released images for benchmark scoring. Loading the released LoRA
or archival training checkpoint does not make this adapter production-faithful;
the production inference path, original source images, masks, and runtime are
also needed for historical regeneration.

## Training Provenance

The recorded training setup used 9,024 radiographs and a single H200, with LoRA
rank 64 and alpha 128. Loss weights were reconstruction L1 = 1.0, PatchGAN =
0.005, KL = 1e-7, and Stage-2 latent-cycle L1 = 1.0. The VAE was frozen, so
the recorded KL term did not update trainable parameters. This is provenance,
not a runnable training configuration. VinDr-SpineXR and BUU-LSPINE data must be
obtained under their original terms.
