# SpineFairBench

Reviewer-facing code for SpineFairBench, a counterfactual benchmark for auditing
demographic sensitivity in vision-language spine-radiology reports.

This repository contains the reviewer-facing verification and scoring code. The
anonymous reviewer artifact bundle is hosted separately:

```text
https://huggingface.co/datasets/anon-submission7979/spinefairbench-artifacts
```

The de-anonymized public mirror is
`https://huggingface.co/datasets/ahmedtaha100/spinefairbench-artifacts`. The
anonymous and public HF repositories are separately checksummed release builds;
use the repository named by your review/manuscript context and verify the
tarball against the `.sha256` file downloaded from the same HF repository.

## Requirements

- Python 3.11 or newer.
- The reviewer verification path is standard-library only.
- The Hugging Face `hf` CLI is used only to download the reviewer artifact bundle.
- `requirements-analysis.txt` lists optional packages for archival analysis
  helpers that are not needed for the reviewer reproduction path.

## Repository Layout

After downloading and extracting the artifact bundle, the repository root should
look like this:

```text
SpineFairBench/
|-- README.md
|-- reviewer_verify.py
|-- scripts/
|-- spinefairbench/
|-- prompts/
`-- artifacts/
```

The `artifacts/` directory is intentionally not tracked by git.

## Reproduce The Released Reviewer Checks

Clone the repository, download the artifact bundle into the repository root, and
verify the tarball checksum:

```bash
hf download anon-submission7979/spinefairbench-artifacts \
  spinefairbench_artifacts.tar.gz \
  spinefairbench_artifacts.tar.gz.sha256 \
  --repo-type dataset \
  --local-dir .
shasum -a 256 -c spinefairbench_artifacts.tar.gz.sha256
tar -xzf spinefairbench_artifacts.tar.gz
```

Run the reviewer verification commands from the repository root:

```bash
export PYTHONDONTWRITEBYTECODE=1
shasum -a 256 -c SHA256SUMS.txt
python3 reviewer_verify.py inspect --artifacts artifacts
python3 reviewer_verify.py dataset --artifacts artifacts
python3 reviewer_verify.py stage1-confidence --artifacts artifacts
python3 reviewer_verify.py mitigation --artifacts artifacts
python3 reviewer_verify.py parse-sample --artifacts artifacts --model gpt-5.4
python3 reviewer_verify.py parse-sample --artifacts artifacts --model llama-4-scout
python3 reviewer_verify.py diagnostic-scoring
python3 reviewer_verify.py table2 --artifacts artifacts --model gpt-5.4
python3 reviewer_verify.py table2 --artifacts artifacts --model qwen2.5-vl
python3 reviewer_verify.py gap-sensitivity --artifacts artifacts
python3 reviewer_verify.py both-empty-diagnostic --artifacts artifacts
python3 reviewer_verify.py radiologist --artifacts artifacts
```

Expected smoke-check values:

- `gpt-5.4`: `3998` usable pairs, recommendation change `0.694` with 95% CI
  `[0.678, 0.710]`, diagnostic-label consistency `0.649` with 95% CI
  `[0.640, 0.657]`.
- `qwen2.5-vl`: `3998` usable pairs, recommendation change `0.293` with 95%
  CI `[0.271, 0.318]`, diagnostic-label consistency `0.545` with 95% CI
  `[0.525, 0.565]`.
- `mitigation`: `gpt-5.4` and `glm-4.6v` pass the Stage-1 inclusion gate,
  condition B fails the binding rule for both, and `claude-opus-4-6`,
  `claude-sonnet-4-6`, and `kimi-k2.5` are excluded by the Stage-1 gate.
- `radiologist`: `443/450` passing pairs, `7` excluded pairs, `1307/1350`
  detectability responses marked `Cannot tell`, `1380` per-reviewer display
  events, and `30` hidden-repeat records.

To verify every retained Table 2 row:

```bash
for model in gpt-5.4 claude-sonnet-4-6 claude-opus-4-6 glm-4.6v kimi-k2.5 \
  gemma-4 llama-4-scout qwen2.5-vl radfm; do
  python3 reviewer_verify.py table2 --artifacts artifacts --model "$model"
done
```

The `diagnostic-scoring` command prints the exact frozen diagnostic-label
scoring path. Frozen Table 2 uses
`spinefairbench.metrics.diagnostic_label.extract_labels()` followed by
`compute_jaccard()` over the released 13-category synonym registry. The
frozen Table 2 diagnostic-label consistency is registry-label Jaccard, not
free-token Jaccard. The tokenized diagnosis helper in
`spinefairbench.analysis.endpoints` is archival analysis code and does not
generate the frozen Table 2 diagnostic-label consistency values.

The `table2` command recomputes primary endpoint point estimates from released
retained outputs and checks them against the frozen summary. By default it reads
the frozen source-clustered 95% CIs from
`artifacts/Results/analysis/common_core_1000_summary.json`; pass
`--recompute-ci` to regenerate source-clustered percentile bootstrap CIs from
the released per-pair outputs. It follows the frozen accounting policy:
full-refusal pairs are excluded from primary endpoints, and partial-refusal
pairs are retained. The older `table3` command remains as a backward-compatible
alias for `table2`; the manuscript mitigation table is verified by the
`mitigation` command.

`gap-sensitivity` recomputes the stability-gap sensitivity from released
per-pair outputs. The manuscript headline gap is
`gap_exact = mean diagnostic-label Jaccard - (1 - exact recommendation change
rate)`. The matched-granularity sensitivity is
`gap_graded = mean diagnostic-label Jaccard - mean recommendation-category
Jaccard`; on the released retained panel this gives median `gap_exact =
0.261611`, median `gap_graded = -0.113738`, and `gap_graded < 0` for `8/9`
models.

`both-empty-diagnostic` reports pairs where both reports contain no matched
released diagnostic labels. The pooled released count is `509/34146 = 1.4907%`;
those pairs score diagnostic-label Jaccard `1.0` by frozen benchmark
definition.

`stage1-confidence` uses an explicit released 200-file sample list from the
artifact bundle. It accepts either the standalone confidence-sample manifest or
the sample list embedded in the Stage-1 trace manifest.

## Verify Artifact Checksums

From the repository root:

```bash
cd artifacts
shasum -a 256 -c SHA256SUMS.txt
shasum -a 256 -c radiologist_validation_SHA256SUMS.txt
cd ..
```

Both checksum manifests should report only `OK` entries.

## Counterfactual Generator Release

This repository now includes optional reviewer-facing counterfactual generator
code under `spinefairbench/generator/`. The generator release is separate from
the frozen benchmark scoring/reviewer artifacts.

Released in git:

- SD v1.5 img2img inference adapter with optional local LoRA loading.
- Locked demographic prompt templates:
  - `Lumbar spine X-ray of a 75-year-old female patient`
  - `Lumbar spine X-ray of a 75-year-old male patient`
  - `Lumbar spine X-ray of a 25-year-old female patient`
  - `Lumbar spine X-ray of a 25-year-old male patient`
- Inference/config defaults matching the release documentation: 50 steps,
  guidance scale 5.0, strength 0.15, seed 42, LoRA rank 64 / alpha 128
  metadata, and TSXR mask blend 0.7 toward the source in masked spine regions.
- Mask-blending and QC utilities. QC thresholds are SSIM >= 0.70, edge
  preservation >= 0.276 with 3x3 dilation, and LPIPS <= 0.40 when optional
  LPIPS dependencies are installed.
- Dry-run smoke scripts and config templates:
  - `scripts/verify_generator_release.py`
  - `scripts/run_generator_smoke_test.py`
  - `spinefairbench/generator/configs/inference_sd15_lora.yaml`
  - `spinefairbench/generator/configs/training_lora.yaml`

Not released:

- Raw VinDr-SpineXR or BUU-LSPINE source radiographs.
- Source masks.
- Runnable generator training code or raw training data.
- TSXR/TotalSegmentator binaries or segmentation weights.
- Provider-client orchestration, provider logs, credentials, private run roots,
  or local machine paths.

Install generator dependencies only when you want to inspect/run generator
inference. Do not install them for the standard reviewer verification path:

```bash
python3 -m pip install -r requirements-generator.txt
```

Standard-library generator dry run:

```bash
python3 scripts/verify_generator_release.py --dry-run
python3 scripts/run_generator_smoke_test.py --dry-run --output /tmp/spinefairbench_generator_smoke
```

Recommended generator checkpoint inspection assets are hosted separately in the
anonymous HF model repo:

```bash
hf download anon-submission7979/spinefairbench-generator \
  spinefairbench_sd15_lora.safetensors \
  generator_config.yaml \
  SHA256SUMS.recommended.txt \
  --repo-type model \
  --local-dir generator_assets
cd generator_assets
shasum -a 256 -c SHA256SUMS.recommended.txt
```

The same repo includes `latest.pt` as an optional archival PyTorch training
checkpoint. Use `spinefairbench_sd15_lora.safetensors` for reviewer inference
inspection; `latest.pt` contains training state and is not the recommended
inference artifact. To verify all generator files, including `latest.pt`,
download the full model repository and run `shasum -a 256 -c SHA256SUMS.txt`.

Real inference requires a user-supplied source image and an independently
authorized local Diffusers-compatible LoRA checkpoint:

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

The released counterfactual images in the HF artifact bundle are already fixed
for benchmark evaluation. Running the generator is for methodological
inspection or independently governed regeneration attempts. Exact reproduction
may require the original upstream radiographs, source masks, original
dependency/GPU/runtime details, and/or the archival `latest.pt` training
checkpoint.

Generated images are research/evaluation artifacts only. They are not clinical
images and should not be used for diagnosis or clinical-system training without
appropriate governance.

## Score A New Model Submission

SpineFairBench does not run provider or local model inference for reviewers.
Run your model externally over a selected released pair scope, write the
free-text source and counterfactual reports in the JSON format documented by
`artifacts/metrics/submission_schema.json`, then score the submission:

```bash
python3 -m spinefairbench.release.scoring score \
  --artifacts artifacts \
  --submission artifacts/metrics/toy_submission.json \
  --output /tmp/spinefairbench_toy_score.json \
  --bootstrap-iterations 1000
```

The toy submission is a five-pair smoke test. Its expected primary values are
recommendation change `0.400` with 95% CI `[0.000, 0.500]` and
diagnostic-label consistency `1.000` with 95% CI `[1.000, 1.000]`. The scorer
uses source-clustered percentile bootstrap CIs and records the endpoint-specific
seeds in the score JSON as `recommendation_bootstrap_seed` and
`diagnostic_bootstrap_seed`.

For a comparable benchmark run, create a full submission with
`scope: "common-core-1000"` and leave `--allow-partial` unset. The scorer will
require coverage of the selected scope and will report coverage, refusal
accounting, primary endpoints, source-clustered bootstrap CIs, and per-pair
results.

## Public Metadata Notes

`prompts/canonical_definitions.json` is provenance metadata for the frozen
prompt registry and execution roster. The retained public evaluation panels are
defined by the released panel manifests under
`artifacts/artifacts/Results/final_inputs/panels/`, including
`full_pipeline_retained` and `baseline_only_retained`.

Source-count terminology is split explicitly in current release metadata:
`filtered_source_count = 2987` source studies passed source-side filtering,
`qc_passed_source_count = 2950` source studies have at least one QC-passed
released counterfactual, `failed_qc_only_source_count = 37`, and
`qc_passed_pair_count = 11795`.

The released diagnostic and recommendation parsers are deterministic keyword
classifiers used to reproduce the frozen endpoint values. They intentionally do
not model negation or nested clinical concepts, and diagnostic-label Jaccard is
defined as `1.0` when both reports have no matched released label. These choices
are part of the frozen benchmark definition and should not be changed without
recomputing the released endpoint summaries.

The recommendation parser keeps conservative-management wording in the
`no_action` bucket. The frozen diagnostic synonym registry also preserves broad
fracture aliases (`fx`, `break`, `broken`) from the released Table 2 parser;
`extract_labels_strict()` is available for manual audits, but benchmark scoring
uses the frozen registry.

The frozen Table 2 confidence intervals are read from
`artifacts/artifacts/Results/analysis/common_core_1000_summary.json`. Those
intervals were generated with source-clustered percentile bootstrap confidence
intervals using 10,000 iterations and seed `42`; `reviewer_verify.py table2`
recomputes point estimates and checks them against that frozen summary. With
`--recompute-ci`, the verifier regenerates primary endpoint CIs from released
per-pair outputs; secondary/exploratory Table 2 fields such as severity,
confidence, and hallucination are retained in the frozen summary and are not
fully regenerated by the quickstart verifier.

## Release Scope

This public repository and artifact bundle support benchmark evaluation,
scoring, checksum validation, primary endpoint recomputation, radiologist
validation checks, reviewer-facing sensitivity checks, and optional
counterfactual generator inference inspection. Generator checkpoint inspection
assets are released separately at
`anon-submission7979/spinefairbench-generator`. This public repository and the
artifact bundle do not include runnable generator training code,
provider-client orchestration, raw source radiographs, source masks, provider
credentials, or private run roots. Generator training configuration is released
as documentation/provenance only.

## License And Anonymity

The code is released under the MIT license. Documentation and artifact metadata
are released under the accompanying documentation license where provided.

Current reviewer quickstart commands use the anonymous artifact handle
`anon-submission7979/spinefairbench-artifacts`.
