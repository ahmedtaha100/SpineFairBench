# Reviewer Quickstart

Run these commands from the repository root after downloading and extracting the
artifact bundle so that `artifacts/` is present next to `reviewer_verify.py`.

The reviewer verification path uses only the Python standard library. Python
3.11 or newer is required.

## 1. Download And Verify The Artifact Bundle

```bash
hf download anon-submission7979/spinefairbench-artifacts \
  spinefairbench_artifacts.tar.gz \
  spinefairbench_artifacts.tar.gz.sha256 \
  --repo-type dataset \
  --local-dir .
shasum -a 256 -c spinefairbench_artifacts.tar.gz.sha256
tar -xzf spinefairbench_artifacts.tar.gz
```

## 2. Verify Code Checksums

```bash
export PYTHONDONTWRITEBYTECODE=1
shasum -a 256 -c SHA256SUMS.txt
```

## 3. Inspect Frozen Artifacts And Locked Prompts

```bash
python3 reviewer_verify.py inspect --artifacts artifacts
python3 reviewer_verify.py dataset --artifacts artifacts
python3 reviewer_verify.py stage1-confidence --artifacts artifacts
python3 reviewer_verify.py mitigation --artifacts artifacts
```

Expected outcome: the commands report the frozen pair/source manifests, retained
panel manifests, locked primary prompt excerpt, dropped-model exclusion
evidence, benchmark metadata, public radiologist-validation artifacts, and the
public counterfactual-image/QC release counts. The Stage-1 command recomputes
the mitigation parsing-confidence gates from released Stage-1 text and
validation trace files using the explicit released 200-file sample list for each
model.

The mitigation command verifies the retained mitigation Table 3 rows for
`gpt-5.4` and `glm-4.6v`, checks their Stage-1 inclusion gates, and confirms the
binding rule result (`b_rule_pass: False`) plus the full exclusions for
`claude-opus-4-6`, `claude-sonnet-4-6`, and `kimi-k2.5`.

## 4. Run Parsers On Included Pairs

```bash
python3 reviewer_verify.py parse-sample --artifacts artifacts --model gpt-5.4
python3 reviewer_verify.py parse-sample --artifacts artifacts --model llama-4-scout
```

Expected outcome: each command loads one matched source/counterfactual output
pair and prints recommendation categories, diagnostic labels, and
diagnostic-label Jaccard overlap. The `llama-4-scout` command is a
baseline-only smoke test and confirms that the verifier routes baseline-only
models through `baseline_only_retained`.

## 5. Reproduce Endpoint-Summary Rows

```bash
python3 reviewer_verify.py diagnostic-scoring
python3 reviewer_verify.py table2 --artifacts artifacts --model gpt-5.4
python3 reviewer_verify.py table2 --artifacts artifacts --model qwen2.5-vl
```

Expected retained rounded values for `gpt-5.4`:

- usable pairs: `3998`
- recommendation change rate: `0.694`
- recommendation change 95% CI: `[0.678, 0.710]`
- diagnostic-label consistency: `0.649`
- diagnostic-label consistency 95% CI: `[0.640, 0.657]`

Expected retained rounded values for `qwen2.5-vl`:

- usable pairs: `3998`
- recommendation change rate: `0.293`
- recommendation change 95% CI: `[0.271, 0.318]`
- diagnostic-label consistency: `0.545`
- diagnostic-label consistency 95% CI: `[0.525, 0.565]`

To check all retained models:

```bash
for model in gpt-5.4 claude-sonnet-4-6 claude-opus-4-6 glm-4.6v kimi-k2.5 \
  gemma-4 llama-4-scout qwen2.5-vl radfm; do
  python3 reviewer_verify.py table2 --artifacts artifacts --model "$model"
done
```

The command recomputes point estimates from retained outputs and reads the
frozen source-clustered CI values from
`artifacts/artifacts/Results/analysis/common_core_1000_summary.json`. It
excludes full-refusal pairs and retains partial-refusal pairs, matching the
frozen accounting policy. The older `table3` command remains as a
backward-compatible alias for `table2`.

To regenerate source-clustered primary endpoint CIs from the released per-pair
outputs, add `--recompute-ci`. The default quickstart mode is faster and labels
the frozen CIs as read from the frozen summary.

The diagnostic-label path used for frozen Table 2 is
`extract_labels()` plus `compute_jaccard()` over the released 13-category
synonym registry. Frozen Table 2 diagnostic-label consistency is registry-label
Jaccard, not free-token Jaccard. The tokenized diagnosis helper in
`spinefairbench.analysis.endpoints` is archival analysis code, not the frozen
Table 2 diagnostic scorer.

## 5a. Run Sensitivity And Parser-Definition Checks

```bash
python3 reviewer_verify.py gap-sensitivity --artifacts artifacts
python3 reviewer_verify.py both-empty-diagnostic --artifacts artifacts
```

Expected summary values:

- `gap-sensitivity`: median `gap_exact = 0.261611`, median `gap_graded =
  -0.113738`, and `gap_graded < 0` for `8/9` retained models.
- `both-empty-diagnostic`: pooled both-empty diagnostic-label pairs
  `509/34146 (1.4907%)`; these score diagnostic-label Jaccard `1.0` by frozen
  benchmark definition.

## 6. Check Public Radiologist-Validation Counts

```bash
python3 reviewer_verify.py radiologist --artifacts artifacts
```

Expected outcome: `443/450` passing validation pairs under the 2-of-3 rule,
`7` excluded pairs, and `1307/1350` detectability responses marked `Cannot
tell`. The command also checks `1350` detectability responses, `1380`
per-reviewer display events, and `30` hidden-repeat records.

## 7. Verify Artifact Checksums

```bash
cd artifacts
shasum -a 256 -c SHA256SUMS.txt
shasum -a 256 -c radiologist_validation_SHA256SUMS.txt
cd ..
```

Both checksum manifests should report only `OK` entries.

## 8. Score A New-Model Submission

SpineFairBench does not run model inference. Run your model externally, write
one JSON result object per scored pair, then run the submission scorer. The
required JSON format is in `artifacts/metrics/submission_schema.json`, with a
starter template at `artifacts/metrics/submission_template.json`.

Smoke-test the scorer with the bundled toy submission:

```bash
python3 -m spinefairbench.release.scoring score \
  --artifacts artifacts \
  --submission artifacts/metrics/toy_submission.json \
  --output /tmp/spinefairbench_toy_score.json \
  --bootstrap-iterations 1000
```

Expected primary values are recommendation change `0.400` with 95% CI
`[0.000, 0.500]` and diagnostic-label consistency `1.000` with 95% CI
`[1.000, 1.000]`. The toy score is a smoke test, not a benchmark-comparable
result. The score JSON records both `recommendation_bootstrap_seed` and
`diagnostic_bootstrap_seed`; current scoring uses the requested `--seed` for
both primary endpoint CIs.

For a comparable new-model run, fill the submission template with all pair IDs
for `scope: "common-core-1000"`. Leave `--allow-partial` unset; missing pairs
are an error for comparable scoring.

## 9. Parser And Bootstrap Notes

The released diagnostic and recommendation parsers are deterministic keyword
classifiers. They do not model negation or nested clinical concepts, and
diagnostic-label Jaccard is defined as `1.0` when both reports have no matched
released label. These are frozen benchmark-definition choices.

The frozen Table 2 confidence intervals were generated with source-clustered
percentile bootstrap confidence intervals using 10,000 iterations and seed
`42`. The new-submission scorer uses
`spinefairbench.release.scoring.source_clustered_bootstrap_ci`; archival
analysis helpers delegate to the same implementation where applicable.

## 10. Optional Counterfactual Generator Inspection

The standard reviewer verification path above remains standard-library only.
The optional generator release is for methodology inspection and separately
governed regeneration attempts; it is not needed to reproduce frozen benchmark
scores.

Released in git:

- generator inference adapter and config templates under
  `spinefairbench/generator/`;
- locked demographic prompt templates;
- mask-blending and QC utilities;
- dry-run scripts under `scripts/`.

Not released:

- raw VinDr-SpineXR or BUU-LSPINE source radiographs;
- source masks;
- runnable training code or raw training data;
- provider credentials, provider logs, private run roots, or local paths.

Recommended generator checkpoint inspection assets are hosted separately at
`anon-submission7979/spinefairbench-generator`:

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

The LoRA-only `spinefairbench_sd15_lora.safetensors` file is the recommended
checkpoint for reviewer inference inspection. The same HF model repo includes
`latest.pt` as an optional archival PyTorch training checkpoint; it contains
training state and is not the recommended inference artifact. To verify all
generator files, including `latest.pt`, download the full model repository and
run `shasum -a 256 -c SHA256SUMS.txt`.

Run the generator dry-run checks without heavy dependencies:

```bash
python3 scripts/verify_generator_release.py --dry-run
python3 scripts/run_generator_smoke_test.py --dry-run --output /tmp/spinefairbench_generator_smoke
```

For real inference, install optional dependencies with
`python3 -m pip install -r requirements-generator.txt`, then provide your own
source image and a local LoRA checkpoint such as the recommended safetensors
download above:

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

The released HF counterfactual images remain the fixed benchmark images.
Generator execution is not required for benchmark scoring, and exact
regeneration may require the original upstream radiographs, source masks,
original dependency/GPU/runtime details, and/or the archival `latest.pt`
training checkpoint.
