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
result.

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
