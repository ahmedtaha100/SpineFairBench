# SpineFairBench

Reviewer-facing code for SpineFairBench, a counterfactual benchmark for auditing
demographic sensitivity in vision-language spine-radiology reports.

This repository contains the public verification and scoring code. The released
artifact bundle is hosted separately:

```text
https://huggingface.co/datasets/anon-submission7979/spinefairbench-artifacts
```

## Requirements

- Python 3.11 or newer.
- The reviewer verification path is standard-library only.
- The Hugging Face `hf` CLI is used only to download the public artifact bundle.
- `requirements-analysis.txt` lists optional packages for archival analysis
  helpers that are not needed for the reviewer reproduction path.

## Repository Layout

After downloading and extracting the artifact bundle, the repository root should
look like this:

```text
SpineFairBench/
|-- README.md
|-- reviewer_verify.py
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
python3 reviewer_verify.py table2 --artifacts artifacts --model gpt-5.4
python3 reviewer_verify.py table2 --artifacts artifacts --model qwen2.5-vl
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

The `table2` command recomputes endpoint point estimates from released retained
outputs and checks them against the frozen summary. It follows the frozen
accounting policy: full-refusal pairs are excluded from primary endpoints, and
partial-refusal pairs are retained. The older `table3` command remains as a
backward-compatible alias for `table2`; the manuscript mitigation table is
verified by the `mitigation` command.

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
diagnostic-label consistency `1.000` with 95% CI `[1.000, 1.000]`.

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
recomputes point estimates and checks them against that frozen summary.

## License And Anonymity

The code is released under the MIT license. Documentation and artifact metadata
are released under the accompanying documentation license where provided.

This repository is prepared for anonymous review. The tracked source tree is not
intended to contain author names, institutional names, private repository URLs,
API keys, local machine paths, or raw source radiographs.
