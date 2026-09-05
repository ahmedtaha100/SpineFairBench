# SpineFairBench

SpineFairBench audits demographic sensitivity in vision-language models that
generate spinal-radiology reports. It pairs a source radiograph with synthetic
edits conditioned on four age/sex targets. The frozen nine-model panel shows
recommendation changes in every model.

The primary endpoints are whole-report recommendation-category set inequality
(seven categories) and diagnostic-category Jaccard overlap (13 categories).
These lexical measurements use different scales. They do not adjudicate clinical
appropriateness or isolate a causal demographic effect. The radiologist study
assessed plausibility and pathology preservation in a stratified sample; it did
not validate perceived target age or sex.

## Verify retained results

Use Python 3.11 or newer. Frozen point-estimate, count, and checksum checks use
only the standard library. Follow the [reviewer quickstart](reviewer_quickstart.md)
for a revision-pinned artifact download and its access boundaries.

After placing the verified bundle at artifacts/, run from this repository:

~~~sh
python reviewer_verify.py checksums
python reviewer_verify.py checksums artifacts/SHA256SUMS.txt
python reviewer_verify.py table2
python reviewer_verify.py dataset
python reviewer_verify.py stage1-confidence
python reviewer_verify.py mitigation
python reviewer_verify.py radiologist
~~~

The table2 command verifies all nine models against [paper_results.json](paper_results.json),
an archived manuscript derivation, including point estimates, usable pairs,
and full/partial refusal counts. It reads the frozen confidence intervals unless
--recompute-ci is explicitly supplied. No model calls or image generation occur.
Use --model gpt-5.4 to check one row and --artifacts PATH for a bundle elsewhere.

All 18 primary point estimates agree with the latest preprint, but six of its
18 confidence intervals differ from this archived derivation. Interval provenance
remains unresolved; the following command verifies the archived record, not yet
the latest paper's intervals. The bundle's common_core_1000_summary.json contains
a different bootstrap record. Optional interval replay requires pinned NumPy:

~~~sh
python -m pip install -r requirements.txt
python reviewer_verify.py table2 --recompute-ci
~~~

This follows the archived manuscript producer: 10,000 source-clustered resamples,
NumPy PCG64 with int32 indices, base seed 20260426, and offsets +11 for
recommendations and +23 for diagnoses. The command fails if any interval differs
by more than 1e-12. It recomputes summaries from saved reports, not model inference.

## Score a new model

Obtain source radiographs under their original dataset terms. Run the model over
the selected source/edit pairs with the primary prompt in
[prompts/canonical_definitions.json](prompts/canonical_definitions.json).
Write reports in the released artifacts/metrics/submission_schema.json format.
Provider and local-model inference are supplied by the participant.

~~~sh
python -m pip install -r requirements.txt
python -m spinefairbench.release.scoring score --artifacts artifacts --submission artifacts/metrics/toy_submission.json --output toy_score.json --bootstrap-iterations 1000
~~~

The five-pair toy fixture yields recommendation change 0.400 [0.000, 0.500]
and diagnostic consistency 1.000 [1.000, 1.000]. For a full benchmark submission,
use scope "common-core-1000", 10,000 bootstrap iterations, and omit --allow-partial.
The default base seed and endpoint offsets match the manuscript verifier.
The scorer rejects incomplete coverage and records refusals, source-clustered
intervals, coverage, and per-pair scores. Full refusals are excluded; partial
refusals remain. Resolve API errors before submitting reports; do not convert
them into clinical text.

## Interpretation and reproduction limits

- The 1,000-source core has 3,998 possible pairs before model-specific exclusions:
  819 BUU-LSPINE and 181 VinDr-SpineXR sources. Of these, 906 have no-finding or
  unascertained labels and 94 have finding labels.
- Scorers match keywords across whole reports, including negated mentions.
  Both-empty diagnostic sets score 1.0. Changing these frozen definitions
  creates a different analysis.
- Source-image calls were repeated for different edits. Their stochastic
  variation and generation artifacts limit demographic attribution.
- The reader pass count, 443/450, applies to a stratified post-QC sample.
  Edited-side detectability was "Cannot tell" for 1,307/1,350 responses.
- Findings-first Condition B failed for GPT-5.4 and GLM-4.6V, the two
  gate-eligible models. This does not localize a perceptual mechanism.
- Historical inference and image generation cannot be recreated exactly from
  this release: provider snapshots, training membership, source data, and masks
  are not all available in the public repository.

## Optional generator inspection

The [generator adapter](spinefairbench/generator/README.md) is an illustrative
Diffusers pipeline with a final pixel-space mask blend. Production used per-step
latent mask blending and CLIP-guided latent drift. The adapter does not reproduce
the released images or production QC. Use the fixed images for benchmark scoring.
QC cannot pass without all three measurements, including LPIPS.

## Development

~~~sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

Four small regression tests cover the archived bootstrap, frozen-result checks,
checksum failures, and missing-LPIPS QC. They use synthetic fixtures and make
no model calls. Optional mitigation analysis requires requirements-analysis.txt;
generator dependencies are in requirements-generator.txt.

The metrics/ and evaluation/ modules retain the original endpoint, prompt, and
Stage-1 parser definitions for inspection. The optional analysis/mitigation.py
retains the mitigation analysis. Unused private-analysis and packaging scaffolds
are available in Git history.

Code uses the [MIT license](LICENSE); documentation uses [LICENSE-DOCS](LICENSE-DOCS).
Data and derived artifacts retain their own terms. The code license does not
grant rights to source radiographs or reader records.
