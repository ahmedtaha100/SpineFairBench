# Release Notes — 2026-06-01

This release-side correction pass updates reviewer-facing code, documentation,
metadata terminology, and checksum manifests. It does not modify the frozen
manuscript or recompute scientific endpoint summaries.

## Corrections

- Current public artifact URL is `ahmedtaha100/spinefairbench-artifacts`.
  Historical anonymous-review archives may still mention
  `anon-submission7979/spinefairbench-artifacts`.
- Repository and artifact metadata now distinguish `filtered_source_count =
  2987`, `qc_passed_source_count = 2950`, `failed_qc_only_source_count = 37`,
  and `qc_passed_pair_count = 11795`.
- Frozen Table 2 diagnostic-label consistency is documented as
  `extract_labels()` plus `compute_jaccard()` over the released 13-category
  diagnostic synonym registry. The tokenized diagnosis helper in
  `spinefairbench.analysis.endpoints` is archival and does not generate frozen
  Table 2.
- `reviewer_verify.py table2` now labels frozen CIs as read from the frozen
  summary and supports optional `--recompute-ci` for source-clustered primary
  endpoint CIs from released per-pair outputs.
- `reviewer_verify.py gap-sensitivity` recomputes the exact-vs-graded stability
  gap sensitivity. Released retained-panel summaries are median `gap_exact =
  0.261611`, median `gap_graded = -0.113738`, and `gap_graded < 0` for `8/9`
  models.
- `reviewer_verify.py both-empty-diagnostic` reports both-empty diagnostic-label
  pairs. Released pooled count is `509/34146 (1.4907%)`, scored as Jaccard
  `1.0` by frozen benchmark definition.
- New-submission scoring now uses the requested bootstrap seed for both
  recommendation-change and diagnostic-label-consistency CIs and records
  endpoint-specific seed fields in score JSON.
- Public release-scope docs explicitly state that the generator checkpoint,
  generator training/inference code, provider-client orchestration, raw source
  radiographs, provider credentials, and private run roots are not released.
- Verifier docs now state that secondary/exploratory Table 2 fields such as
  severity, confidence, and hallucination remain frozen-summary fields unless a
  dedicated regeneration command is added later.

## Verification

Run the reviewer quickstart after downloading the companion artifacts. The
final release pass regenerated repository checksums after all source edits.
