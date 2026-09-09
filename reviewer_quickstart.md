# Reviewer quickstart

Use Python 3.11 or newer from the repository root. These commands work in
PowerShell and a POSIX shell and verify existing files and saved reports.

## Artifact identity and access

The historical anonymous bundle is pinned to Hugging Face revision
fc283334897cc47e07757184f27f00c575537657, archive SHA-256
7fc99c95abcfad823f71dd10ddbf9e82b6e7fd0b8d808ad3b4d62913efa57380.

The machine-readable [release manifest](release_manifest.json) records the
archive's 3,734,719,959-byte size and seven anchor hashes verified directly against
archive members. It also identifies the separately added follow-up supplement.
Its `code_base_commit` records the historical starting point; `SHA256SUMS.txt`
identifies the current code package without requiring Git or a network connection.

The historical archive contains synthetic images, source identifiers, report
text, and pseudonymized reader rows. It is not aggregate-only. Use it under the
applicable dataset and reader-data permissions. Source radiographs and masks are
not included. Aggregate access alone cannot reproduce per-pair scoring.
This guide does not grant redistribution rights.

For an authorized copy, download the two files from the
[pinned Hugging Face revision](https://huggingface.co/datasets/anon-submission7979/spinefairbench-artifacts/tree/fc283334897cc47e07757184f27f00c575537657),
or install the hf CLI and use the command below:

~~~sh
python -m pip install huggingface_hub
hf download anon-submission7979/spinefairbench-artifacts spinefairbench_artifacts.tar.gz spinefairbench_artifacts.tar.gz.sha256 --repo-type dataset --revision fc283334897cc47e07757184f27f00c575537657 --local-dir .
python reviewer_verify.py checksums spinefairbench_artifacts.tar.gz.sha256
tar -xzf spinefairbench_artifacts.tar.gz
python reviewer_verify.py checksums
python reviewer_verify.py release-identity
python reviewer_verify.py checksums artifacts/SHA256SUMS.txt
python reviewer_verify.py checksums artifacts/radiologist_validation_SHA256SUMS.txt
~~~

Each checksum command must exit successfully. Repository text uses LF endings
through .gitattributes. Do not rewrite artifact bytes or manifests to hide a mismatch.
On Windows, extract under a short path such as C:\sfb; deeply nested working
directories can exceed the legacy 260-character path limit for archive members.

`release-identity` must report seven matching anchors and 11 preserved historical
prompt entries. Artifact checksums must report 91,844 files, and the separate
reader checksum manifest 12 files. The identity check pins the manifests; the
checksum commands verify their listed bytes. Newly created files outside those
manifests are not certified by the checksum command.

## Reader-record erratum (2026-09-07)

The frozen reader files and their hashes are preserved. Their reviewer labels
are file-local: all four response fields for the 450 main cases in
artifacts/validation_public/per_reviewer/reviews_R1_public_2026-04-28.csv match R2 in
artifacts/validation_public/validation_pair_results_public_2026-04-28.csv; per-reviewer R2
matches combined R1, and R3 matches R3. This correspondence does not identify
individual radiologists. The hidden-repeat records match the per-reviewer labels;
do not apply the combined-file mapping to those repeat records.

The combined file has 1,350 main responses; the per-reviewer exports contain
1,380 events, including 30 hidden repeats. All combined excluded_final fields
are blank. Use the separate artifacts/radiologist_exclusion_list.json for the
seven excluded pair IDs, which match the majority-rule failures. Pooled acceptance
remains 443/450, and repeat agreements for Q1/Q2/Q3/Q4 remain 30/30, 30/30,
29/30, and 27/30.

The aggregate artifacts/validation_report_public_final_2026-04-28.json also
contains stale fields. Its Q2 Gwet AC1 is
0.9804; the identical Q1/Q2 row-level responses both yield 0.9799846264, rounded
to 0.9800 (raw agreement 0.9807; Fleiss kappa 0.4902). Its release_scope says
raw per-pair responses remain private, although this archive includes the
pseudonymized row-level exports described above. These corrections do not grant
reader-data redistribution rights.

## Retained results and accounting

~~~sh
python reviewer_verify.py inspect
python reviewer_verify.py dataset
python reviewer_verify.py table2
python reviewer_verify.py stage1-confidence
python reviewer_verify.py mitigation
python reviewer_verify.py radiologist
~~~

The table2 command verifies all nine primary rows and full/partial refusals
against artifacts/Results/analysis/common_core_1000_summary.json. It reads the
saved confidence intervals without recomputing them. All 18 primary point
estimates and all 36 displayed interval bounds match the final paper.

| Model | Usable pairs | Recommendation change (95% CI) | Diagnostic consistency (95% CI) |
|---|---:|---|---|
| GPT-5.4 | 3,998 | 0.694 [0.678, 0.710] | 0.649 [0.640, 0.657] |
| Qwen2.5-VL | 3,998 | 0.293 [0.271, 0.318] | 0.545 [0.525, 0.565] |

Dataset checks expect 11,795 QC-passed images and rows, 11,948 attempted QC rows,
and no source PNGs. Reader checks expect 443/450 accepted pairs and 1,307/1,350
"Cannot tell" responses. Stage-1 parsing admits GPT-5.4 and GLM-4.6V at 193/200
each; both fail the Condition B joint rule.

Dataset verification also rejects duplicate pair/source IDs, inconsistent QC
selection, source/image mapping errors, replaced PNG membership, and incorrect
image sizes. Retained-report verification rejects duplicate or orphan reports,
unresolved errors, missing report text, and contradictory source-cluster IDs.
Full refusals remain valid retained responses and receive the frozen exclusion
policy. Baseline-only models are still verified in their appropriate panel.

Optional interval regeneration from saved reports:

~~~sh
python -m pip install -r requirements.txt
python reviewer_verify.py table2 --recompute-ci
~~~

This uses sorted source clusters, NumPy PCG64, 10,000 resamples, and seed 42
independently for each endpoint. The retained July audit identified this RNG
and seed as matching the frozen intervals. The command checks intervals to
absolute tolerance 1e-12 and rejects other settings for frozen-record verification.

## Later aggregate evidence in the updated manuscript

~~~sh
python reviewer_verify.py followup
~~~

[supplement/retained_followup_audit_summary.json](supplement/retained_followup_audit_summary.json)
is a byte-identical extract of retained July audits, separately packaged in
September. It contains aggregate target-estimator and reconstruction results,
finding strata, recommendation transitions, reader reliability, source demographics,
and repeated-call variability. It does not contain individual reader identities
or patient-level reports. Input hashes identify the retained sources; they do not
make unavailable inputs publicly accessible.

The command verifies the supplement's pinned hash, the nine-model set, repeated-call
source accounting, and the stored `T-F` and `(T-F)/(1-F)` arithmetic. It reports
919 eligible sources for Gemma and 1,000 for each other model. It does not generate
images, call VLMs, bootstrap intervals, or reproduce the underlying July audits.
The primary table remains pair-weighted; these post-hoc repeated-call summaries
weight sources equally and have 18 unadjusted raw-excess intervals.

| Manuscript evidence | Retained location or command | What verification establishes |
|---|---|---|
| Primary nine-model endpoints | `table2`; frozen `common_core_1000_summary.json` | Saved-report lexical scores, denominators and refusal accounting |
| Metric-construction sensitivity | `gap-sensitivity` | Binary/graded comparison using the same saved reports |
| Repeated-call variability | `followup`; supplement `repeated_calls` | Aggregate identity and arithmetic, not new inference |
| Source cohorts, finding strata, transitions | Supplement sections of the same names | Accessible retained aggregate evidence with input hashes |
| Target estimator and reconstruction audit | Supplement `target_validity` and `reconstruction` | Retained results and their limitations, not demographic or causal validation |
| Mitigation | `stage1-confidence` and `mitigation` | Saved parse-gate sample, table arithmetic and binding-rule outcome |
| Reader study | `radiologist` and the erratum above | Retained rows under documented file-local labels |

## Optional scorer inspection

~~~sh
python reviewer_verify.py diagnostic-scoring
python reviewer_verify.py parse-sample --model gpt-5.4
python reviewer_verify.py both-empty-diagnostic
python reviewer_verify.py gap-sensitivity
~~~

The last two are optional sensitivities on retained outputs. Both-empty diagnosis
sets score 1.0. The difference between binary recommendation inequality and graded
diagnosis overlap depends on metric construction; it is not a clinical ranking.

The numeric mitigation guardrail first appears in the retained record on
2026-04-25, with code/results on 2026-04-27. It was not preregistered on 2026-04-08.

Frozen-output verification, new-model scoring, and historical regeneration are
different operations. These checks establish agreement with retained records.
They cannot establish missing historical provenance, successful age/sex transfer,
clinical correctness, or a causal demographic effect.
