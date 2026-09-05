# Reviewer quickstart

Use Python 3.11 or newer from the repository root. These commands work in
PowerShell and a POSIX shell and verify existing files and saved reports.

## Artifact identity and access

The historical anonymous bundle is pinned to Hugging Face revision
fc283334897cc47e07757184f27f00c575537657, archive SHA-256
7fc99c95abcfad823f71dd10ddbf9e82b6e7fd0b8d808ad3b4d62913efa57380.
The personal mirror at ee679338f9a54d5c5e281eebeb841edac06332fe has archive SHA-256
f90f8cc54d9ef67d6ffc1bb21d53f9fcf5cff9ca94300353aa83db2ad967e238.
Do not interchange their checksums.

The historical archive contains synthetic images, source identifiers, report
text, and pseudonymized reader rows. It is not aggregate-only. Use it under the
applicable dataset and reader-data permissions. Source radiographs and masks are
not included. Aggregate access alone cannot reproduce per-pair scoring.
This guide does not grant redistribution rights.

For an authorized copy, use the hf CLI or download the two files from the exact
revision through Hugging Face:

~~~sh
hf download anon-submission7979/spinefairbench-artifacts spinefairbench_artifacts.tar.gz spinefairbench_artifacts.tar.gz.sha256 --repo-type dataset --revision fc283334897cc47e07757184f27f00c575537657 --local-dir .
python reviewer_verify.py checksums spinefairbench_artifacts.tar.gz.sha256
tar -xzf spinefairbench_artifacts.tar.gz
python reviewer_verify.py checksums
python reviewer_verify.py checksums artifacts/SHA256SUMS.txt
python reviewer_verify.py checksums artifacts/radiologist_validation_SHA256SUMS.txt
~~~

Each checksum command must exit successfully. Repository text uses LF endings
through .gitattributes. Do not rewrite artifact bytes or manifests to hide a mismatch.

## Paper results and accounting

~~~sh
python reviewer_verify.py inspect
python reviewer_verify.py dataset
python reviewer_verify.py table2
python reviewer_verify.py stage1-confidence
python reviewer_verify.py mitigation
python reviewer_verify.py radiologist
~~~

The table2 command verifies all nine primary rows and full/partial refusals
against paper_results.json. It reads the manuscript's frozen confidence
intervals without recomputing them. Examples:

| Model | Usable pairs | Recommendation change (95% CI) | Diagnostic consistency (95% CI) |
|---|---:|---|---|
| GPT-5.4 | 3,998 | 0.694 [0.678, 0.710] | 0.649 [0.640, 0.657] |
| Qwen2.5-VL | 3,998 | 0.293 [0.270, 0.317] | 0.545 [0.526, 0.564] |

Dataset checks expect 11,795 QC-passed images and rows, 11,948 attempted QC rows,
and no source PNGs. Reader checks expect 443/450 accepted pairs and 1,307/1,350
"Cannot tell" responses. Stage-1 parsing admits GPT-5.4 and GLM-4.6V at 193/200
each; both fail the Condition B joint rule.

Optional interval regeneration from saved reports:

~~~sh
python -m pip install -r requirements.txt
python reviewer_verify.py table2 --recompute-ci
~~~

This uses sorted source clusters, NumPy PCG64, int32 resampling indices, 10,000
resamples, and base seed 20260426 with offsets +11/+23. It checks intervals to
absolute tolerance 1e-12. Other settings are rejected for exact verification.
The old bundle summary contains earlier CIs and remains a historical artifact.

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