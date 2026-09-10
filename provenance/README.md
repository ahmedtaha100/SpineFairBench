# Historical provenance

The September 2026 recovery connects retained training, generation and scoring
records while preserving the published images, reports, endpoint definitions and
results. The [asset manifest](historical_asset_manifest.json) records exact hashes,
repository revisions and access conditions. No training, image generation, VLM
calls or historical confidence-interval replay was performed for this recovery.
Run the commands from the code repository root with Python 3.11 or newer after
following the [reviewer quickstart](../reviewer_quickstart.md).

## What the records establish

| Record | Verified finding | Remaining boundary |
|---|---|---|
| [Eligible corpus](training_membership_reconstruction.json) | The historical final-image inventory and VinDr/BUU filter identify 9,024 filenames: 5,337 VinDr and 3,687 BUU. | The complete original DICOM and annotation bytes are unavailable. |
| [Configured splits](configured_training_split_reconstruction.json) | The recovered loader/configuration assign 7,219 training, 902 validation and 903 test source IDs. | These are prescribed partitions; historical overrides and per-step optimizer consumption are not authenticated. |
| [Training log excerpts](training_log_excerpts.json) | Exact dated lines confirm 7,219/902 train/validation counts, an epoch-2 checkpoint, and later Stage-2 execution. | The retained code includes corrupt-image fallback and drops incomplete training batches; the log does not record per-step consumption or establish that fallback occurred. |
| [Public generator assets](https://huggingface.co/ahmedtaha100/spinefairbench-generator/tree/77335a8c3fdc388020f5582fb04df741b5fe7735) | The public checkpoint equals the retained Stage-1 `epoch_2.pt` bytes; the LoRA header identifies that checkpoint. | The selected checkpoint does not contain the later Stage-2 updates recorded in the full log. The public adapter remains illustrative. |
| [Source PNGs and masks](historical_asset_manifest.json) | All 2,987 generation-source PNGs and 2,987 production masks were downloaded and hash-verified against the pinned backup; every hash also matches the original full-freeze inventory. | These are access-controlled generation assets, not recovery of the complete training DICOM corpus. |
| [Scoring lineage](scoring_lineage.json) | Original April 20 records connect all nine archived exploratory-statistic objects to the frozen summary. An April 20 preservation note documents the pair-ID repair; a later retained repaired source matches all nine means/counts. | The run records `git_dirty=true`; the later source is not hash-bound to the actual April 20 execution. |

The April generation calls record checkpoint paths and settings but do not bind
the runtime weight or code bytes by hash. The public checkpoint's identity as
the selected archival Stage-1 snapshot does not authenticate its loading during
those calls.

[Static checkpoint metadata](checkpoint_static_metadata.json) independently
identifies Stage 1, epoch 2 and global step 902 from pickle opcodes in the
hash-verified checkpoint. No checkpoint was unpickled or loaded into a model;
these fields do not supply a consumed-sample ledger.

All 1,000 common-core source IDs, all 2,000 evaluation-pool source IDs and all
2,950 released QC-pool source IDs occur in the reconstructed eligible corpus.
Within the reconstructed training/validation/test partitions, the common core
contains 395/326/279 source IDs and the evaluation pool 797/606/597. Source-image
membership does not establish patient-level overlap or patient-disjoint evaluation.

## Verify corpus and configured membership

The original inventory remains in an access-controlled Hugging Face repository.
An account explicitly granted access can authenticate through `hf auth login`
and download the pinned file. A repository URL or hash does not itself grant
access. Contact the repository owner through the project repository for access
arrangements; no automatic access approval is promised.

```sh
hf download ahmedtaha100/SpineFairBench-freeze freeze_2026_03_20/file_list.txt --repo-type dataset --revision 1f36d0425d416ac9445f386ec092c1152ceb67f2 --local-dir historical-inputs
python -m pip install -r requirements.txt
python scripts/verify_corpus_provenance.py --historical-inventory historical-inputs/freeze_2026_03_20/file_list.txt --membership provenance/training_membership_reconstruction.json --artifacts artifacts --configured-splits provenance/configured_training_split_reconstruction.json
```

The command checks the inventory SHA-256, every reconstructed source ID and
historical filename, frozen evaluation inputs, configured split ordering and
all recorded split-overlap counts. Omitting `--configured-splits` performs the
eligible-corpus check with the standard library alone. The optional split check
uses the recorded `numpy.random.RandomState(42)` permutation, sorted `.dcm`
paths followed by sorted `.dicom` paths, and the recorded 0.8/0.1/remainder split.
It does not train a model.

## Inspect historical code and scoring records

The [training snapshot](historical_training_source/manifest.json) contains the
three exact source files obtained from the recorded base plus recovered patch,
and the recorded configuration. The [scoring snapshot](historical_scoring_source/manifest.json)
contains five exact files from the recorded scoring base. These are inspection
snapshots, not complete runnable production distributions. The historical commit
IDs are archival identifiers recovered from local Git objects; publicly
retrievable commit URLs are not asserted.

```sh
python scripts/verify_hallucination_provenance.py --artifacts artifacts
```

The scoring audit verifies pinned source/input bytes before reading saved
reports. It reconstructs the exploratory unmatched-prediction fraction with
explicit `pair_id` source/edit joins and checks the mean absolute within-pair
change against the frozen values. It separately evaluates the recorded base
implementation's joining behavior. Agreement of the explicit-pair reconstruction
does not authenticate the dirty historical producer. The recorded base produces
different means for eight models and different denominators for three models on
the same frozen inputs. No retained result is replaced by that comparison.

The [later source recovery](later_scoring_source_recovery.json) narrows this gap.
A note preserved on April 20 at 17:41:09 UTC documents the `pair_id` repair and
names the changed files. Its exact hash and immutable private location are in
the recovery record; the full note contains local paths and remains private.
Five source files recovered from a direct descendant of the recorded base,
with April 27 Git metadata, include the repaired pairing implementation. An
isolated arithmetic check of that source matches all nine frozen means and
denominators. The copied files are inspectable under `later_scoring_source/`.
The remaining requirement is the binding between actual April 20 execution
and source bytes. Later Git metadata and arithmetic agreement cannot provide
that retrospective authentication.

```sh
python scripts/verify_recovered_scoring_candidate.py --artifacts artifacts
```

This command verifies all five candidate source files and six frozen input
files by hash, isolates the retained pairing function, and checks the nine
saved means/counts. It disables the historical aggregate/bootstrap path.

The quantity is lexical, uses the whole-core usable-pair population, and is
neither a conventional false-positive rate nor a clinically adjudicated
hallucination rate. No model calls or bootstrap calculation occur.

Fifteen original training/scoring records were preserved byte-for-byte in the
access-controlled [recovery revision](https://huggingface.co/datasets/ahmedtaha100/SpineFairBench-freeze/tree/c2ab560f157f242ac627a73f0a537298b004bf8f/provenance_recovery_2026-09-10).
All 28,567 prior repository files remain unchanged. The manifest supplies the
exact path/hash for each original log, run manifest, configuration and report.
The public excerpts and reconstructed manifests provide inspectable evidence
without implying public access to those complete private records.

For generation assets, authorized users can download
`freeze-run-2026-04-08/masks/**` and `freeze-run-2026-04-08/pairs/**/source.png`
from `ahmedtaha100/SpineFairBench-freeze-backup` at revision
`aee6e70bddb487fad3c5580033e1e368aaf605af`. The asset manifest lists each of the
5,974 files and its corresponding path in the earlier full freeze. Historical
provider snapshots, complete original training image bytes, the April generator's
runtime code/weight hash binding and the April scoring execution-to-source binding remain
unresolved; available records do not make exact historical
end-to-end regeneration possible.

The [archive comparison](archive_variant_comparison.json) separately records the
anonymous and personal artifact packages. Their scientific files match, but
their archive and manifest hashes differ. Use the explicit archive variant in
the [reviewer quickstart](../reviewer_quickstart.md).
