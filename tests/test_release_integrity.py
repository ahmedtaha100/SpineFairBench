"""Corruption cases for retained-release verification; no model calls."""

import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reviewer_verify as verify
from spinefairbench.release import scoring


class ReleaseIntegrityTests(unittest.TestCase):
    def test_checksum_rejects_duplicates_malformed_entries_and_path_escapes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").write_bytes(b"data")
            digest = hashlib.sha256(b"data").hexdigest()
            manifest = root / "SHA256SUMS.txt"
            for contents, message in [
                (f"{digest}  data\n{digest}  data\n", "Duplicate"),
                ("not-a-hash  data\n", "Malformed"),
                (f"{digest}  ../data\n", "Invalid release-relative path"),
                (f"{digest}  /data\n", "Invalid release-relative path"),
                (f"{digest}  ./data\n", "Invalid release-relative path"),
            ]:
                with self.subTest(message=message, contents=contents):
                    manifest.write_text(contents)
                    with self.assertRaisesRegex(SystemExit, message):
                        verify.command_checksums(argparse.Namespace(manifest=str(manifest)))

    def test_retained_reports_reject_ambiguous_pairs_but_keep_full_refusals(self):
        rows = [{"model": "gpt-5.4", "pair_id": "case__young_female", "source_id": "case",
                 "image_role": role, "response": "Normal spine. Physical therapy recommended."}
                for role in ("source", "generated")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evaluation_results.json"
            with patch.object(verify, "_panel_for_model", return_value="full_pipeline_retained"), \
                    patch.object(verify, "_panel_paths", return_value=(path, path, path)):
                for mutation, message in [
                    (lambda r: r + [r[0]], "Duplicate source"),
                    (lambda r: r + [r[1]], "Duplicate pair_id"),
                    (lambda r: r[:1], "Unpaired"),
                    (lambda r: [dict(r[0], response=None), r[1]], "Missing report"),
                    (lambda r: [dict(r[0], error="timeout"), r[1]], "Unresolved evaluation error"),
                    (lambda r: [dict(r[0], pair_source_id="other"), r[1]], "Inconsistent source cluster"),
                ]:
                    with self.subTest(message=message):
                        path.write_text(json.dumps(mutation(copy.deepcopy(rows))))
                        with self.assertRaisesRegex(SystemExit, message):
                            verify._paired_records(Path(temp), "gpt-5.4")
                # Refusal is an observed response, not a missing/error record.
                rows[0]["response"] = "I cannot interpret medical images."
                path.write_text(json.dumps(rows))
                sources, generated = verify._paired_records(Path(temp), "gpt-5.4")
                self.assertEqual((len(sources), len(generated)), (1, 1))

    def _dataset(self, root):
        dataset = root / "dataset"
        image = dataset / "counterfactual_images/case/young_female.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"synthetic fixture bytes")
        row = {"pair_id": "case__young_female", "source_id": "case", "edit_label": "young_female",
               "passed_qc": True, "counterfactual_image_path": image.relative_to(root).as_posix(),
               "counterfactual_image_size_bytes": image.stat().st_size}
        for name in ("qc_metadata.jsonl", "qc_passed_pair_manifest.jsonl"):
            (dataset / name).write_text(json.dumps(row) + "\n")
        source = {"source_id": "case", "counterfactual_images": {"young_female": row["counterfactual_image_path"]},
                  "qc_passed_counterfactuals": ["young_female"]}
        for name in ("pairs.json", "source_metadata.json"):
            (dataset / name).write_text(json.dumps([source]))
        manifest = {"counterfactual_images": {"included_png_count": 1}, "pair_manifests": {
            "qc_passed_pair_rows": 1, "attempted_qc_rows": 1, "source_level_rows": 1},
            "source_radiographs": {"release_status": "not_redistributed"}}
        (dataset / "release_manifest.json").write_text(json.dumps(manifest))
        return dataset, row

    def test_dataset_rejects_same_count_replacement_and_extra_source_png(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            dataset, row = self._dataset(root)
            args = argparse.Namespace(artifacts=str(root))
            verify.command_dataset(args)
            image = root / row["counterfactual_image_path"]
            image.rename(image.with_name("young_male.png"))
            with self.assertRaisesRegex(SystemExit, "missing or size mismatch"):
                verify.command_dataset(args)
            image.with_name("young_male.png").rename(image)
            (dataset / "source.png").write_bytes(b"unlisted source")
            with self.assertRaisesRegex(SystemExit, "PNG membership"):
                verify.command_dataset(args)

    def test_dataset_rejects_duplicate_qc_and_changed_source_mapping(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            dataset, row = self._dataset(root)
            args = argparse.Namespace(artifacts=str(root))
            qc = dataset / "qc_metadata.jsonl"
            qc.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(SystemExit, "Duplicate pair_id"):
                verify.command_dataset(args)
            qc.write_text(json.dumps(row) + "\n")
            (dataset / "source_metadata.json").write_text(json.dumps([{"source_id": "different"}]))
            with self.assertRaisesRegex(SystemExit, "Source membership"):
                verify.command_dataset(args)

    def test_pinned_identity_rejects_replaced_checksum_manifest(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            code, data = root / "code", root / "data"
            (code / "prompts").mkdir(parents=True)
            data.mkdir()
            registry = {"prompt_registry": {"system": "report", "primary": "image"}}
            (data / "prompts.json").write_text(json.dumps(registry))
            (code / "prompts/canonical_definitions.json").write_text(json.dumps(registry))
            (data / "SHA256SUMS.txt").write_bytes(b"original\n")
            release = {"schema_version": "1.0", "artifact_anchors": {
                "SHA256SUMS.txt": hashlib.sha256(b"original\n").hexdigest()},
                "frozen_prompt_registry": "prompts.json", "artifact_archive": {"revision": "frozen"}}
            (code / "release_manifest.json").write_text(json.dumps(release))
            with patch.object(verify, "CODE_ROOT", code):
                verify.command_release_identity(argparse.Namespace(artifacts=str(data)))
                (data / "SHA256SUMS.txt").write_text("replacement\n")
                with self.assertRaisesRegex(SystemExit, "Pinned release identity mismatch"):
                    verify.command_release_identity(argparse.Namespace(artifacts=str(data)))

    def test_release_identity_requires_explicit_personal_variant(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            code, data = root / "code", root / "data"
            (code / "prompts").mkdir(parents=True)
            data.mkdir()
            registry = {"prompt_registry": {"system": "report", "primary": "image"}}
            for path in (data / "prompts.json", code / "prompts/canonical_definitions.json"):
                path.write_text(json.dumps(registry))
            (data / "SHA256SUMS.txt").write_bytes(b"personal\n")
            release = {"schema_version": "1.0", "frozen_prompt_registry": "prompts.json",
                       "artifact_archive": {"revision": "anonymous"},
                       "artifact_anchors": {"SHA256SUMS.txt": hashlib.sha256(b"anonymous\n").hexdigest()},
                       "archive_variants": {"personal": {"artifact_archive": {"revision": "personal"},
                           "artifact_anchors": {"SHA256SUMS.txt": hashlib.sha256(b"personal\n").hexdigest()}}}}
            (code / "release_manifest.json").write_text(json.dumps(release))
            with patch.object(verify, "CODE_ROOT", code):
                with self.assertRaisesRegex(SystemExit, "Pinned release identity mismatch"):
                    verify.command_release_identity(argparse.Namespace(artifacts=str(data)))
                verify.command_release_identity(argparse.Namespace(artifacts=str(data), archive_variant="personal"))
                with self.assertRaisesRegex(SystemExit, "Unsupported archive variant"):
                    verify.command_release_identity(argparse.Namespace(artifacts=str(data), archive_variant="unknown"))

    def test_complete_qc_collection_is_not_a_retained_panel_scope(self):
        pair = scoring.BenchmarkPair("case__young_female", "case", "young_female")
        payload = {"schema_version": scoring.SUBMISSION_SCHEMA_VERSION, "scope": "qc-passed",
                   "model": {"name": "fixture"}, "results": [{"pair_id": pair.pair_id,
                       "source_report": "Normal spine.", "counterfactual_report": "Normal spine."}]}
        with patch.object(scoring, "load_benchmark_pairs", return_value={pair.pair_id: pair}):
            output = scoring.score_submission_payload(Path("."), payload, bootstrap_iterations=10)
        self.assertTrue(output["coverage"]["coverage_complete"])
        self.assertFalse(output["coverage"]["comparable_to_panel_scope"])
        self.assertEqual(output["primary_endpoints"]["diagnostic_label_consistency"]["point_estimate"], 1.0)

    def test_scorer_rejects_corrupt_qc_manifest_before_computing_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, row = self._dataset(root)
            manifest = dataset / "qc_passed_pair_manifest.jsonl"
            for rows, message in [
                ([row, row], "Duplicate pair_id"),
                ([dict(row, pair_id="")], "Invalid source/edit/pair identity"),
                ([dict(row, source_id="different")], "Invalid source/edit/pair identity"),
                ([dict(row, edit_label=[])], "Invalid source/edit/pair identity"),
                ([dict(row, passed_qc=False)], "Non-passing row"),
            ]:
                with self.subTest(message=message, rows=rows):
                    manifest.write_text("".join(json.dumps(item) + "\n" for item in rows))
                    with self.assertRaisesRegex(scoring.ScoringError, message):
                        scoring.load_benchmark_pairs(root, "qc-passed")

    def test_scorer_rejects_corrupt_scope_manifests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, row = self._dataset(root)
            core = root / "artifacts/freeze_runs/2026-04-09/evaluation_source_subset_core1000.json"
            core.parent.mkdir(parents=True)
            for payload, message in [
                ({"source_ids": ["case", "case"], "actual_n": 2}, "Duplicate source_id"),
                ({"source_ids": [None], "actual_n": 1}, "Invalid source_id"),
                ({"source_ids": ["case"], "actual_n": 2}, "actual_n does not match"),
                ({"source_ids": ["absent"], "actual_n": 1}, "source IDs absent from QC"),
            ]:
                with self.subTest(payload=payload):
                    core.write_text(json.dumps(payload))
                    with self.assertRaisesRegex(scoring.ScoringError, message):
                        scoring.load_benchmark_pairs(root, "common-core-1000")
            core.write_text(json.dumps({"source_ids": ["case"], "actual_n": 1}))
            self.assertEqual(set(scoring.load_benchmark_pairs(root, "common-core-1000")), {row["pair_id"]})
            intersection = root / "artifacts/Results/final_inputs/all_model_intersection_2166_manifest.json"
            intersection.parent.mkdir(parents=True)
            for records, count, message in [
                ([row, row], 2, "Duplicate pair_id"),
                ([dict(row, source_id="different")], 1, "Invalid source/edit/pair identity"),
                ([None], 1, "must be an object"),
                ([row], 2, "pair_count does not match"),
            ]:
                with self.subTest(records=records, count=count):
                    intersection.write_text(json.dumps({"records": records, "pair_count": count}))
                    with self.assertRaisesRegex(scoring.ScoringError, message):
                        scoring.load_benchmark_pairs(root, "all-model-intersection-2166")
            intersection.write_text(json.dumps({"records": [row], "pair_count": 1}))
            self.assertEqual(set(scoring.load_benchmark_pairs(root, "all-model-intersection-2166")), {row["pair_id"]})
            outside_core = dict(row, source_id="other", pair_id="other__young_female")
            (root / "dataset/qc_passed_pair_manifest.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in (row, outside_core)))
            intersection.write_text(json.dumps({"records": [outside_core], "pair_count": 1}))
            with self.assertRaisesRegex(scoring.ScoringError, "outside the common core"):
                scoring.load_benchmark_pairs(root, "all-model-intersection-2166")

    def test_named_scope_rejects_self_consistent_replacement_before_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, row = self._dataset(root)
            core = root / "artifacts/freeze_runs/2026-04-09/evaluation_source_subset_core1000.json"
            core.parent.mkdir(parents=True)
            core.write_text(json.dumps({"source_ids": ["case"], "actual_n": 1}))
            payload = {"schema_version": scoring.SUBMISSION_SCHEMA_VERSION,
                       "scope": "common-core-1000", "model": {"name": "test"},
                       "results": [{"pair_id": row["pair_id"], "source_report": "No fracture.",
                                    "counterfactual_report": "No fracture."}]}
            # Valid internal membership cannot authenticate the named frozen scope.
            self.assertEqual(len(scoring.load_benchmark_pairs(root, "common-core-1000")), 1)
            with patch.object(scoring, "source_clustered_bootstrap_ci") as bootstrap:
                with self.assertRaisesRegex(scoring.ScoringError, "Frozen scope manifest SHA-256 mismatch"):
                    scoring.score_submission_payload(root, payload)
                bootstrap.assert_not_called()

    def test_named_scope_pins_every_required_manifest_and_records_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, row = self._dataset(root)
            core = root / "artifacts/freeze_runs/2026-04-09/evaluation_source_subset_core1000.json"
            core.parent.mkdir(parents=True)
            core.write_text(json.dumps({"source_ids": ["case"], "actual_n": 1}))
            intersection = root / "artifacts/Results/final_inputs/all_model_intersection_2166_manifest.json"
            intersection.parent.mkdir(parents=True)
            intersection.write_text(json.dumps({"records": [row], "pair_count": 1}))
            pins = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                    for relative in scoring.FROZEN_SCOPE_MANIFEST_HASHES}
            with patch.object(scoring, "FROZEN_SCOPE_MANIFEST_HASHES", pins):
                for scope in ("common-core-1000", "all-model-intersection-2166"):
                    payload = {"schema_version": scoring.SUBMISSION_SCHEMA_VERSION,
                               "scope": scope, "model": {"name": "test"},
                               "results": [{"pair_id": row["pair_id"], "source_report": "No fracture.",
                                            "counterfactual_report": "No fracture."}]}
                    with patch.object(scoring, "source_clustered_bootstrap_ci", return_value=(1.0, 1.0)):
                        output = scoring.score_submission_payload(root, payload)
                    identity = output["artifact_identity"]
                    self.assertEqual(identity["status"], "verified_frozen_scope")
                    self.assertEqual(len(identity["manifest_sha256"]), 2 if scope == "common-core-1000" else 3)
                    self.assertTrue(output["coverage"]["comparable_to_panel_scope"])
                    for relative, digest in identity["manifest_sha256"].items():
                        with self.subTest(scope=scope, relative=relative):
                            self.assertEqual(digest, pins[relative])
                            path = root / relative
                            original = path.read_bytes()
                            path.write_bytes(original + b"\n")
                            with self.assertRaisesRegex(scoring.ScoringError, "SHA-256 mismatch"):
                                scoring.score_submission_payload(root, payload)
                            path.write_bytes(original)

    def test_followup_rejects_wrong_aggregate_arithmetic(self):
        source = verify.CODE_ROOT / "supplement/retained_followup_audit_summary.json"
        summary = json.loads(source.read_text())
        summary["repeated_calls"]["models"]["gpt-5.4"]["recommendation"]["excess"] += .01
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            code = Path(temp)
            path = code / "followup.json"
            path.write_text(json.dumps(summary))
            manifest = {"supplemental_artifacts": {"retained_followup_audit": {
                "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}
            (code / "release_manifest.json").write_text(json.dumps(manifest))
            with patch.object(verify, "CODE_ROOT", code):
                with self.assertRaisesRegex(SystemExit, "aggregate arithmetic mismatch"):
                    verify.command_followup(argparse.Namespace())

    def test_mitigation_rejects_nonfinite_table_cells(self):
        for value in ("NaN", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaisesRegex(SystemExit, "Invalid mitigation-table value"):
                verify._cell_float({"rec_change": value}, "rec_change")


if __name__ == "__main__":
    unittest.main()
