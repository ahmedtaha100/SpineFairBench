"""Small regression fixtures; no released study data or model calls."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reviewer_verify as verify
from spinefairbench.generator.qc import passes_qc
from spinefairbench.release.scoring import source_clustered_bootstrap_ci


class ReproductionTests(unittest.TestCase):
    def test_bootstrap_matches_archived_producer(self):
        # Archived run3_derivations._source_cluster_ci, unequal cluster sizes,
        # crossing its 1,024-resample chunk boundary. These are synthetic data.
        actual = source_clustered_bootstrap_ci(
            [0., 1., .2, .4, .8, .3, .7], ["a", "a", "b", "b", "b", "c", "d"],
            iterations=1031, seed=20260437,
        )
        self.assertEqual(actual, (0.38249999999999995, 0.6))

    def test_table2_checks_frozen_intervals_and_refusal_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for panel, models in (("full_pipeline_retained", ["gpt-5.4"]),
                                  ("baseline_only_retained", [])):
                path = root / "artifacts/Results/final_inputs/panels" / panel
                path.mkdir(parents=True)
                (path / "panel_manifest.json").write_text(json.dumps({"models": models}))
                (path / "pairs.json").write_text("[]")
                rows = [{"model": "gpt-5.4", "pair_id": "case__young_female",
                         "source_id": "case", "image_role": role,
                         "response": "Normal spine. Physical therapy recommended."}
                        for role in ("source", "generated")]
                (path / "evaluation_results.json").write_text(json.dumps(rows if models else []))
            frozen = {"pairs": 1, "full_refusals": 0, "partial_refusals": 0,
                      "rec_change": 0., "diag_consistency": 1.,
                      "rec_change_ci": {"lower": 0., "upper": 0.},
                      "diag_consistency_ci": {"lower": 1., "upper": 1.}}
            summary = {"metadata": {"n_resamples": 10000, "seed": 20260426},
                       "models": {"gpt-5.4": frozen}}
            args = argparse.Namespace(artifacts=str(root), model="gpt-5.4",
                                      recompute_ci=True, bootstrap_iterations=10000, seed=20260426)
            with patch.object(verify, "CODE_ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                (root / "paper_results.json").write_text(json.dumps(summary))
                verify.command_table2(args)
                frozen["rec_change_ci"]["upper"] = .1
                (root / "paper_results.json").write_text(json.dumps(summary))
                with self.assertRaisesRegex(SystemExit, "CI does not match"):
                    verify.command_table2(args)
                frozen["rec_change_ci"]["upper"] = 0.
                frozen["full_refusals"] = 1
                (root / "paper_results.json").write_text(json.dumps(summary))
                with self.assertRaisesRegex(SystemExit, "full_refusals does not match"):
                    verify.command_table2(args)

    def test_checksums_detect_changed_and_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "file with spaces.json"
            data.write_bytes(b"{}\n")
            manifest = root / "SHA256SUMS.txt"
            manifest.write_text(f"{hashlib.sha256(data.read_bytes()).hexdigest()}  {data.name}\n")
            args = argparse.Namespace(manifest=str(manifest))
            with contextlib.redirect_stdout(io.StringIO()):
                verify.command_checksums(args)
                data.write_bytes(b"changed")
                with self.assertRaisesRegex(SystemExit, "Checksum mismatch"):
                    verify.command_checksums(args)
                data.unlink()
                with self.assertRaisesRegex(SystemExit, "Checksum file missing"):
                    verify.command_checksums(args)

    def test_qc_requires_all_three_measurements(self):
        metrics = {"ssim": .9, "edge_preservation": .8, "lpips": .2}
        self.assertTrue(passes_qc(metrics))
        metrics["lpips"] = None
        self.assertFalse(passes_qc(metrics))


if __name__ == "__main__":
    unittest.main()
