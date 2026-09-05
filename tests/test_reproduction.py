"""Small regression fixtures; no released study data or model calls."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import reviewer_verify as verify
from spinefairbench.generator.qc import passes_qc
from spinefairbench.release.scoring import source_clustered_bootstrap_ci


class ReproductionTests(unittest.TestCase):
    def test_bootstrap_matches_numpy_cluster_resampling(self):
        # NumPy seed-42 reference from direct resampling of four synthetic
        # source clusters of unequal sizes; crosses the 1,024-draw boundary.
        actual = source_clustered_bootstrap_ci(
            [0., 1., .2, .4, .8, .3, .7], ["a", "a", "b", "b", "b", "c", "d"],
            iterations=1031, seed=42,
        )
        self.assertEqual(actual, (0.3833333333333333, 0.605))

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
            rec_ci = {"lower": 1., "upper": 1.}
            quality = {"pair_usable": 1, "pair_full_refusals": 0, "pair_partial_refusals": 0}
            frozen = {"data_quality": quality, "primary_secondary_stats": {
                "recommendation": {"agreement_rate": 1., "bootstrap_ci": rec_ci},
                "diagnostic_label": {"mean": 1., "bootstrap_ci": {"lower": 1., "upper": 1.}}}}
            summary = {"source_clustered_bootstrap": {"iterations": 10000},
                       "panels": {"full": {"models": {"gpt-5.4": frozen}}}}
            summary_path = root / "artifacts/Results/analysis/common_core_1000_summary.json"
            summary_path.parent.mkdir(parents=True)
            args = argparse.Namespace(artifacts=str(root), model="gpt-5.4",
                                      recompute_ci=True, bootstrap_iterations=10000, seed=42)
            with contextlib.redirect_stdout(io.StringIO()):
                summary_path.write_text(json.dumps(summary))
                verify.command_table2(args)
                rec_ci["lower"] = .9
                summary_path.write_text(json.dumps(summary))
                with self.assertRaisesRegex(SystemExit, "CI does not match"):
                    verify.command_table2(args)
                rec_ci["lower"] = 1.
                quality["pair_full_refusals"] = 1
                summary_path.write_text(json.dumps(summary))
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
