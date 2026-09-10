"""The verification CLI must reject changed source or frozen inputs."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_hallucination_provenance.py"
SOURCE = ROOT / "provenance/historical_scoring_source/spinefairbench"


class ProvenanceRejectionTests(unittest.TestCase):
    def invoke(self, source: Path, artifacts: Path):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--recorded-source-root", str(source), "--artifacts", str(artifacts)],
            text=True, capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )

    def test_changed_source_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            source = temporary / "source"
            shutil.copytree(SOURCE, source)
            marker = temporary / "executed"
            target = source / "metrics/diagnostic_label.py"
            target.write_text(target.read_text() + f"\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('unexpected execution')\n")
            completed = self.invoke(source, temporary / "missing_artifacts")
            self.assertEqual(completed.returncode, 1)
            self.assertIn("SHA-256 mismatch: metrics/diagnostic_label.py", completed.stderr)
            self.assertFalse(marker.exists())

    def test_changed_frozen_summary_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            summary = temporary / "Results/analysis/common_core_1000_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text('{"panels": {}}\n')
            completed = self.invoke(SOURCE, temporary)
            self.assertEqual(completed.returncode, 1)
            self.assertIn("SHA-256 mismatch: Results/analysis/common_core_1000_summary.json", completed.stderr)
            self.assertEqual(completed.stdout, "")

    def test_extracted_bundle_root_resolves_inner_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            summary = temporary / "artifacts/Results/analysis/common_core_1000_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text('{"panels": {}}\n')
            completed = self.invoke(SOURCE, temporary)
            self.assertEqual(completed.returncode, 1)
            self.assertIn("SHA-256 mismatch: Results/analysis/common_core_1000_summary.json", completed.stderr)


if __name__ == "__main__":
    unittest.main()
