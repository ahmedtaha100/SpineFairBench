"""The verification CLI must reject changed source or frozen inputs."""
import hashlib
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.verify_hallucination_provenance import _load_json, load_verified_module

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_hallucination_provenance.py"
SOURCE = ROOT / "provenance/historical_scoring_source/spinefairbench"


class ProvenanceRejectionTests(unittest.TestCase):
    def test_later_candidate_rejects_changed_input_before_source_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for nested in (False, True):
                with self.subTest(nested=nested):
                    artifact_root = root / ("nested" if nested else "direct")
                    inner = artifact_root / "artifacts" if nested else artifact_root
                    summary = inner / "Results/analysis/common_core_1000_summary.json"
                    summary.parent.mkdir(parents=True)
                    summary.write_bytes(b'{"panels": {}}\n')
                    completed = subprocess.run(
                        [sys.executable, str(ROOT / "scripts/verify_recovered_scoring_candidate.py"),
                         "--artifacts", str(artifact_root), "--source-root", str(root / "missing_source")],
                        text=True, capture_output=True, check=False,
                    )
                    self.assertEqual(completed.returncode, 1)
                    self.assertIn("Input SHA-256 mismatch: Results/analysis/common_core_1000_summary.json", completed.stderr)
                    self.assertEqual(completed.stdout, "")

    def test_utf8_report_json_is_independent_of_windows_text_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_bytes('{"response": "No fracture. \u201d"}'.encode("utf-8"))
            with self.assertRaises(UnicodeDecodeError):
                path.read_bytes().decode("cp1252")
            def locale_read_text(file, *args, **kwargs):
                return file.read_bytes().decode("cp1252")
            with patch.object(Path, "read_text", locale_read_text):
                self.assertEqual(_load_json(path), {"response": "No fracture. \u201d"})

    def test_verified_source_execution_ignores_timestamp_valid_bytecode_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.py"
            original = b"value = 'original'\n"
            stale = b"value = 'injected'\n"
            self.assertEqual(len(original), len(stale))
            path.write_bytes(original)
            stat = path.stat()
            path.write_bytes(stale)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            py_compile.compile(str(path), doraise=True,
                               invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
            path.write_bytes(original)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            digest = hashlib.sha256(original).hexdigest()
            module = load_verified_module("_sfb_cache_fixture", path, digest)
            self.assertEqual(module.value, "original")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_verified_module("_sfb_cache_fixture", path, "0" * 64)

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
