from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class ReleaseError(RuntimeError):
    """Raised for release packaging or scoring setup errors."""


LEADERBOARD_COLUMNS = [
    "submission_id",
    "model_name",
    "organization",
    "model_family",
    "scored_at_utc",
    "n_pairs",
    "diagnostic_label_mean_jaccard",
    "severity_language_mean_abs_difference",
    "recommendation_agreement_rate",
    "confidence_mean_abs_difference",
    "hallucination_disparity",
    "score_file",
    "notes",
]


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        raise ReleaseError(f"Output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise ReleaseError(f"Required file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_files_recursive(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file()],
        key=lambda p: p.as_posix(),
    )


def write_checksums(root: Path, output_file: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    for file_path in list_files_recursive(root):
        if file_path == output_file:
            continue
        rel = file_path.relative_to(root).as_posix()
        checksum = sha256_file(file_path)
        size = file_path.stat().st_size
        entries.append(
            {
                "path": rel,
                "sha256": checksum,
                "size_bytes": size,
            }
        )
        lines.append(f"{checksum}  {rel}")
    write_text(output_file, "\n".join(lines) + ("\n" if lines else ""))
    return entries


def append_checksum_entry(
    root: Path,
    checksum_file: Path,
    target_file: Path,
) -> dict[str, Any]:
    rel = target_file.relative_to(root).as_posix()
    checksum = sha256_file(target_file)
    size = target_file.stat().st_size
    with open(checksum_file, "a", encoding="utf-8") as f:
        f.write(f"{checksum}  {rel}\n")
    return {
        "path": rel,
        "sha256": checksum,
        "size_bytes": size,
    }


def create_tar_gz(source_dir: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)
    return archive_path


def normalize_release_tag(value: str) -> str:
    if not value:
        raise ReleaseError("Release version cannot be empty")
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)
    cleaned = cleaned.strip("-_.")
    if not cleaned:
        raise ReleaseError("Release version produced an empty tag after normalization")
    return cleaned
