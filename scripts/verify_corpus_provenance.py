"""Verify historical eligible-corpus membership; never infer optimizer use."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

INVENTORY_SHA256 = "c05122e9c61ba3c1869a37421e0cdf4b4551cbe785115da829cf5046deca2a55"
FINAL_IMAGE_PREFIX = "Spine_Bench/data/unified/03_final/images/"
EXPECTED_CORPUS_COUNTS = {"buu": 3687, "vindr": 5337}
QC_SHA256 = "9897fb84b97d2cf9cf5dd207dc42fe3f0108cbac1d2df42a59bbda0945f2d0cd"
CONFIGURED_SPLIT_IDENTITIES = {
    "source_file_sha256": INVENTORY_SHA256,
    "config_sha256": "62dc4ea9620ceb2427d10248fe39a6ed4d2b5424e7e910e689bc12ac884df839",
    "source_code_commit": "b9bc1c69c1be2acb7e73637b9971c149ddd3be27",
    "source_patch_sha256": "a7e978928c881c4f8279bbb1901cff739fa1b8144e312a8494ff8584338648af",
    "patched_source_sha256": {
        "spinefairbench/data/dataset.py": "00e52244350b146b84ac37367a44ff703493613b2f218b61424bfe5c192334ad",
        "spinefairbench/data/dataloader.py": "81015b0f7d17d0acf1bcb8b6c8562b58bc0ae63692f089bffaa3ca1df409f328",
        "spinefairbench/train.py": "5e22a9a5430bf543919bf6bef90e6543c559371776aad2185203f4ec2970a761",
    },
}
FROZEN_INPUTS = {
    "common_core": ("artifacts/freeze_runs/2026-04-09/evaluation_source_subset_core1000.json",
                    "56d52393844c11d9aa5b9aed9fe9db144dfa8193c2584d7d980d501bedc2af4d", 1000),
    "evaluation_pool": ("artifacts/freeze_runs/2026-04-09/evaluation_source_subset.json",
                        "e2b080d7ee87e3d35aa439aaaa46484ca010ddba348cea9ff88c8ca6fe641f3e", 2000),
    "retained_panel_sources": ("artifacts/Results/final_inputs/panels/full_pipeline_retained/pairs.json",
                               "cb5c29f25c54a086c934c77517e8e28ae695704192f181084c5efdc5a23c58bb", 2000),
    "released_sources": ("dataset/source_metadata.json",
                         "5180a6399ea2459fb2454272cf339a3a3af643126389a21ae0e67329edc67c36", 2950),
}


class ProvenanceError(ValueError):
    """The supplied records fail historical identity or membership checks."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in items:
        if key in out:
            raise ProvenanceError(f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def _json(data: bytes) -> Any:
    return json.loads(data, object_pairs_hook=_unique_object)


def reconstruct_eligible_corpus(data: bytes) -> tuple[dict[str, tuple[str, str]], int]:
    """Reconstruct names using the retained final-images inventory/source filter."""
    if _sha256(data) != INVENTORY_SHA256:
        raise ProvenanceError("Historical inventory SHA-256 does not match the pinned March 20 file")
    final_paths: set[str] = set()
    members: dict[str, tuple[str, str]] = {}
    for name in data.decode("utf-8").splitlines():
        if not name.startswith(FINAL_IMAGE_PREFIX) or not name.endswith((".dcm", ".dicom")):
            continue
        path = PurePosixPath(name)
        if str(path) != name or ".." in path.parts or str(path.parent) + "/" != FINAL_IMAGE_PREFIX:
            raise ProvenanceError(f"Invalid final-image inventory path: {name}")
        if name in final_paths:
            raise ProvenanceError(f"Duplicate final-image inventory path: {name}")
        final_paths.add(name)
        source_id = path.stem
        source = source_id.split("_", 1)[0]
        if source not in EXPECTED_CORPUS_COUNTS:
            continue
        if not source_id[len(source) + 1:]:
            raise ProvenanceError(f"Empty source identifier: {source_id}")
        if source_id in members:
            raise ProvenanceError(f"Duplicate eligible source_id: {source_id}")
        members[source_id] = (source, name)
    counts = dict(Counter(source for source, _ in members.values()))
    if counts != EXPECTED_CORPUS_COUNTS:
        raise ProvenanceError(f"Eligible corpus source counts differ from the retained record: {counts}")
    return members, len(final_paths)


def verify_membership(payload: Any, expected: dict[str, tuple[str, str]], total_final_images: int) -> None:
    if not isinstance(payload, dict) or payload.get("source_sha256") != INVENTORY_SHA256:
        raise ProvenanceError("Membership manifest does not identify the pinned historical inventory")
    if type(payload.get("total_final_images")) is not int or payload["total_final_images"] != total_final_images:
        raise ProvenanceError("Membership total_final_images differs from the historical inventory")
    rows = payload.get("members")
    if not isinstance(rows, list) or not rows:
        raise ProvenanceError("Membership members must be a nonempty list")
    actual: dict[str, tuple[str, str]] = {}
    paths: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ProvenanceError("Membership row must be an object")
        source_id, source, path = (row.get(key) for key in ("source_id", "source", "historical_path"))
        if (not isinstance(source_id, str) or not source_id or source_id != source_id.strip()
                or not isinstance(source, str) or source not in EXPECTED_CORPUS_COUNTS
                or not source_id.startswith(source + "_") or not isinstance(path, str) or not path):
            raise ProvenanceError("Invalid membership source identity or historical path")
        if source_id in actual or path in paths:
            raise ProvenanceError(f"Duplicate membership source_id or path: {source_id}")
        actual[source_id] = (source, path)
        paths.add(path)
    if actual != expected:
        raise ProvenanceError("Membership records differ from the reconstructed historical eligible corpus")


def _source_ids(payload: Any, label: str, expected_count: int) -> set[str]:
    if isinstance(payload, dict):
        values = payload.get("source_ids")
        if type(payload.get("actual_n")) is not int or payload["actual_n"] != expected_count:
            raise ProvenanceError(f"Invalid recorded source count in {label}")
    elif isinstance(payload, list):
        if any(not isinstance(row, dict) for row in payload):
            raise ProvenanceError(f"Invalid source record in {label}")
        values = [row.get("source_id") for row in payload]
    else:
        raise ProvenanceError(f"Invalid source manifest in {label}")
    if (not isinstance(values, list)
            or any(not isinstance(value, str) or not value or value != value.strip() for value in values)):
        raise ProvenanceError(f"Invalid source_id in {label}")
    if len(set(values)) != len(values):
        raise ProvenanceError(f"Duplicate source_id in {label}")
    if len(values) != expected_count:
        raise ProvenanceError(f"Source count mismatch in {label}")
    return set(values)


def _qc_source_ids(data: bytes) -> set[str]:
    if _sha256(data) != QC_SHA256:
        raise ProvenanceError("Attempted-QC manifest SHA-256 mismatch")
    sources, pairs = set(), set()
    for line in data.splitlines():
        row = _json(line)
        if not isinstance(row, dict):
            raise ProvenanceError("Invalid attempted-QC row")
        pair, source, edit = (row.get(key) for key in ("pair_id", "source_id", "edit_label"))
        if (not isinstance(source, str) or not source or not isinstance(edit, str)
                or edit not in {"young_female", "young_male", "elderly_female", "elderly_male"}
                or pair != f"{source}__{edit}" or type(row.get("passed_qc")) is not bool):
            raise ProvenanceError("Invalid attempted-QC pair identity")
        if pair in pairs:
            raise ProvenanceError(f"Duplicate attempted-QC pair: {pair}")
        pairs.add(pair)
        sources.add(source)
    if len(pairs) != 11948 or len(sources) != 2987:
        raise ProvenanceError("Attempted-QC pair/source counts differ from the frozen record")
    return sources


def configured_split_ids(members: dict[str, tuple[str, str]]) -> dict[str, list[str]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ProvenanceError("Configured-split verification requires pinned NumPy from requirements.txt") from exc
    ordered = sorted(members, key=lambda source_id: (
        0 if members[source_id][1].endswith(".dcm") else 1, members[source_id][1]))
    indices = np.random.RandomState(42).permutation(len(ordered))
    n_train, n_val = int(len(ordered) * .8), int(len(ordered) * .1)
    return {name: [ordered[index] for index in selected] for name, selected in {
        "train": indices[:n_train], "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }.items()}


def verify_configured_splits(payload: Any, members: dict[str, tuple[str, str]],
                             scope_ids: dict[str, set[str]]) -> dict[str, Any]:
    if (not isinstance(payload, dict) or payload.get("schema_version") != "1.0"
            or payload.get("record_type") != "configured_split_reconstruction"):
        raise ProvenanceError("Unsupported configured-split manifest")
    for key, expected in CONFIGURED_SPLIT_IDENTITIES.items():
        if payload.get(key) != expected:
            raise ProvenanceError(f"Configured-split provenance identity mismatch: {key}")
    recipe = payload.get("recipe")
    if not isinstance(recipe, dict):
        raise ProvenanceError("Configured split recipe is missing")
    for key, expected in {"rng": "numpy.random.RandomState", "seed": 42, "train_ratio": .8,
                          "val_ratio": .1, "test_ratio": .1, "operation": f"permutation({len(members)})",
                          "training_batch_size": 16, "training_drop_last": True}.items():
        if recipe.get(key) != expected or type(recipe.get(key)) is not type(expected):
            raise ProvenanceError(f"Configured split recipe mismatch: {key}")
    if type(payload.get("corpus_count")) is not int or payload["corpus_count"] != len(members):
        raise ProvenanceError("Configured-split corpus count mismatch")
    expected_splits = configured_split_ids(members)
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(expected_splits):
        raise ProvenanceError("Configured-split manifest must contain train, val and test")
    output = {}
    for name, expected_ids in expected_splits.items():
        row = splits[name]
        if not isinstance(row, dict):
            raise ProvenanceError(f"Invalid configured split: {name}")
        actual_ids = row.get("source_ids")
        if (not isinstance(actual_ids, list) or any(not isinstance(value, str) for value in actual_ids)
                or len(set(actual_ids)) != len(actual_ids)):
            raise ProvenanceError(f"Invalid or duplicate configured split source IDs: {name}")
        if actual_ids != expected_ids or type(row.get("count")) is not int or row["count"] != len(expected_ids):
            raise ProvenanceError(f"Configured split membership/order/count mismatch: {name}")
        source_counts = dict(sorted(Counter(members[source][0] for source in expected_ids).items()))
        overlap_counts = {label: len(set(expected_ids) & values) for label, values in scope_ids.items()}
        if row.get("source_counts") != source_counts or row.get("overlap_counts") != overlap_counts:
            raise ProvenanceError(f"Configured split source/overlap counts mismatch: {name}")
        output[name] = {"count": len(expected_ids), "source_counts": source_counts, "overlap_counts": overlap_counts}
    return {"status": "verified_configured_split_reconstruction", "splits": output,
            "scope": "Recomputed prescribed filename partition and source-level overlaps; does not authenticate actual runtime use or consumed samples."}


def audit(inventory: Path, membership: Path, artifacts: Path, configured_splits: Path | None = None) -> dict[str, Any]:
    historical_bytes, membership_bytes = inventory.read_bytes(), membership.read_bytes()
    members, final_count = reconstruct_eligible_corpus(historical_bytes)
    verify_membership(_json(membership_bytes), members, final_count)
    corpus_ids = set(members)
    overlap = {}
    scope_ids = {}
    for label, (relative_path, expected_hash, expected_count) in FROZEN_INPUTS.items():
        data = (artifacts / relative_path).read_bytes()
        if _sha256(data) != expected_hash:
            raise ProvenanceError(f"Frozen source-manifest SHA-256 mismatch: {relative_path}")
        ids = _source_ids(_json(data), label, expected_count)
        scope_ids[label] = ids
        overlap[label] = {
            "path": relative_path, "sha256": expected_hash, "source_count": len(ids),
            "overlap_with_eligible_training_corpus": len(ids & corpus_ids),
            "source_ids_outside_eligible_training_corpus": sorted(ids - corpus_ids),
            "source_counts": dict(sorted(Counter(members[source_id][0] for source_id in ids & corpus_ids).items())),
        }
    if scope_ids["evaluation_pool"] != scope_ids["retained_panel_sources"]:
        raise ProvenanceError("Evaluation pool and retained panel source membership differ")
    if not scope_ids["common_core"] <= scope_ids["evaluation_pool"] <= scope_ids["released_sources"]:
        raise ProvenanceError("Frozen core/pool/released source nesting is inconsistent")
    result = {
        "status": "verified_eligible_corpus_reconstruction", "schema_version": "1.0",
        "historical_inventory_sha256": INVENTORY_SHA256, "membership_manifest_sha256": _sha256(membership_bytes),
        "total_final_images_in_inventory": final_count,
        "eligible_training_corpus": {"source_count": len(members), "source_counts": EXPECTED_CORPUS_COUNTS},
        "evaluation_overlap": overlap,
        "limits": [
            "This verifies filename membership and the documented VinDr/BUU filter, not original image bytes.",
            "The eligible corpus is not the optimizer training partition; filenames alone do not authenticate historical split use or per-step sample consumption.",
            "Source-image overlap does not establish patient overlap or patient-level disjointness.",
        ],
    }
    if configured_splits is not None:
        split_bytes = configured_splits.read_bytes()
        split_scopes = {"common_core_1000": scope_ids["common_core"],
                        "evaluation_pool_2000": scope_ids["evaluation_pool"],
                        "released_sources_2950": scope_ids["released_sources"],
                        "source_filtered_generation_2987": _qc_source_ids((artifacts / "dataset/qc_metadata.jsonl").read_bytes())}
        result["configured_split_reconstruction"] = verify_configured_splits(_json(split_bytes), members, split_scopes)
        result["configured_split_reconstruction"]["manifest_sha256"] = _sha256(split_bytes)
        result["configured_split_reconstruction"]["attempted_qc_sha256"] = QC_SHA256
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-inventory", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--configured-splits", type=Path, help="Optionally verify the prescribed partition; requires NumPy, never trains a model.")
    args = parser.parse_args()
    try:
        result = audit(args.historical_inventory, args.membership, args.artifacts, args.configured_splits)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"Corpus provenance verification failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
