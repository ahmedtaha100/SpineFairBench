"""Negative provenance cases with synthetic names; no training or model calls."""

import copy
import hashlib
import unittest
from unittest.mock import patch

from scripts import verify_corpus_provenance as provenance


class CorpusProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.inventory = ((provenance.FINAL_IMAGE_PREFIX + "buu_case.dicom\n")
                          + (provenance.FINAL_IMAGE_PREFIX + "vindr_case.dcm\n")).encode()
        self.digest = hashlib.sha256(self.inventory).hexdigest()
        self.expected = {
            "buu_case": ("buu", provenance.FINAL_IMAGE_PREFIX + "buu_case.dicom"),
            "vindr_case": ("vindr", provenance.FINAL_IMAGE_PREFIX + "vindr_case.dcm"),
        }
        self.membership = {"source_sha256": self.digest, "total_final_images": 2,
                           "members": [{"source_id": key, "source": value[0], "historical_path": value[1]}
                                       for key, value in self.expected.items()]}

    def test_inventory_hash_and_duplicate_source_detection(self):
        with self.assertRaisesRegex(provenance.ProvenanceError, "SHA-256"):
            provenance.reconstruct_eligible_corpus(self.inventory)
        with patch.object(provenance, "INVENTORY_SHA256", self.digest), \
                patch.object(provenance, "EXPECTED_CORPUS_COUNTS", {"buu": 1, "vindr": 1}):
            self.assertEqual(provenance.reconstruct_eligible_corpus(self.inventory), (self.expected, 2))
        duplicate = self.inventory + (provenance.FINAL_IMAGE_PREFIX + "buu_case.dcm\n").encode()
        with patch.object(provenance, "INVENTORY_SHA256", hashlib.sha256(duplicate).hexdigest()), \
                self.assertRaisesRegex(provenance.ProvenanceError, "Duplicate eligible source_id"):
            provenance.reconstruct_eligible_corpus(duplicate)

    def test_membership_rejects_duplicates_missing_rows_wrong_identity_and_count(self):
        with patch.object(provenance, "INVENTORY_SHA256", self.digest):
            provenance.verify_membership(self.membership, self.expected, 2)
            mutations = [
                (lambda obj: obj["members"].append(obj["members"][0]), "Duplicate membership"),
                (lambda obj: obj["members"].pop(), "records differ"),
                (lambda obj: obj["members"][0].update(source="vindr"), "Invalid membership"),
                (lambda obj: obj["members"][0].update(historical_path="different.dicom"), "records differ"),
                (lambda obj: obj.update(total_final_images=True), "total_final_images"),
                (lambda obj: obj.update(source_sha256="0" * 64), "pinned historical inventory"),
            ]
            for mutate, message in mutations:
                obj = copy.deepcopy(self.membership)
                mutate(obj)
                with self.subTest(message=message), self.assertRaisesRegex(provenance.ProvenanceError, message):
                    provenance.verify_membership(obj, self.expected, 2)

    def test_duplicate_json_keys_and_scope_ids_are_rejected(self):
        with self.assertRaisesRegex(provenance.ProvenanceError, "Duplicate JSON key"):
            provenance._json(b'{"members":[],"members":[]}')
        for values, message in [(["case", "case"], "Duplicate source_id"), (["case", None], "Invalid source_id")]:
            with self.subTest(values=values), self.assertRaisesRegex(provenance.ProvenanceError, message):
                provenance._source_ids({"source_ids": values, "actual_n": 2}, "fixture", 2)

    def test_configured_split_checks_order_identity_and_overlap(self):
        members = {f"buu_{index:02d}": ("buu", provenance.FINAL_IMAGE_PREFIX + f"buu_{index:02d}."
                   + ("dcm" if index % 2 else "dicom")) for index in range(10)}
        # RandomState(42) permutes ten indices as 8,1,5,0,7,2,9,4,3,6.
        # The historical loader puts all sorted .dcm paths before .dicom paths.
        expected = {"train": [f"buu_{index:02d}" for index in (6, 3, 0, 1, 4, 5, 8, 9)],
                    "val": ["buu_07"], "test": ["buu_02"]}
        self.assertEqual(provenance.configured_split_ids(members), expected)
        scopes = {"fixture": {"buu_06", "buu_07"}}
        payload = {**provenance.CONFIGURED_SPLIT_IDENTITIES, "schema_version": "1.0",
                   "record_type": "configured_split_reconstruction", "corpus_count": 10,
                   "recipe": {"rng": "numpy.random.RandomState", "seed": 42, "train_ratio": .8,
                              "val_ratio": .1, "test_ratio": .1, "operation": "permutation(10)",
                              "training_batch_size": 16, "training_drop_last": True},
                   "splits": {name: {"count": len(ids), "source_ids": ids,
                                     "source_counts": {"buu": len(ids)},
                                     "overlap_counts": {"fixture": len(set(ids) & scopes["fixture"])}}
                              for name, ids in expected.items()}}
        result = provenance.verify_configured_splits(payload, members, scopes)
        self.assertEqual(result["splits"]["train"]["count"], 8)
        for mutate, message in [
            (lambda obj: obj["recipe"].update(rng="numpy.random.default_rng"), "recipe mismatch"),
            (lambda obj: obj["splits"]["train"]["source_ids"].reverse(), "membership/order/count"),
            (lambda obj: obj["splits"]["train"]["source_ids"].append("buu_06"), "duplicate configured split"),
            (lambda obj: obj["splits"]["val"]["overlap_counts"].update(fixture=0), "source/overlap counts"),
            (lambda obj: obj.update(source_patch_sha256="0" * 64), "provenance identity mismatch"),
        ]:
            mutated = copy.deepcopy(payload)
            mutate(mutated)
            with self.subTest(message=message), self.assertRaisesRegex(provenance.ProvenanceError, message):
                provenance.verify_configured_splits(mutated, members, scopes)


if __name__ == "__main__":
    unittest.main()
