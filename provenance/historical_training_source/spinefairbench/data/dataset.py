from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    import pandas as pd
from torch.utils.data import Dataset

from spinefairbench.data.annotations import (
    get_label_vector,
    parse_annotations,
)
from spinefairbench.data.dicom_loader import load_dicom
from spinefairbench.data.preprocessing import preprocess_pipeline
from spinefairbench.exceptions import DataError

logger = logging.getLogger(__name__)

AGE_NORMALIZE_MAX = 90.0


class SpineXRDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        annotations_csv: Path | None = None,
        image_size: int = 512,
        normalize_range: str = "0_1",
        use_clahe: bool = False,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        cache_dir: Path | None = None,
        source_filter: list[str] | None = None,
    ) -> None:
        self._data_root = Path(data_root)
        self._image_size = image_size
        self._normalize_range = normalize_range
        self._use_clahe = use_clahe
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._split = split

        if not self._data_root.exists():
            raise DataError(f"Data root not found: {self._data_root}")

        dicom_files = sorted(self._data_root.glob("**/*.dcm")) + sorted(
            self._data_root.glob("**/*.dicom")
        )
        if source_filter:
            prefixes = tuple(p + "_" for p in source_filter)
            dicom_files = [f for f in dicom_files if f.name.startswith(prefixes)]
        if not dicom_files:
            raise DataError(f"No DICOM files found in {self._data_root}")

        self._annotations_df: pd.DataFrame | None = None
        if annotations_csv is not None and annotations_csv.exists():
            self._annotations_df = parse_annotations(annotations_csv)

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(dicom_files))
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train : n_train + n_val]
        elif split == "test":
            selected = indices[n_train + n_val :]
        else:
            raise DataError(f"Invalid split: {split}. Must be train/val/test")

        self._files = [dicom_files[i] for i in selected]

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("SpineXRDataset [%s]: %d images", split, len(self._files))

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self._files[idx]

        cached = self._load_cached(path)
        if cached is not None:
            return cached

        try:
            record = load_dicom(path)
            image = preprocess_pipeline(
                record.pixel_data,
                target_size=self._image_size,
                bits_stored=record.bits_stored,
                normalize_range=self._normalize_range,
                use_clahe=self._use_clahe,
            )
        except Exception as e:
            logger.warning("Skipping corrupted DICOM %s: %s", path.stem, e)
            fallback_idx = (idx + 1) % len(self._files)
            if fallback_idx == idx:
                raise DataError("No valid DICOM files available") from e
            return self.__getitem__(fallback_idx)

        age_normalized = 0.5
        if record.age is not None:
            age_normalized = float(record.age) / AGE_NORMALIZE_MAX

        sex_encoded = 0
        if record.sex == "F":
            sex_encoded = 1

        pathology_labels: list[int] = []
        if self._annotations_df is not None:
            label_vec = get_label_vector(record.image_id, self._annotations_df)
            pathology_labels = [int(i) for i in np.where(label_vec > 0)[0]]

        result = {
            "image": image,
            "age": torch.tensor(age_normalized, dtype=torch.float32),
            "sex": torch.tensor(sex_encoded, dtype=torch.long),
            "pathology_labels": pathology_labels,
            "study_id": record.study_id,
            "image_id": record.image_id,
        }

        self._save_cached(path, result)
        return result

    def _cache_path(self, dicom_path: Path) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{dicom_path.stem}.pt"

    def _load_cached(self, dicom_path: Path) -> dict | None:
        cp = self._cache_path(dicom_path)
        if cp is None or not cp.exists():
            return None
        try:
            return torch.load(cp, weights_only=False)
        except Exception:
            return None

    def _save_cached(self, dicom_path: Path, data: dict) -> None:
        cp = self._cache_path(dicom_path)
        if cp is None:
            return
        try:
            torch.save(data, cp)
        except Exception as e:
            logger.warning("Failed to cache %s: %s", dicom_path.stem, e)
