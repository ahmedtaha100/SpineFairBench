from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pathlib import Path

from spinefairbench.data.dataset import SpineXRDataset

logger = logging.getLogger(__name__)


def _spine_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    elem = batch[0]
    result: dict[str, Any] = {}
    for key in elem:
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def create_dataloader(
    data_root: Path,
    annotations_csv: Path | None = None,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 512,
    normalize_range: str = "0_1",
    use_clahe: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    cache_dir: Path | None = None,
    source_filter: list[str] | None = None,
) -> DataLoader:
    dataset = SpineXRDataset(
        data_root=data_root,
        annotations_csv=annotations_csv,
        image_size=image_size,
        normalize_range=normalize_range,
        use_clahe=use_clahe,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        cache_dir=cache_dir,
        source_filter=source_filter,
    )

    shuffle = split == "train"

    logger.info(
        "DataLoader [%s]: batch_size=%d, num_workers=%d, shuffle=%s",
        split,
        batch_size,
        num_workers,
        shuffle,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        collate_fn=_spine_collate,
    )
