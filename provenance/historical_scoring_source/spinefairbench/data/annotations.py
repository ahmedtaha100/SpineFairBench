from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from spinefairbench.exceptions import DataError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

ABNORMALITY_CATEGORIES = [
    "Osteophytes",
    "Foraminal stenosis",
    "Disc space narrowing",
    "Spondylolysthesis",
    "Surgical implant",
    "Vertebral collapse",
    "Other lesions",
    "No finding",
    "Posterior osteophyte",
    "Endplate sclerosis",
    "Schmorl's node",
    "Disc calcification",
    "Fracture",
]

CATEGORY_TO_INDEX = {cat: idx for idx, cat in enumerate(ABNORMALITY_CATEGORIES)}
INDEX_TO_CATEGORY = {idx: cat for cat, idx in CATEGORY_TO_INDEX.items()}
NUM_CATEGORIES = len(ABNORMALITY_CATEGORIES)


def parse_annotations(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise DataError(f"Annotation file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise DataError(f"Failed to read annotation CSV: {e}") from e

    required_columns = {"image_id", "lesion_type"}
    missing = required_columns - set(df.columns)
    if missing:
        raise DataError(f"Missing required columns in annotation CSV: {missing}")

    unknown = set(df["lesion_type"].unique()) - set(ABNORMALITY_CATEGORIES)
    if unknown:
        logger.warning("Unknown abnormality categories found: %s", unknown)

    logger.info(
        "Parsed %d annotations for %d unique images",
        len(df),
        df["image_id"].nunique(),
    )
    return df


def get_label_vector(image_id: str, annotations_df: pd.DataFrame) -> np.ndarray:
    label_vec: np.ndarray = np.zeros(NUM_CATEGORIES, dtype=np.float32)
    image_annots = annotations_df[annotations_df["image_id"] == image_id]
    for _, row in image_annots.iterrows():
        lesion = row["lesion_type"]
        if lesion in CATEGORY_TO_INDEX:
            label_vec[CATEGORY_TO_INDEX[lesion]] = 1.0
    return label_vec


def get_image_labels(image_id: str, annotations_df: pd.DataFrame) -> list[str]:
    image_annots = annotations_df[annotations_df["image_id"] == image_id]
    labels = []
    for _, row in image_annots.iterrows():
        lesion = row["lesion_type"]
        if lesion in CATEGORY_TO_INDEX and lesion not in labels:
            labels.append(lesion)
    return labels


def get_bounding_boxes(
    image_id: str, annotations_df: pd.DataFrame
) -> list[dict[str, float]]:
    image_annots = annotations_df[annotations_df["image_id"] == image_id]
    boxes = []
    for _, row in image_annots.iterrows():
        if all(col in row.index for col in ["x_min", "y_min", "x_max", "y_max"]):
            boxes.append({
                "lesion_type": row["lesion_type"],
                "x_min": float(row["x_min"]),
                "y_min": float(row["y_min"]),
                "x_max": float(row["x_max"]),
                "y_max": float(row["y_max"]),
            })
    return boxes
