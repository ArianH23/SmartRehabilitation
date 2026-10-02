"""Feature layout of the grasp classifier and its normalisation.

A pipeline result row holds 75 columns::

    hand box (4) | object box (4) | 21 landmarks x (dx, dy, z) (63) | depth_dist | handedness | picture_name | grasp

The classifier input has 74 features: the first 72 numeric columns plus the one-hot encoded handedness
(thesis section 4.7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HAND_BOX = ["hand_min_x", "hand_min_y", "hand_width", "hand_height"]
OBJECT_BOX = ["object_min_x", "object_min_y", "object_width", "object_height"]
LANDMARKS = [f"{i}{axis}" for i in range(1, 22) for axis in "xyz"]
DEPTH = "depth_dist"
META = ["handedness", "picture_name", "grasp"]

#: Columns written by the depth stage (and read by `combine` / `train` / `evaluate`).
RESULT_COLUMNS = HAND_BOX + OBJECT_BOX + LANDMARKS + [DEPTH] + META
#: Number of classifier inputs: box + landmark + depth columns plus the two handedness dummies.
N_FEATURES = len(HAND_BOX + OBJECT_BOX + LANDMARKS) + 1 + 2

CLASSES = ("none", "power", "precision")

#: Yale frames are 640x480; every other dataset is 1920x1080.
YALE_SIZE = (640, 480)
DEFAULT_SIZE = (1920, 1080)


def is_yale(picture_name: str) -> bool:
    """Yale frames are the only ones stored with a relative ``../data/...`` path."""
    return ".." in picture_name


def frame_sizes(picture_names: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (width, height) of the source frames."""
    yale = picture_names.map(is_yale).to_numpy()
    width = np.where(yale, YALE_SIZE[0], DEFAULT_SIZE[0])
    height = np.where(yale, YALE_SIZE[1], DEFAULT_SIZE[1])
    return width, height


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a result table into raw features (handedness one-hot encoded) and metadata
    (``picture_name``, ``grasp``, ``handedness``)."""
    missing = [c for c in RESULT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Result table is missing columns: {missing}")
    df = df[RESULT_COLUMNS]
    features = df.drop(columns=META)
    features["handedness_left"] = (df["handedness"] == "left").astype(float)
    features["handedness_right"] = (df["handedness"] == "right").astype(float)
    return features, df[["picture_name", "grasp", "handedness"]]


def normalize(features: pd.DataFrame, picture_names: pd.Series, depth_max: float) -> pd.DataFrame:
    """Scale pixel columns to [0, 1] by the frame size and depth by ``depth_max``.

    x / y distances are divided by the frame width / height, the landmark z value is left untouched,
    and the depth difference is divided by the largest absolute depth difference of the training set.
    """
    width, height = frame_sizes(picture_names.reset_index(drop=True))
    out = features.astype(float).copy()
    for cols in (HAND_BOX, OBJECT_BOX):
        out[cols[0]] /= width
        out[cols[1]] /= height
        out[cols[2]] /= width
        out[cols[3]] /= height
    for i in range(1, 22):
        out[f"{i}x"] /= width
        out[f"{i}y"] /= height
    out[DEPTH] /= depth_max
    return out


def combine_results(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the result tables of several datasets, dropping rows with missing values."""
    for table in tables:
        missing = [c for c in RESULT_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"Result table is missing columns: {missing}")
    combined = pd.concat([t[RESULT_COLUMNS] for t in tables], ignore_index=True)
    return combined.dropna()
