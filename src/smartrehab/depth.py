"""Depth stage: hand/object depth difference from MiDaS ``.pfm`` depth maps."""
from __future__ import annotations

import logging
import os
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DatasetConfig
from .features import RESULT_COLUMNS

log = logging.getLogger(__name__)


def read_pfm(filename: str | Path) -> np.ndarray:
    """Read a PFM image (the format MiDaS writes its float depth maps in)."""
    with Path(filename).open("rb") as pfm_file:
        line1, line2, line3 = (pfm_file.readline().decode("latin-1").strip() for _ in range(3))
        assert line1 in ("PF", "Pf"), "not a PFM file"
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)
        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4, "unexpected PFM payload size"
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale


def _median(depth: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    region = depth[y1:y2, x1:x2]
    return float(np.median(region)) if region.size else float("nan")


def depth_difference(depth: np.ndarray, hand_box, object_box, width: int, height: int) -> float:
    """Median depth inside the object box minus median depth inside the hand box.

    Boxes are (min_x, min_y, box_width, box_height). Coordinates that sit on the far frame edge are moved one
    pixel inside. A zero-sized object box is the "phantom object" (no detection): the depth at its corner is used.
    """

    def corners(box):
        x1, y1 = box[0], box[1]
        x2, y2 = x1 + box[2], y1 + box[3]
        return (
            x1 - 1 if x1 == width else x1,
            y1 - 1 if y1 == height else y1,
            x2 - 1 if x2 == width else x2,
            y2 - 1 if y2 == height else y2,
        )

    hx1, hy1, hx2, hy2 = corners(hand_box)
    ox1, oy1, ox2, oy2 = corners(object_box)

    if object_box[2] == 0:
        object_depth = float(depth[oy1][ox1])
    else:
        object_depth = _median(depth, ox1, oy1, ox2, oy2)
    return object_depth - _median(depth, hx1, hy1, hx2, hy2)


def pfm_path(cfg: DatasetConfig, picture_name: str) -> str:
    stem = os.path.splitext(picture_name)[0][cfg.depth.strip_prefix_chars:]
    return os.path.join(cfg.depth.dir, stem + cfg.depth.suffix)


def _row_depth(args) -> float:
    cfg, row = args
    path = pfm_path(cfg, row["picture_name"])
    depth = read_pfm(path)
    hand = (row["hand_min_x"], row["hand_min_y"], row["hand_width"], row["hand_height"])
    obj = (row["object_min_x"], row["object_min_y"], row["object_width"], row["object_height"])
    return depth_difference(depth, hand, obj, cfg.width, cfg.height)


def add_depth(cfg: DatasetConfig, landmarks_df: pd.DataFrame, workers: int = 1) -> pd.DataFrame:
    """Append ``depth_dist`` to the landmark table and return it in the final column order."""
    jobs = [(cfg, row) for _, row in landmarks_df.iterrows()]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            values = list(tqdm(pool.map(_row_depth, jobs, chunksize=16), total=len(jobs), desc="depth"))
    else:
        values = [_row_depth(job) for job in tqdm(jobs, desc="depth")]

    result = landmarks_df.copy()
    result["depth_dist"] = values
    n_missing = int(result["depth_dist"].isna().sum())
    if n_missing:
        log.warning("%d rows have an empty hand box and no depth value; `combine` will drop them", n_missing)
    return result[RESULT_COLUMNS]
