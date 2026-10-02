"""YOLO stage (step 2): pick the object the hand is interacting with.

Among the YOLOv8 detections of the configured classes, the object with the largest overlap with the hand box is
chosen; if no object overlaps the hand enough, the object whose centre is closest to the hand centre is chosen;
if nothing is detected the object box is all zeros (the landmark stage then substitutes a phantom object).
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DatasetConfig
from .debug import save_debug_image
from .frames import locate_image

log = logging.getLogger(__name__)

#: An overlap (in pixels, ``interArea``) must exceed this to count as "the hand touches the object".
MIN_OVERLAP = 5000

OUTPUT_COLUMNS = [
    "hand_min_x", "hand_min_y", "hand_width", "hand_height",
    "object_min_x", "object_min_y", "object_width", "object_height",
    "handedness", "picture_name", "grasp",
]


def yolo_to_pixel_box(xc: float, yc: float, w: float, h: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Normalised YOLO centre box -> pixel ``(min_x, min_y, box_width, box_height)``, clipped at the top-left."""
    box_w = int(np.round(w * width))
    min_x = max(0, int(np.round(xc * width)) - box_w // 2)
    max_x = int(np.round(xc * width)) + box_w // 2

    box_h = int(np.round(h * height))
    min_y = max(0, int(np.round(yc * height)) - box_h // 2)
    max_y = int(np.round(yc * height)) + box_h // 2
    return min_x, min_y, max_x - min_x, max_y - min_y


def select_object(hand_box: Sequence[int], detections: Sequence[Sequence[int]], width: int, height: int) -> list[int]:
    """Choose the object box ``[min_x, min_y, box_width, box_height]`` for a hand, or zeros if there are none."""
    hand_min_x, hand_min_y = hand_box[0], hand_box[1]
    hand_max_x, hand_max_y = hand_min_x + hand_box[2], hand_min_y + hand_box[3]
    hand_center = (hand_min_x + hand_box[2] // 2, hand_min_y + hand_box[3] // 2)

    best_overlap, best_box = MIN_OVERLAP, None
    closest_distance, closest_box = width * height, [0, 0, 0, 0]
    for min_x, min_y, box_w, box_h in detections:
        max_x, max_y = min_x + box_w, min_y + box_h
        center = ((min_x + max_x) // 2, (min_y + max_y) // 2)

        inter_w = min(hand_max_x, max_x) - max(hand_min_x, min_x) + 1
        inter_h = min(hand_max_y, max_y) - max(hand_min_y, min_y) + 1
        overlap = max(0, inter_w) * max(0, inter_h)
        distance = np.hypot(center[0] - hand_center[0], center[1] - hand_center[1])

        if overlap > best_overlap:
            best_overlap, best_box = overlap, [min_x, min_y, box_w, box_h]
        if distance < closest_distance:
            closest_distance, closest_box = distance, [min_x, min_y, box_w, box_h]

    return best_box if best_box is not None else closest_box


def detect_objects(cfg: DatasetConfig, mp_df: pd.DataFrame, debug_dir: Optional[str] = None) -> pd.DataFrame:
    """Step 2: add the object box next to each hand box."""
    from ultralytics import YOLO

    model = YOLO(cfg.objects.weights)
    predict_args = {"classes": cfg.objects.classes, "verbose": False}
    if cfg.objects.device is not None:
        predict_args["device"] = cfg.objects.device
    if cfg.objects.conf is not None:
        predict_args["conf"] = cfg.objects.conf

    rows, missing, detected = [], 0, 0
    for _, row in tqdm(mp_df.iterrows(), total=len(mp_df), desc=f"{cfg.name}: objects"):
        name, grasp = row["picture_name"], row["grasp"]
        hand_box = [int(row[c]) for c in OUTPUT_COLUMNS[:4]]

        image = locate_image(name, grasp, cfg.objects.image_dir)
        object_box = [0, 0, 0, 0]
        if image is None:
            missing += 1
        else:
            boxes = model.predict(image, **predict_args)[0].boxes.xywhn.cpu().numpy()
            detections = [yolo_to_pixel_box(*b, cfg.width, cfg.height) for b in boxes]
            if detections:
                detected += 1
                object_box = select_object(hand_box, detections, cfg.width, cfg.height)
            if debug_dir:
                save_debug_image(image, os.path.join(debug_dir, "yolo", os.path.splitext(os.path.basename(name))[0] + ".png"),
                                 hand_box, object_box)
        rows.append(hand_box + object_box + [row["handedness"], name, grasp])

    if missing:
        log.warning("%s: %d frames not found under objects.image_dir; they got an empty object box", cfg.name, missing)
    log.info("%s: objects detected in %d of %d frames", cfg.name, detected, len(rows))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
