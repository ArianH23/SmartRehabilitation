"""MediaPipe Hands stages: hand bounding box (step 1) and hand-landmark features (step 3).

MediaPipe is run on the horizontally flipped frame (it reports the handedness of the mirrored image), so all
coordinates are mirrored back to the original frame before they are stored.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DatasetConfig, Variant
from .debug import FINGERTIP_LANDMARKS, save_debug_image
from .frames import Frame, iter_frames, locate_image

log = logging.getLogger(__name__)

HAND_BOX_COLUMNS = ["hand_min_x", "hand_min_y", "hand_width", "hand_height", "handedness", "picture_name", "grasp"]


def to_frame_pixels(xs, ys, kind: str, width: int, height: int):
    """Map landmarks normalised to a resolution variant onto pixels of the original (still mirrored) frame.

    A centre crop (``crop``) keeps the full height and loses (width - height) / 2 pixels on each side; a
    square pad (``pad``) is ``width`` wide and ``width`` tall with (width - height) / 2 pixels of padding.
    """
    offset = (width - height) // 2
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    px = xs * height + offset if kind == "crop" else xs * width
    py = ys * width - offset if kind == "pad" else ys * height
    return px, py


def hand_bbox(xs, ys, kind: str, width: int, height: int) -> list[int]:
    """Hand bounding box ``[min_x, min_y, box_width, box_height]`` of the 21 landmarks, in original-frame pixels."""
    px, py = to_frame_pixels(xs, ys, kind, width, height)
    limit = max(width, height)
    max_x, max_y = max(0.0, px.max()), max(0.0, py.max())
    min_x, min_y = min(limit, px.min()), min(limit, py.min())

    # Undo the horizontal mirroring.
    min_x = int(np.round(width - min_x))
    max_x = int(np.round(width - max_x))
    min_x, max_x = max_x, min_x

    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(width, max_x), min(height, max_y)
    return [min_x, int(np.round(min_y)), max_x - min_x, int(np.round(max_y)) - int(np.round(min_y))]


def landmark_features(xs, ys, zs, kind: str, width: int, height: int, object_center) -> list:
    """Per-landmark ``(object_x - x, object_y - y, z)`` triplets, flattened (63 values for 21 landmarks)."""
    px, py = to_frame_pixels(xs, ys, kind, width, height)
    px = width - px
    features: list = []
    for x, y, z in zip(px, py, zs):
        features.extend([object_center[0] - int(round(x)), object_center[1] - int(round(y)), z])
    return features


def phantom_object_center(hand_box, width: int, height: int) -> tuple[int, int]:
    """Centre of the imaginary object used when YOLO finds nothing: the frame corner opposite to the hand."""
    center_x = hand_box[0] + hand_box[2] // 2
    center_y = hand_box[1] + hand_box[3] // 2
    return (width if center_x < width // 2 else 0), (height if center_y < height // 2 else 0)


def open_hands(min_detection_confidence: float):
    import mediapipe as mp  # imported lazily so the geometry above works without MediaPipe installed

    return mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=min_detection_confidence
    )


def _labels(results) -> list[str]:
    return [h.classification[0].label.lower() for h in results.multi_handedness]


def find_hand(hands, candidates: Iterable[tuple[Variant, Optional[str]]], matches: Callable[[str], bool]):
    """Run MediaPipe on each ``(variant, image_path)`` until one of the first two hands satisfies ``matches``.

    Returns ``(variant, results)`` or None. Unreadable images are skipped.
    """
    for variant, path in candidates:
        image = cv2.imread(path) if path else None
        if image is None:
            continue
        results = hands.process(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks and any(matches(label) for label in _labels(results)[:2]):
            return variant, results
    return None


def _hand_matcher(frame: Frame) -> Callable[[str], bool]:
    if frame.hand_hint is None:           # right_only datasets (Yale)
        return lambda label: label == "right"
    return lambda label: label in frame.hand_hint


def detect_hand_boxes(cfg: DatasetConfig, debug_dir: Optional[str] = None) -> pd.DataFrame:
    """Step 1: find the expected hand in each frame and store its bounding box."""
    rows, used, no_detection = [], Counter(), 0
    with open_hands(cfg.hands.min_detection_confidence) as hands:
        for frame in tqdm(list(iter_frames(cfg.frames)), desc=f"{cfg.name}: hand boxes"):
            matches = _hand_matcher(frame)
            candidates = (
                (v, frame.path if v.dir is None else locate_image(frame.name, frame.grasp, v.dir))
                for v in cfg.hands.variants
            )
            found = find_hand(hands, candidates, matches)
            if found is None:
                no_detection += 1
                continue
            variant, results = found

            boxes = [
                (hand_bbox([m.x for m in lm.landmark], [m.y for m in lm.landmark], variant.kind, cfg.width, cfg.height), label)
                for lm, label in zip(results.multi_hand_landmarks, _labels(results))
                if matches(label)
            ]
            if not boxes:
                no_detection += 1
                continue
            box, label = boxes[-1] if cfg.hands.pick == "last" else boxes[0]
            rows.append(box + [label, frame.name, frame.grasp])
            used[variant.name] += 1
            if debug_dir:
                save_debug_image(frame.path, os.path.join(debug_dir, "bbox", os.path.splitext(frame.name)[0] + ".png"), box)

    log.info("%s: %d boxes, %d frames without the expected hand; resolution used: %s",
             cfg.name, len(rows), no_detection, dict(used))
    return pd.DataFrame(rows, columns=HAND_BOX_COLUMNS)


def add_landmarks(cfg: DatasetConfig, mp_yolo: pd.DataFrame, debug_dir: Optional[str] = None) -> pd.DataFrame:
    """Step 3: append per-landmark distance to the object (plus MediaPipe's z) for each hand/object row.

    Rows whose hand is not found again are dropped.
    """
    from .features import HAND_BOX, LANDMARKS, META, OBJECT_BOX

    columns = HAND_BOX + OBJECT_BOX + LANDMARKS + META
    rows, no_detection = [], 0
    with open_hands(cfg.hands.min_detection_confidence) as hands:
        for _, row in tqdm(mp_yolo.iterrows(), total=len(mp_yolo), desc=f"{cfg.name}: landmarks"):
            handedness, name, grasp = row["handedness"], row["picture_name"], row["grasp"]
            candidates = (
                (v, locate_image(name, grasp, v.dir) if v.dir is not None else name)
                for v in cfg.landmark_variants
            )
            found = find_hand(hands, candidates, lambda label: label in handedness)
            if found is None:
                no_detection += 1
                continue
            variant, results = found

            hand_box = [row[c] for c in HAND_BOX]
            object_box = [row[c] for c in OBJECT_BOX]
            if object_box[2] == 0:
                object_center = phantom_object_center(hand_box, cfg.width, cfg.height)
                object_box[0], object_box[1] = object_center
            else:
                object_center = (object_box[0] + object_box[2] // 2, object_box[1] + object_box[3] // 2)

            # The first detected hand with the expected handedness provides the landmarks.
            for lm, label in zip(results.multi_hand_landmarks, _labels(results)):
                if label != handedness:
                    continue
                marks = lm.landmark
                features = landmark_features(
                    [m.x for m in marks], [m.y for m in marks], [m.z for m in marks],
                    variant.kind, cfg.width, cfg.height, object_center,
                )
                rows.append(hand_box + object_box + features + [handedness, name, grasp])
                if debug_dir:
                    tips = [
                        ((object_center[0] - features[3 * (i - 1)], object_center[1] - features[3 * (i - 1) + 1]), object_center)
                        for i in FINGERTIP_LANDMARKS
                    ]
                    save_debug_image(
                        name if variant.dir is None else locate_image(name, grasp, variant.dir),
                        os.path.join(debug_dir, "landmarks", os.path.splitext(os.path.basename(name))[0] + ".png"),
                        hand_box, row[OBJECT_BOX].tolist(), tips,
                    )
                break

    log.info("%s: %d rows with landmarks, %d frames where the hand was not found again", cfg.name, len(rows), no_detection)
    return pd.DataFrame(rows, columns=columns)
