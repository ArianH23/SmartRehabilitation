"""Optional debug images (boxes and landmark-to-object lines drawn on the frames)."""
from __future__ import annotations

import os
from typing import Optional, Sequence

import cv2

HAND_COLOR = (255, 0, 255)
OBJECT_COLOR = (255, 255, 0)
LINE_COLOR = (0, 255, 0)
FINGERTIP_LANDMARKS = (1, 5, 9, 13, 17, 21)   # wrist and the five finger tips (1-based)


def _rectangle(image, box, color):
    x, y, w, h = (int(v) for v in box)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)


def save_debug_image(
    frame_path: str,
    out_path: str,
    hand_box: Optional[Sequence[int]] = None,
    object_box: Optional[Sequence[int]] = None,
    lines: Sequence[tuple] = (),
) -> None:
    """Draw the hand box, object box and ``((x1, y1), (x2, y2))`` lines on a frame and save it."""
    image = cv2.imread(frame_path)
    if image is None:
        return
    if hand_box is not None:
        _rectangle(image, hand_box, HAND_COLOR)
    if object_box is not None and object_box[2] and object_box[3]:
        _rectangle(image, object_box, OBJECT_COLOR)
    for start, end in lines:
        cv2.line(image, tuple(int(v) for v in start), tuple(int(v) for v in end), LINE_COLOR, 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, image)
