"""Frame discovery and image lookup shared by the pipeline stages."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd

from .config import FramesConfig

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
GRASPS = ("none", "precision", "power")


@dataclass
class Frame:
    name: str                         # value written to the `picture_name` column
    path: str
    grasp: str
    hand_hint: Optional[str]          # text the expected hand label is looked up in (None: right hands only)


def grasp_from_text(text: str) -> str:
    if "power" in text:
        return "power"
    if "precision" in text:
        return "precision"
    return "none"


def iter_frames(cfg: FramesConfig) -> Iterator[Frame]:
    """Enumerate the frames of a dataset in a deterministic order."""
    if cfg.kind == "walk":
        for path, subdirs, files in os.walk(cfg.root):
            subdirs.sort()
            for name in sorted(files):
                if not name.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                hint = {"path": path, "name": name}.get(cfg.handedness_from)
                yield Frame(name, os.path.join(path, name), grasp_from_text(path), hint)
    elif cfg.kind == "csv":
        table = pd.read_csv(cfg.csv)
        for _, row in table.iterrows():
            name = row[cfg.filename_column]
            yield Frame(name, name, grasp_from_text(str(row[cfg.category_column])), None)
    else:
        raise ValueError(f"Unknown frames kind: {cfg.kind!r}")


def locate_image(name: str, grasp: str, directory: Optional[str]) -> Optional[str]:
    """Find a frame by name inside ``directory`` (a template that may contain ``{grasp}``).

    A missing directory means ``name`` is already a usable path. With a ``{grasp}`` template the frame's own
    class is tried first, then the other classes. Returns None if the frame cannot be found.
    """
    if not directory:
        return name if os.path.exists(name) else None
    grasps = [grasp] + [g for g in GRASPS if g != grasp] if "{grasp}" in directory else [grasp]
    for g in grasps:
        candidate = os.path.join(directory.format(grasp=g), name)
        if os.path.exists(candidate):
            return candidate
    return None
