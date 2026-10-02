"""Per-dataset and training configuration, loaded from YAML (see ``configs/``)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Variant:
    """One resolution variant of a frame that MediaPipe is run on.

    kind: "plain" (resized frame), "pad" (padded to a square) or "crop" (centre crop).
    dir: directory template holding the variant; None means the frame's own path.
    """

    name: str
    kind: str = "plain"
    dir: Optional[str] = None


@dataclass
class FramesConfig:
    kind: str                       # "walk" (scan a directory tree) or "csv" (frame list)
    handedness_from: str            # "path", "name" or "right_only"
    root: Optional[str] = None
    csv: Optional[str] = None
    filename_column: str = "Filename"
    category_column: str = "SmallCategories"


@dataclass
class HandsConfig:
    min_detection_confidence: float
    variants: list[Variant]
    pick: str = "last"              # which matching hand provides the box: "first" or "last"


@dataclass
class ObjectsConfig:
    classes: list[int]
    weights: str = "yolov8x6.pt"
    device: Optional[object] = None
    conf: Optional[float] = None
    image_dir: Optional[str] = None


@dataclass
class DepthConfig:
    dir: str
    suffix: str = "-dpt_beit_large_512.pfm"
    strip_prefix_chars: int = 0


@dataclass
class DatasetConfig:
    name: str
    width: int
    height: int
    frames: FramesConfig
    hands: HandsConfig
    objects: ObjectsConfig
    depth: DepthConfig
    landmark_variants: list[Variant] = field(default_factory=list)


@dataclass
class TrainConfig:
    data_csv: str
    output_dir: str = "outputs/training"
    seed: int = 42
    test_size: float = 0.15
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 0.01
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    class_weights: list[float] = field(default_factory=lambda: [0.6382, 0.64986, 0.71198])


def _variants(raw: list[dict]) -> list[Variant]:
    return [Variant(name=str(v["name"]), kind=v.get("kind", "plain"), dir=v.get("dir")) for v in raw]


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    hands = HandsConfig(
        min_detection_confidence=raw["hands"]["min_detection_confidence"],
        variants=_variants(raw["hands"]["variants"]),
        pick=raw["hands"].get("pick", "last"),
    )
    landmark_raw = (raw.get("landmarks") or {}).get("variants")
    return DatasetConfig(
        name=raw["name"],
        width=raw["frame_size"]["width"],
        height=raw["frame_size"]["height"],
        frames=FramesConfig(**raw["frames"]),
        hands=hands,
        objects=ObjectsConfig(**raw["objects"]),
        depth=DepthConfig(**raw["depth"]),
        landmark_variants=_variants(landmark_raw) if landmark_raw else hands.variants,
    )


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
