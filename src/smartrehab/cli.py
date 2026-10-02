"""Command line entry point: ``python -m smartrehab <stage> ...`` (or the ``smartrehab`` script).

Dataset stages (run once per dataset, each reads the CSV written by the previous one)::

    bbox       -> <out>/mp.csv          MediaPipe hand bounding box
    yolo       -> <out>/mp_yolo.csv     + YOLO object bounding box
    landmarks  -> <out>/mp_yolo_lm.csv  + 21 hand landmarks relative to the object
    depth      -> <out>/results.csv     + MiDaS depth difference (needs the .pfm depth maps)

Then ``combine`` merges the ``results.csv`` of several datasets, ``train`` fits the classifier and ``evaluate``
scores a saved model.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from .config import load_dataset_config, load_train_config

STAGE_FILES = {"bbox": "mp.csv", "yolo": "mp_yolo.csv", "landmarks": "mp_yolo_lm.csv", "depth": "results.csv"}


def _dataset_stage(args: argparse.Namespace) -> None:
    cfg = load_dataset_config(args.config)
    out_dir = args.out or os.path.join("outputs", cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "debug") if args.debug_images else None

    def read(stage: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(out_dir, STAGE_FILES[stage]))

    if args.stage == "bbox":
        from .hands import detect_hand_boxes
        result = detect_hand_boxes(cfg, debug_dir)
    elif args.stage == "yolo":
        from .objects import detect_objects
        result = detect_objects(cfg, read("bbox"), debug_dir)
    elif args.stage == "landmarks":
        from .hands import add_landmarks
        result = add_landmarks(cfg, read("yolo"), debug_dir)
    else:
        from .depth import add_depth
        result = add_depth(cfg, read("landmarks"), workers=args.workers)

    path = os.path.join(out_dir, STAGE_FILES[args.stage])
    result.to_csv(path, index=False)
    print(f"Wrote {len(result)} rows to {path}")


def _combine(args: argparse.Namespace) -> None:
    from .features import combine_results

    combined = combine_results([pd.read_csv(p) for p in args.inputs])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined)} rows to {args.output}")


def _train(args: argparse.Namespace) -> None:
    from .train import train

    cfg = load_train_config(args.config)
    if args.data:
        cfg.data_csv = args.data
    train(cfg)


def _evaluate(args: argparse.Namespace) -> None:
    from .evaluate import evaluate_checkpoint

    evaluate_checkpoint(args.checkpoint, args.data, args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smartrehab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    for stage in STAGE_FILES:
        p = sub.add_parser(stage, help=f"dataset stage writing {STAGE_FILES[stage]}")
        p.add_argument("--config", required=True, help="dataset config, e.g. configs/ek.yaml")
        p.add_argument("--out", help="output directory (default: outputs/<dataset name>)")
        p.add_argument("--debug-images", action="store_true", help="also save annotated frames under <out>/debug")
        if stage == "depth":
            p.add_argument("--workers", type=int, default=1, help="processes used to read the depth maps")
        p.set_defaults(func=_dataset_stage)

    p = sub.add_parser("combine", help="merge the results.csv files of several datasets")
    p.add_argument("inputs", nargs="+", help="results.csv files")
    p.add_argument("--output", default="outputs/combined.csv")
    p.set_defaults(func=_combine)

    p = sub.add_parser("train", help="train the grasp classifier")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--data", help="overrides data_csv of the config")
    p.set_defaults(func=_train)

    p = sub.add_parser("evaluate", help="evaluate a saved model on a result table")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="result table, same columns as the training table")
    p.add_argument("--out", default="outputs/evaluation")
    p.set_defaults(func=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
