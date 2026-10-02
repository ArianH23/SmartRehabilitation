"""Evaluation: predictions, F1 scores, per-sample validation table and confusion matrices."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from .features import CLASSES, N_FEATURES, normalize, split_features  # noqa: E402
from .model import Net  # noqa: E402

log = logging.getLogger(__name__)


def predict(model: Net, features: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(features).argmax(dim=1).numpy()


def report(y_true, y_pred, meta: pd.DataFrame, out_dir: str | Path) -> dict:
    """Print F1 scores and write the per-sample table and the confusion matrices to ``out_dir``.

    ``meta`` holds ``picture_name`` and ``handedness`` for each sample, in the same order as the predictions.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [c.capitalize() for c in CLASSES]
    table = meta[["picture_name", "handedness"]].copy().reset_index(drop=True)
    table["Prediction"] = [labels[i] for i in y_pred]
    table["True"] = [labels[i] for i in y_true]
    counts = table["picture_name"].map(table["picture_name"].value_counts())
    table = table.assign(counts=counts).sort_values(["counts", "picture_name"], ascending=False).drop(columns="counts")
    table["Equal"] = np.where(table["Prediction"] == table["True"], " ", "X")
    table.to_csv(out_dir / "validation_results_sorted.csv", index=False)

    scores = {
        "F1": f1_score(y_true, y_pred, average=None),
        "F1-micro": f1_score(y_true, y_pred, average="micro"),
        "F1-macro": f1_score(y_true, y_pred, average="macro"),
        "F1-weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    print("0:none 1:power 2:precision", dict(Counter(int(v) for v in y_true)))
    for key, value in scores.items():
        print(key, value)

    counts_matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    # Row-normalised: each row (true class) sums to 1.
    percent_matrix = counts_matrix / counts_matrix.sum(axis=1, keepdims=True)
    for matrix, fmt, filename in ((percent_matrix, ".2f", "confusion_matrix_percent.png"),
                                  (counts_matrix, "d", "confusion_matrix_counts.png")):
        plt.figure(figsize=(12, 7))
        sns.heatmap(pd.DataFrame(matrix, index=CLASSES, columns=CLASSES), annot=True, cmap="Blues", fmt=fmt)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(out_dir / filename)
        plt.close()
    return scores


def evaluate_checkpoint(checkpoint_path: str | Path, data_csv: str | Path, out_dir: str | Path,
                        hidden_dims: tuple[int, int] = (64, 32)) -> dict:
    """Evaluate a saved model on a result table (same columns as the training table)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Net(N_FEATURES, *hidden_dims, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])

    features, meta = split_features(pd.read_csv(data_csv))
    depth_max = checkpoint.get("depth_max")
    if depth_max is None:
        # Checkpoints from the thesis do not store the training depth scale; the thesis evaluation normalised
        # with the largest depth difference of the evaluated set itself.
        depth_max = features["depth_dist"].abs().max()
        log.warning("Checkpoint has no depth_max; using the evaluation set's own maximum (%.4f)", depth_max)

    x = torch.tensor(normalize(features, meta["picture_name"], depth_max).to_numpy(), dtype=torch.float32)
    y_true = meta["grasp"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return report(y_true, predict(model, x), meta, out_dir)
