import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from smartrehab.config import TrainConfig  # noqa: E402
from smartrehab.evaluate import evaluate_checkpoint  # noqa: E402
from smartrehab.train import train  # noqa: E402

from test_features import make_table  # noqa: E402


def test_train_and_evaluate_smoke(tmp_path):
    csv = tmp_path / "combined.csv"
    make_table(120).to_csv(csv, index=False)

    cfg = TrainConfig(data_csv=str(csv), output_dir=str(tmp_path / "out"), epochs=2)
    checkpoint = train(cfg)

    assert checkpoint.exists() and ":" not in checkpoint.name
    for name in ("validation_results_sorted.csv", "confusion_matrix_percent.png", "confusion_matrix_counts.png"):
        assert (tmp_path / "out" / name).exists()

    scores = evaluate_checkpoint(checkpoint, csv, tmp_path / "eval")
    assert 0.0 <= scores["F1-macro"] <= 1.0
    assert len(pd.read_csv(tmp_path / "eval" / "validation_results_sorted.csv")) == 120
