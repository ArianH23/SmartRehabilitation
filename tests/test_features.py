import numpy as np
import pandas as pd

from smartrehab.features import (
    CLASSES, N_FEATURES, RESULT_COLUMNS, combine_results, normalize, split_features,
)


def make_table(n=40, yale_share=0.25, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yale = rng.random(n) < yale_share
    width, height = np.where(yale, 640, 1920), np.where(yale, 480, 1080)
    data = {}
    for prefix in ("hand", "object"):
        data[f"{prefix}_min_x"] = (rng.random(n) * width).astype(int)
        data[f"{prefix}_min_y"] = (rng.random(n) * height).astype(int)
        data[f"{prefix}_width"] = (rng.random(n) * width / 4).astype(int)
        data[f"{prefix}_height"] = (rng.random(n) * height / 4).astype(int)
    for i in range(1, 22):
        data[f"{i}x"] = rng.integers(-width, width)
        data[f"{i}y"] = rng.integers(-height, height)
        data[f"{i}z"] = rng.normal(0, 0.1, n)
    data["depth_dist"] = rng.normal(0, 3, n)
    data["handedness"] = rng.choice(["left", "right"], n)
    data["picture_name"] = [f"../data/frame_{i}.jpg" if y else f"frame_{i}.jpg" for i, y in enumerate(yale)]
    data["grasp"] = rng.choice(CLASSES, n)
    return pd.DataFrame(data)[RESULT_COLUMNS]


def thesis_apply_func_adapted(row, dist_max):
    """Row-wise normalisation exactly as written in the thesis' nn.py (positional column indices)."""
    w, h = (640, 480) if ".." in row["picture_name"] else (1920, 1080)
    for i in range(0, 8):
        row.iloc[i] = row.iloc[i] / (w if i % 2 == 0 else h)
    for i in range(8, 71):
        if i % 3 == 2:
            row.iloc[i] = row.iloc[i] / w
        elif i % 3 == 0:
            row.iloc[i] = row.iloc[i] / h
    row.iloc[71] = row.iloc[71] / dist_max
    return row


def test_feature_count_is_74():
    assert N_FEATURES == 74
    features, meta = split_features(make_table())
    assert features.shape[1] == 74
    assert list(features.columns[-2:]) == ["handedness_left", "handedness_right"]
    assert list(meta.columns) == ["picture_name", "grasp", "handedness"]


def test_normalize_matches_thesis_implementation():
    table = make_table()
    features, meta = split_features(table)
    depth_max = features["depth_dist"].abs().max()

    expected = features.copy()
    expected["picture_name"] = meta["picture_name"]
    expected = expected.apply(thesis_apply_func_adapted, axis=1, args=(depth_max,)).drop(columns="picture_name")

    actual = normalize(features, meta["picture_name"], depth_max)
    np.testing.assert_allclose(actual.to_numpy(), expected.astype(float).to_numpy())


def test_single_handedness_still_gives_two_dummies():
    table = make_table()
    table["handedness"] = "left"
    features, _ = split_features(table)
    assert features.shape[1] == 74
    assert (features["handedness_right"] == 0).all()


def test_combine_drops_rows_with_missing_values():
    a, b = make_table(10, seed=1), make_table(10, seed=2)
    b.loc[3, "depth_dist"] = np.nan
    combined = combine_results([a, b])
    assert len(combined) == 19
    assert list(combined.columns) == RESULT_COLUMNS
