"""Training of the grasp classifier (thesis chapter 5)."""
from __future__ import annotations

import copy
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from .config import TrainConfig
from .evaluate import predict, report
from .features import CLASSES, N_FEATURES, normalize, split_features
from .model import Net

log = logging.getLogger(__name__)


def prepare_data(cfg: TrainConfig):
    """Load the combined table, split it and normalise it. Returns tensors, the depth scale and validation metadata."""
    features, meta = split_features(pd.read_csv(cfg.data_csv))
    if features.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {features.shape[1]}")
    labels = meta["grasp"].map({c: i for i, c in enumerate(CLASSES)})
    if labels.isna().any():
        raise ValueError(f"`grasp` must be one of {CLASSES}")

    x_train, x_val, y_train, y_val, _, meta_val = train_test_split(
        features, labels.to_numpy(), meta, test_size=cfg.test_size, random_state=cfg.seed, shuffle=True
    )
    # Both sets are scaled with the training set's largest absolute depth difference.
    depth_max = x_train["depth_dist"].abs().max()
    x_train = normalize(x_train, meta.loc[x_train.index, "picture_name"], depth_max)
    x_val = normalize(x_val, meta_val["picture_name"], depth_max)

    to_x = lambda frame: torch.tensor(frame.to_numpy(), dtype=torch.float32)  # noqa: E731
    to_y = lambda labels_: torch.nn.functional.one_hot(torch.tensor(labels_, dtype=torch.long), len(CLASSES)).float()  # noqa: E731
    return to_x(x_train), to_y(y_train), to_x(x_val), to_y(y_val), float(depth_max), meta_val


def train(cfg: TrainConfig) -> Path:
    """Train the classifier, save the checkpoint and write the validation report. Returns the checkpoint path."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    x_train, y_train, x_val, y_val, depth_max, meta_val = prepare_data(cfg)
    out_dir = Path(cfg.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "runs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    if len(cfg.hidden_dims) != 2:
        raise ValueError("hidden_dims must have exactly two entries")
    model = Net(N_FEATURES, *cfg.hidden_dims, len(CLASSES), dropout=cfg.dropout)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cfg.class_weights, dtype=torch.float))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)

    best_acc, best_weights = -np.inf, None
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, epoch_acc = [], []
        for x_batch, y_batch in tqdm.tqdm(loader, desc=f"Epoch {epoch}", unit="batch"):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            epoch_acc.append(float((y_pred.argmax(1) == y_batch.argmax(1)).float().mean()))

        model.eval()
        with torch.no_grad():
            y_pred = model(x_val)
            # Known limitation kept from the thesis code: the validation loss is computed on the last sample only.
            val_loss = float(loss_fn(y_pred[-1], y_val[-1]))
            val_acc = float((y_pred.argmax(1) == y_val.argmax(1)).float().mean())

        writer.add_scalar("Loss/train", np.mean(epoch_loss), epoch)
        writer.add_scalar("Loss/test", val_loss, epoch)
        writer.add_scalar("Accuracy/train", np.mean(epoch_acc), epoch)
        writer.add_scalar("Accuracy/test", val_acc, epoch)
        if val_acc > best_acc:
            best_acc, best_weights = val_acc, copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch} validation: Cross-entropy={val_loss}, Accuracy={val_acc}")
    writer.close()

    # Known limitation kept from the thesis code: the checkpoint holds the last-epoch weights, while the report
    # below uses the best-epoch weights.
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = out_dir / "models" / f"{stamp}_{Path(cfg.data_csv).stem}_acc{str(best_acc)[:5]}.pth"
    torch.save({"model_state_dict": model.state_dict(), "depth_max": depth_max}, checkpoint_path)
    print(f"Training complete. Checkpoint: {checkpoint_path}")

    model.load_state_dict(best_weights)
    report(y_val.argmax(1).numpy(), predict(model, x_val), meta_val, out_dir)
    return checkpoint_path
