"""Train an evidence-based CNN-Attention-BiLSTM neonatal sepsis predictor.

The architecture is informed by:
    - Das et al. (2025), "An attention-based bidirectional LSTM-CNN architecture
      for the early prediction of sepsis", Int. J. Data Sci. & Analytics.
    - Saqib et al. (2020), "SSP: Early prediction of sepsis using a fully
      connected LSTM-CNN model", Computers in Biology and Medicine.
    - Beaulieu-Jones et al. and follow-up MIMIC-III sepsis modelling work.
    - Caicedo-Torres & Gutierrez (2020), "ISeeU: a hybrid CNN-BiLSTM mortality
      predictor on ICU vital signs".

It uses:
    - 2 stacked dilated 1D convolutional blocks (BatchNorm + GELU + dropout)
      to capture local physiologic motifs.
    - A lightweight temporal attention gate over CNN features.
    - A 2-layer bidirectional LSTM to model sequence-level dynamics.
    - Additive (Bahdanau-style) attention pooling combined with the final
      forward and backward hidden states.
    - A dropout MLP head producing a single sepsis logit.
    - Class-imbalance handling via `BCEWithLogitsLoss(pos_weight=...)` and an
      optional `WeightedRandomSampler`.
    - AdamW + ReduceLROnPlateau, gradient clipping, and early stopping on
      validation AUPRC.

Loads the prepared `neonatal_sepsis_data_v1.npz` artifact and trains one model
per `(horizon, channel_set)` combination requested via CLI.

Outputs (under --out-dir, default ``results/cnn_bilstm``):
    - cnn_bilstm_results.csv           tidy per-experiment summary
    - cnn_bilstm_results.md / .html    readable ranked tables
    - <exp_id>/training_history.csv    per-epoch metrics and loss
    - <exp_id>/best_model.pt           best validation-AUPRC checkpoint
    - <exp_id>/predictions.csv         per-admission test+val predictions
    - <exp_id>/plots/*.png + *.pdf     loss curves, val AUROC/AUPRC,
                                       ROC, PR, confusion matrix, attention map
    - plots/horizon_comparison_*.png   horizon and ablation comparisons
    - run_config.json                  serialized CLI config and metadata

Example:
    python train_cnn_bilstm.py \
        --data neonatal_sepsis_data_v1.npz \
        --out-dir results/cnn_bilstm \
        --horizons 6 12 18 24 \
        --channel-sets values values_masks full

    python train_cnn_bilstm.py --horizons 6 --channel-sets values \
        --epochs 2 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Constants and channel groupings (must match preprocessing notebook output)
# ---------------------------------------------------------------------------

NUM_VITAL_CHANNELS = 5
NUM_OBSERVED_MASK_CHANNELS = 5
NUM_SOURCE_MASK_CHANNELS = 8
TOTAL_CHANNELS = NUM_VITAL_CHANNELS + NUM_OBSERVED_MASK_CHANNELS + NUM_SOURCE_MASK_CHANNELS  # 18

CHANNEL_SETS: dict[str, slice] = {
    "values": slice(0, NUM_VITAL_CHANNELS),
    "values_masks": slice(0, NUM_VITAL_CHANNELS + NUM_OBSERVED_MASK_CHANNELS),
    "full": slice(0, TOTAL_CHANNELS),
}

DEFAULT_HORIZONS: tuple[int, ...] = (6, 12, 18, 24)
DEFAULT_CHANNEL_SETS: tuple[str, ...] = ("values", "values_masks", "full")

logger = logging.getLogger("cnn_bilstm")


def format_duration(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS`` or ``MM:SS`` for log readability."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def human_int(value: int) -> str:
    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class SepsisData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    channel_names: list[str]
    label_names: list[str]
    train_hadm_ids: np.ndarray
    val_hadm_ids: np.ndarray
    test_hadm_ids: np.ndarray
    train_subject_ids: np.ndarray
    val_subject_ids: np.ndarray
    test_subject_ids: np.ndarray
    horizon_max: int = 24

    def summary(self) -> dict[str, Any]:
        return {
            "channel_names": self.channel_names,
            "label_names": self.label_names,
            "splits": {
                "train": int(self.y_train.shape[0]),
                "val": int(self.y_val.shape[0]),
                "test": int(self.y_test.shape[0]),
            },
            "label_balance": {
                "train": np.bincount(self.y_train.astype(int)).tolist(),
                "val": np.bincount(self.y_val.astype(int)).tolist(),
                "test": np.bincount(self.y_test.astype(int)).tolist(),
            },
            "horizon_max": self.horizon_max,
        }


def load_data(npz_path: Path) -> SepsisData:
    if not npz_path.exists():
        raise FileNotFoundError(f"Data file not found: {npz_path}")

    raw = np.load(npz_path, allow_pickle=True)
    required = [
        "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
        "channel_names", "label_names",
        "train_hadm_ids", "val_hadm_ids", "test_hadm_ids",
        "train_subject_ids", "val_subject_ids", "test_subject_ids",
    ]
    missing = [k for k in required if k not in raw.files]
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {missing}")

    data = SepsisData(
        X_train=raw["X_train"].astype(np.float32),
        y_train=raw["y_train"].astype(np.int64),
        X_val=raw["X_val"].astype(np.float32),
        y_val=raw["y_val"].astype(np.int64),
        X_test=raw["X_test"].astype(np.float32),
        y_test=raw["y_test"].astype(np.int64),
        channel_names=[str(s) for s in raw["channel_names"].tolist()],
        label_names=[str(s) for s in raw["label_names"].tolist()],
        train_hadm_ids=raw["train_hadm_ids"],
        val_hadm_ids=raw["val_hadm_ids"],
        test_hadm_ids=raw["test_hadm_ids"],
        train_subject_ids=raw["train_subject_ids"],
        val_subject_ids=raw["val_subject_ids"],
        test_subject_ids=raw["test_subject_ids"],
        horizon_max=int(raw["X_train"].shape[1]),
    )

    for name, arr in [("X_train", data.X_train), ("X_val", data.X_val), ("X_test", data.X_test)]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"{name} contains NaN/Inf")
        if arr.shape[-1] != len(data.channel_names):
            raise ValueError(
                f"{name} channels {arr.shape[-1]} mismatch channel_names "
                f"length {len(data.channel_names)}"
            )

    return data


def slice_view(X: np.ndarray, channel_set: str, horizon: int) -> np.ndarray:
    if channel_set not in CHANNEL_SETS:
        raise KeyError(f"Unknown channel set: {channel_set}")
    if not (1 <= horizon <= X.shape[1]):
        raise ValueError(f"Horizon {horizon} out of range")
    return np.ascontiguousarray(X[:, :horizon, CHANNEL_SETS[channel_set]])


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class TemporalConvBlock(nn.Module):
    """Conv1D -> BatchNorm -> GELU -> Dropout, residual when shapes allow.

    Operates on `(batch, channels, time)` tensors, padding="same".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual_proj: nn.Module
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h + residual


class TemporalAttentionGate(nn.Module):
    """Lightweight per-time-step attention gate, mirroring the gating used in
    Das et al. (2025) and the lightweight CNN-Attention-BiLSTM ECG model."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.gate(x))


class AdditiveAttentionPool(nn.Module):
    """Bahdanau-style additive attention pooling over a sequence."""

    def __init__(self, hidden_size: int, attn_size: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, attn_size)
        self.score = nn.Linear(attn_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, H)
        e = torch.tanh(self.proj(x))
        scores = self.score(e).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled, weights


class CNNAttentionBiLSTM(nn.Module):
    """Hybrid CNN -> attention gate -> BiLSTM -> attention pool -> MLP head."""

    def __init__(
        self,
        in_channels: int,
        conv_channels: tuple[int, int] = (64, 128),
        kernel_sizes: tuple[int, int] = (3, 5),
        dilations: tuple[int, int] = (1, 2),
        lstm_hidden: int = 96,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        head_hidden: int = 96,
        input_feature_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_feature_dropout = input_feature_dropout

        self.conv1 = TemporalConvBlock(
            in_channels, conv_channels[0],
            kernel_size=kernel_sizes[0],
            dilation=dilations[0],
            dropout=dropout * 0.5,
        )
        self.conv2 = TemporalConvBlock(
            conv_channels[0], conv_channels[1],
            kernel_size=kernel_sizes[1],
            dilation=dilations[1],
            dropout=dropout * 0.5,
        )
        self.attn_gate = TemporalAttentionGate(conv_channels[1])

        self.lstm = nn.LSTM(
            input_size=conv_channels[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attn_pool = AdditiveAttentionPool(hidden_size=2 * lstm_hidden)

        head_in = 2 * lstm_hidden + 2 * lstm_hidden  # pooled + final fwd/bwd states
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def _maybe_feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if self.training and self.input_feature_dropout > 0:
            keep = 1.0 - self.input_feature_dropout
            mask = (torch.rand(x.size(0), 1, x.size(2), device=x.device) < keep).to(x.dtype)
            x = x * mask
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: (B, T, C)
        x = self._maybe_feature_dropout(x)
        conv_in = x.transpose(1, 2)  # (B, C, T)
        h = self.conv1(conv_in)
        h = self.conv2(h)
        h = self.attn_gate(h)
        seq = h.transpose(1, 2)  # (B, T, C')

        lstm_out, (h_n, _) = self.lstm(seq)
        # h_n: (num_layers * 2, B, H) -> last layer fwd & bwd states
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        last_states = torch.cat([h_fwd, h_bwd], dim=-1)

        pooled, attn_weights = self.attn_pool(lstm_out)
        rep = torch.cat([pooled, last_states], dim=-1)

        logit = self.head(rep).squeeze(-1)
        return logit, attn_weights if return_attention else None


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
    drop_last: bool = False,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=drop_last,
    )


def make_balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(y.astype(int))
    weights = 1.0 / np.where(counts == 0, 1, counts)
    sample_weights = weights[y.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float64)),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=2)
    neg, pos = counts[0], counts[1]
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(float(neg) / float(pos))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, 0.0
    eps = 1e-12
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + eps)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    preds = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    horizon: int
    channel_set: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    early_stop_patience: int
    use_sampler: bool
    use_pos_weight: bool
    monitor: str
    lstm_hidden: int
    lstm_layers: int
    conv_channels: tuple[int, int]
    kernel_sizes: tuple[int, int]
    dilations: tuple[int, int]
    dropout: float
    head_hidden: int
    seed: int
    device: str


@dataclass
class TrainArtifacts:
    best_state_dict: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    val_threshold: float
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    val_attention: np.ndarray | None
    test_attention: np.ndarray | None
    n_params: int
    train_seconds: float


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_attention: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, attn = model(xb, return_attention=return_attention)
            probs = torch.sigmoid(logits)
            scores.append(probs.detach().cpu().numpy())
            targets.append(yb.numpy())
            if return_attention and attn is not None:
                attentions.append(attn.detach().cpu().numpy())
    y_score = np.concatenate(scores)
    y_true = np.concatenate(targets).astype(np.int64)
    attn_arr = np.concatenate(attentions) if (return_attention and attentions) else None
    return y_score, y_true, attn_arr


def train_one_experiment(
    data: SepsisData,
    config: TrainConfig,
    out_dir: Path,
) -> TrainArtifacts:
    set_seed(config.seed)
    device = select_device(config.device)

    X_tr = slice_view(data.X_train, config.channel_set, config.horizon)
    X_va = slice_view(data.X_val, config.channel_set, config.horizon)
    X_te = slice_view(data.X_test, config.channel_set, config.horizon)
    in_channels = X_tr.shape[-1]

    sampler = make_balanced_sampler(data.y_train) if config.use_sampler else None
    train_loader = make_dataloader(
        X_tr, data.y_train,
        batch_size=config.batch_size, shuffle=True, sampler=sampler,
    )
    val_loader = make_dataloader(X_va, data.y_val, batch_size=512, shuffle=False)
    test_loader = make_dataloader(X_te, data.y_test, batch_size=512, shuffle=False)

    model = CNNAttentionBiLSTM(
        in_channels=in_channels,
        conv_channels=config.conv_channels,
        kernel_sizes=config.kernel_sizes,
        dilations=config.dilations,
        lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
        head_hidden=config.head_hidden,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    pos_weight_tensor = compute_pos_weight(data.y_train) if config.use_pos_weight else None
    pos_weight_value = float(pos_weight_tensor.item()) if pos_weight_tensor is not None else None
    if pos_weight_tensor is not None:
        pos_weight_tensor = pos_weight_tensor.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, config.early_stop_patience // 2),
        min_lr=1e-6,
    )

    train_pos = int(np.sum(data.y_train == 1))
    train_neg = int(np.sum(data.y_train == 0))
    train_pos_frac = train_pos / max(train_pos + train_neg, 1)
    batches_per_epoch = max(len(train_loader), 1)

    logger.info("-" * 78)
    logger.info("Experiment: horizon=%dh | channel_set=%s", config.horizon, config.channel_set)
    logger.info("Device     : %s", device)
    logger.info("Input      : (T=%d, C=%d) | train=%s val=%s test=%s",
                config.horizon, in_channels,
                X_tr.shape, X_va.shape, X_te.shape)
    logger.info("Class bal  : train pos=%s neg=%s (pos_frac=%.3f) | val pos=%s neg=%s | test pos=%s neg=%s",
                human_int(train_pos), human_int(train_neg), train_pos_frac,
                human_int(int(np.sum(data.y_val == 1))), human_int(int(np.sum(data.y_val == 0))),
                human_int(int(np.sum(data.y_test == 1))), human_int(int(np.sum(data.y_test == 0))))
    logger.info("Loss       : BCE | pos_weight=%s | sampler=%s",
                f"{pos_weight_value:.3f}" if pos_weight_value is not None else "off",
                "WeightedRandomSampler" if config.use_sampler else "off")
    logger.info("Optim      : AdamW lr=%.2e wd=%.2e | grad_clip=%.2f", config.lr, config.weight_decay, config.grad_clip)
    logger.info("Schedule   : ReduceLROnPlateau (monitor=%s, factor=0.5, patience=%d)",
                config.monitor, max(2, config.early_stop_patience // 2))
    logger.info("Stopping   : early_stop_patience=%d | epochs<=%d", config.early_stop_patience, config.epochs)
    logger.info("Model      : params=%s | conv=%s | kernels=%s | dilations=%s | lstm=%dx%d | head=%d | dropout=%.2f",
                human_int(n_params), config.conv_channels, config.kernel_sizes, config.dilations,
                config.lstm_layers, config.lstm_hidden, config.head_hidden, config.dropout)
    logger.info("Train      : batch_size=%d | batches/epoch=%d", config.batch_size, batches_per_epoch)
    logger.info("-" * 78)

    history: list[dict[str, float]] = []
    epoch_seconds_history: list[float] = []
    best_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_seen = 0

        batch_pbar = tqdm(
            train_loader,
            total=batches_per_epoch,
            desc=f"Epoch {epoch:>3}/{config.epochs}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=2.0,
            file=sys.stdout,
            leave=False,
        )
        for xb, yb in batch_pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch_loss = float(loss.detach().item())
            running_loss += batch_loss * xb.size(0)
            n_seen += xb.size(0)
            batch_pbar.set_postfix(
                {"loss": f"{batch_loss:.4f}", "avg": f"{running_loss / max(n_seen, 1):.4f}"},
                refresh=False,
            )
        batch_pbar.close()

        train_loss = running_loss / max(n_seen, 1)

        val_score, val_true, _ = evaluate(model, val_loader, device)
        threshold, _ = tune_threshold(val_true, val_score)
        val_metrics = compute_metrics(val_true, val_score, threshold)

        monitor_value = val_metrics.get(config.monitor, val_metrics["auprc"])
        scheduler.step(monitor_value)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_seconds = time.time() - epoch_start
        epoch_seconds_history.append(epoch_seconds)
        elapsed_total = time.time() - start
        avg_epoch = sum(epoch_seconds_history) / len(epoch_seconds_history)
        epochs_remaining = max(config.epochs - epoch, 0)
        patience_remaining = max(config.early_stop_patience - epochs_without_improvement, 0)
        eta_epochs = min(epochs_remaining, patience_remaining + 1)
        eta_seconds = avg_epoch * eta_epochs

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall_sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_threshold": val_metrics["threshold"],
            "lr": current_lr,
            "epoch_seconds": epoch_seconds,
        })

        improved = monitor_value > best_metric + 1e-5
        if improved:
            delta = monitor_value - best_metric if math.isfinite(best_metric) else float("inf")
            best_metric = monitor_value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            improvement_str = f"NEW BEST {config.monitor}={monitor_value:.4f}"
            if math.isfinite(delta):
                improvement_str += f" (+{delta:.4f})"
        else:
            epochs_without_improvement += 1
            improvement_str = (
                f"no improve ({config.monitor}={monitor_value:.4f}, best={best_metric:.4f}, "
                f"stale={epochs_without_improvement}/{config.early_stop_patience})"
            )

        logger.info(
            "Epoch %3d/%d | %s | train_loss=%.4f | val AUROC=%.4f AUPRC=%.4f F1=%.4f sens=%.4f spec=%.4f | thr=%.3f | lr=%.2e | %s | elapsed=%s eta=%s",
            epoch, config.epochs,
            f"{epoch_seconds:5.1f}s",
            train_loss,
            val_metrics["auroc"], val_metrics["auprc"], val_metrics["f1"],
            val_metrics["recall_sensitivity"], val_metrics["specificity"],
            val_metrics["threshold"], current_lr,
            improvement_str,
            format_duration(elapsed_total),
            format_duration(eta_seconds),
        )

        if epochs_without_improvement >= config.early_stop_patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs; best was epoch %d, %s=%.4f)",
                epoch, config.early_stop_patience, best_epoch, config.monitor, best_metric,
            )
            break

    train_seconds = time.time() - start

    logger.info(
        "Training summary: epochs_run=%d | best_epoch=%d | best_%s=%.4f | total_time=%s | avg_epoch=%.2fs",
        len(history), best_epoch, config.monitor, best_metric,
        format_duration(train_seconds),
        train_seconds / max(len(history), 1),
    )

    model.load_state_dict(best_state)
    logger.info("Final eval: running validation and test passes with best checkpoint ...")
    val_score, val_true, val_attn = evaluate(model, val_loader, device, return_attention=True)
    threshold, _ = tune_threshold(val_true, val_score)
    val_metrics = compute_metrics(val_true, val_score, threshold)

    test_score, test_true, test_attn = evaluate(model, test_loader, device, return_attention=True)
    test_metrics = compute_metrics(test_true, test_score, threshold)

    logger.info(
        "Final test  | AUROC=%.4f AUPRC=%.4f F1=%.4f sens=%.4f spec=%.4f bal_acc=%.4f thr=%.3f",
        test_metrics["auroc"], test_metrics["auprc"], test_metrics["f1"],
        test_metrics["recall_sensitivity"], test_metrics["specificity"],
        test_metrics["balanced_accuracy"], threshold,
    )
    logger.info(
        "Final val   | AUROC=%.4f AUPRC=%.4f F1=%.4f sens=%.4f spec=%.4f",
        val_metrics["auroc"], val_metrics["auprc"], val_metrics["f1"],
        val_metrics["recall_sensitivity"], val_metrics["specificity"],
    )

    return TrainArtifacts(
        best_state_dict=best_state,
        history=history,
        val_threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_predictions=val_score,
        test_predictions=test_score,
        val_attention=val_attn,
        test_attention=test_attn,
        n_params=n_params,
        train_seconds=train_seconds,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRecord:
    horizon: int
    channel_set: str
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    val_threshold: float
    n_params: int
    train_seconds: float
    n_features: int
    epochs_run: int

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "horizon": self.horizon,
            "channel_set": self.channel_set,
            "model": "CNN-Attn-BiLSTM",
            "n_features": self.n_features,
            "n_params": self.n_params,
            "train_seconds": round(self.train_seconds, 3),
            "epochs_run": self.epochs_run,
            "val_threshold": round(self.val_threshold, 4),
        }
        for split, metrics in (("val", self.val_metrics), ("test", self.test_metrics)):
            for k, v in metrics.items():
                row[f"{split}_{k}"] = v
        return row


def _format_md_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if formatted[col].dtype.kind == "f":
            formatted[col] = formatted[col].map(lambda v: f"{v:.4f}" if pd.notnull(v) else "")
    return formatted.to_markdown(index=False)


def write_markdown_report(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "model",
        "test_auroc", "test_auprc",
        "test_balanced_accuracy", "test_recall_sensitivity",
        "test_specificity", "test_f1", "test_brier",
        "val_threshold", "epochs_run", "n_params", "train_seconds",
    ]
    lines = ["# CNN-Attention-BiLSTM Results", ""]

    overall = (
        df.sort_values(by=["test_auprc", "test_auroc"], ascending=False)
        .head(10)[["horizon", "channel_set"] + cols]
        .copy()
    )
    lines.append("## Top configurations by test AUPRC")
    lines.append("")
    lines.append(_format_md_table(overall))
    lines.append("")

    for horizon in sorted(df["horizon"].unique()):
        for channel_set in sorted(df["channel_set"].unique()):
            sub = df[(df.horizon == horizon) & (df.channel_set == channel_set)].copy()
            if sub.empty:
                continue
            lines.append(f"## Horizon {horizon}h, channel set: `{channel_set}`")
            lines.append("")
            lines.append(_format_md_table(sub[cols]))
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_html_report(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "horizon", "channel_set", "model",
        "test_auroc", "test_auprc",
        "test_balanced_accuracy", "test_recall_sensitivity",
        "test_specificity", "test_f1", "test_brier",
        "val_threshold", "epochs_run", "n_params", "train_seconds",
    ]
    styled = (
        df[cols]
        .style.format({
            "test_auroc": "{:.4f}", "test_auprc": "{:.4f}",
            "test_balanced_accuracy": "{:.4f}",
            "test_recall_sensitivity": "{:.4f}", "test_specificity": "{:.4f}",
            "test_f1": "{:.4f}", "test_brier": "{:.4f}",
            "val_threshold": "{:.4f}", "train_seconds": "{:.2f}",
        })
        .background_gradient(subset=["test_auroc", "test_auprc"], cmap="Greens")
        .set_caption("Neonatal sepsis CNN-Attention-BiLSTM — test set metrics")
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#f4f4f8"), ("text-align", "left")]},
            {"selector": "caption", "props": [("font-size", "1.2em"), ("padding", "8px")]},
        ])
    )
    out_path.write_text(styled.to_html(), encoding="utf-8")


def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_training_history(history: list[dict[str, float]], out_dir: Path) -> None:
    sns.set_style("whitegrid")
    df = pd.DataFrame(history)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss", color="#2b8cbe")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    _save_fig(fig, out_dir, "training_loss")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["val_auroc"], label="Val AUROC", color="#2ca25f")
    ax.plot(df["epoch"], df["val_auprc"], label="Val AUPRC", color="#e34a33")
    ax.plot(df["epoch"], df["val_f1"], label="Val F1", color="#756bb1", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Validation metrics over epochs")
    ax.legend()
    _save_fig(fig, out_dir, "training_validation_metrics")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["lr"], color="#ec7014")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    ax.set_title("Learning rate schedule")
    _save_fig(fig, out_dir, "learning_rate")


def plot_test_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    out_dir: Path,
) -> None:
    sns.set_style("whitegrid")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}", color="#2ca25f")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC")
    ax.legend(loc="lower right")
    _save_fig(fig, out_dir, "roc_test")

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#e34a33", label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test Precision-Recall")
    ax.legend(loc="lower left")
    _save_fig(fig, out_dir, "pr_test")

    preds = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred: control", "Pred: sepsis"],
        yticklabels=["True: control", "True: sepsis"],
        cbar=False, ax=ax,
    )
    ax.set_title(f"Test confusion matrix @ thr={threshold:.3f}")
    _save_fig(fig, out_dir, "confusion_matrix_test")


def plot_attention_examples(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    attention: np.ndarray,
    out_dir: Path,
    horizon: int,
    title_prefix: str = "Test",
) -> None:
    if attention is None or attention.size == 0:
        return
    sns.set_style("white")

    preds = (y_score >= threshold).astype(int)
    pick: dict[str, int | None] = {"TP": None, "FP": None, "TN": None, "FN": None}
    for idx in range(len(y_true)):
        yt, yp = int(y_true[idx]), int(preds[idx])
        if yt == 1 and yp == 1 and pick["TP"] is None:
            pick["TP"] = idx
        elif yt == 0 and yp == 1 and pick["FP"] is None:
            pick["FP"] = idx
        elif yt == 0 and yp == 0 and pick["TN"] is None:
            pick["TN"] = idx
        elif yt == 1 and yp == 0 and pick["FN"] is None:
            pick["FN"] = idx
        if all(v is not None for v in pick.values()):
            break

    examples = [(name, idx) for name, idx in pick.items() if idx is not None]
    if not examples:
        return

    fig, axes = plt.subplots(len(examples), 1, figsize=(8, 1.6 * len(examples)), squeeze=False)
    for i, (name, idx) in enumerate(examples):
        ax = axes[i, 0]
        weights = attention[idx]
        ax.bar(range(len(weights)), weights, color="#3182bd")
        ax.set_title(
            f"{name}: y_true={int(y_true[idx])} y_score={y_score[idx]:.2f}",
            fontsize=10,
        )
        ax.set_xlim(-0.5, horizon - 0.5)
        ax.set_ylim(0, max(0.05, weights.max() * 1.15))
        ax.set_xlabel("Hour" if i == len(examples) - 1 else "")
        ax.set_ylabel("Attention weight")
    fig.suptitle(f"{title_prefix} attention weights — horizon={horizon}h")
    fig.tight_layout()
    _save_fig(fig, out_dir, f"attention_examples_h{horizon:02d}")


def plot_horizon_and_ablation(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_style("whitegrid")

    for metric, title in [
        ("test_auroc", "Test AUROC"),
        ("test_auprc", "Test AUPRC"),
        ("test_recall_sensitivity", "Test Sensitivity"),
        ("test_specificity", "Test Specificity"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=df, x="horizon", y=metric, hue="channel_set", ax=ax,
        )
        ax.set_title(f"{title} by horizon and channel set")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Hours of data used")
        ax.set_ylabel(title)
        ax.legend(title="Channel set")
        _save_fig(fig, plots_dir, f"horizon_comparison_{metric}")

    for metric, title in [
        ("test_auroc", "Test AUROC"),
        ("test_auprc", "Test AUPRC"),
        ("test_f1", "Test F1"),
    ]:
        pivot = df.pivot_table(
            index="channel_set", columns="horizon", values=metric, aggfunc="max",
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title(f"Best {title} per ablation cell")
        ax.set_xlabel("Hours of data used")
        ax.set_ylabel("Channel set")
        _save_fig(fig, plots_dir, f"ablation_heatmap_{metric}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("neonatal_sepsis_data_v1.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/cnn_bilstm"))
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS),
    )
    parser.add_argument(
        "--channel-sets", type=str, nargs="+",
        choices=list(CHANNEL_SETS.keys()),
        default=list(DEFAULT_CHANNEL_SETS),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--use-sampler", action="store_true",
                        help="Use WeightedRandomSampler instead of pos_weight")
    parser.add_argument("--no-pos-weight", action="store_true",
                        help="Disable pos_weight in BCE loss")
    parser.add_argument("--monitor", type=str, default="auprc",
                        choices=["auprc", "auroc", "f1", "balanced_accuracy"])
    parser.add_argument("--lstm-hidden", type=int, default=96)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--conv-channels", type=int, nargs=2, default=[64, 128])
    parser.add_argument("--kernel-sizes", type=int, nargs=2, default=[3, 5])
    parser.add_argument("--dilations", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--head-hidden", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    handler_file = logging.FileHandler(out_dir / "run.log", mode="w")
    handler_console = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler_file.setFormatter(fmt)
    handler_console.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(handler_file)
    logger.addHandler(handler_console)
    logger.setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out_dir)

    logger.info("Loading dataset from %s", args.data)
    data = load_data(args.data)
    summary = data.summary()
    logger.info("Dataset summary: %s", json.dumps(summary, indent=2))

    horizons = sorted({int(h) for h in args.horizons if 1 <= h <= data.horizon_max})
    if not horizons:
        raise SystemExit("No valid horizons provided")
    channel_sets = list(args.channel_sets)

    config_serialised = {
        "data": str(args.data),
        "out_dir": str(out_dir),
        "horizons": horizons,
        "channel_sets": channel_sets,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "early_stop_patience": args.early_stop_patience,
        "use_sampler": args.use_sampler,
        "use_pos_weight": not args.no_pos_weight,
        "monitor": args.monitor,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "conv_channels": list(args.conv_channels),
        "kernel_sizes": list(args.kernel_sizes),
        "dilations": list(args.dilations),
        "dropout": args.dropout,
        "head_hidden": args.head_hidden,
        "seed": args.seed,
        "device": args.device,
        "dataset_summary": summary,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config_serialised, indent=2), encoding="utf-8")

    records: list[ExperimentRecord] = []
    total = len(horizons) * len(channel_sets)
    counter = 0

    logger.info("=" * 78)
    logger.info(
        "Plan: %d CNN-Attn-BiLSTM experiments = %d horizon(s) x %d channel set(s)",
        total, len(horizons), len(channel_sets),
    )
    logger.info("Horizons      : %s", horizons)
    logger.info("Channel sets  : %s", channel_sets)
    logger.info(
        "Per-experiment: epochs<=%d | batch_size=%d | early_stop=%d | monitor=%s | device=%s",
        args.epochs, args.batch_size, args.early_stop_patience, args.monitor, args.device,
    )
    logger.info("=" * 78)

    run_started = time.time()

    for horizon in horizons:
        for channel_set in channel_sets:
            counter += 1
            exp_id = f"h{horizon:02d}__{channel_set}"
            exp_dir = out_dir / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)

            elapsed_so_far = time.time() - run_started
            avg_so_far = elapsed_so_far / max(counter - 1, 1)
            eta_pre = avg_so_far * (total - counter + 1) if counter > 1 else 0.0
            logger.info(
                "[%d/%d] Starting %s | elapsed=%s | eta=%s",
                counter, total, exp_id,
                format_duration(elapsed_so_far),
                format_duration(eta_pre) if counter > 1 else "??:??",
            )

            tcfg = TrainConfig(
                horizon=horizon,
                channel_set=channel_set,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                early_stop_patience=args.early_stop_patience,
                use_sampler=args.use_sampler,
                use_pos_weight=not args.no_pos_weight,
                monitor=args.monitor,
                lstm_hidden=args.lstm_hidden,
                lstm_layers=args.lstm_layers,
                conv_channels=tuple(args.conv_channels),
                kernel_sizes=tuple(args.kernel_sizes),
                dilations=tuple(args.dilations),
                dropout=args.dropout,
                head_hidden=args.head_hidden,
                seed=args.seed,
                device=args.device,
            )

            try:
                artifacts = train_one_experiment(data=data, config=tcfg, out_dir=exp_dir)
            except Exception as exc:
                logger.exception("Experiment %s failed: %s", exp_id, exc)
                continue

            torch.save(
                {
                    "state_dict": artifacts.best_state_dict,
                    "config": config_serialised,
                    "experiment": {
                        "horizon": horizon,
                        "channel_set": channel_set,
                        "in_channels": int(slice_view(data.X_train, channel_set, horizon).shape[-1]),
                    },
                    "val_threshold": artifacts.val_threshold,
                    "val_metrics": artifacts.val_metrics,
                    "test_metrics": artifacts.test_metrics,
                },
                exp_dir / "best_model.pt",
            )

            history_df = pd.DataFrame(artifacts.history)
            history_df.to_csv(exp_dir / "training_history.csv", index=False)

            pred_rows: list[dict[str, Any]] = []
            for split_name, hadm_ids, subject_ids, y_true, y_score in [
                ("val", data.val_hadm_ids, data.val_subject_ids, data.y_val, artifacts.val_predictions),
                ("test", data.test_hadm_ids, data.test_subject_ids, data.y_test, artifacts.test_predictions),
            ]:
                preds = (y_score >= artifacts.val_threshold).astype(int)
                for hadm, subj, yt, ys, p in zip(hadm_ids, subject_ids, y_true, y_score, preds):
                    pred_rows.append({
                        "split": split_name,
                        "hadm_id": int(hadm),
                        "subject_id": int(subj),
                        "y_true": int(yt),
                        "y_score": float(ys),
                        "y_pred": int(p),
                        "threshold": float(artifacts.val_threshold),
                    })
            pd.DataFrame(pred_rows).to_csv(exp_dir / "predictions.csv", index=False)

            if not args.no_plots:
                plots_dir = exp_dir / "plots"
                plot_training_history(artifacts.history, plots_dir)
                plot_test_curves(
                    data.y_test, artifacts.test_predictions, artifacts.val_threshold, plots_dir,
                )
                plot_attention_examples(
                    data.y_test, artifacts.test_predictions, artifacts.val_threshold,
                    artifacts.test_attention, plots_dir, horizon=horizon, title_prefix="Test",
                )

            records.append(
                ExperimentRecord(
                    horizon=horizon,
                    channel_set=channel_set,
                    val_metrics=artifacts.val_metrics,
                    test_metrics=artifacts.test_metrics,
                    val_threshold=artifacts.val_threshold,
                    n_params=artifacts.n_params,
                    train_seconds=artifacts.train_seconds,
                    n_features=int(slice_view(data.X_train, channel_set, horizon).shape[-1]),
                    epochs_run=len(artifacts.history),
                )
            )

            elapsed = time.time() - run_started
            avg_per_exp = elapsed / counter
            eta = avg_per_exp * (total - counter)

            logger.info(
                "  -> %s | test AUROC=%.4f AUPRC=%.4f F1=%.4f sens=%.4f spec=%.4f thr=%.3f | trained_in=%s",
                exp_id,
                artifacts.test_metrics["auroc"],
                artifacts.test_metrics["auprc"],
                artifacts.test_metrics["f1"],
                artifacts.test_metrics["recall_sensitivity"],
                artifacts.test_metrics["specificity"],
                artifacts.val_threshold,
                format_duration(artifacts.train_seconds),
            )
            logger.info(
                "  progress | done=%d/%d | elapsed=%s | avg/exp=%s | eta=%s",
                counter, total,
                format_duration(elapsed),
                format_duration(avg_per_exp),
                format_duration(eta),
            )

    if not records:
        raise SystemExit("No CNN-BiLSTM experiments produced results")

    df = pd.DataFrame([r.to_row() for r in records]).sort_values(
        by=["horizon", "channel_set", "test_auprc"], ascending=[True, True, False],
    ).reset_index(drop=True)
    df.to_csv(out_dir / "cnn_bilstm_results.csv", index=False)
    write_markdown_report(df, out_dir / "cnn_bilstm_results.md")
    write_html_report(df, out_dir / "cnn_bilstm_results.html")
    logger.info("Saved cross-experiment summary with %d rows", len(df))

    if not args.no_plots and len(df) > 1:
        plots_dir = out_dir / "plots"
        plot_horizon_and_ablation(df, plots_dir)
        logger.info("Saved cross-experiment plots to %s", plots_dir)

    logger.info("=" * 78)
    logger.info("Top 5 by test AUPRC:")
    top = df.sort_values("test_auprc", ascending=False).head(5)
    for _, row in top.iterrows():
        logger.info(
            "  h=%2d | %-12s | AUROC=%.3f AUPRC=%.3f F1=%.3f sens=%.3f spec=%.3f | epochs=%d | params=%s",
            int(row.horizon), row.channel_set,
            row.test_auroc, row.test_auprc, row.test_f1,
            row.test_recall_sensitivity, row.test_specificity,
            int(row.epochs_run), human_int(int(row.n_params)),
        )
    logger.info("=" * 78)
    logger.info("Done. Total wall-clock time: %s", format_duration(time.time() - run_started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
