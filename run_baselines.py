"""Baseline and ablation experiments for neonatal sepsis prediction.

Loads the prepared `neonatal_sepsis_data_v1.npz` artifact and runs a battery of
classical machine-learning baselines across multiple time horizons (e.g. first
6, 12, 18, 24 hours) and channel-set ablations (`values` only, `values + masks`,
or `full` 18-channel tensor).

For each (horizon, channel set, model) combination it:
    1. Builds tabular features from the time-series prefix.
    2. Fits the model on train, with class-imbalance handling.
    3. Tunes a decision threshold on validation by maximizing F1.
    4. Locks the threshold and evaluates on the test set.
    5. Records per-admission predictions and a rich set of metrics.

Outputs (in --out-dir, default ``results/baselines``):
    - baseline_results.csv          tidy long-format results
    - baseline_results.md           ranked tables grouped by horizon/channel set
    - baseline_results.html         styled HTML version of the same tables
    - predictions/<exp_id>.csv      per-admission probabilities and labels
    - plots/*.png and plots/*.pdf   AUROC/AUPRC bars, ROC/PR curves, confusion
                                    matrices, sensitivity/specificity tradeoffs,
                                    and ablation heatmaps
    - run_config.json               serialized CLI config and dataset metadata

Example:
    python run_baselines.py \
        --data neonatal_sepsis_data_v1.npz \
        --out-dir results/baselines

    python run_baselines.py --horizons 6 --channel-sets values --quick
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Constants and configuration
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

logger = logging.getLogger("baselines")


def format_duration(seconds: float) -> str:
    """Format a number of seconds as ``HH:MM:SS`` or ``MM:SS``.

    Used in progress messages so batch-job log files contain readable elapsed
    and ETA values without relying on ANSI cursor control.
    """
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def human_int(value: int) -> str:
    """Render an integer with thousands separators."""
    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------


@dataclass
class SepsisData:
    """Container for the loaded `.npz` dataset."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    norm_params: np.ndarray
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
            "n_channels": self.X_train.shape[-1],
        }


def load_data(npz_path: Path) -> SepsisData:
    """Load the dataset and run integrity checks on it."""

    if not npz_path.exists():
        raise FileNotFoundError(f"Data file not found: {npz_path}")

    raw = np.load(npz_path, allow_pickle=True)

    required = [
        "X_train", "y_train",
        "X_val", "y_val",
        "X_test", "y_test",
        "norm_params", "channel_names", "label_names",
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
        norm_params=raw["norm_params"],
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

    for name, arr in [
        ("X_train", data.X_train),
        ("X_val", data.X_val),
        ("X_test", data.X_test),
    ]:
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN values")
        if np.isinf(arr).any():
            raise ValueError(f"{name} contains inf values")
        if arr.shape[-1] != len(data.channel_names):
            raise ValueError(
                f"{name} channel count {arr.shape[-1]} does not match "
                f"channel_names length {len(data.channel_names)}"
            )

    return data


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _slice_channel_set(X: np.ndarray, channel_set: str, horizon: int) -> np.ndarray:
    """Return tensor prefix sliced to the configured horizon and channels."""
    if channel_set not in CHANNEL_SETS:
        raise KeyError(f"Unknown channel set: {channel_set}")
    if not (1 <= horizon <= X.shape[1]):
        raise ValueError(f"Horizon {horizon} is out of range for tensor of length {X.shape[1]}")
    return X[:, :horizon, CHANNEL_SETS[channel_set]]


def build_tabular_features(X: np.ndarray) -> np.ndarray:
    """Create a tabular feature matrix from a 3D `(N, T, C)` tensor.

    Features per channel:
        - mean, std, min, max
        - first observation
        - last observation
        - delta = last - first
        - slope estimated by simple finite difference on the channel mean trend
    Plus the full flattened time-channel grid for fine-grained signal that
    tree models can exploit.
    """

    n, t, c = X.shape
    means = X.mean(axis=1)
    stds = X.std(axis=1)
    mins = X.min(axis=1)
    maxs = X.max(axis=1)
    firsts = X[:, 0, :]
    lasts = X[:, -1, :]
    deltas = lasts - firsts

    if t > 1:
        diffs = np.diff(X, axis=1)
        slopes = diffs.mean(axis=1)
    else:
        slopes = np.zeros((n, c), dtype=X.dtype)

    flat = X.reshape(n, -1)

    summary = np.concatenate(
        [means, stds, mins, maxs, firsts, lasts, deltas, slopes],
        axis=1,
    )
    return np.concatenate([summary, flat], axis=1)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def make_models(seed: int, quick: bool = False) -> dict[str, Any]:
    """Instantiate baseline classifiers, all imbalance-aware where possible."""

    n_estimators = 100 if quick else 400

    models: dict[str, Any] = {
        "Dummy_StratifiedFloor": DummyClassifier(strategy="stratified", random_state=seed),
        "LogReg_L2_balanced": Pipeline(
            steps=[
                ("scale", StandardScaler(with_mean=True)),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "ExtraTrees_balanced": ExtraTreesClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "HistGradBoost": HistGradientBoostingClassifier(
            max_iter=200 if quick else 600,
            learning_rate=0.05,
            max_depth=6,
            l2_regularization=0.0,
            early_stopping=True,
            random_state=seed,
        ),
    }

    if not quick:
        models["MLP_tabular"] = Pipeline(
            steps=[
                ("scale", StandardScaler(with_mean=True)),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=seed,
                    ),
                ),
            ]
        )

    return models


def supports_sample_weight(name: str) -> bool:
    """Whether to pass sample_weight to .fit (HGB benefits from it)."""
    return name == "HistGradBoost"


def fit_model(name: str, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Fit a model with appropriate imbalance handling."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)

        if supports_sample_weight(name):
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
    return model


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return probability of the positive class with a stable fallback."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Model has neither predict_proba nor decision_function")


# ---------------------------------------------------------------------------
# Metrics and threshold tuning
# ---------------------------------------------------------------------------


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Pick the threshold that maximises F1 on the validation set."""

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
    """Compute the full metric panel for a given probability/threshold pair."""

    preds = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    n = len(y_true)
    metrics = {
        "n": int(n),
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
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return metrics


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    horizon: int
    channel_set: str
    model_name: str
    train_seconds: float
    n_features: int
    val_threshold: float
    feature_seconds: float = 0.0
    predict_seconds: float = 0.0
    total_seconds: float = 0.0
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "horizon": self.horizon,
            "channel_set": self.channel_set,
            "model": self.model_name,
            "train_seconds": round(self.train_seconds, 3),
            "feature_seconds": round(self.feature_seconds, 3),
            "predict_seconds": round(self.predict_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "n_features": self.n_features,
            "val_threshold": round(self.val_threshold, 4),
        }
        for split, metrics in (("val", self.val_metrics), ("test", self.test_metrics)):
            for k, v in metrics.items():
                row[f"{split}_{k}"] = v
        return row


def run_single_experiment(
    data: SepsisData,
    horizon: int,
    channel_set: str,
    model_name: str,
    model: Any,
    out_dir: Path,
    save_predictions: bool = True,
) -> ExperimentResult:
    exp_start = time.time()

    feat_start = time.time()
    X_tr = build_tabular_features(_slice_channel_set(data.X_train, channel_set, horizon))
    X_va = build_tabular_features(_slice_channel_set(data.X_val, channel_set, horizon))
    X_te = build_tabular_features(_slice_channel_set(data.X_test, channel_set, horizon))
    feature_seconds = time.time() - feat_start

    logger.info(
        "  features: train=%s val=%s test=%s | dim=%s | feat_time=%.2fs",
        X_tr.shape, X_va.shape, X_te.shape, human_int(X_tr.shape[1]), feature_seconds,
    )

    fit_start = time.time()
    fit_model(model_name, model, X_tr, data.y_train)
    train_seconds = time.time() - fit_start

    pred_start = time.time()
    val_score = predict_proba(model, X_va)
    test_score = predict_proba(model, X_te)
    predict_seconds = time.time() - pred_start

    threshold, _ = tune_threshold(data.y_val, val_score)
    val_metrics = compute_metrics(data.y_val, val_score, threshold)
    test_metrics = compute_metrics(data.y_test, test_score, threshold)

    total_seconds = time.time() - exp_start

    result = ExperimentResult(
        horizon=horizon,
        channel_set=channel_set,
        model_name=model_name,
        train_seconds=train_seconds,
        feature_seconds=feature_seconds,
        predict_seconds=predict_seconds,
        total_seconds=total_seconds,
        n_features=int(X_tr.shape[1]),
        val_threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    if save_predictions:
        exp_id = f"h{horizon:02d}__{channel_set}__{model_name}"
        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for split_name, hadm_ids, subject_ids, y_true, y_score in [
            ("val", data.val_hadm_ids, data.val_subject_ids, data.y_val, val_score),
            ("test", data.test_hadm_ids, data.test_subject_ids, data.y_test, test_score),
        ]:
            preds = (y_score >= threshold).astype(int)
            for hadm, subj, yt, ys, p in zip(hadm_ids, subject_ids, y_true, y_score, preds):
                rows.append(
                    {
                        "split": split_name,
                        "hadm_id": int(hadm),
                        "subject_id": int(subj),
                        "y_true": int(yt),
                        "y_score": float(ys),
                        "y_pred": int(p),
                        "threshold": float(threshold),
                    }
                )
        pd.DataFrame(rows).to_csv(pred_dir / f"{exp_id}.csv", index=False)

    return result


# ---------------------------------------------------------------------------
# Reporting: tables and plots
# ---------------------------------------------------------------------------


def results_to_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.to_row() for r in results])
    df = df.sort_values(
        by=["horizon", "channel_set", "test_auprc", "test_auroc"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    return df


def write_markdown_report(df: pd.DataFrame, out_path: Path) -> None:
    """Write a human-friendly markdown report grouped by horizon and channel set."""

    cols = [
        "model",
        "test_auroc",
        "test_auprc",
        "test_balanced_accuracy",
        "test_recall_sensitivity",
        "test_specificity",
        "test_f1",
        "test_brier",
        "val_threshold",
        "train_seconds",
    ]

    lines: list[str] = ["# Baseline Results", ""]

    overall = (
        df.sort_values(by=["test_auprc", "test_auroc"], ascending=False)
        .head(10)[["horizon", "channel_set"] + cols]
        .copy()
    )
    lines.append("## Top 10 overall by test AUPRC")
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


def _format_md_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if formatted[col].dtype.kind == "f":
            formatted[col] = formatted[col].map(lambda v: f"{v:.4f}" if pd.notnull(v) else "")
    return formatted.to_markdown(index=False)


def write_html_report(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "horizon", "channel_set", "model",
        "test_auroc", "test_auprc",
        "test_balanced_accuracy", "test_recall_sensitivity",
        "test_specificity", "test_f1", "test_brier",
        "val_threshold", "train_seconds",
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
        .set_caption("Neonatal sepsis baselines — test set metrics")
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


def plot_metric_grid(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_style("whitegrid")

    metrics_to_plot = [
        ("test_auroc", "Test AUROC"),
        ("test_auprc", "Test AUPRC"),
        ("test_balanced_accuracy", "Test Balanced Accuracy"),
        ("test_recall_sensitivity", "Test Sensitivity"),
        ("test_specificity", "Test Specificity"),
        ("test_f1", "Test F1"),
    ]

    for metric, title in metrics_to_plot:
        for channel_set in sorted(df["channel_set"].unique()):
            sub = df[df["channel_set"] == channel_set].copy()
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(
                data=sub,
                x="horizon", y=metric, hue="model", ax=ax,
            )
            ax.set_title(f"{title} by horizon — channel set: {channel_set}")
            ax.set_ylim(0, 1)
            ax.set_ylabel(title)
            ax.set_xlabel("Hours of data used")
            ax.legend(loc="lower right", frameon=True, fontsize=8)
            _save_fig(fig, plots_dir, f"bar_{metric}_{channel_set}")


def plot_sens_spec_tradeoff(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_style("whitegrid")
    for channel_set in sorted(df["channel_set"].unique()):
        sub = df[df["channel_set"] == channel_set].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=sub,
            x="test_specificity",
            y="test_recall_sensitivity",
            hue="model",
            style="horizon",
            s=110,
            ax=ax,
        )
        ax.plot([0, 1], [1, 0], color="grey", linestyle=":", alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Sensitivity vs specificity — channel set: {channel_set}")
        ax.set_xlabel("Test specificity")
        ax.set_ylabel("Test sensitivity")
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
        _save_fig(fig, plots_dir, f"sens_spec_{channel_set}")


def plot_ablation_heatmaps(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_style("white")
    for metric, title in [
        ("test_auroc", "Test AUROC"),
        ("test_auprc", "Test AUPRC"),
        ("test_f1", "Test F1"),
    ]:
        best = (
            df.groupby(["horizon", "channel_set"])[metric]
            .max()
            .reset_index()
            .pivot(index="channel_set", columns="horizon", values=metric)
        )
        if best.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(
            best, annot=True, fmt=".3f", cmap="viridis",
            cbar_kws={"label": metric}, ax=ax,
        )
        ax.set_title(f"Best {title} per ablation cell")
        ax.set_xlabel("Hours of data used")
        ax.set_ylabel("Channel set")
        _save_fig(fig, plots_dir, f"ablation_heatmap_{metric}")


def plot_curves_for_top_models(
    data: SepsisData,
    df: pd.DataFrame,
    pred_dir: Path,
    plots_dir: Path,
) -> None:
    sns.set_style("whitegrid")

    top_per_horizon = (
        df.sort_values("test_auprc", ascending=False)
        .groupby("horizon")
        .head(3)
    )

    for horizon in sorted(top_per_horizon["horizon"].unique()):
        rows = top_per_horizon[top_per_horizon["horizon"] == horizon]
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        for _, row in rows.iterrows():
            exp_id = f"h{int(row.horizon):02d}__{row.channel_set}__{row.model}"
            path = pred_dir / f"{exp_id}.csv"
            if not path.exists():
                continue
            preds = pd.read_csv(path)
            test = preds[preds["split"] == "test"]
            if test.empty:
                continue
            fpr, tpr, _ = roc_curve(test["y_true"], test["y_score"])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(
                fpr, tpr,
                label=f"{row.model} ({row.channel_set}) — AUROC={roc_auc:.3f}",
            )
            prec, rec, _ = precision_recall_curve(test["y_true"], test["y_score"])
            ax_pr.plot(
                rec, prec,
                label=f"{row.model} ({row.channel_set}) — AP={row.test_auprc:.3f}",
            )

        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax_roc.set_title(f"Top models — ROC at {horizon}h")
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.legend(loc="lower right", fontsize=8)
        _save_fig(fig_roc, plots_dir, f"roc_top_h{horizon:02d}")

        ax_pr.set_title(f"Top models — PR at {horizon}h")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_ylim(0, 1.05)
        ax_pr.legend(loc="lower left", fontsize=8)
        _save_fig(fig_pr, plots_dir, f"pr_top_h{horizon:02d}")


def plot_confusion_matrices(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_style("white")
    best_per_horizon = (
        df.sort_values("test_auprc", ascending=False)
        .groupby("horizon")
        .head(1)
    )
    for _, row in best_per_horizon.iterrows():
        cm = np.array(
            [
                [row["test_tn"], row["test_fp"]],
                [row["test_fn"], row["test_tp"]],
            ]
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: control", "Pred: sepsis"],
            yticklabels=["True: control", "True: sepsis"],
            cbar=False, ax=ax,
        )
        ax.set_title(
            f"Best @ h={int(row.horizon)} | {row.model} ({row.channel_set})\n"
            f"AUPRC={row.test_auprc:.3f}, AUROC={row.test_auroc:.3f}"
        )
        _save_fig(
            fig, plots_dir,
            f"confmat_best_h{int(row.horizon):02d}_{row.channel_set}_{row.model}",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("neonatal_sepsis_data_v1.npz"),
        help="Path to the prepared .npz dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/baselines"),
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Time horizons in hours (1-24)",
    )
    parser.add_argument(
        "--channel-sets",
        type=str,
        nargs="+",
        choices=list(CHANNEL_SETS.keys()),
        default=list(DEFAULT_CHANNEL_SETS),
        help="Channel-set ablations to run",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Subset of model names to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller, faster set of models for smoke tests",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating PNG/PDF plots",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip writing per-admission prediction CSVs",
    )
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

    base_models = make_models(seed=args.seed, quick=args.quick)
    if args.models is not None:
        unknown = set(args.models) - set(base_models)
        if unknown:
            raise SystemExit(f"Unknown models: {unknown}. Known: {sorted(base_models)}")
        base_models = {k: v for k, v in base_models.items() if k in args.models}

    config = {
        "data": str(args.data),
        "out_dir": str(out_dir),
        "horizons": horizons,
        "channel_sets": channel_sets,
        "models": list(base_models),
        "seed": args.seed,
        "quick": args.quick,
        "dataset_summary": summary,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results: list[ExperimentResult] = []
    total = len(horizons) * len(channel_sets) * len(base_models)
    counter = 0

    logger.info("=" * 78)
    logger.info(
        "Plan: %d experiments = %d horizon(s) x %d channel set(s) x %d model(s)",
        total, len(horizons), len(channel_sets), len(base_models),
    )
    logger.info("Horizons      : %s", horizons)
    logger.info("Channel sets  : %s", channel_sets)
    logger.info("Models        : %s", list(base_models))
    logger.info("=" * 78)

    run_started = time.time()
    pbar = tqdm(
        total=total,
        desc="Experiments",
        unit="exp",
        dynamic_ncols=True,
        mininterval=1.0,
        file=sys.stdout,
        leave=True,
    )
    try:
        for horizon in horizons:
            for channel_set in channel_sets:
                for model_name in base_models:
                    counter += 1
                    model = make_models(seed=args.seed, quick=args.quick)[model_name]

                    elapsed_so_far = time.time() - run_started
                    avg_so_far = elapsed_so_far / max(counter - 1, 1)
                    eta_pre = avg_so_far * (total - counter + 1) if counter > 1 else 0.0
                    logger.info(
                        "[%d/%d] horizon=%dh | channels=%s | model=%s | elapsed=%s | eta=%s",
                        counter, total, horizon, channel_set, model_name,
                        format_duration(elapsed_so_far),
                        format_duration(eta_pre) if counter > 1 else "??:??",
                    )
                    try:
                        result = run_single_experiment(
                            data=data,
                            horizon=horizon,
                            channel_set=channel_set,
                            model_name=model_name,
                            model=model,
                            out_dir=out_dir,
                            save_predictions=not args.no_predictions,
                        )
                    except Exception as exc:
                        logger.exception("Experiment failed: %s", exc)
                        pbar.update(1)
                        continue

                    elapsed = time.time() - run_started
                    avg_per_exp = elapsed / counter
                    eta = avg_per_exp * (total - counter)

                    logger.info(
                        "  timings  | feat=%.2fs fit=%.2fs predict=%.2fs total=%.2fs",
                        result.feature_seconds, result.train_seconds,
                        result.predict_seconds, result.total_seconds,
                    )
                    logger.info(
                        "  test     | AUROC=%.4f AUPRC=%.4f F1=%.4f sens=%.4f spec=%.4f bal_acc=%.4f thr=%.3f",
                        result.test_metrics["auroc"],
                        result.test_metrics["auprc"],
                        result.test_metrics["f1"],
                        result.test_metrics["recall_sensitivity"],
                        result.test_metrics["specificity"],
                        result.test_metrics["balanced_accuracy"],
                        result.val_threshold,
                    )
                    logger.info(
                        "  val      | AUROC=%.4f AUPRC=%.4f F1=%.4f sens=%.4f spec=%.4f",
                        result.val_metrics["auroc"],
                        result.val_metrics["auprc"],
                        result.val_metrics["f1"],
                        result.val_metrics["recall_sensitivity"],
                        result.val_metrics["specificity"],
                    )
                    logger.info(
                        "  progress | done=%d/%d | elapsed=%s | avg/exp=%.1fs | eta=%s",
                        counter, total,
                        format_duration(elapsed),
                        avg_per_exp,
                        format_duration(eta),
                    )
                    results.append(result)

                    best_auprc = max(r.test_metrics["auprc"] for r in results)
                    pbar.set_postfix(
                        {
                            "h": f"{horizon}",
                            "ch": channel_set,
                            "model": model_name[:18],
                            "auprc": f"{result.test_metrics['auprc']:.3f}",
                            "best": f"{best_auprc:.3f}",
                            "eta": format_duration(eta),
                        },
                        refresh=False,
                    )
                    pbar.update(1)
    finally:
        pbar.close()

    total_elapsed = time.time() - run_started
    logger.info("=" * 78)
    logger.info("Completed %d experiments in %s", len(results), format_duration(total_elapsed))

    if not results:
        raise SystemExit("No experiments produced results")

    df = results_to_dataframe(results)
    df.to_csv(out_dir / "baseline_results.csv", index=False)
    write_markdown_report(df, out_dir / "baseline_results.md")
    write_html_report(df, out_dir / "baseline_results.html")
    joblib.dump(df, out_dir / "baseline_results.joblib")
    logger.info("Saved results table with %d rows", len(df))

    plots_started = time.time()
    if not args.no_plots:
        plots_dir = out_dir / "plots"
        logger.info("Generating plots in %s ...", plots_dir)
        plot_metric_grid(df, plots_dir)
        plot_sens_spec_tradeoff(df, plots_dir)
        plot_ablation_heatmaps(df, plots_dir)
        plot_confusion_matrices(df, plots_dir)
        if not args.no_predictions:
            plot_curves_for_top_models(data, df, out_dir / "predictions", plots_dir)
        logger.info("Plots saved (%.1fs)", time.time() - plots_started)

    logger.info("=" * 78)
    logger.info("Top 5 by test AUPRC:")
    top = df.sort_values("test_auprc", ascending=False).head(5)
    for _, row in top.iterrows():
        logger.info(
            "  h=%2d | %-12s | %-22s | AUROC=%.3f AUPRC=%.3f F1=%.3f sens=%.3f spec=%.3f",
            int(row.horizon), row.channel_set, row.model,
            row.test_auroc, row.test_auprc, row.test_f1,
            row.test_recall_sensitivity, row.test_specificity,
        )
    logger.info("=" * 78)
    logger.info("Done. Total wall-clock time: %s", format_duration(time.time() - run_started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
