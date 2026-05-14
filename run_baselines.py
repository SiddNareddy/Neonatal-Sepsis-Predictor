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

@dataclass
class SepsisData:
    """Container for the loaded `.npz` dataset."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    norm_params: np.ndarray | None
    channel_names: list[str]
    label_names: list[str]
    train_hadm_ids: np.ndarray
    val_hadm_ids: np.ndarray
    test_hadm_ids: np.ndarray
    train_subject_ids: np.ndarray
    val_subject_ids: np.ndarray
    test_subject_ids: np.ndarray
    X_static_train: np.ndarray | None = None
    X_static_val: np.ndarray | None = None
    X_static_test: np.ndarray | None = None
    static_feature_names: list[str] = field(default_factory=list)
    horizon_max: int = 24

    def summary(self) -> dict[str, Any]:
        has_static = self.X_static_train is not None
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
            "has_static_features": has_static,
            "n_static_features": int(self.X_static_train.shape[1]) if has_static else 0,
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
        "channel_names", "label_names",
        "train_hadm_ids", "val_hadm_ids", "test_hadm_ids",
        "train_subject_ids", "val_subject_ids", "test_subject_ids",
    ]
    missing = [k for k in required if k not in raw.files]
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {missing}")

    static_arrays: dict[str, np.ndarray | None] = {
        "X_static_train": None,
        "X_static_val": None,
        "X_static_test": None,
    }
    static_feature_names: list[str] = []
    has_all_static = all(
        k in raw.files for k in ("X_static_train", "X_static_val", "X_static_test")
    )
    if has_all_static:
        static_arrays = {
            "X_static_train": raw["X_static_train"].astype(np.float32),
            "X_static_val": raw["X_static_val"].astype(np.float32),
            "X_static_test": raw["X_static_test"].astype(np.float32),
        }
        if "static_feature_names" in raw.files:
            static_feature_names = [str(s) for s in raw["static_feature_names"].tolist()]
        else:
            static_dim = int(static_arrays["X_static_train"].shape[1])
            static_feature_names = [f"static_{i}" for i in range(static_dim)]

    data = SepsisData(
        X_train=raw["X_train"].astype(np.float32),
        y_train=raw["y_train"].astype(np.int64),
        X_val=raw["X_val"].astype(np.float32),
        y_val=raw["y_val"].astype(np.int64),
        X_test=raw["X_test"].astype(np.float32),
        y_test=raw["y_test"].astype(np.int64),
        norm_params=raw["norm_params"] if "norm_params" in raw.files else None,
        channel_names=[str(s) for s in raw["channel_names"].tolist()],
        label_names=[str(s) for s in raw["label_names"].tolist()],
        train_hadm_ids=raw["train_hadm_ids"],
        val_hadm_ids=raw["val_hadm_ids"],
        test_hadm_ids=raw["test_hadm_ids"],
        train_subject_ids=raw["train_subject_ids"],
        val_subject_ids=raw["val_subject_ids"],
        test_subject_ids=raw["test_subject_ids"],
        X_static_train=static_arrays["X_static_train"],
        X_static_val=static_arrays["X_static_val"],
        X_static_test=static_arrays["X_static_test"],
        static_feature_names=static_feature_names,
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

    if data.X_static_train is not None:
        for split_name, s_arr, y_arr in [
            ("train", data.X_static_train, data.y_train),
            ("val", data.X_static_val, data.y_val),
            ("test", data.X_static_test, data.y_test),
        ]:
            if s_arr is None:
                raise ValueError("Static arrays must be present for all splits if provided")
            if np.isnan(s_arr).any():
                raise ValueError(f"X_static_{split_name} contains NaN values")
            if np.isinf(s_arr).any():
                raise ValueError(f"X_static_{split_name} contains inf values")
            if s_arr.ndim != 2:
                raise ValueError(f"X_static_{split_name} must be 2D; got shape {s_arr.shape}")
            if s_arr.shape[0] != y_arr.shape[0]:
                raise ValueError(
                    f"X_static_{split_name} rows {s_arr.shape[0]} do not match y_{split_name} rows {y_arr.shape[0]}"
                )
        static_dim = int(data.X_static_train.shape[1])
        if int(data.X_static_val.shape[1]) != static_dim or int(data.X_static_test.shape[1]) != static_dim:
            raise ValueError("Static feature dimensions mismatch across splits")
        if data.static_feature_names and len(data.static_feature_names) != static_dim:
            raise ValueError(
                "static_feature_names length mismatch static dim: "
                f"{len(data.static_feature_names)} != {static_dim}"
            )

    return data

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

THRESHOLD_MODES = (
    "max_f1",
    "max_balanced_accuracy",
    "sensitivity_at_specificity",
    "specificity_at_sensitivity",
    "fixed",
)


def candidate_thresholds(y_score: np.ndarray) -> np.ndarray:
    """Build stable threshold candidates: all unique validation scores + 0.5."""
    return np.sort(np.union1d(np.unique(y_score), [0.5]))


def _sweep_metrics(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, float]]:
    """Compute sens/spec/balanced-accuracy/F1 at every candidate threshold."""
    thrs = candidate_thresholds(y_score)
    eps = 1e-12
    rows: list[dict[str, float]] = []
    for t in thrs:
        preds = (y_score >= t).astype(int)
        tp = float(((preds == 1) & (y_true == 1)).sum())
        fp = float(((preds == 1) & (y_true == 0)).sum())
        tn = float(((preds == 0) & (y_true == 0)).sum())
        fn = float(((preds == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn + eps)
        spec = tn / (tn + fp + eps)
        ba = (sens + spec) / 2
        prec_v = tp / (tp + fp + eps)
        f1 = 2 * prec_v * sens / (prec_v + sens + eps)
        rows.append({
            "threshold": float(t),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "balanced_accuracy": float(ba),
            "f1": float(f1),
        })
    return rows


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mode: str,
    min_sensitivity: float = 0.90,
    min_specificity: float = 0.70,
    fixed_threshold: float = 0.5,
) -> tuple[float, float, bool]:
    """Select a decision threshold on *validation* data using the given mode.

    Returns
    -------
    (selected_threshold, objective_value, threshold_fallback)
        objective_value is the primary optimised quantity at the selected threshold
        (NaN for ``fixed`` mode).
        threshold_fallback is True when a constrained mode found no feasible
        threshold and fell back to max balanced accuracy.
    """
    if mode == "fixed":
        return float(fixed_threshold), float("nan"), False

    sweep = _sweep_metrics(y_true, y_score)
    if not sweep:
        return 0.5, 0.0, False

    if mode == "max_f1":
        best = max(sweep, key=lambda r: (r["f1"], r["balanced_accuracy"]))
        return best["threshold"], best["f1"], False

    if mode == "max_balanced_accuracy":
        best = max(sweep, key=lambda r: (r["balanced_accuracy"], r["f1"]))
        return best["threshold"], best["balanced_accuracy"], False

    if mode == "sensitivity_at_specificity":
        valid = [r for r in sweep if r["specificity"] >= min_specificity - 1e-9]
        if not valid:
            logger.warning(
                "select_threshold: no threshold satisfies specificity>=%.2f "
                "(mode=%s); falling back to max_balanced_accuracy",
                min_specificity, mode,
            )
            best = max(sweep, key=lambda r: (r["balanced_accuracy"], r["f1"]))
            return best["threshold"], best["balanced_accuracy"], True
        best = max(valid, key=lambda r: (r["sensitivity"], r["balanced_accuracy"], r["f1"]))
        return best["threshold"], best["sensitivity"], False

    if mode == "specificity_at_sensitivity":
        valid = [r for r in sweep if r["sensitivity"] >= min_sensitivity - 1e-9]
        if not valid:
            logger.warning(
                "select_threshold: no threshold satisfies sensitivity>=%.2f "
                "(mode=%s); falling back to max_balanced_accuracy",
                min_sensitivity, mode,
            )
            best = max(sweep, key=lambda r: (r["balanced_accuracy"], r["f1"]))
            return best["threshold"], best["balanced_accuracy"], True
        best = max(valid, key=lambda r: (r["specificity"], r["balanced_accuracy"], r["f1"]))
        return best["threshold"], best["specificity"], False

    raise ValueError(f"Unknown threshold mode: {mode!r}. Must be one of {THRESHOLD_MODES}")


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Backward-compatible wrapper: max-F1 threshold selection."""
    t, obj, _ = select_threshold(y_true, y_score, mode="max_f1")
    return t, obj

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

@dataclass
class ExperimentResult:
    horizon: int
    channel_set: str
    static_mode: str
    uses_static: bool
    static_dim: int
    model_name: str
    train_seconds: float
    n_dynamic_features: int
    n_features: int
    val_threshold: float
    feature_seconds: float = 0.0
    predict_seconds: float = 0.0
    total_seconds: float = 0.0
    threshold_mode: str = "max_f1"
    threshold_objective_value: float = 0.0
    threshold_fallback: bool = False
    min_sensitivity: float = 0.90
    min_specificity: float = 0.70
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "horizon": self.horizon,
            "channel_set": self.channel_set,
            "static_mode": self.static_mode,
            "uses_static": self.uses_static,
            "static_dim": self.static_dim,
            "model": self.model_name,
            "threshold_mode": self.threshold_mode,
            "selected_threshold": round(self.val_threshold, 4),
            "threshold_objective_value": round(self.threshold_objective_value, 6) if not (self.threshold_objective_value != self.threshold_objective_value) else float("nan"),
            "threshold_fallback": self.threshold_fallback,
            "min_sensitivity": round(self.min_sensitivity, 4),
            "min_specificity": round(self.min_specificity, 4),
            "train_seconds": round(self.train_seconds, 3),
            "feature_seconds": round(self.feature_seconds, 3),
            "predict_seconds": round(self.predict_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "n_dynamic_features": self.n_dynamic_features,
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
    static_mode: str,
    threshold_modes: list[str],
    min_sensitivity: float = 0.90,
    min_specificity: float = 0.70,
    fixed_threshold: float = 0.5,
    save_predictions: bool = True,
) -> list[ExperimentResult]:
    """Fit one model and evaluate it under every requested threshold mode.

    Returns one ``ExperimentResult`` per threshold mode without retraining.
    """
    exp_start = time.time()

    feat_start = time.time()
    X_tr = build_tabular_features(_slice_channel_set(data.X_train, channel_set, horizon))
    X_va = build_tabular_features(_slice_channel_set(data.X_val, channel_set, horizon))
    X_te = build_tabular_features(_slice_channel_set(data.X_test, channel_set, horizon))
    n_dynamic_features = int(X_tr.shape[1])
    static_dim = 0
    if static_mode == "static":
        if data.X_static_train is None or data.X_static_val is None or data.X_static_test is None:
            raise ValueError("static_mode='static' requested but static arrays are missing in dataset")
        static_dim = int(data.X_static_train.shape[1])
        X_tr = np.concatenate([X_tr, data.X_static_train], axis=1)
        X_va = np.concatenate([X_va, data.X_static_val], axis=1)
        X_te = np.concatenate([X_te, data.X_static_test], axis=1)
    feature_seconds = time.time() - feat_start

    logger.info(
        "  features: mode=%s train=%s val=%s test=%s | dynamic_dim=%s static_dim=%s total_dim=%s | feat_time=%.2fs",
        static_mode, X_tr.shape, X_va.shape, X_te.shape,
        human_int(n_dynamic_features), human_int(static_dim), human_int(X_tr.shape[1]), feature_seconds,
    )

    fit_start = time.time()
    fit_model(model_name, model, X_tr, data.y_train)
    train_seconds = time.time() - fit_start

    pred_start = time.time()
    val_score = predict_proba(model, X_va)
    test_score = predict_proba(model, X_te)
    predict_seconds = time.time() - pred_start

    total_seconds = time.time() - exp_start

    pred_dir = out_dir / "predictions"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    exp_results: list[ExperimentResult] = []
    for thr_mode in threshold_modes:
        threshold, obj_val, fallback = select_threshold(
            data.y_val, val_score,
            mode=thr_mode,
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
            fixed_threshold=fixed_threshold,
        )
        val_metrics = compute_metrics(data.y_val, val_score, threshold)
        test_metrics = compute_metrics(data.y_test, test_score, threshold)

        result = ExperimentResult(
            horizon=horizon,
            channel_set=channel_set,
            static_mode=static_mode,
            uses_static=(static_mode == "static"),
            static_dim=static_dim,
            model_name=model_name,
            train_seconds=train_seconds,
            feature_seconds=feature_seconds,
            predict_seconds=predict_seconds,
            total_seconds=total_seconds,
            n_dynamic_features=n_dynamic_features,
            n_features=int(X_tr.shape[1]),
            val_threshold=threshold,
            threshold_mode=thr_mode,
            threshold_objective_value=float(obj_val),
            threshold_fallback=fallback,
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
        exp_results.append(result)

        if save_predictions:
            exp_id = (
                f"h{horizon:02d}__{channel_set}__static_{static_mode}"
                f"__thr_{thr_mode}__{model_name}"
            )
            rows = []
            for split_name, hadm_ids, subject_ids, y_true, y_sc in [
                ("val", data.val_hadm_ids, data.val_subject_ids, data.y_val, val_score),
                ("test", data.test_hadm_ids, data.test_subject_ids, data.y_test, test_score),
            ]:
                preds = (y_sc >= threshold).astype(int)
                for hadm, subj, yt, ys, p in zip(hadm_ids, subject_ids, y_true, y_sc, preds):
                    rows.append({
                        "split": split_name,
                        "hadm_id": int(hadm),
                        "subject_id": int(subj),
                        "static_mode": static_mode,
                        "threshold_mode": thr_mode,
                        "y_true": int(yt),
                        "y_score": float(ys),
                        "y_pred": int(p),
                        "threshold": float(threshold),
                    })
            pd.DataFrame(rows).to_csv(pred_dir / f"{exp_id}.csv", index=False)

    return exp_results

def results_to_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.to_row() for r in results])
    df = df.sort_values(
        by=["horizon", "channel_set", "static_mode", "threshold_mode", "test_auprc", "test_auroc"],
        ascending=[True, True, True, True, False, False],
    ).reset_index(drop=True)
    return df


def write_markdown_report(df: pd.DataFrame, out_path: Path) -> None:
    """Write a human-friendly markdown report grouped by horizon and channel set."""

    cols = [
        "model", "static_mode", "threshold_mode",
        "test_auroc",
        "test_auprc",
        "test_balanced_accuracy",
        "test_recall_sensitivity",
        "test_specificity",
        "test_f1",
        "test_brier",
        "selected_threshold",
        "threshold_objective_value",
        "threshold_fallback",
        "static_dim",
        "n_dynamic_features",
        "n_features",
        "train_seconds",
    ]
    cols = [c for c in cols if c in df.columns]

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
    base_cols = [
        "horizon", "channel_set", "static_mode", "model",
        "threshold_mode", "selected_threshold", "threshold_fallback",
        "test_auroc", "test_auprc",
        "test_balanced_accuracy", "test_recall_sensitivity",
        "test_specificity", "test_f1", "test_brier",
        "static_dim", "n_dynamic_features", "n_features", "train_seconds",
    ]
    cols = [c for c in base_cols if c in df.columns]
    fmt = {
        "test_auroc": "{:.4f}", "test_auprc": "{:.4f}",
        "test_balanced_accuracy": "{:.4f}",
        "test_recall_sensitivity": "{:.4f}", "test_specificity": "{:.4f}",
        "test_f1": "{:.4f}", "test_brier": "{:.4f}",
        "train_seconds": "{:.2f}",
    }
    if "selected_threshold" in cols:
        fmt["selected_threshold"] = "{:.4f}"
    styled = (
        df[cols]
        .style.format(fmt)
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
    df = df.copy()
    df["model_ablation"] = (
        df["model"].astype(str) + " | " + df["static_mode"].astype(str)
        + " | " + df.get("threshold_mode", pd.Series("max_f1", index=df.index)).astype(str)
    )

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
                x="horizon", y=metric, hue="model_ablation", ax=ax,
            )
            ax.set_title(f"{title} by horizon — channel set: {channel_set}")
            ax.set_ylim(0, 1)
            ax.set_ylabel(title)
            ax.set_xlabel("Hours of data used")
            ax.legend(loc="lower right", frameon=True, fontsize=8)
            _save_fig(fig, plots_dir, f"bar_{metric}_{channel_set}")


def plot_sens_spec_tradeoff(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_style("whitegrid")
    df = df.copy()
    df["model_ablation"] = (
        df["model"].astype(str) + " | " + df["static_mode"].astype(str)
        + " | " + df.get("threshold_mode", pd.Series("max_f1", index=df.index)).astype(str)
    )
    for channel_set in sorted(df["channel_set"].unique()):
        sub = df[df["channel_set"] == channel_set].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=sub,
            x="test_specificity",
            y="test_recall_sensitivity",
            hue="model_ablation",
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
    df = df.copy()
    df["ablation_cell"] = (
        df["channel_set"].astype(str) + " | " + df["static_mode"].astype(str)
        + " | " + df.get("threshold_mode", pd.Series("max_f1", index=df.index)).astype(str)
    )
    for metric, title in [
        ("test_auroc", "Test AUROC"),
        ("test_auprc", "Test AUPRC"),
        ("test_f1", "Test F1"),
    ]:
        best = (
            df.groupby(["horizon", "ablation_cell"])[metric]
            .max()
            .reset_index()
            .pivot(index="ablation_cell", columns="horizon", values=metric)
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

    thr_col = df.get("threshold_mode", pd.Series("max_f1", index=df.index))
    preferred = "max_f1"
    df_curves = df.copy()
    df_curves["_thr_sort"] = (thr_col != preferred).astype(int)
    top_per_horizon = (
        df_curves.sort_values(
            by=["test_auprc", "test_auroc", "_thr_sort"],
            ascending=[False, False, True],
        )
        .drop_duplicates(subset=["horizon", "channel_set", "static_mode", "model"])
        .groupby("horizon")
        .head(3)
    )

    for horizon in sorted(top_per_horizon["horizon"].unique()):
        rows = top_per_horizon[top_per_horizon["horizon"] == horizon]
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        for _, row in rows.iterrows():
            thr_mode = row.get("threshold_mode", "max_f1")
            exp_id = f"h{int(row.horizon):02d}__{row.channel_set}__static_{row.static_mode}__thr_{thr_mode}__{row.model}"
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
                label=f"{row.model} ({row.channel_set}, {row.static_mode}) — AUROC={roc_auc:.3f}",
            )
            prec, rec, _ = precision_recall_curve(test["y_true"], test["y_score"])
            ax_pr.plot(
                rec, prec,
                label=f"{row.model} ({row.channel_set}, {row.static_mode}) — AP={row.test_auprc:.3f}",
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
        thr_label = row.get("threshold_mode", "max_f1")
        ax.set_title(
            f"Best @ h={int(row.horizon)} | {row.model} ({row.channel_set}, {row.static_mode}, {thr_label})\n"
            f"AUPRC={row.test_auprc:.3f}, AUROC={row.test_auroc:.3f}"
        )
        _save_fig(
            fig, plots_dir,
            f"confmat_best_h{int(row.horizon):02d}_{row.channel_set}_{row.static_mode}_{thr_label}_{row.model}",
        )

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
    parser.add_argument(
        "--static-modes",
        type=str,
        nargs="+",
        choices=["none", "static"],
        default=["none"],
        help="Ablation modes: none (dynamic tabular only), static (append static feature block).",
    )
    parser.add_argument(
        "--threshold-modes",
        type=str,
        nargs="+",
        choices=list(THRESHOLD_MODES),
        default=["max_f1"],
        help=(
            "Threshold selection strategies to evaluate. One ExperimentResult row is produced "
            "per mode without retraining. Default: max_f1 (preserves previous behaviour)."
        ),
    )
    parser.add_argument(
        "--min-specificity",
        type=float,
        default=0.70,
        help="Minimum validation specificity for sensitivity_at_specificity mode.",
    )
    parser.add_argument(
        "--min-sensitivity",
        type=float,
        default=0.90,
        help="Minimum validation sensitivity for specificity_at_sensitivity mode.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
        help="Decision threshold used for the 'fixed' mode.",
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
    static_modes = list(dict.fromkeys(args.static_modes))
    if "static" in static_modes and data.X_static_train is None:
        raise SystemExit(
            "Requested static mode but dataset has no X_static_* arrays. "
            "Use the late-fusion NPZ generated by the notebook addendum."
        )
    threshold_modes = list(dict.fromkeys(args.threshold_modes))

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
        "static_modes": static_modes,
        "threshold_modes": threshold_modes,
        "min_sensitivity": args.min_sensitivity,
        "min_specificity": args.min_specificity,
        "fixed_threshold": args.fixed_threshold,
        "seed": args.seed,
        "quick": args.quick,
        "dataset_summary": summary,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results: list[ExperimentResult] = []
    total = len(horizons) * len(channel_sets) * len(static_modes) * len(base_models)
    counter = 0

    logger.info("=" * 78)
    logger.info(
        "Plan: %d training runs = %d horizon(s) x %d channel set(s) x %d static mode(s) x %d model(s)",
        total, len(horizons), len(channel_sets), len(static_modes), len(base_models),
    )
    logger.info(
        "      %d result rows = training runs x %d threshold mode(s)",
        total * len(threshold_modes), len(threshold_modes),
    )
    logger.info("Horizons        : %s", horizons)
    logger.info("Channel sets    : %s", channel_sets)
    logger.info("Static modes    : %s", static_modes)
    logger.info("Threshold modes : %s", threshold_modes)
    logger.info("Models          : %s", list(base_models))
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
                for static_mode in static_modes:
                    for model_name in base_models:
                        counter += 1
                        model = make_models(seed=args.seed, quick=args.quick)[model_name]

                        elapsed_so_far = time.time() - run_started
                        avg_so_far = elapsed_so_far / max(counter - 1, 1)
                        eta_pre = avg_so_far * (total - counter + 1) if counter > 1 else 0.0
                        logger.info(
                            "[%d/%d] horizon=%dh | channels=%s | static=%s | model=%s | elapsed=%s | eta=%s",
                            counter, total, horizon, channel_set, static_mode, model_name,
                            format_duration(elapsed_so_far),
                            format_duration(eta_pre) if counter > 1 else "??:??",
                        )
                        try:
                            exp_results = run_single_experiment(
                                data=data,
                                horizon=horizon,
                                channel_set=channel_set,
                                static_mode=static_mode,
                                model_name=model_name,
                                model=model,
                                out_dir=out_dir,
                                threshold_modes=threshold_modes,
                                min_sensitivity=args.min_sensitivity,
                                min_specificity=args.min_specificity,
                                fixed_threshold=args.fixed_threshold,
                                save_predictions=not args.no_predictions,
                            )
                        except Exception as exc:
                            logger.exception("Experiment failed: %s", exc)
                            pbar.update(1)
                            continue

                        elapsed = time.time() - run_started
                        avg_per_exp = elapsed / counter
                        eta = avg_per_exp * (total - counter)

                        first = exp_results[0] if exp_results else None
                        if first:
                            logger.info(
                                "  timings  | feat=%.2fs fit=%.2fs predict=%.2fs total=%.2fs",
                                first.feature_seconds, first.train_seconds,
                                first.predict_seconds, first.total_seconds,
                            )
                        for res in exp_results:
                            fallback_tag = " [FALLBACK]" if res.threshold_fallback else ""
                            logger.info(
                                "  [%s] test | AUROC=%.4f AUPRC=%.4f F1=%.4f sens=%.4f spec=%.4f bal_acc=%.4f thr=%.3f%s",
                                res.threshold_mode,
                                res.test_metrics["auroc"],
                                res.test_metrics["auprc"],
                                res.test_metrics["f1"],
                                res.test_metrics["recall_sensitivity"],
                                res.test_metrics["specificity"],
                                res.test_metrics["balanced_accuracy"],
                                res.val_threshold,
                                fallback_tag,
                            )
                            logger.info(
                                "  [%s] val  | AUROC=%.4f AUPRC=%.4f F1=%.4f sens=%.4f spec=%.4f",
                                res.threshold_mode,
                                res.val_metrics["auroc"],
                                res.val_metrics["auprc"],
                                res.val_metrics["f1"],
                                res.val_metrics["recall_sensitivity"],
                                res.val_metrics["specificity"],
                            )
                        logger.info(
                            "  progress | done=%d/%d | elapsed=%s | avg/exp=%.1fs | eta=%s",
                            counter, total,
                            format_duration(elapsed),
                            avg_per_exp,
                            format_duration(eta),
                        )
                        results.extend(exp_results)

                        best_auprc = max(r.test_metrics["auprc"] for r in results)
                        ref = first if first else exp_results[0] if exp_results else None
                        pbar.set_postfix(
                            {
                                "h": f"{horizon}",
                                "ch": channel_set,
                                "sm": static_mode,
                                "model": model_name[:14],
                                "auprc": f"{ref.test_metrics['auprc']:.3f}" if ref else "n/a",
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
        thr_mode = row.get("threshold_mode", "max_f1")
        logger.info(
            "  h=%2d | %-12s | static=%-6s | thr=%-28s | %-22s | AUROC=%.3f AUPRC=%.3f F1=%.3f sens=%.3f spec=%.3f",
            int(row.horizon), row.channel_set, row.static_mode, thr_mode, row.model,
            row.test_auroc, row.test_auprc, row.test_f1,
            row.test_recall_sensitivity, row.test_specificity,
        )
    logger.info("=" * 78)
    logger.info("Done. Total wall-clock time: %s", format_duration(time.time() - run_started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
