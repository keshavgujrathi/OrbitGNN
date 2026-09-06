"""
train.py
--------
Training and evaluation for OrbitGNN on real TLE benchmark data.

Usage
-----
  # Real benchmark (primary)
  python train.py --dataset real \\
                  --dataset-path /path/to/TLE_observation_benchmark_dataset-main

  # Synthetic (regression)
  python train.py --dataset synthetic

  # Just run baselines (no model training)
  python train.py --baselines-only

Key improvements over previous version
---------------------------------------
* Per-satellite residual normalization (FIX F2): GEO and LEO satellites
  are now scaled independently so GEO Δa (≈1 km) doesn't swamp LEO Δa (≈0.001 km).
* Correct propagation dt (FIX F1): actual inter-TLE elapsed time is passed
  to compute_residual_sequences() via dt_seconds_grid.
* Baseline models for comparison: residual-magnitude, rolling-z-score, IsolationForest.
* Detection-offset metric is now correctly defined and documented.
* Reproducibility: random seed, version printing, best-epoch checkpointing.
* Tighter maneuver tolerance (24 h instead of 36 h).
* PR-AUC, F1, confusion matrix, event-level evaluation all retained.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dataset import (
    load_real_benchmark_dataset,
    simulate_constellation,
    compute_residual_sequences,
    build_orbital_neighbor_graph,
    fit_per_satellite_scaler,
    apply_per_satellite_scaler,
)
from model import OrbitGNN, anomaly_score

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Results directory ─────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"

# ── Default hyper-parameters ──────────────────────────────────────────
DEFAULT_WINDOW       = 8
DEFAULT_EPOCHS       = 40
DEFAULT_LR           = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_MC_SAMPLES   = 15
DEFAULT_SEED         = 42

# Chronological split fractions
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
# TEST  = 1 - TRAIN_FRAC - VAL_FRAC


# ════════════════════════════════════════════════════════════════════
# Reproducibility
# ════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_env(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("OrbitGNN  —  environment")
    logger.info("  Python   : %s", sys.version.split()[0])
    logger.info("  NumPy    : %s", np.__version__)
    logger.info("  PyTorch  : %s", torch.__version__)
    logger.info("  Device   : %s", args.device)
    logger.info("  Seed     : %d", args.seed)
    logger.info("  Dataset  : %s", args.dataset)
    if args.dataset == "real":
        logger.info("  Path     : %s", args.dataset_path)
        logger.info("  Window   : %s → %s", args.start_date, args.end_date)
    logger.info("=" * 60)


# ════════════════════════════════════════════════════════════════════
# Windowing
# ════════════════════════════════════════════════════════════════════

def make_windows(
    residuals: np.ndarray,
    labels:    np.ndarray,
    window:    int,
) -> Tuple[List, List, List, List]:
    """
    Slide a window of length `window` over the residual sequence.

    For index t in [window, T-1]:
      X[t] = residuals[:, t-window : t, :]   (S, window, 6)  ← input history
      Y[t] = residuals[:, t, :]              (S, 6)           ← target residual
      L[t] = labels[:, t+1]  if t+1 < T     (S,)             ← ground-truth label

    NOTE: label index is t+1 because residual[t] = obs[t+1] − predicted[t→t+1].
    """
    _, n_steps, _ = residuals.shape
    X, Y, L, T_idx = [], [], [], []
    for t in range(window, n_steps):
        X.append(residuals[:, t - window:t, :])
        Y.append(residuals[:, t, :])
        label_t = min(t + 1, labels.shape[1] - 1)
        L.append(labels[:, label_t])
        T_idx.append(t)
    return X, Y, L, T_idx


def chronological_split(n_windows: int) -> Tuple[int, int]:
    train_end = int(n_windows * TRAIN_FRAC)
    val_end   = int(n_windows * (TRAIN_FRAC + VAL_FRAC))
    return train_end, val_end


# ════════════════════════════════════════════════════════════════════
# Metrics (pure NumPy, no sklearn required)
# ════════════════════════════════════════════════════════════════════

def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney-U ROC-AUC."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under precision-recall curve (trapezoidal)."""
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    thresholds = np.sort(np.unique(scores))[::-1]
    prec, rec = [], []
    for thr in thresholds:
        pred = scores >= thr
        tp   = int(np.logical_and(pred, y_true == 1).sum())
        fp   = int(np.logical_and(pred, y_true == 0).sum())
        fn   = n_pos - tp
        p    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec.append(p); rec.append(r)
    pairs = sorted(zip(rec, prec))
    rs    = [p[0] for p in pairs]
    ps    = [p[1] for p in pairs]
    return float(np.trapezoid(ps, rs))


def prf1(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    pred  = (scores >= threshold).astype(int)
    tp    = int(np.logical_and(pred == 1, y_true == 1).sum())
    fp    = int(np.logical_and(pred == 1, y_true == 0).sum())
    fn    = int(np.logical_and(pred == 0, y_true == 1).sum())
    tn    = int(np.logical_and(pred == 0, y_true == 0).sum())
    p     = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return dict(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find threshold maximising F1 over a grid of score percentiles."""
    candidates = np.percentile(scores, np.linspace(30, 99, 70))
    best_f1, best_thr = 0.0, candidates[0]
    for thr in candidates:
        m = prf1(y_true, scores, thr)
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], thr
    return best_thr


def print_metrics(
    label: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict:
    auc  = roc_auc(y_true, scores)
    pauc = pr_auc(y_true, scores)
    thr  = threshold if threshold is not None else best_f1_threshold(y_true, scores)
    m    = prf1(y_true, scores, thr)

    logger.info(
        "%s  ROC-AUC=%.4f  PR-AUC=%.4f  P=%.3f  R=%.3f  F1=%.3f  "
        "(thr=%.4f  TP=%d FP=%d FN=%d)",
        label, auc, pauc, m["precision"], m["recall"], m["f1"],
        thr, m["tp"], m["fp"], m["fn"],
    )
    return {"roc_auc": auc, "pr_auc": pauc, "threshold": thr, **m}


# ════════════════════════════════════════════════════════════════════
# Event-level detection
# ════════════════════════════════════════════════════════════════════

def event_level_detection(
    maneuver_events:        Dict[str, List[datetime.datetime]],
    sat_names:              List[str],
    score_matrix:           np.ndarray,        # (S, T_test)
    test_timestamps:        List[datetime.datetime],
    threshold:              float,
    detection_window_hours: float = 72.0,
) -> Dict:
    """
    Evaluate manoeuvre detection at the event level.

    For each real manoeuvre event, check whether the model's anomaly score
    exceeds `threshold` at any grid slot within ±detection_window_hours.

    Detection-offset definition (FIX F5)
    -------------------------------------
    offset_hours = (grid_slot_time − manoeuvre_event_time) / 3600
    Negative → alarm BEFORE the official manoeuvre timestamp.
    Positive → alarm AFTER.

    NOTE: TLE-derived manoeuvre timestamps are approximate (ground-station
    fitting can lag the actual burn by 12–36 h).  A negative offset does NOT
    necessarily mean the model "predicted" the manoeuvre; it may simply mean
    the model detected the orbital state change before the ground station
    published updated elements.  We report offsets neutrally as
    "prediction-to-event time offset" without claiming predictive capability.
    """
    ts_arr    = np.array([t.timestamp() for t in test_timestamps])
    det_win_s = detection_window_hours * 3600.0

    n_events   = 0
    n_detected = 0
    offsets: List[float] = []

    for s_idx, sat_name in enumerate(sat_names):
        for ev in maneuver_events.get(sat_name, []):
            ev_ts = ev.timestamp()
            # Grid slots within detection window
            within = np.where(np.abs(ts_arr - ev_ts) <= det_win_s)[0]
            if len(within) == 0:
                continue
            n_events += 1
            if np.any(score_matrix[s_idx, within] >= threshold):
                n_detected += 1
                # First slot in window that exceeds threshold
                first_idx = within[np.argmax(score_matrix[s_idx, within] >= threshold)]
                offsets.append((ts_arr[first_idx] - ev_ts) / 3600.0)

    # False alarms: flagged slots with no manoeuvre within detection window
    false_alarms = 0
    S, T_t = score_matrix.shape
    for s_idx, sat_name in enumerate(sat_names):
        ev_ts_list = [ev.timestamp() for ev in maneuver_events.get(sat_name, [])]
        for t_idx in range(T_t):
            if score_matrix[s_idx, t_idx] >= threshold:
                near = any(abs(ts_arr[t_idx] - ets) <= det_win_s for ets in ev_ts_list)
                if not near:
                    false_alarms += 1

    fa_rate_per_day = false_alarms / max((ts_arr[-1] - ts_arr[0]) / 86400.0, 1.0)

    return {
        "n_events":              n_events,
        "n_detected":            n_detected,
        "n_missed":              n_events - n_detected,
        "detection_rate":        n_detected / n_events if n_events > 0 else float("nan"),
        "false_alarms_total":    false_alarms,
        "fa_rate_per_day":       fa_rate_per_day,
        "mean_offset_hours":     float(np.mean(offsets))  if offsets else float("nan"),
        "median_offset_hours":   float(np.median(offsets)) if offsets else float("nan"),
    }


# ════════════════════════════════════════════════════════════════════
# BASELINE MODELS
# ════════════════════════════════════════════════════════════════════

def baseline_residual_magnitude(
    residuals:  np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Baseline 1: L2 norm of the physics residual vector.

    Score(s, t) = ||residual[s, t]||₂

    valid_mask should be (S, T-1) aligned to the residuals array.
    """
    scores = np.linalg.norm(residuals, axis=-1)   # (S, T-1)
    # Set forward-filled slots to NaN
    if valid_mask.shape == scores.shape:
        scores[~valid_mask] = np.nan
    return scores   # (S, T-1)


def baseline_rolling_zscore(
    residuals:       np.ndarray,
    window:          int   = 8,
    min_periods:     int   = 3,
) -> np.ndarray:
    """
    Baseline 2: rolling z-score of residual L2 norm.

    At each step t, score = (norm[t] − mean(norm[t-w:t])) / std(norm[t-w:t]).
    Large positive scores indicate sudden residual increases (anomalies).
    """
    S, T_minus1, _ = residuals.shape
    norms  = np.linalg.norm(residuals, axis=-1)   # (S, T-1)
    scores = np.zeros_like(norms)

    for t in range(T_minus1):
        start = max(0, t - window)
        hist  = norms[:, start:t]
        if hist.shape[1] < min_periods:
            scores[:, t] = 0.0
            continue
        mu  = np.nanmean(hist, axis=1)
        std = np.nanstd(hist,  axis=1) + 1e-8
        scores[:, t] = (norms[:, t] - mu) / std

    return scores   # (S, T-1)


def baseline_isolation_forest(
    residuals:  np.ndarray,
    train_end:  int,
    seed:       int = 42,
) -> np.ndarray:
    """
    Baseline 3: Isolation Forest on the per-step residual vector.

    Trained on training split; scored on all time steps.
    Returns anomaly scores where higher = more anomalous
    (negated sklearn decision_function).
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        logger.warning("scikit-learn not installed; skipping IsolationForest baseline")
        return np.zeros(residuals.shape[:2])

    S, T_minus1, F = residuals.shape
    scores = np.zeros((S, T_minus1))

    for k in range(S):
        X_train = residuals[k, :train_end, :]   # (train_T, 6)
        X_all   = residuals[k, :, :]            # (T-1, 6)
        # Drop NaN rows
        valid   = ~np.any(np.isnan(X_train), axis=1)
        if valid.sum() < 10:
            continue
        clf = IsolationForest(n_estimators=100, random_state=seed)
        clf.fit(X_train[valid])
        # Higher score = more anomalous (negate sklearn convention)
        scores[k] = -clf.decision_function(X_all)

    return scores   # (S, T-1)


# ════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════

def plot_roc_curve(
    curves:   Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    filename: str = "roc_curve.png",
) -> None:
    """Plot multiple ROC curves for comparison."""
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=1.8, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random (0.500)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — held-out test split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=150); plt.close(fig)
    logger.info("Saved: %s", out)


def _compute_roc(y_true: np.ndarray, scores: np.ndarray):
    thresholds = np.sort(np.unique(scores))[::-1]
    n_pos = int(y_true.sum()); n_neg = len(y_true) - n_pos
    tpr, fpr = [0.0], [0.0]
    for thr in thresholds:
        pred = scores >= thr
        tp   = int(np.logical_and(pred, y_true == 1).sum())
        fp   = int(np.logical_and(pred, y_true == 0).sum())
        tpr.append(tp / n_pos if n_pos > 0 else 0.0)
        fpr.append(fp / n_neg if n_neg > 0 else 0.0)
    tpr.append(1.0); fpr.append(1.0)
    return np.array(fpr), np.array(tpr)


def plot_pr_curve(
    curves:   Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    baseline: float,
    filename: str = "pr_curve.png",
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, (rec, prec, auc) in curves.items():
        ax.plot(rec, prec, lw=1.8, label=f"{label} (PR-AUC={auc:.3f})")
    ax.axhline(baseline, color="grey", ls="--", lw=1,
               label=f"no-skill ({baseline:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — held-out test split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=150); plt.close(fig)
    logger.info("Saved: %s", out)


def _compute_pr(y_true: np.ndarray, scores: np.ndarray):
    n_pos = int(y_true.sum())
    thresholds = np.sort(np.unique(scores))[::-1]
    prec, rec = [1.0], [0.0]
    for thr in thresholds:
        pred = scores >= thr
        tp   = int(np.logical_and(pred, y_true == 1).sum())
        fp   = int(np.logical_and(pred, y_true == 0).sum())
        fn   = n_pos - tp
        p    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec.append(p); rec.append(r)
    prec.append(n_pos / len(y_true) if len(y_true) > 0 else 0.0)
    rec.append(1.0)
    return np.array(rec), np.array(prec)


def plot_anomaly_timeline(
    score_matrix:     np.ndarray,
    label_matrix:     np.ndarray,
    test_timestamps:  List[datetime.datetime],
    sat_names:        List[str],
    shell_id:         np.ndarray,
    maneuver_events:  Dict[str, List[datetime.datetime]],
    n_to_plot:        int = 4,
    filename:         str = "anomaly_timeline.png",
) -> None:
    """
    Plot anomaly score over the test window with real manoeuvre markers.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    # Prefer satellites with at least one manoeuvre in the test window
    ts0 = test_timestamps[0] if test_timestamps else None
    ts1 = test_timestamps[-1] if test_timestamps else None
    has_man = [
        any(ts0 <= m <= ts1 for m in maneuver_events.get(sat_names[k], []))
        if ts0 and ts1 else False
        for k in range(len(sat_names))
    ]
    picks = [k for k, h in enumerate(has_man) if h][:n_to_plot]
    if not picks:
        picks = list(range(min(n_to_plot, len(sat_names))))

    fig, axes = plt.subplots(len(picks), 1,
                             figsize=(13, 3.5 * len(picks)), sharex=True)
    if len(picks) == 1:
        axes = [axes]

    for ax, k in zip(axes, picks):
        name   = sat_names[k]
        scores = score_matrix[k]
        ts_list = test_timestamps[:len(scores)]

        ax.plot(ts_list, scores, color="steelblue", lw=1.2, label="anomaly score")

        man_plotted = False
        for mt in maneuver_events.get(name, []):
            if ts0 and ts1 and ts0 <= mt <= ts1:
                lbl = "manoeuvre (benchmark)" if not man_plotted else "_"
                ax.axvline(mt, color="crimson", alpha=0.75, lw=1.5, ls="--", label=lbl)
                man_plotted = True

        ax.set_ylabel(f"{name}\n(shell {shell_id[k]})", fontsize=8)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.legend(loc="upper left", fontsize=7)

    axes[-1].set_xlabel("UTC date")
    fig.suptitle("OrbitGNN anomaly score vs. real manoeuvre ground truth", fontsize=10)
    fig.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=150); plt.close(fig)
    logger.info("Saved: %s", out)


def plot_residuals_over_time(
    residuals:   np.ndarray,
    timestamps:  List[datetime.datetime],
    labels:      np.ndarray,
    sat_names:   List[str],
    filename:    str = "residuals_over_time.png",
) -> None:
    """Plot Δa and ΔM residuals for each satellite with manoeuvre labelling."""
    RESULTS_DIR.mkdir(exist_ok=True)
    S = residuals.shape[0]
    ts = timestamps[1:]   # residuals align to t+1

    fig, axes = plt.subplots(S, 2, figsize=(16, 2.5 * S), sharex=True)
    if S == 1:
        axes = [axes]

    for k in range(S):
        ax_a, ax_M = axes[k]
        da = residuals[k, :, 0]  # Δa (km)
        dM = residuals[k, :, 5]  # ΔM (rad)
        ts_k = ts[:len(da)]

        man_mask = (labels[k, 1:len(da)+1] > 0)

        ax_a.plot(ts_k, da, color="steelblue", lw=0.8)
        ax_a.scatter([ts_k[i] for i in range(len(ts_k)) if man_mask[i]],
                     da[man_mask], color="crimson", s=10, zorder=3, label="manoeuvre")
        ax_a.set_ylabel(f"{sat_names[k]}\nΔa (km)", fontsize=7)

        ax_M.plot(ts_k, dM, color="darkorange", lw=0.8)
        ax_M.scatter([ts_k[i] for i in range(len(ts_k)) if man_mask[i]],
                     dM[man_mask], color="crimson", s=10, zorder=3)
        ax_M.set_ylabel("ΔM (rad)", fontsize=7)

    axes[-1][0].set_xlabel("UTC date")
    axes[-1][1].set_xlabel("UTC date")
    for a in axes:
        for ax in a:
            ax.tick_params(axis="x", rotation=25, labelsize=6)

    fig.suptitle("Physics residuals Δa and ΔM (manoeuvre slots in red)", fontsize=10)
    fig.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=150); plt.close(fig)
    logger.info("Saved: %s", out)


# ════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ════════════════════════════════════════════════════════════════════

def save_checkpoint(model: nn.Module, path: Path, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(model: nn.Module, path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("meta", {})


# ════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    print_env(args)
    device = torch.device(args.device)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    if args.dataset == "real":
        dataset_path = args.dataset_path
        if dataset_path is None:
            candidates = [
                Path(__file__).parent.parent / "TLE_observation_benchmark_dataset-main",
                Path.home() / "Documents/dl/TLE_observation_benchmark_dataset-main",
            ]
            dataset_path = next((str(p) for p in candidates if p.is_dir()), None)
        if dataset_path is None or not Path(dataset_path).is_dir():
            logger.error("Cannot find TLE benchmark dataset. Use --dataset-path.")
            sys.exit(1)

        logger.info("Loading real benchmark from: %s", dataset_path)
        result = load_real_benchmark_dataset(
            dataset_path,
            start_date=args.start_date,
            end_date=args.end_date,
            dt_hours=args.dt_hours,
            max_tle_gap_hours=args.max_tle_gap_hours,
            maneuver_tolerance_hours=args.maneuver_tolerance_hours,
        )
        elements_obs    = result["elements_obs"]
        labels          = result["labels"]
        shell_id        = result["shell_id"]
        timestamps      = result["timestamps"]
        sat_names       = result["sat_names"]
        maneuver_events = result["maneuver_events"]
        dt_seconds_grid = result["dt_seconds_grid"]
        valid_mask      = result["valid_mask"]
        md              = result["metadata"]
        dt_hours        = args.dt_hours

        logger.info("── Dataset Summary ──────────────────────────────────────")
        logger.info("  Satellites   : %d  %s", md["n_satellites"], md["satellite_names"])
        logger.info("  Grid steps   : %d", md["n_grid_steps"])
        logger.info("  Date range   : %s → %s", md["grid_start"], md["grid_end"])
        logger.info("  Maneuvers    : %d events → %d labeled slots",
                    md["total_maneuver_events"], md["labeled_maneuver_slots"])
        logger.info("  Missing slots: %d (%.1f%%)",
                    md["missing_slots"], 100 * md["missing_fraction"])
        logger.info("  Shells       : %s", md["shell_distribution"])
        logger.info("  dt range     : %.1fh – %.1fh",
                    md["dt_seconds_grid_range"][0] / 3600,
                    md["dt_seconds_grid_range"][1] / 3600)

    else:  # synthetic
        logger.info("Loading synthetic constellation...")
        _, elements_obs, labels, shell_id = simulate_constellation(
            n_shells=3, sats_per_shell=14, n_steps=150,
            anomaly_rate=0.08, seed=args.seed,
        )
        timestamps      = None
        sat_names       = [f"SAT_{k}" for k in range(elements_obs.shape[0])]
        maneuver_events = {}
        dt_seconds_grid = None
        valid_mask      = np.ones(elements_obs.shape[:2], dtype=bool)
        dt_hours        = 6.0
        logger.info("  Satellites: %d  Grid steps: %d",
                    elements_obs.shape[0], elements_obs.shape[1])

    S, T, _ = elements_obs.shape

    # ── Compute physics residuals with actual dt ──────────────────────
    logger.info("Computing physics residuals (S=%d, T=%d)...", S, T)
    residuals = compute_residual_sequences(
        elements_obs, dt_seconds_grid=dt_seconds_grid, dt_hours=dt_hours
    )   # (S, T-1, 6)
    logger.info("  Residual ΔM: median=%.4f rad, p95=%.4f rad",
                np.median(np.abs(residuals[:, :, 5])),
                np.percentile(np.abs(residuals[:, :, 5]), 95))

    # ── Graph ─────────────────────────────────────────────────────────
    adj    = build_orbital_neighbor_graph(elements_obs[:, -1, :], shell_id,
                                          k_neighbors=3)
    adj_t  = torch.tensor(adj, dtype=torch.float32, device=device)
    n_edges = int(adj.sum() // 2)
    logger.info("Graph: %d nodes, %d edges", S, n_edges)
    for sh in np.unique(shell_id):
        members = [sat_names[k] for k in range(S) if shell_id[k] == sh]
        logger.info("  Shell %d (%d sats): %s", sh, len(members), members)

    # ── Windows & split ───────────────────────────────────────────────
    X, Y, L, T_idx = make_windows(residuals, labels, window=args.window)
    n_windows       = len(X)
    train_end, val_end = chronological_split(n_windows)

    if timestamps:
        def _ts(w): return timestamps[min(T_idx[w] + 1, T - 1)]
        logger.info("Split dates:")
        logger.info("  TRAIN      : %s → %s  (%d windows)",
                    _ts(0).date(), _ts(train_end - 1).date(), train_end)
        logger.info("  VALIDATION : %s → %s  (%d windows)",
                    _ts(train_end).date(), _ts(val_end - 1).date(), val_end - train_end)
        logger.info("  TEST       : %s → %s  (%d windows)",
                    _ts(val_end).date(), _ts(n_windows - 1).date(), n_windows - val_end)

    # ── Per-satellite normalisation (FIX F2) ─────────────────────────
    # Fit on training residuals ONLY, apply globally.
    train_res_seq = residuals[:, :train_end, :]   # (S, train_T, 6)
    sat_median, sat_scale = fit_per_satellite_scaler(residuals, train_end)

    # Normalised full residual sequence (S, T-1, 6)
    res_normed = apply_per_satellite_scaler(residuals, sat_median, sat_scale)

    # Rebuild windows from normalised residuals
    Xn, Yn, _, _ = make_windows(res_normed, labels, window=args.window)

    # ── Baselines ─────────────────────────────────────────────────────
    logger.info("── Baselines ────────────────────────────────────────────")

    valid_res_mask = valid_mask[:, :-1]   # (S, T-1): True when 'from' slot has real TLE

    b1_scores = baseline_residual_magnitude(residuals, valid_res_mask)
    b2_scores = baseline_rolling_zscore(residuals, window=args.window)
    b3_scores = baseline_isolation_forest(residuals, train_end, seed=args.seed)

    # Align all baseline scores to test split
    # Each score array is (S, T-1); windows index into T-1
    def _flatten_test(score_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flatten test-split scores and labels for evaluation."""
        sc_list, lab_list = [], []
        for w in range(val_end, n_windows):
            t = T_idx[w]   # residual index
            sc_list.append(score_arr[:, t])
            lab_list.append((L[w] > 0).astype(int))
        sc   = np.nan_to_num(np.concatenate(sc_list), nan=0.0)
        lab  = np.concatenate(lab_list)
        return sc, lab

    sc_b1, lab_test = _flatten_test(b1_scores)
    sc_b2, _        = _flatten_test(b2_scores)
    sc_b3, _        = _flatten_test(b3_scores)

    res_b1 = print_metrics("  Baseline-1 (ResidMag):  ", lab_test, sc_b1)
    res_b2 = print_metrics("  Baseline-2 (RollingZ):  ", lab_test, sc_b2)
    res_b3 = print_metrics("  Baseline-3 (IsoForest): ", lab_test, sc_b3)

    if args.baselines_only:
        logger.info("--baselines-only: skipping model training")
        return

    # ── Model training ─────────────────────────────────────────────────
    logger.info("── Training OrbitGNN (%d epochs) ────────────────────────", args.epochs)
    model      = OrbitGNN(in_dim=6, d_model=64, n_heads=4, n_layers=2,
                          dropout=0.2).to(device)
    n_params   = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)
    opt        = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    loss_fn    = nn.MSELoss()
    ckpt_path  = RESULTS_DIR / "best_model.pt"

    best_val_loss  = float("inf")
    best_val_epoch = 0
    model.train()

    for epoch in range(args.epochs):
        # Training: nominal windows only (self-supervised)
        epoch_loss, n_batches = 0.0, 0
        for w in np.random.permutation(train_end):
            nom_mask = (L[w] == 0)
            if nom_mask.sum() == 0:
                continue
            x    = torch.tensor(Xn[w], dtype=torch.float32, device=device)
            y    = torch.tensor(Yn[w], dtype=torch.float32, device=device)
            mask = torch.tensor(nom_mask, device=device)

            sf, pf, _ = model(x, adj_t)
            loss       = loss_fn(sf[mask], y[mask]) + loss_fn(pf[mask], y[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1

        # Validation (all windows, for loss monitoring only)
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for w in range(train_end, val_end):
                x    = torch.tensor(Xn[w], dtype=torch.float32, device=device)
                y    = torch.tensor(Yn[w], dtype=torch.float32, device=device)
                sf, pf, _ = model(x, adj_t)
                val_loss += (loss_fn(sf, y) + loss_fn(pf, y)).item()
                val_n    += 1
        model.train()

        cur_val = val_loss / max(val_n, 1)
        if cur_val < best_val_loss:
            best_val_loss  = cur_val
            best_val_epoch = epoch + 1
            save_checkpoint(model, ckpt_path,
                            {"epoch": epoch + 1, "val_loss": cur_val})

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("  epoch %3d/%d  train=%.5f  val=%.5f  (best val epoch %d)",
                        epoch + 1, args.epochs,
                        epoch_loss / max(n_batches, 1),
                        cur_val, best_val_epoch)

    logger.info("Best validation epoch: %d (val_loss=%.5f)", best_val_epoch, best_val_loss)

    # Load best checkpoint
    load_checkpoint(model, ckpt_path)
    model.eval()

    # ── Evaluation on TEST split ──────────────────────────────────────
    logger.info("── Evaluating on TEST split (%d windows) ────────────────",
                n_windows - val_end)

    all_scores, all_labels = [], []
    per_sat_scores: List[np.ndarray] = []
    per_sat_labels: List[np.ndarray] = []
    test_ts_list:   List[datetime.datetime] = []

    for w in range(val_end, n_windows):
        x   = torch.tensor(Xn[w], dtype=torch.float32, device=device)
        y   = torch.tensor(Yn[w], dtype=torch.float32, device=device)

        with torch.no_grad():
            sf, pf, _     = model(x, adj_t)
            mean_pf, std_pf = model.mc_dropout_forecast(x, adj_t,
                                                         n_samples=args.mc_samples)

        sc      = anomaly_score(y, sf, pf, std_pf).cpu().numpy()
        all_scores.append(sc)
        all_labels.append((L[w] > 0).astype(int))
        per_sat_scores.append(sc)
        per_sat_labels.append(L[w])

        if timestamps:
            t_offset = min(T_idx[w] + 1, T - 1)
            test_ts_list.append(timestamps[t_offset])

    sc_orbitgnn = np.concatenate(all_scores)
    lab_all     = np.concatenate(all_labels)

    # ── Metrics comparison ────────────────────────────────────────────
    logger.info("── Test metrics comparison ──────────────────────────────")
    thr_orbitgnn = best_f1_threshold(lab_all, sc_orbitgnn)
    res_gnn      = print_metrics("  OrbitGNN:               ", lab_all, sc_orbitgnn)
    print_metrics("  Baseline-1 (ResidMag):  ", lab_test, sc_b1)
    print_metrics("  Baseline-2 (RollingZ):  ", lab_test, sc_b2)
    if not np.all(sc_b3 == 0):
        print_metrics("  Baseline-3 (IsoForest): ", lab_test, sc_b3)

    # ── Event-level evaluation (real data only) ───────────────────────
    if args.dataset == "real" and test_ts_list:
        sm = np.stack(per_sat_scores, axis=1)  # (S, n_test)
        evr = event_level_detection(
            maneuver_events, sat_names, sm,
            test_ts_list, thr_orbitgnn,
        )
        logger.info("── Event-level manoeuvre detection ──────────────────────")
        logger.info("  Total events in test : %d", evr["n_events"])
        logger.info("  Detected             : %d (%.1f%%)",
                    evr["n_detected"], 100 * evr["detection_rate"])
        logger.info("  Missed               : %d", evr["n_missed"])
        logger.info("  False alarms total   : %d (%.2f /day)",
                    evr["false_alarms_total"], evr["fa_rate_per_day"])
        logger.info("  Prediction-to-event offset (detected only):")
        logger.info("    mean=%.1f h, median=%.1f h",
                    evr["mean_offset_hours"], evr["median_offset_hours"])
        logger.info("  NOTE: negative offset = alarm before official timestamp.")
        logger.info("        This may reflect ground-station fitting lag, not")
        logger.info("        genuine prediction of future manoeuvres.")

    # ── Plots ──────────────────────────────────────────────────────────
    roc_curves: Dict[str, Tuple] = {}
    pr_curves:  Dict[str, Tuple] = {}

    for name, sc, lab in [
        ("OrbitGNN",            sc_orbitgnn, lab_all),
        ("Baseline-1 ResidMag", sc_b1,       lab_test),
        ("Baseline-2 RollingZ", sc_b2,       lab_test),
    ]:
        fpr, tpr = _compute_roc(lab, sc)
        auc      = roc_auc(lab, sc)
        roc_curves[name] = (fpr, tpr, auc)

        rec, prec = _compute_pr(lab, sc)
        pauc      = pr_auc(lab, sc)
        pr_curves[name] = (rec, prec, pauc)

    plot_roc_curve(roc_curves)
    plot_pr_curve(pr_curves, baseline=float(lab_all.mean()))

    if args.dataset == "real" and test_ts_list:
        sm_labels = np.stack(per_sat_labels, axis=1)
        plot_anomaly_timeline(
            sm, sm_labels, test_ts_list, sat_names, shell_id, maneuver_events,
        )
        plot_residuals_over_time(
            residuals, timestamps or [], labels, sat_names,
        )

    # ── Final summary ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("  OrbitGNN ROC-AUC : %.4f", res_gnn["roc_auc"])
    logger.info("  OrbitGNN PR-AUC  : %.4f", res_gnn["pr_auc"])
    logger.info("  Baseline-1 AUC   : %.4f", res_b1["roc_auc"])
    logger.info("  Baseline-2 AUC   : %.4f", res_b2["roc_auc"])
    gnn_beats = res_gnn["roc_auc"] > max(res_b1["roc_auc"], res_b2["roc_auc"])
    logger.info("  OrbitGNN beats simple baselines: %s", gnn_beats)
    if not gnn_beats:
        logger.info("  → OrbitGNN does NOT outperform simple residual baselines.")
        logger.info("    This is scientifically honest and should be reported as such.")
    logger.info("=" * 60)


# ════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate OrbitGNN")
    p.add_argument("--dataset",       choices=["real", "synthetic"], default="real")
    p.add_argument("--dataset-path",  dest="dataset_path", default=None)
    p.add_argument("--start-date",    dest="start_date",   default="2020-01-01")
    p.add_argument("--end-date",      dest="end_date",     default="2022-01-01")
    p.add_argument("--dt-hours",      dest="dt_hours",     type=float, default=24.0)
    p.add_argument("--max-tle-gap",   dest="max_tle_gap_hours",
                   type=float, default=48.0)
    p.add_argument("--maneuver-tol",  dest="maneuver_tolerance_hours",
                   type=float, default=24.0)
    p.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",            type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay",  dest="weight_decay",
                   type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--window",        type=int,   default=DEFAULT_WINDOW)
    p.add_argument("--mc-samples",    dest="mc_samples",
                   type=int, default=DEFAULT_MC_SAMPLES)
    p.add_argument("--seed",          type=int,   default=DEFAULT_SEED)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--baselines-only", dest="baselines_only",
                   action="store_true", default=False,
                   help="Run only baseline models, skip OrbitGNN training")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
