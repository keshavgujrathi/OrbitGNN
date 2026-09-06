"""
train.py
--------
Trains OrbitGNN to forecast next-step physics residuals using ONLY nominal
(non-anomalous) windows -- this is standard self-supervised anomaly
detection practice: the model never sees an anomaly label during training,
it only learns "what normal orbital-residual behavior looks like." At test
time, anything the model can't forecast well (self OR peer) or is unsure
about (MC-dropout variance) gets flagged. We use the injected labels ONLY
for evaluation (AUC/ROC), never for training -- which is exactly how you'd
have to operate on real satellites, since you don't have failure labels
in advance either.

Run:  python train.py
Needs: pip install torch   (numpy/scipy already used elsewhere)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import simulate_constellation, compute_residual_sequences, build_orbital_neighbor_graph
from model import OrbitGNN, anomaly_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

WINDOW = 8          # timesteps of residual history fed to the encoder per sample
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_windows(residuals, labels, window=WINDOW):
    """
    residuals: (S, T-1, 6), labels: (S, T)
    Slice into overlapping windows so the model sees a short residual
    history and forecasts the residual right after it. Also carries the
    ground-truth label of the *target* timestep for eval only.
    """
    n_sats, n_steps, _ = residuals.shape
    X, Y, L, T_idx = [], [], [], []
    for t in range(window, n_steps):
        X.append(residuals[:, t - window:t, :])   # (S, window, 6)
        Y.append(residuals[:, t, :])              # (S, 6)  <- target to forecast
        L.append(labels[:, t + 1])                # label aligns with residual index t -> obs t+1
        T_idx.append(t)
    return X, Y, L, T_idx


def train():
    print(f"device: {DEVICE}")
    elements_true, elements_obs, labels, shell_id = simulate_constellation(
        n_shells=3, sats_per_shell=14, n_steps=150, anomaly_rate=0.08, seed=42
    )
    residuals = compute_residual_sequences(elements_obs)          # (S, T-1, 6)
    n_sats = residuals.shape[0]

    # build ONE adjacency from the (roughly static) orbital-plane geometry
    adj = build_orbital_neighbor_graph(elements_obs[:, -1, :], k_neighbors=4)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=DEVICE)

    X, Y, L, T_idx = make_windows(residuals, labels)
    n_windows = len(X)
    split = int(n_windows * 0.7)

    # normalize residuals using TRAIN-split statistics only
    train_res = np.concatenate([x.reshape(-1, 6) for x in X[:split]], axis=0)
    mu, sigma = train_res.mean(0), train_res.std(0) + 1e-8

    def norm(a):
        return (a - mu) / sigma

    model = OrbitGNN(in_dim=6, d_model=64, n_heads=4, n_layers=2, dropout=0.2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    EPOCHS = 40
    model.train()
    for epoch in range(EPOCHS):
        # only train on windows whose TARGET is labeled nominal (label==0)
        # -> the model only ever learns "what normal looks like"
        epoch_loss, n_batches = 0.0, 0
        idx_order = np.random.permutation(split)
        for w in idx_order:
            target_labels = L[w]
            nominal_mask = (target_labels == 0)
            if nominal_mask.sum() == 0:
                continue

            x = torch.tensor(norm(X[w]), dtype=torch.float32, device=DEVICE)   # (S, window, 6)
            y = torch.tensor(norm(Y[w]), dtype=torch.float32, device=DEVICE)   # (S, 6)

            self_forecast, peer_forecast, _ = model(x, adj_t)

            mask = torch.tensor(nominal_mask, device=DEVICE)
            loss = loss_fn(self_forecast[mask], y[mask]) + loss_fn(peer_forecast[mask], y[mask])

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d}/{EPOCHS}  train_loss={epoch_loss / max(n_batches,1):.4f}")

    # ---- evaluation on held-out windows (includes anomalies) ----
    model.eval()
    all_scores, all_labels = [], []
    per_sat_scores, per_sat_labels, eval_t_idx = [], [], []  # kept per-satellite for plotting
    for w in range(split, n_windows):
        x = torch.tensor(norm(X[w]), dtype=torch.float32, device=DEVICE)
        y = torch.tensor(norm(Y[w]), dtype=torch.float32, device=DEVICE)

        self_forecast, peer_forecast, _ = model(x, adj_t)
        mean_pf, std_pf = model.mc_dropout_forecast(x, adj_t, n_samples=15)

        scores = anomaly_score(y, self_forecast, peer_forecast, std_pf)
        scores_np = scores.detach().cpu().numpy()

        all_scores.append(scores_np)
        all_labels.append((L[w] > 0).astype(int))
        per_sat_scores.append(scores_np)          # (S,) for this timestep
        per_sat_labels.append(L[w])               # (S,) raw label (0/1/2/3) for this timestep
        eval_t_idx.append(T_idx[w])

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc(all_labels, all_scores)
    print(f"\nHeld-out anomaly-detection ROC-AUC: {auc:.3f}")
    print(f"Held-out anomaly rate: {all_labels.mean():.3f}  |  n={len(all_labels)}")

    top_k = np.argsort(all_scores)[::-1][:20]
    hit_rate = all_labels[top_k].mean()
    print(f"Precision@20 (top-20 highest-scored windows): {hit_rate:.2f}")

    # per_sat_scores/labels: list of (S,) arrays -> stack into (n_eval_steps, S) -> transpose to (S, n_eval_steps)
    score_matrix = np.stack(per_sat_scores, axis=0).T   # (S, n_eval_steps)
    label_matrix = np.stack(per_sat_labels, axis=0).T   # (S, n_eval_steps)
    eval_t_idx = np.array(eval_t_idx)

    plot_anomaly_timeline(score_matrix, label_matrix, eval_t_idx, shell_id)
    plot_roc_curve(all_labels, all_scores)

    return model, adj_t, mu, sigma


def plot_anomaly_timeline(score_matrix, label_matrix, t_idx, shell_id, n_sats_to_plot=3):
    """
    Saves results/anomaly_timeline.png: for a few example satellites, plots
    the anomaly score over the held-out timesteps with injected-anomaly
    timesteps marked. This is the single chart that makes the detector's
    behavior legible in a live demo instead of a printed AUC number.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # pick satellites with at least one anomaly in the held-out window, so the
    # demo chart actually shows a spike instead of a flat nominal line
    has_anomaly = (label_matrix > 0).any(axis=1)
    candidates = np.where(has_anomaly)[0]
    if len(candidates) == 0:
        candidates = np.arange(label_matrix.shape[0])
    picks = candidates[:n_sats_to_plot]

    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3 * len(picks)), sharex=True)
    if len(picks) == 1:
        axes = [axes]

    for ax, sat_idx in zip(axes, picks):
        scores = score_matrix[sat_idx]
        labels = label_matrix[sat_idx]
        ax.plot(t_idx, scores, color="tab:blue", label="anomaly score")
        anomaly_steps = t_idx[labels > 0]
        if len(anomaly_steps) > 0:
            for step in anomaly_steps:
                ax.axvline(step, color="tab:red", alpha=0.3, linestyle="--")
            ax.axvline(anomaly_steps[0], color="tab:red", alpha=0.3, linestyle="--",
                       label="injected anomaly")
        ax.set_ylabel(f"sat #{sat_idx}\n(shell {shell_id[sat_idx]})")
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("timestep")
    fig.suptitle("OrbitGNN anomaly score vs. injected ground-truth events")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "anomaly_timeline.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_roc_curve(y_true, scores):
    """Saves results/roc_curve.png — plain-numpy ROC curve, no sklearn needed."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    thresholds = np.sort(np.unique(scores))[::-1]
    tpr, fpr = [], []
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    for thr in thresholds:
        pred = scores >= thr
        tp = np.logical_and(pred, y_true == 1).sum()
        fp = np.logical_and(pred, y_true == 0).sum()
        tpr.append(tp / n_pos if n_pos > 0 else 0)
        fpr.append(fp / n_neg if n_neg > 0 else 0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="tab:blue", label="OrbitGNN")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve — held-out anomaly detection")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "roc_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def roc_auc(y_true, scores):
    """Plain-numpy ROC-AUC (rank-based, no sklearn dependency needed)."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


if __name__ == "__main__":
    train()
