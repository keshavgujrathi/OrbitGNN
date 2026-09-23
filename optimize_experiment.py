"""
optimize_experiment.py
----------------------
Run a controlled comparison between baseline OrbitGNN and improved OrbitGNN v2.

Rules:
  - Chronological 60/20/20 split.  No test-set access during training or tuning.
  - Threshold tuned on VALIDATION set only.
  - 3 random seeds, report mean ± std.
  - No metric fabrication.
  - Only improvements that genuinely improve VALIDATION metrics are kept.
"""

import sys, warnings, pathlib, csv
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from dataset import (load_real_benchmark_dataset, compute_residual_sequences,
                     build_orbital_neighbor_graph, fit_per_satellite_scaler,
                     apply_per_satellite_scaler)
from train import make_windows, chronological_split, best_f1_threshold
from model import OrbitGNN, anomaly_score
from model_v2 import OrbitGNNv2, anomaly_score_v2

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_PATH = "../TLE_observation_benchmark_dataset-main"
OUT_DIR = pathlib.Path("results/optimization")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGS = pathlib.Path("results/paper_figures")
FIGS.mkdir(exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
WINDOW        = 8
EPOCHS        = 120          # more epochs — we have early stopping
LR            = 8e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 20           # early stopping patience (on val loss)
MC_SAMPLES    = 20
SEEDS         = [1, 2, 3]

# ─── Data loading (once) ─────────────────────────────────────────────────────
print("Loading data...")
result = load_real_benchmark_dataset(
    DATASET_PATH, start_date="2020-01-01", end_date="2022-01-01",
    dt_hours=24.0, max_tle_gap_hours=48.0, maneuver_tolerance_hours=24.0)
eo     = result["elements_obs"]
labels = result["labels"]
sid    = result["shell_id"]
dts    = result["dt_seconds_grid"]
meta   = result["metadata"]
T, S   = meta["n_grid_steps"], meta["n_satellites"]

res    = compute_residual_sequences(eo, dts)
T_train = int(0.60 * (T - 1))
sat_mean, sat_scale = fit_per_satellite_scaler(res, train_end=T_train)
res_n  = apply_per_satellite_scaler(res, sat_mean, sat_scale)
X, Y, L, T_idx = make_windows(res_n, labels, window=WINDOW)
n_win  = len(X)
train_end, val_end = chronological_split(n_win)

X_np, Y_np, L_np = np.array(X), np.array(Y), np.array(L)

# Per-satellite normalisation (applied on top of scaler, using train only)
Xn = X_np.copy()
for k in range(S):
    tr = X_np[:train_end, k].reshape(-1, 6)
    mu = tr.mean(0); sg = tr.std(0) + 1e-8
    Xn[:, k] = ((X_np[:, k] - mu) / sg).clip(-10, 10)

X_tr = Xn[:train_end];  L_tr = L_np[:train_end]
X_va = Xn[train_end:val_end]; L_va = L_np[train_end:val_end]
X_te = Xn[val_end:];   L_te = L_np[val_end:]

# Per-satellite test labels (1305 samples = 145 windows × 9 sats)
y_ps_te  = L_te.flatten()
y_ps_va  = L_va.flatten()

print(f"Train={train_end}, Val={val_end-train_end}, Test={n_win-val_end}")
print(f"Test positives: {y_ps_te.sum()} / {len(y_ps_te)}")
print(f"Val  positives: {y_ps_va.sum()} / {len(y_ps_va)}")

# Graph (fixed for all experiments)
adj_np = build_orbital_neighbor_graph(eo[:, T // 2, :], sid,
                                      k_neighbors=3, cross_shell=False)
adj_t  = torch.tensor(adj_np, dtype=torch.float32)

# ─── Metrics helper ───────────────────────────────────────────────────────────
def full_metrics(y_true, scores, threshold=None, name=""):
    if threshold is None:
        threshold = best_f1_threshold(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    roc  = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
    pr   = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    bal  = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return dict(roc_auc=roc, pr_auc=pr, f1=f1, precision=prec, recall=rec,
                mcc=mcc, bal_acc=bal, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                threshold=float(threshold))

# ─── Training function (generic for both v1 and v2) ──────────────────────────
def train_model(model, seed, version="v1", lr=LR, wd=WEIGHT_DECAY,
                epochs=EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_val, patience_cnt, best_sd = float("inf"), 0, None

    for ep in range(epochs):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        perm = torch.randperm(train_end)
        ep_loss = 0.0
        for i in range(train_end):
            w = int(perm[i])
            x = torch.tensor(Xn[w], dtype=torch.float32)
            y = torch.tensor(Y_np[w], dtype=torch.float32)
            sf, pf, _ = model(x, adj_t)
            loss = nn.functional.mse_loss(sf, y) + nn.functional.mse_loss(pf, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()

        # ── Validation loss (for early stopping only) ──────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for w in range(train_end, val_end):
                x = torch.tensor(Xn[w], dtype=torch.float32)
                y = torch.tensor(Y_np[w], dtype=torch.float32)
                sf, pf, _ = model(x, adj_t)
                val_loss += (nn.functional.mse_loss(sf, y) +
                             nn.functional.mse_loss(pf, y)).item()

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    Early stop at epoch {ep+1}")
                break

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"    ep {ep+1:3d}  train={ep_loss/train_end:.5f}  val={val_loss:.5f}")

    model.load_state_dict(best_sd)
    return model

# ─── Scoring for v1 ───────────────────────────────────────────────────────────
def score_v1(model, X_split, Y_split, start_idx):
    model.eval()
    sc_ps = []
    with torch.no_grad():
        for i in range(len(X_split)):
            x = torch.tensor(X_split[i], dtype=torch.float32)
            y = torch.tensor(Y_split[start_idx + i], dtype=torch.float32)
            sf, pf, _ = model(x, adj_t)
            _, std_pf  = model.mc_dropout_forecast(x, adj_t, n_samples=MC_SAMPLES)
            sc_ps.append(anomaly_score(y, sf, pf, std_pf).numpy())
    return np.array(sc_ps).flatten()

# ─── Scoring for v2 ───────────────────────────────────────────────────────────
def score_v2(model, X_split, Y_split, start_idx):
    model.eval()
    sc_ps = []
    log_w = model.log_w.detach()
    with torch.no_grad():
        for i in range(len(X_split)):
            x = torch.tensor(X_split[i], dtype=torch.float32)
            y = torch.tensor(Y_split[start_idx + i], dtype=torch.float32)
            sf, pf, _ = model(x, adj_t)
            _, std_pf  = model.mc_dropout_forecast(x, adj_t, n_samples=MC_SAMPLES)
            sc_ps.append(anomaly_score_v2(y, sf, pf, std_pf, log_w=log_w).numpy())
    return np.array(sc_ps).flatten()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════
results_v1, results_v2 = [], []

for seed in SEEDS:
    print(f"\n{'='*55}")
    print(f"  SEED {seed}")
    print(f"{'='*55}")

    # ── v1 baseline ────────────────────────────────────────────────────
    print("  [v1] Training baseline OrbitGNN ...")
    m1 = OrbitGNN(in_dim=6, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
    m1 = train_model(m1, seed, version="v1")

    sc1_va = score_v1(m1, X_va, Y_np, train_end)
    sc1_te = score_v1(m1, X_te, Y_np, val_end)
    thr1   = best_f1_threshold(y_ps_va, sc1_va)
    m1_val = full_metrics(y_ps_va, sc1_va, threshold=thr1, name="v1-val")
    m1_te  = full_metrics(y_ps_te, sc1_te, threshold=thr1, name="v1-test")
    print(f"  [v1] VAL  ROC={m1_val['roc_auc']:.4f}  F1={m1_val['f1']:.4f}  MCC={m1_val['mcc']:.4f}")
    print(f"  [v1] TEST ROC={m1_te['roc_auc']:.4f}  F1={m1_te['f1']:.4f}  MCC={m1_te['mcc']:.4f}")
    results_v1.append(m1_te)

    # ── v2 improved ────────────────────────────────────────────────────
    print("  [v2] Training improved OrbitGNN v2 ...")
    m2 = OrbitGNNv2(in_dim=6, d_model=96, n_heads=4, n_layers=3, dropout=0.2, ff_mult=3)
    m2 = train_model(m2, seed, version="v2", lr=LR, epochs=EPOCHS, patience=PATIENCE)

    sc2_va = score_v2(m2, X_va, Y_np, train_end)
    sc2_te = score_v2(m2, X_te, Y_np, val_end)
    thr2   = best_f1_threshold(y_ps_va, sc2_va)
    m2_val = full_metrics(y_ps_va, sc2_va, threshold=thr2, name="v2-val")
    m2_te  = full_metrics(y_ps_te, sc2_te, threshold=thr2, name="v2-test")
    print(f"  [v2] VAL  ROC={m2_val['roc_auc']:.4f}  F1={m2_val['f1']:.4f}  MCC={m2_val['mcc']:.4f}")
    print(f"  [v2] TEST ROC={m2_te['roc_auc']:.4f}  F1={m2_te['f1']:.4f}  MCC={m2_te['mcc']:.4f}")
    results_v2.append(m2_te)

    # Save best v2 model (seed with best val ROC)
    if seed == SEEDS[0] or m2_val['roc_auc'] > best_val_roc_v2:
        best_val_roc_v2 = m2_val['roc_auc']
        torch.save({'state_dict': m2.state_dict(), 'seed': seed},
                   'results/best_model_v2.pt')
        # also save its scores for plotting
        best_sc2_te = sc2_te.copy()
        best_sc1_te = sc1_te.copy()

# ─── Summary statistics ───────────────────────────────────────────────────────
def summarise(results, name):
    keys = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'mcc', 'bal_acc']
    print(f"\n{'='*60}")
    print(f"  {name}  (mean ± std over {len(SEEDS)} seeds)")
    print(f"{'='*60}")
    summary = {}
    for k in keys:
        vals = [r[k] for r in results]
        mu, sd = np.mean(vals), np.std(vals)
        summary[k] = (mu, sd)
        print(f"  {k:<15} {mu:.4f} ± {sd:.4f}")
    return summary

s1 = summarise(results_v1, "OrbitGNN v1 (baseline)")
s2 = summarise(results_v2, "OrbitGNN v2 (improved)")

print(f"\n{'='*60}")
print("  IMPROVEMENT (v2 − v1, mean)")
print(f"{'='*60}")
for k in s1:
    delta = s2[k][0] - s1[k][0]
    sign  = "+" if delta >= 0 else ""
    print(f"  {k:<15} {sign}{delta:+.4f}")

# ─── Save comparison CSV ─────────────────────────────────────────────────────
rows = []
for i, (r1, r2) in enumerate(zip(results_v1, results_v2)):
    rows.append({"model": "v1_baseline", "seed": SEEDS[i], **r1})
    rows.append({"model": "v2_improved", "seed": SEEDS[i], **r2})

with open(OUT_DIR / "comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"\nSaved: {OUT_DIR/'comparison.csv'}")

# ─── Figure: comparison bar chart ────────────────────────────────────────────
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
metrics_plot = ['roc_auc', 'pr_auc', 'f1', 'mcc', 'bal_acc']
metric_labels = ['ROC-AUC', 'PR-AUC', 'F1', 'MCC', 'Bal. Acc.']

fig, axes = plt.subplots(1, len(metrics_plot), figsize=(14, 5))
x = np.arange(2)
for ax_i, (mk, ml) in enumerate(zip(metrics_plot, metric_labels)):
    ax = axes[ax_i]
    v1_vals = [r[mk] for r in results_v1]
    v2_vals = [r[mk] for r in results_v2]
    mu1, sd1 = np.mean(v1_vals), np.std(v1_vals)
    mu2, sd2 = np.mean(v2_vals), np.std(v2_vals)
    bars = ax.bar([0, 1], [mu1, mu2], yerr=[sd1, sd2],
                  color=['#AAAAAA', '#CC2222'], edgecolor='k',
                  linewidth=0.8, capsize=5, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline\n(v1)', 'Improved\n(v2)'], fontsize=10)
    ax.set_title(f'({chr(97+ax_i)}) {ml}', fontweight='bold', fontsize=11)
    ax.set_ylabel(ml, fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.axhline(0, color='k', linewidth=0.5)
    # Annotate
    for bar, mu in zip(bars, [mu1, mu2]):
        ax.text(bar.get_x()+bar.get_width()/2, mu + (0.005 if mu >= 0 else -0.02),
                f'{mu:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('OrbitGNN v1 (Baseline) vs v2 (Improved) — Test Set\n'
             'Mean ± Std over 3 Seeds, Per-Satellite Evaluation',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGS / 'fig14_v1_vs_v2_comparison.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIGS / 'fig14_v1_vs_v2_comparison.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig14 saved")

# ─── Final summary table to stdout ───────────────────────────────────────────
print(f"\n{'='*75}")
print(f"{'Metric':<15} {'v1 baseline':>18} {'v2 improved':>18} {'Δ':>10}")
print(f"{'='*75}")
for k in s1:
    m1_str = f"{s1[k][0]:.4f} ± {s1[k][1]:.4f}"
    m2_str = f"{s2[k][0]:.4f} ± {s2[k][1]:.4f}"
    delta  = s2[k][0] - s1[k][0]
    sign   = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "~")
    print(f"{k:<15} {m1_str:>18} {m2_str:>18} {sign}{delta:+.4f}")
print(f"{'='*75}")
