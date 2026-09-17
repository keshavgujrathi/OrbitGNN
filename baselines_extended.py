"""
Extended baselines for OrbitGNN comparison study.
All models receive the same 8-step physics residual windows as OrbitGNN.
Train/val/test split is identical (chronological 60/20/20).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')

# ── Extended metrics ──────────────────────────────────────────────────────────
def extended_metrics(y_true, y_score, threshold=None):
    """Compute full metric suite from scores + labels."""
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    pr  = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    # Best-F1 threshold
    if threshold is None:
        from train import best_f1_threshold
        threshold = best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    acc   = accuracy_score(y_true, y_pred)
    bal   = balanced_accuracy_score(y_true, y_pred)
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc   = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    kappa = cohen_kappa_score(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    return dict(roc_auc=roc, pr_auc=pr, precision=prec, recall=rec, f1=f1,
                accuracy=acc, balanced_accuracy=bal, specificity=spec,
                mcc=mcc, kappa=kappa, tp=int(tp), fp=int(fp),
                fn=int(fn), tn=int(tn), threshold=float(threshold))

# ── Feature helpers ───────────────────────────────────────────────────────────
def windows_to_flat(X):
    """Flatten (N, S, W, 6) → (N, S*W*6), then mean over S → (N, W*6)."""
    N, S, W, F = X.shape
    return X.mean(axis=1).reshape(N, W * F)  # (N, W*6)

def windows_to_per_sat_flat(X):
    """Returns (N*S, W*F) — treat each satellite independently."""
    N, S, W, F = X.shape
    return X.reshape(N * S, W * F)

# ── LSTM baseline ─────────────────────────────────────────────────────────────
class LSTMDetector(nn.Module):
    def __init__(self, input_size=6, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                   nn.Linear(32, 1))
    def forward(self, x):  # x: (B, W, 6)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

def run_lstm(X_train, L_train, X_val, L_val, X_test, L_test, n_epochs=40, lr=1e-3):
    """Train LSTM on averaged-satellite windows."""
    # Average over satellite dimension → (N, W, 6)
    Xtr = torch.tensor(X_train.mean(1), dtype=torch.float32)
    Xva = torch.tensor(X_val.mean(1),   dtype=torch.float32)
    Xte = torch.tensor(X_test.mean(1),  dtype=torch.float32)
    Ltr = torch.tensor(L_train.any(1).astype(float), dtype=torch.float32)
    model = LSTMDetector()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1-Ltr.mean())/Ltr.mean()]))
    best_val, best_sd = 1e9, None
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i+32]
            loss = crit(model(Xtr[idx]), Ltr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva), torch.tensor(L_val.any(1).astype(float), dtype=torch.float32))
        if vl.item() < best_val:
            best_val = vl.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(Xte)).numpy()
    return scores

# ── LSTM-Autoencoder ──────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=6, hidden=32):
        super().__init__()
        self.enc = nn.LSTM(input_size, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, input_size, batch_first=True)
    def forward(self, x):  # x: (B, W, 6)
        _, (h, c) = self.enc(x)
        # Expand hidden to full sequence for decoder
        h_exp = h[-1:].permute(1, 0, 2).expand(-1, x.size(1), -1)
        out, _ = self.dec(h_exp)
        return out

def run_lstm_ae(X_train, L_train, X_val, L_val, X_test, L_test, n_epochs=40, lr=1e-3):
    """Train LSTM-AE on NORMAL windows only; score by reconstruction error."""
    Xtr_all = X_train.mean(1)  # (N, W, 6)
    normal_mask = ~L_train.any(1)
    Xtr = torch.tensor(Xtr_all[normal_mask], dtype=torch.float32)
    Xte = torch.tensor(X_test.mean(1), dtype=torch.float32)
    model = LSTMAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_sd = 1e9, None
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        ep_loss = 0
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i+32]
            xb = Xtr[idx]
            loss = nn.MSELoss()(model(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        rec = model(Xte)
        scores = ((Xte - rec) ** 2).mean(dim=(1, 2)).numpy()
    return scores

# ── 1D-CNN baseline ───────────────────────────────────────────────────────────
class CNN1DDetector(nn.Module):
    def __init__(self, input_size=6, seq_len=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):  # x: (B, W, 6) → needs (B, 6, W)
        return self.head(self.conv(x.permute(0, 2, 1)).squeeze(-1)).squeeze(-1)

def run_cnn1d(X_train, L_train, X_val, L_val, X_test, L_test, n_epochs=40, lr=1e-3):
    Xtr = torch.tensor(X_train.mean(1), dtype=torch.float32)
    Xva = torch.tensor(X_val.mean(1),   dtype=torch.float32)
    Xte = torch.tensor(X_test.mean(1),  dtype=torch.float32)
    Ltr = torch.tensor(L_train.any(1).astype(float), dtype=torch.float32)
    model = CNN1DDetector()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_w = torch.tensor([(1-Ltr.mean())/Ltr.mean().clamp(min=0.01)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best_val, best_sd = 1e9, None
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i+32]
            loss = crit(model(Xtr[idx]), Ltr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva), torch.tensor(L_val.any(1).astype(float), dtype=torch.float32))
        if vl.item() < best_val:
            best_val = vl.item(); best_sd = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(Xte)).numpy()
    return scores

# ── Random Forest ─────────────────────────────────────────────────────────────
def run_random_forest(X_train, L_train, X_val, L_val, X_test, L_test):
    Xtr_f = windows_to_flat(X_train); Ltr = L_train.any(1).astype(int)
    Xte_f = windows_to_flat(X_test)
    sc = StandardScaler().fit(Xtr_f)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced',
                                  random_state=42, n_jobs=-1)
    clf.fit(sc.transform(Xtr_f), Ltr)
    scores = clf.predict_proba(sc.transform(Xte_f))[:, 1]
    return scores

# ── XGBoost ───────────────────────────────────────────────────────────────────
def run_xgboost(X_train, L_train, X_val, L_val, X_test, L_test):
    Xtr_f = windows_to_flat(X_train); Ltr = L_train.any(1).astype(int)
    Xte_f = windows_to_flat(X_test)
    sc = StandardScaler().fit(Xtr_f)
    n_pos = Ltr.sum(); n_neg = len(Ltr) - n_pos
    scale = n_neg / max(n_pos, 1)
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              scale_pos_weight=scale, eval_metric='logloss',
                              random_state=42, verbosity=0)
    clf.fit(sc.transform(Xtr_f), Ltr)
    scores = clf.predict_proba(sc.transform(Xte_f))[:, 1]
    return scores
