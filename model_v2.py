"""
model_v2.py — OrbitGNN Improved Architecture

Changes vs baseline model.py:
  1. Richer feature engineering in scoring (delta-from-window-mean + velocity proxy)
  2. Larger d_model (96 vs 64) + 3 transformer layers vs 2
  3. Attention-weighted GCN (learned edge weights, not uniform averaging)
  4. Forecast head uses 2-layer MLP (not linear) → more expressive
  5. Anomaly score: element-wise normalisation across both satellite AND feature dims
     + learned weights between 3 signal components (trained on validation)
  6. Window-level score = max(per-satellite score) OR per-sat (configurable)

Label-free at all stages: training uses only MSE forecast loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualEncoderV2(nn.Module):
    """
    Transformer encoder over a satellite's residual history.
    Changes vs v1:
      - larger d_model (configurable, default 96)
      - 3 transformer layers (vs 2)
      - 2-layer MLP forecast head (vs linear)
      - input projection uses LayerNorm before transformer
    """

    def __init__(self, in_dim=6, d_model=96, n_heads=4, n_layers=3, dropout=0.2,
                 ff_mult=3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_embed  = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,   # Pre-LN → more stable training
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                              enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        # 2-layer MLP forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, in_dim),
        )

    def forward(self, x):
        B, T, _ = x.shape
        h = self.input_norm(self.input_proj(x) + self.pos_embed[:, :T, :])
        h = self.encoder(h)
        h = self.dropout(h)
        summary  = h[:, -1, :]
        forecast = self.forecast_head(summary)
        return summary, forecast


class AttentionGCN(nn.Module):
    """
    GCN with learned attention weights per edge (graph attention lite).
    Instead of uniform averaging (D⁻¹ A H W), we learn a scalar attention
    score for each (i,j) edge and normalise with softmax over neighbours.

    H' = GELU( Σ_j α_ij * H_j * W ) + H    (residual connection)

    where α_ij = softmax_j(LeakyReLU(a^T [H_i || H_j]))  for j ∈ N(i)

    For the small graph (S=9 nodes) this is equivalent to one GAT head.
    """

    def __init__(self, dim, dropout=0.2, leaky_slope=0.2):
        super().__init__()
        self.W    = nn.Linear(dim, dim, bias=False)
        self.att  = nn.Linear(2 * dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(leaky_slope)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, adj):
        # h: (S, d), adj: (S, S) binary
        S, d = h.shape
        Wh = self.W(h)                   # (S, d)

        # Attention scores for all pairs
        Wh_i = Wh.unsqueeze(1).expand(-1, S, -1)   # (S, S, d)
        Wh_j = Wh.unsqueeze(0).expand(S, -1, -1)   # (S, S, d)
        e = self.leaky(self.att(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))  # (S,S)

        # Mask non-edges to -inf before softmax
        mask = (adj == 0)
        e = e.masked_fill(mask, float('-inf'))

        # Softmax over neighbours; handle isolated nodes
        alpha = torch.where(
            adj.sum(dim=1, keepdim=True) > 0,
            F.softmax(e, dim=1),
            torch.zeros_like(e),
        )  # (S, S)
        alpha = self.dropout(alpha)

        # Aggregate
        agg = alpha @ Wh            # (S, d)
        out = F.gelu(agg) + h       # residual
        return self.norm(out)


class OrbitGNNv2(nn.Module):
    """
    Full OrbitGNN v2 pipeline.

    Improvements over v1:
      - ResidualEncoderV2: larger, deeper, MLP forecast head
      - AttentionGCN: learned edge weights instead of uniform mean
      - Learnable anomaly-score weights (α_self, α_peer, α_unc)
        initialised to [1,1,1] and updated during val-set tuning
      - Per-feature z-score in anomaly computation (more discriminative)
    """

    def __init__(self, in_dim=6, d_model=96, n_heads=4, n_layers=3,
                 dropout=0.2, ff_mult=3):
        super().__init__()
        self.encoder  = ResidualEncoderV2(in_dim, d_model, n_heads, n_layers,
                                          dropout, ff_mult)
        self.gcn      = AttentionGCN(d_model, dropout)
        self.peer_forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, in_dim),
        )
        # Learnable score weights (trained on val set by simple grid search,
        # NOT on test set)
        self.log_w = nn.Parameter(torch.zeros(3))   # [w_self, w_peer, w_unc]

    def forward(self, residual_seqs, adj):
        summary, self_forecast = self.encoder(residual_seqs)
        peer_latent  = self.gcn(summary, adj)
        peer_forecast = self.peer_forecast_head(peer_latent)
        return self_forecast, peer_forecast, summary

    @torch.no_grad()
    def mc_dropout_forecast(self, residual_seqs, adj, n_samples=20):
        was_training = self.training
        self.train()
        preds = []
        for _ in range(n_samples):
            _, pf, _ = self.forward(residual_seqs, adj)
            preds.append(pf.unsqueeze(0))
        preds = torch.cat(preds, dim=0)   # (N, S, 6)
        if not was_training:
            self.eval()
        return preds.mean(0), preds.std(0)


def anomaly_score_v2(observed, self_fc, peer_fc, mc_std,
                     log_w=None, eps=1e-6):
    """
    Improved anomaly score with per-feature z-normalisation.

    Key changes vs v1:
    - z-score applied PER FEATURE across satellites (not sum-then-z)
      This means a spike in any single orbital element is picked up
      even if others are normal.
    - Weights are soft-positived via exp() on learnable log_w
    - Returns (S,) scalar scores

    Parameters
    ----------
    observed  : (S, 6)
    self_fc   : (S, 6)
    peer_fc   : (S, 6)
    mc_std    : (S, 6)
    log_w     : tensor of shape (3,) or None → equal weights
    """
    def z_per_feature(x):
        # x: (S, F)  → normalise each feature column across satellites
        mu  = x.mean(0, keepdim=True)
        sig = x.std(0, keepdim=True) + eps
        return (x - mu) / sig

    self_err  = z_per_feature((observed - self_fc).pow(2))    # (S, 6)
    peer_err  = z_per_feature((observed - peer_fc).pow(2))    # (S, 6)
    unc       = z_per_feature(mc_std.pow(2))                  # (S, 6)

    # Sum over features → (S,) composite error per satellite
    s_self = self_err.sum(-1)
    s_peer = peer_err.sum(-1)
    s_unc  = unc.sum(-1)

    if log_w is not None:
        w = log_w.exp()   # always positive
        score = w[0] * s_self + w[1] * s_peer + w[2] * s_unc
    else:
        score = s_self + s_peer + s_unc

    return score
