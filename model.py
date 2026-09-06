"""
model.py
--------
OrbitGNN: a three-part architecture. Each part answers one question a
professor will actually ask you in the Q&A after your 10-minute talk.

  1. "Why is this deep learning and not just an SGP4 diff?"
     -> ResidualEncoder: a small Transformer that learns the TEMPORAL
        pattern of a satellite's own physics-residual history and forecasts
        the next residual. Nominal satellites have boring, near-zero,
        near-stationary residual sequences (physics explains them fine).
        Degrading/maneuvering satellites have structured, non-stationary
        residual sequences. Forecasting error on this residual = anomaly
        signal #1.

  2. "Why compare a satellite to itself? Real anomaly detection needs
      context." -> NeighborGCN: a graph layer over the orbital-plane-
      neighbor graph from dataset.py. A satellite's predicted residual is
      also compared against what its orbital neighbors are doing. If your
      whole shell is drifting the same way, that's probably solar-cycle
      drag, not an anomaly. If ONE satellite diverges from a shell that's
      otherwise stable, that's the interesting case. This is the "graph"
      idea nobody else in the room will have: anomaly-by-deviation-from-
      peers instead of anomaly-by-deviation-from-history.

  3. "How do you know when to trust the model?" -> MC-Dropout uncertainty:
      dropout stays ON at inference time and we run N stochastic forward
      passes. The SPREAD across those passes is the model's own confidence.
      A satellite whose predicted uncertainty suddenly balloons is one the
      model itself doesn't understand anymore -- which is, definitionally,
      an anomaly. This turns "the model is confused" from a bug into the
      actual detection mechanism, and it's a genuinely nice line to say out
      loud in a presentation.

Final anomaly score = weighted combination of:
    (a) residual forecast error         (self-consistency)
    (b) deviation from GNN-pooled peers (peer-consistency)
    (c) MC-dropout predictive variance  (model self-doubt)

All three are physically grounded, all three are diagnosable, and the sum
is far more robust than any one alone -- three uncorrelated ways for an
anomaly to get caught beats one clever way.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualEncoder(nn.Module):
    """Small Transformer encoder over a satellite's own residual history.
    Input:  (B, T, 6) physics-residual sequence
    Output: (B, d_model) latent summary + (B, 6) 1-step-ahead residual forecast
    """

    def __init__(self, in_dim=6, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Linear(d_model, in_dim)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :T, :]
        h = self.encoder(h)
        h = self.dropout(h)
        summary = h[:, -1, :]                      # latent state at last timestep
        forecast = self.forecast_head(summary)      # predicted next residual
        return summary, forecast


class NeighborGCN(nn.Module):
    """
    One graph-convolution hop over the orbital-neighbor adjacency matrix.
    Implemented by hand (plain matmuls) on purpose -- no torch_geometric
    dependency, so this runs anywhere torch runs (including a bare Colab),
    and it makes the mechanism transparent for your slides:

        H' = sigma( D^-1 A H W )

    i.e. every satellite's new representation is a learned mix of its own
    latent state and the AVERAGE latent state of its orbital-plane peers.
    """

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        # h:   (S, d)   per-satellite latent summaries at this snapshot
        # adj: (S, S)   symmetric orbital-neighbor adjacency (0/1)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg                         # row-normalize -> mean over neighbors
        agg = adj_norm @ h                           # (S, d) neighbor-averaged features
        out = F.gelu(self.lin(agg))
        out = self.dropout(out)
        return out + h                               # residual connection


class OrbitGNN(nn.Module):
    """
    Full pipeline for one time-snapshot across a constellation:

        residual sequences (S, T, 6)
                |
        ResidualEncoder (shared weights across satellites)
                |
        latent summaries (S, d) ------ orbital-neighbor adjacency (S, S)
                |                              |
                +---------- NeighborGCN -------+
                |
        peer-aware latent (S, d)
                |
        forecast head -> predicted next residual per satellite (S, 6)

    forward() returns both the "self" forecast (from the encoder alone) and
    the "peer-aware" forecast (after graph mixing). The gap between them,
    plus MC-dropout variance, is what train.py turns into the anomaly score.
    """

    def __init__(self, in_dim=6, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = ResidualEncoder(in_dim, d_model, n_heads, n_layers, dropout)
        self.gcn = NeighborGCN(d_model, dropout)
        self.peer_forecast_head = nn.Linear(d_model, in_dim)

    def forward(self, residual_seqs, adj):
        # residual_seqs: (S, T, 6), adj: (S, S)
        summary, self_forecast = self.encoder(residual_seqs)   # (S,d), (S,6)
        peer_latent = self.gcn(summary, adj)                   # (S,d)
        peer_forecast = self.peer_forecast_head(peer_latent)   # (S,6)
        return self_forecast, peer_forecast, summary

    @torch.no_grad()
    def mc_dropout_forecast(self, residual_seqs, adj, n_samples=20):
        """
        Run N stochastic forward passes with dropout FORCED ON (even though
        we're in eval/no_grad mode) to get a predictive distribution instead
        of a point estimate. Returns mean + std of the peer-aware forecast.
        std is our per-satellite, per-timestep "model confidence" signal.
        """
        was_training = self.training
        self.train()  # keep dropout active
        preds = []
        for _ in range(n_samples):
            _, peer_forecast, _ = self.forward(residual_seqs, adj)
            preds.append(peer_forecast.unsqueeze(0))
        preds = torch.cat(preds, dim=0)  # (n_samples, S, 6)
        if not was_training:
            self.eval()
        return preds.mean(0), preds.std(0)


def anomaly_score(observed_residual, self_forecast, peer_forecast, mc_std,
                   w_self=1.0, w_peer=1.0, w_uncertainty=1.0):
    """
    Combine the three signals described at the top of this file into one
    scalar per satellite. Each term is normalized (mean/std across the
    batch) before combining so no single term dominates just because of
    units (e.g. semi-major axis residual in km vs angles in radians).
    """
    def z(x):
        return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)

    self_err = z((observed_residual - self_forecast).pow(2).sum(-1, keepdim=True))
    peer_err = z((observed_residual - peer_forecast).pow(2).sum(-1, keepdim=True))
    uncertainty = z(mc_std.pow(2).sum(-1, keepdim=True))

    score = w_self * self_err + w_peer * peer_err + w_uncertainty * uncertainty
    return score.squeeze(-1)
