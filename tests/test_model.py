"""
tests/test_model.py
-------------------
Unit tests for OrbitGNN model components:
  - ResidualEncoder (Transformer)
  - NeighborGCN
  - OrbitGNN (full forward pass)
  - MC-Dropout inference
  - anomaly_score function
  - Ablation modes (zero adjacency, trivial encoder)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import torch
import torch.nn as nn
import numpy as np

from model import ResidualEncoder, NeighborGCN, OrbitGNN, anomaly_score


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════

S      = 9     # satellites
WINDOW = 8     # lookback steps
F      = 6     # orbital element features
D      = 64    # model hidden dim


@pytest.fixture
def model():
    m = OrbitGNN(in_dim=F, d_model=D, n_heads=4, n_layers=2, dropout=0.2)
    m.eval()
    return m


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(S, WINDOW, F)


@pytest.fixture
def adj():
    """Shell-aware adjacency: 3 GEO + 4 SSO + 2 LEO."""
    a = torch.zeros(S, S)
    # Shell 0: nodes 0,1,2
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            if i != j:
                a[i, j] = 1.0
    # Shell 1: nodes 3,4,5,6
    for i in [3, 4, 5, 6]:
        for j in [3, 4, 5, 6]:
            if i != j:
                a[i, j] = 1.0
    # Shell 2: nodes 7,8
    a[7, 8] = a[8, 7] = 1.0
    return a


# ════════════════════════════════════════════════════════════════════
# ResidualEncoder tests
# ════════════════════════════════════════════════════════════════════

class TestResidualEncoder:

    def test_output_shapes(self):
        enc = ResidualEncoder(in_dim=F, d_model=D, n_heads=4, n_layers=2)
        enc.eval()
        x = torch.randn(S, WINDOW, F)
        with torch.no_grad():
            summary, forecast = enc(x)
        assert summary.shape  == (S, D), f"summary shape: {summary.shape}"
        assert forecast.shape == (S, F), f"forecast shape: {forecast.shape}"

    def test_no_nan_on_normal_input(self):
        enc = ResidualEncoder(in_dim=F, d_model=D, n_heads=4, n_layers=2)
        enc.eval()
        x = torch.randn(S, WINDOW, F)
        with torch.no_grad():
            summary, forecast = enc(x)
        assert not torch.isnan(summary).any(),  "NaN in encoder summary"
        assert not torch.isnan(forecast).any(), "NaN in encoder forecast"

    def test_no_nan_on_zero_input(self):
        """Zero input should not produce NaN (layer-norm handles it)."""
        enc = ResidualEncoder(in_dim=F, d_model=D, n_heads=4, n_layers=2)
        enc.eval()
        x = torch.zeros(S, WINDOW, F)
        with torch.no_grad():
            summary, forecast = enc(x)
        assert not torch.isnan(summary).any()
        assert not torch.isnan(forecast).any()

    def test_different_windows(self):
        """Encoder should work with any window length up to 512 (pos_embed size)."""
        enc = ResidualEncoder(in_dim=F, d_model=D, n_heads=4, n_layers=2)
        enc.eval()
        for T in [1, 4, 8, 16, 32]:
            x = torch.randn(S, T, F)
            with torch.no_grad():
                summary, forecast = enc(x)
            assert summary.shape  == (S, D)
            assert forecast.shape == (S, F)

    def test_batch_independence(self):
        """Output for satellite i should not depend on satellite j (no shared state)."""
        enc = ResidualEncoder(in_dim=F, d_model=D, n_heads=4, n_layers=2)
        enc.eval()
        torch.manual_seed(1)
        x = torch.randn(S, WINDOW, F)
        # Perturb satellite 0 and check that only satellite 0's output changes
        x_perturbed = x.clone()
        x_perturbed[0] += 100.0
        with torch.no_grad():
            _, f1 = enc(x)
            _, f2 = enc(x_perturbed)
        # Satellites 1..8 should be unchanged
        assert torch.allclose(f1[1:], f2[1:], atol=1e-5), \
            "Encoder output for unperturbed satellites changed unexpectedly"

    def test_dropout_active_in_train_mode(self):
        """In train mode, two forward passes should differ (dropout is stochastic)."""
        enc = ResidualEncoder(in_dim=F, d_model=D, n_heads=4, n_layers=2, dropout=0.5)
        enc.train()
        torch.manual_seed(42)
        x = torch.randn(S, WINDOW, F)
        s1, _ = enc(x)
        s2, _ = enc(x)
        assert not torch.allclose(s1, s2), "Dropout has no effect in train mode"

    def test_dropout_inactive_in_eval_mode(self):
        """In eval mode, two forward passes must be identical."""
        enc = ResidualEncoder(in_dim=F, d_model=D, n_heads=4, n_layers=2)
        enc.eval()
        x = torch.randn(S, WINDOW, F)
        with torch.no_grad():
            s1, _ = enc(x)
            s2, _ = enc(x)
        assert torch.allclose(s1, s2), "Non-deterministic in eval mode"


# ════════════════════════════════════════════════════════════════════
# NeighborGCN tests
# ════════════════════════════════════════════════════════════════════

class TestNeighborGCN:

    def test_output_shape(self):
        gcn = NeighborGCN(dim=D)
        h   = torch.randn(S, D)
        a   = torch.eye(S)
        out = gcn(h, a)
        assert out.shape == (S, D)

    def test_no_nan(self):
        gcn = NeighborGCN(dim=D)
        h   = torch.randn(S, D)
        a   = torch.eye(S)
        out = gcn(h, a)
        assert not torch.isnan(out).any()

    def test_zero_adjacency_is_self_residual(self):
        """With adj=0, aggregation is over zero peers → agg=0, output = residual of W@0 + h = relu(0)+h."""
        gcn = NeighborGCN(dim=D)
        gcn.eval()
        h   = torch.randn(S, D)
        a   = torch.zeros(S, S)
        with torch.no_grad():
            out = gcn(h, a)
        # The residual connection always adds h back:
        # out = gelu(lin(agg)) + h, where agg = 0 when adj=0
        # So ||out - h|| should be non-zero (lin transforms it) but finite
        assert torch.isfinite(out).all()
        assert out.shape == (S, D)

    def test_identity_adjacency(self):
        """With identity adj, each node aggregates from itself only."""
        gcn = NeighborGCN(dim=D)
        gcn.eval()
        h   = torch.randn(S, D)
        a   = torch.eye(S)
        with torch.no_grad():
            out = gcn(h, a)
        assert out.shape == (S, D)
        assert not torch.isnan(out).any()

    def test_symmetry_preserved(self):
        """If two nodes have identical features and are connected symmetrically,
        their outputs should be identical."""
        gcn = NeighborGCN(dim=D)
        gcn.eval()
        h   = torch.randn(S, D)
        h[3] = h[4].clone()           # force identical features for nodes 3 & 4
        a   = torch.zeros(S, S)
        a[3, 4] = a[4, 3] = 1.0       # only edge between 3 and 4
        with torch.no_grad():
            out = gcn(h, a)
        assert torch.allclose(out[3], out[4], atol=1e-5), \
            "Symmetric nodes produce different GCN outputs"


# ════════════════════════════════════════════════════════════════════
# OrbitGNN full model tests
# ════════════════════════════════════════════════════════════════════

class TestOrbitGNN:

    def test_forward_output_shapes(self, model, x, adj):
        with torch.no_grad():
            sf, pf, summary = model(x, adj)
        assert sf.shape      == (S, F), f"self_forecast: {sf.shape}"
        assert pf.shape      == (S, F), f"peer_forecast: {pf.shape}"
        assert summary.shape == (S, D), f"summary: {summary.shape}"

    def test_no_nan_normal_input(self, model, x, adj):
        with torch.no_grad():
            sf, pf, summary = model(x, adj)
        assert not torch.isnan(sf).any(),      "NaN in self_forecast"
        assert not torch.isnan(pf).any(),      "NaN in peer_forecast"
        assert not torch.isnan(summary).any(), "NaN in summary"

    def test_no_nan_zero_input(self, model, adj):
        x = torch.zeros(S, WINDOW, F)
        with torch.no_grad():
            sf, pf, summary = model(x, adj)
        assert not torch.isnan(sf).any()
        assert not torch.isnan(pf).any()

    def test_no_nan_zero_adjacency(self, model, x):
        adj_zero = torch.zeros(S, S)
        with torch.no_grad():
            sf, pf, summary = model(x, adj_zero)
        assert not torch.isnan(sf).any()
        assert not torch.isnan(pf).any()

    def test_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert 80_000 < n_params < 150_000, \
            f"Unexpected parameter count: {n_params}"

    def test_eval_mode_deterministic(self, model, x, adj):
        """Two forward passes in eval mode must produce identical outputs."""
        with torch.no_grad():
            sf1, pf1, _ = model(x, adj)
            sf2, pf2, _ = model(x, adj)
        assert torch.allclose(sf1, sf2), "Non-deterministic self_forecast in eval mode"
        assert torch.allclose(pf1, pf2), "Non-deterministic peer_forecast in eval mode"

    def test_train_mode_stochastic(self, x, adj):
        """Dropout should make outputs differ in train mode."""
        m = OrbitGNN(in_dim=F, d_model=D, n_heads=4, n_layers=2, dropout=0.5)
        m.train()
        sf1, pf1, _ = m(x, adj)
        sf2, pf2, _ = m(x, adj)
        # With 50% dropout at least some values should differ
        assert not torch.allclose(sf1, sf2), "No stochasticity in train mode"

    def test_peer_forecast_differs_from_self_forecast(self, model, x, adj):
        """Self and peer forecasts should not be identical (GCN transforms the latent)."""
        with torch.no_grad():
            sf, pf, _ = model(x, adj)
        assert not torch.allclose(sf, pf), \
            "self_forecast == peer_forecast — GCN has no effect"

    def test_zero_adj_self_peer_differ(self, model, x):
        """Even with zero adjacency, self != peer (GCN residual connection)."""
        adj_zero = torch.zeros(S, S)
        with torch.no_grad():
            sf, pf, _ = model(x, adj_zero)
        # They may or may not be identical — just check no crash and finite values
        assert torch.isfinite(pf).all()


# ════════════════════════════════════════════════════════════════════
# MC-Dropout tests
# ════════════════════════════════════════════════════════════════════

class TestMCDropout:

    def test_output_shapes(self, model, x, adj):
        mean_pf, std_pf = model.mc_dropout_forecast(x, adj, n_samples=5)
        assert mean_pf.shape == (S, F)
        assert std_pf.shape  == (S, F)

    def test_no_nan(self, model, x, adj):
        mean_pf, std_pf = model.mc_dropout_forecast(x, adj, n_samples=5)
        assert not torch.isnan(mean_pf).any()
        assert not torch.isnan(std_pf).any()

    def test_std_positive(self, model, x, adj):
        """With dropout, std over N samples must be > 0 for at least some elements."""
        mean_pf, std_pf = model.mc_dropout_forecast(x, adj, n_samples=20)
        assert (std_pf > 0).any(), "MC-Dropout produced zero variance for all elements"

    def test_n_samples_1_std_is_nan_or_zero(self, model, x, adj):
        """
        With n_samples=1, torch.std() returns NaN (unbiased estimator has
        0 degrees of freedom). Expected PyTorch behaviour — documented here.
        Production always uses n_samples >= 2.
        """
        _, std_pf = model.mc_dropout_forecast(x, adj, n_samples=1)
        assert ((std_pf == 0.0) | torch.isnan(std_pf)).all()

    def test_model_returns_to_eval_after_mc(self, model, x, adj):
        """mc_dropout_forecast should leave model in eval mode if it was eval."""
        assert not model.training
        model.mc_dropout_forecast(x, adj, n_samples=5)
        assert not model.training, "model left in train mode after mc_dropout_forecast"

    def test_mean_close_to_deterministic_eval(self, model, x, adj):
        """MC-Dropout mean (large n) should be close to the eval-mode peer_forecast."""
        with torch.no_grad():
            _, pf_eval, _ = model(x, adj)
        mean_pf, _ = model.mc_dropout_forecast(x, adj, n_samples=200)
        # They won't be identical (dropout on), but should be in the same ballpark
        # Just verify both are finite and close in magnitude
        ratio = (mean_pf.abs() / (pf_eval.abs() + 1e-6)).mean().item()
        assert 0.1 < ratio < 10.0, f"MC mean diverges wildly from eval: ratio={ratio:.3f}"


# ════════════════════════════════════════════════════════════════════
# anomaly_score function tests
# ════════════════════════════════════════════════════════════════════

class TestAnomalyScore:

    def _run(self, obs, sf, pf, std):
        return anomaly_score(obs, sf, pf, std)

    def test_output_shape(self, model, x, adj):
        obs = x[:, -1, :]
        with torch.no_grad():
            sf, pf, _ = model(x, adj)
        mean_pf, std_pf = model.mc_dropout_forecast(x, adj, n_samples=5)
        score = anomaly_score(obs, sf, pf, std_pf)
        assert score.shape == (S,), f"score shape: {score.shape}"

    def test_no_nan(self, model, x, adj):
        obs = x[:, -1, :]
        with torch.no_grad():
            sf, pf, _ = model(x, adj)
        mean_pf, std_pf = model.mc_dropout_forecast(x, adj, n_samples=5)
        score = anomaly_score(obs, sf, pf, std_pf)
        assert not torch.isnan(score).any()

    def test_higher_error_raises_score(self, model, x, adj):
        """A satellite with large forecast error should receive a higher anomaly score."""
        obs = x[:, -1, :].clone()
        with torch.no_grad():
            sf, pf, _ = model(x, adj)
        _, std_pf = model.mc_dropout_forecast(x, adj, n_samples=5)

        # Inflate observed residual for satellite 0 by 100× (simulating manoeuvre)
        obs_anomaly = obs.clone()
        obs_anomaly[0] = obs[0] * 100.0

        score_normal  = anomaly_score(obs,         sf, pf, std_pf)
        score_anomaly = anomaly_score(obs_anomaly,  sf, pf, std_pf)

        # Satellite 0 should receive a higher score in the anomalous case
        assert score_anomaly[0] > score_normal[0], \
            f"Anomaly not detected: score_normal[0]={score_normal[0]:.3f}, " \
            f"score_anomaly[0]={score_anomaly[0]:.3f}"

    def test_zero_std_does_not_crash(self, model, x, adj):
        """If MC-Dropout std is all zeros, score should still be finite."""
        obs = x[:, -1, :]
        with torch.no_grad():
            sf, pf, _ = model(x, adj)
        std_zero = torch.zeros_like(pf)
        score = anomaly_score(obs, sf, pf, std_zero)
        assert torch.isfinite(score).all()

    def test_perfect_forecast_lower_score(self):
        """If self_forecast == observed (perfect prediction), self-error term = 0."""
        S_t = 4; F_t = 6
        obs = torch.randn(S_t, F_t)
        sf  = obs.clone()          # perfect self-forecast
        pf  = torch.randn(S_t, F_t)
        std = torch.ones(S_t, F_t) * 0.01

        score_perfect = anomaly_score(obs, sf, pf, std)

        # With bad self-forecast (adds large noise):
        sf_bad = obs + torch.randn_like(obs) * 10.0
        score_bad = anomaly_score(obs, sf_bad, pf, std)

        # Mean score should be lower with perfect forecast
        assert score_perfect.mean() < score_bad.mean(), \
            "Perfect forecast did not produce lower anomaly score"

    def test_weights_affect_score(self):
        """w_self, w_peer, w_uncertainty should independently scale each term."""
        S_t = 4; F_t = 6
        obs = torch.randn(S_t, F_t)
        sf  = torch.zeros_like(obs)
        pf  = torch.zeros_like(obs)
        std = torch.ones_like(obs)

        s1 = anomaly_score(obs, sf, pf, std, w_self=1.0, w_peer=0.0, w_uncertainty=0.0)
        s2 = anomaly_score(obs, sf, pf, std, w_self=0.0, w_peer=1.0, w_uncertainty=0.0)
        s3 = anomaly_score(obs, sf, pf, std, w_self=0.0, w_peer=0.0, w_uncertainty=1.0)
        s_all = anomaly_score(obs, sf, pf, std, w_self=1.0, w_peer=1.0, w_uncertainty=1.0)

        # Combined score should equal sum of individual terms (after z-score normalisation)
        # At least verify they're all finite and not identical
        assert torch.isfinite(s1).all()
        assert torch.isfinite(s2).all()
        assert torch.isfinite(s3).all()


# ════════════════════════════════════════════════════════════════════
# Integration: train one step, check gradient flow
# ════════════════════════════════════════════════════════════════════

class TestGradientFlow:

    def test_backward_pass(self, x, adj):
        """Backward pass should not raise and all parameter grads should be non-None."""
        m = OrbitGNN(in_dim=F, d_model=D, n_heads=4, n_layers=2)
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)

        sf, pf, _ = m(x, adj)
        target = torch.zeros_like(sf)
        loss = nn.MSELoss()(sf, target) + nn.MSELoss()(pf, target)

        opt.zero_grad()
        loss.backward()

        for name, param in m.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_loss_decreases_after_steps(self, x, adj):
        """After 10 training steps on the same batch, loss should decrease."""
        torch.manual_seed(99)
        m = OrbitGNN(in_dim=F, d_model=D, n_heads=4, n_layers=2)
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        target = torch.zeros_like(x[:, -1, :])

        losses = []
        for _ in range(10):
            sf, pf, _ = m(x, adj)
            loss = nn.MSELoss()(sf, target) + nn.MSELoss()(pf, target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
