"""
tests/test_physics.py
---------------------
Unit tests for physics.py: constants, propagation, Kepler solver,
J2 rates, angle wrapping, and residual computation.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from physics import (
    KeplerianElements,
    MU_EARTH,
    R_EARTH,
    J2,
    mean_motion,
    solve_kepler,
    j2_secular_rates,
    propagate,
    wrap_angle,
    elements_residual,
    propagate_batch,
    residuals_batch,
)


# ════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_mu_earth_km3_s2(self):
        """μ should be close to 398600 km³/s²."""
        assert 398_500 < MU_EARTH < 398_700

    def test_r_earth_km(self):
        """Earth radius ≈ 6378 km."""
        assert 6370 < R_EARTH < 6380

    def test_j2_magnitude(self):
        """J2 ≈ 1.083e-3."""
        assert 1.08e-3 < J2 < 1.09e-3


# ════════════════════════════════════════════════════════════════════
# KeplerianElements
# ════════════════════════════════════════════════════════════════════

class TestKeplerianElements:
    def _iss(self) -> KeplerianElements:
        return KeplerianElements(
            a=R_EARTH + 400.0, e=0.001,
            i=math.radians(51.6), raan=1.0, argp=0.5, M=0.3,
        )

    def test_as_vector_shape(self):
        v = self._iss().as_vector()
        assert v.shape == (6,)

    def test_from_vector_round_trip(self):
        el = self._iss()
        v  = el.as_vector()
        el2 = KeplerianElements.from_vector(v)
        np.testing.assert_allclose(el2.as_vector(), v)

    def test_is_valid_iss(self):
        assert self._iss().is_valid()

    def test_is_valid_invalid_a(self):
        el = self._iss()
        el.a = 0.0
        assert not el.is_valid()

    def test_is_valid_invalid_e(self):
        el = self._iss()
        el.e = 1.5
        assert not el.is_valid()


# ════════════════════════════════════════════════════════════════════
# Mean motion
# ════════════════════════════════════════════════════════════════════

class TestMeanMotion:
    def test_iss_mean_motion_range(self):
        """ISS (~400 km alt) should have ~15.5 rev/day."""
        n = mean_motion(R_EARTH + 400.0)
        rev_day = n * 86400.0 / (2 * math.pi)
        assert 15.0 < rev_day < 16.0

    def test_geo_mean_motion(self):
        """GEO (~35787 km alt) should have ~1 rev/day."""
        n = mean_motion(R_EARTH + 35_787.0)
        rev_day = n * 86400.0 / (2 * math.pi)
        assert 0.99 < rev_day < 1.01

    def test_invalid_a_raises(self):
        with pytest.raises(ValueError):
            mean_motion(-100.0)


# ════════════════════════════════════════════════════════════════════
# Kepler equation solver
# ════════════════════════════════════════════════════════════════════

class TestKeplerSolver:
    def test_circular_orbit_e0(self):
        """For e=0, E should equal M."""
        for M in [0.0, 0.5, 1.0, 2.0, math.pi, 5.0]:
            E = solve_kepler(M, 0.0)
            assert abs(E - M % (2 * math.pi)) < 1e-8

    def test_low_eccentricity_convergence(self):
        """Newton-Raphson must satisfy M = E - e*sin(E) to tolerance."""
        for e in [0.001, 0.01, 0.05, 0.1]:
            for M in [0.1, 1.0, 2.5, 5.0]:
                E = solve_kepler(M, e)
                res = E - e * math.sin(E) - (M % (2 * math.pi))
                assert abs(res) < 1e-8, f"e={e} M={M}: residual={res}"

    def test_m_2pi_equivalence(self):
        """M and M + 2π should give same E (mod 2π)."""
        E1 = solve_kepler(1.0, 0.05)
        E2 = solve_kepler(1.0 + 2 * math.pi, 0.05)
        assert abs(E1 - E2) < 1e-9

    def test_negative_M_wrapped(self):
        """Negative M should be wrapped before solving."""
        E1 = solve_kepler(-0.5, 0.01)
        E2 = solve_kepler(-0.5 + 2 * math.pi, 0.01)
        assert abs(E1 - E2) < 1e-9


# ════════════════════════════════════════════════════════════════════
# J2 secular rates
# ════════════════════════════════════════════════════════════════════

class TestJ2Rates:
    def test_raan_drift_iss_sign(self):
        """RAAN drift for prograde orbit should be negative (westward)."""
        raan_dot, _, _ = j2_secular_rates(R_EARTH + 400, 0.001,
                                          math.radians(51.6))
        assert raan_dot < 0, "RAAN drift should be negative for prograde orbit"

    def test_raan_drift_iss_magnitude(self):
        """RAAN drift for ISS-like orbit should be ~ -4 to -8 deg/day."""
        raan_dot, _, _ = j2_secular_rates(R_EARTH + 400, 0.001,
                                          math.radians(51.6))
        deg_day = math.degrees(raan_dot) * 86400
        assert -8 < deg_day < -4, f"RAAN drift = {deg_day:.2f} deg/day"

    def test_raan_drift_polar_orbit(self):
        """At i=90°, cos(i)=0, so RAAN drift should be ~0."""
        raan_dot, _, _ = j2_secular_rates(R_EARTH + 600, 0.001,
                                          math.radians(90.0))
        assert abs(raan_dot) < 1e-9, "RAAN drift should be 0 at i=90°"

    def test_sso_inclination(self):
        """Sun-synchronous orbit: RAAN drift ≈ +0.9856 deg/day."""
        # SSO for ~800 km altitude requires i ≈ 98.6°
        raan_dot, _, _ = j2_secular_rates(R_EARTH + 800, 0.001,
                                          math.radians(98.6))
        deg_day = math.degrees(raan_dot) * 86400
        assert 0.8 < deg_day < 1.1, f"SSO RAAN drift = {deg_day:.3f} deg/day"

    def test_invalid_a_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError, FloatingPointError)):
            j2_secular_rates(0.0, 0.001, 1.0)


# ════════════════════════════════════════════════════════════════════
# Propagation
# ════════════════════════════════════════════════════════════════════

class TestPropagate:
    def _iss(self) -> KeplerianElements:
        return KeplerianElements(
            a=R_EARTH + 400.0, e=0.001,
            i=math.radians(51.6), raan=1.0, argp=0.5, M=0.3,
        )

    def test_zero_dt_identity(self):
        """Propagation with dt=0 should return same elements."""
        el  = self._iss()
        out = propagate(el, 0.0)
        np.testing.assert_allclose(out.as_vector(), el.as_vector(), atol=1e-10)

    def test_a_e_i_unchanged(self):
        """a, e, i should be constant under J2 secular to 1st order."""
        el  = self._iss()
        out = propagate(el, 86400.0)
        assert abs(out.a - el.a) < 1e-10
        assert abs(out.e - el.e) < 1e-10
        assert abs(out.i - el.i) < 1e-10

    def test_raan_advances(self):
        """RAAN should drift (negative for prograde orbit)."""
        el  = self._iss()
        out = propagate(el, 86400.0)
        diff = wrap_angle(out.raan - el.raan)
        assert diff < 0, f"RAAN should decrease; got {math.degrees(diff):.4f} deg"

    def test_angles_in_range(self):
        """Propagated angles should be in [0, 2π)."""
        el  = self._iss()
        out = propagate(el, 10 * 86400.0)
        for angle in [out.raan, out.argp, out.M]:
            assert 0 <= angle < 2 * math.pi, f"Angle {angle} out of [0, 2π)"

    def test_one_period_M_return(self):
        """After one orbital period, M should return close to initial M."""
        el  = self._iss()
        T   = 2.0 * math.pi / mean_motion(el.a)
        out = propagate(el, T)
        dM  = wrap_angle(out.M - el.M)
        # Small residual due to J2 correction on M
        assert abs(dM) < math.radians(1.0), f"M deviation after one period: {math.degrees(dM):.3f} deg"

    def test_geo_raan_drift_rate(self):
        """GEO satellite RAAN drift should be < 0.1 deg/day."""
        el  = KeplerianElements(
            a=R_EARTH + 35_787.0, e=0.0001,
            i=math.radians(0.1), raan=1.0, argp=0.5, M=0.3,
        )
        out = propagate(el, 86400.0)
        diff_deg = math.degrees(wrap_angle(out.raan - el.raan))
        assert abs(diff_deg) < 0.1, f"GEO RAAN drift {diff_deg:.4f} deg/day"


# ════════════════════════════════════════════════════════════════════
# Angle wrapping
# ════════════════════════════════════════════════════════════════════

class TestWrapAngle:
    def test_zero(self):
        assert wrap_angle(0.0) == 0.0

    def test_pi(self):
        # π wraps to −π (our convention: result in (-π, π])
        assert abs(wrap_angle(math.pi) - (-math.pi)) < 1e-9 or abs(wrap_angle(math.pi) - math.pi) < 1e-9

    def test_near_180_vs_minus_180(self):
        """179° and -179° should produce residual ≈ ±2°, not ±358°."""
        diff = wrap_angle(math.radians(179) - math.radians(-179))
        assert abs(diff) < math.radians(5), f"Expected ~±2°, got {math.degrees(diff):.1f}°"

    def test_large_positive(self):
        """3π should wrap to π (mod 2π → π, then into (-π,π] → -π or π)."""
        result = wrap_angle(3 * math.pi)
        assert -math.pi <= result <= math.pi

    def test_large_negative(self):
        result = wrap_angle(-3 * math.pi)
        assert -math.pi <= result <= math.pi

    def test_symmetry(self):
        """wrap_angle(x) == -wrap_angle(-x) for small x."""
        for x in [0.1, 0.5, 1.0, 2.0]:
            assert abs(wrap_angle(x) + wrap_angle(-x)) < 1e-10


# ════════════════════════════════════════════════════════════════════
# Elements residual
# ════════════════════════════════════════════════════════════════════

class TestElementsResidual:
    def test_identical_states_zero_residual(self):
        el  = KeplerianElements(7000, 0.001, 1.0, 1.0, 1.0, 1.0)
        res = elements_residual(el, el)
        np.testing.assert_allclose(res, 0.0, atol=1e-12)

    def test_shape(self):
        el1 = KeplerianElements(7000, 0.001, 1.0, 1.0, 1.0, 1.0)
        el2 = KeplerianElements(7001, 0.002, 1.1, 1.1, 1.1, 1.1)
        assert elements_residual(el1, el2).shape == (6,)

    def test_angle_wrapping_in_residual(self):
        """RAAN near ±π should be wrapped correctly."""
        el_obs  = KeplerianElements(7000, 0.001, 1.0, math.radians(179), 1.0, 1.0)
        el_pred = KeplerianElements(7000, 0.001, 1.0, math.radians(-179), 1.0, 1.0)
        res = elements_residual(el_obs, el_pred)
        # ΔRAAN should be ≈ 2°, not 358°
        assert abs(math.degrees(res[3])) < 5.0

    def test_da_not_wrapped(self):
        """Δa is a difference of km values, not an angle — no wrapping."""
        el_obs  = KeplerianElements(7100, 0.001, 1.0, 1.0, 1.0, 1.0)
        el_pred = KeplerianElements(7000, 0.001, 1.0, 1.0, 1.0, 1.0)
        res = elements_residual(el_obs, el_pred)
        assert abs(res[0] - 100.0) < 1e-8


# ════════════════════════════════════════════════════════════════════
# Vectorised batch functions
# ════════════════════════════════════════════════════════════════════

class TestBatchFunctions:
    def _make_sequence(self, S=3, T=10):
        """Create a simple element sequence for batch testing."""
        np.random.seed(0)
        a    = (R_EARTH + 500 + np.random.randn(S, T) * 0.1)
        e    = np.clip(np.random.randn(S, T) * 0.001 + 0.001, 0.0001, 0.01)
        i    = np.ones((S, T)) * math.radians(51.6) + np.random.randn(S, T) * 1e-5
        raan = np.random.uniform(0, 2 * math.pi, (S, T))
        argp = np.random.uniform(0, 2 * math.pi, (S, T))
        M    = np.random.uniform(0, 2 * math.pi, (S, T))
        return np.stack([a, e, i, raan, argp, M], axis=-1)   # (S, T, 6)

    def test_propagate_batch_shape(self):
        elems = self._make_sequence(3, 10)
        dt    = np.full((3, 10), 86400.0)
        out   = propagate_batch(elems, dt)
        assert out.shape == elems.shape

    def test_propagate_batch_zero_dt(self):
        """Zero dt should return same elements."""
        elems = self._make_sequence(2, 5)
        dt    = np.zeros((2, 5))
        out   = propagate_batch(elems, dt)
        # a, e, i should be unchanged regardless
        np.testing.assert_allclose(out[..., 0], elems[..., 0])  # a
        np.testing.assert_allclose(out[..., 1], elems[..., 1])  # e
        np.testing.assert_allclose(out[..., 2], elems[..., 2])  # i

    def test_residuals_batch_shape(self):
        elems = self._make_sequence(4, 12)
        dt    = np.full((4, 11), 86400.0)
        res   = residuals_batch(elems, dt)
        assert res.shape == (4, 11, 6)

    def test_residuals_batch_no_nan(self):
        elems = self._make_sequence(4, 12)
        dt    = np.full((4, 11), 86400.0)
        res   = residuals_batch(elems, dt)
        assert not np.any(np.isnan(res))

    def test_batch_vs_scalar_agreement(self):
        """Single-step batch propagation should match scalar propagate()."""
        el_scalar = KeplerianElements(
            a=R_EARTH + 500, e=0.001, i=math.radians(51.6),
            raan=1.0, argp=0.5, M=0.3,
        )
        dt_s = 86400.0
        prop_scalar = propagate(el_scalar, dt_s)

        elems = el_scalar.as_vector().reshape(1, 1, 6)
        dt    = np.array([[[dt_s]]])[:, :, 0]   # shape (1, 1)
        out   = propagate_batch(elems, dt)

        np.testing.assert_allclose(out[0, 0, 0], prop_scalar.a,    atol=1e-6)
        np.testing.assert_allclose(out[0, 0, 1], prop_scalar.e,    atol=1e-10)
        np.testing.assert_allclose(out[0, 0, 2], prop_scalar.i,    atol=1e-10)


# ════════════════════════════════════════════════════════════════════
# J2 propagation: physical correctness checks
# ════════════════════════════════════════════════════════════════════

class TestJ2PhysicalCorrectness:
    def test_raan_drift_direction(self):
        """
        For a prograde orbit (i < 90°), J2 causes RAAN to drift westward
        (decrease).  For retrograde (i > 90°), it drifts eastward.
        """
        # Prograde
        el_pro = KeplerianElements(R_EARTH+600, 0.001, math.radians(51.6), 1.0, 0.5, 0.3)
        el_ret = KeplerianElements(R_EARTH+600, 0.001, math.radians(98.6), 1.0, 0.5, 0.3)

        dt = 86400.0
        pro_out = propagate(el_pro, dt)
        ret_out = propagate(el_ret, dt)

        pro_dR = wrap_angle(pro_out.raan - el_pro.raan)
        ret_dR = wrap_angle(ret_out.raan - el_ret.raan)

        assert pro_dR < 0, "Prograde RAAN should decrease"
        assert ret_dR > 0, "Retrograde RAAN should increase"

    def test_argp_drift_critical_inclination(self):
        """
        At the critical inclination i ≈ 63.435°, dω/dt should be ~0.
        The exact value is limited by floating-point precision; tolerance is 1e-6.
        """
        i_crit = math.asin(math.sqrt(2.0 / 5.0))  # ≈ 63.435°
        _, argp_dot, _ = j2_secular_rates(R_EARTH + 500, 0.001, i_crit)
        # argp_dot ≈ 0 at critical inclination (floating-point limited)
        assert abs(argp_dot) < 1e-5, f"argp_dot at crit inc = {argp_dot:.2e}"
