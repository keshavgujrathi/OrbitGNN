"""
physics.py
----------
Simplified two-body + J2-secular orbital mechanics for OrbitGNN.

PURPOSE
-------
This module is NOT a precision orbit-determination (OD) propagator.
It is a *residual generator*: given a satellite's observed Keplerian
state at time t, it predicts where the satellite should be at time t+dt
under the simplest physically meaningful model (two-body + J2 secular).
The difference between prediction and next real observation is the
"physics residual" that OrbitGNN learns to characterise for anomaly detection.

DESIGN CONSTRAINTS
------------------
* Public interface (KeplerianElements, propagate, elements_residual,
  mean_motion) is preserved for backward-compatibility with dataset.py,
  model.py, and train.py.
* No external orbital-mechanics libraries (sgp4, poliastro, etc.)
  are used, keeping the project self-contained.
* Internal units: km and seconds throughout unless explicitly noted.
  Angles: radians throughout.  Degrees are only accepted at TLE-parsing
  time (in dataset.py) and immediately converted.

ORBITAL MODEL
-------------
Two-body Kepler motion (central gravitational force only) plus the
first-order secular perturbations from Earth's J2 oblateness:

  d(RAAN)/dt  = -3/2 · n · J2 · (R_E/p)² · cos(i)
  d(ω)/dt     = -3/2 · n · J2 · (R_E/p)² · (2 - 5/2·sin²(i))
  d(M)/dt_add = -3/2 · n · J2 · (R_E/p)² · √(1-e²) · (1 - 3/2·sin²(i))

  where p = a(1-e²), n = √(μ/a³).

Semi-major axis, eccentricity, and inclination are assumed secular-
constant to first order (they do drift slowly, but the relevant signal
for manoeuvre detection is the discrete jump, not the long-term trend).

This model is adequate for computing residuals over inter-TLE intervals
of 6-48 hours.  It does NOT replace SGP4 for precision use cases.

REFERENCES
----------
Bate, Mueller & White (1971), Fundamentals of Astrodynamics.
Vallado (2013), Fundamentals of Astrodynamics and Applications, §9.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ── Physical constants (SI/km-s units) ──────────────────────────────
# Source: IAU 2012 / WGS-84

MU_EARTH: float = 398_600.4418   # km³ s⁻²  — Earth gravitational parameter
R_EARTH:  float = 6_378.137      # km        — Earth equatorial radius (WGS-84)
J2:       float = 1.082_626_68e-3 # dimensionless — Earth second zonal harmonic

# Numerical tolerances
_KEPLER_TOL:     float = 1e-10   # eccentric-anomaly convergence tolerance (rad)
_KEPLER_MAX_IT:  int   = 50      # maximum Newton-Raphson iterations


# ════════════════════════════════════════════════════════════════════
# Keplerian element container
# ════════════════════════════════════════════════════════════════════

@dataclass
class KeplerianElements:
    """
    Mean Keplerian orbital elements compatible with TLE mean-element
    representation.

    Attributes
    ----------
    a    : float  Semi-major axis (km).  For circular orbits, a ≈ R_E + altitude.
    e    : float  Eccentricity (dimensionless, 0 ≤ e < 1).
    i    : float  Inclination (rad, 0 ≤ i ≤ π).
    raan : float  Right ascension of ascending node Ω (rad).
    argp : float  Argument of perigee ω (rad).
    M    : float  Mean anomaly M (rad).

    Note
    ----
    These are *mean* elements as published in TLE records.  They differ
    from osculating elements by short-period and long-period perturbation
    terms.  The J2 secular propagator in this module operates consistently
    in mean-element space.
    """
    a:    float  # km
    e:    float  # dimensionless
    i:    float  # rad
    raan: float  # rad
    argp: float  # rad
    M:    float  # rad

    # ── vector interface (expected by model.py / train.py) ──────────
    def as_vector(self) -> np.ndarray:
        """Return [a, e, i, raan, argp, M] as float64 array."""
        return np.array([self.a, self.e, self.i, self.raan, self.argp, self.M],
                        dtype=np.float64)

    @staticmethod
    def from_vector(v: np.ndarray) -> "KeplerianElements":
        """Construct from a length-6 array [a, e, i, raan, argp, M]."""
        return KeplerianElements(
            a=float(v[0]), e=float(v[1]), i=float(v[2]),
            raan=float(v[3]), argp=float(v[4]), M=float(v[5]),
        )

    def is_valid(self) -> bool:
        """Basic sanity check (not exhaustive)."""
        return (
            self.a > R_EARTH          # satellite above ground
            and 0.0 <= self.e < 1.0   # bound elliptic orbit
            and 0.0 <= self.i <= math.pi
        )


# ════════════════════════════════════════════════════════════════════
# Core orbital mechanics
# ════════════════════════════════════════════════════════════════════

def mean_motion(a: float) -> float:
    """
    Keplerian mean motion n = √(μ / a³)  [rad s⁻¹].

    Parameters
    ----------
    a : float  Semi-major axis (km).

    Returns
    -------
    n : float  Mean motion (rad s⁻¹).
    """
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive; got {a} km")
    return math.sqrt(MU_EARTH / a ** 3)


def solve_kepler(M: float, e: float) -> float:
    """
    Solve Kepler's equation  M = E − e·sin(E)  for eccentric anomaly E.

    Uses Newton-Raphson iteration with:
    * Initial guess: E₀ = M  (adequate for e < 0.1, which covers all TLE orbits)
    * Convergence criterion: |ΔE| < _KEPLER_TOL
    * Maximum iterations: _KEPLER_MAX_IT

    For near-circular orbits (e → 0), the method converges in 1-2 iterations.

    Parameters
    ----------
    M : float  Mean anomaly (rad, any real — normalised internally).
    e : float  Eccentricity (0 ≤ e < 1).

    Returns
    -------
    E : float  Eccentric anomaly (rad, in [0, 2π)).
    """
    M = math.fmod(M, 2.0 * math.pi)
    if M < 0:
        M += 2.0 * math.pi

    # Initial guess
    E = M
    for _ in range(_KEPLER_MAX_IT):
        f  = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < _KEPLER_TOL:
            break

    return math.fmod(E, 2.0 * math.pi)


def j2_secular_rates(
    a: float, e: float, i: float
) -> Tuple[float, float, float]:
    """
    Secular drift rates of RAAN, argument of perigee, and mean anomaly
    due to Earth's J2 oblateness perturbation.

    Equations (Bate et al. 1971, §9.2; Vallado 2013, §9.6):

      dΩ/dt = -(3/2) · n · J2 · (R_E/p)² · cos i
      dω/dt = -(3/2) · n · J2 · (R_E/p)² · (2 - 5/2·sin²i)
      dM/dt_J2 = -(3/2) · n · J2 · (R_E/p)² · √(1−e²) · (1 − 3/2·sin²i)

    where p = a(1−e²) is the semi-latus rectum.

    Parameters
    ----------
    a : float  Semi-major axis (km).
    e : float  Eccentricity.
    i : float  Inclination (rad).

    Returns
    -------
    raan_dot : float  dΩ/dt (rad s⁻¹)
    argp_dot : float  dω/dt (rad s⁻¹)
    M_dot_J2 : float  additional dM/dt due to J2 (rad s⁻¹)
    """
    n = mean_motion(a)
    p = a * (1.0 - e * e)           # semi-latus rectum (km)
    if p <= 0:
        raise ValueError(f"Non-positive semi-latus rectum p={p:.3f} km")

    k = -1.5 * n * J2 * (R_EARTH / p) ** 2
    sin_i = math.sin(i)
    cos_i = math.cos(i)

    raan_dot = k * cos_i
    argp_dot = k * (2.0 - 2.5 * sin_i ** 2)
    M_dot_J2 = k * math.sqrt(1.0 - e * e) * (1.0 - 1.5 * sin_i ** 2)

    return raan_dot, argp_dot, M_dot_J2


def propagate(elements: KeplerianElements, dt_seconds: float) -> KeplerianElements:
    """
    Propagate Keplerian mean elements forward by dt_seconds using
    two-body + J2 secular corrections.

    This is the "null-hypothesis physics model": if nothing perturbs the
    satellite, this is where it should be after dt seconds.  Any deviation
    between this prediction and the next real TLE observation forms the
    physics residual used for anomaly detection.

    First-order secular assumption: a, e, i do not change (J2 secular
    perturbations on these elements are second-order).  RAAN, argp, and M
    advance at their J2-corrected rates.

    Parameters
    ----------
    elements   : KeplerianElements  State at epoch t.
    dt_seconds : float              Propagation interval (seconds).

    Returns
    -------
    KeplerianElements  Predicted state at epoch t + dt_seconds.
    """
    if dt_seconds <= 0:
        return KeplerianElements(
            a=elements.a, e=elements.e, i=elements.i,
            raan=elements.raan, argp=elements.argp, M=elements.M,
        )

    n = mean_motion(elements.a)
    raan_dot, argp_dot, M_dot_J2 = j2_secular_rates(
        elements.a, elements.e, elements.i
    )

    # Advance angles, normalise to [0, 2π)
    new_raan = math.fmod(elements.raan + raan_dot * dt_seconds,  2.0 * math.pi)
    new_argp = math.fmod(elements.argp + argp_dot * dt_seconds,  2.0 * math.pi)
    new_M    = math.fmod(elements.M    + (n + M_dot_J2) * dt_seconds, 2.0 * math.pi)

    if new_raan < 0: new_raan += 2.0 * math.pi
    if new_argp < 0: new_argp += 2.0 * math.pi
    if new_M    < 0: new_M    += 2.0 * math.pi

    return KeplerianElements(
        a=elements.a,       # a, e, i unchanged (J2 secular ≈ constant to 1st order)
        e=elements.e,
        i=elements.i,
        raan=new_raan,
        argp=new_argp,
        M=new_M,
    )


def wrap_angle(x: float) -> float:
    """
    Wrap angle to (-π, π].

    Used for angular residuals so that  179° vs −179°  gives ~±2°, not ±358°.
    """
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def elements_residual(
    observed: KeplerianElements,
    predicted: KeplerianElements,
) -> np.ndarray:
    """
    Compute the signed physics residual vector:
        r = observed − predicted

    Angles are wrapped to (−π, π] to eliminate discontinuities.

    Element ordering: [Δa, Δe, Δi, ΔRAAN, Δargp, ΔM]
    Units:            [km, —,  rad,  rad,   rad,   rad]

    NOTE on ΔM
    ----------
    Mean anomaly residuals computed between consecutive TLEs are physically
    meaningful only when the EXACT inter-TLE elapsed time is used for
    propagation.  If the propagation dt matches the actual time difference
    between TLE epochs, |ΔM| for a nominal satellite should be in the range
    0.01–0.1 rad.  If a wrong or approximate dt is used, ΔM becomes
    dominated by the dt error and carries no manoeuvre information.
    Dataset.py ensures the actual inter-TLE dt is used; this function simply
    performs the wrapped subtraction.

    Parameters
    ----------
    observed  : KeplerianElements  Next observed TLE state.
    predicted : KeplerianElements  Physics-propagated prediction.

    Returns
    -------
    residual : np.ndarray  Shape (6,) signed residual vector.
    """
    return np.array([
        observed.a    - predicted.a,
        observed.e    - predicted.e,
        wrap_angle(observed.i    - predicted.i),
        wrap_angle(observed.raan - predicted.raan),
        wrap_angle(observed.argp - predicted.argp),
        wrap_angle(observed.M    - predicted.M),
    ], dtype=np.float64)


# ════════════════════════════════════════════════════════════════════
# Vectorised propagation (used by dataset.py for speed)
# ════════════════════════════════════════════════════════════════════

def propagate_batch(
    elements_array: np.ndarray,
    dt_seconds_array: np.ndarray,
) -> np.ndarray:
    """
    Vectorised J2+Kepler propagation over a batch of states and
    (potentially different) propagation intervals.

    This is the fast path used by compute_residual_sequences() in
    dataset.py.  It replicates the scalar propagate() logic using
    NumPy broadcasting and avoids Python loops.

    Parameters
    ----------
    elements_array  : np.ndarray  Shape (S, T, 6)  — element sequences.
    dt_seconds_array: np.ndarray  Shape (S, T)     — propagation dt per step.
        Each dt_seconds_array[s, t] is the actual elapsed time between
        grid slot t and grid slot t+1 for satellite s.

    Returns
    -------
    predicted : np.ndarray  Shape (S, T, 6) — predicted next states.

    Note
    ----
    Only the "from" elements (S, T-1, 6) and dts (S, T-1) should be
    passed when computing residuals; the returned shape matches the input.
    """
    a    = elements_array[..., 0]   # (..., ) broadcast-safe
    e    = elements_array[..., 1]
    i    = elements_array[..., 2]
    raan = elements_array[..., 3]
    argp = elements_array[..., 4]
    M_in = elements_array[..., 5]

    n    = np.sqrt(MU_EARTH / a ** 3)
    p    = a * (1.0 - e ** 2)
    k    = -1.5 * n * J2 * (R_EARTH / p) ** 2

    sin_i = np.sin(i)
    raan_dot = k * np.cos(i)
    argp_dot = k * (2.0 - 2.5 * sin_i ** 2)
    M_dot_J2 = k * np.sqrt(np.clip(1.0 - e ** 2, 0, None)) * (1.0 - 1.5 * sin_i ** 2)

    dt = dt_seconds_array

    pred_raan = np.mod(raan + raan_dot * dt, 2.0 * np.pi)
    pred_argp = np.mod(argp + argp_dot * dt, 2.0 * np.pi)
    pred_M    = np.mod(M_in + (n + M_dot_J2) * dt, 2.0 * np.pi)

    return np.stack([a, e, i, pred_raan, pred_argp, pred_M], axis=-1)


def residuals_batch(
    elements_obs:    np.ndarray,
    dt_seconds_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute physics residuals for a full observation sequence.

    For each satellite k and time step t:
        residual[k, t] = observed[k, t+1] − propagate(observed[k, t], dt[k, t])

    Using the ACTUAL propagation dt for each inter-observation step.

    Parameters
    ----------
    elements_obs    : np.ndarray  Shape (S, T, 6)  — observed element sequences.
    dt_seconds_grid : np.ndarray  Shape (S, T-1)   — actual elapsed seconds
                      between observation t and observation t+1 for satellite s.

    Returns
    -------
    residuals : np.ndarray  Shape (S, T-1, 6) — signed, angle-wrapped residuals.
    """
    prev    = elements_obs[:, :-1, :]    # (S, T-1, 6)  from-state
    obs_nxt = elements_obs[:, 1:,  :]   # (S, T-1, 6)  to-state (observed)

    predicted = propagate_batch(prev, dt_seconds_grid)   # (S, T-1, 6)

    def wrap(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    return np.stack([
        obs_nxt[..., 0] - predicted[..., 0],          # Δa   (km)
        obs_nxt[..., 1] - predicted[..., 1],          # Δe
        wrap(obs_nxt[..., 2] - predicted[..., 2]),    # Δi   (rad)
        wrap(obs_nxt[..., 3] - predicted[..., 3]),    # ΔRAAN (rad)
        wrap(obs_nxt[..., 4] - predicted[..., 4]),    # Δargp (rad)
        wrap(obs_nxt[..., 5] - predicted[..., 5]),    # ΔM   (rad)
    ], axis=-1)


# ════════════════════════════════════════════════════════════════════
# ECI conversion (optional — not used in main pipeline)
# ════════════════════════════════════════════════════════════════════

def elements_to_state_vector(
    elements: KeplerianElements,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian mean elements to ECI Cartesian position/velocity.

    Uses the osculating interpretation (short-period terms ignored).
    Intended for visualisation, NOT for precision orbit determination.

    Returns
    -------
    pos : np.ndarray  ECI position (km), shape (3,).
    vel : np.ndarray  ECI velocity (km s⁻¹), shape (3,).
    """
    a, e, i = elements.a, elements.e, elements.i
    raan, argp, M = elements.raan, elements.argp, elements.M

    E  = solve_kepler(M, e)
    nu = 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(E / 2.0),
        math.sqrt(1.0 - e) * math.cos(E / 2.0),
    )
    r  = a * (1.0 - e * math.cos(E))
    p  = a * (1.0 - e * e)

    # Perifocal frame
    x_pf = r * math.cos(nu)
    y_pf = r * math.sin(nu)
    vx_pf = -math.sqrt(MU_EARTH / p) * math.sin(nu)
    vy_pf =  math.sqrt(MU_EARTH / p) * (e + math.cos(nu))

    # Rotation matrix (perifocal → ECI)
    cO, sO = math.cos(raan), math.sin(raan)
    cw, sw = math.cos(argp), math.sin(argp)
    ci, si = math.cos(i),    math.sin(i)

    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                  cw * si,                  ci     ],
    ])

    pf_pos = np.array([x_pf, y_pf, 0.0])
    pf_vel = np.array([vx_pf, vy_pf, 0.0])

    return R @ pf_pos, R @ pf_vel


# ════════════════════════════════════════════════════════════════════
# Manoeuvre Δv Estimation
# ════════════════════════════════════════════════════════════════════

def estimate_delta_v(delta_a_km: float, a_km: float) -> float:
    """Estimate manoeuvre delta-V from TLE-derived semi-major axis change.

    Derivation
    ----------
    For a tangential (prograde/retrograde) burn applied to a near-circular
    orbit, the change in semi-major axis relates to delta-V via the
    vis-viva / Gauss equations (linearised for small Δa):

        ΔV ≈ (n · Δa) / 2

    where n = √(μ / a³) is the mean motion [rad/s] and Δa is the change
    in semi-major axis [km].  Converting: ΔV [m/s] = 1000 × ΔV [km/s].

    IMPORTANT CAVEATS
    -----------------
    * TLEs represent *mean* elements averaged over short-period terms.
      The Δa between two consecutive TLEs reflects the combined effect of
      the manoeuvre burn AND the measurement noise of the ground-station
      orbit determination fit (typically ±0.001–0.050 km for LEO).
    * This formula assumes a tangential burn. Radial burns change the
      orbit shape differently and cannot be decomposed from TLEs alone.
    * GEO satellite burns often include an inclination component (N/S
      station-keeping) which produces a RAAN/inclination change, not
      captured here.
    * The estimate is order-of-magnitude only (±50% error is typical).
    * Use ONLY for post-hoc characterisation of detected events, not for
      operational manoeuvre planning.

    Args:
        delta_a_km : Change in semi-major axis between two TLE epochs [km].
                     Positive = orbit raised, negative = orbit lowered.
        a_km       : Reference semi-major axis (pre-manoeuvre) [km].

    Returns:
        Estimated |ΔV| [m/s].  Always non-negative.

    Example
    -------
    >>> dv = estimate_delta_v(delta_a_km=1.5, a_km=7156.0)   # Sentinel-like
    >>> print(f"ΔV ≈ {dv:.2f} m/s")
    ΔV ≈ 0.52 m/s
    """
    if a_km <= 0:
        raise ValueError(f"a_km must be positive, got {a_km}")
    n       = mean_motion(a_km)                   # rad/s
    dv_km_s = (n * abs(delta_a_km)) / 2.0         # km/s  (linearised Gauss)
    return dv_km_s * 1000.0                        # → m/s


def estimate_delta_v_batch(
    delta_a_array: "np.ndarray",
    a_array:       "np.ndarray",
) -> "np.ndarray":
    """Vectorised version of estimate_delta_v.

    Args:
        delta_a_array : shape (...,) change in semi-major axis [km]
        a_array       : shape (...,) reference semi-major axis [km]

    Returns:
        Estimated |ΔV| [m/s], same shape as inputs.
    """
    import numpy as _np
    a_arr      = _np.asarray(a_array,       dtype=float)
    da_arr     = _np.asarray(delta_a_array, dtype=float)
    n_arr      = _np.sqrt(MU_EARTH / a_arr ** 3)   # rad/s
    dv_km_s    = (n_arr * _np.abs(da_arr)) / 2.0
    return dv_km_s * 1000.0                         # → m/s


# ════════════════════════════════════════════════════════════════════
# Self-test
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("physics.py self-test")
    print("=" * 50)

    # 1. mean_motion sanity
    # ISS: ~400 km altitude → a ≈ 6778 km → n ≈ 0.00113 rad/s → ~15.5 rev/day
    a_iss = R_EARTH + 400.0
    n_iss = mean_motion(a_iss)
    rev_per_day = n_iss * 86400 / (2 * math.pi)
    print(f"ISS mean motion: {n_iss:.6f} rad/s  ({rev_per_day:.2f} rev/day, expected ~15.5)")
    assert 15.0 < rev_per_day < 16.0, "ISS mean motion out of range"

    # 2. Kepler solver: E=1 rad, e=0.01
    M_test = 1.0 - 0.01 * math.sin(1.0)
    E_sol = solve_kepler(M_test, 0.01)
    residual_kepler = E_sol - 0.01 * math.sin(E_sol) - M_test
    print(f"Kepler solver residual: {residual_kepler:.2e}  (expected < {_KEPLER_TOL:.0e})")
    assert abs(residual_kepler) < _KEPLER_TOL * 10, "Kepler solver not converged"

    # 3. propagate: circular orbit, dt=one period → should return to ~same M
    a_circ = R_EARTH + 500.0
    period = 2.0 * math.pi / mean_motion(a_circ)
    elems  = KeplerianElements(a=a_circ, e=0.001, i=math.radians(51.6),
                               raan=1.0, argp=0.5, M=0.3)
    prop   = propagate(elems, period)
    dM = wrap_angle(prop.M - elems.M)
    print(f"One-period M advance: {math.degrees(dM):.3f}° (expected ≈ 0° modulo period)")

    # 4. J2 RAAN drift for ISS-like orbit
    raan_dot, argp_dot, M_corr = j2_secular_rates(a_circ, 0.001, math.radians(51.6))
    raan_deg_day = math.degrees(raan_dot) * 86400
    print(f"ISS-like RAAN drift: {raan_deg_day:.4f} deg/day (expected ~ -7 deg/day)")
    assert -10 < raan_deg_day < -4, f"RAAN drift out of expected range: {raan_deg_day}"

    # 5. wrap_angle
    assert abs(wrap_angle(math.radians(181.0)) - math.radians(-179.0)) < 1e-9
    assert abs(wrap_angle(math.radians(-181.0)) - math.radians(179.0)) < 1e-9
    print("wrap_angle: OK")

    # 6. elements_residual is zero for identical states
    e0 = KeplerianElements(a=7000.0, e=0.001, i=1.0, raan=1.0, argp=1.0, M=1.0)
    r0 = elements_residual(e0, e0)
    assert np.allclose(r0, 0.0, atol=1e-12), "Residual of identical states should be zero"
    print("elements_residual self-consistency: OK")

    # 7. as_vector / from_vector round-trip
    v   = elems.as_vector()
    e2  = KeplerianElements.from_vector(v)
    assert np.allclose(v, e2.as_vector())
    print("as_vector / from_vector: OK")

    print()
    print("All tests passed.")
