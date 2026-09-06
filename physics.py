"""
physics.py
----------
A from-scratch two-body + J2-secular orbital propagator.

Why write our own instead of importing `sgp4`?
Because the whole pitch of this project is that the NEURAL NETWORK's job is to
learn the residual between "what pure physics says should happen" and "what
actually happened" to a satellite. If we import a black-box propagator, we
can't explain what the residual physically means. If we derive it ourselves
(two-body Kepler motion + the dominant J2 oblateness perturbation), the
residual has a clean physical interpretation:

    residual(t) = observed_state(t) - physics_predicted_state(t)

Anything NOT explained by gravity + Earth's oblateness (drag, maneuvers,
degradation, tumbling, fragmentation) shows up in that residual. That's the
signal we hand to the network. Nothing here is a mystery box.

Units: km, seconds, radians (unless noted).
"""

import numpy as np

MU_EARTH = 398600.4418          # km^3 / s^2, Earth's gravitational parameter
R_EARTH = 6378.137               # km, equatorial radius
J2 = 1.08262668e-3               # Earth's J2 oblateness coefficient


class KeplerianElements:
    """A minimal orbital-element state: [a, e, i, raan, argp, M]."""

    def __init__(self, a, e, i, raan, argp, M):
        self.a = a          # semi-major axis (km)
        self.e = e          # eccentricity
        self.i = i          # inclination (rad)
        self.raan = raan    # right ascension of ascending node (rad)
        self.argp = argp    # argument of perigee (rad)
        self.M = M          # mean anomaly (rad)

    def as_vector(self):
        return np.array([self.a, self.e, self.i, self.raan, self.argp, self.M])

    @staticmethod
    def from_vector(v):
        return KeplerianElements(*v)


def mean_motion(a):
    """Rad/s. n = sqrt(mu / a^3)"""
    return np.sqrt(MU_EARTH / a ** 3)


def solve_kepler(M, e, tol=1e-10, max_iter=50):
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E."""
    M = np.mod(M, 2 * np.pi)
    E = M.copy() if isinstance(M, np.ndarray) else M
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def j2_secular_rates(a, e, i):
    """
    Secular drift rates of RAAN, argument of perigee, and mean anomaly
    caused by Earth's oblateness (J2). This is the #1 real-world reason a
    satellite's orbit drifts away from a pure two-body prediction over
    days/weeks, so a naive Keplerian-only baseline WILL look like it has
    "anomalies" everywhere unless we correct for this. Modeling J2 is what
    separates "we used Kepler's laws" (freshman-year physics) from
    "we understand orbital perturbation theory" (what actually impresses
    an aerospace-adjacent professor).
    """
    n = mean_motion(a)
    p = a * (1 - e ** 2)
    factor = -1.5 * n * J2 * (R_EARTH / p) ** 2

    raan_dot = factor * np.cos(i)
    argp_dot = factor * (2.5 * np.sin(i) ** 2 - 2) * -1  # standard J2 argp drift
    argp_dot = factor * (2 - 2.5 * np.sin(i) ** 2)
    M_dot_correction = factor * np.sqrt(1 - e ** 2) * (1 - 1.5 * np.sin(i) ** 2)
    return raan_dot, argp_dot, M_dot_correction


def propagate(elements: KeplerianElements, dt_seconds: float) -> KeplerianElements:
    """
    Propagate a Keplerian state forward by dt_seconds under two-body motion
    + J2 secular perturbation. This is our "null hypothesis" physics model:
    if nothing weird is happening to the satellite, this is where it should be.
    """
    n = mean_motion(elements.a)
    raan_dot, argp_dot, M_dot_corr = j2_secular_rates(elements.a, elements.e, elements.i)

    new_raan = elements.raan + raan_dot * dt_seconds
    new_argp = elements.argp + argp_dot * dt_seconds
    new_M = elements.M + (n + M_dot_corr) * dt_seconds

    return KeplerianElements(
        a=elements.a,       # a, e, i are (to first order / secular approx) unperturbed by J2
        e=elements.e,
        i=elements.i,
        raan=np.mod(new_raan, 2 * np.pi),
        argp=np.mod(new_argp, 2 * np.pi),
        M=np.mod(new_M, 2 * np.pi),
    )


def elements_to_state_vector(elements: KeplerianElements):
    """Convert Keplerian elements -> ECI position/velocity (km, km/s)."""
    a, e, i = elements.a, elements.e, elements.i
    raan, argp, M = elements.raan, elements.argp, elements.M

    E = solve_kepler(M, e)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * np.cos(E))

    # position in perifocal frame
    x_pf = r * np.cos(nu)
    y_pf = r * np.sin(nu)

    n = mean_motion(a)
    p = a * (1 - e ** 2)
    vx_pf = -np.sqrt(MU_EARTH / p) * np.sin(nu)
    vy_pf = np.sqrt(MU_EARTH / p) * (e + np.cos(nu))

    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_w, sin_w = np.cos(argp), np.sin(argp)
    cos_i, sin_i = np.cos(i), np.sin(i)

    R11 = cos_O * cos_w - sin_O * sin_w * cos_i
    R12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    R21 = sin_O * cos_w + cos_O * sin_w * cos_i
    R22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    R31 = sin_w * sin_i
    R32 = cos_w * sin_i

    pos = np.array([R11 * x_pf + R12 * y_pf,
                     R21 * x_pf + R22 * y_pf,
                     R31 * x_pf + R32 * y_pf])
    vel = np.array([R11 * vx_pf + R12 * vy_pf,
                     R21 * vx_pf + R22 * vy_pf,
                     R31 * vx_pf + R32 * vy_pf])
    return pos, vel


def elements_residual(observed: KeplerianElements, predicted: KeplerianElements):
    """
    Physically-meaningful residual vector between what we observed and what
    pure physics predicted. Angles are wrapped to [-pi, pi] so a satellite
    that's 359 deg vs 1 deg doesn't look like a huge anomaly.
    """
    def wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    return np.array([
        observed.a - predicted.a,
        observed.e - predicted.e,
        wrap(observed.i - predicted.i),
        wrap(observed.raan - predicted.raan),
        wrap(observed.argp - predicted.argp),
        wrap(observed.M - predicted.M),
    ])
