"""
dataset.py
----------
Two jobs:

1. Pull real TLE history for a constellation from CelesTrak (no auth needed)
   -> `fetch_celestrak_group()`. Run this on your own machine / Colab where
   you have internet; the sandbox this was written in has network disabled,
   so it can't be executed here, but it's a plain requests.get() against a
   public, documented endpoint.

2. A synthetic constellation simulator -> `simulate_constellation()` that
   generates physically-realistic multi-satellite TLE-like histories WITH
   ground-truth-labeled anomalies (drag decay creep, thruster maneuvers,
   attitude/tumble events). This is what makes the project demoable and
   gradeable in a 10-minute slot: real CelesTrak anomalies are rare, unlabeled,
   and slow (decay takes months). You need controlled ground truth to show a
   confusion matrix / ROC curve in your presentation. Use synthetic for the
   live demo + metrics, use real CelesTrak pulls as the "look, it also flags
   something interesting on live Starlink data" wow-moment at the end.

Everything downstream (model.py, train.py) only cares about the array shapes
produced here, so swapping the synthetic simulator for real fetched TLEs is
a drop-in replacement once you have a Space-Track/CelesTrak account.
"""

import numpy as np
from physics import KeplerianElements, propagate, elements_residual, mean_motion

EARTH_MU = 398600.4418


def fetch_celestrak_group(group="starlink", fmt="json"):
    """
    Real-data path (needs internet — run outside this sandbox).
    CelesTrak's supplemental GP data API, no login required:
        https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json
    Returns a list of dicts with orbital elements per satellite at fetch time.
    To build a *history* (needed for residual/anomaly detection), call this
    on a cron/schedule (e.g. every 6-12h for a week) and append to a local
    cache — TLE epochs update roughly daily per satellite anyway.
    """
    import requests
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT={fmt}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _random_shell(n_sats, altitude_km, inclination_deg, rng):
    """Place n_sats in circular orbits at one altitude/inclination shell,
    spread across RAAN + mean anomaly like a real constellation shell."""
    a = np.full(n_sats, R_EARTH_PLUS(altitude_km)) + rng.normal(0, 0.5, n_sats)
    e = np.clip(rng.normal(0.0008, 0.0003, n_sats), 0.0001, 0.01)
    i = np.radians(inclination_deg) + np.radians(rng.normal(0, 0.05, n_sats))
    raan = rng.uniform(0, 2 * np.pi, n_sats)
    argp = rng.uniform(0, 2 * np.pi, n_sats)
    M = rng.uniform(0, 2 * np.pi, n_sats)
    return [KeplerianElements(a[k], e[k], i[k], raan[k], argp[k], M[k]) for k in range(n_sats)]


def R_EARTH_PLUS(alt_km):
    return 6378.137 + alt_km


def simulate_constellation(
    n_shells=3,
    sats_per_shell=12,
    n_steps=120,
    dt_hours=6.0,
    anomaly_rate=0.08,
    seed=0,
):
    """
    Builds a synthetic multi-shell constellation (think: 3 Starlink-like
    orbital shells at different altitude/inclination) and rolls it forward
    in time. At each step, the TRUE physics propagation is applied, then for
    a random subset of (satellite, time) pairs we inject a labeled anomaly:

      - 'decay_drag'   : semi-major axis creeps down faster than J2 predicts
                         (atmospheric drag increasing -> deorbiting satellite)
      - 'maneuver'     : a sudden delta-v style jump in a/e/argp (station-
                         keeping burn or, if large, a collision-avoidance burn)
      - 'tumble'       : mean anomaly / argp noise spikes (attitude control
                         failure -> satellite is no longer nicely Keplerian)

    Returns
    -------
    elements_true : (S, T, 6) array of ground-truth Keplerian elements
    elements_obs  : (S, T, 6) array of "observed" elements (true + sensor noise)
    labels        : (S, T) int array, 0 = nominal, 1/2/3 = anomaly type
    shell_id      : (S,) which orbital shell each satellite belongs to
    """
    rng = np.random.default_rng(seed)
    dt_sec = dt_hours * 3600.0

    shell_altitudes = np.linspace(540, 570, n_shells)          # km, close-ish shells
    shell_inclinations = np.linspace(53.0, 97.6, n_shells)     # deg

    all_sats, shell_id = [], []
    for s in range(n_shells):
        sats = _random_shell(sats_per_shell, shell_altitudes[s], shell_inclinations[s], rng)
        all_sats.extend(sats)
        shell_id.extend([s] * sats_per_shell)
    shell_id = np.array(shell_id)
    n_sats = len(all_sats)

    elements_true = np.zeros((n_sats, n_steps, 6))
    labels = np.zeros((n_sats, n_steps), dtype=int)

    state = all_sats
    for t in range(n_steps):
        for k in range(n_sats):
            state[k] = propagate(state[k], dt_sec)

            # --- inject anomalies on top of clean physics ---
            if t > 5 and rng.random() < anomaly_rate / n_steps * 15:
                kind = rng.choice(["decay_drag", "maneuver", "tumble"])
                if kind == "decay_drag":
                    state[k].a -= rng.uniform(0.05, 0.3)          # km, extra drag decay
                    labels[k, t:] = np.maximum(labels[k, t:], 1)  # persists forward
                elif kind == "maneuver":
                    state[k].a += rng.normal(0, 0.15)
                    state[k].argp = np.mod(state[k].argp + rng.normal(0, 0.05), 2 * np.pi)
                    labels[k, t] = 2                              # instantaneous event
                elif kind == "tumble":
                    state[k].M = np.mod(state[k].M + rng.normal(0, 0.2), 2 * np.pi)
                    labels[k, t] = 3

            elements_true[k, t] = state[k].as_vector()

    # sensor/tracking noise on top of the "true" trajectory -> what an
    # observer (radar/telescope -> TLE fit) would actually report
    noise_std = np.array([0.02, 0.00002, 1e-5, 1e-5, 1e-5, 2e-5])
    elements_obs = elements_true + rng.normal(0, noise_std, elements_true.shape)

    return elements_true, elements_obs, labels, shell_id


def compute_residual_sequences(elements_obs, dt_hours=6.0):
    """
    For every satellite/timestep, propagate the PREVIOUS observed state
    forward with pure physics and diff it against what was actually observed
    next. This is the core "physics-informed" feature: a (S, T-1, 6) tensor
    of residuals that a naive Keplerian model cannot explain.
    """
    n_sats, n_steps, _ = elements_obs.shape
    dt_sec = dt_hours * 3600.0
    residuals = np.zeros((n_sats, n_steps - 1, 6))
    for k in range(n_sats):
        for t in range(n_steps - 1):
            prev = KeplerianElements.from_vector(elements_obs[k, t])
            predicted = propagate(prev, dt_sec)
            observed_next = KeplerianElements.from_vector(elements_obs[k, t + 1])
            residuals[k, t] = elements_residual(observed_next, predicted)
    return residuals


def build_orbital_neighbor_graph(elements_snapshot, k_neighbors=4):
    """
    THE graph-construction step behind OrbitGNN's core idea: two satellites
    are "neighbors" if they sit in similar orbital planes (close inclination
    + RAAN + altitude), regardless of where they physically are right now.
    That's the aerospace-correct notion of "peers" for a constellation
    satellite — not spatial proximity at an instant, but orbital-plane
    similarity. A satellite that starts diverging from its orbital-plane
    peers is far more suspicious than one that just looks different from
    its own noisy history.

    Returns a (S, S) symmetric adjacency matrix (0/1).
    """
    n_sats = elements_snapshot.shape[0]
    a = elements_snapshot[:, 0]
    i = elements_snapshot[:, 2]
    raan = elements_snapshot[:, 3]

    # normalize features so distance isn't dominated by a's larger scale
    feat = np.stack([
        (a - a.mean()) / (a.std() + 1e-6),
        (i - i.mean()) / (i.std() + 1e-6),
        np.sin(raan), np.cos(raan),
    ], axis=1)

    dist = np.linalg.norm(feat[:, None, :] - feat[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)

    adj = np.zeros((n_sats, n_sats))
    for k in range(n_sats):
        nn = np.argsort(dist[k])[:k_neighbors]
        adj[k, nn] = 1
    adj = np.maximum(adj, adj.T)  # symmetrize
    return adj


if __name__ == "__main__":
    # quick smoke test — runs entirely offline
    et, eo, labels, shell_id = simulate_constellation(n_shells=2, sats_per_shell=6, n_steps=40)
    res = compute_residual_sequences(eo)
    adj = build_orbital_neighbor_graph(eo[:, -1, :])
    print("elements_true", et.shape, "| residuals", res.shape, "| adjacency", adj.shape)
    print("anomaly rate:", (labels > 0).mean().round(4))
    print("avg neighbors per satellite:", adj.sum(1).mean())
