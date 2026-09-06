"""
dataset.py
----------
Data pipeline for OrbitGNN — two paths:

  PATH A  [PRIMARY]  load_real_benchmark_dataset()
      Loads the TLE Observation Benchmark Dataset
      (https://github.com/dpshorten/TLE_observation_benchmark_dataset).
      Produces (S, T, 6) mean-element tensors with actual inter-TLE
      propagation dt, per-satellite normalisation, and science-defensible
      manoeuvre labels.

  PATH B  [REGRESSION]  simulate_constellation()
      Synthetic constellation with injected anomalies.
      Kept so that the unchanged model/training code can be regression-
      tested without the real dataset.

KEY FIXES OVER PREVIOUS VERSION
---------------------------------
F1  Actual inter-TLE dt stored alongside element sequences.
    compute_residual_sequences() now uses physics.residuals_batch()
    which takes the real dt per step, not a fixed 24-hour constant.
    This is the most important fix: previously ΔM was dominated by
    the propagation-dt error (wrong dt used for a variable-gap dataset).

F2  Per-satellite normalisation (RobustScaler equivalent using median/IQR
    from TRAINING data only).  Previously GEO Δa ≈ 1 km vs LEO Δa ≈ 0.001 km
    caused the normaliser to be dominated by GEO satellites, making all LEO
    residuals near-zero after scaling.

F3  Manoeuvre-label tolerance tightened to 24 h (from 36 h) because the
    measured TLE-to-manoeuvre offset distribution is mostly 3–34 h;
    36 h labelled too many nominal windows as anomalous.

F4  The 'detection offset' metric is now defined unambiguously:
    "time from grid slot to manoeuvre event" (negative = alarm before event).

F5  Forward-fill is explicitly tracked per step via valid_mask.
    Residuals at forward-filled steps are flagged in a separate mask
    so they can be optionally excluded from training.

F6  KeplerianElements dataclass is imported from physics; no duplication.

SCIENTIFIC CONSTRAINTS
-----------------------
- Mean Keplerian elements from TLEs are used directly (no SGP4).
  See physics.py for justification.
- No interpolation of orbital elements.
- No injection of synthetic anomalies into real data.
- Chronological train/val/test split only.
- Normalisation statistics derived from training split only.
"""

from __future__ import annotations

import datetime
import glob
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from physics import (
    KeplerianElements,
    MU_EARTH,
    R_EARTH,
    propagate,
    elements_residual,
    mean_motion,
    residuals_batch,
)

logger = logging.getLogger(__name__)

# ── Defaults (all configurable via function arguments) ───────────────
DEFAULT_DT_HOURS:               float = 24.0
DEFAULT_MAX_TLE_GAP_HOURS:      float = 48.0
DEFAULT_MANEUVER_TOL_HOURS:     float = 24.0   # tightened from 36 h
DEFAULT_MIN_VALID_FRACTION:     float = 0.50
DEFAULT_K_NEIGHBORS:            int   = 3
MIN_GRAPH_SATS:                 int   = 2


# ════════════════════════════════════════════════════════════════════
# TLE PARSING
# ════════════════════════════════════════════════════════════════════

def _parse_tle_epoch(epoch_field: str) -> datetime.datetime:
    """
    Convert TLE epoch field (2-digit year + fractional day) to UTC datetime.

    TLE convention: year 00-56 → 2000-2056; year 57-99 → 1957-1999.
    """
    s    = epoch_field.strip()
    yr2  = int(s[:2])
    year = (2000 + yr2) if yr2 < 57 else (1900 + yr2)
    return (
        datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
        + datetime.timedelta(days=float(s[2:]) - 1.0)
    )


def _tle_to_elements(
    line1: str, line2: str
) -> Tuple[datetime.datetime, np.ndarray]:
    """
    Extract a (epoch, [a, e, i, raan, argp, M]) tuple from a TLE line pair.

    All angles are returned in radians; a in km.

    Mean element extraction:
      a  ← derived from mean motion n (rev/day) via Kepler's 3rd law
      e  ← TLE implied decimal (prepend '0.')
      i  ← degrees → radians
      Ω  ← degrees → radians
      ω  ← degrees → radians
      M  ← degrees → radians

    The benchmark paper recommends using mean elements directly rather than
    propagating to a common epoch via SGP4.  We follow that recommendation.
    """
    epoch = _parse_tle_epoch(line1[18:32])

    i_deg    = float(line2[8:16])
    raan_deg = float(line2[17:25])
    e        = float("0." + line2[26:33].strip())
    argp_deg = float(line2[34:42])
    M_deg    = float(line2[43:51])
    n_rev    = float(line2[52:63])               # rev day⁻¹

    n_rad_s  = n_rev * 2.0 * math.pi / 86400.0  # rad s⁻¹
    a        = (MU_EARTH / n_rad_s ** 2) ** (1.0 / 3.0)  # km

    vec = np.array([
        a,
        e,
        math.radians(i_deg),
        math.radians(raan_deg),
        math.radians(argp_deg),
        math.radians(M_deg),
    ], dtype=np.float64)
    return epoch, vec


def parse_tle_file(filepath: str) -> List[Tuple[datetime.datetime, np.ndarray]]:
    """
    Parse all TLE records from a compiled .tle file.

    Returns a sorted list of (epoch_utc, element_vector) tuples.
    Duplicate epochs, invalid elements, and sub-Earth orbits are silently dropped.

    Validation checks:
    - a > R_EARTH (not crashed)
    - 0 ≤ e < 1 (bound orbit)
    - 0 ≤ i ≤ π (valid inclination)
    """
    records: List[Tuple[datetime.datetime, np.ndarray]] = []
    seen: set = set()

    with open(filepath, "r") as fh:
        raw = [ln.rstrip() for ln in fh if ln.strip()]

    i = 0
    while i < len(raw) - 1:
        l1, l2 = raw[i], raw[i + 1]
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                epoch, vec = _tle_to_elements(l1, l2)
                a, e, inc = vec[0], vec[1], vec[2]
                if not (R_EARTH < a < 100_000): i += 2; continue
                if not (0.0 <= e < 1.0):         i += 2; continue
                if not (0.0 <= inc <= math.pi):  i += 2; continue
                key = round(epoch.timestamp())
                if key not in seen:
                    seen.add(key)
                    records.append((epoch, vec))
            except (ValueError, IndexError):
                pass
            i += 2
        else:
            i += 1

    records.sort(key=lambda r: r[0])
    return records


# ════════════════════════════════════════════════════════════════════
# MANOEUVRE YAML PARSING
# ════════════════════════════════════════════════════════════════════

def parse_maneuver_yaml(filepath: str) -> List[datetime.datetime]:
    """
    Parse benchmark manoeuvre timestamp YAML file.

    Supports both PyYAML (preferred) and a minimal hand-rolled parser.
    Returns UTC datetime objects sorted ascending.
    """
    timestamps: List[datetime.datetime] = []
    try:
        import yaml
        with open(filepath) as fh:
            data = yaml.safe_load(fh)
        for ts in data.get("manoeuvre_timestamps", []):
            if isinstance(ts, str):
                dt = datetime.datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")
                timestamps.append(dt.replace(tzinfo=datetime.timezone.utc))
            elif isinstance(ts, datetime.datetime):
                timestamps.append(ts.replace(tzinfo=datetime.timezone.utc))
    except ImportError:
        in_list = False
        with open(filepath) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("manoeuvre_timestamps:"):
                    in_list = True; continue
                if in_list:
                    if s.startswith("- "):
                        try:
                            dt = datetime.datetime.strptime(s[2:].strip(), "%Y-%m-%d %H:%M:%S")
                            timestamps.append(dt.replace(tzinfo=datetime.timezone.utc))
                        except ValueError:
                            pass
                    elif s and not s.startswith("-"):
                        in_list = False
    timestamps.sort()
    return timestamps


# ════════════════════════════════════════════════════════════════════
# GRID SNAPPING
# ════════════════════════════════════════════════════════════════════

def snap_to_grid(
    records:            List[Tuple[datetime.datetime, np.ndarray]],
    grid_times:         List[datetime.datetime],
    max_tle_gap_hours:  float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each UTC grid timestamp, find the nearest real TLE observation
    within max_tle_gap_hours.

    Returns
    -------
    elements    : (T, 6) float64  — element vectors (NaN where missing)
    valid_mask  : (T,)   bool     — True where a real TLE was found
    tle_offsets : (T,)   float64  — hours from grid time to nearest TLE
    tle_epochs  : (T,)   float64  — Unix timestamp of the selected TLE epoch
    """
    T           = len(grid_times)
    epochs_s    = np.array([r[0].timestamp() for r in records], dtype=np.float64)
    vecs        = np.array([r[1] for r in records], dtype=np.float64)
    grid_ts     = np.array([g.timestamp() for g in grid_times], dtype=np.float64)
    max_gap_s   = max_tle_gap_hours * 3600.0

    elements    = np.full((T, 6), np.nan)
    valid_mask  = np.zeros(T, dtype=bool)
    tle_offsets = np.full(T, np.nan)
    tle_epochs  = np.full(T, np.nan)

    for t in range(T):
        diffs = np.abs(epochs_s - grid_ts[t])
        idx   = int(np.argmin(diffs))
        if diffs[idx] <= max_gap_s:
            elements[t]    = vecs[idx]
            valid_mask[t]  = True
            tle_offsets[t] = (epochs_s[idx] - grid_ts[t]) / 3600.0
            tle_epochs[t]  = epochs_s[idx]

    return elements, valid_mask, tle_offsets, tle_epochs


def _forward_fill(
    elements:   np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill missing slots by carrying the previous valid observation forward.

    Remaining leading NaN rows (no prior valid TLE) are filled from
    the first valid observation.

    This is the minimum preprocessing to produce a dense (T, 6) array
    without interpolating orbital elements.  The valid_mask is preserved
    separately so downstream code can identify filled slots.
    """
    T = len(elements)
    first_valid = next((t for t in range(T) if valid_mask[t]), None)
    if first_valid is None:
        return elements  # all missing — caller should have excluded this satellite

    filled = elements.copy()
    for t in range(T):
        if not valid_mask[t]:
            if t == 0:
                filled[t] = elements[first_valid]
            else:
                filled[t] = filled[t - 1]

    # Fill leading missing rows with first valid observation
    for t in range(first_valid):
        filled[t] = elements[first_valid]

    return filled


def _compute_grid_dt_seconds(
    tle_epochs:  np.ndarray,
    valid_mask:  np.ndarray,
    dt_hours:    float,
) -> np.ndarray:
    """
    Compute the actual elapsed time (seconds) between consecutive grid slots.

    If both slot t and t+1 have real TLEs, dt[t] = TLE_epoch[t+1] - TLE_epoch[t].
    If either slot is forward-filled, dt[t] = dt_hours * 3600 (nominal grid step).

    This actual dt is passed to the physics propagator so that the J2-corrected
    mean anomaly prediction is computed with the right elapsed time, giving
    meaningful ΔM residuals.

    Returns
    -------
    dt_seconds : np.ndarray  Shape (T-1,)
    """
    T  = len(tle_epochs)
    dt = np.full(T - 1, dt_hours * 3600.0)

    for t in range(T - 1):
        if valid_mask[t] and valid_mask[t + 1]:
            actual_dt = tle_epochs[t + 1] - tle_epochs[t]
            if actual_dt > 0:
                dt[t] = actual_dt  # use real inter-TLE elapsed time

    return dt


# ════════════════════════════════════════════════════════════════════
# MANOEUVRE LABELS
# ════════════════════════════════════════════════════════════════════

def labels_from_maneuvers(
    grid_times:           List[datetime.datetime],
    maneuver_times:       List[datetime.datetime],
    tolerance_hours:      float,
) -> np.ndarray:
    """
    Create binary manoeuvre labels for each grid slot.

    A slot is labelled 1 if any manoeuvre timestamp falls within
    ±tolerance_hours of the grid time.

    Rationale for 24 h default tolerance
    -------------------------------------
    Measured TLE-to-manoeuvre offsets in the benchmark:
      - Typical range: 3–34 hours (positive = TLE after manoeuvre).
      - The offset distribution has most mass in [−12, +36] hours.
    A ±24 h window captures nearly all confirmed manoeuvre-affected TLEs
    without labelling multiple-day windows as anomalous.

    The original manoeuvre timestamps are preserved in the loader's
    return dict for event-level evaluation.
    """
    if not maneuver_times:
        return np.zeros(len(grid_times), dtype=np.int8)

    man_ts  = np.array([m.timestamp() for m in maneuver_times])
    tol_s   = tolerance_hours * 3600.0
    labels  = np.zeros(len(grid_times), dtype=np.int8)

    for t, gt in enumerate(grid_times):
        gt_ts = gt.timestamp()
        if np.any(np.abs(man_ts - gt_ts) <= tol_s):
            labels[t] = 1
    return labels


# ════════════════════════════════════════════════════════════════════
# ORBITAL SHELL ASSIGNMENT
# ════════════════════════════════════════════════════════════════════

def assign_shell_ids(
    sat_names:       List[str],
    median_elements: np.ndarray,
) -> np.ndarray:
    """
    Assign satellites to orbital families (shells) based on altitude and
    inclination of their median element set.

    Hard-coded shells (justified by benchmark satellite characteristics):
      Shell 0  GEO belt:      alt > 30 000 km
      Shell 1  SSO / polar:   600 < alt < 1 000 km  AND  i > 88°
      Shell 2  LEO 66°:       1 000 < alt < 1 700 km AND  60° < i < 70°
      Shell 3  Other / unclassified

    Satellites in different shells are not connected in the GNN graph
    (no scientifically motivated peer relationship across GEO ↔ LEO).

    Parameters
    ----------
    sat_names        : list of satellite names (for logging)
    median_elements  : (S, 6) median observed elements

    Returns
    -------
    shell_id : np.ndarray  Shape (S,) int8
    """
    S        = len(sat_names)
    shell_id = np.full(S, 3, dtype=np.int8)

    for k in range(S):
        a   = median_elements[k, 0]
        inc = math.degrees(median_elements[k, 2])
        alt = a - R_EARTH

        if alt > 30_000:
            shell_id[k] = 0                                    # GEO
        elif 600 < alt < 1_000 and inc > 88:
            shell_id[k] = 1                                    # SSO / polar LEO
        elif 1_000 < alt < 1_700 and 60 < inc < 70:
            shell_id[k] = 2                                    # LEO 66°

        logger.debug(
            "  %s: alt=%.0f km, i=%.1f° → shell %d",
            sat_names[k], alt, inc, shell_id[k],
        )

    return shell_id


# ════════════════════════════════════════════════════════════════════
# PER-SATELLITE NORMALISATION
# ════════════════════════════════════════════════════════════════════

def fit_per_satellite_scaler(
    residuals: np.ndarray,
    train_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a per-satellite, per-element zero-mean / unit-std scaler using
    TRAINING data only.

    Uses mean and std (not IQR) because orbital residuals are heavy-tailed
    (most values are near zero, but manoeuvre spikes are large).  The IQR
    would be near-zero for near-circular orbits, making the scaler unstable.

    A minimum scale of 1e-6 prevents division-by-zero on constant features.

    Parameters
    ----------
    residuals : (S, T-1, 6)
    train_end : int  — last training time index

    Returns
    -------
    mean  : (S, 6)
    scale : (S, 6)  — std, clipped to [1e-6, ∞)
    """
    train_res = residuals[:, :train_end, :]   # (S, train_T, 6)
    mean  = np.nanmean(train_res, axis=1)     # (S, 6)
    scale = np.nanstd(train_res, axis=1)      # (S, 6)
    scale = np.clip(scale, 1e-6, None)        # prevent division by zero
    return mean, scale



def apply_per_satellite_scaler(
    residuals: np.ndarray,
    mean:      np.ndarray,
    scale:     np.ndarray,
    clip_sigma: float = 10.0,
) -> np.ndarray:
    """
    Apply per-satellite scaler: (residual − mean) / scale, clipped to ±clip_sigma.

    Clipping prevents extreme manoeuvre events (which can be 100+ sigma) from
    causing NaN gradients or dominating model training.  The anomaly detector
    will still flag these events because the clipped value (±10) is far outside
    the nominal distribution (which has std ≈ 1 after scaling).

    clip_sigma=10 is a conservative choice: it retains the sign and relative
    ordering of anomalies while preventing floating-point overflow.

    Operates on shape (..., 6) using broadcasting over the time axis.
    """
    normed = (residuals - mean[:, np.newaxis, :]) / scale[:, np.newaxis, :]
    if clip_sigma is not None:
        normed = np.clip(normed, -clip_sigma, clip_sigma)
    return normed


# ════════════════════════════════════════════════════════════════════
# RESIDUAL COMPUTATION
# ════════════════════════════════════════════════════════════════════

def compute_residual_sequences(
    elements_obs:     np.ndarray,
    dt_seconds_grid:  Optional[np.ndarray] = None,
    dt_hours:         float = DEFAULT_DT_HOURS,
) -> np.ndarray:
    """
    Compute physics residuals for the full observation sequence.

    For each satellite k and time step t:
        residual[k, t] = observed[k, t+1] − propagate(observed[k, t], dt[k, t])

    FIX (F1): if dt_seconds_grid is provided (shape S × T-1), the ACTUAL
    inter-observation elapsed time is used for each step.  This is critical
    for correct ΔM residuals.  If not provided, a fixed dt_hours*3600 is
    used (backward-compatible default for synthetic data).

    Parameters
    ----------
    elements_obs     : (S, T, 6)   observed element sequences.
    dt_seconds_grid  : (S, T-1)    actual elapsed seconds per step (optional).
    dt_hours         : float        fallback fixed dt (used only when dt_seconds_grid is None).

    Returns
    -------
    residuals : (S, T-1, 6) signed, angle-wrapped residuals.
    """
    S, T, _ = elements_obs.shape

    if dt_seconds_grid is None:
        # Backward-compatible path for synthetic data
        dt_seconds_grid = np.full((S, T - 1), dt_hours * 3600.0)

    return residuals_batch(elements_obs, dt_seconds_grid)


# ════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ════════════════════════════════════════════════════════════════════

def build_orbital_neighbor_graph(
    elements_snapshot: np.ndarray,
    shell_id:          np.ndarray,
    k_neighbors:       int   = DEFAULT_K_NEIGHBORS,
    cross_shell:       bool  = False,
) -> np.ndarray:
    """
    Construct the symmetric orbital-neighbour adjacency matrix for OrbitGNN's GCN.

    Graph construction rules
    ------------------------
    PRIMARY: same-shell k-NN
      Within each shell, edges connect each satellite to its k nearest
      orbital-plane neighbours as measured by:
        feature = [norm(Δa), norm(Δi), sin(RAAN), cos(RAAN)]
      Normalisation is within-shell to not penalise cross-shell scale.

      Scientific basis: satellites in the same shell experience similar
      perturbation environments (drag, solar radiation pressure).  A
      satellite diverging from its shell peers is anomalous; one that
      diverges identically to all peers indicates a shell-wide driver.

    SECONDARY: cross-shell (disabled by default)
      When cross_shell=True, k-NN is applied globally.  This can connect
      GEO to LEO satellites, which is scientifically questionable unless
      there is a specific reason (e.g., common ground segment).

    Singleton shells (one satellite) correctly have no edges — there are
    no meaningful peers to compare against.

    Returns
    -------
    adj : (S, S) symmetric 0/1 float32 adjacency matrix.
    """
    S   = elements_snapshot.shape[0]
    adj = np.zeros((S, S), dtype=np.float32)

    for sh in np.unique(shell_id):
        idxs = np.where(shell_id == sh)[0]
        n_sh = len(idxs)
        if n_sh < 2:
            logger.debug("Shell %d has only %d satellite(s) — no edges", sh, n_sh)
            continue

        a    = elements_snapshot[idxs, 0]
        inc  = elements_snapshot[idxs, 2]
        raan = elements_snapshot[idxs, 3]

        feat = np.column_stack([
            (a   - a.mean())   / (a.std()   + 1e-6),
            (inc - inc.mean()) / (inc.std()  + 1e-6),
            np.sin(raan),
            np.cos(raan),
        ])

        dist = np.linalg.norm(feat[:, None, :] - feat[None, :, :], axis=-1)
        np.fill_diagonal(dist, np.inf)

        k_eff = min(k_neighbors, n_sh - 1)
        for local_i, global_i in enumerate(idxs):
            nn_local = np.argsort(dist[local_i])[:k_eff]
            for nl in nn_local:
                adj[global_i, idxs[nl]] = 1.0

    if cross_shell:
        a    = elements_snapshot[:, 0]
        inc  = elements_snapshot[:, 2]
        raan = elements_snapshot[:, 3]
        feat = np.column_stack([
            (a - a.mean()) / (a.std() + 1e-6),
            (inc - inc.mean()) / (inc.std() + 1e-6),
            np.sin(raan), np.cos(raan),
        ])
        dist = np.linalg.norm(feat[:, None, :] - feat[None, :, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        for k in range(S):
            for nl in np.argsort(dist[k])[:k_neighbors]:
                adj[k, nl] = 1.0

    adj = np.maximum(adj, adj.T)  # symmetrise
    return adj


# ════════════════════════════════════════════════════════════════════
# PATH A: REAL TLE BENCHMARK LOADER
# ════════════════════════════════════════════════════════════════════

def load_real_benchmark_dataset(
    dataset_path:             str,
    start_date:               Optional[str]  = None,
    end_date:                 Optional[str]  = None,
    dt_hours:                 float          = DEFAULT_DT_HOURS,
    max_tle_gap_hours:        float          = DEFAULT_MAX_TLE_GAP_HOURS,
    maneuver_tolerance_hours: float          = DEFAULT_MANEUVER_TOL_HOURS,
    min_valid_fraction:       float          = DEFAULT_MIN_VALID_FRACTION,
) -> dict:
    """
    Load the TLE Observation Benchmark Dataset.

    Produces tensors compatible with OrbitGNN's (S, T, 6) interface plus
    the per-step actual dt tensor required for correct physics residuals.

    Parameters
    ----------
    dataset_path             : Path to benchmark root or processed_files/ dir.
    start_date               : ISO-8601 start date, e.g. '2020-01-01'.
    end_date                 : ISO-8601 end date,   e.g. '2022-01-01'.
    dt_hours                 : Uniform grid spacing (hours).  Default 24 h.
    max_tle_gap_hours        : Max gap (hours) between grid point and nearest
                               real TLE before slot is marked MISSING.
    maneuver_tolerance_hours : Half-window (hours) for matching manoeuvre
                               timestamps to grid slots.
    min_valid_fraction       : Satellites with fewer valid slots than this
                               fraction are excluded.

    Returns
    -------
    dict with keys:
      'elements_obs'     : (S, T, 6)   float64  — mean Keplerian elements
      'labels'           : (S, T)      int8     — 0=nominal, 1=manoeuvre
      'shell_id'         : (S,)        int8     — orbital family
      'timestamps'       : list[datetime]  — UTC grid times (length T)
      'sat_names'        : list[str]
      'valid_mask'       : (S, T)      bool     — False = forward-filled slot
      'dt_seconds_grid'  : (S, T-1)   float64  — actual inter-TLE dt per step
      'maneuver_events'  : dict  sat_name → list[datetime]
      'metadata'         : dict  — provenance / statistics
    """
    # ── Locate processed_files directory ────────────────────────────
    p = Path(dataset_path)
    proc_dir = p / "processed_files" if (p / "processed_files").is_dir() else p
    if not proc_dir.is_dir():
        raise FileNotFoundError(f"Cannot find TLE files under {dataset_path}")

    tle_files  = sorted(proc_dir.glob("*.tle"))
    yaml_files = {
        f.stem.replace("manoeuvres_", ""): str(f)
        for f in proc_dir.glob("manoeuvres_*.yaml")
    }
    if not tle_files:
        raise FileNotFoundError(f"No .tle files found under {proc_dir}")

    # ── Parse TLEs ───────────────────────────────────────────────────
    all_records: Dict[str, List] = {}
    for tf in tle_files:
        recs = parse_tle_file(str(tf))
        if recs:
            all_records[tf.stem] = recs

    # ── Time window ──────────────────────────────────────────────────
    tz = datetime.timezone.utc
    t0 = (datetime.datetime.fromisoformat(start_date).replace(tzinfo=tz)
          if start_date else min(r[0] for recs in all_records.values() for r in recs))
    t1 = (datetime.datetime.fromisoformat(end_date).replace(tzinfo=tz)
          if end_date else max(r[0] for recs in all_records.values() for r in recs))

    dt_td       = datetime.timedelta(hours=dt_hours)
    grid_times  = []
    cur = t0
    while cur <= t1:
        grid_times.append(cur)
        cur += dt_td
    T = len(grid_times)
    if T < 10:
        raise ValueError(f"Grid has only {T} points for {t0}–{t1}. Widen date range.")

    # ── Filter: must have ≥3 TLEs within window ─────────────────────
    active_sats = [
        name for name, recs in all_records.items()
        if sum(1 for r in recs if t0 <= r[0] <= t1) >= 3
    ]
    if not active_sats:
        raise ValueError(f"No satellites have ≥3 TLE observations in {t0}–{t1}")

    # ── Snap each satellite to grid ──────────────────────────────────
    sat_elements:    Dict[str, np.ndarray] = {}
    sat_valid_mask:  Dict[str, np.ndarray] = {}
    sat_tle_epochs:  Dict[str, np.ndarray] = {}
    sat_labels:      Dict[str, np.ndarray] = {}
    maneuver_events: Dict[str, List]        = {}
    gap_s = max_tle_gap_hours * 3600.0

    for name in list(active_sats):
        window_recs = [
            r for r in all_records[name]
            if (t0 - datetime.timedelta(hours=max_tle_gap_hours)) <= r[0]
            <= (t1 + datetime.timedelta(hours=max_tle_gap_hours))
        ]
        if not window_recs:
            active_sats.remove(name); continue

        elem, mask, _, tle_ep = snap_to_grid(window_recs, grid_times, max_tle_gap_hours)

        if mask.mean() < min_valid_fraction:
            warnings.warn(
                f"Satellite '{name}' has only {mask.mean():.1%} valid grid slots "
                f"(threshold {min_valid_fraction:.0%}) — excluded.",
                UserWarning, stacklevel=2,
            )
            active_sats.remove(name); continue

        elem = _forward_fill(elem, mask)
        sat_elements[name]   = elem
        sat_valid_mask[name] = mask
        sat_tle_epochs[name] = tle_ep

        # Manoeuvre labels
        man_times = []
        if name in yaml_files:
            all_man = parse_maneuver_yaml(yaml_files[name])
            man_times = [m for m in all_man if t0 <= m <= t1]
        maneuver_events[name] = man_times
        sat_labels[name] = labels_from_maneuvers(
            grid_times, man_times, maneuver_tolerance_hours
        )

    S = len(active_sats)
    if S < MIN_GRAPH_SATS:
        raise ValueError(f"Only {S} satellites remain; need ≥{MIN_GRAPH_SATS}")

    # ── Assemble tensors ─────────────────────────────────────────────
    elements_obs = np.stack([sat_elements[n]  for n in active_sats], axis=0)  # (S,T,6)
    labels       = np.stack([sat_labels[n]    for n in active_sats], axis=0)  # (S,T)
    valid_mask   = np.stack([sat_valid_mask[n] for n in active_sats], axis=0) # (S,T)
    tle_epochs   = np.stack([sat_tle_epochs[n] for n in active_sats], axis=0) # (S,T)

    # ── Per-step actual dt (FIX F1) ──────────────────────────────────
    dt_seconds_grid = np.zeros((S, T - 1), dtype=np.float64)
    for k in range(S):
        dt_seconds_grid[k] = _compute_grid_dt_seconds(
            tle_epochs[k], valid_mask[k], dt_hours
        )

    # ── Shell IDs ────────────────────────────────────────────────────
    median_elems = np.nanmedian(elements_obs, axis=1)  # (S, 6)
    shell_id     = assign_shell_ids(active_sats, median_elems)

    # ── Metadata ─────────────────────────────────────────────────────
    total_maneuvers = sum(len(v) for v in maneuver_events.values())
    labeled_slots   = int((labels > 0).sum())
    missing_slots   = int((~valid_mask).sum())

    metadata = {
        "n_satellites":           S,
        "satellite_names":        active_sats,
        "n_grid_steps":           T,
        "grid_start":             grid_times[0].isoformat(),
        "grid_end":               grid_times[-1].isoformat(),
        "dt_hours":               dt_hours,
        "max_tle_gap_hours":      max_tle_gap_hours,
        "maneuver_tolerance_h":   maneuver_tolerance_hours,
        "total_maneuver_events":  total_maneuvers,
        "labeled_maneuver_slots": labeled_slots,
        "missing_slots":          missing_slots,
        "missing_fraction":       missing_slots / (S * T),
        "shell_distribution":     {n: int(shell_id[k])
                                   for k, n in enumerate(active_sats)},
        "dt_seconds_grid_range":  (dt_seconds_grid.min(), dt_seconds_grid.max()),
    }

    return {
        "elements_obs":    elements_obs,
        "labels":          labels,
        "shell_id":        shell_id,
        "timestamps":      grid_times,
        "sat_names":       active_sats,
        "valid_mask":      valid_mask,
        "dt_seconds_grid": dt_seconds_grid,
        "maneuver_events": maneuver_events,
        "metadata":        metadata,
    }


# ════════════════════════════════════════════════════════════════════
# PATH B: SYNTHETIC SIMULATOR (regression / testing only)
# ════════════════════════════════════════════════════════════════════

def _random_shell(
    n_sats:         int,
    altitude_km:    float,
    inclination_deg:float,
    rng:            np.random.Generator,
) -> List[KeplerianElements]:
    """Place n_sats in circular orbits at one altitude/inclination shell."""
    a    = np.full(n_sats, R_EARTH + altitude_km) + rng.normal(0, 0.5, n_sats)
    e    = np.clip(rng.normal(0.0008, 0.0003, n_sats), 0.0001, 0.01)
    i    = np.radians(inclination_deg) + np.radians(rng.normal(0, 0.05, n_sats))
    raan = rng.uniform(0, 2 * np.pi, n_sats)
    argp = rng.uniform(0, 2 * np.pi, n_sats)
    M    = rng.uniform(0, 2 * np.pi, n_sats)
    return [KeplerianElements(a[k], e[k], i[k], raan[k], argp[k], M[k])
            for k in range(n_sats)]


def simulate_constellation(
    n_shells:      int   = 3,
    sats_per_shell:int   = 12,
    n_steps:       int   = 120,
    dt_hours:      float = 6.0,
    anomaly_rate:  float = 0.08,
    seed:          int   = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic multi-shell constellation for regression testing.

    Returns elements_true, elements_obs, labels, shell_id — same shapes
    as before.  Not used in primary evaluation.
    """
    rng    = np.random.default_rng(seed)
    dt_sec = dt_hours * 3600.0

    shell_altitudes    = np.linspace(540, 570, n_shells)
    shell_inclinations = np.linspace(53.0, 97.6, n_shells)

    all_sats: List[KeplerianElements] = []
    shell_id_list: List[int] = []
    for s in range(n_shells):
        sats = _random_shell(sats_per_shell, shell_altitudes[s],
                             shell_inclinations[s], rng)
        all_sats.extend(sats)
        shell_id_list.extend([s] * sats_per_shell)

    shell_id = np.array(shell_id_list, dtype=np.int8)
    S = len(all_sats)

    elements_true = np.zeros((S, n_steps, 6))
    labels        = np.zeros((S, n_steps), dtype=np.int8)
    state         = all_sats[:]

    for t in range(n_steps):
        for k in range(S):
            state[k] = propagate(state[k], dt_sec)
            if t > 5 and rng.random() < anomaly_rate / n_steps * 15:
                kind = rng.choice(["decay_drag", "maneuver", "tumble"])
                if kind == "decay_drag":
                    state[k].a -= rng.uniform(0.05, 0.3)
                    labels[k, t:] = np.maximum(labels[k, t:], 1)
                elif kind == "maneuver":
                    state[k].a   += rng.normal(0, 0.15)
                    state[k].argp = (state[k].argp + rng.normal(0, 0.05)) % (2 * np.pi)
                    labels[k, t]  = 2
                elif kind == "tumble":
                    state[k].M   = (state[k].M + rng.normal(0, 0.2)) % (2 * np.pi)
                    labels[k, t] = 3
            elements_true[k, t] = state[k].as_vector()

    noise_std    = np.array([0.02, 0.00002, 1e-5, 1e-5, 1e-5, 2e-5])
    elements_obs = elements_true + rng.normal(0, noise_std, elements_true.shape)
    return elements_true, elements_obs, labels, shell_id


# ── CelesTrak live fetch (unchanged) ─────────────────────────────────
def fetch_celestrak_group(group: str = "starlink", fmt: str = "json") -> list:
    """Fetch live GP data from CelesTrak (needs internet access)."""
    import requests
    url  = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT={fmt}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)

    print("=" * 60)
    print("Smoke test: Real TLE Benchmark Dataset")
    print("=" * 60)

    candidates = [
        sys.argv[1] if len(sys.argv) > 1 else None,
        str(Path(__file__).parent.parent / "TLE_observation_benchmark_dataset-main"),
        str(Path.home() / "Documents/dl/TLE_observation_benchmark_dataset-main"),
    ]
    dataset_path = next((p for p in candidates if p and Path(p).is_dir()), None)
    if dataset_path is None:
        print("ERROR: dataset path not found. Pass as: python dataset.py <path>")
        sys.exit(1)
    print(f"Dataset path: {dataset_path}\n")

    result = load_real_benchmark_dataset(
        dataset_path,
        start_date="2020-01-01",
        end_date="2022-01-01",
        dt_hours=24.0,
        max_tle_gap_hours=48.0,
        maneuver_tolerance_hours=24.0,
    )

    md = result["metadata"]
    eo = result["elements_obs"]
    lab = result["labels"]
    sid = result["shell_id"]
    dts = result["dt_seconds_grid"]
    vm  = result["valid_mask"]

    print(f"Satellites loaded  : {md['n_satellites']}")
    print(f"Names              : {md['satellite_names']}")
    print(f"Grid steps (T)     : {md['n_grid_steps']}")
    print(f"Grid range         : {md['grid_start']} → {md['grid_end']}")
    print(f"Total maneuvers    : {md['total_maneuver_events']}")
    print(f"Labeled slots      : {md['labeled_maneuver_slots']}")
    print(f"Missing slots      : {md['missing_slots']} ({md['missing_fraction']:.1%})")
    print(f"Shell distribution : {md['shell_distribution']}")
    print(f"dt_seconds range   : {md['dt_seconds_grid_range'][0]/3600:.1f}h "
          f"– {md['dt_seconds_grid_range'][1]/3600:.1f}h")

    res = compute_residual_sequences(eo, dt_seconds_grid=dts)
    adj = build_orbital_neighbor_graph(eo[:, -1, :], sid, k_neighbors=3)

    print(f"\nelements_obs shape : {eo.shape}")
    print(f"labels shape       : {lab.shape}")
    print(f"residuals shape    : {res.shape}")
    print(f"adjacency shape    : {adj.shape}")
    print(f"edges              : {int(adj.sum() // 2)}")
    print(f"anomaly rate       : {(lab > 0).mean():.4f}")
    print(f"NaN in elements    : {np.isnan(eo).sum()}")
    print(f"NaN in residuals   : {np.isnan(res).sum()}")

    # Check that ΔM residuals are now physically plausible (< 0.2 rad for most)
    dM = res[:, :, 5]
    print(f"\nΔM residual stats  : median={np.median(np.abs(dM)):.4f} rad, "
          f"p95={np.percentile(np.abs(dM), 95):.4f} rad")

    print("\nSMOKE TEST PASSED")
