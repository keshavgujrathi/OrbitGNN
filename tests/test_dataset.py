"""
tests/test_dataset.py
---------------------
Unit tests for dataset.py: TLE parsing, grid snapping, maneuver labels,
graph construction, tensor shapes, and data integrity checks.
"""

import datetime
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from physics import R_EARTH, MU_EARTH
from dataset import (
    _parse_tle_epoch,
    _tle_to_elements,
    parse_tle_file,
    parse_maneuver_yaml,
    snap_to_grid,
    _forward_fill,
    _compute_grid_dt_seconds,
    labels_from_maneuvers,
    assign_shell_ids,
    build_orbital_neighbor_graph,
    compute_residual_sequences,
    fit_per_satellite_scaler,
    apply_per_satellite_scaler,
    simulate_constellation,
    DEFAULT_DT_HOURS,
)

UTC = datetime.timezone.utc
DAY = datetime.timedelta(days=1)


# ════════════════════════════════════════════════════════════════════
# TLE epoch parsing
# ════════════════════════════════════════════════════════════════════

class TestParseTleEpoch:
    def test_year_2000_day_1(self):
        ep = _parse_tle_epoch("00001.00000000")
        assert ep.year == 2000 and ep.month == 1 and ep.day == 1

    def test_year_2020(self):
        ep = _parse_tle_epoch("20001.50000000")
        assert ep.year == 2020
        assert ep.month == 1 and ep.day == 1
        # 0.5 day = noon
        assert ep.hour == 12

    def test_year_1999_pre_2000(self):
        ep = _parse_tle_epoch("99001.00000000")
        assert ep.year == 1999

    def test_utc_aware(self):
        ep = _parse_tle_epoch("21001.00000000")
        assert ep.tzinfo is not None


# ════════════════════════════════════════════════════════════════════
# TLE element extraction
# ════════════════════════════════════════════════════════════════════

JASON3_LINE1 = "1 41240U 16002A   21001.50000000  .00000000  00000-0  00000+0 0  9999"
JASON3_LINE2 = "2 41240  66.0416 313.6082 0009500 289.8740 070.1284 12.80831803 99999"

class TestTleToElements:
    def test_returns_epoch_and_vec(self):
        ep, vec = _tle_to_elements(JASON3_LINE1, JASON3_LINE2)
        assert isinstance(ep, datetime.datetime)
        assert vec.shape == (6,)

    def test_a_in_leo_range(self):
        _, vec = _tle_to_elements(JASON3_LINE1, JASON3_LINE2)
        a = vec[0]
        assert 7_000 < a < 8_000, f"Jason-3 a should be ~7716 km, got {a:.1f}"

    def test_e_low(self):
        _, vec = _tle_to_elements(JASON3_LINE1, JASON3_LINE2)
        e = vec[1]
        assert 0 < e < 0.01, f"Jason-3 e should be near-circular, got {e:.5f}"

    def test_inclination_radians(self):
        _, vec = _tle_to_elements(JASON3_LINE1, JASON3_LINE2)
        i_deg = math.degrees(vec[2])
        assert 65 < i_deg < 67, f"Jason-3 inclination should be ~66°, got {i_deg:.2f}"

    def test_mean_anomaly_in_range(self):
        _, vec = _tle_to_elements(JASON3_LINE1, JASON3_LINE2)
        M = vec[5]
        assert 0 <= M <= 2 * math.pi


# ════════════════════════════════════════════════════════════════════
# TLE file parsing
# ════════════════════════════════════════════════════════════════════

class TestParseTleFile:
    def _write_tle_file(self, lines, tmp_path=None):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False)
        f.write("\n".join(lines))
        f.close()
        return f.name

    def test_basic_parse(self):
        fname = self._write_tle_file([JASON3_LINE1, JASON3_LINE2])
        recs  = parse_tle_file(fname)
        os.unlink(fname)
        assert len(recs) == 1
        assert recs[0][1].shape == (6,)

    def test_multiple_records_sorted(self):
        # Two records: day 10 then day 5 — should be sorted by epoch
        l1a = JASON3_LINE1.replace("21001", "21010")
        l1b = JASON3_LINE1.replace("21001", "21005")
        fname = self._write_tle_file([l1a, JASON3_LINE2, l1b, JASON3_LINE2])
        recs  = parse_tle_file(fname)
        os.unlink(fname)
        assert len(recs) == 2
        assert recs[0][0] < recs[1][0]  # sorted ascending

    def test_duplicate_epochs_deduplicated(self):
        fname = self._write_tle_file([JASON3_LINE1, JASON3_LINE2,
                                      JASON3_LINE1, JASON3_LINE2])
        recs  = parse_tle_file(fname)
        os.unlink(fname)
        assert len(recs) == 1


# ════════════════════════════════════════════════════════════════════
# Maneuver YAML parsing
# ════════════════════════════════════════════════════════════════════

class TestParseManeuverYaml:
    def _write_yaml(self, content):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_basic_yaml(self):
        fname = self._write_yaml(
            "manoeuvre_timestamps:\n"
            "- 2020-03-12 09:11:00\n"
            "- 2020-06-18 10:13:00\n"
        )
        times = parse_maneuver_yaml(fname)
        os.unlink(fname)
        assert len(times) == 2
        assert times[0].year == 2020
        assert times[0].month == 3

    def test_empty_yaml(self):
        fname = self._write_yaml("manoeuvre_timestamps: []\n")
        times = parse_maneuver_yaml(fname)
        os.unlink(fname)
        assert len(times) == 0

    def test_timestamps_utc_aware(self):
        fname = self._write_yaml(
            "manoeuvre_timestamps:\n- 2021-01-01 00:00:00\n"
        )
        times = parse_maneuver_yaml(fname)
        os.unlink(fname)
        assert times[0].tzinfo is not None

    def test_timestamps_sorted(self):
        fname = self._write_yaml(
            "manoeuvre_timestamps:\n"
            "- 2021-06-01 00:00:00\n"
            "- 2021-01-01 00:00:00\n"
        )
        times = parse_maneuver_yaml(fname)
        os.unlink(fname)
        assert times[0] < times[1]


# ════════════════════════════════════════════════════════════════════
# Grid snapping
# ════════════════════════════════════════════════════════════════════

class TestSnapToGrid:
    def _make_records(self, start, n, gap_hours=24.0):
        """Create synthetic TLE records at uniform intervals."""
        recs = []
        for i in range(n):
            ep  = start + datetime.timedelta(hours=gap_hours * i)
            vec = np.array([7000 + i * 0.001, 0.001, 1.7, 1.0, 0.5, 0.3])
            recs.append((ep, vec))
        return recs

    def test_exact_match(self):
        start = datetime.datetime(2021, 1, 1, tzinfo=UTC)
        grid  = [start + datetime.timedelta(days=i) for i in range(5)]
        recs  = self._make_records(start, 5)
        elem, mask, _, _ = snap_to_grid(recs, grid, max_tle_gap_hours=2.0)
        assert mask.all(), "All slots should be valid for exact match"
        assert elem.shape == (5, 6)

    def test_gap_too_large_marked_missing(self):
        start = datetime.datetime(2021, 1, 1, tzinfo=UTC)
        grid  = [start + datetime.timedelta(days=i) for i in range(5)]
        # Records only for days 0 and 4; max_gap=12h means days 1,2,3 are missing
        # (nearest record is 24h away, beyond the 12h threshold)
        recs  = [
            (start, np.ones(6)),
            (start + datetime.timedelta(days=4), np.ones(6) * 2),
        ]
        elem, mask, _, _ = snap_to_grid(recs, grid, max_tle_gap_hours=12.0)
        assert mask[0] and mask[4]
        assert not mask[1] and not mask[2] and not mask[3]

    def test_no_nan_in_valid_slots(self):
        start = datetime.datetime(2021, 1, 1, tzinfo=UTC)
        grid  = [start + datetime.timedelta(days=i) for i in range(10)]
        recs  = self._make_records(start, 10)
        elem, mask, _, _ = snap_to_grid(recs, grid, max_tle_gap_hours=25.0)
        assert not np.any(np.isnan(elem[mask]))


# ════════════════════════════════════════════════════════════════════
# Forward fill
# ════════════════════════════════════════════════════════════════════

class TestForwardFill:
    def test_no_missing_unchanged(self):
        elems = np.arange(12, dtype=float).reshape(4, 3)
        mask  = np.ones(4, dtype=bool)
        out   = _forward_fill(elems, mask)
        np.testing.assert_array_equal(out, elems)

    def test_missing_filled_from_prev(self):
        elems = np.array([[1.0, 0.0, 0.0],
                          [np.nan]*3,
                          [np.nan]*3,
                          [4.0, 0.0, 0.0]])
        mask  = np.array([True, False, False, True])
        out   = _forward_fill(elems, mask)
        np.testing.assert_allclose(out[1], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(out[2], [1.0, 0.0, 0.0])

    def test_leading_missing_filled(self):
        elems = np.array([[np.nan]*3,
                          [np.nan]*3,
                          [3.0, 0.0, 0.0]])
        mask  = np.array([False, False, True])
        out   = _forward_fill(elems, mask)
        np.testing.assert_allclose(out[0], [3.0, 0.0, 0.0])
        np.testing.assert_allclose(out[1], [3.0, 0.0, 0.0])


# ════════════════════════════════════════════════════════════════════
# Grid dt computation
# ════════════════════════════════════════════════════════════════════

class TestComputeGridDt:
    def test_both_valid_uses_actual_dt(self):
        start  = datetime.datetime(2021, 1, 1, tzinfo=UTC)
        ep1_ts = start.timestamp()
        ep2_ts = (start + datetime.timedelta(hours=26)).timestamp()
        tle_ep    = np.array([ep1_ts, ep2_ts])
        valid     = np.array([True, True])
        dt        = _compute_grid_dt_seconds(tle_ep, valid, 24.0)
        assert len(dt) == 1
        assert abs(dt[0] - 26 * 3600) < 1.0

    def test_missing_slot_uses_nominal_dt(self):
        start  = datetime.datetime(2021, 1, 1, tzinfo=UTC)
        tle_ep = np.array([start.timestamp(), np.nan])
        valid  = np.array([True, False])
        dt     = _compute_grid_dt_seconds(tle_ep, valid, 24.0)
        assert abs(dt[0] - 24 * 3600) < 1.0


# ════════════════════════════════════════════════════════════════════
# Maneuver labels
# ════════════════════════════════════════════════════════════════════

class TestLabelsFromManeuvers:
    def test_no_maneuvers_all_zero(self):
        grid = [datetime.datetime(2021, 1, i + 1, tzinfo=UTC) for i in range(10)]
        lab  = labels_from_maneuvers(grid, [], tolerance_hours=24.0)
        assert (lab == 0).all()

    def test_exact_match_labeled(self):
        t0   = datetime.datetime(2021, 6, 1, tzinfo=UTC)
        grid = [t0 + datetime.timedelta(days=i) for i in range(10)]
        maneuver = [t0 + datetime.timedelta(days=3)]
        lab  = labels_from_maneuvers(grid, maneuver, tolerance_hours=1.0)
        assert lab[3] == 1
        assert lab[2] == 0 and lab[4] == 0

    def test_tolerance_window(self):
        t0   = datetime.datetime(2021, 6, 1, tzinfo=UTC)
        grid = [t0 + datetime.timedelta(days=i) for i in range(10)]
        # Maneuver 12h after day 3 grid point
        maneuver = [t0 + datetime.timedelta(days=3, hours=12)]
        lab_24 = labels_from_maneuvers(grid, maneuver, tolerance_hours=24.0)
        lab_6  = labels_from_maneuvers(grid, maneuver, tolerance_hours=6.0)
        assert lab_24[3] == 1  # within 24h tolerance
        assert lab_6[3]  == 0  # outside 6h tolerance

    def test_label_shape(self):
        grid = [datetime.datetime(2021, 1, i + 1, tzinfo=UTC) for i in range(30)]
        lab  = labels_from_maneuvers(grid, [], tolerance_hours=24.0)
        assert lab.shape == (30,)


# ════════════════════════════════════════════════════════════════════
# Shell ID assignment
# ════════════════════════════════════════════════════════════════════

class TestAssignShellIds:
    def _elements_for(self, alt_km, inc_deg):
        """Create a median element row for a satellite."""
        a   = R_EARTH + alt_km
        n   = (MU_EARTH / a**3) ** 0.5
        return np.array([a, 0.001, math.radians(inc_deg), 1.0, 0.5, 0.3])

    def test_geo_satellite(self):
        names   = ["GEO-1"]
        elems   = self._elements_for(35_787, 0.1).reshape(1, 6)
        ids     = assign_shell_ids(names, elems)
        assert ids[0] == 0

    def test_sso_satellite(self):
        names   = ["SSO-1"]
        elems   = self._elements_for(800, 98.6).reshape(1, 6)
        ids     = assign_shell_ids(names, elems)
        assert ids[0] == 1

    def test_leo66_satellite(self):
        names   = ["LEO66-1"]
        elems   = self._elements_for(1_330, 66.0).reshape(1, 6)
        ids     = assign_shell_ids(names, elems)
        assert ids[0] == 2

    def test_mixed_constellation(self):
        names = ["GEO", "SSO", "LEO66"]
        elems = np.stack([
            self._elements_for(35_787, 0.1),
            self._elements_for(800,    98.6),
            self._elements_for(1_330,  66.0),
        ])
        ids = assign_shell_ids(names, elems)
        assert ids[0] == 0
        assert ids[1] == 1
        assert ids[2] == 2


# ════════════════════════════════════════════════════════════════════
# Graph construction
# ════════════════════════════════════════════════════════════════════

class TestBuildOrbitalNeighborGraph:
    def _simple_snapshot(self, S=4, alt=500, inc_deg=51.6):
        a = R_EARTH + alt
        return np.array([
            [a + i * 0.5, 0.001, math.radians(inc_deg), i * 0.1, 0.5, 0.3]
            for i in range(S)
        ])

    def test_output_shape(self):
        snap = self._simple_snapshot(4)
        sid  = np.zeros(4, dtype="int8")
        adj  = build_orbital_neighbor_graph(snap, sid, k_neighbors=2)
        assert adj.shape == (4, 4)

    def test_symmetric(self):
        snap = self._simple_snapshot(4)
        sid  = np.zeros(4, dtype="int8")
        adj  = build_orbital_neighbor_graph(snap, sid, k_neighbors=2)
        np.testing.assert_array_equal(adj, adj.T)

    def test_no_self_loops(self):
        snap = self._simple_snapshot(4)
        sid  = np.zeros(4, dtype="int8")
        adj  = build_orbital_neighbor_graph(snap, sid, k_neighbors=2)
        np.testing.assert_array_equal(np.diag(adj), 0)

    def test_different_shells_not_connected(self):
        # 2 GEO (shell 0) + 2 SSO (shell 1)
        geo  = np.array([[R_EARTH + 35787, 0.0001, math.radians(0.1), i, 0.5, 0.3]
                         for i in range(2)])
        sso  = np.array([[R_EARTH + 800,   0.001,  math.radians(98.6), i, 0.5, 0.3]
                         for i in range(2)])
        snap = np.vstack([geo, sso])
        sid  = np.array([0, 0, 1, 1], dtype="int8")
        adj  = build_orbital_neighbor_graph(snap, sid, k_neighbors=2, cross_shell=False)
        # GEO[0]↔SSO[2] should not be connected
        assert adj[0, 2] == 0 and adj[0, 3] == 0
        assert adj[1, 2] == 0 and adj[1, 3] == 0

    def test_singleton_shell_no_edges(self):
        snap = self._simple_snapshot(2)
        sid  = np.array([0, 1], dtype="int8")   # each satellite alone in its shell
        adj  = build_orbital_neighbor_graph(snap, sid, k_neighbors=2)
        assert adj.sum() == 0, "Singleton shells should have no edges"

    def test_values_binary(self):
        snap = self._simple_snapshot(5)
        sid  = np.zeros(5, dtype="int8")
        adj  = build_orbital_neighbor_graph(snap, sid, k_neighbors=2)
        unique_vals = set(adj.flatten().tolist())
        assert unique_vals <= {0.0, 1.0}


# ════════════════════════════════════════════════════════════════════
# Residual computation
# ════════════════════════════════════════════════════════════════════

class TestComputeResidualSequences:
    def _make_elements(self, S=3, T=20):
        """Synthetic near-circular LEO constellation."""
        np.random.seed(1)
        a    = R_EARTH + 500 + np.random.randn(S, T) * 0.01
        e    = np.full((S, T), 0.001) + np.random.randn(S, T) * 0.00001
        i    = np.full((S, T), math.radians(51.6)) + np.random.randn(S, T) * 1e-5
        raan = (np.cumsum(np.random.randn(S, T) * 1e-4, axis=1)
                + np.random.uniform(0, 2*math.pi, (S, 1))) % (2 * math.pi)
        argp = np.random.uniform(0, 2*math.pi, (S, T))
        M    = np.random.uniform(0, 2*math.pi, (S, T))
        return np.stack([a, e, i, raan, argp, M], axis=-1)

    def test_output_shape(self):
        el  = self._make_elements(3, 20)
        res = compute_residual_sequences(el, dt_hours=24.0)
        assert res.shape == (3, 19, 6)

    def test_no_nan(self):
        el  = self._make_elements(3, 20)
        res = compute_residual_sequences(el, dt_hours=24.0)
        assert not np.any(np.isnan(res))

    def test_with_variable_dt(self):
        el  = self._make_elements(3, 10)
        dt  = np.full((3, 9), 86400.0)
        dt[0, 3] = 172800.0  # one 48h gap
        res = compute_residual_sequences(el, dt_seconds_grid=dt)
        assert res.shape == (3, 9, 6)
        assert not np.any(np.isnan(res))

    def test_da_residual_small_for_nominal(self):
        """Δa should be near-zero for a smoothly propagated nominal orbit."""
        el = self._make_elements(1, 10)
        res = compute_residual_sequences(el, dt_hours=24.0)
        # Δa residual should be much smaller than the orbit altitude variation
        assert np.abs(res[0, :, 0]).mean() < 1.0


# ════════════════════════════════════════════════════════════════════
# Per-satellite normalisation
# ════════════════════════════════════════════════════════════════════

class TestPerSatelliteScaler:
    def test_fit_shape(self):
        res      = np.random.randn(4, 100, 6)
        med, sc  = fit_per_satellite_scaler(res, train_end=60)
        assert med.shape == (4, 6)
        assert sc.shape  == (4, 6)

    def test_scale_positive(self):
        res     = np.random.randn(4, 100, 6)
        _, sc   = fit_per_satellite_scaler(res, train_end=60)
        assert (sc > 0).all()

    def test_apply_centered(self):
        """Applying the scaler to training data should give mean ≈ 0, std ≈ 1."""
        np.random.seed(7)
        res          = np.random.randn(2, 100, 6) * 10 + 5
        med, sc      = fit_per_satellite_scaler(res, train_end=80)
        normed       = apply_per_satellite_scaler(res[:, :80, :], med, sc,
                                                   clip_sigma=None)
        train_mean   = normed.mean(axis=1)
        np.testing.assert_allclose(train_mean, 0.0, atol=0.1)

    def test_no_test_leakage(self):
        """Scaler fitted on train should not see test data."""
        res         = np.random.randn(2, 100, 6)
        med1, sc1   = fit_per_satellite_scaler(res, train_end=60)
        # Changing test data should not affect scaler
        res_modified       = res.copy()
        res_modified[:, 60:, :] *= 100
        med2, sc2          = fit_per_satellite_scaler(res_modified, train_end=60)
        np.testing.assert_allclose(med1, med2)
        np.testing.assert_allclose(sc1, sc2)


# ════════════════════════════════════════════════════════════════════
# Synthetic simulator (regression)
# ════════════════════════════════════════════════════════════════════

class TestSimulateConstellation:
    def test_output_shapes(self):
        _, eo, lab, sid = simulate_constellation(
            n_shells=2, sats_per_shell=5, n_steps=30, seed=0
        )
        S = 10  # 2 shells × 5 sats
        assert eo.shape  == (S, 30, 6)
        assert lab.shape == (S, 30)
        assert sid.shape == (S,)

    def test_no_nan_in_elements(self):
        _, eo, _, _ = simulate_constellation(
            n_shells=2, sats_per_shell=4, n_steps=20, seed=1
        )
        assert not np.any(np.isnan(eo))

    def test_shell_ids_valid(self):
        _, _, _, sid = simulate_constellation(
            n_shells=3, sats_per_shell=3, n_steps=10, seed=2
        )
        assert set(sid.tolist()).issubset({0, 1, 2})

    def test_deterministic_with_seed(self):
        _, eo1, lab1, _ = simulate_constellation(n_shells=2, sats_per_shell=3, n_steps=10, seed=42)
        _, eo2, lab2, _ = simulate_constellation(n_shells=2, sats_per_shell=3, n_steps=10, seed=42)
        np.testing.assert_array_equal(eo1, eo2)
        np.testing.assert_array_equal(lab1, lab2)


# ════════════════════════════════════════════════════════════════════
# Integration test (requires real dataset)
# ════════════════════════════════════════════════════════════════════

DATASET_PATH = os.environ.get(
    "TLE_DATASET_PATH",
    os.path.expanduser("~/Documents/dl/TLE_observation_benchmark_dataset-main"),
)

@pytest.mark.skipif(
    not os.path.isdir(DATASET_PATH),
    reason="Real TLE benchmark dataset not found",
)
class TestRealDatasetIntegration:
    @pytest.fixture(scope="class")
    def result(self):
        from dataset import load_real_benchmark_dataset
        return load_real_benchmark_dataset(
            DATASET_PATH,
            start_date="2020-01-01",
            end_date="2021-01-01",  # 1 year for speed
            dt_hours=24.0,
        )

    def test_shapes(self, result):
        eo   = result["elements_obs"]
        lab  = result["labels"]
        dts  = result["dt_seconds_grid"]
        S, T, F = eo.shape
        assert F == 6
        assert lab.shape  == (S, T)
        assert dts.shape  == (S, T - 1)

    def test_no_nan_in_elements(self, result):
        assert not np.any(np.isnan(result["elements_obs"]))

    def test_labels_binary(self, result):
        lab = result["labels"]
        assert set(lab.flatten().tolist()).issubset({0, 1})

    def test_dt_seconds_positive(self, result):
        dts = result["dt_seconds_grid"]
        assert (dts > 0).all()

    def test_residuals_no_nan(self, result):
        eo  = result["elements_obs"]
        dts = result["dt_seconds_grid"]
        res = compute_residual_sequences(eo, dt_seconds_grid=dts)
        assert not np.any(np.isnan(res))

    def test_dm_residual_physically_plausible(self, result):
        """ΔM median should be < 0.3 rad with correct actual dt."""
        eo  = result["elements_obs"]
        dts = result["dt_seconds_grid"]
        res = compute_residual_sequences(eo, dt_seconds_grid=dts)
        dM  = res[:, :, 5]
        # With actual dt, ΔM should be much less than π
        assert np.median(np.abs(dM)) < 0.3, \
            f"ΔM median = {np.median(np.abs(dM)):.4f} rad — likely wrong propagation dt"

    def test_adjacency_no_geo_leo_edges(self, result):
        eo  = result["elements_obs"]
        sid = result["shell_id"]
        adj = build_orbital_neighbor_graph(eo[:, -1, :], sid, k_neighbors=3)
        sat_names = result["sat_names"]
        # GEO sats have shell 0; LEO have shells 1,2
        for k_geo in range(len(sat_names)):
            if sid[k_geo] != 0: continue
            for k_leo in range(len(sat_names)):
                if sid[k_leo] == 0: continue
                assert adj[k_geo, k_leo] == 0, \
                    f"GEO {sat_names[k_geo]} should not be connected to LEO {sat_names[k_leo]}"
