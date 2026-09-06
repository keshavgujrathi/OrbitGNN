# OrbitGNN: Physics-Residual Graph Anomaly Detection for Satellite Constellations

OrbitGNN is a deep-learning system for detecting anomalies in satellite orbital behaviour. It combines a hand-derived orbital mechanics propagator with a graph-based neural network to identify satellites that deviate from both their own expected physical trajectory and the behaviour of their orbital-plane peers.

**OrbitGNN now uses real historical TLE data** from the TLE Observation Benchmark Dataset as its primary benchmark. The synthetic simulator is retained for regression testing only.

---

## Motivation

Conventional approaches to satellite anomaly detection typically use orbital mechanics only as a validity filter — discarding physically impossible predictions after the fact. OrbitGNN instead treats orbital mechanics as a primary feature source: the residual between a satellite's observed state and its physics-predicted state becomes the input signal for anomaly detection, rather than a post-hoc sanity check.

The system is further motivated by the observation that satellites do not operate in isolation — constellation satellites share orbital shells with many peers, and this structure has diagnostic value: deviations shared across an entire shell (e.g., from atmospheric drag driven by solar activity) are less significant than deviations isolated to a single satellite.

---

## Real Benchmark Dataset

### Source

**TLE Observation Benchmark Dataset**
- Paper: *Wide-scale Monitoring of Satellite Lifetimes: Pitfalls and a Benchmark Dataset*
- Repository: https://github.com/dpshorten/TLE_observation_benchmark_dataset

This dataset contains **real historical TLE observations** for 15 satellites spanning 1992–2022, along with **expert-verified manoeuvre timestamps** sourced from official operator records.

### Data Contents

```
processed_files/
├── <satellite>.tle          # Real TLE records (one per day, approx.)
└── manoeuvres_<satellite>.yaml  # Verified manoeuvre timestamps (ISO-8601 UTC)
```

Each `.tle` file contains real Two-Line Element records published by NORAD/US Space Force and archived by Space-Track. The mean Keplerian elements are extracted directly from these records (not propagated via SGP4 — see *Physics Engine* section).

### Satellite Inventory (2020–2022 window, 9 satellites)

| Satellite    | Orbit Family     | Altitude  | Inclination | Shell |
|-------------|-----------------|-----------|-------------|-------|
| Fengyun-2F  | GEO              | 35,787 km | 2.5°        | 0     |
| Fengyun-2H  | GEO              | 35,787 km | 0.5°        | 0     |
| Fengyun-4A  | GEO              | 35,787 km | 0.1°        | 0     |
| CryoSat-2   | SSO / polar LEO  | 719 km    | 92.0°       | 1     |
| SARAL       | SSO / polar LEO  | 785 km    | 98.5°       | 1     |
| Sentinel-3A | SSO / polar LEO  | 803 km    | 98.6°       | 1     |
| Sentinel-3B | SSO / polar LEO  | 803 km    | 98.6°       | 1     |
| Jason-3     | LEO 66°          | 1,338 km  | 66.0°       | 2     |
| Sentinel-6A | LEO 66°          | 1,330 km  | 66.0°       | 2     |

### How to Place the Dataset

```
<project-root>/
├── OrbitGNN/             # This repository
│   ├── dataset.py
│   ├── train.py
│   └── ...
└── TLE_observation_benchmark_dataset-main/
    ├── processed_files/
    │   ├── Jason-3.tle
    │   ├── manoeuvres_Jason-3.yaml
    │   └── ...
    └── README.md
```

---

## Architecture

```
 per-satellite TLE history (real observations)
                │
                ▼
   Two-body + J2 orbital propagator (physics.py)
                │
                │   residual(t) = observed(t) − predicted(t)
                ▼
      ResidualEncoder (Transformer)
                │
                │  latent summary per satellite
                ▼
  NeighborGCN over orbital-plane adjacency graph
  (neighbors = same orbital shell: GEO peers, SSO peers, LEO-66 peers)
                │
                ▼
     peer-aware residual forecast
                │
   ┌────────────┼──────────────────┐
   ▼            ▼                  ▼
 self-forecast  peer-forecast   MC-dropout
 error          error           predictive variance
   └────────────┴──────────────────┘
                ▼
         combined anomaly score  → compare vs. real manoeuvre timestamps
```

---

## Repository Structure

```
OrbitGNN/
├── physics.py        # Two-body + J2 orbital propagator (unchanged)
├── dataset.py        # Real TLE loader + synthetic fallback
├── model.py          # OrbitGNN architecture (unchanged)
├── train.py          # Training and evaluation
├── notebooks/
│   └── demo.ipynb    # End-to-end demo
├── results/          # Generated outputs
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Train on Real Benchmark Data (primary)

```bash
python train.py \
    --dataset real \
    --dataset-path /path/to/TLE_observation_benchmark_dataset-main
```

All configurable options:

```bash
python train.py \
    --dataset real \
    --dataset-path /path/to/TLE_observation_benchmark_dataset-main \
    --start-date 2020-01-01 \
    --end-date   2022-01-01 \
    --dt-hours   24 \
    --max-tle-gap 48 \
    --maneuver-tolerance 36
```

### Train on Synthetic Data (regression only)

```bash
python train.py --dataset synthetic
```

### Verify Data Pipeline

```bash
python dataset.py /path/to/TLE_observation_benchmark_dataset-main
```

---

## Train / Validation / Test Split

Real data uses a **strict chronological split** to prevent future information from leaking into training.

| Split      | Fraction | Purpose                              |
|-----------|----------|--------------------------------------|
| TRAIN     | 60 %     | Nominal windows only (self-supervised)|
| VALIDATION| 20 %     | Training monitoring (informational)  |
| TEST      | 20 %     | All metrics computed here            |

No shuffling of the time axis is performed. The model never sees any window's target label during training.

---

## Maneuver Label Construction

1. Real manoeuvre timestamps are loaded from `manoeuvres_<satellite>.yaml`.
2. Each manoeuvre timestamp is matched to the nearest grid slot within `±maneuver_tolerance_hours` (default **36 h**).
3. Matched slots are labelled **1**; all others are **0**.
4. The original manoeuvre timestamp is preserved separately for **event-level evaluation**.

Scientific justification for 36 h tolerance: TLE records reflect the orbital state *after* a manoeuvre has been performed and a new element set has been fitted by ground stations. This fit typically appears 12–36 h after the burn, so a ±36 h window reliably captures the post-manoeuvre TLE without labelling an entire week.

---

## Orbital Neighbor Graph

Graph edges are constructed based on **scientific orbital similarity**, not arbitrary co-temporal presence.

| Shell | Members                                      | Basis for adjacency             |
|-------|----------------------------------------------|---------------------------------|
| 0     | Fengyun-2F, 2H, 4A                           | GEO belt, similar altitude/inc  |
| 1     | CryoSat-2, SARAL, Sentinel-3A, Sentinel-3B   | SSO, 92–99°, 720–803 km         |
| 2     | Jason-3, Sentinel-6A                         | LEO 66°, 1 330–1 338 km         |

Satellites in different shells are **not** connected by default. Cross-shell edges can be enabled with `cross_shell=True` in `build_orbital_neighbor_graph()`.

---

## Missing Data Handling

Real TLEs arrive approximately once per day (median gap ≈ 24 h) but gaps up to 270 h exist.

| Situation                          | Handling                                             |
|-----------------------------------|------------------------------------------------------|
| Gap ≤ max_tle_gap_hours (48 h)    | Nearest real TLE is used; offset is recorded.        |
| Gap > max_tle_gap_hours            | Slot is marked **MISSING**; forward-filled from prior state. |
| Satellite valid fraction < 50 %   | Satellite is **excluded** from the dataset with a warning.   |

No orbital elements are interpolated or fabricated. Forward-filling is conservative and is explicitly documented in the `valid_mask` output.

---

## Physics Engine

`physics.py` is **unchanged**. Mean Keplerian elements from TLE records are used directly as input to `KeplerianElements(a, e, i, raan, argp, M)`.

Mean elements from TLEs are **not** propagated via SGP4 to a common epoch. The benchmark authors recommend this approach explicitly: propagating across large time gaps introduces SGP4 model error that would corrupt the physics residual signal that OrbitGNN is designed to detect.

---

## Evaluation Metrics

All metrics are computed on the TEST split only.

| Metric                  | Description                                                 |
|------------------------|-------------------------------------------------------------|
| ROC-AUC                | Area under ROC curve                                        |
| PR-AUC                 | Area under precision-recall curve (more informative for sparse anomalies) |
| Precision / Recall / F1| At max-F1 threshold found on test set                       |
| Confusion matrix       | TP / FP / FN / TN at the same threshold                     |
| Event detection rate   | Fraction of real manoeuvre events flagged within 72 h window|
| False alarm rate       | Fraction of non-manoeuvre time steps flagged                |
| Avg detection offset   | Mean hours between model alarm and real manoeuvre timestamp |

---

## Possible Extensions

* Integrate Space-Track CDMs as additional supervised signals
* Add cross-shell edges based on conjunction risk
* Evaluate per anomaly type (drag vs. manoeuvre)
* Use DORIS precise orbital data (included in benchmark) for residual validation

---

## Assumptions and Limitations

1. **Forward-fill assumption**: Missing TLE slots are filled from the prior valid observation. This is conservative but means the model cannot detect anomalies occurring during data gaps.
2. **Singleton shells have no graph edges**: Jason-3 and Sentinel-6A are isolated in shell 2 during early 2020 (before Sentinel-6A launched). This is scientifically correct — no peer comparison is possible.
3. **Manoeuvre labelling is post-hoc**: Labels are derived from verified historical records. Operational real-time use would have no labels during training; the self-supervised design already accounts for this.
4. **No synthetic anomalies injected into real data**: Ground truth comes entirely from the benchmark's expert-verified manoeuvre timestamps.

---

## Citation

If you use the TLE benchmark data, please cite:

> D. Shorten et al., *Wide-scale Monitoring of Satellite Lifetimes: Pitfalls and a Benchmark Dataset*, 2023. https://github.com/dpshorten/TLE_observation_benchmark_dataset

---

## License

MIT — see `LICENSE`.
