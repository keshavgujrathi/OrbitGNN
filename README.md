# OrbitGNN: Physics-Residual Graph Anomaly Detection for Satellite Constellations

OrbitGNN is a deep-learning system for detecting anomalies in satellite orbital behaviour. It combines a hand-derived orbital mechanics propagator with a graph-based neural network to identify satellites that deviate from both their own expected physical trajectory and the behaviour of their orbital-plane peers.

**OrbitGNN uses real historical TLE data** from the TLE Observation Benchmark Dataset as its primary benchmark. The synthetic simulator is retained for regression testing only.

---

## Motivation

Conventional approaches to satellite anomaly detection typically use orbital mechanics only as a validity filter. OrbitGNN instead treats orbital mechanics as a primary feature source: the residual between a satellite's observed state and its physics-predicted state becomes the input signal for anomaly detection.

The system is further motivated by the observation that satellites do not operate in isolation — constellation satellites share orbital shells with many peers, and this structure has diagnostic value: deviations shared across an entire shell (e.g., from atmospheric drag driven by solar activity) are less significant than deviations isolated to a single satellite.

---

## Real Benchmark Dataset

### Source

**TLE Observation Benchmark Dataset**
- Paper: *Wide-scale Monitoring of Satellite Lifetimes: Pitfalls and a Benchmark Dataset*
- Repository: https://github.com/dpshorten/TLE_observation_benchmark_dataset

This dataset contains **real historical TLE observations** for 15 satellites spanning 1992–2022, along with **expert-verified manoeuvre timestamps** sourced from official operator records.

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

---

## Architecture

```
 per-satellite TLE history (real observations)
                │
                ▼
   Two-body + J2 orbital propagator  (physics.py)
                │
                │   residual(t) = observed(t+1) − propagate(observed(t), Δt_actual)
                ▼
      ResidualEncoder (Transformer, d_model=64, 4 heads, 2 layers)
                │
                │  latent summary per satellite
                ▼
  NeighborGCN over orbital-plane adjacency graph
  (same orbital shell: GEO peers / SSO peers / LEO-66° peers)
                │
                ▼
     peer-aware residual forecast
                │
   ┌────────────┼──────────────────┐
   ▼            ▼                  ▼
 self-forecast  peer-forecast   MC-dropout (15 passes)
 error          error           predictive variance
   └────────────┴──────────────────┘
                ▼
         combined anomaly score  → compare vs. real manoeuvre timestamps
```

---

## Repository Structure

```
OrbitGNN/
├── physics.py            # Two-body + J2 orbital propagator + Δv estimation
├── dataset.py            # Real TLE loader + synthetic fallback
├── model.py              # OrbitGNN architecture
├── train.py              # Training + evaluation + baselines
├── run_experiments.py    # Full validation study (multi-seed + ablations)
├── tests/
│   ├── test_physics.py   # 30 unit tests for physics module
│   ├── test_dataset.py   # 70 unit tests for data pipeline
│   ├── test_graph.py     # 22 unit tests for graph construction
│   └── test_model.py     # 35 unit tests for model components
├── .github/
│   └── workflows/
│       └── tests.yml     # GitHub Actions CI (Python 3.10/3.11/3.12)
├── results/
│   ├── validation/       # Scientific validation outputs
│   │   ├── seed_results.csv
│   │   ├── ablation_results.csv
│   │   ├── baseline_results.csv
│   │   ├── event_detection_results.csv
│   │   ├── per_satellite_results.csv
│   │   ├── error_analysis.md
│   │   ├── final_report.md
│   │   └── *.png         # 8 plots
│   └── *.png             # Training run plots
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/keshavgujrathi/OrbitGNN
git checkout real-tle-pipeline
pip install -r requirements.txt
```

---

## Usage

### Single Training Run

```bash
# Real benchmark (primary)
python train.py \
    --dataset real \
    --dataset-path /path/to/TLE_observation_benchmark_dataset-main \
    --seed 42

# Synthetic (regression only)
python train.py --dataset synthetic

# Baselines only (no model training)
python train.py --dataset real --dataset-path /path/to/... --baselines-only
```

### Full Scientific Validation Study

```bash
# Run 5 seeds + ablation study + all reports (≈20 min on CPU)
python run_experiments.py \
    --dataset-path /path/to/TLE_observation_benchmark_dataset-main \
    --seeds 1 2 3 4 5

# All outputs saved to results/validation/
```

### Tests (157 tests, all pass)

```bash
python -m pytest tests/ -v      # runs all 157 tests (physics + dataset + graph + model)
python physics.py                # physics self-test (also runs in CI)
```

---

## Scientific Validation Results

### Multi-Seed Results (5 seeds, Full OrbitGNN)

| Metric | Mean ± Std |
|--------|-----------|
| ROC-AUC | see `results/validation/seed_results.csv` |
| PR-AUC | see `results/validation/seed_results.csv` |
| F1 | see `results/validation/seed_results.csv` |

### Baseline Comparison (TEST split 2021-08-10 → 2022-01-01)

| Method | ROC-AUC | PR-AUC | F1 | Type |
|--------|---------|--------|-----|------|
| ResidMag | 0.562 | 0.050 | 0.106 | causal, no learning |
| RollingZ | 0.505 | 0.050 | 0.094 | causal, no learning |
| IsolationForest | 0.575 | 0.122 | 0.201 | batch (non-causal) |
| OrbitGNN | see validation/ | see validation/ | see validation/ | causal, learned |

### Ablation Study

| Configuration | Physics | Transformer | GNN |
|--------------|---------|-------------|-----|
| Physics Only | ✅ | ❌ | ❌ |
| Physics + Transformer | ✅ | ✅ | ❌ |
| Physics + GNN | ✅ | ❌ | ✅ |
| Full OrbitGNN | ✅ | ✅ | ✅ |

See `results/validation/ablation_results.csv` for numerical results.

### Event-Level Detection

OrbitGNN correctly detected **25/30 manoeuvre events (83.3%)** within ±72h in the test split (seed=42).
Detection rates at ±24h, ±48h, ±72h are reported in `results/validation/event_detection_results.csv`.

---

## Key Technical Details

### Physics Corrections Applied

| Bug | Fix |
|-----|-----|
| Fixed 24h propagation dt for all steps | Now uses actual inter-TLE elapsed time per step |
| Global normalisation (GEO + LEO mixed) | Per-satellite mean/std normalisation, training-data only |
| Angular residuals not wrapped | `wrap_angle()` ensures Δθ ∈ (−π, π] |
| Manoeuvre tolerance 36h (too wide) | Tightened to 24h based on measured TLE-to-event offsets |
| No baselines | Added ResidMag, RollingZ, IsolationForest |

### Orbital Physics

- **Model**: Two-body Kepler + first-order J₂ secular perturbations
- **Constants**: μ = 398,600.4418 km³/s² (IAU 2012), R_E = 6,378.137 km, J₂ = 1.082626680 × 10⁻³
- **Kepler solver**: Newton-Raphson, tolerance 10⁻¹⁰ rad, converges in ≤5 iterations for e < 0.01
- **Residual ΔM median**: 0.057 rad (corrected; was ~3 rad with fixed 24h dt — 50× improvement)

---

## Train / Validation / Test Split

Real data uses a **strict chronological split** (no temporal leakage):

| Split | Period | Windows |
|-------|--------|---------|
| TRAIN (60%) | 2020-01-10 → 2021-03-17 | 433 |
| VALIDATION (20%) | 2021-03-18 → 2021-08-09 | 145 |
| TEST (20%) | 2021-08-10 → 2022-01-01 | 145 |

---

## Limitations

1. **Small dataset**: 9 satellites × 2 years limits statistical power.
2. **Simplified physics**: J₂-only; atmospheric drag, SRP, and lunisolar perturbations are unmodelled.
3. **TLE timing uncertainty**: Ground-station solutions lag manoeuvre events by 6–36h.
4. **Graph sparsity**: Shell 2 (LEO-66°) has only 2 nodes and 1 edge.
5. **Unsupervised**: The model never sees anomaly examples during training.
6. **IsolationForest is non-causal**: Batch access to all history gives it an advantage over the online OrbitGNN.

---

## Citation

**This implementation**:
```
OrbitGNN Real-TLE Pipeline (2026)
https://github.com/keshavgujrathi/OrbitGNN (branch: real-tle-pipeline)
```

**TLE Benchmark Dataset**:
```
Shorten, D. & Abdurahimov, A. (2023). TLE Observation Benchmark Dataset.
https://github.com/dpshorten/TLE_observation_benchmark_dataset
```

**Orbital mechanics**:
```
Vallado, D.A. (2013). Fundamentals of Astrodynamics and Applications, 4th ed. §9.6
Bate, R.R., Mueller, D.D., White, J.E. (1971). Fundamentals of Astrodynamics. Dover.
```

---

## License

MIT — see `LICENSE`.
