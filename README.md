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

> All results on TEST split: 2021-08-10 → 2022-01-01 (145 windows, 30 manoeuvre events)

### Baseline Comparison (deterministic, fixed)

| Method | ROC-AUC | PR-AUC | Precision | Recall | F1 | Type |
|--------|---------|--------|-----------|--------|----|------|
| ResidMag | 0.5616 | 0.0497 | 0.0569 | 0.7931 | 0.1061 | causal, no learning |
| RollingZ | 0.5054 | 0.0501 | 0.0550 | 0.3103 | 0.0935 | causal, no learning |
| IsolationForest | 0.5746 | 0.1222 | 0.1450 | 0.3276 | 0.2011 | **batch, non-causal** |

### Multi-Seed Results — Full OrbitGNN (5 seeds, causal)

| Seed | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|------|---------|--------|-----------|--------|----|
| 1 | 0.5880 | 0.0757 | 0.1429 | 0.2586 | 0.1840 |
| 2 | 0.5720 | 0.0861 | 0.1772 | 0.2414 | 0.2044 |
| 3 | 0.5877 | 0.0747 | 0.1522 | 0.2414 | 0.1867 |
| 4 | 0.5815 | 0.0807 | 0.1519 | 0.2069 | 0.1752 |
| 5 | **0.6036** | **0.0907** | 0.1522 | 0.2414 | 0.1867 |
| **Mean ± Std** | **0.5866 ± 0.0103** | **0.0816 ± 0.0061** | 0.1553 ± 0.0115 | 0.2379 ± 0.0169 | **0.1874 ± 0.0095** |

### Ablation Study (mean ± std, 3 seeds each)

| Configuration | Physics | Transformer | GNN | ROC-AUC | PR-AUC | F1 |
|--------------|:-------:|:-----------:|:---:|---------|--------|-----|
| Physics Only | ✅ | ❌ | ❌ | 0.5616 ± 0.0000 | 0.0497 ± 0.0000 | 0.1061 ± 0.0000 |
| Physics + Transformer | ✅ | ✅ | ❌ | 0.5788 ± 0.0058 | 0.0830 ± 0.0028 | 0.1860 ± 0.0104 |
| Physics + GNN | ✅ | ❌ | ✅ | 0.5587 ± 0.0060 | **0.1168 ± 0.0119** | **0.2404 ± 0.0045** |
| **Full OrbitGNN** | ✅ | ✅ | ✅ | **0.5866 ± 0.0103** | 0.0816 ± 0.0061 | 0.1874 ± 0.0095 |

> The Transformer adds +0.017 ROC-AUC over physics-only. The GNN improves PR-AUC by 2.35×. The full model achieves best ROC-AUC.

### Event-Level Detection (mean ± std across 5 seeds)

| Tolerance | Detection Rate | False Alarms/day |
|-----------|---------------|-----------------|
| ±24h | 39.3% ± 3.3% | 0.29 |
| ±48h | 51.3% ± 7.5% | 0.27 |
| ±72h | **61.3% ± 8.6%** | 0.24 |

Detection offset: mean +3.9h, median −6.5h (negative = alarm before official timestamp, explained by TLE publication lag).

### Per-Satellite Results (seed=5, ±72h tolerance)

| Satellite | Shell | Man | Det | Det% | FA/day | ΔV est (m/s) |
|-----------|-------|-----|-----|------|--------|-------------|
| CryoSat-2 | SSO/Polar | 4 | 3 | 75.0% | 0.069 | ~0.000 |
| Fengyun-2F | GEO | 3 | 2 | 66.7% | 0.014 | 0.254 |
| Fengyun-2H | GEO | 1 | 1 | **100%** | 0.014 | 0.001 |
| Fengyun-4A | GEO | 8 | 5 | 62.5% | 0.014 | 0.117 |
| Jason-3 | LEO-66° | 1 | 1 | **100%** | 0.007 | 0.005 |
| SARAL | SSO/Polar | 1 | 1 | **100%** | 0.090 | 0.044 |
| Sentinel-3A | SSO/Polar | 7 | 6 | 85.7% | **0.000** | 0.011 |
| Sentinel-3B | SSO/Polar | 3 | 2 | 66.7% | 0.014 | 0.011 |
| Sentinel-6A | LEO-66° | 2 | 0 | 0.0% | 0.035 | N/A |

ΔV estimated via linearised Gauss tangential burn: `ΔV ≈ (n·|Δa|)/2` where `n=√(μ/a³)`.

---

## Key Technical Details

### Physics Corrections Applied (vs. original synthetic pipeline)

| Bug | Original | Fixed |
|-----|----------|-------|
| Propagation dt | Fixed 24h for all steps | Actual inter-TLE elapsed time per step |
| Normalisation | Global GEO+LEO mixed | Per-satellite mean/std, training data only |
| Angular residuals | Not wrapped | `wrap_angle()` → Δθ ∈ (−π, π] |
| Manoeuvre tolerance | 36h | Tightened to 24h |
| Baselines | None | ResidMag, RollingZ, IsolationForest |
| ΔM residual after fix | ~3 rad (50× too large) | **0.057 rad** |

### Orbital Physics

- **Model**: Two-body Kepler + first-order J₂ secular perturbations
- **Constants**: μ = 398,600.4418 km³/s², R_E = 6,378.137 km, J₂ = 1.082626680 × 10⁻³
- **Kepler solver**: Newton-Raphson, tolerance 10⁻¹⁰ rad
- **Δv estimation**: `ΔV ≈ (n·|Δa|)/2` — linearised Gauss tangential burn formula

### New in This Branch

| Feature | Details |
|---------|---------|
| `estimate_delta_v()` | Post-hoc ΔV characterisation from TLE Δa |
| `per_satellite_analysis()` | Per-satellite detection / FA / ΔV table |
| `tests/test_model.py` | 35 model unit tests (shapes, NaN, gradients) |
| `.github/workflows/tests.yml` | CI on Python 3.10/3.11/3.12 |
| `notebooks/demo.ipynb` | 8-cell real-TLE end-to-end walkthrough |

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
2. **Simplified physics**: J₂-only; atmospheric drag, SRP, and lunisolar perturbations unmodelled.
3. **TLE timing uncertainty**: Ground-station solutions lag manoeuvre events by 6–36h.
4. **Graph sparsity**: Shell 2 (LEO-66°) has only 2 nodes and 1 edge.
5. **Unsupervised**: Model never sees anomaly examples during training.
6. **IsolationForest is non-causal**: Batch access to full history gives it an advantage over causal OrbitGNN.
7. **Sentinel-6A coverage**: Only 9% valid TLE slots in 2020 (satellite was recently launched) — detection is 0%.

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
Curtis, H.D. (2013). Orbital Mechanics for Engineering Students, 3rd ed. §6.3
```

---

## License

MIT — see `LICENSE`.
