# FORM 2
## PROVISIONAL SPECIFICATION
### (See Section 9, Rule 13)
### Indian Patent Office

---

**Title of Invention:**

# OrbitGNN: A Physics-Informed Graph Neural Network System and Method for Satellite Manoeuvre Detection and Delta-V Estimation Using Publicly Available Two-Line Element Orbital Data

**Inventors:**
1. Chunduri Sai Kaushik
2. Keshav Gujrathi

**Applicant:** [INSTITUTION NAME], [City], India

---

## 1. FIELD OF THE INVENTION

The present invention relates to the field of **space situational awareness, satellite tracking, and machine learning**. More particularly, the invention provides a novel system and method that combines orbital physics propagation with Graph Neural Networks (GNNs) to automatically detect satellite manoeuvres and estimate the associated delta-velocity (ΔV) using only publicly available Two-Line Element (TLE) orbital data, without access to proprietary satellite telemetry or propulsion system data.

---

## 2. BACKGROUND OF THE INVENTION

### 2.1 The Problem

As of 2024, more than 27,000 artificial objects are tracked in Earth orbit by ground-based surveillance networks. Among these are active satellites that periodically perform **orbital manoeuvres** — controlled burns of their propulsion systems — to:
- Maintain their operational position (station-keeping)
- Avoid collisions with debris or other satellites
- Adjust their ground track for mission requirements

When a satellite performs an unplanned or unexpected manoeuvre, **ground operators must be alerted quickly** to update collision avoidance calculations. A delayed detection — even by a few hours — can result in stale conjunction screening and increased collision risk, contributing to the Kessler cascade phenomenon that could render certain orbital regimes unusable.

The primary source of publicly available orbital state information is the **Two-Line Element (TLE) set**, a two-line ASCII format published by the United States Space Surveillance Network (US SSN) via the Space-Track.org catalogue. TLEs are routinely updated for tracked objects and provide six mean Keplerian orbital elements: semi-major axis (a), eccentricity (e), inclination (i), right ascension of ascending node (Ω), argument of perigee (ω), and mean anomaly (M).

### 2.2 Limitations of Existing Methods

Existing methods for satellite manoeuvre detection suffer from one or more of the following critical limitations:

**(a) Reliance on Synthetic Data:**
Most prior machine learning approaches are trained and evaluated on synthetically generated TLE data with idealised Gaussian noise models. Such models do not capture the real-world irregularities of TLE publication cadences, ground station visibility gaps, or measurement noise from the actual radar and optical tracking systems. Consequently, models trained on synthetic data do not generalise well to real operational scenarios.

**(b) Ignoring Orbital Physics:**
Machine learning methods that treat TLE elements as raw numerical features without applying orbital physics fail to exploit the known mathematical structure of Keplerian dynamics. The variation in orbital elements between successive TLEs is governed by well-understood physical equations involving mean motion, J₂ zonal harmonic perturbation, atmospheric drag, and solar radiation pressure. Ignoring these physics leads to models that are confounded by natural orbital evolution and cannot distinguish a genuine manoeuvre signature from expected orbital drift.

**(c) No Peer Satellite Context:**
No existing method exploits the fact that **satellites in similar orbital regimes experience similar environmental perturbations** (same atmospheric density layer, same solar radiation environment). If a single satellite shows an anomalous change in its semi-major axis while its same-regime neighbours do not, this asymmetry is a strong indicator of a manoeuvre. Existing single-satellite detection methods discard this diagnostic signal entirely.

**(d) Incorrect Elapsed Time Handling:**
All known prior methods assume a fixed time step (e.g., exactly 24 hours) when propagating orbital elements forward for residual computation. In reality, TLEs are published at irregular intervals, and the actual elapsed time between consecutive TLE epochs varies from hours to several days. Using a fixed Δt introduces systematic errors in the propagated mean anomaly (ΔM), which is the most sensitive indicator of thruster activity.

**(e) No Delta-V Estimation:**
No prior TLE-based detection method provides an estimate of the manoeuvre magnitude in physical units (meters per second). Such ΔV estimates are operationally valuable for fuel budget tracking, insurance assessment, and threat characterisation, but have previously required access to proprietary propulsion system telemetry or high-precision orbit determination data.

### 2.3 Need for the Invention

There is therefore a need for a method that:
1. Operates purely on **publicly available TLE data** — no proprietary sensors or telemetry required
2. Applies **correct orbital physics** using actual inter-TLE elapsed times
3. Exploits **peer satellite context** through a graph-structured neural network
4. Provides **uncertainty-quantified anomaly scores**
5. Estimates **manoeuvre ΔV** from TLE-derived residuals alone

The present invention — **OrbitGNN** — fulfils all five of these requirements.

---

## 3. OBJECTS OF THE INVENTION

The principal objects of the present invention are:

1. To provide a method for detecting satellite orbital manoeuvres using only publicly available Two-Line Element (TLE) data without requiring access to proprietary satellite telemetry, ground station data, or propulsion system information.

2. To provide a physics-informed residual computation method that uses the actual elapsed time between consecutive TLE observations for J₂-corrected Keplerian propagation, eliminating the fixed Δt assumption of prior art.

3. To provide a Transformer-based encoder architecture (ResidualEncoder) that captures temporal patterns in orbital physics residual sequences for improved anomaly detection.

4. To provide an orbital-shell-aware graph construction method that connects satellites within the same orbital regime (GEO, SSO, LEO) without cross-regime edges, enabling physically meaningful peer context aggregation.

5. To provide a composite anomaly scoring method that combines self-forecast error, peer-graph-forecast error, and Monte Carlo Dropout epistemic uncertainty in a z-normalised form across the satellite batch.

6. To provide a method for estimating manoeuvre delta-velocity (ΔV) from TLE-derived semi-major axis residuals using the linearised Gauss tangential burn equation, without access to propulsion telemetry.

7. To provide a self-supervised training method that requires no labelled manoeuvre annotations during model training.

---

## 4. STATEMENT OF THE INVENTION

According to the present invention, there is provided:

**A system and method for satellite manoeuvre detection and delta-velocity estimation comprising:**

(a) A **TLE loading and grid-snapping module** that parses Two-Line Element records for a plurality of satellites, extracts six mean Keplerian orbital elements at each observation epoch, maps observations to a regular temporal grid, and records the actual elapsed time Δt between consecutive TLE epochs;

(b) A **physics residual computation module** implementing J₂-corrected Keplerian propagation using the said actual elapsed time, computing signed residuals between observed and propagated orbital elements, and applying angular wrapping to maintain residuals within (−π, π];

(c) A **ResidualEncoder module** comprising a Transformer encoder with positional embeddings that maps a window of residual sequences per satellite to a latent embedding and a self-forecast of the next residual step;

(d) An **orbital-shell-aware graph construction module** that assigns each satellite to an orbital shell based on altitude and inclination thresholds, constructs a k-nearest-neighbour adjacency graph within each shell, and excludes edges between satellites of different shells;

(e) A **NeighborGCN module** that performs graph convolution over the said adjacency graph with a residual connection to produce peer-context-aware forecast embeddings;

(f) A **composite anomaly scoring module** that combines z-normalised self-forecast error, peer-forecast error, and Monte Carlo Dropout uncertainty into a per-satellite anomaly score;

(g) A **delta-velocity estimation module** that computes ΔV ≈ (n·|Δa|)/2 from the detected semi-major axis residual Δa and mean motion n; and

(h) A **self-supervised training procedure** that minimises mean-squared forecast errors on training data without requiring manoeuvre labels.

---

## 5. DETAILED DESCRIPTION OF THE INVENTION

### 5.1 System Overview

Figure 1 shows the complete OrbitGNN pipeline. The system processes TLE data for S satellites over a period of T time steps, producing per-satellite anomaly scores at each step that indicate the likelihood of an orbital manoeuvre.

```
INPUT: TLE files (.tle) + Manoeuvre timestamps (.yaml)
         │
         ▼
┌─────────────────────────────────────────────┐
│  MODULE 1: TLE LOADING & GRID SNAPPING      │
│  • Parse TLE epoch → 6 Keplerian elements   │
│  • Snap to daily grid (T steps)             │
│  • Record actual Δt between TLE epochs      │
│  • Generate valid-observation binary mask   │
└─────────────────────┬───────────────────────┘
                      │  elements_obs: (S, T, 6)
                      │  dt_seconds_grid: (S, T-1)
                      ▼
┌─────────────────────────────────────────────┐
│  MODULE 2: PHYSICS RESIDUALS                │
│  • J₂ + Kepler propagation per actual Δt    │
│  • Residual δ = a_obs(t+1) - â(t+1)         │
│  • Angular wrap all angle components        │
│  • Per-satellite zero-mean/unit-std scaling │
└─────────────────────┬───────────────────────┘
                      │  residuals: (S, T-1, 6)
          ┌───────────┴──────────┐
          ▼                      ▼
┌──────────────────┐    ┌────────────────────┐
│ MODULE 3A:       │    │ MODULE 3B:         │
│ ResidualEncoder  │    │ Graph Construction │
│ Transformer:     │    │ • Assign shells    │
│ d=64, h=4, L=2   │    │   GEO/SSO/LEO-66° │
│ → summary (S,64) │    │ • k-NN within      │
│ → sf (S,6)       │    │   same shell only  │
└────────┬─────────┘    │ → adjacency A(S,S) │
         └──────────────┴────────┬───────────┘
                                 ▼
              ┌─────────────────────────────────┐
              │  MODULE 4: NeighborGCN          │
              │  H' = GELU(D⁻¹AHW) + H         │
              │  → peer_forecast pf (S,6)       │
              └────────────────┬────────────────┘
                               ▼
              ┌─────────────────────────────────┐
              │  MODULE 5: ANOMALY SCORING      │
              │  MC-Dropout (15 passes) → std   │
              │  score = z(self) + z(peer)      │
              │        + z(uncertainty)         │
              └────────────────┬────────────────┘
                               ▼
              ┌─────────────────────────────────┐
              │  MODULE 6: ΔV ESTIMATION        │
              │  ΔV ≈ (n·|Δa|)/2  [m/s]        │
              │  n = √(μ/a³) mean motion        │
              └─────────────────────────────────┘
```

### 5.2 Module 1: TLE Loading and Grid Snapping

For each satellite k ∈ {1, ..., S}, the TLE archive file is parsed to extract the six mean Keplerian elements at each available epoch:

**a** = [a, e, i, Ω, ω, M]

where:
- a = semi-major axis (km)
- e = eccentricity (dimensionless)  
- i = inclination (radians)
- Ω = right ascension of ascending node (radians)
- ω = argument of perigee (radians)
- M = mean anomaly (radians)

TLE epochs are mapped to the nearest daily grid point within a tolerance of 48 hours. For each occupied grid step t, the **actual elapsed time** between consecutive TLE epochs is recorded as Δt_{k,t} in seconds. Grid steps without a TLE observation within ±48 hours are marked as invalid via binary mask v_{k,t} ∈ {0,1}.

**Key Innovation:** Unlike all prior art methods that assume a fixed propagation time step (typically 24 hours), the present invention records and uses the actual Δt for each satellite-step pair. This is critical because TLEs are published at irregular intervals, and forcing a 24-hour Δt introduces systematic errors proportional to the difference between actual and assumed elapsed time.

### 5.3 Module 2: Physics Residual Computation

#### 5.3.1 J₂-Corrected Keplerian Propagation

For each valid satellite-step pair (k, t), the observed orbital state **a**(t) is propagated forward by the actual elapsed time Δt_{k,t} using a two-body + first-order J₂ secular perturbation model:

**J₂ secular perturbation rates:**

```
n = √(μ/a³)                              (mean motion)
p = a(1 - e²)                            (semi-latus rectum)

dΩ/dt = -(3/2) · n · J₂ · (Rₑ/p)² · cos(i)
dω/dt =  (3/4) · n · J₂ · (Rₑ/p)² · (5cos²(i) - 1)
dM/dt =   n   [mean motion propagation]
```

where J₂ = 1.08262668 × 10⁻³ (Earth's second zonal harmonic coefficient), Rₑ = 6378.137 km (Earth's equatorial radius), and μ = 398600.4418 km³/s² (Earth's gravitational parameter).

The propagated elements at time t+1 are:

```
â(t+1) = [a(t),
           e(t),
           i(t),
           Ω(t) + (dΩ/dt)·Δt,
           ω(t) + (dω/dt)·Δt,
           M̃(t) + n·Δt]
```

where M̃ is computed from M via the Newton-Raphson Kepler equation solver:
```
E - e·sin(E) = M    (Kepler's equation)
```

#### 5.3.2 Residual Computation

The signed residual for satellite k at step t is:

```
δ_{k,t} = a_obs_{k,t+1} - â_{k,t+1}
```

For angular elements (i, Ω, ω, M), the angular wrapping operation is applied:

```
wrap(θ) = ((θ + π) mod 2π) - π
```

ensuring all angular residuals lie in (−π, π].

#### 5.3.3 Per-Satellite Normalisation

A per-satellite, per-element zero-mean/unit-standard-deviation scaler is fitted on the training split only:

```
μ_{k,f} = mean(δ_{k,:,f})  over training steps
σ_{k,f} = std(δ_{k,:,f})   over training steps (clipped to ≥ 10⁻⁶)

δ̂_{k,t,f} = (δ_{k,t,f} - μ_{k,f}) / σ_{k,f}
```

### 5.4 Module 3A: ResidualEncoder (Transformer)

A sliding window of W = 8 consecutive normalised residuals for satellite k forms the input sequence:

```
X_k = [δ̂_{k,t-W}, δ̂_{k,t-W+1}, ..., δ̂_{k,t-1}]  ∈ ℝ^{W×6}
```

The ResidualEncoder is a Transformer encoder with:
- Input projection: Linear(6 → 64)
- Positional embedding: learned (1 × 512 × 64), truncated to W
- Two Transformer encoder layers: d_model=64, n_heads=4, dim_feedforward=128, GELU activation, dropout=0.2
- Forecast head: Linear(64 → 6)

Output:
```
h_k = summary embedding  ∈ ℝ^64      (CLS token representation)
ŷ^self_k = self-forecast  ∈ ℝ^6      (predicted next residual)
```

### 5.5 Module 3B: Orbital-Shell-Aware Graph Construction

Each satellite is assigned to one of three orbital shells based on its mean altitude and inclination:

| Shell | Name | Altitude | Inclination |
|-------|------|----------|-------------|
| 0 | GEO | > 30,000 km | any |
| 1 | SSO/Polar | 600–1,000 km | > 88° |
| 2 | LEO-66° | 1,000–1,700 km | 60°–70° |

**Key Innovation:** The adjacency graph A ∈ {0,1}^{S×S} is constructed using k-nearest-neighbours (k=3) based on mean-motion similarity, but **only within the same orbital shell**. No edges are created between satellites in different shells. This design choice ensures that the GNN does not propagate GEO manoeuvre signatures to LEO neighbours (who experience fundamentally different perturbation environments), preventing physically meaningless information mixing.

### 5.6 Module 4: NeighborGCN

The graph convolution layer aggregates peer satellite embeddings:

```
H' = GELU(D⁻¹ · A · H · W) + H
```

where:
- H ∈ ℝ^{S×64} = stacked summary embeddings from ResidualEncoder
- W ∈ ℝ^{64×64} = learned weight matrix
- D = diagonal degree matrix, D_{kk} = Σⱼ A_{kj}
- The residual connection H preserves each satellite's own information

A peer-forecast head (Linear(64→6)) maps the output to:
```
ŷ^peer_k = peer-context forecast  ∈ ℝ^6
```

### 5.7 Module 5: Composite Anomaly Scoring

#### 5.7.1 Monte Carlo Dropout Uncertainty

The model is run N=15 times in training mode (dropout active) to produce stochastic peer forecast samples:

```
{ŷ^peer_k^(1), ..., ŷ^peer_k^(N)} → μ̄_k, σ̄_k  (mean and std per element)
```

σ̄_k represents the model's **epistemic uncertainty** about the peer forecast.

#### 5.7.2 Z-Normalised Composite Score

For each satellite k at each time step t, the anomaly score is:

```
score_k = w_self · z(||y_k - ŷ^self_k||²)
         + w_peer · z(||y_k - ŷ^peer_k||²)
         + w_unc  · z(||σ̄_k||²)
```

where z(·) denotes zero-mean, unit-standard-deviation normalisation **across the satellite batch at each time step**:

```
z(x_k) = (x_k - mean_k(x)) / (std_k(x) + ε)
```

**Key Innovation:** The z-normalisation across the satellite batch is a critical design choice. It measures **relative** anomalousness: a satellite is flagged when its error is anomalous compared to its constellation peers at the same time step. This automatically accounts for time-varying environmental perturbations that affect all satellites simultaneously.

### 5.8 Module 6: Delta-V Estimation

**Key Innovation:** For a detected manoeuvre at time step t with semi-major axis residual Δa_{k,t} (km), the tangential ΔV is estimated using the linearised Gauss equation for near-circular orbits:

```
ΔV ≈ (n · |Δa|) / 2    [m/s]
```

where n = √(μ/a³) is the mean motion in rad/s. This formula is derived from the Gauss variational equations for a purely tangential impulsive burn and provides first-order ΔV estimates without any propulsion telemetry. The accuracy is approximately ±50% for tangential burns; radial or inclination-changing burns produce larger estimation errors.

### 5.9 Training Procedure

OrbitGNN is trained in a **self-supervised** manner without manoeuvre labels:

```
Loss = MSE(ŷ^self, y_next) + MSE(ŷ^peer, y_next)
```

Chronological train/validation/test split is used to prevent temporal data leakage:
- Training: first 60% of time series
- Validation: next 20% (used for threshold selection)
- Test: final 20% (held out, never seen during training or threshold selection)

This chronological split is mandatory for TLE-based methods because TLE data is temporally autocorrelated; random splits would constitute data leakage.

---

## 6. BEST MODE OF CARRYING OUT THE INVENTION

### 6.1 Experimental Setup

The invention was implemented in Python 3.11 using PyTorch 2.x and evaluated on the **TLE Observation Benchmark Dataset** (Shorten & Abdurahimov, 2023) — the first publicly available benchmark of real historical TLE records with verified manoeuvre timestamps.

**Dataset:**
- S = 9 satellites (3 GEO: Fengyun-2F/2H/4A; 4 SSO: CryoSat-2, SARAL, Sentinel-3A/3B; 2 LEO-66°: Jason-3, Sentinel-6A)
- T = 732 daily grid steps (2020-01-01 to 2022-01-01)
- 125 total verified manoeuvre events
- Training: 433 sliding-window samples
- Test: 145 sliding-window samples (held out)

**Model Configuration:**
- Parameters: 105,100 (lightweight, suitable for edge deployment)
- Window size: W = 8 steps
- d_model = 64, n_heads = 4, n_layers = 2
- Dropout = 0.2, MC samples = 15
- Anomaly score weights: w_self = w_peer = w_uncertainty = 1.0

### 6.2 Quantitative Results

**Table 1: Multi-Seed Performance (5 seeds, Test Split)**

| Metric | Mean | Std | 95% CI Lower | 95% CI Upper |
|--------|------|-----|-------------|-------------|
| ROC-AUC | 0.587 | 0.010 | 0.482 | 0.660 |
| PR-AUC | 0.082 | 0.006 | 0.045 | 0.120 |
| F1 Score | 0.187 | 0.010 | 0.110 | 0.256 |

**Table 2: Comparison with Baseline Methods (Test Split)**

| Method | ROC-AUC | PR-AUC | F1 | Type |
|--------|---------|--------|-----|------|
| ResidMag (threshold only) | 0.562 | 0.050 | 0.106 | Causal |
| RollingZ (statistical) | 0.505 | 0.050 | 0.094 | Causal |
| IsolationForest | 0.575 | 0.122 | 0.201 | Non-causal |
| **OrbitGNN (proposed)** | **0.587** | **0.082** | **0.187** | **Causal** |

OrbitGNN outperforms all causal baselines. The improvement over ResidMag is statistically significant (one-sample t-test: p = 0.0084, Cohen's d = 2.16, large effect size).

**Table 3: Event-Level Detection Rate (30 events in test split)**

| Detection Window | Detection Rate | False Alarms/Day |
|-----------------|---------------|-----------------|
| ±24 hours | 39.3% ± 3.3% | 0.29 |
| ±48 hours | 51.3% ± 7.5% | 0.27 |
| **±72 hours** | **61.3% ± 8.6%** | **0.24** |

**Table 4: Per-Satellite Results (Best Seed, ±72h)**

| Satellite | Shell | Manoeuvres | Detected | Rate | ΔV Est. (m/s) |
|-----------|-------|-----------|---------|------|--------------|
| Sentinel-3A | SSO | 7 | 6 | 85.7% | 0.011 |
| Fengyun-2H | GEO | 1 | 1 | 100% | 0.001 |
| Jason-3 | LEO | 1 | 1 | 100% | 0.005 |
| CryoSat-2 | SSO | 4 | 3 | 75.0% | ~0.000 |
| Fengyun-4A | GEO | 8 | 5 | 62.5% | 0.117 |
| Fengyun-2F | GEO | 3 | 2 | 66.7% | 0.254 |
| SARAL | SSO | 1 | 1 | 100% | 0.044 |
| Sentinel-3B | SSO | 3 | 2 | 66.7% | 0.011 |
| Sentinel-6A | LEO | 2 | 0 | 0.0% | N/A* |

*Sentinel-6A: recently launched (2020), sparse early TLE coverage reduces detection signal.

**Table 5: Ablation Study (Mean ± Std, 3 Seeds)**

| Configuration | ROC-AUC | PR-AUC | F1 |
|--------------|---------|--------|-----|
| Physics residuals only | 0.562 | 0.050 | 0.106 |
| + Transformer encoder | 0.579 | 0.083 | 0.186 |
| + GNN only (no Transformer) | 0.559 | 0.117 | 0.240 |
| **Full OrbitGNN** | **0.587** | 0.082 | **0.187** |

**Table 6: ΔV Estimates for Test-Split Manoeuvres**

| Satellite | Manoeuvre Date | |Δa| (km) | ΔV Estimate (m/s) |
|-----------|---------------|----------|------------------|
| Fengyun-2F | 2021-08-04 | 0.196 | 0.007 |
| Fengyun-2F | 2021-09-22 | 0.266 | 0.010 |
| Fengyun-4A | 2021-08-20 | 0.091 | 0.003 |
| Fengyun-4A | 2021-09-21 | 0.192 | 0.007 |
| Sentinel-3A | 2021-09-15 | 0.031 | 0.011 |

These ΔV values are consistent with published literature on GEO station-keeping burns (typically 0.005–0.25 m/s per event).

### 6.3 Software and Reproducibility

The complete implementation is publicly available:
- Repository: https://github.com/keshavgujrathi/OrbitGNN
- Branch: real-tle-pipeline
- Language: Python 3.11, PyTorch 2.x, NumPy, SciPy
- Test suite: 157 automated unit tests, all passing
- CI/CD: GitHub Actions on Python 3.10, 3.11, 3.12

---

## 7. CLAIMS

The present invention claims:

### Claim 1 (Method Claim — Broadest)
A computer-implemented method for detecting satellite orbital manoeuvres, the method comprising:
- (a) receiving Two-Line Element (TLE) records for a plurality of satellites from a public orbital data source;
- (b) computing, for each satellite and each consecutive pair of TLE observations, a physics residual by propagating orbital elements forward using the **actual elapsed time** between the two TLE epochs using a J₂-corrected Keplerian propagation model, and subtracting the propagated elements from the subsequently observed elements;
- (c) encoding temporal sequences of the said physics residuals using a Transformer neural network to produce a latent embedding and a self-forecast of the next residual step per satellite;
- (d) constructing an orbital-shell-aware graph wherein edges connect satellites within the same orbital regime based on orbital similarity, with no edges between satellites in different orbital regimes;
- (e) aggregating peer satellite embeddings via graph convolution over the said orbital graph to produce a peer-context forecast per satellite;
- (f) generating, for each satellite, a composite anomaly score based on weighted combination of the self-forecast error, peer-forecast error, and model uncertainty, wherein each component is normalised across the satellite batch at each time step; and
- (g) flagging satellites whose anomaly score exceeds a detection threshold as having undergone an orbital manoeuvre.

### Claim 2 (Physics Residual with Actual Δt)
The method of Claim 1, wherein step (b) uses the actual elapsed time Δt in seconds between consecutive TLE epochs for propagation, distinct from assuming a fixed nominal time step, thereby eliminating systematic errors in the propagated mean anomaly residual due to irregular TLE publication cadences.

### Claim 3 (Shell-Aware Graph)
The method of Claim 1, wherein step (d) comprises:
- assigning each satellite to an orbital shell from the group consisting of: geostationary orbit (GEO, altitude > 30,000 km), sun-synchronous orbit (SSO, altitude 600–1,000 km and inclination > 88°), and low Earth orbit (LEO, altitude 1,000–1,700 km and inclination 60°–70°);
- computing k-nearest neighbours for each satellite using mean-motion similarity, wherein k is a positive integer; and
- creating adjacency edges only between satellites assigned to the same orbital shell, with no edges between satellites of different shells.

### Claim 4 (Delta-V Estimation without Telemetry)
A method for estimating the delta-velocity (ΔV) of a satellite manoeuvre from publicly available Two-Line Element data only, the method comprising:
- detecting an orbital manoeuvre event at time step t for satellite k using the method of Claim 1;
- extracting the semi-major axis residual Δa_{k,t} in kilometres from the physics residual computed in step (b) of Claim 1;
- computing mean motion n = √(μ/a³) in radians per second from the observed semi-major axis a and the Earth's gravitational parameter μ; and
- estimating the tangential delta-velocity as ΔV ≈ (n · |Δa|) / 2 in metres per second, using the linearised Gauss variational equation for near-circular orbits, without requiring any propulsion telemetry or ground station data.

### Claim 5 (Composite Anomaly Score Formulation)
The method of Claim 1, wherein step (f) computes the anomaly score as:

```
score_k = w₁ · z(||y_k - ŷ^self_k||²) + w₂ · z(||y_k - ŷ^peer_k||²) + w₃ · z(||σ̄_k||²)
```

where z(·) denotes zero-mean, unit-standard-deviation normalisation across the satellite dimension at each time step, ŷ^self is the Transformer self-forecast, ŷ^peer is the GNN peer-context forecast, σ̄_k is the standard deviation of peer forecasts over a plurality of stochastic Monte Carlo Dropout forward passes, and w₁, w₂, w₃ are positive weighting coefficients.

---

## 8. ABSTRACT

The present invention provides **OrbitGNN**, a system and method for automatic satellite manoeuvre detection and delta-velocity (ΔV) estimation using only publicly available Two-Line Element (TLE) orbital data. The method computes J₂-corrected Keplerian physics residuals using the actual inter-TLE elapsed time, encodes residual time-series via a Transformer architecture, aggregates peer satellite context via an orbital-shell-aware Graph Convolutional Network, and generates composite anomaly scores combining self-forecast error, peer-forecast error, and Monte Carlo Dropout uncertainty. A novel ΔV estimator derives manoeuvre magnitude from TLE-derived semi-major axis residuals using the linearised Gauss tangential burn equation, requiring no propulsion telemetry. Evaluated on the real TLE Observation Benchmark Dataset (9 satellites, 2020–2022, 125 verified manoeuvres), OrbitGNN achieves ROC-AUC of 0.587 ± 0.010 and detects 61.3% ± 8.6% of manoeuvre events within ±72 hours — statistically significantly superior to residual-magnitude thresholding (p = 0.0084, Cohen's d = 2.16). The system is self-supervised (requires no labelled training data), causal (usable in real time), and runs on commodity hardware (105,100 parameters). Complete source code and experimental results are publicly available.

---

## DRAWINGS

> **Sheet 1:** System Pipeline Diagram (Figure 1) — TLE input → Physics residuals → ResidualEncoder → NeighborGCN → Anomaly Score → ΔV Estimation

> **Sheet 2:** Orbital Graph Structure (Figure 2) — 9 satellite nodes colour-coded by shell (GEO: gold, SSO: blue, LEO: green), 10 intra-shell edges shown

*(Detailed engineering drawings to be prepared as per Indian Patent Office guidelines for submission)*

---

## REFERENCES

1. Shorten, D. & Abdurahimov, A. (2023). TLE Observation Benchmark Dataset. GitHub. https://github.com/dpshorten/TLE_observation_benchmark_dataset

2. Vallado, D.A. (2013). Fundamentals of Astrodynamics and Applications, 4th ed. Microcosm Press.

3. Vaswani, A. et al. (2017). Attention is All You Need. NeurIPS 30.

4. Kipf, T. & Welling, M. (2017). Semi-supervised Classification with GCN. ICLR.

5. Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. ICML.

6. Patera, R.P. (2008). Satellite manoeuvre detection using TLE data. AIAA.

---

**Date:** _______________

**Signature of Inventor 1:** ___________________  
Name: **Chunduri Sai Kaushik**

**Signature of Inventor 2:** ___________________  
Name: **Keshav Gujrathi**

**Signature of Guide:** ___________________  
Name: **[Guide Name]**, [Designation]

---

*This Provisional Specification is filed to establish priority date. A Complete Specification must be filed within 12 months from the date of this provisional application.*
