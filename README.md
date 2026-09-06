# OrbitGNN: Physics-Residual Graph Anomaly Detection for Satellite Constellations

OrbitGNN is a deep learning system for detecting anomalies in satellite orbital behavior. It combines a hand-derived orbital mechanics propagator with a graph-based neural network to identify satellites that deviate from both their own expected physical trajectory and the behavior of their orbital-plane peers.

---

## Motivation

Conventional approaches to satellite anomaly detection typically use orbital mechanics only as a validity filter — discarding physically impossible predictions after the fact. OrbitGNN instead treats orbital mechanics as a primary feature source: the residual between a satellite's observed state and its physics-predicted state becomes the input signal for anomaly detection, rather than a post-hoc sanity check.

The system is further motivated by the observation that satellites do not operate in isolation — constellation satellites share orbital shells with many peers, and this structure has diagnostic value: deviations shared across an entire shell (e.g., from atmospheric drag driven by solar activity) are less significant than deviations isolated to a single satellite.

---

## Approach

| Limitation of Baseline Approach                            | OrbitGNN Design                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Physics used only to reject impossible outputs             | Physics residual (observed − Kepler+J2 prediction) is the primary input feature |
| Anomaly defined relative to a satellite's own history only | Orbital-neighbor graph compares each satellite to its shell peers               |
| Requires supervised failure labels (sparse and delayed)    | Self-supervised training on nominal residual forecasting                        |
| No prediction confidence                                   | Monte Carlo dropout provides predictive uncertainty                             |

---

## Architecture

```
 per-satellite orbital element history
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
   (neighbors defined by inclination / RAAN / altitude
    similarity, not spatial proximity)
                 │
                 ▼
      peer-aware residual forecast
                 │
   ┌─────────────┼──────────────────┐
   ▼             ▼                  ▼
 self-forecast  peer-forecast   MC-dropout
 error          error           predictive variance
   └─────────────┴──────────────────┘
                 ▼
          combined anomaly score
```

---

## Repository Structure

```
orbitgnn/
├── physics.py        # Two-body + J2 orbital propagator
├── dataset.py        # Data pipeline and simulation
├── model.py          # OrbitGNN architecture
├── train.py          # Training and evaluation
├── notebooks/
│   └── demo.ipynb    # End-to-end demo
├── results/          # Generated outputs
├── requirements.txt
└── LICENSE
```

---

## Module Details

### `physics.py`

Implements a two-body Keplerian propagator with J2 secular perturbation corrections, derived from orbital mechanics rather than external libraries (e.g., `sgp4`). This ensures interpretability and generates the residual signal used throughout the pipeline.

### `dataset.py`

Provides two data paths:

* **`fetch_celestrak_group()`**
  Retrieves live TLE/GP data from CelesTrak’s public API.

* **`simulate_constellation()`**
  Generates a synthetic multi-shell constellation with labeled anomalies:

  * Drag-induced decay
  * Station-keeping maneuvers
  * Attitude/tumble events

* **`build_orbital_neighbor_graph()`**
  Constructs adjacency matrices based on orbital-plane similarity.

---

### `model.py`

Defines `OrbitGNN`, composed of:

* Transformer-based residual encoder
* Custom graph convolution layer (no `torch_geometric`)
* MC-dropout uncertainty estimator
* Combined anomaly scoring function

---

### `train.py`

* Self-supervised training on nominal residual windows
* Evaluation using:

  * ROC-AUC
  * Precision@k

Outputs:

* `results/anomaly_timeline.png`
* `results/roc_curve.png`

---

## Installation and Usage

```bash
pip install -r requirements.txt

python dataset.py     # Verify physics and data pipeline
python train.py       # Train model and generate results
```

Alternatively:

* Run `notebooks/demo.ipynb` for a full walkthrough
* Includes optional live CelesTrak data integration

---

## Using Real Data

To replace synthetic data:

1. Call `fetch_celestrak_group()` regularly (daily updates)
2. Accumulate history
3. Pass into `compute_residual_sequences()`

No further pipeline changes required — input format remains `(S, T, 6)`.

---

## Evaluation Methodology

Real satellite anomaly labels are:

* Rare
* Delayed (weeks/months)

Therefore:

* Training and evaluation are performed on **synthetic constellations**
* Injected anomalies provide controlled ground truth

Live data support is included for real-world validation.

---

## Possible Extensions

* Integrate Space-Track CDMs as supervised signals
* Add cross-shell edges based on conjunction risk
* Evaluate detection performance per anomaly type

---

## License

MIT — see `LICENSE`.
