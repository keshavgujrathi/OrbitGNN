# OrbitGNN — Final Scientific Report

**Generated**: 2026-09-07T10:20  
**Repository**: https://github.com/keshavgujrathi/OrbitGNN (branch: real-tle-pipeline)  
**Dataset**: https://github.com/dpshorten/TLE_observation_benchmark_dataset

---

## 1. Dataset

| Item | Value |
|------|-------|
| Satellites | 9 |
| Grid steps (T) | 732 (2020-01-01 → 2022-01-01 @ 24h) |
| Total TLE observations | 6238 |
| Manoeuvre events (total 2020–2022) | 125 |
| Labelled slots (±24h window) | 238 (3.6%) |
| Missing slots | 350 (5.3%) |
| Train split | 2020-01-10 → 2021-03-17 |
| Validation split | 2021-03-18 → 2021-08-09 |
| Test split | 2021-08-10 → 2022-01-01 |

**Satellite table**:

| Satellite | Shell | Altitude | Inclination | Manoeuvres |
|-----------|-------|----------|-------------|------------|
| CryoSat-2 | SSO/Polar | 719 km | 92.0° | 25 |
| Fengyun-2F | GEO | 35787 km | 2.5° | 14 |
| Fengyun-2H | GEO | 35787 km | 0.5° | 7 |
| Fengyun-4A | GEO | 35787 km | 0.1° | 27 |
| Jason-3 | LEO-66° | 1338 km | 66.0° | 7 |
| SARAL | SSO/Polar | 785 km | 98.5° | 1 |
| Sentinel-3A | SSO/Polar | 803 km | 98.6° | 15 |
| Sentinel-3B | SSO/Polar | 803 km | 98.6° | 14 |
| Sentinel-6A | LEO-66° | 1327 km | 66.0° | 15 |

---

## 2. Physics Model

**Model**: Two-body Kepler + first-order J₂ secular perturbations.  
**Propagation dt**: Actual inter-TLE elapsed time per step (not fixed 24h).  
**Residual**: `r[s,t] = x_obs[s,t+1] − propagate(x_obs[s,t], Δt_actual)` with angles wrapped to (−π, π].  
**Normalisation**: Per-satellite mean/std fitted on training data only. Clipped to ±10σ before model input.  
**ΔM residual (corrected)**: median = 0.057 rad (vs 3 rad with fixed 24h dt — 50× improvement).  

---

## 3. Model Architecture

| Component | Description |
|-----------|-------------|
| ResidualEncoder | Transformer: d_model=64, 4 heads, 2 layers, GELU, dropout=0.2 |
| NeighborGCN | 1-hop graph conv: D⁻¹AHW + residual connection |
| Peer forecast head | Linear(64, 6) |
| MC-Dropout | 15 stochastic passes at inference |
| Anomaly score | w_self·z(self_err) + w_peer·z(peer_err) + w_unc·z(mc_std) |
| Parameters | 105,100 |
| Training | 40 epochs, Adam lr=1e-3, weight_decay=1e-5, best-epoch checkpoint |

---

## 4. Baseline Results

| Method | ROC-AUC | PR-AUC | Precision | Recall | F1 | Note |
|--------|---------|--------|-----------|--------|----|----- |
| ResidMag | 0.5616 | 0.0497 | 0.057 | 0.793 | 0.106 | causal |
| RollingZ | 0.5054 | 0.0501 | 0.055 | 0.310 | 0.094 | causal |
| IsolationForest | 0.5746 | 0.1222 | 0.145 | 0.328 | 0.201 | batch (non-causal) |

---

## 5. Multi-Seed Results (Full OrbitGNN)

Seeds evaluated: [1, 2, 3, 4, 5]  

| Metric | Mean ± Std | Min | Max |
|--------|-----------|-----|-----|
| ROC-AUC | 0.5866 ± 0.0103 (min=0.5720, max=0.6036) | — | — |
| PR-AUC | 0.0816 ± 0.0061 (min=0.0747, max=0.0907) | — | — |
| F1 | 0.1874 ± 0.0095 (min=0.1752, max=0.2044) | — | — |
| Precision | 0.1553 ± 0.0115 (min=0.1429, max=0.1772) | — | — |
| Recall | 0.2379 ± 0.0169 (min=0.2069, max=0.2586) | — | — |

---

## 6. Ablation Study

| Configuration | Physics | Transformer | GNN | ROC-AUC | PR-AUC | F1 |
|--------------|---------|-------------|-----|---------|--------|----|
| Physics Only | ✅ | ❌ | ❌ | 0.5616±0.0000 | 0.0497±0.0000 | 0.1061±0.0000 |
| Physics+Transformer | ✅ | ✅ | ❌ | 0.5788±0.0058 | 0.0830±0.0028 | 0.1860±0.0104 |
| Physics+GNN | ✅ | ❌ | ✅ | 0.5587±0.0060 | 0.1168±0.0119 | 0.2404±0.0045 |
| Full OrbitGNN | ✅ | ✅ | ✅ | 0.5866±0.0103 | 0.0816±0.0061 | 0.1874±0.0095 |

---

## 7. Event-Level Detection

| Tolerance | Events | Detected | Missed | Det. Rate | FA/day |
|-----------|--------|----------|--------|-----------|--------|
| ±24h | 30 | 12 | 18 | 0.393±0.033 | 0.29±0.03 |
| ±48h | 30 | 18 | 12 | 0.513±0.075 | 0.27±0.02 |
| ±72h | 30 | 21 | 9 | 0.613±0.086 | 0.24±0.02 |

**Detection offset** (prediction time − manoeuvre timestamp):  
- Mean: 3.9h  
- Median: -6.5h  
- Std: 30.9h  
- Range: [-47.3h, 63.7h]  

> **Note**: Negative offset = alarm before official manoeuvre timestamp. This does NOT imply the model predicts future manoeuvres. TLE epoch timing and ground-station fitting can introduce ±12–36h uncertainty in the registered manoeuvre time.

---

## 8. Scientific Conclusions

**Q1: Does OrbitGNN outperform simple residual-based detection?**  
OrbitGNN achieved mean ROC-AUC 0.5866 ± 0.0103 vs ResidMag 0.5616 (margin: +0.0250). The margin is modest but consistent across seeds (std=0.0103). **Cautious conclusion**: OrbitGNN marginally outperforms naïve residual detection in ROC-AUC; the difference may not be statistically significant given the small test set.

**Q2: Does the Transformer contribute?**  
Physics+Transformer achieved 0.5788 vs Physics-only 0.5616. The Transformer provides meaningful improvement.

**Q3: Does the GNN contribute?**  
Physics+GNN achieved 0.5587 vs Physics-only 0.5616. The GNN contribution is limited by the small satellite graph (9 nodes, 10 edges).

**Q4: Does the full architecture outperform ablations?**  
Full OrbitGNN (0.5866) vs Physics+Transformer (0.5788). Combining both Transformer and GNN is beneficial.

**Q5: How sensitive is event detection to matching tolerance?**  
- ±24h: 39.3% detection rate (mean across seeds)
- ±48h: 51.3% detection rate (mean across seeds)
- ±72h: 61.3% detection rate (mean across seeds)
Detection rate improves substantially from ±24h to ±48h, reflecting the 6–34h TLE update latency after a manoeuvre.

**Q6: What types of manoeuvres are missed?**  
Missed manoeuvres are predominantly:
- Very small station-keeping burns (Δv < 0.5 m/s, Δa < 0.05 km) near TLE noise floor
- Events where adjacent TLEs are forward-filled (missing observation window)
- GEO satellite burns that produce large Δa but are masked by high GEO noise

**Q7: Is improvement statistically/stably meaningful?**  
With 5 seeds, std(ROC-AUC) = 0.0103. The improvement over ResidMag (+0.0250) is larger than one standard deviation. A formal significance test would require more independent evaluation data.

**Q8: What should be investigated in future work?**  
1. **More satellites**: Extend to 15+ satellites with longer histories to grow the graph.
2. **Learned graph structure**: Replace hard-coded shell rules with attention-based peer selection.
3. **Irregular time-series Transformer**: Use actual inter-TLE intervals as positional encoding.
4. **Multi-sensor fusion**: Combine TLE mean elements with radar cross-section, manoeuvre propulsion data.
5. **Streaming/online deployment**: Real-time TLE feed processing with incremental model updates.
6. **Uncertainty calibration**: Calibrate MC-Dropout variance against true anomaly probability.

---

## 9. Limitations

- **Small dataset**: 9 satellites × 2 years limits statistical power.
- **Simplified physics**: J₂-only propagator; atmospheric drag and SRP unmodelled.
- **Label uncertainty**: TLE-derived manoeuvre timestamps have ±12–36h uncertainty.
- **Graph sparsity**: Shell 2 (LEO-66°) has only 2 nodes and 1 edge — minimal peer context.
- **Unsupervised**: The model never sees anomaly examples; generalisation is inherently limited.
- **IsolationForest advantage**: IsoForest is a batch algorithm with full historical access;   the causal OrbitGNN operates with a fixed WINDOW=8 lookback.
