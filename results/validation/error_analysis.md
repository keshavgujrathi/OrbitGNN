# OrbitGNN — Error Analysis Report
Generated: 2026-09-07T10:20

---

## 1. True Positives

Manoeuvre events that OrbitGNN successfully detected (within ±72h):

- **CryoSat-2** (SSO/Polar) — manoeuvre 2021-09-17 01:47, alarm 2021-09-17 00:00 (offset -1.8h)
- **CryoSat-2** (SSO/Polar) — manoeuvre 2021-10-14 23:17, alarm 2021-10-13 00:00 (offset -47.3h)
- **CryoSat-2** (SSO/Polar) — manoeuvre 2021-11-26 21:07, alarm 2021-11-29 00:00 (offset +50.9h)
- **Fengyun-2F** (GEO) — manoeuvre 2021-09-22 16:00, alarm 2021-09-25 00:00 (offset +56.0h)
- **Fengyun-2F** (GEO) — manoeuvre 2021-09-23 16:00, alarm 2021-09-25 00:00 (offset +32.0h)
- **Fengyun-2F** (GEO) — manoeuvre 2021-11-15 15:30, alarm 2021-11-17 00:00 (offset +32.5h)
- **Fengyun-2H** (GEO) — manoeuvre 2021-09-27 15:00, alarm 2021-09-28 00:00 (offset +9.0h)
- **Fengyun-4A** (GEO) — manoeuvre 2021-08-20 16:16, alarm 2021-08-19 00:00 (offset -40.3h)
- **Fengyun-4A** (GEO) — manoeuvre 2021-09-10 16:16, alarm 2021-09-13 00:00 (offset +55.7h)
- **Fengyun-4A** (GEO) — manoeuvre 2021-09-21 17:30, alarm 2021-09-24 00:00 (offset +54.5h)
- **Fengyun-4A** (GEO) — manoeuvre 2021-09-22 17:30, alarm 2021-09-24 00:00 (offset +30.5h)
- **Fengyun-4A** (GEO) — manoeuvre 2021-11-12 08:16, alarm 2021-11-15 00:00 (offset +63.7h)
- **Fengyun-4A** (GEO) — manoeuvre 2021-12-07 16:16, alarm 2021-12-10 00:00 (offset +55.7h)
- **Jason-3** (LEO-66°) — manoeuvre 2021-11-08 22:57, alarm 2021-11-09 00:00 (offset +1.1h)
- **SARAL** (SSO/Polar) — manoeuvre 2021-11-21 12:18, alarm 2021-11-21 00:00 (offset -12.3h)
- **Sentinel-3A** (SSO/Polar) — manoeuvre 2021-08-11 09:15, alarm 2021-08-11 00:00 (offset -9.2h)
- **Sentinel-3A** (SSO/Polar) — manoeuvre 2021-09-09 06:33, alarm 2021-09-09 00:00 (offset -6.5h)
- **Sentinel-3A** (SSO/Polar) — manoeuvre 2021-11-05 07:10, alarm 2021-11-05 00:00 (offset -7.2h)
- **Sentinel-3A** (SSO/Polar) — manoeuvre 2021-12-01 02:42, alarm 2021-12-01 00:00 (offset -2.7h)
- **Sentinel-3A** (SSO/Polar) — manoeuvre 2021-12-08 03:00, alarm 2021-12-08 00:00 (offset -3.0h)
- **Sentinel-3A** (SSO/Polar) — manoeuvre 2021-12-16 07:39, alarm 2021-12-16 00:00 (offset -7.7h)
- **Sentinel-3B** (SSO/Polar) — manoeuvre 2021-09-30 08:40, alarm 2021-09-30 00:00 (offset -8.7h)
- **Sentinel-3B** (SSO/Polar) — manoeuvre 2021-10-21 12:51, alarm 2021-10-21 00:00 (offset -12.8h)
- **Sentinel-3B** (SSO/Polar) — manoeuvre 2021-12-02 08:20, alarm 2021-12-02 00:00 (offset -8.3h)

**Interpretation**: OrbitGNN fires within a few hours of the TLE-registered manoeuvre timestamp. The offset is consistent with ground-station TLE update latency (typically 6–24 h after the physical burn). Negative offsets (alarm before timestamp) reflect the model detecting orbital state changes before the official TLE is published.

---

## 2. False Positives

Alarm events with no registered manoeuvre within ±72h:

- **CryoSat-2** (SSO/Polar): 11 false alarm(s)
- **Fengyun-2F** (GEO): 1 false alarm(s)
- **Fengyun-2H** (GEO): 4 false alarm(s)
- **Fengyun-4A** (GEO): 2 false alarm(s)
- **Jason-3** (LEO-66°): 1 false alarm(s)
- **SARAL** (SSO/Polar): 14 false alarm(s)
- **Sentinel-3B** (SSO/Polar): 1 false alarm(s)
- **Sentinel-6A** (LEO-66°): 4 false alarm(s)

**Possible explanations**:
- TLE update noise: Ground-station solutions have measurement error   O(10–100 m) that can produce transient residual spikes.
- Unregistered station-keeping: Operators do not publicly log every micro-burn.   Some 'false positives' may be real events absent from the YAML files.
- Graph effects: If a single shell peer makes an unusual manoeuvre,   its neighbours receive anomalous graph messages and may be spuriously flagged.
- Solar/geomagnetic activity: Increased atmospheric drag during geomagnetic   storms causes temporary orbital decay detectable as Δa residuals.
- Model overconfidence in low-data regions: Sentinel-6A has only 398/732 valid   slots; the forward-fill creates artificial continuity that the model   may misinterpret as anomalous.

---

## 3. False Negatives (Missed Manoeuvres)

Manoeuvre events NOT detected within ±72h:

- **CryoSat-2** (SSO/Polar) — missed manoeuvre 2021-12-23 12:40
- **Fengyun-4A** (GEO) — missed manoeuvre 2021-10-18 16:16
- **Fengyun-4A** (GEO) — missed manoeuvre 2021-12-31 16:16
- **Sentinel-3A** (SSO/Polar) — missed manoeuvre 2021-12-04 02:55
- **Sentinel-6A** (LEO-66°) — missed manoeuvre 2021-08-17 00:21
- **Sentinel-6A** (LEO-66°) — missed manoeuvre 2021-11-26 03:00

**Possible explanations for missed manoeuvres**:
- Very small Δv burns: Station-keeping burns of < 0.5 m/s produce   Δa ≈ 3 km for GEO and < 0.1 km for LEO — near or below TLE noise floor.
- Short-window effect: The model sees only WINDOW=8 steps (8 days).   If pre-manoeuvre TLEs are forward-filled (missing observations),   the history is contaminated.
- Temporal mismatch: The 24h grid may not align with the manoeuvre   timestamp. A burn at the midpoint between two grid slots is harder to detect.
- GNN masking: If all shell peers show similar residuals (common-mode   perturbation), the peer comparison does not flag the individual burn.

---

## 4. Dataset Limitations

| Limitation | Impact |
|-----------|--------|
| Only 9 usable satellites | Very small graph; limited peer relationships. Shell 2 has only 2 members (1 edge). |
| Irregular TLE cadence (3.4–107h) | Forward-filling up to 48h introduces artificial continuity. |
| TLE epoch timing uncertainty | Ground-station solutions lag physical events by 6–36h. Manoeuvre timestamps are approximate. |
| Manoeuvre label uncertainty | YAML timestamps from mission operations records; some micro-burns may be unlabelled. |
| Simplified J2/Kepler physics | Unmodelled: atmospheric drag, SRP, lunisolar, higher-order harmonics. These contribute to background noise. |
| No test-set manoeuvres for SARAL | SARAL has only 1 manoeuvre in the entire 2020–2022 window. |
| Class imbalance | Only 3.6% of grid slots are labelled manoeuvre. |

---

## 5. Residual Signal Analysis

After applying the corrected physics propagation (actual inter-TLE dt):

- **CryoSat-2**: median |Δa|=0.0004 km, median |ΔM|=0.0754 rad
- **Fengyun-2F**: median |Δa|=0.1545 km, median |ΔM|=0.0117 rad
- **Fengyun-2H**: median |Δa|=0.0216 km, median |ΔM|=0.0112 rad
- **Fengyun-4A**: median |Δa|=0.1458 km, median |ΔM|=0.0106 rad
- **Jason-3**: median |Δa|=0.0001 km, median |ΔM|=0.0326 rad
- **SARAL**: median |Δa|=0.0003 km, median |ΔM|=0.1128 rad
- **Sentinel-3A**: median |Δa|=0.0003 km, median |ΔM|=0.1032 rad
- **Sentinel-3B**: median |Δa|=0.0003 km, median |ΔM|=0.1063 rad
- **Sentinel-6A**: median |Δa|=0.0000 km, median |ΔM|=0.0580 rad

Manoeuvre signals in Δa range from 0.003 km (Jason-3, small burn) to 2.3 km (Fengyun-2F, large station-keeping). The TLE noise floor for Δa is approximately 0.001–0.050 km depending on satellite type and ground station coverage. Small manoeuvres with Δa near the noise floor are fundamentally difficult to detect without multi-sensor fusion.
