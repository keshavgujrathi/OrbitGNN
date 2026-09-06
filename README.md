# OrbitGNN — "Does this satellite still agree with physics, and does it still agree with its neighbors?"

## The one-line pitch

Instead of asking *"can a neural net predict satellite failure?"* (what
everyone else in the room will pitch), OrbitGNN asks a sharper question:

> **What can't gravity explain — and is the satellite alone in that, or is
> its whole orbital shell doing it too?**

That reframing turns your project from "anomaly classifier" into a
mini physics-informed **graph** anomaly detector, which is a genuinely
different animal, still built entirely on public TLE data, still buildable
in a course timeline.

---

## Why this beats the original idea, concretely

| Original idea | What was missing | OrbitGNN's fix |
|---|---|---|
| Physics filters *impossible* predictions | Doesn't use physics as a *signal*, just a sanity clamp | Physics residual (obs − Kepler+J2 prediction) **is** the primary feature, not a filter bolted on at the end |
| One satellite's history vs itself | No notion of "normal for this kind of orbit" | Orbital-neighbor graph — a satellite is compared to its **shell peers**, not just its own past |
| Point-prediction of failure | Sparse, delayed, noisy failure labels (decay can take months) | Self-supervised: train only on "predict next residual," never on failure labels at all — labels only used to *score* you at the end |
| No confidence signal | A model can be confidently wrong | MC-Dropout gives an explicit uncertainty term — "the model doesn't know" *is* an anomaly signal, not a caveat |

Everything above is implemented, not hand-waved. Three files, ~500 lines,
all runnable.

---

## Architecture

```
 per-satellite TLE history
          │
          ▼
 ┌─────────────────────┐
 │   physics.py         │   two-body Kepler + J2 propagator (hand-derived,
 │   (no black box)      │   not sgp4-import — you can explain every line)
 └─────────┬────────────┘
           │  residual(t) = observed(t) − physics_predicted(t)
           ▼
 ┌─────────────────────┐        orbital-neighbor graph
 │ ResidualEncoder       │        (inclination + RAAN + altitude
 │ (small Transformer)   │         similarity, NOT spatial distance)
 └─────────┬────────────┘                 │
           │ latent summary               │
           ▼                              ▼
 ┌───────────────────────────────────────────┐
 │              NeighborGCN                    │  <- satellite's forecast
 │  H' = GELU( D⁻¹ A H W ) + H                 │     is corrected using its
 └─────────┬───────────────────────────────────┘     shell peers' behavior
           │
           ▼
   peer-aware residual forecast
           │
   ┌───────┼────────────────┐
   ▼       ▼                ▼
self-error peer-error   MC-dropout
                          variance
   └───────┴────────────────┘
           ▼
      anomaly_score  (per satellite, per timestep)
```

## Files

- `physics.py` — from-scratch orbital propagator (two-body + J2 secular
  perturbation). Fully tested offline, no dependencies beyond numpy.
- `dataset.py` — (a) real CelesTrak fetcher [`fetch_celestrak_group()`],
  (b) synthetic multi-shell constellation simulator with **ground-truth
  labeled anomalies** (drag decay, maneuvers, tumble events) for training
  and evaluation, (c) orbital-neighbor graph builder. Tested offline.
- `model.py` — `OrbitGNN`: ResidualEncoder (Transformer) + NeighborGCN
  (hand-written graph conv, no `torch_geometric` needed) + MC-Dropout
  uncertainty head + `anomaly_score()`.
- `train.py` — self-supervised training loop (trains only on nominal
  windows) + held-out ROC-AUC / precision@k evaluation.

## Running it

```bash
pip install -r requirements.txt
python dataset.py          # smoke test: physics + data pipeline (offline, no torch needed)
python train.py            # trains OrbitGNN, prints ROC-AUC + precision@20,
                            # and saves two charts into results/:
                            #   results/anomaly_timeline.png  (score vs. injected events)
                            #   results/roc_curve.png
```

For the live demo, open `notebooks/demo.ipynb` instead — it runs the same
pipeline and displays both charts inline, plus an optional cell that pulls
real live Starlink data from CelesTrak.

To point it at real Starlink data instead of the synthetic simulator, call
`fetch_celestrak_group("starlink")` on a schedule (e.g. every 6–12h for a
week or two — that's roughly how often TLE epochs actually update per
satellite) and feed the accumulated history into `compute_residual_sequences`
in place of the synthetic array. No other code changes needed — that's the
whole point of keeping the interface as plain `(S, T, 6)` arrays.

## For the 10-minute talk

1. **Hook (30s):** "Everyone doing space this semester will use Kepler's
   laws as a filter. We use it as a *sensor*." Show the residual equation.
2. **The graph idea (2 min):** show the shell-vs-neighbor diagram. One
   sentence that lands: *"A satellite that drifts with its whole shell is
   probably just solar weather. A satellite that drifts alone is the one
   you call an engineer about."*
3. **Live demo (3–4 min):** run `train.py` in front of them (or show the
   pre-run output + the ROC curve). Point at a specific injected maneuver
   event and show the anomaly score spiking on that exact timestep.
4. **The uncertainty punchline (1 min):** "The model doesn't just say
   'anomaly' — it says 'anomaly, and here's how sure I am,' which is the
   difference between a lab curiosity and something you'd actually trust
   to page someone at 3am."
5. **Close (30s):** what you'd do with more time — real Space-Track
   conjunction data (CDMs) as an extra supervised signal, extending the
   graph to cross-shell debris-conjunction edges.

## Honest scope note (say this if asked — it builds credibility, not doubt)

Real orbital anomalies (actual decay, actual failures) are rare and the
labels are slow/sparse in the real world, which is *why* the synthetic
injected-anomaly benchmark exists — it's standard practice in anomaly
detection research when ground truth is scarce, not a shortcut. The real
CelesTrak pull is there to show the pipeline works on live data; the ROC-AUC
number you report should come from the synthetic ground-truth benchmark
where you actually know what's true.
