# Superconductor@home — distributed uncertainty-aware screening

A **citizen-science compute** extension to this project: volunteers donate spare
CPU to run massively parallel, **uncertainty-aware** virtual screening of
candidate superconductors, feeding an expert-reviewed synthesis funnel.

`superconductor_at_home.py` is a runnable prototype of **one volunteer work
unit** — the exact computation a single donated machine would perform.

## Why *inference*, not training, is the right citizen-science task
Training a QNN has a per-step synchronisation barrier that volunteer grids
handle badly. **Inference is embarrassingly parallel** — every candidate material
is scored independently, so the workload is latency- and straggler-tolerant, the
classic BOINC / Folding@home shape. Two more properties make it a clean fit:

- **Deterministic → trivially verifiable.** Classical inference is reproducible,
  so the server re-runs a random sample of each returned work unit and checks a
  result hash — errors or cheating are caught for free.
- **No privacy problem.** The model and candidate generator are public; it is a
  public-good screen.

## What one work unit does (`superconductor_at_home.py`)
For each candidate composition:

1. **Featurize** the formula → 81 UCI features (via `featurize_formula.py`) →
   a **hybrid 8-feature encoding** (4 physics features + 4 PLS components) — a
   richer input space than the shipped PLS-4 champion QNN.
2. **Ensemble predict** with `K = 12` models → **mean Tc + spread (uncertainty)**.
   In production each ensemble member is a QNN variant; here fast bootstrap
   surrogates on the same features demonstrate the pipeline.
3. **OOD / novelty score** = distance to the training distribution; candidates
   beyond the 99th-percentile distance are flagged untrustworthy and dropped.
4. **Acquisition score** `LCB = mean_Tc − uncertainty` — reward high Tc, *penalise
   uncertain extrapolation*. Return the ranked in-distribution shortlist + a hash.

## Why uncertainty is the crux (and why it justifies a grid)
A single-model screen is **too cheap to need volunteers** *and* scientifically
dishonest — it hands you confidently-wrong extrapolations. Making each candidate
cost an **ensemble** (plus optional noise Monte-Carlo) does two things at once:
it multiplies the per-candidate cost by 10³–10⁴× (so distributing is genuinely
worthwhile) **and** produces the **calibrated uncertainty** that out-of-distribution
triage requires.

The prototype makes this concrete: **high-Tc predictions carry ~2× the
uncertainty of the rest** — the model is uncertain *exactly* where it predicts a
high Tc. Ranking by the lower confidence bound is what stops that from becoming
false hope sent to a lab.

## What the prototype shows (from a run of 4,000 candidates)
- **Directed search beats brute force.** Family-directed (cuprate / iron-based)
  candidates average ~40 K predicted Tc vs ~9 K for random compositions; ~40% of
  all candidates are flagged out-of-distribution and dropped.
- **It re-discovers the champion chemistry.** With no labels, the top of the
  acquisition-ranked shortlist is dominated by **Hg–Ba–Cu–O** compositions
  (predicted ~85–92 K) — the real-world record-holding cuprate family. A clean
  sanity check that the funnel works.

## Honest limits
- **Bounded by predictor reliability.** The screen is only as good as the model,
  and the model is weakest out-of-distribution — which is where discovery lives.
  Uncertainty filtering *manages* this; it does not remove it.
- **The real bottleneck is the wet lab.** Distributed compute amplifies reach and
  quantifies confidence; it does **not** replace synthesis and measurement. This
  is a **triage funnel**, not a discovery oracle.

## Run it
```bash
# raw UCI CSVs in data/ (see README.md); element table is already cached
python superconductor_at_home.py
```
Output: a printed shortlist + `data/screening_workunit_results.csv` — the result
a volunteer node would return to the server.
