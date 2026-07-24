# Final-week hardware campaign (July 22 to 24, 2026)

BasQ extended the hackathon's IBM Quantum access to July 24. We spent the
remaining allocation on a systematic campaign: close to 90 jobs, roughly
3,900 QPU-seconds, five quantum computers, three processor families. Every
job ID, expectation value, and per-point uncertainty is in this repository
(`data/final_runs_jobs.json`, `data/final_burn_jobs.json`,
`data/burn_evs_*.npy`, `data/burn_stds_*.npy`), and every number below can
be recomputed offline from those files with `burn_collect.py`.

## Headlines

| Experiment | Result |
|---|---|
| Full 200-point test set, tiled, `ibm_fez` | **20.78 K** vs 19.83 K ideal (r = 0.991) |
| The whole cleaned dataset: 15,170 materials, `ibm_fez` | test split **21.24 K** (r = 0.990) |
| The whole dataset again on the next processor generation, `ibm_pittsburgh` | test split **21.18 K** (r = 0.991) |
| Preregistered 300-material holdout, `ibm_fez` | **23.20 K** vs 21.30 K ideal committed before the run ([HOLDOUT_PREREG.md](HOLDOUT_PREREG.md)) |
| 4,000 generated candidates screened on hardware | rank agreement with simulation **rho = 0.98**; the top 10 are all Hg-cuprates ([candidates_screen.csv](data/candidates_screen.csv)) |

The candidate screen deserves one sentence of context: the model, running on
a quantum computer over hypothetical formulas it had never seen, put the
mercury-cuprate family at the top of its shortlist. That family holds the
real-world ambient-pressure Tc record.

## Replication: one circuit, five quantum computers

Identical circuit, weights, and settings; tiled 200-point test set.

| Backend | Family | RMSE | r vs ideal |
|---|---|---|---|
| `ibm_fez` | Heron r2 | 20.78 K | 0.991 |
| `ibm_kingston` | Heron r2 | 21.02 K | 0.992 |
| `ibm_marrakesh` | Heron r2 | 21.52 K | 0.987 |
| `ibm_pittsburgh` | Heron r3 | 20.75 K | 0.992 |
| `ibm_miami` | Nighthawk | 21.71 K | 0.989 |

The layout is recomputed from live calibration on every device; no per-device
tuning of any kind. On the Nighthawk's square lattice the packer found 15
tiles instead of 25 and still transpiled SWAP-free at depth 21. A sixth
device, `ibm_boston`, accepted the job but its public queue (480+ pending)
never reached it before access ended; the job ID is in the registry.

Day-to-day stability on `ibm_fez`, identical anchor job: 20.78 K (Jul 22),
20.84 K and 20.84 K (Jul 23, twice). The honest drift bound for this setup
is about 0.1 K, far tighter than the 0.7 K anecdote from July 11.

## Hardware-awareness, measured instead of asserted

- **Per-tile error map.** Ten identical inputs broadcast to all 25 tiles in
  one job: per-tile RMSE correlates with the tile's summed calibration error
  score at **r = 0.99** (bias at r = 0.97). IBM's published calibration data
  predicts realized per-region error almost perfectly.
- **Tile selection ablation.** The 12 best-ranked tiles score 20.15 K
  (r = 0.999); the 12 worst score 20.60 K (r = 0.972); all 25 score 20.78 K.
  Pruning to good tiles is a free accuracy upgrade, and placement quality is
  worth about half a Kelvin end to end.

## Depth and error mitigation, closed out

- At 6 CZ (3 layers), full mitigation lands exactly on the ideal: 23.28 K vs
  23.28 K (r = 1.000) on the frozen 50. Unmitigated is 22.85 K. Doubling
  depth opens no hardware gap on today's Herons, and on the full test set
  the 3-layer model runs at 21.15 K vs 20.31 K ideal.
- The 2-layer PCA cell (29.87 K vs 30.43 K ideal) completes the
  depth-by-features factorial: feature choice, not circuit depth, explains
  the accuracy spread between configurations.
- Every mitigation family in Qiskit Runtime is now measured on this circuit:
  TREX readout mitigation recovers the ideal alone; DD, gate twirling, tuned
  ZNE (noise factors 1/3/5, gate folding, exponential extrapolation, 2.4x
  cost), and PEC (23.03 K vs 23.00 K ideal, r = 1.000, at 231 QPU-seconds
  against TREX's 41) add nothing below 6 CZ. Readout error is the entire
  budget at this depth.
- Shot ladder at 64/256/1024 shots: 23.75/24.01/23.54 K against the 23.0 K
  ideal. The model sits near its noise floor even at 64 shots; per-job
  overhead, not shots, dominates cost below about 1024.

## Honesty checks on the model itself

- **Multiseed on hardware.** Seeds 7/17/27 score 20.78/21.47/21.54 K against
  ideals of 19.83/20.13/20.13 K: the hardware gap is seed-independent, so
  the headline is not an optimizer accident. In simulation, seeds 17 and 27
  reach identical accuracy from weight vectors 3.3 radians apart, so the
  training landscape is degenerate and robust.
- **No-entanglement control.** The same circuit with its 3 CZ gates deleted,
  retrained from scratch: 20.09 K in simulation (champion 19.83 K) and
  23.65 K on hardware frozen-50 (champion 22.90 K). Entanglement contributes
  a few tenths of a Kelvin here, not the headline. The claim of this project
  is parameter efficiency and hardware viability, not quantum advantage, and
  this control is why we can say that precisely.

## Hardware-in-the-loop fine-tuning

We ran SPSA on the quantum computer itself: each iteration evaluates the
loss at two perturbed weight vectors inside ONE tiled job (both arms share a
calibration; shot noise partially cancels in the gradient), checkpointing
every step. 46 iterations completed in a Qiskit Runtime Session on
`ibm_fez` before the budget floor stopped the loop.

**Verdict (same job, same calibration, full 200-point test set): original
weights 21.46 K, fine-tuned weights 21.01 K.** Forty-six SPSA iterations
against the live device improved the model by 0.45 K under identical
conditions. The day's calibration ran slightly hot (the anchors put the
same original weights at 20.78 to 20.84 K on the two prior days), which is
exactly why the A/B shares one job: the comparison is calibration-free.
The improvement was learned entirely from 150 training points evaluated on
hardware; the test set was never touched during fine-tuning. Weights:
`data/hw_finetune_weights.npy`; per-iteration log with every job ID:
`data/hw_finetune_log.jsonl`.

## Cost ledger and lessons

- Registry jobs: 37 submitted, 36 executed, 1,945 QPU-seconds total.
- Fine-tuning: about 1,900 QPU-seconds. Lesson learned the expensive way:
  Runtime Sessions bill wall-clock while the session is open, so an
  iterative loop pays roughly double its pure execution time.
- The Nighthawk (`ibm_miami`) is about 15x slower per shot than a Heron:
  its single 200-point job cost 249 QPU-seconds against 16 on `ibm_fez`.
- One IBM API outage (July 23, several hours, undeclared) was ridden out by
  idempotent submission scripts that retry until the write path returns.
