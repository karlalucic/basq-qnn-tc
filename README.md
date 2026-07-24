# OpenSuperLab: a hardware-aware QNN for superconductor Tc prediction

**Winning entry, CERN Quantum Materials Hackathon 2026 (BasQ challenge)**

We trained a 32-parameter quantum neural network to predict superconductor
critical temperatures, designed the circuit around a real IBM Heron chip, and
built a screening instrument on top of the trained model:
[opensuperlab.vercel.app](https://opensuperlab.vercel.app), with a public
citizen-science mode at [/contribute](https://opensuperlab.vercel.app/contribute).
The app runs the model in the browser; its source is maintained separately.

## Results

| What | Measured |
|---|---|
| Champion QNN vs parameter-matched MLP (3 seeds, identical inputs) | **19.84 ± 0.03 K** vs 26.7 ± 4.6 K test RMSE |
| Ideal simulator → noisy simulator → real QPU (`ibm_fez`, same 50 materials) | 23.00 → 22.86 → **22.89 K** (r = 0.999) |
| QPU time per 50-material screen (25 circuit copies tiled on the 156-qubit chip) | 41 → **12 s** (+0.55 K) |
| Selection quality on held-out materials (top-10 vs random pick) | 8/10 vs 3.8, a **2.1× enrichment** |
| Full 200-material test set on hardware, replicated on five devices | **20.75 to 21.71 K** vs 19.83 K ideal (r = 0.99 on all five) |
| The whole cleaned dataset, 15,170 materials, on two processor generations | test split **21.24 / 21.18 K** (`ibm_fez` / `ibm_pittsburgh`) |
| Preregistered 300-material fresh holdout on hardware, never touched before the run | **23.20 K** vs 21.30 K ideal, committed in advance |
| 4,000 generated candidate materials screened on hardware | rank agreement with simulation **rho = 0.98**; top 10 all Hg-cuprates |

The champion circuit: 4 qubits × 2 layers, 32 parameters, supervised PLS-4
inputs, one chain of 3 CZ gates. It transpiles to ISA depth 21 with zero SWAPs
on IBM Heron. Gradient boosting trained on the full 12k-row training split of
the same four inputs reaches 15.9 K; we report it because the claim here is
parameter efficiency and hardware viability, not best-in-class accuracy.

When BasQ extended QPU access through July 24 we ran a closing campaign:
close to 90 jobs on five IBM devices across three processor families,
including a preregistered holdout (hashes and ideal predictions committed
before submission), a five-device replication, a per-tile error map that
shows IBM's calibration data predicting realized error at r = 0.99, the
complete error-mitigation matrix, a no-entanglement control, and SPSA
fine-tuning running on the QPU itself. The full record with every job ID:
[`qnn_tc/FINAL_WEEK.md`](qnn_tc/FINAL_WEEK.md) and
[`qnn_tc/HOLDOUT_PREREG.md`](qnn_tc/HOLDOUT_PREREG.md).

## Start here

[`qnn_tc/BasQ_QNN_Tc.ipynb`](qnn_tc/BasQ_QNN_Tc.ipynb) is the study notebook:
52 cells, 13 figures, runs end to end in about 4 minutes. It covers the
architecture comparison, the parameter-binding bug forensics, the mitigation
ablation (readout correction alone recovers simulation-grade accuracy), chip
tiling, training dynamics, and a physics-informed Section 15 (isotope-effect
reversal, electron-hub ansatz). Every hardware RMSE in Sections 1-14 is
recomputed live from stored expectation values; IBM job IDs are included.

## Repository map

| Path | What it is |
|---|---|
| [`qnn_tc/`](qnn_tc/) | The full pipeline: data prep, classical baselines, the hardware-aware QNN, real-QPU inference, mitigation ablation, chip tiling, multiseed protocol, QUBO feature selection, post-2018 out-of-time validation. Trained weights and measured hardware data ship with the repo; setup and the full results table are in [`qnn_tc/README.md`](qnn_tc/README.md). |
| [`qnn_tc/superconductor_at_home.py`](qnn_tc/superconductor_at_home.py) | One volunteer work unit for distributed screening: ensemble prediction with uncertainty, out-of-distribution triage, ranking by lower confidence bound. Run blind over 4,000 generated candidates, it re-discovers the Hg-Ba-Cu-O record family. Explainer: [`superconductor_at_home.md`](qnn_tc/superconductor_at_home.md). |
| [`scaling_study/`](scaling_study/) | Qubit-count scaling: PCA vs supervised Top-N encodings across 2-10 qubits, ideal vs real hardware. Ideal RMSE keeps improving with more qubits; depth and gate noise erode the gain. [`README_results.md`](scaling_study/README_results.md). |
| [`reference/`](reference/) | The original challenge brief and the mentors' starter notebook. |

## Reproduce

```bash
cd qnn_tc
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download train.csv (link below) into qnn_tc/data/, then:
jupyter lab BasQ_QNN_Tc.ipynb   # Run All, about 4 min, no IBM account needed
```

Expensive stages (retraining, noisy simulation, QPU runs) are gated behind
flags and default to shipped measured data. Training histories
(`data/loss_curves.npz`) regenerate in about 40 minutes via `python loss_curves.py`.

## Reporting choices

- Failed experiments are stored as censored bounds ("below X K"), never as Tc = 0.
- Enrichment is claimed only from a held-out reveal, and the app re-measures it per campaign.
- The strongest classical baseline beats the QNN on raw accuracy and is reported anyway.
- Volunteer screening counts as throughput, not as research staff.
- High-pressure hydrides fail for every model we tried (about 29 K RMSE); the notebook shows it.

Dataset: [UCI Superconductivity Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data) (21,263 materials, 81 features)

Team: Loïc, Haripriya, Abbas, Khadija, Karla, Sam · Mentors: Benjamin Tirado & Unai Aseguinolaza (BasQ)

© 2026 the OpenSuperLab team. Shared for review and reproduction; for other uses, ask first.
