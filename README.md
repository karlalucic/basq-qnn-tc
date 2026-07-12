# OpenSuperLab — Hardware-Aware QNN for Superconductor Tc Prediction

**BasQ Challenge (Challenge 5) · CERN Quantum Materials Hackathon 2026**

A quantum neural network that predicts superconductor critical temperatures and
*survives real hardware* — plus [OpenSuperLab](https://opensuperlab.vercel.app),
a working screening instrument built on the same trained model, with a public
citizen-science door at [/contribute](https://opensuperlab.vercel.app/contribute).

## The result in four numbers

| What | Measured |
|---|---|
| Champion QNN vs parameter-matched MLP (3 seeds, identical inputs) | **19.84 ± 0.03 K** vs 26.7 ± 4.6 K test RMSE |
| Ideal simulator → noisy simulator → real QPU (`ibm_fez`, same 50 materials) | 23.00 → 22.86 → **22.89 K** (r = 0.999) |
| QPU time per 50-material screen (25 circuit copies tiled on the 156-qubit chip) | 41 → **12 s** (+0.55 K) |
| Selection quality on held-out materials (top-10 vs random pick) | 8/10 vs 3.8 → **2.1× enrichment** |

The champion circuit: 4 qubits × 2 layers, 32 parameters, supervised PLS-4
inputs, one chain of 3 CZ gates — transpiles to ISA depth 21 with **zero SWAPs**
on IBM Heron. Gradient boosting with all 81 features beats every small model on
raw RMSE (15.9 K) — we show it on purpose: our claim is parameter efficiency
and hardware viability, not supremacy.

## Start here

**[`qnn_tc/BasQ_QNN_Tc.ipynb`](qnn_tc/BasQ_QNN_Tc.ipynb)** — the reproducible
study notebook: 52 cells, 13 figures, runs end-to-end in ~4 minutes. Equations,
architecture comparison, the parameter-binding bug forensics, mitigation
ablation (readout correction alone recovers simulation-grade accuracy), chip
tiling, training dynamics, and a physics-informed §15 (isotope-effect reversal,
electron-hub ansatz). Every hardware RMSE in §§1–14 is recomputed live from
stored expectation values; IBM job IDs included.

## Repository map

| Path | What it is |
|---|---|
| [`qnn_tc/`](qnn_tc/) | The full pipeline: data prep, classical baselines, the hardware-aware QNN, real-QPU inference, mitigation ablation, chip tiling, multiseed protocol, QUBO feature selection, post-2018 out-of-time validation. Trained weights and measured hardware data ship with the repo — see [`qnn_tc/README.md`](qnn_tc/README.md) for setup and the full results table. |
| [`qnn_tc/superconductor_at_home.py`](qnn_tc/superconductor_at_home.py) | One volunteer *work unit* for distributed screening: ensemble prediction with uncertainty, out-of-distribution triage, ranking by lower confidence bound. Run blind over 4,000 generated candidates it re-discovers the Hg–Ba–Cu–O record family. Explainer: [`superconductor_at_home.md`](qnn_tc/superconductor_at_home.md). |
| [`scaling_study/`](scaling_study/) | Qubit-count scaling study: PCA vs supervised Top-N encodings across 2–10 qubits, ideal vs real hardware. Finding: ideal RMSE keeps improving with more qubits, but depth/gate noise erodes the gain — hardware-aware design, not qubit count, is the lever. [`README_results.md`](scaling_study/README_results.md). |
| [`opensuperlab/`](opensuperlab/) | The web instrument (Next.js). **Workbench**: campaign briefs in lab terms, experiment prioritization with the exact trained QNN running client-side, measured enrichment-vs-random reporting, censored negatives preserved, deterministic protocols, local lab notebook. **`/contribute`**: in-browser volunteer screening in 25-candidate work units, leads ranked by confidence floor (estimate − uncertainty). No accounts, no server-side model — see [`opensuperlab/README.md`](opensuperlab/README.md). |
| [`Challenge_proposal_BasQ.md`](Challenge_proposal_BasQ.md) | The original challenge brief and evaluation criteria. |
| [`quantum_regressor_v2.ipynb`](quantum_regressor_v2.ipynb) | The mentors' starter notebook. |

## Reproduce

```bash
cd qnn_tc
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download train.csv (link below) into qnn_tc/data/, then:
jupyter lab BasQ_QNN_Tc.ipynb   # Run All — ~4 min, no IBM account needed
```

Expensive stages (retraining, noisy simulation, QPU runs) are gated behind
flags and default to shipped measured data. Training histories
(`data/loss_curves.npz`) regenerate in ~40 min via `python loss_curves.py`.

```bash
cd opensuperlab
npm install && npm run dev      # the app; model runs in the browser
```

## Honesty rules we kept

- Failed experiments are stored censored ("below X K"), never as `Tc = 0`.
- Enrichment is claimed only from a held-out reveal, and the app re-measures it per campaign.
- The strongest classical model is shown beating us on raw accuracy.
- Volunteer screening is reported as throughput — volunteers are not FTE researchers (SDG 9.5 precision).
- High-pressure hydrides fail for every model (~29 K); it's in the notebook.

Dataset: [UCI Superconductivity Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data) (21,263 materials, 81 features)

Team: Loïc, Haripriya, Abbas, Khadija, Karla, Sam · Mentors: Benjamin Tirado & Unai Aseguinolaza (BasQ)
