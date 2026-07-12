# BasQ QNN Challenge — Team Repo

CERN Quantum Materials Hackathon 2026 · Predicting critical temperatures in
superconductors using Quantum Neural Networks (Basque Quantum / BasQ).

Original challenge materials:

- [Challenge_proposal_BasQ.md](Challenge_proposal_BasQ.md) — the challenge brief,
  evaluation criteria, and resources.
- [quantum_regressor_v2.ipynb](quantum_regressor_v2.ipynb) — the mentors' starter
  notebook (single-qubit QNN regressor).

Our pipeline:

- [qnn_tc/BasQ_QNN_Tc.ipynb](qnn_tc/BasQ_QNN_Tc.ipynb) — **the reproducible
  study notebook**: equations, architecture comparison, real-QPU results with
  charts; runs end-to-end in ~4 min (see its appendix for required data files —
  everything is included here except `train.csv`, download link below).

- [qnn_tc/](qnn_tc/) — the full working pipeline: data prep, classical
  baselines, the hardware-aware 4-qubit QNN, real-QPU inference, mitigation
  ablation, chip tiling, and validation experiments. Trained weights and
  measured hardware results included; see [qnn_tc/README.md](qnn_tc/README.md)
  for setup and the measured-results table.

- [scaling_study/](scaling_study/) — qubit-count scaling study (Haripriya):
  PCA vs supervised Top-N encodings across 2-10 qubits, ideal simulation vs
  real hardware. Conclusion: ideal RMSE keeps improving with more qubits, but
  depth/gate noise erodes the gain on hardware - hardware-aware design, not
  qubit count, is the lever. See [scaling_study/README_results.md](scaling_study/README_results.md).

Dataset: [UCI Superconductivity Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data)

Team: Loïc, Haripriya, Abbas, Khadija, Karla, Sam · Mentors: Benjamin Tirado & Unai Aseguinolaza
