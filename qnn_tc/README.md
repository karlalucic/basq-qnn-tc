# qnn_tc: QNN pipeline for the BasQ Tc-regression challenge

Ready-to-run pipeline for the CERN Quantum Materials Hackathon 2026.
Predicts superconductor critical temperature with a hardware-aware
4-qubit data re-uploading QNN, trained in simulation and evaluated on
IBM Heron QPUs.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Large raw datasets are not in the repo; download them once:

- UCI superconductivity data (`train.csv`, `unique_m.csv` -> `data/`):
  https://archive.ics.uci.edu/dataset/464/superconductivty+data
- 3DSC (only needed for the structure experiments; `3DSC_MP.csv` ->
  `data/external/`): https://github.com/aimat-lab/3DSC

All prepared inputs (`data/prepared_*.npz`), trained weights
(`data/trained_weights_*.npy`), and measured hardware expectation values
(`data/*_evs*.npy`) ARE included, so steps 4-8 run without the raw data.

## Pipeline

| Step | Command | What it does |
|---|---|---|
| 1. Data | `prep_data.py` | dedupe -> log-Tc -> PCA or PLS -> angle scaling -> shared splits (seed 42) |
| 2. Baselines | `baselines.py` | linear / param-matched MLP / RF + full-data GBM ceiling, RMSE in Kelvin |
| 3. Train QNN | `qnn.py` | data re-uploading ansatz + EstimatorQNN; saves weights |
| 4. Transpile + noisy sim | `hardware.py` | ISA cost report, noisy Aer (FakeFez) inference |
| 5. Real QPU | `connect_ibm.py save <KEY> [CRN]` then `qpu_corrected_rerun.py` | batched inference, one PUB for the whole test set |
| 6. Mitigation ablation | `mitigation_ablation.py` | raw -> TREX -> +DD -> +twirling -> ZNE, one job each |
| 7. Chip tiling | `tiling_experiment.py` (sim), `tiled_qpu_ab.py` (real A/B) | 25 parallel copies of the QNN, 25x fewer parameter rows |
| 8. Robustness | `multiseed.py`, `fresh_holdout.py`, `frozen50_check.py` | seed variance, untouched holdout, controlled 50-point comparisons |
| 9. Input experiments | `qubo_select.py`, `featurize_formula.py`, `post2018_eval.py`, `dsc_ablation.py`, `struct_qnn_experiment.py` | QUBO feature selection, out-of-time validation, 3DSC structure ablation |
| 10. Training curves | `loss_curves.py` | regenerates `data/loss_curves.npz` (QNN + MLP histories, ~40 min); `team_training_curves.ipynb` is the team runbook for the same job |
| 11. Final-week QPU runs | `holdout_prereg.py`, `final_qpu_submit.py`, `final_qpu_collect.py` | preregistered holdout + full-test tiled runs on two Herons; job IDs in `data/final_runs_jobs.json` |
| 12. Closing campaign | `final_burn.py`, `wave2.py`, `day2_models.py`, `candidates_qpu_screen.py`, `hw_finetune.py`, `burn_collect.py` | five-device replication, whole-dataset inference, tile ablations, mitigation matrix, candidate screen, QPU fine-tuning; results in [FINAL_WEEK.md](FINAL_WEEK.md) |

## Architecture (qnn.py)

Data re-uploading ansatz with trainable frequencies: per layer, each qubit gets
`Ry(s*x_f + o)` (feature f round-robin, s and o trainable) -> `Rz`, `Ry`
trainable -> CZ linear chain (skipped after the last layer, since CZ is diagonal
in the measurement basis). Observable = mean of single-qubit Z.

Champion: **4 qubits x 2 layers, 32 params, PLS-4 inputs** -> ISA depth 21,
**3 CZ, 0 SWAPs** on Heron.

## Key measured results (IBM Heron QPUs, July 2026)

| Experiment | Result |
|---|---|
| Ideal test RMSE (200 pts) | champion 19.8 K (linear 20.2, param-matched MLP 21.3) |
| Full test set on hardware, five devices (200 pts) | **20.75 to 21.71 K** vs ideal 19.83 K, r = 0.99 everywhere ([FINAL_WEEK.md](FINAL_WEEK.md)) |
| Whole cleaned dataset (15,170 materials) | test split fez **21.24 K**, pittsburgh **21.18 K** |
| Preregistered fresh holdout (300 materials) | **23.20 K** vs ideal 21.30 K committed before the run ([HOLDOUT_PREREG.md](HOLDOUT_PREREG.md)) |
| Candidate screen (4,000 generated formulas) | hardware vs simulation rank agreement rho = 0.98 |
| Per-tile error map (25 tiles, one job) | per-tile error vs calibration score r = 0.99 |
| No-entanglement control (CZ removed, retrained) | sim 20.09 K vs champion 19.83 K; hardware 23.65 K vs 22.90 K |
| Noise ladder (frozen 50) | ideal 23.00 -> noisy sim 22.86 -> **real QPU 22.89 K** |
| Mitigation ablation | raw 24.62 -> **TREX 22.90** -> +DD/+twirl/ZNE: no further gain |
| Tiled A/B (25 copies) | single 23.61 vs tiled 24.16 K, 50 -> 2 rows, 12 vs 41 QPU-s |
| Seed variance (3 seeds) | QNN 19.84 +/- 0.03 K vs MLP 26.68 +/- 4.61 K |
| Out-of-time (post-2018, n=14) | QNN 3.93 K RMSE (pressure-dependent rows fail ~29 K for all models) |

## Shared experimental protocol (do not break)

- One split for every model: seed 42, fixed in `prep_data.py`.
- Fit scalers/reducers on train only. Test set is sacred; final numbers can be
  quoted on `data/fresh_holdout_*.npz` (never used for any decision).
- Report RMSE/MAE in Kelvin (`baselines.rmse_kelvin` inverts scaling + log).
- Architecture sweeps: 300 train samples, maxiter 60. Champion: full budget.
- **Never bind a bare NumPy array to a Qiskit Estimator PUB.** Bare arrays bind
  in `circuit.parameters` order (sorted by NAME, so `x[...]` binds last), which
  silently scrambles every angle. Use `EstimatorQNN.forward` or
  `hardware.param_matrix()`, and pre-flight any new binding path against exact
  simulation (see `tiled_qpu_ab.py`) before spending QPU time.
