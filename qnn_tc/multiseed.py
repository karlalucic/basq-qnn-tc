"""Multi-seed champion-vs-baseline comparison (audit finding (b)).

Single-seed numbers conflate model quality with optimizer luck: L-BFGS-B is
deterministic given its start, so the initial_point seed alone moves the
QNN into a different local minimum, and the MLP's weight init is likewise
seeded. This trains the champion QNN and the parameter-matched MLP across
three seeds at the SWEEP budget and reports mean +/- std, so the QNN-vs-MLP
comparison rests on a distribution rather than one draw.

Champion QNN: 4 qubits, 2 layers, PLS inputs (data/prepared_4d_pls.npz).
Budget: first 300 training rows, L_BFGS_B(maxiter=60).
Baseline: parameter-matched MLP from baselines.py (h=5 hidden, 31 params vs
the QNN's 32), same 300 rows.
Reported per seed: test RMSE (K) on the full 200-point test set AND the frozen
first-50 subset used by the hardware runs, then mean +/- std across seeds.
"""

import time

import numpy as np
from sklearn.neural_network import MLPRegressor

from baselines import rmse_kelvin
from qnn import train

SEEDS = (7, 17, 27)
N_TRAIN = 300
MAXITER = 60
N_QUBITS, N_LAYERS = 4, 2


def _mspm(vals):
    """mean and sample std (ddof=1) of a list of scalars."""
    a = np.asarray(vals, float)
    return a.mean(), a.std(ddof=1)


def mlp_sizing(k, param_budget):
    """Same hidden-width rule as baselines.run_baselines."""
    h = max(1, round((param_budget - 1) / (k + 2)))
    return h, (k + 2) * h + 1


def run():
    d = dict(np.load("data/prepared_4d_pls.npz"))
    d_small = dict(d)
    d_small["X_train"] = d["X_train"][:N_TRAIN]
    d_small["y_train"] = d["y_train"][:N_TRAIN]
    Xtr, ytr = d_small["X_train"], d_small["y_train"]
    Xte_full, yte_full = d["X_test"], d["y_test"]
    Xte_50, yte_50 = d["X_test"][:50], d["y_test"][:50]
    k = Xtr.shape[1]

    print(f"champion QNN q={N_QUBITS} L={N_LAYERS} PLS | budget {N_TRAIN} rows, "
          f"L_BFGS_B(maxiter={MAXITER}) | seeds {SEEDS}\n")

    q_full, q_50, m_full, m_50 = [], [], [], []

    # -- Champion QNN, one fit per seed (sequential; ~10 min each) -----------
    for i, s in enumerate(SEEDS, 1):
        t0 = time.time()
        reg, _ = train(d_small, n_qubits=N_QUBITS, n_layers=N_LAYERS,
                       maxiter=MAXITER, seed=s, verbose=False)
        # NeuralNetworkRegressor.predict routes through EstimatorQNN.forward,
        # which binds parameters in the correct order (never a bare ndarray).
        rf, _ = rmse_kelvin(yte_full, reg.predict(Xte_full).ravel(), d)
        r50, _ = rmse_kelvin(yte_50, reg.predict(Xte_50).ravel(), d)
        q_full.append(rf)
        q_50.append(r50)
        print(f"[QNN {i}/{len(SEEDS)}] seed {s:2d} | fit {time.time()-t0:5.0f}s "
              f"| test200 RMSE {rf:6.2f} K | frozen50 RMSE {r50:6.2f} K")

    # -- Parameter-matched MLP, same seeds & rows ----------------------------
    h, mlp_params = mlp_sizing(k, param_budget=32)  # 32 = QNN weight count
    print(f"\nparameter-matched MLP: hidden={h} ({mlp_params} params vs QNN 32)\n")
    for i, s in enumerate(SEEDS, 1):
        t0 = time.time()
        mlp = MLPRegressor(hidden_layer_sizes=(h,), max_iter=4000,
                           random_state=s).fit(Xtr, ytr)
        rf, _ = rmse_kelvin(yte_full, mlp.predict(Xte_full), d)
        r50, _ = rmse_kelvin(yte_50, mlp.predict(Xte_50), d)
        m_full.append(rf)
        m_50.append(r50)
        print(f"[MLP {i}/{len(SEEDS)}] seed {s:2d} | fit {time.time()-t0:5.1f}s "
              f"| test200 RMSE {rf:6.2f} K | frozen50 RMSE {r50:6.2f} K")

    # -- Summary --------------------------------------------------------------
    def line(name, vals):
        m, sd = _mspm(vals)
        per = "  ".join(f"{v:6.2f}" for v in vals)
        return f"{name:22s} [{per}]  mean {m:6.2f}  std {sd:5.2f} K"

    print("\n=== per-seed RMSE (K), seeds 7/17/27 ===")
    print(line("QNN  test-200", q_full))
    print(line("QNN  frozen-50", q_50))
    print(line("MLP  test-200", m_full))
    print(line("MLP  frozen-50", m_50))

    results = dict(seeds=list(SEEDS), qnn_test200=q_full, qnn_frozen50=q_50,
                   mlp_test200=m_full, mlp_frozen50=m_50,
                   qnn_params=32, mlp_params=mlp_params)
    np.savez("data/multiseed_results.npz", **{k: np.asarray(v)
             for k, v in results.items()})
    print("\nsaved data/multiseed_results.npz")
    return results


if __name__ == "__main__":
    run()
