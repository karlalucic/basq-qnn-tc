"""Training-dynamics histories for the judge notebook's loss-curve figure.

Trains the champion QNN configuration (4q x 2L, PLS-4) and the parameter-matched
MLP at the sweep budget (300 training rows) for seeds 7/17/27, recording:
  - the raw training objective at every optimizer evaluation (the loss curve), and
  - the 200-point test RMSE in Kelvin at thinned weight checkpoints,
so the notebook can plot honest training dynamics without retraining.

Output: data/loss_curves.npz
"""

import time

import numpy as np
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.optimizers import L_BFGS_B
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
import warnings

from prep_data import inverse_target
from qnn import make_qnn

SEEDS = (7, 17, 27)
N_TRAIN = 300
MAXITER = 60
MLP_EPOCHS = 400
MAX_RMSE_CHECKPOINTS = 40

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def rmse_kelvin_from_pred(y_true_scaled, y_pred_scaled, d):
    t = inverse_target(np.asarray(y_true_scaled).ravel(), d["y_min"], d["y_max"])
    p = inverse_target(np.clip(np.asarray(y_pred_scaled).ravel(), -1, 1),
                       d["y_min"], d["y_max"])
    return float(np.sqrt(np.mean((t - p) ** 2)))


def train_qnn_with_history(d, seed):
    qnn, _ = make_qnn(4, 2, d["X_train"].shape[1])
    objs, weight_log = [], []

    def callback(weights, obj):
        objs.append(float(obj))
        weight_log.append(np.array(weights, copy=True))

    rng = np.random.default_rng(seed)
    reg = NeuralNetworkRegressor(
        neural_network=qnn, loss="squared_error",
        optimizer=L_BFGS_B(maxiter=MAXITER),
        initial_point=rng.uniform(-0.3, 0.3, qnn.num_weights),
        callback=callback)
    t0 = time.time()
    reg.fit(d["X_train"][:N_TRAIN], d["y_train"][:N_TRAIN])
    fit_s = time.time() - t0

    # Test-RMSE trajectory at thinned checkpoints (forward passes only).
    idx = np.unique(np.linspace(0, len(weight_log) - 1,
                                min(MAX_RMSE_CHECKPOINTS, len(weight_log)),
                                dtype=int))
    rmses = []
    for i in idx:
        evs = np.asarray(qnn.forward(d["X_test"], weight_log[i])).ravel()
        rmses.append(rmse_kelvin_from_pred(d["y_test"], evs, d))
    final = rmse_kelvin_from_pred(
        d["y_test"], np.asarray(qnn.forward(d["X_test"], reg.weights)).ravel(), d)
    print(f"QNN seed {seed}: {len(objs)} evals, fit {fit_s:.0f}s, "
          f"final test RMSE {final:.2f} K")
    return np.array(objs), idx, np.array(rmses), final


def train_mlp_with_history(d, seed):
    mlp = MLPRegressor(hidden_layer_sizes=(5,), max_iter=1, warm_start=True,
                       random_state=seed)
    losses, rmses = [], []
    for _ in range(MLP_EPOCHS):
        mlp.fit(d["X_train"][:N_TRAIN], d["y_train"][:N_TRAIN])
        losses.append(float(mlp.loss_))
        rmses.append(rmse_kelvin_from_pred(
            d["y_test"], mlp.predict(d["X_test"]), d))
    print(f"MLP seed {seed}: {MLP_EPOCHS} epochs, final test RMSE {rmses[-1]:.2f} K")
    return np.array(losses), np.array(rmses)


def main():
    d = dict(np.load("data/prepared_4d_pls.npz"))
    out = {"seeds": np.array(SEEDS), "n_train": N_TRAIN, "maxiter": MAXITER,
           "mlp_epochs": MLP_EPOCHS}
    for seed in SEEDS:
        objs, idx, rmses, final = train_qnn_with_history(d, seed)
        out[f"qnn_obj_s{seed}"] = objs
        out[f"qnn_ckpt_idx_s{seed}"] = idx
        out[f"qnn_rmse_s{seed}"] = rmses
        out[f"qnn_final_s{seed}"] = final
        losses, mlp_rmses = train_mlp_with_history(d, seed)
        out[f"mlp_loss_s{seed}"] = losses
        out[f"mlp_rmse_s{seed}"] = mlp_rmses
    np.savez("data/loss_curves.npz", **out)
    print("saved data/loss_curves.npz")


if __name__ == "__main__":
    main()
