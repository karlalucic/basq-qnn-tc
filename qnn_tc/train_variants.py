"""Simulation-side trainings feeding the final-week hardware runs.

Trains at the champion's full budget (800 rows, L_BFGS_B maxiter=150):
  1. champion seeds 17 and 27 (seed 7 is the deployed champion), so the
     hardware headline can carry a cross-seed error bar
  2. a no-entanglement control (entangle=False, seed 7): same circuit
     minus the 3 CZ gates, the "is the quantum part doing anything" arm

Run: .venv/bin/python train_variants.py
"""

import numpy as np
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.optimizers import L_BFGS_B

from baselines import rmse_kelvin
from qnn import make_qnn, train

d = dict(np.load("data/prepared_4d_pls.npz"))

for s in (17, 27):
    reg, stats = train(d, n_qubits=4, n_layers=2, maxiter=150, seed=s)
    np.save(f"data/trained_weights_q4_l2_pls_seed{s}.npy", reg.weights)
    print(f"seed {s}: saved (test RMSE {stats['rmse_test']:.2f} K)")

# no-entanglement control: train() does not expose the entangle knob
qnn, _ = make_qnn(4, 2, 4, entangle=False)
rng = np.random.default_rng(7)
reg = NeuralNetworkRegressor(
    neural_network=qnn, loss="squared_error", optimizer=L_BFGS_B(maxiter=150),
    initial_point=rng.uniform(-0.3, 0.3, qnn.num_weights))
reg.fit(d["X_train"], d["y_train"])
rm, ma = rmse_kelvin(d["y_test"], reg.predict(d["X_test"]).ravel(), d)
np.save("data/trained_weights_q4_l2_pls_noent.npy", reg.weights)
print(f"no-entanglement control: test RMSE {rm:.2f} K MAE {ma:.2f} K (saved)")
