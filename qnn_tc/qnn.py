"""Hardware-efficient data re-uploading QNN for Tc regression.

Architecture (per layer):
  1. encoding sublayer  : Ry(w_ij * x_j + b_ij) on each qubit — trainable
                          input scaling, so re-uploads become a Fourier
                          series with learnable frequencies
  2. variational sublayer: Rz, Ry rotations per qubit
  3. entanglement       : CZ gates on a linear chain — CZ is native on
                          IBM Heron QPUs and a chain embeds directly into
                          the heavy-hex lattice with zero SWAPs

Output: mean of single-qubit Z expectations (readout-noise friendly).
"""

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.optimizers import L_BFGS_B, COBYLA

from prep_data import inverse_target
from baselines import rmse_kelvin


def build_circuit(n_qubits: int, n_layers: int, n_features: int,
                  entangle: bool = True):
    """Returns (circuit, input_params, weight_params)."""
    x = ParameterVector("x", n_features)
    qc = QuantumCircuit(n_qubits)
    weights = []

    def w(name):
        p = Parameter(name)
        weights.append(p)
        return p

    for layer in range(n_layers):
        # encoding: features round-robin over qubits, trainable scale+shift
        for q in range(n_qubits):
            f = (q + layer) % n_features
            qc.ry(w(f"s{layer}_{q}") * x[f] + w(f"o{layer}_{q}"), q)
        # variational
        for q in range(n_qubits):
            qc.rz(w(f"a{layer}_{q}"), q)
            qc.ry(w(f"c{layer}_{q}"), q)
        # entanglement: linear chain of CZ (SWAP-free on heavy-hex)
        if entangle and n_qubits > 1 and layer < n_layers - 1:
            for q in range(n_qubits - 1):
                qc.cz(q, q + 1)

    obs = SparsePauliOp.from_sparse_list(
        [("Z", [q], 1 / n_qubits) for q in range(n_qubits)],
        num_qubits=n_qubits)
    return qc, list(x), weights, obs


def make_qnn(n_qubits, n_layers, n_features, estimator=None, pass_manager=None,
             gradient=None, entangle=True):
    qc, inputs, weights, obs = build_circuit(
        n_qubits, n_layers, n_features, entangle)
    if pass_manager is not None:
        qc_isa = pass_manager.run(qc)
        obs = obs.apply_layout(qc_isa.layout)
        qc = qc_isa
    qnn = EstimatorQNN(circuit=qc, observables=obs, input_params=inputs,
                       weight_params=weights, estimator=estimator,
                       gradient=gradient, pass_manager=pass_manager)
    return qnn, qc


def train(d, n_qubits=4, n_layers=3, maxiter=150, optimizer=None, seed=7,
          verbose=True):
    n_features = d["X_train"].shape[1]
    qnn, qc = make_qnn(n_qubits, n_layers, n_features)
    opt = optimizer or L_BFGS_B(maxiter=maxiter)
    rng = np.random.default_rng(seed)
    reg = NeuralNetworkRegressor(
        neural_network=qnn, loss="squared_error", optimizer=opt,
        initial_point=rng.uniform(-0.3, 0.3, qnn.num_weights))

    t0 = time.time()
    reg.fit(d["X_train"], d["y_train"])
    fit_s = time.time() - t0

    rm_tr, _ = rmse_kelvin(d["y_train"], reg.predict(d["X_train"]).ravel(), d)
    rm_te, ma_te = rmse_kelvin(d["y_test"], reg.predict(d["X_test"]).ravel(), d)
    if verbose:
        print(f"QNN q={n_qubits} L={n_layers} params={qnn.num_weights} "
              f"| fit {fit_s:.0f}s | train RMSE {rm_tr:.2f} K "
              f"| test RMSE {rm_te:.2f} K MAE {ma_te:.2f} K")
    return reg, dict(rmse_test=rm_te, mae_test=ma_te, rmse_train=rm_tr,
                     n_params=qnn.num_weights, fit_seconds=fit_s)


if __name__ == "__main__":
    d = dict(np.load("data/prepared_4d.npz"))
    reg, stats = train(d, n_qubits=4, n_layers=3, maxiter=150)
    np.save("data/trained_weights_q4_l3.npy", reg.weights)
