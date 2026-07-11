"""Hardware-aware experiments: transpilation cost, noisy simulation,
and (once credentials exist) real-QPU inference of a trained QNN.

Strategy: train in ideal simulation (qnn.py), then run *inference only*
on noisy/real backends. Training on a QPU is not feasible in a weekend;
inference of a trained circuit is one batched job.
"""

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime.fake_provider import FakeTorino

from baselines import rmse_kelvin
from qnn import build_circuit


def param_matrix(X, weights, inputs, wparams, circuit):
    """Parameter-value matrix column-ordered to match circuit.parameters.

    A bare ndarray in a PUB is bound in circuit.parameters order (sorted by
    parameter NAME, so our x[...] inputs come last), not in the
    [inputs..., weights...] order the matrix is naturally built in. Binding
    an unordered hstack scrambles every angle in the circuit.
    """
    pv = np.hstack([X, np.tile(weights, (len(X), 1))])
    col = {p: i for i, p in enumerate(list(inputs) + list(wparams))}
    return pv[:, [col[p] for p in circuit.parameters]]


def transpile_report(n_qubits, n_layers, n_features, backend=None):
    """Depth / 2q-gate cost of the ansatz before and after ISA transpilation."""
    backend = backend or FakeTorino()
    qc, _, _, _ = build_circuit(n_qubits, n_layers, n_features)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(qc)
    ops = isa.count_ops()
    print(f"q={n_qubits} L={n_layers} -> logical depth {qc.depth()}, "
          f"ISA depth {isa.depth()}, CZ count {ops.get('cz', 0)}, "
          f"ops {dict(ops)}")
    return isa, pm


def noisy_inference(d, weights, n_qubits, n_layers, shots=4096,
                    backend=None, seed=11):
    """Run the trained circuit on a noisy Aer model of a Heron QPU."""
    backend = backend or FakeTorino()
    n_features = d["X_test"].shape[1]
    qc, inputs, wparams, obs = build_circuit(n_qubits, n_layers, n_features)

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(qc)
    obs_isa = obs.apply_layout(isa.layout)

    noise_est = AerEstimator.from_backend(
        backend, options={"run_options": {"seed_simulator": seed}})
    X = d["X_test"]
    # one PUB: circuit + (n_test, n_params) parameter array, broadcast,
    # column-ordered to the circuit's parameter binding order
    param_values = param_matrix(X, weights, inputs, wparams, isa)
    job = noise_est.run([(isa, obs_isa, param_values)], precision=1 / np.sqrt(shots))
    ev = job.result()[0].data.evs
    rm, ma = rmse_kelvin(d["y_test"], np.asarray(ev).ravel(), d)
    print(f"noisy inference ({backend.name}, {shots} shots): "
          f"RMSE {rm:.2f} K  MAE {ma:.2f} K")
    return rm, ma


def qpu_inference(d, weights, n_qubits, n_layers, backend_name=None,
                  shots=4096, resilience_level=1, dd=True, twirl=True,
                  save_evs=None):
    """Same, on real hardware via Qiskit Runtime. Needs saved credentials.

    resilience_level 0 = raw, 1 = TREX readout mitigation, 2 = +ZNE.
    dd / twirl toggle dynamical decoupling and gate twirling independently
    so mitigation ablations can credit each technique separately.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

    service = QiskitRuntimeService()
    backend = (service.backend(backend_name) if backend_name
               else service.least_busy(operational=True, simulator=False))
    print("Using backend:", backend.name)

    n_features = d["X_test"].shape[1]
    qc, inputs, wparams, obs = build_circuit(n_qubits, n_layers, n_features)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(qc)
    obs_isa = obs.apply_layout(isa.layout)

    est = EstimatorV2(mode=backend)
    est.options.resilience_level = resilience_level
    est.options.default_shots = shots
    if dd:
        est.options.dynamical_decoupling.enable = True
        est.options.dynamical_decoupling.sequence_type = "XY4"
    est.options.twirling.enable_gates = twirl

    X = d["X_test"]
    param_values = param_matrix(X, weights, inputs, wparams, isa)
    job = est.run([(isa, obs_isa, param_values)])
    print("job id:", job.job_id())
    ev = np.asarray(job.result()[0].data.evs).ravel()
    if save_evs:
        np.save(save_evs, ev)
    rm, ma = rmse_kelvin(d["y_test"], ev, d)
    print(f"QPU inference ({backend.name}): RMSE {rm:.2f} K  MAE {ma:.2f} K")
    return rm, ma


if __name__ == "__main__":
    d = dict(np.load("data/prepared_4d.npz"))
    transpile_report(4, 3, 4)
    try:
        w = np.load("data/trained_weights_q4_l3.npy")
        noisy_inference(d, w, 4, 3)
    except FileNotFoundError:
        print("train first: .venv/bin/python qnn.py")
