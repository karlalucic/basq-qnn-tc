"""Corrected-binding QPU rerun (invalidates jobs d97p1ccq... / d98k5c2f...).

Both models, identical frozen 50 test points, 2048 shots, TREX+DD+twirling
on ibm_fez — the first hardware numbers with correctly bound parameters.
"""

import numpy as np

from hardware import qpu_inference

RUNS = [
    ("PCA-L3", "data/prepared_4d.npz", "data/trained_weights_q4_l3.npy", 3),
    ("PLS-L2 champion", "data/prepared_4d_pls.npz",
     "data/trained_weights_q4_l2_pls.npy", 2),
]

for name, npz, wfile, L in RUNS:
    d = dict(np.load(npz))
    d50 = dict(d)
    d50["X_test"] = d["X_test"][:50]
    d50["y_test"] = d["y_test"][:50]
    w = np.load(wfile)
    print(f"=== {name} corrected QPU run (frozen 50, 2048 shots) ===")
    rm, ma = qpu_inference(d50, w, n_qubits=4, n_layers=L,
                           backend_name="ibm_fez", shots=2048,
                           resilience_level=1, dd=True)
    print(f"CORRECTED {name}: RMSE {rm:.2f} K  MAE {ma:.2f} K\n")
