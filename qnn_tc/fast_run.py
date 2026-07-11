"""Fast end-to-end check: small training budget, then noisy Heron inference.
Use this configuration style for architecture sweeps; retrain the champion
with the full budget (qnn.py) overnight.
"""

import numpy as np

from hardware import noisy_inference, transpile_report
from qnn import train

d = dict(np.load("data/prepared_4d.npz"))
# sweep-sized budget: 300 train samples, shallower model, fewer iterations
d_small = dict(d)
d_small["X_train"] = d["X_train"][:300]
d_small["y_train"] = d["y_train"][:300]

reg, stats = train(d_small, n_qubits=4, n_layers=2, maxiter=60)
np.save("data/trained_weights_q4_l2.npy", reg.weights)

transpile_report(4, 2, 4)
noisy_inference(d_small, reg.weights, 4, 2, shots=4096)
