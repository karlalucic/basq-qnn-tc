"""Pre-hackathon dress rehearsal: real-QPU inference of the trained QNN.

Conservative budget: 50 test points, 2048 shots, one batched PUB.
Validates the entire hardware path (credentials -> transpile -> mitigation
options -> batched job -> Kelvin metrics) before the event.
"""

import numpy as np

from hardware import qpu_inference

d = dict(np.load("data/prepared_4d.npz"))
w = np.load("data/trained_weights_q4_l3.npy")

d_sub = dict(d)
d_sub["X_test"] = d["X_test"][:50]
d_sub["y_test"] = d["y_test"][:50]

rm, ma = qpu_inference(d_sub, w, n_qubits=4, n_layers=3,
                       backend_name="ibm_fez", shots=2048,
                       resilience_level=1, dd=True)
print(f"DRESS REHEARSAL RESULT: RMSE {rm:.2f} K, MAE {ma:.2f} K")
