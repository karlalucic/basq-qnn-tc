"""5-setting mitigation ablation on the champion QNN (D4 in the plan).

raw -> TREX -> +DD -> +gate twirling -> ZNE, each as one batched job on the
identical frozen 50 test points, 2048 shots, ibm_fez. Ideal reference on
these points: 23.00 K. Answers: which mitigation earns its keep for a
3-CZ circuit, with per-setting evs saved for the recovery plot.
"""

import numpy as np

from hardware import qpu_inference

SETTINGS = [
    ("raw",          dict(resilience_level=0, dd=False, twirl=False)),
    ("TREX",         dict(resilience_level=1, dd=False, twirl=False)),
    ("TREX+DD",      dict(resilience_level=1, dd=True,  twirl=False)),
    ("TREX+DD+twirl", dict(resilience_level=1, dd=True,  twirl=True)),
    ("ZNE(res2)",    dict(resilience_level=2, dd=True,  twirl=True)),
]

d = dict(np.load("data/prepared_4d_pls.npz"))
d50 = dict(d)
d50["X_test"] = d["X_test"][:50]
d50["y_test"] = d["y_test"][:50]
w = np.load("data/trained_weights_q4_l2_pls.npy")

results = {}
for name, kw in SETTINGS:
    print(f"=== {name} ===")
    tag = name.lower().replace("+", "_").replace("(", "").replace(")", "")
    rm, ma = qpu_inference(d50, w, n_qubits=4, n_layers=2,
                           backend_name="ibm_fez", shots=2048,
                           save_evs=f"data/ablation_evs_{tag}.npy", **kw)
    results[name] = (rm, ma)

print("\n=== ABLATION SUMMARY (champion, frozen 50, ideal = 23.00 K) ===")
print(f"{'setting':16s} {'RMSE (K)':>9s} {'MAE (K)':>9s}")
for name, (rm, ma) in results.items():
    print(f"{name:16s} {rm:9.2f} {ma:9.2f}")
