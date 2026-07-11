"""Controlled ranking-flip check (Codex-audit P0).

The QPU runs used X_test[:50]; ideal RMSEs were reported on the full
200-point test set. This recomputes exact (statevector) predictions for
both models on the identical frozen 50 points, so ideal vs. hardware is
finally apples-to-apples. Zero QPU cost.
"""

import numpy as np

from qnn import make_qnn
from baselines import rmse_kelvin

d_pca = dict(np.load("data/prepared_4d.npz"))
d_pls = dict(np.load("data/prepared_4d_pls.npz"))

# Same seed -> same materials in the same order in both files.
same = np.allclose(d_pca["y_test"], d_pls["y_test"])
print(f"y_test identical across PCA/PLS files: {same}")
assert same, "test rows differ between npz files — comparison invalid"

models = [
    ("PCA-L3", d_pca, np.load("data/trained_weights_q4_l3.npy"), 3),
    ("PLS-L2 (champion)", d_pls, np.load("data/trained_weights_q4_l2_pls.npy"), 2),
]

for name, d, w, L in models:
    q, _ = make_qnn(4, L, 4)   # default estimator = exact statevector
    for scope, X, y in [("200 pts", d["X_test"], d["y_test"]),
                        ("frozen 50", d["X_test"][:50], d["y_test"][:50])]:
        ev = np.asarray(q.forward(X, w)).ravel()
        rm, ma = rmse_kelvin(y, ev, d)
        print(f"{name:18s} ideal on {scope:9s}: RMSE {rm:6.2f} K  MAE {ma:6.2f} K")
    np.save(f"data/ideal_evs50_{'pls_l2' if L == 2 else 'pca_l3'}.npy",
            np.asarray(q.forward(d["X_test"][:50], w)).ravel())
print("ideal evs on the frozen 50 saved to data/ideal_evs50_*.npy")
