"""Single-vs-tiled A/B on real hardware (D2/D5 payoff experiment).

Packs K disjoint 4-qubit chains (with buffer qubits) onto the live ibm_fez
target, loads the champion ansatz onto every tile, and evaluates the frozen
50 test points in ceil(50/K) parameter rows instead of 50 — same shots, same
mitigation. A fresh single-circuit reference runs in the same session so the
comparison shares one calibration. Success = tiled RMSE ~= single RMSE at a
fraction of the QPU seconds.
"""

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

import tiling_experiment as te
from baselines import rmse_kelvin
from hardware import qpu_inference

te.N_LAYERS = 2                     # champion config (3 CZ per tile)
SHOTS = 2048

d = dict(np.load("data/prepared_4d_pls.npz"))
X = d["X_test"][:50]
y = d["y_test"][:50]
w = np.load("data/trained_weights_q4_l2_pls.npy")

service = QiskitRuntimeService()
backend = service.backend("ibm_fez")
print("backend:", backend.name)

# --- pack + rank tiles on today's calibration -----------------------------
adj = te.adjacency(backend)
tiles = te.find_disjoint_tiles(adj, backend.num_qubits, buffer=True)
te.validate_tiles(tiles, adj)
ranked = te.rank_tiles_by_calibration(tiles, backend)
K = min(len(ranked), 25)
tiles = [t for _, t in ranked[:K]]          # best-calibrated K tiles
print(f"K={K} buffered tiles (best score {ranked[0][0]:.4f}, "
      f"worst kept {ranked[K-1][0]:.4f})")

# --- wide circuit, pinned layout, SWAP-free check --------------------------
wide, tin, twt = te.build_wide_circuit(K)
phys = [q for t in tiles for q in t]
layout = {wide.qubits[i]: phys[i] for i in range(len(phys))}
pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                  initial_layout=layout)
isa = pm.run(wide)
ops = isa.count_ops()
assert ops.get("swap", 0) == 0, f"SWAPs injected: {ops.get('swap')}"
assert ops.get("cz", 0) == 3 * K, f"expected {3*K} CZ, got {ops.get('cz')}"
print(f"wide ISA: {K*4} qubits, depth {isa.depth()}, cz {ops['cz']}, 0 swaps")

obs_isa = [o.apply_layout(isa.layout) for o in te.tile_observables(K)]

# --- parameter rows: row r evaluates points [r*K, (r+1)*K) ------------------
def build_pv(tin_, twt_, X_, w_, k_tiles, params_order):
    """(rows, 1, n_params) value tensor; row r, tile k <- point X_[r*k_tiles+k].
    Column order follows params_order (bare arrays bind by parameter order)."""
    n_rows = int(np.ceil(len(X_) / k_tiles))
    Xp_ = np.vstack([X_, np.repeat(X_[-1:], n_rows * k_tiles - len(X_), axis=0)])
    vals = {}
    for k in range(k_tiles):
        for r in range(n_rows):
            pt = Xp_[r * k_tiles + k]
            for p, v in zip(tin_[k], pt):
                vals.setdefault(p, []).append(v)
            for p, v in zip(twt_[k], w_):
                vals.setdefault(p, []).append(v)
    pv_ = np.stack([np.array(vals[p]) for p in params_order], axis=-1)
    return pv_.reshape(n_rows, 1, -1), n_rows


# --- pre-flight: exact statevector check of the SAME binding code at K=3 ----
from qiskit.primitives import StatevectorEstimator
from qnn import make_qnn

wide3, tin3, twt3 = te.build_wide_circuit(3)          # 12 qubits: simulable
pv3, _ = build_pv(tin3, twt3, X[:9], w, 3, list(wide3.parameters))
ev_check = np.asarray(
    StatevectorEstimator().run([(wide3, te.tile_observables(3), pv3)])
    .result()[0].data.evs).reshape(-1)[:9]
ref = np.asarray(make_qnn(4, 2, 4)[0].forward(X[:9], w)).ravel()
assert np.allclose(ev_check, ref, atol=1e-9), \
    f"tiled binding mismatch! max diff {np.abs(ev_check - ref).max():.3e}"
print(f"pre-flight PASS: K=3 tiled statevector == EstimatorQNN.forward "
      f"(max diff {np.abs(ev_check - ref).max():.2e})")

pv, rows = build_pv(tin, twt, X, w, K, list(isa.parameters))

est = EstimatorV2(mode=backend)
est.options.resilience_level = 1
est.options.default_shots = SHOTS
est.options.dynamical_decoupling.enable = True
est.options.dynamical_decoupling.sequence_type = "XY4"
est.options.twirling.enable_gates = True

job = est.run([(isa, obs_isa, pv)])
print("tiled job id:", job.job_id())
evs = np.asarray(job.result()[0].data.evs)   # (rows, K)
print("evs shape:", evs.shape)
ev_flat = evs.reshape(-1)[: len(X)]          # row-major = point order
rm_t, ma_t = rmse_kelvin(y, ev_flat, d)
try:
    qs_t = job.metrics()["usage"]["quantum_seconds"]
except Exception:
    qs_t = float("nan")
print(f"TILED   : RMSE {rm_t:.2f} K  MAE {ma_t:.2f} K  "
      f"({rows} rows, {qs_t:.1f} QPU-s)")
np.save("data/tiled_evs50.npy", ev_flat)

# --- same-session single-circuit reference ---------------------------------
d50 = dict(d); d50["X_test"] = X; d50["y_test"] = y
print("\nsingle-circuit reference (same calibration):")
rm_s, ma_s = qpu_inference(d50, w, n_qubits=4, n_layers=2,
                           backend_name="ibm_fez", shots=SHOTS,
                           resilience_level=1, dd=True, twirl=True,
                           save_evs="data/single_evs50_ab.npy")

print(f"\nA/B: single {rm_s:.2f} K vs tiled {rm_t:.2f} K "
      f"| rows 50 -> {rows} ({50/rows:.0f}x fewer)")
