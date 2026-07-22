"""Final-week QPU runs (access ends 2026-07-24). Submits and exits.

Three tiled-inference jobs, ~65 QPU-s total:
  1. ibm_fez        full 200-point test set          (headline on full test set)
  2. ibm_fez        300-point fresh holdout          (preregistered, see HOLDOUT_PREREG.md)
  3. ibm_marrakesh  full 200-point test set          (second-device replication)

Jobs 1+2 share one Batch so they see one calibration. Every binding path
passes the K=3 statevector pre-flight before anything is sent (the
parameter-binding rule in the README). Poll with final_qpu_collect.py.

Usage:
  .venv/bin/python final_qpu_submit.py            # submit all three
  .venv/bin/python final_qpu_submit.py --dry-run  # pre-flight + packing on FakeFez only
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import tiling_experiment as te
from qnn import make_qnn

te.N_LAYERS = 2                     # champion config (3 CZ per tile)
SHOTS = 2048
MAX_K = 25
WEIGHTS = "data/trained_weights_q4_l2_pls.npy"
JOBS_FILE = "data/final_runs_jobs.json"

DATASETS = {
    "test200": ("data/prepared_4d_pls.npz", "X_test", "y_test"),
    "holdout300": ("data/fresh_holdout_pls.npz", "X_hold", "y_hold"),
}


def load_x(name):
    path, xk, _ = DATASETS[name]
    return dict(np.load(path))[xk]


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


def preflight(X, w):
    """Exact statevector check of the SAME binding code at K=3 (12 qubits)."""
    wide3, tin3, twt3 = te.build_wide_circuit(3)
    pv3, _ = build_pv(tin3, twt3, X[:9], w, 3, list(wide3.parameters))
    ev = np.asarray(
        StatevectorEstimator().run([(wide3, te.tile_observables(3), pv3)])
        .result()[0].data.evs).reshape(-1)[:9]
    ref = np.asarray(make_qnn(4, 2, 4)[0].forward(X[:9], w)).ravel()
    assert np.allclose(ev, ref, atol=1e-9), \
        f"tiled binding mismatch! max diff {np.abs(ev - ref).max():.3e}"
    print(f"pre-flight PASS: K=3 tiled statevector == EstimatorQNN.forward "
          f"(max diff {np.abs(ev - ref).max():.2e})")


def pack_and_transpile(backend):
    """Pack + rank tiles on today's calibration, pin layout, verify SWAP-free."""
    adj = te.adjacency(backend)
    tiles = te.find_disjoint_tiles(adj, backend.num_qubits, buffer=True)
    te.validate_tiles(tiles, adj)
    ranked = te.rank_tiles_by_calibration(tiles, backend)
    k = min(len(ranked), MAX_K)
    tiles = [t for _, t in ranked[:k]]
    print(f"{backend.name}: K={k} buffered tiles "
          f"(best score {ranked[0][0]:.4f}, worst kept {ranked[k-1][0]:.4f})")

    wide, tin, twt = te.build_wide_circuit(k)
    phys = [q for t in tiles for q in t]
    layout = {wide.qubits[i]: phys[i] for i in range(len(phys))}
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      initial_layout=layout)
    isa = pm.run(wide)
    ops = isa.count_ops()
    assert ops.get("swap", 0) == 0, f"SWAPs injected: {ops.get('swap')}"
    assert ops.get("cz", 0) == 3 * k, f"expected {3*k} CZ, got {ops.get('cz')}"
    print(f"  wide ISA: {k*4} qubits, depth {isa.depth()}, cz {ops['cz']}, 0 swaps")
    obs_isa = [o.apply_layout(isa.layout) for o in te.tile_observables(k)]
    meta = {"K": k, "depth": isa.depth(), "cz": int(ops["cz"]),
            "tile_score_best": round(ranked[0][0], 4),
            "tile_score_worst_kept": round(ranked[k - 1][0], 4)}
    return isa, obs_isa, tin, twt, k, meta


def make_estimator(mode):
    from qiskit_ibm_runtime import EstimatorV2
    est = EstimatorV2(mode=mode)
    est.options.resilience_level = 1
    est.options.default_shots = SHOTS
    est.options.dynamical_decoupling.enable = True
    est.options.dynamical_decoupling.sequence_type = "XY4"
    est.options.twirling.enable_gates = True
    return est


def submit(est, isa, obs_isa, tin, twt, k, dataset, w, backend_name, meta, records):
    X = load_x(dataset)
    pv, rows = build_pv(tin, twt, X, w, k, list(isa.parameters))
    job = est.run([(isa, obs_isa, pv)])
    print(f"  submitted {dataset} on {backend_name}: job {job.job_id()} "
          f"({len(X)} points -> {rows} rows)")
    records.append({"job_id": job.job_id(), "backend": backend_name,
                    "dataset": dataset, "n_points": len(X), "rows": rows,
                    "shots": SHOTS, "weights": WEIGHTS,
                    "options": "res1 TREX + DD XY4 + gate twirling",
                    "submitted_utc": datetime.now(timezone.utc).isoformat(),
                    **meta})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="pre-flight + FakeFez packing only, no submission")
    args = ap.parse_args()

    w = np.load(WEIGHTS)
    preflight(load_x("test200"), w)

    if args.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        isa, obs, tin, twt, k, _ = pack_and_transpile(FakeFez())
        for name in DATASETS:
            X = load_x(name)
            pv, rows = build_pv(tin, twt, X, w, k, list(isa.parameters))
            print(f"  {name}: pv shape {pv.shape} ({len(X)} points -> {rows} rows)")
        print("dry run OK, nothing submitted")
        return

    from qiskit_ibm_runtime import Batch, QiskitRuntimeService
    service = QiskitRuntimeService()
    records = []

    fez = service.backend("ibm_fez")
    isa, obs, tin, twt, k, meta = pack_and_transpile(fez)
    batch = Batch(backend=fez)
    try:
        est = make_estimator(batch)
        submit(est, isa, obs, tin, twt, k, "test200", w, "ibm_fez", meta, records)
        submit(est, isa, obs, tin, twt, k, "holdout300", w, "ibm_fez", meta, records)
    finally:
        batch.close()      # stops accepting new jobs; queued jobs still run

    mk = service.backend("ibm_marrakesh")
    isa, obs, tin, twt, k, meta = pack_and_transpile(mk)
    est = make_estimator(mk)
    submit(est, isa, obs, tin, twt, k, "test200", w, "ibm_marrakesh", meta, records)

    with open(JOBS_FILE, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nwrote {JOBS_FILE}; poll with: .venv/bin/python final_qpu_collect.py")


if __name__ == "__main__":
    main()
