"""Hardware-in-the-loop SPSA fine-tuning of the champion QNN.

The champion was trained in ideal simulation; the QPU sees it through its
own systematic biases (readout tilt, angle miscalibration, crosstalk). This
runs SPSA *on the quantum computer*: each iteration evaluates the loss at
w + c*delta and w - c*delta in ONE tiled job (both weight sets ride the same
wide circuit as separate parameter rows), so the two arms share a calibration
and shot-noise partially cancels in the gradient estimate.

Protocol:
  - loss set: first 150 training rows (fixed), loss = MSE in scaled ev space
  - K <= 25 calibration-ranked tiles on ibm_fez, 2048 shots,
    TREX + DD(XY4) + gate twirling (identical to the headline runs)
  - SPSA: c_k = C0/(k+1)^0.101, a_k = A0/(k+1+5)^0.602, per-coordinate step
    clipped to 0.05 rad
  - every iteration checkpoints weights + a jsonl log line, so the loop is
    resumable after any interruption
  - before/after verdict: one job evaluates ORIGINAL and FINE-TUNED weights
    on the full 200-point test set (two segments, same wide circuit, same
    calibration) -- that A/B is the deliverable
  - budget guard: stops when the shared instance drops below --floor QPU-s

Usage:
  .venv/bin/python hw_finetune.py --iters 40 --floor 2500
  .venv/bin/python hw_finetune.py --dry-run       # statevector pre-flight only
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import tiling_experiment as te
from baselines import rmse_kelvin
from qnn import make_qnn

SHOTS = 2048
N_LOSS = 150
C0, A0, A_STAB, STEP_CLIP = 0.10, 0.03, 5, 0.05
W_CHAMP = "data/trained_weights_q4_l2_pls.npy"
CKPT = "data/hw_finetune_weights.npy"
LOG = "data/hw_finetune_log.jsonl"
PLS = "data/prepared_4d_pls.npz"


def build_pv_multi(tin_, twt_, segments, k_tiles, params_order):
    """Stack segments of (X, weights): each contributes ceil(len(X)/K) rows
    with its OWN weight vector on every tile. Returns (pv, boundaries) where
    boundaries[i] = (row0, n_points) locates segment i in the row-major evs."""
    vals, boundaries, row0 = {}, [], 0
    for X_, w_ in segments:
        n_rows = int(np.ceil(len(X_) / k_tiles))
        Xp_ = np.vstack([X_, np.repeat(X_[-1:], n_rows * k_tiles - len(X_), axis=0)])
        for k in range(k_tiles):
            for r in range(n_rows):
                pt = Xp_[r * k_tiles + k]
                for p, v in zip(tin_[k], pt):
                    vals.setdefault(p, []).append(v)
                for p, v in zip(twt_[k], w_):
                    vals.setdefault(p, []).append(v)
        boundaries.append((row0, len(X_)))
        row0 += n_rows
    pv_ = np.stack([np.array(vals[p]) for p in params_order], axis=-1)
    return pv_.reshape(row0, 1, -1), boundaries


def split_evs(evs, boundaries, k_tiles):
    """Row-major flatten per segment, truncating each segment's padding."""
    out = []
    for row0, n in boundaries:
        n_rows = int(np.ceil(n / k_tiles))
        out.append(np.asarray(evs[row0:row0 + n_rows]).reshape(-1)[:n])
    return out


def preflight_multi(X, w):
    te.N_LAYERS = 2
    wide3, tin3, twt3 = te.build_wide_circuit(3)
    segs = [(X[:5], w), (X[:4], 0.9 * w)]
    pv3, bnd = build_pv_multi(tin3, twt3, segs, 3, list(wide3.parameters))
    evs = np.asarray(
        StatevectorEstimator().run([(wide3, te.tile_observables(3), pv3)])
        .result()[0].data.evs)
    got = split_evs(evs, bnd, 3)
    qnn = make_qnn(4, 2, 4)[0]
    for (X_, w_), g in zip(segs, got):
        ref = np.asarray(qnn.forward(X_, w_)).ravel()
        assert np.allclose(g, ref, atol=1e-9), \
            f"multi-segment binding mismatch! {np.abs(g - ref).max():.3e}"
    print("pre-flight PASS (multi-segment): both weight sets match statevector")


def remaining(service):
    try:
        return service.usage()["usage_remaining_seconds"]
    except Exception:
        return None


def log_line(**kw):
    kw["utc"] = datetime.now(timezone.utc).isoformat()
    with open(LOG, "a") as f:
        f.write(json.dumps(kw) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--floor", type=int, default=2500,
                    help="stop when shared-instance QPU-s drop below this")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = dict(np.load(PLS))
    X_ft, y_ft = d["X_train"][:N_LOSS], d["y_train"][:N_LOSS]
    X_te, y_te = d["X_test"], d["y_test"]
    w0 = np.load(W_CHAMP)
    preflight_multi(X_ft, w0)
    if args.dry_run:
        return

    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService, Session
    service = QiskitRuntimeService()
    fez = service.backend("ibm_fez")

    te.N_LAYERS = 2
    adj = te.adjacency(fez)
    ranked = te.rank_tiles_by_calibration(
        te.find_disjoint_tiles(adj, fez.num_qubits, buffer=True), fez)
    k = min(len(ranked), 25)
    tiles = [t for _, t in ranked[:k]]
    wide, tin, twt = te.build_wide_circuit(k)
    layout = {wide.qubits[i]: q for i, q in
              enumerate(q for t in tiles for q in t)}
    pm = generate_preset_pass_manager(optimization_level=3, backend=fez,
                                      initial_layout=layout)
    isa = pm.run(wide)
    ops = isa.count_ops()
    assert ops.get("swap", 0) == 0 and ops.get("cz", 0) == 3 * k
    obs = [o.apply_layout(isa.layout) for o in te.tile_observables(k)]
    order = list(isa.parameters)
    print(f"K={k} tiles, wide ISA depth {isa.depth()}, budget floor {args.floor}")

    # resume from checkpoint if one exists
    w = np.load(CKPT) if os.path.exists(CKPT) else w0.copy()
    it0 = sum(1 for _ in open(LOG)) if os.path.exists(LOG) else 0
    if it0:
        print(f"resuming from checkpoint at iteration {it0}")

    rng = np.random.default_rng(2026_07_23)
    for _ in range(it0):
        rng.integers(0, 2, size=w.size)      # replay RNG stream on resume

    with Session(backend=fez) as session:
        est = EstimatorV2(mode=session)
        est.options.resilience_level = 1
        est.options.default_shots = SHOTS
        est.options.dynamical_decoupling.enable = True
        est.options.dynamical_decoupling.sequence_type = "XY4"
        est.options.twirling.enable_gates = True

        for it in range(it0, args.iters):
            rem = remaining(service)
            if rem is not None and rem < args.floor:
                print(f"budget floor hit ({rem} < {args.floor}), stopping")
                break
            ck = C0 / (it + 1) ** 0.101
            ak = A0 / (it + 1 + A_STAB) ** 0.602
            delta = rng.integers(0, 2, size=w.size) * 2.0 - 1.0
            pv, bnd = build_pv_multi(tin, twt,
                                     [(X_ft, w + ck * delta),
                                      (X_ft, w - ck * delta)], k, order)
            job = est.run([(isa, obs, pv)])
            evp, evm = split_evs(job.result()[0].data.evs, bnd, k)
            lp = float(np.mean((evp - y_ft) ** 2))
            lm = float(np.mean((evm - y_ft) ** 2))
            step = np.clip(ak * (lp - lm) / (2 * ck) * delta,
                           -STEP_CLIP, STEP_CLIP)
            w = w - step
            np.save(CKPT, w)
            log_line(iter=it, loss_plus=lp, loss_minus=lm,
                     step_norm=float(np.linalg.norm(step)),
                     job_id=job.job_id())
            print(f"iter {it:3d}: L+ {lp:.5f} L- {lm:.5f} "
                  f"|step| {np.linalg.norm(step):.4f} job {job.job_id()}",
                  flush=True)

        # before/after verdict on the full test set, one job, one calibration
        pv, bnd = build_pv_multi(tin, twt, [(X_te, w0), (X_te, w)], k, order)
        job = est.run([(isa, obs, pv)])
        ev0, ev1 = split_evs(job.result()[0].data.evs, bnd, k)

    rm0, _ = rmse_kelvin(y_te, ev0, d)
    rm1, _ = rmse_kelvin(y_te, ev1, d)
    np.save("data/qpu_evs_test200_finetune_ab.npy", np.stack([ev0, ev1]))
    log_line(verdict=True, rmse_original=round(rm0, 2),
             rmse_finetuned=round(rm1, 2), job_id=job.job_id())
    print(f"\nVERDICT (same job, same calibration): original {rm0:.2f} K "
          f"vs fine-tuned {rm1:.2f} K on the full 200-point test set")
    print(f"fine-tuned weights: {CKPT}; log: {LOG}")


if __name__ == "__main__":
    main()
