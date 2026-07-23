"""Final-allocation experiment battery (July 22-24, access ends July 24).

Six experiment families, submitted per subcommand; all jobs append to
data/final_burn_jobs.json and are scored later by burn_collect.py:

  fullset   every cleaned material (12,136 train_full + 3,034 test_full =
            15,170 points) through the champion on ibm_fez, tiled, chunked
            into ~90-row jobs sharing one Batch
  devices   the 200-point test set on every other exposed backend
            (Heron r3 ibm_pittsburgh/ibm_boston, Heron r2 ibm_kingston,
            Nighthawk ibm_miami); fez + marrakesh already recorded
  tilesel   200-point test set on the 12 BEST vs 12 WORST calibration-ranked
            tiles, same day: hardware-aware placement measured in Kelvin
  tilemap   10 frozen points broadcast identically to all 25 tiles: per-tile
            error vs calibration score, 25 spatial datapoints in one job
  shots     frozen 50, TREX only, at 64 / 256 / 1024 shots: shot-noise floor
            vs device-noise floor (2048-shot arm already recorded)
  depth     the 6-CZ question on PLS 3-layer weights: raw, full mitigation,
            tuned ZNE (noise factors 1/3/5, gate folding, exponential
            extrapolator), tiled; plus PCA 2-layer to complete the
            depth-by-features factorial

Every new binding path is pre-flighted against exact statevector at K=3
before submission (the parameter-binding rule in the README).

Usage:
  .venv/bin/python final_burn.py <fullset|devices|tilesel|tilemap|shots|depth> [--dry-run]
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import tiling_experiment as te
from hardware import param_matrix
from qnn import build_circuit, make_qnn

SHOTS = 2048
REG = "data/final_burn_jobs.json"

PLS = "data/prepared_4d_pls.npz"
PCA = "data/prepared_4d.npz"
W_CHAMP = "data/trained_weights_q4_l2_pls.npy"
W_3L_PLS = "data/trained_weights_q4_l3_pls.npy"
W_2L_PCA = "data/trained_weights_q4_l2.npy"

DEVICES = ["ibm_pittsburgh", "ibm_boston", "ibm_kingston", "ibm_miami"]


# ---------------------------------------------------------------- registry --
def log_job(rec):
    recs = json.load(open(REG)) if os.path.exists(REG) else []
    recs.append(rec)
    with open(REG, "w") as f:
        json.dump(recs, f, indent=2)


def base_rec(kind, desc, backend, job, mode, layers, weights, data, xkey, ykey,
             i0, i1, K, rows, shots, options, **extra):
    return {"kind": kind, "desc": desc, "backend": backend,
            "job_id": job.job_id(), "mode": mode, "layers": layers,
            "weights": weights, "data": data, "xkey": xkey, "ykey": ykey,
            "i0": i0, "i1": i1, "K": K, "rows": rows, "shots": shots,
            "options": options,
            "submitted_utc": datetime.now(timezone.utc).isoformat(), **extra}


# ------------------------------------------------------- tiled machinery ----
def build_pv(tin_, twt_, X_, w_, k_tiles, params_order):
    """Row r, tile k <- point X_[r*k_tiles+k]; identical to the recorded runs."""
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


def build_pv_broadcast(tin_, twt_, X_, w_, k_tiles, params_order):
    """Row r sends the SAME point X_[r] to every tile (per-tile error map)."""
    vals = {}
    for k in range(k_tiles):
        for r in range(len(X_)):
            for p, v in zip(tin_[k], X_[r]):
                vals.setdefault(p, []).append(v)
            for p, v in zip(twt_[k], w_):
                vals.setdefault(p, []).append(v)
    pv_ = np.stack([np.array(vals[p]) for p in params_order], axis=-1)
    return pv_.reshape(len(X_), 1, -1), len(X_)


def preflight_standard(X, w, n_layers):
    te.N_LAYERS = n_layers          # module global read by build_wide_circuit
    wide3, tin3, twt3 = te.build_wide_circuit(3)
    pv3, _ = build_pv(tin3, twt3, X[:9], w, 3, list(wide3.parameters))
    ev = np.asarray(
        StatevectorEstimator().run([(wide3, te.tile_observables(3), pv3)])
        .result()[0].data.evs).reshape(-1)[:9]
    ref = np.asarray(make_qnn(4, n_layers, 4)[0].forward(X[:9], w)).ravel()
    assert np.allclose(ev, ref, atol=1e-9), \
        f"tiled binding mismatch (L={n_layers})! {np.abs(ev - ref).max():.3e}"
    print(f"pre-flight PASS (standard, L={n_layers}): "
          f"max diff {np.abs(ev - ref).max():.2e}")


def preflight_broadcast(X, w, n_layers):
    te.N_LAYERS = n_layers          # module global read by build_wide_circuit
    wide3, tin3, twt3 = te.build_wide_circuit(3)
    pv3, _ = build_pv_broadcast(tin3, twt3, X[:2], w, 3, list(wide3.parameters))
    ev = np.asarray(
        StatevectorEstimator().run([(wide3, te.tile_observables(3), pv3)])
        .result()[0].data.evs)                       # (2 rows, 3 tiles)
    ref = np.asarray(make_qnn(4, n_layers, 4)[0].forward(X[:2], w)).ravel()
    assert np.allclose(ev, ref[:, None], atol=1e-9), \
        f"broadcast binding mismatch! {np.abs(ev - ref[:, None]).max():.3e}"
    print(f"pre-flight PASS (broadcast): max diff "
          f"{np.abs(ev - ref[:, None]).max():.2e}")


def pack(backend, n_layers, k_max=25, subset="best"):
    """Pack + rank buffered tiles, pin layout, verify SWAP-free ISA."""
    te.N_LAYERS = n_layers
    cz_tile = (n_layers - 1) * 3
    adj = te.adjacency(backend)
    all_tiles = te.find_disjoint_tiles(adj, backend.num_qubits, buffer=True)
    te.validate_tiles(all_tiles, adj)
    ranked = te.rank_tiles_by_calibration(all_tiles, backend)
    k = min(len(ranked), k_max)
    chosen = ranked[:k] if subset == "best" else ranked[-k:]
    tiles = [t for _, t in chosen]
    scores = [round(s, 4) for s, _ in chosen]
    print(f"{backend.name}: K={k} ({subset} of {len(ranked)} tiles, "
          f"scores {scores[0]:.3f}..{scores[-1]:.3f})")

    wide, tin, twt = te.build_wide_circuit(k)
    phys = [q for t in tiles for q in t]
    layout = {wide.qubits[i]: phys[i] for i in range(len(phys))}
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      initial_layout=layout)
    isa = pm.run(wide)
    ops = isa.count_ops()
    assert ops.get("swap", 0) == 0, f"SWAPs injected: {ops.get('swap')}"
    assert ops.get("cz", 0) == cz_tile * k, \
        f"expected {cz_tile * k} CZ, got {ops.get('cz')}"
    print(f"  wide ISA: {k*4} qubits, depth {isa.depth()}, cz {ops['cz']}, 0 swaps")
    obs_isa = [o.apply_layout(isa.layout) for o in te.tile_observables(k)]
    return isa, obs_isa, tin, twt, k, scores


# ------------------------------------------------------ estimator options ---
def make_est(mode, res=1, dd=True, twirl=True, shots=SHOTS, zne=False):
    from qiskit_ibm_runtime import EstimatorV2
    est = EstimatorV2(mode=mode)
    est.options.resilience_level = res
    est.options.default_shots = shots
    est.options.dynamical_decoupling.enable = bool(dd)
    if dd:
        est.options.dynamical_decoupling.sequence_type = "XY4"
    est.options.twirling.enable_gates = bool(twirl)
    if zne:
        est.options.resilience.measure_mitigation = True
        est.options.resilience.zne_mitigation = True
        est.options.resilience.zne.noise_factors = (1, 3, 5)
        est.options.resilience.zne.amplifier = "gate_folding"
        est.options.resilience.zne.extrapolator = ("exponential", "linear")
    desc = f"res{res}" + ("+DD(XY4)" if dd else "") + \
           ("+gate_twirl" if twirl else "") + \
           ("+ZNE(1,3,5;gate_folding;exp/lin)" if zne else "") + f" shots={shots}"
    return est, desc


def submit_untiled(est, backend_obj, X, w, n_layers):
    qc, inputs, wparams, obs = build_circuit(4, n_layers, 4)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend_obj)
    isa = pm.run(qc)
    obs_isa = obs.apply_layout(isa.layout)
    pv = param_matrix(X, w, inputs, wparams, isa)
    return est.run([(isa, obs_isa, pv)])


def usage_line(service):
    try:
        u = service.usage()
        print(f"[usage] {u['usage_remaining_seconds']} QPU-s remaining "
              f"on the shared instance")
    except Exception as e:
        print(f"[usage] check failed: {type(e).__name__}")


# ------------------------------------------------------------ experiments ---
def run_fullset(service, dry):
    d = dict(np.load(PLS))
    X = np.vstack([d["X_train_full"], d["X_test_full"]])
    w = np.load(W_CHAMP)
    preflight_standard(X, w, 2)
    if dry:
        print(f"fullset: {len(X)} points, would chunk into "
              f"{int(np.ceil(len(X) / (25 * 90)))} jobs of <=90 rows")
        return
    from qiskit_ibm_runtime import Batch
    fez = service.backend("ibm_fez")
    isa, obs, tin, twt, k, scores = pack(fez, 2)
    per_job = k * 90
    batch = Batch(backend=fez)
    try:
        est, odesc = make_est(batch)
        for i0 in range(0, len(X), per_job):
            i1 = min(i0 + per_job, len(X))
            pv, rows = build_pv(tin, twt, X[i0:i1], w, k, list(isa.parameters))
            job = est.run([(isa, obs, pv)])
            print(f"  fullset[{i0}:{i1}] -> job {job.job_id()} ({rows} rows)")
            log_job(base_rec("fullset", f"all cleaned materials [{i0}:{i1}]",
                             "ibm_fez", job, "tiled", 2, W_CHAMP, PLS,
                             "train_full+test_full", "train_full+test_full",
                             i0, i1, k, rows, SHOTS, odesc))
    finally:
        batch.close()


def run_devices(service, dry):
    d = dict(np.load(PLS))
    X, w = d["X_test"], np.load(W_CHAMP)
    preflight_standard(X, w, 2)
    if dry:
        print(f"devices: would submit test200 tiled on {DEVICES}")
        return
    for name in DEVICES:
        try:
            b = service.backend(name)
            isa, obs, tin, twt, k, scores = pack(b, 2)
            est, odesc = make_est(b)
            pv, rows = build_pv(tin, twt, X, w, k, list(isa.parameters))
            job = est.run([(isa, obs, pv)])
            print(f"  test200 on {name}: job {job.job_id()} ({rows} rows)")
            log_job(base_rec("devices", f"test200 replication on {name}",
                             name, job, "tiled", 2, W_CHAMP, PLS,
                             "X_test", "y_test", 0, 200, k, rows, SHOTS, odesc))
        except Exception as e:
            print(f"  {name} FAILED to submit: {type(e).__name__}: {e}")


def run_tilesel(service, dry):
    d = dict(np.load(PLS))
    X, w = d["X_test"], np.load(W_CHAMP)
    preflight_standard(X, w, 2)
    if dry:
        print("tilesel: would submit test200 on best-12 and worst-12 tiles")
        return
    from qiskit_ibm_runtime import Batch
    fez = service.backend("ibm_fez")
    batch = Batch(backend=fez)
    try:
        for subset in ("best", "worst"):
            isa, obs, tin, twt, k, scores = pack(fez, 2, k_max=12, subset=subset)
            est, odesc = make_est(batch)
            pv, rows = build_pv(tin, twt, X, w, k, list(isa.parameters))
            job = est.run([(isa, obs, pv)])
            print(f"  test200 {subset}-12 tiles: job {job.job_id()}")
            log_job(base_rec("tilesel", f"test200 on {subset}-12 tiles",
                             "ibm_fez", job, "tiled", 2, W_CHAMP, PLS,
                             "X_test", "y_test", 0, 200, k, rows, SHOTS, odesc,
                             tile_subset=subset, tile_scores=scores))
    finally:
        batch.close()


def run_tilemap(service, dry):
    d = dict(np.load(PLS))
    X, w = d["X_test"][:10], np.load(W_CHAMP)
    preflight_broadcast(X, w, 2)
    if dry:
        print("tilemap: would broadcast 10 points to all 25 tiles (10 rows)")
        return
    fez = service.backend("ibm_fez")
    isa, obs, tin, twt, k, scores = pack(fez, 2)
    est, odesc = make_est(fez)
    pv, rows = build_pv_broadcast(tin, twt, X, w, k, list(isa.parameters))
    job = est.run([(isa, obs, pv)])
    print(f"  tilemap: job {job.job_id()} ({rows} rows x {k} tiles)")
    log_job(base_rec("tilemap", "10 frozen points broadcast to all tiles",
                     "ibm_fez", job, "broadcast", 2, W_CHAMP, PLS,
                     "X_test", "y_test", 0, 10, k, rows, SHOTS, odesc,
                     tile_scores=scores))


def run_shots(service, dry):
    d = dict(np.load(PLS))
    X, w = d["X_test"][:50], np.load(W_CHAMP)
    if dry:
        print("shots: would submit frozen50 untiled TREX-only at 64/256/1024")
        return
    from qiskit_ibm_runtime import Batch
    fez = service.backend("ibm_fez")
    batch = Batch(backend=fez)
    try:
        for s in (64, 256, 1024):
            est, odesc = make_est(batch, res=1, dd=False, twirl=False, shots=s)
            job = submit_untiled(est, fez, X, w, 2)
            print(f"  shots={s}: job {job.job_id()}")
            log_job(base_rec("shots", f"frozen50 TREX-only at {s} shots",
                             "ibm_fez", job, "untiled", 2, W_CHAMP, PLS,
                             "X_test", "y_test", 0, 50, None, 50, s, odesc))
    finally:
        batch.close()


def run_depth(service, dry):
    d_pls = dict(np.load(PLS))
    d_pca = dict(np.load(PCA))
    X50 = d_pls["X_test"][:50]
    w3 = np.load(W_3L_PLS)
    w2pca = np.load(W_2L_PCA)
    preflight_standard(X50, w3, 3)
    if dry:
        print("depth: PLS-3L raw/full-mit/ZNE untiled + tiled, PCA-2L untiled")
        return
    from qiskit_ibm_runtime import Batch
    fez = service.backend("ibm_fez")
    batch = Batch(backend=fez)
    try:
        arms = [("raw", dict(res=0, dd=False, twirl=False)),
                ("full_mitigation", dict(res=1, dd=True, twirl=True)),
                ("zne_tuned", dict(res=1, dd=False, twirl=True, zne=True))]
        for name, kw in arms:
            est, odesc = make_est(batch, **kw)
            job = submit_untiled(est, fez, X50, w3, 3)
            print(f"  PLS-3L {name}: job {job.job_id()}")
            log_job(base_rec("depth", f"PLS-3L (6 CZ) frozen50 {name}",
                             "ibm_fez", job, "untiled", 3, W_3L_PLS, PLS,
                             "X_test", "y_test", 0, 50, None, 50, SHOTS, odesc,
                             arm=name))
        isa, obs, tin, twt, k, scores = pack(fez, 3)
        est, odesc = make_est(batch)
        pv, rows = build_pv(tin, twt, X50, w3, k, list(isa.parameters))
        job = est.run([(isa, obs, pv)])
        print(f"  PLS-3L tiled: job {job.job_id()}")
        log_job(base_rec("depth", "PLS-3L (6 CZ) frozen50 tiled",
                         "ibm_fez", job, "tiled", 3, W_3L_PLS, PLS,
                         "X_test", "y_test", 0, 50, k, rows, SHOTS, odesc,
                         arm="tiled"))
        est, odesc = make_est(batch)
        job = submit_untiled(est, fez, d_pca["X_test"][:50], w2pca, 2)
        print(f"  PCA-2L full mitigation: job {job.job_id()}")
        log_job(base_rec("depth", "PCA-2L (3 CZ) frozen50 full mitigation",
                         "ibm_fez", job, "untiled", 2, W_2L_PCA, PCA,
                         "X_test", "y_test", 0, 50, None, 50, SHOTS, odesc,
                         arm="pca_2l"))
    finally:
        batch.close()


def run_anchor(service, dry):
    """Identical tiled test200 on ibm_fez, resubmitted on later days: gives
    the headline number a cross-calibration, cross-day error bar."""
    d = dict(np.load(PLS))
    X, w = d["X_test"], np.load(W_CHAMP)
    preflight_standard(X, w, 2)
    if dry:
        print("anchor: would submit tiled test200 on ibm_fez")
        return
    b = service.backend("ibm_fez")
    isa, obs, tin, twt, k, scores = pack(b, 2)
    est, odesc = make_est(b)
    pv, rows = build_pv(tin, twt, X, w, k, list(isa.parameters))
    job = est.run([(isa, obs, pv)])
    print(f"  anchor test200 on ibm_fez: job {job.job_id()}")
    log_job(base_rec("anchor", "cross-day stability anchor (test200)",
                     "ibm_fez", job, "tiled", 2, W_CHAMP, PLS,
                     "X_test", "y_test", 0, 200, k, rows, SHOTS, odesc))


RUNNERS = {"fullset": run_fullset, "devices": run_devices,
           "tilesel": run_tilesel, "tilemap": run_tilemap,
           "shots": run_shots, "depth": run_depth, "anchor": run_anchor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=sorted(RUNNERS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        fake = FakeFez()
        service = None
        # dry runs only need pre-flights + local packing checks
        if args.kind in ("fullset", "devices", "tilesel", "tilemap", "depth"):
            pack(fake, 3 if args.kind == "depth" else 2)
        RUNNERS[args.kind](service, dry=True)
        return
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    usage_line(service)
    RUNNERS[args.kind](service, dry=False)
    usage_line(service)


if __name__ == "__main__":
    main()
