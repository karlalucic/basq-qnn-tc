"""Wave-2 burn: everything that does not depend on pending results.

  fullset_pitt  all 15,170 cleaned materials on ibm_pittsburgh (Heron r3):
                the whole dataset on a second processor generation
  depth200      the 6-CZ 3-layer model, tiled, on the FULL 200-point test
                set (the depth study so far used the frozen 50)
  pec           probabilistic error cancellation on the champion frozen 50,
                untiled, pec.max_overhead=8: the one mitigation family never
                measured, completing raw/TREX/DD/twirl/ZNE/PEC
  anchor2       same-day repeat of the tiled test200 anchor (intra-day
                repeatability, complementing the cross-day chain)

Deliberately absent: any holdout rerun. The preregistered 300-material
result stands exactly as committed; re-running it on friendlier tiles would
be post-hoc shopping.

Usage: .venv/bin/python wave2.py [--dry-run]
"""

import argparse
import json
import os

import numpy as np

import final_burn as fb

W_3L = fb.W_3L_PLS


def submitted_keys():
    """(kind, i0, desc) of every record already in the registry, so retry
    runs after partial API failures never double-submit."""
    if not os.path.exists(fb.REG):
        return set()
    return {(r["kind"], r["i0"], r["desc"]) for r in json.load(open(fb.REG))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = dict(np.load(fb.PLS))
    X_full = np.vstack([d["X_train_full"], d["X_test_full"]])
    X200, X50 = d["X_test"], d["X_test"][:50]
    w = np.load(fb.W_CHAMP)
    w3 = np.load(W_3L)

    fb.preflight_standard(X_full, w, 2)
    fb.preflight_standard(X200, w3, 3)
    if args.dry_run:
        print("dry run: fullset_pitt x7, depth200, pec, anchor2")
        return

    from qiskit_ibm_runtime import Batch, QiskitRuntimeService
    service = QiskitRuntimeService()
    fb.usage_line(service)

    # --- whole dataset on Heron r3 -----------------------------------------
    pitt = service.backend("ibm_pittsburgh")
    isa, obs, tin, twt, k, _ = fb.pack(pitt, 2)
    per_job = k * 90
    try:
        batch = Batch(backend=pitt)
    except Exception as e:
        # sessions API flaking (HTTP 500s): plain job mode still works
        print(f"  Batch creation failed ({type(e).__name__}); using job mode")
        batch = None
    try:
        mode = batch if batch is not None else pitt
        seen = submitted_keys()
        est, odesc = fb.make_est(mode)
        for i0 in range(0, len(X_full), per_job):
            i1 = min(i0 + per_job, len(X_full))
            desc = f"all cleaned materials on r3 [{i0}:{i1}]"
            if ("fullset_pitt", i0, desc) in seen:
                print(f"  fullset_pitt[{i0}:{i1}] already submitted, skipping")
                continue
            pv, rows = fb.build_pv(tin, twt, X_full[i0:i1], w, k,
                                   list(isa.parameters))
            job = est.run([(isa, obs, pv)])
            print(f"  fullset_pitt[{i0}:{i1}] -> job {job.job_id()}")
            fb.log_job(fb.base_rec(
                "fullset_pitt", f"all cleaned materials on r3 [{i0}:{i1}]",
                "ibm_pittsburgh", job, "tiled", 2, fb.W_CHAMP, fb.PLS,
                "train_full+test_full", "train_full+test_full",
                i0, i1, k, rows, fb.SHOTS, odesc))
    finally:
        if batch is not None:
            batch.close()

    fez = service.backend("ibm_fez")
    seen = submitted_keys()

    # --- 6-CZ model on the full test set -----------------------------------
    if ("depth200", 0, "PLS-3L tiled on full test200") in seen:
        print("  depth200 already submitted, skipping")
    else:
        isa, obs, tin, twt, k, _ = fb.pack(fez, 3)
        est, odesc = fb.make_est(fez)
        pv, rows = fb.build_pv(tin, twt, X200, w3, k, list(isa.parameters))
        job = est.run([(isa, obs, pv)])
        print(f"  depth200 (PLS-3L tiled, test200): job {job.job_id()}")
        fb.log_job(fb.base_rec("depth200", "PLS-3L tiled on full test200",
                               "ibm_fez", job, "tiled", 3, W_3L, fb.PLS,
                               "X_test", "y_test", 0, 200, k, rows, fb.SHOTS,
                               odesc))

    # --- PEC on the champion frozen 50 -------------------------------------
    if ("pec", 0, "champion frozen50 with PEC") in seen:
        print("  pec already submitted, skipping")
    else:
        est, odesc = fb.make_est(fez, res=1, dd=False, twirl=True)
        est.options.resilience.pec_mitigation = True
        est.options.resilience.pec.max_overhead = 8
        odesc += "+PEC(max_overhead=8)"
        job = fb.submit_untiled(est, fez, X50, w, 2)
        print(f"  PEC frozen50: job {job.job_id()}")
        fb.log_job(fb.base_rec("pec", "champion frozen50 with PEC", "ibm_fez",
                               job, "untiled", 2, fb.W_CHAMP, fb.PLS,
                               "X_test", "y_test", 0, 50, None, 50, fb.SHOTS,
                               odesc))

    # --- same-day repeat anchor --------------------------------------------
    if ("anchor", 0, "same-day repeat anchor (test200)") in seen:
        print("  anchor2 already submitted, skipping")
    else:
        isa, obs, tin, twt, k, _ = fb.pack(fez, 2)
        est, odesc = fb.make_est(fez)
        pv, rows = fb.build_pv(tin, twt, X200, w, k, list(isa.parameters))
        job = est.run([(isa, obs, pv)])
        print(f"  anchor2 test200: job {job.job_id()}")
        fb.log_job(fb.base_rec("anchor", "same-day repeat anchor (test200)",
                               "ibm_fez", job, "tiled", 2, fb.W_CHAMP, fb.PLS,
                               "X_test", "y_test", 0, 200, k, rows, fb.SHOTS,
                               odesc))
    fb.usage_line(service)


if __name__ == "__main__":
    main()
