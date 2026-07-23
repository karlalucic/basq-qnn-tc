"""Day-2 model jobs: hardware multiseed A/B + no-entanglement control.

  multiseed  one tiled job carries seed-17 AND seed-27 champion weights as
             separate segments over the full 200-point test set (same wide
             circuit, same calibration), giving the hardware headline a
             cross-seed comparison to go with seed 7's recorded runs
  noent      the entangle=False control (identical circuit minus the 3 CZ
             gates, retrained) on the frozen 50, untiled, full mitigation:
             the same protocol as the 22.89 K headline job

Usage: .venv/bin/python day2_models.py [--dry-run]
"""

import argparse

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import final_burn as fb
from hardware import param_matrix
from hw_finetune import build_pv_multi, preflight_multi
from qnn import build_circuit

W17 = "data/trained_weights_q4_l2_pls_seed17.npy"
W27 = "data/trained_weights_q4_l2_pls_seed27.npy"
WNO = "data/trained_weights_q4_l2_pls_noent.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = dict(np.load(fb.PLS))
    X, X50 = d["X_test"], d["X_test"][:50]
    w17, w27, wno = np.load(W17), np.load(W27), np.load(WNO)
    preflight_multi(X, w17)

    # no-entanglement ISA sanity: the control must have zero 2-qubit gates
    qc, inputs, wparams, obs = build_circuit(4, 2, 4, entangle=False)
    assert qc.count_ops().get("cz", 0) == 0

    if args.dry_run:
        print("dry run: multiseed 2x200 tiled + noent frozen50 untiled")
        return

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    fb.usage_line(service)
    fez = service.backend("ibm_fez")

    isa, obs_t, tin, twt, k, _ = fb.pack(fez, 2)
    est, odesc = fb.make_est(fez)
    pv, bnd = build_pv_multi(tin, twt, [(X, w17), (X, w27)], k,
                             list(isa.parameters))
    job = est.run([(isa, obs_t, pv)])
    print(f"  multiseed (17+27) test200: job {job.job_id()}")
    fb.log_job(fb.base_rec("multiseed", "seeds 17+27 on test200, one job",
                           "ibm_fez", job, "tiled", 2, W17, fb.PLS,
                           "X_test", "y_test", 0, 200, k,
                           int(pv.shape[0]), fb.SHOTS, odesc,
                           seg_weights=[W17, W27]))

    pm = generate_preset_pass_manager(optimization_level=3, backend=fez)
    isa_n = pm.run(qc)
    assert isa_n.count_ops().get("cz", 0) == 0, "CZ appeared in noent ISA"
    obs_n = obs.apply_layout(isa_n.layout)
    est, odesc = fb.make_est(fez)
    pv_n = param_matrix(X50, wno, inputs, wparams, isa_n)
    job = est.run([(isa_n, obs_n, pv_n)])
    print(f"  noent frozen50: job {job.job_id()}")
    fb.log_job(fb.base_rec("noent", "no-entanglement control, frozen50",
                           "ibm_fez", job, "untiled", 2, WNO, fb.PLS,
                           "X_test", "y_test", 0, 50, None, 50, fb.SHOTS,
                           odesc, entangle=False))
    fb.usage_line(service)


if __name__ == "__main__":
    main()
