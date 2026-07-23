"""Quantum-hardware virtual screen of generated superconductor candidates.

Generates hypothetical formulas with the same generator families as the
volunteer work unit (cuprate-biased, iron-biased, random), featurizes them
into the UCI 81-feature space, projects them through the champion's EXACT
input pipeline (prep_data internals: StandardScaler -> PLS-4 -> angle
scaling, fitted on the seed-42 training split and verified against the
shipped npz before use), and submits them through the tiled champion on
ibm_fez. burn_collect.py scores the jobs: hardware Tc per candidate,
hardware-vs-ideal rank agreement, and the top of the shortlist.

Candidates whose PLS angles fall outside the encoding range [-pi/2, pi/2]
are clipped and flagged out_of_range in the CSV (the circuit never saw
angles beyond that range in training).

Usage: .venv/bin/python candidates_qpu_screen.py [--n 4000] [--dry-run]
"""

import argparse
import json

import numpy as np
import pandas as pd

import final_burn as fb
from prep_data import inverse_target, prepare
from qnn import make_qnn

SEED = 20260723
CSV = "data/candidates_screen.csv"
NPZ = "data/candidates_screen_inputs.npz"


def generate(n, etable, basis, rng):
    from featurize_formula import featurize
    cuprate_a = [e for e in ["La", "Ba", "Sr", "Ca", "Y", "Bi", "Tl", "Hg", "Nd"]
                 if e in basis]
    iron_x = [e for e in ["As", "Se", "Te", "P"] if e in basis]
    pool = [e for e in ["O", "Cu", "La", "Ba", "Sr", "Ca", "Y", "Bi", "Tl", "Hg",
                        "Fe", "As", "Se", "Te", "H", "Mg", "B", "Nb", "Ti", "Sn",
                        "Pb", "K", "Rb", "Ni", "Co", "P", "S", "Sb", "Ge", "Si",
                        "F", "C", "Zr", "V", "Mo"] if e in basis]
    modes = rng.choice(["cuprate", "iron", "random"], n, p=[0.35, 0.15, 0.50])
    forms, v81s, gens = [], [], []
    feat_cols = None
    for mode in modes:
        if mode == "cuprate":
            els = list(rng.choice(cuprate_a, rng.integers(1, 3),
                                  replace=False)) + ["Cu", "O"]
        elif mode == "iron":
            els = list(dict.fromkeys(
                ["Fe", str(rng.choice(iron_x))] + list(rng.choice(cuprate_a, 1))))
        else:
            els = list(rng.choice(pool, rng.integers(2, 6), replace=False))
        amts = np.round(rng.uniform(0.1, 6.0, len(els)), 2)
        formula = "".join(f"{e}{a}" for e, a in zip(els, amts))
        try:
            f = featurize(formula, etable)
        except ValueError:
            continue
        if feat_cols is None:
            feat_cols = [c for c in pd.read_csv("data/train.csv", nrows=1).columns
                         if c != "critical_temp"]
        forms.append(formula)
        v81s.append(f[feat_cols].values.astype(float))
        gens.append(mode)
    return forms, np.array(v81s), gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # champion input pipeline, refit deterministically and verified
    out = prepare(reduction="pls", save=False, return_internals=True)
    shipped = dict(np.load(fb.PLS))
    assert np.allclose(out["X_test"], shipped["X_test"], atol=1e-10), \
        "refit pipeline does not reproduce shipped X_test"
    print("pipeline verified: refit X_test == shipped X_test")
    internals = out["_internals"]

    etable = pd.read_csv("data/element_properties_uci.csv", index_col=0)
    basis = [e for e in etable.index if not etable.loc[e].isna().any()]
    rng = np.random.default_rng(SEED)
    forms, V81, gens = generate(args.n, etable, basis, rng)
    print(f"generated + featurized {len(forms)}/{args.n} candidates")

    Z = internals["x_scaler"].transform(internals["project"](V81))
    oor = (np.abs(Z) > np.pi / 2).any(axis=1)
    X_cand = np.clip(Z, -np.pi / 2, np.pi / 2)
    print(f"angles clipped for {oor.sum()} candidates (flagged out_of_range)")

    w = np.load(fb.W_CHAMP)
    ideal = np.asarray(make_qnn(4, 2, 4)[0].forward(X_cand, w)).ravel()
    tc_ideal = inverse_target(np.clip(ideal, -1, 1),
                              out["y_min"], out["y_max"])

    np.savez(NPZ, X_cand=X_cand, y_min=out["y_min"], y_max=out["y_max"])
    pd.DataFrame({"formula": forms, "gen": gens, "out_of_range": oor,
                  "tc_ideal_K": np.round(tc_ideal, 2)}).to_csv(CSV, index=False)
    print(f"saved {CSV} + {NPZ}; ideal Tc: median "
          f"{np.median(tc_ideal):.1f} K, max {tc_ideal.max():.1f} K")

    fb.preflight_standard(X_cand, w, 2)
    if args.dry_run:
        print(f"dry run: would submit {len(X_cand)} candidates in tiled chunks")
        return

    from qiskit_ibm_runtime import Batch, QiskitRuntimeService
    service = QiskitRuntimeService()
    fb.usage_line(service)
    fez = service.backend("ibm_fez")
    isa, obs, tin, twt, k, _ = fb.pack(fez, 2)
    per_job = k * 90
    batch = Batch(backend=fez)
    try:
        est, odesc = fb.make_est(batch)
        for i0 in range(0, len(X_cand), per_job):
            i1 = min(i0 + per_job, len(X_cand))
            pv, rows = fb.build_pv(tin, twt, X_cand[i0:i1], w, k,
                                   list(isa.parameters))
            job = est.run([(isa, obs, pv)])
            print(f"  candidates[{i0}:{i1}] -> job {job.job_id()} ({rows} rows)")
            fb.log_job(fb.base_rec(
                "candidates", f"hardware screen of generated candidates [{i0}:{i1}]",
                "ibm_fez", job, "tiled", 2, fb.W_CHAMP, NPZ,
                "X_cand", None, i0, i1, k, rows, fb.SHOTS, odesc))
    finally:
        batch.close()
    fb.usage_line(service)


if __name__ == "__main__":
    main()
