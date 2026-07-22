"""Collect the final-week QPU jobs recorded in data/final_runs_jobs.json.

Safe to run repeatedly: pending jobs are reported and skipped, finished jobs
are scored against Kelvin labels and the ideal statevector reference, and
their expectation values, per-point stds, and QPU seconds are saved.

Usage: .venv/bin/python final_qpu_collect.py
"""

import json

import numpy as np

from baselines import rmse_kelvin
from qnn import make_qnn

JOBS_FILE = "data/final_runs_jobs.json"

DATASETS = {
    "test200": ("data/prepared_4d_pls.npz", "X_test", "y_test"),
    "holdout300": ("data/fresh_holdout_pls.npz", "X_hold", "y_hold"),
}


def main():
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    with open(JOBS_FILE) as f:
        records = json.load(f)

    qnn, _ = make_qnn(4, 2, 4)
    done = 0
    for r in records:
        tag = f"{r['dataset']}@{r['backend']}"
        job = service.job(r["job_id"])
        status = str(job.status())
        if "DONE" not in status.upper():
            print(f"{tag}: job {r['job_id']} status {status}")
            continue

        path, xk, yk = DATASETS[r["dataset"]]
        d = dict(np.load(path))
        X, y = d[xk], d[yk]
        w = np.load(r["weights"])

        data = job.result()[0].data
        evs = np.asarray(data.evs).reshape(-1)[: len(X)]
        try:
            stds = np.asarray(data.stds).reshape(-1)[: len(X)]
        except Exception:
            stds = None
        try:
            qs = job.metrics()["usage"]["quantum_seconds"]
        except Exception:
            qs = float("nan")

        ideal = np.asarray(qnn.forward(X, w)).ravel()
        rm, ma = rmse_kelvin(y, evs, d)
        rm_i, _ = rmse_kelvin(y, ideal, d)
        r_ev = np.corrcoef(evs, ideal)[0, 1]

        stem = f"{r['dataset']}_{r['backend'].replace('ibm_', '')}"
        np.save(f"data/qpu_evs_{stem}.npy", evs)
        if stds is not None:
            np.save(f"data/qpu_stds_{stem}.npy", stds)

        print(f"{tag}: RMSE {rm:.2f} K  MAE {ma:.2f} K  "
              f"(ideal {rm_i:.2f} K, r={r_ev:.3f}, {r['rows']} rows, "
              f"{qs:.1f} QPU-s, job {r['job_id']})")
        r.update({"rmse_k": round(rm, 2), "mae_k": round(ma, 2),
                  "ideal_rmse_k": round(rm_i, 2), "r_vs_ideal": round(r_ev, 4),
                  "quantum_seconds": qs, "status": "DONE"})
        done += 1

    with open(JOBS_FILE, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n{done}/{len(records)} jobs finished; results merged into {JOBS_FILE}")


if __name__ == "__main__":
    main()
