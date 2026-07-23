"""Score the final-burn jobs recorded in data/final_burn_jobs.json.

Safe to run repeatedly: pending jobs are skipped, finished ones are scored
against Kelvin labels and per-config ideal statevector references, raw
expectation values and stds are saved as data/burn_evs_<job_id>.npy, and
metrics are merged back into the registry. When all fullset chunks are done
they are stitched into data/qpu_evs_fullset_fez.npy and scored on the
12,136-row train split and the 3,034-row full test split separately.

Usage: .venv/bin/python burn_collect.py
"""

import json

import numpy as np

from baselines import rmse_kelvin
from qnn import make_qnn

REG = "data/final_burn_jobs.json"


def load_xy(rec):
    d = dict(np.load(rec["data"]))
    if rec["xkey"] == "train_full+test_full":
        X = np.vstack([d["X_train_full"], d["X_test_full"]])
        y = np.concatenate([d["y_train_full"], d["y_test_full"]])
    else:
        X, y = d[rec["xkey"]], d[rec["ykey"]]
    return X[rec["i0"]:rec["i1"]], y[rec["i0"]:rec["i1"]], d


_qnn_cache = {}
def ideal_evs(layers, weights_path, X):
    key = (layers, weights_path)
    if key not in _qnn_cache:
        _qnn_cache[key] = (make_qnn(4, layers, 4)[0], np.load(weights_path))
    qnn, w = _qnn_cache[key]
    return np.asarray(qnn.forward(X, w)).ravel()


def main():
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    recs = json.load(open(REG))

    n_done = 0
    for r in recs:
        tag = f"{r['kind']}:{r.get('arm') or r.get('tile_subset') or r['backend']}"
        if r.get("status") == "DONE":
            n_done += 1
            continue
        job = service.job(r["job_id"])
        status = str(job.status())
        if "DONE" not in status.upper():
            print(f"{tag}: {status}")
            continue

        if r["kind"] == "candidates":
            # no labels: save evs, stitch + rank-compare once all chunks land
            data = job.result()[0].data
            np.save(f"data/burn_evs_{r['job_id']}.npy", np.asarray(data.evs))
            try:
                qs = job.metrics()["usage"]["quantum_seconds"]
            except Exception:
                qs = float("nan")
            print(f"{tag} [{r['i0']}:{r['i1']}]: DONE ({qs:.1f} QPU-s)")
            r.update({"status": "DONE", "quantum_seconds": qs})
            n_done += 1
            continue

        X, y, d = load_xy(r)
        data = job.result()[0].data
        raw = np.asarray(data.evs)
        np.save(f"data/burn_evs_{r['job_id']}.npy", raw)
        try:
            np.save(f"data/burn_stds_{r['job_id']}.npy", np.asarray(data.stds))
        except Exception:
            pass
        try:
            qs = job.metrics()["usage"]["quantum_seconds"]
        except Exception:
            qs = float("nan")

        ideal = ideal_evs(r["layers"], r["weights"], X)

        if r["mode"] == "broadcast":
            # raw is (rows=points, K tiles): per-tile error vs calibration score
            per_tile_bias = (raw - ideal[:, None]).mean(axis=0)
            per_tile_rmse = np.sqrt(((raw - ideal[:, None]) ** 2).mean(axis=0))
            scores = np.array(r["tile_scores"])
            r_bias = np.corrcoef(scores, np.abs(per_tile_bias))[0, 1]
            r_rmse = np.corrcoef(scores, per_tile_rmse)[0, 1]
            print(f"{tag}: per-tile |bias| vs calib score r={r_bias:.3f}, "
                  f"per-tile ev-RMSE vs score r={r_rmse:.3f} ({qs:.1f} QPU-s)")
            r.update({"status": "DONE", "quantum_seconds": qs,
                      "r_bias_vs_score": round(float(r_bias), 4),
                      "r_rmse_vs_score": round(float(r_rmse), 4),
                      "per_tile_bias": [round(float(b), 5) for b in per_tile_bias],
                      "per_tile_ev_rmse": [round(float(b), 5) for b in per_tile_rmse]})
            n_done += 1
            continue

        evs = raw.reshape(-1)[: len(X)]
        rm, ma = rmse_kelvin(y, evs, d)
        rm_i, _ = rmse_kelvin(y, ideal, d)
        r_ev = float(np.corrcoef(evs, ideal)[0, 1])
        print(f"{tag}: RMSE {rm:.2f} K MAE {ma:.2f} K "
              f"(ideal {rm_i:.2f} K, r={r_ev:.3f}, {qs:.1f} QPU-s)")
        r.update({"status": "DONE", "rmse_k": round(rm, 2), "mae_k": round(ma, 2),
                  "ideal_rmse_k": round(rm_i, 2), "r_vs_ideal": round(r_ev, 4),
                  "quantum_seconds": qs})
        n_done += 1

    # stitch the fullset once every chunk is in
    chunks = sorted([r for r in recs if r["kind"] == "fullset"],
                    key=lambda r: r["i0"])
    if chunks and all(r.get("status") == "DONE" for r in chunks):
        evs = np.concatenate([
            np.asarray(np.load(f"data/burn_evs_{r['job_id']}.npy"))
            .reshape(-1)[: r["i1"] - r["i0"]] for r in chunks])
        np.save("data/qpu_evs_fullset_fez.npy", evs)
        d = dict(np.load(chunks[0]["data"]))
        n_tr = len(d["y_train_full"])
        rm_tr, _ = rmse_kelvin(d["y_train_full"], evs[:n_tr], d)
        rm_te, ma_te = rmse_kelvin(d["y_test_full"], evs[n_tr:], d)
        X_all = np.vstack([d["X_train_full"], d["X_test_full"]])
        ideal = ideal_evs(2, chunks[0]["weights"], X_all)
        r_all = float(np.corrcoef(evs, ideal)[0, 1])
        print(f"\nFULLSET stitched ({len(evs)} materials on hardware): "
              f"train_full RMSE {rm_tr:.2f} K, test_full RMSE {rm_te:.2f} K "
              f"MAE {ma_te:.2f} K, r_vs_ideal={r_all:.4f}")

    # stitch the candidate screen once every chunk is in
    cand = sorted([r for r in recs if r["kind"] == "candidates"],
                  key=lambda r: r["i0"])
    if cand and all(r.get("status") == "DONE" for r in cand):
        import pandas as pd
        from scipy.stats import spearmanr

        from prep_data import inverse_target
        evs = np.concatenate([
            np.asarray(np.load(f"data/burn_evs_{r['job_id']}.npy"))
            .reshape(-1)[: r["i1"] - r["i0"]] for r in cand])
        meta = np.load(cand[0]["data"])
        tc_hw = inverse_target(np.clip(evs, -1, 1),
                               float(meta["y_min"]), float(meta["y_max"]))
        df = pd.read_csv("data/candidates_screen.csv")
        df["tc_hw_K"] = np.round(tc_hw, 2)
        df.to_csv("data/candidates_screen.csv", index=False)
        rho = spearmanr(df["tc_ideal_K"], df["tc_hw_K"]).statistic
        top = df[~df["out_of_range"]].nlargest(10, "tc_hw_K")
        print(f"\nCANDIDATE SCREEN stitched ({len(df)} candidates on hardware): "
              f"Spearman rank agreement hardware-vs-ideal rho={rho:.4f}")
        print(top[["formula", "gen", "tc_ideal_K", "tc_hw_K"]]
              .to_string(index=False))

    with open(REG, "w") as f:
        json.dump(recs, f, indent=2)
    print(f"\n{n_done}/{len(recs)} scored; registry updated")


if __name__ == "__main__":
    main()
