"""Fresh, never-touched holdout for final reporting (audit remediation).

After all model selection is done, quote error on a set of rows that were
NEVER seen by any stage of the workflow: not the 800-row train subsample,
not the 200-row test subsample, and not the full 3034-row test split. Such
rows exist inside the TRAINING split — the ~11.3k rows that were never
subsampled into X_train. We draw 300 of them and project them through the
identical seed-42 PCA and PLS pipelines (reusing prepare()'s internals), so
the holdout materials are the same in both files, only the projection differs.

Saves data/fresh_holdout_pca.npz and data/fresh_holdout_pls.npz with X_hold/
y_hold in the same scaled space as X_test/y_test, plus y_min/y_max/log_target
so baselines.rmse_kelvin / prep_data.inverse_target map predictions to Kelvin.
"""

import numpy as np

from prep_data import prepare

SEED = 42
N_HOLD = 300


def build_holdout(reduction: str, n_hold: int = N_HOLD, seed: int = SEED):
    d = prepare(n_features=4, reduction=reduction, n_val=0, save=False,
                return_internals=True)
    it = d["_internals"]
    project, x_scaler, y_scaler = it["project"], it["x_scaler"], it["y_scaler"]
    X_split, y_split = it["X_train_split"], it["y_train_split"]  # raw / log-Tc
    i_tr = it["i_tr"]

    # Rows in the training split that were never subsampled into X_train.
    never = np.setdiff1d(np.arange(len(X_split)), i_tr)
    rng = np.random.default_rng(seed)
    i_hold = rng.choice(never, size=n_hold, replace=False)

    # Disjointness guarantees (train subsample lives in this same index space;
    # the test subsample and full test split are a different partition entirely).
    assert set(i_hold.tolist()).isdisjoint(set(i_tr.tolist())), \
        "holdout overlaps the 800-row train subsample"

    X_hold = x_scaler.transform(project(X_split[i_hold]))
    y_hold = y_scaler.transform(y_split[i_hold].reshape(-1, 1)).ravel()

    out = dict(
        X_hold=X_hold, y_hold=y_hold,
        y_min=y_scaler.data_min_[0], y_max=y_scaler.data_max_[0],
        log_target=d["log_target"],
        hold_index=i_hold,  # index into the deduped training split
    )
    np.savez(f"data/fresh_holdout_{reduction}.npz", **out)
    print(f"[{reduction}] training split={len(X_split)}  train-subsampled={len(i_tr)}"
          f"  never-touched pool={len(never)}  ->  holdout {X_hold.shape}")
    return out


if __name__ == "__main__":
    pca = build_holdout("pca")
    pls = build_holdout("pls")
    # Same materials, different projection: identical holdout indices.
    same_rows = np.array_equal(pca["hold_index"], pls["hold_index"])
    print(f"PCA/PLS holdout draw identical rows: {same_rows}")
    assert same_rows, "holdout materials differ across pipelines"
    print("saved data/fresh_holdout_pca.npz and data/fresh_holdout_pls.npz")
