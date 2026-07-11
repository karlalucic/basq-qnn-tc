"""Data preparation for the BasQ QNN Tc-regression challenge.

Loads the UCI superconductivity dataset, reduces it to a QPU-compatible
number of features, and produces train/test splits that every model
(classical baseline, ideal QNN, noisy QNN, hardware QNN) must share.

Outputs data/prepared_<k>d.npz with scaled features in [-pi/2, pi/2]
and targets in [-0.8, 0.8] (log-transformed Tc), plus the scalers needed
to invert predictions back to Kelvin.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_CSV = "data/train.csv"
SEED = 42


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    # The dataset contains duplicated feature rows (same material measured
    # more than once). Keep one row per unique feature vector, averaging Tc.
    feat_cols = [c for c in df.columns if c != "critical_temp"]
    df = df.groupby(feat_cols, as_index=False)["critical_temp"].mean()
    return df


def prepare(n_features: int = 4, n_train: int = 800, n_test: int = 200,
            log_target: bool = True, seed: int = SEED, save: bool = True,
            reduction: str = "pca", n_val: int = 0,
            return_internals: bool = False):
    """Reduce to n_features and subsample a hackathon-sized split.

    reduction="pca": unsupervised (keeps feature variance).
    reduction="pls": supervised — PLS components are the directions most
    predictive of Tc, fitted on ALL training rows. The full dataset's
    information is absorbed classically; the QNN reads the distilled inputs.

    Full-data splits are kept for the classical baselines; the subsample is
    what the QNN trains on (statevector training on 21k rows is pointless
    and slow — the comparison only needs a common subset).

    n_val > 0 carves an extra validation subsample from the TRAINING side
    only (never from the test split), disjoint from the n_train rows. It is
    drawn from the same rng stream AFTER the n_train/n_test draws, so with
    n_val=0 the stream is untouched and X_train/X_test are byte-identical to
    the original files. Saved under X_val/y_val when present. Used for
    architecture selection so the test set is never touched during model
    selection (audit finding (a): test-set leakage).

    return_internals=True adds a non-saved "_internals" entry to the returned
    dict (fitted scalers/reducer, the project() callable, the subsample
    index arrays, and the raw train/test splits). fresh_holdout.py reuses
    these to project never-subsampled rows through the identical pipeline.
    """
    df = load_raw()
    X = df.drop(columns="critical_temp").values
    y = df["critical_temp"].values
    if log_target:
        y = np.log1p(y)  # Tc is heavily right-skewed; ln(1+Tc) evens it out

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    # Fit scaling + reducer on training data only (no leakage).
    std = StandardScaler().fit(X_train_full)
    if reduction == "pls":
        from sklearn.cross_decomposition import PLSRegression
        red = PLSRegression(n_components=n_features).fit(
            std.transform(X_train_full), y_train_full)
        explained = np.array([np.nan])  # PLS has no variance-ratio notion

        def project(A):
            return red.transform(std.transform(A))
    else:
        red = PCA(n_components=n_features, random_state=seed).fit(
            std.transform(X_train_full))
        explained = red.explained_variance_ratio_

        def project(A):
            return red.transform(std.transform(A))

    Z_train, Z_test = project(X_train_full), project(X_test_full)

    # Angle-scale features and bound targets inside the Z-observable range.
    x_scaler = MinMaxScaler((-np.pi / 2, np.pi / 2)).fit(Z_train)
    y_scaler = MinMaxScaler((-0.8, 0.8)).fit(y_train_full.reshape(-1, 1))

    Zs_train = x_scaler.transform(Z_train)
    Zs_test = x_scaler.transform(Z_test)
    ys_train = y_scaler.transform(y_train_full.reshape(-1, 1)).ravel()
    ys_test = y_scaler.transform(y_test_full.reshape(-1, 1)).ravel()

    rng = np.random.default_rng(seed)
    i_tr = rng.choice(len(Zs_train), size=n_train, replace=False)
    i_te = rng.choice(len(Zs_test), size=n_test, replace=False)

    out = dict(
        X_train=Zs_train[i_tr], y_train=ys_train[i_tr],
        X_test=Zs_test[i_te], y_test=ys_test[i_te],
        X_train_full=Zs_train, y_train_full=ys_train,
        X_test_full=Zs_test, y_test_full=ys_test,
        y_min=y_scaler.data_min_[0], y_max=y_scaler.data_max_[0],
        explained_var=explained,
        log_target=log_target,
    )

    # Optional validation subsample: extra rows from the TRAINING side, drawn
    # AFTER i_tr/i_te on the same stream so n_val=0 leaves everything above
    # bit-for-bit unchanged. Disjoint from the n_train rows by construction.
    i_val = None
    if n_val > 0:
        remaining = np.setdiff1d(np.arange(len(Zs_train)), i_tr)
        if n_val > len(remaining):
            raise ValueError(
                f"n_val={n_val} exceeds {len(remaining)} untrained rows")
        i_val = rng.choice(remaining, size=n_val, replace=False)
        out["X_val"] = Zs_train[i_val]
        out["y_val"] = ys_train[i_val]

    if save:
        suffix = "" if reduction == "pca" else f"_{reduction}"
        np.savez(f"data/prepared_{n_features}d{suffix}.npz", **out)

    if return_internals:
        # Never saved — only attached to the in-memory return value.
        out["_internals"] = dict(
            project=project, std=std, reducer=red,
            x_scaler=x_scaler, y_scaler=y_scaler,
            i_tr=i_tr, i_te=i_te, i_val=i_val,
            X_train_split=X_train_full, y_train_split=y_train_full,
            X_test_split=X_test_full, y_test_split=y_test_full,
        )
    return out


def _verify_backward_compatible(n_features: int = 4, reduction: str = "pca"):
    """n_val=0 must reproduce the committed prepared file exactly.

    Asserts np.allclose (and exact array equality) against every array in the
    existing npz, guarding audit-remediation edits from silently perturbing
    the shared train/test split.
    """
    suffix = "" if reduction == "pca" else f"_{reduction}"
    path = f"data/prepared_{n_features}d{suffix}.npz"
    old = dict(np.load(path))
    new = prepare(n_features=n_features, reduction=reduction, n_val=0,
                  save=False)
    assert set(old) == {k for k in new if k != "_internals"}, (
        f"key set changed vs {path}")
    all_exact = True
    for k in old:
        close = np.allclose(old[k], new[k], equal_nan=True)
        exact = np.array_equal(old[k], new[k], equal_nan=np.issubdtype(
            np.asarray(old[k]).dtype, np.floating))
        all_exact = all_exact and exact
        assert close, f"MISMATCH in {k}: regenerated array differs from {path}"
    print(f"[verify] {path}: n_val=0 reproduces all {len(old)} arrays "
          f"(np.allclose=True, exact-equal={all_exact})")
    return all_exact


def inverse_target(ys, y_min, y_max, log_target=True):
    """Map scaled predictions back to Kelvin."""
    y = (ys + 0.8) / 1.6 * (y_max - y_min) + y_min
    return np.expm1(y) if log_target else y


if __name__ == "__main__":
    # Guard the shared split first: regenerating with n_val=0 must be
    # byte-identical to the committed files (does not overwrite them).
    _verify_backward_compatible(4, "pca")
    _verify_backward_compatible(4, "pls")
    for k in (4, 5, 8):
        d = prepare(n_features=k)
        print(f"k={k}: PCA explains {d['explained_var'].sum():.1%} of variance "
              f"| train {d['X_train'].shape} test {d['X_test'].shape}")
