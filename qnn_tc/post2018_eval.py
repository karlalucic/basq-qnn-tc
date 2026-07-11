"""Out-of-time validation: post-2018 superconductors through the frozen pipeline.

The 17 literature-verified post-2018 materials (data/external/post2018_validation.csv)
are featurized with the exact UCI 81-feature scheme (featurize_formula.py),
pushed through the PCA-4 and PLS-4 pipelines refit identically to
prep_data.prepare (seed 42, fit on train side only — verified bit-identical
against the cached prepared_4d*.npz), and predicted with:

  * linear baseline           (baselines.py config, fit on the QNN's 800-row subsample)
  * random forest             (n_estimators=200, random_state=0 — baselines.py config)
  * champion QNN              (make_qnn(4, 2, 4) + trained_weights_q4_l2_pls.npy,
                               evaluated via EstimatorQNN.forward)

Outputs data/post2018_predictions.csv and prints per-material tables + metrics.
UTe2 is dropped (U, Z=92, is outside the UCI 86-element basis).
High-pressure-synthesis / strain-stabilized rows are flagged and metrics are
reported with and without them — composition-only features cannot know
pressure/strain, so those rows are the expected failure mode.
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from featurize_formula import feature_columns, featurize_many, recover_element_table
from prep_data import SEED, inverse_target, load_raw

VALIDATION_CSV = "data/external/post2018_validation.csv"
OUT_CSV = "data/post2018_predictions.csv"
CHAMPION_WEIGHTS = "data/trained_weights_q4_l2_pls.npy"

# Rows whose Tc depends on pressure/strain conditions invisible to
# composition-only features (see features_validation.md caveats).
FLAGGED = {
    "La3Ni2O7": "strain-stabilized film (bulk SC needs >14 GPa)",
    "Ba2CuO3.2": "metastable, synthesized at 18 GPa (SC measured at ambient)",
}


def refit_pipeline(reduction: str, n_features: int = 4):
    """Reproduce prep_data.prepare() exactly, but keep the fitted transforms.

    prep_data.prepare() only saves arrays, not the scaler/reducer objects,
    so we rerun the identical code path (same seed, same fit-on-train-only
    order) and return a project() closure for new external data.
    """
    df = load_raw()
    X = df.drop(columns="critical_temp").values
    y = np.log1p(df["critical_temp"].values)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    std = StandardScaler().fit(X_train_full)
    if reduction == "pls":
        red = PLSRegression(n_components=n_features).fit(
            std.transform(X_train_full), y_train_full)
    else:
        red = PCA(n_components=n_features, random_state=SEED).fit(
            std.transform(X_train_full))

    Z_train = red.transform(std.transform(X_train_full))
    x_scaler = MinMaxScaler((-np.pi / 2, np.pi / 2)).fit(Z_train)
    y_scaler = MinMaxScaler((-0.8, 0.8)).fit(y_train_full.reshape(-1, 1))

    def project(A):
        return x_scaler.transform(red.transform(std.transform(A)))

    return project, y_scaler


def verify_against_cache(project, reduction):
    """Assert the refit pipeline reproduces the cached prepared npz."""
    suffix = "" if reduction == "pca" else "_pls"
    d = dict(np.load(f"data/prepared_4d{suffix}.npz"))
    df = load_raw()
    X = df.drop(columns="critical_temp").values
    y = np.log1p(df["critical_temp"].values)
    X_train_full, X_test_full, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    assert np.allclose(project(X_train_full), d["X_train_full"], atol=1e-8), \
        f"{reduction}: refit pipeline does not match cached npz (train)"
    assert np.allclose(project(X_test_full), d["X_test_full"], atol=1e-8), \
        f"{reduction}: refit pipeline does not match cached npz (test)"
    print(f"  [{reduction}] refit pipeline == cached prepared_4d{suffix}.npz  OK")
    return d


def to_kelvin(pred_scaled, d):
    """Scaled model output -> Kelvin (same clipping convention as baselines)."""
    return inverse_target(np.clip(pred_scaled, -1, 1), d["y_min"], d["y_max"])


def qnn_predict(X_scaled, weights_path=CHAMPION_WEIGHTS):
    """Champion QNN forward pass. forward() handles parameter ordering —
    never bind a bare ndarray to a raw Qiskit PUB."""
    from qnn import make_qnn
    weights = np.load(weights_path)
    qnn, _ = make_qnn(4, 2, 4)
    assert qnn.num_weights == len(weights), \
        f"weight count mismatch: circuit {qnn.num_weights} vs file {len(weights)}"
    return np.asarray(qnn.forward(X_scaled, weights)).ravel()


def metrics(y_true, y_pred):
    return (np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred))


def main():
    val = pd.read_csv(VALIDATION_CSV)
    print(f"Loaded {len(val)} post-2018 materials")

    print("\nFeaturizing with the UCI 81-feature scheme...")
    E = recover_element_table(verbose=False)
    feats, skipped = featurize_many(val["formula"].tolist(), E)
    for f, reason in skipped:
        print(f"  DROPPED {f}: {reason}")
    val = val[val["formula"].isin(feats.index)].reset_index(drop=True)
    X81 = feats.loc[val["formula"], feature_columns()].values
    print(f"  {len(val)} materials featurized")

    out = val[["formula", "tc_kelvin", "year", "family"]].copy()
    out["flag"] = out["formula"].map(FLAGGED).fillna("")

    results = {}
    for reduction in ("pca", "pls"):
        print(f"\n=== {reduction.upper()}-4 pipeline ===")
        project, _ = refit_pipeline(reduction)
        d = verify_against_cache(project, reduction)

        X4 = project(X81)
        lo, hi = X4.min(), X4.max()
        print(f"  projected features span [{lo:.2f}, {hi:.2f}] "
              f"(training range [-1.57, 1.57]) — "
              f"{(np.abs(X4) > np.pi / 2).mean():.0%} of entries out of range")

        # Models fit on the exact arrays the QNN trained on (800-row subsample)
        lin = LinearRegression().fit(d["X_train"], d["y_train"])
        rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(
            d["X_train"], d["y_train"])

        out[f"pred_linear_{reduction}"] = to_kelvin(lin.predict(X4), d)
        out[f"pred_rf_{reduction}"] = to_kelvin(rf.predict(X4), d)

        if reduction == "pls":
            print("  running champion QNN (q=4, L=2, PLS-4 weights)...")
            out["pred_qnn_pls"] = to_kelvin(qnn_predict(X4), d)
        results[reduction] = d

    # ---- report ----
    pred_cols = [c for c in out.columns if c.startswith("pred_")]
    pd.set_option("display.width", 200)
    print("\nPer-material predictions (Kelvin):")
    show = out[["formula", "family", "tc_kelvin"] + pred_cols].copy()
    show[pred_cols] = show[pred_cols].round(1)
    print(show.to_string(index=False))

    def report(mask, label):
        print(f"\nMetrics {label} (n={mask.sum()}):")
        for c in pred_cols:
            rm, ma = metrics(out.loc[mask, "tc_kelvin"], out.loc[mask, c])
            print(f"  {c:22s} RMSE {rm:6.2f} K   MAE {ma:6.2f} K")

    all_rows = np.ones(len(out), bool)
    clean = out["flag"].values == ""
    report(all_rows, "ALL rows")
    report(clean, "excluding flagged high-pressure/strain rows")
    report(~clean, "flagged rows only")

    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")
    return out


if __name__ == "__main__":
    main()
