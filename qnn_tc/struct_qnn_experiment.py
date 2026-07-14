"""Does structure enrichment survive the quantum model?

dsc_ablation.py measured (linear on PLS-4, grouped split): composition-only
13.97 K vs composition+structure 12.37 K. This trains the champion QNN config
(4 qubits x 2 layers, sweep budget) on the SAME two input pipelines and the
SAME grouped split, so the deltas are directly comparable.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dsc_ablation import build_joined_table, STRUCT_COLS, kelvin_metrics
from featurize_formula import feature_columns
from qnn import train

SEED = 42
N_SWEEP = 300          # sweep-budget training rows (protocol standard)
MAXITER = 60


def make_d(X_raw, y_log, tr_idx, te_idx, sweep_rows=N_SWEEP, seed=SEED):
    """Build a qnn.train-compatible dict from raw features + log1p targets,
    fitting impute/scale/PLS/minmax on the training side only."""
    imp = SimpleImputer(strategy="median").fit(X_raw[tr_idx])
    std = StandardScaler().fit(imp.transform(X_raw[tr_idx]))
    Ztr = std.transform(imp.transform(X_raw[tr_idx]))
    Zte = std.transform(imp.transform(X_raw[te_idx]))
    pls = PLSRegression(n_components=4).fit(Ztr, y_log[tr_idx])
    Str, Ste = pls.transform(Ztr), pls.transform(Zte)

    xs = MinMaxScaler((-np.pi / 2, np.pi / 2)).fit(Str)
    ys = MinMaxScaler((-0.8, 0.8)).fit(y_log[tr_idx].reshape(-1, 1))

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(tr_idx), size=sweep_rows, replace=False)
    return dict(
        X_train=xs.transform(Str)[sub],
        y_train=ys.transform(y_log[tr_idx].reshape(-1, 1)).ravel()[sub],
        X_test=xs.transform(Ste),
        y_test=ys.transform(y_log[te_idx].reshape(-1, 1)).ravel(),
        y_min=ys.data_min_[0], y_max=ys.data_max_[0], log_target=True,
    ), (Str[sub], Ste, y_log[tr_idx][sub], y_log[te_idx])


def main():
    joined = build_joined_table()
    y = np.log1p(joined["critical_temp"].values)
    X_comp = joined[feature_columns()].values.astype(float)
    X_both = np.hstack([X_comp, joined[STRUCT_COLS].values.astype(float)])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(joined, groups=joined["chem_system"]))
    print(f"grouped split: train {len(tr_idx)} / test {len(te_idx)} "
          f"(sweep subsample {N_SWEEP} rows, maxiter {MAXITER})\n")

    results = {}
    for label, X_raw in (("composition-only", X_comp),
                         ("composition+structure", X_both)):
        d, (Str, Ste, ytr, yte) = make_d(X_raw, y, tr_idx, te_idx)

        # matched classical reference on the identical 300 rows
        lin = LinearRegression().fit(Str, ytr)
        rm_lin, _ = kelvin_metrics(yte, lin.predict(Ste))

        print(f"=== QNN 4q x 2L on PLS-4 <- {label} ===")
        reg, stats = train(d, n_qubits=4, n_layers=2, maxiter=MAXITER, seed=7)
        results[label] = (stats["rmse_test"], rm_lin)
        np.save(f"data/trained_weights_struct_{label.split('-')[0][:4]}.npy",
                reg.weights)

    print("\nSummary (grouped split, Kelvin):")
    print(f"{'inputs':24s} {'QNN 4qx2L':>10s} {'linear-300':>11s}")
    for label, (rm_q, rm_l) in results.items():
        print(f"{label:24s} {rm_q:10.2f} {rm_l:11.2f}")
    dq = results["composition-only"][0] - results["composition+structure"][0]
    print(f"\nstructure gain for the QNN: {dq:+.2f} K "
          f"(classical full-train reference was +1.60 K)")


if __name__ == "__main__":
    main()
