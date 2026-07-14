"""Classical baselines on the same prepared data the QNN sees,
plus a full-featured reference model to show the honest ceiling.

The challenge scores 'comparison against a classical baseline with a
similar number of trainable parameters' — that is the small MLP.
The full-data gradient-boosting run is context: what classical ML does
with all 81 features and all rows.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor

from prep_data import inverse_target


def rmse_kelvin(y_true_s, y_pred_s, d):
    yt = inverse_target(y_true_s, d["y_min"], d["y_max"])
    yp = inverse_target(np.clip(y_pred_s, -1, 1), d["y_min"], d["y_max"])
    return (np.sqrt(mean_squared_error(yt, yp)),
            mean_absolute_error(yt, yp))


def run_baselines(d, param_budget=30):
    """Baselines on the QNN's own (PCA-reduced, subsampled) data."""
    Xtr, ytr = d["X_train"], d["y_train"]
    Xte, yte = d["X_test"], d["y_test"]
    k = Xtr.shape[1]

    # MLP sized to roughly match the QNN's trainable parameter count:
    # hidden width h gives (k+2)h + 1 params.
    h = max(1, round((param_budget - 1) / (k + 2)))
    models = {
        "linear": LinearRegression(),
        f"mlp[{h}] (~{(k + 2) * h + 1} params)": MLPRegressor(
            hidden_layer_sizes=(h,), max_iter=4000, random_state=0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=0),
    }
    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        rm, ma = rmse_kelvin(yte, m.predict(Xte), d)
        results[name] = (rm, ma)
        print(f"{name:35s} RMSE {rm:6.2f} K   MAE {ma:6.2f} K")
    return results


def run_reference_ceiling(d):
    """Strong classical model on ALL reduced-space training rows."""
    m = HistGradientBoostingRegressor(random_state=0)
    m.fit(d["X_train_full"], d["y_train_full"])
    rm, ma = rmse_kelvin(d["y_test_full"], m.predict(d["X_test_full"]), d)
    print(f"{'hist_gbm (full rows, reduced inputs)':35s} RMSE {rm:6.2f} K   MAE {ma:6.2f} K")
    return rm, ma


if __name__ == "__main__":
    for k in (4, 8):
        print(f"\n=== {k} PCA features ===")
        d = dict(np.load(f"data/prepared_{k}d.npz"))
        run_baselines(d)
        run_reference_ceiling(d)
