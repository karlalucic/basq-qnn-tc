"""3DSC structure-feature ablation (classical only).

Question: how much predictive information does crystal structure / DFT add
that composition-statistics features lack? And would a structure-enriched
4-dim PLS bottleneck feed the QNN better inputs than composition alone?

Method:
  1. Join 3DSC_MP.csv to UCI rows by exact normalized composition
     (the convention measured in external_datasets.md: fractions rounded
     to 3 dp, elements sorted). One row per unique composition:
     UCI 81 features are constant per composition; Tc is averaged
     (protocol invariant); for 3DSC the best structure match is kept
     (min totreldiff, tiebreak min e_above_hull_2).
  2. Fixed split, seed 42, GROUPED by chemical system (sorted element set)
     so doping series (e.g. all YBCO variants) never straddle the split.
  3. HistGradientBoostingRegressor(random_state=0) on
       (i)  UCI-81 composition features only
       (ii) UCI-81 + 3DSC structural/DFT columns
     Same rows, same split, y = log1p(Tc), RMSE/MAE reported in Kelvin.
  4. PLS-4 bottleneck: StandardScaler + PLSRegression(4) fit on train,
     LinearRegression on the 4 scores — composition-only vs
     structure-enriched, mirroring the QNN's 4-feature input pipeline.

Leakage control: no column derived from measured Tc enters the features
(drop tc, sc_class and all match-quality/provenance metadata).
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from featurize_formula import feature_columns, parse_formula

DSC_CSV = "data/external/3DSC_MP.csv"
TRAIN_CSV = "data/train.csv"
UNIQUE_M_CSV = "data/unique_m.csv"
SEED = 42

# Structural / DFT columns taken from 3DSC (physics only — no Tc-derived,
# no match-quality metadata, no free-text/ID columns).
STRUCT_NUMERIC = [
    "lata_2", "latb_2", "latc_2", "cell_volume_2", "density_2", "nsites_2",
    "band_gap_2", "efermi_2", "energy_per_atom_2",
    "formation_energy_per_atom_2", "e_above_hull_2",
    "total_magnetization_2", "total_magnetization_normalized_vol_2",
]
STRUCT_ONEHOT = [
    "cubic", "hexagonal", "monoclinic", "orthorhombic", "tetragonal",
    "triclinic", "trigonal",
    "primitive", "base-centered", "body-centered", "face-centered",
]
STRUCT_BOOL = ["is_magnetic_2"]
STRUCT_COLS = STRUCT_NUMERIC + STRUCT_ONEHOT + STRUCT_BOOL


def comp_key_from_fractions(els, fracs):
    """Normalized composition key: sorted (element, fraction rounded 3dp)."""
    return tuple(sorted((e, round(float(f), 3)) for e, f in zip(els, fracs)))


def build_joined_table():
    """UCI (81 features + mean Tc) inner-joined to best 3DSC structure match,
    one row per unique composition."""
    tr = pd.read_csv(TRAIN_CSV)
    um = pd.read_csv(UNIQUE_M_CSV)
    elem_cols = [c for c in um.columns if c not in ("critical_temp", "material")]

    # --- UCI side: composition key from the fraction columns -------------
    F = um[elem_cols].values.astype(float)
    P = F / F.sum(1, keepdims=True)
    keys = []
    for i in range(len(um)):
        idx = np.where(F[i] > 0)[0]
        keys.append(comp_key_from_fractions(
            [elem_cols[j] for j in idx], P[i, idx]))
    uci = tr.copy()
    uci["comp_key"] = keys
    n_rows_before = len(uci)
    # One row per composition: features are deterministic per composition,
    # Tc duplicates are averaged (same protocol invariant as prep_data).
    uci = uci.groupby("comp_key", as_index=False).agg(
        {**{c: "first" for c in feature_columns()}, "critical_temp": "mean"})
    print(f"UCI: {n_rows_before} rows -> {len(uci)} unique compositions")

    # --- 3DSC side: parse formula_sc, keep best structure match ----------
    dsc = pd.read_csv(DSC_CSV, comment="#")
    dsc_keys = []
    for f in dsc["formula_sc"]:
        try:
            c = parse_formula(str(f))
            tot = sum(c.values())
            dsc_keys.append(comp_key_from_fractions(
                list(c), [v / tot for v in c.values()]))
        except ValueError:
            dsc_keys.append(None)
    dsc["comp_key"] = dsc_keys
    dsc = dsc[dsc["comp_key"].notna()]
    dsc = dsc.sort_values(["totreldiff", "e_above_hull_2"]).groupby(
        "comp_key", as_index=False).first()
    print(f"3DSC: {len(dsc)} unique compositions (best structure match kept)")

    joined = uci.merge(
        dsc[["comp_key", "tc"] + STRUCT_COLS], on="comp_key", how="inner")
    joined["is_magnetic_2"] = joined["is_magnetic_2"].astype(float)
    joined["chem_system"] = ["-".join(sorted(e for e, _ in k))
                             for k in joined["comp_key"]]
    print(f"Joined: {len(joined)} compositions, "
          f"{joined['chem_system'].nunique()} chemical systems")
    # Tc label sanity between the two sources on the matched rows
    r = np.corrcoef(joined["critical_temp"], joined["tc"])[0, 1]
    print(f"UCI-Tc vs 3DSC-Tc on matches: Pearson r = {r:.3f} "
          f"(median |dTc| = {np.median(np.abs(joined['critical_temp'] - joined['tc'])):.2f} K)")
    return joined


def kelvin_metrics(y_true_log, y_pred_log):
    yt, yp = np.expm1(y_true_log), np.expm1(np.clip(y_pred_log, 0, None))
    return (np.sqrt(mean_squared_error(yt, yp)), mean_absolute_error(yt, yp))


def run_split(joined, tr_idx, te_idx, label):
    """HistGBM ablation + PLS-4 bottleneck comparison on one split."""
    y = np.log1p(joined["critical_temp"].values)
    X_comp = joined[feature_columns()].values.astype(float)
    X_both = np.hstack([X_comp, joined[STRUCT_COLS].values.astype(float)])

    print(f"\n--- split: {label} | train {len(tr_idx)} / test {len(te_idx)} ---")
    results = {}
    for name, X in (("UCI-81 (composition only)", X_comp),
                    ("UCI-81 + 3DSC structure/DFT", X_both)):
        m = HistGradientBoostingRegressor(random_state=0)
        m.fit(X[tr_idx], y[tr_idx])
        rm, ma = kelvin_metrics(y[te_idx], m.predict(X[te_idx]))
        results[name] = (rm, ma)
        print(f"  HistGBM  {name:32s} RMSE {rm:6.2f} K   MAE {ma:6.2f} K")

    # ---- PLS-4 bottleneck: would structure help the 4-feature QNN input? --
    for name, X in (("PLS-4 <- composition only", X_comp),
                    ("PLS-4 <- composition + structure", X_both)):
        imp = SimpleImputer(strategy="median").fit(X[tr_idx])
        std = StandardScaler().fit(imp.transform(X[tr_idx]))
        Xtr = std.transform(imp.transform(X[tr_idx]))
        Xte = std.transform(imp.transform(X[te_idx]))
        pls = PLSRegression(n_components=4).fit(Xtr, y[tr_idx])
        lin = LinearRegression().fit(pls.transform(Xtr), y[tr_idx])
        rm, ma = kelvin_metrics(y[te_idx], lin.predict(pls.transform(Xte)))
        results[name] = (rm, ma)
        print(f"  Linear   {name:32s} RMSE {rm:6.2f} K   MAE {ma:6.2f} K")
    return results


def main():
    joined = build_joined_table()

    # Primary split: grouped by chemical system (no doping-series leakage).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(joined, groups=joined["chem_system"]))
    res_grouped = run_split(joined, tr_idx, te_idx,
                            "grouped by chemical system, seed 42")

    # Secondary sanity split: plain random (matches the main pipeline style).
    tr_idx, te_idx = train_test_split(np.arange(len(joined)), test_size=0.2,
                                      random_state=SEED)
    res_random = run_split(joined, tr_idx, te_idx, "random, seed 42")

    print("\nDelta (composition -> +structure), RMSE Kelvin:")
    for res, label in ((res_grouped, "grouped"), (res_random, "random")):
        d_gbm = res["UCI-81 (composition only)"][0] - \
            res["UCI-81 + 3DSC structure/DFT"][0]
        d_pls = res["PLS-4 <- composition only"][0] - \
            res["PLS-4 <- composition + structure"][0]
        print(f"  [{label:7s}] HistGBM {d_gbm:+.2f} K | PLS-4 linear {d_pls:+.2f} K")
    return res_grouped, res_random


if __name__ == "__main__":
    main()
