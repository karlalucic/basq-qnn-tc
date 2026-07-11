"""Reproduce the UCI/Hamidieh (2018) 81-feature scheme from a chemical formula.

The 81 features = number_of_elements + 10 statistics x 8 elemental properties
(atomic_mass, fie, atomic_radius, Density, ElectronAffinity, FusionHeat,
ThermalConductivity, Valence).

Element property values are NOT taken from an external table. They are
recovered from the UCI data itself: for every row, train.csv's
    mean_<prop>     = (1/k) * sum_{elements present} t_e
    wtd_mean_<prop> = sum_e p_e * t_e          (p_e = normalized fraction)
are LINEAR in the unknown element values t_e, and unique_m.csv provides the
compositions. Stacking 2 x 21,263 equations and solving least squares
recovers Hamidieh's exact table (residuals ~1e-12; recovered values match
known physical constants, e.g. Cu thermal conductivity = 400 W/mK).

Statistic definitions (verified to machine precision against train.csv):
    mean         = arithmetic mean of t over elements present
    wtd_mean     = sum p_i t_i
    gmean        = exp(mean(ln t))
    wtd_gmean    = exp(sum p_i ln t_i)
    entropy      = -sum w_i ln w_i,  w_i = t_i / sum t_j
    wtd_entropy  = -sum a_i ln a_i,  a_i = p_i t_i / sum p_j t_j
    range        = max t - min t
    wtd_range    = max(p_i t_i) - min(p_i t_i)
    std          = population std of t (ddof=0)
    wtd_std      = sqrt(sum p_i (t_i - wtd_mean)^2)

Run as a script to (re)build the cached element table and validate the
featurizer against 20 random unique_m.csv formulas.
"""

import re

import numpy as np
import pandas as pd

TRAIN_CSV = "data/train.csv"
UNIQUE_M_CSV = "data/unique_m.csv"
ELEMENT_TABLE_CSV = "data/element_properties_uci.csv"

PROPS = ["atomic_mass", "fie", "atomic_radius", "Density", "ElectronAffinity",
         "FusionHeat", "ThermalConductivity", "Valence"]
STATS = ["mean", "wtd_mean", "gmean", "wtd_gmean", "entropy", "wtd_entropy",
         "range", "wtd_range", "std", "wtd_std"]

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def feature_columns():
    """The 81 feature names in exact train.csv order."""
    cols = ["number_of_elements"]
    for p in PROPS:
        cols += [f"{s}_{p}" for s in STATS]
    return cols


def parse_formula(formula: str) -> dict:
    """'Ba0.2La1.8Cu1O4' -> {'Ba': 0.2, 'La': 1.8, 'Cu': 1.0, 'O': 4.0}.

    Repeated element symbols accumulate. Raises ValueError on unparseable
    residue so bad strings fail loudly instead of silently dropping atoms.
    """
    counts = {}
    consumed = 0
    for m in _FORMULA_RE.finditer(formula):
        if m.group(0) == "":
            continue
        consumed += len(m.group(0))
        el = m.group(1)
        amt = float(m.group(2)) if m.group(2) else 1.0
        counts[el] = counts.get(el, 0.0) + amt
    if consumed != len(formula):
        raise ValueError(f"could not fully parse formula {formula!r}")
    if not counts:
        raise ValueError(f"empty formula {formula!r}")
    return counts


def recover_element_table(train_csv=TRAIN_CSV, unique_m_csv=UNIQUE_M_CSV,
                          cache=ELEMENT_TABLE_CSV, verbose=True):
    """Solve the linear system for the 8 element-property vectors.

    Returns a DataFrame indexed by element symbol (the 86 unique_m columns,
    H..Rn) with one column per property. Elements that never appear in the
    dataset are unrecoverable and stored as NaN.
    """
    import os
    if cache and os.path.exists(cache):
        return pd.read_csv(cache, index_col=0)

    tr = pd.read_csv(train_csv)
    um = pd.read_csv(unique_m_csv)
    elem_cols = [c for c in um.columns if c not in ("critical_temp", "material")]

    F = um[elem_cols].values.astype(float)      # raw subscripts
    present = F > 0
    k = present.sum(1)
    P = F / F.sum(1, keepdims=True)             # normalized fractions
    I = present / np.maximum(k, 1)[:, None]     # rows for 'mean' equations

    A = np.vstack([I, P])                       # (2N x 86)
    table = {}
    for prop in PROPS:
        b = np.concatenate([tr[f"mean_{prop}"].values,
                            tr[f"wtd_mean_{prop}"].values])
        sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        resid = np.abs(A @ sol - b)
        if verbose:
            print(f"  recover {prop:20s} rank={rank} "
                  f"max|resid|={resid.max():.2e}")
        table[prop] = sol

    E = pd.DataFrame(table, index=elem_cols)
    never = present.sum(0) == 0
    E.loc[never, :] = np.nan                    # unrecoverable, be explicit
    if verbose:
        print(f"  unrecoverable (never appear): "
              f"{[e for e, n in zip(elem_cols, never) if n]}")
    if cache:
        E.to_csv(cache)
    return E


def _stats_vector(t: np.ndarray, p: np.ndarray) -> list:
    """The 10 Hamidieh statistics for one property over one material."""
    wtd_mean = float((p * t).sum())
    pt = p * t
    with np.errstate(divide="ignore", invalid="ignore"):
        log_t = np.log(t)
        gmean = float(np.exp(log_t.mean()))
        wtd_gmean = float(np.exp((p * log_t).sum()))
        w = t / t.sum()
        entropy = float(-(w * np.log(w)).sum())
        a = pt / pt.sum()
        wtd_entropy = float(-(a * np.log(a)).sum())
    return [
        float(t.mean()), wtd_mean, gmean, wtd_gmean, entropy, wtd_entropy,
        float(t.max() - t.min()), float(pt.max() - pt.min()),
        float(t.std(ddof=0)),
        float(np.sqrt((p * (t - wtd_mean) ** 2).sum())),
    ]


def featurize(formula, element_table: pd.DataFrame) -> pd.Series:
    """Formula string (or {element: amount} dict) -> 81-feature Series.

    Raises ValueError if the formula contains an element outside the UCI
    basis (or one whose properties are unrecoverable), e.g. U in UTe2.
    """
    counts = parse_formula(formula) if isinstance(formula, str) else dict(formula)
    els = list(counts)
    missing = [e for e in els if e not in element_table.index
               or element_table.loc[e].isna().any()]
    if missing:
        raise ValueError(f"formula {formula!r}: element(s) {missing} outside "
                         f"the recoverable UCI basis")
    amounts = np.array([counts[e] for e in els], float)
    p = amounts / amounts.sum()

    vals = [float(len(els))]
    for prop in PROPS:
        t = element_table.loc[els, prop].values.astype(float)
        vals += _stats_vector(t, p)
    return pd.Series(vals, index=feature_columns())


def featurize_many(formulas, element_table: pd.DataFrame) -> pd.DataFrame:
    """Featurize a list of formulas; rows with unsupported elements are
    returned in the second value (list of (formula, reason))."""
    rows, skipped = {}, []
    for f in formulas:
        try:
            rows[f] = featurize(f, element_table)
        except ValueError as e:
            skipped.append((f, str(e)))
    return pd.DataFrame(rows).T, skipped


def validate(n=20, seed=42, element_table=None, verbose=True):
    """Featurize n random unique_m.csv formulas and compare with train.csv.

    Returns a DataFrame with per-feature median/max relative error.
    """
    tr = pd.read_csv(TRAIN_CSV)
    um = pd.read_csv(UNIQUE_M_CSV)
    E = element_table if element_table is not None else recover_element_table()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(um), size=n, replace=False)
    cols = feature_columns()

    rel_errs = []
    used = []
    for i in idx:
        formula = um.loc[i, "material"]
        try:
            f = featurize(formula, E)
        except ValueError as e:
            if verbose:
                print(f"  skip row {i} ({formula}): {e}")
            continue
        truth = tr.loc[i, cols].values.astype(float)
        err = np.abs(f.values - truth) / (np.abs(truth) + 1e-12)
        rel_errs.append(err)
        used.append((i, formula, np.nanmax(err)))

    R = np.array(rel_errs)
    out = pd.DataFrame({"median_rel_err": np.median(R, 0),
                        "max_rel_err": R.max(0)}, index=cols)
    if verbose:
        print(f"\nValidation on {len(used)} formulas (seed {seed}):")
        for i, formula, mx in used:
            print(f"  row {i:6d} {formula:30s} worst feature rel err {mx:.2e}")
        print(f"\nOverall: median rel err {np.median(R):.2e}, "
              f"max {R.max():.2e}")
        by_stat = {}
        for j, c in enumerate(cols):
            stat = "number_of_elements" if c == cols[0] else \
                next(s for s in sorted(STATS, key=len, reverse=True)
                     if c.startswith(s + "_"))
            by_stat.setdefault(stat, []).append(R[:, j])
        print("\nPer-statistic median (max) relative error:")
        for stat, chunks in by_stat.items():
            v = np.concatenate(chunks)
            print(f"  {stat:20s} {np.median(v):.2e}  ({v.max():.2e})")
    return out


if __name__ == "__main__":
    print("Recovering element property table from UCI data...")
    E = recover_element_table()
    known = E.loc[["H", "O", "Cu", "Ba", "Nb", "Fe", "La"]].round(3)
    print("\nSanity check, recovered values (should match periodictable.com):")
    print(known)
    validate(n=20, seed=42, element_table=E)
