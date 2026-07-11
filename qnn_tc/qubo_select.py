"""Quantum-annealing-inspired QUBO feature selection for the Tc QNN.

Implements the mutual-information QUBO of Ibarra-Hoyos et al.
(npj Comput. Mater. 2026): pick a k-subset of the 81 raw features that
maximises relevance to the target while minimising pairwise redundancy,

    min_{w in {0,1}^81}  -sum_i w_i I(X_i; y)
                         + lambda * sum_{i<j} w_i w_j I(X_i; X_j),
    subject to sum_i w_i = k = 4.

The QUBO is solved with CLASSICAL simulated annealing using
cardinality-preserving swap moves (the identical QUBO could run on a
D-Wave annealer; no quantum hardware is used here, and IBM gate-model
hardware does not perform annealing). Because C(81,4) = 1,663,740 is
enumerable, the SA result is additionally certified against the exact
global optimum by brute force.

Protocol matches prep_data.py exactly: dedupe by feature columns
(averaging Tc), y = log1p(Tc), train_test_split(test_size=0.2,
random_state=42); all fitting (scalers, MI, selection) on the training
split only; the 800/200 subsample uses np.random.default_rng(42).choice
in the same order as prep_data.prepare, so test rows align with
prepared_4d.npz / prepared_4d_pls.npz.

Outputs
  data/qmi_matrix.npz        cached MI relevance vector + 81x81 redundancy
  data/prepared_4d_qmi.npz   QNN-ready data, same keys as prepared_4d.npz
"""

import os
from itertools import combinations

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from prep_data import SEED, load_raw  # noqa: E402

MI_CACHE = "data/qmi_matrix.npz"
OUT_NPZ = "data/prepared_4d_qmi.npz"
K = 4
LAMBDAS = (0.25, 0.5, 1.0, 2.0)
N_RESTARTS = 20


# --------------------------------------------------------------------------
# Data (identical split to prep_data.prepare)
# --------------------------------------------------------------------------
def get_split():
    df = load_raw()
    feat_names = [c for c in df.columns if c != "critical_temp"]
    X = df.drop(columns="critical_temp").values
    y = np.log1p(df["critical_temp"].values)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    return X_tr, X_te, y_tr, y_te, feat_names


# --------------------------------------------------------------------------
# Mutual information: relevance (sklearn kNN) + redundancy (binned, fast)
# --------------------------------------------------------------------------
def binned_mi_matrix(X, n_bins=16):
    """Pairwise I(X_i; X_j) via equal-frequency binning (plug-in estimate).

    3240 pairs on ~17k rows in seconds; the plug-in estimator is biased
    upward but only the relative ordering matters inside the QUBO.
    """
    n, p = X.shape
    B = np.empty((n, p), dtype=np.int64)
    n_levels = np.empty(p, dtype=np.int64)
    for j in range(p):
        qs = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1))
        edges = np.unique(qs)[1:-1]  # interior edges only
        B[:, j] = np.searchsorted(edges, X[:, j], side="right")
        n_levels[j] = B[:, j].max() + 1

    def entropy(counts):
        pr = counts[counts > 0] / n
        return -np.sum(pr * np.log(pr))

    H = np.array([entropy(np.bincount(B[:, j], minlength=n_levels[j]))
                  for j in range(p)])
    M = np.zeros((p, p))
    for i in range(p):
        bi, ni = B[:, i], n_levels[i]
        for j in range(i + 1, p):
            joint = np.bincount(bi * n_levels[j] + B[:, j],
                                minlength=ni * n_levels[j])
            mi = max(0.0, H[i] + H[j] - entropy(joint))
            M[i, j] = M[j, i] = mi
    return M


def get_mi(X_tr_std, y_tr):
    if os.path.exists(MI_CACHE):
        c = np.load(MI_CACHE)
        print(f"[mi] loaded cache {MI_CACHE}", flush=True)
        return c["relevance"], c["redundancy"]
    print("[mi] computing relevance I(X_i; y) via mutual_info_regression "
          "(kNN, random_state=42) ...", flush=True)
    rel = mutual_info_regression(X_tr_std, y_tr, random_state=SEED)
    print("[mi] computing 81x81 redundancy I(X_i; X_j) via binned "
          "estimator ...", flush=True)
    red = binned_mi_matrix(X_tr_std, n_bins=16)
    np.savez(MI_CACHE, relevance=rel, redundancy=red)
    print(f"[mi] cached to {MI_CACHE}", flush=True)
    return rel, red


# --------------------------------------------------------------------------
# QUBO objective + simulated annealing with cardinality-preserving swaps
# --------------------------------------------------------------------------
def objective(idx, rel, red, lam):
    idx = np.asarray(idx)
    pair = red[np.ix_(idx, idx)].sum() / 2.0  # diagonal of red is 0
    return -rel[idx].sum() + lam * pair


def sa_once(rel, red, lam, rng, k=K, t_levels=150, iters=60, alpha=0.95):
    n = len(rel)
    S = list(rng.choice(n, size=k, replace=False))
    f = objective(S, rel, red, lam)
    best_S, best_f = sorted(S), f

    # Calibrate T0 from the spread of random swap deltas.
    deltas = []
    for _ in range(80):
        pos = rng.integers(k)
        a = rng.integers(n)
        while a in S:
            a = rng.integers(n)
        T = S.copy()
        T[pos] = a
        deltas.append(abs(objective(T, rel, red, lam) - f))
    T0 = max(float(np.mean(deltas)), 1e-9)

    temp = T0
    for _ in range(t_levels):
        for _ in range(iters):
            pos = rng.integers(k)
            o = S[pos]
            a = rng.integers(n)
            while a in S:
                a = rng.integers(n)
            rest = [x for x in S if x != o]
            delta = (rel[o] - rel[a]
                     + lam * (red[a, rest].sum() - red[o, rest].sum()))
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                S[pos] = a
                f += delta
                if f < best_f - 1e-12:
                    best_f, best_S = f, sorted(S)
        temp *= alpha
    # re-evaluate exactly (kill accumulated float drift)
    return tuple(best_S), objective(best_S, rel, red, lam)


def sa_select(rel, red, lam, restarts=N_RESTARTS, seed=SEED):
    master = np.random.default_rng(seed)
    hits = {}
    best_S, best_f = None, np.inf
    for r in range(restarts):
        S, f = sa_once(rel, red, lam, np.random.default_rng(master.integers(2**32)))
        hits[S] = hits.get(S, 0) + 1
        if f < best_f:
            best_S, best_f = S, f
    stability = hits[best_S] / restarts
    return best_S, best_f, stability, hits


def brute_force_terms(n, k=K):
    combos = np.array(list(combinations(range(n), k)), dtype=np.int64)
    return combos


def brute_force(rel, red, lam, combos):
    rel_sum = rel[combos].sum(axis=1)
    pair_sum = np.zeros(len(combos))
    for a, b in combinations(range(combos.shape[1]), 2):
        pair_sum += red[combos[:, a], combos[:, b]]
    obj = -rel_sum + lam * pair_sum
    i = int(np.argmin(obj))
    return tuple(combos[i]), float(obj[i])


# --------------------------------------------------------------------------
# Lambda choice on TRAIN ONLY: 5-fold CV linear-regression RMSE in Kelvin
# --------------------------------------------------------------------------
def cv_rmse_kelvin(X_tr, y_tr_log, idx, seed=SEED):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.empty_like(y_tr_log)
    for tr, va in kf.split(X_tr):
        m = LinearRegression().fit(X_tr[tr][:, list(idx)], y_tr_log[tr])
        pred[va] = m.predict(X_tr[va][:, list(idx)])
    return float(np.sqrt(mean_squared_error(np.expm1(y_tr_log),
                                            np.expm1(pred))))


# --------------------------------------------------------------------------
# Build prepared_4d_qmi.npz — replicates prep_data.prepare verbatim,
# with column selection replacing the PCA/PLS projection.
# --------------------------------------------------------------------------
def prepare_qmi(selected_idx, n_train=800, n_test=200, seed=SEED):
    df = load_raw()
    X = df.drop(columns="critical_temp").values
    y = np.log1p(df["critical_temp"].values)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    std = StandardScaler().fit(X_train_full)
    sel = list(selected_idx)

    def project(A):
        return std.transform(A)[:, sel]

    Z_train, Z_test = project(X_train_full), project(X_test_full)

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
        explained_var=np.array([np.nan]),
        log_target=True,
    )
    np.savez(OUT_NPZ, **out)
    return out


# --------------------------------------------------------------------------
def main():
    X_tr, X_te, y_tr, y_te, names = get_split()
    print(f"[data] deduped rows: {len(X_tr) + len(X_te)} "
          f"(train {len(X_tr)}, test {len(X_te)}), features {X_tr.shape[1]}",
          flush=True)

    X_tr_std = StandardScaler().fit_transform(X_tr)
    rel, red = get_mi(X_tr_std, y_tr)
    print(f"[mi] relevance: min {rel.min():.3f}  max {rel.max():.3f}  "
          f"(top: {names[int(np.argmax(rel))]})", flush=True)
    off = red[~np.eye(len(rel), dtype=bool)]
    print(f"[mi] redundancy off-diag: min {off.min():.3f}  "
          f"max {off.max():.3f}  mean {off.mean():.3f}", flush=True)

    combos = brute_force_terms(len(rel))
    print(f"[qubo] enumerating C(81,4) = {len(combos)} subsets for "
          "certification", flush=True)

    sweep = {}
    for lam in LAMBDAS:
        S, f, stab, hits = sa_select(rel, red, lam)
        S_bf, f_bf = brute_force(rel, red, lam, combos)
        cv = cv_rmse_kelvin(X_tr, y_tr, S)
        certified = (S == S_bf)
        sweep[lam] = dict(S=S, f=f, stability=stab, hits=hits,
                          S_bf=S_bf, f_bf=f_bf, certified=certified, cv=cv)
        print(f"\n[lambda={lam}] SA best obj {f:.4f} | "
              f"stability {stab:.0%} of {N_RESTARTS} restarts | "
              f"global-opt certified: {certified} (brute obj {f_bf:.4f})",
              flush=True)
        for i in S:
            print(f"    {i:2d}  {names[i]:45s} I(X;y)={rel[i]:.3f}")
        print(f"    5-fold CV linear RMSE (train only): {cv:.2f} K")
        if not certified:
            print("    brute-force optimum instead:",
                  [names[i] for i in S_bf])

    # champion = lowest train-CV RMSE (test set untouched for selection)
    lam_star = min(LAMBDAS, key=lambda l: sweep[l]["cv"])
    S_star = sweep[lam_star]["S_bf"]  # certified global optimum
    print(f"\n[champion] lambda={lam_star}  features:", flush=True)
    for i in S_star:
        print(f"    {i:2d}  {names[i]}")

    print(f"\n[npz] building {OUT_NPZ} (prep_data protocol, aligned "
          "subsample)", flush=True)
    d_qmi = prepare_qmi(S_star)

    # alignment check vs existing reductions
    d_pca = dict(np.load("data/prepared_4d.npz"))
    d_pls = dict(np.load("data/prepared_4d_pls.npz"))
    assert np.allclose(d_qmi["y_test"], d_pca["y_test"]), "test rows misaligned!"
    assert np.allclose(d_qmi["y_test"], d_pls["y_test"]), "test rows misaligned!"
    assert np.allclose(d_qmi["y_test"][:50], d_pca["y_test"][:50])
    print("[npz] alignment verified: y_test identical to prepared_4d.npz "
          "and prepared_4d_pls.npz (same 200 materials, same order; "
          "first 50 match)", flush=True)

    from baselines import run_baselines
    results = {}
    for tag, d in (("QMI-4", d_qmi), ("PCA-4", d_pca), ("PLS-4", d_pls)):
        print(f"\n=== {tag} baselines (800 train / 200 test subsample) ===",
              flush=True)
        results[tag] = run_baselines(d)

    print("\n[summary] test RMSE in Kelvin")
    header = None
    for tag, res in results.items():
        if header is None:
            header = list(res.keys())
            print(f"{'reduction':10s} " +
                  " ".join(f"{h:>28s}" for h in header))
        print(f"{tag:10s} " +
              " ".join(f"{res[h][0]:22.2f} K RMSE" for h in header))
    return sweep, lam_star, S_star, names, results


if __name__ == "__main__":
    main()
