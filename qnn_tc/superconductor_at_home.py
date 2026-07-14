"""Superconductor@home -- one volunteer WORK UNIT for distributed, uncertainty-
aware virtual screening of superconductor candidates.

Uses a HYBRID 8-feature encoding (4 physics features + 4 PLS components), a
richer input space than the shipped PLS-4 champion QNN (19.84 K, see README).

Pipeline per candidate: featurize -> 81 features -> [4 physics | 4 PLS] ->
ensemble predict (mean Tc + uncertainty) -> OOD flag -> acquisition rank.
In production each ensemble member is a QNN variant; here K=12 fast bootstrap
surrogates on the hybrid features demonstrate the pipeline.
"""
import os, sys, hashlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from featurize_formula import featurize, feature_columns

# Raw UCI CSVs are downloaded into data/ (see qnn_tc/README.md), same as every other script.
def data(name): return os.path.join(HERE, "data", name)

RNG = np.random.default_rng(42)
etable = pd.read_csv(data("element_properties_uci.csv"), index_col=0)
BASIS = [e for e in etable.index if not etable.loc[e].isna().any()]
FEATS = feature_columns()

def physics4(f, cu, o):
    return np.array([1.0/np.sqrt(f["wtd_mean_atomic_mass"]), f["wtd_mean_Valence"],
                     0.5*(f["range_fie"]+f["range_ElectronAffinity"]),
                     2*np.sqrt(max(cu,0)*max(o,0))])

# ---------- training data: build HYBRID 8-feature matrix ----------
tr = pd.read_csv(data("train.csv")); um = pd.read_csv(data("unique_m.csv"))
y = tr["critical_temp"].values
X81 = tr[FEATS].values
phys_tr = np.column_stack([1.0/np.sqrt(tr["wtd_mean_atomic_mass"]), tr["wtd_mean_Valence"],
            0.5*(tr["range_fie"]+tr["range_ElectronAffinity"]),
            2*np.sqrt(np.clip(um["Cu"],0,None)*np.clip(um["O"],0,None))])
sc81 = StandardScaler().fit(X81)
pls = PLSRegression(n_components=4).fit(sc81.transform(X81), y)
Xtrain = np.column_stack([phys_tr, pls.transform(sc81.transform(X81))])   # 8 hybrid features
print(f"[work unit] hybrid encoding: {Xtrain.shape[1]} features (4 physics + 4 PLS)", flush=True)

K = 12
members = []
for m in range(K):
    idx = RNG.choice(len(Xtrain), 5000, replace=True)
    members.append(ExtraTreesRegressor(n_estimators=60, n_jobs=-1, random_state=m).fit(Xtrain[idx], y[idx]))
print(f"[work unit] trained {K}-member ensemble on {len(Xtrain)} known superconductors", flush=True)

scaler = StandardScaler().fit(Xtrain)
nn = NearestNeighbors(n_neighbors=8).fit(scaler.transform(Xtrain))
ood_cut = np.quantile(nn.kneighbors(scaler.transform(Xtrain))[0].mean(1), 0.99)

# ---------- candidate generation ----------
CUPRATE_A = [e for e in ["La","Ba","Sr","Ca","Y","Bi","Tl","Hg","Nd"] if e in BASIS]
IRON_X    = [e for e in ["As","Se","Te","P"] if e in BASIS]
POOL = [e for e in ["O","Cu","La","Ba","Sr","Ca","Y","Bi","Tl","Hg","Fe","As","Se","Te",
                    "H","Mg","B","Nb","Ti","Sn","Pb","K","Rb","Ni","Co","P","S","Sb",
                    "Ge","Si","F","C","Zr","V","Mo"] if e in BASIS]
def gen(mode):
    if mode=="cuprate":  els=list(RNG.choice(CUPRATE_A,RNG.integers(1,3),replace=False))+["Cu","O"]
    elif mode=="iron":   els=list(dict.fromkeys(["Fe",str(RNG.choice(IRON_X))]+list(RNG.choice(CUPRATE_A,1))))
    else:                els=list(RNG.choice(POOL,RNG.integers(2,6),replace=False))
    amts=np.round(RNG.uniform(0.1,6.0,len(els)),2)
    return "".join(f"{e}{a}" for e,a in zip(els,amts)), els, amts

N=4000; modes=RNG.choice(["cuprate","iron","random"],N,p=[0.35,0.15,0.50])
forms=[]; phys=[]; v81s=[]; gens=[]
for mode in modes:
    formula, els, amts = gen(mode)
    try: f = featurize(formula, etable)
    except ValueError: continue
    frac = amts/amts.sum()
    cu = frac[els.index("Cu")] if "Cu" in els else 0.0
    o  = frac[els.index("O")]  if "O"  in els else 0.0
    forms.append(formula); phys.append(physics4(f,cu,o)); v81s.append(f[FEATS].values); gens.append(mode)
# batch: build hybrid features for candidates
Xc = np.column_stack([np.array(phys), pls.transform(sc81.transform(np.array(v81s)))])
print(f"[work unit] featurized {len(forms)}/{N} candidates -> hybrid 8-feature", flush=True)

# ---------- ensemble scoring ----------
preds = np.stack([m.predict(Xc) for m in members])
mean_tc, uncert = preds.mean(0), preds.std(0)
novelty = nn.kneighbors(scaler.transform(Xc))[0].mean(1)
res = pd.DataFrame({"formula":forms, "gen":gens, "pred_Tc":mean_tc, "uncertainty":uncert,
                    "novelty":novelty, "in_distribution":novelty<=ood_cut})
res["lcb"] = res.pred_Tc - res.uncertainty
HIGH = np.quantile(res.pred_Tc, 0.90)
shortlist = res[res.in_distribution].sort_values("lcb", ascending=False)

print(f"\n[work unit] predicted-Tc: median {np.median(mean_tc):.0f} K, 90th {HIGH:.0f} K, max {mean_tc.max():.0f} K")
print(f"[work unit] screened {len(res)}: OOD-dropped {(~res.in_distribution).sum()}")
hi,lo = res[res.pred_Tc>HIGH], res[res.pred_Tc<=HIGH]
print(f"   avg uncertainty high-Tc {hi.uncertainty.mean():.1f} K vs rest {lo.uncertainty.mean():.1f} K")
print("   directed vs brute force (mean predicted Tc):")
for g in ["cuprate","iron","random"]:
    s=res[res.gen==g]; print(f"     {g:8s}: {s.pred_Tc.mean():5.1f} K over {len(s)}")
print("\nTop 8 by acquisition score (hybrid model, uncertainty-penalised, in-distribution):")
print(shortlist.head(8)[["formula","gen","pred_Tc","uncertainty","lcb","novelty"]].to_string(index=False,
      formatters={"pred_Tc":"{:.1f}".format,"uncertainty":"{:.1f}".format,"lcb":"{:.1f}".format,"novelty":"{:.2f}".format}))
res.to_csv(data("screening_workunit_results.csv"), index=False)
print(f"\n[work unit] hash {hashlib.sha256(np.round(mean_tc,4).tobytes()).hexdigest()[:16]}  | saved results")
