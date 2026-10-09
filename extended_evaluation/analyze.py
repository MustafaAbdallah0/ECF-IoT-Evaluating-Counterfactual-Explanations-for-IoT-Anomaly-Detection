import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, ".")
from common import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

REUSE = '--reuse' in sys.argv      # --reuse: recompute all statistics from the shipped out/instance_metrics_*.csv (no model retraining)
K = 3; B = 10000; RNG = np.random.RandomState(2024)
METHODS = {"dice_mlp": "DiCE (restricted features)", "dice_mlp_free": "DiCE (all features free)", "alibi_mlp": "Alibi-Explain",
           "gs_mlp": "Growing Spheres", "face_mlp": "FACE-style graph", "dice_rf": "DiCE on Random Forest"}
res = {}

def boot_ci(v, stat=np.mean):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    bs = np.array([stat(v[RNG.randint(0, len(v), len(v))]) for _ in range(B)])
    return (float(stat(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

def wilson(k, n, z=1.959964):
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

for name in ["mems", "d9"]:
    d9 = name == "d9"; normal = 0 if d9 else 1
    df = load_d9() if d9 else load_mems(); cols, tr, te = split(df)
    rf, mlp = (None, None) if REUSE else make_models(df, cols, tr, d9=d9)
    mpath = f"out/instance_metrics_{name}.csv"; have_tab = REUSE and os.path.exists(mpath)
    reuse_tab = pd.read_csv(mpath, index_col=[0, 1]) if have_tab else None
    lo = df[cols].min().values; hi = df[cols].max().values; rg = np.where(hi - lo == 0, 1.0, hi - lo)
    inst = json.load(open("./instances.json"))[name]
    benign = df[df["label"] == normal]
    btr = benign.loc[benign.index.isin(tr), cols].values; bte = benign.loc[benign.index.isin(te), cols].values
    # ---- three plausibility detectors: fit on benign TRAIN rows, 5% rejection threshold calibrated on held-out benign TEST rows
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(btr)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(btr)
    knn = NearestNeighbors(n_neighbors=5).fit(btr)
    kd = lambda X: knn.kneighbors(X)[0][:, -1]
    thr = {"IF": np.percentile(iso.decision_function(bte), 5), "LOF": np.percentile(lof.score_samples(bte), 5), "kNN": np.percentile(kd(bte), 95)}
    plaus = lambda X: {"IF": iso.decision_function(X) >= thr["IF"], "LOF": lof.score_samples(X) >= thr["LOF"], "kNN": kd(X) <= thr["kNN"]}
    X0 = df.loc[inst, cols].values
    base = {k: float(v.mean()) for k, v in plaus(X0).items()}
    heldout = {k: float(v.mean()) for k, v in plaus(bte).items()}
    cen = btr.mean(axis=0); dist0 = (np.abs(X0 - cen) / rg).sum(axis=1)
    out = {"n_instances": len(inst), "baseline_original": base, "baseline_heldout_benign": heldout, "methods": {}}
    inst_tab = {}
    for me, label in METHODS.items():
        f = f"./out/{name}_{me}.csv"
        if not os.path.exists(f): continue
        o = pd.read_csv(f); n_done = o["inst"].nunique()
        if n_done == 0: continue
        clf = rf if me == "dice_rf" else mlp   # unused in --reuse mode
        rows = []
        for i, g in ([] if have_tab else o.groupby("inst")):
            g = g[g.cf_id >= 0]
            x = df.loc[i, cols].values.astype(float); t = float(o[o.inst == i]["time"].iloc[0])
            if len(g):
                C = g[cols].values
                valid = (clf.predict(pd.DataFrame(C, columns=cols)) if me == "dice_rf" else clf.predict(C)) == normal
                dl = np.abs(C - x) / rg; l1 = dl.sum(1); nch = (dl > SPARSITY_TOL).sum(1)
                P = plaus(C)
            else:
                valid = np.zeros(0, bool); l1 = nch = np.zeros(0); P = {k: np.zeros(0, bool) for k in thr}
            nv = int(valid.sum())
            r = {"inst": i, "validity": nv / K, "L1": l1[valid].mean() if nv else np.nan, "spars": (nch[valid] / len(cols)).mean() if nv else np.nan,
                 "time": t, "dist": dist0[inst.index(i)]}
            for k in P:
                r["pl_" + k] = (P[k][valid]).mean() if nv else np.nan            # P(plausible | valid)
                r["vp_" + k] = (P[k] & valid).sum() / K                          # valid-and-plausible per requested CF
            rows.append(r)
        T = reuse_tab.loc[me].copy() if have_tab else pd.DataFrame(rows).set_index("inst"); inst_tab[me] = T
        m = {"label": label, "n_inst": int(n_done), "n_valid_cf": int(round(T["validity"].sum() * K))}
        for c in ["validity", "L1", "spars", "time", "pl_IF", "pl_LOF", "pl_kNN", "vp_IF", "vp_LOF", "vp_kNN"]:
            m[c] = boot_ci(T[c].values)
        out["methods"][me] = m
    # ---- paired tests DiCE (restricted) vs Alibi on the same instances (shared MLP)
    if "dice_mlp" in inst_tab and "alibi_mlp" in inst_tab:
        A, Bm = inst_tab["dice_mlp"], inst_tab["alibi_mlp"]; common = A.index.intersection(Bm.index)
        tests = {}
        for c in ["validity", "L1", "spars", "pl_IF", "vp_IF", "vp_LOF", "vp_kNN"]:
            a, b = A.loc[common, c].values, Bm.loc[common, c].values; ok = ~np.isnan(a) & ~np.isnan(b)
            a, b = a[ok], b[ok]; diff = a - b
            if len(diff) < 2 or np.allclose(diff, 0): p = 1.0
            else: p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
            mean, l, u = boot_ci(diff)
            tests[c] = {"n_pairs": int(len(diff)), "mean_diff": mean, "ci": [l, u], "p": p, "median_dice": float(np.median(a)), "median_alibi": float(np.median(b))}
        keys = list(tests); adj = holm([tests[k]["p"] for k in keys])
        for k, a_ in zip(keys, adj): tests[k]["p_holm"] = float(a_)
        SIX = ["validity", "L1", "spars", "vp_IF", "vp_LOF", "vp_kNN"]          # the six comparisons shown in Table X of the paper
        for k, a_ in zip(SIX, holm([tests[k]["p"] for k in SIX])): tests[k]["p_holm_six"] = float(a_)
        out["paired_dice_vs_alibi"] = tests
    # ---- classifier effect: DiCE on Random Forest vs DiCE on MLP, same instances
    if "dice_rf" in inst_tab and "dice_mlp" in inst_tab:
        R_, M_ = inst_tab["dice_rf"], inst_tab["dice_mlp"]; common = R_.index.intersection(M_.index); ce = {"n_common": int(len(common))}
        for c in ["validity", "L1", "spars"]:
            a_, b_ = R_.loc[common, c].values, M_.loc[common, c].values; ok = ~np.isnan(a_) & ~np.isnan(b_); a_, b_ = a_[ok], b_[ok]; diff = a_ - b_
            p = 1.0 if (len(diff) < 2 or np.allclose(diff, 0)) else float(stats.wilcoxon(a_, b_).pvalue)
            m_, l_, u_ = boot_ci(diff); ce[c] = {"n_pairs": int(len(diff)), "mean_rf": float(a_.mean()), "mean_mlp": float(b_.mean()), "mean_diff_rf_minus_mlp": m_, "ci": [l_, u_], "p": p}
        # same-instance validity of the other methods, for a like-for-like table
        for me in ["alibi_mlp"]:
            ce["alibi_validity_on_common"] = boot_ci(inst_tab[me].loc[common, "validity"].values)
        ce["dice_mlp_validity_on_common"] = boot_ci(M_.loc[common, "validity"].values); ce["dice_rf_validity_on_common"] = boot_ci(R_.loc[common, "validity"].values)
        out["classifier_effect_dice"] = ce
    # ---- distance vs validity (supports the centroid-distance claim)
    if "alibi_mlp" in inst_tab and "dice_mlp" in inst_tab:
        dd = {}
        for me in ["dice_mlp", "alibi_mlp", "gs_mlp", "face_mlp"]:
            if me not in inst_tab: continue
            T = inst_tab[me]; rho, p = stats.spearmanr(T["dist"], T["validity"])
            full = T[T.validity == 1]["dist"]; part = T[T.validity < 1]["dist"]
            mw = stats.mannwhitneyu(full, part, alternative="two-sided") if len(full) > 1 and len(part) > 1 else None
            dd[me] = {"spearman_rho": float(rho), "spearman_p": float(p), "n_full": int(len(full)), "n_not_full": int(len(part)),
                      "mw_p": float(mw.pvalue) if mw else None, "median_dist_full": float(full.median()) if len(full) else None, "median_dist_notfull": float(part.median()) if len(part) else None}
        out["distance_vs_validity"] = dd
    # ---- original paper run: CF-level counts -> Wilson CI + Fisher exact (CFs are not independent; indicative only)
    orig = {"mems": {"DiCE": (70, 72), "Alibi-Explain": (66, 72)}, "d9": {"DiCE": (40, 78), "Alibi-Explain": (49, 78)}}[name]
    ow = {k: {"k": v[0], "n": v[1], "rate": v[0] / v[1], "wilson": list(wilson(*v))} for k, v in orig.items()}
    (kd_, nd_), (ka_, na_) = orig["DiCE"], orig["Alibi-Explain"]
    ow["fisher_p"] = float(stats.fisher_exact([[kd_, nd_ - kd_], [ka_, na_ - ka_]])[1]); out["original_run"] = ow
    res[name] = out
    inst_tab_all = {me: T for me, T in inst_tab.items()}
    if not have_tab: pd.concat(inst_tab_all, names=["method", "inst"]).to_csv(f"./out/instance_metrics_{name}.csv")
json.dump(res, open("./results.json", "w"), indent=1, default=float)
print("saved results.json")
