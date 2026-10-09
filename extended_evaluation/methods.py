"""Counterfactual generation for the extended evaluation. Usage: python methods.py <mems|d9> <method> [max_instances]
Methods: dice_mlp, dice_mlp_free, dice_rf, alibi_mlp, gs_mlp, face_mlp
Writes out/<dataset>_<method>.csv (one row per counterfactual; resumable)."""
import os, sys, time, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from common import *

name, method = sys.argv[1], sys.argv[2]
maxn = int(sys.argv[3]) if len(sys.argv) > 3 else None
d9 = name == "d9"
NORMAL = 0 if d9 else 1                       # label that means "normal / benign"
df = load_d9() if d9 else load_mems()
cols, tr, te = split(df)
rf, mlp = make_models(df, cols, tr, d9=d9)
lo = df[cols].min().values; hi = df[cols].max().values; rng_ = np.where(hi - lo == 0, 1.0, hi - lo)
inst = json.load(open("./instances.json"))[name]
if method == "dice_rf" and d9:                 # interleave attack types so a partial run stays stratified
    T_ = df.loc[inst, "atype"]; order = T_.groupby(T_).cumcount().sort_values(kind="stable").index
    inst = list(order)
if maxn: inst = inst[:maxn]
clf = rf if method == "dice_rf" else mlp
predict = (lambda X: clf.predict(pd.DataFrame(X, columns=cols))) if method == "dice_rf" else (lambda X: clf.predict(np.asarray(X)))
shap_top10 = json.load(open("./shap_top10.json"))
K = 3

# ---------------- generators: each returns an array (<=3, d) of counterfactuals ----------------
if method.startswith("dice"):
    import dice_ml
    from dice_ml import Dice
    d = dice_ml.Data(dataframe=df[cols + ["label"]], continuous_features=cols, outcome_name="label")
    m = dice_ml.Model(model=clf, backend="sklearn")
    exp = Dice(d, m, method="random")
    if method == "dice_mlp_free": ftv = cols
    elif d9: ftv = shap_top10
    else: ftv = ["x", "z"]
    permitted = None if d9 else {c: [float(df[c].min()), float(df[c].max())] for c in cols}
    def gen(x):
        q = pd.DataFrame([x], columns=cols)
        kw = dict(total_CFs=K, desired_class=0, random_seed=42, features_to_vary=ftv)
        if permitted: kw["permitted_range"] = permitted
        cf = exp.generate_counterfactuals(q, **kw)
        c = cf.cf_examples_list[0].final_cfs_df
        return np.zeros((0, len(cols))) if c is None or len(c) == 0 else c[cols].values.astype(float)

elif method == "alibi_mlp":
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution(); tf.compat.v1.reset_default_graph()
    pf = lambda X: mlp.predict_proba(X).astype(np.float64)
    if not d9:
        from alibi.explainers import Counterfactual
        ex = Counterfactual(pf, shape=(1, len(cols)), target_proba=0.51, tol=0.05, target_class=0, max_iter=1000,
                            lam_init=1e-1, max_lam_steps=10, learning_rate_init=0.1,
                            feature_range=(df[cols].values.min(axis=0).reshape(1, -1), df[cols].values.max(axis=0).reshape(1, -1)))
    else:
        from alibi.explainers import CounterfactualProto
        ex = CounterfactualProto(pf, shape=(1, len(cols)), use_kdtree=True, kappa=0.1, beta=0.1, theta=10.0, c_init=1.0,
                                 c_steps=5, max_iterations=500, learning_rate_init=0.01,
                                 feature_range=(df[cols].values.min(axis=0).reshape(1, -1), df[cols].values.max(axis=0).reshape(1, -1)))
        ex.fit(df.loc[tr][df.loc[tr, "label"] == 0][cols].values, d_type="abdm", disc_perc=[25, 50, 75])
    def gen(x):
        q = np.asarray(x, dtype=np.float64).reshape(1, -1); cands = []
        try:
            e = ex.explain(q)
            if e.cf is not None: cands.append(np.asarray(e.cf["X"]).reshape(-1))
            if hasattr(e, "all") and e.all:
                for _, lst in e.all.items():
                    for c in lst: cands.append(np.asarray(c["X"] if isinstance(c, dict) else c).reshape(-1))
        except Exception as err:
            print("alibi error:", err, flush=True)
        seen, sc = set(), []
        for c in cands:
            key = tuple(np.round(c, 4))
            if key in seen: continue
            seen.add(key)
            if mlp.predict(c.reshape(1, -1))[0] != NORMAL: continue
            sc.append((float(np.sum(np.abs(c - q.reshape(-1)) / rng_)), c))
        sc.sort(key=lambda t: t[0])
        return np.array([c for _, c in sc[:K]]).reshape(-1, len(cols))

elif method == "gs_mlp":                      # Growing Spheres (Laugel et al., 2018), own implementation
    def gen(x):
        z0 = (np.asarray(x) - lo) / rng_; dim = len(z0); out = []
        for seed in range(K):                 # 3 independent runs -> 3 counterfactuals
            r_ = np.random.RandomState(seed); n = 1000
            def shell(a, b):
                v = r_.normal(size=(n, dim)); v /= np.linalg.norm(v, axis=1, keepdims=True)
                rad = (a ** dim + r_.uniform(size=n) * (b ** dim - a ** dim)) ** (1.0 / dim)
                return np.clip(z0 + v * rad[:, None], 0, 1)
            enemy = lambda Z: predict(Z * rng_ + lo) == NORMAL
            eta = 0.1; Z = shell(0, eta); s = 0
            while enemy(Z).any() and s < 20: eta /= 2; Z = shell(0, eta); s += 1
            a, b, found = 0.0, eta, None
            for _ in range(2000):
                Z = shell(a, b); e = enemy(Z)
                if e.any():
                    cand = Z[e]; found = cand[np.argmin(np.linalg.norm(cand - z0, axis=1))]; break
                a, b = b, b + eta
                if a > np.sqrt(dim): break
            if found is None: continue
            order = np.argsort(np.abs(found - z0))           # sparsification: reset features, smallest change first
            for j in order:
                t = found.copy(); t[j] = z0[j]
                if enemy(t.reshape(1, -1))[0]: found = t
            out.append(found * rng_ + lo)
        return np.array(out).reshape(-1, dim)

elif method == "face_mlp":                    # FACE-style graph search (Poyiadzi et al., 2020), simplified own implementation
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix, bmat
    from scipy.sparse.csgraph import dijkstra
    T = df.loc[te]; T = T[~T.index.isin(inst)]
    nodes = T.sample(n=min(3000, len(T)), random_state=42)[cols].values
    Zn = (nodes - lo) / rng_; M = len(Zn); kk = 10
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(Zn)
    rk = nn.kneighbors(Zn)[0][:, -1]; rho = float(np.median(rk[rk > 0])) if (rk > 0).any() else 1.0   # duplicates give rk=0
    def dens_w(dist, mid):                     # distance x (1 + (r_k(midpoint)/rho)^2): sparse regions cost more
        rm = nn.kneighbors(mid, n_neighbors=kk)[0][:, -1]
        return np.maximum(dist, 1e-6) * (1.0 + (rm / rho) ** 2)
    dist, idx = nn.kneighbors(Zn); rows, cs, ws = [], [], []
    for i in range(M):
        j = idx[i, 1:]; dd = dist[i, 1:]
        w = dens_w(dd, (Zn[i] + Zn[j]) / 2)
        rows += [i] * len(j); cs += list(j); ws += list(w)
    from scipy.spatial.distance import cdist
    from scipy.sparse.csgraph import minimum_spanning_tree
    Dm = cdist(Zn, Zn); Dm[Dm == 0] = 1e-9; np.fill_diagonal(Dm, 0)     # duplicate rows -> tiny positive edge
    mst = minimum_spanning_tree(csr_matrix(Dm)).tocoo()                  # backbone guarantees a connected graph
    wm = dens_w(mst.data, (Zn[mst.row] + Zn[mst.col]) / 2)
    rows += list(mst.row); cs += list(mst.col); ws += list(wm)
    G = csr_matrix((ws, (rows, cs)), shape=(M, M))          # kNN + MST edges; Dijkstra runs with directed=False
    npred = predict(nodes); ok = (npred == NORMAL) & (rk <= np.percentile(rk, 95))
    def gen(x):
        z = ((np.asarray(x) - lo) / rng_).reshape(1, -1)
        dq, jq = nn.kneighbors(z, n_neighbors=kk); dq, jq = dq[0], jq[0]
        wq = dens_w(dq, (z + Zn[jq]) / 2)
        r = csr_matrix((wq, (np.zeros(kk, dtype=int), jq)), shape=(1, M))
        G2 = bmat([[G, r.T], [r, None]], format='csr')
        cost = dijkstra(G2, directed=False, indices=M)[:M]
        cost = np.where(ok & np.isfinite(cost), cost, np.inf)
        best = np.argsort(cost)[:K]; best = best[np.isfinite(cost[best])]
        return nodes[best].reshape(-1, len(cols))
else:
    raise SystemExit("unknown method")

# ---------------- run (resumable) ----------------
path = f"./out/{name}_{method}.csv"
done = set(pd.read_csv(path)["inst"]) if os.path.exists(path) else set()
first = not os.path.exists(path)
for n_, i in enumerate(inst):
    if i in done: continue
    x = df.loc[i, cols].values.astype(float)
    t0 = time.time()
    try: C = gen(x)
    except Exception as err:
        print("generator error:", err, flush=True); C = np.zeros((0, len(cols)))
    dt = time.time() - t0
    rows = [dict(inst=i, cf_id=k, time=dt, **dict(zip(cols, c))) for k, c in enumerate(C)] or [dict(inst=i, cf_id=-1, time=dt, **{c: np.nan for c in cols})]
    pd.DataFrame(rows).to_csv(path, mode="a", header=first, index=False); first = False
    print(f"[{name}/{method}] {n_+1}/{len(inst)} inst={i} cfs={len(C)} time={dt:.1f}s", flush=True)
print("DONE", flush=True)
