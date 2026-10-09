import sys, json, numpy as np, pandas as pd
sys.path.insert(0, ".")
from common import *
out = {}
for name in ["mems", "d9"]:
    df = load_mems() if name == "mems" else load_d9()
    cols, tr, te = split(df)
    rf, mlp = make_models(df, cols, tr, d9=(name == "d9"))
    T = df.loc[te].copy()
    T["pred_rf"] = rf.predict(T[cols]); T["pred_mlp"] = mlp.predict(T[cols].values)
    normal = 1 if name == "mems" else 0
    anom = T[T["label"] != normal]
    print(f"\n== {name}: test rows {len(T)}; MLP acc {(T['pred_mlp']==T['label']).mean():.4f}; RF acc {(T['pred_rf']==T['label']).mean():.4f}")
    g = anom.assign(mlp_ok=anom["pred_mlp"] != normal, rf_ok=anom["pred_rf"] != normal)
    tab = g.groupby("atype").agg(n_test=("label", "size"), mlp_pred_anom=("mlp_ok", "sum"), rf_pred_anom=("rf_ok", "sum"), both=("mlp_ok", lambda s: int((s & g.loc[s.index, "rf_ok"]).sum())))
    print(tab)
    # sample: up to 50 per class (MEMS) / 10 per type (D9) among instances BOTH models predict anomalous
    per = 50 if name == "mems" else 10
    chosen = []
    for t, grp in g[g["mlp_ok"] & g["rf_ok"]].groupby("atype"):
        chosen += grp.sample(n=min(per, len(grp)), random_state=42).index.tolist()
    print("selected:", len(chosen), "instances")
    out[name] = [int(i) for i in chosen]
json.dump(out, open("./instances.json", "w"))
