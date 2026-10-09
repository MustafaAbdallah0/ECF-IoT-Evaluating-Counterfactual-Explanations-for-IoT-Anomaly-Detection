import warnings; warnings.filterwarnings("ignore")
import sys, json, numpy as np, pandas as pd, shap
sys.path.insert(0, ".")
from common import *
df = load_d9(); cols, tr, te = split(df)
rf, _ = make_models(df, cols, tr, d9=True)
print("RF test acc:", rf.score(df.loc[te, cols], df.loc[te, "label"]))
rng = np.random.RandomState(0)
atk_tr = df.loc[tr][df.loc[tr, "label"] == 1]
samp = atk_tr.sample(n=500, random_state=0)           # TRAINING attack rows, disjoint from evaluated test instances
sv = shap.TreeExplainer(rf).shap_values(samp[cols])
sv = sv[1] if isinstance(sv, list) else sv[:, :, 1]
imp = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
print(imp.head(12))
json.dump(imp.head(10).index.tolist(), open("./shap_top10.json", "w"))
print("saved:", imp.head(10).index.tolist())
