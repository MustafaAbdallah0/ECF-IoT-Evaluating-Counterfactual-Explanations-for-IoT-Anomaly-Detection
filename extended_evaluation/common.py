"""Shared data prep, models and metrics for the extended evaluation.
Reproduces the preprocessing of DiCE.ipynb (cells 2,3,5,6,37) exactly."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

DATA = "../"
SPARSITY_TOL = 1e-3

def downsample_to_min_class(df, label_col="label", random_state=42):
    counts = df[label_col].value_counts(); n_min = counts.min(); parts = []
    for cls, n in counts.items():
        parts.append(resample(df[df[label_col] == cls], replace=False, n_samples=n_min, random_state=random_state))
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)

def load_mems():
    df = downsample_to_min_class(pd.read_csv(DATA + "mems_dataset.csv")).select_dtypes(include=[np.number])
    X = df.drop(columns=["label"]); y = df["label"]
    out = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns.tolist()); out["label"] = y.values
    out["atype"] = out["label"]          # true class (1 normal, 2/3 anomalous)
    return out                            # label: 1=normal, 2,3 anomalies

def load_d9():
    raw = pd.read_csv(DATA + "device9_top_20_features.csv")
    d = raw.select_dtypes(include=[np.number]).copy()
    d.replace([np.inf, -np.inf], np.nan, inplace=True); d.dropna(inplace=True)
    X = d.drop(columns=["label"]); y = d["label"]
    Xs = StandardScaler().fit_transform(X)
    feature_cols = raw.select_dtypes(include=[np.number]).drop(columns=["label"]).columns
    dc = pd.DataFrame(Xs, columns=feature_cols); dc["label"] = y.values
    ATT = list(range(2, 12))
    benign_part = dc[dc["label"] == 1]
    attack_parts = [dc[dc["label"] == l].sample(n=1900, random_state=42) for l in ATT]
    dc = pd.concat([benign_part] + attack_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    # notebook cell 37 (binarise), carrying the attack type along
    ben = dc[dc["label"] == 1].copy(); ben["atype"] = 1; ben["label"] = 0
    atk = []
    for l in ATT:
        p = dc[dc["label"] == l].sample(n=1900, random_state=42).copy(); p["atype"] = l; p["label"] = 1; atk.append(p)
    return pd.concat([ben, pd.concat(atk)]).sample(frac=1, random_state=42).reset_index(drop=True)  # label 0 benign, 1 attack

def split(df):
    cols = [c for c in df.columns if c not in ("label", "atype")]
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=df["label"].values)
    return cols, tr, te

def make_models(df, cols, tr, d9=False):
    import joblib, os
    path = f"cache_{'d9' if d9 else 'mems'}.joblib"      # training is deterministic (seed 42); cache avoids retraining
    if os.path.exists(path): return joblib.load(path)
    Xtr = df.loc[tr, cols]; ytr = df.loc[tr, "label"].values
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, **({"max_features": None} if d9 else {})).fit(Xtr, ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42).fit(Xtr.values, ytr)
    joblib.dump((rf, mlp), path)
    return rf, mlp
