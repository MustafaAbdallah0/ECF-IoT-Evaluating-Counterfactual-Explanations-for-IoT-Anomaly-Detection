# Extended evaluation

Shared-classifier re-evaluation of DiCE and Alibi-Explain on **held-out** instances, with two additional baselines (Growing Spheres, a
FACE-style graph search), three plausibility detectors, bootstrap confidence intervals and paired significance tests
(Sections V-G and VI-D of the paper).

**Always run commands from inside this folder.** The scripts read `../mems_dataset.csv` and `../device9_top_20_features.csv` from the
repository root.

## Quick check (no training, about 20 s)

```bash
python analyze.py --reuse && python make_tables.py
```

This recomputes every number of `results.json` from `out/instance_metrics_*.csv` and writes the paper's four tables to `tables/`.
Needs only numpy, pandas, scikit-learn and scipy.

## Files

| File | Purpose |
|---|---|
| `common.py` | Data preparation identical to `DiCE.ipynb`, 80/20 stratified split (seed 42), Random Forest and MLP training (models are cached in `cache_*.joblib`, not committed) |
| `select_instances.py` -> `instances.json` | Held-out instances that both models predict as anomalous: 50 + 50 for MEMS, 10 per attack type (types 2 and 5-11) for N-BaIoT |
| `shap_top10.py` -> `shap_top10.json` | SHAP ranking for N-BaIoT, recomputed on 500 *training* attack rows (needs `shap`; use `requirements-shap.txt`) |
| `methods.py <mems\|d9> <method> [n]` | Generates counterfactuals: `dice_mlp`, `dice_mlp_free` (all features free), `dice_rf`, `alibi_mlp`, `gs_mlp` (Growing Spheres), `face_mlp` (FACE-style). Writes `out/<dataset>_<method>.csv`; resumable |
| `analyze.py [--reuse]` | Per-instance metrics, three plausibility detectors (IsolationForest, LOF, kNN), bootstrap CIs, Wilcoxon + Holm, Spearman, Mann-Whitney, Wilson/Fisher -> `results.json`, `out/instance_metrics_*.csv` |
| `make_tables.py` | Writes the LaTeX tables of Section VI-D from `results.json` |
| `run_all.sh`, `run_all.ps1` | Run the whole pipeline |

## Full rerun

```bash
pip install -r ../requirements.txt                       # Python 3.11
python select_instances.py                               # or keep the shipped instances.json
python methods.py mems dice_mlp                          # repeat for alibi_mlp, gs_mlp, face_mlp, dice_mlp_free, then for d9
python methods.py d9 dice_rf 45                          # slow; the paper uses the first 45 of the 80 instances (attack types interleaved)
python analyze.py && python make_tables.py
```

Typical times on one CPU core: Alibi-Explain 4-6 s per instance, DiCE on the MLP 1-3 s, Growing Spheres and FACE under 1 s, DiCE on the
Random Forest up to 2 minutes per instance when it finds counterfactuals.

## Notes

* Growing Spheres and the FACE-style search are our own implementations (see Section V-G); the FACE-style search is simplified.
* DiCE on the Random Forest was run on a stratified 45 of the 80 N-BaIoT instances and not on MEMS, because it is slow.
* `results.json` stores two Holm columns for the paired tests: `p_holm` (over 7 comparisons) and `p_holm_six` (over the 6 comparisons
  shown in Table X of the paper, which is what the paper reports).
* **Retraining sensitivity.** Rerunning `analyze.py` without `--reuse` retrains the MLP in your environment. Boundary-hugging
  counterfactuals can change class under slightly different weights, so some N-BaIoT numbers move (36 of 590 quantities in our check;
  MEMS identical). The shipped `out/` files and `results.json` are the canonical results.
* Model caches (`cache_*.joblib`) are created on first run and are not committed.
