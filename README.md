# ECF-IoT: Evaluating Counterfactual Explanations for IoT Anomaly Detection

Code, data and results for the paper

> **Evaluating Counterfactual Explanations for IoT Anomaly Detection: A Comparative Study**
> Joseph Rizzo and Mustafa Abdallah, Purdue University (corresponding author: M. Abdallah).
> Paper: [`paper/Counterfactual_Explanations_IoT_Rizzo_Abdallah.pdf`](paper/Counterfactual_Explanations_IoT_Rizzo_Abdallah.pdf)

Explainable-AI methods such as SHAP, LIME, Anchor and RuleFit tell an operator *why* a model flagged an IoT device, but not *what would
have to change* for it to look normal. Counterfactual explanations answer the second question. This project compares two counterfactual
generators, **DiCE** (randomized search) and **Alibi-Explain** (gradient search: Wachter-style on MEMS, CounterfactualProto on N-BaIoT),
on two IoT anomaly-detection datasets, and scores every counterfactual on **validity**, **proximity** (L1), **sparsity**,
**plausibility** and generation time.

## Getting the code

```bash
git clone https://github.com/MustafaAbdallah0/ECF-IoT-Evaluating-Counterfactual-Explanations-for-IoT-Anomaly-Detection.git
cd ECF-IoT-Evaluating-Counterfactual-Explanations-for-IoT-Anomaly-Detection
```

## Repository layout

```
.
├── README.md
├── DiCE.ipynb                      main experiments (DiCE on a Random Forest, Alibi-Explain on an MLP; 24 MEMS / 26 N-BaIoT instances)
├── mems_dataset.csv                MEMS vibration data (3 features, 3 classes)
├── device9_top_20_features.csv     N-BaIoT Device 9 (security camera): 20 features, 11 classes, 19,000 rows per class
├── extended_evaluation/            shared-classifier re-evaluation on held-out instances (Section VI-D of the paper)
│   ├── README.md                   how to run it, file by file
│   ├── methods.py, analyze.py ...  code
│   ├── instances.json              the evaluated instances (100 MEMS, 80 N-BaIoT)
│   ├── out/                        every counterfactual that was generated, plus per-instance metrics
│   └── results.json                all statistics reported in the paper
├── paper/                          the manuscript (PDF)
├── requirements.txt                environment for the notebook and the extended evaluation (Python 3.11)
└── requirements-shap.txt           separate environment for the SHAP step (Python 3.12)
```

## Main results at a glance

**Main experiments** (`DiCE.ipynb`; DiCE explains a Random Forest, Alibi-Explain explains an MLP):

| | MEMS DiCE | MEMS Alibi | N-BaIoT DiCE | N-BaIoT Alibi |
|---|---:|---:|---:|---:|
| Validity | 97.2% | 91.67% | 51.28% | 62.82% |
| Proximity (L1) | 0.226 | 0.127 | 1.409 | 1.310 |
| Sparsity (% features changed) | 40.00% | 64.14% | 18.25% | 43.47% |
| Plausibility (IsolationForest, 5%) | 80.00% | 90.91% | 0% | 0% |
| Generation time | 24m 41s | 1m 51s | 33m 1s | 2m 59s |

**Extended evaluation** (`extended_evaluation/`; every method explains the *same* MLP; held-out test instances that the model predicts as
anomalous; 95% bootstrap confidence intervals and paired Wilcoxon tests in the paper):

| | MEMS (n=100) DiCE / Alibi | N-BaIoT (n=80) DiCE / Alibi |
|---|---:|---:|
| Validity | 0.99 / 0.95 (not significant) | **0.98 / 0.71** (p < 0.001) |
| Proximity (L1) | 0.258 / **0.121** | **1.004** / 1.163 |
| Sparsity (% features changed) | **37.8** / 70.9 | **14.9** / 48.9 |

* The N-BaIoT validity ranking of the main experiments **reverses** on a shared classifier. For the same 45 instances DiCE reaches
  validity 0.37 on the Random Forest and 0.98 on the MLP, so the gap in the main experiments is mostly a classifier effect.
* The two validity differences of the main experiments are not statistically significant (Fisher exact test, p = 0.275 and p = 0.196).
* On N-BaIoT no counterfactual of DiCE, Alibi-Explain or Growing Spheres is accepted by the plausibility detectors under the protocol of
  the paper; this depends on how the detectors are fitted (see *Known data properties* below).
* The association between distance from the benign region and validity seen in the main experiments is not statistically supported on a
  shared classifier (Spearman rho = -0.16 for DiCE and +0.20 for Alibi-Explain on N-BaIoT, both not significant).

## Data

| File | Content |
|---|---|
| `mems_dataset.csv` | MEMS accelerometer data from a motor test bed; features `x`, `y`, `z`; label 1 = normal, labels 2 and 3 = the two anomalous states (near-failure and failure) |
| `device9_top_20_features.csv` | N-BaIoT Device 9: 20 statistical traffic features (`HH_L1/L3/L5_*`), label 1 = benign, 2-11 = ten botnet attack types (Gafgyt and Mirai); 209,000 rows |

Both files are the data used in the paper (see the paper for the original sources: Meidan et al., 2018 for N-BaIoT, and the MEMS test-bed
data of the E-RXAI-IoT benchmark). They remain subject to the terms of those sources; please check them before redistributing.

### Known data properties (measured on `device9_top_20_features.csv`)

* The benign class has only **77 distinct feature vectors** among its 19,000 rows (64 among the 15,200 training rows); 99.7% of held-out
  benign rows duplicate a training row.
* About a third of all attack rows are **identical to a benign vector**. Attack types 3 and 4 have only 2 distinct vectors each and 99.9%
  of their rows equal a benign vector, so no classifier can detect them in this 20-feature representation (types 2, 5, 7, 8 and 11: 22-36%;
  type 6: 6%; types 9 and 10: none). This is consistent with the classifiers' accuracy of about 84% and explains why those types have no held-out detections.
* Because of this structure, density-based plausibility detectors behave differently depending on whether duplicates are kept when they
  are fitted, and a kNN detector with k = 5 reduces to "exact match with a benign training row". Read the plausibility results as
  protocol-dependent.

## Reproducing the results

### 1. Verify the statistics of the paper (about 20 seconds, no TensorFlow)

```bash
pip install numpy pandas scikit-learn scipy
cd extended_evaluation
python analyze.py --reuse       # recomputes every statistic from the shipped per-instance metrics -> results.json
python make_tables.py           # regenerates the four LaTeX tables of Section VI-D -> tables/
```

`--reuse` reproduces all 590 numbers of `results.json` exactly (checked under NumPy 1.26.4 and 2.4.4) and regenerates the paper's four
tables character for character.

### 2. Main experiments (`DiCE.ipynb`)

```powershell
# Windows PowerShell (Linux/macOS: python3.11 -m venv .venv && source .venv/bin/activate)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab DiCE.ipynb
```

Run the notebook from the repository root so that it finds the two CSV files. DiCE on the Random Forest is slow (24 and 33 minutes in the
paper). SHAP and TensorFlow 2.14 need incompatible NumPy versions, so run the SHAP cells (they `pip install shap`) in an environment built
from `requirements-shap.txt` and the Alibi cells in the environment above.

### 3. Rerun the extended evaluation from scratch (about 1-1.5 hours on one CPU core)

See [`extended_evaluation/README.md`](extended_evaluation/README.md) or run `extended_evaluation/run_all.sh` / `run_all.ps1`.

### Reproducibility notes

* **Retraining changes boundary-hugging counterfactuals.** Counterfactuals that sit right at the decision boundary (notably Growing
  Spheres on N-BaIoT) can flip class when the MLP is retrained in a different environment. In our check, re-scoring with a retrained MLP
  changed 36 of 590 reported quantities, all on N-BaIoT (for example Growing Spheres validity 1.00 to 0.88); the MEMS results reproduced
  exactly. The shipped `out/` files and `results.json` are the canonical results; use `--reuse` to verify them.
* Seeds: 42 for the split, models, DiCE and SHAP sampling unless stated in the paper (the plausibility IsolationForests use seed 0 except
  the N-BaIoT DiCE scorer).
* Growing Spheres and the FACE-style search in the extended evaluation are our own implementations, not reference code.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`). The manuscript details will be updated here when it is published.

## License

Code: MIT (see `LICENSE`). Data files: see the note in the *Data* section.
