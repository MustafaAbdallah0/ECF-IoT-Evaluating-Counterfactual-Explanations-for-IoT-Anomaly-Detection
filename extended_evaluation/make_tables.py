import json, os
os.makedirs('tables', exist_ok=True)
R = json.load(open('./results.json'))
M = ["dice_mlp", "dice_mlp_free", "alibi_mlp", "gs_mlp", "face_mlp"]
NAME = {"dice_mlp": "DiCE (restricted)", "dice_mlp_free": "DiCE (all features)", "alibi_mlp": "Alibi-Explain", "gs_mlp": "Growing Spheres", "face_mlp": "FACE-style graph"}
def ci(t, d=2, s=1.0): return f"{t[0]*s:.{d}f} [{t[1]*s:.{d}f}, {t[2]*s:.{d}f}]"
def pct(x, d=1): return f"{100*x:.{d}f}"

# ---- Table A: shared-classifier main results (table*)
rows = []
for me in M:
    a, b = R["mems"]["methods"][me], R["d9"]["methods"][me]
    rows.append(f"{NAME[me]} & {ci(a['validity'])} & {ci(a['L1'],3)} & {ci(a['spars'],1,100)} & {ci(b['validity'])} & {ci(b['L1'],3)} & {ci(b['spars'],1,100)} \\\\")
tabA = r"""\begin{table*}[t]
\centering
\caption{Extended evaluation on a shared MLP classifier: held-out instances predicted as anomalous (MEMS: $n=100$; N-BaIoT: $n=80$, attack types 2 and 5--11). Entries are instance-level means with 95\% bootstrap confidence intervals (10{,}000 resamples). Validity is the fraction of the three requested counterfactuals that flip the prediction; proximity (L1) and sparsity (\% of features changed) are averaged over valid counterfactuals.}
\label{tab:ext-main}
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l ccc ccc@{}}
\toprule
 & \multicolumn{3}{c}{\textbf{MEMS}} & \multicolumn{3}{c}{\textbf{N-BaIoT}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Method} & Validity & Proximity (L1) & Sparsity (\%) & Validity & Proximity (L1) & Sparsity (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open('tables/tab_ext_main.tex', 'w').write(tabA)

# ---- Table B: plausibility under three detectors
def prow(label, a, b): return f"{label} & " + " & ".join(a + b) + r" \\"
rows = []
for me in M:
    a, b = R["mems"]["methods"][me], R["d9"]["methods"][me]
    rows.append(prow(NAME[me], [pct(a["pl_IF"][0]), pct(a["pl_LOF"][0]), pct(a["pl_kNN"][0])], [pct(b["pl_IF"][0]), pct(b["pl_LOF"][0]), pct(b["pl_kNN"][0])]))
bo_a, bo_b = R["mems"]["baseline_original"], R["d9"]["baseline_original"]; bh_a, bh_b = R["mems"]["baseline_heldout_benign"], R["d9"]["baseline_heldout_benign"]
base = [prow("Original anomalies (baseline)", [pct(bo_a[k]) for k in ["IF", "LOF", "kNN"]], [pct(bo_b[k]) for k in ["IF", "LOF", "kNN"]]),
        prow("Held-out benign (reference)", [pct(bh_a[k]) for k in ["IF", "LOF", "kNN"]], [pct(bh_b[k]) for k in ["IF", "LOF", "kNN"]])]
tabB = r"""\begin{table}[t]
\centering
\caption{Plausibility of valid counterfactuals under three detectors (\% accepted; IF: IsolationForest, LOF: Local Outlier Factor, kNN: distance to the fifth nearest benign neighbor). Each detector is fitted on benign training rows and its 5\% rejection threshold is calibrated on held-out benign rows. The last two rows show the acceptance rate of the original anomalous instances and of held-out benign rows.}
\label{tab:ext-plaus}
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}l ccc ccc@{}}
\toprule
 & \multicolumn{3}{c}{\textbf{MEMS}} & \multicolumn{3}{c}{\textbf{N-BaIoT}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Method} & IF & LOF & kNN & IF & LOF & kNN \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + "\n".join(base) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open('tables/tab_ext_plaus.tex', 'w').write(tabB)

# ---- Table C: paired tests DiCE (restricted) vs Alibi, shared MLP (table*)
lab = [("validity", "Validity", 1, 2), ("L1", "Proximity (L1)", 1, 3), ("spars", "Sparsity (pct. points)", 100, 1),
       ("vp_IF", "Valid and plausible (IF)", 1, 2), ("vp_LOF", "Valid and plausible (LOF)", 1, 2), ("vp_kNN", "Valid and plausible (kNN)", 1, 2)]
def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"
rows = []
for k, nm, sc, dg in lab:
    ta, tb = R["mems"]["paired_dice_vs_alibi"][k], R["d9"]["paired_dice_vs_alibi"][k]
    f_ = lambda t: f"{t['mean_diff']*sc:+.{dg}f} [{t['ci'][0]*sc:+.{dg}f}, {t['ci'][1]*sc:+.{dg}f}]"
    rows.append(f"{nm} & {f_(ta)} & {pfmt(ta['p_holm_six'])} & {f_(tb)} & {pfmt(tb['p_holm_six'])} \\\\")
tabC = r"""\begin{table*}[t]
\centering
\caption{Paired comparison of DiCE (restricted features) and Alibi-Explain on the same instances and the same MLP. Entries are the mean difference (DiCE $-$ Alibi-Explain) with a 95\% bootstrap confidence interval, and the Holm-adjusted $p$-value of a two-sided Wilcoxon signed-rank test over six comparisons per dataset. ``Valid and plausible'' is the fraction of the three requested counterfactuals that are both valid and accepted by the detector. Proximity and sparsity use only instances for which both methods returned a valid counterfactual (MEMS: 96; N-BaIoT: 66).}
\label{tab:ext-tests}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}l cc cc@{}}
\toprule
 & \multicolumn{2}{c}{\textbf{MEMS} ($n=100$)} & \multicolumn{2}{c}{\textbf{N-BaIoT} ($n=80$)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\textbf{Metric} & Difference [95\% CI] & $p_{\mathrm{Holm}}$ & Difference [95\% CI] & $p_{\mathrm{Holm}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open('tables/tab_ext_tests.tex', 'w').write(tabC)

# ---- Table D: classifier effect (N-BaIoT, same 45 instances)
R9 = R["d9"]; ce = R9["classifier_effect_dice"]
dr, dm = R9["methods"]["dice_rf"], R9["methods"]["dice_mlp"]
tabD = r"""\begin{table}[t]
\centering
\caption{Effect of the classifier on DiCE (N-BaIoT, the same 45 held-out instances, stratified over attack types). The Random Forest is the classifier used with DiCE in the main experiments; the MLP is the classifier used with Alibi-Explain. Validity is the fraction of three requested counterfactuals that flip the prediction (95\% bootstrap confidence interval).}
\label{tab:ext-clf}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}l c@{}}
\toprule
\textbf{Method / classifier} & \textbf{Validity [95\% CI]} \\
\midrule
DiCE on Random Forest & """ + ci(ce["dice_rf_validity_on_common"]) + r""" \\
DiCE on MLP & """ + ci(ce["dice_mlp_validity_on_common"]) + r""" \\
Alibi-Explain on MLP & """ + ci(ce["alibi_validity_on_common"]) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
open('tables/tab_ext_clf.tex', 'w').write(tabD)
print("tables written\n"); print(tabA.split("\\midrule")[1].split("\\bottomrule")[0]); print(tabB.split("\\midrule")[1].split("\\bottomrule")[0]); print(tabC.split("\\midrule")[1].split("\\bottomrule")[0]); print(tabD.split("\\midrule")[1].split("\\bottomrule")[0])
