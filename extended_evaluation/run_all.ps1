# Regenerates all counterfactuals and statistics of the extended evaluation (about 1-1.5 hours on one CPU core).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python select_instances.py            # shap_top10.py needs shap (numpy>=2): run it in the requirements-shap.txt environment, or keep the shipped shap_top10.json
foreach ($ds in "mems", "d9") {
  foreach ($m in "dice_mlp", "alibi_mlp", "gs_mlp", "face_mlp", "dice_mlp_free") { python methods.py $ds $m }
}
python methods.py d9 dice_rf 45       # slow; the paper uses the first 45 of the 80 N-BaIoT instances
python analyze.py
python make_tables.py
