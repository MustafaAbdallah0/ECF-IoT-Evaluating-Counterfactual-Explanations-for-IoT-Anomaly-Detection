#!/usr/bin/env bash
# Regenerates all counterfactuals and statistics of the extended evaluation (about 1-1.5 hours on one CPU core).
set -euo pipefail
cd "$(dirname "$0")"
python select_instances.py            # shap_top10.py needs shap (numpy>=2): run it in the requirements-shap.txt environment, or keep the shipped shap_top10.json
for ds in mems d9; do
  for m in dice_mlp alibi_mlp gs_mlp face_mlp dice_mlp_free; do python methods.py "$ds" "$m"; done
done
python methods.py d9 dice_rf 45       # slow; the paper uses the first 45 of the 80 N-BaIoT instances
python analyze.py
python make_tables.py
