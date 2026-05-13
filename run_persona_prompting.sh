#!/bin/bash
# End-to-end runner for the persona-prompting experiment.
# Usage (inside the conda env, e.g. via deploy.sh):
#   ./run_persona_prompting.sh
#
# Steps (each is idempotent — skip if the trained detector / outputs already exist):
#   1. Train the RepE probe (configs_all/repe.yaml, layer 22)
#   2. Train the Roleplaying probe (configs/my_roleplaying_layer22.yaml, layer 22)
#   3. Run the persona-prompting eval (189 dialogues, score with both probes)
#   4. Generate the 6 plots per probe

set -euo pipefail

cd "$(dirname "$(realpath "$0")")"

echo "============================================================"
echo "[1/4] Training RepE probe (configs_all/repe.yaml)"
echo "============================================================"
python -m deception_detection.scripts.experiment run \
    --config_file=deception_detection/scripts/configs_all/repe.yaml

echo
echo "============================================================"
echo "[2/4] Training Roleplaying probe (configs/my_roleplaying_layer22.yaml)"
echo "============================================================"
python -m deception_detection.scripts.experiment run \
    --config_file=deception_detection/scripts/configs/my_roleplaying_layer22.yaml

echo
echo "============================================================"
echo "[3/4] Running persona-prompting eval (scoring 189 dialogues with both probes)"
echo "============================================================"
python -m deception_detection.scripts.persona_prompting_experiment run

echo
echo "============================================================"
echo "[4/4] Generating plots"
echo "============================================================"
python -m deception_detection.scripts.persona_prompting_plots run

echo
echo "Done. Outputs under results/persona_prompting/"
ls -la results/persona_prompting/ || true
