#!/bin/bash
# CPU postprocessing for the data-sanitization study. NO GPU needed.
# Run AFTER the GPU array job (submit_sanitization.sbatch) has finished.
#
#   1. Bootstrap CIs over each run's saved scores      (Step 2)
#   2. Aggregate the matrix + make plots with error bars
#
# Output: results/sanitization/{summary.csv, *_test_auroc.png, *_regression.png,
#         cats_dogs_collateral.png} and results/sanitization_bootstrap_ci.csv.

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

PY="${PY:-python}"

echo "[1/2] Bootstrap confidence intervals over results/san_*"
$PY -m deception_detection.scripts.bootstrap_ci run --results_glob "results/san_*"

echo "[2/2] Aggregating + plotting"
$PY -m deception_detection.scripts.plot_sanitization run --results_glob "results/san_*"

echo "Postprocessing done. See results/sanitization/."
