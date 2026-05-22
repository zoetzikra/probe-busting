#!/bin/bash
# CPU + OpenAI-API prep for the data-sanitization study. NO GPU needed.
# Produces every cleaned dataset the GPU array job will train on, plus the
# label-preservation audit. Run this FIRST, on a node with internet + OPENAI_API_KEY.
#
#   1. Clean each poison at every guidance level         (Step 1)
#   2. Build cats/dogs deterministic clean-data ceiling  (Step 3)
#   3. Clean the clean-data ceiling at every level        (Step 4, collateral)
#   4. Label-preservation audit of all cleaned sets       (Step 5)
#
# Requires OPENAI_API_KEY in the environment or in .env. See note at the bottom.

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

# Pick a python with openai/fire (local login node has deception_env; cluster has probe-env).
PY="${PY:-python}"

if ! grep -q "OPENAI_API_KEY" .env 2>/dev/null && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: no OPENAI_API_KEY found in .env or environment."
  echo "Add a line  OPENAI_API_KEY=sk-...  to .env  (it is git-ignored), then re-run."
  exit 1
fi

echo "[1/4] Cleaning all poisons x guidance levels"
$PY -m deception_detection.sanitization clean_all

echo "[2/4] Building deterministic clean-data ceiling (cats/dogs)"
$PY -m deception_detection.sanitization make_clean_ceiling --poison cats_dogs

echo "[3/4] Cleaning the clean-data ceiling (collateral cells, cats/dogs)"
$PY -m deception_detection.sanitization make_collateral --poison cats_dogs

echo "[4/4] Label-preservation audit (gpt-4o judge)"
$PY -m deception_detection.label_audit audit_all

echo "Prep done. Cleaned data under data/<poison>/cleaned/. Now submit submit_sanitization.sbatch."
