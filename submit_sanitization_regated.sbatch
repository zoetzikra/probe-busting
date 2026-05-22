#!/bin/bash
# GPU array for the 6 cells re-cleaned with the label-preservation gate (gpt-4.1-mini).
# These had first-run flip rates >5% and were re-cleaned gated; their stale ungated result
# folders were deleted. Submit:  sbatch --array=0-5 submit_sanitization_regated.sbatch
#
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=san_regate
#SBATCH --output=logs/san_regate_%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --hint=nomultithread

set -euo pipefail

echo "Job ${SLURM_ARRAY_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} on $(hostname) at $(date)"
nvidia-smi || true

module purge
module load 2023
module load CUDA/12.1.1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate probe-env

export HF_HOME=/scratch-shared/ztzifa/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch-shared/ztzifa/hf_cache/hub
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat $HOME/.cache/huggingface/token)"
fi

cd /gpfs/home6/ztzifa/probe-busting

CFG_DIR=deception_detection/scripts/configs/sanitization
CONFIGS=(
  "$CFG_DIR/cats_dogs__training_dist_minimal.yaml"
  "$CFG_DIR/confidence__training_dist_minimal.yaml"
  "$CFG_DIR/confidence__training_dist_medium.yaml"
  "$CFG_DIR/confidence__training_dist_maximal.yaml"
  "$CFG_DIR/tense__training_dist_minimal.yaml"
  "$CFG_DIR/tense__training_dist_diagnosed.yaml"
)
IDX=${SLURM_ARRAY_TASK_ID:-0}
N=${#CONFIGS[@]}
[ "$IDX" -ge "$N" ] && { echo "idx $IDX >= $N; nothing to do"; exit 0; }

CFG="${CONFIGS[$IDX]}"
echo "=== [$IDX/$N] Running $CFG ==="
python -m deception_detection.scripts.experiment run --config_file="$CFG"
echo "Task $IDX finished at $(date)"
