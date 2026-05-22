#!/bin/bash
# GPU array job: trains+evaluates one sanitization config per array task, in parallel.
# Each task loads Llama-3.3-70B once and runs `experiment.py run` for its config.
#
# Prerequisite: run ./run_sanitization_prep.sh first so the cleaned datasets exist.
#
# Submit with the array sized to the number of configs (16):
#   sbatch --array=0-15 submit_sanitization.sbatch
# Then, after it finishes, run postprocessing:
#   ./run_sanitization_postprocess.sh
#
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=san_matrix
#SBATCH --output=logs/san_matrix_%A_%a.out
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

# Deterministic, sorted list of configs -> index by array task id.
mapfile -t CONFIGS < <(ls deception_detection/scripts/configs/sanitization/*.yaml | sort)
N=${#CONFIGS[@]}
echo "Found $N configs"

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "$N" ]; then
    echo "Task index $IDX >= $N configs; nothing to do."
    exit 0
fi

CFG="${CONFIGS[$IDX]}"
echo "=== [$IDX/$N] Running $CFG ==="
python -m deception_detection.scripts.experiment run --config_file="$CFG"

echo "Task $IDX finished at $(date)"
