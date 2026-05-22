#!/bin/bash
# CPU postprocessing job (bootstrap CIs + plots). No GPU, no API.
# Submit with an afterok dependency on the GPU array so it runs automatically:
#   sbatch --dependency=afterok:<array_job_id> submit_sanitization_postprocess.sbatch
#
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=8
#SBATCH --job-name=san_post
#SBATCH --output=logs/san_post_%j.out
#SBATCH --time=00:30:00
#SBATCH --mem=16G

set -euo pipefail

echo "Postprocess started at $(date) on $(hostname)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate probe-env

cd /gpfs/home6/ztzifa/probe-busting
./run_sanitization_postprocess.sh

echo "Postprocess finished at $(date)"
