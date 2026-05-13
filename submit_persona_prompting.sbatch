#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=persona_prompting
#SBATCH --output=logs/persona_prompting_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --hint=nomultithread

# --mem=180G note: Llama-3.3-70B is ~140 GB in fp16. cut_at_layer=22 only takes
# effect AFTER from_pretrained loads the full model, so we still need enough
# CPU+GPU memory to stage the full 140 GB during load. 180 GB is the default
# per-GPU allocation on gpu_h100 (see `scontrol show partition gpu_h100`).

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)"
nvidia-smi || true

# Snellius module stack
module purge
module load 2023
module load CUDA/12.1.1

# Activate our miniconda env (created earlier under ~/miniconda3)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate probe-env

# Redirect HuggingFace cache to scratch (home only has ~78 GB free;
# Llama-3.3-70B is ~140 GB). The token at ~/.cache/huggingface/token is
# still picked up via the HF_TOKEN env var.
export HF_HOME=/scratch-shared/ztzifa/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch-shared/ztzifa/hf_cache/hub
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat $HOME/.cache/huggingface/token)"
fi

# Sanity: verify env
echo "Python: $(which python) $(python --version)"
python -c "import torch; print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

cd /gpfs/home6/ztzifa/probe-busting

# Run the end-to-end pipeline:
#   1. Train RepE probe   (configs_all/repe.yaml)
#   2. Train Roleplaying probe at layer 22 (configs/my_roleplaying_layer22.yaml)
#   3. Persona-prompting eval (189 dialogues, both probes)
#   4. Generate the 6 plots per probe
./run_persona_prompting.sh

echo "Job finished at: $(date)"
