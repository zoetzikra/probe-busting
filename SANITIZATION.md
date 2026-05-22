# Data-Sanitization Blue-Team Study (re-run from scratch)

This is the systematic, version-controlled re-implementation of the "Universal Data
Sanitization" experiment (final report Section 2.4.1 / 3.3.1 / Appendix A), rebuilt to
address the red-team review of the original setup. It implements Steps 1–5 of that review.

## What changed vs. the original report

| # | Red-team gap | What this adds |
|---|---|---|
| 1 | Only cats/dogs evaluated | All three poisons run end-to-end: **cats/dogs, confidence, tense** |
| 2 | Bare AUROC bars (no error bars) | **Bootstrap CIs** over the eval set for auroc / auroc_alpaca / recall@1% / fpr@1% |
| 3 | No clean-data ceiling | **clean_ceiling** cell: probe trained on never-poisoned data, eval on test_dist |
| 4 | No collateral-damage control | **clean_{minimal,medium,maximal}**: cleaner run over *clean* data to isolate rephrasing damage |
| 5 | Label preservation only asserted | **Label-preservation audit**: a separate gpt-4o judge measures the flip rate |

## Files

- `deception_detection/sanitization.py` — cleaning pipeline (prompts verbatim from Appendix A),
  GPT-4o-mini, `temperature=0`, both dataset formats. Builds cleaned / ceiling / collateral data.
- `deception_detection/label_audit.py` — gpt-4o truth-value judge → flip-rate per cell.
- `deception_detection/scripts/gen_sanitization_configs.py` — emits the 16 experiment configs
  into `deception_detection/scripts/configs/sanitization/`.
- `deception_detection/scripts/bootstrap_ci.py` — bootstrap CIs over each run's `scores.json`.
- `deception_detection/scripts/plot_sanitization.py` — aggregates the matrix + plots with CIs.
- Dataset variants added to `data/{repe_cats_dogs,confidence,tense}.py` (training_dist_{level},
  clean_ceiling, clean_{level}).

## The experiment matrix (16 cells)

Every cell trains a layer-22 LR probe and evaluates on the fixed reversed-correlation
`test_dist` (+ the auto `_val` split). The cleaning only ever touches **training** data.

- **cats/dogs (8):** training_dist (baseline) · training_dist_{minimal,medium,maximal} ·
  clean_ceiling · clean_{minimal,medium,maximal}
- **confidence (4):** training_dist · training_dist_{minimal,medium,maximal}
- **tense (4):** training_dist · training_dist_{minimal,medium,maximal}

Confidence/tense have no deterministic never-poisoned source, so their matched ceiling is
the maximal-guidance cleaned set — documented, not run as a separate cell.

## How to run

**Prereqs:** an `OPENAI_API_KEY` line in `.env` (git-ignored) for the cleaning + audit steps;
SLURM GPU access for the 70B activations.

```bash
# 1. CPU + OpenAI (no GPU) — produces cleaned data + label audit. ~minutes.
#    On the login node:  PY=~/.conda/envs/deception_env/bin/python ./run_sanitization_prep.sh
./run_sanitization_prep.sh

# 2. GPU array job — one config per task, in parallel (16 configs).
sbatch --array=0-15 submit_sanitization.sbatch

# 3. CPU postprocessing — bootstrap CIs + plots. After the array finishes.
./run_sanitization_postprocess.sh
```

Outputs land in `results/sanitization/`: `summary.csv`, `<poison>_test_auroc.png`
(Step 1 + ceiling), `<poison>_regression.png` (Step 6 auroc vs auroc_alpaca),
`cats_dogs_collateral.png` (Step 4), plus `results/label_audit_summary.csv`.

## Scope caveats to state in the writeup

- **Single seed.** Cleaning uses `temperature=0` and one probe-training run. The bootstrap CIs
  therefore capture **evaluation-set + control sampling noise only** — not cleaning randomness
  or probe-training-seed variance. Say this in the figure captions.
- **Guidance levels co-vary with prompt length / the truth-preservation clause** (accepted as
  inevitable). Treat "guidance level" as a single specificity axis, not a clean one-factor knob.
- **"Universal"** describes only the minimal prompt (shared across poisons); medium and maximal
  are attack-flavoured.
