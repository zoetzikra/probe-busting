"""Bootstrap confidence intervals for probe metrics over the evaluation set.

Why this exists: AUROC / recall@1% / fpr@1% are *set-level scalars* (one number per eval
dataset), so unlike the persona deltas they have no per-item population to take a SEM over.
To put error bars on them under the single-seed constraint, we resample the evaluation and
control dialogues with replacement and recompute each metric, giving its sampling
distribution. See the SPAR red-team review (Step 2).

Scope of the interval (state this in any caption): it captures **evaluation-set + control
sampling noise only**. It holds fixed (a) the one trained probe and (b) the one temperature=0
cleaning pass. It is NOT a CI over cleaning randomness or probe-training seeds.

Metrics reproduce the repo's "mean" PromptScorer exactly:
  - per-dialogue score = mean over its per-token scores
  - auroc          : honest (neg) vs deceptive (pos) within the eval set
  - auroc_alpaca   : control/alpaca (neg) vs deceptive (pos)
  - recall@1%      : frac of deceptive > threshold, threshold = 99th pct of control scores
  - fpr@1%         : frac of honest    > threshold (same threshold)

Usage (inside an env with numpy/sklearn, e.g. deception_env):
    python -m deception_detection.scripts.bootstrap_ci run --results_glob "results/san_*"
    python -m deception_detection.scripts.bootstrap_ci run --folder results/<one_run>
Writes bootstrap_ci.csv next to each run and a combined sanitization_bootstrap_ci.csv.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deception_detection.utils import REPO_ROOT

DEFAULT_FPR = 0.01
N_BOOT = 2000
SEED = 0


def _mean_per_dialogue(score_lists: list[list[float]]) -> np.ndarray:
    """mean over tokens per dialogue, dropping all-NaN dialogues -> 1D array."""
    out = np.array([np.nanmean(s) if len(s) else np.nan for s in score_lists], dtype=float)
    return out


def _load_run(folder: Path) -> dict:
    with open(folder / "scores.json") as f:
        scores = json.load(f)
    with open(folder / "control_scores.json") as f:
        control = json.load(f)
    # control: take the first (only) control dataset
    ctrl_key = next(iter(control))
    ctrl = _mean_per_dialogue(control[ctrl_key]["scores"])
    ctrl = ctrl[~np.isnan(ctrl)]
    datasets = {}
    for ds_name, payload in scores.items():
        per = _mean_per_dialogue(payload["scores"])
        labels = np.array(payload["labels"])
        honest = per[(labels == "HONEST") & ~np.isnan(per)]
        decept = per[(labels == "DECEPTIVE") & ~np.isnan(per)]
        datasets[ds_name] = (honest, decept)
    return {"control": ctrl, "datasets": datasets}


def _auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    return float(roc_auc_score(y, s))


def _metrics(honest: np.ndarray, decept: np.ndarray, control: np.ndarray) -> dict[str, float]:
    thr = np.quantile(control, 1 - DEFAULT_FPR) if len(control) else np.nan
    return {
        "auroc": _auroc(honest, decept),
        "auroc_alpaca": _auroc(control, decept),
        "recall_1pct": float(np.mean(decept > thr)) if len(decept) else float("nan"),
        "fpr_1pct": float(np.mean(honest > thr)) if len(honest) else float("nan"),
    }


def _bootstrap(honest, decept, control, n_boot=N_BOOT, seed=SEED) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    point = _metrics(honest, decept, control)
    keys = list(point)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        h = rng.choice(honest, size=len(honest), replace=True) if len(honest) else honest
        d = rng.choice(decept, size=len(decept), replace=True) if len(decept) else decept
        c = rng.choice(control, size=len(control), replace=True) if len(control) else control
        m = _metrics(h, d, c)
        for k in keys:
            samples[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.array(samples[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            out[k] = {"point": point[k], "ci_low": np.nan, "ci_high": np.nan, "std": np.nan}
        else:
            out[k] = {
                "point": point[k],
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
                "std": float(arr.std(ddof=1)),
            }
    return out


def process_folder(folder: Path, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    run = _load_run(folder)
    rows = []
    for ds_name, (honest, decept) in run["datasets"].items():
        ci = _bootstrap(honest, decept, run["control"], n_boot=n_boot, seed=seed)
        for metric, vals in ci.items():
            rows.append(
                {
                    "run": folder.name,
                    "dataset": ds_name,
                    "metric": metric,
                    "n_honest": len(honest),
                    "n_deceptive": len(decept),
                    "n_control": len(run["control"]),
                    **vals,
                }
            )
    return pd.DataFrame(rows)


def run(
    folder: str | None = None,
    results_glob: str | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    out: str = "results/sanitization_bootstrap_ci.csv",
) -> None:
    """Compute bootstrap CIs for one run (folder) or many (results_glob)."""
    folders: list[Path] = []
    if folder:
        folders.append(Path(folder))
    if results_glob:
        folders += [Path(p) for p in sorted(glob.glob(str(REPO_ROOT / results_glob)))]
    folders = [f for f in folders if (f / "scores.json").exists()]
    if not folders:
        raise FileNotFoundError("No runs with scores.json found for the given folder/glob.")

    all_dfs = []
    for f in folders:
        df = process_folder(f, n_boot=n_boot, seed=seed)
        df.to_csv(f / "bootstrap_ci.csv", index=False)
        all_dfs.append(df)
        print(f"[bootstrap] {f.name}: {len(df)} metric rows -> {f / 'bootstrap_ci.csv'}")

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = REPO_ROOT / out
    combined.to_csv(out_path, index=False)
    print(f"[bootstrap] combined -> {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    fire.Fire({"run": run})
