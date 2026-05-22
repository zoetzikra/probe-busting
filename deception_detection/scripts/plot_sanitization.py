"""Aggregate the sanitization experiment matrix and plot it with bootstrap error bars.

Reads every results/san_* run: its cfg.yaml (to recover poison + condition) and its
bootstrap_ci.csv (produced by bootstrap_ci.py). Emits:

  results/sanitization/summary.csv         tidy table: poison, condition, dataset, metric, point, ci
  results/sanitization/<poison>_test_auroc.png   Step 1 headline: test_dist AUROC by guidance
                                                  level, CI bars, clean-ceiling line (Step 3)
  results/sanitization/<poison>_regression.png    Step 6: auroc vs auroc_alpaca per condition
  results/sanitization/cats_dogs_collateral.png    Step 4: clean-data probe with/without cleaning

Run AFTER bootstrap_ci.py:
    python -m deception_detection.scripts.plot_sanitization run --results_glob "results/san_*"
"""

from __future__ import annotations

import glob
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from deception_detection.utils import REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "sanitization"

BASE_TO_POISON = {"repe_dog_cat": "cats_dogs", "confidence": "confidence", "tense": "tense"}

# display order + labels for the Step-1 / Step-3 conditions
STEP1_ORDER = [
    "training_dist",
    "training_dist_minimal",
    "training_dist_medium",
    "training_dist_maximal",
    "training_dist_diagnosed",
]
COLLATERAL_ORDER = ["clean_ceiling", "clean_minimal", "clean_medium", "clean_maximal"]
COND_LABEL = {
    "training_dist": "Poisoned\nbaseline",
    "training_dist_minimal": "Minimal",
    "training_dist_medium": "Medium",
    "training_dist_maximal": "Maximal",
    "training_dist_diagnosed": "Diagnosed\n(Idea 1)",
    "clean_ceiling": "Clean\n(no clean)",
    "clean_minimal": "Clean\n+Minimal",
    "clean_medium": "Clean\n+Medium",
    "clean_maximal": "Clean\n+Maximal",
}
COND_COLOR = {
    "training_dist": "#9e9e9e",
    "training_dist_minimal": "#90caf9",
    "training_dist_medium": "#42a5f5",
    "training_dist_maximal": "#1565c0",
    "training_dist_diagnosed": "#ab47bc",
    "clean_ceiling": "#a5d6a7",
    "clean_minimal": "#66bb6a",
    "clean_medium": "#43a047",
    "clean_maximal": "#2e7d32",
}


def _parse_run(folder: Path) -> tuple[str, str, str] | None:
    """Return (poison, condition, test_dist_name) from a run's cfg.yaml, or None."""
    cfg_path = folder / "cfg.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    train = cfg.get("train_data", "")
    if "__" not in train:
        return None
    base, _, variant = train.partition("__")
    if base not in BASE_TO_POISON:
        return None
    test_name = f"{base}__test_dist"
    return BASE_TO_POISON[base], variant, test_name


def aggregate(results_glob: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(REPO_ROOT / results_glob))):
        folder = Path(p)
        ci_path = folder / "bootstrap_ci.csv"
        parsed = _parse_run(folder)
        if parsed is None or not ci_path.exists():
            continue
        poison, condition, test_name = parsed
        ci = pd.read_csv(ci_path)
        for _, r in ci.iterrows():
            # tag the dataset as either the held-out test_dist or the in-dist val split
            ds = r["dataset"]
            kind = "test_dist" if ds == test_name else ("val" if ds.endswith("_val") else ds)
            rows.append(
                {
                    "poison": poison,
                    "condition": condition,
                    "dataset_kind": kind,
                    "metric": r["metric"],
                    "point": r["point"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "std": r["std"],
                }
            )
    return pd.DataFrame(rows)


def _yerr(sub: pd.DataFrame) -> np.ndarray:
    return np.array([
        (sub["point"] - sub["ci_low"]).to_numpy(),
        (sub["ci_high"] - sub["point"]).to_numpy(),
    ])


def _bar(ax, sub: pd.DataFrame, order: list[str], title: str, ylabel: str):
    sub = sub.set_index("condition").reindex([c for c in order if c in sub["condition"].values])
    sub = sub.reset_index().dropna(subset=["point"])
    x = np.arange(len(sub))
    ax.bar(
        x, sub["point"], yerr=_yerr(sub),
        color=[COND_COLOR.get(c, "#777") for c in sub["condition"]],
        capsize=4, edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL.get(c, c) for c in sub["condition"]], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.5, color="red", ls=":", lw=1, label="chance (0.5)")
    ax.grid(axis="y", alpha=0.3)


def plot_test_auroc(df: pd.DataFrame, poison: str):
    """Step 1 headline + Step 3 ceiling line: test_dist AUROC by guidance level."""
    sub = df[(df.poison == poison) & (df.dataset_kind == "test_dist") & (df.metric == "auroc")]
    step1 = sub[sub.condition.isin(STEP1_ORDER)]
    if step1.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _bar(ax, step1, STEP1_ORDER, f"{poison}: test_dist AUROC by guidance level",
         "AUROC (test_dist)")

    # clean-data ceiling as a horizontal band (point + CI), if present
    ceil = sub[sub.condition == "clean_ceiling"]
    if not ceil.empty:
        c = ceil.iloc[0]
        ax.axhline(c["point"], color="green", ls="--", lw=1.5, label="clean-data ceiling")
        ax.axhspan(c["ci_low"], c["ci_high"], color="green", alpha=0.10)
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / f"{poison}_test_auroc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_regression(df: pd.DataFrame, poison: str):
    """Step 6: put auroc and auroc_alpaca side by side per condition to surface whether
    cleaning recovers discrimination while degrading the alpaca-referenced metric."""
    sub = df[(df.poison == poison) & (df.dataset_kind == "test_dist")]
    sub = sub[sub.condition.isin(STEP1_ORDER)]
    if sub.empty:
        return
    metrics = ["auroc", "auroc_alpaca"]
    conds = [c for c in STEP1_ORDER if c in sub["condition"].values]
    x = np.arange(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, m in enumerate(metrics):
        ms = sub[sub.metric == m].set_index("condition").reindex(conds).reset_index()
        ax.bar(
            x + (i - 0.5) * w, ms["point"], w, yerr=_yerr(ms),
            capsize=3, label=m,
            color=("#1565c0" if m == "auroc" else "#ef9a9a"), edgecolor="black", linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL.get(c, c) for c in conds], fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_title(f"{poison}: auroc vs auroc_alpaca on test_dist (Step 6 regression check)")
    ax.axhline(0.5, color="red", ls=":", lw=1)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / f"{poison}_regression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_collateral(df: pd.DataFrame, poison: str = "cats_dogs"):
    """Step 4: does cleaning a *clean* dataset hurt the probe? Compare clean_ceiling (no
    cleaning) to clean_{minimal,medium,maximal} on the test_dist."""
    sub = df[(df.poison == poison) & (df.dataset_kind == "test_dist") & (df.metric == "auroc")]
    sub = sub[sub.condition.isin(COLLATERAL_ORDER)]
    if sub.empty or len(sub) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _bar(ax, sub, COLLATERAL_ORDER, f"{poison}: collateral damage of cleaning clean data",
         "AUROC (test_dist)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / f"{poison}_collateral.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def run(results_glob: str = "results/san_*") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = aggregate(results_glob)
    if df.empty:
        print(f"No runs matched {results_glob} with cfg.yaml + bootstrap_ci.csv. "
              "Run the experiments and bootstrap_ci.py first.")
        return
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'summary.csv'} ({len(df)} rows)")
    for poison in sorted(df.poison.unique()):
        plot_test_auroc(df, poison)
        plot_regression(df, poison)
        plot_collateral(df, poison)


if __name__ == "__main__":
    fire.Fire({"run": run})
