"""Generate the six persona-prompting plots per probe.

Reads <out_dir>/<probe_name>/results.csv and writes:
    a_scores_all.png        - per-fact line plot, raw scores, 7 series
    b_scores_yes_only.png   - same as a, but only base + 3 yes conditions
    c_delta_all.png         - per-fact line plot, score - base, 6 series
    d_delta_yes_only.png    - same as c, but only 3 yes conditions
    e_category_bars_all.png - grouped bars, mean delta per (category, condition), 6 bars per category
    f_category_bars_yes_only.png - same as e, but only 3 yes conditions

Color convention follows the design doc:
    base         -> red
    naive_yes    -> dark green   | naive_no    -> light green
    detailed_yes -> dark blue    | detailed_no -> light blue
    dishonest_yes-> dark orange  | dishonest_no-> light orange

Usage:
    python -m deception_detection.scripts.persona_prompting_plots \\
        [--out_dir=results/persona_prompting]
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deception_detection.utils import REPO_ROOT


CATEGORY_ORDER = ["logical_impossibility", "conspiracy_theory", "contested_claim"]
CATEGORY_LABEL = {
    "logical_impossibility": "Logical Impossibilities",
    "conspiracy_theory": "Conspiracy Theories",
    "contested_claim": "Contested Claims",
}

CONDITION_ORDER_ALL = [
    "base",
    "naive_yes",
    "naive_no",
    "detailed_yes",
    "detailed_no",
    "dishonest_yes",
    "dishonest_no",
]
CONDITION_ORDER_YES = ["base", "naive_yes", "detailed_yes", "dishonest_yes"]
DELTA_ORDER_ALL = CONDITION_ORDER_ALL[1:]
DELTA_ORDER_YES = ["naive_yes", "detailed_yes", "dishonest_yes"]

CONDITION_COLOR = {
    "base": "#d62728",            # red
    "naive_yes": "#1b7837",       # dark green
    "naive_no": "#7fbc41",        # light green
    "detailed_yes": "#08306b",    # dark blue
    "detailed_no": "#6baed6",     # light blue
    "dishonest_yes": "#b35900",   # dark orange
    "dishonest_no": "#fdae6b",    # light orange
}

CONDITION_LABEL = {
    "base": "D_base",
    "naive_yes": "D_naive(yes)",
    "naive_no": "D_naive(no)",
    "detailed_yes": "D_detailed(yes)",
    "detailed_no": "D_detailed(no)",
    "dishonest_yes": "D_dishonest(yes)",
    "dishonest_no": "D_dishonest(no)",
}


def _pivot_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: row per fact_id, column per condition, value = score.

    Rows are ordered so that facts in CATEGORY_ORDER appear contiguously.
    """
    df = df.copy()
    df["_cat_order"] = df["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)})
    fact_order = (
        df.sort_values(["_cat_order", "fact_id"])
        .drop_duplicates("fact_id")["fact_id"]
        .tolist()
    )
    wide = df.pivot_table(
        index="fact_id", columns="condition", values="score", aggfunc="first"
    )
    wide = wide.reindex(fact_order)
    return wide


def _fact_category(df: pd.DataFrame) -> dict[int, str]:
    return dict(zip(df["fact_id"], df["category"]))


def _category_boundaries(facts_in_order: list[int], fact_cat: dict[int, str]):
    """Return (boundary_xs, label_centers, labels) for vertical separators between categories."""
    cats = [fact_cat[f] for f in facts_in_order]
    boundaries: list[int] = []
    label_centers: list[float] = []
    labels: list[str] = []
    start = 0
    for i in range(1, len(cats)):
        if cats[i] != cats[i - 1]:
            boundaries.append(i - 0.5)
            label_centers.append((start + i - 1) / 2)
            labels.append(CATEGORY_LABEL[cats[i - 1]])
            start = i
    label_centers.append((start + len(cats) - 1) / 2)
    labels.append(CATEGORY_LABEL[cats[-1]])
    return boundaries, label_centers, labels


def _line_plot(
    wide: pd.DataFrame,
    fact_cat: dict[int, str],
    conditions: list[str],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(wide.index))
    for cond in conditions:
        if cond not in wide.columns:
            continue
        ax.plot(
            x,
            wide[cond].values,
            label=CONDITION_LABEL[cond],
            color=CONDITION_COLOR[cond],
            marker="o",
            markersize=4,
            linewidth=1.4,
        )

    boundaries, centers, cat_labels = _category_boundaries(
        wide.index.tolist(), fact_cat
    )
    for b in boundaries:
        ax.axvline(b, color="grey", linestyle=":", alpha=0.5)
    # Place category labels in axes coordinates just above the plot so
    # bbox_inches="tight" doesn't blow up the figure height.
    for cx, lbl in zip(centers, cat_labels):
        ax.text(
            cx / max(len(wide.index) - 1, 1),
            1.02,
            lbl,
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            transform=ax.transAxes,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in wide.index], fontsize=7, rotation=0)
    ax.set_xlabel("Fact ID (grouped by category)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bar_plot(
    df: pd.DataFrame,
    conditions: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Grouped bars: x = category, group = condition, height = mean delta over facts."""
    summary_rows = []
    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat]
        # delta per fact = score - score(base)
        wide_sub = sub.pivot_table(
            index="fact_id", columns="condition", values="score", aggfunc="first"
        )
        for cond in conditions:
            if cond not in wide_sub.columns:
                continue
            deltas = wide_sub[cond] - wide_sub["base"]
            summary_rows.append(
                {
                    "category": cat,
                    "condition": cond,
                    "mean_delta": float(deltas.mean()),
                    "sem_delta": float(
                        deltas.std(ddof=1) / np.sqrt(len(deltas))
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)

    n_cats = len(CATEGORY_ORDER)
    n_conds = len(conditions)
    bar_w = 0.8 / n_conds
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, cond in enumerate(conditions):
        rows = [summary[(summary.category == c) & (summary.condition == cond)]
                for c in CATEGORY_ORDER]
        means = [
            r["mean_delta"].values[0] if len(r) else 0.0 for r in rows
        ]
        sems = [
            r["sem_delta"].values[0] if len(r) else 0.0 for r in rows
        ]
        xs = np.arange(n_cats) + (i - (n_conds - 1) / 2) * bar_w
        ax.bar(
            xs,
            means,
            bar_w,
            yerr=sems,
            color=CONDITION_COLOR[cond],
            label=CONDITION_LABEL[cond],
            capsize=3,
        )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(np.arange(n_cats))
    ax.set_xticklabels([CATEGORY_LABEL[c] for c in CATEGORY_ORDER])
    ax.set_ylabel("Mean Δ score vs D_base")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_plots_for_probe(results_csv: Path, probe_label: str) -> None:
    df = pd.read_csv(results_csv)
    fact_cat = _fact_category(df)
    wide_scores = _pivot_scores(df)

    # Deltas wide
    wide_delta = wide_scores.subtract(wide_scores["base"], axis=0)

    out_dir = results_csv.parent

    # (a) raw scores, all 7 conditions
    _line_plot(
        wide_scores,
        fact_cat,
        conditions=CONDITION_ORDER_ALL,
        ylabel="Deception score (mean over prefill tokens)",
        title=f"{probe_label}: raw scores, all conditions",
        out_path=out_dir / "a_scores_all.png",
    )

    # (b) raw scores, base + yes-only
    _line_plot(
        wide_scores,
        fact_cat,
        conditions=CONDITION_ORDER_YES,
        ylabel="Deception score (mean over prefill tokens)",
        title=f"{probe_label}: raw scores, yes-prefill conditions only",
        out_path=out_dir / "b_scores_yes_only.png",
    )

    # (c) deltas, all 6 non-baseline conditions
    _line_plot(
        wide_delta,
        fact_cat,
        conditions=DELTA_ORDER_ALL,
        ylabel="Δ deception score (vs D_base)",
        title=f"{probe_label}: Δ from baseline, all conditions",
        out_path=out_dir / "c_delta_all.png",
    )

    # (d) deltas, yes-only (3 conditions)
    _line_plot(
        wide_delta,
        fact_cat,
        conditions=DELTA_ORDER_YES,
        ylabel="Δ deception score (vs D_base)",
        title=f"{probe_label}: Δ from baseline, yes-prefill only",
        out_path=out_dir / "d_delta_yes_only.png",
    )

    # (e) grouped bars per category, all 6
    _bar_plot(
        df,
        conditions=DELTA_ORDER_ALL,
        title=f"{probe_label}: mean Δ by category, all conditions",
        out_path=out_dir / "e_category_bars_all.png",
    )

    # (f) grouped bars per category, yes-only
    _bar_plot(
        df,
        conditions=DELTA_ORDER_YES,
        title=f"{probe_label}: mean Δ by category, yes-prefill only",
        out_path=out_dir / "f_category_bars_yes_only.png",
    )

    print(f"[{probe_label}] wrote 6 plots to {out_dir}")


def run(out_dir: str | None = None):
    """Generate plots for each probe under <out_dir>/<probe_name>/results.csv."""
    if out_dir is None:
        root = REPO_ROOT / "results" / "persona_prompting"
    else:
        root = Path(out_dir)
    assert root.exists(), f"Output dir {root} does not exist"

    any_found = False
    for probe_dir in sorted(root.iterdir()):
        if not probe_dir.is_dir():
            continue
        results_csv = probe_dir / "results.csv"
        if not results_csv.exists():
            continue
        any_found = True
        _make_plots_for_probe(results_csv, probe_label=probe_dir.name)

    if not any_found:
        raise FileNotFoundError(
            f"No <probe>/results.csv under {root}. "
            "Run persona_prompting_experiment.py first."
        )


if __name__ == "__main__":
    fire.Fire({"run": run})
