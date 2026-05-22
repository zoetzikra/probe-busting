"""Cleaning-fidelity audit: how often did the cleaner change a sentence's claim?

This is the *measurement* counterpart to the corrective gate in sanitization.py. For each
statement it compares the CLEANED rewrite to the ORIGINAL sentence with an independent model
and asks "SAME claim, or CHANGED?" — exactly the gate's question, but (a) run after gating as
an independent check, and (b) with a stronger model than the gate's.

Crucially it does NOT judge world-truth and NEVER consults the dataset's labels — so it cannot
"disagree with the dataset's ground truth." It only measures cleaning-induced corruption:
the fraction of sentences whose central claim the cleaning altered (negation, number/entity
change, etc.). Reverted sentences are identical to the original, so they count as SAME.

Independence: the gate uses gpt-4o-mini; this audit uses gpt-4o (stronger, different) so a
residual >0 is meaningful (it catches changes the gate's weaker model missed). Equivalence is
a near-objective textual question, so this is not the circular "same judge" problem the old
truth-value audit had.

Usage:
    python -m deception_detection.label_audit audit --poison cats_dogs --level minimal
    python -m deception_detection.label_audit audit_all
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
import pandas as pd
from tqdm import tqdm

from deception_detection.sanitization import (
    LEVELS,
    POISONS,
    _make_client,
    _read_json,
    claim_preserved,
    cleaned_dir,
    cleaned_train_path,
)
from deception_detection.utils import REPO_ROOT

AUDIT_MODEL = "gpt-4o"  # independent + stronger than the gpt-4o-mini gate


def _load_original_and_cleaned(poison: str, level: str) -> tuple[list[str], list[str]]:
    """Return (original_statements, cleaned_statements) aligned 1:1 in the same order."""
    spec = POISONS[poison]
    orig = _read_json(spec.train_path)
    cleaned_path = (
        cleaned_dir(poison) / "diagnosed_training_dist.json"
        if level == "diagnosed"
        else cleaned_train_path(poison, level)
    )
    if not cleaned_path.exists():
        raise FileNotFoundError(f"{cleaned_path} missing; run cleaning first.")
    cln = _read_json(cleaned_path)
    if spec.fmt == "contrastive_dict":
        orig_list = list(orig["true_facts"]) + list(orig["false_facts"])
        cln_list = list(cln["true_facts"]) + list(cln["false_facts"])
    else:
        orig_list = [d["statement"] for d in orig]
        cln_list = [d["statement"] for d in cln]
    if len(orig_list) != len(cln_list):
        raise ValueError(
            f"length mismatch for {poison}/{level}: {len(orig_list)} vs {len(cln_list)}"
        )
    return orig_list, cln_list


def audit(poison: str, level: str, model: str = AUDIT_MODEL, max_workers: int = 8) -> dict:
    """Audit one cleaned (poison, level) cell: fraction of sentences whose claim was CHANGED."""
    orig_list, cln_list = _load_original_and_cleaned(poison, level)
    client = _make_client()

    def check(i: int) -> tuple[int, bool]:
        o, c = orig_list[i], cln_list[i]
        if o.strip() == c.strip():
            return i, False  # reverted / unchanged -> SAME
        return i, (not claim_preserved(client, o, c, model))  # True = CHANGED

    changed_flags: list[bool] = [False] * len(orig_list)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(check, i) for i in range(len(orig_list))]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"audit {poison}/{level}"):
            i, changed = fut.result()
            changed_flags[i] = changed

    rows = [
        {"poison": poison, "level": level, "changed": changed_flags[i],
         "original": orig_list[i], "cleaned": cln_list[i]}
        for i in range(len(orig_list))
    ]
    df = pd.DataFrame(rows)
    out_csv = cleaned_dir(poison) / f"fidelity_audit_{level}.csv"
    df.to_csv(out_csv, index=False)
    n = len(orig_list)
    n_changed = int(sum(changed_flags))
    n_reverted = int(sum(1 for i in range(n) if orig_list[i].strip() == cln_list[i].strip()))
    rate = n_changed / n if n else float("nan")
    print(f"[audit] {poison}/{level}: claim_changed={rate:.3%} ({n_changed}/{n}; "
          f"{n_reverted} reverted/unchanged) -> {out_csv}")
    return {"poison": poison, "level": level, "n": n, "n_changed": n_changed,
            "n_reverted": n_reverted, "claim_changed_rate": rate}


def audit_all(model: str = AUDIT_MODEL, out: str = "results/fidelity_audit_summary.csv") -> None:
    """Audit every cleaned cell (all levels + diagnosed) that exists on disk."""
    summaries = []
    for poison in POISONS:
        for level in (*LEVELS, "diagnosed"):
            path = (cleaned_dir(poison) / "diagnosed_training_dist.json" if level == "diagnosed"
                    else cleaned_train_path(poison, level))
            if path.exists():
                summaries.append(audit(poison, level, model=model))
    if not summaries:
        print("No cleaned datasets found to audit.")
        return
    df = pd.DataFrame(summaries)
    out_path = REPO_ROOT / out
    df.to_csv(out_path, index=False)
    print(f"\n[audit] summary -> {out_path}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    fire.Fire({"audit": audit, "audit_all": audit_all})
