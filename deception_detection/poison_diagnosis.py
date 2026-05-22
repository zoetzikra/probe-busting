"""Idea 1: dataset-level poison diagnosis + targeted cleaning.

A per-statement cleaner is structurally blind to a *correlation* — a spurious feature is a
dataset-level property (feature <-> label), invisible when you look at one statement at a
time. That is why blind (minimal/medium) cleaning failed. This module splits the job:

  (1) DIAGNOSE — show a small model a batch of LABELLED examples and ask for its top-3 guesses
      of the non-semantic feature that separates the classes (other than truthfulness).
  (2) CLEAN — turn the chosen guess into a targeted, maximal-style cleaning prompt (wrapped in
      the truth-preservation guardrails we learned are necessary) and strip the feature.

This tests whether cheap dataset-level diagnosis reaches maximal-quality recovery WITHOUT
being told the attack in advance — the positive counterpart to the negative result that blind
cleaning fails.

CLI (env with openai, e.g. deception_env):
    python -m deception_detection.poison_diagnosis diagnose --poison cats_dogs
    python -m deception_detection.poison_diagnosis diagnose_all
    python -m deception_detection.poison_diagnosis clean_with_diagnosis --poison=cats_dogs
"""

from __future__ import annotations

import json
import random

import fire

from deception_detection.sanitization import (
    CLEAN_MODEL,
    GATE_JUDGE_MODEL,
    POISONS,
    _clean_many_preserving,
    _dump_statements,
    _load_statements,
    _make_client,
    _read_json,
    _write_json,
    cleaned_dir,
)

DIAGNOSE_MODEL = "gpt-4o-mini"  # the "small model" — keep diagnosis lightweight

DIAGNOSIS_SYSTEM = (
    "You are auditing a dataset used to train a truthfulness probe. Each statement is "
    "labelled TRUE or FALSE according to whether its claim is factually correct. An adversary "
    "may have injected a SPURIOUS CORRELATION: an incidental surface feature that happens to "
    "track the label but is NOT the actual truth of the claim — e.g. a recurring word or "
    "topic, a stylistic tic, hedging vs. confident phrasing, verb tense, punctuation, or "
    "length. Your job is to find such a shortcut a probe could exploit.\n\n"
    "IGNORE whether the claims are actually true or false — that is the legitimate signal. "
    "Look only for incidental patterns that co-vary with the label. Give your THREE most "
    "likely hypotheses, most likely first. If you see no spurious pattern, say so.\n\n"
    "Respond with JSON only: {\"hypotheses\": [{\"feature\": <short name>, \"evidence\": "
    "<what you observed across the examples>, \"cleaning_instruction\": <one imperative "
    "sentence telling a rewriter how to strip this feature while keeping the claim and its "
    "truth value unchanged>}]} — exactly three hypotheses."
)


def _sample(poison: str, n_per_class: int, seed: int) -> tuple[list[str], list[str]]:
    statements, labels = _load_statements(POISONS[poison])
    rng = random.Random(seed)
    trues = [s for s, lab in zip(statements, labels, strict=True) if lab == 0]
    falses = [s for s, lab in zip(statements, labels, strict=True) if lab == 1]
    rng.shuffle(trues)
    rng.shuffle(falses)
    return trues[:n_per_class], falses[:n_per_class]


def _user_prompt(trues: list[str], falses: list[str]) -> str:
    t = "\n".join(f"  [TRUE]  {s}" for s in trues)
    f = "\n".join(f"  [FALSE] {s}" for s in falses)
    return (
        f"Here are {len(trues)} TRUE-labelled and {len(falses)} FALSE-labelled statements.\n\n"
        f"{t}\n{f}\n\n"
        "What incidental surface feature(s) separate the TRUE set from the FALSE set? "
        "Give your top 3 hypotheses as specified."
    )


def diagnose(
    poison: str, model: str = DIAGNOSE_MODEL, n_per_class: int = 50, seed: int = 0
) -> dict:
    """Show the model labelled examples; return + save its top-3 spurious-feature guesses."""
    if poison not in POISONS:
        raise ValueError(f"Unknown poison {poison!r}; choose from {list(POISONS)}")
    trues, falses = _sample(poison, n_per_class, seed)
    client = _make_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DIAGNOSIS_SYSTEM},
            {"role": "user", "content": _user_prompt(trues, falses)},
        ],
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    hyps = parsed.get("hypotheses", [])[:3]
    out = {"poison": poison, "model": model, "n_per_class": n_per_class, "seed": seed,
           "hypotheses": hyps}
    path = cleaned_dir(poison) / "diagnosis.json"
    _write_json(out, path)
    print(f"\n=== diagnosis: {poison} (model={model}, n={n_per_class}/class) ===")
    for i, h in enumerate(hyps):
        print(f"  [{i}] feature: {h.get('feature','?')}")
        print(f"      evidence: {h.get('evidence','')}")
        print(f"      cleaning: {h.get('cleaning_instruction','')}")
    print(f"  saved -> {path}")
    return out


def diagnose_all(model: str = DIAGNOSE_MODEL, n_per_class: int = 50, seed: int = 0) -> None:
    for poison in POISONS:
        diagnose(poison, model=model, n_per_class=n_per_class, seed=seed)


def _targeted_prompt(cleaning_instruction: str) -> str:
    """Wrap the diagnosed cleaning instruction in the truth-preservation guardrails that the
    label audit showed are necessary (no negating, no relabelling, no stating the truth value)."""
    return (
        "This dataset contains a spurious correlation. " + cleaning_instruction.strip().rstrip(".")
        + ". CRITICAL: keep the core factual claim and its truth value EXACTLY as given — do "
        "NOT correct, negate, fix, or relabel any statement even if it is false, and do NOT "
        "append or state whether it is true or false. Change only what is needed to remove the "
        "feature above. Return only the cleaned statement."
    )


def clean_with_diagnosis(
    poison: str,
    choice: int = 0,
    model: str = CLEAN_MODEL,
    overwrite: bool = False,
    label_preserving: bool = True,
    judge_model: str = GATE_JUDGE_MODEL,
    max_workers: int = 8,
) -> str:
    """Clean the poisoned training set using the chosen diagnosis hypothesis. Writes
    data/<poison>/cleaned/diagnosed_training_dist.json (variant: training_dist_diagnosed).

    With label_preserving=True (default) the cleaner is gated: a wrong hypothesis strips the
    wrong feature (leaving the poison in -> low recovery) but can never flip truth labels."""
    diag_path = cleaned_dir(poison) / "diagnosis.json"
    if not diag_path.exists():
        raise FileNotFoundError(f"{diag_path} missing; run `diagnose --poison {poison}` first.")
    diag = _read_json(diag_path)
    hyps = diag["hypotheses"]
    if not 0 <= choice < len(hyps):
        raise ValueError(f"choice {choice} out of range (have {len(hyps)} hypotheses)")
    chosen = hyps[choice]
    prompt = _targeted_prompt(chosen["cleaning_instruction"])
    print(f"[diagnosed-clean] {poison}: using hypothesis [{choice}] '{chosen.get('feature')}'"
          f" gate={label_preserving}")
    print(f"  cleaning prompt: {prompt}")

    out_path = cleaned_dir(poison) / "diagnosed_training_dist.json"
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists (use overwrite=True)")
        return str(out_path)

    spec = POISONS[poison]
    statements, labels = _load_statements(spec)
    if label_preserving:
        cleaned, _ = _clean_many_preserving(statements, prompt, model, judge_model, max_workers)
    else:
        from deception_detection.sanitization import _clean_many
        cleaned = _clean_many(statements, prompt, model)
    _dump_statements(spec, cleaned, labels, out_path)
    # record which hypothesis was used, for the writeup
    _write_json({**diag, "chosen_index": choice, "chosen": chosen, "clean_prompt": prompt},
                cleaned_dir(poison) / "diagnosis_used.json")
    print(f"[done] wrote {out_path}")
    return str(out_path)


if __name__ == "__main__":
    fire.Fire(
        {
            "diagnose": diagnose,
            "diagnose_all": diagnose_all,
            "clean_with_diagnosis": clean_with_diagnosis,
        }
    )
