"""Persona-prompting evaluation driver.

Loads two trained probes (RepE-style and roleplaying-style) from their result folders,
runs the model once on the 189-dialogue persona-prompting dataset, scores each dialogue
with both probes, and writes per-(fact, condition) score CSVs that the plotting script
consumes.

Usage:
    python -m deception_detection.scripts.persona_prompting_experiment \\
        --repe_folder=results/repe_layer_22_lr__... \\
        --roleplaying_folder=results/roleplaying_layer_22_lr__... \\
        [--batch_size=4] [--out_dir=results/persona_prompting]

If --repe_folder or --roleplaying_folder is omitted, the most recent folder matching
the standard id prefix is auto-discovered under <REPO_ROOT>/results/.
"""

import json
from pathlib import Path

import fire
import pandas as pd
import torch

from deception_detection.activations import Activations
from deception_detection.data.persona_prompting import PersonaPrompting
from deception_detection.detectors import get_detector_class
from deception_detection.experiment import ExperimentConfig
from deception_detection.log import logger
from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.utils import REPO_ROOT


def _find_latest(prefix: str, results_root: Path) -> Path:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root {results_root} does not exist")
    candidates = sorted(
        [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No results folder matching prefix '{prefix}' under {results_root}"
        )
    return candidates[0]


def _load_detector(folder: Path):
    cfg = ExperimentConfig.from_path(folder)
    detector_path = folder / "detector.pt"
    assert detector_path.exists(), f"No detector.pt in {folder}"
    detector_class = get_detector_class(cfg.method)
    detector = detector_class.load(detector_path)
    return cfg, detector


def run(
    repe_folder: str | None = None,
    roleplaying_folder: str | None = None,
    out_dir: str | None = None,
    batch_size: int = 4,
    layer: int = 22,
):
    """Score the persona-prompting dataset with both trained probes."""
    results_root = REPO_ROOT / "results"
    if out_dir is None:
        out_path = results_root / "persona_prompting"
    else:
        out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    repe_path = (
        Path(repe_folder)
        if repe_folder is not None
        else _find_latest("repe_layer_22_lr__repe_honesty", results_root)
    )
    rp_path = (
        Path(roleplaying_folder)
        if roleplaying_folder is not None
        else _find_latest("roleplaying_layer_22_lr__roleplaying", results_root)
    )
    logger.info(f"RepE probe folder: {repe_path}")
    logger.info(f"Roleplaying probe folder: {rp_path}")

    # Load both detectors first so any pickle/config error fails fast before
    # paying the cost of a 70B forward pass.
    repe_cfg, repe_detector = _load_detector(repe_path)
    rp_cfg, rp_detector = _load_detector(rp_path)
    assert repe_detector.layers is not None and rp_detector.layers is not None
    assert layer in repe_detector.layers, (
        f"Layer {layer} not in repe detector layers {repe_detector.layers}"
    )
    assert layer in rp_detector.layers, (
        f"Layer {layer} not in roleplaying detector layers {rp_detector.layers}"
    )

    # Build dataset (shuffle_upon_init=False to keep fact_id order for grouping)
    dataset = PersonaPrompting(variant="plain", shuffle_upon_init=False)
    assert dataset.metadata is not None
    n_facts = len(set(dataset.metadata["fact_id"]))
    logger.info(
        f"PersonaPrompting: {len(dataset)} dialogues over "
        f"{n_facts} facts x 7 conditions"
    )

    # Save experiment config snapshot
    config_snapshot = {
        "repe_folder": str(repe_path),
        "roleplaying_folder": str(rp_path),
        "model_name": ModelName.LLAMA_70B_3_3.value,
        "layer": layer,
        "batch_size": batch_size,
        "n_dialogues": len(dataset),
        "n_facts": n_facts,
        "repe_detector_layers": list(repe_detector.layers),
        "rp_detector_layers": list(rp_detector.layers),
        "repe_train_data": repe_cfg.train_data,
        "rp_train_data": rp_cfg.train_data,
        "repe_reg_coeff": repe_cfg.reg_coeff,
        "rp_reg_coeff": rp_cfg.reg_coeff,
    }
    with open(out_path / "cfg.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # Load model truncated at the target layer. We only need layer-22 residual
    # activations — layers 23..79 and the LM head are never used for probe
    # scoring — so loading only the first `layer` transformer blocks shrinks
    # the in-memory model from ~140 GB to ~40 GB (fits on one H100).
    # This mirrors what the training pipeline already does (see
    # Experiment.model property in experiment.py).
    model, tokenizer = get_model_and_tokenizer(
        ModelName.LLAMA_70B_3_3, cut_at_layer=layer
    )
    tokenized = TokenizedDataset.from_dataset(dataset, tokenizer)

    # Single forward pass at the target layer
    acts = Activations.from_model(
        model, tokenized, batch_size=batch_size, layers=[layer], verbose=True
    )

    # Free the model — we're done with it
    del model
    torch.cuda.empty_cache()

    # Score with each probe and persist
    for probe_name, detector in [
        ("repe_probe", repe_detector),
        ("roleplaying_probe", rp_detector),
    ]:
        probe_out = out_path / probe_name
        probe_out.mkdir(parents=True, exist_ok=True)

        scores = detector.score(acts)

        # Per-token scores -> mean per dialogue
        per_dialogue_means = [float(s.mean().item()) for s in scores.scores]
        per_token_scores = [s.tolist() for s in scores.scores]

        # Save full per-token scores
        with open(probe_out / "scores.json", "w") as f:
            json.dump(
                {
                    "scores": per_token_scores,
                    "labels": [lbl.name for lbl in scores.labels],
                },
                f,
            )

        # Write a tidy results.csv with metadata aligned to scores
        rows = []
        for i, score_val in enumerate(per_dialogue_means):
            rows.append(
                {
                    "fact_id": dataset.metadata["fact_id"][i],
                    "category": dataset.metadata["category"][i],
                    "condition": dataset.metadata["condition"][i],
                    "statement": dataset.metadata["statement"][i],
                    "question": dataset.metadata["question"][i],
                    "score": score_val,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(probe_out / "results.csv", index=False)
        logger.info(f"Wrote {probe_out}/results.csv ({len(df)} rows)")

    logger.info(f"Done. Outputs in {out_path}")


if __name__ == "__main__":
    fire.Fire({"run": run})
