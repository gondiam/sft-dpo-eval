"""Predictions -> the README tables. Needs no GPU or API, only results/predictions/.

Layout: results/predictions/<task>/<system>_seed<k>.jsonl, one line per item
with `id`, `gold`, `pred` (parsed label, null if unparseable) and `output`.
"""

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from llm_judge_stats import Comparison, adjust_pvalues, bootstrap_ci, compare, compare_many, f1
from numpy.typing import NDArray

from .data import LABELS

PREDICTIONS = Path(__file__).resolve().parent.parent / "results" / "predictions"
STAGES = ("base", "sft", "sft_dpo")
CHAIN = (("base", "sft"), ("sft", "sft_dpo"), ("base", "sft_dpo"))

Preds = tuple[NDArray[np.str_], NDArray[np.object_]]  # gold (items,), preds (items, seeds)


def load(task: str, root: Path = PREDICTIONS) -> dict[str, Preds]:
    """system -> (gold, preds), items sorted by id so systems line up."""
    runs: dict[str, list[dict[int, dict[str, str]]]] = defaultdict(list)
    for path in sorted((root / task).glob("*_seed*.jsonl")):
        system = path.stem.rsplit("_seed", 1)[0]
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        runs[system].append({r["id"]: r for r in lines})
    out = {}
    for system, seeds in runs.items():
        ids = sorted(seeds[0])
        if any(sorted(s) != ids for s in seeds):
            raise ValueError(f"{task}/{system}: seeds cover different items")
        gold = np.array([seeds[0][i]["gold"] for i in ids])
        preds = np.array([[s[i]["pred"] for s in seeds] for i in ids], dtype=object)
        out[system] = (gold, preds)
    return out


def correct(p: Preds) -> NDArray[np.float64]:
    """Per-item accuracy averaged over seeds (0/1 with one seed)."""
    gold, preds = p
    return np.asarray((preds == gold[:, None]).mean(axis=1), float)


def macro_f1(gold: NDArray[np.str_], preds: NDArray[np.object_]) -> float:
    """Macro-F1 over yes/no/maybe, averaged over seeds."""
    return float(
        np.mean(
            [
                np.nanmean([f1(gold, preds[:, k], lab) for lab in LABELS])
                for k in range(preds.shape[1])
            ]
        )
    )


def compare_stages(scores: dict[str, NDArray[np.float64]]) -> list[Comparison]:
    """The preregistered comparisons, paired by item, Holm-adjusted together.

    With several seeds the per-item score is the share of seeds that got it
    right, so the test is a paired t test; the CI covers item sampling for
    these seeds. Seed spread is reported separately.
    """
    res = [compare(scores[a], scores[b], names=(a, b)) for a, b in CHAIN if {a, b} <= scores.keys()]
    adj = adjust_pvalues([r.p_value for r in res])
    return [
        dataclasses.replace(r, p_adjusted=float(p), correction="Holm")
        for r, p in zip(res, adj, strict=True)
    ]


def _ci(est: float, lo: float, hi: float) -> str:
    return f"{est:.1%} [{lo:.1%}, {hi:.1%}]"


def report(root: Path = PREDICTIONS, seed: int = 0) -> str:
    out = []
    pub = load("pubmedqa", root)
    systems = [s for s in STAGES if s in pub]
    if systems:
        rows: dict[str, list[str]] = defaultdict(list)
        for s in systems:
            gold, preds = pub[s]
            acc = (preds == gold[:, None]).mean(axis=0)
            maybe = gold == "maybe"
            rows["Accuracy (95% CI)"].append(
                _ci(*bootstrap_ci(np.mean, correct(pub[s]), seed=seed))
            )
            rows["Macro-F1 (95% CI)"].append(
                _ci(*bootstrap_ci(macro_f1, gold, preds, strata=gold, seed=seed))
            )
            rows['Recall on "maybe"'].append(f"{(preds[maybe] == 'maybe').mean():.1%}")
            rows["Unparseable outputs"].append(f"{(preds == None).mean():.1%}")  # noqa: E711
            rows["Seed range (accuracy)"].append(
                f"{acc.min():.1%}–{acc.max():.1%} ({len(acc)} seeds)" if len(acc) > 1 else "n/a"
            )
        out += [f"## PubMedQA test ({len(pub[systems[0]][0])} items)", ""]
        out += ["| Metric | " + " | ".join(systems) + " |", "|---" * (len(systems) + 1) + "|"]
        out += [f"| {k} | " + " | ".join(v) + " |" for k, v in rows.items()]
        out += [
            "",
            "```",
            *(f"{c}\n" for c in compare_stages({s: correct(pub[s]) for s in systems})),
            "```",
        ]
    mmlu = load("mmlu", root)
    if "base" in mmlu and len(mmlu) > 1:
        scores = {s: correct(mmlu[s]) for s in STAGES if s in mmlu}
        out += ["", f"## MMLU ({len(scores['base'])} questions)", "", "```"]
        out += [f"{c}\n" for c in compare_many(scores, "base")]
        out += ["```"]
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
