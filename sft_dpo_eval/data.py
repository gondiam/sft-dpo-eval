"""PubMedQA and MMLU splits, prompt format and answer parsing.

`python -m sft_dpo_eval.data` writes data/splits.json (item IDs only). Every
other step loads rows by those IDs, so the splits are fixed in git and the test
IDs can be checked never to reach training.
"""

import json
import random
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = ("yes", "no", "maybe")
CHOICES = ("A", "B", "C", "D")

PUBMEDQA_INSTRUCTIONS = (
    "Answer the research question using only the abstract. Reply in exactly this format:\n"
    "Reasoning: <1-3 sentences>\n"
    "Answer: <yes, no or maybe>"
)
MMLU_INSTRUCTIONS = (
    "Answer the multiple-choice question. Reply in exactly this format:\n"
    "Reasoning: <1-3 sentences>\n"
    "Answer: <A, B, C or D>"
)

Messages = list[dict[str, str]]


def make_splits(
    labeled: dict[int, str],
    artificial: dict[int, str],
    test_ids: Iterable[int],
    n_mmlu_pool: int,
    *,
    n_dev: int = 100,
    n_sft: int = 10_000,
    n_dpo: int = 2_500,
    n_mmlu: int = 500,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Split item IDs. `labeled` / `artificial` map PubMedQA pubid -> label.

    test: the official 500. dev: n_dev of the other expert-labelled items, the
    rest go to SFT. Artificial items have no "maybe"; SFT and DPO draw disjoint
    sets balanced between yes and no. mmlu: indices into the MMLU test split.
    """
    rng = random.Random(seed)
    test = sorted(test_ids)
    if not set(test) <= labeled.keys():
        raise ValueError("test IDs must all be in the labelled set")
    rest = sorted(labeled.keys() - set(test))
    rng.shuffle(rest)
    by_label = {lab: sorted(i for i, v in artificial.items() if v == lab) for lab in ("yes", "no")}
    for ids in by_label.values():
        rng.shuffle(ids)
    half_sft, half_dpo = n_sft // 2, n_dpo // 2
    if any(len(ids) < half_sft + half_dpo for ids in by_label.values()):
        raise ValueError("not enough artificial items for balanced SFT and DPO sets")
    return {
        "test": test,
        "dev": sorted(rest[:n_dev]),
        "sft_labeled": sorted(rest[n_dev:]),
        "sft_artificial": sorted(by_label["yes"][:half_sft] + by_label["no"][:half_sft]),
        "dpo": sorted(
            by_label["yes"][half_sft : half_sft + half_dpo]
            + by_label["no"][half_sft : half_sft + half_dpo]
        ),
        "mmlu": sorted(rng.sample(range(n_mmlu_pool), n_mmlu)),
    }


def pubmedqa_messages(question: str, contexts: Sequence[str]) -> Messages:
    # reasoning-required setting: the abstract without its conclusion (which is long_answer)
    abstract = "\n".join(contexts)
    return [
        {"role": "system", "content": PUBMEDQA_INSTRUCTIONS},
        {"role": "user", "content": f"Abstract:\n{abstract}\n\nQuestion: {question}"},
    ]


def mmlu_messages(question: str, options: Sequence[str]) -> Messages:
    lines = "\n".join(f"{c}. {o}" for c, o in zip(CHOICES, options, strict=True))
    return [
        {"role": "system", "content": MMLU_INSTRUCTIONS},
        {"role": "user", "content": f"{question}\n\n{lines}"},
    ]


def target(reasoning: str, answer: str) -> str:
    """The completion SFT trains on."""
    return f"Reasoning: {reasoning}\nAnswer: {answer}"


def parse(text: str, labels: Sequence[str] = LABELS) -> str | None:
    """The last `Answer: <label>` in the text, or None (scored as wrong)."""
    pattern = r"answer:\s*\**\s*(" + "|".join(map(re.escape, labels)) + r")\b"
    found = re.findall(pattern, text, re.IGNORECASE)
    if not found:
        return None
    return next(lab for lab in labels if lab.lower() == found[-1].lower())


def load_splits() -> dict[str, list[int]]:
    splits: dict[str, list[int]] = json.loads((DATA / "splits.json").read_text())
    return splits


def rows(split: str) -> list[dict[str, Any]]:
    """Rows of one split: id, messages, gold, plus long_answer for PubMedQA."""
    from datasets import load_dataset

    ids = load_splits()[split]
    if split == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        return [
            {
                "id": i,
                "messages": mmlu_messages(ds[i]["question"], ds[i]["choices"]),
                "gold": CHOICES[ds[i]["answer"]],
                "subject": ds[i]["subject"],
            }
            for i in ids
        ]
    config = "pqa_artificial" if split in ("sft_artificial", "dpo") else "pqa_labeled"
    ds = load_dataset("qiaojin/PubMedQA", config, split="train")
    index = {p: n for n, p in enumerate(ds["pubid"])}
    out = []
    for i in ids:
        r = ds[index[i]]
        out.append(
            {
                "id": i,
                "messages": pubmedqa_messages(r["question"], r["context"]["contexts"]),
                "gold": r["final_decision"],
                "long_answer": r["long_answer"],
            }
        )
    return out


def main() -> None:
    from datasets import load_dataset

    def labels(config: str) -> dict[int, str]:
        ds = load_dataset("qiaojin/PubMedQA", config, split="train")
        return dict(zip(ds["pubid"], ds["final_decision"], strict=True))

    test_ids = [int(i) for i in json.loads((DATA / "test_ground_truth.json").read_text())]
    n_mmlu = load_dataset("cais/mmlu", "all", split="test").num_rows
    splits = make_splits(labels("pqa_labeled"), labels("pqa_artificial"), test_ids, n_mmlu)
    (DATA / "splits.json").write_text(json.dumps(splits) + "\n")
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
