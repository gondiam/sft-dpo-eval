import json
from pathlib import Path

import numpy as np

from sft_dpo_eval.analyze import compare_stages, correct, load, report

N = 500
WRONG = {"yes": "no", "no": "yes", "maybe": "yes"}


def _write(root: Path, rng: np.random.Generator, gold: list[str], name: str, p: float) -> None:
    rows = []
    for i, g in enumerate(gold):
        ok = rng.random() < p
        rows.append({"id": i, "gold": g, "pred": g if ok else WRONG[g], "output": ""})
    path = root / "pubmedqa" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_recovers_known_effect_and_null(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    gold = list(rng.choice(["yes", "no", "maybe"], N, p=[0.55, 0.34, 0.11]))
    _write(tmp_path, rng, gold, "base_seed0", 0.60)
    for k in range(3):
        _write(tmp_path, rng, gold, f"sft_seed{k}", 0.75)
        _write(tmp_path, rng, gold, f"sft_dpo_seed{k}", 0.75)

    preds = load("pubmedqa", tmp_path)
    c = {r.label: r for r in compare_stages({s: correct(p) for s, p in preds.items()})}
    up, null = c["sft vs base"], c["sft_dpo vs sft"]
    assert up.significant and up.ci[0] < 0.15 < up.ci[1]
    assert not null.significant and null.ci[0] < 0 < null.ci[1]
    assert null.p_adjusted is not None and null.p_adjusted >= null.p_value

    text = report(tmp_path)
    assert "Accuracy (95% CI)" in text and "sft_dpo vs sft" in text
