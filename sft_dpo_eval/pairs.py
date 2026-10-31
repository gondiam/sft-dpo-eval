"""Turn the SFT model's own samples into DPO preference pairs (on-policy, no API)."""

from collections.abc import Mapping, Sequence
from typing import Any

from .data import parse, target


def build_pairs(
    items: Sequence[Mapping[str, Any]], samples: Mapping[int, Sequence[str]]
) -> list[dict[str, Any]]:
    """One pair per item whose samples include a wrong answer.

    chosen: the first sample with the gold label, or, if none has it, the
    reference conclusion plus the gold label. rejected: the first sample
    without it (unparseable counts as wrong). Items the model always gets
    right give no pair: there is nothing to prefer against.
    """
    pairs = []
    for it in items:
        outs = samples[it["id"]]
        right = [o for o in outs if parse(o) == it["gold"]]
        wrong = [o for o in outs if parse(o) != it["gold"]]
        if not wrong:
            continue
        chosen = right[0] if right else target(it["long_answer"], it["gold"])
        pairs.append(
            {
                "id": it["id"],
                "prompt": it["messages"],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": wrong[0]}],
                "chosen_source": "model" if right else "reference",
            }
        )
    return pairs
