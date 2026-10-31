"""Greedy generation and scoring on the GPU. Writes results/predictions/<task>/."""

import json
from pathlib import Path
from typing import Any

from .data import CHOICES, LABELS, Messages, parse, rows

RESULTS = Path(__file__).resolve().parent.parent / "results"
TASKS = {"pubmedqa": ("test", LABELS), "dev": ("dev", LABELS), "mmlu": ("mmlu", CHOICES)}


def prompt(tok: Any, messages: Messages) -> str:
    """Chat-formatted prompt, thinking disabled. SFT and DPO train on exactly this text."""
    text: str = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return text


def generate(
    model: Any,
    tok: Any,
    prompts: list[str],
    *,
    n: int = 1,
    sample: bool = False,
    max_new_tokens: int = 256,
    batch_size: int = 16,
) -> list[list[str]]:
    """n completions per prompt: greedy, or sampled with Qwen3's non-thinking defaults."""
    import torch

    tok.padding_side = "left"
    kw: dict[str, Any] = (
        {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
        if sample
        else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    )
    # batching similar lengths together wastes less compute on padding
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[list[str]] = [[] for _ in prompts]
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = tok(
            [prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.no_grad():
            g = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tok.pad_token_id,
                **kw,
            )
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for k, i in enumerate(idx):
            out[i] = texts[k * n : (k + 1) * n]
    return out


def evaluate(
    model: Any,
    tok: Any,
    system: str,
    seed: int,
    tasks: tuple[str, ...] = ("pubmedqa", "dev", "mmlu"),
    limit: int | None = None,
    **gen: Any,
) -> None:
    model.eval()
    for task in tasks:
        split, labels = TASKS[task]
        items = rows(split)[:limit]
        outs = generate(model, tok, [prompt(tok, it["messages"]) for it in items], **gen)
        path = RESULTS / "predictions" / task / f"{system}_seed{seed}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for it, (o,) in zip(items, outs, strict=True):
                r = {"id": it["id"], "gold": it["gold"], "pred": parse(o, labels), "output": o}
                f.write(json.dumps(r) + "\n")
