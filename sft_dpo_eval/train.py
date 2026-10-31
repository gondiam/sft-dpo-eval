"""The GPU stages, run on Kaggle. Needs the [train] extra.

python -m sft_dpo_eval.train smoke           # real model, a few steps: memory and timings
python -m sft_dpo_eval.train base            # evaluate the untrained model
python -m sft_dpo_eval.train sft <seed>      # SFT, evaluate, sample DPO pairs
python -m sft_dpo_eval.train dpo <seed> <sft_adapter_dir>

Weights train in fp32 with fp16 autocast (a T4 has no bf16); generation runs
in fp16. Each stage logs config, versions and wall time to results/runs/.
"""

import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .data import rows, target
from .evaluate import RESULTS, evaluate, generate, prompt
from .pairs import build_pairs

ROOT = Path(__file__).resolve().parent.parent
CONFIG: dict[str, Any] = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
CHECKPOINTS = RESULTS / "checkpoints"


def load(model_id: str, adapters: tuple[str, ...] = (), fp16: bool = True) -> tuple[Any, Any]:
    """Base model with LoRA adapters merged in order (sft, then dpo)."""
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # the previous stage's model went out of scope when its function returned
    gc.collect()
    torch.cuda.empty_cache()
    dtype = torch.float16 if fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to("cuda")
    for a in adapters:
        model = PeftModel.from_pretrained(model, a).merge_and_unload()
    return model, AutoTokenizer.from_pretrained(model_id)


def lora() -> Any:
    from peft import LoraConfig

    return LoraConfig(**CONFIG["lora"], task_type="CAUSAL_LM")


def train_sft(seed: int, out: Path, n: int | None = None, max_steps: int = -1) -> dict[str, Any]:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = load(CONFIG["model"], fp16=False)
    items = rows("sft_labeled") + rows("sft_artificial")
    items = random.Random(seed).sample(items, n or len(items))
    ds = Dataset.from_list(
        [
            {
                "prompt": prompt(tok, it["messages"]),
                "completion": target(it["long_answer"], it["gold"]),
            }
            for it in items
        ]
    )
    args = SFTConfig(
        output_dir=str(out), seed=seed, max_steps=max_steps, fp16=True,
        gradient_checkpointing=True, logging_steps=25, save_strategy="no", report_to="none",
        **CONFIG["sft"],
    )  # fmt: skip
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora()
    )
    trainer.train()
    trainer.save_model(str(out))
    log = trainer.state.log_history
    return {"examples": len(ds), "log_history": log}


def sample_pairs(sft: str, seed: int, n: int | None = None) -> list[dict[str, Any]]:
    """On-policy pairs: the SFT model samples k answers per DPO question."""
    import torch

    torch.manual_seed(seed)
    model, tok = load(CONFIG["model"], (sft,))
    items = rows("dpo")[:n]
    outs = generate(
        model, tok, [prompt(tok, it["messages"]) for it in items],
        n=CONFIG["samples_per_item"], sample=True, **CONFIG["generation"],
    )  # fmt: skip
    del model
    pairs = build_pairs(items, {it["id"]: o for it, o in zip(items, outs, strict=True)})
    # DPO trains on the same prompt text the model is evaluated with
    return [
        {
            "prompt": prompt(tok, p["prompt"]),
            "chosen": p["chosen"][0]["content"],
            "rejected": p["rejected"][0]["content"],
            "chosen_source": p["chosen_source"],
        }
        for p in pairs
    ]


def train_dpo(
    seed: int, sft: str, pairs: list[dict[str, Any]], out: Path, max_steps: int = -1
) -> dict[str, Any]:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = load(CONFIG["model"], (sft,), fp16=False)
    ds = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])
    args = DPOConfig(
        output_dir=str(out), seed=seed, max_steps=max_steps, fp16=True,
        gradient_checkpointing=True, logging_steps=10, save_strategy="no", report_to="none",
        **CONFIG["dpo"],
    )  # fmt: skip
    # with a peft_config and no ref_model, the reference is the model with the
    # new adapter disabled: the merged SFT model
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora()
    )
    trainer.train()
    trainer.save_model(str(out))
    log = trainer.state.log_history
    return {"pairs": len(ds), "log_history": log}


def run_eval(
    system: str,
    seed: int,
    adapters: tuple[str, ...],
    limit: int | None = None,
    tasks: tuple[str, ...] = ("pubmedqa", "dev", "mmlu"),
) -> None:
    model, tok = load(CONFIG["model"], adapters)
    evaluate(model, tok, system, seed, tasks, limit, **CONFIG["generation"])


def versions() -> dict[str, str]:
    import peft
    import torch
    import transformers
    import trl

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }


def main(argv: list[str]) -> None:
    stage, seed = argv[0], int(argv[1]) if len(argv) > 1 else 0
    log: dict[str, Any] = {"stage": stage, "seed": seed, "config": CONFIG, "versions": versions()}
    timings: dict[str, float] = {}
    t = time.time()

    def lap(name: str) -> None:
        nonlocal t
        timings[name] = round((time.time() - t) / 3600, 3)
        t = time.time()

    sft_dir, dpo_dir = CHECKPOINTS / f"sft_seed{seed}", CHECKPOINTS / f"dpo_seed{seed}"
    if stage == "smoke":
        s = CONFIG["smoke"]
        steps, n_eval = s["max_steps"], s["eval_items"]
        run_eval("smoke_base", 0, (), n_eval, ("dev",))
        lap("eval_base")
        log["sft"] = train_sft(seed, sft_dir, s["items"], steps)
        lap("sft")
        run_eval("smoke_sft", seed, (str(sft_dir),), n_eval, ("dev",))
        lap("eval_sft")
        pairs = sample_pairs(str(sft_dir), seed, n_eval)
        lap("pairs")
        log["dpo"] = train_dpo(seed, str(sft_dir), pairs or _dummy_pairs(), dpo_dir, steps)
        lap("dpo")
        run_eval("smoke_sft_dpo", seed, (str(sft_dir), str(dpo_dir)), n_eval, ("dev",))
        lap("eval_sft_dpo")
        log["pairs_from_items"] = [len(pairs), n_eval]
    elif stage == "base":
        run_eval("base", 0, ())
        lap("eval")
    elif stage == "sft":
        log["sft"] = train_sft(seed, sft_dir)
        lap("train")
        run_eval("sft", seed, (str(sft_dir),))
        lap("eval")
        pairs = sample_pairs(str(sft_dir), seed)
        lap("pairs")
        (sft_dir / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
        log["pairs"] = {
            "n": len(pairs),
            "chosen_from_reference": sum(p["chosen_source"] == "reference" for p in pairs),
        }
    elif stage == "dpo":
        sft = argv[2]
        pairs = [json.loads(x) for x in (Path(sft) / "pairs.jsonl").read_text().splitlines()]
        log["dpo"] = train_dpo(seed, sft, pairs, dpo_dir)
        lap("train")
        run_eval("sft_dpo", seed, (sft, str(dpo_dir)))
        lap("eval")
    else:
        raise SystemExit(__doc__)
    log["gpu_hours"] = timings
    path = RESULTS / "runs" / f"{stage}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=1, default=str) + "\n")
    print(json.dumps({"stage": stage, "seed": seed, "gpu_hours": timings}))


def _dummy_pairs() -> list[dict[str, Any]]:
    # smoke only: 10 SFT steps can leave no wrong samples among 20 items
    return [{"prompt": "Q", "chosen": "Answer: yes", "rejected": "Answer: no"}] * 8


if __name__ == "__main__":
    main(sys.argv[1:])
