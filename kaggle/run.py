"""Kaggle script kernel: clone this repo at a pinned commit and run one stage.

push.sh fills in the placeholders; do not run this locally.
"""

import glob
import subprocess

ARGS = "@ARGS@"
COMMIT = "@COMMIT@"
REPO = "/kaggle/working/sft-dpo-eval"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone -q https://github.com/gondiam/sft-dpo-eval {REPO}")
sh(f"git -C {REPO} checkout -q {COMMIT}")
sh(f"pip install -q -e '{REPO}[train]'")
stage, seed = (ARGS.split() + ["0"])[:2]
extra = ""
if stage == "dpo":
    # the SFT kernel's output is mounted read-only somewhere under /kaggle/input
    (cfg,) = glob.glob(f"/kaggle/input/**/sft_seed{seed}/adapter_config.json", recursive=True)
    extra = cfg.rsplit("/", 1)[0]
# one T4 of the two: the Trainer would otherwise wrap the model in DataParallel
sh(f"cd {REPO} && CUDA_VISIBLE_DEVICES=0 python -m sft_dpo_eval.train {ARGS} {extra}")
