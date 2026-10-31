"""Kaggle script kernel: clone this repo at a pinned commit and run one stage.

push.sh fills in the placeholders; do not run this locally.
"""

import glob
import shutil
import subprocess

ARGS = "@ARGS@"
COMMIT = "@COMMIT@"
# clone outside /kaggle/working so the kernel output holds only results/
REPO = "/tmp/sft-dpo-eval"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone -q https://github.com/gondiam/sft-dpo-eval {REPO}")
sh(f"git -C {REPO} checkout -q {COMMIT}")
sh(f"pip install -q -e '{REPO}[train]'")
# Kaggle's preinstalled torchao is older than peft accepts, and nothing here uses it
sh("pip uninstall -q -y torchao")
stage, seed = (ARGS.split() + ["0"])[:2]
extra = ""
if stage == "dpo":
    # the SFT kernel's output is mounted read-only somewhere under /kaggle/input
    (cfg,) = glob.glob(f"/kaggle/input/**/sft_seed{seed}/adapter_config.json", recursive=True)
    extra = cfg.rsplit("/", 1)[0]
try:
    # one T4 of the two: the Trainer would otherwise wrap the model in DataParallel
    sh(f"cd {REPO} && CUDA_VISIBLE_DEVICES=0 python -m sft_dpo_eval.train {ARGS} {extra}")
finally:
    shutil.copytree(f"{REPO}/results", "/kaggle/working/results", dirs_exist_ok=True)
