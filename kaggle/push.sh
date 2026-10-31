#!/bin/sh
# Run one stage on a Kaggle T4: kaggle/push.sh smoke | base | sft <seed> | dpo <seed>
# Kaggle clones the repo from GitHub at HEAD, so HEAD must be pushed first.
# Fetch the results afterwards with:
#   uvx kaggle kernels output $KAGGLE_USER/<slug> -p /tmp/out && cp -r /tmp/out/sft-dpo-eval/results/{predictions,runs} results/
set -eu
user=${KAGGLE_USER:-gonzalodazamor}
stage=$1
seed=${2:-0}
cd "$(dirname "$0")/.."
git fetch -q
[ "$(git rev-parse HEAD)" = "$(git rev-parse '@{u}')" ] || { echo "push HEAD to GitHub first"; exit 1; }

slug="sft-dpo-eval-$stage"
sources="[]"
case $stage in
  sft) slug="$slug-$seed" ;;
  dpo) slug="$slug-$seed"; sources="[\"$user/sft-dpo-eval-sft-$seed\"]" ;;
esac
dir=$(mktemp -d)
sed -e "s|@ARGS@|$stage $seed|" -e "s|@COMMIT@|$(git rev-parse HEAD)|" kaggle/run.py > "$dir/run.py"
cat > "$dir/kernel-metadata.json" <<JSON
{"id": "$user/$slug", "title": "$slug", "code_file": "run.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "kernel_sources": $sources, "dataset_sources": [], "competition_sources": []}
JSON
uvx kaggle kernels push -p "$dir" --accelerator NvidiaTeslaT4
echo "status: uvx kaggle kernels status $user/$slug"
