# Preregistration

Committed and pushed before any training run and before any model has been
evaluated on the test set. The git history is the timestamp. Any later change
to what is below is listed under Deviations, with its reason.

## Question

On PubMedQA (reasoning-required setting), does SFT improve Qwen3-1.7B, and does
DPO on top of SFT improve it further?

## Fixed before training

- Data: `data/splits.json` (500 official test IDs, 100 dev, 10,400 SFT, 2,500
  DPO questions, 500 MMLU questions), generated with seed 0 by
  `python -m sft_dpo_eval.data`. Test items are never used for training,
  prompt design or checkpoint choice; a test checks the splits are disjoint.
- Training: `configs/train.yaml` as of this commit. One run per seed, seeds
  0, 1, 2; the last checkpoint is the one evaluated (no early stopping, no
  checkpoint selection).
- Decoding: greedy, at most 256 new tokens, thinking disabled.
- Scoring: an output is correct when its last `Answer: <label>` matches the
  gold label. No answer is wrong.

## Primary analysis

Outcome: accuracy on the 500 test items. Per item, the share of the 3 seeds
that answered correctly (base has one deterministic run).

Three paired comparisons, item by item, with `llm_judge_stats.compare`
(paired t test on per-item differences, 95% CI, Cohen's d_z):

1. sft vs base
2. sft_dpo vs sft
3. sft_dpo vs base

p-values Holm-adjusted over these three, alpha = 0.05. A stage "helped" only if
its adjusted p < 0.05 and the difference is positive. Every difference is
reported with its CI whatever its p-value; a null result is reported as the
range of effects the CI rules out.

Detectability: before SFT results are seen, the smallest difference the test
set can detect with 80% power is computed with
`smallest_detectable_difference(500, p_base=<base test accuracy>, discordance=d)`
for d = 0.15 and 0.30, and added to the README. Differences below it are
reported but not claimed.

## Secondary analyses (reported, not used for the main claim)

- Macro-F1 over yes/no/maybe with a stratified bootstrap 95% CI.
- Recall on "maybe" per stage. Expected to fall after SFT, since the artificial
  training data has no "maybe".
- Unparseable-output rate per stage.
- Seed spread: min to max test accuracy across seeds.
- MMLU (500 questions): each stage vs base, paired, Holm over the two
  comparisons. A significant drop is reported as damage to general ability.
- GPU hours per stage against the gain over the previous stage.

## Deviations

None to the design above. For the record: SFT seeds 0 and 1 ran at `ccde61e`;
SFT seed 2 and all DPO runs ran at `7f9cc20`. Those commits differ only in
`README.md` and the base model's results, not in code, data or config.
