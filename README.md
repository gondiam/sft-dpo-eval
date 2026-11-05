# sft-dpo-eval

Post-training a small open model with SFT and then DPO on biomedical question
answering, and measuring whether each stage helped by more than noise.

Most fine-tuning repos report one number per stage from one training run. This
one asks for each stage: is the gain distinguishable from noise? Is it more
than training-seed variance? What got worse? Was it worth the compute? The
evaluation plan is preregistered before any training run.

**Finding:** SFT lifted PubMedQA accuracy from 38.6% to 71.7% (+33.1 points,
95% CI +27.4 to +38.9). DPO on top did not help (−1.0 point, 95% CI −1.9 to −0.1):
it barely moved the model. Both stages cost about 7 points of MMLU, mostly because
the fine-tuned model now falls into repetition loops on out-of-domain questions.

## Question

On PubMedQA (yes / no / maybe questions about a PubMed abstract), does
supervised fine-tuning improve a 1.7B model, and does DPO on top of it improve
it further?

## Results

Accuracy on the 500-item official PubMedQA test set. Each fine-tuned system is
trained with 3 seeds. Differences are paired by item, with 95% CIs, and p-values
are Holm-adjusted over the 3 comparisons.

| Metric | base | sft | sft_dpo | sft − base | sft_dpo − sft |
|---|---|---|---|---|---|
| Accuracy | 38.6% | **71.7%** | 70.7% | **+33.1** [+27.4, +38.9] | −1.0 [−1.9, −0.1] |
| Macro-F1 | 35.7% | 50.8% | 50.2% | +15.1 | −0.6 |
| Recall on "maybe" | 80.0% | 0.0% | 0.0% | −80.0 | 0.0 |
| Unparseable outputs | 0.0% | 1.5% | 1.9% | | |
| Seed spread (min–max accuracy) | n/a | 70.8–72.6% | 70.0–71.8% | | |

Differences are in points, with 95% CIs for accuracy. Accuracy CIs per system
and the Macro-F1 CIs are in the output of `sft_dpo_eval.analyze`.

**Did SFT help? Yes.** Holm-adjusted p < 0.001, Cohen's d_z = 0.51, well above
the detectable difference below and above the always-"yes" reference. The gap
between seeds (1.8 points) is small next to the gain.

**Did DPO help? No.** It had to be positive to count, and it was not. The −1.0
point is distinguishable from zero (Holm-adjusted p = 0.032, d_z = −0.10) because
few items changed, but it is below the preregistered detectable difference, so
it is reported and not claimed as harm. Across seeds DPO changed 17–23 of 500
predictions, fixing 17 and breaking 32 in total. SFT + DPO against base is
+32.1 points [+26.5, +37.8], p < 0.001.

**The 71.7% has a ceiling.** After training the model almost never says "maybe"
(the training data has no such class, as predicted in the preregistration), so
it misses all 55 "maybe" items and cannot exceed 89%. That is why Macro-F1 stays
near 50%.

**Reference point:** always answering "yes" scores **55.2%** on this test set
(276 yes / 169 no / 55 maybe). A model below that has learned nothing useful.

**What this test set can detect:** the base model scores 38.6% on test (below
the always-"yes" reference: it answers "maybe" on 330 of 500 items). From that,
with 500 paired items, 80% power and alpha = 0.05, the smallest detectable change
is 4.8 points if 15% of items change outcome between stages, and 6.9 points if
30% do (`llm_judge_stats.smallest_detectable_difference`). Computed before any
SFT result was seen, as preregistered. Smaller differences are reported, but not
claimed.

## What got worse

Fine-tuning on a narrow task can damage general ability. Each stage is also
evaluated on 500 MMLU questions (exact-match accuracy, paired against base).

| | base | sft | sft_dpo |
|---|---|---|---|
| MMLU accuracy (500 questions) | 64.2% | 57.2% | 56.6% |
| Unparseable outputs | 0.8% | 10.9% | 11.2% |

General ability got measurably worse: −7.0 points for SFT (95% CI −11 to −3) and
−7.6 for SFT + DPO (95% CI −11 to −4), both Holm-adjusted p < 0.001.

Most of the drop looks like broken output rather than lost knowledge. In SFT
seed 0, none of the 64 unparseable answers reached an `Answer:` line, and 62 of
them repeat the same six-word phrase at least three times: repetition loops that
ran out of the 256-token budget. On answers that did parse, accuracy is 63.4–64.9% for SFT
against 64.7% for base. That last comparison is exploratory, not preregistered,
and it conditions on parsing, so questions that trigger loops may simply be harder.

## Cost

| Stage | GPU hours (T4) | Gain over previous stage (95% CI) |
|---|---|---|
| SFT | 1.9 | +33.1 points [+27.4, +38.9] |
| DPO | 1.8 | −1.0 point [−1.9, −0.1] |

Hours are per seed. SFT is 1.7 training plus 0.2 evaluation. DPO is 1.4 sampling
preference pairs, 0.2 training and 0.2 evaluation. The whole study, with 3 seeds
and the base evaluation, took 11.3 T4 hours. DPO cost nearly as much as SFT and
bought nothing here.

API spend: 0. The DPO preference pairs come from the SFT model's own samples,
not from an API model.

## Setup

**Model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
(Apache-2.0), with thinking mode disabled. LoRA adapters (r = 16, alpha = 32)
are trained in fp16 on one free Kaggle T4. Code and smoke tests use
Qwen/Qwen3-0.6B.

**Data:** [PubMedQA](https://github.com/pubmedqa/pubmedqa) (MIT), in the
reasoning-required setting: the question plus the abstract without its
conclusion.

| Split | Source | Items | Used for |
|---|---|---|---|
| test | `pqa_labeled`, official test IDs | 500 | final evaluation only, frozen |
| dev | `pqa_labeled`, rest | 100 | checkpoint and prompt sanity checks |
| sft | `pqa_artificial` (balanced yes/no) + remaining `pqa_labeled` | 10,000 + 400 | SFT |
| dpo | `pqa_artificial`, disjoint from sft | 2,500 questions | DPO pairs |

SFT and DPO sizes were halved (from 20,000 and 5,000) after a smoke run on
the T4 measured the full plan at about 22 GPU hours, against a weekly
Kaggle quota of 30. Nothing had been trained or evaluated on test yet.

**Output format:** `Reasoning: <1-3 sentences>` then `Answer: yes|no|maybe`.
Outputs that do not parse count as wrong, and their rate is reported.

**DPO pairs (on-policy):** the SFT model samples 4 answers per question. A
correct answer becomes `chosen` (or, if none is correct, the dataset's long
answer plus the gold label), and an incorrect one becomes `rejected`. Items the
model always gets right give no pair, so 2,500 questions yielded only 402–414
pairs per seed.

## Limitations

- **`pqa_artificial` labels are heuristic and noisy**, and it has no "maybe"
  class. SFT and DPO learn from that noise, and recall on "maybe" is expected
  to drop. It is measured, not hidden.
- **One dataset, one model size, LoRA only.** Nothing here says how full
  fine-tuning or a larger model would behave.
- **English only.**
- **Three seeds** separate training variance from evaluation noise only roughly.
- **MMLU is a coarse probe** of general ability. A flat MMLU score does not
  prove nothing else got worse.
- **The DPO null is about this setup, not about DPO.** About 400 pairs at
  effective batch 16 is 26 steps. Loss went from 0.693 to about 0.685, with a
  reward margin near 0.02: the policy hardly left the reference. A larger pair
  set or learning rate might behave differently; none was tried, because that
  would be tuning after seeing test results.
- **SFT and DPO share the same noisy label source**, so neither can teach "maybe".

## Reproduce

Every prediction behind the tables is in `results/predictions/`, so the analysis
needs no GPU:

```
uv sync
uv run pytest
uv run python -m sft_dpo_eval.analyze   # rebuilds the tables from results/predictions/
```

To rerun the training, `uv run python -m sft_dpo_eval.data` rebuilds the splits
in `data/`, and each stage runs as a Kaggle T4 kernel pinned to the pushed commit
(needs a Kaggle token; set `KAGGLE_USER` to your username):

```
sh kaggle/push.sh base
sh kaggle/push.sh sft 0    # seeds 0, 1, 2
sh kaggle/push.sh dpo 0    # after sft of the same seed has finished
```

Fetch each stage's output with the command at the top of `kaggle/push.sh`. Run
logs with versions, config and GPU hours are in `results/runs/`.

## License

Code: MIT. Data: PubMedQA (MIT), MMLU (MIT). Model: Qwen3 (Apache-2.0).
