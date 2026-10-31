# sft-dpo-eval

Post-training a small open model with SFT and then DPO on biomedical question
answering, and measuring whether each stage helped by more than noise.

Most fine-tuning repos report one number per stage from one training run. This
one asks for each stage: is the gain distinguishable from noise? Is it more
than training-seed variance? What got worse? Was it worth the compute? The
evaluation plan is preregistered before any training run.

> **Status:** scaffold. Every result below is `TODO` until it has been run.

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
| Accuracy | TODO | TODO | TODO | TODO | TODO |
| Macro-F1 | TODO | TODO | TODO | TODO | TODO |
| Recall on "maybe" | TODO | TODO | TODO | TODO | TODO |
| Unparseable outputs | TODO | TODO | TODO | | |
| Seed spread (min–max accuracy) | n/a | TODO | TODO | | |

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
| MMLU accuracy (500 questions) | TODO | TODO | TODO |

TODO: what got worse, if anything.

## Cost

| Stage | GPU hours (T4) | Gain over previous stage (95% CI) |
|---|---|---|
| SFT | TODO | TODO |
| DPO | TODO | TODO |

API spend: 0. The DPO preference pairs come from the SFT model's own samples,
not from an API model.

## Setup

**Model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
(Apache-2.0), with thinking mode disabled. LoRA adapters (r = 16, alpha = 32)
are trained in fp16 on one free Kaggle/Colab T4. Code and smoke tests use
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
answer plus the gold label), and an incorrect one becomes `rejected`.

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
- TODO: anything learned while running it.

## Reproduce

TODO: install, data preparation, Kaggle notebook, analysis.

```
uv run python -m sft_dpo_eval.analyze   # rebuilds every table and figure from results/predictions/
```

## License

Code: MIT. Data: PubMedQA (MIT), MMLU (MIT). Model: Qwen3 (Apache-2.0).
