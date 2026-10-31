import itertools

import pytest

from sft_dpo_eval.data import CHOICES, load_splits, make_splits, parse


def test_parse_takes_last_answer() -> None:
    assert parse("Reasoning: x.\nAnswer: yes") == "yes"
    assert parse("Answer: no\n...\nANSWER: **Maybe**") == "maybe"
    assert parse("Reasoning: unclear.") is None
    assert parse("Answer: yesterday") is None
    assert parse("Answer: c", CHOICES) == "C"


def test_make_splits_disjoint_and_balanced() -> None:
    labeled = {i: "yes" for i in range(1000)}
    artificial = {i: ("yes" if i % 2 else "no") for i in range(10_000, 12_000)}
    s = make_splits(labeled, artificial, range(500), 100, n_sft=1000, n_dpo=400, n_mmlu=10)
    assert len(s["test"]) == 500 and len(s["dev"]) == 100 and len(s["sft_labeled"]) == 400
    for a, b in itertools.combinations(["test", "dev", "sft_labeled", "sft_artificial", "dpo"], 2):
        assert not set(s[a]) & set(s[b]), (a, b)
    for k, n in (("sft_artificial", 1000), ("dpo", 400)):
        assert len(s[k]) == n and sum(artificial[i] == "no" for i in s[k]) == n // 2
    with pytest.raises(ValueError):
        make_splits(labeled, artificial, range(500), 100, n_sft=2000)


def test_committed_splits_never_train_on_test() -> None:
    s = load_splits()
    train = set(s["sft_labeled"]) | set(s["sft_artificial"]) | set(s["dpo"])
    assert len(s["test"]) == 500
    assert not set(s["test"]) & (train | set(s["dev"]))
    assert not set(s["dev"]) & train
