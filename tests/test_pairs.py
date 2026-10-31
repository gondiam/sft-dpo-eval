from sft_dpo_eval.pairs import build_pairs


def test_build_pairs() -> None:
    items = [
        {"id": i, "messages": [{"role": "user", "content": "q"}], "gold": "yes", "long_answer": "L"}
        for i in (1, 2, 3)
    ]
    samples = {
        1: ["Answer: no", "Answer: yes", "garbage"],  # mixed -> model chosen
        2: ["Answer: yes", "Answer: yes"],  # all right -> no pair
        3: ["Answer: no", "no answer"],  # all wrong -> reference chosen
    }
    p = {x["id"]: x for x in build_pairs(items, samples)}
    assert set(p) == {1, 3}
    assert p[1]["chosen"][0]["content"] == "Answer: yes"
    assert p[1]["rejected"][0]["content"] == "Answer: no"
    assert p[1]["chosen_source"] == "model"
    assert p[3]["chosen"][0]["content"] == "Reasoning: L\nAnswer: yes"
    assert p[3]["chosen_source"] == "reference"
