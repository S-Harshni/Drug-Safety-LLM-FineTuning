import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from drugsafety import data, metrics, prompts  # noqa: E402


def test_split_depends_only_on_the_sentence():
    assert data.bucket("Aspirin caused  bleeding.") == data.bucket("aspirin caused bleeding.")      # case and spacing ignored
    buckets = pd.Series([data.bucket(f"sentence number {i}") for i in range(5000)]).value_counts(normalize=True)
    assert abs(buckets["train"] - 0.7) < 0.03 and abs(buckets["test"] - 0.2) < 0.03 and abs(buckets["validation"] - 0.1) < 0.03


def test_repeated_sentences_are_merged_before_splitting(tmp_path, monkeypatch):
    frame = pd.DataFrame({"text": ["Drug A caused rash.", "drug a caused  rash.", "No event here."], "label": [0, 1, 0]})
    frame.to_parquet(tmp_path / "classification.parquet")
    pairs = pd.DataFrame({"text": ["Drug A caused rash and fever."] * 3, "drug": ["Drug A"] * 3, "effect": ["rash", "fever", "rash"]})
    pairs.to_parquet(tmp_path / "drug_ade_relation.parquet")
    monkeypatch.setattr(data, "DATA", tmp_path)
    detect = data.detection()
    assert len(detect) == 2 and detect.label.tolist() == [1, 0]               # one copy kept; positive if any copy was
    extract = data.extraction()
    assert len(extract) == 1 and extract.pairs[0] == [("Drug A", "fever"), ("Drug A", "rash")]


def test_binary_scores_on_a_hand_worked_case():
    labels, predicted = [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0]        # 2 hits, 1 miss, 1 false alarm
    assert metrics.binary(labels, predicted) == {"precision": 0.6667, "recall": 0.6667, "f1": 0.6667, "accuracy": 0.75}
    assert metrics.binary([0, 0], [0, 0])["f1"] == 0.0


def test_best_threshold_separates_a_separable_case():
    labels, scores = [0] * 50 + [1] * 50, list(np.linspace(-2, -0.1, 50)) + list(np.linspace(0.1, 2, 50))
    threshold = metrics.best_threshold(labels, scores)
    assert -0.1 < threshold <= 0.1 + 1e-9
    assert metrics.binary(labels, np.asarray(scores) >= threshold)["f1"] == 1.0


def test_bootstrap_interval_brackets_the_score():
    rng = np.random.default_rng(0)
    labels = rng.random(400) < 0.3
    predicted = np.where(rng.random(400) < 0.85, labels, ~labels)
    low, high = metrics.f1_interval(labels, predicted, draws=300)
    assert low < metrics.binary(labels, predicted)["f1"] < high and high - low < 0.2


def test_pair_scores_need_both_drug_and_effect():
    expected = [{("aspirin", "bleeding")}, {("warfarin", "necrosis"), ("warfarin", "rash")}]
    found = [{("aspirin", "bleeding")}, {("warfarin", "necrosis"), ("heparin", "rash")}]
    assert metrics.pairs(expected, found) == {"precision": 0.6667, "recall": 0.6667, "f1": 0.6667, "sentences_fully_right": 0.5}
    assert metrics.pairs(expected, [set(), set()])["f1"] == 0.0


def test_pairs_round_trip_through_text():
    pairs = [("warfarin", "skin necrosis"), ("heparin", "thrombocytopenia")]
    assert prompts.parse_pairs(prompts.format_pairs(pairs)) == set(pairs)
    assert prompts.parse_pairs("Here are the events:\n- Warfarin | Skin necrosis\nnothing else | \n| x\nno pipe") == {("warfarin", "skin necrosis")}


def test_prompts_put_examples_before_the_sentence():
    m = prompts.messages(prompts.EXTRACT_SYSTEM, "X caused Y.", prompts.EXTRACT_SHOTS)
    assert m[0]["role"] == "system" and m[-1] == {"role": "user", "content": "X caused Y."}
    assert len(m) == 2 + 2 * len(prompts.EXTRACT_SHOTS) and m[2]["content"] == "isoniazid | acute liver failure"
    assert len(prompts.messages(prompts.DETECT_SYSTEM, "X")) == 2


def test_lora_trains_only_the_adapter_and_can_be_removed():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import GPT2Config, GPT2LMHeadModel

    from drugsafety import finetune

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=16, n_embd=16, n_layer=1, n_head=2))
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ids = torch.randint(0, 50, (2, 8))
    reference = model(ids).logits.detach().clone()
    finetune.TARGETS, saved = ["c_attn"], finetune.TARGETS
    try:
        tuned = finetune.add_lora(model, rank=2)
    finally:
        finetune.TARGETS = saved
    counts = finetune.count_parameters(tuned)
    assert 0 < counts["trainable"] < 0.05 * counts["total"]
    assert all("lora" in name for name, p in tuned.named_parameters() if p.requires_grad)
    tuned.eval()
    assert torch.allclose(tuned(ids).logits, reference, atol=1e-6)          # a fresh adapter changes nothing: B starts at zero
    restored = tuned.unload()
    assert all(torch.equal(v, before[k]) for k, v in restored.state_dict().items())


def test_training_labels_cover_only_the_answer():
    pytest.importorskip("torch")
    from drugsafety import finetune

    class Tok:
        eos_token, pad_token_id = "<e>", 0

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return " ".join(m["content"] for m in messages)

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [len(w) for w in text.replace("<e>", " <e>").split()]}

    ids, labels = finetune.encode(Tok(), [{"role": "user", "content": "one two three"}], "yes")
    assert ids == [3, 3, 5, 3, 3] and labels == [-100, -100, -100, 3, 3]
    batch_ids, mask, batch_labels = finetune.collate(Tok(), [(ids, labels), ([7], [7])])
    assert batch_ids.tolist()[1] == [0, 0, 0, 0, 7] and mask.tolist()[1] == [0, 0, 0, 0, 1] and batch_labels.tolist()[1] == [-100] * 4 + [7]
