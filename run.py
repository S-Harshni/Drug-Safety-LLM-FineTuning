"""All fine-tuning experiments. Each finished run is saved, so the script can be stopped and resumed.

    python run.py            # writes results/runs/*.json, then results/results.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from drugsafety import data, finetune, metrics, prompts  # noqa: E402

RUNS = ROOT / "results" / "runs"
EPOCHS, SEED = 2, 0
SIZES, RANKS, DEFAULT_RANK, RANK_STUDY_SIZE = (250, 1000, 4000), (2, 8, 32), 8, 1000


def saved(name: str, compute):
    path = RUNS / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    print(f"--- {name}", flush=True)
    result = compute()
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k not in ("scores", "outputs")}), flush=True)
    return result


sample = data.sample


def detection_report(labels, validation_labels, validation_scores, scores, seconds: float | None = None) -> dict:
    threshold = metrics.best_threshold(validation_labels, validation_scores)
    predicted = np.asarray(scores) >= threshold
    out = {**metrics.binary(labels, predicted), "f1_interval": metrics.f1_interval(labels, predicted),
           "roc_auc": round(float(roc_auc_score(labels, scores)), 4), "threshold": round(threshold, 4)}
    if seconds is not None:
        out["ms_per_sentence"] = round(1000 * seconds / len(scores), 2)
    return out


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    data.download()
    sets = data.evaluation_sets()
    detect, extract, train, validation, test = (sets[k] for k in ("detect", "extract", "train", "validation", "test"))
    x_train, x_test = sets["extract_train"], sets["extract_test"]
    detect_prompt = lambda text, shots=(): prompts.messages(prompts.DETECT_SYSTEM, text, shots)   # noqa: E731
    extract_prompt = lambda text, shots=(): prompts.messages(prompts.EXTRACT_SYSTEM, text, shots)  # noqa: E731
    expected_pairs = [{(d.lower(), e.lower()) for d, e in pairs} for pairs in x_test.pairs]

    def baseline() -> dict:
        vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        model = LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(vectoriser.fit_transform(train.text), train.label)
        t0 = time.time()
        scores = model.decision_function(vectoriser.transform(test.text))
        seconds = time.time() - t0
        return {"method": "TF-IDF + logistic regression", "train_sentences": len(train),
                **detection_report(test.label, validation.label, model.decision_function(vectoriser.transform(validation.text)), scores, seconds)}

    tokenizer, base = finetune.load()

    def score(model, shots=()) -> tuple[list[float], list[float], float]:
        val = finetune.yes_no_scores(model, tokenizer, [detect_prompt(t, shots) for t in validation.text])
        t0 = time.time()
        got = finetune.yes_no_scores(model, tokenizer, [detect_prompt(t, shots) for t in test.text])
        return val, got, time.time() - t0

    def untuned(shots, label) -> dict:
        val, got, seconds = score(base, shots)
        return {"method": label, "train_sentences": 0, **detection_report(test.label, validation.label, val, got, seconds)}

    def lora_detection(size: int, rank: int) -> dict:
        nonlocal base
        subset = sample(train, size, seed=SEED + 1)
        model = finetune.add_lora(base, rank)
        parameters = finetune.count_parameters(model)
        log = finetune.train(model, tokenizer, [(detect_prompt(t), "yes" if y else "no") for t, y in zip(subset.text, subset.label, strict=True)],
                             epochs=EPOCHS, seed=SEED)
        val, got, seconds = score(model)
        if (size, rank) == (max(SIZES), DEFAULT_RANK):
            model.save_pretrained(ROOT / "out" / "detection_adapter")
        base = model.unload()
        report = detection_report(test.label, validation.label, val, got, seconds)
        return {"method": "LoRA fine-tuned", "train_sentences": size, "rank": rank, **parameters, "training": log, **report,
                "scores": [round(s, 3) for s in got]}

    def pair_scores(outputs: list[str], seconds: float) -> dict:
        return {**metrics.pairs(expected_pairs, [prompts.parse_pairs(o) for o in outputs]),
                "ms_per_sentence": round(1000 * seconds / len(outputs), 1), "outputs": outputs}

    def untuned_extraction() -> dict:
        t0 = time.time()
        outputs = finetune.generate(base, tokenizer, [extract_prompt(t, prompts.EXTRACT_SHOTS) for t in x_test.text])
        return {"method": "Base model, 4 examples in the prompt", "train_sentences": 0, **pair_scores(outputs, time.time() - t0)}

    def lora_extraction() -> dict:
        nonlocal base
        model = finetune.add_lora(base, DEFAULT_RANK)
        parameters = finetune.count_parameters(model)
        log = finetune.train(model, tokenizer, [(extract_prompt(t), prompts.format_pairs(p)) for t, p in zip(x_train.text, x_train.pairs, strict=True)],
                             epochs=EPOCHS, seed=SEED)
        t0 = time.time()
        outputs = finetune.generate(model, tokenizer, [extract_prompt(t) for t in x_test.text])
        seconds = time.time() - t0
        model.save_pretrained(ROOT / "out" / "extraction_adapter")
        base = model.unload()
        return {"method": "LoRA fine-tuned", "train_sentences": len(x_train), "rank": DEFAULT_RANK, **parameters, "training": log,
                **pair_scores(outputs, seconds)}

    results = {
        "baseline": saved("detect_baseline", baseline),
        "zero_shot": saved("detect_base_zero_shot", lambda: untuned((), "Base model, no examples")),
        "few_shot": saved("detect_base_few_shot", lambda: untuned(prompts.DETECT_SHOTS, "Base model, 6 examples in the prompt")),
        "extract_few_shot": saved("extract_base_few_shot", untuned_extraction),
        "extract_lora": saved("extract_lora", lora_extraction),
        "sizes": [saved(f"detect_lora_n{n}_r{DEFAULT_RANK}", lambda n=n: lora_detection(n, DEFAULT_RANK)) for n in SIZES],
        "ranks": [saved(f"detect_lora_n{RANK_STUDY_SIZE}_r{r}", lambda r=r: lora_detection(RANK_STUDY_SIZE, r)) for r in RANKS],
    }
    setup = {
        "base_model": finetune.BASE_MODEL, "lora_targets": finetune.TARGETS, "epochs": EPOCHS, "device": finetune.device(),
        "detection": {"sentences": len(detect), "positive_share": round(float(detect.label.mean()), 4),
                      "train": int((detect.split == "train").sum()), "validation_used": len(validation), "test_used": len(test),
                      "test_positive_share": round(float(test.label.mean()), 4)},
        "extraction": {"sentences": len(extract), "train": len(x_train), "test_used": len(x_test),
                       "pairs_per_sentence": round(float(extract.pairs.map(len).mean()), 2)},
    }
    (ROOT / "results" / "results.json").write_text(json.dumps({"setup": setup, **results}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
