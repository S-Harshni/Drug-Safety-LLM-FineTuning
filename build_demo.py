"""Merge the experiment results into docs/data.json for the demo page.

    python run.py && python eval_prompting.py && python demo_examples.py && python build_demo.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from drugsafety import data  # noqa: E402


def slim(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ("scores", "outputs")}


def load_results() -> dict:
    """Every finished run in results/runs, so the page can be built while later runs are still training."""
    import run as experiments
    from drugsafety import finetune

    runs = ROOT / "results" / "runs"
    read = lambda name: json.loads((runs / f"{name}.json").read_text())  # noqa: E731
    done = lambda name: (runs / f"{name}.json").exists()  # noqa: E731
    sets = data.evaluation_sets()
    detect, extract = sets["detect"], sets["extract"]
    setup = {
        "base_model": finetune.BASE_MODEL, "lora_targets": finetune.TARGETS, "epochs": experiments.EPOCHS, "device": finetune.device(),
        "detection": {"sentences": len(detect), "positive_share": round(float(detect.label.mean()), 4),
                      "train": int((detect.split == "train").sum()), "validation_used": len(sets["validation"]),
                      "test_used": len(sets["test"]), "test_positive_share": round(float(sets["test"].label.mean()), 4)},
        "extraction": {"sentences": len(extract), "train": len(sets["extract_train"]), "test_used": len(sets["extract_test"]),
                       "pairs_per_sentence": round(float(extract.pairs.map(len).mean()), 2)},
    }
    size_names = [f"detect_lora_n{n}_r{experiments.DEFAULT_RANK}" for n in experiments.SIZES]
    rank_names = [f"detect_lora_n{experiments.RANK_STUDY_SIZE}_r{r}" for r in experiments.RANKS]
    return {
        "setup": setup, "baseline": read("detect_baseline"), "zero_shot": read("detect_base_zero_shot"),
        "few_shot": read("detect_base_few_shot"), "extract_few_shot": read("extract_base_few_shot"), "extract_lora": read("extract_lora"),
        "sizes": [read(n) for n in size_names if done(n)], "ranks": [read(n) for n in rank_names if done(n)],
    }


def main() -> None:
    results = load_results()
    prompting_file = ROOT / "results" / "prompting.json"
    prompting = json.loads(prompting_file.read_text()) if prompting_file.exists() else None
    # Sentences shown on the page are written for it (demo_examples.py); corpus sentences are not republished.
    examples_file = ROOT / "results" / "demo_examples.json"
    examples = json.loads(examples_file.read_text()) if examples_file.exists() else {"detection": [], "extraction": []}

    out = {
        "setup": results["setup"],
        "detection": {"baseline": slim(results["baseline"]), "zero_shot": slim(results["zero_shot"]), "few_shot": slim(results["few_shot"]),
                      "sizes": [slim(r) for r in results["sizes"]], "ranks": [slim(r) for r in results["ranks"]],
                      "prompted": [slim(r) for r in prompting["detection"]] if prompting else []},
        "extraction": {"few_shot": slim(results["extract_few_shot"]), "lora": slim(results["extract_lora"]),
                       "prompted": [slim(r) for r in prompting["extraction"]] if prompting else []},
        "prompted_model": prompting and {k: prompting[k] for k in ("model", "runtime", "detect_sentences", "extract_sentences")},
        "detect_examples": examples["detection"], "extract_examples": examples["extraction"],
    }
    (ROOT / "docs" / "data.json").write_text(json.dumps(out, separators=(",", ":")))
    print("written", (ROOT / "docs" / "data.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
