"""Run the fine-tuned adapters on sentences written for the demo page. None of them is from the corpus.

    python run.py && python demo_examples.py        # writes results/demo_examples.json

The corpus itself is not republished; these sentences show the models on text they have never seen.
"""
import json
import sys
from pathlib import Path

import httpx
from peft import PeftModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from drugsafety import finetune, prompts  # noqa: E402

BIG_MODEL, URL = "qwen2.5:3b", "http://localhost:11434/v1/chat/completions"

# (sentence, 1 if it describes an adverse drug event)
DETECTION = [
    ("A 62-year-old woman developed severe hyponatremia three days after starting sertraline.", 1),
    ("The patient received amoxicillin for ten days and recovered fully.", 0),
    ("Rhabdomyolysis occurred in a man taking simvastatin together with clarithromycin.", 1),
    ("Metformin is the first-line treatment for type 2 diabetes.", 0),
    ("We report a case of lithium-induced nephrogenic diabetes insipidus.", 1),
    ("Blood cultures were negative and the chest radiograph was normal.", 0),
    ("Two weeks after the first infusion of infliximab she presented with a lupus-like rash.", 1),
    ("Warfarin was continued at the same dose after discharge.", 0),
    ("Hair loss resolved when valproate was withdrawn.", 1),
    ("The tumour was resected and adjuvant cisplatin was planned.", 0),
    ("Acute pancreatitis associated with azathioprine therapy in a patient with Crohn's disease.", 1),
    ("Fever and cough improved after treatment with levofloxacin.", 0),
    ("Tendon rupture has been described in elderly patients receiving ciprofloxacin.", 1),
    ("She had a history of hypertension treated with amlodipine.", 0),
    ("Visual hallucinations began shortly after the dose of ropinirole was increased.", 1),
    ("The study enrolled 120 adults with moderate asthma.", 0),
]

# (sentence, the (drug, effect) pairs a careful reader would mark)
EXTRACTION = [
    ("A 62-year-old woman developed severe hyponatremia three days after starting sertraline.", [("sertraline", "severe hyponatremia")]),
    ("Rhabdomyolysis occurred in a man taking simvastatin together with clarithromycin.",
     [("simvastatin", "rhabdomyolysis"), ("clarithromycin", "rhabdomyolysis")]),
    ("We report a case of lithium-induced nephrogenic diabetes insipidus.", [("lithium", "nephrogenic diabetes insipidus")]),
    ("Hair loss resolved when valproate was withdrawn.", [("valproate", "hair loss")]),
    ("Acute pancreatitis associated with azathioprine therapy in a patient with Crohn's disease.", [("azathioprine", "acute pancreatitis")]),
    ("Visual hallucinations and confusion began shortly after the dose of ropinirole was increased.",
     [("ropinirole", "visual hallucinations"), ("ropinirole", "confusion")]),
    ("Tendon rupture has been described in elderly patients receiving ciprofloxacin.", [("ciprofloxacin", "tendon rupture")]),
    ("Two weeks after the first infusion of infliximab she presented with a lupus-like rash.", [("infliximab", "lupus-like rash")]),
    ("Methotrexate was stopped because of pneumonitis and mouth ulcers.", [("methotrexate", "pneumonitis"), ("methotrexate", "mouth ulcers")]),
    ("Gingival overgrowth is a recognised complication of phenytoin.", [("phenytoin", "gingival overgrowth")]),
]


def ask_big_model(sentence: str) -> list | None:
    """The larger prompted model, if a local Ollama server is running; otherwise the column is left out."""
    try:
        reply = httpx.post(URL, timeout=120, json={
            "model": BIG_MODEL, "messages": prompts.messages(prompts.EXTRACT_SYSTEM, sentence, prompts.EXTRACT_SHOTS),
            "temperature": 0, "seed": 0, "max_tokens": 96})
        reply.raise_for_status()
        return sorted(prompts.parse_pairs(reply.json()["choices"][0]["message"]["content"]))
    except httpx.HTTPError:
        return None


def main() -> None:
    tokenizer, base = finetune.load()
    best = max((json.loads(p.read_text()) for p in (ROOT / "results" / "runs").glob("detect_lora_n*_r8.json")), key=lambda r: r["train_sentences"])

    model = PeftModel.from_pretrained(base, ROOT / "out" / "detection_adapter")
    scores = finetune.yes_no_scores(model, tokenizer, [prompts.messages(prompts.DETECT_SYSTEM, text) for text, _ in DETECTION])
    base = model.unload()
    detection = [{"text": text, "label": label, "score": round(score, 3), "predicted": int(score >= best["threshold"])}
                 for (text, label), score in zip(DETECTION, scores, strict=True)]

    sentences = [text for text, _ in EXTRACTION]
    untuned = finetune.generate(base, tokenizer, [prompts.messages(prompts.EXTRACT_SYSTEM, t, prompts.EXTRACT_SHOTS) for t in sentences])
    model = PeftModel.from_pretrained(base, ROOT / "out" / "extraction_adapter")
    tuned = finetune.generate(model, tokenizer, [prompts.messages(prompts.EXTRACT_SYSTEM, t) for t in sentences])
    extraction = []
    for (text, pairs), before, after in zip(EXTRACTION, untuned, tuned, strict=True):
        row = {"text": text, "expected": sorted([d.lower(), e.lower()] for d, e in pairs),
               "tuned": sorted(prompts.parse_pairs(after)), "base": sorted(prompts.parse_pairs(before))}
        big = ask_big_model(text)
        if big is not None:
            row["prompted"] = big
        extraction.append(row)

    (ROOT / "results" / "demo_examples.json").write_text(json.dumps({"detection": detection, "extraction": extraction}, indent=1))
    right = sum(d["label"] == d["predicted"] for d in detection)
    print(f"detection: {right} of {len(detection)} right; extraction: "
          f"{sum(sorted(map(list, r['tuned'])) == r['expected'] for r in extraction)} of {len(extraction)} sentences fully right")


if __name__ == "__main__":
    main()
