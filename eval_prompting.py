"""The alternative to fine-tuning: prompt a model six times larger, with and without examples.

    ollama pull qwen2.5:3b
    python eval_prompting.py        # writes results/prompting.json (replies are cached, so it can be resumed)

Scored on the first 500 sentences of the detection test sample and on the full extraction test sample.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from drugsafety import data, metrics, prompts  # noqa: E402

MODEL, URL, N_DETECT = "qwen2.5:3b", "http://localhost:11434/v1/chat/completions", 500
CACHE = ROOT / "results" / "prompting_cache.jsonl"


def main() -> None:
    sets = data.evaluation_sets()
    test, x_test = sets["test"].head(N_DETECT), sets["extract_test"]
    store = {r["key"]: r for r in map(json.loads, CACHE.read_text().splitlines())} if CACHE.exists() else {}
    client = httpx.Client(timeout=180)

    def chat(messages: list[dict], max_tokens: int) -> tuple[str, float]:
        key = hashlib.sha256(json.dumps([MODEL, messages, max_tokens]).encode()).hexdigest()
        if key not in store:
            t0 = time.time()
            for _ in range(3):
                try:
                    r = client.post(URL, json={"model": MODEL, "messages": messages, "temperature": 0, "seed": 0, "max_tokens": max_tokens})
                    r.raise_for_status()
                    reply = r.json()["choices"][0]["message"]["content"].strip()
                    break
                except httpx.HTTPError:
                    time.sleep(5)
            else:
                reply = ""
            store[key] = {"key": key, "reply": reply, "seconds": round(time.time() - t0, 3)}
            with CACHE.open("a") as f:
                f.write(json.dumps(store[key]) + "\n")
        return store[key]["reply"], store[key]["seconds"]

    out = {"model": MODEL, "runtime": "Ollama, 4-bit quantised, local", "detect_sentences": len(test), "extract_sentences": len(x_test), "detection": [], "extraction": []}
    for label, shots in (("No examples", ()), ("6 examples in the prompt", prompts.DETECT_SHOTS)):
        replies = [chat(prompts.messages(prompts.DETECT_SYSTEM, t, shots), 4) for t in test.text]
        predicted = [r.lower().startswith("yes") for r, _ in replies]
        unreadable = sum(not r.lower().startswith(("yes", "no")) for r, _ in replies)
        out["detection"].append({"method": label, **metrics.binary(test.label, predicted), "f1_interval": metrics.f1_interval(test.label, predicted),
                                 "unreadable_replies": unreadable, "ms_per_sentence": round(1000 * sum(s for _, s in replies) / len(replies), 1)})
        print(out["detection"][-1], flush=True)
    expected = [{(d.lower(), e.lower()) for d, e in pairs} for pairs in x_test.pairs]
    for label, shots in (("No examples", ()), ("4 examples in the prompt", prompts.EXTRACT_SHOTS)):
        replies = [chat(prompts.messages(prompts.EXTRACT_SYSTEM, t, shots), 96) for t in x_test.text]
        out["extraction"].append({"method": label, **metrics.pairs(expected, [prompts.parse_pairs(r) for r, _ in replies]),
                                  "ms_per_sentence": round(1000 * sum(s for _, s in replies) / len(replies), 1),
                                  "outputs": [r for r, _ in replies]})
        print({k: v for k, v in out["extraction"][-1].items() if k != "outputs"}, flush=True)
    (ROOT / "results" / "prompting.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
