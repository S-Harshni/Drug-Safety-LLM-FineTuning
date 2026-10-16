"""Real-world check of the fine-tuned model: are the (drug, effect) pairs it extracts from case reports confirmed by
FDA adverse event reports (FAERS), and are they already on the FDA label?

    python faers_signals.py            # calls openFDA only for what is not in data/faers_cache.json.gz
    python faers_signals.py --offline  # cache only

Three sets of pairs from the same 400 held-out sentences are checked the same way:
- model:   pairs written by the LoRA fine-tuned extraction model (results/runs/extract_lora.json);
- human:   the pairs annotators marked in those sentences;
- control: each model pair with its effect swapped for one from a different sentence, a pairing nobody wrote.
If the model's pairs are real drug-safety knowledge, they should look like the human pairs and unlike the controls.

The result also triages the model's pairs the way a safety team would: known (FAERS signal and on the label),
candidate (FAERS signal but not on the label: worth a safety review) and unconfirmed.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from drugsafety import data, faers, prompts  # noqa: E402

REACTIONS = "patient.reaction.reactionmeddrapt.exact"
CALL_BUDGET = 950   # openFDA allows 1,000 requests a day without a key
SENTENCES = 120     # the first 120 of the 400 held-out sentences (a fixed sample), to fit that budget
TOP = 100           # most reported terms per count without a key


def pair_sets() -> dict[str, list[tuple[str, str]]]:
    test = data.evaluation_sets()["extract_test"]
    outputs = json.loads((ROOT / "results" / "runs" / "extract_lora.json").read_text())["outputs"]
    # sorted: parse_pairs returns a set, whose order changes between runs, and the shuffle below must be reproducible
    norm = lambda pairs: sorted({(d.strip().lower(), e.strip().lower()) for d, e in pairs if d.strip() and e.strip()})  # noqa: E731
    model = [norm(prompts.parse_pairs(o)) for o in outputs[:SENTENCES]]
    human = [norm(p) for p in list(test.pairs)[:SENTENCES]]
    rng = random.Random(0)
    control = []
    for i, pairs in enumerate(model):
        out = []
        for d, _ in pairs:
            while True:
                j = rng.randrange(len(model))
                if j != i and model[j] and all(dd != d for dd, _ in model[j]):
                    out.append((d, rng.choice(model[j])[1]))
                    break
        control.append(out)
    unique = lambda sets: sorted({p for s in sets for p in s})  # noqa: E731
    return {"model": unique(model), "human": unique(human), "control": unique(control)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--budget", type=int, default=CALL_BUDGET, help="most API calls this run may make")
    args = parser.parse_args()
    client = faers.Client(offline=args.offline)
    sets = pair_sets()
    drugs = sorted({d for pairs in sets.values() for d, _ in pairs})

    n_reports = client.get("event", limit=1)["meta"]["results"]["total"]
    global_rx = {r["term"]: r["count"] for r in client.get("event", count=REACTIONS, limit=TOP)["results"]}
    info = {}
    for drug in drugs:
        q = faers.drug_query(drug)
        total = client.get("event", search=q, limit=1)
        if not total:
            q = faers.reported_query(drug)
            total = client.get("event", search=q, limit=1)
        if not total:
            info[drug] = None
            continue
        rx = client.get("event", search=q, count=REACTIONS, limit=TOP)
        info[drug] = {"total": total["meta"]["results"]["total"], "query": q,
                      "rx": {r["term"]: r["count"] for r in (rx or {}).get("results", [])},
                      "label": faers.label_text(client, drug)}
        client.save()
    print(f"{sum(v is not None for v in info.values())} of {len(drugs)} drugs found in FAERS; {client.calls} API calls so far", flush=True)

    reaction_total = dict(global_rx)
    over = lambda: client.calls >= args.budget and not args.offline  # noqa: E731

    def count(search: str) -> int | None:
        """Number of reports matching `search`; None when it would exceed today's budget."""
        if over() and ("event?limit=1&search=" + search) not in client.cache:
            return None
        got = client.get("event", search=search, limit=1)
        return got["meta"]["results"]["total"] if got else 0

    def total_for(pt: str) -> int | None:
        if pt not in reaction_total:
            got = count(f'{REACTIONS}:"{pt}"')
            if got is None:
                return None
            reaction_total[pt] = got
        return reaction_total[pt] or None

    def check(drug: str, effect: str) -> dict:
        d = info.get(drug)
        if d is None:
            return {"drug": drug, "effect": effect, "status": "drug not in FAERS"}
        options = faers.candidates(effect)
        pt = next((c for c in options if c in d["rx"] or c in global_rx), None) or next((c for c in options if total_for(c)), None)
        if pt is None:
            return {"drug": drug, "effect": effect, "status": "not resolved today" if over() else "effect not a FAERS term"}
        a = d["rx"].get(pt)
        if a is None:  # not among the drug's most reported terms: ask for the exact count
            a = count(f'{d["query"]} AND {REACTIONS}:"{pt}"')
        reaction = total_for(pt)
        if a is None or reaction is None:
            return {"drug": drug, "effect": effect, "status": "not resolved today"}
        s = faers.prr(a, d["total"], reaction, n_reports)
        labelled = None if d["label"] is None else faers.on_label(d["label"], effect, pt)
        return {"drug": drug, "effect": effect, "term": pt, "status": "checked", **s, "signal": faers.is_signal(s), "on_label": labelled}

    checked = {name: [check(d, e) for d, e in pairs] for name, pairs in sets.items()}
    client.save()

    def summary(rows: list[dict]) -> dict:
        ok = [r for r in rows if r["status"] == "checked"]
        lab = [r for r in ok if r["on_label"] is not None]
        sig = sum(r["signal"] for r in ok)
        return {"pairs": len(rows), "status": dict(Counter(r["status"] for r in rows)), "checked": len(ok),
                "signal": sig, "signal_share": round(sig / max(1, len(ok)), 4),
                "with_label": len(lab), "on_label": sum(r["on_label"] for r in lab),
                "on_label_share": round(sum(r["on_label"] for r in lab) / max(1, len(lab)), 4),
                "known": sum(r["signal"] and r["on_label"] for r in lab),
                "candidate": sum(r["signal"] and not r["on_label"] for r in lab),
                "label_only": sum(r["on_label"] and not r["signal"] for r in lab),
                "neither": sum(not r["signal"] and not r["on_label"] for r in lab),
                "median_prr": round(float(np.median([r["prr"] for r in ok])), 2) if ok else None}

    model_rows = [r for r in checked["model"] if r["status"] == "checked" and r["on_label"] is not None]
    pick = lambda cond: sorted([r for r in model_rows if cond(r)], key=lambda r: -r["prr"])[:12]  # noqa: E731
    out = {"source": {"sentences": SENTENCES, "faers_reports": n_reports, "faers_last_updated": client.get("event", limit=1)["meta"]["last_updated"],
                      "drugs": len(drugs), "drugs_in_faers": sum(v is not None for v in info.values()),
                      "drugs_with_label": sum(bool(v and v["label"]) for v in info.values())},
           "summary": {name: summary(rows_) for name, rows_ in checked.items()},
           "examples": {"known": pick(lambda r: r["signal"] and r["on_label"]), "candidate": pick(lambda r: r["signal"] and not r["on_label"])}}
    (ROOT / "results" / "faers.json").write_text(json.dumps(out, indent=1))
    for name, s in out["summary"].items():
        print(f"{name:8s} checked {s['checked']}/{s['pairs']}  signal {s['signal_share']:.1%}  on label {s['on_label_share']:.1%}  "
              f"known {s['known']} candidate {s['candidate']} median PRR {s['median_prr']}")
    print("API calls this run:", client.calls)


if __name__ == "__main__":
    main()
