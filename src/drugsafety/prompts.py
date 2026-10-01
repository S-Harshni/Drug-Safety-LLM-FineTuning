"""The instructions every model gets: the fine-tuned one, the prompted one, with or without examples."""

DETECT_SYSTEM = ("You review sentences from medical case reports for drug safety. "
                 "Answer yes if the sentence describes an adverse event caused by a drug, otherwise answer no. "
                 "Answer with one word: yes or no.")

EXTRACT_SYSTEM = ("You review sentences from medical case reports for drug safety. "
                  "List every adverse drug event in the sentence, one per line, as: drug | adverse effect. "
                  "Copy the drug and the effect exactly as they are written in the sentence. No other text.")

# Worked examples for few-shot prompting, written for this project (they are not corpus sentences).
DETECT_SHOTS = [
    ("A 54-year-old man developed acute liver failure two weeks after starting isoniazid.", "yes"),
    ("The patient was treated with intravenous ceftriaxone for seven days.", "no"),
    ("Severe hyponatraemia associated with carbamazepine therapy in an elderly woman.", "yes"),
    ("Magnetic resonance imaging showed no abnormality.", "no"),
    ("Symptoms resolved after amiodarone was withdrawn, suggesting drug-induced thyrotoxicosis.", "yes"),
    ("Methotrexate remains the first-line treatment for this condition.", "no"),
]
EXTRACT_SHOTS = [
    ("A 54-year-old man developed acute liver failure two weeks after starting isoniazid.", [("isoniazid", "acute liver failure")]),
    ("Severe hyponatraemia and confusion associated with carbamazepine therapy.",
     [("carbamazepine", "confusion"), ("carbamazepine", "hyponatraemia")]),
    ("We describe a case of warfarin-induced skin necrosis.", [("warfarin", "skin necrosis")]),
    ("Seizures occurred in two children receiving theophylline and ciprofloxacin.",
     [("ciprofloxacin", "Seizures"), ("theophylline", "Seizures")]),
]


def format_pairs(pairs) -> str:
    return "\n".join(f"{drug} | {effect}" for drug, effect in pairs)


def parse_pairs(text: str) -> set[tuple[str, str]]:
    """Read 'drug | effect' lines back; anything else the model wrote is ignored."""
    found = set()
    for line in text.splitlines():
        parts = [p.strip().strip("-*• ").strip() for p in line.split("|")]
        if len(parts) == 2 and all(parts):
            found.add((parts[0].lower(), parts[1].lower()))
    return found


def messages(system: str, sentence: str, shots=()) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for text, answer in shots:
        out += [{"role": "user", "content": text},
                {"role": "assistant", "content": answer if isinstance(answer, str) else format_pairs(answer)}]
    return out + [{"role": "user", "content": sentence}]
