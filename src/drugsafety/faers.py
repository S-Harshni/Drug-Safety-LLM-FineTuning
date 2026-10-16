"""Check (drug, adverse effect) pairs against real post-market data from the FDA, the way a drug-safety team would.

- FAERS, the FDA Adverse Event Reporting System (via the openFDA API): about 20 million reports from doctors,
  pharmacists, patients and manufacturers. A pair is a *signal* when the effect is reported with the drug more often than
  with other drugs: proportional reporting ratio (PRR) >= 2, chi-squared >= 4 and at least 3 reports (Evans et al., 2001).
- FDA drug labels (openFDA): is the effect already written in the adverse reactions, warnings or boxed warning section?

Every API response is cached in data/faers_cache.json.gz, so results reproduce without calling the API again.
Without a key openFDA allows 1,000 requests a day and 100 terms per count; set OPENFDA_API_KEY (free) for more.
"""
import gzip
import json
import os
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "faers_cache.json.gz"
API = "https://api.fda.gov/drug"
LABEL_SECTIONS = ("boxed_warning", "warnings_and_cautions", "warnings", "adverse_reactions", "precautions", "contraindications")
# MedDRA preferred terms use British spelling.
UK = [("diarrhea", "diarrhoea"), ("edema", "oedema"), ("anemia", "anaemia"), ("hemorrhag", "haemorrhag"), ("leukemia", "leukaemia"),
      ("esophag", "oesophag"), ("hematoma", "haematoma"), ("hemolys", "haemolys"), ("hemoglobin", "haemoglobin"),
      ("hepatitis", "hepatitis"), ("tumor", "tumour"), ("estrogen", "oestrogen"), ("pediatric", "paediatric"), ("hypokalemia", "hypokalaemia"),
      ("hyperkalemia", "hyperkalaemia"), ("hyponatremia", "hyponatraemia"), ("hypernatremia", "hypernatraemia"), ("hypocalcemia", "hypocalcaemia"),
      ("hypercalcemia", "hypercalcaemia"), ("hypoglycemia", "hypoglycaemia"), ("hyperglycemia", "hyperglycaemia"), ("ischemi", "ischaemi"),
      ("septicemia", "septicaemia"), ("anesthe", "anaesthe"), ("dyspnea", "dyspnoea"), ("apnea", "apnoea"), ("hematuria", "haematuria"),
      ("leukopenia", "leukopenia"), ("thrombocytopenia", "thrombocytopenia"), ("neutropenia", "neutropenia")]


class Client:
    """openFDA GET with a persistent cache and polite pacing. Cached answers never touch the network."""

    def __init__(self, cache: Path = CACHE, offline: bool = False):
        self.path, self.offline, self.calls = cache, offline, 0
        self.cache = json.loads(gzip.decompress(cache.read_bytes())) if cache.exists() else {}

    def get(self, endpoint: str, **params) -> dict | None:
        key = endpoint + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        if key not in self.cache:
            if self.offline:
                return None
            if os.environ.get("OPENFDA_API_KEY"):
                params = {**params, "api_key": os.environ["OPENFDA_API_KEY"]}  # never part of the cache key
            for attempt in range(4):
                r = httpx.get(f"{API}/{endpoint}.json", params=params, timeout=60)
                if r.status_code == 429:
                    raise RuntimeError("openFDA daily limit reached; rerun tomorrow, the cache keeps what is done")
                if r.status_code == 403 or (r.status_code == 400 and "API_KEY" in r.text):
                    raise RuntimeError(f"openFDA refused the request: {r.text[:200]}")
                if r.status_code < 500:
                    break
                time.sleep(2 ** attempt)
            self.calls += 1
            self.cache[key] = r.json() if r.status_code == 200 else None  # 404 = nothing matches
            time.sleep(0.3)
        return self.cache[key]

    def save(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_bytes(gzip.compress(json.dumps(self.cache, separators=(",", ":")).encode()))


# Abbreviations and non-US names used in the case reports -> the US generic name openFDA harmonises to.
DRUG_ALIASES = {
    "paracetamol": "acetaminophen", "cbz": "carbamazepine", "tmp-smx": "sulfamethoxazole and trimethoprim",
    "cotrimoxazole": "sulfamethoxazole and trimethoprim", "cpa": "cyclophosphamide", "cyclosposphamide": "cyclophosphamide",
    "ifn": "interferon alfa-2b", "ifn-alpha": "interferon alfa-2b", "interferon alpha": "interferon alfa-2b",
    "interferon-alpha": "interferon alfa-2b", "interferon alfa": "interferon alfa-2b", "ppa": "phenylpropanolamine",
    "hu": "hydroxyurea", "mitomycin c": "mitomycin", "interleukin-2": "aldesleukin", "l-asparaginase": "asparaginase",
    "5-fu": "fluorouracil", "5-fluorouracil": "fluorouracil", "adriamycin": "doxorubicin", "6-tg": "thioguanine",
    "actinomycin d": "dactinomycin", "actinomycin-d": "dactinomycin", "az": "azathioprine", "mtx": "methotrexate",
    "adrenaline": "epinephrine", "salbutamol": "albuterol", "frusemide": "furosemide", "ciclosporin": "cyclosporine",
}


def clean(drug: str) -> str:
    d = re.sub(r'["\\]', "", drug.strip().lower())
    return DRUG_ALIASES.get(d, d)


def drug_query(drug: str) -> str:
    """Match the drug as a generic or brand name in openFDA's harmonised fields."""
    d = clean(drug)
    return f'(patient.drug.openfda.generic_name:"{d}" OR patient.drug.openfda.brand_name:"{d}")'


def reported_query(drug: str) -> str:
    """Fallback: the product name as written on the report (covers older and non-US drugs openFDA does not harmonise)."""
    return f'patient.drug.medicinalproduct:"{clean(drug)}"'


# Everyday wording -> the MedDRA preferred term FAERS uses. Applied the same way to every set of pairs.
SYNONYMS = {
    "FEVER": "PYREXIA", "ALLERGY": "HYPERSENSITIVITY", "ALLERGIC REACTION": "HYPERSENSITIVITY",
    "HYPERSENSITIVITY REACTION": "HYPERSENSITIVITY", "SKIN RASH": "RASH", "SKIN ERUPTION": "RASH", "PSYCHOSIS": "PSYCHOTIC DISORDER",
    "CONGESTIVE HEART FAILURE": "CARDIAC FAILURE CONGESTIVE", "HEART FAILURE": "CARDIAC FAILURE", "AML": "ACUTE MYELOID LEUKAEMIA",
    "LUNG CANCER": "LUNG NEOPLASM MALIGNANT", "HEPATOTOXIC EFFECT": "HEPATOTOXICITY", "HEPATOTOXIC REACTION": "HEPATOTOXICITY",
    "LIVER TOXICITY": "HEPATOTOXICITY", "LIVER DAMAGE": "LIVER INJURY", "TOXICITY": "TOXICITY TO VARIOUS AGENTS",
    "COGNITIVE IMPAIRMENT": "COGNITIVE DISORDER", "ADRENOCORTICAL INSUFFICIENCY": "ADRENAL INSUFFICIENCY",
    "HYPERPIGMENTATION": "SKIN HYPERPIGMENTATION", "ELECTROLYTE DISORDER": "ELECTROLYTE IMBALANCE", "SEIZURE": "SEIZURE",
    "CONVULSION": "SEIZURE", "KIDNEY FAILURE": "RENAL FAILURE", "LIVER FAILURE": "HEPATIC FAILURE", "BLEEDING": "HAEMORRHAGE",
    "HEART ATTACK": "MYOCARDIAL INFARCTION", "LOW BLOOD PRESSURE": "HYPOTENSION", "HIGH BLOOD PRESSURE": "HYPERTENSION",
    "MACROPHAGIC ACTIVATION SYNDROME": "HAEMOPHAGOCYTIC LYMPHOHISTIOCYTOSIS", "CARDIAC COLLAPSE": "CARDIOVASCULAR COLLAPSE",
}
MODIFIERS = ("WORSENING ", "SEVERE ", "FATAL ", "ACUTE ", "CHRONIC ", "DELAYED ", "BILATERAL ", "RAPID ONSET OF ", "TRANSIENT ",
             "DRUG-INDUCED ", "DRUG INDUCED ", "PERSISTENT ", "RECURRENT ", "PROFOUND ", "MARKED ", "MILD ")


def to_pt(effect: str) -> str:
    """Free-text effect -> the form of a MedDRA preferred term (upper case, British spelling)."""
    e = re.sub(r"\s+", " ", effect.strip().lower())
    for us, uk in UK:
        e = e.replace(us, uk)
    e = re.sub(r"(?<![a])emia\b", "aemia", e)
    return e.upper()


def candidates(effect: str) -> list[str]:
    """Preferred-term forms to try, most literal first: as written, without modifiers, singular, then synonyms."""
    pt = to_pt(effect)
    forms = [pt]
    stripped = re.sub(r"^[A-Z0-9-]+-INDUCED ", "", pt)
    changed = True
    while changed:
        changed = False
        for m in MODIFIERS:
            if stripped.startswith(m):
                stripped, changed = stripped[len(m):], True
    forms.append(stripped)
    for f in list(forms):
        if f.endswith("ES") and not f.endswith("SES"):
            forms += [f[:-1], f[:-2]]
        elif f.endswith("S") and not f.endswith(("SS", "SIS", "US")):
            forms.append(f[:-1])
    forms += [SYNONYMS[f] for f in list(forms) if f in SYNONYMS]
    return list(dict.fromkeys(forms))


def prr(a: int, drug_total: int, reaction_total: int, n: int) -> dict:
    """2x2 disproportionality: a = reports with drug and effect; drug_total = reports with the drug;
    reaction_total = reports with the effect; n = all reports."""
    b, c = drug_total - a, reaction_total - a
    d = n - a - b - c
    if a == 0 or min(b, c, d) <= 0:
        return {"a": a, "prr": 0.0, "prr_low": 0.0, "chi2": 0.0}
    ratio = (a / (a + b)) / (c / (c + d))
    se = (1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)) ** 0.5
    low = ratio * pow(2.718281828459045, -1.96 * se)
    expected = (a + b) * (a + c) / n
    chi2 = (abs(a - expected) - 0.5) ** 2 / expected if expected > 0 else 0.0  # Yates-corrected, one cell (Evans)
    return {"a": a, "prr": round(ratio, 3), "prr_low": round(low, 3), "chi2": round(chi2, 2)}


def is_signal(s: dict) -> bool:
    return s["a"] >= 3 and s["prr"] >= 2 and s["chi2"] >= 4


def label_text(client: Client, drug: str) -> str | None:
    d = clean(drug)
    got = client.get("label", search=f'(openfda.generic_name:"{d}" OR openfda.brand_name:"{d}")', limit=1)
    if not got or not got.get("results"):
        return None
    r = got["results"][0]
    return " ".join(" ".join(r.get(k, [])) for k in LABEL_SECTIONS).lower()


def on_label(text: str, effect: str, pt: str | None) -> bool:
    """Is the effect written in the label? Checks the effect as written, its US/UK spellings and the matched term."""
    forms = {effect.lower().strip(), to_pt(effect).lower()}
    if pt:
        forms.add(pt.lower())
    for us, uk in UK:
        forms |= {f.replace(uk, us) for f in list(forms)}
    return any(re.search(r"\b" + re.escape(f) + r"\b", text) for f in forms if len(f) >= 4)
