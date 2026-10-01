"""Load the ADE corpus, remove repeated sentences and split it the same way every time.

The corpus (Gurulingappa et al., 2012) is 23,516 sentences from MEDLINE case reports, each marked as
describing an adverse drug event or not, plus the (drug, effect) pairs in the sentences that do.
"""
import hashlib
import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BASE = "https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2/resolve/main"
FILES = {"classification": "Ade_corpus_v2_classification", "drug_ade_relation": "Ade_corpus_v2_drug_ade_relation"}


def download() -> None:
    DATA.mkdir(exist_ok=True)
    for name, folder in FILES.items():
        target = DATA / f"{name}.parquet"
        if not target.exists():
            urllib.request.urlretrieve(f"{BASE}/{folder}/train-00000-of-00001.parquet", target)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def bucket(text: str) -> str:
    """train / validation / test, decided by a hash of the sentence so it never depends on row order."""
    slot = int(hashlib.sha256(normalise(text).encode()).hexdigest(), 16) % 10
    return "test" if slot < 2 else "validation" if slot == 2 else "train"


def detection() -> pd.DataFrame:
    """One row per distinct sentence: text, label (1 = describes an adverse drug event), split."""
    frame = pd.read_parquet(DATA / "classification.parquet")
    frame["key"] = frame["text"].map(normalise)
    frame = frame.groupby("key", sort=True).agg(text=("text", "first"), label=("label", "max")).reset_index(drop=True)
    frame["split"] = frame["text"].map(bucket)
    return frame


def extraction() -> pd.DataFrame:
    """One row per distinct sentence: text, pairs (sorted list of (drug, effect)), split."""
    frame = pd.read_parquet(DATA / "drug_ade_relation.parquet")
    frame["key"] = frame["text"].map(normalise)
    frame["pair"] = list(zip(frame["drug"].str.strip(), frame["effect"].str.strip(), strict=True))
    frame = frame.groupby("key", sort=True).agg(text=("text", "first"), pairs=("pair", lambda p: sorted(set(p)))).reset_index(drop=True)
    frame["split"] = frame["text"].map(bucket)
    return frame


N_VALIDATION, N_TEST, N_EXTRACT_TEST, SEED = 600, 2000, 400, 0


def sample(frame: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    return frame.sample(n=min(n, len(frame)), random_state=seed).reset_index(drop=True)


def evaluation_sets() -> dict:
    """The fixed samples every method is scored on, so all numbers are comparable."""
    detect, extract = detection(), extraction()
    return {
        "detect": detect, "extract": extract,
        "train": detect[detect.split == "train"],
        "validation": sample(detect[detect.split == "validation"], N_VALIDATION),
        "test": sample(detect[detect.split == "test"], N_TEST),
        "extract_train": sample(extract[extract.split == "train"], 10**9),
        "extract_test": sample(extract[extract.split == "test"], N_EXTRACT_TEST),
    }
