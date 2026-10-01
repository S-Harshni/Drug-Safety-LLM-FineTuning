"""Scores, written out so each one can be tested on a hand-made case."""
import numpy as np


def binary(labels, predicted) -> dict:
    labels, predicted = np.asarray(labels).astype(bool), np.asarray(predicted).astype(bool)
    tp, fp, fn = int((labels & predicted).sum()), int((~labels & predicted).sum()), int((labels & ~predicted).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 4),
            "accuracy": round(float((labels == predicted).mean()), 4)}


def best_threshold(labels, scores) -> float:
    """The cut-off with the highest F1. Chosen on validation data, then applied unchanged to the test set."""
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    return float(max(candidates, key=lambda t: binary(labels, scores >= t)["f1"]))


def f1_interval(labels, predicted, draws: int = 1000, seed: int = 0) -> list[float]:
    """95% bootstrap interval for F1: how much the score would move with a different test sample."""
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(labels), (draws, len(labels)))
    values = [binary(labels[i], predicted[i])["f1"] for i in picks]
    return [round(float(np.quantile(values, q)), 4) for q in (0.025, 0.975)]


def pairs(expected: list[set], found: list[set]) -> dict:
    """Micro-averaged scores over (drug, effect) pairs; a pair counts only if both parts match exactly."""
    tp = sum(len(e & f) for e, f in zip(expected, found, strict=True))
    n_found, n_expected = sum(map(len, found)), sum(map(len, expected))
    precision, recall = tp / max(n_found, 1), tp / max(n_expected, 1)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 4),
            "sentences_fully_right": round(sum(e == f for e, f in zip(expected, found, strict=True)) / max(len(expected), 1), 4)}
