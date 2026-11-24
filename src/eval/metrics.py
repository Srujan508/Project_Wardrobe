"""Evaluation metrics shared by every model (baselines and trained).

Every model is reduced to a single function:
    pair_score(item_a: str, item_b: str) -> float     (higher = more compatible)
Outfit-level scores are the mean over all item pairs, so the same metric code
works for random, color-histogram, ResNet, FashionCLIP and trained models.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

PairScore = Callable[[str, str], float]


def outfit_score(items: Sequence[str], pair_score: PairScore) -> float:
    pairs = list(combinations(items, 2))
    if not pairs:
        return 0.0
    return float(np.mean([pair_score(a, b) for a, b in pairs]))


def compat_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return float(roc_auc_score(labels, scores))


def fitb_candidate_scores(question: Sequence[str], candidates: Sequence[str], pair_score: PairScore) -> list[float]:
    """Score each candidate by its mean compatibility with the partial outfit."""
    return [float(np.mean([pair_score(q, c) for q in question])) for c in candidates]


def fitb_accuracy(questions: Iterable, pair_score: PairScore) -> float:
    """questions: objects with .question, .candidates, .answer (see src.data.polyvore.FitbQuestion).
    Credit = fraction of the top-scoring candidates that ARE the answer item. So ties are split
    evenly (slot order never biases the result), and when the official file lists the answer item
    twice (it happens in a few questions) picking either copy counts as correct."""
    correct, total = 0.0, 0
    for q in questions:
        s = np.asarray(fitb_candidate_scores(q.question, q.candidates, pair_score))
        top = np.flatnonzero(s == s.max())
        answer_item = q.candidates[q.answer]
        correct += sum(q.candidates[i] == answer_item for i in top) / len(top)
        total += 1
    return correct / max(total, 1)


def evaluate(pair_score: PairScore, compat_questions, fitb_questions) -> dict[str, float]:
    scores = [outfit_score(items, pair_score) for items, _ in compat_questions]
    labels = [label for _, label in compat_questions]
    return {
        "compat_auc": compat_auc(scores, labels),
        "fitb_acc": fitb_accuracy(fitb_questions, pair_score),
    }
