"""Sanity checks for whichever trained compatibility model exists (skips if none yet)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.data.polyvore import load_compatibility
from src.eval.metrics import outfit_score
from src.eval.scoring import batched_scorer, needed_pairs

pytestmark = pytest.mark.integration

N_QUESTIONS = 300


@pytest.fixture(scope="module")
def scored(real_cfg, trained_model, embedding_files, split_dir):
    """(scorer, compat sample, pair-score lookup) for 300 random official valid compat questions."""
    from src.inference import CompatScorer, load_dataset_assets, make_row_fn

    scorer = CompatScorer(trained_model, real_cfg)
    emb, ids, row, cats, colors = load_dataset_assets(real_cfg)
    if scorer.use_color and colors is None:
        pytest.skip(f"{trained_model} uses color features but colors_hsv.npy is missing (run scripts.extract_colors)")
    fn, _ = make_row_fn(scorer, emb, cats, colors)
    compat = random.Random(0).sample(load_compatibility("valid", real_cfg), N_QUESTIONS)
    pairs = {p for p in needed_pairs(compat, []) if p[0] != p[1]}
    score = batched_scorer(pairs, row, fn, scorer.device)
    return scorer, compat, score, pairs, fn, len(ids)


def test_compatible_outfits_score_higher(scored, trained_model):
    _, compat, score, _, _, _ = scored
    s = np.array([outfit_score(items, score) for items, _ in compat])
    y = np.array([label for _, label in compat])
    assert 0 < y.sum() < len(y)
    assert np.isfinite(s).all()
    pos, neg = s[y == 1].mean(), s[y == 0].mean()
    assert pos > neg, f"{trained_model}: mean score label-1 {pos:.4f} <= label-0 {neg:.4f}"


def test_calibrated_probabilities_in_range_and_monotonic(scored):
    scorer, _, score, pairs, _, _ = scored
    raw = np.array([score(a, b) for a, b in pairs])
    grid = np.linspace(raw.min() - 3 * raw.std(), raw.max() + 3 * raw.std(), 200)
    for x in (raw, grid):
        p = scorer.calibrate(x)
        assert p.shape == x.shape
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        order = np.argsort(x, kind="stable")
        assert (np.diff(p[order]) >= -1e-9).all(), "calibrated probability decreases as the raw score rises"
    p = scorer.calibrate(grid)
    assert p[-1] > p[0], "calibration is flat: every outfit would get the same %"


def test_scores_are_symmetric(scored):
    """score(a, b) == score(b, a): the app and benchmarks treat pairs as unordered."""
    import torch

    scorer, _, _, _, fn, n = scored
    rng = np.random.default_rng(0)
    a = torch.as_tensor(rng.integers(0, n, 500), device=scorer.device)
    b = torch.as_tensor(rng.integers(0, n, 500), device=scorer.device)
    assert torch.allclose(fn(a, b).float(), fn(b, a).float(), atol=1e-4)
