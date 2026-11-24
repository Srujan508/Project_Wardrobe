"""Step 9: outfit recommendation with occasion/weather filters, diversity and explanations.

    recommend(items, scorer, occasion="casual", weather="hot", n=5)

Outfit = top + bottom (+ shoes) (+ outerwear). Outfit score = mean calibrated
compatibility over all item pairs. Diversity: greedy pick, penalising outfits that
reuse an already-chosen top or bottom.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import torch

from src.color import is_neutral
from src.wardrobe_rules import OCCASIONS, WEATHER


@dataclass
class Item:
    id: int
    category: str
    type: str
    color_name: str
    color_hsv: tuple[float, float, float]
    formality: int
    warmth: int
    embedding: np.ndarray


# ------------------------------------------------------------------ pair scores
def _fallback_calibrate(raw: np.ndarray) -> np.ndarray:
    """Untrained fallback: squash raw FashionCLIP cosine (~0.2-0.9) into [0, 1]."""
    return 1 / (1 + np.exp(-(np.asarray(raw) - 0.5) * 10))


def pair_matrix(items: list[Item], scorer=None) -> np.ndarray:
    """N x N matrix of RAW pair scores. Outfits are ranked on the mean raw score (exactly what
    the benchmarks evaluate); calibration is applied afterwards, only for display."""
    if not items:
        return np.zeros((0, 0))
    E = np.stack([it.embedding for it in items]).astype(np.float32)
    if scorer is None:                      # untrained fallback: raw FashionCLIP cosine
        return E @ E.T
    cats = [it.category for it in items]
    Z = scorer.project(E, cats)
    C = torch.tensor([it.color_hsv for it in items], dtype=torch.float32, device=scorer.device)
    n = len(items)
    ia, ib = torch.meshgrid(torch.arange(n, device=scorer.device), torch.arange(n, device=scorer.device),
                            indexing="ij")
    ia, ib = ia.reshape(-1), ib.reshape(-1)
    return scorer.raw_scores(Z[ia], Z[ib], C[ia], C[ib]).float().cpu().numpy().reshape(n, n)


# ------------------------------------------------------------------ filters
def allowed(it: Item, occasion: str, weather: str) -> bool:
    lo, hi = OCCASIONS[occasion]
    (wlo, whi), outer_ok = WEATHER[weather]
    if it.category == "outerwear" and not outer_ok:
        return False
    return lo <= it.formality <= hi and wlo <= it.warmth <= whi


# ------------------------------------------------------------------ explanations
def color_relation(a: Item, b: Item) -> str:
    na, nb = is_neutral(a.color_hsv), is_neutral(b.color_hsv)
    if na and nb:
        return "an easy all-neutral pairing"
    if na or nb:
        return "a neutral that lets the color stand out"
    dh = abs(a.color_hsv[0] - b.color_hsv[0]); dh = min(dh, 1 - dh) * 360
    if dh < 35:
        return "analogous colors that blend smoothly"
    if dh > 150:
        return "complementary colors for contrast"
    return "a bold color mix"


def formality_note(items: list[Item]) -> str:
    f = [it.formality for it in items]
    if max(f) - min(f) >= 3:
        return "mixes dressy and casual pieces"
    return {1: "relaxed and casual", 2: "casual", 3: "smart casual", 4: "polished", 5: "formal"}[round(np.mean(f))]


def explain(outfit: list[Item], score: float) -> str:
    # colour story between the two main pieces: top + bottom, or one-piece + its first accessory
    parts = [color_relation(outfit[0], outfit[1]) if len(outfit) > 1 else "a one-piece look",
             formality_note(outfit)]
    pieces = " + ".join(f"{it.color_name} {it.type}" for it in outfit)   # base pieces, shoes, layer
    return f"{pieces}: " + ", ".join(parts) + f". {score * 100:.0f}% compatibility estimate."


# ------------------------------------------------------------------ recommend
def recommend(items: list[Item], scorer=None, occasion: str = "any", weather: str = "any", n: int = 5,
              reuse_penalty: float = 0.15) -> list[dict]:
    pool = [it for it in items if allowed(it, occasion, weather)]
    by_cat = {c: [i for i, it in enumerate(pool) if it.category == c]
              for c in ("tops", "bottoms", "all-body", "shoes", "outerwear")}
    # an outfit's base is EITHER (top, bottom) OR (one-piece,)
    bases = [(t, b) for t in by_cat["tops"] for b in by_cat["bottoms"]] + [(d,) for d in by_cat["all-body"]]
    if not bases:
        return []
    S = pair_matrix(pool, scorer)
    shoes = by_cat["shoes"] or [None]
    outer = (by_cat["outerwear"] + [None]) if by_cat["outerwear"] else [None]
    if weather == "cold" and by_cat["outerwear"]:
        outer = by_cat["outerwear"]          # cold: always add a layer when one is available

    cands = []
    for base, sh, ow in product(bases, shoes, outer):
        idx = [i for i in (*base, sh, ow) if i is not None]
        pairs = list(combinations(idx, 2))
        # a lone one-piece with no shoes/layer has no pairs to score: neutral, ranks last
        score = float(np.mean([S[a, b] for a, b in pairs])) if pairs else float("-inf")
        cands.append((score, base, sh, ow))
    cands.sort(key=lambda x: -x[0])

    calibrate = scorer.calibrate if scorer is not None else _fallback_calibrate
    finite = np.array([c[0] for c in cands if np.isfinite(c[0])])
    spread = float(finite.max() - finite.min()) if len(finite) > 1 else 1.0
    spread = spread or 1.0                                  # penalty in units of the score range
    chosen, used = [], {}                                  # reuse count per base piece
    remaining = cands[: max(2000, n * 50)]
    while remaining and len(chosen) < n:
        def adjusted(c):
            return c[0] - reuse_penalty * spread * sum(used.get(i, 0) for i in c[1])
        best = max(remaining, key=adjusted)
        remaining.remove(best)
        raw_score, base, sh, ow = best
        score = float(calibrate(np.array([raw_score]))[0]) if np.isfinite(raw_score) else 0.5
        for i in base:
            used[i] = used.get(i, 0) + 1
        outfit = [pool[i] for i in (*base, sh, ow) if i is not None]
        chosen.append({
            "item_ids": [it.id for it in outfit],
            "score": round(score, 4),
            "explanation": explain(outfit, score),
        })
    return chosen
