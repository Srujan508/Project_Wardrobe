"""Batched pair scoring: score every pair a benchmark needs in one vectorized pass,
then hand the metric code a cheap dictionary lookup."""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import torch

# vectorized: (rows_a, rows_b) LongTensors -> scores Tensor
BatchFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def needed_pairs(compat, fitb) -> set[tuple[str, str]]:
    pairs = set()
    for items, _ in compat:
        pairs.update(combinations(items, 2))
    for q in fitb:
        pairs.update((a, c) for a in q.question for c in q.candidates)
    return pairs


@torch.no_grad()
def batched_scorer(pairs: set[tuple[str, str]], row: dict[str, int], fn: BatchFn,
                   device: torch.device, batch: int = 65536):
    pairs = list(pairs)
    a = torch.tensor([row[x] for x, _ in pairs], device=device)
    b = torch.tensor([row[y] for _, y in pairs], device=device)
    out = np.concatenate([fn(a[i:i + batch], b[i:i + batch]).float().cpu().numpy()
                          for i in range(0, len(pairs), batch)])
    table = {p: float(s) for p, s in zip(pairs, out)}

    def score(x: str, y: str) -> float:
        s = table.get((x, y))
        return s if s is not None else table[(y, x)]
    return score
