"""Step 5 models.

Model A  CompatModel: one projector per category into a shared compatibility space,
         so a top and a bottom can land close together without looking alike.
Model B  PairScorer: an MLP on top of Model A's embeddings (+ optional color features),
         trained with BCE to output a compatibility logit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompatModel(nn.Module):
    def __init__(self, n_categories: int, d_in: int = 512, d_hidden: int = 256, d_out: int = 128):
        super().__init__()
        # Stacked per-category weights, so a batch with mixed categories is one gather + bmm.
        self.w1 = nn.Parameter(torch.empty(n_categories, d_in, d_hidden))
        self.b1 = nn.Parameter(torch.zeros(n_categories, d_hidden))
        self.w2 = nn.Parameter(torch.empty(n_categories, d_hidden, d_out))
        self.b2 = nn.Parameter(torch.zeros(n_categories, d_out))
        for w in (self.w1, self.w2):
            for c in range(n_categories):
                nn.init.kaiming_uniform_(w[c].T, a=5 ** 0.5)

    def forward(self, x: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        """x: (B, d_in) encoder embeddings; cat: (B,) category ids -> (B, d_out) unit vectors.
        Projects through every category's layer with one small matmul, then picks each row's own
        category. Same function as a per-row gather of weights, but without copying a full weight
        matrix per sample (which made training memory-bound)."""
        C, d_in, h = self.w1.shape
        rows = torch.arange(x.shape[0], device=x.device)
        h_all = (x @ self.w1.permute(1, 0, 2).reshape(d_in, C * h)).view(-1, C, h)
        hid = F.relu(h_all[rows, cat] + self.b1[cat])
        d_out = self.w2.shape[2]
        z_all = (hid @ self.w2.permute(1, 0, 2).reshape(h, C * d_out)).view(-1, C, d_out)
        return F.normalize(z_all[rows, cat] + self.b2[cat], dim=-1)


def triplet_loss(a: torch.Tensor, p: torch.Tensor, n: torch.Tensor, margin: float) -> torch.Tensor:
    """Margin loss on cosine distance (inputs are unit vectors)."""
    d_ap = 1 - (a * p).sum(-1)
    d_an = 1 - (a * n).sum(-1)
    return F.relu(d_ap - d_an + margin).mean()


def batch_hard_triplet_loss(a, p, n, pos_cat, outfit, margin: float, pos_item=None,
                            semi_hard: bool = False) -> torch.Tensor:
    """For each anchor, the negative is the hardest of: its sampled negative, or any other
    in-batch positive of the SAME category as its positive, from a DIFFERENT outfit, and not
    the same item as its own positive (items can appear in several outfits).
    Known gap: an in-batch positive that appeared with this anchor in some OTHER outfit is not
    masked (needs per-batch CPU lookups; measured ~0.17% of anchors on Polyvore train)."""
    sim = a @ p.T                                                     # (B, B) anchor vs all positives
    valid = (pos_cat[None, :] == pos_cat[:, None]) & (outfit[None, :] != outfit[:, None])
    if pos_item is not None:
        valid &= pos_item[None, :] != pos_item[:, None]
    if semi_hard:
        # FaceNet semi-hard: only negatives still FARTHER than the positive. Pure batch-hard from
        # a cold start collapses every embedding to one point (loss stuck at the margin).
        valid &= sim < (a * p).sum(-1, keepdim=True)
    sim = sim.masked_fill(~valid, -2.0)
    hardest_in_batch = sim.max(dim=1).values
    sampled = (a * n).sum(-1)
    hardest = torch.maximum(hardest_in_batch, sampled)
    d_ap = 1 - (a * p).sum(-1)
    d_an = 1 - hardest
    return F.relu(d_ap - d_an + margin).mean()


class PairScorer(nn.Module):
    def __init__(self, d: int = 128, n_color: int = 0, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.n_color = n_color
        self.mlp = nn.Sequential(nn.Linear(d * 4 + n_color, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, a, b, color_feats=None):
        x = [a, b, a * b, (a - b).abs()]
        if self.n_color:
            x.append(color_feats)
        return self.mlp(torch.cat(x, -1)).squeeze(-1)

    def symmetric(self, a, b, ca=None, cb=None):
        """Order-independent score: average both directions."""
        return 0.5 * (self(a, b, ca) + self(b, a, cb if cb is not None else ca))
