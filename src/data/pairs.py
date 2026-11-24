"""Step 3: positive pairs, random negatives, hard negatives, and the training Dataset.

Training runs on cached encoder embeddings (Step 4), so the Dataset returns
embedding row indices, not images. Negatives are re-sampled every epoch.

Rules:
  * positive  = two items of DIFFERENT categories from the same outfit
  * negative  = an item of the POSITIVE's category from a different outfit
                (never a different category, otherwise the model learns "top vs bottom")
  * hard neg  = close to the positive in embedding space, but skipping the closest
                `skip` items, which are often equally good matches (false negatives)
"""
from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
from torch.utils.data import Dataset


def positive_pairs(outfits: list[dict], cat: dict[str, str], keep: set[str] | None = None,
                   max_pairs: int | None = None, seed: int = 42) -> list[tuple[str, str, str]]:
    """Return (anchor, positive, outfit_id) for all cross-category pairs, both directions."""
    pairs = []
    for o in outfits:
        ids = [it["item_id"] for it in o["items"] if keep is None or cat.get(it["item_id"]) in keep]
        for a, b in combinations(ids, 2):
            if cat.get(a) != cat.get(b):
                pairs.append((a, b, o["set_id"]))
                pairs.append((b, a, o["set_id"]))
    if max_pairs and len(pairs) > max_pairs:
        pairs = random.Random(seed).sample(pairs, max_pairs)
    return pairs


class PairIndex:
    """Lookup tables shared by negative samplers."""

    def __init__(self, outfits: list[dict], cat: dict[str, str], emb_index: dict[str, int]):
        self.emb_index = emb_index
        self.cat = cat
        self.item_outfits: dict[str, set[str]] = defaultdict(set)
        self.by_cat: dict[str, list[str]] = defaultdict(list)
        seen = set()
        for o in outfits:
            for it in o["items"]:
                iid = it["item_id"]
                self.item_outfits[iid].add(o["set_id"])
                if iid not in seen and iid in emb_index:
                    self.by_cat[cat.get(iid, "unknown")].append(iid)
                    seen.add(iid)
        self.categories = sorted(self.by_cat)
        self.cat_id = {c: i for i, c in enumerate(self.categories)}

    def co_occurs(self, anchor: str, candidate: str) -> bool:
        """True if the two items ever appeared in the same outfit (items can be in several)."""
        return bool(self.item_outfits[anchor] & self.item_outfits[candidate])

    def random_negative(self, anchor: str, pos_cat: str, rng: random.Random) -> str:
        """An item of the positive's category that never appeared in an outfit with the anchor."""
        pool = self.by_cat[pos_cat]
        for _ in range(50):
            n = rng.choice(pool)
            if not self.co_occurs(anchor, n):
                return n
        raise RuntimeError(f"could not sample a negative for category {pos_cat}")


@torch.no_grad()
def mine_hard_negatives(pairs: list[tuple[str, str, str]], index: PairIndex, emb: np.ndarray,
                        topk: int = 20, skip: int = 5, device: str = "cpu",
                        seed: int = 42) -> list[str]:
    """For each (anchor, positive) pick a negative among the positive's nearest same-category
    neighbours, ranks [skip, skip+topk), excluding anything that ever co-occurred with the anchor.
    The `skip` closest are never used (they are often equally good matches); if too few remain,
    fall back to a random negative."""
    rng = random.Random(seed)
    E = torch.from_numpy(emb).to(device)
    out: list[str | None] = [None] * len(pairs)
    by_cat_pairs: dict[str, list[int]] = defaultdict(list)
    for i, (_, p, _) in enumerate(pairs):
        by_cat_pairs[index.cat[p]].append(i)
    for c, pair_ids in by_cat_pairs.items():
        pool = index.by_cat[c]
        pool_rows = torch.tensor([index.emb_index[x] for x in pool], device=device)
        P = E[pool_rows]                                                   # (M, D)
        k = min(len(pool), skip + topk + 10)                               # +10 slack for same-outfit hits
        for start in range(0, len(pair_ids), 4096):
            chunk = pair_ids[start:start + 4096]
            q = E[torch.tensor([index.emb_index[pairs[i][1]] for i in chunk], device=device)]
            nn_idx = (q @ P.T).topk(k, dim=1).indices.cpu().numpy()
            for row, i in enumerate(chunk):
                anchor, pos, _ = pairs[i]
                cands = [pool[j] for j in nn_idx[row]
                         if pool[j] != pos and not index.co_occurs(anchor, pool[j])]
                cands = cands[skip:skip + topk]
                out[i] = rng.choice(cands) if cands else index.random_negative(anchor, c, rng)
    return out  # type: ignore[return-value]


class TripletEmbeddingDataset(Dataset):
    """Yields (anchor_row, pos_row, neg_row, anchor_cat_id, pos_cat_id, outfit_id) as tensors.
    outfit_id lets in-batch mining avoid using items from the anchor's own outfit as negatives."""

    def __init__(self, pairs, index: PairIndex, hard_negatives: list[str] | None = None,
                 hard_fraction: float = 0.0, seed: int = 42):
        self.pairs, self.index = pairs, index
        self.outfit_id = {o: i for i, o in enumerate(sorted({p[2] for p in pairs}))}
        self.hard = hard_negatives
        self.hard_fraction = hard_fraction if hard_negatives is not None else 0.0
        self.rng = random.Random(seed)
        self.resample()

    def resample(self) -> None:
        """Draw fresh negatives; call once per epoch."""
        negs = []
        for i, (a, p, _) in enumerate(self.pairs):
            if self.hard is not None and self.rng.random() < self.hard_fraction:
                negs.append(self.hard[i])
            else:
                negs.append(self.index.random_negative(a, self.index.cat[p], self.rng))
        self.negs = negs

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The whole epoch as 6 index tensors (same content as __getitem__), for fast batching."""
        ei, cid, cat = self.index.emb_index, self.index.cat_id, self.index.cat
        a = torch.tensor([ei[x] for x, _, _ in self.pairs])
        p = torch.tensor([ei[y] for _, y, _ in self.pairs])
        n = torch.tensor([ei[z] for z in self.negs])
        ca = torch.tensor([cid[cat[x]] for x, _, _ in self.pairs])
        cp = torch.tensor([cid[cat[y]] for _, y, _ in self.pairs])
        oid = torch.tensor([self.outfit_id[o] for _, _, o in self.pairs])
        return a, p, n, ca, cp, oid

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a, p, outfit = self.pairs[index]
        ei, cid, cat = self.index.emb_index, self.index.cat_id, self.index.cat
        return (torch.tensor(ei[a]), torch.tensor(ei[p]), torch.tensor(ei[self.negs[index]]),
                torch.tensor(cid[cat[a]]), torch.tensor(cid[cat[p]]), torch.tensor(self.outfit_id[outfit]))


class TensorBatches:
    """Iterate a TripletEmbeddingDataset in shuffled batches by slicing index tensors
    (no per-item Python work). Re-reads the dataset's current negatives on every epoch."""

    def __init__(self, ds: TripletEmbeddingDataset, batch_size: int, drop_last: bool = True,
                 device: torch.device | str = "cpu", seed: int = 42):
        self.ds, self.bs, self.drop_last, self.device = ds, batch_size, drop_last, device
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        n = len(self.ds)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def __iter__(self):
        T = [t.to(self.device) for t in self.ds.tensors()]
        perm = torch.randperm(len(self.ds), generator=self.gen).to(self.device)
        for i in range(len(self)):
            idx = perm[i * self.bs:(i + 1) * self.bs]
            yield tuple(t[idx] for t in T)
