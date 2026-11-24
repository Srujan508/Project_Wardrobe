import random

import numpy as np

from src.data.pairs import PairIndex, TripletEmbeddingDataset, mine_hard_negatives, positive_pairs

# 30 synthetic outfits, each with a top, bottom and shoe.
CATS = ["tops", "bottoms", "shoes"]
OUTFITS = [{"set_id": f"o{k}", "items": [{"item_id": f"{c}{k}", "index": j + 1} for j, c in enumerate(CATS)]}
           for k in range(30)]
CAT = {it["item_id"]: c for o in OUTFITS for it, c in zip(o["items"], CATS)}
EMB_INDEX = {iid: i for i, iid in enumerate(sorted(CAT))}
EMB = np.random.default_rng(0).normal(size=(len(EMB_INDEX), 16)).astype(np.float32)
EMB /= np.linalg.norm(EMB, axis=1, keepdims=True)


def test_positive_pairs_are_cross_category_and_symmetric():
    pairs = positive_pairs(OUTFITS, CAT)
    assert len(pairs) == 30 * 3 * 2
    assert all(CAT[a] != CAT[b] for a, b, _ in pairs)
    assert {(a, b) for a, b, _ in pairs} == {(b, a) for a, b, _ in pairs}


def test_random_negatives_match_positive_category_and_other_outfit():
    pairs = positive_pairs(OUTFITS, CAT)
    idx = PairIndex(OUTFITS, CAT, EMB_INDEX)
    ds = TripletEmbeddingDataset(pairs, idx)
    for (a, p, outfit), n in zip(pairs, ds.negs):
        assert CAT[n] == CAT[p]
        assert outfit not in idx.item_outfits[n]


def test_negatives_never_co_occur_with_anchor_in_any_outfit():
    # tops0 is reused in outfit o1, so bottoms1/shoes1 are compatible with it too.
    outfits = [dict(o) for o in OUTFITS[:4]]
    outfits[1] = {"set_id": "o1", "items": [{"item_id": "tops0", "index": 1},
                                            {"item_id": "bottoms1", "index": 2},
                                            {"item_id": "shoes1", "index": 3}]}
    idx = PairIndex(outfits, CAT, EMB_INDEX)
    pairs = [p for p in positive_pairs(outfits, CAT) if p[0] == "tops0"]
    rng = random.Random(0)
    for _ in range(200):
        for a, p, _ in pairs:
            n = idx.random_negative(a, CAT[p], rng)
            assert n not in {"bottoms0", "shoes0", "bottoms1", "shoes1"}
    for (a, p, _), n in zip(pairs, mine_hard_negatives(pairs, idx, EMB, topk=5, skip=0)):
        assert not idx.co_occurs(a, n)


def test_hard_negatives_skip_closest_and_never_fall_back_to_them():
    pairs = positive_pairs(OUTFITS, CAT)
    idx = PairIndex(OUTFITS, CAT, EMB_INDEX)
    # skip >= pool size: every candidate is "too close", so all must come from the random sampler
    hard = mine_hard_negatives(pairs, idx, EMB, topk=5, skip=1000)
    for (a, p, _), n in zip(pairs, hard):
        assert CAT[n] == CAT[p] and not idx.co_occurs(a, n)


def test_hard_negatives_valid_and_dataset_shapes():
    pairs = positive_pairs(OUTFITS, CAT)
    idx = PairIndex(OUTFITS, CAT, EMB_INDEX)
    hard = mine_hard_negatives(pairs, idx, EMB, topk=5, skip=2)
    for (a, p, outfit), n in zip(pairs, hard):
        assert CAT[n] == CAT[p] and n != p and outfit not in idx.item_outfits[n]
    ds = TripletEmbeddingDataset(pairs, idx, hard_negatives=hard, hard_fraction=0.5)
    a, p, n, ca, cp, oid = ds[0]
    assert a.ndim == p.ndim == n.ndim == ca.ndim == cp.ndim == oid.ndim == 0
    ds.resample()
    assert len(ds.negs) == len(pairs)


def test_tensor_batches_match_getitem():
    import torch
    from src.data.pairs import TensorBatches
    pairs = positive_pairs(OUTFITS, CAT)
    ds = TripletEmbeddingDataset(pairs, PairIndex(OUTFITS, CAT, EMB_INDEX))
    T = ds.tensors()
    for i in (0, 7, len(ds) - 1):
        assert all(int(t[i]) == int(g) for t, g in zip(T, ds[i]))
    batches = list(TensorBatches(ds, 32, drop_last=False))
    assert sum(len(b[0]) for b in batches) == len(ds)
    seen = torch.cat([b[0] * 10_000 + b[1] for b in batches]).sort().values
    assert torch.equal(seen, (T[0] * 10_000 + T[1]).sort().values)   # every pair exactly once
