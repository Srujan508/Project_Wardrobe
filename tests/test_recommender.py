import numpy as np

from src.recommender import Item, recommend
from src.wardrobe_rules import TYPES

rng = np.random.default_rng(0)


def make(i, t, hsv=(0.0, 0.0, 0.1)):
    info = TYPES[t]
    e = rng.normal(size=16).astype(np.float32)
    return Item(i, info.category, t, "black", hsv, info.formality, info.warmth, e / np.linalg.norm(e))


WARDROBE = [make(1, "t-shirt"), make(2, "sweater"), make(3, "button-up shirt"),
            make(4, "jeans"), make(5, "shorts"), make(6, "trousers"),
            make(7, "sneakers"), make(8, "boots"), make(9, "coat"), make(10, "blazer")]
CAT = {it.id: it.category for it in WARDROBE}


def test_valid_outfits_no_duplicates_or_two_tops():
    recs = recommend(WARDROBE, n=5)
    assert 1 <= len(recs) <= 5
    for r in recs:
        ids = r["item_ids"]
        cats = [CAT[i] for i in ids]
        assert len(set(ids)) == len(ids)
        assert cats.count("tops") == 1 and cats.count("bottoms") == 1
        assert cats.count("shoes") <= 1 and cats.count("outerwear") <= 1
        assert 0.0 <= r["score"] <= 1.0 and r["explanation"]


def test_hot_weather_excludes_outerwear_and_warm_items():
    for r in recommend(WARDROBE, occasion="any", weather="hot", n=5):
        for i in r["item_ids"]:
            it = next(x for x in WARDROBE if x.id == i)
            assert it.category != "outerwear" and it.warmth <= 3


def test_casual_filter_and_diversity():
    recs = recommend(WARDROBE, occasion="casual", n=3)
    for r in recs:
        assert all(next(x for x in WARDROBE if x.id == i).formality <= 2 for i in r["item_ids"])
    tops = [r["item_ids"][0] for r in recommend(WARDROBE, n=3)]
    assert len(set(tops)) > 1, "diversity penalty should vary the top"


def test_empty_when_missing_bottoms():
    assert recommend([it for it in WARDROBE if it.category != "bottoms"]) == []


def test_explanation_names_every_piece():
    for r in recommend(WARDROBE, n=3):
        for i in r["item_ids"]:
            it = next(x for x in WARDROBE if x.id == i)
            assert it.type in r["explanation"]


DRESSY = WARDROBE + [make(11, "dress"), make(12, "jumpsuit")]
CAT2 = {it.id: it.category for it in DRESSY}


def test_one_piece_outfits_are_valid():
    recs = recommend(DRESSY, n=12)
    assert any("all-body" in [CAT2[i] for i in r["item_ids"]] for r in recs), "dresses should be suggested"
    for r in recs:
        cats = [CAT2[i] for i in r["item_ids"]]
        two_piece = cats.count("tops") == 1 and cats.count("bottoms") == 1 and "all-body" not in cats
        one_piece = cats.count("all-body") == 1 and "tops" not in cats and "bottoms" not in cats
        assert two_piece or one_piece, cats
        assert len(set(r["item_ids"])) == len(r["item_ids"])


def test_dress_only_wardrobe_works():
    recs = recommend([make(1, "dress"), make(2, "heels")], n=3)
    assert recs and recs[0]["item_ids"] == [1, 2] and "dress" in recs[0]["explanation"]
    lone = recommend([make(1, "dress")], n=3)
    assert len(lone) == 1 and lone[0]["explanation"].count("one-piece") == 1
