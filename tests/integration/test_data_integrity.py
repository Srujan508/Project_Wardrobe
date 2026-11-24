"""Integrity of the real Polyvore Outfits download, items.csv and the FashionCLIP embedding cache."""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.data.polyvore import SPLITS, load_compatibility, load_fitb, load_outfits

pytestmark = pytest.mark.integration

OFFICIAL_OUTFITS = {"train": 16995, "valid": 3000, "test": 15145}   # disjoint split


@pytest.fixture(scope="module")
def outfits(split_dir, real_cfg):
    return {s: load_outfits(s, real_cfg) for s in SPLITS}


@pytest.fixture(scope="module")
def questions(split_dir, real_cfg):
    """Parsed once: {split: (compat questions, FITB questions)} via the project loaders.
    load_fitb itself asserts exactly one candidate shares the question's set_id."""
    return {s: (load_compatibility(s, real_cfg), load_fitb(s, real_cfg)) for s in SPLITS}


@pytest.fixture(scope="module")
def referenced_items(outfits, questions) -> set[str]:
    """Every item id referenced by any outfit, compat question or FITB question, in any split."""
    ids = set()
    for s in SPLITS:
        ids |= {it["item_id"] for o in outfits[s] for it in o["items"]}
        compat, fitb = questions[s]
        for items, _ in compat:
            ids.update(items)
        for q in fitb:
            ids.update(q.question + q.candidates)
    return ids


# ------------------------------------------------------------------ official splits
@pytest.mark.parametrize("split", SPLITS)
def test_official_outfit_counts(outfits, split):
    assert len(outfits[split]) == OFFICIAL_OUTFITS[split]
    assert len({o["set_id"] for o in outfits[split]}) == OFFICIAL_OUTFITS[split], "duplicate set_ids"
    assert all(len(o["items"]) >= 2 for o in outfits[split])


@pytest.mark.parametrize("split", SPLITS)
def test_compat_questions_are_twice_the_outfits(questions, split):
    compat = questions[split][0]
    assert len(compat) == 2 * OFFICIAL_OUTFITS[split]
    labels = [label for _, label in compat]
    assert set(labels) == {0, 1}
    assert labels.count(1) == labels.count(0) == OFFICIAL_OUTFITS[split]
    assert all(len(items) >= 2 for items, _ in compat)


@pytest.mark.parametrize("split", SPLITS)
def test_fitb_has_four_candidates_and_one_correct(questions, split):
    fitb = questions[split][1]          # loading already checked "exactly one correct" per question
    assert len(fitb) == OFFICIAL_OUTFITS[split]
    for q in fitb:
        assert len(q.candidates) == 4 and q.question
        assert 0 <= q.answer < 4
    # Known quirk of the official files: a handful of questions repeat an item among the
    # candidates (sometimes the correct one) or contain the answer in the question. Keep it rare.
    dup = sum(len(set(q.candidates)) < 4 for q in fitb)
    in_question = sum(q.candidates[q.answer] in q.question for q in fitb)
    assert dup / len(fitb) < 0.01, f"{dup} FITB questions with duplicate candidates"
    assert in_question / len(fitb) < 0.001, f"{in_question} FITB answers already in the question"
    assert len({q.answer for q in fitb}) > 1, "answer position never varies"


def test_train_and_test_items_barely_overlap(outfits):
    """The 'disjoint' split keeps train and test (almost) item-disjoint."""
    items = {s: {it["item_id"] for o in outfits[s] for it in o["items"]} for s in SPLITS}
    overlap = items["train"] & items["test"]
    assert len(overlap) / len(items["test"]) < 0.005, f"{len(overlap)} test items also appear in train"


# ------------------------------------------------------------------ items.csv
def test_items_csv_schema(items_df):
    assert list(items_df.columns) == ["item_id", "split", "splits", "category", "path", "valid"]
    assert items_df["item_id"].is_unique
    assert set(items_df["split"]) <= set(SPLITS)
    # `splits` lists every split an item appears in; `split` is the first of them
    all_splits = items_df["splits"].str.split(";")
    assert all(set(s) <= set(SPLITS) for s in all_splits)
    assert (items_df["split"] == all_splits.str[0]).all()


def test_every_referenced_item_has_valid_image_row_and_category(referenced_items, items_df, images_dir, metadata):
    df = items_df.set_index("item_id")
    missing = referenced_items - set(df.index)
    assert not missing, f"{len(missing)} referenced items missing from items.csv, e.g. {sorted(missing)[:5]}"

    sub = df.loc[sorted(referenced_items)]
    invalid = sub.index[~sub["valid"].astype(bool)]
    assert len(invalid) == 0, f"{len(invalid)} referenced items have invalid images, e.g. {list(invalid[:5])}"

    bad_cat = sub.index[sub["category"].isna() | sub["category"].isin(["", "unknown"])]
    assert len(bad_cat) == 0, f"{len(bad_cat)} referenced items have no category, e.g. {list(bad_cat[:5])}"
    wrong_cat = [i for i in sub.index[:5000] if metadata.get(i, {}).get("semantic_category") !=sub.at[i, "category"]]
    assert not wrong_cat, f"items.csv category disagrees with metadata for {wrong_cat[:5]}"

    on_disk = {e.name for e in os.scandir(images_dir)}
    no_file = [i for i in sub.index if f"{i}.jpg" not in on_disk]
    assert not no_file, f"{len(no_file)} images missing on disk, e.g. {no_file[:5]}"
    assert all(p.replace("\\", "/").endswith(f"/{i}.jpg") for i, p in zip(sub.index, sub["path"]))


# ------------------------------------------------------------------ embeddings
@pytest.fixture(scope="module")
def embeddings(embedding_files):
    npy, ids_file = embedding_files
    emb = np.load(npy)                  # ~300 MB, read once; random memmap reads are much slower
    ids = ids_file.read_text().split()
    return emb, ids


def test_embedding_rows_match_ids(embeddings, items_df, real_cfg):
    emb, ids = embeddings
    assert emb.ndim == 2 and emb.shape == (len(ids), real_cfg["encoder"]["embed_dim"])
    assert emb.dtype == np.float32
    assert len(set(ids)) == len(ids), "duplicate ids in embedding_ids.txt"
    valid = set(items_df.loc[items_df["valid"].astype(bool), "item_id"])
    assert set(ids) == valid, "embedding ids should be exactly the valid items of items.csv"


def test_embeddings_finite_and_l2_normalized(embeddings):
    emb, _ = embeddings
    for i in range(0, len(emb), 20000):
        chunk = emb[i:i + 20000]
        assert np.isfinite(chunk).all(), f"NaN/inf in embeddings rows {i}..{i + len(chunk)}"
        norms = np.linalg.norm(chunk, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), f"rows {i}..: norms in [{norms.min():.4f}, {norms.max():.4f}]"


def test_embeddings_are_not_degenerate(embeddings, items_df):
    """Same-category items should be closer than cross-category ones (catches a shuffled id file)."""
    emb, ids = embeddings
    cat = dict(zip(items_df["item_id"], items_df["category"]))
    rng = np.random.default_rng(0)
    rows = rng.choice(len(ids), size=2000, replace=False)
    E = emb[rows]
    cats = np.array([cat[ids[r]] for r in rows])
    S = E @ E.T
    same = cats[:, None] == cats[None, :]
    np.fill_diagonal(same, False)
    off = ~np.eye(len(E), dtype=bool)
    assert S[same].mean() > S[off & ~same].mean() + 0.05
    assert np.std(E, axis=0).mean() > 1e-3, "all embeddings are (nearly) identical"
