"""Real tagger (rembg + FashionCLIP zero-shot) on real Polyvore product photos of known category."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.color import NAMED_COLORS
from src.wardrobe_rules import APP_CATEGORIES, TYPES

pytestmark = pytest.mark.integration

PLAN = {"tops": 2, "bottoms": 2, "shoes": 1, "outerwear": 1}


@pytest.fixture(scope="module")
def tagged(sample_images):
    """Run the real tagger once on 6 images: [(true_category, path, TaggedItem)]."""
    from src.tagger import process_upload
    out = []
    for cat, k in PLAN.items():
        for p in sample_images[cat][:k]:
            with Image.open(p) as im:
                im.load()
                out.append((cat, p, process_upload(im, remove_bg=True)))
    return out


def test_coarse_category_mostly_right(tagged):
    hits = [(cat, t.category, t.type, p.name) for cat, p, t in tagged]
    correct = sum(c == pred for c, pred, _, _ in hits)
    assert correct >= 4, f"only {correct}/6 correct: {hits}"


def test_type_and_rules_are_consistent(tagged):
    for _, _, t in tagged:
        assert t.type in TYPES
        assert t.category in APP_CATEGORIES and t.category == TYPES[t.type].category
        assert (t.formality, t.warmth) == (TYPES[t.type].formality, TYPES[t.type].warmth)
        assert 0.0 < t.type_confidence <= 1.0


def test_embedding_is_512d_unit_vector(tagged):
    for _, _, t in tagged:
        assert t.embedding.shape == (512,) and t.embedding.dtype == np.float32
        assert np.isfinite(t.embedding).all()
        assert abs(float(np.linalg.norm(t.embedding)) - 1.0) < 1e-3


def test_color_is_valid(tagged):
    for _, _, t in tagged:
        assert t.color_name in NAMED_COLORS
        assert len(t.color_rgb) == 3 and all(0 <= v <= 255 for v in t.color_rgb)
        assert all(0.0 <= v <= 1.0 for v in t.color_hsv)


def test_clean_image_is_white_square(tagged):
    for _, _, t in tagged:
        im = t.clean_image
        assert im.mode == "RGB" and im.width == im.height
        corner = np.asarray(im)[:4, :4].reshape(-1, 3)
        assert (corner >= 240).all(), "background should be composited onto white"


def test_upload_embedding_close_to_cached_dataset_embedding(tagged, embedding_files):
    """The app's embedding of a product photo should match the offline cache for the same item
    (rembg + white canvas changes framing a little, so only a loose bound)."""
    npy, ids_file = embedding_files
    ids = ids_file.read_text().split()
    row = {iid: i for i, iid in enumerate(ids)}
    emb = np.load(npy, mmap_mode="r")
    sims = [float(np.asarray(emb[row[p.stem]]) @ t.embedding) for _, p, t in tagged if p.stem in row]
    assert sims, "sampled images are not in the embedding cache"
    assert np.median(sims) > 0.8, f"upload vs cached cosine: {np.round(sims, 3)}"
