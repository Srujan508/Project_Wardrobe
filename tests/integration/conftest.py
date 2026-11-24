"""Shared fixtures for the integration tests (real Polyvore data, real tagger, trained models).

Run:   venv\\Scripts\\python -m pytest -m integration -q

A plain `pytest` run skips these tests so the unit suite stays fast; select them with
`-m integration` (or set WARDROBE_INTEGRATION=1). Every test skips with a reason when
the data or model it needs does not exist yet.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pytest

from src.utils import ROOT, load_config, project_path

REAL_CONFIG = ROOT / "configs" / "config.yaml"
SPLIT_FILES = ("{s}.json", "compatibility_{s}.txt", "fill_in_blank_{s}.json")

# url_name keywords used to pick unambiguous real product photos per coarse category.
PICK_RULES = {
    "tops": (r"t-?shirt|tee|blouse|shirt|sweater|top",
             r"dress|jacket|coat|cardigan|blazer|vest|swim|bikini|bra|set|skirt|pant|short"),
    "bottoms": (r"jeans|skirt|trousers|pants|shorts",
                r"dress|set|jumpsuit|romper|top|shirt|jacket|swim|bikini"),
    "shoes": (r"sneakers?|boots?|sandals?|pumps?|heels?|loafers?",
              r"sock|bag|dress|jean|skirt|top|shirt|jacket|coat"),
    "outerwear": (r"jacket|coat|blazer",
                  r"vest|dress|skirt|pant|jean|shoe|boot|sneaker|bag|top|shirt"),
}


def pytest_collection_modifyitems(config, items):
    if "integration" in (config.getoption("markexpr") or "") or os.environ.get("WARDROBE_INTEGRATION"):
        return
    skip = pytest.mark.skip(reason="integration test: run with `pytest -m integration`")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)


# ------------------------------------------------------------------ config / data
@pytest.fixture(scope="session")
def real_cfg() -> dict:
    """The real project config (never $WARDROBE_CONFIG, which unit tests point at temp files)."""
    return load_config(REAL_CONFIG)


@pytest.fixture(scope="session")
def split_dir(real_cfg) -> Path:
    d = project_path(real_cfg["paths"]["polyvore_root"]) / real_cfg["paths"]["split"]
    missing = [f.format(s=s) for s in ("train", "valid", "test") for f in SPLIT_FILES
               if not (d / f.format(s=s)).exists()]
    if missing:
        pytest.skip(f"Polyvore split files missing under {d}: {missing[:3]} (run scripts.download_polyvore)")
    return d


@pytest.fixture(scope="session")
def images_dir(real_cfg) -> Path:
    d = project_path(real_cfg["paths"]["polyvore_root"]) / "images"
    if not d.is_dir():
        pytest.skip(f"{d} missing (run src.data.prepare)")
    return d


@pytest.fixture(scope="session")
def metadata(real_cfg) -> dict:
    p = project_path(real_cfg["paths"]["polyvore_root"]) / "polyvore_item_metadata.json"
    if not p.exists():
        pytest.skip(f"{p} missing")
    from src.data.polyvore import load_metadata
    return load_metadata(str(p.parent))


@pytest.fixture(scope="session")
def items_df(real_cfg) -> pd.DataFrame:
    p = project_path(real_cfg["paths"]["processed"]) / "items.csv"
    if not p.exists():
        pytest.skip(f"{p} missing (run src.data.prepare)")
    return pd.read_csv(p, dtype={"item_id": str})


@pytest.fixture(scope="session")
def embedding_files(real_cfg) -> tuple[Path, Path]:
    proc = project_path(real_cfg["paths"]["processed"])
    npy, ids = proc / "embeddings.npy", proc / "embedding_ids.txt"
    if not (npy.exists() and ids.exists()):
        pytest.skip("embeddings.npy / embedding_ids.txt not written yet (run scripts.embed_items)")
    return npy, ids


# ------------------------------------------------------------------ real images
@pytest.fixture(scope="session")
def sample_images(items_df, metadata, images_dir) -> dict[str, list[Path]]:
    """4 deterministic, unambiguous product photos per app category (picked by url_name keywords)."""
    out = {}
    valid = items_df[items_df["valid"]].sort_values("item_id")
    for cat, (inc, exc) in PICK_RULES.items():
        inc_re, exc_re = re.compile(rf"\b({inc})\b"), re.compile(rf"\b({exc})")
        picks = []
        for iid in valid.loc[valid["category"] == cat, "item_id"]:
            name = (metadata.get(iid, {}).get("url_name") or "").lower()
            if inc_re.search(name) and not exc_re.search(name) and (images_dir / f"{iid}.jpg").exists():
                picks.append(images_dir / f"{iid}.jpg")
            if len(picks) >= 40:
                break
        if len(picks) < 4:
            pytest.skip(f"not enough clearly named '{cat}' images to sample from")
        out[cat] = picks[::10][:4]       # spread the picks a little
    return out


# ------------------------------------------------------------------ trained models
def find_trained_model(cfg: dict) -> str | None:
    """Prefer config app.model; else any kind-B checkpoint; else kind A. Ignores the overfit `debug` run."""
    import torch

    mdir = project_path(cfg["paths"]["models"])
    if not mdir.is_dir():
        return None
    preferred = cfg["app"]["model"]
    if (mdir / f"{preferred}.pt").exists():
        return preferred
    kinds = {}
    for p in sorted(mdir.glob("*.pt")):
        if p.stem.startswith("debug"):
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:                 # half-written by a training run
            continue
        if ck.get("kind") == "B" and not (mdir / f"{ck.get('base')}.pt").exists():
            continue
        kinds.setdefault(ck.get("kind"), []).append(p.stem)
    for kind in ("B", "A"):
        if kinds.get(kind):
            # prefer the colour variant, then any
            return sorted(kinds[kind], key=lambda n: ("color" not in n, n))[0]
    return None


@pytest.fixture(scope="session")
def trained_model(real_cfg) -> str:
    name = find_trained_model(real_cfg)
    if name is None:
        pytest.skip(f"no trained model in {project_path(real_cfg['paths']['models'])} yet (run scripts.train)")
    return name
