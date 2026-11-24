"""Loaders for the official Polyvore Outfits splits (Vasileva et al., ECCV 2018).

File formats (from the reference implementation, mvasil/fashion-compatibility):
  <split>.json                 list of {"set_id": str, "items": [{"item_id": str, "index": int}, ...]}
  compatibility_<split>.txt    "<label> <setid_index> <setid_index> ..."   label 1 = compatible
  fill_in_blank_<split>.json   list of {"question": [setid_index...], "answers": [setid_index x4], ...}
                               the correct answer is the candidate whose set_id matches the question's
  polyvore_item_metadata.json  {item_id: {"semantic_category": ..., "title": ..., "url_name": ...}}

Questions reference items as "setid_index"; we always resolve them to item_id.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.utils import load_config, project_path

SPLITS = ("train", "valid", "test")


def polyvore_root(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return project_path(cfg["paths"]["polyvore_root"])


def split_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return polyvore_root(cfg) / cfg["paths"]["split"]


@dataclass
class FitbQuestion:
    question: list[str]      # item_ids of the partial outfit
    candidates: list[str]    # item_ids of the 4 candidates
    answer: int              # index of the correct candidate


def load_outfits(split: str, cfg: dict | None = None) -> list[dict]:
    with open(split_dir(cfg) / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def setid_index_map(outfits: list[dict]) -> dict[str, str]:
    """Map "setid_index" -> item_id for one split."""
    return {f"{o['set_id']}_{it['index']}": it["item_id"] for o in outfits for it in o["items"]}


@lru_cache(maxsize=1)
def load_metadata(root: str) -> dict[str, dict]:
    with open(Path(root) / "polyvore_item_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def item_categories(cfg: dict | None = None) -> dict[str, str]:
    """item_id -> semantic_category (tops, bottoms, shoes, outerwear, bags, jewellery, ...)."""
    meta = load_metadata(str(polyvore_root(cfg)))
    return {iid: m.get("semantic_category", "unknown") for iid, m in meta.items()}


def load_compatibility(split: str, cfg: dict | None = None) -> list[tuple[list[str], int]]:
    """Official outfit-compatibility questions: (item_ids, label)."""
    id_map = setid_index_map(load_outfits(split, cfg))
    out = []
    with open(split_dir(cfg) / f"compatibility_{split}.txt", "r", encoding="utf-8") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            out.append(([id_map[t] for t in tok[1:]], int(tok[0])))
    return out


def load_fitb(split: str, cfg: dict | None = None) -> list[FitbQuestion]:
    """Official fill-in-the-blank questions with the correct candidate identified by set_id."""
    id_map = setid_index_map(load_outfits(split, cfg))
    with open(split_dir(cfg) / f"fill_in_blank_{split}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for q in data:
        gt_set = q["question"][0].split("_")[0]
        correct = [i for i, a in enumerate(q["answers"]) if a.split("_")[0] == gt_set]
        assert len(correct) == 1, f"expected exactly one correct FITB answer, got {correct}"
        out.append(FitbQuestion(
            question=[id_map[t] for t in q["question"]],
            candidates=[id_map[a] for a in q["answers"]],
            answer=correct[0],
        ))
    return out
