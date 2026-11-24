"""Build a tiny synthetic dataset with the exact Polyvore Outfits file layout.

Each outfit has a hidden "style". An item's embedding = its category direction
+ a category-specific rotation of the style + noise. So items of one outfit are
compatible, but raw cosine across categories is weak: the per-category projections
of Model A must learn to align them. A working pipeline should beat 0.5 AUC clearly.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CATS = ["tops", "bottoms", "shoes", "outerwear", "bags"]


def build(root: Path, n_styles: int = 12, outfits_per_split=(400, 80, 80), dim: int = 32, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    styles = rng.normal(size=(n_styles, dim))
    cat_dir = {c: rng.normal(size=dim) * 2 for c in CATS}
    rot = {c: np.linalg.qr(rng.normal(size=(dim, dim)))[0] for c in CATS}

    poly = root / "raw" / "polyvore_outfits"
    (poly / "disjoint").mkdir(parents=True)
    proc = root / "processed"
    proc.mkdir(parents=True)
    meta, emb, ids, rows = {}, [], [], []
    counter = 0

    for split, n_out in zip(("train", "valid", "test"), outfits_per_split):
        outfits = []
        for k in range(n_out):
            s = rnd.randrange(n_styles)
            cats = rnd.sample(CATS, rnd.randint(3, 4))
            items = []
            for j, c in enumerate(cats):
                counter += 1
                iid = str(counter)
                meta[iid] = {"semantic_category": c, "title": "", "url_name": ""}
                v = cat_dir[c] + rot[c] @ styles[s] + rng.normal(size=dim) * 0.3
                emb.append(v / np.linalg.norm(v)); ids.append(iid)
                rows.append({"item_id": iid, "split": split, "category": c, "path": "", "valid": True})
                items.append({"item_id": iid, "index": j + 1, "_style": s, "_cat": c})
            outfits.append({"set_id": f"{split}{k}", "items": items})

        by_cat = {}
        for o in outfits:
            for it in o["items"]:
                by_cat.setdefault(it["_cat"], []).append((o["set_id"], it))
        compat_lines, fitb = [], []
        for o in outfits:
            ref = [f"{o['set_id']}_{it['index']}" for it in o["items"]]
            compat_lines.append("1 " + " ".join(ref))
            # negative outfit: same categories, items drawn from random other outfits
            neg = []
            for it in o["items"]:
                sid, other = rnd.choice([x for x in by_cat[it["_cat"]] if x[0] != o["set_id"]])
                neg.append(f"{sid}_{other['index']}")
            compat_lines.append("0 " + " ".join(neg))
            blank = rnd.randrange(len(o["items"]))
            correct = o["items"][blank]
            wrong = [x for x in by_cat[correct["_cat"]] if x[0] != o["set_id"]]
            answers = [f"{o['set_id']}_{correct['index']}"] + [f"{sid}_{it['index']}" for sid, it in rnd.sample(wrong, 3)]
            rnd.shuffle(answers)
            fitb.append({"question": [r for i, r in enumerate(ref) if i != blank], "answers": answers,
                         "blank_position": blank + 1})
        clean = [{"set_id": o["set_id"], "items": [{"item_id": it["item_id"], "index": it["index"]}
                                                    for it in o["items"]]} for o in outfits]
        (poly / "disjoint" / f"{split}.json").write_text(json.dumps(clean))
        (poly / "disjoint" / f"compatibility_{split}.txt").write_text("\n".join(compat_lines) + "\n")
        (poly / "disjoint" / f"fill_in_blank_{split}.json").write_text(json.dumps(fitb))

    (poly / "polyvore_item_metadata.json").write_text(json.dumps(meta))
    np.save(proc / "embeddings.npy", np.asarray(emb, dtype=np.float32))
    (proc / "embedding_ids.txt").write_text("\n".join(ids))
    pd.DataFrame(rows).to_csv(proc / "items.csv", index=False)

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "config.yaml").read_text())
    cfg["paths"] |= {"polyvore_root": str(poly), "processed": str(proc),
                     "models": str(root / "models"), "results": str(root / "results")}
    cfg["logging"] |= {"experiments_file": str(root / "results" / "experiments.jsonl"),
                       "comparison_file": str(root / "results" / "comparison.csv")}
    cfg["train"] |= {"batch_size": 64, "epochs": 15, "num_positive_pairs": 5000}
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path
