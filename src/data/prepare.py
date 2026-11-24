"""Step 1: turn the raw Polyvore Outfits download into data/processed/items.csv.

  1. Extract images from the parquet files to <polyvore_root>/images/<item_id>.jpg (once).
  2. Collect every item referenced by the official split files (outfits, compat, FITB).
  3. Attach the semantic category from polyvore_item_metadata.json.
  4. Verify each image opens; mark broken/missing ones invalid.
  5. Write items.csv (item_id, split, category, path, valid) and print split stats.

Run:  venv\\Scripts\\python -m src.data.prepare
"""
from __future__ import annotations

import io
import os
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.polyvore import SPLITS, item_categories, load_compatibility, load_fitb, load_outfits, polyvore_root
from src.utils import load_config, project_path

ID_COLUMNS = ("item_id", "id", "image_id", "name", "filename", "file_name")


def _find_columns(schema) -> tuple[str, str]:
    """Guess (id_column, image_column) from a parquet schema; fail loudly if unsure."""
    import pyarrow as pa

    names = schema.names
    id_col = next((c for c in ID_COLUMNS if c in names), None)
    img_col = None
    for field in schema:
        t = field.type
        if pa.types.is_struct(t) and any(f.name == "bytes" for f in t):
            img_col = field.name
            break
        if pa.types.is_binary(t) or pa.types.is_large_binary(t):
            img_col = field.name
    if img_col is None or id_col is None:
        raise RuntimeError(f"Could not identify id/image columns. Schema:\n{schema}\n"
                           "Set them explicitly in src/data/prepare.py.")
    return id_col, img_col


def extract_images(cfg: dict) -> Path:
    import pyarrow.parquet as pq

    root = polyvore_root(cfg)
    img_dir = root / "images"
    img_dir.mkdir(exist_ok=True)
    files = sorted((root / "data" / cfg["paths"]["split"]).glob("*.parquet"))
    if not files:
        print(f"No parquet files under {root / 'data'}; assuming images/ already populated.")
        return img_dir
    for f in files:
        pf = pq.ParquetFile(f)
        id_col, img_col = _find_columns(pf.schema_arrow)
        written = 0
        for batch in tqdm(pf.iter_batches(columns=[id_col, img_col], batch_size=512),
                          total=pf.metadata.num_row_groups, desc=f"extract {f.name}"):
            for iid, img in zip(batch.column(id_col).to_pylist(), batch.column(img_col).to_pylist()):
                raw = img["bytes"] if isinstance(img, dict) else img
                stem = Path(str(iid)).stem
                out = img_dir / f"{stem}.jpg"
                if raw is None or out.exists():
                    continue
                out.write_bytes(raw)
                written += 1
        print(f"{f.name}: wrote {written} new images")
    return img_dir


def _is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def build_items(cfg: dict) -> pd.DataFrame:
    img_dir = polyvore_root(cfg) / "images"
    cats = item_categories(cfg)
    rows = {}
    for split in SPLITS:
        ids = {it["item_id"] for o in load_outfits(split, cfg) for it in o["items"]}
        for items, _ in load_compatibility(split, cfg):
            ids.update(items)
        for q in load_fitb(split, cfg):
            ids.update(q.question)
            ids.update(q.candidates)
        for iid in ids:
            # "disjoint" means OUTFITS don't overlap; a few items still appear in several splits
            # (3,781 valid items are also in train). `split` = first split seen; `splits` = all.
            r = rows.setdefault(iid, {"item_id": iid, "split": split, "splits": [],
                                      "category": cats.get(iid, "unknown"),
                                      "path": str(img_dir / f"{iid}.jpg")})
            r["splits"].append(split)
    for r in rows.values():
        r["splits"] = ";".join(r["splits"])
    df = pd.DataFrame(rows.values())
    paths = [Path(p) for p in df["path"]]
    with ThreadPoolExecutor(max_workers=16) as ex:
        df["valid"] = list(tqdm(ex.map(_is_valid, paths), total=len(paths), desc="verify images"))
    return df


def report(df: pd.DataFrame, cfg: dict) -> dict:
    stats: dict[str, Any] = {"items_total": len(df), "items_invalid": int((~df["valid"]).sum())}
    for split in SPLITS:
        outfits = load_outfits(split, cfg)
        sizes = Counter(len(o["items"]) for o in outfits)
        sub = df[df["splits"].str.split(";").apply(lambda s, sp=split: sp in s)]
        stats[split] = {
            "outfits": len(outfits),
            "items": len(sub),
            "items_shared_with_other_splits": int((sub["splits"] != split).sum()),
            "invalid_images": int((~sub["valid"]).sum()),
            "outfit_size_hist": dict(sorted(sizes.items())),
            "categories": sub["category"].value_counts().to_dict(),
        }
    return stats


def main() -> None:
    cfg = load_config()
    extract_images(cfg)
    df = build_items(cfg)
    out_dir = project_path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "items.tmp.csv", index=False)
    os.replace(out_dir / "items.tmp.csv", out_dir / "items.csv")   # atomic: readers never see half a file
    stats = report(df, cfg)
    res = project_path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    (res / "data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "train"} | {"train": {k: v for k, v in stats["train"].items() if k != "categories"}}, indent=2))
    print(f"\nSaved {out_dir / 'items.csv'} and {res / 'data_stats.json'}")


if __name__ == "__main__":
    main()
