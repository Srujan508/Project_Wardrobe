"""Step 4: embed every valid item once with FashionCLIP and run a sanity check.

Run:  venv\\Scripts\\python -m scripts.embed_items
Writes data/processed/embeddings.npy and data/processed/embedding_ids.txt.

Sanity check: for 1,000 random items, what fraction of the 5 nearest neighbours
share the item's category? It should be high (well above chance).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.encoder import embed_images
from src.utils import load_config, project_path, set_seed


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    proc_dir = project_path(cfg["paths"]["processed"])
    items = pd.read_csv(proc_dir / "items.csv", dtype={"item_id": str})
    items = items[items["valid"]].reset_index(drop=True)
    print(f"Embedding {len(items)} items...")

    emb = embed_images([str(p) for p in items["path"]], batch_size=cfg["encoder"]["batch_size"])
    # write both to temp names, then swap them in, so the pair is never out of sync
    tmp_npy, tmp_ids = proc_dir / "embeddings.tmp.npy", proc_dir / "embedding_ids.tmp.txt"
    np.save(tmp_npy, emb)
    tmp_ids.write_text("\n".join(str(x) for x in items["item_id"]))
    os.replace(tmp_ids, proc_dir / "embedding_ids.txt")
    os.replace(tmp_npy, proc_dir / "embeddings.npy")
    print(f"Saved embeddings {emb.shape} -> {proc_dir / 'embeddings.npy'}")

    # Sanity check: category purity of nearest neighbours.
    cats = items["category"].to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    q = rng.choice(len(items), size=min(1000, len(items)), replace=False)
    sims = emb[q] @ emb.T
    sims[np.arange(len(q)), q] = -np.inf
    nn = np.argsort(-sims, axis=1)[:, :5]
    purity = float((cats[nn] == cats[q][:, None]).mean())
    chance = float((pd.Series(cats).value_counts(normalize=True) ** 2).sum())
    print(f"Top-5 neighbour category purity: {purity:.3f} (chance = {chance:.3f})")


if __name__ == "__main__":
    main()
