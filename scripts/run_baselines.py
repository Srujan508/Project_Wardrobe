"""Step 2: evaluate the three baselines on the official compat-AUC and FITB tasks.

Run:  venv\\Scripts\\python -m scripts.run_baselines --split test
Appends one row per baseline to results/comparison.csv.

Baselines are untrained, so reporting them on the test split is fine.
Trained models are developed on 'valid' and reported on 'test' once.
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd

from src.data.polyvore import load_compatibility, load_fitb
from src.eval.metrics import evaluate
from src.features import cached_features, cosine_pair_scorer
from src.utils import append_comparison, load_config, log_experiment, project_path, set_seed


def random_scorer(a: str, b: str) -> float:
    """Deterministic pseudo-random score per unordered pair."""
    key = "|".join(sorted((a, b))).encode()
    return zlib.crc32(key) / 2**32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--only", nargs="*", default=["random", "color_hist", "resnet50"])
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    compat = load_compatibility(args.split, cfg)
    fitb = load_fitb(args.split, cfg)
    needed = {i for items, _ in compat for i in items} | {i for q in fitb for i in q.question + q.candidates}

    items = pd.read_csv(project_path(cfg["paths"]["processed"]) / "items.csv", dtype={"item_id": str})
    items = items[items["item_id"].isin(needed)]
    missing = needed - set(items["item_id"])
    bad = items[~items["valid"]]
    if missing or len(bad):
        raise SystemExit(f"{len(missing)} referenced items missing from items.csv, {len(bad)} invalid images. "
                         "Re-run src.data.prepare and check the download.")
    print(f"{args.split}: {len(compat)} compat questions, {len(fitb)} FITB questions, {len(items)} items")

    for name in args.only:
        proc = project_path(cfg["paths"]["processed"])
        if name == "random":
            scorer = random_scorer
        elif name == "fashion_clip" and (proc / "embeddings.npy").exists():
            # reuse the Step 4 cache instead of re-encoding every image
            ids = (proc / "embedding_ids.txt").read_text().split()
            scorer = cosine_pair_scorer(np.load(proc / "embeddings.npy"), {iid: i for i, iid in enumerate(ids)})
        else:
            feats, idx = cached_features(name, items, cfg, tag=args.split)
            scorer = cosine_pair_scorer(feats, idx)
        m = evaluate(scorer, compat, fitb)
        print(f"{name:12s}  AUC={m['compat_auc']:.4f}  FITB={m['fitb_acc']:.4f}")
        append_comparison(f"baseline_{name}", args.split, m, cfg)
        log_experiment(f"baseline_{name}", {"split": args.split}, m, cfg)


if __name__ == "__main__":
    main()
