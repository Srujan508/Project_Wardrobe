"""Step 7: evaluate a trained model on the official benchmarks and append to comparison.csv.

  compat AUC, FITB accuracy  (official files, all categories, comparable with papers)
  Recall@K                   "what bottom goes with this top?": rank the true bottom among
                             all bottoms of the split (app-style retrieval)

Run:  venv\\Scripts\\python -m scripts.run_eval --model A_hard --split test
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.polyvore import load_compatibility, load_fitb, load_outfits
from src.eval.metrics import evaluate
from src.eval.scoring import batched_scorer, needed_pairs
from src.inference import CompatScorer, load_dataset_assets, make_row_fn
from src.utils import append_comparison, load_config, log_experiment, set_seed


@torch.no_grad()
def recall_at_k(fn, row, cat, split, cfg, device, q_cat="tops", g_cat="bottoms", ks=(1, 10, 50)):
    outfits = load_outfits(split, cfg)
    queries = []
    for o in outfits:
        tops = [i["item_id"] for i in o["items"] if cat.get(i["item_id"]) == q_cat and i["item_id"] in row]
        bots = [i["item_id"] for i in o["items"] if cat.get(i["item_id"]) == g_cat and i["item_id"] in row]
        if tops and bots:
            queries.append((tops[0], bots[0]))
    gallery = sorted({i["item_id"] for o in outfits for i in o["items"]
                      if cat.get(i["item_id"]) == g_cat and i["item_id"] in row})
    g_rows = torch.tensor([row[g] for g in gallery], device=device)
    g_pos = {g: i for i, g in enumerate(gallery)}
    ranks = []
    for q, truth in queries:
        s = fn(torch.full_like(g_rows, row[q]), g_rows)
        ranks.append(int((s > s[g_pos[truth]]).sum().item()))
    ranks = np.array(ranks)
    return {f"recall@{k}": float((ranks < k).mean()) for k in ks} | {"recall_queries": len(queries),
                                                                    "gallery": len(gallery)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])

    emb, ids, row, cats, colors = load_dataset_assets(cfg)
    cat = dict(zip(ids, cats))
    scorer = CompatScorer(args.model, cfg)
    fn, _ = make_row_fn(scorer, emb, cats, colors)

    compat, fitb = load_compatibility(args.split, cfg), load_fitb(args.split, cfg)
    pairs = needed_pairs(compat, fitb)
    missing = {x for pair in pairs for x in pair} - row.keys()
    if missing:
        raise SystemExit(f"{len(missing)} {args.split} items have no embedding; re-run scripts.embed_items")
    m = evaluate(batched_scorer(pairs, row, fn, scorer.device), compat, fitb)
    m |= recall_at_k(fn, row, cat, args.split, cfg, scorer.device)
    print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()))
    append_comparison(args.model, args.split, m, cfg)
    log_experiment(f"eval_{args.model}_{args.split}", {"model": args.model, "split": args.split}, m, cfg)


if __name__ == "__main__":
    main()
