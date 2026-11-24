"""Step 5: train the compatibility models on cached FashionCLIP embeddings.

Experiments (run in order, one change at a time):
  python -m scripts.train --model A --neg random    --name A_random
  python -m scripts.train --model A --neg hard      --name A_hard
  python -m scripts.train --model A --neg batchhard --name A_batchhard
  python -m scripts.train --model B --base A_hard   --name B
  python -m scripts.train --model B --base A_hard   --color --name B_color

Debug first:  --overfit 100   (the loss must go near zero on 100 pairs, or there is a bug)

Every run: early stopping on validation compat AUC, best checkpoint saved to
models/<name>.pt, metrics appended to results/experiments.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


from src.color import N_COLOR_FEATS, pair_color_features
from src.data.pairs import PairIndex, TensorBatches, TripletEmbeddingDataset, mine_hard_negatives, positive_pairs
from src.data.polyvore import load_compatibility, load_fitb, load_outfits
from src.eval.metrics import evaluate
from src.eval.scoring import batched_scorer, needed_pairs
from src.models.compat import CompatModel, PairScorer, batch_hard_triplet_loss, triplet_loss
from src.utils import get_device, load_config, log_experiment, project_path, set_seed


# ---------------------------------------------------------------- assets
def load_assets(cfg):
    proc = project_path(cfg["paths"]["processed"])
    emb = np.load(proc / "embeddings.npy")
    ids = (proc / "embedding_ids.txt").read_text().split()
    row = {iid: i for i, iid in enumerate(ids)}
    items = pd.read_csv(proc / "items.csv", dtype={"item_id": str}).set_index("item_id")
    cat: dict[str, str] = {str(k): str(v) for k, v in items["category"].items()}
    categories = sorted(set(cat.values()))
    cat_index = {c: i for i, c in enumerate(categories)}
    cat_of_row = torch.tensor([cat_index[cat[i]] for i in ids])
    colors = None
    if (proc / "colors_hsv.npy").exists():
        colors = torch.from_numpy(np.load(proc / "colors_hsv.npy")).float()
    return emb, ids, row, cat, categories, cat_of_row, colors


@torch.no_grad()
def project_all(model: CompatModel, E: torch.Tensor, cat_of_row: torch.Tensor, bs: int = 8192) -> torch.Tensor:
    model.eval()
    return torch.cat([model(E[i:i + bs], cat_of_row[i:i + bs]) for i in range(0, len(E), bs)])


def validate(fn, row, val, device) -> dict[str, float]:
    compat, fitb, pairs = val
    return evaluate(batched_scorer(pairs, row, fn, device), compat, fitb)


def load_compat_model(name: str, n_cat: int, device, cfg) -> CompatModel:
    ck = torch.load(project_path(cfg["paths"]["models"]) / f"{name}.pt", map_location=device, weights_only=True)
    assert ck["kind"] == "A", f"{name} is not a Model A checkpoint"
    m = CompatModel(n_cat, **ck["dims"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()


# ---------------------------------------------------------------- training loops
def train_model_a(args, cfg, E, row, cat_of_row, categories, train_pairs, index, val, device):
    tc = cfg["train"]
    model = CompatModel(len(categories), d_in=E.shape[1]).to(device)
    if args.init:   # curriculum: start from a trained Model A (e.g. random negatives first)
        model.load_state_dict(load_compat_model(args.init, len(categories), device, cfg).state_dict())
        print(f"initialised from {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr or tc["lr"], fused=device.type == "cuda")
    hard = None
    if args.neg == "hard":
        print("Mining hard negatives with the frozen encoder embeddings...")
        hard = mine_hard_negatives(train_pairs, index, E.cpu().numpy(), topk=tc["hard_negative_topk"],
                                   skip=args.hard_skip, device=str(device), seed=cfg["seed"])
    ds = TripletEmbeddingDataset(train_pairs, index, hard_negatives=hard,
                                 hard_fraction=args.hard_fraction if hard else 0.0, seed=cfg["seed"])
    dl = TensorBatches(ds, tc["batch_size"], drop_last=len(ds) > tc["batch_size"], device=device, seed=cfg["seed"])
    cat_global = torch.tensor([categories.index(c) for c in index.categories], device=device)

    def fn_from(Z):
        return lambda a, b: (Z[a] * Z[b]).sum(-1)

    return _fit(args, cfg, model, opt, ds, dl, val, row, device,
                step=lambda batch: _step_a(model, E, batch, cat_global, tc["margin"], args.neg, device),
                score_fn=lambda: fn_from(project_all(model, E, cat_of_row)),
                ckpt=lambda: {"kind": "A", "state": model.state_dict(), "categories": categories,
                              "dims": {"d_in": E.shape[1]}})


def _step_a(model, E, batch, cat_global, margin, neg, device):
    a, p, n, ca, cp, oid = (t.to(device) for t in batch)
    ca, cp = cat_global[ca], cat_global[cp]   # dataset ids -> global category ids
    za, zp, zn = model(E[a], ca), model(E[p], cp), model(E[n], cp)
    if neg in ("batchhard", "semihard"):
        return batch_hard_triplet_loss(za, zp, zn, cp, oid, margin, pos_item=p, semi_hard=neg == "semihard")
    return triplet_loss(za, zp, zn, margin)


def train_model_b(args, cfg, E, row, cat_of_row, categories, train_pairs, index, val, device, colors):
    tc = cfg["train"]
    base = load_compat_model(args.base, len(categories), device, cfg)
    Z = project_all(base, E, cat_of_row)                    # frozen Model A embeddings
    use_color = args.color
    if use_color and colors is None:
        raise SystemExit("colors_hsv.npy missing: run python -m scripts.extract_colors first")
    C = colors.to(device) if use_color else None
    model = PairScorer(d=Z.shape[1], n_color=N_COLOR_FEATS if use_color else 0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], fused=device.type == "cuda")
    ds = TripletEmbeddingDataset(train_pairs, index, seed=cfg["seed"])
    dl = TensorBatches(ds, tc["batch_size"], drop_last=len(ds) > tc["batch_size"], device=device, seed=cfg["seed"])

    def cf(a, b):
        return pair_color_features(C[a], C[b]) if C is not None else None

    def step(batch):
        a, p, n = (t.to(device) for t in batch[:3])
        pos = model.symmetric(Z[a], Z[p], cf(a, p), cf(p, a))
        neg = model.symmetric(Z[a], Z[n], cf(a, n), cf(n, a))
        logits = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        return F.binary_cross_entropy_with_logits(logits, labels)

    def score_fn():
        model.eval()
        return lambda a, b: model.symmetric(Z[a], Z[b], cf(a, b), cf(b, a))

    base_ckpt = {"state": base.state_dict(), "dims": {"d_in": E.shape[1]}}   # frozen copy of A
    return _fit(args, cfg, model, opt, ds, dl, val, row, device, step=step, score_fn=score_fn,
                ckpt=lambda: {"kind": "B", "state": model.state_dict(), "categories": categories,
                              "base": args.base, "base_ckpt": base_ckpt, "color": use_color,
                              "d": Z.shape[1]})


def _fit(args, cfg, model, opt, ds, dl, val, row, device, step, score_fn, ckpt):
    tc = cfg["train"]
    best, best_metrics, bad_epochs = -1.0, {}, 0
    out = project_path(cfg["paths"]["models"]) / f"{args.name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.overfit:
        # a calibrator fit on a previous version of this model would be wrong for the new one
        out.with_name(f"{args.name}_calibrator.json").unlink(missing_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        if epoch > 1:
            ds.resample()
        for batch in dl:
            loss = step(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        msg = f"epoch {epoch:3d} | loss {np.mean(losses):.4f} | {time.time() - t0:.1f}s"
        if args.overfit:
            print(msg)
            continue
        m = validate(score_fn(), row, val, device)
        print(f"{msg} | val AUC {m['compat_auc']:.4f} FITB {m['fitb_acc']:.4f}")
        if m["compat_auc"] > best:
            best, best_metrics, bad_epochs = m["compat_auc"], m | {"epoch": epoch}, 0
            torch.save(ckpt(), out)
        else:
            bad_epochs += 1
            if bad_epochs >= tc["early_stop_patience"]:
                print(f"early stop (no val AUC gain for {bad_epochs} epochs)")
                break
    return best_metrics, out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["A", "B"], required=True)
    ap.add_argument("--neg", choices=["random", "hard", "batchhard", "semihard"], default="random")
    ap.add_argument("--init", help="Model A checkpoint to start from (curriculum fine-tuning)")
    ap.add_argument("--lr", type=float, help="override config train.lr (use ~1e-4 when fine-tuning)")
    ap.add_argument("--hard-fraction", type=float, default=0.5)
    ap.add_argument("--hard-skip", type=int, default=5, help="skip the N closest (likely false negatives)")
    ap.add_argument("--base", help="Model A checkpoint name, required for --model B")
    ap.add_argument("--color", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--name", required=True)
    ap.add_argument("--overfit", type=int, default=0, help="train on N pairs only, no validation")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    args.epochs = args.epochs or (300 if args.overfit else cfg["train"]["epochs"])
    device = get_device()

    emb, ids, row, cat, categories, cat_of_row, colors = load_assets(cfg)
    E, cat_of_row = torch.from_numpy(emb).to(device), cat_of_row.to(device)
    train_outfits = [o for o in load_outfits("train", cfg)]
    pairs = positive_pairs(train_outfits, cat, max_pairs=args.overfit or cfg["train"]["num_positive_pairs"],
                           seed=cfg["seed"])
    pairs = [p for p in pairs if p[0] in row and p[1] in row]
    index = PairIndex(train_outfits, cat, row)
    print(f"{len(pairs)} positive training pairs | {len(categories)} categories | device {device}")

    val_compat, val_fitb = load_compatibility("valid", cfg), load_fitb("valid", cfg)
    val = (val_compat, val_fitb, needed_pairs(val_compat, val_fitb))
    missing = {x for pair in val[2] for x in pair} - row.keys()
    if missing:
        raise SystemExit(f"{len(missing)} validation items have no embedding; re-run scripts.embed_items")

    if args.model == "A":
        metrics, path = train_model_a(args, cfg, E, row, cat_of_row, categories, pairs, index, val, device)
    else:
        if not args.base:
            raise SystemExit("--base <Model A name> is required for Model B")
        metrics, path = train_model_b(args, cfg, E, row, cat_of_row, categories, pairs, index, val,
                                      device, colors)
    if not args.overfit:
        params = {k: v for k, v in vars(args).items()} | {"train_pairs": len(pairs)}
        log_experiment(args.name, params, metrics, cfg)
        print(f"best: {json.dumps(metrics)} -> {path}")


if __name__ == "__main__":
    main()
