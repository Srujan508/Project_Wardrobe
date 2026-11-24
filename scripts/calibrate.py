"""Step 6: Platt-scale a trained model's scores into probabilities.

Fit on VALID pairs, report on TEST pairs (never fit on what you report on).
Pairs: every cross-category pair inside an outfit (label 1), each matched with
a same-category replacement from another outfit (label 0).

Run:  venv\\Scripts\\python -m scripts.calibrate --model B_color
Writes models/<model>_calibrator.json and results/reliability_<model>.png
"""
from __future__ import annotations

import argparse
import json
import random

import matplotlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.pairs import PairIndex, positive_pairs  # noqa: E402
from src.data.polyvore import load_outfits  # noqa: E402
from src.inference import CompatScorer, calibrator_path, file_sha256, load_dataset_assets, make_row_fn  # noqa: E402
from src.utils import load_config, log_experiment, project_path, set_seed  # noqa: E402


def labelled_pairs(split, cfg, cat, row, n_max=50000, seed=0):
    outfits = load_outfits(split, cfg)
    pos = [p for p in positive_pairs(outfits, cat, max_pairs=n_max, seed=seed) if p[0] in row and p[1] in row]
    idx = PairIndex(outfits, cat, row)
    rng = random.Random(seed)
    neg = [(a, idx.random_negative(a, cat[p], rng)) for a, p, _ in pos]
    pairs = [(a, p) for a, p, _ in pos] + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))
    return pairs, labels


def score_pairs(fn, row, pairs, device):
    a = torch.tensor([row[x] for x, _ in pairs], device=device)
    b = torch.tensor([row[y] for _, y in pairs], device=device)
    return np.concatenate([fn(a[i:i + 65536], b[i:i + 65536]).float().cpu().numpy()
                           for i in range(0, len(pairs), 65536)])


def ece(prob, labels, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo)
        if m.any():
            total += m.mean() * abs(prob[m].mean() - labels[m].mean())
    return float(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])

    emb, ids, row, cats, colors = load_dataset_assets(cfg)
    cat = dict(zip(ids, cats))
    scorer = CompatScorer(args.model, cfg)
    fn, _ = make_row_fn(scorer, emb, cats, colors)

    vp, vl = labelled_pairs("valid", cfg, cat, row, seed=cfg["seed"])
    tp, tl = labelled_pairs("test", cfg, cat, row, seed=cfg["seed"] + 1)
    vs, ts = score_pairs(fn, row, vp, scorer.device), score_pairs(fn, row, tp, scorer.device)

    cal = LogisticRegression().fit(vs.reshape(-1, 1), vl)
    mdir = project_path(cfg["paths"]["models"])
    out = calibrator_path(mdir, args.model)
    # JSON, not pickle; tied to the exact checkpoint so a retrained model can't reuse it
    out.write_text(json.dumps({"coef": float(cal.coef_[0, 0]), "intercept": float(cal.intercept_[0]),
                               "model_sha256": file_sha256(mdir / f"{args.model}.pt"),
                               "fit_on": "valid", "n_pairs": len(vp)}, indent=2))

    prob = cal.predict_proba(ts.reshape(-1, 1))[:, 1]
    test_ece = ece(prob, tl)
    print(f"test ECE = {test_ece:.4f}  ({len(tp)} test pairs)  -> {out}")

    edges = np.linspace(0, 1, 11)
    centers, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi)
        if m.any():
            centers.append(prob[m].mean()); accs.append(tl[m].mean())
    res = project_path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfectly calibrated")
    ax.plot(centers, accs, "o-", color="#2a6fdb", label=f"{args.model} (ECE {test_ece:.3f})")
    ax.set_xlabel("predicted compatibility"); ax.set_ylabel("observed fraction compatible")
    ax.set_title("Reliability diagram (test pairs)"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(res / f"reliability_{args.model}.png", dpi=150)
    log_experiment(f"calibrate_{args.model}", {"model": args.model}, {"test_ece": test_ece}, cfg)


if __name__ == "__main__":
    main()
