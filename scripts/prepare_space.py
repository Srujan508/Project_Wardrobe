"""Assemble a ready-to-push Hugging Face Space (Docker SDK) in dist/space/.

    venv\\Scripts\\python -m scripts.prepare_space --model B

Copies only what the app needs (Dockerfile, configs, src, app, the chosen model + calibrator)
and writes the Space card. Nothing is uploaded: you push it yourself (see deploy/HUGGINGFACE.md).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils import ROOT, load_config, project_path

CARD = """---
title: AI Wardrobe Stylist
emoji: 👗
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

Upload photos of your clothes and get outfit suggestions from a compatibility model trained on
Polyvore Outfits (test AUC {auc}, FITB {fitb}).

**Set these Space secrets before use** (Settings -> Variables and secrets):
- `WARDROBE_BASIC_AUTH` = `user:long-random-password` (the app has no accounts; without this anyone
  with the link can see and delete the wardrobe)
- `WARDROBE_ALLOWED_HOSTS` = `<your-space-subdomain>.hf.space`

Storage on a free Space is not persistent: uploads reset when the Space restarts.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model name in models/ (default: config app.model)")
    ap.add_argument("--auc", default="0.894")
    ap.add_argument("--fitb", default="0.622")
    args = ap.parse_args()
    cfg = load_config()
    name = args.model or cfg["app"]["model"]
    mdir = project_path(cfg["paths"]["models"])
    needed = [mdir / f"{name}.pt", mdir / f"{name}_calibrator.json"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing}: train and calibrate the model first")

    out = ROOT / "dist" / "space"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for d in ("configs", "src", "app"):
        shutil.copytree(ROOT / d, out / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("Dockerfile", ".dockerignore", "requirements.lock.txt"):
        shutil.copy2(ROOT / f, out / f)
    (out / "models").mkdir()
    for p in needed:
        shutil.copy2(p, out / "models" / p.name)
    (out / "README.md").write_text(CARD.format(auc=args.auc, fitb=args.fitb), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Space ready in {out} ({size:.1f} MB). Push it with the steps in deploy/HUGGINGFACE.md")


if __name__ == "__main__":
    main()
