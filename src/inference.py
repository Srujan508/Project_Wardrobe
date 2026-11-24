"""Load a trained model (A or B) as a vectorized pair scorer over embedding rows.

Used by calibration (Step 6), evaluation (Step 7) and the recommender/app (Steps 9-10).
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from src.color import pair_color_features
from src.models.compat import CompatModel, PairScorer
from src.utils import get_device, load_config, project_path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibrator_path(mdir: Path, name: str) -> Path:
    return mdir / f"{name}_calibrator.json"


class CompatScorer:
    def __init__(self, name: str, cfg: dict | None = None, device: torch.device | None = None):
        self.cfg = cfg or load_config()
        self.device = device or get_device()
        self.name = name
        mdir = project_path(self.cfg["paths"]["models"])
        ck_path = mdir / f"{name}.pt"
        # weights_only=True: checkpoints hold only tensors, dicts, lists, strings and numbers,
        # so loading never unpickles arbitrary objects.
        ck = torch.load(ck_path, map_location=self.device, weights_only=True)
        self.kind = ck["kind"]
        self.categories: list[str] = ck["categories"]
        self.cat_id = {c: i for i, c in enumerate(self.categories)}
        base = ck if self.kind == "A" else ck["base_ckpt"]     # B carries its own frozen copy of A
        self.proj = CompatModel(len(self.categories), **base["dims"]).to(self.device)
        self.proj.load_state_dict(base["state"])
        self.proj.eval()
        self.head = None
        self.use_color = False
        if self.kind == "B":
            self.use_color = ck["color"]
            self.head = PairScorer(d=ck["d"], n_color=6 if self.use_color else 0).to(self.device)
            self.head.load_state_dict(ck["state"])
            self.head.eval()
        self.calibrator: dict | None = None
        cal_path = calibrator_path(mdir, name)
        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            if cal.get("model_sha256") == file_sha256(ck_path):
                self.calibrator = cal
            else:
                warnings.warn(f"{cal_path.name} was fit on a different version of {name}.pt; "
                              "ignoring it. Re-run scripts.calibrate.")

    # ---- embedding helpers
    @torch.no_grad()
    def project(self, emb: np.ndarray | torch.Tensor, cats: list[str]) -> torch.Tensor:
        E = torch.as_tensor(emb, dtype=torch.float32, device=self.device)
        c = torch.tensor([self.cat_id[x] for x in cats], device=self.device)
        return torch.cat([self.proj(E[i:i + 8192], c[i:i + 8192]) for i in range(0, len(E), 8192)])

    # ---- scoring
    @torch.no_grad()
    def raw_scores(self, Za: torch.Tensor, Zb: torch.Tensor, Ca: torch.Tensor | None = None,
                   Cb: torch.Tensor | None = None) -> torch.Tensor:
        """Za, Zb: (N, d) projected embeddings; Ca, Cb: (N, 3) HSV colors (Model B + color only)."""
        if self.head is None:
            return (Za * Zb).sum(-1)
        cab = cba = None
        if self.use_color:
            if Ca is None or Cb is None:
                raise ValueError(f"model {self.name} uses color features: pass Ca and Cb")
            cab, cba = pair_color_features(Ca, Cb), pair_color_features(Cb, Ca)
        return self.head.symmetric(Za, Zb, cab, cba)

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        """Raw score -> probability in [0, 1]. Falls back to a logistic squash if uncalibrated."""
        raw = np.asarray(raw, dtype=np.float64).ravel()
        if self.calibrator is None:
            return 1 / (1 + np.exp(-raw))
        return 1 / (1 + np.exp(-(self.calibrator["coef"] * raw + self.calibrator["intercept"])))


def load_dataset_assets(cfg: dict):
    """Embeddings, row map, categories and colors for the Polyvore items."""
    import pandas as pd

    proc = project_path(cfg["paths"]["processed"])
    emb = np.load(proc / "embeddings.npy")
    ids = (proc / "embedding_ids.txt").read_text().split()
    items = pd.read_csv(proc / "items.csv", dtype={"item_id": str}).set_index("item_id")
    cats = [items.at[i, "category"] for i in ids]
    colors = np.load(proc / "colors_hsv.npy") if (proc / "colors_hsv.npy").exists() else None
    return emb, ids, {iid: i for i, iid in enumerate(ids)}, cats, colors


def make_row_fn(scorer: CompatScorer, emb, cats, colors):
    """Return fn(rows_a, rows_b) -> raw scores, for src.eval.scoring.batched_scorer."""
    Z = scorer.project(emb, cats)
    C = torch.as_tensor(colors, device=scorer.device) if (colors is not None and scorer.use_color) else None
    if scorer.use_color and C is None:
        raise SystemExit("This model uses color features: run python -m scripts.extract_colors first")

    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return scorer.raw_scores(Z[a], Z[b], C[a] if C is not None else None, C[b] if C is not None else None)
    return fn, Z
