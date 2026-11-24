"""Per-item feature extractors with an on-disk cache, plus a cosine pair scorer.

Each extractor maps a list of image paths to an (N, D) float32 array.
Features are L2-normalized so cosine similarity is a dot product.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils import get_device, project_path


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def load_rgb_on_white(path: str | Path) -> Image.Image:
    """Open an image as RGB, compositing any transparency onto white (Polyvore's background)."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


# ---------- color histogram ----------
def color_hist_features(paths: list[str], bins: int = 8) -> np.ndarray:
    """8x8x8 HSV histogram. Near-white pixels are masked out so the product
    background does not dominate every histogram."""
    import cv2

    feats = np.zeros((len(paths), bins ** 3), dtype=np.float32)
    for i, p in enumerate(tqdm(paths, desc="color hist")):
        rgb = np.asarray(load_rgb_on_white(p).resize((128, 128)))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = ~((hsv[..., 1] < 20) & (hsv[..., 2] > 235))
        mask = (mask.astype(np.uint8) * 255) if mask.any() else None
        h = cv2.calcHist([hsv], [0, 1, 2], mask, [bins] * 3, [0, 180, 0, 256, 0, 256])
        feats[i] = h.flatten()
    return l2norm(feats)


# ---------- CNN / CLIP features ----------
class _ImageDS(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(load_rgb_on_white(self.paths[index]))


@torch.no_grad()
def resnet50_features(paths: list[str], batch_size: int = 128, num_workers: int = 4) -> np.ndarray:
    import torchvision.models as tvm
    import torchvision.transforms as T

    device = get_device()
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dl = DataLoader(_ImageDS(paths, tf), batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    out = []
    for x in tqdm(dl, desc="resnet50"):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return l2norm(np.concatenate(out))


def fashion_clip_features(paths: list[str]) -> np.ndarray:
    from src.encoder import embed_images  # local import: avoids a circular import
    return embed_images(paths)


EXTRACTORS: dict[str, Callable[[list[str]], np.ndarray]] = {
    "color_hist": color_hist_features,
    "resnet50": resnet50_features,
    "fashion_clip": fashion_clip_features,
}


def cached_features(name: str, items: pd.DataFrame, cfg: dict, tag: str = "all") -> tuple[np.ndarray, dict[str, int]]:
    """Compute (or load) features for the given items. Returns (matrix, item_id -> row).
    `tag` names the cache file (e.g. the split) so different item sets don't overwrite each other."""
    out_dir = project_path(cfg["paths"]["processed"]) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = items["item_id"].astype(str).tolist()
    f_npy, f_ids = out_dir / f"{name}_{tag}.npy", out_dir / f"{name}_{tag}_ids.txt"
    if f_npy.exists() and f_ids.exists():
        cached_ids = f_ids.read_text().split()
        if set(ids) <= set(cached_ids):
            idx = {iid: i for i, iid in enumerate(cached_ids)}
            return np.load(f_npy), idx
    feats = EXTRACTORS[name](items["path"].tolist())
    np.save(f_npy, feats)
    f_ids.write_text("\n".join(ids))
    return feats, {iid: i for i, iid in enumerate(ids)}


def cosine_pair_scorer(feats: np.ndarray, idx: dict[str, int]):
    def score(a: str, b: str) -> float:
        return float(feats[idx[a]] @ feats[idx[b]])
    return score
