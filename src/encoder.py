"""Step 4: FashionCLIP image (and text) encoder.

The encoder stays frozen; its 512-d embeddings are computed once and cached,
so training the compatibility head takes minutes instead of hours.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.features import load_rgb_on_white
from src.utils import get_device, load_config


def _as_tensor(out) -> torch.Tensor:
    """transformers versions differ: get_*_features may return a tensor or a ModelOutput."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(f"Unexpected encoder output type: {type(out)}")


@lru_cache(maxsize=1)
def load_fashion_clip(name: str | None = None):
    from transformers import CLIPModel, CLIPProcessor

    name = name or load_config()["encoder"]["name"]
    device = get_device()
    model = CLIPModel.from_pretrained(name).eval().to(device)
    proc = CLIPProcessor.from_pretrained(name)
    return model, proc, device


class _ClipImageDS(Dataset):
    def __init__(self, paths, image_processor):
        self.paths, self.ip = paths, image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.ip(images=load_rgb_on_white(self.paths[index]), return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def embed_images(paths: list[str], batch_size: int = 64, num_workers: int = 4) -> np.ndarray:
    model, proc, device = load_fashion_clip()
    dl = DataLoader(_ClipImageDS(paths, proc.image_processor), batch_size=batch_size,
                    num_workers=num_workers, pin_memory=device.type == "cuda")
    out = []
    for px in tqdm(dl, desc="fashion-clip"):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            e = _as_tensor(model.get_image_features(pixel_values=px.to(device, non_blocking=True)))
        out.append(F.normalize(e.float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def embed_pil(images: list, batch_size: int = 32) -> np.ndarray:
    """Embed in-memory PIL images (used by the app for uploads)."""
    model, proc, device = load_fashion_clip()
    out = []
    for i in range(0, len(images), batch_size):
        px = proc.image_processor(images=images[i:i + batch_size], return_tensors="pt")["pixel_values"].to(device)
        out.append(F.normalize(_as_tensor(model.get_image_features(pixel_values=px)).float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def embed_texts(texts: list[str]) -> np.ndarray:
    """Text embeddings, for zero-shot category tagging (Step 8)."""
    model, proc, device = load_fashion_clip()
    tok = proc.tokenizer(texts, padding=True, return_tensors="pt").to(device)
    return F.normalize(_as_tensor(model.get_text_features(**tok)).float(), dim=-1).cpu().numpy()
