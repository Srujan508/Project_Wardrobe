"""Step 8: turn an uploaded photo into {category, type, color, embedding}.

  1. Background removal (rembg), then composite onto WHITE so the photo looks like
     Polyvore's product images (the model was trained on those).
  2. Zero-shot type with FashionCLIP: compare the image with "a photo of a <type>".
  3. Dominant color (k-means on foreground pixels) + nearest color name.
  4. FashionCLIP embedding of the cleaned image.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

from src.color import color_name, dominant_color, rgb_to_hsv01
from src.encoder import embed_pil, embed_texts
from src.wardrobe_rules import TYPES


@dataclass
class TaggedItem:
    category: str
    type: str
    type_confidence: float
    color_rgb: tuple[int, int, int]
    color_name: str
    color_hsv: tuple[float, float, float]
    formality: int
    warmth: int
    embedding: np.ndarray = field(repr=False)
    clean_image: Image.Image = field(repr=False)


@lru_cache(maxsize=1)
def _rembg_session():
    from rembg import new_session
    return new_session("u2net")


@lru_cache(maxsize=1)
def _type_text_embeddings() -> tuple[list[str], np.ndarray]:
    names = list(TYPES)
    return names, embed_texts([f"a photo of a {t}" for t in names])


def remove_background(img: Image.Image, max_side: int = 768) -> Image.Image:
    """Return the item cut out and centred on a white square canvas (RGB)."""
    from rembg import remove

    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    cut = remove(img, session=_rembg_session())                       # RGBA
    bbox = cut.getchannel("A").point(lambda a: 255 if a > 20 else 0).getbbox()
    if bbox:
        cut = cut.crop(bbox)
    side = int(max(cut.size) * 1.1)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    canvas.alpha_composite(cut, ((side - cut.width) // 2, (side - cut.height) // 2))
    return canvas.convert("RGB")


def classify(embedding: np.ndarray) -> tuple[str, float]:
    names, T = _type_text_embeddings()
    logits = 100.0 * (T @ embedding)
    p = np.exp(logits - logits.max()); p /= p.sum()
    i = int(np.argmax(p))
    return names[i], float(p[i])


def process_upload(data: bytes | Image.Image, remove_bg: bool = True) -> TaggedItem:
    img = data if isinstance(data, Image.Image) else Image.open(io.BytesIO(data))
    clean = remove_background(img) if remove_bg else ImageOps.exif_transpose(img).convert("RGB")
    emb = embed_pil([clean])[0]
    t, conf = classify(emb)
    rgb = dominant_color(clean)
    info = TYPES[t]
    return TaggedItem(category=info.category, type=t, type_confidence=conf, color_rgb=rgb,
                      color_name=color_name(rgb), color_hsv=rgb_to_hsv01(rgb),
                      formality=info.formality, warmth=info.warmth, embedding=emb, clean_image=clean)
