"""Dominant color extraction, color names, and pairwise color-harmony features."""
from __future__ import annotations

import colorsys

import numpy as np
import torch
from PIL import Image

NAMED_COLORS = {
    "black": (20, 20, 20), "white": (245, 245, 245), "grey": (128, 128, 128),
    "light grey": (200, 200, 200), "cream": (240, 230, 205),
    "navy": (25, 35, 80), "blue": (40, 90, 200), "light blue": (150, 190, 230), "teal": (20, 120, 120),
    "red": (190, 30, 40), "burgundy": (110, 20, 40), "pink": (235, 150, 180), "light pink": (245, 205, 215),
    "purple": (110, 50, 140), "lilac": (200, 170, 215),
    "green": (40, 130, 60), "light green": (170, 215, 165), "olive": (110, 110, 50),
    "yellow": (235, 210, 60), "orange": (230, 120, 40),
    "brown": (110, 70, 40), "tan": (190, 150, 105), "beige": (215, 195, 160),
}
_NAMES = list(NAMED_COLORS)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, ..., 3) -> CIELAB (D65). Distances in Lab match perceived color difference
    far better than RGB distance does (e.g. a pale green no longer reads as beige)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


_LAB = _srgb_to_lab(np.array(list(NAMED_COLORS.values())))


def dominant_color(img: Image.Image, k: int = 3) -> tuple[int, int, int]:
    """Largest k-means cluster over foreground pixels. Foreground = non-transparent
    and not near-white (product photos and rembg output both use white/transparent bg)."""
    import cv2

    im = img.convert("RGBA").resize((64, 64))
    arr = np.asarray(im).reshape(-1, 4).astype(np.float32)
    rgb, alpha = arr[:, :3], arr[:, 3]
    near_white = (rgb.min(axis=1) > 235)
    fg = rgb[(alpha > 128) & ~near_white]
    if len(fg) < k * 10:
        fg = rgb[alpha > 128] if (alpha > 128).sum() >= k * 10 else rgb
    _, labels, centers = cv2.kmeans(fg, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                    3, cv2.KMEANS_PP_CENTERS)
    c = centers[np.bincount(labels.ravel(), minlength=k).argmax()]
    return tuple(int(v) for v in c)  # type: ignore[return-value]


def color_name(rgb) -> str:
    """Nearest named color by perceptual (CIELAB) distance."""
    return _NAMES[int(np.argmin(((_LAB - _srgb_to_lab(np.asarray(rgb))) ** 2).sum(-1)))]


def rgb_to_hsv01(rgb) -> tuple[float, float, float]:
    r, g, b = (v / 255.0 for v in rgb)
    return colorsys.rgb_to_hsv(r, g, b)


def is_neutral(hsv) -> bool:
    h, s, v = hsv
    return s < 0.2 or v < 0.2 or (v > 0.9 and s < 0.1)


def pair_color_features(hsv_a: torch.Tensor, hsv_b: torch.Tensor) -> torch.Tensor:
    """(B,3),(B,3) HSV in [0,1] -> (B,6): circular hue diff, sat a/b, val a/b, any-neutral flag."""
    dh = (hsv_a[:, 0] - hsv_b[:, 0]).abs()
    dh = torch.minimum(dh, 1 - dh) * 2                         # 0 = same hue, 1 = complementary

    def neutral(x):
        return (x[:, 1] < 0.2) | (x[:, 2] < 0.2) | ((x[:, 2] > 0.9) & (x[:, 1] < 0.1))

    flag = (neutral(hsv_a) | neutral(hsv_b)).float()
    return torch.stack([dh, hsv_a[:, 1], hsv_b[:, 1], hsv_a[:, 2], hsv_b[:, 2], flag], dim=-1)


N_COLOR_FEATS = 6
