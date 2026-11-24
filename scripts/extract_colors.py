"""Compute each item's dominant color (HSV in [0,1]) once, aligned with embedding_ids.txt.

Run:  venv\\Scripts\\python -m scripts.extract_colors
Writes data/processed/colors_hsv.npy, used by `scripts.train --color`.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.color import dominant_color, rgb_to_hsv01
from src.utils import load_config, project_path


def _hsv(path: str) -> tuple[float, float, float]:
    try:
        with Image.open(path) as im:
            return rgb_to_hsv01(dominant_color(im))
    except Exception:
        return (0.0, 0.0, 0.5)   # unreadable: neutral grey


def main() -> None:
    cfg = load_config()
    proc = project_path(cfg["paths"]["processed"])
    ids = (proc / "embedding_ids.txt").read_text().split()
    paths = pd.read_csv(proc / "items.csv", dtype={"item_id": str}).set_index("item_id")["path"]
    todo = [paths[i] for i in ids]
    with ProcessPoolExecutor() as ex:
        hsv = list(tqdm(ex.map(_hsv, todo, chunksize=256), total=len(todo), desc="dominant colors"))
    np.save(proc / "colors_hsv.npy", np.asarray(hsv, dtype=np.float32))
    print(f"Saved {len(hsv)} colors -> {proc / 'colors_hsv.npy'}")


if __name__ == "__main__":
    main()
