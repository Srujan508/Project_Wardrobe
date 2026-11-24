"""Step 1 EDA: category counts, outfit sizes, and a grid of sample images.

Run:  venv\\Scripts\\python -m scripts.eda
Writes results/eda/*.png (static figures for the report).
"""
from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.polyvore import SPLITS, load_outfits  # noqa: E402
from src.utils import load_config, project_path  # noqa: E402

SERIES = "#2a78d6"     # categorical slot 1 (single series -> no legend)
SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
GRID = "#e6e5e1"


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_axisbelow(True)


def main() -> None:
    cfg = load_config()
    items = pd.read_csv(project_path(cfg["paths"]["processed"]) / "items.csv", dtype={"item_id": str})
    out = project_path(cfg["paths"]["results"]) / "eda"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Items per category (train split), horizontal bars sorted by count
    counts = items[items["split"] == "train"]["category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    bars = ax.barh(counts.index, counts.values, color=SERIES, height=0.6)
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], padding=4, color=INK2, fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("items in train split", color=INK2, fontsize=9)
    style(ax, "Items per category (train)")
    fig.tight_layout(); fig.savefig(out / "items_per_category.png", dpi=160); plt.close(fig)

    # 2. Outfit size distribution (train)
    sizes = Counter(len(o["items"]) for o in load_outfits("train", cfg))
    xs = sorted(sizes)
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    bars = ax.bar([str(x) for x in xs], [sizes[x] for x in xs], color=SERIES, width=0.6)
    ax.bar_label(bars, labels=[f"{sizes[x]:,}" for x in xs], padding=3, color=INK2, fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("items per outfit", color=INK2, fontsize=9)
    style(ax, "Outfit sizes (train)")
    fig.tight_layout(); fig.savefig(out / "outfit_sizes.png", dpi=160); plt.close(fig)

    # 3. 20 random item images with their category
    sample = items[items["valid"] & (items["split"] == "train")].sample(20, random_state=cfg["seed"])
    fig, axes = plt.subplots(4, 5, figsize=(10, 8.6), facecolor=SURFACE)
    for ax, (_, r) in zip(axes.flat, sample.iterrows()):
        ax.imshow(Image.open(r["path"]).convert("RGB"))
        ax.set_title(r["category"], fontsize=9, color=INK2)
        ax.axis("off")
    fig.suptitle("Random training items", x=0.02, ha="left", color=INK, fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "sample_items.png", dpi=130); plt.close(fig)

    # 4. Split summary table
    rows = []
    for s in SPLITS:
        sub = items[items["split"] == s]
        rows.append({"split": s, "outfits": len(load_outfits(s, cfg)), "items": len(sub),
                     "invalid_images": int((~sub["valid"]).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(out / "split_summary.csv", index=False)
    print(table.to_string(index=False))
    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
