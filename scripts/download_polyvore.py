"""Download the Polyvore Outfits dataset (disjoint split) from Hugging Face.

The dataset is gated. One-time setup:
  1. Log in at https://huggingface.co and accept the terms on
     https://huggingface.co/datasets/mvasil/polyvore-outfits
  2. Create a read token at https://huggingface.co/settings/tokens
  3. Run:  venv\\Scripts\\hf auth login      (paste the token when asked)

Then:  venv\\Scripts\\python -m scripts.download_polyvore
"""
from __future__ import annotations

from huggingface_hub import snapshot_download

from src.utils import load_config, project_path

REPO = "mvasil/polyvore-outfits"


def main() -> None:
    cfg = load_config()
    split = cfg["paths"]["split"]
    dest = project_path(cfg["paths"]["polyvore_root"])
    dest.mkdir(parents=True, exist_ok=True)
    patterns = [
        "README.md",
        "categories.csv",
        "polyvore_item_metadata.json",
        "polyvore_outfit_titles.json",
        f"{split}/*",
        f"data/{split}/*",     # images, stored as parquet (~1.5 GB for disjoint)
    ]
    path = snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=dest, allow_patterns=patterns)
    print(f"Downloaded to {path}")

    # Show the parquet schema so the image-extraction step can be checked against it.
    import pyarrow.parquet as pq
    for f in sorted((dest / "data" / split).glob("*.parquet")):
        pf = pq.ParquetFile(f)
        print(f"{f.name}: {pf.metadata.num_rows} rows | schema: {pf.schema_arrow}")


if __name__ == "__main__":
    main()
