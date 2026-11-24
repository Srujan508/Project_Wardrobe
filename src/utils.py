"""Shared helpers: project paths, config loading, seeding, experiment logging."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/config.yaml, or the file named by $WARDROBE_CONFIG (used by tests)."""
    path = path or os.environ.get("WARDROBE_CONFIG") or ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(rel: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_experiment(name: str, params: dict[str, Any], metrics: dict[str, float], cfg: dict | None = None) -> None:
    """Append one run to the JSONL experiment log (lightweight alternative to wandb)."""
    cfg = cfg or load_config()
    out = project_path(cfg["logging"]["experiments_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "params": params, "metrics": metrics}
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_comparison(model: str, split: str, metrics: dict[str, float], cfg: dict | None = None) -> None:
    """Append one row to results/comparison.csv (the table that goes in the report)."""
    cfg = cfg or load_config()
    out = project_path(cfg["logging"]["comparison_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "split", "compat_auc", "fitb_acc", "recall@10"]
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({"model": model, "split": split,
                    **{k: round(metrics[k], 4) if k in metrics else "" for k in fields[2:]}})
