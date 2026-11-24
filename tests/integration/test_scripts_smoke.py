"""Cheap smoke run of a pipeline script on the real data, with all logging redirected to a temp dir."""
from __future__ import annotations

import copy
import csv
import os
import subprocess
import sys

import pytest
import yaml

from src.utils import ROOT, project_path

pytestmark = pytest.mark.integration


def _rows(path, model, split):
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(r["model"] == model and r["split"] == split for r in csv.DictReader(f))


def test_run_baselines_random_on_valid(real_cfg, split_dir, items_df, tmp_path):
    cfg = copy.deepcopy(real_cfg)
    cfg["logging"] = {"experiments_file": str(tmp_path / "experiments.jsonl"),
                      "comparison_file": str(tmp_path / "comparison.csv")}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    real_cmp = project_path(real_cfg["logging"]["comparison_file"])
    before = _rows(real_cmp, "baseline_random", "valid")

    env = os.environ | {"WARDROBE_CONFIG": str(tmp_path / "config.yaml"), "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-m", "scripts.run_baselines", "--split", "valid", "--only", "random"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-4000:]}"
    assert "valid: 6000 compat questions, 3000 FITB questions" in r.stdout

    with open(tmp_path / "comparison.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(x["model"], x["split"]) for x in rows] == [("baseline_random", "valid")]
    auc, fitb = float(rows[0]["compat_auc"]), float(rows[0]["fitb_acc"])
    assert 0.45 < auc < 0.55, f"random baseline AUC {auc} should be ~0.5"
    assert 0.20 < fitb < 0.30, f"random baseline FITB {fitb} should be ~0.25"
    assert (tmp_path / "experiments.jsonl").read_text().count("baseline_random") == 1
    assert _rows(real_cmp, "baseline_random", "valid") == before, "wrote to the real results/comparison.csv"
