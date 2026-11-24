"""End-to-end check of Steps 3 and 5 on synthetic data with the real file layout."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.synthetic import build

ROOT = Path(__file__).resolve().parent.parent


def run(cfg_path, *args, script="scripts.train"):
    env = os.environ | {"WARDROBE_CONFIG": str(cfg_path), "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-m", script, *args], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    return root, build(root)


def best(root, name):
    lines = (root / "results" / "experiments.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines if json.loads(x)["name"] == name][-1]["metrics"]


def test_overfit_tiny_set(synth):
    _, cfg = synth
    out = run(cfg, "--model", "A", "--neg", "random", "--name", "ovf", "--overfit", "100", "--epochs", "150")
    final_loss = float(out.strip().splitlines()[-1].split("loss")[1].split("|")[0])
    assert final_loss < 0.05, out[-500:]


@pytest.mark.parametrize("neg", ["random", "hard", "batchhard"])
def test_model_a_learns(synth, neg):
    root, cfg = synth
    run(cfg, "--model", "A", "--neg", neg, "--name", f"A_{neg}", "--hard-skip", "1")
    m = best(root, f"A_{neg}")
    assert m["compat_auc"] > 0.8 and m["fitb_acc"] > 0.5, m


def test_model_b_learns(synth):
    root, cfg = synth
    run(cfg, "--model", "A", "--neg", "random", "--name", "A_base")
    run(cfg, "--model", "B", "--base", "A_base", "--name", "B")
    m = best(root, "B")
    assert m["compat_auc"] > 0.8, m


def test_calibrate_and_eval(synth):
    root, cfg = synth
    run(cfg, "--model", "A", "--neg", "random", "--name", "A_eval")
    out = run(cfg, "--model", "A_eval", script="scripts.calibrate")
    assert "test ECE" in out and (root / "models" / "A_eval_calibrator.json").exists()
    assert (root / "results" / "reliability_A_eval.png").exists()
    ece = best(root, "calibrate_A_eval")["test_ece"]
    assert 0.0 <= ece < 0.15, ece

    run(cfg, "--model", "A_eval", "--split", "test", script="scripts.run_eval")
    m = best(root, "eval_A_eval_test")
    assert m["compat_auc"] > 0.8 and m["recall@50"] >= m["recall@10"] >= m["recall@1"]
    rows = (root / "results" / "comparison.csv").read_text().splitlines()
    assert rows[0] == "model,split,compat_auc,fitb_acc,recall@10" and rows[-1].startswith("A_eval,test,")

    # calibrated probabilities stay in [0, 1]
    import numpy as np
    from src.inference import CompatScorer
    from src.utils import load_config
    sc = CompatScorer("A_eval", load_config(cfg))
    p = sc.calibrate(np.linspace(-1, 1, 21))
    assert ((p >= 0) & (p <= 1)).all() and np.all(np.diff(p) >= 0)
