"""Full API flow with the REAL tagger (rembg + FashionCLIP) on real Polyvore photos, in a temp DB/uploads dir."""
from __future__ import annotations

import contextlib
import copy
import importlib
import sys

import pytest
import yaml

from src.utils import project_path

pytestmark = pytest.mark.integration

UPLOAD_PLAN = {"tops": 2, "bottoms": 2, "shoes": 1}


@contextlib.contextmanager
def make_client(real_cfg, tmp_path, monkeypatch, model: str | None):
    cfg = copy.deepcopy(real_cfg)
    cfg["app"] |= {"uploads": str(tmp_path / "uploads"), "db": str(tmp_path / "w.db"),
                   "model": model or "does_not_exist", "allowed_hosts": ["testserver"]}
    cfg["paths"]["models"] = str(project_path(real_cfg["paths"]["models"]) if model else tmp_path / "no_models")
    cfg["logging"] = {"experiments_file": str(tmp_path / "exp.jsonl"),
                      "comparison_file": str(tmp_path / "comparison.csv")}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("WARDROBE_CONFIG", str(tmp_path / "config.yaml"))
    for var in ("WARDROBE_DB", "WARDROBE_MODEL", "WARDROBE_ALLOWED_HOSTS"):
        monkeypatch.delenv(var, raising=False)

    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    from fastapi.testclient import TestClient
    try:
        with TestClient(main.app) as c:
            yield c
    finally:
        sys.modules.pop("app.main", None)


def upload_wardrobe(client, sample_images) -> list[dict]:
    out = []
    for cat, k in UPLOAD_PLAN.items():
        for p in sample_images[cat][:k]:
            r = client.post("/items", files={"file": (p.name, p.read_bytes(), "image/jpeg")})
            assert r.status_code == 200, r.text
            out.append(r.json() | {"_true": cat})
    return out


def check_outfits(resp: dict, items: list[dict], n: int):
    by_id = {it["id"]: it for it in items}
    outfits = resp["outfits"]
    assert 1 <= len(outfits) <= n
    for o in outfits:
        ids = o["item_ids"]
        assert len(ids) == len(set(ids)), f"duplicate items in {ids}"
        assert set(ids) <= set(by_id)
        cats = [by_id[i]["category"] for i in ids]
        two_piece = cats.count("tops") == 1 and cats.count("bottoms") == 1 and "all-body" not in cats
        one_piece = cats.count("all-body") == 1 and "tops" not in cats and "bottoms" not in cats
        assert two_piece or one_piece, cats
        assert cats.count("shoes") <= 1 and cats.count("outerwear") <= 1, cats
        assert [it["id"] for it in o["items"]] == ids
        assert 0.0 <= o["score"] <= 1.0
        assert isinstance(o["explanation"], str) and o["explanation"].strip()
    keys = [tuple(sorted(o["item_ids"])) for o in outfits]
    assert len(keys) == len(set(keys)), "the same outfit was recommended twice"


def test_full_flow_untrained(real_cfg, sample_images, tmp_path, monkeypatch):
    with make_client(real_cfg, tmp_path, monkeypatch, model=None) as client:
        assert client.get("/options").json()["model"] is None
        uploaded = upload_wardrobe(client, sample_images)
        assert len({u["id"] for u in uploaded}) == 5
        for u in uploaded:
            assert u["image"].startswith("/uploads/clean/")
            assert client.get(u["image"]).status_code == 200
            assert 1 <= u["formality"] <= 5 and 1 <= u["warmth"] <= 5
        right = sum(u["category"] == u["_true"] for u in uploaded)
        assert right >= 3, f"tagger got {right}/5 categories right: {[(u['_true'], u['type']) for u in uploaded]}"
        cats = [u["category"] for u in uploaded]
        if "tops" not in cats or "bottoms" not in cats:
            pytest.fail(f"tagger produced no top or no bottom, cannot recommend: {cats}")

        items = client.get("/items").json()
        assert sorted(i["id"] for i in items) == sorted(u["id"] for u in uploaded)
        assert len(client.get("/items?category=tops").json()) == cats.count("tops")

        resp = client.post("/recommend", json={"occasion": "any", "weather": "any", "n": 3})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] is None
        check_outfits(body, items, 3)

        # feedback / save / list / delete round trip
        o = body["outfits"][0]
        assert client.post("/feedback", json={"item_ids": o["item_ids"], "vote": 1}).json() == {"ok": True}
        assert client.post("/feedback", json={"item_ids": o["item_ids"], "vote": -1}).json() == {"ok": True}
        sid = client.post("/saved", json={"item_ids": o["item_ids"], "score": o["score"],
                                          "explanation": o["explanation"][:500]}).json()["id"]
        saved = client.get("/saved").json()
        assert [s["id"] for s in saved] == [sid]
        assert [it["id"] for it in saved[0]["items"]] == o["item_ids"]
        assert client.delete(f"/saved/{sid}").status_code == 200
        assert client.get("/saved").json() == []
        assert client.delete(f"/saved/{sid}").status_code == 404

        # deleting an item removes its files and it disappears from listings
        victim = items[0]
        clean_file = tmp_path / "uploads" / victim["image"].removeprefix("/uploads/")
        assert clean_file.exists()
        assert client.delete(f"/items/{victim['id']}").status_code == 200
        assert not clean_file.exists()
        assert len(client.get("/items").json()) == 4
        assert client.delete(f"/items/{victim['id']}").status_code == 404


def test_full_flow_with_trained_model(real_cfg, trained_model, sample_images, tmp_path, monkeypatch):
    with make_client(real_cfg, tmp_path, monkeypatch, model=trained_model) as client:
        assert client.get("/options").json()["model"] == trained_model
        upload_wardrobe(client, sample_images)
        items = client.get("/items").json()
        cats = [i["category"] for i in items]
        if "tops" not in cats or "bottoms" not in cats:
            pytest.fail(f"tagger produced no top or no bottom, cannot recommend: {cats}")
        for occasion, weather in [("any", "any"), ("casual", "mild")]:
            resp = client.post("/recommend", json={"occasion": occasion, "weather": weather, "n": 3})
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["model"] == trained_model
            if occasion == "any":
                check_outfits(body, items, 3)
            elif body["outfits"]:
                check_outfits(body, items, 3)
