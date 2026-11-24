"""API test with a stubbed tagger (no rembg / FashionCLIP download needed)."""
import importlib
import json
import io
import sys

import numpy as np
import pytest
import yaml
from PIL import Image

from src.tagger import TaggedItem
from src.wardrobe_rules import TYPES


def png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from pathlib import Path
    base = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "config.yaml").read_text())
    base["app"] |= {"uploads": str(tmp_path / "uploads"), "db": str(tmp_path / "w.db"), "model": "does_not_exist",
                    "allowed_hosts": ["testserver"]}
    base["paths"]["models"] = str(tmp_path / "models")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(base))
    monkeypatch.setenv("WARDROBE_CONFIG", str(cfg))

    # Stub the tagger: type is chosen from the image color so tests are deterministic.
    import src.tagger as tagger
    order = iter(["t-shirt", "jeans", "sneakers", "button-up shirt", "shorts"])
    rng = np.random.default_rng(0)

    def fake_process(img, remove_bg=True):
        t = next(order)
        e = rng.normal(size=512).astype(np.float32)
        return TaggedItem(TYPES[t].category, t, 0.9, (20, 20, 20), "black", (0.0, 0.0, 0.08),
                          TYPES[t].formality, TYPES[t].warmth, e / np.linalg.norm(e), img.convert("RGB"))
    monkeypatch.setattr(tagger, "process_upload", fake_process)

    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c


def test_full_flow(client):
    for col in ["red", "blue", "white", "green", "grey"]:
        r = client.post("/items", files={"file": (f"../../evil_{col}.png", png_bytes(col), "image/png")})
        assert r.status_code == 200, r.text
        assert "evil" not in r.json()["image"]          # client filename is never used
    items = client.get("/items").json()
    assert len(items) == 5
    assert client.get(items[0]["image"]).status_code == 200

    assert client.get("/items?category=bottoms").json().__len__() == 2

    r = client.post("/recommend", json={"occasion": "any", "weather": "any", "n": 3}).json()
    assert r["model"] is None and 1 <= len(r["outfits"]) <= 3
    o = r["outfits"][0]
    assert len(o["items"]) == len(o["item_ids"]) and o["explanation"]

    assert client.post("/feedback", json={"item_ids": o["item_ids"], "vote": 1}).json() == {"ok": True}
    sid = client.post("/saved", json={"item_ids": o["item_ids"], "score": o["score"],
                                      "explanation": o["explanation"]}).json()["id"]
    assert client.get("/saved").json()[0]["id"] == sid

    jid = items[0]["id"]
    assert client.patch(f"/items/{jid}", json={"type": "boots"}).json()["category"] == "shoes"
    assert client.patch(f"/items/{jid}", json={"type": "spacesuit"}).status_code == 400
    assert client.delete(f"/items/{jid}").status_code == 200
    assert len(client.get("/items").json()) == 4


def test_rejects_non_images(client):
    r = client.post("/items", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


# ---------------------------------------------------------------- security regressions
def _upload(client, data, name="x.png", **kw):
    return client.post("/items", files={"file": (name, data, "image/png")}, **kw)


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_decompression_bomb_rejected(client):
    # 7000 x 7000 = 49M pixels: above the 25M cap -> 400, never a 500 or an accepted upload
    buf = io.BytesIO()
    Image.new("1", (7000, 7000)).save(buf, format="PNG")
    r = _upload(client, buf.getvalue())
    assert r.status_code == 400, r.text


def test_oversized_body_rejected_before_reading(client):
    r = _upload(client, b"\0" * (16 * 1024 * 1024))
    assert r.status_code == 413


def test_only_allowed_image_formats(client):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, format="BMP")
    assert _upload(client, buf.getvalue(), name="x.bmp").status_code == 400


def test_foreign_host_and_cross_origin_refused(client):
    assert client.get("/items", headers={"host": "attacker.example"}).status_code == 400
    r = _upload(client, png_bytes("red"), headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/feedback", json={"item_ids": [1], "vote": 1}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403


def test_request_bounds(client):
    bad = [
        ("/saved", {"item_ids": [1], "score": float("inf"), "explanation": "x"}),
        ("/saved", {"item_ids": [1], "score": 1.5, "explanation": "x"}),
        ("/saved", {"item_ids": [1], "score": 0.5, "explanation": "x" * 501}),
        ("/saved", {"item_ids": list(range(1, 50)), "score": 0.5, "explanation": "x"}),
        ("/feedback", {"item_ids": [], "vote": 1}),
        ("/feedback", {"item_ids": [0], "vote": 1}),
    ]
    for url, body in bad:
        # json.dumps would refuse inf; send raw text the way a hostile client would
        raw = json.dumps(body, allow_nan=True).replace("Infinity", "1e999")
        r = client.post(url, content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 422, (url, body, r.status_code)
    assert client.delete("/items/99999999999999999999").status_code == 422
    assert client.get("/saved").status_code == 200


def test_color_edit_updates_features(client):
    _upload(client, png_bytes("red"))
    item = client.get("/items").json()[0]
    assert client.patch(f"/items/{item['id']}", json={"color": "not-a-color"}).status_code == 400
    assert client.patch(f"/items/{item['id']}", json={"color": "white"}).json()["color"] == "white"


def test_same_origin_required_including_port(client):
    body = {"item_ids": [1], "vote": 1}
    assert client.post("/feedback", json=body, headers={"origin": "http://testserver"}).status_code == 200
    # another port on the same host (e.g. a local dev server) is a different origin
    assert client.post("/feedback", json=body, headers={"origin": "http://testserver:3000"}).status_code == 403
    # scheme isn't compared (TLS proxies), host and port are


def test_chunked_body_without_length_refused(client):
    r = client.post("/feedback", content=iter([b'{"item_ids": [1], "vote": 1}']),
                    headers={"content-type": "application/json"})
    assert r.status_code == 411


def test_large_phone_photo_accepted_and_downsized(client, tmp_path):
    # 4032 x 3024 (12 MP phone photo): allowed, decoded at reduced size, stored <= 2048 px
    buf = io.BytesIO()
    Image.new("RGB", (4032, 3024), (30, 40, 90)).save(buf, format="JPEG", quality=80)
    r = client.post("/items", files={"file": ("phone.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200, r.text
    stored = client.get(r.json()["image"])
    assert max(Image.open(io.BytesIO(stored.content)).size) <= 2048


def test_optional_basic_auth(client, monkeypatch):
    import base64
    monkeypatch.setenv("WARDROBE_BASIC_AUTH", "me:s3cret-pass")
    r = client.get("/items")
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic")
    bad = base64.b64encode(b"me:wrong").decode()
    assert client.get("/items", headers={"authorization": f"Basic {bad}"}).status_code == 401
    assert client.get("/items", headers={"authorization": "Basic !!notbase64"}).status_code == 401
    good = base64.b64encode(b"me:s3cret-pass").decode()
    assert client.get("/items", headers={"authorization": f"Basic {good}"}).status_code == 200
    assert client.get("/", headers={"authorization": f"Basic {good}"}).status_code == 200
    monkeypatch.delenv("WARDROBE_BASIC_AUTH")
    assert client.get("/items").status_code == 200          # off by default (local use)


def test_security_headers_and_proxy_origin(client):
    r = client.get("/")
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "object-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    # behind a TLS proxy the browser's Origin is https while the app sees http: same host -> allowed
    body = {"item_ids": [1], "vote": 1}
    assert client.post("/feedback", json=body, headers={"origin": "https://testserver"}).status_code == 200
    assert client.post("/feedback", json=body, headers={"origin": "https://evil.example"}).status_code == 403


def test_refuses_to_run_open_on_a_space(tmp_path, monkeypatch):
    import importlib, sys
    monkeypatch.setenv("SPACE_ID", "someone/wardrobe")
    monkeypatch.delenv("WARDROBE_BASIC_AUTH", raising=False)
    sys.modules.pop("app.main", None)
    with pytest.raises(RuntimeError, match="WARDROBE_BASIC_AUTH"):
        importlib.import_module("app.main")
    sys.modules.pop("app.main", None)
