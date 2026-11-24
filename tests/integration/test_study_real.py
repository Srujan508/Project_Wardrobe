"""Style study endpoints on the real held-out test items with the real trained model."""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils import project_path
from tests.integration.test_api_real import make_client

pytestmark = pytest.mark.integration


@pytest.fixture()
def client(real_cfg, trained_model, tmp_path, monkeypatch):
    with make_client(real_cfg, tmp_path, monkeypatch, model=trained_model) as c:
        yield c


def test_study_round_vote_and_results(client, real_cfg, monkeypatch):
    import app.study
    monkeypatch.setattr(app.study, "MIN_SECONDS_TO_VOTE", 0.0)      # test votes instantly
    assert client.get("/options").json()["study_available"] is True
    test_ids = set(pd.read_csv(project_path(real_cfg["paths"]["processed"]) / "items.csv",
                               dtype={"item_id": str}).query("splits == 'test'")["item_id"])
    sides = []
    for i in range(6):
        r = client.post("/study/next", json={}).json()
        urls = [r["top"], r["left"], r["right"]]
        assert len(set(urls)) == 3                                   # model pick != control
        for u in urls:
            assert u.split("/")[-1] in test_ids                      # held-out items only
        img = client.get(r["left"])
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
        choice = "left" if i % 2 == 0 else "right"
        assert client.post("/study/vote", json={"token": r["token"], "choice": choice}).json() == {"ok": True}
        # one vote per round
        assert client.post("/study/vote", json={"token": r["token"], "choice": choice}).status_code == 409
        sides.append(choice)
    skip = client.post("/study/next", json={}).json()
    client.post("/study/vote", json={"token": skip["token"], "choice": "skip"})

    res = client.get("/study/results").json()
    assert res["votes"] == 6 and res["skipped"] == 1
    assert 0 <= res["model_preferred"] <= 6
    assert res["rate"] == round(res["model_preferred"] / 6, 4)
    lo, hi = res["ci95"]
    assert 0.0 <= lo <= res["rate"] <= hi <= 1.0


def test_study_refuses_bad_input(client):
    assert client.post("/study/vote", json={"token": "x" * 32, "choice": "left"}).status_code == 404
    assert client.post("/study/vote", json={"token": "short", "choice": "left"}).status_code == 422
    assert client.post("/study/vote", json={"token": "a" * 32, "choice": "middle"}).status_code == 422
    # only digits, and only held-out test items, are served
    assert client.get("/study/image/..%2F..%2Fconfigs").status_code in (404, 422)
    assert client.get("/study/image/abc").status_code == 422
    assert client.get("/study/image/1").status_code == 404
    assert client.get("/study/results").json()["votes"] == 0


def test_study_next_is_not_a_get_and_fast_votes_are_excluded(client):
    # a GET must not create rounds (an <img src> on any website could trigger it)
    assert client.get("/study/next").status_code == 405
    r = client.post("/study/next", json={}).json()
    assert client.post("/study/vote", json={"token": r["token"], "choice": "left"}).json() == {"ok": True}
    res = client.get("/study/results").json()
    assert res["votes"] == 0 and res["too_fast"] == 1            # cast within 1 s: left out


def test_concurrent_votes_on_one_round_only_one_wins(client):
    from concurrent.futures import ThreadPoolExecutor
    r = client.post("/study/next", json={}).json()
    with ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(lambda c: client.post("/study/vote", json={"token": r["token"], "choice": c}).status_code,
                            ["left", "right"] * 4))
    assert codes.count(200) == 1 and codes.count(409) == 7, codes
