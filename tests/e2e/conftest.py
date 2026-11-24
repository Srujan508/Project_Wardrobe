"""Real-browser end-to-end tests for the web app (Playwright + Chromium).

Run:   venv\\Scripts\\python -m pytest -m e2e -q          (add --headed to watch)

A plain `pytest` run skips these. The session starts the real app (uvicorn subprocess, real
configs and trained model, real tagger) with a TEMPORARY database and uploads folder, so the
user's wardrobe is never touched. Uploads use real Polyvore test-split product photos.
"""
from __future__ import annotations

import csv as csv_mod
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = ROOT / "results" / "screenshots"

# Real Polyvore test-split photos, chosen so the zero-shot tagger gets the category right.
# One top is deliberately low-confidence (~0.43) so the "please check the type" hint shows up.
UPLOADS = {
    "top_1": "203105203",      # t-shirt
    "top_2": "40356127",       # blouse, low tagger confidence
    "bottom_1": "153243850",   # jeans
    "bottom_2": "88312863",    # jeans
    "shoe": "163417587",       # sneakers
    "dress": "98170098",       # sundress
}
EXTRA_IMAGE = "148061828"     # sneakers: dropped onto the drop zone, then deleted
EXPECTED_CATEGORY = {"top_1": "tops", "top_2": "tops", "bottom_1": "bottoms", "bottom_2": "bottoms",
                     "shoe": "shoes", "dress": "all-body"}
# After upload, items are set to known casual types so filters/empty states are deterministic.
CASUAL_TYPES = {"top_1": "t-shirt", "top_2": "tank top", "bottom_1": "jeans", "bottom_2": "shorts",
                "shoe": "sneakers", "dress": "sundress"}


def pytest_collection_modifyitems(config, items):
    if "e2e" in (config.getoption("markexpr") or "") or os.environ.get("WARDROBE_E2E"):
        return
    skip = pytest.mark.skip(reason="browser e2e test: run with `pytest -m e2e`")
    for item in items:
        if item.get_closest_marker("e2e"):
            item.add_marker(skip)


# ------------------------------------------------------------------ server
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_json(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    """Start uvicorn on a free port with a temp DB + uploads dir; yield its base URL."""
    tmp = tmp_path_factory.mktemp("wardrobe_e2e")
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    cfg["app"]["uploads"] = str(tmp / "uploads")
    cfg["app"]["db"] = str(tmp / "wardrobe.db")
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    port = _free_port()
    env = {k: v for k, v in os.environ.items()
           if k not in ("WARDROBE_DB", "WARDROBE_MODEL", "WARDROBE_ALLOWED_HOSTS")}
    env["WARDROBE_CONFIG"] = str(cfg_path)
    env["PYTHONUNBUFFERED"] = "1"
    log_path = tmp / "server.log"
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 240
        while True:
            if proc.poll() is not None:
                log.flush()
                pytest.fail(f"uvicorn exited early:\n{log_path.read_text(encoding='utf-8')[-3000:]}")
            try:
                _get_json(base + "/options", timeout=2)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                if time.time() > deadline:
                    pytest.fail(f"server didn't answer /options in time:\n{log_path.read_text(encoding='utf-8')[-3000:]}")
                time.sleep(0.5)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log.close()


@pytest.fixture(scope="session")
def options(server) -> dict:
    return _get_json(server + "/options")


# ------------------------------------------------------------------ images
@pytest.fixture(scope="session")
def upload_paths() -> dict[str, Path]:
    csv = ROOT / "data" / "processed" / "items.csv"
    if not csv.exists():
        pytest.skip(f"{csv} missing (run src.data.prepare)")
    wanted = {**UPLOADS, "extra": EXTRA_IMAGE}
    with open(csv, newline="", encoding="utf-8") as f:
        paths = {row["item_id"]: Path(row["path"]) for row in csv_mod.DictReader(f) if row["item_id"] in wanted.values()}
    out = {}
    for role, iid in wanted.items():
        if iid not in paths or not paths[iid].exists():
            pytest.skip(f"Polyvore image {iid} missing")
        out[role] = paths[iid]
    return out


# ------------------------------------------------------------------ browser pages
class ConsoleWatch:
    """Collects console errors, uncaught page errors and failed same-origin requests."""

    def __init__(self, page):
        self.errors: list[str] = []
        page.on("console", lambda m: m.type == "error" and self.errors.append(f"console: {m.text}"))
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        # requests the page itself abandons (reload, superseded fetch) are not errors
        page.on("requestfailed", lambda r: "ERR_ABORTED" not in (r.failure or "")
                and self.errors.append(f"requestfailed: {r.url} {r.failure}"))


@pytest.fixture()
def make_page(browser, server):
    """make_page(viewport=..., color_scheme=...) -> page; asserts no console errors at teardown."""
    made = []

    def _make(width=1280, height=900, color_scheme="light", reduced_motion="reduce"):
        ctx = browser.new_context(base_url=server, viewport={"width": width, "height": height},
                                  color_scheme=color_scheme, reduced_motion=reduced_motion)
        ctx.set_default_timeout(20_000)
        page = ctx.new_page()
        watch = ConsoleWatch(page)
        made.append((ctx, page, watch))
        return page

    yield _make
    errors = [e for _, _, w in made for e in w.errors]
    for ctx, _, _ in made:
        ctx.close()
    assert not errors, "browser errors:\n" + "\n".join(errors)


@pytest.fixture()
def page(make_page):
    return make_page()


def open_app(page, tab: str | None = None):
    page.goto("/" + (f"#{tab}" if tab else ""))
    page.wait_for_selector("body[data-ready='true']")
    return page


# ------------------------------------------------------------------ seeded wardrobe
@pytest.fixture(scope="session")
def seeded(browser, server, upload_paths) -> dict[str, dict]:
    """Upload the six photos through the real file input, check the UI, then pin casual types.

    Returns {role: item} with the items as they are after the type normalisation.
    """
    from playwright.sync_api import expect

    ctx = browser.new_context(base_url=server, viewport={"width": 1280, "height": 900})
    page = ctx.new_page()
    watch = ConsoleWatch(page)
    responses = []
    page.on("response", lambda r: r.request.method == "POST" and r.url.endswith("/items") and responses.append(r))
    try:
        open_app(page, "add")
        expect(page.get_by_role("tab", name="Add Clothes")).to_have_attribute("aria-selected", "true")
        roles = list(UPLOADS)
        page.locator("#file-input").set_input_files([str(upload_paths[r]) for r in roles])

        rows = page.locator("#upload-list .upload-row")
        expect(rows).to_have_count(len(roles))
        expect(page.locator("#first-upload-note")).to_be_visible()     # first upload loads the models
        expect(page.locator('#upload-list .upload-row[data-state="done"]')).to_have_count(len(roles), timeout=240_000)
        expect(page.locator('#upload-list .upload-row[data-state="error"]')).to_have_count(0)
        expect(page.locator("#first-upload-note")).to_be_hidden()

        assert all(r.ok for r in responses), [r.text() for r in responses if not r.ok]
        uploaded = [r.json() for r in responses]
        assert len(uploaded) == len(roles)

        # new cards are editable and show detection + confidence; low-confidence ones show a hint
        cards = page.locator("#just-added .item-card")
        expect(cards).to_have_count(len(roles))
        for it in uploaded:
            card = page.locator(f'#just-added .item-card[data-item-id="{it["id"]}"]')
            expect(card.locator(".item-title")).to_have_text(f'{it["color"]} {it["type"]}')
            expect(card.locator(".confidence")).to_contain_text(f'{round(it["type_confidence"] * 100)}% confidence')
            expect(card.get_by_label("Type")).to_have_value(it["type"])
            expect(card.get_by_label("Color")).to_have_value(it["color"])
            expect(card.locator(".low-conf")).to_be_visible() if it["type_confidence"] < 0.5 \
                else expect(card.locator(".low-conf")).to_be_hidden()

        # map uploads back to roles through the upload list (file name -> item id)
        by_role = {}
        for role in roles:
            row = page.locator("#upload-list .upload-row", has_text=upload_paths[role].name)
            iid = int(row.get_attribute("data-item-id"))
            by_role[role] = next(it for it in uploaded if it["id"] == iid)
        SCREENSHOTS.mkdir(parents=True, exist_ok=True)
        page.evaluate("() => document.querySelectorAll('img[loading=lazy]').forEach(i => i.loading = 'eager')")
        page.wait_for_function("() => [...document.querySelectorAll('#just-added img')]"
                               ".every(i => i.complete && i.naturalWidth > 0)")
        page.screenshot(path=SCREENSHOTS / "desktop-light-add-clothes.png", full_page=True)

        wrong = {r: it["category"] for r, it in by_role.items() if it["category"] != EXPECTED_CATEGORY[r]}
        assert not wrong, f"tagger put photos in unexpected categories: {wrong}"

        for role, t in CASUAL_TYPES.items():
            res = page.request.patch(f"/items/{by_role[role]['id']}", data={"type": t})
            assert res.ok, res.text()
            by_role[role] = res.json()
        assert not watch.errors, watch.errors
        return by_role
    finally:
        ctx.close()
