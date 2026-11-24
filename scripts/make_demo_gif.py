"""Record results/demo.gif: a scripted walk through the real app in Chromium.

    venv\\Scripts\\python -m scripts.make_demo_gif

Starts its own server on a free port with a temp DB/uploads dir (your wardrobe is untouched),
uploads real Polyvore test-split photos, and screenshots each step into an animated GIF.
"""
from __future__ import annotations

import csv
import io
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml
from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "demo.gif"
# Polyvore test-split items: 2 tops, 2 bottoms, sneakers, a dress, a coat-style layer
ITEMS = ["203105203", "40356127", "153243850", "88312863", "163417587", "98170098"]
WIDTH, HEIGHT, GIF_WIDTH = 1180, 760, 900


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(tmp: Path) -> tuple[subprocess.Popen, str]:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    cfg["app"] |= {"uploads": str(tmp / "uploads"), "db": str(tmp / "w.db")}
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("WARDROBE_")}
    env |= {"WARDROBE_CONFIG": str(tmp / "config.yaml"), "PYTHONUNBUFFERED": "1"}
    port = free_port()
    with open(tmp / "server.log", "w") as log:          # child keeps its own handle; ours closes
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                                 "--port", str(port)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(480):
        try:
            urllib.request.urlopen(base + "/options", timeout=2)
            return proc, base
        except OSError:
            if proc.poll() is not None:
                raise SystemExit((tmp / "server.log").read_text()[-2000:])
            time.sleep(0.5)
    raise SystemExit("server did not start")


def image_paths() -> list[str]:
    with open(ROOT / "data" / "processed" / "items.csv", newline="", encoding="utf-8") as f:
        paths = {r["item_id"]: r["path"] for r in csv.DictReader(f) if r["item_id"] in ITEMS}
    return [paths[i] for i in ITEMS]


def main() -> None:
    frames: list[tuple[Image.Image, int]] = []          # (frame, duration ms)

    def shot(page, ms=1600):
        frames.append((Image.open(io.BytesIO(page.screenshot())).convert("RGB"), ms))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        proc, base = start_server(Path(td))
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, color_scheme="light")
                page.goto(base)
                expect(page.locator("#suggest-btn")).to_be_enabled(timeout=60_000)
                shot(page, 1800)                                        # empty Get Styled

                page.get_by_role("tab", name="Add Clothes").click()
                page.locator("#file-input").set_input_files(image_paths())
                page.wait_for_timeout(1200)
                shot(page, 1400)                                        # uploads in progress
                expect(page.locator('#upload-list .upload-row[data-state="done"]')).to_have_count(
                    len(ITEMS), timeout=300_000)
                page.locator("#just-added .item-card").first.scroll_into_view_if_needed()
                page.wait_for_timeout(600)
                shot(page, 2400)                                        # tagged, editable cards

                page.get_by_role("tab", name="My Wardrobe").click()
                page.wait_for_timeout(800)
                shot(page, 2200)                                        # wardrobe grid

                page.get_by_role("tab", name="Get Styled").click()
                page.locator("#suggest-btn").click()
                expect(page.locator("#styled-results .outfit-card").first).to_be_visible(timeout=60_000)
                page.wait_for_timeout(800)
                shot(page, 2600)                                        # outfit suggestions
                page.mouse.wheel(0, 520)
                page.wait_for_timeout(600)
                shot(page, 2000)

                study_tab = page.get_by_role("tab", name="Style Study")
                if study_tab.is_visible():
                    study_tab.click()
                    expect(page.locator("#study-left")).to_be_enabled(timeout=60_000)
                    page.wait_for_timeout(1200)
                    shot(page, 2400)                                    # style study
                browser.close()
        finally:
            proc.terminate()
            proc.wait(timeout=20)

    imgs = [f.resize((GIF_WIDTH, round(f.height * GIF_WIDTH / f.width)), Image.LANCZOS) for f, _ in frames]
    pal = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in imgs]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=[d for _, d in frames], loop=0,
                optimize=True)
    print(f"Saved {OUT} ({len(frames)} frames, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
