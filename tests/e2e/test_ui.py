"""Browser end-to-end tests of the wardrobe app (see conftest.py for the live server and seeding).

Tests share one server and run in file order: the first test sees an empty wardrobe; the
`seeded` fixture uploads six real photos through the file input the first time it is used.
"""
from __future__ import annotations

import base64
import re

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.conftest import SCREENSHOTS, open_app

pytestmark = pytest.mark.e2e
expect.set_options(timeout=15_000)

TAB_NAMES = ["Get Styled", "My Wardrobe", "Add Clothes", "Saved Outfits"]
IMAGES_LOADED = """sel => {
  const imgs = [...document.querySelectorAll(sel)];
  return imgs.length > 0 && imgs.every(i => i.complete && i.naturalWidth > 0);
}"""
NO_HSCROLL = "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth"
UNLABELLED_CONTROLS = """() => [...document.querySelectorAll('input:not([type=hidden]), select, textarea, button')]
  .filter(el => !el.closest('[hidden]') && !el.closest('dialog:not([open])'))
  .filter(el => {
    const name = (el.labels && [...el.labels].map(l => l.textContent).join('').trim())
      || el.getAttribute('aria-label') || el.getAttribute('aria-labelledby')
      || (el.tagName === 'BUTTON' && el.textContent.trim());
    return !name;
  }).map(el => el.outerHTML.slice(0, 120))"""


def tab(page: Page, name: str):
    return page.get_by_role("tab", name=name, exact=True)


def run_suggest(page: Page, occasion="any", weather="any", count=4):
    page.get_by_label("Occasion").select_option(occasion)
    page.get_by_label("Weather").select_option(weather)
    page.get_by_label("How many outfits").fill(str(count))
    page.get_by_role("button", name="Suggest outfits").click()
    expect(page.locator("#styled-results")).to_have_attribute("aria-busy", "false", timeout=60_000)


# ------------------------------------------------------------------ shell, tabs, keyboard
def test_tabs_keyboard_and_header(page: Page, options):
    open_app(page)
    expected = TAB_NAMES + (["Style Study"] if options["study_available"] else [])
    expect(page.get_by_role("tab")).to_have_text(expected)
    expect(page.locator("#model-text")).to_have_text(
        f"Model {options['model']}" if options["model"] else "Untrained fallback")

    first, second, last = tab(page, expected[0]), tab(page, expected[1]), tab(page, expected[-1])
    expect(first).to_have_attribute("aria-selected", "true")
    expect(page.locator('[role="tab"][tabindex="0"]')).to_have_count(1)      # roving tabindex

    first.focus()
    page.keyboard.press("ArrowRight")
    expect(second).to_be_focused()
    expect(second).to_have_attribute("aria-selected", "true")
    expect(first).to_have_attribute("tabindex", "-1")
    expect(page.locator("#panel-wardrobe")).to_be_visible()
    expect(page.locator("#panel-styled")).to_be_hidden()
    assert page.url.endswith("#wardrobe")

    page.keyboard.press("ArrowLeft")
    expect(first).to_be_focused()
    page.keyboard.press("ArrowLeft")                                       # wraps around
    expect(last).to_be_focused()
    expect(last).to_have_attribute("aria-selected", "true")
    page.keyboard.press("Home")
    expect(first).to_have_attribute("aria-selected", "true")
    page.keyboard.press("End")
    expect(last).to_be_focused()
    page.keyboard.press("Home")

    page.keyboard.press("Tab")                                             # Tab leaves the tablist
    expect(page.locator("#panel-styled")).to_be_focused()
    page.keyboard.press("ArrowRight")                                      # ...so arrows no longer switch tabs
    expect(first).to_have_attribute("aria-selected", "true")

    # the selected tab survives a reload (hash)
    tab(page, "Saved Outfits").click()
    page.reload()
    page.wait_for_selector("body[data-ready='true']")
    expect(tab(page, "Saved Outfits")).to_have_attribute("aria-selected", "true")

    for name in expected:
        tab(page, name).click()
        assert page.evaluate(UNLABELLED_CONTROLS) == [], name


def test_empty_wardrobe_states(page: Page):
    if page.request.get("/items").json():
        pytest.skip("wardrobe already has items (test run out of order)")
    open_app(page)
    empty = page.locator("#styled-results .empty-state")
    expect(empty).to_contain_text("Your wardrobe is empty")
    empty.get_by_role("button", name="Add clothes").click()
    expect(tab(page, "Add Clothes")).to_have_attribute("aria-selected", "true")
    expect(tab(page, "Add Clothes")).to_be_focused()

    tab(page, "My Wardrobe").click()
    expect(page.locator("#wardrobe-grid .empty-state")).to_contain_text("Your wardrobe is empty")
    tab(page, "Saved Outfits").click()
    expect(page.locator("#saved-list .empty-state")).to_contain_text("No saved outfits yet")


# ------------------------------------------------------------------ uploads
def test_upload_via_file_input(page: Page, seeded):
    items = page.request.get("/items").json()
    assert sorted(i["id"] for i in items) == sorted(i["id"] for i in seeded.values())
    for it in items:
        assert page.request.get(it["image"]).ok
    assert {it["category"] for it in items} == {"tops", "bottoms", "shoes", "all-body"}


def test_drag_and_drop_upload_then_delete_with_confirm(page: Page, seeded, upload_paths):
    open_app(page, "add")
    data = base64.b64encode(upload_paths["extra"].read_bytes()).decode()
    page.evaluate("""b64 => {
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const dt = new DataTransfer();
        dt.items.add(new File([bytes], 'dropped-shoe.jpg', {type: 'image/jpeg'}));
        const zone = document.getElementById('dropzone');
        for (const type of ['dragenter', 'dragover', 'drop'])
          zone.dispatchEvent(new DragEvent(type, {dataTransfer: dt, bubbles: true, cancelable: true}));
    }""", data)
    row = page.locator("#upload-list .upload-row", has_text="dropped-shoe.jpg")
    expect(row).to_have_attribute("data-state", "done", timeout=60_000)
    expect(row.locator(".upload-state")).to_have_text("Added")
    item_id = row.get_attribute("data-item-id")
    assert item_id
    card = page.locator(f'#just-added .item-card[data-item-id="{item_id}"]')
    expect(card).to_be_visible()

    # cancel keeps the item and returns focus to the Delete button
    delete = card.get_by_role("button", name="Delete")
    delete.click()
    dialog = page.get_by_role("dialog", name="Delete this item?")
    expect(dialog).to_be_visible()
    expect(dialog.get_by_role("button", name="Cancel")).to_be_focused()
    page.keyboard.press("Escape")
    expect(dialog).to_be_hidden()
    expect(delete).to_be_focused()
    expect(card).to_be_visible()

    delete.click()
    with page.expect_response(lambda r: r.request.method == "DELETE"):
        dialog.get_by_role("button", name="Delete").click()
    expect(card).to_have_count(0)
    assert int(item_id) not in [i["id"] for i in page.request.get("/items").json()]


def test_rejects_bad_file_inline(page: Page, seeded, tmp_path):
    open_app(page, "add")
    bad = tmp_path / "notes.txt"
    bad.write_text("not an image")
    page.locator("#file-input").set_input_files(str(bad))
    row = page.locator("#upload-list .upload-row", has_text="notes.txt")
    expect(row).to_have_attribute("data-state", "error")
    expect(row.locator(".upload-error")).to_have_text("Not a JPEG, PNG or WEBP image.")


# ------------------------------------------------------------------ wardrobe editing
def test_edit_type_and_color_persists_after_reload(page: Page, seeded):
    target = seeded["top_1"]
    open_app(page, "wardrobe")
    card = page.locator(f'#wardrobe-grid .item-card[data-item-id="{target["id"]}"]')
    expect(card.get_by_label("Type")).to_have_value(target["type"])

    with page.expect_response(lambda r: r.request.method == "PATCH" and r.ok):
        card.get_by_label("Type").select_option("polo shirt")
    expect(card.locator(".inline-msg")).to_have_text("Saved")
    with page.expect_response(lambda r: r.request.method == "PATCH" and r.ok):
        card.get_by_label("Color").select_option("red")
    expect(card.locator(".item-title")).to_have_text("red polo shirt")

    page.reload()
    page.wait_for_selector("body[data-ready='true']")
    card = page.locator(f'#wardrobe-grid .item-card[data-item-id="{target["id"]}"]')
    expect(card.get_by_label("Type")).to_have_value("polo shirt")
    expect(card.get_by_label("Color")).to_have_value("red")
    expect(card.locator("img")).to_have_attribute("alt", "red polo shirt")
    # type options are grouped by category
    groups = card.get_by_label("Type").locator("optgroup")
    expect(groups.first).to_have_attribute("label", "Tops")
    assert groups.count() == 5

    # a low-confidence detection shows a hint until the user picks the type themselves
    unsure = next((it for it in seeded.values() if it["type_confidence"] < 0.5), None)
    if unsure is not None:
        card = page.locator(f'#wardrobe-grid .item-card[data-item-id="{unsure["id"]}"]')
        expect(card.locator(".low-conf")).to_be_visible()
        type_select = card.get_by_label("Type")
        type_select.focus()
        with page.expect_response(lambda r: r.request.method == "PATCH" and r.ok):
            type_select.select_option(unsure["type"])
        expect(card.locator(".low-conf")).to_be_hidden()
        expect(card.locator(".confidence")).to_have_text("Type set by you")
        expect(type_select).to_be_focused()                      # focus is not lost while saving


def test_category_filter_counts(page: Page, seeded):
    open_app(page, "wardrobe")
    expected = {"all": 6, "tops": 2, "bottoms": 2, "all-body": 1, "shoes": 1, "outerwear": 0}
    for cat, n in expected.items():
        expect(page.locator(f'.chip[data-category="{cat}"] .count')).to_have_text(str(n))
    expect(page.locator('.chip[data-category="all"]')).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#wardrobe-grid .item-card")).to_have_count(6)

    page.locator('.chip[data-category="tops"]').click()
    expect(page.locator('.chip[data-category="tops"]')).to_have_attribute("aria-pressed", "true")
    expect(page.locator('.chip[data-category="all"]')).to_have_attribute("aria-pressed", "false")
    expect(page.locator("#wardrobe-grid .item-card")).to_have_count(2)

    page.locator('.chip[data-category="outerwear"]').click()
    expect(page.locator("#wardrobe-grid .empty-state")).to_contain_text("No outerwear yet")

    page.locator('.chip[data-category="all"]').click()
    expect(page.locator("#wardrobe-grid .item-card")).to_have_count(6)
    page.wait_for_function(IMAGES_LOADED, arg="#wardrobe-grid img")
    for img in page.locator("#wardrobe-grid img").all():
        assert re.fullmatch(r"[a-z ]+ [a-z -]+", img.get_attribute("alt") or "")


# ------------------------------------------------------------------ get styled, feedback, saved
def test_get_styled_feedback_save_and_remove(page: Page, seeded):
    open_app(page)
    run_suggest(page, count=4)
    cards = page.locator("#styled-results .outfit-card")
    n = cards.count()
    assert 1 <= n <= 4
    page.wait_for_function(IMAGES_LOADED, arg="#styled-results img")
    for i in range(n):
        expect(cards.nth(i).locator(".score")).to_have_text(re.compile(r"^\d{1,3}% compatibility estimate$"))
        expect(cards.nth(i).locator(".explanation")).not_to_be_empty()
        pieces = cards.nth(i).locator(".pieces li")
        assert 1 <= pieces.count() <= 4
    assert "accuracy" not in page.locator("body").inner_text().lower()

    first = cards.first
    like, dislike = first.get_by_role("button", name="Like"), first.get_by_role("button", name="Not for me")
    like.click()
    expect(like).to_have_attribute("aria-pressed", "true")
    expect(dislike).to_have_attribute("aria-pressed", "false")
    dislike.click()
    expect(dislike).to_have_attribute("aria-pressed", "true")
    expect(like).to_have_attribute("aria-pressed", "false")

    pieces = first.locator("figcaption").all_text_contents()
    save = first.get_by_role("button", name="Save outfit")
    save.click()
    saved_btn = first.locator(".save-btn")
    expect(saved_btn).to_have_text("Saved ✓")
    expect(saved_btn).to_be_disabled()
    expect(saved_btn).to_be_focused()

    tab(page, "Saved Outfits").click()
    saved = page.locator("#saved-list .outfit-card")
    expect(saved).to_have_count(1)
    expect(saved.first.locator("figcaption")).to_have_text(pieces)
    expect(saved.first.locator(".score")).to_have_text(re.compile(r"^\d{1,3}% compatibility estimate$"))
    page.wait_for_function(IMAGES_LOADED, arg="#saved-list img")

    # a re-run marks the already-saved outfit as saved
    tab(page, "Get Styled").click()
    run_suggest(page, count=4)
    expect(page.locator("#styled-results .save-btn[data-saved='true']")).to_have_count(1)

    tab(page, "Saved Outfits").click()
    saved.first.get_by_role("button", name="Remove").click()
    expect(page.locator("#saved-list .empty-state")).to_contain_text("No saved outfits yet")
    assert page.request.get("/saved").json() == []


def test_impossible_filter_shows_empty_state(page: Page, seeded):
    open_app(page)
    run_suggest(page, occasion="formal", weather="hot")
    empty = page.locator("#styled-results .empty-state")
    expect(empty).to_contain_text("No outfits fit a formal occasion in hot weather")
    expect(page.locator("#styled-results .outfit-card")).to_have_count(0)
    empty.get_by_role("button", name="Show any outfit").click()
    expect(page.locator("#styled-results .outfit-card").first).to_be_visible(timeout=60_000)
    expect(page.get_by_label("Occasion")).to_have_value("any")


# ------------------------------------------------------------------ style study
def test_style_study_keyboard_votes(page: Page, options):
    if not options["study_available"]:
        pytest.skip("style study not available (needs the trained model + Polyvore test data)")
    open_app(page, "study")
    expect(page.locator("#panel-study")).to_contain_text("Which bottom goes better with this top?")
    expect(page.locator("#study-left")).to_be_enabled(timeout=60_000)
    page.wait_for_function(IMAGES_LOADED, arg="#study-stage img")
    expect(page.locator("#study-round")).to_have_text("Round 1")
    expect(page.locator("#stat-votes")).not_to_have_text("–")
    votes0 = int(page.locator("#stat-votes").inner_text())
    skipped0 = int(page.locator("#stat-skipped").inner_text())

    # nothing on the page says which option is the model's pick
    stage = page.locator("#study-stage").evaluate("el => el.outerHTML").lower()
    assert "model" not in stage and "ai pick" not in stage

    page.locator("#panel-study .panel-lead").click()       # focus inside the panel, not on the tabs
    page.wait_for_timeout(1100)                                 # vote like a person (>= 1 s); faster votes are excluded
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#stat-votes")).to_have_text(str(votes0 + 1))
    expect(page.locator("#study-round")).to_have_text("Round 2")
    expect(page.locator("#study-right")).to_be_enabled()
    expect(tab(page, "Style Study")).to_have_attribute("aria-selected", "true")

    page.keyboard.press("s")
    expect(page.locator("#stat-skipped")).to_have_text(str(skipped0 + 1))
    expect(page.locator("#study-round")).to_have_text("Round 3")
    expect(page.locator("#study-left")).to_be_enabled()

    page.wait_for_timeout(1100)                                 # vote like a person (>= 1 s); faster votes are excluded
    page.keyboard.press("ArrowRight")
    expect(page.locator("#stat-votes")).to_have_text(str(votes0 + 2))
    expect(page.locator("#stat-rate")).to_have_text(re.compile(r"^\d{1,3}% \(\d+ of \d+\)$"))
    expect(page.locator("#stat-ci")).to_have_text(re.compile(r"^\d{1,3}%–\d{1,3}%$"))
    # a mouse vote keeps focus on the clicked button for the next pair
    page.wait_for_timeout(1100)                                 # vote like a person (>= 1 s); faster votes are excluded
    page.locator("#study-right").click()
    expect(page.locator("#stat-votes")).to_have_text(str(votes0 + 3))
    expect(page.locator("#study-right")).to_be_focused()

    res = page.request.get("/study/results").json()
    lo, hi = res["ci95"]
    reading = page.locator("#study-reading")
    if lo <= 0.5 <= hi:
        expect(reading).to_contain_text("No clear preference yet")
    else:
        expect(reading).to_contain_text("prefer")


# ------------------------------------------------------------------ responsive + screenshots
@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("device", ["desktop", "mobile"])
def test_layout_and_screenshots(make_page, seeded, options, device, scheme):
    width, height = (1280, 860) if device == "desktop" else (360, 780)
    page = make_page(width=width, height=height, color_scheme=scheme)
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    open_app(page)
    run_suggest(page, count=3)
    page.wait_for_function(IMAGES_LOADED, arg="#styled-results img")
    assert page.evaluate(NO_HSCROLL), "horizontal scroll on Get Styled"
    page.screenshot(path=SCREENSHOTS / f"{device}-{scheme}-get-styled.png")

    names = ["My Wardrobe", "Add Clothes", "Saved Outfits"] + (["Style Study"] if options["study_available"] else [])
    for name in names:
        tab(page, name).click()
        slug = name.lower().replace(" ", "-")
        if name == "My Wardrobe":
            expect(page.locator("#wardrobe-grid .item-card")).to_have_count(6)
            page.evaluate("() => document.querySelectorAll('img[loading=lazy]').forEach(i => i.loading = 'eager')")
            page.wait_for_function(IMAGES_LOADED, arg="#wardrobe-grid img")
        if name == "Style Study":
            expect(page.locator("#study-left")).to_be_enabled(timeout=60_000)
            page.wait_for_function(IMAGES_LOADED, arg="#study-stage img")
        page.wait_for_timeout(100)
        assert page.evaluate(NO_HSCROLL), f"horizontal scroll on {name} at {width}px"
        if name in ("My Wardrobe", "Style Study") or device == "mobile":
            page.screenshot(path=SCREENSHOTS / f"{device}-{scheme}-{slug}.png")
