// AI Wardrobe Stylist frontend: plain ES module, no framework, no build step.
// Security: server data is only ever inserted with textContent / attributes via createElement,
// never innerHTML. All requests are same-origin.

const CATEGORY_LABELS = {
  tops: "Tops",
  bottoms: "Bottoms",
  "all-body": "Dresses & jumpsuits",
  shoes: "Shoes",
  outerwear: "Outerwear",
};
const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
const ALLOWED_MIME = ["image/jpeg", "image/png", "image/webp"];
const LOW_CONFIDENCE = 0.5;

const state = {
  options: null,
  items: [],               // newest first, as returned by GET /items
  itemsLoaded: false,
  filter: "all",
  savedKeys: new Map(),    // outfit key ("1-4-7") -> saved id
  uploadedOnce: false,
  uploadQueue: [],
  uploadsActive: 0,
  study: { token: null, busy: false, round: 0, loaded: false },
};

let resolveOptions;
const optionsReady = new Promise((r) => { resolveOptions = r; });

let uid = 0;
const nextId = (prefix) => `${prefix}-${++uid}`;
const $ = (sel, root = document) => root.querySelector(sel);

// ------------------------------------------------------------------ DOM helper
/** h("div", {class: "x", "aria-label": "y"}, child, "text") -> element. Text is always text. */
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on")) {                 // handlers only as functions, never as strings
      if (typeof v === "function") el.addEventListener(k.slice(2), v);
    }
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

/** Only allow same-origin image URLs from the server. Resolving with the URL parser (rather than
 *  string checks) also catches tricks like "/\t/evil.example", which browsers read as "//evil.example". */
function safeUrl(url) {
  const s = String(url ?? "");
  try {
    const u = new URL(s, location.href);
    return u.origin === location.origin ? u.pathname + u.search : "";
  } catch {
    return "";
  }
}

const itemLabel = (it) => `${it.color} ${it.type}`;
const pct = (x) => `${Math.round(Number(x) * 100)}%`;
const outfitKey = (ids) => [...ids].map(Number).sort((a, b) => a - b).join("-");

// ------------------------------------------------------------------ API
class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

function errorText(data, status, statusText) {
  const d = data && data.detail;
  if (typeof d === "string" && d) return d;
  if (Array.isArray(d) && d.length) {
    return d.map((e) => {
      const where = (e.loc || []).filter((x) => x !== "body" && x !== "query" && x !== "path").join(".");
      return where ? `${where}: ${e.msg}` : String(e.msg || "Invalid value");
    }).join("; ");
  }
  if (status === 413) return "Image too large (max 15 MB)";
  return `Request failed (${status}${statusText ? " " + statusText : ""})`;
}

async function api(path, { method = "GET", json, signal } = {}) {
  const opts = { method, headers: { Accept: "application/json" }, credentials: "same-origin", signal };
  if (json !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(json);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("Can't reach the server. Is it still running?", 0);
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(errorText(data, res.status, res.statusText), res.status);
  return data;
}

/** Multipart upload with progress (fetch has no upload progress events). */
function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/items");
    xhr.responseType = "json";
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.addEventListener("progress", (e) => { if (e.lengthComputable) onProgress("upload", e.loaded / e.total); });
    xhr.upload.addEventListener("load", () => onProgress("analyzing", 1));
    xhr.addEventListener("load", () => {
      const data = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300 && data) resolve(data);
      else reject(new ApiError(errorText(data, xhr.status, xhr.statusText), xhr.status));
    });
    xhr.addEventListener("error", () => reject(new ApiError("Network error: the upload did not reach the server.", 0)));
    const fd = new FormData();
    fd.append("file", file, file.name || "photo");
    xhr.send(fd);
  });
}

// ------------------------------------------------------------------ small UI utilities
// aria-disabled instead of `disabled`: a disabled button loses keyboard focus to <body>.
const isInactive = (btn) => btn.disabled || btn.getAttribute("aria-disabled") === "true";
function setBusy(btn, busy, busyLabel) {
  if (!btn) return;
  if (busy) btn.setAttribute("aria-disabled", "true");
  else btn.removeAttribute("aria-disabled");
  btn.setAttribute("aria-busy", String(busy));
  const label = btn.querySelector(".btn-label");
  if (label) {
    if (busy && busyLabel) { btn.dataset.idleLabel = label.textContent; label.textContent = busyLabel; }
    else if (!busy && btn.dataset.idleLabel) { label.textContent = btn.dataset.idleLabel; delete btn.dataset.idleLabel; }
  }
}

function announce(el, text) {
  // clear first so identical consecutive messages are still announced
  el.textContent = "";
  window.setTimeout(() => { el.textContent = text; }, 30);
}

let toastTimer = 0;
function toast(text) {
  const t = $("#toast");
  t.textContent = text;
  t.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { t.hidden = true; }, 2600);
}

function showError(box, message, retry) {
  box.replaceChildren(h("span", { text: message }));
  if (retry) box.append(h("button", { type: "button", class: "btn small", onclick: retry }, "Try again"));
  box.hidden = false;
}
const hideError = (box) => { box.hidden = true; box.replaceChildren(); };

function emptyState(title, body, actions = []) {
  return h("div", { class: "empty-state" },
    h("p", { class: "empty-title", text: title }),
    body ? h("p", { text: body }) : null,
    actions.length ? h("div", { class: "empty-actions" }, actions) : null);
}

function actionButton(label, onClick, primary = false) {
  return h("button", { type: "button", class: primary ? "btn primary" : "btn", onclick: onClick }, label);
}

function pieceImage(it, { lazy = false } = {}) {
  return h("img", {
    src: safeUrl(it.image), alt: itemLabel(it), width: 300, height: 300, decoding: "async",
    loading: lazy ? "lazy" : null,
  });
}

let confirmResolve = null;
function confirmDialog({ title, body, confirmLabel = "Delete" }) {
  const dlg = $("#confirm-dialog");
  $("#confirm-title").textContent = title;
  $("#confirm-desc").textContent = body;
  $("#confirm-ok").textContent = confirmLabel;
  const opener = document.activeElement;
  dlg.returnValue = "";
  dlg.showModal();
  return new Promise((resolve) => {
    confirmResolve = (ok) => {
      confirmResolve = null;
      if (opener && opener.isConnected && typeof opener.focus === "function") opener.focus();
      resolve(ok);
    };
  });
}
$("#confirm-dialog").addEventListener("close", () => {
  if (confirmResolve) confirmResolve($("#confirm-dialog").returnValue === "confirm");
});

// ------------------------------------------------------------------ tabs (ARIA tabs, roving tabindex)
const tablist = $("#tablist");
const tabs = () => [...tablist.querySelectorAll('[role="tab"]')].filter((t) => !t.hidden);
let currentTab = "styled";

function selectTab(name, { focus = false, updateHash = true } = {}) {
  const tab = tabs().find((t) => t.dataset.tab === name) || tabs()[0];
  name = tab.dataset.tab;
  for (const t of tablist.querySelectorAll('[role="tab"]')) {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $(`#${t.getAttribute("aria-controls")}`).hidden = !on;
  }
  if (focus) {
    tab.focus();
    tab.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
  if (updateHash && location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  const changed = currentTab !== name;
  currentTab = name;
  onShowTab(name, changed);
}

tablist.addEventListener("click", (e) => {
  const tab = e.target.closest('[role="tab"]');
  if (tab) selectTab(tab.dataset.tab);
});
tablist.addEventListener("keydown", (e) => {
  const list = tabs();
  const i = list.indexOf(document.activeElement);
  if (i < 0) return;
  let j = null;
  if (e.key === "ArrowRight") j = (i + 1) % list.length;
  else if (e.key === "ArrowLeft") j = (i - 1 + list.length) % list.length;
  else if (e.key === "Home") j = 0;
  else if (e.key === "End") j = list.length - 1;
  if (j === null) return;
  e.preventDefault();
  selectTab(list[j].dataset.tab, { focus: true });
});
window.addEventListener("hashchange", () => {
  const name = location.hash.slice(1);
  if (name && name !== currentTab) selectTab(name, { updateHash: false });
});

function goTo(name) {
  selectTab(name, { focus: true });
}

function onShowTab(name, changed) {
  if (!state.options) return;          // init() renders once options arrive
  if (name === "wardrobe") loadWardrobe();
  else if (name === "saved") loadSaved();
  else if (name === "study") startStudy();
}

// ------------------------------------------------------------------ items state
async function fetchItems() {
  state.items = await api("/items");
  state.itemsLoaded = true;
  return state.items;
}

function upsertItem(item) {
  const i = state.items.findIndex((x) => x.id === item.id);
  if (i >= 0) state.items[i] = item;
  else state.items.unshift(item);
}

// ------------------------------------------------------------------ item card (editable)
function typeSelect(current, id) {
  const sel = h("select", { id, name: "type" });
  const byCat = new Map(state.options.categories.map((c) => [c, []]));
  for (const [t, info] of Object.entries(state.options.types)) {
    if (!byCat.has(info.category)) byCat.set(info.category, []);
    byCat.get(info.category).push(t);
  }
  for (const [cat, types] of byCat) {
    if (!types.length) continue;
    const group = h("optgroup", { label: CATEGORY_LABELS[cat] || cat });
    for (const t of types) group.append(h("option", { value: t, text: t }));
    sel.append(group);
  }
  sel.value = current;
  return sel;
}

function colorSelect(current, id) {
  const sel = h("select", { id, name: "color" });
  const colors = [...state.options.colors];
  if (current && !colors.includes(current)) colors.unshift(current);
  for (const c of colors) sel.append(h("option", { value: c, text: c }));
  sel.value = current;
  return sel;
}

function itemCard(item) {
  const typeId = nextId("type"), colorId = nextId("color"), titleId = nextId("item-title");
  const card = h("article", { class: "item-card", "aria-labelledby": titleId, dataset: { itemId: item.id } });
  const img = h("img", { width: 300, height: 300, decoding: "async", loading: "lazy" });
  const title = h("h3", { class: "item-title", id: titleId });
  const meta = h("p", { class: "item-meta" });
  const conf = h("p", { class: "confidence" });
  const hint = h("p", { class: "low-conf", hidden: true });
  const tSel = typeSelect(item.type, typeId);
  const cSel = colorSelect(item.color, colorId);
  const msg = h("p", { class: "inline-msg", role: "status", "aria-live": "polite" });
  const del = h("button", { type: "button", class: "btn small" }, "Delete");
  del.setAttribute("aria-describedby", titleId);

  card.append(
    h("div", { class: "img-well" }, img),
    h("div", { class: "item-body" },
      title, meta, conf, hint,
      h("div", { class: "item-controls" },
        h("div", { class: "field" }, h("label", { for: typeId, text: "Type" }), tSel),
        h("div", { class: "field" }, h("label", { for: colorId, text: "Color" }), cSel)),
      msg,
      h("div", { class: "item-actions" }, del)));

  card._item = item;
  fillCard(card, item);

  const patch = async (body, sel) => {
    const prev = card._item;
    // selects stay enabled: disabling the focused <select> would drop keyboard focus on <body>
    card.setAttribute("aria-busy", "true");
    del.setAttribute("aria-disabled", "true");
    msg.className = "inline-msg";
    msg.textContent = "Saving…";
    try {
      const updated = await api(`/items/${encodeURIComponent(prev.id)}`, { method: "PATCH", json: body });
      upsertItem(updated);
      if (body.type) userTyped.add(updated.id);
      for (const c of document.querySelectorAll(".item-card")) if (c._item && c._item.id === updated.id) fillCard(c, updated);
      renderChips();
      msg.className = "inline-msg is-ok";
      msg.textContent = "Saved";
    } catch (e) {
      sel.value = body.type ? prev.type : prev.color;       // revert
      msg.className = "inline-msg is-error";
      msg.textContent = e.message;
    } finally {
      card.removeAttribute("aria-busy");
      del.removeAttribute("aria-disabled");
    }
  };
  tSel.addEventListener("change", () => patch({ type: tSel.value }, tSel));
  cSel.addEventListener("change", () => patch({ color: cSel.value }, cSel));

  del.addEventListener("click", async () => {
    const it = card._item;
    const ok = await confirmDialog({
      title: "Delete this item?",
      body: `The ${itemLabel(it)} and its photo will be removed from your wardrobe. This can't be undone.`,
    });
    if (!ok || isInactive(del)) return;
    del.setAttribute("aria-disabled", "true");
    card.setAttribute("aria-busy", "true");
    try {
      await api(`/items/${encodeURIComponent(it.id)}`, { method: "DELETE" });
      state.items = state.items.filter((x) => x.id !== it.id);
      removeItemCards(it.id);
      renderChips();
      if (!$("#wardrobe-grid .item-card")) renderWardrobeGrid();
      toast(`Deleted ${itemLabel(it)}`);
    } catch (e) {
      msg.className = "inline-msg is-error";
      msg.textContent = e.message;
      del.removeAttribute("aria-disabled");
      card.removeAttribute("aria-busy");
    }
  });
  return card;
}

const userTyped = new Set();       // item ids whose type the user picked in this session

function fillCard(card, item) {
  card._item = item;
  const img = $("img", card);
  const src = safeUrl(item.image);
  if (img.getAttribute("src") !== src) img.src = src;
  img.alt = itemLabel(item);
  $(".item-title", card).textContent = itemLabel(item);
  $(".item-meta", card).textContent =
    `${CATEGORY_LABELS[item.category] || item.category} · formality ${item.formality}/5 · warmth ${item.warmth}/5`;
  const c = Number(item.type_confidence);
  const mine = userTyped.has(item.id);         // the tagger confidence no longer applies
  $(".confidence", card).textContent = mine ? "Type set by you"
    : Number.isFinite(c) ? `Type detected with ${pct(c)} confidence` : "";
  const hint = $(".low-conf", card);
  const low = !mine && Number.isFinite(c) && c < LOW_CONFIDENCE;
  hint.hidden = !low;
  hint.textContent = low ? "The AI isn't sure about this type. Please check it." : "";
  const [tSel, cSel] = card.querySelectorAll("select");
  if (![...cSel.options].some((o) => o.value === item.color)) cSel.prepend(h("option", { value: item.color, text: item.color }));
  tSel.value = item.type;
  cSel.value = item.color;
}

function removeItemCards(id) {
  for (const c of [...document.querySelectorAll(".item-card")]) {
    if (!c._item || c._item.id !== id) continue;
    // keep keyboard focus nearby instead of dropping it on <body>
    const hadFocus = c.contains(document.activeElement) || document.activeElement === document.body;
    const next = c.nextElementSibling || c.previousElementSibling;
    c.remove();
    if (hadFocus && next && next.isConnected && !next.closest("[hidden]")) $("select, button", next)?.focus();
  }
  if (!$("#just-added .item-card")) $("#just-added-wrap").hidden = true;
}

// ------------------------------------------------------------------ My Wardrobe
function renderChips() {
  const box = $("#filter-chips");
  const counts = { all: state.items.length };
  for (const c of state.options.categories) counts[c] = 0;
  for (const it of state.items) counts[it.category] = (counts[it.category] || 0) + 1;
  const entries = [["all", "All"], ...state.options.categories.map((c) => [c, CATEGORY_LABELS[c] || c])];
  const focusedCat = document.activeElement?.closest?.(".chip")?.dataset.category;
  box.replaceChildren(...entries.map(([cat, label]) => {
    const n = counts[cat] || 0;
    const chip = h("button", {
      type: "button", class: "chip", "aria-pressed": String(state.filter === cat), dataset: { category: cat },
      "aria-label": `${label}, ${n} item${n === 1 ? "" : "s"}`,
    }, h("span", { class: "chip-label", text: label }), h("span", { class: "count", "aria-hidden": "true", text: n }));
    chip.addEventListener("click", () => {
      state.filter = cat;
      renderChips();
      renderWardrobeGrid();
      announce($("#wardrobe-status"), `Showing ${label.toLowerCase()}: ${n} item${n === 1 ? "" : "s"}`);
    });
    return chip;
  }));
  if (focusedCat) box.querySelector(`[data-category="${CSS.escape(focusedCat)}"]`)?.focus();
}

function renderWardrobeGrid() {
  const grid = $("#wardrobe-grid");
  const shown = state.filter === "all" ? state.items : state.items.filter((it) => it.category === state.filter);
  if (!state.items.length) {
    grid.replaceChildren(emptyState("Your wardrobe is empty",
      "Add photos of your clothes and they'll show up here, ready to mix into outfits.",
      [actionButton("Add clothes", () => goTo("add"), true)]));
    return;
  }
  if (!shown.length) {
    const label = (CATEGORY_LABELS[state.filter] || state.filter).toLowerCase();
    grid.replaceChildren(emptyState(`No ${label} yet`, "Add some, or look at everything you have.", [
      actionButton("Add clothes", () => goTo("add"), true),
      actionButton("Show all", () => { state.filter = "all"; renderChips(); renderWardrobeGrid(); }),
    ]));
    return;
  }
  grid.replaceChildren(...shown.map(itemCard));
}

function skeletonCards(n) {
  return Array.from({ length: n }, () => h("div", { class: "item-card", "aria-hidden": "true" },
    h("div", { class: "skeleton sk-img" }),
    h("div", { class: "item-body" }, h("div", { class: "skeleton sk-line" }), h("div", { class: "skeleton sk-line short" }))));
}

let wardrobeLoad = 0;
async function loadWardrobe() {
  const grid = $("#wardrobe-grid");
  const err = $("#wardrobe-error");
  const ticket = ++wardrobeLoad;
  hideError(err);
  if (!state.itemsLoaded) {
    grid.setAttribute("aria-busy", "true");
    grid.replaceChildren(...skeletonCards(6));
  }
  try {
    await fetchItems();
    if (ticket !== wardrobeLoad) return;
    renderChips();
    // don't rebuild under the user's focus (e.g. mid-edit) if nothing changed
    const ids = [...grid.querySelectorAll(".item-card")].map((c) => c._item?.id).join(",");
    const want = (state.filter === "all" ? state.items : state.items.filter((it) => it.category === state.filter))
      .map((it) => it.id).join(",");
    if (ids !== want || !ids) renderWardrobeGrid();
    else for (const c of grid.querySelectorAll(".item-card")) fillCard(c, state.items.find((x) => x.id === c._item.id));
  } catch (e) {
    if (ticket !== wardrobeLoad) return;
    grid.replaceChildren();
    showError(err, `Couldn't load your wardrobe: ${e.message}`, loadWardrobe);
  } finally {
    if (ticket === wardrobeLoad) grid.setAttribute("aria-busy", "false");
  }
}

// ------------------------------------------------------------------ outfit card
function stripScoreSentence(text) {
  // the server's explanation ends with "NN% compatibility estimate." which the card already shows
  return String(text ?? "").replace(/\s*\d{1,3}% compatibility estimate\.?\s*$/i, "").trim();
}

function outfitCard(outfit, { index, saved = false } = {}) {
  const titleId = nextId("outfit");
  const items = outfit.items || [];
  const score = Math.max(0, Math.min(1, Number(outfit.score) || 0));
  const card = h("article", { class: "outfit-card", "aria-labelledby": titleId });
  const name = saved ? `Saved outfit${index ? " " + index : ""}` : `Outfit ${index}`;
  card.append(
    h("div", { class: "outfit-head" },
      h("h3", { id: titleId, text: name }),
      h("p", { class: "score" }, h("strong", { text: pct(score) }), " compatibility estimate")),
    h("div", { class: "score-bar", "aria-hidden": "true" }, h("span", { style: `width:${Math.round(score * 100)}%` })),
    items.length
      ? h("ul", { class: "pieces", "aria-label": "Pieces" }, items.map((it) =>
        h("li", {}, h("figure", {}, h("div", { class: "img-well" }, pieceImage(it)), h("figcaption", { text: itemLabel(it) })))))
      : h("p", { class: "hint", text: "The pieces in this outfit were deleted from your wardrobe." }),
    h("p", { class: "explanation", text: stripScoreSentence(outfit.explanation) }));

  const actions = h("div", { class: "outfit-actions" });
  const msg = h("p", { class: "inline-msg", role: "status", "aria-live": "polite" });
  const ids = (outfit.item_ids || items.map((i) => i.id)).map(Number);

  if (!saved) {
    const like = h("button", { type: "button", class: "btn small vote up", "aria-pressed": "false" },
      h("span", { "aria-hidden": "true", text: "👍" }), " Like");
    const dislike = h("button", { type: "button", class: "btn small vote down", "aria-pressed": "false" },
      h("span", { "aria-hidden": "true", text: "👎" }), " Not for me");
    like.setAttribute("aria-describedby", titleId);
    dislike.setAttribute("aria-describedby", titleId);
    const vote = async (btn, other, value) => {
      if (btn.getAttribute("aria-pressed") === "true" || isInactive(btn)) return;   // already recorded / in flight
      like.setAttribute("aria-disabled", "true");
      dislike.setAttribute("aria-disabled", "true");
      msg.className = "inline-msg";
      msg.textContent = "";
      try {
        await api("/feedback", { method: "POST", json: { item_ids: ids, vote: value } });
        btn.setAttribute("aria-pressed", "true");
        other.setAttribute("aria-pressed", "false");
        msg.className = "inline-msg is-ok";
        msg.textContent = value > 0 ? "Thanks! We'll remember you liked this." : "Got it. Thanks for the feedback.";
      } catch (e) {
        msg.className = "inline-msg is-error";
        msg.textContent = e.message;
      } finally {
        like.removeAttribute("aria-disabled");
        dislike.removeAttribute("aria-disabled");
      }
    };
    like.addEventListener("click", () => vote(like, dislike, 1));
    dislike.addEventListener("click", () => vote(dislike, like, -1));

    const save = h("button", { type: "button", class: "btn small save-btn" },
      h("span", { class: "spinner", "aria-hidden": "true" }), h("span", { class: "btn-label", text: "Save outfit" }));
    save.setAttribute("aria-describedby", titleId);
    const markSaved = () => {
      save.dataset.saved = "true";
      save.setAttribute("aria-disabled", "true");
      $(".btn-label", save).textContent = "Saved ✓";
    };
    if (state.savedKeys.has(outfitKey(ids))) markSaved();
    save.addEventListener("click", async () => {
      if (isInactive(save)) return;
      setBusy(save, true, "Saving…");
      msg.className = "inline-msg";
      msg.textContent = "";
      try {
        const res = await api("/saved", {
          method: "POST",
          json: { item_ids: ids, score, explanation: String(outfit.explanation ?? "").slice(0, 500) },
        });
        state.savedKeys.set(outfitKey(ids), res.id);
        setBusy(save, false);
        markSaved();
        toast("Outfit saved");
      } catch (e) {
        setBusy(save, false);
        msg.className = "inline-msg is-error";
        msg.textContent = e.message;
      }
    });
    actions.append(like, dislike, save);
  } else {
    const remove = h("button", { type: "button", class: "btn small" },
      h("span", { class: "spinner", "aria-hidden": "true" }), h("span", { class: "btn-label", text: "Remove" }));
    remove.setAttribute("aria-describedby", titleId);
    remove.addEventListener("click", async () => {
      if (isInactive(remove)) return;
      setBusy(remove, true, "Removing…");
      try {
        await api(`/saved/${encodeURIComponent(outfit.id)}`, { method: "DELETE" });
        for (const [k, v] of state.savedKeys) if (v === outfit.id) state.savedKeys.delete(k);
        const next = card.nextElementSibling || card.previousElementSibling;
        card.remove();
        announce($("#saved-status"), "Outfit removed");
        if (!$("#saved-list .outfit-card")) renderSavedEmpty();
        else $("button", next)?.focus();
      } catch (e) {
        setBusy(remove, false);
        msg.className = "inline-msg is-error";
        msg.textContent = e.message;
      }
    });
    actions.append(remove);
  }
  card.append(actions, msg);
  return card;
}

function outfitSkeletons(n) {
  return Array.from({ length: n }, () => h("div", { class: "outfit-card sk-outfit", "aria-hidden": "true" },
    h("div", { class: "skeleton sk-line short" }),
    h("ul", { class: "pieces" }, [0, 1, 2].map(() => h("li", {}, h("div", { class: "skeleton" })))),
    h("div", { class: "skeleton sk-line" })));
}

// ------------------------------------------------------------------ Get Styled
async function refreshSavedKeys() {
  try {
    const saved = await api("/saved");
    state.savedKeys = new Map(saved.map((s) => [outfitKey(s.items.map((i) => i.id)), s.id]));
  } catch { /* only used to mark already-saved outfits */ }
}

function readCount() {
  const input = $("#count");
  const n = Number(input.value);
  const ok = Number.isInteger(n) && n >= 1 && n <= 20;
  input.setAttribute("aria-invalid", String(!ok));
  const err = $("#count-error");
  err.hidden = ok;
  err.textContent = ok ? "" : "How many outfits: enter a whole number from 1 to 20.";
  if (!ok) input.setAttribute("aria-describedby", "count-error");
  else input.removeAttribute("aria-describedby");
  return ok ? n : null;
}

let styledController = null;
$("#styled-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const n = readCount();
  if (n === null) { $("#count").focus(); return; }
  const occasion = $("#occasion").value, weather = $("#weather").value;
  const box = $("#styled-results"), btn = $("#suggest-btn");
  if (isInactive(btn)) return;
  styledController?.abort();
  const ctrl = styledController = new AbortController();
  setBusy(btn, true, "Finding outfits…");
  box.setAttribute("aria-busy", "true");
  box.replaceChildren(...outfitSkeletons(Math.min(n, 3)));
  announce($("#styled-status"), "Finding outfits…");
  try {
    const [res] = await Promise.all([
      api("/recommend", { method: "POST", json: { occasion, weather, n }, signal: ctrl.signal }),
      fetchItems(),
      refreshSavedKeys(),
    ]);
    if (ctrl !== styledController) return;
    const outfits = res.outfits || [];
    if (!outfits.length) {
      box.replaceChildren(noOutfitsState(occasion, weather));
      announce($("#styled-status"), "No outfits found.");
    } else {
      box.replaceChildren(...outfits.map((o, i) => outfitCard(o, { index: i + 1 })));
      announce($("#styled-status"), `${outfits.length} outfit${outfits.length === 1 ? "" : "s"} found.`);
    }
  } catch (err) {
    if (err.name === "AbortError" || ctrl !== styledController) return;
    box.replaceChildren(h("div", { class: "error-inline", role: "alert" },
      h("span", { text: `Couldn't get suggestions: ${err.message}` }),
      h("button", { type: "button", class: "btn small", onclick: () => $("#styled-form").requestSubmit() }, "Try again")));
  } finally {
    if (ctrl === styledController) {
      setBusy(btn, false);
      box.setAttribute("aria-busy", "false");
    }
  }
});
$("#count").addEventListener("input", () => { if ($("#count").getAttribute("aria-invalid") === "true") readCount(); });

function noOutfitsState(occasion, weather) {
  if (!state.items.length) {
    return emptyState("Your wardrobe is empty",
      "Add a top and a bottom (or a dress) and you'll get outfit suggestions right away.",
      [actionButton("Add clothes", () => goTo("add"), true)]);
  }
  const what = [occasion !== "any" ? `a ${occasion} occasion` : null, weather !== "any" ? `${weather} weather` : null]
    .filter(Boolean).join(" in ");
  const filtered = occasion !== "any" || weather !== "any";
  return emptyState(
    filtered ? `No outfits fit ${what}` : "No complete outfits yet",
    "An outfit needs a top and a bottom, or one dress or jumpsuit, that suit the occasion and weather. "
      + "Correct item types in My Wardrobe, add more clothes, or loosen the filters.",
    [
      filtered ? actionButton("Show any outfit", () => {
        $("#occasion").value = "any"; $("#weather").value = "any"; $("#styled-form").requestSubmit();
      }, true) : null,
      actionButton("Add clothes", () => goTo("add"), !filtered),
      actionButton("Edit wardrobe", () => goTo("wardrobe")),
    ].filter(Boolean));
}

// ------------------------------------------------------------------ Add Clothes
const dropzone = $("#dropzone");
const fileInput = $("#file-input");

fileInput.addEventListener("change", () => {
  queueFiles([...fileInput.files]);
  fileInput.value = "";
});
for (const ev of ["dragenter", "dragover"]) {
  dropzone.addEventListener(ev, (e) => {
    if (![...(e.dataTransfer?.types || [])].includes("Files")) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    dropzone.classList.add("is-over");
  });
}
dropzone.addEventListener("dragleave", (e) => {
  if (!dropzone.contains(e.relatedTarget)) dropzone.classList.remove("is-over");
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("is-over");
  queueFiles([...(e.dataTransfer?.files || [])]);
});
// a file dropped next to the drop zone must not navigate away from the app
window.addEventListener("dragover", (e) => { if (!dropzone.contains(e.target)) e.preventDefault(); });
window.addEventListener("drop", (e) => { if (!dropzone.contains(e.target)) e.preventDefault(); });

function queueFiles(files) {
  if (!files.length) return;
  const list = $("#upload-list");
  for (const file of files) {
    const stateEl = h("span", { class: "upload-state", text: "Waiting" });
    const bar = h("progress", { max: 100, value: 0, "aria-label": `Upload progress for ${file.name}` });
    const row = h("li", { class: "upload-row", dataset: { state: "queued" } },
      h("span", { class: "upload-name", title: file.name, text: file.name || "photo" }), stateEl, bar);
    list.prepend(row);
    const problem = !ALLOWED_MIME.includes(file.type)
      ? "Not a JPEG, PNG or WEBP image."
      : file.size > MAX_UPLOAD_BYTES ? "Too large: the limit is 15 MB." : null;
    if (problem) { failRow(row, problem); continue; }
    state.uploadQueue.push({ file, row, stateEl, bar });
  }
  announce($("#upload-status"), `${files.length} photo${files.length === 1 ? "" : "s"} queued.`);
  pumpUploads();
}

function failRow(row, message) {
  row.dataset.state = "error";
  $(".upload-state", row).textContent = "Failed";
  row.append(h("p", { class: "upload-error", text: message }));
}

function pumpUploads() {
  // The first upload of a session loads the AI models: run it alone, then two at a time
  // (the server tags at most two photos at once).
  const limit = state.uploadedOnce ? 2 : 1;
  while (state.uploadsActive < limit && state.uploadQueue.length) {
    runUpload(state.uploadQueue.shift());
  }
  $("#first-upload-note").hidden = state.uploadedOnce || !(state.uploadsActive || state.uploadQueue.length);
}

async function runUpload(job) {
  const { file, row, stateEl, bar } = job;
  state.uploadsActive++;
  row.dataset.state = "uploading";
  stateEl.textContent = "Uploading 0%";
  try {
    const item = await uploadFile(file, (phase, frac) => {
      if (phase === "upload") {
        bar.value = Math.round(frac * 90);
        stateEl.textContent = `Uploading ${Math.round(frac * 100)}%`;
      } else {
        bar.removeAttribute("value");          // indeterminate while the server analyses the photo
        row.dataset.state = "analyzing";
        stateEl.textContent = "Analyzing…";
      }
    });
    await optionsReady;                         // the editable card needs the type/color lists
    state.uploadedOnce = true;
    upsertItem(item);
    row.dataset.state = "done";
    bar.value = 100;
    stateEl.textContent = "Added";
    row.dataset.itemId = item.id;
    $("#just-added-wrap").hidden = false;
    $("#just-added").prepend(itemCard(item));
    announce($("#upload-status"), `Added ${itemLabel(item)}.`);
  } catch (e) {
    failRow(row, e.message);
    announce($("#upload-status"), `${file.name} failed: ${e.message}`);
  } finally {
    state.uploadsActive--;
    pumpUploads();
  }
}

// ------------------------------------------------------------------ Saved Outfits
function renderSavedEmpty() {
  $("#saved-list").replaceChildren(emptyState("No saved outfits yet",
    "Save the outfits you like from Get Styled and they'll be kept here.",
    [actionButton("Get styled", () => goTo("styled"), true)]));
}

let savedLoad = 0;
async function loadSaved() {
  const list = $("#saved-list"), err = $("#saved-error");
  const ticket = ++savedLoad;
  hideError(err);
  list.setAttribute("aria-busy", "true");
  if (!list.querySelector(".outfit-card")) list.replaceChildren(...outfitSkeletons(2));
  try {
    const saved = await api("/saved");
    if (ticket !== savedLoad) return;
    state.savedKeys = new Map(saved.map((s) => [outfitKey(s.items.map((i) => i.id)), s.id]));
    if (!saved.length) renderSavedEmpty();
    else list.replaceChildren(...saved.map((o, i) => outfitCard(o, { index: i + 1, saved: true })));
  } catch (e) {
    if (ticket !== savedLoad) return;
    list.replaceChildren();
    showError(err, `Couldn't load saved outfits: ${e.message}`, loadSaved);
  } finally {
    if (ticket === savedLoad) list.setAttribute("aria-busy", "false");
  }
}

// ------------------------------------------------------------------ Style Study
const studyButtons = () => [$("#study-left"), $("#study-right"), $("#study-skip")];

function setStudyEnabled(on) {
  for (const b of studyButtons()) b.disabled = !on;
}

function loadImage(img, url) {
  return new Promise((resolve) => {
    const src = safeUrl(url);
    if (!src) { img.removeAttribute("src"); resolve(); return; }
    const done = () => { img.removeEventListener("load", done); img.removeEventListener("error", done); resolve(); };
    img.addEventListener("load", done);
    img.addEventListener("error", done);
    img.src = src;
  });
}

async function startStudy() {
  // keep an unvoted pair when the user comes back to the tab; always refresh the tally
  const pending = state.study.token || state.study.busy ? null : nextStudyRound();
  await Promise.all([pending, loadStudyResults()]);
}

async function nextStudyRound() {
  const stage = $("#study-stage"), err = $("#study-error");
  state.study.busy = true;
  state.study.token = null;
  setStudyEnabled(false);
  stage.setAttribute("aria-busy", "true");
  hideError(err);
  try {
    const r = await api("/study/next", { method: "POST", json: {} });   // POST: creates a round
    await Promise.all([
      loadImage($("#study-top-img"), r.top),
      loadImage($("#study-left-img"), r.left),
      loadImage($("#study-right-img"), r.right),
    ]);
    for (const b of studyButtons()) delete b.dataset.picked;
    state.study.token = r.token;
    state.study.round++;
    $("#study-round").textContent = `Round ${state.study.round}`;
    stage.setAttribute("aria-busy", "false");
    setStudyEnabled(true);
  } catch (e) {
    showError(err, `Couldn't load the next pair: ${e.message}`, nextStudyRound);
  } finally {
    state.study.busy = false;
  }
}

async function loadStudyResults() {
  try {
    renderStudyResults(await api("/study/results"));
  } catch (e) {
    $("#study-reading").textContent = `Couldn't load results: ${e.message}`;
  }
}

function renderStudyResults(r) {
  const votes = Number(r.votes) || 0;
  $("#stat-votes").textContent = String(votes);
  $("#stat-skipped").textContent = String(Number(r.skipped) || 0);
  const point = $("#ci-point"), range = $("#ci-range");
  if (!votes || r.rate === null || r.rate === undefined) {
    $("#stat-rate").textContent = "–";
    $("#stat-ci").textContent = "–";
    point.style.display = range.style.display = "none";
    $("#study-reading").textContent = "No votes yet. Your first vote starts the tally.";
    return;
  }
  const rate = Number(r.rate);
  $("#stat-rate").textContent = `${pct(rate)} (${Number(r.model_preferred) || 0} of ${votes})`;
  let reading;
  if (Array.isArray(r.ci95) && r.ci95.length === 2) {
    const [lo, hi] = r.ci95.map(Number);
    $("#stat-ci").textContent = `${pct(lo)}–${pct(hi)}`;
    range.style.display = point.style.display = "block";
    range.style.left = `${lo * 100}%`;
    range.style.width = `${Math.max(0, hi - lo) * 100}%`;
    point.style.left = `${rate * 100}%`;
    if (lo > 0.5) reading = "People prefer the AI's picks: the whole interval is above 50%.";
    else if (hi < 0.5) reading = "People prefer the alternative: the whole interval is below 50%.";
    else reading = `No clear preference yet: the interval still includes 50%. ${votes < 30 ? "More votes will narrow it." : ""}`.trim();
  } else {
    $("#stat-ci").textContent = "–";
    reading = "Not enough votes for an interval yet.";
  }
  $("#study-reading").textContent = reading;
}

async function studyVote(choice) {
  const token = state.study.token;
  if (!token || state.study.busy) return;
  state.study.busy = true;
  state.study.token = null;
  // disabling the buttons drops focus on <body>; put it back on the same button afterwards
  const refocus = studyButtons().includes(document.activeElement) ? document.activeElement : null;
  setStudyEnabled(false);
  const btn = studyButtons().find((b) => b.dataset.choice === choice);
  if (btn) btn.dataset.picked = "true";
  try {
    await api("/study/vote", { method: "POST", json: { token, choice } });
    announce($("#study-status"), choice === "skip" ? "Skipped. Loading the next pair." : `Voted ${choice}. Loading the next pair.`);
  } catch (e) {
    if (e.status !== 409) {                // 409 = this round was already counted; just move on
      state.study.busy = false;
      state.study.token = token;
      setStudyEnabled(true);
      if (btn) delete btn.dataset.picked;
      showError($("#study-error"), `Your vote wasn't recorded: ${e.message}`);
      return;
    }
  }
  state.study.busy = false;
  await Promise.all([loadStudyResults(), nextStudyRound()]);
  if (refocus && !refocus.disabled && document.activeElement === document.body) refocus.focus();
}

for (const b of studyButtons()) b.addEventListener("click", () => studyVote(b.dataset.choice));

document.addEventListener("keydown", (e) => {
  if (currentTab !== "study" || e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.repeat) return;                       // a held-down key must not keep voting on new rounds
  if ($("#confirm-dialog").open) return;
  const t = e.target;
  if (t.closest?.('[role="tab"], input, select, textarea, [contenteditable="true"]')) return;
  const choice = { ArrowLeft: "left", ArrowRight: "right", s: "skip", S: "skip" }[e.key];
  if (!choice) return;
  e.preventDefault();
  studyVote(choice);
});

// ------------------------------------------------------------------ init
function fillSelect(sel, values, fallback = "any") {
  sel.replaceChildren(...values.map((v) => h("option", { value: v, text: v })));
  sel.value = values.includes(fallback) ? fallback : values[0];
}

function renderModelBadge(model) {
  const badge = $("#model-badge");
  if (model) {
    badge.dataset.state = "trained";
    $("#model-text").textContent = `Model ${model}`;
    badge.title = "Outfits are scored by the trained compatibility model";
  } else {
    badge.dataset.state = "fallback";
    $("#model-text").textContent = "Untrained fallback";
    badge.title = "No trained model found: scores use raw FashionCLIP similarity";
  }
}

async function init() {
  const banner = $("#global-error");
  banner.hidden = true;
  try {
    const opts = await api("/options");
    state.options = opts;
    fillSelect($("#occasion"), opts.occasions);
    fillSelect($("#weather"), opts.weather);
    $("#occasion").disabled = $("#weather").disabled = false;
    $("#suggest-btn").disabled = false;
    renderModelBadge(opts.model);
    $("#tab-study").hidden = !opts.study_available;
    resolveOptions();
  } catch (e) {
    $("#model-badge").dataset.state = "error";
    $("#model-text").textContent = "Offline";
    $("#global-error-text").textContent = `Couldn't connect to the stylist server: ${e.message}`;
    banner.hidden = false;
    return;
  }
  const want = location.hash.slice(1);
  selectTab(tabs().some((t) => t.dataset.tab === want) ? want : "styled", { updateHash: Boolean(want) });
  if (currentTab === "styled") {
    // keep the empty state honest: tell first-time users to add clothes before asking for outfits
    try {
      await fetchItems();
      if (!state.items.length && !$("#styled-results .outfit-card")) {
        $("#styled-results").replaceChildren(emptyState("Your wardrobe is empty",
          "Add a few photos of your clothes first; then come back here for outfit ideas.",
          [actionButton("Add clothes", () => goTo("add"), true)]));
      }
    } catch { /* the form still works; errors show on submit */ }
  }
  document.body.dataset.ready = "true";
}

$("#global-retry").addEventListener("click", init);
init();
