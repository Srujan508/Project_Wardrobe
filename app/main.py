"""Step 10: FastAPI backend.

Run:  venv\\Scripts\\uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs     App: http://127.0.0.1:8000
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import os
import secrets
import threading
import uuid
from urllib.parse import urlsplit
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Path as PathParam, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.db import FeedbackRow, ItemRow, SavedOutfitRow, make_session_factory
from app.study import build_router as build_study_router
from src.color import NAMED_COLORS, rgb_to_hsv01
from src.recommender import Item, recommend
from src.utils import load_config, project_path
from src.wardrobe_rules import APP_CATEGORIES, OCCASIONS, TYPES, WEATHER

CFG = load_config()
UPLOADS = project_path(CFG["app"]["uploads"])
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000            # 5000x5000 (> any phone photo); larger images are refused
ALLOWED_FORMATS = ["JPEG", "PNG", "WEBP"]
# Decoding + background removal + CLIP is memory-heavy (~hundreds of MB for a large image);
# cap how many uploads do it at once so concurrent requests can't exhaust memory.
UPLOAD_SLOTS = threading.BoundedSemaphore(2)
ALLOWED_HOSTS = ([h.strip() for h in os.environ["WARDROBE_ALLOWED_HOSTS"].split(",") if h.strip()]
                 if os.environ.get("WARDROBE_ALLOWED_HOSTS")
                 else CFG["app"].get("allowed_hosts", ["127.0.0.1", "localhost"]))
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
if os.environ.get("SPACE_ID") and not os.environ.get("WARDROBE_BASIC_AUTH"):
    # Hosted on a Hugging Face Space: the app has no accounts, so refuse to run without a login.
    raise RuntimeError("Set the WARDROBE_BASIC_AUTH secret (user:password) before running on a public host")
STATE: dict = {}


def _load_scorer():
    """The trained model if present, else None (recommender falls back to raw FashionCLIP)."""
    name = os.environ.get("WARDROBE_MODEL", CFG["app"]["model"])
    if not (project_path(CFG["paths"]["models"]) / f"{name}.pt").exists():
        print(f"[app] model '{name}' not found; using untrained FashionCLIP similarity")
        return None
    from src.inference import CompatScorer
    print(f"[app] loaded model '{name}'")
    return CompatScorer(name, CFG)


@asynccontextmanager
async def lifespan(_: FastAPI):
    (UPLOADS / "original").mkdir(parents=True, exist_ok=True)
    (UPLOADS / "clean").mkdir(parents=True, exist_ok=True)
    STATE["sessions"] = make_session_factory(os.environ.get("WARDROBE_DB"))
    STATE["scorer"] = _load_scorer()
    yield


app = FastAPI(title="AI Wardrobe Stylist v2", lifespan=lifespan)

# DNS-rebinding protection: only answer requests addressed to an allowed host name.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


def _basic_auth_ok(header: str | None) -> bool:
    """Optional login for hosted deployments: set WARDROBE_BASIC_AUTH="user:password"."""
    expected = os.environ.get("WARDROBE_BASIC_AUTH")
    if not expected:
        return True                                  # local use: no login
    scheme, _, value = (header or "").partition(" ")
    if scheme.lower() != "basic" or not value:
        return False
    try:
        given = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return secrets.compare_digest(given, expected.encode())


@app.middleware("http")
async def guard_requests(request: Request, call_next):
    if not _basic_auth_ok(request.headers.get("authorization")):
        return JSONResponse({"detail": "Login required"}, status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="wardrobe", charset="UTF-8"'})
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return await call_next(request)
    # CSRF: browsers send cross-site form posts without a CORS check. Require the same host AND port
    # as the request (another site, or another local port, can't change data). The scheme is not
    # compared: behind a TLS-terminating proxy the browser says https while the app sees http.
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host", ""):
        return JSONResponse({"detail": "Cross-origin request refused"}, status_code=403)
    # Refuse oversized bodies before they are spooled to disk / read into memory. Bodies without
    # a declared length (chunked) could bypass that, so they are refused too.
    length = request.headers.get("content-length")
    if length is None and request.method in ("POST", "PUT", "PATCH"):
        return JSONResponse({"detail": "Content-Length required"}, status_code=411)
    if length and (not length.isdigit() or int(length) > MAX_UPLOAD_BYTES + 64 * 1024):
        return JSONResponse({"detail": "Request too large (max 15 MB)"}, status_code=413)
    return await call_next(request)


SECURITY_HEADERS = {
    # only same-origin scripts/images/requests; no plugins, no <base> tricks, no framing
    "Content-Security-Policy": ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
                                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        if k == "Content-Security-Policy" and request.url.path in ("/docs", "/redoc"):
            continue                       # FastAPI's API docs page loads its UI from a CDN
        response.headers.setdefault(k, v)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    # FastAPI's default 422 body echoes the rejected input back; values like 1e999 (inf) then
    # crash JSON serialization with a 500. Return only where and why validation failed.
    errors = [{"loc": [str(x) for x in e.get("loc", ())], "msg": e.get("msg", ""), "type": e.get("type", "")}
              for e in exc.errors()]
    return JSONResponse({"detail": errors}, status_code=422)


UPLOADS.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def get_db():
    db = STATE["sessions"]()
    try:
        yield db
    finally:
        db.close()


app.include_router(build_study_router(get_db, CFG, lambda: _model_name()))


# ---------------------------------------------------------------- schemas
Id = Annotated[int, Field(ge=1, le=2**62)]
ItemIds = Annotated[list[Id], Field(min_length=1, max_length=8)]
PathId = Annotated[int, PathParam(ge=1, le=2**62)]


class ItemUpdate(BaseModel):
    type: str | None = Field(None, max_length=40)
    category: Literal["tops", "bottoms", "all-body", "shoes", "outerwear"] | None = None
    color: str | None = Field(None, min_length=1, max_length=40)
    formality: int | None = Field(None, ge=1, le=5)
    warmth: int | None = Field(None, ge=1, le=5)


class RecReq(BaseModel):
    occasion: Literal["any", "casual", "smart casual", "formal"] = "any"
    weather: Literal["any", "hot", "mild", "cold"] = "any"
    n: int = Field(5, ge=1, le=20)


class FeedbackReq(BaseModel):
    item_ids: ItemIds
    vote: Literal[1, -1]


class SaveReq(BaseModel):
    item_ids: ItemIds
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    explanation: str = Field(max_length=500)


def decode_upload(data: bytes) -> Image.Image:
    """Decode an uploaded image safely: known formats only, pixel cap checked from the header
    BEFORE decoding (explicit check: warning filters are process-wide, so not thread-safe)."""
    try:
        img = Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS)    # reads the header only
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise HTTPException(400, "Image has too many pixels")
        if img.format == "JPEG":
            img.draft("RGB", (2048, 2048))   # JPEG can decode straight at reduced size: far less memory
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Not a readable JPEG/PNG/WEBP image (or too many pixels)")
    img.thumbnail((2048, 2048))          # cap memory for everything downstream
    return img


# ---------------------------------------------------------------- items
@app.post("/items")
def add_item(file: UploadFile = File(...), db=Depends(get_db)):
    # sync handler: FastAPI runs it in a worker thread, so a slow upload doesn't block the server
    if (file.size or 0) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 15 MB)")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 15 MB)")
    from src.tagger import process_upload
    with UPLOAD_SLOTS:                   # at most 2 heavy decodes/taggings at once
        img = decode_upload(data)
        tagged = process_upload(img, remove_bg=CFG["app"]["remove_background"])
    if tagged.category not in APP_CATEGORIES:
        raise HTTPException(400, f"Unsupported category {tagged.category}")

    stem = uuid.uuid4().hex                      # never trust the client's filename
    orig, clean = f"original/{stem}.jpg", f"clean/{stem}.jpg"   # relative to UPLOADS
    img.convert("RGB").save(UPLOADS / orig, quality=90)
    tagged.clean_image.save(UPLOADS / clean, quality=90)

    row = ItemRow(path=clean, original_path=orig,
                  category=tagged.category, type=tagged.type, type_confidence=tagged.type_confidence,
                  color_name=tagged.color_name, color_hsv=json.dumps(tagged.color_hsv),
                  formality=tagged.formality, warmth=tagged.warmth,
                  embedding=tagged.embedding.astype("float32").tobytes())
    db.add(row)
    db.commit()
    return row.to_dict()


@app.get("/items")
def list_items(category: str | None = None, db=Depends(get_db)):
    q = db.query(ItemRow)
    if category:
        q = q.filter(ItemRow.category == category)
    return [r.to_dict() for r in q.order_by(ItemRow.created.desc()).all()]


@app.patch("/items/{item_id}")
def update_item(item_id: PathId, upd: ItemUpdate, db=Depends(get_db)):
    row = db.get(ItemRow, item_id)
    if not row:
        raise HTTPException(404, "Item not found")
    if upd.type:
        if upd.type not in TYPES:
            raise HTTPException(400, f"Unknown type. Choose from: {', '.join(TYPES)}")
        t = TYPES[upd.type]
        row.type, row.category, row.formality, row.warmth = upd.type, t.category, t.formality, t.warmth
    for field in ("category", "formality", "warmth"):
        if getattr(upd, field) is not None:
            setattr(row, field, getattr(upd, field))
    if upd.color:
        if upd.color not in NAMED_COLORS:
            raise HTTPException(400, f"Unknown color. Choose from: {', '.join(NAMED_COLORS)}")
        row.color_name = upd.color
        row.color_hsv = json.dumps(rgb_to_hsv01(NAMED_COLORS[upd.color]))   # keep features in sync
    db.commit()
    return row.to_dict()


@app.delete("/items/{item_id}")
def delete_item(item_id: PathId, db=Depends(get_db)):
    row = db.get(ItemRow, item_id)
    if not row:
        raise HTTPException(404, "Item not found")
    for p in (row.path, row.original_path):
        try:
            (UPLOADS / p).unlink(missing_ok=True)
        except OSError:
            pass
    db.delete(row)
    db.commit()
    return {"deleted": item_id}


@app.get("/options")
def options():
    proc = project_path(CFG["paths"]["processed"])
    return {"types": {t: {"category": v.category, "formality": v.formality, "warmth": v.warmth}
                      for t, v in TYPES.items()},
            "categories": list(APP_CATEGORIES), "colors": list(NAMED_COLORS),
            "occasions": list(OCCASIONS), "weather": list(WEATHER),
            "model": _model_name(),
            "study_available": _model_name() is not None and (proc / "embeddings.npy").exists()}


def _model_name() -> str | None:
    return None if STATE.get("scorer") is None else STATE["scorer"].name


# ---------------------------------------------------------------- recommend
@app.post("/recommend")
def recommend_ep(req: RecReq, db=Depends(get_db)):
    rows = db.query(ItemRow).all()
    items = [Item(r.id, r.category, r.type, r.color_name, r.hsv(), r.formality, r.warmth, r.emb()) for r in rows]
    recs = recommend(items, STATE.get("scorer"), req.occasion, req.weather, req.n)
    by_id = {r.id: r.to_dict() for r in rows}
    for rec in recs:
        rec["items"] = [by_id[i] for i in rec["item_ids"]]
    return {"model": None if STATE.get("scorer") is None else STATE["scorer"].name, "outfits": recs}


# ---------------------------------------------------------------- feedback & saved
@app.post("/feedback")
def feedback(req: FeedbackReq, db=Depends(get_db)):
    key = "-".join(str(i) for i in sorted(req.item_ids))
    db.add(FeedbackRow(outfit_key=key, vote=req.vote))
    db.commit()
    return {"ok": True}


@app.post("/saved")
def save_outfit(req: SaveReq, db=Depends(get_db)):
    row = SavedOutfitRow(item_ids=json.dumps(req.item_ids), score=req.score, explanation=req.explanation)
    db.add(row)
    db.commit()
    return {"id": row.id}


@app.get("/saved")
def list_saved(db=Depends(get_db)):
    items = {r.id: r.to_dict() for r in db.query(ItemRow).all()}
    out = []
    for s in db.query(SavedOutfitRow).order_by(SavedOutfitRow.ts.desc()).all():
        ids = json.loads(s.item_ids)
        out.append({"id": s.id, "score": s.score, "explanation": s.explanation,
                    "items": [items[i] for i in ids if i in items]})
    return out


@app.delete("/saved/{saved_id}")
def delete_saved(saved_id: PathId, db=Depends(get_db)):
    row = db.get(SavedOutfitRow, saved_id)
    if not row:
        raise HTTPException(404, "Not found")
    db.delete(row)
    db.commit()
    return {"deleted": saved_id}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
