"""Human style study (Step 7): does the model pick better bottoms than chance?

Each round: a random Polyvore TEST top + 20 random TEST bottoms. The model picks its best
bottom; a control bottom is drawn at random from the SAME 20 (never the model's pick). The two
are shown left/right in random order; the voter doesn't know which is which. Result = share of
votes for the model's pick, with a 95% Wilson interval. Uses held-out test items only.
"""
from __future__ import annotations

import math
import random
import secrets
import threading
import time
from functools import lru_cache
from typing import Annotated, Literal

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Path as PathParam
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from src.utils import project_path

N_CANDIDATES = 20
MIN_SECONDS_TO_VOTE = 1.0        # faster votes are treated as scripted and left out of the results
STALE_ROUND_SECONDS = 3600       # unvoted rounds older than this are purged
_LOCK = threading.Lock()         # one GPU scoring at a time
_LOAD_LOCK = threading.Lock()    # build the study data once


class StudyRound(Base):
    __tablename__ = "study_rounds"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token: Mapped[str] = mapped_column(String, unique=True, index=True)
    model: Mapped[str] = mapped_column(String)
    top_id: Mapped[str] = mapped_column(String)
    model_pick: Mapped[str] = mapped_column(String)
    control_pick: Mapped[str] = mapped_column(String)
    model_side: Mapped[str] = mapped_column(String)            # "left" | "right"
    choice: Mapped[str | None] = mapped_column(String, nullable=True)   # left | right | skip
    created: Mapped[float] = mapped_column(Float, default=time.time)
    voted: Mapped[float | None] = mapped_column(Float, nullable=True)


class VoteReq(BaseModel):
    token: str = Field(min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    choice: Literal["left", "right", "skip"]


@lru_cache(maxsize=1)
def _study_data(processed: str, models_dir: str, model_name: str):
    """Test-split tops/bottoms with embeddings, projected once with the app's model (lazy)."""
    import pandas as pd

    from src.inference import CompatScorer

    proc = project_path(processed)
    items = pd.read_csv(proc / "items.csv", dtype={"item_id": str})
    test = items[(items["splits"] == "test") & items["valid"]]          # test-only items
    ids = (proc / "embedding_ids.txt").read_text().split()
    row = {iid: i for i, iid in enumerate(ids)}
    emb = np.load(proc / "embeddings.npy", mmap_mode="r")
    tops = [i for i in test[test["category"] == "tops"]["item_id"] if i in row]
    bottoms = [i for i in test[test["category"] == "bottoms"]["item_id"] if i in row]
    scorer = CompatScorer(model_name, {"paths": {"models": models_dir}})
    paths = dict(zip(test["item_id"], test["path"]))
    colors = np.load(proc / "colors_hsv.npy") if scorer.use_color else None
    return scorer, emb, row, tops, bottoms, paths, colors


def build_router(get_db, cfg: dict, scorer_name) -> APIRouter:
    router = APIRouter(prefix="/study", tags=["study"])

    def data():
        name = scorer_name()
        proc = project_path(cfg["paths"]["processed"])
        if name is None or not (proc / "embeddings.npy").exists():
            raise HTTPException(503, "Style study needs the trained model and the Polyvore test data")
        with _LOAD_LOCK:
            return _study_data(cfg["paths"]["processed"], str(project_path(cfg["paths"]["models"])), name)

    @router.post("/next")          # POST: creates state, so it must not be triggerable by <img src>
    def next_round(db=Depends(get_db)):
        scorer, emb, row, tops, bottoms, _, colors = data()
        db.query(StudyRound).filter(StudyRound.choice.is_(None),
                                    StudyRound.created < time.time() - STALE_ROUND_SECONDS).delete()
        rng = random.Random(secrets.randbits(64))
        top = rng.choice(tops)
        cands = rng.sample(bottoms, N_CANDIDATES)
        E = np.stack([emb[row[top]]] + [emb[row[b]] for b in cands])
        with _LOCK:                                       # one GPU scoring at a time
            Z = scorer.project(E, ["tops"] + ["bottoms"] * N_CANDIDATES)
            Ca = Cb = None
            if colors is not None:                         # colour-feature models (e.g. B_color)
                C = torch.as_tensor(np.stack([colors[row[x]] for x in [top] + cands]), device=scorer.device)
                Ca, Cb = C[:1].expand(N_CANDIDATES, -1), C[1:]
            raw = scorer.raw_scores(Z[:1].expand(N_CANDIDATES, -1), Z[1:], Ca, Cb)
        best = int(raw.argmax())
        control = rng.choice([i for i in range(N_CANDIDATES) if i != best])
        side = rng.choice(["left", "right"])
        rnd = StudyRound(token=secrets.token_urlsafe(24), model=scorer.name, top_id=top,
                         model_pick=cands[best], control_pick=cands[control], model_side=side)
        db.add(rnd)
        db.commit()
        left, right = (rnd.model_pick, rnd.control_pick) if side == "left" else (rnd.control_pick, rnd.model_pick)
        return {"token": rnd.token, "top": f"/study/image/{top}",
                "left": f"/study/image/{left}", "right": f"/study/image/{right}"}

    @router.post("/vote")
    def vote(req: VoteReq, db=Depends(get_db)):
        # single conditional UPDATE: two concurrent votes on one round can't both succeed
        updated = (db.query(StudyRound)
                   .filter(StudyRound.token == req.token, StudyRound.choice.is_(None))
                   .update({"choice": req.choice, "voted": time.time()}, synchronize_session=False))
        db.commit()
        if not updated:
            exists = db.query(StudyRound.id).filter(StudyRound.token == req.token).first()
            raise HTTPException(409 if exists else 404,
                                "Already voted on this round" if exists else "Unknown round")
        return {"ok": True}

    @router.get("/results")
    def results(db=Depends(get_db)):
        decided = db.query(StudyRound).filter(StudyRound.choice.in_(["left", "right"])).all()
        votes = [v for v in decided if (v.voted or 0) - v.created >= MIN_SECONDS_TO_VOTE]
        n = len(votes)
        k = sum(v.choice == v.model_side for v in votes)
        skipped = db.query(StudyRound).filter(StudyRound.choice == "skip").count()
        out = {"votes": n, "model_preferred": k, "skipped": skipped, "too_fast": len(decided) - n,
               "rate": round(k / n, 4) if n else None, "ci95": None}
        if n:
            z, p = 1.96, k / n                       # Wilson score interval
            den = 1 + z * z / n
            mid = (p + z * z / (2 * n)) / den
            half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            out["ci95"] = [round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)]
        return out

    @router.get("/image/{item_id}")
    def image(item_id: Annotated[str, PathParam(pattern=r"^[0-9]{1,20}$")]):
        paths = data()[5]
        path = paths.get(item_id)                  # only held-out test items are served
        if path is None:
            raise HTTPException(404, "Not a study image")
        return FileResponse(path, media_type="image/jpeg")

    return router
