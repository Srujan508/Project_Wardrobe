"""SQLite tables: items, feedback, saved_outfits."""
from __future__ import annotations

import json
import time

import numpy as np
from sqlalchemy import Float, Integer, LargeBinary, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.utils import load_config, project_path


class Base(DeclarativeBase):
    pass


class ItemRow(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, default="me", index=True)
    path: Mapped[str] = mapped_column(String)               # cleaned image, relative to uploads dir
    original_path: Mapped[str] = mapped_column(String)      # relative to uploads dir
    category: Mapped[str] = mapped_column(String)
    type: Mapped[str] = mapped_column(String)
    type_confidence: Mapped[float] = mapped_column(Float)
    color_name: Mapped[str] = mapped_column(String)
    color_hsv: Mapped[str] = mapped_column(String)          # JSON [h, s, v]
    formality: Mapped[int] = mapped_column(Integer)
    warmth: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)   # float32 bytes
    created: Mapped[float] = mapped_column(Float, default=time.time)

    def emb(self) -> np.ndarray:
        return np.frombuffer(self.embedding, dtype=np.float32)

    def hsv(self) -> tuple[float, float, float]:
        return tuple(json.loads(self.color_hsv))  # type: ignore[return-value]

    def to_dict(self) -> dict:
        return {"id": self.id, "image": "/uploads/" + self.path, "category": self.category,
                "type": self.type, "type_confidence": round(self.type_confidence, 3),
                "color": self.color_name, "formality": self.formality, "warmth": self.warmth}


class FeedbackRow(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    outfit_key: Mapped[str] = mapped_column(String, index=True)   # sorted item ids, "1-4-7"
    vote: Mapped[int] = mapped_column(Integer)                    # +1 / -1
    ts: Mapped[float] = mapped_column(Float, default=time.time)


class SavedOutfitRow(Base):
    __tablename__ = "saved_outfits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, default="me")
    item_ids: Mapped[str] = mapped_column(Text)                   # JSON list
    score: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str] = mapped_column(Text)
    ts: Mapped[float] = mapped_column(Float, default=time.time)


def make_session_factory(db_path: str | None = None):
    cfg = load_config()
    path = project_path(db_path or cfg["app"]["db"])
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
