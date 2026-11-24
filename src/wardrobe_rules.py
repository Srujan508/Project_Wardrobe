"""Clothing types the app recognises, with coarse category, formality (1-5) and warmth (1-5).

The zero-shot tagger picks one of these types; the recommender's occasion and
weather filters read formality and warmth from here. Edit freely: it is a lookup table.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClothingType:
    category: str     # tops | bottoms | shoes | outerwear
    formality: int    # 1 = very casual ... 5 = formal
    warmth: int       # 1 = very light ... 5 = very warm


TYPES: dict[str, ClothingType] = {
    "t-shirt":        ClothingType("tops", 1, 1),
    "tank top":       ClothingType("tops", 1, 1),
    "polo shirt":     ClothingType("tops", 2, 2),
    "blouse":         ClothingType("tops", 3, 2),
    "button-up shirt": ClothingType("tops", 4, 2),
    "sweater":        ClothingType("tops", 2, 4),
    "hoodie":         ClothingType("tops", 1, 3),
    "jeans":          ClothingType("bottoms", 2, 3),
    "trousers":       ClothingType("bottoms", 4, 3),
    "shorts":         ClothingType("bottoms", 1, 1),
    "skirt":          ClothingType("bottoms", 3, 2),
    "joggers":        ClothingType("bottoms", 1, 2),
    "sneakers":       ClothingType("shoes", 1, 2),
    "boots":          ClothingType("shoes", 3, 4),
    "heels":          ClothingType("shoes", 4, 2),
    "loafers":        ClothingType("shoes", 4, 2),
    "sandals":        ClothingType("shoes", 1, 1),
    "denim jacket":   ClothingType("outerwear", 2, 3),
    "coat":           ClothingType("outerwear", 3, 5),
    "blazer":         ClothingType("outerwear", 5, 3),
    "cardigan":       ClothingType("outerwear", 2, 3),
    "puffer jacket":  ClothingType("outerwear", 1, 5),
    "dress":          ClothingType("all-body", 3, 2),
    "sundress":       ClothingType("all-body", 1, 1),
    "evening gown":   ClothingType("all-body", 5, 2),
    "jumpsuit":       ClothingType("all-body", 2, 2),
}

# "all-body" = one-piece items (dresses, jumpsuits): an outfit is EITHER top + bottom OR one all-body piece.
APP_CATEGORIES = ("tops", "bottoms", "all-body", "shoes", "outerwear")

OCCASIONS = {            # allowed formality range per item
    "any": (1, 5),
    "casual": (1, 2),
    "smart casual": (2, 4),
    "formal": (3, 5),
}

WEATHER = {              # allowed item warmth range, and whether outerwear is allowed
    "any": ((1, 5), True),
    "hot": ((1, 3), False),
    "mild": ((1, 4), True),
    "cold": ((2, 5), True),
}
