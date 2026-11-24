import torch
from PIL import Image

from src.color import color_name, dominant_color, pair_color_features, rgb_to_hsv01


def _square(rgb, bg=(0, 0, 0, 0)):
    im = Image.new("RGBA", (100, 100), bg)
    im.paste(rgb + (255,), (20, 20, 80, 80))
    return im


def test_dominant_color_ignores_transparent_and_white_background():
    assert color_name(dominant_color(_square((190, 30, 40)))) == "red"
    assert color_name(dominant_color(_square((25, 35, 80), bg=(255, 255, 255, 255)))) == "navy"


def test_pair_color_features():
    red, cyan, black = (rgb_to_hsv01(c) for c in [(255, 0, 0), (0, 255, 255), (10, 10, 10)])
    f = pair_color_features(torch.tensor([red, red]), torch.tensor([cyan, black]))
    assert f.shape == (2, 6)
    assert abs(f[0, 0].item() - 1.0) < 1e-5          # complementary hues
    assert f[0, 5].item() == 0.0 and f[1, 5].item() == 1.0   # black counts as neutral


def test_color_names_are_perceptual():
    cases = {(185, 215, 175): "light green", (25, 35, 80): "navy", (130, 25, 45): "burgundy",
             (250, 250, 248): "white", (20, 20, 22): "black", (128, 130, 128): "grey",
             (235, 150, 180): "pink", (185, 145, 100): "tan", (30, 90, 210): "blue"}
    for rgb, name in cases.items():
        assert color_name(rgb) == name, (rgb, color_name(rgb))
