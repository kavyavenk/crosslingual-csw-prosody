"""Soothing blue / purple / turquoise palette for paper figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Core palette
DARK_BLUE = "#2F4A7A"
LIGHT_BLUE = "#A8CCE8"
DARK_PURPLE = "#5E4B8B"
LIGHT_PURPLE = "#CBB8E8"
TURQUOISE = "#4FAFA6"
LIGHT_TURQUOISE = "#9FD9D3"
MID_BLUE = "#4A6FA5"
MID_PURPLE = "#8A74B8"

CORPUS_COLORS = {
    "seame": DARK_PURPLE,
    "masac": DARK_BLUE,
}

COMPARISON_COLORS = [DARK_BLUE, MID_BLUE, DARK_PURPLE, MID_PURPLE, TURQUOISE, LIGHT_TURQUOISE]


def apply_paper_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FAFBFD",
            "axes.facecolor": "#FAFBFD",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "text.color": "#1E293B",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#E2E8F0",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def soothing_cmap(name: str = "significance") -> LinearSegmentedColormap:
    if name == "significance":
        colors = [LIGHT_BLUE, LIGHT_TURQUOISE, TURQUOISE, MID_PURPLE, DARK_PURPLE]
    else:
        colors = [LIGHT_BLUE, TURQUOISE, MID_BLUE, MID_PURPLE, DARK_PURPLE]
    return LinearSegmentedColormap.from_list("soothing", colors, N=256)


def corpus_color(corpus: str) -> str:
    return CORPUS_COLORS.get(corpus.lower(), MID_BLUE)
