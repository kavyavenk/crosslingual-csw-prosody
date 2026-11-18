#!/usr/bin/env python3
"""Category significance bar chart + SEAME p-value heatmap (paper palette)."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figure_theme import COMPARISON_COLORS, apply_paper_theme, soothing_cmap


def disvoice_cat(f: str) -> str:
    if f[0].isdigit() or "f0" in f.lower() or "tilt" in f.lower():
        return "f0"
    if any(x in f.lower() for x in ["dur", "pause", "rate", "silence"]):
        return "duration"
    if "E" in f or "energy" in f.lower():
        return "energy"
    return "other"


def category_bar_chart(out: pathlib.Path) -> None:
    apply_paper_theme()
    rows = []
    specs = [
        ("seame", "CS_vs_EN", "SEAME CS–EN"),
        ("seame", "CS_vs_ZH", "SEAME CS–ZH"),
        ("masac", "CS_vs_EN", "MaSaC CS–EN"),
        ("masac", "CS_vs_HI", "MaSaC CS–HI"),
    ]
    for corp, pair, label in specs:
        df = pd.read_csv(f"features/first_contrast_{corp}_{pair}.csv")
        df["cat"] = df.feature.map(disvoice_cat)
        for c, g in df.groupby("cat"):
            rows.append(
                {
                    "comparison": label,
                    "category": c,
                    "pct_sig": 100 * g.sig.mean(),
                }
            )
    catdf = pd.DataFrame(rows)
    cats = ["duration", "energy", "f0", "other"]
    x = np.arange(len(cats))
    w = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, comp in enumerate(catdf.comparison.unique()):
        sub = catdf[catdf.comparison == comp].set_index("category").reindex(cats)
        ax.bar(x + i * w, sub.pct_sig, width=w, label=comp, color=COMPARISON_COLORS[i % len(COMPARISON_COLORS)])
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(cats, fontsize=16)
    ax.set_ylabel("% features BH-FDR significant", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(fontsize=13, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def pvalue_heatmap(out: pathlib.Path) -> None:
    apply_paper_theme()
    df = pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv").sort_values("feature")
    pvals = -np.log10(df.p.clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(4.5, 10))
    cmap = soothing_cmap("significance")
    im = ax.imshow(pvals.values.reshape(-1, 1), aspect="auto", cmap=cmap, vmin=0, vmax=12)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.feature, fontsize=4)
    ax.set_xticks([0])
    ax.set_xticklabels(["CS vs EN"])
    ax.set_title("SEAME: $-\\log_{10}(p)$ per feature")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("$-\\log_{10}(p)$")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/paper/figures")
    args = p.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    category_bar_chart(out_dir / "fig_category_pct_significant.png")
    pvalue_heatmap(out_dir / "fig_seame_cs_en_pvalue_heatmap.png")
    print(f"Wrote themed figures under {out_dir}")


if __name__ == "__main__":
    main()
