#!/usr/bin/env python3
"""Summarize CSW quantity, frequency, and strategy metrics for RQ2 replication."""

import argparse
import json
import pathlib

import pandas as pd


def summarize_corpus(integrated_path: pathlib.Path, corpus: str) -> dict:
    df = pd.read_csv(integrated_path, low_memory=False)
    cs = df[df["lang"] == "CS"].copy()
    with_dir = cs[cs["switch_direction_class"].notna()]

    summary = {
        "corpus": corpus,
        "n_cs_total": int(len(cs)),
        "n_cs_with_direction": int(len(with_dir)),
        "direction_l1_to_l2_pct": round(
            100 * (with_dir["switch_direction_class"] == "L1→L2").mean(), 1
        )
        if len(with_dir)
        else None,
        "n_switches_mean": round(float(cs["n_switches"].mean()), 2),
        "n_switches_median": float(cs["n_switches"].median()),
        "m_index_mean": round(float(cs["m_index"].mean()), 3)
        if "m_index" in cs.columns and cs["m_index"].notna().any()
        else None,
        "m_index_median": round(float(cs["m_index"].median()), 3)
        if "m_index" in cs.columns and cs["m_index"].notna().any()
        else None,
        "i_index_mean": round(float(cs["i_index"].mean()), 3)
        if "i_index" in cs.columns and cs["i_index"].notna().any()
        else None,
        "i_index_median": round(float(cs["i_index"].median()), 3)
        if "i_index" in cs.columns and cs["i_index"].notna().any()
        else None,
    }

    if "cs_strategy" in cs.columns:
        strat = cs["cs_strategy"].value_counts()
        summary["strategy_distribution"] = {
            str(k): int(v) for k, v in strat.items()
        }
        labeled = cs[cs["cs_strategy"].isin(["insertional", "alternational"])]
        summary["n_cs_with_strategy_label"] = int(len(labeled))
        summary["strategy_labeled_pct"] = round(100 * len(labeled) / len(cs), 1)
        if len(labeled):
            summary["insertional_pct_of_labeled"] = round(
                100 * (labeled["cs_strategy"] == "insertional").mean(), 1
            )

    if "switch_type" in cs.columns:
        st = cs["switch_type"].value_counts(dropna=False)
        summary["heuristic_switch_type"] = {str(k): int(v) for k, v in st.items()}

    return summary


def model_fit_summary(model_dir: pathlib.Path, corpus: str) -> dict:
    out = {}
    for name, file in [
        ("direction", f"{corpus}_cs_directionality_models_ols.csv"),
        ("quantity_n_switches", f"{corpus}_cs_quantity_models_ols.csv"),
        ("m_i_index", f"{corpus}_cs_mi_index_models_ols.csv"),
        ("strategy", f"{corpus}_cs_strategy_models_ols.csv"),
    ]:
        path = model_dir / file
        if not path.exists():
            out[name] = {"available": False}
            continue
        mdf = pd.read_csv(path)
        ok = mdf[mdf["success"] == True]  # noqa: E712
        out[name] = {
            "available": True,
            "n_features_fitted": int(len(ok)),
            "example_n": int(ok["n"].iloc[0]) if len(ok) and "n" in ok.columns else None,
        }
    return out


def write_markdown(summaries: list, model_summaries: dict, output: pathlib.Path) -> None:
    lines = [
        "# CSW Quantity / Frequency / Strategy — Replication Status",
        "",
        "**Short answer:** We now have direction, `n_switches`, M-index, I-index, and strategy "
        "labels (where annotated), with OLS models for each predictor family.",
        "",
    ]

    for s in summaries:
        corp = s["corpus"].upper()
        lines.extend(
            [
                f"## {corp}",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| CS utterances (analyzed subset) | {s['n_cs_total']:,} |",
                f"| CS with direction (`is_l1_to_l2`) | {s['n_cs_with_direction']:,} |",
                f"| L1→L2 share (of directed CS) | {s.get('direction_l1_to_l2_pct', '—')}% |",
                f"| Mean `n_switches` per CS utterance | {s['n_switches_mean']} |",
                f"| Median `n_switches` | {s['n_switches_median']} |",
                f"| Mean M-index | {s.get('m_index_mean', '—')} |",
                f"| Mean I-index | {s.get('i_index_mean', '—')} |",
            ]
        )
        if "n_cs_with_strategy_label" in s:
            lines.append(
                f"| CS with insertional/alternational label | "
                f"{s['n_cs_with_strategy_label']:,} ({s['strategy_labeled_pct']}%) |"
            )
            if "insertional_pct_of_labeled" in s:
                lines.append(
                    f"| Insertional share (of labeled) | {s['insertional_pct_of_labeled']}% |"
                )
        lines.append("")
        if "strategy_distribution" in s:
            lines.append("**Strategy distribution (`cs_strategy`):**")
            for k, v in s["strategy_distribution"].items():
                lines.append(f"- {k}: {v:,}")
            lines.append("")

    lines.extend(["## Model outputs", ""])
    for corpus, ms in model_summaries.items():
        lines.append(f"### {corpus.upper()}")
        for family, info in ms.items():
            if info.get("available"):
                lines.append(
                    f"- `{family}`: {info['n_features_fitted']} features fitted "
                    f"(n≈{info.get('example_n')})"
                )
            else:
                lines.append(f"- `{family}`: not available")
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Summarize CSW quantity/frequency/strategy metrics")
    p.add_argument(
        "--output",
        type=str,
        default="results/csw_quantity_frequency_strategy_summary.md",
    )
    args = p.parse_args()

    corpora = {
        "seame": pathlib.Path("results/functional_anchors/seame/seame_integrated_features.csv"),
        "masac": pathlib.Path("results/functional_anchors/masac/masac_integrated_features.csv"),
    }
    summaries = [summarize_corpus(path, name) for name, path in corpora.items() if path.exists()]
    model_summaries = {
        name: model_fit_summary(pathlib.Path("results/directionality_models") / name, name)
        for name in corpora
    }

    out = pathlib.Path(args.output)
    write_markdown(summaries, model_summaries, out)
    print(json.dumps({"summaries": summaries, "models": model_summaries}, indent=2))
    print(f"\nWrote: {out}")
