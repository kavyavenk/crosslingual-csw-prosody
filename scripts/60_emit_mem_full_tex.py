"""Emit app:mem-full summary + full LaTeX table from mem_full_*.csv.

Category membership is recomputed here from the shared ``disvoice_category``
rule (the corrected DisVoice convention in which energy-contour tilt/MSE
features such as ``avgtiltEunvoiced`` are Energy, not F0), so this table matches
the categorization used everywhere else in the paper (Table 2, the ablation
table, and the per-category social table). We do NOT trust the stored
``category`` column, which predates that fix.
"""
import pandas as pd
import numpy as np

try:  # works whether run as `python3 scripts/60_...py` or imported as a module
    from paper_feature_selection import disvoice_category
except ImportError:
    from scripts.paper_feature_selection import disvoice_category

CATN = {"f0": "F0", "duration": "Duration", "energy": "Energy", "other": "Other"}


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def _load(corpus):
    d = pd.read_csv(f"results/paper/mem_full_{corpus}.csv")
    # Recompute category with the corrected rule (energy tilt/MSE -> energy).
    d["category"] = d["feature"].map(disvoice_category)
    return d


def summary(corpus):
    d = _load(corpus)
    print(f"\n== {corpus.upper()} ({len(d)} feats) ==")
    for c in ("f0", "duration", "energy"):
        s = d[d["category"] == c]
        se = (s["p_en"] < 0.05).sum()
        sl = (s["p_l1"] < 0.05).sum()
        print(f"  {c:8s} n={len(s):2d} | mono-EN sig {se}/{len(s)} | mono-L1 sig {sl}/{len(s)}")
    print(f"  ALL      n={len(d):2d} | mono-EN sig {(d['p_en']<0.05).sum()}/{len(d)} | "
          f"mono-L1 sig {(d['p_l1']<0.05).sum()}/{len(d)}")


def emit_tex(l1lab):
    rows = []
    # Order categories F0, Duration, Energy, Other for readability; within each,
    # rank by between-condition separation variance (descending).
    cat_order = {"f0": 0, "duration": 1, "energy": 2, "other": 3}
    for corpus, l1 in [("seame", "ZH"), ("masac", "HI")]:
        d = _load(corpus)
        d["_catord"] = d["category"].map(cat_order)
        d = d.sort_values(["_catord", "sep_var"], ascending=[True, False])
        for _, r in d.iterrows():
            rows.append((r["feature"], CATN[r["category"]],
                         "SEAME" if corpus == "seame" else "MaSaC",
                         f"${r['beta_en']:.1f}^{{{stars(r['p_en'])}}}$" if r['beta_en']==r['beta_en'] else "--",
                         f"${r['beta_l1']:.1f}^{{{stars(r['p_l1'])}}}$" if r['beta_l1']==r['beta_l1'] else "--"))
    caption = (
        "Full ranked monolingual-vs-CSW mixed-effects coefficients for all 103 "
        "DisVoice features (CSW reference; by-speaker random intercept; Eq.~1), "
        "ranked within prosodic category by "
        "between-condition separation variance. "
        "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (BH-FDR). Category "
        "membership follows the DisVoice convention used throughout, in which "
        "energy-contour tilt/MSE features (e.g., \\texttt{avgtiltEunvoiced}) are "
        "grouped with energy rather than F0."
    )
    header = ("\\textbf{Outcome} & \\textbf{Cat.} & \\textbf{Corpus} & "
              "\\textbf{Mono EN} & \\textbf{Mono ZH/HI} \\\\")
    out = "results/paper/mem_full_table.tex"
    with open(out, "w") as f:
        f.write("\\begin{longtable}{lllrr}\n")
        f.write(f"\\caption{{{caption}}}\\label{{tab:long}}\\\\\n")
        f.write("\\toprule\n" + header + "\n\\midrule\n\\endfirsthead\n")
        f.write("\\multicolumn{5}{c}{\\tablename~\\thetable{} -- continued} \\\\\n")
        f.write("\\toprule\n" + header + "\n\\midrule\n\\endhead\n")
        f.write("\\midrule\n\\endfoot\n")
        f.write("\\bottomrule\n\\endlastfoot\n")
        for feat, cat, cor, be, bl in rows:
            f.write(f"\\texttt{{{feat}}} & {cat} & {cor} & {be} & {bl} \\\\\n")
        f.write("\\end{longtable}\n")
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    summary("seame")
    summary("masac")
    emit_tex("ZH/HI")
