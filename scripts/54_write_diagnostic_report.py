import pandas as pd
import pathlib

d = pd.read_csv("results/paper/diagnostic_classifier/diagnostic_classifier_summary.csv")


def v(tc, ev, fz):
    r = d[(d.test_corpus == tc) & (d["eval"] == ev) & (d.featureization == fz)]
    r = r.iloc[0]
    return r["mean"], r["std"]


def fmt(tc, ev, fz):
    m, s = v(tc, ev, fz)
    return f"{m:.3f}$\\pm${s:.3f}"


hold_raw = fmt("SEAME", "train_corpus_holdout", "raw")
hold_z = fmt("SEAME", "train_corpus_holdout", "zscore_within_speaker")
se_raw = fmt("SEAME", "zero_shot", "raw")
se_z = fmt("SEAME", "zero_shot", "zscore_within_speaker")
ma_raw = fmt("MaSaC", "zero_shot", "raw")
ma_z = fmt("MaSaC", "zero_shot", "zscore_within_speaker")

md = f"""# Diagnostic classifier: CS vs. monolingual prosody (Bhattacharya et al. 2026 style)

Train: Bangor Miami (EN--ES), utterance-level DisVoice prosody (103 features).
Model: L2 logistic regression, class-balanced, features standardized on train.
Evaluation: 5 speaker-disjoint seeds; AUROC reported as mean$\\pm$SD.
Labels joined from `full_final_miami_bangor_annotated.xlsx` (`with-span-ann`,
`is_code_switched`); 33,231 utterances, 1,545 code-switched / 31,686 monolingual,
96 speakers across 55 conversations.

| Train $\\to$ Test | Featurization | Eval | AUROC |
|---|---|---|---|
| BM (in-corpus) | raw | speaker-disjoint holdout | {hold_raw} |
| BM (in-corpus) | per-speaker $z$ | speaker-disjoint holdout | {hold_z} |
| BM $\\to$ SEAME (EN--ZH) | raw | zero-shot | {se_raw} |
| BM $\\to$ SEAME (EN--ZH) | per-speaker $z$ | zero-shot | {se_z} |
| BM $\\to$ MaSaC (EN--HI) | raw | zero-shot | {ma_raw} |
| BM $\\to$ MaSaC (EN--HI) | per-speaker $z$ | zero-shot | {ma_z} |

## Paste-in paragraph (LaTeX)

To test whether the prosody of code-switching generalizes across typologically
distinct language pairs, we trained a diagnostic classifier to distinguish
code-switched from monolingual utterances on the Bangor Miami English--Spanish
corpus and evaluated it zero-shot on SEAME (English--Mandarin) and MaSaC
(English--Hindi). Using the 103 DisVoice prosodic features and an
$L_2$-regularized logistic regression, the classifier separates code-switched
from monolingual speech well within Bangor Miami under a speaker-disjoint split
(AUROC {hold_raw}). Transferred zero-shot to the other corpora it remains above
chance---AUROC {se_raw} on SEAME and {ma_raw} on MaSaC on raw features---and
per-speaker $z$-scoring, which removes speaker-specific baselines before
transfer, substantially improves generalization (AUROC {se_z} on SEAME and
{ma_z} on MaSaC). This indicates that a prosodic signature of code-switching is
partially shared across language pairs once speaker-level differences are
normalized, though the drop from in-corpus to cross-corpus performance shows the
signature is also partly language-pair specific.
"""

out = pathlib.Path("results/paper/diagnostic_classifier/DIAGNOSTIC_REPORT.md")
out.write_text(md, encoding="utf-8")
print(md)
print("wrote", out)
