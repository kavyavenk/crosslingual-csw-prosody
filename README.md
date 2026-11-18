# Cross-Lingual Prosodic Profiles of Conversational Code-switched Speech

Analysis code for Bhattacharya*, Venkatesh*, and Hirschberg, *Cross-Lingual Prosodic Profiles of Conversational Code-switched Speech* (Findings of EACL 2027).

The code compares utterance-level prosody (103 DisVoice F0, energy, and duration features) between code-switched and monolingual speech in Mandarin-English (SEAME) and Hindi-English (MaSaC), with a Spanish-English (Bangor Miami) diagnostic classifier.

## Data

No corpus data is included in this repository. Obtain each corpus from its original source:

- **SEAME**: [LDC2015S04](https://catalog.ldc.upenn.edu/LDC2015S04) (LDC license required)
- **MaSaC**: <https://github.com/LCS2-IIITD/MSH-COMICS>
- **Bangor Miami**: <https://biling.talkbank.org/access/Bangor/Miami.html>

Scripts expect audio and transcripts under `data/`, and write features to `features/` and outputs to `results/`.

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/jcvasquezc/DisVoice
```

## Pipeline

Scripts are numbered roughly in run order. Main groups, by paper section:

| Scripts | Purpose |
|---|---|
| `00`-`03`, `fix_disvoice_compatibility.py` | Audio normalization, manifests, utterance slicing, DisVoice feature extraction |
| `16`, `17`, `19`, `21` | Discourse marker and repetition labels, switch direction, feature integration |
| `04`, `12`-`15`, `26` | CSW vs. monolingual Welch's *t*-tests with BH-FDR (Section 5.1) |
| `18`, `23`, `50`, `55`, `59`, `60` | Primary mixed-effects models and full ranked table (Table 3, Appendix A.5) |
| `44`, `51`-`54`, `62`, `63` | Bangor Miami diagnostic classifier and category ablations |
| `32_summarize_csw_quantity_frequency.py`, `48` | CSW property subgroup comparisons (Section 5.2) |
| `24`, `25`, `57` | Switch direction x discourse marker analyses |
| `28`-`39`, `41_run_stratified_interlocutor_gender_masac.py`, `49`, `56`, `61`, `64` | Speaker and interlocutor identity analyses (Section 5.3) |
| `65`-`67` | Validation of discourse marker and repetition labels |
| `68`-`72` | Interaction models, speaker counts, separate baselines, divergence model |
| `27`, `42`, `43`, `58` | Figures and appendix tables |

All statistical models use `statsmodels`; the classifier uses `scikit-learn` (versions in `requirements.txt`).
