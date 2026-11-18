#!/usr/bin/env bash
# Re-run SEAME analysis after full DisVoice extraction (106,842 utterances).
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST="manifests/seame_manifest_audio.csv"
FEATURES="features/seame_disvoice_utt.csv"
INTEGRATED="results/functional_anchors/seame/seame_integrated_features.csv"
META_MAN="results/functional_anchors/seame/seame_manifest_with_metadata.csv"

echo "=== SEAME full pipeline rerun ==="
echo "Manifest: $MANIFEST"
echo "Features: $FEATURES ($(wc -l < "$FEATURES") lines)"

python3 scripts/16_detect_discourse_markers_repetitions.py \
  --manifest "$MANIFEST" \
  --output results/functional_anchors/seame/seame_manifest_with_anchors.csv \
  --corpus seame

python3 scripts/17_extract_switch_directionality.py \
  --manifest results/functional_anchors/seame/seame_manifest_with_anchors.csv \
  --output results/functional_anchors/seame/seame_manifest_with_direction.csv \
  --corpus seame

python3 scripts/28_preprocess_speaker_interaction_metadata.py \
  --manifest-in results/functional_anchors/seame/seame_manifest_with_direction.csv \
  --manifest-out "$META_MAN" \
  --corpus seame

python3 scripts/19_integrate_features_for_analysis.py \
  --features "$FEATURES" \
  --manifest "$META_MAN" \
  --output "$INTEGRATED" \
  --corpus seame

python3 scripts/04_first_contrasts.py --corpus seame

python3 scripts/18_mixed_effects_analysis.py \
  --features "$FEATURES" \
  --manifest "$META_MAN" \
  --output-dir results/mixed_effects/seame_full \
  --corpus seame \
  --n-features 15

python3 scripts/25_run_directionality_models.py \
  --corpus seame \
  --integrated "$INTEGRATED"

python3 scripts/32_summarize_csw_quantity_frequency.py

python3 scripts/40_duration_controlled_mixed_effects.py --corpus seame --n-features 10

python3 scripts/20_generate_functional_anchors_report.py \
  --integrated "$INTEGRATED" \
  --output results/functional_anchors/seame/seame_functional_anchors_report.md \
  --corpus seame

python3 scripts/27_paper_figures_and_summary.py || true

python3 scripts/14_generate_seame_prosody_report.py || true

echo "=== Done ==="
