#!/usr/bin/env bash
# Re-run MaSaC analysis after applying the FINAL annotation sheet
# (scripts/46_apply_masac_final_sheet.py). Mirrors the SEAME rerun script.
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST="manifests/masac_manifest.csv"
FEATURES="features/masac_disvoice_utt.csv"
OUT="results/functional_anchors/masac"
INTEGRATED="$OUT/masac_integrated_features.csv"
META_MAN="$OUT/masac_manifest_with_metadata.csv"
WORDS="data/masac_raw/masac_data_compiled.csv"

echo "=== MaSaC full pipeline rerun (final sheet) ==="
echo "Manifest: $MANIFEST"
echo "Features: $FEATURES ($(wc -l < "$FEATURES") lines)"

python3 scripts/16_detect_discourse_markers_repetitions.py \
  --manifest "$MANIFEST" \
  --output "$OUT/masac_manifest_with_anchors.csv" \
  --corpus masac

python3 scripts/17_extract_switch_directionality.py \
  --manifest "$OUT/masac_manifest_with_anchors.csv" \
  --output "$OUT/masac_manifest_with_direction.csv" \
  --corpus masac

python3 scripts/28_preprocess_speaker_interaction_metadata.py \
  --manifest-in "$OUT/masac_manifest_with_direction.csv" \
  --manifest-out "$META_MAN" \
  --corpus masac

python3 scripts/19_integrate_features_for_analysis.py \
  --features "$FEATURES" \
  --manifest "$META_MAN" \
  --output "$INTEGRATED" \
  --corpus masac

python3 scripts/04_first_contrasts.py --corpus masac

python3 scripts/18_mixed_effects_analysis.py \
  --features "$FEATURES" \
  --manifest "$META_MAN" \
  --output-dir results/mixed_effects/masac_full \
  --corpus masac \
  --n-features 15

python3 scripts/25_run_directionality_models.py \
  --corpus masac \
  --integrated "$INTEGRATED" || true

python3 scripts/32_summarize_csw_quantity_frequency.py || true

python3 scripts/40_duration_controlled_mixed_effects.py --corpus masac --n-features 10 || true

python3 scripts/20_generate_functional_anchors_report.py \
  --integrated "$INTEGRATED" \
  --output "$OUT/masac_functional_anchors_report.md" \
  --corpus masac

echo "=== Done ==="
