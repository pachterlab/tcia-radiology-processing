#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
NOTEBOOK="$BASE_DIR/tcia_ct_processing_protocol.ipynb"
DATA_DIR="$BASE_DIR/data/radiogenomics/tcga-kirc/imaging"

run_case () {
  do_masking=$1
  orient=$2
  clip=$3
  resample=$4
  normalize=$5

  echo "Running: masking=$do_masking orient=$orient clip=$clip resample=$resample normalize=$normalize"

  # Build suffix
  suffix=""
  [[ "$orient" == "True" ]] && suffix="${suffix}_oriented"
  [[ "$clip" == "True" ]] && suffix="${suffix}_clipped"
  [[ "$resample" == "True" ]] && suffix="${suffix}_resampled"
  [[ "$do_masking" == "True" ]] && suffix="${suffix}_maskedapplied"
  [[ "$normalize" == "True" ]] && suffix="${suffix}_normalized"

  # Handle "all False" case
  [[ -z "$suffix" ]] && suffix="_raw"

  OUTPUT_NOTEBOOK="$BASE_DIR/tcia_ct_processing_protocol_output${suffix}.ipynb"

  echo "Running: $suffix"

  papermill "$NOTEBOOK" "$OUTPUT_NOTEBOOK" \
    -p data_dir "$DATA_DIR" \
    -p do_radiomics False \
    -p image_dimensionality 3D \
    -p do_masking $do_masking \
    -p orient $orient \
    -p clip $clip \
    -p resample $resample \
    -p normalize $normalize
  
  # Rename .npy files
  find "$DATA_DIR" -type f -name "imaging_final_3D_masked.npy" \
    -execdir mv {} "imaging${suffix}.npy" \;

  find "$DATA_DIR" -type f -name "imaging_final_3D.npy" \
    -execdir mv {} "imaging${suffix}.npy" \;

  find "$DATA_DIR" -type f -name "segmentation_final_3D_masked.npy" \
    -execdir mv {} "segmentation${suffix}.npy" \;

  find "$DATA_DIR" -type f -name "segmentation_final_3D.npy" \
    -execdir mv {} "segmentation${suffix}.npy" \; RRPP_2026

  # change filenames and column names in CSV
  sed -i \
    -e "s/imaging_final_3D_masked\b/imaging${suffix}/g" \
    -e "s/segmentation_final_3D_masked\b/segmentation${suffix}/g" \
    -e "s/imaging_final_3D\b/imaging${suffix}/g" \
    -e "s/segmentation_final_3D\b/segmentation${suffix}/g" \
    "$DATA_DIR/metadata_usc.csv"

  # Delete intermediate nii.gz
  find "$DATA_DIR" -type f -name "imaging_final_3D_masked.nii.gz" -delete
  find "$DATA_DIR" -type f -name "segmentation_final_3D_masked.nii.gz" -delete

  # Cleanup other files
  find "$DATA_DIR" -type f \( -name "*.nii" -o -name "*.nii.gz" -o -name "*.npy" \) \
    ! -name "0502_VENOUS.nii" \
    ! -name "ROI_602_Tumor_a.nii" \
    ! -name "*_final*" \
    -delete
}

# -------------------------
# CASES
# -------------------------

# 1. All False
run_case False False False False False

# 2. One-at-a-time True
run_case True  False False False False
run_case False True  False False False
run_case False False True  False False
run_case False False False True  False
run_case False False False False True

# 3. All True
run_case True True True True True