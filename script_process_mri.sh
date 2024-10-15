#!/bin/bash

# x [-35 35] span 70
# y [-55 35] span 90
# z [-25 35] span 60
#max thickness ~5
export PATH="$2:$PATH"
ROOT="$1"
RIBBON="$ROOT/ribbonSpace.nii.gz"
# reference volume space
wb_command -volume-create 281 361 241 "$RIBBON" -plumb XYZ .25 .25 .25 -35 -55 -25
# unzipped version for some matlab nifti implementations that don't do gzipped
gunzip -c "$RIBBON" > "$ROOT/ribbonSpace.nii"

# L_PIAL="$ROOT"/fsaverage_LR164k/*.L.pial.164k_fs_LR.surf.gii
# R_PIAL="$ROOT"/fsaverage_LR164k/*.R.pial.164k_fs_LR.surf.gii

# L_WHITE="$ROOT"/fsaverage_LR164k/*.L.white.164k_fs_LR.surf.gii
# R_WHITE="$ROOT"/fsaverage_LR164k/*.R.white.164k_fs_LR.surf.gii

L_PIAL=$(find "$ROOT/fsaverage_LR164k" -name "*.L.pial.164k_fs_LR.surf.gii" -print -quit)
R_PIAL=$(find "$ROOT/fsaverage_LR164k" -name "*.R.pial.164k_fs_LR.surf.gii" -print -quit)
L_WHITE=$(find "$ROOT/fsaverage_LR164k" -name "*.L.white.164k_fs_LR.surf.gii" -print -quit)
R_WHITE=$(find "$ROOT/fsaverage_LR164k" -name "*.R.white.164k_fs_LR.surf.gii" -print -quit)


# pial surface signed distance
wb_command -create-signed-distance-volume "$L_PIAL" "$RIBBON" "$ROOT/pialSD_L.nii.gz" -exact-limit 1 -approx-limit 7
wb_command -create-signed-distance-volume "$R_PIAL" "$RIBBON" "$ROOT/pialSD_R.nii.gz" -exact-limit 1 -approx-limit 7

# white surface signed distance
wb_command -create-signed-distance-volume "$L_WHITE" "$RIBBON" "$ROOT/whiteSD_L.nii.gz" -exact-limit 1 -approx-limit 7
wb_command -create-signed-distance-volume "$R_WHITE" "$RIBBON" "$ROOT/whiteSD_R.nii.gz" -exact-limit 1 -approx-limit 7

# make ribbon volumes
# WB_VARS_L="-var pial \"$ROOT/pialSD_L.nii.gz\" -var white \"$ROOT/whiteSD_L.nii.gz\""
# WB_VARS_R="-var pial \"$ROOT/pialSD_R.nii.gz\" -var white \"$ROOT/whiteSD_R.nii.gz\""
# WB_VARS="-var pial_L \"$ROOT/pialSD_L.nii.gz\" -var white_L \"$ROOT/whiteSD_L.nii.gz\"
# -var pial_R \"$ROOT/pialSD_R.nii.gz\" -var \"$ROOT/white_R whiteSD_R.nii.gz\""

WB_VARS_L=(-var pial "$ROOT/pialSD_L.nii.gz" -var white "$ROOT/whiteSD_L.nii.gz")
WB_VARS_R=(-var pial "$ROOT/pialSD_R.nii.gz" -var white "$ROOT/whiteSD_R.nii.gz")
WB_VARS=(-var pial_L "$ROOT/pialSD_L.nii.gz" -var white_L "$ROOT/whiteSD_L.nii.gz"
-var pial_R "$ROOT/pialSD_R.nii.gz" -var white_R "$ROOT/whiteSD_R.nii.gz")


wb_command -volume-math '(pial < 0) && (white > 0)' "$ROOT/ribbon_L_GM.nii.gz" "${WB_VARS_L[@]}"
wb_command -volume-math '(pial < 0) && (white > 0)' "$ROOT/ribbon_R_GM.nii.gz" "${WB_VARS_R[@]}"

wb_command -volume-math '(white < 0)' "$ROOT/ribbon_L_WM.nii.gz" "${WB_VARS_L[@]}"
wb_command -volume-math '(white < 0)' "$ROOT/ribbon_R_WM.nii.gz" "${WB_VARS_R[@]}"

wb_command -volume-math '(pial < 0)' "$ROOT/ribbon_L_PIAL.nii.gz" "${WB_VARS_L[@]}"
wb_command -volume-math '(pial < 0)' "$ROOT/ribbon_R_PIAL.nii.gz" "${WB_VARS_R[@]}"

wb_command -volume-math '((pial_L < 0) && (white_L > 0)) || ((pial_R < 0) && (white_R > 0))' \
	"$ROOT/ribbon_both_GM.nii.gz" "${WB_VARS[@]}"

wb_command -volume-math '(white_L < 0) || (white_R < 0)' "$ROOT/ribbon_both_WM.nii.gz" "${WB_VARS[@]}"

wb_command -volume-math '(pial_L < 0) || (pial_R < 0)' "$ROOT/ribbon_both_PIAL.nii.gz" "${WB_VARS[@]}"
