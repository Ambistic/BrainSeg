#!/bin/bash

# Exit immediately if any command returns a non-zero status
set -e

old_location="$(pwd)"
script_dir="$(dirname "$0")"

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <script_to_run> <argument>"
  exit 1
fi

# Change the current working directory to the script's directory
cd "$script_dir"

# Assign the first argument to a variable
script_to_run="$1"

# Check which script to run and execute it with the second argument
if [ "$script_to_run" = "forward" ]; then
  poetry run python script_forward_histo_mri_v2.py --config "$2"
  poetry run python script_reconstruct_density.py --config "$2"
  poetry run python script_parcellate_surfaces.py --config "$2"
elif [ "$script_to_run" = "backward" ]; then
  poetry run python script_backward_mri_histo_v2.py --config "$2"
  poetry run python script_count_neurons.py --config "$2"
elif [ "$script_to_run" = "forward_backward" ]; then
  poetry run python script_forward_histo_mri_v2.py --config "$2"
  poetry run python script_reconstruct_density.py --config "$2"
  poetry run python script_parcellate_surfaces.py --config "$2"
  poetry run python script_backward_mri_histo_v2.py --config "$2"
  poetry run python script_count_neurons.py --config "$2"
elif [ "$script_to_run" = "segmentation" ]; then
  poetry run python apply_qupath_version3.py --config "$2"
  poetry run python convert_png_svg_qupath_version3.py --config "$2"
elif [ "$script_to_run" = "make_sections" ]; then
  poetry run python script_create_mri_sections.py --config "$2"
  poetry run python script_create_atlas_sections.py --config "$2"
elif [ "$script_to_run" = "make_transform" ]; then
  poetry run python script_create_transforms_v3.py --config "$2"
elif [ "$script_to_run" = "select_angle" ]; then
  poetry run python script_angle_selection.py --config "$2"
elif [ "$script_to_run" = "merge" ]; then
  poetry run python register_plotfast_cv_qupath_version2.py --config "$2"
elif [ "$script_to_run" = "merge_fluo" ]; then
  poetry run python register_fluo_cv_qupath_version2.py --config "$2"
elif [ "$script_to_run" = "qc" ]; then
  poetry run python script_qc.py --config "$2"
elif [ "$script_to_run" = "build_image_transform" ]; then
  poetry run python script_create_image_transform.py --config "$2"
else
  echo "Invalid script name. Please choose 'scriptA' or 'scriptB'."
  exit 1
fi

cd "$old_location"
