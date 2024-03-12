#!/bin/bash

# Exit immediately if any command returns a non-zero status
set -e

old_location="$(pwd)"

# Check if at least two arguments are provided
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <root_descriptor> <output_dir> <name>"
  exit 1
fi

# Change the current working directory to the script's directory

for _ in $(seq 1 100);
  do poetry run python script_train_trires_qupath_version3.py -d "$1" -r "$2" -n "$3";
done

