#!/bin/bash

set -e
set +x

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 input_folder output_folder [tcpoptions]"
    exit 1
fi

# Get the directory where the current script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

input_folder="$1"
output_folder="$2"
tcpoptions=0
if [ "$#" -eq 3 ]; then
    tcpoptions="$3"
fi


# Check if input_folder exists and is a directory
if [ ! -d "$input_folder" ]; then
    echo "Error: Input folder '$input_folder' does not exist or is not a directory."
    exit 1
fi

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Check if 3_field_extraction script exists in the same directory as the current script
field_extraction_script="$script_dir/3_field_extraction"
if [ ! -f "$field_extraction_script" ]; then
    echo "Error: 3_field_extraction script not found in $script_dir, run make all"
    exit 1
fi

# Create output directories for each subdirectory in the input folder
find "$input_folder" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' dir; do
    dir_name="$(basename "$dir")"
    mkdir -p "$output_folder/$dir_name"
done

find "$input_folder" -mindepth 1 -maxdepth 1 -type d -print0 | parallel -0 "$field_extraction_script {} $output_folder/{/} $tcpoptions"
