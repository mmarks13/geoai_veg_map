#!/bin/bash

# Define the directory
INPUT_DIR="uavlidar/original_las"
OUTPUT_DIR="${INPUT_DIR}/compressed"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all .las files and compress them to .laz
for file in "$INPUT_DIR"/*.las; do
    if [[ -f "$file" ]]; then
        laszip -i "$file" -o "$OUTPUT_DIR/$(basename "${file%.las}.laz")"
    fi
done

echo "Compression complete. Compressed files are in: $OUTPUT_DIR"

