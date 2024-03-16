#!/bin/bash

# Define the directory containing the video files
video_dir="$HOME/first-impression/videos-train/"

# Define the output directory for features
output_dir="$HOME/first-impression/openface/features-train/"

# Check if the output directory exists, if not create it
mkdir -p "$output_dir"

# Loop through each .mp4 file in the videos directory
for file in "$video_dir"*.mp4; do
    if [ -f "$file" ]; then
        # Extract filename without extension
        filename=$(basename -- "$file")
        filename_no_ext="${filename%.*}"
        
        # Run FeatureExtraction command
        FeatureExtraction -f "$file" -out_dir "$output_dir"
        
        echo "Features extracted for $filename"
    fi
done

echo "Script execution complete."
