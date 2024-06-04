#!/bin/bash

in_dir="/Users/raineyaberle/Research/PhD/SnowDEMs/DCEW/lidar/20220215"
out_dir=$in_dir
resolution=10

# Count total number of files for tqdm
total_files=$(find "$in_dir" -maxdepth 1 -type f -name '*.laz' | wc -l)
progress=0

# Iterate over files
for fn in "$in_dir"/*.laz; do
    bn=$(basename "$fn")
    out_fn="$out_dir"/"${bn/.laz/_${resolution}m.tif}"
    # Run the pipeline if TIF file doesn't already exist
    if [ ! -f "$out_fn" ]; then
        cmd="pdal pipeline pdal_pipeline.json --readers.las.filename=\"$fn\" --writers.gdal.resolution=$resolution --writers.gdal.filename=\"$out_fn\""
        eval "$cmd"
    fi
    # Print progress
    ((progress++))
    echo -ne "Progress: $progress/$total_files\r"
done
echo ""
