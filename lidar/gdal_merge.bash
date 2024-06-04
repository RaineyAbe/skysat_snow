#!/bin/bash

in_dir="/Users/raineyaberle/Research/PhD/SnowDEMs/DCEW/lidar/20220215"
out_dir=$in_dir
pixel_size=10
nodata_value=-9999

out_fn="20220215_mosaic_${resolution}m.tif"

# Create command
cmd="gdal_merge.py -ps ${pixel_size} -a_nodata ${nodata_value} -o ${out_fn}"

# Iterate over files
for fn in "$in_dir"/*.tif; do
    # Add file to command
    cmd+=" ${fn}"

done

eval "$cmd"
