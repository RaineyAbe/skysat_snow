#!/usr/bin/python
import os
import glob
import shutil
import subprocess

# -----Define input and output directories
in_dir = '/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/SitKusa/20220906/proc_out/final_pinhole_stereo'
out_dir = '/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/SitKusa/20220906/proc_out/point_clouds'
# Create the output directory if it doesn't exist
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# -----Construct and move the final point clouds into out_dir
# Get first folders
folders_first = [x for x in sorted(os.listdir(in_dir)) if '20' in x]
# Iterate over the first folders
for folder_first in folders_first:
    # Get folder names for all pairs
    folders_second = [x for x in sorted(os.listdir(os.path.join(in_dir, folder_first))) if '20' in x]
    # Iterate over all pairs
    for folder_second in folders_second:
        # Convert point cloud TIF to LAZ
        tif_fn = os.path.join(in_dir, folder_first, folder_second, 'run-PC.tif')
        if os.path.exists(tif_fn):
            laz_fn = os.path.join(in_dir, folder_first, folder_second, 'run-PC.laz')
            new_laz_fn = os.path.join(out_dir, folder_second + '-PC.laz')
            if not os.path.exists(new_laz_fn):
                args = ['point2las', tif_fn, '--compressed', '-o', os.path.splitext(laz_fn)[0]]
                output = subprocess.run(args, shell=False, capture_output=True)
                # Move to out_dir
                shutil.move(laz_fn, new_laz_fn)

# -----Merge point clouds
# Get all LAZ files
laz_fns = glob.glob(os.path.join(out_dir, '*.laz'))
# Run pdal merge
cmd = ['pdal', 'merge'] + laz_fns + ['pc_merged.laz']
output = subprocess.run(cmd, shell=False, capture_output=True)
print(output)

print('DONE! :)')
