#!/usr/bin/python
import os
import glob
import shutil
import subprocess
import json
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def getparser():
    parser = argparse.ArgumentParser(description='Script to construct, merge, and filter final point clouds from skysat_stereo.')
    parser.add_argument('-in_dir',default=None,type=str,help='path to final_pinhole_stereo folder')
    parser.add_argument('-out_dir',default=None,type=str,help='path to folder where outputs will be saved')
    parser.add_argument('-refdem_fn',default=None,type=str,help='path to reference DEM file')
    parser.add_argument('-job_name',default='',type=str,help='job name used for output file names')
    return parser

def run_cmd(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end='')  
    for line in process.stderr:
        print(line, end='')
    process.wait()
    return

def convert_and_move(tif_fn, laz_fn, new_laz_fn):
    """Convert TIF to LAZ and move it to the output directory."""
    if os.path.exists(tif_fn) and not os.path.exists(new_laz_fn):
        cmd = ['point2las', tif_fn, '--compressed', '-o', os.path.splitext(laz_fn)[0]]
        run_cmd(cmd)
        # Move the file after the process completes
        shutil.move(laz_fn, new_laz_fn)

def process_folder_second(folder_first, folder_second, in_dir, out_dir):
    """Process a single folder pair (folder_first, folder_second)."""
    tif_fn = os.path.join(in_dir, folder_first, folder_second, 'run-PC.tif')
    laz_fn = os.path.join(in_dir, folder_first, folder_second, 'run-PC.laz')
    new_laz_fn = os.path.join(out_dir, folder_second + '-PC.laz')
    convert_and_move(tif_fn, laz_fn, new_laz_fn)


def main():
    # -----Get user arguments
    parser = getparser()
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    refdem_fn = args.refdem_fn
    job_name = args.job_name
    print(job_name)

    # -----Define output file names
    pc_merged_las_fn = os.path.join(out_dir, f"{job_name}_pc_merged.laz")
    pc_merged_tif_fn = pc_merged_las_fn.replace('.laz', '.tif')
    pc_filtered_las_fn = os.path.join(out_dir, f"{job_name}_pc_merged_filtered.laz")
    pc_filtered_tif_fn = pc_filtered_las_fn.replace('.laz', '.tif')
    json1_fn = os.path.join(out_dir, 'las2tif.json')
    json2_fn = os.path.join(out_dir, 'las2unaligned.json')
    fig_fn = pc_filtered_las_fn.replace('.laz', '.png')
    out_res = 2 # m
    # Create the output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('Made directory for output point clouds:', out_dir)


    # -----Construct point clouds and merge
    if not os.path.exists(pc_merged_las_fn):
        print('\nConstructing point clouds from .tif files')
        folders_first = [x for x in sorted(os.listdir(in_dir)) if '20' in x]
        for folder_first in tqdm(folders_first, desc="Processing first-level folders"):
            folders_second = [x for x in sorted(os.listdir(os.path.join(in_dir, folder_first))) if '20' in x]
            # Use ProcessPoolExecutor to parallelize processing of folder_second
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(process_folder_second, folder_first, folder_second, in_dir, out_dir)
                    for folder_second in folders_second
                ]
                # Show progress with tqdm
                for _ in tqdm(as_completed(futures), total=len(folders_second), desc=f"Processing pairs in {folder_first}"):
                    pass

        # Merge point clouds
        print('\nMerging point clouds')
        pc_fns = sorted(glob.glob(os.path.join(out_dir, '*-PC.laz')))
        cmd = ['pdal', 'merge'] + pc_fns + [pc_merged_las_fn]
        run_cmd(cmd)

    else:
        print('\nMerged point cloud already exists, skipping.')


    # -----Filter point clouds
    ### Rasterize initial point cloud for reference ###
    # Create PDAL pipeline JSON file
    if not os.path.exists(pc_merged_tif_fn):
        print('\nRasterizing initial point cloud for reference')
        reader = {"type": "readers.las", "filename": pc_merged_las_fn}
        # Write tif file
        tif_writer = {
            "type": "writers.gdal",
            "filename": pc_merged_tif_fn,
            "resolution": out_res,
            "output_type": "idw"
        }
        pipeline = [reader, tif_writer]
        # write json out
        with open(json1_fn,'w') as outfile:
            json.dump(pipeline, outfile, indent = 2)
        print('JSON saved to file:', json1_fn)

        # Run pipeline with the JSON
        cmd = ['pdal', 'pipeline', json1_fn]
        run_cmd(cmd)

    else:
        print('\nRasterized point cloud already exists, skipping.')

    ### Filter point cloud ###
    if not os.path.exists(pc_filtered_las_fn):
        print('\nFiltering point cloud')
        # Create pdal pipeline json
        reader = {"type": "readers.las", "filename": pc_merged_las_fn}
        # Filter out points far away from our dem
        dem_filter = {
            "type": "filters.dem",
            "raster": refdem_fn,
            "limits": "Z[25:35]"
        }
        # Extended Local Minimum filter
        elm_filter = {
            "type": "filters.elm"
        }
        # Outlier filter
        outlier_filter = {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": 12,
            "multiplier": 2.2
        }
        # SMRF classifier for ground
        smrf_classifier = {
            "type": "filters.smrf",
            "ignore": "Classification[7:7]" 
        }
        # Select ground points only
        smrf_selecter = { 
            "type": "filters.range",
            "limits": "Classification[2:2]"
        }
        # Write las file
        las_writer = {
            "type": "writers.las",
            "filename": pc_filtered_las_fn
        }
        # Write tif file
        tif_writer = {
            "type": "writers.gdal",
            "filename": pc_filtered_tif_fn,
            "resolution": out_res,
            "output_type": "idw"
        }

        pipeline = [reader, dem_filter, elm_filter, outlier_filter, smrf_classifier, smrf_selecter, las_writer, tif_writer]
        # write json out
        with open(json2_fn,'w') as outfile:
            json.dump(pipeline, outfile, indent = 2)
        print('JSON saved to file:', json2_fn)

        # Run pipeline with the JSON
        print('Filtering point cloud...')
        cmd = ['pdal', 'pipeline', json2_fn]
        run_cmd(cmd)

    else:
        print('\nFiltered point cloud already exists, skipping.')

    # Plot DEM pre- and post-filtering
    if not os.path.exists(fig_fn):
        print('Plotting DEM pre- and post-filtering')
        pc_merged = rxr.open_rasterio(pc_merged_tif_fn).squeeze()
        pc_merged = xr.where(pc_merged==pc_merged.attrs['_FillValue'], np.nan, pc_merged)
        pc_filtered = rxr.open_rasterio(pc_filtered_tif_fn).squeeze()
        pc_filtered = xr.where(pc_filtered==pc_filtered.attrs['_FillValue'], np.nan, pc_filtered)

        fig, ax = plt.subplots(1, 3, figsize=(12,6))
        pc_merged_interp = pc_merged.interp(x=pc_filtered.x, y=pc_filtered.y)
        for i, (dem, cmap, title) in enumerate(list(zip([pc_merged, pc_filtered, pc_merged_interp - pc_filtered],
                                                        ['terrain', 'terrain', 'coolwarm_r'],
                                                        ['Merged point cloud', 'Merged, filtered point cloud', 'Difference']))):
            
            im = ax[i].imshow(dem.data, cmap=cmap,
                            extent=(np.min(dem.x)/1e3, np.max(dem.x)/1e3, np.min(dem.y)/1e3, np.max(dem.y)/1e3))
            if i==2:
                im.set_clim(-5,5)
            fig.colorbar(im, ax=ax[i], orientation='horizontal', label='meters')
            ax[i].set_title(title)
            ax[i].set_xlabel('Easting [km]')
        ax[0].set_ylabel('Northing [km]')
        plt.show()
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
        print('Figure saved to file:', fig_fn)

    print('Done!')


if __name__ == '__main__':
    main()
