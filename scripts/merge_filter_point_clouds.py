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

def get_parser():
    parser = argparse.ArgumentParser(description='Script to construct, merge, and filter final point clouds from skysat_stereo.')
    parser.add_argument('-in_dir', default=None, type=str, help='path to final_pinhole_stereo folder')
    parser.add_argument('-out_dir', default=None, type=str, help='path to folder where outputs will be saved')
    parser.add_argument('-refdem_fn', default=None, type=str, help='path to reference DEM file')
    parser.add_argument('-job_name', default='', type=str, help='job name used for output file names')
    return parser

def run_cmd(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end='')  
    for line in process.stderr:
        print(line, end='')
    process.wait()

def convert_and_move(tif_fn, laz_fn, new_laz_fn):
    if os.path.exists(tif_fn) and not os.path.exists(new_laz_fn):
        cmd = ['point2las', tif_fn, '--compressed', '-o', os.path.splitext(laz_fn)[0]]
        run_cmd(cmd)
        shutil.move(laz_fn, new_laz_fn)

def process_folder_second(folder_first, folder_second, in_dir, out_dir):
    tif_fn = os.path.join(in_dir, folder_first, folder_second, 'run-PC.tif')
    laz_fn = os.path.join(in_dir, folder_first, folder_second, 'run-PC.laz')
    new_laz_fn = os.path.join(out_dir, f"{folder_second}-PC.laz")
    convert_and_move(tif_fn, laz_fn, new_laz_fn)

def construct_point_clouds(in_dir, out_dir):
    folders_first = [x for x in sorted(os.listdir(in_dir)) if '20' in x]
    for folder_first in tqdm(folders_first, desc="Processing first-level folders"):
        folders_second = [x for x in sorted(os.listdir(os.path.join(in_dir, folder_first))) if '20' in x]
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_folder_second, folder_first, folder_second, in_dir, out_dir)
                for folder_second in folders_second
            ]
            for _ in tqdm(as_completed(futures), total=len(folders_second), desc=f"Processing pairs in {folder_first}"):
                pass

def merge_point_clouds(out_dir, pc_merged_las_fn):
    pc_fns = sorted(glob.glob(os.path.join(out_dir, '*-PC.laz')))
    cmd = ['pdal', 'merge'] + pc_fns + [pc_merged_las_fn]
    run_cmd(cmd)

def create_pdal_pipeline_json(filename, reader, writers):
    pipeline = [reader] + writers
    with open(filename, 'w') as outfile:
        json.dump(pipeline, outfile, indent=2)
    print(f'JSON saved to file: {filename}')
    return filename

def rasterize_point_cloud(pc_merged_las_fn, pc_merged_tif_fn, out_res, json1_fn):
    reader = {"type": "readers.las", "filename": pc_merged_las_fn}
    tif_writer = {"type": "writers.gdal", "filename": pc_merged_tif_fn, "resolution": out_res, "output_type": "idw"}
    create_pdal_pipeline_json(json1_fn, reader, [tif_writer])
    run_cmd(['pdal', 'pipeline', json1_fn])

def filter_point_cloud(pc_merged_las_fn, pc_filtered_las_fn, pc_filtered_tif_fn, refdem_fn, out_res, json2_fn):
    reader = {"type": "readers.las", "filename": pc_merged_las_fn}
    dem_filter = {"type": "filters.dem", "raster": refdem_fn, "limits": "Z[25:35]"}
    elm_filter = {"type": "filters.elm"}
    outlier_filter = {"type": "filters.outlier", "method": "statistical", "mean_k": 12, "multiplier": 2.2}
    smrf_classifier = {"type": "filters.smrf", "ignore": "Classification[7:7]"}
    smrf_selecter = {"type": "filters.range", "limits": "Classification[2:2]"}
    las_writer = {"type": "writers.las", "filename": pc_filtered_las_fn}
    tif_writer = {"type": "writers.gdal", "filename": pc_filtered_tif_fn, "resolution": out_res, "output_type": "idw"}
    create_pdal_pipeline_json(json2_fn, reader, [dem_filter, elm_filter, outlier_filter, smrf_classifier, smrf_selecter, las_writer, tif_writer])
    run_cmd(['pdal', 'pipeline', json2_fn])

def plot_dem_comparison(pc_merged_tif_fn, pc_filtered_tif_fn, fig_fn):
    pc_merged = rxr.open_rasterio(pc_merged_tif_fn).squeeze()
    pc_merged = xr.where(pc_merged == pc_merged.attrs['_FillValue'], np.nan, pc_merged)
    pc_filtered = rxr.open_rasterio(pc_filtered_tif_fn).squeeze()
    pc_filtered = xr.where(pc_filtered == pc_filtered.attrs['_FillValue'], np.nan, pc_filtered)

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    pc_merged_interp = pc_merged.interp(x=pc_filtered.x, y=pc_filtered.y)
    for i, (dem, cmap, title) in enumerate(zip([pc_merged, pc_filtered, pc_merged_interp - pc_filtered],
                                               ['terrain', 'terrain', 'coolwarm_r'],
                                               ['Merged point cloud', 'Filtered point cloud', 'Difference'])):
        im = ax[i].imshow(dem.data, cmap=cmap,
                          extent=(np.min(dem.x)/1e3, np.max(dem.x)/1e3, np.min(dem.y)/1e3, np.max(dem.y)/1e3))
        if i == 2:
            im.set_clim(-5, 5)
        fig.colorbar(im, ax=ax[i], orientation='horizontal', label='meters')
        ax[i].set_title(title)
        ax[i].set_xlabel('Easting [km]')
    ax[0].set_ylabel('Northing [km]')
    plt.show()
    fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
    print(f'Figure saved to file: {fig_fn}')

def main():
    parser = get_parser()
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    refdem_fn = args.refdem_fn
    job_name = args.job_name

    pc_merged_las_fn = os.path.join(out_dir, f"{job_name}_pc_merged.laz")
    pc_merged_tif_fn = pc_merged_las_fn.replace('.laz', '.tif')
    pc_filtered_las_fn = os.path.join(out_dir, f"{job_name}_pc_merged_filtered.laz")
    pc_filtered_tif_fn = pc_filtered_las_fn.replace('.laz', '.tif')
    json1_fn = os.path.join(out_dir, 'las2tif.json')
    json2_fn = os.path.join(out_dir, 'las2unaligned.json')
    fig_fn = pc_filtered_las_fn.replace('.laz', '.png')
    out_res = 2  # m

    os.makedirs(out_dir, exist_ok=True)
    print(f'Made directory for output point clouds: {out_dir}')

    if not os.path.exists(pc_merged_las_fn):
        construct_point_clouds(in_dir, out_dir)
        print('Merging point clouds...')
        merge_point_clouds(out_dir, pc_merged_las_fn)
    else:
        print('Merged point cloud already exists, skipping.')

    if not os.path.exists(pc_merged_tif_fn):
        print('Rasterizing initial point cloud for reference...')
        rasterize_point_cloud(pc_merged_las_fn, pc_merged_tif_fn, out_res, json1_fn)
    else:
        print('Rasterized point cloud already exists, skipping.')

    if not os.path.exists(pc_filtered_las_fn):
        print('Filtering point cloud...')
        filter_point_cloud(pc_merged_las_fn, pc_filtered_las_fn, pc_filtered_tif_fn, refdem_fn, out_res, json2_fn)
    else:
        print('Filtered point cloud already exists, skipping.')

    if not os.path.exists(fig_fn):
        print('Plotting DEM pre- and post-filtering...')
        plot_dem_comparison(pc_merged_tif_fn, pc_filtered_tif_fn, fig_fn)

    print('Done!')

if __name__ == '__main__':
    main()
