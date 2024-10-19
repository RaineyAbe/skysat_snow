#! /usr/bin/python/

import argparse
import os
import sys
import xdem
import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import xarray as xr
from skimage.filters import threshold_otsu
import matplotlib


def getparser():
    parser = argparse.ArgumentParser(description="snow_dems_pipeline with arguments passed by the user",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-code_dir', default=None, type=str, help='Path to the snow-dems code root directory')
    parser.add_argument('refdem_fn', default=None, type=str, help='Full path in directory to the reference DEM.')
    parser.add_argument('sourcedem_fn', default=None, type=str, help='Full path in directory to the source ('
                                                                     'to-be-aligned) DEM.')
    parser.add_argument('ortho_fn', default=None, type=str, help='Full path in directory to the orthomosaic image.')
    parser.add_argument('roads_fn', default=None, type=str, help='Full path in directory to geospatial file of roads '
                                                                 'over the input DEMs.')
    parser.add_argument('out_dir', default=None, type=str, help='Directory where outputs will be saved.')
    parser.add_argument('job_name', default=None, type=str, help='Name of job.')
    parser.add_argument('steps_to_run', default=[1, 2, 3, 4, 5], type=list, help='Steps in pipeline to run.')

    return parser


def main():
    # -----SET UP-----
    # Set user arguments as variables
    parser = getparser()
    args = parser.parse_args()
    code_dir = args.code_dir
    refdem_fn = args.refdem_fn
    sourcedem_fn = args.refdem_fn
    out_dir = args.out_dir
    ortho_fn = args.ortho_fn
    roads_fn = args.roads_fn
    job_name = args.job_name
    steps_to_run = args.steps_to_run

    # Check that input files exist
    if not os.path.exists(refdem_fn):
        print('Reference DEM not found, check path to file.')
    if not os.path.exists(sourcedem_fn):
        print('Source DEM not found, check path to file.')
    if not os.path.exists(ortho_fn):
        print('Ortho image not found, check path to file.')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('Created directory for output files:', out_dir)

    # Define output directories and files
    out_dir = os.path.join(out_dir, job_name)
    # step 1: stable surfaces (snow-free) mask by Otsu thresholding
    stable_surfaces_dir = os.path.join(out_dir, 'stable_surfaces')
    roads_mask_fn = os.path.join(stable_surfaces_dir, 'roads_mask.tif')
    ss_mask_fn = os.path.join(stable_surfaces_dir, 'stable_surfaces.tif')
    # step 2: Initial coregistration and differencing
    coreg_init_dir = os.path.join(out_dir, 'coreg_initial_diff')
    # step 3: Quadratic bias correction
    deramp_dir = os.path.join(out_dir, 'deramping')
    # step 4: Final coregistration and differencing
    coreg_final_dir = os.path.join(out_dir, 'coreg_final_diff')
    # Create directories
    for directory in [out_dir, stable_surfaces_dir, coreg_init_dir, deramp_dir, coreg_final_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # Add path to functions
    sys.path.insert(1, os.path.join(code_dir, 'snow-dems'))
    import pipeline_utils as f

    # Crop reference DEM to source DEM extent + 100 m buffer

    if 1 in steps_to_run:
        print('=====================================================')
        print('||            STEP 1: CREATE ROADS MASK            ||')
        print('=====================================================')

        # Convert roads vector to mask according to reference DEM grid
        f.vector_to_mask(roads_fn, refdem_fn, roads_mask_fn)
        print('\n')

    if 2 in steps_to_run:
        print('=====================================================')
        print('||       STEP 2: CREATE STABLE SURFACES MASK       ||')
        print('=====================================================')

        # Create stable surfaces mask
        f.construct_stable_surfaces_mask(ortho_fn, ss_mask_fn)
        print('\n')

    if 3 in steps_to_run:
        print('=====================================================')
        print('||                STEP 3: DERAMPING                ||')
        print('=====================================================')

        # First round
        f.deramp_dem(tba_dem_fn=os.path.join(coreg_init_dir, 'dem_coregistered.tif'),
                     ss_mask_fn=os.path.join(coreg_init_dir, 'stable_surfaces_coregistered.tif'),
                     ref_dem_fn=refdem_fn,
                     out_dir=deramp_dir,
                     poly_order=3)
        # Second round
        f.deramp_dem(tba_dem_fn=os.path.join(deramp_dir, 'dem_coregistered_deramped.tif'),
                     ss_mask_fn=os.path.join(coreg_init_dir, 'stable_surfaces_coregistered.tif'),
                     ref_dem_fn=refdem_fn,
                     out_dir=deramp_dir,
                     poly_order=3)
        print('\n')

    if 4 in steps_to_run:
        print('=====================================================')
        print('||  STEP 4: FINAL COREGISTRATION AND DIFFERENCING  ||')
        print('=====================================================')

        f.coregister_difference_dems(ref_dem_fn=refdem_fn,
                                     source_dem_fn=os.path.join(deramp_dir, 'dem_coregistered_deramped_deramped.tif'),
                                     ss_mask_fn=roads_mask_fn,
                                     out_dir=coreg_final_dir,
                                     coreg_method='NuthKaab',
                                     coreg_stable_only=False,
                                     plot_terrain_results=True)
        print('\n')


if __name__ == '__main__':
    main()
