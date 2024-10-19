#! /usr/bin/python

import os
import glob
import math
import xdem
import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import subprocess
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from shapely import clip_by_rect, unary_union
from scipy.optimize import curve_fit
import laspy
import xdem
import geoutils as gu
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def convert_wgs_to_utm(lon: float, lat: float):
    """
    Return best UTM epsg-code based on WGS84 lat and lon coordinate pair

    Parameters
    ----------
    lon: float
        longitude coordinate
    lat: float
        latitude coordinate

    Returns
    ----------
    epsg_code: str
        optimal UTM zone, e.g. "EPSG:32606"
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return f'EPSG:{epsg_code}'


def preprocess_multispec_refdem_roads(multispec_dir, refdem_fn, roads_fn, roads_buffer, out_res, out_dir):

    # Check if multispec_dir is a directory of images or a single file
    if os.path.isdir(multispec_dir):
        multispec_mosaic_fn = os.path.join(out_dir, '4band_mosaic.tif')
        # Check if mosaic exists
        if not os.path.exists(multispec_mosaic_fn):
            print('Mosacking 4-band SR images...')
            # Grab all 4-band SR image file names
            multispec_fns = sorted(glob.glob(os.path.join(multispec_dir, '*_SR.tif')))
            # Construct gdal_merge command
            cmd = ['gdal_merge'] + multispec_fns + ['-o', multispec_mosaic_fn]
            # Run command
            output = subprocess.run(cmd, shell=False, capture_output=True)
            print(output)
        else:
            print('4-band mosaic already exists, skipping gdal_merge.')
    elif os.path.isfile(multispec_dir):
        multispec_mosaic_fn = multispec_dir

    # Determine optimal UTM zone
    multispec_mosaic = rxr.open_rasterio(multispec_mosaic_fn)
    multispec_mosaic_latlon = rxr.open_rasterio(multispec_mosaic_fn).rio.reproject('EPSG:4326')
    bounds = multispec_mosaic_latlon.rio.bounds()
    cen_lon, cen_lat = (bounds[2] + bounds[0]) / 2, (bounds[3] + bounds[1]) / 2
    crs_utm = convert_wgs_to_utm(cen_lon, cen_lat)
    print('Optimal UTM CRS = ', crs_utm)

    # Determine cropping extent from 4-band image
    multispec_mosaic = multispec_mosaic_latlon.rio.reproject(crs_utm)
    bounds = multispec_mosaic.rio.bounds()
    bounds_buffered = [bounds[0]-1e3, bounds[1]-1e3, bounds[2]+1e3, bounds[3]+1e3]
    
    # Roads: buffer and crop to extent
    roads_adj_fn = os.path.join(out_dir, f"roads_buffered_cropped_{crs_utm.replace(':', '')}.gpkg")
    if not os.path.exists(roads_adj_fn):
        print('Merging, buffering, and cropping roads...')
        roads = gpd.read_file(roads_fn)
        roads = roads.to_crs(crs_utm)
        geom = unary_union(roads['geometry'].values).buffer(roads_buffer)
        geom_crop = clip_by_rect(geom, bounds_buffered[0], bounds_buffered[1], bounds_buffered[2], bounds_buffered[3])
        roads_adj = gpd.GeoDataFrame(geometry=[geom_crop], crs=crs_utm)
        roads_adj.to_file(roads_adj_fn)
        print('Adjusted roads vector saved to file:', roads_adj_fn)
    else:
        print('Adjusted roads vector already exist in file, skipping.')

    # Reference DEM: regrid to desired CRS and crop
    refdem_adj_fn = os.path.join(out_dir, f"refdem_{crs_utm.replace(':', '')}.tif")
    if not os.path.exists(refdem_adj_fn):
        print('Reprojecting and cropping reference DEM...')
        bounds_buffered = list(np.array(bounds_buffered).astype(str))
        cmd = ['gdalwarp', '-t_srs', crs_utm, '-te'] + bounds_buffered + [refdem_fn, refdem_adj_fn]
        output = subprocess.run(cmd, shell=False, capture_output=True)
        print(output)
    else:
        print('Adjusted reference DEM already exists in file, skipping.')

    return multispec_mosaic_fn, roads_adj_fn, refdem_adj_fn


def preprocess_point_clouds(pc_las_fn, ref_dem_fn, resolution, out_dir, plot_results=True):
    """
    Merge, filter, and rasterize point clouds using PDAL and GDAL.

    Parameters
    ----------
    pc_las_fn: str or Path
        path to point cloud .las or .laz file
    ref_dem_fn: str or Path
        full file name of the reference DEM
    resolution: int or float
        spatial resolution of the output raster
    out_dir: str or Path
        path where outputs will be saved

    Returns
    ----------
    pc_filtered_las_fn: str
        full file name of the filtered point cloud LAZ
    pc_filtered_tif_fn: str
        full file name of the filtered point cloud TIF
    """
    #### Define output files ###
    pc_merged_tif_fn = os.path.join(out_dir, 'pc_merged.tif')
    json1_fn = os.path.join(out_dir, 'las2tif.json')
    json2_fn = os.path.join(out_dir, 'las2unaligned.json')
    pc_filtered_las_fn = os.path.join(out_dir, 'pc_merged_filtered.laz')
    pc_filtered_tif_fn = os.path.join(out_dir, 'pc_merged_filtered.tif')
    fig_fn = os.path.join(out_dir, 'pc_filtered.png')

    ### Rasterize initial point cloud for reference ###
    # Create PDAL pipeline JSON file
    if not os.path.exists(json1_fn):
        reader = {"type": "readers.las", "filename": pc_las_fn}
        # Write tif file
        tif_writer = {
            "type": "writers.gdal",
            "filename": pc_merged_tif_fn,
            "resolution": resolution,
            "output_type": "idw"
        }
        pipeline = [reader, tif_writer]
        # write json out
        with open(json1_fn,'w') as outfile:
            json.dump(pipeline, outfile, indent = 2)
        print('JSON saved to file:', json1_fn)
    # Run pipeline with the JSON
    if not os.path.exists(pc_merged_tif_fn):
        print('Rasterizing point cloud...')
        cmd = ['pdal', 'pipeline', json1_fn]
        output = subprocess.run(cmd, shell=False, capture_output=True)
        print(output)
    else:
        print('Rasterized point cloud already exists, skipping.')


    ### Filter point cloud ###
    # Create pdal pipeline json
    if not os.path.exists(json2_fn):
        reader = {"type": "readers.las", "filename": pc_las_fn}
        # Filter out points far away from our dem
        dem_filter = {
            "type": "filters.dem",
            "raster": ref_dem_fn,
            "limits": "Z[25:35]"
        }
        # Extended Local Minimum filter - might be filtering points below trees!
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
            "resolution": resolution,
            "output_type": "idw"
        }

        pipeline = [reader, dem_filter, elm_filter, outlier_filter, smrf_classifier, smrf_selecter, las_writer, tif_writer]
        # write json out
        with open(json2_fn,'w') as outfile:
            json.dump(pipeline, outfile, indent = 2)
        print('JSON saved to file:', json2_fn)
    # Run pipeline with the JSON
    if not os.path.exists(pc_filtered_las_fn):
        print('Filtering point cloud...')
        cmd = ['pdal', 'pipeline', json2_fn]
        output = subprocess.run(cmd, shell=False, capture_output=True)
        print(output)
    else:
        print('Filtered point cloud already exists, skipping.')

    # Plot the difference pre- and post-filtering
    if plot_results & (not os.path.exists(fig_fn)):
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

    return pc_filtered_las_fn, pc_filtered_tif_fn


def construct_land_cover_masks(multispec_mosaic_fn, roads_vector_fn, out_dir, roads_buffer=3, ndvi_threshold=0.5, ndsi_threshold=0.1, plot_results=True):
    # Define output files
    trees_mask_fn = os.path.join(out_dir, 'trees_mask.tif')
    snow_mask_fn = os.path.join(out_dir, 'snow_mask.tif')
    roads_mask_fn = os.path.join(out_dir, 'roads_mask.tif')
    ss_mask_fn = os.path.join(out_dir, 'stable_surfaces_mask.tif')
    fig_fn = os.path.join(out_dir, 'land_cover_masks.png')

    # Function to load 4-band mosaic if needed
    mosaic = None
    def load_mosaic(multispec_mosaic_fn):
        mosaic_xr = rxr.open_rasterio(multispec_mosaic_fn)
        crs = f'EPSG:{mosaic_xr.rio.crs.to_epsg()}'
        mosaic = xr.Dataset(coords={'y': mosaic_xr.y, 'x':mosaic_xr.x})
        bands = ['blue', 'green', 'red', 'NIR']
        for i, b in enumerate(bands):
            mosaic[b] = mosaic_xr.isel(band=i)
        mosaic = mosaic / 1e4
        mosaic = xr.where(mosaic==0, np.nan, mosaic)
        mosaic.rio.write_crs(crs, inplace=True)
        return mosaic, crs

    # Construct trees mask
    if not os.path.exists(trees_mask_fn):
        print('Constructing trees mask...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Calculate NDVI
        ndvi = (mosaic.NIR - mosaic.red) / (mosaic.NIR + mosaic.red)
        # Apply threshold
        trees_mask = (ndvi >= ndvi_threshold).astype(int)
        # Save to file
        trees_mask = xr.where(np.isnan(mosaic.blue), -9999, trees_mask) # set no data values to -9999
        trees_mask = trees_mask.assign_attrs({'Description': 'Trees mask constructing by thresholding the NDVI of the 4-band mosaic image.',
                                            '_FillValue': -9999,
                                            'NDVI bands': 'NIR, green',
                                            'NDVI threshold': ndvi_threshold})
        trees_mask.rio.write_crs(crs, inplace=True)
        trees_mask.rio.to_raster(trees_mask_fn, dtype='int16')
        print('Trees mask saved to file:', trees_mask_fn)
    else:
        print('Trees mask exists in directory, skipping.')

    # Construct snow mask
    if not os.path.exists(snow_mask_fn):
        print('Constructing snow mask...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Calculate NDSI
        ndsi = (mosaic.red - mosaic.NIR) / (mosaic.red + mosaic.NIR)
        # Apply threshold
        snow_mask = (ndsi >= ndsi_threshold).astype(int)
        # Save to file
        snow_mask = xr.where(np.isnan(mosaic.blue), -9999, snow_mask) # set no data values to -9999
        snow_mask = snow_mask.assign_attrs({'Description': 'Snow mask constructed by thresholding the NDSI of the orthomosaic image',
                                            '_FillValue': -9999,
                                            'NDSI bands': 'red, NIR',
                                            'NDSI threshold': ndsi_threshold})
        snow_mask.rio.write_crs(crs, inplace=True)
        snow_mask.rio.to_raster(snow_mask_fn, dtype='int16')
        print('Snow mask saved to file:', snow_mask_fn)
    else:
        print('Snow mask exists in directory, skipping.')

    # Construct roads mask
    if not os.path.exists(roads_mask_fn):
        print('Constructing roads mask...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Load roads vector file
        roads_vector = gpd.read_file(roads_vector_fn) 
        # Convert to mask
        roads_mask = mosaic.blue.rio.clip(roads_vector.geometry.values, roads_vector.crs, drop=False)
        roads_mask = xr.where(np.isnan(roads_mask), 0, 1)
        # Save to file
        roads_mask = xr.where(np.isnan(mosaic.blue), -9999, roads_mask)
        roads_mask = roads_mask.assign_attrs({'Description': 'Roads mask constructed by buffering, rasterizing, and interpolating the geospatial roads file to the 4-band image grid.',
                                            '_FillValue': -9999,
                                            'buffer_m': roads_buffer})
        roads_mask.rio.write_crs(crs, inplace=True)
        roads_mask.rio.to_raster(roads_mask_fn, dtype='int16')
        print('Roads mask saved to file:', roads_mask_fn)
    else: 
        print('Roads mask exists in directory, skipping.')

    # Construct stable surfaces (snow-free and tree-free) mask
    if not os.path.exists(ss_mask_fn):
        print('Constructing stable surfaces mask...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Load trees and snow masks
        trees_mask = rxr.open_rasterio(trees_mask_fn).squeeze()
        snow_mask = rxr.open_rasterio(snow_mask_fn).squeeze()
        # Stable surfaces = snow-free and tree-free
        ss_mask = xr.where((trees_mask==0) & (snow_mask==0), 1, 0)
        # Save to file
        ss_mask = xr.where(np.isnan(mosaic.blue), -9999, ss_mask)
        ss_mask = ss_mask.assign_attrs({'Description': 'Stable surfaces (snow-free and tree-free) mask.',
                                        '_FillValue': -9999})
        ss_mask.rio.write_crs(crs, inplace=True)
        ss_mask.rio.to_raster(ss_mask_fn, dtype='int16')
        print('Stable surfaces mask saved to file:', ss_mask_fn)
    else: 
        print('Stable surfaces mask exists in directory, skipping.')

    # Plot land cover masks
    if plot_results & (not os.path.exists(fig_fn)):
        print('Plotting land cover masks...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Load masks
        trees_mask = rxr.open_rasterio(trees_mask_fn).squeeze()
        snow_mask = rxr.open_rasterio(snow_mask_fn).squeeze()
        roads_mask = rxr.open_rasterio(roads_mask_fn).squeeze()
        ss_mask = rxr.open_rasterio(ss_mask_fn).squeeze()
        # Define land cover colors
        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        # RGB image
        ax[0].imshow(np.dstack([mosaic.red.data, mosaic.green.data, mosaic.blue.data]), clim=(0,0.6),
                     extent=(np.min(mosaic.x)/1e3, np.max(mosaic.x)/1e3, np.min(mosaic.y)/1e3, np.max(mosaic.y)/1e3))
        ax[0].set_title('RGB mosaic')
        ax[0].set_xlabel('Easting [km]')
        ax[0].set_ylabel('Northing [km]')
        # Land cover masks
        colors_dict = {'trees': '#006d2c', 
                       'snow': '#4292c6', 
                       'stable_surfaces': '#bdbdbd', 
                       'roads': '#662506'}
        for mask, label in zip([trees_mask, snow_mask, ss_mask, roads_mask], list(colors_dict.keys())):
            cmap = matplotlib.colors.ListedColormap([(0,0,0,0), matplotlib.colors.to_rgb(colors_dict[label])])
            ax[1].imshow(mask, cmap=cmap, clim=(0,1),
                        extent=(np.min(mask.x)/1e3, np.max(mask.x)/1e3, np.min(mask.y)/1e3, np.max(mask.y)/1e3))
            # dummy point for legend
            ax[1].plot(0, 0, 's', color=colors_dict[label], markersize=10, label=label) 
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())
        ax[1].legend(loc='best')
        ax[1].set_xlabel('Easting [km]')

        # Save figure
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
        print('Figure saved to file:', fig_fn)
        plt.close()
    elif plot_results:
        print('Masks figure exists in directory, skipping plotting.')

    return multispec_mosaic_fn, trees_mask_fn, snow_mask_fn, roads_mask_fn, ss_mask_fn

    
def deramp(ref_dem_fn, tba_dem_fn, tba_pc_fn, out_dir, vmin=-10, vmax=10, plot_results=True):
    # Define output file names
    dem_deramped_fn = os.path.join(out_dir, os.path.basename(tba_dem_fn).replace('.tif', '_deramped.tif'))
    pc_deramped_fn = os.path.join(out_dir, os.path.basename(tba_pc_fn).replace('.laz', '_deramped.laz'))
    fig_fn = os.path.join(out_dir, 'deramping.png')

    # Check if deramped DEM already exists
    if os.path.exists(pc_deramped_fn):
        print('Deramped DEM already exists, skipping.')
        return pc_deramped_fn

    # Load and preprocess DEMs
    print('Loading input files...')
    ref_dem = xdem.DEM(ref_dem_fn)
    tba_dem = xdem.DEM(tba_dem_fn)
    tba_dem = tba_dem.reproject(ref_dem)

    # Calculate difference before deramping
    print('Calculating difference before deramping...')
    diff_before = tba_dem - ref_dem
    # Get the x and y coordinates
    x, y = tba_dem.coords()
    # Create non-nan mask (no NaNs allowed in scipy.optimize.curve_fit)
    ireal = ~diff_before.data.mask

    # Flatten the arrays for scipy.optimize
    x_flat = x[ireal].ravel()
    y_flat = y[ireal].ravel()
    diff_before_flat = diff_before.data.data[ireal].ravel()

    # Fit the quadratic surface
    print('Fitting a quadratic surface to the dDEM...')
    def quadratic_surface(XY, a, b, c, d, e, f): 
        X, Y = XY 
        return a*X**2 + b*Y**2 + c*X*Y + d*X + e*Y + f
    popt, _ = curve_fit(quadratic_surface, (x_flat, y_flat), diff_before_flat)    

    # Deramp the DEM
    print('Deramping the DEM raster...')
    ireal = ~np.isnan(tba_dem.data)
    x_flat, y_flat = x[ireal].ravel(), y[ireal].ravel()
    ramp_flat = quadratic_surface((x_flat, y_flat), *popt)
    ramp = np.nan * np.ones(np.shape(ireal))
    ramp[ireal] = ramp_flat
    dem_deramped = tba_dem - ramp
    dem_deramped.save(dem_deramped_fn)
    print('Deramped DEM saved to file:', dem_deramped_fn)

    # Deramp the point cloud
    print('Deramping the DEM point cloud...')
    pc = laspy.read(tba_pc_fn)
    pc_x, pc_y, pc_z = np.array(pc.x), np.array(pc.y), pc.z
    ramp_flat = quadratic_surface((pc_x, pc_y), *popt)
    pc.z = pc_z - ramp_flat
    pc.write(pc_deramped_fn)
    print('Deramped point cloud saved to file:', pc_deramped_fn)

    # Calculate difference after
    print('Calculating dDEM after deramping...')
    diff_after = dem_deramped - ref_dem

    # Plot deramping results
    if plot_results:
        print('Plotting results...')
        fig, ax = plt.subplots(2, 2, figsize=(12,10), gridspec_kw=dict(height_ratios=[2,1]))
        ax = ax.flatten()
        ax[0].imshow(diff_before.data, cmap='coolwarm_r', vmin=vmin, vmax=vmax,
                    extent=np.divide(diff_before.bounds, 1e3))
        ax[0].set_title('dDEM')
        ax[0].set_xlabel('Easting [km]')
        ax[0].set_ylabel('Northing [km]')
        ax[1].imshow(diff_after.data, cmap='coolwarm_r', vmin=vmin, vmax=vmax,
                    extent=np.divide(diff_after.bounds, 1e3))
        ax[1].set_title('Deramped dDEM')
        ax[1].set_xlabel('Easting [km]')
        bins = np.linspace(vmin, vmax, 100)
        ax[2].hist(diff_before.data.ravel(), bins=bins, color='gray')
        ax[2].set_xlabel('Difference [m]')
        ax[2].set_xlim(vmin, vmax)
        ax[3].hist(diff_after.data.ravel(), bins=bins, color='gray')
        ax[3].set_xlabel('Difference [m]')
        ax[3].set_xlim(vmin, vmax)

        # Save figure
        fig.savefig(fig_fn, dpi=250, bbox_inches='tight')
        print('Figure saved to file:', fig_fn)
        plt.close()

    return pc_deramped_fn


def align_transform_pc(asp_dir, tba_pc_fn, ref_dem_fn, out_res, out_align_dir, out_transform_dir):
    # Clip reference DEM to stable surfaces
    # ref_dem_clip_fn = os.path.join(out_align_dir, os.path.splitext(os.path.basename(ref_dem_fn))[0] + '_ss.tif')

    # Align point clouds
    pc_align_prefix = os.path.join(out_align_dir,'pc-align')
    init_transform = f"{pc_align_prefix}-transform.txt"
    if not os.path.exists(init_transform):
        print('Beginning pc_align...')
        cmd = [os.path.join(asp_dir, 'pc_align'), '--max-displacement', '5', '--highest-accuracy',
               ref_dem_fn, tba_pc_fn, '-o', pc_align_prefix]
        output = subprocess.run(cmd, shell=False, capture_output=True)
        print(output)
    else:
        print('Initial transform already exists, skipping.')

    # Transform point cloud
    pc_transform_prefix = os.path.join(out_transform_dir, 'pc-transform')
    final_transform = pc_transform_prefix + '-trans_source.laz'
    if not os.path.exists(final_transform):
        print('Beginning pc_transform...')
        cmd = [os.path.join(asp_dir, 'pc_align'), '--max-displacement', '-1', '--num-iterations', '0',
            '--initial-transform', init_transform, '--save-transformed-source-points',
            ref_dem_fn, tba_pc_fn, '-o', pc_transform_prefix]
        output = subprocess.run(cmd, shell=False, capture_output=True)
        print(output)
    else:
        print('Final transform already exists, skipping.')

    # Rasterize coregistered DEM
    print('Beginning rasterization of coregistered DEM...')
    cmd = [os.path.join(asp_dir, 'point2dem'), final_transform,
           '--dem-spacing', str(out_res), '--search-radius-factor', '2', 
           '-o', pc_transform_prefix]
    output = subprocess.run(cmd, shell=False, capture_output=True)
    print(output)
    final_tif_fn = glob.glob(os.path.join(pc_transform_prefix + '-DEM.tif'))[0]

    return final_tif_fn


def raster_adjustments(ref_dem_fn, tba_dem_fn, ss_mask_fn, out_res, out_dir, vmin=-5, vmax=5, plot_results=True):

    # Define output files
    out_dem_fn = os.path.join(out_dir, 'final_DEM.tif')
    out_ddem_fn = os.path.join(out_dir, 'final_dDEM.tif')
    fig_fn = os.path.join(out_dir, 'final_dDEM.png')

    # Check if output files already exist
    if os.path.exists(out_dem_fn):
        print('Final DEM already exists, skipping.')
        return out_dem_fn, out_ddem_fn

    # Load input files
    print('Loading input files...')
    ref_dem = xdem.DEM(ref_dem_fn)
    ref_dem = ref_dem.reproject(res=out_res)
    dem = xdem.DEM(tba_dem_fn)
    ss_mask = gu.Raster(ss_mask_fn, load_data=True)
    # Reproject DEM and ss_mask to reference DEM
    dem = dem.reproject(ref_dem)
    ss_mask = ss_mask.reproject(ref_dem)
    ss_mask = (ss_mask==1)

    # Calculate dDEM and stable surface stats before adjustments
    print('Calculating dDEM before adjustments...')
    ddem_before = dem - ref_dem
    ddem_before_ss = ddem_before[ss_mask]
    ddem_before_ss_med, ddem_before_ss_nmad = np.median(ddem_before_ss), xdem.spatialstats.nmad(ddem_before_ss)

    # Deramp
    # print('Deramping...')
    # deramp = xdem.coreg.Deramp(poly_order=2).fit(ref_dem, dem, inlier_mask=ss_mask)
    # dem_deramped = deramp.apply(dem)

    # Coregister
    print('Coregistering...')
    coreg = xdem.coreg.NuthKaab().fit(ref_dem, dem, inlier_mask=ss_mask)
    print('Coregistration fit:\n', coreg.meta)
    dem_coreg = coreg.apply(dem)

    # Calculate differences after adjustments
    print('Calculating dDEM after adjustments...')
    ddem_after = dem_coreg - ref_dem
    ddem_after_ss = ddem_after[ss_mask]
    ddem_after_ss_med, ddem_after_ss_nmad = np.median(ddem_after_ss), xdem.spatialstats.nmad(ddem_after_ss)

    # Subtract the median stable surfaces difference
    print('Removing median dDEM value over stable surfaces...')
    dem_coreg -= ddem_after_ss_med
    ddem_after -= ddem_after_ss_med
    # Re-calculate stable surface stats
    ddem_after_ss = ddem_after[ss_mask]
    ddem_after_ss_med, ddem_after_ss_nmad = np.median(ddem_after_ss), xdem.spatialstats.nmad(ddem_after_ss)

    # Save final DEM and dDEM
    dem_coreg.save(out_dem_fn)
    print('Final DEM saved to file:', out_dem_fn)
    ddem_after.save(out_ddem_fn)
    print('Final dDEM saved to file:', out_ddem_fn)

    # Plot results
    if plot_results:
        print('Plotting results...')
        fig, ax = plt.subplots(2, 2, figsize=(10,8), gridspec_kw=dict(height_ratios=[2,1]))
        ax = ax.flatten()
        bins = np.linspace(vmin, vmax, 100)
        ss_color = 'm'
        for i, (ddem, ddem_ss, ddem_ss_med, ddem_ss_nmad) in enumerate(zip([ddem_before, ddem_after], 
                                                                        [ddem_before_ss, ddem_after_ss],
                                                                        [ddem_before_ss_med, ddem_after_ss_med], 
                                                                        [ddem_before_ss_nmad, ddem_after_ss_nmad])):
            # Map
            ddem.plot(ax=ax[i], cmap='coolwarm_r', vmin=vmin, vmax=vmax)
            ax[i].set_xticks(ax[i].get_xticks())
            ax[i].set_xticklabels(np.divide(ax[i].get_xticks(), 1e3).astype(str))
            ax[i].set_yticks(ax[i].get_yticks())
            ax[i].set_yticklabels(np.divide(ax[i].get_yticks(), 1e3).astype(str))
            ax[i].set_xlabel('Easting [km]')
            # Histogram
            ax[i+2].hist(ddem.data.ravel(), bins=bins, color='k', alpha=0.8, label='All surfaces')
            ax2 = ax[i+2].twinx()
            ax2.hist(ddem_ss.data.ravel(), bins=bins, color=ss_color, alpha=0.8, label='Stable surfaces')
            ax2.spines['right'].set_color(ss_color)
            ax2.yaxis.label.set_color(ss_color)
            ax2.tick_params(colors=ss_color, which='both')
            ax[i+2].set_xlim(vmin, vmax)
            handles1, labels1 = ax[i+2].get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles, labels = handles1+handles2, labels1+labels2
            ax[i+2].legend(handles, labels, loc='best')
            ax[i+2].set_title(f'SS median = {np.round(ddem_ss_med, 3)} m\nSS MAD = {np.round(ddem_ss_nmad, 3)} m')
        # Add some labels
        ax[0].set_ylabel('Northing [km]')
        ax[0].set_title('dDEM')
        ax[1].set_title('dDEM adjusted')
        ax[2].set_ylabel('Counts')
        ax2.set_ylabel('Counts', color=ss_color)
        fig.tight_layout()
        plt.close()
        # Save figure
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
        print('Figure saved to file:', fig_fn)

    return out_dem_fn, out_ddem_fn

def plot_dem_ddem(dem_fn, ddem_fn, ss_mask_fn, out_fn, vmin=-5, vmax=5):
    # Load input files
    dem = xdem.DEM(dem_fn)
    ddem = xdem.DEM(ddem_fn)
    ss_mask = gu.Raster(ss_mask_fn)

    # Drop empty columns (to minimize white space on figure)
    def drop_empty_cols(raster):
        mask = raster.data.mask
        # Identify rows and columns that are completely NaN
        valid_rows = ~np.all(mask, axis=1)  # All-NaN rows
        valid_cols = ~np.all(mask, axis=0)  # All-NaN columns
        # Crop the data to remove empty rows and columns
        cropped_data = raster.data[valid_rows, :][:, valid_cols]
        # Create a new Raster object with the cropped data (keeping original metadata)
        cropped_raster = gu.Raster.from_array(cropped_data, transform=raster.transform, crs=raster.crs)
        return cropped_raster
    dem = drop_empty_cols(dem)
    ddem = drop_empty_cols(ddem)

    # Mask dDEM with stable surfaces
    ss_mask = ss_mask.reproject(ddem)
    ss_mask = (ss_mask==1)
    ddem_ss = ddem[ss_mask]

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(12,5))
    # Shaded relief
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem.data, vert_exag=5, dx=10, dy=10)
    ax[0].imshow(hs, cmap='Greys_r',
                extent=(dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top))
    im = ax[0].imshow(dem.data, cmap='terrain', alpha=0.8,
                    extent=(dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top))
    fig.colorbar(im, ax=ax[0], orientation='horizontal', shrink=0.8, label='Elevation [m]')
    ax[0].set_title('DEM shaded relief')
    # dDEM
    im = ax[1].imshow(ddem.data, cmap='coolwarm_r', vmin=-5, vmax=5,
                    extent=(ddem.bounds.left, ddem.bounds.right, ddem.bounds.bottom, ddem.bounds.top))
    fig.colorbar(im, ax=ax[1], orientation='horizontal', shrink=0.8, label='Difference [m]')
    ax[1].set_title('dDEM')
    # dDEM histogram
    ss_color = '#737373'
    bins = np.linspace(vmin, vmax, 100)
    ax[2].hist(ddem.data.ravel(), bins=bins, color='k', alpha=0.9, label='All surfaces')
    ax[2].set_ylabel('Counts')
    ax2 = ax[2].twinx()
    hist = ax2.hist(ddem_ss.data.ravel(), bins=bins, color=ss_color, alpha=0.9, label='Stable surfaces')
    ax2.set_ylim(0, np.nanmax(hist[0])*1.4)
    ax2.spines['right'].set_color(ss_color)
    ax2.yaxis.label.set_color(ss_color)
    ax2.tick_params(colors=ss_color, which='both')
    # Add legend
    handles1, labels1 = ax[2].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles, labels = handles1+handles2, labels1+labels2
    ax[2].legend(handles, labels, loc='best')
    ax[2].set_xlim(vmin, vmax)
    # Add scalebars, remove axes
    for axis in ax[0:-1]:
        scalebar = AnchoredSizeBar(axis.transData,
                                1e3, '1 km', 'lower left', 
                                pad=0.1,
                                color='k',
                                frameon=False,
                                size_vertical=1,
                                fontproperties=fm.FontProperties(size=14))
        axis.add_artist(scalebar)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.tight_layout()

    # Save figure
    fig.savefig(out_fn, dpi=300, bbox_inches='tight')
    print('Figure saved to file:', out_fn)
    plt.close()

    return
