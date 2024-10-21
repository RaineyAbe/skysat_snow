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

    # Reference DEM: regrid to desired CRS and output resolution
    refdem_adj_fn = os.path.join(out_dir, f"refdem_{out_res}m_{crs_utm.replace(':', '')}.tif")
    if not os.path.exists(refdem_adj_fn):
        print('Reprojecting, cropping, and regridding reference DEM...')
        bounds_buffered = list(np.array(bounds_buffered).astype(str))
        cmd = ['gdalwarp', '-t_srs', crs_utm, '-tr', str(out_res), str(out_res), '-te'] + bounds_buffered + [refdem_fn, refdem_adj_fn]
        output = subprocess.run(cmd, shell=False, capture_output=True)
        print(output)
    else:
        print('Adjusted reference DEM already exists in file, skipping.')

    return multispec_mosaic_fn, roads_adj_fn, refdem_adj_fn


def preprocess_point_clouds(pc_dir, ref_dem_fn, resolution, out_dir):
    """
    Merge, filter, and rasterize point clouds using PDAL and GDAL.

    Parameters
    ----------
    pc_dir: str or Path
        path to folder containing .laz or .las files
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
    pc_merged_las_fn = os.path.join(out_dir, 'pc_merged.laz')
    pc_merged_tif_fn = os.path.join(out_dir, 'pc_merged.tif')
    json1_fn = os.path.join(out_dir, 'las2tif.json')
    json2_fn = os.path.join(out_dir, 'las2unaligned.json')
    pc_filtered_las_fn = os.path.join(out_dir, 'pc_merged_filtered.laz')
    pc_filtered_tif_fn = os.path.join(out_dir, 'pc_merged_filtered.tif')
    fig_fn = os.path.join(out_dir, 'pc_filtered.png')

    ### Merge point clouds ###
    if not os.path.exists(pc_merged_las_fn):
        print('Merging point clouds...')
        # Grab input files
        laz_fns = sorted(glob.glob(os.path.join(pc_dir, '*.laz')) + glob.glob(os.path.join(pc_dir, '*.las')))
        # Construct command
        cmd = ['pdal', 'merge'] + laz_fns + [pc_merged_las_fn]
        # Run command
        output = subprocess.run(cmd, shell=False, capture_output=True)
        print(output)
    else:
        print('Merged point cloud already exists, skipping.')

    ### Rasterize merged point cloud for reference ###
    # Create PDAL pipeline JSON file
    if not os.path.exists(json1_fn):
        reader = {"type": "readers.las", "filename": pc_merged_las_fn}
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
        print('Rasterizing merged point cloud...')
        cmd = ['pdal', 'pipeline', json1_fn]
        output = subprocess.run(cmd, shell=False, capture_output=True)
        print(output)
    else:
        print('Rasterized point cloud already exists, skipping.')


    ### Filter point cloud ###
    # Create pdal pipeline json
    if not os.path.exists(json2_fn):
        reader = {"type": "readers.las", "filename": pc_merged_las_fn}
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

    return pc_filtered_las_fn, pc_filtered_tif_fn


def construct_land_cover_masks(multispec_mosaic_fn, roads_vector_fn, out_dir, roads_buffer=3, ndvi_threshold=0.5, ndsi_threshold=0.1):
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
        trees_mask.rio.to_raster(trees_mask_fn)
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
        snow_mask.rio.to_raster(snow_mask_fn)
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
        roads_mask.rio.to_raster(roads_mask_fn)
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
        ss_mask.rio.to_raster(ss_mask_fn)
        print('Stable surfaces mask saved to file:', ss_mask_fn)
    else: 
        print('Stable surfaces mask exists in directory, skipping.')

    # Plot land cover masks
    if not os.path.exists(fig_fn):
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
        plt.show()

        # Save figure
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
        print('Figure saved to file:', fig_fn)
    else:
        print('Masks figure exists in directory, skipping plotting.')

    return multispec_mosaic_fn, trees_mask_fn, snow_mask_fn, roads_mask_fn, ss_mask_fn

    
def deramp(ref_dem_fn, tba_dem_fn, ss_mask_fn, out_dir, poly_order=2, vmin=-10, vmax=10):
    # Define output file names
    dem_deramped_fn = os.path.join(out_dir, os.path.basename(tba_dem_fn).replace('.tif', '_deramped.tif'))
    fig_fn = os.path.join(out_dir, 'deramping.png')

    # Check if deramped DEM already exists
    if os.path.exists(dem_deramped_fn):
        print('Deramped DEM already exists, skipping.')
        return dem_deramped_fn

    # Load input files
    print('Loading input files')
    tba_dem = xdem.DEM(tba_dem_fn)
    ref_dem = xdem.DEM(ref_dem_fn)
    ss_mask = gu.Raster(ss_mask_fn, load_data=True, nodata=-9999)

    # Reproject tba_dem and ss_mask to ref_dem grid
    tba_dem = tba_dem.reproject(ref_dem)
    ss_mask = ss_mask.reproject(ref_dem)
    ss_mask = (ss_mask==1) # Convert to boolean mask

    # Calculate difference before deramping
    print('Calculating difference before deramping...')
    diff_before = tba_dem - ref_dem

    # Fit deramper
    print('Fitting and applying deramper to DEM...')
    deramper = xdem.coreg.Deramp(poly_order=poly_order)
    deramper.fit(ref_dem, tba_dem, inlier_mask=ss_mask)
    # Apply the deramper to the tba_dem
    dem_deramped = deramper.apply(tba_dem)

    # Save results
    dem_deramped.save(dem_deramped_fn)
    print('Deramped DEM saved to file:', dem_deramped_fn)

    # Calculate difference after
    print('Calculating difference after deramping...')
    diff_after = dem_deramped - ref_dem

    # Plot deramping results
    print('Plotting results...')
    fig, ax = plt.subplots(2, 2, figsize=(12,10), gridspec_kw=dict(height_ratios=[2,1]))
    ax = ax.flatten()
    diff_before.plot(cmap='coolwarm_r', vmin=vmin, vmax=vmax, ax=ax[0])
    ax[0].set_title('dDEM')
    ax[0].set_xlabel('Easting [m]')
    ax[0].set_ylabel('Northing [m]')
    diff_after.plot(cmap='coolwarm_r', vmin=vmin, vmax=vmax, ax=ax[1])
    ax[1].set_title('Deramped dDEM')
    ax[1].set_xlabel('Easting [m]')
    bins = np.linspace(vmin, vmax, 100)
    ax[2].hist(diff_before.data.ravel(), bins=bins, color='gray')
    ax[2].set_xlabel('Difference [m]')
    ax[2].set_xlim(vmin, vmax)
    ax[3].hist(diff_after.data.ravel(), bins=bins, color='gray')
    ax[3].set_xlabel('Difference [m]')
    ax[3].set_xlim(vmin, vmax)
    plt.show()

    # Save figure
    fig.savefig(fig_fn, dpi=250, bbox_inches='tight')
    print('Figure saved to file:', fig_fn)

    return dem_deramped_fn


def create_coreg_object(coreg_name):
    if type(coreg_name) == list:
        try:
            if coreg_name[0]=='BiasCorr':
                coreg_class = getattr(xdem.coreg, coreg_name[0])(bias_vars=["elevation", "slope", "aspect"])
            else:
                coreg_class = getattr(xdem.coreg, coreg_name[0])()
            for i in range(1, len(coreg_name)):
                if coreg_name[i]=='BiasCorr':
                    coreg_class += getattr(xdem.coreg, coreg_name[i])(bias_vars=["elevation", "slope", "aspect"])
                else:
                    coreg_class += getattr(xdem.coreg, coreg_name[i])()
            return coreg_class
        except AttributeError:
            raise ValueError(f"Coregistration method '{coreg_name}' not found.")
    elif type(coreg_name) == str:
        try:
            if coreg_name=='BiasCorr':
                coreg_class = getattr(xdem.coreg, coreg_name)(bias_vars=["elevation", "slope", "aspect"])
            else:
                coreg_class = getattr(xdem.coreg, coreg_name)()
            return coreg_class
        except AttributeError:
            raise ValueError(f"Coregistration method '{coreg_name}' not found.")
    else:
        print('coreg_method format not recognized, exiting...')
        return None


def calculate_stable_surface_stats(diff_dem, ss_mask):
    ss_masked_data = np.where(ss_mask.data==1, diff_dem.data, -9999)  
    diff_dem_ss = gu.Raster.from_array(
        ss_masked_data,
        transform=diff_dem.transform,
        crs=diff_dem.crs,
        nodata=-9999)
    diff_dem_ss_median = np.nanmedian(gu.raster.get_array_and_mask(diff_dem_ss)[0])
    diff_dem_ss_nmad = xdem.spatialstats.nmad(diff_dem)
    return diff_dem_ss, diff_dem_ss_median, diff_dem_ss_nmad


def plot_coreg_dh_results(dh_before, dh_before_ss, dh_before_ss_med, dh_before_ss_nmad,
                          dh_after, dh_after_ss, dh_after_ss_med, dh_after_ss_nmad,
                          dh_after_ss_adj, dh_after_ss_adj_ss, dh_after_ss_adj_ss_med, dh_after_ss_adj_ss_nmad,
                          vmin=-10, vmax=10):
    fig, ax = plt.subplots(3, 2, figsize=(10,16))
    dhs = [dh_before, dh_after, dh_after_ss_adj]
    titles = ['Difference before coreg.', 'Difference after coreg.', 'Difference after coreg. - median SS diff.']
    dhs_ss = [dh_before_ss, dh_after_ss, dh_after_ss_adj_ss]
    ss_meds = [dh_before_ss_med, dh_after_ss_med, dh_after_ss_adj_ss_med]
    ss_nmads = [dh_before_ss_nmad, dh_after_ss_nmad, dh_after_ss_adj_ss_nmad]
    bins = np.linspace(vmin, vmax, num=100)
    for i in range(len(dhs)):
        # plot dh
        dhs[i].plot(cmap="coolwarm_r", ax=ax[i,0], vmin=vmin, vmax=vmax)
        ax[i,0].set_title(f'{titles[i]} \nSS median = {np.round(ss_meds[i], 3)}, SS NMAD = {np.round(ss_nmads[i], 3)}')
        # Adjust map units to km
        ax[i,0].set_xticks(ax[i,0].get_xticks())
        ax[i,0].set_xticklabels(np.divide(ax[i,0].get_xticks(), 1e3).astype(str))
        ax[i,0].set_yticks(ax[i,0].get_yticks())
        ax[i,0].set_yticklabels(np.divide(ax[i,0].get_yticks(), 1e3).astype(str))
        ax[i,0].set_xlabel('Easting [km]')
        ax[i,0].set_ylabel('Northing [km]')
         # plot histograms
        ax[i,1].hist(np.ravel(dhs[i].data), bins=bins, color='grey', alpha=0.8, label='All surfaces')
        ax[i,1].legend(loc='upper left')
        ax[i,1].set_xlabel('Differences [m]')
        ax[i,1].set_ylabel('Counts')
        ax[i,1].set_xlim(vmin,vmax)
        ax2 = ax[i,1].twinx()
        ax2.hist(np.ravel(dhs_ss[i].data), bins=bins, color='m', alpha=0.8, label='Stable surfaces')
        ax2.legend(loc='upper right')
        ax2.spines['right'].set_color('m')
        ax2.set_yticks(ax2.get_yticks())
        ax2.set_yticklabels(ax2.get_yticklabels(), color='m')
    fig.tight_layout()
    plt.show()
    return fig
                                           
def coregister_difference_dems(ref_dem_fn=None, source_dem_fn=None, ss_mask_fn=None, out_dir=None, 
                               coreg_method='NuthKaab', coreg_stable_only=False, vmin=-10, vmax=10):
    # Define output file names
    coreg_meta_fn = os.path.join(out_dir, 'coregistration_fit.json')
    dem_coreg_fn = os.path.join(out_dir, 'dem_coregistered.tif')
    ddem_fn = os.path.join(out_dir, 'ddem.tif')
    fig1_fn = os.path.join(out_dir, 'ddem_results.png')

    # Check if output files already exist
    if os.path.exists(dem_coreg_fn):
        print('Coregistered DEM already exists in directory, skipping.')
        return dem_coreg_fn

    # Load input files
    ref_dem = xdem.DEM(gu.Raster(ref_dem_fn, load_data=True, bands=1))
    tba_dem = xdem.DEM(gu.Raster(source_dem_fn, load_data=True, bands=1))
    ss_mask = gu.Raster(ss_mask_fn, load_data=True, nodata=-9999)

    # Reproject source DEM and stable surfaces mask to reference DEM grid
    tba_dem = tba_dem.reproject(ref_dem)
    ss_mask = ss_mask.reproject(ref_dem)

    # Calculate differences before coregistration
    print('Calculating differences before coregistration...')
    diff_before = tba_dem - ref_dem
    # Calculate stable surface stats
    diff_before_ss, diff_before_ss_median, diff_before_ss_nmad = calculate_stable_surface_stats(diff_before, ss_mask)

    # Create and fit the coregistration object
    print('Coregistering source DEM to reference DEM...')
    coreg_obj = create_coreg_object(coreg_method)
    if coreg_stable_only:
        ss_mask = (ss_mask == 1)
        coreg_obj.fit(ref_dem, tba_dem, ss_mask)   
    else:
        coreg_obj.fit(ref_dem, tba_dem)
    # Save the coregistration fit metadata
    meta = coreg_obj.meta
    with open(coreg_meta_fn, 'w') as f:
        json.dump(meta, f)
    print('Coregistration fit saved to file:', coreg_meta_fn)
        
    # Apply the coregistration object to the source DEM
    aligned_dem = coreg_obj.apply(tba_dem)

    # Calculate differences after coregistration
    print('Calculating differences after coregistration...')
    diff_after = aligned_dem - ref_dem

    # Calculate stable surfaces stats
    diff_after_ss, diff_after_ss_median, diff_after_ss_nmad = calculate_stable_surface_stats(diff_after, ss_mask)

    # Subtract the median difference over stable surfaces
    aligned_dem = aligned_dem - diff_after_ss_median
    diff_after_ss_adj = diff_after - diff_after_ss_median

    # Save coregistered DEM, stable surfaces, and dDEM to file
    aligned_dem.save(dem_coreg_fn)
    print('Coregistered DEM saved to file:', dem_coreg_fn)
    diff_after_ss_adj.save(ddem_fn)
    print('dDEM saved to file:', ddem_fn)

    # Re-calculate stable surfaces stats
    diff_after_ss_adj_ss, diff_after_ss_adj_ss_median, diff_after_ss_adj_ss_nmad = calculate_stable_surface_stats(diff_after_ss_adj, ss_mask)

    # Plot results
    print('Plotting dDEM results...')
    fig1 = plot_coreg_dh_results(diff_before, diff_before_ss, diff_before_ss_median, diff_before_ss_nmad,
                                 diff_after, diff_after_ss, diff_after_ss_median, diff_after_ss_nmad,
                                 diff_after_ss_adj, diff_after_ss_adj_ss, diff_after_ss_adj_ss_median, diff_after_ss_adj_ss_nmad,
                                 vmin=vmin, vmax=vmax)
    fig1.savefig(fig1_fn)
    print('Figure saved to file:', fig1_fn)
    
    return dem_coreg_fn


def mask_trees_dem(dem_fn, trees_mask_fn, out_dir):
    # Define output file
    dem_masked_fn = os.path.join(out_dir, os.path.basename(dem_fn).replace('.tif', '_maskedtrees.tif'))
    if not os.path.exists(dem_masked_fn):
        # Load input files
        dem = rxr.open_rasterio(dem_fn).squeeze()
        trees_mask = rxr.open_rasterio(trees_mask_fn).squeeze()
        # Interpolate trees to DEM grid
        trees_mask = trees_mask.sel(x=dem.x, y=dem.y, method='nearest')
        # Mask trees
        dem_masked = xr.where((trees_mask.data==1) | (trees_mask.data==-9999), np.nan, dem)
        # Save to file
        dem_masked = dem_masked.rio.write_crs(dem.rio.crs)
        dem_masked.rio.to_raster(dem_masked_fn)
        print('DEM with trees masked saved to file:', dem_masked_fn)

    else:
        print('DEM with trees masked already exists, skipping.')

    return dem_masked_fn
