#! /usr/bin/python

import math
import geopandas as gpd
import geedim as gd
import os
import pyproj
import subprocess
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
import ee
import json
from shapely.geometry import Polygon

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
        epsg_code = 'EPSG:326' + utm_band
        return epsg_code
    epsg_code = 'EPSG:327' + utm_band
    return epsg_code


def create_bbox_from_meta(meta_fns, buffer):
    # Iterate over metadata files
    xmin, xmax, ymin, ymax = 1e10, -1e10, 1e10, -1e10
    for meta_fn in meta_fns[0:1]:
        meta = json.load(open(meta_fn))
        bounds = np.array(meta['geometry']['coordinates'])[0]
        xbounds, ybounds = bounds[:,0], bounds[:,1]
        xmin_im, xmax_im, ymin_im, ymax_im = np.min(xbounds), np.max(xbounds), np.min(ybounds), np.max(ybounds)
        if xmin_im < xmin:
            xmin = xmin_im
        if xmax_im > xmax:
            xmax = xmax_im
        if ymin_im < ymin:
            ymin = ymin_im
        if ymax_im > ymax:
            ymax = ymax_im

    # Create bounding geometry and buffer
    bounds_poly = Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]])
    bounds_gdf = gpd.GeoDataFrame(geometry=[bounds_poly], crs='EPSG:4326')
    epsg_utm = convert_wgs_to_utm(bounds_poly.centroid.coords.xy[0][0], bounds_poly.centroid.coords.xy[1][0])
    print(f'Optimal UTM zone = {epsg_utm}')
    bounds_utm_gdf = bounds_gdf.to_crs(epsg_utm)
    bounds_utm_buffer_gdf = bounds_utm_gdf.buffer(buffer)
    bounds_buffer_gdf = bounds_utm_buffer_gdf.to_crs('EPSG:4326')

    # Plot
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.plot(*bounds_gdf.geometry[0].exterior.coords.xy, '-k', label='Image bounds')
    ax.plot(*bounds_buffer_gdf.geometry[0].exterior.coords.xy, '-m', label='Clipping geometry')
    ax.legend(loc='upper right')
    plt.show()

    return bounds_buffer_gdf, epsg_utm


def query_gee_for_copdem(aoi, out_fn=None, crs='EPSG:4326', scale=30):
    """
    Query GEE for the COPDEM, clip to the AOI, and return as xarray.Dataset.

    Parameters
    ----------
    aoi: geopandas.geodataframe.GeoDataFrame
        area of interest used for clipping the DEM 
    out_fn: str
        file name for output DEM
    crs: str
        Coordinate Reference System of output DEM

    Returns
    ----------
    dem_ds: xarray.Dataset
        dataset of elevations over the AOI
    """

    # Reproject AOI to EPSG:4326 if necessary
    aoi_wgs = aoi.to_crs('EPSG:4326')

    # Reformat AOI for querying and clipping DEM
    region = {'type': 'Polygon',
              'coordinates': [[
                  [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.miny[0]],
                  [aoi_wgs.geometry.bounds.maxx[0], aoi_wgs.geometry.bounds.miny[0]],
                  [aoi_wgs.geometry.bounds.maxx[0], aoi_wgs.geometry.bounds.maxy[0]],
                  [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.maxy[0]],
                  [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.miny[0]]
              ]]
              }

    # Query GEE for DEM
    dem_col = gd.MaskedCollection.from_name("COPERNICUS/DEM/GLO30").search(start_date='1900-01-01',
                                                                           end_date='2025-01-01',
                                                                           region=region)
    # Mosaic all images over the region
    dem_im = dem_col.composite(method='mosaic')

    # Download DEM 
    if not os.path.exists(out_fn):
        dem_im.download(out_fn, region=region, scale=scale, bands=['DEM'], crs=crs)

    # Reproject from the EGM96 geoid to the WGS84 ellipsoid
    s_crs = pyproj.CRS.from_epsg(int(crs.split(':')[1]))
    s_proj_string = s_crs.to_proj4() + " +vunits=m +nodefs"
    t_proj_string = s_proj_string 
    s_proj_string += f' +geoidgrids=egm96_15.gtx'
    out_ellip_fn = out_fn.replace('.tif', '_WGS84_ellipsoid.tif')
    cmd = f'''gdalwarp -s_srs "{s_proj_string}" -t_srs "{t_proj_string}" {out_fn} {out_ellip_fn}'''
    output = subprocess.run(cmd, capture_output=True, shell=True)
    print(output)
    print('DEM reprojected to the WGS84 ellipsoid and saved to file:', out_ellip_fn)

    # Simplify CRS to UTM Zone without ellipsoidal height
    out_ellip_utm_fn = out_ellip_fn.replace('.tif', '_UTM.tif')
    cmd = f'''gdalwarp -s_srs "{t_proj_string}" -t_srs "+proj=utm +zone=7 +datum=WGS84" {out_ellip_fn} {out_ellip_utm_fn}'''
    output = subprocess.run(cmd, capture_output=True, shell=True)
    print(output)
    print('DEM reprojected to UTM Zone 7N and saved to file:', out_ellip_utm_fn)

    # Fill holes
    out_ellip_utm_filled_fn = out_ellip_utm_fn.replace('.tif', '_filled.tif')
    cmd = f"gdal_fillnodata {out_ellip_utm_fn} {out_ellip_utm_filled_fn}"
    output = subprocess.run(cmd, capture_output=True, shell=True)
    print(output)
    print('DEM with holes filled saved to file:', out_ellip_utm_filled_fn)

    # Open DEM as xarray.DataArray and plot
    dem = rxr.open_rasterio(out_ellip_utm_filled_fn).squeeze()
    fig, ax = plt.subplots()
    dem_im = ax.imshow(dem.data, cmap='terrain',
              extent=(np.min(dem.x.data), np.max(dem.x.data), 
                      np.min(dem.y.data), np.max(dem.y.data)))
    fig.colorbar(dem_im, ax=ax, label='Elevation [m]')
    ax.set_title(os.path.basename(out_ellip_utm_filled_fn))
    plt.show()
    
    return dem

def query_gee_for_arcticdem(aoi, out_fn=None, crs='EPSG:4326', scale=30):
    """
    Query GEE for the ArcticDEM, clip to the AOI, and return as xarray.Dataset.

    Parameters
    ----------
    aoi: geopandas.geodataframe.GeoDataFrame
        area of interest used for clipping the DEM 
    out_fn: str
        file name for output DEM
    crs: str
        Coordinate Reference System of output DEM

    Returns
    ----------
    dem_ds: xarray.Dataset
        dataset of elevations over the AOI
    """

    # Reproject AOI to EPSG:4326 if necessary
    aoi_wgs = aoi.to_crs('EPSG:4326')

    # Reformat AOI for querying and clipping DEM
    region = ee.Geometry.Polygon(list(zip(aoi.geometry[0].exterior.coords.xy[0], aoi.geometry[0].exterior.coords.xy[1])))

    # Query GEE for DEM
    dem_im = gd.MaskedImage.from_id("UMN/PGC/ArcticDEM/V3/2m_mosaic")

    # Download DEM 
    if not os.path.exists(out_fn):
        dem_im.download(out_fn, region=region, scale=scale, bands=['elevation'], crs=crs)

    # Fill holes
    out_filled_fn = out_fn.replace('.tif', '_filled.tif')
    cmd = f"gdal_fillnodata -md 2000 {out_fn} {out_filled_fn}"
    output = subprocess.run(cmd, capture_output=True, shell=True)
    print(output)

    # Open DEM as xarray.DataArray and plot
    dem = rxr.open_rasterio(out_filled_fn).squeeze()
    fig, ax = plt.subplots()
    dem_im = ax.imshow(dem.data, cmap='terrain',
              extent=(np.min(dem.x.data), np.max(dem.x.data), 
                      np.min(dem.y.data), np.max(dem.y.data)))
    fig.colorbar(dem_im, ax=ax, label='Elevation [m]')
    ax.set_title(os.path.basename(out_filled_fn))
    plt.show()
    
    return dem

