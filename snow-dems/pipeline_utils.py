#! usr/bin/python
# Functions for running the SkySat DEM alignment and differencing pipeline
# Rainey Aberle
# 2024

import os
import xdem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geoutils as gu
import rioxarray as rxr
import xarray as xr
from skimage.filters import threshold_otsu
import matplotlib


def vector_to_mask(vector_fn, ref_raster_fn, out_fn):
    """
    Converts a vector to a mask using the input reference raster grid.

    Args:
        vector_fn (str): file name of input vector
        ref_raster_fn (str): file name of reference raster
        out_fn (str): file name of the output mask

    Returns:
        None
    """
    # Check if mask already exists in directory
    if os.path.exists(out_fn):
        print('Mask already exists in file, skipping.')
    else:
        ref_raster = xdem.DEM(ref_raster_fn)
        vector = gu.Vector(vector_fn)
        raster = vector.rasterize(ref_raster)
        raster.data.mask = ref_raster.data.mask
        raster = raster.astype(np.int16)
        raster.set_nodata(-9999)
        raster.save(out_fn)
        print('Mask saved to file:', out_fn)
    return


def construct_stable_surfaces_mask(ortho_fn, out_fn):
    """
    Construct a stable surface mask by calculating and applying an Otsu threshold to the input orthomosaic.
    Args:
        ortho_fn (str):
        out_fn (str):

    Returns:
        None
    """
    # Check if stable surfaces mask already exists
    if os.path.exists(out_fn):
        print('Stable surfaces mask already exists in file, skipping.')
    else:
        # Open orthomosaic
        ortho = rxr.open_rasterio(ortho_fn)
        # Save CRS for later
        crs = ortho.rio.crs
        # Grab data
        image = ortho.data[0]
        # Create no-data mask
        nodata_mask = image == 0
        # Calculate and apply Otsu threshold
        otsu_thresh = threshold_otsu(image)
        ss_mask = (image < otsu_thresh).astype(float)
        # Apply nodata mask
        ss_mask[nodata_mask] = np.nan
        image[nodata_mask] = np.nan

        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        ax = axes.ravel()
        im = ax[0].imshow(image, cmap='Greys',
                          extent=(np.min(ortho.x.data) / 1e3, np.max(ortho.x.data) / 1e3,
                                  np.min(ortho.y.data) / 1e3, np.max(ortho.y.data) / 1e3))
        fig.colorbar(im, ax=ax[0], orientation='horizontal', shrink=0.9)
        ax[0].set_title('Original')
        ax[1].hist(image.ravel(), bins=100, color='grey')
        ax[1].set_title('Histogram')
        ax[1].axvline(otsu_thresh, color='r')
        ax[1].text(otsu_thresh + (np.nanmax(image) - np.nanmin(image)) * 0.1,
                   ax[1].get_ylim()[0] + (ax[1].get_ylim()[1] - ax[1].get_ylim()[0]) * 0.9,
                   f'Otsu threshold = \n{np.round(otsu_thresh, 2)}', color='r')
        cmap_binary = matplotlib.colors.ListedColormap(['w', 'k'])
        im = ax[2].imshow(ss_mask, cmap=cmap_binary,
                          extent=(np.min(ortho.x.data) / 1e3, np.max(ortho.x.data) / 1e3,
                                  np.min(ortho.y.data) / 1e3, np.max(ortho.y.data) / 1e3))
        cbar = fig.colorbar(im, ax=ax[2], orientation='horizontal', shrink=0.9, ticks=[0.25, 0.75])
        cbar.ax.set_xticklabels(['unstable', 'stable'])
        ax[2].set_title('Stable surfaces mask')
        plt.close()

        # Save results to file
        ss_mask[np.isnan(ss_mask)] = -9999
        ss_mask = ss_mask.astype(int)
        ss_mask_xr = xr.DataArray(data=ss_mask,
                                  coords=dict(y=ortho.y, x=ortho.x),
                                  attrs=dict(
                                      Description='Stable surfaces mask generated using Otsu thresholding of the '
                                                  'input image. 1 = stable, 0 = unstable',
                                      InputImage=os.path.basename(ortho_fn),
                                      OtsuThreshold=otsu_thresh,
                                      _FillValue=-9999)
                                  )
        ss_mask_xr = ss_mask_xr.rio.write_crs(crs)
        ss_mask_xr.rio.to_raster(out_fn)
        print('Stable surfaces mask saved to file:', out_fn)
        fig_fn = os.path.join(os.path.splitext(out_fn)[0], '.png')
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
        print('Stable surfaces figure saved to file:', fig_fn)

    return


def create_coregistration_object(coreg_name):
    if type(coreg_name) == list:
        try:
            if coreg_name[0] == 'BiasCorr':
                coreg_class = getattr(xdem.coreg, coreg_name[0])(bias_vars=["elevation", "slope", "aspect"])
            else:
                coreg_class = getattr(xdem.coreg, coreg_name[0])()
            for i in range(1, len(coreg_name)):
                if coreg_name[i] == 'BiasCorr':
                    coreg_class += getattr(xdem.coreg, coreg_name[i])(bias_vars=["elevation", "slope", "aspect"])
                else:
                    coreg_class += getattr(xdem.coreg, coreg_name[i])()
            return coreg_class
        except AttributeError:
            raise ValueError(f"Coregistration method '{coreg_name}' not found.")
    elif type(coreg_name) == str:
        try:
            if coreg_name == 'BiasCorr':
                coreg_class = getattr(xdem.coreg, coreg_name)(bias_vars=["elevation", "slope", "aspect"])
            else:
                coreg_class = getattr(xdem.coreg, coreg_name)()
            return coreg_class
        except AttributeError:
            raise ValueError(f"Coregistration method '{coreg_name}' not found.")
    else:
        print('coreg_method format not recognized, exiting...')
        return None


def differences_vs_slope_aspect(dem, diffs):
    # Calculate slope and aspect from DEM
    slope = xdem.terrain.slope(dem)
    aspect = xdem.terrain.aspect(dem)

    # Compile differences, slopes, and aspects in a dataframe
    df = pd.DataFrame({'diff': np.ravel(diffs.data),
                       'elev': np.ravel(dem.data),
                       'slope': np.ravel(slope.data),
                       'aspect': np.ravel(aspect.data)})
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create bins for elev, slope, and aspect
    bin_min = np.floor(dem.data.min() / 5) * 5
    bins = np.arange(bin_min, dem.data.max() + 50, step=50)
    df['elev_bin'] = pd.cut(df['elev'], bins=bins)
    df['slope_bin'] = pd.cut(df['slope'], bins=np.arange(0, 41, step=2.5))
    df['aspect_bin'] = pd.cut(df['aspect'], bins=np.arange(0, 361, step=22.5))

    # Plot
    fig2, ax = plt.subplots(3, 1, figsize=(8, 14))
    # elev
    df.boxplot(column='diff', by='elev_bin', showfliers=False, patch_artist=True, ax=ax[0],
               boxprops=dict(color='k'), medianprops=dict(color='w', linewidth=1.5), whiskerprops=dict(color='k'))
    ax[0].set_title('')
    ax[0].set_xlabel('Elevation range [m]')
    # slope
    df.boxplot(column='diff', by='slope_bin', showfliers=False, patch_artist=True, ax=ax[1],
               boxprops=dict(color='k'), medianprops=dict(color='w', linewidth=1.5), whiskerprops=dict(color='k'))
    ax[1].set_title('')
    ax[1].set_xlabel('Slope range [degrees]')
    # aspect
    df.boxplot(column='diff', by='aspect_bin', showfliers=False, patch_artist=True, ax=ax[2],
               boxprops=dict(color='k'), medianprops=dict(color='w', linewidth=1.5), whiskerprops=dict(color='k'))
    ax[2].set_title('')
    ax[2].set_xlabel('Aspect range [degrees from North]')
    for axis in ax:
        axis.set_ylabel('Differences [m]')
    fig2.suptitle('')
    fig2.tight_layout()

    return fig2


def calculate_stable_surface_stats(diff_dem, ss_mask):
    ss_masked_data = np.where(ss_mask.data == 1, diff_dem.data, -9999)
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
    fig, ax = plt.subplots(3, 2, figsize=(10, 16))
    dhs = [dh_before, dh_after, dh_after_ss_adj]
    titles = ['Difference before coreg.', 'Difference after coreg.', 'Difference after coreg. - median SS diff.']
    dhs_ss = [dh_before_ss, dh_after_ss, dh_after_ss_adj_ss]
    ss_meds = [dh_before_ss_med, dh_after_ss_med, dh_after_ss_adj_ss_med]
    ss_nmads = [dh_before_ss_nmad, dh_after_ss_nmad, dh_after_ss_adj_ss_nmad]
    for i in range(len(dhs)):
        # plot dh
        dhs[i].plot(cmap="coolwarm_r", ax=ax[i, 0], vmin=vmin, vmax=vmax)
        ax[i, 0].set_title(f'{titles[i]} \nSS median = {np.round(ss_meds[i], 3)}, SS NMAD = {np.round(ss_nmads[i], 3)}')
        # Adjust map units to km
        ax[i, 0].set_xticks(ax[i, 0].get_xticks())
        ax[i, 0].set_xticklabels(np.divide(ax[i, 0].get_xticks(), 1e3).astype(str))
        ax[i, 0].set_yticks(ax[i, 0].get_yticks())
        ax[i, 0].set_yticklabels(np.divide(ax[i, 0].get_yticks(), 1e3).astype(str))
        ax[i, 0].set_xlabel('Easting [km]')
        ax[i, 0].set_ylabel('Northing [km]')
        # plot histograms
        ax[i, 1].hist(np.ravel(dhs[i].data), bins=50, color='grey', alpha=0.8, label='All surfaces')
        ax[i, 1].legend(loc='upper left')
        ax[i, 1].set_xlabel('Differences [m]')
        ax[i, 1].set_ylabel('Counts')
        ax[i, 1].set_xlim(vmin, vmax)
        ax2 = ax[i, 1].twinx()
        ax2.hist(np.ravel(dhs_ss[i].data), bins=50, color='m', alpha=0.8, label='Stable surfaces')
        ax2.legend(loc='upper right')
        ax2.spines['right'].set_color('m')
        ax2.set_yticks(ax2.get_yticks())
        ax2.set_yticklabels(ax2.get_yticklabels(), color='m')
    fig.tight_layout()
    plt.close()
    return fig


def coregister_difference_dems(ref_dem_fn=None, source_dem_fn=None, ss_mask_fn=None, out_dir=None,
                               coreg_method='NuthKaab', coreg_stable_only=False, plot_terrain_results=True,
                               vmin=-10, vmax=10):
    # Define output file names
    coreg_meta_fn = os.path.join(out_dir, 'coregistration_fit.json')
    dem_coreg_fn = os.path.join(out_dir, 'dem_coregistered.tif')
    ss_mask_shift_fn = os.path.join(out_dir, 'stable_surfaces_coregistered.tif')
    ddem_fn = os.path.join(out_dir, 'ddem.tif')
    fig1_fn = os.path.join(out_dir, 'ddem_results.png')
    fig2_fn = os.path.join(out_dir, 'ddem_terrain_boxplots.png')

    # Load input files
    ref_dem = xdem.DEM(gu.Raster(ref_dem_fn, load_data=True, bands=1))
    tba_dem = xdem.DEM(gu.Raster(source_dem_fn, load_data=True, bands=1))
    ss_mask = gu.Raster(ss_mask_fn, load_data=True, nodata=-9999)
    ss_mask = (ss_mask == 1)  # Convert to boolean mask

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
        coreg_obj.fit(ref_dem, tba_dem, ss_mask)
    else:
        coreg_obj.fit(ref_dem, tba_dem)
    # Save the fit coregistration metadata
    meta = coreg_obj.meta
    with open(coreg_meta_fn, 'w') as f:
        json.dump(meta, f)
    print('Coregistration fit saved to file:', coreg_meta_fn)

    # Apply the coregistration object to the source DEM
    aligned_dem = coreg_obj.apply(tba_dem)

    # Apply horizontal components of the coregistration object to the stable surfaces mask if applicable
    if 'shift_x' in meta.keys():
        dx, dy = meta['shift_x'], meta['shift_y']
        ss_mask_affine = ss_mask.transform
        new_affine = ss_mask_affine * Affine.translation(dx, dy)
        ss_mask_shift = ss_mask.copy()
        ss_mask_shift.transform = new_affine
        ss_mask = ss_mask_shift

    # Save coregistered DEM and stable surfaces to file
    aligned_dem.save(dem_coreg_fn)
    print('Coregistered DEM saved to file:', dem_coreg_fn)
    ss_mask.save(ss_mask_shift_fn)
    print('Coregistered stable surfaces mask saved to file:', ss_mask_shift_fn)

    # Calculate differences after coregistration
    print('Calculating differences after coregistration...')
    diff_after = aligned_dem - ref_dem

    # Calculate stable surfaces stats
    diff_after_ss, diff_after_ss_median, diff_after_ss_nmad = calculate_stable_surface_stats(diff_after, ss_mask)

    # Subtract the median difference over stable surfaces
    diff_after_ss_adj = diff_after - diff_after_ss_median

    # Save dDEM to file
    diff_after_ss_adj.save(ddem_fn)
    print('dDEM saved to file:', ddem_fn)

    # Re-calculate stable surfaces stats
    diff_after_ss_adj_ss, diff_after_ss_adj_ss_median, diff_after_ss_adj_ss_nmad = calculate_stable_surface_stats(
        diff_after_ss_adj, ss_mask)

    # Plot results
    print('Plotting dDEM results...')
    fig1 = plot_coreg_dh_results(diff_before, diff_before_ss, diff_before_ss_median, diff_before_ss_nmad,
                                 diff_after, diff_after_ss, diff_after_ss_median, diff_after_ss_nmad,
                                 diff_after_ss_adj, diff_after_ss_adj_ss, diff_after_ss_adj_ss_median,
                                 diff_after_ss_adj_ss_nmad,
                                 vmin=vmin, vmax=vmax)
    fig1.savefig(fig1_fn)
    print('Figure saved to file:', fig1_fn)

    # Calculate differences as a function of slope and aspect
    if plot_terrain_results:
        print('Plotting differences as a function of elevation, slope, and aspect...')
        fig2 = differences_vs_slope_aspect(ref_dem, diff_after_ss_adj)
        fig2.savefig(fig2_fn, dpi=300, bbox_inches='tight')
        print('Figure saved to file:', fig2_fn)

    return


def deramp_dem(tba_dem_fn=None, ss_mask_fn=None, ref_dem_fn=None, out_dir=None, poly_order=2,
               vmin=-5, vmax=5):
    # Apply a vertical correction using a polynomial 2D surface to the to-be-aligned DEM
    # See example in the XDEM docs: https://xdem.readthedocs.io/en/stable/advanced_examples/plot_deramp.html

    # Define output file names
    dem_corrected_fn = os.path.join(out_dir, os.path.basename(tba_dem_fn).replace('.tif', '_deramped.tif'))
    deramp_meta_fn = os.path.join(out_dir, os.path.basename(tba_dem_fn).replace('.tif', '_deramp_fit.json'))
    fig_fn = os.path.join(out_dir, os.path.basename(tba_dem_fn).replace('.tif', '_deramp_correction.png'))

    # Load input files
    tba_dem = xdem.DEM(tba_dem_fn)
    ss_mask = gu.Raster(ss_mask_fn, load_data=True)
    ss_mask = (ss_mask == 0)  # convert to boolean mask
    ref_dem = xdem.DEM(ref_dem_fn)

    # Calculate difference before
    diff_before = tba_dem - ref_dem

    # Mask values in DEM where dDEM > 5 (probably trees)
    tba_dem.data[diff_before.data > 5] = np.nan

    # Fit and apply Deramp object
    deramp = xdem.coreg.Deramp(poly_order=poly_order)
    deramp.fit(ref_dem, tba_dem, inlier_mask=ss_mask)
    meta = deramp.meta
    print(meta)
    dem_corrected = deramp.apply(tba_dem)

    # Save corrected DEM
    dem_corrected.save(dem_corrected_fn)
    print('Deramped DEM saved to file:', dem_corrected_fn)

    # Save Deramp fit metadata
    # keys_sub = [x for x in list(meta.keys()) if (x!= 'fit_func') & (x!='fit_optimizer')] # can't serialize functions in JSON
    # meta_sub = {key: meta[key] for key in keys_sub}
    # with open(deramp_meta_fn, 'w') as f:
    #     json.dump(meta_sub, f)
    # print('Deramp fit saved to file:', deramp_meta_fn)

    # Calculate difference after
    diff_after = dem_corrected - ref_dem

    # Plot results
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    diff_before.plot(cmap='coolwarm_r', vmin=vmin, vmax=vmax, ax=ax[0])
    ax[0].set_title('dDEM')
    diff_after.plot(cmap='coolwarm_r', vmin=vmin, vmax=vmax, ax=ax[1])
    ax[1].set_title('Deramped dDEM')
    ax[2].hist(np.ravel(diff_before.data), color='grey', bins=100)
    ax[2].set_xlim(vmin, vmax)
    ax[2].set_xlabel('Elevation differences (all surfaces) [m]')
    ax[3].hist(np.ravel(diff_after.data), color='grey', bins=100)
    ax[3].set_xlim(vmin, vmax)
    ax[3].set_xlabel('Elevation differences (all surfaces) [m]')
    fig.tight_layout()
    plt.close()

    # Save figure
    fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
    print('Figure saved to file:', fig_fn)

    return



