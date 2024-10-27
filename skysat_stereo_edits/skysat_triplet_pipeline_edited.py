#! /usr/bin/env python
import subprocess
import argparse
from datetime import datetime
import os,sys,glob,shutil
from rpcm import geo
import numpy as np
import geopandas as gpd
from distutils.spawn import find_executable
from skysat_stereo import misc_geospatial as misc
from skysat_stereo import asp_utils as asp

"""
Script for running the full pipeline based on workflow described in ISPRS 2020 submission
Need to specify input image folder, input refrence DEM folder
"""

def getparser():
    parser = argparse.ArgumentParser(description='Wrapper script to run full triplet stereo workflow')
    parser.add_argument('-in_img',default=None,type=str,help='path to Folder containing L1B imagery')
    parser.add_argument('-code_edit_dir',default=None,type=str,help='path to Folder containing edited skysat-stereo_scripts')
    parser.add_argument('-aoi_bbox',default=None,type=str,help='path to bounding box shapefile if limiting processing to a smaller aoi')
    parser.add_argument('-orthodem',default=None,type=str,help='path to Reference DEM to use in orthorectification and camera resection, if not provided, will use coregdem')
    parser.add_argument('-coregdem',default=None,type=str,help='path to reference DEM to use in coregisteration')
    parser.add_argument('-mask_dem',default=1,type=int,choices=[1,0],help='mask reference DEM for static surfaces before coreg (default: %(default)s)')
    mask_opt = ['glaciers','glaciers+nlcd']
    parser.add_argument('-mask_dem_opt',default='glaciers',choices=mask_opt,help='surfaces to mask if -mask_dem=1, default is glaciers which uses RGI polygons.\
                        If processing in CONUS, the option of glaciers+nlcd also additionaly masks out forest surfaces')
    parser.add_argument('-ortho_workflow',default=1,type=int,choices=[1,0],help='option to orthorectify before stereo or not')
    parser.add_argument('-block_matching',default=0,type=int,choices=[1,0],help='whether to use block matching in final stereo matching, default is 0 (not)')
    parser.add_argument('-job_name',default=None,type=str,help='identifier for output folder and final composite products')
    parser.add_argument('-outfolder',default=None,type=str,help='path to output folder to save results in')
    bin_choice = [1,0]
    parser.add_argument('-full_workflow',choices=bin_choice,type=int,default=1,help='Specify 1 to run full workflow (default: %(default)s)')
    parser.add_argument('-partial_workflow_steps',nargs='+',help='specify steps of workflow to run')
    return parser


def main():
    parser = getparser()
    args = parser.parse_args()
    # import edited scripts
    code_edit_dir = args.code_edit_dir
    sys.path.append(code_edit_dir)
    import skysat_stereo_workflow_edited as workflow
    import bundle_adjustment_lib_edited as ba

    img_folder = args.in_img
    coreg_dem = args.coregdem
    if args.orthodem is not None:
        ortho_dem = args.orthodem
    else:
        ortho_dem = coreg_dem
    # Check for input files
    img_list = glob.glob(os.path.join(img_folder,'*.tif'))+glob.glob(os.path.join(img_folder,'*.tiff'))
    if len(img_list)<2:
        print(f"Only {len(img_list)} images detected, exiting")
        sys.exit()
    if not os.path.exists(coreg_dem):
        print(f"Coreg dem {coreg_dem} could not be located, exiting")
        sys.exit()
    if not os.path.exists(ortho_dem):
        print(f"Ortho dem {ortho_dem} could not be located, exiting")
        sys.exit()

    # structure for output folder
    out_fol = os.path.join(args.outfolder,'proc_out')
    job_name = args.job_name
    
    #Universal Args
    if args.ortho_workflow == 1:
        map = True
    else:
        map = False
    if map:
        init_stereo_session = 'rpcmaprpc'
        init_ortho_session = 'rpc'
        final_stereo_session = 'pinholemappinhole'
        final_ortho_session = 'pinhole'
    else:
        init_stereo_session = 'rpc'
        init_ortho_session = 'rpc'
        final_stereo_session, final_ortho_session = ['nadirpinhole','pinhole']

    # For consistency, lets hardcode expected file names,folder names here :)
    # step 1 outputs
    overlap_full_txt = os.path.join(out_fol,'overlap.txt')
    overlap_full_pkl = os.path.splitext(overlap_full_txt)[0]+'_with_overlap_perc.pkl'
    overlap_stereo_pkl = os.path.splitext(overlap_full_pkl)[0]+'_stereo_only.pkl'
    overlap_stereo_txt = os.path.splitext(overlap_full_pkl)[0]+'_stereo_only.txt'
    bound_fn = os.path.splitext(overlap_full_txt)[0]+'_bound.gpkg'
    bound_buffer_fn = os.path.splitext(bound_fn)[0]+'_1km_buffer.gpkg'
    # step 2 outputs
    cam_gcp_directory = os.path.join(out_fol,'camgen_cam_gcp')
    # step 3 outputs
    init_ortho_dir = os.path.join(out_fol,'init_rpc_ortho')
    init_stereo_dir = os.path.join(out_fol,'init_rpc_stereo')
    # step 4 bundle_adjust dense matches
    init_ba = os.path.join(out_fol,'ba_pinhole')
    ba_prefix = os.path.join(init_ba,'run')
    # step 5 stereo_args
    intermediate_ortho_dir = os.path.join(out_fol,'intermediate_pinhole_ortho')
    final_stereo_dir = os.path.join(out_fol,'final_pinhole_stereo')
    # step 6, dem gridding and mosaicing
    mos_dem_dir = os.path.join(final_stereo_dir,'composite_dems')
    # step 7. dem_alignment
    alignment_dir = os.path.join(out_fol,'georegistered_dem_mos')
    # step 8, camera alignment
    aligned_cam_dir = os.path.join(out_fol,'georegistered_cameras')
    # step 9, final orthorectification
    final_ortho_dir = os.path.join(out_fol,'georegistered_orthomosaics')
    # step 10, plot figure
    final_figure = os.path.join(out_fol,f"{job_name}_result.jpg")
    # step 11, experimental rpc production
    if args.full_workflow == 1:
         steps2run = np.arange(0,11) # run the entire 10 steps
    else:
        steps2run = np.array(args.partial_workflow_steps).astype(int)

    # create output directory
    if not os.path.exists(out_fol):
        os.makedirs(out_fol)

    #copy reference DEM(s) to refdem directory
    # if parallel runs on different nodes use the same DEM, then will have issues
    refdem_dir = os.path.join(out_fol,'refdem')
    if not os.path.exists(refdem_dir):
        os.makedirs(refdem_dir)
    shutil.copy2(coreg_dem,os.path.join(refdem_dir,os.path.basename(coreg_dem)))
    if coreg_dem != ortho_dem:
        diff_dem = True
        shutil.copy2(ortho_dem,os.path.join(refdem_dir,os.path.basename(ortho_dem)))
    else:
        diff_dem = False
    # replace old variable names
    coreg_dem = os.path.join(refdem_dir,os.path.basename(coreg_dem))
    ortho_dem = os.path.join(refdem_dir,os.path.basename(ortho_dem))


    # Run the steps
    if 1 in steps2run:
        print('\n--------------------')
        print('Step 1: Compute overlapping pairs')
        print('--------------------\n')
        overlap_perc = 0.01
        workflow.prepare_stereopair_list(img_folder, overlap_perc, overlap_full_txt, aoi_bbox=None)


    print("Computing Target UTM zones for orthorectification")
    gdf = gpd.read_file(bound_fn)
    clon, clat = [gdf.centroid.x.values, gdf.centroid.y.values]
    epsg_code = f'EPSG:{geo.compute_epsg(clon, clat)}'
    print(f"Detected UTM zone is {epsg_code}")
    if not os.path.exists(bound_buffer_fn):
        print("Creating buffered shapefile")
        gdf_proj = gdf.to_crs(epsg_code)
        gdf_proj['geometry'] = gdf_proj.buffer(1000)
        gdf_proj.to_file(bound_buffer_fn, driver='GPKG')

    print("Cropping reference DEMs to extent of SkySat footprint + 1 km buffer")
    misc.clip_raster_by_shp_disk(coreg_dem, bound_buffer_fn)
    misc.ndvtrim_function(os.path.splitext(coreg_dem)[0] + '_shpclip.tif')
    coreg_dem = os.path.splitext(coreg_dem)[0] + '_shpclip_trim.tif'

    if diff_dem:
        misc.clip_raster_by_shp_disk(ortho_dem, bound_buffer_fn)
        misc.ndvtrim_function(os.path.splitext(ortho_dem)[0] + '_shpclip.tif')
        ortho_dem = os.path.splitext(ortho_dem)[0] + '_shpclip_trim.tif'
    else:
        ortho_dem = coreg_dem


    if 2 in steps2run:
        print('\n--------------------')
        print('Step 2: Generate frame cameras')
        print('--------------------\n')
        cam_gen_log = workflow.skysat_preprocess(img_folder, mode='triplet',
                                                product_level='l1b', overlap_pkl=overlap_stereo_pkl, dem=ortho_dem,
                                                outdir=cam_gcp_directory)
        now = datetime.now()
        log_fn = os.path.join(cam_gcp_directory, 'camgen_{}.log'.format(now))
        print("saving subprocess camgen log at {}".format(log_fn))
        with open(log_fn, 'w') as f:
            for log in cam_gen_log:
                f.write(log)


    if 3 in steps2run:
        # specify whether to run using maprojected sessions or not
        print('\n--------------------')
        print('Step 3: Orthorectify and run stereo using RPC cameras')
        print('--------------------\n')
        if map:
            # orthorectify all the images first
            print("Orthorectifying images using RPC camera")
            workflow.execute_skysat_orhtorectification(images=img_list, data='triplet', session=init_ortho_session,
                                                    outdir=init_ortho_dir, tsrs=epsg_code, dem=ortho_dem, mode='science',
                                                    overlap_list=overlap_stereo_txt, copy_rpc=1, orthomosaic=0)
            init_stereo_input_img_folder = init_ortho_dir
        else:
            init_stereo_input_img_folder = img_folder
        # Remove images from init_stereo_dir that are not in init_
        print("Running stereo using RPC cameras")
        # Note crop_map = 0 option, this does not do warping to common extent and resolution for orthoimages before stereo, because we want to
        # presrve this crucail information for correctly unwarped dense match points
        workflow.execute_skysat_stereo(init_stereo_input_img_folder, init_stereo_dir,
                                    mode='triplet', session=init_stereo_session,
                                    dem=ortho_dem, texture='normal', writeout_only=False,
                                    block=1, crop_map=0, threads=2, overlap_pkl=overlap_stereo_pkl,
                                    cross_track=False)
        # copy dense match file to ba directory
        workflow.dense_match_wrapper(stereo_master_dir=os.path.abspath(init_stereo_dir),
                                    ba_dir=os.path.abspath(init_ba), modify_overlap=0)


    if 4 in steps2run:
        print('\n--------------------')
        print('Step 4: Bundle adjustment')
        print('--------------------\n')
        ba_prefix = os.path.join(init_ba, 'run')
        print("running bundle adjustment")
        ba.bundle_adjust_stable(img=img_folder, ba_prefix=ba_prefix, cam=os.path.abspath(cam_gcp_directory),
                                session='nadirpinhole', overlap_list=overlap_stereo_txt,
                                num_iter=700, num_pass=2, mode='full_triplet')


    # Correct the img_list for images that didn't make it through step 4, use old cameras if they exist
    old_cam_list = [] # keep track of how many images are using old camera model
    no_cam_list = [] # keep track of how many images have no camera model
    for img_fn in img_list:
        cam_fn = os.path.join(cam_gcp_directory, os.path.basename(img_fn).replace('.tif', '_rpc.tsai')) # step 2 camera model
        ba_cam_fn = os.path.join(init_ba, 'run-run-' + os.path.basename(img_fn).replace('.tif', '_rpc.tsai')) # step 4 camera model
        # check if camera exists in ba_pinhole folder
        if os.path.exists(ba_cam_fn):
            continue
        # check if camera exists in initial camgen folder
        elif os.path.exists(cam_fn):
            # copy to ba_pinhole folder
            shutil.copy2(cam_fn, ba_cam_fn)
            old_cam_list += img_fn
        else:
            no_cam_list += img_fn
    # update image list
    cam_list = glob.glob(os.path.join(ba_cam_fn, 'run-run-*_rpc.tsai'))
    img_list = [os.path.join(img_folder, os.path.basename(cam_list[i]).replace('_rpc.tsai', '.tif')) for i in range(len(cam_list))]
    # print camera adjustments
    print(f'Using old cameras for {len(old_cam_list)} images')
    print(f'No cameras for {len(no_cam_list)} images, removed from list')
    

    if 5 in steps2run:
        print('\n--------------------')
        print('Step 5: Final stereo')
        print('--------------------\n')
        if map:
            print("Running intermediate orthorectification with bundle adjusted pinhole cameras")
            workflow.execute_skysat_orhtorectification(images=img_list, data='triplet', session=final_ortho_session,
                                                        outdir=intermediate_ortho_dir, tsrs=epsg_code, dem=ortho_dem,
                                                        ba_prefix=ba_prefix + '-run', mode='science',
                                                        overlap_list=overlap_stereo_txt,
                                                        copy_rpc=1, orthomosaic=0)
            final_stereo_input_img_folder = intermediate_ortho_dir
        else:
            final_stereo_input_img_folder = img_folder
        # now run stereo
        print("Running final stereo reconstruction")
        workflow.execute_skysat_stereo(final_stereo_input_img_folder,
                                        final_stereo_dir, ba_prefix=ba_prefix + '-run',
                                        mode='triplet', session=final_stereo_session,
                                        dem=ortho_dem, texture='normal', writeout_only=False,
                                        block=args.block_matching, crop_map=1, threads=2, overlap_pkl=overlap_stereo_pkl,
                                        cross_track=False)


    if 6 in steps2run:
        print('\n--------------------')
        print('Step 6: Mosaic DEMs')
        print('--------------------\n')
        pc_list = sorted(glob.glob(os.path.join(final_stereo_dir, '20*/2*/run-PC.tif')))
        print(f"Identified {len(pc_list)} clouds")
        # this is dem gridding followed by mosaicing
        workflow.gridding_wrapper(pc_list, tr=2)
        print("Mosaicing DEMs")
        workflow.dem_mosaic_wrapper(dir=os.path.abspath(final_stereo_dir), mode='triplet',
                                    out_folder=os.path.abspath(mos_dem_dir))


    if 7 in steps2run:
        print('\n--------------------')
        print('Step 7: Align DEMs')
        print('--------------------\n')
        # now perform alignment
        median_mos_dem = glob.glob(os.path.join(mos_dem_dir, '*_median_mos.tif'))[0]
        print("Aligning DEMs")
        workflow.alignment_wrapper_single(coreg_dem, source_dem=median_mos_dem, max_displacement=40, outprefix=os.path.join(alignment_dir, 'run'))


    if 8 in steps2run:
        print('\n--------------------')
        print('Step 8: Align frame camera models')
        print('--------------------\n')
        # this steps aligns the frame camera models
        camera_list = sorted(glob.glob(os.path.join(init_ba, 'run-run-*.tsai')))
        print(f"Detected {len(camera_list)} cameras to be registered to DEM")
        alignment_vector = glob.glob(os.path.join(alignment_dir, 'alignment_vector.txt'))[0]
        if not os.path.exists(aligned_cam_dir):
            os.makedirs(aligned_cam_dir)
        print("Aligning cameras")
        workflow.align_cameras_wrapper(input_camera_list=camera_list, transform_txt=alignment_vector, outfolder=aligned_cam_dir)


    if 9 in steps2run:
        print('\n--------------------')
        print('Step 9: Construct final georegistered orthomosaics')
        print('--------------------\n')
        georegistered_median_dem = glob.glob(os.path.join(alignment_dir, 'run-trans_*DEM.tif'))[0]
        print("Running final orthomsaic creation")
        workflow.execute_skysat_orhtorectification(images=img_list, data='triplet', session=final_ortho_session,
                                                   outdir=final_ortho_dir, tsrs=epsg_code, dem=georegistered_median_dem,
                                                   ba_prefix=os.path.join(aligned_cam_dir, 'run-run'), mode='science',
                                                   overlap_list=overlap_stereo_txt, copy_rpc=0, orthomosaic=1)


    if 10 in steps2run:
        print('\n--------------------')
        print('Step 10: Produce final plot of orthoimage, DEM, NMAD, and countmaps')
        print('--------------------\n')
        ortho = glob.glob(os.path.join(final_ortho_dir, '*finest_orthomosaic.tif'))[0]
        count = glob.glob(os.path.join(mos_dem_dir, '*count*.tif'))[0]
        nmad = glob.glob(os.path.join(mos_dem_dir, '*nmad*.tif'))[0]
        georegistered_median_dem = glob.glob(os.path.join(alignment_dir, 'run-trans_*DEM.tif'))[0]
        print("plotting final figure")
        misc.plot_composite_fig(ortho, georegistered_median_dem, count, nmad, outfn=final_figure)


if __name__ == '__main__':
    main()
