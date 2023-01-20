#!/usr/bin/env python3

import os
import requests
import glob
import tarfile
from proteus.dswx_hls import (
    get_dswx_hls_cli_parser,
    generate_dswx_layers,
    create_logger,
    parse_runconfig_file,
    compare_dswx_hls_products
)

FLAG_ALWAYS_DOWNLOAD = True

def test_workflow():

    parser = get_dswx_hls_cli_parser()

    test_data_directory = 'data'

    if not os.path.isdir(test_data_directory):
        os.makedirs(test_data_directory, exist_ok=True)

    dataset_name = 's30_louisiana_mississippi'
    dataset_url = ('https://zenodo.org/record/7555132/files/'
                   's30_louisiana_mississippi.tar.gz?download=1')

    dataset_dir = os.path.join(test_data_directory, dataset_name)
    user_runconfig_file = os.path.join(dataset_dir, 'dswx_hls.yaml')

    if (FLAG_ALWAYS_DOWNLOAD or not os.path.isdir(dataset_dir) or
            not os.path.isfile(user_runconfig_file)):

        print(f'Test dataset {dataset_name} not found. Downloading'
              f' file {dataset_url}.')
        response = requests.get(dataset_url)
        response.raise_for_status()

        compressed_filename = os.path.join(test_data_directory,
                                           os.path.basename(dataset_url))

        open(compressed_filename, 'wb').write(response.content)

        print(f'Extracting downloaded file {compressed_filename}')
        with tarfile.open(compressed_filename) as compressed_file:
            compressed_file.extractall(test_data_directory)

    ref_dir = os.path.join(dataset_dir, 'ref_dir')
    output_dir = os.path.join(dataset_dir, 'output_dir')
    # args.input_list = user_runconfig_file
    args = parser.parse_args([user_runconfig_file])

    create_logger(args.log_file)

    runconfig_constants = parse_runconfig_file(
        user_runconfig_file = user_runconfig_file, args = args)

    args.flag_debug = True

    generate_dswx_layers(
        args.input_list,
        args.output_file,
        hls_thresholds = runconfig_constants.hls_thresholds,
        dem_file=args.dem_file,
        dem_file_description=args.dem_file_description,
        output_interpreted_band=args.output_interpreted_band,
        output_rgb_file=args.output_rgb_file,
        output_infrared_rgb_file=args.output_infrared_rgb_file,
        output_binary_water=args.output_binary_water,
        output_confidence_layer=args.output_confidence_layer,
        output_diagnostic_layer=args.output_diagnostic_layer,
        output_non_masked_dswx=args.output_non_masked_dswx,
        output_shadow_masked_dswx=args.output_shadow_masked_dswx,
        output_landcover=args.output_landcover,
        output_shadow_layer=args.output_shadow_layer,
        output_cloud_layer=args.output_cloud_layer,
        output_dem_layer=args.output_dem_layer,
        output_browse_image=args.output_browse_image,
        browse_image_height=args.browse_image_height,
        browse_image_width=args.browse_image_width,
        landcover_file=args.landcover_file,
        landcover_file_description=args.landcover_file_description,
        worldcover_file=args.worldcover_file,
        worldcover_file_description=args.worldcover_file_description,
        shoreline_shapefile=args.shoreline_shapefile,
        shoreline_shapefile_description=args.shoreline_shapefile_description,
        flag_offset_and_scale_inputs=args.flag_offset_and_scale_inputs,
        scratch_dir=args.scratch_dir,
        product_id=args.product_id,
        product_version=args.product_version,
        shadow_masking_algorithm=args.shadow_masking_algorithm,
        min_slope_angle = args.min_slope_angle,
        max_sun_local_inc_angle=args.max_sun_local_inc_angle,
        mask_adjacent_to_cloud_mode=args.mask_adjacent_to_cloud_mode,
        copernicus_forest_classes=args.copernicus_forest_classes,
        ocean_masking_shoreline_distance_km = \
            args.ocean_masking_shoreline_distance_km,
        flag_debug=args.flag_debug)

    ref_files = glob.glob(os.path.join(ref_dir, '*'))
    for ref_file in ref_files:

        ref_basename = os.path.basename(ref_file)
        output_file = os.path.join(output_dir, ref_basename)

        assert compare_dswx_hls_products(ref_file, output_file)
