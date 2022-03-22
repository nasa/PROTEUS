#!/usr/bin/env python3

import os
import glob
from proteus.dswx_hls import (
    get_dswx_hls_cli_parser,
    generate_dswx_layers,
    create_logger,
    parse_runconfig_file,
    compare_dswx_hls_products
)

def test_workflow():

    parser = get_dswx_hls_cli_parser()

    user_runconfig_file = 'data/s30_louisiana_mississippi/dswx_hls.yaml'
    ref_dir = 'data/s30_louisiana_mississippi/ref_dir'
    output_dir = 'data/s30_louisiana_mississippi/output_dir'
    # args.input_list = user_runconfig_file
    args = parser.parse_args([user_runconfig_file])

    create_logger(args.log_file)


    hls_thresholds = parse_runconfig_file(
        user_runconfig_file = user_runconfig_file, args = args)

    args.flag_debug = True

    generate_dswx_layers(
        args.input_list,
        args.output_file,
        hls_thresholds = hls_thresholds,
        dem_file=args.dem_file,
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
        output_cloud_mask=args.output_cloud_mask,
        output_dem_layer=args.output_dem_layer,
        landcover_file=args.landcover_file,
        built_up_cover_fraction_file=args.built_up_cover_fraction_file,
        flag_offset_and_scale_inputs=args.flag_offset_and_scale_inputs,
        scratch_dir=args.scratch_dir,
        product_id=args.product_id,
        flag_debug=args.flag_debug)

    ref_files = glob.glob(os.path.join(ref_dir, '*'))
    for ref_file in ref_files:

        ref_basename = os.path.basename(ref_file)
        output_file = os.path.join(output_dir, ref_basename)

        assert compare_dswx_hls_products(ref_file, output_file)

    assert True