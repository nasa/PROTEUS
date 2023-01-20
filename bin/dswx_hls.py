#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# References:
#
# [1] J. W. Jones, "Efficient wetland surface water detection and
# monitoring via Landsat: Comparison with in situ data from the Everglades
# Depth Estimation Network", Remote Sensing, 7(9), 12503-12538.
# http://dx.doi.org/10.3390/rs70912503, 2015
#
# [2] R. Dittmeier, "LANDSAT DYNAMIC SURFACE WATER EXTENT (DSWE) ALGORITHM
# DESCRIPTION DOCUMENT (ADD)", USGS, March 2018
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import logging
import mimetypes
from proteus.dswx_hls import (
    get_dswx_hls_cli_parser,
    generate_dswx_layers,
    create_logger,
    parse_runconfig_file
)

logger = logging.getLogger('dswx_hls')

def main():
    parser = get_dswx_hls_cli_parser()

    args = parser.parse_args()

    create_logger(args.log_file, args.full_log_formatting)

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_list[0])[0]

    if len(args.input_list) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    if flag_first_file_is_text:
        user_runconfig_file = args.input_list[0]
    else:
        user_runconfig_file = None

    runconfig_constants = parse_runconfig_file(
        user_runconfig_file = user_runconfig_file, args = args)

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
        exclude_psw_aggressive_in_browse=args.exclude_psw_aggressive_in_browse,
        not_water_in_browse=args.not_water_in_browse,
        cloud_in_browse=args.cloud_in_browse,
        snow_in_browse=args.snow_in_browse,
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
        check_ancillary_inputs_coverage=args.check_ancillary_inputs_coverage,
        shadow_masking_algorithm=args.shadow_masking_algorithm,
        min_slope_angle = args.min_slope_angle,
        max_sun_local_inc_angle=args.max_sun_local_inc_angle,
        mask_adjacent_to_cloud_mode=args.mask_adjacent_to_cloud_mode,
        copernicus_forest_classes=args.copernicus_forest_classes,
        ocean_masking_shoreline_distance_km = \
            args.ocean_masking_shoreline_distance_km,
        flag_debug=args.flag_debug)


if __name__ == '__main__':
    main()