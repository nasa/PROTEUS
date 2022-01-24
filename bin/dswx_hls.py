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
import argparse
import mimetypes
from src.proteus.dswx_hls import generate_dswx_layers, \
                             create_logger, \
                             parse_runconfig_file

logger = logging.getLogger('dswx_hls')


def _get_parser():
    parser = argparse.ArgumentParser(
        description='Generate a DSWx-HLS product from an HLS product',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs
    parser.add_argument('input_list',
                        type=str,
                        nargs='+',
                        help='Input YAML run configuration file or HLS product file(s)')

    parser.add_argument('--dem',    
                        dest='dem_file',
                        type=str,
                        help='Input digital elevation model (DEM)')

    parser.add_argument('--landcover',
                        dest='landcover_file',
                        type=str,
                        help='Input Land Cover Discrete-Classification-map')

    parser.add_argument('--built-up-cover-fraction',
                        '--builtup-cover-fraction',    
                        dest='built_up_cover_fraction_file',
                        type=str,
                        help='Input built-up cover fraction layer')

    # Outputs
    parser.add_argument('-o',
                        '--output-file',
                        dest='output_file',
                        type=str,
                        help='Output DSWx-HLS product (GeoTIFF)')

    parser.add_argument('--wtr',
                        '--interpreted-band',
                        dest='output_interpreted_band',
                        type=str,
                        help='Output interpreted DSWx layer (GeoTIFF)')

    parser.add_argument('--output-rgb',
                        '--output-rgb-file',
                        dest='output_rgb_file',
                        type=str,
                        help='Output RGB reflectance file (GeoTIFF)'
                        ' copied from input HLS product.')

    parser.add_argument('--output-infrared-rgb',
                        '--output-infrared-rgb-file',
                        dest='output_infrared_rgb_file',
                        type=str,
                        help='Output infrared SWIR-1, NIR, and Red RGB'
                        '-color-composition GeoTIFF file')

    parser.add_argument('--bwtr'
                        '--output-binary-water',
                        dest='output_binary_water',
                        type=str,
                        help='Output binary water mask (GeoTIFF)')

    parser.add_argument('--diag',
                        '--output-diagnostic-layer',
                        dest='output_diagnostic_layer',
                        type=str,
                        help='Output diagnostic test layer file (GeoTIFF)')

    parser.add_argument('--intr',
                        '--output-non-masked-dswx',
                        dest='output_non_masked_dswx',
                        type=str,
                        help='Output non-masked DSWx layer file (GeoTIFF)')

    parser.add_argument('--insm',
                        '--output-shadow-masked-dswx',
                        dest='output_shadow_masked_dswx',
                        type=str,
                        help='Output GeoTIFF file with interpreted layer'
                        ' refined using land cover and terrain shadow testing')

    parser.add_argument('--shad',
                        '--output-shadow-layer',
                        dest='output_shadow_layer',
                        type=str,
                        help='Output terrain shadow layer file (GeoTIFF)')

    parser.add_argument('--cloud'
                        '--output-cloud-mask',
                        dest='output_cloud_mask',
                        type=str,
                        help='Output cloud/cloud-shadow classification file'
                        ' (GeoTIFF)')

    # Parameters
    parser.add_argument('--offset-and-scale-inputs',
                        dest='flag_offset_and_scale_inputs',
                        action='store_true',
                        default=False,
                        help='Offset and scale HLS inputs before processing')

    parser.add_argument('--scratch-dir',
                        '--temp-dir',
                        '--temporary-dir',
                        dest='scratch_dir',
                        type=str,
                        help='Scratch (temporary) directory')

    parser.add_argument('--debug',
                        dest='flag_debug',
                        action='store_true',
                        default=False,
                        help='Activate debug mode')

    parser.add_argument('--log',
                        '--log-file',
                        dest='log_file',
                        type=str,
                        help='Log file')

    return parser

def main():
    parser = _get_parser()

    args = parser.parse_args()

    create_logger(args.log_file)

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

    hls_thresholds = parse_runconfig_file(
        user_runconfig_file = user_runconfig_file, args = args)

    generate_dswx_layers(
        args.input_list,
        args.output_file,
        hls_thresholds = hls_thresholds,
        dem_file=args.dem_file, 
        output_interpreted_band=args.output_interpreted_band,
        output_rgb_file=args.output_rgb_file,
        output_infrared_rgb_file=args.output_infrared_rgb_file,
        output_binary_water=args.output_binary_water,
        output_diagnostic_layer=args.output_diagnostic_layer,
        output_non_masked_dswx=args.output_non_masked_dswx,
        output_shadow_masked_dswx=args.output_shadow_masked_dswx,
        output_shadow_layer=args.output_shadow_layer,
        output_cloud_mask=args.output_cloud_mask,
        landcover_file=args.landcover_file, 
        built_up_cover_fraction_file=args.built_up_cover_fraction_file,
        flag_offset_and_scale_inputs=args.flag_offset_and_scale_inputs,
        scratch_dir=args.scratch_dir,
        flag_debug=args.flag_debug)


if __name__ == '__main__':
    main()