#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute Dynamic Surface Water Extent (DSWx)
# 
# OPERA
#
# Copyright 2021, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S.
# export laws and regulations. User has the responsibility to obtain export
# licenses, or other export authority as may be required before exporting such
# information to foreign countries or providing access to foreign persons.
#
# 
# References:
# [1] Jones, J. W. (2015). Efficient wetland surface water detection and 
# monitoring via Landsat: Comparison with in situ data from the Everglades 
# Depth Estimation Network. Remote Sensing, 7(9), 12503-12538. 
# http://dx.doi.org/10.3390/rs70912503.
# 
# [2] R. Dittmeier, LANDSAT DYNAMIC SURFACE WATER EXTENT (DSWE) ALGORITHM 
# DESCRIPTION DOCUMENT (ADD)", USGS, March 2018
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import argparse
from osgeo import gdal
from dswx_hls import save_mask, save_dswx_product


def _get_parser():
    parser = argparse.ArgumentParser(description='',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs
    parser.add_argument('input_file',
                        type=str,
                        nargs=1,
                        help='Input images')

    # Outputs
    parser.add_argument('-o',
                        '--output-file',
                        dest='output_file',
                        type=str,
                        required=True,
                        default='output_file',
                        help='Output file')

    # Parameters
    parser_dataset = parser.add_mutually_exclusive_group()
    parser_dataset.add_argument('--mask',
                                action='store_false',
                                default=None,
                                dest='interpreted_dswx',
                                help='Append color table to mask')

    parser_dataset.add_argument('--interpreted-dswx',
                                action='store_true',
                                dest='interpreted_dswx',
                                help='Append color table to interpreted DSWx layer')

    return parser


def main():
    parser = _get_parser()

    args = parser.parse_args()

    layer_gdal_dataset = gdal.Open(args.input_file[0])
    geotransform = layer_gdal_dataset.GetGeoTransform()
    projection = layer_gdal_dataset.GetProjection()
    if layer_gdal_dataset.RasterCount > 1:
        band = layer_gdal_dataset.GetRasterBand(1)
        image = band.ReadAsArray()
    else:
        image = layer_gdal_dataset.ReadAsArray()

    if args.interpreted_dswx is False:
        print('Appending color table to mask')
        save_mask(image, args.output_file, geotransform, projection)
    else:
        print('Appending color table to interpreted DSWx layer')
        metadata_dict = {}
        save_dswx_product(image, args.output_file, metadata_dict,
                          geotransform, projection)


if __name__ == '__main__':
    main()