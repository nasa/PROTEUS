#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Reference:
# [1] J. W. Jones, "Efficient wetland surface water detection and
# monitoring via Landsat: Comparison with in situ data from the Everglades
# Depth Estimation Network", Remote Sensing, 7(9), 12503-12538.
# http://dx.doi.org/10.3390/rs70912503, 2015
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import argparse
import logging
import os
from osgeo import gdal
from proteus.dswx_hls import create_landcover_mask, create_logger

logger = logging.getLogger('dswx_hls_landcover_mask')

def _get_parser():
    parser = argparse.ArgumentParser(
        description='Create landcover mask LAND combining Copernicus Global'
        ' Land Service (CGLS) Land Cover Layers collection 3 at 100m and ESA'
        ' WorldCover 10m',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs
    parser.add_argument('input_file',
                        type=str,
                        help='Input HLS product')

    parser.add_argument('-c',
                        '--copernicus-landcover-100m',
                        '--landcover', '--land-cover',
                        dest='landcover_file',
                        required=True,
                        type=str,
                        help='Input Copernicus Land Cover'
                        ' Discrete-Classification-map 100m')

    parser.add_argument('-w',
                        '--world-cover-10m', '--worldcover', '--world-cover',
                        dest='worldcover_file',
                        required=True,
                        type=str,
                        help='Input ESA WorldCover 10m')

    # Outputs
    parser.add_argument('-o',
                        '--output-file',
                        dest='output_file',
                        required=True,
                        type=str,
                        default='output_file',
                        help='Output landcover file (GeoTIFF) over input HLS'
                             ' product grid')

    # Parameters
    parser.add_argument('--mask-type',
                        dest='mask_type',
                        type=str,
                        default='standard',
                        help='Options: "Standard" and "Water Heavy"')

    parser.add_argument('--log',
                        '--log-file',
                        dest='log_file',
                        type=str,
                        help='Log file')

    parser.add_argument('--scratch-dir',
                        '--temp-dir',
                        '--temporary-dir',
                        dest='scratch_dir',
                        default='.',
                        type=str,
                        help='Scratch (temporary) directory')
    return parser


def main():
    parser = _get_parser()

    args = parser.parse_args()

    create_logger(args.log_file)

    logger.info(f'Input file: {args.input_file}')
    if not os.path.isfile(args.input_file):
        logger.error(f'ERROR file not found: {args.input_file}')
        return
    layer_gdal_dataset = gdal.Open(args.input_file, gdal.GA_ReadOnly)
    if layer_gdal_dataset is None:
        logger.error(f'ERROR invalid input HLS file: {args.input_file}')
    geotransform = layer_gdal_dataset.GetGeoTransform()
    projection = layer_gdal_dataset.GetProjection()
    length = layer_gdal_dataset.RasterYSize
    width = layer_gdal_dataset.RasterXSize

    create_landcover_mask(args.copernicus_landcover_file,
                          args.worldcover_file, args.output_file,
                          args.scratch_dir, args.mask_type,
                          geotransform, projection, length, width)


if __name__ == '__main__':
    main()