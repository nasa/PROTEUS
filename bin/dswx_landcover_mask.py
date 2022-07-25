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
from osgeo import gdal, osr
import numpy as np
from proteus.dswx_hls import create_landcover_mask, create_logger

logger = logging.getLogger('dswx_hls_landcover_mask')

def _get_parser():
    parser = argparse.ArgumentParser(
        description='Create landcover mask LAND combining Copernicus Global'
        ' Land Service (CGLS) Land Cover Layers collection 3 at 100m and ESA'
        ' WorldCover 10m',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs
    parser.add_argument('-g', '--ref',
                        dest='reference_file',
                        type=str,
                        help='Reference file for geographic grid'
                              ' (e.g. HLS product layer)')

    parser.add_argument('-b', '--bbox',
                        type=float,
                        nargs=4,
                        dest='bbox',
                        metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
                        help='Defines the spatial region in '
                             'the format south north west east.')

    # other parameters
    parser.add_argument('-c', '--copernicus-landcover-100m',
                        '--landcover', '--land-cover',
                        dest='copernicus_landcover_file',
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
    parser.add_argument('-o', '--output-file',
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


def point2epsg(lon, lat):
    if lon >= 180.0:
        lon = lon - 360.0
    if lat >= 60.0:
        return 3413
    elif lat <= -60.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    raise ValueError(
        'Could not determine projection for {0},{1}'.format(lat, lon))

def main():
    parser = _get_parser()

    args = parser.parse_args()

    create_logger(args.log_file)

    if args.bbox is None and args.reference_file is None:
        logger.error('ERROR please, provide either a '
                     'reference file or a bounding box')
        return

    if args.bbox is not None:
        flag_is_geographic = True
        lat_min, lat_max, lon_min, lon_max = args.bbox
        # make sure lat_max > lat_min (since dy < 0)
        if lat_max < lat_min:
            lat_min, lat_max = lat_max, lat_min

    geographic_srs = None
    if args.reference_file:
        print(f'Reference file: {args.reference_file}')

        if not os.path.isfile(args.reference_file):
            logger.error(f'ERROR file not found: {args.reference_file}')
            return
        layer_gdal_dataset = gdal.Open(args.reference_file, gdal.GA_ReadOnly)
        if layer_gdal_dataset is None:
            logger.error(f'ERROR invalid file: {args.reference_file}')

        # read reference image geolocation
        geotransform = layer_gdal_dataset.GetGeoTransform()
        projection = layer_gdal_dataset.GetProjection()
        length = layer_gdal_dataset.RasterYSize
        width = layer_gdal_dataset.RasterXSize

        # check if geolocation is in projected coordinates, i.e.,
        # not geographic
        projection_srs = osr.SpatialReference(wkt=projection)

        flag_is_geographic = projection_srs.IsGeographic()
        if flag_is_geographic:
            geographic_srs = projection_srs
            print('Reference file is provided in geographic'
                  ' coordinates.')
            lat_max = geotransform[3]
            lat_min = lat_max + geotransform[5] * length
            lon_min = geotransform[0]
            lon_max = lon_min + geotransform[1] * width
        else:
            print('Reference file is NOT provided in geographic'
                        ' coordinates')

    if flag_is_geographic:
        mean_y = (lat_max + lat_min) / 2.0
        mean_x = (lon_min + lon_max) / 2.0
        epsg = point2epsg(mean_x, mean_y)

        print('Converting geographic coordinates to UTM:')
        print(f'    lat_min: {lat_min}, lat_max: {lat_max}')
        print(f'    lon_min: {lon_min}, lon_max: {lon_max}')
        print(f'    EPSG code: {epsg}')

        if geographic_srs is None:
            geographic_srs = osr.SpatialReference()
            geographic_srs.SetWellKnownGeogCS("WGS84")

        utm_srs = osr.SpatialReference()
        utm_srs.ImportFromEPSG(epsg)

        # create transformation of coordinates from geographic (lat/lon)
        # to UTM
        transformation = osr.CoordinateTransformation(geographic_srs, utm_srs)
        y_min = None
        y_max = None
        x_min = None
        x_max = None
        for lat in [lat_max, lat_min]:
            for lon in [lon_min, lon_max]:
                x, y, _ = transformation.TransformPoint(lat, lon, 0)
                if y_min is None or y_min > y:
                    y_min = y
                if y_max is None or y_max < y:
                    y_max = y
                if x_min is None or x_min > x:
                    x_min = x
                if x_max is None or x_max < x:
                    x_max = x

        print(f'    y_min: {y_min}, y_max: {y_max}')
        print(f'    x_min: {x_min}, x_max: {x_max}')

        # land cover map step is 30m meters
        dx = 30
        dy = -30
        geotransform = [x_min, dx, 0, y_max, 0, dy]

        width = int(np.ceil((x_max - x_min) / dx))
        length = int(np.ceil((y_min - y_max) / dy))
        projection = utm_srs.ExportToProj4()
        projection = projection.strip()

        print(f'    width: {width}')
        print(f'    length: {length}')
        print(f'    geotransform: {geotransform}')
        print(f'    projection: {projection}')

    create_landcover_mask(args.copernicus_landcover_file,
                          args.worldcover_file, args.output_file,
                          args.scratch_dir, args.mask_type,
                          geotransform, projection, length, width)


if __name__ == '__main__':
    main()