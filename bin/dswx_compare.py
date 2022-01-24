#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# References:
# [1] J. W. Jones, "Efficient wetland surface water detection and
# monitoring via Landsat: Comparison with in situ data from the Everglades 
# Depth Estimation Network", Remote Sensing, 7(9), 12503-12538.
# http://dx.doi.org/10.3390/rs70912503, 2015
#
# [2] R. Dittmeier, "LANDSAT DYNAMIC SURFACE WATER EXTENT (DSWE) ALGORITHM
# DESCRIPTION DOCUMENT (ADD)", USGS, March 2018
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import argparse
from osgeo import gdal
import numpy as np
from proteus.dswx_hls import band_description_dict


def _get_parser():
    parser = argparse.ArgumentParser(
        description='Compare two DSWx-HLS products',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs
    parser.add_argument('input_file',
                        type=str,
                        nargs=2,
                        help='Input images')

    # Outputs
    parser.add_argument('-o',
                        '--output-file',
                        dest='output_file',
                        type=str,
                        default='output_file',
                        help='Output file')

    return parser

def _get_prefix_str(flag_same):
    return '[OK]   ' if flag_same else '[FAIL] '

def main():
    parser = _get_parser()

    args = parser.parse_args()

    if not os.path.isfile(args.input_file[0]):
        print(f'ERROR file not found: {args.input_file[0]}')
        return

    if not os.path.isfile(args.input_file[1]):
        print(f'ERROR file not found: {args.input_file[1]}')
        return

    # TODO: compare projections ds.GetProjection()
    layer_gdal_dataset_1 = gdal.Open(args.input_file[0])
    geotransform_1 = layer_gdal_dataset_1.GetGeoTransform()
    metadata_1 = layer_gdal_dataset_1.GetMetadata()
    nbands_1 = layer_gdal_dataset_1.RasterCount

    layer_gdal_dataset_2 = gdal.Open(args.input_file[1])
    geotransform_2 = layer_gdal_dataset_2.GetGeoTransform()
    metadata_2 = layer_gdal_dataset_2.GetMetadata()
    nbands_2 = layer_gdal_dataset_2.RasterCount

    flag_same_nbands =  nbands_1 == nbands_2
    flag_same_nbands_str = _get_prefix_str(flag_same_nbands)
    prefix = ' ' * 7
    print(f'{flag_same_nbands_str}Comparing number of bands')
    if not flag_same_nbands:
        print(prefix + f'Input 1 has {nbands_1} bands and input 2 has {nbands_2}'
              ' bands')
        return False

    print('Comparing DSWx bands...')
    band_keys = list(band_description_dict.keys())
    band_names = list(band_description_dict.values())
    for b in range(1, nbands_1 + 1):
        gdal_band_1 = layer_gdal_dataset_1.GetRasterBand(b)
        gdal_band_2 = layer_gdal_dataset_2.GetRasterBand(b)
        image_1 = gdal_band_1.ReadAsArray()
        image_2 = gdal_band_2.ReadAsArray()
        flag_bands_are_equal = np.array_equal(image_1, image_2)
        flag_bands_are_equal_str = _get_prefix_str(flag_bands_are_equal)
        print(f'{flag_bands_are_equal_str}     Band {b} -'
              f' {band_keys[b-1]}: "{band_names[b-1]}"')
        if not flag_bands_are_equal:
            flag_error_found = False
            for i in range(image_1.shape[0]):
                for j in range(image_1.shape[1]):
                    if image_1[i, j] == image_2[i, j]:
                        continue
                    print(prefix + f'     * input 1 has value'
                          f' "{image_1[i, j]}" in position'
                          f' (x: {j}, y: {i})'
                          f' whereas input 2 has value "{image_2[i, j]}"'
                          ' in the same position.')
                    flag_error_found = True
                    break
                if flag_error_found:
                    break

    flag_same_geotransforms = np.array_equal(geotransform_1, geotransform_2)
    flag_same_geotransforms_str = _get_prefix_str(flag_same_geotransforms)
    print(f'{flag_same_geotransforms_str}Comparing geotransform')
    if not flag_same_geotransforms:
        print(prefix + f'* input 1 geotransform with content "{geotransform_1}"'
              f' differs from input 2 geotransform with content'
              f' "{geotransform_2}".')

    metadata_error_message = None
    flag_same_metadata = len(metadata_1.keys()) == len(metadata_2.keys())
    if not flag_same_metadata:
        metadata_error_message = (
            f'* input 1 metadata has {len(metadata_1.keys())} entries'
            f' whereas input 2 metadata has {len(metadata_2.keys())} entries.')

        set_1_m_2 = set(metadata_1.keys()) - set(metadata_2.keys())
        if len(set_1_m_2) > 0:
            metadata_error_message += (' Input 1 metadata has extra entries'
                                       ' with keys:'
                                       f' {", ".join(set_1_m_2)}.')
        set_2_m_1 = set(metadata_2.keys()) - set(metadata_1.keys())
        if len(set_2_m_1) > 0:
            metadata_error_message += (' Input 2 metadata has the extra'
                                       ' entries with keys:'
                                       f' {", ".join(set_2_m_1)}.')
    else:
        for k1, v1, in metadata_1.items():
            if k1 not in metadata_2.keys():
                 flag_same_metadata = False
                 metadata_error_message = (
                     f'* the metadata key {k1} is present in'
                     ' but it is not present in input 2')
                 break
            if k1 == 'PROCESSING_DATETIME':
                # Processing datetimes are expected to be different from
                # input 1 and 2
                continue
            if metadata_2[k1] != v1:
                 flag_same_metadata = False
                 metadata_error_message = (
                     f'* contents of metadata key {k1} from'
                     f' input 1 has value "{v1}" whereas the same key in'
                     f' input 2 metadata has value "{metadata_2[k1]}"')
                 break

    flag_same_metadata_str = _get_prefix_str(flag_same_metadata)
    print(f'{flag_same_metadata_str}Comparing metadata')
    if not flag_same_metadata:
        print(prefix + metadata_error_message)


if __name__ == '__main__':
    main()