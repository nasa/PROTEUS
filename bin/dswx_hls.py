# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# References:
#
# [1] J. W. Jones, "Efficient wetland surface water detection and 
# monitoring via Landsat: Comparison with in situ data from the Everglades 
# Depth Estimation Network", Remote Sensing, 7(9), 12503-12538. 
# http://dx.doi.org/10.3390/rs70912503, 2015
# 
# [2] R. Dittmeier, LANDSAT DYNAMIC SURFACE WATER EXTENT (DSWE) ALGORITHM 
# DESCRIPTION DOCUMENT (ADD)", USGS, March 2018
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import argparse
import logging
import tempfile
import os
import shutil
import sys
import glob
import numpy as np
import mimetypes
import yamale
from datetime import datetime
from collections import OrderedDict
from ruamel.yaml import YAML as ruamel_yaml
from osgeo.gdalconst import GDT_Float32
from scipy.ndimage import convolve
from osgeo import gdal, osr
from modules.core import save_as_cog
from modules.dswx_hls import l30_v1_band_dict, \
                             s30_v1_band_dict, \
                             l30_v2_band_dict, \
                             s30_v2_band_dict, \
                             interpreted_dswx_band_dict, \
                             band_description_dict, \
                             METADATA_FIELDS_TO_COPY_FROM_HLS_LIST, \
                             get_mask_ctable, \
                             get_interpreted_dswx_ctable

logger = logging.getLogger('dswx_hls')

# Thresholds
wigt = 0.0124  # Modified Normalized Difference Wetness Index (MNDWI) Threshold
awgt = 0.0  # Automated Water Extent Shadow Threshold
pswt_1_mndwi = -0.44  # Partial Surface Water Test-1 MNDWI Threshold
pswt_1_nir = 1500  # Partial Surface Water Test-1 NIR Threshold
pswt_1_swir1 = 900  # Partial Surface Water Test-1 SWIR1 Threshold
pswt_1_ndvi = 0.7  # Partial Surface Water Test-1 NDVI Threshold
pswt_2_mndwi = -0.5  # Partial Surface Water Test-2 MNDWI Threshold
pswt_2_blue = 1000  # Partial Surface Water Test-2 Blue Threshold
pswt_2_nir = 2500  # Partial Surface Water Test-2 NIR Threshold
pswt_2_swir1 = 3000  # Partial Surface Water Test-2 SWIR1 Threshold
pswt_2_swir2 = 1000  # Partial Surface Water Test-2 SWIR2 Threshold


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
                        help='Output interpreted DSWx layer')

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
                        '-color-composition file')

    parser.add_argument('--bwtr'
                        '--output-binary-water',
                        dest='output_binary_water',
                        type=str,
                        help='Output binary water mask')

    parser.add_argument('--diag',
                        '--output-diagnostic-layer',
                        dest='output_diagnostic_test_band',
                        type=str,
                        help='Output diagnostic test layer file')

    parser.add_argument('--intr',
                        '--output-non-masked-dswx',
                        dest='output_non_masked_dswx',
                        type=str,
                        help='Output non-masked DSWx layer file')

    parser.add_argument('--insm',
                        '--output-shadow-masked-dswx',
                        dest='output_shadow_masked_dswx',
                        type=str,
                        help='Output interpreted layer refined using'
                        ' land cover and terrain shadow testing ')

    parser.add_argument('--shad',
                        '--output-shadow-layer',
                        dest='output_shadow_layer',
                        type=str,
                        help='Output terrain shadow layer file')

    parser.add_argument('--cloud'
                        '--output-cloud-mask',
                        dest='output_cloud_mask',
                        type=str,
                        help='Output cloud/cloud-shadow classification file')

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



def _generate_interpreted_layer(diagnostic_test_band):

    logger.info('step 2 - get interpreted DSWX band')
    shape = diagnostic_test_band.shape
    interpreted_dswx_band = np.zeros(shape, dtype = np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for key, value in interpreted_dswx_band_dict.items():
                if diagnostic_test_band[i, j] == key:
                    interpreted_dswx_band[i, j] = value
                    break
            else:
                interpreted_dswx_band[i, j] = 255

    return interpreted_dswx_band


def _get_binary_water_layer(masked_dswx_band):
 
    binary_water_layer = np.zeros_like(masked_dswx_band)

    # water classes: 1 to 4
    for class_value in range(1, 5):
        ind = np.where(masked_dswx_band == class_value)
        binary_water_layer[ind] = 1

    # invalid classes: 9 (Q/A masked) or 255 (fill value)    
    for class_value in [9, 255]:
        ind = np.where(masked_dswx_band == class_value)
        binary_water_layer[ind] = 255

    return binary_water_layer


def _compute_diagnostic_tests(blue, green, red,
                              nir, swir1, swir2):

    # Temporarily supress RuntimeWarnings:
    # - divide by zero encountered in true_divide
    # - invalid value encountered in true_divide
    old_settings = np.seterr(invalid='ignore', divide='ignore')

    # Modified Normalized Difference Wetness Index (MNDWI)
    mndwi = (green - swir1)/(green + swir1)

    # Multi-band Spectral Relationship Visible (MBSRV)
    mbsrv = green + red

    # Multi-band Spectral Relationship Near-Infrared (MBSRN)
    mbsrn = nir + swir1

    # Automated Water Extent Shadow (AWESH) 
    awesh = blue + (2.5 * green) - (1.5 * mbsrn) - (0.25 * swir2)

    # Normalized Difference Vegetation Index (NDVI) 
    ndvi = (nir - red) / (nir + red)

    # Restore numpy RuntimeWarnings settings
    np.seterr(**old_settings)

    # Diagnostic test band
    shape = blue.shape
    diagnostic_test_band = np.zeros(shape, dtype = np.uint8)

    logger.info('step 1 - compute diagnostic tests')
    for i in range(shape[0]):
        for j in range(shape[1]):

            # Implementation of water tests described in [1, 2]

            # Test 1 
            if (mndwi[i, j] > wigt):
                diagnostic_test_band[i, j] += 1

            # Test 2
            if (mbsrv[i, j] > mbsrn[i, j]):
                diagnostic_test_band[i, j] += 2

            # Test 3
            if (awesh[i, j] > awgt):
                diagnostic_test_band[i, j] += 4

            # Test 4
            if (mndwi[i, j] > pswt_1_mndwi and 
                    swir1[i, j] < pswt_1_swir1 and
                    nir[i, j] < pswt_1_nir and
                    ndvi[i, j] < pswt_1_ndvi):
                diagnostic_test_band[i, j] += 8

            # Test 5
            if (mndwi[i, j] > pswt_2_mndwi and
                    blue[i, j] < pswt_2_blue and
                    swir1[i, j] < pswt_2_swir1 and
                    swir2[i, j] < pswt_2_swir2 and
                    nir[i, j] < pswt_2_nir):
                diagnostic_test_band[i, j] += 16

    return diagnostic_test_band


def _compute_otsu_threshold(image, is_normalized = True):
    '''
    Compute Otsu threshold 
    source: https://learnopencv.com/otsu-thresholding-with-opencv/
    '''
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    logger.info(f"Otsu's algorithm implementation thresholding result: {threshold}")

    return image < threshold

def _compute_mask_and_filter_interpreted_layer(interpreted_dswx_band, qa_band):
    '''
    QA band - Landsat 8
    BITS:
    0 - Cirrus (reserved but not used)
    1 - Cloud (*)
    2 - Adjacent to cloud/shadow
    3 - Cloud shadow (*)
    4 - Snow/ice (*)
    5 - Water
    6-7 - Aerosol quality:
          00 - Climatology aerosol
          01 - Low aerosol
          10 - Moderate aerosol
          11 - High aerosol

    (*) set output as 9
    '''
    shape = interpreted_dswx_band.shape
    masked_dswx_band = interpreted_dswx_band.copy()
    mask = np.zeros(shape, dtype = np.uint8)

    for i in range(shape[0]):
        for j in range(shape[1]):

            # Check QA cloud shadow bit (3) => bit 0
            if np.bitwise_and(2**3, qa_band[i, j]):
                mask[i, j] += 1

            # Check QA snow bit (4) => bit 1
            if np.bitwise_and(2**4, qa_band[i, j]):
                mask[i, j] += 2

            # Check QA cloud bit (1) => bit 2
            if np.bitwise_and(2**1, qa_band[i, j]):
                mask[i, j] += 4

            if mask[i, j] == 0:
                continue
            
            masked_dswx_band[i, j] = 9

    return mask, masked_dswx_band


def _load_hls_from_file(filename, image_dict, offset_dict, scale_dict,
                        dswx_metadata_dict, key,
                        flag_offset_and_scale_inputs, flag_debug = False,
                        band_name = None):

    layer_gdal_dataset = gdal.Open(filename)
    if layer_gdal_dataset is None:
        return None

    if 'hls_dataset_name' not in image_dict.keys():
        hls_dataset_name = os.path.splitext(os.path.basename(filename))[0]
        if band_name:
            hls_dataset_name = hls_dataset_name.replace(f'.{band_name}', '')
        image_dict['hls_dataset_name'] = hls_dataset_name

    offset = 0.0
    scale_factor = 1.

    metadata = layer_gdal_dataset.GetMetadata()

    if 'SPACECRAFT_NAME' not in dswx_metadata_dict.keys():
        for k, v in metadata.items():
            if k.upper() not in METADATA_FIELDS_TO_COPY_FROM_HLS_LIST:
                continue
            dswx_metadata_dict[k.upper()] = v

        sensor = None

        if 'SPACECRAFT_NAME' in metadata:
            spacecraft_name = metadata['SPACECRAFT_NAME'].upper()
            if 'SENTINEL' not in spacecraft_name and 'LANDSAT' not in spacecraft_name:
                logger.info(f'ERROR the platform "{spacecraft_name}" is not supported')
                return False
        elif 'SENSOR' in metadata:
            sensor = metadata['SENSOR']
            if 'OLI' in sensor:
                spacecraft_name = 'LANDSAT-8'
            elif 'MSI' in sensor:
                spacecraft_name = 'SENTINEL-1'
            else:
                logger.info(f'ERROR the sensor "{sensor}" is not supported')
                return False
        else:
            logger.info(f'ERROR could not determine the platorm from metadata')
            return False

        dswx_metadata_dict['SPACECRAFT_NAME'] = spacecraft_name
        if sensor is not None:
            dswx_metadata_dict['SENSOR'] = sensor
        elif 'SENTINEL' in spacecraft_name:
            dswx_metadata_dict['SENSOR'] = 'MSI'
        else:
            dswx_metadata_dict['SENSOR'] = 'OLI'

    if key == 'qa':
        if flag_debug:
            logger.info('reading in debug mode')
            image_dict[key] = layer_gdal_dataset.ReadAsArray(
                xoff=0, yoff=0, xsize=1000, ysize=1000)
        else:
            image_dict[key] = layer_gdal_dataset.ReadAsArray()
    else:
        for metadata_key, metadata_value in metadata.items():
            if metadata_key == 'add_offset':
                offset = float(metadata_value)
            elif metadata_key == 'scale_factor':
               scale_factor = float(metadata_value)
        if flag_debug:
            logger.info('reading in debug mode')
            image = layer_gdal_dataset.ReadAsArray(
                xoff=0, yoff=0, xsize=1000, ysize=1000)
        else:
            image = layer_gdal_dataset.ReadAsArray()
        if flag_offset_and_scale_inputs:
            image = scale_factor * (np.asarray(image, dtype=np.float32) -
                                    offset)
        image_dict[key] = image

    # save offset and scale factor into corresponding dictionaries
    offset_dict[key] = offset
    scale_dict[key] = scale_factor
            
    if 'geotransform' not in image_dict.keys():
        image_dict['geotransform'] = \
            layer_gdal_dataset.GetGeoTransform()
        image_dict['projection'] = \
            layer_gdal_dataset.GetProjection()
        band = layer_gdal_dataset.GetRasterBand(1)
        image_dict['fill_data'] = band.GetNoDataValue()
        image_dict['length'] = layer_gdal_dataset.RasterYSize
        image_dict['width'] = layer_gdal_dataset.RasterXSize
    
    return True


def _load_hls_product_v1(filename, image_dict, offset_dict, 
                         scale_dict, dswx_metadata_dict,
                         flag_offset_and_scale_inputs,
                         flag_debug = False):

    if isinstance(filename, list):
        filename = filename[0]

    logger.info('loading HLS v.1.x layers:')
    for key in l30_v1_band_dict.keys():

        logger.info(f'    {key}')

        # Sensor is undertermined (first band) or LANDSAT
        if ('SPACECRAFT_NAME' not in dswx_metadata_dict.keys() or
                'LANDSAT' in dswx_metadata_dict['SPACECRAFT_NAME']):
            band_name = l30_v1_band_dict[key]
        else:
            band_name = s30_v1_band_dict[key]

        band_ref = f'HDF4_EOS:EOS_GRID:"{filename}":Grid:{band_name}'
        success = _load_hls_from_file(band_ref, image_dict, offset_dict,
                                      scale_dict, dswx_metadata_dict,
                                      key, flag_offset_and_scale_inputs,
                                      flag_debug = flag_debug)
        if not success:
            return False

    return True
        

def _load_hls_product_v2(file_list, image_dict, offset_dict, 
                         scale_dict, dswx_metadata_dict,
                         flag_offset_and_scale_inputs, flag_debug = False):

    logger.info('loading HLS v.2.0 layers:')
    for key in l30_v2_band_dict.keys():

        logger.info(f'    {key}')

        # Sensor is undertermined (first band) or LANDSAT
        if ('SPACECRAFT_NAME' not in dswx_metadata_dict.keys() or
                'LANDSAT' in dswx_metadata_dict['SPACECRAFT_NAME']):
            band_name = l30_v2_band_dict[key]
        else:
            band_name = s30_v2_band_dict[key]

        for filename in file_list:
            if band_name in filename:
                break
        else:
            logger.info(f'ERROR band {key} not found within input file(s)')
            return
        success = _load_hls_from_file(filename, image_dict, offset_dict,
                                      scale_dict, dswx_metadata_dict,
                                      key, flag_offset_and_scale_inputs,
                                      flag_debug = flag_debug,
                                      band_name = band_name)
        if not success:
            return False

    return True

def _get_binary_water_ctable():

    # create color table
    binary_water_ctable = gdal.ColorTable()
    binary_water_ctable.SetColorEntry(0, (255, 255, 255))  # No water
    binary_water_ctable.SetColorEntry(1, (0, 0, 255))  # Water
    binary_water_ctable.SetColorEntry(255, (0, 0, 0, 255))  # Black - Fill value
    return binary_water_ctable


def save_dswx_product(wtr, output_file, dswx_metadata_dict, geotransform,
                      projection, scratch_dir='.', output_files_list = None,
                      description = None, **dswx_processed_bands):

    _makedirs(output_file)
    shape = wtr.shape
    driver = gdal.GetDriverByName("GTiff")

    dswx_processed_bands['wtr'] = wtr
    dswx_processed_bands_keys = dswx_processed_bands.keys()
    dswx_band_names_list = band_description_dict.keys() 

    # check input arrays different than None
    n_valid_bands = int(np.sum([int(dswx_processed_bands[band_key.lower()] is not None) 
                               for band_key in band_description_dict.keys()
                               if band_key.lower() in dswx_processed_bands_keys]))

    if n_valid_bands == 1:
        # save interpreted layer (single band)
        nbands = 1
    else:
        # save DSWx product
        nbands = len(band_description_dict.keys())
    gdal_ds = driver.Create(output_file, shape[1], shape[0], nbands, gdal.GDT_Byte)
    gdal_ds.SetMetadata(dswx_metadata_dict)
    gdal_ds.SetGeoTransform(geotransform)
    gdal_ds.SetProjection(projection)

    for band_index, (band_key, description_from_dict) in enumerate(
            band_description_dict.items()):
        
        # check if band is in the list of processed bands
        if band_key.lower() in dswx_processed_bands:
            band_array = dswx_processed_bands[band_key.lower()]
        else:
            band_array = None
        
        # if band is not in the list of processed bands or it's None
        if band_array is None:
            band_array = np.zeros_like(wtr)

        gdal_band = gdal_ds.GetRasterBand(band_index + 1)
        gdal_band.WriteArray(band_array)
        gdal_band.SetNoDataValue(255)
        if n_valid_bands == 1:
            # set color table and color interpretation
            dswx_ctable = get_interpreted_dswx_ctable()
            gdal_band.SetRasterColorTable(dswx_ctable)
            gdal_band.SetRasterColorInterpretation(
                gdal.GCI_PaletteIndex)
        if description is not None:
            gdal_band.SetDescription(description)
        else:
            gdal_band.SetDescription(description_from_dict)
        gdal_band.FlushCache()
        gdal_band = None
        if n_valid_bands == 1:
            break

    gdal_ds.FlushCache()
    gdal_ds = None

    save_as_cog(output_file, scratch_dir, logger)

    if output_files_list is not None:
        output_files_list.append(output_file)
    logger.info(f'file saved: {output_file}')


def save_mask(mask, output_file, dswx_metadata_dict, geotransform, projection,
              description = None, output_files_list = None):

    _makedirs(output_file)
    shape = mask.shape
    driver = gdal.GetDriverByName("GTiff")
    gdal_ds = driver.Create(output_file, shape[1], shape[0], 1, gdal.GDT_Byte)
    gdal_ds.SetMetadata(dswx_metadata_dict)
    gdal_ds.SetGeoTransform(geotransform)
    gdal_ds.SetProjection(projection)
    mask_band = gdal_ds.GetRasterBand(1)
    mask_band.WriteArray(mask)
    mask_band.SetNoDataValue(255)

    # set color table and color interpretation
    mask_ctable = get_mask_ctable()
    mask_band.SetRasterColorTable(mask_ctable)
    mask_band.SetRasterColorInterpretation(
        gdal.GCI_PaletteIndex)

    if description is not None:
        mask_band.SetDescription(description)

    gdal_ds.FlushCache()
    gdal_ds = None

    if output_files_list is not None:
        output_files_list.append(output_file)
    logger.info(f'file saved: {output_file}')


def _save_binary_water(binary_water_layer, output_file, dswx_metadata_dict,
                       geotransform, projection, description = None,
                       output_files_list = None):
    _makedirs(output_file)
    shape = binary_water_layer.shape
    driver = gdal.GetDriverByName("GTiff")
    gdal_ds = driver.Create(output_file, shape[1], shape[0], 1, gdal.GDT_Byte)
    gdal_ds.SetMetadata(dswx_metadata_dict)
    gdal_ds.SetGeoTransform(geotransform)
    gdal_ds.SetProjection(projection)
    binary_water_band = gdal_ds.GetRasterBand(1)
    binary_water_band.WriteArray(binary_water_layer)
    binary_water_band.SetNoDataValue(255)

    # set color table and color interpretation
    binary_water_ctable = _get_binary_water_ctable()
    binary_water_band.SetRasterColorTable(binary_water_ctable)
    binary_water_band.SetRasterColorInterpretation(
        gdal.GCI_PaletteIndex)

    if description is not None:
        binary_water_band.SetDescription(description)

    gdal_ds.FlushCache()
    gdal_ds = None

    if output_files_list is not None:
        output_files_list.append(output_file)
    logger.info(f'file saved: {output_file}')


def _save_array(input_array, output_file, dswx_metadata_dict, geotransform,
                projection, description = None, output_files_list = None):

    _makedirs(output_file)
    shape = input_array.shape
    driver = gdal.GetDriverByName("GTiff")
    gdal_ds = driver.Create(output_file, shape[1], shape[0], 1, gdal.GDT_Byte)
    gdal_ds.SetMetadata(dswx_metadata_dict)
    gdal_ds.SetGeoTransform(geotransform)
    gdal_ds.SetProjection(projection)

    raster_band = gdal_ds.GetRasterBand(1)
    raster_band.WriteArray(input_array)

    if description is not None:
        gdal_ds.SetDescription(description)

    gdal_ds.FlushCache()
    gdal_ds = None

    if output_files_list is not None:
        output_files_list.append(output_file)
    logger.info(f'file saved: {output_file}')

def _makedirs(input_file):
    output_dir = os.path.dirname(input_file)
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)


def _save_output_rgb_file(red, green, blue, output_file,
                          offset_dict, scale_dict, 
                          flag_offset_and_scale_inputs,
                          geotransform, projection,
                          invalid_ind = None, output_files_list = None,
                          flag_infrared = False):
    _makedirs(output_file)
    shape = blue.shape
    driver = gdal.GetDriverByName("GTiff")
    gdal_dtype = GDT_Float32
    gdal_ds = driver.Create(output_file, shape[1], shape[0], 3, gdal_dtype)
    gdal_ds.SetGeoTransform(geotransform)
    gdal_ds.SetProjection(projection)

    # HLS images were not yet corrected for offset and scale factor
    if not flag_offset_and_scale_inputs:

        if not flag_infrared:
            red_key = 'red'
            green_key = 'green'
            blue_key = 'blue'
        else:
            red_key = 'swir1'
            green_key = 'nir'
            blue_key = 'red'

        red = scale_dict[red_key] * (np.asarray(red, dtype=np.float32) -
                                   offset_dict[red_key])

        green = scale_dict[green_key] * (np.asarray(green, dtype=np.float32) -
                                       offset_dict[green_key])

        blue = scale_dict[blue_key] * (np.asarray(blue, dtype=np.float32) -
                                     offset_dict[blue_key])

    if invalid_ind is not None:
        red[invalid_ind] = np.nan
        green[invalid_ind] = np.nan
        blue[invalid_ind] = np.nan

    # Save red band
    gdal_ds.GetRasterBand(1).WriteArray(red)

    # Save green band
    gdal_ds.GetRasterBand(2).WriteArray(green)

    # Save blue band
    gdal_ds.GetRasterBand(3).WriteArray(blue)

    gdal_ds.FlushCache()
    gdal_ds = None

    if output_files_list is not None:
        output_files_list.append(output_file)
    logger.info(f'file saved: {output_file}')


def get_projection_proj4(projection):
    srs = osr.SpatialReference()
    if projection.upper() == 'WGS84':
        srs.SetWellKnownGeogCS(projection)
    else:
        srs.ImportFromProj4(projection)
    projection = srs.ExportToProj4()
    projection = projection.strip()
    return projection


def _relocate(input_file, geotransform, projection,
              length, width,
              resample_algorithm='nearest',
              relocated_file=None):

    logger.info(f'relocating file: {input_file}')

    dy = geotransform[5]
    dx = geotransform[1]
    y0 = geotransform[3]
    x0 = geotransform[0]

    ds = gdal.Open(input_file)
    xf = x0 + width * dx
    yf = y0 + length * dy

    dstSRS = get_projection_proj4(projection)

    if relocated_file is None:
        relocated_file = tempfile.NamedTemporaryFile(
                    dir='.', suffix='.tif').name
        logger.info(f'temporary file: {relocated_file}')
    else:
        logger.info(f'relocated file: {relocated_file}')

    _makedirs(relocated_file)

    gdal.Warp(relocated_file, input_file, format='GTiff',
              dstSRS=dstSRS,
              outputBounds=[x0, yf, xf, y0], multithread=True,
              xRes=dx, yRes=abs(dy), resampleAlg=resample_algorithm,
              errorThreshold=0)

    gdal_ds = gdal.Open(relocated_file)
    relocated_array = gdal_ds.ReadAsArray()
    # gdal_ds.SetProjection(projection)
    # gdal_ds.FlushCache()
    del gdal_ds

    return relocated_array

def _deep_update(original, update):
    '''
    update default runconfig key with user supplied dict
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    '''
    for key, val in update.items():
        if isinstance(val, dict):
            original[key] = _deep_update(original.get(key, {}), val)
        else:
            original[key] = val

    # return updated original
    return original


def _generate_dswx_layers_runconfig(runconfig_file, args):
    if not os.path.isfile(runconfig_file):
        logger.info(f'ERROR invalid file {runconfig_file}')
        return

    logger.info(f'Input runconfig file: {runconfig_file}')

    bin_dirname = os.path.dirname(__file__)
    source_dirname = os.path.split(bin_dirname)[0]
    default_runconfig_file = f'{source_dirname}/schemas/dswx_hls.yaml'

    logger.info(f'Default runconfig file: {default_runconfig_file}')

    yaml_schema = f'{source_dirname}/schemas/dswx_hls.yaml'
    logger.info(f'YAML schema: {yaml_schema}')

    schema = yamale.make_schema(yaml_schema, parser='ruamel')
    data = yamale.make_data(runconfig_file, parser='ruamel')

    logger.info(f'Validating runconfig file: {runconfig_file}')
    yamale.validate(schema, data)

    # parse default config
    parser = ruamel_yaml(typ='safe')
    with open(default_runconfig_file, 'r') as f:
        default_runconfig = parser.load(f)

    # parse user config
    with open(runconfig_file) as f_yaml:
        user_runconfig = parser.load(f_yaml)

    # copy user suppiled config into default config
    _deep_update(default_runconfig, user_runconfig)

    # copy runconfig parameters from dictionary
    input_file_path = user_runconfig['runconfig']['groups']['input_file_group'][
        'input_file_path']

    ancillary_ds_group = user_runconfig['runconfig']['groups'][
        'dynamic_ancillary_file_group']

    if 'dem_file' not in ancillary_ds_group:
        dem_file = None
    else:
        dem_file = ancillary_ds_group['dem_file']

    if 'landcover_file' not in ancillary_ds_group:
        landcover_file = None
    else:
        landcover_file = ancillary_ds_group['landcover_file']
    
    if 'built_up_cover_fraction_file' not in ancillary_ds_group:
        built_up_cover_fraction_file = None
    else:
        built_up_cover_fraction_file = ancillary_ds_group[
            'built_up_cover_fraction_file']

    output_file = user_runconfig['runconfig']['groups']['product_path_group'][
        'sas_output_file']

    scratch_dir = user_runconfig['runconfig']['groups']['product_path_group'][
        'scratch_path']

    if len(input_file_path) == 1 and os.path.isdir(input_file_path[0]):
        logger.info(f'input HLS files directory: {input_file_path[0]}')
        input_list = glob.glob(os.path.join(input_file_path[0], '*.tif'))
    else:
        input_list = input_file_path

    # print main runconfig parameters
    args.input_list = input_list

    if args.output_file is not None and output_file is not None:
        logger.warning(f'command line output file "{args.output_file}"'
              f' has precedence over runconfig output file "{output_file}"')
    elif args.output_file is None:
        args.output_file = output_file
 
    if args.dem_file is not None and dem_file is not None:
        logger.warning(f'command line output file "{args.dem_file}"'
              f' has precedence over runconfig output file "{dem_file}"')
    elif args.dem_file is None:
        args.dem_file = dem_file
 
    if args.landcover_file is not None and landcover_file is not None:
        logger.warning(f'command line output file "{args.landcover_file}"'
              f' has precedence over runconfig output file "{landcover_file}"')
    elif args.landcover_file is None:
        args.landcover_file = landcover_file
 
    if args.built_up_cover_fraction_file is not None and built_up_cover_fraction_file is not None:
        logger.warning(f'command line output file "{args.built_up_cover_fraction_file}"'
              f' has precedence over runconfig output file "{built_up_cover_fraction_file}"')
    elif args.built_up_cover_fraction_file is None:
        args.built_up_cover_fraction_file = built_up_cover_fraction_file

    if args.scratch_dir is not None and scratch_dir is not None:
        logger.warning(f'command line output file "{args.scratch_dir}"'
              f' has precedence over runconfig output file "{scratch_dir}"')
    elif args.scratch_dir is None:
        args.scratch_dir = scratch_dir


def _get_dswx_metadata_dict(output_file):

    dswx_metadata_dict = OrderedDict()

    # identification
    product_id = os.path.splitext(os.path.basename(output_file))[0]
    dswx_metadata_dict['PRODUCT_ID'] = product_id
    dswx_metadata_dict['PRODUCT_VERSION'] = '0.1'
    dswx_metadata_dict['PROJECT'] = 'OPERA'
    dswx_metadata_dict['LEVEL'] = '3'
    dswx_metadata_dict['PRODUCT_TYPE'] = 'DSWx'
    dswx_metadata_dict['PRODUCT_SOURCE'] = 'HLS'

    # save datetime 'YYYY-MM-DD HH:MM:SS'
    dswx_metadata_dict['PROCESSING_DATETIME'] = \
        datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    return dswx_metadata_dict

def _populate_dswx_metadata_datasets(dswx_metadata_dict, hls_dataset,
                                     dem_file=None, landcover_file=None,
                                     built_up_cover_fraction_file=None):

    # input datasets
    dswx_metadata_dict['HLS_DATASET'] = hls_dataset
    dswx_metadata_dict['DEM_FILE'] = dem_file if dem_file else '(not provided)'
    dswx_metadata_dict['LANDCOVER_FILE'] = \
        landcover_file if landcover_file else '(not provided)'
    dswx_metadata_dict['BUILT_UP_COVER_FRACTION_FILE'] = \
        built_up_cover_fraction_file if built_up_cover_fraction_file \
                                     else '(not provided)'


def configure_log_file(log_file):
    # create logger
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    if log_file:
        file_handler = logging.FileHandler(log_file)

        # Log file format
        msgfmt = ('%(asctime)s.%(msecs)03d, %(levelname)s, DSWx-HLS, '
                  '%(module)s, 999999, %(pathname)s:%(lineno)d, "%(message)s"')

        log_file_formatter = logging.Formatter(msgfmt, "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(log_file_formatter)

        # add file handler to logger
        logger.addHandler(file_handler)

    return logger


def _compute_hillshade(dem_file, scratch_dir, sun_azimuth_angle,
                      sun_elevation_angle):
    shadow_layer_file = tempfile.NamedTemporaryFile(
        dir=scratch_dir, suffix='.tif').name

    gdal.DEMProcessing(shadow_layer_file, dem_file, "hillshade",
                      azimuth=sun_azimuth_angle,
                      altitude=sun_elevation_angle)
    gdal_ds = gdal.Open(shadow_layer_file)
    shadow_layer = gdal_ds.ReadAsArray()
    del gdal_ds
    return shadow_layer


def _apply_shadow_layer(masked_dswx_band, shadow_layer):
    # shadows are set to 0 (not water)
    ind = np.where(shadow_layer == 1)
    masked_dswx_band[ind] = 0
    return masked_dswx_band


# @profile
def generate_dswx_layers(input_list, output_file,
                         dem_file=None,
                         output_interpreted_band=None,
                         output_rgb_file=None,
                         output_infrared_rgb_file=None,
                         output_binary_water=None,
                         output_diagnostic_test_band=None,
                         output_non_masked_dswx=None,
                         output_shadow_masked_dswx=None,
                         output_shadow_layer=None,
                         output_cloud_mask=None,
                         landcover_file=None,
                         built_up_cover_fraction_file=None,
                         flag_offset_and_scale_inputs=False,
                         scratch_dir='.',
                         flag_debug=False):

    if scratch_dir is None:
        scratch_dir = '.'
 
    logger.info('input parameters:')
    logger.info('    file(s):')
    for input_file in input_list:
        logger.info(f'        {input_file}')
    logger.info(f'    output_file: {output_file}')
    logger.info(f'    DEM file: {dem_file}')
    logger.info(f'    scratch directory: {scratch_dir}')

    image_dict = {}
    offset_dict = {}
    scale_dict = {}
    output_files_list = []
    dem = None
    shadow_layer = None

    dswx_metadata_dict = _get_dswx_metadata_dict(output_file)

    version = None
    if not isinstance(input_list, list) or len(input_list) == 1:
        success = _load_hls_product_v1(input_list, image_dict, offset_dict,
                                       scale_dict, dswx_metadata_dict,
                                       flag_offset_and_scale_inputs,
                                       flag_debug = flag_debug)
        if success:
            version = '1.4'
    else:
        success = None

    # If success is None or False:
    if success is not True:
        success = _load_hls_product_v2(input_list, image_dict, offset_dict,
                                       scale_dict, dswx_metadata_dict,
                                       flag_offset_and_scale_inputs,
                                       flag_debug = flag_debug)
        if not success:
            logger.info(f'ERROR could not read file(s): {input_list}')
            return False
        version = '2.0'

    hls_dataset_name = image_dict['hls_dataset_name']
    _populate_dswx_metadata_datasets(dswx_metadata_dict, hls_dataset_name,
        dem_file=None, landcover_file=None, built_up_cover_fraction_file=None)

    spacecraft_name = dswx_metadata_dict['SPACECRAFT_NAME']
    logger.info(f'processing HLS {spacecraft_name[0]}30 dataset v.{version}')
    blue = image_dict['blue'] 
    green = image_dict['green'] 
    red = image_dict['red']
    nir = image_dict['nir'] 
    swir1 = image_dict['swir1'] 
    swir2 = image_dict['swir2']
    qa = image_dict['qa']

    geotransform = image_dict['geotransform']
    projection = image_dict['projection']
    fill_data = image_dict['fill_data']
    length = image_dict['length']
    width = image_dict['width']

    sun_azimuth_angle_meta = dswx_metadata_dict['MEAN_SUN_AZIMUTH_ANGLE'].split(', ')
    sun_zenith_angle_meta = dswx_metadata_dict['MEAN_SUN_ZENITH_ANGLE'].split(', ')

    if len(sun_azimuth_angle_meta) == 2:
        sun_azimuth_angle = (float(sun_azimuth_angle_meta[0]) + 
                            float(sun_azimuth_angle_meta[1])) / 2.0
    else:
        sun_azimuth_angle = float(sun_azimuth_angle_meta[0])
    if len(sun_zenith_angle_meta) == 2:
        sun_zenith_angle = (float(sun_zenith_angle_meta[0]) + 
                            float(sun_zenith_angle_meta[1])) / 2.0
    else:
        sun_zenith_angle = float(sun_zenith_angle_meta[0])

    # Sun elevation and zenith angles are complementary
    sun_elevation_angle = 90 - float(sun_zenith_angle)

    print('Mean Sun azimuth angle:', sun_azimuth_angle)
    print('Mean Sun elevation angle:', sun_elevation_angle)

    if dem_file is not None:
        # DEM
        dem_cropped_file = 'temp_dem.tif'
        dem = _relocate(dem_file, geotransform, projection,
                        length, width,
                        resample_algorithm='cubic',
                        relocated_file=dem_cropped_file)
        hillshade = _compute_hillshade(dem_cropped_file, scratch_dir,
                                         sun_azimuth_angle, sun_elevation_angle)
        shadow_layer = _compute_otsu_threshold(hillshade, is_normalized = True)

        if output_shadow_layer:
            _save_array(shadow_layer, output_shadow_layer,
                        dswx_metadata_dict, geotransform, projection,
                        description=band_description_dict['SHAD'],
                        output_files_list=output_files_list)

    if landcover_file is not None:
        # Land Cover
        landcover = _relocate(landcover_file, geotransform, projection,
                              length, width, 
                              relocated_file='temp_landcover.tif')

    if built_up_cover_fraction_file is not None:
        # Build-up cover fraction
        built_up_cover_fraction = _relocate(built_up_cover_fraction_file, 
                                            geotransform, projection,
                                            length, width,
                                            relocated_file =
                                            'temp_built_up_cover_fraction.tif')

    # Set invalid pixels to fill value (255)
    if not flag_offset_and_scale_inputs:
        invalid_ind = np.where(blue < -5000)
    else:
        invalid_ind = np.where(blue < -0.5)

    if output_rgb_file:
        _save_output_rgb_file(red, green, blue, output_rgb_file,
                              offset_dict, scale_dict,
                              flag_offset_and_scale_inputs,
                              geotransform, projection,
                              invalid_ind=invalid_ind,
                              output_files_list=output_files_list)
    
    if output_infrared_rgb_file:
        _save_output_rgb_file(swir1, nir, red, output_infrared_rgb_file,
                              offset_dict, scale_dict,
                              flag_offset_and_scale_inputs,
                              geotransform, projection,
                              invalid_ind=invalid_ind,
                              output_files_list=output_files_list,
                              flag_infrared=True)

    diagnostic_test_band = _compute_diagnostic_tests(
        blue, green, red, nir, swir1, swir2)

    if output_diagnostic_test_band:
        _save_array(diagnostic_test_band, output_diagnostic_test_band,
                    dswx_metadata_dict, geotransform, projection,
                    description=band_description_dict['DIAG'],
                    output_files_list=output_files_list)

    interpreted_dswx_band = _generate_interpreted_layer(
        diagnostic_test_band)

    if invalid_ind is not None:
        interpreted_dswx_band[invalid_ind] = 255

    if output_non_masked_dswx:
        save_dswx_product(interpreted_dswx_band,
                          output_non_masked_dswx,
                          dswx_metadata_dict,
                          geotransform, 
                          projection,
                          description=band_description_dict['INTR'],
                          scratch_dir=scratch_dir,
                          output_files_list=output_files_list)

    if shadow_layer is not None:
        shadow_masked_dswx = _apply_shadow_layer(
            interpreted_dswx_band, shadow_layer)
    else:
        shadow_masked_dswx = interpreted_dswx_band

    if output_shadow_masked_dswx is not None:
        save_dswx_product(shadow_masked_dswx, output_shadow_masked_dswx,
                          dswx_metadata_dict,
                          geotransform, 
                          projection,
                          description=band_description_dict['INSM'],
                          scratch_dir=scratch_dir,
                          output_files_list=output_files_list)

    cloud, masked_dswx_band = _compute_mask_and_filter_interpreted_layer(
        shadow_masked_dswx, qa)

    if invalid_ind is not None:
        # Set invalid pixels to mask fill value (255)
        cloud[invalid_ind] = 255
        masked_dswx_band[invalid_ind] = 255

    if output_interpreted_band:
        save_dswx_product(masked_dswx_band,
                          output_interpreted_band,
                          dswx_metadata_dict,
                          geotransform, 
                          projection,
                          description=band_description_dict['WTR'],
                          scratch_dir=scratch_dir,
                          output_files_list=output_files_list)

    if output_cloud_mask:
        save_mask(cloud, output_cloud_mask, dswx_metadata_dict, geotransform,
                  projection,
                  description=band_description_dict['CLOUD'],
                  output_files_list=output_files_list)

    binary_water_layer = _get_binary_water_layer(masked_dswx_band)
    if output_binary_water:
        _save_binary_water(binary_water_layer, output_binary_water,
                           dswx_metadata_dict,
                           geotransform, projection,
                           description=band_description_dict['BWTR'],
                           output_files_list=output_files_list)

    save_dswx_product(masked_dswx_band,
                      output_file,
                      dswx_metadata_dict,
                      geotransform, 
                      projection,
                      bwtr=binary_water_layer,
                      diag=diagnostic_test_band,
                      intr=interpreted_dswx_band,
                      insm=shadow_masked_dswx,
                      shad=shadow_layer,
                      cloud=cloud,
                      dem=dem,
                      scratch_dir=scratch_dir,
                      output_files_list=output_files_list)

    logger.info('list of output files:')
    for filename in output_files_list:
        logger.info(filename)
    
    return True

# @profile
def main():
    parser = _get_parser()

    args = parser.parse_args()

    configure_log_file(args.log_file)

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_list[0])[0]

    if len(args.input_list) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return
 
    if flag_first_file_is_text:
        _generate_dswx_layers_runconfig(args.input_list[0], args)


    generate_dswx_layers(
        args.input_list,
        args.output_file,
        dem_file=args.dem_file, 
        output_interpreted_band=args.output_interpreted_band,
        output_rgb_file=args.output_rgb_file,
        output_infrared_rgb_file=args.output_infrared_rgb_file,
        output_binary_water=args.output_binary_water,
        output_diagnostic_test_band=args.output_diagnostic_test_band,
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