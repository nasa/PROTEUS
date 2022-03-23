import logging
import tempfile
import os
import glob
import numpy as np
import argparse
import yamale
from datetime import datetime
from collections import OrderedDict
from ruamel.yaml import YAML as ruamel_yaml
from osgeo.gdalconst import GDT_Float32
from osgeo import gdal, osr
from proteus.core import save_as_cog

PRODUCT_VERSION = '0.1'

logger = logging.getLogger('dswx_hls')

l30_v1_band_dict = {'blue': 'band02',
                    'green': 'band03',
                    'red': 'band04',
                    'nir': 'band05',
                    'swir1': 'band06',
                    'swir2': 'band07',
                    'qa': 'QA'}

s30_v1_band_dict = {'blue': 'band02',
                    'green': 'band03',
                    'red': 'band04',
                    'nir': 'band8A',
                    'swir1': 'band11',
                    'swir2': 'band12',
                    'qa': 'QA'}

l30_v2_band_dict = {'blue': 'B02',
                    'green': 'B03',
                    'red': 'B04',
                    'nir': 'B05',
                    'swir1': 'B06',
                    'swir2': 'B07',
                    'qa': 'Fmask'}

s30_v2_band_dict = {'blue': 'B02',
                    'green': 'B03',
                    'red': 'B04',
                    'nir': 'B8A',
                    'swir1': 'B11',
                    'swir2': 'B12',
                    'qa': 'Fmask'}

interpreted_dswx_band_dict = {
    0b00000 : 0,  # (Not Water)
    0b00001 : 0,
    0b00010 : 0,
    0b00100 : 0,
    0b01000 : 0,
    0b01111 : 1,  # (Water - High Confidence)
    0b10111 : 1,
    0b11011 : 1,
    0b11101 : 1,
    0b11110 : 1,
    0b11111 : 1,
    0b00111 : 2,  # (Water - Moderate Confidence)
    0b01011 : 2,
    0b01101 : 2,
    0b01110 : 2,
    0b10011 : 2,
    0b10101 : 2,
    0b10110 : 2,
    0b11001 : 2,
    0b11010 : 2,
    0b11100 : 2,
    0b11000 : 3,  # (Potential Wetland)
    0b00011 : 4,  #(Low Confidence Water or Wetland)
    0b00101 : 4,
    0b00110 : 4,
    0b01001 : 4,
    0b01010 : 4,
    0b01100 : 4,
    0b10000 : 4,
    0b10001 : 4,
    0b10010 : 4,
    0b10100 : 4}

band_description_dict = {
    'WTR': 'Water classification (WTR)',
    'BWTR': 'Binary Water (BWTR)',
    'CONF': 'TBD Confidence (CONF)',
    'DIAG': 'Diagnostic layer (DIAG)',
    'WTR-1': 'Interpretation of diagnostic layer into water classes (WTR-1)',
    'WTR-2': 'Interpreted layer refined using land cover and terrain shadow testing (WTR-2)',
    'LAND': 'Land cover classification (LAND)',
    'SHAD': 'Terrain shadow layer (SHAD)',
    'CLOUD': 'Cloud/cloud-shadow classification (CLOUD)',
    'DEM': 'Digital elevation model (DEM)'}

layer_names_to_args_dict = {
    'WTR': 'output_interpreted_band',
    'BWTR': 'output_binary_water',
    'CONF': 'output_confidence_layer',
    'DIAG': 'output_diagnostic_layer',
    'WTR-1': 'output_non_masked_dswx',
    'WTR-2': 'output_shadow_masked_dswx',
    'LAND': 'output_landcover',
    'SHAD': 'output_shadow_layer',
    'CLOUD': 'output_cloud_mask',
    'DEM': 'output_dem_layer',
    'RGB': 'output_rgb_file',
    'INFRARED_RGB': 'output_infrared_rgb_file'}
 

METADATA_FIELDS_TO_COPY_FROM_HLS_LIST = ['SENSOR_PRODUCT_ID',
                                         'SENSING_TIME',
                                         'SPATIAL_COVERAGE',
                                         'CLOUD_COVERAGE',
                                         'SPATIAL_RESAMPLING_ALG',
                                         'MEAN_SUN_AZIMUTH_ANGLE',
                                         'MEAN_SUN_ZENITH_ANGLE',
                                         'MEAN_VIEW_AZIMUTH_ANGLE',
                                         'MEAN_VIEW_ZENITH_ANGLE',
                                         'NBAR_SOLAR_ZENITH',
                                         'ACCODE',
                                         'IDENTIFIER_PRODUCT_DOI']


class HlsThresholds:
    """
    Placeholder for HLS reflectance thresholds for generating DSWx-HLS products

    Attributes
    ----------
    wigt : float
        Modified Normalized Difference Wetness Index (MNDWI) Threshold
    awgt : float
        Automated Water Extent Shadow Threshold
    pswt_1_mndwi : float
        Partial Surface Water Test-1 MNDWI Threshold
    pswt_1_nir : float
        Partial Surface Water Test-1 NIR Threshold
    pswt_1_swir1 : float
        Partial Surface Water Test-1 SWIR1 Threshold
    pswt_1_ndvi : float
        Partial Surface Water Test-1 NDVI Threshold
    pswt_2_mndwi : float
        Partial Surface Water Test-2 MNDWI Threshold
    pswt_2_blue : float
        Partial Surface Water Test-2 Blue Threshold
    pswt_2_nir : float
        Partial Surface Water Test-2 NIR Threshold
    pswt_2_swir1 : float
        Partial Surface Water Test-2 SWIR1 Threshold
    pswt_2_swir2 : float
        Partial Surface Water Test-2 SWIR2 Threshold
    """
    def __init__(self):

        self.wigt = None
        self.awgt = None
        self.pswt_1_mndwi = None
        self.pswt_1_nir = None
        self.pswt_1_swir1 = None
        self.pswt_1_ndvi = None
        self.pswt_2_mndwi = None
        self.pswt_2_blue = None
        self.pswt_2_nir = None
        self.pswt_2_swir1 = None
        self.pswt_2_swir2 = None


def get_dswx_hls_cli_parser():
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

    parser.add_argument('--conf'
                        '--output-confidence-layer',
                        dest='output_confidence_layer',
                        type=str,
                        help='Output confidence layer (GeoTIFF)')

    parser.add_argument('--diag',
                        '--output-diagnostic-layer',
                        dest='output_diagnostic_layer',
                        type=str,
                        help='Output diagnostic test layer file (GeoTIFF)')

    parser.add_argument('--wtr-1',
                        '--output-non-masked-dswx',
                        dest='output_non_masked_dswx',
                        type=str,
                        help='Output non-masked DSWx layer file (GeoTIFF)')

    parser.add_argument('--wtr-2',
                        '--output-shadow-masked-dswx',
                        dest='output_shadow_masked_dswx',
                        type=str,
                        help='Output GeoTIFF file with interpreted layer'
                        ' refined using land cover and terrain shadow testing')

    parser.add_argument('--land',
                        '--output-land',
                        dest='output_landcover',
                        type=str,
                        help='Output landcover classification file (GeoTIFF)')

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

    parser.add_argument('--out-dem'
                        '--output-digital-elevation-model',
                        '--output-elevation-layer',
                        dest='output_dem_layer',
                        type=str,
                        help='Output elevation layer file (GeoTIFF)')

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

    parser.add_argument('--pid',
                        '--product-id',
                        dest='product_id',
                        type=str,
                        help='Product ID that will be saved in the output'
                        "product's metadata")

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


def _get_prefix_str(flag_same, flag_all_ok):
    flag_all_ok[0] = flag_all_ok[0] and flag_same
    return '[OK]   ' if flag_same else '[FAIL] '


def compare_dswx_hls_products(file_1, file_2):
    if not os.path.isfile(file_1):
        print(f'ERROR file not found: {file_1}')
        return False

    if not os.path.isfile(file_2):
        print(f'ERROR file not found: {file_2}')
        return False

    flag_all_ok = [True]

    # TODO: compare projections ds.GetProjection()
    layer_gdal_dataset_1 = gdal.Open(file_1, gdal.GA_ReadOnly)
    geotransform_1 = layer_gdal_dataset_1.GetGeoTransform()
    metadata_1 = layer_gdal_dataset_1.GetMetadata()
    nbands_1 = layer_gdal_dataset_1.RasterCount

    layer_gdal_dataset_2 = gdal.Open(file_2, gdal.GA_ReadOnly)
    geotransform_2 = layer_gdal_dataset_2.GetGeoTransform()
    metadata_2 = layer_gdal_dataset_2.GetMetadata()
    nbands_2 = layer_gdal_dataset_2.RasterCount

    flag_same_nbands =  nbands_1 == nbands_2
    flag_same_nbands_str = _get_prefix_str(flag_same_nbands, flag_all_ok)
    prefix = ' ' * 7
    print(f'{flag_same_nbands_str}Comparing number of bands')
    if not flag_same_nbands:
        print(prefix + f'Input 1 has {nbands_1} bands and input 2'
              f' has {nbands_2} bands')
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
        flag_bands_are_equal_str = _get_prefix_str(flag_bands_are_equal,
                                                   flag_all_ok)
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
    flag_same_geotransforms_str = _get_prefix_str(flag_same_geotransforms,
                                                  flag_all_ok)
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

    flag_same_metadata_str = _get_prefix_str(flag_same_metadata,
                                             flag_all_ok)
    print(f'{flag_same_metadata_str}Comparing metadata')
    if not flag_same_metadata:
        print(prefix + metadata_error_message)

    return flag_all_ok[0]


def create_landcover_mask(input_file, copernicus_landcover_file,
                          worldcover_file, output_file, scratch_dir):
    """Create landcover mask LAND combining Copernicus Global Land Service
       (CGLS) Land Cover Layers collection 3 at 100m and ESA WorldCover 10m.

       Parameters
       ----------
       input_file : str
            HLS tile to be used as reference for the map (geographic) grid
       copernicus_landcover_file : str
            Copernicus Global Land Service (CGLS) Land Cover Layers
            collection 3 at 100m
       worldcover_file : str
            ESA WorldCover 10m
       output_file : str
            Output landcover mask (LAND layer)
       scratch_dir : str
              Temporary directory"""
    if not os.path.isfile(input_file):
        logger.error(f'ERROR file not found: {input_file}')
        return

    if not os.path.isfile(copernicus_landcover_file):
        logger.error(f'ERROR file not found: {copernicus_landcover_file}')
        return
    
    if not os.path.isfile(worldcover_file):
        logger.error(f'ERROR file not found: {worldcover_file}')
        return

    '''
    TODO review following classes and update code:
    copernicus_landcover_evergreen_classes = [111, 112, 121, 122]
    copernicus_landcover_buit_up_classses = [50]
    copernicus_landcover_mask_classses = \
        (copernicus_landcover_evergreen_classes +
         copernicus_landcover_buit_up_classses)
    '''
    logger.info('')
    logger.info(f'Input file: {input_file}')
    logger.info(f'Copernicus landcover 100 m file: {copernicus_landcover_file}')
    logger.info(f'World cover 10 m file: {worldcover_file}')
    logger.info('')

    layer_gdal_dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    if layer_gdal_dataset is None:
        logger.error(f'ERROR invalid file: {input_file}')
    geotransform = layer_gdal_dataset.GetGeoTransform()
    projection = layer_gdal_dataset.GetProjection()
    length = layer_gdal_dataset.RasterYSize
    width = layer_gdal_dataset.RasterXSize

    _relocate(copernicus_landcover_file, geotransform, projection,
              length, width, scratch_dir, resample_algorithm='nearest',
              relocated_file=output_file)


def _get_interpreted_dswx_ctable():
    """Create and return GDAL RGB color table for DSWx-HLS
       surface water interpreted layers.

       Returns
       -------
       dswx_ctable : GDAL ColorTable object
            GDAL color table for DSWx-HLS surface water interpreted layers

    """
    # create color table
    dswx_ctable = gdal.ColorTable()

    # set color for each value

    # White - Not water
    dswx_ctable.SetColorEntry(0, (255, 255, 255))
    # Blue - Water (high confidence)
    dswx_ctable.SetColorEntry(1, (0, 0, 255))
    # Light blue - Water (moderate conf.)
    dswx_ctable.SetColorEntry(2, (0, 127, 255))
    # Dark green - Potential wetland
    dswx_ctable.SetColorEntry(3, (0, 127, 0))
    # Green - Low confidence water or wetland
    dswx_ctable.SetColorEntry(4, (0, 255, 0))
    # Cyan - Reserved
    dswx_ctable.SetColorEntry(5, (0, 255, 255))
    # Light green - Reserved
    dswx_ctable.SetColorEntry(6, (127, 255, 0))
    # Yellow - Reserved
    dswx_ctable.SetColorEntry(7, (255, 255, 0))
    # Orange - Reserved
    dswx_ctable.SetColorEntry(8, (255, 127, 0))
    # Gray - QA masked
    dswx_ctable.SetColorEntry(9, (127, 127, 127))
    # Black/transparent - Fill value
    dswx_ctable.SetColorEntry(255, (0, 0, 0, 0))

    return dswx_ctable


def _get_mask_ctable():
    """Create and return GDAL RGB color table for HLS Q/A mask.

       Returns
       -------
       dswx_ctable : GDAL ColorTable object
            GDAL color table for HLS Q/A mask.
     
    """
    # create color table
    mask_ctable = gdal.ColorTable()

    # set color for each value
    # - Mask cloud shadow bit (0)
    # - Mask snow/ice bit (1)
    # - Mask cloud bit (2)

    # White - Not masked
    mask_ctable.SetColorEntry(0, (255, 255, 255))
    # Dark gray - Cloud shadow
    mask_ctable.SetColorEntry(1, (64, 64, 64))
    # Cyan - snow/ice
    mask_ctable.SetColorEntry(2, (0, 255, 255))
    # Blue - Cloud shadow and snow/ice
    mask_ctable.SetColorEntry(3, (0, 0, 255))
    # Light gray - Cloud
    mask_ctable.SetColorEntry(4, (192, 192, 192))
    # Gray - Cloud and cloud shadow
    mask_ctable.SetColorEntry(5, (127, 127, 127))
    # Magenta - Cloud and snow/ice
    mask_ctable.SetColorEntry(6, (255, 0, 255))
    # Light blue - Cloud, cloud shadow, and snow/ice
    mask_ctable.SetColorEntry(7, (127, 127, 255))
    # Black/transparent - Fill value
    mask_ctable.SetColorEntry(255, (0, 0, 0, 0))
    return mask_ctable


def _compute_otsu_threshold(image, is_normalized = True):
    """Compute Otsu threshold
       source: https://learnopencv.com/otsu-thresholding-with-opencv/

       Parameters
       ----------
       image: numpy.ndarray
              Input image
       is_normalized: bool (optional)
              Flag to inform the function if input image is normalized

       Returns
       -------
       binary_array : numpy.ndarray
            Binary array after thresholding input image with Otsu's threshold
     
    """
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


def generate_interpreted_layer(diagnostic_layer):
    """Generate interpreted layer from diagnostic test band

       Parameters
       ----------
       diagnostic_layer: numpy.ndarray
              Diagnostic test band

       Returns
       -------
       interpreted_layer : numpy.ndarray
            Interpreted layer
     
    """

    logger.info('step 2 - get interpreted DSWX band')
    shape = diagnostic_layer.shape
    interpreted_layer = np.zeros(shape, dtype = np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for key, value in interpreted_dswx_band_dict.items():
                if diagnostic_layer[i, j] == key:
                    interpreted_layer[i, j] = value
                    break
            else:
                interpreted_layer[i, j] = 255

    return interpreted_layer


def _get_binary_water_layer(interpreted_water_layer):
    """Generate binary water layer from interpreted water layer

       Parameters
       ----------
       masked_dswx_band: numpy.ndarray
              Interpreted water layer

       Returns
       -------
       binary_water_layer : numpy.ndarray
            Binary water layer
     
    """
    # fill value: 255
    binary_water_layer = np.full_like(interpreted_water_layer, 255)

    # water classes: 0
    ind = np.where(interpreted_water_layer == 0)
    binary_water_layer[ind] = 0

    # water classes: 1 to 4
    for class_value in range(1, 5):
        ind = np.where(interpreted_water_layer == class_value)
        binary_water_layer[ind] = 1

    # Q/A masked: 9
    ind = np.where(interpreted_water_layer == 9)
    binary_water_layer[ind] = 2

    return binary_water_layer


def _compute_diagnostic_tests(blue, green, red, nir, swir1, swir2,
                              hls_thresholds):
    """Compute diagnost tests over reflectance channels: Blue,
    Green, Red, NIR, SWIR-1, and SWIR-2, and return
    diagnostic test band

       Parameters
       ----------
       blue: numpy.ndarray
              Blue channel
       green: numpy.ndarray
              Green channel
       red: numpy.ndarray
              Red channel
       nir: numpy.ndarray
              Near infrared (NIR) channel
       swir1: numpy.ndarray
              Short-wave infrared 1 (SWIR-1) channel
       swir2: numpy.ndarray
              Short-wave infrared 2 (SWIR-2) channel
       hls_thresholds:
              HLS reflectance thresholds for generating DSWx-HLS products

       Returns
       -------
       diagnostic_layer : numpy.ndarray
            Diagnostic test band
     
    """

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
    diagnostic_layer = np.zeros(shape, dtype = np.uint8)

    logger.info('step 1 - compute diagnostic tests')
    for i in range(shape[0]):
        for j in range(shape[1]):

            # Surface water tests (see [1, 2])

            # Test 1
            if (mndwi[i, j] > hls_thresholds.wigt):
                diagnostic_layer[i, j] += 1

            # Test 2
            if (mbsrv[i, j] > mbsrn[i, j]):
                diagnostic_layer[i, j] += 2

            # Test 3
            if (awesh[i, j] > hls_thresholds.awgt):
                diagnostic_layer[i, j] += 4

            # Test 4
            if (mndwi[i, j] > hls_thresholds.pswt_1_mndwi and
                    swir1[i, j] < hls_thresholds.pswt_1_swir1 and
                    nir[i, j] < hls_thresholds.pswt_1_nir and
                    ndvi[i, j] < hls_thresholds.pswt_1_ndvi):
                diagnostic_layer[i, j] += 8

            # Test 5
            if (mndwi[i, j] > hls_thresholds.pswt_2_mndwi and
                    blue[i, j] < hls_thresholds.pswt_2_blue and
                    swir1[i, j] < hls_thresholds.pswt_2_swir1 and
                    swir2[i, j] < hls_thresholds.pswt_2_swir2 and
                    nir[i, j] < hls_thresholds.pswt_2_nir):
                diagnostic_layer[i, j] += 16

    return diagnostic_layer


def _compute_mask_and_filter_interpreted_layer(
        unmasked_interpreted_water_layer, qa_band):
    """Compute cloud/cloud-shadow mask and filter interpreted water layer

       Parameters
       ----------
       unmasked_interpreted_water_layer: numpy.ndarray
              Cloud-unmasked interpreted water layer
       qa_band: numpy ndarray
              HLS Q/A band

       Returns
       -------
       masked_interpreted_water_layer : numpy.ndarray
              Cloud-masked interpreted water layer

    """

    shape = unmasked_interpreted_water_layer.shape
    masked_interpreted_water_layer = unmasked_interpreted_water_layer.copy()
    mask = np.zeros(shape, dtype = np.uint8)

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

    for i in range(shape[0]):
        for j in range(shape[1]):

            # Check QA cloud shadow bit (3) => bit 0
            if np.bitwise_and(2**3, qa_band[i, j]):
                mask[i, j] += 1

            # Check QA adjacent to cloud/cloud shadow bit (2) => bit 0
            # Note: this line differs from original USGS DSWE ADD
            elif np.bitwise_and(2**2, qa_band[i, j]):
                mask[i, j] += 1

            # Check QA snow bit (4) => bit 1
            if np.bitwise_and(2**4, qa_band[i, j]):
                mask[i, j] += 2

            # Check QA cloud bit (1) => bit 2
            if np.bitwise_and(2**1, qa_band[i, j]):
                mask[i, j] += 4

            if mask[i, j] == 0:
                continue
            
            masked_interpreted_water_layer[i, j] = 9

    return mask, masked_interpreted_water_layer


def _load_hls_from_file(filename, image_dict, offset_dict, scale_dict,
                        dswx_metadata_dict, key,
                        flag_offset_and_scale_inputs, flag_debug = False,
                        band_suffix = None):
    """Load HLS band from file into memory

       Parameters
       ----------
       filename: str
              Filename containing HLS band
       image_dict: dict
              Image dictionary that will store HLS band array
       offset_dict: dict
              Offset dictionary that will store band offset
       scale_dict: dict
              Scale dictionary that will store band scaling factor
       dswx_metadata_dict: dict
              Metadata dictionary that will store band metadata
       key: str
              Name of the band (e.g., "blue", "green", "swir1", etc)
       flag_offset_and_scale_inputs: bool
              Flag to indicate if the band should be offseted and scaled
       flag_debug: bool (optional)
              Flag to indicate if execution is for debug purposes. If so,
              only a subset of the image will be loaded into memory
       band_suffix: str (optional)
              Indicate band suffix that should be removed from file
              name to extract band name

       Returns
       -------
       flag_success : bool
              Flag indicating if band was successfuly loaded into memory

    """

    layer_gdal_dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    if layer_gdal_dataset is None:
        return None

    if 'hls_dataset_name' not in image_dict.keys():
        hls_dataset_name = os.path.splitext(os.path.basename(filename))[0]
        if band_suffix:
            hls_dataset_name = hls_dataset_name.replace(f'.{band_suffix}', '')
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

        # HLS Sentinel metadata contain attribute SPACECRAFT_NAME
        if 'SPACECRAFT_NAME' in metadata:
            spacecraft_name = metadata['SPACECRAFT_NAME'].upper()
            if 'SENTINEL' not in spacecraft_name and 'LANDSAT' not in spacecraft_name:
                logger.info(f'ERROR the platform "{spacecraft_name}" is not supported')
                return False

        # HLS Landsat metadata contain attribute SENSOR
        elif 'SENSOR' in metadata:
            sensor = metadata['SENSOR']
            if 'OLI' in sensor:
                spacecraft_name = 'Landsat-8'
            else:
                logger.info(f'ERROR the sensor "{sensor}" is not supported')
                return False

        # Otherwise, could not find HLS Sentinel or Landsat metadata
        else:
            logger.info('ERROR could not determine the platorm from metadata')
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
        image_dict['length'] = image_dict[key].shape[0]
        image_dict['width'] = image_dict[key].shape[1]

    return True


def _load_hls_product_v1(filename, image_dict, offset_dict,
                         scale_dict, dswx_metadata_dict,
                         flag_offset_and_scale_inputs,
                         flag_debug = False):
    """Load a HLS (v.1) product (all required bands) from file
       into memory

       Parameters
       ----------
       filename: str
              Filename containing HLS product
       image_dict: dict
              Image dictionary that will store HLS product's arrays
       offset_dict: dict
              Offset dictionary that will store product's offsets
       scale_dict: dict
              Scale dictionary that will store product's scaling factor
       dswx_metadata_dict: dict
              Metadata dictionary that will store product's metadata
       flag_offset_and_scale_inputs: bool
              Flag to indicate if bands should be offseted and scaled
       flag_debug (optional)
              Flag to indicate if execution is for debug purposes. If so,
              only a subset of the product will be loaded into memory

       Returns
       -------
       flag_success : bool
              Flag indicating if band was successfuly loaded into memory

    """

    if isinstance(filename, list):
        filename = filename[0]

    logger.info('loading HLS v.1.x layers:')
    for key in l30_v1_band_dict.keys():

        logger.info(f'    {key}')

        # Sensor is undertermined (first band) or LANDSAT
        if ('SPACECRAFT_NAME' not in dswx_metadata_dict.keys() or
                'LANDSAT' in dswx_metadata_dict['SPACECRAFT_NAME'].upper()):
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
    """Load a HLS (v.2) product (all required bands) from a list of files
       into memory

       Parameters
       ----------
       file_list: str
              File list containing HLS product
       image_dict: dict
              Image dictionary that will store HLS product's arrays
       offset_dict: dict
              Offset dictionary that will store product's offsets
       scale_dict: dict
              Scale dictionary that will store product's scaling factor
       dswx_metadata_dict: dict
              Metadata dictionary that will store product's metadata
       flag_offset_and_scale_inputs: bool
              Flag to indicate if bands should be offseted and scaled
       flag_debug (optional)
              Flag to indicate if execution is for debug purposes. If so,
              only a subset of the product will be loaded into memory

       Returns
       -------
       flag_success : bool
              Flag indicating if band was successfuly loaded into memory

    """
    logger.info('loading HLS v.2.0 layers:')
    for key in l30_v2_band_dict.keys():

        logger.info(f'    {key}')

        # Sensor is undertermined (first band) or LANDSAT
        if ('SPACECRAFT_NAME' not in dswx_metadata_dict.keys() or
                'LANDSAT' in dswx_metadata_dict['SPACECRAFT_NAME'].upper()):
            band_name = l30_v2_band_dict[key]
        else:
            band_name = s30_v2_band_dict[key]

        for filename in file_list:
            if band_name + '.tif' in filename:
                break
        else:
            logger.info(f'ERROR band {key} not found within input file(s)')
            return
        success = _load_hls_from_file(filename, image_dict, offset_dict,
                                      scale_dict, dswx_metadata_dict,
                                      key, flag_offset_and_scale_inputs,
                                      flag_debug = flag_debug,
                                      band_suffix = band_name)
        if not success:
            return False

    return True

def _get_binary_water_ctable():
    """Get binary water RGB color table

       Returns
       -------
       binary_water_ctable : gdal.ColorTable
              Binary water RGB color table
    """

    # create color table
    binary_water_ctable = gdal.ColorTable()
    # No water
    binary_water_ctable.SetColorEntry(0, (255, 255, 255))
    # Water
    binary_water_ctable.SetColorEntry(1, (0, 0, 255))
    # Gray - QA masked
    binary_water_ctable.SetColorEntry(2, (127, 127, 127))
    # Black - Fill value
    binary_water_ctable.SetColorEntry(255, (0, 0, 0, 255))
    return binary_water_ctable


def save_dswx_product(wtr, output_file, dswx_metadata_dict, geotransform,
                      projection, scratch_dir='.', output_files_list = None,
                      description = None, **dswx_processed_bands):
    """Save DSWx-HLS product

       Parameters
       ----------
       wtr: numpy.ndarray
              Water classification layer WTR
       output_file: str
              Output filename
       dswx_metadata_dict: dict
              Metadata dictionary to be written into the DSWx-HLS product
       geotransform: numpy.ndarray
              Geotransform describing the DSWx-HLS product geolocation
       projection: str
              DSWx-HLS product's projection
       scratch_dir: str (optional)
              Directory for temporary files
       output_files_list: list (optional)
              Mutable list of output files
       description: str (optional)
              Band description
       **dswx_processed_bands: dict
              Remaining bands to be included into the DSWx-HLS product
    """

    _makedirs(output_file)
    shape = wtr.shape
    driver = gdal.GetDriverByName("GTiff")

    dswx_processed_bands['wtr'] = wtr

    # translate dswx_processed_bands_keys to band_description_dict keys
    # example: wtr_1 to WTR-1
    dswx_processed_bands_keys = list(dswx_processed_bands.keys())
    dswx_processed_band_names_list = []
    for dswx_processed_bands_key in dswx_processed_bands_keys:
        dswx_processed_band_names_list.append(
            dswx_processed_bands_key.upper().replace('_', '-'))

    # check input arrays different than None
    n_valid_bands = 0
    band_description_dict_keys = list(band_description_dict.keys())
    for i, band_name in enumerate(dswx_processed_band_names_list):
        if band_name not in band_description_dict_keys:
            continue
        if dswx_processed_bands[dswx_processed_bands_keys[i]] is None:
            continue
        n_valid_bands += 1

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

    for band_index, (band_name, description_from_dict) in enumerate(
            band_description_dict.items()):
        
        # check if band is in the list of processed bands
        if band_name in dswx_processed_band_names_list:

            # index using processed key from band name (e.g., WTR-1 to wtr_1)
            band_array = dswx_processed_bands[
                band_name.replace('-', '_').lower()]
        else:
            logger.warning(f'layer not found "{band_name}".')
            band_array = None
        
        # if band is not in the list of processed bands or it's None
        if band_array is None:
            band_array = np.zeros_like(wtr)

        gdal_band = gdal_ds.GetRasterBand(band_index + 1)
        gdal_band.WriteArray(band_array)
        gdal_band.SetNoDataValue(255)
        if n_valid_bands == 1:
            # set color table and color interpretation
            dswx_ctable = _get_interpreted_dswx_ctable()
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
    """Save DSWx-HLS cloud/cloud-mask layer

       Parameters
       ----------
       mask: numpy.ndarray
              Cloud/cloud-shadow layer
       output_file: str
              Output filename
       dswx_metadata_dict: dict
              Metadata dictionary to be written into the output file
       geotransform: numpy.ndarray
              Geotransform describing the output file geolocation
       projection: str
              Output file's projection
       description: str (optional)
              Band description
       output_files_list: list (optional)
              Mutable list of output files
    """

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
    mask_ctable = _get_mask_ctable()
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
    """Save DSWx-HLS binary water layer

       Parameters
       ----------
       binary_water_layer: numpy.ndarray
              Binary water layer
       output_file: str
              Output filename
       dswx_metadata_dict: dict
              Metadata dictionary to be written into the output file
       geotransform: numpy.ndarray
              Geotransform describing the output file geolocation
       projection: str
              Output file's projection
       description: str (optional)
              Band description
       output_files_list: list (optional)
              Mutable list of output files
    """
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
    """Save a generic DSWx-HLS layer (e.g., diagnostic layer, shadow layer, etc.)

       Parameters
       ----------
       input_array: numpy.ndarray
              DSWx-HLS layer to be saved
       output_file: str
              Output filename
       dswx_metadata_dict: dict
              Metadata dictionary to be written into the output file
       geotransform: numpy.ndarray
              Geotransform describing the output file geolocation
       projection: str
              Output file's projection
       description: str (optional)
              Band description
       output_files_list: list (optional)
              Mutable list of output files
    """

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
    """Save the a three-band reflectance-layer (RGB or infrared RGB) GeoTIFF

       Parameters
       ----------
       red: numpy.ndarray
              Red reflectance layer
       green: numpy.ndarray
              Green reflectance layer
       blue: numpy.ndarray
              Blue reflectance layer
       output_file: str
              Output filename
       offset_dict: dict
              Offset dictionary that stores band offsets
       scale_dict: dict
              Scale dictionary that stores bands scaling factor
       flag_offset_and_scale_inputs: bool
              Flag to indicate if the band has been already offseted and scaled
       geotransform: numpy.ndarray
              Geotransform describing the output file geolocation
       projection: str
              Output file's projection
       invalid_ind: list
              List of invalid indices to be set to NaN
       output_files_list: list (optional)
              Mutable list of output files
       flag_infrared: bool
              Flag to indicate if layer represents infrared reflectance,
              i.e., Red, NIR, and SWIR-1
    """
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
    """Return projection in proj4 format

       projection : str
              Projection

       Returns
       -------
       projection_proj4 : str
              Projection in proj4 format
    """
    srs = osr.SpatialReference()
    if projection.upper() == 'WGS84':
        srs.SetWellKnownGeogCS(projection)
    else:
        srs.ImportFromProj4(projection)
    projection_proj4 = srs.ExportToProj4()
    projection_proj4 = projection_proj4.strip()
    return projection_proj4


def _relocate(input_file, geotransform, projection,
              length, width, scratch_dir = '.',
              resample_algorithm='nearest',
              relocated_file=None):
    """Relocate/reproject a file (e.g., landcover or DEM) based on geolocation
       defined by a geotransform, output dimensions (length and width)
       and projection

       Parameters
       ----------
       input_file: str
              Input filename
       geotransform: numpy.ndarray
              Geotransform describing the output file geolocation
       projection: str
              Output file's projection
       length: int
              Output length
       width: int
              Output width
       scratch_dir: str (optional)
              Temporary directory
       resample_algorithm: str
              Resample algorithm
       relocated_file: str
              Relocated file (output file)

       Returns
       -------
       relocated_array : numpy.ndarray
              Relocated array
    """
    logger.info(f'relocating file: {input_file}')

    dy = geotransform[5]
    dx = geotransform[1]
    y0 = geotransform[3]
    x0 = geotransform[0]

    xf = x0 + width * dx
    yf = y0 + length * dy

    dstSRS = get_projection_proj4(projection)

    if relocated_file is None:
        relocated_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name
        logger.info(f'temporary file: {relocated_file}')
    else:
        logger.info(f'relocated file: {relocated_file}')

    _makedirs(relocated_file)

    gdal.Warp(relocated_file, input_file, format='GTiff',
              dstSRS=dstSRS,
              outputBounds=[x0, yf, xf, y0], multithread=True,
              xRes=dx, yRes=abs(dy), resampleAlg=resample_algorithm,
              errorThreshold=0)

    gdal_ds = gdal.Open(relocated_file, gdal.GA_ReadOnly)
    relocated_array = gdal_ds.ReadAsArray()
    del gdal_ds

    return relocated_array

def _deep_update(main_dict, update_dict):
    """Update input dictionary with a second (update) dictionary
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

       Parameters
       ----------
       main_dict: dict
              Input dictionary
       update_dict: dict
              Update dictionary
       
       Returns
       -------
       updated_dict : dict
              Updated dictionary
    """
    for key, val in update_dict.items():
        if isinstance(val, dict):
            main_dict[key] = _deep_update(main_dict.get(key, {}), val)
        else:
            main_dict[key] = val

    # return updated main_dict
    return main_dict


def parse_runconfig_file(user_runconfig_file = None, args = None):
    """
    Parse run configuration file updating an argument
    (argparse.Namespace) and an HlsThresholds object
    
       Parameters
       ----------
       user_runconfig_file: str (optional)
              Run configuration (runconfig) filename
       args: argparse.Namespace (optional)
              Argument object
    """

    bin_dirname = os.path.dirname(__file__)
    source_dirname = os.path.split(bin_dirname)[0]
    default_runconfig_file = f'{source_dirname}/proteus/defaults/dswx_hls.yaml'

    logger.info(f'Default runconfig file: {default_runconfig_file}')

    yaml_schema = f'{source_dirname}/proteus/schemas/dswx_hls.yaml'
    logger.info(f'YAML schema: {yaml_schema}')

    schema = yamale.make_schema(yaml_schema, parser='ruamel')

    # parse default config
    parser = ruamel_yaml(typ='safe')
    with open(default_runconfig_file, 'r') as f:
        default_runconfig = parser.load(f)

    if user_runconfig_file is not None:
        if not os.path.isfile(user_runconfig_file):
            logger.info(f'ERROR invalid file {user_runconfig_file}')
            return

        logger.info(f'Input runconfig file: {user_runconfig_file}')

        data = yamale.make_data(user_runconfig_file, parser='ruamel')

        logger.info(f'Validating runconfig file: {user_runconfig_file}')
        yamale.validate(schema, data)

        # parse user config
        with open(user_runconfig_file) as f_yaml:
            user_runconfig = parser.load(f_yaml)

        # copy user suppiled config into default config
        runconfig = _deep_update(default_runconfig, user_runconfig)

    else:
        runconfig = default_runconfig

    hls_thresholds = HlsThresholds()
    hls_thresholds_user = runconfig['runconfig']['groups']['hls_thresholds']

    # copy runconfig parameters from dictionary
    if hls_thresholds_user is not None:
        logger.info('HLS thresholds:')
        for key in hls_thresholds_user.keys():
            logger.info(f'     {key}: {hls_thresholds_user[key]}')
            hls_thresholds.__setattr__(key, hls_thresholds_user[key])

    if args is None:
        return hls_thresholds

    input_file_path = runconfig['runconfig']['groups']['input_file_group'][
        'input_file_path']

    ancillary_ds_group = runconfig['runconfig']['groups'][
        'dynamic_ancillary_file_group']

    product_path_group = runconfig['runconfig']['groups'][
        'product_path_group']

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

    scratch_dir = product_path_group['scratch_path']
    output_directory = product_path_group['output_dir']
    product_id = product_path_group['product_id']

    if (input_file_path is not None and len(input_file_path) == 1 and
            os.path.isdir(input_file_path[0])):
        logger.info(f'input HLS files directory: {input_file_path[0]}')
        input_list = glob.glob(os.path.join(input_file_path[0], '*.tif'))
        args.input_list = input_list
    elif input_file_path is not None:
        input_list = input_file_path
        args.input_list = input_list

    # update args with runconfig parameters listed below
    variables_to_update_dict = {
        'dem_file': dem_file,
        'landcover_file': landcover_file,
        'built_up_cover_fraction_file': built_up_cover_fraction_file,
        'scratch_dir': scratch_dir,
        'product_id': product_id}

    for var_name, runconfig_file in variables_to_update_dict.items():
        user_file = getattr(args, var_name)
        if user_file is not None and runconfig_file is not None:
            logger.warning(f'command line {var_name} "{user_file}"'
                f' has precedence over runconfig {var_name}'
                f' "{runconfig_file}".')
        elif user_file is None:
            setattr(args, var_name, runconfig_file)
 
    # If user runconfig was not provided, return
    if user_runconfig_file is None:
        return hls_thresholds

    # Save layers
    if product_id is None:
        product_id = 'dswx_hls'
    for i, (layer_name, args_name) in \
            enumerate(layer_names_to_args_dict.items()):
        layer_number = i + 1
        layer_var_name = layer_name.lower().replace('-', '_')
        runconfig_field = f'save_{layer_var_name}'

        flag_save_layer = product_path_group[runconfig_field]
        arg_name = layer_names_to_args_dict[layer_name]

        # user (command-line interface) layer filename
        user_layer_file = getattr(args, arg_name)

        # runconfig layer filename
        product_basename = (f'{product_id}_v{PRODUCT_VERSION}_B{layer_number:02}'
                            f'_{layer_name}.tif')
        runconfig_layer_file = os.path.join(output_directory,
                                            product_basename)

        if user_layer_file is not None and flag_save_layer:
            logger.warning(f'command line {arg_name} "{user_layer_file}" has'
                           f' precedence over runconfig {arg_name}'
                           f' "{runconfig_layer_file}".')
            continue

        if user_layer_file is not None or not flag_save_layer:
            continue

        setattr(args, args_name, runconfig_layer_file)

    return hls_thresholds

def _get_dswx_metadata_dict(product_id):
    """Create and return metadata dictionary

       Parameters
       ----------
       output_file: str
              Output filename

       Returns
       -------
       dswx_metadata_dict : collections.OrderedDict
              Metadata dictionary
    """
    dswx_metadata_dict = OrderedDict()

    # identification


    dswx_metadata_dict['PRODUCT_ID'] = product_id
    dswx_metadata_dict['PRODUCT_VERSION'] = PRODUCT_VERSION
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
    """Populate metadata dictionary with input files

       Parameters
       ----------
       dswx_metadata_dict : collections.OrderedDict
              Metadata dictionary
       hls_dataset: str
              HLS dataset name
       dem_file: str
              DEM filename
       landcover_file: str
              Landcover filename
       built_up_cover_fraction_file: str
              Built-up cover fraction filename
    """

    # input datasets
    dswx_metadata_dict['HLS_DATASET'] = hls_dataset
    dswx_metadata_dict['DEM_FILE'] = dem_file if dem_file else '(not provided)'
    dswx_metadata_dict['LANDCOVER_FILE'] = \
        landcover_file if landcover_file else '(not provided)'
    dswx_metadata_dict['BUILT_UP_COVER_FRACTION_FILE'] = \
        built_up_cover_fraction_file if built_up_cover_fraction_file \
                                     else '(not provided)'


def create_logger(log_file):
    """Create logger object for a log file

       Parameters
       ----------
       log_file: str
              Log file

       Returns
       -------
       logger : logging.Logger
              Logger object
    """

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
    """Compute hillshade using GDAL's DEMProcessing() function

       Parameters
       ----------
       dem_file: str
              DEM filename
       scratch_dir: str
              Scratch directory
       sun_azimuth_angle: float
              Sun azimuth angle
       sun_elevation_angle: float
              Sun elevation angle

       Returns
       -------
       hillshade : numpy.ndarray
              Hillshade
    """
    shadow_layer_file = tempfile.NamedTemporaryFile(
        dir=scratch_dir, suffix='.tif').name

    gdal.DEMProcessing(shadow_layer_file, dem_file, "hillshade",
                      azimuth=sun_azimuth_angle,
                      altitude=sun_elevation_angle)
    gdal_ds = gdal.Open(shadow_layer_file, gdal.GA_ReadOnly)
    hillshade = gdal_ds.ReadAsArray()
    del gdal_ds
    return hillshade


def _apply_shadow_layer(interpreted_layer, shadow_layer):
    """Apply shadow layer onto interpreted layer

       Parameters
       ----------
       interpreted_layer: numpy.ndarray
              Interpreted layer
       shadow_layer: numpy.ndarray
              Shadow layer

       Returns
       -------
       shadow_masked_interpreted_layer : numpy.ndarray
              Shadow-masked interpreted layer
    """
    # shadows are set to 0 (not water)
    shadow_masked_interpreted_layer = interpreted_layer.copy()
    ind = np.where(shadow_layer == 1)
    shadow_masked_interpreted_layer[ind] = 0
    return shadow_masked_interpreted_layer


def generate_dswx_layers(input_list, output_file,
                         hls_thresholds = None,
                         dem_file=None,
                         output_interpreted_band=None,
                         output_rgb_file=None,
                         output_infrared_rgb_file=None,
                         output_binary_water=None,
                         output_confidence_layer=None,
                         output_diagnostic_layer=None,
                         output_non_masked_dswx=None,
                         output_shadow_masked_dswx=None,
                         output_landcover=None,
                         output_shadow_layer=None,
                         output_cloud_mask=None,
                         output_dem_layer=None,
                         landcover_file=None,
                         built_up_cover_fraction_file=None,
                         flag_offset_and_scale_inputs=False,
                         scratch_dir='.',
                         product_id=None,
                         flag_debug=False):
    """Apply shadow layer onto interpreted layer

       Parameters
       ----------
       input_list: list
              Input file list
       output_file: str
              Output filename
       hls_thresholds: HlsThresholds (optional)
              HLS reflectance thresholds for generating DSWx-HLS products
       dem_file: str (optional)
              DEM filename
       output_interpreted_band: str (optional)
              Output interpreted band filename
       output_rgb_file: str (optional)
              Output RGB filename
       output_infrared_rgb_file: str (optional)
              Output infrared RGB filename
       output_binary_water: str (optional)
              Output binary water filename
       output_confidence_layer: str (optional)
              Output confidence layer filename
       output_diagnostic_layer: str (optional)
              Output diagnostic layer filename
       output_non_masked_dswx: str (optional)
              Output (non-masked) interpreted layer filename
       output_shadow_masked_dswx: str (optional)
              Output shadow-masked filename
       output_landcover: str (optional)
              Output landcover classification file
       output_shadow_layer: str (optional)
              Output shadow layer filename
       output_cloud_mask: str (optional)
              Output cloud/cloud-shadow mask filename
       output_dem_layer: str (optional)
              Output elevation layer filename
       landcover_file: str (optional)
              Output landcover filename
       built_up_cover_fraction_file: str (optional)
              Output built-up cover fraction filename
       flag_offset_and_scale_inputs: bool (optional)
              Flag indicating if DSWx-HLS should be offsetted and scaled
       scratch_dir: str (optional)
              Temporary directory
       product_id: str (optional)
              Product ID that will be saved in the output' product's
              metadata
       flag_debug: bool (optional)
              Flag to indicate if execution is for debug purposes. If so,
              only a subset of the image will be loaded into memory

       Returns
       -------
       success : bool
              Flag success indicating if execution was successful
    """

    if hls_thresholds is None:
        hls_thresholds = parse_runconfig_file()

    if scratch_dir is None:
        scratch_dir = '.'
 
    logger.info('input parameters:')
    logger.info('    file(s):')
    for input_file in input_list:
        logger.info(f'        {input_file}')
    logger.info(f'    output_file: {output_file}')
    logger.info(f'    DEM file: {dem_file}')
    logger.info(f'    scratch directory: {scratch_dir}')

    os.makedirs(scratch_dir, exist_ok=True)

    image_dict = {}
    offset_dict = {}
    scale_dict = {}
    output_files_list = []
    build_vrt_list = []
    dem = None
    shadow_layer = None

    if product_id is None and output_file:
        product_id = os.path.splitext(os.path.basename(output_file))[0]
    elif product_id is None:
        product_id = 'dswx_hls'

    dswx_metadata_dict = _get_dswx_metadata_dict(product_id)

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

    logger.info(f'Mean Sun azimuth angle: {sun_azimuth_angle}')
    logger.info(f'Mean Sun elevation angle: {sun_elevation_angle}')

    if dem_file is not None:
        # DEM
        if output_dem_layer is None:
            dem_cropped_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name
        else:
            dem_cropped_file = output_dem_layer

        dem = _relocate(dem_file, geotransform, projection,
                        length, width, scratch_dir,
                        resample_algorithm='cubic',
                        relocated_file=dem_cropped_file)

        # TODO:
        #     1. crop DEM with a margin
        #     2. save metadata to DEM layer

        hillshade = _compute_hillshade(dem_cropped_file, scratch_dir,
                                         sun_azimuth_angle, sun_elevation_angle)
        shadow_layer = _compute_otsu_threshold(hillshade, is_normalized = True)

        if output_shadow_layer:
            _save_array(shadow_layer, output_shadow_layer,
                        dswx_metadata_dict, geotransform, projection,
                        description=band_description_dict['SHAD'],
                        output_files_list=build_vrt_list)

    if landcover_file is not None:

        if output_landcover is None:
            relocated_landcover_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name
        else:
            relocated_landcover_file = output_landcover

        # Land Cover
        # TODO output_landcover will be the output of create_landcover_mask()
        landcover = _relocate(landcover_file, geotransform, projection,
                              length, width, scratch_dir,
                              relocated_file=relocated_landcover_file)

    if built_up_cover_fraction_file is not None:
        # Build-up cover fraction
        built_up_cover_fraction = _relocate(built_up_cover_fraction_file,
                                            geotransform, projection,
                                            length, width, scratch_dir,
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

    diagnostic_layer = _compute_diagnostic_tests(
        blue, green, red, nir, swir1, swir2, hls_thresholds)

    if output_diagnostic_layer:
        _save_array(diagnostic_layer, output_diagnostic_layer,
                    dswx_metadata_dict, geotransform, projection,
                    description=band_description_dict['DIAG'],
                    output_files_list=build_vrt_list)

    interpreted_dswx_band = generate_interpreted_layer(diagnostic_layer)

    if invalid_ind is not None:
        interpreted_dswx_band[invalid_ind] = 255

    if output_non_masked_dswx:
        save_dswx_product(interpreted_dswx_band,
                          output_non_masked_dswx,
                          dswx_metadata_dict,
                          geotransform,
                          projection,
                          description=band_description_dict['WTR-1'],
                          scratch_dir=scratch_dir,
                          output_files_list=build_vrt_list)

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
                          description=band_description_dict['WTR-2'],
                          scratch_dir=scratch_dir,
                          output_files_list=build_vrt_list)

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
                          output_files_list=build_vrt_list)

    if output_cloud_mask:
        save_mask(cloud, output_cloud_mask, dswx_metadata_dict, geotransform,
                  projection,
                  description=band_description_dict['CLOUD'],
                  output_files_list=build_vrt_list)

    binary_water_layer = _get_binary_water_layer(masked_dswx_band)
    if output_binary_water:
        _save_binary_water(binary_water_layer, output_binary_water,
                           dswx_metadata_dict,
                           geotransform, projection,
                           description=band_description_dict['BWTR'],
                           output_files_list=build_vrt_list)

    # TODO: fix CONF layer!!!
    if output_confidence_layer:
        _save_binary_water(binary_water_layer, output_confidence_layer,
                           dswx_metadata_dict,
                           geotransform, projection,
                           description=band_description_dict['CONF'],
                           output_files_list=build_vrt_list)

    # save output_file as GeoTIFF
    if output_file and not output_file.endswith('.vrt'):
        save_dswx_product(masked_dswx_band,
                          output_file,
                          dswx_metadata_dict,
                          geotransform, 
                          projection,
                          bwtr=binary_water_layer,
                          diag=diagnostic_layer,
                          wtr_1=interpreted_dswx_band,
                          wtr_2=shadow_masked_dswx,
                          shad=shadow_layer,
                          cloud=cloud,
                          dem=dem,
                          scratch_dir=scratch_dir,
                          output_files_list=output_files_list)

    # save output_file as VRT
    elif output_file:
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest')
        gdal.BuildVRT(output_file, build_vrt_list, options=vrt_options)
        build_vrt_list.append(output_file)
        logger.info(f'file saved: {output_file}')

    logger.info('list of output files:')
    for filename in build_vrt_list + output_files_list:
        logger.info(filename)
    
    return True

