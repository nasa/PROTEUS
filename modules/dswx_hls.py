from osgeo import gdal

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
    # 0b255 :  # 255 (Fill)

band_description_dict = {
    'WTR': 'Water classification (WTR)',
    'BWTR': 'Binary Water (BWTR)',
    'CONF': 'TBD Confidence (CONF)',
    'DIAG': 'Diagnostic layer (DIAG)',
    'INTR': 'Interpretation of diagnostic layer into water classes (INTR)',
    'INSM': 'Interpreted layer refined using land cover and terrain shadow testing (INSM)',
    'LAND': 'Land cover classification (LAND)',
    'SHAD': 'Terrain shadow layer (SHAD)',
    'CLOUD': 'Cloud/cloud-shadow classification (CLOUD)',
    'DEM': 'Digital elevation model (DEM)'}


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

def get_interpreted_dswx_ctable():
    """Create and return GDAL color table for DSWx-HLS 
       surface water interpreted layers.

      Returns
      -------
      dswx_ctable : GDAL ColorTable object
           GDAL color table for DSWx-HLS surface water interpreted layers
     
    """
    # create color table
    dswx_ctable = gdal.ColorTable()

    # set color for each value
    dswx_ctable.SetColorEntry(0, (255, 255, 255))  # White - Not water
    dswx_ctable.SetColorEntry(1, (0, 0, 255))  # Blue - Water (high confidence)
    dswx_ctable.SetColorEntry(2, (64, 64, 255))  # Light blue - Water (moderate conf.)
    dswx_ctable.SetColorEntry(3, (0, 255, 0))  # Green - Potential wetland
    dswx_ctable.SetColorEntry(4, (0, 255, 255))  # Cyan - Low confidence 
                                                 # water or wetland
    dswx_ctable.SetColorEntry(9, (128, 128, 128))  # Gray - QA masked
    dswx_ctable.SetColorEntry(255, (0, 0, 0, 255))  # Black - Fill value
    return dswx_ctable


def get_mask_ctable():
    """Create and return GDAL color table for HLS Q/A mask.

      Returns
      -------
      dswx_ctable : GDAL ColorTable object
           GDAL color table for HLS Q/A mask.
     
    """

    # create color table
    mask_ctable = gdal.ColorTable()

    '''
    set color for each value
    - Mask cloud shadow bit (0)
    - Mask snow/ice bit (1)
    - Mask cloud bit (2)
    '''

    mask_ctable.SetColorEntry(0, (255, 255, 255))  # White - Not masked
    mask_ctable.SetColorEntry(1, (64, 64, 64))  # Dark gray - Cloud shadow
    mask_ctable.SetColorEntry(2, (0, 255, 255))  # Cyan - snow/ice
    mask_ctable.SetColorEntry(3, (0, 0, 255))  # Blue - Cloud shadow and snow/ice
    mask_ctable.SetColorEntry(4, (192, 192, 192))  # Light gray - Cloud
    mask_ctable.SetColorEntry(5, (128, 128, 128))  # Gray - Cloud and cloud shadow
    mask_ctable.SetColorEntry(6, (255, 0, 255))  # Magenta - Cloud and snow/ice
    mask_ctable.SetColorEntry(7, (128, 128, 255))  # Light blue - Cloud, cloud shadow, and snow/ice
    mask_ctable.SetColorEntry(255, (0, 0, 0, 255))  # Black - Fill value
    return mask_ctable

