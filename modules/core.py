import os
import shutil
import tempfile
import logging
from osgeo import gdal

def save_as_cog(filename, scratch_dir = '.', logger = None):
    """Save (overwrite) a GeoTIFF file as a cloud-optimized GeoTIFF.
       Parameters
       ----------
       filename: str
              GeoTIFF to be saved as a cloud-optimized GeoTIFF
       scratch_dir: str (optional)
              Temporary Directory
     
    """
    if logger is None:
        logger = logging.getLogger('proteus')

    logger.info('COG step 1: add overviews')
    gdal_ds = gdal.Open(filename, 1)
    gdal_ds.BuildOverviews('NEAREST', [2, 4, 8, 16, 32, 64, 128], gdal.TermProgress_nocb)
    del gdal_ds  # close the dataset (Python object and pointers)
    external_overview_file = filename + '.ovr'
    if os.path.isfile(external_overview_file):
        os.path.remove(external_overview_file)

    logger.info('COG step 2: save as COG')
    temp_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name

    gdal_translate_options = ['TILED=YES',
                              'BLOCKXSIZE=1024',
                              'BLOCKYSIZE=1024',
                              'COMPRESS=DEFLATE',
                              # 'COMPRESS_OVERVIEW=DEFLATE',
                              'PREDICTOR=2',
                              'COPY_SRC_OVERVIEWS=YES',
                              # 'GDAL_TIFF_OVR_BLOCKSIZE=1024'
                              ]
    gdal.Translate(temp_file, filename,
                   creationOptions=gdal_translate_options)

    shutil.move(temp_file, filename)

    logger.info('COG step 3: validate')
    try:
        from extern.validate_cloud_optimized_geotiff import main as validate_cog
    except ModuleNotFoundError:
        logger.info('ERROR could not import module validate_cloud_optimized_geotiff')
        return

    argv = ['--full-check=yes', filename]
    validate_cog_ret = validate_cog(argv)
    if validate_cog_ret == 0:
        logger.info(f'file "{filename}" is a valid cloud optimized'
                    ' GeoTIFF')
    else:
        logger.warning(f'file "{filename}" is NOT a valid cloud'
                       f' optimized GeoTIFF!')


def compute_otsu_threshold(image, is_normalized = True):
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

