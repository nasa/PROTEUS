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
                              'PREDICTOR=2',
                              'COPY_SRC_OVERVIEWS=YES']
    gdal.Translate(temp_file, filename,
                   creationOptions=gdal_translate_options)

    shutil.move(temp_file, filename)

    logger.info('COG step 3: validate')
    try:
        from proteus.extern.validate_cloud_optimized_geotiff import main as validate_cog
    except ModuleNotFoundError:
        logger.info('WARNING could not import module validate_cloud_optimized_geotiff')
        return

    argv = ['--full-check=yes', filename]
    validate_cog_ret = validate_cog(argv)
    if validate_cog_ret == 0:
        logger.info(f'file "{filename}" is a valid cloud optimized'
                    ' GeoTIFF')
    else:
        logger.warning(f'file "{filename}" is NOT a valid cloud'
                       f' optimized GeoTIFF!')

