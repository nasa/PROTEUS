import os
import glob
import argparse
import logging
import datetime
import nasa_hls
import numpy as np

from dswx_hls import generate_dswx_layers, configure_log_file

logger = logging.getLogger('dswx_hls')


def _get_parser():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs
    parser.add_argument('--dem',    
                        dest='dem_file',
                        type=str,
                        help='Input digital elevation model (DEM)')

    parser.add_argument('--landcover',
                        dest='landcover_file',
                        type=str,
                        help='Input Land Cover Discrete-Classification-map')

    parser.add_argument('--build-up-cover-fraction',
                        '--builtup-cover-fraction',    
                        dest='built_up_cover_fraction_file',
                        type=str,
                        help='Input built-up cover fraction layer')

    # Outputs
    parser.add_argument('-o',
                        '--output-dswx',
                        dest='output_dswx_file',
                        type=str,
                        help='Output file')
   
    parser.add_argument('--wtr',
                        '--interpreted-band',
                        dest='output_interpreted_band',
                        type=str,
                        help='Output interpreted DSWx layer')

    parser.add_argument('--output-rgb',
                        '--output-rgb-file',
                        dest='output_rgb_file',
                        type=str,
                        help='Output RGB file')

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
                        help='Output non-masked DSWx layer')

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
                        '--output-mask',
                        dest='output_cloud_mask',
                        type=str,
                        help='Output cloud/cloud-shadow classification file')

    # Parameters
    parser.add_argument('--download-dir',
                        dest='download_dir',
                        type=str,
                        default='downloads',
                        help='Download directory')

    parser.add_argument('-y',
                        '--year',
                        dest='year',
                        type=int,
                        help='Acquisition year',
                        required=True)

    parser.add_argument('-m',
                        '--month',
                        dest='month',
                        type=int,
                        help='Acquisition month')

    parser.add_argument('-d',
                        '--day',
                        dest='day',
                        type=int,
                        help='Acquisition day')

    parser.add_argument('--day-year',
                        '--day-of-the-year',
                        dest='day_of_the_year',
                        type=int,
                        help='Acquisition day of the year')

    # https://hls.gsfc.nasa.gov/wp-content/uploads/2016/03/MGRS_GZD-1.png
    # e.g. 32UNU
    parser.add_argument('-t',
                        '--tile',
                        dest='tile',
                        type=str,
                        help='Tile name to download/process',
                        required=True)

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
                        help='Scratch directory')

    parser.add_argument('--log',
                        '--log-file',
                        dest='log_file',
                        type=str,
                        help='Log file')

    return parser


def _download_dataset(args, year, month, day, day_of_the_year):

    if day_of_the_year is None and month is None and day is None:
        logger.info('ERROR please provide day and month or day of the year')
        return

    if day_of_the_year is None:
        user_datetime = datetime.datetime(year, month, day)
        day_of_the_year = user_datetime.timetuple().tm_yday
    else:
        user_datetime = datetime.datetime.strptime(f'{year} {day_of_the_year}',
                                                   '%Y %j')
        day = user_datetime.day
        month = user_datetime.month                                          

    date_str = f'{year}-{month:02}-{day:02}'
    logger.info(f'date: {date_str}, day of the year: {day_of_the_year}')

    nasa_hls.download(dstdir=args.download_dir,
                          date=date_str,
                          tile=args.tile,
                          product="L30",
                          overwrite=False)
    search_str = os.path.join(
            args.download_dir,
            f'*{args.tile}*{year}*{day_of_the_year}*hdf')

    dataset_filename_list = glob.glob(search_str)
    
    if len(dataset_filename_list) == 0:
        logger.info(f'WARNING tile {args.tile} not found for date {date_str}')

    if len(dataset_filename_list) == 0:
        new_date = _get_min_day_diff_dataset(args.tile, year, month, day)
        dataset_filename_list = _download_dataset(args, year, month, day,
                                                  day_of_the_year)
    return dataset_filename_list
 

def _get_min_day_diff_dataset(tile, year, month, day):
    user_datetime = datetime.datetime(year, month, day)
    urls_datasets = nasa_hls.get_available_datasets(
        products=["L30"], years=[year],
        tiles=[tile],
        return_list=False)
    min_day_diff = np.nan
    min_day_diff_date = None
    for i, date in enumerate(urls_datasets.date):
        day_diff = abs((date - user_datetime).days)
        if (np.isnan(min_day_diff) or 
                day_diff < min_day_diff):
            min_day_diff = day_diff
        
    logger.info(f'closest available dataset from selected date: {date}'
                f' ({min_day_diff} days difference)')
    return date

def main():
    parser = _get_parser()

    args = parser.parse_args()

    configure_log_file(args.log_file)

    dataset_filename_list = \
        _download_dataset(args, args.year, args.month, args.day, 
                          args.day_of_the_year)

    dataset_filename = dataset_filename_list[0]
    logger.info(f'dataset filename: {dataset_filename}')

    generate_dswx_layers(
        dataset_filename,
        args.output_dswx_file, 
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