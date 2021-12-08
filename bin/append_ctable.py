import argparse
from osgeo import gdal
from dswx_hls import save_mask, save_dswx_product, band_description_dict


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
    parser_dataset.add_argument('--cloud'
                                '--cloud-mask',
                                action='store_true',
                                dest='cloud_mask',
                                help='Append color table to cloud mask')

    parser_dataset.add_argument('--wtr',
                                '--interpreted-band',
                                action='store_true',
                                dest='interpreted_dswx',
                                help='Append color table to interpreted DSWx layer')

    parser.add_argument('--intr',
                        '--non-masked-dswx',
                        action='store_true',
                        dest='non_masked_dswx',
                        help='Append color table to non-masked DSWx layer file')

    parser.add_argument('--insm',
                        '--shadow-masked-dswx',
                        action='store_true',
                        dest='shadow_masked_dswx',
                        help='Append color table to interpreted layer refined using'
                        ' land cover and terrain shadow testing ')

    return parser


def main():
    parser = _get_parser()

    args = parser.parse_args()

    layer_gdal_dataset = gdal.Open(args.input_file[0])
    geotransform = layer_gdal_dataset.GetGeoTransform()
    projection = layer_gdal_dataset.GetProjection()
    if layer_gdal_dataset.RasterCount > 1:

        band_description_keys_list = list(band_description_dict.keys())
        if args.non_masked_dswx:
            layer_name = 'INTR'
        elif args.shadow_masked_dswx:
            layer_name = 'INSM'
        elif args.cloud_mask:
            layer_name = 'CLOUD'
        else:
            layer_name = 'WTR'

        band = band_description_keys_list.index(layer_name) + 1
        print(f'Loading DSWx-HLS product (band: {band})')
        band = layer_gdal_dataset.GetRasterBand(band)
        image = band.ReadAsArray()
    else:
        image = layer_gdal_dataset.ReadAsArray()

    metadata_dict = {}
    if args.cloud_mask:
        print('Appending color table to cloud mask')
        save_mask(image, args.output_file, metadata_dict, geotransform, 
                  projection)
    else:
        print('Appending color table to interpreted DSWx layer')
        save_dswx_product(image, args.output_file, metadata_dict,
                          geotransform, projection)


if __name__ == '__main__':
    main()