#!/usr/bin/env python3

import numpy as np
from proteus.dswx_hls import interpreted_dswx_band_dict,\
                             generate_interpreted_layer

def test_units():

    # define test arrays dimensions adding one element for fill value
    length = 1
    width = len(interpreted_dswx_band_dict) + 1

    # declare input_array using an invalid value (`111111`) as fill value
    input_array = np.full((length, width), 111111)

    # declare expected output array using `255` as fill value
    expected_output_array = np.full((length, width), 255)

    # populate arrays
    for i, (key, value) in enumerate(interpreted_dswx_band_dict.items()):
        input_array[0, i] = key
        expected_output_array[0, i] = value

    # run DSWx-HLS function to generate interpreted layer
    output_array = generate_interpreted_layer(input_array)

    # compare both arrays
    assert np.array_equal(output_array, expected_output_array)
