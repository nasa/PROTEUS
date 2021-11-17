#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute Dynamic Surface Water Extent (DSWx) from HLS data
# 
# OPERA
#
# Copyright 2021, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S.
# export laws and regulations. User has the responsibility to obtain export
# licenses, or other export authority as may be required before exporting such
# information to foreign countries or providing access to foreign persons.
#
# 
# References:
# [1] Jones, J. W. (2015). Efficient wetland surface water detection and 
# monitoring via Landsat: Comparison with in situ data from the Everglades 
# Depth Estimation Network. Remote Sensing, 7(9), 12503-12538. 
# http://dx.doi.org/10.3390/rs70912503.
# 
# [2] R. Dittmeier, LANDSAT DYNAMIC SURFACE WATER EXTENT (DSWE) ALGORITHM 
# DESCRIPTION DOCUMENT (ADD)", USGS, March 2018
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
from distutils.core import setup

__version__ = version = VERSION = '0.1'

directory = os.path.abspath(os.path.dirname(__file__))

long_description = ''

package_data_dict = {}

package_data_dict[''] = [
    os.path.join('defaults', 'dswx_hls.yaml'),
    os.path.join('schemas', 'dswx_hls.yaml')]

setup(
    name='DSWx-HLS',
    version=version,
    description='Compute Dynamic Surface Water Extent (DSWx) from HLS data',
    package_dir={'dswx_hls': '.'},
    packages=['dswx_hls',
              'dswx_hls.bin',
              'dswx_hls.extern'],
    package_data=package_data_dict,
    classifiers=['Programming Language :: Python', ],
    # py_modules=['bin/dswx_hls.py'],
    scripts=['bin/dswx_hls.py'],
    install_requires=['argparse', 'numpy', 'yamale', 'ruamel',
                      'osgeo', 'scipy'],
    # 'tempfile' 'os' 'sys' 'glob' 'mimetypes'
    url='https://github-fn.jpl.nasa.gov/OPERA-ADT/DSWX-optical',
    author='Gustavo H. X. Shiroma',
    author_email=('gustavo.h.shiroma@jpl.nasa.gov'),
    license='Copyright by the California Institute of Technology.'
    ' ALL RIGHTS RESERVED.',
    long_description=long_description,
)