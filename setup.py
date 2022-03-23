import os
from distutils.core import setup

__version__ = version = VERSION = '0.1'

directory = os.path.abspath(os.path.dirname(__file__))

long_description = ''

package_data_dict = {}

package_data_dict[''] = [
    os.path.join('src', 'proteus', 'defaults', 'dswx_hls.yaml'),
    os.path.join('src', 'proteus', 'schemas', 'dswx_hls.yaml')]

setup(
    name='proteus',
    version=version,
    description='Compute Dynamic Surface Water Extent (DSWx)'
                ' from optical (HLS) and SAR data',
    package_dir={'dswx_hls': '.'},
    packages=['dswx_hls',
              'dswx_hls.src.proteus',
              'dswx_hls.src.proteus.extern'],
    package_data=package_data_dict,
    classifiers=['Programming Language :: Python', ],
    # py_modules=['src/proteus/core.py'],
    scripts=['bin/dswx_hls.py'],
    install_requires=['argparse', 'numpy', 'yamale', 'ruamel',
                      'osgeo', 'scipy', 'pytest'],
    url='https://github.com/opera-adt/PROTEUS',
    author='Gustavo H. X. Shiroma',
    author_email=('gustavo.h.shiroma@jpl.nasa.gov'),
    license='Copyright by the California Institute of Technology.'
    ' ALL RIGHTS RESERVED.',
    long_description=long_description,
)