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
    url='https://github-fn.jpl.nasa.gov/OPERA-ADT/DSWX-optical',
    author='Gustavo H. X. Shiroma',
    author_email=('gustavo.h.shiroma@jpl.nasa.gov'),
    license='Copyright by the California Institute of Technology.'
    ' ALL RIGHTS RESERVED.',
    long_description=long_description,
)