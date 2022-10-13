import os
from setuptools import setup
from setuptools import Command

__version__ = version = VERSION = '0.1'


class CleanCommand(Command):
    """Custom clean command to tidy up the project root 
    after running `python setup.py install`."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        # Make sure to remove the .egg-info file 
        os.system('rm -vrf .scratch_dir ./build ./dist ./*.pyc ./*.tgz ./*.egg-info ./src/*.egg-info')


long_description = open('README.md').read()

package_data_dict = {}

package_data_dict['proteus'] = [
    os.path.join('defaults', 'dswx_hls.yaml'),
    os.path.join('schemas', 'dswx_hls.yaml')]

setup(
    name='proteus',
    version=version,
    description='Compute Dynamic Surface Water Extent (DSWx)'
                ' from optical (HLS) data',
    # Gather all packages located under `src`.
    # (A package is any directory containing an __init__.py file.)
    package_dir={'': 'src'},
    packages=['proteus',
              'proteus.extern'],
    package_data=package_data_dict,
    classifiers=['Programming Language :: Python',],
    scripts=['bin/dswx_hls.py',
    	     'bin/dswx_compare.py'],
    install_requires=['argparse', 'numpy', 'yamale',
                      'osgeo', 'scipy', 'pytest', 'requests'],
    url='https://github.com/opera-adt/PROTEUS',
    author='Gustavo H. X. Shiroma',
    author_email=('gustavo.h.shiroma@jpl.nasa.gov'),
    license='Copyright by the California Institute of Technology.'
    ' ALL RIGHTS RESERVED.',
    long_description=long_description,
    cmdclass={
        'clean': CleanCommand,
        }
)
