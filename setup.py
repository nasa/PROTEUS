import os
import re
from setuptools import setup
from setuptools import Command


def _get_version():
    """Returns the PROTEUS software version from the
    file `src/proteus/version.py`

       Returns
       -------
       version : str
            PROTEUS software version
    """

    version_file = os.path.join('src','proteus','version.py')

    with open(version_file, 'r') as f:
        text = f.read()

    # Get first match of the version number contained in the version file
    # This regex should match a pattern like: VERSION = '3.2.5', but it
    # allows for varying spaces, number of major/minor versions,
    # and quotation mark styles.
    p = re.search("VERSION[ ]*=[ ]*['\"]\d+([.]\d+)*['\"]", text)

    # Check that the version file contains properly formatted text string
    if p is None:
        raise ValueError(
            f'Version file {version_file} not properly formatted.'
            " It should contain text matching e.g. VERSION = '2.3.4'")

    # Extract just the numeric version number from the string
    p = re.search("\d+([.]\d+)*", p.group(0))

    return p.group(0)

__version__ = version = VERSION = _get_version()

print(f'proteus version {version}')

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
    install_requires=['argparse', 'numpy==1.23.5', 'yamale',
                      'osgeo', 'scipy==1.10.0', 'pytest', 'requests'],
    url='https://github.com/opera-adt/PROTEUS',
    license='Copyright by the California Institute of Technology.'
    ' ALL RIGHTS RESERVED.',
    long_description=long_description,
    cmdclass={
        'clean': CleanCommand,
        }
)
