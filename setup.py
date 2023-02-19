from pathlib import Path
import re
from setuptools import setup, Command, find_packages


def _get_version() -> str:
    """Returns the PROTEUS software version from the
    file `src/proteus/version.py`

    Returns
    -------
    version : str
        PROTEUS software version
    """
    version_file = Path('src', 'proteus', 'version.py')

    with version_file.open('r') as f:
        text = f.read()

    # Get first match of the version number contained in the version file
    # This regex should match a pattern like: VERSION = '3.2.5', but it
    # allows for varying spaces, number of major/minor versions,
    # and quotation mark styles.
    p = re.search(r"VERSION\s*=\s*['\"]\d+(\.\d+)*['\"]", text)

    # Check that the version file contains properly formatted text string
    if p is None:
        raise ValueError(f'Version file {version_file} not properly formatted. '
                         f"It should contain text matching e.g. VERSION = '2.3.4'")

    # Extract just the numeric version number from the string
    p = re.search(r"\d+(\.\d+)*", p.group(0))

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
        Path('.scratch_dir').unlink(missing_ok=True)
        Path('./build').unlink(missing_ok=True)
        Path('./dist').unlink(missing_ok=True)
        Path('./*.pyc').unlink(missing_ok=True)
        Path('./*.tgz').unlink(missing_ok=True)
        Path('./*.egg-info').unlink(missing_ok=True)
        Path('./src/*.egg-info').unlink(missing_ok=True)


long_description = Path('README.md').read_text()

package_data_dict = {'proteus': [Path('defaults', 'dswx_hls.yaml'),
                                 Path('schemas', 'dswx_hls.yaml')]}

setup(
    name='proteus',
    version=version,
    description='Compute Dynamic Surface Water Extent (DSWx) '
                'from optical (HLS) data',
    # Gather all packages located under `src`.
    # (A package is any directory containing an __init__.py file.)
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data=package_data
