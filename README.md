# PROTEUS

The Dynamic Surface Water eXtent from Harmonized Landsat and Sentinel-2 (DSWx-HLS) Science Application Software (SAS).

# License
**Copyright (c) 2022** California Institute of Technology (“Caltech”). U.S. Government
sponsorship acknowledged.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
* Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the
names of its contributors may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

## Installation

### Download the Source Code
Download the source code and change working directory to cloned repository:

```bash
git clone https://github.com/nasa/PROTEUS.git
cd PROTEUS
```

### Standard Installation
Install dependencies (installation via conda is recommended):
```bash
conda install --file docker/requirements.txt
conda install -c conda-forge --file docker/requirements.txt.forge
```

Install via setup.py:

```bash
python setup.py install
python setup.py clean
```

Note: Installation via pip is not currently recommended due to an
issue with the osgeo and gdal dependency.


OR update environment path to run PROTEUS:

```bash
export PROTEUS_HOME=$PWD
export PYTHONPATH=${PYTHONPATH}:${PROTEUS_HOME}/src
export PATH=${PATH}:${PROTEUS_HOME}/bin
```

Run workflow tests to ensure proper installation:

```bash
pytest -rpP tests
```

Process data sets; use a runconfig file to specify the location
of the dataset, the output directory, parameters, etc.

```bash
dswx_hls.py <path to runconfig file>
```

A default runconfig file can be found: `PROTEUS > src > proteus > defaults > dswx_hls.yaml`.
This file can be copied and modified for your needs.
Note: The runconfig must meet this schema: `PROTEUS > src > proteus > schemas > dswx_hls.yaml`.


### Alternate Installation: Docker Image

Skip the standard installation process above.

Then, from inside the cloned repository, build the Docker image:
(This will automatically run the workflow tests.)

```bash
./build_docker_image.sh
```

Load the Docker container image onto your computer:

```bash
docker load -i docker/dockerimg_proteus_final_4.1.tar
```

See DSWx-HLS Science Algorithm Software (SAS) User Guide for instructions on processing via Docker.
