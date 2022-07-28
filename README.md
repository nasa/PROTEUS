# PROTEUS
PROTEUS - Parallelized Radar Optical Toolbox for Estimating dynamic sUrface water extentS

# License
**Copyright (c) 2021** California Institute of Technology (“Caltech”). U.S. Government
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

Download the source code and move working directory to clone repository:

```bash
git clone https://github.com/opera-adt/PROTEUS.git
cd PROTEUS
```

Install PROTEUS via conda/setup.py (recommended):

```bash
conda install --file docker/requirements.txt
conda install -c conda-forge --file docker/requirements.txt.forge
python setup.py install
```

Or via pip:

```bash
pip install .
```

Or via environment path setup:

```bash
export PROTEUS_HOME=$PWD
export PYTHONPATH=${PYTHONPATH}:${PROTEUS_HOME}/src
export PATH=${PATH}:${PROTEUS_HOME}/bin
```

Run workflow and unit tests:

```bash
cd tests
pytest -rpP
```
