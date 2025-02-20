FROM ubuntu:20.04
FROM python:3.9

# Set default UID and GID
ENV USER_ID 1000
ENV GROUP_ID 1000

# Set an encoding to make things work smoothly.
ENV LANG en_US.UTF-8

# create user and install latest miniconda
COPY docker/requirements.txt /tmp/requirements.txt
RUN set -ex \
 && groupadd -r conda -g $GROUP_ID \
 && useradd -u $USER_ID -r -g conda -d /home/conda -s /bin/bash -m conda \
 && chmod 755 /home/conda \
 && echo 'conda ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
 && curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
 && bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/conda \
 && rm Miniforge3-$(uname)-$(uname -m).sh \
 && touch /opt/conda/conda-meta/pinned \
 && export PATH=/opt/conda/bin:$PATH \
 && conda config --prepend channels conda-forge \
 && conda config --append channels nodefaults \
 && conda config --set channel_priority strict \
 && conda config --set show_channel_urls True \
 && conda update --all --yes --channel conda-forge --override-channels \
 && conda install --yes --channel conda-forge --override-channels --file /tmp/requirements.txt \
 && conda clean -tipy \
 && rm -rf /opt/conda/pkgs \
 && rm /tmp/requirements.txt

# set default user and workdir
WORKDIR /home/conda
ADD dist/proteus-1.0.1.tar.gz .

USER root
RUN mkdir -p proteus-1.0.1/build
RUN export PATH=/opt/conda/bin:$PATH
WORKDIR /home/conda/proteus-1.0.1
RUN python3 setup.py install
WORKDIR /home/conda

USER conda

ENV PROJ_LIB /opt/conda/share/proj/
ENV GDAL_DIR /opt/conda
ENV PATH $GDAL_DIR/bin:$PATH
ENV GDAL_DATA $GDAL_DIR/share/gdal

ENV PYTHONPATH /home/conda/proteus-1.0.1/src/:$PYTHONPATH
ENV PATH /home/conda/proteus-1.0.1/bin/:$PATH

CMD [ "/bin/bash" ]
