#!/bin/bash

IMAGE=opera/proteus
t=cal_val
echo "IMAGE is $IMAGE:$t"

# fail on any non-zero exit codes
set -ex

python3 setup.py sdist

# build image
docker build --rm --force-rm --network=host -t ${IMAGE}:$t -f docker/Dockerfile .

 # run tests
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/mnt" -w /mnt -it "${IMAGE}:$t" pytest /mnt/tests/

# create image tar
docker save opera/proteus > docker/dockerimg_proteus_$t.tar

# remove image
docker image rm opera/proteus:$t
