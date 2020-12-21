#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/snn_iir"

docker build -t ${IMAGE_NAME} "$CURRENT_PATH"/docker/gpu && \
docker run -it --rm \
    -u $(id -u):$(id -g) \
    -p 8888:8888 \
    -v /home/muramatsu/Projects/quva-repetition-dataset-generator/data/output:/dataset \
    -v "$CURRENT_PATH"/data:/data \
    -v "$CURRENT_PATH"/notebooks:/workdir \
    -v "$CURRENT_PATH"/src:/src \
    ${IMAGE_NAME}
