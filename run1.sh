#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/snn_iir"

docker build -q -t ${IMAGE_NAME} "$CURRENT_PATH"/docker && \
docker run -it --rm \
    --gpus device=1 \
    -v "$CURRENT_PATH"/src:/workdir \
    -v "$CURRENT_PATH"/dataset:/dataset \
    -v "$CURRENT_PATH"/checkpoint:/checkpoint \
    -v "$CURRENT_PATH"/logs:/logs \
    -v "$CURRENT_PATH"/torch_logs:/torch_logs \
    -w /workdir \
    ${IMAGE_NAME} \
    /bin/bash -c "\
        python ann_snn_poisson.py \
            --model networks.fixed_cnn_poisson_networks.ann6_snn2 \
            --pretrained_model networks.fixed_cnn_poisson_networks.pretrained_model \
            --config_file ann_snn_cnn_poisson.yaml \
            --train \
            --logging && \
        python ann_snn_poisson.py \
            --model networks.fixed_cnn_poisson_networks.ann4_snn4 \
            --pretrained_model networks.fixed_cnn_poisson_networks.pretrained_model \
            --config_file ann_snn_cnn_poisson.yaml \
            --train \
            --logging && \
        python ann_snn_poisson.py \
            --model networks.fixed_cnn_poisson_networks.ann1_snn7 \
            --pretrained_model networks.fixed_cnn_poisson_networks.pretrained_model \
            --config_file ann_snn_cnn_poisson.yaml \
            --train \
            --logging \
    "
