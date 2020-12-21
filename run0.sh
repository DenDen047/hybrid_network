#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/snn_iir"

docker build -q -t ${IMAGE_NAME} "$CURRENT_PATH"/docker && \
docker run -it --rm \
    --gpus device=0 \
    -v "$CURRENT_PATH"/src:/workdir \
    -v "$CURRENT_PATH"/dataset:/dataset \
    -v "$CURRENT_PATH"/checkpoint:/checkpoint \
    -v "$CURRENT_PATH"/logs:/logs \
    -v "$CURRENT_PATH"/torch_logs:/torch_logs \
    -w /workdir \
    ${IMAGE_NAME} \
    /bin/bash -c "\
        python ann_snn.py \
            --model networks.mlp_networks.baseline_snn \
            --rand_seed 2 \
            --config_file ann_snn_mlp.yaml \
            --train \
            --logging && \
        python ann_snn.py \
            --model networks.mlp_networks.baseline_snn \
            --rand_seed 3 \
            --config_file ann_snn_mlp.yaml \
            --train \
            --logging && \
        python ann_snn.py \
            --model networks.mlp_networks.baseline_snn \
            --rand_seed 4 \
            --config_file ann_snn_mlp.yaml \
            --train \
            --logging && \
        python ann_snn.py \
            --model networks.mlp_networks.baseline_snn \
            --rand_seed 5 \
            --config_file ann_snn_mlp.yaml \
            --train \
            --logging && \
        python ann_snn.py \
            --model networks.mlp_networks.baseline_snn \
            --rand_seed 6 \
            --config_file ann_snn_mlp.yaml \
            --train \
            --logging \
    "
