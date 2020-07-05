#!/bin/bash
# to activate the virtual environment, change here
source "/home/mod/.local/share/virtualenvs/Workspace-n_bFWxY_/bin/activate"

for fold in 0 1 2 3;
do
    python -m shopee.main \
          --backbone 'resnext101_32x8d_swsl' \
          --image-size 256 \
          --use-neck 0 \
          --criterion 'ce' \
          --fp16 False \
          --debug 0 \
          --data-path '/home/mod/Workspace/shopee-data' \
          --image-train-dir '/home/mod/Workspace/shopee-data/train_256x256' \
          --image-test-dir '/home/mod/Workspace/shopee-data/test_256x256' \
          --batch-size 64 \
          --fold ${fold} \
          --lr 7e-6 \
          --epochs 30 \
          --device 0 \
          --output-dir "/home/mod/Workspace/shopee-product-detection-2020/output"
done
