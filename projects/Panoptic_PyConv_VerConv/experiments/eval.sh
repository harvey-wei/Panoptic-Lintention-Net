#!/usr/bin/env bash
# source this bash script to run in the current shell

# setup the path to built-in dataset according to Detectron2 Instructions by
export DETECTRON2_DATASETS=/cvhci/temp/wmao/detectron2_datasets/

# train the net (assume 4 gpus)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# config.yaml must be the one to produce the model_final.pth
../train_net.py \
  --config-file ./config.yaml \
  --eval-only MODEL.WEIGHTS ./model_final.pth OUTPUT_DIR "./infer_results"