#!/usr/bin/env bash
# source this bash script to run in the current shell

# setup the path to built-in dataset according to Detectron2 Instructions by
export DETECTRON2_DATASETS=/cvhci/temp/wmao/detectron2_datasets/

# train the net (assume 4 gpus)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# change to directory where the training script is located
cd ..

# Be sure to modify the SOLVER.BASE_LR according to the number of GPUs as indicated by the Linear Scale Rule.
# By default, BASE_LR = 0.02 for IMS_PER_BATCH = 16
./train_net.py \
--config-file ./configs/COCO/panoptic_fpn_verconv_verconv_head_R_50_3x \
--num-gpus 4 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 \
OUTPUT_DIR  "./output_panoptic_fpn_verconv_head_r50_3x"
