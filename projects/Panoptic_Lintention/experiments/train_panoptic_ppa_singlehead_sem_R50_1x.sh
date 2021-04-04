#!/usr/bin/env bash
# source this bash script to run in the current shell

# setup the path to built-in dataset according to Detectron2 Instructions by
export DETECTRON2_DATASETS=/cvhci/temp/wmao/detectron2_datasets/

# train the net (assume 4 gpus)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# change to directory where the training script is located
cd ..

# Be sure to modify the SOLVER.BASE_LR according to the number of GPUs as indicated by the Linear Scale Rule.
# By default, BASE_LR = 0.02 for IMS_PER_BATCH = 16
./train_net.py \
--config-file ./configs/COCO/panoptic_ppa_singlehead_sem_R50_1x.yaml \
--num-gpus 4 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 \
OUTPUT_DIR  "./output_panoptic_ppa_single_head__r50_1x"
