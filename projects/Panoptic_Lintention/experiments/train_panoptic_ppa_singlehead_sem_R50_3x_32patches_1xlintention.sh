#!/usr/bin/env bash
# source this bash script to run in the current shell

# setup the path to built-in dataset according to Detectron2 Instructions by
export DETECTRON2_DATASETS=/cvhci/temp/wmao/detectron2_datasets/

# train the net (assume 4 gpus)
export CUDA_VISIBLE_DEVICES=0,2,4,5

# change to directory where the training script is located
cd ..

# Be sure to modify the SOLVER.BASE_LR according to the number of GPUs as indicated by the Linear Scale Rule.
# By default, BASE_LR = 0.02 for IMS_PER_BATCH = 16
./train_net.py \
--config-file ./configs/COCO/panoptic_ppa_singlehead_sem_R50_3x_1xlintention.yaml \
--num-gpus 4 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 \
MODEL.SEM_SEG_HEAD.PPA_NUM_PATCHES 32 \
MODEL.WEIGHTS  ~/code/detectron2_PFPN/detectron2/tools/R-50.pkl \
OUTPUT_DIR  "./output_panoptic_ppa_single_head_1xLin_32patches_r50_3x"
