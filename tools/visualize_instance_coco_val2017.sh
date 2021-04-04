export DETECTRON2_DATASETS=/cvhci/temp/wmao/detectron2_datasets/
python visualize_data.py \
--source "annotation" \
--config-file ~/code/detectron2_PFPN/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml \
--output-dir  /cvhci/temp/wmao/Results/coco_2017_val_instances_object_detection
