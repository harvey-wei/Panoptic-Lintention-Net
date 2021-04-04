python image_inference_visualize.py \
--config_file /cvhci/temp/wmao/Results/output_panoptic_fpn_verconvsep_head_r50_3x_Dec21_23h26m/config.yaml  \
--model_weights /cvhci/temp/wmao/Results/output_panoptic_fpn_verconvsep_head_r50_3x_Dec21_23h26m/model_final.pth \
--image_dir   /cvhci/temp/wmao/detectron2_datasets/coco/val2017 \
--output_dir '/cvhci/temp/wmao/visualizations/output_RGB_coco_val_w_labels_verconv_sep' \
--device 'cuda'
