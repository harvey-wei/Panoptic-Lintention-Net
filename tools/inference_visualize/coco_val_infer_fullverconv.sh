python image_inference_visualize.py \
--config_file  /cvhci/temp/wmao/Results/output_panoptic_fpn_full_verconv_head_r50_3x_Dec26_15h45m_3x/config.yaml \
--model_weights /cvhci/temp/wmao/Results/output_panoptic_fpn_full_verconv_head_r50_3x_Dec26_15h45m_3x/model_final.pth \
--image_dir   /cvhci/temp/wmao/detectron2_datasets/coco/val2017 \
--output_dir '/cvhci/temp/wmao/visualizations/output_RGB_coco_val_w_labels_fullverconv' \
--device 'cuda'
