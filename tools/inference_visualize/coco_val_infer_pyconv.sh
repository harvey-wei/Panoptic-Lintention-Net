python image_inference_visualize.py \
--config_file   /cvhci/temp/wmao/Results/output_pan_pyconv_v1head_r50_3x_gpus4_batch8_LR0x01_Jan4_21h/config.yaml	\
--model_weights  /cvhci/temp/wmao/Results/output_pan_pyconv_v1head_r50_3x_gpus4_batch8_LR0x01_Jan4_21h/model_final.pth \
--image_dir   /cvhci/temp/wmao/detectron2_datasets/coco/val2017 \
--output_dir '/cvhci/temp/wmao/visualizations/output_RGB_coco_val_w_labels_pyconv' \
--device 'cuda'
