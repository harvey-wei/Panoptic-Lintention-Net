python  image_inference_visualize_ppa.py  \
--config_file   /cvhci/temp/wmao/Results/output_encoderlayer_p5p4_3x/config.yaml \
--model_weights /cvhci/temp/wmao/Results/output_encoderlayer_p5p4_3x/model_final.pth \
--image_dir   /cvhci/temp/wmao/detectron2_datasets/coco/val2017 \
--output_dir '/cvhci/temp/wmao/visualizations/output_RGB_coco_val_w_labels_encoder_p5p4' \
--device 'cuda'
