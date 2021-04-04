python image_inference_visualize.py \
--config_file  ~/code/detectron2_PFPN/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml \
--model_weights /cvhci/temp/wmao/Results/output_panoptic_fpn_official_1x_Jan11_01h/model_final_dbfeb4.pkl \
--image_dir /cvhci/temp/wmao/Street_Scenes_KA/  \
--output_dir '/cvhci/temp/wmao/visualizations/output_RGB_SSKA_model_w_labels_pan_fpn_r50' \
--device 'cuda'





