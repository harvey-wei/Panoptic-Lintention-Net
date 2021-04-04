python  image_inference_visualize_ppa.py \
--config_file   /cvhci/temp/wmao/Results/ablation_study/output_panoptic_ppa_single_head_2xLin_32patches_r50_3x_0603/config.yaml \
--model_weights /cvhci/temp/wmao/Results/ablation_study/output_panoptic_ppa_single_head_2xLin_32patches_r50_3x_0603/model_final.pth \
--image_dir   /cvhci/temp/wmao/Street_Scenes_KA/ \
--output_dir '/cvhci/temp/wmao/visualizations/output_RGB_SSKA_model_w_labels_ppa_2xLin_32patches_r50_3x' \
--device 'cuda'
