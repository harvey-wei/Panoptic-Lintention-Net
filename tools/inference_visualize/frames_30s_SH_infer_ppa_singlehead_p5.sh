python  image_inference_visualize_ppa.py \
--config_file   /cvhci/temp/wmao/Results/output_panoptic_ppa_single_head_atp5_r50_3x/config.yaml \
--model_weights /cvhci/temp/wmao/Results/output_panoptic_ppa_single_head_atp5_r50_3x/model_final.pth \
--image_dir /cvhci/temp/wmao/frames_video-clip-30 \
--output_dir '/cvhci/temp/wmao/SHFrames_30s_shp5_preds' \
--device 'cuda'

