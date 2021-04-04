python ../demo/demo.py \
--config-file /cvhci/temp/wmao/Results/output_panoptic_ppa_single_head_atp5_r50_3x/config.yaml \
--video-input ./ps_on_video/video-clip.mp4  \
--confidence-threshold 0.6 --output ./video-output \
--opts MODEL.WEIGHTS /cvhci/temp/wmao/Results/output_panoptic_ppa_single_head_atp5_r50_3x/model_final.pth