export CUDA_VISIBLE_DEVICES=0,1,2,3
export DETECTRON2_DATASETS=/cvhci/temp/wmao/detectron2_datasets/
python  image_inference_visualize_ppa.py \
--config_file   /cvhci/temp/wmao/Results/ablation_study/output_panoptic_ppa_single_head_2xLin_32patches_r50_3x_0603/config.yaml \
--model_weights /cvhci/temp/wmao/Results/ablation_study/output_panoptic_ppa_single_head_2xLin_32patches_r50_3x_0603/model_final.pth \
--image_dir   /cvhci/temp/wmao/Gardens_Point_GT/leftImg8bit/nightright \
--output_dir '/cvhci/temp/wmao/nightright_shp5_preds_2xLin_32patches' \
--device 'cuda'

