import torch

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import  setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random,tqdm
import matplotlib.pylab as plt
import numpy as np
from panopticapi.utils import IdGenerator, rgb2id
from PIL import Image
import argparse

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from projects.Panoptic_Attention.panoptic_attention import add_panoptic_ppa_config

def parse_args():
    parser = argparse.ArgumentParser(description="Run PS on images and Visualize Results")
    parser.add_argument("--config_file", dest='config', help="path to config file")
    parser.add_argument("--model_weights", dest='weights', help="path to weights")
    parser.add_argument("--output_dir", dest='output_dir', help="path to output directory")
    parser.add_argument("--image_dir", dest='image_dir', help="path to image directory")
    parser.add_argument("--device", dest='device', help="cuda or cpu")

    return parser.parse_args()

def panoptic_inference(config,
                       model_weights,
                       image_folder,
                       output_dir,
                       device,
                       ):
    """
    Vsiualize coco-style panoptic annotatioin json.
    Args:
        config:
        model_weights:
        image_folder:
        output_dir:

    Returns:

    """
    # Load the config, model weights
    # Get a copy of the default configs
    cfg = get_cfg()
    add_panoptic_ppa_config(cfg)

    if device == 'cpu':
        cfg.MODEL.DEVICE = "cpu"
    # _C.MODEL.DEVICE = "cuda"

    # Use the configs for your own model
    cfg.merge_from_file(config)
    # meta =  MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    # # print(type(meta))
    # print(f"label_divisor {meta.label_divisor}")
    # Load the learned parameters for your own model
    cfg.MODEL.WEIGHTS = model_weights

    # Cread a predictor which accepts a np.array image and returns a dict storing prediction
    predictor = DefaultPredictor(cfg)

    # COCO MeataData
    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # list_images is list of image full name including the file extension
    list_images = os.listdir(image_folder)

    os.makedirs(output_dir, exist_ok=True)
    for image_name in tqdm.tqdm(list_images):
        prefix = image_name[:-3]

        # img is in BGR order
        img = cv2.imread(os.path.join(image_folder, image_name))

        # predictor need BGR imag
        panoptic_seg, segments_info = predictor(img)["panoptic_seg"]

        # Give the metadata and RGB image to the visualizer
        v = Visualizer(img[:, :, ::-1], meta, scale=1)

        # Give the segment info to the visualizer, return BGR
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

        # out.ge_image() is numpy
        # print(type(out.get_image()))
        annotated_img = out.get_image()  # RGB
        # plt.imshow(annotated_img)
        # plt.show()

        # receive BGR
        cv2.imwrite(os.path.join(output_dir, prefix + 'jpg'), annotated_img[:, :, ::-1])


if __name__ == '__main__':
    # config = '/home/harvey/Desktop/Master_Thesis/Experiements/panoptic_pyconv_net/panoptic_pyconv_net/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml'
    # weights = 'model_final_dbfeb4.pkl'
    # img_folder = './sample_data/input_images'
    # output_dir = './output_examples'
    # panoptic_inference(config, weights, img_folder, output_dir, 'cuda')
    # config = '/cvhci/temp/wmao/Results/output_panoptic_fpn_official_1x_Jan11_01h/panoptic_fpn_R_50_1x.yaml'
    # weights = '/cvhci/temp/wmao/Results/output_panoptic_fpn_official_1x_Jan11_01h/model_final_dbfeb4.pkl'
    # img_folder = \
    #     '/cvhci/temp/wmao/Results/20200703-13-16-10color'
    # output_dir = \
    #     '/cvhci/temp/wmao/Results/output_vi_model_fpn_official_1x'

    args = parse_args()
    panoptic_inference(args.config, args.weights, args.image_dir, args.output_dir, args.device)
