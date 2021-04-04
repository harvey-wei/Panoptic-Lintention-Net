import torch

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, tqdm
import matplotlib.pylab as plt
import argparse
import numpy as np
from panopticapi.utils import IdGenerator, rgb2id
from PIL import Image

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def coco_seg_info_to_detectron2_seg_info(coco_seg_info, metadata):
    """

    Args:
        coco_seg_info: keys: dict_keys(['id', 'category_id', 'iscrowd', 'bbox', 'area'])


    Returns:
        det_seg_info: such as
        {
        "id": segment_id,
        "category_id": pred_class,
        "isthing": thing,
        }}

        segment_id corresponds to vale in rgb2id(segment.png)
        0 means void in coco but -1 means void in model
        If "isthing" == true, pred_class is the contiguous category id starting at 0
        If "isthing" == true, pred_class is the contiguous category id starting at 0
    """
    seg_id = coco_seg_info['id']
    dataset_cat_id = coco_seg_info['category_id']

    if seg_id == 0:
        coco_seg_info['id'] == -1

    # stuff_dataset_id_to_contiguous_id is dict of stuff_datatset_id : contiguous_id
    stuff_dataset_id_to_contiguous_id = metadata.stuff_dataset_id_to_contiguous_id
    thing_dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id

    if dataset_cat_id in stuff_dataset_id_to_contiguous_id:
        coco_seg_info['isthing'] = False
        coco_seg_info['category_id'] = stuff_dataset_id_to_contiguous_id[dataset_cat_id]

    if dataset_cat_id in thing_dataset_id_to_contiguous_id:
        coco_seg_info['isthing'] = True
        coco_seg_info['category_id'] = thing_dataset_id_to_contiguous_id[dataset_cat_id]

    return coco_seg_info


def dataset_visualize(config,
                      image_folder,
                      annotation_json,
                      segmentation_folder,
                      output_dir,
                      device='cuda'
                      ):
    """
    Visualize coco-style panoptic annotatioin json.
    And store as jpg images in output_dir.
    Args(all are str):
        image_folder:
        annotation_json:
        segmentation_folder:
        output_dir:
    """
    # Load the config, model weights
    # Get a copy of the default configs
    cfg = get_cfg()

    if device == 'cpu':
        cfg.MODEL.DEVICE = 'cpu'

    # Use the configs for your own model
    cfg.merge_from_file(config)
    # meta =  MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    # # print(type(meta))
    # print(f"label_divisor {meta.label_divisor}")
    # Load the learned parameters for your own model
    # cfg.MODEL.WEIGHTS = model_weights

    # Cread a predictor which accepts a np.array image and returns a dict storing prediction
    # predictor = DefaultPredictor(cfg)

    # COCO MeataData
    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Load the annotation json file
    with open(annotation_json, 'r') as ann_json:
        # ann is list of dict , each element of list corresponds to an image
        # dict_keys(['segments_info', 'file_name', 'image_id']
        f = json.load(ann_json)
    annotatioins = f['annotations']

    # list_images is list of image full name including the file extension
    list_images = os.listdir(image_folder)
    list_segs = os.listdir(segmentation_folder)

    os.makedirs(output_dir, exist_ok=True)

    for ann in tqdm.tqdm(annotatioins):
        panoptic_seg_name = ann['file_name']
        prefix = ann['file_name'][:-3]

        img_name = prefix + 'jpg'
        img = np.array(
            Image.open(os.path.join(img_folder, img_name))
        )
        # segmentation (H, W, C) in RGB order
        segmentation = np.array(
            Image.open(os.path.join(segmentation_folder, ann['file_name'])),
            dtype=np.uint8
        )

        # seg_id (H, W)
        segmentation_id = rgb2id(segmentation)
        segmentation_id = torch.from_numpy(segmentation_id)

        # 0 means void in coco but -1 means void in detectron2
        segmentation_id[segmentation_id == 0] = -1

        # list of dict, each dict corresponds to one seg
        segments_info = ann['segments_info']
        num_segs = len(segments_info)
        for seg in range(num_segs):
            segments_info[seg] = coco_seg_info_to_detectron2_seg_info(segments_info[seg], meta)

        # Give the metadata and RGB image to the visualizer
        v = Visualizer(img, meta, scale=1)

        # Give the segment info to the visualizer
        out = v.draw_panoptic_seg_predictions(segmentation_id.to("cpu"), segments_info)

        # out.ge_image() is numpy
        # print(type(out.get_image()))
        annotated_img = out.get_image() # RGB
        # plt.imshow(annotated_img)
        # plt.show()

        # receive RGB
        Image.fromarray(annotated_img).save(os.path.join(output_dir, prefix + 'jpg'))


if __name__ == '__main__':
    # config = '/home/harvey/Desktop/Master_Thesis/Experiements/panoptic_pyconv_net/panoptic_pyconv_net/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml'
    # json_file = './sample_data/panoptic_examples.json'
    # segmentations_folder = './sample_data/panoptic_examples/'
    # img_folder = './sample_data/input_images/'
    # panoptic_coco_categories = './panoptic_coco_categories.json'
    # output_dir = './visualized_images'

    config = "../../configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    json_file = '/cvhci/temp/wmao/detectron2_datasets/coco/annotations/panoptic_val2017.json'
    segmentations_folder = '/cvhci/temp/wmao/detectron2_datasets/coco/panoptic_val2017/'
    img_folder = '/cvhci/temp/wmao/detectron2_datasets/coco/val2017/'
    output_dir = '/cvhci/temp/wmao/visualized_RGB_panoptics_coco_val_2017'

    dataset_visualize(config, img_folder, json_file, segmentations_folder, output_dir)