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
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from typing import List, Dict, Any
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)


class SemanticVisualizer(Visualizer):
    def __init__(self, img_rgb, catID2color, catID2name):
        """

        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            catID2Color: a dict with cat_id as key and color vector as its value.
            catID2name: a dict with cat_id as key and cat_name as its value.
        """
        super().__init__(img_rgb)
        self.catID2color = catID2color
        self.catID2name = catID2name
        self.max_catID = max(catID2name.keys())

    def draw_full_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel defined by COCO official documentation.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        # np.argsort return the indices that sorts the array in ascending order
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        for label in filter(lambda l: l != 0 and l <= self.max_catID, labels):
            try:
                mask_color = [x / 255 for x in self.catID2color[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)
            text = self.catID2name[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output


def pan_ann_to_sem_ann(annotation, segmentation_id):
    """
    Construct  a tensor of (H, W) from annotation_json file with each value as the semantic label
    for each image.

    Args:
        annotation: A dict holds the panoptic annotation info for a image, with keys:
                    'segments_info', 'file_name', 'image_id'. segment_info is also a dict, with keys:
                     ['id', 'category_id', 'iscrowd', 'bbox', 'area']
        segmentation_id: Numpy Array of shape (H, W) with each value as segment ID defined in official COCO.
                    Note:The final seg_id is  obtained by applying rgb2id :`meth' to annotation png file.

    Returns:
        A tensor of (H, W), where H and W are the height and width of the input image and each value is
        the category ID defined in the COCO official documentations.

    """
    # list of dict, each dict corresponds to one seg
    # keys of each dict_keys(['id', 'category_id', 'iscrowd', 'bbox', 'area'])
    segments_info = annotation['segments_info']

    for seg_info in segments_info:
        segmentation_id[segmentation_id == seg_info['id']] = seg_info['category_id']

    return segmentation_id


def COCO_CATID2COLOR(COCO_CATEGORY: List[Dict[str, Any]]):
    # Example of id:color: 2: [0, 255, 0]
    catID2color = {}
    for cat_dict in COCO_CATEGORY:
        catID2color[cat_dict['id']] = cat_dict['color']

    return catID2color


def COCO_CATID2NAME(COCO_CATEGORY: List[Dict[str, Any]]):
    # e.g. id: name : 2:'person'
    catID2name = {}
    for cat_dict in COCO_CATEGORY:
        catID2name[cat_dict['id']] = cat_dict['name'].\
            replace("-other", "").replace("-merged", "")


    return catID2name


def full_sem_visulize(image_folder,
                      annotation_json,
                      segmentation_folder,
                      output_dir,
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
    catID2color = COCO_CATID2COLOR(COCO_CATEGORIES)
    catID2name = COCO_CATID2NAME(COCO_CATEGORIES)
    # print(f'The number of categories {len(catID2name.keys())}')
    # print(f'Max CatID {max(catID2name.keys())}')

    # Load the annotation json file
    with open(annotation_json, 'r') as ann_json:
        # ann is list of dict , each element of list corresponds to an image
        # dict_keys(['segments_info', 'file_name', 'image_id']
        f = json.load(ann_json)
    annotatioins = f['annotations']

    # list_images is list of image full name including the file extension
    # list_images = os.listdir(image_folder)
    # list_segs = os.listdir(segmentation_folder)

    os.makedirs(output_dir, exist_ok=True)

    for ann in tqdm.tqdm(annotatioins):
        prefix = ann['file_name'][:-3]

        img_name = prefix + 'jpg'
        img = np.array(
            Image.open(os.path.join(image_folder, img_name))
        )

        # segmentation (H, W, C) in RGB order
        segmentation = np.array(
            Image.open(os.path.join(segmentation_folder, ann['file_name'])),
            dtype=np.uint8
        )

        # seg_id (H, W)
        segmentation_id = rgb2id(segmentation)
        segmentation_id = torch.from_numpy(segmentation_id)

        segmentation_cat_id = pan_ann_to_sem_ann(ann, segmentation_id)

        # Give the metadata and RGB image to the visualizer
        v = SemanticVisualizer(img_rgb=img, catID2color=catID2color, catID2name=catID2name)

        # Give the segment info to the visualizer
        out = v.draw_full_sem_seg(segmentation_cat_id)

        # assert isinstance(out.get_image(), np.ndarray)
        # print(type(out.get_image()))
        annotated_img = out.get_image()  # RGB
        # plt.imshow((annotated_img))
        # plt.show()

        Image.fromarray(annotated_img).save(os.path.join(output_dir, prefix + 'jpg'))


if __name__ == '__main__':
    # img_folder = "/home/harvey/Desktop/Master_Thesis/Experiements/panoptic_pyconv_net/panoptic_pyconv_net/tools/inference_visualize/sample_data/input_images"
    # ann_json = "/home/harvey/Desktop/Master_Thesis/Experiements/panoptic_pyconv_net/panoptic_pyconv_net/tools/inference_visualize/sample_data/panoptic_examples.json"
    # seg_folder = "/home/harvey/Desktop/Master_Thesis/Experiements/panoptic_pyconv_net/panoptic_pyconv_net/tools/inference_visualize/sample_data/panoptic_examples"
    # out_dir = "./output_examples"

    json_file = '/cvhci/temp/wmao/detectron2_datasets/coco/annotations/panoptic_val2017.json'
    seg_folder = '/cvhci/temp/wmao/detectron2_datasets/coco/panoptic_val2017/'
    img_folder = '/cvhci/temp/wmao/detectron2_datasets/coco/val2017/'
    out_dir = '/cvhci/temp/wmao/visualized_RGB_semantic_panoptics_coco_val_2017'
    full_sem_visulize(img_folder, json_file, seg_folder, out_dir)
