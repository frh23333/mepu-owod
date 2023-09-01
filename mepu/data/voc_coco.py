import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.pascal_voc import load_voc_instances
import logging
import json

SOWOD_CATEGORIES = [
    # t1
    "airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep",
    "train","elephant","bear","zebra","giraffe","truck","person",
    # t2
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","dining table",
    "potted plant","backpack","umbrella","handbag","tie",
    "suitcase","microwave","oven","toaster","sink","refrigerator","bed","toilet","couch",
    # t3
    "frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake",
    # t4
    "laptop","mouse","remote","keyboard","cell phone",
    "book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush","wine glass","cup","fork","knife","spoon","bowl","tv","bottle",
    # Unknown
    "unknown",
]

OWOD_CATEGORIES = [
    # voc
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
    "potted plant", "sheep", "couch", "train", "tv",
    # t2
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator", 
    # t3 
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", 
    # t4
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
    # Unknown
    "unknown",
] 


def load_voc_instances_with_extrabboxes(
    dirname: str, split: str, 
    class_names: Union[List[str], Tuple[str, ...]], 
    extra_bbox_type: str,):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    bbox_file = {}
    logger = logging.getLogger(__name__)
    if extra_bbox_type == "st":
        bbox_file = json.load(open(os.path.join(annotation_dirname, "pseudo_label_st.json")))
        logger.info("loading pseudo labels from self-training as unknown")
    elif extra_bbox_type == "fs":
        bbox_file = json.load(open(os.path.join(annotation_dirname, "pseudo_label_fs.json")))
        logger.info("loading pseudo labels from FreeSOLO as unknown")
    elif extra_bbox_type == "ss":
        bbox_file = json.load(open(os.path.join(annotation_dirname, "pseudo_label_ss.json")))
        logger.info("loading pseudo labels from Selective Search as unknown")
    elif extra_bbox_type == "eb":
        bbox_file = json.load(open(os.path.join(annotation_dirname, "pseudo_label_eb.json")))
        logger.info("loading pseudo labels from edgeBoxes as unknown")
    elif extra_bbox_type == "gop":
        bbox_file = json.load(open(os.path.join(annotation_dirname, "pseudo_label_gop.json")))
        logger.info("loading pseudo labels from Geodesic Object Proposals as unknown")
    elif extra_bbox_type == "detreg":
        bbox_file = json.load(open(os.path.join(annotation_dirname, "pseudo_label_detreg.json")))
        logger.info("loading pseudo labels from Detreg as unknown")
        
    # if len(bbox_file) and bbox_file[fileids[0]].get("scores", -1) != -1:
    #     logger.info("loading soft labels from offline rew model")

    for fileid in fileids:
        
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        
        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, \
                    "bbox_mode": BoxMode.XYXY_ABS, "soft_label": 1.0}, 
            )
            
        bbox_file_img = bbox_file.get(fileid, -1)
        if bbox_file_img != -1:
            bboxes = bbox_file_img.get("bboxes", -1)
            soft_labels = bbox_file_img.get("scores", -1)
            if soft_labels != -1:
                soft_labels = (np.array(soft_labels)).tolist()
        else:
            bboxes = soft_labels = -1
        
        if bboxes == -1:
            bboxes = []
        if soft_labels == -1:
            soft_labels = np.ones([len(bboxes)])
        
        for bbox, soft_label in zip(bboxes, soft_labels):
            instances.append(
                {"category_id": class_names.index("unknown"), "bbox": bbox, \
                    "bbox_mode": BoxMode.XYXY_ABS, "soft_label": soft_label}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_sowod(name, dirname, split, year):
    class_names = SOWOD_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_sowod_with_extrabboxes(name, dirname, split, year, bbox_type):
    class_names = SOWOD_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances_with_extrabboxes(dirname, split, class_names, bbox_type))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split,
    )

def register_mowod(name, dirname, split, year):
    class_names = OWOD_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_mowod_with_pl(name, dirname, split, year, bbox_type):
    class_names = OWOD_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances_with_extrabboxes(dirname, split, class_names, bbox_type))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split,
    )