# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import os
import sys
from collections import OrderedDict, defaultdict
from functools import lru_cache
from tabulate import tabulate

import numpy as np
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from detectron2.evaluation import DatasetEvaluator
import json

np.set_printoptions(threshold=sys.maxsize)

class InferenceST(DatasetEvaluator):

    def __init__(self, dataset_name, cfg=None):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = False
        # self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        if cfg is not None:
            self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        self._predictions = {}  

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.cpu().tolist()
            scores = instances.scores.cpu().tolist()
            self._predictions[image_id] = {"bboxes": boxes, "scores": scores}

    def evaluate(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
        if not comm.is_main_process():
            return
        if not os.path.exists(os.path.join(self.output_dir, "inference")):
            os.mkdir(os.path.join(self.output_dir, "inference"))
        save_path = os.path.join(self.output_dir, "inference", "inference_results.json")
        json.dump(predictions, open(save_path, "w"))
        self._logger.info("saved results to " + save_path)
        
        return {}





