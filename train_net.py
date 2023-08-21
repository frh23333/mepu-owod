#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import  hooks, launch
from detectron2.evaluation import verify_results
from mepu.config.config import add_config
import mepu.data.builtin
import mepu.model.detector.oln_box, mepu.model.rew.offline_rew
from mepu.evaluation.open_world_eval import PascalVOCDetectionEvaluator
from mepu.data.build import build_detection_test_loader, build_detection_train_loader
from mepu.engine.defaults import DefaultTrainer, default_setup, default_argument_parser
from mepu.evaluation.coco_evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from mepu.evaluation.inference_self_train import InferenceST

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if cfg.OPENSET.INFERENCE_SELT_TRAIN:
        return InferenceST(dataset_name, cfg)
    
    if evaluator_type == "coco":
        return COCOEvaluator(dataset_name, cfg, distributed=True, output_dir=output_folder)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name, cfg)
    

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg = add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    output_dir = cfg.OUTPUT_DIR
    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="mepu")

    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if args.inference_rew:
            res = Trainer.inference_rew(cfg, model)
        elif args.update_weibull:
            res = Trainer.update_weibull(cfg, model)
        else:
            res = Trainer.test(cfg, model)
            if comm.is_main_process():
                verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--inference-rew", action="store_true")
    parser.add_argument("--update-weibull", action="store_true")
    args = parser.parse_args()
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
