#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from activeteacher.config import add_activeteacher_config
from activeteacher.engine.trainer import ActiveTeacherTrainer
from activeteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_activeteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


annot_link_for_training="datasets/coco/annotations/coco_training_f_new.json"  # link to the dataset labeled or pseudo labeled training set annotation
#annot_link_for_training="datasets/coco/annotations/coco_training.json"  # link to the dataset labeled or pseudo labeled training set annotation
  
def main(args):
    cfg = setup(args)
    
    print("------------------annot_link_for_training=",annot_link_for_training,"--------------\n")
     
    register_coco_instances("my_dataset_train", {}, annot_link_for_training, "datasets/coco/train") # register the custom dataset
    register_coco_instances("my_dataset_val", {}, "datasets/coco/annotations/coco_validation.json", "datasets/coco/val") # register the custom dataset
    
    # set name for the different class labels
    metadata= MetadataCatalog.get("my_dataset_train").set(thing_classes = ["Angiectasia","Blood-fresh","Erosion","Erythema","Foreign-Body","Lymphangiectasia","Polyp","Ulcer"]); del metadata.thing_classes
    metadata = MetadataCatalog.get("my_dataset_val").set(thing_classes = ["Angiectasia","Blood-fresh","Erosion","Erythema","Foreign-Body","Lymphangiectasia","Polyp","Ulcer"]); del metadata.thing_classes
    
    Trainer = ActiveTeacherTrainer
    print("args.eval_only=",args.eval_only)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )