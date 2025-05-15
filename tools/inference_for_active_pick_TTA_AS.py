import json
import numpy as np
import torch
import torch.nn.functional as F
import os

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data import detection_utils as utils

from activeteacher.config import add_activeteacher_config
from activeteacher.engine.trainer import ActiveTeacherTrainer
from activeteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

#--------------------------------- Package imports For TTA --------------------------------------
import torchvision
import cv2
import matplotlib.pyplot as plt
from odach import *

import odach as oda

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load a model pre-trained pre-trained on COCO
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)


#------------------------------- TTA -----------------------
# Declare TTA Transformation here
#tta_transforms = [oda.VerticalFlip(), oda.Multiply(0.9), oda.Multiply(1.1)]
tta_transforms = [ oda.GaussianBlur(5, 0.1, 5), oda.HorizontalFlip(), oda.VerticalFlip(), oda.Multiply(0.9), oda.Multiply(1.2), oda.Contrast(0.7), oda.Contrast(1.3), oda.Contrast(1.5), oda.Rotate90Left(),oda.Rotate90Right()]

#scale = [0.8, 0.9, 1, 1.1, 1.2]
        

#-------------------------------------------------------------------------

@torch.no_grad()
def uncertainty_entropy(p):
    # p.size() = num_instances of a image, num_classes
    p = F.softmax(p, dim=0)
    p = - torch.log2(p) * p
    entropy_instances = torch.sum(p, dim=0)
    # set uncertainty of image eqs the mean uncertainty of instances
    entropy_image = torch.mean(entropy_instances)
    return entropy_image


data_hook = {}
def box_predictor_hooker(m, i, o):
    data_hook['scores_hooked'] = o[0].clone().detach()
    data_hook['boxes_hooked'] = o[1].clone().detach()


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


def main(args):
    cfg = setup(args)
    register_coco_instances("my_dataset_train", {}, "datasets/coco/annotations/coco_training.json", "datasets/coco/train") # register the custom dataset for inference; "my_dataset_inference" is the chosen percentage for the unlabled data
    register_coco_instances("my_dataset_val", {}, "datasets/coco/annotations/coco_validation.json", "datasets/coco/val") # register the custom dataset
    
    # set name for the different class labels
    metadata= MetadataCatalog.get("my_dataset_train").set(thing_classes = ["Angiectasia","Blood-fresh","Erosion","Erythema","Foreign-Body","Lymphangiectasia","Polyp","Ulcer"]); del metadata.thing_classes
    metadata = MetadataCatalog.get("my_dataset_val").set(thing_classes = ["Angiectasia","Blood-fresh","Erosion","Erythema","Foreign-Body","Lymphangiectasia","Polyp","Ulcer"]); del metadata.thing_classes
    
    Trainer = ActiveTeacherTrainer
    assert args.eval_only is True, "Inference should be eval only."
    inference(Trainer, cfg)


@torch.no_grad()
def inference(Trainer, cfg):
    print('Loading Model named: ', cfg.MODEL.WEIGHTS)
    model = Trainer.build_model(cfg)
    model_teacher = Trainer.build_model(cfg)
    ensem_ts_model = EnsembleTSModel(model_teacher, model)
    
    
    DetectionCheckpointer(
        ensem_ts_model, save_dir=cfg.OUTPUT_DIR
    ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    
   
    ensem_ts_model.modelTeacher.roi_heads.box_predictor.register_forward_hook(box_predictor_hooker)
    ensem_ts_model.modelTeacher.eval()
    ensem_ts_model.modelTeacher.training = False
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dic={}
    from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
    
    for j,item in enumerate(dataset_dicts):
    
        file_name = item['file_name']
        print(j,file_name)
        
        image = utils.read_image(file_name, format='BGR')
        image1 = torch.from_numpy(image.copy()).permute(2,0,1)
        
        image2= image1.unsqueeze(0).to(torch.device("cpu")).float()
        
        #print("--------------- image1 shape=",image1.shape," ----------------------")
        
        score_thresh = 0.4
        boxes = []; scores = []; labels = [];
        for tta in tta_transforms:
            print("--------------TTA transform=",tta,"------------\n")
            # apply TTA transform the image, then Inference the model predictions on each transformed image.
            inf_img = tta.batch_augment(image2.clone())
            #print("--------------- inf_img1 shape=",inf_img.shape," TYPE=",type(inf_img),"----------------------")
            inf_img = inf_img.squeeze(0)
            res = ensem_ts_model.modelTeacher.inference([{'image':inf_img}])
            
            #print("--------------- res =",res,"----------------------\n")
            thescores=res[0]['instances'].scores.to(torch.device("cpu")).detach().numpy()
            box = res[0]['instances'].pred_boxes.to(torch.device("cpu")).tensor.numpy()
            #print("--------------- scores =",thescores,"----------------------\n")
            #print("--------------- box1 =",box,"----------------------\n")
            box = tta.deaugment_boxes(box)
            # scale box to 0-1
            
            if len(box)==0:
                continue
            
            if np.max(box)>1:
                box[:,0] /= image2.shape[3]
                box[:,2] /= image2.shape[3]
                box[:,1] /= image2.shape[2]
                box[:,3] /= image2.shape[2]
           
            
            #print("--------------- box2 =",box,"----------------------\n")
            
            ind= res[0]['instances'].scores.to(torch.device("cpu")).detach().numpy() > score_thresh
            print("--------------- ind =",ind,"----------------------\n")
            boxes.append(box[ind])
            scores.append(res[0]['instances'].scores.cpu().detach().numpy()[ind])
            labels.append(res[0]['instances'].pred_classes.cpu().detach().numpy()[ind])
            
    
        iou_thr = 0.5
        skip_box_thr = 0.5 # prediction score 
        #print("-----------------------BOXES1=",boxes,"----------------------------\n")
        #print("-----------------------LABEL1=",labels,"----------------------------\n")
        #print("-----------------------SCORES1=",scores,"----------------------------\n")
        
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        
        boxes[:,0] *= image2.shape[3]
        boxes[:,2] *= image2.shape[3]
        boxes[:,1] *= image2.shape[2]
        boxes[:,3] *= image2.shape[2]
        
        # score
        print("-----------------------BOXES=",boxes,"----------------------------\n")
        print("-----------------------LABEL=",labels,"----------------------------\n")
        print("-----------------------SCORES=",scores,"----------------------------\n")
        scores1 = data_hook['scores_hooked'].to(torch.device("cpu"))
        
        entropy = uncertainty_entropy(torch.from_numpy(scores))
        
        dic[file_name]=[]
        for i in range(len(labels)):
            box_info = {'confidence score':np.float64(scores[i]),
                        'pred class':np.int64(labels[i]),
                        'pred box':boxes[i],
                        'entropy': entropy.cpu().detach().clone().item()
                        }
            dic[file_name].append(box_info)

        del res
        del image
        del data_hook['scores_hooked']
        del data_hook['boxes_hooked']
        torch.cuda.empty_cache()
    
   
    
    FILE_PATH=cfg.OUTPUT_DIR+"/e_static_by_random.json"
    #FILE_PATH=cfg.static_file
    print("FILE_PATH:",FILE_PATH)
    with open(FILE_PATH, 'w') as f:
        f.write(json.dumps(dic, default=str))
   


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument("--static_file",type=str,default='temp/coco/static_by_random.json') #Json file of the intermediate process
    parser.add_argument("--model_weights",type=str,default='output/model_best.pth')
    parser.add_argument("--config",type=str,default='configs/coco/faster_rcnn_R_50_FPN_sup20_run1.yaml')
   
    args = parser.parse_args()
    args.eval_only = True
    args.resume = True
    args.num_gpus = 1
    FILE_PATH = args.static_file 
    args.config_file=args.config 
    
    
    # you should config MODEL.WEIGHTS and keep other hyperparameters default(Odd-numbered items are keys, even-numbered items are values)
    args.opts = ['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5,'MODEL.ROI_HEADS.NMS_THRESH_TEST', 0.5,
     'TEST.DETECTIONS_PER_IMAGE', 20, 'INPUT.FORMAT', 'RGB','MODEL.WEIGHTS',args.model_weights,'OUTPUT_DIR',args.static_file ]
    print("Command Line Args:", args)
    main(args)
    
    