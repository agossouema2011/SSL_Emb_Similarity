import json
import numpy as np
import torch
import torch.nn.functional as F

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

@torch.no_grad()
def uncertainty_entropy(p):
    # p.size() = num_instances of a image, num_classes
    p = F.softmax(p, dim=1)
    p = - torch.log2(p) * p
    entropy_instances = torch.sum(p, dim=1)
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
    #register_coco_instances("my_dataset_inference", {}, "", "datasets/coco/inference") # register the custom dataset for inference; "my_dataset_inference" is the chosen percentage for the unlabled data
    register_coco_instances("my_dataset_train", {}, "datasets/coco/annotations/coco_training.json", "datasets/coco/train") # register the custom dataset for inference; "my_dataset_train" is the selected percentage for the unlabled data
    
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
    
    """
    DetectionCheckpointer(
        ensem_ts_model, save_dir=cfg.static_file
    ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    """
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
        image = torch.from_numpy(image.copy()).permute(2,0,1)
        res = ensem_ts_model.modelTeacher.inference([{'image':image}])
        
        print("--------------- res =",res,"----------------------\n")
        scores=res[0]['instances'].scores.to(torch.device("cpu")).detach().numpy()
        box = res[0]['instances'].pred_boxes.to(torch.device("cpu")).tensor.numpy()
        print("--------------- scores =",scores,"----------------------\n")
        print("--------------- box1 =",box,"----------------------\n")
            
        # score
        scores = data_hook['scores_hooked'].to(torch.device("cpu"))
        entropy = uncertainty_entropy(scores)

        dic[file_name]=[]
        for i in range(len(res[0]['instances'])):
            box_info = {'confidence score':np.float64(res[0]['instances'].scores.cpu().detach().numpy()[i]),
                        'pred class':np.int64(res[0]['instances'].pred_classes.cpu().detach().numpy()[i]),
                        'pred box':res[0]['instances'].pred_boxes.tensor[i].cpu().detach().numpy().tolist(),
                        'entropy': entropy.cpu().detach().clone().item()
                        }
            dic[file_name].append(box_info)

        del res
        del image
        del data_hook['scores_hooked']
        del data_hook['boxes_hooked']
        torch.cuda.empty_cache()
    FILE_PATH=cfg.OUTPUT_DIR+"/static_by_random.json"
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
    #print("PATH=",FILE_PATH)
    #args.config_file = 'configs/coco/faster_rcnn_R_50_FPN_sup20_run1.yaml' #the config file you used to train this inference model
    args.config_file=args.config 
    
    # you should config MODEL.WEIGHTS and keep other hyperparameters default(Odd-numbered items are keys, even-numbered items are values)
    args.opts = ['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5,'MODEL.ROI_HEADS.NMS_THRESH_TEST', 0.5,
     'TEST.DETECTIONS_PER_IMAGE', 20, 'INPUT.FORMAT', 'RGB','MODEL.WEIGHTS',args.model_weights,'OUTPUT_DIR',args.static_file ]
    print("Command Line Args:", args)
    main(args)