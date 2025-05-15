import json
import os
import argparse


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

#--------------------------------- Package imports For TTA --------------------------------------
import torchvision
import cv2
import matplotlib.pyplot as plt
from odach import *

import odach as oda

#------------------------------- TTA -----------------------
# Declare TTA Transformation here
#tta_transforms = [oda.VerticalFlip(), oda.Multiply(0.9), oda.Multiply(1.1)]
tta_transforms = [ oda.GaussianBlur(5, 0.1, 5), oda.HorizontalFlip(), oda.VerticalFlip(), oda.Multiply(0.9), oda.Multiply(1.2), oda.Contrast(0.7), oda.Contrast(1.3), oda.Contrast(1.5), oda.Rotate90Left(),oda.Rotate90Right()]

#tta_transforms =[oda.HorizontalFlip(), oda.VerticalFlip(),oda.Multiply(1)]
#scale = [0.8, 0.9, 1, 1.1, 1.2]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
    
def loadimg(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (1024,1024))
    img = img.transpose([2,0,1]) / 255 # 0-1 float!
    return torch.from_numpy(img).unsqueeze(0).to(device).float()

# function to perform TTA on an unlabelled image

@torch.no_grad()
def inference_TTA(tableImages,Trainer, cfg):
    print("----------------------------- INFERENCE WITH TTA -------------------------------------\n") 
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

    dic={}
    from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
    
    for elt in tableImages:
        file_name = elt
        print("---------- File:",file_name,"-----------------------\n")
        image = utils.read_image(file_name, format='BGR')
        image1 = torch.from_numpy(image.copy()).permute(2,0,1)
        
        image2= image1.unsqueeze(0).to(torch.device("cpu")).float()
        
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
            #print("--------------- ind =",ind,"----------------------\n")
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
    FILE_PATH=cfg.OUTPUT_DIR+"/d_static_by_random.json"
    #FILE_PATH=cfg.static_file
    print("FILE_PATH:",FILE_PATH)
    with open(FILE_PATH, 'w') as f:
        f.write(json.dumps(dic, default=str))
    
    #--------------------------- UPDATE dataset annotations-----------------------------------
    # create a new datasets/coco/annotations/coco_training_new.json
    # First make coco_training_new.json as a copy of datasets/coco/annotations/coco_training.json
    # for each elt i tableImages get filename and bbox
    #---------- File: datasets/coco/train/image17876.jpg
    # upadate the coco_training_new.json with the bbox for the elt



def generate_pick_merge_random(random_file, random_percent, indicator_file, pick_percent, save_file, static_file, cfg,reverse=True):
    P = []
    with open(indicator_file,'r') as f:
        items = f.readlines()
        P = [float(item) for item in items]
    #print("------------------- P=",P,"len(P)=",len(P),"----------------------\n")
    idx = sorted(range(len(P)), key=lambda k: P[k], reverse=reverse) #  idx is a list of the index of the images sorted from the highest combined active metric to the lowest combined active metric
    #print("------------------- idx=",idx,"len(idx)=",len(idx),"----------------------\n")

    total_imgs = len(idx)
    print('total images: ',total_imgs)
    dic={}
    
    Filtered_images_idx =[] # the index of the images having high Active Sampling score after pseudo labeling by the Teacher model.
     
    dic[str(random_percent+pick_percent)]={}
    print('total chosen initial images: ',len(dic))
    with open(random_file,'r') as f:
        table = json.load(f)
        for i in range(10):
            exist_idx = table[str(random_percent)][str(i)]
            #print("-----------i=",exist_idx,"-----------------\n")
            iddx = []
            for item in idx:
                if item not in exist_idx:  # select from the inference "results/c_coco/random_maxnorm.json" only the results where image is not in the "dataseed/COCO_supervision.txt "
                    iddx.append(item)
            left = int(total_imgs*(random_percent+pick_percent)/100) - len(table[str(random_percent)][str(i)])
            #print("-----------iddx[:left]=",iddx[:left],"------------len(iddx[:left])=",len(iddx[:left]))
            
            #---------------------------------------the TTA block starts from here ----------------------------------------
            TTA_images_idx= iddx[:left] # the index of the images having high Active Sampling score after pseudo labeling by the Teacher model.
            
            # 1) Get each image having it index. 2) Apply TTA transformation 3) Run each transformed image on the Teacher
            
            new_extension = ".json"
            # Split the filename and extension
            base, extension = os.path.splitext(indicator_file)
            
            # Replace the extension with the new extension
            new_filename = base + new_extension # get the .json file of the indicator_file
            with open(new_filename) as f:
              dataContent = json.load(f)
            
            tableImages=[]  # to keep the list of the images location of the speudo labeled images from Active sampling
            imagesLocations = list(dataContent.keys()) # get the keys which are the images locations
            for imgId in range(len(TTA_images_idx)):
                imageIndex=TTA_images_idx[imgId]
                image_url=imagesLocations[imageIndex] # get the corresponding image location
                #print("------------------------------- Image Location :",image_url,"-------------------------\n")
                tableImages.append(image_url)
                
            # get tresholds for TTA
            score_thresh=0.4
            iou_thr = 0.5
            skip_box_thr = 0.5 # prediction score 
            
            Trainer = ActiveTeacherTrainer
            #assert args.eval_only is True, "Inference should be eval only."
            # call Inference with TTA on the images list
            inference_TTA(tableImages,Trainer, cfg)
            
            
            # 4) Average the bounding box to get the final result from the TTA 5) Save this new bounding box for the corresponding image (may be update file "datasets/coco/annotations/coco_training.json")
            
            
            # the block should update more accurately the pseudo bounding boxes of the images which index are in this table iddx[:left].
            #------------------------------------ TTA Block end up here ----------------------------------------------------
            
            arr = iddx[:left] + table[str(random_percent)][str(i)]
            #print("-----------left=",left,"------------\n-------------------arr=",arr,"len(arr)=",len(arr))
            Filtered_images_idx= iddx[:left] # the index of the images having high Active Sampling score after pseudo labeling by the Teacher model.
            dic[str(random_percent+pick_percent)][str(i)] = arr
            

    with open(save_file,'w') as f:
        f.write(json.dumps(dic))
    #print('total chosen final images: ',len(dic))






    # --------------------------- UPDATING ANNOTATION WITH NEW PSEUDO LABELS BBBOX---------------------------
    
    new_extension = ".json"
    # Split the filename and extension
    base, extension = os.path.splitext(indicator_file)
            
    # Replace the extension with the new extension
    new_filename = base + new_extension # get the .json file of the indicator_file
    with open(new_filename) as f:
        dataContent = json.load(f)
            
    tableImages=[]  # to keep the list of the location of the speudo labeled images from Active sampling filtering
    imagesLocations = list(dataContent.keys()) # get the keys which are the images locations
    for imgId in range(len(Filtered_images_idx)):
        imageIndex=Filtered_images_idx[imgId]
        image_url=imagesLocations[imageIndex] # get the corresponding image location
        #print("------------------------------- Image Location :",image_url,"-------------------------\n")
        tableImages.append(image_url)
        
    #print("-----------------tableImages=",tableImages,"--------\n")
    #--------------------------- UPDATE dataset annotations-----------------------------------
    # create a new datasets/coco/annotations/coco_training_d_new.json
    # First make coco_training_new.json as a copy of datasets/coco/annotations/coco_training.json
    with open("datasets/coco/annotations/coco_training.json", "r") as data1:
        data1content=json.load(data1)
        
    with open("datasets/coco/annotations/coco_training_d_new.json", "w") as data2:
            json.dump(data1content, data2)
    
    # Open the prediction json file
    json_annot_prediction_data= {}
    static_file_d_new=static_file+"/d_static_by_random.json"
    with open(static_file_d_new) as f:
        json_annot_prediction_data = json.load(f)
    #print("-----------------Type json_annot_prediction_data=",type(json_annot_prediction_data),"--------\n")    
    # Open the new annotation json file    
    with open('datasets/coco/annotations/coco_training_d_new.json') as f:
        json_annot_data = json.load(f)
    
    json_annot_data_images=json_annot_data["images"] # get the images lists
    #print("-----------------Type json_annot_data_images=",type(json_annot_data_images),"--------\n")
    #print("-----------------json_annot_data_images=",json_annot_data_images,"--------\n")
    
    json_annot_data_annotations=json_annot_data["annotations"] # get the images predicted annotations lists
    #print("-----------------json_annot_data_annotations=",json_annot_data_annotations,"--------\n")
    
    id_new=0 # new id for detections
        
    # for each elt i tableImages get filename and bbox
    for jj in range(len(tableImages)):
        elt=tableImages[jj] # Image path
        imageName=os.path.basename(elt) # get only image name with extension need to "import os"
        print("-----------------elt image=",elt," --- imageName=",imageName,"--------\n")
        
        #***************************************
        bbox_predicted=[] # table for the list of bboxes detected
        pred_class_predicted=[] # table for the list of labels detected
        predictionList=json_annot_prediction_data[elt] #  get the prediction leist for the image
        
        for kk in predictionList:
            #print("------------type kk=", type(kk),"-----------\n")
            #print("------------kk=", kk,"-----------\n")
            bbox_predicted.append(kk.get("pred box"))# the predicted bboxes for the selected image
            #print("---------------pred box=",type(kk.get("pred box")),"----------------\n")
            pred_class_predicted.append(kk.get("pred class"))# the predicted bboxes for the selected image
         
        #print("---------------bbox_predicted=",bbox_predicted,"----------------\n")
        #print("---------------pred_class=",pred_class_predicted,"----------------\n")
        
       # get from  coco_training_new.json the "id" value for "imageName"
        idImage=""
        for item in json_annot_data_images:
            if item.get("file_name")==imageName:
                idImage=item["id"]
                break
        print("-----------------imageName=",imageName," --- idImage=",idImage,"--------\n")
        
        cptDelected=0
        # search idImage in the coco_training_new.json file 
        for eltt in json_annot_data_annotations:
            #print("-----------------this elt id_image=",eltt.get("image_id")," -----------\n")
            if eltt.get("image_id")==idImage:
                # remove this dict from the list
                json_annot_data_annotations.remove(eltt)
                cptDelected=cptDelected+1 # count number of element removed
            else:
                 id_new=eltt.get("id")
        #id_new=id_new - cptDelected # update 
        
        #print("--------------------bbox_predicted=",bbox_predicted,"-------TYPE=",type(bbox_predicted),"------\n")
        for iii in range(len(bbox_predicted)):
            
            bbox_predicted1=bbox_predicted[iii][1:-1] # remove the [ and ] charater at the begining and the end of the string
            bbox_predicted1=bbox_predicted1.strip() # remove first and last space from the string
            bbox_predicted2=bbox_predicted1.split(" ") # table of the predictions boxes
            #print("--------------------bbox_predicted2=",bbox_predicted2,"-------TYPE=",type(bbox_predicted2),"------\n")
            bbox_predicted3=[x.strip() for x in bbox_predicted2 if x.strip()]
            #print("--------------------bbox_predicted3=",bbox_predicted3,"-------TYPE=",type(bbox_predicted3),"------\n")
            the_area=(float(bbox_predicted3[2]) - float(bbox_predicted3[0])) * (float(bbox_predicted3[3]) - float(bbox_predicted3[1]))
            # Convert each item to float
            bbox_predicted3 = [float(item) for item in bbox_predicted3]

            id_new=id_new+1
            tab_elt={"iscrowd": 0, "ignore": 0, "image_id": idImage, "bbox":bbox_predicted3, "area": the_area, "segmentation": [], "category_id": pred_class_predicted[iii], "id": id_new}
            json_annot_data_annotations.append(tab_elt)
        
        # upadate the coco_training_new.json with the bbox for the elt
        # Replace "bbox" coordinantes by "bbox_predicted" where "image_id"=idImage
    
    categories=[{"supercategory": "none", "id": 0, "name": "0"}, {"supercategory": "none", "id": 1, "name": "1"},
    {"supercategory": "none", "id": 2, "name": "10"}, {"supercategory": "none", "id": 3, "name": "11"},
    {"supercategory": "none", "id":4 , "name": "2"}, {"supercategory": "none", "id": 5, "name": "3"}, 
    {"supercategory": "none", "id": 6, "name": "4"}, {"supercategory": "none", "id": 7, "name": "5"}, 
    {"supercategory": "none", "id": 8, "name": "6"}, {"supercategory": "none", "id": 9, "name": "7"}, 
    {"supercategory": "none", "id": 10, "name": "8"}, {"supercategory": "none", "id": 11, "name": "9"}]
    new_data_annotation={"images":json_annot_data_images , "annotations":json_annot_data_annotations, "categories":categories}
        
    # Replace the old content for predicted annotations by the new            
    with open("datasets/coco/annotations/coco_training_d_new.json",'w') as f:
        f.write(json.dumps(new_data_annotation))
            
            
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pick and merge label data partition')
    parser.add_argument("--random_file",type=str,default='dataseed/COCO_supervision.txt')
    parser.add_argument("--random_percent",type=float,default=10.0)
    parser.add_argument("--indicator_file",type=str,default='results/10random_maxnorm.txt')
    parser.add_argument("--pick_percent",type=float,default=10.0)
    parser.add_argument("--reverse",type=bool,default=True)
    parser.add_argument("--save_file",type=str,default='dataseed/pick_maxnorm10+random10.txt')
    
    parser.add_argument("--static_file",type=str,default='temp/coco/static_by_random.json') #Json file of the intermediate process
    parser.add_argument("--model_weights",type=str,default='output/model_best.pth')
    parser.add_argument("--config",type=str,default='configs/coco/faster_rcnn_R_50_FPN_sup35_run1.yaml')
    args = parser.parse_args()
    args.eval_only = True
    args.resume = True
    args.num_gpus = 1
    #FILE_PATH = "temp/coco/static_by_random.json"
    FILE_PATH =args.static_file
    args.config_file=args.config 
    
    # you should config MODEL.WEIGHTS and keep other hyperparameters default(Odd-numbered items are keys, even-numbered items are values)
    args.opts = ['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5,'MODEL.ROI_HEADS.NMS_THRESH_TEST', 0.5,
     'TEST.DETECTIONS_PER_IMAGE', 20, 'INPUT.FORMAT', 'RGB','MODEL.WEIGHTS',args.model_weights,'OUTPUT_DIR',FILE_PATH ]
    
    cfg = setup(args)
    print("------------cfg=",cfg,"-------------------------")
    generate_pick_merge_random(
        args.random_file,
        args.random_percent,
        args.indicator_file,
        args.pick_percent,
        args.save_file,
        args.static_file,
        cfg,
        args.reverse
    )