# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
import cv2
from info_nce import InfoNCE, info_nce

from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.structures import PolygonMasks, ROIMasks, BitMasks
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from activeteacher.data.build import (
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from activeteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from activeteacher.engine.hooks import LossEvalHook
from activeteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from activeteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from activeteacher.solver.build import build_lr_scheduler

from torchvision import transforms
from PIL import Image, ImageDraw

from img2vec_pytorch import Img2Vec

import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
 
from PIL import Image
import torchvision.transforms.functional as con
from PIL import Image, ImageEnhance
from PIL import Image, ImageFilter

import logging

# Configure the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

#-------------------------------------------------- Added functions --------------------------------------------------------

import torch
import numpy as np
import json

# Function to get the largest bboxe which is the box having minimum and maximum x, y coordinates
def get_largest_bounding_box(bboxes):
    x_min = min(bbox[0] for bbox in bboxes)  # Minimum x (left)
    y_min = min(bbox[1] for bbox in bboxes)  # Minimum y (top)
    x_max = max(bbox[2] for bbox in bboxes)  # Maximum x (right)
    y_max = max(bbox[3] for bbox in bboxes)  # Maximum y (bottom)
    return x_min, y_min, x_max, y_max

    
def crop_image(image, bbox):  # get read image with PIL.open() and the bbox cordinates and return the cropped image 
    # Crop the image using the bounding box coordinates
    cropped_image = image.crop(bbox)
    # Ensure the crop is at least 224x224
    min_size = (224, 224)
    cropped_image = cropped_image.resize(min_size, Image.Resampling.LANCZOS)
    return cropped_image

# This function creates empty image of 224x224
def empty_image():
    # Define the size and color (white) of the empty image
    width, height = 224, 224
    color = (255, 255, 255)  # RGB for white
    # Create an empty (white) image
    empty_image = Image.new("RGB", (width, height), color)
"""
def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2
"""

# Function to convert (x, y, width, height) to (x1, y1, x2, y2)
def convert_bbox(x, y, width, height):
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    return x1, y1, x2, y2
    
class Boxes_ext:
    def __init__(self, tensor):
        self.tensor = tensor

class Instances_ext:
    def __init__(self, num_instances, image_height, image_width, fields):
        self.num_instances = num_instances
        self.image_height = image_height
        self.image_width = image_width
        self.fields = fields

# Function to extract 'Boxes' array values from each instance
def extract_boxes(instance):
    #boxes_list = []
    #print("---------------------------- instance*******=",instance,"----------TYPE=",type(instance),"------------\n")
    gt_boxes = instance.gt_boxes.tensor.cpu().numpy()  # Convert tensor to numpy array and move to CPU
    #boxes_list.append(gt_boxes)
    return gt_boxes
    
# Function to convert arrays to tuples
def convert_boxes_to_tuples(boxes_list):
    return [tuple(box.flatten()) for box in boxes_list if box.size > 0]
    
    
# Function to draw a bounding box on the image
def draw_bounding_box(pil_image, bbox):
    # Create an ImageDraw object
    draw = ImageDraw.Draw(pil_image)
    # Draw the bounding box
    draw.rectangle(bbox, outline="red", width=3)
    return pil_image
    
# Function to read the image and convert it to a tensor
def read_image(image_path):
    image = Image.open(image_path)
    #resized_image=image.resize((224,224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image)
    return image_tensor

# Function to get image IDs, image names, and bounding boxes for each category
def get_image_info_per_category(coco_data):
    category_dict = {}
    # Initialize the category dictionary
    for category in coco_data['categories']:
        category_dict[category['id']] = {
            'name': category['name'],
            'images': []
        }

    # Create a dictionary for image names by their IDs
    image_id_to_name = {image['id']: image['file_name'] for image in coco_data['images']}

    # Iterate through annotations and populate the dictionary
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        image_name = image_id_to_name.get(image_id)

        category_dict[category_id]['images'].append({
            'image_id': image_id,
            'image_name': image_name,
            'bbox': bbox
        })

    return category_dict


# Fuction to get a random interger between 0 and number of pathology class excluding a given number
import random

def select_random_excluding(excludeTab, nbClass):
    choices = [i for i in range(nbClass) if i not in excludeTab]
    return choices
    #sample_size = min(3, len(choices))
    #return random.sample(choices, k=sample_size)


   
# Function to get labeled images of given classes with their annontations. Get the positives and the negative samples
def get_same_class_labelled_images(predClasses):
    # Example usage
    with open('./././datasets/coco/annotations/coco_training.json', 'r') as f:
        coco_data = json.load(f)
    
    category_dict = get_image_info_per_category(coco_data)
    
    # Print the result
    selected_images_ids=[]
    selected_images=[]
    selected_images_bbox=[]
    for category_id, data in category_dict.items():
        #print(f"Category: {data['name']} (ID: {category_id})")
        image_info=data['images'][0]
        selected_images_ids.append(image_info['image_id'])
        selected_images.append(image_info['image_name'])
        selected_images_bbox.append(image_info['bbox'])
    #print("----------------------- selected_images=",selected_images,"\n")
    #print("------------------------ selected_images_bbox=",selected_images_bbox,"\n")
    #print("------------------------ predClasses 111 =",predClasses,"\n")
    
    labeled_image=[]
    labeled_bbox=[]
    
    neg_labeled_image=[]
    neg_labeled_bbox=[]
    
    for i in range(len(predClasses)):
        indice_class=predClasses[i]
        #
        # Check in the initial labeled txt file and return file_path, the_labeled_bbox
        file_link="./././datasets/coco/train/"+selected_images[indice_class]
        #print("------------------------  file_link selected_image=",file_link,"\n")
        the_labeled_image=Image.open(file_link)
        labeled_image.append(the_labeled_image)
        labeled_bbox.append(selected_images_bbox[indice_class])
    
    neg_indice_classes=select_random_excluding(predClasses, 8) # get classID for all the negatives (pathologies which are not present in the detected pathologies list)
        
    for j in range(len(neg_indice_classes)):
        neg_indice_class=neg_indice_classes[j]
        #
        # Check in the initial labeled txt file and return file_path, the_labeled_bbox
        neg_file_link="./././datasets/coco/train/"+selected_images[neg_indice_class] # get the image link for a negative sample which is not from the samen pathology class
        the_neg_labeled_image=Image.open(neg_file_link)
        neg_labeled_image.append(the_neg_labeled_image)
        neg_labeled_bbox.append(selected_images_bbox[neg_indice_class])  
    
    return labeled_image,labeled_bbox,neg_labeled_image,neg_labeled_bbox
    

# Function to combine two images side by side. The two images are already read with PIL function Image.open()
def combine_images(image1, image2):
    # Get dimensions
    width1, height1 = image1.size
    width2, height2 = image2.size
    # Create a new image with combined width and max height
    combined_image = Image.new("RGB", (width1 + width2, max(height1, height2)))
    # Paste the two images into the combined image
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1, 0))
    return combined_image
    

# Function to perform a horizontal flip
def horizontal_flip(image_tensor):
    transform = transforms.RandomHorizontalFlip(p=1)  # p=1.0 ensures the image is always flipped
    H_flipped_image_tensor = transform(image_tensor)
    return H_flipped_image_tensor
    
def vertical_flip(image_tensor):
    transform = transforms.RandomVerticalFlip(p=1)  # p=1.0 ensures the image is always flipped
    V_flipped_image_tensor = transform(image_tensor)
    return V_flipped_image_tensor

def gaussian_blur(input_img,kernel_size,thesigma):
    from torchvision.transforms import v2
    # Image Gaussian blur
    blurrer = v2.GaussianBlur(kernel_size=kernel_size, sigma=thesigma)
    "kernel_size (int or sequence) – Size of the Gaussian kernel. " \
    "sigma (float or tuple of python:float (min, max)) – " \
    "Standard deviation to be used for creating kernel to perform blurring. " \
    "If float, sigma is fixed. If it is tuple of float (min, max), " \
    "sigma is chosen uniformly at random to lie in the given range." \
    "The size of the Gaussian kernel depends on the noise level in the image. If the kernel size is too large, small features within the image may get suppressed, and the image may look blurred." \
    " Hence, the quality of the details of the image will be affected."
    output_img = blurrer(input_img)
    return output_img 
    
    
def constrast_enhancement(input_img, contast_level):
    # adjust the contrast of the image
    output_img = con.adjust_contrast(input_img, contast_level)
    return output_img
    
    
def brightness_enhancement(input_img, brightness_level):
    # adjust the contrast of the image
    output_img = con.adjust_brightness (input_img, brightness_level)
    return output_img


def constrast_enh(image,level):
    # Enhance the contrast of the image
    enhancer = ImageEnhance.Contrast(image)
    contrast_enhanced_image = enhancer.enhance(level)  # Adjust the factor (e.g., 1.5) for desired enhancement
    # Transpose the image (e.g., flip left-to-right)
    #transposed_image = contrast_enhanced_image.transpose(Image.FLIP_LEFT_RIGHT)
    return contrast_enhanced_image
    
def brightness_enh(image, level):
    # Enhance the brightness of the image
    enhancer = ImageEnhance.Brightness(image)
    brightness_enhanced_image = enhancer.enhance(level)  # Adjust the factor (e.g., 1.5) for desired enhancement
    # Transpose the image (e.g., flip top-to-bottom)
    #transposed_image = brightness_enhanced_image.transpose(Image.FLIP_TOP_BOTTOM)
    return brightness_enhanced_image


# Create a Gaussian kernel with different sigma values for x and y
def gaussian_kernel(size, sigma_x, sigma_y):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma_x*sigma_y)) * np.exp(-((x-(size-1)/2)**2 / (2*sigma_x**2) + (y-(size-1)/2)**2 / (2*sigma_y**2))),
        (size, size)
    )
    return kernel / np.sum(kernel)

from PIL import Image, ImageFilter
import numpy as np

def gaussian_Blurr(image,kernel_size,sigma_x,sigma_y):
    # Apply Gaussian Blur with custom kernel
    kernel = gaussian_kernel(kernel_size, sigma_x, sigma_y)
    gaussian_filter = ImageFilter.Kernel((kernel_size, kernel_size), kernel.flatten(), scale=np.sum(kernel))
    blurred_image = image.filter(gaussian_filter)
    return blurred_image


# -------Loss function -----------------
# We use the Unsupervised InfoNCE available at https://github.com/arashkhoeini/infonce/tree/main
#from infonce import InfoNCE
from info_nce import InfoNCE, info_nce

def loss_func(query, positive_key,negative_keys):
    query=np.array(query) # convert list into numpy array first
    query=query.reshape(1, len(query)) # convert 1 dimenssional array into 2 dimensions
    query=torch.from_numpy(query) # convert the numpy arrays into tensors
    
    positive_key=np.array(positive_key) # convert list into numpy array first
    positive_key=positive_key.reshape(1, len(positive_key))
    positive_key=torch.from_numpy(positive_key)
    
    ng_elt=len(negative_keys)
    negative_keys=np.array(negative_keys) # convert list into numpy array first
    negative_keys=negative_keys.reshape(1, len(negative_keys))
    negative_keys=torch.from_numpy(negative_keys)
    
    #print("\n----------- Dim query=",query.dim(),"------------ Dim positive_key=",positive_key.dim(),"------------ Dim negative_keys=",negative_keys.dim(),"----------\n" )
    #print("\n----------- Query=",query,"\n\n------------  positive_key=",positive_key,"\n\n------------ negative_keys=",negative_keys,"----------\n" )
    loss = InfoNCE(temperature=0.1, negative_mode='unpaired') # negative_mode='unpaired' is the default value
    theloss =0
    if ng_elt==0:
        theloss =0
    else:
        theloss =loss(query, positive_key, negative_keys).item() # the result is a tensor
    return theloss
    
# Function to convert tensor values to integers
def convert_to_int(tensor):
    int_tensor = tensor.mul(255).byte()
    return int_tensor
    
# function to generate the tensor from a PIL image
def tensor_values(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    theimage_tensor = transform(image)
    return theimage_tensor
#--------------------------------------------- End added Function block -----------------------------------------
    
class ActiveTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        if cfg.TEST.EVALUATOR == 'COCOeval':
            evaluator_type = "coco"
        elif cfg.TEST.EVALUATOR == 'VOCeval':
            evaluator_type = "pascal_voc"
        else:
            raise NotImplementedError(
                "Evaluator for the dataset {} with the type {} not implemented".format(
                    dataset_name, evaluator_type
                )
            )

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    #print("-------------------- ITTERATION Nb=",self.iter,"-----------\n")
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

            # add masks to instances
            if hasattr(proposal_bbox_inst,'pred_masks'):
                roi_masks = ROIMasks(proposal_bbox_inst.pred_masks[valid_map, 0, :, :])
                mask_threshold = 0.5
                bitmasks = roi_masks.to_bitmasks(
                    new_proposal_inst.gt_boxes, image_shape[0], image_shape[1], mask_threshold
                )
                polygons_masks = [self.mask_to_polygons(bitmask)[0] for bitmask in bitmasks.tensor.cpu().numpy()]
                new_proposal_inst.gt_masks = PolygonMasks(polygons_masks)

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

  
    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[ActiveTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        
        #data_loader = self._trainer._data_loader_iter
      
        data = next(self._trainer._data_loader_iter)
        
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # remove unlabeled data labels
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)
        
        
        #---------------- ------------- New Bloc strats from here ----------------------------------------------------
        original_images=[] # array of unlabeled images
        image_ids=[]  # array of unlabeled images image_ids
        #print("-------------------- original unlabel_data_k len=",len(unlabel_data_k)," ------------------------\n")
        for ii in range(len(unlabel_data_k)):
            original_images.append(unlabel_data_k[ii]['file_name']) # Get the original unlabel image name 
            image_ids.append(unlabel_data_k[ii]['image_id']) # Get the original unlabel image id
        
        images=[] #  Array for the Original unlabeled labeled images
        H_flipped_images=[] # TArray for the horizontal fliped unlabeled labeled images
        V_flipped_images=[] # Arrays for the vertical fliped unlabeled labeled images
        C1_images=[] # Arrays for the Contrast enhanced 1
        C2_images=[] # Arrays for the Contrast enhanced 2
        C3_images=[] # Arrays for the Contrast enhanced 3
        B1_images=[] # Arrays for the Brigthness enhanced 1
        B2_images=[] # Arrays for the Brigthness enhanced 2
        GB1_images=[] # Arrays for the Gaussian Blured 1
        
        thewidth=0  # image width
        theheight=0     # image height
        
        for ii in range(len(original_images)):
            link_elt="./././"+original_images[ii]
            #print("-------------------- original unlabel images Link=",link_elt," ------------------------\n")
            theimage=Image.open(link_elt)
            thewidth,theheight = theimage.size 
            images.append(theimage)
            H_flip_img = theimage.transpose(Image.FLIP_LEFT_RIGHT) # horizontal fliping
            V_flip_img = theimage.transpose(Image.FLIP_TOP_BOTTOM) # vertical fliping
            H_flipped_images.append(H_flip_img)
            V_flipped_images.append(V_flip_img)
            
            enhancer = ImageEnhance.Contrast(theimage)
            C1_images.append(enhancer.enhance(0.7))
            enhancer = ImageEnhance.Contrast(theimage)
            C2_images.append(enhancer.enhance(1.3))
            enhancer = ImageEnhance.Contrast(theimage)
            C3_images.append(enhancer.enhance(1.5))

            enhancer1 = ImageEnhance.Brightness(theimage)
            B1_images.append(enhancer1.enhance(0.8))
            enhancer1 = ImageEnhance.Brightness(theimage)
            B2_images.append(enhancer1.enhance(1.2))
            GB1_images.append(gaussian_Blurr(theimage,5,0.1,5))
        
        image_tensors=[] # Tensor vectors for the Original unlabeled labeled images
        H_flipped_image_tensors=[] # Tensor vectors for the horizontal fliped unlabeled labeled images
        V_flipped_image_tensors=[] # Tensor vectors for the vertical fliped unlabeled labeled images
        C_enhancement1_image_tensors=[] # Tensor vectors for the contrast enhancement unlabeled labeled images
        C_enhancement2_image_tensors=[] # Tensor vectors for the contrast enhancement unlabeled labeled images
        C_enhancement3_image_tensors=[] # Tensor vectors for the contrast enhancement unlabeled labeled images
        B_enhancement1_image_tensors=[] # Tensor vectors for the Brightness enhancement unlabeled labeled images
        B_enhancement2_image_tensors=[] # Tensor vectors for the Brightness enhancement unlabeled labeled images
        GaussianBlur1_image_tensors=[] # Tensor vectors for the Gaussian Blur unlabeled labeled images
        
        
        for ii in range(len(original_images)):
            link_elt="./././"+original_images[ii]
            #print("-------------------- original unlabel images Link=",link_elt," ------------------------\n")
            image_tensor=convert_to_int(read_image(link_elt))
            image_tensors.append(image_tensor)
            H_flipped_image_tensors.append(horizontal_flip(image_tensor))
            V_flipped_image_tensors.append(vertical_flip(image_tensor))
            C_enhancement1_image_tensors.append(constrast_enhancement(image_tensor,0.7))
            C_enhancement2_image_tensors.append(constrast_enhancement(image_tensor,1.3))
            C_enhancement3_image_tensors.append(constrast_enhancement(image_tensor,1.5))
            B_enhancement1_image_tensors.append(brightness_enhancement(image_tensor,0.8))
            B_enhancement2_image_tensors.append(brightness_enhancement(image_tensor,2))
            GaussianBlur1_image_tensors.append(gaussian_blur(image_tensor,(5,5),(0.1,5)))
            
        #weak unlabel_data_k = [{'file_name': 'datasets/coco/train/image08649.jpg', 'height': 576, 'width': 576, 'image_id': 4590, 'image': tensor([[[ 0,  0,  0,  ..., 16, 16, 16]]])}]
        
        # put the original images and the flipped images into the same data type and format as unlabel_data_k for the teacher model to process them
        
        copy_unlabel_data_k=[]
        H_flipped_unlabel_data_k=[]
        V_flipped_unlabel_data_k=[]
        C_enhancement1_unlabel_data_k=[]
        C_enhancement2_unlabel_data_k=[]
        C_enhancement3_unlabel_data_k=[]
        B_enhancement1_unlabel_data_k=[]
        B_enhancement2_unlabel_data_k=[]
        GaussianBlur1_unlabel_data_k=[]
        
        
        for ii in range(len(image_tensors)):
            theimage=original_images[ii]
            theimage_id=image_ids[ii]
            H_flip_image_id=theimage_id*100000
            V_flip_image_id=theimage_id*1000000
            C1_image_id=theimage_id*10000000
            C2_image_id=theimage_id*100000000
            C3_image_id=theimage_id*1000000000
            B1_image_id=theimage_id*10000000000
            B2_image_id=theimage_id*100000000000
            GB1_image_id=theimage_id*1000000000000
            
            elt_img={}
            elt_H_flip_img={}
            elt_V_flip_img={}
            elt_C1_img={}
            elt_C2_img={}
            elt_C3_img={}
            elt_B1_img={}
            elt_B2_img={}
            elt_GB1_img={}
            
            elt_img = {'file_name': theimage, 'height': theheight, 'width': thewidth, 'image_id':theimage_id , 'image': image_tensors[ii]}
            elt_H_flip_img = {'file_name': theimage, 'height': theheight, 'width': thewidth, 'image_id':H_flip_image_id , 'image': H_flipped_image_tensors[ii]}
            elt_V_flip_img = {'file_name': theimage, 'height': theheight, 'width': thewidth, 'image_id': V_flip_image_id , 'image': V_flipped_image_tensors[ii]}
            elt_C1_img = {'file_name': theimage, 'height': theheight, 'width': thewidth, 'image_id': C1_image_id , 'image': C_enhancement1_image_tensors[ii]}
            elt_C2_img = {'file_name': theimage, 'height': theheight, 'width': thewidth, 'image_id': C2_image_id , 'image': C_enhancement2_image_tensors[ii]}
            elt_C3_img = {'file_name': theimage, 'height': theheight, 'width': thewidth, 'image_id': C3_image_id , 'image': C_enhancement3_image_tensors[ii]}
            elt_B1_img = {'file_name': theimage, 'height': theheight, 'width': thewidth, 'image_id': B1_image_id , 'image': B_enhancement1_image_tensors[ii]}
            elt_B2_img = {'file_name': theimage, 'height': theheight, 'width': thewidth, 'image_id': B2_image_id , 'image': B_enhancement2_image_tensors[ii]}
            elt_GB1_img = {'file_name': theimage, 'height': theheight, 'width': thewidth, 'image_id': GB1_image_id , 'image': GaussianBlur1_image_tensors[ii]}
            
            copy_unlabel_data_k.append(elt_img)
            H_flipped_unlabel_data_k.append(elt_H_flip_img)
            V_flipped_unlabel_data_k.append(elt_V_flip_img)
            C_enhancement1_unlabel_data_k.append(elt_C1_img)
            C_enhancement2_unlabel_data_k.append(elt_C2_img)
            C_enhancement3_unlabel_data_k.append(elt_C3_img)
            B_enhancement1_unlabel_data_k.append(elt_B1_img)
            B_enhancement2_unlabel_data_k.append(elt_B2_img)
            GaussianBlur1_unlabel_data_k.append(elt_GB1_img)
            
        #-------------------------------------New Bloc ends here -----------------------------------------------
        
       
            
        
        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
            
            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
            
            
            
            
            
            
            #--------------------------------------------- Pseudo labeling for the copy of original images --------------------
            with torch.no_grad():
                (
                    _,
                    copy_proposals_rpn_unsup_k,
                    copy_proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(copy_unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            copy_joint_proposal_dict = {}
            copy_joint_proposal_dict["proposals_rpn"] = copy_proposals_rpn_unsup_k
            (
                copy_pesudo_proposals_rpn_unsup_k,
                copy_nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                copy_proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            copy_joint_proposal_dict["proposals_pseudo_rpn"] = copy_pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            copy_pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                copy_proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            copy_joint_proposal_dict["proposals_pseudo_roih"] = copy_pesudo_proposals_roih_unsup_k
            
            
            
            
            #--------------------------------------------- Pseudo labeling for the Horizontal Fliped for the copy of original images --------------------
            with torch.no_grad():
                (
                    _,
                    H_flipped_proposals_rpn_unsup_k,
                    H_flipped_proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(H_flipped_unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            H_flipped_joint_proposal_dict = {}
            H_flipped_joint_proposal_dict["proposals_rpn"] = H_flipped_proposals_rpn_unsup_k
            (
                H_flipped_pesudo_proposals_rpn_unsup_k,
                H_flipped_nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                H_flipped_proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            H_flipped_joint_proposal_dict["proposals_pseudo_rpn"] = H_flipped_pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            H_flipped_pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                H_flipped_proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            H_flipped_joint_proposal_dict["proposals_pseudo_roih"] = H_flipped_pesudo_proposals_roih_unsup_k
            
            
            #--------------------------------------------- Pseudo labeling for the Vertical Fliped for the copy of original images --------------------
            with torch.no_grad():
                (
                    _,
                    V_flipped_proposals_rpn_unsup_k,
                    V_flipped_proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(V_flipped_unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            V_flipped_joint_proposal_dict = {}
            V_flipped_joint_proposal_dict["proposals_rpn"] = V_flipped_proposals_rpn_unsup_k
            (
                V_flipped_pesudo_proposals_rpn_unsup_k,
                V_flipped_nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                V_flipped_proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            V_flipped_joint_proposal_dict["proposals_pseudo_rpn"] = V_flipped_pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            V_flipped_pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                V_flipped_proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            V_flipped_joint_proposal_dict["proposals_pseudo_roih"] = V_flipped_pesudo_proposals_roih_unsup_k
            
            
            #--------------------------------------------- Pseudo labeling for the contrast 1 for the copy of original images --------------------
            with torch.no_grad():
                (
                    _,
                    C1_proposals_rpn_unsup_k,
                    C1_proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(C_enhancement1_unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            C1_joint_proposal_dict = {}
            C1_joint_proposal_dict["proposals_rpn"] = C1_proposals_rpn_unsup_k
            (
                C1_pesudo_proposals_rpn_unsup_k,
                C1_nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                C1_proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            C1_joint_proposal_dict["proposals_pseudo_rpn"] = C1_pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            C1_pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                C1_proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            C1_joint_proposal_dict["proposals_pseudo_roih"] = C1_pesudo_proposals_roih_unsup_k
          
            
            
            #--------------------------------------------- Pseudo labeling for the contrast 2 for the copy of original images --------------------
            with torch.no_grad():
                (
                    _,
                    C2_proposals_rpn_unsup_k,
                    C2_proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(C_enhancement2_unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            C2_joint_proposal_dict = {}
            C2_joint_proposal_dict["proposals_rpn"] = C2_proposals_rpn_unsup_k
            (
                C2_pesudo_proposals_rpn_unsup_k,
                C2_nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                C2_proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            C2_joint_proposal_dict["proposals_pseudo_rpn"] = C2_pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            C2_pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                C2_proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            C2_joint_proposal_dict["proposals_pseudo_roih"] = C2_pesudo_proposals_roih_unsup_k
            
            
            
            #--------------------------------------------- Pseudo labeling for the contrast 3 for the copy of original images --------------------
            with torch.no_grad():
                (
                    _,
                    C3_proposals_rpn_unsup_k,
                    C3_proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(C_enhancement3_unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            C3_joint_proposal_dict = {}
            C3_joint_proposal_dict["proposals_rpn"] = C3_proposals_rpn_unsup_k
            (
                C3_pesudo_proposals_rpn_unsup_k,
                C3_nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                C3_proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            C3_joint_proposal_dict["proposals_pseudo_rpn"] = C3_pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            C3_pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                C3_proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            C3_joint_proposal_dict["proposals_pseudo_roih"] = C3_pesudo_proposals_roih_unsup_k
            
            
            
            #--------------------------------------------- Pseudo labeling for the Brightness 1 for the copy of original images --------------------
            with torch.no_grad():
                (
                    _,
                    B1_proposals_rpn_unsup_k,
                    B1_proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(B_enhancement1_unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            B1_joint_proposal_dict = {}
            B1_joint_proposal_dict["proposals_rpn"] = B1_proposals_rpn_unsup_k
            (
                B1_pesudo_proposals_rpn_unsup_k,
                B1_nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                B1_proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            B1_joint_proposal_dict["proposals_pseudo_rpn"] = B1_pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            B1_pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                B1_proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            B1_joint_proposal_dict["proposals_pseudo_roih"] = B1_pesudo_proposals_roih_unsup_k
            
            
            
            #--------------------------------------------- Pseudo labeling for the Brightness 2 for the copy of original images --------------------
            with torch.no_grad():
                (
                    _,
                    B2_proposals_rpn_unsup_k,
                    B2_proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(B_enhancement2_unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            B2_joint_proposal_dict = {}
            B2_joint_proposal_dict["proposals_rpn"] = B2_proposals_rpn_unsup_k
            (
                B2_pesudo_proposals_rpn_unsup_k,
                B2_nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                B2_proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            B2_joint_proposal_dict["proposals_pseudo_rpn"] = B2_pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            B2_pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                B2_proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            B2_joint_proposal_dict["proposals_pseudo_roih"] = B2_pesudo_proposals_roih_unsup_k
            
            
            
            
            
            #--------------------------------------------- Pseudo labeling for the Gaussian Blur 1  for the copy of original images --------------------
            with torch.no_grad():
                (
                    _,
                    GB1_proposals_rpn_unsup_k,
                    GB1_proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(GaussianBlur1_unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            GB1_joint_proposal_dict = {}
            GB1_joint_proposal_dict["proposals_rpn"] = GB1_proposals_rpn_unsup_k
            (
                GB1_pesudo_proposals_rpn_unsup_k,
                GB1_nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                GB1_proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            GB1_joint_proposal_dict["proposals_pseudo_rpn"] = GB1_pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            GB1_pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                GB1_proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            GB1_joint_proposal_dict["proposals_pseudo_roih"] = GB1_pesudo_proposals_roih_unsup_k
            
            
            
            
            
            
            # -------------------------------------- Using Img2Vec for processing images embeddings starts here -----------------------------------------
            
            # For each image extract the bbounding boxes coordinates as tuples and draw them on the image
            
            loss_tab=[] # array of the similarity scores of the pseudo labelled images embeddings
            #   -------------------------------------- Initialize Img2Vec with GPU for processing images embeddings -----------------------------------------
            img2vec = Img2Vec(cuda=True, model='resnet50', layer='default', layer_output_size=2304) # call for the Feacture extractor
            #img2vec = Img2Vec(cuda=True, model='densenet', layer='default', layer_output_size=2304) # call for the Feacture extractor
            print("------------------- Feacture Extractor Model=",img2vec.model_name,"------------------------------\n")
            
            vec_labeled_images=[] # array of the embeddings for selected labeled images having the same classes as the predicted classes
            neg_vec_labeled_images=[] 
            vec_H_labeled_images=[] # Horizontal flip image array
            vec_V_labeled_images=[]
            vec_C1_labeled_images=[]
            vec_C2_labeled_images=[]
            vec_C3_labeled_images=[]
            vec_B1_labeled_images=[]
            vec_B2_labeled_images=[]
            vec_GB1_labeled_images=[]
            
            for jj in range(len(pesudo_proposals_roih_unsup_k)):
                vec_labeled=[] # array of the embeddings of the selected labeled images having same classe(s) as predicted on this unlabled image 
                neg_vec_labeled=[]
                vec_H_labeled=[]
                vec_V_labeled=[]
                vec_C1_labeled=[]
                vec_C2_labeled=[]
                vec_C3_labeled=[]
                vec_B1_labeled=[]
                vec_B2_labeled=[]
                vec_GB1_labeled=[]
                
                
                image_path = images[jj]  #  your original image array, image read with PIL (Image.open())
                copy_image_path = images[jj]  #  The image copy array
                H_flipped_image_path=H_flipped_images[jj] #  The array of the image copy horizontal flipped
                V_flipped_image_path=V_flipped_images[jj] #  The array of the image copy vertical flipped
                
                C1_image_path=C1_images[jj]
                C2_image_path=C2_images[jj]
                C3_image_path=C3_images[jj]
                B1_image_path=B1_images[jj]
                B2_image_path=B2_images[jj]
                GB1_image_path=GB1_images[jj]
                
                #print("-------------Org_Bbox ",jj," =",pesudo_proposals_roih_unsup_k[jj],"---------------------\n")
                #
                #
                # --------------Get the Predictions Classes for the unlabeled image and get the corresponding labeled image embedding---------------
                
                predClasses = pesudo_proposals_roih_unsup_k[jj].gt_classes.cpu().numpy()  # Convert tensor to numpy array and move to CPU
                #print("------------------------------ predClasses=",predClasses,"------------------------\n")
                labeled_Images, labeled_bbox,neg_labeled_Images, neg_labeled_bbox= get_same_class_labelled_images(predClasses)  # get the same classes labeled images and their bounding boxes
                
                #print("------------------labeled_Images=",labeled_Images,"-------------------------\n")
                #print("------------------labeled_bbox=",labeled_bbox,"-------------------------\n")
                labeled_bbox_array= np.array(labeled_bbox) # convert list into numpy array
                labeled_bbox_tuples_values = convert_boxes_to_tuples(labeled_bbox_array) # convert labeled bboxes into tuples
                #print("------------------labeled_bbox_tuples_values=",labeled_bbox_tuples_values,"-------------------------\n")
                # if there are bboxes for the labeled image
                labeled_cropped_images=[] # Array of cropped labeled images
                NbItems=len(labeled_bbox_tuples_values)
                
                neg_labeled_bbox_array= np.array(neg_labeled_bbox) # convert list into numpy array
                neg_labeled_bbox_tuples_values = convert_boxes_to_tuples(neg_labeled_bbox_array) # convert labeled bboxes into tuples
                neg_labeled_cropped_images=[] # Array of cropped negative sample of the labeled images
                neg_NbItems=len(neg_labeled_bbox_tuples_values)
                
                if NbItems!= 0:
                    for iii in range(len(labeled_Images)): 
                        bbbox=labeled_bbox_tuples_values[iii]
                        xmin=bbbox[0]
                        ymin=bbbox[1]
                        w=bbbox[2]
                        h=bbbox[3]
                        x1,y1,x2,y2=convert_bbox(xmin,ymin,w,h)
                        #x1,y1,x2,y2=yolobbox2bbox(xmin,ymin,w,h)
                            
                        bbox = (x1,y1,x2, y2) #  the coordinates
                        #print("------------------bbox=",bbox,"-------------------------\n")
                        the_image_path=labeled_Images[iii] # get the image array (PIL.open())
                        # Crop the image using the bounding box coordinates
                        labeled_cropped_image= the_image_path.crop(bbox)
                        labeled_cropped_images.append(labeled_cropped_image)
                    # combined all the cropped labeled images for all the detected classes
                   
                    combined_image =labeled_cropped_images[0]
                    #combined_image = combine_images(labeled_cropped_images[0], labeled_cropped_images[1]) # combined all the cropped selected labeled images for all the detected classes
                    if NbItems >= 2:
                        cpt=NbItems-1
                        while cpt>0:
                            combined_image=combine_images(combined_image, labeled_cropped_images[NbItems-cpt])
                            cpt=cpt-1
               
                    resized_labeled_images = combined_image.resize((224,224), Image.Resampling.LANCZOS) # final labeled image containing all the cropped from all the detected classes
                    vec_labeled = img2vec.get_vec(resized_labeled_images, tensor=False) # get the embeddings vectors
                    vec_labeled = np.array(vec_labeled) # convert list into array to be able to use reshape()
                    vec_labeled_images.append(vec_labeled)    
                    
                    # Perform the horizontal flip
                    H_flipped_image_Labeled = resized_labeled_images.transpose(Image.FLIP_LEFT_RIGHT)
                    vec_H_labeled = img2vec.get_vec(H_flipped_image_Labeled, tensor=False) # get the embeddings vectors
                    vec_H_labeled = np.array(vec_H_labeled) # convert list into array to be able to use reshape()
                    vec_H_labeled_images.append(vec_H_labeled)    
                    
                    # Perform the vertical flip
                    V_flipped_image_Labeled = resized_labeled_images.transpose(Image.FLIP_TOP_BOTTOM)
                    vec_V_labeled = img2vec.get_vec(V_flipped_image_Labeled, tensor=False) # get the embeddings vectors
                    vec_V_labeled = np.array(vec_V_labeled) # convert list into array to be able to use reshape()
                    vec_V_labeled_images.append(vec_V_labeled)  
                    
                    # Perform the contrast enhancement 1
                    C1_image_Labeled = constrast_enh(resized_labeled_images,0.7)
                    vec_C1_labeled = img2vec.get_vec(C1_image_Labeled, tensor=False) # get the embeddings vectors
                    vec_C1_labeled = np.array(vec_C1_labeled) # convert list into array to be able to use reshape()
                    vec_C1_labeled_images.append(vec_C1_labeled) 
                    
                    # Perform the contrast enhancement 2
                    C2_image_Labeled = constrast_enh(resized_labeled_images,1.3)
                    vec_C2_labeled = img2vec.get_vec(C2_image_Labeled, tensor=False) # get the embeddings vectors
                    vec_C2_labeled = np.array(vec_C2_labeled) # convert list into array to be able to use reshape()
                    vec_C2_labeled_images.append(vec_C2_labeled) 
                    
                    # Perform the contrast enhancement 3
                    C3_image_Labeled = constrast_enh(resized_labeled_images,1.5)
                    vec_C3_labeled = img2vec.get_vec(C3_image_Labeled, tensor=False) # get the embeddings vectors
                    vec_C3_labeled = np.array(vec_C3_labeled) # convert list into array to be able to use reshape()
                    vec_C3_labeled_images.append(vec_C3_labeled) 
                    
                    # Perform the Brightness enhancement 1
                    B1_image_Labeled = brightness_enh(resized_labeled_images,0.8)
                    vec_B1_labeled = img2vec.get_vec(B1_image_Labeled, tensor=False) # get the embeddings vectors
                    vec_B1_labeled = np.array(vec_B1_labeled) # convert list into array to be able to use reshape()
                    vec_B1_labeled_images.append(vec_B1_labeled)
                    
                    # Perform the Brightness enhancement 2
                    B2_image_Labeled = brightness_enh(resized_labeled_images,1.2)
                    vec_B2_labeled = img2vec.get_vec(B2_image_Labeled, tensor=False) # get the embeddings vectors
                    vec_B2_labeled = np.array(vec_B2_labeled) # convert list into array to be able to use reshape()
                    vec_B2_labeled_images.append(vec_B2_labeled)
                    
                    # Perform the Gaussian Blur 1
                    GB1_image_Labeled = gaussian_Blurr(resized_labeled_images,5,0.1,5)
                    vec_GB1_labeled = img2vec.get_vec(GB1_image_Labeled, tensor=False) # get the embeddings vectors
                    vec_GB1_labeled = np.array(vec_GB1_labeled) # convert list into array to be able to use reshape()
                    vec_GB1_labeled_images.append(vec_GB1_labeled)
                #--------------------------------------
                
                
                 
                if neg_NbItems!= 0:
                    for iii in range(len(neg_labeled_Images)): 
                        bbbox=neg_labeled_bbox_tuples_values[iii]
                        xmin=bbbox[0]
                        ymin=bbbox[1]
                        w=bbbox[2]
                        h=bbbox[3]
                        x1,y1,x2,y2=convert_bbox(xmin,ymin,w,h)
        
                        bbox = (x1,y1,x2, y2) #  the coordinates
                        #print("------------------bbox=",bbox,"-------------------------\n")
                        the_image_path=neg_labeled_Images[iii] # get the image array (PIL.open())
                        # Crop the image using the bounding box coordinates
                        neg_labeled_cropped_image= the_image_path.crop(bbox)
                        neg_labeled_cropped_images.append(neg_labeled_cropped_image)
                    # combined all the cropped negative samples of the labeled images for all the detected classes
                   
                    neg_combined_image =neg_labeled_cropped_images[0]
                    if neg_NbItems >= 2:
                        cpt=neg_NbItems-1
                        while cpt>0:
                            neg_combined_image=combine_images(neg_combined_image, neg_labeled_cropped_images[neg_NbItems-cpt])
                            cpt=cpt-1
               
                    resized_neg_labeled_images = neg_combined_image.resize((224,224), Image.Resampling.LANCZOS) # final labeled image containing all the cropped from all the detected classes
                    neg_vec_labeled = img2vec.get_vec(resized_neg_labeled_images, tensor=False) # get the embeddings vectors
                    neg_vec_labeled = np.array(neg_vec_labeled) # convert list into array to be able to use reshape()
                    neg_vec_labeled_images.append(neg_vec_labeled)   
                
                org_boxes_tuples_values = convert_boxes_to_tuples(extract_boxes(pesudo_proposals_roih_unsup_k[jj]))
                org_copy_boxes_tuples_values = convert_boxes_to_tuples(extract_boxes(copy_pesudo_proposals_roih_unsup_k[jj]))
                H_flipped_copy_boxes_tuples_values = convert_boxes_to_tuples(extract_boxes(H_flipped_pesudo_proposals_roih_unsup_k[jj]))
                V_flipped_copy_boxes_tuples_values = convert_boxes_to_tuples(extract_boxes(V_flipped_pesudo_proposals_roih_unsup_k[jj]))
                
                C1_boxes_tuples_values = convert_boxes_to_tuples(extract_boxes(C1_pesudo_proposals_roih_unsup_k[jj]))
                C2_boxes_tuples_values = convert_boxes_to_tuples(extract_boxes(C2_pesudo_proposals_roih_unsup_k[jj]))
                C3_boxes_tuples_values = convert_boxes_to_tuples(extract_boxes(C3_pesudo_proposals_roih_unsup_k[jj]))
                B1_boxes_tuples_values = convert_boxes_to_tuples(extract_boxes(B1_pesudo_proposals_roih_unsup_k[jj]))
                B2_boxes_tuples_values = convert_boxes_to_tuples(extract_boxes(B2_pesudo_proposals_roih_unsup_k[jj]))
                GB1_boxes_tuples_values = convert_boxes_to_tuples(extract_boxes(GB1_pesudo_proposals_roih_unsup_k[jj]))
                
                # By defaut there is no predicted pseudo bounding boxes cropped
                org_cropped_image=empty_image()
                copy_org_cropped_image=empty_image()
                H_flipped_copy_cropped_image=empty_image()
                V_flipped_copy_cropped_image=empty_image() 
                
                C1_cropped_image=empty_image() 
                C2_cropped_image=empty_image()
                C3_cropped_image=empty_image()
                B1_cropped_image=empty_image()
                B2_cropped_image=empty_image()
                GB1_cropped_image=empty_image()
                
                # By defaut all similarities scores between different cropped pseudo labels  are equal to 0
                loss_org_copy_org=0.0 
                loss_HF_VF=0.0
                loss_org_HF=0.0
                loss_org_VF=0.0
                loss_copy_org_HF=0.0 
                loss_copy_org_VF=0.0
                
                loss_labeled_org=0.0
                loss_labeled_copy_org=0.0
                loss_labeled_HF=0.0
                loss_labeled_VF=0.0
                
                loss_org_C1=0.0
                loss_org_C2=0.0
                loss_org_C3=0.0
                loss_org_B1=0.0
                loss_org_B2=0.0
                loss_org_GB1=0.0
                
                loss_copy_org_C1=0.0 
                loss_copy_org_C2=0.0 
                loss_copy_org_C3=0.0 
                loss_copy_org_B1=0.0 
                loss_copy_org_B2=0.0 
                loss_copy_org_GB1=0.0 
                
                loss_HF_C1=0.0 
                loss_HF_C2=0.0 
                loss_HF_C3=0.0 
                loss_HF_B1=0.0 
                loss_HF_B2=0.0 
                loss_HF_GB1=0.0 
                
                loss_VF_C1=0.0
                loss_VF_C2=0.0
                loss_VF_C3=0.0
                loss_VF_B1=0.0
                loss_VF_B2=0.0
                loss_VF_GB1=0.0
                
                loss_labeled_C1=0.0
                loss_labeled_C2=0.0
                loss_labeled_C3=0.0
                loss_labeled_B1=0.0
                loss_labeled_B2=0.0
                loss_labeled_GB1=0.0
                
                loss_C1_C2=0.0
                loss_C1_C3=0.0
                loss_C1_B1=0.0
                loss_C1_B2=0.0
                loss_C1_GB1=0.0
                
                loss_C2_C3=0.0
                loss_C2_B1=0.0
                loss_C2_B2=0.0
                loss_C2_GB1=0.0
                
                loss_C3_B1=0.0
                loss_C3_B2=0.0
                loss_C3_GB1=0.0
                
                loss_B1_B2=0.0
                loss_B1_GB1=0.0
                
                loss_B2_GB1=0.0
                
                
                
                # if there is a predicted bounding boxes for the unlabeled image with its augmented versions by the Teacher model
                # do the following
                
                # if there are pseudo bboxes detected for the original image
                if len(org_boxes_tuples_values)!= 0:
                    # Get the minimum and maximum coordinates
                    x_min, y_min, x_max, y_max = get_largest_bounding_box(org_boxes_tuples_values) #get the coordinates of the biggest bbox covering all candidates bboxes.
                    bbox = (x_min, y_min, x_max, y_max)
                    org_cropped_image=crop_image(image_path,bbox)
                
                # if there are pseudo bboxes detected for the copy of the original image
                if len(org_copy_boxes_tuples_values)!= 0:
                    # Get the minimum and maximum coordinates
                    x_min, y_min, x_max, y_max = get_largest_bounding_box(org_copy_boxes_tuples_values) #get the coordinates of the biggest bbox covering all candidates bboxes.
                    bbox = (x_min, y_min, x_max, y_max)
                    copy_org_cropped_image=crop_image(copy_image_path,bbox)
                    
                # if there are pseudo bboxes detected for the horizontal fliped of the image    
                if len(H_flipped_copy_boxes_tuples_values)!= 0:
                    # Get the minimum and maximum coordinates
                    x_min, y_min, x_max, y_max = get_largest_bounding_box(H_flipped_copy_boxes_tuples_values) #get the coordinates of the biggest bbox covering all candidates bboxes.
                    bbox = (x_min, y_min, x_max, y_max)
                    H_flipped_copy_cropped_image=crop_image(H_flipped_image_path,bbox)
                
                # if there are pseudo bboxes detected for the vertical fliped of the image     
                if len(V_flipped_copy_boxes_tuples_values)!= 0:
                    # Get the minimum and maximum coordinates
                    x_min, y_min, x_max, y_max = get_largest_bounding_box(V_flipped_copy_boxes_tuples_values) #get the coordinates of the biggest bbox covering all candidates bboxes.
                    bbox = (x_min, y_min, x_max, y_max)
                    V_flipped_copy_cropped_image=crop_image(V_flipped_image_path,bbox)
               
                
                # if there are pseudo bboxes detected      
                if len(C1_boxes_tuples_values)!= 0:
                    # Get the minimum and maximum coordinates
                    x_min, y_min, x_max, y_max = get_largest_bounding_box(C1_boxes_tuples_values) #get the coordinates of the biggest bbox covering all candidates bboxes.
                    bbox = (x_min, y_min, x_max, y_max)
                    C1_cropped_image=crop_image(C1_image_path,bbox)
                
                # if there are pseudo bboxes detected      
                if len(C2_boxes_tuples_values)!= 0:
                    # Get the minimum and maximum coordinates
                    x_min, y_min, x_max, y_max = get_largest_bounding_box(C2_boxes_tuples_values) #get the coordinates of the biggest bbox covering all candidates bboxes.
                    bbox = (x_min, y_min, x_max, y_max)
                    C2_cropped_image=crop_image(C2_image_path,bbox)
                    
                # if there are pseudo bboxes detected      
                if len(C3_boxes_tuples_values)!= 0:
                    # Get the minimum and maximum coordinates
                    x_min, y_min, x_max, y_max = get_largest_bounding_box(C3_boxes_tuples_values) #get the coordinates of the biggest bbox covering all candidates bboxes.
                    bbox = (x_min, y_min, x_max, y_max)
                    C3_cropped_image=crop_image(C3_image_path,bbox)
                    
                # if there are pseudo bboxes detected      
                if len(B1_boxes_tuples_values)!= 0:
                    # Get the minimum and maximum coordinates
                    x_min, y_min, x_max, y_max = get_largest_bounding_box(B1_boxes_tuples_values) #get the coordinates of the biggest bbox covering all candidates bboxes.
                    bbox = (x_min, y_min, x_max, y_max)
                    B1_cropped_image=crop_image(B1_image_path,bbox)
                
                # if there are pseudo bboxes detected      
                if len(B2_boxes_tuples_values)!= 0:
                    # Get the minimum and maximum coordinates
                    x_min, y_min, x_max, y_max = get_largest_bounding_box(B2_boxes_tuples_values) #get the coordinates of the biggest bbox covering all candidates bboxes.
                    bbox = (x_min, y_min, x_max, y_max)
                    B2_cropped_image=crop_image(B2_image_path,bbox)
                
                # if there are pseudo bboxes detected      
                if len(GB1_boxes_tuples_values)!= 0:
                    # Get the minimum and maximum coordinates
                    x_min, y_min, x_max, y_max = get_largest_bounding_box(GB1_boxes_tuples_values) #get the coordinates of the biggest bbox covering all candidates bboxes.
                    bbox = (x_min, y_min, x_max, y_max)
                    GB1_cropped_image=crop_image(GB1_image_path,bbox)
                    
                    
                # Generate the embbedings vectors for each using img2vec, returned as a torch FloatTensor 
                # Using PyTorch Cosine Similarity 
                
                vec_org=np.array([])
                vec_copy_org=np.array([])
                vec_H_fliped=np.array([])
                vec_V_fliped=np.array([])
                
                vec_C1=np.array([])
                vec_C2=np.array([])
                vec_C3=np.array([])
                vec_B1=np.array([])
                vec_B2=np.array([])
                vec_GB1=np.array([])
                
                if (len(org_boxes_tuples_values)!= 0 and NbItems!= 0):
                    vec_org = img2vec.get_vec(org_cropped_image, tensor=False)
                    loss_labeled_org=loss_func(vec_org,vec_labeled,neg_vec_labeled)
                    #cos_sim_labeled_org=cosine_similarity(vec_labeled.reshape((1, -1)), vec_org.reshape((1, -1)))[0][0] # similarity with the labeled embeddings
                if (len(org_copy_boxes_tuples_values)!= 0 and NbItems!= 0):    
                    vec_copy_org = img2vec.get_vec(copy_org_cropped_image, tensor=False)
                    loss_labeled_copy_org=loss_func(vec_copy_org,vec_labeled,neg_vec_labeled)
                    #cos_sim_labeled_copy_org=cosine_similarity(vec_labeled.reshape((1, -1)), vec_copy_org.reshape((1, -1)))[0][0]
                if (len(H_flipped_copy_boxes_tuples_values)!= 0 and NbItems!= 0):
                    vec_H_fliped = img2vec.get_vec(H_flipped_copy_cropped_image, tensor=False)
                    loss_labeled_HF=loss_func(vec_H_fliped,vec_labeled,neg_vec_labeled)
                    #cos_sim_labeled_HF=cosine_similarity(vec_labeled.reshape((1, -1)), vec_H_fliped.reshape((1, -1)))[0][0]
                if (len(V_flipped_copy_boxes_tuples_values)!= 0 and NbItems!= 0):
                    vec_V_fliped = img2vec.get_vec(V_flipped_copy_cropped_image, tensor=False)
                    loss_labeled_VF=loss_func( vec_V_fliped,vec_labeled,neg_vec_labeled)
                    #cos_sim_labeled_VF=cosine_similarity(vec_labeled.reshape((1, -1)), vec_V_fliped.reshape((1, -1)))[0][0]
                
                if (len(C1_boxes_tuples_values)!= 0 and NbItems!= 0):    
                    vec_C1 = img2vec.get_vec(C1_cropped_image, tensor=False)
                    loss_labeled_C1=loss_func(vec_C1, vec_labeled,  neg_vec_labeled)
                    #cos_sim_labeled_C1=cosine_similarity(vec_labeled.reshape((1, -1)), vec_C1.reshape((1, -1)))[0][0]
                
                if (len(C2_boxes_tuples_values)!= 0 and NbItems!= 0):    
                    vec_C2 = img2vec.get_vec(C2_cropped_image, tensor=False)
                    loss_labeled_C2=loss_func( vec_C2,vec_labeled, neg_vec_labeled)
                    #cos_sim_labeled_C2=cosine_similarity(vec_labeled.reshape((1, -1)), vec_C2.reshape((1, -1)))[0][0]
                
                if (len(C3_boxes_tuples_values)!= 0 and NbItems!= 0):    
                    vec_C3 = img2vec.get_vec(C3_cropped_image, tensor=False)
                    loss_labeled_C3=loss_func(vec_C3,vec_labeled, neg_vec_labeled)
                    #cos_sim_labeled_C3=cosine_similarity(vec_labeled.reshape((1, -1)), vec_C3.reshape((1, -1)))[0][0]
                    
                if (len(B1_boxes_tuples_values)!= 0 and NbItems!= 0):    
                    vec_B1 = img2vec.get_vec(B1_cropped_image, tensor=False)
                    loss_labeled_B1=loss_func(vec_B1, vec_labeled, neg_vec_labeled)
                    #cos_sim_labeled_B1=cosine_similarity(vec_labeled.reshape((1, -1)), vec_B1.reshape((1, -1)))[0][0]
                
                if (len(B2_boxes_tuples_values)!= 0 and NbItems!= 0):    
                    vec_B2 = img2vec.get_vec(B2_cropped_image, tensor=False)
                    loss_labeled_B2=loss_func(vec_B2,vec_labeled,neg_vec_labeled)
                    #cos_sim_labeled_B2=cosine_similarity(vec_labeled.reshape((1, -1)), vec_B2.reshape((1, -1)))[0][0]
                
                if (len(GB1_boxes_tuples_values)!= 0 and NbItems!= 0):    
                    vec_GB1 = img2vec.get_vec(GB1_cropped_image, tensor=False)
                    loss_labeled_GB1=loss_func(vec_GB1,vec_labeled, neg_vec_labeled)
                    #cos_sim_labeled_GB1=cosine_similarity(vec_labeled.reshape((1, -1)), vec_GB1.reshape((1, -1)))[0][0]
                    
                    
                # Using PyTorch Cosine Similarity 
                # For each unlabled image Compute the similarity scores (cosinus sim) for between each (1x6)
                if (len(org_boxes_tuples_values)!= 0 and len(org_copy_boxes_tuples_values)!= 0):
                    vec_org = img2vec.get_vec(org_cropped_image, tensor=False)
                    vec_copy_org = img2vec.get_vec(copy_org_cropped_image, tensor=False)
                    loss_org_copy_org =loss_func(vec_copy_org,vec_org,neg_vec_labeled)
                    #cos_sim_org_copy_org = cosine_similarity(vec_org.reshape((1, -1)), vec_copy_org.reshape((1, -1)))[0][0]
                if (len(H_flipped_copy_boxes_tuples_values)!= 0 and len(V_flipped_copy_boxes_tuples_values)!= 0):
                    vec_H_fliped = img2vec.get_vec(H_flipped_copy_cropped_image, tensor=False)
                    vec_V_fliped = img2vec.get_vec(V_flipped_copy_cropped_image, tensor=False)
                    loss_HF_VF=loss_func(vec_H_fliped, vec_V_fliped,neg_vec_labeled)
                    #cos_sim_HF_VF = cosine_similarity(vec_H_fliped.reshape((1, -1)), vec_V_fliped.reshape((1, -1)))[0][0]
                if (len(org_boxes_tuples_values)!= 0 and  len(H_flipped_copy_boxes_tuples_values)!= 0):
                    vec_org = img2vec.get_vec(org_cropped_image, tensor=False)
                    vec_H_fliped = img2vec.get_vec(H_flipped_copy_cropped_image, tensor=False)
                    loss_org_HF=loss_func(vec_org, vec_H_fliped,neg_vec_labeled)
                    #cos_sim_org_HF = cosine_similarity(vec_org.reshape((1, -1)), vec_H_fliped.reshape((1, -1)))[0][0]
                if (len(org_boxes_tuples_values)!= 0 and  len(V_flipped_copy_boxes_tuples_values)!= 0): 
                    vec_org = img2vec.get_vec(org_cropped_image, tensor=False)
                    vec_V_fliped = img2vec.get_vec(V_flipped_copy_cropped_image, tensor=False)
                    loss_org_VF=loss_func(vec_org, vec_V_fliped,neg_vec_labeled)
                    #cos_sim_org_VF = cosine_similarity(vec_org.reshape((1, -1)), vec_V_fliped.reshape((1, -1)))[0][0]
                if (len(org_copy_boxes_tuples_values)!= 0 and  len(H_flipped_copy_boxes_tuples_values)!= 0):
                    vec_H_fliped = img2vec.get_vec(H_flipped_copy_cropped_image, tensor=False)
                    vec_copy_org = img2vec.get_vec(copy_org_cropped_image, tensor=False)
                    loss_copy_org_HF=loss_func(vec_H_fliped, vec_copy_org,neg_vec_labeled)
                    #cos_sim_copy_org_HF = cosine_similarity(vec_H_fliped.reshape((1, -1)), vec_copy_org.reshape((1, -1)))[0][0]
                if (len(org_copy_boxes_tuples_values)!= 0 and  len(V_flipped_copy_boxes_tuples_values)!= 0):
                    vec_V_fliped = img2vec.get_vec(V_flipped_copy_cropped_image, tensor=False)
                    vec_copy_org = img2vec.get_vec(copy_org_cropped_image, tensor=False)
                    loss_copy_org_VF =loss_func(vec_V_fliped, vec_copy_org,neg_vec_labeled)
                    #cos_sim_copy_org_VF = cosine_similarity(vec_V_fliped.reshape((1, -1)), vec_copy_org.reshape((1, -1)))[0][0]
                    
                # simmilarrity metrics ,.......    
                    
                if (len(org_boxes_tuples_values)!= 0 and len(C1_boxes_tuples_values)!= 0):
                    vec_org = img2vec.get_vec(org_cropped_image, tensor=False)
                    vec_C1 = img2vec.get_vec(C1_cropped_image, tensor=False)
                    loss_org_C1 =loss_func(vec_org, vec_C1,neg_vec_labeled)
                    #cos_sim_org_C1 = cosine_similarity(vec_org.reshape((1, -1)), vec_C1.reshape((1, -1)))[0][0]

                if (len(org_boxes_tuples_values)!= 0 and len(C2_boxes_tuples_values)!= 0):
                    vec_org = img2vec.get_vec(org_cropped_image, tensor=False)
                    vec_C2 = img2vec.get_vec(C2_cropped_image, tensor=False)
                    loss_org_C2 =loss_func(vec_org, vec_C2,neg_vec_labeled)
                    #cos_sim_org_C2 = cosine_similarity(vec_org.reshape((1, -1)), vec_C2.reshape((1, -1)))[0][0]
                
                if (len(org_boxes_tuples_values)!= 0 and len(C3_boxes_tuples_values)!= 0):
                    vec_org = img2vec.get_vec(org_cropped_image, tensor=False)
                    vec_C3 = img2vec.get_vec(C3_cropped_image, tensor=False)
                    loss_org_C3 =loss_func(vec_org, vec_C3,neg_vec_labeled)
                    #cos_sim_org_C3 = cosine_similarity(vec_org.reshape((1, -1)), vec_C3.reshape((1, -1)))[0][0]
                    
                if (len(org_boxes_tuples_values)!= 0 and len(B1_boxes_tuples_values)!= 0):
                    vec_org = img2vec.get_vec(org_cropped_image, tensor=False)
                    vec_B1 = img2vec.get_vec(B1_cropped_image, tensor=False)
                    loss_org_B1 =loss_func(vec_org, vec_B1,neg_vec_labeled)
                    #cos_sim_org_B1 = cosine_similarity(vec_org.reshape((1, -1)), vec_B1.reshape((1, -1)))[0][0]
                
                if (len(org_boxes_tuples_values)!= 0 and len(B2_boxes_tuples_values)!= 0):
                    vec_org = img2vec.get_vec(org_cropped_image, tensor=False)
                    vec_B2 = img2vec.get_vec(B2_cropped_image, tensor=False)
                    loss_org_B2 =loss_func(vec_org, vec_B2,neg_vec_labeled)
                    #cos_sim_org_B2 = cosine_similarity(vec_org.reshape((1, -1)), vec_B2.reshape((1, -1)))[0][0]
                
                if (len(org_boxes_tuples_values)!= 0 and len(GB1_boxes_tuples_values)!= 0):
                    vec_org = img2vec.get_vec(org_cropped_image, tensor=False)
                    vec_GB1 = img2vec.get_vec(GB1_cropped_image, tensor=False)
                    loss_org_GB1 =loss_func(vec_org, vec_GB1,neg_vec_labeled)
                    #cos_sim_org_GB1 = cosine_similarity(vec_org.reshape((1, -1)), vec_GB1.reshape((1, -1)))[0][0]
                    
                    
                if (len(org_copy_boxes_tuples_values)!= 0 and  len(C1_boxes_tuples_values)!= 0):
                    vec_C1 = img2vec.get_vec(C1_cropped_image, tensor=False)
                    vec_copy_org = img2vec.get_vec(copy_org_cropped_image, tensor=False)
                    loss_copy_org_C1 =loss_func(vec_C1, vec_copy_org,neg_vec_labeled)
                    #cos_sim_copy_org_C1 = cosine_similarity(vec_C1.reshape((1, -1)), vec_copy_org.reshape((1, -1)))[0][0]    
                    
                if (len(org_copy_boxes_tuples_values)!= 0 and  len(C2_boxes_tuples_values)!= 0):
                    vec_C2 = img2vec.get_vec(C2_cropped_image, tensor=False)
                    vec_copy_org = img2vec.get_vec(copy_org_cropped_image, tensor=False)
                    loss_copy_org_C2 =loss_func(vec_C2, vec_copy_org,neg_vec_labeled)
                    #cos_sim_copy_org_C2 = cosine_similarity(vec_C2.reshape((1, -1)), vec_copy_org.reshape((1, -1)))[0][0]
                
                if (len(org_copy_boxes_tuples_values)!= 0 and  len(C3_boxes_tuples_values)!= 0):
                    vec_C3 = img2vec.get_vec(C3_cropped_image, tensor=False)
                    vec_copy_org = img2vec.get_vec(copy_org_cropped_image, tensor=False)
                    loss_copy_org_C3 =loss_func(vec_C3, vec_copy_org,neg_vec_labeled)
                    #cos_sim_copy_org_C3 = cosine_similarity(vec_C3.reshape((1, -1)), vec_copy_org.reshape((1, -1)))[0][0]
                
                if (len(org_copy_boxes_tuples_values)!= 0 and  len(B1_boxes_tuples_values)!= 0):
                    vec_B1 = img2vec.get_vec(B1_cropped_image, tensor=False)
                    vec_copy_org = img2vec.get_vec(copy_org_cropped_image, tensor=False)
                    loss_copy_org_B1 =loss_func(vec_B1, vec_copy_org,neg_vec_labeled)
                    #cos_sim_copy_org_B1 = cosine_similarity(vec_B1.reshape((1, -1)), vec_copy_org.reshape((1, -1)))[0][0]
                
                if (len(org_copy_boxes_tuples_values)!= 0 and  len(B2_boxes_tuples_values)!= 0):
                    vec_B2 = img2vec.get_vec(B2_cropped_image, tensor=False)
                    vec_copy_org = img2vec.get_vec(copy_org_cropped_image, tensor=False)
                    loss_copy_org_B2 =loss_func(vec_B2, vec_copy_org,neg_vec_labeled)
                    #cos_sim_copy_org_B2 = cosine_similarity(vec_B2.reshape((1, -1)), vec_copy_org.reshape((1, -1)))[0][0]
                    
                if (len(org_copy_boxes_tuples_values)!= 0 and  len(GB1_boxes_tuples_values)!= 0):
                    vec_GB1 = img2vec.get_vec(GB1_cropped_image, tensor=False)
                    vec_copy_org = img2vec.get_vec(copy_org_cropped_image, tensor=False)
                    loss_copy_org_GB1 =loss_func(vec_GB1, vec_copy_org,neg_vec_labeled)
                    #cos_sim_copy_org_GB1 = cosine_similarity(vec_GB1.reshape((1, -1)), vec_copy_org.reshape((1, -1)))[0][0]
                    
                if (len(C1_boxes_tuples_values)!= 0 and  len(H_flipped_copy_boxes_tuples_values)!= 0):
                    vec_H_fliped = img2vec.get_vec(H_flipped_copy_cropped_image, tensor=False)
                    vec_C1 = img2vec.get_vec(C1_cropped_image, tensor=False)
                    loss_HF_C1 =loss_func(vec_H_fliped, vec_C1,neg_vec_labeled)
                    #cos_sim_HF_C1 = cosine_similarity(vec_H_fliped.reshape((1, -1)), vec_C1.reshape((1, -1)))[0][0]    
                
                if (len(C2_boxes_tuples_values)!= 0 and  len(H_flipped_copy_boxes_tuples_values)!= 0):
                    vec_H_fliped = img2vec.get_vec(H_flipped_copy_cropped_image, tensor=False)
                    vec_C2 = img2vec.get_vec(C2_cropped_image, tensor=False)
                    loss_HF_C2 =loss_func(vec_H_fliped, vec_C2,neg_vec_labeled)
                    #cos_sim_HF_C2 = cosine_similarity(vec_H_fliped.reshape((1, -1)), vec_C2.reshape((1, -1)))[0][0]
                
                if (len(C3_boxes_tuples_values)!= 0 and  len(H_flipped_copy_boxes_tuples_values)!= 0):
                    vec_H_fliped = img2vec.get_vec(H_flipped_copy_cropped_image, tensor=False)
                    vec_C3 = img2vec.get_vec(C3_cropped_image, tensor=False)
                    loss_HF_C3 =loss_func(vec_H_fliped, vec_C3,neg_vec_labeled)
                    #cos_sim_HF_C3 = cosine_similarity(vec_H_fliped.reshape((1, -1)), vec_C3.reshape((1, -1)))[0][0]
                
                if (len(B1_boxes_tuples_values)!= 0 and  len(H_flipped_copy_boxes_tuples_values)!= 0):
                    vec_H_fliped = img2vec.get_vec(H_flipped_copy_cropped_image, tensor=False)
                    vec_B1 = img2vec.get_vec(B1_cropped_image, tensor=False)
                    loss_HF_B1 =loss_func(vec_H_fliped, vec_B1,neg_vec_labeled)
                    #cos_sim_HF_B1 = cosine_similarity(vec_H_fliped.reshape((1, -1)), vec_B1.reshape((1, -1)))[0][0]
                
                if (len(B2_boxes_tuples_values)!= 0 and  len(H_flipped_copy_boxes_tuples_values)!= 0):
                    vec_H_fliped = img2vec.get_vec(H_flipped_copy_cropped_image, tensor=False)
                    vec_B2 = img2vec.get_vec(B2_cropped_image, tensor=False)
                    loss_HF_B2 =loss_func(vec_H_fliped, vec_B2,neg_vec_labeled)
                    #cos_sim_HF_B2 = cosine_similarity(vec_H_fliped.reshape((1, -1)), vec_B2.reshape((1, -1)))[0][0]
                
                if (len(GB1_boxes_tuples_values)!= 0 and  len(H_flipped_copy_boxes_tuples_values)!= 0):
                    vec_H_fliped = img2vec.get_vec(H_flipped_copy_cropped_image, tensor=False)
                    vec_GB1 = img2vec.get_vec(GB1_cropped_image, tensor=False)
                    loss_HF_GB1 =loss_func(vec_H_fliped, vec_GB1,neg_vec_labeled)
                    #cos_sim_HF_GB1 = cosine_similarity(vec_H_fliped.reshape((1, -1)), vec_GB1.reshape((1, -1)))[0][0]
                    
                    
                if (len(C1_boxes_tuples_values)!= 0 and  len(V_flipped_copy_boxes_tuples_values)!= 0):
                    vec_V_fliped = img2vec.get_vec(V_flipped_copy_cropped_image, tensor=False)
                    vec_C1 = img2vec.get_vec(C1_cropped_image, tensor=False)
                    loss_VF_C1 =loss_func(vec_V_fliped, vec_C1,neg_vec_labeled)
                    #cos_sim_VF_C1 = cosine_similarity(vec_V_fliped.reshape((1, -1)), vec_C1.reshape((1, -1)))[0][0]
                        
                if (len(C2_boxes_tuples_values)!= 0 and  len(V_flipped_copy_boxes_tuples_values)!= 0):
                    vec_V_fliped = img2vec.get_vec(V_flipped_copy_cropped_image, tensor=False)
                    vec_C2 = img2vec.get_vec(C2_cropped_image, tensor=False)
                    loss_VF_C2 =loss_func(vec_V_fliped, vec_C2,neg_vec_labeled)
                    #cos_sim_VF_C2 = cosine_similarity(vec_V_fliped.reshape((1, -1)), vec_C2.reshape((1, -1)))[0][0]
                
                if (len(C3_boxes_tuples_values)!= 0 and  len(V_flipped_copy_boxes_tuples_values)!= 0):
                    vec_V_fliped = img2vec.get_vec(V_flipped_copy_cropped_image, tensor=False)
                    vec_C3 = img2vec.get_vec(C3_cropped_image, tensor=False)
                    loss_VF_C3 =loss_func(vec_V_fliped, vec_C3,neg_vec_labeled)
                    #cos_sim_VF_C3 = cosine_similarity(vec_V_fliped.reshape((1, -1)), vec_C3.reshape((1, -1)))[0][0]
                    
                if (len(B1_boxes_tuples_values)!= 0 and  len(V_flipped_copy_boxes_tuples_values)!= 0):
                    vec_V_fliped = img2vec.get_vec(V_flipped_copy_cropped_image, tensor=False)
                    vec_B1 = img2vec.get_vec(B1_cropped_image, tensor=False)
                    loss_VF_B1 =loss_func(vec_V_fliped, vec_B1,neg_vec_labeled)
                    #cos_sim_VF_B1 = cosine_similarity(vec_V_fliped.reshape((1, -1)), vec_B1.reshape((1, -1)))[0][0]  
                
                if (len(B2_boxes_tuples_values)!= 0 and  len(V_flipped_copy_boxes_tuples_values)!= 0):
                    vec_V_fliped = img2vec.get_vec(V_flipped_copy_cropped_image, tensor=False)
                    vec_B2 = img2vec.get_vec(B2_cropped_image, tensor=False)
                    loss_VF_B2 =loss_func(vec_V_fliped, vec_B2,neg_vec_labeled)
                    #cos_sim_VF_B2 = cosine_similarity(vec_V_fliped.reshape((1, -1)), vec_B2.reshape((1, -1)))[0][0]
                    
                if (len(GB1_boxes_tuples_values)!= 0 and  len(V_flipped_copy_boxes_tuples_values)!= 0):
                    vec_V_fliped = img2vec.get_vec(V_flipped_copy_cropped_image, tensor=False)
                    vec_GB1 = img2vec.get_vec(GB1_cropped_image, tensor=False)
                    loss_VF_GB1 =loss_func(vec_V_fliped, vec_GB1,neg_vec_labeled)
                    #cos_sim_VF_GB1 = cosine_similarity(vec_V_fliped.reshape((1, -1)), vec_GB1.reshape((1, -1)))[0][0]    
                    
                if (len(C1_boxes_tuples_values)!= 0 and  len(C2_boxes_tuples_values)!= 0):
                    vec_C1 = img2vec.get_vec(C1_cropped_image, tensor=False)
                    vec_C2 = img2vec.get_vec(C2_cropped_image, tensor=False)
                    loss_C1_C2  =loss_func(vec_C1, vec_C2,neg_vec_labeled)
                    #cos_sim_C1_C2 = cosine_similarity(vec_C1.reshape((1, -1)), vec_C2.reshape((1, -1)))[0][0]    
                    
                if (len(C1_boxes_tuples_values)!= 0 and  len(C3_boxes_tuples_values)!= 0):
                    vec_C1 = img2vec.get_vec(C1_cropped_image, tensor=False)
                    vec_C3 = img2vec.get_vec(C3_cropped_image, tensor=False)
                    loss_C1_C3  =loss_func(vec_C1, vec_C3,neg_vec_labeled)
                    #cos_sim_C1_C3 = cosine_similarity(vec_C1.reshape((1, -1)), vec_C3.reshape((1, -1)))[0][0] 
                
                if (len(C1_boxes_tuples_values)!= 0 and  len(B1_boxes_tuples_values)!= 0):
                    vec_C1 = img2vec.get_vec(C1_cropped_image, tensor=False)
                    vec_B1 = img2vec.get_vec(B1_cropped_image, tensor=False)
                    loss_C1_B1  =loss_func(vec_C1, vec_B1,neg_vec_labeled)
                    #cos_sim_C1_B1 = cosine_similarity(vec_C1.reshape((1, -1)), vec_B1.reshape((1, -1)))[0][0] 
                    
                if (len(C1_boxes_tuples_values)!= 0 and  len(B2_boxes_tuples_values)!= 0):
                    vec_C1 = img2vec.get_vec(C1_cropped_image, tensor=False)
                    vec_B2 = img2vec.get_vec(B2_cropped_image, tensor=False)
                    loss_C1_B2  =loss_func(vec_C1, vec_B2,neg_vec_labeled)
                    #cos_sim_C1_B2 = cosine_similarity(vec_C1.reshape((1, -1)), vec_B2.reshape((1, -1)))[0][0] 
                       
                if (len(C1_boxes_tuples_values)!= 0 and  len(GB1_boxes_tuples_values)!= 0):
                    vec_C1 = img2vec.get_vec(C1_cropped_image, tensor=False)
                    vec_GB1 = img2vec.get_vec(GB1_cropped_image, tensor=False)
                    loss_C1_GB1  =loss_func(vec_C1, vec_GB1,neg_vec_labeled)
                    #cos_sim_C1_GB1 = cosine_similarity(vec_C1.reshape((1, -1)), vec_GB1.reshape((1, -1)))[0][0] 
                        
                if (len(C2_boxes_tuples_values)!= 0 and  len(C3_boxes_tuples_values)!= 0):
                    vec_C2 = img2vec.get_vec(C2_cropped_image, tensor=False)
                    vec_C3 = img2vec.get_vec(C3_cropped_image, tensor=False)
                    loss_C2_C3  =loss_func(vec_C2, vec_C3,neg_vec_labeled)
                    #cos_sim_C2_C3 = cosine_similarity(vec_C2.reshape((1, -1)), vec_C3.reshape((1, -1)))[0][0]    
                    
                if (len(C2_boxes_tuples_values)!= 0 and  len(B1_boxes_tuples_values)!= 0):
                    vec_C2 = img2vec.get_vec(C2_cropped_image, tensor=False)
                    vec_B1 = img2vec.get_vec(B1_cropped_image, tensor=False)
                    loss_C2_B1  =loss_func(vec_C2, vec_B1,neg_vec_labeled)
                    #cos_sim_C2_B1 = cosine_similarity(vec_C2.reshape((1, -1)), vec_B1.reshape((1, -1)))[0][0]    
                    
                if (len(C2_boxes_tuples_values)!= 0 and  len(B2_boxes_tuples_values)!= 0):
                    vec_C2 = img2vec.get_vec(C2_cropped_image, tensor=False)
                    vec_B2 = img2vec.get_vec(B2_cropped_image, tensor=False)
                    loss_C2_B2  =loss_func(vec_C2, vec_B2,neg_vec_labeled)
                    #cos_sim_C2_B2 = cosine_similarity(vec_C2.reshape((1, -1)), vec_B2.reshape((1, -1)))[0][0]    
                
                if (len(C2_boxes_tuples_values)!= 0 and  len(GB1_boxes_tuples_values)!= 0):
                    vec_C2 = img2vec.get_vec(C2_cropped_image, tensor=False)
                    vec_GB1 = img2vec.get_vec(GB1_cropped_image, tensor=False)
                    loss_C2_GB1  =loss_func(vec_C2, vec_GB1,neg_vec_labeled)
                    #cos_sim_C2_GB1 = cosine_similarity(vec_C2.reshape((1, -1)), vec_GB1.reshape((1, -1)))[0][0]    
                
                if (len(C3_boxes_tuples_values)!= 0 and  len(B1_boxes_tuples_values)!= 0):
                    vec_C3 = img2vec.get_vec(C3_cropped_image, tensor=False)
                    vec_B1 = img2vec.get_vec(B1_cropped_image, tensor=False)
                    loss_C3_B1  =loss_func(vec_C3, vec_B1,neg_vec_labeled)
                    #cos_sim_C3_B1 = cosine_similarity(vec_C3.reshape((1, -1)), vec_B1.reshape((1, -1)))[0][0]
                
                if (len(C3_boxes_tuples_values)!= 0 and  len(B2_boxes_tuples_values)!= 0):
                    vec_C3 = img2vec.get_vec(C3_cropped_image, tensor=False)
                    vec_B2 = img2vec.get_vec(B2_cropped_image, tensor=False)
                    loss_C3_B2  =loss_func(vec_C3, vec_B2,neg_vec_labeled)
                    #cos_sim_C3_B2 = cosine_similarity(vec_C3.reshape((1, -1)), vec_B2.reshape((1, -1)))[0][0]
                
                if (len(C3_boxes_tuples_values)!= 0 and  len(GB1_boxes_tuples_values)!= 0):
                    vec_C3 = img2vec.get_vec(C3_cropped_image, tensor=False)
                    vec_GB1 = img2vec.get_vec(GB1_cropped_image, tensor=False)
                    loss_C3_GB1  =loss_func(vec_C3, vec_GB1,neg_vec_labeled)
                    #cos_sim_C3_GB1 = cosine_similarity(vec_C3.reshape((1, -1)), vec_GB1.reshape((1, -1)))[0][0]
                
                if (len(B1_boxes_tuples_values)!= 0 and  len(B2_boxes_tuples_values)!= 0):
                    vec_B1 = img2vec.get_vec(B1_cropped_image, tensor=False)
                    vec_B2 = img2vec.get_vec(B2_cropped_image, tensor=False)
                    loss_B1_B2  =loss_func(vec_B1, vec_B2,neg_vec_labeled)
                    #cos_sim_B1_B2 = cosine_similarity(vec_B1.reshape((1, -1)), vec_B2.reshape((1, -1)))[0][0]
                
                if (len(B1_boxes_tuples_values)!= 0 and  len(GB1_boxes_tuples_values)!= 0):
                    vec_B1 = img2vec.get_vec(B1_cropped_image, tensor=False)
                    vec_GB1 = img2vec.get_vec(GB1_cropped_image, tensor=False)
                    loss_B1_GB1 =loss_func(vec_B1, vec_GB1,neg_vec_labeled)
                    #cos_sim_B1_GB1 = cosine_similarity(vec_B1.reshape((1, -1)), vec_GB1.reshape((1, -1)))[0][0]   
                
                if (len(B2_boxes_tuples_values)!= 0 and  len(GB1_boxes_tuples_values)!= 0):
                    vec_B2 = img2vec.get_vec(B2_cropped_image, tensor=False)
                    vec_GB1 = img2vec.get_vec(GB1_cropped_image, tensor=False)
                    loss_B2_GB1 =loss_func(vec_B2, vec_GB1,neg_vec_labeled)
                    cos_sim_B2_GB1 = cosine_similarity(vec_B2.reshape((1, -1)), vec_GB1.reshape((1, -1)))[0][0]    
                
                # Put all the cosinus similarity metrics here in the array cos_Sim
                
                """
                loss_tab.append([loss_org_copy_org,loss_org_HF,loss_org_VF,loss_copy_org_HF,loss_copy_org_VF,loss_HF_VF,loss_labeled_org,loss_labeled_copy_org,
                loss_labeled_HF,loss_labeled_VF, loss_org_C1, loss_org_C2, loss_org_C3, loss_org_B1, loss_org_B2, loss_org_GB1, loss_copy_org_C1, loss_copy_org_C2,
                loss_copy_org_C3, loss_copy_org_B1, loss_copy_org_B2, loss_copy_org_GB1, loss_HF_C1, loss_HF_C2, loss_HF_C3, loss_HF_B1,loss_HF_B2,
                loss_HF_GB1, loss_VF_C1, loss_VF_C2, loss_VF_C3, loss_VF_B1, loss_VF_B2 , loss_VF_GB1, loss_labeled_C1, loss_labeled_C2, loss_labeled_C3, loss_labeled_B1,
                loss_labeled_B2, loss_labeled_GB1, loss_C1_C2, loss_C1_C3,  loss_C1_B1, loss_C1_B2, loss_C1_GB1, loss_C2_B1, loss_C2_B2, loss_C2_GB1,
                loss_C3_B1, loss_C3_B2, loss_C3_GB1, loss_B1_B2, loss_B1_GB1, loss_B2_GB1 ])
                """
                
                loss_tab.append([loss_labeled_org,loss_labeled_copy_org, loss_labeled_HF,loss_labeled_VF, loss_labeled_C1, loss_labeled_C2, loss_labeled_C3, loss_labeled_B1,loss_labeled_B2, loss_labeled_GB1])
                
                
               
            print("\n-------------------------------- LOSS FOR THE BATCH loss_tab=",loss_tab,"---------------------\n")
            logger.info("\n-------------------------------- LOSS FOR THE BATCH  loss_tab={}".format(loss_tab))
            
            
            
            avg_loss_sim=[] # the average loss based on embeddings for each image of the whole batch
           
            for kk in range(len(loss_tab)):
                avg=0
                row_loss=loss_tab[kk]
                for kkk in range(len(row_loss)):
                    # put a filtering condiction here for the similarity selection row_loss[kkk] if possible
                    avg=avg + row_loss[kkk]
                if len(row_loss)!=0:
                    avg=avg/len(row_loss)
                else:
                    avg=0
                    
                avg_loss_sim.append(avg)
               
                
            # Compute average loss derived from constistency
            avg_cst_loss_cos_Sim=0
            for jjj in range(len(avg_loss_sim)):
                avg_cst_loss_cos_Sim=avg_cst_loss_cos_Sim + avg_loss_sim[jjj]
            avg_cst_loss_cos_Sim=avg_cst_loss_cos_Sim/len(avg_loss_sim)
            
             
            #  add pseudo-label to unlabeled data

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]  #---------------------- Here we can decide to add the pseudo labels or not to the initial labeled set for the student training based on the value of 'avg_cos_Sim' or 'avg_cst_loss_cos_Sim' 
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)
            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            def adjust_pseudo_weight(cur_iter, max_iter, ori_weight, method="linear"):
                if method == "linear":
                    return cur_iter / max_iter * ori_weight
                elif method == "cosine":
                    import math
                    cos = math.cos(math.pi * cur_iter / max_iter)
                    w = ori_weight - (0.5 * ori_weight * (cos + 1))
                    return w
                else:
                    raise NotImplementedError(method+" has not implemented")

            # weight losses
            
            alpha=1
            print("---------------------aplha=",alpha,"---------------------\n")
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif "mask_pseudo" in key:
                        # loss_dict[key] = record_dict[key] * 0
                        loss_dict[key] = record_dict[key] * adjust_pseudo_weight(self.iter, self.max_iter, 1, method='cosine')
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        print("\n-------------------------------- UNSUPERVISED LOSS record_dict=",record_dict[key],"---------------------\n")
                        print("\n-------------------------------- CONSISTENCY LOSS Lc =",avg_cst_loss_cos_Sim,"---------------------")
                        
                        loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        ) + avg_cst_loss_cos_Sim*alpha                       # add consistency loss here
                        
                        logger.info("\n-------------------------------- UNSUPERVISED LOSS Lu={}".format(record_dict[key]*self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT))
                        logger.info("\n-------------------------------- Lc or avg_cst_loss_cos_Sim={}".format(avg_cst_loss_cos_Sim*alpha))
                       
                        
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1
                        print("\n-------------------------------- SUPERVISED LOSS=", record_dict[key],"---------------------")
                        logger.info("\n-------------------------------- SUPERVISED LOSS Ls={}".format(record_dict[key]))
                        
            
            losses = sum(loss_dict.values())
            #print("\n-------------------------------- TOTAL LOSSES with consistency loss=",losses,"---------------------\n")
            
        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )
            

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)

            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            ret.append(
                hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD,self.checkpointer,"bbox/AP")
            )
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

