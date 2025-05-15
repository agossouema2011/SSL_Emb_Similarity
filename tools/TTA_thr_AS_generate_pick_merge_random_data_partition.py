import json
import argparse
import os  

def generate_pick_merge_random(random_file, random_percent, indicator_file, pick_percent, save_file, static_file, reverse=True):
    P = []
    with open(indicator_file,'r') as f:
        items = f.readlines()
        P = [float(item) for item in items]
    #print("------------------- P=",P,"len(P)=",len(P),"----------------------\n")
    idx = sorted(range(len(P)), key=lambda k: P[k], reverse=reverse) # idx is a list of the index of the images sorted from the highest combined active metric to the lowest combined active metric
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
                if item not in exist_idx:  # select from the inference "results/e_coco/random_maxnorm.json" only the results where image is not in the "dataseed/COCO_supervision.txt "
                    iddx.append(item)
            left = int(total_imgs*(random_percent+pick_percent)/100) - len(table[str(random_percent)][str(i)])
            #print("-----------iddx[:left]=",iddx[:left],"------------len(iddx[:left])=",len(iddx[:left]))
            arr = iddx[:left] + table[str(random_percent)][str(i)]
            Filtered_images_idx= iddx[:left] # the index of the images having high Active Sampling score after pseudo labeling by the Teacher model.
            #print("-----------left=",left,"------------\n-------------------arr=",arr,"len(arr)=",len(arr))
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
            
    tableImages=[]  # to keep the list of the location of the speudo labeled images from Active Teacher filtering
    imagesLocations = list(dataContent.keys()) # get the keys which are the images locations
    for imgId in range(len(Filtered_images_idx)):
        imageIndex=Filtered_images_idx[imgId]
        image_url=imagesLocations[imageIndex] # get the corresponding image location
        #print("------------------------------- Image Location :",image_url,"-------------------------\n")
        tableImages.append(image_url)
        
    #print("-----------------tableImages=",tableImages,"--------\n")
    #--------------------------- UPDATE dataset annotations-----------------------------------
    # create a new datasets/coco/annotations/coco_training_e_thr_new.json
    # First make coco_training_new.json as a copy of datasets/coco/annotations/coco_training.json
    with open("datasets/coco/annotations/coco_training.json", "r") as data1:
        data1content=json.load(data1)
        
    with open("datasets/coco/annotations/coco_training_e_thr_new.json", "w") as data2:
            json.dump(data1content, data2)
    
    # Open the prediction json file
    json_annot_prediction_data= {}
    with open(static_file) as f:
        json_annot_prediction_data = json.load(f)
    #print("-----------------Type json_annot_prediction_data=",type(json_annot_prediction_data),"--------\n")    
    # Open the new annotation json file    
    with open('datasets/coco/annotations/coco_training_e_thr_new.json') as f:
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
        
        for iii in range(len(bbox_predicted)):
            
            #print("------------------bbox_predicted=",type(bbox_predicted[iii]))
            #print("------------------bbox_predicted=",bbox_predicted[iii])
            bbox_predicted_tab=bbox_predicted[iii][1:-1] # remove the fist and the last characters [ and ] from the string
            #print("------------------bbox_predicted111=",bbox_predicted_tab)
            bbox_predicted_tab=bbox_predicted_tab.split()
            
            the_area=(float(bbox_predicted_tab[2]) - float(bbox_predicted_tab[0])) * (float(bbox_predicted_tab[3]) - float(bbox_predicted_tab[1]))
            id_new=id_new+1
            bbox_predicted_new=[]
            bbox_predicted_new.append(float(bbox_predicted_tab[0]))
            bbox_predicted_new.append(float(bbox_predicted_tab[1]))
            bbox_predicted_new.append(float(bbox_predicted_tab[2]))
            bbox_predicted_new.append(float(bbox_predicted_tab[3]))
            #print("------------------bbox_predicted_new=",type(bbox_predicted_new))
            #print("------------------bbox_predicted_new=",bbox_predicted_new)
            
            tab_elt={"iscrowd": 0, "ignore": 0, "image_id": idImage, "bbox": bbox_predicted_new, "area": the_area, "segmentation": [], "category_id": pred_class_predicted[iii], "id": id_new}
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
    with open("datasets/coco/annotations/coco_training_e_thr_new.json",'w') as f:
        f.write(json.dumps(new_data_annotation))
            
        
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pick and merge label data partition')
    parser.add_argument("--random_file",type=str,default='dataseed/COCO_supervision.txt')
    parser.add_argument("--random_percent",type=float,default=10.0)
    parser.add_argument("--indicator_file",type=str,default='results/e_coco/10random_maxnorm.txt')
    parser.add_argument("--pick_percent",type=float,default=10.0)
    parser.add_argument("--reverse",type=bool,default=True)
    parser.add_argument("--save_file",type=str,default='dataseed/e_coco_pick/pick_maxnorm10+random10.txt')
    parser.add_argument("--static_file",type=str,default='temp/coco/static_by_random.json/e_static_by_random.json')
    args = parser.parse_args()
    generate_pick_merge_random(
        args.random_file,
        args.random_percent,
        args.indicator_file,
        args.pick_percent,
        args.save_file,
        args.static_file,
        args.reverse
    )